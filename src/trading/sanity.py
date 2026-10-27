# -*- coding: utf-8 -*-
"""报价合法性守卫（Quote Sanity Guard）
======================================

为什么需要它
------------
自动交易系统里，**一个坏价格就是一笔坏交易**。而坏价格比想象中常见：

- 数据源偶发返回复权价/前收盘价/错位合约（实测 2026-10-20 那次日线兜底事故：
  盘中估值退化成昨收，买入却用实时价，9 笔持仓凭空「浮亏」¥5.5 万，直接把
  单日亏损熔断打爆、闭环当天停摆）；
- 停牌股仍在推送最后成交价，系统当它「还活着」继续估值与判定止损；
- 新上市/退市整理期价格跳变，与昨收偏离远超涨跌停限制；
- 单位错误（分 vs 元、手 vs 股）导致价格放大 100 倍。

这些都不会抛异常，只会让 **止损在错误的价格上触发**、或让**权益曲线凭空断崖**。

设计原则
--------
1. **按板块判定涨跌停**，不用一刀切 10%：主板 10%、创业板/科创板 20%、
   北交所 30%、ST 5%。偏差超过「板限 + 容差」即视为可疑。
2. **可疑不立刻采信，也不立刻丢弃**：连续 N 个 tick 同向偏离才认定为
   「真实跳空/涨跌停」，否则沿用上一次可信价 —— 单次抖动不驱动风控动作。
3. **停牌可估值、不可交易**：`tradable=False` 只拦新开仓，不拦止损判定。
4. **无昨收时退化**：只做「相对上一次可信价」的跳变检查，宁松不严。
5. 纯函数 + 内存状态，零 I/O、零网络，可在任何进程里用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ======================================================================
# 涨跌停判定
# ======================================================================

ST_KEYWORDS = ("ST", "*ST", "退")


def is_st(name: str) -> bool:
    """名称含 ST / *ST / 退 字样即按 5% 板限处理（宁严不松）。"""
    n = (name or "").upper().replace(" ", "")
    return ("ST" in n) or ("退" in (name or ""))


def board_limit_pct(code: str, name: str = "") -> float:
    """该标的的涨跌停幅度（百分比，正数）。

    A 股现行规则（2026）：
      · 科创板 688/689、创业板 300/301 → 20%
      · 北交所 43/83/87/92 开头 → 30%
      · ST / *ST / 退市整理 → 5%
      · 其余主板 → 10%
    """
    c = str(code or "").strip()
    if is_st(name):
        return 5.0
    if c.startswith(("688", "689", "300", "301")):
        return 20.0
    if c.startswith(("43", "83", "87", "92")):
        return 30.0
    return 10.0


@dataclass
class Verdict:
    """一次报价校验的结论。"""
    ok: bool                 # 价格是否可采信（可用于估值/风控）
    tradable: bool           # 是否可用于下单（停牌/零成交 → False）
    price: float             # 建议采用的价（可疑时=上一次可信价）
    why: str = ""            # 人话说明（写日志/报告）


class QuoteSanityGuard:
    """跨 tick 记住「上一次可信价」，用于识别并吸收异常报价。"""

    def __init__(
        self,
        tolerance_pct: float = 2.0,
        confirm_ticks: int = 2,
        jump_pct: float = 20.0,
        stale_suspect: int = 2,
    ):
        """
        Args:
            tolerance_pct: 在板限之上额外容忍的百分比（价格取整/四舍五入误差）。
            confirm_ticks: 连续几次同向偏离才认定为真实跳空。1=立刻采信。
            jump_pct: 没有昨收时，相对上次可信价的最大允许跳变（%）。
            stale_suspect: 连续多少次可疑后强制采信（防止永久卡在旧价上）。
        """
        self.tolerance_pct = max(0.0, float(tolerance_pct))
        self.confirm_ticks = max(1, int(confirm_ticks))
        self.jump_pct = max(1.0, float(jump_pct))
        self.stale_suspect = max(self.confirm_ticks, int(stale_suspect))
        self._last: Dict[str, float] = {}       # code → 上一次可信价
        self._suspect: Dict[str, tuple] = {}    # code → (方向, 连续次数)

    # ---------------- 状态 ----------------
    def last_price(self, code: str) -> float:
        return float(self._last.get(code) or 0.0)

    def reset(self, code: Optional[str] = None) -> None:
        if code is None:
            self._last.clear()
            self._suspect.clear()
        else:
            self._last.pop(code, None)
            self._suspect.pop(code, None)

    def _accept(self, code: str, price: float) -> None:
        self._last[code] = float(price)
        self._suspect.pop(code, None)

    def _bump(self, code: str, direction: int) -> int:
        prev_dir, prev_n = self._suspect.get(code, (0, 0))
        n = prev_n + 1 if direction == prev_dir else 1
        self._suspect[code] = (direction, n)
        return n

    # ---------------- 主校验 ----------------
    def check(
        self,
        code: str,
        price: float,
        pre_close: Optional[float] = None,
        volume: Optional[float] = None,
        amount: Optional[float] = None,
        name: str = "",
    ) -> Verdict:
        code = str(code or "")
        try:
            price = float(price or 0)
        except (TypeError, ValueError):
            price = 0.0
        last = self.last_price(code)

        # 1) 非正价：直接不可用（不要用 0 覆盖已有估值）
        if price <= 0:
            return Verdict(ok=False, tradable=False, price=last,
                           why="报价缺失/非正")

        # 2) 停牌识别：零成交且价格未动 → 可估值，不可交易
        tradable = True
        note = ""
        try:
            if volume is not None and amount is not None:
                if float(volume) <= 0 and float(amount) <= 0:
                    tradable = False
                    note = "零成交（疑似停牌）"
        except (TypeError, ValueError):
            pass

        # 3) 昨收已知 → 按板块涨跌停判定
        try:
            pc = float(pre_close or 0)
        except (TypeError, ValueError):
            pc = 0.0
        if pc > 0:
            limit = board_limit_pct(code, name) + self.tolerance_pct
            dev = (price / pc - 1.0) * 100.0
            if abs(dev) <= limit:
                self._accept(code, price)
                return Verdict(True, tradable, price, note)
            direction = 1 if dev > 0 else -1
            n = self._bump(code, direction)
            if n >= self.confirm_ticks:
                # 连续确认 → 认定真实跳空/涨跌停，采信并清账
                self._accept(code, price)
                return Verdict(True, tradable, price,
                               f"偏离昨收 {dev:+.2f}% 已连续 {n} 次确认，采信")
            return Verdict(
                False, False, last or price,
                f"偏离昨收 {dev:+.2f}% 超板限 {limit:.1f}%"
                f"（第 {n}/{self.confirm_ticks} 次确认，暂不采信）",
            )

        # 4) 无昨收 → 只做相对上次可信价的跳变检查
        if last > 0:
            jump = abs(price / last - 1.0) * 100.0
            if jump > self.jump_pct:
                direction = 1 if price > last else -1
                n = self._bump(code, direction)
                if n < self.stale_suspect:
                    return Verdict(
                        False, False, last,
                        f"相对上次可信价跳变 {jump:.1f}%（第 {n}/{self.stale_suspect} 次），暂不采信",
                    )
                self._accept(code, price)
                return Verdict(True, tradable, price, f"跳变 {jump:.1f}% 已确认，采信")
        self._accept(code, price)
        return Verdict(True, tradable, price, note)


def filter_prices(prices: Dict[str, float], guard: Optional[QuoteSanityGuard] = None
                  ) -> Dict[str, float]:
    """便捷函数：过滤一份 {code: price}，只保留正价（不做涨跌停判定）。"""
    g = guard or QuoteSanityGuard()
    out: Dict[str, float] = {}
    for c, p in (prices or {}).items():
        v = g.check(c, p)
        if v.ok and v.price > 0:
            out[c] = v.price
    return out

# -*- coding: utf-8 -*-
r"""A 股交易规则：涨跌幅限制、停牌判断、T+1 可卖数量、滑点/冲击成本

补齐三处实盘硬规则（原先缺失，会持续产生废单）：

1. **涨跌停**：涨停时买单基本无法成交、跌停时卖单无法成交。
   原实现完全不判断，导致废单堆积、占用资金/仓位额度。
2. **停牌**：停牌股票不能交易，下单必然失败。
3. **T+1**：A 股当日买入的股票当日不可卖出。原实现在 ``paper.py`` 里
   买入后直接把 ``available`` 设为全量持仓，等于允许当日卖 —— 实盘会废单。

板块涨跌幅限制（现行规则）：
    主板（60/000/001/002/003）10%；ST / *ST 主板股 5%
    创业板（30x）、科创板（688/689）20%
    北交所（4xx/8xx/92x）30%

滑点：真实成交价不会等于盘口价，买入更贵、卖出更便宜。
    原实现按盘口价成交，模拟收益系统性虚高。
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, Union

# 浮点比较容差（价格保留 2 位小数，1 分钱以内视为触及）
_EPS = 0.005


def _code_num(code: str) -> str:
    """从各种格式的证券代码里提取 6 位数字

    兼容 ``600000`` / ``600000.SH`` / ``sh600000`` / ``000001.SZ``。
    """
    m = re.search(r"(\d{6})", str(code or ""))
    return m.group(1) if m else ""


def board_limit_pct(code: str, name: str = "") -> float:
    """返回该股票的涨跌幅限制比例（0.10 表示 ±10%）"""
    num = _code_num(code)

    # 创业板 / 科创板：20%（即使是 ST 也是 20%）
    if num.startswith(("30", "688", "689")):
        return 0.20

    # 北交所：30%
    if num.startswith(("43", "83", "87", "88", "92", "4", "8")):
        return 0.30

    # 主板：ST / *ST 为 5%
    if "ST" in str(name or "").upper():
        return 0.05

    return 0.10


def limit_prices(pre_close: float, code: str, name: str = "") -> Tuple[float, float]:
    """返回 ``(跌停价, 涨停价)``"""
    if not pre_close or pre_close <= 0:
        return 0.0, 0.0
    pct = board_limit_pct(code, name)
    return (
        round(pre_close * (1 - pct), 2),
        round(pre_close * (1 + pct), 2),
    )


def is_limit_up(price: float, pre_close: float, code: str, name: str = "") -> bool:
    """是否涨停（涨停时买入极难成交）"""
    if not price or not pre_close:
        return False
    _, up = limit_prices(pre_close, code, name)
    return up > 0 and price >= up - _EPS


def is_limit_down(price: float, pre_close: float, code: str, name: str = "") -> bool:
    """是否跌停（跌停时卖出极难成交）"""
    if not price or not pre_close:
        return False
    dn, _ = limit_prices(pre_close, code, name)
    return dn > 0 and price <= dn + _EPS


def is_suspended(tick_or_price, volume: Optional[int] = None) -> bool:
    """是否停牌

    接受 ``TickData``，或直接传 ``(price, volume)``。
    判据：无有效最新价，或当日成交量为 0。
    """
    price: Optional[float] = None
    vol: Optional[int] = volume

    if tick_or_price is None:
        return True
    if hasattr(tick_or_price, "last_price"):
        price = getattr(tick_or_price, "last_price", 0.0) or 0.0
        if vol is None:
            vol = getattr(tick_or_price, "volume", None)
    else:
        price = float(tick_or_price or 0)

    if not price or price <= 0:
        return True
    if vol is not None and int(vol) <= 0:
        return True
    return False


# ==================== 下单前置校验 ====================

def check_buyable(
    code: str,
    name: str = "",
    price: float = 0.0,
    pre_close: float = 0.0,
    volume: Optional[int] = None,
) -> Tuple[bool, str]:
    """买入前置校验：停牌 / 涨停 都不该下单"""
    if is_suspended(price if not hasattr(price, "last_price") else price, volume):
        return False, f"⏸ {name or code} 停牌或无有效报价，跳过买入"

    if pre_close and price and is_limit_up(price, pre_close, code, name):
        return False, f"🔒 {name or code} 已涨停({price:.2f})，买入难成交，跳过"

    return True, ""


def check_sellable(
    code: str,
    name: str = "",
    price: float = 0.0,
    pre_close: float = 0.0,
    volume: Optional[int] = None,
) -> Tuple[bool, str]:
    """卖出前置校验：停牌 / 跌停 都不该下单"""
    if is_suspended(price if not hasattr(price, "last_price") else price, volume):
        return False, f"⏸ {name or code} 停牌或无有效报价，跳过卖出"

    if pre_close and price and is_limit_down(price, pre_close, code, name):
        return False, f"🔒 {name or code} 已跌停({price:.2f})，卖出难成交，跳过"

    return True, ""


# ==================== T+1 可卖数量 ====================

def sellable_quantity(position: Union[dict, object]) -> int:
    """计算 T+1 规则下的可卖数量

    兼容两种形态：
      - ``src.trading.broker.Position``（有 available / quantity / today_quantity）
      - 纸交易的 holdings 字典（有 quantity / available / today_quantity）

    取 ``available`` 与 ``quantity - today_quantity`` 的较小值 ——
    双保险：券商返回的 available 若未扣除当日买入，这里也能兜住。
    """
    try:
        if isinstance(position, dict):
            qty = int(position.get("quantity", 0) or 0)
            avail = int(position.get("available", qty) or 0)
            today = int(position.get("today_quantity", 0) or 0)
        else:
            qty = int(getattr(position, "quantity", 0) or 0)
            avail = int(getattr(position, "available", qty) or 0)
            today = int(getattr(position, "today_quantity", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0

    by_t1 = max(0, qty - today)
    return max(0, min(avail, by_t1, qty))


# ==================== 滑点 / 冲击成本 ====================

# 基础滑点（买卖价差 + 撮合延迟），单位：万分之一(bps)
BASE_SLIPPAGE_BPS = 3.0

# A 股最小价格变动单位
MIN_TICK = 0.01

# 冲击成本分档：成交额越大，对市场冲击越强 —— (成交额上限, 冲击率)
_IMPACT_TIERS = [
    (100_000, 0.0002),      # < 10 万：0.02%
    (500_000, 0.0005),      # 10–50 万：0.05%
    (2_000_000, 0.0010),    # 50–200 万：0.10%
    (float("inf"), 0.0015),  # > 200 万：0.15%
]


def impact_rate(amount: float) -> float:
    """按成交额估算冲击成本比例"""
    for cap, rate in _IMPACT_TIERS:
        if amount <= cap:
            return rate
    return _IMPACT_TIERS[-1][1]


def slippage_rate(amount: float, base_bps: float = BASE_SLIPPAGE_BPS) -> float:
    """总滑点比例 = 基础滑点 + 冲击成本"""
    return base_bps / 10_000.0 + impact_rate(amount)


def apply_slippage(price: float, side: str, amount: float,
                   base_bps: float = BASE_SLIPPAGE_BPS) -> float:
    """对委托价施加滑点，返回更接近真实的成交价

    买入往上滑（更贵）、卖出往下滑（更便宜），结果保留 2 位小数。
    ``side`` 接受 ``"buy"/"sell"`` 或 OrderSide。
    """
    if not price or price <= 0:
        return price
    s = str(getattr(side, "value", side)).lower()
    rate = slippage_rate(amount, base_bps)
    if s.startswith("buy"):
        adj = price * (1 + rate)
        # A 股最小变动单位 0.01 元：比例滑点在低价股上会不足 1 分钱而被
        # round() 抹平，这里保证至少滑 1 个 tick，否则滑点等于没生效
        adj = max(adj, price + MIN_TICK)
    else:
        adj = price * (1 - rate)
        adj = min(adj, price - MIN_TICK)
    return round(adj, 2)


def check_t1_sell(position: Union[dict, object], quantity: int) -> Tuple[bool, str]:
    """校验卖出数量是否满足 T+1"""
    name = ""
    code = ""
    if isinstance(position, dict):
        name = position.get("name", "")
        code = position.get("code", "")
    else:
        name = getattr(position, "name", "") or ""
        code = getattr(position, "code", "") or ""

    can = sellable_quantity(position)
    if can <= 0:
        return False, f"🔒 {name or code} 当日无可用仓位（T+1 锁仓），无法卖出"
    if quantity > can:
        return False, (
            f"🔒 {name or code} 可卖 {can} 股（T+1 锁定 "
            f"{int((position.get('quantity', 0) if isinstance(position, dict) else getattr(position, 'quantity', 0)) or 0) - can} 股），"
            f"委托 {quantity} 股超限"
        )
    return True, ""

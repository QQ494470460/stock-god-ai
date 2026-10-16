# -*- coding: utf-8 -*-
"""
闭环全自动交易引擎（Trading Loop）
==================================

把之前"半截"的自动化补成一个真正能自己转起来的闭环：

    ┌──────────────────────────────────────────────────────────────┐
    │  盘前：读昨日 AI 选股 → 生成交易计划（目标组合 + 权重）        │
    │        ↓                                                      │
    │  盘中：每 tick 同步账户/持仓 → 熔断监控 → 止盈止损 → 目标调仓   │
    │        ↓                                                      │
    │  收盘：结算当日盈亏 → 生成日报 → 推送 → 归档                   │
    │        ↓                                                      │
    │  盘后：daily_pick 生成明日候选（已有调度器）→ 回到盘前          │
    └──────────────────────────────────────────────────────────────┘

设计要点
--------
1. **一切下单必经安全中枢**（guard）。未传 guard 时自动构建一个默认的
   纸交易（PAPER）中枢 —— 所以「忘了配 guard」永远不会变成裸奔下单。
2. **计划定数量、guard 定准否**。计划给的股数通过 explicit_quantity 传入，
   guard 仍会用单票上限 / 可用现金 / T+1 可卖量做硬封顶。
3. **纪律优先于观点**。止盈止损与调仓清仓走 risk_exit=True，不被
   「AI 评分仍高不卖」拦下。
4. **行情有兜底**。券商 tick 拿不到（模拟盘/离线）时回退日线收盘价，
   保证纸交易闭环也能离线跑通。
5. **可演练**。dry_run=True 时只产出「本该下什么单」的预演，不真下单。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from loguru import logger

from config.settings import DATA_DIR, config as app_config
from src.trading.broker import BrokerAPI, OrderSide, Position, AccountInfo, Order
from src.trading.guard import RiskSupervisor, build_supervisor, TradingMode


# ----------------------------------------------------------------------------
# 时钟接缝
# ----------------------------------------------------------------------------
# 闭环的全部时间判断（盘中/收盘/已结算日）都走这两个函数，而不是直接调
# datetime.now()。原因有两个，都不是洁癖：
#
# 1. **可测**。`_should_settle` 判断的是墙上时钟「是否已过 15:00」。
#    测试若不能控制时钟，那么同一份测试在 14:00 跑是通过、在 16:00 跑就是
#    失败 —— 这是真实踩过的坑：整组交易闭环用例在收盘后必然全红。
#    时钟依赖的测试不是「偶尔不稳」，是**每天都有大半天必然是错的**。
# 2. **可重放**。交易闭环是安全关键路径，能给出一个确定的时钟，
#    才有可能把某天的行情原样重放一遍。

def _now() -> datetime:
    """当前时刻。测试可 monkeypatch 本函数以冻结时钟。"""
    return datetime.now()


def _today() -> date:
    """当前日期（与 :func:`_now` 同源，避免两者互相打架）。"""
    return _now().date()


# ----------------------------------------------------------------------------
# 状态与配置
# ----------------------------------------------------------------------------

class LoopState(str, Enum):
    STOPPED = "stopped"      # 未启动
    PREOPEN = "preopen"      # 盘前（已备计划，等开盘）
    RUNNING = "running"      # 盘中执行
    HALTED = "halted"        # 熔断冻结
    SETTLED = "settled"      # 当日已结算，等下一交易日
    IDLE = "idle"            # 非交易时段空闲


@dataclass
class LoopConfig:
    """闭环引擎配置（安全默认：纸交易 + 演练关闭调仓前先看预演）"""
    scan_interval_sec: float = 30.0     # 主循环 tick 间隔
    auto_build_plan: bool = True        # 自动生成/刷新交易计划
    top_n: Optional[int] = None          # 计划取前 N 只（None=用 max_positions）
    max_positions: int = 10              # 目标持仓上限（与 guard 对齐）
    cash_reserve_pct: float = 0.10       # 现金保留比例
    weight_cap: float = 0.30             # 单票权重上限（= guard.max_single_position_pct）
    weighting: str = "equal"             # equal / score
    enable_rebalance: bool = True        # 是否执行目标调仓
    enable_stop_orders: bool = True      # 是否执行止盈止损
    band_pct: float = 0.25               # 调仓容忍带
    min_trade_amount: float = 5000.0     # 最小交易金额（元）
    max_orders_per_tick: int = 10        # 单 tick 最多下多少单（防雪崩）
    settle_hour: int = 15                # 收盘结算时刻
    settle_minute: int = 5
    notify: bool = True                  # 是否推送事件/日报
    dry_run: bool = False                # True=只预演不下单
    price_fallback: bool = True          # 无 tick 时回退日线收盘价

    @classmethod
    def from_app_config(cls) -> "LoopConfig":
        """从全局配置的 trading_loop 段构造（缺失则用安全默认）。"""
        try:
            sec = getattr(app_config, "trading_loop", None)
            if sec is None:
                return cls()
            return cls(**{k: v for k, v in asdict(sec).items()
                          if k in cls.__dataclass_fields__})
        except Exception:  # noqa: BLE001
            return cls()


@dataclass
class TickReport:
    """单次 tick 的执行报告（UI / 日志 / 推送共用）"""
    at: str = ""
    state: str = ""
    market_status: str = ""
    plan_date: str = ""
    plan_built: bool = False
    n_positions: int = 0
    equity: float = 0.0
    cash: float = 0.0
    market_value: float = 0.0
    n_stops: int = 0
    n_intents: int = 0
    n_executed: int = 0
    n_blocked: int = 0
    alerts: List[str] = field(default_factory=list)
    actions: List[dict] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = [f"[{self.at}] {self.market_status} · 状态={self.state}"]
        if self.plan_date:
            parts.append(f"计划={self.plan_date}")
        parts.append(f"净值¥{self.equity:,.0f}/现金¥{self.cash:,.0f}/持仓{self.n_positions}只")
        if self.n_stops or self.n_intents or self.n_executed:
            parts.append(
                f"止损{self.n_stops} 意图{self.n_intents} 成交{self.n_executed} 拦截{self.n_blocked}"
            )
        if self.alerts:
            parts.append("告警:" + "; ".join(self.alerts))
        if self.note:
            parts.append(self.note)
        return " | ".join(parts)


# ----------------------------------------------------------------------------
# 闭环引擎
# ----------------------------------------------------------------------------

class TradingLoop:
    """闭环全自动交易引擎。

    典型用法：
        loop = TradingLoop(broker, guard=guard, paper_account=paper)
        loop.run_once()          # 手动跑一个 tick（调试/UI 按钮）
        loop.start()             # 后台线程按 scan_interval 自动跑
        loop.stop()

    未传 guard 时会自动 build_supervisor(...)（默认 PAPER 纸交易），
    确保任何情况下都不会绕过安全中枢直接下单。
    """

    def __init__(
        self,
        broker: BrokerAPI,
        guard: Optional[RiskSupervisor] = None,
        paper_account=None,
        config: Optional[LoopConfig] = None,
        notifier=None,
        price_fn: Optional[Callable[[str], float]] = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
    ):
        self.broker = broker
        self.config = config or LoopConfig.from_app_config()

        # 安全中枢：未提供则自动构建（默认 PAPER，绝不裸奔）
        self.guard = guard or build_supervisor(broker, paper_account=paper_account)
        if paper_account is not None and getattr(self.guard, "paper_account", None) is None:
            self.guard.paper_account = paper_account

        self.notifier = notifier
        self._price_fn = price_fn
        self._on_event = event_callback

        # 计划与状态
        self.plan = None
        self._state: LoopState = LoopState.STOPPED
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._price_cache: Dict[str, float] = {}

        # 结算记账
        self._settled_dates: set = set()
        self._last_report: Optional[TickReport] = None
        self._day_start_equity: Dict[str, float] = {}
        self._day_trades: Dict[str, List[dict]] = {}

        # 事件回调接到 guard，让中枢事件也能透传到 UI
        if self._on_event:
            try:
                self.guard.set_event_callback(self._on_event)
            except Exception:  # noqa: BLE001
                pass

        logger.info(
            f"🔁 TradingLoop 已初始化 | mode={self.guard.safety.mode.value} "
            f"| 间隔={self.config.scan_interval_sec}s | dry_run={self.config.dry_run}"
        )

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _set_state(self, s: LoopState):
        if self._state != s:
            logger.info(f"🔁 闭环状态: {self._state.value} → {s.value}")
            self._state = s
            self._emit("loop_state", {"state": s.value})

    def _emit(self, event: str, data: dict):
        if not self._on_event:
            return
        try:
            self._on_event(event, data)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 账户 / 持仓 / 行情（PAPER 与 LIVE 统一口径）
    # ------------------------------------------------------------------
    def _account(self) -> Optional[AccountInfo]:
        try:
            return self.guard.account()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"取账户失败: {e}")
            return self.broker.query_account()

    def _positions(self) -> List[Position]:
        try:
            return list(self.guard.positions())
        except Exception as e:  # noqa: BLE001
            logger.debug(f"取持仓失败: {e}")
            return list(self.broker.query_positions() or [])

    def _price_of(self, code: str) -> float:
        if code in self._price_cache:
            return self._price_cache[code]
        price = 0.0
        # 1) 券商实时行情
        try:
            tick = self.broker.get_tick(code)
            if tick and getattr(tick, "last_price", 0) > 0:
                price = float(tick.last_price)
        except Exception:  # noqa: BLE001
            pass
        # 2) 兜底：日线收盘价（模拟盘 / 离线可用）
        if price <= 0 and self.config.price_fallback and self._price_fn is None:
            try:
                from src.data.fetcher import fetcher
                df = fetcher.get_daily_kline(code)
                if df is not None and len(df) > 0:
                    price = float(df["close"].iloc[-1])
            except Exception:  # noqa: BLE001
                pass
        # 3) 外部注入的价格函数
        if price <= 0 and self._price_fn is not None:
            try:
                price = float(self._price_fn(code) or 0)
            except Exception:  # noqa: BLE001
                price = 0.0
        self._price_cache[code] = price
        return price

    def _prices_for(self, codes: Sequence[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in codes:
            p = self._price_of(c)
            if p > 0:
                out[c] = p
        return out

    # ------------------------------------------------------------------
    # 计划
    # ------------------------------------------------------------------
    def _build_plan(self):
        from src.trading.trade_plan import build_trade_plan
        cap = float(getattr(self.guard.config, "max_single_position_pct",
                            self.config.weight_cap) or self.config.weight_cap)
        try:
            plan = build_trade_plan(
                top_n=self.config.top_n,
                max_positions=self.config.max_positions,
                cash_reserve_pct=self.config.cash_reserve_pct,
                weighting=self.config.weighting,
                weight_cap=min(cap, self.config.weight_cap),
                write=True,
            )
            return plan
        except Exception as e:  # noqa: BLE001
            logger.error(f"交易计划生成失败: {e}")
            return None

    def _ensure_plan(self, report: TickReport):
        """确保手上有一份**属于今天**的交易计划。

        时点约定（很容易搞错，故写在这里）：计划是**收盘后为下一个交易日**
        生成的，`plan_date` 因此恒等于「生成日的下一个交易日」。所以：

        * 在第 D 天盘中，我们要用的是 `plan_date == D` 的那份（D-1 收盘后写的）；
        * 第 D 天收盘后调度器写出 `plan_date == D+1` 的新计划，此时内存里那份
          仍然是 D，对 D 依然有效 —— 不该被替换掉。

        ⚠️ 这里曾经有一个真 bug：计划**过期后直接重建**，而没有先回磁盘看一眼。
        常驻进程跨到第 D+1 天时，内存里的 D 计划失效 → 立刻 `_build_plan()`
        → 它按 `resolve_target_date()` 生成的是 **D+2** 的计划，
        于是「用后天的计划交易今天」，而调度器昨晚为 D+1 写好的那份
        `trade_plan_D+1.json` 压根没被读过。
        实跑证据：每次 tick 都会重新生成一份计划并重复写盘。

        现在改成：过期时先**回磁盘重读**，只有确实没有「今天的」计划时才重建；
        若手上一份是「为下一个交易日准备的」待生效计划，则保留它 ——
        重建只会得到同一份，白白重复写盘。
        """
        from src.trading.trade_plan import load_latest_plan, load_trade_plan

        today = _today()
        today_s = today.strftime("%Y-%m-%d")

        # ⚠️ 启动时就要按「今天」精确加载，不能用 load_latest_plan()。
        # 后者挑的是**文件最新**的那份，而计划是「收盘后为下一交易日」写的 ——
        # 于是昨晚/今早写好的 trade_plan_明天.json 反而成了"最新"，
        # 闭环一启动就抱着明天的计划交易今天。
        # 实跑证据（2026-10-16）：磁盘上明明有 trade_plan_20261016.json，
        # 闭环却全天显示「计划 2026-10-19」，结算报告 plan_date 也是 2026-10-19。
        if self.plan is None:
            self.plan = load_trade_plan(today_s) or load_latest_plan()

        if not self.config.auto_build_plan:
            return

        if self.plan is not None and self.plan.is_valid_for(today):
            return

        # ① 过期/缺失 → 先回磁盘按**日期**找「今天的」计划（常驻进程跨日必经此处）。
        #    同样不能用 load_latest_plan()：它会拿到明天的那份，
        #    而明天的 is_valid_for(今天) 恒为 False，于是永远找不到今天的。
        fresh = load_trade_plan(today_s)
        if fresh is not None and fresh.is_valid_for(today):
            self.plan = fresh
            report.plan_built = False
            return

        # ② 手上这份是「下一个交易日」的待生效计划 → 保留，别重建
        #    （重建的结果与它同日，只是把同一个文件又写了一遍）
        if self.plan is not None and (self.plan.plan_date or "") > today_s:
            return

        # ③ 真的没有任何可用计划 → 按选股结果生成一份
        self.plan = self._build_plan()
        report.plan_built = True

    # ------------------------------------------------------------------
    # 下单统一入口
    # ------------------------------------------------------------------
    def _place(
        self,
        code: str, name: str, side: OrderSide, quantity: int,
        price: float, reason: str,
        ai_score: float = 50.0, risk_exit: bool = False, tag: str = "loop",
    ) -> tuple:
        """所有下单的唯一出口 —— 必经 guard 裁决。

        Returns: (ok, msg, order, proposal_dict)
        """
        if quantity <= 0:
            return False, "数量为 0，跳过", None, None
        if self.config.dry_run:
            detail = {
                "code": code, "name": name, "side": side.value,
                "quantity": quantity, "price": price, "reason": reason,
                "dry_run": True,
            }
            self._emit("loop_dry_run", detail)
            return True, f"[演练] {side.value} {code} {quantity}股 @ {price:.2f}", None, detail

        prop = self.guard.propose_trade(
            code=code, name=name, side=side,
            ai_score=ai_score, agreement=1.0, signal_strength=1.0,
            strategy_tag=tag, risk_exit=risk_exit,
            explicit_quantity=int(quantity),
        )
        if prop.status == "blocked":
            return False, prop.reason or "被安全中枢拦截", None, {
                "code": code, "side": side.value, "quantity": quantity,
                "reason": prop.reason, "blocked": True,
            }
        ok, msg, order = self.guard.approve_and_execute(
            prop.proposal_id, operator="auto(loop)")
        return ok, msg, order, {
            "code": code, "name": name, "side": side.value,
            "quantity": prop.quantity, "price": prop.price,
            "reason": reason, "ok": ok, "msg": msg,
        }

    # ------------------------------------------------------------------
    # ① 止盈止损
    # ------------------------------------------------------------------
    def _run_exits(self, positions, prices, report: TickReport):
        try:
            alerts = self.guard.executor.check_stop_conditions(
                positions=positions, prices=prices)
        except Exception as e:  # noqa: BLE001
            logger.error(f"止损检查失败: {e}")
            return
        for a in alerts[: self.config.max_orders_per_tick]:
            code = a.get("code", "")
            qty = int(a.get("quantity", 0) or 0)
            price = float(prices.get(code) or self._price_of(code) or 0)
            if qty <= 0 or price <= 0:
                continue
            ok, msg, order, detail = self._place(
                code=code, name=a.get("name", code), side=OrderSide.SELL,
                quantity=qty, price=price, reason=a.get("reason", "止盈止损"),
                ai_score=50.0, risk_exit=True, tag="stop",
            )
            report.n_stops += 1
            report.actions.append(detail or {"code": code, "side": "sell", "ok": ok, "msg": msg})
            if ok:
                report.n_executed += 1
            else:
                report.n_blocked += 1

    # ------------------------------------------------------------------
    # ② 目标组合调仓
    # ------------------------------------------------------------------
    def _run_rebalance(self, positions, account, prices, report: TickReport):
        if self.plan is None or not self.plan.targets:
            report.note = "无有效交易计划，跳过调仓"
            return
        from src.trading.rebalance import compute_rebalance_orders

        intents = compute_rebalance_orders(
            self.plan, positions, account, prices,
            band_pct=self.config.band_pct,
            min_trade_amount=self.config.min_trade_amount,
            allow_buy=True, allow_sell=True,
        )
        report.n_intents = len(intents)
        if not intents:
            return

        score_map = {t.code: t.score for t in self.plan.targets}
        for it in intents[: self.config.max_orders_per_tick]:
            ok, msg, order, detail = self._place(
                code=it.code, name=it.name, side=it.side,
                quantity=it.quantity, price=it.price, reason=it.reason,
                ai_score=float(score_map.get(it.code, 50.0)),
                risk_exit=(it.side == OrderSide.SELL),
                tag="rebalance",
            )
            report.actions.append(detail or {})
            if ok:
                report.n_executed += 1
            else:
                report.n_blocked += 1

    # ------------------------------------------------------------------
    # ③ 收盘结算
    # ------------------------------------------------------------------
    def _should_settle(self, now: datetime) -> bool:
        if now.date() in self._settled_dates:
            return False
        try:
            from src.trading.calendar import is_trading_day
            if not is_trading_day(now.date()):
                return False
        except Exception:  # noqa: BLE001
            pass
        return (now.hour, now.minute) >= (self.config.settle_hour, self.config.settle_minute)

    def _settle(self, account, positions, prices, report: TickReport):
        today = _today().strftime("%Y-%m-%d")
        equity = float(account.total_asset) if account else 0.0
        start_eq = self._day_start_equity.get(today, equity)
        day_pnl = equity - start_eq

        # 成交记录：纸交易读纸账户，实盘读结构化日志
        trades: List[dict] = []
        try:
            pa = getattr(self.guard, "paper_account", None)
            if pa is not None and self.guard.safety.mode == TradingMode.PAPER:
                trades = [
                    (t.to_dict() if hasattr(t, "to_dict") else dict(t.__dict__))
                    for t in pa.get_recent_trades(n=100)
                ]
            else:
                trades = [e for e in self.guard.journal.tail(200)
                          if e.get("event") in ("fill", "order_sent")]
        except Exception as e:  # noqa: BLE001
            logger.debug(f"结算取成交记录失败: {e}")

        snapshot = {
            "date": today,
            "settled_at": _now().isoformat(timespec="seconds"),
            "state": self._state.value,
            "guard_mode": self.guard.safety.mode.value,
            "plan_date": self.plan.plan_date if self.plan else "",
            "equity": round(equity, 2),
            "day_start_equity": round(start_eq, 2),
            "day_pnl": round(day_pnl, 2),
            "day_pnl_pct": round(day_pnl / start_eq * 100, 2) if start_eq else 0.0,
            "cash": round(float(account.available_cash), 2) if account else 0.0,
            "market_value": round(float(account.market_value), 2) if account else 0.0,
            "n_positions": len(positions),
            "positions": [
                {"code": p.code, "name": p.name, "quantity": p.quantity,
                 "avg_cost": round(float(p.avg_cost), 3),
                 "price": round(float(prices.get(p.code) or p.current_price), 3),
                 "pnl_pct": round(float(p.pnl_pct), 2)}
                for p in positions
            ],
            "trades": trades[-100:],
            "stats": {
                "stops": report.n_stops,
                "intents": report.n_intents,
                "executed": report.n_executed,
                "blocked": report.n_blocked,
            },
        }

        # 落盘
        try:
            out_dir = Path(DATA_DIR) / "loop_reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = today.replace("-", "")
            (out_dir / f"loop_report_{stamp}.json").write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            (out_dir / f"loop_report_{stamp}.md").write_text(
                self._settle_markdown(snapshot), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"结算报告写入失败: {e}")

        self._settled_dates.add(_today())
        report.note = (f"✅ 已收盘结算：当日盈亏 ¥{day_pnl:,.2f} "
                       f"({snapshot['day_pnl_pct']:+.2f}%)")

        # 推送
        if self.config.notify:
            self._push("闭环交易日报", self._settle_markdown(snapshot), snapshot)

        self._emit("loop_settled", snapshot)
        logger.info(report.note)

    @staticmethod
    def _settle_markdown(s: dict) -> str:
        lines = [
            f"# 🔁 闭环交易日报 {s['date']}",
            "",
            f"- 结算时间：{s['settled_at']}",
            f"- 安全中枢：{s['guard_mode']} | 计划：{s['plan_date'] or '-'}",
            f"- 总资产：¥{s['equity']:,.2f}（当日 ¥{s['day_pnl']:,.2f} / "
            f"{s['day_pnl_pct']:+.2f}%）",
            f"- 现金：¥{s['cash']:,.2f} | 持仓市值：¥{s['market_value']:,.2f}",
            f"- 持仓 {s['n_positions']} 只"
            f"（止损{s['stats']['stops']} / 意图{s['stats']['intents']} / "
            f"成交{s['stats']['executed']} / 拦截{s['stats']['blocked']}）",
            "",
        ]
        if s["positions"]:
            lines.append("| 代码 | 名称 | 数量 | 成本 | 现价 | 盈亏% |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for p in s["positions"]:
                lines.append(
                    f"| {p['code']} | {p['name']} | {p['quantity']} | "
                    f"{p['avg_cost']} | {p['price']} | {p['pnl_pct']:+.2f} |")
        else:
            lines.append("_（空仓）_")
        return "\n".join(lines)

    def _push(self, title: str, markdown: str, payload: dict):
        if not self.notifier:
            return
        try:
            self.notifier.send(title, markdown, payload=payload)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"推送失败（已忽略）: {e}")

    # ------------------------------------------------------------------
    # 主 tick
    # ------------------------------------------------------------------
    def tick(self) -> TickReport:
        """执行一个完整的闭环周期。线程安全（可重入保护）。"""
        with self._lock:
            self._price_cache.clear()
            now = _now()
            report = TickReport(
                at=now.strftime("%H:%M:%S"),
                state=self._state.value,
            )

            # --- 市场阶段 ---
            phase, trading_ok, why = self._phase(now)
            report.market_status = phase

            # --- ① 计划 ---
            try:
                self._ensure_plan(report)
            except Exception as e:  # noqa: BLE001
                logger.error(f"准备交易计划失败: {e}")
            report.plan_date = self.plan.plan_date if self.plan else ""

            # --- ② 同步账户/持仓 ---
            account = self._account()
            positions = self._positions()
            report.n_positions = len(positions)
            if account:
                report.equity = float(account.total_asset)
                report.cash = float(account.available_cash)
                report.market_value = float(account.market_value)
                today = _today().strftime("%Y-%m-%d")
                self._day_start_equity.setdefault(today, report.equity)

            # --- ③ 熔断监控 ---
            if self.guard is not None:
                try:
                    alerts = self.guard.tick_monitor()
                    report.alerts.extend(alerts or [])
                except Exception as e:  # noqa: BLE001
                    logger.error(f"熔断监控异常: {e}")
                if self.guard.safety.halted:
                    self._set_state(LoopState.HALTED)
                    report.state = self._state.value
                    report.note = "🛑 安全中枢已熔断，停止一切交易"
                    self._push("⛔ 闭环交易已熔断",
                               f"熔断原因：{self.guard.safety._state.halt_reason}\n"
                               f"时间：{now.isoformat(timespec='seconds')}",
                               {"halted": True})
                    self._last_report = report
                    return report

            # --- ④ 收盘结算（不受交易时段限制） ---
            if self._should_settle(now):
                self._prices_for([p.code for p in positions])
                prices = dict(self._price_cache)
                self._settle(account, positions, prices, report)
                self._set_state(LoopState.SETTLED)
                report.state = self._state.value
                self._last_report = report
                return report

            # --- ⑤ 非交易时段：待命 ---
            if not trading_ok:
                self._set_state(
                    LoopState.PREOPEN if phase in ("盘前", "开盘集合竞价") else LoopState.IDLE)
                report.state = self._state.value
                report.note = why
                self._last_report = report
                return report

            self._set_state(LoopState.RUNNING)
            report.state = self._state.value

            # --- ⑥ 准备价格 ---
            need = list({p.code for p in positions} |
                        ({t.code for t in self.plan.targets} if self.plan else set()))
            prices = self._prices_for(need)

            # --- ⑦ 止盈止损（优先，纪律性离场） ---
            if self.config.enable_stop_orders:
                self._run_exits(positions, prices, report)

            # --- ⑧ 目标组合调仓 ---
            if self.config.enable_rebalance and account:
                self._run_rebalance(positions, account, prices, report)

            # --- ⑨ 事件沉淀 ---
            if report.n_executed:
                self._emit("loop_cycle", report.to_dict())
            self._last_report = report
            return report

    def _phase(self, now: datetime) -> tuple:
        """返回 (市场阶段文本, 是否可下单, 原因)"""
        try:
            from src.trading import calendar as cal
            phase = cal.market_status(now)
            ok, why = cal.order_allowed(now)
            return phase, ok, why
        except Exception as e:  # noqa: BLE001
            logger.debug(f"日历不可用，按工作日兜底: {e}")
            is_wd = now.weekday() < 5
            t = now.time()
            from datetime import time as _t
            in_hours = (_t(9, 30) <= t <= _t(11, 30)) or (_t(13, 0) <= t <= _t(15, 0))
            return ("工作日盘中(兜底)" if in_hours else "非交易时段(兜底)"), (is_wd and in_hours), "日历降级"

    # ------------------------------------------------------------------
    # 运行控制
    # ------------------------------------------------------------------
    def start(self, interval_seconds: Optional[float] = None):
        """启动后台循环线程。"""
        if self.is_running:
            logger.warning("TradingLoop 已在运行")
            return
        interval = float(interval_seconds or self.config.scan_interval_sec)
        self._stop_event.clear()

        def _loop():
            logger.info(f"🚀 闭环自动交易启动（每 {interval:.0f}s 一个周期）")
            while not self._stop_event.is_set():
                try:
                    rep = self.tick()
                    if rep.n_executed or rep.n_stops or rep.alerts:
                        logger.info("🔁 " + rep.summary())
                except Exception as e:  # noqa: BLE001
                    logger.error(f"闭环周期异常: {e}")
                self._stop_event.wait(interval)
            logger.info("🛑 闭环自动交易已停止")

        self._thread = threading.Thread(target=_loop, daemon=True, name="TradingLoop")
        self._thread.start()

    def stop(self, timeout: float = 10.0):
        """停止后台循环。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._set_state(LoopState.STOPPED)

    def run_once(self) -> TickReport:
        """手动跑一个周期（不下单也可用于预览）。"""
        return self.tick()

    # ------------------------------------------------------------------
    # 预演 / 状态
    # ------------------------------------------------------------------
    def preview(self) -> dict:
        """不产生任何副作用的「本该下什么单」预演。"""
        from src.trading.rebalance import summarize_intents, compute_rebalance_orders

        self._ensure_plan(TickReport())
        account = self._account()
        positions = self._positions()
        need = list({p.code for p in positions} |
                    ({t.code for t in self.plan.targets} if self.plan else set()))
        self._price_cache.clear()
        prices = self._prices_for(need)

        intents = compute_rebalance_orders(
            self.plan, positions, account, prices,
            band_pct=self.config.band_pct,
            min_trade_amount=self.config.min_trade_amount,
        ) if (self.plan and account) else []

        stops = []
        if account is not None:
            try:
                stops = self.guard.executor.check_stop_conditions(
                    positions=positions, prices=prices)
            except Exception:  # noqa: BLE001
                stops = []

        return {
            "plan": self.plan.to_dict() if self.plan else None,
            "plan_summary": self.plan.summary() if self.plan else "（无计划）",
            "positions": [
                {"code": p.code, "name": p.name, "quantity": p.quantity,
                 "pnl_pct": round(float(p.pnl_pct), 2)} for p in positions
            ],
            "equity": round(float(account.total_asset), 2) if account else 0.0,
            "cash": round(float(account.available_cash), 2) if account else 0.0,
            "stops": stops,
            "intents": [i.to_dict() for i in intents],
            "intent_summary": summarize_intents(intents),
            "guard_mode": self.guard.safety.mode.value,
            "dry_run": self.config.dry_run,
        }

    def get_status(self) -> dict:
        """给 UI 的状态快照。"""
        g = None
        try:
            g = self.guard.get_status()
        except Exception:  # noqa: BLE001
            pass
        return {
            "running": self.is_running,
            "state": self._state.value,
            "dry_run": self.config.dry_run,
            "interval": self.config.scan_interval_sec,
            "plan_date": self.plan.plan_date if self.plan else "",
            "plan_targets": len(self.plan.targets) if self.plan else 0,
            "settled_today": _today() in self._settled_dates,
            "guard": g,
            "last_report": self._last_report.to_dict() if self._last_report else None,
        }


if __name__ == "__main__":  # 手动跑一个周期
    from src.trading.yuekai import YuekaiBroker
    b = YuekaiBroker()
    b.connect(mode="sim")
    lp = TradingLoop(b, config=LoopConfig(dry_run=True))
    rep = lp.run_once()
    print(rep.summary())
    print(json.dumps(lp.preview(), ensure_ascii=False, indent=2, default=str)[:2000])

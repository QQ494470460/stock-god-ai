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
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from loguru import logger

from src.core.single_instance import SingleInstanceLock

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
    # 组合回撤 → 连续渐进降仓（**不是**清仓）。回测实证：触线全清能把回撤
    # 从 -34% 压到 -14%，但年化也从 +26% 掉到 +4.7%、夏普腰斩，故保留敞口在场。
    enable_risk_scaling: bool = True
    risk_scale_floor: float = 0.30       # 最深回撤时仍保留的敞口比例（绝不为 0）
    band_pct: float = 0.25               # 调仓容忍带
    min_trade_amount: float = 5000.0     # 最小交易金额（元）
    max_orders_per_tick: int = 10        # 单 tick 最多下多少单（防雪崩）
    settle_hour: int = 15                # 收盘结算时刻
    settle_minute: int = 5
    notify: bool = True                  # 是否推送事件/日报
    dry_run: bool = False                # True=只预演不下单
    price_fallback: bool = True          # 无 tick 时回退日线收盘价
    heartbeat_file: str = ""             # tick 心跳落盘路径（空=data/loop_heartbeat.json）

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
        lock_path: Optional[str] = None,
    ):
        self.broker = broker

        # 单实例锁：传了 lock_path 就表示「本实例启动时必须先抢到这把锁」。
        # 看板内嵌闭环与常驻 run_loop.py 共用 data/.run_loop.lock，
        # 抢不到 = 已经有一个实例在跑 → 拒绝启动，避免重复下单。
        self._instance_lock = SingleInstanceLock(lock_path) if lock_path else None
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

        # 日内累计记账：结算报告的 stats 必须反映「一整天」的止损/意图/成交/拦截，
        # 而不是「结算那一刻」的单 tick。每 tick 的计数在盘中累加，结算时一并落盘，
        # 然后清零（跨日也自动重置，见 tick() 顶部的日期切换判断）。
        self._day_stats_key: str = ""
        self._day_stats: Dict[str, int] = {
            "stops": 0, "intents": 0, "executed": 0, "blocked": 0,
        }

        # ⚠️ 这两项原本是**纯内存**状态，常驻进程崩溃/被开机自启重启后会丢失：
        #   · _settled_dates 丢失 → 收盘后重复结算，把当天真实的拦截/成交计数清零、
        #     并重推一遍日报；同时 get_status().settled_today 永远 False（--until-close 永不退）。
        #   · _day_start_equity 丢失 → 重启后当日盈亏从 0 算起（day_pnl 失真）。
        # 无人值守闭环的设计就是要自动重启（≤5 次），所以必须把它们持久化到磁盘。
        # 即便持久化失败，_should_settle 还有「今日报告文件已存在即视为已结算」的
        # 兜底幂等保护（见 _should_settle）。
        self._runtime_file: Path = (
            Path(self.config.heartbeat_file).parent / "loop_runtime.json"
            if self.config.heartbeat_file
            else Path(DATA_DIR) / "loop_runtime.json"
        )
        self._halt_notified: bool = False   # 熔断通知只发一次，解除后再熔断才重发
        self._load_runtime()
        # 纸账户本金基线（initial_cash）的上一 tick 记忆：用于检测「重置」这类
        # 非交易性基线变更并自动对齐风控净值基线，避免误熔断。
        self._last_pa_baseline: Optional[float] = None

        # 心跳：每次 tick 落盘一次，用于回答「今天到底有没有在跑」。
        self.heartbeat_file: Path = (
            Path(self.config.heartbeat_file) if self.config.heartbeat_file
            else (Path(DATA_DIR) / "loop_heartbeat.json")
        )

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
    # 心跳（判断「没跑」还是「跑了但没成交」）
    # ------------------------------------------------------------------
    def _write_heartbeat(self, report: TickReport) -> None:
        """把最近一次 tick 的摘要落盘。

        为什么需要：闭环「①选股→②计划→③交易→④业绩→⑤进化」里，③交易这一步
        没有任何定时任务驱动，全靠人工启动 `run_loop.py` 或点看板按钮。10-14~10-16
        三天的结算报告都是 ``intents=0``，事后翻日志才分清到底是「根本没跑」还是
        「跑了但没满足调仓条件」—— 这两种情况的处置方式完全不同。

        心跳文件字段：时间戳、状态、市场阶段、计划日、权益、意图/成交/拦截计数、pid。
        读取用模块级 :func:`read_heartbeat`。
        """
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "state": report.state,
            "market_status": report.market_status,
            "plan_date": report.plan_date,
            "n_positions": int(report.n_positions),
            "equity": round(float(report.equity or 0), 2),
            "n_intents": int(report.n_intents),
            "n_executed": int(report.n_executed),
            "n_blocked": int(report.n_blocked),
            "dry_run": bool(self.config.dry_run),
            "guard_mode": self.guard.safety.mode.value,
        }
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # 跨重启持久化（常驻进程崩溃/被开机自启重启后，结算状态 / 当日起始权益不能丢）
    # ------------------------------------------------------------------
    def _load_runtime(self) -> None:
        """把 _settled_dates / _day_start_equity 从磁盘读回（无则保持内存默认值）。

        为什么需要：这两项原先是纯内存状态。无人值守闭环的设计就是要**自动重启**
        （开机自启 .bat 在崩溃后 ≤5 次内自动拉起），重启后内存清零会导致：

        * 重复结算：把当天真实的拦截/成交计数清零、并重推一遍日报；
        * ``settled_today`` 永远 False（``--until-close`` 永不退出）；
        * 当日盈亏从 0 算起（day_pnl 失真）。

        读盘失败（文件不存在/损坏）一律静默忽略 —— 宁可退化成「未结算/从零计」，
        也绝不让一次读盘异常把整个 tick 打挂（交易主流程优先级最高）。
        """
        try:
            if not self._runtime_file.exists():
                return
            data = json.loads(self._runtime_file.read_text(encoding="utf-8"))
            self._day_start_equity = {
                k: float(v) for k, v in (data.get("day_start_equity") or {}).items()
            }
            self._settled_dates = set(d for d in (data.get("settled_dates") or []) if d)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"闭环运行态读盘失败（已忽略，按内存默认值）: {e}")

    def _save_runtime(self) -> None:
        """把 _settled_dates / _day_start_equity 落盘（仅在状态变化时调用，频率极低）。"""
        try:
            self._runtime_file.parent.mkdir(parents=True, exist_ok=True)
            self._runtime_file.write_text(
                json.dumps({
                    "day_start_equity": self._day_start_equity,
                    "settled_dates": sorted(self._settled_dates),
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"闭环运行态落盘失败（已忽略）: {e}")

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
        # 2) 公开实时行情兜底（券商 tick 不可用时的**首选**兜底）
        #    ⚠️ 这一步不能省，也不能排在日线后面。2026-10-20 实测：模拟盘券商
        #    tick 恒不可用，于是估值一路退到日线 close —— 盘中那就是**昨收**，
        #    而买入用的是实时价。结果每笔持仓一买进去就"浮亏" 1%~3%，
        #    9 笔合计凭空 -¥5.5 万，直接把单日亏损熔断打爆（闭环当天停摆）。
        #    买入口径与估值口径必须同源，否则风控读到的是幽灵亏损。
        if price <= 0 and self.config.price_fallback:
            try:
                from src.data.fetcher import fetcher
                q = fetcher.get_realtime_quote(code)
                if isinstance(q, dict):
                    p = float(q.get("最新价") or 0)
                    if p > 0:
                        price = p
                elif q is not None:
                    p = float(getattr(q, "last_price", 0) or getattr(q, "price", 0) or 0)
                    if p > 0:
                        price = p
            except Exception:  # noqa: BLE001
                pass
        # 3) 兜底：日线收盘价（离线 / 盘后可用；盘中会退化为昨收，仅最后手段）
        if price <= 0 and self.config.price_fallback and self._price_fn is None:
            try:
                from src.data.fetcher import fetcher
                df = fetcher.get_daily_kline(code)
                if df is not None and len(df) > 0:
                    price = float(df["close"].iloc[-1])
            except Exception:  # noqa: BLE001
                pass
        # 4) 外部注入的价格函数
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

    def _mark_paper_to_market(self, prices: Dict[str, float]) -> bool:
        """把纸账户持仓按最新价 mark 一遍，返回是否真的动了。

        为什么必须有：纸账户的 ``last_price`` 只在买入/卖出时更新，平时没有任何
        人推它 —— 于是「持仓现价」可能停在上一次成交那一刻。实测 600519 停在
        1501.95，而当日实际收盘 1329.00，同一屏上出现：

          · 持仓卡片显示浮盈 +0.01%（用的是纸账户旧价）
          · 风控止损判定 -11.5%（用的是实时取价）

        两个数字互相矛盾，看到的人只会以为程序疯了。统一在拿到行情后 mark，
        权益 / 盈亏 / 止损三个口径就对上了。

        ⚠️ 只用 >0 的价，缺价不覆盖 —— 行情拿不到时保持上一次的估值，
        绝不能把持仓市值打成 0。
        """
        pa = getattr(self.guard, "paper_account", None)
        if pa is None:
            return False
        clean = {k: float(v) for k, v in (prices or {}).items() if v and float(v) > 0}
        if not clean:
            return False
        try:
            before = {c: h.get("last_price") for c, h in pa.holdings.items()}
            pa.mark_to_market(clean)
            changed = any(pa.holdings[c].get("last_price") != before.get(c)
                          for c in pa.holdings)
            # 必须落盘：看板是另一个进程，只改内存的话它永远读到旧价，
            # 同一屏上会出现「持仓浮盈 +x%」与「当日亏损 -y%」互相矛盾。
            if changed and hasattr(pa, "save"):
                try:
                    pa.save()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"纸账户 mark 后落盘失败（已忽略）: {e}")
            return changed
        except Exception as e:  # noqa: BLE001
            logger.debug(f"纸账户 mark_to_market 失败（已忽略）: {e}")
            return False

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

        risk_scale = self._portfolio_risk_scale(account)
        intents = compute_rebalance_orders(
            self.plan, positions, account, prices,
            band_pct=self.config.band_pct,
            min_trade_amount=self.config.min_trade_amount,
            allow_buy=True, allow_sell=True,
            risk_scale=risk_scale,
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

    def _portfolio_risk_scale(self, account) -> float:
        """按组合当前回撤推导敞口系数（连续渐进降仓，非清仓）。

        用 guard 记录的峰值权益与当前权益算峰谷回撤，再交给
        ``sizing.portfolio_risk_scale`` 线性映射到 [floor, 1.0]。

        取不到峰值/权益时返回 1.0（不降仓）—— **数据缺失不能当成「回撤很深」**
        而误砍仓位，这一点与「未知波动率时降级为固定比例」同源。
        """
        if not self.config.enable_risk_scaling:
            return 1.0
        try:
            from src.trading.sizing import drawdown_pct, portfolio_risk_scale

            peak = float(getattr(self.guard.safety._state, "peak_equity", 0) or 0)
            equity = float(getattr(account, "total_asset", 0) or 0)
            if peak <= 0 or equity <= 0:
                return 1.0
            dd = drawdown_pct(equity, peak)
            scale = portfolio_risk_scale(
                dd,
                self.guard.config.max_portfolio_drawdown_pct,
                floor=self.config.risk_scale_floor,
            )
            if scale < 1.0:
                logger.info(
                    f"📉 组合回撤 {dd:.2f}% → 敞口系数 {scale:.2f}"
                    f"（连续降仓，保留 {scale:.0%} 仓位在场）"
                )
            return scale
        except Exception as e:  # noqa: BLE001
            logger.debug(f"敞口系数计算失败，按满仓处理：{e}")
            return 1.0

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
        if (now.hour, now.minute) < (self.config.settle_hour, self.config.settle_minute):
            return False
        # 幂等兜底：今日结算报告已落盘即视为已结算。这是防止**常驻进程崩溃/重启后**
        # 重复结算的关键 —— 即使运行态文件读盘失败，磁盘上那份报告也能拦住第二次结算，
        # 避免把当天真实的拦截/成交计数清零、并重推日报。
        stamp = now.date().strftime("%Y%m%d")
        report_file = Path(DATA_DIR) / "loop_reports" / f"loop_report_{stamp}.json"
        if report_file.exists():
            self._settled_dates.add(now.date())     # 同步内存，使本进程与 get_status() 一致
            try:
                self._save_runtime()
            except Exception:  # noqa: BLE001
                pass
            return False
        return True

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
                # 用「当日累计」而非「结算那一刻」的 tick 计数，
                # 否则盘中发生的止损/拦截/调仓意图全都不会出现在报告里。
                "stops": self._day_stats.get("stops", 0),
                "intents": self._day_stats.get("intents", 0),
                "executed": self._day_stats.get("executed", 0),
                "blocked": self._day_stats.get("blocked", 0),
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
        # 落盘：常驻进程重启后不会重复结算 / --until-close 能正确退出
        try:
            self._save_runtime()
        except Exception:  # noqa: BLE001
            pass
        # 结算报告已落盘，当日累计清零（键保持为今天，避免同一天反复重置）
        self._day_stats = {
            "stops": 0, "intents": 0, "executed": 0, "blocked": 0,
        }
        self._day_stats_key = today
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
        """执行一个完整的闭环周期，并落盘一次心跳。

        心跳失败绝不影响交易主流程（写盘异常会被吞掉并记 debug）。
        """
        report = self._tick_impl()
        try:
            self._write_heartbeat(report)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"心跳写入失败（已忽略）: {e}")
        return report

    def _tick_impl(self) -> TickReport:
        """执行一个完整的闭环周期。线程安全（可重入保护）。"""
        with self._lock:
            self._price_cache.clear()
            # 纸账户权威态以磁盘为准：外部（看板/用户）可能重置或手动调整了纸账户，
            # 若不每 tick 从磁盘重载，常驻进程会一直用内存里的陈旧副本交易 ——
            # 既忽略用户的重置操作，又会在「重置改了 initial_cash」后触发误熔断。
            _pa = getattr(self.guard, "paper_account", None)
            if _pa is not None and hasattr(_pa, "reload"):
                try:
                    _pa.reload()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"纸账户重载失败（已忽略）: {e}")
            # 纸账户基线变更检测：initial_cash 是「本金基线」，全代码里只有
            # reset() 会改它。若本 tick 与上个 tick 的基线不同，说明用户/看板
            # 在常驻进程运行期间重置了账户（清仓+改本金）—— 这是「与交易无关」
            # 的资金基数变更，必须同步把风控的净值基线（peak_equity / 当日起始）
            # 对齐到新权益，否则旧 peak 远高于新权益会立刻被误判成腰斩→熔断，
            # 且无法自愈（实测：纸账户 10 万 vs 遗留 peak 100 万 → 首 tick 报
            # 89.95% 回撤直接 halt，把当天剩余时间全冻结）。
            # 注意：真实腰斩只会改变 cash/market_value，绝不会动 initial_cash，
            # 所以用 initial_cash 作判据不会把真回撤误当成重置而静默放宽熔断。
            if _pa is not None and hasattr(_pa, "get_total_asset"):
                _cur_baseline = getattr(_pa, "initial_cash", None)
                if _cur_baseline is not None:
                    if self._last_pa_baseline is None:
                        self._last_pa_baseline = _cur_baseline
                    elif abs(_cur_baseline - self._last_pa_baseline) > 1e-6:
                        try:
                            _new_eq = _pa.get_total_asset()
                            if self.guard is not None and getattr(self.guard, "safety", None) is not None:
                                self.guard.safety.rebaseline_peak(
                                    _new_eq, operator="autoloop",
                                    reason=(
                                        f"检测到纸账户重置(initial_cash "
                                        f"{self._last_pa_baseline:,.0f}→{_cur_baseline:,.0f})，"
                                        f"自动对齐风控净值基线，避免误熔断"
                                    ),
                                )
                                logger.warning(
                                    f"♻️ 因纸账户重置自动对齐风控基线：净值基数→"
                                    f"¥{_new_eq:,.0f}（原 ¥{self._last_pa_baseline:,.0f}）"
                                )
                            self._last_pa_baseline = _cur_baseline
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"纸账户重置后对齐风控基线失败（下个 tick 会重试）: {e}")
            now = _now()
            report = TickReport(
                at=now.strftime("%H:%M:%S"),
                state=self._state.value,
            )

            # 跨日 → 日内累计清零（避免把昨天的拦截/成交带进今天的结算报告）
            today_s = now.date().strftime("%Y-%m-%d")
            if today_s != self._day_stats_key:
                self._day_stats = {
                    "stops": 0, "intents": 0, "executed": 0, "blocked": 0,
                }
                self._day_stats_key = today_s

            # --- 市场阶段 ---
            phase, trading_ok, why = self._phase(now)
            report.market_status = phase

            # --- 市场状态(regime)感知：每交易日推算一次，推给 guard 做策略动态加权 ---
            try:
                self._update_regime()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"市场状态感知跳过（已忽略）: {e}")

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
                if today not in self._day_start_equity:
                    self._day_start_equity[today] = report.equity
                    # 落盘：常驻进程重启后当日起始权益仍正确（否则 day_pnl 从 0 算起）
                    self._save_runtime()

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
                    # 关键修正：熔断期间仍执行「纪律性离场」（止损/止盈）。
                    # 自动熔断只应冻结新开仓/调仓，不能冻结斩仓——否则亏损会越拖
                    # 越大（实测曾因此出现 32% 回撤）。这里按最新价 mark 后跑一遍
                    # 止损检查；能否成交由 guard 判定（人工 KILL SWITCH 仍全冻）。
                    try:
                        need = [p.code for p in positions]
                        prices = self._prices_for(need) if need else {}
                        if self._mark_paper_to_market(prices):
                            positions = self._positions()
                            account = self._account()
                            if account:
                                report.equity = float(account.total_asset)
                                report.cash = float(account.available_cash)
                                report.market_value = float(account.market_value)
                        if self.config.enable_stop_orders:
                            self._run_exits(positions, prices, report)
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"熔断期间止损检查异常: {e}")
                    report.note = "🛑 安全中枢已熔断（冻结新开仓/调仓；止损离场照常执行）"
                    # 只在「进入熔断」那一刻推一次，避免每个 tick 都推（默认 notify=True，
                    # 一旦接了通知渠道就会变成通知轰炸）。熔断解除/重启后会重置该标志，
                    # 下一次再熔断可重新推送。
                    if not self._halt_notified:
                        self._halt_notified = True
                        self._push("⛔ 闭环交易已熔断",
                                   f"熔断原因：{self.guard.safety._state.halt_reason}\n"
                                   f"时间：{now.isoformat(timespec='seconds')}",
                                   {"halted": True})
                    self._last_report = report
                    return report
                # 未熔断 → 允许下次再触发时重新推送（自愈/手动恢复后重熔断会再报一次）
                self._halt_notified = False

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

            # --- ⑥.5 纸账户按最新价 mark ---
            # 必须在止损/调仓之前：否则「持仓浮盈」用旧价、「止损判定」用实时价，
            # 同一份报告里两个口径互相矛盾（实测 +0.01% vs -11.5%）。
            if self._mark_paper_to_market(prices):
                positions = self._positions()
                account = self._account()
                report.n_positions = len(positions)
                if account:
                    report.equity = float(account.total_asset)
                    report.cash = float(account.available_cash)
                    report.market_value = float(account.market_value)

            # --- ⑦ 止盈止损（优先，纪律性离场） ---
            if self.config.enable_stop_orders:
                self._run_exits(positions, prices, report)

            # --- ⑧ 目标组合调仓 ---
            if self.config.enable_rebalance and account:
                self._run_rebalance(positions, account, prices, report)

            # 日内累计：把本 tick 的止损/意图/成交/拦截并入「当日」计数器，
            # 供收盘结算报告使用（否则只记结算那一刻，盘中拦截全丢）。
            self._day_stats["stops"] += report.n_stops
            self._day_stats["intents"] += report.n_intents
            self._day_stats["executed"] += report.n_executed
            self._day_stats["blocked"] += report.n_blocked

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

    def _update_regime(self) -> None:
        """尽力而为地推算当前市场状态(regime)并推给 guard 做策略动态加权。

        设计：每天只算一次（``_regime_day`` 去重）；取不到指数日线（本机无网络 /
        非交易时段接口受限）时静默跳过，regime 保持默认 sideways（权重≈1.0）。
        任何异常都被上层 try 吞掉，绝不因「感知不到市场」而中断交易闭环。

        Batch 3 扩展：同一天里（在 regime 去重守卫之前）还会——

        - 从纸账户成交历史增量同步「策略信誉」（``guard.sync_credibility_from_paper``），
          让 Feature #7 的多策略集成随真实战绩自我调权；与指数感知相互独立，
          任一步失败都不影响另一步。
        - 把基准近期日波动喂给 guard（``guard.set_market_vol``），供「波动自适应
          熔断线」动态缩放单日亏损熔断阈值。
        """
        if self.guard is None:
            return
        today_s = _today().strftime("%Y-%m-%d")
        if getattr(self, "_regime_day", "") == today_s:
            return  # 一天只推算一次
        # 每日一次：同步策略信誉（独立于指数感知，失败不影响后续）
        try:
            pa = getattr(self.guard, "paper_account", None)
            self.guard.sync_credibility_from_paper(pa)
        except Exception:  # noqa: BLE001
            pass
        self._regime_day = today_s
        try:
            from datetime import timedelta
            from src.data.fetcher import get_benchmark_kline
            from src.trading.regime import detect_regime
        except Exception:  # noqa: BLE001
            return
        start = (_today() - timedelta(days=450)).strftime("%Y-%m-%d")
        bench = get_benchmark_kline("000300.SH", start_date=start)
        if bench is None or len(bench) < 60:
            return
        regime = detect_regime(bench)
        if regime:
            self.guard.set_regime(regime)
            logger.info(f"🌐 市场状态感知更新：{regime}（策略权重已按状态调整）")
        # 基准近期日波动 → 波动自适应熔断线（独立于 regime 是否判定成功）
        try:
            import numpy as _np
            closes = bench["close"].astype(float).to_numpy()
            if len(closes) >= 5:
                r = _np.diff(_np.log(closes[-30:]))
                if len(r) > 0:
                    self.guard.set_market_vol(float(_np.std(r, ddof=0)) * 100.0)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 运行控制
    # ------------------------------------------------------------------
    def start(self, interval_seconds: Optional[float] = None) -> bool:
        """启动后台循环线程。

        返回 True=已启动，False=**拒绝启动**（另一个实例持有单实例锁）。
        调用方（看板「启动闭环」按钮）必须检查返回值并提示用户，
        否则两个实例会对同一份计划各下一遍单。
        """
        if self.is_running:
            logger.warning("TradingLoop 已在运行")
            return True
        if self._instance_lock is not None and not self._instance_lock.acquire():
            logger.error(
                f"⛔ 已有另一个闭环实例在运行（单实例锁 {self._instance_lock.path} "
                f"被占用），拒绝启动第二个实例 —— 否则会对同一份计划重复下单"
            )
            return False
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
        return True

    def stop(self, timeout: float = 10.0):
        """停止后台循环，并释放单实例锁（让别的实例可以接手）。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._set_state(LoopState.STOPPED)
        if self._instance_lock is not None:
            self._instance_lock.release()

    def run_once(self) -> TickReport:
        """手动跑一个周期（不下单也可用于预览）。"""
        return self.tick()

    # ------------------------------------------------------------------
    # 预演 / 状态
    # ------------------------------------------------------------------
    def preview(self) -> dict:
        """预演「本该下什么单」：不下单、不写盘（不产生任何持久化副作用）。

        注意：会顺带把纸账户按最新价 mark 一遍（仅内存），
        这样预演出来的盈亏/止损口径与真实 tick 一致。
        """
        from src.trading.rebalance import summarize_intents, compute_rebalance_orders

        self._ensure_plan(TickReport())
        account = self._account()
        positions = self._positions()
        need = list({p.code for p in positions} |
                    ({t.code for t in self.plan.targets} if self.plan else set()))
        self._price_cache.clear()
        prices = self._prices_for(need)

        # 预演也要 mark：否则预览出来的持仓盈亏和止损判定同样会打架
        if self._mark_paper_to_market(prices):
            account = self._account()
            positions = self._positions()

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
            "heartbeat": read_heartbeat(self.heartbeat_file),
        }


def read_heartbeat(path: Optional[Path] = None) -> dict:
    """读取 tick 心跳（缺失/损坏返回空 dict）。

    附带 ``age_sec``（距现在多少秒），调用方据此判断「是否还活着」。
    """
    p = Path(path) if path else (Path(DATA_DIR) / "loop_heartbeat.json")
    try:
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = data.get("ts")
        if ts:
            try:
                data["age_sec"] = round(
                    (datetime.now() - datetime.fromisoformat(ts)).total_seconds(), 1)
            except Exception:  # noqa: BLE001
                data["age_sec"] = None
        return data
    except Exception:  # noqa: BLE001
        return {}


def is_resident_loop_running(stale_sec: float = 120.0) -> dict:
    """判断是否有独立的常驻 ``run_loop.py`` 进程在跑（与看板 in-process 线程抢同一把锁）。

    看板里的 ``TradingLoop`` 是 in-process 后台线程，``is_running`` 只看自己；而真正的
    无人值守闭环通常跑在独立的 ``run_loop.py`` 常驻进程里。两者抢同一个
    ``data/.run_loop.lock``，所以看板点「启动闭环」会被拒（"已有另一个闭环实例在运行"）。

    这里读心跳文件判断常驻进程是否活着，让看板如实显示状态，而不是给一个永远点不动的
    「启动闭环」按钮。返回 dict：``{running, pid, age_sec, state}``。
    """
    hb = read_heartbeat()
    if not hb:
        return {"running": False, "pid": None, "age_sec": None, "state": ""}
    age = hb.get("age_sec")
    if age is not None and age <= stale_sec:
        return {"running": True, "pid": hb.get("pid"),
                "age_sec": age, "state": hb.get("state") or ""}
    return {"running": False, "pid": hb.get("pid"),
            "age_sec": age, "state": hb.get("state") or ""}


if __name__ == "__main__":  # 手动跑一个周期
    from src.trading.yuekai import YuekaiBroker
    b = YuekaiBroker()
    b.connect(mode="sim")
    lp = TradingLoop(b, config=LoopConfig(dry_run=True))
    rep = lp.run_once()
    print(rep.summary())
    print(json.dumps(lp.preview(), ensure_ascii=False, indent=2, default=str)[:2000])

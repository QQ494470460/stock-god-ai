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

import hashlib
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

from config.settings import DATA_DIR, ExecutionConfig, config as app_config
from src.trading.broker import BrokerAPI, OrderSide, Position, AccountInfo, Order
from src.trading.guard import RiskSupervisor, build_supervisor, TradingMode
from src.trading.sanity import QuoteSanityGuard, wrap_broker


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


#: 退出原因 → 可归因的策略标签。
#:
#: 为什么要分细：过去所有风控离场都记成同一个 ``stop``，于是
#: **「止盈从来没触发过」这个决定性事实在归因表里完全看不见** ——
#: 你只看得到「stop 腿 9 笔全亏 -23 万」，却不知道配的止盈线是 +15%、
#: 而实际最大盈利只有 +1.62%（全部由调仓卖出）。三者是完全不同的失败模式：
#:   · stop_profit 不触发 → 止盈线过远 / 盈利仓被调仓提前换出
#:   · stop_loss 频繁触发 → 入场质量问题
#:   · trail_stop 频繁触发 → 移动止损过紧，把趋势单砍在半路
#: 混在一个桶里就没法判断到底该修入场还是修退出。
def _exit_tag(reason: str) -> str:
    r = str(reason or "")
    # 「移动止损」里含「止损」，必须先判。
    if "移动止损" in r:
        return "trail_stop"
    tp, sl = ("止盈" in r), ("止损" in r)
    # 两者都出现（如兜底文案「止盈止损」）= 说不清，记 exit 而不是瞎猜 ——
    # 猜错的标签会污染归因表，比缺失更难发现。
    if tp and not sl:
        return "stop_profit"
    if sl and not tp:
        return "stop_loss"
    return "exit"


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
    # 生产权威值在 config/settings.py:TradingLoopConfig（此处仅 app_config 缺失时兜底，
    # 两处须一致，否则「改哪份生效」全看运气）。
    max_positions: int = 8               # 目标持仓上限（与 guard 对齐）
    cash_reserve_pct: float = 0.10       # 现金保留比例
    weight_cap: float = 0.30             # 单票权重上限（= guard.max_single_position_pct）
    # equal / score / risk_parity / min_variance / vol_target
    # 默认风险平价：等权与按评分加权都只看 alpha，不看波动与相关性，会让
    # 组合的风险集中在少数高波动标的上；风险平价让每只票贡献相同风险，
    # 与 guard 的 VaR 预算同源。数据不足时 build_trade_plan 自动退回等权。
    weighting: str = "risk_parity"
    tilt_strength: float = 0.0           # 风险权重上叠加 alpha 倾斜的力度（0=纯风险平价；真OOS验证 0.0 全面更优）
    enable_rebalance: bool = True        # 是否执行目标调仓
    enable_stop_orders: bool = True      # 是否执行止盈止损
    # 组合回撤 → 连续渐进降仓（**不是**清仓）。
    # ⚠️ 2026-10-31 关闭（原 True + floor 0.30）。执行级回测（1140 天 × 200 只，
    # 10bp，15 日周期 + slack3）把这道闸门算进去之后：
    #   无闸门       年化 +26.8% / 夏普 1.19 / 回撤 -25.0% / 年换手 10.6x
    #   渐进降仓0.30 年化 +14.7% / 夏普 0.99 / 回撤 -15.0% / 年换手  4.5x
    #   + 8% 熔断    年化  +6.5% / 夏普 0.42 / 回撤 -16.3% / 年换手  1.8x  ← 实盘现状
    # 即：**这两层「越跌越轻」的顺周期机制吃掉了 3/4 的收益**，而回撤只从 -25.0%
    # 改善到 -16.3%。代价与收益完全不成比例。而且实盘只在 8% 熔断线下工作，
    # 回测的最大回撤是 -25% ⇒ 正常历史里必然反复触发，等于换了一个策略。
    # 实测证据：10-27 回撤 7.90% 时日志写着「敞口系数 0.31」—— 新仓直接砍到三成。
    # 现在风险边界交给「15 日周期 + 8 只分散 + 8% 熔断」，不再额外压新仓规模。
    enable_risk_scaling: bool = False
    risk_scale_floor: float = 0.30       # 关闭时无效；保留字段供快速回滚/实验
    # ---- T3 组合层风控覆盖层（**默认关闭**）。生产权威值在
    # config/settings.py:TradingLoopConfig 的 derisk_* / vol_* / risk_blend。
    # ⚠️ 已实测在生产口径（hold_rank_slack=3.0）下为**负贡献**，故关闭：
    #    它与「排名滞后 3.0」是替代关系（都在降换手/降回撤），叠加重复扣分。
    #    详见 settings.py 里那张 slack=3.0 的对照表。
    derisk_mode: str = "off"             # off / asymmetric / procyclical
    derisk_engage_pct: float = 12.0      # ⚠️ 启用时不得 < 10%，否则退回顺周期
    derisk_max_dd_pct: float = 20.0
    derisk_floor: float = 0.60
    derisk_release_hysteresis_pct: float = 2.0
    vol_target_pct: float = 0.0          # 0=关闭；百分数
    vol_scale_floor: float = 0.5
    vol_window_days: int = 20
    vol_min_obs: int = 10
    risk_blend: str = "mul"              # mul（回测口径）/ avg（未验证）
    # 调仓容忍带。生产权威值在 config/settings.py:TradingLoopConfig.band_pct
    # （此处只是 app_config 缺失时的兜底，两处须保持一致，否则谁读谁生效全看运气）。
    band_pct: float = 0.15
    # 调仓周期（交易日，0/1=每 tick 都可调）。生产权威值在
    # config/settings.py:TradingLoopConfig.rebalance_days（此处仅兜底）。
    # 依据见 settings.py 里的注解：5 天周期在 50bp 成本下夏普塌到 0.41，
    # 15 天周期在三档成本下 1.19/1.06/0.95 —— 周期是成本敏感性的决定性参数。
    rebalance_days: int = 15
    # 排名滞后系数（1.0=关闭）。持仓只要还在「前 top_n×slack 名」以内就保留。
    # 生产权威值在 config/settings.py:TradingLoopConfig.hold_rank_slack。
    # 依据：执行级回测 15 天周期下 slack3.0 vs 1.0 在 50bp 成本下
    # 年化 21.1% vs 14.7%、夏普 0.95 vs 0.68（换手 9.3x vs 14.3x）。
    hold_rank_slack: float = 3.0
    min_trade_amount: float = 5000.0     # 最小交易金额（元）
    # 持有纪律阈值（0=关闭）。持仓不在目标组合但浮盈≥该比例时不强制清仓，
    # 改由止损/止盈托管。实测「每日换票必卖赢家」是盈亏比 0.15 的根因之一。
    # 调高前须经 walk-forward 验证，否则退回「卖赢家、留输家」。
    hold_if_profitable_pct: float = 0.0
    # 单 tick 最多下多少单（防雪崩闸门）。
    # ⚠️ 必须 ≥ 2×max_positions：一次完整调仓是「卖 ≤N + 买 ≤N」共 2N 笔，
    # N=8 时最多 16 笔。原来 10 会在满仓换手时把排在后面的买单截掉，
    # 而周期标记已写死 ⇒ 剩下的票要等一整个周期（15 个交易日），账户长期半仓。
    max_orders_per_tick: int = 30
    # 调仓日「未建满」顺延重试（**2026-11-06 经用户确认打开**，默认关闭的同名
    # 字段在 config/settings.py:TradingLoopConfig 是生产权威值）。要点：一天最多
    # 重试一次、整轮最多顺延 `rebalance_retry_max_days` 个交易日，用尽即收尾，
    # 绝不退化成逐 tick 重试。
    rebalance_retry_enabled: bool = True
    rebalance_retry_max_days: int = 2
    rebalance_retry_min_fill_ratio: float = 0.85
    settle_hour: int = 15                # 收盘结算时刻
    settle_minute: int = 5
    notify: bool = True                  # 是否推送事件/日报
    dry_run: bool = False                # True=只预演不下单
    price_fallback: bool = True          # 无 tick 时回退日线收盘价
    heartbeat_file: str = ""             # tick 心跳落盘路径（空=data/loop_heartbeat.json）
    # 执行层智能下单（T2）：默认 None（关闭）；启用后由 _build_exec_scheduler 接管调仓分片。
    execution: "ExecutionConfig" = None

    @classmethod
    def from_app_config(cls) -> "LoopConfig":
        """从全局配置的 trading_loop 段构造（缺失则用安全默认）。

        ⚠️ 嵌套配置必须还原成对象：``asdict`` 会把 ``execution`` 深度转成
        ``dict``，而 ``_build_exec_scheduler`` 是用 ``getattr(ec, "smart_routing")``
        读的 —— 拿到 dict 会恒为 False，等于「配置里开了、进程里没开」。
        这类静默失效正是本项目最大的坑，故在此显式重建。
        """
        try:
            sec = getattr(app_config, "trading_loop", None)
            if sec is None:
                return cls()
            kwargs = {k: v for k, v in asdict(sec).items()
                      if k in cls.__dataclass_fields__}
            ex = kwargs.get("execution")
            if isinstance(ex, dict):
                from dataclasses import fields as _dc_fields
                allowed = {f.name for f in _dc_fields(ExecutionConfig)}
                kwargs["execution"] = ExecutionConfig(
                    **{k: v for k, v in ex.items() if k in allowed})
            return cls(**kwargs)
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
        # 单实例锁：传了 lock_path 就表示「本实例启动时必须先抢到这把锁」。
        # 看板内嵌闭环与常驻 run_loop.py 共用 data/.run_loop.lock，
        # 抢不到 = 已经有一个实例在跑 → 拒绝启动，避免重复下单。
        self._instance_lock = SingleInstanceLock(lock_path) if lock_path else None
        self.config = config or LoopConfig.from_app_config()
        # 执行层智能下单（T2）：opt-in，默认 None（行为不变）；启用后接管调仓分片。
        self._exec_scheduler = self._build_exec_scheduler()
        # 组合层风控覆盖层（T3）：带恢复滞回的非顺周期降仓斜坡，懒建；默认不建。
        self._derisk_ramp = None
        # 一启动就把覆盖层口径打进日志 —— 否则「到底有没有生效」只能靠猜，
        # 而这类静默失效（配置改了、进程没重读）本项目已经踩过多次。
        _dm = str(getattr(self.config, "derisk_mode", "off") or "off")
        _vt = float(getattr(self.config, "vol_target_pct", 0.0) or 0.0)
        if _dm != "off" or _vt > 0:
            logger.info(
                f"🛡 组合层风控覆盖层: 回撤腿={_dm}"
                f"({getattr(self.config, 'derisk_engage_pct', 0):.0f}%/"
                f"{getattr(self.config, 'derisk_max_dd_pct', 0):.0f}%/"
                f"floor {getattr(self.config, 'derisk_floor', 0):.2f})"
                f" 波动腿={'off' if _vt <= 0 else f'{_vt:.0f}%'}"
                f"(floor {getattr(self.config, 'vol_scale_floor', 0):.2f})"
                f" 合成={getattr(self.config, 'risk_blend', 'mul')}"
            )
        else:
            logger.info("🛡 组合层风控覆盖层: 关闭（回撤腿/波动腿均未启用）")

        # 报价合法性守卫：一个坏价格就是一笔坏交易。
        # 数据源会偶发返回昨收/复权价/错位价，而止损是按价格判定的 ——
        # 不校验就会在错误的价格上斩仓，或让权益曲线凭空断崖。
        # 连续 confirm 次同向偏离才采信（区分「数据抖动」与「真实涨跌停」）。
        #
        # ⚠️ 顺序有讲究：守卫必须先建，再拿去包券商，最后回灌给 guard。
        # 三处取价链路（闭环估值 / propose_trade 预览 / executor 止损）必须
        # 共享**同一个**实例 —— 「上一次可信价」是跨 tick 状态，各记一份就会
        # 互相打架：这边判异常沿用旧价，那边没见过这个价反手就采信。
        self._sanity = QuoteSanityGuard(confirm_ticks=2, tolerance_pct=2.0)
        self._sanity_warned: set = set()   # (code, why) 去重，避免每 tick 刷日志
        self.broker = wrap_broker(
            broker, guard=self._sanity, on_reject=self._warn_bad_quote)

        # 安全中枢：未提供则自动构建（默认 PAPER，绝不裸奔）
        self.guard = guard or build_supervisor(self.broker, paper_account=paper_account)
        # 外部传入的 guard（run_loop.py / 看板）握的是**原始** broker，
        # 取价会绕过代理。整体切到代理上，否则刚才那层包装等于白套。
        try:
            self.guard.rebind_broker(self.broker)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"风控中枢切到报价代理失败（取价将绕过校验）: {e}")
        else:
            # 启动即留痕：出问题时第一件事就是确认「校验到底有没有接管」，
            # 靠推理不如靠一行日志。
            logger.info(
                f"🔍 报价校验已接管取价通道（券商代理统一收口，"
                f"连续 {self._sanity.confirm_ticks} 次同向偏离才采信，"
                f"容差 ±{self._sanity.tolerance_pct:g}%）")
        if paper_account is not None and getattr(self.guard, "paper_account", None) is None:
            self.guard.paper_account = paper_account

        self.notifier = notifier
        self._price_fn = price_fn
        self._on_event = event_callback

        # 计划与状态
        self.plan = None
        # 已经喊过「这批票是旧的」的计划日期。告警只在计划**换了一份**时喊一次：
        # _ensure_plan 每 tick 都跑，不设这个就会把日志刷成一片。
        self._stale_picks_warned: set = set()
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
        # 上次**实际发起调仓**的交易日（YYYY-MM-DD）。配合 config.rebalance_days
        # 控制调仓节奏：到期才动，其余时间只跑止损/结算。
        # 必须落盘：闭环本来就设计成会自愈重启，纯内存状态一重启就「立刻再调一次仓」，
        # 换手与成本会翻倍 —— 而换手正是 50bp 成本下的死因。
        self._last_rebalance_date: str = ""
        # 顺延重试的两个状态（见 config.settings.rebalance_retry_*）：
        # `_rebalance_defer_from` = 本轮周期**第一次尝试**的交易日（顺延锚点，
        #   收尾时用它回填周期标记，从而不把 15 天的节奏往后推）；
        # `_rebalance_defer_date` = 最近一次判「未建满」的日期（实现「一天只重试一次」）。
        # 必须落盘：与 `_last_rebalance_date` 同理会自愈重启。
        self._rebalance_defer_from: str = ""
        self._rebalance_defer_date: str = ""
        self._load_runtime()
        # 纸账户本金基线（initial_cash）的上一 tick 记忆：用于检测「重置」这类
        # 非交易性基线变更并自动对齐风控净值基线，避免误熔断。
        self._last_pa_baseline: Optional[float] = None
        # 上次见到的「重置证据」时间戳（PaperAccount.reset_at）。配合 _last_pa_baseline
        # 判断 initial_cash 变化是人工重置还是状态被写坏，见 _tick_impl 里的对齐分支。
        self._last_pa_reset_at: str = ""

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
            self._last_rebalance_date = str(data.get("last_rebalance_date") or "")
            self._rebalance_defer_from = str(data.get("rebalance_defer_from") or "")
            self._rebalance_defer_date = str(data.get("rebalance_defer_date") or "")
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
                    "last_rebalance_date": self._last_rebalance_date,
                    "rebalance_defer_from": self._rebalance_defer_from,
                    "rebalance_defer_date": self._rebalance_defer_date,
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

    def _warn_bad_quote(self, code: str, raw: float, v) -> None:
        """报价被代理拦下时的告警（按 code + 原因去重，避免每 tick 刷屏）。

        双写：``logger.warning`` 给人看实时日志，``quote_reject`` 进交易流水
        给事后复盘 —— 「今天数据源骗了几次、骗在哪几只票上」必须能查。
        ⚠️ 闭环走 ``self._emit`` 是**空转**（常驻 run_loop.py 没注册 UI 回调），
        所以这里改走风控中枢的 ``_emit``，它同时写 journal 与 UI 回调。
        """
        try:
            key = (str(code or ""), str(v.why or "").split("（")[0])
            if key in self._sanity_warned:
                return
            self._sanity_warned.add(key)
            payload = {
                "code": code,
                "raw": round(float(raw or 0), 4),
                "used": round(float(v.price or 0), 4),
                "why": v.why,
            }
            logger.warning(
                f"⚠️ 报价异常已拦截 {code}: 原始价 {raw:.3f} → {v.why}；"
                f"本次沿用 ¥{float(v.price or 0):.3f}")
            g = getattr(self, "guard", None)
            if g is not None:
                g._emit("quote_reject", payload)
            else:
                self._emit("quote_reject", payload)
        except Exception:  # noqa: BLE001
            pass

    def _price_of(self, code: str) -> float:
        if code in self._price_cache:
            return self._price_cache[code]
        price = 0.0
        pre_close = 0.0
        volume = amount = None
        from_broker = False
        # 1) 券商实时行情（已被 SanityBroker 净化过 —— 这里拿到的就是可信价）
        try:
            tick = self.broker.get_tick(code)
            if tick and getattr(tick, "last_price", 0) > 0:
                price = float(tick.last_price)
                pre_close = float(getattr(tick, "pre_close", 0) or 0)
                volume = getattr(tick, "volume", None)
                amount = getattr(tick, "amount", None)
                from_broker = True
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
                        try:
                            pre_close = float(q.get("昨收") or q.get("昨收价") or 0)
                        except (TypeError, ValueError):
                            pre_close = 0.0
                elif q is not None:
                    p = float(getattr(q, "last_price", 0) or getattr(q, "price", 0) or 0)
                    if p > 0:
                        price = p
                        pre_close = float(getattr(q, "pre_close", 0) or 0)
            except Exception:  # noqa: BLE001
                pass
        # 3) 兜底：日线收盘价（离线 / 盘后可用；盘中会退化为昨收，仅最后手段）
        if price <= 0 and self.config.price_fallback and self._price_fn is None:
            try:
                from src.data.fetcher import fetcher
                df = fetcher.get_daily_kline(code)
                if df is not None and len(df) > 0:
                    price = float(df["close"].iloc[-1])
                    # 日线兜底拿到的就是「收盘价」，没有当日盘中信息 →
                    # 不提供 pre_close，让守卫退化为「相对上次可信价的跳变检查」
            except Exception:  # noqa: BLE001
                pass
        # 4) 外部注入的价格函数
        if price <= 0 and self._price_fn is not None:
            try:
                price = float(self._price_fn(code) or 0)
            except Exception:  # noqa: BLE001
                price = 0.0

        # 5) 合法性校验：可疑价不采信，沿用上一次可信价（好过用 0 让止损失明）
        #    ⚠️ 券商那条路已在 SanityBroker 里校验过（from_broker），这里不能再检
        #    一次 —— 同一个价被 check 两遍会让「连续 N 次才采信」变成 N/2 次，
        #    相当于偷偷把 confirm_ticks=2 降成 1，一次抖动就能改写风控判断。
        #    公开行情 / 日线 / 外部价格函数这三条兜底路径绕过代理，必须补检。
        if price > 0 and not from_broker:
            v = self._sanity.check(code, price, pre_close=pre_close,
                                   volume=volume, amount=amount)
            if not v.ok:
                self._warn_bad_quote(code, price, v)
            if v.price > 0:
                price = v.price

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
    def _build_plan(self, target_date: str | None = None):
        """生成交易计划。

        ``target_date`` 为 None 时走默认语义（= 今天之后的**下一个交易日**，
        由 ``resolve_target_date`` 决定）；显式传入则强制指定 ``plan_date``。
        后者是「补建今天的计划」所必需的 —— 默认语义**永远补不出今天**。
        """
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
                tilt_strength=self.config.tilt_strength,
                plan_date=target_date,
                write=True,
            )
            return plan
        except Exception as e:  # noqa: BLE001
            logger.error(f"交易计划生成失败: {e}")
            return None

    def _warn_if_picks_stale(self) -> None:
        """计划吃的选股结果太旧就喊一声（同一份计划只喊一次）。

        ``build_trade_plan`` 在**生成时**已经喊过，但从磁盘读回来的旧计划没人管 ——
        而这类计划恰恰最可疑：它可能在几天前就用旧票建好了
        （2026-11-05 挖到的那批 `trade_plan_20261106.json` 就是 11-03 的 picks
        生成的，与 11-04 那份目标逐位相同），当天加载时毫无征兆。

        滞后 0/1 是正常的（D-1 收盘选股 → D 日计划），>=2 才说明中间漏了作业。
        按 ``plan_date:lag`` 去重：`_ensure_plan` 每 tick 都跑，不去重会刷屏。
        """
        plan = self.plan
        if plan is None:
            return
        lag = int(getattr(plan, "picks_lag_days", 0) or 0)
        if lag < 2:
            return
        key = f"{plan.plan_date}:{lag}"
        if key in self._stale_picks_warned:
            return
        self._stale_picks_warned.add(key)
        logger.warning(
            f"⚠️ 计划 {plan.plan_date} 吃的是 "
            f"{getattr(plan, 'picks_date', '') or '未知日期'} 的选股结果，"
            f"滞后 {lag} 个交易日（正常应为 1）—— 这期间有选股作业没跑成。"
            "目标组合可能已不代表今天的市场。"
        )

    def _ensure_plan(self, report: TickReport):
        """确保手上有一份**属于今天**的交易计划（实现见 ``_ensure_plan_impl``）。

        这层薄包装只为了一件事：不管内部从哪条分支提前 return，
        都要对**最终选定的那份计划**做一次陈旧度检查。
        """
        self._ensure_plan_impl(report)
        self._warn_if_picks_stale()

    def _ensure_plan_impl(self, report: TickReport):
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

        # ② 手上这份是「下一个交易日」的待生效计划，而磁盘上**没有今天的**
        #    → 说明昨天收盘后那次作业没跑成功（正常情况这里应该已在 ① 命中）。
        #    以前这里直接 return，代价是**整个交易日空转**：
        #    `_build_plan()` 默认只生成「下一个交易日」，永远补不出「今天的」，
        #    于是闭环抱着明天那份计划，今天一单不发，界面还显示得好好的。
        #    实跑证据（2026-11-05 01:04）：调度器与 loop 同进程，11-04 晚 loop
        #    崩溃重启 → 18:30 数据更新 / 18:45 选股 / 19:00 进化三个作业全没触发
        #    → trade_plan_20261105.json 从未生成。loop 重启后 _build_plan() 按
        #    resolve_target_date(11-05)=11-06 生成了「11-06 的计划」，且用的是
        #    11-03 的旧 picks（目标与 11-04 那份逐位相同）—— 今天(11-05)无计划。
        #    现在改为补建**今天**那一份；建成后 is_valid_for(今天) 成立，
        #    下一 tick 走不到这里，不会重复写盘。
        if self.plan is not None and (self.plan.plan_date or "") > today_s:
            logger.warning(
                f"⚠️ 今天({today_s})没有交易计划，手上是 {self.plan.plan_date} 的待生效计划 —— "
                "多半是昨日收盘后的选股作业未成功。正在补建今天的计划。")
            backfilled = self._build_plan(target_date=today_s)
            if backfilled is not None:
                self.plan = backfilled
                report.plan_built = True
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
            reason = a.get("reason", "止盈止损")
            ok, msg, order, detail = self._place(
                code=code, name=a.get("name", code), side=OrderSide.SELL,
                quantity=qty, price=price, reason=reason,
                ai_score=50.0, risk_exit=True, tag=_exit_tag(reason),
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
    def _strategy_fingerprint(self) -> str:
        """当前策略形状的短指纹（决定收益分布的那些参数）。

        为什么要有它：回撤熔断按「峰值→当前权益」算，而峰值是**旧策略**
        打出来的成绩。策略形状一变（周期、权重、退出规则），旧峰值就不再
        是同一个策略的基准，继续拿它考核会直接把闭环锁死。
        这里只放「改变收益分布」的参数，**不放风控阈值**（熔断线本身变了
        不该触发基线重置，否则调风控就等于自己给自己发免死金牌）。
        """
        c = self.config
        body = {
            "top_n": int(getattr(c, "max_positions", 0) or 0),
            "band": round(float(getattr(c, "band_pct", 0.0) or 0.0), 4),
            "rebal": int(getattr(c, "rebalance_days", 0) or 0),
            "weight": str(getattr(c, "weighting", "") or ""),
            "cash": round(float(getattr(c, "cash_reserve_pct", 0.0) or 0.0), 4),
            "tilt": round(float(getattr(c, "tilt_strength", 0.0) or 0.0), 4),
            "hold_if_pf": round(
                float(getattr(c, "hold_if_profitable_pct", 0.0) or 0.0), 4),
        }
        g = getattr(self.guard, "config", None)
        if g is not None:
            body.update({
                "sl": round(float(getattr(g, "stop_loss_pct", 0.0) or 0.0), 4),
                "tp": round(float(getattr(g, "stop_profit_pct", 0.0) or 0.0), 4),
                "trail": round(
                    float(getattr(g, "trailing_stop_pct", 0.0) or 0.0), 4),
                "atr": bool(getattr(g, "atr_stop_enabled", False)),
            })
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _maybe_sync_strategy_version(self, equity: float) -> None:
        """策略形状变更时对齐熔断净值基线（细则见 guard.sync_strategy_version）。"""
        safety = getattr(self.guard, "safety", None) if self.guard else None
        if safety is None or not hasattr(safety, "sync_strategy_version"):
            return
        try:
            version = self._strategy_fingerprint()
            changed, why = safety.sync_strategy_version(version, equity)
            if changed:
                logger.warning(
                    f"🧬 策略形状变更（{why}）⇒ 熔断净值基线已对齐当前权益 "
                    f"¥{equity:,.0f}，回撤归零。旧峰值属另一策略，不再考核新策略。")
        except Exception as e:  # noqa: BLE001
            # 这条链路只影响「熔断基线是否跟着策略走」，失败绝不能挡住交易主流程
            logger.error(f"策略指纹同步失败（已忽略，下个 tick 重试）: {e}")

    def _rebalance_due(self, now: Optional[datetime] = None) -> bool:
        """距上次调仓是否已满 ``config.rebalance_days`` 个交易日。

        为什么按**交易日**而不是自然日：15 个自然日含两个周末，实际只过了 11 个
        交易日，等于把周期悄悄缩短 27%。也为什么必须落盘：自愈重启后内存清零会
        立刻再调一次仓，换手与成本翻倍 —— 而换手就是 50bp 假设下的死因。
        """
        n = int(getattr(self.config, "rebalance_days", 0) or 0)
        if n <= 1:
            return True
        last = (self._last_rebalance_date or "").strip()
        if not last:
            return True
        try:
            from datetime import date as _date

            from src.trading.calendar import trading_days_between

            d0 = _date.fromisoformat(last)
            d1 = (now or datetime.now()).date()
            return trading_days_between(d0, d1) >= n
        except Exception as e:  # noqa: BLE001
            # 数据不全时宁可允许调仓（机会成本 < 永久卡死的风险）
            logger.debug(f"调仓周期判定失败，按「到期」处理: {e}")
            return True

    # ------------------------------------------------------------------
    # 调仓日「未建满」顺延重试（opt-in，默认关闭）
    # ------------------------------------------------------------------
    def _retry_enabled(self) -> bool:
        return bool(getattr(self.config, "rebalance_retry_enabled", False))

    def _retry_max_days(self) -> int:
        try:
            return max(int(getattr(self.config, "rebalance_retry_max_days", 0) or 0), 0)
        except (TypeError, ValueError):
            return 0

    def _retry_min_ratio(self) -> float:
        try:
            r = float(getattr(self.config, "rebalance_retry_min_fill_ratio", 1.0) or 1.0)
        except (TypeError, ValueError):
            return 1.0
        # 越界一律按「不重试」处理：>1 会让每次调仓都顺延，<=0 等于永远满意。
        return r if 0.0 < r <= 1.0 else 1.0

    def _defer_days_used(self, today: str) -> int:
        """从顺延锚点算起，已经过去几个**交易日**（0 = 今天就是首次尝试日）。"""
        anchor = (self._rebalance_defer_from or "").strip()
        if not anchor:
            return 0
        try:
            from datetime import date as _date

            from src.trading.calendar import trading_days_between

            return int(trading_days_between(_date.fromisoformat(anchor),
                                            _date.fromisoformat(today)))
        except Exception as e:  # noqa: BLE001
            # 日历不可用时**不重试**（宁可维持旧行为，也不要在算不清预算时反复下单）
            logger.debug(f"顺延天数计算失败，本轮不顺延: {e}")
            return 10 ** 6

    def _clear_defer(self) -> None:
        self._rebalance_defer_from = ""
        self._rebalance_defer_date = ""

    def _close_cycle(self, today: str) -> None:
        """收尾本轮调仓周期：周期标记回填到**锚点**（第一次尝试日）。

        为什么不是 `today`：回测里调仓是严格的「每 N 个交易日一次」，顺延只是
        把**执行**补上，不该把**节奏**往后推 —— 否则每顺延一次，下一次调仓就
        整体平移，几次之后与回测口径差出一个周期。
        """
        anchor = (self._rebalance_defer_from or "").strip()
        if anchor:
            try:
                from datetime import date as _date

                _date.fromisoformat(anchor)
            except Exception:  # noqa: BLE001
                # 锚点坏掉时**绝不能**把它写进周期标记：`_rebalance_due` 解析失败
                # 会 fail-open 返回「到期」⇒ 每个 tick 都调仓，比不顺延糟得多。
                logger.warning(f"顺延锚点 {anchor!r} 不是合法日期，周期标记回退到今天")
                anchor = ""
        self._last_rebalance_date = anchor or today
        self._clear_defer()
        self._save_runtime()

    def _build_exec_scheduler(self):
        """构建执行调度器（T2）。仅当 execution.smart_routing 开启时返回实例，
        否则返回 None（闭环走原「单 tick 一次性下单」路径，行为完全不变）。
        """
        ec = getattr(self.config, "execution", None)
        if ec is None or not getattr(ec, "smart_routing", False):
            return None
        try:
            import os as _os
            from src.trading.execution.scheduler import ExecutionScheduler
            from src.trading.execution.slippage import (
                SimSlippageModel, set_active_slippage_model,
            )
            # 增强滑点模型（opt-in，仅 participation 模式注册）
            if getattr(ec, "slippage_model", "simple") == "participation":
                set_active_slippage_model(SimSlippageModel(
                    base_bps=2.0,
                    participation=float(getattr(ec, "slippage_participation", 0.01)),
                    partial_fill=bool(getattr(ec, "partial_fill_sim", False)),
                ))
            persist = None
            if getattr(ec, "persist_state", True):
                data_dir = _os.path.join(
                    _os.path.dirname(_os.path.dirname(_os.path.dirname(
                        _os.path.abspath(__file__)))), "data")
                persist = _os.path.join(data_dir, "exec_scheduler_state.json")
                if _os.path.exists(persist):
                    try:
                        import json as _json
                        with open(persist, encoding="utf-8") as f:
                            return ExecutionScheduler.from_dict(
                                _json.load(f), persist_path=persist)
                    except Exception:
                        pass
            return ExecutionScheduler(
                strategy=getattr(ec, "slice_strategy", "twap"),
                window_min=int(getattr(ec, "window_min", 300)),
                max_slices=int(getattr(ec, "max_slices", 10)),
                overnight=bool(getattr(ec, "overnight_preopen", True)),
                retry_cfg={
                    "max_attempts": int(getattr(ec, "retry_max_attempts", 3)),
                    "step_bps": float(getattr(ec, "retry_step_bps", 8.0)),
                    "timeout_sec": float(getattr(ec, "retry_timeout_sec", 60.0)),
                    "escalate_to_market": bool(getattr(ec, "escalate_to_market", True)),
                },
                persist_path=persist,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"执行调度器构建失败（回退为一次性下单）: {e}")
            return None

    def _run_rebalance(self, positions, account, prices, report: TickReport):
        if self.plan is None or not self.plan.targets:
            report.note = "无有效交易计划，跳过调仓"
            return
        if not self._rebalance_due():
            report.note = (f"距上次调仓不足 {getattr(self.config, 'rebalance_days', 0)} "
                           f"个交易日（上次 {self._last_rebalance_date}），本次只跑止损/结算")
            return
        _retry = self._retry_enabled()
        _today = datetime.now().strftime("%Y-%m-%d")
        if _retry:
            # ① 顺延预算用尽 ⇒ 就地收尾。放在下单之前：结构性拦截（集中度/资金）
            #    不会因为再试一次而消失，继续下单只是把同样的日志再刷一遍。
            _used = self._defer_days_used(_today)
            if _used > self._retry_max_days():
                logger.warning(
                    f"⚠️ 调仓顺延预算用尽（锚点 {self._rebalance_defer_from}，"
                    f"已顺延 {_used} 个交易日 > 上限 {self._retry_max_days()}）⇒ "
                    f"本轮收尾，周期标记回填到锚点。未建满的部分要等下一个周期，"
                    f"请查 data/trading_journal.jsonl 里被拦的原因。")
                self._close_cycle(_today)
                report.note = "调仓顺延预算用尽，本轮收尾"
                return
            # ② 一天只重试一次。放开的后果：结构性拦截会让每个 tick 都重下一遍，
            #    换手被悄悄改成「逐 tick」，15 天周期的成本优势直接作废。
            if self._rebalance_defer_date == _today:
                report.note = (
                    f"本轮调仓已顺延（锚点 {self._rebalance_defer_from}，"
                    f"已用 {_used}/{self._retry_max_days()} 个交易日），"
                    f"下个交易日再试")
                return
        from src.trading.rebalance import compute_rebalance_orders

        risk_scale = self._portfolio_risk_scale(account)
        intents = compute_rebalance_orders(
            self.plan, positions, account, prices,
            band_pct=self.config.band_pct,
            min_trade_amount=self.config.min_trade_amount,
            allow_buy=True, allow_sell=True,
            risk_scale=risk_scale,
            adv_provider=self._adv_provider(),
            participation=self._participation(),
            hold_if_profitable_pct=self.config.hold_if_profitable_pct,
            hold_rank_slack=float(
                getattr(self.config, "hold_rank_slack", 1.0) or 1.0),
        )
        report.n_intents = len(intents)
        # T2 智能下单（opt-in）：切片在时间维度摊薄，天然规避单 tick 雪崩，
        # 因此不受 max_orders_per_tick 截断；入队后即记周期标记，避免重复入队。
        if self._exec_scheduler is not None and intents:
            n_q = self._exec_scheduler.enqueue(intents, datetime.now().date())
            # ⚠️ T2 执行调度接管后，下面的「周期记账 + 顺延重试」逻辑整体跳过（return）：
            # 调度器在盘中按切片逐 tick 发出，日内填不满由调度器自身重发处理，与「按周期
            # 驱动的顺延重试」严格互斥，两套机制叠加会重复推动同一批委托。
            # 但若入队本身被截断（n_q < 待入队数），与上方「闸门截断」同义：
            # **不写周期标记**，下个 tick 继续补入队，否则账户长期半仓、现金闲置。
            if n_q >= len(intents):
                # 与下面同口径：顺延期间记账用锚点（不推节奏）
                self._last_rebalance_date = (
                    (self._rebalance_defer_from or _today) if _retry else _today)
                self._save_runtime()
            else:
                logger.warning(
                    f"⚠️ T2 入队被截断：{n_q}/{len(intents)} 笔入队，"
                    f"不记周期标记，下个 tick 继续补入队（记了会锁账户）")
            report.note = (
                f"⚙️ 执行调度已接管 {n_q}/{len(intents)} 笔调仓意图，"
                f"将于盘中按 {getattr(self.config.execution, 'slice_strategy', 'twap')} 分片发出")
            return
        # ⚠️ 「防雪崩闸门」与「周期记账」的先后顺序很容易搞错，这里钉死：
        # 一次完整调仓需要「卖 ≤N + 买 ≤N」共 2N 笔（N = max_positions）。
        # N=8 时最多 16 笔，若闸门 < 16，排在后面（**全是买单**）的委托会被
        # 静默丢弃；而周期标记若已经写死，就变成「8 只只建了 2 只，剩下 6 只
        # 等 15 个交易日」—— 账户长期半仓、大量现金闲置，收益自然上不去。
        # 两道保险：① 闸门默认值已提到 30（≥ 2×8 + 余量）；
        #           ② 万一仍被截断，**不写周期标记**，下个 tick 接着做。
        _limit = int(getattr(self.config, "max_orders_per_tick", 0) or 0)
        _truncated = bool(_limit > 0 and len(intents) > _limit)
        # 周期标记先写再下单（崩溃安全：宁可「记成已调过」也不要重复下单）。
        # 值取「顺延锚点」而非今天：顺延只补执行，不推节奏（见 _close_cycle）。
        _mark_value = ((self._rebalance_defer_from or _today) if _retry else _today)
        if not _truncated:
            # 空转（带内/资金不足）也要记账：调仓是按**周期**驱动的，不是
            # 「想调就调」。只在有成交时记账，会让空转的调仓日每 tick 重算，
            # 直到某天碰巧成交 —— 等于把周期悄悄改回逐 tick。
            self._last_rebalance_date = _mark_value
            self._save_runtime()
        else:
            logger.warning(
                f"⚠️ 本轮调仓委托 {len(intents)} 笔超过单 tick 上限 {_limit} 笔，"
                f"只执行前 {_limit} 笔；**不记周期标记**，下个 tick 继续"
                f"（记了的话剩下的委托要等 "
                f"{getattr(self.config, 'rebalance_days', '?')} 个交易日）")
        if not intents:
            if _retry and (self._rebalance_defer_from or self._rebalance_defer_date):
                # 顺延后的重试日跑出「零意图」= 已经对齐目标，顺延结束。
                self._clear_defer()
                self._save_runtime()
            return

        score_map = {t.code: t.score for t in self.plan.targets}
        _planned = 0.0
        _filled = 0.0
        for it in (intents[:_limit] if _limit > 0 else intents):
            _amt = abs(float(it.quantity or 0) * float(it.price or 0))
            _planned += _amt
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
                # 用「实际成交量×成交价」而不是意愿量：guard 可能下调数量
                _dq = float((detail or {}).get("quantity") or 0)
                _dp = float((detail or {}).get("price") or 0)
                _filled += abs(_dq * _dp) if (_dq > 0 and _dp > 0) else _amt
            else:
                report.n_blocked += 1
        # 顺延判定：只看「本轮委托有多少真成交了」。被截断的那条路径已经用
        # 「不写周期标记」实现了 tick 级续做，两条机制不叠加（否则同一批委托
        # 会被两套重试逻辑各推一次）。
        if _retry and not _truncated:
            _ratio = (_filled / _planned) if _planned > 0 else 1.0
            if _ratio < self._retry_min_ratio():
                anchor = self._rebalance_defer_from or _today
                self._rebalance_defer_from = anchor
                self._rebalance_defer_date = _today
                self._last_rebalance_date = ""      # 撤回：本轮不算「调过」
                self._save_runtime()
                logger.warning(
                    f"⚠️ 本轮调仓未建满：成交 ¥{_filled:,.0f} / 计划 ¥{_planned:,.0f}"
                    f"（{_ratio:.0%} < {self._retry_min_ratio():.0%}），"
                    f"{report.n_blocked} 笔被拦 ⇒ **不记周期标记**，顺延到下一个交易日"
                    f"重试（锚点 {anchor}，已用 {self._defer_days_used(_today)}"
                    f"/{self._retry_max_days()} 个交易日）。"
                    f"若拦截原因结构性不变，预算用尽后会自动收尾。")
                report.note = (
                    f"调仓未建满（{_ratio:.0%}），已顺延到下一交易日"
                    f"（{self._defer_days_used(_today)}/{self._retry_max_days()}）")
                return
            if self._rebalance_defer_from or self._rebalance_defer_date:
                self._clear_defer()
                self._save_runtime()

    def _drain_exec_scheduler(self, now, report: TickReport):
        """T2：每个交易 tick 让调度器吐出「此刻该发」的切片，并经 guard 下单。"""
        try:
            due = self._exec_scheduler.release_due(now)
            if not due:
                return
            score_map = ({t.code: t.score for t in self.plan.targets}
                         if self.plan and self.plan.targets else {})
            for ro in due:
                ok, msg, order, detail = self._place(
                    code=ro.code, name=ro.name, side=ro.side,
                    quantity=ro.qty, price=ro.price, reason=ro.reason,
                    ai_score=float(score_map.get(ro.code, 50.0)),
                    risk_exit=(ro.side == OrderSide.SELL),
                    tag="exec_slice",
                )
                report.actions.append(detail or {})
                if ok:
                    report.n_executed += 1
                    self._exec_scheduler.on_fill(ro.slice_id, ro.qty, now)
                else:
                    report.n_blocked += 1
        except Exception as e:  # noqa: BLE001
            logger.error(f"执行调度出队失败: {e}")

    # ------------------------------------------------------------------
    # 组合层风控覆盖层（T3）
    # ------------------------------------------------------------------
    def _portfolio_drawdown_pct(self, account) -> Optional[float]:
        """当前组合峰谷回撤（百分数，正数表示回撤）；取不到返回 None。

        **绝不把「取不到」当成「回撤很深」**：guard 状态文件缺失/权益为 0 时
        返回 None，调用方按「不干预」处理。反过来（缺失→满仓）与（缺失→空仓）
        两种误判里，只有前者是可接受的错误方向。
        """
        try:
            from src.trading.sizing import drawdown_pct

            peak = float(getattr(self.guard.safety._state, "peak_equity", 0) or 0)
            equity = float(getattr(account, "total_asset", 0) or 0)
            if peak <= 0 or equity <= 0:
                return None
            dd = float(drawdown_pct(equity, peak))
            if not (dd == dd) or dd < 0:   # NaN / 负值
                return None
            return dd
        except Exception as e:  # noqa: BLE001
            logger.debug(f"组合回撤计算失败，本 tick 不干预：{e}")
            return None

    def _realized_portfolio_vol(self) -> float:
        """用逐日起始权益序列估组合年化波动率；样本不足返回 0.0（fail-open）。

        数据源是 ``self._day_start_equity``（``{date: equity}``，已随
        ``data/loop_runtime.json`` 落盘），所以重启不清零、不需要新状态文件。

        ⚠️ 与回测的头寸无关，只差估计量：回测用窗口内**简单标准差**，这里用
        EWMA(span=window)（等价 ewm，近期权重更大，对波动骤升反应更快）。两者
        在同一窗口上的差别通常 < 10%，落在敞口系数上约 ±2%，不影响方向；
        若以后发现「实盘波动腿砍得比回测狠」，第一个要怀疑的就是这里，
        届时把 ``vol_estimator`` 切成简单标准差即可，不必改回测。
        """
        try:
            from src.trading.risk.volatility import ewma_vol_annualized

            series = getattr(self, "_day_start_equity", None) or {}
            window = max(int(getattr(self.config, "vol_window_days", 20) or 20), 3)
            min_obs = max(int(getattr(self.config, "vol_min_obs", 10) or 10), 2)

            pts = []
            for k, v in series.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv > 0:
                    pts.append((str(k), fv))
            pts.sort(key=lambda kv: kv[0])
            eq = [v for _, v in pts[-(window + 1):]]
            if len(eq) < min_obs + 1:
                return 0.0
            rets = [eq[i] / eq[i - 1] - 1.0 for i in range(1, len(eq)) if eq[i - 1] > 0]
            if len(rets) < min_obs:
                return 0.0
            return float(ewma_vol_annualized(rets, span=window, periods_per_year=252))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"组合波动率估计失败，波动腿不干预：{e}")
            return 0.0

    def _derisk_scale_from_dd(self, dd: Optional[float]) -> float:
        """回撤腿：非顺周期斜坡（介入线 12%，不是 0%）。"""
        if dd is None:
            return 1.0
        mode = str(getattr(self.config, "derisk_mode", "off") or "off").strip().lower()
        try:
            if mode == "asymmetric":
                if self._derisk_ramp is None:
                    from src.trading.risk.derisk import DeriskRamp
                    self._derisk_ramp = DeriskRamp(
                        engage_pct=float(getattr(self.config, "derisk_engage_pct", 12.0)),
                        max_dd_pct=float(getattr(self.config, "derisk_max_dd_pct", 20.0)),
                        floor=float(getattr(self.config, "derisk_floor", 0.60)),
                        release_hysteresis_pct=float(getattr(
                            self.config, "derisk_release_hysteresis_pct", 2.0)),
                    )
                before = self._derisk_ramp.current()
                scale = self._derisk_ramp.update(dd)
                if scale < 1.0 and scale != before:
                    logger.info(
                        f"📉 回撤 {dd:.2f}%（介入以来最深 "
                        f"{self._derisk_ramp.peak_drawdown_pct:.2f}%）→ 新买入系数 "
                        f"{scale:.2f}（只缩新仓，不强制卖出）")
                return float(scale)
            if mode == "procyclical":
                # 旧口径（已被回测证伪，仅保留供 A/B 回滚）
                from src.trading.sizing import portfolio_risk_scale
                return float(portfolio_risk_scale(
                    dd, self.guard.config.max_portfolio_drawdown_pct,
                    floor=self.config.risk_scale_floor))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"回撤腿计算失败，不干预：{e}")
        return 1.0

    def _vol_target_scale(self) -> float:
        """波动腿：实测组合波动 > 目标时按比例缩新买入（只缩不放）。"""
        target = float(getattr(self.config, "vol_target_pct", 0.0) or 0.0)
        if target <= 0.0:
            return 1.0
        rv = self._realized_portfolio_vol()
        if rv <= 0.0:
            return 1.0
        try:
            from src.trading.risk.portfolio_vol import portfolio_vol_target_scale

            floor = float(getattr(self.config, "vol_scale_floor", 0.5) or 0.5)
            scale = float(portfolio_vol_target_scale(
                rv, target / 100.0, floor=floor, ceil_=1.0))
            if scale < 1.0:
                logger.info(
                    f"🌊 实测组合波动 {rv * 100:.1f}% > 目标 {target:.1f}% → "
                    f"新买入系数 {scale:.2f}（只缩不放）")
            return scale
        except Exception as e:  # noqa: BLE001
            logger.debug(f"波动腿计算失败，不干预：{e}")
            return 1.0

    def _portfolio_risk_scale(self, account) -> float:
        """组合层敞口系数：缩放**可投敞口**（deployable = 总资产 × 系数）。

        ⚠️ 语义澄清（很重要，别被「只缩新买入」这种说法误导）：
        系数最终传给 ``rebalance`` 的 ``risk_scale``，那里算的是
        ``deployable = total_asset × risk_scale``，目标金额 = deployable × 权重。
        所以系数一旦调低，**已持仓超过「目标 × (1 + band_pct)」的部分会被减仓**。
        也就是说它缩的是「整个组合的目标敞口」，不是「这一笔买单的手数」。
        这一点与 ``exec_backtest`` 里 ``invest = eq × (1-cash) × scale`` 的口径
        完全一致 —— 回测 E 方案就是按这个语义跑出来的，实盘必须同口径，
        否则又是一处「回测里有、实盘不一样」的漂移。

        两条正交的腿，都可独立开关，默认全关：

        * **回撤腿**（``derisk_mode``）：非顺周期斜坡，回撤 < 12% 完全不干预，
          12%→20% 线性压到 0.60，且带 2% 恢复滞回（修到 10% 才完全放开）。
          与旧的 ``enable_risk_scaling`` 的唯一区别就是「介入线 0 还是 12%」——
          从 0 开始砍 = 顺周期 = 卖在低点，回测里吃掉 9pp 收益。
        * **波动腿**（``vol_target_pct``）：实测组合年化波动超目标时按
          ``target/realized`` 缩，地板 0.5，**只缩不放**（低波动不加杠杆）。

        合成方式由 ``risk_blend`` 决定，默认 ``mul``（= 回测 E 方案口径，
        最差 0.60×0.50=0.30）；``avg`` 是算术平均（地板 0.55，但**未验证**）。

        取不到峰值/权益/波动样本时**不干预**（返回 1.0）—— 数据缺失不能当成
        「回撤很深」，否则一次读盘失败就会把仓位砍半。
        """
        mode = str(getattr(self.config, "derisk_mode", "off") or "off").strip().lower()
        vol_target = float(getattr(self.config, "vol_target_pct", 0.0) or 0.0)
        legacy = bool(getattr(self.config, "enable_risk_scaling", False))

        if mode == "off" and not legacy:
            # 旧开关没开时，derisk_mode=off 就彻底不介入（不必算 dd）。
            if vol_target <= 0.0:
                return 1.0
            return self._vol_target_scale()

        dd = self._portfolio_drawdown_pct(account)
        scales = []

        if mode == "asymmetric":
            scales.append(self._derisk_scale_from_dd(dd))
        elif mode == "procyclical" or legacy:
            if dd is not None:
                try:
                    from src.trading.sizing import portfolio_risk_scale
                    s = float(portfolio_risk_scale(
                        dd, self.guard.config.max_portfolio_drawdown_pct,
                        floor=self.config.risk_scale_floor))
                    if s < 1.0:
                        logger.info(
                            f"📉 组合回撤 {dd:.2f}% → 敞口系数 {s:.2f}"
                            f"（顺周期渐进降仓）")
                    scales.append(s)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"顺周期降仓计算失败，不干预：{e}")

        if vol_target > 0.0:
            scales.append(self._vol_target_scale())

        if not scales:
            return 1.0
        if len(scales) == 1:
            return max(0.0, min(float(scales[0]), 1.0))

        blend = str(getattr(self.config, "risk_blend", "mul") or "mul").strip().lower()
        a, b = float(scales[0]), float(scales[1])
        if blend == "avg":
            try:
                from src.trading.risk.portfolio_vol import blend_scales
                return float(blend_scales(a, b, 0.5))
            except Exception:  # noqa: BLE001
                pass
        return max(0.0, min(a * b, 1.0))

    def _adv_provider(self):
        """流动性上限提供者（计划层事前封顶用）；关闭或失败时返回 None。

        与 ``guard._apply_execution_limits`` 共用同一份 ADV 数据：计划层先按上限
        摊薄、执行层再兜一次。两处封顶不冲突——执行层拿到的量已在上限内，
        不会出现「计划 8000 股 / 风控砍成 3200」的来回打架。
        """
        if getattr(self, "_adv_cache_done", False):
            return self._adv_cache
        self._adv_cache = None
        self._adv_cache_done = True
        try:
            cfg = getattr(self.guard, "config", None)
            if cfg is not None and not getattr(cfg, "execution_limits_enabled", True):
                return None
            from src.trading.liquidity import AdvProvider

            self._adv_cache = AdvProvider()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"流动性提供者初始化失败，计划层不封顶：{e}")
            self._adv_cache = None
        return self._adv_cache

    def _participation(self) -> float:
        try:
            return float(self.guard.config.max_participation_rate)
        except Exception:  # noqa: BLE001
            return 0.01

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

    def _maybe_settle(self, now: datetime, account, positions, report: TickReport) -> bool:
        """收盘结算的唯一入口：**任何** return 分支之前都必须先过它。

        历史坑：结算原本只挂在 ④ 段，而 ③ 熔断分支在它之前就 ``return`` 了。
        熔断不是「一 tick 的事」——它可能连续多天平息不下来，于是这些天的
        结算报告一份都不落盘：日报断更、绩效曲线缺段、进化面板没有当日样本，
        而这些恰恰是**判断系统是否还活着**的证据。结算纯记账，不涉及下单，
        没有任何理由跟着风控一起冻结。

        幂等：``_should_settle`` 内部有「今日报告已落盘即视为已结算」兜底，
        熔断分支与 ④ 段各调一次也不会重复结算。
        """
        if not self._should_settle(now):
            return False
        try:
            self._prices_for([p.code for p in positions])
            prices = dict(self._price_cache)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"结算取价失败（按持仓现价结算）: {e}")
            prices = {}
        try:
            self._settle(account, positions, prices, report)
        except Exception as e:  # noqa: BLE001
            # 结算异常绝不能把闭环打挂：报告写失败也要让状态机继续走，
            # 否则第二天 _should_settle 会再次尝试（文件不存在），形成死循环重试。
            logger.error(f"收盘结算失败（已跳过，下个 tick 重试）: {e}")
            return False
        self._set_state(LoopState.SETTLED)
        report.state = self._state.value
        self._last_report = report
        return True

    def _settle(self, account, positions, prices, report: TickReport):
        today = _today().strftime("%Y-%m-%d")
        # 账户快照的两种失效方式都必须挡住，否则报告会写出**假数据**：
        #   · account 为 None（券商/纸账户读取失败）→ 旧实现 equity=0，
        #     而 start_eq 是真实基线 ¥4.7M ⇒ day_pnl=-4.7M、day_pnl_pct=-100%，
        #     日报渲染出来就是「今日爆仓」，用户只能靠翻代码才知道是取数失败。
        #   · 当日基线缺失 → 旧实现回退到当前权益 ⇒ day_pnl 恒为 0，
        #     把真实亏损静默抹平（比报错更危险：看起来一切正常）。
        equity_ok = bool(account) and float(getattr(account, "total_asset", 0) or 0) > 0
        raw_equity = float(account.total_asset) if equity_ok else 0.0

        start_eq = self._day_start_equity.get(today)
        try:
            start_eq = float(start_eq) if start_eq is not None else None
        except (TypeError, ValueError):
            start_eq = None
        if not start_eq or start_eq <= 0:
            # 基线缺失：用当刻权益补建，并**落盘**，让同日后续 tick 不再走兜底。
            start_eq = raw_equity
            if raw_equity > 0:
                self._day_start_equity[today] = raw_equity
                try:
                    self._save_runtime()
                except Exception:  # noqa: BLE001
                    pass

        if equity_ok:
            equity = raw_equity
            day_pnl = equity - start_eq
        else:
            # 取不到账户 → 权益按基线计（当天视为「无变化」），并在报告里显式标注。
            # 宁可写 0，也不能写一个看起来像爆仓的负数。
            equity = float(start_eq or 0.0)
            day_pnl = 0.0

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
            "equity_ok": equity_ok,   # False = 账户快照取不到，权益为占位值
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

        # 超级集成健康审计：把各腿可用/降级状态并入日报（用户唯一主动翻的产物），
        # 让「腿静默降级」在结算日报里显眼（呼应 LSTM c10.dll / Alpha 1114 坑）。
        # 默认 use_ensemble=False 时不注入（日报保持干净）；启用时只读、零行为改变、
        # 不触网、不重训。任何异常都被吞掉，绝不影响日报落盘。
        try:
            from config.settings import config as _app_cfg
            if getattr(_app_cfg.ml, "use_ensemble", False):
                from src.models.predictor import predictor as _pred
                snapshot["ensemble_health"] = _pred.ensemble_report()
                snapshot["ensemble_alarm"] = _pred.ensemble_degradation_alert()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"超级集成健康审计并入日报失败（不影响日报）: {e}")

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

        # 业绩档案：先落当天快照，再重算累计（此刻今天的点已在内），
        # 然后把累计摘要补回日报并重写文件。
        # 为什么值得多写一次盘：日报是用户唯一会主动翻的产物，而「今天 -2.39%」
        # 单独看毫无意义 —— 必须配上「累计 -4.51% / 最大回撤 -4.9%」才知道
        # 系统整体在往哪走。少了这一块，日报永远只有当天视角。
        perf_rep = self._update_performance()
        if perf_rep:
            snapshot["performance"] = self._perf_brief(perf_rep)
            try:
                out_dir = Path(DATA_DIR) / "loop_reports"
                stamp = today.replace("-", "")
                (out_dir / f"loop_report_{stamp}.json").write_text(
                    json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                (out_dir / f"loop_report_{stamp}.md").write_text(
                    self._settle_markdown(snapshot), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"日报补写累计摘要失败（已忽略）: {e}")

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
        # 追加而非覆盖：熔断分支现在也会走到这里，若直接覆盖，「🛑 已熔断」
        # 这条提示就从当日最后一份 tick 报告里消失了 —— 而熔断恰恰是最该被看见的。
        _settle_note = (f"✅ 已收盘结算：当日盈亏 ¥{day_pnl:,.2f} "
                        f"({snapshot['day_pnl_pct']:+.2f}%)")
        report.note = f"{report.note} | {_settle_note}" if report.note else _settle_note

        # 推送
        if self.config.notify:
            self._push("闭环交易日报", self._settle_markdown(snapshot), snapshot)

        self._emit("loop_settled", snapshot)
        logger.info(report.note)

        # 业绩档案：结算后自动重算一次，看板/日报/进化门禁直接读 data/performance.json。
        # ⚠️ 路径全部显式取自本模块的 DATA_DIR（测试里会被 monkeypatch 到 tmp），
        # 绝不能让 performance 模块用它自己的默认路径 —— 那会把测试的虚拟权益
        # 写进真实 data/performance.json（同一类事故已经发生过一次）。
        self._update_performance()

    def _update_performance(self) -> dict:
        """重算并落盘业绩档案；失败只记 debug（业绩统计不该影响交易主流程）。"""
        try:
            from src.trading import performance as _perf
            _root = Path(DATA_DIR)
            rep = _perf.build_report(
                reports_dir=_root / "loop_reports",
                history_file=_root / "paper_trade_history.json",
                industry_file=_root / "industry_map.json",
                account_file=_root / "paper_account.json",
            )
            _perf.save_report(rep, _root / "performance.json")
            return rep
        except Exception as e:  # noqa: BLE001
            logger.debug(f"业绩档案更新失败（已忽略）: {e}")
            return {}

    @staticmethod
    def _perf_brief(rep: dict) -> dict:
        """从完整业绩档案里摘出日报需要的几项（None 原样保留，不冒充 0）。"""
        eq = rep.get("equity") or {}
        rk = rep.get("risk") or {}
        tr = rep.get("trades") or {}
        return {
            "period": rep.get("period") or {},
            "total_return_pct": eq.get("total_return_pct"),
            "annualized_return_pct": eq.get("annualized_return_pct"),
            "sharpe": rk.get("sharpe"),
            "max_drawdown_pct": rk.get("max_drawdown_pct"),
            "win_rate": tr.get("win_rate"),
            "payoff_ratio": tr.get("payoff_ratio"),
            "n_closes": tr.get("n_closes"),
            "exit_reason_distribution": rep.get("exit_reason_distribution") or [],
        }


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
        # 熔断/取数异常必须在日报里显眼 —— 这是用户唯一会主动翻的产物。
        # 反例：2026-10-22 起连续熔断、报告压根没生成（结算被熔断分支跳过），
        # 事后没有任何一份文件记录「这几天系统是停摆的」。补上结算之后，
        # 还要保证报告本身说清「当日是否在熔断态/权益是否可信」。
        if s.get("state") == "halted":
            lines.append("> ⚠️ **当日处于熔断态**：新开仓/调仓已冻结"
                         "（止损离场仍在执行），需人工或跨日自愈后才恢复交易。")
            lines.append("")
        if s.get("equity_ok") is False:
            lines.append("> ⚠️ **账户快照取不到**：权益/当日盈亏为占位值，不可用于业绩归因。")
            lines.append("")
        pf = s.get("performance") or {}
        if pf:
            _per = pf.get("period") or {}

            def _f(v, suf=""):
                return "—" if v is None else f"{v:+.2f}{suf}"

            _sharpe = "—" if pf.get("sharpe") is None else f"{pf['sharpe']:.2f}"
            _wr = ("—" if pf.get("win_rate") is None
                   else f"{pf['win_rate'] * 100:.0f}%")
            lines.append(
                f"**累计表现**（{_per.get('start') or '—'} ~ {_per.get('end') or '—'}，"
                f"{_per.get('n_days', 0)} 个交易日）："
                f"累计 {_f(pf.get('total_return_pct'), '%')}"
                f" ｜ 年化 {_f(pf.get('annualized_return_pct'), '%')}"
                f" ｜ 夏普 {_sharpe}"
                f" ｜ 最大回撤 {_f(pf.get('max_drawdown_pct'), '%')}"
                f" ｜ 平仓胜率 {_wr}（{pf.get('n_closes', 0)} 笔）"
            )
            # 退出原因分布：固定桶（含 0 值）。让「止盈 0 次 / 移动止损 0 次」
            # 这类决定性事实在日报里一眼可见 —— 实测曾因只有一个笼统的 stop 桶，
            # 「+15% 止盈线从没走到、亏损一路跑到 -5% 止损」被完全掩盖。
            erd = pf.get("exit_reason_distribution") or []
            if erd:
                lines.append("")
                lines.append("| 退出原因 | 已实现盈亏 | 笔数 | 胜率 |")
                lines.append("| --- | --- | --- | --- |")
                for e in erd:
                    _warn = " ⚠️未触发" if e["count"] == 0 else ""
                    lines.append(
                        f"| {e['label']}{_warn} | ¥{e['pnl']:,.0f} | "
                        f"{e['count']} | {('%.0f%%' % (e['win_rate'] * 100)) if e['count'] else '—'} |"
                    )
            lines.append("")
        if s["positions"]:
            lines.append("| 代码 | 名称 | 数量 | 成本 | 现价 | 盈亏% |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for p in s["positions"]:
                lines.append(
                    f"| {p['code']} | {p['name']} | {p['quantity']} | "
                    f"{p['avg_cost']} | {p['price']} | {p['pnl_pct']:+.2f} |")
        else:
            lines.append("_（空仓）_")
        # 超级集成健康审计：降级必须显眼（呼应 LSTM c10.dll / Alpha 1114 静默降级坑）。
        # 仅当 use_ensemble=True 时 snapshot 才带这两个字段（见 _settle 注入逻辑）。
        eh = s.get("ensemble_health")
        if eh:
            lines.append("")
            lines.append("## 🧩 超级集成健康")
            lines.append("")
            lines.extend(eh.splitlines())
        ea = s.get("ensemble_alarm")
        if ea:
            lines.append("")
            lines.append("> ⚠️ **超级集成有腿静默降级**：")
            for _al in ea.splitlines():
                lines.append(f"> {_al}")
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
                # ★ 「本金基线变了」本身**不足以**证明是人工重置：账户文件被写坏时
                # initial_cash 也会变。唯一可信的证据是 PaperAccount.reset() 落盘的
                # reset_at 时间戳前进了。2026-10-30 09:42 的空白单例覆盖正是反例 ——
                # 基线跟着变成 10 万，本分支照样 rebaseline，回撤归零、**熔断被静默解除**。
                _cur_reset_at = str(getattr(_pa, "reset_at", "") or "")
                _reset_evidenced = bool(
                    _cur_reset_at and _cur_reset_at != (self._last_pa_reset_at or "")
                )
                if _cur_baseline is not None:
                    if self._last_pa_baseline is None:
                        self._last_pa_baseline = _cur_baseline
                        self._last_pa_reset_at = _cur_reset_at
                    elif abs(_cur_baseline - self._last_pa_baseline) > 1e-6:
                        if not _reset_evidenced:
                            logger.error(
                                f"🛑 本金基线变了（{self._last_pa_baseline:,.0f}→"
                                f"{_cur_baseline:,.0f}）但账户文件里没有新的 reset_at —— "
                                f"这不是人工重置，疑似状态被写坏。**拒绝对齐风控基线**，"
                                f"保留原峰值 ¥{self.guard.safety._state.peak_equity:,.0f} "
                                f"与当前熔断状态，请人工核对 data/paper_account.json。"
                            )
                            try:
                                self.guard._emit("rebaseline_blocked", {
                                    "old_baseline": self._last_pa_baseline,
                                    "new_baseline": _cur_baseline,
                                    "reset_at": _cur_reset_at,
                                    "peak_equity": self.guard.safety._state.peak_equity,
                                    "reason": "初始本金变更但无 reset() 证据，疑似状态损坏",
                                })
                            except Exception:  # noqa: BLE001
                                pass
                            # 记录下来避免每 tick 重复告警；期间不改风控基线。
                            self._last_pa_baseline = _cur_baseline
                        else:
                            try:
                                _new_eq = _pa.get_total_asset()
                                if self.guard is not None and getattr(self.guard, "safety", None) is not None:
                                    self.guard.safety.rebaseline_peak(
                                        _new_eq, operator="autoloop",
                                        reason=(
                                            f"检测到纸账户重置(reset_at {_cur_reset_at}，initial_cash "
                                            f"{self._last_pa_baseline:,.0f}→{_cur_baseline:,.0f})，"
                                            f"自动对齐风控净值基线，避免误熔断"
                                        ),
                                    )
                                    logger.warning(
                                        f"♻️ 因纸账户重置自动对齐风控基线：净值基数→"
                                        f"¥{_new_eq:,.0f}（原 ¥{self._last_pa_baseline:,.0f}）"
                                    )
                                self._last_pa_baseline = _cur_baseline
                                self._last_pa_reset_at = _cur_reset_at
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

            # --- ②.5 策略形状 → 熔断基线对齐 ---
            # 必须早于 ③ 熔断监控：否则本 tick 会先按旧峰值判熔断、再重置基线，
            # 白白多冻结一个 tick；且若本 tick 正是交易日尾盘，会丢掉一次调仓机会。
            if account:
                try:
                    self._maybe_sync_strategy_version(report.equity)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"策略指纹同步异常（已忽略）: {e}")

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
                    # ★ 熔断期间仍必须结算 —— 这个 early-return 现在会越过 ④ 段，
                    # 而熔断可以持续**多天**（实测 2026-10-22 起因回撤熔断，之后
                    # data/loop_reports/ 一直停在 10-21，日报、绩效快照、进化面板
                    # 的当日样本全部断更）。结算是记账，不是交易：它不该被风控冻结。
                    if self._maybe_settle(now, account, positions, report):
                        return report
                    self._last_report = report
                    return report
                # 未熔断 → 允许下次再触发时重新推送（自愈/手动恢复后重熔断会再报一次）
                self._halt_notified = False

            # --- ④ 收盘结算（不受交易时段限制） ---
            if self._maybe_settle(now, account, positions, report):
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

            # T2 智能下单：每个交易 tick 让调度器吐出「此刻该发」的片
            if self._exec_scheduler is not None:
                self._drain_exec_scheduler(now, report)

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
            # 预演必须与真实 tick 用同一套参数（敞口系数 + 流动性封顶 + 持有纪律），
            # 否则「预演说买 8000 股、实盘只买 3200」这种不一致永远查不出来。
            risk_scale=self._portfolio_risk_scale(account),
            adv_provider=self._adv_provider(),
            participation=self._participation(),
            hold_if_profitable_pct=self.config.hold_if_profitable_pct,
            hold_rank_slack=float(
                getattr(self.config, "hold_rank_slack", 1.0) or 1.0),
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

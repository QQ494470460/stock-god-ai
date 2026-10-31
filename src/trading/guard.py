# -*- coding: utf-8 -*-
"""
交易安全与风控中枢 (Trading Guard)

为「全自动实盘(分阶段)」补齐三块致命的安全地基：

1. 武装状态机 (SafetyController)
   OFF → PAPER → LIVE_PENDING → LIVE 四级显式切换，状态持久化到磁盘。
   默认 PAPER，**绝不**在重启后自动进入实盘；进入 LIVE 需要两步确认令牌。

2. 全局 Kill Switch + 组合级熔断
   一键紧急停止（持久化，重启仍保持 halt），峰谷最大回撤熔断、
   单日亏损熔断，触发的瞬间冻结所有下单。

3. 人工确认闸门 (OrderGate) + 结构化成交日志 (TradeJournal)
   LIVE_PENDING 模式下每一笔实盘委托必须先「预览 → 审批 → 执行」；
   所有决策（信号/风控/审批/成交/熔断）写入 append-only 日志供合规审计。

另含高级仓位管理 (PositionSizer)：固定比例 / 波动率目标 / 半 Kelly。

本模块是「防御纵深」的最外层：它复用现有 TradeExecutor 的逐单风控
（仓位上限、涨跌停/T+1 拦截、止损止盈、AI 评分过滤），在其之上叠加
状态机、确认闸门与组合级熔断。两者都拦，才算通过。
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

from loguru import logger

from src.trading.broker import (
    BrokerAPI, Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo, TickData,
)
from src.trading.executor import TradeExecutor, RiskConfig
from src.trading.paper import PaperAccount
from src.trading.rules import check_buyable, check_sellable, sellable_quantity
from src.trading.regime import RegimeWeighter, RegimeSmoother
from src.trading.intelligence import StrategyCredibility
from src.trading.sanity import wrap_broker

try:  # 与 calendar.py 一致：打包/测试环境下 config 可能不在 sys.path 顶层
    from config.settings import ROOT_DIR, DATA_DIR
except Exception:  # noqa: BLE001
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"

# 策略信誉引擎的持久化快照（由 guard 每个有效 tick 落盘，看板只读不写）。
CRED_STATE_PATH = DATA_DIR / "credibility_state.json"

# 当日买入换手累计（跨进程重启必须保留，否则「重启一次预算就满血」）。
TURNOVER_STATE_PATH = DATA_DIR / "turnover_state.json"

# 测试环境下不落盘：sync_credibility_from_paper 原本是「零 IO」方法，测试套件里
# 大量实例化 guard 并调用它，若真实写盘会污染生产 data/ 目录。
#
# 注意：必须在**调用时**判定，不能在模块 import 时算一次 —— pytest 的
# PYTEST_CURRENT_TEST 是「每个用例执行期间」才注入的，而 guard 模块在收集
# （import）阶段就被导入了，那时该变量还不存在，import-time 的快照永远是 False
# （实测：曾因此让一次回归把 data/credibility_state.json 写成 momentum n=30）。
# 用 `"pytest" in sys.modules` 作为判据最稳：跑 pytest 时它必然已导入。
def _running_under_pytest() -> bool:
    import sys
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


# 行业映射缺失时的兜底桶名。所有「在 industry_map 里查不到」的标的都归到这里，
# 让集中度检查对它们**依然生效**（fail-safe），而不是整段跳过（fail-open）。
UNKNOWN_SECTOR = "__UNKNOWN__"


def _load_industry_map() -> Dict[str, str]:
    """默认从 ``data/industry_map.json`` 装载行业映射。

    为什么必须自动装载：行业集中度是本中枢的硬风控之一，但构造器此前的
    ``industry_map or {}`` 意味着**调用方不传就恒为空** —— 于是
    ``_check_portfolio_limits`` 里的行业检查被静默跳过（fail-open），
    这条风控实际上从未生效。现在改为默认装载，缺失/损坏才退回空表并告警。

    测试环境跳过：避免用例依赖生产 data/ 的当前内容（hermetic）。
    """
    if _running_under_pytest():
        return {}
    path = DATA_DIR / "industry_map.json"
    try:
        if not path.exists():
            logger.warning(f"行业映射缺失（{path}）：行业集中度风控将被跳过")
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            clean = {str(k): str(v) for k, v in raw.items() if v}
            if clean:
                return clean
        logger.warning("行业映射为空：行业集中度风控将被跳过")
        return {}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"行业映射载入失败（集中度风控跳过）: {e}")
        return {}


# ======================================================================
# 武装模式
# ======================================================================

class TradingMode(str, Enum):
    """交易武装模式 —— 越往后风险越高，切换必须显式且可审计"""
    OFF = "off"                  # 完全停止，连纸交易都不跑（硬保险）
    PAPER = "paper"              # 纸交易：零风险验证策略
    LIVE_PENDING = "live_pending"  # 已连接实盘，但每笔委托需人工确认
    LIVE = "live"                # 全自动实盘：系统自主下单


# ======================================================================
# 配置
# ======================================================================

@dataclass
class GuardConfig:
    """安全与风控配置（安全优先，默认值全部保守）"""

    # ---- 武装模式 ----
    default_mode: str = "paper"            # 默认纸交易，绝不默认实盘
    require_live_confirmation: bool = True  # 进入 LIVE 必须两步确认令牌

    # ---- 组合级风控（熔断）----
    max_portfolio_drawdown_pct: float = 8.0   # 峰谷最大回撤熔断线(%)
    max_daily_loss_pct: float = 3.0           # 单日最大亏损比例(%)
    max_daily_loss_abs: float = 5000.0        # 单日最大亏损金额(元)；0=不设固定金额线
    # 金额线也可按「当前权益 × 百分比」推导，账户规模变化时自动缩放。
    # >0 时生效，并与 max_daily_loss_abs 取大者（固定值当兜底下限）。
    max_daily_loss_abs_pct: float = 0.0       # 单日最大亏损金额线(占权益%)

    # ---- 熔断自诊断 + 智能自愈（Feature #5）----
    # 仅对「自动触发」(halted_by 以 "auto(" 开头) 且当前为 PAPER 的熔断生效；
    # 人工 KILL 与任何实盘熔断绝不自动恢复（对齐 resume 铁律）。
    auto_resume_cooldown_sec: int = 3600      # 熔断后至少冷却多久才允许自愈评估
    auto_resume_require_drawdown_clear: bool = True   # 自愈前回撤必须已回落到熔断线之下
    auto_resume_require_dailyloss_clear: bool = True  # 自愈前单日亏损必须已收复
    auto_resume_throttle_sec: int = 1800      # 自愈后「降权冷静期」时长（throttle_days<=0 时生效）
    auto_resume_risk_fraction: float = 0.5    # 冷静期内 signal_strength 折扣（部分解冻）
    # ★ 自愈滞后带：解冻要求回撤回落到「熔断线 − margin」之下，而不是「刚好不超线」。
    # 没有滞后带时账户必在 8% 线上反复横跳：实测 10-26 收 7.54% 解冻 → 10-27 建仓
    # 138 万 → 收 8.70% 再熔断 → 10-28 收 8.06%（差 ¥2,754 未达线），用户体感就是
    # 「怎么老是这样」。留 2 个百分点的余量，解冻才算真出坑。
    auto_resume_drawdown_margin_pct: float = 2.0
    # 解冻后按**交易日**计的半仓冷静期（<=0 时退回 auto_resume_throttle_sec）。
    # 必须按交易日：周五解冻 + 2 个自然日 = 周日，等于没有冷静期。
    auto_resume_throttle_days: float = 2.0

    # ---- 持仓约束 ----
    max_positions: int = 20                  # 最大持仓只数
    max_single_position_pct: float = 0.30    # 单票最大仓位(占总资产)
    max_sector_pct: float = 0.50             # 单一行业最大集中度(占总资产)

    # ---- 止盈止损（透传给 TradeExecutor 的 RiskConfig）----
    # 放在这里而不是写死在 RiskConfig 默认值里，是为了「改配置即生效，无需改代码」。
    # atr_stop_* 见 src/trading/stops.py：固定 -5% 止损在 A 股是结构性错误，
    # 70% 年化波动的题材股日内振幅本身就有 5%，固定线等于开盘就被扫。
    stop_loss_pct: float = -5.0
    stop_profit_pct: float = 15.0
    trailing_stop_pct: float = -3.0
    atr_stop_enabled: bool = False
    atr_stop_k: float = 2.0                  # 止损宽度 = max(|配置值|, k × ATR%)
    atr_trailing_k: float = 1.5              # 移动止损用更紧的倍数
    atr_stop_min_pct: float = 2.0            # 止损宽度下限(%)
    atr_stop_max_pct: float = 15.0           # 止损宽度上限(%)

    # ---- 下单频率 ----
    min_order_interval_sec: float = 2.0      # 两笔委托最小间隔(秒)
    max_orders_per_minute: int = 10          # 每分钟最大委托数

    # ---- 执行约束（流动性 / 风险预算 / 换手率，2026-10-26 新增）----
    # 为什么需要：仓位比例只回答「最多放多少」，不回答「这笔能不能吃掉」。
    # 单笔金额可能是中小市值标的日成交额的一大截（冲击成本、甚至打板），
    # 而「固定 % 仓位」隐含了「单笔潜在亏损一致」的假设，对高波动个股并不成立。
    execution_limits_enabled: bool = True
    max_participation_rate: float = 0.01     # 单笔金额 ≤ 20日平均成交额 × 1%
    max_risk_per_trade_pct: float = 0.005    # 单笔潜在亏损 ≤ 总资产 × 0.5%
    risk_budget_stop_pct: float = 0.05       # 换算止损距离（对应 -5% 止损线）
    # 预算按「止损可能放宽到的上限」算，而不是按建仓当时的宽度算。
    # ATR 自适应止损只增不减（最宽 atr_stop_max_pct），建仓时按 -5% 放行的仓位，
    # 后续止损被放宽到 -15% 时实际风险翻 3 倍 —— 预算形同虚设。
    risk_budget_use_stop_ceiling: bool = True
    # 当日买入总额 ≤ 总资产 × 该比例（仅买侧）。
    # ⚠️ 必须 ≥ 1 - cash_reserve_pct：一次「从现金建满仓」需要一天内买入
    # ``(1 - 现金保留)`` 的敞口。原值 0.30 会让每次调仓只能建 1/3 仓位，
    # 而调仓周期是 15 个交易日 ⇒ 账户要 45 个交易日（两个多月）才建满，
    # 期间长期低仓、大量现金闲置。**回测是 T+1 一次性建满的，没有这个 30% 闸门。**
    # 实测证据：data/turnover_state.json 记录 2026-10-27 当日买入 ¥1,382,444，
    # 当时权益约 ¥4.55M —— 正好卡死在 30% 上，不是巧合。
    # 「抑制来回翻仓」这件事现在由 rebalance_days=15 + band_pct=0.15 承担，
    # 本项退化为「一天绝不能倒两次仓」的物理上限（一天最多建满一次）。
    max_daily_turnover_pct: float = 0.95

    # ---- 组合尾部风险（相关性 VaR 约束，2026-10-26 新增）----
    # 「单票 X%、最多 N 只」只约束名义敞口，不约束相关性：20 只同涨同跌的票
    # 名义分散、风险等价于 1 只加杠杆。这里按协方差算组合 1 日 VaR，
    # 超预算就**连续降新开仓强度**（不清仓），把尾部风险摁在预算内。
    var_monitor_enabled: bool = True
    max_portfolio_var_pct: float = 0.025     # 组合 1 日 95% VaR 上限（占总资产）
    var_confidence_z: float = 1.65           # 95% 单尾
    var_scale_floor: float = 0.40            # 超预算时最低缩放到 40%（不清仓）

    # ---- 仓位规模 ----
    sizing_method: str = "fixed_fractional"  # fixed_fractional | volatility_target | kelly
    base_risk_fraction: float = 0.02         # 每笔风险敞口占净值比例(固定比例法)
    volatility_target_pct: float = 0.15      # 年化波动率目标(波动率目标法)
    kelly_cap: float = 0.10                  # Kelly 仓位上限(半 Kelly 后再封顶)

    # ---- 市场状态感知 + 策略动态加权（Feature #4）----
    # 默认开启；未接入指数数据前 regime=sideways → 所有权重≈1.0，零行为变化。
    regime_weighting_enabled: bool = True

    # ---- 波动率尾部风险自适应降仓（Vol-Control，Feature #6）----
    # 监控组合净值已实现波动率，单边上行不限制、只在「波动率异常飙涨」时自动
    # 压低**新开仓**强度（不碰止损/风控离场），相当于给系统装一个「恐慌阻尼器」。
    # 默认开启；纯加法、fail-safe：数据不足/计算异常时退化为不干预（系数=1.0）。
    vol_control_enabled: bool = True
    vol_control_window: int = 20          # 滚动计算已实现波动的样本窗口(交易日)
    vol_control_spike_multiple: float = 2.5  # 当前波动 ≥ 基线 × 该倍数 → 触发降仓
    vol_control_min_strength: float = 0.3    # 触发后新开仓强度下限（不完全停手，留流动性）

    # ---- 市场状态粘性（Regime Hysteresis，Batch 3）----
    # 要求同一新 regime 信号连续出现 N 个交易日才真正切换，避免基准均线在
    # 横盘/假突破时天天横跳、策略随之忽攻忽守（whipsaw）。
    regime_hysteresis_days: int = 2

    # ---- 策略信誉自适应权重（Online Credibility，Batch 3）----
    # 用每笔已平仓交易的「信号方向 vs 实际盈亏」一致性（类 IC 的 EWMA 代理）
    # 为每个策略标签维护信誉，映射成 ic_weights 喂给多策略集成器（Feature #7），
    # 让集成权重随真实战绩自我调优。默认开启；未知标签按中性 1.0 兜底。
    credibility_enabled: bool = True
    credibility_halflife: int = 20         # EWMA 半衰期（笔）

    # ---- 波动自适应熔断线（Volatility-scaled kill-switch，Batch 3）----
    # 单日亏损熔断线不再写死 3%，而是随「基准近期日波动」动态缩放：
    # 平静市场波动小 → 线收紧（更敏感，更早保护）；动荡市场波动大 → 线放宽
    # （抗噪声，避免被正常波动误杀）。始终夹在 [floor, cap] 之间，且回撤熔断
    # 线(8%) 作为绝对底线始终生效，分层安全不被削弱。
    adaptive_kill_switch: bool = True
    kill_switch_ref_vol_pct: float = 1.5   # 参考日波动(%)，作为缩放中枢
    kill_switch_floor_pct: float = 2.0     # 下限（绝不更松）
    kill_switch_cap_pct: float = 4.5       # 上限（绝不更紧）

    # ---- 人工确认 ----
    confirm_timeout_sec: float = 300.0       # 待确认委托超时(秒)，超时自动撤销

    # ---- 持久化路径 ----
    state_file: str = ""                     # 武装状态文件(默认 data/trading_guard_state.json)
    journal_file: str = ""                   # 成交日志(默认 data/trading_journal.jsonl)

    def __post_init__(self):
        if not self.state_file:
            self.state_file = str(DATA_DIR / "trading_guard_state.json")
        if not self.journal_file:
            self.journal_file = str(DATA_DIR / "trading_journal.jsonl")

    @classmethod
    def from_yaml(cls, path: str) -> "GuardConfig":
        """从 YAML 覆盖默认值（缺字段用默认）"""
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            logger.info(f"🛡️ 已加载安全配置: {path}")
            return cfg
        except FileNotFoundError:
            logger.warning(f"安全配置文件不存在 {path}，使用默认保守配置")
            return cls()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"安全配置解析失败({e})，使用默认保守配置")
            return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    def daily_loss_abs_limit(self, equity: float) -> float:
        """单日亏损「绝对金额线」的实际阈值。

        历史坑：写死 ¥5,000 对小额账户合理，但账户做到 ~¥500 万后只占 0.10%，
        几乎每天都熔断，闭环事实上跑不起来（2026-10-20 就是这么停的）。
        因此支持按权益比例推导：

        - max_daily_loss_abs_pct > 0 → 金额线 = 权益 × pct%，并取 max(固定值, 推导值)
          （固定值只当兜底下限，设 0 即完全交给比例推导）
        - 否则 → 沿用固定金额线 max_daily_loss_abs
        - 两者都为 0 → 返回 inf，等于「不设金额线」，只由 3% 比例线把关
        """
        fixed = float(self.max_daily_loss_abs or 0.0)
        pct = float(self.max_daily_loss_abs_pct or 0.0)
        if pct > 0:
            # ⚠️ 耦合告警：金额线比比例线更紧时，比例线形同虚设。
            # 2026-10-26 实测踩到：abs_pct=1.0 而 max_daily_loss_pct=3.0，
            # 结果每天在 -1% 就熔断，3% 那条线一次都没机会生效，闭环下午全静默。
            # 两条线是「或」的关系，紧的那条才是真实阈值 —— 必须显式提醒。
            if pct < float(self.max_daily_loss_pct or 0.0) and not getattr(
                self, "_warned_abs_vs_pct", False
            ):
                object.__setattr__(self, "_warned_abs_vs_pct", True)
                logger.warning(
                    f"⚠️ 单日亏损金额线({pct:.2f}% 权益) 比比例线"
                    f"({self.max_daily_loss_pct:.2f}%) 更紧 → 实际生效阈值是 {pct:.2f}%。"
                    f"如非本意，把 max_daily_loss_abs_pct 调到 >= max_daily_loss_pct 或设 0。"
                )
            derived = max(0.0, float(equity or 0.0)) * pct / 100.0
            return max(fixed, derived)
        if fixed > 0:
            return fixed
        return float("inf")


# ======================================================================
# 武装状态机 + 全局 Kill Switch
# ======================================================================

@dataclass
class GuardState:
    """持久化的武装状态"""
    mode: str = "paper"
    armed_by: str = ""
    armed_at: str = ""
    halted: bool = False
    halt_reason: str = ""
    halt_at: str = ""
    halted_by: str = ""
    peak_equity: float = 0.0         # 组合净值峰值(用于回撤熔断，跨会话保留)
    pending_token: str = ""          # arm_live 生成的待确认令牌
    pending_operator: str = ""
    # arm_live(target=...) 请求的目标模式，confirm_arm 时据此落地。
    # 旧状态文件没有此字段 → 空串 → confirm_arm 默认落 LIVE_PENDING（最保守）。
    pending_target: str = ""
    today_start_equity: float = 0.0  # 当日起始净值(用于单日亏损熔断)
    today_date: str = ""
    # 峰值/当日起始净值所对应的资金账号。
    # 用途：账号切换（纸交易 ↔ 实盘、换券商、纸账户重置）时净值基数变化巨大，
    # 若沿用旧 peak 会立刻误判「巨额回撤」并熔断（曾真实发生：纸账户 10 万
    # 对上遗留 peak 100 万 → 首 tick 报 89.95% 回撤直接 halt）。
    # 旧状态文件无此字段时为空串，表示「未记录」，此时不做自动重置。
    baseline_account: str = ""
    # 峰值所依据的「策略形状指纹」（由 loop 计算，含周期/权重/退出规则等）。
    # 回撤熔断的语义是「这个策略亏得太多」；策略一换，旧峰值就是**另一个
    # 策略**的成绩，拿它考核新策略会直接把闭环锁死（实测 2026-10-31 改造后
    # 回撤 10.86% 卡在 6% 解冻线下方，93% 现金根本回不来，新规则一行都跑不到）。
    # 空串 = 旧状态文件，只登记不重置（见 SafetyController.sync_strategy_version）。
    strategy_version: str = ""


class SafetyController:
    """武装状态机 + 全局 Kill Switch

    设计铁律：
    - 默认 PAPER，重启后**绝不**自动实盘（halt 状态也会持久化）。
    - 进入 LIVE 必须两步：arm_live() 生成令牌 → confirm_arm(token) 完成。
    - halt() 一旦触发，必须 resume() 显式解除；若曾实盘，resume 后强制回到 PAPER，
      需重新武装才能再上实盘。
    """

    def __init__(self, config: GuardConfig, broker=None):
        self.config = config
        # broker 只用于「武装实盘前确认券商真的是实盘连接」这一项校验。
        # 传 None = 没有券商接入 → 按模拟处理（fail-closed），拒绝武装实盘。
        self._broker = broker
        self._state_file = Path(config.state_file)
        self._lock = threading.RLock()
        self._state = GuardState()
        self._load()
        # 若磁盘无状态，套用默认模式
        if not self._state.mode:
            self._state.mode = config.default_mode
            self._save()

    # ---------- 持久化 ----------
    def _load(self):
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if hasattr(self._state, k):
                        setattr(self._state, k, v)
                logger.info(
                    f"🛡️ 武装状态已恢复: mode={self._state.mode} "
                    f"halted={self._state.halted}"
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"武装状态读取失败({e})，使用默认 PAPER")

    def _save(self):
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(
                json.dumps(asdict(self._state), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"武装状态写入失败({e})")

    # ---------- 查询 ----------
    @property
    def mode(self) -> TradingMode:
        try:
            return TradingMode(self._state.mode)
        except ValueError:
            return TradingMode.PAPER

    @property
    def halted(self) -> bool:
        return bool(self._state.halted)

    def can_trade(self) -> bool:
        """是否允许交易：未熔断 且 模式非 OFF"""
        return (not self._state.halted) and self._state.mode != TradingMode.OFF.value

    def requires_per_order_confirmation(self) -> bool:
        """该模式下每笔实盘委托是否需要人工确认"""
        if not self.config.require_live_confirmation:
            return False
        return self._state.mode in (
            TradingMode.LIVE_PENDING.value, TradingMode.LIVE.value
        )

    def is_live(self) -> bool:
        return self._state.mode in (
            TradingMode.LIVE_PENDING.value, TradingMode.LIVE.value
        )

    # ---------- 模式切换 ----------
    def set_mode(self, mode: TradingMode, operator: str = "system") -> bool:
        """直接设置模式（仅用于 OFF/PAPER 之间，或测试）"""
        with self._lock:
            # 任何向实盘的切换都必须走 arm 流程，不允许 set_mode 直接进 LIVE
            if mode in (TradingMode.LIVE, TradingMode.LIVE_PENDING):
                logger.warning("⛔ 不允许直接 set_mode 进入实盘，请使用 arm_live()")
                return False
            old = self._state.mode
            self._state.mode = mode.value
            self._state.armed_by = operator
            self._state.armed_at = datetime.now().isoformat()
            self._save()
            if old != mode.value:
                self._journal_event("mode_change", {
                    "from": old, "to": mode.value, "operator": operator,
                    "reason": "set_mode",
                })
            logger.info(f"🛡️ 模式切换 → {mode.value} (by {operator})")
            return True

    def _broker_is_simulated(self) -> bool:
        """券商连接是否为模拟（或无法确认）。

        ⚠️ 这里修的是一个**静默失效**：武装 LIVE 之前从不校验券商到底连的
        是实盘还是模拟。``is_simulated`` 一直只在看板上显示，guard / loop /
        executor 谁都没看它。后果是「mode=live、界面写着实盘、单子却全进了
        模拟通道」——用户以为在拿真钱跑，实际什么都没发生。

        缺失 ``is_simulated`` 属性时按**模拟**处理（fail-closed）：宁可拒绝
        武装、让人去查，也不能给一个「看起来在实盘」的假象。
        """
        try:
            return bool(getattr(self._broker, "is_simulated", True))
        except Exception:  # noqa: BLE001
            return True

    def arm_live(self, operator: str, target: TradingMode = TradingMode.LIVE_PENDING) -> str:
        """第一步：请求武装实盘，返回需回传的确认令牌

        Args:
            operator: 操作员标识（人名/工号），记入审计
            target:   目标模式，默认 LIVE_PENDING（逐单确认），最安全
        Returns:
            确认令牌字符串；调用 confirm_arm(token) 才能完成切换

        Raises:
            ValueError: 目标模式非法，或券商仍是模拟连接（拒绝武装实盘）
        """
        if target not in (TradingMode.LIVE, TradingMode.LIVE_PENDING):
            raise ValueError("arm_live 目标只能是 LIVE 或 LIVE_PENDING")
        if self._broker_is_simulated():
            reason = str(getattr(self._broker, "sim_reason", "") or "券商连接处于模拟模式，或无法确认")
            logger.error(f"⛔ 拒绝武装实盘：{reason}")
            raise ValueError(
                f"拒绝武装实盘：{reason}。"
                "武装前必须先用 live 模式连上真实券商（本机可用 "
                "`python scripts/check_live_ready.py` 自检）。"
            )
        with self._lock:
            token = secrets.token_hex(4)  # 8 位十六进制，足够人工核对
            self._state.pending_token = token
            self._state.pending_operator = operator
            # 记住请求的目标模式：confirm_arm 必须按它落地，否则
            # promote_to_live() 会被静默钉死在 LIVE_PENDING，永远进不了 LIVE。
            self._state.pending_target = target.value
            self._save()
            self._journal_event("arm_request", {
                "operator": operator, "target": target.value, "token": token,
            })
            logger.warning(
                f"⚠️ 收到武装实盘请求 → {target.value} (by {operator})\n"
                f"   请人工核对后执行 confirm_arm(token='{token}') 完成切换"
            )
            return token

    def confirm_arm(self, token: str, operator: str) -> bool:
        """第二步：回传令牌确认，完成武装"""
        with self._lock:
            if not self._state.pending_token:
                logger.error("⛔ 无待确认令牌，请先调用 arm_live()")
                return False
            if token != self._state.pending_token:
                logger.error("⛔ 确认令牌不匹配，武装被拒绝（保留令牌可重试）")
                return False

            # 二次校验：拿到令牌后券商可能被重新连成模拟（或断线降级）。
            # 只校验第一步是不够的 —— 令牌有效期里连接状态会变。
            if self._broker_is_simulated():
                reason = str(getattr(self._broker, "sim_reason", "") or "券商连接已变为模拟，或无法确认")
                logger.error(f"⛔ 武装被拒：{reason}")
                self._journal_event("arm_reject", {"operator": operator, "reason": reason})
                return False

            # 按 arm_live 请求的目标落地模式。
            # ⚠️ 曾在此处硬编码 LIVE_PENDING，导致 promote_to_live()
            #    （arm_live(target=LIVE) + confirm_arm）永远停在 LIVE_PENDING、
            #    LIVE 成了不可达状态 —— 界面/调用方以为升到了全自动实盘，实际没有。
            #    旧状态文件没有 pending_target → 默认 LIVE_PENDING（最保守）。
            req = (self._state.pending_target or "").strip()
            if req not in (TradingMode.LIVE.value, TradingMode.LIVE_PENDING.value):
                req = TradingMode.LIVE_PENDING.value

            # 进入实盘前强制清空当日起始净值，确保单日亏损熔断从零点计
            self._state.mode = req
            self._state.armed_by = operator
            self._state.armed_at = datetime.now().isoformat()
            self._state.pending_token = ""
            self._state.pending_operator = ""
            self._state.pending_target = ""
            self._save()
            self._journal_event("arm_confirm", {
                "operator": operator, "token": token, "mode": self._state.mode,
            })
            if req == TradingMode.LIVE.value:
                logger.warning(
                    f"🔴🔴 已武装【全自动实盘 LIVE】by {operator} —— "
                    f"此后系统自主下单，不再逐单人工确认"
                )
            else:
                logger.warning(
                    f"🔴 已武装实盘(LIVE_PENDING) by {operator}，"
                    f"此后每笔委托需人工确认"
                )
            return True

    def promote_to_live(self, operator: str) -> bool:
        """从 LIVE_PENDING 升级到 LIVE 全自动（仍需令牌确认，复用 arm 流程）"""
        token = self.arm_live(operator, target=TradingMode.LIVE)
        # 提示调用方需 confirm_arm；此处不自动确认，保持显式
        return bool(token)

    def disarm(self, operator: str = "system", reason: str = "") -> bool:
        """解除实盘武装，回到纸交易（安全方向，无需令牌）

        任何从 LIVE/LIVE_PENDING 退回 PAPER 的动作都写入 journal（mode_change），
        确保「收紧风控」也能被审计追溯 —— 不能只改状态文件而留不下痕迹。
        """
        with self._lock:
            if self._state.mode in (TradingMode.LIVE.value, TradingMode.LIVE_PENDING.value):
                old = self._state.mode
                self._state.mode = TradingMode.PAPER.value
                self._state.armed_by = operator
                self._state.armed_at = datetime.now().isoformat()
                self._save()
                self._journal_event("mode_change", {
                    "from": old, "to": TradingMode.PAPER.value,
                    "operator": operator,
                    "reason": reason or "disarm→PAPER (security default)",
                })
                logger.warning(f"🟢 已解除实盘武装 → PAPER (by {operator})")
                return True
            return True

    # ---------- Kill Switch ----------
    def halt(self, reason: str, operator: str = "system") -> bool:
        """全局紧急停止：持久化，重启仍保持

        **幂等**：同因同操作者重复调用不再落盘/打 WARNING。``tick_monitor`` 每
        30s 被调一次，熔断期间条件仍成立 → 旧实现每次都会写一次状态文件并打一条
        WARNING。实测一天刷出 1,250 条「KILL SWITCH 触发」，日志 4,671 条里绝大
        部分是同一个状态在自言自语，看板/日志永远在滚「已熔断」，用户无法判断
        这是新的一次熔断还是旧状态。幂等后「触发」只发生在真正的跃迁上。
        """
        with self._lock:
            if (self._state.halted
                    and self._state.halt_reason == reason
                    and self._state.halted_by == operator):
                logger.debug(f"KILL SWITCH 已处于熔断（同因重复触发，忽略）: {reason}")
                return True
            self._state.halted = True
            self._state.halt_reason = reason
            self._state.halt_at = datetime.now().isoformat()
            self._state.halted_by = operator
            self._save()
            logger.warning(f"🛑 KILL SWITCH 触发: {reason} (by {operator})")
            return True

    def resume(self, operator: str = "system") -> bool:
        """解除 halt；若曾实盘，强制回到 PAPER，需重新武装"""
        with self._lock:
            if not self._state.halted:
                return True
            self._state.halted = False
            self._state.halt_reason = ""
            self._state.halt_at = ""
            self._state.halted_by = ""
            # 安全方向：解除熔断后绝不自动回实盘
            if self._state.mode in (TradingMode.LIVE.value, TradingMode.LIVE_PENDING.value):
                self._state.mode = TradingMode.PAPER.value
                logger.warning(f"🟢 KILL SWITCH 解除 → 强制回到 PAPER (by {operator})，需重新武装才能实盘")
            else:
                logger.info(f"🟢 KILL SWITCH 解除 (by {operator})")
            self._save()
            return True

    # ---------- 净值追踪（供熔断使用） ----------
    def update_equity(self, total_equity: float):
        """更新净值峰值（用于峰谷回撤熔断）"""
        with self._lock:
            if total_equity > self._state.peak_equity:
                self._state.peak_equity = total_equity
            self._save()

    def current_drawdown_pct(self, total_equity: float) -> float:
        peak = self._state.peak_equity
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - total_equity) / peak * 100)

    def set_today_start(self, equity: float):
        with self._lock:
            self._state.today_start_equity = equity
            self._state.today_date = datetime.now().strftime("%Y-%m-%d")
            self._save()

    def daily_loss_pct(self, total_equity: float) -> float:
        start = self._state.today_start_equity
        if start <= 0:
            return 0.0
        return (total_equity - start) / start * 100

    def rebaseline_peak(self, equity: float, operator: str = "system",
                        reason: str = "", account_id: str = "") -> None:
        """把净值基数（峰值 + 当日起始）重置为当前权益。

        **什么时候需要**：账户的资金基数发生了「与交易无关」的变化 ——
        切换券商/账号、纸账户重置、出入金。此时旧 peak 相对新权益会
        显得异常高，回撤熔断会立刻误触发（实测：纸账户 10 万 vs 遗留
        peak 100 万 → 首 tick 报 89.95% 回撤直接 halt，且无法自行恢复）。

        **为什么不做成自动**：真·腰斩与「换了账号」在净值曲线上无法区分，
        静默放宽熔断线是危险行为。故仅提供显式入口，由操作者决定。
        （唯一例外见 ``tick_monitor``：资金账号发生明确切换时自动重置。）

        Args:
            equity: 当前总资产，作为新基线
            operator: 操作者，写入 journal
            reason: 重置原因（人工排查时很有用）
            account_id: 若给定，同时更新 baseline_account
        """
        with self._lock:
            old_peak = self._state.peak_equity
            self._state.peak_equity = float(equity)
            self._state.today_start_equity = float(equity)
            self._state.today_date = datetime.now().strftime("%Y-%m-%d")
            if account_id:
                self._state.baseline_account = account_id
            elif not self._state.baseline_account:
                self._state.baseline_account = ""
            self._save()
        logger.warning(
            f"♻️ 净值基数已重置为 ¥{equity:,.0f}（原峰值 ¥{old_peak:,.0f}）"
            f" by {operator}" + (f"｜原因：{reason}" if reason else "")
        )
        self._journal_event("rebaseline_peak", {
            "operator": operator, "old_peak": old_peak, "new_baseline": equity,
            "reason": reason, "account_id": account_id,
        })

    def sync_strategy_version(self, version: str, equity: float,
                              reason: str = "") -> Tuple[bool, str]:
        """策略形状变了 ⇒ 把净值基线对齐到当前权益。

        **为什么必须做**：回撤熔断的语义是「这个策略亏得太多」。策略形状
        （周期 / 权重 / 退出规则）一换，旧峰值就是**另一个策略**的成绩，
        拿它考核新策略会直接把闭环锁死 —— 实测 2026-10-31 改造后回撤
        10.86% 卡在 6% 解冻线下方，93% 现金根本回不来，新规则一行代码都
        执行不到。

        **为什么不是无条件重置**（三条 fail-safe）：
        - ``version`` 为空 / 权益非法 → 直接不动，返回未重置；
        - 旧状态文件没有该字段（空串）→ **只登记不重置**。首次引入本机制
          就静默放宽一次熔断是不可接受的，那一次必须由操作者显式
          ``rebaseline_peak``（CLI ``--rebaseline-peak``）；
        - 版本相同 → 不动。

        Returns:
            ``(是否重置, 说明)``
        """
        version = (version or "").strip()
        if not version:
            return False, "策略指纹为空，跳过"
        try:
            equity = float(equity)
        except (TypeError, ValueError):
            return False, "权益非法，跳过"
        if equity <= 0:
            return False, "权益非正，跳过"
        with self._lock:
            prev = (self._state.strategy_version or "").strip()
            if not prev:
                self._state.strategy_version = version
                self._save()
                logger.info(
                    f"🧬 已登记策略指纹 {version}（首见，**不改动**净值基线；"
                    f"若确因策略变更需要重置，请显式 --rebaseline-peak）")
                return False, "首见，仅登记"
            if prev == version:
                return False, "策略指纹未变"
            old_peak = float(self._state.peak_equity or 0.0)
            self._state.strategy_version = version
        # rebaseline_peak 自带锁与落盘，且会写 rebaseline_peak 审计事件
        self.rebaseline_peak(
            equity, operator="autoloop",
            reason=(reason or (
                f"策略形状变更 {prev}→{version}；旧峰值 ¥{old_peak:,.0f} "
                f"属于另一个策略，不对新策略计回撤")),
        )
        return True, f"{prev}→{version}"

    def _journal_event(self, event: str, data: dict) -> None:
        """把关键状态变更落到 TradeJournal（append-only，便于事后审计）。

        `TradeJournal` 在本模块后面才定义，但方法体在调用时求值，
        那时模块已加载完毕，直接引用全局名即可。
        """
        try:
            TradeJournal(self.config).log(event, data)  # noqa: F821
        except Exception:  # noqa: BLE001  记录失败不影响主流程
            pass

    def snapshot(self) -> dict:
        return asdict(self._state)


# ======================================================================
# 订单预览 / 审批闸门
# ======================================================================

@dataclass
class OrderProposal:
    proposal_id: str
    code: str
    name: str
    side: str
    price: float
    quantity: int
    est_amount: float
    ai_score: float
    agreement: float
    signal_strength: float
    strategy_tag: str
    status: str = "pending"   # pending|approved|rejected|executed|blocked|expired
    reason: str = ""
    created_at: str = ""
    decided_at: str = ""
    decided_by: str = ""
    order_id: str = ""
    risk_exit: bool = False   # 风控性离场（止盈止损/调仓清仓）：跳过「AI评分仍高不卖」过滤
    raw: dict = field(default_factory=dict)


class OrderGate:
    """人工确认闸门

    PAPER/OFF 模式：propose 后由 supervisor 自动批准（无真实资金风险）。
    LIVE_PENDING 模式：propose 后必须 approve() 才能执行，超时自动撤销。
    """

    def __init__(self, config: GuardConfig, safety: SafetyController):
        self.config = config
        self.safety = safety
        self._proposals: Dict[str, OrderProposal] = {}
        self._lock = threading.RLock()

    def create(self, **kwargs) -> OrderProposal:
        pid = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        prop = OrderProposal(proposal_id=pid, created_at=now, **kwargs)
        with self._lock:
            self._proposals[pid] = prop
        return prop

    def get(self, proposal_id: str) -> Optional[OrderProposal]:
        return self._proposals.get(proposal_id)

    def approve(self, proposal_id: str, operator: str) -> bool:
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if not prop:
                return False
            if prop.status not in ("pending",):
                logger.warning(f"委托 {proposal_id} 已决定({prop.status})，忽略重复审批")
                return False
            prop.status = "approved"
            prop.decided_at = datetime.now().isoformat()
            prop.decided_by = operator
            return True

    def reject(self, proposal_id: str, operator: str, reason: str = "") -> bool:
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if not prop:
                return False
            prop.status = "rejected"
            prop.reason = reason
            prop.decided_at = datetime.now().isoformat()
            prop.decided_by = operator
            return True

    def mark_executed(self, proposal_id: str, order_id: str):
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if prop:
                prop.status = "executed"
                prop.order_id = order_id

    def mark_blocked(self, prop: OrderProposal, reason: str):
        with self._lock:
            prop.status = "blocked"
            prop.reason = reason

    def reap_expired(self) -> int:
        """清理超时未确认的待审批委托（仅标记 expired，不删除，保留审计）"""
        now = datetime.now().timestamp()
        cnt = 0
        with self._lock:
            for prop in self._proposals.values():
                if prop.status == "pending":
                    created = datetime.fromisoformat(prop.created_at).timestamp()
                    if now - created > self.config.confirm_timeout_sec:
                        prop.status = "expired"
                        prop.reason = "确认超时自动撤销"
                        cnt += 1
        return cnt

    def pending_count(self) -> int:
        return sum(1 for p in self._proposals.values() if p.status == "pending")


# ======================================================================
# 高级仓位管理
# ======================================================================

class PositionSizer:
    """仓位规模计算器

    三种方法：
    - fixed_fractional：每笔预算 = 净值 × base_risk_fraction × 信号强度
    - volatility_target：按目标年化波动率缩放预算（vol 越高买越少）
    - kelly：半 Kelly 仓位，上限 kelly_cap
    """

    def __init__(self, config: GuardConfig,
                 vol_provider: Optional[callable] = None):
        self.config = config
        # vol_provider(code) -> 年化波动率(float) | None。仅当 sizing_method==
        # "volatility_target" 且未显式传入 annualized_vol 时调用（固定比例法下永不触发，
        # 零 IO，不改变既有行为）。
        self.vol_provider = vol_provider

    def size(
        self,
        side: OrderSide,
        price: float,
        total_equity: float,
        signal_strength: float = 1.0,
        annualized_vol: Optional[float] = None,
        win_prob: Optional[float] = None,
        payoff: Optional[float] = None,
        existing_value: float = 0.0,
        max_single_pct: Optional[float] = None,
        code: Optional[str] = None,
    ) -> Tuple[int, float]:
        """返回 (整手数量, 预算金额)

        Args:
            side: 买卖方向
            price: 委托价
            total_equity: 当前总资产
            signal_strength: 信号强度 0~1（融合多模型概率/评分）
            annualized_vol: 标的年化波动率（volatility_target 用）。为 None 且配置了
                vol_provider 时，自动按 code 估计 —— 把原本「死代码」的波动率目标法盘活。
            win_prob / payoff: Kelly 用（payoff=avg_win/avg_loss）
            existing_value: 该标的已有市值（用于加仓上限）
            max_single_pct: 单票上限（默认取配置）
            code: 标的代码（波动率目标法自动估 vol 时需要）
        """
        if price <= 0 or total_equity <= 0:
            return 0, 0.0

        max_single_pct = max_single_pct or self.config.max_single_position_pct
        strength = max(0.0, min(1.0, signal_strength))

        # 波动率目标法：优先用显式传入的 vol，否则尝试按 code 实时估计
        if self.config.sizing_method == "volatility_target":
            if annualized_vol is None and self.vol_provider is not None and code:
                try:
                    annualized_vol = self.vol_provider(code)
                except Exception:  # noqa: BLE001
                    annualized_vol = None
            if annualized_vol:
                # 波动率越高，预算越小。
                #
                # ⚠️ 原实现是 target/vol 再封顶 base_risk_fraction*2（=0.04）。
                # A 股个股年化波动率普遍 0.25~0.60，target/vol 恒大于 0.04，
                # 于是**每只票都吃满同一个上限**——既没有区分度，又等于把
                # 固定比例法的仓位直接翻倍。
                #
                # 现改由 sizing 按「组合目标 / (个股波动 × √N)」推导，
                # 高波动票真的买得少（详见该函数文档）。
                from src.trading.sizing import volatility_target_fraction

                frac = volatility_target_fraction(
                    self.config.volatility_target_pct,
                    annualized_vol,
                    n_positions=getattr(self.config, "max_positions", 1) or 1,
                    max_single_pct=max_single_pct,
                )
                if frac <= 0:  # 异常输入兜底，退回固定比例法
                    frac = self.config.base_risk_fraction
            else:
                # 未知波动率（无历史）→ 优雅降级为固定比例法，绝不用极小值 vol 撑爆仓位
                frac = self.config.base_risk_fraction
        elif self.config.sizing_method == "kelly" and win_prob and payoff:
            kelly = win_prob - (1 - win_prob) / max(payoff, 1e-6)
            frac = max(0.0, min(kelly * 0.5, self.config.kelly_cap))  # 半 Kelly + 封顶
        else:
            # 固定比例法
            frac = self.config.base_risk_fraction

        budget = total_equity * frac * (0.5 + 0.5 * strength)  # 强度弱则减半

        # 单票上限约束
        cap_value = total_equity * max_single_pct - existing_value
        budget = max(0.0, min(budget, cap_value))

        qty = int(budget / price / 100) * 100  # 整手
        return qty, budget


# ======================================================================
# 结构化成交日志
# ======================================================================

class TradeJournal:
    """append-only 交易日志（审计/合规用）

    每条记录一行 JSON：时间戳、事件类型、载荷。
    事件类型：signal / risk_block / proposal / approve / reject /
             order_sent / fill / halt / resume / arm / drawdown_breach
    """

    def __init__(self, config: GuardConfig):
        self.path = Path(config.journal_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, data: Optional[dict] = None):
        try:
            record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "event": event,
                "data": data or {},
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.error(f"交易日志写入失败({e})")

    def tail(self, n: int = 50) -> List[dict]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            out = []
            for ln in lines[-n:]:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
            return out
        except Exception:
            return []


# ======================================================================
# 风控编排中枢
# ======================================================================

class RiskSupervisor:
    """安全与风控编排中枢

    对外暴露精简 API：
        propose_trade(...)      → 生成预览（不下单）
        approve_and_execute(...) → 审批后下单（实盘受闸门+熔断约束）
        kill_switch(reason)     → 全局紧急停止
        tick_monitor()          → 周期性净值/回撤/亏损检查与熔断
        get_status()            → 给 UI 的状态快照
    """

    def __init__(
        self,
        broker: BrokerAPI,
        config: Optional[GuardConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        paper_account: Optional[PaperAccount] = None,
        industry_map: Optional[Dict[str, str]] = None,
        adv_provider: Optional[Callable[[str], Optional[float]]] = None,
        risk_monitor=None,
        turnover_file: Optional[Path] = None,
    ):
        # 报价校验收口：任何构造路径都必须过它，而不只是闭环那一条。
        # propose_trade 自己会 broker.get_tick() 且不接受外部传价，止损判定
        # 也在 executor 内部独立取价 —— 只修闭环等于没修。
        self.broker = wrap_broker(broker)
        self.sanity = getattr(self.broker, "guard", None)
        self.config = config or GuardConfig()
        self.paper_account = paper_account
        # 行业映射默认自动装载：不传就空表会让「行业集中度」风控静默失效
        self.industry_map = (
            industry_map if industry_map is not None else _load_industry_map()
        )

        # 流动性约束用的 ADV provider（code → 近20日平均成交额）。None = 不约束。
        # 默认只在非测试环境构建：读盘估计 ADV 会让用例依赖生产 parquet。
        if adv_provider is not None:
            self._adv_provider = adv_provider
        elif _running_under_pytest():
            self._adv_provider = None
        else:
            try:
                from src.trading.liquidity import AdvProvider
                _p = AdvProvider()
                self._adv_provider = _p.adv_amount
            except Exception as e:  # noqa: BLE001
                logger.warning(f"ADV provider 初始化失败（流动性约束将跳过）: {e}")
                self._adv_provider = None

        # 当日买入换手累计（用于 max_daily_turnover_pct）
        # ⚠️ 必须落盘：这是「当日已用预算」的记账，不是缓存。存在内存里的话，
        # 常驻进程每重启一次预算就归零 —— 一天重启三次就能买出三倍换手，
        # 而重启在生产里是常态（换代码、崩溃自愈、手工拉起）。
        self._turnover_day: str = ""
        self._turnover_amount: float = 0.0
        self._turnover_file = (
            Path(turnover_file) if turnover_file
            else (None if _running_under_pytest() else TURNOVER_STATE_PATH)
        )
        self._load_turnover()

        # 组合尾部风险监控（相关性 VaR）。None = 不约束（测试/数据缺失时降级）。
        if risk_monitor is not None:
            self._risk_monitor = risk_monitor
        elif _running_under_pytest():
            self._risk_monitor = None
        else:
            try:
                from src.trading.risk_model import PortfolioRiskMonitor
                self._risk_monitor = PortfolioRiskMonitor()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"组合风险模型初始化失败（VaR 约束将跳过）: {e}")
                self._risk_monitor = None

        # 复用现有逐单风控执行器（防御纵深内层）
        # risk_config 为 None 时按「本配置的止损段」构造，而不是用 RiskConfig 的
        # 硬编码默认值 —— 否则 trading_guard.yaml 里的止损参数永远不会生效。
        if risk_config is None:
            try:
                from src.trading.executor import RiskConfig
                risk_config = RiskConfig(
                    stop_loss_pct=float(self.config.stop_loss_pct),
                    stop_profit_pct=float(self.config.stop_profit_pct),
                    trailing_stop_pct=float(self.config.trailing_stop_pct),
                    atr_stop_enabled=bool(self.config.atr_stop_enabled),
                    atr_stop_k=float(self.config.atr_stop_k),
                    atr_trailing_k=float(self.config.atr_trailing_k),
                    atr_stop_min_pct=float(self.config.atr_stop_min_pct),
                    atr_stop_max_pct=float(self.config.atr_stop_max_pct),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"按配置构造 RiskConfig 失败，改用默认值: {e}")
                risk_config = None
        self.executor = TradeExecutor(self.broker, risk_config)
        self.risk_config = self.executor.config

        # 安全层组件（把 broker 注入进去：武装实盘前要能确认它是真实盘通道）
        self.safety = SafetyController(self.config, broker=self.broker)
        self.gate = OrderGate(self.config, self.safety)

        # 波动率目标法需要的「按 code 实时估 vol」provider（固定比例法下不会被调用）
        try:
            from src.trading.analytics import VolatilityEstimator
            self._vol_estimator = VolatilityEstimator()
            vol_provider = self._vol_estimator.annualized_vol
        except Exception:  # noqa: BLE001
            vol_provider = None
        self.sizer = PositionSizer(self.config, vol_provider=vol_provider)
        self.journal = TradeJournal(self.config)

        self._lock = threading.RLock()
        self._last_order_ts: float = 0.0
        self._orders_this_minute: List[float] = []
        self._on_event: Optional[Callable] = None  # UI 事件回调

        # 信号日志去重：{(code, side): (评分档, agreement, strength)}，跨日清空
        self._signal_dedupe: Dict[Tuple[str, str], Tuple] = {}
        self._signal_dedupe_day: str = ""

        # 市场状态感知：维护当前 regime 并对策略标签做动态加权（Feature #4）
        self.regime = RegimeWeighter(
            enabled=getattr(self.config, "regime_weighting_enabled", True))

        # 熔断自诊断：自愈后的「降权冷静期」截止时间戳（epoch 秒），0=无
        self._heal_throttle_until: float = 0.0

        # 波动率尾部风险自适应降仓（Feature #6）：滚动净值收益缓冲 + 当前阻尼系数
        self._equity_window: List[float] = []   # 仅存「收益率」，不混装净值水位
        self._last_equity: Optional[float] = None  # 上一 tick 净值，用于算收益率
        self._vol_control_factor: float = 1.0  # 1.0=不干预；<1.0=降仓

        # 市场状态粘性（Batch 3）：对原始 detect_regime 信号做 N 日确认，防横跳
        self._regime_smoother = RegimeSmoother(
            hold_days=getattr(self.config, "regime_hysteresis_days", 2),
            initial=self.regime.regime,
        )

        # 策略信誉自适应权重（Batch 3）：在线学习每标签 IC，喂给多策略集成器
        self._credibility = StrategyCredibility(
            halflife=getattr(self.config, "credibility_halflife", 20),
        )
        self._ic_weights: Dict[str, float] = {}   # 当前生效的 ic_weights
        self._cred_fed: set = set()               # 已喂给信誉引擎的成交 id（防重复）

        # 波动自适应熔断线（Batch 3）：基准近期日波动，由闭环每交易日推送
        self._market_vol: Optional[float] = None

        logger.info(
            f"🛡️ RiskSupervisor 初始化 | mode={self.safety.mode.value} "
            f"halted={self.safety.halted}"
        )

    # ---------- 事件回调 ----------
    def set_event_callback(self, cb: Callable[[str, dict], None]):
        self._on_event = cb

    def rebind_broker(self, broker) -> None:
        """把下单/取价通道整体切到另一个 broker（通常是报价代理）。

        为什么需要：:class:`~src.trading.loop.TradingLoop` 会在构造时才把原始
        券商包成 ``SanityBroker``，而外部（``run_loop.py`` / 看板）已经先建好了
        本中枢并注入了**原始** broker。不整体回灌的话，``propose_trade`` 与
        ``check_stop_conditions`` 仍走原始通道 —— 报价校验等于只覆盖了闭环估值
        那一条路，而止损恰恰是拿另外那条路的价在判。

        三处引用必须一起换（本类 / 内层 executor / 安全层），漏一个就留一个
        绕过校验的入口。
        """
        self.broker = broker
        # 顺手把「当前生效的报价守卫」指到代理上那份 —— 两处各留一份状态
        # 正是这个模块要防的事（A 判异常沿用旧价，B 没见过该价反手采信）。
        self.sanity = getattr(broker, "guard", getattr(self, "sanity", None))
        try:
            self.executor.broker = broker
        except Exception as e:  # noqa: BLE001
            logger.warning(f"executor 切通道失败（止损取价仍绕过校验）: {e}")
        try:
            if getattr(self, "safety", None) is not None:
                self.safety._broker = broker
        except Exception as e:  # noqa: BLE001
            logger.warning(f"安全层切通道失败（武装前券商校验仍看旧对象）: {e}")

    def _emit(self, event: str, data: dict):
        self.journal.log(event, data)
        if self._on_event:
            try:
                self._on_event(event, data)
            except Exception:
                pass

    # ---------- 信号日志（去重） ----------
    def _log_signal(
        self,
        code: str,
        name: str,
        side: OrderSide,
        ai_score: float,
        agreement: float,
        signal_strength: float,
    ) -> bool:
        """写 ``signal`` 事件，按「交易日 × 标的 × 方向 × 评分档」去重。

        背景：``propose_trade`` 在盘中每个 tick 都会对每只标的调用一次。早期实现
        在函数第一行**无条件**落盘 signal，于是 10-15 那天固定刷了 1440 条
        （每分钟 5 只 × 288 分钟），整个 journal 1659 条里 1642 条（99%）是同一条
        信号的重复。噪声把日志灌满后，真正的 fill / risk_block / halt 事件被淹没，
        反馈引擎按 ``signal`` 计数也完全失真。

        去重口径：同一交易日内，同一标的同一方向，评分落在同一个 5 分档
        （且 agreement / strength 未变）就只记一次。评分跨档、方向反转、
        换交易日都会重新落盘 —— 观测「信号在变」的能力保留，纯 tick 重复被去掉。

        Returns:
            True 表示本次真的落盘（未被去重）。
        """
        today = datetime.now().strftime("%Y-%m-%d")
        bucket = int(round(float(ai_score) / 5.0) * 5)
        sig = (bucket, round(float(agreement), 2), round(float(signal_strength), 2))
        key = (code, side.value if isinstance(side, OrderSide) else str(side))

        # propose_trade 本身不持锁（它要调券商/算仓位，不该长时间占用），
        # 但去重表是共享可变状态（看板线程 + 调度线程都可能进来），单独加锁。
        with self._lock:
            if self._signal_dedupe_day != today:
                self._signal_dedupe.clear()
                self._signal_dedupe_day = today
            if self._signal_dedupe.get(key) == sig:
                return False
            self._signal_dedupe[key] = sig

        self._emit("signal", {
            "code": code, "name": name, "side": key[1],
            "ai_score": ai_score, "agreement": agreement,
            "signal_strength": signal_strength,
        })
        return True

    # ---------- 账户/持仓公开访问（PAPER 走纸账户，LIVE 走券商） ----------
    def account(self) -> Optional[AccountInfo]:
        """当前账户快照（自动区分 PAPER 纸账户 / LIVE 券商账户）。"""
        return self._account()

    def positions(self) -> List[Position]:
        """当前持仓（自动区分 PAPER 纸账户 / LIVE 券商账户）。"""
        return self._positions()

    # ---------- 信号强度自适应（regime 加权 + 自愈降权 + 波动率降仓） ----------
    def _adjust_signal_strength(self, strategy_tag: str, raw: float,
                                is_exit: bool = False) -> float:
        """把「市场状态 / 熔断自愈冷静期 / 波动率尾部风险」折算进下单强度。

        - regime 加权：牛市放大进攻、熊市放大防御，由 ``self.regime`` 查表得到
          标签系数（风控类标签 stop 恒为 1.0，止损纪律不受市场状态影响）。
        - 自愈降权：若正处于熔断自愈后的「降权冷静期」，再乘一个折扣系数，
          实现「部分解冻」而非全仓位立刻放开。
        - 波动率降仓（Vol-Control）：仅对**新开仓**（``is_exit=False``）生效，
          当组合净值已实现波动率异常飙涨时压低强度；止损/风控离场永远是 1.0
          （恐慌时更该果断离场，绝不能因为波动大就卡住止损）。
        输出裁剪到 [0, 2.0]（允许 regime 适度放大，但不过冲）。
        """
        s = float(raw)
        if self.regime.enabled:
            s *= self.regime.weight(strategy_tag, 1.0)
        if self._heal_throttle_until and time.time() < self._heal_throttle_until:
            s *= float(self.config.auto_resume_risk_fraction)
        # Vol-Control：只压新开仓，不碰离场
        if not is_exit and self.config.vol_control_enabled and self._vol_control_factor < 1.0:
            s *= self._vol_control_factor
        return max(0.0, min(2.0, s))

    def _update_vol_control(self, equity: float) -> None:
        """滚动维护净值收益窗口，计算「波动率尾部风险」阻尼系数。

        实现（稳健、零分支依赖）：
        - 用 ``_last_equity`` 记录上一 tick 净值水位，每 tick 算出**这一笔**的
          tick 级收益率并追加进 ``_equity_window``（窗口只装收益率，不混装水位，
          避免首笔 0.0 种子导致后续永远算不出收益率的死窗口）。
        - 以窗口后半段的收益波动（std）为「当前波动」，以全窗口收益中位数
          （绝对值）为「常态波动」基线；当前波动 ≥ 基线 × ``vol_control_spike_multiple``
          即判定波动异常，把系数线性压到 ``vol_control_min_strength``；否则系数=1.0。
        - 数据不足（窗口未满）或计算异常 → 系数保持 1.0（fail-safe，不干预）。
        """
        try:
            import numpy as _np
            if equity <= 0:
                return
            # 记录收益率（首 tick 无前值，仅打底，不装窗口）
            if self._last_equity is not None and self._last_equity > 0:
                self._equity_window.append(
                    (equity - self._last_equity) / self._last_equity)
            self._last_equity = equity
            n = int(self.config.vol_control_window)
            if len(self._equity_window) > n:
                self._equity_window = self._equity_window[-n:]
            if len(self._equity_window) < max(5, n // 2):
                self._vol_control_factor = 1.0
                return
            arr = _np.asarray([float(x) for x in self._equity_window])
            cur = float(_np.std(arr[-max(2, n // 2):], ddof=0))
            base = max(abs(float(_np.median(arr))), 1e-6)
            if cur >= base * float(self.config.vol_control_spike_multiple):
                # 线性压到下限：偏离越多压得越狠，但保底 min_strength
                over = min((cur / base) / float(self.config.vol_control_spike_multiple), 3.0)
                factor = max(float(self.config.vol_control_min_strength),
                            1.0 - (over - 1.0) * 0.35)
                self._vol_control_factor = factor
            else:
                self._vol_control_factor = 1.0
        except Exception:  # noqa: BLE001  // 任何异常都不该中断主流程
            self._vol_control_factor = 1.0

    def set_regime(self, regime: Optional[str]) -> None:
        """由外部（交易闭环 / 调度）每 tick 推送当前市场状态。

        Batch 3：原始信号先经过 ``RegimeSmoother`` 做 N 日确认，避免基准均线在
        横盘/假突破时天天横跳导致策略权重反复翻转（whipsaw）。最终生效状态写入
        ``self.regime``（供 ``_adjust_signal_strength`` / ``fuse_signals`` 查表）。
        """
        smoothed = self._regime_smoother.update(regime)
        self.regime.update(smoothed)

    def set_market_vol(self, vol_pct: Optional[float]) -> None:
        """由闭环每交易日推送「基准近期日波动(%)」，供波动自适应熔断线使用。

        异常/缺失值一律置 None → ``_effective_daily_loss_pct`` 退化为静态配置。
        """
        try:
            self._market_vol = float(vol_pct) if vol_pct is not None else None
        except (TypeError, ValueError):
            self._market_vol = None

    def _effective_daily_loss_pct(self) -> float:
        """波动自适应单日亏损熔断线（Batch 3）。

        以 ``config.max_daily_loss_pct`` 为中枢，按「基准近期日波动 / 参考波动」
        等比缩放：平静市场波动小 → 线收紧（更敏感），动荡市场波动大 → 线放宽
        （抗噪声）。结果始终夹在 [floor, cap] 之间，且回撤熔断线(8%) 作为绝对
        底线始终独立生效。任何异常 / 未推送波动 → 退化为静态 ``max_daily_loss_pct``。
        """
        if not self.config.adaptive_kill_switch:
            return float(self.config.max_daily_loss_pct)
        mv = self._market_vol
        ref = self.config.kill_switch_ref_vol_pct
        if mv is None or mv <= 0 or ref <= 0:
            return float(self.config.max_daily_loss_pct)
        ratio = mv / float(ref)
        ratio = min(max(ratio, 0.7), 1.5)  # 缩放倍率限幅 ±50%
        eff = float(self.config.max_daily_loss_pct) * ratio
        return min(max(eff, self.config.kill_switch_floor_pct),
                   self.config.kill_switch_cap_pct)

    def observe_strategy_outcome(self, tag: str, entry_strength: float,
                                 realized_return: float) -> None:
        """向策略信誉引擎喂入一笔已平仓结果（Batch 3）。

        随后刷新 ``_ic_weights``，使 ``fuse_signals`` 的多策略集成随真实战绩调权。
        """
        if not self.config.credibility_enabled:
            return
        self._credibility.observe(tag, entry_strength, realized_return)
        self._ic_weights = self._credibility.weights()
        if self._credibility.is_dirty() and not _running_under_pytest():
            self._credibility.save_state(CRED_STATE_PATH)

    def sync_credibility_from_paper(self, paper) -> None:
        """从纸账户成交历史增量同步策略信誉（Batch 3）。

        配对每笔 sell 的 close_id 找到对应买入的入场强度(ai_score)，用实际盈亏
        喂养信誉引擎；已喂过的成交用 trade_id 去重。零 IO、异常静默吞掉。
        """
        if not self.config.credibility_enabled or paper is None:
            return
        try:
            trades = getattr(paper, "trade_history", None)
            if not trades:
                return
            by_id = {t.trade_id: t for t in trades
                     if getattr(t, "trade_id", None)}
            for t in trades:
                if getattr(t, "side", "") != "sell":
                    continue
                if not getattr(t, "pnl", 0):
                    continue
                tid = getattr(t, "close_id", "") or ""
                buy = by_id.get(tid)
                entry_strength = (float(getattr(buy, "ai_score", 0.0) or 0.0)
                                  if buy else 0.0)
                ret = float(getattr(t, "pnl_pct", 0.0) or 0.0) / 100.0
                tag = getattr(t, "strategy", "") or ""
                key = tid or (f"{getattr(t, 'code', '')}-"
                              f"{getattr(t, 'filled_at', '')}")
                if not key or key in self._cred_fed:
                    continue
                self._credibility.observe(tag, entry_strength, ret)
                self._cred_fed.add(key)
            self._ic_weights = self._credibility.weights()
            # 只有真的喂入新样本（dirty）才落盘，避免每 tick 空写；
            # 测试环境下不落盘（保持 sync 零 IO，避免污染生产 data/）
            if self._credibility.is_dirty() and not _running_under_pytest():
                self._credibility.save_state(CRED_STATE_PATH)
        except Exception:  # noqa: BLE001  // 学习失败绝不中断主流程
            pass

    def fuse_signals(self, scores: Dict[str, float]) -> Dict:
        """多策略信号集成（Feature #4 + #2 + Batch 3）：用当前市场状态权重与
        在线策略信誉权重融合各策略评分。

        ``scores`` 为 ``{策略标签: AI 评分(0~100)}``。返回一个含综合分、决策、
        各策略权重与贡献拆解的 dict，可直接用于下单前的信号融合或绩效归因。
        """
        from src.trading.ensemble import fuse_strategies
        return fuse_strategies(
            scores,
            regime_weights=self.regime.describe(),
            ic_weights=getattr(self, "_ic_weights", None) or {},
        )

    # ---------- 账户/持仓适配（PAPER 走纸账户，LIVE 走券商） ----------
    def _paper_quote_fallback(self, code: str,
                              tick: Optional[TickData]) -> Optional[TickData]:
        """PAPER 模式下用**公开行情**兜底券商 tick。

        事故（2026-10-20，静默失效）：闭环跑在 PAPER，但 propose_trade 的行情取自
        **真实券商** ``YuekaiBroker.get_tick()``（走 xtquant / QMT）。本机没装 QMT，
        该方法恒返回 None（"无法连接xtquant服务，请检查QMT…是否开启"）→
        每笔意图都被「无法获取行情或账户」拦掉。实测 2674 次拦截 / 仅 4 笔成交，
        纸面闭环等于从没真正交易过；而心跳、界面一切正常，完全看不出异常。

        纸交易本来就不该依赖真实券商链路：这里在 PAPER 下退回公开行情源
        （腾讯，实测可用）。LIVE / LIVE_PENDING **绝不兜底** —— 实盘拿不到券商
        行情就必须停手，那是正确的 fail-closed，不能拿公开行情去下真单。
        """
        if self.safety.mode != TradingMode.PAPER:
            return tick
        try:
            from src.data.fetcher import fetcher
            q = fetcher.get_realtime_quote(code) or {}
            price = float(q.get("最新价") or 0)
            if price <= 0:
                return tick
            logger.info(
                f"📡 PAPER 行情兜底：{code} 券商 tick 不可用，改用公开行情 "
                f"¥{price:.2f}（{q.get('名称') or ''}）")
            return TickData(
                code=code,
                name=str(q.get("名称") or ""),
                last_price=price,
                open=float(q.get("今开") or 0),
                high=float(q.get("最高") or 0),
                low=float(q.get("最低") or 0),
                pre_close=float(q.get("昨收") or 0),
                volume=int(float(q.get("成交量") or 0)),
                amount=float(q.get("成交额") or 0),
                change_pct=float(q.get("涨跌幅") or 0),
                time=datetime.now(),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"PAPER 行情兜底失败 {code}: {e}")
            return tick

    def _account(self) -> Optional[AccountInfo]:
        if self.safety.mode == TradingMode.PAPER and self.paper_account:
            pa = self.paper_account
            mv = sum(h.get("market_value", 0.0) for h in pa.holdings.values())
            return AccountInfo(
                account_id="PAPER",
                total_asset=pa.get_total_asset(),
                available_cash=pa.cash,
                frozen_cash=pa.frozen_cash,
                market_value=mv,
                total_pnl=pa.get_total_pnl(),
                today_pnl=pa.today_pnl,
            )
        return self.broker.query_account()

    def _positions(self) -> List[Position]:
        if self.safety.mode == TradingMode.PAPER and self.paper_account:
            out = []
            for h in self.paper_account.get_positions():
                out.append(Position(
                    code=h["code"], name=h.get("name", h["code"]),
                    quantity=h["quantity"], available=h["available"],
                    avg_cost=h["avg_cost"], current_price=h.get("last_price", h["avg_cost"]),
                    market_value=h["market_value"],
                    pnl=h.get("pnl", 0.0), pnl_pct=h.get("pnl_pct", 0.0),
                    today_quantity=h.get("today_quantity", 0),
                ))
            return out
        return self.broker.query_positions()

    # ---------- 预览（不下单） ----------
    def propose_trade(
        self,
        code: str,
        name: str,
        side: OrderSide,
        ai_score: float,
        agreement: float = 1.0,
        signal_strength: float = 1.0,
        strategy_tag: str = "ai",
        annualized_vol: Optional[float] = None,
        win_prob: Optional[float] = None,
        payoff: Optional[float] = None,
        risk_exit: bool = False,
        explicit_quantity: Optional[int] = None,
    ) -> OrderProposal:
        """生成一笔交易预览（不实际下单）

        Args:
            risk_exit: 风控性离场（止盈止损 / 调仓清仓）。为 True 时，
                卖出不再受「AI评分仍高不卖」的过滤约束 —— 止损纪律优先于
                模型观点。涨跌停 / 停牌 / T+1 等硬规则仍然生效。
            explicit_quantity: 由交易计划指定的目标股数。提供时跳过通用
                仓位算法（固定比例/波动率/凯利），仍受硬约束封顶：
                单票上限、可卖量、可用现金。用于「计划定数量、guard 定准否」。

        Returns:
            OrderProposal，status 可能为 blocked（含拒绝原因）或 pending
        """
        self._log_signal(code, name, side, ai_score, agreement, signal_strength)

        # 0. 市场状态动态加权 + 熔断自愈降权 + 波动率降仓（统一在入口缩放
        #    signal_strength，后续预算公式 budget = equity*frac*(0.5+0.5*strength) 自动跟随）
        signal_strength = self._adjust_signal_strength(
            strategy_tag, signal_strength, is_exit=risk_exit)

        # 1. 全局开关
        if not self.safety.can_trade():
            # 风控性离场（止损/止盈，risk_exit=True）在「自动熔断」场景下仍放行：
            # 自动熔断（operator 以 "auto(" 开头，如 auto(dailyloss)/auto(drawdown)）
            # 只应冻结「新开仓/调仓」，绝不能冻结「斩仓」——否则会像实测那样把
            # 亏损越拖越大（曾出现 32% 回撤）。人工 KILL SWITCH（user/dashboard/
            # system）仍冻结一切，那是操作者明确的「全部停止」意图，必须尊重。
            if risk_exit and str(self.safety._state.halted_by or "").startswith("auto("):
                pass  # 放行，继续走常规 propose 流程
            else:
                reason = "KILL SWITCH 已触发，禁止交易" if self.safety.halted else "模式为 OFF，禁止交易"
                prop = self.gate.create(
                    code=code, name=name, side=side.value, price=0.0,
                    quantity=0, est_amount=0.0, ai_score=ai_score,
                    agreement=agreement, signal_strength=signal_strength,
                    strategy_tag=strategy_tag, risk_exit=risk_exit,
                )
                self.gate.mark_blocked(prop, reason)
                self._emit("risk_block", {"code": code, "reason": reason})
                return prop

        # 2. 交易时段（实盘必须有日历约束）
        if self.safety.is_live():
            try:
                from src.trading.calendar import order_allowed
                ok, why = order_allowed()
                if not ok:
                    prop = self.gate.create(
                        code=code, name=name, side=side.value, price=0.0,
                        quantity=0, est_amount=0.0, ai_score=ai_score,
                        agreement=agreement, signal_strength=signal_strength,
                        strategy_tag=strategy_tag, risk_exit=risk_exit,
                    )
                    self.gate.mark_blocked(prop, f"非可交易时段: {why}")
                    self._emit("risk_block", {"code": code, "reason": why})
                    return prop
            except Exception:  # noqa: BLE001
                pass

        # 3. 取行情/账户
        tick = self.broker.get_tick(code)
        # PAPER 下券商行情拿不到时（如本机没开 QMT）用公开行情兜底，见
        # _paper_quote_fallback 的说明。LIVE 绝不兜底：实盘拿不到券商行情就该停手。
        if tick is None or getattr(tick, "last_price", 0) <= 0:
            tick = self._paper_quote_fallback(code, tick)
        account = self._account()
        if not tick or tick.last_price <= 0 or not account:
            reason = "无法获取行情或账户"
            prop = self.gate.create(
                code=code, name=name, side=side.value, price=0.0,
                quantity=0, est_amount=0.0, ai_score=ai_score,
                agreement=agreement, signal_strength=signal_strength,
                strategy_tag=strategy_tag, risk_exit=risk_exit,
            )
            self.gate.mark_blocked(prop, reason)
            self._emit("risk_block", {"code": code, "reason": reason})
            return prop

        price = tick.last_price
        positions = {p.code: p for p in self._positions()}

        # 3.5 信号级预检（评分/一致性/涨跌停/T+1），纸交易也走这道关
        ok_sig, why_sig = self._precheck_signal(
            code, name, side, ai_score, agreement, tick, risk_exit=risk_exit)
        if not ok_sig:
            prop = self.gate.create(
                code=code, name=name, side=side.value, price=price,
                quantity=0, est_amount=0.0, ai_score=ai_score,
                agreement=agreement, signal_strength=signal_strength,
                strategy_tag=strategy_tag, risk_exit=risk_exit,
            )
            self.gate.mark_blocked(prop, why_sig)
            self._emit("risk_block", {"code": code, "reason": why_sig})
            return prop

        # 4. 组合级约束（持仓数 / 单票 / 行业）
        block_reason = self._check_portfolio_limits(code, side, account, positions)
        if block_reason:
            prop = self.gate.create(
                code=code, name=name, side=side.value, price=price,
                quantity=0, est_amount=0.0, ai_score=ai_score,
                agreement=agreement, signal_strength=signal_strength,
                strategy_tag=strategy_tag, risk_exit=risk_exit,
            )
            self.gate.mark_blocked(prop, block_reason)
            self._emit("risk_block", {"code": code, "reason": block_reason})
            return prop

        # 4.5 组合尾部风险：相关性 VaR 超预算 → 连续降新开仓强度（不清仓）
        var_scale, var_detail = 1.0, {}
        if side == OrderSide.BUY:
            var_scale, var_detail = self._portfolio_var_scale(account, positions)
            if var_scale < 1.0:
                signal_strength = max(0.05, signal_strength * var_scale)
                logger.info(
                    f"[VaR约束] 组合VaR {var_detail.get('var_pct')} > "
                    f"{self.config.max_portfolio_var_pct}（相关性{var_detail.get('avg_corr')}），"
                    f"新开仓强度 ×{var_scale:.2f}"
                )

        # 5. 仓位规模
        #    · 交易计划给了明确股数 → 用计划的（计划定数量）
        #    · 否则走通用仓位算法（固定比例 / 波动率目标 / 半 Kelly）
        existing_value = positions[code].market_value if code in positions else 0.0
        # 清仓式卖出允许零股（A 股：卖出可不足一手，但须一次性卖光）。
        # 只在「请求量 >= 可卖全量」时放开，部分卖出仍整手，否则实盘废单。
        full_sellable = 0
        if side == OrderSide.SELL and code in positions:
            full_sellable = sellable_quantity(positions[code])
        allow_odd_lot = (
            side == OrderSide.SELL and full_sellable > 0
            and explicit_quantity is not None
            and int(explicit_quantity) >= full_sellable
        )
        if explicit_quantity is not None and explicit_quantity > 0:
            # 计划数量同样要跟随 VaR 缩放，否则「计划定数量」会绕过尾部风险约束
            if var_scale < 1.0:
                explicit_quantity = max(0, int(int(explicit_quantity) * var_scale))
                # 被 VaR 缩过就不再是「清仓」了，零股豁免随之失效
                if int(explicit_quantity) < full_sellable:
                    allow_odd_lot = False
            qty, budget = self._size_explicit(
                code, side, price, int(explicit_quantity),
                account, positions, existing_value,
            )
        else:
            qty, budget = self.sizer.size(
                side, price, account.total_asset, signal_strength,
                annualized_vol=annualized_vol, win_prob=win_prob, payoff=payoff,
                existing_value=existing_value,
                max_single_pct=self.config.max_single_position_pct,
                code=code,
            )
        # 5.5 执行约束：流动性（ADV参与率）/ 单笔风险预算 / 当日换手预算
        qty, limit_why = self._apply_execution_limits(
            code, side, price, qty, account, allow_odd_lot=allow_odd_lot)
        if limit_why:
            logger.info(f"[执行约束] {name or code} {side.value}: {limit_why}")
        min_qty = 1 if allow_odd_lot else 100
        if qty < min_qty:
            reason = limit_why or f"计算下单量不足1手 ({qty}股)"
            prop = self.gate.create(
                code=code, name=name, side=side.value, price=price,
                quantity=0, est_amount=0.0, ai_score=ai_score,
                agreement=agreement, signal_strength=signal_strength,
                strategy_tag=strategy_tag, risk_exit=risk_exit,
            )
            self.gate.mark_blocked(prop, reason)
            self._emit("risk_block", {
                "code": code, "side": side.value, "reason": reason,
            })
            return prop

        # 5.6 组合约束复核（带本次成交额）
        # 第 4 步只能看「已有持仓」：那时 est_amount 还是 0，行业检查退化成
        # 「已占多少」。于是「行业已占 29%、上限 30%」时，一笔把它顶到 50% 的
        # 大单照样放行 —— 单票上限管的是单只，不是行业。这里用真实下单额再核一次。
        if side == OrderSide.BUY and qty > 0:
            block2 = self._check_portfolio_limits(
                code, side, account, positions, delta_value=qty * price)
            if block2:
                prop = self.gate.create(
                    code=code, name=name, side=side.value, price=price,
                    quantity=0, est_amount=0.0, ai_score=ai_score,
                    agreement=agreement, signal_strength=signal_strength,
                    strategy_tag=strategy_tag, risk_exit=risk_exit,
                )
                self.gate.mark_blocked(prop, block2)
                self._emit("risk_block", {"code": code, "reason": block2})
                return prop

        prop = self.gate.create(
            code=code, name=name, side=side.value, price=price,
            quantity=qty, est_amount=qty * price, ai_score=ai_score,
            agreement=agreement, signal_strength=signal_strength,
            strategy_tag=strategy_tag, risk_exit=risk_exit,
            raw={"budget": round(budget, 2)},
        )
        self._emit("proposal", {
            "proposal_id": prop.proposal_id, "code": code, "side": side.value,
            "qty": qty, "price": price, "est_amount": round(qty * price, 2),
        })

        # 纸交易 / 关闭确认：自动批准
        if not self.safety.requires_per_order_confirmation():
            self.gate.approve(prop.proposal_id, operator="auto(paper/off)")
        return prop

    def _size_explicit(
        self, code: str, side: OrderSide, price: float, want_qty: int,
        account: AccountInfo, positions: Dict[str, Position],
        existing_value: float, lot: int = 100,
    ) -> Tuple[int, float]:
        """按计划给定量级定单，但仍受硬约束封顶。

        硬约束（不可被计划突破）：
          · 卖出 ≤ T+1 可卖量
          · 买入 ≤ 单票上限剩余额度（max_single_position_pct - 已持市值）
          · 买入 ≤ 可用现金
          · 买入一律整手；卖出**清仓时**允许零股（A 股规则），部分卖出仍整手
        """
        if price <= 0 or want_qty <= 0:
            return 0, 0.0

        if side == OrderSide.SELL:
            pos = positions.get(code)
            full = sellable_quantity(pos) if pos else 0
            # 请求量 >= 可卖全量 = 清仓：零股合法，不能取整
            if full > 0 and int(want_qty) >= full:
                qty = int(full)
            else:
                qty = (int(want_qty) // lot) * lot
                if qty <= 0:
                    return 0, 0.0
            return qty, qty * price

        qty = (int(want_qty) // lot) * lot
        if qty <= 0:
            return 0, 0.0
        cap_value = (account.total_asset * self.config.max_single_position_pct
                     - existing_value)
        max_qty = int(max(0.0, cap_value) / price / lot) * lot
        affordable = int(max(0.0, account.available_cash) / price / lot) * lot
        qty = min(qty, max_qty, affordable)
        qty = max(0, (qty // lot) * lot)
        return qty, qty * price

    def _precheck_signal(
        self, code: str, name: str, side: OrderSide,
        ai_score: float, agreement: float, tick,
        risk_exit: bool = False,
    ) -> Tuple[bool, str]:
        """信号级预检：复用 executor 的同套规则（AI评分/一致性/涨跌停/T+1）

        PAPER 模式下 executor 不会被调用（走纸账户），所以这里单独把关，
        避免纸交易绕过评分过滤下垃圾单。LIVE 模式执行时 executor 会再查一遍。

        risk_exit=True（止盈止损/调仓清仓）时跳过「AI评分仍高不卖」——
        纪律性离场不该被模型观点否决；其余硬规则（涨跌停/停牌/T+1）照旧。
        """
        rc = self.risk_config
        if side == OrderSide.BUY and ai_score < rc.min_ai_score_buy:
            return False, f"AI评分不足: {ai_score:.0f} < {rc.min_ai_score_buy}"
        if (side == OrderSide.SELL and not risk_exit
                and ai_score > rc.max_ai_score_sell):
            return False, f"AI评分仍高: {ai_score:.0f} > {rc.max_ai_score_sell}"
        if rc.require_agreement and agreement < 0.8:
            return False, f"模型分歧较大(一致性={agreement:.2f})"
        if rc.enable_market_rules and tick is not None:
            pre_close = getattr(tick, "pre_close", 0) or 0
            vol = getattr(tick, "volume", None)
            if side == OrderSide.BUY:
                ok, why = check_buyable(code, name, tick.last_price, pre_close, vol)
            else:
                ok, why = check_sellable(code, name, tick.last_price, pre_close, vol)
            if not ok:
                return False, why
        return True, ""

    def _check_portfolio_limits(
        self, code: str, side: OrderSide, account: AccountInfo,
        positions: Dict[str, Position], delta_value: float = 0.0,
    ) -> str:
        """返回 "" 表示通过；否则返回拦截原因。

        ``delta_value``：本次委托的预估成交额。第 4 步预检时还不知道下单量
        （sizing 在后面），只能拿 0 去比「已有持仓」；于是「行业已占 29%、
        上限 30%」时一笔能把它顶到 50% 的大单照样放行 —— 事后复核必须把这笔
        算进去，见 ``propose_trade`` 的 5.6。
        """
        if side == OrderSide.BUY:
            # 持仓只数
            active = [p for p in positions.values() if p.quantity > 0]
            # 只数上限只约束「是否新增一只有名有姓的新票」；给已有持仓加仓
            # 并不会增加只数，若一并拦掉会让调仓/加仓永远无法把权重加到目标。
            if (len(active) >= self.config.max_positions
                    and code not in {p.code for p in active}):
                return f"持仓只数已达上限 {self.config.max_positions}"
            # 行业集中度
            # 未知行业归入**同一个**「未知」桶，而不是跳过检查。
            # 旧实现 ``sector = industry_map.get(code, "")`` 后 ``if sector:``
            # —— 映射缺失就直接放行（fail-open）。实测 2026-10-26：
            # industry_map 2616 条只覆盖部分市场，当日 4 只选股里 603688 缺、
            # 2 只持仓里 300037 也缺 ⇒ 这两笔买入完全不受行业集中度约束，
            # 而「我不知道它是什么行业」恰恰应该按最保守的假设处理。
            sector = self.industry_map.get(code, "") or UNKNOWN_SECTOR
            if sector == UNKNOWN_SECTOR:
                self._warn_missing_sector([code])
            sector_val = sum(
                p.market_value for p in active
                if (self.industry_map.get(p.code, "") or UNKNOWN_SECTOR) == sector
            )
            # 估算新增：delta_value = 本次委托金额（预检时=0，复核时=实际下单额）
            cap = account.total_asset * self.config.max_sector_pct
            if sector_val + max(0.0, float(delta_value or 0.0)) >= cap:
                label = "未知行业(映射缺失)" if sector == UNKNOWN_SECTOR else f"行业 {sector}"
                return f"{label} 集中度已达上限 {self.config.max_sector_pct*100:.0f}%"
        return ""

    def _warn_missing_sector(self, codes: List[str]) -> None:
        """行业映射缺失一次性告警（同一代码只报一次，避免每 tick 刷日志）。"""
        try:
            bucket = getattr(self, "_missing_sector_warned", None)
            if bucket is None:
                bucket = set()
                self._missing_sector_warned = bucket
        except Exception:  # noqa: BLE001
            return
        fresh = [c for c in codes if c and c not in bucket]
        if not fresh:
            return
        bucket.update(fresh)
        logger.warning(
            f"⚠️ 行业映射缺失 {len(fresh)} 只（{', '.join(fresh[:8])}）→ 已归入"
            f"「未知行业」桶参与集中度检查。补全请跑 scripts/build_industry_map.py"
        )

    def _portfolio_var_scale(self, account, positions) -> Tuple[float, dict]:
        """按持仓相关性估组合 1 日 VaR，超预算则给出新开仓缩放系数。

        返回 ``(scale, detail)``；``scale`` ∈ [var_scale_floor, 1.0]。
        **只影响新开仓强度，绝不清仓、不影响止损离场。**
        """
        cfg = self.config
        if not getattr(cfg, "var_monitor_enabled", True) or self._risk_monitor is None:
            return 1.0, {}
        if not account or float(getattr(account, "total_asset", 0) or 0) <= 0:
            return 1.0, {}

        codes: List[str] = []
        values: List[float] = []
        for c, p in (positions or {}).items():
            if int(getattr(p, "quantity", 0) or 0) <= 0:
                continue
            v = float(getattr(p, "market_value", 0) or 0)
            if v <= 0:
                v = float(getattr(p, "quantity", 0)) * float(getattr(p, "current_price", 0) or 0)
            if v > 0:
                codes.append(c)
                values.append(v)
        if len(codes) == 0:
            return 1.0, {}
        total = sum(values)
        if total <= 0:
            return 1.0, {}
        # 权重口径：市值 / **总资产**（不是 / 持仓合计）。低仓位本身就该是低风险，
        # 归一化到持仓合计会把「只用 18% 仓位」误算成满仓风险，导致过度降仓。
        equity = float(getattr(account, "total_asset", 0) or 0)
        if equity <= 0:
            return 1.0, {}
        weights = [v / equity for v in values]

        # 单票满仓时也要算：这是最集中的状态，恰恰不能因为「凑不出协方差矩阵」
        # 就跳过风险检查（旧实现 len<2 直接 return 1.0 = 静默不约束）。
        if len(codes) == 1:
            try:
                var_pct = self._risk_monitor.var_pct_single(
                    codes[0], weights[0], z=float(cfg.var_confidence_z))
            except Exception as e:  # noqa: BLE001
                logger.debug(f"单标的 VaR 计算失败（已忽略）: {e}")
                return 1.0, {}
            if var_pct is None:
                return 1.0, {}
            detail = {"var_pct": round(float(var_pct), 4), "avg_corr": 0.0, "n": 1}
            cap = float(cfg.max_portfolio_var_pct)
            if cap <= 0 or var_pct <= cap:
                return 1.0, detail
            scale = min(1.0, max(float(cfg.var_scale_floor), cap / float(var_pct)))
            detail["scale"] = round(scale, 3)
            return scale, detail

        try:
            res = self._risk_monitor.var_pct(
                codes, weights, z=float(cfg.var_confidence_z))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"VaR 计算失败（已忽略）: {e}")
            return 1.0, {}
        if not res:
            return 1.0, {}

        var_pct, avg_corr = float(res[0]), float(res[1])
        detail = {
            "var_pct": round(var_pct, 4),
            "avg_corr": round(avg_corr, 3),
            "n": len(codes),
        }
        cap = float(cfg.max_portfolio_var_pct)
        if cap <= 0 or var_pct <= cap:
            return 1.0, detail

        scale = min(1.0, max(float(cfg.var_scale_floor), cap / var_pct))
        detail["scale"] = round(scale, 3)
        return scale, detail

    def _load_turnover(self) -> None:
        """从磁盘恢复「当日已用换手」。失败/跨日/文件缺失都等价于从 0 开始。"""
        if not self._turnover_file:
            return
        try:
            if not self._turnover_file.exists():
                return
            data = json.loads(self._turnover_file.read_text(encoding="utf-8"))
            day = str(data.get("day") or "")
            amt = float(data.get("amount") or 0)
            today = datetime.now().strftime("%Y-%m-%d")
            if day == today and amt > 0:
                self._turnover_day = today
                self._turnover_amount = amt
                logger.info(f"📒 当日换手累计已恢复: ¥{amt:,.0f}（{today}）")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"换手状态读取失败（从 0 计）: {e}")

    def _save_turnover(self) -> None:
        """原子落盘当日换手累计。失败只告警 —— 记账坏了不能拖垮交易主流程。"""
        if not self._turnover_file:
            return
        try:
            self._turnover_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._turnover_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "day": self._turnover_day,
                "amount": round(float(self._turnover_amount), 2),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._turnover_file)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"换手状态落盘失败（下次从头累计）: {e}")

    def _bump_turnover(self, amount: float) -> None:
        """累计当日买入金额（跨自然日自动清零），供换手预算约束使用。"""
        try:
            amt = float(amount)
        except (TypeError, ValueError):
            return
        if amt <= 0:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            if self._turnover_day != today:
                self._turnover_day = today
                self._turnover_amount = 0.0
            self._turnover_amount += amt
            self._save_turnover()

    def _apply_execution_limits(
        self, code: str, side: OrderSide, price: float, qty: int,
        account: Optional[AccountInfo], allow_odd_lot: bool = False,
    ) -> Tuple[int, str]:
        """对已算好的下单量再施加三道执行约束，返回 ``(最终股数, 约束说明)``。

        1) **流动性**：单笔金额 ≤ 20 日 ADV × ``max_participation_rate``
           （买卖同限 —— 卖出同样不能一口气砸穿盘口）；
        2) **风险预算**：单笔潜在亏损（股数 × 价 × 止损距离）≤ 总资产 ×
           ``max_risk_per_trade_pct``。仓位比例相等 ≠ 风险相等，高波动个股
           在同样 5% 止损下亏得更多，这里让每笔交易的风险敞口大致相等；
        3) **换手率**：当日买入累计金额 ≤ 总资产 × ``max_daily_turnover_pct``，
           抑制「每天为微小偏差来回翻仓」的成本漏损。

        ADV 拿不到时只跳过第 1 条（不阻断交易）；任一条把数量压到 0 时返回原因，
        供 propose 明确记录「为什么没下」。

        ``allow_odd_lot``：仅对**清仓式卖出**为 True —— A 股卖出允许不足一手，
        但必须一次性把零股全部卖掉。买入永远整手，别把这个开关用到买侧。
        """
        cfg = self.config
        if not getattr(cfg, "execution_limits_enabled", True):
            return int(qty), ""
        if qty <= 0 or price <= 0:
            return int(qty), ""

        reasons: List[str] = []
        lot = 100
        odd_ok = bool(allow_odd_lot) and side == OrderSide.SELL
        if not odd_ok:
            qty = int(qty // lot * lot)

        # 1) 流动性约束
        adv = None
        if self._adv_provider is not None:
            try:
                adv = self._adv_provider(code)
            except Exception:  # noqa: BLE001
                adv = None
        if adv and adv > 0:
            max_val = float(adv) * max(0.0, float(cfg.max_participation_rate))
            cap_qty = int(max_val / price // lot * lot)
            if cap_qty < qty:
                reasons.append(
                    f"流动性：单笔≤ADV×{cfg.max_participation_rate:.1%}"
                    f"（ADV≈¥{adv:,.0f} → {cap_qty}股）"
                )
                qty = cap_qty

        # 2) 风险预算（仅买入：卖出是减风险，不该被预算卡住）
        if side == OrderSide.BUY and account and account.total_asset > 0:
            stop_pct = max(float(cfg.risk_budget_stop_pct), 1e-4)
            # 止损距离必须取**实际生效**的那条线（executor 里 ATR 会把 -5% 放宽到
            # 最多 -15%）。用静态 5% 算预算而实际按 15% 止损，等于单笔风险超标 3 倍，
            # 风险预算形同虚设 —— 这是「开了 ATR 之后」才会暴露的口径不一致。
            try:
                _line, _ = self.executor._effective_stop_lines(code)  # 负值，如 -12.5
                stop_pct = max(stop_pct, abs(float(_line)) / 100.0)
            except Exception:  # noqa: BLE001
                pass
            # ★ 再放宽到「止损可能到达的上限」：ATR 自适应止损**只增不减**，建仓时
            # 按 -5% 放行的仓位，后续会被放宽到 -15%，实际风险翻 3 倍。用当前宽度
            # 算预算 = 预算只能保证「今天合规」，保证不了「这笔交易最终会亏多少」。
            # 实测 300757：建仓按 5% 止损距放行 ¥41 万，之后止损放宽到 -8.4%，
            # 实亏 ¥3.78 万 = 权益 0.81%，是 0.5% 预算的 1.6 倍。
            if (bool(getattr(cfg, "risk_budget_use_stop_ceiling", True))
                    and bool(getattr(cfg, "atr_stop_enabled", False))):
                stop_pct = max(stop_pct, float(getattr(cfg, "atr_stop_max_pct", 15.0)) / 100.0)
            risk_budget = float(account.total_asset) * float(cfg.max_risk_per_trade_pct)
            # ⚠️ 预算必须扣掉**同标的已有持仓**的风险敞口。本方法按「单笔」算上限，
            # 分多笔建仓时每笔单独看都合规、合起来却成倍超标 —— 预算被「分批」掏空。
            # 实测 300757 用 3 笔建到 800 股：200股×¥397×止损15% ≈ ¥1.2万，每笔都在
            # ¥2.35万预算内，但合计敞口 ¥4.76万 = 净值 1.03%，是 0.5% 预算的 2 倍。
            # 只对买入生效；不持有该标的时 held_risk=0，行为与改动前逐字节一致。
            held_risk = 0.0
            try:
                for _p in (self._positions() or []):
                    _pq = int(getattr(_p, "quantity", 0) or 0)
                    if str(getattr(_p, "code", "")) != code or _pq <= 0:
                        continue
                    _px = float(getattr(_p, "avg_cost", 0) or 0) or price
                    held_risk = _pq * _px * stop_pct
                    break
            except Exception:  # noqa: BLE001
                held_risk = 0.0
            remain = max(0.0, risk_budget - held_risk)
            cap_qty = int(remain / (price * stop_pct) // lot * lot)
            if cap_qty < qty:
                reasons.append(
                    f"风险预算：单笔亏损≤{cfg.max_risk_per_trade_pct:.2%}净值"
                    f"（止损{stop_pct:.0%}"
                    + (f"，含已有持仓风险¥{held_risk:,.0f}" if held_risk > 0 else "")
                    + f" → {cap_qty}股）"
                )
                qty = cap_qty

        # 3) 当日换手预算（仅买入）
        if side == OrderSide.BUY and account and account.total_asset > 0:
            today = datetime.now().strftime("%Y-%m-%d")
            with self._lock:
                if self._turnover_day != today:
                    self._turnover_day = today
                    self._turnover_amount = 0.0
                used = self._turnover_amount
            budget = float(account.total_asset) * float(cfg.max_daily_turnover_pct) - used
            if budget <= 0:
                return 0, f"当日换手预算已用尽（已买 ¥{used:,.0f}）"
            cap_qty = int(max(budget, 0.0) / price // lot * lot)
            if cap_qty < qty:
                reasons.append(f"当日换手预算：剩余 ¥{budget:,.0f} → {cap_qty}股")
                qty = cap_qty

        return max(0, int(qty)), "；".join(reasons)

    # ---------- 频率限制 ----------
    def _check_rate_limit(self) -> str:
        """取得一个下单时间片：**短间隔等待**，每分钟硬上限才拒绝。

        旧实现把「两笔间隔不足」直接当拒绝原因返回。可闭环一次 tick 会连着下
        多笔（一次调仓最多 10 笔），于是同一 tick 里只有第一笔能成交、其余 9 笔
        静默变成 blocked —— 计划看起来"执行了"，实际只成交 1/10，下一轮 gap
        重算又会重复同一幕，表现为「调仓永远补不到位」。

        间隔是**节奏**约束（避免打爆柜台/被限流），不该退化成**丢弃**：这里改成
        等到间隔满足再放行，最多等 ``min(interval, 5s)``；每分钟笔数上限仍然
        硬拒绝（那是真正的容量约束）。
        """
        interval = max(0.0, float(self.config.min_order_interval_sec))
        deadline = time.time() + min(interval, 5.0)
        while True:
            now = time.time()
            with self._lock:
                # 每分钟上限
                self._orders_this_minute = [t for t in self._orders_this_minute if now - t < 60]
                if len(self._orders_this_minute) >= self.config.max_orders_per_minute:
                    return f"每分钟委托数已达上限 {self.config.max_orders_per_minute}"
                gap = now - self._last_order_ts
            if gap >= interval or now >= deadline:
                return ""
            time.sleep(min(interval - gap, 0.2))

    # ---------- 审批 + 执行 ----------
    def approve_and_execute(
        self, proposal_id: str, operator: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Order]]:
        """审批并执行一笔委托

        - 纸交易：propose 时已自动批准，可直接执行（记纸账户）
        - LIVE_PENDING：必须已 approve() 才能执行
        - 任何模式：执行前再跑一遍熔断检查
        """
        prop = self.gate.get(proposal_id)
        if not prop:
            return False, "委托不存在", None

        # 实盘必须人工批准
        if self.safety.requires_per_order_confirmation() and prop.status != "approved":
            return False, f"委托未获人工批准 (status={prop.status})", None
        if prop.status in ("rejected", "expired", "blocked", "executed"):
            return False, prop.reason or f"委托已 {prop.status}，无法执行", None

        # 执行前再熔断一次（防御纵深）
        if self.safety.halted:
            # 风控性离场（止损/止盈，risk_exit=True）在「自动熔断」场景下仍放行：
            # 自动熔断只应冻结新开仓/调仓，不能冻结斩仓（实测曾因此出现 32% 回撤）。
            # 人工 KILL SWITCH 仍冻结一切。
            if not (prop.risk_exit and str(self.safety._state.halted_by or "").startswith("auto(")):
                return False, "KILL SWITCH 已触发，禁止执行", None

        rate_reason = self._check_rate_limit()
        if rate_reason:
            return False, rate_reason, None

        side = OrderSide(prop.side)
        code, name = prop.code, prop.name

        if self.safety.mode == TradingMode.PAPER and self.paper_account:
            # 防御：执行前再跑一次信号预检（避免 propose→execute 之间状态变化）
            tick = self.broker.get_tick(code)
            # 与 propose 阶段一致：PAPER 下券商行情不可用（本机没开 QMT 时恒如此）
            # 要退回公开行情，否则 tick=None 会让「停牌/涨跌停」规则整段被跳过
            # （fail-open），等于纸交易可以买停牌股。
            if tick is None or getattr(tick, "last_price", 0) <= 0:
                tick = self._paper_quote_fallback(code, tick)
            ok_sig, why_sig = self._precheck_signal(
                code, name, side, prop.ai_score, prop.agreement, tick,
                risk_exit=prop.risk_exit)
            if not ok_sig:
                self.gate.mark_blocked(prop, why_sig)
                self._emit("risk_block", {
                    "code": code, "side": side.value, "reason": why_sig,
                    "phase": "execute_precheck",
                })
                return False, why_sig, None
            # 纸交易：走纸账户
            if side == OrderSide.BUY:
                ok, msg, rec = self.paper_account.buy(
                    code, name, prop.price, prop.quantity,
                    strategy=prop.strategy_tag, ai_score=prop.ai_score,
                )
            else:
                ok, msg, rec = self.paper_account.sell(
                    code, prop.price, prop.quantity,
                    strategy=prop.strategy_tag, ai_score=prop.ai_score,
                )
            if ok:
                self.gate.mark_executed(proposal_id, rec.trade_id if rec else "")
                with self._lock:
                    self._last_order_ts = time.time()
                    self._orders_this_minute.append(self._last_order_ts)
                if side == OrderSide.BUY:
                    # 用**实际成交金额**（含滑点）记账：用委托价会系统性低估换手，
                    # 高波动票每次少记十几个 bp，一天下来预算会「莫名剩一点」。
                    _filled = float(getattr(rec, "amount", 0) or 0) if rec else 0.0
                    self._bump_turnover(_filled or prop.quantity * prop.price)
                self._emit("fill", {
                    "proposal_id": proposal_id, "code": code, "side": side.value,
                    "qty": prop.quantity, "price": prop.price, "paper": True,
                    "operator": operator,
                })
                # 构造规范 Order 供下游(AutoTrader 事件)消费，保持接口一致
                order = Order(
                    order_id=rec.trade_id if rec else proposal_id,
                    code=code, name=name, side=side,
                    order_type=OrderType.MARKET,
                    price=prop.price, quantity=prop.quantity,
                    filled_quantity=prop.quantity,
                    status=OrderStatus.FILLED,
                    ai_score=prop.ai_score, strategy_tag=prop.strategy_tag,
                )
                return True, msg, order
            return False, msg, None

        # 实盘：复用 TradeExecutor（内层风控 + 券商下单）
        # 必须把计划股数一并传下去：否则 executor 会按自己的通用仓位算法决定
        # 数量，导致「部分止盈」变成整仓清仓、买入也不按计划预算。
        ok, msg, order = self.executor.execute_ai_signal(
            code=code, name=name, ai_score=prop.ai_score,
            agreement=prop.agreement, side=side,
            risk_exit=prop.risk_exit,
            explicit_quantity=prop.quantity,
        )
        if ok and order:
            self.gate.mark_executed(proposal_id, order.order_id)
            with self._lock:
                self._last_order_ts = time.time()
                self._orders_this_minute.append(self._last_order_ts)
            if side == OrderSide.BUY:
                self._bump_turnover(order.quantity * order.price)
            self._emit("fill", {
                "proposal_id": proposal_id, "code": code, "side": side.value,
                "qty": order.quantity, "price": order.price, "paper": False,
                "order_id": order.order_id, "operator": operator,
            })
        return ok, msg, order

    # ---------- 周期性监控 / 熔断 ----------
    def _heal_throttle_window(self) -> float:
        """解冻后「降权冷静期」的截止时间戳。

        优先按**交易日**计（``auto_resume_throttle_days``）：半仓期必须是交易的
        天数 —— 周五解冻 + 2 个自然日恰好落在周日，等于没有冷静期。
        日历不可用时退回秒级配置（``auto_resume_throttle_sec``），绝不因日历
        异常而放弃冷静期。
        """
        now = time.time()
        days = float(getattr(self.config, "auto_resume_throttle_days", 0) or 0)
        if days <= 0:
            return now + float(self.config.auto_resume_throttle_sec)
        try:
            from src.trading.calendar import next_trading_day

            d = datetime.now().date()
            for i in range(max(1, int(round(days)))):
                d = next_trading_day(d, include_today=(i == 0))   # 第 i+1 个半仓交易日
            end_day = next_trading_day(d)                          # 半仓期结束的次一交易日
            return datetime.combine(end_day, datetime.min.time()).timestamp()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"按交易日计算冷静期失败，回退秒级配置: {e}")
            return now + float(self.config.auto_resume_throttle_sec)

    def _heal_condition_cleared(self, equity: float) -> bool:
        """熔断自诊断：触发熔断的条件是否已经真正解除。

        自动熔断（auto(...)）在跨日自愈前先调用本方法做「智能诊断」——
        只有当回撤 / 单日亏损都已回落到熔断线之下，才允许解冻；否则保持熔断，
        避免「市场仍在恶化却无人值守盲目放开」。人工 KILL 与实盘熔断不走这里
        （它们由 ``tick_monitor`` 的新交易日分支直接判定不恢复）。
        """
        if self.config.auto_resume_require_drawdown_clear:
            dd = self.safety.current_drawdown_pct(equity)
            # 滞后带：必须回落到 (熔断线 − margin) 之下才算清除。margin=0 时
            # 退化为旧行为「不超线即可」，仍可配置回退。
            _clear_line = float(self.config.max_portfolio_drawdown_pct) - float(
                getattr(self.config, "auto_resume_drawdown_margin_pct", 0.0) or 0.0)
            if dd >= _clear_line:
                return False  # 回撤尚未回落到滞后带之下 → 未清除
        if self.config.auto_resume_require_dailyloss_clear:
            dl = self.safety.daily_loss_pct(equity)
            if dl <= -self._effective_daily_loss_pct():
                return False  # 单日亏损仍在线上 → 未清除
        return True

    def tick_monitor(self) -> List[str]:
        """周期性检查：净值峰值、回撤、单日亏损。触发则 halt。返回告警列表"""
        alerts: List[str] = []
        account = self._account()
        if not account:
            return alerts
        equity = account.total_asset
        acct_id = str(getattr(account, "account_id", "") or "")

        # 账号切换 → 净值基数必须重置，否则旧 peak 会立刻误判巨额回撤。
        # 仅在「新旧账号都明确」时才自动重置：任一方为空说明信息不足，
        # 宁可让操作者用 rebaseline_peak() 显式处理，也不静默放宽熔断。
        prev_acct = self.safety._state.baseline_account
        if acct_id and prev_acct and acct_id != prev_acct:
            self.safety.rebaseline_peak(
                equity, operator="auto(account-switch)",
                reason=f"资金账号 {prev_acct} → {acct_id}，净值基数不可比",
                account_id=acct_id,
            )
            alerts.append(f"资金账号已切换（{prev_acct} → {acct_id}），净值基数已重置为 ¥{equity:,.0f}")
        elif acct_id and not prev_acct:
            # 首次记录账号身份（旧状态文件没有这个字段）
            with self.safety._lock:
                self.safety._state.baseline_account = acct_id
                self.safety._save()

        # ---- 新交易日边界 ----
        # 自动熔断（单日亏损 / 组合回撤）本质是「按日 / 按状态」的风险断路器，
        # 隔夜后应自愈，否则无人值守循环会在任何一次熔断后永久停摆、再也不
        # 交易，直到有人手动 resume —— 这直接违背了「无人值守」的初衷。
        #
        # 安全约束（守住铁律，缺一不可）：
        #  1) 仅自愈「自动触发」的熔断（halted_by 以 "auto(" 开头）。
        #     人工 KILL SWITCH（kill_switch / 看板手动，operator=user/dashboard/system）
        #     绝不自动恢复，必须人工 resume。
        #  2) 实盘(LIVE/LIVE_PENDING)模式下的任何熔断都不自动恢复 ——
        #     自动恢复等于「无人值守自动回实盘」，违反「绝不自动实盘」。
        #     这类熔断需人工 resume（resume 会强制回到 PAPER）。
        #  3) 自愈后保持当前模式（通常是 PAPER），不会因此进入实盘。
        #  4) legacy 状态（halted_by 为空）按「非自动」处理 —— 宁可不动，
        #     也不在信息不足时擅自放行，fail-safe。
        #  5) 智能诊断（Feature #5）：自动熔断在新交易日并不「无条件」放开，
        #     而是先经 ``_heal_condition_cleared`` 确认回撤/单日亏损均已回落到
        #     熔断线之下才解冻；条件未清除则保持熔断、仅刷新基线，下个交易日
        #     再评估。解冻后进入「降权冷静期」（``_heal_throttle_until``），
        #     期间 ``_adjust_signal_strength`` 把下单强度打折，实现部分解冻。
        #
        # 注：回撤熔断是「峰值相对当前」的，自愈后下一 tick 若仍超阈值会
        # 立即重新熔断 —— 等于「每天重新评估一次」，既恢复交易能力、
        # 又不丢失对持续深跌的保护；单日亏损熔断则随今日基线重置自然归零。
        today = datetime.now().strftime("%Y-%m-%d")
        st = self.safety._state
        if st.today_date != today:
            if st.halted:
                _by = st.halted_by or ""
                _is_auto = _by.startswith("auto(")
                _is_paper = self.safety.mode == TradingMode.PAPER
                if _is_auto and _is_paper:
                    _reason = st.halt_reason or _by
                    # 智能自愈：先诊断触发条件是否已真正解除。仅当回撤/单日亏损
                    # 均已回落到熔断线之下才解冻；否则保持熔断（市场仍在恶化），
                    # 仅刷新当日起始净值，下个交易日再评估。这把原本「隔夜无条件
                    # 放开」升级为「条件清除才放开」，杜绝在深跌中继盲目恢复交易。
                    if self._heal_condition_cleared(equity):
                        with self.safety._lock:
                            st.halted = False
                            st.halt_reason = ""
                            st.halt_at = ""
                            st.halted_by = ""
                            self.safety._save()
                        # 部分解冻：进入「降权冷静期」，期间下单强度打折扣
                        self._heal_throttle_until = self._heal_throttle_window()
                        _until_txt = datetime.fromtimestamp(
                            self._heal_throttle_until).strftime("%m-%d %H:%M")
                        self._emit("auto_resume", {
                            "reason": _reason, "halted_by": _by,
                            "equity": equity,
                            "drawdown_pct": self.safety.current_drawdown_pct(equity),
                            "clear_line_pct": (
                                float(self.config.max_portfolio_drawdown_pct)
                                - float(getattr(self.config,
                                                "auto_resume_drawdown_margin_pct", 0.0) or 0.0)),
                            "throttle_until": _until_txt,
                            "risk_fraction": float(self.config.auto_resume_risk_fraction),
                        })
                        logger.warning(
                            f"♻️ 自动熔断已智能自愈（条件已清除，缘由 {_by}：{_reason}），"
                            f"回撤 {self.safety.current_drawdown_pct(equity):.2f}% "
                            f"< 滞后带 {float(self.config.max_portfolio_drawdown_pct) - float(getattr(self.config, 'auto_resume_drawdown_margin_pct', 0.0) or 0.0):.2f}%，"
                            f"无人值守循环恢复交易（保持 PAPER，"
                            f"{float(self.config.auto_resume_risk_fraction):.0%} 仓位至 {_until_txt}）"
                        )
                        alerts.append(
                            f"自动熔断智能自愈：{_reason}（触发条件已清除，"
                            f"{float(self.config.auto_resume_risk_fraction):.0%} 仓位至 {_until_txt}）")
                        self.safety.set_today_start(equity)
                    else:
                        # 条件未清除 → 保持熔断，仅刷新当日起始净值避免重复误触发
                        self.safety.set_today_start(equity)
                        alerts.append(
                            f"自动熔断保留（触发条件未清除）：{_reason}")
                else:
                    # 人工 / 实盘熔断：不解除，仅刷新当日起始净值
                    self.safety.set_today_start(equity)
            else:
                self.safety.set_today_start(equity)

        self.safety.update_equity(equity)

        # 波动率尾部风险自适应降仓（Feature #6）：每 tick 更新净值波动阻尼系数
        try:
            self._update_vol_control(equity)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"波动率降仓更新跳过（已忽略）: {e}")

        # 回撤熔断
        dd = self.safety.current_drawdown_pct(equity)
        # 本 tick 内是否刚由「回撤」这条线拉下闸。同一个 tick 里两条线可能同时
        # 成立，那时允许「单日亏损」这条更具体的线复核原因（只复核一次，不算
        # 重复触发）；跨 tick 则一律不再改写，避免熔断期间每 30s 刷日志。
        halted_this_tick = False
        if dd >= self.config.max_portfolio_drawdown_pct:
            reason = f"组合回撤 {dd:.2f}% ≥ 熔断线 {self.config.max_portfolio_drawdown_pct:.1f}%"
            # ⚠️ 只在「首次进入熔断」时落状态 + 告警。熔断会持续多天，条件每 tick
            # 都成立，无条件调 halt() 会让同一个状态每 30s 重写一次文件、重打一条
            # WARNING（实测 4,671 条），且会把峰谷回撤的实时值不断覆盖进
            # halt_reason，事后无法看出「第一次触发时回撤是多少」。
            # 另：已在熔断中（无论是人工 KILL 还是别的规则）不得被本条覆盖原因 ——
            # 人工停手的原因被改写成「组合回撤」会让审计看不出有人动过手。
            if not self.safety.halted:
                self.safety.halt(reason, operator="auto(drawdown)")
                alerts.append(reason)
                halted_this_tick = True

        # 单日亏损熔断（比例线 / 绝对金额线，任一触发即停）
        dl = self.safety.daily_loss_pct(equity)
        loss_abs = -dl / 100.0 * self.safety._state.today_start_equity
        # ⚠️ 基线缺失/异常时**不能**拿 0 当基线去比金额线：limit 会算成 ¥0，
        # 于是 loss_abs(≥0) ≥ 0 恒成立 → 每天开盘第一 tick 就误熔断。
        # 基线不可信时只让比例线说话（它不依赖基线金额）。
        _base = self.safety._state.today_start_equity
        _base_ok = isinstance(_base, (int, float)) and _base > 0
        abs_limit = self.config.daily_loss_abs_limit(_base if _base_ok else 0.0)
        abs_limit = abs_limit if _base_ok else float("inf")
        eff_line = self._effective_daily_loss_pct()
        hit_pct = dl <= -eff_line
        hit_abs = _base_ok and loss_abs >= abs_limit
        if hit_pct or hit_abs:
            # ⚠️ 必须说清是**哪条线**触发的。只报百分比会严重误导排障：
            # 2026-10-20 实际是绝对金额线（¥11,340 ≥ ¥5,000）触发的，
            # 但消息写成「单日亏损 -0.23% 触发熔断」，而比例线是 3% ——
            # 看上去像「亏 0.23% 就熔断」，让人以为阈值被改坏或算错。
            which = []
            if hit_pct:
                which.append(f"比例线 {dl:.2f}% ≤ -{eff_line:.1f}%")
            if hit_abs:
                abs_desc = ("∞（未设金额线）" if abs_limit == float("inf")
                            else f"¥{abs_limit:,.0f}"
                            + (f" = 权益×{self.config.max_daily_loss_abs_pct:.2f}%"
                               if float(self.config.max_daily_loss_abs_pct or 0) > 0 else ""))
                which.append(f"金额线 ¥{loss_abs:,.0f} ≥ {abs_desc}")
            reason = (f"单日亏损熔断（{'；'.join(which)}；"
                      f"当日亏损 {dl:.2f}% / ¥{loss_abs:,.0f}）")
            # 同回撤分支：只在跃迁时触发，避免每 tick 重复落盘/刷日志。
            # 例外 = 本 tick 刚被「回撤」拉下闸：两条线同时成立时，更具体的
            # 「单日亏损（哪条线）」应覆盖宽泛的「组合回撤」，便于事后归因。
            if (not self.safety.halted) or halted_this_tick:
                self.safety.halt(reason, operator="auto(dailyloss)")
                if not halted_this_tick:
                    alerts.append(reason)

        # 清理超时委托
        expired = self.gate.reap_expired()
        if expired:
            alerts.append(f"{expired} 笔待确认委托超时撤销")

        return alerts

    # ---------- Kill Switch 对外接口 ----------
    def kill_switch(self, reason: str, operator: str = "user") -> bool:
        self._emit("halt", {"reason": reason, "operator": operator})
        return self.safety.halt(reason, operator)

    def resume(self, operator: str = "user") -> bool:
        self._emit("resume", {"operator": operator})
        return self.safety.resume(operator)

    # ---------- 状态快照 ----------
    def get_status(self) -> dict:
        account = self._account()
        return {
            "mode": self.safety.mode.value,
            "halted": self.safety.halted,
            "halt_reason": self.safety._state.halt_reason,
            "armed_by": self.safety._state.armed_by,
            "armed_at": self.safety._state.armed_at,
            "peak_equity": self.safety._state.peak_equity,
            "current_drawdown_pct": (
                self.safety.current_drawdown_pct(account.total_asset) if account else 0.0
            ),
            "pending_confirmations": self.gate.pending_count(),
            "requires_confirmation": self.safety.requires_per_order_confirmation(),
            "account": {
                "total_asset": account.total_asset if account else 0,
                "available_cash": account.available_cash if account else 0,
                "market_value": account.market_value if account else 0,
            } if account else None,
            "positions": len(self._positions()),
        }


# 便捷构造：从默认/指定 YAML 创建 supervisor
def build_supervisor(
    broker: BrokerAPI,
    config_path: Optional[str] = None,
    **kwargs,
) -> RiskSupervisor:
    """从 YAML 配置构建 RiskSupervisor（配置缺失则用默认保守值）"""
    cfg = GuardConfig.from_yaml(config_path) if config_path else GuardConfig()
    return RiskSupervisor(broker, cfg, **kwargs)

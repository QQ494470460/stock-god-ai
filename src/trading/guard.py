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

try:  # 与 calendar.py 一致：打包/测试环境下 config 可能不在 sys.path 顶层
    from config.settings import ROOT_DIR, DATA_DIR
except Exception:  # noqa: BLE001
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"


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

    # ---- 持仓约束 ----
    max_positions: int = 20                  # 最大持仓只数
    max_single_position_pct: float = 0.30    # 单票最大仓位(占总资产)
    max_sector_pct: float = 0.50             # 单一行业最大集中度(占总资产)

    # ---- 下单频率 ----
    min_order_interval_sec: float = 2.0      # 两笔委托最小间隔(秒)
    max_orders_per_minute: int = 10          # 每分钟最大委托数

    # ---- 仓位规模 ----
    sizing_method: str = "fixed_fractional"  # fixed_fractional | volatility_target | kelly
    base_risk_fraction: float = 0.02         # 每笔风险敞口占净值比例(固定比例法)
    volatility_target_pct: float = 0.15      # 年化波动率目标(波动率目标法)
    kelly_cap: float = 0.10                  # Kelly 仓位上限(半 Kelly 后再封顶)

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
        """全局紧急停止：持久化，重启仍保持"""
        with self._lock:
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

    def __init__(self, config: GuardConfig):
        self.config = config

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
    ) -> Tuple[int, float]:
        """返回 (整手数量, 预算金额)

        Args:
            side: 买卖方向
            price: 委托价
            total_equity: 当前总资产
            signal_strength: 信号强度 0~1（融合多模型概率/评分）
            annualized_vol: 标的年化波动率（volatility_target 用）
            win_prob / payoff: Kelly 用（payoff=avg_win/avg_loss）
            existing_value: 该标的已有市值（用于加仓上限）
            max_single_pct: 单票上限（默认取配置）
        """
        if price <= 0 or total_equity <= 0:
            return 0, 0.0

        max_single_pct = max_single_pct or self.config.max_single_position_pct
        strength = max(0.0, min(1.0, signal_strength))

        if self.config.sizing_method == "volatility_target" and annualized_vol:
            # 波动率越高，预算越小：budget_frac = target / vol（封顶 base 的 2 倍）
            vol = max(annualized_vol, 1e-6)
            frac = min(self.config.volatility_target_pct / vol,
                       self.config.base_risk_fraction * 2)
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
    ):
        self.broker = broker
        self.config = config or GuardConfig()
        self.paper_account = paper_account
        self.industry_map = industry_map or {}

        # 复用现有逐单风控执行器（防御纵深内层）
        self.executor = TradeExecutor(broker, risk_config)
        self.risk_config = self.executor.config

        # 安全层组件（把 broker 注入进去：武装实盘前要能确认它是真实盘通道）
        self.safety = SafetyController(self.config, broker=broker)
        self.gate = OrderGate(self.config, self.safety)
        self.sizer = PositionSizer(self.config)
        self.journal = TradeJournal(self.config)

        self._lock = threading.RLock()
        self._last_order_ts: float = 0.0
        self._orders_this_minute: List[float] = []
        self._on_event: Optional[Callable] = None  # UI 事件回调

        # 信号日志去重：{(code, side): (评分档, agreement, strength)}，跨日清空
        self._signal_dedupe: Dict[Tuple[str, str], Tuple] = {}
        self._signal_dedupe_day: str = ""

        logger.info(
            f"🛡️ RiskSupervisor 初始化 | mode={self.safety.mode.value} "
            f"halted={self.safety.halted}"
        )

    # ---------- 事件回调 ----------
    def set_event_callback(self, cb: Callable[[str, dict], None]):
        self._on_event = cb

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

        # 1. 全局开关
        if not self.safety.can_trade():
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

        # 5. 仓位规模
        #    · 交易计划给了明确股数 → 用计划的（计划定数量）
        #    · 否则走通用仓位算法（固定比例 / 波动率目标 / 半 Kelly）
        existing_value = positions[code].market_value if code in positions else 0.0
        if explicit_quantity is not None and explicit_quantity > 0:
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
            )
        if qty < 100:
            prop = self.gate.create(
                code=code, name=name, side=side.value, price=price,
                quantity=0, est_amount=0.0, ai_score=ai_score,
                agreement=agreement, signal_strength=signal_strength,
                strategy_tag=strategy_tag, risk_exit=risk_exit,
            )
            self.gate.mark_blocked(prop, f"计算下单量不足1手 ({qty}股)")
            self._emit("risk_block", {
                "code": code, "side": side.value,
                "reason": f"计算下单量不足1手 ({qty}股)",
            })
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
          · 一律整手
        """
        if price <= 0 or want_qty <= 0:
            return 0, 0.0
        qty = (int(want_qty) // lot) * lot
        if qty <= 0:
            return 0, 0.0

        if side == OrderSide.SELL:
            pos = positions.get(code)
            qty = min(qty, sellable_quantity(pos)) if pos else 0
        else:
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
        positions: Dict[str, Position],
    ) -> str:
        """返回 "" 表示通过；否则返回拦截原因"""
        if side == OrderSide.BUY:
            # 持仓只数
            active = [p for p in positions.values() if p.quantity > 0]
            if len(active) >= self.config.max_positions:
                return f"持仓只数已达上限 {self.config.max_positions}"
            # 行业集中度
            sector = self.industry_map.get(code, "")
            if sector:
                sector_val = sum(
                    p.market_value for p in active
                    if self.industry_map.get(p.code, "") == sector
                )
                # 估算新增
                cap = account.total_asset * self.config.max_sector_pct
                if sector_val >= cap:
                    return f"行业 {sector} 集中度已达上限 {self.config.max_sector_pct*100:.0f}%"
        return ""

    # ---------- 频率限制 ----------
    def _check_rate_limit(self) -> str:
        now = time.time()
        with self._lock:
            # 最小间隔
            if now - self._last_order_ts < self.config.min_order_interval_sec:
                return f"下单间隔过短(<{self.config.min_order_interval_sec}s)"
            # 每分钟上限
            self._orders_this_minute = [t for t in self._orders_this_minute if now - t < 60]
            if len(self._orders_this_minute) >= self.config.max_orders_per_minute:
                return f"每分钟委托数已达上限 {self.config.max_orders_per_minute}"
        return ""

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
            return False, "KILL SWITCH 已触发，禁止执行", None

        rate_reason = self._check_rate_limit()
        if rate_reason:
            return False, rate_reason, None

        side = OrderSide(prop.side)
        code, name = prop.code, prop.name

        if self.safety.mode == TradingMode.PAPER and self.paper_account:
            # 防御：执行前再跑一次信号预检（避免 propose→execute 之间状态变化）
            tick = self.broker.get_tick(code)
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
        ok, msg, order = self.executor.execute_ai_signal(
            code=code, name=name, ai_score=prop.ai_score,
            agreement=prop.agreement, side=side,
            risk_exit=prop.risk_exit,
        )
        if ok and order:
            self.gate.mark_executed(proposal_id, order.order_id)
            with self._lock:
                self._last_order_ts = time.time()
                self._orders_this_minute.append(self._last_order_ts)
            self._emit("fill", {
                "proposal_id": proposal_id, "code": code, "side": side.value,
                "qty": order.quantity, "price": order.price, "paper": False,
                "order_id": order.order_id, "operator": operator,
            })
        return ok, msg, order

    # ---------- 周期性监控 / 熔断 ----------
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

        # 首次记录当日起始净值
        if self.safety._state.today_date != datetime.now().strftime("%Y-%m-%d"):
            self.safety.set_today_start(equity)

        self.safety.update_equity(equity)

        # 回撤熔断
        dd = self.safety.current_drawdown_pct(equity)
        if dd >= self.config.max_portfolio_drawdown_pct:
            reason = f"组合回撤 {dd:.2f}% ≥ 熔断线 {self.config.max_portfolio_drawdown_pct:.1f}%"
            self.safety.halt(reason, operator="auto(drawdown)")
            alerts.append(reason)

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
        hit_pct = dl <= -self.config.max_daily_loss_pct
        hit_abs = _base_ok and loss_abs >= abs_limit
        if hit_pct or hit_abs:
            # ⚠️ 必须说清是**哪条线**触发的。只报百分比会严重误导排障：
            # 2026-10-20 实际是绝对金额线（¥11,340 ≥ ¥5,000）触发的，
            # 但消息写成「单日亏损 -0.23% 触发熔断」，而比例线是 3% ——
            # 看上去像「亏 0.23% 就熔断」，让人以为阈值被改坏或算错。
            which = []
            if hit_pct:
                which.append(f"比例线 {dl:.2f}% ≤ -{self.config.max_daily_loss_pct:.1f}%")
            if hit_abs:
                abs_desc = ("∞（未设金额线）" if abs_limit == float("inf")
                            else f"¥{abs_limit:,.0f}"
                            + (f" = 权益×{self.config.max_daily_loss_abs_pct:.2f}%"
                               if float(self.config.max_daily_loss_abs_pct or 0) > 0 else ""))
                which.append(f"金额线 ¥{loss_abs:,.0f} ≥ {abs_desc}")
            reason = (f"单日亏损熔断（{'；'.join(which)}；"
                      f"当日亏损 {dl:.2f}% / ¥{loss_abs:,.0f}）")
            self.safety.halt(reason, operator="auto(dailyloss)")
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

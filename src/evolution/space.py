# -*- coding: utf-8 -*-
"""自我进化 · 参数空间与安全边界。

设计铁律
========
自我进化能动的是**策略参数**（选什么、买多少、调仓多勤），
绝不能动**风控硬线**（熔断线、单票上限、下单频率限制等）。
后者一旦被"自动优化"，就等于让系统自己拆掉刹车。

因此本模块把参数分成两类：

* :data:`PARAM_SPACE` —— 白名单。只有登记在这里的参数才允许被进化触碰，
  且每个都有类型、取值范围、`risk` 方向与 `auto_adopt` 许可。
* :func:`is_protected` —— 黑名单判定。任何路径命中风控命名空间
  （``guard.*`` / ``trading_guard.*`` / ``risk.*``）或风控叶子名，
  一律拒绝，不看白名单。

取值方向的安全性（``risk``）：

* ``neutral``      —— 双向都不可怕（如调仓容忍带、扫描间隔）
* ``risky_up``     —— 值**调高**更激进（如单票权重上限、目标持仓数）
* ``risky_down``   —— 值**调低**更激进（如现金保留比例、选股阈值）

任何「朝更激进方向」的改动一律 **不允许自动采纳**，只能进待批准队列由人拍板。
即便是在安全方向，也只有 `auto_adopt=True` 的参数才允许自动落地。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "ParamSpec", "ParamError", "PARAM_SPACE", "PROTECTED_NAMESPACES",
    "PROTECTED_LEAVES", "is_protected", "get_spec", "get_default",
    "get_value", "set_value", "coerce", "validate", "sample_domain",
    "risk_direction_ok", "check_changes_adoptable", "describe_space",
]

KIND_INT = "int"
KIND_FLOAT = "float"
KIND_BOOL = "bool"
KIND_CATEGORY = "category"
KIND_STR = "str"


class ParamError(ValueError):
    """参数非法（越界 / 受保护 / 未登记）。"""


# ---------------------------------------------------------------------------
# 风控黑名单 —— 永远不许自动改
# ---------------------------------------------------------------------------

#: 命名空间前缀：出现即拒绝（guard 的风控配置都在这些命名空间下）
PROTECTED_NAMESPACES: Tuple[str, ...] = (
    "guard.", "trading_guard.", "risk.", "safety.", "executor.",
)

#: 叶子名：即使不带前缀也拒绝（这些名字本身就是风控指标）
PROTECTED_LEAVES: Tuple[str, ...] = (
    "max_portfolio_drawdown_pct",
    "max_daily_loss_pct",
    "max_daily_loss_abs",
    "max_daily_loss_abs_pct",
    "max_single_position_pct",
    "max_orders_per_minute",
    "max_orders_per_tick",          # 防雪崩闸门，属安全阀
    "base_risk_fraction",
    "kelly_cap",
    "min_ai_score_buy",
    "min_ai_score_sell",
    "max_ai_score_sell",
    "require_live_confirmation",
    "require_per_order_confirmation",
    "default_mode",
    "stop_loss_pct",
    "take_profit_pct",
    "trailing_stop_pct",
    "max_industry_pct",
    "position_sizing_method",
    "sizing_method",
)

#: 裸名歧义保护：这些名字在不同命名空间含义不同，裸用一律拒绝
_AMBIGUOUS_BARE = ("max_positions",)


def is_protected(key: str) -> bool:
    """是否为风控硬线（永不允许自动改）。"""
    k = (key or "").strip()
    if not k:
        return True
    if any(k.startswith(ns) for ns in PROTECTED_NAMESPACES):
        return True
    leaf = k.rsplit(".", 1)[-1]
    if leaf in PROTECTED_LEAVES:
        return True
    if k in _AMBIGUOUS_BARE:
        return True
    return False


# ---------------------------------------------------------------------------
# 参数规格
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    """一个可进化参数的规格。"""
    key: str                       # 点分路径，如 "trading_loop.max_positions"
    kind: str
    default: Any
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: Tuple[Any, ...] = ()
    risk: str = "neutral"          # neutral | risky_up | risky_down
    auto_adopt: bool = True        # 是否允许「安全方向」的改动自动落地
    doc: str = ""

    # ---- 取值 ----

    def clamp(self, value: Any) -> Any:
        v = self.cast(value)
        if self.kind in (KIND_INT, KIND_FLOAT):
            if self.low is not None:
                v = max(self.low, v)
            if self.high is not None:
                v = min(self.high, v)
            if self.kind == KIND_INT:
                v = int(round(v))
            else:
                v = round(float(v), 6)
        return v

    def cast(self, value: Any) -> Any:
        """按 kind 做类型转换，失败抛 ParamError。

        注意：``KIND_INT`` 走 **四舍五入** 而非截断。
        截断会让 ``7.999999999999999``（浮点误差的常见长相）掉到 7，
        也会让生成器产出的 7.6 悄悄偏向低估；且与 `clamp()` 的取整
        语义保持一致，避免「同一路径两种取整」。
        """
        try:
            if self.kind == KIND_BOOL:
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "y", "on")
                return bool(value)
            if self.kind == KIND_INT:
                return int(round(float(value)))
            if self.kind == KIND_FLOAT:
                return float(value)
            if self.kind == KIND_CATEGORY:
                if value not in self.choices:
                    raise ParamError(
                        f"{self.key}: 取值 {value!r} 不在允许集合 {self.choices}")
                return value
            if self.kind == KIND_STR:
                # 自由字符串（如模型版本 id）。这里刻意**不**做格式白名单，
                # 因为版本 id 是动态产生的；安全性由「值只用来查表」保证 ——
                # 查不到就回退默认行为，而不是被当成路径/代码执行。
                return "" if value is None else str(value)
        except ParamError:
            raise
        except (TypeError, ValueError) as e:
            raise ParamError(f"{self.key}: 无法把 {value!r} 转成 {self.kind}: {e}") from e
        return value

    def in_range(self, value: Any) -> bool:
        try:
            v = self.cast(value)
        except ParamError:
            return False
        if self.kind in (KIND_INT, KIND_FLOAT):
            if self.low is not None and v < self.low:
                return False
            if self.high is not None and v > self.high:
                return False
        return True

    # ---- 风险方向 ----

    def is_risk_increasing(self, old: Any, new: Any) -> bool:
        """改动是否朝「更激进」方向。中性参数恒为 False。"""
        if self.kind in (KIND_STR, KIND_CATEGORY, KIND_BOOL):
            # 无序取值（版本 id / 枚举 / 开关）没有「更大」的概念。
            # 声明成 risky_up/risky_down 只会得到字典序比较这种无意义结果，
            # 因此这里直接判定为「非渐进」（真正的高风险靠 auto_adopt=False 拦住）。
            return False
        try:
            o, n = self.cast(old), self.cast(new)
        except ParamError:
            return True          # 说不清就当作有风险
        if self.risk == "risky_up":
            return n > o
        if self.risk == "risky_down":
            return n < o
        return False

    # ---- 枚举 ----

    def domain(self) -> List[Any]:
        """给出候选取值集合（供生成器做网格/变异）。"""
        if self.kind == KIND_CATEGORY:
            return list(self.choices)
        if self.kind == KIND_BOOL:
            return [True, False]
        if self.low is None or self.high is None:
            return [self.default]
        step = self.step or (1 if self.kind == KIND_INT else 0.05)
        out, v, guard = [], float(self.low), 0
        while v <= float(self.high) + 1e-9 and guard < 1000:
            out.append(int(round(v)) if self.kind == KIND_INT else round(v, 6))
            v += step
            guard += 1
        if self.default not in out and self.in_range(self.default):
            out.append(self.default)
            out.sort(key=lambda x: float(x))
        return out


# ---------------------------------------------------------------------------
# 白名单
# ---------------------------------------------------------------------------

PARAM_SPACE: Dict[str, ParamSpec] = {s.key: s for s in (
    # ===== 组合参数（闭环引擎）=====
    ParamSpec("trading_loop.max_positions", KIND_INT, 10, low=1, high=20, step=1,
              risk="risky_up", auto_adopt=False,
              doc="目标持仓只数。越多越分散但也越难管理，调高需人工确认"),
    ParamSpec("trading_loop.cash_reserve_pct", KIND_FLOAT, 0.10, low=0.0, high=0.50,
              step=0.05, risk="risky_down", auto_adopt=False,
              doc="现金保留比例。调低=满仓更激进"),
    ParamSpec("trading_loop.weight_cap", KIND_FLOAT, 0.30, low=0.05, high=0.30,
              step=0.05, risk="risky_up", auto_adopt=False,
              doc="单票权重上限（不得超过风控 max_single_position_pct）"),
    ParamSpec("trading_loop.band_pct", KIND_FLOAT, 0.15, low=0.05, high=0.50,
              step=0.05, risk="neutral", auto_adopt=True,
              doc="调仓容忍带。越大越少交易（省成本），越小越贴近计划。"
                  "默认 0.15 经真样本外面板（panel_oos）三档成本验证"),
    ParamSpec("trading_loop.min_trade_amount", KIND_FLOAT, 5000.0, low=1000.0,
              high=50000.0, step=1000.0, risk="neutral", auto_adopt=True,
              doc="最小交易金额，过滤碎单"),
    ParamSpec("trading_loop.weighting", KIND_CATEGORY, "risk_parity",
              choices=("risk_parity", "equal", "score", "min_variance", "vol_target"),
              risk="neutral", auto_adopt=True,
              doc="权重分配方式（风险平价/等权/按评分/最小方差/波动率目标）"),
    ParamSpec("trading_loop.tilt_strength", KIND_FLOAT, 0.25, low=0.0, high=0.6,
              step=0.05, risk="neutral", auto_adopt=True,
              doc="风险权重上叠加 alpha 倾斜的力度（秩空间，0=纯风险平价）"),
    ParamSpec("trading_loop.enable_rebalance", KIND_BOOL, True,
              risk="neutral", auto_adopt=True, doc="是否执行目标调仓"),
    ParamSpec("trading_loop.enable_stop_orders", KIND_BOOL, True,
              risk="neutral", auto_adopt=False, doc="是否执行止盈止损（关掉=放弃纪律离场）"),
    ParamSpec("trading_loop.scan_interval_sec", KIND_FLOAT, 30.0, low=10.0,
              high=300.0, step=10.0, risk="neutral", auto_adopt=True,
              doc="盘中扫描间隔（秒）"),

    # ===== 选股阈值（与已有 EvolutionEngine 兼容）=====
    ParamSpec("evolution.min_score_base", KIND_FLOAT, 50.0, low=30.0, high=60.0,
              step=1.0, risk="risky_down", auto_adopt=False,
              doc="选股基准最低评分。调低=放宽进池标准"),
    ParamSpec("evolution.threshold_adjustment", KIND_FLOAT, 0.0, low=-10.0,
              high=15.0, step=0.5, risk="risky_down", auto_adopt=False,
              doc="阈值进化调整量"),

    # ===== 模型超参（供重训/基因进化）=====
    ParamSpec("ml.active_version", KIND_STR, "", risk="neutral", auto_adopt=False,
              doc="当前启用的模型产物键（注册表里的 ModelInfo.key）；"
                  "留空 = 由注册表按样本外 RankIC 自动挑选。"
                  "改它等于换掉整套打分逻辑，影响面最大，故禁止自动落地"),
    ParamSpec("ml.xgb_params.n_estimators", KIND_INT, 500, low=100, high=1200,
              step=50, risk="neutral", auto_adopt=False, doc="XGBoost 树数量"),
    ParamSpec("ml.xgb_params.max_depth", KIND_INT, 5, low=3, high=10, step=1,
              risk="neutral", auto_adopt=False, doc="XGBoost 树深度（越大越易过拟合）"),
    ParamSpec("ml.xgb_params.learning_rate", KIND_FLOAT, 0.03, low=0.005, high=0.20,
              step=0.005, risk="neutral", auto_adopt=False, doc="学习率"),
    ParamSpec("ml.xgb_params.subsample", KIND_FLOAT, 0.75, low=0.5, high=1.0,
              step=0.05, risk="neutral", auto_adopt=False, doc="行采样比例"),
    ParamSpec("ml.xgb_params.colsample_bytree", KIND_FLOAT, 0.75, low=0.5, high=1.0,
              step=0.05, risk="neutral", auto_adopt=False, doc="列采样比例"),
)}


# ---------------------------------------------------------------------------
# 读写点分路径
# ---------------------------------------------------------------------------

def get_spec(key: str, *, allow_protected: bool = False) -> ParamSpec:
    if not allow_protected and is_protected(key):
        raise ParamError(f"{key} 属风控硬线，不在可进化范围内")
    spec = PARAM_SPACE.get(key)
    if spec is None:
        raise ParamError(f"{key} 未登记在可进化参数空间里")
    return spec


def get_default(key: str) -> Any:
    return get_spec(key).default


def get_value(target: Any, key: str) -> Any:
    """按点分路径从对象/字典里取值；取不到返回 spec.default。"""
    node = target
    try:
        for part in key.split("."):
            if isinstance(node, dict):
                node = node.get(part)
            else:
                node = getattr(node, part)
            if node is None:
                break
    except (AttributeError, KeyError):
        return get_spec(key).default
    return get_spec(key).default if node is None else node


def set_value(target: Any, key: str, value: Any) -> None:
    """按点分路径写值（支持 dataclass 属性与嵌套 dict）。"""
    parts = key.split(".")
    node = target
    for part in parts[:-1]:
        if isinstance(node, dict):
            node = node.setdefault(part, {})
        else:
            node = getattr(node, part)
    leaf = parts[-1]
    if isinstance(node, dict):
        node[leaf] = value
    else:
        setattr(node, leaf, value)


def coerce(key: str, value: Any) -> Any:
    return get_spec(key).clamp(value)


def validate(key: str, value: Any) -> Any:
    """校验并规整一个取值；非法则抛 ParamError。"""
    spec = get_spec(key)
    v = spec.cast(value)
    if not spec.in_range(v):
        raise ParamError(
            f"{key}={v!r} 越界（允许 {spec.low} ~ {spec.high}）")
    return v


def sample_domain(key: str) -> List[Any]:
    return get_spec(key).domain()


# ---------------------------------------------------------------------------
# 采纳策略
# ---------------------------------------------------------------------------

def risk_direction_ok(key: str, old: Any, new: Any) -> bool:
    """改动是否**未**朝更激进方向（即可以自动采纳的方向）。"""
    return not get_spec(key).is_risk_increasing(old, new)


def check_changes_adoptable(changes: List[Tuple[str, Any, Any]]) -> Tuple[bool, str]:
    """判断一组改动能否**自动**落地。

    Args:
        changes: [(key, old, new), ...]

    Returns:
        (是否可自动采纳, 原因)
    """
    if not changes:
        return False, "无改动"
    for key, old, new in changes:
        try:
            spec = get_spec(key)
        except ParamError as e:
            return False, str(e)
        if old == new:
            return False, f"{key} 新旧值相同，无需变更"
        if spec.is_risk_increasing(old, new):
            return False, (f"{key} 的改动朝更激进方向（{old} → {new}），"
                           f"必须人工批准")
        if not spec.auto_adopt:
            return False, f"{key} 被标记为不可自动采纳"
    return True, "全部为安全方向的改动，可自动采纳"


def describe_space() -> str:
    """人类可读的参数空间清单（CLI / 文档用）。"""
    lines = [f"可进化参数 {len(PARAM_SPACE)} 项："]
    for spec in PARAM_SPACE.values():
        if spec.kind == KIND_CATEGORY:
            rng = "{" + ", ".join(map(str, spec.choices)) + "}"
        elif spec.kind == KIND_BOOL:
            rng = "{true, false}"
        else:
            rng = f"{spec.low} ~ {spec.high} (step {spec.step})"
        flag = "" if spec.auto_adopt else "  [须人工批准]"
        lines.append(f"  {spec.key:<38} {spec.kind:<8} {rng}{flag}")
        if spec.doc:
            lines.append(f"      {spec.doc}")
    lines.append("")
    lines.append("永久受保护（风控硬线，绝不自动改）：")
    lines.append("  " + ", ".join(PROTECTED_LEAVES))
    return "\n".join(lines)

# -*- coding: utf-8 -*-
"""市场状态（牛/熊/震荡）感知 + 策略动态加权。

把「市场处于什么状态」变成一个可复用的标量，再据此对**不同策略标签**的
下单强度做差异化缩放 —— 牛市放大进攻（动量/追涨）、熊市放大防御与风控离场、
震荡市放大均值回归。

设计要点：
- 纯逻辑、零 IO：``detect_regime`` 直接复用 ``src/models/evolution.py`` 的
  ``detect_regime(benchmark_df)``（基于沪深300 日线的 MA20/MA60 斜率），
  本模块只负责「状态 → 权重」的映射，不重复造指数检测轮子。
- 权重作用在 ``signal_strength`` 上（经 ``RiskSupervisor._adjust_signal_strength``
  在 ``propose_trade`` 入口统一缩放），自动走现有预算公式
  ``budget = equity * frac * (0.5 + 0.5 * strength)``，不改动任何下单路径。
- 风控类标签（stop / risk_exit）永远权重 1.0 —— 止损纪律不受市场状态影响。
- 默认 regime 为 ``sideways``，所有权重≈1.0，故**未接指数数据前零行为变化**，
  不会在没人配置的情况下悄悄改变仓位。
"""

from __future__ import annotations

from typing import Dict, Optional

# 各策略标签在不同市场状态下的相对强度权重。
# 只列会被缩放的标签；未列出的标签一律按 1.0 处理。
REGIME_STRATEGY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "bull": {
        "loop": 1.15, "ai": 1.15, "rebalance": 1.10,
        "momentum": 1.30, "mean_revert": 0.80, "defensive": 0.60,
        "stop": 1.0,  # 风控离场永远执行
    },
    "sideways": {
        "loop": 1.0, "ai": 1.0, "rebalance": 1.0,
        "momentum": 0.90, "mean_revert": 1.20, "defensive": 0.90,
        "stop": 1.0,
    },
    "bear": {
        "loop": 0.85, "ai": 0.85, "rebalance": 0.80,
        "momentum": 0.70, "mean_revert": 0.90, "defensive": 1.30,
        "stop": 1.0,
    },
}

# 未知/未列出标签的兜底权重
DEFAULT_TAG_WEIGHT = 1.0
VALID_REGIMES = ("bull", "sideways", "bear")
DEFAULT_REGIME = "sideways"


def detect_regime(benchmark_df) -> Optional[str]:
    """用基准指数（如沪深300）日线判定当前市场状态。

    基于 MA20/MA60 与 MA20 斜率：价格在 MA60 之上且短期均线向上 → 牛市；
    价格在 MA60 之下且短期均线向下 → 熊市；其余 → 震荡。逻辑与进化引擎的
    ``detect_regime`` 对齐，但本实现**无状态、零副作用**（不写盘），
    适合在交易闭环里每个交易日调用。

    Returns:
        ``"bull"`` / ``"sideways"`` / ``"bear"`` 之一；数据不足或判定失败返回
        ``None``，调用方应回退到默认 regime（sideways，权重≈1.0）。
    """
    if benchmark_df is None:
        return None
    try:
        import pandas as pd
        col = "close" if "close" in getattr(benchmark_df, "columns", []) else None
        if col is None:
            return None
        close = pd.to_numeric(benchmark_df[col], errors="coerce").dropna()
        if len(close) < 60:
            return None
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])
        ma20_slope = float(close.rolling(20).mean().diff(5).iloc[-1])
        last = float(close.iloc[-1])
        if last > ma60 and ma20_slope > 0:
            return "bull"
        if last < ma60 and ma20_slope < 0:
            return "bear"
        return "sideways"
    except Exception:  # noqa: BLE001  // 判定失败绝不抛，回退 default
        return None


class RegimeWeighter:
    """维护当前市场状态，并提供「标签 → 缩放系数」查询。"""

    def __init__(self, regime: str = DEFAULT_REGIME, enabled: bool = True):
        self.regime = regime if regime in VALID_REGIMES else DEFAULT_REGIME
        self.enabled = enabled

    def set_regime(self, regime: Optional[str]) -> None:
        if regime in VALID_REGIMES:
            self.regime = regime

    def update(self, regime: Optional[str]) -> None:
        self.set_regime(regime)

    def weight(self, tag: str, base: float = 1.0) -> float:
        """返回某策略标签在「当前市场状态」下的强度缩放系数。

        ``enabled=False`` 时直接返回 ``base``（关闭功能，零行为变化）。
        """
        if not self.enabled:
            return float(base)
        table = REGIME_STRATEGY_WEIGHTS.get(self.regime, {})
        return float(base) * float(table.get(tag, DEFAULT_TAG_WEIGHT))

    def describe(self) -> Dict[str, float]:
        return dict(REGIME_STRATEGY_WEIGHTS.get(self.regime, {}))


class RegimeSmoother:
    """市场状态粘性（Status Hysteresis）。

    背景：``detect_regime`` 每天基于基准均线重新判断，遇到横盘 / 假突破会
    在牛/熊/震荡之间反复横跳，导致 ``RegimeWeighter`` 权重天天变、策略随之
    忽攻忽守（whipsaw）。本类要求**同一新信号连续出现 ``hold_days`` 个交易日**
    才真正切换，期间维持原状态，滤掉噪声翻转。

    设计：纯状态机、零 IO；非法/未知信号一律当作「维持现状」；与
    ``RegimeWeighter`` 解耦（本类只产出「已确认的 regime 字符串」）。
    """

    def __init__(self, hold_days: int = 2, initial: str = DEFAULT_REGIME):
        self.hold_days = max(1, int(hold_days))
        self.current = initial if initial in VALID_REGIMES else DEFAULT_REGIME
        self._pending: Optional[str] = None
        self._count: int = 0

    def update(self, signal: Optional[str]) -> str:
        """喂入今日原始信号，返回经过粘性处理后的「生效 regime」。

        - 信号 == 当前状态：清空待定，维持现状。
        - 信号 != 当前且与待定一致：计数 +1；达到 ``hold_days`` 才翻盘。
        - 信号非法（None / 未知）：等同维持现状，绝不引入无效状态。
        """
        sig = signal if signal in VALID_REGIMES else self.current
        if sig == self.current:
            self._pending = None
            self._count = 0
            return self.current
        if self._pending != sig:
            self._pending = sig
            self._count = 1
        else:
            self._count += 1
        if self._count >= self.hold_days:
            self.current = sig
            self._pending = None
            self._count = 0
        return self.current

    def describe(self) -> Dict[str, float]:
        """便于与 ``RegimeWeighter.describe`` 对齐：返回当前生效状态的权重表。"""
        return dict(REGIME_STRATEGY_WEIGHTS.get(self.current, {}))

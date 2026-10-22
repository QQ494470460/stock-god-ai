# -*- coding: utf-8 -*-
"""市场状态(regime)感知 + 策略动态加权 单元测试。"""

import numpy as np
import pandas as pd
import pytest

from src.trading.regime import (
    RegimeWeighter, detect_regime, REGIME_STRATEGY_WEIGHTS,
)


def _close_series(values):
    return pd.DataFrame({"close": values})


def test_default_regime_is_sideways_and_weight_one():
    w = RegimeWeighter()  # 默认 sideways
    assert w.regime == "sideways"
    # sideways 下 loop/stop/rebalance/ai 等闭环实际使用的标签权重≈1.0（零行为变化）
    assert w.weight("loop") == 1.0
    assert w.weight("stop") == 1.0
    assert w.weight("rebalance") == 1.0
    # 而 momentum/mean_revert 这类「概念标签」在震荡市被差异化（动量降权、均值回归加权）
    assert w.weight("momentum") < 1.0
    assert w.weight("mean_revert") > 1.0


def test_bull_amplifies_momentum_defensive_dampened():
    w = RegimeWeighter(regime="bull")
    assert w.weight("momentum") > 1.0
    assert w.weight("defensive") < 1.0
    # 风控离场永远 1.0，不受市场状态影响
    assert w.weight("stop") == 1.0


def test_bear_amplifies_defensive_dampens_momentum():
    w = RegimeWeighter(regime="bear")
    assert w.weight("defensive") > 1.0
    assert w.weight("momentum") < 1.0
    assert w.weight("stop") == 1.0


def test_disabled_returns_base():
    w = RegimeWeighter(regime="bull", enabled=False)
    assert w.weight("momentum", 0.8) == 0.8
    assert w.weight("defensive", 0.8) == 0.8


def test_unknown_tag_falls_back_to_one():
    w = RegimeWeighter(regime="bull")
    assert w.weight("nonexistent_strategy") == 1.0


def test_set_regime_validates():
    w = RegimeWeighter()
    w.set_regime("bear")
    assert w.regime == "bear"
    w.set_regime("not_a_regime")  # 非法 → 保持原值
    assert w.regime == "bear"


# ----------------------------------------------------------------------
# detect_regime（无状态、零副作用）
# ----------------------------------------------------------------------
def test_detect_bull():
    # 单调上升 > 60 日 → 价格在 MA60 之上且 MA20 斜率为正
    df = _close_series(np.arange(1, 121, dtype=float))
    assert detect_regime(df) == "bull"


def test_detect_bear():
    # 单调下降 → 价格在 MA60 之下且 MA20 斜率为负
    df = _close_series(np.arange(120, 0, -1, dtype=float))
    assert detect_regime(df) == "bear"


def test_detect_sideways():
    # 恒定 → 既不突破 MA60 也不跌破 → 震荡
    df = _close_series(np.full(120, 100.0))
    assert detect_regime(df) == "sideways"


def test_detect_insufficient_data_returns_none():
    df = _close_series(np.arange(1, 40, dtype=float))
    assert detect_regime(df) is None


def test_detect_none_input_returns_none():
    assert detect_regime(None) is None

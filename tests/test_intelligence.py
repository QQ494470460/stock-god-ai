# -*- coding: utf-8 -*-
"""Batch 3 智能层单元测试：状态粘性 + 策略信誉自适应权重。

纯逻辑、零 IO、零网络。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading.regime import RegimeSmoother
from src.trading.intelligence import StrategyCredibility


# ----------------------------------------------------------------------
# 状态粘性（Regime Hysteresis）
# ----------------------------------------------------------------------
def test_smoother_keeps_state_on_single_flip():
    s = RegimeSmoother(hold_days=2, initial="sideways")
    assert s.update("bull") == "sideways"   # 单次翻转被挡住
    assert s.current == "sideways"


def test_smoother_switches_after_hold_days():
    s = RegimeSmoother(hold_days=2, initial="sideways")
    assert s.update("bull") == "sideways"   # 第 1 次
    assert s.update("bull") == "bull"       # 第 2 次 → 切换
    assert s.current == "bull"


def test_smoother_resets_pending_on_revert():
    s = RegimeSmoother(hold_days=2, initial="sideways")
    s.update("bull")          # pending=bull, count=1
    s.update("sideways")      # 回到原状态 → pending 清空
    assert s.current == "sideways"
    # 再翻转需要重新累计
    assert s.update("bull") == "sideways"
    assert s.update("bull") == "bull"


def test_smoother_ignores_invalid_signal():
    s = RegimeSmoother(hold_days=1, initial="bull")
    assert s.update(None) == "bull"
    assert s.update("mars") == "bull"
    assert s.current == "bull"


def test_smoother_hold_days_one_flips_immediately():
    s = RegimeSmoother(hold_days=1, initial="sideways")
    assert s.update("bear") == "bear"


# ----------------------------------------------------------------------
# 策略信誉自适应权重（Strategy Credibility）
# ----------------------------------------------------------------------
def test_credibility_default_neutral_weight_is_one():
    c = StrategyCredibility()
    assert c.weight("unknown") == pytest.approx(1.0)
    assert c.weights() == {}


def test_credibility_rises_with_consistent_wins():
    c = StrategyCredibility(halflife=10)
    for _ in range(40):
        c.observe("momentum", 1.0, 0.04)   # 看多 + 正收益 → 一致
    assert c.weight("momentum") > 1.0
    assert c.weight("momentum") <= c.cap


def test_credibility_falls_with_consistent_losses():
    c = StrategyCredibility(halflife=10)
    for _ in range(40):
        c.observe("defensive", 1.0, -0.05)  # 看多 + 负收益 → 不一致
    assert c.weight("defensive") < 1.0
    assert c.weight("defensive") >= c.floor


def test_credibility_bounds_respected():
    c = StrategyCredibility(floor=0.5, cap=1.5)
    for _ in range(100):
        c.observe("x", 1.0, 0.02)
    assert c.weight("x") <= 1.5 + 1e-9
    for _ in range(100):
        c.observe("y", 1.0, -0.02)
    assert c.weight("y") >= 0.5 - 1e-9


def test_credibility_short_signal_inverts_consistency():
    c = StrategyCredibility()
    # entry_strength < 0 视为看空；实际下跌 → 一致 → 权重上升
    for _ in range(40):
        c.observe("z", -1.0, -0.03)
    assert c.weight("z") > 1.0


def test_credibility_weights_dict_and_reset():
    c = StrategyCredibility()
    c.observe("a", 1.0, 0.05)
    c.observe("b", 1.0, -0.05)
    w = c.weights()
    assert set(w.keys()) == {"a", "b"}
    c.reset()
    assert c.weights() == {}

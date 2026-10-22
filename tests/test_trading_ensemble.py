# -*- coding: utf-8 -*-
"""多策略信号集成器（src.trading.ensemble）单元测试。

验证「市场状态权重 × 历史 IC 权重」融合逻辑、买卖决策阈值、贡献拆解，
以及未知标签 / 空输入的 fail-safe 行为。纯逻辑、零 IO、零网络。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading.ensemble import fuse_strategies, StrategyEnsemble
from src.trading.regime import RegimeWeighter


# ----------------------------------------------------------------------
# 基础融合：等权即简单平均
# ----------------------------------------------------------------------
def test_fuse_equal_weights_is_simple_mean():
    # 无 regime / 无 IC 权重 → 所有标签权重 1.0 → fused = 各分平均
    out = fuse_strategies({"momentum": 60.0, "defensive": 40.0})
    assert out["n_strategies"] == 2
    assert out["fused_score"] == pytest.approx(50.0)
    assert out["weights"] == pytest.approx({"momentum": 0.5, "defensive": 0.5})
    assert out["contributions"]["momentum"] == pytest.approx(30.0)
    assert out["contributions"]["defensive"] == pytest.approx(20.0)


# ----------------------------------------------------------------------
# 市场状态加权：牛市放大动量、熊市放大防御
# ----------------------------------------------------------------------
def test_bull_amplifies_momentum_contribution():
    regime = RegimeWeighter(regime="bull").describe()  # momentum 1.30 > defensive 0.60
    out = fuse_strategies(
        {"momentum": 50.0, "defensive": 50.0},
        regime_weights=regime,
    )
    assert out["weights"]["momentum"] > out["weights"]["defensive"]
    assert out["contributions"]["momentum"] > out["contributions"]["defensive"]


def test_bear_amplifies_defensive_contribution():
    regime = RegimeWeighter(regime="bear").describe()  # defensive 1.30 > momentum 0.70
    out = fuse_strategies(
        {"momentum": 50.0, "defensive": 50.0},
        regime_weights=regime,
    )
    assert out["weights"]["defensive"] > out["weights"]["momentum"]
    assert out["contributions"]["defensive"] > out["contributions"]["momentum"]


# ----------------------------------------------------------------------
# 决策阈值
# ----------------------------------------------------------------------
def test_decision_buy_and_sell_thresholds():
    # 高分 → buy
    buy = fuse_strategies({"momentum": 80.0, "defensive": 80.0})
    assert buy["fused_score"] == pytest.approx(80.0)
    assert buy["decision"] == "buy"

    # 低分 → sell
    sell = fuse_strategies({"momentum": 20.0, "defensive": 20.0})
    assert sell["fused_score"] == pytest.approx(20.0)
    assert sell["decision"] == "sell"

    # 中间 → hold
    hold = fuse_strategies({"momentum": 50.0, "defensive": 50.0})
    assert hold["fused_score"] == pytest.approx(50.0)
    assert hold["decision"] == "hold"


# ----------------------------------------------------------------------
# IC 权重偏移融合结果
# ----------------------------------------------------------------------
def test_ic_weights_shift_fusion():
    scores = {"momentum": 100.0, "defensive": 0.0}
    base = fuse_strategies(scores)  # 等权 → 50.0
    weighted = fuse_strategies(scores, ic_weights={"momentum": 3.0})
    # momentum 被 IC 抬权 → 综合分上移
    assert weighted["fused_score"] > base["fused_score"]
    assert weighted["fused_score"] == pytest.approx(75.0)


# ----------------------------------------------------------------------
# 未知 / 未列出标签 fail-safe
# ----------------------------------------------------------------------
def test_unknown_tag_handled():
    # 未知标签不在 regime 表里 → 默认权重 1.0，不抛、不放大
    out = fuse_strategies({"mystery": 50.0, "momentum": 50.0})
    assert out["n_strategies"] == 2
    assert out["weights"]["mystery"] == pytest.approx(0.5)
    assert out["fused_score"] == pytest.approx(50.0)


def test_empty_scores_returns_hold():
    out = fuse_strategies({})
    assert out["decision"] == "hold"
    assert out["n_strategies"] == 0
    assert out["fused_score"] == pytest.approx(50.0)
    assert out["contributions"] == {}


# ----------------------------------------------------------------------
# 面向对象的 StrategyEnsemble：复用 RegimeWeighter
# ----------------------------------------------------------------------
def test_strategy_ensemble_class_uses_regime():
    ens = StrategyEnsemble(regime_weighter=RegimeWeighter(regime="bull"))
    out = ens.fuse({"momentum": 50.0, "defensive": 50.0})
    # 牛市：动量权重高于防御
    assert out["contributions"]["momentum"] > out["contributions"]["defensive"]

    ens.set_regime("bear")
    out2 = ens.fuse({"momentum": 50.0, "defensive": 50.0})
    # 熊市：防御权重反超动量
    assert out2["contributions"]["defensive"] > out2["contributions"]["momentum"]


def test_strategy_ensemble_ic_weights_update():
    ens = StrategyEnsemble(
        regime_weighter=RegimeWeighter(regime="sideways"),
        ic_weights={"momentum": 2.0},
    )
    out = ens.fuse({"momentum": 100.0, "defensive": 0.0})
    assert out["fused_score"] == pytest.approx(2.0 / 3.0 * 100.0)
    ens.set_ic_weights({"momentum": 1.0})
    out2 = ens.fuse({"momentum": 100.0, "defensive": 0.0})
    assert out2["fused_score"] == pytest.approx(50.0)

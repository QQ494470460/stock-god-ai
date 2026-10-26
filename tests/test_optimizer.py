# -*- coding: utf-8 -*-
"""组合权重优化器：风险平价 / 最小方差 / 波动率目标 / alpha 倾斜 / 换手封顶。

背景：等权与按评分加权都只看 alpha，不看波动与相关性。
- 等权：一只 25% 年化波动的银行股与一只 70% 的题材股拿一样多的钱，
  组合 8 成风险来自后者 —— 名义分散、实质单押。
- 按评分加权更糟：模型偏爱动量，高分往往就是高波动，等于在波动最大处下最重的注。

这些用例的核心不是「算得对」，而是锁住几条**不变量**：
1. 风险平价真的把风险贡献拉平（不是把权重拉平）；
2. 单票上限是软约束，且不可行时不硬套（否则留下无谓现金）；
3. 任何非法输入都退化为等权，绝不抛异常（fail-safe）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest

from src.trading.optimizer import (
    alpha_tilt,
    cap_turnover,
    explain,
    min_variance_weights,
    optimize_weights,
    portfolio_variance,
    risk_contributions,
    risk_parity_weights,
    vol_target_weights,
    weights_to_dict,
)


# ----------------------------------------------------------------------
# 风险平价
# ----------------------------------------------------------------------
def test_risk_parity_equalizes_risk_contribution_not_weight():
    """波动差 3 倍时，权重必须反比于波动，且风险贡献相等。"""
    cov = np.diag([0.0004, 0.0036])  # 日波动 2% vs 6%
    w = risk_parity_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    # 逆波动：2% 的票拿 3 倍权重
    assert w[0] > w[1]
    assert w[0] == pytest.approx(0.75, abs=1e-3)
    rc = risk_contributions(w, cov)
    assert rc[0] == pytest.approx(rc[1], rel=1e-6)


def test_risk_parity_handles_correlation():
    """有相关性时权重不再是简单逆波动，但风险贡献仍需拉平。"""
    cov = np.array([[0.0004, 0.0003], [0.0003, 0.0009]])
    w = risk_parity_weights(cov)
    rc = risk_contributions(w, cov)
    assert rc[0] == pytest.approx(rc[1], rel=1e-5)
    # 高相关会把高波动那只的权重进一步压低
    assert w[0] + w[1] == pytest.approx(1.0)


def test_risk_parity_respects_cap():
    cov = np.diag([0.0001, 0.0100])  # 极不对称 → 无约束解会想给 0.9+
    raw = risk_parity_weights(cov)
    assert raw.max() > 0.7
    capped = risk_parity_weights(cov, cap=0.6)
    assert capped.max() <= 0.6 + 1e-9
    assert capped.sum() == pytest.approx(1.0)


def test_risk_parity_infeasible_cap_is_ignored():
    """3 只票 cap=0.2 → 顶格也凑不满 100%，放弃 cap 而不是留 40% 无谓现金。"""
    cov = np.diag([0.0004, 0.0004, 0.0004])
    w = risk_parity_weights(cov, cap=0.2)
    assert w.sum() == pytest.approx(1.0)
    assert w.max() > 0.2


def test_risk_parity_degenerate_input_falls_back():
    assert risk_parity_weights(None).sum() == pytest.approx(1.0)
    assert risk_parity_weights([[1.0, 2.0], [3.0, np.nan]]).sum() == pytest.approx(1.0)
    # 全零协方差：无信息 → 等权
    w = risk_parity_weights(np.zeros((2, 2)))
    assert w == pytest.approx([0.5, 0.5])


# ----------------------------------------------------------------------
# 最小方差 / 波动率目标
# ----------------------------------------------------------------------
def test_min_variance_concentrates_on_low_vol_asset():
    cov = np.diag([0.0001, 0.0100])
    w = min_variance_weights(cov)
    assert w[0] > 0.99
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_all_weights_non_negative():
    cov = np.array([[0.0004, -0.0002], [-0.0002, 0.0004]])
    w = min_variance_weights(cov)
    assert np.all(w >= 0)
    assert w.sum() == pytest.approx(1.0)


def test_vol_target_scales_down_when_vol_too_high():
    """组合波动高于目标 → 总权重 < 1（剩下的留现金），这是波动率目标法的本意。"""
    daily = 0.05  # 日波动 5% → 年化 ~78%
    w = vol_target_weights([daily, daily], target_vol=0.15)
    assert w.sum() < 1.0
    assert w.sum() > 0.0


def test_vol_target_never_levers_up():
    """低波动时不会加杠杆：总权重封顶 1.0。"""
    w = vol_target_weights([0.001, 0.001], target_vol=0.90)
    assert w.sum() <= 1.0 + 1e-9


def test_vol_target_inverse_vol_without_target():
    w = vol_target_weights([0.01, 0.03])
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.75, abs=1e-6)


def test_vol_target_missing_vol_gets_zero_weight():
    w = vol_target_weights([0.01, 0.0], target_vol=None)
    assert w[1] == pytest.approx(0.0)
    assert w[0] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# alpha 倾斜 / 换手封顶
# ----------------------------------------------------------------------
def test_alpha_tilt_is_rank_based_not_value_based():
    """99 分与 98 分不该拿到天差地别的权重（抗极值）。"""
    w1 = alpha_tilt([0.5, 0.5], [99, 98], strength=0.5)
    w2 = alpha_tilt([0.5, 0.5], [99, 60], strength=0.5)
    assert w1 == pytest.approx(w2)
    assert w1.sum() == pytest.approx(1.0)


def test_alpha_tilt_strength_zero_is_noop():
    base = [0.6, 0.4]
    out = alpha_tilt(base, [10, 90], strength=0.0)
    assert out == pytest.approx(base)


def test_alpha_tilt_moves_weight_toward_best_score():
    out = alpha_tilt([0.5, 0.5], [90, 60], strength=0.5)
    assert out[0] > out[1]
    # strength=0.5 → 首尾乘子 1.5 / 0.5
    assert out[0] == pytest.approx(0.75, abs=1e-6)


def test_cap_turnover_limits_single_rebalance():
    out = cap_turnover([0.5, 0.5], [0.9, 0.1], max_turnover=0.2)
    turnover = 0.5 * float(np.sum(np.abs(out - np.array([0.5, 0.5]))))
    assert turnover == pytest.approx(0.2, abs=1e-3)
    assert out[0] > 0.5  # 方向正确：往目标走，只是没走完


def test_cap_turnover_noop_when_within_budget():
    out = cap_turnover([0.5, 0.5], [0.55, 0.45], max_turnover=0.5)
    assert out == pytest.approx([0.55, 0.45])


# ----------------------------------------------------------------------
# 一站式入口
# ----------------------------------------------------------------------
def test_optimize_weights_dispatch_and_fallback():
    cov = np.diag([0.0004, 0.0036])
    w = optimize_weights(cov=cov, method="risk_parity")
    assert w.sum() == pytest.approx(1.0)

    # 方法未知 / 数据缺失 → 等权
    w2 = optimize_weights(cov=None, vols=None, method="risk_parity")
    assert w2 == pytest.approx([1.0])

    # risk_parity 但只给 vols → 退回等权（不含 0 权重）
    w3 = optimize_weights(vols=[0.01, 0.02], method="risk_parity")
    assert w3 == pytest.approx([0.5, 0.5])


def test_optimize_weights_tilt_and_turnover_pipeline():
    cov = np.diag([0.0004, 0.0004, 0.0004])
    w = optimize_weights(
        cov=cov,
        scores=[90.0, 60.0, 30.0],
        method="risk_parity",
        tilt_strength=0.5,
        old_weights=[1 / 3, 1 / 3, 1 / 3],
        max_turnover=0.1,
    )
    assert w.sum() == pytest.approx(1.0)
    turnover = 0.5 * float(np.sum(np.abs(w - np.full(3, 1 / 3))))
    assert turnover <= 0.1 + 1e-6
    assert w[0] > w[2]


def test_portfolio_variance_and_explain():
    cov = np.diag([0.0004, 0.0036])
    w = [0.75, 0.25]
    assert portfolio_variance(w, cov) > 0
    rows = explain(w, cov, codes=["A", "B"])
    assert rows[0]["code"] in ("A", "B")
    assert abs(sum(r["risk_pct"] for r in rows) - 1.0) < 1e-6
    assert weights_to_dict(w, ["A", "B"]) == {"A": 0.75, "B": 0.25}

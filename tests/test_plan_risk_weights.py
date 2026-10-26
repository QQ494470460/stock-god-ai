# -*- coding: utf-8 -*-
"""计划层风险加权：risk_parity / min_variance / vol_target 落到 PlanTarget.weight。

为什么值得单独测
----------------
权重是「选股」与「交易」之间唯一的数字接口。它错了，后面所有风控都只是在
给一个错的组合打补丁。这里锁住：

1. 风险加权真的让**低波动标的**拿更多权重，且总权重不超过可投比例；
2. 风险统计覆盖不全时**回退等权**（缺数据的那只往往正是流动性差的那只，
   绝不能给它一个未经风险调整的权重）；
3. ``vol_target`` 允许总权重 < 1（波动高就留现金），其余方法强制归一。
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

from src.trading.trade_plan import PlanTarget, TradePlan, _assign_weights


def _targets(n=3):
    return [PlanTarget(code=f"60000{i}", name=f"票{i}", score=60.0 + i) for i in range(n)]


def _stats(codes, vols, corr=0.0):
    vols = np.asarray(vols, dtype=float)
    n = vols.size
    corr_m = np.full((n, n), corr) + np.eye(n) * (1 - corr)
    cov = np.outer(vols, vols) * corr_m
    return {"codes": list(codes), "cov": cov, "vols": vols}


# ----------------------------------------------------------------------
# _assign_weights
# ----------------------------------------------------------------------
def test_risk_parity_gives_low_vol_more_weight():
    ts = _targets(2)
    # 日波动 2% vs 6%（3 倍）
    _assign_weights(ts, 0.9, "risk_parity", tilt_strength=0.0,
                    risk_stats=_stats(["600000", "600001"], [0.02, 0.06]))
    assert ts[0].weight > ts[1].weight
    assert sum(t.weight for t in ts) == pytest.approx(0.9)
    assert ts[0].weight == pytest.approx(0.9 * 0.75, abs=1e-3)


def test_risk_parity_never_exceeds_investable():
    ts = _targets(3)
    _assign_weights(ts, 0.8, "risk_parity", tilt_strength=0.0,
                    risk_stats=_stats([t.code for t in ts], [0.02, 0.03, 0.05]))
    assert sum(t.weight for t in ts) == pytest.approx(0.8)
    assert all(t.weight > 0 for t in ts)


def test_risk_parity_respects_weight_cap():
    ts = _targets(2)
    _assign_weights(ts, 1.0, "risk_parity", weight_cap=0.55, tilt_strength=0.0,
                    risk_stats=_stats(["600000", "600001"], [0.01, 0.09]))
    assert max(t.weight for t in ts) <= 0.55 + 1e-9


def test_coverage_gap_falls_back_to_equal():
    """风险统计只覆盖 2 只、计划有 3 只 → 必须回退等权，不能给缺数据那只瞎配。"""
    ts = _targets(3)
    _assign_weights(ts, 0.9, "risk_parity",
                    risk_stats=_stats(["600000", "600001"], [0.02, 0.06]))
    assert [t.weight for t in ts] == pytest.approx([0.3, 0.3, 0.3])


def test_missing_stats_falls_back_to_equal():
    ts = _targets(2)
    _assign_weights(ts, 0.8, "risk_parity", risk_stats=None)
    assert [t.weight for t in ts] == pytest.approx([0.4, 0.4])


def test_vol_target_can_leave_cash_uninvested():
    ts = _targets(2)
    stats = _stats(["600000", "600001"], [0.05, 0.05])
    stats["target_vol"] = 0.10  # 远低于组合实际波动（~78% 年化）
    _assign_weights(ts, 1.0, "vol_target", risk_stats=stats)
    total = sum(t.weight for t in ts)
    assert 0.0 < total < 1.0


def test_min_variance_concentrates_on_lowest_vol():
    ts = _targets(2)
    _assign_weights(ts, 1.0, "min_variance", tilt_strength=0.0,
                    risk_stats=_stats(["600000", "600001"], [0.01, 0.08]))
    assert ts[0].weight > 0.9


def test_equal_and_score_still_work():
    ts = _targets(2)
    _assign_weights(ts, 1.0, "equal")
    assert [t.weight for t in ts] == pytest.approx([0.5, 0.5])

    ts2 = _targets(2)
    ts2[0].score, ts2[1].score = 80.0, 20.0
    _assign_weights(ts2, 1.0, "score")
    assert ts2[0].weight == pytest.approx(0.8)


# ----------------------------------------------------------------------
# build_trade_plan 端到端（不写盘）
# ----------------------------------------------------------------------
_PICKS = [
    {"code": "600000", "name": "甲", "score": 80},
    {"code": "600001", "name": "乙", "score": 70},
    {"code": "600002", "name": "丙", "score": 60},
]


def test_build_trade_plan_uses_injected_risk_stats():
    from src.trading.trade_plan import build_trade_plan

    stats = _stats(["600000", "600001", "600002"], [0.01, 0.04, 0.08])
    plan = build_trade_plan(
        picks=_PICKS, plan_date="2026-10-27", weighting="risk_parity",
        risk_stats=stats, cash_reserve_pct=0.10, max_positions=3,
        write=False, weight_cap=1.0,
    )
    assert isinstance(plan, TradePlan)
    ws = [t.weight for t in plan.targets]
    assert sum(ws) == pytest.approx(0.9)
    # 波动最小的拿最多
    assert ws[0] > ws[1] > ws[2]


def test_build_trade_plan_falls_back_when_no_data(monkeypatch):
    from src.trading import trade_plan as tp

    monkeypatch.setattr(tp, "_load_risk_stats", lambda codes, data_dir=None: None)
    plan = tp.build_trade_plan(
        picks=_PICKS, plan_date="2026-10-27", weighting="risk_parity",
        cash_reserve_pct=0.10, max_positions=3, write=False, weight_cap=1.0,
    )
    assert [t.weight for t in plan.targets] == pytest.approx([0.3, 0.3, 0.3])


def test_load_risk_stats_single_code_returns_none():
    from src.trading.trade_plan import _load_risk_stats

    assert _load_risk_stats(["600000"]) is None
    assert _load_risk_stats([]) is None

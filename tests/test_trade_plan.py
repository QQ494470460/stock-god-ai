# -*- coding: utf-8 -*-
"""
交易计划（trade_plan）单元测试
==============================
覆盖：picks 归一化（中文键/英文键/DataFrame）、权重分配（等权/评分加权）、
权重上限截断、持仓上限收口、持久化往返、有效期、差异计算。
零网络、零真实资金。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading.trade_plan import (
    TradePlan, PlanTarget, build_trade_plan, save_trade_plan,
    load_trade_plan, plan_diff, _normalize_picks,
)

DAY = "2026-10-15"


def _picks_cn():
    return [
        {"排名": 1, "代码": "600519", "名称": "贵州茅台", "评分": 82.5, "推荐理由": "动量强"},
        {"排名": 2, "代码": "000001", "名称": "平安银行", "评分": 75.0},
        {"排名": 3, "代码": "300750", "名称": "宁德时代", "评分": 71.2},
    ]


# ---------------------------------------------------------------- 归一化

def test_normalize_chinese_keys():
    rows = _normalize_picks(_picks_cn())
    assert [r["code"] for r in rows] == ["600519", "000001", "300750"]
    assert rows[0]["name"] == "贵州茅台"
    assert rows[0]["score"] == 82.5
    assert rows[0]["reason"] == "动量强"


def test_normalize_english_keys_and_sorting():
    rows = _normalize_picks([
        {"code": "000001", "name": "A", "score": 60},
        {"code": "600519", "name": "B", "score": 90},
    ])
    # 必须按评分降序
    assert [r["code"] for r in rows] == ["600519", "000001"]


def test_normalize_dataframe():
    import pandas as pd
    df = pd.DataFrame(_picks_cn())
    rows = _normalize_picks(df)
    assert len(rows) == 3
    assert rows[0]["code"] == "600519"


def test_normalize_skips_missing_code():
    rows = _normalize_picks([{"名称": "无代码"}, {"代码": "600000", "评分": 50}])
    assert [r["code"] for r in rows] == ["600000"]


def test_normalize_tolerates_garbage():
    assert _normalize_picks(None) == []
    assert _normalize_picks("not-a-list") == []


# ---------------------------------------------------------------- 权重

def test_equal_weights_and_cash_reserve():
    p = build_trade_plan(picks=_picks_cn(), plan_date=DAY, max_positions=5,
                         cash_reserve_pct=0.10, weighting="equal", write=False)
    assert len(p.targets) == 3
    # 等权：可投 90% / 3 = 30%
    assert all(abs(t.weight - 0.30) < 1e-9 for t in p.targets)
    assert abs(sum(t.weight for t in p.targets) - 0.90) < 1e-9


def test_score_weighting_orders_by_score():
    p = build_trade_plan(picks=_picks_cn(), plan_date=DAY, max_positions=5,
                         cash_reserve_pct=0.10, weighting="score", write=False)
    ws = [t.weight for t in p.targets]
    assert ws[0] > ws[1] > ws[2]            # 高分多得
    assert abs(sum(ws) - 0.90) < 1e-9       # 总量仍守住可投比例


def test_weight_cap_clamps_and_keeps_cash():
    """2 只目标 + 30% 上限 → 各自封顶 30%，剩余留现金（更保守）。"""
    p = build_trade_plan(
        picks=[{"code": "600519", "score": 90}, {"code": "000001", "score": 80}],
        plan_date=DAY, max_positions=5, cash_reserve_pct=0.10,
        weighting="equal", weight_cap=0.30, write=False,
    )
    assert all(abs(t.weight - 0.30) < 1e-9 for t in p.targets)
    assert abs(sum(t.weight for t in p.targets) - 0.60) < 1e-9


def test_max_positions_limits_targets():
    picks = [{"code": f"{600000+i}", "score": 90 - i} for i in range(10)]
    p = build_trade_plan(picks=picks, plan_date=DAY, max_positions=4,
                         top_n=None, write=False)
    assert len(p.targets) == 4


def test_top_n_further_limits():
    picks = [{"code": f"{600000+i}", "score": 90 - i} for i in range(10)]
    p = build_trade_plan(picks=picks, plan_date=DAY, max_positions=8,
                         top_n=3, write=False)
    assert len(p.targets) == 3


# ---------------------------------------------------------------- 持久化

def test_save_load_roundtrip(tmp_path):
    p = build_trade_plan(picks=_picks_cn(), plan_date=DAY, max_positions=5,
                         cash_reserve_pct=0.10, write=True, plan_dir=tmp_path)
    assert (tmp_path / f"trade_plan_{DAY.replace('-', '')}.json").exists()

    p2 = load_trade_plan(DAY, plan_dir=tmp_path)
    assert p2 is not None
    assert p2.codes == p.codes
    assert p2.cash_reserve_pct == p.cash_reserve_pct
    assert abs(sum(t.weight for t in p2.targets) - sum(t.weight for t in p.targets)) < 1e-9


def test_load_missing_returns_none(tmp_path):
    assert load_trade_plan("2020-01-01", plan_dir=tmp_path) is None


# ---------------------------------------------------------------- 有效期 / 差异

def test_is_valid_for_exact_date():
    p = TradePlan(plan_date=DAY, targets=[PlanTarget(code="600519")])
    from datetime import date
    assert p.is_valid_for(date(2026, 10, 15))
    assert not p.is_valid_for(date(2026, 10, 16))


def test_is_valid_for_bad_date():
    p = TradePlan(plan_date="", targets=[])
    from datetime import date
    assert not p.is_valid_for(date(2026, 10, 15))
    p2 = TradePlan(plan_date="not-a-date", targets=[])
    assert not p2.is_valid_for(date(2026, 10, 15))


def test_plan_diff():
    p = TradePlan(plan_date=DAY, targets=[
        PlanTarget(code="600519"), PlanTarget(code="000001")])
    d = plan_diff(p, ["000001", "600000"])
    assert d["to_buy"] == ["600519"]
    assert d["to_hold"] == ["000001"]
    assert d["to_sell"] == ["600000"]


def test_empty_plan_summary_no_crash():
    p = build_trade_plan(picks=[], plan_date=DAY, write=False)
    assert p.targets == []
    assert "无目标标的" in p.summary()

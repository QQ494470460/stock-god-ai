"""交易计划「参考价」行为回归测试。

历史 bug：daily_pick 的 report.picks 没有把最新收盘价透传出来，导致
build_trade_plan 生成的计划里 10 只目标 price 全是 0.0，字段名存实亡，
执行侧的「跳空不追」校验根本无从做起。

修复后的契约（与 tests/test_signal_and_plan_hygiene.py 里的
test_trade_plan_carries_reference_price 对齐）：

  · build_trade_plan 的 price **只**来自选股报告（daily_pick 透传的「最新价」/
    price / 收盘价 / close）。报告没给价就保持 0.0 —— **不**去行情兜底，
    因为「缺价=未知」本身就是有用信号，自动编造会让执行侧误以为拿到真实锚点。
  · daily_pick 负责把最新收盘价带进 picks（见 test_daily_pick_price.py）。
  · 历史计划用 backfill_prices() 手动补价（不改动标的/权重）。

这些用例把上面的契约钉死，避免有人再回退。
"""

from __future__ import annotations

import importlib

import pytest

TP = importlib.import_module("src.trading.trade_plan")
build_trade_plan = TP.build_trade_plan
backfill_prices = TP.backfill_prices
load_trade_plan = TP.load_trade_plan


@pytest.fixture
def fake_closes(monkeypatch):
    """让 _fetch_close_prices 走桩，不碰网络（仅 backfill_prices 测试需要）。"""
    prices = {"600177": 8.45, "600901": 7.09, "002966": 9.0}

    def _fake(codes):
        return {c: prices.get(c, 0.0) for c in codes}

    monkeypatch.setattr(TP, "_fetch_close_prices", _fake)
    return prices


def test_build_trade_plan_keeps_report_price():
    """报告里带了价（最新价），计划要原样接住，不能丢。"""
    picks = [
        {"代码": "600177", "名称": "雅戈尔", "评分": 62.5,
         "推荐理由": "个股牛市顺风", "最新价": 8.12},
        {"代码": "600901", "名称": "江苏金租", "评分": 61.1,
         "推荐理由": "个股震荡", "price": 7.09},
    ]
    plan = build_trade_plan(
        picks=picks, pool="zz500", min_score=55.0, regime="bear",
        max_positions=10, cash_reserve_pct=0.1, weighting="equal",
        weight_cap=0.2, write=False,
    )
    by_code = {t.code: t for t in plan.targets}
    # 两种带价写法都接住
    assert by_code["600177"].price == pytest.approx(8.12)
    assert by_code["600901"].price == pytest.approx(7.09)


def test_build_trade_plan_no_backfill_when_missing():
    """报告没带价时，计划 price 必须保持 0.0 —— 不编造，也不去行情兜底。"""
    picks = [
        {"代码": "600177", "名称": "雅戈尔", "评分": 62.5, "推荐理由": "个股顺风"},
        {"代码": "600901", "名称": "江苏金租", "评分": 61.1, "推荐理由": "个股震荡"},
    ]
    plan = build_trade_plan(
        picks=picks, regime="bear", max_positions=10, cash_reserve_pct=0.10,
        weighting="equal", write=False,
    )
    by_code = {t.code: t for t in plan.targets}
    # 缺价 -> 0.0（不编造），但权重照常分配（2 只标的，可投 90% -> 各 45%）
    assert by_code["600177"].price == 0.0
    assert by_code["600901"].price == 0.0
    assert abs(by_code["600177"].weight - 0.45) < 1e-9


def test_backfill_prices_persists(tmp_path, fake_closes):
    """backfill_prices 给历史计划补价，落盘后仍可读到，且不改动标的/权重。"""
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()

    from src.trading.trade_plan import PlanTarget, TradePlan, save_trade_plan

    stale = TradePlan(
        plan_date="2026-10-19", source="daily_pick", pool="zz500",
        targets=[
            PlanTarget(code="600177", name="雅戈尔", score=62.5, price=0.0),
            PlanTarget(code="002966", name="苏州银行", score=57.7, price=0.0),
        ],
    )
    save_trade_plan(stale, plan_dir=plan_dir)

    updated = backfill_prices(plan_date="2026-10-19", plan_dir=plan_dir)
    assert updated is not None
    by_code = {t.code: t for t in updated.targets}
    assert by_code["600177"].price == 8.45
    assert by_code["002966"].price == 9.0

    reloaded = load_trade_plan(plan_date="2026-10-19", plan_dir=plan_dir)
    rb = {t.code: t for t in reloaded.targets}
    assert rb["600177"].price == 8.45
    assert rb["600177"].score == 62.5


def test_backfill_prices_no_plan_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(TP, "_fetch_close_prices", lambda codes: {})
    assert backfill_prices(plan_date="2099-01-01", plan_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# 市场状态（指数级 regime）自适应：熊市降权、牛市放开
# ---------------------------------------------------------------------------

def _make_picks(n: int, price: float = 9.0):
    return [
        {"代码": f"{600000 + i:06d}", "名称": f"标的{i}", "评分": 60.0 - i * 0.1,
         "推荐理由": "个股顺风", "最新价": price}
        for i in range(n)
    ]


def test_build_trade_plan_bear_deweights():
    """熊市 regime **不再**自动收敛仓位（2026-10-31 起停用顺周期覆盖）。

    历史行为是 bear → 6 只 / 20% 现金。停用理由：执行级回测（1140 天，含
    2022 熊市与 2024 年初急跌）始终用固定 8 只 / 10% 现金，年化 +25%；
    实盘自己加的这套「越跌越轻」在回测里不存在，反弹时仓位恰好最轻，
    还会让手动 --build-plan（命中覆盖）与闭环重建（显式传参）产出两个
    不同的组合。regime 现在只做**标注**，不再改动组合参数。
    """
    picks = _make_picks(8)
    plan = build_trade_plan(
        picks=picks, pool="zz500", min_score=55.0, regime="bear",
        weighting="equal", write=False,
    )
    assert plan.regime == "bear", "regime 仍要如实记录，只是不再改仓位"
    assert plan.max_positions == 10                   # 函数默认，未被覆盖
    assert abs(plan.cash_reserve_pct - 0.10) < 1e-9   # 默认现金 10%
    assert len(plan.targets) == 8                     # 候选有多少给多少
    for t in plan.targets:
        assert abs(t.weight - 0.90 / 8) < 1e-9
        assert t.price == 9.0                         # 参考价照常透传


def test_build_trade_plan_bull_expands():
    """牛市 regime 同样不再放开 —— 口径必须与回测一致（见上一用例说明）。"""
    picks = _make_picks(14)
    plan = build_trade_plan(
        picks=picks, pool="zz500", min_score=50.0, regime="bull",
        weighting="equal", write=False,
    )
    assert plan.max_positions == 10
    assert abs(plan.cash_reserve_pct - 0.10) < 1e-9
    assert len(plan.targets) == 10


def test_build_trade_plan_sideways_unchanged():
    """震荡/未知：保持默认 10 只 / 现金 10%。"""
    picks = _make_picks(12)
    plan = build_trade_plan(
        picks=picks, pool="zz500", min_score=50.0, regime="sideways",
        weighting="equal", write=False,
    )
    assert plan.max_positions == 10
    assert abs(plan.cash_reserve_pct - 0.10) < 1e-9
    assert len(plan.targets) == 10


def test_build_trade_plan_explicit_beats_regime():
    """显式传参优先：即使 regime=bear，传了 max_positions/cash 也不被覆盖。"""
    picks = _make_picks(10)
    plan = build_trade_plan(
        picks=picks, pool="zz500", min_score=55.0, regime="bear",
        max_positions=10, cash_reserve_pct=0.10,
        weighting="equal", write=False,
    )
    # 显式指定 -> 熊市覆盖被跳过
    assert plan.max_positions == 10
    assert abs(plan.cash_reserve_pct - 0.10) < 1e-9
    assert len(plan.targets) == 10

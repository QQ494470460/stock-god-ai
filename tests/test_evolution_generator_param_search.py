# -*- coding: utf-8 -*-
"""自我进化 · 生成器 A（交易结果驱动参数寻优）测试。

生成器最容易出的两种错：

* **拍脑袋**：没证据也提改动，把待批队列塞满噪声；
* **乱开方**：症状看得对，但处方方向反了（比如「回撤逼近熔断」
  却去调高单票上限），或者引用了不存在的数字。

因此断言集中在「处方方向正确 + 理由含真实数字 + 无证据不动手」。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evolution import overrides as ov_mod  # noqa: E402
from src.evolution.feedback import PerformanceSnapshot  # noqa: E402
from src.evolution.generators import param_search as ps  # noqa: E402


# ------------------------------------------------------------------ 夹具

CURRENT = {
    "trading_loop.band_pct": 0.25,
    "trading_loop.cash_reserve_pct": 0.10,
    "trading_loop.max_positions": 10,
    "trading_loop.weight_cap": 0.30,
    "trading_loop.weighting": "equal",
    "trading_loop.min_trade_amount": 5000.0,
}


@pytest.fixture
def fixed_current(monkeypatch):
    """固定「当前生效值」，让处方结果可预期。"""
    monkeypatch.setattr(ov_mod, "read_effective",
                        lambda k: CURRENT.get(k))
    return CURRENT


def _snap(**kw) -> PerformanceSnapshot:
    base = dict(
        generated_at="2026-10-14T13:00:00",
        n_verified=10,
        n_settlements=20,
        equity_series=[
            {"date": f"2026-09-{i:02d}", "equity": 100000.0,
             "cash": 10000.0, "n_positions": 10}
            for i in range(1, 21)
        ],
        trade_stats={"fill": 40, "risk_block": 2},
        metrics={"insufficient": False},
    )
    base.update(kw)
    return PerformanceSnapshot(**base)


# ------------------------------------------------------------------ 证据门槛

def test_no_candidate_without_evidence(fixed_current):
    """证据不足 → 一条候选都不提。"""
    snap = _snap(n_verified=0, n_settlements=0, equity_series=[])
    assert ps.generate(snap) == []


def test_no_candidate_without_settlements(fixed_current):
    """有 17 批预测验证、但闭环一天都没结算 → 组合参数无从判断。"""
    snap = _snap(n_verified=17, n_settlements=0, equity_series=[])
    assert ps.generate(snap) == []


# ------------------------------------------------------------------ 症状识别

def test_diagnose_turnover_high():
    snap = _snap(metrics={"insufficient": False, "turnover": 5.2})
    symptoms = {s["symptom"] for s in ps.diagnose(snap)}
    assert "turnover_high" in symptoms


def test_diagnose_cash_drag():
    snap = _snap(equity_series=[
        {"date": f"2026-09-{i:02d}", "equity": 100000.0, "cash": 45000.0,
         "n_positions": 10} for i in range(1, 11)])
    symptoms = {s["symptom"] for s in ps.diagnose(snap)}
    assert "cash_drag" in symptoms


def test_diagnose_positions_underfilled():
    snap = _snap(equity_series=[
        {"date": f"2026-09-{i:02d}", "equity": 100000.0, "cash": 10000.0,
         "n_positions": 3} for i in range(1, 11)])
    symptoms = {s["symptom"] for s in ps.diagnose(snap)}
    assert "positions_underfilled" in symptoms


def test_diagnose_drawdown_near_halt():
    # 熔断线 8%，回撤达 80% 即 6.4%
    snap = _snap(metrics={"insufficient": False, "max_drawdown": -0.07})
    symptoms = {s["symptom"] for s in ps.diagnose(snap)}
    assert "drawdown_near_halt" in symptoms


def test_diagnose_ignores_small_drawdown():
    snap = _snap(metrics={"insufficient": False, "max_drawdown": -0.01})
    symptoms = {s["symptom"] for s in ps.diagnose(snap)}
    assert "drawdown_near_halt" not in symptoms


def test_diagnose_weighting_dilutes_winners():
    snap = _snap(metrics={"insufficient": False, "daily_win_rate": 0.35,
                          "profit_loss_ratio": 2.4})
    symptoms = {s["symptom"] for s in ps.diagnose(snap)}
    assert "weighting_dilutes_winners" in symptoms


def test_diagnose_healthy_system_has_no_symptom():
    """健康系统不该被诊断出病 —— 否则会无端改参数。"""
    snap = _snap(metrics={"insufficient": False, "turnover": 0.8,
                          "max_drawdown": -0.01, "daily_win_rate": 0.55,
                          "profit_loss_ratio": 1.2})
    assert ps.diagnose(snap) == []


# ------------------------------------------------------------------ 处方方向

def test_prescribe_turnover_raises_band(fixed_current):
    p = ps.prescribe("turnover_high", {"detail": "年化换手 5.2 倍"})
    assert p["values"]["trading_loop.band_pct"] == 0.30      # 0.25 → 0.30
    assert "5.2" in p["rationale"]


def test_prescribe_cash_drag_lowers_reserve(fixed_current):
    """现金拖累 → 降低保留比例（更激进方向，必须标注需人工批准）。"""
    p = ps.prescribe("cash_drag", {"detail": "现金占比 45%"})
    assert p["values"]["trading_loop.cash_reserve_pct"] == 0.05
    assert "人工批准" in p["rationale"]


def test_prescribe_underfilled_lowers_target(fixed_current):
    p = ps.prescribe("positions_underfilled", {"detail": "平均持仓 30%"})
    assert p["values"]["trading_loop.max_positions"] == 8


def test_prescribe_drawdown_is_defensive(fixed_current):
    """回撤逼近熔断 → 处方必须是**保守**方向，不许反向加杠杆。"""
    p = ps.prescribe("drawdown_near_halt",
                     {"detail": "回撤 7.0%", "halt_limit": 0.08})
    assert p["values"]["trading_loop.weight_cap"] == 0.25     # 降
    assert p["values"]["trading_loop.cash_reserve_pct"] == 0.15   # 升
    assert "保守方向" in p["rationale"]


def test_prescribe_drawdown_respects_bounds(fixed_current, monkeypatch):
    """已到边界时不能再降 → 不产出越界值。"""
    monkeypatch.setattr(ov_mod, "read_effective", lambda k: {
        "trading_loop.weight_cap": 0.05,
        "trading_loop.cash_reserve_pct": 0.10,
    }.get(k))

    p = ps.prescribe("drawdown_near_halt", {"detail": "回撤 7%"})
    assert p["values"]["trading_loop.weight_cap"] == 0.05      # 已是下限
    assert "trading_loop.cash_reserve_pct" in p["values"]      # 现金仍可升


def test_prescribe_accuracy_low_is_deferred(fixed_current):
    """选股阈值属 evolution.* 命名空间，由阈值闸门管 —— 本生成器不插手。"""
    assert ps.prescribe("accuracy_low", {"detail": "准确率 38%"}) is None


def test_prescribe_unknown_symptom_returns_none(fixed_current):
    assert ps.prescribe("火星来的症状", {}) is None


def test_prescribe_weighting_skips_when_already_score(fixed_current, monkeypatch):
    monkeypatch.setattr(ov_mod, "read_effective",
                        lambda k: "score" if k == "trading_loop.weighting" else None)
    assert ps.prescribe("weighting_dilutes_winners", {"detail": "x"}) is None


# ------------------------------------------------------------------ 端到端

def test_generate_produces_candidate_with_real_old_value(fixed_current):
    snap = _snap(metrics={"insufficient": False, "turnover": 5.2})
    cands = ps.generate(snap)

    assert len(cands) == 1
    c = cands[0]
    assert c.source == "param_search"
    ch = c.changes[0]
    assert ch.key == "trading_loop.band_pct"
    assert ch.old == 0.25 and ch.new == 0.30
    assert c.meta["signature"]


def test_generate_respects_max_candidates(fixed_current):
    snap = _snap(
        metrics={"insufficient": False, "turnover": 5.2, "max_drawdown": -0.07,
                 "daily_win_rate": 0.30, "profit_loss_ratio": 3.0},
        equity_series=[
            {"date": f"2026-09-{i:02d}", "equity": 100000.0, "cash": 45000.0,
             "n_positions": 3} for i in range(1, 11)],
    )
    cands = ps.generate(snap, max_candidates=2)
    assert len(cands) == 2


def test_generate_dedupes_signature(fixed_current):
    """同一症状重复生成 → 指纹一致，便于 runner 跳过重复项。"""
    snap = _snap(metrics={"insufficient": False, "turnover": 5.2})
    a = ps.generate(snap)[0]
    b = ps.generate(snap)[0]
    assert a.meta["signature"] == b.meta["signature"]


def test_generate_never_writes_files(fixed_current):
    """生成器必须是纯函数：不碰 config/ 也不碰 data/。"""
    ov_file = ROOT / "config" / "evolution_overrides.json"
    cand_file = ROOT / "data" / "evolution" / "candidates.json"
    before = (ov_file.exists(), cand_file.exists())

    ps.generate(_snap(metrics={"insufficient": False, "turnover": 5.2}))

    assert (ov_file.exists(), cand_file.exists()) == before

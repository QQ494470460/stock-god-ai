# -*- coding: utf-8 -*-
"""Walk-Forward 样本外验证闸门（防过拟合，Feature #3）单元测试。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import pytest

import src.evolution.gate as gate
from src.evolution.gate import GateCriteria, _walk_forward_verdict
from src.evolution.types import Candidate, Change


def _panels(n_days=200, n_codes=5, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D")
    cols = [f"C{i}" for i in range(n_codes)]
    score = pd.DataFrame(rng.random((n_days, n_codes)), index=idx, columns=cols)
    close = pd.DataFrame(rng.random((n_days, n_codes)) * 100 + 50,
                         index=idx, columns=cols)
    bench = pd.Series(rng.random(n_days) * 100 + 3000, index=idx)
    return score, close, bench


def _fake_run(win_top_n):
    """返回一个可被 monkeypatch 的 _run：top_n==win_top_n 视为「更优模型」。"""
    def _run(sp, cp, b, params):
        win = params.get("top_n") == win_top_n
        return {
            "sharpe": 1.2 if win else 0.5,
            "top_excess": 0.12 if win else 0.0,
            "n_periods": 10,
        }
    return _run


def test_wf_passes_when_candidate_wins_all_folds(monkeypatch):
    score, close, bench = _panels()
    monkeypatch.setattr(gate, "_run", _fake_run(win_top_n=5))
    base = {"top_n": 10}
    cand = {"top_n": 5}  # 候选永远更优
    ok, msg = _walk_forward_verdict(score, close, bench, base, cand,
                                    GateCriteria(require_walk_forward=True))
    assert ok is True
    assert "walk-forward" in msg


def test_wf_fails_when_candidate_never_wins(monkeypatch):
    score, close, bench = _panels()
    monkeypatch.setattr(gate, "_run", _fake_run(win_top_n=10))
    base = {"top_n": 10}   # 基准更优
    cand = {"top_n": 5}    # 候选从不胜出
    ok, msg = _walk_forward_verdict(score, close, bench, base, cand,
                                    GateCriteria(require_walk_forward=True,
                                                 min_wf_folds=3))
    assert ok is False
    assert "walk-forward" in msg


def test_wf_degrades_when_panel_too_short(monkeypatch):
    score, close, bench = _panels(n_days=20)  # 太短，无法切折
    monkeypatch.setattr(gate, "_run", _fake_run(win_top_n=5))
    ok, msg = _walk_forward_verdict(score, close, bench, {"top_n": 10},
                                    {"top_n": 5}, GateCriteria())
    # 面板不足 → 退化为不拦（与旧行为一致，不误杀）
    assert ok is True


def test_wf_degrades_when_runner_returns_none(monkeypatch):
    score, close, bench = _panels()
    monkeypatch.setattr(gate, "_run", lambda sp, cp, b, p: None)
    ok, msg = _walk_forward_verdict(score, close, bench, {"top_n": 10},
                                    {"top_n": 5}, GateCriteria())
    assert ok is True  # 折都跑不出结果 → 不拦


# ---------------------------------------------------------------------------
# 经 evaluate() 真正拦截：开关接通后，过拟合候选必须被拒
# ---------------------------------------------------------------------------

def _cand(changes, source="param_search") -> Candidate:
    return Candidate.create(source=source, title="测试候选", changes=changes)


def _provider(panels):
    return lambda: panels


def _fake_run_overfit(sp, cp, b, params):
    """模拟过拟合：全样本上候选「看起来」更优，但每个样本外折都更差。

    用面板长度区分「全样本」(200 行) 与 walk-forward 的「折切片」(~6 行)：
    全样本走 if 分支（候选胜），折切片走 else 分支（基准胜）。
    """
    full = len(sp) > 50
    cand = params.get("top_n") == 5
    if full:
        sharpe = 1.5 if cand else 0.8
        excess = 0.20 if cand else 0.05
    else:
        sharpe = 0.4 if cand else 1.2
        excess = 0.0 if cand else 0.15
    return {"sharpe": sharpe, "top_excess": excess,
            "n_periods": 12 if full else 6}


def _fake_run_robust(sp, cp, b, params):
    """模拟真稳健：候选在全样本与每个样本外折上都更优。"""
    cand = params.get("top_n") == 5
    sharpe = 1.5 if cand else 0.8
    excess = 0.20 if cand else 0.05
    return {"sharpe": sharpe, "top_excess": excess,
            "n_periods": 12 if len(sp) > 50 else 6}


def test_evaluate_enforces_walk_forward_and_rejects_overfit(monkeypatch):
    """命脉：门禁接通后，A/B 通过但样本外全崩的过拟合候选必须被拒。

    这条用例把「开关」和「真正的拦截」连起来 —— 此前 _walk_forward_verdict
    虽被单测过，但 production 永远传 DEFAULT_CRITERIA（WF 关），等于没接电。
    """
    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 20})
    monkeypatch.setattr(gate, "_run", _fake_run_overfit)

    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 20, 5)]),
        provider=_provider(_panels(n_days=200)),
        criteria=GateCriteria(require_walk_forward=True, min_wf_folds=3))

    assert ev.passed is False
    assert ev.evaluator == "gate:backtest"
    assert any("walk-forward" in r.lower() for r in ev.reasons), ev.reasons
    # 必须同时说明「A/B 通过」与「WF 未通过」，让人知道不是被 A/B 拦的
    assert any("walk-forward 未通过" in r for r in ev.reasons), ev.reasons


def test_evaluate_walk_forward_off_passes_overfit_as_before(monkeypatch):
    """回归护栏：门禁默认关闭时，同一过拟合候选按旧逻辑通过（行为不变）。"""
    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 20})
    monkeypatch.setattr(gate, "_run", _fake_run_overfit)

    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 20, 5)]),
        provider=_provider(_panels(n_days=200)),
        criteria=GateCriteria(require_walk_forward=False))

    assert ev.passed is True, ev.reasons
    # 关闭时原因里不应出现 walk-forward 拦截
    assert not any("walk-forward 未通过" in r for r in ev.reasons), ev.reasons


def test_evaluate_walk_forward_passes_genuinely_robust(monkeypatch):
    """正向：门禁开启且候选真稳健（全样本与折都胜）→ 通过。"""
    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 20})
    monkeypatch.setattr(gate, "_run", _fake_run_robust)

    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 20, 5)]),
        provider=_provider(_panels(n_days=200)),
        criteria=GateCriteria(require_walk_forward=True, min_wf_folds=3))

    assert ev.passed is True, ev.reasons
    assert any("walk-forward" in r.lower() for r in ev.reasons), ev.reasons


def test_load_criteria_missing_file_falls_back_to_default():
    """没有配置文件 → 回退 DEFAULT_CRITERIA（WF 关闭，fail-safe）。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        crit = gate.load_criteria(path=Path(d) / "nope.json")
    assert crit.require_walk_forward is False
    assert crit == gate.DEFAULT_CRITERIA


def test_load_criteria_reads_file():
    """配置文件显式开启 WF → 生效。"""
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "evolution_gate.json"
        p.write_text(json.dumps(
            {"require_walk_forward": True, "min_wf_folds": 4}),
            encoding="utf-8")
        crit = gate.load_criteria(path=p)
    assert crit.require_walk_forward is True
    assert crit.min_wf_folds == 4
    # 其余门槛保持默认
    assert crit.min_sharpe_gain == gate.DEFAULT_CRITERIA.min_sharpe_gain

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

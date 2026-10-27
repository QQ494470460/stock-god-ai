# -*- coding: utf-8 -*-
"""进化闸门 · 第五层「回测-实盘一致性」测试。

这一层存在的理由：前四层全在**模拟**里打转。回测按固定滑点估、按收盘价成交、
按 T+1 开盘执行 —— 任何一处失真，「候选比基准好 0.05 夏普」都可能只是模拟
假设的产物。真实业绩是唯一的外部参照，它必须能一票否决。

而它的失效方式同样致命，两种都要钉死：

* **太松**：背离 3 个夏普还放行 → 在错的坐标系里调参，越调越远；
* **太紧**：4 个交易日的业绩就把候选全毙掉 → 用噪声管噪声，进化永久冻结。

所以本文件的断言一半在「必须拦」，一半在「必须不拦」。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evolution import gate  # noqa: E402
from src.evolution.gate import GateCriteria  # noqa: E402
from src.evolution.types import Candidate, Change  # noqa: E402


# ------------------------------------------------------------------ 造数据

def _perf(**over) -> dict:
    """一份最小可用的业绩报告（结构与 performance.py 的产物一致）。"""
    d = {
        "period": {"start": "2026-10-19", "end": "2026-10-26", "n_days": 40},
        "equity": {"total_return_pct": -4.51, "volatility_pct": 18.55},
        "risk": {"sharpe": -2.0, "max_drawdown_pct": -4.92,
                 "max_dd_recovered": False},
        "trades": {"payoff_ratio": 0.15, "win_rate": 0.31,
                   "max_consecutive_losses": 9, "n_closes": 16},
    }
    for k, v in over.items():
        d[k] = v
    return d


def _write(tmp_path, data, name="perf.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _bt(sharpe):
    return {"sharpe": sharpe, "n_periods": 60}


ON = GateCriteria(require_live_consistency=True, min_live_days=20,
                  max_live_gap_sharpe=2.0)


# ------------------------------------------------------------------ 读取

def test_load_missing_report_returns_none(tmp_path):
    assert gate.load_live_performance(tmp_path / "nope.json") is None


def test_load_corrupt_report_returns_none_not_raise(tmp_path):
    """半截 JSON 不能让进化回路崩掉 —— 进化的失败方式必须是「不动」。"""
    p = tmp_path / "bad.json"
    p.write_text("{ this is not json", encoding="utf-8")
    assert gate.load_live_performance(p) is None


def test_load_non_dict_report_returns_none(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert gate.load_live_performance(p) is None


# ------------------------------------------------------------------ 只报告不拦

def test_missing_report_still_passes(tmp_path):
    ok, reasons = gate._live_consistency_verdict(
        _bt(1.0), ON, path=tmp_path / "nope.json")
    assert ok is True
    assert any("无业绩报告" in r for r in reasons)


def test_summary_written_even_when_switch_off(tmp_path):
    """开关关闭也必须留一行真实业绩 —— 否则事后复盘看不到当时的现实处境。"""
    p = _write(tmp_path, _perf())
    ok, reasons = gate._live_consistency_verdict(
        _bt(1.0), GateCriteria(require_live_consistency=False), path=p)
    assert ok is True
    joined = " ".join(reasons)
    assert "（实盘）" in joined
    assert "-4.51%" in joined
    assert "最长连亏 9 笔" in joined


def test_short_live_sample_does_not_block(tmp_path):
    """4 个交易日的业绩否定不了任何东西 —— 只报告、不拦截。"""
    p = _write(tmp_path, _perf(period={"start": "2026-10-19",
                                       "end": "2026-10-26", "n_days": 4}))
    ok, reasons = gate._live_consistency_verdict(_bt(9.0), ON, path=p)
    assert ok is True
    assert any("不足以判定模拟失真" in r for r in reasons)


def test_missing_sharpe_does_not_block(tmp_path):
    """样本不足 20 日 / 零波动时夏普无定义 —— 不能因此拦住进化。"""
    p = _write(tmp_path, _perf(risk={"sharpe": None,
                                     "max_drawdown_pct": -4.92}))
    ok, reasons = gate._live_consistency_verdict(_bt(1.0), ON, path=p)
    assert ok is True
    assert any("夏普缺失" in r for r in reasons)

    ok2, _ = gate._live_consistency_verdict(_bt(None), ON, path=p)
    assert ok2 is True


# ------------------------------------------------------------------ 必须拦

def test_large_gap_blocks(tmp_path):
    """回测夏普 1.5 vs 真实 -2.0 → 模拟与现实差 3.5 个夏普，A/B 差异不可信。"""
    p = _write(tmp_path, _perf())
    ok, reasons = gate._live_consistency_verdict(_bt(1.5), ON, path=p)
    assert ok is False
    joined = " ".join(reasons)
    assert "回测-实盘背离过大" in joined
    assert "3.50" in joined, joined


def test_gap_exactly_at_cap_passes(tmp_path):
    """边界：背离恰好等于容忍值不算超 —— 阈值是「超过才拦」。"""
    p = _write(tmp_path, _perf())          # 真实夏普 -2.0
    ok, _ = gate._live_consistency_verdict(_bt(0.0), ON, path=p)  # 背离 2.0
    assert ok is True


def test_small_gap_passes(tmp_path):
    p = _write(tmp_path, _perf())
    ok, reasons = gate._live_consistency_verdict(_bt(-1.5), ON, path=p)
    assert ok is True
    assert any("一致性通过" in r for r in reasons)


# ------------------------------------------------------------------ 端到端

def _panels(n_days=80, n_codes=20, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{600000 + i}" for i in range(n_codes)]
    quality = np.linspace(1.0, 0.0, n_codes)
    close = pd.DataFrame(1.0, index=dates, columns=codes)
    for j, c in enumerate(codes):
        r = 0.004 * quality[j] + rng.normal(0, 0.0008, len(dates))
        close[c] = (1.0 + r).cumprod()
    score = pd.DataFrame(
        np.tile(quality, (n_days, 1)), index=dates, columns=codes)
    return score, close, None


def test_evaluate_end_to_end_blocks_on_divergence(monkeypatch, tmp_path):
    """端到端：候选在回测里真的更好，但实盘背离过大 → 仍必须被拒。"""
    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 20})
    p = _write(tmp_path, _perf(risk={"sharpe": -2.0,
                                     "max_drawdown_pct": -4.92}))

    cand = Candidate.create(
        source="param_search", title="减少持仓数",
        changes=[Change("trading_loop.max_positions", 20, 5)])
    ev = gate.evaluate(cand, provider=lambda: _panels(),
                       criteria=ON, live_perf_path=p)

    assert ev.candidate["sharpe"] > ev.baseline["sharpe"], "前提：回测确实变好了"
    assert ev.passed is False, "背离过大却放行了"
    assert any("背离过大" in r for r in ev.reasons)


def test_evaluate_end_to_end_passes_when_consistent(monkeypatch, tmp_path):
    """同样的候选，实盘夏普与回测接近 → 放行，且理由里带真实业绩摘要。

    合成面板波动为 0，回测夏普会退化成三位数（面板构造的已知特性，不是本层
    要考的事）。所以先跑一遍拿到真实回测夏普，再据此造一份「与之接近」的
    实盘业绩 —— 这才是被测语义：**背离小就放行**。
    """
    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 20})
    cand = Candidate.create(
        source="param_search", title="减少持仓数",
        changes=[Change("trading_loop.max_positions", 20, 5)])

    probe = gate.evaluate(cand, provider=lambda: _panels(),
                          criteria=GateCriteria(require_live_consistency=False))
    bt = probe.baseline["sharpe"]
    p = _write(tmp_path, _perf(risk={"sharpe": bt - 0.5,
                                     "max_drawdown_pct": -4.92}))

    ev = gate.evaluate(cand, provider=lambda: _panels(),
                       criteria=ON, live_perf_path=p)

    assert ev.passed is True, ev.reasons
    assert any("（实盘）" in r for r in ev.reasons)
    assert any("一致性通过" in r for r in ev.reasons)


# ------------------------------------------------------------------ 配置接线

def test_load_criteria_reads_live_fields(tmp_path):
    p = tmp_path / "gate.json"
    p.write_text(json.dumps({
        "require_live_consistency": True,
        "max_live_gap_sharpe": 1.5,
        "min_live_days": 30,
    }), encoding="utf-8")
    c = gate.load_criteria(p)
    assert c.require_live_consistency is True
    assert c.max_live_gap_sharpe == pytest.approx(1.5)
    assert c.min_live_days == 30


def test_production_config_arms_the_gate():
    """生产配置必须真的把这一层打开，否则等于没做。"""
    c = gate.load_criteria(ROOT / "config" / "evolution_gate.json")
    assert c.require_live_consistency is True
    assert c.min_live_days >= 20

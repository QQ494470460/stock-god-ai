# -*- coding: utf-8 -*-
"""多策略投票融合（StrategyMode.ENSEMBLE）。

背景
----
改造前 `QuantSignalGenerator.generate()` 是**三选一**：按 mode 走
`_multi_factor` / `_mean_rev` / `_hybrid` 其中一个，三者互斥。
等于把赌注压在「当前是哪种市」上——趋势市多因子强、震荡市均值回归强，
选错了整条链路就系统性跑偏。

ENSEMBLE 让三个策略同时投票，方向一致才下手，一致度即置信度。

全部用 monkeypatch 注入策略返回值，不依赖真实行情与模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.trading.signals import QuantSignal, QuantSignalGenerator, StrategyMode  # noqa: E402


def _gen() -> QuantSignalGenerator:
    return QuantSignalGenerator(mode=StrategyMode.ENSEMBLE)


def _sig(signal: int, score: float = 60.0, agreement: float = 0.8,
         strategy: str = "x", reasons=None) -> QuantSignal:
    return QuantSignal(
        code="600000", name="测试", score=score, agreement=agreement,
        signal=signal, strategy=strategy, reasons=reasons or ["理由"],
    )


def _df() -> pd.DataFrame:
    return pd.DataFrame({"close": [10.0 + i for i in range(80)]})


def _bind(g: QuantSignalGenerator, multi=None, mr=None, hybrid=None):
    """绑定三个策略的返回值；None 表示该腿抛异常。"""
    def _mk(ret):
        if ret is None:
            def _raise(*_a, **_k):
                raise RuntimeError("策略腿不可用（模拟 AI 缺模型 / 数据不足）")
            return _raise
        return lambda *_a, **_k: ret

    g._multi_factor_signal = _mk(multi)
    g._mean_rev_signal = _mk(mr)
    g._hybrid_signal = _mk(hybrid)


def test_unanimous_buy_passes_with_full_agreement():
    g = _gen()
    _bind(g, _sig(1, 70), _sig(1, 60), _sig(1, 65))

    r = g.generate_ensemble(_df(), "600000", "测试")

    assert r is not None
    assert r.signal == 1
    assert r.agreement == 1.0, "三票全同 → 一致度 1.0"
    assert r.source == "ensemble"
    assert r.quant_details["votes"] == {"buy": 3, "sell": 0, "total": 3}


def test_majority_wins_and_agreement_reflects_ratio():
    g = _gen()
    _bind(g, _sig(1, 70), _sig(1, 60), _sig(-1, 65))

    r = g.generate_ensemble(_df(), "600000", "测试")

    assert r is not None
    assert r.signal == 1, "2 买 1 卖应取买入"
    assert r.agreement == pytest.approx(2 / 3, abs=1e-3)
    assert r.quant_details["votes"]["sell"] == 1


def test_fully_split_vote_abstains():
    """买卖票数相等 → 不下手。分歧本身就是不确定性的信号。"""
    g = _gen()
    _bind(g, _sig(1, 70), _sig(-1, 60), None)  # 第三腿挂掉，剩 1:1

    r = g.generate_ensemble(_df(), "600000", "测试")

    assert r is None, "1 买 1 卖不应硬选方向"


def test_broken_leg_does_not_kill_the_stock():
    """单个策略抛异常时，其余策略照常投票。"""
    g = _gen()
    _bind(g, _sig(1, 70), None, _sig(1, 65))  # 均值回归腿抛异常

    r = g.generate_ensemble(_df(), "600000", "测试")

    assert r is not None
    assert r.signal == 1
    assert r.agreement == 1.0
    assert "mean_rev" not in r.strategy
    assert r.quant_details["votes"]["total"] == 2


def test_no_leg_produces_signal_returns_none():
    g = _gen()
    _bind(g, _sig(0, 70), _sig(0, 60), _sig(0, 65))

    assert g.generate_ensemble(_df(), "600000", "测试") is None


def test_score_is_confidence_weighted_not_plain_average():
    """高分低置信度不应压过高分高置信度。"""
    g = _gen()
    _bind(g, _sig(1, score=90.0, agreement=0.1),
             _sig(1, score=60.0, agreement=0.9),
             None)

    r = g.generate_ensemble(_df(), "600000", "测试")
    plain_avg = (90.0 + 60.0) / 2

    assert r is not None
    assert r.score < plain_avg, "置信度加权后应低于算术平均（弱票不该拉高）"
    assert r.score > 60.0


def test_generate_dispatches_to_ensemble_mode():
    """mode=ENSEMBLE 时 generate() 必须走融合分支。"""
    g = _gen()
    _bind(g, _sig(1, 70), _sig(1, 60), _sig(1, 65))

    class _FakeFetcher:
        @staticmethod
        def get_daily_kline(_code):
            return _df()

    g._fetcher = _FakeFetcher()
    g._name_map = {"600000": "测试"}

    r = g.generate("600000")

    assert r is not None
    assert r.source == "ensemble"


# ---------------------------------------------------------------- 策略信誉加权

def test_unknown_tag_credibility_is_neutral():
    """没观测过的策略按中性权重 1.0 —— 启用信誉不会改变冷启动行为。"""
    g = _gen()
    assert g.credibility.weight("never_seen") == pytest.approx(1.0)


def test_observe_outcome_raises_weight_of_accurate_strategy():
    g = _gen()
    base = g.credibility.weight("x")
    for _ in range(200):
        g.observe_outcome("x", 1.0, 0.02)  # 看多且赚 → 命中

    assert g.credibility.weight("x") > base + 0.3


def test_credibility_shifts_vote_toward_track_record():
    """分数高的策略若战绩差，话语权应当下降。

    multi_factor 给 90 分但连错，mean_rev 给 50 分但连对 →
    融合分应明显偏向 mean_rev，而不是被高分绑架。
    """
    g = _gen()
    _bind(g, _sig(1, score=90.0, agreement=0.5),
             _sig(1, score=50.0, agreement=0.5), None)

    before = g.generate_ensemble(_df(), "600000", "测试")
    assert before.score == pytest.approx(70.0, abs=0.5), "冷启动应等权（信誉均 1.0）"

    for _ in range(200):
        g.observe_outcome("mean_rev", 1.0, 0.02)        # 看多且赚
        g.observe_outcome("multi_factor", 1.0, -0.02)   # 看多却亏

    after = g.generate_ensemble(_df(), "600000", "测试")

    assert after.score < before.score - 5, "战绩差的 90 分策略不该继续主导"
    cred = after.quant_details["leg_credibility"]
    assert cred["mean_rev"] > cred["multi_factor"]

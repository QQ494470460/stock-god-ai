# -*- coding: utf-8 -*-
"""多策略复核层：`src.core.daily_pick._apply_ensemble_vote`。

定位
----
主选股走 `predictor.get_top_recommendations()`（AI 打分），与
`src.trading.signals` 的量化三策略是两条独立链路。本层把后者接到主航道
**之后**，只做「否决 + 加权」，不替换 AI 打分。

三条不误伤的兜底是本文件的重点 —— 复核层的风险不在于算错，
而在于**把好候选误杀掉，或一次性把推荐清空**。

全部用 monkeypatch 注入投票结果，不依赖真实行情与模型。
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

from src.core.daily_pick import _apply_ensemble_vote  # noqa: E402
from src.trading.signals import QuantSignal, QuantSignalGenerator  # noqa: E402


def _kline(n: int = 80):
    return pd.DataFrame({"close": [10.0 + i for i in range(n)]})


def _cand(rows):
    return pd.DataFrame(rows, columns=["代码", "评分"])


def _review(df, data):
    """复核后只取候选表（统计另行断言）"""
    out, _ = _apply_ensemble_vote(df, data)
    return out


@pytest.fixture
def votes(monkeypatch):
    """按代码注入投票结果：1=一致看多，-1=集体看空，0=无共识/无信号。"""
    table: dict = {}

    def _fake(self, df, code, name):
        v = table.get(str(code), 0)
        if v == 0:
            return None
        return QuantSignal(code=code, name=name, score=60.0, agreement=1.0,
                           signal=v, strategy="t")

    monkeypatch.setattr(QuantSignalGenerator, "generate_ensemble", _fake)
    return table


def test_collective_bearish_vetoes_candidate(votes):
    votes.update({"600001": -1, "600002": 1})
    df = _cand([["600001", 70.0], ["600002", 65.0]])
    data = {"600001": _kline(), "600002": _kline()}

    out = _review(df, data)

    assert list(out["代码"]) == ["600002"], "集体看空的应被否决"


def test_unanimous_bullish_raises_score(votes):
    votes["600001"] = 1
    df = _cand([["600001", 60.0]])
    data = {"600001": _kline()}

    out = _review(df, data)

    # 一致度 1.0 → 加成系数 0.9 + 0.2*1.0 = 1.1
    assert out.iloc[0]["评分"] == pytest.approx(66.0)


def test_insufficient_data_is_kept_not_penalized(votes):
    """拿不到足够 K 线 → 保留，绝不因为数据缺失而误杀。"""
    votes["600001"] = 1
    df = _cand([["600001", 60.0]])
    data = {"600001": _kline(30)}   # 不足 60 根

    out = _review(df, data)

    assert len(out) == 1
    assert out.iloc[0]["评分"] == 60.0, "数据不足应原样保留，不提分也不否决"


def test_no_consensus_is_kept_unchanged(votes):
    """策略分歧（无信号）→ 既不否决也不提分。"""
    votes["600001"] = 0
    df = _cand([["600001", 60.0]])
    data = {"600001": _kline()}

    out = _review(df, data)

    assert len(out) == 1
    assert out.iloc[0]["评分"] == 60.0


def test_all_vetoed_returns_original_instead_of_empty(votes):
    """全部被否决 → 返回原始候选，绝不让推荐被清空。

    与 rebalance「空计划=按兵不动、绝不误会成清仓」是同一条原则：
    一次数据抖动不该让今日推荐归零。
    """
    votes.update({"600001": -1, "600002": -1})
    df = _cand([["600001", 70.0], ["600002", 65.0]])
    data = {"600001": _kline(), "600002": _kline()}

    out, st = _apply_ensemble_vote(df, data)

    assert len(out) == 2, "全票否决时不能返回空表"
    assert list(out["代码"]) == ["600001", "600002"]
    assert st["fallback"] is True
    assert st["vetoed"] == 2, "回退也要如实记录被否决数，不能显示 0"


def test_fallback_is_distinguishable_from_nothing_vetoed(votes):
    """20→20 有两种可能：一票没否决，或全票否决后回退。必须能区分。"""
    votes.update({"600001": 1, "600002": 1})
    df = _cand([["600001", 70.0], ["600002", 65.0]])
    data = {"600001": _kline(), "600002": _kline()}

    _, st = _apply_ensemble_vote(df, data)

    assert st["before"] == st["after"] == 2
    assert st["vetoed"] == 0
    assert st["fallback"] is False
    assert st["boosted"] == 2


def test_skipped_candidates_are_counted_not_confused_with_kept(votes):
    """数据不足被跳过的，单独计数 —— 否则「复核覆盖了几只」永远是笔糊涂账。"""
    votes.update({"600001": 1, "600002": -1})
    df = _cand([["600001", 70.0], ["600002", 65.0], ["600003", 60.0]])
    data = {"600001": _kline(), "600002": _kline(), "600003": _kline(10)}

    out, st = _apply_ensemble_vote(df, data)

    assert len(out) == 2 and st["skipped"] == 1 and st["vetoed"] == 1


def test_kept_candidates_are_re_ranked_by_score(votes):
    votes.update({"600001": 1, "600002": 0})
    df = _cand([["600001", 60.0], ["600002", 65.0]])
    data = {"600001": _kline(), "600002": _kline()}

    out = _review(df, data)

    # 600001 提分到 66，应反超未提分的 600002(65)
    assert list(out["代码"]) == ["600001", "600002"]


def test_empty_frame_is_safe(votes):
    df = pd.DataFrame(columns=["代码", "评分"])
    assert len(_review(df, {})) == 0

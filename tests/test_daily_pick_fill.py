# -*- coding: utf-8 -*-
"""选股门槛：达标优先 + 排名补足（不再一票否决）。

背景（实测数据，2026-10-26 / 10-27）：

    池 = zz500（扫描 425 只），min_score 被 regime 从 50 抬到 55。
    10-26 通过 4 只、10-27 通过 6 只 —— 组合只有 4~6 只票，而回测假设是 8 只。

两层后果：

1. **组合被动集中**：4 只票单票权重翻倍，波动与回撤都被放大。回测的
   +25%/年、-25% 回撤按 8 只算，实盘根本不是同一个组合，数字不可比。
2. **顺周期开关**：越跌 → 全市场分数越低 → 通过越少 → 仓位越轻，反弹时
   恰好仓位最轻。回测里没有这个机制（永远按排名取前 N）。

修法：达标的照常优先；不足目标只数时按评分从高到低补足，但补位票不得低于
硬底线 ``THRESHOLD_MIN``（30），避免把明显不合格的标的塞进组合。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

from src.core.daily_pick import _apply_score_threshold
from src.models.evolution import THRESHOLD_MIN


def _df(scores):
    return pd.DataFrame({
        "代码": [f"S{i:02d}" for i in range(len(scores))],
        "评分": list(scores),
    })


def test_fills_up_to_target_when_few_pass():
    """只有 2 只达标、目标 4 只 ⇒ 补足到 4 只。"""
    report = {}
    out = _apply_score_threshold(_df([70.0, 58.0, 52.0, 48.0, 33.0]), 55.0, 4, report)

    assert list(out["代码"]) == ["S00", "S01", "S02", "S03"]
    assert report["n_passed_strict"] == 2
    assert report["n_filled"] == 2
    assert "补足" in report["fill_note"]


def test_no_fill_when_enough_pass():
    """达标数已够 ⇒ 一只都不补，行为与旧版完全一致。"""
    report = {}
    out = _apply_score_threshold(_df([70.0, 60.0, 56.0, 40.0]), 55.0, 2, report)

    assert list(out["代码"]) == ["S00", "S01", "S02"]
    assert report["n_passed_strict"] == 3
    assert report["n_filled"] == 0
    assert "fill_note" not in report


def test_fill_respects_hard_floor():
    """补位票不得低于硬底线 30 —— 垃圾票不能被凑数塞进组合。"""
    below = THRESHOLD_MIN - 5.0
    out = _apply_score_threshold(_df([70.0, below, below - 5]), 75.0, 3, {})

    assert list(out["代码"]) == ["S00"], "低于底线的票一只都不该补进来"


def test_fill_stops_at_available_candidates():
    """候选池不够目标数时就给多少算多少，不报错、不造票。"""
    report = {}
    out = _apply_score_threshold(_df([70.0, 40.0]), 55.0, 8, report)

    assert len(out) == 2
    assert report["n_filled"] == 1


def test_keeps_descending_score_order():
    out = _apply_score_threshold(_df([70.0, 58.0, 52.0, 48.0]), 55.0, 4, {})
    scores = list(out["评分"])
    assert scores == sorted(scores, reverse=True)


def test_empty_and_malformed_inputs_are_passthrough():
    empty = pd.DataFrame()
    assert _apply_score_threshold(empty, 55.0, 8, {}) is empty

    no_score_col = pd.DataFrame({"代码": ["S00"], "其他": [1]})
    assert _apply_score_threshold(no_score_col, 55.0, 8, {}) is no_score_col

    assert _apply_score_threshold(None, 55.0, 8, {}) is None


def test_nan_score_treated_as_zero_not_crash():
    df = pd.DataFrame({"代码": ["S00", "S01", "S02"], "评分": [70.0, None, 45.0]})
    out = _apply_score_threshold(df, 55.0, 3, {})
    # NaN 视为 0 ⇒ 低于底线，不参与补位
    assert list(out["代码"]) == ["S00", "S02"]

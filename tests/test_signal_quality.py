# -*- coding: utf-8 -*-
"""信号质量（截面中性化 + 集成主腿尺度）离线单测。

不联网、不加载任何模型 / xgboost / sklearn，纯函数 + 轻量打桩。
"""
import numpy as np
import pandas as pd
import pytest

from src.models.signal_quality import neutralize_cross_section, cross_sectional_zscore
from src.models.predictor import predictor, StockPrediction
from src.models.ensemble import EnsemblePredictor
import src.data.fetcher as fetcher_mod
from config.settings import config as app_config


# ---------------------------------------------------------------------------
# 1) neutralize_cross_section 纯函数
# ---------------------------------------------------------------------------

def test_neutralize_removes_industry_bias():
    """两组行业原本整体水位差很多，中性化后组均值应被拉平到接近总均值。"""
    scores = pd.Series({
        "A1": 90.0, "A2": 80.0,          # 行业甲：整体偏高
        "B1": 20.0, "B2": 10.0,          # 行业乙：整体偏低
    })
    ind_map = {"A1": "ind甲", "A2": "ind甲", "B1": "ind乙", "B2": "ind乙"}
    neu = neutralize_cross_section(scores, ind_map)
    # 组内去均值后，两组的均值应相等（= 总均值 50）
    grp = neu.groupby(pd.Series(ind_map)).mean()
    assert abs(grp["ind甲"] - grp["ind乙"]) < 1e-6
    assert abs(grp.mean() - 50.0) < 1e-6


def test_neutralize_preserves_within_group_order():
    """组内去均值只平移、不改组内相对序。"""
    scores = pd.Series({"A1": 100.0, "A2": 60.0, "A3": 40.0})
    ind_map = {"A1": "X", "A2": "X", "A3": "X"}
    neu = neutralize_cross_section(scores, ind_map)
    # 平移常数 = mean(100,60,40)=66.67；残差 = [33.33, -6.67, -26.67]，序保持
    assert neu["A1"] > neu["A2"] > neu["A3"]


def test_neutralize_does_not_collapse_singleton_industry():
    """只含 1 只股票的行业不能被拉平成总均值，应保留原分。"""
    scores = pd.Series({"A1": 95.0, "B1": 5.0})
    # A1 自己是唯一行业，B1 也是唯一行业 → 都被判为小组，原分保留
    ind_map = {"A1": "onlyA", "B1": "onlyB"}
    neu = neutralize_cross_section(scores, ind_map)
    assert abs(neu["A1"] - 95.0) < 1e-9
    assert abs(neu["B1"] - 5.0) < 1e-9


def test_neutralize_keeps_grand_mean_level():
    """中性化不改变整体均值水平（只剔行业水位）。"""
    scores = pd.Series({"A1": 90.0, "A2": 70.0, "B1": 30.0, "B2": 10.0})
    ind_map = {"A1": "甲", "A2": "甲", "B1": "乙", "B2": "乙"}
    neu = neutralize_cross_section(scores, ind_map)
    assert abs(neu.mean() - scores.mean()) < 1e-9


def test_neutralize_empty_map_returns_original():
    """无行业映射时原样返回，绝不抛错。"""
    scores = pd.Series({"A1": 90.0, "B1": 10.0})
    neu = neutralize_cross_section(scores, {})
    assert neu.equals(scores)
    neu2 = neutralize_cross_section(scores, None)
    assert neu2.equals(scores)


def test_zscore_basic():
    z = cross_sectional_zscore(pd.Series([1.0, 2.0, 3.0]))
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 2) predictor._neutralize_scores 接入
# ---------------------------------------------------------------------------

def _make_pred(code, score):
    return StockPrediction(code=code, name=code, prob_up=0.0, prob_neutral=0.0, prob_down=0.0,
                           score=float(score), score_raw=float(score))


def test_predictor_neutralize_scores_applies():
    """predict_batch 内的中性化经 _neutralize_scores 生效：组均值被拉平。"""
    old_flag = app_config.ml.cross_sectional_neutralize
    old_cache = predictor._industry_map_cache
    fake_map = {"A1": "甲", "A2": "甲", "B1": "乙", "B2": "乙"}
    orig = fetcher_mod.fetcher.get_industry_map
    try:
        app_config.ml.cross_sectional_neutralize = True
        predictor._industry_map_cache = None
        fetcher_mod.fetcher.get_industry_map = lambda *a, **k: fake_map
        preds = [_make_pred(c, s) for c, s in
                 [("A1", 90.0), ("A2", 80.0), ("B1", 20.0), ("B2", 10.0)]]
        predictor._neutralize_scores(preds)
        by_code = {p.code: p.score for p in preds}
        mean_甲 = (by_code["A1"] + by_code["A2"]) / 2
        mean_乙 = (by_code["B1"] + by_code["B2"]) / 2
        assert abs(mean_甲 - mean_乙) < 1e-6
    finally:
        app_config.ml.cross_sectional_neutralize = old_flag
        predictor._industry_map_cache = old_cache
        fetcher_mod.fetcher.get_industry_map = orig


def test_predictor_neutralize_disabled_is_noop():
    old_flag = app_config.ml.cross_sectional_neutralize
    old_cache = predictor._industry_map_cache
    orig = fetcher_mod.fetcher.get_industry_map
    try:
        app_config.ml.cross_sectional_neutralize = False
        predictor._industry_map_cache = None
        fetcher_mod.fetcher.get_industry_map = lambda *a, **k: {"A1": "甲", "B1": "乙"}
        preds = [_make_pred("A1", 90.0), _make_pred("B1", 10.0)]
        predictor._neutralize_scores(preds)
        assert preds[0].score == 90.0 and preds[1].score == 10.0
    finally:
        app_config.ml.cross_sectional_neutralize = old_flag
        predictor._industry_map_cache = old_cache
        fetcher_mod.fetcher.get_industry_map = orig


# ---------------------------------------------------------------------------
# 3) 集成主腿尺度一致性（用 score_raw 而非 score 融合）
# ---------------------------------------------------------------------------

class _FakePrimary:
    def predict_single(self, df, code="000000", name="未知"):
        # score 含 regime/risk 加成（被夸大），score_raw 才是纯概率分
        return StockPrediction(code=code, name=name, prob_up=0.0, prob_neutral=0.0,
                               prob_down=0.0, score=100.0, score_raw=50.0)


class _FakeSub:
    def __init__(self, name, score):
        self.name = name
        self.available = True
        self._score = score

    def score_for(self, df, code):
        return float(self._score)


def test_ensemble_blends_on_score_raw():
    """集成主腿必须用 score_raw 融合：blend 应在 (50,30) 而非 (100,30)。"""
    ep = EnsemblePredictor(_FakePrimary())
    ep.add_sub(_FakeSub("alpha_xgboost", 30.0), weight=1.0)
    res = ep.predict_batch({"AAA": pd.DataFrame({"close": [1, 2, 3]})})
    assert "AAA" in res
    blended = res["AAA"].score
    # 真值（修复后）= average([50, 30], weights=[1,1]) = 40
    assert abs(blended - 40.0) < 1e-6
    # 若错误地用了 score(=100)，会是 65；这里断言它一定不等于 65
    assert abs(blended - 65.0) > 1e-6

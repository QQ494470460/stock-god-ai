# -*- coding: utf-8 -*-
"""超级集成单元测试（离线，用 Fake 子预测器，零网络/零真实模型）"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models.ensemble import EnsemblePredictor, SubPredictor, _proba_to_score  # noqa: E402


class FakePred:
    def __init__(self, score):
        self.score = score
        self.code = ""
        self.name = ""
        self.xgb_signal = 1
        self.agreement = 1.0


class FakePrimary:
    def __init__(self, score=60.0):
        self.score = score

    def predict_single(self, df, code="x", name="x"):
        p = FakePred(self.score)
        p.code = code
        p.name = name
        return p


class FakeSub(SubPredictor):
    name = "fake"

    def __init__(self, score):
        self._val = score

    def _ensure(self):
        pass

    def _score(self, df, code):
        return self._val


class UnavailableSub(SubPredictor):
    name = "bad"

    def _ensure(self):
        raise RuntimeError("model file missing")


def _df():
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]})


def test_blend_weighted_average():
    ep = EnsemblePredictor(FakePrimary(60.0))
    ep.add_sub(FakeSub(90.0), weight=1.0)
    out = ep.predict_batch({"000001": _df()})
    p = out["000001"]
    assert abs(p.score - 75.0) < 1e-6
    assert abs(p.ensemble_detail["spread"] - 30.0) < 1e-6
    assert p.agreement >= 0.8  # 不破坏交易执行器硬闸门


def test_weight_respected():
    ep = EnsemblePredictor(FakePrimary(0.0))
    ep.add_sub(FakeSub(100.0), weight=3.0)  # 权重 3:1 → 75
    out = ep.predict_batch({"a": _df()})
    assert abs(out["a"].score - 75.0) < 1e-6


def test_graceful_degradation_no_subs():
    ep = EnsemblePredictor(FakePrimary(42.0))
    out = ep.predict_batch({"a": _df()})
    assert out["a"].score == 42.0
    assert not hasattr(out["a"], "ensemble_detail") or out["a"].ensemble_detail is None


def test_unavailable_sub_skipped():
    ep = EnsemblePredictor(FakePrimary(60.0))
    ep.add_sub(UnavailableSub(), weight=1.0)
    out = ep.predict_batch({"a": _df()})
    # 唯一子模型不可用 → 回退主模型分
    assert out["a"].score == 60.0


def test_proba_to_score_5class():
    s = _proba_to_score(np.array([0.1, 0.1, 0.2, 0.3, 0.3]), [0, 1, 2, 3, 4])
    # prob_up=0.6, prob_down=0.2 -> (0.6*0.7 - 0.2*0.3 + 0.3)*100 = 66.0
    assert abs(s - 66.0) < 1e-6


def test_proba_to_score_3class():
    s = _proba_to_score(np.array([0.2, 0.3, 0.5]), [-1, 0, 1])
    # prob_up=0.5, prob_down=0.2 -> (0.35 - 0.06 + 0.3)*100 = 59.0
    assert abs(s - 59.0) < 1e-6

# -*- coding: utf-8 -*-
"""个股分析页口径统一：predict_single 接入超级集成（离线，Fake 组件，零模型/零网络）。

背景（2026-10-18 用户问「单模型可靠吗」时查证）：
- `predict_single` 原先**全程只用 XGBoost**，`agreement` 恒 1.0，从不调 `_get_ensemble()`；
  只有 `predict_batch`（选股）走三腿融合 → 个股分析页与选股**口径不一致**。
- 而单模型实测 **mean IC 0.0575 / t=1.40（统计不显著）**，集成才是 t≈4.6 的那个。
  用户会把不显著的单模型分误当成集成分来下单。

本测试钉死改造后的契约：
1. `blend_one` 成功融合 → `model_mode == "ensemble"`；
2. 无可用腿 → 保留单模型分、`model_mode` 仍为 "single"；
3. **`predict_batch` 不得双重融合**（它内部会调 primary.predict_single，
   故 predict_single 的 apply_ensemble 必须默认 False）；
4. `Predictor.predict_single` 确实提供 `apply_ensemble` 且默认为 False。
"""
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models.ensemble import EnsemblePredictor, SubPredictor  # noqa: E402
from config.settings import config as app_config  # noqa: E402


@pytest.fixture(autouse=True)
def _pin_no_shrinkage():
    old = getattr(app_config.ml, "ensemble_shrinkage", 0.0)
    app_config.ml.ensemble_shrinkage = 0.0
    try:
        yield
    finally:
        app_config.ml.ensemble_shrinkage = old


class FakePred:
    """最小 StockPrediction 替身：带 score_raw（blend_one 用 score_raw 作主腿分）。"""

    def __init__(self, score):
        self.score = score
        self.score_raw = score
        self.code = ""
        self.name = ""
        self.xgb_signal = 1
        self.agreement = 1.0
        self.model_mode = "single"
        self.ensemble_detail = {}


class FakePrimary:
    def __init__(self, score=60.0):
        self.score = score
        self.calls = 0

    def predict_single(self, df, code="x", name="x"):
        self.calls += 1
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


def _df():
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]})


# --------------------------------------------------------------------------
# 1) 融合成功 → 标成 ensemble
# --------------------------------------------------------------------------
def test_blend_one_marks_ensemble_and_blends_score():
    p = FakePred(60.0)
    ep = EnsemblePredictor(FakePrimary(60.0))
    ep.add_sub(FakeSub(80.0), weight=1.0)

    ok = ep.blend_one(p, _df(), "000001")
    assert ok is True
    assert p.model_mode == "ensemble", "融合成功必须标成 ensemble，供 UI 标注口径"
    assert abs(p.score - 70.0) < 1e-6          # (60 主 + 80 子) 等权
    assert p.agreement >= 0.8                  # 不破坏交易执行器硬闸门
    assert p.ensemble_detail["scores"]["fake"] == 80.0


# --------------------------------------------------------------------------
# 2) 无可用腿 → 保留单模型分，绝不伪装成 ensemble
# --------------------------------------------------------------------------
def test_blend_one_without_subs_keeps_single_mode():
    p = FakePred(42.0)
    ep = EnsemblePredictor(FakePrimary(42.0))   # 不挂任何子模型

    ok = ep.blend_one(p, _df(), "000001")
    assert ok is False
    assert p.score == 42.0
    assert p.model_mode == "single", "没融合就不能标 ensemble，否则 UI 会误导"
    assert p.ensemble_detail == {}


# --------------------------------------------------------------------------
# 3) 关键：predict_batch 不得双重融合
# --------------------------------------------------------------------------
def test_predict_batch_blends_exactly_once():
    """predict_batch 内部先调 primary.predict_single（其 apply_ensemble 默认 False），
    再调 blend_one。若 predict_single 默认 True，这里就会融两次 → 分数失真。
    期望：主 60 + 子 90 等权 = 75（而非再融一次后的 82.5）。
    """
    primary = FakePrimary(60.0)
    ep = EnsemblePredictor(primary)
    ep.add_sub(FakeSub(90.0), weight=1.0)

    out = ep.predict_batch({"a": _df()})
    assert abs(out["a"].score - 75.0) < 1e-6, "发生了双重融合"
    assert out["a"].model_mode == "ensemble"
    assert primary.calls == 1


# --------------------------------------------------------------------------
# 4) Predictor.predict_single 必须提供 apply_ensemble，且默认 False
# --------------------------------------------------------------------------
def test_predict_single_exposes_apply_ensemble_defaulting_false():
    from src.models.predictor import Predictor

    sig = inspect.signature(Predictor.predict_single)
    assert "apply_ensemble" in sig.parameters, "个股分析页需要显式开启集成"
    assert sig.parameters["apply_ensemble"].default is False, (
        "默认必须 False：EnsemblePredictor.predict_batch 会调 predict_single 再自行 "
        "blend，默认 True 会导致双重融合"
    )


def test_model_mode_field_defaults_to_single():
    """StockPrediction 必须自带 model_mode 字段，默认 single（历史行为不变）。"""
    from src.models.predictor import StockPrediction

    p = StockPrediction(
        code="000001", name="x", prob_up=0.4, prob_neutral=0.3, prob_down=0.3,
        score=50.0,
    )
    assert p.model_mode == "single"
    assert p.ensemble_detail == {}

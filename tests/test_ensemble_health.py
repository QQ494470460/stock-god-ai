# -*- coding: utf-8 -*-
"""超级集成健康审计单元测试（离线，Fake 组件，零网络/零真实模型）。

验证 2026-11-12 加的「超级集成健康审计」能力：
1. EnsemblePredictor.health_report() 结构化透出每腿 available / reason / weight / LLM 实时健康；
2. Predictor.ensemble_report() 在 use_ensemble=False 时返回「未启用」，开启时渲染逐腿审计；
3. LLM 腿（第五腿）health() 实时快照被透传；
4. 默认不触网、不重训、不改动任何权重/评分行为（纯观测）。
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models.ensemble import EnsemblePredictor, SubPredictor  # noqa: E402
from config.settings import config as app_config  # noqa: E402


@pytest.fixture(autouse=True)
def _pin_dynamic_off():
    """把动态权重钉为关，让 health_report 的 dynamic_weights 字段确定可断言。

    动态融合是独立旋钮（默认 False），与本次健康审计无关，钉住避免随全局配置漂移。
    """
    old = getattr(app_config.ml, "ensemble_dynamic_weights", False)
    app_config.ml.ensemble_dynamic_weights = False
    try:
        yield
    finally:
        app_config.ml.ensemble_dynamic_weights = old


class FakePred:
    """最小 StockPrediction 替身：blend_one 用 score_raw 作主腿分。"""

    def __init__(self, score):
        self.score = score
        self.score_raw = score
        self.agreement = 1.0
        self.model_mode = "single"
        self.ensemble_detail = {}


class FakePrimary:
    def __init__(self, score=60.0):
        self.score = score

    def predict_single(self, df, code="x", name="x"):
        p = FakePred(self.score)
        p.code = code
        p.name = name
        return p


class FakeSub(SubPredictor):
    name = "fake_xgboost"

    def __init__(self, score=80.0):
        self._val = score

    def _ensure(self):
        pass

    def _score(self, df, code):
        return self._val


class UnavailableSub(SubPredictor):
    name = "bad"

    def _ensure(self):
        raise RuntimeError("model file missing")


class FakeLLMSub(SubPredictor):
    """第五腿替身：available=True，health() 返回预置实时快照。"""

    name = "llm_alpha"

    def __init__(self, health=None):
        self._health = health or {
            "total_calls": 10, "success": 8, "failures": 2,
            "consecutive_failures": 1, "degraded": True,
        }

    def _ensure(self):
        pass

    def _score(self, df, code):
        return 50.0

    def health(self):
        return self._health


def _df():
    import pandas as pd
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]})


# --------------------------------------------------------------------------
# 1) health_report 结构化透出每腿状态
# --------------------------------------------------------------------------
def test_health_report_transmits_available_and_reason():
    ep = EnsemblePredictor(FakePrimary())
    ep.add_sub(FakeSub(90.0), weight=1.0)        # 可用
    ep.add_sub(UnavailableSub(), weight=0.8)     # 静默降级

    rep = ep.health_report()
    assert rep["enabled"] is True
    assert rep["n_subs"] == 2
    assert rep["n_available"] == 1
    assert rep["dynamic_weights"] is False

    by_name = {l["name"]: l for l in rep["legs"]}
    # 可用腿：available=True，reason 应为 None，权重透传
    good = by_name["fake_xgboost"]
    assert good["available"] is True
    assert good["reason"] is None
    assert good["weight"] == 1.0
    # 降级腿：available=False，reason 必须留档（否则看板看不出为什么被剔）
    bad = by_name["bad"]
    assert bad["available"] is False
    assert "model file missing" in (bad["reason"] or "")
    assert bad["weight"] == 0.8
    # 非 LLM 腿不应出现 llm_health 字段
    assert "llm_health" not in good
    assert "llm_health" not in bad


def test_health_report_llm_leg_health_passthrough():
    ep = EnsemblePredictor(FakePrimary())
    ep.add_sub(FakeLLMSub(), weight=0.5)

    rep = ep.health_report()
    leg = rep["legs"][0]
    assert leg["name"] == "llm_alpha"
    assert leg["available"] is True
    h = leg["llm_health"]
    assert h is not None
    assert h["total_calls"] == 10
    assert h["success"] == 8
    assert h["failures"] == 2
    assert h["consecutive_failures"] == 1
    assert h["degraded"] is True


def test_health_report_no_llm_leg_has_no_health_field():
    """未装配 LLM 腿时，report 不应含任何 llm_health（默认关闭零行为改变）。"""
    ep = EnsemblePredictor(FakePrimary())
    ep.add_sub(FakeSub(70.0), weight=1.0)

    rep = ep.health_report()
    assert rep["legs"]
    for leg in rep["legs"]:
        assert "llm_health" not in leg


# --------------------------------------------------------------------------
# 2) Predictor.ensemble_report 渲染层
# --------------------------------------------------------------------------
def test_predictor_ensemble_report_disabled():
    """use_ensemble=False → 直接返回「未启用」，不构建集成、不触网。"""
    old = app_config.ml.use_ensemble
    app_config.ml.use_ensemble = False
    try:
        from src.models.predictor import Predictor
        p = Predictor()
        out = p.ensemble_report()
        assert "未启用" in out
        assert "use_ensemble=False" in out
    finally:
        app_config.ml.use_ensemble = old


def test_predictor_ensemble_report_renders_audit():
    """use_ensemble=True 时 ensemble_report 渲染逐腿审计（含降级原因 + LLM 实时健康）。

    用 monkeypatch 把 _get_ensemble 替成 Fake 集成，避免触发真实模型/torch 加载，
    但仍走真实的 ensemble_report 渲染逻辑。
    """
    old = app_config.ml.use_ensemble
    app_config.ml.use_ensemble = True
    try:
        from src.models.predictor import Predictor
        p = Predictor()

        fake = EnsemblePredictor(FakePrimary())
        fake.add_sub(FakeSub(90.0), weight=1.0)
        fake.add_sub(UnavailableSub(), weight=0.8)
        fake.add_sub(FakeLLMSub(), weight=0.5)

        with patch.object(Predictor, "_get_ensemble", return_value=fake):
            out = p.ensemble_report()

        # 头部：启用腿计数 + 动态权重关
        assert "超级集成健康审计" in out
        assert "2/3" in out          # 3 只子腿里 2 只可用（fake_xgboost + llm_alpha）
        # 可用腿：✅ 可用
        assert "fake_xgboost" in out and "✅ 可用" in out
        # 降级腿：⚠️ 降级 + 真实原因透出（否则就是静默降级）
        assert "bad" in out and "⚠️ 降级" in out
        assert "model file missing" in out
        # LLM 腿实时健康：调用/成功/失败/连续失败 + 已降级 标记
        assert "llm_alpha" in out
        assert "调用 10" in out and "失败 2" in out
        assert "连续失败 1" in out and "已降级" in out
    finally:
        app_config.ml.use_ensemble = old


def test_health_report_is_pure_observation_no_side_effects_on_scores():
    """健康审计是纯观测：构造前后不应改权重/评分行为，且不触网（Fake 路径不触网）。"""
    primary = FakePrimary(60.0)
    ep = EnsemblePredictor(primary)
    ep.add_sub(FakeSub(80.0), weight=1.0)

    rep = ep.health_report()
    # 审计后仍能正常融合（证明没有「探测把腿搞坏」之类副作用）
    out = ep.predict_batch({"a": _df()})
    assert abs(out["a"].score - 70.0) < 1e-6   # (60 主 + 80 子) 等权
    assert rep["n_available"] == 1


# --------------------------------------------------------------------------
# 3) Predictor.ensemble_degradation_alert 聚焦告警
# --------------------------------------------------------------------------
def _fake_ep_with(legs):
    """构造一个 EnsemblePredictor 替身，health_report 返回给定 legs。"""
    class _FakeEP:
        def health_report(self):
            return {
                "enabled": True,
                "n_subs": len(legs),
                "n_available": sum(1 for l in legs if l["available"]),
                "dynamic_weights": False,
                "legs": legs,
            }
    return _FakeEP()


def test_predictor_degradation_alert_none_when_all_healthy():
    old = app_config.ml.use_ensemble
    app_config.ml.use_ensemble = True
    try:
        from src.models.predictor import Predictor
        p = Predictor()
        fake = _fake_ep_with([
            {"name": "alpha_xgboost", "available": True, "reason": None, "weight": 1.0},
            {"name": "lstm", "available": True, "reason": None, "weight": 0.8},
        ])
        with patch.object(Predictor, "_get_ensemble", return_value=fake):
            assert p.ensemble_degradation_alert() is None
    finally:
        app_config.ml.use_ensemble = old


def test_predictor_degradation_alert_text_when_a_leg_down():
    old = app_config.ml.use_ensemble
    app_config.ml.use_ensemble = True
    try:
        from src.models.predictor import Predictor
        p = Predictor()
        fake = _fake_ep_with([
            {"name": "alpha_xgboost", "available": True, "reason": None, "weight": 1.0},
            {"name": "lstm", "available": False, "reason": "RuntimeError: lstm_best.pt 缺失",
             "weight": 0.8},
        ])
        with patch.object(Predictor, "_get_ensemble", return_value=fake):
            alert = p.ensemble_degradation_alert()
        assert alert is not None
        assert "lstm" in alert
        assert "lstm_best.pt 缺失" in alert
        assert "静默降级" in alert
    finally:
        app_config.ml.use_ensemble = old


def test_predictor_degradation_alert_disabled_returns_none():
    old = app_config.ml.use_ensemble
    app_config.ml.use_ensemble = False
    try:
        from src.models.predictor import Predictor
        p = Predictor()
        # use_ensemble=False → _get_ensemble 返回 None → 直接 None（不构建集成、不触网）
        assert p.ensemble_degradation_alert() is None
    finally:
        app_config.ml.use_ensemble = old

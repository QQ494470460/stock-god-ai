# -*- coding: utf-8 -*-
"""
LLM 另类 alpha 实时集成腿（超级集成第五腿）单元测试
================================================
1) 默认关闭：enable_llm_alpha_leg=False 时 available=False，绝不注入。
2) 无真实 LLM key（自动转 mock）时优雅降级为不可用，绝不注入离线伪评分。
3) 启用 + 注入真实（非 mock）provider 时，把 sentiment/risk 正确映射成 0~100 评分。
4) 风险事件扣一档（利空 risk=1 拉低评分）。
5) 启用 + 真实 provider 时 available 链路打通（score_for 出分）。

全程不联网、不调用真实 LLM；用注入 fake provider 隔离外部依赖。
"""

import types

import pandas as pd
import pytest


@pytest.fixture
def ml_config():
    """安全切换 enable_llm_alpha_leg 并返回 app_config.ml，测试后还原。"""
    from config.settings import config as app_config
    ml = app_config.ml
    old = getattr(ml, "enable_llm_alpha_leg", False)
    yield ml
    ml.enable_llm_alpha_leg = old


def test_llm_alpha_disabled_by_default(ml_config):
    """默认开关关闭 → 不可用，绝不注入。"""
    from src.models.ensemble import LLMAlphaSubPredictor

    ml_config.enable_llm_alpha_leg = False
    sub = LLMAlphaSubPredictor()
    assert sub.available is False, "默认关闭时必须不可用"
    # score_for 在不可用时返回 None，不抛
    assert sub.score_for(None, "600000") is None


def test_llm_alpha_refuses_mock_provider(ml_config, monkeypatch):
    """无真实 key 时 LLMFeatureProvider 自动转 mock，本腿应主动拒绝激活。"""
    from src.models.ensemble import LLMAlphaSubPredictor
    from src.data import llm_features as llm_mod

    ml_config.enable_llm_alpha_leg = True
    ml_config.llm_alpha_provider = "deepseek"

    # 让 ensemble 在 _ensure 里 `from src.data.llm_features import LLMFeatureProvider`
    # 拿到一个立刻带 mock=True 的假 provider（模拟「无 key 自动转 mock」）
    class _FakeProvider:
        def __init__(self, *a, **k):
            self.cfg = types.SimpleNamespace(mock=True)

    monkeypatch.setattr(llm_mod, "LLMFeatureProvider", _FakeProvider)
    sub = LLMAlphaSubPredictor()
    assert sub.available is False, "自动转 mock 时必须拒绝激活（不注入离线伪评分）"


def test_llm_alpha_mapping_bullish(ml_config):
    """乐观情绪 → 高分。sentiment=0.6, risk=0 → score = 50 + 50*0.6 = 80。"""
    from src.models.ensemble import LLMAlphaSubPredictor

    ml_config.enable_llm_alpha_leg = True

    class FakeProv:
        def features_for_stock(self, code, end_date, days=20):
            return {"llm_sentiment": 0.6, "llm_risk": 0, "llm_confidence": 0.9}

    sub = LLMAlphaSubPredictor(provider=FakeProv(), lookback_days=10)
    sub._ensure()  # 注入路径：跳过真实构造与 mock 校验
    df = pd.DataFrame({"close": [1, 2, 3]},
                      index=pd.date_range("2026-01-01", periods=3))
    score = sub._score(df, "000001")
    assert abs(score - 80.0) < 1e-6, f"预期 80.0，实际 {score}"


def test_llm_alpha_risk_discount(ml_config):
    """中性情绪 + 利空风险事件 → 拉低。sentiment=0, risk=1 → raw=-0.3 → 35。"""
    from src.models.ensemble import LLMAlphaSubPredictor

    ml_config.enable_llm_alpha_leg = True

    class FakeProv:
        def features_for_stock(self, code, end_date, days=20):
            return {"llm_sentiment": 0.0, "llm_risk": 1, "llm_confidence": 0.9}

    sub = LLMAlphaSubPredictor(provider=FakeProv(), lookback_days=10)
    sub._ensure()
    df = pd.DataFrame({"close": [1, 2, 3]},
                      index=pd.date_range("2026-01-01", periods=3))
    score = sub._score(df, "000001")
    assert abs(score - 35.0) < 1e-6, f"预期 35.0，实际 {score}"


def test_llm_alpha_clamped_at_bounds(ml_config):
    """极端值被夹到 [0,100]。sentiment=-1, risk=1 → raw=-1.3 → 夹到 -1 → 0。"""
    from src.models.ensemble import LLMAlphaSubPredictor

    ml_config.enable_llm_alpha_leg = True

    class FakeProv:
        def features_for_stock(self, code, end_date, days=20):
            return {"llm_sentiment": -1.0, "llm_risk": 1, "llm_confidence": 0.9}

    sub = LLMAlphaSubPredictor(provider=FakeProv(), lookback_days=10)
    sub._ensure()
    df = pd.DataFrame({"close": [1, 2, 3]},
                      index=pd.date_range("2026-01-01", periods=3))
    score = sub._score(df, "000001")
    assert score == 0.0, f"预期 0.0，实际 {score}"


def test_llm_alpha_available_with_injected_provider(ml_config):
    """启用 + 注入非 mock provider → available 打通，score_for 出分。"""
    from src.models.ensemble import LLMAlphaSubPredictor

    ml_config.enable_llm_alpha_leg = True

    class FakeProv:
        def __init__(self):
            self.cfg = types.SimpleNamespace(mock=False)

        def features_for_stock(self, code, end_date, days=20):
            return {"llm_sentiment": 0.2, "llm_risk": 0, "llm_confidence": 0.5}

    sub = LLMAlphaSubPredictor(provider=FakeProv(), lookback_days=10)
    assert sub.available is True, "启用且注入非 mock provider 时应可用"
    df = pd.DataFrame({"close": [1, 2, 3]},
                      index=pd.date_range("2026-01-01", periods=3))
    sc = sub.score_for(df, "000001")
    assert sc is not None and 0.0 <= sc <= 100.0, f"score_for 应在 [0,100]，实际 {sc}"

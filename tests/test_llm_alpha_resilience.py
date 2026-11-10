# -*- coding: utf-8 -*-
"""第五腿 LLM 调用韧性与健康计数：退避/限流识别/失败计数/health 透传。

保持「默认关闭、零行为改变」铁律：所有计数与退避仅在 enable_llm_alpha_leg=True
（即真实 LLM 路径）时生效；测试均走 mock 或注入，不触网。
"""

import time

from src.data.llm_features import LLMFeatureProvider, LLMFeatureConfig


def _fake_client(content: str):
    """构造一个返回固定 JSON 内容的伪 openai 客户端（不触网）。"""
    msg = type("M", (), {"content": content})()
    choice = type("C", (), {"message": msg})()
    resp = type("R", (), {"choices": [choice]})()
    create = staticmethod(lambda **k: resp)
    completions = type("Comp", (), {"create": create})()
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


def test_empty_text_not_counted():
    p = LLMFeatureProvider(LLMFeatureConfig(mock=True))
    p.score_text("", as_of_date="2026-03-31")
    assert p.stats()["total_calls"] == 0


def test_real_success_counts_and_resets(monkeypatch):
    # 真实路径（mock=False）但注入伪客户端，验证计数逻辑
    p = LLMFeatureProvider(LLMFeatureConfig(mock=False, retry=1, rate_limit=0.0))
    p._client = _fake_client(
        '{"sentiment_score": 0.8, "risk_flag": 0, "confidence": 0.9, "rationale": "x"}')
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)
    r = p.score_text("利好：业绩大增", as_of_date="2026-03-31")
    s = p.stats()
    assert s["total_calls"] == 1
    assert s["success"] == 1
    assert s["consecutive_failures"] == 0
    assert s["degraded"] is False
    assert r["sentiment_score"] == 0.8


def test_failure_counts_and_zero_fallback(monkeypatch):
    class FakeCompletions:
        def create(self, *a, **k):
            raise RuntimeError("LLM down")
    class FakeChat:
        completions = FakeCompletions()
    class BoomClient:
        chat = FakeChat()

    p = LLMFeatureProvider(LLMFeatureConfig(mock=False, retry=2, rate_limit=0.0))
    p._client = BoomClient()  # 绕过真实 openai 构造
    sleeps = {"n": 0}
    monkeypatch.setattr(time, "sleep", lambda *a, **k: sleeps.__setitem__("n", sleeps["n"] + 1))

    r = p.score_text("任意文本", as_of_date="2026-03-31")
    s = p.stats()
    assert s["total_calls"] == 1
    assert s["success"] == 0
    assert s["failures"] == 1
    assert s["consecutive_failures"] == 1
    assert s["last_error"] is not None
    assert r == {"sentiment_score": 0.0, "risk_flag": 0,
                 "confidence": 0.0, "rationale": ""}
    # 第 1 次失败 → 退避 1 次（第 2 次失败不再退避）
    assert sleeps["n"] >= 1
    # degraded 标记：连续失败 >= retry
    assert s["degraded"] is True


def test_rate_limit_recognized(monkeypatch):
    class FakeCompletions:
        def create(self, *a, **k):
            raise RuntimeError("Rate limit exceeded (429)")
    class FakeChat:
        completions = FakeCompletions()
    class BoomClient:
        chat = FakeChat()

    p = LLMFeatureProvider(LLMFeatureConfig(mock=False, retry=1, rate_limit=0.0))
    p._client = BoomClient()
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)
    p.score_text("文本", as_of_date="2026-03-31")
    assert "RateLimitError" in p.stats()["last_error"] or "Rate limit" in p.stats()["last_error"]


def test_backoff_is_exponential(monkeypatch):
    p = LLMFeatureProvider(LLMFeatureConfig(backoff_base=1.0, backoff_cap=30.0))
    delays = []
    monkeypatch.setattr(time, "sleep", lambda d: delays.append(d))
    p._backoff(0)
    p._backoff(1)
    p._backoff(2)
    # 底数 1/2/4，叠加 0~25% 抖动
    assert 1.0 <= delays[0] < 2.0
    assert 2.0 <= delays[1] < 3.0
    assert 4.0 <= delays[2] < 5.0


def test_backoff_capped(monkeypatch):
    p = LLMFeatureProvider(LLMFeatureConfig(backoff_base=1.0, backoff_cap=5.0))
    delays = []
    monkeypatch.setattr(time, "sleep", lambda d: delays.append(d))
    p._backoff(10)  # 2^10 远超上限
    assert delays[0] <= 5.0 * 1.25


def test_health_surfaced_via_sub(monkeypatch):
    from config.settings import config as app_config
    from src.data import llm_features as llm_mod
    from src.models import ensemble as ens_mod

    class FakeProvider:
        def __init__(self, config):
            self.cfg = config
            self.cfg.mock = False
        def stats(self):
            return {"total_calls": 5, "success": 4, "failures": 1,
                    "consecutive_failures": 1, "last_error": "x",
                    "last_success_ts": 0.0, "degraded": False}

    monkeypatch.setattr(llm_mod, "LLMFeatureProvider", FakeProvider)
    monkeypatch.setattr(llm_mod, "LLMFeatureConfig",
                        lambda **kw: type("C", (), kw)())
    monkeypatch.setattr(app_config.ml, "enable_llm_alpha_leg", True)
    monkeypatch.setattr(app_config.ml, "llm_alpha_cache", True)
    try:
        sub = ens_mod.LLMAlphaSubPredictor()
        assert sub.available
        h = sub.health()
        assert h["failures"] == 1
        assert h["success"] == 4
    finally:
        monkeypatch.setattr(app_config.ml, "enable_llm_alpha_leg", False)

# -*- coding: utf-8 -*-
"""第五腿（LLM alpha）磁盘缓存：命中/回退/跨日 miss/关闭 + 集成装配。

保持「默认关闭、零行为改变」铁律：所有缓存逻辑仅在 enable_llm_alpha_leg=True
且 llm_alpha_cache=True 时生效；命中仍强制校验 end_date 防零泄漏。
"""

import json

from src.data import llm_features as llm_mod
from src.data.llm_features import LLMFeatureProvider, LLMFeatureConfig
from src.models import ensemble as ens_mod


def _provider(cache_dir, monkeypatch):
    """构造 mock 模式的 provider，并对 score_text 做调用计数（用于验证命中跳过）。"""
    calls = {"n": 0}
    base = LLMFeatureProvider(LLMFeatureConfig(cache_dir=cache_dir, mock=True))

    def spy_score_text(text, as_of_date=""):
        calls["n"] += 1
        return base._mock_score(text)
    monkeypatch.setattr(base, "score_text", spy_score_text)
    base._calls = calls
    return base


def test_cache_hit_skips_llm_call(tmp_path, monkeypatch):
    p = _provider(str(tmp_path), monkeypatch)
    r1 = p.features_for_stock("600000", "2026-03-31", days=20)
    assert p._calls["n"] == 1
    r2 = p.features_for_stock("600000", "2026-03-31", days=20)
    # 第二次命中缓存，不应再调 score_text
    assert p._calls["n"] == 1
    assert r1 == r2
    # 落盘文件存在且内容合法
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    obj = json.loads(files[0].read_text(encoding="utf-8"))
    assert obj["code"] == "600000"
    assert obj["end_date"] == "2026-03-31"
    assert set(obj["features"].keys()) == {"llm_sentiment", "llm_risk", "llm_confidence"}


def test_cache_miss_on_forged_end_date(tmp_path, monkeypatch):
    # 预写一个「end_date 更晚」的伪造缓存（含未来新闻的隐患）→ 必须作废重算
    f = tmp_path / "600000__2026-04-01__d20.json"
    f.write_text(json.dumps({"code": "600000", "end_date": "2026-04-01",
                             "days": 20, "features": {"llm_sentiment": 9.9,
                             "llm_risk": 1, "llm_confidence": 1.0}}),
                 encoding="utf-8")
    p = _provider(str(tmp_path), monkeypatch)
    r = p.features_for_stock("600000", "2026-03-31", days=20)
    # 因 end_date 不匹配，走了真实 mock 计算（sentiment 不会是伪造的 9.9）
    assert p._calls["n"] == 1
    assert r["llm_sentiment"] != 9.9


def test_cache_miss_on_corrupt_file(tmp_path, monkeypatch):
    (tmp_path / "600000__2026-03-31__d20.json").write_text("{not json", encoding="utf-8")
    p = _provider(str(tmp_path), monkeypatch)
    r = p.features_for_stock("600000", "2026-03-31", days=20)
    assert p._calls["n"] == 1  # 损坏文件当作 miss
    assert isinstance(r["llm_sentiment"], float)


def test_cross_day_miss(tmp_path, monkeypatch):
    p = _provider(str(tmp_path), monkeypatch)
    p.features_for_stock("600000", "2026-03-31", days=20)
    p.features_for_stock("600000", "2026-04-01", days=20)
    # 不同 as_of_date → 不同缓存文件 → 两次都走计算
    assert p._calls["n"] == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_cache_disabled_recomputes(tmp_path, monkeypatch):
    # cache_dir 为空 = 关闭缓存：每次都重算，且不落盘任何文件
    p = _provider("", monkeypatch)
    p.features_for_stock("600000", "2026-03-31", days=20)
    p.features_for_stock("600000", "2026-03-31", days=20)
    assert p._calls["n"] == 2  # 没命中
    assert len(list(tmp_path.glob("*.json"))) == 0  # 没写盘


def _wire_cache_test(monkeypatch, cache_on):
    """拦截 LLMFeatureProvider/Config，验证 _ensure 透传的 cache_dir。"""
    from config.settings import config as app_config

    captured = {}

    class FakeProvider:
        def __init__(self, config):
            self.cfg = config
            self.cfg.mock = False  # 绕过「无 key 自动转 mock」降级
            captured["cfg"] = config

    monkeypatch.setattr(llm_mod, "LLMFeatureProvider", FakeProvider)
    monkeypatch.setattr(llm_mod, "LLMFeatureConfig",
                        lambda **kw: type("C", (), kw)())

    monkeypatch.setattr(app_config.ml, "enable_llm_alpha_leg", True)
    monkeypatch.setattr(app_config.ml, "llm_alpha_cache", cache_on)
    try:
        sub = ens_mod.LLMAlphaSubPredictor()
        assert sub.available  # 触发 _ensure
        return captured["cfg"].cache_dir
    finally:
        monkeypatch.setattr(app_config.ml, "enable_llm_alpha_leg", False)


def test_ensemble_wires_cache_dir_when_enabled(monkeypatch):
    from config.settings import DATA_DIR
    cache_dir = _wire_cache_test(monkeypatch, cache_on=True)
    assert cache_dir == str(DATA_DIR / "llm_alpha_cache")


def test_ensemble_cache_disabled_when_flag_off(monkeypatch):
    cache_dir = _wire_cache_test(monkeypatch, cache_on=False)
    assert cache_dir == ""

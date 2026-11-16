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
    """构造一个 EnsemblePredictor 替身，health_report 返回给定 legs。

    未给 ``configured`` 的腿按「已配置」处理（向后兼容旧用例）。
    """
    class _FakeEP:
        def health_report(self):
            return {
                "enabled": True,
                "n_subs": len(legs),
                "n_available": sum(1 for l in legs if l["available"]),
                "n_configured": sum(1 for l in legs if l.get("configured", True)),
                "n_not_configured": sum(1 for l in legs if not l.get("configured", True)),
                "n_degraded": sum(1 for l in legs
                                  if l.get("configured", True) and not l["available"]),
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


# --------------------------------------------------------------------------
# 4) 降级告警去重：同腿只告警一次，避免逐股票刷屏淹没真实信号
# --------------------------------------------------------------------------
def test_available_subs_dedupes_warning_per_leg():
    """同一腿降级只告警一次（而非逐股票刷屏），避免淹没真正重要的那条告警。"""
    ep = EnsemblePredictor(FakePrimary())
    ep.add_sub(UnavailableSub(), weight=1.0)
    with patch("src.models.ensemble.logger") as mock_log:
        # 5 只股票各触发一次 blend_one → _available_subs 被调用 5 次
        ep.predict_batch({f"s{i}": _df() for i in range(5)})
    warns = [c for c in mock_log.warning.call_args_list
             if c.args and "不可用，已跳过" in c.args[0]]
    assert len(warns) == 1, f"期望降级告警仅 1 次，实际 {len(warns)} 次（去重失效）"


# --------------------------------------------------------------------------
# 5) 不可用腿进程内自愈（瞬时故障 / 模型文件后来补上，不必重启）
# --------------------------------------------------------------------------
class FlakySub(SubPredictor):
    """前 N 次 _ensure 抛错、之后成功的替身，模拟瞬时故障自愈。"""

    name = "flaky"

    def __init__(self, fail_times=1):
        self._fail_times = fail_times
        self._calls = 0

    def _ensure(self):
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("transient: c10.dll 初始化失败")


class AlwaysDownSub(SubPredictor):
    """永远 _ensure 失败的替身（文件真缺失），用于验证冷却 + 退避。"""

    name = "always_down"

    def __init__(self):
        self._calls = 0

    def _ensure(self):
        self._calls += 1
        raise RuntimeError("model file missing")


def test_sub_recover_succeeds_on_retry():
    """已探测失败（_ready=False）的腿，重试一次成功 → 重新可用、清原因。"""
    sub = FlakySub(fail_times=1)
    assert sub.available is False          # 首次懒探测失败
    assert sub._ready is False
    ok = sub._try_recover()
    assert ok is True
    assert sub.available is True
    assert sub._reason is None
    assert sub._recover_at is None
    assert sub._fail_count == 0
    assert sub._calls == 2                 # 首次探测 + 1 次重试


def test_sub_recover_ignores_unprobed_and_available():
    """从未探测（_ready=None）或已可用（_ready=True）的腿，_try_recover 不介入。"""
    fresh = FlakySub()
    assert fresh._try_recover() is False   # 未探测，不动
    good = FakeSub(80.0)
    assert good.available is True
    assert good._try_recover() is False    # 已可用，不动


def test_sub_recover_respects_cooldown_and_backoff():
    """持久故障：冷却内不重试（不反复昂贵重载），越过冷却才再试，退避翻倍。"""
    from datetime import datetime, timedelta

    sub = AlwaysDownSub()
    assert sub.available is False
    now = datetime(2026, 11, 14, 10, 0, 0)
    ok1 = sub._try_recover(now)
    assert ok1 is False
    assert sub._recover_at is not None and sub._recover_at > now
    calls1 = sub._calls
    # 冷却内（now 不变）→ 不再调 _ensure
    assert sub._try_recover(now) is False
    assert sub._calls == calls1
    # 越过冷却（now+61s）→ 再试一次 _ensure（仍失败）
    ok3 = sub._try_recover(now + timedelta(seconds=61))
    assert ok3 is False
    assert sub._calls == calls1 + 1
    # 退避应翻倍到 120s：下一次排期 = 本次 now + 120s
    assert sub._recover_at == (now + timedelta(seconds=61)) + timedelta(seconds=120)


def test_ensemble_try_recover_reports_recovered_and_still_down():
    """EnsemblePredictor.try_recover 汇总：瞬时故障腿归队、持久故障腿留在 still_down。"""
    ep = EnsemblePredictor(FakePrimary())
    flaky = FlakySub(fail_times=1)
    down = AlwaysDownSub()
    ep.add_sub(flaky, weight=1.0)
    ep.add_sub(down, weight=0.8)
    _ = flaky.available                   # 预探测：flaky 失败、down 失败
    _ = down.available
    recovered, still = ep.try_recover()
    assert "flaky" in recovered
    assert "always_down" in still
    # 再跑一次：flaky 已可用 → recovered 空；down 仍在冷却/失败
    recovered2, still2 = ep.try_recover()
    assert recovered2 == []
    assert "always_down" in still2


# --------------------------------------------------------------------------
# 6) 可选腿未配置（文件缺失/未训练）≠ 降级，不报警
# --------------------------------------------------------------------------
class OptionalMissingSub(SubPredictor):
    """模拟未训练的第四腿：optional=True 且主模型文件不存在。"""

    name = "transformer"
    optional = True
    model_file = "__definitely_absent_model__.pt"   # 必不存在

    def _ensure(self):
        raise FileNotFoundError("transformer_best.pt 缺失")


def test_optional_missing_leg_is_not_configured_not_degraded():
    ep = EnsemblePredictor(FakePrimary())
    ep.add_sub(FakeSub(90.0), weight=1.0)          # 核心可用
    ep.add_sub(OptionalMissingSub(), weight=0.8)   # 可选、未配置
    rep = ep.health_report()
    assert rep["n_subs"] == 2
    assert rep["n_available"] == 1
    assert rep["n_degraded"] == 0                  # 未配置 ≠ 降级
    assert rep["n_not_configured"] == 1
    by = {l["name"]: l for l in rep["legs"]}
    assert by["transformer"]["configured"] is False
    assert by["transformer"]["optional"] is True
    assert by["fake_xgboost"]["configured"] is True


def test_optional_missing_leg_does_not_raise_degradation_alert():
    """只有未配置的可选腿不可用 → 不算降级，alert 返回 None（不报警）。"""
    old = app_config.ml.use_ensemble
    app_config.ml.use_ensemble = True
    try:
        from src.models.predictor import Predictor
        p = Predictor()
        fake = _fake_ep_with([
            {"name": "alpha_xgboost", "available": True, "configured": True,
             "optional": False, "reason": None, "weight": 1.0},
            {"name": "transformer", "available": False, "configured": False,
             "optional": True, "reason": "FileNotFoundError: transformer_best.pt",
             "weight": 0.8},
        ])
        with patch.object(Predictor, "_get_ensemble", return_value=fake):
            assert p.ensemble_degradation_alert() is None
    finally:
        app_config.ml.use_ensemble = old


def test_optional_missing_leg_renders_not_configured_in_report():
    """报告里未配置腿显示「⚪ 未配置」，不得出现「⚠️ 降级」。"""
    old = app_config.ml.use_ensemble
    app_config.ml.use_ensemble = True
    try:
        from src.models.predictor import Predictor
        p = Predictor()
        fake = _fake_ep_with([
            {"name": "alpha_xgboost", "available": True, "configured": True,
             "optional": False, "reason": None, "weight": 1.0},
            {"name": "transformer", "available": False, "configured": False,
             "optional": True, "reason": "FileNotFoundError: transformer_best.pt",
             "weight": 0.8},
        ])
        with patch.object(Predictor, "_get_ensemble", return_value=fake):
            out = p.ensemble_report()
        assert "⚪ 未配置" in out
        assert "⚠️ 降级" not in out
        assert "1/1" in out                       # n_available/n_configured
        assert "另有 1 条可选腿未配置" in out
    finally:
        app_config.ml.use_ensemble = old


def test_core_missing_leg_still_alerts():
    """核心腿缺文件仍是真故障 → 必须报警（不能靠 optional 掩盖）。"""
    old = app_config.ml.use_ensemble
    app_config.ml.use_ensemble = True
    try:
        from src.models.predictor import Predictor
        p = Predictor()
        fake = _fake_ep_with([
            {"name": "alpha_xgboost", "available": True, "configured": True,
             "optional": False, "reason": None, "weight": 1.0},
            {"name": "lstm", "available": False, "configured": True,
             "optional": False, "reason": "FileNotFoundError: lstm_best.pt",
             "weight": 0.8},
        ])
        with patch.object(Predictor, "_get_ensemble", return_value=fake):
            alert = p.ensemble_degradation_alert()
        assert alert is not None and "lstm" in alert
    finally:
        app_config.ml.use_ensemble = old


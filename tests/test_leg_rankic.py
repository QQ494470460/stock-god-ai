# -*- coding: utf-8 -*-
"""
第六腿·动态融合（RankIC 动态权重）单元测试
========================================
1) aggregate_leg_rankic：逐日 IC → 每腿均值 RankIC / IR（纯函数，合成数据）。
2) dynamic_weights_from_stats：RankIC 越高权重越大；floor 截负；缺腿回退 fallback；归一化。
3) load_leg_rankic：缺失/过期/格式错 → None（触发回退静态权重）。
4) EnsemblePredictor._dynamic_weights：默认关 → None（零行为改变）；开关开 + 有效统计 → 派生权重。

全部离线，不联网、不跑真实集成。
"""

import json


# ---------------------------------------------------------------------------
# 1) aggregate_leg_rankic
# ---------------------------------------------------------------------------
def test_aggregate_perfect_positive():
    from src.core.leg_rankic import aggregate_leg_rankic

    per_day = {
        "xgboost": [([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])],
    }
    out = aggregate_leg_rankic(per_day)
    assert "xgboost" in out
    assert abs(out["xgboost"]["rankic"] - 1.0) < 1e-6
    assert out["xgboost"]["n_days"] == 1


def test_aggregate_positive_vs_negative_legs():
    from src.core.leg_rankic import aggregate_leg_rankic

    per_day = {
        # 两日都完美正相关
        "good": [([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
                 ([3.0, 2.0, 1.0], [0.3, 0.2, 0.1])],
        # 两日都完美负相关
        "bad": [([1.0, 2.0, 3.0], [0.3, 0.2, 0.1]),
                ([3.0, 2.0, 1.0], [0.1, 0.2, 0.3])],
    }
    out = aggregate_leg_rankic(per_day)
    assert out["good"]["rankic"] > 0.99
    assert out["bad"]["rankic"] < -0.99
    assert out["good"]["n_days"] == 2 and out["bad"]["n_days"] == 2


def test_aggregate_skips_insufficient_samples():
    from src.core.leg_rankic import aggregate_leg_rankic

    per_day = {"weak": [([1.0], [0.1])]}  # 单点，样本<3，应被跳过
    out = aggregate_leg_rankic(per_day)
    assert "weak" not in out


# ---------------------------------------------------------------------------
# 2) dynamic_weights_from_stats
# ---------------------------------------------------------------------------
def test_dynamic_weights_better_leg_gets_more():
    from src.core.leg_rankic import dynamic_weights_from_stats

    stats = {
        "alpha_xgboost": {"rankic": 0.08},
        "lstm": {"rankic": 0.04},
        "transformer": {"rankic": 0.02},
    }
    names = ["alpha_xgboost", "lstm", "transformer"]
    fb = [1.0, 0.8, 0.8]
    w = dynamic_weights_from_stats(stats, names, fb, floor=0.0)
    assert abs(sum(w) - 1.0) < 1e-9
    # 严格按 rankic 比例：0.08 : 0.04 : 0.02 = 4 : 2 : 1
    assert abs(w[0] - 4 / 7) < 1e-9
    assert abs(w[1] - 2 / 7) < 1e-9
    assert abs(w[2] - 1 / 7) < 1e-9
    assert w[0] > w[1] > w[2]


def test_dynamic_weights_floor_clips_negative():
    from src.core.leg_rankic import dynamic_weights_from_stats

    stats = {
        "alpha_xgboost": {"rankic": 0.06},
        "lstm": {"rankic": -0.03},  # 负 IC
    }
    names = ["alpha_xgboost", "lstm"]
    fb = [1.0, 0.8]
    w0 = dynamic_weights_from_stats(stats, names, fb, floor=0.0)
    # floor=0：负 IC 腿权重被夹到 0 → 完全剔除
    assert abs(w0[0] - 1.0) < 1e-9 and abs(w0[1] - 0.0) < 1e-9

    w1 = dynamic_weights_from_stats(stats, names, fb, floor=0.02)
    # floor=0.02：负 IC 腿保留最小正权重
    assert abs(w1[1] - 0.02 / 0.08) < 1e-9


def test_dynamic_weights_missing_leg_uses_fallback():
    from src.core.leg_rankic import dynamic_weights_from_stats

    stats = {"alpha_xgboost": {"rankic": 0.06}}  # lstm 无统计
    names = ["alpha_xgboost", "lstm"]
    fb = [1.0, 0.8]
    w = dynamic_weights_from_stats(stats, names, fb, floor=0.0)
    # raw = [0.06, 0.8]（lstm 用静态 fallback）→ 归一化
    assert abs(w[0] - 0.06 / 0.86) < 1e-9
    assert abs(w[1] - 0.8 / 0.86) < 1e-9


def test_dynamic_weights_all_zero_fallback_to_equal():
    from src.core.leg_rankic import dynamic_weights_from_stats

    # 极端：所有腿 rankic 全为负 + floor=0 → raw 全 0 → 退化为等权
    stats = {"a": {"rankic": -0.1}, "b": {"rankic": -0.2}}
    names = ["a", "b"]
    fb = [0.0, 0.0]
    w = dynamic_weights_from_stats(stats, names, fb, floor=0.0)
    assert abs(w[0] - 0.5) < 1e-9 and abs(w[1] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# 3) load_leg_rankic
# ---------------------------------------------------------------------------
def test_load_leg_rankic_missing(tmp_path):
    from src.core.leg_rankic import load_leg_rankic

    assert load_leg_rankic(path=tmp_path / "nope.json") is None


def test_load_leg_rankic_stale(tmp_path):
    from src.core.leg_rankic import load_leg_rankic

    f = tmp_path / "leg_rankic.json"
    f.write_text(json.dumps({
        "last_date": "2000-01-01",  # 远超 30 天
        "legs": {"xgboost": {"rankic": 0.06, "n_days": 10}},
    }), encoding="utf-8")
    assert load_leg_rankic(path=f, max_age_days=30) is None


def test_load_leg_rankic_fresh(tmp_path):
    from src.core.leg_rankic import load_leg_rankic

    f = tmp_path / "leg_rankic.json"
    f.write_text(json.dumps({
        "last_date": "2099-01-01",  # 未来日期，不过期
        "legs": {"xgboost": {"rankic": 0.06, "n_days": 10}},
    }), encoding="utf-8")
    legs = load_leg_rankic(path=f, max_age_days=30)
    assert legs is not None and "xgboost" in legs


def test_load_leg_rankic_bad_format(tmp_path):
    from src.core.leg_rankic import load_leg_rankic

    f = tmp_path / "leg_rankic.json"
    f.write_text("not json", encoding="utf-8")
    assert load_leg_rankic(path=f) is None


# ---------------------------------------------------------------------------
# 4) EnsemblePredictor._dynamic_weights 行为
# ---------------------------------------------------------------------------
class _FakeSub:
    name = "alpha_xgboost"
    available = True

    def score_for(self, df, code):
        return 50.0


def _make_ep():
    from src.models.ensemble import EnsemblePredictor

    class _DummyPrimary:
        def predict_single(self, df, code=None):
            return None

    return EnsemblePredictor(_DummyPrimary())


def test_dynamic_weights_off_returns_none():
    """默认开关关闭 → 返回 None（blend_one 回退静态权重，零行为改变）。"""
    from config.settings import config as app_config

    old = app_config.ml.ensemble_dynamic_weights
    app_config.ml.ensemble_dynamic_weights = False
    try:
        ep = _make_ep()
        avail = [(_FakeSub(), 1.0)]
        assert ep._dynamic_weights(avail) is None
    finally:
        app_config.ml.ensemble_dynamic_weights = old


def test_dynamic_weights_on_derives(monkeypatch):
    """开关开 + 有效统计 → 按 RankIC 派生权重。"""
    from config.settings import config as app_config
    from src.core import leg_rankic as lr

    old = app_config.ml.ensemble_dynamic_weights
    app_config.ml.ensemble_dynamic_weights = True
    try:
        stats = {
            "alpha_xgboost": {"rankic": 0.08, "n_days": 20},
            "lstm": {"rankic": 0.04, "n_days": 20},
        }
        monkeypatch.setattr(lr, "load_leg_rankic", lambda *a, **k: stats)
        ep = _make_ep()
        avail = [(_FakeSub(), 1.0), (type("S", (), {"name": "lstm", "available": True})(), 0.8)]
        w = ep._dynamic_weights(avail)
        assert w is not None
        assert abs(sum(w) - 1.0) < 1e-9
        # 0.08 : 0.04 = 2 : 1
        assert abs(w[0] - 2 / 3) < 1e-9 and abs(w[1] - 1 / 3) < 1e-9
    finally:
        app_config.ml.ensemble_dynamic_weights = old


# ---------------------------------------------------------------------------
# 5) 第六腿基线生成器修复 + 增量回填（纯函数层，离线）
# ---------------------------------------------------------------------------
def test_aggregate_day_builds_full_cross_section():
    from src.core.leg_rankic import _aggregate_day

    # 3 只股票、2 条腿；验证「按腿聚合成整段横截面」，而非逐票单点
    by_code = {
        "600000": {"xgboost": 0.9, "lstm": 0.7},
        "600001": {"xgboost": 0.5, "lstm": 0.4},
        "600002": {"xgboost": 0.2, "lstm": 0.1},
    }
    valid = ["600000", "600001", "600002"]
    excess = [0.01, -0.02, 0.03]
    day = _aggregate_day(by_code, valid, excess)
    assert set(day.keys()) == {"xgboost", "lstm"}
    for leg, (scs, fwd) in day.items():
        # 修复前是逐票单点 → 样本<3 → _spearman 恒 nan → 基线恒空
        assert len(scs) == 3, "应按腿聚合成整段横截面"
        assert len(fwd) == 3
        assert fwd == [0.01, -0.02, 0.03]


def test_old_per_stock_shape_would_be_rejected():
    """回归护栏：单票单点 ( [x], [y] ) 必须被 len>=3 守卫剔除（即旧 bug 的形状）。"""
    from src.core.leg_rankic import _spearman
    ic = _spearman([0.5], [0.1])  # 旧实现每票都这么传
    assert ic != ic  # nan → 不会被计入


def test_aggregate_from_daily_matches_aggregate():
    from src.core.leg_rankic import (aggregate_from_daily,
                                    aggregate_leg_rankic, _spearman)
    day1 = ([-1.0, 0.0, 1.0], [-0.1, 0.0, 0.1])   # 正 IC
    day2 = ([1.0, 0.0, -1.0], [-0.1, 0.0, 0.1])   # 负 IC
    daily = {"xgboost": [("2026-01-01", _spearman(*day1)),
                         ("2026-01-02", _spearman(*day2))]}
    out = aggregate_from_daily(daily)
    ref = aggregate_leg_rankic({"xgboost": [day1, day2]})
    assert abs(out["xgboost"]["rankic"] - ref["xgboost"]["rankic"]) < 1e-9
    assert out["xgboost"]["n_days"] == 2


def test_merge_daily_records_appends_and_trims():
    from src.core.leg_rankic import merge_daily_records

    existing = {"xgboost": [{"date": "2026-01-01", "ic": 0.05},
                            {"date": "2026-01-02", "ic": 0.06}]}
    new = {"xgboost": [{"date": "2026-01-03", "ic": 0.07}]}
    merged = merge_daily_records(existing, new, max_days=3)
    assert [r["date"] for r in merged["xgboost"]] == [
        "2026-01-01", "2026-01-02", "2026-01-03"]
    # 裁剪：超过 max_days 只留最近
    merged2 = merge_daily_records(existing, new, max_days=2)
    assert [r["date"] for r in merged2["xgboost"]] == ["2026-01-02", "2026-01-03"]


def test_merge_daily_records_new_overrides_existing_date():
    from src.core.leg_rankic import merge_daily_records

    existing = {"xgboost": [{"date": "2026-01-02", "ic": 0.06}]}
    new = {"xgboost": [{"date": "2026-01-02", "ic": 0.99}]}  # 同一天重算
    merged = merge_daily_records(existing, new, max_days=10)
    assert len(merged["xgboost"]) == 1
    assert merged["xgboost"][0]["ic"] == 0.99

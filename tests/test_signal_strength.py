"""信号强度诊断（``src/core/signal_strength.py``）的契约测试。

背景证据（写死在测试里，防止有人"顺手"改阈值把弱信号说成强信号）：
  2026-11-07 榜单：达标 9/425、σ=2.43、最高分 62.4 → 健康
  2026-11-13 榜单：达标 0/425、σ=1.38、最高分 56.3 → 弱
"""

from __future__ import annotations

import pytest

from src.core import signal_strength as ss


def _batch(scores, prob_up=None, prob_down=None):
    n = len(scores)
    return {
        "scores": list(scores),
        "prob_up": list(prob_up) if prob_up else [0.5] * n,
        "prob_down": list(prob_down) if prob_down else [0.4] * n,
    }


def _normal_scores(n=425, mean=52.7, std=1.0, seed=7):
    # 均值取 11-13 实测的 52.7：这才是真实榜单的分布中心（阈值 58 在 3.8σ 外）
    import random
    rnd = random.Random(seed)
    return [rnd.gauss(mean, std) for _ in range(n)]


# ---------------------------------------------------------------- 基本分级

def test_healthy_batch_is_strong():
    """11-07 那种分布（σ 2.43、有票过 58 线）必须判为强。"""
    batch = _batch(_normal_scores(mean=55.3, std=2.43))
    st = ss.compute_signal_strength(batch, min_score=58.0, target_n=8)
    assert st["level"] == "strong", st
    assert st["strength"] >= ss.LEVEL_STRONG
    assert st["strict_pass"] > 0
    assert st["dispersion"] >= 2.0


def test_collapsed_batch_is_weak():
    """11-13 那种分布（σ 1.38、达标 0）必须判为弱，且不得判成强/良。"""
    batch = _batch(_normal_scores(mean=52.7, std=1.38))
    st = ss.compute_signal_strength(batch, min_score=58.0, target_n=8)
    assert st["level"] == "weak", st
    assert st["strict_pass"] == 0
    assert "补足" in st["note"]


def test_flat_batch_is_none():
    """几乎无区分度（σ≈0）→ 无信号。"""
    batch = _batch([50.0] * 200)
    st = ss.compute_signal_strength(batch, min_score=58.0, target_n=8)
    assert st["level"] == "none", st
    assert "随机" in st["note"]


def test_strength_is_monotone_in_dispersion():
    """区分度越高，强度分必须越高（同达标率下）。"""
    lo = ss.compute_signal_strength(_batch(_normal_scores(std=1.0)), 60.0, 8)["strength"]
    hi = ss.compute_signal_strength(_batch(_normal_scores(std=2.5)), 60.0, 8)["strength"]
    assert hi > lo


def test_missing_or_bad_batch_never_raises():
    """诊断代码绝不能拖垮选股：垃圾输入只返回 unknown。"""
    for bad in (None, {}, {"scores": []}, {"scores": ["x", None]},
                {"scores": [1, 2, 3]}, "not a dict", 123):
        st = ss.compute_signal_strength(bad, 58.0, 8)
        assert st["level"] == "unknown", (bad, st)
        assert st["strength"] is None


def test_top_up_prob_edge_uses_cross_section():
    """TOP-N 上涨概率对比的是**全池均值**，而不是 50%。"""
    scores = list(range(100, 0, -1))            # 降序
    ups = [0.60] * 8 + [0.40] * 92              # 前 8 名更看涨
    st = ss.compute_signal_strength(_batch(scores, ups), 58.0, 8)
    assert st["top_up_prob"] == pytest.approx(60.0, abs=0.2)
    # 全池均值 0.4*0.92+0.6*0.08 = 41.6% → 边际约 +18.4pp
    assert st["top_edge_pp"] == pytest.approx(18.4, abs=0.5)


# ---------------------------------------------------------------- 台账 / 趋势

def test_history_roundtrip_and_same_day_dedup(tmp_path):
    stats = ss.compute_signal_strength(_batch(_normal_scores(std=1.4)), 58.0, 8)
    p1 = ss.record_signal_history(stats, data_dir=tmp_path, as_of="2026-11-13")
    assert p1 and p1.exists()
    # 同一天重复写 → 覆盖，不追加
    ss.record_signal_history(stats, data_dir=tmp_path, as_of="2026-11-13")
    hist = ss.recent_signal_history(limit=10, data_dir=tmp_path)
    assert len(hist) == 1
    assert hist[0]["date"] == "2026-11-13"
    assert hist[0]["level"] == "weak"


def test_history_skips_unknown(tmp_path):
    assert ss.record_signal_history({"level": "unknown"}, data_dir=tmp_path) is None
    assert ss.recent_signal_history(data_dir=tmp_path) == []


def test_history_ignores_corrupt_lines(tmp_path):
    p = tmp_path / "signal_strength_history.jsonl"
    p.write_text('{"date":"2026-11-10","strength":80,"dispersion":2.2}\n'
                 'this-is-not-json\n'
                 '{"date":"2026-11-12","strength":40,"dispersion":1.4}\n',
                 encoding="utf-8")
    hist = ss.recent_signal_history(data_dir=tmp_path)
    assert [h["date"] for h in hist] == ["2026-11-10", "2026-11-12"]


def test_trend_detects_decay(tmp_path):
    for d, s, disp in [("2026-11-07", 90, 2.4), ("2026-11-10", 85, 2.3),
                       ("2026-11-11", 88, 2.35), ("2026-11-12", 80, 2.1)]:
        ss.record_signal_history({"level": "ok", "strength": s, "dispersion": disp,
                                  "strict_pass": 3, "n": 400, "min_score": 58},
                                 data_dir=tmp_path, as_of=d)
    cur = {"date": "2026-11-13", "strength": 34.5, "dispersion": 1.38}
    tr = ss.signal_trend(cur, lookback=5, data_dir=tmp_path)
    assert tr["state"] == "decaying", tr
    assert "衰减" in tr["note"]


def test_trend_unknown_with_insufficient_samples(tmp_path):
    cur = {"date": "2026-11-13", "strength": 34.5, "dispersion": 1.38}
    assert ss.signal_trend(cur, data_dir=tmp_path)["state"] == "unknown"


# ---------------------------------------------------------------- 渲染

def test_alert_only_for_weak_or_none():
    strong = ss.compute_signal_strength(_batch(_normal_scores(std=2.5)), 55.0, 8)
    assert strong["level"] == "strong"
    assert ss.signal_strength_alert(strong) is None

    weak = ss.compute_signal_strength(_batch(_normal_scores(std=1.4)), 58.0, 8)
    msg = ss.signal_strength_alert(weak)
    assert msg and "偏弱" in msg
    assert "require_strict_pass" in msg      # 必须给出可执行的下一步

    none = ss.compute_signal_strength(_batch([50.0] * 100), 58.0, 8)
    assert "基本无信号" in (ss.signal_strength_alert(none) or "")


def test_format_contains_key_numbers():
    st = ss.compute_signal_strength(_batch(_normal_scores(std=1.38)), 58.0, 8)
    txt = ss.format_signal_strength(st)
    assert "信号强度" in txt
    assert str(st["dispersion"]) in txt
    assert "严格达标 0 只" in txt


def test_unknown_renders_gracefully():
    assert "数据不足" in ss.format_signal_strength({"level": "unknown"})
    assert ss.signal_strength_alert(None) is None


def test_zero_strict_pass_caps_level_at_weak():
    """2026-11-16 实测：σ=3.16 很漂亮，但全池 0 只达到阈值 58。

    区分度再好，一只都没达标就说明今天只能靠排名补足 —— 不能判成"良/强"
    误导用户按常规仓位建仓。
    """
    batch = _batch(_normal_scores(mean=48.0, std=3.2))     # 高分尾也够不到 58
    st = ss.compute_signal_strength(batch, min_score=58.0, target_n=8)
    assert st["strict_pass"] == 0
    assert st["dispersion"] >= 2.0                          # 区分度满格
    assert st["level"] == "weak", st
    assert "没有一只达到阈值" in st["note"]

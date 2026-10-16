# -*- coding: utf-8 -*-
"""tracker_v2 的「两代格式兼容」与汇总一致性。

背景
----
`data/prediction_history.json` 里**同时存在两种格式**：

* 旧 `tracker.py`：``entry["verification"] = {...}``，单槽位，
  `lookback_days` 标窗口，T+5 与 T+10 会互相覆盖；
* 现行 `tracker_v2`：``entry["verifications"] = {"5": {...}, "10": {...}}``。

看板与调度本轮已统一走 v2。如果 v2 只认新格式，那么历史上那 17 批
「旧版验证过」的记录会在看板上凭空消失（汇总从 17 批掉到 0 批）——
这正是本文件第一条断言要钉死的东西。

同时钉住「兼容规则只有一份」：``src/evolution/feedback.py`` 的
``_entry_verifications`` 必须与 ``tracker_v2.entry_verifications`` 完全一致。
两边写歪的后果很隐蔽 —— 看板显示 T+5 有 17 批，而生成器 C 只看到 3 批，
于是「要不要重训」的判断建立在错误样本上。
"""

from __future__ import annotations

import json

import pytest

from src.models import tracker_v2


def _legacy_entry(scan_date: str = "2026-10-09", horizon: int = 5) -> dict:
    return {
        "scan_date": scan_date,
        "logged_at": f"{scan_date}T13:07:12",
        "n_predictions": 20,
        "predictions": [{"code": "600000", "score": 80.0}],
        # 旧版：单槽位 + lookback_days
        "verification": {
            "scan_date": scan_date,
            "lookback_days": horizon,
            "accuracy": 0.20,
            "avg_return": -0.012,
            "excess_return": -0.008,
            "win_rate": 0.30,
            "checked": 20,
            "correct": 4,
        },
    }


def _v2_entry(scan_date: str = "2026-10-10") -> dict:
    return {
        "scan_date": scan_date,
        "logged_at": f"{scan_date}T13:10:00",
        "n_predictions": 15,
        "predictions": [{"code": "000001", "score": 71.0}],
        "price_snapshot": {"000001": 12.34},
        "verifications": {
            "5": {
                "base_date": scan_date,
                "hold_days": 5,
                "accuracy": 0.55,
                "avg_return": 0.021,
                "excess_return": 0.013,
                "win_rate": 0.60,
                "checked": 15,
                "correct": 8,
            },
        },
    }


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker_v2, "TRACK_FILE", tmp_path / "prediction_history.json")
    return tracker_v2.TRACK_FILE


def _write(path, entries):
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# entry_verifications —— 唯一的兼容规则
# ---------------------------------------------------------------------------

def test_reads_v2_format():
    out = tracker_v2.entry_verifications(_v2_entry())
    assert set(out) == {"5"}
    assert out["5"]["accuracy"] == 0.55


def test_reads_legacy_format_and_maps_horizon_from_lookback_days():
    out = tracker_v2.entry_verifications(_legacy_entry(horizon=5))
    assert set(out) == {"5"}, "旧版 lookback_days 必须映射成窗口键"
    assert out["5"]["accuracy"] == 0.20


def test_reads_legacy_format_t10():
    out = tracker_v2.entry_verifications(_legacy_entry(horizon=10))
    assert set(out) == {"10"}


def test_v2_wins_when_both_present_for_same_horizon():
    """同一窗口两种格式都有时，以 v2 为准 —— 新结果不该被旧结果盖回去。"""
    e = _legacy_entry()
    e["verifications"] = {"5": {"hold_days": 5, "accuracy": 0.99, "checked": 20,
                                "correct": 20, "avg_return": 0.05,
                                "excess_return": 0.04, "win_rate": 1.0}}
    out = tracker_v2.entry_verifications(e)
    assert out["5"]["accuracy"] == 0.99


def test_merged_when_horizons_differ():
    """旧版存了 T+5、v2 存了 T+10 → 两个窗口都要在。"""
    e = _legacy_entry(horizon=5)
    e["verifications"] = {"10": {"hold_days": 10, "accuracy": 0.40, "checked": 20,
                                 "correct": 8, "avg_return": 0.01,
                                 "excess_return": 0.0, "win_rate": 0.5}}
    out = tracker_v2.entry_verifications(e)
    assert set(out) == {"5", "10"}


def test_tolerates_garbage():
    assert tracker_v2.entry_verifications({}) == {}
    assert tracker_v2.entry_verifications({"verifications": None}) == {}
    assert tracker_v2.entry_verifications({"verification": "not-a-dict"}) == {}
    assert tracker_v2.entry_verifications({"verifications": {"5": "not-a-dict"}}) == {}


# ---------------------------------------------------------------------------
# 汇总：旧格式的记录不能消失（这是本轮修复的核心动机）
# ---------------------------------------------------------------------------

def test_summary_counts_legacy_verifications(isolated):
    _write(isolated, [_legacy_entry("2026-10-09", 5)])

    s = tracker_v2.get_accuracy_summary()
    assert s["n_verified"] == 1, "旧版验证过的记录必须在汇总里可见"
    assert s["accuracy"] == pytest.approx(0.20)
    assert s["avg_excess"] == pytest.approx(-0.008)
    assert s["history"][0]["hold_days"] == 5, "旧版行也要带上 hold_days，表格才同构"


def test_summary_merges_both_generations(isolated):
    _write(isolated, [_legacy_entry("2026-10-09", 5), _v2_entry("2026-10-10")])

    s = tracker_v2.get_accuracy_summary()
    assert s["n_verified"] == 2
    # 加权总体准确率 = (4 + 8) / (20 + 15)
    assert s["accuracy"] == pytest.approx(12 / 35, abs=1e-4)


def test_summary_empty_shape_uses_none(isolated):
    s = tracker_v2.get_accuracy_summary()
    assert s["n_verified"] == 0
    assert s["accuracy"] is None and s["avg_excess"] is None


# ---------------------------------------------------------------------------
# 待验证判定：不重复计算已经验证过的窗口
# ---------------------------------------------------------------------------

def test_verify_all_pending_skips_legacy_verified_horizons(isolated, monkeypatch):
    _write(isolated, [_legacy_entry("2026-10-09", 5)])

    called = []

    def _fake_verify(scan_date, horizon=5, persist=True):
        called.append((scan_date, horizon))
        return None

    monkeypatch.setattr(tracker_v2, "verify_predictions", _fake_verify)
    tracker_v2.verify_all_pending(horizons=[5, 10])

    assert (("2026-10-09", 5)) not in called, \
        "旧版已验证过的 T+5 不该重复拉行情"
    assert ("2026-10-09", 10) in called, "T+10 从未验证过，必须补验"


# ---------------------------------------------------------------------------
# 兼容规则只有一份
# ---------------------------------------------------------------------------

def test_feedback_delegates_to_tracker_v2():
    """feedback 的兼容读取必须与 tracker_v2 逐字节等价。

    两边写歪的后果：同样的数据在看板与生成器 C 那里被归到不同窗口上，
    汇总数字对不上，而且没人会立刻发现。
    """
    from src.evolution import feedback

    samples = [
        _legacy_entry(5),
        _legacy_entry(10),
        _v2_entry(),
        {},
        {"verification": {"hold_days": 10, "accuracy": 0.5}, "verifications": {}},
    ]
    for e in samples:
        assert feedback._entry_verifications(e) == tracker_v2.entry_verifications(e)

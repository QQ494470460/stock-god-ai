# -*- coding: utf-8 -*-
"""自我进化 · 反馈收集层测试。

这一层最容易出的错不是崩溃，而是**静默失真**：

* 把「0 条数据」当成「0 分准确率」，生成器据此做出错误决策；
* 只读采集里混进写盘动作，跑一次进化就篡改了历史验证记录；
* 两代追踪器字段不兼容（``verification`` vs ``verifications``），
  导致「明明验证过 17 批，系统却认为 0 批」——线上真实发生过。

因此本文件的断言集中在：**采集结果必须与事实一致，且绝不写盘**。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evolution import feedback  # noqa: E402
from src.evolution.feedback import PerformanceSnapshot  # noqa: E402

REAL_HISTORY = ROOT / "data" / "prediction_history.json"
REAL_JOURNAL = ROOT / "data" / "trading_journal.jsonl"


# ------------------------------------------------------------------ 造数据

def _batch(scan_date: str, horizon: int = 5, accuracy: float = 0.5,
           excess: float = 0.01, win_rate: float = 0.5,
           n: int = 20, legacy: bool = False, buckets=None) -> dict:
    v = {
        "scan_date": scan_date,
        "hold_days": horizon,
        "accuracy": accuracy,
        "excess_return": excess,
        "win_rate": win_rate,
        "checked": n,
        "correct": int(n * accuracy),
    }
    if buckets:
        v["by_score_bucket"] = buckets
    preds = [{"code": f"60000{i}", "signal": 1, "score": 70.0} for i in range(3)]
    e = {"scan_date": scan_date, "n_predictions": 3, "predictions": preds,
         "logged_at": scan_date + "T15:00:00"}
    if legacy:
        v["lookback_days"] = horizon
        e["verification"] = v
    else:
        e["verifications"] = {str(horizon): v}
    return e


# ------------------------------------------------------------------ ① 预测反馈

def test_reads_v2_verifications():
    h = [_batch("2026-10-01", 5, 0.6), _batch("2026-10-02", 5, 0.4)]
    out = feedback.collect_prediction_feedback(h)

    assert out["n_batches"] == 2
    assert out["n_verified"] == 2
    assert out["accuracy_by_horizon"]["5"] == [0.6, 0.4]


def test_reads_legacy_single_slot_verification():
    """兼容旧 tracker 的单槽位字段（T+5/T+10 会互相覆盖的那一版）。"""
    h = [_batch("2026-10-01", 5, 0.55, legacy=True)]
    out = feedback.collect_prediction_feedback(h)

    assert out["n_verified"] == 1
    assert out["accuracy_by_horizon"]["5"] == [0.55]


def test_separates_horizons():
    h = [_batch("2026-10-01", 5, 0.60), _batch("2026-10-01", 10, 0.30)]
    out = feedback.collect_prediction_feedback(h)

    assert out["accuracy_by_horizon"]["5"] == [0.60]
    assert out["accuracy_by_horizon"]["10"] == [0.30]


def test_recent_accuracy_is_tail_mean():
    h = [_batch(f"2026-10-{i:02d}", 5, a)
         for i, a in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], start=1)]
    out = feedback.collect_prediction_feedback(h)

    # 近 5 次 = 0.3,0.4,0.5,0.6,0.7 → 0.5
    assert out["recent_accuracy"]["5"] == pytest.approx(0.5)


def test_zero_verified_is_not_silently_zero_accuracy():
    """25 批预测但 0 批验证 ≠ 准确率 0。必须显式区分。"""
    h = [_batch(f"2026-10-{i:02d}") for i in range(1, 6)]
    for e in h:
        e.pop("verifications")

    out = feedback.collect_prediction_feedback(h)

    assert out["n_batches"] == 5
    assert out["n_verified"] == 0
    assert out["accuracy_by_horizon"]["5"] == []       # 空，而不是 [0.0]
    assert out["recent_accuracy"]["5"] is None          # None，而不是 0.0


def test_empty_history_does_not_crash():
    out = feedback.collect_prediction_feedback([])
    assert out["n_batches"] == 0
    assert out["n_verified"] == 0
    assert out["pending_verify"] == 0


def test_corrupt_entries_are_skipped():
    h = [{"scan_date": "2026-10-01", "verifications": {"5": "垃圾数据"}},
         {"no_scan_date": 1}, "完全不是字典", _batch("2026-10-02", 5, 0.5)]
    out = feedback.collect_prediction_feedback(h)
    assert out["n_verified"] == 1


def test_score_buckets_taken_from_latest():
    h = [_batch("2026-10-01", 5, 0.5, buckets={">=80": -0.026, "60-80": -0.02}),
         _batch("2026-10-02", 5, 0.5, buckets={">=80": 0.01})]
    out = feedback.collect_prediction_feedback(h)

    assert out["score_buckets"]["scan_date"] == "2026-10-02"
    assert out["score_buckets"]["buckets"] == {">=80": 0.01}


def test_pending_counts_only_matured_batches():
    """待验证只算「已走满持有期」的 —— 昨天刚出的批次不该算进去。"""
    from datetime import datetime, timedelta

    today = datetime.now().strftime("%Y-%m-%d")
    old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
    fresh = today

    h = [_batch(old), _batch(fresh)]
    for e in h:
        e.pop("verifications")

    out = feedback.collect_prediction_feedback(h, today=today)

    # 老批次 5/10 两个窗口都到期 → 2；当天批次 0
    assert out["pending_verify"] == 2


def test_pending_excludes_already_verified():
    from datetime import datetime, timedelta

    today = datetime.now().strftime("%Y-%m-%d")
    old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")

    out = feedback.collect_prediction_feedback([_batch(old, 5)], today=today)
    # T+5 已验证，仅 T+10 待验
    assert out["pending_verify"] == 1


def test_pending_excludes_empty_unverifiable_batches():
    """空批次（没标的、也没快照）永远验不出来，不能算进「待验证」。

    真实事故（2026-10-16）：`--doctor` 报「另有 10 个已走满持有期但未验证 →
    跑 --run --verify」，而真跑了**永远是「新增 0 条」**。那 10 个来自 5 个
    `n_predictions: 0` 的空批次（7 月没选出标的的日子）：
    `tracker_v2.verify_predictions` 一进门就 `if not codes: return None`。

    计数口径必须和「能不能真的验」对齐 —— 否则就是一条自我循环的错误指令：
    天天催人补验，补了永远没结果。
    """
    from datetime import datetime, timedelta

    today = datetime.now().strftime("%Y-%m-%d")
    old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")

    empty = _batch(old)
    empty.pop("verifications")
    empty["predictions"] = []
    empty["n_predictions"] = 0

    out = feedback.collect_prediction_feedback([empty], today=today)
    assert out["pending_verify"] == 0, (
        f"空批次不该计为待验证（它永远验不出来），实际 {out['pending_verify']}")
    assert out["n_unverifiable"] == 1, "空批次应被单独报出来，免得以为数据丢了"


def test_batch_with_snapshot_but_no_predictions_is_still_verifiable():
    """有 price_snapshot 的批次即使 predictions 为空，也算可验证。

    快照里就有基准价，tracker 能反推收益 —— 不能一刀切按空批次丢掉。
    """
    from datetime import datetime, timedelta

    today = datetime.now().strftime("%Y-%m-%d")
    old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")

    e = _batch(old)
    e.pop("verifications")
    e["predictions"] = []
    e["n_predictions"] = 0
    e["price_snapshot"] = {"600000": 10.0}

    out = feedback.collect_prediction_feedback([e], today=today)
    assert out["pending_verify"] == 2, (
        f"有快照的批次两个窗口都应计为待验证，实际 {out['pending_verify']}")
    assert out["n_unverifiable"] == 0


def test_missing_snapshot_is_counted():
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
    h = [_batch(old), _batch(old)]
    out = feedback.collect_prediction_feedback(h)
    assert out["missing_snapshot"] == 2


# ------------------------------------------------------------------ ② 交易回流

def test_trade_feedback_without_loop_reports(tmp_path, monkeypatch):
    """闭环从未结算时必须如实报 0，而不是编一个收益出来。"""
    monkeypatch.setattr(feedback, "LOOP_REPORTS_DIR", tmp_path / "nope")
    monkeypatch.setattr(feedback, "PAPER_ACCOUNT_FILE", tmp_path / "nope.json")
    monkeypatch.setattr(feedback, "_guard_paths",
                        lambda: (tmp_path / "s.json", tmp_path / "j.jsonl"))

    out = feedback.collect_trade_feedback()

    assert out["n_settlements"] == 0
    assert out["equity_series"] == []
    assert out["trade_stats"] == {}


def test_trade_feedback_reads_reports_and_journal(tmp_path, monkeypatch):
    rd = tmp_path / "loop_reports"
    rd.mkdir()
    for d, eq in (("2026-10-01", 100000.0), ("2026-10-02", 101000.0),
                  ("2026-10-03", 99500.0)):
        (rd / f"loop_report_{d.replace('-', '')}.json").write_text(json.dumps({
            "date": d, "equity": eq, "day_pnl": 0.0, "day_pnl_pct": 0.0,
            "n_positions": 3, "cash": 1000.0, "market_value": eq - 1000.0,
        }), encoding="utf-8")

    j = tmp_path / "j.jsonl"
    j.write_text("\n".join(json.dumps(r) for r in [
        {"ts": "t", "event": "fill", "data": {"side": "buy", "qty": 100, "price": 10}},
        {"ts": "t", "event": "fill", "data": {"side": "sell", "qty": 100, "price": 11}},
        {"ts": "t", "event": "risk_block", "data": {"code": "600000"}},
        {"ts": "t", "event": "halt", "data": {"reason": "手动", "operator": "alice"}},
    ]), encoding="utf-8")

    monkeypatch.setattr(feedback, "LOOP_REPORTS_DIR", rd)
    monkeypatch.setattr(feedback, "PAPER_ACCOUNT_FILE", tmp_path / "pa.json")
    monkeypatch.setattr(feedback, "_guard_paths", lambda: (tmp_path / "s.json", j))

    out = feedback.collect_trade_feedback()

    assert out["n_settlements"] == 3
    assert [r["date"] for r in out["equity_series"]] == \
        ["2026-10-01", "2026-10-02", "2026-10-03"]
    assert out["trade_stats"]["fill"] == 2
    assert out["trade_stats"]["risk_block"] == 1
    assert out["notional_buy"] == 1000.0
    assert out["notional_sell"] == 1100.0
    assert out["last_halt"]["data"]["operator"] == "alice"


def test_trade_feedback_tolerates_corrupt_report(tmp_path, monkeypatch):
    rd = tmp_path / "loop_reports"
    rd.mkdir()
    (rd / "loop_report_bad.json").write_text("{ 坏 JSON", encoding="utf-8")
    (rd / "loop_report_20261001.json").write_text(
        json.dumps({"date": "2026-10-01", "equity": 1.0}), encoding="utf-8")

    monkeypatch.setattr(feedback, "LOOP_REPORTS_DIR", rd)
    monkeypatch.setattr(feedback, "PAPER_ACCOUNT_FILE", tmp_path / "pa.json")
    monkeypatch.setattr(feedback, "_guard_paths",
                        lambda: (tmp_path / "s.json", tmp_path / "j.jsonl"))

    out = feedback.collect_trade_feedback()
    assert out["n_settlements"] == 1


# ------------------------------------------------------------------ ③ 绩效指标

def test_metrics_insufficient_below_two_points():
    m = feedback.compute_metrics([])
    assert m["insufficient"] is True
    assert "total_return" not in m

    m = feedback.compute_metrics([{"date": "2026-10-01", "equity": 100.0}])
    assert m["insufficient"] is True
    assert m["n_days"] == 1


def test_metrics_basic_return_and_drawdown():
    series = [{"date": "2026-10-01", "equity": 100000},
              {"date": "2026-10-02", "equity": 110000},   # +10%
              {"date": "2026-10-03", "equity": 88000}]    # -20% → 回撤 20%
    m = feedback.compute_metrics(series)

    assert m["insufficient"] is False
    assert m["total_return"] == pytest.approx(-0.12)
    assert m["max_drawdown"] == pytest.approx(0.2)
    assert m["n_days"] == 3


def test_metrics_ignores_nonpositive_equity():
    """净值 0/负数（脏数据）不能进序列，否则收益率会除以 0。"""
    series = [{"date": "2026-10-01", "equity": 100000},
              {"date": "2026-10-02", "equity": 0},
              {"date": "2026-10-03", "equity": 105000}]
    m = feedback.compute_metrics(series)
    assert m["n_days"] == 2
    assert m["total_return"] == pytest.approx(0.05)
    assert m["max_drawdown"] == 0.0


def test_metrics_win_rate_and_pl_ratio():
    series = [{"date": "2026-10-01", "equity": 100},
              {"date": "2026-10-02", "equity": 110},   # +10%
              {"date": "2026-10-03", "equity": 105},   # -4.5%
              {"date": "2026-10-04", "equity": 115}]   # +9.5%
    m = feedback.compute_metrics(series)

    assert m["daily_win_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["profit_loss_ratio"] is not None


def test_metrics_turnover_from_notional():
    series = [{"date": "2026-10-01", "equity": 100000},
              {"date": "2026-10-02", "equity": 100000}]
    m = feedback.compute_metrics(series, n_fills=2, notional=50000.0)
    assert m["turnover"] == pytest.approx(0.5)
    assert m["n_fills"] == 2


# ------------------------------------------------------------------ 汇总与诊断

def test_collect_is_read_only(tmp_path, monkeypatch):
    """命脉：collect() 绝不能写盘。"""
    before = (REAL_HISTORY.read_bytes() if REAL_HISTORY.exists() else None,
              REAL_JOURNAL.read_bytes() if REAL_JOURNAL.exists() else None)

    monkeypatch.setattr(feedback, "LOOP_REPORTS_DIR", tmp_path / "none")
    monkeypatch.setattr(feedback, "_guard_paths",
                        lambda: (tmp_path / "s.json", tmp_path / "j.jsonl"))
    snap = feedback.collect()

    assert isinstance(snap, PerformanceSnapshot)
    after = (REAL_HISTORY.read_bytes() if REAL_HISTORY.exists() else None,
             REAL_JOURNAL.read_bytes() if REAL_JOURNAL.exists() else None)
    assert after == before, "collect() 污染了真实文件"


def test_collect_populates_health_and_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "LOOP_REPORTS_DIR", tmp_path / "none")
    monkeypatch.setattr(feedback, "_guard_paths",
                        lambda: (tmp_path / "s.json", tmp_path / "j.jsonl"))

    snap = feedback.collect()

    assert "prediction" in snap.health and "trade" in snap.health
    assert snap.health["trade"]["n_settlements"] == 0
    # 没有结算数据必须留下可执行提示
    assert any("尚未收盘结算" in n for n in snap.notes)


def test_diagnose_flags_zero_verification(tmp_path, monkeypatch):
    snap = PerformanceSnapshot(n_batches=25, n_verified=0)
    health, notes = feedback._diagnose(
        {"n_batches": 25, "n_verified": 0, "pending_verify": 31,
         "missing_snapshot": 0},
        {"n_settlements": 0, "journal_lines": 0, "guard": {}},
        snap)

    assert health["useful"] is False
    assert any("0 批完成验证" in n for n in notes)
    assert any("31" in n for n in notes)


def test_diagnose_flags_halted_guard():
    snap = PerformanceSnapshot(n_verified=5, n_settlements=10)
    _, notes = feedback._diagnose(
        {"n_batches": 5, "n_verified": 5, "pending_verify": 0},
        {"n_settlements": 10, "journal_lines": 100,
         "guard": {"halted": True, "halt_reason": "回撤熔断"}},
        snap)

    assert any("熔断" in n for n in notes)
    assert snap.is_useful() is True


def test_is_useful_thresholds():
    assert PerformanceSnapshot(n_verified=3).is_useful() is True
    assert PerformanceSnapshot(n_settlements=5).is_useful() is True
    assert PerformanceSnapshot(n_verified=2, n_settlements=4).is_useful() is False


def test_summary_and_render_do_not_crash_on_empty():
    snap = feedback.collect_prediction_feedback([])
    s = PerformanceSnapshot(n_batches=snap["n_batches"])
    assert "预测" in s.summary()

    text = feedback.render_text(s)
    assert "进化反馈快照" in text
    assert "数据不足" in text or "无验证结果" in text


def test_render_text_includes_metrics_when_available():
    snap = PerformanceSnapshot(
        generated_at="2026-10-14T13:00:00",
        n_settlements=3,
        metrics=feedback.compute_metrics([
            {"date": "2026-10-01", "equity": 100000},
            {"date": "2026-10-02", "equity": 110000},
        ]),
    )
    text = feedback.render_text(snap)
    assert "累计收益" in text
    assert "+10.00%" in text


def test_snapshot_to_json_roundtrips():
    snap = feedback.collect_prediction_feedback([])
    obj = PerformanceSnapshot(n_batches=snap["n_batches"], n_verified=1)
    parsed = json.loads(feedback.snapshot_to_json(obj))
    assert parsed["n_verified"] == 1

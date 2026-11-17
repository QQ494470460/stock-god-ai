# -*- coding: utf-8 -*-
"""选股质量审计（``src/core/pick_quality.py``）离线单测。

覆盖的都是**本项目真实踩过的形态**，不是虚构的数字：

- 48 日全期均值 +0.0237、t=0.88、近 20 日 −0.0099 → 必须判 ``decayed``
  （曾经可用、现在失效），而不是 ``marginal`` 或 ``effective``。
- ``rolling_ic_status.json`` 停在 ``running`` 超过 100 小时 → 必须判 ``stuck``。
- ``signal_strength`` 说"良"（σ 很分散）但预测力已衰减 → 必须报「假强信号」。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from src.core import pick_quality as pq


def _entries(ics, start="2026-09-01", n=425):
    """按日序列造台账条目（日期连续递增，仅用于排序与分月）。

    ``n`` 默认 425 = 全池口径（≈生产）。要造抽样口径传 ``n=150``。
    """
    d0 = datetime.strptime(start, "%Y-%m-%d")
    return [{"date": (d0 + timedelta(days=i)).strftime("%Y-%m-%d"),
             "ic": v, "n": n} for i, v in enumerate(ics)]


def _series(n, mean, noise=0.0):
    """造 n 个均值≈mean 的 IC：偶数位 +noise、奇数位 -noise。"""
    return [mean + (noise if i % 2 == 0 else -noise) for i in range(n)]


# ---------------------------------------------------------------------------
# 判定：effective / marginal / decayed / insufficient / no_data
# ---------------------------------------------------------------------------

def test_no_data_when_empty():
    s = pq.ic_stats([])
    assert s["verdict"] == "no_data"
    assert s["n"] == 0
    assert s["mean_ic"] is None


def test_insufficient_when_fewer_than_min_days():
    s = pq.ic_stats(_entries([0.10] * 5))
    assert s["verdict"] == "insufficient"
    assert s["n"] == 5


def test_effective_requires_both_significance_and_floor():
    # 均值 0.05、波动很小 → t 很大、过可用线，且近期也为正
    s = pq.ic_stats(_entries(_series(40, 0.05, noise=0.01)))
    assert s["verdict"] == "effective"
    assert s["t_stat"] >= pq.T_SIG
    assert s["mean_ic"] >= pq.IC_FLOOR


def test_marginal_when_positive_but_noise_too_big():
    # 均值 0.03 但波动 0.15 → t≈1.26，不显著；近期仍为正 → 不是 decayed
    s = pq.ic_stats(_entries(_series(40, 0.03, noise=0.15)))
    assert s["verdict"] == "marginal"
    assert s["t_stat"] < pq.T_SIG


def test_decayed_when_overall_mean_negative():
    s = pq.ic_stats(_entries(_series(40, -0.02, noise=0.05)))
    assert s["verdict"] == "decayed"
    assert "反向" in s["note"]


def test_decayed_when_overall_positive_but_recent_turned_negative():
    """本项目 2026-11-18 的真实形态：全期 +0.02 级别，近 20 日转负。"""
    ics = [0.05] * 28 + [-0.01] * 20
    s = pq.ic_stats(_entries(ics))
    assert s["mean_ic"] > 0
    assert s["recent_mean_ic"] < 0
    assert s["verdict"] == "decayed"
    assert "曾经可用" in s["note"]


def test_verdict_boundaries_are_explicit():
    assert pq.ic_verdict({"n": 0, "mean_ic": None}) == "no_data"
    assert pq.ic_verdict({"n": 5, "mean_ic": 0.2}) == "insufficient"
    assert pq.ic_verdict({"n": 40, "mean_ic": 0.0, "t_stat": 3.0}) == "decayed"
    asserted = pq.ic_verdict({"n": 40, "mean_ic": 0.1, "t_stat": 5.0,
                              "recent_mean_ic": 0.05})
    assert asserted == "effective"


# ---------------------------------------------------------------------------
# 统计口径
# ---------------------------------------------------------------------------

def test_recent_window_and_monthly_aggregation():
    ics = [0.1] * 10 + [-0.1] * 10 + [0.2] * 10 + [-0.2] * 10
    s = pq.ic_stats(_entries(ics, start="2026-09-01"), recent_window=20)
    assert s["recent_window"] == 20
    # 最近 20 个 = 第 3、4 段 → 均值 0
    assert abs(s["recent_mean_ic"]) < 1e-9
    assert s["monthly"], "月度分解不应为空"
    assert sum(v["n"] for v in s["monthly"].values()) == 40
    for v in s["monthly"].values():
        assert -1.0 <= v["mean_ic"] <= 1.0
        assert 0.0 <= v["pos_rate"] <= 1.0


def test_bad_rows_are_skipped_not_fatal():
    raw = [{"date": "2026-09-01", "ic": 0.1},
           {"date": "2026-09-02", "ic": None},
           {"date": "2026-09-03", "ic": float("nan")},
           {"date": "2026-09-04", "ic": "abc"},
           {"ic": 0.2},                       # 缺日期
           "not-a-dict"]
    s = pq.ic_stats(raw)
    assert s["n"] == 1
    assert s["verdict"] == "insufficient"


# ---------------------------------------------------------------------------
# 监控存活判定（本项目真故障：running 卡死 103 小时）
# ---------------------------------------------------------------------------

def test_monitor_stuck_when_running_too_long():
    now = datetime(2026, 11, 18, 2, 0)
    f = pq.monitor_freshness(
        _entries([0.1] * 5, start="2026-10-20"),
        {"status": "running", "started_at": "2026-11-13T19:15:00"},
        now=now,
    )
    assert f["state"] == "stuck"
    assert f["stalled"] is True
    assert f["running_hours"] > pq.RUNNING_STALE_HOURS


def test_monitor_running_but_recent_is_not_stuck():
    now = datetime(2026, 11, 18, 2, 0)
    f = pq.monitor_freshness(
        _entries([0.1] * 5, start="2026-11-13"),
        {"status": "running", "started_at": "2026-11-18T01:30:00"},
        now=now,
    )
    assert f["state"] != "stuck"


def test_monitor_stuck_when_running_without_started_at():
    f = pq.monitor_freshness([], {"status": "running"}, now=datetime(2026, 11, 18))
    assert f["state"] == "stuck"


def test_monitor_ok_when_fresh_and_idle():
    now = datetime(2026, 11, 18, 2, 0)
    f = pq.monitor_freshness(
        _entries([0.1] * 5, start="2026-11-16"),
        {"status": "ok", "at": "2026-11-18T01:00:00"},
        now=now,
    )
    assert f["state"] == "ok"
    assert f["stalled"] is False


def test_monitor_empty_when_never_produced():
    f = pq.monitor_freshness([], {"status": "noop"}, now=datetime(2026, 11, 18))
    assert f["state"] == "empty"
    assert f["stalled"] is True


# ---------------------------------------------------------------------------
# 落盘与渲染
# ---------------------------------------------------------------------------

def test_quality_report_writes_file_and_survives_missing_inputs(tmp_path):
    out = tmp_path / "pick_quality.json"
    q = pq.quality_report(log_path=tmp_path / "nope.json",
                          status_path=tmp_path / "nope_status.json",
                          out_path=out, now=datetime(2026, 11, 18, 2, 0))
    assert q["verdict"] == "no_data"
    assert out.exists(), "即使无数据也应落盘，供看板显示「无台账」"
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["verdict"] == "no_data"
    assert not (tmp_path / "pick_quality.json.tmp").exists(), "原子写不应留 .tmp"


def test_quality_report_end_to_end_with_real_shaped_log(tmp_path):
    log = tmp_path / "rolling_ic_log.json"
    log.write_text(json.dumps(_entries([0.05] * 28 + [-0.01] * 20)), encoding="utf-8")
    st = tmp_path / "status.json"
    st.write_text(json.dumps({"status": "running",
                              "started_at": "2026-11-13T19:15:00"}), encoding="utf-8")
    out = tmp_path / "pick_quality.json"
    q = pq.quality_report(log_path=log, status_path=st, out_path=out,
                          now=datetime(2026, 11, 18, 2, 0))
    assert q["verdict"] == "decayed"
    assert q["monitor"]["state"] == "stuck"
    assert out.exists()


def test_format_contains_verdict_and_recent():
    q = pq.ic_stats(_entries([0.05] * 28 + [-0.01] * 20))
    txt = pq.format_pick_quality(q)
    assert "选股预测力" in txt
    assert q["verdict_cn"] in txt
    assert "近 20 日" in txt


def test_format_handles_missing_quality():
    assert "无法评估" in pq.format_pick_quality(None)
    assert "无法评估" in pq.format_pick_quality({"verdict": "no_data"})


def test_alert_flags_false_strong_signal():
    """σ 很分散（signal=ok）但 RankIC 已衰减 → 必须点出「假强信号」。"""
    q = pq.ic_stats(_entries([0.05] * 28 + [-0.01] * 20))
    q["monitor"] = {"stalled": False}
    msg = pq.quality_alert(q, {"level": "ok", "level_cn": "良", "dispersion": 3.27})
    assert msg is not None
    assert "假强信号" in msg
    assert "3.27" in msg


def test_alert_flags_stalled_monitor_even_when_quality_ok():
    q = pq.ic_stats(_entries(_series(40, 0.05, noise=0.01)))
    assert q["verdict"] == "effective"
    q["monitor"] = {"stalled": True, "note": "质量监控停摆：台账最后一条是 2026-11-04"}
    msg = pq.quality_alert(q, {"level": "strong", "level_cn": "强"})
    assert msg is not None
    assert "停摆" in msg


def test_alert_is_none_when_healthy():
    q = pq.ic_stats(_entries(_series(40, 0.05, noise=0.01)))
    q["monitor"] = {"stalled": False}
    assert pq.quality_alert(q, {"level": "strong", "level_cn": "强"}) is None


@pytest.mark.parametrize("verdict", ["effective", "marginal", "decayed",
                                    "insufficient", "no_data"])
def test_every_verdict_has_cn_and_tone(verdict):
    assert verdict in pq._VERDICT_CN
    assert verdict in pq._VERDICT_TONE


# ---------------------------------------------------------------------------
# 状态自愈：被强杀留下的虚假 running 必须被校正（同一个真故障的另一半）
# ---------------------------------------------------------------------------

def test_heal_stale_status_rewrites_stuck_running(tmp_path):
    from src.core import rolling_ic as ri
    st = tmp_path / "status.json"
    st.write_text(json.dumps({"status": "running", "started_at": "2026-11-13T19:15:00",
                              "days": 5, "pool": "zz500"}), encoding="utf-8")
    out = ri.heal_stale_status(now=datetime(2026, 11, 18, 2, 0), path=st)
    assert out and out["status"] == "aborted"
    saved = json.loads(st.read_text(encoding="utf-8"))
    assert saved["status"] == "aborted"
    assert saved["started_at"] == "2026-11-13T19:15:00", "原始起始时间必须留证"
    assert saved["hours"] > 100


def test_heal_stale_status_leaves_fresh_running_alone(tmp_path):
    from src.core import rolling_ic as ri
    st = tmp_path / "status.json"
    body = {"status": "running", "started_at": "2026-11-18T01:30:00"}
    st.write_text(json.dumps(body), encoding="utf-8")
    assert ri.heal_stale_status(now=datetime(2026, 11, 18, 2, 0), path=st) is None
    assert json.loads(st.read_text(encoding="utf-8"))["status"] == "running"


def test_heal_stale_status_noop_when_not_running(tmp_path):
    from src.core import rolling_ic as ri
    st = tmp_path / "status.json"
    for body in ({"status": "ok"}, {"status": "noop"}, {}):
        st.write_text(json.dumps(body), encoding="utf-8")
        assert ri.heal_stale_status(now=datetime(2026, 11, 18), path=st) is None


def test_heal_stale_status_treats_missing_started_at_as_dead(tmp_path):
    from src.core import rolling_ic as ri
    st = tmp_path / "status.json"
    st.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    out = ri.heal_stale_status(now=datetime(2026, 11, 18), path=st)
    assert out and out["status"] == "aborted" and out["hours"] is None


def test_aborted_status_is_reported_as_stalled():
    """校正之后 monitor 仍要判 stalled —— 否则"改了状态"就等于"报平安"。"""
    f = pq.monitor_freshness(
        _entries([0.1] * 5, start="2026-10-20"),
        {"status": "aborted", "reason": "回填进程被强制终止：starting 103.2h 仍未收官"},
        now=datetime(2026, 11, 18),
    )
    assert f["state"] == "aborted"
    assert f["stalled"] is True
    assert "被强制终止" in f["note"]


# ---------------------------------------------------------------------------
# 口径拆分：混算均值就是 Simpson 悖论（本项目台账真实形态）
# ---------------------------------------------------------------------------

def test_sample_universe_never_yields_effective_verdict():
    """抽样口径哪怕 IC 再漂亮也不许判"有效"——它和生产口径不是一把尺子。"""
    s = pq.ic_stats(_entries(_series(40, 0.15, noise=0.01), n=150))
    assert s["universe"] == "sample"
    assert s["confidence"] == "low"
    assert s["verdict"] == "insufficient"
    assert "与生产口径" in s["note"]


def test_mixed_universes_are_reported_not_averaged():
    """复刻真实台账：全池 7 日 +0.157、抽样 41 日 +0.001。

    旧实现把两者混成 +0.0237（既不描述抽样也不描述生产）；新实现只认全池，
    并把两套口径都摊开给人看。
    """
    rows = _entries(_series(7, 0.157, noise=0.11), start="2026-09-07", n=425)
    rows += _entries(_series(41, 0.001, noise=0.19), start="2026-08-21", n=150)
    s = pq.ic_stats(rows)
    assert s["mixed"] is True
    assert s["universe"] == "full"
    assert s["n"] == 7, "主口径必须只含全池那 7 天，不能把 41 天抽样混进来"
    assert s["universes"]["sample"]["n"] == 41
    assert s["mean_ic"] > 0.10, "全池口径才是主结论，不能被抽样拉平"
    assert "混有两种口径" in s["note"]
    assert "混合平均没有意义" in s["note"]


def test_full_universe_with_enough_days_can_be_judged():
    s = pq.ic_stats(_entries(_series(40, 0.05, noise=0.01), n=425))
    assert s["universe"] == "full"
    assert s["confidence"] == "ok"
    assert s["verdict"] == "effective"


# ---------------------------------------------------------------------------
# 名单内名次 IC：从候选里挑前 N 只到底有没有用
# ---------------------------------------------------------------------------

def _hist(periods: int, strength: float, n_stocks: int = 20, horizon: str = "5"):
    """造历史推荐验证明细：strength=1 名次完全有效，0 则名次纯噪声。"""
    recs = []
    for p in range(periods):
        det = []
        for i in range(n_stocks):
            score = 100.0 - i * 3.0
            ret = strength * (score - 50.0) / 50.0 * 0.05 + (0.004 if i % 2 else -0.004)
            det.append({"code": f"{i:06d}", "score": score, "actual_return": ret})
        recs.append({"scan_date": f"2026-10-{p + 1:02d}",
                     "verifications": {horizon: {"details": det}}})
    return recs


def test_list_ic_flags_no_edge_when_rank_is_noise():
    s = pq.list_ic_stats(_hist(21, 0.0), horizon="5")
    assert s["n_periods"] >= 15
    assert abs(s["mean_ic"]) < 0.15
    assert s["verdict"] == "no_edge"
    assert "名次没有预测力" in s["note"]


def test_list_ic_detects_effective_rank():
    s = pq.list_ic_stats(_hist(21, 1.0), horizon="5")
    assert s["verdict"] == "effective"
    assert s["mean_ic"] > 0.8
    assert s["quantiles"]["top"] > s["quantiles"]["mid"]


def test_list_ic_insufficient_with_few_periods():
    s = pq.list_ic_stats(_hist(3, 1.0), horizon="5")
    assert s["verdict"] == "insufficient"


def test_list_ic_handles_missing_and_broken_records():
    assert pq.list_ic_stats(None)["verdict"] == "no_data"
    assert pq.list_ic_stats([])["verdict"] == "no_data"
    bad = [{"scan_date": "x"}, {"verifications": {}}, {"verifications": {"5": {}}},
           {"verifications": {"5": {"details": [{"score": None,
                                                 "actual_return": 0.1}]}}}]
    assert pq.list_ic_stats(bad)["verdict"] == "no_data"


def test_quality_report_includes_list_ic_and_horizon(tmp_path):
    log = tmp_path / "log.json"
    log.write_text(json.dumps(_entries(_series(40, 0.05, noise=0.01))), encoding="utf-8")
    hist = tmp_path / "hist.json"
    hist.write_text(json.dumps(_hist(21, 0.0)), encoding="utf-8")
    q = pq.quality_report(log_path=log, status_path=tmp_path / "s.json",
                          out_path=tmp_path / "q.json", history_path=hist,
                          now=datetime(2026, 11, 18))
    assert q["list_ic"]["n_periods"] == 21
    assert q["list_ic"]["horizon"] == "5"
    saved = json.loads((tmp_path / "q.json").read_text(encoding="utf-8"))
    assert saved["list_ic"]["n_periods"] == 21


def test_alert_prioritises_rank_no_edge_and_gives_action():
    q = {"verdict": "insufficient", "verdict_cn": "样本不足", "note": "口径说明",
         "monitor": {"stalled": False},
         "list_ic": {"verdict": "no_edge", "verdict_cn": "名次无信息",
                     "note": "名次没有预测力"}}
    msg = pq.quality_alert(q, {"level": "weak", "level_cn": "弱"})
    assert msg and "名次没有预测力" in msg
    assert "分散建仓" in msg, "必须给出可执行动作，而不只是抱怨名次没用"
    assert "调阈值或改排名权重都无效" in msg



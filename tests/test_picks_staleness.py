# -*- coding: utf-8 -*-
"""选股结果陈旧度 —— 让「安静地用旧票」变得可见。

为什么这个告警值得有测试
------------------------
``load_latest_picks()`` 会安静地拿「最近一份有 picks 的报告」。选股作业漏跑时，
它就变成几天前的旧票，而**计划文件、界面、审计看板全都显示正常**。
2026-11-05 实测：11-04 傍晚 loop 崩溃 → 18:45 选股作业没触发 → 当天补建的计划
吃到 11-03 的 picks，目标与 11-04 那份**逐位相同**，8 只里 4 只缺行业映射、
合计 43% 权重，全程零报错、零告警。

三条硬约束：
1. **不误报**：滞后 1 个交易日是**正常**的（D-1 收盘选股 → D 日计划）；
   拿不到报告日期时保持安静，只把原因写进 ``reason``。
2. **按交易日算**，不是日历日。周五的报告用于周一的计划 = 滞后 1，不是 3。
3. **落到计划文件里**，不只写日志 —— 否则 UI / 审计工具看不到。

为什么「拿不到报告日期时安静」是对的
------------------------------------
显式传 ``picks=`` 的调用（手动、测试、回放）本来就没有报告元信息，
每次都叫一声等于训练用户忽略告警。陈旧度只在能**证明**滞后时断言。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from loguru import logger  # noqa: E402

import src.trading.trade_plan as tp  # noqa: E402


# ---------------------------------------------------------------------------
# 构造器
# ---------------------------------------------------------------------------

def _picks(n: int = 3):
    return [
        {"代码": f"60{i:04d}", "名称": f"票{i}", "评分": 60.0 - i, "最新价": 10.0 + i}
        for i in range(n)
    ]


def _report(dirp: Path, day: str, hhmm: str = "1850", picks=None) -> Path:
    """写一份 daily_picks 报告；day 形如 ``20261105``。"""
    dirp.mkdir(parents=True, exist_ok=True)
    p = dirp / f"daily_picks_{day}_{hhmm}.json"
    p.write_text(json.dumps({
        "generated_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T{hhmm[:2]}:{hhmm[2:]}:00",
        "pool": "zz500", "min_score": 58.0, "regime": "bear",
        "picks": _picks() if picks is None else picks,
    }, ensure_ascii=False), encoding="utf-8")
    return p


def _capture_warnings():
    """收集 loguru 的 WARNING 及以上（返回 (msgs, cleanup)）。"""
    msgs: list[str] = []
    hid = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    return msgs, lambda: logger.remove(hid)


# ---------------------------------------------------------------------------
# 1. 报告日期从文件名取
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expect", [
    ("daily_picks_20261105_0139.json", "2026-11-05"),
    ("daily_picks_20261103_1850.json", "2026-11-03"),
    ("daily_picks_20261231.json", "2026-12-31"),          # 无时分
    ("daily_picks_20261332_0900.json", ""),               # 非法月日 → 不猜
    ("something_else.json", ""),
])
def test_report_date_comes_from_the_filename(name, expect):
    """报告日期取**文件名**：mtime 会被拷贝改写，generated_at 在历史报告里会缺。"""
    assert tp._report_date_of(Path(name)) == expect


# ---------------------------------------------------------------------------
# 2. 滞后判定
# ---------------------------------------------------------------------------

def test_one_trading_day_lag_is_the_normal_case():
    """D-1 收盘选股 → D 日计划，滞后 1 是设计值，不该报警。"""
    meta = {"report_date": "2026-11-04"}
    s = tp.picks_staleness(meta, "2026-11-05")
    assert s["lag_days"] == 1
    assert s["stale"] is False
    assert s["reason"] == ""


def test_missing_a_whole_session_is_flagged():
    """11-03 的报告拿去生成 11-06 的计划 = 中间漏了 11-04、11-05 两晚。"""
    s = tp.picks_staleness({"report_date": "2026-11-03"}, "2026-11-06")
    assert s["lag_days"] == 3
    assert s["stale"] is True
    assert "2026-11-03" in s["reason"] and "3 个交易日" in s["reason"]


def test_weekend_is_not_counted_as_lag():
    """周五的报告用于周一的计划 = 滞后 1 个交易日，不是 3。"""
    s = tp.picks_staleness({"report_date": "2026-11-06"}, "2026-11-09")
    assert s["lag_days"] == 1
    assert s["stale"] is False


def test_two_trading_days_is_the_threshold():
    """边界：恰好 2 个交易日就报警 —— 那意味着整整一个交易日的作业没跑。"""
    assert tp.picks_staleness({"report_date": "2026-11-05"}, "2026-11-09")["stale"] is True


def test_unknown_report_date_stays_quiet_but_says_why():
    """显式传 picks 的调用没有 meta；每次叫一声只会训练用户忽略告警。"""
    s = tp.picks_staleness({}, "2026-11-06")
    assert s["stale"] is False
    assert "未知" in s["reason"]


def test_calendar_failure_degrades_to_calendar_days_not_to_silence(monkeypatch):
    """日历不可用时退化成自然日差：只会**更容易**报警，绝不漏报。"""
    import src.trading.calendar as cal

    def boom(*a, **k):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr(cal, "trading_days_between", boom)
    s = tp.picks_staleness({"report_date": "2026-11-04"}, "2026-11-06")
    assert s["lag_days"] == 2          # 自然日差
    assert s["stale"] is True


# ---------------------------------------------------------------------------
# 3. load_latest_picks 会跳过空报告 —— 必须留痕
# ---------------------------------------------------------------------------

def test_empty_report_is_skipped_and_counted(tmp_path):
    """最新那份报告选股失败（picks=[]）时，会退到更早那份 —— 但要说出来。

    这是「今天的票选不出来」被悄悄替换成「用前天的票」的路径，
    不记录 skipped 数量就等于把一次失败抹掉了。
    """
    _report(tmp_path, "20261103")
    _report(tmp_path, "20261105", picks=[])          # 选股失败

    picks, meta = tp.load_latest_picks(tmp_path)
    assert picks, "应退回到 11-03 那份"
    assert meta["report_date"] == "2026-11-03"
    assert meta["skipped_reports"] == 1
    # 陈旧度必须把它带进告警文本，否则「为什么是旧票」无从得知
    s = tp.picks_staleness(meta, "2026-11-06")
    assert s["stale"] is True
    assert "空的" in s["reason"]


def test_corrupt_report_file_does_not_break_loading(tmp_path):
    _report(tmp_path, "20261103")
    (tmp_path / "daily_picks_20261105_1850.json").write_text("{not json", encoding="utf-8")
    picks, meta = tp.load_latest_picks(tmp_path)
    assert picks
    assert meta["report_date"] == "2026-11-03"
    assert meta["skipped_reports"] == 1


def test_load_latest_picks_picks_the_newest_by_name(tmp_path):
    _report(tmp_path, "20261103")
    _report(tmp_path, "20261105", "0139")
    _, meta = tp.load_latest_picks(tmp_path)
    assert meta["report_date"] == "2026-11-05"
    assert meta["report_file"] == "daily_picks_20261105_0139.json"


def test_empty_report_dir_is_not_an_error(tmp_path):
    picks, meta = tp.load_latest_picks(tmp_path)
    assert picks == []
    assert meta.get("skipped_reports") == 0


# ---------------------------------------------------------------------------
# 4. build_trade_plan 把新鲜度落到计划里 + 报警
# ---------------------------------------------------------------------------

def _build(monkeypatch, report_date: str, plan_date: str, skipped=0):
    meta = {"pool": "zz500", "regime": "bear", "report_date": report_date,
            "source": "daily_pick", "skipped_reports": skipped}
    monkeypatch.setattr(tp, "load_latest_picks", lambda *a, **k: (_picks(), meta))
    return tp.build_trade_plan(plan_date=plan_date, write=False)


def test_plan_records_which_report_it_ate(monkeypatch):
    """落到计划文件里 —— 只写日志的话 UI 和审计工具看不到。"""
    plan = _build(monkeypatch, "2026-11-04", "2026-11-05")
    assert plan.picks_date == "2026-11-04"
    assert plan.picks_lag_days == 1
    assert plan.plan_date == "2026-11-05"


def test_stale_picks_produce_a_loud_warning(monkeypatch):
    msgs, done = _capture_warnings()
    try:
        plan = _build(monkeypatch, "2026-11-03", "2026-11-06")
    finally:
        done()
    assert plan.picks_lag_days == 3
    assert any("旧选股结果" in m and "3 个交易日" in m for m in msgs), msgs


def test_fresh_picks_do_not_warn(monkeypatch):
    msgs, done = _capture_warnings()
    try:
        _build(monkeypatch, "2026-11-04", "2026-11-05")
    finally:
        done()
    assert not [m for m in msgs if "旧选股结果" in m], msgs


def test_explicit_picks_never_warn(monkeypatch):
    """显式传 picks（回放/手动/测试）没有 meta，不许因此叫一声。"""
    msgs, done = _capture_warnings()
    try:
        plan = tp.build_trade_plan(picks=_picks(), plan_date="2026-11-06", write=False)
    finally:
        done()
    assert plan.picks_lag_days == 0
    assert not [m for m in msgs if "旧选股结果" in m], msgs


def test_plan_date_is_computed_once(monkeypatch):
    """计划的有效日与新鲜度的基准日必须是同一天。

    曾经的两处 ``resolve_target_date()`` 各自调用会在跨午夜时得到两个日期，
    于是「计划属于明天、新鲜度却按今天算」。
    """
    calls = []
    monkeypatch.setattr(tp, "resolve_target_date", lambda *a, **k: calls.append(1) or "2026-11-06")
    monkeypatch.setattr(tp, "load_latest_picks",
                        lambda *a, **k: (_picks(), {"report_date": "2026-11-05"}))
    plan = tp.build_trade_plan(write=False)
    assert plan.plan_date == "2026-11-06"
    assert plan.picks_lag_days == 1
    assert len(calls) == 1, f"resolve_target_date 被调用 {len(calls)} 次"


# ---------------------------------------------------------------------------
# 5. 序列化：旧文件不带新字段，不许崩
# ---------------------------------------------------------------------------

def test_roundtrip_keeps_the_freshness_fields():
    plan = tp.TradePlan(plan_date="2026-11-06", picks_date="2026-11-04", picks_lag_days=2)
    back = tp.TradePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert (back.picks_date, back.picks_lag_days) == ("2026-11-04", 2)


def test_old_plan_files_without_the_new_fields_still_load():
    """历史计划文件没有这两个字段 —— 必须回落到「未知」而不是抛异常。"""
    back = tp.TradePlan.from_dict({"plan_date": "2026-10-30", "targets": []})
    assert back.picks_date == ""
    assert back.picks_lag_days == 0


def test_summary_flags_stale_picks():
    fresh = tp.TradePlan(plan_date="2026-11-05", picks_date="2026-11-04", picks_lag_days=1)
    stale = tp.TradePlan(plan_date="2026-11-06", picks_date="2026-11-03", picks_lag_days=3)
    assert "旧" not in fresh.summary() and "陈旧" not in fresh.summary()
    assert "陈旧" in stale.summary() and "3 个交易日" in stale.summary()


# ---------------------------------------------------------------------------
# 6. 审计工具要把它摆到第一屏
# ---------------------------------------------------------------------------

def test_audit_tool_surfaces_stale_picks():
    import scripts.plan_vs_fill as pvf

    base = {
        "plan_date": "20261106", "pending": False, "plan_source": "daily_pick",
        "plan_regime": "bear", "n_planned": 8, "weight_plan_total": 0.9,
        "weight_filled_total": 0.9, "fill_ratio": 1.0, "healthy": True,
        "n_missing": 0, "n_partially_filled": 0, "unmapped_weight_pct": 0.0,
        "unmapped_codes": [], "cash_pct": 0.1, "equity": 1_000_000.0, "positions": [],
    }
    out = pvf.render({**base, "picks_date": "2026-11-03", "picks_lag_days": 3,
                      "picks_stale": True})
    assert "选股结果陈旧" in out and "3 个交易日" in out

    ok = pvf.render({**base, "picks_date": "2026-11-05", "picks_lag_days": 1,
                     "picks_stale": False})
    assert "陈旧" not in ok and "2026-11-05" in ok

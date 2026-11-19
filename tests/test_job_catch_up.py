# -*- coding: utf-8 -*-
"""作业记账 + 「错过的窗口」补跑。

为什么需要这一层
----------------
APScheduler 用的是**内存 jobstore**：进程死掉 / 机器休眠期间的时间窗，它根本
不知道发生过，恢复后只从「现在」往后算下一次 —— 错过的就永久错过。
``misfire_grace_time`` 也救不了，它只覆盖「进程活着、但触发晚了」的窄窗口。

2026-11-04 实测：常驻 loop 傍晚崩溃（调度器与它同进程），18:30 数据更新 /
18:45 选股 / 19:00 进化三个作业全没触发，第二天**整天没有交易计划**，
而界面、心跳、日志一切正常。

测试要守住的四条边界
--------------------
1. **不许补得太远**：隔了一整天才补一个「收盘后」作业，计划属于后天，对今天
   毫无意义，只会白占 CPU。
2. **不许在盘中补**：补跑要扫全池、占满 CPU，跟盘中闭环抢资源。
3. **没有记账就不补**：无法区分「真的漏了」与「只是刚部署还没记过账」，
   宁可少补也不要每次启动都白跑一遍全池选股。
4. **只补决定「明天有没有计划」的两个作业**：evolution 会写策略参数，
   自动补跑一个会改状态的作业，风险远大于收益。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from loguru import logger  # noqa: E402

import src.scheduler.auto_scheduler as A  # noqa: E402

# 2026-11-05 是周四（交易日）；11-06 周五；11-07 周六；11-09 周一
THU = datetime(2026, 11, 5)


def _at(day: datetime, hh: int, mm: int = 0) -> datetime:
    return day.replace(hour=hh, minute=mm, second=0, microsecond=0)


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    """记账文件指向 tmp_path —— 绝不许碰真实的 data/job_runs.json。"""
    monkeypatch.setattr(A, "JOB_RUNS_PATH", tmp_path / "job_runs.json")
    return tmp_path / "job_runs.json"


def _mark(job: str, at: datetime) -> None:
    A._job_runs_mark(job, at)


def _missed(now: datetime, **kw):
    kw.setdefault("respect_trading_hours", False)
    return A.missed_jobs(now, **kw)


# ---------------------------------------------------------------------------
# cron 反推
# ---------------------------------------------------------------------------

def test_last_fire_time_on_a_normal_evening():
    """19:30 时，今晚的三个作业都已经该跑过了。"""
    now = _at(THU, 19, 30)
    assert A.last_fire_time("30 18 * * mon-fri", now) == _at(THU, 18, 30)
    assert A.last_fire_time("45 18 * * mon-fri", now) == _at(THU, 18, 45)


def test_last_fire_time_before_the_trigger_is_yesterday():
    """18:44 时，18:45 的选股还不到点 —— 最近一次是**昨天**那次。"""
    now = _at(THU, 18, 44)
    assert A.last_fire_time("45 18 * * mon-fri", now) == _at(THU - timedelta(days=1), 18, 45)


def test_last_fire_time_skips_the_weekend():
    """周六中午 → 最近一次是周五 18:45。工作日 cron 不许把周末算进去。"""
    sat = datetime(2026, 11, 7, 12, 0)
    assert A.last_fire_time("45 18 * * mon-fri", sat) == _at(datetime(2026, 11, 6), 18, 45)


def test_last_fire_time_handles_high_frequency_cron():
    """每小时整点：13:20 时最近一次是 13:00。别在枚举上算出 7 天前。"""
    assert A.last_fire_time("0 * * * *", _at(THU, 13, 20)) == _at(THU, 13, 0)


def test_bad_cron_is_not_an_exception():
    assert A.last_fire_time("not a cron", THU) is None


# ---------------------------------------------------------------------------
# 该不该补
# ---------------------------------------------------------------------------

def test_no_record_means_no_catch_up(_state):
    """没有记账记录时不许补 —— 无法区分「真漏了」和「刚部署还没记账」。

    否则每次启动都会白跑一遍全池选股。
    """
    assert _missed(_at(THU, 19, 30)) == []


def test_missed_job_is_detected(_state):
    """昨天 18:45 记过账，今天 19:30 还只有昨天那笔 → 今天的选股漏了。"""
    _mark("data_update", _at(THU - timedelta(days=1), 18, 30))
    _mark("daily_pick", _at(THU - timedelta(days=1), 18, 45))
    got = {m["job"] for m in _missed(_at(THU, 19, 30))}
    assert got == {"data_update", "daily_pick"}


def test_job_that_already_ran_today_is_not_missed(_state):
    _mark("data_update", _at(THU, 18, 31))
    _mark("daily_pick", _at(THU, 18, 52))
    assert _missed(_at(THU, 19, 30)) == []


def test_stale_window_is_not_caught_up(_state):
    """只漏了 data_update 时，第二天早上 08:00 距 18:30 已 13.5 小时 → 不补。

    隔了一整天再补一个「收盘后」作业，生成的计划属于后天，对今天毫无意义。
    """
    _mark("data_update", _at(THU - timedelta(days=7), 18, 30))
    _mark("daily_pick", _at(THU, 18, 52))
    now = _at(THU, 8, 0)                     # 开盘前，但已超过 12h 窗口
    assert _missed(now) == []


def test_catch_up_is_ordered_data_first(_state):
    """数据没更新就去选股 = 用旧行情选今天的票。顺序必须是 data_update 在前。"""
    _mark("data_update", _at(THU - timedelta(days=1), 18, 30))
    _mark("daily_pick", _at(THU - timedelta(days=1), 18, 45))
    assert [m["job"] for m in _missed(_at(THU, 19, 30))] == ["data_update", "daily_pick"]


def test_catch_up_never_touches_evolution(_state):
    """evolution 会写策略参数 —— 绝不能被自动补跑，哪怕它确实漏了。"""
    _mark("evolution", _at(THU - timedelta(days=3), 19, 0))
    assert "evolution" not in {m["job"] for m in _missed(_at(THU, 19, 30))}


# ---------------------------------------------------------------------------
# 盘中不许补
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("when,blocked", [
    (datetime(2026, 11, 5, 8, 59), False),     # 开盘前，允许
    (datetime(2026, 11, 5, 9, 0), True),       # 竞价开始就该让路
    (datetime(2026, 11, 5, 12, 0), True),      # 午休也算盘中
    (datetime(2026, 11, 5, 15, 30), True),
    (datetime(2026, 11, 5, 15, 31), False),    # 收盘后立刻可以补
    (datetime(2026, 11, 7, 12, 0), False),     # 周六：不是交易日，随便补
])
def test_market_window(when, blocked):
    assert A._market_window(when) is blocked


def test_market_window_short_circuits_the_whole_check(_state, monkeypatch):
    """盘中判定发生在最前面：即使真有漏跑的作业，也不许在这里跑。

    补跑要扫全池、占满 CPU 好几分钟，跟盘中闭环抢资源 —— 而盘中的计划
    已经由 ``TradingLoop._ensure_plan`` 兜底补建了。
    """
    _mark("data_update", _at(THU - timedelta(days=1), 18, 30))
    _mark("daily_pick", _at(THU - timedelta(days=1), 18, 45))
    now = _at(datetime(2026, 11, 6), 5, 0)      # 距 18:45 只有 10 小时，确定真的漏了
    assert A.missed_jobs(now, respect_trading_hours=False)    # 前提：不设限就该补

    monkeypatch.setattr(A, "_market_window", lambda *a: True)
    assert A.missed_jobs(now) == []


# ---------------------------------------------------------------------------
# 补跑执行
# ---------------------------------------------------------------------------

def test_catch_up_runs_both_jobs_in_order(_state):
    _mark("data_update", _at(THU - timedelta(days=1), 18, 30))
    _mark("daily_pick", _at(THU - timedelta(days=1), 18, 45))
    calls: list[str] = []

    def runner(jid):
        def _f():
            calls.append(jid)
            return {"job": jid, "status": "ok"}
        return _f

    res = A.catch_up_missed(_at(THU, 19, 30), runner=runner)
    assert calls == ["data_update", "daily_pick"]
    assert [r["status"] for r in res] == ["ok", "ok"]


def test_dry_run_reports_without_running(_state):
    _mark("data_update", _at(THU - timedelta(days=1), 18, 30))
    _mark("daily_pick", _at(THU - timedelta(days=1), 18, 45))
    calls: list[str] = []

    res = A.catch_up_missed(_at(THU, 19, 30), dry_run=True,
                            runner=lambda j: (lambda: calls.append(j)))
    assert calls == []
    assert all(r["status"] == "would_run" for r in res)


def test_one_failing_job_does_not_abort_the_next(_state):
    """数据更新炸了，选股仍要试 —— 选股失败的代价比数据旧更直接。"""
    _mark("data_update", _at(THU - timedelta(days=1), 18, 30))
    _mark("daily_pick", _at(THU - timedelta(days=1), 18, 45))
    ran: list[str] = []

    def runner(jid):
        def _f():
            ran.append(jid)
            if jid == "data_update":
                raise RuntimeError("boom")
            return {"status": "ok"}
        return _f

    res = A.catch_up_missed(_at(THU, 19, 30), runner=runner)
    assert ran == ["data_update", "daily_pick"]
    assert [r["status"] for r in res] == ["error", "ok"]


def test_nothing_missed_means_no_work(_state):
    _mark("data_update", _at(THU, 18, 31))
    _mark("daily_pick", _at(THU, 18, 52))
    assert A.catch_up_missed(_at(THU, 19, 30), runner=lambda j: pytest.fail("不该跑")) == []


# ---------------------------------------------------------------------------
# 记账本身
# ---------------------------------------------------------------------------

def test_tracked_marks_completion(_state):
    @A._tracked("demo")
    def job():
        return {"job": "demo", "status": "ok"}

    job()
    assert A._job_runs_load()["demo"]["at"]


def test_tracked_does_not_mark_a_lease_skip(_state):
    """「被别的进程占着」说明它正在跑，由那个进程记账。

    这里记了等于把「跳过」当成「跑过」，会让真正的漏跑永远发现不了。
    """

    @A._tracked("demo")
    def job():
        return {"job": "demo", "status": "skipped"}

    job()
    assert "demo" not in A._job_runs_load()


def test_tracked_keeps_the_function_name():
    """``run_all_once`` 与日志都按名字认作业，包装后名字不许变。"""

    @A._tracked("demo")
    def job_demo() -> dict:
        return {}

    assert job_demo.__name__ == "job_demo"


def test_marks_land_atomically(_state):
    """落盘不许留 .tmp 残骸（看板/记账会读到半截 JSON）。"""
    A._job_runs_mark("data_update", _at(THU, 18, 31))
    A._job_runs_mark("daily_pick", _at(THU, 18, 52))
    assert not list(_state.parent.glob("*.tmp"))
    assert set(json.loads(_state.read_text(encoding="utf-8"))) == {"data_update", "daily_pick"}


def test_corrupt_state_file_is_treated_as_empty(_state):
    _state.write_text("{not json", encoding="utf-8")
    assert A._job_runs_load() == {}
    # 而且还能重新写回去，不会因为文件坏了就永久瘫痪
    A._job_runs_mark("data_update", _at(THU, 18, 31))
    assert "data_update" in A._job_runs_load()


def test_missing_state_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "JOB_RUNS_PATH", tmp_path / "nope" / "job_runs.json")
    assert A._job_runs_load() == {}


# ---------------------------------------------------------------------------
# 首次建立基准
# ---------------------------------------------------------------------------

def test_first_run_bootstraps_instead_of_replaying_history(_state):
    """新部署的机器从没记过账 —— 补跑一次不许把历史上所有窗口都补一遍。

    否则部署当天就会白跑一遍全池选股（几分钟 CPU），而那天本来没漏任何东西。
    """
    assert not _state.exists()
    res = A.catch_up_missed(_at(THU, 19, 30), runner=lambda j: pytest.fail("不该跑"))
    assert res == []
    assert _state.exists()
    assert set(A._job_runs_load()) >= {"data_update", "daily_pick"}


def test_bootstrap_never_touches_an_existing_record(_state):
    """已有记账时绝不动它 —— 否则每次启动都等于「宣布所有作业刚跑过」，
    真正的漏跑就被抹掉了，而那正是这个机制要防的事。
    """
    _mark("daily_pick", _at(THU - timedelta(days=1), 18, 45))
    before = A._job_runs_load()["daily_pick"]["at"]

    assert A._job_runs_bootstrap(_at(THU, 19, 30)) is False
    assert A._job_runs_load()["daily_pick"]["at"] == before


def test_bootstrap_does_not_suppress_a_real_miss(_state):
    """基准建立**之后**再漏的，照样要补。"""
    A._job_runs_bootstrap(_at(THU - timedelta(days=1), 20, 0))
    got = {m["job"] for m in _missed(_at(THU, 19, 30))}
    assert got == {"data_update", "daily_pick"}


# ---------------------------------------------------------------------------
# 日线作业（daily_pick）按「最近一个已收盘交易日」补跑
#
# 背景（2026-11-19 实测）：``CATCH_UP_MAX_LAG_SEC=12h`` 对 18:45 的选股太短。
# loop 整晚不在是常态，18:45 每天都没触发，而补跑窗口到次日 06:45 就关 ——
# 实测每日选股只覆盖了约一半交易日，11-19 用的是 11-17 收盘数据算的计划，
# 界面却一切正常。判据换成「最近一个已收盘交易日到底做过没有」后，
# 只要没做过，补跑一定产出面向未来交易日的计划，永远不白跑。
# ---------------------------------------------------------------------------

def test_last_closed_trading_day_boundaries():
    """15:30 整还不算收盘（当天数据没落库），15:31 才算。"""
    assert A._last_closed_trading_day(datetime(2026, 11, 5, 15, 30)) == \
        datetime(2026, 11, 4).date()
    assert A._last_closed_trading_day(_at(THU, 15, 31)) == THU.date()
    # 周六 → 最近已收盘的是周五
    assert A._last_closed_trading_day(datetime(2026, 11, 7, 12, 0)) == \
        datetime(2026, 11, 6).date()
    # 周一开盘前 → 上周五
    assert A._last_closed_trading_day(datetime(2026, 11, 9, 8, 0)) == \
        datetime(2026, 11, 6).date()


def test_daily_pick_missed_last_night_is_caught_up_next_morning(_state):
    """**本条是这次改动的全部意义**：昨晚 18:45 漏了，今早 08:00 仍要补。

    按旧的 12h 窗口，08:00 距 18:45 已 13.25h → 判「作废」，于是整整一个
    交易日用的是两天前的信号。而补跑产出的是「面向下一个交易日」的计划，
    开盘前补完全有意义。
    """
    _mark("data_update", _at(THU, 18, 30))
    _mark("daily_pick", _at(THU - timedelta(days=2), 18, 45))   # 前天做过
    nxt = _at(datetime(2026, 11, 6), 8, 0)                      # 次日开盘前
    got = {m["job"]: m for m in _missed(nxt)}
    assert "daily_pick" in got
    assert got["daily_pick"]["basis"] == "closed_day"


def test_daily_pick_before_closed_hour_waits_for_cron(_state):
    """15:31 时不补 —— 还没到「该做完」的 17:00，等 cron 自己跑。

    若此处就补，补完 at=15:31 < 当日 17:00，十分钟后又判「没做」→ 死循环。
    """
    _mark("data_update", _at(THU, 18, 30))
    _mark("daily_pick", _at(THU - timedelta(days=1), 18, 45))
    assert [m["job"] for m in _missed(_at(THU, 15, 31))] == []


def test_daily_pick_caught_up_once_closed_hour_passes(_state):
    _mark("data_update", _at(THU, 18, 30))
    _mark("daily_pick", _at(THU - timedelta(days=1), 18, 45))
    got = [m for m in _missed(_at(THU, 17, 1)) if m["job"] == "daily_pick"]
    assert got and got[0]["basis"] == "closed_day"


def test_daily_pick_not_repeated_after_it_ran(_state):
    """补跑/正常跑过之后（at 已过当日 17:00）不许再补 —— 防每 10 分钟扫一次全池。"""
    _mark("data_update", _at(THU, 18, 30))
    _mark("daily_pick", _at(THU, 17, 5))
    assert [m["job"] for m in _missed(_at(THU, 19, 30))] == []


def test_daily_pick_without_record_is_still_skipped(_state):
    """无记账照旧不补 —— 无法区分「真漏了」和「刚部署没记过账」。"""
    assert _missed(_at(THU, 19, 30)) == []


def test_data_update_keeps_the_old_window(_state):
    """data_update 仍按 12h 窗口走，不被这次改动波及。

    次日 08:00 距昨天 18:30 已 13.5h → 超窗口，不补；daily_pick 昨晚已做。
    """
    _mark("data_update", _at(THU, 18, 30))
    _mark("daily_pick", _at(THU, 18, 52))
    nxt = _at(datetime(2026, 11, 6), 8, 0)
    assert _missed(nxt) == []


def test_calendar_unavailable_falls_back_to_lag_window(_state, monkeypatch):
    """日历挂掉时退回原来的秒数窗口，绝不因此少补或多补。"""
    monkeypatch.setattr(A, "_last_closed_trading_day", lambda *a, **k: None)
    _mark("data_update", _at(THU, 18, 31))
    _mark("daily_pick", _at(THU - timedelta(days=1), 18, 45))
    got = _missed(_at(THU, 19, 30))
    assert [m["job"] for m in got] == ["daily_pick"]
    assert got[0].get("basis") is None      # 走的是 lag 分支

# -*- coding: utf-8 -*-
"""调度器必须把 pick_ensemble 真的传给 generate_daily_picks。

为什么单独测这个
----------------
UI 开关 + 持久化都通过了，也不代表复核真的会跑：中间还隔着
「调度作业有没有把参数传下去」。漏传的话，开关会变成一个完美保存、
永久生效、但完全无害的假开关。这一层专门堵这个漏。

用 monkeypatch 替换掉选股与交易计划，只验证参数传递，不碰网络与磁盘。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import src.scheduler.auto_scheduler as sch  # noqa: E402


@pytest.fixture
def spy(monkeypatch):
    """替换选股与交易计划，记录收到的 use_ensemble。"""
    seen = {}

    def _fake_picks(write_report=True, use_ensemble=False, **kw):
        seen["use_ensemble"] = bool(use_ensemble)
        seen["write_report"] = bool(write_report)
        return None, {"status": "ok", "n_passed": 0, "picks": [],
                      "pool": "", "min_score": 0, "regime": ""}

    monkeypatch.setattr(sch, "generate_daily_picks", _fake_picks)

    class _P:
        plan_date = "2000-01-01"
        targets: list = []

    def _fake_plan(**kw):
        seen["plan_written"] = bool(kw.get("write"))
        return _P()

    import src.trading.trade_plan as tp
    monkeypatch.setattr(tp, "build_trade_plan", _fake_plan)
    return seen


@pytest.mark.parametrize("on", [True, False])
def test_flag_reaches_daily_picks(spy, monkeypatch, on):
    monkeypatch.setattr(sch.app_config.scheduler, "pick_ensemble", on)

    info = sch._job_daily_pick_impl(dry_run=True)

    assert spy["use_ensemble"] is on, f"开关 {'开' if on else '关'}了，却没传下去"
    assert info["status"] == "ok"


def test_dry_run_does_not_write(spy):
    sch._job_daily_pick_impl(dry_run=True)

    assert spy["write_report"] is False
    assert spy["plan_written"] is False

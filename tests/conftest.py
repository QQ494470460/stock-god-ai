# -*- coding: utf-8 -*-
"""全局测试夹具。

为什么需要一条「作业记账不许落生产」的兜底
------------------------------------------
``@_tracked`` 会在每个作业跑完后往 ``data/job_runs.json`` 记一笔，而
``test_scheduler`` 一族本来就会直接调用 ``job_*()`` / ``run_all_once()``。
于是**跑一次回归就把生产记账改写成测试时间戳** —— 而这份记账正是「该不该补跑
错过的作业」的唯一依据：被污染之后，真正的漏跑会被当成「刚跑过」而永远不补，
且现场看起来完全正常。

2026-11-05 实测：一次全量回归把 ``data/job_runs.json`` 写成了一组错乱时间戳
（``industry_map`` 02:00:03 / ``data_update`` 02:01:32 / ``drift_retrain`` 02:03:37），
而当时生产上跑着的进程一个作业都没跑过。

这与「测试不许碰五大状态文件」是同一条纪律，只是这份记账是新加的。

2026-11-18 又踩到同一类：新增的选股质量诊断 ``quality_report()`` 与
``rolling_ic.heal_stale_status()`` 都按**模块级常量路径**落盘，而
``test_scheduler`` 一族会真的调 ``start_scheduler()`` / ``job_daily_pick()``。
实测跑一次 ``tests/test_scheduler.py`` 就把生产 ``data/rolling_ic_status.json``
从 ``running`` 改成了 ``aborted``、把 ``data/pick_quality.json`` 重写了一遍 ——
**诊断文件被测试改写**，正是这份兜底要防的事。所以这里一并指向 tmp_path。
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_job_runs(tmp_path, monkeypatch):
    """把所有测试的作业记账指向 tmp_path，绝不碰生产的 data/job_runs.json。"""
    try:
        import src.scheduler.auto_scheduler as auto_scheduler
    except Exception:  # noqa: BLE001  个别测试环境可能装不上依赖，交给用例自己失败
        return
    monkeypatch.setattr(auto_scheduler, "JOB_RUNS_PATH", tmp_path / "job_runs.json")


@pytest.fixture(autouse=True)
def _isolate_diagnostics(tmp_path, monkeypatch):
    """观测类落盘（质量诊断 / IC 状态）也不许碰生产 data/。

    两处都改**模块级常量**：
    - ``pick_quality.OUT_FILE``    —— ``quality_report()`` 的默认落盘目标
    - ``rolling_ic.STATUS_FILE``   —— ``heal_stale_status()`` / ``run_backfill()`` 的状态文件
    """
    try:
        import src.core.pick_quality as pick_quality
        monkeypatch.setattr(pick_quality, "OUT_FILE", tmp_path / "pick_quality.json")
    except Exception:  # noqa: BLE001
        pass
    try:
        import src.core.rolling_ic as rolling_ic
        monkeypatch.setattr(rolling_ic, "STATUS_FILE", tmp_path / "rolling_ic_status.json")
    except Exception:  # noqa: BLE001
        pass


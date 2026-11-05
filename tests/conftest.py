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

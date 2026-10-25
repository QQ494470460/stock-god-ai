# -*- coding: utf-8 -*-
"""常驻调度进程 ``scripts/scheduler_daemon.py``。

为什么值得测
------------
收盘后作业（数据更新 / 每日推荐 / 进化验证）此前只活在看板进程里，
它挂掉的表现**不是报错，而是「今天没有新报告」** —— 极其容易被当成
"今天没行情"。所以这个进程必须留下可判定的存活证据（心跳），
并且绝不能多开一份把选股跑两遍。

心跳的判据必须是文件里的 ``ts``，不是文件 mtime：进程被杀时文件会保留
最后的内容，mtime 也会停在那一刻，看上去"刚刚还活着"，实际早就死了。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_daemon():
    spec = importlib.util.spec_from_file_location(
        "scheduler_daemon_under_test", ROOT / "scripts" / "scheduler_daemon.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


daemon = _load_daemon()


class _FakeJob:
    def __init__(self, jid, next_run):
        self.id = jid
        self.next_run_time = next_run


class _FakeSched:
    def __init__(self, jobs):
        self._jobs = jobs

    def get_jobs(self):
        return self._jobs


@pytest.fixture
def hb_path(tmp_path, monkeypatch):
    p = tmp_path / "scheduler_heartbeat.json"
    monkeypatch.setattr(daemon, "_heartbeat_path", lambda: p)
    return p


def _read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def test_heartbeat_records_pid_and_ownership(hb_path):
    daemon._write_heartbeat(None, False, hb_path)
    d = _read(hb_path)
    assert d["pid"] == os.getpid()
    assert d["holding"] is False


def test_heartbeat_carries_next_run_times(hb_path):
    nxt = datetime(2026, 10, 26, 18, 45, 0)
    sched = _FakeSched([_FakeJob("daily_pick", nxt), _FakeJob("evolution", None)])
    daemon._write_heartbeat(sched, True, hb_path)

    d = _read(hb_path)
    assert d["holding"] is True
    jobs = {j["id"]: j["next_run"] for j in d["jobs"]}
    assert jobs["daily_pick"] == "2026-10-26 18:45:00"
    assert jobs["evolution"] is None, "没有下次时间的作业要显式留空，不能编一个"


def test_heartbeat_age_is_computable(hb_path):
    """看板靠 ts 算年龄判断死活 —— ts 必须是可解析的 ISO 时间。"""
    daemon._write_heartbeat(None, False, hb_path)
    d = _read(hb_path)
    age = (datetime.now() - datetime.fromisoformat(d["ts"])).total_seconds()
    assert 0 <= age < 10, f"心跳时间戳解析后年龄异常: {age}"


def test_heartbeat_write_is_atomic(hb_path):
    """先写 .tmp 再 os.replace：看板不会读到写了一半的 JSON。"""
    daemon._write_heartbeat(None, False, hb_path)
    assert hb_path.exists()
    assert not hb_path.with_suffix(".tmp").exists(), "临时文件没被替换掉"
    _read(hb_path)      # 能正常解析


def test_heartbeat_never_crashes_on_bad_sched(hb_path):
    class _Broken:
        def get_jobs(self):
            raise RuntimeError("APScheduler 内部炸了")

    daemon._write_heartbeat(_Broken(), True, hb_path)
    d = _read(hb_path)
    assert d["jobs"] == [], "拿不到作业列表也要留下存活证据，不能整个不写心跳"


def test_daemon_uses_its_own_lock_not_the_scheduler_lock():
    """常驻进程锁 vs APScheduler 归属锁是两回事，不能混用同一个文件。

    混用的话：常驻进程占着 ``.scheduler.lock`` 当"我在跑"，看板的
    ``is_running()`` 就会一直返回 True，即使 APScheduler 根本没起来。
    """
    src = (ROOT / "scripts" / "scheduler_daemon.py").read_text(encoding="utf-8")
    lines = [ln for ln in src.splitlines() if "lock_path = " in ln]
    assert len(lines) == 1, f"期望只有一处定义锁路径，实际 {len(lines)} 处"
    assert ".scheduler_daemon.lock" in lines[0]
    assert "lock_path" in "".join(
        ln for ln in src.splitlines() if "acquire_lock(" in ln and "lock_path)" in ln
    ), "拿锁时必须用上面那把锁，不能图省事复用 .scheduler.lock"

# -*- coding: utf-8 -*-
"""共用单实例锁 ``src.core.singleton``。

重点不是「能上锁」，而是**探针与主程序口径一致**：
批处理启动器靠探针的退出码决定要不要再拉一份。要是探针以为空闲、
主程序却会被拒（或反过来），就会出现

* 探针报空闲 → 启动器拉起 → 主程序被拒 → 白跑（良性）；
* 更糟的是反过来：探针报占用 → 启动器永远不拉 → **常驻进程整天没起来**。

所以这里全部用**真实的另一个进程**持锁来验证，而不是同进程内自己锁自己。
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.core.singleton import LOCK_DISABLED, acquire_lock, probe_lock, release_lock  # noqa: E402

PY = sys.executable

# 子进程脚本：拿锁 → 持 N 秒 → 退出。持锁期间父进程去探测。
_HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    sys.path.insert(0, r"{root}")
    from src.core.singleton import acquire_lock
    fd = acquire_lock(Path(r"{lock}"))
    if fd is None:
        sys.exit(9)
    print("HELD", flush=True)
    time.sleep({hold})
    """
)


def _holder(lock: Path, hold: float = 3.0) -> subprocess.Popen:
    code = _HOLDER.format(root=ROOT, lock=lock, hold=hold)
    return subprocess.Popen([PY, "-c", code],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)


def test_cross_process_probe_reports_busy(tmp_path):
    """别的进程真持着锁时，探针必须报「占用」—— 这是启动器的唯一判据。"""
    lock = tmp_path / ".x.lock"
    proc = _holder(lock)
    try:
        assert proc.stdout.readline().strip() == "HELD"
        assert probe_lock(lock) is False, "别人持锁却报空闲，启动器会再拉一份"
        assert acquire_lock(lock, quiet=True) is None, "别人持锁却拿到了锁"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_cross_process_probe_reports_free_after_release(tmp_path):
    """锁被释放后探针要报空闲，否则常驻进程永远不会被拉起。"""
    lock = tmp_path / ".x.lock"
    proc = _holder(lock, hold=0.2)
    try:
        assert proc.stdout.readline().strip() == "HELD"
    finally:
        proc.wait(timeout=10)
    assert probe_lock(lock) is True


def test_probe_does_not_leave_the_lock_held(tmp_path):
    """探针用完必须立刻放手，否则后面的真实例会被自己的探针挡住。"""
    lock = tmp_path / ".x.lock"
    assert probe_lock(lock) is True
    fd = acquire_lock(lock)
    assert fd is not None
    release_lock(fd)


def test_acquire_then_probe_is_busy_in_process(tmp_path):
    lock = tmp_path / ".x.lock"
    fd = acquire_lock(lock)
    try:
        assert fd is not None
        assert probe_lock(lock) is False
    finally:
        release_lock(fd)
    assert probe_lock(lock) is True


def test_distinct_paths_are_independent(tmp_path):
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    fa = acquire_lock(a)
    fb = acquire_lock(b)
    try:
        assert fa is not None and fb is not None
    finally:
        release_lock(fa)
        release_lock(fb)


def test_lock_file_records_pid(tmp_path):
    lock = tmp_path / ".x.lock"
    fd = acquire_lock(lock)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        content = os.read(fd, 4096).decode("utf-8")
        assert f"pid={os.getpid()}" in content
        assert "since=" in content
    finally:
        release_lock(fd)


def test_unwritable_dir_fails_open_with_sentinel(tmp_path, monkeypatch):
    """连锁文件都建不出来 → 返回哨兵值（放行），不是 None（拒绝）。"""
    import src.core.singleton as sg

    lock = tmp_path / "nope" / ".x.lock"
    lock.parent.mkdir()

    def _noent(path, flags, *a, **k):
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(sg.os, "open", _noent)
    assert acquire_lock(lock) == LOCK_DISABLED


def test_write_open_denied_is_refused(tmp_path, monkeypatch):
    """写打开被拒 = 别人正持锁 = 必须拒绝，绝不能 fail-open。"""
    import src.core.singleton as sg

    lock = tmp_path / ".x.lock"
    lock.write_text("pid=99999\n")

    def _denied(path, flags, *a, **k):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(sg.os, "open", _denied)
    assert acquire_lock(lock) is None


def test_release_of_sentinel_is_a_no_op():
    """哨兵值不是真 fd，release 必须忽略它，否则会关掉一个非法句柄。"""
    release_lock(LOCK_DISABLED)
    release_lock(None)


def test_missing_parent_dir_is_created(tmp_path):
    lock = tmp_path / "nested" / "deep" / ".x.lock"
    fd = acquire_lock(lock)
    try:
        assert lock.exists()
    finally:
        release_lock(fd)


@pytest.mark.parametrize("name", [".run_loop.lock", ".scheduler_daemon.lock"])
def test_probe_cli_agrees_with_main_program(tmp_path, name):
    """闭环与调度常驻进程各有一个锁文件，探针与主程序必须对每个都说同一句话。"""
    lock = tmp_path / name
    fd = acquire_lock(lock)
    try:
        assert fd is not None
        assert probe_lock(lock) is False
    finally:
        release_lock(fd)
    assert probe_lock(lock) is True

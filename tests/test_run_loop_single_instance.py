# -*- coding: utf-8 -*-
"""闭环单实例锁测试。

背景：加了开机自启之后，最容易出现「计划任务拉起一份 + 手动又开一份」。
闭环多开一个实例，就会对同一份交易计划**再下一遍单**，而且两边的纸账户与
风控各有一份内存副本，对账必然错乱。

用 OS 级文件锁（Windows msvcrt / POSIX flock）而不是 PID 文件：
进程无论正常退出还是崩溃，操作系统都会释放锁，不会留下"假死锁"。
也**不能**用 ``os.kill(pid, 0)`` 探活 —— Windows 上它会真的 TerminateProcess。
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_run_loop():
    spec = importlib.util.spec_from_file_location(
        "run_loop_under_test", ROOT / "scripts" / "run_loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_loop = _load_run_loop()


def test_acquire_then_second_acquire_fails(tmp_path):
    """同一个锁文件只能被一个持有者拿到 —— 第二个必须失败。"""
    lock = tmp_path / ".run_loop.lock"
    assert run_loop.acquire_single_instance(lock) is True
    assert lock.exists()
    assert run_loop.acquire_single_instance(lock) is False, \
        "第二个实例竟然也拿到了锁，会重复下单"


def test_different_lock_paths_are_independent(tmp_path):
    """不同锁文件互不影响（测试/演练可以各用各的）。"""
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    assert run_loop.acquire_single_instance(a) is True
    assert run_loop.acquire_single_instance(b) is True


def test_lock_file_records_pid(tmp_path):
    """锁文件里写 pid + since，便于排查"谁占着"。"""
    import os
    lock = tmp_path / ".run_loop.lock"
    assert run_loop.acquire_single_instance(lock) is True
    # 自己持有锁时其他句柄读不了被锁字节，用已持有的句柄读
    h = run_loop._instance_lock_handle
    h.seek(0)
    content = h.read()
    assert f"pid={os.getpid()}" in content
    assert "since=" in content


def test_missing_parent_dir_is_created(tmp_path):
    """锁目录不存在时要自动创建，不能因为目录缺失就拒绝启动。"""
    lock = tmp_path / "nested" / "deep" / ".run_loop.lock"
    assert run_loop.acquire_single_instance(lock) is True
    assert lock.exists()


# ---------------------------------------------------------------------------
# 跨进程路径：Windows 上「写打开被拒」= 别人正锁着 = 必须 fail-CLOSED
# ---------------------------------------------------------------------------

def test_write_open_denied_is_treated_as_locked(tmp_path, monkeypatch):
    """回归：真机第二个实例拿到 Permission denied 后竟放行了，真跑了一轮 tick。

    背景：``open(lock_path, "a+")`` 在另一个进程持锁时会在 **open 阶段**就抛
    PermissionError（Windows 拒绝任何可写打开）。当时那段把它归到
    「锁不可用 → fail-open」，于是第二个实例照常启动、照常 tick、还覆盖了心跳。
    「写打开被拒」必须是 fail-CLOSED。
    """
    lock = tmp_path / ".run_loop.lock"
    lock.write_text("pid=99999\n")          # 文件已存在 = 疑似别人在用

    real_open = run_loop.os.open

    def _denied(path, flags, *a, **k):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(run_loop.os, "open", _denied)
    try:
        assert run_loop.acquire_single_instance(lock) is False, \
            "写打开被拒却放行了 —— 会造成双实例重复下单"
    finally:
        monkeypatch.setattr(run_loop.os, "open", real_open)


def test_open_error_without_file_fails_open(tmp_path, monkeypatch):
    """锁文件根本建不出来（如只读目录）时放行 —— 宁可能跑，不可不跑。"""
    lock = tmp_path / "readonly" / ".run_loop.lock"
    lock.parent.mkdir()

    def _noent(path, flags, *a, **k):
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(run_loop.os, "open", _noent)
    assert run_loop.acquire_single_instance(lock) is True

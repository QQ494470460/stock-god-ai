# -*- coding: utf-8 -*-
"""进程级单实例锁 —— 常驻进程（交易闭环 / 收盘后调度）共用同一套机制。

为什么抽成模块
--------------
三个进程要用同一把锁的思路：

* ``scripts/run_loop.py``      交易闭环，多开会**对同一份计划再下一遍单**
* ``scripts/scheduler_daemon.py`` 收盘后调度，多开会把选股/进化作业跑两遍
* ``scripts/loop_lock_probe.py``  批处理启动器的探针，必须和上面**机制完全一致**

曾经这段逻辑被复制过一次，结果两处的判定口径悄悄分叉：一处用
「``os.open`` 是否成功」，另一处用「``msvcrt.locking`` 是否抛错」。
Windows 上前者在别人持锁时**照样成功**，于是探针永远报「空闲」，
两个实例同时在跑却毫无察觉。共用一份实现，口径就不可能再分叉。

机制（Windows / POSIX 通用）
----------------------------
``msvcrt.locking(LK_NBLCK)`` / ``fcntl.flock(LOCK_EX|LOCK_NB)``
对文件首字节加排他锁。进程正常退出、崩溃、被 TerminateProcess，
操作系统都会释放，不会留下「假死锁」。

⚠️ 判定只能用 locking 的结果：

* ``os.open(path, O_RDWR)`` 在别人持锁时**照旧成功**（字节区间锁只拦
  对锁定区间的读写，不拦 open）；
* ``open(path, "a+")`` 走 C 运行时 fopen，共享模式不同，它才会报
  ``PermissionError``。两个 API 语义不一样，**别混用**。

⚠️ 别用 ``os.kill(pid, 0)`` 探活 —— Windows 上它会真的 TerminateProcess，
把正在跑的进程杀掉。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# 锁不可用（连文件都建不出来，例如只读目录/权限受限）时的返回值。
# 语义是「宁可让进程跑起来，也不能因为锁这件小事让交易停摆」——
# 闭环整天不跑的代价比极端情况下的双实例高得多，所以这条要保持 fail-open。
LOCK_DISABLED = -1


def _lock(fd) -> bool:
    """对 fd 的首字节加排他锁。成功 True，已被别人持有 False。"""
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:                # pragma: no cover - POSIX
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def probe_lock(lock_path: Path) -> bool:
    """探测锁是否空闲（**不持有**）。True = 空闲可启动，False = 已有实例。

    探针用完立刻关闭句柄，不会把锁留给后面真正的实例。
    """
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 连目录都建不出来：这是主程序该炸的错，探针不背这个锅。
        return True
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    except PermissionError:
        return False
    except OSError:
        # 连锁文件都建不出来：交给主程序自己判断，不在这里拦死。
        return True
    try:
        return _lock(fd)
    finally:
        os.close(fd)


def acquire_lock(lock_path: Path, quiet: bool = False) -> Optional[int]:
    """拿锁。

    返回值（**都必须由调用方持有，别关**）：

    * ``>= 0``  拿到锁，返回值就是 fd —— 进程存活期间不能 close，否则锁立刻失效
    * ``LOCK_DISABLED(-1)`` 锁机制不可用，但**允许继续运行**（fail-open，见上）
    * ``None``  已被另一个实例持有，**必须拒绝启动**

    崩溃时 OS 会自动释放锁，所以不需要 atexit 兜底。
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    existed = lock_path.exists()
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    except PermissionError:
        if not quiet:
            print(f"⚠️  {lock_path} 正被另一个实例锁定（写打开被拒）")
        return None
    except OSError as e:
        if existed:
            if not quiet:
                print(f"⚠️  无法打开单实例锁 {lock_path}: {e}")
            return None
        if not quiet:
            print(f"⚠️  无法创建单实例锁 {lock_path}: {e}（锁不可用，继续运行）")
        return LOCK_DISABLED
    if not _lock(fd):
        os.close(fd)
        return None
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"pid={os.getpid()}\n"
                     f"since={datetime.now().isoformat(timespec='seconds')}\n"
                     .encode("utf-8"))
    except OSError:
        pass
    return fd


def release_lock(fd: Optional[int]) -> None:
    """释放锁（优雅退出与测试用；崩溃时由 OS 兜底）。"""
    if fd is not None and fd != LOCK_DISABLED:
        try:
            os.close(fd)
        except OSError:
            pass

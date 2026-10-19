# -*- coding: utf-8 -*-
"""进程级单实例文件锁（全机唯一）。

放在 src/core 是为了让「交易闭环」和「收盘后调度器」共用同一份实现，
同时避免调度器为了一把锁去 import src.trading.loop（那会连带拉起 xtquant，
而打包版已排除该依赖）。
"""

import os
from pathlib import Path
from datetime import datetime

from loguru import logger


class SingleInstanceLock:
    """「同一台机器上只允许一个闭环实例」的进程级文件锁。

    为什么必须有：闭环多开一份，就会对同一份交易计划**再下一遍单**，
    而且两边的风控/纸账户各有一份内存副本，对账也会乱。

    2026-10-19 实测踩到过：``data/loop_heartbeat.json`` 里的 pid 在
    ``20508``（常驻 ``scripts/run_loop.py``）和 ``21204``（看板进程内嵌闭环）
    之间来回跳 —— 两个实例在交替 tick。常驻脚本早在启动时就用 OS 级锁挡住了
    第二个进程，但**跑在看板进程里的内嵌闭环此前完全没有这道闸**，于是形成了
    「看板点一次启动 + 开机自启又拉一份」的双实例。本类把这道闸补到引擎内部，
    让两种启动方式共用同一把锁（``data/.run_loop.lock``）。

    用 OS 级文件锁而不是 PID 文件：进程正常退出或崩溃，操作系统都会释放锁，
    不会留下"假死锁"。

    ⚠️ 判据只能用 ``msvcrt.locking`` 的结果 —— 实测 Windows 上
    ``os.open(O_RDWR)`` 在别人持锁时**照样成功**（字节区间锁只拦对锁定区间
    的读写，不拦 open）；``open(path, "a+")`` 走 C 运行时 fopen 才会报
    PermissionError。两个 API 语义不同，混用会得出错误结论。
    ⚠️ 别用 ``os.kill(pid, 0)`` 探活，Windows 上那是真的 TerminateProcess。
    """

    def __init__(self, path):
        self.path = Path(path)
        self._fh = None

    @property
    def acquired(self) -> bool:
        return self._fh is not None

    def acquire(self) -> bool:
        """抢锁；抢到返回 True，被别人占用返回 False。"""
        if self._fh is not None:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 必须可写打开（POSIX 的 flock 同样要求可写句柄）
            fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o644)
        except PermissionError:
            logger.warning(f"⚠️  {self.path} 正被另一个实例锁定（写打开被拒）")
            return False
        except OSError as e:
            # 拿不到锁就无法证明唯一性 —— 宁可不跑，不可重复下单
            logger.warning(f"⚠️  无法创建/打开单实例锁 {self.path}: {e}")
            return False

        fh = os.fdopen(fd, "r+", encoding="utf-8")
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:               # pragma: no cover - POSIX
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False

        try:
            fh.seek(0)
            fh.truncate()
            fh.write(
                f"pid={os.getpid()}\n"
                f"since={datetime.now().isoformat(timespec='seconds')}\n"
            )
            fh.flush()
        except OSError:
            pass
        self._fh = fh
        return True

    def release(self) -> None:
        """主动释放（正常退出路径）；崩溃时由 OS 兜底。"""
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass

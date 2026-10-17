"""闭环单实例锁探针 —— 给批处理启动器用的。

为什么需要单独一个脚本：
    cmd.exe 的 ``ERRORLEVEL`` **不反映重定向失败**（实测向被占用文件
    ``>> echo.`` 之后 ``ERRORLEVEL`` 仍然是 0），所以
    ``run_loop_daemon.bat`` 无法用 ``if errorlevel 1`` 判断「实例是不是
    已经在跑」。唯一可靠的通道是**进程退出码**，于是把探测放进 Python。

⚠️ 这里必须复用 ``run_loop.acquire_single_instance`` 的**真实机制**：
    Windows 上 ``os.open(path, O_RDWR)`` 在别人持锁时**照样成功**（字节区间锁
    只拦读写，不拦 open）。真正挡住第二个实例的是 ``msvcrt.locking(LK_NBLCK)``
    抛 ``OSError``。曾经用「可写打开是否成功」来判定，结果永远判成「空闲」。
    注意 ``open(path, "a+")``（C 运行时 fopen）的共享模式不同，它才会报
    ``PermissionError`` —— 两个 API 语义不一样，别混用。

退出码：
    0  锁空闲（或锁文件根本建不出来，宁可能跑不可不跑）
    1  已有实例持锁
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "data" / ".run_loop.lock"


def probe(lock_path: Path = LOCK_PATH) -> int:
    """返回 0 = 可启动；1 = 已有实例。"""
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 连 data/ 都建不出来：这是主程序该炸的错，探针不背这个锅，交给它自己判断。
        return 0
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    except PermissionError:
        return 1
    except OSError:
        # 连锁文件都建不出来：交给主程序自己判断，不在这里拦死。
        return 0

    f = os.fdopen(fd, "r+", encoding="utf-8")
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 1
    finally:
        # 关闭句柄即释放锁，不会把锁留给启动器后面的真实实例。
        f.close()
    return 0


def main() -> int:
    return probe()


if __name__ == "__main__":
    sys.exit(main())

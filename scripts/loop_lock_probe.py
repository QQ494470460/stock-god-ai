"""闭环单实例锁探针 —— 给批处理启动器用的。

为什么需要单独一个脚本：
    cmd.exe 的 ``ERRORLEVEL`` **不反映重定向失败**（实测向被占用文件
    ``>> echo.`` 之后 ``ERRORLEVEL`` 仍然是 0），所以
    ``run_loop_daemon.bat`` 无法用 ``if errorlevel 1`` 判断「实例是不是
    已经在跑」。唯一可靠的通道是**进程退出码**，于是把探测放进 Python。

⚠️ 这里必须复用主程序的**真实机制**，所以直接调用
``src.core.singleton.probe_lock`` —— 与 ``run_loop.acquire_single_instance``
是同一个实现，两个进程不可能对「锁是否空闲」得出不同结论。

（曾经复制过一份，两处口径分叉：一处用「``os.open`` 是否成功」，
在 Windows 上别人持锁时**照样成功**，于是探针永远报「空闲」，
两个实例同时在跑却毫无察觉。）

用法（一个探针服务所有常驻进程，避免每个进程复制一份、口径再次分叉）：
    pythonw scripts\\loop_lock_probe.py                              # 交易闭环
    pythonw scripts\\loop_lock_probe.py --name .scheduler_daemon.lock  # 收盘后调度

退出码：
    0  锁空闲（或锁文件根本建不出来，宁可能跑不可不跑）
    1  已有实例持锁
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "data" / ".run_loop.lock"

# 本脚本以 ``pythonw scripts\loop_lock_probe.py`` 直接执行，sys.path[0] 是
# scripts/，拿不到 src 包。少了这段，下面的 import 会抛 ModuleNotFoundError。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def probe(lock_path: Path = LOCK_PATH) -> int:
    """返回 0 = 可启动；1 = 已有实例。"""
    try:
        from src.core.singleton import probe_lock
    except Exception:  # noqa: BLE001
        # 探针自己炸了也不能挡住闭环 —— fail-open，交给主程序的锁去判。
        return 0
    return 0 if probe_lock(lock_path) else 1


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="常驻进程单实例锁探针")
    ap.add_argument("--name", default="",
                    help="data/ 下的锁文件名；不给则用 .run_loop.lock")
    ns = ap.parse_args(argv)
    if ns.name:
        return probe(ROOT / "data" / ns.name)
    return probe()


if __name__ == "__main__":
    sys.exit(main())

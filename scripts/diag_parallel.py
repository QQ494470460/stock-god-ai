# -*- coding: utf-8 -*-
"""并行启动诊断：精确复现 launcher 的真实时序，验证不再出现 pandas 循环导入竞争。

背景（实测踩过的坑）：
    launcher 为了加快首屏，把「数据新鲜度扫描」放在服务线程启动之后并行执行。
    旧实现的扫描函数会 ``import pandas`` 并读 parquet，而服务线程此刻正在
    ``import streamlit``（内部也 import pandas）。两个线程同时初始化 pandas 会触发：

        partially initialized module 'pandas' has no attribute 'Series'
        (most likely due to a circular import)

    → 服务线程崩溃 → wait_for_server 干等满 180s → 用户看到"任务栏在运行但没界面"。

本脚本验证两件事：
    1. 扫描函数不再 import pandas（无竞争，服务能正常就绪）
    2. 万一服务线程真的崩了，wait_for_server 能快速失败而不是傻等

用法（源码模式）:   py -3.12 scripts/diag_parallel.py
用法（冻结模式）:   选股神AI.exe --run-script scripts/diag_parallel.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import launcher  # noqa: E402


def main() -> int:
    print("[diag] === 并行启动诊断 ===", flush=True)

    port = launcher.find_free_port()
    print(f"[diag] 端口 {port}", flush=True)

    # --- 阶段 1：与真实 launcher 完全一致的时序 ---
    t0 = time.time()
    box, thread = launcher.start_streamlit_thread(port)

    fresh = launcher.check_data_freshness()          # ← 曾经的崩溃点
    pandas_loaded = "pandas" in sys.modules
    print(f"[diag] 扫描结果: {fresh}", flush=True)
    print(f"[diag] 扫描耗时: {time.time() - t0:.2f}s", flush=True)
    print(f"[diag] 扫描后 pandas 是否已加载: {pandas_loaded}", flush=True)

    # --- 阶段 2：等待服务就绪（带快速失败） ---
    t1 = time.time()
    ok = launcher.wait_for_server(port, server_box=box, timeout=90)
    dt = time.time() - t1
    print(f"[diag] 服务就绪={ok}  等待耗时={dt:.1f}s", flush=True)

    if box.get("error"):
        print(f"[diag] ✗ 服务线程异常: {box['error']!r}", flush=True)

    launcher.stop_streamlit(box, thread)
    time.sleep(1)
    print(f"[diag] 停止后线程存活: {thread.is_alive()}", flush=True)

    if not ok:
        print("[diag] ✗ 失败：服务未能就绪（若等待耗时接近 90s 说明快速失效灵）", flush=True)
        return 1
    if dt > 60:
        print("[diag] ⚠️ 服务就绪过慢，疑似仍有阻塞", flush=True)
        return 1

    print("[diag] ✓ 通过：并行扫描不再干扰服务启动", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

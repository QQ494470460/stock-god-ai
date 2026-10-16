# -*- coding: utf-8 -*-
"""无窗口启动诊断：复用 launcher 的服务启动逻辑，只探测健康接口。

验证三件事：
  1. 端口探测不会误抢已被占用的端口
  2. 服务能在合理时间内就绪
  3. 停止后线程真的退出（老版本会留下孤儿进程占着端口）

用法: py -3.12 scripts/diag_launch.py
"""
import sys
import time
import socket
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import launcher  # noqa: E402


def main() -> int:
    # 1) 端口探测正确性：8501 上现在有孤儿服务在听，必须被判为"占用"
    print(f"[diag] 8501 被占用? {launcher.port_in_use(8501)}  (期望 True)", flush=True)
    port = launcher.find_free_port()
    print(f"[diag] 分配到的端口 {port}  (期望 != 8501)", flush=True)
    if port == 8501:
        print("[diag] ✗ 端口探测失效：仍选中了被占用的 8501", flush=True)
        return 1

    # 2) 启动耗时
    t0 = time.time()
    box, thread = launcher.start_streamlit_thread(port)
    ok = launcher.wait_for_server(port, timeout=120)
    dt = time.time() - t0
    print(f"[diag] 服务就绪={ok} 耗时={dt:.1f}s  (新起服务通常 >2s)", flush=True)
    print(f"[diag] 后台线程存活={thread.is_alive()}", flush=True)
    if box.get("error"):
        print(f"[diag] 线程内异常: {box['error']!r}", flush=True)

    if ok:
        with launcher._localhost_opener().open(
            f"http://127.0.0.1:{port}/_stcore/health", timeout=5
        ) as r:
            print(f"[diag] /_stcore/health -> {r.status} {r.read()[:40]}", flush=True)

    # 3) 停止是否干净
    print("[diag] 停止服务…", flush=True)
    launcher.stop_streamlit(box, thread)
    print(f"[diag] 停止后线程存活={thread.is_alive()}  (期望 False)", flush=True)

    # 端口是否真的释放
    time.sleep(1)
    released = not launcher.port_in_use(port)
    print(f"[diag] 端口 {port} 已释放={released}  (期望 True)", flush=True)

    return 0 if (ok and not thread.is_alive() and released) else 1


if __name__ == "__main__":
    sys.exit(main())

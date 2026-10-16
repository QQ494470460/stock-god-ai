# -*- coding: utf-8 -*-
"""冻结环境下的并行启动诊断（自包含，不 import launcher）。

复现 launcher 的真实时序：
    主线程起 Streamlit 服务线程  →  紧接着主线程做「数据新鲜度扫描」
    （旧实现在此处 import pandas，与服务线程的 import streamlit 抢锁 →
      pandas 循环导入 → 服务线程崩溃 → 用户看到"任务栏在跑但没界面"）

用法（冻结，推荐）:
    选股神AI.exe --run-script scripts/diag_parallel_frozen.py
用法（源码）:
    py -3.12 scripts/diag_parallel_frozen.py
"""
import os
import sys
import time
import socket
import threading
import asyncio
import datetime

P = lambda m: print(f"[diag] {m}", flush=True)


def find_free_port(start=8531, tries=40):
    for p in range(start, start + tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    P("=== 冻结环境并行启动诊断 ===")
    # 注意：exe --run-script 模式下 sys.path[0] 是 base_library.zip，
    # 不能用它推导项目根；必须从脚本自身位置反推（本脚本位于 _internal/scripts/）。
    _here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(_here) if os.path.basename(_here) == "scripts" else _here
    if root not in sys.path:
        sys.path.insert(0, root)
    base = os.path.dirname(root) if os.path.basename(root) == "_internal" else root
    os.environ.setdefault("STOCK_GOD_DATA_DIR", os.path.join(base, "data"))
    os.environ.setdefault("STOCK_GOD_MODEL_DIR", os.path.join(base, "models"))
    os.environ.setdefault("STOCK_GOD_LOG_DIR", os.path.join(base, "logs"))
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    app_path = os.path.join(root, "src", "ui", "app.py")
    if not os.path.exists(app_path):
        app_path = os.path.join(root, "_internal", "src", "ui", "app.py")
    P(f"app 入口存在: {os.path.exists(app_path)}  ({app_path})")
    if not os.path.exists(app_path):
        return 2

    port = find_free_port()
    from streamlit import config as st_config
    for opt, val in (("server.headless", True), ("server.fileWatcherType", "none"),
                     ("browser.gatherUsageStats", False),
                     ("global.developmentMode", False)):
        try:
            st_config.set_option(opt, val)
        except Exception as e:
            P(f"设置 {opt} 失败: {e}")
    st_config.set_option("server.port", port)
    st_config.set_option("server.address", "127.0.0.1")
    P(f"端口 {port}")

    # ---- 1) 起服务线程（与 launcher 一致）----
    box: dict = {}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        box["loop"] = loop

        async def _serve():
            from streamlit.web.server import Server
            srv = Server(main_script_path=app_path, is_hello=False)
            box["server"] = srv
            await srv.start()
            box["ready"] = True
            await asyncio.Event().wait()

        try:
            loop.run_until_complete(_serve())
        except BaseException as e:  # noqa: BLE001
            box["error"] = e
            box["finished"] = True
            P(f"✗ 服务线程异常: {type(e).__name__}: {e}")
        finally:
            box["finished"] = True

    t = threading.Thread(target=_run, name="st-serve", daemon=True)
    t0 = time.time()
    t.start()

    # ---- 2) 主线程：复刻 launcher.check_data_freshness（纯 os 扫描）----
    data_dir = os.environ["STOCK_GOD_DATA_DIR"]
    stocks, latest, stale = 0, "", None
    try:
        from pathlib import Path
        files = sorted(Path(data_dir).glob("daily_*_qfq.parquet"))
        stocks = len(files)
        if files:
            newest = datetime.datetime.fromtimestamp(
                max(f.stat().st_mtime for f in files)).date()
            latest = newest.strftime("%Y-%m-%d")
            stale = (datetime.date.today() - newest).days
    except Exception as e:  # noqa: BLE001
        P(f"扫描异常: {e}")
    P(f"扫描: {stocks} 只 / 最新 {latest or '—'} / 滞后 {stale} 天，耗时 {time.time()-t0:.2f}s")
    P(f"扫描期间 pandas 是否被加载: {'pandas' in sys.modules}")

    # ---- 3) 等健康（带快速失败）----
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    health = False
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=3) as r:
                if r.status == 200:
                    health = True
                    break
        except Exception:
            pass
        if box.get("finished") and not box.get("ready"):
            P("✗ 服务线程已结束且未就绪（快速失败生效）")
            break
        time.sleep(0.5)

    dt = time.time() - t0
    P(f"服务就绪={health} 总耗时={dt:.1f}s")
    if box.get("error"):
        P(f"服务端错误: {box['error']!r}")

    if health:
        try:
            with opener.open(f"http://127.0.0.1:{port}/", timeout=5) as r:
                html = r.read().decode("utf-8", "ignore")
            P(f"✓ 首页 HTTP 200，长度 {len(html)}")
        except Exception as e:
            P(f"首页抓取失败: {e}")

    try:
        srv, loop = box.get("server"), box.get("loop")
        if srv and loop:
            loop.call_soon_threadsafe(srv.stop)
    except Exception:
        pass

    ok = health and dt < 60
    P("✓ 通过：并行扫描未干扰服务启动" if ok else "✗ 失败：服务未能就绪")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

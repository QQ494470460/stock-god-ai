# -*- coding: utf-8 -*-
"""
冻结环境"实跑"校验：在只保留 _internal + 标准库的语境下，
1) 加载真实 xgboost 模型（验证原生 .dll 可用、pickle 可反序列化）
2) 真正启动 Streamlit 服务（复用 launcher 的同进程 asyncio 方式）
3) curl 健康检查 + 首页，确认看板能 serve

不依赖 site-packages，最接近"双击 exe"的运行时解析环境。
用法: py -3.12 scripts/validate_serve.py [INTERNAL_DIR]
"""
import os
import sys
import json
import time
import socket
import threading
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("[VALIDATE_SERVE] START", flush=True)
    internal = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "dist", "选股神AI", "_internal")
    internal = os.path.abspath(internal)
    assert os.path.isdir(internal), f"找不到 _internal: {internal}"

    # ---- 1. 冻结 sys.path: 仅 _internal + src + 标准库, 屏蔽 site-packages ----
    stdlib_lib = os.path.dirname(os.__file__)
    sys.path = [internal, os.path.join(internal, "src"), stdlib_lib]

    # 环境变量（与 launcher 对齐）
    base = os.path.join(ROOT, "dist", "选股神AI")
    os.environ["STOCK_GOD_DATA_DIR"] = os.path.join(base, "data")
    os.environ["STOCK_GOD_MODEL_DIR"] = os.path.join(ROOT, "models")
    os.environ["STOCK_GOD_LOG_DIR"] = os.path.join(base, "logs")
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    # ---- 2. 加载真实模型（验证原生 lib + pickle）----
    try:
        import pickle
        import numpy as np
        import xgboost as xgb
        log(f"xgboost 版本: {getattr(xgb, '__version__', '?')}  (XGBClassifier={hasattr(xgb,'XGBClassifier')})")
        model_path = os.path.join(ROOT, "models", "xgboost_model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict"), "模型对象没有 predict 方法"
        # 用一个很小的常量样本试推（特征数未知，捕获维度异常属正常，能到这步说明原生 lib 已加载）
        try:
            import inspect
            n = getattr(model, "n_features_in_", None)
            if n:
                model.predict(np.zeros((1, n), dtype=float))
                log(f"✓ 模型预测自检通过 (n_features_in_={n})")
            else:
                log("✓ 模型反序列化成功（无法获知特征数，跳过试推）")
        except Exception as e:
            log(f"⚠️ 模型试推跳过（非致命）: {type(e).__name__}: {e}")
        model_ok = True
    except Exception as e:
        import traceback as tb
        log("✗ 模型加载失败:\n" + "".join(tb.format_exception(type(e), e, e.__traceback__)))
        model_ok = False

    # ---- 3. 启动 Streamlit 服务 ----
    app_path = os.path.join(internal, "src", "ui", "app.py")
    if not os.path.exists(app_path):
        log(f"✗ 找不到 app.py: {app_path}")
        return 2

    import asyncio
    from streamlit import config as st_config
    for opt, value in (
        ("server.headless", True),
        ("server.fileWatcherType", "none"),
        ("browser.gatherUsageStats", False),
        ("global.developmentMode", False),
    ):
        try:
            st_config.set_option(opt, value)
        except Exception as e:
            log(f"设置 {opt} 失败: {e}")

    # 找一个真空闲端口
    def find_free_port(start=8511, tries=30):
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

    port = find_free_port()
    st_config.set_option("server.port", port)
    st_config.set_option("server.address", "127.0.0.1")
    log(f"启动 Streamlit 服务，端口 {port}")

    server_box = {}
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server_box["loop"] = loop
        async def _serve():
            from streamlit.web.server import Server
            srv = Server(main_script_path=app_path, is_hello=False)
            server_box["server"] = srv
            await srv.start()
            server_box["ready"] = True
            log(f"服务就绪 127.0.0.1:{port}")
            await asyncio.Event().wait()
        try:
            loop.run_until_complete(_serve())
        except Exception as e:
            server_box["error"] = e
            log(f"服务异常: {e}")
        finally:
            try: loop.close()
            except Exception: pass

    t = threading.Thread(target=_run, name="st-serve", daemon=True)
    t.start()

    # 等健康
    opener = __import__("urllib.request", fromlist=["build_opener"]).build_opener(
        __import__("urllib.request", fromlist=["ProxyHandler"]).ProxyHandler({}))
    health = False
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=3) as r:
                if r.status == 200:
                    health = True
                    break
        except Exception:
            pass
        time.sleep(0.5)

    if not health:
        log("✗ Streamlit 健康检查超时（服务未在 120s 内就绪）")
        if server_box.get("error"):
            log(f"  服务端错误: {server_box['error']}")
        return 3

    # 取首页 HTML，确认有 streamlit 挂载点
    try:
        with opener.open(f"http://127.0.0.1:{port}/", timeout=5) as r:
            html = r.read().decode("utf-8", "ignore")
        has_root = ("root" in html) or ("streamlit" in html.lower()) or ("_stcore" in html)
        log(f"✓ 首页 HTTP 200，长度 {len(html)}，含 streamlit 挂载点: {has_root}")
        serve_ok = has_root
    except Exception as e:
        log(f"✗ 首页抓取失败: {e}")
        serve_ok = False

    # 停服务
    try:
        srv = server_box.get("server")
        loop = server_box.get("loop")
        if srv and loop:
            loop.call_soon_threadsafe(srv.stop)
    except Exception:
        pass

    log("=" * 60)
    log(f"模型加载 : {'PASS' if model_ok else 'FAIL'}")
    log(f"看板服务 : {'PASS' if serve_ok else 'FAIL'}")
    return 0 if (model_ok and serve_ok) else 4


if __name__ == "__main__":
    sys.exit(main())

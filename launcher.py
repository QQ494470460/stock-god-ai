# -*- coding: utf-8 -*-
"""
选股神 AI —— 桌面应用启动器

双击运行本文件（或打包后的 exe）会：
  1. 检查是否已有实例在跑（有就直接开窗，不重复起服务）
  2. 在后台线程启动 Streamlit 服务
  3. 等待服务就绪
  4. 弹出内嵌窗口显示看板
  5. 关闭窗口时彻底结束服务并退出，不留孤儿进程

设计要点：
- 兼容 PyInstaller 打包（sys.frozen）与源码直接运行两种模式
- 端口探测用「真实连接」而非 bind：Windows 下 SO_REUSEADDR 允许抢占
  已被监听的端口，老版本因此会误判 8501 空闲、然后连到别人的服务上
- 停服务走 Streamlit 官方 Server.stop()，不再用 loop.stop() 硬掐
  （硬掐会抛 "Event loop stopped before Future completed" 且线程不退，
  进程就永远挂着、端口一直被占）
- 大数据目录（data/）不打包进 exe，作为外挂目录放在 exe 同级
"""

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Windows 打包环境编码兜底（与 src/ui/app.py 中的同名函数保持一致）
# ---------------------------------------------------------------------------
# 冻结运行时 sys.stdout/stderr 默认 GBK(cp936)，日志/页面文案里的 emoji
# （🤖 🔍 📊 🐞 …）一写出就抛 UnicodeEncodeError，启动器直接崩。
# 就地 reconfigure 成 UTF-8 + errors="replace"，保证任何字符都不会再炸。
def _force_utf8_stdio() -> None:
    import io as _io
    for _name in ("stdout", "stderr"):
        _stream = getattr(sys, _name, None)
        if _stream is None:
            continue
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except Exception:
            pass
        try:
            _buf = getattr(_stream, "buffer", None)
            if _buf is None:
                continue
            setattr(sys, _name, _io.TextIOWrapper(
                _buf, encoding="utf-8", errors="replace", line_buffering=True))
        except Exception:
            pass


_force_utf8_stdio()
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

APP_NAME = "选股神 AI"
DEFAULT_PORT = 8501
STARTUP_TIMEOUT = 180  # 秒，模型加载较慢
LOCK_FILE = "launch.lock"


# 进程启动时刻（模块导入完成即记录），用于在各阶段日志上打出"距启动 X.XXs"，
# 这样双击到出界面的耗时分布一眼可见，不用猜。
_BOOT_T0 = time.perf_counter()


def _elapsed() -> float:
    """距进程启动的秒数"""
    return time.perf_counter() - _BOOT_T0


def log(msg: str) -> None:
    """同时输出到控制台和日志文件"""
    line = f"[{time.strftime('%H:%M:%S')} +{_elapsed():.2f}s] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        log_dir = get_base_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "launcher.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_base_dir() -> Path:
    """应用根目录：打包后是 exe 所在目录，源码运行时是项目根"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


# ==================== 端口探测 ====================

def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """端口上是否真有服务在听。

    用 connect 而不是 bind：Windows 的 SO_REUSEADDR 让 bind 能抢占
    已被监听的端口，bind 成功 ≠ 端口空闲。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.35)
        return s.connect_ex((host, port)) == 0


def find_free_port(start: int = DEFAULT_PORT, tries: int = 25) -> int:
    """找一个真正没人用的端口：先探连接，再实绑一次确认"""
    for port in range(start, start + tries):
        if port_in_use(port):
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # 不设 SO_REUSEADDR，bind 成功才代表端口真的可用
                s.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    # 全被占则让系统随机分配
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _localhost_opener():
    """访问本机专用的 opener —— 强制绕过系统代理。

    环境里可能设了 HTTP_PROXY（本例是 127.0.0.1:59719），urllib 会老老实实
    把 127.0.0.1:8502 的请求也发给代理，代理返回 502，健康探测就永远失败，
    表现成"双击没反应"。探测本机必须显式禁用代理。
    """
    import urllib.request

    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def health_ok(port: int) -> bool:
    """目标端口是否在提供 Streamlit 健康响应"""
    try:
        with _localhost_opener().open(
            f"http://127.0.0.1:{port}/_stcore/health", timeout=2
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


# ==================== 单实例保护 ====================

def _lock_path() -> Path:
    d = get_base_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / LOCK_FILE


def read_alive_instance() -> int | None:
    """读锁文件，若记录的实例仍在提供服务则返回其端口"""
    p = _lock_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        port = int(data.get("port", 0))
        pid = int(data.get("pid", 0))
    except Exception:
        return None
    if not port:
        return None
    # 进程还在 + 端口还活着，才算真的有实例
    if pid and pid != os.getpid() and not _pid_alive(pid):
        return None
    return port if health_ok(port) else None


def write_instance_lock(port: int) -> None:
    try:
        _lock_path().write_text(
            json.dumps({"port": port, "pid": os.getpid(),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")}),
            encoding="utf-8",
        )
    except Exception:
        pass


def clear_instance_lock() -> None:
    try:
        _lock_path().unlink(missing_ok=True)
    except Exception:
        pass


def _pid_alive(pid: int) -> bool:
    """Windows / POSIX 通用的进程存活判断"""
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(code)
            )
            # 259 = STILL_ACTIVE
            return bool(ok) and code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ==================== 数据新鲜度 ====================

def check_data_freshness() -> dict:
    """抽样扫描本地行情缓存，判断数据是否陈旧（只读、不做网络请求）"""
    result = {"stocks": 0, "latest": "", "stale_days": None, "ok": True}
    try:
        data_dir = get_base_dir() / "data"
        if not data_dir.exists():
            result["ok"] = False
            return result
        files = sorted(data_dir.glob("daily_*_qfq.parquet"))
        result["stocks"] = len(files)
        if not files:
            result["ok"] = False
            return result

        # 关键：这里**绝不能 import pandas**。
        # 本函数与服务线程并行执行，服务线程此刻正在 import streamlit
        # （会 import pandas）。两个线程同时初始化 pandas 会触发
        # "partially initialized module 'pandas' has no attribute 'Series'"
        # 的循环导入竞争 → 服务线程崩溃 → 窗口永远等不到服务（实测踩过）。
        # 因此改用文件修改时间判断新鲜度：纯 os 调用，零重库、毫秒级。
        import datetime as _dt

        newest_mtime = max(f.stat().st_mtime for f in files)
        newest = _dt.datetime.fromtimestamp(newest_mtime).date()
        result["latest"] = newest.strftime("%Y-%m-%d")
        # 时钟异常 / dist 目录跨机器拷贝时，文件 mtime 可能落在未来，
        # 直接相减会得到负数（实测 -3、-400），下游会显示"滞后 -3 天"。
        # 这里夹到 0：未来时间戳按"最新、不陈旧"处理。
        result["stale_days"] = max(0, (_dt.date.today() - newest).days)
    except Exception as e:  # noqa: BLE001
        log(f"数据新鲜度检查失败: {e}")
        result["ok"] = False
    return result


# ==================== Streamlit 服务 ====================

def wait_for_server(port: int, timeout: int = STARTUP_TIMEOUT,
                   server_box: dict | None = None) -> bool:
    """轮询等待 Streamlit 服务就绪

    以 ``/_stcore/health`` 返回 200 为准 —— 只探测 TCP 端口不够，
    端口监听 ≠ 服务可用（Uvicorn 起来到能响应请求还有一小段窗口）。

    传入 ``server_box`` 时开启**快速失败**：服务线程若已结束且未就绪，
    立刻返回 False，不再干等满 timeout（否则用户会看到"卡住无界面"）。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if health_ok(port):
            return True
        if server_box is not None and server_box.get("finished") \
                and not server_box.get("ready"):
            return False
        time.sleep(0.5)
    return False


def find_app_entry() -> Path:
    """定位看板入口，兼容 PyInstaller onedir 两种布局

    PyInstaller 会把 add-data 的内容放到 ``_internal/`` 下，
    所以源码既可能在 ``<base>/src`` 也可能在 ``<base>/_internal/src``。
    """
    base = get_base_dir()
    candidates = [
        base / "src" / "ui" / "app.py",
        base / "_internal" / "src" / "ui" / "app.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def resolve_workdir() -> Path:
    """确定工作目录（cwd）：要能同时看到 src/ 和 data/"""
    base = get_base_dir()
    if (base / "src" / "ui" / "app.py").exists():
        return base
    internal = base / "_internal"
    if (internal / "src" / "ui" / "app.py").exists():
        return internal
    return base


def start_streamlit_thread(port: int):
    """在**当前进程的后台线程**里运行 Streamlit

    打包成 exe 后 sys.executable 指向 exe 自己，无法再 fork 一个 python
    子进程跑 streamlit（没有独立解释器）。所以必须同进程加载：
    在后台线程里调用 streamlit 的 Server API。

    返回 ``(server_box, thread)``。
    """
    app_path = find_app_entry()
    if not app_path.exists():
        raise FileNotFoundError(
            f"找不到看板入口文件:\n  {app_path}\n\n"
            "请确认应用目录完整（应包含 _internal/ 或 src/）。"
        )

    # 环境变量必须在 streamlit 加载前设好
    base = get_base_dir()
    os.environ["STREAMLIT_GLOBAL_DEVELOPER_MODE"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STOCK_GOD_DATA_DIR"] = str(base / "data")
    os.environ["STOCK_GOD_MODEL_DIR"] = str(base / "models")
    os.environ["STOCK_GOD_LOG_DIR"] = str(base / "logs")
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
    os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

    # 保证项目包可 import（打包后源码在 _internal 下）
    workdir = resolve_workdir()
    if str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))

    server_box: dict = {}

    def _run():
        """在后台线程跑 asyncio 事件循环，驱动 Streamlit 服务器"""
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server_box["loop"] = loop
        stop_event = asyncio.Event()
        server_box["stop_event"] = stop_event

        async def _serve():
            # 端口/地址必须用 config API 显式设置。
            # 实测：仅设 STREAMLIT_SERVER_PORT 环境变量不生效 —— Streamlit 1.5x
            # 会走默认 8501，导致健康探测永远等不到（表现为"启动卡住 2 分钟"）。
            from streamlit import config as st_config

            for opt, value in (
                ("server.port", port),
                ("server.address", "127.0.0.1"),
                ("server.headless", True),
                ("server.fileWatcherType", "none"),
                ("browser.gatherUsageStats", False),
                ("global.developmentMode", False),
            ):
                try:
                    st_config.set_option(opt, value)
                except Exception as e:  # noqa: BLE001
                    log(f"设置 {opt}={value} 失败: {e}")

            from streamlit.web.server import Server

            srv = Server(main_script_path=str(app_path), is_hello=False)
            server_box["server"] = srv
            await srv.start()
            server_box["ready"] = True
            log(f"服务已就绪，监听 127.0.0.1:{port}")
            # 等停止信号，而不是靠 loop.stop() 硬掐
            await stop_event.wait()
            try:
                srv.stop()
            except Exception as e:  # noqa: BLE001
                log(f"停止服务时出错: {e}")
            # 给 uvicorn 一点收尾时间，再取消残留任务。
            # 不取消的话 loop.close() 会硬杀 pending 任务，抛
            # "Task was destroyed but it is pending" 且 socket 延迟释放。
            await asyncio.sleep(0.5)
            pending = [
                t for t in asyncio.all_tasks() if t is not asyncio.current_task()
            ]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            log("服务已停止")

        try:
            loop.run_until_complete(_serve())
        except Exception as e:  # noqa: BLE001
            server_box["error"] = e
            log(f"Streamlit 运行失败: {e}")
        finally:
            try:
                loop.close()
            except Exception:
                pass
            server_box["finished"] = True

    t = threading.Thread(target=_run, name="streamlit-server", daemon=True)
    t.start()
    return server_box, t


def stop_streamlit(server_box: dict, thread: threading.Thread | None = None,
                   timeout: float = 10.0) -> None:
    """干净地停掉同进程的 Streamlit 服务并等待线程退出"""
    server_box["stopping"] = True
    srv = server_box.get("server")
    loop = server_box.get("loop")
    if srv is None or loop is None:
        return
    log("关闭服务…")

    def _shutdown():
        """在事件循环线程内执行：先置停止事件，再调官方 stop()"""
        try:
            ev = server_box.get("stop_event")
            if ev is not None:
                ev.set()
        except Exception as e:  # noqa: BLE001
            log(f"置停止事件失败: {e}")

    try:
        loop.call_soon_threadsafe(_shutdown)
    except Exception as e:  # noqa: BLE001
        log(f"通知停止失败: {e}")
        return

    if thread is not None:
        thread.join(timeout=timeout)
        log(f"服务线程已退出: {not thread.is_alive()}")
    else:
        deadline = time.time() + timeout
        while time.time() < deadline and not server_box.get("finished"):
            time.sleep(0.1)


# ==================== 窗口 ====================

def build_loading_html(url: str, fresh: dict) -> str:
    """启动画面：轮询探测服务，就绪后自动跳转"""
    tip = ""
    if fresh.get("latest"):
        tip = f"本地数据最新 {fresh['latest']}（{fresh['stocks']} 只）"
        if fresh.get("stale_days") and fresh["stale_days"] > 5:
            tip += f" · 已滞后 {fresh['stale_days']} 天，建议进应用后更新"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0.5;url={url}">
<style>
  body{{margin:0;display:flex;align-items:center;justify-content:center;
       height:100vh;font-family:"Microsoft YaHei",sans-serif;
       background:linear-gradient(135deg,#5b21b6,#7c3aed);color:#fff}}
  .box{{text-align:center}}
  .spin{{width:44px;height:44px;margin:0 auto 18px;border:3px solid rgba(255,255,255,.3);
       border-top-color:#fff;border-radius:50%;animation:spin .9s linear infinite}}
  @keyframes spin{{to{{transform:rotate(360deg)}}}}
  h2{{margin:0 0 8px;font-size:17px;font-weight:600}}
  p{{margin:0;font-size:13px;opacity:.8}}
  .tip{{margin-top:14px;font-size:12px;opacity:.65}}
</style></head>
<!-- 用浏览器原生 meta refresh 跳转，比 JS fetch().then(location.href) 可靠：
     在 pywebview 注入的 about:blank 上下文里，跨域 location.href 常被 WebView2
     导航策略拦掉，导致永远卡在本加载页。meta refresh 是原生导航，必定放行。
     Streamlit 未就绪时会挂起本请求，刷新期间自动重试直到返回主页。 -->
<body><div class="box">
  <div class="spin"></div>
  <h2>{APP_NAME}</h2>
  <p>正在启动服务…</p>
  <div class="tip">{tip}</div>
</div></body></html>"""


def open_window(url: str, fresh: dict) -> None:
    """优先用内嵌窗口，失败则退回系统浏览器"""
    loading_html = build_loading_html(url, fresh)
    try:
        _t = time.perf_counter()
        import webview
        log(f"  [计时] import webview      {time.perf_counter() - _t:.2f}s")

        _t = time.perf_counter()
        window = webview.create_window(
            APP_NAME, url,
            width=1560, height=980,
            min_size=(1100, 700),
            text_select=True,
            on_top=True,
        )
        log(f"  [计时] create_window       {time.perf_counter() - _t:.2f}s")

        # 把启动画面盖在 Streamlit 主页之上，meta refresh 会自动跳回真实页面
        _t = time.perf_counter()
        try:
            window.load_html(loading_html)
        except Exception:
            pass
        log(f"  [计时] load_html(加载页)   {time.perf_counter() - _t:.2f}s")

        # webview.start() 会阻塞到窗口关闭；它返回前的耗时就是 WebView2
        # 运行时初始化 + 首屏导航的时间（这段一直是黑盒，现在记下来）
        _t = time.perf_counter()
        webview.start()
        log(f"  [计时] webview.start       {time.perf_counter() - _t:.2f}s")
        return
    except Exception as e:  # noqa: BLE001
        log(f"内嵌窗口不可用（{e}），改用系统浏览器打开")

    import webbrowser

    webbrowser.open(url)


# ==================== 主流程 ====================

def _show_error(msg: str) -> None:
    """弹窗显示错误（打包后没有控制台，弹窗是唯一反馈渠道）"""
    log(f"错误弹窗: {msg}")
    # 同时把错误详情落盘，方便用户直接把文件发回来定位
    try:
        err_path = get_base_dir() / "logs" / "launcher.err.log"
        with open(err_path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg + "\n\n")
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, msg, APP_NAME, 0x10)
    except Exception:
        pass


def _run_script_mode(argv: list) -> int:
    """脚本模式：不开窗、不起 Streamlit 服务，直接在进程内跑一个项目脚本。

    为什么需要它：
    打包后 ``sys.executable`` 就是本 exe（也就是启动器本身）。看板里的后台
    任务（模型训练 / 离线重训）用 ``[sys.executable, "scripts/train_xxx.py"]``
    起子进程 —— 结果启动器被当成主程序又跑了一遍：检查单实例、起服务、
    弹 pywebview 窗口。用户看到的「一点训练就弹出新窗口」就是这个原因。
    带上 ``--run-script`` 后，子进程直接走这里执行脚本，全程无窗口。
    """
    import runpy

    if not argv:
        print("用法: 选股神AI.exe --run-script <脚本.py> [参数...]",
              file=sys.stderr)
        return 2

    script = argv[0]
    if not os.path.isabs(script):
        script = str(get_base_dir() / script)
    if not os.path.isfile(script):
        # 打包后 add-data 的内容（含 scripts/）实际落在 _internal/ 下，
        # 相对路径按 exe 目录拼会找不到 —— 这里回退一次再试。
        # （看板 UI 传的是绝对路径，本就不会走到这；手工调用时才会用到）
        _alt = str(get_base_dir() / "_internal" / argv[0])
        if os.path.isfile(_alt):
            script = _alt
        else:
            print(f"找不到脚本: {script}", file=sys.stderr)
            return 2

    # 脚本里的 argparse 读的是 sys.argv[1:]，这里把它伪装成「脚本自己当 argv[0]」
    sys.argv = [script] + [str(a) for a in argv[1:]]
    # 项目根（打包后即 _internal）进 sys.path，保证 import src.* 正常
    _root = str(Path(script).parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    print(f"[{time.strftime('%H:%M:%S')}] 脚本模式: {script}", file=sys.stderr, flush=True)
    try:
        runpy.run_path(script, run_name="__main__")
        return 0
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0
    except BaseException:  # noqa: BLE001
        import traceback as _tb
        _tb.print_exc()
        return 1


def main() -> int:
    base = get_base_dir()
    log(f"{APP_NAME} 启动中… 目录: {base}")

    # 已有实例在跑：直接复用它的端口，不重复起服务
    alive_port = read_alive_instance()
    if alive_port:
        log(f"检测到已在运行的实例（端口 {alive_port}），直接开窗")
        open_window(f"http://127.0.0.1:{alive_port}", check_data_freshness())
        return 0

    # 打包后检查关键目录
    data_dir = base / "data"
    model_dir = base / "models"
    if not data_dir.exists():
        log(f"提示: 未找到 data 目录 ({data_dir})，首次使用会自动下载行情数据")
    if not model_dir.exists():
        log(f"警告: 未找到 models 目录 ({model_dir})，AI 选股功能将不可用")

    port = find_free_port()
    log(f"选用端口 {port}")
    server_box: dict = {}
    thread: threading.Thread | None = None
    try:
        server_box, thread = start_streamlit_thread(port)
    except Exception as e:  # noqa: BLE001
        log(f"服务启动失败: {e}")
        _show_error(f"{APP_NAME} 启动失败：\n\n{e}")
        return 1

    # 数据新鲜度扫描已改为纯 os 调用（不 import pandas），
    # 与服务启动并行完全安全；这里再补一层保险：扫描放在线程里跑，
    # 即便将来有人改回重库实现，也不会阻塞首屏。
    fresh_box: dict = {}
    _ft = threading.Thread(
        target=lambda: fresh_box.update(check_data_freshness()),
        name="data-freshness", daemon=True,
    )
    _ft.start()
    _ft.join(timeout=5)
    fresh = fresh_box or {}
    if fresh.get("latest"):
        log(f"本地数据: {fresh['stocks']} 只，最新 {fresh['latest']}"
            + (f"，滞后 {fresh['stale_days']} 天" if fresh.get("stale_days") else ""))

    url = f"http://127.0.0.1:{port}"
    log("等待服务就绪，首次启动需加载模型，约需 10–60 秒…")
    ok = wait_for_server(port, server_box=server_box)
    if not ok:
        err = server_box.get("error")
        if err:
            log(f"服务异常: {err}")
            _show_error(f"{APP_NAME} 服务启动失败：\n\n{err}")
            stop_streamlit(server_box, thread)
            return 1
        log("服务启动超时，仍尝试打开窗口")
    else:
        log(f"服务就绪: {url}")
        write_instance_lock(port)

    # 不等固定 0.8s 再开窗 —— 那只是把"浏览器开始请求首页"往后推，
    # 而首页脚本执行本来就在窗口创建后与加载页并行进行，白等纯属浪费。
    # 加载页 meta refresh 已经负责掩盖首屏渲染，无需额外 sleep。
    try:
        open_window(url, fresh)
    finally:
        # 窗口关闭 → 结束服务，避免残留后台进程占端口
        stop_streamlit(server_box, thread)
        clear_instance_lock()
        log("已退出")
    return 0


if __name__ == "__main__":
    # Windows 冻结程序必须调用：spawn 出来的子进程会重新执行本 exe，
    # freeze_support() 负责接住 --multiprocessing-fork 参数并转入 spawn 逻辑，
    # 否则每个子进程都会把启动器当主程序跑一遍 → 疯狂弹窗。
    try:
        import multiprocessing

        multiprocessing.freeze_support()
    except Exception:  # noqa: BLE001
        pass

    # 后台任务模式（模型训练等）：不进 GUI，直接跑脚本
    if len(sys.argv) > 1 and sys.argv[1] == "--run-script":
        _code = _run_script_mode(sys.argv[2:])
        try:
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(_code)

    try:
        code = main()
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb

        tb_text = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
        log(f"未捕获异常:\n{tb_text}")
        _show_error(f"{APP_NAME} 运行出错：\n\n{exc}\n\n（完整堆栈见 logs/launcher.log）")
        code = 1
    # 强制退出：Streamlit/pywebview 可能留下非守护线程，
    # 正常 return 会让进程挂住、端口一直被占（老版本的孤儿进程就是这样来的）
    sys.stdout.flush()
    os._exit(code)

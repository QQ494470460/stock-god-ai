# -*- coding: utf-8 -*-
"""
PyInstaller 打包脚本 —— 选股神AI 桌面应用

架构说明（重要）：
  代码（Python 依赖 + 项目源码）→ 打进exe
  数据（data/ 615MB 行情、models/ 20MB 模型）→ **不进 exe**，作为外挂目录

为什么不把数据打进 exe：
  PyInstaller 单文件模式每次启动都要把内嵌数据解压到临时目录，
  615MB 会导致启动长达数分钟甚至超时，且每次运行都重复解压。
  外挂后启动只解压代码（约 200MB），数据直接读取，快很多。

用法：
  python build_exe.py            # 正常打包（不含数据拷贝）
  python build_exe.py --with-data # 打包并复制数据到 dist（耗时久）
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
# PyInstaller 的 COLLECT 步骤会先 rmtree 掉 distpath 下的同名目录。
# 若 distpath 直接指向 dist/，它会把外挂的 606MB data/ 一起删掉（8335 个文件），
# 触发批量删除守卫而中断。所以先构建到这个不含数据的暂存目录，再增量同步过去。
#
# 暂存目录必须**每次换新**：复用同一个路径时，COLLECT 会先删掉上次的产物
# （5000+ 文件），同样触发批量删除守卫导致打包直接失败。
# 代价是磁盘上会留历史副本，确认新版没问题后可手动清理 build/dist_stage_*。
STAGE = BUILD / f"dist_stage_{time.strftime('%Y%m%d_%H%M%S')}"
APP_NAME = "选股神AI"

# 预测看板运行时真正 import 到的第三方包（实测自 src/ui/app.py 导入闭包）。
# 打不全会 ModuleNotFoundError 导致黑屏（imblearn / matplotlib / blinker /
# six / typing_extensions / threadpoolctl / jsonpath / py 都踩过）。
#
# 关键背景（曾因此反复黑屏）：
#   本项目的打包产物里**没有 .pyz 归档** —— 所有模块必须以「散文件」形式存在于
#   _internal/ 才能被 import。因此三类模块要用不同方式收集：
#   · 包（目录，如 numpy / pandas / streamlit）→ --collect-all 整包散拷
#   · 单文件模块（typing_extensions.py / six.py / threadpoolctl.py / jsonpath.py /
#     py.py）→ --collect-all 会报 "not a package" 直接跳过，必须显式 --add-data
#     把该 .py 散拷进 _internal，否则运行时 ModuleNotFoundError
#   · 顶层编译扩展（个别 mypyc 产物，如 81d243…__mypyc.cp312-win_amd64.pyd）→
#     同样 --collect-all 收不到，也走 --add-data 散拷
#   · 命名空间包（google）→ --hidden-import；其下的 protobuf 走 --collect-all protobuf
#
# 为避免「手写清单漏掉某个单文件模块」再次黑屏，不再纯靠猜测：打包时实跑一遍
# 导入闭包（__import__ 钩子记录所有第三方顶层模块），自动分类收包 / 散拷，
# 下面的显式清单仅作兜底（动态分析失败时回退）。
COLLECT_ALL = [
    # 核心 ML / 数据
    "numpy", "scipy", "pandas", "sklearn", "joblib", "cloudpickle",
    # 绘图 / 前端
    "streamlit", "plotly", "altair", "pydeck", "matplotlib", "PIL", "narwhals",
    # 数据源 akshare 及其依赖树（含请求/解析库）
    "akshare", "requests", "curl_cffi", "bs4", "lxml", "openpyxl", "html5lib",
    "pyarrow", "kaleido", "anyio", "blinker", "starlette", "python_multipart",
    "certifi", "urllib3", "idna", "charset_normalizer", "click", "cffi",
    "cycler", "dateutil", "kiwisolver", "pyparsing", "pycparser",
    "soupsieve", "webencodings", "win32_setctime",
    "colorama", "packaging", "tqdm", "toml",
    "simplejson", "py_mini_racer", "cachetools", "pytz",
    # 显式补充：保障命名空间包下的子模块（注意包名是 google.protobuf，
    # 不是 protobuf —— find_spec("protobuf") 会返回 None 导致漏收）
    "google.protobuf",
    # Streamlit 1.5x 的 ASGI 服务依赖（--collect-all streamlit 只收自身、
    # 不会收传递依赖，缺了会导致「服务起不来 → 健康检查超时 → 黑屏」）。
    # 这些只在 Server.start() 运行时才 import，import 闭包分析抓不到，必须显式收。
    "uvicorn", "h11",
    "itsdangerous", "rich", "tenacity", "orjson", "jinja2",
    "python-dotenv", "watchdog", "exceptiongroup",
    # 注：httptools 故意不收（见下方 --exclude-module httptools 注释）。
]
# 自带 testing 子模块会 importorskip 缺失包（如 hypothesis），collect-all 必崩，
# 改用 --collect-data 只收资源，编译扩展与 .py 由 PyInstaller 正常分析收集。
#
# 注：optuna / shap 必须在此（而非只放 COLLECT_ALL）。scripts/train_optuna.py
# 在模块顶层就 `import optuna` / `import shap`，但 build_exe.py 是把 scripts/
# 整体 --add-data 当「数据」打进包的（不做 import 分析），所以这两个包不会被
# 静态分析发现；漏掉它们会导致 exe 里点 Optuna 训练立即 ModuleNotFoundError。
COLLECT_DATA = ["xgboost", "imblearn", "optuna", "shap"]
# 命名空间包 / 子模块显式隐藏导入兜底。
HIDDEN_PKGS = [
    "google",
    "loguru", "yaml", "pywebview",
    "plotly.graph_objs", "plotly.subplots",
    "imblearn.over_sampling", "imblearn.under_sampling",
    "optuna", "shap", "shap.explainers",
    "streamlit.runtime.scriptrunner.magic_funcs", "streamlit.web.cli",
]
# 不参与收包的重 / 测试依赖（节省体积 + 避免 testing 子模块崩溃）
EXCLUDE_PKGS = {
    "torch", "tensorflow", "notebook", "IPython",
    "pytest", "hypothesis", "dask", "langchain",
    # xtquant：券商 miniQMT 的原生交易库（34MB wheel，自带 .pyd/DLL）。
    # 收进 PyInstaller 包会在 find_binary_dependencies 的隔离子进程里触发
    # 访问冲突（0xC0000005）导致打包失败；且它本就依赖本机 QMT 客户端，
    # 打包进 exe 意义不大。改为「外挂加载」：
    #   把 xtquant 文件夹放到 exe 同级 libs/ 下即可启用实盘
    #   （由 config/settings.py 的 _extend_sys_path_for_external_libs 注入 sys.path）
    "xtquant",
}


def discover_closure():
    """实跑 app 入口 + 重库，用 __import__ 钩子记录所有第三方顶层模块。

    返回 (packages:set, single_files:set)：
      single_files 既含单文件 .py 模块，也含顶层 .pyd/.so 扩展（它们都无法被
      --collect-all 收，必须 --add-data 散拷）。
    失败时回退到空集合，由上面的显式清单兜底。
    """
    import builtins
    import importlib.util as _ilu

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    seen = set()
    real_import = builtins.__import__

    def _hook(name, *a, **k):
        seen.add(name.split(".")[0])
        return real_import(name, *a, **k)

    builtins.__import__ = _hook
    seeds = [
        "src.ui.app",
        "streamlit", "pandas", "numpy", "scipy", "sklearn", "xgboost",
        "imblearn", "optuna", "shap",
        "plotly", "akshare", "pywebview",
        "matplotlib", "altair", "pyarrow", "openpyxl", "requests", "curl_cffi",
    ]
    for s in seeds:
        try:
            real_import(s)
        except Exception:
            pass
    builtins.__import__ = real_import

    pkgs, singles = set(), set()
    for top in seen:
        try:
            spec = _ilu.find_spec(top)
        except Exception:
            continue
        if spec is None:
            continue
        origin = spec.origin or ""
        if "site-packages" not in origin:
            continue
        loc = getattr(spec, "submodule_search_locations", None)
        if loc:
            pkgs.add(top)
        elif origin.endswith((".py", ".pyd", ".so")):
            singles.add(top)
    return pkgs, singles


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_icon() -> Path | None:
    """生成应用图标（无外部素材时用内置默认图标）"""
    icon_path = ROOT / "build_icon" / "app.ico"
    if icon_path.exists():
        return icon_path
    return None


# 打包后 CONFIG_DIR = <exe 同级>/config/，与源码 config/ 是**两套独立目录**。
# settings.py / trading_guard.yaml / watchlist.json 已在包内
# （后两者由 settings._seed_external_config 在首启时播种到外挂目录）。
# 但下面这些「用户数据」不在包里，必须由构建脚本搬过去：
#   .salt                 密钥盐 —— 没有它 broker_accounts.enc 解不开
#   broker_accounts.enc   券商账号（密文）
#   deepseek.yaml         LLM 接口配置（含 api_key）
#   broker.yaml.template  券商配置模板
# 不搬的话，用户切到 exe 后会发现「账号没了 / AI 功能没了」——曾真实踩坑。
#
# 安全边界：只拷到 exe 同级的**外挂**目录（本机文件），**绝不**打进 exe。
# 并且只补缺失项，永不覆盖用户在 exe 里改过的版本。
# ⚠️ 刻意排除 settings.py：外挂 config/ 若也放一份，可能遮蔽包内模块。
RUNTIME_CFG_FILES = (".salt", "broker_accounts.enc", "deepseek.yaml",
                     "broker.yaml.template")


def _sync_runtime_config(target: Path) -> list[str]:
    """把源码 config/ 里的运行期用户配置补到 <target>/config/。

    返回本次实际拷贝的文件名列表（已存在的不动）。
    """
    cfg_src = ROOT / "config"
    cfg_dst = target / "config"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in RUNTIME_CFG_FILES:
        src = cfg_src / name
        dst = cfg_dst / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def build(with_data: bool, clean: bool) -> int:
    # 不做任何 rmtree：dist/ 里有 8335 个外挂数据文件、build/ 里有上次产物，
    # 批量删除会被守卫拦截导致打包失败。构建一律走"新暂存目录 + 增量同步"。
    if clean:
        log("⚠️ --clean 已禁用（避免批量删除被守卫拦截），改为增量构建")
    STAGE.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",              # 无控制台窗口
        "--name", APP_NAME,
        "--distpath", str(STAGE),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
        # 项目源码
        "--add-data", f"{ROOT / 'src'};src",
        "--add-data", f"{ROOT / 'config' / 'settings.py'};config",
        "--add-data", f"{ROOT / 'config' / 'watchlist.json'};config",
        # 风控配置模板：打进包后由 settings.py 在首次运行时拷到「exe 同级 config/」，
        # 用户改风控参数（回撤线/单票上限/下单频率等）无需重新打包。
        "--add-data", f"{ROOT / 'config' / 'trading_guard.yaml'};config",
        "--add-data", f"{ROOT / 'scripts'};scripts",
        # Streamlit 主题（品牌紫主色）。不打包这个目录，打包版会用默认西瓜红主色，
        # 和源码版观感不一致 —— 排查起来还以为是 CSS 没生效。
        "--add-data", f"{ROOT / '.streamlit'};.streamlit",
        # 依赖库动态资源：下方按导入闭包自动收集（--collect-all / --collect-data /
        # --add-data），不再在此硬编码。
    ]
    import importlib.util as _ilu

    def _have(mod: str) -> bool:
        try:
            return _ilu.find_spec(mod) is not None
        except Exception:
            return False

    # 动态分析导入闭包，自动补齐可能被手写清单漏掉的包 / 单文件模块
    _dyn_pkgs, _dyn_singles = set(), set()
    try:
        _dyn_pkgs, _dyn_singles = discover_closure()
        log(f"ℹ️ 导入闭包分析完成：{len(_dyn_pkgs)} 个包 / "
            f"{len(_dyn_singles)} 个单文件模块")
    except Exception as e:  # noqa: BLE001
        log(f"⚠️ 闭包分析失败，退回显式清单：{e}")

    # 1) 普通包：--collect-all（整包散拷进 _internal）
    _collect = (set(COLLECT_ALL) | _dyn_pkgs) - EXCLUDE_PKGS - set(COLLECT_DATA) - {"google"}
    for _pkg in sorted(_collect):
        if _have(_pkg):
            cmd += ["--collect-all", _pkg, "--hidden-import", _pkg]
        else:
            log(f"ℹ️ 跳过未安装包（不收集）: {_pkg}")

    # 2) 自带 testing 子模块会 importorskip 缺失包（如 hypothesis）的包：
    #    不能用 --collect-all（会触发 testing 导入崩溃），也不能用 --collect-data
    #    （只收数据文件、漏收 .py，导致 xgboost.XGBClassifier 缺失）。
    #    正确做法：整目录 --add-data 散拷（含全部 .py / 编译扩展 / 数据资源），
    #    并 --exclude-module 阻止 PyInstaller 自身去分析该包（避免 testing importorskip）。
    for _pkg in COLLECT_DATA:
        if _have(_pkg):
            _spec = _ilu.find_spec(_pkg)
            _origin = getattr(_spec, "origin", None)
            _loc = getattr(_spec, "submodule_search_locations", None)
            _dir = _loc[0] if _loc else (os.path.dirname(_origin) if _origin else None)
            if _dir and os.path.isdir(_dir):
                cmd += ["--add-data", f"{_dir};{_pkg}"]
                cmd += ["--exclude-module", _pkg]
                log(f"ℹ️ 整目录散拷（含 .py）: {_pkg} -> _internal/{_pkg}")
            else:
                cmd += ["--collect-all", _pkg, "--hidden-import", _pkg]
        else:
            log(f"ℹ️ 跳过未安装包（不收集）: {_pkg}")

    # 3) 命名空间包：--hidden-import
    for _pkg in ("google",):
        if _have(_pkg):
            cmd += ["--hidden-import", _pkg]

    # 4) 单文件模块处理
    #    · 顶层 .py 单文件模块：--collect-all 报 "not a package" 跳过，--add-data 散拷。
    #    · 顶层编译扩展（.pyd / .so，含 numpy 的 mypyc 产物 81d243…__mypyc.cp312….pyd）：
    #      PyInstaller 会把带点的文件名误当成「包路径」而试图 mkdir，与我们的散拷冲突，
    #      因此 --add-data 散拷 + --exclude-module 阻止 PyInstaller 自身的错误收集。
    _single_extra = {"typing_extensions", "six", "threadpoolctl", "jsonpath", "py"}

    def _handle_single(_mod):
        try:
            _spec = _ilu.find_spec(_mod)
        except Exception:
            return
        if _spec is None or not _spec.origin or not os.path.exists(_spec.origin):
            return
        _base = os.path.basename(_spec.origin)
        cmd.append("--add-data")
        cmd.append(f"{_spec.origin};{_base}")
        log(f"ℹ️ 单文件模块散拷: {_mod} -> _internal/{_base}")
        if _spec.origin.endswith((".pyd", ".so")):
            # 阻止 PyInstaller 把带点文件名当包路径 mkdir（否则与散拷文件冲突）
            cmd.append("--exclude-module")
            cmd.append(_mod)

    for _mod in sorted(_dyn_singles | _single_extra):
        _handle_single(_mod)

    # 4b) 补充扫描：site-packages 顶层编译扩展。运行时追踪（discovery）可能漏掉
    #     仅被 PyInstaller 静态分析捕获的 mypyc 产物（如 81d243…__mypyc.cp312….pyd）。
    #     这些顶层 .pyd/.so 必须显式散拷，并 --exclude-module 阻止 PyInstaller 误建目录。
    try:
        import site as _site
        for _sp in _site.getsitepackages():
            if not os.path.isdir(_sp):
                continue
            for _fn in os.listdir(_sp):
                if not _fn.endswith((".pyd", ".so")):
                    continue
                _full = os.path.join(_sp, _fn)
                if not os.path.isfile(_full):
                    continue
                # 导入名 = 去掉 ABI 标签（foo.cp312-win_amd64.pyd -> foo）
                _mod = _fn.split(".cp")[0] if ".cp" in _fn else _fn.rsplit(".", 1)[0]
                cmd.append("--add-data")
                cmd.append(f"{_full};{_fn}")
                cmd.append("--exclude-module")
                cmd.append(_mod)
                log(f"ℹ️ 顶层扩展散拷: {_mod} -> _internal/{_fn}")
    except Exception as e:  # noqa: BLE001
        log(f"⚠️ 顶层扩展扫描失败: {e}")

    # 5) 其余显式隐藏导入兜底
    for _pkg in HIDDEN_PKGS:
        if _have(_pkg.split(".")[0]):
            cmd += ["--hidden-import", _pkg]
    cmd += [
        # Streamlit 前端静态资源（必需，否则界面无样式）
        "--collect-data", "streamlit",
        "--copy-metadata", "streamlit",
        "--copy-metadata", "akshare",
        "--copy-metadata", "scikit-learn",
        "--copy-metadata", "xgboost",
        "--copy-metadata", "optuna",
        "--copy-metadata", "shap",
        # 减小体积 / 排除测试与重依赖
        # 注意：matplotlib 之前被 exclude，但实测 src/ui/app.py 导入闭包在加载时
        # 就拖进了 matplotlib（akshare 绘图模块），必须打进包，否则黑屏。
        "--exclude-module", "torch",
        "--exclude-module", "tensorflow",
        "--exclude-module", "notebook",
        "--exclude-module", "IPython",
        "--exclude-module", "pytest",
        "--exclude-module", "hypothesis",
        "--exclude-module", "dask",
        "--exclude-module", "langchain",
        # 各库的 tests/testing 子模块会 import pytest/hypothesis，打包必失败，排掉
        "--exclude-module", "numpy.tests",
        "--exclude-module", "numpy.testing",
        "--exclude-module", "scipy.tests",
        "--exclude-module", "pandas.tests",
        "--exclude-module", "pandas.testing",
        "--exclude-module", "sklearn.tests",
        "--exclude-module", "matplotlib.tests",
        "--exclude-module", "matplotlib.testing",
        "--exclude-module", "xgboost.testing",
        "--exclude-module", "optuna.testing",
        "--exclude-module", "akshare.tests",
        # httptools 是 uvicorn 的可选加速件；本项目打包里它的 .pyd 会被收成残包
        # （缺 __init__.py → import 成 namespace 包 → uvicorn 误用而崩，报
        # "module 'httptools' has no attribute 'HttpRequestParser'"）。
        # 必须显式 --exclude-module 阻止 PyInstaller 把它当 uvicorn 的传递依赖收进来；
        # 排除后 uvicorn 自动回退纯 Python 的 h11，本地看板完全够用。
        "--exclude-module", "httptools",
    ]

    icon = make_icon()
    if icon:
        cmd += ["--icon", str(icon)]

    cmd.append(str(ROOT / "launcher.py"))

    log("开始打包（首次约需 3–8 分钟，请耐心等待）…")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        log("❌ 打包失败")
        return proc.returncode

    # PyInstaller onedir 模式产物结构：
    #   <distpath>/选股神AI/选股神AI.exe   ← 启动器
    #   <distpath>/选股神AI/_internal/     ← 依赖库（含 add-data 的 src/config/scripts）
    staged = STAGE / APP_NAME
    exe_staged = staged / f"{APP_NAME}.exe"
    if not exe_staged.exists():
        log(f"❌ 未生成 exe（期望位置: {exe_staged}）")
        return 1

    # PyInstaller 会把「带点文件名的顶层编译扩展」（如
    # _cffi_backend.cp312-win_amd64.pyd、81d243…__mypyc.cp312-win_amd64.pyd）
    # 或单文件模块误当成「包路径」在 _internal 下建成目录，导致运行时
    # import 该模块得到空命名空间包而失败。这里在同步前把这类「目录伪装成的
    # 模块」还原成真实文件（取自 site-packages）。
    _internal = staged / "_internal"
    if _internal.is_dir():
        try:
            import site as _site
            _sp_dirs = _site.getsitepackages()
        except Exception:
            _sp_dirs = []
        for _entry in os.listdir(_internal):
            if not _entry.endswith((".pyd", ".so", ".py")):
                continue
            _p = _internal / _entry
            if not _p.is_dir():
                continue
            _real = None
            for _sp in _sp_dirs:
                _cand = os.path.join(_sp, _entry)
                if os.path.isfile(_cand):
                    _real = _cand
                    break
            if _real:
                shutil.rmtree(_p, ignore_errors=True)
                shutil.copy2(_real, str(_internal / _entry))
                log(f"ℹ️ 修正单文件模块: {_entry}（坏目录→真实文件）")
            else:
                log(f"⚠️ 无法修正目录型模块 {_entry}（site-packages 未找到对应文件）")

    # 保险：httptools 若仍被误收进 staged（理论上 --exclude-module 已阻止），
    # 删掉它让 uvicorn 回退纯 Python 的 h11，避免残包 namespace 包导致服务崩溃。
    _ht_staged = staged / "_internal" / "httptools"
    if _ht_staged.is_dir():
        shutil.rmtree(_ht_staged, ignore_errors=True)
        log("ℹ️ 移除残留 httptools 目录（回退 h11）")

    # ---- 增量同步到正式发布目录 ----
    # 只同步 PyInstaller 产物，绝不动外挂的 data/ 与 models/
    import filecmp
    target = DIST / APP_NAME
    target.mkdir(parents=True, exist_ok=True)
    # 同步前清掉 target 里的旧 httptools（robust_sync 只增不删，不清理会残留）
    _ht_target = target / "_internal" / "httptools"
    if _ht_target.is_dir():
        shutil.rmtree(_ht_target, ignore_errors=True)
        log("ℹ️ 移除 target 残留 httptools 目录（回退 h11）")

    def _robust_sync() -> tuple[int, list[str]]:
        """把 staged 整体同步到 target，逐文件处理、绝不抛出：
        · 目标是目录而源是文件（旧构建残留坏目录，如 81d243…__mypyc.cp312….pyd）
          → 先删目录再拷入文件
        · 文件被旧实例锁住（WinError 32）→ 跳过，保留旧文件（通常内容一致）
        · 源是目录而目标是文件 → 先删文件再建目录

        注意：主 exe 不在此处同步，由 _sync_main_exe() 单独处理（关键文件
        不容许静默失败，见下）。
        返回 (跳过的文件数, 被跳过文件的描述列表)。"""
        synced = 0
        skipped = 0
        skipped_paths: list[str] = []
        main_exe_rel = f"{APP_NAME}.exe"
        for root, _dirs, files in os.walk(staged):
            rel = Path(root).relative_to(staged)
            dst_dir = target / rel
            try:
                dst_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            for f in files:
                # 主 exe 由 _sync_main_exe() 负责（被占用时要换名而非跳过）
                if (rel / f).as_posix() == main_exe_rel:
                    continue
                s = Path(root) / f
                d = dst_dir / f
                # 类型冲突：目标是目录
                try:
                    if d.exists() and d.is_dir() and not d.is_symlink():
                        shutil.rmtree(d, ignore_errors=True)
                except OSError:
                    pass
                try:
                    shutil.copy2(s, d)
                    synced += 1
                except OSError as e:
                    skipped += 1
                    skipped_paths.append(f"{d}  ({e})")
                    log(f"⚠️ 跳过（{e}）: {d}")
        log(f"ℹ️ 同步：更新 {synced} 个文件，跳过 {skipped} 个（用时 {time.time() - t0:.0f}s）")
        return skipped, skipped_paths

    def _sync_main_exe() -> Path | None:
        """同步主 exe —— 关键文件，失败绝不能静默。

        target/选股神AI.exe 若被正在运行的旧实例锁住（WinError 32/13），
        拷贝会失败。此时不能像普通文件那样「跳过保留旧版」（用户会拿到
        带旧 Bug 的 exe 却毫不知情），而是写入带时间戳的新文件名绕开占用，
        并醒目提示用户。

        返回最终可用的 exe 绝对路径；新旧文件名都写不进去时返回 None。"""
        dst = target / f"{APP_NAME}.exe"
        try:
            shutil.copy2(exe_staged, dst)
            log(f"✅ 主程序已更新: {dst}")
            return dst
        except OSError as e_first:
            # Python 在 except ... as X: 块结束时会隐式 `del X`，
            # 因此先把异常对象存到块外仍可见的变量，供下面的失败分支引用
            # （否则第二次拷贝也失败时会抛 UnboundLocalError，
            #  导致 return None 变成死代码、操作指引打印不出来）。
            err_first: OSError | None = e_first
            log(f"⚠️ 主程序 {dst.name} 更新失败（{e_first}）")

        # 主文件名被占用 → 改用带时间戳的新文件名，绕开文件锁
        alt = target / f"{APP_NAME}_{time.strftime('%Y%m%d_%H%M%S')}.exe"
        try:
            shutil.copy2(exe_staged, alt)
        except OSError as e_alt:
            log("=" * 68)
            log("❌ 主程序无法写入：新旧文件名都被拒绝，打包失败")
            log(f"   · 旧文件 {dst}: {err_first}")
            log(f"   · 新文件 {alt}: {e_alt}")
            log("   请先关闭正在运行的旧实例，再重新打包。")
            log("=" * 68)
            return None

        log("=" * 68)
        log(f"⚠️  旧版 {APP_NAME}.exe 被占用（很可能还有实例正在运行），未能覆盖")
        log(f"    已改用新文件名：{alt.name}")
        log(f"    → 请关闭旧实例后，双击「{alt.name}」使用新版本")
        log(f"    → 或关闭旧实例后把 {alt.name} 改名覆盖为 {APP_NAME}.exe")
        log("=" * 68)
        return alt

    log("同步构建产物到 dist/ …")
    skipped, skipped_paths = _robust_sync()
    if skipped:
        log("=" * 68)
        log(f"⚠️  同步阶段有 {skipped} 个文件未更新（被占用/冲突，已保留旧文件）")
        log("   典型路径：")
        for _p in skipped_paths[:5]:
            log(f"   · {_p}")
        if skipped > 5:
            log(f"   · … 其余 {skipped - 5} 个，见上方逐条「⚠️ 跳过」日志")
        log("   常见原因：旧实例仍占用 dist/ 下的文件。")
        log("   若这些文件在上次构建后未改动，保留旧版是安全的；")
        log("   否则请关闭旧实例后重新打包，使 dist/ 与本次构建完全一致。")
        log("=" * 68)

    exe = _sync_main_exe()
    if exe is None:
        log("❌ 打包失败：dist/ 下没有任何可用的 exe（旧版被占用且无法写入新文件名）")
        return 1

    size_mb = sum(
        f.stat().st_size for f in target.rglob("*") if f.is_file()
    ) / 1024 / 1024
    log(f"✅ 打包完成: {exe}  (整体 {size_mb:.0f} MB, 用时 {time.time() - t0:.0f}s)")

    # 外挂模型目录（必须随包，AI 选股依赖）
    # ⚠️ 绝对不能 rmtree(dist/models) 再 copytree：整目录删除会触发批量删除守卫
    # （dist/models 文件数 > 阈值），守卫拦截删除后 copytree 因目标已存在而静默失败，
    # 最终打进包的是【旧模型】—— 这正是「修复没生效 / 双击仍是旧版」陷阱的根因。
    # 改为覆盖式同步（dirs_exist_ok=True）：已存在的文件原地覆盖更新，缺失的才新增，
    # 整目录永不删除，既不触发守卫，也保证源码模型 100% 进包。
    models_dst = target / "models"
    log("同步 models/ （覆盖式，不删旧目录）…")
    shutil.copytree(
        ROOT / "models", models_dst,
        ignore=shutil.ignore_patterns("backup_*", "*.log"),
        dirs_exist_ok=True,
    )
    m_mb = sum(f.stat().st_size for f in models_dst.rglob("*") if f.is_file()) / 1024 / 1024
    log(f"✅ models 已就位 ({m_mb:.0f} MB)")

    if with_data:
        data_dst = target / "data"
        log("同步 data/ （覆盖式，615MB，耗时较久）…")
        t1 = time.time()
        shutil.copytree(
            ROOT / "data", data_dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.tmp"),
            dirs_exist_ok=True,
        )
        d_mb = sum(f.stat().st_size for f in data_dst.rglob("*") if f.is_file()) / 1024 / 1024
        log(f"✅ data 已就位 ({d_mb:.0f} MB, {time.time() - t1:.0f}s)")
    else:
        log("ℹ️  未拷贝 data/（用户选择不含数据），首次使用会自动下载")

    # ---- 同步「运行期配置」到外挂 config/ ----------------------------------
    copied_cfg = _sync_runtime_config(target)
    if copied_cfg:
        log(f"✅ 运行期配置已同步到 config/: {', '.join(copied_cfg)}")
    else:
        log("ℹ️  config/ 运行期配置已是最新（跳过，不覆盖用户改动）")

    # 快捷启动方式说明
    (target / "使用说明.txt").write_text(
        f"""{APP_NAME} 使用说明
==================

启动方式
--------
双击「{exe.name}」即可启动，无需安装 Python 或任何依赖。

首次启动
--------
需要 10–60 秒加载模型，属正常现象。之后启动会快很多。

如果双击没反应
--------------
1. 检查 Windows 防火墙是否拦截（首次会弹窗询问，请点「允许」）
2. 右键 exe →「以管理员身份运行」
3. 查看 logs/launcher.log 了解详细原因

目录说明
--------
  data/    历史行情数据、交易计划、闭环日报、风控状态（首次使用会自动下载）
  models/  AI 模型文件（请勿删除）
  config/  外挂配置（改完即生效，重建 exe 不会丢）
             trading_guard.yaml   风控参数
             broker_accounts.enc  券商账号（密文）
             .salt                密钥盐 —— 与上者配对，删了就解不开账号
             deepseek.yaml        AI 对话接口配置
  libs/    外挂第三方库（可选，默认不存在）
           —— 想用【实盘自动交易】需把 xtquant 文件夹放到这里：
              libs/xtquant/...
              获取方式：pip download xtquant，解压出 xtquant 目录即可
  logs/    运行日志

风控参数怎么调
--------------
  编辑 config/trading_guard.yaml，可改：
    max_portfolio_drawdown_pct  组合回撤熔断线(%)
    max_daily_loss_pct          单日最大亏损比例(%)
    max_single_position_pct     单票最大仓位占比
    max_positions               最大持仓只数
    max_orders_per_minute       每分钟最大委托数
  改完保存即可，无需重启系统。

数据说明
--------
· 行情数据联网实时获取，本工具仅做本地缓存加速
· 所有数据仅供参考，不构成投资建议
· 市场有风险，决策需谨慎
""",
        encoding="utf-8",
    )
    log("✅ 使用说明已生成")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--with-data", action="store_true", help="同时拷贝 data/ 数据目录")
    p.add_argument("--no-clean", action="store_true", help="保留上次构建缓存（更快）")
    args = p.parse_args()
    return build(args.with_data, clean=not args.no_clean)


if __name__ == "__main__":
    sys.exit(main())
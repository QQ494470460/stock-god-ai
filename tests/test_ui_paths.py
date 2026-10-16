# -*- coding: utf-8 -*-
"""`src/ui/paths.py` 测试 —— 对「旧进程缓存模块」的免疫能力。

真实故障（用户实测）
------------------
看板进程启动后，我改了源码：给 `config/settings.py` 加了 `CONFIG_DIR`，
又在 `app.py` 里新加了 `from config.settings import CONFIG_DIR`。
Streamlit 只重跑 `app.py`、**不重载已导入模块**，于是新脚本撞上旧缓存：

    ImportError: cannot import name 'CONFIG_DIR' from 'config.settings'

`src/ui/paths.py` 是新增文件，不在任何旧进程的缓存里，因此总会从磁盘重读；
它内部对 `config.settings` 的导入做了兜底，并置位 `STALE_PROCESS` 供 UI 提示。

覆盖点：
  1. 正常进程：STALE_PROCESS=False，CONFIG_DIR = <项目根>/config
  2. 旧进程（settings 缺 CONFIG_DIR）：不抛异常、路径仍正确、STALE_PROCESS=True
  3. 旧进程 + 环境变量 STOCK_GOD_CONFIG_DIR：环境变量优先
  4. 旧进程 + 打包模式：<exe 同级>/config
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STALE_SETTINGS = str(ROOT / "config" / "settings.py")


@pytest.fixture
def paths_mod():
    """给出 paths 模块，并在测试后 reload 复原全局状态。

    teardown 顺序：本 fixture 先于 monkeypatch 建立，故晚于它拆除 ——
    reload 发生在 monkeypatch 复原 sys.modules 之后，不会残留污染。
    """
    import src.ui.paths as p
    yield p
    importlib.reload(p)


def _make_stale_settings(monkeypatch):
    """把 sys.modules 里的 config.settings 换成「缺 CONFIG_DIR 的旧版」。"""
    stale = types.ModuleType("config.settings")
    stale.__file__ = STALE_SETTINGS          # 旧模块仍有 __file__（兜底要用）
    assert not hasattr(stale, "CONFIG_DIR")
    monkeypatch.setitem(sys.modules, "config.settings", stale)


def test_normal_process(paths_mod):
    """正常进程：直接用 config.settings 的 CONFIG_DIR，不算 stale。"""
    importlib.reload(paths_mod)
    assert paths_mod.STALE_PROCESS is False
    assert Path(paths_mod.CONFIG_DIR).resolve() == (ROOT / "config").resolve()


def test_stale_settings_still_resolves(paths_mod, monkeypatch):
    """旧进程缓存缺 CONFIG_DIR 时，不得抛异常，且路径仍正确。"""
    _make_stale_settings(monkeypatch)
    monkeypatch.delenv("STOCK_GOD_CONFIG_DIR", raising=False)

    m = importlib.reload(paths_mod)          # 不应抛 ImportError

    assert m.STALE_PROCESS is True
    assert Path(m.CONFIG_DIR).resolve() == (ROOT / "config").resolve()


def test_stale_settings_respects_env(paths_mod, monkeypatch, tmp_path):
    """兜底分支同样遵守环境变量优先级（与 _resolve_dir 一致）。"""
    _make_stale_settings(monkeypatch)
    monkeypatch.setenv("STOCK_GOD_CONFIG_DIR", str(tmp_path))

    m = importlib.reload(paths_mod)

    assert m.STALE_PROCESS is True
    assert Path(m.CONFIG_DIR) == tmp_path


def test_stale_settings_frozen_uses_exe_dir(paths_mod, monkeypatch, tmp_path):
    """打包模式：配置目录在 exe 同级，而非包内的 _internal/config。"""
    _make_stale_settings(monkeypatch)
    monkeypatch.delenv("STOCK_GOD_CONFIG_DIR", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "选股神AI.exe"))

    m = importlib.reload(paths_mod)

    assert m.STALE_PROCESS is True
    assert Path(m.CONFIG_DIR) == tmp_path / "config"


def test_ui_modules_do_not_import_config_directly():
    """回归保护：UI 页面里不得再出现 `from config.settings import CONFIG_DIR`
    （会重新引入旧进程 ImportError 的隐患）。

    `paths.py` 是唯一许可的例外 —— 它就是那层兜底本身。
    """
    offenders = []
    for name in ("app.py", "trading_ui.py"):
        text = (ROOT / "src" / "ui" / name).read_text(encoding="utf-8")
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith("#"):
                continue
            if s.startswith("from config.settings import") and "CONFIG_DIR" in s:
                offenders.append(f"src/ui/{name}: {s}")
    assert offenders == [], "发现直连 config.settings 的 CONFIG_DIR 导入：" + "; ".join(offenders)

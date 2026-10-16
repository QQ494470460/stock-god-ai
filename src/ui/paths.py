# -*- coding: utf-8 -*-
"""配置目录解析（对「旧进程缓存」免疫）。

背景 —— 一个真实的坑
====================
Streamlit 每次交互只**重新执行 `app.py`**，但**不会重新导入 `sys.modules`
里已缓存的模块**。因此当你在源码里新增了 `config/settings.py` 的
`CONFIG_DIR`，又同时在 `app.py` 里新加 `from config.settings import CONFIG_DIR`
时，一个**启动于改动之前**的看板进程会这样崩：

    ImportError: cannot import name 'CONFIG_DIR' from 'config.settings'
    File "src/ui/app.py", line 2540, in render_safety_panel

新脚本（从磁盘重读）撞上旧模块（内存缓存）—— 重启看板即可消失，
但报错信息对使用者毫无指向性。

本模块的解法
============
`src/ui/paths.py` 是**新增文件**，不可能存在于任何旧进程的缓存里，
所以 `from src.ui.paths import CONFIG_DIR` 一定会从磁盘重新加载。
它内部对 `config.settings` 的导入做了兜底，即使拿到的是缺字段的旧模块，
也能按与 `config.settings._resolve_dir` 完全一致的规则算出正确目录。

优先级（与 `_resolve_dir` 保持一致）：
    1. 环境变量 ``STOCK_GOD_CONFIG_DIR``
    2. 打包模式：``<exe 同级>/config``
    3. 源码模式：``config/settings.py`` 所在目录
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["CONFIG_DIR", "STALE_PROCESS", "resolve_config_dir"]

# 是否检测到「旧进程里缓存的 config.settings 已过时」。
# 供 UI 提示使用者重启，而不是让人对着 ImportError 发呆。
STALE_PROCESS = False


def resolve_config_dir() -> Path:
    """解析配置目录，兼容旧进程缓存的 `config.settings`。"""
    global STALE_PROCESS

    try:
        from config.settings import CONFIG_DIR as _c
        return Path(_c)
    except ImportError:
        # 旧进程缓存：模块存在但没有 CONFIG_DIR 字段
        STALE_PROCESS = True

    env_val = os.environ.get("STOCK_GOD_CONFIG_DIR")
    if env_val:
        return Path(env_val)

    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "config"

    try:
        import config.settings as _cs
        return Path(_cs.__file__).resolve().parent
    except Exception:  # noqa: BLE001  连模块都拿不到，退回项目根 config/
        return Path(__file__).resolve().parent.parent.parent / "config"


CONFIG_DIR = resolve_config_dir()

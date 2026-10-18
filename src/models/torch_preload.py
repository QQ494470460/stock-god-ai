# -*- coding: utf-8 -*-
"""在 xtquant 之前把 torch 拉起来（Windows 原生 DLL 的初始化顺序问题）。

现象（已复现）
------------
先 ``import xtquant``（``xttrader`` / ``xttype``）再 ``import torch``：::

    OSError [WinError 1114] 动态链接库(DLL)初始化例程失败。
    Error loading ".../torch/lib/c10.dll" or one of its dependencies.

反过来（先 ``torch`` 后 ``xtquant``）两者都能正常加载。

影响
----
``LSTMSubPredictor`` 是**懒加载**的，等到真正预测时才 ``import torch`` —— 那时
``src.trading.yuekai`` 早已把 xtquant 拉进进程（券商账号一加载就触发），于是
LSTM 这条腿在实践中**必死**：集成悄悄从三条腿退化成两条，而日志里只有一行
DEBUG，看板完全看不出来。

用法
----
在**任何** import xtquant 之前调一次 ``preload_torch()``（幂等，失败静默）。
设环境变量 ``STOCKGOD_PRELOAD_TORCH=0`` 可关闭（排查用）。
"""
from __future__ import annotations

import os

from loguru import logger

_done = False


def preload_torch(reason: str = "") -> bool:
    """把 torch 提前 import 进来；已加载或不可用时都安全返回。"""
    global _done
    if _done:
        return True
    _done = True
    if os.environ.get("STOCKGOD_PRELOAD_TORCH", "1") == "0":
        logger.debug("[torch_preload] 已按环境变量关闭")
        return False
    try:
        import torch  # noqa: F401
        logger.debug(f"[torch_preload] torch {torch.__version__} 已预载"
                     + (f"（{reason}）" if reason else ""))
        return True
    except Exception as e:  # noqa: BLE001
        # 预载不是硬要求：torch 装不上时，LSTM 腿本就该降级，不能拖垮主流程
        logger.debug(f"[torch_preload] 预载失败，LSTM 腿将降级: {e}")
        return False

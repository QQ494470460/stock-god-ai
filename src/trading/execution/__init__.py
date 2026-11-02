# -*- coding: utf-8 -*-
"""执行层智能下单（T2）—— TWAP/VWAP 切片、限价重试、隔夜预挂、滑点建模。

设计原则
--------
1. **完全增量、opt-in、默认关闭**：`TradingLoopConfig.execution.smart_routing=False`
   时，闭环行为与改造前 100% 一致（所有委托在单个 tick 内一次性发出）。
2. **不动运行中的 loop**：本包被 loop 加载进内存后才生效；旧进程不受影响。
3. **可回测 / 可单测**：核心逻辑是纯函数 + 状态机，不碰网络，便于独立测试。
4. **不破坏风控**：切片只是把"同一笔调仓"在时间维度摊薄，不改变目标组合、
   权重、现金底线，也不绕过 guard 裁决（每片仍经 `_place`→`propose_trade`）。
"""
from src.trading.execution.slice import twap_slices, vwap_slices, choose_n_slices
from src.trading.execution.retry import LimitRetry
from src.trading.execution.overnight import preopen_release_dt, is_preopen
from src.trading.execution.slippage import (
    SimSlippageModel,
    set_active_slippage_model,
    get_active_slippage_model,
)
from src.trading.execution.scheduler import ExecutionScheduler, Slice, PendingOrder

__all__ = [
    "twap_slices", "vwap_slices", "choose_n_slices",
    "LimitRetry", "preopen_release_dt", "is_preopen",
    "SimSlippageModel", "set_active_slippage_model", "get_active_slippage_model",
    "ExecutionScheduler", "Slice", "PendingOrder",
]

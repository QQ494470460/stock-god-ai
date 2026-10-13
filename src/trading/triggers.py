# -*- coding: utf-8 -*-
"""风控触发结构化日志

把「日内亏损熔断 / 日成交上限 / 止损 / 止盈 / 移动止损」等触发事件，
以 JSONL 形式追加写入 ``data/audit_triggers.jsonl``。

用途：
    ``scripts/paper_shadow_audit.py`` 的 Panel 3 可直接读出
    「实盘/纸盘曾几次被熔断、几次触发止损」，而无需从盈亏反推。

设计：
    - 只追加、不覆盖，便于长期累积。
    - 任何写入异常都被吞掉（风控日志绝不能影响主交易链路）。
    - DATA_DIR 独立计算，避免与 paper.py 形成 import 环。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# 与 src/trading/paper.DATA_DIR 同义：项目根 / "data"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TRIGGER_LOG = DATA_DIR / "audit_triggers.jsonl"

_LOCK = threading.Lock()


def log_trigger(event_type: str, **fields) -> None:
    """追加一条触发事件。

    Args:
        event_type: daily_loss_circuit | daily_trade_cap | stop_loss |
                    stop_profit | trailing_stop
        **fields: 任意附加字段（code/name/pnl_pct/reason/quantity ...）
    """
    rec = {"ts": datetime.now().isoformat(), "type": event_type}
    rec.update(fields)
    try:
        with _LOCK:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with TRIGGER_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001  风控日志失败绝不影响主链路
        pass


def _classify(reason: str) -> str:
    """从 executor 的 alert reason 文本推断触发类型"""
    if "移动止损" in (reason or ""):
        return "trailing_stop"
    if "止盈" in (reason or ""):
        return "stop_profit"
    return "stop_loss"

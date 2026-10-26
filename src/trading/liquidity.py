# -*- coding: utf-8 -*-
r"""流动性约束：ADV（平均日成交额）→ 单笔可交易上限。

为什么要单独一层
================
纸交易与调仓计划默认「挂多少成交多少」，但真实市场里单笔吃掉日成交额的
较高比例就会显著冲击价格（冲击成本分档见 :mod:`src.trading.rules`），
极端情况下甚至打上涨停板、完全成交不了。

本仓当前规模 ≈ ¥470 万 / 20 只 ≈ 每只 ¥38 万。对大盘股（日成交额数十亿）
这不构成约束；但对中小市值标的，¥38 万可能是其日成交额的 3%~15%，
不做约束等价于**假设无限流动性**，回测与实盘会系统性偏离。

口径
----
* ADV 用**近 N 个交易日的成交额均值**（元），只取 ``amount > 0`` 的有效日；
* 单笔上限 = ``ADV × participation_rate``（默认 1%，机构常见 1%~5%）；
* 取整到整手；数据缺失返回 ``None``，调用方据此**不施加**该约束（fail-open
  但可观测——绝不因为缺数据就拒绝交易，也绝不假装数据存在）。
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

try:  # 打包/测试环境下 config 可能不在 sys.path 顶层
    from config.settings import DATA_DIR as _DATA_DIR
except Exception:  # noqa: BLE001
    _DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

__all__ = ["AdvProvider", "PARTICIPATION_DEFAULT"]

#: 单笔最多吃掉多大的日成交额（1%）。越低越保守，但会限制可建仓规模。
PARTICIPATION_DEFAULT = 0.01


class AdvProvider:
    """从本地日线 parquet 估计平均日成交额，带进程内缓存（按自然日失效）。"""

    def __init__(self, data_dir: Optional[Path] = None, lookback: int = 20,
                 cache: bool = True):
        self.data_dir = Path(data_dir) if data_dir else Path(_DATA_DIR)
        self.lookback = max(int(lookback), 1)
        self._cache: Dict[str, Optional[float]] = {}
        self._cache_day = ""
        self._cache_on = cache

    # ------------------------------------------------------------------
    def _roll_day_if_needed(self) -> None:
        today = datetime.date.today().isoformat()
        if self._cache_day != today:
            self._cache.clear()
            self._cache_day = today

    def adv_amount(self, code: str) -> Optional[float]:
        """近 ``lookback`` 日平均成交额（元）；数据缺失返回 None。"""
        code = str(code or "").strip()
        if not code:
            return None
        self._roll_day_if_needed()
        if self._cache_on and code in self._cache:
            return self._cache[code]
        val = self._compute(code)
        if self._cache_on:
            self._cache[code] = val
        return val

    def _compute(self, code: str) -> Optional[float]:
        p = self.data_dir / f"daily_{code}_qfq.parquet"
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ADV 读取失败 {code}: {e}")
            return None
        if df is None or len(df) == 0:
            return None

        series = None
        if "amount" in df.columns:
            series = pd.to_numeric(df["amount"], errors="coerce")
        elif "volume" in df.columns and "close" in df.columns:
            # 兜底：成交额 ≈ 成交量 × 收盘价（对没有 amount 列的数据源）
            series = (pd.to_numeric(df["volume"], errors="coerce")
                      * pd.to_numeric(df["close"], errors="coerce"))
        if series is None:
            return None

        s = series.dropna()
        s = s[s > 0]
        if s.empty:
            return None
        tail = s.to_numpy(dtype=float)[-self.lookback:]
        try:
            return float(tail.mean())
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    def max_order_value(self, code: str,
                        participation: float = PARTICIPATION_DEFAULT) -> Optional[float]:
        """单笔金额上限（元）；ADV 未知返回 None（调用方据此不约束）。"""
        adv = self.adv_amount(code)
        if adv is None or adv <= 0:
            return None
        rate = max(0.0, float(participation))
        return float(adv) * rate

    def max_order_quantity(self, code: str, price: float,
                           participation: float = PARTICIPATION_DEFAULT,
                           lot: int = 100) -> Optional[int]:
        """单笔股数上限（整手）；ADV 未知或价格非法返回 None。"""
        try:
            px = float(price)
        except (TypeError, ValueError):
            return None
        if px <= 0:
            return None
        mv = self.max_order_value(code, participation)
        if mv is None:
            return None
        lot = max(int(lot), 1)
        return int(mv / px // lot * lot)

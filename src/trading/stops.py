# -*- coding: utf-8 -*-
"""自适应止损 —— 用 ATR 把「固定百分比」换成「波动率归一」。

问题
----
执行侧现在只有一条固定止损线（``stop_loss_pct = -5%``）。这在 A 股是**结构性错误**：

- 年化波动 20% 的白马，日内 ±3% 就算大波动，-5% 止损基本只会在趋势真坏了时触发——合理；
- 年化波动 70% 的题材股，日内振幅本身就有 ±5%，-5% 止损等于「开盘就被扫出」。
  2026-10 的实盘复盘里，多笔亏损单的共同特征是：**被日常噪声打掉止损，
  随后股价按原方向走回去**。这不是选股错，是止损宽度没跟波动对齐。

做法
----
以 ``ATR/close`` 为基准，止损宽度 = ``max(|基础止损|, k × ATR%)``，
再用上下限封顶（下限防「太紧」、上限防「形同虚设」）。
``k=2`` 的含义是「容忍两倍日均真实波幅」，这是海龟/趋势跟踪的传统参数。

同一套逻辑也用于移动止损：波动大的票回撤容忍度更宽，不会被一根阴线洗出去。

缓存与降级
----------
``AtrProvider`` 按标的缓存（同一 tick 内多次调用只读一次盘）；读不到
parquet 或列不全时返回 ``None``，调用方回退到固定百分比 —— **fail-open**，
数据缺失不能让止损失效，也不能让它变成 0。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# 容忍度上限：再高的波动也不允许止损宽到 15%（否则等于没有止损）
MAX_STOP_PCT = 15.0
# 容忍度下限：再低的波动也不允许止损紧到 2%（否则交易成本就能触发）
MIN_STOP_PCT = 2.0


def adaptive_stop_pct(
    base_pct: float,
    atr_pct: Optional[float],
    k: float = 2.0,
    floor_pct: float = MIN_STOP_PCT,
    ceil_pct: float = MAX_STOP_PCT,
) -> float:
    """返回**负的**止损线（与 ``RiskConfig.stop_loss_pct`` 同口径，如 -7.2）。

    Args:
        base_pct: 配置里的固定止损（负数，如 -5.0）
        atr_pct: ATR / 最新收盘价（小数，如 0.035 表示 3.5%）；None = 无数据
        k: ATR 倍数，2.0 = 容忍两倍日均波幅
        floor_pct: 止损宽度的下限（正数，表示至少这么宽）
        ceil_pct: 止损宽度的上限（正数）

    Returns:
        负数止损线；``atr_pct`` 缺失时原样返回 ``base_pct``（不猜、不放大）。
    """
    try:
        base = float(base_pct)
    except (TypeError, ValueError):
        return -MIN_STOP_PCT
    if atr_pct is None:
        return base
    try:
        a = float(atr_pct)
    except (TypeError, ValueError):
        return base
    if not np.isfinite(a) or a <= 0:
        return base

    width = max(abs(base), float(k) * a * 100.0)
    lo = max(float(floor_pct), 0.0)
    hi = max(float(ceil_pct), lo)
    width = min(max(width, lo), hi)
    return -width


def adaptive_trailing_pct(
    base_pct: float,
    atr_pct: Optional[float],
    k: float = 1.5,
    floor_pct: float = MIN_STOP_PCT,
    ceil_pct: float = MAX_STOP_PCT,
) -> float:
    """移动止损宽度（负数）。默认 k=1.5：移动止损应该比硬止损更紧一点，
    否则已经赚到的浮盈会在回吐时全部还回去。"""
    return adaptive_stop_pct(base_pct, atr_pct, k=k, floor_pct=floor_pct, ceil_pct=ceil_pct)


class AtrProvider:
    """从 ``data/daily_{code}_qfq.parquet`` 估 ATR 占收盘价的比例（带缓存）。"""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        periods: int = 14,
        lookback: int = 60,
        cache: bool = True,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.periods = max(int(periods), 1)
        self.lookback = max(int(lookback), self.periods + 1)
        self._cache: Dict[str, Optional[float]] = {}
        self._cache_on = cache
        self._cache_day = ""

    def _roll(self) -> None:
        """跨自然日清缓存：常驻进程不会重启，不清就会拿昨天的 ATR 做今天的止损。"""
        import datetime

        today = datetime.date.today().isoformat()
        if self._cache_day != today:
            self._cache.clear()
            self._cache_day = today

    # ---------- 读盘 ----------
    def _read_ohlc(self, code: str):
        p = self.data_dir / f"daily_{code}_qfq.parquet"
        if not p.exists():
            return None
        try:
            import pandas as pd

            df = pd.read_parquet(p)
            if df is None or len(df) == 0:
                return None
            cols = {}
            for want, cands in (
                ("high", ("high", "最高")),
                ("low", ("low", "最低")),
                ("close", ("close", "收盘")),
            ):
                for c in cands:
                    if c in df.columns:
                        cols[want] = c
                        break
                else:
                    return None  # 缺任一列就放弃，绝不拿 close 近似 high/low
            h = pd.to_numeric(df[cols["high"]], errors="coerce").dropna()
            l = pd.to_numeric(df[cols["low"]], errors="coerce").dropna()
            c = pd.to_numeric(df[cols["close"]], errors="coerce").dropna()
            n = min(len(h), len(l), len(c))
            if n < 2:
                return None
            return (
                h.to_numpy(dtype=float)[-self.lookback:][-n:],
                l.to_numpy(dtype=float)[-self.lookback:][-n:],
                c.to_numpy(dtype=float)[-self.lookback:][-n:],
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ATR 读盘失败 {code}: {e}")
            return None

    # ---------- 对外 ----------
    def atr_pct(self, code: str) -> Optional[float]:
        """ATR / 最新收盘价；无数据返回 None（调用方回退固定止损）。"""
        code = str(code)
        self._roll()
        if self._cache_on and code in self._cache:
            return self._cache[code]

        val: Optional[float] = None
        ohlc = self._read_ohlc(code)
        if ohlc is not None:
            h, l, c = ohlc
            try:
                from src.trading.analytics import atr_percent

                v = float(atr_percent(h, l, c, periods=self.periods))
                val = v if np.isfinite(v) and v > 0 else None
            except Exception as e:  # noqa: BLE001
                logger.debug(f"ATR 计算失败 {code}: {e}")
                val = None

        if self._cache_on:
            self._cache[code] = val
        return val

    @staticmethod
    def from_ohlc(high, low, close, periods: int = 14) -> Optional[float]:
        """直接喂 OHLC（测试 / 实时数据），不读盘。"""
        try:
            from src.trading.analytics import atr_percent

            v = float(atr_percent(high, low, close, periods=periods))
            return v if np.isfinite(v) and v > 0 else None
        except Exception:  # noqa: BLE001
            return None

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_day = ""

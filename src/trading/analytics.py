# -*- coding: utf-8 -*-
"""风险与绩效分析工具（自动交易闭环的可观测性基座）。

把原本散落在 guard / paper / evolution 里的「波动率估算」「绩效归因」「walk-forward
切分」等能力收敛到一处，作为后续「波动率自适应仓位」「防过拟合进化门禁」的共用底座。

设计原则：
- 纯函数 + 可注入数据，便于单元测试（不依赖实时行情/网络）。
- 不触碰生产状态文件；只读 ``data/daily_{code}_qfq.parquet`` 做历史波动率估计。
- 与纸账户现有 ``close_id`` 配对、``today_pnl_market`` 浮盈口径互不冲突：
  此处只做「已实现 + 已平仓配对」的归因（卖出才有 pnl），回答「到底哪套策略在赚钱」。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    from src.trading.paper import TradeRecord  # 仅类型标注，运行时不做 IO
except Exception:  # noqa: BLE001
    TradeRecord = None  # type: ignore

DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ======================================================================
# 波动率估计
# ======================================================================
def annualized_volatility(returns: Sequence[float], periods_per_year: int = 242) -> float:
    """由收益率序列估计年化波动率。

    returns 为空或常数 → 返回 0.0（无波动）。数值稳定：标准差用样本估计并防 0。
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    return std * np.sqrt(periods_per_year)


def atr_percent(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    periods: int = 14,
) -> float:
    """由 OHLC 估计 ATR 占最近收盘价的百分比（用于 per-share 风险预算）。"""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if h.size < 2:
        return 0.0
    prev_close = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([
        (h - l),
        np.abs(h - prev_close),
        np.abs(l - prev_close),
    ])
    atr = float(pd.Series(tr).rolling(periods, min_periods=1).mean().iloc[-1])
    last = float(c[-1])
    return (atr / last) if last > 0 else 0.0


class VolatilityEstimator:
    """从历史日线估计单标的年化波动率，带进程内内存缓存（不重复读盘）。"""

    def __init__(self, data_dir: Optional[Path] = None, lookback: int = 60, cache: bool = True):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.lookback = lookback
        self._cache: Dict[str, Optional[float]] = {}
        self._cache_on = cache

    def _read_closes(self, code: str) -> Optional[np.ndarray]:
        p = self.data_dir / f"daily_{code}_qfq.parquet"
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
            col = "close" if "close" in df.columns else ("收盘" if "收盘" in df.columns else None)
            if col is None or df.empty:
                return None
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                return None
            return s.to_numpy(dtype=float)[-self.lookback:]
        except Exception as e:  # noqa: BLE001
            logger.debug(f"波动率估计读盘失败 {code}: {e}")
            return None

    def annualized_vol(self, code: str, periods_per_year: int = 242) -> Optional[float]:
        """返回该标的年化波动率；无历史数据返回 None（调用方据此优雅降级）。"""
        if self._cache_on and code in self._cache:
            return self._cache[code]
        closes = self._read_closes(code)
        vol: Optional[float] = None
        if closes is not None and closes.size >= 2:
            rets = np.diff(closes) / closes[:-1]
            vol = annualized_volatility(rets, periods_per_year)
        if self._cache_on:
            self._cache[code] = vol
        return vol

    @staticmethod
    def from_prices(prices: Sequence[float], periods_per_year: int = 242) -> float:
        """直接喂价格序列（测试 / 实时行情）估计年化波动率。"""
        arr = np.asarray(prices, dtype=float)
        if arr.size < 2:
            return 0.0
        rets = np.diff(arr) / arr[:-1]
        return annualized_volatility(rets, periods_per_year)


# ======================================================================
# 绩效归因
# ======================================================================
def attribute_performance(trades: List["TradeRecord"]) -> Dict:
    """对成交序列做最朴素的「信号归因」：按策略标签拆解贡献。

    返回：total_pnl / win_rate / profit_factor / avg_win / avg_loss /
    by_strategy{tag: {pnl, count, win_rate, avg_pnl}} / n_trades / n_sells。
    """
    sells = [t for t in trades if getattr(t, "side", None) == "sell" and getattr(t, "pnl", 0) != 0]
    total = sum(float(t.pnl) for t in sells)
    wins = [t for t in sells if t.pnl > 0]
    losses = [t for t in sells if t.pnl < 0]
    win_rate = len(wins) / len(sells) if sells else 0.0
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = -gross_loss / len(losses) if losses else 0.0

    by_strategy: Dict[str, Dict[str, float]] = {}
    for t in sells:
        tag = getattr(t, "strategy", "") or "unknown"
        d = by_strategy.setdefault(tag, {"pnl": 0.0, "count": 0, "wins": 0})
        d["pnl"] += float(t.pnl)
        d["count"] += 1
        if t.pnl > 0:
            d["wins"] += 1
    for d in by_strategy.values():
        d["win_rate"] = d["wins"] / d["count"] if d["count"] else 0.0
        d["avg_pnl"] = d["pnl"] / d["count"] if d["count"] else 0.0

    return {
        "n_trades": len(trades),
        "n_sells": len(sells),
        "total_pnl": round(total, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 2) if pf != float("inf") else 999.0,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "by_strategy": {
            k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in by_strategy.items()
        },
    }


# ======================================================================
# Walk-forward 切分（进化门禁用）
# ======================================================================
def walk_forward_splits(
    n_folds: int = 5,
    train_size: int = 60,
    test_size: int = 20,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """生成 walk-forward 的 ``(train, test)`` 闭区间下标元组列表。

    用于进化门禁：模型先在 train 窗口拟合，必须在紧邻的 test 窗口上达标才允许 pin，
    杜绝「全样本回测好看、实盘拉胯」的过拟合。每个 fold 在样本上滑动一格，
    train 与 test 紧邻且不重叠。

    Returns:
        ``[((t0,t1),(e0,e1)), ...]``，下标可直接喂 pandas/numpy slice。
    """
    folds: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for k in range(n_folds):
        t0 = k * test_size
        t1 = t0 + train_size - 1
        e0 = t1 + 1
        e1 = e0 + test_size - 1
        folds.append(((t0, t1), (e0, e1)))
    return folds

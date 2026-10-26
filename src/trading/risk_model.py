# -*- coding: utf-8 -*-
r"""组合风险模型：相关性矩阵 → 组合波动率 → 参数法 VaR。

为什么仓位比例不够
==================
「单票 8%、最多 20 只」只约束了**名义敞口**，没有约束**风险敞口**。
20 只同涨同跌的票，名义分散、风险却等于 1 只加杠杆；反之低相关组合
同样的名义仓位，回撤会小得多。

本模块把「持仓之间的相关性」显式算出来，得到：
* 组合日波动率 σ_p = sqrt(wᵀ Σ w)
* 参数法 1 日 VaR（95%）= 1.65 × σ_p × 组合市值
* 平均两两相关性（拥挤度代理：越高说明越像同一笔交易）

用途：guard 在**新开仓**前检查组合 VaR 是否超预算，超了就按比例缩放
下单强度（连续降仓，不是清仓）。

数据来源：本地日线 parquet（与 :mod:`src.trading.analytics` 同源），
按自然日缓存；缺数据的标的自动从矩阵里剔除（不编数据）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:  # 打包/测试环境下 config 可能不在 sys.path 顶层
    from config.settings import DATA_DIR as _DATA_DIR
except Exception:  # noqa: BLE001
    _DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

__all__ = [
    "daily_returns", "returns_matrix", "covariance", "portfolio_daily_vol",
    "parametric_var_pct", "avg_pairwise_corr", "PortfolioRiskMonitor",
]


# ======================================================================
# 纯函数（可单测，无 IO）
# ======================================================================
def daily_returns(closes: Sequence[float]) -> np.ndarray:
    """由收盘价序列算简单日收益率。不足 2 个点返回空数组。"""
    arr = np.asarray(closes, dtype=float)
    arr = arr[arr > 0]
    if arr.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(arr) / arr[:-1]


def returns_matrix(series: Dict[str, Sequence[float]]) -> Tuple[List[str], np.ndarray]:
    """把多只标的的价格序列对齐成收益率矩阵（按尾部对齐，缺数据即剔除）。

    Returns
    -------
    (codes, rets)：``rets`` 形状 ``(T-1, N)``，``T`` 为所有可用序列的最短长度。
    """
    usable: Dict[str, np.ndarray] = {}
    for code, closes in (series or {}).items():
        arr = np.asarray(closes, dtype=float)
        arr = arr[arr > 0]
        if arr.size >= 2:
            usable[str(code)] = arr

    if not usable:
        return [], np.empty((0, 0))

    n = min(len(v) for v in usable.values())
    codes = list(usable.keys())
    mat = np.column_stack([usable[c][-n:] for c in codes])
    rets = np.diff(mat, axis=0) / mat[:-1]
    return codes, rets


def covariance(rets: np.ndarray) -> np.ndarray:
    """样本协方差矩阵；列数 < 2 或样本不足时返回零矩阵。"""
    arr = np.asarray(rets, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 1:
        return np.zeros((arr.shape[1] if arr.ndim == 2 else 0,) * 2)
    return np.cov(arr, rowvar=False, ddof=1)


def portfolio_daily_vol(weights: Sequence[float], cov: np.ndarray) -> float:
    """组合日波动率 σ_p = sqrt(wᵀ Σ w)。"""
    w = np.asarray(weights, dtype=float)
    c = np.asarray(cov, dtype=float)
    if w.size == 0 or c.shape != (w.size, w.size):
        return 0.0
    var = float(w @ c @ w)
    return float(np.sqrt(max(var, 0.0)))


def parametric_var_pct(daily_vol: float, z: float = 1.65) -> float:
    """参数法 VaR 占组合市值的比例（默认 95% 单尾，z=1.65）。"""
    return float(max(daily_vol, 0.0) * max(z, 0.0))


def avg_pairwise_corr(rets: np.ndarray) -> float:
    """平均两两相关系数（拥挤度代理）。样本不足返回 0。"""
    arr = np.asarray(rets, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 3:
        return 0.0
    corr = np.corrcoef(arr, rowvar=False)
    if corr.ndim != 2:
        return 0.0
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = corr[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0


# ======================================================================
# 读盘 + 缓存
# ======================================================================
class PortfolioRiskMonitor:
    """从本地日线估计持仓组合的 VaR 与平均相关性（按自然日缓存）。"""

    def __init__(self, data_dir: Optional[Path] = None, lookback: int = 120,
                 cache: bool = True):
        self.data_dir = Path(data_dir) if data_dir else Path(_DATA_DIR)
        self.lookback = max(int(lookback), 20)
        self._cache: Dict[str, Optional[np.ndarray]] = {}
        self._cache_day = ""
        self._cache_on = cache

    def _roll(self) -> None:
        import datetime
        today = datetime.date.today().isoformat()
        if self._cache_day != today:
            self._cache.clear()
            self._cache_day = today

    def closes(self, code: str) -> Optional[np.ndarray]:
        code = str(code or "").strip()
        if not code:
            return None
        self._roll()
        if self._cache_on and code in self._cache:
            return self._cache[code]
        out: Optional[np.ndarray] = None
        p = self.data_dir / f"daily_{code}_qfq.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                col = "close" if "close" in df.columns else None
                if col:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    s = s[s > 0]
                    if s.size:
                        out = s.to_numpy(dtype=float)[-self.lookback:]
            except Exception as e:  # noqa: BLE001
                logger.debug(f"风险模型读盘失败 {code}: {e}")
        if self._cache_on:
            self._cache[code] = out
        return out

    def var_pct(
        self, codes: Sequence[str], weights: Sequence[float], z: float = 1.65,
    ) -> Optional[Tuple[float, float]]:
        """返回 ``(VaR 占**总资产**比例, 平均两两相关性)``；数据不足返回 None。

        ``weights`` 必须是 **相对总资产** 的权重（``市值 / 总资产``），
        本方法**不做归一化** —— 归一化会把「只用了 18% 仓位的低风险状态」
        算成满仓风险，从而错误地触发降仓。

        拿不到历史数据的标的会被剔除，其权重不计入（相当于按「已实现风险」
        统计，略偏乐观），因此返回值里附带的可用只数应一并观察。
        """
        pairs = []
        for c, w in zip(codes, weights):
            closes = self.closes(c)
            if closes is None or closes.size < 21:
                continue
            pairs.append((c, float(w), closes))
        if len(pairs) < 2:
            return None

        usable_codes, rets = returns_matrix({c: cl for c, _w, cl in pairs})
        if rets.size == 0:
            return None

        wmap: Dict[str, float] = {}
        for c, w, _cl in pairs:
            wmap[c] = wmap.get(c, 0.0) + w
        w = np.asarray([wmap.get(c, 0.0) for c in usable_codes], dtype=float)
        if not np.any(w > 0):
            return None

        cov = covariance(rets)
        vol = portfolio_daily_vol(w, cov)
        return parametric_var_pct(vol, z), avg_pairwise_corr(rets)

    def var_pct_single(
        self, code: str, weight: float, z: float = 1.65,
    ) -> Optional[float]:
        """单标的组合的 1 日 VaR（占**总资产**比例）。

        为什么需要单独一条路：多标的版本要求「可用标的 ≥ 2」，于是**满仓单票**
        （最集中的状态！）时 VaR 约束会静默返回 None 而不生效 —— 风险最大的
        场景恰好是唯一不检查的场景。单资产不需要协方差，直接 ``z·|w|·σ``。
        """
        closes = self.closes(str(code))
        if closes is None or closes.size < 21:
            return None
        rets = daily_returns(closes)
        if rets.size < 2:
            return None
        vol = float(np.std(rets, ddof=1))
        if not np.isfinite(vol) or vol <= 0:
            return None
        return float(abs(float(weight)) * vol * float(z))

    def cov_and_vols(
        self, codes: Sequence[str], min_days: int = 21,
    ) -> Optional[Tuple[List[str], np.ndarray, np.ndarray]]:
        """返回 ``(可用代码, 协方差矩阵, 各标的日波动率)``，供组合优化器用。

        与 ``var_pct`` 共用一个读盘缓存。可用标的不足 2 只、或历史长度不够时
        返回 None —— 让调用方回退等权，**绝不用残缺样本硬算权重**
        （2 只票 10 天数据算出的风险平价与随机数没有区别）。
        """
        pairs = []
        for c in codes:
            closes = self.closes(str(c))
            if closes is None or closes.size < max(int(min_days), 5):
                continue
            pairs.append((str(c), closes))
        if len(pairs) < 2:
            return None

        usable, rets = returns_matrix(dict(pairs))
        if rets.size == 0 or rets.shape[1] < 2:
            return None
        cov = covariance(rets)
        if cov.size == 0:
            return None
        vols = np.asarray([float(np.std(rets[:, i], ddof=1)) for i in range(rets.shape[1])])
        return usable, cov, vols

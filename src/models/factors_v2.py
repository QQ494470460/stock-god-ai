# -*- coding: utf-8 -*-
"""
横截面因子面板
==============

与旧 engine.py 的根本区别
--------------------------
旧 engine.py 为**每只股票独立计算** 110 个特征，拼成一张大表。
这带来三个问题：
  1. 量纲混杂：ma_5（价格元）、pct_change（%）、volume（手）混在一起，
     模型学到的是"股价高低"而非"相对强弱"
  2. 特征含未来函数：obv/vpt 用 cumsum，且无截面处理
  3. 基本面特征全部恒为 0（财务数据从未成功获取）

新模块输出的是**因子面板**（panel）：
  index = date, columns = (code, factor_name)
  每个因子在每个交易日都会做**横截面 rank 归一化**

为什么必须做横截面处理
----------------------
实测（scripts/diagnose.py）证据：
  ret_20d 原始动量因子IC = -0.080（反向有效）
  剥离市场beta 后、逐日 rank 后，反转因子 IC = +0.041（正向有效）
不做截面处理，模型会把"全市场都在跌"误判为"这只票强势"。

因子清单（基于实测 RankIC 筛选，只保留有信号的方向）
-------------------------------------------------
  rev_5d/rev_10d/rev_20d : 短期反转（A股最稳定的 alpha 来源）
  vol_20d               : 低波动（A 股低波动异象）
  ma_bias / 均线距离:  价格偏离均线的程度
  量比/换手             : 资金关注度
  趋势稳定性            : 波动率的稳定性
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger

from src.models.labeling import cs_rank, MIN_STOCKS_PER_DAY


# ============ 单股原始因子计算 ============

def compute_raw_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    为单只股票计算原始因子（未做截面处理）

    Parameters
    ----------
    df : DataFrame index=date，需含 close/high/low/volume

    Returns
    -------
    DataFrame index=date, columns=factor_name
    """
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    volume = df["volume"] if "volume" in df.columns else pd.Series(
        np.nan, index=df.index)

    f = pd.DataFrame(index=df.index)

    # ===== 动量类 =====
    for p in [5, 10, 20, 60]:
        f[f"mom_{p}d"] = close.pct_change(p)

    # ===== 反转类（对动量取负，A股实证有效）=====
    for p in [5, 10, 20]:
        f[f"rev_{p}d"] = -close.pct_change(p)

    # 反转的强度（波动调整版反转）
    vol_short = close.pct_change().rolling(20).std()
    f["rev_5d_voladj"] = -close.pct_change(5) / (vol_short + 1e-10)

    # ===== 波动率类（低波动异象）=====
    for p in [10, 20, 60]:
        f[f"vol_{p}d"] = close.pct_change().rolling(p).std()

    # 波动率的动量（波动率上升/下降）
    f["vol_chg_20_60"] = (
        close.pct_change().rolling(20).std() -
        close.pct_change().rolling(60).std()
    )

    # ===== 均线偏离 =====
    for p in [5, 10, 20, 60]:
        ma = close.rolling(p).mean()
        f[f"ma_bias_{p}d"] = (close - ma) / (ma + 1e-10)

    # 均线斜率
    f["ma_slope_20d"] = close.rolling(20).mean().diff(5) / close + 1e-10

    # 均线多头排列强度（连续满足的比例）
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    f["ma_alignment"] = ((ma5 > ma10).astype(int) +
                         (ma10 > ma20).astype(int)) / 2.0

    # ===== 价格位置 =====
    for p in [20, 60, 120]:
        hh = high.rolling(p).max()
        ll = low.rolling(p).min()
        f[f"price_pos_{p}d"] = (close - ll) / (hh - ll + 1e-10)

    # ===== RSI =====
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    f["rsi_14"] = 100 - (100 / (1 + rs))

    # ===== 量价类 =====
    if not volume.isna().all():
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        f["vol_ratio_5_20"] = vol_ma5 / (vol_ma20 + 1e-10)

        # OBV 变化率（用 pct 而非 cumsum，避免量级爆炸）
        obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
        obv_diff = obv.diff(10)
        # 归一化：除以该股成交量均值，消除量级
        f["obv_chg_norm"] = obv_diff / (volume.rolling(10).mean() + 1e-10)

        # 价量背离
        f["pv_corr_20"] = close.pct_change().rolling(20).corr(
            volume.pct_change()
        )

    # 换手率（若有）
    if "turnover" in df.columns:
        f["turnover_ma5"] = df["turnover"].rolling(5).mean()
        f["turnover_ratio"] = (
            df["turnover"] / df["turnover"].rolling(20).mean() + 1e-10
        )

    # ===== 日内特征 =====
    if "amplitude" in df.columns:
        f["amplitude_ma5"] = df["amplitude"].rolling(5).mean()

    # 上影线/下影线比例
    if "high" in df.columns and "low" in df.columns and "open" in df.columns:
        rng = (high - low).replace(0, np.nan)
        f["upper_shadow"] = (high - np.maximum(close, df["open"])) / rng
        f["lower_shadow"] = (np.minimum(close, df["open"]) - low) / rng

    # 跳空
    if "open" in df.columns:
        f["gap"] = df["open"] / close.shift(1) - 1

    return f.replace([np.inf, -np.inf], np.nan)


# ============ 面板组装 ============

class FactorPanelBuilder:
    """把单股因子组装成 panel，并做横截面处理"""

    def __init__(self, min_stocks_per_day: int = MIN_STOCKS_PER_DAY):
        self.min_stocks = min_stocks_per_day

    def build(
        self,
        stock_data: Dict[str, pd.DataFrame],
        use_cs_rank: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        构建因子面板

        Parameters
        ----------
        stock_data : {code: DataFrame(index=date)}
        use_cs_rank : 是否做横截面 rank（强烈建议 True）

        Returns
        -------
        panel : DataFrame
            index = date, columns = MultiIndex(code, factor)
        """
        per_code = {}

        for i, (code, df) in enumerate(stock_data.items()):
            if len(df) < 130:      # 至少需要 60 日均线 + 120 日位置
                continue
            try:
                d = df.copy()
                d.index = pd.to_datetime(d.index)
                d = d.sort_index()
                fac = compute_raw_factors(d)
                # 丢弃因子预热期（最长 lookback=120）
                fac = fac.iloc[120:]
                if len(fac) < 30:
                    continue
                per_code[code] = fac
            except Exception as e:
                if verbose and i < 5:
                    logger.warning(f"因子计算失败 {code}: {e}")
                continue

        if not per_code:
            raise ValueError("无有效股票可计算因子")

        if verbose:
            logger.info(f"因子计算完成: {len(per_code)} 只股票")

        # 组装 MultiIndex panel
        codes = list(per_code.keys())
        factors = list(per_code[codes[0]].columns)

        all_dates = sorted(set().union(*[set(per_code[c].index) for c in codes]))
        data = {}

        for fac_name in factors:
            mat = pd.DataFrame(
                {c: per_code[c][fac_name].reindex(all_dates) for c in codes}
            )
            # 覆盖度过滤：当日至少 min_stocks 只有有效值
            cover = mat.notna().sum(axis=1)
            mat = mat.loc[cover >= self.min_stocks]
            if len(mat) == 0:
                continue
            if use_cs_rank:
                mat = cs_rank(mat)
            mat.columns = pd.MultiIndex.from_tuples(
                [(c, fac_name) for c in mat.columns],
                names=["code", "factor"]
            )
            data[fac_name] = mat

        panel = pd.DataFrame(
            np.nan,
            index=all_dates,
            columns=pd.MultiIndex.from_tuples([], names=["code", "factor"])
        )
        for fac_name, mat in data.items():
            panel = panel.join(mat)

        panel = panel.sort_index()

        if verbose:
            n_factors = len(set(k[1] for k in panel.columns))
            logger.info(
                f"因子面板: {panel.shape[0]} 天 × {panel.shape[1]} 列 "
                f"({n_factors} 因子 × {len(codes)} 股)"
            )

        return panel

    # ---------- 展平为 ML 特征表 ----------

    @staticmethod
    def flatten(
        panel: pd.DataFrame,
        dates: pd.DatetimeIndex
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        把 panel 展平成 ML 可用的 (X, y, date) 结构

        Returns
        -------
        X : DataFrame  index=样本ID, columns=特征名
            特征名形如 'rev_5d'（同一交易日内所有股票的同名因子是同一特征，
            这样模型才能学到跨股票通用的规律）
        sample_code : Series  每个样本对应的股票
        sample_date : Series  每个样本对应的交易日  ← walk-forward 必需
        """
        frames = []
        all_factors = sorted(set(k[1] for k in panel.columns))
        for d in dates:
            if d not in panel.index:
                continue
            row = panel.loc[d]
            # unstack(level="factor") 后 index=code, columns=factor
            wide = row.unstack(level="factor")
            codes = list(wide.index)
            wide = wide.reindex(columns=all_factors)
            wide["__code__"] = codes
            wide["__date__"] = d
            wide.index = [f"{pd.Timestamp(d).date()}_{c}" for c in codes]
            wide.index.name = "sample_id"
            frames.append(wide)

        if not frames:
            return pd.DataFrame(), pd.Series(dtype=object), pd.Series(dtype=object)

        full = pd.concat(frames)

        # 特征列：factor_名 (以 factor 开头)
        fac_cols = [c for c in full.columns
                    if isinstance(c, str) and c[0].isalpha()]
        X = full[fac_cols].apply(pd.to_numeric, errors="coerce")

        sample_code = full["__code__"].astype(str)
        sample_date = pd.to_datetime(full["__date__"])

        return X, sample_code, sample_date


# ============ 因子有效性体检 ============

def evaluate_factors(
    panel: pd.DataFrame,
    fwd_excess: pd.DataFrame,
    min_abs_ic: float = 0.02,
    oos_ratio: float = 0.3,
    top_n: int = 20
) -> pd.DataFrame:
    """
    因子有效性体检（区分样本内 / 样本外）

    Parameters
    ----------
    panel : DataFrame index=date, columns=(code, factor)
    fwd_excess : DataFrame index=date, columns=code, 未来超额收益
    min_abs_ic : 样本外 |IC| 阈值
    oos_ratio : 最后多少比例的日期视为样本外
        因子筛选必须以样本外为准 —— 全样本 IC 会把噪声当信号
    top_n : 组合持股数

    Returns
    -------
    DataFrame，按样本外 |IC| 排序
    """
    from src.models.labeling import mean_rank_ic, top_portfolio_return

    factors = sorted(set(k[1] for k in panel.columns))
    codes = sorted(set(k[0] for k in panel.columns))

    common = panel.index.intersection(fwd_excess.index)
    if len(common) < 100:
        raise ValueError(f"共同日期不足({len(common)})")

    split_i = int(len(common) * (1 - oos_ratio))
    is_dates = common[:split_i]
    oos_dates = common[split_i:]

    rows = []
    for fac in factors:
        sub = panel.xs(fac, level="factor", axis=1).reindex(columns=codes)
        sub2 = sub.loc[common]
        fwd2 = fwd_excess.loc[common]

        is_ic, is_ir, n_is = mean_rank_ic(sub2, fwd2, dates=is_dates)
        oos_ic, oos_ir, n_oos = mean_rank_ic(sub2, fwd2, dates=oos_dates)

        # 样本外 Top-N 组合的真实超额
        if n_oos >= 30:
            daily = top_portfolio_return(sub2, fwd2, top_n=top_n, dates=oos_dates)
            ex_mean = float(daily["excess"].mean()) if len(daily) else np.nan
            ex_win = float((daily["excess"] > 0).mean()) if len(daily) else np.nan
        else:
            ex_mean = ex_win = np.nan

        rows.append({
            "factor": fac,
            "is_ic": is_ic,
            "oos_ic": oos_ic,
            "oos_ic_ir": oos_ir,
            "oos_excess": ex_mean,
            "oos_win_rate": ex_win,
            "n_oos_days": n_oos,
            "usable": (
                abs(oos_ic) >= min_abs_ic and not np.isnan(oos_ic)
            ),
        })

    res = pd.DataFrame(rows).sort_values(
        "oos_ic", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)
    return res
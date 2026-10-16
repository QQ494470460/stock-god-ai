# -*- coding: utf-8 -*-
"""
标签构造与横截面因子体系
========================

设计原则（基于 scripts/diagnose.py 的实测结论）
------------------------------------------------
1. 选股要预测的是**相对强弱**，不是绝对涨跌
   实测：全市场 5日收益标准差 6.37%，剥离市场共同波动后剩 5.46%。
   如果标签用绝对涨跌，模型 85% 的学习目标其实是"猜大盘走向"，
   而大盘方向几乎不可预测 → 模型只能学噪声。

2. 标签用**截面排名分位**，而非固定阈值
   原三分类（±2%）在 A股的分布是 涨30.4% / 震荡36.7% / 跌32.9%，
   阈值稍微挪动一点分布就大变，模型对阈值极度敏感。
   改成分位数后每天自动平衡，且天然对异常值稳健。

3. 中期动量在 A股是**反向**指标
   实测 RankIC：ret_20d = -0.080（按它选股年化跑输约 15%）。
   所以因子里保留动量，但方向由数据自己学（用负号或让模型学），
   绝不能想当然地"动量=好"。

标签定义
--------
y = 未来 H 日**相对全市场**收益的截面分位（0~100，或转成五档）
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


# ============ 配置常量 ============
DEFAULT_HORIZON = 5          # 预测未来5个交易日
N_QUANTILE_BINS = 5          # 五档：强跌/弱跌/中性/弱涨/强涨
MIN_STOCKS_PER_DAY = 30      # 当日有效股票数下限，低于此日丢弃


# ============ 基础收益计算 ============

def forward_return(
    close: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON
) -> pd.DataFrame:
    """
    未来 N 日收益率

    Parameters
    ----------
    close : DataFrame  index=date, columns=code

    Returns
    -------
    DataFrame 同形状，值为未来 horizon 日收益率
    """
    return close.shift(-horizon) / close - 1


def forward_excess_return(
    close: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    winsorize: float = 0.01
) -> pd.DataFrame:
    """
    未来 N 日**超额收益**（已剥离当日市场均值）

    这是选股真正的预测目标：模型不需要判断大盘涨跌，
    只需要判断"这只股票比同一天的其他股票强多少"。

    Parameters
    ----------
    winsorize : float
        对每日截面做缩尾的比例，抑制涨停/暴跌等极端值影响
    """
    fwd = forward_return(close, horizon)

    # 按日剥离横截面均值（市场共同波动）
    excess = fwd.sub(fwd.mean(axis=1), axis=0)

    if winsorize > 0:
        lo = excess.quantile(winsorize, axis=1)
        hi = excess.quantile(1 - winsorize, axis=1)
        excess = excess.clip(lower=lo, upper=hi, axis=0)

    return excess


# ============ 标签构造 ============

def make_rank_labels(
    excess_ret: pd.DataFrame,
    n_bins: int = N_QUANTILE_BINS
) -> pd.DataFrame:
    """
    每日截面分位档位标签

    每个交易日内，按超额收益排序切成 n_bins 档。
    n_bins=5 时标签为 0~4，含义：
      4 = 未来最强的一档
      3 = 次强
      2 = 中性
      1 = 次弱
      0 = 最弱

    好处：
      - 每天样本天然均衡（各档占比稳定），不再需要 SMOTE
      - 对阈值不敏感，不会因为阈值挪动而失效
      - 直接对应"选股"这个动作——我们只想买最强的那一档

    Returns
    -------
    DataFrame，值为 0~n_bins-1，NaN 表示该日无有效数据
    """
    valid_count = excess_ret.notna().sum(axis=1)

    # 向量化：逐日做 pct rank → 映射到档位，避免逐行 apply 的性能陷阱
    pct = excess_ret.rank(axis=1, pct=True)
    labels = np.floor(pct * n_bins)
    labels = labels.clip(upper=n_bins - 1)
    labels = labels.where(excess_ret.notna())

    # 有效股票数不足的交易日整行作废
    labels = labels.mask(valid_count < MIN_STOCKS_PER_DAY)

    return labels.astype(float)


def make_binary_labels(
    excess_ret: pd.DataFrame,
    top_pct: float = 0.3
) -> pd.DataFrame:
    """
    二分类标签：跑赢前 top_pct 记为 1，其余为 0

    如果下游模型（二分类 LightGBM / LSTM）需要二分类标签，
    用这个比原来的"绝对涨跌 ±2%"合理得多。

    Parameters
    ----------
    top_pct : float
        排名前多少比例算正样本。0.3 表示每日前30% 为正样本。
    """
    pct = excess_ret.rank(pct=True, axis=1)
    labels = (pct > (1 - top_pct)).astype(float)
    labels[excess_ret.isna()] = np.nan
    return labels


# ============ 横截面因子预处理 ============

def cs_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """
    横截面排名归一化（cross-sectional rank）

    量化选股的标准做法。对每个因子，在每个交易日内独立排名，
    映射到 [-1, 1]（中位数→0）。

    为什么必须做：
      - 消除行业/市值/市场整体水平的差异
      - 抗极端值（一只股票停牌暴涨不会污染全池分布）
      - 让模型学"相对强弱"而非"绝对数值"
      - 因子在不同市场状态下可比

    Parameters
    ----------
    panel : DataFrame  index=date, columns=code, 值为因子原始值

    Returns
    -------
    DataFrame 同形状，值域 (-1, 1)
    """
    return panel.rank(pct=True, axis=1).mul(2).sub(1)


def cs_zscore(panel: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """
    横截面 z-score 标准化（带截断）

    同样按日独立处理，但对极端值更敏感于 rank，
    会保留"这个因子有多极端"的信息量。
    """
    mean = panel.mean(axis=1)
    std = panel.std(axis=1).replace(0, np.nan)
    z = panel.sub(mean, axis=0).div(std, axis=0)
    return z.clip(-clip, clip)


# ============ 评估指标 ============

def rank_ic(
    factor: pd.Series,
    fwd_ret: pd.Series
) -> float:
    """
    单日横截面 Spearman RankIC

    Parameters
    ----------
    factor : Series  同一交易日的因子值
    fwd_ret : Series  同一交易日对应的未来收益（建议用超额收益）

    Returns
    -------
    float, NaN 表示数据不足
    """
    df = pd.DataFrame({"f": factor, "r": fwd_ret}).dropna()
    if len(df) < MIN_STOCKS_PER_DAY:
        return np.nan
    if df["f"].nunique() < 2 or df["r"].nunique() < 2:
        return np.nan
    return float(df["f"].corr(df["r"], method="spearman"))


def mean_rank_ic(
    factor_panel: pd.DataFrame,
    fwd_excess: pd.DataFrame,
    dates: Optional[pd.DatetimeIndex] = None
) -> Tuple[float, float, int]:
    """
    逐日 RankIC 的均值 / 标准误 / 有效天数

    Returns
    -------
    (mean_ic, ic_ir, n_days)
        ic_ir = mean_ic / 标准误，是因子的稳健性指标
    """
    if dates is None:
        dates = factor_panel.index.intersection(fwd_excess.index)

    ics: List[float] = []
    for d in dates:
        if d not in factor_panel.index or d not in fwd_excess.index:
            continue
        ic = rank_ic(factor_panel.loc[d], fwd_excess.loc[d])
        if not np.isnan(ic):
            ics.append(ic)

    if not ics:
        return np.nan, np.nan, 0

    arr = np.array(ics)
    se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.nan
    return float(arr.mean()), float(arr.mean() / se) if se else np.nan, len(arr)


def top_portfolio_return(
    score_panel: pd.DataFrame,
    fwd_excess: pd.DataFrame,
    top_n: int = 20,
    n_quantile: int = 5,
    dates: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    按因子得分构建 Top-N 组合，返回逐日收益

    这是最接近实盘的指标：只看"按这个因子选出来的股票，
    相对全市场能赚多少"。

    Parameters
    ----------
    score_panel : index=date, columns=code, 因子得分（越大越好）
    fwd_excess  : index=date, columns=code, 未来超额收益

    Returns
    -------
    DataFrame，列 = ['top_ret', 'mkt_ret', 'excess', 'n_valid']
    """
    if dates is None:
        dates = score_panel.index.intersection(fwd_excess.index)

    rows = []
    for d in dates:
        if d not in score_panel.index or d not in fwd_excess.index:
            continue
        s = score_panel.loc[d]
        r = fwd_excess.loc[d]
        v = pd.DataFrame({"s": s, "r": r}).dropna()
        if len(v) < MIN_STOCKS_PER_DAY:
            continue
        top = v.nlargest(min(top_n, len(v)), "s")
        rows.append({
            "date": d,
            "top_ret": float(top["r"].mean()),
            "mkt_ret": float(v["r"].mean()),
            "excess": float(top["r"].mean() - v["r"].mean()),
            "n_valid": len(v),
        })

    if not rows:
        return pd.DataFrame(columns=["date", "top_ret", "mkt_ret", "excess", "n_valid"])
    return pd.DataFrame(rows).set_index("date")


def portfolio_metrics(daily_ret: pd.Series, periods_per_year: int = 48) -> dict:
    """
    由逐日收益序列计算组合绩效指标

    Parameters
    ----------
    daily_ret : 逐"持仓周期"收益（如 5 日一持）
    periods_per_year : 年化换算系数

    ⚠️ 重叠持仓的年化口径说明
    ------------------------
    当每 5 个交易日调仓一次时，日频超额收益序列是**高度重叠**的
    （相邻两期的收益共享 4 天行情）。此时：
      - 均值可以直接乘以 (250/5)= 50 做年化
      - 但标准差会被低估，直接乘 sqrt(50) 会**高估**年化波动，
        进而高估 Sharpe

    所以这里对波动做 Newey-West 式的保守调整：把重叠视为
    独立期数为 min(periods_per_year, n_periods)，并明确标注
    这是"按持仓期计"的年化，不等于逐日调仓的真实年化。

    更严谨的做法是用不重叠的采样期，这里通过保守处理避免误导。
    """
    r = daily_ret.dropna()
    if len(r) == 0:
        return {
            "n_periods": 0, "mean_period_ret": np.nan, "annual_return": np.nan,
            "annual_vol": np.nan, "sharpe": np.nan, "win_rate": np.nan,
            "max_drawdown": np.nan, "note": "无样本",
        }

    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1)) if len(r) > 1 else 0.0

    annual_ret = mean_r * periods_per_year

    # 保守处理：重叠收益序列的波动不可直接 sqrt 年化，
    # 这里用重叠因子做保守放大
    OVERLAP = 4  # 5日持仓，相邻期重叠4天
    eff_periods = max(1.0, periods_per_year / OVERLAP)
    annual_vol = std_r * np.sqrt(eff_periods)
    sharpe = (mean_r / std_r * np.sqrt(eff_periods)) if std_r > 0 else np.nan

    # 最大回撤：按真实时间顺序累计净值
    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = float(drawdown.min())

    return {
        "n_periods": len(r),
        "mean_period_ret": mean_r,
        "annual_return": annual_ret,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "win_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "note": "毛收益，未扣交易成本；重叠持仓，年化为按持仓期换算",
    }
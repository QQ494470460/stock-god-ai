# -*- coding: utf-8 -*-
"""
信号稳定化层（Signal Fusion / Stabilization Layer）
===================================================

背景
----
单日模型评分噪声很大（信息比率 ≈ 0.35），却直接驱动每日选股。本模块在**完全可
选（opt-in）、默认关闭、零行为改变**的前提下，提供一组纯函数/类，对 *已有* 的
每日评分做后处理，目的是在不重训模型、不改动现有推理/交易链路（predictor.py /
trading/loop.py / executor.py / broker.py / paper.py / config/*.py 一律不碰）的
情况下，**降低单日评分的噪声与板块偏差**。

所有函数均为：
  · 确定性（deterministic）
  · 无副作用（side-effect free）
  · 无网络 / 无文件 IO（pure in-memory）
  · 严格无前视（no lookahead）：跨日运算只使用 <= 目标日期的历史/当日数据

输入契约（score DataFrame）
---------------------------
约定每张「评分表」至少含列：
    code   : 股票代码 (str)
    score  : 模型综合评分 (float, 0~100)
可选列：
    name   : 股票名称 (str)

公开 API
--------
  · cross_day_ema(history, alpha=0.3, as_of=None) -> pd.DataFrame
  · confidence_weighted_fusion(legs, weights=None, confidences=None) -> pd.DataFrame
  · reneutralize(scores, industry_map, by="sector") -> pd.DataFrame
  · class SignalFusion(enabled=False, ema_alpha=0.3, neutralize=True)
        .fuse(today_scores, history_scores, industry_map=None) -> pd.DataFrame

设计原则：默认 enabled=False 时 SignalFusion.fuse 原样返回输入，相当于零改动。
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
_DateLike = Union[str, date, datetime, "pd.Timestamp"]


def _norm_key(k: _DateLike):
    """把日期键归一化为可排序的 pd.Timestamp（仅用于排序，绝不做时区/前视变换）。"""
    return pd.Timestamp(k)


def _ema_values(values: Sequence[Optional[float]], alpha: float) -> List[float]:
    """对一条标量序列做指数平滑，遇到缺失值（None/NaN）则沿用上一期水平。

    使用递归定义：ema_t = alpha * x_t + (1 - alpha) * ema_{t-1}，
    ema_0 = x_0。严格只用过去/当前值，无任何前视。
    """
    out: List[float] = []
    prev: Optional[float] = None
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(prev if prev is not None else 0.0)
            continue
        if prev is None:
            prev = float(v)
        else:
            prev = alpha * float(v) + (1.0 - alpha) * prev
        out.append(prev)
    return out


# --------------------------------------------------------------------------- #
# 1) 跨日指数平滑
# --------------------------------------------------------------------------- #
def cross_day_ema(
    history: Dict[_DateLike, pd.DataFrame],
    alpha: float = 0.3,
    as_of: Optional[_DateLike] = None,
) -> pd.DataFrame:
    """对每只股票的多日评分做指数平滑，输出目标日期的平滑评分。

    Parameters
    ----------
    history : dict[date, DataFrame]
        键为交易日（str/date/datetime/pd.Timestamp 均可），值为当日评分表
        （含 ``code``、``score``，可选 ``name``）。
    alpha : float, default 0.3
        平滑系数（越大越看重最近一日；0.3 ≈ 约 3 日半衰期）。
    as_of : date-like, optional
        目标日期。默认取 ``history`` 中**最晚**的日期。返回的平滑值只依赖
        ``<= as_of`` 的历史，绝不前视。

    Returns
    -------
    pd.DataFrame
        列：``code``、``name``（若原始表提供）、``score``（目标日期的平滑值）。
        仅包含目标日期评分表中出现的股票；其历史缺失处沿用最近已知水平。
    """
    if not history:
        return pd.DataFrame(columns=["code", "name", "score"])

    # 按日期排序，构建 (sorted_dates, per-code value序列)
    items = sorted(history.items(), key=lambda kv: _norm_key(kv[0]))
    dates = [k for k, _ in items]
    frames = [v for _, v in items]

    as_of_key = _norm_key(as_of) if as_of is not None else _norm_key(dates[-1])
    # 只保留 <= as_of 的日期（无前视保证）
    keep = [(d, f) for d, f in zip(dates, frames) if _norm_key(d) <= as_of_key]
    if not keep:
        return pd.DataFrame(columns=["code", "name", "score"])

    # 目标日期的评分表（用于决定输出 universe 与 name）
    last_d, last_f = keep[-1]
    last_f = last_f if isinstance(last_f, pd.DataFrame) else pd.DataFrame(last_f)
    if "code" not in last_f.columns or "score" not in last_f.columns:
        raise KeyError("每个评分表必须包含 'code' 与 'score' 列")

    codes = list(last_f["code"].astype(str))

    # 收集每只股票在 keep 序列中的 score 轨迹
    # 用 dict-of-lists 累积，按 keep 顺序
    traj: Dict[str, List[Optional[float]]] = {c: [] for c in codes}
    # 先按 keep 顺序逐日填充（保持时间顺序）
    for d, f in keep:
        f = f if isinstance(f, pd.DataFrame) else pd.DataFrame(f)
        if "code" not in f.columns or "score" not in f.columns:
            continue
        s = f.set_index(f["code"].astype(str))["score"]
        for c in codes:
            traj[c].append(float(s[c]) if c in s.index and pd.notna(s[c]) else None)

    rows = []
    name_map = dict(zip(last_f["code"].astype(str),
                        last_f["name"] if "name" in last_f.columns
                        else [None] * len(last_f)))
    for c in codes:
        ema = _ema_values(traj[c], alpha)
        rows.append({
            "code": c,
            "name": name_map.get(c),
            "score": ema[-1] if ema else 0.0,
        })
    return pd.DataFrame(rows, columns=["code", "name", "score"])


# --------------------------------------------------------------------------- #
# 2) 置信度加权多腿融合
# --------------------------------------------------------------------------- #
def confidence_weighted_fusion(
    legs: List[pd.DataFrame],
    weights: Optional[List[float]] = None,
    confidences: Optional[List[float]] = None,
) -> pd.DataFrame:
    """多模型腿（leg）置信度加权融合。

    适用于 xgboost / alpha_xgboost / lstm 等多腿评分的横向融合。权重优先级：
        ``confidences``（如近期 RankIC）> ``weights`` > 等权。

    Parameters
    ----------
    legs : list[DataFrame]
        各腿评分表，每张含 ``code``、``score``，可选 ``name``。
        缺失腿（None / 空表）会被安全丢弃，绝不抛错。
    weights : list[float], optional
        显式权重；与上述优先级共存时，``confidences`` 优先。
    confidences : list[float], optional
        每条腿的置信度（如近期 RankIC / 样本外 IC）。归一化后作为权重；
        若存在负值或全 0，则回退等权。

    Returns
    -------
    pd.DataFrame
        列：``code``、``name``、``score``（融合评分）。
        缺失某腿的股票：该腿贡献计 0，按其余腿有效权重求和。
    """
    # 与对应 weights / confidences 一并过滤，保证三者长度对齐（缺失腿安全降级）
    paired = [
        (l, (weights[i] if weights is not None else 1.0),
         (confidences[i] if confidences is not None else None))
        for i, l in enumerate(legs)
        if l is not None and not (isinstance(l, pd.DataFrame) and l.empty)
    ]
    if not paired:
        return pd.DataFrame(columns=["code", "name", "score"])
    clean = [p[0] for p in paired]
    w_meta = [p[1] for p in paired]
    c_meta = [p[2] for p in paired]
    n = len(clean)

    if any(c is not None for c in c_meta):
        conf = np.asarray(
            [float(max(c, 0.0)) if c is not None else 0.0 for c in c_meta],
            dtype=float,
        )
        if conf.size != n or conf.sum() <= 0:
            w = np.ones(n) / n
        else:
            w = conf / conf.sum()
    elif any(wt is not None for wt in w_meta):
        wt = np.asarray(
            [float(x) if x is not None else 1.0 for x in w_meta], dtype=float
        )
        if wt.sum() <= 0:
            w = np.ones(n) / n
        else:
            w = wt / wt.sum()
    else:
        w = np.ones(n) / n

    # 对齐到所有出现过的 code 的并集
    all_codes = pd.unique(pd.concat([l["code"] for l in clean]).astype(str))
    mat = pd.DataFrame(index=all_codes)
    name_map: Dict[str, str] = {}
    for i, l in enumerate(clean):
        l = l if isinstance(l, pd.DataFrame) else pd.DataFrame(l)
        s = l.set_index(l["code"].astype(str))["score"]
        mat[i] = s.reindex(all_codes)
        if "name" in l.columns:
            nm = l.set_index(l["code"].astype(str))["name"]
            for c in all_codes:
                if c in nm.index and pd.notna(nm[c]) and c not in name_map:
                    name_map[c] = nm[c]

    # 加权求和：缺失腿按 0 贡献，并只按该股票可用的腿重新归一（避免被缺失腿稀释）
    wmat = mat.notna().astype(float).mul(w, axis=1)
    denom = wmat.sum(axis=1).replace(0.0, np.nan)
    fused = mat.mul(w, axis=1).sum(axis=1) / denom

    out = pd.DataFrame({
        "code": all_codes,
        "name": [name_map.get(c) for c in all_codes],
        "score": fused.values,
    })
    out["score"] = out["score"].astype(float)
    return out


# --------------------------------------------------------------------------- #
# 3) 截面行业中性化（去行业均值，消除板块 beta 泄漏到排名）
# --------------------------------------------------------------------------- #
def reneutralize(
    scores: pd.DataFrame,
    industry_map: Optional[Dict[str, str]] = None,
    by: str = "sector",
) -> pd.DataFrame:
    """对评分做横截面行业中性化：减去各行业（分组）内的均值。

    注意：与 signal_quality.neutralize_cross_section 不同，本函数直接把
    分组均值减到 0（不强制加回总均值），目的是让排名不再受「行业整体水位」影响。

    Parameters
    ----------
    scores : pd.DataFrame
        含 ``code``、``score``，可选 ``name``。
    industry_map : dict | None
        ``code -> 行业名`` 的映射。若为 None / 空，则原样返回。
    by : str, default "sector"
        分组来源：若该列已存在于 ``scores`` 中则直接用该列；否则用
        ``code`` 经 ``industry_map`` 映射得到分组。

    Returns
    -------
    pd.DataFrame
        同输入结构，``score`` 已减去所在分组均值。
    """
    if scores is None or not isinstance(scores, pd.DataFrame) or scores.empty:
        return scores if scores is not None else pd.DataFrame(columns=["code", "name", "score"])
    if industry_map is None or len(industry_map) == 0:
        return scores.copy()

    df = scores.copy()
    if by in df.columns:
        grp = df[by].astype(str)
    else:
        grp = df["code"].astype(str).map(industry_map)
        grp = grp.fillna("_UNKNOWN_")

    means = df.groupby(grp)["score"].transform("mean")
    df["score"] = df["score"] - means
    return df


# --------------------------------------------------------------------------- #
# 4) 便捷封装类
# --------------------------------------------------------------------------- #
class SignalFusion:
    """信号融合便捷封装。

    默认 ``enabled=False``：``fuse`` 原样返回 ``today_scores``，对调用方零行为改变。
    开启后链式执行：跨日指数平滑 ->（可选）行业中性化。

    调用方职责：维护滚动的 ``history_scores``（dict[date, DataFrame]），
    其**最晚键即为当日**，值等于当日 ``today_scores``。本类不做任何状态修改。

    Examples
    --------
    >>> sf = SignalFusion(enabled=True, ema_alpha=0.3, neutralize=True)
    >>> fused = sf.fuse(today_df, history, industry_map)
    """

    def __init__(
        self,
        enabled: bool = False,
        ema_alpha: float = 0.3,
        neutralize: bool = True,
    ):
        self.enabled = bool(enabled)
        self.ema_alpha = float(ema_alpha)
        self.neutralize = bool(neutralize)

    def fuse(
        self,
        today_scores: pd.DataFrame,
        history_scores: Optional[Dict[_DateLike, pd.DataFrame]] = None,
        industry_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """稳定化当日评分。

        Parameters
        ----------
        today_scores : pd.DataFrame
            当日评分表（``code``、``score``，可选 ``name``）。
        history_scores : dict[date, DataFrame], optional
            历史评分缓冲；其最晚键应为当日（等于 ``today_scores``）。
            为空时直接返回 ``today_scores``（即便 enabled=True）。
        industry_map : dict | None, optional
            行业映射，仅当 ``neutralize=True`` 时生效。

        Returns
        -------
        pd.DataFrame
            稳定化后的评分；``enabled=False`` 时原样返回输入。
        """
        if not self.enabled:
            return today_scores
        if history_scores is None or len(history_scores) == 0:
            return today_scores

        # 1) 跨日指数平滑（仅用 <= 当日的历史，无前视）
        smoothed = cross_day_ema(history_scores, alpha=self.ema_alpha)
        if smoothed.empty:
            return today_scores

        # 2) 截面行业中性化
        if self.neutralize and industry_map:
            smoothed = reneutralize(smoothed, industry_map, by="sector")
        return smoothed

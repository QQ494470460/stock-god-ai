# -*- coding: utf-8 -*-
"""信号质量增强（截面层）。

集中放「不重训模型、只对预测评分做后处理」的纯函数，便于离线单测与回测复用：

  - neutralize_cross_section：行业中性化（去均值），消除模型对个别行业的系统性
    偏高/偏低造成的排名偏差，降低单日 IC 的板块噪声，提升每日 IC 稳定性
    （Grinold-Kahn IR = mean/std）。

设计要点
--------
- 全部为纯函数（输入 Series / dict，输出 Series），不触碰任何全局状态，便于测试。
- 行业中性化的标准做法是「按行业去均值、再加回总均值」：组内排序完全保留，
  只把「行业整体水位」从评分里剔除。这样选股不再被「恰好全仓某个行业」带偏。
- 对只含 1 只股票的行业（或无法映射到行业的个股），不强拉平到总均值
  （否则该股票的信号会被直接抹成总均值），而是保留其原分，仅随整体去均值。
- ⚠️ 行业映射缺失的个股统一归入 ``_UNKNOWN_`` 伪组，该伪组**无论数量多少都按小组
  处理、保留原分**；绝不把它当成「一个行业」去做组内去均值（否则 ≥2 只未映射股
  会被合并平移，等于用数据缺失惩罚分散化）。
- 无行业映射（map 为空 / 缺失）时直接返回原序列，绝不抛错、不影响排名。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# 行业组最小样本：低于该值的行业视为「单只/极少」，不做组内去均值，保留原分。
_MIN_GROUP = 2

# 无法映射到行业的个股归入该虚拟组（与真实行业名碰撞概率极低）。
_UNKNOWN = "_UNKNOWN_"


def neutralize_cross_section(
    scores: pd.Series,
    industry_map: Optional[Dict[str, str]],
    min_group: int = _MIN_GROUP,
) -> pd.Series:
    """对评分做行业中性化。

    Parameters
    ----------
    scores : pd.Series
        索引为股票代码，值为模型评分（0~100）。
    industry_map : dict | None
        code -> 行业名 的映射；为 None / 空时直接返回原序列。
    min_group : int
        行业组内最少样本数；低于此值的组不做组内去均值（保留原分，仅随整体去均值）。

    Returns
    -------
    pd.Series
        中性化后的评分（索引同 scores）。总均值水平保持不变，组内排序不变，
        仅「行业整体水位」被剔除。
    """
    if scores is None or len(scores) == 0:
        return scores if scores is not None else pd.Series(dtype=float)
    if not industry_map:
        return scores

    grand = float(scores.mean())
    ind = pd.Series({c: str(industry_map.get(c, _UNKNOWN)) for c in scores.index})

    grp_mean = scores.groupby(ind, group_keys=False).transform("mean")
    grp_count = scores.groupby(ind, group_keys=False).transform("count")

    resid = scores.copy()
    # ⚠️ 关键：_UNKNOWN_ 伪组**永远**按「小组」处理（保留原分），绝不参与组内去均值。
    # 否则一旦 ≥2 只未映射股落入同一个伪组，它们会被当成一个「行业」整体平移，
    # 等于用「数据缺失」去惩罚分散化 —— 这正是排名偏差的沉默来源。
    # 只有真实行业组且样本数 ≥ min_group 才做组内去均值。
    is_unknown = ind == _UNKNOWN
    big = (~is_unknown) & (grp_count >= min_group)
    # 大组：score - 组均值 + 总均值（剔除行业水位，保留组内序与总水平）
    resid[big] = scores[big] - grp_mean[big] + grand
    # 小组 / 未知组：保留原分（未知组缺失行业信息，强行平移只会引入伪信号）
    resid[~big] = scores[~big]
    return resid


def cross_sectional_zscore(
    scores: pd.Series,
    clip: float = 3.0,
) -> pd.Series:
    """把评分转成横截面 z-score（均值 0、标准差 1），用于下游拼接组合权重时对齐量纲。

    可选 clip 防止极端值主导。不改变排名方向。
    """
    if scores is None or len(scores) == 0:
        return scores if scores is not None else pd.Series(dtype=float)
    mean = float(scores.mean())
    std = float(scores.std(ddof=0))
    if std <= 1e-9:
        return pd.Series(0.0, index=scores.index)
    z = (scores - mean) / std
    if clip and clip > 0:
        z = z.clip(-clip, clip)
    return z

# -*- coding: utf-8 -*-
"""组合波动率目标：温和的敞口叠加层，**只降不升**。

定位
----
这是「波动率目标（vol targeting）」在组合层面的一个薄封装：实测组合波动率
高于目标时，按 ``target / realized`` 线性缩小新买入规模。它不是恐慌开关：

* ``ceil_`` 默认 1.0 —— **只可能向下调整**，绝不在低波动时加杠杆。回测里
  「低波动加杠杆」这条腿在 A 股几乎从不赚钱，却会放大尾部风险。
* ``floor`` 默认 0.5 —— 再高的实测波动也保留一半买入力，与
  :mod:`src.trading.risk.derisk` 的非顺周期地板保持一致。
* 实测波动 <= 0 或数据不可用 → 返回 1.0（fail-open），而不是「数据坏了就
  停止买入」。

与 derisk 的关系
----------------
两者是**正交**的两条腿：derisk 看回撤（路径风险），本模块看波动（尺度风险）。
同时启用时用 :func:`blend_scales` 做**算术**平均——注意不能用几何平均，
几何会把两个地板相乘（0.6 × 0.5 = 0.3），直接击穿「地板 >= 0.5」的硬约束。
"""

from __future__ import annotations

import math

__all__ = ["portfolio_vol_target_scale", "blend_scales"]

MIN_FLOOR = 0.5


def _to_float(value, default=None):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def portfolio_vol_target_scale(
    realized_portfolio_vol: float,
    target_vol: float = 0.15,
    floor: float = 0.5,
    ceil_: float = 1.0,
) -> float:
    """``target / realized`` 缩放系数，夹在 ``[floor, ceil_]`` 内。

    Parameters
    ----------
    realized_portfolio_vol
        实测组合年化波动率（0.30 表示 30%）。
    target_vol
        目标年化波动率，默认 0.15。
    floor
        系数下限，默认 0.5（**不会低于 0.5**）。
    ceil_
        系数上限，默认 1.0（即只缩不放）。

    Returns
    -------
    缩放系数。实测波动 <= 0 / 参数非法 → ``1.0``（fail-open）。
    """
    realized = _to_float(realized_portfolio_vol, None)
    target = _to_float(target_vol, None)
    if realized is None or realized <= 0.0:
        return 1.0
    if target is None or target <= 0.0:
        return 1.0

    lo = _to_float(floor, MIN_FLOOR)
    if lo is None:
        lo = MIN_FLOOR
    lo = max(MIN_FLOOR, min(lo, 1.0))

    hi = _to_float(ceil_, 1.0)
    if hi is None:
        hi = 1.0
    hi = max(lo, min(hi, 1.0))

    ratio = target / realized
    return max(lo, min(ratio, hi))


def blend_scales(a: float, b: float, weight_b: float = 0.5) -> float:
    """两个敞口系数的**算术**加权平均，夹在 ``[0, 1]``。

    ``result = a * (1 - weight_b) + b * weight_b``

    为什么是算术而不是几何：几何平均会让两个各自合规的地板相乘后跌到
    0.5 以下（0.6 × 0.5 = 0.3），违背「任何组合下都保留至少一半买入力」的
    非顺周期约束。算术平均的保底性质与单调性都更符合要求。

    非法输入（None / NaN / 非数值）按 ``1.0`` 处理，即「该腿不干预」。
    """
    fa = _to_float(a, 1.0)
    fb = _to_float(b, 1.0)
    if fa is None:
        fa = 1.0
    if fb is None:
        fb = 1.0
    fa = max(0.0, min(fa, 1.0))
    fb = max(0.0, min(fb, 1.0))

    w = _to_float(weight_b, 0.5)
    if w is None:
        w = 0.5
    w = max(0.0, min(w, 1.0))

    return max(0.0, min(fa * (1.0 - w) + fb * w, 1.0))

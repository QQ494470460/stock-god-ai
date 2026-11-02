# -*- coding: utf-8 -*-
"""波动率估计：给「波动率目标仓位」喂真实输入，而不是让它退化成固定比例。

为什么要有这个文件
------------------
``src/trading/sizing.py`` 的 :func:`volatility_target_fraction` 已经实现
``w_i = target / (vol_i * sqrt(N))``，但没有生产调用方稳定地提供
``annualized_vol``。缺省值一旦写死成常数（例如 0.30），该函数就退化成
「固定比例 × 常数」：高波动票和低波动票拿一样的仓位，波动率目标的全部
意义丢失。

本模块只做一件事：把「收盘价序列」变成「年化波动率」，并且**永不抛异常**
（fail-open 返回 ``None``，调用方自然回落到固定比例仓位）。

设计约束
--------
* 纯函数 + 一个闭包工厂，无网络、无磁盘 IO、导入期无副作用。
* 全部使用标准库；结果对同一输入完全确定。
* 无效/非正价格被跳过，而不是让整个估计崩掉——我们宁可要一个略嘈杂的
  波动率，也不要回测因为一条脏数据而中断。
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

__all__ = [
    "ewma_vol_annualized",
    "realized_vol_from_prices",
    "make_vol_provider",
]

_DEFAULT_PPY = 252.0


def _finite_floats(values) -> list:
    """把任意可迭代对象转成有限浮点列表，跳过 None / NaN / inf / 不可转换项。"""
    out = []
    if values is None:
        return out
    try:
        iterator = iter(values)
    except TypeError:
        return out
    for v in iterator:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _periods_per_year(periods_per_year) -> float:
    try:
        ppy = float(periods_per_year)
    except (TypeError, ValueError):
        return _DEFAULT_PPY
    if not math.isfinite(ppy) or ppy <= 0:
        return _DEFAULT_PPY
    return ppy


def ewma_vol_annualized(
    returns: Sequence[float],
    span: int = 20,
    periods_per_year: int = 252,
) -> float:
    """指数加权（EWMA）收益率标准差，年化。

    权重取 ``(1 - alpha) ** k``，``alpha = 2 / (span + 1)``，最近一期权重最大。
    采用**加权去均值**的方差（不是 ``sum(w * r^2)``）：否则一条恒定的正收益率
    序列会被算成 ``|r| * sqrt(ppy)`` 这种「零波动却有波动率」的假信号。

    Parameters
    ----------
    returns
        逐期简单/对数收益率序列（小数，不是百分数）。非有限值会被跳过。
    span
        EWMA 衰减跨度，等价于 pandas ``ewm(span=...)``。
    periods_per_year
        年化因子。A 股日频取 252。

    Returns
    -------
    年化波动率（0.28 表示 28%）。样本 < 2 个点时返回 ``0.0``。
    """
    r = _finite_floats(returns)
    if len(r) < 2:
        return 0.0

    try:
        sp = float(span)
    except (TypeError, ValueError):
        sp = 20.0
    if not math.isfinite(sp) or sp < 1.0:
        sp = 1.0

    ppy = _periods_per_year(periods_per_year)

    alpha = 2.0 / (sp + 1.0)
    decay = 1.0 - alpha

    # 最近一期的权重为 1.0，越早衰减越多。
    newest_first = list(reversed(r))
    weights = [decay ** k for k in range(len(newest_first))]
    wsum = sum(weights)
    if wsum <= 0:
        return 0.0

    mean = sum(w * x for w, x in zip(weights, newest_first)) / wsum
    var = sum(w * (x - mean) ** 2 for w, x in zip(weights, newest_first)) / wsum
    if var <= 0.0:
        return 0.0
    return math.sqrt(var) * math.sqrt(ppy)


def realized_vol_from_prices(
    closes: Sequence[float],
    window: int = 20,
    periods_per_year: int = 252,
) -> float:
    """用最近 ``window + 1`` 个收盘价算对数收益率标准差，年化。

    * 价格 <= 0 或非有限的点被**跳过**（脏数据不应让整条序列作废）。
    * 有效点数 < 3 时返回 ``0.0``（2 个点只能得到 1 个收益率，样本方差无定义）。

    Returns
    -------
    年化波动率（0.28 表示 28%）。
    """
    try:
        w = int(window)
    except (TypeError, ValueError):
        w = 20
    if w < 2:
        w = 2

    ppy = _periods_per_year(periods_per_year)

    prices = [p for p in _finite_floats(closes) if p > 0.0]
    if len(prices) < 3:
        return 0.0

    tail = prices[-(w + 1):]
    if len(tail) < 3:
        return 0.0

    rets = [
        math.log(tail[i] / tail[i - 1]) for i in range(1, len(tail))
    ]
    if len(rets) < 2:
        return 0.0

    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    if var <= 0.0:
        return 0.0
    return math.sqrt(var) * math.sqrt(ppy)


def make_vol_provider(
    price_history_fn: Callable[[str], Sequence[float]],
    window: int = 20,
) -> Callable[[str], Optional[float]]:
    """把「按代码取收盘价序列」的函数包装成波动率 provider。

    返回的 ``provider(code)`` 供 ``volatility_target_fraction(annualized_vol=...)``
    直接使用：

    * 价格历史取不到 / 抛异常 / 有效点不足 / 估计结果非正 → ``None``
      （**fail-open**：调用方回落到固定比例仓位，而不是把仓位压到 0）。
    * 正常情况返回年化波动率（正浮点数）。

    返回 ``None`` 而不是 ``0.0`` 很关键：``volatility_target_fraction`` 对
    ``vol <= 0`` 的处理是返回 ``0.0`` 仓位，那是「波动率数据坏了就清仓」的
    顺周期行为，正是本项目回测证伪过的方向。
    """
    try:
        w = int(window)
    except (TypeError, ValueError):
        w = 20
    if w < 2:
        w = 2

    def provider(code: str) -> Optional[float]:
        try:
            hist = price_history_fn(code)
        except Exception:  # noqa: BLE001 - provider 必须 fail-open
            return None
        if hist is None:
            return None
        try:
            if len(hist) < 3:
                return None
        except TypeError:
            return None

        vol = realized_vol_from_prices(hist, window=w)
        if not math.isfinite(vol) or vol <= 0.0:
            return None
        return vol

    provider.__name__ = "vol_provider"
    provider.__qualname__ = "vol_provider"
    return provider

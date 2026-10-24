# -*- coding: utf-8 -*-
"""组合级仓位缩放：连续、渐进，拒绝开关式全清。

为什么要有这个文件
------------------
2026-10-23 用真实面板（200 只 × 2015→2026）做过的回测实证：

    现状（不止损）      年化 +26.0%  夏普 1.05  最大回撤 -34.4%
    组合 -10% 回撤清仓  年化  +4.7%  夏普 0.52  最大回撤 -13.9%

「触线全清」确实把回撤砍掉一大截，但因为清仓后长期空仓，年化从 26% 掉到
4.7%、夏普腰斩——**回撤治好了，收益也治没了**。

结论（已写入项目记忆）：回撤要治，方向是**连续渐进降仓**，不是开关式全清。
本模块把这条结论固化成可单测、可调参的一段代码。
"""

from __future__ import annotations


def drawdown_pct(equity: float, peak_equity: float) -> float:
    """当前峰谷回撤百分比（4.0 表示 -4%）。输入非法时返回 0（不退化为负）。"""
    try:
        eq = float(equity)
        peak = float(peak_equity)
    except (TypeError, ValueError):
        return 0.0
    if peak <= 0 or eq <= 0:
        return 0.0
    return max(0.0, (1.0 - eq / peak) * 100.0)


def portfolio_risk_scale(
    current_drawdown_pct: float,
    max_drawdown_pct: float,
    floor: float = 0.30,
    start_cut_pct: float = 0.0,
) -> float:
    """按组合当前回撤，连续缩放总敞口系数。

    返回 1.0 表示满仓参与，floor 表示最深回撤时仍保留的敞口比例。
    与「触线全清」的根本区别：**始终留一部分在场**，市场恢复时不踏空，
    也不会因为一次性全清而在反弹初期追高。

    Parameters
    ----------
    current_drawdown_pct : 当前峰谷回撤百分比（如 4.0 表示 -4%）
    max_drawdown_pct : 回撤熔断线（对应配置 max_portfolio_drawdown_pct）
    floor : 最深回撤时保留的敞口比例，默认 0.30（绝不降到 0）
    start_cut_pct : 回撤超过该值才开始降仓，默认 0（立即开始）

    Returns
    -------
    敞口系数，恒在 [floor, 1.0]
    """
    if max_drawdown_pct <= 0:
        return 1.0
    try:
        dd = float(current_drawdown_pct)
    except (TypeError, ValueError):
        return 1.0
    dd = max(0.0, dd)

    floor = min(max(float(floor), 0.0), 1.0)
    if dd <= start_cut_pct:
        return 1.0

    span = max(float(max_drawdown_pct) - float(start_cut_pct), 1e-9)
    progress = min((dd - start_cut_pct) / span, 1.0)
    return max(floor, 1.0 - (1.0 - floor) * progress)


def volatility_target_fraction(
    target_vol: float,
    annualized_vol: float,
    n_positions: int = 1,
    max_single_pct: float = 0.30,
) -> float:
    """按组合波动率目标，推导**单笔**仓位比例。

    为什么不是简单的 ``target / vol``
    --------------------------------
    ``volatility_target_pct`` 是**组合**年化波动率目标（默认 0.15），而
    ``annualized_vol`` 是**单只个股**的年化波动率。直接拿组合目标除以个股
    波动，得到的量级是「整只票占满组合」（A 股个股 vol 普遍 0.25~0.60，
    → 0.25~0.60 的仓位），必须靠封顶救场；而封顶一旦偏松，所有票都吃满
    同一个上限，**波动率目标法就退化成了「固定比例 × 常数」**，完全失去
    「高波动买得少」的意义。

    正确做法：单笔的波动贡献只应占组合目标的 ``1/sqrt(N)``
    （N = 持仓分散度），即 ``w_i = target / (vol_i * sqrt(N))``。
    这样同为 15% 的组合目标下，vol=0.30 的票约 11%，vol=0.60 的票约 5.6%
    —— 区分度真实存在。

    Returns
    -------
    单笔仓位比例，已按单票上限的 60% 封顶
    """
    try:
        tv = float(target_vol)
        vol = float(annualized_vol)
    except (TypeError, ValueError):
        return 0.0
    if tv <= 0 or vol <= 0:
        return 0.0

    n = max(int(n_positions or 1), 1)
    frac = tv / (vol * (n ** 0.5))

    cap = max(float(max_single_pct), 0.0) * 0.6
    return max(0.0, min(frac, cap))

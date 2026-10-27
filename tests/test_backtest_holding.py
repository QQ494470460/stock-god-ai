# -*- coding: utf-8 -*-
"""回测引擎「持有纪律」参数测试。

``BacktestConfig.hold_if_profitable_pct`` 是实盘 ``trading_loop.hold_if_profitable_pct``
在回测里的对应物：跌出目标组合、但浮盈达标时不强制卖出。默认 0 必须与旧行为
**完全等价**（否则会悄悄改变所有历史回测口径）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.core.backtest import run_backtest


def _panel():
    """构造：WIN 仅第 0 天在榜首且每天 +10%；之后垫底但持续上涨。

    这样「跌出目标 + 浮盈 10%」两个条件同时成立，正好用来区分
    「持有纪律开/关」——关则卖掉，开则保留。
    """
    dates = pd.date_range("2024-01-01", periods=10)
    codes = ["WIN"] + [f"S{i}" for i in range(6)]
    score = pd.DataFrame(0.1, index=dates, columns=codes, dtype=float)
    score.loc[dates[0], "WIN"] = 0.9
    score.loc[dates[0], "S0"] = 0.8
    for d in dates[1:]:
        score.loc[d, "WIN"] = -0.9
        score.loc[d, "S1"] = 0.9
        score.loc[d, "S2"] = 0.8
    close = pd.DataFrame(10.0, index=dates, columns=codes, dtype=float)
    close["WIN"] = [10.0 * (1.10 ** i) for i in range(len(dates))]
    return score, close, dates


def test_hold_param_zero_is_noop():
    """显式传 0 必须与不传完全一致（默认即旧口径）。"""
    score, close, _ = _panel()
    a = run_backtest(score, close, top_n=2, rebalance_days=1)
    b = run_backtest(score, close, top_n=2, rebalance_days=1,
                     hold_if_profitable_pct=0.0)
    for k in ("ann_return", "sharpe", "turnover", "n_periods"):
        av, bv = a.metrics.get(k), b.metrics.get(k)
        if isinstance(av, float) and np.isnan(av):
            continue
        assert av == pytest.approx(bv), f"{k} 在 pct=0 下不一致"


def test_hold_keeps_winner_out_of_target():
    """pct>0 时，跌出目标但浮盈 10% 的 WIN 应被保留继续持有。"""
    score, close, dates = _panel()
    res = run_backtest(score, close, top_n=2, rebalance_days=1,
                       hold_if_profitable_pct=5.0)
    # 第 1 天（跌幅出目标那天）持仓里仍应有 WIN
    day1 = str(pd.Timestamp(dates[1]).date())
    hist = {h["date"]: h["top"] for h in (res.holdings or [])}
    assert day1 in hist, "缺少第 1 天持仓记录"
    assert "WIN" in hist[day1], "盈利仓跌出目标后被强制卖出了（持有纪律未生效）"


def test_hold_disabled_sells_winner():
    """pct=0（默认）时，跌出目标的 WIN 必须被卖出（旧行为）。"""
    score, close, dates = _panel()
    res = run_backtest(score, close, top_n=2, rebalance_days=1,
                       hold_if_profitable_pct=0.0)
    day1 = str(pd.Timestamp(dates[1]).date())
    hist = {h["date"]: h["top"] for h in (res.holdings or [])}
    assert "WIN" not in hist.get(day1, []), "默认应在跌出目标时卖出"


def test_hold_only_keeps_profitable_not_losers():
    """持有纪律只保盈利仓：浮盈不足阈值（亏损）的票仍被卖出。"""
    score, close, dates = _panel()
    # 把 WIN 改成下跌，浮盈变成亏损 → 即使 pct>0 也不应保留
    close["WIN"] = [10.0 * (0.95 ** i) for i in range(len(dates))]
    res = run_backtest(score, close, top_n=2, rebalance_days=1,
                       hold_if_profitable_pct=5.0)
    day1 = str(pd.Timestamp(dates[1]).date())
    hist = {h["date"]: h["top"] for h in (res.holdings or [])}
    assert "WIN" not in hist.get(day1, []), "亏损仓不应被持有纪律保留"

# -*- coding: utf-8 -*-
"""严谨回测引擎单元测试（离线，合成数据）"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.core.backtest import WalkForwardBacktester, BacktestConfig, run_backtest  # noqa: E402


def _make_panel(n_days=120, n_codes=12):
    """构造：code0 稳定上涨，其余横盘；打分始终把 code0 排第一。"""
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    codes = [f"{i:06d}" for i in range(n_codes)]
    data = {}
    for i, c in enumerate(codes):
        if i == 0:
            data[c] = 10.0 * np.cumprod(np.full(n_days, 1.008))
        else:
            data[c] = 10.0 * np.ones(n_days)
    close = pd.DataFrame(data, index=dates)
    scores = pd.DataFrame(0.0, index=dates, columns=codes)
    scores[codes[0]] = 100.0
    for j, c in enumerate(codes[1:], start=1):
        scores[c] = 50.0 - j  # 稳定序，避免并列
    return scores, close


def test_backtest_picks_winner_positive():
    scores, close = _make_panel()
    bt = WalkForwardBacktester(BacktestConfig(top_n=2, rebalance_days=5))
    res = bt.backtest_scores(scores, close)
    assert res.metrics["n_periods"] > 0
    assert res.metrics["ann_return"] > 0
    assert res.metrics["sharpe"] > 0
    assert res.metrics["max_drawdown"] <= 0
    assert res.equity is not None and res.equity.iloc[-1] > 1.0


def test_costs_reduce_return():
    scores, close = _make_panel()
    hi = WalkForwardBacktester(BacktestConfig(
        top_n=3, rebalance_days=5, commission=0.01, stamp_tax=0.01,
        slippage=0.01, min_commission=0.0))
    lo = WalkForwardBacktester(BacktestConfig(
        top_n=3, rebalance_days=5, commission=0.0, stamp_tax=0.0,
        slippage=0.0, min_commission=0.0))
    rh = hi.backtest_scores(scores, close).metrics["top_excess"]
    rl = lo.backtest_scores(scores, close).metrics["top_excess"]
    # 注：top_excess 为毛超额，成本体现在 net；此处比较年化收益更直观
    ar_hi = hi.backtest_scores(scores, close).metrics["ann_return"]
    ar_lo = lo.backtest_scores(scores, close).metrics["ann_return"]
    assert ar_lo > ar_hi
    assert rl >= rh - 1e-9  # 毛超额不应因成本而改变符号方向


def test_rank_ic_sign():
    scores, close = _make_panel()
    # 测试面板仅 12 只，调低每日最少股票数阈值（生产默认 30）
    bt = WalkForwardBacktester(BacktestConfig(top_n=2, rebalance_days=5, min_stocks_per_day=5))
    res = bt.backtest_scores(scores, close)
    # code0 上涨且打分最高 → IC 应为正
    assert res.metrics["rank_ic"] > 0


def test_insufficient_data():
    dates = pd.date_range("2023-01-02", periods=3, freq="B")
    scores = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=dates)
    close = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=dates)
    res = run_backtest(scores, close, top_n=1, rebalance_days=5)
    assert res.metrics.get("n_periods", 0) == 0

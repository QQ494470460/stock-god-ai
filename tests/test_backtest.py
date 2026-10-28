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


# ---------------------------------------------------------------------------
# 风险平价权重支持（与生产 optimizer 口径一致）
# ---------------------------------------------------------------------------
def _make_volatile_panel(n_days=200, n_codes=10, seed=42):
    """所有票都有正波动的随机游走面板（风险平价才能算出有意义权重）。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    codes = [f"V{i:04d}" for i in range(n_codes)]
    data = {
        c: 10.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.015 + 0.004 * i, n_days))
        for i, c in enumerate(codes)
    }
    close = pd.DataFrame(data, index=dates)
    scores = pd.DataFrame(0.0, index=dates, columns=codes)
    for j, c in enumerate(codes):
        scores[c] = 100.0 - j  # 固定排序，避免并列
    return scores, close


def test_backtestconfig_has_tilt_strength():
    assert hasattr(BacktestConfig, "tilt_strength")
    assert BacktestConfig().tilt_strength == 0.25


def test_backtest_risk_parity_distinct_from_equal():
    """风险平价应产生与等权不同的组合（权重口径真正生效，非走兜底）。"""
    scores, close = _make_volatile_panel()
    cfg_eq = BacktestConfig(top_n=8, rebalance_days=5, weighting="equal")
    cfg_rp = BacktestConfig(top_n=8, rebalance_days=5, weighting="risk_parity", tilt_strength=0.0)
    res_eq = WalkForwardBacktester(cfg_eq).backtest_scores(scores, close)
    res_rp = WalkForwardBacktester(cfg_rp).backtest_scores(scores, close)
    assert res_eq.metrics["n_periods"] > 0
    assert res_rp.metrics["n_periods"] > 0
    # 风险平价按波动率反比分配，与等权结果应有可测差异
    assert abs(res_eq.metrics["sharpe"] - res_rp.metrics["sharpe"]) > 1e-4


def test_backtest_risk_parity_no_future_leak():
    """截至 t 日的风险权重不得受 t 日之后的极端行情影响（无未来泄露）。"""
    scores, base = _make_volatile_panel(seed=7)
    cfg = BacktestConfig(top_n=8, rebalance_days=5, weighting="risk_parity", tilt_strength=0.0)
    bt = WalkForwardBacktester(cfg)
    bt.backtest_scores(scores, base)
    early_date = base.index[40]  # 第 40 个交易日，窗口内无未来数据
    top = scores.loc[early_date].nlargest(8).index.tolist()
    w_base = bt._target_weights(scores.loc[early_date], top, t_date=early_date)

    # 在 early_date 之后很远注入极端值
    close2 = base.copy()
    close2.loc[base.index[150], base.columns[0]] = (
        close2.loc[base.index[150], base.columns[0]] * 1000.0)
    bt2 = WalkForwardBacktester(cfg)
    bt2.backtest_scores(scores, close2)
    w_leak = bt2._target_weights(scores.loc[early_date], top, t_date=early_date)

    assert set(w_base) == set(w_leak)
    for k in w_base:
        assert abs(w_base[k] - w_leak[k]) < 1e-9, f"未来数据泄露到 {k}: {w_base[k]} vs {w_leak[k]}"


def test_backtest_risk_parity_fallback_safe():
    """历史窗口不足 / 含 NaN 时 fail-safe 回退等权，不崩溃。"""
    dates = pd.date_range("2023-01-02", periods=10, freq="B")
    codes = [f"V{i:04d}" for i in range(6)]
    rng = np.random.default_rng(1)
    close = pd.DataFrame(
        {c: 10.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, 10)) for c in codes}, index=dates)
    scores = pd.DataFrame(0.0, index=dates, columns=codes)
    for j, c in enumerate(codes):
        scores[c] = 100.0 - j
    cfg = BacktestConfig(top_n=4, rebalance_days=5, weighting="risk_parity", tilt_strength=0.25)
    res = WalkForwardBacktester(cfg).backtest_scores(scores, close)
    assert res.metrics is not None  # 不抛异常即通过

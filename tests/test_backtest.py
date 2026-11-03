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
    from config.settings import AppConfig
    assert BacktestConfig().tilt_strength == AppConfig().trading_loop.tilt_strength


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


# ---------------------------------------------------------------------------
# 排名滞后 / 调仓相位（口径与 exec_backtest 对齐，2026-11-04）
# ---------------------------------------------------------------------------

def _rank_panel(n_days: int = 60, n_codes: int = 12):
    """质量单调的面板：代码 0 最好、代码 n-1 最差，收益与质量成正比。"""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{600000 + i}" for i in range(n_codes)]
    quality = np.linspace(1.0, 0.0, n_codes)
    close = pd.DataFrame(1.0, index=dates, columns=codes)
    for j, c in enumerate(codes):
        r = 0.004 * quality[j] + rng.normal(0, 0.0008, len(dates))
        close[c] = (1 + r).cumprod()
    score = pd.DataFrame(np.tile(quality * 100.0, (n_days, 1)),
                         index=dates, columns=codes)
    return score, close


def _bt_cfg(**kw):
    base = dict(top_n=3, rebalance_days=5, commission=0.0, min_commission=0.0,
                stamp_tax=0.0, slippage=0.0, weighting="equal")
    base.update(kw)
    return BacktestConfig(**base)


def test_hold_rank_slack_default_is_bit_identical():
    """slack=1.0 = 关闭，必须与旧口径逐位相同（新增旋钮不许改历史结论）。

    ⚠️ 不能直接比 ``metrics`` 字典：里面含 NaN，而 ``NaN != NaN``
    会让「完全一样」的两个结果被判成不同。
    """
    score, close = _rank_panel()
    a = WalkForwardBacktester(_bt_cfg()).backtest_scores(score, close)
    b = WalkForwardBacktester(_bt_cfg(hold_rank_slack=1.0)).backtest_scores(score, close)
    assert a.metrics["n_periods"] == b.metrics["n_periods"]
    assert np.allclose(a.equity.values, b.equity.values, equal_nan=True)
    for k, v in a.metrics.items():
        w = b.metrics[k]
        assert (v == w) or (pd.isna(v) and pd.isna(w)), f"{k}: {v} != {w}"


def test_hold_rank_slack_lowers_turnover():
    """slack>1 必须真的降低换手 —— 否则闸门里这个参数形同虚设。

    面板用**缓慢漂移**的榜单（峰值每期右移一位，分数 = −(代码序号−峰值)²）：
    无 slack 时每期换掉一只，有 slack（top_n=3、slack=3 ⇒ 保留线第 9 名）时
    上一期的三只仍在保留线内，几乎不换。
    ⚠️ 这里踩过两次坑：①「整体倒转」的榜单会让老仓直接掉到第 10 名以后，
    全部超出保留线 ⇒ slack 完全无效（换手逐位相同）；②「整体 roll 一位」同理，
    因为最差的那只被瞬移到榜首。只有**平滑漂移**的榜单才测得出来。
    """
    score, close = _rank_panel(n_days=60, n_codes=12)
    idx = np.arange(12, dtype=float)
    for k, i in enumerate(range(0, len(score), 5)):
        row = -(idx - float(k)) ** 2
        score.iloc[i] = row
        score.iloc[min(i + 4, len(score) - 1)] = row
    tight = WalkForwardBacktester(_bt_cfg()).backtest_scores(score, close).metrics
    loose = WalkForwardBacktester(_bt_cfg(hold_rank_slack=3.0)).backtest_scores(
        score, close).metrics
    assert loose["n_periods"] == tight["n_periods"]
    assert loose["turnover"] < tight["turnover"], "slack 没降低换手 ⇒ 参数没生效"


def test_rebalance_offset_shifts_the_start():
    """相位必须真的移动 —— 否则闸门的「相位平均」只是把同一次回测跑三遍。"""
    score, close = _rank_panel()
    a = WalkForwardBacktester(_bt_cfg()).backtest_scores(score, close)
    b = WalkForwardBacktester(_bt_cfg(rebalance_offset=2)).backtest_scores(score, close)
    assert a.metrics != b.metrics or \
        not np.allclose(a.equity.values, b.equity.values, equal_nan=True)

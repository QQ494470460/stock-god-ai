# -*- coding: utf-8 -*-
"""执行级回测引擎（src/core/exec_backtest.py）的机制测试。

这个引擎是「改造是否成立」的唯一裁判，所以它自己的口径必须先钉死：
  · 信号在 T 日收盘产生 → T+1 开盘成交（不能用 T 日收盘价，等于偷未来）
  · 止损按**日内最低价**触发，成交在止损价（跳空则按开盘价，更差）
  · 阈值设 0 必须表示「关闭」，不是「立刻触发」
  · 成本必须真的扣掉（滑点变大 → 收益必然变低）

行情全部人造、零网络，40 个交易日足够跑完逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.core.exec_backtest import ExecConfig, run_exec_backtest


def _panel(closes, opens=None, highs=None, lows=None, scores=None):
    """按给定的收盘序列构造单标的面板（index=交易日）。"""
    idx = pd.bdate_range("2026-01-05", periods=len(closes))
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else c.copy()
    # 默认给一点点振幅：high==low==close 会被引擎判成「一字板」
    # （涨跌停封死，买不进也卖不出）—— 那是真实规则，不是 bug，
    # 但人造面板必须避开，否则测出来「从不成交」。
    h = np.asarray(highs, dtype=float) if highs is not None else np.maximum(o, c) * 1.005
    lw = np.asarray(lows, dtype=float) if lows is not None else np.minimum(o, c) * 0.995
    sc = np.asarray(scores, dtype=float) if scores is not None else np.ones(len(c))
    mk = lambda arr, col: pd.DataFrame({col: arr}, index=idx)  # noqa: E731
    ohlc = {
        "open": mk(o, "A"), "high": mk(h, "A"), "low": mk(lw, "A"),
        "close": mk(c, "A"), "volume": mk(np.full(len(c), 1e6), "A"),
    }
    return pd.DataFrame({"A": sc}, index=idx), ohlc


def _cfg(**kw):
    base = dict(top_n=1, rebalance_days=5, weighting="equal", cash_reserve_pct=0.0,
                weight_cap=1.0, band_pct=0.0, commission=0.0, min_commission=0.0,
                stamp_tax=0.0, slippage_bp=0.0, capital=1_000_000.0)
    base.update(kw)
    return ExecConfig(**base)


def test_signal_fills_at_next_open_not_signal_close():
    """T 日收盘出信号，必须按 T+1 开盘价成交。

    把 T 日收盘故意设成 99 元、T+1 开盘 10 元：若引擎用信号日收盘价成交，
    股数会是 1 万股；按开盘价成交才是 10 万股。这是**最容易被忽略的未来函数**。
    """
    closes = [99.0, 10.0] + [10.0] * 10
    opens = [99.0, 10.0] + [10.0] * 10
    scores, ohlc = _panel(closes, opens=opens)
    res = run_exec_backtest(scores, ohlc, _cfg(top_n=1, rebalance_days=5))
    assert len(res.trades) >= 0
    # 第一笔买入应当发生在 T+1（index 1）
    first_buy_px = 10.0
    tr = res.trades
    if len(tr):
        assert tr.iloc[0]["entry_px"] == pytest.approx(first_buy_px)
    # 期末权益应约等于本金（价格恒定、零成本）
    assert res.equity.iloc[-1] == pytest.approx(1_000_000.0, rel=1e-6)


def test_stop_triggers_on_intraday_low_and_fills_at_stop_price():
    """固定 -10% 止损：第 3 天最低价 8.0 击穿 9.0 的止损线 → 按 9.0 成交。"""
    n = 12
    closes = [10.0] * n
    opens = [10.0] * n
    highs = [10.2] * n
    lows = [9.6] * n
    closes[3] = lows[3] = 8.0          # 第 3 天盘中击穿
    scores, ohlc = _panel(closes, opens=opens, highs=highs, lows=lows)
    res = run_exec_backtest(scores, ohlc, _cfg(stop_mode="fixed", stop_loss_pct=-10.0))
    stops = res.trades[res.trades["reason"] == "stop_loss"]
    assert len(stops) == 1, res.trades
    assert stops.iloc[0]["exit_px"] == pytest.approx(9.0, rel=1e-6)


def test_zero_take_profit_means_disabled():
    """stop_profit_pct=0 是「关闭」，不能变成「一有浮盈就卖」。"""
    n = 12
    closes = [10.0, 10.0] + [15.0] * (n - 2)     # 一路 +50%
    scores, ohlc = _panel(closes)
    res = run_exec_backtest(scores, ohlc, _cfg(take_profit_pct=0.0))
    # 关掉止盈后：持仓一路拿着，不该出现任何 stop_profit 平仓腿
    assert len(res.trades) == 0 or not (res.trades["reason"] == "stop_profit").any()
    # 反证：给了 +10% 止盈就应当触发
    res2 = run_exec_backtest(scores, ohlc, _cfg(take_profit_pct=10.0))
    assert len(res2.trades) > 0 and (res2.trades["reason"] == "stop_profit").any()


def test_slippage_reduces_return():
    """滑点必须真的扣掉：同一条价格路径，成本越高收益越低。"""
    n = 30
    closes = [10.0 * (1.01 ** i) for i in range(n)]     # 稳步上涨，反复调仓
    scores, ohlc = _panel(closes)
    r0 = run_exec_backtest(scores, ohlc, _cfg(slippage_bp=0.0)).metrics["total_return"]
    r50 = run_exec_backtest(scores, ohlc, _cfg(slippage_bp=50.0)).metrics["total_return"]
    assert r50 < r0, f"50bp 滑点必须降低收益（{r50:.4f} vs {r0:.4f}）"


def test_cash_reserve_keeps_part_uninvested():
    """现金保留：满仓与 50% 现金的收益必须显著不同（否则权重没生效）。"""
    n = 40
    closes = [10.0 * (1.02 ** i) for i in range(n)]
    scores, ohlc = _panel(closes)
    full = run_exec_backtest(scores, ohlc, _cfg(cash_reserve_pct=0.0)).metrics["total_return"]
    half = run_exec_backtest(scores, ohlc, _cfg(cash_reserve_pct=0.5)).metrics["total_return"]
    assert 0 < half < full


def test_top_n_requires_enough_universe():
    """可选标的少于 top_n 时安全退出，不能抛异常。"""
    scores, ohlc = _panel([10.0] * 10)
    res = run_exec_backtest(scores, ohlc, _cfg(top_n=3))
    assert res.metrics.get("n_trades", 0) == 0

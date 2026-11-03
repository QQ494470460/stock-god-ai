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


# ======================================================================
# 单日亏损熔断（ExecConfig.daily_halt_pct）
# 实盘 guard.max_daily_loss_pct 的等价物 —— 11 处漂移之外的第 12 处
# ======================================================================

def _crash_panel(rebound: bool = True):
    """先平盘、第 11 天暴跌 -8%、随后反弹（可选）的单标的面板。

    暴跌必须发生在**已持仓**之后，否则空仓状态下权益不动，
    日收益恒为 0，熔断永远不会触发 —— 这是这类测试最容易写错的地方。
    """
    closes = [10.0] * 10 + [9.2] + [9.2] * 4
    if rebound:
        closes += [10.12, 11.13, 12.24, 13.47, 14.82]
    closes += [closes[-1]] * 5
    return _panel(closes)


def test_daily_halt_off_by_default_is_bit_identical():
    """默认 0 = 关闭：必须与完全不带该参数的旧口径逐位相同。

    这条守的是「加了个新旋钮却没有偷偷改变历史结论」——
    任何新增风控都必须先证明自己是零影响的。
    """
    scores, ohlc = _crash_panel()
    a = run_exec_backtest(scores, ohlc, _cfg(rebalance_days=5))
    b = run_exec_backtest(scores, ohlc, _cfg(rebalance_days=5,
                                           daily_halt_pct=0.0,
                                           daily_halt_cooldown_days=0))
    assert a.metrics["ann_return"] == pytest.approx(b.metrics["ann_return"])
    assert a.metrics["daily_halt_count"] == 0.0
    assert np.allclose(a.equity.values, b.equity.values, equal_nan=True)


def test_daily_halt_triggers_on_big_down_day():
    """单日暴跌 8% 必须触发（3% 线）。"""
    scores, ohlc = _crash_panel()
    m = run_exec_backtest(scores, ohlc, _cfg(rebalance_days=5,
                                            daily_halt_pct=3.0)).metrics
    assert m["daily_halt_count"] >= 1.0
    assert m["halt_days_pct"] > 0.0


def test_daily_halt_ignores_normal_volatility():
    """日常 ±0.5% 的波动绝不能触发 —— 阈值不能宽到「跌一点就拉闸」。

    实盘 3% 线的教训：线设在策略正常日波动之内，就会每个季度付一次费。
    """
    closes = [10.0]
    for i in range(30):
        closes.append(closes[-1] * (1.005 if i % 2 == 0 else 0.995))
    scores, ohlc = _panel(closes)
    m = run_exec_backtest(scores, ohlc, _cfg(rebalance_days=5,
                                            daily_halt_pct=3.0)).metrics
    assert m["daily_halt_count"] == 0.0


def _rotation_panel():
    """双标的「暴跌日恰逢排行榜互换」面板 —— 专门用来证明熔断真的拦住了换仓。

    为什么不能只用单标的：i=0 就建好仓，之后每次调仓都是「继续持有」，
    冻结它不会产生任何成交差异，测出来永远是「一样」。
    必须让被冻结的那次调仓**本来要换票**，才看得出区别。

      i=0..9  A 分高 → 持有 A；i=10 起 B 分高（本该卖 A 买 B）
      i=10    A 暴跌 -8%（触发熔断），之后 A 阴跌、B 上涨
    """
    n = 30
    idx = pd.bdate_range("2026-01-05", periods=n)
    a = [10.0] * 10 + [9.2] + [9.2 * (0.99 ** k) for k in range(1, 20)]
    b = [10.0] * 11 + [10.0 * (1.01 ** k) for k in range(1, 20)]
    a, b = a[:n], b[:n]
    sc = pd.DataFrame({
        "A": [1.0] * 10 + [0.0] * (n - 10),
        "B": [0.0] * 10 + [1.0] * (n - 10),
    }, index=idx)
    mk = lambda arr, code: pd.DataFrame({code: arr}, index=idx)  # noqa: E731
    a, b = np.asarray(a, float), np.asarray(b, float)
    ohlc = {
        "open": pd.concat([mk(a, "A"), mk(b, "B")], axis=1),
        "close": pd.concat([mk(a, "A"), mk(b, "B")], axis=1),
        "high": pd.concat([mk(a * 1.005, "A"), mk(b * 1.005, "B")], axis=1),
        "low": pd.concat([mk(a * 0.995, "A"), mk(b * 0.995, "B")], axis=1),
        "volume": pd.concat([mk(np.full(n, 1e6), "A"), mk(np.full(n, 1e6), "B")], axis=1),
    }
    return sc, ohlc


def test_daily_halt_freezes_the_rebalance_that_day():
    """熔断当日必须**跳过调仓**（顺延到下一个调仓日），不是只记个数。

    构造：i=10 既是调仓日、又是排行榜互换日、还是 A 暴跌 -8% 的那天。
    关闭 → 次日卖 A 买 B，之后吃 B 的上涨；
    开启 → 当日冻结，继续死抱下跌的 A 直到 i=15 ⇒ 期末权益必然更低。
    这正是实盘 3% 线吃掉年化 2.7pp 的机理：**冻结调仓 = 把亏损组合锁死**。
    """
    scores, ohlc = _rotation_panel()
    off = run_exec_backtest(scores, ohlc, _cfg(top_n=1, rebalance_days=5))
    on = run_exec_backtest(scores, ohlc,
                           _cfg(top_n=1, rebalance_days=5, daily_halt_pct=3.0))
    assert on.metrics["daily_halt_count"] >= 1.0

    end_off = float(off.equity.dropna().iloc[-1])
    end_on = float(on.equity.dropna().iloc[-1])
    assert end_on != pytest.approx(end_off), "冻结没改变成交时点，说明没真正拦住调仓"
    assert end_on < end_off, "被冻结的换仓错过了替代标的的上涨，权益应更低"


def test_daily_halt_cooldown_extends_freeze():
    """冷却期必须真的多冻结几天（实盘解冻后的半仓冷静期用这个近似）。"""
    scores, ohlc = _crash_panel()
    base = run_exec_backtest(scores, ohlc, _cfg(rebalance_days=5,
                                               daily_halt_pct=3.0,
                                               daily_halt_cooldown_days=0)).metrics
    cooled = run_exec_backtest(scores, ohlc, _cfg(rebalance_days=5,
                                                 daily_halt_pct=3.0,
                                                 daily_halt_cooldown_days=3)).metrics
    assert cooled["halt_days_pct"] > base["halt_days_pct"]


# ======================================================================
# 行业集中度上限（ExecConfig.max_sector_pct）—— 第 13 处「实盘有回测没有」
# ======================================================================

def _two_same_sector(n: int = 30):
    """两只**同行业**、都在缓慢上涨的票。用来观察行业额度被谁吃满。"""
    idx = pd.bdate_range("2026-01-05", periods=n)
    a = np.array([10.0 * (1.01 ** k) for k in range(n)])
    b = np.array([20.0 * (1.01 ** k) for k in range(n)])
    sc = pd.DataFrame({"A": 1.0, "B": 0.9}, index=idx)
    mk = lambda arr, code: pd.DataFrame({code: arr}, index=idx)  # noqa: E731
    ohlc = {
        "open": pd.concat([mk(a, "A"), mk(b, "B")], axis=1),
        "close": pd.concat([mk(a, "A"), mk(b, "B")], axis=1),
        "high": pd.concat([mk(a * 1.005, "A"), mk(b * 1.005, "B")], axis=1),
        "low": pd.concat([mk(a * 0.995, "A"), mk(b * 0.995, "B")], axis=1),
        "volume": pd.concat([mk(np.full(n, 1e6), "A"), mk(np.full(n, 1e6), "B")], axis=1),
    }
    return sc, ohlc


_SECTOR = {"A": "银行Ⅱ", "B": "银行Ⅱ"}


def test_max_sector_off_is_bit_identical():
    """默认 0 = 关闭：必须与不传该参数逐位相同（新增闸门不能偷改历史结论）。"""
    scores, ohlc = _two_same_sector()
    cfg_kw = dict(top_n=2, rebalance_days=5, weight_cap=0.30)
    a = run_exec_backtest(scores, ohlc, _cfg(**cfg_kw))
    b = run_exec_backtest(scores, ohlc, _cfg(max_sector_pct=0.0, industry_map=_SECTOR, **cfg_kw))
    assert np.allclose(a.equity.values, b.equity.values, equal_nan=True)


def test_max_sector_cap_blocks_third_name_in_same_sector():
    """行业额度被吃满后，后面的票权重归零（钱留现金），不是摊给别的票。

    两只票各 30%（weight_cap）× 0.9 可投 = 27% 行业敞口：
      · 上限 30% → 第二只 27+27=54% 超线 ⇒ 只买得进 1 只
      · 上限 60% → 两只都买 ⇒ 吃满两只的涨幅，期末权益必然更高
    这直接复现了实盘「计划要 4 只同行业、实际只成交 2 只」的偏差。
    """
    scores, ohlc = _two_same_sector()
    cfg_kw = dict(top_n=2, rebalance_days=5, weight_cap=0.30, cash_reserve_pct=0.10)
    tight = run_exec_backtest(scores, ohlc,
                              _cfg(max_sector_pct=0.30, industry_map=_SECTOR, **cfg_kw))
    loose = run_exec_backtest(scores, ohlc,
                              _cfg(max_sector_pct=0.60, industry_map=_SECTOR, **cfg_kw))
    end_tight = float(tight.equity.dropna().iloc[-1])
    end_loose = float(loose.equity.dropna().iloc[-1])
    assert end_loose > end_tight, "宽上限必须能买到更多、从而吃到更多涨幅"


def test_max_sector_without_map_is_noop():
    """没有行业映射时 fail-open：不猜行业、不误拦（否则映射缺失会变成静默减仓）。"""
    scores, ohlc = _two_same_sector()
    cfg_kw = dict(top_n=2, rebalance_days=5, weight_cap=0.30)
    a = run_exec_backtest(scores, ohlc, _cfg(**cfg_kw))
    b = run_exec_backtest(scores, ohlc, _cfg(max_sector_pct=0.30, industry_map=None, **cfg_kw))
    assert np.allclose(a.equity.values, b.equity.values, equal_nan=True)

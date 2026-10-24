# -*- coding: utf-8 -*-
"""组合级连续降仓（src/trading/sizing.py + rebalance.risk_scale）。

为什么是「连续」而不是「触线全清」
--------------------------------
2026-10-23 的回测实证（200 只 × 2015→2026）：

    现状（不止损）      年化 +26.0%  夏普 1.05  最大回撤 -34.4%
    组合 -10% 回撤清仓  年化  +4.7%  夏普 0.52  最大回撤 -13.9%

开关式全清确实把回撤砍掉一大截，但因为清仓后长期空仓，年化从 26% 掉到
4.7%、夏普腰斩。所以降仓必须**保留敞口在场**：回撤越深仓位越低，但绝不为 0。

本文件钉死这条设计，并防住「数据缺失被误判为深回撤」的坑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.trading.broker import AccountInfo  # noqa: E402
from src.trading.rebalance import compute_rebalance_orders  # noqa: E402
from src.trading.sizing import (  # noqa: E402
    drawdown_pct,
    portfolio_risk_scale,
    volatility_target_fraction,
)
from src.trading.trade_plan import PlanTarget, TradePlan  # noqa: E402

MAX_DD = 8.0  # 与 config/trading_guard.yaml 的 max_portfolio_drawdown_pct 一致


# ---------------------------------------------------------------- 缩放曲线

def test_no_drawdown_keeps_full_exposure():
    assert portfolio_risk_scale(0.0, MAX_DD) == 1.0


def test_at_circuit_line_drops_to_floor():
    assert portfolio_risk_scale(MAX_DD, MAX_DD, floor=0.30) == pytest.approx(0.30)


def test_scaling_is_continuous_and_monotonic():
    """回撤越深，敞口单调下降；中途不能跳变。"""
    prev = 1.0
    for dd in [0.0, 1.0, 2.0, 4.0, 6.0, 7.5, 8.0]:
        cur = portfolio_risk_scale(dd, MAX_DD, floor=0.30)
        assert cur <= prev + 1e-9, f"回撤 {dd}% 时敞口回升了"
        prev = cur
    # 半程回撤应落在中间，而不是已跌到底
    assert 0.55 < portfolio_risk_scale(4.0, MAX_DD, floor=0.30) < 0.80


def test_never_reaches_zero_even_beyond_circuit_line():
    """回撤远超熔断线也保留敞口——这是「连续降仓」与「全清」的分界线。"""
    for dd in [8.0, 20.0, 99.0]:
        assert portfolio_risk_scale(dd, MAX_DD, floor=0.30) >= 0.30


def test_bad_input_falls_back_to_full_exposure():
    """数据缺失不能当成「回撤很深」而误砍仓位。"""
    assert portfolio_risk_scale(float("nan"), MAX_DD) == 1.0
    assert portfolio_risk_scale(None, MAX_DD) == 1.0      # type: ignore[arg-type]
    assert portfolio_risk_scale(4.0, 0) == 1.0            # 未配置熔断线


# ---------------------------------------------------------------- 回撤计算

def test_drawdown_pct_is_peak_to_current():
    assert drawdown_pct(920_000.0, 1_000_000.0) == pytest.approx(8.0)
    assert drawdown_pct(1_050_000.0, 1_000_000.0) == 0.0  # 创新高不算回撤


def test_drawdown_pct_rejects_bad_peak():
    assert drawdown_pct(100.0, 0) == 0.0
    assert drawdown_pct(0, 1_000_000.0) == 0.0


# ---------------------------------------------------------------- 调仓接入

def _plan(targets):
    return TradePlan(
        plan_date="2026-10-24",
        targets=[PlanTarget(code=c, name=c, weight=w) for c, w in targets],
        cash_reserve_pct=0.0,
        weighting="equal",
    )


def _acct(total=1_000_000.0):
    return AccountInfo(
        account_id="T", total_asset=total, available_cash=total,
        frozen_cash=0.0, market_value=0.0, total_pnl=0.0, today_pnl=0.0,
    )


def test_risk_scale_shrinks_target_value_not_liquidates():
    """降仓缩小买入目标市值，但绝不是清仓——仍会买入。"""
    plan = _plan([("600000", 1.0)])
    prices = {"600000": 10.0}

    full = compute_rebalance_orders(plan, [], _acct(), prices,
                                    allow_sell=False, risk_scale=1.0)
    scaled = compute_rebalance_orders(plan, [], _acct(), prices,
                                      allow_sell=False, risk_scale=0.30)

    full_amt = sum(i.amount for i in full)
    scaled_amt = sum(i.amount for i in scaled)

    assert full_amt > 0 and scaled_amt > 0, "降仓后仍应买入，不是清仓"
    assert scaled_amt == pytest.approx(full_amt * 0.30, rel=0.02)


def test_risk_scale_one_unchanged_from_before():
    """默认 1.0 必须与改造前行为一致（向后兼容）。"""
    plan = _plan([("600000", 1.0)])
    prices = {"600000": 10.0}

    default = compute_rebalance_orders(plan, [], _acct(), prices, allow_sell=False)
    explicit = compute_rebalance_orders(plan, [], _acct(), prices,
                                        allow_sell=False, risk_scale=1.0)

    assert sum(i.amount for i in default) == sum(i.amount for i in explicit)


# ---------------------------------------------------------------- 波动率目标法

def test_higher_volatility_gets_smaller_position():
    """波动率翻倍 → 仓位减半，这是波动率目标法的立身之本。"""
    low = volatility_target_fraction(0.15, 0.30, n_positions=20, max_single_pct=0.30)
    high = volatility_target_fraction(0.15, 0.60, n_positions=20, max_single_pct=0.30)

    assert high < low
    assert high == pytest.approx(low / 2, rel=0.01)


def test_common_volatility_range_does_not_always_hit_cap():
    """原实现的缺陷：所有票都吃满同一个封顶值，等于没有区分度。

    封顶 = max_single_pct * 0.6 = 0.18；常见 vol 0.25~0.60 的结果都应低于它。
    """
    cap = 0.30 * 0.6
    for vol in [0.25, 0.30, 0.45, 0.60]:
        frac = volatility_target_fraction(0.15, vol, n_positions=20,
                                          max_single_pct=0.30)
        assert frac < cap, f"vol={vol} 吃满了封顶，失去区分度"


def test_extreme_low_volatility_still_capped():
    """异常低的波动率（脏数据）不该给出荒谬的大仓位。"""
    assert volatility_target_fraction(0.15, 0.001, 20, 0.30) <= 0.30 * 0.6


def test_volatility_target_rejects_bad_input():
    assert volatility_target_fraction(0, 0.30, 20, 0.30) == 0.0
    assert volatility_target_fraction(0.15, 0.0, 20, 0.30) == 0.0
    assert volatility_target_fraction(0.15, -0.3, 20, 0.30) == 0.0

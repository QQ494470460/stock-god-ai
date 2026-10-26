# -*- coding: utf-8 -*-
"""调仓层的流动性事前封顶。

为什么在计划层封顶
------------------
执行层（guard）已经会拦超量委托，但那种拦截是**事后**的：
计划里写着「买 8000 股」，风控砍成 3200，日志里全是 blocked，
人看不出到底哪一环在缩量，也没法在事前判断这笔到底能不能做成。

在 ``compute_rebalance_orders`` 里按 ADV 上限先摊薄，剩下的一直由后续
tick 的 gap 补仓（每次 tick 重算 gap，天然分批），好处是：
- 计划显示的股数就是**真能成交**的股数；
- 不需要引入"分片状态机"，靠 gap 自愈；
- 卖出永不受流动性限制（风控绝不阻止减仓）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading.broker import AccountInfo, OrderSide, Position
from src.trading.rebalance import compute_rebalance_orders
from src.trading.trade_plan import PlanTarget, TradePlan

DAY = "2026-10-27"


class _Adv:
    """固定上限的 ADV 替身。"""

    def __init__(self, cap_qty):
        self.cap_qty = cap_qty
        self.calls = []

    def max_order_quantity(self, code, price, participation=0.01, lot=100):
        self.calls.append((code, price, participation, lot))
        return self.cap_qty


class _BoomAdv:
    def max_order_quantity(self, *a, **kw):
        raise RuntimeError("boom")


def _acct(total=1_000_000.0, cash=1_000_000.0):
    return AccountInfo(
        account_id="T", total_asset=total, available_cash=cash,
        frozen_cash=0.0, market_value=total - cash, total_pnl=0.0, today_pnl=0.0,
    )


def _plan(targets, reserve=0.0):
    return TradePlan(
        plan_date=DAY,
        targets=[PlanTarget(code=c, name=c, weight=w) for c, w in targets],
        cash_reserve_pct=reserve,
        weighting="equal",
    )


def _pos(code, qty, price):
    return Position(code=code, name=code, quantity=qty, available=qty,
                    avg_cost=price, current_price=price, market_value=qty * price,
                    pnl=0.0, pnl_pct=0.0, today_quantity=0)


def test_buy_capped_to_liquidity_limit():
    plan = _plan([("600000", 0.5)])          # 目标 50 万
    adv = _Adv(100)                          # 上限 100 股 @10 元
    intents = compute_rebalance_orders(
        plan, [], _acct(), {"600000": 10.0},
        adv_provider=adv, participation=0.01,
    )
    assert len(intents) == 0 or intents[0].capped_by == "liquidity"
    if intents:
        assert intents[0].quantity == 100


def test_buy_uncapped_when_limit_is_loose():
    plan = _plan([("600000", 0.5)])
    intents = compute_rebalance_orders(
        plan, [], _acct(), {"600000": 10.0},
        adv_provider=_Adv(10_000_000),
    )
    assert len(intents) == 1
    assert intents[0].capped_by == ""
    assert intents[0].quantity == 50_000   # 50 万 / 10 元


def test_capped_flag_and_reason_text():
    plan = _plan([("600000", 0.5)])
    intents = compute_rebalance_orders(
        plan, [], _acct(), {"600000": 10.0}, adv_provider=_Adv(2_000),
    )
    assert len(intents) == 1
    it = intents[0]
    assert it.quantity == 2_000
    assert it.capped_by == "liquidity"
    assert "流动性" in it.reason


def test_cap_below_one_lot_is_skipped():
    """流动性上限连一手都买不到 → 不下单（而不是下 0 股或非整手）。"""
    plan = _plan([("600000", 0.5)])
    intents = compute_rebalance_orders(
        plan, [], _acct(), {"600000": 10.0}, adv_provider=_Adv(50),
    )
    assert intents == []


def test_no_adv_provider_means_no_cap():
    plan = _plan([("600000", 0.5)])
    intents = compute_rebalance_orders(plan, [], _acct(), {"600000": 10.0})
    assert intents[0].quantity == 50_000
    assert intents[0].capped_by == ""


def test_adv_provider_error_is_fail_open():
    """ADV 取值异常 → 不封顶（数据故障不能变成交易停摆）。"""
    plan = _plan([("600000", 0.5)])
    intents = compute_rebalance_orders(
        plan, [], _acct(), {"600000": 10.0}, adv_provider=_BoomAdv(),
    )
    assert intents[0].quantity == 50_000


def test_sell_side_never_capped_by_liquidity():
    """持有一只已跌出目标组合的票 → 清仓，即使 ADV 极小也不许只卖一手。"""
    plan = _plan([("600001", 1.0)])          # 600000 不在目标里
    pos = _pos("600000", 5_000, 10.0)
    intents = compute_rebalance_orders(
        plan, [pos], _acct(cash=0.0, total=1_000_000.0), {"600000": 10.0, "600001": 10.0},
        adv_provider=_Adv(100),
    )
    sells = [i for i in intents if i.side == OrderSide.SELL]
    assert len(sells) == 1
    assert sells[0].quantity == 5_000
    assert sells[0].capped_by == ""

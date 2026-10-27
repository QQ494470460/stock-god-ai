# -*- coding: utf-8 -*-
"""
调仓计算（rebalance）单元测试
=============================
覆盖：退出清仓、容忍带内不动手、减仓、建仓/补仓、现金底线、
先卖后买、整手取整、最小交易金额、T+1 可卖量封顶、边界输入。
零网络、零真实资金。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading.broker import Position, AccountInfo, OrderSide
from src.trading.trade_plan import TradePlan, PlanTarget
from src.trading.rebalance import (
    compute_rebalance_orders, summarize_intents, net_cash_flow, RebalanceIntent,
)

DAY = "2026-10-15"


def mk_pos(code, qty, cost, price, today=0, avail=None):
    return Position(
        code=code, name=code, quantity=qty,
        available=qty - today if avail is None else avail,
        avg_cost=cost, current_price=price, market_value=qty * price,
        pnl=0.0, pnl_pct=0.0, today_quantity=today,
    )


def mk_acct(total=1_000_000.0, cash=400_000.0):
    return AccountInfo(
        account_id="T", total_asset=total, available_cash=cash,
        frozen_cash=0.0, market_value=total - cash, total_pnl=0.0, today_pnl=0.0,
    )


def mk_plan(targets, reserve=0.10, weighting="equal", cap=None):
    """targets: [(code, weight)]"""
    return TradePlan(
        plan_date=DAY,
        targets=[PlanTarget(code=c, name=c, weight=w) for c, w in targets],
        cash_reserve_pct=reserve,
        weighting=weighting,
    )


PRICES = {"600519": 10.0, "000001": 10.0, "300750": 10.0, "600000": 10.0}


# ---------------------------------------------------------------- 退出

def test_sell_position_not_in_plan():
    plan = mk_plan([("600519", 0.30), ("000001", 0.30), ("300750", 0.30)])
    positions = [mk_pos("600000", 5000, 11.0, 10.0)]
    acct = mk_acct(cash=1_000_000.0)
    intents = compute_rebalance_orders(plan, positions, acct, PRICES)

    sells = [i for i in intents if i.side == OrderSide.SELL]
    assert len(sells) == 1
    assert sells[0].code == "600000"
    assert sells[0].quantity == 5000           # 全清
    assert "跌出目标组合" in sells[0].reason


def test_orders_are_sells_first_then_buys():
    plan = mk_plan([("600519", 0.30)])
    positions = [mk_pos("600000", 5000, 11.0, 10.0)]
    intents = compute_rebalance_orders(plan, positions, mk_acct(cash=1_000_000.0), PRICES)
    sides = [i.side for i in intents]
    assert sides == sorted(sides, key=lambda s: 0 if s == OrderSide.SELL else 1)


# ---------------------------------------------------------------- 容忍带

def test_within_band_no_action():
    """持仓已接近目标市值 → 不动手，避免空转成本。"""
    plan = mk_plan([("600519", 0.30)])          # 目标 30 万
    positions = [mk_pos("600519", 29000, 10.0, 10.0)]   # 现值 29 万，偏离 3.3%
    intents = compute_rebalance_orders(plan, positions, mk_acct(), PRICES, band_pct=0.25)
    assert intents == []


def test_below_band_buys_gap():
    plan = mk_plan([("600519", 0.30)])          # 目标 30 万
    positions = [mk_pos("600519", 10000, 10.0, 10.0)]   # 现值 10 万
    acct = mk_acct(total=1_000_000.0, cash=500_000.0)
    intents = compute_rebalance_orders(plan, positions, acct, PRICES, band_pct=0.25)
    assert len(intents) == 1
    i = intents[0]
    assert i.side == OrderSide.BUY
    assert i.quantity == 20000                  # 补 20 万 / 10 元
    assert "补仓" in i.reason


def test_above_band_sells_excess():
    plan = mk_plan([("600519", 0.30)])          # 目标 30 万
    positions = [mk_pos("600519", 60000, 10.0, 10.0)]   # 现值 60 万
    acct = mk_acct(total=1_000_000.0, cash=0.0)
    intents = compute_rebalance_orders(plan, positions, acct, PRICES, band_pct=0.25)
    assert len(intents) == 1
    i = intents[0]
    assert i.side == OrderSide.SELL
    assert i.quantity == 30000                  # 减 30 万 / 10 元
    assert "超出目标仓位" in i.reason


# ---------------------------------------------------------------- 现金底线 / 预算

def test_cash_reserve_respected():
    """买完现金不得低于 总资产 × 现金保留比例。"""
    plan = mk_plan([("600519", 0.45)])          # 目标 45 万
    acct = mk_acct(total=1_000_000.0, cash=600_000.0)
    intents = compute_rebalance_orders(plan, [], acct, PRICES)
    spent = sum(i.amount for i in intents if i.side == OrderSide.BUY)
    assert acct.available_cash - spent >= acct.total_asset * plan.cash_reserve_pct - 1


def test_budget_exhausted_stops_buying():
    """预算有限时，缺口大的先买满，后续候选拿不到钱就不下。"""
    plan = mk_plan([("600519", 0.30), ("000001", 0.30), ("300750", 0.30)])
    acct = mk_acct(total=1_000_000.0, cash=400_000.0)
    intents = compute_rebalance_orders(plan, [], acct, PRICES)
    buys = [i for i in intents if i.side == OrderSide.BUY]
    spent = sum(i.amount for i in buys)
    # 可用 40 万 - 保留 10 万 = 30 万预算
    assert abs(spent - 300_000.0) < 1e-6
    assert len(buys) < 3                        # 钱不够，不可能三只都买满


def test_sell_proceeds_expand_budget():
    """卖出回款应计入买入预算（先卖后买）。"""
    plan = mk_plan([("600519", 0.30)])
    positions = [mk_pos("600000", 5000, 11.0, 10.0)]     # 卖 5 万
    acct = mk_acct(total=1_000_000.0, cash=150_000.0)    # 保留需 10 万
    intents = compute_rebalance_orders(plan, positions, acct, PRICES)
    buys = [i for i in intents if i.side == OrderSide.BUY]
    # 现金 15 万 + 回款 5 万 - 保留 10 万 = 10 万可买
    assert sum(i.amount for i in buys) == pytest.approx(100_000.0)


# ---------------------------------------------------------------- A股规则

def test_lot_size_rounding():
    plan = mk_plan([("600519", 0.10)])          # 目标 10 万
    # 价格 33.33 → 100000/33.33 = 3000.3 股 → 向下取整手 = 3000
    intents = compute_rebalance_orders(
        plan, [], mk_acct(total=1_000_000.0, cash=1_000_000.0), {"600519": 33.33})
    assert intents[0].quantity == 3000
    assert intents[0].quantity % 100 == 0


def test_min_trade_amount_filters_tiny():
    plan = mk_plan([("600519", 0.002)])         # 目标仅 2000 元 < 5000 门槛
    intents = compute_rebalance_orders(
        plan, [], mk_acct(total=1_000_000.0, cash=1_000_000.0), PRICES,
        min_trade_amount=5000.0)
    assert intents == []


def test_t1_today_buy_not_sellable():
    """今日买入的股票不可卖（退出信号也要遵守 T+1）。

    注意用“有目标但持仓跌出目标”的场景；空计划是「按兵不动」语义，
    见 test_empty_plan_is_noop_not_liquidation。
    """
    plan = mk_plan([("600519", 0.30)])
    positions = [mk_pos("600000", 5000, 10.0, 10.0, today=5000)]
    # cash=0 → 买不起目标，只剩“卖出 600000”这一个潜在意图，而它被 T+1 拦掉
    intents = compute_rebalance_orders(plan, positions, mk_acct(cash=0.0), PRICES)
    assert intents == []


def test_t1_partial_sellable():
    plan = mk_plan([("600519", 0.30)])
    positions = [mk_pos("600000", 5000, 10.0, 10.0, today=2000)]
    intents = compute_rebalance_orders(plan, positions, mk_acct(cash=0.0), PRICES)
    assert len(intents) == 1
    assert intents[0].quantity == 3000          # 只卖可用的 3000


def test_empty_plan_is_noop_not_liquidation():
    """安全语义：空计划 = 按兵不动，绝不清仓。

    模型某天没选出票（或数据故障导致 picks 为空）不应触发全仓卖出，
    否则一次数据抖动就能把整个组合清掉。
    """
    plan = mk_plan([])
    positions = [mk_pos("600000", 5000, 10.0, 10.0)]
    assert compute_rebalance_orders(plan, positions, mk_acct(), PRICES) == []


# ---------------------------------------------------------------- 边界

def test_no_account_returns_empty():
    plan = mk_plan([("600519", 0.30)])
    assert compute_rebalance_orders(plan, [], None, PRICES) == []


def test_zero_equity_returns_empty():
    plan = mk_plan([("600519", 0.30)])
    assert compute_rebalance_orders(plan, [], mk_acct(total=0.0, cash=0.0), PRICES) == []


def test_empty_plan_returns_empty():
    plan = mk_plan([])
    assert compute_rebalance_orders(plan, [mk_pos("600000", 100, 10, 10)], mk_acct(), PRICES) == []


def test_missing_price_for_new_target_skipped():
    """新标的没有价格 → 跳过买入（无法定价就不下单）。"""
    plan = mk_plan([("999999", 0.30)])
    intents = compute_rebalance_orders(plan, [], mk_acct(cash=1_000_000.0), {})
    assert intents == []


def test_held_position_falls_back_to_current_price():
    """持仓标的缺 tick 价时回退持仓现价。"""
    plan = mk_plan([("600519", 0.30)])
    positions = [mk_pos("600519", 10000, 10.0, 10.0)]
    intents = compute_rebalance_orders(plan, positions, mk_acct(cash=500_000.0), {})
    assert len(intents) == 1
    assert intents[0].side == OrderSide.BUY


def test_allow_flags_disable_direction():
    plan = mk_plan([("600519", 0.30)])
    positions = [mk_pos("600000", 5000, 11.0, 10.0)]
    intents = compute_rebalance_orders(
        plan, positions, mk_acct(cash=1_000_000.0), PRICES,
        allow_buy=False, allow_sell=False)
    assert intents == []


# ---------------------------------------------------------------- 辅助函数

def test_summarize_and_net_cash_flow():
    plan = mk_plan([("600519", 0.30)])
    positions = [mk_pos("600000", 5000, 11.0, 10.0)]
    intents = compute_rebalance_orders(plan, positions, mk_acct(cash=1_000_000.0), PRICES)
    txt = summarize_intents(intents)
    assert "卖出" in txt and "买入" in txt
    # 卖 5 万 - 买 30 万 = -25 万
    assert net_cash_flow(intents) == pytest.approx(-250_000.0)


def test_summarize_empty():
    assert "无需调仓" in summarize_intents([])


def test_intent_amount_and_dict():
    i = RebalanceIntent(code="600519", name="X", side=OrderSide.BUY,
                        quantity=1000, price=10.0)
    assert i.amount == 10_000.0
    assert i.delta_value == 10_000.0
    d = i.to_dict()
    assert d["side"] == "buy" and d["amount"] == 10_000.0


# ---------------------------------------------------------------- 退出纪律

def test_exit_ignores_min_trade_amount():
    """跌出目标组合的小仓位必须清掉 —— 退出纪律不能被碎单阈值吃掉。

    回归背景：A 段（退出）套用了 ``min_trade_amount``，于是「100 股 × ¥30 =
    ¥3000」这类小仓位永远不满足 ¥5000 门槛 ⇒ 永久滞留在组合里：占着持仓名额、
    每天被重算一遍却从不执行，且因为不在目标组合里也不会被加仓，只能等它自己
    涨过阈值（等于把纪律交给了运气）。碎单过滤是给**建仓**用的，不是退出用的。
    """
    plan = mk_plan([("600519", 0.30)])
    positions = [mk_pos("600000", 100, 30.0, 30.0)]        # 仅 ¥3000
    intents = compute_rebalance_orders(
        plan, positions, mk_acct(cash=1_000_000.0), PRICES, min_trade_amount=5000.0)

    sells = [i for i in intents if i.side == OrderSide.SELL]
    assert len(sells) == 1, "小仓位退出被最小交易金额过滤掉了"
    assert sells[0].code == "600000"
    assert sells[0].quantity == 100                        # 全清，不是部分减
    assert "跌出目标组合" in sells[0].reason


# ---------------------------------------------------------------- 持有纪律

def test_hold_if_profitable_keeps_winner():
    """持仓不在目标组合、但浮盈达标 → 不强制清仓，交止损/止盈托管。

    这是「卖赢家、留输家」逆向选择的直接解药：每天换票时，原本会把刚盈利的
    赢家甩掉（只赚 1%）、把还没跌的输家留在组合里（一路亏到止损）。实测盈亏比
    设计 3:1、实测 0.15 的根因之一就在这里。阈值需 walk-forward 验证后再上调。
    """
    plan = mk_plan([("000001", 0.30)])           # 600519 跌出目标组合
    positions = [mk_pos("600519", 1000, cost=100.0, price=110.0)]  # 浮盈 +10%
    intents = compute_rebalance_orders(
        plan, positions, mk_acct(), {"600519": 110.0},
        hold_if_profitable_pct=5.0)
    sells = [i for i in intents if i.side == OrderSide.SELL]
    assert all(i.code != "600519" for i in sells), "盈利仓被持有纪律外的强制清仓卖掉了"


def test_hold_disabled_default_sells_winner():
    """默认阈值 0 = 关闭持有纪律，保持原「跌出榜单即清仓」语义（向后兼容）。"""
    plan = mk_plan([("000001", 0.30)])
    positions = [mk_pos("600519", 1000, cost=100.0, price=110.0)]
    intents = compute_rebalance_orders(
        plan, positions, mk_acct(), {"600519": 110.0},
        hold_if_profitable_pct=0.0)
    sells = [i for i in intents if i.side == OrderSide.SELL and i.code == "600519"]
    assert sells, "默认应保留原有清仓纪律"


def test_hold_does_not_keep_loser():
    """持有纪律只保盈利仓：亏损仓即便浮盈阈值很高也必须清仓。"""
    plan = mk_plan([("000001", 0.30)])
    positions = [mk_pos("600519", 1000, cost=100.0, price=90.0)]   # 浮亏 -10%
    intents = compute_rebalance_orders(
        plan, positions, mk_acct(), {"600519": 90.0},
        hold_if_profitable_pct=5.0)
    sells = [i for i in intents if i.side == OrderSide.SELL and i.code == "600519"]
    assert sells, "亏损仓必须清仓，持有纪律不该保它"


def test_summarize_intents_reports_capped_count():
    """流动性封顶笔数应在预演摘要里明示，否则「计划 8000 股 / 实盘只买 3200」类
    不一致会被静默吞掉。"""
    intents = [
        RebalanceIntent(code="600519", name="x", side=OrderSide.BUY,
                        quantity=100, price=10.0, capped_by="liquidity"),
    ]
    txt = summarize_intents(intents)
    assert "流动性上限" in txt

# -*- coding: utf-8 -*-
"""排名滞后（hold_rank_slack）：老持仓只要还在带宽内就不换出。

**为什么要有它**（执行级回测，1140 天 × 200 只，OIHC 撮合）：

    15 天周期下         年化(10bp)  夏普  年换手 | 年化(50bp)  夏普
    slack = 1.0          24.7%     1.09  17.8x  |  14.7%     0.68
    slack = 3.0          26.8%     1.19  10.6x  |  21.1%     0.95

实盘原先只有「跌出目标组合即清仓」（等价 slack=1.0），于是换手是回测最优配置
的 1.7 倍。成本假设越差，缺口越大 —— 这不是「不止损」，它只影响**换出时机**。

参数来源：``TradePlan.raw_rank``（每日候选池的评分名次，必须在截断成 top_n
**之前**生成，否则 9~(top_n×slack) 名信息永久丢失）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.trading.broker import AccountInfo, OrderSide, Position
from src.trading.rebalance import compute_rebalance_orders
from src.trading.trade_plan import PlanTarget, TradePlan

DAY = "2026-11-02"
PRICES = {"600000": 10.0, "600519": 10.0, "000001": 10.0, "300750": 10.0}


def mk_pos(code, qty, price):
    return Position(
        code=code, name=code, quantity=qty, available=qty,
        avg_cost=price, current_price=price, market_value=qty * price,
        pnl=0.0, pnl_pct=0.0, today_quantity=0,
    )


def mk_acct(total=1_000_000.0, cash=400_000.0):
    return AccountInfo(
        account_id="T", total_asset=total, available_cash=cash,
        frozen_cash=0.0, market_value=total - cash, total_pnl=0.0, today_pnl=0.0,
    )


def mk_plan(codes, reserve=0.10, raw_rank=None, max_positions=8):
    w = (1.0 - reserve) / max(len(codes), 1)
    return TradePlan(
        plan_date=DAY,
        targets=[PlanTarget(code=c, name=c, weight=w) for c in codes],
        cash_reserve_pct=reserve,
        max_positions=max_positions,
        raw_rank=dict(raw_rank or {}),
    )


# ---------------------------------------------------------------- 关闭态

def test_slack_off_behaves_like_before():
    """slack=1.0（默认）⇒ 跌出目标组合就清仓，与旧行为逐字节一致。"""
    plan = mk_plan(["600519"], raw_rank={"600000": 12})   # 老仓在第 12 名
    intents = compute_rebalance_orders(
        plan, [mk_pos("600000", 5000, 10.0)], mk_acct(),
        PRICES, hold_rank_slack=1.0,
    )
    sells = [i for i in intents if i.side == OrderSide.SELL]
    assert [s.code for s in sells] == ["600000"]
    assert "跌出目标组合" in sells[0].reason


def test_no_raw_rank_degrades_to_off():
    """计划里没有 raw_rank ⇒ 自动退化为关闭，绝不能误判成「保留」。"""
    plan = mk_plan(["600519"])          # raw_rank 为空
    intents = compute_rebalance_orders(
        plan, [mk_pos("600000", 5000, 10.0)], mk_acct(),
        PRICES, hold_rank_slack=3.0,
    )
    sells = [i for i in intents if i.side == OrderSide.SELL]
    assert [s.code for s in sells] == ["600000"]


# ---------------------------------------------------------------- 开启态

def test_keeps_position_inside_rank_band():
    """slack=3.0、cap=8 ⇒ 保留线 24 名。第 12 名的老仓应被保留，不产生委托。"""
    plan = mk_plan(["600519"], raw_rank={"600000": 12})
    intents = compute_rebalance_orders(
        plan, [mk_pos("600000", 5000, 10.0)], mk_acct(),
        PRICES, hold_rank_slack=3.0,
    )
    assert not [i for i in intents if i.code == "600000"], \
        "带宽内的老仓不该出现任何委托（既不卖也不再买）"


def test_evicts_position_outside_rank_band():
    """第 30 名已在保留线（24）之外 ⇒ 照常清仓。"""
    plan = mk_plan(["600519"], raw_rank={"600000": 30})
    intents = compute_rebalance_orders(
        plan, [mk_pos("600000", 5000, 10.0)], mk_acct(),
        PRICES, hold_rank_slack=3.0,
    )
    sells = [i for i in intents if i.side == OrderSide.SELL]
    assert [s.code for s in sells] == ["600000"]


def test_band_boundary_is_exclusive():
    """名次 == 保留线时应当卖出（回调用 rank < keep_rank，严格小于）。"""
    plan = mk_plan(["600519"], raw_rank={"600000": 24}, max_positions=8)
    intents = compute_rebalance_orders(
        plan, [mk_pos("600000", 5000, 10.0)], mk_acct(),
        PRICES, hold_rank_slack=3.0,
    )
    sells = [i for i in intents if i.side == OrderSide.SELL]
    assert [s.code for s in sells] == ["600000"]


def test_kept_position_does_not_consume_buy_budget():
    """保留仓位不产生买单 —— 它的目标权重 = 当前权重，偏离为 0。"""
    plan = mk_plan(["600519"], raw_rank={"600000": 12})
    intents = compute_rebalance_orders(
        plan, [mk_pos("600000", 5000, 10.0)], mk_acct(),
        PRICES, hold_rank_slack=3.0,
    )
    assert not [i for i in intents if i.code == "600000"]
    # 新票仍应被买入
    assert [i.code for i in intents if i.side == OrderSide.BUY] == ["600519"]


def test_kept_count_capped_by_max_positions():
    """带宽内老仓多于 cap 时按名次截断，名次最差的先被换出。"""
    codes = ("600000", "000001", "300750")
    # cap=2、slack=10 ⇒ 保留线 20 名，三只（9/10/11）都在带内，但只留 2 只
    plan = mk_plan(["600519"], raw_rank={
        "600000": 9, "000001": 10, "300750": 11,
    }, max_positions=2)
    intents = compute_rebalance_orders(
        plan, [mk_pos(c, 1000, 10.0) for c in codes],
        mk_acct(), PRICES, hold_rank_slack=10.0,
    )
    sold = {i.code for i in intents if i.side == OrderSide.SELL} & set(codes)
    kept = set(codes) - sold
    assert len(kept) == 2, "保留数必须被 max_positions 截断"
    assert kept == {"600000", "000001"}, "名次最好的两只留下"
    assert "300750" in sold, "名次最差的那只必须先被换出"


# ---------------------------------------------------------------- 计划层

def test_plan_rank_map_keeps_truncated_candidates():
    """排名表必须在截断成 top_n **之前**生成，否则 9~24 名信息永久丢失。"""
    from src.trading.trade_plan import build_trade_plan

    picks = [{"code": f"S{i:02d}", "name": f"N{i}", "score": 100 - i}
             for i in range(12)]
    plan = build_trade_plan(picks=picks, top_n=3, max_positions=3,
                            weighting="equal", write=False)

    assert len(plan.targets) == 3, "目标只数照旧受 top_n 约束"
    assert len(plan.raw_rank) == 12, "排名表应覆盖整个候选池"
    assert plan.raw_rank["S00"] == 0
    assert plan.raw_rank["S11"] == 11


def test_plan_rank_map_survives_serialization():
    """计划要落盘再读回（隔日使用），raw_rank 不能丢。"""
    from src.trading.trade_plan import build_trade_plan

    picks = [{"code": f"S{i:02d}", "name": f"N{i}", "score": 100 - i}
             for i in range(10)]
    plan = build_trade_plan(picks=picks, top_n=3, max_positions=3,
                            weighting="equal", write=False)

    again = TradePlan.from_dict(plan.to_dict())

    assert again.raw_rank == plan.raw_rank
    assert again.raw_rank["S07"] == 7

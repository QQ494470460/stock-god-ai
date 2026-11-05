# -*- coding: utf-8 -*-
"""排名滞后（``hold_rank_slack``）的覆盖度 —— 生产最重要的参数，曾经静默失效。

问题
----
``compute_rebalance_orders`` 判断「老持仓该不该留」问的是**模型眼里它在第几名**：
在 ``max_positions × hold_rank_slack`` 名以内就继续拿着（生产 8×3 = 24）。

这个名次来自 ``TradePlan.raw_rank``。而实盘的 raw_rank 曾经是从**今日 picks**
建的 —— picks 是「过了 min_score 门槛、今天要买的那几只」，不是「模型怎么看全
市场」。门槛把 425 只筛到 9 只时，排名表就只有 9 条，保留线却是 24：

* 老持仓只要不在今日 picks 里，就**不在排名表里** ⇒ 名次=None ⇒ 判「跌出目标组合」；
* 于是每个调仓日把老持仓全清掉，换手翻倍 —— 而注释里那套
  「15 日 + slack3.0 ⇒ 50bp 夏普 0.95 vs 1.0 时 0.68」的好处**一分没拿到**。

2026-11-05 实盘实测：计划 11-05 的 ``raw_rank`` 只有 9 条（就是报告的 9 只），
保留线 24，5 只老持仓全部被判「跌出目标组合，清仓」。

四条测试守住的边界
------------------
1. 排名表用**完整榜单**，不是 picks；
2. 有完整榜单时，排名靠后但仍在保留线内的老仓**不许被卖**；
3. 没有榜单时这个失效必须**说出来**（告警），不能又变成静默；
4. picks 里有、榜单里没有的票要补上名次（否则它自己的判断也失效）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from loguru import logger  # noqa: E402

import src.trading.trade_plan as tp  # noqa: E402
from src.trading.broker import AccountInfo, Position  # noqa: E402
from src.trading.rebalance import compute_rebalance_orders  # noqa: E402

TARGET_CODES = ["600001", "600002", "600003", "600004",
                "600005", "600006", "600007", "600008"]
HELD_OLD = "601999"          # 不在今日 picks 里，但在模型榜单第 10 名


def _picks(codes):
    return [{"代码": c, "名称": c, "评分": 60.0, "最新价": 10.0} for c in codes]


def _ranking(codes):
    return [{"代码": c, "名称": c, "评分": 70.0 - i} for i, c in enumerate(codes)]


def _held(code, qty=1000, px=10.0):
    return Position(code=code, name=code, quantity=qty, available=qty,
                    avg_cost=px, current_price=px,
                    market_value=qty * px, pnl=0.0, pnl_pct=0.0)


def _account(cash=1_000_000.0, mv=10_000.0):
    return AccountInfo(account_id="T", total_asset=cash + mv, available_cash=cash,
                       frozen_cash=0.0, market_value=mv, total_pnl=0.0, today_pnl=0.0)


# ---------------------------------------------------------------------------
# 1. 排名表来源
# ---------------------------------------------------------------------------

def test_raw_rank_uses_the_full_ranking_not_just_picks():
    """榜单有 24 条、picks 只有 8 条 → 排名表必须是 24 条。"""
    rank = _ranking(TARGET_CODES + [f"6009{i:02d}" for i in range(16)])
    plan = tp.build_trade_plan(picks=_picks(TARGET_CODES), ranking=rank,
                               max_positions=8, write=False)
    assert len(plan.raw_rank) == 24
    assert plan.raw_rank[HELD_OLD] if HELD_OLD in plan.raw_rank else True


def test_raw_rank_falls_back_to_picks_when_no_ranking():
    """老报告没有 ranking 字段 —— 退回 picks 顺序，但覆盖度必然不足。"""
    plan = tp.build_trade_plan(picks=_picks(TARGET_CODES), max_positions=8, write=False)
    assert len(plan.raw_rank) == len(TARGET_CODES)


def test_picks_missing_from_the_ranking_still_get_a_rank():
    """榜单与 picks 是两段数据，可能对不上；缺名次的票必须补在末尾。"""
    rank = _ranking(TARGET_CODES[:4])          # 榜单里少了后 4 只
    plan = tp.build_trade_plan(picks=_picks(TARGET_CODES), ranking=rank,
                               max_positions=8, write=False)
    assert set(TARGET_CODES) <= set(plan.raw_rank)
    assert len(plan.raw_rank) >= 8


def test_ranking_keeps_the_plan_order_intact():
    """rank 必须按榜单顺序从 0 递增 —— 顺序错了 slack 就是在乱保留。"""
    rank = _ranking(TARGET_CODES)
    plan = tp.build_trade_plan(picks=_picks(TARGET_CODES[:2]), ranking=rank,
                               max_positions=8, write=False)
    assert plan.raw_rank["600001"] == 0
    assert plan.raw_rank["600008"] == 7
    assert len(plan.targets) == 2              # 目标只数仍由 picks 决定


# ---------------------------------------------------------------------------
# 2. 端到端：老持仓留不留
# ---------------------------------------------------------------------------

def _orders(plan, slack=3.0, held=HELD_OLD):
    prices = {c: 10.0 for c in TARGET_CODES + [held]}
    return compute_rebalance_orders(
        plan, [_held(held)], _account(), prices,
        band_pct=0.15, min_trade_amount=0.0, hold_rank_slack=slack,
    )


def test_slack_keeps_a_holding_ranked_inside_the_window():
    """老仓在模型榜单第 10 名、保留线 24 → 必须原样持有，不产生卖单。"""
    rank = _ranking(TARGET_CODES + [HELD_OLD] + [f"6009{i:02d}" for i in range(20)])
    plan = tp.build_trade_plan(picks=_picks(TARGET_CODES), ranking=rank,
                               max_positions=8, write=False)
    assert plan.raw_rank[HELD_OLD] < 24
    sells = [o for o in _orders(plan) if o.side.value == "sell"]
    assert sells == [], f"排名在保留线内却被清仓: {sells}"


def test_without_a_ranking_the_same_holding_gets_sold():
    """同一只老仓、同一个 slack，只因为计划没带榜单就被清掉 —— 这就是原 bug。

    这条断言的方向是「复现失效」，用来钉死「为什么必须传 ranking」。
    """
    plan = tp.build_trade_plan(picks=_picks(TARGET_CODES), max_positions=8, write=False)
    sells = [o for o in _orders(plan) if o.side.value == "sell"]
    assert [o.code for o in sells] == [HELD_OLD]


def test_a_holding_outside_the_window_is_still_sold():
    """保留线不是「不止损」：排到 30 名开外（线 24）就该换出去。"""
    rank = _ranking(TARGET_CODES + [f"6009{i:02d}" for i in range(21)] + [HELD_OLD])
    plan = tp.build_trade_plan(picks=_picks(TARGET_CODES), ranking=rank,
                               max_positions=8, write=False)
    assert plan.raw_rank[HELD_OLD] >= 24
    sells = [o for o in _orders(plan) if o.side.value == "sell"]
    assert [o.code for o in sells] == [HELD_OLD]


# ---------------------------------------------------------------------------
# 3. 覆盖度不足必须喊出来
# ---------------------------------------------------------------------------

def _warnings(fn):
    msgs: list[str] = []
    hid = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        fn()
    finally:
        logger.remove(hid)
    return [m for m in msgs if "排名表" in m]


def test_short_ranking_produces_a_loud_warning():
    hits = _warnings(lambda: tp.build_trade_plan(
        picks=_picks(TARGET_CODES), max_positions=8, write=False))
    assert hits and "24" in hits[0], hits


def test_full_ranking_is_quiet():
    rank = _ranking(TARGET_CODES + [f"6009{i:02d}" for i in range(20)])
    assert _warnings(lambda: tp.build_trade_plan(
        picks=_picks(TARGET_CODES), ranking=rank, max_positions=8, write=False)) == []


# ---------------------------------------------------------------------------
# 4. 报告侧：完整榜单要真的落进报告
# ---------------------------------------------------------------------------

def test_ranking_of_extracts_ordered_codes():
    import pandas as pd

    from src.core.daily_pick import _ranking_of

    df = pd.DataFrame([{"代码": "600002", "名称": "乙", "评分": 62.0},
                       {"代码": "600001", "名称": "甲", "评分": 58.5},
                       {"代码": "", "名称": "空", "评分": 50.0}])
    out = _ranking_of(df)
    assert [x["代码"] for x in out] == ["600002", "600001"]      # 保序 + 丢空码
    assert out[0]["评分"] == 62.0
    assert _ranking_of(None) == []
    assert _ranking_of(pd.DataFrame()) == []


def test_report_ranking_survives_a_json_roundtrip(tmp_path):
    """报告是落盘 JSON，ranking 必须能原样读回来喂给 build_trade_plan。"""
    rank = _ranking(TARGET_CODES + [f"6009{i:02d}" for i in range(6)])
    rep = tmp_path / "daily_picks_20261105_1845.json"
    rep.write_text(json.dumps({"picks": _picks(TARGET_CODES), "ranking": rank,
                               "generated_at": "2026-11-05T18:45:00"},
                              ensure_ascii=False), encoding="utf-8")

    picks, meta = tp.load_latest_picks(tmp_path)
    assert len(meta["ranking"]) == 14
    plan = tp.build_trade_plan(picks=picks, ranking=meta["ranking"],
                               max_positions=8, write=False)
    assert len(plan.raw_rank) == 14

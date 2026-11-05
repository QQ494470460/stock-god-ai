# -*- coding: utf-8 -*-
"""``generate_daily_picks`` 必须产出**完整宽榜单**的 ranking —— 这是 11-02 / 11-05
「5 只老持仓被一次性清仓」的真正根因所在。

之前的单测（test_rank_slack_coverage）只测了 *下游*：``build_trade_plan(ranking=...)``
确实会用完整榜单保留老仓。但**上游**—— ``generate_daily_picks`` 到底有没有真的把
完整榜单塞进 ``report["ranking"]``—— 一直没被测。如果那天那个字段没写进去，
下游再对也没用：计划拿到的还是只有 9 只 picks 的残缺排名表。

本文件补上这块拼图，且把两端用**同一份真实产出的 ranking** 串起来：
从 ``generate_daily_picks`` 拿到的 ranking 直接喂给 ``build_trade_plan``，
证明「排在第 20 名（在 picks 之外、保留线之内）的老持仓」不再被清。

全程 mock，不碰网络、不写生产状态。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.core.daily_pick as dp  # noqa: E402
import src.trading.trade_plan as tp  # noqa: E402
from src.trading.broker import AccountInfo, Position  # noqa: E402
from src.trading.rebalance import compute_rebalance_orders  # noqa: E402


# ---------------------------------------------------------------------------
# Mock 依赖
# ---------------------------------------------------------------------------

class _MockFetcher:
    def __init__(self, n=50, wide=40):
        self._codes = [f"600{i:03d}" for i in range(1, n + 1)]
        self._wide = wide

    def get_stock_pool(self, pool="all"):
        return list(self._codes)

    def get_batch_daily_kline(self, codes, max_workers=8):
        out = {}
        for c in codes:
            out[c] = pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=120, freq="D"),
                "close": [10.0 + i * 0.01 for i in range(120)],
            })
        return out

    def get_stock_list(self, force_refresh=False):
        return pd.DataFrame({
            "code": self._codes,
            "name": [f"股{c}" for c in self._codes],
        })

    def get_benchmark_kline(self, *a, **kw):
        # regime 检测失败无所谓（被 try 包住），返回空让它自然走降级
        return pd.DataFrame()


class _MockPredictor:
    def __init__(self, wide=40):
        self._wide = wide

    def get_top_recommendations(self, stock_data, stock_names=None,
                                top_n=20, min_score=50.0):
        # 前 9 只高分（过门槛）、后 31 只低分（不过门槛但仍在模型榜单里）
        rows = []
        for i in range(self._wide):
            score = 60.0 if i < 9 else 40.0
            rows.append({"代码": f"600{i+1:03d}", "名称": f"股{i+1:03d}", "评分": float(score)})
        return pd.DataFrame(rows[:top_n])


def _gen(wide=40, min_score=55.0):
    """跑一次（mock）选股，返回 (df, report)。"""
    fet = _MockFetcher(wide=wide)
    pred = _MockPredictor(wide=wide)
    df, rep = dp.generate_daily_picks(
        fetcher_inst=fet, predictor_inst=pred,
        min_score=min_score, write_report=False,
    )
    return df, rep


# ---------------------------------------------------------------------------
# 上游：generate_daily_picks 必须吐出完整宽榜单
# ---------------------------------------------------------------------------

def test_ranking_is_the_full_wide_board_not_the_thresholded_picks():
    """ranking 宽度 = wide（40），picks 宽度 = 过门槛的 9 只。两者必须不同。"""
    _, rep = _gen(wide=40)
    assert rep["status"] == "ok"
    assert "ranking" in rep, "报告缺 ranking 字段 —— 9e2bb94 修复没接上"
    # 关键不变量：ranking 是完整榜单，picks 是筛过的子集
    assert len(rep["ranking"]) == 40, f"ranking 宽度应为 40，实际 {len(rep['ranking'])}"
    assert len(rep["picks"]) == 9, f"picks 应为 9（过门槛），实际 {len(rep['picks'])}"
    rank_codes = {r["代码"] for r in rep["ranking"]}
    pick_codes = {p["代码"] for p in rep["picks"]}
    assert pick_codes <= rank_codes, "picks 必须全是 ranking 的子集"
    # ranking 必须覆盖保留线（生产 8 × 3 = 24）
    assert len(rep["ranking"]) >= 8 * 3, "ranking 太窄，够不到保留线"


def test_ranking_entries_are_well_formed():
    _, rep = _gen(wide=40)
    for r in rep["ranking"]:
        assert "代码" in r and "名称" in r and "评分" in r
        assert isinstance(r["评分"], (int, float))


def test_short_wide_board_still_reports_ranking():
    """即使 wide 给得小（如 12），ranking 也要等于 wide、且区别于 picks。"""
    _, rep = _gen(wide=12)
    assert len(rep["ranking"]) == 12
    assert 0 < len(rep["picks"]) <= 12


# ---------------------------------------------------------------------------
# 端到端：把真实产出的 ranking 喂回 build_trade_plan，老持仓必须保住
# ---------------------------------------------------------------------------

def _held(code, qty=1000, px=10.0):
    return Position(code=code, name=code, quantity=qty, available=qty,
                    avg_cost=px, current_price=px,
                    market_value=qty * px, pnl=0.0, pnl_pct=0.0)


def _account(cash=1_000_000.0, mv=10_000.0):
    return AccountInfo(account_id="T", total_asset=cash + mv, available_cash=cash,
                       frozen_cash=0.0, market_value=mv, total_pnl=0.0, today_pnl=0.0)


def test_e2e_real_ranking_preserves_an_old_holding_outside_picks():
    """从 generate_daily_picks 真实产出的 ranking 里取一只「在榜单第 20 名、
    不在 picks 里」的票当老持仓：slack=3.0 时保留线 24，它必须原样持有。

    这正是 11-05 现场重演的修复：旧代码 ranking 只有 9 条 → 这只票名次 None
    → 被判「跌出目标组合」清仓；新代码 ranking 有 40 条 → 名次 20 < 24 → 保留。
    """
    _, rep = _gen(wide=40)
    held_code = rep["ranking"][20]["代码"]          # 第 20 名，分数 40，不在 9 只 picks 里
    assert held_code not in {p["代码"] for p in rep["picks"]}

    plan = tp.build_trade_plan(
        picks=rep["picks"], ranking=rep["ranking"],
        max_positions=8, write=False,
    )
    assert plan.raw_rank[held_code] == 20
    assert plan.raw_rank[held_code] < 24

    prices = {c: 10.0 for c in [p["代码"] for p in rep["picks"]] + [held_code]}
    orders = compute_rebalance_orders(
        plan, [_held(held_code)], _account(), prices,
        band_pct=0.15, min_trade_amount=0.0, hold_rank_slack=3.0,
    )
    sells = [o for o in orders if getattr(o.side, "value", str(o.side)) == "sell"]
    assert [o.code for o in sells] == [], f"排名在保留线内却被清仓: {sells}"


def test_e2e_real_ranking_still_sells_a_holding_beyond_the_window():
    """对照：取一只排到第 30 名开外（保留线 24）的老仓，仍应换出 —— 证明
    ranking 不是「一律保留」，只是把判断权交回模型排名。"""
    _, rep = _gen(wide=40)
    # 榜单只有 40 条，取最后一只（第 39 名）当老持仓
    held_code = rep["ranking"][39]["代码"]
    plan = tp.build_trade_plan(
        picks=rep["picks"], ranking=rep["ranking"],
        max_positions=8, write=False,
    )
    assert plan.raw_rank[held_code] >= 24
    prices = {c: 10.0 for c in [p["代码"] for p in rep["picks"]] + [held_code]}
    orders = compute_rebalance_orders(
        plan, [_held(held_code)], _account(), prices,
        band_pct=0.15, min_trade_amount=0.0, hold_rank_slack=3.0,
    )
    sells = [o for o in orders if getattr(o.side, "value", str(o.side)) == "sell"]
    assert [o.code for o in sells] == [held_code]

# -*- coding: utf-8 -*-
"""执行层智能下单（T2）单测：切片 / 重试 / 隔夜窗口 / 调度滴灌 / 滑点 / 默认关闭。"""
from datetime import date, datetime

import pytest

from src.trading.broker import OrderSide
from src.trading.rebalance import RebalanceIntent
from src.trading.execution.slice import twap_slices, vwap_slices, choose_n_slices, LOT_SIZE
from src.trading.execution.retry import LimitRetry
from src.trading.execution.overnight import schedule_window, preopen_release_dt
from src.trading.execution.scheduler import ExecutionScheduler
from src.trading.execution.slippage import (
    SimSlippageModel, set_active_slippage_model, get_active_slippage_model,
)


def _intent(code="600000", qty=1000, price=10.0, side=OrderSide.BUY):
    return RebalanceIntent(
        code=code, name=code, side=side, quantity=qty, price=price,
        reason="test", target_weight=0.1,
    )


# ---------------- 切片 ----------------
def test_twap_slices_sum_and_lot():
    q = twap_slices(1000, 5)
    assert len(q) == 5
    assert sum(q) == 1000
    assert all(x % LOT_SIZE == 0 for x in q)
    # 余数补到前几片
    q2 = twap_slices(1050, 4)
    assert sum(q2) == 1000 or sum(q2) == 1050  # 整手对齐后可能损失零散
    assert all(x % LOT_SIZE == 0 for x in q2)


def test_choose_n_slices_respects_lot():
    # 总量仅 300 股，最多 10 片 → 每片不足 1 手，应下调
    n = choose_n_slices(300, 10)
    assert n == 3
    assert choose_n_slices(100, 10) == 1
    assert choose_n_slices(5000, 10) == 10


def test_vwap_slices_sum():
    prof = [1, 2, 3, 2, 1]
    q = vwap_slices(1000, prof)
    assert len(q) == 5
    assert abs(sum(q) - 1000) <= LOT_SIZE
    assert all(x % LOT_SIZE == 0 for x in q)
    # 权重大的片分到的量应不少于权重小的片
    assert q[2] >= q[0]


# ---------------- 重试 ----------------
def test_limit_retry_escalation():
    r = LimitRetry(max_attempts=2, step_bps=10.0, timeout_sec=60.0, escalate_to_market=True)
    # 首发：原价、限价
    p, mkt = r.next_price(10.0, "buy", now_ts=0.0)
    assert p == 10.0 and mkt is False
    # 未到超时：原价
    p, mkt = r.next_price(10.0, "buy", now_ts=30.0)
    assert p == 10.0 and mkt is False
    # 超时：买价上抬
    p, mkt = r.next_price(10.0, "buy", now_ts=61.0)
    assert p > 10.0 and mkt is False
    # 超过重试上限：转市价
    p, mkt = r.next_price(10.0, "buy", now_ts=200.0)
    assert mkt is True


def test_limit_retry_sell_lowers():
    r = LimitRetry(max_attempts=2, step_bps=10.0, timeout_sec=60.0)
    r.next_price(10.0, "sell", 0.0)
    p, _ = r.next_price(10.0, "sell", 61.0)
    assert p < 10.0


# ---------------- 隔夜窗口 ----------------
def test_schedule_window_overnight():
    d = date(2026, 11, 3)
    start, end = schedule_window(d, window_min=300, overnight=True)
    assert start == preopen_release_dt(d)          # 09:15
    assert start.hour == 9 and start.minute == 15
    # 末片不晚于 14:57
    assert end.hour <= 15


def test_schedule_window_no_overnight():
    d = date(2026, 11, 3)
    start, _ = schedule_window(d, window_min=300, overnight=False)
    assert start.hour == 9 and start.minute == 30


# ---------------- 调度器滴灌 ----------------
def test_scheduler_drip_and_fill():
    d = date(2026, 11, 3)
    sc = ExecutionScheduler(
        strategy="twap", window_min=300, max_slices=5, overnight=True,
        persist_path=None,
    )
    n = sc.enqueue([_intent(qty=1000)], d)
    assert n == 1 and not sc.is_done()

    # 09:10 早于首片(09:15) → 无
    assert sc.release_due(datetime(2026, 11, 3, 9, 10, 0)) == []
    # 09:15:01 → 首片
    due1 = sc.release_due(datetime(2026, 11, 3, 9, 15, 1))
    assert len(due1) == 1
    ro = due1[0]
    assert ro.code == "600000" and ro.qty == 200 and ro.side == OrderSide.BUY
    # 同刻再问：已 sent，不重复
    assert sc.release_due(datetime(2026, 11, 3, 9, 15, 2)) == []
    # 回报全成
    sc.on_fill(ro.slice_id, ro.qty, datetime(2026, 11, 3, 9, 15, 3))

    # 推进到收盘前，取剩余所有片
    collected = 0
    for hh, mm in [(10, 40), (12, 0), (13, 20), (14, 40)]:
        for r in sc.release_due(datetime(2026, 11, 3, hh, mm, 0)):
            sc.on_fill(r.slice_id, r.qty, datetime(2026, 11, 3, hh, mm, 0))
            collected += 1
    assert collected == 4
    assert sc.is_done()


def test_scheduler_partial_fill_retry_price_rises():
    d = date(2026, 11, 3)
    sc = ExecutionScheduler(strategy="twap", window_min=300, max_slices=2,
                            overnight=True, persist_path=None,
                            retry_cfg={"max_attempts": 3, "step_bps": 20.0, "timeout_sec": 60.0})
    sc.enqueue([_intent(qty=1000, price=10.0)], d)
    due = sc.release_due(datetime(2026, 11, 3, 9, 15, 1))
    ro = due[0]
    orig_price = ro.price
    # 部分成交（只成一半）→ 应抬价并保持待发
    sc.on_fill(ro.slice_id, ro.qty // 2, datetime(2026, 11, 3, 9, 15, 2))
    # 超时后重新放开，价格应上抬
    due2 = sc.release_due(datetime(2026, 11, 3, 9, 20, 0))
    assert due2, "超时后应重新发出"
    assert due2[0].price > orig_price


# ---------------- 滑点模型 ----------------
def test_slippage_buy_up_sell_down():
    m = SimSlippageModel(base_bps=2.0, participation=0.01)
    assert m.fill_price(100.0, "buy", 10000) > 100.0
    assert m.fill_price(100.0, "sell", 10000) < 100.0


def test_slippage_participation_impact_grows():
    adv = 1_000_000.0
    m = SimSlippageModel(base_bps=2.0, participation=0.01, impact_coef=0.5,
                         adv_provider=lambda code, amt: adv)
    small = m.fill_price(100.0, "buy", amount=10_000)      # 1% of adv
    large = m.fill_price(100.0, "buy", amount=500_000)     # 50% of adv
    assert large > small


def test_slippage_registry_and_rules_hook():
    from src.trading.rules import apply_slippage
    base = apply_slippage(100.0, "buy", 10000)
    m = SimSlippageModel(base_bps=50.0, participation=0.0)  # 明显更高的基础滑点
    set_active_slippage_model(m)
    try:
        hooked = apply_slippage(100.0, "buy", 10000)
        assert hooked == m.fill_price(100.0, "buy", 10000)
        assert hooked >= base
    finally:
        set_active_slippage_model(None)
    # 注销后回到原逻辑
    assert apply_slippage(100.0, "buy", 10000) == base


# ---------------- 默认关闭 ----------------
def test_loop_default_execution_is_off():
    from src.trading.loop import LoopConfig
    cfg = LoopConfig()
    assert getattr(cfg, "execution", None) is None  # 默认 None = 不启用智能下单

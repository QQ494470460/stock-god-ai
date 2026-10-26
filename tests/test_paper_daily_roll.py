# -*- coding: utf-8 -*-
"""跨日滚动：T+1 解锁必须在**读路径**上也能触发。

缺陷（真会静默吃掉止损）
------------------------
``_check_daily_reset`` 原本只写在 ``buy`` / ``sell`` 里。常驻闭环跨日后：

    guard._positions() → paper.get_positions() → today_quantity 仍是昨天的
      → sellable_quantity() = 数量 - 昨买量
      → 卖出在**信号预检**就被「T+1 锁仓」拦下
      → 被拦下就不会调用 sell()，解锁永远不触发
      → 昨天买的仓位**整天卖不掉**（含止损），直到某次买入把它唤醒

这类 bug 不会报错、不会告警，只会表现为「止损没执行」，是最难查的一类。

同时锁住零股规则：A 股卖出允许不足一手，但必须一次性卖光；
旧实现把卖出也强行整手取整，150 股只能卖 100，剩 50 股永久卡死。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading.paper import PaperAccount


def _mk(tmp_path, holdings, cash=100_000.0):
    pa = PaperAccount(
        initial_cash=cash,
        account_file=tmp_path / "acct.json",
        history_file=tmp_path / "hist.json",
    )
    pa.cash = cash
    pa.holdings = holdings
    pa._save()
    return pa


def _holding(qty=1000, avail=1000, today=0, cost=10.0, price=10.0):
    return {
        "name": "TEST", "quantity": qty, "available": avail,
        "avg_cost": cost, "last_price": price,
        "market_value": qty * price, "today_quantity": today,
    }


def _set_last_reset(pa: PaperAccount, when: str):
    pa._last_reset_date = when
    pa._save()


# ----------------------------------------------------------------------
# 跨日解锁
# ----------------------------------------------------------------------
def test_get_positions_rolls_day_and_unlocks_t1(tmp_path):
    pa = _mk(tmp_path, {"600000": _holding(qty=1000, avail=0, today=1000)})
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    _set_last_reset(pa, yesterday)

    # 读路径（guard 正是这么取持仓的）
    pos = pa.get_positions()[0]
    assert pos["today_quantity"] == 0, "跨日后昨买量必须清零"
    assert pos["available"] == 1000
    assert pos["sellable"] == 1000

    # 解锁必须落盘：另一个进程（看板）读到的也得是解锁后的状态
    saved = json.loads((tmp_path / "acct.json").read_text(encoding="utf-8"))
    assert saved["holdings"]["600000"]["today_quantity"] == 0
    assert saved["last_reset_date"] == datetime.now().strftime("%Y-%m-%d")


def test_mark_to_market_rolls_day_too(tmp_path):
    pa = _mk(tmp_path, {"600000": _holding(qty=1000, avail=0, today=1000)})
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    _set_last_reset(pa, yesterday)

    pa.mark_to_market({"600000": 11.0})
    assert pa.holdings["600000"]["today_quantity"] == 0
    assert pa.holdings["600000"]["market_value"] == pytest.approx(11_000.0)


def test_same_day_does_not_unlock(tmp_path):
    """当天买入的必须仍然锁定 —— 别把修复做成「解锁一切」。"""
    pa = _mk(tmp_path, {"600000": _holding(qty=1000, avail=0, today=1000)})
    pos = pa.get_positions()[0]
    assert pos["today_quantity"] == 1000
    assert pos["sellable"] == 0


def test_roll_is_cheap_noop_when_day_unchanged(tmp_path):
    pa = _mk(tmp_path, {"600000": _holding()})
    before = (tmp_path / "acct.json").read_text(encoding="utf-8")
    pa.get_positions()
    after = (tmp_path / "acct.json").read_text(encoding="utf-8")
    assert before == after, "同一天内读持仓不应改写磁盘"


# ----------------------------------------------------------------------
# 零股卖出
# ----------------------------------------------------------------------
def test_odd_lot_full_liquidation_is_allowed(tmp_path):
    pa = _mk(tmp_path, {"600000": _holding(qty=150, avail=150)})
    ok, msg, rec = pa.sell("600000", 10.0, 150)
    assert ok, msg
    assert rec.quantity == 150
    assert "600000" not in pa.holdings


def test_odd_lot_partial_sell_still_needs_lot(tmp_path):
    """持有 250 股、只卖 150 股 → 部分卖出必须整手（取整到 100）。"""
    pa = _mk(tmp_path, {"600000": _holding(qty=250, avail=250)})
    ok, msg, rec = pa.sell("600000", 10.0, 150)
    assert ok, msg
    assert rec.quantity == 100
    assert pa.holdings["600000"]["quantity"] == 150


def test_small_holding_below_one_lot_can_be_closed(tmp_path):
    """只有 30 股（送股零头）也必须能清掉，不能让仓位烂在账上。"""
    pa = _mk(tmp_path, {"600000": _holding(qty=30, avail=30)})
    ok, msg, rec = pa.sell("600000", 10.0, 30)
    assert ok, msg
    assert rec.quantity == 30
    assert "600000" not in pa.holdings


def test_non_lot_partial_below_100_still_rejected(tmp_path):
    pa = _mk(tmp_path, {"600000": _holding(qty=1000, avail=1000)})
    ok, msg, _ = pa.sell("600000", 10.0, 50)
    assert not ok
    assert "不足1手" in msg

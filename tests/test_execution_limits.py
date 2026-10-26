# -*- coding: utf-8 -*-
"""执行约束层：流动性（ADV 参与率）/ 单笔风险预算 / 当日换手预算。

背景：仓位比例只回答「最多放多少」，不回答「这笔能不能吃掉」。
纸交易默认「挂多少成交多少」，等于假设无限流动性；而固定 % 仓位也隐含了
「每笔潜在亏损相同」的假设，对高波动个股并不成立。这三道约束补齐这块。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import pytest

from src.trading.broker import (
    BrokerAPI, Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo, TickData,
)
from src.trading.liquidity import AdvProvider


# ----------------------------------------------------------------------
# ADV 估计
# ----------------------------------------------------------------------
def _write_parquet(path: Path, amounts) -> None:
    pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(amounts), freq="D"),
        "close": [10.0] * len(amounts),
        "volume": [1_000_000] * len(amounts),
        "amount": amounts,
    }).to_parquet(path)


def test_adv_amount_is_mean_of_recent_amount(tmp_path):
    _write_parquet(tmp_path / "daily_600000_qfq.parquet", [100.0, 200.0, 300.0, 400.0])
    p = AdvProvider(data_dir=tmp_path, lookback=2)
    assert p.adv_amount("600000") == pytest.approx(350.0)  # 最后两日均值


def test_adv_ignores_zero_and_missing_amount(tmp_path):
    _write_parquet(tmp_path / "daily_600001_qfq.parquet", [0.0, 500.0, 0.0, 500.0])
    p = AdvProvider(data_dir=tmp_path, lookback=5)
    assert p.adv_amount("600001") == pytest.approx(500.0)


def test_adv_missing_file_returns_none(tmp_path):
    p = AdvProvider(data_dir=tmp_path)
    assert p.adv_amount("999999") is None
    assert p.max_order_value("999999") is None
    assert p.max_order_quantity("999999", 10.0) is None


def test_max_order_quantity_is_lot_rounded(tmp_path):
    _write_parquet(tmp_path / "daily_600000_qfq.parquet", [1_000_000.0] * 5)
    p = AdvProvider(data_dir=tmp_path, lookback=5)
    # 1% × 100万 = 1万元；价格 33 → 303 股 → 整手 300
    assert p.max_order_quantity("600000", 33.0, participation=0.01) == 300


# ----------------------------------------------------------------------
# Guard 三道约束
# ----------------------------------------------------------------------
class _Broker(BrokerAPI):
    is_simulated = False

    def __init__(self, price=10.0, total_asset=1_000_000.0):
        super().__init__()
        self._connected = True
        self.price = price
        self.total_asset = total_asset
        self.orders = []

    def connect(self, **kw): self._connected = True; return True
    def disconnect(self): self._connected = False
    def is_trading_time(self): return True
    def query_positions(self): return []
    def query_orders(self, status_filter="all"): return []
    def cancel_order(self, order_id): return True
    def cancel_all(self): return 0
    def get_broker_name(self): return "fake"

    def query_account(self):
        return AccountInfo(
            account_id="FAKE", total_asset=self.total_asset,
            available_cash=self.total_asset * 0.6, frozen_cash=0.0,
            market_value=0.0, total_pnl=0.0, today_pnl=0.0,
        )

    def get_tick(self, code):
        return TickData(
            code=code, name=code, last_price=self.price, open=self.price,
            high=self.price, low=self.price, pre_close=self.price,
            volume=50_000_000, amount=self.price * 50_000_000,
            bid1=self.price - 0.01, ask1=self.price + 0.01,
        )

    def place_order(self, code, side, order_type, price, quantity, strategy_tag=""):
        o = Order(order_id=f"O{len(self.orders)}", code=code, name=code, side=side,
                  order_type=order_type, price=price, quantity=quantity,
                  filled_quantity=quantity, status=OrderStatus.FILLED,
                  strategy_tag=strategy_tag)
        self.orders.append(o)
        return o


def _guard(tmp_path, adv=None, **cfg_kw):
    from src.trading.guard import GuardConfig, RiskSupervisor
    cfg = GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
        **cfg_kw,
    )
    return RiskSupervisor(
        _Broker(), cfg,
        industry_map={},
        adv_provider=(lambda code: adv),
    )


def _account(total=1_000_000.0):
    return AccountInfo(account_id="P", total_asset=total, available_cash=total,
                       frozen_cash=0.0, market_value=0.0, total_pnl=0.0, today_pnl=0.0)


def test_participation_cap_limits_buy_quantity(tmp_path):
    g = _guard(tmp_path, adv=1_000_000.0, max_participation_rate=0.01)
    qty, why = g._apply_execution_limits("600000", OrderSide.BUY, 10.0, 5000, _account())
    # 1% × 100万 = 1万元 → 1000 股
    assert qty == 1000
    assert "流动性" in why


def test_risk_budget_caps_position_by_stop_distance(tmp_path):
    g = _guard(tmp_path, adv=1e9, max_participation_rate=0.01,
               max_risk_per_trade_pct=0.005, risk_budget_stop_pct=0.05)
    qty, why = g._apply_execution_limits("600000", OrderSide.BUY, 10.0, 20_000, _account())
    # 0.5% × 100万 = 5000 元风险；止损 5% → 最多 10000 股
    assert qty == 10_000
    assert "风险预算" in why


def test_risk_budget_does_not_cap_sells(tmp_path):
    g = _guard(tmp_path, adv=1e9, max_risk_per_trade_pct=0.005)
    qty, _why = g._apply_execution_limits("600000", OrderSide.SELL, 10.0, 20_000, _account())
    assert qty == 20_000, "卖出是减风险，不该被风险预算卡住"


def test_daily_turnover_budget_caps_and_blocks(tmp_path):
    g = _guard(tmp_path, adv=1e9, max_daily_turnover_pct=0.30)
    # 预算 30 万，已买 29.5 万 → 只剩 5000 元 → 500 股
    g._turnover_amount = 295_000.0
    g._turnover_day = __import__("datetime").date.today().isoformat()
    qty, why = g._apply_execution_limits("600000", OrderSide.BUY, 10.0, 5000, _account())
    assert qty == 500
    assert "换手预算" in why

    g._turnover_amount = 300_000.0
    qty2, why2 = g._apply_execution_limits("600000", OrderSide.BUY, 10.0, 5000, _account())
    assert qty2 == 0 and "已用尽" in why2


def test_limits_can_be_disabled(tmp_path):
    g = _guard(tmp_path, adv=1.0, execution_limits_enabled=False)
    qty, why = g._apply_execution_limits("600000", OrderSide.BUY, 10.0, 5000, _account())
    assert qty == 5000 and why == ""


def test_adv_unavailable_does_not_block_trading(tmp_path):
    g = _guard(tmp_path, adv=None)
    qty, why = g._apply_execution_limits("600000", OrderSide.BUY, 10.0, 3000, _account())
    assert qty == 3000, "ADV 缺失应只跳过流动性约束，不能拦交易"


def test_propose_trade_applies_participation_cap(tmp_path):
    g = _guard(tmp_path, adv=1_000_000.0, max_participation_rate=0.01)
    prop = g.propose_trade(
        code="600000", name="600000", side=OrderSide.BUY,
        ai_score=90.0, agreement=1.0, strategy_tag="ai",
        explicit_quantity=5000,
    )
    assert prop.status != "blocked", prop.reason
    assert prop.quantity == 1000

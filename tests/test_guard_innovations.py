# -*- coding: utf-8 -*-
"""守护侧创新单元测试：市场状态动态加权 + 熔断自诊断/智能自愈。

仅用 Fake 对象，零资金、零网络。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from src.trading.broker import (
    BrokerAPI, AccountInfo, Order, OrderSide, OrderStatus, Position, TickData,
)
from src.trading.guard import GuardConfig, RiskSupervisor


class FakeBroker(BrokerAPI):
    is_simulated = False

    def __init__(self, total_asset=1_000_000.0, last_price=10.0):
        super().__init__()
        self._connected = True
        self.total_asset = total_asset
        self.available_cash = total_asset * 0.6
        self.last_price = last_price

    def connect(self, **kwargs):
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def query_account(self):
        return AccountInfo(
            account_id="FAKE", total_asset=self.total_asset,
            available_cash=self.available_cash, frozen_cash=0.0,
            market_value=self.total_asset - self.available_cash,
            total_pnl=0.0, today_pnl=0.0,
        )

    def query_positions(self):
        return []

    def query_orders(self, status_filter="all"):
        return []

    def place_order(self, code, side, order_type, price, quantity, strategy_tag=""):
        return Order(
            order_id="O1", code=code, side=side, order_type=order_type,
            price=price, quantity=quantity, filled_quantity=quantity,
            status=OrderStatus.FILLED, strategy_tag=strategy_tag,
        )

    def cancel_order(self, order_id):
        return True

    def cancel_all(self):
        return 0

    def get_tick(self, code):
        return TickData(
            code=code, name=code, last_price=self.last_price, open=10.0,
            high=10.5, low=9.5, pre_close=10.0, volume=100000, amount=1e6,
            bid1=self.last_price - 0.01, ask1=self.last_price + 0.01,
            change_pct=0.0,
        )

    def get_broker_name(self):
        return "FakeBroker"

    def is_trading_time(self):
        return True


class FakePaper:
    def __init__(self, total=1_000_000.0):
        self.cash = total
        self.frozen_cash = 0.0
        self.holdings = {}
        self.trades = 0
        self.today_pnl = 0.0

    def get_total_asset(self):
        mv = sum(h["qty"] * h["last"] for h in self.holdings.values())
        return self.cash + mv

    def get_total_pnl(self):
        return 0.0

    def mark_to_market(self, prices):
        for code, price in prices.items():
            if code in self.holdings:
                self.holdings[code]["last"] = price

    def get_positions(self):
        return []

    def buy(self, code, name, price, quantity, strategy="", ai_score=0.0):
        cost = price * quantity
        if self.cash < cost:
            return False, "资金不足", None
        self.cash -= cost
        h = self.holdings.setdefault(code, {"qty": 0, "cost": 0.0, "last": price})
        h["qty"] += quantity
        h["cost"] = price
        h["last"] = price
        self.trades += 1
        return True, "ok", None

    def sell(self, code, price, quantity, strategy="", ai_score=0.0):
        if code not in self.holdings or self.holdings[code]["qty"] < quantity:
            return False, "无持仓", None
        h = self.holdings[code]
        self.cash += price * quantity
        h["qty"] -= quantity
        if h["qty"] <= 0:
            del self.holdings[code]
        self.trades += 1
        return True, "ok", None


@pytest.fixture
def sup(tmp_path):
    cfg = GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
        regime_weighting_enabled=True,
        auto_resume_risk_fraction=0.5,
        auto_resume_throttle_sec=1800,
    )
    return RiskSupervisor(FakeBroker(), config=cfg, paper_account=FakePaper())


# ----------------------------------------------------------------------
# 市场状态动态加权（Feature #4）
# ----------------------------------------------------------------------
def test_regime_amplifies_momentum_in_bull(sup):
    sup.regime.set_regime("bull")
    s = sup._adjust_signal_strength("momentum", 1.0)
    assert s > 1.0  # 牛市放大进攻


def test_risk_exit_tag_never_scaled(sup):
    sup.regime.set_regime("bull")
    assert sup._adjust_signal_strength("stop", 1.0) == 1.0


def test_regime_disabled_returns_base(sup):
    sup.regime.enabled = False
    sup.regime.set_regime("bull")
    assert sup._adjust_signal_strength("momentum", 0.8) == 0.8


def test_signal_strength_clamped(sup):
    sup.regime.set_regime("bull")
    # 极端输入也不应越界
    assert 0.0 <= sup._adjust_signal_strength("loop", 5.0) <= 2.0
    assert sup._adjust_signal_strength("loop", -3.0) == 0.0


# ----------------------------------------------------------------------
# 熔断自诊断 / 智能自愈（Feature #5）
# ----------------------------------------------------------------------
def test_heal_condition_cleared_when_recovered(sup):
    sup.safety._state.peak_equity = 1_000_000.0
    sup.safety._state.today_start_equity = 1_000_000.0
    # 权益已回到峰值、当日无亏损 → 条件清除
    assert sup._heal_condition_cleared(1_000_000.0) is True


def test_heal_condition_not_cleared_on_drawdown(sup):
    sup.safety._state.peak_equity = 1_000_000.0
    sup.safety._state.today_start_equity = 1_000_000.0
    # 回撤 10% ≥ 8% → 未清除
    assert sup._heal_condition_cleared(900_000.0) is False


def test_heal_condition_not_cleared_on_daily_loss(sup):
    sup.safety._state.peak_equity = 1_000_000.0
    sup.safety._state.today_start_equity = 1_000_000.0
    # 当日亏损 3% ≤ -3% → 未清除
    assert sup._heal_condition_cleared(970_000.0) is False


def test_heal_throttle_reduces_signal_strength(sup):
    sup.regime.enabled = False  # 隔离：只看冷静期折扣
    sup._heal_throttle_until = time.time() + 100
    s = sup._adjust_signal_strength("loop", 1.0)
    assert abs(s - 0.5) < 1e-6  # auto_resume_risk_fraction = 0.5


def test_heal_throttle_expires(sup):
    sup.regime.enabled = False
    sup._heal_throttle_until = time.time() - 10  # 已过期
    assert sup._adjust_signal_strength("loop", 1.0) == 1.0

# -*- coding: utf-8 -*-
"""止盈止损 / 计划股数 的回归测试。

覆盖三类真实缺陷（均曾在评审中发现）：

1. **移动止损参照高点污染持仓对象**：旧实现把历史最高价写回
   ``Position.current_price``，等于篡改调用方的市值/盈亏；且已清仓的标的
   不会从高点表剔除 —— 同一只票卖出后再买回，建仓当天就被上一轮峰值
   判成「回撤」而直接卖出。
2. **止损卖出量未过 T+1**：用 ``pos.available`` 直接下单，实盘会废单。
3. **LIVE 执行丢弃计划股数**：``execute_ai_signal`` 按自己的通用仓位算法
   定量，导致「部分止盈」被执行成整仓清仓。
4. **AutoTrader 止盈止损绕开 guard**：直接向券商下单（PAPER 下还会绕开
   纸账户，账实分裂）。

全部内存替身，零网络、零真实资金。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading.broker import (
    BrokerAPI, Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo, TickData,
)
from src.trading.executor import TradeExecutor, RiskConfig


def _pos(code="600000", avg_cost=10.0, qty=1000, available=None,
         price=10.0, today=0) -> Position:
    avail = qty if available is None else available
    return Position(
        code=code, name=code, quantity=qty, available=avail,
        avg_cost=avg_cost, current_price=price,
        market_value=qty * price, pnl=(price - avg_cost) * qty,
        pnl_pct=(price / avg_cost - 1) * 100 if avg_cost else 0.0,
        today_quantity=today,
    )


class FakeBroker(BrokerAPI):
    """最小内存券商：返回固定行情/账户，记录下单。"""

    is_simulated = False  # 明确「非模拟」以免被安全中枢按模拟盘处理

    def __init__(self, price=10.0, positions=None, total_asset=1_000_000.0):
        super().__init__()
        self._connected = True
        self.price = price
        self.positions = list(positions or [])
        self.orders = []
        self.total_asset = total_asset
        self.available_cash = 600_000.0

    def connect(self, **kw): self._connected = True; return True
    def disconnect(self): self._connected = False
    def is_trading_time(self): return True

    def query_account(self):
        mv = sum(p.market_value for p in self.positions)
        return AccountInfo(
            account_id="FAKE", total_asset=self.total_asset + mv,
            available_cash=self.available_cash, frozen_cash=0.0,
            market_value=mv, total_pnl=0.0, today_pnl=0.0,
        )

    def query_positions(self):
        return list(self.positions)

    def query_orders(self, status_filter="all"):
        return list(self.orders)

    def cancel_order(self, order_id):
        return True

    def cancel_all(self):
        return 0

    def get_broker_name(self):
        return "fake"

    def get_tick(self, code):
        return TickData(
            code=code, name=code, last_price=self.price, open=self.price,
            high=self.price, low=self.price, pre_close=self.price,
            volume=10_000_000, amount=self.price * 10_000_000,
            bid1=self.price - 0.01, ask1=self.price + 0.01,
        )

    def place_order(self, code, side, order_type, price, quantity,
                    strategy_tag=""):
        o = Order(
            order_id=f"O{len(self.orders)}", code=code, name=code, side=side,
            order_type=order_type, price=price, quantity=quantity,
            filled_quantity=quantity, status=OrderStatus.FILLED,
            strategy_tag=strategy_tag,
        )
        self.orders.append(o)
        return o


# ----------------------------------------------------------------------
# 1. 移动止损：不污染持仓对象 + 清仓后剔除旧高点
# ----------------------------------------------------------------------

def test_stop_check_does_not_mutate_position_price():
    broker = FakeBroker()
    ex = TradeExecutor(broker, RiskConfig())
    pos = _pos(avg_cost=10.0, price=10.0)

    ex.check_stop_conditions(positions=[pos], prices={"600000": 12.0})

    assert pos.current_price == pytest.approx(10.0), "历史高点绝不能写回持仓对象"


def test_trailing_high_pruned_after_position_closed():
    """卖出后再买回，不能被上一轮的峰值误判成回撤。"""
    broker = FakeBroker()
    ex = TradeExecutor(broker, RiskConfig())

    # 第一轮：成本 10，最高涨到 12 → 记录高点 12
    ex.check_stop_conditions(positions=[_pos(avg_cost=10.0, price=10.0)],
                             prices={"600000": 12.0})
    assert ex._high_water["600000"] == pytest.approx(12.0)

    # 清仓（positions 为空）→ 高点必须被剔除
    ex.check_stop_conditions(positions=[], prices={})
    assert "600000" not in ex._high_water

    # 重新买回：成本 10，现价 10.5（+5%）。若旧高点 12 仍在，
    # 回撤 -12.5% 会让它立刻触发移动止损。
    alerts = ex.check_stop_conditions(
        positions=[_pos(avg_cost=10.0, price=10.5)],
        prices={"600000": 10.5},
    )
    assert alerts == [], f"重新建仓不应被上一轮高点误杀，实际: {alerts}"


def test_trailing_stop_still_fires_within_same_position():
    """同一持仓内回撤仍要正常触发（别把修复做成失效）。"""
    broker = FakeBroker()
    ex = TradeExecutor(broker, RiskConfig(trailing_stop_pct=-3.0))
    pos = _pos(avg_cost=10.0, price=10.0)

    ex.check_stop_conditions(positions=[pos], prices={"600000": 11.0})  # +10%
    alerts = ex.check_stop_conditions(positions=[pos], prices={"600000": 10.5})
    assert len(alerts) == 1
    assert "移动止损" in alerts[0]["reason"]


# ----------------------------------------------------------------------
# 2. 止损卖出量必须过 T+1
# ----------------------------------------------------------------------

def test_stop_alert_quantity_respects_t1_lock():
    broker = FakeBroker()
    ex = TradeExecutor(broker, RiskConfig())
    # 持仓 1000，其中今日买入 600 → 可卖 400
    pos = _pos(avg_cost=10.0, qty=1000, available=1000, today=600, price=9.0)

    alerts = ex.check_stop_conditions(positions=[pos], prices={"600000": 9.0})

    assert len(alerts) == 1
    assert alerts[0]["quantity"] == 400, "止盈止损也不能卖当日买入的份额"


def test_stop_alert_skipped_when_all_locked():
    broker = FakeBroker()
    ex = TradeExecutor(broker, RiskConfig())
    pos = _pos(avg_cost=10.0, qty=1000, available=1000, today=1000, price=9.0)

    assert ex.check_stop_conditions(positions=[pos], prices={"600000": 9.0}) == []


# ----------------------------------------------------------------------
# 3. LIVE 执行必须尊重计划股数（部分止盈不能变清仓）
# ----------------------------------------------------------------------

def test_execute_ai_signal_caps_sell_to_planned_quantity():
    pos = _pos(avg_cost=10.0, qty=1000, available=1000, price=10.0)
    broker = FakeBroker(price=10.0, positions=[pos])
    ex = TradeExecutor(broker, RiskConfig())

    ok, msg, order = ex.execute_ai_signal(
        code="600000", name="600000", ai_score=40.0, agreement=1.0,
        side=OrderSide.SELL, risk_exit=True, explicit_quantity=300,
    )

    assert ok, msg
    assert order is not None
    assert order.quantity == 300, "计划卖 300 股就该只卖 300 股"
    assert broker.orders[-1].quantity == 300


def test_execute_ai_signal_planned_quantity_cannot_exceed_sellable():
    pos = _pos(avg_cost=10.0, qty=1000, available=1000, price=10.0)
    broker = FakeBroker(price=10.0, positions=[pos])
    ex = TradeExecutor(broker, RiskConfig())

    ok, _msg, order = ex.execute_ai_signal(
        code="600000", name="600000", ai_score=40.0, agreement=1.0,
        side=OrderSide.SELL, risk_exit=True, explicit_quantity=5000,
    )

    assert ok
    assert order.quantity == 1000, "计划超量也只能卖到可卖上限"


# ----------------------------------------------------------------------
# 4. AutoTrader 止盈止损必须走 guard 审批链
# ----------------------------------------------------------------------

class _FakeProposal:
    status = "pending"
    proposal_id = "p1"
    quantity = 400
    reason = ""


class _FakeGuard:
    def __init__(self):
        self.propose_calls = []
        self.approve_calls = 0

    def positions(self):
        return [_pos(avg_cost=10.0, qty=1000, available=400, today=600)]

    def propose_trade(self, **kw):
        self.propose_calls.append(kw)
        return _FakeProposal()

    def approve_and_execute(self, proposal_id, operator=None):
        self.approve_calls += 1
        return True, "ok", Order(
            order_id="P1", code="600000", name="600000", side=OrderSide.SELL,
            order_type=OrderType.MARKET, price=9.0, quantity=400,
            filled_quantity=400, status=OrderStatus.FILLED,
        )


def test_autotrader_stops_route_through_guard():
    from src.trading.autotrader import AutoTrader

    broker = FakeBroker()
    trader = AutoTrader(broker=broker, risk_config=RiskConfig())
    guard = _FakeGuard()
    trader.guard = guard

    trader.executor.check_stop_conditions = lambda **kw: [{
        "code": "600000", "name": "600000", "reason": "止损 -10%",
        "action": "sell", "pnl_pct": -10.0, "quantity": 400,
    }]

    orders = trader._execute_stops()

    assert guard.propose_calls and guard.propose_calls[0]["risk_exit"] is True
    assert guard.propose_calls[0]["explicit_quantity"] == 400
    assert guard.approve_calls == 1
    assert len(orders) == 1
    assert broker.orders == [], "接入 guard 后不得再直接向券商下单"


def test_autotrader_stops_fall_back_without_guard():
    from src.trading.autotrader import AutoTrader

    broker = FakeBroker(price=9.0, positions=[
        _pos(avg_cost=10.0, qty=1000, available=1000, price=9.0)])
    trader = AutoTrader(broker=broker, risk_config=RiskConfig())

    orders = trader._execute_stops()  # guard is None → 原有直连行为

    assert len(orders) == 1
    assert broker.orders, "未接入 guard 时应保持旧行为"


# ----------------------------------------------------------------------
# 5. 组合级风控：只数上限不该误伤加仓；行业集中度不该静默失效
# ----------------------------------------------------------------------

def _sup(tmp_path, **cfg_kw):
    from src.trading.guard import GuardConfig, RiskSupervisor
    cfg = GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
        **cfg_kw,
    )
    return RiskSupervisor(FakeBroker(), cfg)


def test_max_positions_allows_adding_to_existing_holding(tmp_path):
    sup = _sup(tmp_path)
    sup.config.max_positions = 2
    positions = {
        "600000": _pos("600000", qty=1000, price=10.0),
        "600001": _pos("600001", qty=1000, price=10.0),
    }
    account = sup._account()

    # 给已有持仓加仓：只数不变 → 必须放行
    assert sup._check_portfolio_limits(
        "600000", OrderSide.BUY, account, positions) == ""

    # 新增第三只：才该被只数上限拦下
    blocked = sup._check_portfolio_limits(
        "600002", OrderSide.BUY, account, positions)
    assert "上限" in blocked


def test_industry_map_autoload_is_skipped_under_pytest(tmp_path, monkeypatch):
    from src.trading import guard as g

    (tmp_path / "industry_map.json").write_text(
        '{"600000": "银行"}', encoding="utf-8")
    monkeypatch.setattr(g, "DATA_DIR", tmp_path)

    # pytest 下必须跳过（hermetic），即使文件存在
    assert g._load_industry_map() == {}

    # 非测试环境才真的装载
    monkeypatch.setattr(g, "_running_under_pytest", lambda: False)
    assert g._load_industry_map() == {"600000": "银行"}


def test_sector_concentration_blocks_when_map_present(tmp_path):
    from src.trading.guard import RiskSupervisor, GuardConfig

    cfg = GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
    )
    sup = RiskSupervisor(
        FakeBroker(total_asset=1_000_000.0), cfg,
        industry_map={"600000": "银行", "600001": "银行"},
    )
    sup.config.max_sector_pct = 0.2
    # 两只银行股各 15 万 → 行业 30 万，已超 20% 上限（20 万）
    positions = {
        "600000": _pos("600000", qty=1500, price=100.0),
        "600001": _pos("600001", qty=1500, price=100.0),
    }
    account = sup._account()

    reason = sup._check_portfolio_limits(
        "600000", OrderSide.BUY, account, positions)
    assert "行业" in reason and "上限" in reason


# ----------------------------------------------------------------------
# 6. 执行前预检必须也用 PAPER 行情兜底（否则停牌/涨跌停规则整段跳过）
# ----------------------------------------------------------------------

def test_execute_precheck_uses_paper_quote_fallback(tmp_path, monkeypatch):
    from src.trading.guard import GuardConfig, RiskSupervisor

    cfg = GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
    )
    sup = RiskSupervisor(FakeBroker(), cfg, paper_account=object())
    monkeypatch.setattr(sup.broker, "get_tick", lambda code: None)
    # 兜底行情：成交量为 0 → 停牌
    monkeypatch.setattr(sup, "_paper_quote_fallback", lambda code, tick: TickData(
        code=code, name=code, last_price=10.0, open=10.0, high=10.0, low=10.0,
        pre_close=10.0, volume=0, amount=0.0))

    prop = sup.gate.create(
        code="600000", name="600000", side="buy", price=10.0,
        quantity=100, est_amount=1000.0, ai_score=90.0, agreement=1.0,
        signal_strength=1.0, strategy_tag="ai", risk_exit=False,
    )
    sup.gate.approve(prop.proposal_id, operator="test")

    ok, msg, _order = sup.approve_and_execute(prop.proposal_id, operator="test")

    assert not ok
    assert "停牌" in msg

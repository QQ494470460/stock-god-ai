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


# ----------------------------------------------------------------------
# Batch 3：市场状态粘性（Regime Hysteresis）
# ----------------------------------------------------------------------
def test_set_regime_applies_hysteresis(sup):
    # 初始 sideways；单次翻转被粘性挡住
    sup.set_regime("bull")
    assert sup.regime.regime == "sideways"
    # 连续第二次才切换
    sup.set_regime("bull")
    assert sup.regime.regime == "bull"
    # 切换后动量权重应放大
    assert sup._adjust_signal_strength("momentum", 1.0) > 1.0


def test_set_regime_revert_needs_hold_days(sup):
    sup.set_regime("bull")
    sup.set_regime("bull")  # 切到 bull
    assert sup.regime.regime == "bull"
    # 回到 sideways 也要连续两次
    sup.set_regime("sideways")
    assert sup.regime.regime == "bull"
    sup.set_regime("sideways")
    assert sup.regime.regime == "sideways"


# ----------------------------------------------------------------------
# Batch 3：波动自适应熔断线（Volatility-scaled kill-switch）
# ----------------------------------------------------------------------
def test_effective_daily_loss_uses_market_vol(sup):
    sup.config.adaptive_kill_switch = True
    sup.config.kill_switch_ref_vol_pct = 1.5
    # 未推送波动 → 静态 3.0
    sup.set_market_vol(None)
    assert sup._effective_daily_loss_pct() == pytest.approx(3.0)
    # 高波动 → 放宽（ratio 2.0 截到 1.5 → 4.5，封顶 4.5）
    sup.set_market_vol(3.0)
    assert sup._effective_daily_loss_pct() == pytest.approx(4.5)
    # 低波动 → 收紧（ratio 0.5 截到 0.7 → 2.1）
    sup.set_market_vol(0.75)
    assert sup._effective_daily_loss_pct() == pytest.approx(2.1)
    # 关闭 → 静态
    sup.config.adaptive_kill_switch = False
    assert sup._effective_daily_loss_pct() == pytest.approx(3.0)


def test_effective_daily_loss_caps_and_floors(sup):
    sup.config.adaptive_kill_switch = True
    sup.config.kill_switch_ref_vol_pct = 1.5
    sup.config.kill_switch_floor_pct = 2.0
    sup.config.kill_switch_cap_pct = 4.5
    sup.set_market_vol(100.0)   # 极端高 → 封顶 4.5
    assert sup._effective_daily_loss_pct() <= 4.5 + 1e-9
    sup.set_market_vol(0.001)   # 极端低 → 封底 2.0
    assert sup._effective_daily_loss_pct() >= 2.0 - 1e-9


# ----------------------------------------------------------------------
# Batch 3：策略信誉自适应权重 → 多策略集成（Feature #7）
# ----------------------------------------------------------------------
def test_credibility_feeds_fuse_signals(sup):
    # 给 momentum 灌一串正收益 → 抬高其信誉权重
    for _ in range(30):
        sup.observe_strategy_outcome("momentum", 1.0, 0.05)
    assert sup._ic_weights.get("momentum", 1.0) > 1.0
    # 等评分的两标签融合：momentum 信誉高 → 贡献应高于 defensive
    out = sup.fuse_signals({"momentum": 50.0, "defensive": 50.0})
    assert out["contributions"]["momentum"] > out["contributions"]["defensive"]


def test_credibility_disabled_is_noop(sup):
    sup.config.credibility_enabled = False
    for _ in range(30):
        sup.observe_strategy_outcome("momentum", 1.0, 0.05)
    assert sup._ic_weights == {}  # 关闭时不应产出生效权重


# ----------------------------------------------------------------------
# 波动率尾部风险自适应降仓（Feature #6）
# ----------------------------------------------------------------------
def _feed_equity(sup, values):
    """依次把一串净值喂给 _update_vol_control，触发滚动窗口更新。"""
    for v in values:
        sup._update_vol_control(v)


def test_vol_control_factor_stays_one_with_steady_equity(sup):
    # 净值几乎不动 → 波动极小 → 系数恒为 1.0（不干预）
    sup.config.vol_control_window = 20
    steady = [1_000_000.0 + i * 100.0 for i in range(30)]  # 极缓涨
    _feed_equity(sup, steady)
    assert sup._vol_control_factor == pytest.approx(1.0)


def test_vol_control_factor_suppresses_new_entry_on_spike(sup):
    # 先给一段平稳净值铺满窗口，再砸一根大跌，制造「近期波动 >> 常态基线」
    sup.config.vol_control_window = 20
    sup.config.vol_control_spike_multiple = 2.5
    sup.config.vol_control_min_strength = 0.3
    base = [1_000_000.0] * 24  # 全平 → 常态波动基线≈0，但 median 会被 0 拉满 1e-6 兜底
    # 制造一段有微小正波动的正常序列（避免基线=0 退化），再接暴跌
    normal = [1_000_000.0 * (1 + 0.001 * i) for i in range(24)]
    crash = normal[-1] * 0.80  # 单 tick -20% 暴跌
    _feed_equity(sup, normal + [crash])
    # 窗口足够 + 出现 -20% 极端收益 → 近期 std 远超基线 → 系数被压到 <1.0
    assert sup._vol_control_factor < 1.0
    assert sup._vol_control_factor >= sup.config.vol_control_min_strength


def test_vol_control_never_suppresses_exit(sup):
    # 即便波动降仓系数 <1（压新开仓），离场/止损永远是 1.0
    sup.config.vol_control_window = 20
    normal = [1_000_000.0 * (1 + 0.001 * i) for i in range(24)]
    crash = normal[-1] * 0.80
    _feed_equity(sup, normal + [crash])
    assert sup._vol_control_factor < 1.0  # 确认已进入降仓状态
    # 离场信号不打折
    assert sup._adjust_signal_strength("loop", 1.0, is_exit=True) == 1.0
    # 新开仓信号打折
    assert sup._adjust_signal_strength("loop", 1.0, is_exit=False) < 1.0


# ----------------------------------------------------------------------
# 策略信誉落盘：pytest 下绝不写盘（曾因 import-time 判据失效污染生产 data/）
# ----------------------------------------------------------------------
def _paper_with_closed_trade():
    from types import SimpleNamespace
    return SimpleNamespace(trade_history=[
        SimpleNamespace(side="buy", pnl=0.0, pnl_pct=0.0, strategy="rebalance",
                        trade_id="B1", close_id="", code="000001",
                        filled_at="2026-10-20T10:00:00", ai_score=55.0),
        SimpleNamespace(side="sell", pnl=100.0, pnl_pct=1.0, strategy="rebalance",
                        trade_id="S1", close_id="B1", code="000001",
                        filled_at="2026-10-21T10:00:00", ai_score=0.0),
    ])


def test_sync_credibility_does_not_write_under_pytest(sup, tmp_path, monkeypatch):
    """回归护栏：判据必须在**调用时**取，否则 import-time 快照恒为 False，
    一次 `pytest` 就把生产 data/credibility_state.json 覆盖成测试数据。"""
    import src.trading.guard as g
    target = tmp_path / "cred.json"
    monkeypatch.setattr(g, "CRED_STATE_PATH", target, raising=False)
    sup.sync_credibility_from_paper(_paper_with_closed_trade())
    assert not target.exists(), "pytest 下绝不应落盘"
    assert sup._credibility.count("rebalance") == 1  # 但内存信誉照常更新


def test_sync_credibility_persists_when_not_under_pytest(sup, tmp_path, monkeypatch):
    import json
    import src.trading.guard as g
    target = tmp_path / "cred.json"
    monkeypatch.setattr(g, "CRED_STATE_PATH", target, raising=False)
    monkeypatch.setattr(g, "_running_under_pytest", lambda: False)
    sup.sync_credibility_from_paper(_paper_with_closed_trade())
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["scores"]["rebalance"]["n"] == 1

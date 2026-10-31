# -*- coding: utf-8 -*-
"""风控中枢加固：零股清仓 / 单票满仓 VaR / 换手记账持久化 / 限频不丢单。

四个都是「不报错、只会静默做错事」的缺陷：

1. **零股清仓被整手化**：150 股只能卖 100，剩 50 股永远卖不掉（含止损）。
   A 股规则是「卖出可不足一手，但须一次性卖光」。
2. **单票满仓时 VaR 静默失效**：多标的 VaR 要求「可用标的 ≥ 2」，
   于是**最集中**的状态反而唯一不检查。
3. **换手预算只存在内存**：常驻进程重启一次预算就归零，一天重启三次
   就能买出三倍换手 —— 而重启在生产里是常态。
4. **限频把「间隔不足」当拒绝**：一个 tick 里连下多笔时，除第一笔外全部
   静默 blocked，表现为「调仓永远补不到位」、止损排队 30s。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading.broker import (
    AccountInfo, BrokerAPI, Order, OrderSide, OrderStatus, OrderType,
    Position, TickData,
)
from src.trading.guard import GuardConfig, RiskSupervisor


class _Broker(BrokerAPI):
    is_simulated = False

    def __init__(self, price=10.0, total=1_000_000.0, positions=None):
        super().__init__()
        self._connected = True
        self.price = price
        self.total = total
        self.positions = list(positions or [])

    def connect(self, **kw): return True
    def disconnect(self): pass
    def is_connected(self): return True
    def is_trading_time(self): return True
    def query_positions(self): return list(self.positions)
    def query_orders(self, status_filter="all"): return []
    def cancel_order(self, order_id): return True
    def cancel_all(self): return 0
    def get_broker_name(self): return "fake"

    def query_account(self):
        mv = sum(p.market_value for p in self.positions)
        return AccountInfo(account_id="FAKE", total_asset=self.total,
                           available_cash=self.total - mv, frozen_cash=0.0,
                           market_value=mv, total_pnl=0.0, today_pnl=0.0)

    def get_tick(self, code):
        return TickData(code=code, name=code, last_price=self.price,
                        open=self.price, high=self.price, low=self.price,
                        pre_close=self.price, volume=50_000_000,
                        amount=self.price * 50_000_000,
                        bid1=self.price - 0.01, ask1=self.price + 0.01)

    def place_order(self, code, side, order_type, price, quantity, strategy_tag=""):
        return Order(order_id="O1", code=code, name=code, side=side,
                     order_type=order_type, price=price, quantity=quantity,
                     filled_quantity=quantity, status=OrderStatus.FILLED)


class _Monitor:
    """风险监控替身：给定单票 VaR，验证 guard 是否真的用上了。"""

    def __init__(self, single=None, multi=None):
        self.single = single
        self.multi = multi

    def var_pct_single(self, code, weight, z=1.65):
        return self.single

    def var_pct(self, codes, weights, z=1.65):
        return self.multi


def _guard(tmp_path, adv=None, monitor=None, turnover_file=None, **cfg_kw):
    cfg = GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
        **cfg_kw,
    )
    return RiskSupervisor(
        _Broker(), cfg, industry_map={},
        adv_provider=(lambda code: adv),
        risk_monitor=monitor,
        turnover_file=turnover_file,
    )


def _account(total=1_000_000.0):
    return AccountInfo(account_id="P", total_asset=total, available_cash=total,
                       frozen_cash=0.0, market_value=0.0, total_pnl=0.0, today_pnl=0.0)


def _pos(qty, price=10.0, today=0, avail=None):
    return Position(code="600000", name="600000", quantity=qty,
                    available=qty - today if avail is None else avail,
                    avg_cost=price, current_price=price,
                    market_value=qty * price, pnl=0.0, pnl_pct=0.0,
                    today_quantity=today)


# ----------------------------------------------------------------------
# 1. 零股清仓
# ----------------------------------------------------------------------
def test_size_explicit_keeps_odd_lot_on_full_liquidation(tmp_path):
    g = _guard(tmp_path)
    qty, amt = g._size_explicit(
        "600000", OrderSide.SELL, 10.0, 150, _account(),
        {"600000": _pos(150)}, existing_value=1500.0)
    assert qty == 150
    assert amt == pytest.approx(1500.0)


def test_size_explicit_floors_partial_odd_sell(tmp_path):
    g = _guard(tmp_path)
    qty, _ = g._size_explicit(
        "600000", OrderSide.SELL, 10.0, 100, _account(),
        {"600000": _pos(150)}, existing_value=1500.0)
    assert qty == 100


def test_size_explicit_caps_sell_to_sellable(tmp_path):
    """150 股里 50 股当日买入锁定 → 可卖 100，仍是整手。"""
    g = _guard(tmp_path)
    pos = _pos(150, today=50)
    qty, _ = g._size_explicit(
        "600000", OrderSide.SELL, 10.0, 150, _account(), {"600000": pos}, 1500.0)
    assert qty == 100


def test_apply_limits_keeps_odd_lot_when_allowed(tmp_path):
    g = _guard(tmp_path, adv=1e12)     # ADV 极大 → 流动性不限制
    qty, why = g._apply_execution_limits(
        "600000", OrderSide.SELL, 10.0, 150, _account(), allow_odd_lot=True)
    assert qty == 150
    assert why == ""


def test_apply_limits_floors_odd_lot_by_default(tmp_path):
    g = _guard(tmp_path, adv=1e12)
    qty, _ = g._apply_execution_limits(
        "600000", OrderSide.SELL, 10.0, 150, _account())
    assert qty == 100


def test_apply_limits_never_lets_buy_use_odd_lot(tmp_path):
    """买入永远整手：即使调用方误传 allow_odd_lot 也要取整。"""
    g = _guard(tmp_path, adv=1e12)
    qty, _ = g._apply_execution_limits(
        "600000", OrderSide.BUY, 10.0, 150, _account(), allow_odd_lot=True)
    assert qty == 100


# ----------------------------------------------------------------------
# 2. 单票满仓 VaR
# ----------------------------------------------------------------------
def test_single_position_var_is_enforced(tmp_path):
    """满仓单票 VaR 超预算 → 新开仓强度被压（旧实现直接跳过检查）。"""
    mon = _Monitor(single=0.06)          # 6% > 2.5% 预算
    g = _guard(tmp_path, monitor=mon, max_portfolio_var_pct=0.025,
               var_scale_floor=0.4, var_monitor_enabled=True)
    scale, detail = g._portfolio_var_scale(
        _account(), {"600000": _pos(90_000)})     # 90% 仓位
    assert detail["n"] == 1
    assert scale < 1.0
    assert scale >= 0.4
    assert scale == pytest.approx(0.025 / 0.06, rel=0.05)


def test_single_position_var_within_budget_is_neutral(tmp_path):
    mon = _Monitor(single=0.01)
    g = _guard(tmp_path, monitor=mon, max_portfolio_var_pct=0.025,
               var_monitor_enabled=True)
    scale, detail = g._portfolio_var_scale(_account(), {"600000": _pos(50_000)})
    assert scale == 1.0
    assert detail["var_pct"] == pytest.approx(0.01)


def test_single_position_var_none_is_fail_open(tmp_path):
    """历史数据不足 → 不约束（不能因为算不出来就把交易停掉）。"""
    mon = _Monitor(single=None)
    g = _guard(tmp_path, monitor=mon)
    scale, detail = g._portfolio_var_scale(_account(), {"600000": _pos(50_000)})
    assert scale == 1.0


# ----------------------------------------------------------------------
# 3. 换手记账持久化
# ----------------------------------------------------------------------
def test_turnover_survives_restart(tmp_path):
    f = tmp_path / "turnover.json"
    g1 = _guard(tmp_path, turnover_file=f)
    g1._bump_turnover(120_000.0)
    assert f.exists()

    # 模拟进程重启：新实例（同一文件）
    # 关掉风险预算这条，隔离出「换手预算」单独验证。
    # 换手比例也显式钉成 0.30：生产默认已放宽到 0.95（见 GuardConfig 里的
    # 口径说明），这里验的是「记账能否跨重启恢复」，不该被默认值变动带跑。
    g2 = _guard(tmp_path, turnover_file=f, max_risk_per_trade_pct=1.0,
                max_daily_turnover_pct=0.30)
    assert g2._turnover_amount == pytest.approx(120_000.0)

    # 预算应被历史消耗打掉：总资产 100 万 × 30% = 30 万，已用 12 万 → 剩 18 万
    qty, why = g2._apply_execution_limits(
        "600000", OrderSide.BUY, 10.0, 50_000, _account(), allow_odd_lot=False)
    assert qty == 18_000
    assert "换手" in why


def test_turnover_ignores_stale_day(tmp_path):
    f = tmp_path / "turnover.json"
    f.write_text(json.dumps({"day": "1999-01-01", "amount": 999_999.0}), encoding="utf-8")
    g = _guard(tmp_path, turnover_file=f)
    assert g._turnover_amount == 0.0


def test_turnover_corrupt_file_is_fail_open(tmp_path):
    f = tmp_path / "turnover.json"
    f.write_text("{not json", encoding="utf-8")
    g = _guard(tmp_path, turnover_file=f)
    assert g._turnover_amount == 0.0


# ----------------------------------------------------------------------
# 4. 限频不丢单
# ----------------------------------------------------------------------
def test_rate_limit_waits_instead_of_dropping(tmp_path):
    g = _guard(tmp_path, min_order_interval_sec=0.3)
    assert g._check_rate_limit() == ""          # 第一笔直接通过
    with g._lock:
        g._last_order_ts = time.time()
        g._orders_this_minute.append(g._last_order_ts)

    t0 = time.time()
    assert g._check_rate_limit() == ""          # 第二笔等待后通过，而不是被拒
    assert time.time() - t0 >= 0.25


def test_rate_limit_still_enforces_per_minute_cap(tmp_path):
    g = _guard(tmp_path, min_order_interval_sec=0.0, max_orders_per_minute=3)
    now = time.time()
    with g._lock:
        g._orders_this_minute = [now, now, now]
    assert "上限" in g._check_rate_limit()


# ----------------------------------------------------------------------
# 5. 实盘下单路径同样要能清掉零股
# ----------------------------------------------------------------------
def _executor(positions, price=10.0):
    from src.trading.executor import RiskConfig, TradeExecutor

    broker = _Broker(price=price, positions=positions)
    return TradeExecutor(broker, RiskConfig(min_ai_score_buy=0, max_ai_score_sell=100,
                                            require_agreement=False)), broker


def test_execute_ai_signal_sells_odd_lot_on_full_liquidation():
    ex, broker = _executor([_pos(150)])
    ok, msg, order = ex.execute_ai_signal(
        code="600000", name="600000", ai_score=50, agreement=1.0,
        side=OrderSide.SELL, risk_exit=True, explicit_quantity=150)
    assert ok, msg
    assert order.quantity == 150


def test_execute_ai_signal_floors_partial_odd_sell():
    ex, broker = _executor([_pos(1000)])
    ok, msg, order = ex.execute_ai_signal(
        code="600000", name="600000", ai_score=50, agreement=1.0,
        side=OrderSide.SELL, risk_exit=True, explicit_quantity=130)
    assert ok, msg
    assert order.quantity == 100


def test_execute_ai_signal_buy_still_needs_lot():
    ex, broker = _executor([])
    ok, msg, order = ex.execute_ai_signal(
        code="600000", name="600000", ai_score=90, agreement=1.0,
        side=OrderSide.BUY, explicit_quantity=150)
    if ok:                      # 资金充足时会成交，但必须整手
        assert order.quantity % 100 == 0
    else:
        assert "不足1手" in msg or "仓位" in msg


# ----------------------------------------------------------------------
# 6. 风险预算必须用「实际生效」的止损距离
# ----------------------------------------------------------------------
class _FakeAtr:
    def __init__(self, value):
        self.value = value

    def atr_pct(self, code):
        return self.value


def _guard_with_atr(tmp_path, atr):
    """开启 ATR 的 guard：止损被放宽后，单笔风险预算应随之收紧。"""
    from src.trading.executor import RiskConfig, TradeExecutor

    cfg = GuardConfig(state_file=str(tmp_path / "s.json"),
                      journal_file=str(tmp_path / "j.jsonl"),
                      max_risk_per_trade_pct=0.005, risk_budget_stop_pct=0.05,
                      max_participation_rate=1.0)      # 让流动性不干扰
    rc = RiskConfig(atr_stop_enabled=True, atr_stop_k=2.0,
                    atr_stop_min_pct=2.0, atr_stop_max_pct=15.0)
    broker = _Broker()
    broker._atr = atr
    sup = RiskSupervisor(broker, cfg, risk_config=rc, industry_map={},
                         adv_provider=lambda code: None)
    # 把带 ATR 的执行器换进去（guard 内部按 risk_config 构造了默认执行器）
    sup.executor = TradeExecutor(broker, rc, atr_provider=atr)
    sup.risk_config = rc
    return sup


def test_risk_budget_uses_atr_widened_stop(tmp_path):
    # ATR 8% × 2 = 16% → 夹到 15% → 预算 5000/(10×0.15) = 3333 → 3300 股
    g = _guard_with_atr(tmp_path, _FakeAtr(0.08))
    qty, why = g._apply_execution_limits("600000", OrderSide.BUY, 10.0, 20_000, _account())
    assert qty == 3_300
    assert "15%" in why


def test_risk_budget_uses_static_stop_when_atr_off(tmp_path):
    g = _guard(tmp_path, adv=None, max_risk_per_trade_pct=0.005,
               risk_budget_stop_pct=0.05)
    qty, _ = g._apply_execution_limits("600000", OrderSide.BUY, 10.0, 20_000, _account())
    assert qty == 10_000        # 5% 止损 → 5000/(10*0.05)


# ----------------------------------------------------------------------
# 5. 行业映射缺失时集中度检查不能被整段跳过
# ----------------------------------------------------------------------
def test_unknown_sector_is_bucketed_not_skipped(tmp_path):
    """映射查不到的标的必须归入同一个「未知行业」桶继续受约束。

    回归背景（2026-10-26 实测）：``sector = industry_map.get(code, "")`` 之后
    ``if sector:`` —— 查不到就整段放行（fail-open）。当时 industry_map 2616 条
    只覆盖部分市场：当日 4 只选股里 603688 缺、2 只持仓里 300037 也缺，
    这些买入完全不受集中度约束，而「不知道它是什么行业」本该按最保守假设处理。
    """
    g = _guard(tmp_path, max_sector_pct=0.30)
    acct = _account(1_000_000.0)
    positions = {"600000": _pos(40_000, price=10.0)}      # 未知行业已占 40%

    why = g._check_portfolio_limits("300111", OrderSide.BUY, acct, positions)
    assert why, "映射缺失时集中度检查被跳过了（fail-open）"
    assert "未知行业" in why


def test_unknown_sector_below_cap_still_passes(tmp_path):
    """归桶不等于一刀切：未知行业合计仍在阈值内时必须放行。"""
    g = _guard(tmp_path, max_sector_pct=0.30)
    acct = _account(1_000_000.0)
    positions = {"600000": _pos(10_000, price=10.0)}      # 10% < 30%
    assert g._check_portfolio_limits("300111", OrderSide.BUY, acct, positions) == ""


def test_known_sector_checked_against_same_sector_only(tmp_path):
    """有映射时仍按「同行业」聚合，不会把不同行业混成一个桶。"""
    g = _guard(tmp_path, max_sector_pct=0.30)
    g.industry_map = {"600000": "银行", "300111": "医药"}
    acct = _account(1_000_000.0)
    positions = {"600000": _pos(40_000, price=10.0)}      # 银行 40%
    # 买医药：与银行不同桶 → 放行
    assert g._check_portfolio_limits("300111", OrderSide.BUY, acct, positions) == ""
    # 再买银行：同桶已 40% ≥ 30% → 拦截
    why = g._check_portfolio_limits("601988", OrderSide.BUY, acct, positions)
    assert why == "" or "银行" in why   # 601988 无映射 → 进未知桶，不受银行约束
    why2 = g._check_portfolio_limits("600000", OrderSide.BUY, acct, positions)
    assert "银行" in why2

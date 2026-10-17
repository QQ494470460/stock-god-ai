# -*- coding: utf-8 -*-
"""
交易安全与风控中枢 单元测试
覆盖：武装状态机、Kill Switch 持久化、人工确认闸门、回撤/亏损熔断、
仓位规模计算、纸交易 vs 实盘路由。
全部使用 Fake 对象，零真实资金、零网络。
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 确保项目根与 src 在 sys.path（pytest 从不同 cwd 运行时也能 import）
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from src.trading.broker import (
    BrokerAPI, Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo, TickData,
)
from src.trading.guard import (
    GuardConfig, SafetyController, OrderGate, PositionSizer,
    RiskSupervisor, TradingMode,
)


# ======================================================================
# Fake 对象
# ======================================================================

class _LiveBrokerStub:
    """最小替身：只回答「券商是不是模拟盘」这一项。

    arm_live 现在会 fail-closed 地校验券商真伪（缺 ``is_simulated`` 一律当模拟），
    所以凡是要走武装流程的用例都必须显式给一个「真实盘」替身。
    """
    is_simulated = False


class FakeBroker(BrokerAPI):
    """内存券商：记录下单、返回固定行情/账户"""

    # 武装实盘要校验券商真伪：缺这个属性会被按「模拟盘」拒绝（fail-closed）
    is_simulated = False

    def __init__(self, total_asset=1_000_000.0, last_price=10.0):
        super().__init__()
        self._connected = True
        self.total_asset = total_asset
        self.available_cash = total_asset * 0.6
        self.last_price = last_price
        self.orders = []
        self.positions = {}

    def connect(self, **kwargs): self._connected = True; return True
    def disconnect(self): self._connected = False
    def query_account(self):
        return AccountInfo(
            account_id="FAKE", total_asset=self.total_asset,
            available_cash=self.available_cash, frozen_cash=0.0,
            market_value=self.total_asset - self.available_cash,
            total_pnl=0.0, today_pnl=0.0,
        )
    def query_positions(self):
        return [Position(
            code=c, name=c, quantity=p["qty"], available=p["qty"],
            avg_cost=p["cost"], current_price=self.last_price,
            market_value=p["qty"] * self.last_price, pnl=0.0, pnl_pct=0.0,
            today_quantity=0,
        ) for c, p in self.positions.items()]
    def query_orders(self, status_filter="all"): return []
    def place_order(self, code, side, order_type, price, quantity, strategy_tag=""):
        o = Order(
            order_id=f"ORD{len(self.orders)+1:03d}", code=code, side=side,
            order_type=order_type, price=price, quantity=quantity,
            filled_quantity=quantity, status=OrderStatus.FILLED,
            strategy_tag=strategy_tag,
        )
        self.orders.append(o)
        return o
    def cancel_order(self, order_id): return True
    def cancel_all(self): return 0
    def get_tick(self, code):
        return TickData(
            code=code, name=code, last_price=self.last_price, open=10.0,
            high=10.5, low=9.5, pre_close=10.0, volume=100000, amount=1e6,
            bid1=self.last_price - 0.01, ask1=self.last_price + 0.01,
            change_pct=0.0,
        )
    def get_broker_name(self): return "FakeBroker"
    def is_trading_time(self): return True


class FakePaper:
    """极简纸账户，仅实现 supervisor 用到的接口"""

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
        return [{
            "code": c, "name": c, "quantity": h["qty"], "available": h["qty"],
            "today_quantity": 0, "avg_cost": h["cost"], "last_price": h["last"],
            "market_value": h["qty"] * h["last"], "pnl": 0.0, "pnl_pct": 0.0,
        } for c, h in self.holdings.items()]

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


# ======================================================================
# 测试
# ======================================================================

def test_default_mode_is_paper(tmp_path):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    ctl = SafetyController(cfg, broker=_LiveBrokerStub())
    assert ctl.mode == TradingMode.PAPER
    assert ctl.can_trade() is True
    assert ctl.halted is False


def test_set_mode_cannot_enter_live_directly(tmp_path):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    ctl = SafetyController(cfg, broker=_LiveBrokerStub())
    assert ctl.set_mode(TradingMode.LIVE) is False
    assert ctl.set_mode(TradingMode.LIVE_PENDING) is False
    assert ctl.mode == TradingMode.PAPER


def test_arm_flow_requires_two_step_token(tmp_path):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    ctl = SafetyController(cfg, broker=_LiveBrokerStub())
    token = ctl.arm_live("alice")
    assert ctl.mode == TradingMode.PAPER  # 第一步仍是 paper
    assert ctl.confirm_arm("wrong-token", "alice") is False
    assert ctl.mode == TradingMode.PAPER
    assert ctl.confirm_arm(token, "alice") is True
    assert ctl.mode == TradingMode.LIVE_PENDING


def test_kill_switch_persists_across_restart(tmp_path):
    state = str(tmp_path / "s.json")
    cfg = GuardConfig(state_file=state, journal_file=str(tmp_path / "j.jsonl"))
    ctl = SafetyController(cfg, broker=_LiveBrokerStub())
    ctl.arm_live("alice")
    ctl.confirm_arm(ctl._state.pending_token, "alice")
    assert ctl.mode == TradingMode.LIVE_PENDING
    ctl.halt("emergency", operator="bob")
    assert ctl.halted is True

    # 模拟重启：新建实例从磁盘读取
    ctl2 = SafetyController(cfg)
    assert ctl2.halted is True
    assert ctl2.mode == TradingMode.LIVE_PENDING  # halt 不降级模式，只冻结

    # resume 后曾实盘 → 强制回 PAPER
    ctl2.resume("bob")
    assert ctl2.halted is False
    assert ctl2.mode == TradingMode.PAPER


def test_position_sizer_fixed_fractional(tmp_path):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    s = PositionSizer(cfg)
    qty, budget = s.size(OrderSide.BUY, 10.0, 1_000_000, signal_strength=1.0)
    assert budget == pytest.approx(20_000.0)   # 1e6 * 0.02 * 1.0
    assert qty == 2000


def test_position_sizer_volatility_target(tmp_path):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    cfg.sizing_method = "volatility_target"
    s = PositionSizer(cfg)
    qty, budget = s.size(OrderSide.BUY, 10.0, 1_000_000, signal_strength=1.0,
                         annualized_vol=0.30)
    # frac = min(0.15/0.30, 0.02*2)=min(0.5,0.04)=0.04 → budget 40000
    assert budget == pytest.approx(40_000.0)
    assert qty == 4000


def test_position_sizer_kelly(tmp_path):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    cfg.sizing_method = "kelly"
    s = PositionSizer(cfg)
    qty, budget = s.size(OrderSide.BUY, 10.0, 1_000_000, signal_strength=1.0,
                         win_prob=0.6, payoff=2.0)
    # kelly=0.6-0.4/2=0.4 → 半 Kelly 0.2 → 封顶 kelly_cap 0.10 → budget 100000
    assert budget == pytest.approx(100_000.0)
    assert qty == 10_000


def test_paper_execution_routes_to_paper(tmp_path):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    broker = FakeBroker()
    paper = FakePaper()
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    # PAPER 模式：propose 自动批准
    prop = sup.propose_trade("600000", "浦发", OrderSide.BUY, ai_score=70, agreement=0.9)
    assert prop.status == "approved"
    ok, msg, order = sup.approve_and_execute(prop.proposal_id, "auto")
    assert ok is True
    assert len(broker.orders) == 0          # 没走券商
    assert "600000" in paper.holdings        # 走了纸账户


def test_paper_insufficient_score_blocked(tmp_path):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    broker = FakeBroker()
    paper = FakePaper()
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    # 默认 min_ai_score_buy=55，评分 40 应被 executor 内层拦截
    prop = sup.propose_trade("600000", "浦发", OrderSide.BUY, ai_score=40, agreement=0.9)
    # propose 阶段不拦评分（由 executor 在执行时拦），但执行会失败
    ok, msg, order = sup.approve_and_execute(prop.proposal_id, "auto")
    assert ok is False
    assert "评分不足" in msg


def test_live_pending_requires_human_confirmation(tmp_path, monkeypatch):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"),
                       require_live_confirmation=True)
    broker = FakeBroker()
    paper = FakePaper()
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    # 武装到 LIVE_PENDING
    token = sup.safety.arm_live("alice")
    assert sup.safety.confirm_arm(token, "alice") is True
    assert sup.safety.mode == TradingMode.LIVE_PENDING

    # 绕过交易时段日历检查（测试环境可能非交易时段）
    import src.trading.guard as G
    monkeypatch.setattr(G, "TradeJournal", lambda *a, **k: _NullJournal())
    import src.trading.calendar as cal
    monkeypatch.setattr(cal, "order_allowed", lambda: (True, "test"))

    prop = sup.propose_trade("600000", "浦发", OrderSide.BUY, ai_score=70, agreement=0.9)
    assert prop.status == "pending"   # 未自动批准，需人工确认

    # 未批准直接执行 → 拒绝
    ok, msg, order = sup.approve_and_execute(prop.proposal_id, "bob")
    assert ok is False
    assert "未获人工批准" in msg

    # 人工批准后再执行 → 走券商
    assert sup.gate.approve(prop.proposal_id, "bob") is True
    ok, msg, order = sup.approve_and_execute(prop.proposal_id, "bob")
    assert ok is True
    assert len(broker.orders) == 1


def test_drawdown_circuit_breaker(tmp_path):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"),
                       max_portfolio_drawdown_pct=8.0)
    broker = FakeBroker(total_asset=900_000)  # 当前净值
    paper = FakePaper(total=900_000)
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    # 设峰值为 100 万，当前 90 万 → 回撤 10% ≥ 8% 熔断
    sup.safety._state.peak_equity = 1_000_000.0
    sup.safety._state.today_date = "2000-01-01"  # 强制刷新当日起始
    alerts = sup.tick_monitor()
    assert sup.safety.halted is True
    assert any("回撤" in a for a in alerts)


def test_kill_switch_blocks_trading(tmp_path):
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    broker = FakeBroker()
    paper = FakePaper()
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    sup.kill_switch("manual stop", operator="user")
    assert sup.safety.halted is True
    prop = sup.propose_trade("600000", "浦发", OrderSide.BUY, ai_score=70, agreement=0.9)
    assert prop.status == "blocked"


class _NullJournal:
    def log(self, *a, **k): pass
    def tail(self, n=50): return []


# ======================================================================
# AutoTrader ↔ RiskSupervisor 接线集成测试
# ======================================================================

from src.trading.autotrader import AutoTrader  # noqa: E402


def _make_candidate(code="600000", score=70, xgb_signal=1):
    return {
        "code": code, "name": code, "score": score,
        "agreement": 1.0, "xgb_signal": xgb_signal,
    }


def _wire_autotrader(broker, paper, sup=None, monkeypatch=None):
    at = AutoTrader(broker=broker)  # 显式传 broker，避免实例化真实券商
    at.paper_account = paper
    # 测试环境关闭交易时段门控，确保可交易场景可被复现（生产环境默认仍生效）
    at.executor.config.trading_hours_only = False
    if sup is not None:
        at.set_guard(sup)
    at.set_ai_predictor(lambda code: _make_candidate(code))
    if monkeypatch is not None:
        # 绕过 watchlist.json 读取，直接用固定候选
        monkeypatch.setattr(at, "_scan_buy_candidates", lambda: [_make_candidate()])
    else:
        at._scan_buy_candidates = lambda: [_make_candidate()]
    return at


def test_autotrader_guard_routes_paper_through_supervisor(tmp_path, monkeypatch):
    """接入安全中枢后，买入信号应走 guard→纸账户，绝不落券商"""
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    broker = FakeBroker()
    paper = FakePaper()
    sup = RiskSupervisor(broker, cfg, paper_account=paper)

    at = _wire_autotrader(broker, paper, sup, monkeypatch)
    assert at.guard is sup
    assert at.paper_account is paper  # 共享引用

    res = at.scan_and_trade()
    assert any(r.get("type") == "buy_signal" and r.get("success") for r in res)
    assert "600000" in paper.holdings          # 安全中枢执行了纸交易
    assert len(broker.orders) == 0             # 未走券商（PAPER 模式正确）
    assert at.get_status()["guard"]["mode"] == "paper"


def test_autotrader_guard_kill_switch_halts_scan(tmp_path, monkeypatch):
    """KILL SWITCH 触发后，scan_and_trade 不得产生任何交易"""
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    broker = FakeBroker()
    paper = FakePaper()
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    sup.kill_switch("test emergency", operator="user")

    at = _wire_autotrader(broker, paper, sup, monkeypatch)
    res = at.scan_and_trade()
    assert "600000" not in paper.holdings
    assert all(not r.get("success") for r in res if r.get("type") == "buy_signal")
    assert sup.safety.halted is True


def test_autotrader_default_no_guard_preserves_behavior(tmp_path):
    """不接入安全中枢时，行为必须与接入前完全一致（下单走 executor→券商，并手动记纸账户）"""
    broker = FakeBroker()
    paper = FakePaper()
    at = _wire_autotrader(broker, paper, sup=None)
    assert at.guard is None

    res = at.scan_and_trade()
    assert any(r.get("type") == "buy_signal" and r.get("success") for r in res)
    assert len(broker.orders) == 1             # 默认路径走 executor→broker 下单
    assert "600000" in at.paper_account.holdings  # 默认路径手动记纸账户


def test_autotrader_status_exposes_guard(tmp_path, monkeypatch):
    """get_status / print_status 应能在接入后安全读取 guard 状态"""
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"), journal_file=str(tmp_path / "j.jsonl"))
    broker = FakeBroker()
    paper = FakePaper()
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    at = _wire_autotrader(broker, paper, sup, monkeypatch)
    status = at.get_status()
    assert status["guard"] is not None
    assert status["guard"]["mode"] == "paper"
    at.print_status()  # 不应抛异常



# ======================================================================
# 净值基线失真导致误熔断（曾真实发生）
# ----------------------------------------------------------------------
# 场景：纸账户 10 万，而状态文件里遗留 peak=100 万（换账号/出入金/纸账户
# 重置都会造成）→ 首 tick 立刻判「回撤 89.95%」并 halt，且无法自行恢复。
# 修复：① 显式 rebaseline_peak()；② 资金账号明确切换时自动重置。
# ======================================================================

def _sup_with_peak(tmp_path, peak, today_start=None,
                   baseline_account="PAPER", equity=100_000.0):
    """构造一个「状态文件里带指定 peak、账户权益为 equity」的 supervisor。

    注意：PAPER 模式下 `guard._account()` 走的是**纸账户**（account_id="PAPER"），
    不是 broker，所以权益要设在 FakePaper 上。
    """
    state = tmp_path / "s.json"
    state.write_text(json.dumps({
        "mode": "paper", "halted": False,
        "peak_equity": peak,
        "today_start_equity": today_start if today_start is not None else peak,
        "today_date": datetime.now().strftime("%Y-%m-%d"),
        "baseline_account": baseline_account,
    }, ensure_ascii=False), encoding="utf-8")
    cfg = GuardConfig(state_file=str(state), journal_file=str(tmp_path / "j.jsonl"))
    paper = FakePaper(total=equity)
    sup = RiskSupervisor(FakeBroker(total_asset=equity), cfg, paper_account=paper)
    return sup, paper, state


def test_stale_peak_triggers_false_halt(tmp_path):
    """复现：遗留 peak=100 万 vs 实际 10 万 → 回撤 90% 直接熔断。"""
    sup, paper, _ = _sup_with_peak(tmp_path, peak=1_000_000.0)
    alerts = sup.tick_monitor()
    assert sup.safety.halted is True
    assert any("回撤" in a for a in alerts)


def test_rebaseline_peak_clears_false_halt(tmp_path):
    """rebaseline_peak 应以当前权益重建基线，熔断条件随之消失。"""
    sup, paper, _ = _sup_with_peak(tmp_path, peak=1_000_000.0)
    sup.tick_monitor()
    assert sup.safety.halted is True

    sup.safety.rebaseline_peak(100_000.0, operator="alice", reason="纸账户重置")

    snap = sup.safety.snapshot()
    assert snap["peak_equity"] == pytest.approx(100_000.0)
    assert snap["today_start_equity"] == pytest.approx(100_000.0)
    assert sup.safety.current_drawdown_pct(100_000.0) == pytest.approx(0.0)
    assert sup.safety.daily_loss_pct(100_000.0) == pytest.approx(0.0)


def test_rebaseline_peak_writes_journal(tmp_path):
    """重置基线必须留痕（事后审计能看出谁在什么时候放宽了基线）。"""
    sup, paper, _ = _sup_with_peak(tmp_path, peak=1_000_000.0)
    sup.safety.rebaseline_peak(100_000.0, operator="alice", reason="换了券商")
    lines = (tmp_path / "j.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines, "journal 应至少有一条记录"
    rec = json.loads(lines[-1])
    assert rec["event"] == "rebaseline_peak"
    assert rec["data"]["operator"] == "alice"
    assert rec["data"]["new_baseline"] == pytest.approx(100_000.0)


def test_account_switch_auto_rebaselines(tmp_path):
    """资金账号明确切换 → 自动重置基线，不再误熔断。"""
    sup, paper, _ = _sup_with_peak(
        tmp_path, peak=1_000_000.0, baseline_account="FAKE_LIVE_888")
    alerts = sup.tick_monitor()
    assert sup.safety.halted is False, "账号切换不该触发熔断"
    assert any("账号已切换" in a for a in alerts)
    assert sup.safety.snapshot()["baseline_account"] == "PAPER"
    assert sup.safety.snapshot()["peak_equity"] == pytest.approx(100_000.0)


def test_no_auto_rebaseline_when_prev_account_unknown(tmp_path):
    """旧状态文件没有账号字段时，**不得**静默放宽熔断 —— 宁可 halt。"""
    sup, paper, _ = _sup_with_peak(
        tmp_path, peak=1_000_000.0, baseline_account="")
    sup.tick_monitor()
    assert sup.safety.snapshot()["baseline_account"] == "PAPER"          # 仅记录身份
    assert sup.safety.snapshot()["peak_equity"] == pytest.approx(1_000_000.0)
    assert sup.safety.halted is True, "基线不可比时应保持保守（halt）"


def test_normal_drawdown_still_halts(tmp_path):
    """回归保护：真实回撤（同账号）必须仍然熔断，不能被修复改坏。"""
    sup, paper, _ = _sup_with_peak(tmp_path, peak=100_000.0)
    paper.cash = 90_000.0          # 同账号，权益跌到 9 万 → 回撤 10% ≥ 8% 熔断线
    alerts = sup.tick_monitor()
    assert sup.safety.halted is True
    assert any("回撤" in a for a in alerts)


# ======================================================================
# 风控拦截必须留痕（审计）
# ----------------------------------------------------------------------
# 曾有一整天的「意图 N / 拦截 N / 成交 0」在 journal 里没有任何 risk_block 记录，
# 原因无法追溯。根因：propose_trade 里有两条拦截路径（拿不到行情/账户、计算下单量
# 不足 1 手）只 mark_blocked 却没 emit("risk_block")。下面把这两处钉死。
# ======================================================================

def test_propose_logs_risk_block_when_qty_below_one_lot(tmp_path):
    """下单量不足 1 手被拦截时，必须写 risk_block 到 journal。"""
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"),
                      journal_file=str(tmp_path / "j.jsonl"))
    cfg.base_risk_fraction = 1e-6          # 预算被压到 < 100 股 → 触发 qty<100 拦截
    broker = FakeBroker()
    paper = FakePaper()
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    prop = sup.propose_trade("600000", "浦发", OrderSide.BUY,
                             ai_score=70, agreement=1.0)
    assert prop.status == "blocked", prop.reason
    lines = (tmp_path / "j.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines, "journal 应至少有一条记录"
    assert any(json.loads(l).get("event") == "risk_block" for l in lines)


def test_propose_logs_risk_block_when_no_market_data(tmp_path, monkeypatch):
    """拿不到行情/账户被拦截时，也必须写 risk_block（LIVE 模式走券商账户）。"""
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"),
                      journal_file=str(tmp_path / "j.jsonl"),
                      require_live_confirmation=True)
    broker = FakeBroker()
    paper = FakePaper()
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    token = sup.safety.arm_live("alice")
    assert sup.safety.confirm_arm(token, "alice") is True

    monkeypatch.setattr(broker, "get_tick", lambda code: None)
    monkeypatch.setattr(broker, "query_account", lambda: None)

    prop = sup.propose_trade("600000", "浦发", OrderSide.BUY,
                             ai_score=70, agreement=1.0)
    assert prop.status == "blocked"
    lines = (tmp_path / "j.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(l).get("event") == "risk_block" for l in lines)


def test_signal_precheck_block_is_journaled(tmp_path):
    """信号级预检（如 AI 评分不足）拦截时同样留 risk_block。"""
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"),
                      journal_file=str(tmp_path / "j.jsonl"))
    broker = FakeBroker()
    paper = FakePaper()
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    # 默认 min_ai_score_buy=55，评分 30 应被信号预检拦下
    prop = sup.propose_trade("600000", "浦发", OrderSide.BUY,
                             ai_score=30, agreement=1.0)
    assert prop.status == "blocked"
    lines = (tmp_path / "j.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(l).get("event") == "risk_block" for l in lines)


# ======================================================================
# 武装模式切换必须留痕（审计）
# ----------------------------------------------------------------------
# 曾出现 trading_guard_state.json 是 live_pending、但 journal 里查不到任何 arm
# 记录的情况（违反「放宽风控须显式留痕」铁律）。根因：SafetyController 的
# set_mode/arm_live/confirm_arm/disarm 全部只改状态文件、从不写 journal。
# 下面把这些切换钉死。
# ======================================================================

def test_mode_transitions_are_journaled(tmp_path):
    """武装/解除/切换模式必须写 journal（arm_request / arm_confirm / mode_change）。"""
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"),
                      journal_file=str(tmp_path / "j.jsonl"))
    ctl = SafetyController(cfg, broker=_LiveBrokerStub())
    token = ctl.arm_live("alice")
    assert ctl.confirm_arm(token, "alice") is True
    assert ctl.disarm("alice") is True

    lines = (tmp_path / "j.jsonl").read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(l).get("event") for l in lines]
    assert "arm_request" in events
    assert "arm_confirm" in events
    assert "mode_change" in events


def test_disarm_records_audit_reason(tmp_path):
    """disarm 回 PAPER 时，mode_change 必须带操作者与原因，便于事后追溯。"""
    cfg = GuardConfig(state_file=str(tmp_path / "s.json"),
                      journal_file=str(tmp_path / "j.jsonl"))
    ctl = SafetyController(cfg, broker=_LiveBrokerStub())
    token = ctl.arm_live("bob")
    ctl.confirm_arm(token, "bob")
    ctl.disarm("bob", reason="audit-cleanup: 无 arm 留痕，回安全默认")
    lines = (tmp_path / "j.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["event"] == "mode_change"
    assert rec["data"]["to"] == "paper"
    assert rec["data"]["operator"] == "bob"
    assert "audit-cleanup" in rec["data"]["reason"]


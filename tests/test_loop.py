# -*- coding: utf-8 -*-
"""
闭环交易引擎（loop）单元测试
============================
覆盖：状态机推进、非交易时段待命、熔断冻结、预演无副作用、
演练模式(dry_run)不下单、PAPER 模式真实记账、收盘结算写报告、
未传 guard 时自动构建安全中枢（绝不裸奔下单）。
零网络、零真实资金（纸账户落盘到 tmp）。
"""
import sys
import copy
import json
from datetime import datetime, date, timedelta
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
from src.trading.loop import TradingLoop, LoopConfig, LoopState, TickReport
from src.trading.trade_plan import TradePlan, PlanTarget, save_trade_plan
from src.trading.guard import build_supervisor, TradingMode

DAY = "2026-10-15"


# ======================================================================
# Fake 依赖
# ======================================================================

class FakeBroker(BrokerAPI):
    """内存券商：固定行情、记录下单"""

    def __init__(self, total=1_000_000.0, cash=400_000.0, price=10.0):
        super().__init__()
        self._connected = True
        self.total_asset = total
        self.available_cash = cash
        self.price = price
        self.orders = []
        self.positions = {}          # code -> (qty, cost)

    def connect(self, **kw): self._connected = True; return True
    def disconnect(self): self._connected = False

    def query_account(self):
        mv = sum(q * self.price for q, _ in self.positions.values())
        return AccountInfo(
            account_id="FAKE", total_asset=self.total_asset,
            available_cash=self.available_cash, frozen_cash=0.0,
            market_value=mv, total_pnl=0.0, today_pnl=0.0,
        )

    def query_positions(self):
        return [Position(
            code=c, name=c, quantity=q, available=q,
            avg_cost=cost, current_price=self.price, market_value=q * self.price,
            pnl=0.0, pnl_pct=0.0, today_quantity=0,
        ) for c, (q, cost) in self.positions.items()]

    def query_orders(self, status_filter="all"): return []

    def place_order(self, code, side, order_type, price, quantity, strategy_tag=""):
        o = Order(order_id=f"O{len(self.orders)+1}", code=code, side=side,
                  order_type=order_type, price=price, quantity=quantity,
                  filled_quantity=quantity, status=OrderStatus.FILLED,
                  strategy_tag=strategy_tag)
        self.orders.append(o)
        return o

    def cancel_order(self, order_id): return True
    def cancel_all(self): return 0

    def get_tick(self, code):
        return TickData(code=code, name=code, last_price=self.price, open=self.price,
                        high=self.price, low=self.price, pre_close=self.price,
                        volume=100000, amount=1e6,
                        bid1=self.price - 0.01, ask1=self.price + 0.01)

    def get_broker_name(self): return "FakeBroker"
    def is_trading_time(self): return True


class FakePaper:
    """极简纸账户：实现 guard PAPER 分支所需接口"""

    def __init__(self, total=1_000_000.0):
        self.cash = total
        self.frozen_cash = 0.0
        self.holdings = {}
        self.today_pnl = 0.0
        self._trades = []

    def get_total_asset(self):
        return self.cash + sum(h["qty"] * h["last"] for h in self.holdings.values())

    def get_total_pnl(self): return 0.0

    def get_positions(self):
        return [{
            "code": c, "name": c, "quantity": h["qty"], "available": h["qty"],
            "today_quantity": 0, "avg_cost": h["cost"], "last_price": h["last"],
            "market_value": h["qty"] * h["last"], "pnl": 0.0, "pnl_pct": 0.0,
        } for c, h in self.holdings.items()]

    def get_recent_trades(self, n=50):
        return self._trades[-n:]

    def mark_to_market(self, prices):
        for c, p in prices.items():
            if c in self.holdings:
                self.holdings[c]["last"] = p

    def buy(self, code, name, price, quantity, strategy="", ai_score=0.0):
        cost = price * quantity
        if self.cash < cost:
            return False, "资金不足", None
        self.cash -= cost
        h = self.holdings.setdefault(code, {"qty": 0, "cost": 0.0, "last": price})
        h["qty"] += quantity
        h["cost"] = price
        h["last"] = price
        rec = _Rec()
        self._trades.append(rec)
        return True, "ok", rec

    def sell(self, code, price, quantity, strategy="", ai_score=0.0):
        h = self.holdings.get(code)
        if not h or h["qty"] < quantity:
            return False, "无持仓", None
        self.cash += price * quantity
        h["qty"] -= quantity
        if h["qty"] <= 0:
            del self.holdings[code]
        rec = _Rec()
        self._trades.append(rec)
        return True, "ok", rec


class _Rec:
    trade_id = "T1"


def mk_plan(targets, reserve=0.10):
    return TradePlan(
        plan_date=DAY,
        targets=[PlanTarget(code=c, name=c, score=s, weight=w) for c, s, w in targets],
        cash_reserve_pct=reserve,
    )


@pytest.fixture(autouse=True)
def isolate_production_persistence(monkeypatch, tmp_path):
    """把本模块会碰到的**全部生产落盘目录**隔离到 tmp —— 绝不污染真实 data/。

    已知两个落盘口，都必须掐掉：

    1. **guard 状态/日志**：``GuardConfig`` 未显式指定 state_file/journal_file 时
       落到 ``config.settings.DATA_DIR``（真实 data/），一旦触发熔断就会把
       ``halted=True`` 写进用户真实的风控状态文件。
    2. **loop 收盘结算报告**：``TradingLoop._settle()`` 往
       ``Path(DATA_DIR)/"loop_reports"`` 写 ``loop_report_YYYYMMDD.json``。
       曾踩过 —— 夹具只冻结日历没冻结时钟时，一组用例在收盘后跑到
       ``tick()`` → ``_settle()``，把 ``FakeBroker`` 的虚拟权益（95 万）
       当成真实结算写进 ``data/loop_reports/loop_report_20261014.json``，
       一份看起来完全正常的伪造报告。``_doctor`` 会把它当成真实业绩读。

    ``loop.py`` 里 ``DATA_DIR`` 只有 ``:509`` 一处在用，且是模块级名字查找
    （不是默认参数绑定），所以改模块属性即可生效。
    """
    import src.trading.guard as guardmod
    import src.trading.loop as loopmod
    monkeypatch.setattr(guardmod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(loopmod, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def force_trading_time(monkeypatch):
    """把日历**和时钟**都强制为「可下单的交易时段」。

    只改日历是不够的：``_should_settle`` 比的是墙上时钟是否已过 15:00，
    于是同一份用例在 14:00 跑是通过、在 16:00 跑就必然被判 SETTLED ——
    这组用例曾整组在收盘后变红，看起来像代码坏了，其实只是天晚了。
    把时钟一起冻结到 10:00，用例才真正与运行时刻无关。
    """
    import src.trading.calendar as cal
    import src.trading.loop as loopmod
    monkeypatch.setattr(cal, "market_status", lambda now=None: "早盘交易中")
    monkeypatch.setattr(cal, "order_allowed", lambda now=None: (True, "早盘交易中"))
    monkeypatch.setattr(cal, "is_trading_day", lambda d=None: True)
    monkeypatch.setattr(loopmod, "_now", lambda: datetime(2026, 10, 14, 10, 0, 0))
    return cal


@pytest.fixture
def force_closed(monkeypatch):
    """把日历强制为「休市」，时钟冻结在**盘中**时刻。

    时钟刻意停在盘中：这样「不下单」只可能来自休市判定本身，
    而不是因为恰好跑到收盘之后 —— 否则这个用例就从「验证休市」
    悄悄退化成「验证现在几点了」。
    """
    import src.trading.calendar as cal
    import src.trading.loop as loopmod
    monkeypatch.setattr(cal, "market_status", lambda now=None: "休市（非交易日）")
    monkeypatch.setattr(cal, "order_allowed", lambda now=None: (False, "非交易日"))
    monkeypatch.setattr(cal, "is_trading_day", lambda d=None: False)
    monkeypatch.setattr(loopmod, "_now", lambda: datetime(2026, 10, 14, 10, 0, 0))
    return cal


def mk_loop(broker=None, paper=None, cfg=None, guard=None):
    b = broker or FakeBroker()
    p = paper if paper is not None else FakePaper()
    g = guard or build_supervisor(b, paper_account=p)
    return TradingLoop(b, guard=g, paper_account=p, config=cfg)


# ======================================================================
# 安全默认
# ======================================================================

def test_auto_builds_paper_guard_when_none_given():
    """未传 guard 时自动构建，且默认是 PAPER 纸交易 —— 绝不裸奔下单。"""
    b = FakeBroker()
    loop = TradingLoop(b, guard=None, paper_account=FakePaper())
    assert loop.guard is not None
    assert loop.guard.safety.mode == TradingMode.PAPER
    assert not loop.guard.safety.is_live()


def test_config_defaults_are_conservative():
    cfg = LoopConfig()
    assert cfg.enable_rebalance is True
    assert cfg.enable_stop_orders is True
    assert cfg.cash_reserve_pct == 0.10
    assert cfg.weight_cap == 0.30
    assert cfg.dry_run is False


# ======================================================================
# 状态机 / 时段
# ======================================================================

def test_initial_state_stopped():
    loop = mk_loop()
    assert loop.state == LoopState.STOPPED
    assert not loop.is_running


def test_closed_market_goes_idle_and_does_not_trade(force_closed):
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    rep = loop.tick()
    assert rep.state in (LoopState.IDLE.value, LoopState.PREOPEN.value)
    assert rep.n_executed == 0
    assert rep.n_intents == 0


def test_trading_time_goes_running(force_trading_time):
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    rep = loop.tick()
    assert rep.state == LoopState.RUNNING.value


def test_halted_freezes_everything(force_trading_time):
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    loop.guard.kill_switch("测试熔断")
    rep = loop.tick()
    assert rep.state == LoopState.HALTED.value
    assert rep.n_executed == 0
    assert "熔断" in rep.note


# ======================================================================
# dry_run 演练
# ======================================================================

def test_dry_run_places_no_order(force_trading_time):
    b = FakeBroker()
    loop = mk_loop(broker=b, cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    rep = loop.tick()
    assert b.orders == []                       # 一张单都没发
    assert rep.n_intents >= 1                   # 但确实算出了意图
    assert rep.n_executed >= 1                  # 记为「演练成交」


def test_dry_run_does_not_touch_paper_account(force_trading_time):
    paper = FakePaper()
    loop = mk_loop(paper=paper, cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    loop.tick()
    assert paper.holdings == {}                 # 纸账户未被改动
    assert paper.cash == 1_000_000.0


# ======================================================================
# PAPER 真实记账
# ======================================================================

def test_ensure_plan_uses_todays_plan_not_the_newer_tomorrow_plan(monkeypatch, tmp_path):
    """磁盘上「明天的计划」更新时，也必须加载**今天**的那份。

    计划是「收盘后为下一交易日」写的，所以 trade_plan_明天.json 的文件 mtime
    天然比 trade_plan_今天.json 新。而 ``load_latest_plan()`` 挑的是**文件最新**
    的那份 → 闭环一启动就抱着明天的计划交易今天。

    实跑证据（2026-10-16）：磁盘上明明有 ``trade_plan_20261016.json``
    （当天 14:17 刚生成），闭环却从 14:36 到 15:00 全天显示「计划 2026-10-19」，
    结算报告里 ``plan_date`` 也是 2026-10-19 —— 用后天的计划交易今天。

    这正是「跨日拿错计划」那个老 bug 没修完的残留：上次只堵了「过期就重建」，
    没堵住「按文件新旧挑计划」这个更上游的口子。
    """
    import src.trading.loop as loopmod
    import src.trading.trade_plan as tpmod

    today, tomorrow = "2026-10-16", "2026-10-19"
    # plan 目录 = <DATA_DIR>/trade_plans，两个模块各有自己的 DATA_DIR，都要指过去
    monkeypatch.setattr(tpmod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(loopmod, "_now", lambda: datetime(2026, 10, 16, 10, 0, 0))

    save_trade_plan(TradePlan(
        plan_date=today,
        targets=[PlanTarget(code="600177", name="雅戈尔", score=62.0, weight=0.09)],
        cash_reserve_pct=0.10,
    ))
    save_trade_plan(TradePlan(          # 后写 → 它是「文件最新」的那份
        plan_date=tomorrow,
        targets=[PlanTarget(code="000001", name="平安银行", score=70.0, weight=0.09)],
        cash_reserve_pct=0.10,
    ))

    loop = mk_loop(cfg=LoopConfig(auto_build_plan=True))
    loop.plan = None                     # 模拟冷启动
    loop._ensure_plan(TickReport())

    assert loop.plan is not None, "没加载到任何计划"
    assert loop.plan.plan_date == today, (
        f"应当用今天({today})的计划，实际拿到 {loop.plan.plan_date} —— "
        "又变成「拿明天的计划交易今天」了")


def test_paper_mode_actually_buys(force_trading_time):
    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    loop = mk_loop(broker=b, paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, dry_run=False))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    rep = loop.tick()
    assert rep.n_executed >= 1
    assert "600519" in paper.holdings            # 纸账户真的持仓了
    assert paper.cash < 1_000_000.0
    assert b.orders == []                        # PAPER 模式不打真券商单


def test_paper_mode_respects_cash_reserve(force_trading_time):
    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    loop = mk_loop(broker=b, paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, dry_run=False))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    loop.tick()
    # 纸账户现金不得低于 总资产(约100万) × 10%
    assert paper.cash >= paper.get_total_asset() * 0.10 - 1000


def test_paper_mode_sells_dropped_position(force_trading_time):
    paper = FakePaper(total=1_000_000.0)
    paper.cash = 700_000.0
    paper.holdings["600000"] = {"qty": 30000, "cost": 10.0, "last": 10.0}
    b = FakeBroker(total=1_000_000.0, cash=700_000.0)
    loop = mk_loop(broker=b, paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, dry_run=False))
    loop.plan = mk_plan([("600519", 90, 0.30)])   # 600000 不在计划里
    loop.tick()
    assert "600000" not in paper.holdings         # 已清仓


# ======================================================================
# 止盈止损（risk_exit 不被评分过滤拦住）
# ======================================================================

def test_stop_loss_sells_even_when_score_high(force_trading_time):
    """止损是纪律性离场：AI 评分再高也必须卖。"""
    paper = FakePaper(total=1_000_000.0)
    paper.cash = 900_000.0
    # 成本 20，现价 10 → 亏 50%，远超 -5% 止损线
    paper.holdings["600000"] = {"qty": 5000, "cost": 20.0, "last": 10.0}
    b = FakeBroker(total=1_000_000.0, cash=900_000.0, price=10.0)
    loop = mk_loop(broker=b, paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, enable_rebalance=False,
                                  dry_run=False))
    loop.plan = mk_plan([])
    rep = loop.tick()
    assert rep.n_stops >= 1
    assert "600000" not in paper.holdings          # 止损已执行


def test_stop_orders_can_be_disabled(force_trading_time):
    paper = FakePaper(total=1_000_000.0)
    paper.cash = 900_000.0
    paper.holdings["600000"] = {"qty": 5000, "cost": 20.0, "last": 10.0}
    b = FakeBroker(total=1_000_000.0, cash=900_000.0, price=10.0)
    loop = mk_loop(broker=b, paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, enable_rebalance=False,
                                  enable_stop_orders=False, dry_run=False))
    loop.plan = mk_plan([])
    rep = loop.tick()
    assert rep.n_stops == 0
    assert "600000" in paper.holdings


# ======================================================================
# 预演
# ======================================================================

def test_preview_has_no_side_effects(force_trading_time):
    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    loop = mk_loop(broker=b, paper=paper, cfg=LoopConfig(auto_build_plan=False))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    before = copy.deepcopy(paper.holdings)
    pv = loop.preview()
    assert paper.holdings == before
    assert b.orders == []
    assert pv["plan_summary"]
    assert pv["guard_mode"] == "paper"
    assert pv["intent_summary"]


# ======================================================================
# 收盘结算
# ======================================================================

def test_should_settle_after_close_time(force_trading_time):
    loop = mk_loop(cfg=LoopConfig(settle_hour=15, settle_minute=5))
    assert loop._should_settle(datetime(2026, 10, 15, 15, 10)) is True
    assert loop._should_settle(datetime(2026, 10, 15, 14, 0)) is False


def test_should_settle_once_per_day(force_trading_time):
    loop = mk_loop(cfg=LoopConfig(settle_hour=15, settle_minute=5))
    assert loop._should_settle(datetime(2026, 10, 15, 15, 10)) is True
    loop._settled_dates.add(date(2026, 10, 15))
    assert loop._should_settle(datetime(2026, 10, 15, 15, 20)) is False


def test_settle_writes_report(monkeypatch, tmp_path, force_trading_time):
    import src.trading.loop as loopmod
    monkeypatch.setattr(loopmod, "DATA_DIR", tmp_path)

    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    loop = mk_loop(broker=b, paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, notify=False))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    loop.tick()                                  # 先建仓

    loop._settled_dates.clear()
    monkeypatch.setattr(loop, "_should_settle", lambda now: True)
    rep = loop.tick()

    out = tmp_path / "loop_reports"
    files = list(out.glob("loop_report_*.json"))
    assert files, "结算报告未生成"
    import json
    snap = json.loads(files[0].read_text(encoding="utf-8"))
    assert snap["equity"] > 0
    assert "positions" in snap and "trades" in snap
    assert loop.state == LoopState.SETTLED
    assert "结算" in rep.note


def test_rebalance_period_gate_skips_and_persists(force_trading_time, tmp_path, monkeypatch):
    """调仓必须按「交易日周期」驱动，且上次调仓日期要跨进程存活。

    背景：2026-10-31 依据 src/core/exec_backtest.py 的网格把周期从 5 天拉到 15 天
    —— 5 天周期在 50bp 成本下夏普塌到 0.41（每期边际收益 ~0.5%，一次往返成本 ~1.1%），
    15 天在三档成本下 1.19/1.06/0.95。这里钉两件事：
      ① 周期内不重复调仓（否则等于逐 tick 调，成本翻倍）；
      ② 日期要落盘 —— 闭环设计上会自愈重启，纯内存状态一重启就立刻再调一次。
    """
    import src.trading.loop as loopmod
    monkeypatch.setattr(loopmod, "DATA_DIR", tmp_path)

    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    cfg = LoopConfig(auto_build_plan=False, dry_run=True, rebalance_days=15,
                     heartbeat_file=str(tmp_path / "hb.json"))
    loop = mk_loop(broker=b, paper=paper, cfg=cfg)
    loop.plan = mk_plan([("600519", 90, 0.30)])

    loop.tick()
    first = loop._last_rebalance_date
    assert first, "调仓日必须记下来"
    assert loop._rebalance_due() is False, "同一天不该再次调仓"

    rep = loop.tick()
    assert rep.n_intents == 0, "周期内不得产生调仓委托"
    assert "不足" in (rep.note or ""), f"应说明为何跳过调仓，实际 note={rep.note!r}"

    # 模拟自愈重启：新实例从 loop_runtime.json 读回上次调仓日
    loop2 = mk_loop(broker=FakeBroker(total=1_000_000.0, cash=1_000_000.0),
                    paper=FakePaper(total=1_000_000.0), cfg=cfg)
    assert loop2._last_rebalance_date == first, "重启后不得丢失调仓节奏（否则立刻再调一次）"
    assert loop2._rebalance_due() is False


# ======================================================================
# 防雪崩闸门 vs 周期记账（顺序陷阱）
# ======================================================================

def _four_target_plan():
    """4 只等权 20% 的目标组合：满仓换手需要「卖 4 + 买 4」共 8 笔。"""
    return mk_plan([(c, 90 - i, 0.20) for i, c in enumerate(
        ["600519", "600000", "000858", "601318"])])


def test_truncated_rebalance_does_not_mark_period_done(force_trading_time, tmp_path):
    """闸门截断委托时**绝不能**写周期标记 —— 否则剩下的票要等一整个周期。

    回归背景（2026-10-31 实测）：闸门原值 10，而一次完整调仓是「卖 ≤8 + 买 ≤8」
    共 16 笔。截断发生在**排序之后**，被截掉的恰好是排在后面的**全部买单**；
    而 ``_last_rebalance_date`` 在循环之前就写死了 ⇒ 「8 只只建了 2 只，剩下
    6 只等 15 个交易日」。账户因此长期半仓、九成现金闲置，收益自然上不去 ——
    这类 bug 不会报错、不会告警，只会安静地让仓位变轻。
    """
    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    cfg = LoopConfig(auto_build_plan=False, dry_run=False, rebalance_days=15,
                     max_orders_per_tick=2,
                     heartbeat_file=str(tmp_path / "hb.json"))
    loop = mk_loop(broker=b, paper=paper, cfg=cfg)
    loop.plan = _four_target_plan()

    rep = loop.tick()

    assert rep.n_intents > 2, f"本用例需要构造出被截断的场景，实际 {rep.n_intents} 笔"
    assert loop._last_rebalance_date == "", "被截断就不算调完，不得写周期标记"
    assert loop._rebalance_due() is True, "下个 tick 必须能接着做剩下的委托"

    # 第二道保险的语义：截断只影响「有没有调完」，不影响已下单部分的成交
    assert rep.n_executed == 2, f"应只执行前 2 笔，实际 {rep.n_executed}"


def test_untruncated_rebalance_marks_period_done(force_trading_time, tmp_path):
    """未截断（正常情况）仍要写周期标记 —— 空转也必须记账。

    与上一条互为对照：修 bug 时最容易的错法是「干脆不写周期标记」，
    那会把 15 天周期悄悄变成逐 tick 调仓，换手与成本直接翻倍。
    """
    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    cfg = LoopConfig(auto_build_plan=False, dry_run=False, rebalance_days=15,
                     max_orders_per_tick=30,
                     heartbeat_file=str(tmp_path / "hb.json"))
    loop = mk_loop(broker=b, paper=paper, cfg=cfg)
    loop.plan = _four_target_plan()

    rep = loop.tick()

    assert rep.n_intents <= 30
    assert loop._last_rebalance_date, "未截断时必须记下本次调仓日"
    assert loop._rebalance_due() is False, "周期内不得再次调仓"
    assert rep.n_executed == rep.n_intents, "未截断时不该有委托被闸门吞掉"


def test_default_order_gate_covers_a_full_rebalance():
    """默认闸门必须 ≥ 2×max_positions，且**两处默认值**都要满足。

    这是算术不变量，不是口味问题：一次完整调仓 = 卖 ≤N + 买 ≤N = 2N 笔。
    生产权威值是 ``config.settings.AppConfig().trading_loop``（``LoopConfig``
    只是 app_config 缺失时的兜底），所以两处一起断言 —— 只改一处等于白改。
    """
    from config.settings import AppConfig

    cfg = LoopConfig()
    assert cfg.max_orders_per_tick >= 2 * cfg.max_positions, (
        f"LoopConfig 闸门 {cfg.max_orders_per_tick} < 2×{cfg.max_positions}")

    tl = AppConfig().trading_loop
    assert tl.max_orders_per_tick >= 2 * tl.max_positions, (
        f"TradingLoopConfig 闸门 {tl.max_orders_per_tick} < 2×{tl.max_positions}")


def test_daily_turnover_budget_allows_building_the_whole_plan():
    """当日换手预算必须够「从现金一天建满目标组合」—— 否则建仓要花两个多月。

    第五处「实盘 ≠ 回测」缺口（2026-10-31 由一条断言意外暴露）。
    回测是 T+1 一次性把仓位建满，**没有**当日买入上限；实盘却把当日买入压在
    总资产 30%。于是「8 只 / 保留 10% 现金」需要 90% 敞口的一天建不完，
    只能建 1/3，而调仓周期是 15 个交易日 ⇒ 满仓要 45 个交易日。
    实测证据：``data/turnover_state.json`` 记录 2026-10-27 当日买入 ¥1,382,444，
    当时权益约 ¥4.55M，正好是 30.4% —— 卡死在上限上。

    「抑制来回翻仓」的职责已由 ``rebalance_days`` + ``band_pct`` 承担；
    本条只需保证「一天最多建满一次」这个物理上限不误伤正常建仓。
    """
    from config.settings import AppConfig
    from src.trading.guard import GuardConfig

    tl = AppConfig().trading_loop
    need = 1.0 - float(tl.cash_reserve_pct)

    # 生产权威来源是 YAML（scripts/run_loop.py 显式加载），先验它
    yaml_cfg = GuardConfig.from_yaml(str(ROOT / "config" / "trading_guard.yaml"))
    assert float(yaml_cfg.max_daily_turnover_pct) >= need, (
        f"生产 YAML 换手预算 {yaml_cfg.max_daily_turnover_pct:.0%} < 建满所需 {need:.0%}")
    # 再验兜底默认值（YAML 缺失时生效，两处不一致 = 改哪份生效全看运气）
    assert float(GuardConfig().max_daily_turnover_pct) >= need


def test_risk_scaling_is_off_so_drawdown_does_not_shrink_new_positions():
    """组合回撤**不得**再压新仓规模 —— 这是被回测判死的顺周期开关。

    第六处「实盘 ≠ 回测」缺口，也是杀伤力最大的一处。
    执行级回测（1140 天 × 200 只，10bp，15 日 + slack3）：

        无闸门         年化 +26.8%  夏普 1.19  回撤 -25.0%  换手 10.6x
        渐进降仓 floor.3 年化 +14.7%  夏普 0.99  回撤 -15.0%  换手  4.5x
        ＋8% 熔断       年化  +6.5%  夏普 0.42  回撤 -16.3%  换手  1.8x  ← 实盘现状

    「越跌越轻」吃掉 3/4 的收益，只换来 8.7pp 回撤改善；而实盘只在 8% 熔断线
    以下工作，回测最大回撤却是 -25% ⇒ 正常历史里必然反复触发，等于换了策略。
    实测证据：10-27 回撤 7.90% 时日志写着「敞口系数 0.31」。
    """
    from config.settings import AppConfig

    assert LoopConfig().enable_risk_scaling is False
    assert AppConfig().trading_loop.enable_risk_scaling is False

    # 行为层：**浅回撤不得砍新仓**（顺周期最烧钱的地方就在这一段）。
    # 2026-11-03 起了 T3 非顺周期覆盖层，但它 12% 才介入，8% 仍必须是 1.0。
    paper = FakePaper(total=800_000.0)
    b = FakeBroker(total=800_000.0, cash=800_000.0)
    loop = mk_loop(broker=b, paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    loop.guard.safety._state.peak_equity = 869_565.0     # 回撤 8%
    acct = b.query_account()
    assert loop._portfolio_risk_scale(acct) == 1.0

    # 覆盖层整体关掉时（derisk_mode=off 且 vol_target_pct=0）也必须回到 1.0
    b2 = FakeBroker(total=800_000.0, cash=800_000.0)
    loop2 = mk_loop(broker=b2, paper=FakePaper(total=800_000.0),
                    cfg=LoopConfig(auto_build_plan=False, dry_run=True,
                                   derisk_mode="off", vol_target_pct=0.0))
    loop2._day_start_equity = {}
    loop2.guard.safety._state.peak_equity = 1_000_000.0   # 回撤 20%
    assert loop2._portfolio_risk_scale(b2.query_account()) == 1.0


def test_settle_report_aggregates_intraday_counts(force_trading_time, tmp_path, monkeypatch):
    """结算报告 stats 必须反映「一整天」的拦截/意图，而非结算那一刻的 0。

    回归背景：旧实现把 report.n_stops/intents/executed/blocked 直接写进报告，
    而 report 是「结算 tick」新建的，盘中发生的拦截全都不见了 —— 报告里永远
    是「风平浪静」的 0/0/0/0，与当天实际「意图 N/拦截 N」对不上。
    """
    import src.trading.loop as loopmod
    monkeypatch.setattr(loopmod, "DATA_DIR", tmp_path)

    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    # rebalance_days=1 → 每个 tick 都发起调仓。本用例要验的是**日内累计计数**，
    # 与「按周期调仓」无关；生产默认 15 天会让第 2、3 个 tick 直接跳过调仓。
    loop = mk_loop(broker=b, paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, notify=False, rebalance_days=1))
    # 目标评分 40 < 风控下限 55 → 每次调仓都会被 guard 拦截（盘中持续发生）
    loop.plan = mk_plan([("600519", 40, 0.30)])

    n_ticks = 3
    for _ in range(n_ticks):
        loop.tick()

    # 当日累计应已累加：每个 tick 1 个意图 + 1 次拦截
    assert loop._day_stats["intents"] == n_ticks, loop._day_stats
    assert loop._day_stats["blocked"] == n_ticks, loop._day_stats

    loop._settled_dates.clear()
    monkeypatch.setattr(loop, "_should_settle", lambda now: True)
    loop.tick()                                  # 触发结算

    files = list((tmp_path / "loop_reports").glob("loop_report_*.json"))
    assert files, "结算报告未生成"
    import json
    snap = json.loads(files[0].read_text(encoding="utf-8"))
    assert snap["stats"]["intents"] == n_ticks, snap["stats"]
    assert snap["stats"]["blocked"] == n_ticks, snap["stats"]
    # 结算后当日累计已清零
    assert loop._day_stats["blocked"] == 0, loop._day_stats


# ======================================================================
# 结算路径的三种「静默失效」
# ======================================================================

def test_halted_loop_still_settles(monkeypatch, tmp_path, force_trading_time):
    """熔断态也必须收盘结算。

    回归背景（2026-10-26 实测）：``_tick_impl`` 的熔断分支在 ④ 收盘结算**之前**
    就 ``return``，于是熔断期间的结算被整体跳过。而熔断不是一 tick 的事 ——
    实测从 10-22 持续到 10-26，``data/loop_reports/`` 一直停在 10-21：
    日报断更、绩效曲线缺段、进化面板没有当日样本，全都悄无声息。
    结算是记账不是交易，不该被风控冻结。
    """
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False, notify=False))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    loop.guard.kill_switch("测试熔断")

    monkeypatch.setattr(loop, "_should_settle", lambda now: True)
    rep = loop.tick()

    files = list((Path(tmp_path) / "loop_reports").glob("loop_report_*.json"))
    assert files, "熔断态下没有生成结算报告 —— 熔断把记账一起冻住了"
    snap = json.loads(files[0].read_text(encoding="utf-8"))
    assert snap["state"] == "halted"          # 报告如实记录「当时是熔断态」
    assert "结算" in rep.note
    assert "熔断" in rep.note                  # 熔断提示不被结算覆盖


def test_settle_missing_account_does_not_fake_total_loss(monkeypatch, tmp_path):
    """账户快照取不到时，绝不能写出「当日 -100%」。

    旧实现 ``equity = account.total_asset if account else 0.0``，而 ``start_eq``
    是真实基线：account 为 None ⇒ day_pnl = -4.7M、day_pnl_pct = -100%，
    日报渲染出来就是「今日爆仓」。取数失败被伪装成灾难，比报错危险得多。
    """
    import src.trading.loop as loopmod
    monkeypatch.setattr(loopmod, "DATA_DIR", tmp_path)

    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False, notify=False))
    today = loopmod._today().strftime("%Y-%m-%d")
    loop._day_start_equity[today] = 1_000_000.0

    rep = TickReport(at="15:10:00", state=loop.state.value)
    loop._settle(None, [], {}, rep)

    snap = json.loads(
        (Path(tmp_path) / "loop_reports" / f"loop_report_{today.replace('-', '')}.json")
        .read_text(encoding="utf-8")
    )
    assert snap["equity_ok"] is False
    assert snap["day_pnl"] == 0.0, f"账户缺失却算出当日盈亏 {snap['day_pnl']}"
    assert snap["day_pnl_pct"] == 0.0
    assert snap["equity"] == snap["day_start_equity"]


def test_settle_missing_baseline_backfills_not_zeroes_pnl(monkeypatch, tmp_path):
    """当日基线缺失时应**补建并落盘**，而不是静默把当日盈亏压成 0。

    旧实现 ``start_eq = self._day_start_equity.get(today, equity)`` 取不到就
    退回当前权益 ⇒ day_pnl 恒为 0，真实亏损被抹平（看起来一切正常）。
    """
    import src.trading.loop as loopmod
    monkeypatch.setattr(loopmod, "DATA_DIR", tmp_path)

    b = FakeBroker(total=900_000.0, cash=900_000.0)
    loop = mk_loop(broker=b, cfg=LoopConfig(auto_build_plan=False, notify=False))
    today = loopmod._today().strftime("%Y-%m-%d")
    loop._day_start_equity.pop(today, None)

    rep = TickReport(at="15:10:00", state=loop.state.value)
    loop._settle(b.query_account(), [], {}, rep)

    snap = json.loads(
        (Path(tmp_path) / "loop_reports" / f"loop_report_{today.replace('-', '')}.json")
        .read_text(encoding="utf-8")
    )
    assert snap["equity_ok"] is True
    assert snap["day_start_equity"] == snap["equity"]
    # 基线已被补建并落到运行态文件，同日后续 tick 不再走兜底
    assert loop._day_start_equity[today] == 900_000.0


def test_settle_markdown_flags_halted_and_bad_equity():
    """日报必须自己说清「当日是否熔断 / 权益是否可信」。

    背景：2026-10-22 起连续熔断，而结算被熔断分支跳过 → 这几天**没有任何文件**
    记录「系统是停摆的」。日报是用户唯一会主动翻的产物，必须把这两件事写在脸上，
    否则一份格式完好的报告看起来就像「今天没行情，正常」。
    """
    base = {
        "date": "2026-10-26", "settled_at": "2026-10-26T15:10:00",
        "state": "settled", "guard_mode": "paper", "plan_date": "2026-10-26",
        "equity": 1_000_000.0, "day_start_equity": 1_000_000.0,
        "day_pnl": 0.0, "day_pnl_pct": 0.0, "equity_ok": True,
        "cash": 1_000_000.0, "market_value": 0.0, "n_positions": 0,
        "positions": [], "trades": [],
        "stats": {"stops": 0, "intents": 0, "executed": 0, "blocked": 0},
    }
    normal = TradingLoop._settle_markdown(dict(base))
    assert "熔断" not in normal and "取不到" not in normal

    halted = TradingLoop._settle_markdown({**base, "state": "halted"})
    assert "处于熔断态" in halted

    bad = TradingLoop._settle_markdown({**base, "equity_ok": False})
    assert "账户快照取不到" in bad


# ======================================================================
# 线程生命周期
# ======================================================================

def test_start_stop_thread(force_closed):
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False, scan_interval_sec=0.05))
    loop.start(interval_seconds=0.05)
    assert loop.is_running
    loop.stop()
    assert not loop.is_running
    assert loop.state == LoopState.STOPPED


def test_start_twice_is_safe(force_closed):
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False, scan_interval_sec=0.05))
    loop.start(0.05)
    loop.start(0.05)                             # 第二次应被忽略
    try:
        assert loop.is_running
    finally:
        loop.stop()


# ======================================================================
# 状态快照
# ======================================================================

def test_get_status_shape(force_closed):
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    st = loop.get_status()
    assert st["state"] == LoopState.STOPPED.value
    assert st["plan_date"] == DAY
    assert st["plan_targets"] == 1
    assert st["guard"]["mode"] == "paper"


def test_tick_report_summary_readable(force_closed):
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    rep = loop.tick()
    txt = rep.summary()
    assert "状态=" in txt and "净值" in txt


# ======================================================================
# 时钟接缝（回归：这组用例曾整组在收盘后变红）
# ======================================================================

def test_force_trading_time_also_freezes_the_clock(force_trading_time):
    """夹具必须把**时钟**也冻住，只改日历是不够的。

    回归背景：``_should_settle`` 比的是墙上时钟是否已过 15:00。
    夹具原先只改 `calendar`，于是同一份用例 14:00 跑通过、16:00 跑必红 ——
    看起来像代码坏了，实际只是天晚了。这条断言把「时钟已冻结」钉住，
    以后谁把这段 monkeypatch 删掉，这里会立刻报出来。
    """
    import src.trading.loop as loopmod

    assert loopmod._now() == datetime(2026, 10, 14, 10, 0, 0), \
        "force_trading_time 没有冻结时钟 —— 这组用例又开始依赖运行时刻了"
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    assert loop._should_settle(loopmod._now()) is False


def test_loop_time_source_is_injectable(monkeypatch):
    """闭环的时间来源必须是可注入的 ``_now()``，而不是散落的 ``datetime.now()``。

    交易闭环是安全关键路径：能给出一个确定的时钟，才有可能把某天的行情
    原样重放一遍；否则「重放」这件事在结构上就做不到。
    """
    import src.trading.loop as loopmod

    monkeypatch.setattr(loopmod, "_now", lambda: datetime(2026, 3, 2, 9, 45, 0))
    assert loopmod._today() == date(2026, 3, 2)

    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    rep = loop.tick()
    assert rep.at == "09:45:00", "tick 报告的时刻没有走可注入时钟"


def test_settle_cannot_write_into_real_data_dir(force_trading_time, tmp_path):
    """**端到端**验证：一次真实结算 tick 不会在真实 data/loop_reports 留下任何东西。

    这条断言有两半，缺一不可：

    * 前半 —— 结算**确实发生了**（临时目录里生成了报告）。
      没有这一半，「真实目录没变」可能只是因为结算分支压根没跑，
      断言变成永真的空话。
    * 后半 —— 真实 ``data/loop_reports`` 逐字节没变。

    回归背景见 ``isolate_production_persistence`` 夹具的 docstring：
    曾有一份伪造的 ``loop_report_20261014.json`` 就这样落进了真实目录，
    还被 ``--doctor`` 当成真实业绩读。
    """
    import src.trading.loop as loopmod
    from config.settings import DATA_DIR as real_data_dir

    real_out = Path(real_data_dir) / "loop_reports"
    before = sorted(p.name for p in real_out.glob("loop_report_*")) if real_out.exists() else []

    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    loop = mk_loop(broker=b, paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, notify=False))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    loop.tick()                                   # 建仓
    loop._settled_dates.clear()
    # 恒真：强制走结算分支，使这条用例与运行时刻彻底解耦
    # （直接给实例属性赋值，随对象销毁自然还原，不动模块）。
    loop._should_settle = lambda now: True
    loop.tick()

    # 前半：结算真的发生了，只是落在了被隔离的临时目录
    assert list((Path(tmp_path) / "loop_reports").glob("loop_report_*.json")), \
        "结算分支没跑 —— 这条断言的后半就变成空话了"

    # 后半：真实目录纹丝不动
    after = sorted(p.name for p in real_out.glob("loop_report_*")) if real_out.exists() else []
    assert after == before, (
        "结算报告落进了真实 data/loop_reports —— 隔离夹具失效了。"
        f"新增 {sorted(set(after) - set(before))}"
    )
    assert Path(loopmod.DATA_DIR).resolve() != Path(real_data_dir).resolve()


class FakeNotifier:
    """记录所有推送调用，用于断言「某条通知只发一次」。"""

    def __init__(self):
        self.sent = []

    def send(self, title, markdown, payload=None):
        self.sent.append({"title": title, "markdown": markdown, "payload": payload})
        return True


def test_halt_push_fires_only_once_per_episode(force_trading_time):
    """熔断期间每个 tick 都推通知 = 轰炸（默认 notify=True 且接了通知渠道时）。

    必须在「进入熔断」那一刻只推一次；解除/自愈后**再次**熔断才重发。
    """
    notifier = FakeNotifier()
    b = FakeBroker()
    paper = FakePaper()
    loop = TradingLoop(
        b, guard=build_supervisor(b, paper_account=paper),
        paper_account=paper,
        config=LoopConfig(auto_build_plan=False), notifier=notifier,
    )
    loop.plan = mk_plan([("600519", 90, 0.30)])
    loop.guard.kill_switch("测试中控熔断")

    # 连续 5 个 tick：应只在第一次推送，其余 tick 保持静默
    for _ in range(5):
        loop.tick()
    halts = [s for s in notifier.sent if "已熔断" in s["title"]]
    assert len(halts) == 1, f"熔断通知被推送了 {len(halts)} 次，应仅 1 次"

    # 解除 → 重置标志位；再次熔断 → 必须能重发一次
    loop.guard.resume(operator="test")
    loop.tick()                                  # 未熔断 → _halt_notified 复位
    loop.guard.kill_switch("二次熔断")
    loop.tick()
    halts = [s for s in notifier.sent if "已熔断" in s["title"]]
    assert len(halts) == 2, f"解除后再熔断应重发，实际推送 {len(halts)} 次"


def test_settle_is_idempotent_across_process_restart(force_trading_time, tmp_path, monkeypatch):
    """常驻进程崩溃/重启后，今日结算报告已落盘 → 不再重复结算（防重推日报 + 防计数清零）。

    模拟：① 第一个实例结算一次并落盘运行态；② 新建第二个实例（=重启）后
    ``_should_settle`` 必须 False —— 即便内存清零，磁盘报告也能拦住重复结算。
    """
    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)

    loop1 = TradingLoop(
        b, guard=build_supervisor(b, paper_account=paper),
        paper_account=paper,
        config=LoopConfig(auto_build_plan=False, notify=False))
    loop1.plan = mk_plan([("600519", 90, 0.30)])
    loop1.tick()                                  # 建仓
    loop1._settled_dates.clear()
    loop1._should_settle = lambda now: True
    loop1.tick()                                  # 结算 → 写报告 + 落盘运行态

    report = tmp_path / "loop_reports" / "loop_report_20261014.json"
    assert report.exists(), "结算报告未生成，无法验证幂等"

    # ② 重启：新建实例（__init__ 会 _load_runtime 从磁盘读回 settled_dates）
    loop2 = TradingLoop(
        b, guard=build_supervisor(b, paper_account=paper),
        paper_account=paper,
        config=LoopConfig(auto_build_plan=False, notify=False))
    # 极端兜底：即便内存没读回运行态，磁盘报告也必须拦住重复结算
    loop2._settled_dates.clear()
    assert loop2._should_settle(datetime(2026, 10, 14, 15, 30)) is False, \
        "重启后不应重复结算（今日报告已存在 / 运行态已落盘）"


def test_day_start_equity_persists_across_restart(force_trading_time, tmp_path):
    """当日起始权益落盘后，重启的新实例能读回，避免 day_pnl 从 0 重算。"""
    paper = FakePaper(total=1_000_000.0)
    b = FakeBroker(total=1_200_000.0, cash=1_200_000.0)

    loop1 = TradingLoop(
        b, guard=build_supervisor(b, paper_account=paper),
        paper_account=paper,
        config=LoopConfig(auto_build_plan=False, notify=False))
    loop1.plan = mk_plan([("600519", 90, 0.30)])
    loop1.tick()                                  # 写入 _day_start_equity 并落盘
    today_s = "2026-10-14"
    assert today_s in loop1._day_start_equity

    # 重启：读回运行态
    loop2 = TradingLoop(
        b, guard=build_supervisor(b, paper_account=paper),
        paper_account=paper,
        config=LoopConfig(auto_build_plan=False, notify=False))
    assert loop2._day_start_equity.get(today_s) == loop1._day_start_equity[today_s], \
        "重启后当日起始权益未读回，day_pnl 会失真"


# ======================================================================
# 计划的时点语义（跨日不串台）
# ======================================================================

def _plan_for(day, targets=(("600519", 90, 0.30),)):
    """构造一份指定 plan_date 的计划。"""
    return TradePlan(
        plan_date=day,
        targets=[PlanTarget(code=c, name=c, score=s, weight=w)
                 for c, s, w in targets],
        cash_reserve_pct=0.10,
    )


def _counting_build(loop, monkeypatch, result=None):
    """把 ``_build_plan`` 换成计数器，用来断言「有没有重建」以及「重建的是哪一天」。"""
    box = {"n": 0, "targets": []}

    def _fake(target_date=None):
        box["n"] += 1
        box["targets"].append(target_date)
        return result

    monkeypatch.setattr(loop, "_build_plan", _fake)
    return box


def _fake_disk(monkeypatch, tp_mod, by_date=None, latest=None):
    """把「磁盘上的计划」替换成注入的计划（两个入口都要接管）。

    ⚠️ 必须**同时** patch ``load_trade_plan``（按日期精确查）和
    ``load_latest_plan``（按文件新旧查）。实现现在**优先按今天日期**加载，
    只接管其中一个的话，注入会失效、测试会读到真实的 ``data/trade_plans/``
    —— 而且今天恰好有真实计划时会“碰巧通过”，是最坏的那种假绿。
    """
    by_date = dict(by_date or {})
    monkeypatch.setattr(tp_mod, "load_trade_plan",
                        lambda d=None, plan_dir=None: by_date.get(d))
    monkeypatch.setattr(tp_mod, "load_latest_plan",
                        lambda plan_dir=None: latest)


def test_expired_in_memory_plan_is_reloaded_from_disk_before_rebuild(monkeypatch):
    """内存里是过期计划、磁盘上有「今天的」→ 必须用磁盘那份，不重建。

    回归背景（实跑发现）：常驻进程跨日时内存里的 D 计划失效，旧代码**直接**
    ``_build_plan()``。而 ``build_trade_plan()`` 按 ``resolve_target_date()``
    生成的是 **D+2** 的计划 —— 结果「拿后天的计划交易今天」，
    调度器昨晚为 D+1 写好的 ``trade_plan_D+1.json`` 从未被读过。
    """
    import src.trading.trade_plan as tp
    import src.trading.loop as loopmod

    today = loopmod._today()
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    today_s = today.strftime("%Y-%m-%d")

    loop = mk_loop(cfg=LoopConfig(auto_build_plan=True))
    loop.plan = _plan_for(yesterday)
    fresh = _plan_for(today_s, targets=[("000001", 80, 0.20)])
    _fake_disk(monkeypatch, tp, by_date={today_s: fresh}, latest=fresh)

    box = _counting_build(loop, monkeypatch)
    loop._ensure_plan(TickReport())

    assert box["n"] == 0, "磁盘上有今天的计划，却仍然重建了"
    assert loop.plan is fresh
    assert loop.plan.plan_date == today_s


def test_pending_plan_for_next_session_triggers_a_backfill_for_today(monkeypatch):
    """手上只有「下一交易日」的计划、磁盘上没有今天的 → 必须**补建今天的**。

    以前这条断言的是「保留待生效计划、不重建」（``plan_built is False``），
    理由是「重建结果与它同日，只是把同一个文件又写一遍」。但那只对
    ``_build_plan()`` 的默认语义成立 —— 它生成的是**下一个交易日**，所以
    「不重建」等于让今天整天没有计划。而这条分支**只在今天确实没有计划时**
    才会走到（正常情况第 ① 步就按日期命中并返回了）。

    实跑证据（2026-11-05 01:04）：调度器与 loop 同进程，11-04 晚 loop 崩溃重启，
    18:30/18:45/19:00 三个作业全没触发 → ``trade_plan_20261105.json`` 从未生成。
    loop 重启后 ``_build_plan()`` 按 ``resolve_target_date(11-05)=11-06`` 生成了
    11-06 的计划（用的还是 11-03 的旧 picks），今天(11-05)整天空转。

    现在改为补建今天那一份 —— 关键是**必须显式传 target_date=今天**，
    靠默认语义永远补不出来。
    """
    import src.trading.trade_plan as tp
    import src.trading.loop as loopmod

    today = loopmod._today()
    today_s = today.strftime("%Y-%m-%d")
    pending_day = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    loop = mk_loop(cfg=LoopConfig(auto_build_plan=True))
    loop.plan = _plan_for(pending_day)
    _fake_disk(monkeypatch, tp)          # 磁盘上什么都没有

    box = _counting_build(loop, monkeypatch, result=_plan_for(today_s))
    rep = TickReport()
    loop._ensure_plan(rep)

    assert box["n"] == 1, "今天没有计划却不补建 —— 这个交易日会一单不发"
    assert box["targets"] == [today_s], \
        f"补建时必须显式指定目标日=今天，实际传了 {box['targets']}"
    assert rep.plan_built is True
    assert loop.plan.plan_date == today_s


def test_backfilled_today_plan_is_not_rebuilt_on_every_tick(monkeypatch):
    """补建之后今天这份就是有效的，下一 tick 不许再重建（否则每 30 秒写一次盘）。"""
    import src.trading.trade_plan as tp
    import src.trading.loop as loopmod

    today_s = loopmod._today().strftime("%Y-%m-%d")
    pending_day = (loopmod._today() + timedelta(days=1)).strftime("%Y-%m-%d")

    loop = mk_loop(cfg=LoopConfig(auto_build_plan=True))
    loop.plan = _plan_for(pending_day)
    _fake_disk(monkeypatch, tp)

    box = _counting_build(loop, monkeypatch, result=_plan_for(today_s))
    loop._ensure_plan(TickReport())
    assert box["n"] == 1

    # 第二次 tick：手上已是今天的计划 → 直接返回
    loop._ensure_plan(TickReport())
    assert box["n"] == 1, "补建后的当天计划被判定为无效，导致每个 tick 都重建"


def _with_picks(plan, picks_date: str, lag: int):
    plan.picks_date = picks_date
    plan.picks_lag_days = lag
    return plan


def _warnings_of(fn):
    from loguru import logger as _lg
    msgs: list[str] = []
    hid = _lg.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        fn()
    finally:
        _lg.remove(hid)
    return [m for m in msgs if "选股结果" in m]


def test_stale_picks_on_a_disk_loaded_plan_are_reported(monkeypatch):
    """**从磁盘读回来的**旧计划也必须喊出来。

    这类计划最可疑：它可能在几天前就用旧票建好了（11-06 那批就是 11-03 的
    picks 生成的，与 11-04 那份目标逐位相同），当天加载时毫无征兆。
    ``build_trade_plan`` 的告警只在**生成时**触发，覆盖不到这条路径。
    """
    import src.trading.loop as loopmod

    today_s = loopmod._today().strftime("%Y-%m-%d")
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False))
    loop.plan = _with_picks(_plan_for(today_s), "2026-11-03", 3)

    hits = _warnings_of(lambda: loop._ensure_plan(TickReport()))
    assert len(hits) == 1, hits
    assert "滞后 3 个交易日" in hits[0]


def test_stale_picks_warning_fires_once_per_plan(monkeypatch):
    """``_ensure_plan`` 每 tick 都跑 —— 不去重会把日志刷成一片。"""
    import src.trading.loop as loopmod

    today_s = loopmod._today().strftime("%Y-%m-%d")
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False))
    loop.plan = _with_picks(_plan_for(today_s), "2026-11-03", 3)

    def _three_ticks():
        for _ in range(3):
            loop._ensure_plan(TickReport())

    assert len(_warnings_of(_three_ticks)) == 1


def test_fresh_picks_do_not_warn(monkeypatch):
    """滞后 1 是设计值（D-1 收盘选股 → D 日计划），不许报警。"""
    import src.trading.loop as loopmod

    today_s = loopmod._today().strftime("%Y-%m-%d")
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False))
    loop.plan = _with_picks(_plan_for(today_s), "2026-11-04", 1)

    assert _warnings_of(lambda: loop._ensure_plan(TickReport())) == []


def test_plan_is_built_when_nothing_usable_exists(monkeypatch):
    """确实没有任何可用计划时才重建 —— 这条保证上面那些「不重建」不是永真。"""
    import src.trading.trade_plan as tp
    import src.trading.loop as loopmod

    loop = mk_loop(cfg=LoopConfig(auto_build_plan=True))
    loop.plan = None
    _fake_disk(monkeypatch, tp)          # 磁盘上什么都没有

    built = _plan_for(loopmod._today().strftime("%Y-%m-%d"))
    monkeypatch.setattr(loop, "_build_plan", lambda target_date=None: built)

    rep = TickReport()
    loop._ensure_plan(rep)

    assert rep.plan_built is True
    assert loop.plan is built


def test_no_auto_build_never_rebuilds_and_skips_validity_check(monkeypatch):
    """``auto_build_plan=False`` 时只读盘、绝不重建，也不做过期判定。

    这是「人工管计划」的模式：运维把计划文件放好，进程照做，不该自作主张。
    """
    import src.trading.trade_plan as tp
    import src.trading.loop as loopmod

    stale = _plan_for((loopmod._today() - timedelta(days=30)).strftime("%Y-%m-%d"))
    loop = mk_loop(cfg=LoopConfig(auto_build_plan=False))
    loop.plan = None
    _fake_disk(monkeypatch, tp, latest=stale)   # 磁盘上只有这份手放的旧计划

    box = _counting_build(loop, monkeypatch)
    loop._ensure_plan(TickReport())

    assert box["n"] == 0
    assert loop.plan is stale, "auto_build_plan=False 时不该替换手放的计划"



# ======================================================================
# 取价口径：买入口径必须与估值口径同源（2026-10-20 幽灵亏损事故）
# ======================================================================

class DeadTickBroker(FakeBroker):
    """券商 tick 恒不可用（模拟盘实测如此），逼出兜底路径。"""

    def get_tick(self, code):
        return TickData(code=code, name=code, last_price=0.0, open=0.0,
                        high=0.0, low=0.0, pre_close=0.0,
                        volume=0, amount=0.0, bid1=0.0, ask1=0.0)


def test_price_of_prefers_realtime_over_stale_daily_close(monkeypatch, tmp_path):
    """券商 tick 挂了 → 必须取实时公开行情，不能一路退到日线 close(=昨收)。

    事故复盘：买入用实时价、估值用昨收，每笔持仓一买进去就"浮亏" 1%~3%，
    9 笔合计凭空 -¥5.5 万，直接把单日亏损熔断打爆，闭环当天停摆。
    """
    import src.data.fetcher as fetcher_mod

    monkeypatch.setattr(fetcher_mod.fetcher, "get_realtime_quote",
                        lambda code: {"代码": code, "最新价": 64.13, "昨收": 63.20})
    monkeypatch.setattr(
        fetcher_mod.fetcher, "get_daily_kline",
        lambda code: _FakeFrame(63.20))          # 昨收，退化值

    loop = TradingLoop(DeadTickBroker(), config=LoopConfig(
        dry_run=True, heartbeat_file=str(tmp_path / "hb.json")))
    loop._price_cache.clear()

    assert loop._price_of("002008") == pytest.approx(64.13), \
        "必须取实时价 64.13，而不是日线昨收 63.20"


def test_price_of_falls_back_to_daily_close_only_when_realtime_missing(monkeypatch, tmp_path):
    """实时行情也拿不到时，才允许退到日线 close（离线/盘后）。"""
    import src.data.fetcher as fetcher_mod

    monkeypatch.setattr(fetcher_mod.fetcher, "get_realtime_quote", lambda code: None)
    monkeypatch.setattr(fetcher_mod.fetcher, "get_daily_kline",
                        lambda code: _FakeFrame(63.20))

    loop = TradingLoop(DeadTickBroker(), config=LoopConfig(
        dry_run=True, heartbeat_file=str(tmp_path / "hb.json")))
    loop._price_cache.clear()

    assert loop._price_of("002008") == pytest.approx(63.20)


def test_mark_to_market_persists_so_other_processes_agree(tmp_path):
    """mark 完必须落盘：看板是另一个进程，只改内存它永远读到旧价。"""
    from src.trading.paper import PaperAccount

    pa = PaperAccount(account_file=tmp_path / "pa.json",
                      history_file=tmp_path / "ph.json")
    pa.holdings["002008"] = {"code": "002008", "name": "大族激光",
                             "quantity": 100, "avg_cost": 63.20,
                             "last_price": 63.20, "market_value": 6320.0}
    pa.mark_to_market({"002008": 64.13})
    pa.save()

    reread = PaperAccount(account_file=tmp_path / "pa.json",
                          history_file=tmp_path / "ph.json")
    assert reread.holdings["002008"]["last_price"] == pytest.approx(64.13), \
        "落盘后重开必须读到 mark 后的价，否则跨进程口径不一致"


class _FakeFrame:
    """最小 DataFrame 替身，只需支持 df['close'].iloc[-1]。"""

    def __init__(self, close: float):
        self.close = _FakeSeries(close)

    def __getitem__(self, key):
        return self.close if key == "close" else _FakeSeries(0.0)

    def __len__(self):
        return 1


class _FakeSeries:
    def __init__(self, v):
        self._v = v
        self.iloc = self

    def __getitem__(self, i):
        return self._v


# ======================================================================
# 无人值守闭环的纸账户重置安全（关键回归）
# ======================================================================

def test_paper_reset_external_disk_change_realigns_guard_and_keeps_running(force_trading_time, tmp_path):
    """回归：常驻进程运行期间，用户从看板「重置账户」改了 initial_cash，
    闭环每 tick 从磁盘重载纸账户，必须①把外部重置反映进内存，②检测本金
    基线(initial_cash)变更并自动把风控净值基线(peak_equity)对齐到新权益。

    若缺①：常驻进程永远用内存里的陈旧副本交易，用户点了「重置」也视而不见。
    若缺②：旧 peak(100万) 远高于新权益(10万) 会被误判成 90% 腰斩→熔断，
            且无法自愈，当天剩余时间全冻结 —— 这是无人值守循环最危险的静默停摆。

    本用例用「第二个实例写盘」模拟看板进程重置，从而同时锁住①和②：
      · 断言 cash 被重载为 10 万 → 抓「不重载」的退化；
      · 断言未进入 HALTED 且 peak_equity 对齐到 10 万 → 抓「不重对齐」的误熔断。
    """
    from src.trading.paper import PaperAccount

    account_file = tmp_path / "paper_account.json"
    history_file = tmp_path / "paper_history.json"
    paper = PaperAccount(initial_cash=1_000_000.0,
                         account_file=account_file, history_file=history_file)
    b = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    loop = mk_loop(broker=b, paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    loop.plan = mk_plan([("600519", 90, 0.30)])

    # tick 1：建立基线，guard peak → 1,000,000
    rep1 = loop.tick()
    assert rep1.state == LoopState.RUNNING.value
    assert loop.guard.safety._state.peak_equity == pytest.approx(1_000_000.0, rel=1e-3)

    # 模拟「另一个进程（看板）」重置账户到 10 万（initial_cash 跟着变，默认保留历史）
    PaperAccount(initial_cash=1_000_000.0,
                 account_file=account_file, history_file=history_file).reset(100_000.0)

    # tick 2：闭环从磁盘重载 → 检测到 initial_cash 变更 → 自动对齐基线
    rep2 = loop.tick()
    # ① 重载必须反映外部重置
    assert loop.guard.paper_account.cash == pytest.approx(100_000.0, rel=1e-3), \
        "闭环未从磁盘重载纸账户，外部「重置账户」被无视"
    assert loop.guard.paper_account.holdings == {}, \
        "重置后持仓应被清空"
    # ② 绝不能因旧 peak 误判腰斩而熔断
    assert rep2.state != LoopState.HALTED.value, (
        f"重置账户后误熔断！state={rep2.state}, note={rep2.note}")
    assert loop.guard.safety._state.peak_equity == pytest.approx(100_000.0, rel=1e-3), \
        "风控净值基线必须对齐到重置后的权益，否则旧 peak 仍会触发误熔断"
    # 且闭环继续正常运行（不是卡死在 HALTED/PREOPEN/IDLE）
    assert rep2.state == LoopState.RUNNING.value


def test_loop_refuses_rebaseline_when_baseline_changes_without_reset_evidence(force_trading_time, tmp_path):
    """回归（2026-10-30 事故）：本金基线变了但**没有 reset() 证据** → 拒绝重对齐。

    实况：09:42 `data/paper_account.json` 从「现金 ¥4,218,917 / 1 只持仓 /
    本金基线 ¥4,995,224」被空白单例覆盖成「10 万 / 0 持仓 / 10 万」，随后闭环
    把这个变化当成「用户重置」，rebaseline_peak 把峰值对齐到 10 万 —— 回撤从
    10.86% 变成 0%，**熔断被静默解除**，且 journal 里只有一条 autoloop 的
    rebaseline_peak，没有任何告警。

    判据：只有 `PaperAccount.reset()` 会写 `reset_at`。基线变了而 reset_at 没动
    ⇒ 不是人工重置，是状态被写坏 ⇒ 必须拒绝放宽基线、保留熔断并留痕。
    """
    from src.trading.guard import GuardConfig, RiskSupervisor
    from src.trading.paper import PaperAccount

    account_file = tmp_path / "paper_account.json"
    history_file = tmp_path / "paper_history.json"
    paper = PaperAccount(initial_cash=1_000_000.0,
                         account_file=account_file, history_file=history_file)
    paper.save()                                    # dry_run 不落盘，手动建盘
    broker = FakeBroker(total=1_000_000.0, cash=1_000_000.0)
    guard = RiskSupervisor(broker, GuardConfig(
        state_file=str(tmp_path / "gs.json"),
        journal_file=str(tmp_path / "gj.jsonl"),
    ), paper_account=paper)
    loop = mk_loop(broker=broker, paper=paper, guard=guard,
                   cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    loop.plan = mk_plan([("600519", 90, 0.30)])

    loop.tick()                                     # 建立基线 peak=1,000,000
    peak_before = loop.guard.safety._state.peak_equity
    assert peak_before == pytest.approx(1_000_000.0, rel=1e-3)

    # 绕过 PaperAccount 直接改盘 —— 这就是「状态被写坏」的样子：基线变了，
    # reset_at 还是旧的（空），没有 reset() 留下的证据。
    raw = json.loads(account_file.read_text(encoding="utf-8"))
    raw.update({"cash": 100_000.0, "initial_cash": 100_000.0, "holdings": {}})
    account_file.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    calls, events = [], []
    _orig_rb = loop.guard.safety.rebaseline_peak
    _orig_emit = loop.guard._emit
    loop.guard.safety.rebaseline_peak = lambda *a, **k: (calls.append(a), _orig_rb(*a, **k))[1]
    loop.guard._emit = lambda ev, data: (events.append(ev), _orig_emit(ev, data))[1]

    rep = loop.tick()

    assert calls == [], "没有 reset() 证据就重对齐基线 = 静默放宽熔断线"
    assert loop.guard.safety._state.peak_equity == pytest.approx(peak_before), \
        "峰值必须保持原值，否则回撤会被重新算成 0"
    assert "rebaseline_blocked" in events, "拒绝重对齐必须留痕（事后审计要看得见）"
    # 90% 的假回撤照常熔断 —— 宁可停摆也不静默放开
    assert loop.guard.safety.halted is True
    assert rep.state == LoopState.HALTED.value


def test_loop_allows_rebaseline_when_reset_evidence_present(force_trading_time, tmp_path):
    """对照：确实走过 reset()（reset_at 前进）→ 照常对齐基线，不误熔断。"""
    from src.trading.paper import PaperAccount

    account_file = tmp_path / "paper_account.json"
    history_file = tmp_path / "paper_history.json"
    paper = PaperAccount(initial_cash=1_000_000.0,
                         account_file=account_file, history_file=history_file)
    loop = mk_loop(broker=FakeBroker(total=1_000_000.0, cash=1_000_000.0),
                   paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    loop.plan = mk_plan([("600519", 90, 0.30)])
    loop.tick()
    assert paper.reset_at == "", "reset() 之外的路径不该写 reset_at"

    paper.reset(500_000.0)          # 人工重置：reset_at 前进
    rep = loop.tick()
    assert loop.guard.safety._state.peak_equity == pytest.approx(500_000.0, rel=1e-3)
    assert rep.state != LoopState.HALTED.value


def test_loop_only_rebaselines_on_initial_cash_change_not_on_cash_change(force_trading_time, tmp_path):
    """回归：本金基线(_last_pa_baseline)只在 initial_cash 变更时才重对齐，
    普通市值波动（cash 变化）绝不应触发 rebaseline —— 否则会把真实的腰斩
    当成「重置」而静默放宽熔断线，违反回撤熔断纪律。

    用 spy 记录 rebaseline_peak 的调用：普通亏损一 tick 不应调用；重置改了
    initial_cash 的那一 tick 才调用一次。
    """
    from src.trading.paper import PaperAccount

    account_file = tmp_path / "paper_account.json"
    history_file = tmp_path / "paper_history.json"
    paper = PaperAccount(initial_cash=1_000_000.0,
                         account_file=account_file, history_file=history_file)
    loop = mk_loop(broker=FakeBroker(total=1_000_000.0, cash=1_000_000.0),
                   paper=paper,
                   cfg=LoopConfig(auto_build_plan=False, dry_run=True))
    loop.plan = mk_plan([("600519", 90, 0.30)])

    loop.tick()  # 建立基线 _last_pa_baseline=1,000,000，guard peak=1,000,000

    calls = []
    orig = loop.guard.safety.rebaseline_peak
    def _spy(*a, **k):
        calls.append((a, k))
        return orig(*a, **k)
    loop.guard.safety.rebaseline_peak = _spy

    # 普通市值波动：只改 cash（写盘），不动 initial_cash
    pa2 = PaperAccount(initial_cash=1_000_000.0,
                       account_file=account_file, history_file=history_file)
    pa2.cash = 950_000.0
    pa2._save()
    loop.tick()
    assert calls == [], (
        "普通市值波动不应触发 rebaseline_peak，否则会把真实腰斩当成重置放宽熔断")

    # 重置账户：initial_cash 变更 → 必须触发一次
    paper.reset(500_000.0)
    loop.tick()
    assert len(calls) == 1, (
        f"initial_cash 变更必须触发一次 rebaseline_peak，实际 {len(calls)} 次")
    assert loop.guard.safety._state.peak_equity == pytest.approx(500_000.0, rel=1e-3)


# ======================================================================
# 退出原因细分（归因可观测性）
# ======================================================================

def test_exit_tag_maps_reason_to_attributable_label():
    """止盈 / 止损 / 移动止损必须是三个不同标签。

    否则归因表里只剩一个 ``stop`` 桶，**「止盈从来没触发过」这个决定性事实
    完全看不见** —— 实测就是这情况：配的止盈 +15%，实际最大盈利 +1.62%。
    """
    from src.trading.loop import _exit_tag

    assert _exit_tag("止盈 15.0%") == "stop_profit"
    assert _exit_tag("止损 -5.2%") == "stop_loss"
    assert _exit_tag("移动止损 回撤-3.1% (最高+8.2%)") == "trail_stop"
    # 「移动止损」含「止损」二字，优先级必须高于 stop_loss
    assert _exit_tag("移动止损 -3%") != "stop_loss"


def test_exit_tag_does_not_guess_when_ambiguous():
    """兜底文案「止盈止损」说不清方向 → 记 exit，不许瞎猜。

    猜错的标签会污染归因表，比缺失更难发现。
    """
    from src.trading.loop import _exit_tag

    assert _exit_tag("止盈止损") == "exit"
    assert _exit_tag("") == "exit"
    assert _exit_tag("封板无量") == "exit"


@pytest.mark.parametrize("reason,expect", [
    ("止盈 15.0%", "stop_profit"),
    ("止损 -5.2%", "stop_loss"),
    ("移动止损 回撤-3.1%", "trail_stop"),
])
def test_run_exits_passes_fine_grained_tag(monkeypatch, reason, expect):
    """闭环真正下单时带的是细分标签，而不是笼统的 stop。"""
    loop = mk_loop()
    seen = {}

    monkeypatch.setattr(
        loop.guard.executor, "check_stop_conditions",
        lambda positions=None, prices=None: [{
            "code": "600519", "name": "贵州茅台", "reason": reason,
            "action": "sell", "pnl_pct": -5.0, "quantity": 100,
        }])

    def _fake_place(**kw):
        seen.update(kw)
        return True, "ok", None, {"code": kw.get("code")}

    monkeypatch.setattr(loop, "_place", _fake_place)
    loop._run_exits(positions=[], prices={"600519": 10.0}, report=TickReport())

    assert seen.get("tag") == expect
    assert seen.get("reason") == reason, "reason 原样透传给日志/日报"


def test_regime_weights_keep_risk_exit_always_allowed():
    """风控离场在任何市场状态下权重都是 1.0（不被 regime 降权）。

    新加的三个细分标签必须显式登记 —— 靠 DEFAULT_TAG_WEIGHT 兜底虽然也是
    1.0，但「永远执行」这条保证不该依赖默认值，那是会被后人改掉的。
    """
    from src.trading.regime import REGIME_STRATEGY_WEIGHTS

    for regime in ("bull", "sideways", "bear"):
        w = REGIME_STRATEGY_WEIGHTS[regime]
        for tag in ("stop", "stop_loss", "stop_profit", "trail_stop"):
            assert w.get(tag) == pytest.approx(1.0), f"{regime}/{tag} 会被降权"


# ======================================================================
# T3 组合层风控覆盖层（非顺周期降仓 + 波动率目标）
# ======================================================================

def _loop_with_dd(peak: float, total: float, **cfg_kw):
    """造一个「峰值 peak / 当前权益 total」的 loop，用于直接喂回撤。

    ⚠️ 默认把波动腿关掉并清空 ``_day_start_equity``：loop 构造时会 ``_load_runtime()``
    读到**真实的** ``data/loop_runtime.json``，那里的波动率会让波动腿随机介入，
    使「只验回撤腿」的断言变成薛定谔的测试。
    """
    cfg_kw.setdefault("vol_target_pct", 0.0)
    b = FakeBroker(total=total, cash=total)
    loop = mk_loop(broker=b, paper=FakePaper(total=total),
                   cfg=LoopConfig(auto_build_plan=False, dry_run=True, **cfg_kw))
    loop._day_start_equity = {}
    loop.guard.safety._state.peak_equity = float(peak)
    return loop, b.query_account()


def test_t3_production_switches_are_off_after_slack3_evidence():
    """T3 覆盖层**生产关闭** + 参数留档（slack=3.0 实测负贡献）。

    踩过的坑：在 slack=1.0 口径下 vt22 看起来是正贡献，换上生产口径 slack=3.0
    之后收益/夏普/Calmar 三档全输。原因是「波动率目标」和「排名滞后」都在降
    换手、降回撤，**是替代关系不是叠加关系**，叠加等于重复扣分。
    参数保留是为了让后续实验有个已记录的起点，不是因为它们在生效。
    """
    from config.settings import AppConfig

    for c in (LoopConfig(), AppConfig().trading_loop):
        assert c.derisk_mode == "off", "生产必须关闭：slack=3.0 下是负贡献"
        assert c.vol_target_pct == 0.0
        assert c.risk_blend == "mul"
        # 参数留档（改这些不会生效，但开回来时要保持非顺周期）
        assert (c.derisk_engage_pct, c.derisk_max_dd_pct, c.derisk_floor) == (12.0, 20.0, 0.60)
        assert (c.derisk_release_hysteresis_pct, c.vol_scale_floor) == (2.0, 0.5)


def test_t3_params_never_regress_to_procyclical():
    """硬约束：介入线不得 < 10%、两个地板不得 < 0.5。

    把 engage 调回 0 就是被证伪的顺周期版（回测里吃掉 9pp 收益）；
    地板调到 0.5 以下就变成「跌了就清仓」。将来谁要重新启用，先过这条线。
    """
    from config.settings import AppConfig

    for c in (LoopConfig(), AppConfig().trading_loop):
        assert c.derisk_engage_pct >= 10.0, "介入线过低 = 顺周期，回测已证伪"
        assert c.derisk_floor >= 0.5, "地板 < 0.5 = 接近清仓"
        assert c.vol_scale_floor >= 0.5, "波动腿地板 < 0.5 = 波动一高就停手"
        assert c.derisk_max_dd_pct > c.derisk_engage_pct, "斜坡上限必须大于介入线"


def test_t3_off_mode_never_shrinks_even_at_deep_drawdown():
    """开关关闭时，20% 回撤也必须返回 1.0（回滚路径必须真的关得掉）。"""
    loop, acct = _loop_with_dd(1_000_000.0, 800_000.0, derisk_mode="off")
    assert loop._portfolio_risk_scale(acct) == 1.0


def test_t3_asymmetric_does_not_engage_below_engage_line():
    """回撤 < 12% 完全不干预 —— 这是「非顺周期」与旧版的唯一区别所在。

    旧的 portfolio_risk_scale 从 dd=0 就开始砍，8% 回撤时系数已 0.55；
    本方案 8% 必须仍是 1.0，否则又变成「跌一点就缩手」。
    """
    loop, acct = _loop_with_dd(1_000_000.0, 920_000.0, derisk_mode="asymmetric")
    assert loop._portfolio_risk_scale(acct) == 1.0
    # 11.9% 也不该介入
    loop2, acct2 = _loop_with_dd(1_000_000.0, 881_000.0, derisk_mode="asymmetric")
    assert loop2._portfolio_risk_scale(acct2) == 1.0


def test_t3_asymmetric_ramps_then_floors():
    """12%→20% 线性压到 0.60；超过 20% 一直是 0.60（不清仓、不归零）。"""
    loop, acct = _loop_with_dd(1_000_000.0, 840_000.0, derisk_mode="asymmetric")  # dd 16%
    s16 = loop._portfolio_risk_scale(acct)
    assert 0.60 < s16 < 1.0
    assert s16 == pytest.approx(0.8, abs=0.02)

    loop2, acct2 = _loop_with_dd(1_000_000.0, 700_000.0, derisk_mode="asymmetric")  # dd 30%
    assert loop2._portfolio_risk_scale(acct2) == pytest.approx(0.60)


def test_t3_asymmetric_ratchets_and_hysteresis():
    """棘轮 + 滞回：介入后小反弹不放开，修到 10% 以下才完全回 1.0。

    没有滞回就会出现「15% 减仓 → 反弹到 12.5% 满仓 → 再跌回 15% 再减仓」
    的来回摩擦，正是顺周期最烧钱的地方。
    """
    loop, _ = _loop_with_dd(1_000_000.0, 840_000.0, derisk_mode="asymmetric")
    assert loop._portfolio_risk_scale(_acct(840_000.0)) == pytest.approx(0.8, abs=0.02)
    # 反弹到 88.5 万（dd 11.5%，已低于介入线但高于放开线 10%）
    s = loop._portfolio_risk_scale(_acct(885_000.0))
    assert s == pytest.approx(0.8, abs=0.02), "滞回区间内应保持介入以来的水平"
    # 修到 9%（< 10% 放开线）→ 完全放开
    assert loop._portfolio_risk_scale(_acct(910_000.0)) == 1.0


class _Acct:
    def __init__(self, total):
        self.total_asset = float(total)


def _acct(total: float):
    return _Acct(total)


def test_t3_vol_target_only_shrinks_and_fails_open():
    """波动腿：高波动才缩，低波动绝不放杠杆；样本不足时不干预。"""
    loop = mk_loop(broker=FakeBroker(total=1_000_000.0, cash=1_000_000.0),
                   cfg=LoopConfig(auto_build_plan=False, dry_run=True,
                                  vol_target_pct=15.0))
    # 样本不足（< min_obs）→ fail-open
    loop._day_start_equity = {"2026-01-0%d" % i: 1_000_000.0 for i in range(1, 6)}
    assert loop._vol_target_scale() == 1.0

    # 日波动 ±5% → 年化 ~80% → 0.15/0.80 低于地板 0.5 → 夹到 0.5
    eq = {}
    v = 1_000_000.0
    for i in range(25):
        eq["2026-02-%02d" % (i + 1)] = v
        v *= 1.05 if i % 2 == 0 else 0.95
    loop._day_start_equity = eq
    assert loop._vol_target_scale() == pytest.approx(0.5)

    # 日波动 ±0.3% → 年化 ~5% < 目标 15% → 只缩不放，返回 1.0
    eq2 = {}
    v = 1_000_000.0
    for i in range(25):
        eq2["2026-03-%02d" % (i + 1)] = v
        v *= 1.003 if i % 2 == 0 else 0.997
    loop._day_start_equity = eq2
    assert loop._vol_target_scale() == 1.0


def test_t3_blend_is_multiplicative_by_default():
    """默认 mul（= 回测 E 方案口径）：两腿都在时系数是两者相乘。"""
    loop, acct = _loop_with_dd(1_000_000.0, 700_000.0,
                               derisk_mode="asymmetric", vol_target_pct=15.0)
    eq = {}
    v = 1_000_000.0
    for i in range(25):
        eq["2026-02-%02d" % (i + 1)] = v
        v *= 1.05 if i % 2 == 0 else 0.95
    loop._day_start_equity = eq
    # 回撤 30% → 0.60；波动腿 → 0.50；mul → 0.30
    assert loop._portfolio_risk_scale(acct) == pytest.approx(0.30)


def test_from_app_config_restores_execution_object():
    """回归：asdict 会把嵌套配置变 dict，必须还原成对象。

    否则 ``getattr(dict, "smart_routing")`` 恒 False —— T2 智能下单
    「配置里开了、进程里没开」，而且不报任何错。
    """
    import src.trading.loop as loopmod
    from config.settings import ExecutionConfig, TradingLoopConfig

    sec = TradingLoopConfig(
        execution=ExecutionConfig(smart_routing=True, slice_strategy="vwap",
                                  max_slices=6))
    # asdict 只认真 dataclass，所以必须用 TradingLoopConfig 实例当替身
    old = loopmod.app_config
    try:
        loopmod.app_config = type("C", (), {"trading_loop": sec})()
        cfg = loopmod.LoopConfig.from_app_config()
        assert isinstance(cfg.execution, ExecutionConfig)
        assert cfg.execution.smart_routing is True
        assert cfg.execution.slice_strategy == "vwap"
        assert cfg.execution.max_slices == 6
        # 且真的能建出调度器（原来 dict 时这里恒为 None）
        loop = mk_loop(cfg=cfg)
        assert loop._exec_scheduler is not None
    finally:
        loopmod.app_config = old


def test_backtest_scripts_must_pass_production_slack():
    """口径守卫：回测脚本必须显式传 hold_rank_slack，且等于生产值。

    2026-11-03 踩过的坑：`ExecConfig.hold_rank_slack` 默认 **1.0**，而生产是 **3.0**。
    一个漏传就让「波动率目标有效」的结论翻转成「无效」—— 同一批数据、同一个策略。
    这类口径漂移在代码里看不出来（不报错、不告警），只能靠断言守着。

    ``rebalance_days`` 同理：它和 ``hold_rank_slack`` 都**不在可进化参数空间里**，
    却决定策略身份。脚本里只要提到周期，就必须有一个方案等于生产周期，
    否则「15 日周期」的结论就来自一个别的周期。
    """
    import re
    from config.settings import AppConfig

    prod = float(AppConfig().trading_loop.hold_rank_slack)
    assert prod == 3.0, "生产 slack 变了就必须重新验证所有 slack 敏感结论"
    prod_rb = int(AppConfig().trading_loop.rebalance_days)

    for name in ("validate_risk_overlay.py", "exec_ab.py", "exec_phase.py"):
        p = ROOT / "scripts" / name
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        hits = re.findall(r"hold_rank_slack\s*=\s*([0-9.]+)", src)
        assert hits, f"{name} 没显式传 hold_rank_slack，会用 ExecConfig 默认 1.0"
        assert any(abs(float(h) - prod) < 1e-9 for h in hits), (
            f"{name} 里没有任何方案用生产值 {prod}")
        rb_hits = re.findall(r"rebalance_days\s*=\s*([0-9]+)", src)
        if rb_hits:
            assert any(int(h) == prod_rb for h in rb_hits), (
                f"{name} 提到 rebalance_days 但没有一个等于生产值 {prod_rb} ⇒ "
                "结论来自别的周期")

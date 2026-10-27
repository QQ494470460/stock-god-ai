# -*- coding: utf-8 -*-
"""报价合法性守卫测试。

核心保证：**一个坏价格不能驱动一笔交易**。四类真实事故各自一条用例：
板限越界、真实涨跌停（需二次确认）、停牌零成交、无昨收时的相对跳变。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading.sanity import QuoteSanityGuard, board_limit_pct, filter_prices, is_st


# ---------------------------------------------------------------- 板限
@pytest.mark.parametrize("code,name,expect", [
    ("600519", "贵州茅台", 10.0),
    ("000001", "平安银行", 10.0),
    ("300750", "宁德时代", 20.0),
    ("301155", "某某科技", 20.0),
    ("688981", "中芯国际", 20.0),
    ("689009", "九号公司", 20.0),
    ("830799", "艾融软件", 30.0),
    ("430047", "诺思兰德", 30.0),
    ("600519", "ST茅台", 5.0),
])
def test_board_limit_by_code(code, name, expect):
    assert board_limit_pct(code, name) == expect


def test_is_st_detects_prefix_and_delist():
    assert is_st("*ST海航")
    assert is_st("ST长生")
    assert is_st("退市锐电")
    assert not is_st("贵州茅台")


# ---------------------------------------------------------------- 主校验
def test_normal_quote_passes():
    g = QuoteSanityGuard()
    v = g.check("600519", 100.0, pre_close=99.0, volume=1000, amount=1e5)
    assert v.ok and v.tradable and v.price == 100.0


def test_beyond_board_limit_rejected_once_then_confirmed():
    """单次越界不采信（防数据抖动），连续两次同向才认定为真实跳空。"""
    g = QuoteSanityGuard(confirm_ticks=2, tolerance_pct=2.0)
    g.check("600519", 100.0, pre_close=100.0)          # 建立可信价
    v1 = g.check("600519", 200.0, pre_close=100.0)     # +100% 明显异常
    assert v1.ok is False
    assert v1.price == 100.0, "异常时应沿用上一次可信价，而不是用 0 让止损失明"
    assert "板限" in v1.why

    v2 = g.check("600519", 200.0, pre_close=100.0)     # 第二次同向 → 采信
    assert v2.ok is True
    assert v2.price == 200.0


def test_opposite_direction_resets_counter():
    """方向一正一负交替 → 计数重置，永远不会被确认（这才是抖动）。"""
    g = QuoteSanityGuard(confirm_ticks=2)
    g.check("600519", 100.0, pre_close=100.0)
    assert g.check("600519", 130.0, pre_close=100.0).ok is False
    assert g.check("600519", 70.0, pre_close=100.0).ok is False
    assert g.check("600519", 130.0, pre_close=100.0).ok is False


def test_limit_up_inside_board_is_accepted():
    """真实涨停（创业板 +20%）在板限内，直接采信。"""
    g = QuoteSanityGuard()
    v = g.check("300750", 120.0, pre_close=100.0)
    assert v.ok is True


def test_suspended_quote_tradable_false():
    """停牌：可估值、不可交易（不能因为停牌就把持仓市值打成 0）。"""
    g = QuoteSanityGuard()
    v = g.check("600519", 100.0, pre_close=100.0, volume=0, amount=0)
    assert v.ok is True
    assert v.tradable is False
    assert "停牌" in v.why


def test_missing_price_keeps_last_known():
    g = QuoteSanityGuard()
    g.check("600519", 100.0, pre_close=100.0)
    v = g.check("600519", 0.0)
    assert v.ok is False
    assert v.price == 100.0


def test_no_preclose_uses_jump_check():
    """没有昨收时退化为「相对上次可信价」的跳变检查，宁松不严。"""
    g = QuoteSanityGuard(jump_pct=15.0, stale_suspect=3)
    g.check("600519", 100.0)
    v = g.check("600519", 200.0)
    assert v.ok is False and v.price == 100.0
    # 连续 3 次后强制采信，避免永久卡在旧价上
    g.check("600519", 200.0)
    assert g.check("600519", 200.0).ok is True


def test_reset_clears_state():
    g = QuoteSanityGuard()
    g.check("600519", 100.0, pre_close=100.0)
    g.reset()
    assert g.last_price("600519") == 0.0


def test_filter_prices_drops_nonpositive():
    out = filter_prices({"600519": 100.0, "000001": 0.0, "300750": -1.0})
    assert out == {"600519": 100.0}


# ---------------------------------------------------------------- 接入闭环
class _BadQuoteBroker:
    """模拟数据源给出 100 倍错位价（分/元单位错误）。"""

    is_simulated = False

    def __init__(self, price, pre_close=None):
        self.price = price
        self.pre_close = pre_close if pre_close is not None else price

    def get_tick(self, code):
        from src.trading.broker import TickData
        return TickData(code=code, name=code, last_price=self.price,
                        open=self.price, high=self.price, low=self.price,
                        pre_close=self.pre_close, volume=1000, amount=1e5)

    # guard 构造 TradeExecutor 时会查账户，缺了就 AttributeError
    def query_account(self):
        from src.trading.broker import AccountInfo
        return AccountInfo(account_id="FAKE", total_asset=1_000_000.0,
                           available_cash=1_000_000.0, frozen_cash=0.0,
                           market_value=0.0, total_pnl=0.0, today_pnl=0.0)

    def query_positions(self): return []
    def query_orders(self, status_filter="all"): return []
    def cancel_order(self, order_id): return True
    def cancel_all(self): return 0
    def place_order(self, *a, **k): return None
    def get_broker_name(self): return "bad-quote"
    def is_trading_time(self): return True


def test_loop_price_guard_rejects_absurd_quote(tmp_path, monkeypatch):
    """闭环取价链路必须拦住 100 倍错位价 —— 否则止损会在假价格上触发。"""
    from src.trading.guard import GuardConfig, RiskSupervisor
    from src.trading.loop import LoopConfig, TradingLoop
    from src.trading.paper import PaperAccount

    broker = _BadQuoteBroker(price=10.0, pre_close=10.0)
    cfg = GuardConfig(state_file=str(tmp_path / "g.json"),
                      journal_file=str(tmp_path / "j.jsonl"))
    # 纸账户必须显式指向 tmp：不传就会用生产 data/paper_account.json
    pa = PaperAccount(initial_cash=100_000.0,
                      account_file=tmp_path / "pa.json",
                      history_file=tmp_path / "ph.json")
    loop = TradingLoop(broker, guard=RiskSupervisor(broker, cfg, paper_account=pa),
                       paper_account=pa,
                       config=LoopConfig(auto_build_plan=False,
                                         heartbeat_file=str(tmp_path / "hb.json")))

    assert loop._price_of("600519") == pytest.approx(10.0)   # 建立可信价
    broker.price = 1000.0                                    # 数据源抽风（×100）
    loop._price_cache.clear()
    assert loop._price_of("600519") == pytest.approx(10.0), "异常报价被采信了"
    loop._price_cache.clear()
    # 连续第二次同向 → 说明是真跳空，采信
    assert loop._price_of("600519") == pytest.approx(1000.0)

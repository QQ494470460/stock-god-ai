# -*- coding: utf-8 -*-
"""环路心跳 + 纸账户市值归一化测试。

两个都是「看不见的失真」：

1. 闭环的 ③交易 没有定时任务驱动，全靠人工启动 ``run_loop.py`` / 点看板按钮。
   10-14~10-16 三天的结算报告都是 ``intents=0``，事后翻日志才分清是「根本没跑」
   还是「跑了但没满足调仓条件」。心跳文件把这两种情况分开。
2. 纸账户 ``market_value`` 买入时按成交价写死，之后 ``last_price`` 更新它不跟，
   实测出现 quantity=300 / last_price=1501.95 / market_value=450000（应 450585）。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading.broker import (
    BrokerAPI, AccountInfo, Order, OrderSide, OrderStatus, TickData,
)
from src.trading.guard import GuardConfig, RiskSupervisor
from src.trading.loop import LoopConfig, TickReport, TradingLoop, read_heartbeat
from src.trading.paper import PaperAccount


class _FakeBroker(BrokerAPI):
    def __init__(self, last_price: float = 10.0):
        super().__init__()
        self._connected = True
        self.last_price = last_price
        self.orders = []

    def connect(self, **k): self._connected = True; return True
    def disconnect(self): self._connected = False

    def query_account(self):
        return AccountInfo("FAKE", 1_000_000.0, 600_000.0, 0.0, 400_000.0, 0.0, 0.0)

    def query_positions(self): return []
    def query_orders(self, *a, **k): return []

    def place_order(self, code, side, order_type, price, quantity, strategy_tag=""):
        o = Order(f"ORD{len(self.orders)+1:03d}", code, side, order_type, price,
                  quantity, quantity, OrderStatus.FILLED, strategy_tag=strategy_tag)
        self.orders.append(o)
        return o

    def cancel_order(self, oid): return True
    def cancel_all(self): return 0

    def get_tick(self, code):
        p = self.last_price
        return TickData(code, code, p, p, p, p, p, 100000, 1e6, p, p, 0.0)

    def get_broker_name(self): return "FakeBroker"
    def is_trading_time(self): return True


def _make_loop(tmp_path, hb_path: Path):
    cfg = LoopConfig(
        heartbeat_file=str(hb_path),
        dry_run=True,
        auto_build_plan=False,      # 绝不在测试里生成/写盘交易计划
        enable_rebalance=False,
        enable_stop_orders=False,
    )
    gcfg = GuardConfig(
        state_file=str(tmp_path / "guard_state.json"),
        journal_file=str(tmp_path / "guard_journal.jsonl"),
    )
    guard = RiskSupervisor(_FakeBroker(), gcfg)
    return TradingLoop(_FakeBroker(), guard=guard, config=cfg)


# ======================================================================
# 心跳
# ======================================================================
def test_tick_writes_heartbeat(tmp_path, monkeypatch):
    """每次 tick 都要落一次心跳，字段可供 UI/告警判断「还活着吗」。"""
    hb = tmp_path / "loop_heartbeat.json"
    loop = _make_loop(tmp_path, hb)

    fake = TickReport(at="09:31:00", state="running", market_status="盘中",
                      plan_date="2026-10-19", n_positions=1, equity=4_943_923.78,
                      n_intents=10, n_executed=3, n_blocked=7)
    monkeypatch.setattr(loop, "_tick_impl", lambda: fake)

    rep = loop.tick()
    assert rep is fake

    data = read_heartbeat(hb)
    assert data["state"] == "running"
    assert data["plan_date"] == "2026-10-19"
    assert data["n_intents"] == 10 and data["n_executed"] == 3
    assert data["guard_mode"] == "paper"
    assert data["dry_run"] is True
    assert data["age_sec"] is not None and data["age_sec"] < 60


def test_heartbeat_missing_returns_empty(tmp_path):
    """心跳文件不存在时返回空 dict，不抛异常（UI 首屏/新部署都会遇到）。"""
    assert read_heartbeat(tmp_path / "nope.json") == {}


def test_heartbeat_failure_does_not_break_tick(tmp_path, monkeypatch):
    """心跳写盘失败（比如路径是个目录）绝不能影响交易主流程。"""
    bad = tmp_path / "hb_dir"
    bad.mkdir()
    loop = _make_loop(tmp_path, bad)          # 指向目录 → write_text 必失败

    fake = TickReport(at="09:31:00", state="idle", market_status="非交易时段")
    monkeypatch.setattr(loop, "_tick_impl", lambda: fake)

    assert loop.tick() is fake                # 不抛异常


# ======================================================================
# 纸账户市值归一化
# ======================================================================
def test_market_value_normalized_on_load(tmp_path):
    """载入时把陈旧的 market_value 拉回 quantity × last_price。"""
    f = tmp_path / "paper_account.json"
    f.write_text(json.dumps({
        "cash": 1000.0,
        "holdings": {"600519": {
            "name": "贵州茅台", "quantity": 300, "available": 300,
            "today_quantity": 0, "avg_cost": 1501.725,
            "market_value": 450000.0,          # 陈旧：按成交价 1500 算的
            "last_price": 1501.95,
        }},
        "today_pnl": 0.0, "today_trades": 0, "last_reset_date": "2026-10-15",
    }), encoding="utf-8")

    pa = PaperAccount(account_file=f, history_file=tmp_path / "hist.json")
    h = pa.holdings["600519"]
    assert h["market_value"] == pytest.approx(300 * 1501.95)
    assert pa.get_total_asset() == pytest.approx(1000.0 + 300 * 1501.95)


def test_market_value_normalized_before_save(tmp_path):
    """外部把 last_price 改了、market_value 没跟 → 落盘前必须拉正。"""
    f = tmp_path / "paper_account.json"
    pa = PaperAccount(initial_cash=1_000_000.0,
                      account_file=f, history_file=tmp_path / "hist.json")
    pa.holdings["000001"] = {
        "name": "平安银行", "quantity": 1000, "available": 1000,
        "today_quantity": 0, "avg_cost": 10.0,
        "market_value": 10000.0, "last_price": 12.5,   # 故意不一致
    }
    pa._save()

    saved = json.loads(f.read_text(encoding="utf-8"))
    assert saved["holdings"]["000001"]["market_value"] == pytest.approx(12500.0)


def test_missing_last_price_falls_back_to_cost(tmp_path):
    """没有 last_price 时退化成成本价，不留 0 市值。"""
    f = tmp_path / "paper_account.json"
    f.write_text(json.dumps({
        "cash": 0.0,
        "holdings": {"600000": {
            "name": "浦发银行", "quantity": 500, "available": 500,
            "avg_cost": 8.0, "market_value": 0.0,
        }},
        "today_pnl": 0.0, "today_trades": 0, "last_reset_date": "2026-10-15",
    }), encoding="utf-8")

    pa = PaperAccount(account_file=f, history_file=tmp_path / "hist.json")
    h = pa.holdings["600000"]
    assert h["last_price"] == pytest.approx(8.0)
    assert h["market_value"] == pytest.approx(4000.0)

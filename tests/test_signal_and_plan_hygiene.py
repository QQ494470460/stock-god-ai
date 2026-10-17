# -*- coding: utf-8 -*-
"""观测/计划卫生测试：信号日志去重、计划参考价、牛熊文案层级。

三个都是「数据看着正常、其实失真」的静默问题：

1. ``guard.propose_trade`` 早期在函数第一行无条件写 ``signal`` 事件，而盘中
   每 tick 都会对每只标的调一次 —— 实测 10-15 一天刷 1440 条，整个 journal
   1659 条里 1642 条（99%）是同一条信号重复。噪声淹没 fill / risk_block。
2. 交易计划的 ``price`` 字段从来没被填过，10 只目标全是 0.0，字段名存实亡。
3. 计划顶层 ``regime`` 取自指数基准，而每条推荐的 ``reason`` 取自**个股**特征，
   文案又都写成「牛市顺风」→ 同一份计划里 ``regime: bear`` 配
   ``reason: 牛市顺风``，自相矛盾。
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


# ======================================================================
# 极简假件（不下单、不出网）
# ======================================================================
class _FakeBroker(BrokerAPI):
    def __init__(self, last_price: float = 10.0):
        super().__init__()
        self._connected = True
        self.last_price = last_price
        self.orders = []

    def connect(self, **kwargs): self._connected = True; return True
    def disconnect(self): self._connected = False

    def query_account(self):
        return AccountInfo("FAKE", 1_000_000.0, 600_000.0, 0.0, 400_000.0, 0.0, 0.0)

    def query_positions(self):
        return []

    def query_orders(self, *a, **k):
        return []

    def place_order(self, code, side, order_type, price, quantity, strategy_tag=""):
        o = Order(f"ORD{len(self.orders) + 1:03d}", code, side, order_type,
                  price, quantity, quantity, OrderStatus.FILLED,
                  strategy_tag=strategy_tag)
        self.orders.append(o)
        return o

    def cancel_order(self, oid): return True
    def cancel_all(self): return 0

    def get_tick(self, code):
        p = self.last_price
        return TickData(code, code, p, p, p, p, p, 100000, 1e6, p, p, 0.0)

    def get_broker_name(self): return "FakeBroker"
    def is_trading_time(self): return True


def _make_guard(tmp_path, last_price: float = 10.0):
    cfg = GuardConfig(
        state_file=str(tmp_path / "state.json"),
        journal_file=str(tmp_path / "journal.jsonl"),
    )
    cfg.min_order_interval_sec = 0.0
    return RiskSupervisor(_FakeBroker(last_price), cfg), tmp_path / "journal.jsonl"


def _events(jpath: Path, name: str = None):
    rows = []
    if not jpath.exists():
        return rows
    for line in jpath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if name is None or o.get("event") == name:
            rows.append(o)
    return rows


# ======================================================================
# 1. 信号日志去重
# ======================================================================
def test_repeated_same_signal_logged_once(tmp_path):
    """同一标的、同方向、同评分档反复调用 → 只落一条。"""
    guard, jpath = _make_guard(tmp_path)

    for _ in range(20):
        guard._log_signal("600177", "雅戈尔", OrderSide.BUY, 62.5, 1.0, 1.0)

    sigs = _events(jpath, "signal")
    assert len(sigs) == 1, f"20 次同参调用应只落 1 条，实际 {len(sigs)}"
    assert sigs[0]["data"]["code"] == "600177"
    assert sigs[0]["data"]["side"] == "buy"


def test_signal_logged_again_when_score_crosses_bucket(tmp_path):
    """评分跨 5 分档（62.5 → 67.0）说明信号实质变化 → 再落一条。"""
    guard, jpath = _make_guard(tmp_path)

    guard._log_signal("600177", "雅戈尔", OrderSide.BUY, 62.5, 1.0, 1.0)
    guard._log_signal("600177", "雅戈尔", OrderSide.BUY, 62.5, 1.0, 1.0)   # 重复
    guard._log_signal("600177", "雅戈尔", OrderSide.BUY, 67.0, 1.0, 1.0)   # 跨档

    assert len(_events(jpath, "signal")) == 2


def test_signal_logged_again_when_side_flips(tmp_path):
    """同标的方向反转（买 → 卖）必须分别记账。"""
    guard, jpath = _make_guard(tmp_path)

    guard._log_signal("600177", "雅戈尔", OrderSide.BUY, 62.5, 1.0, 1.0)
    guard._log_signal("600177", "雅戈尔", OrderSide.SELL, 62.5, 1.0, 1.0)

    sides = [e["data"]["side"] for e in _events(jpath, "signal")]
    assert sides == ["buy", "sell"]


def test_dedupe_cache_resets_on_new_day(tmp_path):
    """跨交易日缓存清空，第二天同一条信号仍要落盘（否则会永久静音）。"""
    guard, jpath = _make_guard(tmp_path)

    guard._log_signal("600177", "雅戈尔", OrderSide.BUY, 62.5, 1.0, 1.0)
    assert len(_events(jpath, "signal")) == 1

    # 模拟跨日：把去重日戳改到过去
    guard._signal_dedupe_day = "2000-01-01"
    guard._log_signal("600177", "雅戈尔", OrderSide.BUY, 62.5, 1.0, 1.0)

    assert len(_events(jpath, "signal")) == 2


def test_propose_trade_does_not_flood_journal(tmp_path):
    """端到端：连续 propose 同一信号，signal 事件不应随调用次数线性增长。"""
    guard, jpath = _make_guard(tmp_path)

    for _ in range(10):
        guard.propose_trade("600177", "雅戈尔", OrderSide.BUY, ai_score=62.5, agreement=1.0)

    sigs = _events(jpath, "signal")
    assert len(sigs) == 1, f"10 次 propose 应只留 1 条 signal，实际 {len(sigs)}"


# ======================================================================
# 2. 交易计划参考价
# ======================================================================
def test_trade_plan_carries_reference_price():
    """推荐里带最新价 → 计划 target.price 要接住，不能恒为 0。"""
    from src.trading.trade_plan import build_trade_plan

    picks = [
        {"代码": "600177", "名称": "雅戈尔", "评分": 62.5,
         "推荐理由": "个股牛市顺风", "最新价": 8.12},
        {"代码": "600901", "名称": "江苏金租", "评分": 61.1,
         "推荐理由": "个股震荡"},                       # 无价格
    ]
    plan = build_trade_plan(
        picks=picks, pool="zz500", min_score=55.0, regime="bear",
        max_positions=10, cash_reserve_pct=0.1, weighting="equal",
        weight_cap=0.2, write=False,
    )

    by_code = {t.code: t for t in plan.targets}
    assert by_code["600177"].price == pytest.approx(8.12)
    assert by_code["600901"].price == 0.0     # 缺价时仍为 0（不编造）


def test_trade_plan_price_from_english_keys():
    """英文键 price / close 同样要接住（兼容 DataFrame 来源）。"""
    from src.trading.trade_plan import build_trade_plan

    plan = build_trade_plan(
        picks=[{"code": "600519", "name": "贵州茅台", "score": 70.0, "close": 1501.95}],
        pool="zz500", min_score=55.0, regime="bear",
        max_positions=5, cash_reserve_pct=0.1, weighting="equal",
        weight_cap=0.2, write=False,
    )
    assert plan.targets[0].price == pytest.approx(1501.95)


# ======================================================================
# 3. 牛熊文案层级（个股 vs 指数）
# ======================================================================
def _score_with(snapshot, prob_up=0.6, prob_down=0.2):
    from src.models.scoring import compute_score

    score, breakdown = compute_score(
        prob_up=prob_up, prob_neutral=1 - prob_up - prob_down, prob_down=prob_down,
        confidence=0.5, agreement=1.0, risk_pct=0.02, volatility=0.25,
        feature_snapshot=snapshot, n_classes=3,
    )
    return score, breakdown


def test_regime_note_is_stock_level_not_index_level():
    """文案必须点明是「个股」口径，避免与计划顶层指数 regime 打架。"""
    _, bd = _score_with({"regime_bull": 1})
    assert bd["regime_note"] == "个股牛市顺风"

    _, bd = _score_with({"regime_bear": 1}, prob_up=0.2, prob_down=0.6)
    assert bd["regime_note"] == "个股熊市顺风"

    _, bd = _score_with({})
    assert bd["regime_note"] == "个股震荡"

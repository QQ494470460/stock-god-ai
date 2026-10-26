# -*- coding: utf-8 -*-
"""ATR 自适应止损：把固定百分比换成「按个股波动归一」的宽度。

核心缺陷（2026-10 复盘发现）：固定 -5% 止损在 A 股是结构性错误。
- 年化波动 20% 的白马，-5% 只在趋势真坏时触发，合理；
- 年化波动 70% 的题材股，日内振幅本身就有 ±5%，-5% 止损等于开盘被扫，
  亏损单的共同特征就是「被日常噪声打掉、随后原方向走回去」。

不变量（本文件重点锁住的）：
1. **宽度只增不减**：ATR 开启后止损永远不会比配置值更紧，升级不会让止损更频繁。
2. **无数据完全等价旧行为**：ATR 取不到时逐字节退回固定线，绝不能变成 0。
3. 上下限夹紧：极低波动不紧于 2%，极高波动不宽于 15%。
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
from src.trading.executor import RiskConfig, TradeExecutor
from src.trading.stops import AtrProvider, adaptive_stop_pct, adaptive_trailing_pct


# ----------------------------------------------------------------------
# 纯函数
# ----------------------------------------------------------------------
def test_no_atr_data_returns_base_untouched():
    """无数据 → 原样返回配置值（不猜、不放大、不缩成 0）。"""
    assert adaptive_stop_pct(-5.0, None) == -5.0
    assert adaptive_stop_pct(-8.0, 0.0) == -8.0
    assert adaptive_stop_pct(-8.0, float("nan")) == -8.0


def test_stop_widens_with_volatility():
    # ATR 3.5% × k2 = 7% > 5% → 放宽到 7%
    assert adaptive_stop_pct(-5.0, 0.035) == pytest.approx(-7.0)
    # ATR 6% × k2 = 12%
    assert adaptive_stop_pct(-5.0, 0.06) == pytest.approx(-12.0)


def test_stop_never_tighter_than_configured():
    """低波动标的：ATR 给 1.2% 也不许把止损收紧到 -5% 以内。"""
    assert adaptive_stop_pct(-5.0, 0.006) == -5.0
    assert adaptive_stop_pct(-5.0, 0.012) == -5.0


def test_stop_clamped_by_floor_and_ceil():
    # ATR 0.1% × k2 = 0.2%，但配置是 -20% → 下限 2% 不该把用户设的 -20% 变紧？
    # 不变量优先：宽度 = max(|base|, k*atr) 后夹在 [floor, ceil]
    assert adaptive_stop_pct(-20.0, 0.001, floor_pct=2.0, ceil_pct=15.0) == -15.0
    assert adaptive_stop_pct(-1.0, 0.001, floor_pct=2.0, ceil_pct=15.0) == -2.0
    # ATR 20% × 2 = 40% → 上限 15%
    assert adaptive_stop_pct(-5.0, 0.20) == -15.0


def test_trailing_uses_smaller_multiplier():
    """同一 ATR 下移动止损比硬止损更紧（k 更小）。"""
    hard = adaptive_stop_pct(-5.0, 0.035, k=2.0)
    trail = adaptive_trailing_pct(-3.0, 0.035, k=1.5)
    assert abs(trail) < abs(hard)
    assert trail == pytest.approx(-5.25)


def test_bad_input_never_raises():
    assert adaptive_stop_pct(None, 0.03) == -2.0
    assert adaptive_stop_pct(-5.0, "abc") == -5.0


# ----------------------------------------------------------------------
# AtrProvider 读盘
# ----------------------------------------------------------------------
def _write_ohlc(path: Path, n=40, base=10.0, rng=0.02, close=10.0):
    rows = []
    for i in range(n):
        c = base + (i % 5) * 0.01
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "high": c * (1 + rng),
            "low": c * (1 - rng),
            "close": c if i < n - 1 else close,
        })
    pd.DataFrame(rows).to_parquet(path)


def test_atr_provider_reads_parquet(tmp_path):
    _write_ohlc(tmp_path / "daily_600000_qfq.parquet", rng=0.03)
    p = AtrProvider(data_dir=tmp_path, periods=14)
    v = p.atr_pct("600000")
    assert v is not None and 0.0 < v < 0.2


def test_atr_provider_missing_file_returns_none(tmp_path):
    p = AtrProvider(data_dir=tmp_path)
    assert p.atr_pct("000001") is None


def test_atr_provider_missing_column_returns_none(tmp_path):
    """只有 close 没有 high/low → 返回 None，绝不拿 close 近似。"""
    pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=30),
        "close": [10.0] * 30,
    }).to_parquet(tmp_path / "daily_600001_qfq.parquet")
    p = AtrProvider(data_dir=tmp_path)
    assert p.atr_pct("600001") is None


def test_atr_provider_cache(tmp_path):
    _write_ohlc(tmp_path / "daily_600002_qfq.parquet")
    p = AtrProvider(data_dir=tmp_path)
    first = p.atr_pct("600002")
    # 删掉文件后仍应返回缓存值，证明只读一次盘
    (tmp_path / "daily_600002_qfq.parquet").unlink()
    assert p.atr_pct("600002") == first


def test_atr_cache_rolls_to_next_day(tmp_path):
    """常驻进程不重启，缓存必须跨自然日失效，否则拿昨天 ATR 做今天止损。"""
    _write_ohlc(tmp_path / "daily_600003_qfq.parquet")
    p = AtrProvider(data_dir=tmp_path)
    assert p.atr_pct("600003") is not None
    p._cache_day = "1999-01-01"                    # 伪造「昨天」
    (tmp_path / "daily_600003_qfq.parquet").unlink()
    assert p.atr_pct("600003") is None             # 缓存已清 + 文件已删


# ----------------------------------------------------------------------
# 与 TradeExecutor 的集成
# ----------------------------------------------------------------------
def _pos(code="600000", avg_cost=10.0, qty=1000, price=10.0) -> Position:
    return Position(
        code=code, name=code, quantity=qty, available=qty,
        avg_cost=avg_cost, current_price=price,
        market_value=qty * price, pnl=(price - avg_cost) * qty,
        pnl_pct=(price / avg_cost - 1) * 100 if avg_cost else 0.0,
        today_quantity=0,
    )


class _FakeBroker(BrokerAPI):
    is_simulated = False

    def __init__(self, price=10.0, positions=None):
        super().__init__()
        self._connected = True
        self.price = price
        self.positions = list(positions or [])

    def connect(self): return True
    def disconnect(self): pass
    def is_connected(self): return True
    def get_broker_name(self): return "fake"
    def place_order(self, **kwargs): return None
    def cancel_order(self, order_id): return True
    def cancel_all(self): return True
    def query_orders(self): return []
    def query_positions(self): return list(self.positions)
    def query_account(self):
        return AccountInfo(account_id="fake", total_asset=1_000_000,
                           available_cash=1_000_000, frozen_cash=0.0,
                           market_value=0.0, total_pnl=0.0, today_pnl=0.0)

    def get_tick(self, code):
        return TickData(code=code, last_price=self.price, bid1=self.price,
                        ask1=self.price, volume=0, open=0, high=0, low=0,
                        pre_close=self.price, timestamp="")


class _FakeAtr:
    """固定 ATR 的替身，避免测试读盘。"""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def atr_pct(self, code):
        self.calls += 1
        return self.value


def test_executor_atr_disabled_keeps_configured_lines():
    ex = TradeExecutor(_FakeBroker(), RiskConfig(atr_stop_enabled=False))
    assert ex._effective_stop_lines("600000") == (-5.0, -3.0)


def test_executor_atr_enabled_widens_stop():
    cfg = RiskConfig(atr_stop_enabled=True, atr_stop_k=2.0, atr_trailing_k=1.5)
    ex = TradeExecutor(_FakeBroker(), cfg, atr_provider=_FakeAtr(0.04))
    stop, trail = ex._effective_stop_lines("600000")
    assert stop == pytest.approx(-8.0)     # 4% × 2
    assert trail == pytest.approx(-6.0)    # 4% × 1.5


def test_executor_atr_none_falls_back_to_fixed():
    cfg = RiskConfig(atr_stop_enabled=True)
    ex = TradeExecutor(_FakeBroker(), cfg, atr_provider=_FakeAtr(None))
    assert ex._effective_stop_lines("600000") == (-5.0, -3.0)


def test_executor_atr_provider_error_falls_back():
    class _Boom:
        def atr_pct(self, code):
            raise RuntimeError("boom")

    cfg = RiskConfig(atr_stop_enabled=True)
    ex = TradeExecutor(_FakeBroker(), cfg, atr_provider=_Boom())
    assert ex._effective_stop_lines("600000") == (-5.0, -3.0)


def test_check_stop_conditions_respects_wider_atr_line():
    """-6% 浮亏：固定 -5% 会止损，ATR 放宽到 -8% 则不该止损。"""
    pos = _pos(avg_cost=10.0, price=9.4)  # -6%
    broker = _FakeBroker(price=9.4, positions=[pos])

    fixed = TradeExecutor(broker, RiskConfig(atr_stop_enabled=False))
    assert len(fixed.check_stop_conditions(positions=[pos], prices={"600000": 9.4})) == 1

    wide = TradeExecutor(broker, RiskConfig(atr_stop_enabled=True),
                         atr_provider=_FakeAtr(0.04))
    assert wide.check_stop_conditions(positions=[pos], prices={"600000": 9.4}) == []


def test_check_stop_conditions_still_fires_beyond_atr_line():
    pos = _pos(avg_cost=10.0, price=9.1)  # -9% < -8%
    broker = _FakeBroker(price=9.1, positions=[pos])
    ex = TradeExecutor(broker, RiskConfig(atr_stop_enabled=True),
                       atr_provider=_FakeAtr(0.04))
    alerts = ex.check_stop_conditions(positions=[pos], prices={"600000": 9.1})
    assert len(alerts) == 1 and alerts[0]["action"] == "sell"

# -*- coding: utf-8 -*-
"""Analytics 底座（波动率估计 / 绩效归因 / walk-forward 切分）单测。

覆盖范围：
- VolatilityEstimator：价格序列估计、parquet 历史估计、缺数据时优雅返回 None
- attribute_performance：按策略标签拆解盈亏贡献
- walk_forward_splits：滑动窗口区间不重叠且紧邻
- PositionSizer 打通 vol_provider：波动率目标法真正生效 + 未知 vol 优雅降级
"""

import numpy as np
import pandas as pd
import pytest

from src.trading.analytics import (
    annualized_volatility,
    atr_percent,
    attribute_performance,
    walk_forward_splits,
    VolatilityEstimator,
)
from src.trading.broker import OrderSide
from src.trading.guard import GuardConfig, PositionSizer


# ----------------------------------------------------------------------
# 波动率估计
# ----------------------------------------------------------------------
def test_annualized_volatility_basic():
    # 交替 ±1% 日收益 → 标准差≈1% → 年化 ≈ 0.01 * sqrt(242) ≈ 0.156
    rets = [0.01 if i % 2 == 0 else -0.01 for i in range(100)]
    vol = annualized_volatility(rets)
    assert 0.14 < vol < 0.17


def test_annualized_volatility_constant_returns_is_zero():
    assert annualized_volatility([0.0, 0.0, 0.0]) == 0.0


def test_annualized_volatility_too_short_is_zero():
    assert annualized_volatility([0.01]) == 0.0


def test_annualized_volatility_realistic_range():
    rng = np.random.default_rng(42)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.02, 250))
    vol = annualized_volatility(np.diff(prices) / prices[:-1])
    # A股个股年化波动常见 0.2~0.8
    assert 0.1 < vol < 1.5


def test_atr_percent_basic():
    high = [10, 11, 12, 11, 10]
    low = [9, 10, 11, 10, 9]
    close = [9.5, 10.5, 11.5, 10.5, 9.5]
    atr = atr_percent(high, low, close, periods=3)
    assert 0.0 < atr < 1.0  # 占收盘价比例


def test_volatility_estimator_from_prices():
    prices = list(np.linspace(10, 20, 80))  # 单调上涨 → 有波动
    vol = VolatilityEstimator.from_prices(prices)
    assert vol > 0


def test_volatility_estimator_from_parquet(tmp_path):
    # 造一份 daily_{code}_qfq.parquet（含 close 列）
    code = "600519"
    closes = list(np.linspace(1000, 1500, 90)) + list(np.linspace(1500, 1200, 30))
    df = pd.DataFrame({"date": range(len(closes)), "close": closes})
    df.to_parquet(tmp_path / f"daily_{code}_qfq.parquet", index=False)

    est = VolatilityEstimator(data_dir=tmp_path, lookback=60)
    vol = est.annualized_vol(code)
    assert vol is not None and vol > 0


def test_volatility_estimator_missing_file_returns_none(tmp_path):
    est = VolatilityEstimator(data_dir=tmp_path)
    assert est.annualized_vol("999999") is None


def test_volatility_estimator_caches(tmp_path):
    code = "600519"
    df = pd.DataFrame({"close": list(np.linspace(1000, 1500, 90))})
    df.to_parquet(tmp_path / f"daily_{code}_qfq.parquet", index=False)
    est = VolatilityEstimator(data_dir=tmp_path)
    v1 = est.annualized_vol(code)
    v2 = est.annualized_vol(code)  # 第二次应命中缓存（同值）
    assert v1 == v2


# ----------------------------------------------------------------------
# 绩效归因
# ----------------------------------------------------------------------
class _Trade:
    def __init__(self, side, pnl=0.0, strategy=""):
        self.side = side
        self.pnl = pnl
        self.strategy = strategy


def test_attribute_performance_sums_and_by_strategy():
    trades = [
        _Trade("buy"),                       # 买入（无 pnl，不算）
        _Trade("sell", pnl=100.0, strategy="momentum"),
        _Trade("sell", pnl=-50.0, strategy="mean_revert"),
        _Trade("sell", pnl=-30.0, strategy="momentum"),  # 计为负贡献
    ]
    r = attribute_performance(trades)
    assert r["n_trades"] == 4
    assert r["n_sells"] == 3
    assert r["total_pnl"] == 20.0           # 100 - 50 - 30
    assert r["win_rate"] == pytest.approx(1 / 3, abs=1e-3)  # 函数内 round(.,4)→0.3333
    assert r["profit_factor"] == pytest.approx(100.0 / 80.0)  # 100 / (50+30)
    assert r["by_strategy"]["momentum"]["pnl"] == pytest.approx(70.0)   # 100 - 30
    assert r["by_strategy"]["momentum"]["count"] == 2
    assert r["by_strategy"]["momentum"]["win_rate"] == 0.5
    assert r["by_strategy"]["mean_revert"]["pnl"] == pytest.approx(-50.0)
    assert r["by_strategy"]["mean_revert"]["win_rate"] == 0.0


def test_attribute_performance_empty():
    r = attribute_performance([])
    assert r["total_pnl"] == 0.0
    assert r["win_rate"] == 0.0
    assert r["profit_factor"] == 999.0  # inf 的兜底表示


# ----------------------------------------------------------------------
# walk-forward 切分
# ----------------------------------------------------------------------
def test_walk_forward_splits_contiguous_non_overlapping():
    folds = walk_forward_splits(n_folds=3, train_size=10, test_size=5)
    assert len(folds) == 3
    for (t0, t1), (e0, e1) in folds:
        assert e0 == t1 + 1          # 训练与测试紧邻
        assert e0 > t1               # 不重叠
        assert t1 >= t0 and e1 >= e0


# ----------------------------------------------------------------------
# PositionSizer 打通 vol_provider
# ----------------------------------------------------------------------
def test_sizer_volatility_target_uses_provider():
    cfg = GuardConfig(
        sizing_method="volatility_target",
        base_risk_fraction=0.02,
        volatility_target_pct=0.15,
    )
    # vol=0.8 → frac = min(0.15/0.8=0.1875, base*2=0.04) = 0.04（波动率目标法封顶 2 倍基准）
    sizer = PositionSizer(cfg, vol_provider=lambda c: 0.8)
    qty, budget = sizer.size(OrderSide.BUY, price=10.0, total_equity=1_000_000.0, code="X")
    # 强度=1 → budget = 1e6 * 0.04
    assert budget == pytest.approx(1_000_000.0 * 0.04, rel=1e-6)


def test_sizer_volatility_target_unknown_vol_falls_back_to_fixed():
    cfg = GuardConfig(
        sizing_method="volatility_target",
        base_risk_fraction=0.02,
        volatility_target_pct=0.15,
    )
    # provider 返回 None（无历史）→ 优雅降级为固定比例法
    sizer = PositionSizer(cfg, vol_provider=lambda c: None)
    _, budget = sizer.size(OrderSide.BUY, price=10.0, total_equity=1_000_000.0, code="X")
    assert budget == pytest.approx(1_000_000.0 * 0.02, rel=1e-6)


def test_sizer_fixed_fractional_ignores_provider():
    cfg = GuardConfig(sizing_method="fixed_fractional", base_risk_fraction=0.02)
    # 固定比例法下即便给了 provider 也不该触发 vol IO/估算
    sizer = PositionSizer(cfg, vol_provider=lambda c: 0.8)
    _, budget = sizer.size(OrderSide.BUY, price=10.0, total_equity=1_000_000.0, code="X")
    assert budget == pytest.approx(1_000_000.0 * 0.02, rel=1e-6)

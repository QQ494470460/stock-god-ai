# -*- coding: utf-8 -*-
"""风险增强模块单测：波动率估计 / 非顺周期降风险 / 组合波动率目标。

这些模块目前**未接线**（生产代码不 import 它们），只供
``src/core/exec_backtest.py`` 做参数验证。测试纯用合成数据，无网络、无磁盘写。

核心断言是三条设计红线：
1. 降风险的**地板 >= 0.5**（不允许清仓式空仓）；
2. 降风险**只在有意义回撤（>= 12%）介入**，且恢复带滞回（不在噪音里抖动）；
3. 波动率数据缺失时 **fail-open**（返回 None / 1.0），而不是把仓位压到 0。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.trading.risk import (  # noqa: E402
    DeriskRamp,
    asymmetric_derisk_scale,
    blend_scales,
    ewma_vol_annualized,
    make_vol_provider,
    portfolio_vol_target_scale,
    realized_vol_from_prices,
)


# ---------------------------------------------------------------------------
# volatility: ewma
# ---------------------------------------------------------------------------

def test_ewma_constant_returns_is_zero_vol():
    """恒定收益率 = 零波动。若用 sum(w*r^2) 会错算成 |r|*sqrt(252)。"""
    rets = [0.001] * 120
    assert abs(ewma_vol_annualized(rets, span=20)) < 1e-12


def test_ewma_known_series_positive_finite():
    rets = [0.01, -0.02, 0.03, -0.01, 0.02, -0.005, 0.015]
    vol = ewma_vol_annualized(rets, span=5)
    assert math.isfinite(vol)
    assert vol > 0.0
    # 日波动量级 ~1.7%，年化不可能超过 200%
    assert vol < 2.0


def test_ewma_handles_empty_and_short_series():
    assert ewma_vol_annualized([]) == 0.0
    assert ewma_vol_annualized([0.01]) == 0.0
    assert ewma_vol_annualized(None) == 0.0  # type: ignore[arg-type]


def test_ewma_skips_dirty_values():
    clean = [0.01, -0.02, 0.03, -0.01, 0.02]
    dirty = [0.01, None, -0.02, float("nan"), 0.03, float("inf"), -0.01, 0.02]
    assert ewma_vol_annualized(dirty, span=5) == pytest.approx(
        ewma_vol_annualized(clean, span=5), rel=1e-12
    )


# ---------------------------------------------------------------------------
# volatility: realized from prices
# ---------------------------------------------------------------------------

def test_realized_vol_linear_uptrend_is_zero():
    closes = [100.0 * (1.01 ** i) for i in range(60)]
    assert realized_vol_from_prices(closes, window=20) < 1e-9


def test_realized_vol_noisy_is_positive():
    closes = []
    px = 100.0
    for i in range(60):
        px *= 1.02 if i % 2 == 0 else 0.975
        closes.append(px)
    vol = realized_vol_from_prices(closes, window=20)
    assert math.isfinite(vol)
    assert vol > 0.05


def test_realized_vol_needs_three_points():
    assert realized_vol_from_prices([]) == 0.0
    assert realized_vol_from_prices([100.0]) == 0.0
    assert realized_vol_from_prices([100.0, 101.0]) == 0.0


def test_realized_vol_skips_bad_prices():
    closes = [100.0, 101.0, 0.0, 102.0, -5.0, 103.0, 104.0, 105.0]
    vol = realized_vol_from_prices(closes, window=20)
    assert math.isfinite(vol)
    assert vol >= 0.0


# ---------------------------------------------------------------------------
# volatility: provider (fail-open)
# ---------------------------------------------------------------------------

def test_vol_provider_returns_positive_for_good_history():
    prices = [100.0 * (1.0 + 0.01 * ((-1) ** i)) for i in range(40)]
    provider = make_vol_provider(lambda code: prices, window=20)
    vol = provider("600000")
    assert vol is not None and vol > 0.0


def test_vol_provider_fails_open():
    def boom(code):
        raise RuntimeError("数据源挂了")

    assert make_vol_provider(boom)("600000") is None
    assert make_vol_provider(lambda c: None)("600000") is None
    assert make_vol_provider(lambda c: [100.0, 101.0])("600000") is None
    # 完全持平的行情：波动率为 0 -> 必须返回 None（否则 sizing 会给 0 仓位）
    assert make_vol_provider(lambda c: [100.0] * 40)("600000") is None


# ---------------------------------------------------------------------------
# derisk: pure ramp
# ---------------------------------------------------------------------------

def test_derisk_no_action_below_engage_line():
    assert asymmetric_derisk_scale(0.0) == 1.0
    assert asymmetric_derisk_scale(5.0) == 1.0
    assert asymmetric_derisk_scale(11.9) == 1.0


def test_derisk_hits_floor_at_max_dd():
    assert asymmetric_derisk_scale(20.0) == pytest.approx(0.60)
    assert asymmetric_derisk_scale(35.0) == pytest.approx(0.60)
    # 中间点是 1.0 与 floor 的中点
    assert asymmetric_derisk_scale(16.0) == pytest.approx(0.80)


def test_derisk_monotone_non_increasing():
    prev = 1.0
    for dd in [x * 0.5 for x in range(0, 81)]:
        cur = asymmetric_derisk_scale(dd)
        assert cur <= prev + 1e-12
        assert cur <= 1.0
        prev = cur


def test_derisk_floor_never_below_half():
    for floor in (0.0, 0.1, 0.3, 0.49, -1.0, float("nan")):
        assert asymmetric_derisk_scale(50.0, floor=floor) >= 0.5
    assert asymmetric_derisk_scale(50.0, engage_pct=12.0, max_dd_pct=12.0) >= 0.5


def test_derisk_ignores_invalid_drawdown():
    assert asymmetric_derisk_scale(None) == 1.0  # type: ignore[arg-type]
    assert asymmetric_derisk_scale(float("nan")) == 1.0


# ---------------------------------------------------------------------------
# derisk: stateful ramp with recovery hysteresis
# ---------------------------------------------------------------------------

def test_derisk_ramp_hysteresis_path():
    ramp = DeriskRamp(engage_pct=12.0, max_dd_pct=20.0, floor=0.60,
                      release_hysteresis_pct=2.0)
    assert ramp.current() == 1.0
    assert not ramp.engaged

    s13 = ramp.update(13.0)
    assert s13 < 1.0
    assert ramp.engaged

    # 回到 11%：仍在滞回带内（release=10%），棘轮保持 13% 的水平
    s11 = ramp.update(11.0)
    assert s11 < 1.0
    assert s11 == pytest.approx(s13)
    assert ramp.engaged

    # 修复到 9%（<= 12-2）：解除介入，完全放开
    s9 = ramp.update(9.0)
    assert s9 == 1.0
    assert not ramp.engaged
    assert ramp.current() == 1.0


def test_derisk_ramp_ratchets_deeper_then_holds():
    ramp = DeriskRamp(engage_pct=12.0, max_dd_pct=20.0, floor=0.60)
    s14 = ramp.update(14.0)
    s18 = ramp.update(18.0)
    assert s18 < s14  # 加深回撤继续降
    s12 = ramp.update(12.2)  # 反弹但没修复到 10% 以下
    assert s12 == pytest.approx(s18)  # 不因小反弹而松开
    assert ramp.peak_drawdown_pct == pytest.approx(18.0)


def test_derisk_ramp_reset():
    ramp = DeriskRamp()
    ramp.update(16.0)
    assert ramp.current() < 1.0
    ramp.reset()
    assert ramp.current() == 1.0
    assert not ramp.engaged
    assert ramp.peak_drawdown_pct == 0.0


def test_derisk_ramp_never_below_floor():
    ramp = DeriskRamp(engage_pct=12.0, max_dd_pct=20.0, floor=0.60)
    for dd in (13, 16, 20, 28, 45, 60):
        s = ramp.update(dd)
        assert 0.5 <= s <= 1.0


def test_derisk_ramp_non_procyclical_example_path():
    """报告用样例：一条「跌深→反弹」路径上系数只在 0.60~1.0，从不归零。"""
    ramp = DeriskRamp()
    path = [0.0, 6.0, 12.5, 15.0, 19.0, 22.0, 16.0, 11.0, 9.5, 4.0]
    scales = [ramp.update(dd) for dd in path]
    assert all(0.60 <= s <= 1.0 for s in scales)
    assert scales[0] == 1.0
    assert scales[4] == pytest.approx(0.65)  # 19% 斜坡
    assert scales[5] == pytest.approx(0.60)  # 22% 落地板
    assert scales[6] == pytest.approx(0.60)  # 16% 棘轮不放宽
    assert scales[7] < 1.0  # 11% 滞回未放开
    assert scales[8] == pytest.approx(1.0)  # 9.5% <= 10% 完全放开
    assert scales[9] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# portfolio vol target
# ---------------------------------------------------------------------------

def test_portfolio_vol_scale_high_vol_hits_floor():
    assert portfolio_vol_target_scale(0.30, target_vol=0.15) == pytest.approx(0.50)


def test_portfolio_vol_scale_low_vol_never_levers_up():
    assert portfolio_vol_target_scale(0.10, target_vol=0.15) == pytest.approx(1.0)
    assert portfolio_vol_target_scale(0.15, target_vol=0.15) == pytest.approx(1.0)


def test_portfolio_vol_scale_fails_open():
    assert portfolio_vol_target_scale(0.0) == 1.0
    assert portfolio_vol_target_scale(-0.2) == 1.0
    assert portfolio_vol_target_scale(None) == 1.0  # type: ignore[arg-type]
    assert portfolio_vol_target_scale(0.30, target_vol=0.0) == 1.0


def test_portfolio_vol_scale_floor_clamped():
    assert portfolio_vol_target_scale(1.0, target_vol=0.15, floor=0.2) >= 0.5


# ---------------------------------------------------------------------------
# blending
# ---------------------------------------------------------------------------

def test_blend_scales_arithmetic_mean():
    assert blend_scales(0.6, 0.5) == pytest.approx(0.55)
    assert blend_scales(1.0, 1.0) == pytest.approx(1.0)
    assert blend_scales(0.60, 1.0, weight_b=0.0) == pytest.approx(0.60)
    # 算术混合保证两个 >=0.5 的地板不会跌破 0.5（几何会掉到 0.30）
    assert blend_scales(0.6, 0.5) >= 0.5


def test_blend_scales_clamps_and_fails_open():
    assert blend_scales(None, 0.5) == pytest.approx(0.75)  # type: ignore[arg-type]
    assert 0.0 <= blend_scales(1.5, -0.5) <= 1.0

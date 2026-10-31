# -*- coding: utf-8 -*-
"""组合风险模型：相关性 → 组合波动率 → 参数法 VaR。

核心断言：**名义分散 ≠ 风险分散**。两只完全同涨同跌的票各占 50%，
组合波动率等于单只的波动率（不是 1/√2）。这正是「单票 X%、最多 N 只」
管不到的尾部风险。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import pytest

from src.trading.risk_model import (
    daily_returns, returns_matrix, covariance, portfolio_daily_vol,
    parametric_var_pct, avg_pairwise_corr, PortfolioRiskMonitor,
)


def test_daily_returns_basic():
    r = daily_returns([100.0, 110.0, 99.0])
    assert r == pytest.approx(np.array([0.10, -0.10]))


def test_returns_matrix_aligns_tail_and_drops_short():
    codes, rets = returns_matrix({
        "A": [10, 11, 12, 13],
        "B": [20, 22, 24],          # 更短 → 对齐到 3 个点
        "C": [0, 0, 0],             # 无有效价 → 剔除
    })
    assert codes == ["A", "B"]
    assert rets.shape == (2, 2)


def test_correlated_assets_have_no_diversification():
    """两只完全同涨同跌的票，各 50% 的组合波动率 == 单只波动率。"""
    closes = [100.0, 101.0, 102.0, 101.0, 103.0, 104.0]
    _, rets = returns_matrix({"A": closes, "B": closes})
    cov = covariance(rets)
    vol = portfolio_daily_vol([0.5, 0.5], cov)

    single = float(np.std(daily_returns(closes), ddof=1))
    assert vol == pytest.approx(single, rel=1e-9)


def test_opposite_assets_cancel_out():
    """完全负相关等权 → 组合波动率≈0（风险真的被对冲掉了）。"""
    # 直接构造互为相反数的收益率矩阵，避免价格序列的复利误差
    rets = np.array([
        [0.010, -0.010],
        [-0.020, 0.020],
        [0.030, -0.030],
        [0.015, -0.015],
    ])
    assert portfolio_daily_vol([0.5, 0.5], covariance(rets)) == pytest.approx(0.0, abs=1e-15)


def test_avg_pairwise_corr_extremes():
    same = np.array([[0.01, 0.01], [0.02, 0.02], [-0.01, -0.01], [0.03, 0.03]])
    assert avg_pairwise_corr(same) == pytest.approx(1.0, abs=1e-9)

    opp = np.array([[0.01, -0.01], [0.02, -0.02], [-0.01, 0.01], [0.03, -0.03]])
    assert avg_pairwise_corr(opp) == pytest.approx(-1.0, abs=1e-9)


def test_parametric_var_pct_scales_with_z():
    assert parametric_var_pct(0.02, 1.65) == pytest.approx(0.033)
    assert parametric_var_pct(0.02, 0.0) == 0.0


def _write(tmp_path, code, closes):
    pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
        "close": closes, "volume": [1e6] * len(closes),
        "amount": [1e7] * len(closes),
    }).to_parquet(tmp_path / f"daily_{code}_qfq.parquet")


def test_monitor_drops_missing_code_without_renormalizing(tmp_path):
    # A/B 有数据且同涨；C 没有任何历史 → 被剔除，且剩余权重**保持原比例**
    _write(tmp_path, "600000", [10 + i * 0.1 for i in range(60)])
    _write(tmp_path, "600001", [10 + i * 0.1 for i in range(60)])
    m = PortfolioRiskMonitor(data_dir=tmp_path)

    res = m.var_pct(["600000", "600001", "999999"], [0.3, 0.3, 0.4])
    assert res is not None
    var_pct, corr = res
    assert var_pct > 0
    assert corr == pytest.approx(1.0, abs=1e-6)

    # 可用权重翻倍 → VaR 应恰好翻倍：证明没有「归一化到持仓合计」
    res2 = m.var_pct(["600000", "600001"], [0.6, 0.6])
    assert res2 is not None
    assert res2[0] == pytest.approx(var_pct * 2, rel=1e-6)


def test_monitor_returns_none_with_insufficient_history(tmp_path):
    _write(tmp_path, "600000", [10 + i * 0.1 for i in range(60)])
    m = PortfolioRiskMonitor(data_dir=tmp_path)
    assert m.var_pct(["600000", "888888"], [0.5, 0.5]) is None


# ----------------------------------------------------------------------
# Guard 集成：VaR 超预算 → 新开仓强度连续缩放
# ----------------------------------------------------------------------
class _Monitor:
    def __init__(self, var_pct, corr=0.8):
        self._v = var_pct
        self._c = corr

    def var_pct(self, codes, weights, z=1.65):
        return self._v, self._c


class _Pos:
    def __init__(self, code, value):
        self.code = code
        self.quantity = 1000
        self.market_value = value
        self.current_price = value / 1000
        self.available = 1000
        self.avg_cost = self.current_price
        self.name = code
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.today_quantity = 0


def _guard(tmp_path, monitor):
    from src.trading.guard import GuardConfig, RiskSupervisor
    cfg = GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
        var_monitor_enabled=True,   # 风险模型单测隔离：显式开，避免默认值改动牵连
    )
    g = RiskSupervisor.__new__(RiskSupervisor)  # 只测纯逻辑，免构造券商/安全层
    g.config = cfg
    g._risk_monitor = monitor
    g._lock = __import__("threading").RLock()
    g._turnover_day = ""
    g._turnover_amount = 0.0
    return g


def test_var_scale_shrinks_when_over_budget(tmp_path):
    g = _guard(tmp_path, _Monitor(var_pct=0.05))  # 5% > 2.5% 上限
    positions = {"A": _Pos("A", 300_000), "B": _Pos("B", 300_000)}
    account = type("Acct", (), {"total_asset": 1_000_000.0})()

    scale, detail = g._portfolio_var_scale(account, positions)

    assert scale == pytest.approx(0.5)          # 0.025 / 0.05
    assert detail["var_pct"] == pytest.approx(0.05)
    assert detail["avg_corr"] == pytest.approx(0.8)


def test_var_scale_floor_prevents_total_shutdown(tmp_path):
    g = _guard(tmp_path, _Monitor(var_pct=0.20))
    positions = {"A": _Pos("A", 300_000), "B": _Pos("B", 300_000)}
    account = type("Acct", (), {"total_asset": 1_000_000.0})()

    scale, _ = g._portfolio_var_scale(account, positions)
    assert scale == pytest.approx(0.40), "再糟也不清零，留 40% 敞口"


def test_var_scale_is_one_when_within_budget(tmp_path):
    g = _guard(tmp_path, _Monitor(var_pct=0.01))
    positions = {"A": _Pos("A", 300_000), "B": _Pos("B", 300_000)}
    account = type("Acct", (), {"total_asset": 1_000_000.0})()
    assert g._portfolio_var_scale(account, positions)[0] == pytest.approx(1.0)


def test_var_scale_noop_for_single_position(tmp_path):
    g = _guard(tmp_path, _Monitor(var_pct=0.50))
    positions = {"A": _Pos("A", 300_000)}
    account = type("Acct", (), {"total_asset": 1_000_000.0})()
    assert g._portfolio_var_scale(account, positions)[0] == pytest.approx(1.0)

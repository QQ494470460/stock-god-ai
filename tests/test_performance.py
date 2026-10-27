# -*- coding: utf-8 -*-
"""业绩与归因引擎测试。

这套指标直接决定「这套策略该不该继续跑」，所以每条口径都要钉住：
分母用当日起始权益（不是前一日）、只统计平仓腿、占位快照必须剔除。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from src.trading import performance as perf


def _write_report(d: Path, date: str, equity: float, start: float,
                  **extra) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    snap = {
        "date": date, "settled_at": f"{date}T15:10:00", "state": "settled",
        "equity": equity, "day_start_equity": start,
        "day_pnl": equity - start,
        "day_pnl_pct": (equity / start - 1) * 100 if start else 0.0,
        "cash": 0.0, "market_value": equity, "n_positions": 1,
        "stats": {"stops": 0, "intents": 0, "executed": 0, "blocked": 0},
    }
    snap.update(extra)
    p = d / f"loop_report_{date.replace('-', '')}.json"
    p.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 装载
def test_load_curve_sorted_and_deduped(tmp_path):
    _write_report(tmp_path, "2026-10-21", 102_000, 100_000)
    _write_report(tmp_path, "2026-10-20", 100_000, 100_000)
    # 同日第二份（补发/重跑），settled_at 更晚应胜出
    _write_report(tmp_path, "2026-10-21", 103_000, 100_000,
                  settled_at="2026-10-21T16:00:00")
    curve = perf.load_equity_curve(reports_dir=tmp_path)
    assert [p["date"] for p in curve] == ["2026-10-20", "2026-10-21"]
    assert curve[-1]["equity"] == 103_000


def test_load_curve_drops_placeholder_snapshot(tmp_path):
    """equity_ok=False 的点必须剔除 —— 一条占位值能把回撤算成 -100%。"""
    _write_report(tmp_path, "2026-10-20", 100_000, 100_000)
    _write_report(tmp_path, "2026-10-21", 0.0, 100_000, equity_ok=False)
    curve = perf.load_equity_curve(reports_dir=tmp_path)
    assert len(curve) == 1
    assert perf.max_drawdown(curve)["pct"] == 0.0


def test_load_curve_empty_falls_back_to_account(tmp_path):
    acct = tmp_path / "paper.json"
    acct.write_text(json.dumps({"cash": 50_000.0, "holdings": {
        "600519": {"market_value": 50_000.0}}}), encoding="utf-8")
    curve = perf.load_equity_curve(reports_dir=tmp_path / "nope", account_file=acct)
    assert len(curve) == 1 and curve[0]["equity"] == 100_000.0


# ---------------------------------------------------------------- 指标
def test_daily_returns_uses_day_start(tmp_path):
    """分母必须是当日起始权益，不是前一日收盘（否则隔夜跳空被算进当天）。"""
    _write_report(tmp_path, "2026-10-20", 110_000, 100_000)   # +10%
    _write_report(tmp_path, "2026-10-21", 99_000, 110_000)    # -10%
    curve = perf.load_equity_curve(reports_dir=tmp_path)
    rets = perf.daily_returns(curve)
    assert rets[0] == pytest.approx(0.10)
    assert rets[1] == pytest.approx(-0.10)


def test_max_drawdown_peak_trough(tmp_path):
    for i, eq in enumerate([100_000, 120_000, 90_000, 95_000, 125_000]):
        _write_report(tmp_path, f"2026-10-{20 + i:02d}", eq,
                      100_000 if i == 0 else eq)
    curve = perf.load_equity_curve(reports_dir=tmp_path)
    dd = perf.max_drawdown(curve)
    assert dd["pct"] == pytest.approx(-25.0)          # 120k → 90k
    assert dd["peak_date"] == "2026-10-21"
    assert dd["trough_date"] == "2026-10-22"
    assert dd["recovered"] is True                     # 125k > 120k
    assert dd["days"] == 3


def test_sharpe_none_when_sample_too_small():
    """样本 < 20 个交易日一律 None —— 拿 4 天数据报夏普是自欺。"""
    assert perf.sharpe_ratio([0.01] * 5) is None
    assert perf.sortino_ratio([0.01] * 5) is None


def test_sharpe_positive_for_steady_gains():
    rets = [0.0015 if i % 3 else 0.0005 for i in range(30)]   # 稳步上行且有波动
    s = perf.sharpe_ratio(rets)
    assert s is not None and s > 0
    # 零波动（每天的收益都一样）无法定义夏普 → None，不能用 inf 冒充
    assert perf.sharpe_ratio([0.001] * 30) is None


def test_annualized_none_for_short_sample(tmp_path):
    _write_report(tmp_path, "2026-10-20", 100_000, 100_000)
    curve = perf.load_equity_curve(reports_dir=tmp_path)
    assert perf.annualized_return(curve) is None


def test_monthly_matrix(tmp_path):
    _write_report(tmp_path, "2026-09-30", 100_000, 100_000)
    _write_report(tmp_path, "2026-10-31", 110_000, 100_000)
    curve = perf.load_equity_curve(reports_dir=tmp_path)
    m = perf.monthly_matrix(curve)
    assert m["2026-10"] == pytest.approx(10.0)


# ---------------------------------------------------------------- 交易统计
def _trades():
    return [
        {"side": "buy", "code": "600519", "name": "贵州茅台", "amount": 10_000,
         "commission": 5, "strategy": "ai"},
        {"side": "sell", "code": "600519", "name": "贵州茅台", "amount": 12_000,
         "commission": 5, "stamp_tax": 12, "pnl": 1_985, "strategy": "ai"},
        {"side": "buy", "code": "300750", "name": "宁德时代", "amount": 10_000,
         "commission": 5, "strategy": "stop"},
        {"side": "sell", "code": "300750", "name": "宁德时代", "amount": 5_000,
         "commission": 5, "stamp_tax": 5, "pnl": -5_010, "strategy": "stop"},
        # 平仓腿但 pnl=0（如手工调仓）→ 不计入胜率
        {"side": "sell", "code": "000001", "name": "平安银行", "amount": 1_000,
         "commission": 5, "pnl": 0},
    ]


def test_trade_stats_only_counts_closed_legs():
    s = perf.trade_stats(_trades())
    assert s["n_closes"] == 2          # pnl=0 那条不计
    assert s["n_buys"] == 2
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["profit_factor"] == pytest.approx(1985 / 5010, rel=1e-3)
    assert s["expectancy"] == pytest.approx((1985 - 5010) / 2)
    assert s["fees"] > 0


def test_trade_stats_max_consecutive_losses():
    t = [{"side": "sell", "pnl": -1}, {"side": "sell", "pnl": -1},
         {"side": "sell", "pnl": 5}, {"side": "sell", "pnl": -1}]
    assert perf.trade_stats(t)["max_consecutive_losses"] == 2


def test_attribution_by_sector_marks_unknown():
    """映射缺失要归入「未知行业」而不是消失 —— 消失会让归因表凭空虚高。"""
    a = perf.attribute(_trades(), industry_map={"600519": "白酒"})
    assert "白酒" in a["by_sector"]
    assert "未知行业" in a["by_sector"]        # 300750 / 000001 无映射
    assert a["by_sector"]["白酒"]["pnl"] == pytest.approx(1985)
    assert "ai" in a["by_strategy"] and "stop" in a["by_strategy"]


# ---------------------------------------------------------------- 汇总
def test_build_report_survives_empty_data(tmp_path):
    r = perf.build_report(reports_dir=tmp_path / "none",
                          history_file=tmp_path / "none.json",
                          industry_file=tmp_path / "none_ind.json",
                          account_file=tmp_path / "none_acct.json")
    assert r["period"]["n_days"] == 0
    assert r["equity"]["total_return_pct"] is None
    assert r["risk"]["max_drawdown_pct"] == 0.0
    assert "业绩档案" in perf.format_text(r)


def test_build_report_end_to_end(tmp_path):
    for i, eq in enumerate([100_000, 101_000, 99_000, 102_000]):
        _write_report(tmp_path, f"2026-10-{20 + i:02d}", eq,
                      100_000 if i == 0 else eq)
    hist = tmp_path / "hist.json"
    hist.write_text(json.dumps(_trades()), encoding="utf-8")
    r = perf.build_report(reports_dir=tmp_path, history_file=hist,
                          industry_file=tmp_path / "none.json")
    assert r["equity"]["total_return_pct"] == pytest.approx(2.0)
    assert r["trades"]["n_closes"] == 2
    assert len(r["curve"]) == 4 and len(r["drawdown"]) == 4
    txt = perf.format_text(r)
    assert "最大回撤" in txt and "行业贡献" in txt


def test_save_report_is_atomic(tmp_path):
    r = {"a": 1}
    p = perf.save_report(r, tmp_path / "perf.json")
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
    assert not (tmp_path / "perf.json.tmp").exists()


def test_exit_reason_distribution_fixed_buckets_with_zero():
    """退出原因必须固定桶（含 0 值），否则「止盈 0 次 / 移动止损 0 次」会被藏起来。

    实测：配 +15% 止盈线、实际最大盈利仅 +1.62%，止盈一次都没触发；
    旧归因只有一个笼统的 stop 桶，这个事实完全不可见。
    """
    trades = [
        {"side": "sell", "code": "A", "pnl": 100.0, "strategy": "stop_profit"},
        {"side": "sell", "code": "B", "pnl": -200.0, "strategy": "stop_loss"},
        {"side": "sell", "code": "C", "pnl": 50.0, "strategy": "rebalance"},
        {"side": "buy", "code": "D", "pnl": 999.0, "strategy": "rebalance"},   # 买入腿忽略
        {"side": "sell", "code": "E", "pnl": 0.0, "strategy": "stop_loss"},    # pnl=0 忽略
    ]
    erd = perf.exit_reason_distribution(trades)
    tags = [e["tag"] for e in erd]
    # 固定顺序：调仓 / 止盈 / 止损 / 移动止损 / 旧标签 / 其他
    assert tags == ["rebalance", "stop_profit", "stop_loss", "trail_stop",
                    "stop", "exit"], tags
    by = {e["tag"]: e for e in erd}
    assert by["stop_profit"]["count"] == 1 and by["stop_profit"]["pnl"] == 100.0
    assert by["stop_loss"]["count"] == 1 and by["stop_loss"]["pnl"] == -200.0
    assert by["rebalance"]["count"] == 1
    # 从未触发的桶必须显式存在（0 值），这是这条指标存在的全部意义
    assert by["trail_stop"]["count"] == 0, "移动止损从未触发必须显式可见"
    assert by["exit"]["count"] == 0


def test_exit_reason_distribution_legacy_stop_is_not_other():
    """旧标签 "stop" 必须单列「风控离场(旧标签)」，绝不能并入「其他退出」。

    在真实数据上验证过：9 笔亏损腿的历史标签是旧的 "stop"，若并入「其他退出」
    会看起来像「根本没有止损」，直接误导归因。
    """
    trades = [{"side": "sell", "code": "A", "pnl": -5000.0, "strategy": "stop"}]
    by = {e["tag"]: e for e in perf.exit_reason_distribution(trades)}
    assert by["stop"]["count"] == 1 and by["stop"]["pnl"] == -5000.0
    assert by["exit"]["count"] == 0, "旧 stop 被误归入其他退出"


def test_exit_reason_distribution_unknown_tag_falls_to_exit():
    """未知标签归入「其他退出」，不丢弃。"""
    trades = [{"side": "sell", "code": "A", "pnl": 10.0, "strategy": "weird_tag"}]
    erd = perf.exit_reason_distribution(trades)
    by = {e["tag"]: e for e in erd}
    assert by["exit"]["count"] == 1 and by["exit"]["pnl"] == 10.0

# -*- coding: utf-8 -*-
"""看板「组合风险（VaR / 流动性）」视图测试。

同样用 AST 切片：``src/ui/app.py`` 是 Streamlit 顶层脚本，直接 import 有副作用。
被测的两个函数必须留在 app.py 内（冻结包只打包被引用模块），故切片 exec。

重点保护两件事：
1. **口径**：VaR 权重是「市值 / 总资产」，不是「市值 / 持仓合计」；
2. **fail-safe**：账户文件缺失/损坏/holdings 为空 → 返回 ``{}``，页面显示空态而不是崩。
"""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

_WANT = ("_fmt_money_short", "load_portfolio_risk_view", "render_portfolio_risk_panel")


def _load_fragment(reader):
    src = (ROOT / "src" / "ui" / "app.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _WANT]
    missing = set(_WANT) - {n.name for n in nodes}
    assert not missing, f"app.py 里找不到 {sorted(missing)}（改名了？同步更新本测试）"

    ns = {"Path": Path, "json": json, "_status_read_json": reader}
    frag = types.ModuleType("app_risk_fragment")
    frag.__dict__.update(ns)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<app-frag>", "exec"),
         frag.__dict__)
    return frag


def _reader(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


frag = _load_fragment(_reader)


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def _write_kline(data_dir: Path, code: str, closes) -> None:
    pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
        "close": closes,
        "volume": [1_000_000] * len(closes),
        "amount": [5e7] * len(closes),
    }).to_parquet(data_dir / f"daily_{code}_qfq.parquet")


def _write_account(path: Path, cash, holdings) -> None:
    path.write_text(json.dumps({
        "cash": cash,
        "holdings": holdings,
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# load_portfolio_risk_view
# ---------------------------------------------------------------------------
def test_missing_account_returns_empty(tmp_path):
    view = frag.load_portfolio_risk_view(
        account_path=tmp_path / "nope.json", data_dir=tmp_path)
    assert view == {}


def test_empty_holdings_returns_empty(tmp_path):
    acct = tmp_path / "pa.json"
    _write_account(acct, 1_000_000.0, {})
    assert frag.load_portfolio_risk_view(
        account_path=acct, data_dir=tmp_path) == {}


def test_var_weights_are_relative_to_total_assets(tmp_path):
    """权重口径回归：持仓 20% → 权重合计 20%，VaR 不该按满仓算。"""
    d = tmp_path
    _write_kline(d, "600000", [10 + i * 0.05 for i in range(80)])
    _write_kline(d, "600001", [20 + i * 0.04 for i in range(80)])
    acct = d / "pa.json"
    _write_account(acct, 800_000.0, {
        "600000": {"quantity": 10_000, "market_value": 100_000.0, "last_price": 10.0},
        "600001": {"quantity": 5_000, "market_value": 100_000.0, "last_price": 20.0},
    })

    view = frag.load_portfolio_risk_view(account_path=acct, data_dir=d)

    assert view["equity"] == pytest.approx(1_000_000.0)
    assert view["invested_pct"] == pytest.approx(0.2)
    assert len(view["rows"]) == 2
    assert sum(r["weight"] for r in view["rows"]) == pytest.approx(0.2, abs=1e-9)
    assert view["var_pct"] is not None and view["var_pct"] > 0
    assert view["var_pct"] < 0.2, "20% 仓位的 VaR 不可能接近满仓量级"


def test_rows_expose_adv_and_cap(tmp_path):
    d = tmp_path
    _write_kline(d, "600000", [10 + i * 0.05 for i in range(80)])
    _write_kline(d, "600001", [20 + i * 0.04 for i in range(80)])
    acct = d / "pa.json"
    _write_account(acct, 0.0, {
        "600000": {"quantity": 10_000, "market_value": 100_000.0, "last_price": 10.0},
        "600001": {"quantity": 5_000, "market_value": 100_000.0, "last_price": 20.0},
    })

    view = frag.load_portfolio_risk_view(account_path=acct, data_dir=d)
    row = next(r for r in view["rows"] if r["code"] == "600000")

    assert row["adv"] == pytest.approx(5e7)
    assert row["cap"] == pytest.approx(5e5)  # 1% × 5000万


def test_fmt_money_short():
    assert frag._fmt_money_short(2.5e8) == "2.50亿"
    assert frag._fmt_money_short(3.2e4) == "3.2万"
    assert frag._fmt_money_short(None) == "N/A"


# ---------------------------------------------------------------------------
# render（假 st，验证不抛 + 关键文案）
# ---------------------------------------------------------------------------
class _Col:
    def __init__(self):
        self.metrics = []

    def metric(self, label, value, **_kw):
        self.metrics.append((label, value))


class _FakeSt:
    def __init__(self):
        self.markdowns = []
        self.captions = []
        self._cols = []

    def markdown(self, s, **_kw):
        self.markdowns.append(str(s))

    def caption(self, s, **_kw):
        self.captions.append(str(s))

    def columns(self, n):
        self._cols = [_Col() for _ in range(n if isinstance(n, int) else len(n))]
        return self._cols


def _render_with(tmp_path, monkeypatch):
    monkeypatch.setattr(frag, "load_portfolio_risk_view", lambda *a, **k: {
        "equity": 1_000_000.0, "invested_pct": 0.2,
        "var_pct": 0.0123, "avg_corr": 0.45, "var_cap": 0.025,
        "rows": [{"code": "600000", "weight": 0.1, "adv": 5e7, "cap": 5e5}],
    })
    fake = _FakeSt()
    monkeypatch.setattr(frag, "st", fake, raising=False)
    frag.render_portfolio_risk_panel()
    return fake


def test_render_shows_metrics_and_rows(tmp_path, monkeypatch):
    fake = _render_with(tmp_path, monkeypatch)
    labels = [m[0] for c in fake._cols for m in c.metrics]
    assert "组合 1日95% VaR" in labels
    assert "平均相关性" in labels
    body = "\n".join(fake.markdowns)
    assert "600000" in body and "单笔上限" in body


def test_render_empty_state_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(frag, "load_portfolio_risk_view", lambda *a, **k: {})
    fake = _FakeSt()
    monkeypatch.setattr(frag, "st", fake, raising=False)
    frag.render_portfolio_risk_panel()
    assert any("不可用" in c for c in fake.captions)

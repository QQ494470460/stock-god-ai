# -*- coding: utf-8 -*-
"""PaperAccount 本金基线一致性测试（防止「双账户态不一致」幽灵收益）。

历史存档若未记录 initial_cash（账户被外部注入了资金/持仓），加载时必须把
initial_cash 对齐到「载入时总资产」，否则盈亏/收益率会被按默认 ¥100,000 误算，
出现 4890% 这类幽灵收益，与风控基线的真实权益对不上。
"""

import json

import pytest

from src.trading.paper import PaperAccount, PAPER_ACCOUNT_FILE, PAPER_HISTORY_FILE


def _patch_paths(monkeypatch, tmp_path):
    acct = tmp_path / "paper_account.json"
    hist = tmp_path / "paper_history.json"
    monkeypatch.setattr("src.trading.paper.PAPER_ACCOUNT_FILE", acct)
    monkeypatch.setattr("src.trading.paper.PAPER_HISTORY_FILE", hist)
    return acct


def test_seeded_account_reconciles_initial_cash(monkeypatch, tmp_path):
    """被外部注入资金的「历史存档」（无 initial_cash 字段）加载后，
    本金基线必须等于载入时总资产，账户级盈亏接近 0。"""
    f = _patch_paths(monkeypatch, tmp_path)
    data = {
        "cash": 4545223.775,
        "holdings": {
            "600519": {
                "name": "贵州茅台", "quantity": 300, "available": 0,
                "today_quantity": 300, "avg_cost": 1501.725,
                "market_value": 450000.0, "last_price": 1501.95,
            }
        },
        "today_pnl": -3883.29, "today_trades": 2, "last_reset_date": "2026-10-15",
    }
    f.write_text(json.dumps(data), encoding="utf-8")

    pa = PaperAccount()
    expected_equity = 4545223.775 + 450000.0
    assert abs(pa.initial_cash - expected_equity) < 0.01
    # 注入资本不算盈利：账户级盈亏应接近 0
    assert abs(pa.get_total_pnl()) < 1.0
    assert abs(pa.get_total_pnl_pct()) < 0.01


def test_reconciliation_is_persisted_on_save(monkeypatch, tmp_path):
    """reconcile 后的 initial_cash 应随下一次 save 落盘，保证重启后仍一致。"""
    f = _patch_paths(monkeypatch, tmp_path)
    data = {
        "cash": 2000000.0, "holdings": {},
        "today_pnl": 0.0, "today_trades": 0, "last_reset_date": "2026-10-15",
    }
    f.write_text(json.dumps(data), encoding="utf-8")

    pa = PaperAccount()
    assert abs(pa.initial_cash - 2000000.0) < 0.01
    pa.buy("000001", "测试股", 10.0, 100)  # 触发 _save
    saved = json.loads(f.read_text(encoding="utf-8"))
    assert abs(float(saved["initial_cash"]) - 2000000.0) < 0.01


def test_fresh_account_keeps_explicit_initial_cash(monkeypatch, tmp_path):
    """全新账户（无存档文件）沿用显式传入的 initial_cash，不被篡改。"""
    _patch_paths(monkeypatch, tmp_path)  # 文件不存在 -> 新建
    pa = PaperAccount(initial_cash=100000.0)
    assert pa.initial_cash == 100000.0
    assert pa.cash == 100000.0
    ok, _, _ = pa.buy("000001", "测试股", 10.0, 100)
    assert ok
    # 含费用，账户级盈亏应略小于 0
    assert pa.get_total_pnl() < 0

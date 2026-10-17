# -*- coding: utf-8 -*-
"""PaperAccount 本金基线一致性 + 生产文件隔离测试。

历史存档若未记录 initial_cash（账户被外部注入了资金/持仓），加载时必须把
initial_cash 对齐到「载入时总资产」，否则盈亏/收益率会被按默认 ¥100,000 误算，
出现 4890% 这类幽灵收益，与风控基线的真实权益对不上。

同时钉死一条纪律：构造纸账户必须显式传 ``account_file`` / ``history_file``
（测试传 tmp 路径）。默认路径指向生产的 ``data/paper_account.json``，
一旦漏传，单元测试就会把真实纸账户覆盖掉。
"""

import hashlib
import json
from pathlib import Path

import pytest

from src.trading.paper import PaperAccount, PAPER_ACCOUNT_FILE, PAPER_HISTORY_FILE

ROOT = Path(__file__).resolve().parent.parent


def _md5(path: Path) -> str:
    if not path.exists():
        return "<absent>"
    return hashlib.md5(path.read_bytes()).hexdigest()


def _new_account(tmp_path, initial_cash: float = 100000.0, seed: dict = None):
    """建一个完全指向 tmp 的纸账户（可选预置存档）。"""
    acct = tmp_path / "paper_account.json"
    hist = tmp_path / "paper_history.json"
    if seed is not None:
        acct.write_text(json.dumps(seed), encoding="utf-8")
    return PaperAccount(initial_cash=initial_cash, account_file=acct, history_file=hist), acct


# ======================================================================
# 本金基线
# ======================================================================
def test_seeded_account_reconciles_initial_cash(tmp_path):
    """被外部注入资金的「历史存档」（无 initial_cash 字段）加载后，
    本金基线必须等于载入时总资产，账户级盈亏接近 0。"""
    seed = {
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
    pa, _ = _new_account(tmp_path, seed=seed)

    expected_equity = 4545223.775 + 450000.0
    assert abs(pa.initial_cash - expected_equity) < 0.01
    # 注入资本不算盈利：账户级盈亏应接近 0
    assert abs(pa.get_total_pnl()) < 1.0
    assert abs(pa.get_total_pnl_pct()) < 0.01


def test_reconciliation_is_persisted_on_save(tmp_path):
    """reconcile 后的 initial_cash 应随下一次 save 落盘，保证重启后仍一致。"""
    seed = {
        "cash": 2000000.0, "holdings": {},
        "today_pnl": 0.0, "today_trades": 0, "last_reset_date": "2026-10-15",
    }
    pa, acct = _new_account(tmp_path, seed=seed)

    assert abs(pa.initial_cash - 2000000.0) < 0.01
    pa.buy("000001", "测试股", 10.0, 100)  # 触发 _save
    saved = json.loads(acct.read_text(encoding="utf-8"))
    assert abs(float(saved["initial_cash"]) - 2000000.0) < 0.01


def test_fresh_account_keeps_explicit_initial_cash(tmp_path):
    """全新账户（无存档文件）沿用显式传入的 initial_cash，不被篡改。"""
    pa, _ = _new_account(tmp_path, initial_cash=100000.0)
    assert pa.initial_cash == 100000.0
    assert pa.cash == 100000.0
    ok, _, _ = pa.buy("000001", "测试股", 10.0, 100)
    assert ok
    # 含费用，账户级盈亏应略小于 0
    assert pa.get_total_pnl() < 0


# ======================================================================
# 生产文件隔离（负向对照：真实文件逐字节不变）
# ======================================================================
def test_account_file_argument_never_touches_production(tmp_path):
    """显式传 account_file 后，生产 data/paper_account.json 必须逐字节不变。"""
    prod_acct = ROOT / "data" / "paper_account.json"
    prod_hist = ROOT / "data" / "paper_trade_history.json"
    before = (_md5(prod_acct), _md5(prod_hist))

    pa, acct = _new_account(tmp_path, initial_cash=1_000_000.0)
    pa.buy("600519", "贵州茅台", 1500.0, 100)
    pa.sell("600519", 1550.0, 100)
    pa._save()          # 强制落盘
    pa._save_history()  # 强制写历史

    after = (_md5(prod_acct), _md5(prod_hist))
    assert before == after, "测试写到了生产纸账户文件！"
    assert acct.exists(), "tmp 路径下应该真的写下存档"


def test_default_paths_point_at_production():
    """默认路径必须仍是生产的 data/ 目录（防止有人把默认值改歪）。"""
    assert PAPER_ACCOUNT_FILE == ROOT / "data" / "paper_account.json"
    assert PAPER_HISTORY_FILE == ROOT / "data" / "paper_trade_history.json"

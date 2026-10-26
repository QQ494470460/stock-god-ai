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
    本金基线必须等于载入时总资产，账户级盈亏接近 0。

    注意：载入时会先把 market_value 拉回 quantity × last_price，
    所以「载入时总资产」按现价算，不按存档里可能陈旧的 market_value 算。
    """
    seed = {
        "cash": 4545223.775,
        "holdings": {
            "600519": {
                "name": "贵州茅台", "quantity": 300, "available": 0,
                "today_quantity": 300, "avg_cost": 1501.725,
                # 故意留一个陈旧市值：归一化应把它拉回 300 × 1501.95
                "market_value": 450000.0, "last_price": 1501.95,
            }
        },
        "today_pnl": -3883.29, "today_trades": 2, "last_reset_date": "2026-10-15",
    }
    pa, acct = _new_account(tmp_path, seed=seed)

    assert pa.holdings["600519"]["market_value"] == 300 * 1501.95
    expected_equity = 4545223.775 + 300 * 1501.95
    assert abs(pa.initial_cash - expected_equity) < 0.01
    assert abs(pa.get_total_asset() - expected_equity) < 0.01
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


# ======================================================================
# 当日盈亏口径（2026-10-20 排障：today_pnl 是已实现口径，被当成当日盈亏展示）
# ======================================================================

def test_day_start_equity_is_captured_before_first_buy(tmp_path):
    """当日基线必须在**第一笔建仓之前**定格，否则当天第一笔的损益会被吞掉。"""
    pa = PaperAccount(account_file=tmp_path / "pa.json",
                      history_file=tmp_path / "ph.json", initial_cash=100_000.0)
    assert pa.day_start_equity() is None, "新建账户当日基线未定，不应伪造一个数"

    pa.buy("002008", "大族激光", price=10.0, quantity=1000)   # 花 ~1万建仓
    # 现金被抽走，但基线必须仍是 10 万（不能变成 9 万）
    assert pa.day_start_equity() == 100_000.0


def test_today_pnl_market_reflects_floating_loss_not_only_sells(tmp_path):
    """只买不卖时，已实现口径恒为 0，但当日浮亏必须能被看见。"""
    pa = PaperAccount(account_file=tmp_path / "pa.json",
                      history_file=tmp_path / "ph.json", initial_cash=100_000.0)
    pa.buy("002008", "大族激光", price=10.0, quantity=1000)
    pa.mark_to_market({"002008": 9.0})          # 浮亏 1000

    assert pa.today_pnl == 0.0, "已实现口径：本例没有卖出，应为 0"
    # 市值口径必须看到浮亏（不能被建仓抽走的现金当成亏损）
    assert pa.today_pnl_market() < -900.0
    assert pa.today_return_pct() < -0.9


def test_today_pnl_market_ignores_cash_outflow_from_building_position(tmp_path):
    """建仓本身不是亏损：把现金换成都市值不应产生当日亏损。"""
    pa = PaperAccount(account_file=tmp_path / "pa.json",
                      history_file=tmp_path / "ph.json", initial_cash=100_000.0)
    pa.buy("002008", "大族激光", price=10.0, quantity=1000)
    pa.mark_to_market({"002008": 10.0})         # 价格与成本一致

    assert pa.today_pnl_market() < 0, "有手续费，应为小额负数"
    assert pa.today_pnl_market() > -100.0, "只应是手续费量级，绝不是 -9000 的现金缺口"


def test_day_start_equity_survives_restart(tmp_path):
    """跨进程重启后当日基线不能丢，否则收益率会闪回 0 看起来像系统失忆。"""
    pa = PaperAccount(account_file=tmp_path / "pa.json",
                      history_file=tmp_path / "ph.json", initial_cash=100_000.0)
    pa.buy("002008", "大族激光", price=10.0, quantity=1000)
    pa.mark_to_market({"002008": 9.5})
    pa.save()
    before = pa.today_pnl_market()

    pa2 = PaperAccount(account_file=tmp_path / "pa.json",
                       history_file=tmp_path / "ph.json")
    assert pa2.day_start_equity() == 100_000.0
    assert pa2.today_pnl_market() == pytest.approx(before, abs=1.0)


# ======================================================================
# 成交历史落盘（2026-10-21：看板「交易记录」tab 长期空白）
# ======================================================================

def test_trade_history_persisted_on_every_fill(tmp_path):
    """每笔成交都必须把历史落盘 —— 旧实现只在 reset() 落盘，导致看板
    「交易记录」tab 始终空白，且常驻闭环重启后历史全丢。"""
    pa, _ = _new_account(tmp_path, initial_cash=1_000_000.0)
    pa.buy("600519", "贵州茅台", 1500.0, 100)
    pa.buy("000001", "平安银行", 12.0, 1000)
    hist = tmp_path / "paper_history.json"
    assert hist.exists(), "成交历史未落盘"
    recs = json.loads(hist.read_text(encoding="utf-8"))
    assert len(recs) == 2
    assert all(r["side"] == "buy" for r in recs)
    # 必须带成交时间，否则「交易记录」tab 的时间列空白
    assert all(r.get("filled_at") for r in recs), "成交记录缺少 filled_at 时间戳"


def test_sell_fill_is_persisted(tmp_path):
    """卖出成交同样必须落盘（T+1：只能卖『昨日及更早』买入的持仓）。"""
    seed = {
        "cash": 500000.0,
        "holdings": {
            "600519": {"name": "贵州茅台", "quantity": 100, "available": 100,
                       "today_quantity": 0, "avg_cost": 1500.0,
                       "market_value": 150000.0, "last_price": 1500.0},
        },
        "today_pnl": 0.0, "today_trades": 0, "last_reset_date": "2026-10-21",
    }
    pa, _ = _new_account(tmp_path, seed=seed, initial_cash=500000.0)
    ok, _, _ = pa.sell("600519", 1550.0, 100)
    assert ok, "昨日持仓应可卖"
    recs = json.loads((tmp_path / "paper_history.json").read_text(encoding="utf-8"))
    assert any(r["side"] == "sell" for r in recs), "卖出成交未落盘"
    assert recs[-1].get("filled_at"), "卖出记录缺少时间戳"


def test_reload_reads_disk_trades(tmp_path):
    """看板是独立进程：reload() 必须能从磁盘读回常驻闭环写入的成交。"""
    pa, _ = _new_account(tmp_path, initial_cash=1_000_000.0)
    pa.buy("002008", "大族激光", 10.0, 1000)
    # 模拟另一个进程（看板）重新载入
    pa2 = PaperAccount(account_file=tmp_path / "pa.json",
                       history_file=tmp_path / "paper_history.json")
    assert len(pa2.trade_history) == 1, "新进程未从磁盘读回成交历史"
    assert pa2.trade_history[0].filled_at, "读回的成交缺少时间戳"


# ======================================================================
# 重置账户不误删成交历史（2026-10-21：点「重置账户」会把配对历史清空）
# ======================================================================

def test_reset_keeps_history_by_default(tmp_path):
    """重置账户默认保留成交历史，只有显式 keep_history=False 才清空。"""
    pa, _ = _new_account(tmp_path, initial_cash=1_000_000.0)
    pa.buy("600519", "贵州茅台", 1500.0, 100)
    pa.buy("000001", "平安银行", 12.0, 1000)
    assert len(pa.trade_history) == 2

    # 默认：保留历史，只清仓/重置本金
    pa.reset(500_000.0)
    assert pa.cash == 500_000.0
    assert pa.initial_cash == 500_000.0
    assert len(pa.holdings) == 0, "重置应清掉持仓"
    assert len(pa.trade_history) == 2, "默认重置误删了成交历史！"

    # 显式清空：才连历史一起抹掉
    pa.reset(200_000.0, keep_history=False)
    assert len(pa.trade_history) == 0, "keep_history=False 应清空历史"


# ======================================================================
# 买卖配对（2026-10-21：交易记录没有开仓/平仓对应链接）
# ======================================================================

def test_sell_links_to_fifo_buy_via_close_id(tmp_path):
    """卖出必须按 FIFO 把 close_id 勾回最早的那笔开仓买入（双向打标）。"""
    # 预置一笔「昨日」买入（可卖），并把它写进成交历史，模拟已存在的开仓记录
    seed = {
        "cash": 500000.0,
        "holdings": {
            "600519": {"name": "贵州茅台", "quantity": 100, "available": 100,
                       "today_quantity": 0, "avg_cost": 1500.0,
                       "market_value": 150000.0, "last_price": 1500.0},
        },
        "today_pnl": 0.0, "today_trades": 0, "last_reset_date": "2026-10-21",
    }
    hist = tmp_path / "paper_history.json"
    buy_rec = {
        "trade_id": "PB202610210900001000", "code": "600519", "name": "贵州茅台",
        "side": "buy", "price": 1500.0, "quantity": 100, "amount": 150000.0,
        "commission": 5.0, "stamp_tax": 0.0, "filled_at": "2026-10-21T09:00:00",
        "strategy": "seed", "ai_score": 0.0, "pnl": 0.0, "pnl_pct": 0.0, "close_id": "",
    }
    hist.write_text(json.dumps([buy_rec]), encoding="utf-8")
    pa, _ = _new_account(tmp_path, seed=seed, initial_cash=500000.0)

    ok, _, rec = pa.sell("600519", 1550.0, 100)
    assert ok, "昨日持仓应可卖"

    # 卖出记录应指向被平的开仓买入
    assert rec.close_id == "PB202610210900001000", "卖出未配对到开仓买入"
    # 买入记录应被反向打标为已被该卖出平仓
    buy_in_mem = next(t for t in pa.trade_history if t.trade_id == "PB202610210900001000")
    assert buy_in_mem.close_id == rec.trade_id, "开仓买入未反向配对到平仓卖出"

    # 配对必须落盘：重新载入后仍能读回
    pa2 = PaperAccount(account_file=tmp_path / "pa.json",
                       history_file=tmp_path / "paper_history.json")
    sell2 = next(t for t in pa2.trade_history if t.side == "sell")
    buy2 = next(t for t in pa2.trade_history if t.trade_id == "PB202610210900001000")
    assert sell2.close_id == "PB202610210900001000"
    assert buy2.close_id == sell2.trade_id


def test_sell_pairs_to_earliest_open_buy_fifo(tmp_path):
    """同标的多次买入后，卖出应先配对最早（FIFO）那笔开仓。"""
    seed = {
        "cash": 500000.0,
        "holdings": {
            "000001": {"name": "平安银行", "quantity": 200, "available": 200,
                       "today_quantity": 0, "avg_cost": 12.0,
                       "market_value": 2400.0, "last_price": 12.0},
        },
        "today_pnl": 0.0, "today_trades": 0, "last_reset_date": "2026-10-21",
    }
    hist = tmp_path / "paper_history.json"
    hist.write_text(json.dumps([
        {"trade_id": "BUY1", "code": "000001", "name": "平安银行", "side": "buy",
         "price": 10.0, "quantity": 100, "amount": 1000.0, "commission": 5.0,
         "stamp_tax": 0.0, "filled_at": "2026-10-20T09:00:00", "strategy": "s",
         "ai_score": 0.0, "pnl": 0.0, "pnl_pct": 0.0, "close_id": ""},
        {"trade_id": "BUY2", "code": "000001", "name": "平安银行", "side": "buy",
         "price": 14.0, "quantity": 100, "amount": 1400.0, "commission": 5.0,
         "stamp_tax": 0.0, "filled_at": "2026-10-21T09:00:00", "strategy": "s",
         "ai_score": 0.0, "pnl": 0.0, "pnl_pct": 0.0, "close_id": ""},
    ]), encoding="utf-8")
    pa, _ = _new_account(tmp_path, seed=seed, initial_cash=500000.0)

    ok, _, rec = pa.sell("000001", 13.0, 100)
    assert ok
    assert rec.close_id == "BUY1", "应 FIFO 配对到最早的开仓 BUY1，而非 BUY2"


# ======================================================================
# 落盘健壮性（2026-10-21 漏洞修补：两个进程并发写 + taskkill /F 强杀 → 文件被截断）
# ======================================================================

def test_atomic_write_leaves_recoverable_bak(tmp_path):
    """每次落盘都应为上一份完整文件留 .bak，供损坏时回滚。"""
    acct = tmp_path / "paper_account.json"
    hist = tmp_path / "paper_history.json"
    pa = PaperAccount(initial_cash=500_000.0, account_file=acct, history_file=hist)
    pa.buy("600519", "贵州茅台", 1500.0, 100)
    pa.save()  # 第二次落盘 → 此时会为上一份完整文件留 .bak
    bak = acct.with_name(acct.name + ".bak")
    assert bak.exists(), "原子写应为上一份完整文件留 .bak"
    # .bak 必须是合法 JSON（能解析），否则回滚无意义
    recovered = json.loads(bak.read_text(encoding="utf-8"))
    assert "cash" in recovered and "holdings" in recovered


def test_corrupt_main_file_recovers_from_bak(tmp_path):
    """主账户文件被截断/损坏时，必须从 .bak 回滚，而非把账户清空。

    模拟 ``taskkill /F`` 在 ``write_text`` 中途杀进程导致文件半截；下次加载若直接
    抛错，账户会被当成空账户（现金/持仓/历史全丢，2026-10 实测过的灾难）。
    """
    acct = tmp_path / "paper_account.json"
    hist = tmp_path / "paper_history.json"
    pa = PaperAccount(initial_cash=1_000_000.0, account_file=acct, history_file=hist)
    pa.buy("600519", "贵州茅台", 1500.0, 100)
    pa.save()  # 再落盘一次 → 为完整状态留 .bak
    # 人为截断主文件（模拟中途强杀）
    acct.write_text('{ "cash": 12', encoding="utf-8")
    # 新进程应从 .bak 恢复，而非清空账户
    pa2 = PaperAccount(initial_cash=1_000_000.0, account_file=acct, history_file=hist)
    assert "600519" in pa2.holdings, "应从 .bak 恢复持仓"
    assert len(pa2.trade_history) == 1, "应从 .bak 恢复成交历史"
    assert pa2.initial_cash == pytest.approx(1_000_000.0, rel=1e-6)


def test_corrupt_main_and_bak_resets_without_crash(tmp_path):
    """主文件与 .bak 都损坏时，必须优雅退化为新账户，绝不能抛异常卡死加载。"""
    acct = tmp_path / "paper_account.json"
    hist = tmp_path / "paper_history.json"
    acct.write_text("{bad", encoding="utf-8")
    acct.with_name(acct.name + ".bak").write_text("{bad", encoding="utf-8")
    hist.write_text("[bad", encoding="utf-8")
    hist.with_name(hist.name + ".bak").write_text("[bad", encoding="utf-8")
    pa = PaperAccount(initial_cash=100_000.0, account_file=acct, history_file=hist)
    assert pa.cash == 100_000.0
    assert pa.initial_cash == 100_000.0
    assert pa.trade_history == []


def test_buy_preserves_other_process_trade_on_disk(tmp_path):
    """常驻进程(pa1)买入后，另一进程(pa2)再买入不得覆盖 pa1 的成交/持仓。

    这是「跨进程 clobber」漏洞的回归钉：两进程共享同一份文件，pa2 必须在落盘前
    reload 到 pa1 已写入磁盘的状态，再加自己的成交，最终磁盘上两者都在。
    """
    acct = tmp_path / "paper_account.json"
    hist = tmp_path / "paper_history.json"
    pa1 = PaperAccount(initial_cash=1_000_000.0, account_file=acct, history_file=hist)
    pa1.buy("600519", "贵州茅台", 1500.0, 100)   # pa1 落盘

    pa2 = PaperAccount(initial_cash=1_000_000.0, account_file=acct, history_file=hist)
    pa2.buy("000001", "平安银行", 12.0, 1000)     # pa2 应 reload 到 pa1 的状态再买

    pa3 = PaperAccount(initial_cash=1_000_000.0, account_file=acct, history_file=hist)
    assert "600519" in pa3.holdings, "pa1 的持仓被 pa2 覆盖丢失"
    assert "000001" in pa3.holdings, "pa2 的持仓未写入"
    assert len(pa3.trade_history) == 2, "应累计两笔成交，而非互相覆盖"


def test_reset_keeps_other_process_trade_on_disk(tmp_path):
    """看板(pa2)重置账户时，必须先在磁盘对齐常驻进程(pa1)已写入的成交再清仓，
    避免把 pa1 的成交历史覆盖掉（keep_history 默认 True）。"""
    acct = tmp_path / "paper_account.json"
    hist = tmp_path / "paper_history.json"
    pa1 = PaperAccount(initial_cash=1_000_000.0, account_file=acct, history_file=hist)
    pa1.buy("600519", "贵州茅台", 1500.0, 100)   # pa1 落盘
    pa2 = PaperAccount(initial_cash=1_000_000.0, account_file=acct, history_file=hist)
    pa2.reset(500_000.0)                          # pa2 重置（默认保留历史）
    pa3 = PaperAccount(initial_cash=1_000_000.0, account_file=acct, history_file=hist)
    assert pa3.cash == pytest.approx(500_000.0, rel=1e-3), "重置后现金应为新本金"
    assert len(pa3.holdings) == 0, "重置应清仓"
    assert len(pa3.trade_history) == 1, "重置不得覆盖 pa1 的成交历史"



# ======================================================================
# 成本基准口径：买入佣金必须资本化（否则卖出 pnl 系统性高估一个佣金）
# ======================================================================
def test_buy_commission_is_capitalized_into_cost_basis(tmp_path):
    pa = PaperAccount(initial_cash=1_000_000.0,
                      account_file=tmp_path / "a.json",
                      history_file=tmp_path / "h.json")

    ok, _msg, rec = pa.buy("600519", "贵州茅台", 1500.0, 100)
    assert ok and rec is not None

    # 单位成本 = (成交金额 + 佣金) / 股数，与券商「摊薄成本」同口径
    expected = (rec.amount + rec.commission) / rec.quantity
    assert pa.holdings["600519"]["avg_cost"] == pytest.approx(expected, rel=1e-9)
    assert rec.commission >= 5.0, "佣金有最低 5 元"


def test_repeat_buy_blends_commission_into_average_cost(tmp_path):
    pa = PaperAccount(initial_cash=1_000_000.0,
                      account_file=tmp_path / "a.json",
                      history_file=tmp_path / "h.json")

    ok1, _m1, r1 = pa.buy("600519", "贵州茅台", 1500.0, 100)
    ok2, _m2, r2 = pa.buy("600519", "贵州茅台", 1600.0, 100)
    assert ok1 and ok2

    cost1 = r1.amount + r1.commission
    cost2 = r2.amount + r2.commission
    expected = (cost1 + cost2) / (r1.quantity + r2.quantity)
    assert pa.holdings["600519"]["avg_cost"] == pytest.approx(expected, rel=1e-9)


def test_sell_pnl_does_not_double_count_buy_commission(tmp_path):
    """买入佣金已进成本，卖出 pnl 只需再扣卖出费，不应重复计买入佣金。"""
    pa = PaperAccount(initial_cash=1_000_000.0,
                      account_file=tmp_path / "a.json",
                      history_file=tmp_path / "h.json")
    _ok, _m, buy_rec = pa.buy("600519", "贵州茅台", 100.0, 1000)
    # 模拟「昨日买入」：解锁 T+1，否则当日不可卖。必须落盘 ——
    # sell() 开头会 _load() 对齐磁盘最新态，只改内存会被覆盖。
    _h = pa.holdings["600519"]
    _h["today_quantity"] = 0
    _h["available"] = _h["quantity"]
    pa._save()
    ok, _m2, sell_rec = pa.sell("600519", 100.0, 1000)
    assert ok

    # 恒等式：pnl = 卖出净收入 - (买入金额 + 买入佣金) - 卖出佣金 - 印花税
    # 若成本基准漏计买入佣金，这里会多出一笔 comm 的差额。
    expected = (
        sell_rec.amount
        - (buy_rec.amount + buy_rec.commission)
        - sell_rec.commission
        - sell_rec.stamp_tax
    )
    assert sell_rec.pnl == pytest.approx(expected, rel=1e-9)


# ======================================================================
# T+1 字段不变量 / 卖出市值口径
# ======================================================================

def test_load_clamps_t1_fields_into_range(tmp_path):
    """载入时必须把 available / today_quantity 夹进 [0, quantity]。

    来源可以是旧存档、手工编辑、外部注入，也可以是 ``taskkill /F`` 打断的
    半截写入。越界值看起来只是显示难看，实际会**掐死止损**：
    ``sellable_quantity()`` 是 ``max(0, min(available, quantity - today, quantity))``，
    负 available 会被夹成 0 ⇒ 该标的当天 sellable=0 ⇒ 卖出在预检阶段就被
    「T+1 锁仓」拦下，且永远不会自愈（只有次日 reset 才修）。
    """
    seed = {
        "cash": 1000.0,
        "holdings": {
            "600519": {
                "name": "贵州茅台", "quantity": 100,
                "available": 9999,          # 越上界
                "today_quantity": 5000,     # 越上界
                "avg_cost": 100.0, "last_price": 100.0, "market_value": 10000.0,
            },
            "000001": {
                "name": "平安银行", "quantity": 200,
                "available": -80,           # 越下界（会让止损永久失效）
                "today_quantity": -5,
                "avg_cost": 10.0, "last_price": 10.0, "market_value": 2000.0,
            },
        },
        "today_pnl": 0.0, "today_trades": 0, "last_reset_date": "2026-10-15",
    }
    pa, _ = _new_account(tmp_path, seed=seed)

    # 越界值按「quantity 权威、今日买入量次之」重建：available = quantity - today
    assert pa.holdings["600519"]["today_quantity"] == 100   # 5000 → 100
    assert pa.holdings["600519"]["available"] == 0          # 100 - 100
    assert pa.holdings["000001"]["today_quantity"] == 0     # -5 → 0
    assert pa.holdings["000001"]["available"] == 200        # 200 - 0（不是夹成 0！）
    # 关键：可卖量必须回到「真实可卖」，而不是被负值夹死
    from src.trading.rules import sellable_quantity
    assert sellable_quantity(pa.holdings["000001"]) == 200
    assert sellable_quantity(pa.holdings["600519"]) == 0


def test_sell_refreshes_mark_and_keeps_available_in_range(tmp_path):
    """卖出后剩余仓位必须同步刷新 ``last_price``（否则本次标价会被陈价覆盖）。

    ``_normalize_holdings`` 每次落盘都按 ``quantity × last_price`` 重算市值，
    卖出时若只改 ``market_value`` 而不动 ``last_price``，本次标价会在落盘那一刻
    被**隔夜价**覆盖 —— 看板手动卖出（不经 loop 的 mark_to_market）正是这条路，
    结果总资产/收益率继续按昨天的价算，且完全看不出来源。
    """
    from src.trading.rules import apply_slippage

    seed = {
        "cash": 0.0,
        "holdings": {
            "600519": {
                "name": "贵州茅台", "quantity": 300, "available": 300,
                "today_quantity": 0,
                "avg_cost": 10.0, "last_price": 10.0, "market_value": 3000.0,
            },
        },
        "today_pnl": 0.0, "today_trades": 0, "last_reset_date": "2026-10-15",
    }
    pa, _ = _new_account(tmp_path, seed=seed)

    ok, msg, rec = pa.sell("600519", 20.0, 100)
    assert ok, msg
    h = pa.holdings["600519"]
    assert h["quantity"] == 200
    assert h["available"] == 200                 # 不越界
    assert rec.price == pytest.approx(apply_slippage(20.0, "sell", 20.0 * 100))
    # 标价按盘口价 20（市场价），且落盘后不会被陈价 10 覆盖
    assert h["last_price"] == 20.0
    assert h["market_value"] == pytest.approx(200 * 20.0)
    pa.reload()
    assert pa.holdings["600519"]["market_value"] == pytest.approx(200 * 20.0)

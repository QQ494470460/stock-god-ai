# -*- coding: utf-8 -*-
"""「计划 vs 实际成交」审计工具 —— 仓位建满度。

为什么这个工具值得有测试
------------------------
它是第 14 处漂移（未知行业桶限仓）的取证工具，本身出错就会误导诊断方向。
两条硬约束：

1. **不能误报**：计划日期在未来、或当天还没开盘时，一律 ``pending``，
   不许把「还没到执行」渲染成「执行失败」。凌晨跑批时最容易踩。
2. **必须能定位**：建满度低时要能回答「哪只票没买进、被什么原因拦的」，
   否则只知道「仓位建不满」却无从下手。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.plan_vs_fill as pvf  # noqa: E402

PLAN_DAY = "20261102"


# ---------------------------------------------------------------------------
# 构造器
# ---------------------------------------------------------------------------

def _env(tmp_path: Path) -> Path:
    ddir = tmp_path / "data"
    (ddir / "trade_plans").mkdir(parents=True, exist_ok=True)
    return ddir


def _plan(ddir: Path, date: str, targets, source="daily_pick", regime="neutral"):
    """targets: [(code, name, weight), ...]"""
    p = ddir / "trade_plans" / f"trade_plan_{date}.json"
    p.write_text(json.dumps({
        "source": source,
        "regime": regime,
        "targets": [{"code": c, "name": n, "weight": w} for c, n, w in targets],
    }, ensure_ascii=False), encoding="utf-8")


def _account(ddir: Path, holdings, cash: float):
    """holdings: {code: (quantity, last_price)}"""
    pa = {
        "cash": cash,
        "holdings": {c: {"quantity": q, "last_price": px} for c, (q, px) in holdings.items()},
    }
    (ddir / "paper_account.json").write_text(json.dumps(pa, ensure_ascii=False), encoding="utf-8")


def _journal(ddir: Path, records):
    (ddir / "trading_journal.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")


def _after_open() -> datetime:
    return datetime(2026, 11, 2, 10, 0, 0)


_THREE = [("600000", "浦发银行", 0.30), ("600036", "招商银行", 0.30), ("601318", "中国平安", 0.30)]


# ---------------------------------------------------------------------------
# 建满度算得对
# ---------------------------------------------------------------------------

def test_fully_filled_plan_is_healthy(tmp_path):
    """8 只按计划买满 → 1.0，健康。总额 100 万：90 万持仓 + 10 万现金。"""
    ddir = _env(tmp_path)
    _plan(ddir, PLAN_DAY, _THREE)
    _account(ddir, {"600000": (30000, 10.0), "600036": (30000, 10.0),
                    "601318": (30000, 10.0)}, 100_000)

    rep = pvf.audit(PLAN_DAY, ddir, now=_after_open())

    assert rep["weight_plan_total"] == 0.9
    assert rep["weight_filled_total"] == 0.9
    assert rep["fill_ratio"] == 1.0
    assert rep["healthy"] is True
    assert rep["n_missing"] == 0 and rep["n_partially_filled"] == 0
    assert {r["status"] for r in rep["positions"]} == {"ok"}


def test_one_missing_buy_lowers_fill_ratio_to_two_thirds(tmp_path):
    """3 只计划只买进 2 只 → 0.6/0.9 ≈ 0.667，判为偏空（11-02 的形态）。"""
    ddir = _env(tmp_path)
    _plan(ddir, PLAN_DAY, _THREE)
    _account(ddir, {"600000": (30000, 10.0), "600036": (30000, 10.0)}, 400_000)

    rep = pvf.audit(PLAN_DAY, ddir, now=_after_open())

    assert abs(rep["fill_ratio"] - 0.6 / 0.9) < 1e-4
    assert rep["healthy"] is False
    assert rep["n_missing"] == 1
    missing = [r for r in rep["positions"] if r["status"] == "missing"]
    assert [r["code"] for r in missing] == ["601318"]


def test_partial_fill_is_flagged_but_not_missing(tmp_path):
    """买到一半算 partially_filled —— 和「完全没买」是两种病，不能混为一谈。"""
    ddir = _env(tmp_path)
    _plan(ddir, PLAN_DAY, _THREE[:2])
    _account(ddir, {"600000": (15000, 10.0)}, 850_000)

    rep = pvf.audit(PLAN_DAY, ddir, now=_after_open())

    by_code = {r["code"]: r for r in rep["positions"]}
    assert by_code["600000"]["status"] == "partially_filled"
    assert by_code["600000"]["weight_actual"] == 0.15
    assert by_code["600036"]["status"] == "missing"
    assert rep["n_partially_filled"] == 1 and rep["n_missing"] == 1


def test_positions_outside_the_plan_do_not_count_as_filled(tmp_path):
    """老仓不是「建满」。否则满手老票时建满度虚高，缺口被掩盖。"""
    ddir = _env(tmp_path)
    _plan(ddir, PLAN_DAY, _THREE)
    _account(ddir, {"600000": (30000, 10.0),     # 计划内，已建满
                    "600519": (6000, 100.0)},    # 计划外的老仓 60 万
            100_000)

    rep = pvf.audit(PLAN_DAY, ddir, now=_after_open())

    assert rep["equity"] == 1_000_000
    assert rep["weight_filled_total"] == 0.30          # 只有计划内那 30 万算数
    assert abs(rep["fill_ratio"] - 1 / 3) < 1e-4
    assert rep["n_missing"] == 2


# ---------------------------------------------------------------------------
# 拦截原因要能定位
# ---------------------------------------------------------------------------

def test_blocked_reason_is_attached_to_the_code(tmp_path):
    """journal 里的 risk_block 原因必须挂回对应票 —— 这是工具存在的全部意义。"""
    ddir = _env(tmp_path)
    _plan(ddir, PLAN_DAY, _THREE)
    _account(ddir, {"600000": (30000, 10.0), "600036": (30000, 10.0)}, 400_000)
    _journal(ddir, [
        {"ts": "2026-11-02T09:31:05", "event": "risk_block",
         "data": {"code": "601318", "reason": "未知行业(映射缺失) 集中度已达上限 30%"}},
    ])

    rep = pvf.audit(PLAN_DAY, ddir, now=_after_open())

    by_code = {r["code"]: r for r in rep["positions"]}
    assert "集中度" in by_code["601318"]["blocked_reason"]
    assert by_code["600000"]["blocked_reason"] == ""


def test_blocks_on_ignores_entries_from_other_days(tmp_path):
    """昨天的拦截不能算到今天的头上（同代码跨日复现时最容易张冠李戴）。"""
    ddir = _env(tmp_path)
    _plan(ddir, PLAN_DAY, _THREE[:1])
    _account(ddir, {"600000": (30000, 10.0)}, 700_000)
    _journal(ddir, [
        {"ts": "2026-11-01T09:31:05", "event": "risk_block",
         "data": {"code": "600000", "reason": "昨天的旧原因"}},
        {"ts": "2026-11-02T09:31:05", "event": "risk_block",
         "data": {"code": "600000", "reason": "今天的原因"}},
    ])

    blocks = pvf.blocks_on(PLAN_DAY, ddir)

    assert blocks["600000"] == "今天的原因"


# ---------------------------------------------------------------------------
# pending：不许把「还没到执行」报成缺口
# ---------------------------------------------------------------------------

def test_future_plan_is_pending_not_a_gap(tmp_path):
    """前一晚生成的明天计划，今天建满度必然是 0 —— 那不是缺口。"""
    ddir = _env(tmp_path)
    _plan(ddir, "20261105", _THREE)
    _account(ddir, {}, 1_000_000)

    rep = pvf.audit(None, ddir, now=datetime(2026, 11, 4, 22, 0, 0))

    assert rep["plan_date"] == "20261105"
    assert rep["pending"] is True
    assert rep["healthy"] is True
    assert rep["fill_ratio"] == 0.0
    assert rep["n_missing"] == 3


def test_same_day_before_open_is_pending(tmp_path):
    """当天计划在 00:34 被审计时还没开盘，不能判偏空（曾真实误报过一次）。"""
    ddir = _env(tmp_path)
    _plan(ddir, "20261104", _THREE)
    _account(ddir, {"600519": (600, 100.0)}, 940_000)

    rep = pvf.audit(None, ddir, now=datetime(2026, 11, 4, 0, 34, 0))

    assert rep["pending"] is True
    assert rep["healthy"] is True


def test_same_day_after_open_is_a_real_gap(tmp_path):
    """同样的数据，开盘后就不再是 pending —— 这条守住「pending 不会掩盖真缺口」。"""
    ddir = _env(tmp_path)
    _plan(ddir, "20261104", _THREE)
    _account(ddir, {"600519": (600, 100.0)}, 940_000)

    rep = pvf.audit(None, ddir, now=datetime(2026, 11, 4, 10, 0, 0))

    assert rep["pending"] is False
    assert rep["healthy"] is False


def test_render_marks_pending_instead_of_warning(tmp_path):
    ddir = _env(tmp_path)
    _plan(ddir, "20261105", _THREE)
    _account(ddir, {}, 1_000_000)

    text = pvf.render(pvf.audit(None, ddir, now=datetime(2026, 11, 4, 22, 0, 0)))

    assert "尚未开始执行" in text
    assert "偏空" not in text


def test_render_warns_when_really_short(tmp_path):
    ddir = _env(tmp_path)
    _plan(ddir, PLAN_DAY, _THREE)
    _account(ddir, {"600000": (30000, 10.0), "600036": (30000, 10.0)}, 400_000)

    text = pvf.render(pvf.audit(PLAN_DAY, ddir, now=_after_open()))

    assert "偏空" in text
    assert "完全没买 1 只" in text


# ---------------------------------------------------------------------------
# 空环境不许炸
# ---------------------------------------------------------------------------

def test_no_plan_at_all_returns_empty(tmp_path):
    rep = pvf.audit(None, _env(tmp_path), now=_after_open())

    assert rep["plan_date"] == ""
    assert rep["n_planned"] == 0
    assert rep["fill_ratio"] == 0.0


def test_missing_account_file_does_not_crash(tmp_path):
    """账户文件缺失（首次装、或清理过）时权益为 0，不能除零崩掉。"""
    ddir = _env(tmp_path)
    _plan(ddir, PLAN_DAY, _THREE)

    rep = pvf.audit(PLAN_DAY, ddir, now=_after_open())

    assert rep["equity"] == 0.0
    assert rep["cash_pct"] == 0.0
    assert rep["fill_ratio"] == 0.0

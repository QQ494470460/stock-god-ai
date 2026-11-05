# -*- coding: utf-8 -*-
"""计划 vs 实际成交 审计 —— 量化「仓位建满度」。

为什么需要它
------------
2026-11-03 挖到的第 14 处「实盘有、回测没有」：实盘长期跑不过回测，
根因之一不是选股，是**仓位建不满**。11-02 的计划是 8 只、权重合计 90%，
实际持仓只有 55% —— 三笔买入被「未知行业(映射缺失) 集中度已达上限」
误拦，少掉 35pp 敞口。这类偏差在收益报表里只表现为「少赚一点」，
完全看不出是执行问题，所以必须单独审计。

执行级回测假设「计划买什么就能买到什么、且买满 90%」；
只要这个假设不成立，回测的收益率就永远兑现不了。

用法
----
    python scripts/plan_vs_fill.py                 # 最近一份计划
    python scripts/plan_vs_fill.py --date 20261104
    python scripts/plan_vs_fill.py --json
    python scripts/plan_vs_fill.py --save          # 另存 data/fill_audit.json

判定
----
``fill_ratio`` = 实际总持仓权重 / 计划总权重。
  >= 0.90  健康（把「建仓当日尚未完全成交」容忍在内）
  <  0.90  警告：有票没买进，逐只看拦截原因
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 打包版需要先注入 libs 路径（与其它 scripts 一致）
try:
    import config.settings  # noqa: F401
except Exception:  # noqa: BLE001
    pass

DATA = ROOT / "data"
PLANS = DATA / "trade_plans"
HEALTHY_RATIO = 0.90


def _load(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _iso(date: str) -> str:
    d = str(date or "")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else d


def resolve_plan(date: str | None = None, plans_dir: Path | None = None):
    """返回 (date_str_yyyymmdd, plan_dict)。"""
    pdir = Path(plans_dir) if plans_dir else PLANS
    if date:
        key = str(date).replace("-", "")
        return key, _load(pdir / f"trade_plan_{key}.json", {}) or {}
    files = sorted(pdir.glob("trade_plan_*.json"))
    if not files:
        return "", {}
    f = files[-1]
    return f.stem.replace("trade_plan_", ""), _load(f, {}) or {}


def account_snapshot(data_dir: Path | None = None) -> dict:
    """纸账户快照：逐票市值 + 现金 + 合计。"""
    ddir = Path(data_dir) if data_dir else DATA
    pa = _load(ddir / "paper_account.json", {}) or {}
    holdings = pa.get("holdings") or {}
    mv: dict[str, float] = {}
    for code, h in holdings.items():
        if not isinstance(h, dict):
            continue
        qty = float(h.get("quantity") or 0)
        px = float(h.get("last_price") or h.get("price") or h.get("avg_cost") or 0)
        mv[code] = qty * px
    cash = float(pa.get("cash") or 0)
    return {"mv": mv, "cash": cash, "total": sum(mv.values()) + cash}


def blocks_on(date: str, data_dir: Path | None = None) -> dict[str, str]:
    """当天每只票的最后一条风控拦截原因（用于解释「为什么没买进」）。"""
    ddir = Path(data_dir) if data_dir else DATA
    out: dict[str, str] = {}
    p = ddir / "trading_journal.jsonl"
    if not p.exists() or not date:
        return out
    iso = _iso(date)
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not str(rec.get("ts", "")).startswith(iso):
            continue
        if rec.get("event") != "risk_block":
            continue
        d = rec.get("data") or {}
        code = d.get("code")
        if code:
            out[str(code)] = str(d.get("reason") or "")
    return out


def unmapped_weight(plan_w: dict, ddir: Path) -> tuple[float, list[str]]:
    """计划权重里「没有行业映射」的那部分占比。

    为什么必须单独报：行业集中度上限（``guard.max_sector_pct``）与回测里的
    ``max_sector_pct`` 对未映射票都是 **fail-open**（豁免、不占额度）。
    2026-11-04 的计划 8 只里有 4 只没有映射、合计占仓位 **43.4%** ——
    也就是说那条线对近一半仓位是免检的，而这件事**在任何地方都不会报错**。

    这是一条数据质量问题，不是策略问题：出路是补全 ``data/industry_map.json``
    （只覆盖 2621 / 约 5400 只），不是调阈值。
    """
    ind = _load(ddir / "industry_map.json", {}) or {}
    missing = [c for c in plan_w if not str(ind.get(c) or "").strip()]
    tot = sum(plan_w.values())
    share = (sum(plan_w[c] for c in missing) / tot) if tot > 0 else 0.0
    return share, sorted(missing)


def audit(date: str | None = None, data_dir: Path | None = None,
          now: datetime | None = None) -> dict:
    """now 可注入 —— 否则「凌晨不算缺口」这条判定在测试里无法确定性复现。"""
    now = now or datetime.now()
    ddir = Path(data_dir) if data_dir else DATA
    if date:
        key, plan = resolve_plan(date, ddir / "trade_plans")
    else:
        # 默认审计**今天**那份，而不是文件名最大的那份。
        # 计划是收盘后为下一交易日生成的，所以磁盘上「最大」的那份往往属于明天；
        # 拿它做建满度审计只会得到「尚未开始执行」，把今天真实的缺口藏起来。
        # 2026-11-05 实测：11-06 那份是 11-05 01:04 用 **11-03 的旧 picks** 生成的，
        # 报「48.2% 无行业映射」，而今天 11-05 那份是 0% —— 第一屏就把诊断带偏了。
        # 与「重读计划必须 load_trade_plan(今天)、勿用 load_latest_plan()」同源。
        key, plan = resolve_plan(now.strftime("%Y%m%d"), ddir / "trade_plans")
        if not plan:                    # 今天没有计划 → 退回最新一份（报表会如实写日期）
            key, plan = resolve_plan(None, ddir / "trade_plans")
    snap = account_snapshot(ddir)
    total = snap["total"]
    targets = plan.get("targets") or []
    plan_w = {str(t.get("code")): float(t.get("weight") or 0) for t in targets if t.get("code")}
    plan_sum = sum(plan_w.values())

    blocks = blocks_on(key, ddir)
    rows = []
    filled_sum = 0.0
    for t in targets:
        code = str(t.get("code"))
        w_plan = float(t.get("weight") or 0)
        mv = float(snap["mv"].get(code) or 0)
        w_act = mv / total if total > 0 else 0.0
        # 只统计「计划内」的实际权重，避免把不在计划里的老仓算成建满
        filled_sum += w_act
        gap = w_plan - w_act
        rows.append({
            "code": code,
            "name": t.get("name") or "",
            "weight_plan": round(w_plan, 4),
            "weight_actual": round(w_act, 4),
            "gap": round(gap, 4),
            "status": ("ok" if gap <= max(0.01, w_plan * 0.15)
                       else ("partially_filled" if w_act > 0 else "missing")),
            "blocked_reason": blocks.get(code, ""),
        })

    n_missing = sum(1 for r in rows if r["status"] == "missing")
    n_partial = sum(1 for r in rows if r["status"] == "partially_filled")
    ratio = (filled_sum / plan_sum) if plan_sum > 0 else 0.0

    # ⚠️ 两种「还没到执行」的情况，都不该报缺口：
    #   ① 计划日期晚于今天（前一晚生成、等明天执行）
    #   ② 计划日期就是今天、但还没开盘（凌晨跑）——开盘后才谈得上成交
    today = now.strftime("%Y%m%d")
    pre_open = now.hour < 9 or (now.hour == 9 and now.minute < 30)
    pending = bool(key) and (key > today or (key == today and pre_open))
    healthy = pending or ratio >= HEALTHY_RATIO

    unmapped_share, unmapped_codes = unmapped_weight(plan_w, ddir)

    # 这份计划吃的是哪天的选股结果 —— 与「建满度」并列，因为「用旧票建满仓」
    # 和「用好票没建满」都是实盘跑不过回测的独立原因，且都不报错。
    # 滞后 >=2 个交易日 = 中间有选股作业没跑成（计划文件本身看不出来）。
    picks_lag = int(plan.get("picks_lag_days") or 0)

    return {
        "at": now.isoformat(timespec="seconds"),
        "plan_date": key,
        "pending": pending,
        "plan_source": plan.get("source") or "",
        "plan_regime": plan.get("regime") or "",
        "picks_date": plan.get("picks_date") or "",
        "picks_lag_days": picks_lag,
        "picks_stale": picks_lag >= 2,
        "n_planned": len(targets),
        "weight_plan_total": round(plan_sum, 4),
        "weight_filled_total": round(filled_sum, 4),
        "fill_ratio": round(ratio, 4),
        "healthy": healthy,
        "n_missing": n_missing,
        "n_partially_filled": n_partial,
        "unmapped_weight_pct": round(unmapped_share, 4),
        "unmapped_codes": unmapped_codes,
        "cash_pct": round((snap["cash"] / total) if total > 0 else 0.0, 4),
        "equity": round(total, 2),
        "positions": rows,
    }


def retry_hint(rep: dict) -> str:
    """「未建满会不会被补上」—— 只有这里能把「缺口」和「还有没有下一次机会」连起来。

    ⚠️ 缺口的**大小**与缺口的**后果**是两件事：11-02 那次建满度 65%，看起来只是
    「少买三只」，实际后果是「剩下 41% 现金要等 15 个交易日」。审计工具如果只报
    百分比，读的人根本不知道这轮调仓已经结束了。
    """
    if rep.get("pending"):
        return ""
    try:
        from config.settings import app_config

        tl = getattr(app_config, "trading_loop", None)
        enabled = bool(getattr(tl, "rebalance_retry_enabled", False))
        min_ratio = float(getattr(tl, "rebalance_retry_min_fill_ratio", 0.85) or 0.85)
        max_days = int(getattr(tl, "rebalance_retry_max_days", 2) or 0)
    except Exception:  # noqa: BLE001
        return ""
    if not (0.0 < min_ratio <= 1.0):
        min_ratio = 1.0
    ratio = float(rep.get("fill_ratio") or 0.0)
    if ratio >= min_ratio:
        return ""
    if enabled:
        return (f"↩️ 建满度 {ratio:.0%} < 顺延阈值 {min_ratio:.0%}：顺延重试**已开启**，"
                f"若被拦的量还能补，loop 会在下一个交易日重试（最多 {max_days} 个交易日）")
    return (f"↩️ 建满度 {ratio:.0%} < 顺延阈值 {min_ratio:.0%}：顺延重试**未开启**"
            f"（rebalance_retry_enabled=false）—— 这轮调仓到此为止，缺口要等满 "
            f"rebalance_days 个交易日才有下一次机会。11-02 就是这种形态。")


def render(rep: dict) -> str:
    lines = []
    lines.append(f"计划日期 {rep['plan_date']}（{rep['plan_source']}，regime={rep['plan_regime']}）")
    lines.append(f"权益 ¥{rep['equity']:,.0f}   现金占比 {rep['cash_pct']:.1%}")
    if rep.get("picks_stale"):
        lines.append(f"🗞 选股结果陈旧：报告 {rep.get('picks_date') or '?'} 距今 "
                     f"{rep.get('picks_lag_days')} 个交易日（正常应为 1）"
                     f" —— 计划吃的是旧票，先查当日选股作业有没有跑成")
    elif rep.get("picks_date"):
        lines.append(f"🗞 选股报告 {rep['picks_date']}（滞后 {rep.get('picks_lag_days') or 0} 个交易日）")
    share = rep.get("unmapped_weight_pct") or 0.0
    if share > 0:
        flag = "⚠️ 集中度上限对该部分免检" if share >= 0.30 else "（集中度上限对其免检）"
        lines.append(f"🏭 无行业映射：{share:.1%} 的计划权重、{len(rep.get('unmapped_codes') or [])} 只 "
                     f"{flag} —— 数据质量问题，修 industry_map 而不是调阈值")
    if rep.get("pending"):
        lines.append(f"⏳ 该计划尚未开始执行（未到执行日 / 今天还没开盘）："
                     f"计划权重 {rep['weight_plan_total']:.1%}，当前建满 {rep['weight_filled_total']:.1%}"
                     f" —— 属于正常，不是缺口")
    else:
        lines.append(f"计划权重合计 {rep['weight_plan_total']:.1%} → 实际建满 {rep['weight_filled_total']:.1%}"
                     f"   建满度 {rep['fill_ratio']:.1%}"
                     + ("   ✅ 健康" if rep["healthy"] else "   ⚠️ 偏空"))
        if rep["n_missing"] or rep["n_partially_filled"]:
            lines.append(f"未建满：完全没买 {rep['n_missing']} 只 / 只买到一部分 {rep['n_partially_filled']} 只")
        hint = retry_hint(rep)
        if hint:
            lines.append(hint)
    lines.append("")
    lines.append(f"{'代码':<8}{'名称':<10}{'计划':>7}{'实际':>7}{'差额':>7}  状态 / 拦截原因")
    for r in rep["positions"]:
        name = (r["name"] or "")[:8]
        tag = {"ok": "✅", "partially_filled": "◐", "missing": "❌"}[r["status"]]
        why = r["blocked_reason"][:44] if r["blocked_reason"] else ""
        lines.append(f"{r['code']:<8}{name:<10}{r['weight_plan']:>6.2%}{r['weight_actual']:>7.2%}"
                     f"{r['gap']:>7.2%}  {tag} {why}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="计划 vs 实际成交审计（仓位建满度）")
    ap.add_argument("--date", default=None, help="计划日期 yyyymmdd（默认取最新一份）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--save", action="store_true", help="另存 data/fill_audit.json")
    args = ap.parse_args()

    rep = audit(args.date)
    if not rep["plan_date"]:
        print("找不到任何交易计划（data/trade_plans/ 为空）")
        return 1
    if args.save:
        (DATA / "fill_audit.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2) if args.json else render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""纸盘影子期验收脚本

用途：在把「选股神 AI」自动交易从纸盘切到实盘之前，做一轮影子期验收，
回答三个核心问题：

  1. 收益验收  —— 纸盘已实现收益，对比生产模型承诺的预期超额
                 （production_metrics.json: Top20 每期超额 top_excess）
  2. 信号质量  —— 模型在「实际被选中的标的」上是否仍有效：
                 用 entry ai_score 与 已实现 pnl_pct 算 Spearman（已实现 RankIC），
                 并对最近 N 笔做滚动窗口监控（<=0 即告警）
  3. 风控熔断  —— 扫描历史成交，统计日内亏损熔断 / 日成交上限 / 单票仓位 /
                 止损(-5%) / 止盈(+15%) 的触发次数

纯离线可用（只依赖本地 paper_trade_history.json / production_metrics.json /
RiskConfig），不联网、不依赖券商。另有可选 ``--live-ic`` 用本地缓存 parquet
做市场级单日 RankIC 并累积成滚动日志（ZZ500 成分股需联网获取，失败则退化为
本地缓存样本）。

用法：
    python scripts/paper_shadow_audit.py                 # 完整离线验收
    python scripts/paper_shadow_audit.py --live-ic       # 追加市场级滚动 IC
    python scripts/paper_shadow_audit.py --json out.json # 指定 JSON 输出
    python scripts/paper_shadow_audit.py --history H --account A --metrics M
    python scripts/paper_shadow_audit.py --rolling 30    # 滚动窗口笔数
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 把仓库根加入 sys.path，保证 ``from src...`` 可用
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- 路径常量（与 paper.py 保持一致，避免硬编码漂移） ----
try:
    from src.trading.paper import PAPER_HISTORY_FILE, PAPER_ACCOUNT_FILE
except Exception:  # noqa: BLE001
    _DATA = ROOT / "data"
    PAPER_HISTORY_FILE = _DATA / "paper_trade_history.json"
    PAPER_ACCOUNT_FILE = _DATA / "paper_account.json"

MODELS_DIR = ROOT / "models"
PROD_METRICS_FILE = MODELS_DIR / "production_metrics.json"
ROLLING_IC_LOG = ROOT / "data" / "rolling_ic_log.json"
TRIGGER_LOG = ROOT / "data" / "audit_triggers.jsonl"

# ---- RiskConfig 真阈值（优先从 executor 取，失败退化为文档默认值） ----
try:
    from src.trading.executor import RiskConfig
    _RC = RiskConfig()
    RISK = {
        "max_position_pct": _RC.max_position_pct,
        "stop_loss_pct": _RC.stop_loss_pct,
        "stop_profit_pct": _RC.stop_profit_pct,
        "trailing_stop_pct": _RC.trailing_stop_pct,
        "max_daily_trades": _RC.max_daily_trades,
        "max_daily_loss": _RC.max_daily_loss,
        "max_daily_loss_pct": _RC.max_daily_loss_pct,
        "source": "RiskConfig(executor)",
    }
except Exception as e:  # noqa: BLE001
    RISK = {
        "max_position_pct": 0.30,
        "stop_loss_pct": -5.0,
        "stop_profit_pct": 15.0,
        "trailing_stop_pct": -3.0,
        "max_daily_trades": 20,
        "max_daily_loss": 5000.0,
        "max_daily_loss_pct": -3.0,
        "source": f"builtin-fallback({e})",
    }


# ============================ 工具函数 ============================

def _load_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:19], fmt)
            except Exception:
                continue
    return None


def _rank(a: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）"""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    uniq = np.unique(a)
    if len(uniq) == len(a):
        return ranks
    out = np.empty(len(a))
    for v in uniq:
        idx = np.where(a == v)[0]
        out[idx] = ranks[idx].mean()
    return out


def _spearman(x, y) -> float:
    """Spearman 秩相关（无 scipy 依赖）"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _count_trading_days(start: date, end: date) -> int:
    """区间内交易日数（含端点）"""
    try:
        from src.trading.calendar import is_trading_day
    except Exception:
        is_trading_day = lambda d: d.weekday() < 5  # 降级：仅工作日
    cnt = 0
    d = start
    guard = 0
    while d <= end and guard < 400:
        if is_trading_day(d):
            cnt += 1
        d = d + __import__("datetime").timedelta(days=1)
        guard += 1
    return cnt


def _read_trigger_log(path: Path = TRIGGER_LOG) -> List[dict]:
    """读取 audit_triggers.jsonl（AutoTrader 风控触发结构化日志）"""
    out: List[dict] = []
    try:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:  # noqa: BLE001
        pass
    return out


def _trigger_summary(log: List[dict]) -> dict:
    counts: Dict[str, int] = {}
    for r in log:
        t = r.get("type", "?")
        counts[t] = counts.get(t, 0) + 1
    return {"total": len(log), "by_type": counts,
            "recent": log[-5:] if log else []}


# ============================ 成交解析 ============================

def _load_history(path: Path) -> List[dict]:
    obj = _load_json(path)
    if not obj:
        return []
    if isinstance(obj, dict):  # 容错：某些存档可能是 {trades:[...]}
        obj = obj.get("trades", [])
    return [r for r in obj if isinstance(r, dict)]


def fifo_close_lots(history: List[dict]) -> List[dict]:
    """把买/卖记录按 code FIFO 配对成整手平仓，返回已实现样本。

    每个样本: code, name, entry_ai_score, entry_price, close_price,
              qty, pnl_pct(用成交填充价算，已含滑点), entry_date, close_date
    """
    trades = sorted(
        (t for t in history if t.get("side") in ("buy", "sell")),
        key=lambda t: _parse_dt(t.get("filled_at") or "") or datetime.min,
    )
    open_lots: Dict[str, List[dict]] = {}
    closed: List[dict] = []

    for t in trades:
        code = str(t.get("code", ""))
        side = t.get("side")
        qty = int(float(t.get("quantity", 0) or 0))
        if qty <= 0:
            continue
        dt = _parse_dt(t.get("filled_at") or "")
        if side == "buy":
            open_lots.setdefault(code, []).append({
                "qty": qty,
                "entry_price": float(t.get("price", 0) or 0),
                "entry_ai_score": float(t.get("ai_score", 0) or 0),
                "entry_date": dt,
                "name": t.get("name", code),
            })
        else:  # sell
            remaining = qty
            price = float(t.get("price", 0) or 0)
            for lot in list(open_lots.get(code, [])):
                if remaining <= 0:
                    break
                take = min(lot["qty"], remaining)
                if lot["entry_price"] > 0 and price > 0:
                    pnl_pct = (price - lot["entry_price"]) / lot["entry_price"] * 100.0
                else:
                    pnl_pct = 0.0
                closed.append({
                    "code": code,
                    "name": lot.get("name", code),
                    "entry_ai_score": lot["entry_ai_score"],
                    "entry_price": lot["entry_price"],
                    "close_price": price,
                    "qty": take,
                    "pnl_pct": pnl_pct,
                    "entry_date": lot["entry_date"],
                    "close_date": dt,
                })
                lot["qty"] -= take
                remaining -= take
                if lot["qty"] <= 0:
                    open_lots[code].remove(lot)
            # 卖多于买（手动平仓/分红等）的剩余部分忽略，不计入配对
    return closed


# ============================ Panel 1: 收益验收 ============================

def panel_pnl(history: List[dict], account: Optional[dict], metrics: dict) -> dict:
    wf = (metrics or {}).get("walk_forward", {})
    top_excess = float(wf.get("top_excess", 0.0) or 0.0)      # 每期 Top20 超额（小数）
    top_win_rate = float(wf.get("top_win_rate", 0.0) or 0.0)
    top_n = int((metrics or {}).get("config", {}).get("top_n", 20) or 20)
    horizon = int((metrics or {}).get("config", {}).get("horizon", 5) or 5)

    initial_cash = float((account or {}).get("initial_cash", 100000.0) or 100000.0)

    sells = [t for t in history if t.get("side") == "sell" and float(t.get("pnl", 0) or 0) != 0]
    realized_pnl = sum(float(t.get("pnl", 0) or 0) for t in sells)

    # 未实现（用 account 当前持仓的 last_price vs avg_cost）
    unreal = 0.0
    cash = initial_cash
    mv_total = 0.0
    pos_breach = []
    if account:
        cash = float(account.get("cash", initial_cash))
        holdings = account.get("holdings", {}) or {}
        for code, h in holdings.items():
            q = float(h.get("quantity", 0) or 0)
            avg = float(h.get("avg_cost", 0) or 0)
            lp = float(h.get("last_price", avg) or avg)
            mv = q * lp
            mv_total += mv
            unreal += (lp - avg) * q
        total_asset = cash + mv_total
        # 单票仓位检查
        for code, h in holdings.items():
            q = float(h.get("quantity", 0) or 0)
            lp = float(h.get("last_price", 0) or 0)
            mv = q * lp
            if total_asset > 0 and mv / total_asset > RISK["max_position_pct"]:
                pos_breach.append({"code": code, "pct": round(mv / total_asset * 100, 2)})
    else:
        total_asset = initial_cash + realized_pnl + unreal

    total_pnl = total_asset - initial_cash
    realized_return_pct = realized_pnl / initial_cash * 100.0 if initial_cash else 0.0
    total_return_pct = total_pnl / initial_cash * 100.0 if initial_cash else 0.0

    # 期数：首个成交日到今天（或最后一个成交日）的交易日数
    trade_dates = [_parse_dt(t.get("filled_at") or "") for t in history]
    trade_dates = [d for d in trade_dates if d]
    if trade_dates:
        first = min(trade_dates).date()
        last = max(trade_dates).date()
        periods = _count_trading_days(first, last)
        periods_to_today = _count_trading_days(first, date.today())
    else:
        first = last = None
        periods = periods_to_today = 0

    # 预期累计超额（用 Top20 每期超额折算）
    expected_cum_excess_pct = periods_to_today * top_excess * 100.0
    # 你的策略每期实现多少（用已实现收益 / 期数）
    realized_per_period_pct = (realized_return_pct / periods) if periods else float("nan")
    gap_pct = realized_return_pct - expected_cum_excess_pct

    # 胜率 / 利润因子（已实现平仓）
    closed = fifo_close_lots(history)
    wins = [c for c in closed if c["pnl_pct"] > 0]
    losses = [c for c in closed if c["pnl_pct"] <= 0]
    win_rate = len(wins) / len(closed) if closed else 0.0
    gross_profit = sum(c["pnl_pct"] for c in wins)
    gross_loss = abs(sum(c["pnl_pct"] for c in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "initial_cash": round(initial_cash, 2),
        "total_asset": round(total_asset, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unreal, 2),
        "total_pnl": round(total_pnl, 2),
        "realized_return_pct": round(realized_return_pct, 3),
        "total_return_pct": round(total_return_pct, 3),
        "first_trade": first.isoformat() if first else None,
        "last_trade": last.isoformat() if last else None,
        "periods_traded": periods,
        "periods_to_today": periods_to_today,
        "expected_top_excess_per_period_pct": round(top_excess * 100, 4),
        "expected_cum_excess_pct": round(expected_cum_excess_pct, 3),
        "realized_per_period_pct": round(realized_per_period_pct, 3) if periods else None,
        "gap_vs_expected_pct": round(gap_pct, 3),
        "win_rate_realized": round(win_rate, 3),
        "win_rate_expected_top": round(top_win_rate, 3),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999,
        "closed_lots": len(closed),
        "top_n": top_n,
        "horizon": horizon,
        "single_position_breach": pos_breach,
    }


# ============================ Panel 2: 信号质量 / 滚动 IC ============================

def panel_signal(history: List[dict], rolling: int) -> dict:
    closed = fifo_close_lots(history)
    # 仅保留有 entry 评分的样本（评分=0 视为无效，不参与排序）
    scored = [c for c in closed if c["entry_ai_score"] > 0]
    if not scored:
        return {
            "closed_lots": len(closed),
            "scored_lots": 0,
            "overall_ic": None,
            "rolling_ic": None,
            "rolling_window": rolling,
            "note": "无足够带评分的平仓样本（需 autotrader 写入 ai_score）",
        }

    xs = [c["entry_ai_score"] for c in scored]
    ys = [c["pnl_pct"] for c in scored]
    overall = _spearman(xs, ys)

    # 滚动窗口（按平仓时间倒序取最近 rolling 笔）
    ordered = sorted(scored, key=lambda c: c["close_date"] or datetime.min)
    win = ordered[-rolling:] if len(ordered) >= rolling else ordered
    rolling_ic = _spearman([c["entry_ai_score"] for c in win],
                           [c["pnl_pct"] for c in win]) if len(win) >= 3 else float("nan")

    # 最近 30 自然日窗口
    cutoff = datetime.now() - __import__("datetime").timedelta(days=30)
    recent = [c for c in ordered if (c["close_date"] or datetime.min) >= cutoff]
    recent_ic = _spearman([c["entry_ai_score"] for c in recent],
                          [c["pnl_pct"] for c in recent]) if len(recent) >= 3 else float("nan")

    return {
        "closed_lots": len(closed),
        "scored_lots": len(scored),
        "overall_ic": round(overall, 4) if overall == overall else None,
        "rolling_ic": round(rolling_ic, 4) if rolling_ic == rolling_ic else None,
        "rolling_window": rolling,
        "recent30d_ic": round(recent_ic, 4) if recent_ic == recent_ic else None,
        "recent30d_samples": len(recent),
        "verdict": _ic_verdict(overall, rolling_ic, len(scored)),
    }


def _ic_verdict(overall, rolling, n) -> str:
    if overall != overall or n < 10:
        return "样本不足（需 >=10 笔带评分平仓）"
    if overall <= 0:
        return "🔴 信号失效：整体已实现 IC<=0（高分未带来高收益）"
    if rolling == rolling and rolling <= 0:
        return "🟡 滚动 IC<=0：近期信号走弱，需观察"
    if overall < 0.05:
        return "🟡 信号偏弱：IC>0 但偏低（<0.05）"
    return "🟢 信号有效：IC>0 且健康"


# ============================ Panel 3: 风控熔断统计 ============================

def panel_risk(history: List[dict], account: Optional[dict]) -> dict:
    initial_cash = float((account or {}).get("initial_cash", 100000.0) or 100000.0)
    daily_loss_cap = RISK["max_daily_loss"]
    daily_loss_pct = RISK["max_daily_loss_pct"] / 100.0
    daily_trade_cap = RISK["max_daily_trades"]
    stop_loss = RISK["stop_loss_pct"]
    stop_profit = RISK["stop_profit_pct"]

    # 按交易日聚合
    by_day: Dict[str, dict] = {}
    for t in history:
        dt = _parse_dt(t.get("filled_at") or "")
        if not dt:
            continue
        key = dt.strftime("%Y-%m-%d")
        d = by_day.setdefault(key, {"pnl": 0.0, "trades": 0, "buys": 0, "sells": 0})
        d["trades"] += 1
        if t.get("side") == "buy":
            d["buys"] += 1
        elif t.get("side") == "sell":
            d["sells"] += 1
            d["pnl"] += float(t.get("pnl", 0) or 0)

    breach_days = []
    trade_cap_days = []
    for day, d in sorted(by_day.items()):
        # 日内亏损熔断：以当日已实现卖出盈亏近似（保守：未含盘中未实现回撤）
        if d["pnl"] <= -daily_loss_cap or d["pnl"] <= daily_loss_pct * initial_cash:
            breach_days.append({"date": day, "realized_pnl": round(d["pnl"], 2)})
        if d["trades"] > daily_trade_cap:
            trade_cap_days.append({"date": day, "trades": d["trades"]})

    # 止损 / 止盈触发（在平仓样本层面统计）
    closed = fifo_close_lots(history)
    sl_hits = [c for c in closed if c["pnl_pct"] <= stop_loss]
    tp_hits = [c for c in closed if c["pnl_pct"] >= stop_profit]

    return {
        "thresholds": {
            "max_daily_loss": daily_loss_cap,
            "max_daily_loss_pct": RISK["max_daily_loss_pct"],
            "max_daily_trades": daily_trade_cap,
            "stop_loss_pct": stop_loss,
            "stop_profit_pct": stop_profit,
            "max_position_pct": RISK["max_position_pct"],
            "source": RISK["source"],
        },
        "days_traded": len(by_day),
        "daily_loss_breach_days": breach_days,
        "daily_loss_breach_count": len(breach_days),
        "daily_trade_cap_days": trade_cap_days,
        "daily_trade_cap_breach_count": len(trade_cap_days),
        "stop_loss_trigger_lots": len(sl_hits),
        "stop_profit_trigger_lots": len(tp_hits),
        "single_position_breach": (account and [
            {"code": c, "pct": p} for c, p in []
        ]) or [],
        "note": "日内亏损按当日已实现卖出盈亏近似，未含盘中未实现回撤（实盘会更严）",
    }


# ============================ 汇总与打印 ============================

def _verdict_pnl(p: dict) -> str:
    if p["periods_traded"] == 0:
        return "样本不足（尚无成交）"
    if p["total_return_pct"] < 0:
        return "🟡 累计收益为负"
    if p["gap_vs_expected_pct"] < -1.0:
        return "🟡 实际收益显著低于模型预期"
    return "🟢 收益达成/超预期"


def run_audit(args) -> dict:
    history = _load_history(args.history)
    account = _load_json(args.account)
    metrics = _load_json(args.metrics) or {}

    if not history:
        report = {
            "generated_at": datetime.now().isoformat(),
            "status": "NO_DATA",
            "message": "纸盘影子期尚未开始：paper_trade_history.json 无成交记录。"
                       " 先让自动交易（纸盘模式）跑一段时间再来验收。",
        }
        return report

    pnl = panel_pnl(history, account, metrics)
    sig = panel_signal(history, args.rolling)
    risk = panel_risk(history, account)
    risk["structured_triggers"] = _trigger_summary(_read_trigger_log(args.trigger_log))

    # 全局验收判定
    critical = []
    if sig.get("overall_ic") is not None and sig["scored_lots"] >= 20 and sig["overall_ic"] <= 0:
        critical.append("信号失效：整体已实现 RankIC<=0")
    if risk["daily_loss_breach_count"] >= 1:
        critical.append(f"触发日内亏损熔断 {risk['daily_loss_breach_count']} 天")
    warn = []
    if pnl["total_return_pct"] < 0:
        warn.append("累计收益为负")
    if pnl["win_rate_realized"] and pnl["win_rate_expected_top"] and \
       pnl["win_rate_realized"] < pnl["win_rate_expected_top"] - 0.05:
        warn.append(f"胜率 {pnl['win_rate_realized']:.1%} 低于模型预期 {pnl['win_rate_expected_top']:.1%}")

    status = "FAIL" if critical else ("WARN" if warn else "PASS")

    report = {
        "generated_at": datetime.now().isoformat(),
        "status": status,
        "critical": critical,
        "warn": warn,
        "panel_pnl": pnl,
        "panel_signal": sig,
        "panel_risk": risk,
    }
    return report


def _print(report: dict):
    if report.get("status") == "NO_DATA":
        print("\n" + "=" * 64)
        print("  📋 纸盘影子期验收")
        print("=" * 64)
        print("  " + report["message"])
        print("=" * 64 + "\n")
        return

    p, s, r = report["panel_pnl"], report["panel_signal"], report["panel_risk"]
    print("\n" + "=" * 64)
    print(f"  📋 纸盘影子期验收报告  [{report['status']}]")
    print(f"  生成时间: {report['generated_at']}")
    print("=" * 64)

    print("\n【1】收益验收  " + _verdict_pnl(p))
    print(f"  期初资金        : ¥{p['initial_cash']:,.0f}")
    print(f"  当前总资产      : ¥{p['total_asset']:,.0f}")
    print(f"  累计盈亏        : ¥{p['total_pnl']:,.2f} ({p['total_return_pct']:+.2f}%)")
    print(f"  已实现 / 未实现 : ¥{p['realized_pnl']:,.2f} / ¥{p['unrealized_pnl']:,.2f}")
    if p["periods_traded"]:
        print(f"  交易期数        : {p['periods_traded']}（至今天数 {p['periods_to_today']}）")
        print(f"  模型预期/期超额 : {p['expected_top_excess_per_period_pct']:.4f}%  → 累计预期 {p['expected_cum_excess_pct']:+.2f}%")
        if p['realized_per_period_pct'] is not None:
            print(f"  你每期实现      : {p['realized_per_period_pct']}%")
        print(f"  与预期差距      : {p['gap_vs_expected_pct']:+.2f} pct")
    print(f"  胜率(已实现)    : {p['win_rate_realized']:.1%}  (模型 Top{int(p['top_n'])} 预期 {p['win_rate_expected_top']:.1%})")
    print(f"  利润因子        : {p['profit_factor']}")
    if p["single_position_breach"]:
        print(f"  ⚠️ 单票仓位超限  : {p['single_position_breach']}")

    print("\n【2】信号质量 / 滚动 IC  " + s.get("verdict", ""))
    print(f"  平仓样本        : {s['closed_lots']}（带评分 {s['scored_lots']}）")
    if s.get("overall_ic") is not None:
        print(f"  整体已实现 IC   : {s['overall_ic']:+.4f}  (entry评分 vs 实现收益)")
        print(f"  滚动 IC(近{s['rolling_window']}笔): {s['rolling_ic']}")
        print(f"  近30日 IC       : {s['recent30d_ic']}  (样本 {s['recent30d_samples']})")
    else:
        print("  " + s.get("note", "无样本"))

    print("\n【3】风控熔断统计")
    th = r["thresholds"]
    print(f"  阈值: 日亏≤¥{th['max_daily_loss']:.0f} 或 ≤{th['max_daily_loss_pct']}% | "
          f"日成交≤{th['max_daily_trades']} | 单票≤{th['max_position_pct']:.0%} | "
          f"止损{th['stop_loss_pct']}% / 止盈{th['stop_profit_pct']}%")
    print(f"  交易天数        : {r['days_traded']}")
    print(f"  日内亏损熔断    : {r['daily_loss_breach_count']} 天 {r['daily_loss_breach_days']}")
    print(f"  日成交上限突破  : {r['daily_trade_cap_breach_count']} 天 {r['daily_trade_cap_days']}")
    print(f"  止损触发(平仓)  : {r['stop_loss_trigger_lots']} 笔 | 止盈触发: {r['stop_profit_trigger_lots']} 笔")
    st = r.get("structured_triggers")
    if st and st["total"]:
        print(f"  结构化触发日志  : 共 {st['total']} 条 {st['by_type']}")
        for rec in st["recent"]:
            print(f"      · {rec.get('ts','')[:19]} [{rec.get('type','')}] "
                  f"{rec.get('code','')} {rec.get('reason','')}")
    else:
        print(f"  结构化触发日志  : 无（data/audit_triggers.jsonl 为空或不存在）")
    print(f"  ⓘ {r['note']}")

    print("\n【判定】")
    if report["critical"]:
        print("  🔴 CRITICAL: " + "；".join(report["critical"]))
    if report["warn"]:
        print("  🟡 WARN: " + "；".join(report["warn"]))
    if not report["critical"] and not report["warn"]:
        print("  🟢 全部通过，可进入下一阶段（建议先 1~2 成仓实盘 + 继续监控）")
    print("=" * 64 + "\n")


# ============================ 可选: 市场级滚动 IC ============================

def cmd_live_ic(args) -> int:
    """用本地缓存 parquet 对股票池做市场级单日 RankIC，并累积滚动日志。

    返回 0（无论是否算出，失败仅告警，不阻断主验收）。
    """
    try:
        import pandas as pd  # noqa: F401
        from src.models.predictor import predictor
        from src.data.fetcher import fetcher
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ live-ic 需要模型/数据模块，当前不可用: {e}")
        return 0

    # 股票池：优先 zz500（需联网），失败退化为本地缓存样本
    try:
        codes = list(fetcher.get_stock_pool("zz500"))
        offline = False
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 获取 zz500 成分股失败（需联网）: {e}；改用本地缓存样本(前{args.sample}只)")
        codes = [p.name.split("_")[1] for p in sorted((ROOT / "data").glob("daily_*_qfq.parquet"))][:args.sample]
        offline = True
    if not codes:
        print("⚠️ 无可用股票池，跳过 live-ic")
        return 0

    # 取样本内所有股票共同的最后一个交易日作为评分日 d
    from src.data.fetcher import fetcher as _fk
    last_dates = {}
    closes = {}
    for code in codes:
        try:
            df = _fk.get_daily_kline(code)
            if df is None or len(df) < 60:
                continue
            dcol = pd.to_datetime(df["date"])
            df = df.assign(_d=dcol).set_index("_d").sort_index()
            last_dates[code] = df.index[-1]
            closes[code] = df["close"]
        except Exception:
            continue
    if not last_dates:
        print("⚠️ 无可用行情数据，跳过 live-ic")
        return 0

    from src.trading.calendar import next_trading_day
    horizon = int((args.horizon if hasattr(args, "horizon") else 5))

    # 评分日取「最新可评分日」：需满足其后 horizon 个交易日的前向收益已可实现。
    # 缓存最新日的前向窗口往往落在未来，若直接取最新日则永远算不出 RankIC；
    # 这里回看，使 d 与 d+horizon 都落在已缓存行情内，从而能用已实现前向收益计算 RankIC。
    def _latest_score_date(ser):
        idx = set(ser.index)
        dates = list(ser.index)
        for i in range(len(dates) - 1, -1, -1):
            t = dates[i]
            t5 = t
            ok = True
            for _ in range(horizon):
                t5 = pd.Timestamp(next_trading_day(t5, include_today=False))
                if t5 not in idx:
                    ok = False
                    break
            if ok:
                return t
        return None

    latest = {c: _latest_score_date(ser) for c, ser in closes.items()}
    latest = {c: d0 for c, d0 in latest.items() if d0 is not None}
    if not latest:
        print("⚠️ 无可用评分日（数据不足以覆盖前向窗口），跳过 live-ic")
        return 0
    # 取所有股票均可评分的最近日期（个别不含该日的股票会在后续循环被跳过）
    d = min(latest.values())
    d5 = d
    for _ in range(horizon):
        d5 = pd.Timestamp(next_trading_day(d5, include_today=False))

    scores, fwd = [], []
    for code in codes:
        if code not in closes:
            continue
        ser = closes[code]
        if d not in ser.index or d5 not in ser.index:
            continue
        try:
            df_code = _fk.get_daily_kline(code)
            dcol = pd.to_datetime(df_code["date"])
            df_code = df_code.assign(_d=dcol).set_index("_d").sort_index()
            df_d = df_code.loc[:d]
            if len(df_d) < 60:
                continue
            pred = predictor.predict_single(df_d, code)
            c0, c1 = float(ser.loc[d]), float(ser.loc[d5])
            if c0 <= 0:
                continue
            ret = c1 / c0 - 1.0
            scores.append(pred.score)
            fwd.append(ret)
        except Exception:
            continue

    if len(scores) < 10:
        print(f"⚠️ 有效评分样本不足({len(scores)})，跳过 live-ic")
        return 0

    fwd_arr = np.asarray(fwd)
    excess = fwd_arr - fwd_arr.mean()  # 相对等权宇宙的超额（与训练标签口径一致）
    ic = _spearman(scores, excess)

    # 累积滚动日志（按评分日去重：非交易日/同日重复跑不追加，避免污染滚动序列）
    log = _load_json(ROLLING_IC_LOG) or []
    d_str = d.strftime("%Y-%m-%d")
    if log and log[-1].get("date") == d_str:
        appended = False
    else:
        log.append({"date": d_str, "ic": round(float(ic), 4),
                    "n": len(scores), "offline": offline})
        log = log[-60:]
        try:
            ROLLING_IC_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        appended = True

    ics = [x["ic"] for x in log if x["ic"] == x["ic"]]
    roll_mean = float(np.mean(ics[-args.rolling:])) if len(ics) >= args.rolling else float(np.mean(ics))
    print(f"\n📡 市场级 RankIC（评分日 {d.strftime('%Y-%m-%d')}，样本 {len(scores)}，"
          f"{'离线样本' if offline else 'zz500'}）")
    print(f"  今日 IC      : {ic:+.4f}")
    print(f"  滚动{args.rolling}日均值: {roll_mean:+.4f}  (日志共 {len(ics)} 点，{'已追加' if appended else '今日已记录·跳过追加'})")
    print(f"  生产基线参考 : RankIC={_load_json(PROD_METRICS_FILE).get('walk_forward', {}).get('rank_ic')}")
    if ic <= 0:
        print("  🔴 今日市场级 IC<=0：模型全市场排序能力暂失，需警惕")
    elif roll_mean <= 0:
        print("  🟡 滚动 IC<=0：近期走弱")
    else:
        print("  🟢 市场级排序能力正常")
    return 0


# ============================ 入口 ============================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="纸盘影子期验收脚本")
    ap.add_argument("--history", type=Path, default=PAPER_HISTORY_FILE, help="纸盘成交历史 JSON")
    ap.add_argument("--account", type=Path, default=PAPER_ACCOUNT_FILE, help="纸盘账户 JSON")
    ap.add_argument("--metrics", type=Path, default=PROD_METRICS_FILE, help="生产指标 JSON")
    ap.add_argument("--trigger-log", type=Path, default=TRIGGER_LOG, help="风控触发日志 JSONL")
    ap.add_argument("--json", type=Path, default=ROOT / "data" / "paper_shadow_audit.json", help="JSON 输出")
    ap.add_argument("--rolling", type=int, default=20, help="Panel2 滚动窗口笔数")
    ap.add_argument("--live-ic", action="store_true", help="追加市场级滚动 IC（需模型+本地缓存）")
    ap.add_argument("--sample", type=int, default=200, help="live-ic 离线样本数")
    args = ap.parse_args(argv)

    report = run_audit(args)
    _print(report)
    try:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        report["inputs"] = {
            "history": str(args.history), "account": str(args.account),
            "metrics": str(args.metrics),
        }
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📄 JSON 已写入: {args.json}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ JSON 写出失败: {e}")

    if args.live_ic:
        cmd_live_ic(args)

    # 退出码：FAIL=1，其余 0（CI 可据此拦截）
    return 1 if report.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

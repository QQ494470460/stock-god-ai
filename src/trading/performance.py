# -*- coding: utf-8 -*-
"""业绩与归因引擎（Performance & Attribution）
================================================

系统此前只会「做事」，不会「结账」。

结算报告是**分散的**：`data/loop_reports/loop_report_YYYYMMDD.json` 一天一个文件，
各自只回答「今天多少」。没有任何地方回答三个真正决定这套系统该不该继续跑下去
的问题：

1. **它到底赚不赚钱？** —— 需要净值曲线、年化、夏普、最大回撤，而不是一堆日盈亏。
2. **钱是怎么赚/亏的？** —— 需要按策略、按行业、按个股拆解贡献，否则没法知道
   该保留哪条腿、砍掉哪条腿。
3. **风险是不是在偷偷变大？** —— 需要滚动夏普、回撤持续期、盈亏比漂移。

本模块把 `loop_reports/`（每日净值快照）+ `paper_trade_history.json`（成交流水）
汇成一份可判断的业绩档案，落盘 `data/performance.json` 供看板/日报/进化门禁读取。

设计纪律
--------
- **纯计算、零网络、零副作用**：只读文件、只写一份 JSON。
- **fail-safe**：任何缺失/损坏的数据源都退化为「少算一项」，绝不抛错。
  业绩统计挂掉不该影响交易主流程。
- **口径必须自证**：`equity_ok=False` 的快照（账户取不到时的占位值）直接剔除，
  否则一条占位点就能把最大回撤算成 -100%。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

from config.settings import DATA_DIR

PERF_FILE = Path(DATA_DIR) / "performance.json"
REPORTS_DIR = Path(DATA_DIR) / "loop_reports"
PAPER_HISTORY = Path(DATA_DIR) / "paper_trade_history.json"
INDUSTRY_MAP_FILE = Path(DATA_DIR) / "industry_map.json"
TRADING_DAYS_PER_YEAR = 242


# ======================================================================
# 数据装载
# ======================================================================

def _read_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"业绩引擎读取失败 {path.name}: {e}")
    return None


def load_equity_curve(
    reports_dir: Optional[Path] = None,
    account_file: Optional[Path] = None,
) -> List[dict]:
    """从结算报告装配净值曲线（按日期升序）。

    剔除规则（每条都对应一次真实事故）：
    - 文件名不是 ``loop_report_YYYYMMDD`` → 跳过（.md、临时文件）；
    - ``equity_ok is False`` → 剔除：账户快照取不到时 equity 是占位值，
      一条就能把回撤算成 -100%；
    - ``equity <= 0`` → 剔除；
    - 同一日期出现多份 → 保留 ``settled_at`` 最新的那份。

    报告不足时退化为「纸账户单点」，让看板至少能显示当前权益而不是空白。
    """
    reports_dir = Path(reports_dir or REPORTS_DIR)
    by_date: Dict[str, dict] = {}
    try:
        files = sorted(reports_dir.glob("loop_report_*.json"))
    except Exception:  # noqa: BLE001
        files = []
    for f in files:
        snap = _read_json(f)
        if not isinstance(snap, dict):
            continue
        day = str(snap.get("date") or "").strip()
        if not day:
            continue
        if snap.get("equity_ok") is False:
            continue
        try:
            eq = float(snap.get("equity") or 0)
            start = float(snap.get("day_start_equity") or 0)
        except (TypeError, ValueError):
            continue
        if eq <= 0:
            continue
        prev = by_date.get(day)
        if prev and str(prev.get("settled_at", "")) > str(snap.get("settled_at", "")):
            continue
        by_date[day] = {
            "date": day,
            "equity": eq,
            "day_start_equity": start,
            "day_pnl": float(snap.get("day_pnl") or (eq - start)),
            "day_pnl_pct": float(
                snap.get("day_pnl_pct")
                if snap.get("day_pnl_pct") is not None
                else ((eq / start - 1) * 100 if start > 0 else 0.0)
            ),
            "n_positions": int(snap.get("n_positions") or 0),
            "cash": float(snap.get("cash") or 0),
            "market_value": float(snap.get("market_value") or 0),
            "state": snap.get("state", ""),
            "stats": snap.get("stats") or {},
        }

    curve = [by_date[k] for k in sorted(by_date)]
    if curve:
        return curve

    # 退化：没有任何结算报告时，用纸账户当前权益作单点
    acct = _read_json(Path(account_file or (Path(DATA_DIR) / "paper_account.json")))
    if isinstance(acct, dict):
        try:
            cash = float(acct.get("cash") or 0)
            mv = sum(float(h.get("market_value") or 0)
                     for h in (acct.get("holdings") or {}).values())
            eq = cash + mv
            if eq > 0:
                return [{
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "equity": eq, "day_start_equity": eq, "day_pnl": 0.0,
                    "day_pnl_pct": 0.0, "n_positions": len(acct.get("holdings") or {}),
                    "cash": cash, "market_value": mv, "state": "snapshot", "stats": {},
                }]
        except (TypeError, ValueError):
            pass
    return []


def load_trades(history_file: Optional[Path] = None) -> List[dict]:
    """读纸账户成交流水（list[dict]，失败返回空表）。"""
    raw = _read_json(Path(history_file or PAPER_HISTORY))
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    return []


# ======================================================================
# 基础统计
# ======================================================================

def daily_returns(curve: Sequence[dict]) -> List[float]:
    """日收益率序列（小数，非百分比）。

    优先用 ``day_start_equity`` 做分母（真实日内基数）；缺失时退回前一日权益。
    首日没有前值 → 用 day_pnl_pct 兜，仍缺失则不计入。
    """
    out: List[float] = []
    prev: Optional[float] = None
    for p in curve:
        eq = float(p.get("equity") or 0)
        start = float(p.get("day_start_equity") or 0)
        if start > 0:
            out.append(eq / start - 1.0)
        elif prev and prev > 0:
            out.append(eq / prev - 1.0)
        prev = eq
    return out


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sharpe_ratio(returns: Sequence[float], rf_annual: float = 0.0,
                 periods: int = TRADING_DAYS_PER_YEAR) -> Optional[float]:
    """年化夏普。样本 < 20 个交易日时返回 None（不足以免疫噪声）。"""
    if len(returns) < 20:
        return None
    sd = _std(returns)
    if sd <= 0:
        return None
    rf_daily = rf_annual / periods
    return (_mean(returns) - rf_daily) / sd * math.sqrt(periods)


def sortino_ratio(returns: Sequence[float], rf_annual: float = 0.0,
                  periods: int = TRADING_DAYS_PER_YEAR) -> Optional[float]:
    """年化索提诺（只惩罚下行波动）。"""
    if len(returns) < 20:
        return None
    rf_daily = rf_annual / periods
    downs = [min(0.0, r - rf_daily) for r in returns]
    dd = math.sqrt(sum(d * d for d in downs) / len(downs)) if downs else 0.0
    if dd <= 0:
        return None
    return (_mean(returns) - rf_daily) / dd * math.sqrt(periods)


def drawdown_series(curve: Sequence[dict]) -> List[dict]:
    """逐日回撤（相对历史峰值）。"""
    peak = 0.0
    out: List[dict] = []
    for p in curve:
        eq = float(p.get("equity") or 0)
        peak = max(peak, eq)
        dd = (eq / peak - 1.0) * 100 if peak > 0 else 0.0
        out.append({"date": p.get("date"), "equity": eq,
                    "peak": round(peak, 2), "dd_pct": round(dd, 4)})
    return out


def max_drawdown(curve: Sequence[dict]) -> dict:
    """最大回撤 + 峰谷日期 + 持续（交易日）数。"""
    if not curve:
        return {"pct": 0.0, "peak_date": "", "trough_date": "", "days": 0, "recovered": False}
    peak = 0.0
    peak_date = ""
    best = {"pct": 0.0, "peak_date": "", "trough_date": "", "days": 0}
    trough_idx = 0
    for i, p in enumerate(curve):
        eq = float(p.get("equity") or 0)
        if eq > peak:
            peak, peak_date = eq, str(p.get("date"))
        dd = (eq / peak - 1.0) * 100 if peak > 0 else 0.0
        if dd < best["pct"]:
            best = {"pct": dd, "peak_date": peak_date,
                    "trough_date": str(p.get("date")), "days": 0}
            trough_idx = i
    # 回撤持续期：从峰值日到恢复日（或到现在）
    if best["pct"] < 0 and best["peak_date"]:
        try:
            i_peak = next(i for i, p in enumerate(curve)
                          if str(p.get("date")) == best["peak_date"])
            peak_val = float(curve[i_peak].get("equity") or 0)
            recovered = False
            for j in range(trough_idx + 1, len(curve)):
                if float(curve[j].get("equity") or 0) >= peak_val:
                    best["days"] = j - i_peak
                    recovered = True
                    break
            if not recovered:
                best["days"] = len(curve) - 1 - i_peak
            best["recovered"] = recovered
        except StopIteration:
            pass
    return best


def annualized_return(curve: Sequence[dict]) -> Optional[float]:
    """年化收益率（%）。样本 < 5 个交易日返回 None —— 一周的数据推不出年化。"""
    if len(curve) < 5:
        return None
    first_start = float(curve[0].get("day_start_equity") or 0) or float(curve[0].get("equity") or 0)
    last = float(curve[-1].get("equity") or 0)
    if first_start <= 0 or last <= 0:
        return None
    n = len(curve)
    total = last / first_start
    try:
        return (total ** (TRADING_DAYS_PER_YEAR / n) - 1.0) * 100
    except (OverflowError, ZeroDivisionError):
        return None


def monthly_matrix(curve: Sequence[dict]) -> Dict[str, float]:
    """月度收益（%）：{YYYY-MM: pct}，用当月最后一日权益 / 上月末权益。"""
    if not curve:
        return {}
    out: Dict[str, float] = {}
    prev_equity = float(curve[0].get("day_start_equity") or curve[0].get("equity") or 0)
    month_last: Dict[str, float] = {}
    for p in curve:
        m = str(p.get("date"))[:7]
        month_last[m] = float(p.get("equity") or 0)
    for m in sorted(month_last):
        cur = month_last[m]
        if prev_equity > 0 and cur > 0:
            out[m] = round((cur / prev_equity - 1.0) * 100, 4)
        prev_equity = cur
    return out


def rolling_sharpe(curve: Sequence[dict], window: int = 20) -> List[dict]:
    """滚动夏普（默认 20 日窗口）—— 用来观察「优势是否在衰减」。"""
    rets = daily_returns(curve)
    out: List[dict] = []
    if window <= 1 or len(rets) < window:
        return out
    for i in range(window - 1, len(rets)):
        win = rets[i - window + 1: i + 1]
        sd = _std(win)
        val = (None if sd <= 0 else _mean(win) / sd * math.sqrt(TRADING_DAYS_PER_YEAR))
        out.append({"date": curve[i + 1]["date"] if i + 1 < len(curve) else "",
                    "sharpe": None if val is None else round(val, 3)})
    return out


# ======================================================================
# 交易统计与归因
# ======================================================================

def trade_stats(trades: Sequence[dict]) -> dict:
    """成交层面的胜率/盈亏比/期望值/连亏/成本占比。

    只统计**平仓腿**（side=sell 且 pnl != 0）：买入腿没有盈亏，混进来会把胜率
    稀释成 0，这是最常见的归因错误。
    """
    sells = [t for t in trades
             if str(t.get("side")) == "sell" and float(t.get("pnl") or 0) != 0]
    buys = [t for t in trades if str(t.get("side")) == "buy"]
    pnls = [float(t.get("pnl") or 0) for t in sells]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    fees = sum(float(t.get("commission") or 0) + float(t.get("stamp_tax") or 0)
               for t in trades)
    turnover = sum(float(t.get("amount") or 0) for t in trades)

    # 最长连续亏损（心理/策略失效的先行指标）
    worst_streak = cur = 0
    for x in pnls:
        cur = cur + 1 if x < 0 else 0
        worst_streak = max(worst_streak, cur)

    return {
        "n_buys": len(buys),
        "n_closes": len(sells),
        "total_pnl": round(sum(pnls), 2),
        "win_rate": round(len(wins) / len(sells), 4) if sells else None,
        "profit_factor": (round(gross_win / gross_loss, 3) if gross_loss > 0
                          else (999.0 if gross_win > 0 else None)),
        "avg_win": round(gross_win / len(wins), 2) if wins else None,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else None,
        "payoff_ratio": (round((gross_win / len(wins)) / (gross_loss / len(losses)), 3)
                         if wins and losses else None),
        "expectancy": round(sum(pnls) / len(sells), 2) if sells else None,
        "max_consecutive_losses": worst_streak,
        "fees": round(fees, 2),
        "turnover": round(turnover, 2),
        "fee_ratio_pct": round(fees / turnover * 100, 4) if turnover > 0 else None,
    }


def attribute(
    trades: Sequence[dict],
    industry_map: Optional[Dict[str, str]] = None,
) -> dict:
    """按策略 / 行业 / 个股拆解已实现盈亏。

    没有这张表，「哪条腿在赚钱」只能靠猜 —— 而组合里最贵的错误就是
    「砍掉赚钱的、留下亏钱的」。
    """
    sells = [t for t in trades
             if str(t.get("side")) == "sell" and float(t.get("pnl") or 0) != 0]
    industry_map = industry_map or {}

    def _agg(key_fn) -> Dict[str, dict]:
        acc: Dict[str, dict] = {}
        for t in sells:
            k = key_fn(t) or "unknown"
            d = acc.setdefault(k, {"pnl": 0.0, "count": 0, "wins": 0, "fees": 0.0})
            pnl = float(t.get("pnl") or 0)
            d["pnl"] += pnl
            d["count"] += 1
            d["wins"] += 1 if pnl > 0 else 0
            d["fees"] += float(t.get("commission") or 0) + float(t.get("stamp_tax") or 0)
        for d in acc.values():
            d["pnl"] = round(d["pnl"], 2)
            d["fees"] = round(d["fees"], 2)
            d["win_rate"] = round(d["wins"] / d["count"], 4) if d["count"] else None
        return acc

    total = sum(float(t.get("pnl") or 0) for t in sells)
    by_sector = _agg(lambda t: industry_map.get(str(t.get("code")), "") or "未知行业")
    for d in by_sector.values():
        d["pnl_share_pct"] = round(d["pnl"] / total * 100, 2) if total else None

    return {
        "by_strategy": _agg(lambda t: str(t.get("strategy") or "")),
        "by_sector": by_sector,
        "by_code": _agg(lambda t: f"{t.get('code')} {t.get('name') or ''}".strip()),
    }


# 退出原因标签 → 人话。固定桶：即使某个桶为 0 也要展示，
# 否则「止盈 0 次 / 移动止损 0 次」这种决定性事实会被悄悄藏起来
# （实测：配 +15% 止盈线，实际最大盈利仅 +1.62%，止盈一次都没触发，
# 而在旧归因里只看到一个笼统的「stop」桶，根本看不出来）。
EXIT_REASON_LABELS = {
    "rebalance": "调仓换出",
    "stop_profit": "止盈",
    "stop_loss": "止损",
    "trail_stop": "移动止损",
    # 2026-10-27 之前所有风控离场都记成笼统的 "stop"，分不清止盈/止损/移动止损。
    # 单列「旧标签」桶：既不误导成「没有止损」，也不假装知道它到底是哪一种。
    "stop": "风控离场(旧标签)",
    "exit": "其他退出",
}


def exit_reason_distribution(trades: Sequence[dict]) -> List[dict]:
    """按退出原因（strategy 标签）聚合平仓盈亏，所有固定桶都在（含 0 值）。"""
    sells = [t for t in trades
             if str(t.get("side")) == "sell" and float(t.get("pnl") or 0) != 0]
    acc: Dict[str, dict] = {k: {"label": v, "pnl": 0.0, "count": 0, "wins": 0}
                            for k, v in EXIT_REASON_LABELS.items()}
    for t in sells:
        raw = str(t.get("strategy") or "") or "exit"
        # 未知标签归并到「其他退出」，不凭空造新桶（看板/日报按固定桶渲染）。
        tag = raw if raw in EXIT_REASON_LABELS else "exit"
        d = acc[tag]
        pnl = float(t.get("pnl") or 0)
        d["pnl"] += pnl
        d["count"] += 1
        d["wins"] += 1 if pnl > 0 else 0
    out = []
    for k, d in acc.items():
        out.append({
            "tag": k,
            "label": d["label"],
            "pnl": round(d["pnl"], 2),
            "count": d["count"],
            "win_rate": round(d["wins"] / d["count"], 4) if d["count"] else 0.0,
        })
    # 固定顺序：调仓 / 止盈 / 止损 / 移动止损 / 旧标签 / 其他
    order = ["rebalance", "stop_profit", "stop_loss", "trail_stop", "stop", "exit"]
    out.sort(key=lambda x: order.index(x["tag"]) if x["tag"] in order else 99)
    return out


# ======================================================================
# 汇总报告
# ======================================================================

def build_report(
    reports_dir: Optional[Path] = None,
    history_file: Optional[Path] = None,
    industry_file: Optional[Path] = None,
    rf_annual: float = 0.0,
    account_file: Optional[Path] = None,
) -> dict:
    """装配完整业绩档案。任何一项算不出来就是 None，绝不用 0 冒充。"""
    curve = load_equity_curve(reports_dir=reports_dir, account_file=account_file)
    trades = load_trades(history_file)
    ind_raw = _read_json(Path(industry_file or INDUSTRY_MAP_FILE))
    industry_map = {str(k): str(v) for k, v in ind_raw.items()} if isinstance(ind_raw, dict) else {}

    rets = daily_returns(curve)
    dd_series = drawdown_series(curve)
    dd = max_drawdown(curve)
    ann = annualized_return(curve)
    vol = _std(rets) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100 if len(rets) >= 2 else None
    shp = sharpe_ratio(rets, rf_annual=rf_annual)
    srt = sortino_ratio(rets, rf_annual=rf_annual)
    calmar = (ann / abs(dd["pct"]) if ann is not None and dd["pct"] < 0 else None)

    first_start = (float(curve[0].get("day_start_equity") or 0) or
                   float(curve[0].get("equity") or 0)) if curve else 0.0
    last_eq = float(curve[-1].get("equity") or 0) if curve else 0.0
    win_days = sum(1 for r in rets if r > 0)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period": {
            "start": curve[0]["date"] if curve else "",
            "end": curve[-1]["date"] if curve else "",
            "n_days": len(curve),
        },
        "equity": {
            "start": round(first_start, 2),
            "end": round(last_eq, 2),
            "total_return_pct": round((last_eq / first_start - 1) * 100, 4) if first_start > 0 else None,
            "annualized_return_pct": None if ann is None else round(ann, 3),
            "volatility_pct": None if vol is None else round(vol, 3),
        },
        "risk": {
            "sharpe": None if shp is None else round(shp, 3),
            "sortino": None if srt is None else round(srt, 3),
            "calmar": None if calmar is None else round(calmar, 3),
            "max_drawdown_pct": round(dd["pct"], 3),
            "max_dd_peak_date": dd["peak_date"],
            "max_dd_trough_date": dd["trough_date"],
            "max_dd_days": dd["days"],
            "max_dd_recovered": dd.get("recovered", False),
            "up_days": win_days,
            "down_days": len(rets) - win_days,
            "up_day_rate": round(win_days / len(rets), 4) if rets else None,
        },
        "trades": trade_stats(trades),
        "attribution": attribute(trades, industry_map),
        "exit_reason_distribution": exit_reason_distribution(trades),
        "monthly": monthly_matrix(curve),
        "rolling_sharpe": rolling_sharpe(curve, window=20),
        "curve": [{"date": p["date"], "equity": round(float(p["equity"]), 2),
                   "dd_pct": d["dd_pct"]}
                  for p, d in zip(curve, dd_series)],
        "drawdown": dd_series,
    }
    return report


def save_report(report: dict, path: Optional[Path] = None) -> Path:
    """原子落盘（先写 .tmp 再 replace），避免看板读到半截 JSON。"""
    path = Path(path or PERF_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_report(path: Optional[Path] = None) -> dict:
    data = _read_json(Path(path or PERF_FILE))
    return data if isinstance(data, dict) else {}


# ======================================================================
# 文本报告（终端 / 日报 / 推送共用）
# ======================================================================

def _fmt(v, nd: int = 2, suffix: str = "") -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def format_text(r: dict) -> str:
    eq, risk, tr = r.get("equity") or {}, r.get("risk") or {}, r.get("trades") or {}
    per = r.get("period") or {}
    lines = [
        f"业绩档案 {per.get('start','—')} ~ {per.get('end','—')}"
        f"（{per.get('n_days',0)} 个交易日）",
        "─" * 46,
        f"期末权益   ¥{_fmt(eq.get('end'))}   起点 ¥{_fmt(eq.get('start'))}",
        f"累计收益   {_fmt(eq.get('total_return_pct'), 2, '%')}"
        f"    年化 {_fmt(eq.get('annualized_return_pct'), 2, '%')}"
        f"    年化波动 {_fmt(eq.get('volatility_pct'), 2, '%')}",
        f"夏普 {_fmt(risk.get('sharpe'), 2)}"
        f"  索提诺 {_fmt(risk.get('sortino'), 2)}"
        f"  卡玛 {_fmt(risk.get('calmar'), 2)}",
        f"最大回撤 {_fmt(risk.get('max_drawdown_pct'), 2, '%')}"
        f"（{risk.get('max_dd_peak_date') or '—'} → {risk.get('max_dd_trough_date') or '—'}，"
        f"{risk.get('max_dd_days', 0)} 日，"
        f"{'已恢复' if risk.get('max_dd_recovered') else '未恢复'}）",
        f"上涨天数 {risk.get('up_days', 0)} / 下跌 {risk.get('down_days', 0)}"
        f"（胜日率 {_fmt((risk.get('up_day_rate') or 0) * 100, 1, '%')}）",
        "─" * 46,
        f"平仓 {tr.get('n_closes', 0)} 笔   胜率 {_fmt((tr.get('win_rate') or 0) * 100, 1, '%')}"
        f"   盈亏比 {_fmt(tr.get('payoff_ratio'), 2)}"
        f"   期望 {_fmt(tr.get('expectancy'))} 元/笔",
        f"累计已实现 {_fmt(tr.get('total_pnl'))} 元"
        f"   手续费 {_fmt(tr.get('fees'))} 元"
        f"（占成交额 {_fmt(tr.get('fee_ratio_pct'), 3, '%')}）"
        f"   最长连亏 {tr.get('max_consecutive_losses', 0)} 笔",
    ]
    by_sector = (r.get("attribution") or {}).get("by_sector") or {}
    if by_sector:
        lines.append("─" * 46)
        lines.append("行业贡献：")
        for k, v in sorted(by_sector.items(), key=lambda kv: kv[1]["pnl"], reverse=True):
            lines.append(f"  {k:<10} {_fmt(v['pnl'], 0)} 元"
                         f"  {v['count']} 笔  胜率 {_fmt((v['win_rate'] or 0) * 100, 0, '%')}")
    # 退出原因分布：固定桶（含 0 值），让「止盈 0 次」之类的事实一眼可见。
    erd = r.get("exit_reason_distribution") or []
    if erd:
        lines.append("─" * 46)
        lines.append("退出原因分布：")
        for e in erd:
            flag = "  ⚠️未触发" if e["count"] == 0 else ""
            lines.append(f"  {e['label']:<6} {_fmt(e['pnl'], 0)} 元"
                         f"  {e['count']} 笔  胜率 {_fmt(e['win_rate'] * 100, 0, '%')}{flag}")
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - CLI 入口
    import argparse
    ap = argparse.ArgumentParser(description="生成业绩档案")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--history", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rf", type=float, default=0.0, help="无风险年化利率(%%)")
    a = ap.parse_args()
    r = build_report(
        reports_dir=Path(a.reports_dir) if a.reports_dir else None,
        history_file=Path(a.history) if a.history else None,
        rf_annual=a.rf / 100.0,
    )
    p = save_report(r, Path(a.out) if a.out else None)
    print(format_text(r))
    print(f"\n已落盘: {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

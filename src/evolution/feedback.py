# -*- coding: utf-8 -*-
"""自我进化 · 反馈收集层。

一句话：**把「系统最近到底干得怎么样」变成一个结构化对象。**

四个候选生成器（参数寻优 / LLM 复盘 / 重训 / 遗传算法）都要吃同一份事实，
所以事实必须先被统一采集，否则会出现「A 生成器说准确率 0.62、
B 生成器说 0.48」的自相矛盾。

三路信号
========

1. **预测验证批次** —— ``data/prediction_history.json``
   每批「选出 20 只 → N 个交易日后实际涨跌」，给出准确率 / 超额 / 胜率 /
   评分分档表现。这是判断「模型还准不准」的唯一一手依据。
2. **交易结果回流** —— ``data/loop_reports/*.json``（每日收盘结算）
   + ``data/trading_journal.jsonl``（审计日志：下单/成交/熔断/拒单）
   + ``data/paper_account.json``。这是判断「策略赚不赚钱」的依据。
   注意：闭环引擎至今可能**一次都没结算过**，此时该路数据为空，
   必须显式标注而不是当成 0 收益。
3. **组合绩效指标** —— 从结算净值序列算出收益 / 最大回撤 / 日胜率 /
   盈亏比 / 换手，作为样本外闸门的对照基线。

设计铁律
========

* **默认只读**。:func:`collect` 绝不写任何文件。
  会写盘的 ``prediction_history.json`` 验证动作被隔离在
  :func:`verify_pending` 里，必须显式调用（由进化 runner 每日触发一次）。
* **缺数据要说话，不要装哑巴**。每路数据源都产出 ``health`` 状态与
  人类可读的 ``notes``，前端/CLI 直接展示，避免「0 条数据被当成 0 分」。
* 任何路径都经 ``config.settings`` 的 ``DATA_DIR`` / ``CONFIG_DIR``，
  打包后可外挂，绝不写死相对路径。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from config.settings import CONFIG_DIR, DATA_DIR

__all__ = [
    "LOOP_REPORTS_DIR", "PREDICTION_FILE", "PerformanceSnapshot",
    "collect", "collect_prediction_feedback", "collect_trade_feedback",
    "compute_metrics", "verify_pending", "render_text", "snapshot_to_json",
]

LOOP_REPORTS_DIR = Path(DATA_DIR) / "loop_reports"
PREDICTION_FILE = Path(DATA_DIR) / "prediction_history.json"
PAPER_ACCOUNT_FILE = Path(DATA_DIR) / "paper_account.json"

#: 单次采集的读取上限，防止文件无限增长拖垮进化回路
MAX_REPORTS = 180          # 最近 180 个交易日结算
MAX_JOURNAL_LINES = 5000   # 审计日志只取尾部
MAX_BATCHES = 120          # 最近 120 批预测

VERIFY_HORIZONS = (5, 10)


# ---------------------------------------------------------------------------
# 低层读取（全部容错，绝不抛异常）
# ---------------------------------------------------------------------------

def _read_json(path: Path, default: Any) -> Any:
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"读取 {path} 失败（按空处理）: {e}")
    return default


def _read_jsonl(path: Path, limit: int = MAX_JOURNAL_LINES) -> List[dict]:
    out: List[dict] = []
    try:
        p = Path(path)
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                continue
    except Exception as e:  # noqa: BLE001
        logger.warning(f"读取 {path} 失败（按空处理）: {e}")
    return out


def _guard_paths() -> Tuple[Path, Path]:
    """解析生效的风控状态/日志路径（尊重 ``config/trading_guard.yaml``）。"""
    fallback = (Path(DATA_DIR) / "trading_guard_state.json",
                Path(DATA_DIR) / "trading_journal.jsonl")
    try:
        from src.trading.guard import GuardConfig

        yml = Path(CONFIG_DIR) / "trading_guard.yaml"
        cfg = GuardConfig.from_yaml(str(yml)) if yml.exists() else GuardConfig()
        return Path(cfg.state_file), Path(cfg.journal_file)
    except Exception:  # noqa: BLE001
        return fallback


# ---------------------------------------------------------------------------
# 快照结构
# ---------------------------------------------------------------------------

@dataclass
class PerformanceSnapshot:
    """进化回路的事实底稿。所有生成器只读它，不各自去翻文件。"""

    generated_at: str = ""
    # ---- ① 预测验证 ----
    n_batches: int = 0
    n_verified: int = 0
    pending_verify: int = 0
    accuracy_by_horizon: Dict[str, List[float]] = field(default_factory=dict)
    recent_accuracy: Dict[str, Optional[float]] = field(default_factory=dict)
    excess_by_horizon: Dict[str, List[float]] = field(default_factory=dict)
    win_rate_by_horizon: Dict[str, List[float]] = field(default_factory=dict)
    score_buckets: Dict[str, Any] = field(default_factory=dict)
    # ---- ② 交易回流 ----
    n_settlements: int = 0
    equity_series: List[Dict[str, Any]] = field(default_factory=list)
    trade_stats: Dict[str, int] = field(default_factory=dict)
    last_halt: Optional[Dict[str, Any]] = None
    guard: Dict[str, Any] = field(default_factory=dict)
    paper_account: Dict[str, Any] = field(default_factory=dict)
    # ---- ③ 绩效 ----
    metrics: Dict[str, Any] = field(default_factory=dict)
    # ---- 数据完备度 ----
    health: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ---- 给生成器用的便捷取值 ----

    def latest_accuracy(self, horizon: int = 5) -> Optional[float]:
        vals = self.accuracy_by_horizon.get(str(horizon)) or []
        return vals[-1] if vals else None

    def is_useful(self) -> bool:
        """是否已经积累了足以驱动进化的证据。"""
        return bool(self.n_verified >= 3 or self.n_settlements >= 5)

    def summary(self) -> str:
        m = self.metrics or {}
        lines = [
            f"预测：{self.n_verified}/{self.n_batches} 批已验证"
            f"（待验证 {self.pending_verify}）",
            f"  近 {len(self.accuracy_by_horizon.get('5', []))} 批 T+5 准确率"
            f" {_fmt_pct_list(self.accuracy_by_horizon.get('5'))}",
            f"交易：{self.n_settlements} 个结算日，"
            f"成交 {self.trade_stats.get('fill', 0)} / "
            f"拒单 {self.trade_stats.get('risk_block', 0)}",
        ]
        if m.get("total_return") is not None:
            lines.append(
                f"绩效：累计收益 {m['total_return']:+.2%}，"
                f"最大回撤 {m.get('max_drawdown', 0):.2%}，"
                f"日胜率 {m.get('daily_win_rate', 0):.1%}")
        for n in self.notes:
            lines.append(f"⚠️ {n}")
        return "\n".join(lines)


def _fmt_pct_list(vals: Optional[List[float]]) -> str:
    if not vals:
        return "无数据"
    return "[" + ", ".join(f"{v:.3f}" for v in vals[-5:]) + "]"


# ---------------------------------------------------------------------------
# ① 预测验证反馈
# ---------------------------------------------------------------------------

def _entry_verifications(entry: dict) -> Dict[str, dict]:
    """兼容两代追踪器的结果字段。

    **权威规则不在这里** —— 转发到 ``tracker_v2.entry_verifications``。
    兼容逻辑一旦写成两份，迟早会写歪：例如一边按 ``lookback_days`` 取窗口、
    另一边按 ``hold_days`` 取，同一批数据在两处就会归到不同窗口上，
    于是「看板显示 T+5 有 17 批」而「生成器 C 看到 T+5 只有 3 批」。
    `tests/test_evolution_feedback.py` 有一条断言钉住两处实现一致。
    """
    from src.models.tracker_v2 import entry_verifications

    return entry_verifications(entry)


def _horizon_of(v: dict) -> str:
    return str(v.get("hold_days") or v.get("lookback_days") or 5)


def collect_prediction_feedback(
    history: Optional[List[dict]] = None,
    *,
    today: Optional[str] = None,
) -> Dict[str, Any]:
    """采集预测验证批次（只读）。

    Args:
        history: 直接注入历史（测试用）；缺省读 ``prediction_history.json``。
        today: 计算「待验证」时用的基准日（ISO），缺省取当天。

    Returns:
        含 ``accuracy_by_horizon`` 等字段的字典。
    """
    if history is None:
        raw = _read_json(PREDICTION_FILE, [])
        history = raw if isinstance(raw, list) else []

    batches = [e for e in history if isinstance(e, dict) and e.get("scan_date")]
    batches.sort(key=lambda e: e.get("scan_date", ""))
    batches = batches[-MAX_BATCHES:]

    acc: Dict[str, List[float]] = {str(h): [] for h in VERIFY_HORIZONS}
    excess: Dict[str, List[float]] = {str(h): [] for h in VERIFY_HORIZONS}
    win: Dict[str, List[float]] = {str(h): [] for h in VERIFY_HORIZONS}
    n_verified = 0

    for e in batches:
        for h, v in _entry_verifications(e).items():
            n_verified += 1
            bucket = h if h in acc else None
            if v.get("accuracy") is not None:
                if bucket:
                    acc[bucket].append(float(v["accuracy"]))
            if v.get("excess_return") is not None:
                if bucket:
                    excess[bucket].append(float(v["excess_return"]))
            if v.get("win_rate") is not None:
                if bucket:
                    win[bucket].append(float(v["win_rate"]))

    # 待验证：已走满 horizon 个交易日但还没有结果
    pending = 0
    if batches:
        pending = _count_pending(batches, today=today)

    # 评分分档：最近一批有分档数据的验证结果
    score_buckets: Dict[str, Any] = {}
    for e in reversed(batches):
        for h in sorted(_entry_verifications(e), key=lambda x: -int(x or 0)):
            v = _entry_verifications(e)[h]
            if v.get("by_score_bucket"):
                score_buckets = {
                    "scan_date": e.get("scan_date"),
                    "horizon": h,
                    "buckets": v["by_score_bucket"],
                }
                break
        if score_buckets:
            break

    def _recent(vals: List[float], n: int = 5) -> Optional[float]:
        if not vals:
            return None
        tail = vals[-n:]
        return round(sum(tail) / len(tail), 4)

    return {
        "n_batches": len(batches),
        "n_verified": n_verified,
        "pending_verify": pending,
        "accuracy_by_horizon": acc,
        "recent_accuracy": {k: _recent(v) for k, v in acc.items()},
        "excess_by_horizon": excess,
        "win_rate_by_horizon": win,
        "score_buckets": score_buckets,
        "date_range": ([batches[0].get("scan_date"), batches[-1].get("scan_date")]
                       if batches else []),
        "missing_snapshot": sum(
            1 for e in batches if not e.get("price_snapshot")),
        # 空批次：没标的也没快照，永远验不出来。单独报出来，
        # 免得用户以为「数据丢了」——它们本来就是当天没选出标的的日子。
        "n_unverifiable": sum(
            1 for e in batches
            if e.get("scan_date") and not _is_verifiable(e)),
    }


def _is_verifiable(entry: dict) -> bool:
    """这个批次**有没有可能被验证**（不看日期，只看有没有可验的东西）。

    空批次（当天没选出标的、也没留 price_snapshot）**永远**验不出来：
    ``tracker_v2.verify_predictions`` 一进门就 ``if not codes: return None``。

    把它们算进 pending 会造出一个自我循环的错误指令：``--doctor`` 天天催
    「另有 N 个已走满持有期但未验证 → 跑 --run --verify」，而真跑了
    **永远是「新增 0 条」**（实测 2026-10-16：报 10 个，补验 0 个）。
    计数口径必须和「能不能真的验」对齐，否则就是在骗人。
    """
    if entry.get("price_snapshot"):
        return True
    preds = entry.get("predictions") or []
    return any(isinstance(p, dict) and p.get("code") for p in preds)


def _count_pending(batches: List[dict], today: Optional[str] = None) -> int:
    """统计「持有期已走满但尚无验证结果」的 (批次, 窗口) 数量。

    只统计**可验证**的批次（见 ``_is_verifiable``）—— 空批次不计。
    """
    try:
        from src.trading.calendar import trading_days_between
    except Exception:  # noqa: BLE001
        return 0

    base = today or datetime.now().strftime("%Y-%m-%d")
    n = 0
    for e in batches:
        scan = e.get("scan_date")
        if not scan:
            continue
        if not _is_verifiable(e):
            continue
        done = set(_entry_verifications(e))
        for h in VERIFY_HORIZONS:
            if str(h) in done:
                continue
            try:
                if trading_days_between(scan, base) >= h:
                    n += 1
            except Exception:  # noqa: BLE001
                continue
    return n


# ---------------------------------------------------------------------------
# ② 交易结果回流
# ---------------------------------------------------------------------------

def collect_trade_feedback() -> Dict[str, Any]:
    """采集闭环结算 + 审计日志 + 纸账户（只读）。"""
    state_path, journal_path = _guard_paths()

    # ---- 结算报告 ----
    reports: List[dict] = []
    try:
        if LOOP_REPORTS_DIR.exists():
            files = sorted(LOOP_REPORTS_DIR.glob("loop_report_*.json"))
            for f in files[-MAX_REPORTS:]:
                d = _read_json(f, None)
                if isinstance(d, dict) and d.get("date"):
                    reports.append(d)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"扫描闭环结算报告失败（按空处理）: {e}")
    reports.sort(key=lambda d: d.get("date", ""))

    equity_series = [
        {
            "date": d.get("date"),
            "equity": d.get("equity"),
            "day_pnl": d.get("day_pnl"),
            "day_pnl_pct": d.get("day_pnl_pct"),
            "n_positions": d.get("n_positions"),
            "cash": d.get("cash"),
            "market_value": d.get("market_value"),
        }
        for d in reports
    ]

    # ---- 审计日志 ----
    journal = _read_jsonl(journal_path)
    counts: Dict[str, int] = {}
    notional_buy = notional_sell = 0.0
    last_halt: Optional[dict] = None
    fills: List[dict] = []
    for rec in journal:
        ev = rec.get("event") or "unknown"
        counts[ev] = counts.get(ev, 0) + 1
        data = rec.get("data") or {}
        if ev == "fill":
            fills.append(rec)
            try:
                amt = float(data.get("qty", 0)) * float(data.get("price", 0))
            except Exception:  # noqa: BLE001
                amt = 0.0
            if str(data.get("side", "")).lower() in ("sell", "s"):
                notional_sell += amt
            else:
                notional_buy += amt
        elif ev == "halt":
            last_halt = rec

    # ---- 纸账户 / 风控状态 ----
    paper = _read_json(PAPER_ACCOUNT_FILE, {})
    guard_state = _read_json(state_path, {})

    return {
        "n_settlements": len(reports),
        "equity_series": equity_series,
        "trade_stats": counts,
        "notional_buy": round(notional_buy, 2),
        "notional_sell": round(notional_sell, 2),
        "n_fills": len(fills),
        "last_halt": last_halt,
        "journal_lines": len(journal),
        "guard": {
            "mode": guard_state.get("mode"),
            "halted": guard_state.get("halted"),
            "halt_reason": guard_state.get("halt_reason"),
            "peak_equity": guard_state.get("peak_equity"),
            "today_start_equity": guard_state.get("today_start_equity"),
            "baseline_account": guard_state.get("baseline_account"),
        },
        "paper_account": paper if isinstance(paper, dict) else {},
        "reports_source": str(LOOP_REPORTS_DIR),
        "journal_source": str(journal_path),
    }


# ---------------------------------------------------------------------------
# ③ 组合绩效指标
# ---------------------------------------------------------------------------

def compute_metrics(equity_series: List[dict],
                    n_fills: int = 0,
                    notional: float = 0.0) -> Dict[str, Any]:
    """从结算净值序列算绩效指标。

    净值序列不足 2 天时**不编造收益率**，只返回计数与 ``insufficient`` 标记。
    """
    pts: List[Tuple[str, float]] = []
    for row in equity_series or []:
        try:
            eq = float(row.get("equity"))
        except (TypeError, ValueError):
            continue
        if eq > 0:
            pts.append((str(row.get("date") or ""), eq))

    if len(pts) < 2:
        return {
            "n_days": len(pts),
            "insufficient": True,
            "note": ("净值样本不足 2 个结算日，无法计算收益/回撤。"
                     "闭环引擎尚未收盘结算过，或结算报告缺失。"),
            "n_fills": n_fills,
            "notional": round(float(notional), 2),
        }

    eqs = [p[1] for p in pts]
    total_return = eqs[-1] / eqs[0] - 1.0

    # 最大回撤（真实时序，不是排序后累计）
    peak = eqs[0]
    max_dd = 0.0
    for v in eqs:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)

    # 日收益
    rets = [eqs[i] / eqs[i - 1] - 1.0 for i in range(1, len(eqs)) if eqs[i - 1] > 0]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

    # 年化（按 244 个交易日）
    n = len(rets)
    ann_return = ((1 + total_return) ** (244 / n) - 1) if n > 0 else None
    vol = None
    if n > 1:
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / (n - 1)
        vol = (var ** 0.5) * (244 ** 0.5)
    sharpe = (ann_return / vol) if (ann_return is not None and vol and vol > 0) else None

    avg_equity = sum(eqs) / len(eqs)
    turnover = (float(notional) / avg_equity) if avg_equity > 0 else None

    return {
        "n_days": len(pts),
        "insufficient": False,
        "start_date": pts[0][0],
        "end_date": pts[-1][0],
        "start_equity": round(eqs[0], 2),
        "end_equity": round(eqs[-1], 2),
        "total_return": round(total_return, 6),
        "max_drawdown": round(max_dd, 6),
        "daily_win_rate": round(len(wins) / len(rets), 4) if rets else None,
        "profit_loss_ratio": round(avg_win / avg_loss, 4) if avg_loss > 0 else None,
        "annualized_return": round(ann_return, 6) if ann_return is not None else None,
        "annualized_vol": round(vol, 6) if vol is not None else None,
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "calmar": (round(ann_return / max_dd, 4)
                   if (ann_return is not None and max_dd > 0) else None),
        "turnover": round(turnover, 4) if turnover is not None else None,
        "n_fills": n_fills,
        "notional": round(float(notional), 2),
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def collect(with_predictions: bool = True,
            with_trades: bool = True) -> PerformanceSnapshot:
    """采集全部反馈信号（**只读**，不写任何文件）。"""
    snap = PerformanceSnapshot(
        generated_at=datetime.now().isoformat(timespec="seconds"))

    pred: Dict[str, Any] = {}
    trade: Dict[str, Any] = {}

    if with_predictions:
        pred = collect_prediction_feedback()
        snap.n_batches = pred["n_batches"]
        snap.n_verified = pred["n_verified"]
        snap.pending_verify = pred["pending_verify"]
        snap.accuracy_by_horizon = pred["accuracy_by_horizon"]
        snap.recent_accuracy = pred["recent_accuracy"]
        snap.excess_by_horizon = pred["excess_by_horizon"]
        snap.win_rate_by_horizon = pred["win_rate_by_horizon"]
        snap.score_buckets = pred["score_buckets"]

    if with_trades:
        trade = collect_trade_feedback()
        snap.n_settlements = trade["n_settlements"]
        snap.equity_series = trade["equity_series"]
        snap.trade_stats = trade["trade_stats"]
        snap.last_halt = trade["last_halt"]
        snap.guard = trade["guard"]
        snap.paper_account = trade["paper_account"]

    notional = float(trade.get("notional_buy", 0)) + float(trade.get("notional_sell", 0))
    snap.metrics = compute_metrics(
        snap.equity_series,
        n_fills=int(trade.get("n_fills", 0)),
        notional=notional,
    )

    snap.health, snap.notes = _diagnose(pred, trade, snap)
    return snap


def _diagnose(pred: Dict[str, Any], trade: Dict[str, Any],
              snap: PerformanceSnapshot) -> Tuple[Dict[str, Any], List[str]]:
    """把「数据缺了什么」翻译成可执行提示。

    这是本模块最容易被人忽略、但最有价值的部分：
    没有它，0 条数据会被静默当成 0 分准确率，生成器据此做出错误决策。
    """
    health: Dict[str, Any] = {}
    notes: List[str] = []

    # --- 预测侧 ---
    health["prediction"] = {
        "file": str(PREDICTION_FILE),
        "exists": PREDICTION_FILE.exists(),
        "n_batches": pred.get("n_batches", 0),
        "n_verified": pred.get("n_verified", 0),
        "pending_verify": pred.get("pending_verify", 0),
        "missing_snapshot": pred.get("missing_snapshot", 0),
    }
    if not pred:
        notes.append("未采集预测反馈。")
    elif pred.get("n_batches", 0) == 0:
        notes.append("预测历史为空：系统还没产出过任何一批选股，"
                     "无准确率可依据，进化应保持观望。")
    else:
        pending = pred.get("pending_verify", 0)
        if pred.get("n_verified", 0) == 0:
            if pending > 0:
                # 最关键的一条：既有事实（到期了）又有动作（去补验）
                notes.append(
                    f"有 {pred['n_batches']} 批预测但**0 批完成验证**，"
                    f"其中 {pending} 个（批次×窗口）已走满持有期却未验证。"
                    "→ 准确率历史为空，阈值进化与自动重训都不会触发。"
                    "请运行 verify_pending() 补齐。")
            else:
                notes.append(
                    f"有 {pred['n_batches']} 批预测但**0 批完成验证**："
                    "准确率历史为空 → 阈值进化与自动重训都不会触发。"
                    "（当前还没有任何批次走满持有期，属正常等待。）")
        elif pending > 0:
            notes.append(f"还有 {pending} 个（批次×窗口）"
                         "已走满持有期但未验证，建议立即补验。")
        if pred.get("missing_snapshot", 0) > 0:
            notes.append(
                f"{pred['missing_snapshot']} 批预测缺 price_snapshot，"
                "验证时会回退用复权价反推，收益可能失真。")

    # --- 交易侧 ---
    health["trade"] = {
        "loop_reports_dir": trade.get("reports_source"),
        "n_settlements": trade.get("n_settlements", 0),
        "journal_source": trade.get("journal_source"),
        "journal_lines": trade.get("journal_lines", 0),
        "n_fills": trade.get("n_fills", 0),
        "guard_mode": (trade.get("guard") or {}).get("mode"),
        "guard_halted": (trade.get("guard") or {}).get("halted"),
    }
    if not trade:
        notes.append("未采集交易反馈。")
    else:
        if trade.get("n_settlements", 0) == 0:
            notes.append(
                "闭环尚未收盘结算过任何一天（loop_reports 为空）："
                "拿不到净值序列 → 收益/回撤/换手全部不可用，"
                "任何「候选策略更优」的结论都缺乏一手依据。")
        if (trade.get("guard") or {}).get("halted"):
            notes.append(
                f"风控当前处于熔断状态：{trade['guard'].get('halt_reason')}。"
                "熔断期间不应落地任何改动。")
        if trade.get("journal_lines", 0) == 0:
            notes.append("交易审计日志为空：从未有过下单尝试，"
                         "交易成本与拒单率无从评估。")

    # --- 可进化性 ---
    health["useful"] = snap.is_useful()
    if not snap.is_useful():
        notes.append("证据量不足（需 ≥3 批验证或 ≥5 个结算日），"
                     "进化回路应只做观察，不产出候选。")

    return health, notes


# ---------------------------------------------------------------------------
# 唯一会写盘的动作：补验
# ---------------------------------------------------------------------------

def verify_pending(horizons: Optional[List[int]] = None,
                   feed_engine: bool = True) -> List[Dict[str, Any]]:
    """补齐所有「已走满持有期但未验证」的预测批次。

    **这是本模块唯一会写文件的函数**（写 ``prediction_history.json``；
    ``feed_engine=True`` 时还会写 ``data/evolution_state.json`` 的
    ``performance`` 计数）。之所以单独拎出来，是因为它需要读盘/联网、
    耗时长，必须由进化 runner 按日触发，而绝不能混在只读采集里被反复调用。

    走 ``tracker_v2``（按交易日精确计算持有期、T+5/T+10 分开存），
    而不是旧 ``tracker``（用「今天减 N 天」反推日期，命中率极低，
    且两个窗口互相覆盖 —— 实测 25 批预测只验证上 2 批）。

    Args:
        horizons: 要补验的持有期窗口，默认 (5, 10)。
        feed_engine: 是否把结果回灌给 ``EvolutionEngine.record_verification``
            （只记准确率序列，**不改任何参数**），供自动重训判定使用。
    """
    hs = list(horizons or VERIFY_HORIZONS)
    try:
        from src.models import tracker_v2
    except Exception as e:  # noqa: BLE001
        logger.error(f"无法加载 tracker_v2，跳过补验: {e}")
        return []

    try:
        results = tracker_v2.verify_all_pending(horizons=hs) or []
    except Exception as e:  # noqa: BLE001
        logger.error(f"补验失败: {e}")
        return []

    if results and feed_engine:
        try:
            from src.models.evolution import EvolutionEngine

            eng = EvolutionEngine()
            for r in results:
                eng.record_verification(r)
            eng.save()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"验证结果回灌进化引擎失败（不影响验证本身）: {e}")

    logger.info(f"✅ 预测补验完成：新增 {len(results)} 条验证结果")
    return results


# ---------------------------------------------------------------------------
# 序列化 / 展示
# ---------------------------------------------------------------------------

def snapshot_to_json(snap: PerformanceSnapshot, indent: int = 2) -> str:
    return json.dumps(snap.to_dict(), ensure_ascii=False, indent=indent, default=str)


def render_text(snap: PerformanceSnapshot) -> str:
    """人类可读报告（CLI 与 LLM 复盘共用同一份事实）。"""
    L: List[str] = []
    L.append(f"# 进化反馈快照 {snap.generated_at}")
    L.append("")
    L.append("## ① 预测验证")
    L.append(f"- 批次：{snap.n_batches} 批（已验证 {snap.n_verified}，"
             f"待验证 {snap.pending_verify}）")
    for h in VERIFY_HORIZONS:
        k = str(h)
        accs = snap.accuracy_by_horizon.get(k) or []
        if not accs:
            L.append(f"- T+{h}：无验证结果")
            continue
        recent = snap.recent_accuracy.get(k)
        L.append(f"- T+{h}：{len(accs)} 次验证，近 5 次均值 "
                 f"{recent:.1%}，序列 {_fmt_pct_list(accs)}")
    if snap.score_buckets:
        L.append(f"- 评分分档（{snap.score_buckets.get('scan_date')} "
                 f"T+{snap.score_buckets.get('horizon')}）：")
        for k, v in (snap.score_buckets.get("buckets") or {}).items():
            L.append(f"    · {k}: {v:+.2%}" if v is not None else f"    · {k}: -")
    L.append("")
    L.append("## ② 交易回流")
    L.append(f"- 结算日：{snap.n_settlements} 个")
    ts = snap.trade_stats or {}
    L.append(f"- 事件计数：订单信号 {ts.get('signal', 0)}，提案 {ts.get('proposal', 0)}，"
             f"成交 {ts.get('fill', 0)}，拒单 {ts.get('risk_block', 0)}，"
             f"熔断 {ts.get('halt', 0)}")
    g = snap.guard or {}
    L.append(f"- 风控：模式 {g.get('mode')}，熔断 {g.get('halted')}"
             f"（峰值净值 {g.get('peak_equity')}）")
    L.append("")
    L.append("## ③ 绩效")
    m = snap.metrics or {}
    if m.get("insufficient"):
        L.append(f"- 数据不足：{m.get('note')}")
    else:
        L.append(f"- 区间：{m.get('start_date')} ~ {m.get('end_date')}"
                 f"（{m.get('n_days')} 个结算日）")
        L.append(f"- 累计收益：{m.get('total_return', 0):+.2%}"
                 f"（年化 {_pct(m.get('annualized_return'))}）")
        L.append(f"- 最大回撤：{m.get('max_drawdown', 0):.2%}，"
                 f"Calmar {_num(m.get('calmar'))}，Sharpe {_num(m.get('sharpe'))}")
        L.append(f"- 日胜率：{_pct(m.get('daily_win_rate'))}，"
                 f"盈亏比 {_num(m.get('profit_loss_ratio'))}，"
                 f"换手 {_num(m.get('turnover'))}")
    if snap.notes:
        L.append("")
        L.append("## ⚠️ 数据完备度提示")
        for n in snap.notes:
            L.append(f"- {n}")
    return "\n".join(L)


def _pct(v: Optional[float]) -> str:
    return f"{v:.2%}" if isinstance(v, (int, float)) else "-"


def _num(v: Optional[float]) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"

# -*- coding: utf-8 -*-
"""RankIC 滚动监控核心（可导入，供调度器 / 看板 / CLI 复用）。

两个能力：
  1. run_backfill()  —— 用真实预测器对历史评分日打分（严格 OOS：只用 d 及之前数据），
     算「评分日 → d+horizon 交易日」的超额收益，逐日算横截面 Spearman IC，
     累积进 data/rolling_ic_log.json。脚本 scripts/rolling_ic_backfill.py 是它的 CLI 壳。
  2. render_curve() —— 读日志产出稳定性曲线 SVG（每日 IC 散点 / 0 线 / 生产基线 /
     滚动均值±1σ 带）；看板「📈 模型稳定性」页与 scripts/rolling_ic_curve.py 共用。

前视防护：评分日 d 的打分只用 d 及之前数据，前向收益用 d+horizon（交易日），
绝不回看未来之外的信息，所以算出的每日 IC 是真实 OOS 排序能力。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IC_LOG = ROOT / "data" / "rolling_ic_log.json"
PROD_METRICS = ROOT / "models" / "production_metrics.json"
STATUS_FILE = ROOT / "data" / "rolling_ic_status.json"
OUT_HTML = ROOT / "data" / "rolling_ic_curve.html"
OUT_SVG = ROOT / "data" / "rolling_ic_curve.svg"


def _load_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _next_trading_day(d: pd.Timestamp, n: int) -> pd.Timestamp:
    from src.trading.calendar import next_trading_day
    cur = d
    for _ in range(n):
        cur = pd.Timestamp(next_trading_day(cur, include_today=False))
    return cur


def _write_status(d: Dict) -> None:
    try:
        STATUS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _stats_from_entries(entries: List[Dict]) -> Dict:
    """从 IC 日志条目算出 {total, mean_ic, ir_mean_std, t_stat, pos_rate, last_date}。"""
    ics = [float(r.get("ic")) for r in entries
           if r.get("ic") is not None and r.get("ic") == r.get("ic")]
    if not ics:
        return {"total": len(entries), "mean_ic": None, "ir_mean_std": None,
                "t_stat": None, "pos_rate": None,
                "last_date": entries[-1]["date"] if entries else None}
    m = float(np.mean(ics))
    sd = float(np.std(ics, ddof=1)) if len(ics) > 1 else float("nan")
    se = sd / (len(ics) ** 0.5) if len(ics) > 1 and sd == sd else float("nan")
    ir = m / sd if sd and sd > 0 else float("nan")
    tstat = m / se if se and se > 0 else float("nan")
    return {
        "total": len(entries),
        "mean_ic": round(m, 4),
        "ir_mean_std": round(ir, 3) if ir == ir else None,
        "t_stat": round(tstat, 2) if tstat == tstat else None,
        "pos_rate": round(sum(1 for v in ics if v > 0) / len(ics), 3),
        "last_date": entries[-1]["date"] if entries else None,
    }


def refresh_status(log_path: Optional[Path] = None, extra: Optional[Dict] = None) -> Dict:
    """按**当前日志**重算并落盘 ``rolling_ic_status.json``。

    为什么需要这个入口：``rolling_ic_log.json`` 有**两个生产者** ——
    ``run_backfill``（收盘后批量回填）和 ``scripts/paper_shadow_audit.py``
    （盘后算当天 live-IC）。2026-10-19 之前后者只往日志里追加、从不更新状态，
    于是看板从状态文件读到「20 条 / 最后 2026-10-09」，而日志里其实已经有
    31 条 / 最后 2026-10-12 —— 显示和真实数据对不上，且没人知道该信哪个。
    """
    path = Path(log_path) if log_path else IC_LOG
    log = _load_json(path) or []
    entries = sorted(
        [r for r in log if isinstance(r, dict) and r.get("date")],
        key=lambda r: r.get("date", ""),
    )
    status: Dict = {"status": "ok" if entries else "empty"}
    status.update(_stats_from_entries(entries))
    status["at"] = datetime.now().isoformat(timespec="seconds")
    if extra:
        status.update(extra)
    _write_status(status)
    return status


def run_backfill(
    days: int = 30,
    sample: int = 150,
    horizon: int = 5,
    min_stocks: int = 30,
    pool: str = "zz500",
    out: Optional[Path] = None,
    dry_run: bool = False,
    progress_cb: Optional[Callable[[int, int, int], None]] = None,
) -> Dict:
    """回填逐日 RankIC 到 rolling_ic_log.json（可重入：已存在的评分日跳过）。

    Returns
    -------
    dict: {status, added, total, mean_ic, ir_mean_std, t_stat, pos_rate, last_date}
    """
    from src.data.fetcher import fetcher
    from src.models.predictor import predictor

    out = Path(out) if out else IC_LOG
    _write_status({"status": "running", "started_at": datetime.now().isoformat(timespec="seconds"),
                   "days": days, "pool": pool})
    try:
        codes = list(fetcher.get_stock_pool(pool))
        if sample and sample < len(codes):
            codes = codes[:sample]

        raw: Dict[str, pd.DataFrame] = {}
        for i, code in enumerate(codes, 1):
            try:
                df = fetcher.get_daily_kline(code)
                if df is None or len(df) < 120:
                    continue
                df = df.copy()
                df.index = pd.to_datetime(df["date"])
                df = df.sort_index()
                raw[code] = df
            except Exception:  # noqa: BLE001
                pass
            if progress_cb and i % 50 == 0:
                progress_cb(i, len(codes), 0)

        if not raw:
            _write_status({"status": "error", "error": "无任何可用行情",
                           "at": datetime.now().isoformat(timespec="seconds")})
            return {"status": "error", "error": "no_usable_klines"}

        close = pd.DataFrame({c: df["close"] for c, df in raw.items()}).sort_index()
        all_dates = list(close.index)
        candidates = []
        i = len(all_dates) - 1
        while i >= 0 and len(candidates) < days:
            d = all_dates[i]
            d_fwd = _next_trading_day(d, horizon)
            if d_fwd in close.index:
                candidates.append(d)
            i -= 1
        candidates = list(reversed(candidates))

        log = _load_json(out) or []
        done = {r.get("date") for r in log if isinstance(r, dict)}
        todo = [d for d in candidates if d.strftime("%Y-%m-%d") not in done]

        if dry_run:
            _write_status({"status": "dry_run", "candidates": len(candidates),
                           "todo": len(todo), "at": datetime.now().isoformat(timespec="seconds")})
            return {"status": "dry_run", "candidates": len(candidates), "todo": len(todo)}

        if not todo:
            _write_status({"status": "noop", "total": len(log),
                           "at": datetime.now().isoformat(timespec="seconds")})
            return {"status": "noop", "added": 0, "total": len(log)}

        score_rows, fwd_rows = [], []
        for k, d in enumerate(todo, 1):
            d_fwd = _next_trading_day(d, horizon)
            sliced = {c: df.loc[df.index <= d] for c, df in raw.items() if df.index[0] <= d}
            try:
                preds = predictor.predict_batch(sliced)
            except Exception:  # noqa: BLE001
                preds = []
            sc = {p.code: float(p.score) for p in preds}
            c0 = close.loc[d]
            c1 = close.loc[d_fwd]
            valid = [c for c in raw if c in sc and c in c0 and c in c1
                     and pd.notna(c0[c]) and pd.notna(c1[c]) and c0[c] > 0]
            if len(valid) < min_stocks:
                continue
            rets = np.array([c1[c] / c0[c] - 1.0 for c in valid], dtype=float)
            excess = rets - rets.mean()
            for c, s, e in zip(valid, [sc[c] for c in valid], excess):
                score_rows.append({"date": d, "code": c, "score": s})
                fwd_rows.append({"date": d, "code": c, "excess": e})
            if progress_cb:
                progress_cb(len(codes), len(codes), k)

        if not score_rows:
            _write_status({"status": "error", "error": "no_valid_scores",
                           "at": datetime.now().isoformat(timespec="seconds")})
            return {"status": "error", "error": "no_valid_scores"}

        sp = pd.DataFrame(score_rows).pivot(index="date", columns="code", values="score")
        fr = pd.DataFrame(fwd_rows).pivot(index="date", columns="code", values="excess")

        new_entries = []
        for d in sp.index:
            s = sp.loc[d]
            r = fr.loc[d]
            v = pd.DataFrame({"s": s, "r": r}).dropna()
            if len(v) < min_stocks:
                continue
            ic = _spearman(v["s"].values, v["r"].values)
            if ic == ic:
                new_entries.append({"date": d.strftime("%Y-%m-%d"),
                                    "ic": round(float(ic), 4),
                                    "n": int(len(v)),
                                    "offline": False})

        merged = {r.get("date"): r for r in log if isinstance(r, dict)}
        for e in new_entries:
            merged[e["date"]] = e
        merged_list = sorted(merged.values(), key=lambda r: r.get("date", ""))[-250:]

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged_list, ensure_ascii=False, indent=2), encoding="utf-8")

        # 统计口径与 refresh_status 完全一致（共用一个函数，避免两个生产者
        # 各算一套、对不上账）。非有限值一律落成 None，别往 JSON 里写 NaN。
        result: Dict = {"status": "ok", "added": len(new_entries)}
        result.update(_stats_from_entries(merged_list))
        result["at"] = datetime.now().isoformat(timespec="seconds")
        _write_status(dict(result))
        return result
    except Exception as e:  # noqa: BLE001
        _write_status({"status": "error", "error": str(e),
                       "at": datetime.now().isoformat(timespec="seconds")})
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# 曲线渲染
# ---------------------------------------------------------------------------

def _rolling(seq, window):
    out = []
    for i in range(len(seq)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(seq[lo:i + 1])))
    return out


def _build_svg(points: List[Tuple[str, float, int]], baseline, window) -> str:
    W, H = 940, 460
    ml, mr, mt, mb = 64, 24, 40, 56
    plot_w = W - ml - mr
    plot_h = H - mt - mb
    n = len(points)
    if n == 0:
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">'
                '<rect width="100%%" height="100%%" fill="#0f1420"/>'
                '<text x="%d" y="%d" fill="#9aa4b2" font-size="18" text-anchor="middle">'
                '暂无 RankIC 数据（先跑回填或等收盘后自动任务）</text></svg>' % (W, H, W // 2, H // 2))
    ics = [p[1] for p in points]
    dates = [p[0] for p in points]
    x = [ml + (plot_w * i / (n - 1)) for i in range(n)]

    rmean = _rolling(ics, window)
    band = []
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = ics[lo:i + 1]
        sd = float(np.std(seg)) if len(seg) > 1 else 0.0
        band.append((float(np.mean(seg)) - sd, float(np.mean(seg)) + sd))
    ymin = min(0.0, *(b[0] for b in band), *ics, (baseline if baseline is not None else 0))
    ymax = max(0.0, *(b[1] for b in band), *ics, (baseline if baseline is not None else 0))
    pad = (ymax - ymin) * 0.12 or 0.05
    ymin -= pad
    ymax += pad

    def yv(v):
        return mt + plot_h * (1 - (v - ymin) / (ymax - ymin))

    m = float(np.mean(ics))
    sd = float(np.std(ics, ddof=1)) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n > 1 and sd == sd else 0.0
    ir = m / sd if sd > 0 else float("nan")
    tstat = m / se if se > 0 else float("nan")
    pos_rate = sum(1 for v in ics if v > 0) / n
    stat = (f'样本 {n} 点 ｜ 均值 {m:+.4f} ｜ 日频IR(mean/std) {ir:.2f} ｜ '
            f't统计量(mean/SE) {tstat:.2f} ｜ 正IC占比 {pos_rate*100:.0f}% ｜ 滚动窗口 {window}')

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                 f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Microsoft YaHei,sans-serif">')
    parts.append(f'<rect width="100%" height="100%" fill="#0f1420"/>')
    parts.append(f'<text x="{ml}" y="24" fill="#e6edf3" font-size="16" font-weight="700">RankIC 稳定性曲线</text>')
    parts.append(f'<text x="{ml}" y="40" fill="#9aa4b2" font-size="12">{stat}</text>')

    parts.append(f'<line x1="{ml}" y1="{yv(0):.1f}" x2="{ml+plot_w}" y2="{yv(0):.1f}" '
                 f'stroke="#3a4250" stroke-width="1"/>')
    parts.append(f'<text x="{ml-8}" y="{yv(0)+4:.1f}" fill="#6b7280" font-size="10" text-anchor="end">0</text>')
    if baseline is not None:
        yy = yv(baseline)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+plot_w}" y2="{yy:.1f}" '
                     f'stroke="#5b9dff" stroke-width="1" stroke-dasharray="6 4" opacity="0.7"/>')
        parts.append(f'<text x="{ml+plot_w}" y="{yy-4:.1f}" fill="#5b9dff" font-size="10" text-anchor="end">'
                     f'生产基线 {baseline:+.4f}</text>')

    if n > 1:
        up = " ".join(f"{x[i]:.1f},{yv(band[i][1]):.1f}" for i in range(n))
        dn = " ".join(f"{x[i]:.1f},{yv(band[i][0]):.1f}" for i in range(n - 1, -1, -1))
        parts.append(f'<polygon points="{up} {dn}" fill="#5b9dff" opacity="0.12"/>')
        rm = " ".join(f"{x[i]:.1f},{yv(rmean[i]):.1f}" for i in range(n))
        parts.append(f'<polyline points="{rm}" fill="none" stroke="#7cc4ff" stroke-width="2" opacity="0.9"/>')

    for i in range(n):
        col = "#3fb950" if ics[i] >= 0 else "#f85149"
        parts.append(f'<circle cx="{x[i]:.1f}" cy="{yv(ics[i]):.1f}" r="3.2" fill="{col}" opacity="0.85">'
                     f'<title>{dates[i]}: IC {ics[i]:+.4f} (n={points[i][2]})</title></circle>')

    xticks = max(1, n // min(n, 8))
    for i in range(0, n, xticks):
        parts.append(f'<text x="{x[i]:.1f}" y="{mt+plot_h+18:.1f}" fill="#6b7280" font-size="10" '
                     f'text-anchor="middle">{dates[i][5:]}</text>')

    lx, ly = ml + 8, mt + 14
    parts.append(f'<circle cx="{lx}" cy="{ly}" r="3.2" fill="#3fb950"/><text x="{lx+8}" y="{ly+4}" fill="#9aa4b2" font-size="10">IC&gt;0</text>')
    parts.append(f'<circle cx="{lx+60}" cy="{ly}" r="3.2" fill="#f85149"/><text x="{lx+68}" y="{ly+4}" fill="#9aa4b2" font-size="10">IC&lt;0</text>')
    parts.append(f'<line x1="{lx+118}" y1="{ly}" x2="{lx+138}" y2="{ly}" stroke="#7cc4ff" stroke-width="2"/>'
                 f'<text x="{lx+144}" y="{ly+4}" fill="#9aa4b2" font-size="10">滚动均值±σ</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def render_curve(log_path=None, metrics_path=None, window: int = 20) -> Tuple[str, Dict]:
    """读 rolling_ic_log.json 渲染 SVG，并返回统计 dict。"""
    log_path = Path(log_path) if log_path else IC_LOG
    metrics_path = Path(metrics_path) if metrics_path else PROD_METRICS
    log = _load_json(log_path) or []
    log = [r for r in log if isinstance(r, dict) and r.get("ic") == r.get("ic")]
    log.sort(key=lambda r: r.get("date", ""))
    points = [(r["date"], float(r["ic"]), int(r.get("n", 0))) for r in log]

    baseline = None
    m = _load_json(metrics_path)
    if m:
        b = (m.get("walk_forward") or {}).get("rank_ic")
        if b is not None:
            try:
                baseline = float(b)
            except Exception:
                baseline = None

    n = len(points)
    ics = [p[1] for p in points] if n else []
    m_ = float(np.mean(ics)) if n else float("nan")
    sd = float(np.std(ics, ddof=1)) if n > 1 else float("nan")
    se = sd / (n ** 0.5) if n > 1 and sd == sd else float("nan")
    stats = {
        "n_points": n,
        "mean_ic": round(m_, 4) if n else None,
        "std_daily": round(sd, 4) if n > 1 else None,
        "ir_mean_std": round(m_ / sd, 3) if (n > 1 and sd and sd > 0) else None,
        "t_stat": round(m_ / se, 2) if (n > 1 and se and se > 0) else None,
        "pos_rate": round(sum(1 for v in ics if v > 0) / n, 3) if n else None,
        "last_date": points[-1][0] if points else None,
        "window": max(2, min(window, n or window)),
    }
    svg = _build_svg(points, baseline, stats["window"])
    return svg, stats


def write_curve_files(out_html: Optional[Path] = None, out_svg: Optional[Path] = None,
                      window: int = 20, log_path=None, metrics_path=None) -> Dict:
    """渲染曲线并写 HTML+SVG 文件；返回 stats dict。供 CLI / 调度器复用。"""
    out_html = Path(out_html) if out_html else OUT_HTML
    out_svg = Path(out_svg) if out_svg else OUT_SVG
    svg, stats = render_curve(log_path, metrics_path, window)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text(svg, encoding="utf-8")
    if out_html:
        html = (
            "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
            "<title>RankIC 稳定性曲线</title>"
            "<style>body{background:#0b0f17;color:#e6edf3;font-family:-apple-system,"
            "Segoe UI,Microsoft YaHei,sans-serif;margin:24px}"
            "h2{font-weight:700}.sub{color:#9aa4b2;font-size:13px;margin-bottom:12px}</style>"
            "</head><body>"
            "<h2>📈 RankIC 稳定性曲线</h2>"
            "<div class='sub'>数据源：data/rolling_ic_log.json ｜ 由每日回填累积"
            "（逐日滚动、按评分日去重）</div>"
            f"{svg}"
            "</body></html>"
        )
        out_html.write_text(html, encoding="utf-8")
    return stats

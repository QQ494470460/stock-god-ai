# -*- coding: utf-8 -*-
"""RankIC 稳定性曲线生成器（离线，纯 SVG/HTML，无第三方依赖）。

读取 ``data/rolling_ic_log.json``（由 ``paper_shadow_audit.py --live-ic`` 或
``rolling_ic_backfill.py`` 累积），画一张「逐日 RankIC 稳定性曲线」：

  · 每日 IC 散点（正绿 / 负红）
  · 0 基准线
  · 生产基线 RankIC（来自 production_metrics.json，虚线）
  · 滚动均值线（窗口可配，默认 20 点）
  · 滚动均值 ±1σ 阴影带（直观看稳定性）

用法：
    python scripts/rolling_ic_curve.py                 # 默认读 data/rolling_ic_log.json
    python scripts/rolling_ic_curve.py --window 10     # 滚动窗口 10 点
    python scripts/rolling_ic_curve.py --out out.html  # 指定输出
    python scripts/rolling_ic_curve.py --no-html       # 只写 SVG

输出：
    data/rolling_ic_curve.html  （内联 SVG，浏览器/看板可直接看）
    data/rolling_ic_curve.svg   （同一张图，便于嵌入文档）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IC_LOG = ROOT / "data" / "rolling_ic_log.json"
PROD_METRICS = ROOT / "models" / "production_metrics.json"
OUT_HTML = ROOT / "data" / "rolling_ic_curve.html"
OUT_SVG = ROOT / "data" / "rolling_ic_curve.svg"


def _load_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _rolling(seq, window):
    out = []
    for i in range(len(seq)):
        lo = max(0, i - window + 1)
        out.append(mean(seq[lo:i + 1]))
    return out


def _build_svg(points, baseline, window):
    """points: list[(date_str, ic, n)]，按日期升序。返回 SVG 字符串。"""
    W, H = 940, 460
    ml, mr, mt, mb = 64, 24, 40, 56
    plot_w = W - ml - mr
    plot_h = H - mt - mb

    n = len(points)
    if n == 0:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">'
                f'<rect width="100%" height="100%" fill="#0f1420"/>'
                f'<text x="{W//2}" y="{H//2}" fill="#9aa4b2" font-size="18" '
                f'text-anchor="middle">暂无 RankIC 数据（先跑 --live-ic 或 rolling_ic_backfill）</text>'
                f'</svg>')

    ics = [p[1] for p in points]
    dates = [p[0] for p in points]
    x = [ml + (plot_w * i / (n - 1)) for i in range(n)]

    # Y 轴范围：容纳 IC、滚动均值±σ、0、基线
    rmean = _rolling(ics, window)
    band = []
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = ics[lo:i + 1]
        sd = pstdev(seg) if len(seg) > 1 else 0.0
        band.append((mean(seg) - sd, mean(seg) + sd))
    ymin = min(0.0, *(b[0] for b in band), *(ics), (baseline if baseline is not None else 0))
    ymax = max(0.0, *(b[1] for b in band), *(ics), (baseline if baseline is not None else 0))
    pad = (ymax - ymin) * 0.12 or 0.05
    ymin -= pad
    ymax += pad

    def yv(v):
        return mt + plot_h * (1 - (v - ymin) / (ymax - ymin))

    # ---- 统计 ----
    # 注意口径：这里 ic_ir = mean/std（Grinold-Kahn 信息比率，反映日频稳定性）。
    # production_metrics.json 那个 "ic_ir=14" 是 mean/SE（t 统计量），口径不同，别混。
    ic_ir = (mean(ics) / pstdev(ics)) if n > 1 and pstdev(ics) > 0 else float("nan")
    se = pstdev(ics) / (n ** 0.5) if n > 1 and pstdev(ics) > 0 else float("nan")
    tstat = (mean(ics) / se) if se and se > 0 else float("nan")
    pos_rate = sum(1 for v in ics if v > 0) / n

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                 f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Microsoft YaHei,sans-serif">')
    parts.append(f'<rect width="100%" height="100%" fill="#0f1420"/>')

    # 标题 + 统计
    parts.append(f'<text x="{ml}" y="24" fill="#e6edf3" font-size="16" font-weight="700">'
                 f'RankIC 稳定性曲线</text>')
    stat = (f'样本 {n} 点 ｜ 均值 {mean(ics):+.4f} ｜ 日频IR(mean/std) {ic_ir:.2f} ｜ '
            f't统计量(mean/SE) {tstat:.2f} ｜ 正 IC 占比 {pos_rate*100:.0f}% ｜ 滚动窗口 {window}')
    parts.append(f'<text x="{ml}" y="40" fill="#9aa4b2" font-size="12">{stat}</text>')

    # 网格 + Y 轴刻度（0 与基线）
    for gv in (0.0,):
        yy = yv(gv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+plot_w}" y2="{yy:.1f}" '
                     f'stroke="#3a4250" stroke-width="1"/>')
        parts.append(f'<text x="{ml-8}" y="{yy+4:.1f}" fill="#6b7280" font-size="10" '
                     f'text-anchor="end">0</text>')
    if baseline is not None:
        yy = yv(baseline)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+plot_w}" y2="{yy:.1f}" '
                     f'stroke="#5b9dff" stroke-width="1" stroke-dasharray="6 4" opacity="0.7"/>')
        parts.append(f'<text x="{ml+plot_w}" y="{yy-4:.1f}" fill="#5b9dff" font-size="10" '
                     f'text-anchor="end">生产基线 {baseline:+.4f}</text>')

    # ±1σ 阴影带（多边形）
    if n > 1:
        up = " ".join(f"{x[i]:.1f},{yv(band[i][1]):.1f}" for i in range(n))
        dn = " ".join(f"{x[i]:.1f},{yv(band[i][0]):.1f}" for i in range(n - 1, -1, -1))
        parts.append(f'<polygon points="{up} {dn}" fill="#5b9dff" opacity="0.12"/>')
        # 滚动均值线
        rm = " ".join(f"{x[i]:.1f},{yv(rmean[i]):.1f}" for i in range(n))
        parts.append(f'<polyline points="{rm}" fill="none" stroke="#7cc4ff" '
                     f'stroke-width="2" opacity="0.9"/>')

    # 每日 IC 散点
    for i in range(n):
        col = "#3fb950" if ics[i] >= 0 else "#f85149"
        parts.append(f'<circle cx="{x[i]:.1f}" cy="{yv(ics[i]):.1f}" r="3.2" fill="{col}" '
                     f'opacity="0.85"><title>{dates[i]}: IC {ics[i]:+.4f} (n={points[i][2]})</title></circle>')

    # X 轴标签（最多 ~8 个）
    xticks = max(1, n // min(n, 8))
    for i in range(0, n, xticks):
        parts.append(f'<text x="{x[i]:.1f}" y="{mt+plot_h+18:.1f}" fill="#6b7280" '
                     f'font-size="10" text-anchor="middle">{dates[i][5:]}</text>')

    # 图例
    lx = ml + 8
    ly = mt + 14
    parts.append(f'<circle cx="{lx}" cy="{ly}" r="3.2" fill="#3fb950"/>'
                 f'<text x="{lx+8}" y="{ly+4}" fill="#9aa4b2" font-size="10">IC&gt;0</text>')
    parts.append(f'<circle cx="{lx+60}" cy="{ly}" r="3.2" fill="#f85149"/>'
                 f'<text x="{lx+68}" y="{ly+4}" fill="#9aa4b2" font-size="10">IC&lt;0</text>')
    parts.append(f'<line x1="{lx+118}" y1="{ly}" x2="{lx+138}" y2="{ly}" stroke="#7cc4ff" stroke-width="2"/>'
                 f'<text x="{lx+144}" y="{ly+4}" fill="#9aa4b2" font-size="10">滚动均值±σ</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="RankIC 稳定性曲线生成器")
    ap.add_argument("--log", type=Path, default=IC_LOG)
    ap.add_argument("--metrics", type=Path, default=PROD_METRICS)
    ap.add_argument("--out", type=Path, default=OUT_HTML, help="HTML 输出")
    ap.add_argument("--svg", type=Path, default=OUT_SVG, help="SVG 输出")
    ap.add_argument("--window", type=int, default=20, help="滚动均值窗口（点）")
    ap.add_argument("--no-html", action="store_true", help="只写 SVG")
    args = ap.parse_args()

    log = _load_json(args.log) or []
    log = [r for r in log if isinstance(r, dict) and r.get("ic") == r.get("ic")]
    log.sort(key=lambda r: r.get("date", ""))
    points = [(r["date"], float(r["ic"]), int(r.get("n", 0))) for r in log]

    baseline = None
    m = _load_json(args.metrics)
    if m:
        baseline = (m.get("walk_forward") or {}).get("rank_ic")
        if baseline is not None:
            try:
                baseline = float(baseline)
            except Exception:
                baseline = None

    window = max(2, min(args.window, len(points) or args.window))
    svg = _build_svg(points, baseline, window)

    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(svg, encoding="utf-8")
    print(f"✅ SVG 已写入: {args.svg}（{len(points)} 点，窗口 {window}）")

    if not args.no_html:
        html = (
            "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
            "<title>RankIC 稳定性曲线</title>"
            "<style>body{background:#0b0f17;color:#e6edf3;font-family:-apple-system,"
            "Segoe UI,Microsoft YaHei,sans-serif;margin:24px}"
            "h2{font-weight:700}.sub{color:#9aa4b2;font-size:13px;margin-bottom:12px}</style>"
            "</head><body>"
            "<h2>📈 RankIC 稳定性曲线</h2>"
            "<div class='sub'>数据源：data/rolling_ic_log.json ｜ 由 --live-ic / rolling_ic_backfill 累积"
            "（逐日滚动、按评分日去重）</div>"
            f"{svg}"
            "</body></html>"
        )
        args.out.write_text(html, encoding="utf-8")
        print(f"✅ HTML 已写入: {args.out}")

    if not points:
        print("⚠️ 日志为空或无有效 IC 点，曲线仅显示占位。先跑每日 --live-ic 或 rolling_ic_backfill。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

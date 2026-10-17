# -*- coding: utf-8 -*-
"""RankIC 稳定性曲线生成器 CLI —— 薄壳，核心在 ``src/core/rolling_ic.py``。

读取 ``data/rolling_ic_log.json``（由 ``rolling_ic_backfill.py`` 累积），画一张
「逐日 RankIC 稳定性曲线」：

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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.rolling_ic import (
    IC_LOG, OUT_HTML, OUT_SVG, PROD_METRICS,
    render_curve, write_curve_files,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="RankIC 稳定性曲线生成器")
    ap.add_argument("--log", type=Path, default=IC_LOG)
    ap.add_argument("--metrics", type=Path, default=PROD_METRICS)
    ap.add_argument("--out", type=Path, default=OUT_HTML, help="HTML 输出")
    ap.add_argument("--svg", type=Path, default=OUT_SVG, help="SVG 输出")
    ap.add_argument("--window", type=int, default=20, help="滚动均值窗口（点）")
    ap.add_argument("--no-html", action="store_true", help="只写 SVG")
    args = ap.parse_args()

    if args.no_html:
        svg, stats = render_curve(args.log, args.metrics, args.window)
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(svg, encoding="utf-8")
        print(f"✅ SVG 已写入: {args.svg}（{stats['n_points']} 点，窗口 {stats['window']}）")
    else:
        stats = write_curve_files(args.out, args.svg, args.window, args.log, args.metrics)
        print(f"✅ HTML: {args.out} ｜ SVG: {args.svg}（{stats['n_points']} 点，窗口 {stats['window']}）")

    if not stats["n_points"]:
        print("⚠️ 日志为空或无有效 IC 点，曲线仅显示占位。先跑每日 --live-ic 或 rolling_ic_backfill。")
    else:
        print(f"📊 均值 {stats['mean_ic']:+.4f} ｜ 日频IR {stats['ir_mean_std']} ｜ "
              f"t统计量 {stats['t_stat']} ｜ 正IC占比 {stats['pos_rate']*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

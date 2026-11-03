# -*- coding: utf-8 -*-
"""调仓相位的敏感性扫描 —— 检验「某个结论是不是相位运气」。

为什么必须有这个工具
--------------------
11-04 挖第 15 处漂移时的意外发现：**同一个策略，只把调仓日整体平移几天，
1140 天年化就在 12.7% ~ 29.4% 之间跳**（基准 c15 扫 15 个相位，均值 21.9%）。
也就是说此前所有「A 比 B 好 2pp」的结论，都可能只是 A 抽到了顺手的相位。

更具体的坑：回撤熔断只在深回撤时触发。如果那几天恰好没有调仓日，
这条熔断线的历史成本就是 0 —— 看起来「免费」，其实是没撞上。
实测 20% 线在 offset=0 时冻结了 2 天、零成本；扫到别的相位就会撞上调仓日，
最差相位 −2.23pp。

用法
----
    python scripts/exec_phase.py                       # 基准 + 生产风控，15 相位，10bp
    python scripts/exec_phase.py --only c15,prod_risk
    python scripts/exec_phase.py --slippage-bp 50 --phases 15

判定标准
--------
不看单点，看三件事：
  · **均值**：期望成本/收益（真实决策依据）
  · **最差相位**：尾部风险（策略不能只在好相位上活着）
  · **负相位个数**：结论稳不稳（若 6/15 相位为负、9/15 为 0，说明是纯尾部成本）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.core.exec_backtest import build_ohlc_panel  # noqa: E402
from scripts.exec_ab import (  # noqa: E402
    BASE, PHASE_SPREAD_PP, SCENARIOS, fmt_t, needs_industry_map, summarise,
    sweep_phases,
)

DEFAULT_CASES = ["c15", "prod_halt20", "prod_halt20_m2", "prod_risk"]

# 生产口径的硬约束：显式写死，不能依赖 SCENARIOS 里每个方案都记得覆盖。
# ``ExecConfig.hold_rank_slack`` 默认 1.0 而生产是 3.0，漏传会把结论正反颠倒
# （见 memory §11）。tests/test_loop.py 有静态守卫盯着这一行。
PROD_BASE = dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="调仓相位敏感性扫描")
    ap.add_argument("--panel", default="data/evolution/panel_oos")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--slippage-bp", type=float, default=10.0)
    ap.add_argument("--phases", type=int, default=15, help="扫描的相位个数（=rebalance_days）")
    ap.add_argument("--only", default="", help="逗号分隔的方案名（默认 c15 + 生产风控四件套）")
    ap.add_argument("--industry-map", default="",
                    help="行业映射 JSON 路径。中性化/行业上限类方案必须给，"
                         "否则会被跳过（否则会静默退化成「没打开」）")
    ap.add_argument("--baseline", default="c15", help="作为基准相减的方案名")
    ap.add_argument("--quick", action="store_true", help="只跑最近 400 个交易日")
    args = ap.parse_args()

    panel = Path(args.panel)
    scores = pd.read_parquet(panel / "score_panel.parquet")
    codes = list(scores.columns)
    ohlc = build_ohlc_panel(codes, str(scores.index.min().date()),
                            str(scores.index.max().date()), args.data_dir)
    if not ohlc or ohlc["close"].empty:
        print("❌ 没有可用 OHLC 数据")
        return 1
    if args.quick:
        keep = scores.index[-400:]
        scores = scores.loc[keep]
        ohlc = {k: v.loc[v.index.isin(keep)] for k, v in ohlc.items()}

    names = [n for n in (args.only.split(",") if args.only else DEFAULT_CASES) if n]
    missing = [n for n in names if n not in SCENARIOS]
    if missing:
        print(f"⚠️ 未知方案 {missing}")
        return 1
    # 基准必须在样本里，否则算不了逐相位差
    if args.baseline not in names:
        names.insert(0, args.baseline)

    industry_map = None
    if args.industry_map:
        import json
        mp = Path(args.industry_map)
        if not mp.exists():
            print(f"❌ 行业映射不存在：{mp}")
            return 1
        industry_map = json.loads(mp.read_text(encoding="utf-8"))
        print(f"🏭 行业映射 {len(industry_map)} 条")

    print(f"📦 面板 {scores.shape[0]} 天 × {len(codes)} 只 ｜ 滑点 {args.slippage_bp:.0f}bp ｜ "
          f"扫 {args.phases} 个相位")

    acc: dict[str, list[dict]] = {}
    for name in names:
        over = dict(BASE)
        over.update(PROD_BASE)
        over.update(SCENARIOS[name])
        if industry_map:
            over["industry_map"] = industry_map
        if needs_industry_map(over):
            print(f"⚠️ 跳过 {name}：需要 --industry-map（否则中性化/行业上限静默失效）")
            continue
        acc[name] = sweep_phases(scores, ohlc, over,
                                 slippage_bp=args.slippage_bp, phases=args.phases)
        s = summarise(acc[name])
        print(f"  ✓ {name:<16} 年化均值 {s['ann_mean']:>7.2f}%  "
              f"最差 {s['ann_min']:>7.2f}%  夏普 {s['sharpe_mean']:>5.3f}")
    names = [n for n in names if n in acc]
    if args.baseline not in acc:
        print(f"❌ 基准 {args.baseline} 没跑成，无法算逐相位差")
        return 1

    print("\n" + "=" * 92)
    print(f"{'方案':<18}{'年化均值':>10}{'最差':>9}{'最好':>9}{'夏普均值':>10}"
          f"{'夏普最差':>10}{'回撤均值':>10}")
    print("-" * 92)
    for name in names:
        s = summarise(acc[name])
        print(f"{name:<18}{s['ann_mean']:>9.2f}%{s['ann_min']:>8.2f}%{s['ann_max']:>8.2f}%"
              f"{s['sharpe_mean']:>10.3f}{s['sharpe_min']:>10.3f}{s['dd_mean']:>9.2f}%")

    base = acc[args.baseline]
    print(f"\n相对基准【{args.baseline}】的逐相位差（年化 pp / 夏普）：")
    print(f"{'方案':<18}{'年化均值':>10}{'最差':>9}{'最好':>9}{'负相位':>9}{'配对t':>8}{'夏普均值':>10}")
    print("-" * 92)
    for name in names:
        if name == args.baseline:
            continue
        s = summarise(acc[name], base)
        print(f"{name:<18}{s['d_ann_mean']:>+9.2f}pp{s['d_ann_min']:>+8.2f}pp"
              f"{s['d_ann_max']:>+8.2f}pp"
              f"{s['neg_phases']:>6}/{args.phases}{fmt_t(s['t']):>9}"
              f"{s['d_sharpe_mean']:>+10.3f}")

    spread = [float(r["metrics"]["ann_return"]) * 100 for r in base]
    print(f"\n⚠️ 基准自身的相位离散度：{max(spread) - min(spread):.2f}pp 年化"
          f"（{min(spread):.2f}% ~ {max(spread):.2f}%）")
    print("   任何「A 比 B 好 X pp」的结论，若 X 小于这个离散度，都必须用相位均值说话。")
    if abs((max(spread) - min(spread)) - PHASE_SPREAD_PP) > 1.0:
        print(f"   （注意：本次离散度与 11-04 记录的 {PHASE_SPREAD_PP:.1f}pp 差超过 1pp，"
              "面板或口径可能变了，先查清再引用结论。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

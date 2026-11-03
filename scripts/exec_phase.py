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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.core.exec_backtest import (  # noqa: E402
    ExecConfig, build_ohlc_panel, run_exec_backtest,
)
from scripts.exec_ab import BASE, SCENARIOS  # noqa: E402

DEFAULT_CASES = ["c15", "prod_halt20", "prod_halt20_m2", "prod_risk"]

# 生产口径的硬约束：显式写死，不能依赖 SCENARIOS 里每个方案都记得覆盖。
# ``ExecConfig.hold_rank_slack`` 默认 1.0 而生产是 3.0，漏传会把结论正反颠倒
# （见 memory §11）。tests/test_loop.py 有静态守卫盯着这一行。
PROD_BASE = dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0)


def _fmt(v: float, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v*100:+.2f}%" if pct else f"{v:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="调仓相位敏感性扫描")
    ap.add_argument("--panel", default="data/evolution/panel_oos")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--slippage-bp", type=float, default=10.0)
    ap.add_argument("--phases", type=int, default=15, help="扫描的相位个数（=rebalance_days）")
    ap.add_argument("--only", default="", help="逗号分隔的方案名（默认 c15 + 生产风控四件套）")
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

    print(f"📦 面板 {scores.shape[0]} 天 × {len(codes)} 只 ｜ 滑点 {args.slippage_bp:.0f}bp ｜ "
          f"扫 {args.phases} 个相位")

    acc: dict[str, list[tuple[float, float, float, float]]] = {n: [] for n in names}
    for off in range(args.phases):
        for name in names:
            over = dict(BASE)
            over.update(PROD_BASE)
            over.update(SCENARIOS[name])
            m = run_exec_backtest(
                scores, ohlc,
                ExecConfig(slippage_bp=args.slippage_bp, rebalance_offset=off, **over),
            ).metrics
            acc[name].append((m["ann_return"] * 100, m["sharpe"],
                              m["max_drawdown"] * 100, m["total_return"]))

    print("\n" + "=" * 92)
    print(f"{'方案':<18}{'年化均值':>10}{'最差':>9}{'最好':>9}{'夏普均值':>10}"
          f"{'夏普最差':>10}{'回撤均值':>10}")
    print("-" * 92)
    for name in names:
        a = np.array(acc[name])
        print(f"{name:<18}{a[:, 0].mean():>9.2f}%{a[:, 0].min():>8.2f}%{a[:, 0].max():>8.2f}%"
              f"{a[:, 1].mean():>10.3f}{a[:, 1].min():>10.3f}{a[:, 2].mean():>9.2f}%")

    base = np.array(acc[args.baseline])
    print(f"\n相对基准【{args.baseline}】的逐相位差（年化 pp / 夏普）：")
    print(f"{'方案':<18}{'年化均值':>10}{'最差':>9}{'最好':>9}{'负相位':>9}{'配对t':>8}{'夏普均值':>10}")
    print("-" * 92)
    for name in names:
        if name == args.baseline:
            continue
        a = np.array(acc[name])
        d = a[:, 0] - base[:, 0]
        ds = a[:, 1] - base[:, 1]
        # 配对 t：同一相位下两方案相减，消掉「哪个相位本身好坏」这个共同项。
        # 注意 15 个相位是重叠样本、不算独立，t 值只用来区分「系统性差异」
        # 和「方向随机的噪声」，不当成严格显著性检验。
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        t = float(np.mean(d) / (sd / np.sqrt(len(d)))) if sd > 1e-12 else float("nan")
        print(f"{name:<18}{d.mean():>+9.2f}pp{d.min():>+8.2f}pp{d.max():>+8.2f}pp"
              f"{int((d < -0.01).sum()):>6}/{args.phases}{t:>8.2f}{ds.mean():>+10.3f}")

    spread = np.array(acc[args.baseline])[:, 0]
    print(f"\n⚠️ 基准自身的相位离散度：{spread.max() - spread.min():.2f}pp 年化"
          f"（{spread.min():.2f}% ~ {spread.max():.2f}%）")
    print("   任何「A 比 B 好 X pp」的结论，若 X 小于这个离散度，都必须用相位均值说话。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

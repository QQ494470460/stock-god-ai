# -*- coding: utf-8 -*-
"""验证 T3 非顺周期风控叠加 vs 权威 exec_backtest。

⚠️ 血的教训（2026-11-03）：本脚本最初漏传 ``hold_rank_slack``，于是跑在
``ExecConfig`` 的默认值 1.0 上，而不是生产的 3.0。同一批数据、同一个策略，
只换这一个参数，结论完全反转（slack1.0 下 vt22 是正贡献，slack3.0 下是负贡献）。
原因：「波动率目标」与「排名滞后」是**替代关系**，都在降换手/降回撤，叠加重复扣分。
现在 BASE 里钉死 ``hold_rank_slack=3.0``，与 ``TradingLoopConfig`` 保持一致。

对比方案（三档成本 10/30/50bp 下的 年化/夏普/回撤/Calmar/换手）：
  A 基线      : 生产 c15（不止损 + 15 日 + slack3，无风险叠加）← **采纳**
  B 顺周期    : dd_scale_floor=0.30, at=8%, start=0（历史吃掉收益那种）
  C 非顺周期  : dd_scale_floor=0.60, at=20%, start=12
  D 波动率目标: vol_target_pct=22, floor=0.5
  E C+D 叠加  : C 与 D 同时开启

用法：
    python scripts/validate_risk_overlay.py --quick     # 最近 400 天，快速看信号
    python scripts/validate_risk_overlay.py             # 全 1140 天
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.core.exec_backtest import ExecConfig, build_ohlc_panel, run_exec_backtest  # noqa: E402

BASE = dict(
    top_n=8, rebalance_days=15, weighting="risk_parity",
    cash_reserve_pct=0.10, band_pct=0.15, weight_cap=0.30, stop_mode="none",
    # ⚠️ 必须与 TradingLoopConfig.hold_rank_slack 一致（生产 3.0）。
    # 漏这一行 = 全套结论作废：slack1.0 与 3.0 的结论会正反颠倒。
    hold_rank_slack=3.0,
)

SCEN = {
    "A_baseline":       dict(),
    "B_dd_procyclical": dict(dd_scale_floor=0.30, dd_scale_at_pct=8.0, dd_start_cut_pct=0.0),
    "C_dd_noncyc":      dict(dd_scale_floor=0.60, dd_scale_at_pct=20.0, dd_start_cut_pct=12.0),
    "D_voltarget":      dict(vol_target_pct=22.0, vol_scale_floor=0.5),
    "E_C_plus_D":       dict(dd_scale_floor=0.60, dd_scale_at_pct=20.0, dd_start_cut_pct=12.0,
                             vol_target_pct=22.0, vol_scale_floor=0.5),
}


def _fmt(v, pct: bool = False, d: int = 2) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v*100:+.{d}f}%" if pct else f"{v:.{d}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/evolution/panel_oos")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--quick", action="store_true", help="只跑最近 400 个交易日")
    args = ap.parse_args()

    panel = Path(args.panel)
    scores = pd.read_parquet(panel / "score_panel.parquet")
    codes = list(scores.columns)
    start, end = scores.index.min(), scores.index.max()
    print(f"📦 面板 {scores.shape[0]} 天 × {len(codes)} 只  {start.date()} → {end.date()}")
    ohlc = build_ohlc_panel(codes, str(start.date()), str(end.date()), args.data_dir)
    if not ohlc or ohlc["close"].empty:
        print("❌ 无可用 OHLC")
        return 1
    if args.quick:
        keep = scores.index[-400:]
        scores = scores.loc[keep]
        ohlc = {k: v.loc[v.index.isin(keep)] for k, v in ohlc.items()}
        print(f"⚡ quick：最近 {len(keep)} 个交易日")

    for bp in (10.0, 30.0, 50.0):
        print(f"\n=== 滑点 {bp:.0f}bp/边 ｜ 调仓15日 ｜ TopN8 ｜ risk_parity ｜ cash10% ===")
        hdr = f"{'方案':<20}{'年化':>10}{'夏普':>8}{'回撤':>10}{'Calmar':>9}{'年换手':>9}"
        print(hdr)
        print("-" * 66)
        for name, over in SCEN.items():
            cfg = ExecConfig(slippage_bp=bp, **{**BASE, **over})
            m = run_exec_backtest(scores, ohlc, cfg).metrics
            print(f"{name:<20}"
                  f"{_fmt(m.get('ann_return'), True):>10}"
                  f"{_fmt(m.get('sharpe')):>8}"
                  f"{_fmt(m.get('max_drawdown'), True):>10}"
                  f"{_fmt(m.get('calmar')):>9}"
                  f"{_fmt(m.get('turnover_per_year'), False, 1):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

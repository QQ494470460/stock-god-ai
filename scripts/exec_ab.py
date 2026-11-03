# -*- coding: utf-8 -*-
"""执行级 A/B：把实盘的退出规则逐条搬进回测，看谁真的在赚钱、谁在毁灭收益。

用法：
    python scripts/exec_ab.py                        # 默认面板 + 10bp 滑点，跑全部方案
    python scripts/exec_ab.py --slippage-bp 50       # 成本敏感性
    python scripts/exec_ab.py --only live,no_exit    # 只跑指定方案
    python scripts/exec_ab.py --quick                # 只跑最近 400 个交易日（迭代用）

**为什么要有这个脚本**：实盘 −6.75%/7 天、盈亏比 0.153、10 连亏，而样本外面板
RankIC .061 / 胜率 61.5%。两个数字不可能同时对 —— 差异只可能来自执行层：
止损把波动实现成了亏损。这个脚本就是用来证明/证伪这句话的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.core.exec_backtest import (  # noqa: E402
    ExecConfig, build_ohlc_panel, run_exec_backtest,
)

# 生产默认（真·样本外验证过的那组）
BASE = dict(
    top_n=8, rebalance_days=5, weighting="risk_parity",
    cash_reserve_pct=0.10, band_pct=0.15, weight_cap=0.30,
)

# 方案表：名字 → ExecConfig 覆盖项
# 第一轮结论（1140 天，10bp）：止损单独用有益（回撤 -20.6%→-16.7%），
# 止盈 +15% + 移动止损 -3% 是灾难（年化 -27.6%、胜率 13.9%、持有 1.8 天）。
# 第二轮起围绕「去掉止盈/移动止损 + 降换手 + 择时 + 信号门槛」做组合搜索。
SCENARIOS: dict[str, dict] = {
    # ---- 对照组 ----
    "no_exit":        dict(stop_mode="none"),
    "live_now":       dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           take_profit_pct=15.0, trailing_stop_pct=-3.0),
    "only_stop":      dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0),
    # ---- 换手控制：排名滞后（降成本敏感性的决定性杠杆）----
    "slack15":        dict(stop_mode="none", hold_rank_slack=1.5),
    "slack20":        dict(stop_mode="none", hold_rank_slack=2.0),
    "slack30":        dict(stop_mode="none", hold_rank_slack=3.0),
    "slack20_stop":   dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           hold_rank_slack=2.0),
    # ---- 信号强度门槛（z 分数）----
    "slack20_z15":    dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           hold_rank_slack=2.0, min_score_z=1.5),
    "slack20_gate60": dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           hold_rank_slack=2.0, market_gate="ma60", market_gate_scale=0.5),
    # ---- 最终候选：15 日周期 + 排名滞后 3.0（网格里唯一三档成本都站得住的格子）----
    "c15":            dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0),
    "c15_stop":       dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           rebalance_days=15, hold_rank_slack=3.0),
    "c15_fix20":      dict(stop_mode="fixed", stop_loss_pct=-20.0,
                           rebalance_days=15, hold_rank_slack=3.0),
    "c15_fix12":      dict(stop_mode="fixed", stop_loss_pct=-12.0,
                           rebalance_days=15, hold_rank_slack=3.0),
    "c15_budget":     dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           risk_budget_pct=0.005, band_mode="rel", band_pct=0.35,
                           rebalance_days=15, hold_rank_slack=3.0),
    # ---- 风险预算阶梯（不止损 + 15 日 + slack3）：单票上限 = rb / 15% ----
    "c15_rb075":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           risk_budget_pct=0.0075, band_mode="rel", band_pct=0.35),
    "c15_rb100":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           risk_budget_pct=0.010, band_mode="rel", band_pct=0.35),
    "c15_rb150":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           risk_budget_pct=0.015, band_mode="rel", band_pct=0.35),
    # ---- 波动率目标（组合层压回撤）----
    "c15_vol15":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           vol_target_pct=15.0),
    "c15_vol20":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           vol_target_pct=20.0),
    "c15_vol25":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           vol_target_pct=25.0),
    # ---- 回撤闸门（第六处口径漂移：实盘两层「越跌越轻」，回测一个都没有）----
    # 基准 c15 = 不止损 + 15 日周期 + 排名滞后 3.0（唯一三档成本都站得住的格子）。
    # dd_live 复现实盘现状：渐进降仓 floor 0.30 + 8% 熔断冻结整轮调仓。
    "dd_live":        dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           dd_scale_floor=0.30, dd_scale_at_pct=8.0, halt_dd_pct=8.0),
    "dd_scale":       dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           dd_scale_floor=0.30, dd_scale_at_pct=8.0),
    "dd_scale50":     dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           dd_scale_floor=0.50, dd_scale_at_pct=8.0),
    "dd_halt8":       dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=8.0),
    "dd_halt12":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=12.0),
    "dd_halt15":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=15.0),
    "dd_halt20":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=20.0),
    # ---- 熔断自愈的完整生命周期（第 15 处口径漂移）----
    # 生产现状的四件套：回撤熔断 20% + 解冻滞后带 2% + 自愈后 2 日半仓冷静期
    # + 单日亏损熔断 5%。基准 = c15（不止损 + 15 日 + slack3）。
    # ⚠️ halt_dd_pct=20 不是「理论上才有」的闸门：策略最大回撤 -25.5% ⇒
    #    1140 天里必然撞线。以前所有「回测没有这条线」的结论都默认它零成本，未经验证。
    "prod_risk":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=20.0, resume_dd_margin_pct=2.0,
                           resume_throttle_days=2, resume_risk_fraction=0.5,
                           daily_halt_pct=5.0),
    "prod_halt20":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=20.0),
    "prod_halt20_m2": dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=20.0, resume_dd_margin_pct=2.0),
    "prod_throttle":  dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=20.0, resume_throttle_days=2,
                           resume_risk_fraction=0.5),
    "prod_thr_1d":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           daily_halt_pct=5.0, resume_throttle_days=1,
                           resume_risk_fraction=0.5),
    "prod_thr_3d":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           daily_halt_pct=5.0, resume_throttle_days=3,
                           resume_risk_fraction=0.5),
    "prod_thr_f75":   dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           daily_halt_pct=5.0, resume_throttle_days=2,
                           resume_risk_fraction=0.75),
    "prod_daily5":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           daily_halt_pct=5.0),
    "prod_halt25":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=25.0, resume_dd_margin_pct=2.0,
                           resume_throttle_days=2, resume_risk_fraction=0.5,
                           daily_halt_pct=5.0),
    "prod_halt30":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=30.0, resume_dd_margin_pct=2.0,
                           resume_throttle_days=2, resume_risk_fraction=0.5,
                           daily_halt_pct=5.0),
    # ---- 相位平均复核用（11-04 发现单相位结论会被 16.6pp 的相位噪声淹没）----
    # 这三组当初都是在 offset=0 上单点选出来的，差值只有 1~4pp，必须用相位均值重判。
    "tilt10":         dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           tilt_strength=0.10),
    "tilt25":         dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           tilt_strength=0.25),
    "cash00":         dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           cash_reserve_pct=0.0),
    "cash05":         dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           cash_reserve_pct=0.05),
    "cash15":         dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           cash_reserve_pct=0.15),
}


def _fmt(v: float, pct: bool = False, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v*100:+.{digits}f}%" if pct else f"{v:.{digits}f}"


def _run_grid(scores, ohlc, args) -> None:
    """决定性网格：持有周期 × 排名滞后 × 成本三档。

    5 天周期下每期边际收益约 0.5%，一次往返成本在 50bp 假设下约 1.1% ——
    结构上必亏。这个网格回答「周期要拉到多长，edge 才能覆盖成本」。
    """
    print("\n【网格】不止损（no_exit）｜行=持有周期，列=排名滞后，单元格=50bp 下 夏普/年化")
    for bp in (10.0, 30.0, 50.0):
        print(f"\n— 滑点 {bp:.0f}bp/边 " + "—" * 62)
        print(f"{'周期':<8}" + "".join(f"{'slack=' + str(s):>20}" for s in (1.0, 1.5, 2.0, 3.0)))
        for rb in (5, 10, 15, 20, 30):
            cells = []
            for sl in (1.0, 1.5, 2.0, 3.0):
                over = dict(BASE)
                over.update(dict(stop_mode="none", rebalance_days=rb, hold_rank_slack=sl))
                m = run_exec_backtest(scores, ohlc,
                                      ExecConfig(slippage_bp=bp, **over)).metrics
                cells.append(f"{_fmt(m.get('sharpe')):>7}/{_fmt(m.get('ann_return'), True):>9}"
                             f"{'':>4}")
            print(f"{rb:<8}" + "".join(f"{c:>20}" for c in cells))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/evolution/panel_oos")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--slippage-bp", type=float, default=10.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true", help="只跑最近 400 个交易日")
    ap.add_argument("--top-n", type=int, default=0)
    ap.add_argument("--grid", action="store_true", help="跑持有周期×排名滞后网格")
    args = ap.parse_args()

    panel = Path(args.panel)
    scores = pd.read_parquet(panel / "score_panel.parquet")
    codes = list(scores.columns)
    start, end = scores.index.min(), scores.index.max()
    print(f"📦 面板 {scores.shape[0]} 天 × {len(codes)} 只  {start.date()} → {end.date()}")

    ohlc = build_ohlc_panel(codes, str(start.date()), str(end.date()), args.data_dir)
    if not ohlc or ohlc["close"].empty:
        print("❌ 没有可用 OHLC 数据")
        return 1

    if args.quick:
        keep = scores.index[-400:]
        scores = scores.loc[keep]
        ohlc = {k: v.loc[v.index.isin(keep)] for k, v in ohlc.items()}
        print(f"⚡ quick 模式：{len(keep)} 个交易日")

    if args.grid:
        _run_grid(scores, ohlc, args)
        return 0

    names = [n for n in (args.only.split(",") if args.only else SCENARIOS) if n]
    rows = []
    for name in names:
        if name not in SCENARIOS:
            print(f"⚠️ 未知方案 {name}，跳过")
            continue
        over = dict(BASE)
        over.update(SCENARIOS[name])
        if args.top_n:
            over["top_n"] = args.top_n
        cfg = ExecConfig(slippage_bp=args.slippage_bp, **over)
        res = run_exec_backtest(scores, ohlc, cfg)
        m = res.metrics
        rows.append({
            "方案": name,
            "年化": m.get("ann_return"),
            "夏普": m.get("sharpe"),
            "最大回撤": m.get("max_drawdown"),
            "Calmar": m.get("calmar"),
            "胜率": m.get("win_rate"),
            "盈亏比": m.get("payoff"),
            "盈利因子": m.get("profit_factor"),
            "平均持有": m.get("avg_hold_days"),
            "年换手": m.get("turnover_per_year"),
            "_result": res,
        })
        print(f"  ✓ {name:<16} 年化 {_fmt(m.get('ann_return'), True)}  "
              f"夏普 {_fmt(m.get('sharpe'))}  回撤 {_fmt(m.get('max_drawdown'), True)}")

    if not rows:
        return 1

    print("\n" + "=" * 100)
    print(f"成本假设：滑点 {args.slippage_bp:.0f}bp/边 ｜ 佣金 3bp ｜ 印花税 5bp(卖出) ｜ 调仓 5 日 ｜ TopN 8")
    print("=" * 100)
    df = pd.DataFrame(rows).sort_values("夏普", ascending=False, na_position="last")
    hdr = f"{'方案':<16}{'年化':>9}{'夏普':>7}{'回撤':>9}{'Calmar':>8}{'胜率':>8}{'盈亏比':>8}{'盈利因子':>9}{'持有日':>7}{'年换手':>8}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r['方案']:<16}"
              f"{_fmt(r['年化'], True):>9}{_fmt(r['夏普']):>7}{_fmt(r['最大回撤'], True):>9}"
              f"{_fmt(r['Calmar']):>8}{_fmt(r['胜率'], True, 1):>8}{_fmt(r['盈亏比']):>8}"
              f"{_fmt(r['盈利因子']):>9}{_fmt(r['平均持有'], False, 1):>7}{_fmt(r['年换手'], False, 1):>8}")

    best = df.iloc[0]
    res = best["_result"]
    if res.exit_stats is not None and len(res.exit_stats):
        print(f"\n【{best['方案']}】退出原因归因（按盈亏排序）")
        print(res.exit_stats.to_string(float_format=lambda x: f"{x:,.2f}"))

    # 成本敏感性：前 3 名在 30/50bp 下是否还成立（唯一判据）
    if not args.only:
        print("\n【成本敏感性】前 3 名在更差成本假设下是否仍成立（50bp 崩掉 = 不可用）")
        for _, row in df.head(3).iterrows():
            over = dict(BASE)
            over.update(SCENARIOS[row["方案"]])
            line = f"  {row['方案']:<18}"
            for bp in (10.0, 30.0, 50.0):
                cfg = ExecConfig(slippage_bp=bp, **over)
                m = run_exec_backtest(scores, ohlc, cfg).metrics
                line += (f"| {bp:>2.0f}bp 年化 {_fmt(m.get('ann_return'), True):>8} "
                         f"夏普 {_fmt(m.get('sharpe')):>5} ")
            print(line + "|")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

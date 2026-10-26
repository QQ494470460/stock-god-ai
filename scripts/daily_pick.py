# -*- coding: utf-8 -*-
"""每日选股 CLI —— 让「选股 → 交易计划 → 闭环」这条链可以手动、可复现地跑通。

为什么需要这个脚本
------------------
`generate_daily_picks()`（`src/core/daily_pick.py`）本来只有两个触发点：

1. 调度器 `auto_scheduler.job_daily_pick` —— 但 `scheduler.enabled` 默认 **False**，
   而且它只活在**看板进程**里，关掉看板就没了；
2. 看板上那个「开始 AI 选股」按钮 —— 手动、无参数、产物不落 `daily_reports`（见下）。

于是出现了一个很隐蔽的死循环：

    daily_reports/ 从没被生成 → load_latest_picks() 恒返回空
      → build_trade_plan() 的目标恒为 []
      → TradingLoop 每次都「无有效交易计划，跳过调仓」
      → loop_reports/ 永远没有真实业绩
      → 生成器 A 的换手 / 现金拖累 / 持仓不满 全部无从判断

闭环「从未真正运转」不是闭环本身坏了，是它的上游一直是空的。
本脚本把上游补上，并且可以放进计划任务里无人值守运行。

注意：`scripts/daily_recommend.py` 是**另一条路径**（只写 CSV 到 logs/），
不会产出 `daily_reports/daily_picks_*.json`，因此喂不饱闭环。别用错。

用法
----
    py -3.12 scripts/daily_pick.py                      # 按配置：zz500 / 20 只
    py -3.12 scripts/daily_pick.py --top 30 --min-score 52
    py -3.12 scripts/daily_pick.py --pool hs300
    py -3.12 scripts/daily_pick.py --build-plan         # 选完立刻刷新交易计划
    py -3.12 scripts/daily_pick.py --ensemble           # 多策略投票复核候选
    py -3.12 scripts/daily_pick.py --json               # 机器可读

退出码：0 = 有达标标的；3 = 选股成功但无标的（空仓是合法结果，不是错误）；
        2 = 生成过程出错（数据/模型异常）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:  # 与 run_loop.py 一致：先导入 config.settings 以注入外部 libs 路径
    import config.settings  # noqa: F401
except Exception:  # noqa: BLE001
    pass


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="每日 AI 选股（产出 daily_reports/daily_picks_*.json，供交易计划消费）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pool", default="", help="股票池（hs300/zz500/all，默认取配置）")
    ap.add_argument("--top", type=int, default=0, help="推荐只数上限（默认取配置）")
    ap.add_argument("--min-score", type=float, default=None,
                    help="最低评分；不给则用「基线+进化+市场状态」的有效阈值")
    ap.add_argument("--build-plan", action="store_true",
                    help="选完立刻据此刷新交易计划（写 data/trade_plans/）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出报告")
    ap.add_argument("--ensemble", action="store_true",
                    help="启用多策略复核：量化三策略（多因子/均值回归/融合）对 AI 候选投票，"
                         "集体看空则剔除、一致看多则加权提分（默认关闭）")
    ap.add_argument("--weighting", default="",
                    help="交易计划权重方式：equal | score | risk_parity | min_variance | "
                         "vol_target（默认取 GuardConfig.weighting 或 equal）")
    ap.add_argument("--tilt", type=float, default=None,
                    help="风险权重之上叠加 alpha 倾斜的力度（0=纯风险平价，默认 0.5）")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    from src.core.daily_pick import generate_daily_picks

    df, rep = generate_daily_picks(
        pool=args.pool or None,
        max_picks=args.top or None,
        min_score=args.min_score,
        write_report=True,
        use_ensemble=args.ensemble,
    )

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    else:
        print("=" * 72)
        print("每日 AI 选股")
        print("=" * 72)
        print(f"  状态       {rep['status']}")
        print(f"  股票池     {rep['pool']}｜扫描 {rep['n_scanned']} 只 → 达标 {rep['n_passed']} 只")
        print(f"  有效阈值   {rep['min_score']:.1f}")
        print(f"  市场状态   {rep.get('regime') or '未知'}")
        print(f"  生成时间   {rep['generated_at']}")
        ens = rep.get("ensemble")
        if ens:
            tag = "⚠️ 全部被否决→原样保留" if ens.get("fallback") else (
                f"否决 {ens['vetoed']}｜提分 {ens['boosted']}｜数据不足跳过 {ens['skipped']}")
            print(f"  多策略复核 {ens['before']} → {ens['after']} 只（{tag}）")
        picks = rep.get("picks") or []
        if picks:
            print()
            print(f"  {'代码':<9}{'名称':<12}{'评分':>7}  {'XGB信号':<10}理由")
            print("  " + "-" * 66)
            for r in picks:
                print(f"  {str(r.get('代码','')):<9}{str(r.get('名称',''))[:10]:<12}"
                      f"{float(r.get('评分') or 0):>7.1f}  {str(r.get('XGB信号','')):<10}"
                      f"{str(r.get('推荐理由',''))[:20]}")
        else:
            print("\n  （无达标标的 —— 空仓是合法结果，闭环会按兵不动而不是清仓）")

    # ---- 可选：立刻刷新交易计划 ----
    if args.build_plan:
        from src.trading.trade_plan import build_trade_plan

        kwargs = {}
        if args.weighting:
            kwargs["weighting"] = args.weighting
        if args.tilt is not None:
            kwargs["tilt_strength"] = max(0.0, min(float(args.tilt), 1.0))
        plan = build_trade_plan(write=True, **kwargs)
        if not args.json:
            print()
            print("📋 交易计划已刷新")
            print(f"  {plan.summary()}")
        if plan.targets:
            print(f"  ✅ 计划含 {len(plan.targets)} 只目标标的，闭环下次 tick 即可调仓")
        else:
            print("  ⚠️ 计划仍为空 —— 闭环将「按兵不动」（这是设计，不是故障）")

    if rep["status"] == "error":
        print(f"\n❌ 生成失败：{rep.get('error')}", file=sys.stderr)
        return 2
    if not rep.get("picks"):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

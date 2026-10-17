# -*- coding: utf-8 -*-
"""RankIC 历史回填 CLI —— 薄壳，核心在 ``src/core/rolling_ic.py::run_backfill``。

为什么需要它
------------
paper_shadow_audit.py --live-ic 只在「跑那一下」产出 1 个点，所以日志长期只有
零星 2 个点，画不出稳定性曲线。本脚本用**真实预测器**对历史上每个交易日打分，
算「评分日 → 后 horizon 个交易日」的超额收益（与训练标签口径一致：超额 =
个股收益 - 等权宇宙均值），再逐日算横截面 Spearman IC，累积进 rolling_ic_log.json。

前视（look-ahead）防护（与回测引擎同一套纪律）：
  · 评分日 d 的打分只用 d 及之前的数据（predict_batch 喂的是 df.loc[df.index <= d]）。
  · 前向收益用 d+horizon（交易日），绝不回看未来之外的信息。
  · 因此算出的每日 IC 是真实的 OOS 排序能力，可直接拿去画稳定性曲线 / 判稳定性。

性能与边界
----------
  · 成本 ≈ (评分日数) × (宇宙只数) 次单股打分。默认 --days 30 --sample 150，
    约 4500 次，后台跑几分钟到十几分钟。机器够快可调大。
  · --sample 截断宇宙（取前 N 只），只为控制成本，不影响 IC 口径。
  · 可重入：已存在于日志的评分日自动跳过；日志只保留最近 250 点。
  · 纯离线无需券商；但需要联网拿行情（fetcher），这是本机看板已具备的能力。

用法
----
    python scripts/rolling_ic_backfill.py                 # 默认 zz500，近 30 交易日，150 只
    python scripts/rolling_ic_backfill.py --days 60 --sample 300
    python scripts/rolling_ic_backfill.py --pool hs300 --horizon 5
    python scripts/rolling_ic_backfill.py --dry-run       # 只报告规模，不跑模型
    python scripts/rolling_ic_backfill.py --out data/rolling_ic_log.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.rolling_ic import IC_LOG, run_backfill


def main() -> int:
    ap = argparse.ArgumentParser(description="RankIC 历史回填（累积 rolling_ic_log.json）")
    ap.add_argument("--pool", default="zz500", help="股票池 hs300/zz500/zz1000/all")
    ap.add_argument("--days", type=int, default=30, help="回溯的评分日数量（交易日）")
    ap.add_argument("--sample", type=int, default=150, help="宇宙截断只数（取前 N 只，控制成本）")
    ap.add_argument("--horizon", type=int, default=5, help="前向收益跨度（交易日）")
    ap.add_argument("--min-stocks", type=int, default=30, help="单日有效样本下限，低于则跳过该日")
    ap.add_argument("--out", type=Path, default=IC_LOG, help="输出日志 JSON")
    ap.add_argument("--dry-run", action="store_true", help="只报告规模不跑模型")
    args = ap.parse_args()

    res = run_backfill(
        days=args.days, sample=args.sample, horizon=args.horizon,
        min_stocks=args.min_stocks, pool=args.pool, out=args.out,
        dry_run=args.dry_run,
    )

    status = res.get("status")
    if status == "error":
        print(f"❌ 回填失败: {res.get('error')}")
        return 1
    if status == "dry_run":
        print(f"🛑 dry-run：候选 {res.get('candidates')} 个评分日，待算 {res.get('todo')} 个（不跑模型）。")
        return 0
    if status == "noop":
        print(f"✅ 全部已存在，无需回填（日志共 {res.get('total')} 点）。")
        return 0

    print(f"\n✅ 回填完成：新增 {res['added']} 个评分日，日志共 {res['total']} 点")
    print(f"   全样本 RankIC 均值    {res['mean_ic']:+.4f}")
    print(f"   日频 IR (mean/std)    {res['ir_mean_std']:.2f}   ← 真正反映「单日排序稳不稳」")
    print(f"   t 统计量 (mean/SE)    {res['t_stat']:.2f}   ← production_metrics 里那个 'ic_ir=14' 就是这个口径")
    print(f"   正 IC 占比            {res['pos_rate']*100:.0f}%")
    print(f"   末点日期              {res.get('last_date')}")
    print(f"   下一步：python scripts/rolling_ic_curve.py  # 画稳定性曲线")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""逐腿样本外 RankIC 计算（超级集成第六腿·动态融合的基线统计生成器）

产出 data/leg_rankic.json：
  { "last_date": "2026-11-10", "legs": { leg: {rankic, ic_ir, n_days}, ... } }

融合时（config.ml.ensemble_dynamic_weights=True）由 EnsemblePredictor 读取该文件，
按各腿 RankIC 派生融合权重（见 src/core/leg_rankic.py / src/models/ensemble.py）。

⚠️ 需联网拉行情 + 跑全集成推理，属长任务，请在耐久上下文跑（终端双击 /
授权后注册每日收盘后作业），别在 agent 会话内直接拉起（会被回收）。

用法：
  python scripts/compute_leg_rankic.py [--days 30] [--sample 150] [--pool zz500]
                                       [--horizon 5] [--legs xgboost,alpha_xgboost,lstm,transformer]
                                       [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="计算各腿样本外 RankIC → data/leg_rankic.json")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--pool", type=str, default="zz500")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-stocks", type=int, default=30)
    ap.add_argument("--legs", type=str, default="",
                    help="逗号分隔的关注腿名（仅用于提醒哪些腿无统计）；留空=全部")
    ap.add_argument("--dry-run", action="store_true", help="只列候选评分日，不真算")
    args = ap.parse_args()

    from src.core.leg_rankic import compute_leg_rankic

    legs = [s.strip() for s in args.legs.split(",") if s.strip()] or None
    res = compute_leg_rankic(
        days=args.days,
        sample=args.sample,
        horizon=args.horizon,
        min_stocks=args.min_stocks,
        pool=args.pool,
        legs=legs,
        dry_run=args.dry_run,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

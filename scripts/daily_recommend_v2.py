# -*- coding: utf-8 -*-
"""
每日选股（V2）
============

用法：
    py -3.12 scripts/daily_recommend_v2.py --top 20
    py -3.12 scripts/daily_recommend_v2.py --top 20 --track   # 同时记录快照以便日后验证

与旧daily_recommend.py 的区别
---------------------------
1. 用 V2 因子管道，不是旧的 110 维混杂特征
2. 输出附带**历史样本外 IC**，让你知道这个信号靠不靠谱
3. `--track` 会把当天的预测 + **价格快照** 存进追踪文件
   —— 这是修复"验证链条断裂"的关键：以后能算出真实准确率
4. 不输出"78.5 分"这种人为评分，改为相对强弱排序 + 可信度说明
"""

import sys
import warnings
import argparse
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd
from loguru import logger

from config.settings import LOG_DIR
from src.models.predictor_v2 import PredictorV2


def load_pool(max_stocks: int = 400) -> Dict[str, pd.DataFrame]:
    """从本地缓存加载行情（不联网）"""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    store = {}
    for f in sorted(data_dir.glob("daily_*_qfq.parquet")):
        try:
            df = pd.read_parquet(f)
            if len(df) < 250:
                continue
            df["date"] = pd.to_datetime(df["date"])
            store[f.stem.split("_")[1]] = df.set_index("date").sort_index()
            if len(store) >= max_stocks:
                break
        except Exception:
            continue
    return store


def load_stock_names() -> Dict[str, str]:
    """载入股票名称映射（用于可读输出）"""
    p = Path(__file__).resolve().parent.parent / "data" / "stock_list.parquet"
    out = {}
    if p.exists():
        try:
            df = pd.read_parquet(p)
            if "code" in df.columns and "name" in df.columns:
                out = {
                    str(r.code).zfill(6): str(r.name)
                    for r in df[["code", "name"]].itertuples(index=False)
                }
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--max-stocks", type=int, default=400)
    ap.add_argument("--track", action="store_true",
                    help="记录预测与价格快照，供日后验证真实准确率")
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    logger.remove()
    logger.add(LOG_DIR / "recommend_v2.log", rotation="20 MB", level="INFO")

    print("=" * 72)
    print(" 选股神 V2 · 每日选股")
    print("=" * 72)

    # ---------- 模型真实表现 ----------
    predictor = PredictorV2()
    print("\n" + predictor.model_report())

    if not predictor.is_ready:
        print("\n  !! V2 模型未就绪，请先运行: py -3.12 scripts/train_v2.py")
        return

    # ---------- 加载数据 ----------
    print("\n加载本地行情...")
    store = load_pool(args.max_stocks)
    print(f"  载入 {len(store)} 只股票")
    if len(store) < 50:
        print("  !! 数据不足")
        return

    latest = max(d.index.max() for d in store.values())
    print(f"  最新交易日: {latest.date()}")

    names = load_stock_names()

    # ---------- 选股 ----------
    print("\n计算因子并预测...")
    try:
        df = predictor.predict_batch(store, stock_names=names, top_n=args.top)
    except Exception as e:
        print(f"  !! 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return

    if len(df) == 0:
        print("  !! 无有效选股结果")
        return

    # ---------- 输出 ----------
    print(f"\n{'=' * 72}")
    print(f" TOP {len(df)} 推荐")
    print("=" * 72)
    show = df[["代码", "名称", "预测档位", "强度分", "上涨概率", "下跌概率", "可靠性"]]
    print(show.to_string(index=False))

    print("\n  关键因子明细:")
    for _, r in df.head(10).iterrows():
        print(f"    {r['代码']} {r['名称']:<8} {r['关键因子']}")

    rel_counts = df["可靠性"].value_counts().to_dict()
    print(f"\n  信号可靠性分布: {rel_counts}")
    if "不足参考" in rel_counts:
        print("  ⚠ 部分信号历史样本外表现较弱，请谨慎参考")

    print("\n" + "=" * 72)
    print(" 说明：")
    print("  · 强度分 = 模型看涨强度，仅用于相对排序，不是上涨概率")
    print("  · 预测档位 4/4 表示模型认为该股在当日全市场中最强的一档")
    print("  · 可靠性基于历史**样本外**表现，非模型自评")
    print("  · 本输出不构成投资建议，请自行核实后决策")
    print("=" * 72)

    # ---------- 可选：记录快照 ----------
    if args.track:
        from src.models.tracker_v2 import save_predictions
        preds = []
        snapshot = {}
        for _, r in df.iterrows():
            preds.append({
                "code": r["代码"],
                "name": r["名称"],
                "score": float(r["强度分"]),
                "signal": 1,
                "predicted_bin": r["预测档位"],
                "reliability": r["可靠性"],
            })
            d = store.get(r["代码"])
            if d is not None and latest in d.index:
                snapshot[r["代码"]] = float(d.loc[latest, "close"])

        save_predictions(
            preds,
            scan_date=str(latest.date()),
            price_snapshot=snapshot
        )
        print(f"\n  ✅ 已记录 {len(preds)} 条预测 + {len(snapshot)} 个价格快照")
        print(f"  持有 {args.horizon} 个交易日后运行验证即可得到真实准确率")
        print(f"  验证命令: py -3.12 scripts/verify_v2.py")


if __name__ == "__main__":
    main()
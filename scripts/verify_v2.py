# -*- coding: utf-8 -*-
"""
预测验证（V2）
==============

用法：
    py -3.12 scripts/verify_v2.py# 验证所有已满持有期的记录
    py -3.12 scripts/verify_v2.py --horizons 5 10

与旧版的根本区别
----------------
旧版用"今天减N天"反推日期去匹配预测记录，
而预测只在交易日产生 → 周末/节假日必然匹配不上，
实测 22 批预测 0 批验证成功。

V2 遍历每一条历史记录，逐条检查是否真的走够N 个交易日。
"""

import sys
import warnings
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from loguru import logger
from config.settings import LOG_DIR
from src.models.tracker_v2 import verify_all_pending, get_accuracy_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10])
    args = ap.parse_args()

    logger.remove()
    logger.add(LOG_DIR / "verify_v2.log", level="INFO")
    logger.add(sys.stderr, level="WARNING")

    print("=" * 70)
    print(" 选股神 V2 · 预测验证")
    print("=" * 70)

    results = verify_all_pending(horizons=args.horizons)

    if not results:
        print("\n  暂无已满持有期的预测记录可验证。")
        print("  首次记录后需满 5 个交易日才会出现结果。")
    else:
        print(f"\n验证了 {len(results)} 条记录:\n")
        for r in results:
            ex = r.get("excess_return")
            ex_s = f"{ex:+.2%}" if ex is not None else "基准缺失"
            print(f"  {r['scan_date']} (持{r['hold_days']}日) "
                  f"准确率={r['accuracy']:.1%} "
                  f"平均={r['avg_return']:+.2%} "
                  f"超额={ex_s} "
                  f"胜率={r['win_rate']:.1%} "
                  f"回撤={r['max_drawdown']:.2%}")

            buckets = r.get("by_score_bucket") or {}
            if buckets:
                parts = [f"{k}:{v:+.2%}" for k, v in buckets.items() if v is not None]
                if parts:
                    print(f"      分档收益: {'  '.join(parts)}")

    # ---------- 汇总 ----------
    summary = get_accuracy_summary()
    print("\n" + "=" * 70)
    if summary["n_verified"] == 0:
        print(" 尚无已验证记录")
        print(f" {summary.get('note', '')}")
    else:
        print(f" 历史汇总（{summary['n_verified']} 次验证, "
              f"{summary['n_checked']} 个样本）")
        print(f"   总体准确率    {summary['accuracy']:.1%}" if summary["accuracy"] is not None else "")
        if summary["avg_return"] is not None:
            print(f"   平均收益      {summary['avg_return']:+.2%}")
        if summary["avg_excess"] is not None:
            print(f"   平均超额      {summary['avg_excess']:+.2%}")
        if summary["avg_win_rate"] is not None:
            print(f"   平均胜率      {summary['avg_win_rate']:.1%}")

        print("\n   历史明细:")
        for h in summary["history"][:20]:
            ex = h.get("excess_return")
            ex_s = f"{ex:+.2%}" if ex is not None else "  n/a"
            print(f"     {h['scan_date']} 持{h['hold_days']}日  "
                  f"准确率={h['accuracy']:.1%} 平均={h['avg_return']:+.2%} "
                  f"超额={ex_s} 样本={h['checked']}")
    print("=" * 70)

    print("\n 解读提醒：")
    print("  · 选股场景下准确率 50% 左右属正常，别期待 70%+")
    print("  · 更该关注「超额收益」和「平均收益」是否持续为正")
    print("  · 样本数少于 100 个时，统计噪声大于信号，不足以定论")


if __name__ == "__main__":
    main()
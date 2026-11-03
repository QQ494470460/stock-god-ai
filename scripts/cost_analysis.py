"""
交易成本测算：把毛超额换算成"扣成本后还剩多少"。

为什么必须算这一步
------------------
RankIC 0.02 听起来有信号，但选股策略的成败取决于**换手成本**是否能被
超额覆盖。5 日换仓意味着一年换手 50 次，A股双边成本约 0.15%~0.25%，
一年就是 7.5%~12.5%。这个数量级足以吃掉大部分量化选股策略的毛超额。

公式
----
    年换手次数 = 252 / horizon
    年化毛超额 = 单期超额 × 年换手次数
    年化成本   = 年换手次数 × 单次双边成本
    净超额     = 毛超额 - 成本

用法
----
    py -3.12 scripts/cost_analysis.py
    py -3.12 scripts/cost_analysis.py --metric leaksafe_h20_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

# A股双边交易成本假设（2026年）
COST_ITEMS = {
    "佣金(万2.5双边)": 0.05 / 100,
    "印花税(卖出0.05%)": 0.05 / 100,
    "过户费(沪市双边)": 0.002 / 100,
    "滑点(保守0.10%)": 0.10 / 100,
}


def analyze(metrics_path: Path, horizon: int) -> dict:
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_period = m.get("oos_top_excess")
    if per_period is None:
        raise ValueError(f"{metrics_path.name} 缺少 oos_top_excess")
    turns = 252 / horizon
    gross = per_period * turns
    cost_low = turns * 0.15 / 100
    cost_base = turns * sum(COST_ITEMS.values())
    cost_high = turns * 0.30 / 100
    return {
        "name": metrics_path.stem,
        "horizon": horizon,
        "rank_ic": m.get("oos_rank_ic"),
        "ic_ir": m.get("oos_ic_ir"),
        "win_rate": m.get("oos_win_rate"),
        "per_period": per_period,
        "turns_per_year": turns,
        "gross_annual": gross,
        "cost_low": cost_low,
        "cost_base": cost_base,
        "cost_high": cost_high,
        "net_base": gross - cost_base,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", action="append", default=None,
                    help="指标文件名（可多次指定），默认自动找 leaksafe*_metrics.json")
    args = ap.parse_args()

    files = ([MODEL_DIR / n for n in args.metric] if args.metric
             else sorted(MODEL_DIR.glob("leaksafe*_metrics.json")))
    files = [f for f in files if f.exists()]
    if not files:
        print("未找到指标文件")
        return 1

    print("单次双边成本假设:")
    for k, v in COST_ITEMS.items():
        print(f"  {k:<20}{v*100:.3f}%")
    print(f"  {'合计':<20}{sum(COST_ITEMS.values())*100:.3f}%")

    rows = []
    for f in files:
        # horizon 从指标文件名里的 _h20 推断，缺省按 5
        h = 5
        stem = f.stem
        if "_h" in stem:
            tail = stem.split("_h")[-1].split("_")[0]
            if tail.isdigit():
                h = int(tail)
        # leaksafe_metrics.json 对应 horizon=5 的旧版
        try:
            rows.append(analyze(f, h))
        except Exception as e:
            print(f"跳过 {f.name}: {e}")

    print()
    hdr = (f"{'模型':<24}{'窗口':>5}{'单期超额':>10}{'年换手':>7}"
           f"{'毛超额/年':>11}{'净(乐观)':>10}{'净(中性)':>10}{'净(保守)':>10}")
    print(hdr)
    print("-" * 88)
    for r in rows:
        print(f"{r['name']:<24}{r['horizon']:>5}{r['per_period']:>9.3%}"
              f"{r['turns_per_year']:>7.0f}{r['gross_annual']:>10.2%}"
              f"{r['gross_annual']-r['cost_low']:>10.2%}"
              f"{r['net_base']:>10.2%}"
              f"{r['gross_annual']-r['cost_high']:>10.2%}")

    print()
    print("成本情景：乐观=万1佣金+0.05%滑点(0.15%)，中性=0.202%，保守=0.30%")
    print()
    print("结论:")
    for r in rows:
        lo = r["gross_annual"] - r["cost_low"]
        hi = r["gross_annual"] - r["cost_high"]
        if hi > 0.02:
            verdict = f"即使保守成本仍有 +{hi:.1%}/年，可小仓位实盘验证"
        elif lo > 0:
            verdict = (f"乐观成本下 +{lo:.2%}/年、保守成本下 {hi:.2%}/年 —— "
                       f"能不能赚钱取决于你的实际费率与滑点")
        else:
            verdict = f"即便乐观成本也是 {lo:.2%}/年，不具备实盘价值"
        print(f"  {r['name']} (h={r['horizon']}): {verdict}")

    print("\n  注：未计冲击成本（大资金建仓会推高成本）、未计融资成本、"
          "未考虑涨跌停无法成交。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

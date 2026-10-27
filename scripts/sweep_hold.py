# -*- coding: utf-8 -*-
"""持有纪律参数扫描：hold_if_profitable_pct 的真实样本外 A/B。

背景
----
实盘诊断：配 +15% 止盈线，6 笔盈利全部 +0.17%~+1.62% 且都由调仓卖出，止盈一次
没触发；亏损一路跑到 -5% 止损。盈亏比设计 3:1、实测 0.15。根因是调仓每日按弱
信号换票 —— 涨过头的跌出榜单被卖（只赚 1%），还没跌的留在榜单被留下（亏到止损）。

``trading_loop.hold_if_profitable_pct`` = 持仓跌出目标组合但浮盈达标时不强制卖，
交止损/止盈托管。本脚本用**同一份已建好的样本外面板**，对它做 0 / 3 / 5 的真实
A/B，拿样本外结果决定是否开启，而不是拍脑袋。

口径说明（重要）
----------------
面板分数由部署模型一次性生成，故「分数」对模型本身可能是样本内的。但本 A/B 只
改变**组合构建规则**、分数三种参数下完全相同，因此模型偏差在三组之间相互抵消，
**相对结论（pct 之间的差）是干净的**；绝对数值仅作参照。

用法
----
    PYTHONPATH=. python scripts/sweep_hold.py --pcts 0,3,5 --folds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_panel():
    from src.evolution.panels import load_panel
    p = load_panel()
    if p is None:
        raise SystemExit("面板缓存缺失：先跑 scripts/evolve.py --build-panel")
    score, close, bench = p
    if score is None or close is None or len(score) == 0:
        raise SystemExit("面板为空")
    return score, close, bench


def _base_params() -> Dict:
    from src.evolution.gate import _current_backtest_params
    try:
        params = _current_backtest_params()
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 取当前参数失败，退回 gate 默认：{e}")
        params = {}
    # 与 gate 一致的回测周期
    params.setdefault("top_n", 20)
    return params


def _run(score, close, bench, params: Dict, pct: float):
    from src.core.backtest import run_backtest
    p = dict(params)
    top_n = int(p.pop("top_n", 20))
    try:
        res = run_backtest(score, close, top_n=top_n,
                           rebalance_days=int(p.pop("rebalance_days", 5)),
                           benchmark_close=bench,
                           hold_if_profitable_pct=pct, **p)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ pct={pct} 回测失败：{e}")
        return None
    m = res.metrics or {}
    if not m.get("n_periods"):
        return None
    return m


def _fmt(v, nd=2, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:+.{nd}f}%" if pct else f"{v:.{nd}f}"


def _chunks(n: int, folds: int):
    """把 [0, n) 切成 folds 个连续块（覆盖全 panel，块间不重叠）。"""
    bounds = np.linspace(0, n, folds + 1).astype(int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(folds)
            if bounds[i + 1] - bounds[i] > 10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcts", default="0,3,5", help="逗号分隔的百分比，如 0,3,5")
    ap.add_argument("--folds", type=int, default=5, help="连续分块数")
    ap.add_argument("--rebalance-days", type=int, default=5,
                    help="调仓周期（实盘日频病理对应 1，gate 默认 5）")
    args = ap.parse_args()
    pcts = [float(x) for x in args.pcts.split(",") if x.strip() != ""]

    score, close, bench = _load_panel()
    n = len(score)
    params = _base_params()
    params["rebalance_days"] = int(args.rebalance_days)
    print(f"面板 {n} 天 × {score.shape[1]} 只")
    print("回测参数：" + ", ".join(f"{k}={v}" for k, v in sorted(params.items())))

    # ---------------- 全样本（参照，不作为决策依据） ----------------
    print("\n【全样本参照】")
    print(f"{'pct':>5} {'sharpe':>8} {'年化':>9} {'最大回撤':>9} {'换手/期':>9} {'单期超额':>9}")
    for pct in pcts:
        m = _run(score, close, bench, params, pct)
        if not m:
            print(f"{pct:>5.0f}  （无有效期）")
            continue
        print(f"{pct:>5.0f} {_fmt(m.get('sharpe')):>8} {_fmt(m.get('ann_return'),2,True):>9} "
              f"{_fmt(m.get('max_drawdown'),2,True):>9} {_fmt(m.get('turnover'),2,True):>9} "
              f"{_fmt(m.get('top_excess'),3,True):>9}")

    # ---------------- 连续分块 OOS 聚合（决策依据） ----------------
    blocks = _chunks(n, args.folds)
    print(f"\n【跨时间 OOS：{len(blocks)} 个连续块，覆盖全 panel】")
    agg: Dict[float, Dict[str, List[float]]] = {p: {"sharpe": [], "excess": [], "ann": [],
                                                    "dd": [], "turn": []} for p in pcts}
    per_block = []   # [(block_idx, {pct: metrics})]
    for bi, (a, b) in enumerate(blocks):
        sp, cp = score.iloc[a:b], close.iloc[a:b]
        bm = bench.iloc[a:b] if bench is not None and hasattr(bench, "iloc") else bench
        row = {}
        for pct in pcts:
            m = _run(sp, cp, bm, params, pct)
            row[pct] = m
            if m:
                agg[pct]["sharpe"].append(m.get("sharpe", np.nan))
                agg[pct]["excess"].append(m.get("top_excess", np.nan))
                agg[pct]["ann"].append(m.get("ann_return", np.nan))
                agg[pct]["dd"].append(m.get("max_drawdown", np.nan))
                agg[pct]["turn"].append(m.get("turnover", np.nan))
        per_block.append((bi, row))
        print(f"  块{bi+1} [{a}:{b}] " + "  ".join(
            f"pct{p:g}:sharpe={_fmt(r.get('sharpe')) if r else '—'}" for p, r in row.items()))

    print(f"\n{'pct':>5} {'均夏普':>8} {'均超额':>9} {'均年化':>9} {'最差回撤':>9} {'均换手':>9}")
    for pct in pcts:
        a = agg[pct]
        if not a["sharpe"]:
            print(f"{pct:>5.0f}  （无有效块）")
            continue
        print(f"{pct:>5.0f} {np.nanmean(a['sharpe']):>8.2f} "
              f"{_fmt(np.nanmean(a['excess']),3,True):>9} "
              f"{_fmt(np.nanmean(a['ann']),2,True):>9} "
              f"{_fmt(np.nanmin(a['dd']),2,True):>9} "
              f"{_fmt(np.nanmean(a['turn']),2,True):>9}")

    # ---------------- 各候选 vs 基准(最小 pct) 逐块胜负 ----------------
    base = min(pcts)
    base_idx = pcts.index(base)
    print(f"\n【逐块胜负 vs pct={base:g}】（判据：夏普≥基准 且 单期超额≥基准）")
    for pct in pcts:
        if pct == base:
            continue
        wins = both = usable = 0
        for _, row in per_block:
            mb, mc = row.get(base), row.get(pct)
            if not mb or not mc:
                continue
            usable += 1
            bsh, csh = mb.get("sharpe") or 0.0, mc.get("sharpe") or 0.0
            bex, cex = mb.get("top_excess") or 0.0, mc.get("top_excess") or 0.0
            if csh >= bsh:
                wins += 1
                if cex >= bex:
                    both += 1
        print(f"  pct={pct:g}: 夏普不输 {wins}/{usable}，夏普+超额双不输 {both}/{usable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

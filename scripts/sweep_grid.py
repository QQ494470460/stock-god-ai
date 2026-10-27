# -*- coding: utf-8 -*-
"""通用回测参数扫描：对任意 BacktestConfig 参数做跨时间 OOS 对比。

与 ``sweep_hold.py`` 同一套口径：覆盖全 panel 的连续分块 OOS，逐块 + 聚合，
用来回答「降换手能不能在样本外真正加分」，而不是看全样本幻觉。

用法
----
    PYTHONPATH=. python scripts/sweep_grid.py --param band_pct --values 0.1,0.15,0.25,0.4,0.6
    PYTHONPATH=. python scripts/sweep_grid.py --param rebalance_days --values 1,3,5,10,20
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

# 各参数的「生产默认」，用作逐块胜负的基准
_DEFAULT = {"band_pct": 0.25, "rebalance_days": 5,
            "weight_cap": 0.30, "cash_reserve_pct": 0.10,
            "min_trade_amount": 5000.0, "hold_if_profitable_pct": 0.0}


def _load_panel(panel_dir: str = ""):
    if panel_dir:
        import pandas as pd
        d = Path(panel_dir)
        sp = d / "score_panel.parquet"
        cp = d / "close_panel.parquet"
        if not (sp.exists() and cp.exists()):
            raise SystemExit(f"指定面板目录缺文件：{d}")
        return pd.read_parquet(sp), pd.read_parquet(cp), None
    from src.evolution.panels import load_panel
    p = load_panel()
    if p is None:
        raise SystemExit("面板缓存缺失：先跑 scripts/evolve.py --build-panel")
    score, close, bench = p
    if score is None or len(score) == 0:
        raise SystemExit("面板为空")
    return score, close, bench


def _base_params() -> Dict:
    from src.evolution.gate import _current_backtest_params
    try:
        params = _current_backtest_params()
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 取当前参数失败，退回默认：{e}")
        params = {}
    params.setdefault("top_n", 20)
    return params


def _run(score, close, bench, params: Dict, param: str, value):
    from src.core.backtest import run_backtest
    p = dict(params)
    top_n = int(p.pop("top_n", 20))
    rb = int(p.pop("rebalance_days", 5))
    if param == "rebalance_days":
        rb = int(value)
    else:
        p[param] = value
    try:
        res = run_backtest(score, close, top_n=top_n, rebalance_days=rb,
                           benchmark_close=bench, **p)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ {param}={value} 回测失败：{e}")
        return None
    m = res.metrics or {}
    return m if m.get("n_periods") else None


def _fmt(v, nd=2, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:+.{nd}f}%" if pct else f"{v:.{nd}f}"


def _chunks(n: int, folds: int):
    b = np.linspace(0, n, folds + 1).astype(int)
    return [(int(b[i]), int(b[i + 1])) for i in range(folds) if b[i + 1] - b[i] > 10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True, help="要扫描的参数名，如 band_pct / rebalance_days")
    ap.add_argument("--values", required=True, help="逗号分隔取值，如 0.1,0.25,0.4")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--panel-dir", default="", help="指定面板目录（默认用 deployed 缓存）")
    ap.add_argument("--slippage", type=float, default=None, help="覆盖单边滑点（如 0.003=30bp）")
    ap.add_argument("--commission", type=float, default=None, help="覆盖单边佣金率")
    ap.add_argument("--stamp-tax", type=float, default=None, help="覆盖卖出印花税")
    args = ap.parse_args()
    param = args.param
    values = [float(x) if param not in ("weighting",) else x
              for x in args.values.split(",") if x.strip() != ""]
    if param == "rebalance_days":
        values = [int(v) for v in values]

    score, close, bench = _load_panel(args.panel_dir)
    n = len(score)
    params = _base_params()
    if args.slippage is not None:
        params["slippage"] = args.slippage
    if args.commission is not None:
        params["commission"] = args.commission
    if args.stamp_tax is not None:
        params["stamp_tax"] = args.stamp_tax
    base_val = params.get(param, _DEFAULT.get(param))
    print(f"面板 {n} 天 × {score.shape[1]} 只；扫描 {param}={values}；基准 {param}={base_val}")

    blocks = _chunks(n, args.folds)
    agg: Dict[float, Dict[str, List[float]]] = {v: {"sharpe": [], "excess": [], "ann": [],
                                                    "dd": [], "turn": []} for v in values}
    per_block: List[Dict] = []
    for a, b in blocks:
        sp, cp = score.iloc[a:b], close.iloc[a:b]
        bm = bench.iloc[a:b] if bench is not None and hasattr(bench, "iloc") else bench
        row = {v: _run(sp, cp, bm, params, param, v) for v in values}
        per_block.append(row)
        for v, m in row.items():
            if m:
                agg[v]["sharpe"].append(m.get("sharpe", np.nan))
                agg[v]["excess"].append(m.get("top_excess", np.nan))
                agg[v]["ann"].append(m.get("ann_return", np.nan))
                agg[v]["dd"].append(m.get("max_drawdown", np.nan))
                agg[v]["turn"].append(m.get("turnover", np.nan))

    # 逐块明细（★ 标当块夏普最高的取值）
    print("\n【逐块 OOS 夏普】")
    hdr = "  块  ".ljust(10) + "".join(f"{param}={v:g}".ljust(16) for v in values)
    print(hdr)
    for bi, row in enumerate(per_block):
        sharpes = [row[v].get("sharpe", np.nan) for v in values if row.get(v)]
        best_sh = max([s for s in sharpes if s is not None and not np.isnan(s)] or [np.nan])
        cells = []
        for v in values:
            m = row.get(v)
            sh = m.get("sharpe") if m else None
            mark = "★" if (sh is not None and not np.isnan(sh)
                           and not np.isnan(best_sh) and sh == best_sh) else " "
            cells.append((mark + (_fmt(sh) if m else "—")).ljust(16))
        print(f"  块{bi+1:<3}".ljust(10) + "".join(cells))

    # 聚合
    print(f"\n【聚合 OOS】")
    print(f"{param:>16} {'均夏普':>8} {'均超额':>9} {'均年化':>9} {'最差回撤':>9} {'均换手':>9}")
    ranked = []
    for v in values:
        a = agg[v]
        if not a["sharpe"]:
            print(f"{v!s:>16}  （无有效块）")
            continue
        ms = float(np.nanmean(a["sharpe"]))
        ranked.append((ms, v))
        print(f"{v!s:>16} {ms:>8.2f} {_fmt(np.nanmean(a['excess']),3,True):>9} "
              f"{_fmt(np.nanmean(a['ann']),2,True):>9} "
              f"{_fmt(np.nanmin(a['dd']),2,True):>9} "
              f"{_fmt(np.nanmean(a['turn']),2,True):>9}")

    if ranked:
        ranked.sort(reverse=True)
        best = ranked[0][1]
        print(f"\n最佳（按均夏普）：{param}={best}（{ranked[0][0]:.2f}）")
        # 逐块 vs 基准
        bv = base_val if base_val in values else (values[0] if values else None)
        if bv is not None:
            print(f"逐块 vs 基准 {param}={bv}（夏普≥基准 且 超额≥基准 = 双赢）")
            for v in values:
                if v == bv:
                    continue
                wins = both = usable = 0
                for row in per_block:
                    mb, mc = row.get(bv), row.get(v)
                    if not mb or not mc:
                        continue
                    usable += 1
                    bsh, csh = mb.get("sharpe") or 0.0, mc.get("sharpe") or 0.0
                    bex, cex = mb.get("top_excess") or 0.0, mc.get("top_excess") or 0.0
                    if csh >= bsh:
                        wins += 1
                        if cex >= bex:
                            both += 1
                print(f"  {param}={v}: 夏普不输 {wins}/{usable}，双不输 {both}/{usable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

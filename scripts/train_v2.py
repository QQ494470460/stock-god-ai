# -*- coding: utf-8 -*-
"""
V2 训练管线（泄漏安全 + 横截面因子）
==================================

用法：
    py -3.12 scripts/train_v2.py --max-stocks 300
    py -3.12 scripts/train_v2.py --factors-only    # 只看因子体检，不训练

与旧管线的差异
--------------
1. 标签：绝对涨跌三分类 → 未来超额收益的截面分位（选股该预测的东西）
2. 特征：单股 110 维混杂量纲 → 因子面板逐日截面 rank 归一化
3. 切分：按行切（实为跨股票切）→ 按日期 walk-forward
4. 标准化：全量 fit → 每折内只用 train 段 fit
5. 采样：全量 SMOTE → class_weight（不动样本，保时序）
6. 评估：accuracy/F1 → RankIC / IC_IR / TopN 超额 / Sharpe / 回撤
"""

import sys
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import MODEL_DIR, LOG_DIR
from src.data.fetcher import fetcher
from src.models.factors_v2 import FactorPanelBuilder, evaluate_factors
from src.models.labeling import (
    forward_excess_return, make_rank_labels, mean_rank_ic,
    top_portfolio_return, portfolio_metrics, cs_rank,
)
from src.models.trainer_v2 import LeakSafeTrainer, walk_forward_split

HORIZON = 5
MIN_ROWS = 400

# 旧口径（按股票代码字典序取前 N 只本地 parquet）的别名
LEGACY_POOL_ALIASES = {"legacy", "code_order", "default", "none", ""}
# 真实成分股池：交给 DataFetcher.get_stock_pool 解析（与线上服务侧同一函数）
KNOWN_POOLS = {"hs300", "zz500", "zz1000", "hs300_zz500", "all"}


def _resolve_pool_codes(pool: str) -> Optional[List[str]]:
    """把 pool 名称解析成股票代码列表（升序）。

    Parameters
    ----------
    pool : str
        "hs300" / "zz500" / "hs300_zz500" / "all" 走 DataFetcher.get_stock_pool；
        "legacy" / "code_order" / "default"（含空串）返回 None 表示走旧口径。

    Returns
    -------
    Optional[List[str]]
        成分股代码升序列表；返回 None 表示旧口径（按代码字典序取前 N 只本地 parquet）。

    Notes
    -----
    任何解析失败都降级为 None（旧口径）并打 warning，绝不中断训练流程。
    """
    key = (pool or "legacy").strip().lower()
    if key in LEGACY_POOL_ALIASES:
        return None
    if key not in KNOWN_POOLS:
        logger.warning(
            f"未知股票池 '{pool}'（可用: {sorted(KNOWN_POOLS)} | "
            f"{sorted(LEGACY_POOL_ALIASES)}），回退到 legacy 口径"
        )
        return None
    try:
        codes = sorted({str(c).zfill(6) for c in fetcher.get_stock_pool(key)})
    except Exception as e:
        logger.warning(f"获取股票池 '{key}' 失败({e})，回退到 legacy 口径")
        return None
    if not codes:
        logger.warning(f"股票池 '{key}' 返回空列表，回退到 legacy 口径")
        return None
    if key in ("hs300", "zz500") and len(codes) > 800:
        logger.warning(
            f"股票池 '{key}' 返回 {len(codes)} 只，远超预期(<=800)，"
            f"疑似成分股接口失败被静默降级为全市场，请检查网络"
        )
    return codes


def load_pool(max_stocks: int, pool: str = "hs300") -> Dict[str, pd.DataFrame]:
    """加载本地缓存行情（不联网取行情；仅成分股名单走 fetcher 缓存）。

    修复说明
    --------
    旧实现**完全忽略 pool 参数**，等价于「按股票代码字典序排序后的前 N 只本地
    parquet」（如 000001/000002/000006/000007/000008），而线上服务侧
    (scripts/daily_recommend.py) 用的是 fetcher.get_stock_pool("hs300") 真成分股
    （000001/000002/000063/000157/000166），两者 150 只池子只重叠 12 只，
    造成 train/serve universe 不一致。本实现让 pool 真正生效。

    Parameters
    ----------
    max_stocks : int
        最多加载多少只股票。
    pool : str
        - "hs300" / "zz500" / "hs300_zz500" / "all"：只加载该池成分股中本地有
          parquet 的股票（本地没有的直接跳过），顺序按成分股代码升序。
        - "legacy" / "code_order" / "default"：旧行为，按代码字典序取前 N 只，
          用于复现历史基线。

    Returns
    -------
    Dict[str, pd.DataFrame]
        code -> 日线 DataFrame（index 为 date，已按日期升序）。
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    all_files = sorted(data_dir.glob("daily_*_qfq.parquet"))
    by_code = {f.stem.split("_")[1]: f for f in all_files}

    pool_codes = _resolve_pool_codes(pool)
    if pool_codes is None:
        # 旧口径：按代码字典序取前 max_stocks*3 个文件（多读一些，滤掉过短的）
        ordered = all_files[: max_stocks * 3]
    else:
        # 真实成分股池：按池内代码升序，只取本地有数据的
        ordered = [by_code[c] for c in pool_codes if c in by_code]
        n_missing = len(pool_codes) - len(ordered)
        if n_missing > 0:
            logger.info(
                f"股票池 '{pool}': 成分股 {len(pool_codes)} 只，"
                f"本地无 parquet {n_missing} 只（已跳过），可用 {len(ordered)} 只"
            )

    store: Dict[str, pd.DataFrame] = {}
    for f in ordered:
        try:
            df = pd.read_parquet(f)
            if len(df) < MIN_ROWS:          # 历史长度不足，跳过
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            code = f.stem.split("_")[1]
            store[code] = df
            if len(store) >= max_stocks:
                break
        except Exception:
            continue

    head5 = list(store.keys())[:5]
    logger.info(
        f"load_pool(pool={pool!r}, max_stocks={max_stocks}) -> "
        f"实际加载 {len(store)} 只, 前5个代码={head5}"
    )
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stocks", type=int, default=300)
    ap.add_argument("--pool", default="legacy",
                    help="股票池: hs300/zz500/zz1000/hs300_zz500/all 为真成分股；"
                         "legacy(默认) 为旧口径「按代码排序取前 N 只」")
    ap.add_argument("--factors-only", action="store_true", help="只做因子体检")
    # 单一模型口径：只用 xgboost
    ap.add_argument("--model", default="xgboost", choices=["xgboost"])
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    logger.remove()
    logger.add(LOG_DIR / f"train_v2.log", rotation="50 MB", level="INFO")
    logger.add(sys.stderr, level="WARNING")

    print("=" * 70)
    print(" 选股神 V2 训练管线")
    print("=" * 70)

    # ---------- 1. 加载数据 ----------
    print("\n[1/5] 加载本地数据...")
    store = load_pool(args.max_stocks, pool=args.pool)
    print(f"  股票池: {args.pool} | 有效股票: {len(store)} 只 | "
          f"前5个代码: {list(store.keys())[:5]}")
    if len(store) < 50:
        print("  !! 数据不足，退出")
        return

    # 价格矩阵
    close = pd.DataFrame({c: d["close"] for c, d in store.items()}).sort_index()
    # 覆盖度过滤
    cover = close.notna().mean(axis=1)
    close = close.loc[cover >= 0.6]
    print(f"  日期轴: {close.index[0].date()} ~ {close.index[-1].date()} ({len(close)} 天)")

    # ---------- 2. 标签 ----------
    print(f"\n[2/5] 构造标签（未来{HORIZON}日超额收益截面分位）...")
    fwd_abs = close.shift(-HORIZON) / close - 1
    fwd_excess = fwd_abs.sub(fwd_abs.mean(axis=1), axis=0)
    fwd_excess = fwd_excess.clip(
        lower=fwd_excess.quantile(0.01, axis=1),
        upper=fwd_excess.quantile(0.99, axis=1),
        axis=0
    )
    labels_panel = make_rank_labels(fwd_excess, n_bins=5)
    valid_days = int((labels_panel.notna().sum(axis=1) >= 30).sum())
    print(f"  有效交易日: {valid_days}")
    _dist = labels_panel.stack().dropna().value_counts(normalize=True).sort_index()
    print(f"  标签分布: " + ", ".join(
        f"{int(k)}档={v:.1%}" for k, v in _dist.items()
    ))

    # ---------- 3. 因子面板 ----------
    print("\n[3/5] 构建因子面板（逐日截面 rank 归一化）...")
    # 缓存文件名必须带 pool：不同股票池的面板不能互相复用
    cache_f = (Path(__file__).resolve().parent.parent / "data"
               / f"panel_cache_{args.pool}_{args.max_stocks}.pkl")
    builder = FactorPanelBuilder()
    if cache_f.exists():
        try:
            panel = pd.read_pickle(cache_f)
            print(f"  使用缓存面板: {panel.shape}")
        except Exception:
            panel = builder.build(store, use_cs_rank=True)
    else:
        panel = builder.build(store, use_cs_rank=True)
        try:
            panel.to_pickle(cache_f)
            print(f"  面板已缓存到 {cache_f.name}")
        except Exception as e:
            print(f"  面板缓存失败: {e}")

    # ---------- 4. 因子体检 ----------
    print("\n[4/5] 因子有效性体检")
    fac_report = evaluate_factors(panel, fwd_excess)
    n_usable = int(fac_report["usable"].sum())
    print(f"  股票池 {len(store)} 只 | 样本外最后30%用于评估")
    print(f"  可用因子(样本外|IC|>0.02): {n_usable}/{len(fac_report)}")
    print()
    print(f"  {'因子':<18}{'样本内IC':>10}{'样本外IC':>10}{'IC_IR':>8}{'TopN超额':>11}{'胜率':>8}")
    print("  " + "-" * 66)
    for _, r in fac_report.head(16).iterrows():
        isic = f"{r['is_ic']:+.4f}" if not np.isnan(r['is_ic']) else "  n/a"
        oic = f"{r['oos_ic']:+.4f}" if not np.isnan(r['oos_ic']) else "  n/a"
        ir = f"{r['oos_ic_ir']:+.1f}" if not np.isnan(r['oos_ic_ir']) else " n/a"
        ex = f"{r['oos_excess']:+.3%}" if not np.isnan(r['oos_excess']) else "   n/a"
        wr = f"{r['oos_win_rate']:.0%}" if not np.isnan(r['oos_win_rate']) else " n/a"
        mark = "✓" if r["usable"] else " "
        print(f" {mark}{r['factor']:<17}{isic:>10}{oic:>10}{ir:>8}{ex:>11}{wr:>8}")

    # 分解：IC 衰减程度
    decay = (fac_report["is_ic"].abs() - fac_report["oos_ic"].abs()).mean()
    print(f"\n  平均 IC 衰减(样本内|IC| - 样本外|IC|): {decay:+.4f}")
    if decay > 0.02:
        print("  → 衰减明显，样本内表现含较大过拟合成分，以样本外为准")
    else:
        print("  → 衰减可控，因子相对稳健")

    if args.factors_only:
        fac_report.to_csv(MODEL_DIR / "factor_report_v2.csv", index=False)
        print(f"\n因子报告已保存: {MODEL_DIR / 'factor_report_v2.csv'}")
        return

    # 只保留可用因子
    good_factors = fac_report[fac_report["usable"]]["factor"].tolist()
    if not good_factors:
        print("\n  !! 无可用因子，改用样本外|IC|最大的前 10 个")
        good_factors = fac_report.head(10)["factor"].tolist()

    panel = panel[[c for c in panel.columns if c[1] in set(good_factors)]]
    print(f"\n  保留因子 {len(good_factors)} 个")

    # ---------- 5. 展平 + 训练 ----------
    print(f"\n[5/5] Walk-forward 训练 ({args.model})...")
    dates = panel.index
    X, sample_code, sample_date = FactorPanelBuilder.flatten(panel, dates)

    # 只保留标签有效的样本
    label_lookup = labels_panel.stack()
    label_lookup.index = [
        f"{pd.Timestamp(d).date()}_{c}" for d, c in label_lookup.index
    ]
    label_lookup = label_lookup[~label_lookup.index.duplicated(keep="last")]

    y = X.index.map(label_lookup).astype(float)
    mask = y.notna()
    X = X[mask]
    y = y[mask].astype(int)
    sample_code = sample_code[mask]
    sample_date = sample_date[mask]

    print(f"  ML 样本: {len(X)} × {X.shape[1]}")
    print(f"  标签分布: " + ", ".join(
        f"{int(k)}档={v:.1%}" for k, v in
        y.value_counts(normalize=True).sort_index().items()
    ))

    trainer = LeakSafeTrainer(n_splits=args.n_splits, horizon=HORIZON)
    result = trainer.train_walk_forward(
        X, pd.Series(y, index=X.index), sample_date,
        model_type=args.model,
        fwd_excess=fwd_excess   # 传入真实前瞻超额收益，避免用档位冒充收益
    )

    # ---------- 6. 输出 ----------
    print("\n" + "=" * 70)
    print(" 训练结果（全部为样本外 walk-forward 指标）")
    print("=" * 70)
    print(result.summary())

    print(f"\n  逐折表现:")
    print(f"  {'折':<4}{'训练区间':<26}{'验证区间':<26}{'样本':>6}{'准确率':>8}")
    print("  " + "-" * 70)
    for f in result.folds:
        print(f"  {f['fold']:<4}{f['train']:<26}{f['valid']:<26}"
              f"{f['n_valid']:>6}{f['accuracy']:>8.1%}")

    print(f"\n  Top-10 重要因子:")
    for _, r in result.feature_importance.head(10).iterrows():
        print(f"    {r['feature']:<20}{r['importance']:.4f}")

    trainer.save(result, "leaksafe")
    fac_report.to_csv(MODEL_DIR / "factor_report_v2.csv", index=False)

    print(f"\n  模型与指标已保存到 {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
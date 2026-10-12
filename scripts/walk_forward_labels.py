# -*- coding: utf-8 -*-
"""
严格、零泄漏的 Walk-Forward 标签×损失 变体对比
==============================================

目的
----
在**完全相同的**特征、切分、零泄漏条件下，对比 4 种「训练标签 + 损失」变体对
A 股选股真实样本外 RankIC 的影响，定位现有模型「准确率不高」的一个可量化改进点。

4 个变体（每折各训一个模型，只换标签与损失，其余完全共用）：
  V0 nominal_balanced : 五档截面分位标签 + XGBClassifier(class_weight=balanced)
  V1 lambdarank       : 连续前瞻超额收益作相关性 + XGBRanker(rank:pairwise)  ★重点
  V2 recent_decay     : 同 V0，但 sample_weight 额外乘时间衰减（越近期越重要）
  V3 ordinal_cost     : 同 V0，但 sample_weight 对头部/尾部档(0/4)加权重、中性档(2)降权

泄漏防护（全部标注在代码里）
----------------------------
  [L1] scaler 只在 train 段 fit_transform，valid 只 transform（绝不全局 fit）
  [L2] 不使用 SMOTE（不动样本，保时序），仅用 class_weight
  [L3] walk_forward_split 的 embargo_days = HORIZON，隔离标签窗口重叠
  [L4] 特征在日期 d 只依赖 d 及之前数据（feature_engine 全因果）
  [L5] XGBRanker 的 group 只由 train 行按日期构成（query 不跨折、不含未来）
  [L6] RankIC / 组合收益计算时，fwd_excess 只取与分数**同日期**的值
  [L7] 所有变体用同一份 X / 切分 / scaler，只换标签与损失 → 对比公平

复用（import，不重写）
----------------------
  scripts.train_v2.load_pool
  src.features.engine.feature_engine
  src.models.labeling.{forward_excess_return, make_rank_labels, mean_rank_ic,
                        top_portfolio_return, portfolio_metrics, rank_ic}
  src.models.trainer_v2.walk_forward_split
  config.settings.MODEL_DIR

运行
----
  py -3.12 scripts/walk_forward_labels.py --max-stocks 150
  py -3.12 scripts/walk_forward_labels.py --variants V3 --max-stocks 150 --pool hs300

--pool 说明
-----------
  legacy(默认)  : 旧口径，按股票代码字典序取前 N 只本地 parquet（复现历史基线）
  hs300/zz500/all : 真成分股池（与线上服务侧 fetcher.get_stock_pool 同源）
"""

from __future__ import annotations

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

from config.settings import MODEL_DIR
from scripts.train_v2 import load_pool
from src.features.engine import feature_engine
from src.models.labeling import (
    forward_excess_return,
    make_rank_labels,
    mean_rank_ic,
    top_portfolio_return,
    rank_ic,
)
from src.models.trainer_v2 import walk_forward_split


# ============ 泄漏防护工具 ============

def balanced_weights(y_enc: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """class_weight='balanced' 的稳健实现。

    优先用 sklearn.compute_class_weight；若某折 train 缺档（会触发除零），
    回退到只在「出现的档」上做平衡，避免崩溃（对应坑：LabelEncoder 在空折/缺档）。
    """
    try:
        cw = compute_class_weight("balanced", classes=classes, y=y_enc)
        m = dict(zip(classes, cw))
        return np.array([m[v] for v in y_enc], dtype=float)
    except Exception:
        present, counts = np.unique(y_enc, return_counts=True)
        n, k = len(y_enc), len(present)
        m = {c: n / (k * cnt) for c, cnt in zip(present, counts)}
        return np.array([m[v] for v in y_enc], dtype=float)


def compute_metrics(
    records: pd.DataFrame,
    fwd_excess: pd.DataFrame,
    top_n: int,
) -> dict:
    """由 (date, code, score) 记录计算样本外指标。

    仅用 records 内的 (date,code) 透视成面板，fwd_excess 严格取同日期值（[L6]）。
    """
    if records is None or len(records) == 0:
        return dict(ic=np.nan, ic_ir=np.nan, ic_pos=np.nan,
                    top_excess=np.nan, top_win=np.nan,
                    n_days=0, n_oos_days=0)
    rec = records.drop_duplicates(["date", "code"], keep="last")
    sp = rec.pivot(index="date", columns="code", values="score")
    fwd_aligned = fwd_excess.reindex(index=sp.index, columns=sp.columns)

    ic, ic_ir, n_days = mean_rank_ic(sp, fwd_aligned)

    # 逐日 IC（用于 IC>0 比例；[L6] 只取同日期）
    daily: List[float] = []
    for d in sp.index:
        try:
            r = rank_ic(sp.loc[d], fwd_aligned.loc[d])
        except Exception:
            r = np.nan
        if not np.isnan(r):
            daily.append(float(r))
    daily_arr = np.array(daily)
    ic_pos = float(np.mean(daily_arr > 0)) if len(daily_arr) else np.nan

    # Top-N 组合超额（真实前瞻超额收益，绝不用档位冒充收益）
    top = top_portfolio_return(sp, fwd_excess, top_n=top_n)
    top_excess = float(top["excess"].mean()) if len(top) else np.nan
    top_win = float((top["excess"] > 0).mean()) if len(top) else np.nan

    return dict(
        ic=float(ic), ic_ir=float(ic_ir), ic_pos=ic_pos,
        top_excess=top_excess, top_win=top_win,
        n_days=int(n_days), n_oos_days=int(len(top)),
    )


def _clean(obj):
    """把 numpy / nan 转成 JSON 安全类型。"""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)) and (np.isnan(obj) if isinstance(obj, float) or isinstance(obj, np.floating) else False):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return _clean(obj.tolist())
    return obj


# ============ 主流程 ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stocks", type=int, default=150)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--variants", nargs="*", default=None,
                    choices=["V0", "V1", "V2", "V3"],
                    help="只跑指定变体子集，默认全跑")
    ap.add_argument("--pool", default="legacy",
                    help="股票池: hs300/zz500/zz1000/hs300_zz500/all 为真成分股；"
                         "legacy(默认) 为旧口径「按代码排序取前 N 只」，用于复现历史基线")
    args = ap.parse_args()

    variants = args.variants or ["V0", "V1", "V2", "V3"]
    HALF_LIFE = 250.0
    ALPHA = 1.0

    print("=" * 72)
    print(" 选股神 · 严格零泄漏 Walk-Forward 标签×损失 变体对比")
    print("=" * 72)
    print(f" 变体: {variants} | max_stocks={args.max_stocks} | "
          f"n_splits={args.n_splits} | horizon={args.horizon} | top_n={args.top_n}")
    print(f" 股票池: {args.pool}")

    # ---------- 1. 加载数据（离线，本地 parquet） ----------
    print("\n[1/5] 加载本地数据 (load_pool)...")
    store = load_pool(args.max_stocks, pool=args.pool)
    print(f"  有效股票: {len(store)} 只 | 前5个代码: {list(store.keys())[:5]}")
    if len(store) < 50:
        print("  !! 数据不足，退出")
        return

    # ---------- 2. 逐股构建特征 + 收 close 面板 ----------
    print("\n[2/5] 逐股构建特征 (feature_engine.build_features)...")
    feat_store: Dict[str, pd.DataFrame] = {}
    close_parts = []
    for code, df in store.items():
        # 特征在日期 d 只依赖 d 及之前数据（因果，[L4]）
        feats = feature_engine.build_features(df).drop(columns=["close"])
        feat_store[code] = feats
        close_parts.append(df["close"].rename(code))

    # 统一特征列顺序（防止「特征列数不齐」坑：不同股票列集不齐）
    feat_cols = list(next(iter(feat_store.values())).columns)
    print(f"  特征列数: {len(feat_cols)}")

    close_panel = pd.concat(close_parts, axis=1).sort_index()
    print(f"  close 面板: {close_panel.shape} | "
          f"{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}")

    # ---------- 3. 目标：真实前瞻超额收益 + 五档截面标签 ----------
    print(f"\n[3/5] 构造标签（未来{args.horizon}日超额收益截面分位）...")
    fwd_excess = forward_excess_return(
        close_panel, horizon=args.horizon, winsorize=0.01
    )
    labels_panel = make_rank_labels(fwd_excess, n_bins=5)
    print(f"  标签面板: {labels_panel.shape}")

    # ---------- 4. 组装 ML 样本（X / y / dates / codes / y_cont） ----------
    print("\n[4/5] 组装 ML 样本（遍历 date×code）...")
    X_parts, y_parts, codes_parts, dates_parts, fwd_parts = [], [], [], [], []
    for code, fdf in feat_store.items():
        lab = labels_panel[code].reindex(fdf.index)
        fwd = fwd_excess[code].reindex(fdf.index)
        keep = lab.notna().values
        fdf_k = fdf[keep].reindex(columns=feat_cols)
        lab_k = lab[keep]
        fwd_k = fwd[keep]
        dts = fdf_k.index
        idx = [f"{d.date()}_{code}" for d in dts]
        fdf_k = fdf_k.copy()
        fdf_k.index = idx
        X_parts.append(fdf_k)
        y_parts.append(lab_k.to_numpy().astype(int))
        codes_parts.append(pd.Series(code, index=idx))
        dates_parts.append(pd.Series(pd.to_datetime(dts), index=idx))
        fwd_parts.append(fwd_k.to_numpy())

    X = pd.concat(X_parts)
    y = pd.Series(np.concatenate(y_parts), index=X.index, dtype=int)
    codes = pd.concat(codes_parts)
    dates = pd.concat(dates_parts)
    y_cont = pd.Series(np.concatenate(fwd_parts), index=X.index)

    # 丢弃「特征全 NaN」的样本；标签 NaN 已在上面过滤（[L? 数据清洗]）
    all_nan = X.isna().all(axis=1)
    if all_nan.any():
        X = X[~all_nan]; y = y[~all_nan]; codes = codes[~all_nan]
        dates = dates[~all_nan]; y_cont = y_cont[~all_nan]

    # 部分 NaN 用 0 填充（XGBoost 原生支持缺失，但 StandardScaler 不支持）
    X = X.fillna(0.0)

    print(f"  ML 样本: {len(X)} × {X.shape[1]}")
    print(f"  标签分布: " + ", ".join(
        f"{int(k)}档={v:.1%}" for k, v in
        y.value_counts(normalize=True).sort_index().items()
    ))
    print(f"  日期范围: {dates.min().date()} ~ {dates.max().date()}")

    # LabelEncoder 在全量 y 上 fit 一次（保证 4 变体档位编码一致）
    le = LabelEncoder()
    le.fit(y.values)
    print(f"  标签类别: {le.classes_.tolist()}")

    # ---------- 5. Walk-Forward 切分（所有变体共用，embargo=HORIZON [L3]） ----------
    print(f"\n[5/5] Walk-Forward 切分 (embargo_days={args.horizon})...")
    splits = walk_forward_split(
        dates.values, n_splits=args.n_splits, embargo_days=args.horizon
    )
    print(f"  折数: {len(splits)}")

    # 每个变体收集：逐折 (date,code,score) 记录 + 逐折明细
    variant_records: Dict[str, List[pd.DataFrame]] = {v: [] for v in variants}
    variant_folds: Dict[str, List[dict]] = {v: [] for v in variants}

    for sp in splits:
        tr_mask = (dates >= sp.train_start) & (dates <= sp.train_end)
        va_mask = (dates >= sp.valid_start) & (dates <= sp.valid_end)
        Xtr, Xva = X[tr_mask], X[va_mask]
        ytr, yva = y[tr_mask], y[va_mask]
        dtr, dva = dates[tr_mask], dates[va_mask]
        cva = codes[va_mask]

        if len(Xtr) < 100 or len(Xva) < 30:
            print(f"  Fold {sp.fold} 样本不足 (tr={len(Xtr)}, va={len(Xva)})，跳过")
            continue

        # [L1] scaler 只在 train 上 fit；valid 只 transform（防泄漏）
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)

        ytr_enc = le.transform(ytr.values)

        # ====== 每折、每变体各训一个模型（只换标签与损失） ======
        for v in variants:
            scores = None
            if v == "V0" or v == "V1":
                # V0 / V2 / V3 共用 XGBClassifier；V1 单独 ranker
                pass

            if v == "V0":
                sw = balanced_weights(ytr_enc, le.classes_)
                clf = xgb.XGBClassifier(
                    n_estimators=400, max_depth=5, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                    gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
                    random_state=42, n_jobs=-1, verbosity=0,
                    eval_metric="mlogloss",
                )
                clf.fit(Xtr_s, ytr_enc, sample_weight=sw, verbose=False)
                proba = clf.predict_proba(Xva_s)
                # 分数 = 预测档位期望（单调，越高越看多）
                scores = proba @ np.arange(len(le.classes_))

            elif v == "V2":
                # 同 V0，但 sample_weight 额外乘时间衰减：越近期权重越高
                age_days = (sp.train_end - dtr).dt.days.to_numpy()
                time_decay = np.exp(-age_days / HALF_LIFE)
                sw = balanced_weights(ytr_enc, le.classes_) * time_decay
                clf = xgb.XGBClassifier(
                    n_estimators=400, max_depth=5, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                    gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
                    random_state=42, n_jobs=-1, verbosity=0,
                    eval_metric="mlogloss",
                )
                clf.fit(Xtr_s, ytr_enc, sample_weight=sw, verbose=False)
                proba = clf.predict_proba(Xva_s)
                scores = proba @ np.arange(len(le.classes_))

            elif v == "V3":
                # 同 V0，但 sample_weight 对头部/尾部档(0/4)加权重、中性档(2)降权
                ord_w = 1.0 + ALPHA * np.abs(ytr.values - 2)
                sw = balanced_weights(ytr_enc, le.classes_) * ord_w
                clf = xgb.XGBClassifier(
                    n_estimators=400, max_depth=5, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                    gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
                    random_state=42, n_jobs=-1, verbosity=0,
                    eval_metric="mlogloss",
                )
                clf.fit(Xtr_s, ytr_enc, sample_weight=sw, verbose=False)
                proba = clf.predict_proba(Xva_s)
                scores = proba @ np.arange(len(le.classes_))

            elif v == "V1":
                # ★ lambdarank：用连续前瞻超额收益作相关性，直接优化日内相对排序
                cont = y_cont[tr_mask].to_numpy()
                # 必须把 train 行**按日期排序**，group 与排序后行序严格对齐（[L5]）
                order = np.argsort(dtr.values)
                Xtr_s_r = Xtr_s[order]
                cont_r = cont[order]
                dtr_s = dtr.values[order]
                # group = 每个交易日(query)的股票数，顺序与排序后行序一致
                groups = pd.Series(dtr_s).value_counts().sort_index().to_numpy().tolist()
                ranker = xgb.XGBRanker(
                    objective="rank:pairwise", n_estimators=400, max_depth=5,
                    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=10, reg_lambda=2.0,
                    random_state=42, n_jobs=-1, eval_metric="ndcg",
                )
                ranker.fit(Xtr_s_r, cont_r, group=groups)
                # 预测用模型直接得排序分数（越高=同日内相对越强）
                scores = ranker.predict(Xva_s)

            # 收集本折 valid 分数
            fold_rec = pd.DataFrame({
                "date": dva.values,
                "code": cva.values,
                "score": scores,
            })
            variant_records[v].append(fold_rec)

            fold_m = compute_metrics(fold_rec, fwd_excess, args.top_n)
            variant_folds[v].append({
                "fold": sp.fold,
                "train": f"{sp.train_start.date()}~{sp.train_end.date()}",
                "valid": f"{sp.valid_start.date()}~{sp.valid_end.date()}",
                "n_train": int(len(Xtr)),
                "n_valid": int(len(Xva)),
                **fold_m,
            })
            print(f"  Fold {sp.fold} [{v}] RankIC={fold_m['ic']:+.4f} "
                  f"Top{args.top_n}超额={fold_m['top_excess']:+.4%}")

    # ---------- 6. 跨折汇总 + 输出 ----------
    print("\n" + "=" * 72)
    print(" 跨折汇总（全部为 walk-forward 样本外指标）")
    print("=" * 72)
    header = f"  {'变体':<16}{'RankIC':>9}{'IC_IR':>8}{'IC>0比例':>9}" \
             f"{'TopN超额':>11}{'TopN胜率':>9}{'天数':>7}"
    print(header)
    print("  " + "-" * 68)

    results: Dict[str, dict] = {}
    for v in variants:
        combined = pd.concat(variant_records[v], ignore_index=True) if variant_records[v] else pd.DataFrame()
        ov = compute_metrics(combined, fwd_excess, args.top_n)
        results[v] = ov
        print(f"  {v:<16}{ov['ic']:>+9.4f}{ov['ic_ir']:>8.2f}"
              f"{ov['ic_pos']:>9.1%}{ov['top_excess']:>11.4%}"
              f"{ov['top_win']:>9.1%}{ov['n_oos_days']:>7}")

    # 赢的变体：RankIC 最高，平手用 TopN 超额破
    def _key(v):
        r = results[v]
        return (r["ic"] if not np.isnan(r["ic"]) else -1e9,
                r["top_excess"] if not np.isnan(r["top_excess"]) else -1e9)

    winner = max(variants, key=_key)
    print("\n  >>> 赢的变体: " + winner +
          f"  (RankIC={results[winner]['ic']:+.4f}, "
          f"Top{args.top_n}超额={results[winner]['top_excess']:+.4%})")

    # 只跑部分变体（如 --variants V3）时 results 里没有 V0/V1，
    # 直接格式化 None 会抛 TypeError: unsupported format string passed to NoneType。
    ic_v1 = results.get("V1", {}).get("ic")
    ic_v0 = results.get("V0", {}).get("ic")
    v1_better = False
    if ic_v1 is None or ic_v0 is None:
        print("  >>> V1 lambdarank 是否优于 V0: 未比较（本次未同时跑 V0 与 V1）")
    else:
        v1_better = bool(ic_v1 > ic_v0)
        print(f"  >>> V1 lambdarank 是否优于 V0: {'是' if v1_better else '否'} "
              f"(V1={ic_v1:+.4f} vs V0={ic_v0:+.4f})")

    # ---------- 7. 写 JSON ----------
    out = {
        "config": {
            "max_stocks": args.max_stocks,
            "n_splits": args.n_splits,
            "horizon": args.horizon,
            "top_n": args.top_n,
            "variants": variants,
            "pool": args.pool,
            "n_stocks_loaded": len(store),
            "codes_head5": list(store.keys())[:5],
            "embargo_days": args.horizon,
            "half_life": HALF_LIFE,
            "alpha": ALPHA,
            "label": "forward_excess_return(horizon=5, winsorize=0.01) -> 5-bin rank label",
            "leak_controls": [
                "scaler fit on train only", "no SMOTE (class_weight only)",
                "embargo_days=HORIZON", "causal features (d depends on <=d)",
                "ranker group from train dates only",
                "fwd_excess aligned by same date",
            ],
        },
        "metrics": {v: results[v] for v in variants},
        "folds": {v: variant_folds[v] for v in variants},
        "winner": winner,
        "v1_better_than_v0": bool(v1_better),
    }
    out_path = MODEL_DIR / "walk_forward_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_clean(out), f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  指标已保存: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
生产部署模型训练（严格零泄漏 walk-forward 方法论落地）
====================================================

把**已严格验证有效**的「诚实零泄漏 walk-forward + 5档截面分位标签 +
V3 序数成本权重」方法论落地为**生产部署模型**，替换旧三分类绝对±2%标签管线
（旧管线样本外 RankIC 仅 0.080，且存在标签泄漏风险）。

核心方法（与 scripts/walk_forward_oos.py / walk_forward_labels.py 一致，已实测
walk-forward RankIC=0.124、IC正向率81.8%）：
  · 生产特征：src/features/engine.feature_engine.build_features -> drop('close')
  · 标签：forward_excess_return(close, horizon) -> make_rank_labels(n_bins=5)
          （4=最强，0=最弱）
  · 损失/权重（V3 ordinal_cost，标签×损失对比赢家）：
        bal = compute_class_weight('balanced', classes, y)  逐样本映射
        ord_w = 1 + ALPHA*|y-2|      # 极端档(0/4)=3x，中性档(2)=1x
        sw = bal * ord_w
  · 切分：walk_forward_split(embargo=horizon)，scaler 仅 train 段 fit
  · 最终模型：全部数据上 StandardScaler fit -> XGBClassifier(400/5/0.03/0.9/0.9/5)

零泄漏防护（详见 scripts/walk_forward_oos.py 顶部说明）：
  [L1] scaler 只 fit 于训练段（最终部署模型则 fit 于全量，供推理口径统一）
  [L2] 无 SMOTE（用 class_weight 替代，保时序）
  [L3] embargo=horizon 隔离标签窗口重叠（train 末端标签伸进 valid）
  [L4] 特征全因果（d 只依赖 d 及之前的数据，无未来函数）
  [L5] 验证段分数与真实收益按 (date,code) 对齐，拼接后再算 RankIC

保存文件（与 src/models/predictor.py 加载口径**完全一致**，务必同名）：
  MODEL_DIR/xgboost_model.pkl   = 最终 XGBClassifier（5类 0..4）
  MODEL_DIR/scaler.pkl          = 全量 fit 的 StandardScaler（feature_names_in_=生产特征列）
  MODEL_DIR/label_encoder.pkl   = 在 [0..4] 上 fit 的 LabelEncoder

用法：
  py -3.12 scripts/train_production_v2.py --max-stocks 60 --no-validate   # 小样本先跑通
  py -3.12 scripts/train_production_v2.py                                  # 默认200 + 诚实验证
"""

from __future__ import annotations

import sys
import os
import json
import shutil
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import MODEL_DIR
from scripts.train_v2 import load_pool
from src.features.engine import feature_engine
from src.models.labeling import (
    forward_excess_return,
    make_rank_labels,
    rank_ic,
)
from src.models.trainer_v2 import walk_forward_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
MIN_STOCKS_PER_DAY = 30          # 与 src/models/labeling.py 保持一致
ALPHA = 1.0                      # V3 序数成本权重的极端档放大系数
XGB_PARAMS: Dict[str, object] = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
)


# ---------------------------------------------------------------------------
# 零泄漏工具函数（直接复用 walk_forward_oos 的已验证实现）
# ---------------------------------------------------------------------------

def _panel_daily_metrics(
    score_panel: pd.DataFrame,
    fwd_excess: pd.DataFrame,
    top_n: int = 20,
) -> dict:
    """逐日计算 RankIC / IC_IR / IC>0 比例 / TopN 超额 / TopN 胜率。

    与 src/models/labeling.py 口径一致：Spearman 只看秩，不同折的分数即便落在
    不同尺度上，在同一交易日内的排序仍然有效（每个交易日只属于一个折）。
    """
    common = score_panel.index.intersection(fwd_excess.index)
    ics: List[float] = []
    top_excess: List[float] = []

    for d in common:
        s = score_panel.loc[d]
        r = fwd_excess.loc[d]
        v = pd.DataFrame({"s": s, "r": r}).dropna()
        if len(v) < MIN_STOCKS_PER_DAY:
            continue
        ic = rank_ic(v["s"], v["r"])
        if np.isnan(ic):
            continue
        ics.append(float(ic))
        top = v.nlargest(min(top_n, len(v)), "s")
        top_excess.append(float(top["r"].mean() - v["r"].mean()))

    if not ics:
        return {
            "rank_ic": float("nan"), "ic_ir": float("nan"),
            "ic_positive_rate": float("nan"), "top_excess": float("nan"),
            "top_win_rate": float("nan"), "n_days": 0,
        }

    arr = np.array(ics)
    tes = np.array(top_excess)
    std_ic = arr.std(ddof=1)
    ic_ir = float(arr.mean() / (std_ic / np.sqrt(len(arr)))) if std_ic > 0 else float("nan")
    return {
        "rank_ic": float(arr.mean()),
        "ic_ir": ic_ir,
        "ic_positive_rate": float((arr > 0).mean()),
        "top_excess": float(tes.mean()),
        "top_win_rate": float((tes > 0).mean()),
        "n_days": int(len(arr)),
    }


def build_feat_store(store: Dict[str, pd.DataFrame]) -> List[str]:
    """逐股构建生产特征并缓存；返回参考列名（ref_cols）。"""
    feat_store: Dict[str, pd.DataFrame] = {}
    ref_cols: Optional[List[str]] = None
    for code, df in store.items():
        try:
            feats = feature_engine.build_features(df).drop(columns=["close"])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  特征构建失败 {code}: {e}")
            continue
        if feats.shape[0] < 130:
            continue
        if feats.isna().all().all():
            continue
        if ref_cols is None:
            ref_cols = list(feats.columns)
        feat_store[code] = feats
    # 用闭包外的可变容器把 feat_store 带到 assemble_samples
    build_feat_store._store = feat_store  # type: ignore[attr-defined]
    return ref_cols or []


def assemble_samples(
    feat_store: Dict[str, pd.DataFrame],
    labels_panel: pd.DataFrame,
    ref_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """把特征与标签拼成 ML 样本（一个 (date, code) 一条）。

    Returns
    -------
    (X_df, y, dates, codes)
        X_df : DataFrame(columns=ref_cols + ['__date__','__code__'])
        y    : np.ndarray(int) 标签 0..4
        dates: np.ndarray(pd.Timestamp)
        codes: np.ndarray(str)
    """
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    date_parts: List[np.ndarray] = []
    code_parts: List[np.ndarray] = []

    for code, feat in feat_store.items():
        if code not in labels_panel.columns:
            continue
        f = feat.reindex(columns=ref_cols)
        lab = labels_panel[code]
        dfc = f.copy()
        dfc["__label__"] = lab
        dfc = dfc.dropna(subset=["__label__"])
        valid_feat = dfc[ref_cols].notna().any(axis=1)
        if int(valid_feat.sum()) < 130:
            continue
        dfc = dfc[valid_feat]

        X_parts.append(dfc[ref_cols].values.astype(float))
        y_parts.append(dfc["__label__"].astype(int).values)
        date_parts.append(dfc.index.to_numpy())
        code_parts.append(np.array([code] * len(dfc)))

    if not X_parts:
        raise RuntimeError("没有可用样本，请检查数据与标签口径")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    dates = np.concatenate(date_parts)
    codes = np.concatenate(code_parts)

    X_df = pd.DataFrame(X, columns=ref_cols)
    X_df["__date__"] = dates
    X_df["__code__"] = codes
    logger.info(
        f"  组装 ML 样本: {len(X_df)} 条 × {len(ref_cols)} 特征 | "
        f"标签分布: " + ", ".join(
            f"{int(k)}={v:.1%}" for k, v in
            pd.Series(y).value_counts(normalize=True).sort_index().items()
        )
    )
    return X_df, y, dates, codes


def make_v3_sample_weights(y: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """V3 ordinal_cost 样本权重（生产落地版）。

        bal   = class_weight('balanced') 逐样本映射
        ord_w = 1 + alpha * |y - 2|      # 极端档(0/4)=3x，中性档(2)=1x
        sw    = bal * ord_w
    """
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_to_w = dict(zip(classes.tolist(), cw.tolist()))
    bal = np.array([class_to_w[int(v)] for v in y], dtype=float)
    ord_w = 1.0 + alpha * np.abs(y - 2)
    return bal * ord_w


# ---------------------------------------------------------------------------
# 诚实 walk-forward 验证（与最终部署模型同口径）
# ---------------------------------------------------------------------------

def run_walk_forward_validation(
    X_df: pd.DataFrame,
    y: np.ndarray,
    dates: np.ndarray,
    codes: np.ndarray,
    fwd_excess: pd.DataFrame,
    ref_cols: List[str],
    args,
) -> dict:
    """逐折重训（同 XGB 参数 + V3 权重，scaler 仅 train 段 fit），拼 OOS 面板算 RankIC。"""
    splits = walk_forward_split(dates, n_splits=args.n_splits, embargo_days=args.horizon)
    logger.info(f"Walk-forward 切分: {len(splits)} 折（embargo={args.horizon} 天）")

    X_feat = X_df[ref_cols]
    fold_records: List[dict] = []
    pool_dates: List[np.ndarray] = []
    pool_codes: List[np.ndarray] = []
    pool_scores: List[np.ndarray] = []

    for sp in splits:
        tr_mask = (dates >= sp.train_start) & (dates <= sp.train_end)
        va_mask = (dates >= sp.valid_start) & (dates <= sp.valid_end)
        n_tr = int(tr_mask.sum())
        n_va = int(va_mask.sum())
        if n_tr < 100 or n_va < 30:
            logger.warning(f"  Fold {sp.fold + 1}: 样本不足(train={n_tr},valid={n_va})，跳过")
            continue

        X_tr = X_feat[tr_mask].fillna(0.0)
        X_va = X_feat[va_mask].fillna(0.0)
        y_tr = y[tr_mask]

        # ★ scaler 仅 train 段 fit（防泄漏）；验证段只 transform
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_tr)
        Xva_s = scaler.transform(X_va)

        # ★ V3 样本权重仅由 train 段构造
        sw = make_v3_sample_weights(y_tr)

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(Xtr_s, y_tr, sample_weight=sw, verbose=False)

        # 分数 = 期望档位（单调，越高越看多），与 eval_model_oos 多分类分支一致
        proba = model.predict_proba(Xva_s)
        score = proba @ np.arange(proba.shape[1], dtype=float)

        fp = pd.DataFrame({
            "date": dates[va_mask], "code": codes[va_mask], "score": score,
        }).drop_duplicates(subset=["date", "code"], keep="last")
        score_panel = fp.pivot(index="date", columns="code", values="score")
        fm = _panel_daily_metrics(score_panel, fwd_excess, top_n=args.top_n)

        fold_records.append({
            "fold": sp.fold + 1,
            "train": f"{sp.train_start.date()}~{sp.train_end.date()}",
            "valid": f"{sp.valid_start.date()}~{sp.valid_end.date()}",
            "n_train": n_tr, "n_valid": n_va,
            **fm,
        })
        logger.info(
            f"  Fold {sp.fold + 1}: train={n_tr} valid={n_va} "
            f"RankIC={fm['rank_ic']:+.4f} IC_IR={fm['ic_ir']:+.2f} "
            f"TopN超额={fm['top_excess']:+.4%} 天数={fm['n_days']}"
        )

        pool_dates.append(dates[va_mask])
        pool_codes.append(codes[va_mask])
        pool_scores.append(score)

    if not fold_records:
        raise RuntimeError("没有有效折可训练，无法进行诚实验证")

    pool = pd.DataFrame({
        "date": np.concatenate(pool_dates),
        "code": np.concatenate(pool_codes),
        "score": np.concatenate(pool_scores),
    }).drop_duplicates(subset=["date", "code"], keep="last")
    pooled_panel = pool.pivot(index="date", columns="code", values="score")
    summary = _panel_daily_metrics(pooled_panel, fwd_excess, top_n=args.top_n)
    summary["n_folds"] = len(fold_records)

    print("\n" + "=" * 78)
    print(" 诚实 Walk-Forward 样本外验证（生产特征 + 5档标签 + V3权重，零泄漏）")
    print("=" * 78)
    print(f" {'折':<4}{'训练区间':<24}{'验证区间':<24}{'样本':>7}"
          f"{'RankIC':>9}{'IC_IR':>8}{'IC>0':>7}{'TopN超额':>11}{'胜率':>7}{'天数':>6}")
    print(" " + "-" * 76)
    for f in fold_records:
        print(f" {f['fold']:<4}{f['train']:<24}{f['valid']:<24}{f['n_train']:>7}"
              f"{f['rank_ic']:>9.4f}{f['ic_ir']:>8.2f}{f['ic_positive_rate']:>7.1%}"
              f"{f['top_excess']:>11.4%}{f['top_win_rate']:>7.1%}{f['n_days']:>6}")
    print(" " + "-" * 76)
    print(f" {'全周期OOS':<4}{'':<48}{summary['rank_ic']:>9.4f}"
          f"{summary['ic_ir']:>8.2f}{summary['ic_positive_rate']:>7.1%}"
          f"{summary['top_excess']:>11.4%}{summary['top_win_rate']:>7.1%}{summary['n_days']:>6}")
    print("=" * 78)

    return {"summary": summary, "folds": fold_records}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="生产部署模型训练（零泄漏 walk-forward + 5档 + V3权重）")
    ap.add_argument("--pool", default="hs300",
                    help="股票池（真实生效，与线上服务侧 fetcher.get_stock_pool 同源）："
                         "hs300/zz500/hs300_zz500/all；legacy=旧口径按代码排序取前 N 只")
    ap.add_argument("--max-stocks", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--no-validate", action="store_true", help="跳过诚实 walk-forward 验证（仅训练+保存）")
    args = ap.parse_args()

    print("=" * 78)
    print(" 选股神 · 生产部署模型训练（零泄漏 walk-forward + 5档截面分位 + V3序数权重）")
    print("=" * 78)

    # 1) 数据（离线本地 parquet）
    print(f"\n[1/4] 加载数据（最多 {args.max_stocks} 只，离线 parquet）...")
    store = load_pool(args.max_stocks, pool=args.pool)
    if len(store) < 50:
        print("  数据不足，退出")
        return 1
    print(f"  股票池: {args.pool} | 有效股票: {len(store)} 只 | "
          f"前5个代码: {list(store.keys())[:5]}")

    # 2) 逐股生产特征
    print("\n[2/4] 构建生产特征（build_features，剔除 close）...")
    ref_cols = build_feat_store(store)
    feat_store = build_feat_store._store  # type: ignore[attr-defined]
    print(f"  特征列数: {len(ref_cols)} | 有效特征股: {len(feat_store)} 只")

    # 3) 标签：未来 horizon 日超额收益的 5档截面分位
    print(f"\n[3/4] 构造标签（未来{args.horizon}日超额收益截面分位，n_bins=5）...")
    close_p = {c: df["close"] for c, df in store.items() if c in feat_store}
    close_panel = pd.DataFrame(close_p).sort_index()
    cover = close_panel.notna().mean(axis=1)
    close_panel = close_panel.loc[cover >= 0.6]
    fwd_excess = forward_excess_return(close_panel, horizon=args.horizon, winsorize=0.01)
    labels_panel = make_rank_labels(fwd_excess, n_bins=5)
    print(f"  close 面板: {close_panel.shape} | 标签面板: {labels_panel.shape}")

    X_df, y, dates, codes = assemble_samples(feat_store, labels_panel, ref_cols)
    print(f"  ML 样本: {len(X_df)} × {len(ref_cols)} | 标签范围 {int(y.min())}~{int(y.max())}")

    # V3 样本权重（全量，供最终模型使用）
    sw_all = make_v3_sample_weights(y)

    # 4) 训练最终部署模型（全量 fit scaler，供部署推理统一口径）
    print("\n[4/4] 训练最终部署模型（全部数据 + V3 序数权重）...")
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_df[ref_cols].fillna(0.0).values)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_all_s, y, sample_weight=sw_all, verbose=False)
    print(f"  最终模型训练完成: n_classes={model.n_classes_}, "
          f"特征={model.n_features_in_}")

    # ---- 备份旧模型文件（若存在），再写新文件（与 predictor 加载口径同名）----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for fn in ("xgboost_model.pkl", "scaler.pkl", "label_encoder.pkl"):
        p = MODEL_DIR / fn
        if p.exists():
            bak = MODEL_DIR / f".bak_{ts}_{fn}"
            shutil.copy(p, bak)
            print(f"  已备份旧 {fn} -> {bak.name}")

    # LabelEncoder 在 [0..4] 上 fit（predictor 用 inverse_transform 还原档位）
    le = LabelEncoder()
    le.fit(np.arange(5))

    import pickle
    with open(MODEL_DIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    # 同步更新 SHAP 特征列表（predictor 可选读取，保证列名与模型一致）
    with open(MODEL_DIR / "features_xgboost.json", "w", encoding="utf-8") as f:
        json.dump({"features": ref_cols, "n_features": len(ref_cols)}, f,
                  ensure_ascii=False, indent=2)
    print(f"  新模型已保存: {MODEL_DIR / 'xgboost_model.pkl'}")
    print(f"  新 scaler 已保存: {MODEL_DIR / 'scaler.pkl'}")
    print(f"  新 label_encoder 已保存: {MODEL_DIR / 'label_encoder.pkl'} (classes={le.classes_.tolist()})")

    # ---- 诚实验证（可选）----
    production_summary: dict = {}
    if not args.no_validate:
        print("\n[验证] 运行诚实 walk-forward 样本外验证...")
        result = run_walk_forward_validation(
            X_df, y, dates, codes, fwd_excess, ref_cols, args
        )
        production_summary = result["summary"]
        if production_summary["rank_ic"] < 0.12:
            print("\n⚠️ WARNING: 汇总 walk-forward RankIC "
                  f"({production_summary['rank_ic']:.4f}) < 0.12，"
                  "仍已保存新模型（请关注是否达标）。")
        production_summary["n_folds"] = result["summary"].get("n_folds", len(result["folds"]))
    else:
        print("\n[验证] 已跳过（--no-validate）")

    # ---- 落盘生产指标 ----
    prod_metrics = {
        "config": {
            "pool": args.pool, "max_stocks": args.max_stocks,
            "horizon": args.horizon, "top_n": args.top_n,
            "n_splits": args.n_splits,
            "label": "forward_excess_return(horizon, winsorize=0.01) -> 5-bin rank label",
            "weight": f"V3 ordinal_cost (balanced x (1+{ALPHA}*|y-2|))",
            "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
            "n_samples": int(len(X_df)),
            "n_features": len(ref_cols),
            "leak_controls": [
                "scaler fit on train only (final model fit on all for inference)",
                "no SMOTE (class_weight only)",
                "embargo_days=HORIZON",
                "causal features (d depends on <=d)",
                "fwd_excess aligned by same date",
            ],
        },
        "walk_forward": production_summary if production_summary else "skipped",
        "saved_at": ts,
        "model_path": str(MODEL_DIR / "xgboost_model.pkl"),
    }
    with open(MODEL_DIR / "production_metrics.json", "w", encoding="utf-8") as f:
        json.dump(prod_metrics, f, ensure_ascii=False, indent=2, default=str)

    # ---- 总结 ----
    print("\n" + "=" * 78)
    print(" 生产模型训练总结")
    print("=" * 78)
    print(f"  样本量          : {len(X_df)} (×{len(ref_cols)} 特征)")
    print(f"  标签体系        : 5档截面分位 (0=最弱 ... 4=最强)")
    print(f"  样本权重        : V3 ordinal_cost，极端档(0/4)=3x，中性档(2)=1x")
    print(f"  XGBoost 参数    : n_est=400, depth=5, lr=0.03, subsample=0.9, "
          f"colsample=0.9, min_child=5")
    if production_summary:
        print(f"  WalkForward RankIC : {production_summary['rank_ic']:+.4f}")
        print(f"  IC_IR               : {production_summary['ic_ir']:+.2f}")
        print(f"  IC>0 比例           : {production_summary['ic_positive_rate']:.1%}")
        print(f"  Top{args.top_n} 超额    : {production_summary['top_excess']:+.4%}")
        print(f"  Top{args.top_n} 胜率    : {production_summary['top_win_rate']:.1%}")
    print(f"  新模型已保存路径 : {MODEL_DIR / 'xgboost_model.pkl'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

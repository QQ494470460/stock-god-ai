# -*- coding: utf-8 -*-
"""
训练「生产+alpha」可部署模型
===========================

把 walk_forward_alpha.py 的"对比实验"升级为"可部署模型"：

- 复用 walk_forward_alpha.py 的零泄漏流程，先在 walk-forward 各折上验证
  「生产+alpha（北向/融资/龙虎榜/行业动量）」模型 B 的样本外 RankIC；
- 若验证 RankIC 优于或等于生产模型，则**在全量样本上训练最终模型 B**，
  落盘为可直接被 predictor 加载的：
      models/alpha_xgboost_model.pkl
      models/alpha_scaler.pkl
      models/alpha_label_encoder.pkl
      models/alpha_model_metrics.json   # 含 RankIC / 特征列 / alpha 子集
- 模型注册中心（src/models/model_registry.py）会自动认出它并标记为「可部署最优」，
  predictor 的 AlphaSubPredictor 即可加载并参与超级集成。

离线能力
--------
数据来自本地 parquet（daily_*_qfq.parquet）与缓存的 alpha_*.parquet，
无需联网即可训练；若某 alpha 子集缓存缺失则自动跳过该子集（rank_ic 略降，不致命）。

用法
----
    py -3.12 scripts/train_alpha_model.py --max-stocks 150 --start-date 2021-01-01
    py -3.12 scripts/train_alpha_model.py --alpha margin lhb northbound --force
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from loguru import logger  # noqa: E402
logger.remove()
logger.add(sys.stderr, level="INFO")

from config.settings import MODEL_DIR  # noqa: E402
from scripts.train_v2 import load_pool  # noqa: E402
from src.features.engine import feature_engine  # noqa: E402
from src.models.labeling import (  # noqa: E402
    forward_excess_return,
    make_rank_labels,
    rank_ic,
)
from src.models.trainer_v2 import walk_forward_split  # noqa: E402
from src.data.akshare_alpha import build_alpha_features, ALPHA_SET_ALL  # noqa: E402
from sklearn.preprocessing import StandardScaler, LabelEncoder  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
import xgboost as xgb  # noqa: E402

MIN_STOCKS_PER_DAY = 30


# ---------------------------------------------------------------------------
# 复用 walk_forward_alpha.py 的零泄漏 helper（直接 import，避免逻辑分叉）
# ---------------------------------------------------------------------------
import scripts.walk_forward_alpha as wfa  # noqa: E402
from scripts.walk_forward_alpha import (  # noqa: E402
    build_feat_store,
    assemble_samples,
    _train_fold,
    _score_panel,
    _panel_daily_metrics,
)


# ---------------------------------------------------------------------------
# 在全部样本上训练"可部署"模型 B
# ---------------------------------------------------------------------------
def train_final_model(
    X_df: pd.DataFrame,
    cols_b: List[str],
    y: np.ndarray,
    n_classes: int,
) -> Tuple[object, StandardScaler, LabelEncoder]:
    """在全部样本上训练最终「生产+alpha」模型，返回 (model, scaler, le)。"""
    # 在 DataFrame 上 fit，使 scaler 保留 feature_names_in_（便于下游对齐输入列）
    X = X_df[cols_b].fillna(0.0).astype(float)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cw = compute_class_weight("balanced", classes=le.classes_, y=y_enc)
    sw = np.take(cw, y_enc)
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        gamma=0.2, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss",
    )
    model.fit(Xs, y_enc, sample_weight=sw, verbose=False)
    return model, scaler, le


def run_walk_forward_b(
    X_df: pd.DataFrame,
    cols_b: List[str],
    y: np.ndarray,
    dates: np.ndarray,
    codes: np.ndarray,
    fwd_excess: pd.DataFrame,
    args,
) -> dict:
    """walk-forward 验证模型 B，返回含 pooled RankIC 等的指标 dict。"""
    n_classes = int(pd.Series(y).nunique())
    splits = walk_forward_split(dates, n_splits=args.n_splits, embargo_days=args.horizon)
    logger.info(f"Walk-forward 切分: {len(splits)} 折（embargo={args.horizon} 天）")

    pool_d, pool_c, pool_sB = [], [], []
    folds = []
    for sp in splits:
        tr_mask = (dates >= sp.train_start) & (dates <= sp.train_end)
        va_mask = (dates >= sp.valid_start) & (dates <= sp.valid_end)
        n_tr, n_va = int(tr_mask.sum()), int(va_mask.sum())
        if n_tr < 100 or n_va < 30:
            logger.warning(f"  Fold {sp.fold + 1}: 样本不足(train={n_tr},valid={n_va})，跳过")
            continue
        scoreB = _train_fold(X_df, cols_b, y, tr_mask, va_mask, n_classes)
        pb = _score_panel(dates[va_mask], codes[va_mask], scoreB)
        fmB = _panel_daily_metrics(pb, fwd_excess, top_n=args.top_n)
        folds.append({"fold": sp.fold + 1, **fmB})
        logger.info(
            f"  Fold {sp.fold + 1}: RankIC(B)={fmB['rank_ic']:+.4f} | "
            f"TopN超额={fmB['top_excess']:+.4%} | 胜率={fmB['top_win_rate']:.1%}"
        )
        pool_d.append(dates[va_mask]); pool_c.append(codes[va_mask]); pool_sB.append(scoreB)

    if not folds:
        raise RuntimeError("没有有效折可训练")

    pooledB = _score_panel(np.concatenate(pool_d), np.concatenate(pool_c), np.concatenate(pool_sB))
    summary = _panel_daily_metrics(pooledB, fwd_excess, top_n=args.top_n)
    summary["n_folds"] = len(folds)
    return {"summary": summary, "folds": folds}


def save_deployable(model, scaler, le, alpha_cols, ref_cols, summary, args, alpha_set) -> Path:
    out_dir = MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(out_dir / "alpha_xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(out_dir / "alpha_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "alpha_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    metrics = {
        "model_type": "alpha_xgboost",
        "rank_ic": float(summary["rank_ic"]),
        "ic_ir": float(summary["ic_ir"]) if not np.isnan(summary["ic_ir"]) else None,
        "top_win_rate": float(summary["top_win_rate"]) if not np.isnan(summary["top_win_rate"]) else None,
        "top_excess": float(summary["top_excess"]) if summary["top_excess"] is not None else None,
        "n_days": int(summary["n_days"]),
        "n_folds": int(summary.get("n_folds", 0)),
        "model_path": "alpha_xgboost_model.pkl",
        "alpha_features": list(alpha_cols),
        "feature_cols": list(ref_cols),
        "alpha_set": sorted(alpha_set),
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "config": {
            "max_stocks": args.max_stocks, "n_splits": args.n_splits,
            "horizon": args.horizon, "top_n": args.top_n,
            "start_date": args.start_date,
            "end_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        },
    }
    mp = out_dir / "alpha_model_metrics.json"
    mp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(f"✅ 可部署 alpha 模型已落盘: {mp}")
    return mp


def main() -> int:
    ap = argparse.ArgumentParser(description="训练「生产+alpha」可部署 XGBoost 模型")
    ap.add_argument("--max-stocks", type=int, default=150)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--start-date", default="2021-01-01")
    ap.add_argument("--alpha", nargs="*", default=None, choices=list(ALPHA_SET_ALL),
                    help="alpha 子集，默认全开 {northbound,margin,lhb,industry}")
    ap.add_argument("--force", action="store_true",
                    help="即便验证 RankIC 低于生产模型也强制保存")
    args = ap.parse_args()

    alpha_set = set(args.alpha) if args.alpha else set(ALPHA_SET_ALL)
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    print("=" * 88)
    print(" 选股神 · 训练「生产+alpha」可部署模型（零泄漏 walk-forward）")
    print("=" * 88)

    # 1) 数据（离线 parquet）
    print(f"\n[1/5] 加载数据（最多 {args.max_stocks} 只）...")
    store = load_pool(args.max_stocks, pool="legacy")
    if len(store) < 50:
        print("  数据不足，退出"); return 1
    print(f"  有效股票: {len(store)} 只")

    # 2) 生产特征
    print("\n[2/5] 构建生产特征...")
    ref_cols = build_feat_store(store)
    print(f"  特征列数: {len(ref_cols)} | 有效特征股: {len(wfa._FEAT_STORE)} 只")

    close_p = {c: df["close"] for c, df in store.items() if c in wfa._FEAT_STORE}
    close_panel = pd.DataFrame(close_p).sort_index()
    cover = close_panel.notna().mean(axis=1)
    close_panel = close_panel.loc[cover >= 0.6]
    fwd_excess = forward_excess_return(close_panel, horizon=args.horizon)
    labels_panel = make_rank_labels(fwd_excess, n_bins=5)

    # 3) alpha 特征（优先读本地缓存，离线可用）
    print(f"\n[3/5] 读取 alpha 特征（区间 {args.start_date}~{end_date}）...")
    codes_universe = [c for c in wfa._FEAT_STORE if c in close_panel.columns]
    try:
        alpha = build_alpha_features(codes_universe, close_panel.index, args.start_date, end_date, alpha_set)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  alpha 特征构建失败（将退化为纯生产模型）: {e}")
        alpha = None
    if alpha is None or len(alpha) == 0:
        alpha_cols: List[str] = []
    else:
        alpha_cols = [c for c in alpha.columns]
    print(f"  alpha 特征列: {alpha_cols}")

    # 4) 组装样本 + walk-forward 验证 B
    print(f"\n[4/5] 组装样本 + walk-forward 验证「生产+alpha」模型 B...")
    X_df, y, dates, codes = assemble_samples(labels_panel, ref_cols)
    if alpha_cols:
        midx = pd.MultiIndex.from_arrays([pd.to_datetime(X_df["__date__"]), X_df["__code__"].astype(str)])
        alpha_a = alpha.reindex(midx)
        for c in alpha_cols:
            X_df[c] = alpha_a[c].values
        X_df[alpha_cols] = X_df[alpha_cols].fillna(0.0)
    else:
        logger.warning("  无 alpha 特征，模型 B == 生产模型，rank_ic 不会提升")
    cols_b = ref_cols + alpha_cols

    result = run_walk_forward_b(X_df, cols_b, y, dates, codes, fwd_excess, args)
    s = result["summary"]
    ricB = s["rank_ic"]
    print(f"\n  验证 RankIC(B)={ricB:+.4f} | IC_IR={s['ic_ir']:+.2f} | "
          f"TopN超额={s['top_excess']:+.4%} | 胜率={s['top_win_rate']:.1%} | 天数={s['n_days']}")

    # 5) 在全部样本上训练最终模型并落盘
    print(f"\n[5/5] 在全部样本上训练最终模型并落盘...")
    model, scaler, le = train_final_model(X_df, cols_b, y, int(pd.Series(y).nunique()))
    mp = save_deployable(model, scaler, le, alpha_cols, ref_cols, s, args, alpha_set)
    print(f"\n✅ 完成。指标文件: {mp}")
    print(f"   注册中心将自动认出此模型（RankIC={ricB:+.4f}）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
严格、零泄漏的 walk-forward 样本外（OOS）评估框架
=================================================

目的
----
把当前 A 股选股模型的**真实** RankIC 钉死，作为后续所有"准确率提升"方向的
回归基线。本脚本复用**生产特征管线**（`src/features/engine.py` 的
`build_features`，逐股技术面特征），但用正确的**按日期 walk-forward 切分**
+ 零泄漏训练，给出诚实的全周期样本外 RankIC。

两套口径
--------
1. walk_forward 模式（核心）：每折只在 train 段 fit scaler、只在 train 内
   编码标签 / 算 class_weight，验证段只 transform；`embargo_days=HORIZON`
   隔离训练段末端标签伸进验证段的重叠。逐折重训，最终拼接全量 OOS 面板算
   RankIC / IC_IR / TopN 超额。
2. deployed 模式：直接加载**真正部署**的 `xgboost_model.pkl` + `scaler.pkl`
   + `features_xgboost.json`，固定模型在全周期向前测（非重训），给出部署
   模型在**全周期**的真实 RankIC。

零泄漏防护（见代码注释 ★）
--------------------------
★ scaler 仅 train 段 fit，验证段只 transform（问题1）
★ 无 SMOTE（默认关；SMOTE 只在 train 内做才有意义，这里直接用 class_weight）
★ embargo = HORIZON，隔离标签窗口重叠（问题：train 末端样本标签伸进 valid）
★ 特征因果：build_features 的值只依赖 d 及之前的数据，无未来函数
★ 目标只用同日期前瞻收益：绝不用 d 的标签预测 d，标签是 d 日的未来 HORIZON 日
★ 样本外预测分数与真实收益按 (date, code) 对齐，绝不在拼接后再 fit

用法
----
    py -3.12 scripts/walk_forward_oos.py --mode both --max-stocks 200 --n-splits 5
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- 项目路径 & 环境 ----
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
    top_portfolio_return,
    rank_ic,
)
from src.models.trainer_v2 import walk_forward_split  # noqa: E402

from sklearn.preprocessing import StandardScaler, LabelEncoder  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
import xgboost as xgb  # noqa: E402


# 最小每日有效股票数（与 src/models/labeling.py 保持一致）
MIN_STOCKS_PER_DAY = 30


def _panel_daily_metrics(
    score_panel: pd.DataFrame,
    fwd_excess: pd.DataFrame,
    top_n: int = 20,
) -> dict:
    """逐日计算 RankIC / IC_IR / IC>0 比例 / TopN 超额 / TopN 胜率。

    与 ``mean_rank_ic`` / ``top_portfolio_return`` 口径一致，但额外返回
    逐日 IC 序列派生的 **IC>0 比例**，且 TopN 超额是相对全市场均值的日频超额。

    Spearman 只看秩，因此不同折的"期望档位"分数即便落在不同尺度上，
    在**同一交易日**内排序仍然有效（每个交易日只属于一个折，不会跨折混合）。
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
        y    : np.ndarray(int) 标签
        dates: np.ndarray(pd.Timestamp) 该样本的交易日
        codes: np.ndarray(str) 该样本的股票代码
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
        # 丢弃标签为 NaN 的样本
        dfc = dfc.dropna(subset=["__label__"])
        # 丢弃特征全 NaN 的样本
        valid_feat = dfc[ref_cols].notna().any(axis=1)
        # ★ 某股有效特征行不足 130 → 整体跳过（已在构建端做，这里再保险）
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


# 全局特征缓存（构建一次，walk_forward / deployed 共用）
_FEAT_STORE: Dict[str, pd.DataFrame] = {}


def build_feat_store(store: Dict[str, pd.DataFrame]) -> List[str]:
    """逐股构建生产特征并缓存到全局；返回参考列名（ref_cols）。"""
    global _FEAT_STORE
    _FEAT_STORE = {}
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
        _FEAT_STORE[code] = feats
    return ref_cols or []


def run_deployed(args) -> Optional[dict]:
    """加载真正部署的模型，固定模型在全周期向前测（非重训）。"""
    model_path = MODEL_DIR / "xgboost_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    feat_path = MODEL_DIR / "features_xgboost.json"

    if not (model_path.exists() and scaler_path.exists()):
        logger.warning("  ⚠️ 缺少 xgboost_model.pkl 或 scaler.pkl，跳过 deployed 模式")
        print("\n[deployed] 跳过：缺少 xgboost_model.pkl 或 scaler.pkl")
        return None

    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # SHAP 特征列表（决定用哪些列）——与部署推理口径一致
    shap_cols: List[str] = []
    if feat_path.exists():
        try:
            with open(feat_path, "r", encoding="utf-8") as f:
                shap_cols = json.load(f).get("features", [])
        except Exception:  # noqa: BLE001
            shap_cols = []

    # 以 scaler 实际训练时使用的列为准（模型输入维度的权威来源），
    # 缺失时退化为 SHAP 列表 / 全特征列
    _fn = getattr(scaler, "feature_names_in_", None)
    scaler_cols = list(_fn) if _fn is not None and len(_fn) > 0 else []
    if scaler_cols:
        dep_cols = scaler_cols
    elif shap_cols:
        dep_cols = shap_cols
    else:
        dep_cols = _FEAT_STORE and list(next(iter(_FEAT_STORE.values())).columns) or []

    logger.info(f"  deployed 特征列: {len(dep_cols)} 个（scaler 维度={getattr(scaler,'n_features_in_',None)}）")

    # 目标面板（与 walk_forward 同一口径，保证可比）
    close_p = {c: df["close"] for c, df in _store().items() if c in _FEAT_STORE}
    close_panel = pd.DataFrame(close_p).sort_index()
    cover = close_panel.notna().mean(axis=1)
    close_panel = close_panel.loc[cover >= 0.6]
    fwd_excess = forward_excess_return(close_panel, horizon=args.horizon)

    # 固定模型：逐股一次性打分（向量化，远快于逐日循环）
    score_by_code: Dict[str, pd.Series] = {}
    for code, feat in _FEAT_STORE.items():
        if code not in close_panel.columns:
            continue
        f = feat.reindex(columns=dep_cols)
        if f.isna().all().all():
            continue
        try:
            Xs = scaler.transform(f.fillna(0.0).values)
            proba = model.predict_proba(Xs)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  deployed 预测失败 {code}: {e}")
            continue
        score = proba @ np.arange(proba.shape[1], dtype=float)
        score_by_code[code] = pd.Series(score, index=f.index)

    records = []
    for code, s in score_by_code.items():
        for d, v in s.items():
            if d in close_panel.index:
                records.append((d, code, float(v)))

    if not records:
        logger.warning("  ⚠️ deployed 模式没有任何可用分数，跳过")
        print("\n[deployed] 跳过：无任何可用分数")
        return None

    sp_df = pd.DataFrame(records, columns=["date", "code", "score"]).drop_duplicates(
        subset=["date", "code"], keep="last"
    )
    score_panel = sp_df.pivot(index="date", columns="code", values="score")
    summary = _panel_daily_metrics(score_panel, fwd_excess, top_n=args.top_n)

    print("\n" + "=" * 78)
    print(" Deployed 模型评估（固定模型全周期向前测，非重训）")
    print("=" * 78)
    print(f" {'模型':<12}{'RankIC':>10}{'IC_IR':>10}{'IC>0比例':>11}"
          f"{'TopN超额':>12}{'TopN胜率':>10}{'天数':>6}")
    print(" " + "-" * 72)
    print(f" {'deployed':<12}{summary['rank_ic']:>10.4f}{summary['ic_ir']:>10.2f}"
          f"{summary['ic_positive_rate']:>11.1%}{summary['top_excess']:>12.4%}"
          f"{summary['top_win_rate']:>10.1%}{summary['n_days']:>6}")
    print("=" * 78)

    return {"summary": summary, "n_features": len(dep_cols)}


# 全局 store 缓存（供 deployed 反查 close）
_STORE: Dict[str, pd.DataFrame] = {}


def _store() -> Dict[str, pd.DataFrame]:
    return _STORE


def main() -> int:
    ap = argparse.ArgumentParser(description="严格零泄漏 walk-forward 样本外评估基线")
    ap.add_argument("--mode", choices=["walk_forward", "deployed", "both"], default="both")
    ap.add_argument("--max-stocks", type=int, default=200)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--label-mode", choices=["rank", "abs"], default="rank")
    ap.add_argument("--pool", default="hs300",
                    help="股票池（真实生效）：hs300/zz500/hs300_zz500/all；"
                         "legacy=旧口径按代码排序取前 N 只")
    args = ap.parse_args()

    print("=" * 78)
    print(" 选股神 · 严格零泄漏 Walk-Forward 样本外基线")
    print("=" * 78)

    # 1) 数据
    print(f"\n[1/3] 加载数据（最多 {args.max_stocks} 只，离线 parquet）...")
    global _STORE
    _STORE = load_pool(args.max_stocks, pool=args.pool)
    if len(_STORE) < 50:
        print("  数据不足，退出")
        return 1
    print(f"  股票池: {args.pool} | 有效股票: {len(_STORE)} 只 | "
          f"前5个代码: {list(_STORE.keys())[:5]}")

    # 2) 特征（构建一次，两种模式共用）
    print("\n[2/3] 构建生产特征（build_features，剔除 close）...")
    ref_cols = build_feat_store(_STORE)
    print(f"  特征列数: {len(ref_cols)} | 有效特征股: {len(_FEAT_STORE)} 只")

    # 3) 评估
    print("\n[3/3] 评估...")
    result: dict = {
        "config": {
            "mode": args.mode, "max_stocks": args.max_stocks,
            "n_splits": args.n_splits, "n_bins": args.n_bins,
            "horizon": args.horizon, "top_n": args.top_n,
            "label_mode": args.label_mode, "pool": args.pool,
        }
    }

    if args.mode in ("walk_forward", "both"):
        result["walk_forward"] = run_walk_forward_shared(args, ref_cols)
        # ★ 增量落盘：walk_forward 是核心基线，先存盘，避免后续 deployed 失败丢失
        _save_baseline(result)

    if args.mode in ("deployed", "both"):
        try:
            result["deployed"] = run_deployed(args)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"deployed 模式异常，跳过: {e}")
            print(f"\n[deployed] 异常跳过: {e}")
            result["deployed"] = None

    # ---- 落盘 ----
    _save_baseline(result)

    # ---- 汇总对照表（模仿 eval_model_oos 表头）----
    print("\n" + "=" * 78)
    print(" 汇总对照表")
    print("=" * 78)
    print(f" {'模型':<12}{'RankIC':>10}{'IC_IR':>10}{'IC>0比例':>11}"
          f"{'TopN超额':>12}{'TopN胜率':>10}{'天数':>6}")
    print(" " + "-" * 72)
    if args.mode in ("walk_forward", "both"):
        s = result["walk_forward"]["summary"]
        print(f" {'walk_forward':<12}{s['rank_ic']:>10.4f}{s['ic_ir']:>10.2f}"
              f"{s['ic_positive_rate']:>11.1%}{s['top_excess']:>12.4%}"
              f"{s['top_win_rate']:>10.1%}{s['n_days']:>6}")
    if args.mode in ("deployed", "both") and result.get("deployed"):
        s = result["deployed"]["summary"]
        print(f" {'deployed':<12}{s['rank_ic']:>10.4f}{s['ic_ir']:>10.2f}"
              f"{s['ic_positive_rate']:>11.1%}{s['top_excess']:>12.4%}"
              f"{s['top_win_rate']:>10.1%}{s['n_days']:>6}")
    print(" " + "-" * 72)
    print(" 参考：随机猜 RankIC≈0；读数异常（≈0 或为负）本身也是真实结论。")
    print("=" * 78)
    return 0


def _save_baseline(result: dict) -> None:
    """把基线结果落盘到 MODEL_DIR/walk_forward_baseline.json。"""
    out_f = MODEL_DIR / "walk_forward_baseline.json"
    try:
        out_f.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n基线已保存: {out_f}")
    except Exception as e:  # noqa: BLE001
        print(f"保存失败: {e}")


def run_walk_forward_shared(args, ref_cols: List[str]) -> dict:
    """与 run_walk_forward 等价的实现，但复用已构建的全局 _FEAT_STORE。"""
    logger.info("构建目标面板...")
    store = _STORE
    close_p = {c: df["close"] for c, df in store.items() if c in _FEAT_STORE}
    close_panel = pd.DataFrame(close_p).sort_index()
    cover = close_panel.notna().mean(axis=1)
    close_panel = close_panel.loc[cover >= 0.6]
    fwd_excess = forward_excess_return(close_panel, horizon=args.horizon)

    labels_panel = make_rank_labels(fwd_excess, n_bins=args.n_bins) if args.label_mode == "rank" \
        else _abs_labels(store, close_panel, args.horizon)

    X_df, y, dates, codes = assemble_samples(_FEAT_STORE, labels_panel, ref_cols)

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

        # ★ 标签编码只在 train 内 fit；无 SMOTE（用 class_weight 替代）
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
        classes = le.classes_
        # ★ compute_class_weight 返回"每类"权重，需映射到每个样本
        cw = compute_class_weight("balanced", classes=classes, y=y_tr_enc)
        sw = np.take(cw, y_tr_enc)

        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss",
        )
        model.fit(Xtr_s, y_tr_enc, sample_weight=sw, verbose=False)

        # ★ 分数 = 期望档位（单调，与 eval_model_oos.score_of 多分类分支一致）
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
        raise RuntimeError("没有有效折可训练")

    pool = pd.DataFrame({
        "date": np.concatenate(pool_dates),
        "code": np.concatenate(pool_codes),
        "score": np.concatenate(pool_scores),
    }).drop_duplicates(subset=["date", "code"], keep="last")
    pooled_panel = pool.pivot(index="date", columns="code", values="score")
    summary = _panel_daily_metrics(pooled_panel, fwd_excess, top_n=args.top_n)
    summary["n_folds"] = len(fold_records)

    print("\n" + "=" * 78)
    print(" Walk-Forward 样本外评估（生产特征管线，零泄漏）")
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


def _abs_labels(store, close_panel, horizon) -> pd.DataFrame:
    labels_panel = pd.DataFrame(index=close_panel.index, columns=close_panel.columns, dtype=float)
    for code, df in store.items():
        if code not in labels_panel.columns:
            continue
        try:
            labels_panel[code] = feature_engine.generate_labels(df, forecast_days=horizon)
        except Exception:  # noqa: BLE001
            continue
    return labels_panel


if __name__ == "__main__":
    sys.exit(main())

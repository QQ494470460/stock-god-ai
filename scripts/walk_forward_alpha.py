# -*- coding: utf-8 -*-
"""
严格、零泄漏的 walk-forward：生产特征 vs 生产+alpha 双模型对比
============================================================

目的
----
用 akshare 免费接口补的「核心 alpha 特征」（北向资金 / 融资余额 / 龙虎榜 / 行业动量）
到底对 RankIC 有没有真实贡献？在严格、零泄漏的 walk-forward 下量化。

做法（同窗口双模型对比）
------------------------
每个 walk-forward 折都训练**两个**模型，共用同一份切分（embargo=HORIZON）：
  - 模型 A「生产仅特征」：只用 engine 生产特征
  - 模型 B「生产+alpha」：engine 生产特征 + alpha 特征
二者 scaler 都只在 train 段 fit（B 的 scaler 覆盖 alpha 列），
sample_weight=compute_class_weight("balanced")，分数=proba @ arange(n_classes)。

核心输出是 Δ = (B - A) 的 RankIC 差（alpha 贡献）。

零泄漏防护（见代码注释 ★）
--------------------------
★ scaler 仅 train 段 fit，验证段只 transform（问题1）
★ 无 SMOTE（用 class_weight）
★ embargo = HORIZON，隔离标签窗口重叠（问题：train 末端样本标签伸进 valid）
★ alpha 特征全部只用 发布/统计日期 <= 预测日 的数据（见 src/data/akshare_alpha.py）
★ 生产特征 causal：build_features 值只依赖 d 及之前的数据，无未来函数
★ 样本外分数与真实收益按 (date, code) 对齐，拼接后只算指标、绝不二次 fit

用法
----
    py -3.12 scripts/walk_forward_alpha.py --max-stocks 120 --start-date 2023-01-01
    py -3.12 scripts/walk_forward_alpha.py --max-stocks 120 --start-date 2023-01-01 \
        --alpha margin lhb          # 只看部分子集
    py -3.12 scripts/walk_forward_alpha.py --max-stocks 60 --n-splits 3 --ablation
"""

import sys
import json
import argparse
import warnings
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
    rank_ic,
)
from src.models.trainer_v2 import walk_forward_split  # noqa: E402
from sklearn.preprocessing import StandardScaler, LabelEncoder  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
import xgboost as xgb  # noqa: E402

from src.data.akshare_alpha import build_alpha_features, ALPHA_SET_ALL  # noqa: E402

# 最小每日有效股票数（与 src/models/labeling.py 保持一致）
MIN_STOCKS_PER_DAY = 30


# ---------------------------------------------------------------------------
# 指标：逐日 RankIC / IC_IR / IC>0 / TopN 超额 / 胜率 / 天数
# ---------------------------------------------------------------------------
def _panel_daily_metrics(
    score_panel: pd.DataFrame,
    fwd_excess: pd.DataFrame,
    top_n: int = 20,
) -> dict:
    """逐日计算 RankIC / IC_IR / IC>0 比例 / TopN 超额 / TopN 胜率。

    Spearman 只看秩，因此不同折的"期望档位"分数即便落在不同尺度上，
    在同一交易日内排序仍然有效（每个交易日只属于一个折，不会跨折混合）。
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


# ---------------------------------------------------------------------------
# 生产特征构建（同 walk_forward_oos）
# ---------------------------------------------------------------------------
_FEAT_STORE: Dict[str, pd.DataFrame] = {}
_STORE: Dict[str, pd.DataFrame] = {}


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


def assemble_samples(
    labels_panel: pd.DataFrame,
    ref_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """把生产特征与标签拼成 ML 样本（一个 (date, code) 一条）。

    Returns (X_df, y, dates, codes)
        X_df : DataFrame(columns=ref_cols + ['__date__','__code__'])
    """
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    date_parts: List[np.ndarray] = []
    code_parts: List[np.ndarray] = []

    for code, feat in _FEAT_STORE.items():
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
        f"  组装 ML 样本: {len(X_df)} 条 × {len(ref_cols)} 生产特征 | "
        f"标签分布: " + ", ".join(
            f"{int(k)}={v:.1%}" for k, v in
            pd.Series(y).value_counts(normalize=True).sort_index().items()
        )
    )
    return X_df, y, dates, codes


# ---------------------------------------------------------------------------
# 单模型训练（A 或 B），返回验证段分数
# ---------------------------------------------------------------------------
def _train_fold(
    X_df: pd.DataFrame,
    cols: List[str],
    y: np.ndarray,
    tr_mask: np.ndarray,
    va_mask: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """训练单折单模型，返回验证段期望档位分数（proba @ arange(n_classes)）。"""
    X_tr = X_df[cols][tr_mask].fillna(0.0)
    X_va = X_df[cols][va_mask].fillna(0.0)
    y_tr = y[tr_mask]

    # ★ scaler 仅 train 段 fit（防泄漏）；验证段只 transform
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xva_s = scaler.transform(X_va)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    classes = le.classes_
    cw = compute_class_weight("balanced", classes=classes, y=y_tr_enc)
    sw = np.take(cw, y_tr_enc)

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        gamma=0.2, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss",
    )
    model.fit(Xtr_s, y_tr_enc, sample_weight=sw, verbose=False)

    proba = model.predict_proba(Xva_s)
    # 分数 = 期望档位（单调）
    return proba @ np.arange(proba.shape[1], dtype=float)


def _score_panel(dates: np.ndarray, codes: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "code": codes.astype(str),
        "score": scores,
    }).drop_duplicates(subset=["date", "code"], keep="last")
    return df.pivot(index="date", columns="code", values="score")


# ---------------------------------------------------------------------------
# 双模型 walk-forward 主流程
# ---------------------------------------------------------------------------
def run_dual(
    args,
    ref_cols: List[str],
    alpha: pd.DataFrame,
    fwd_excess: pd.DataFrame,
    labels_panel: pd.DataFrame,
) -> dict:
    """同窗口训练 A（仅生产）与 B（生产+alpha），返回含 Δ 的结果 dict。"""
    X_df, y, dates, codes = assemble_samples(labels_panel, ref_cols)

    # 对齐 alpha 到 X_df 的 (date, code)
    alpha_cols = list(alpha.columns) if alpha is not None and len(alpha) else []
    if alpha_cols:
        midx = pd.MultiIndex.from_arrays([
            pd.to_datetime(X_df["__date__"]), X_df["__code__"].astype(str)])
        alpha_a = alpha.reindex(midx)
        for c in alpha_cols:
            X_df[c] = alpha_a[c].values
        X_df[alpha_cols] = X_df[alpha_cols].fillna(0.0)
        logger.info(f"  alpha 特征列: {alpha_cols}")
    else:
        logger.warning("  未获得任何 alpha 特征，模型 B == 模型 A（Δ 应为 0）")

    cols_b = ref_cols + alpha_cols
    n_classes = int(pd.Series(y).nunique())

    splits = walk_forward_split(dates, n_splits=args.n_splits, embargo_days=args.horizon)
    logger.info(f"Walk-forward 切分: {len(splits)} 折（embargo={args.horizon} 天）")

    pool_d, pool_c, pool_sA, pool_sB = [], [], [], []
    folds_A, folds_B = [], []

    for sp in splits:
        tr_mask = (dates >= sp.train_start) & (dates <= sp.train_end)
        va_mask = (dates >= sp.valid_start) & (dates <= sp.valid_end)
        n_tr, n_va = int(tr_mask.sum()), int(va_mask.sum())
        if n_tr < 100 or n_va < 30:
            logger.warning(f"  Fold {sp.fold + 1}: 样本不足(train={n_tr},valid={n_va})，跳过")
            continue

        scoreA = _train_fold(X_df, ref_cols, y, tr_mask, va_mask, n_classes)
        scoreB = _train_fold(X_df, cols_b, y, tr_mask, va_mask, n_classes)

        pa = _score_panel(dates[va_mask], codes[va_mask], scoreA)
        pb = _score_panel(dates[va_mask], codes[va_mask], scoreB)
        fmA = _panel_daily_metrics(pa, fwd_excess, top_n=args.top_n)
        fmB = _panel_daily_metrics(pb, fwd_excess, top_n=args.top_n)

        folds_A.append({"fold": sp.fold + 1,
                        "train": f"{sp.train_start.date()}~{sp.train_end.date()}",
                        "valid": f"{sp.valid_start.date()}~{sp.valid_end.date()}",
                        "n_train": n_tr, "n_valid": n_va, **fmA})
        folds_B.append({"fold": sp.fold + 1,
                        "train": f"{sp.train_start.date()}~{sp.train_end.date()}",
                        "valid": f"{sp.valid_start.date()}~{sp.valid_end.date()}",
                        "n_train": n_tr, "n_valid": n_va, **fmB})
        logger.info(
            f"  Fold {sp.fold + 1}: RankIC A={fmA['rank_ic']:+.4f} | "
            f"B={fmB['rank_ic']:+.4f} | Δ={fmB['rank_ic']-fmA['rank_ic']:+.4f}"
        )

        pool_d.append(dates[va_mask]); pool_c.append(codes[va_mask])
        pool_sA.append(scoreA); pool_sB.append(scoreB)

    if not folds_A:
        raise RuntimeError("没有有效折可训练")

    pooledA = _score_panel(np.concatenate(pool_d), np.concatenate(pool_c),
                          np.concatenate(pool_sA))
    pooledB = _score_panel(np.concatenate(pool_d), np.concatenate(pool_c),
                          np.concatenate(pool_sB))
    sumA = _panel_daily_metrics(pooledA, fwd_excess, top_n=args.top_n)
    sumB = _panel_daily_metrics(pooledB, fwd_excess, top_n=args.top_n)
    sumA["n_folds"] = sumB["n_folds"] = len(folds_A)

    delta = {k: (sumB[k] - sumA[k])
             for k in ["rank_ic", "ic_ir", "ic_positive_rate", "top_excess", "top_win_rate"]}
    delta["n_days_A"] = sumA["n_days"]
    delta["n_days_B"] = sumB["n_days"]

    return {
        "model_A": {"summary": sumA, "folds": folds_A},
        "model_B": {"summary": sumB, "folds": folds_B},
        "delta_B_minus_A": delta,
        "alpha_features": alpha_cols,
    }


# ---------------------------------------------------------------------------
# 结果打印 & 落盘
# ---------------------------------------------------------------------------
def _fmt(v, kind="f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  n/a"
    if kind == "pct":
        return f"{v:>+11.4%}"
    if kind == "ir":
        return f"{v:>+8.2f}"
    return f"{v:>+10.4f}"


def print_comparison(result: dict, alpha_set, ablation=None) -> None:
    A = result["model_A"]["summary"]
    B = result["model_B"]["summary"]
    D = result["delta_B_minus_A"]
    print("\n" + "=" * 88)
    print(" Walk-Forward 双模型对比（零泄漏 · 同窗口 · A=生产 / B=生产+alpha）")
    print("=" * 88)
    print(f" {'指标':<12}{'模型A(生产)':>16}{'模型B(生产+α)':>18}{'Δ=B-A':>16}")
    print(" " + "-" * 62)
    print(f" {'RankIC':<12}{_fmt(A['rank_ic']):>16}{_fmt(B['rank_ic']):>18}{_fmt(D['rank_ic']):>16}")
    print(f" {'IC_IR':<12}{_fmt(A['ic_ir'],'ir'):>16}{_fmt(B['ic_ir'],'ir'):>18}{_fmt(D['ic_ir'],'ir'):>16}")
    print(f" {'IC>0比例':<12}{_fmt(A['ic_positive_rate'],'pct'):>16}{_fmt(B['ic_positive_rate'],'pct'):>18}{_fmt(D['ic_positive_rate'],'pct'):>16}")
    print(f" {'TopN超额':<12}{_fmt(A['top_excess'],'pct'):>16}{_fmt(B['top_excess'],'pct'):>18}{_fmt(D['top_excess'],'pct'):>16}")
    print(f" {'TopN胜率':<12}{_fmt(A['top_win_rate'],'pct'):>16}{_fmt(B['top_win_rate'],'pct'):>18}{_fmt(D['top_win_rate'],'pct'):>16}")
    print(f" {'天数':<12}{str(A['n_days']):>16}{str(B['n_days']):>18}{'':>16}")
    print(" " + "-" * 62)
    sign = "✅ alpha 真的加分" if D['rank_ic'] > 0 else "❌ alpha 未加分（甚至拖累）"
    print(f" 结论: Δ(RankIC) = {D['rank_ic']:+.4f}  →  {sign}")
    print(f" 启用 alpha 子集: {sorted(alpha_set)} | alpha 特征列: {result.get('alpha_features')}")

    if ablation:
        print("\n" + "-" * 62)
        print(" 消融（leave-one-out：关掉某子集后 B 相对 A 的 Δ RankIC）：")
        for sub, d in ablation.items():
            print(f"   去掉 {sub:<10} Δ={d:+.4f}  →  "
                  f"{'仍有贡献' if d>0 else '几乎无差/拖累'}")
    print("=" * 88)


def save_result(result: dict, args, alpha_set) -> Path:
    out = MODEL_DIR / "walk_forward_alpha.json"
    payload = {
        "config": {
            "max_stocks": args.max_stocks, "n_splits": args.n_splits,
            "horizon": args.horizon, "top_n": args.top_n,
            "start_date": args.start_date, "end_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "alpha_set": sorted(alpha_set),
        },
        **result,
    }
    try:
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        print(f"\n结果已保存: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"保存失败: {e}")
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="零泄漏 walk-forward：生产 vs 生产+alpha 双模型")
    ap.add_argument("--max-stocks", type=int, default=120)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--start-date", default="2023-01-01",
                    help="限制 akshare 拉取区间的起点（控制耗时）")
    ap.add_argument("--alpha", nargs="*", default=None,
                    choices=list(ALPHA_SET_ALL),
                    help="alpha 子集，默认全开 {northbound,margin,lhb,industry}")
    ap.add_argument("--ablation", action="store_true",
                    help="额外跑 leave-one-out 消融，量化每个子集的单独影响")
    args = ap.parse_args()

    alpha_set = set(args.alpha) if args.alpha else set(ALPHA_SET_ALL)
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    print("=" * 88)
    print(" 选股神 · 零泄漏 Walk-Forward：生产特征 vs 生产+alpha")
    print("=" * 88)

    # 1) 数据
    print(f"\n[1/4] 加载数据（最多 {args.max_stocks} 只，离线 parquet）...")
    global _STORE
    _STORE = load_pool(args.max_stocks)
    if len(_STORE) < 50:
        print("  数据不足，退出")
        return 1
    print(f"  有效股票: {len(_STORE)} 只")

    # 2) 生产特征
    print("\n[2/4] 构建生产特征（build_features，剔除 close）...")
    ref_cols = build_feat_store(_STORE)
    print(f"  特征列数: {len(ref_cols)} | 有效特征股: {len(_FEAT_STORE)} 只")

    # 价格矩阵 & 标签
    close_p = {c: df["close"] for c, df in _STORE.items() if c in _FEAT_STORE}
    close_panel = pd.DataFrame(close_p).sort_index()
    cover = close_panel.notna().mean(axis=1)
    close_panel = close_panel.loc[cover >= 0.6]
    print(f"  日期轴: {close_panel.index[0].date()} ~ {close_panel.index[-1].date()} "
          f"({len(close_panel)} 天)")
    fwd_excess = forward_excess_return(close_panel, horizon=args.horizon)
    labels_panel = make_rank_labels(fwd_excess, n_bins=5)

    # 3) alpha 特征
    print(f"\n[3/4] 拉取 akshare alpha 特征（区间 {args.start_date}~{end_date}）...")
    codes_universe = [c for c in _FEAT_STORE if c in close_panel.columns]
    alpha = build_alpha_features(
        codes_universe, close_panel.index, args.start_date, end_date, alpha_set)
    print(f"  alpha 特征表: shape={alpha.shape} 列={list(alpha.columns)}")
    # 如实报告缺失的 subset
    present = set(c.split("_", 1)[1] if c.startswith("alpha_") else c
                  for c in alpha.columns)
    missing = alpha_set - present
    if missing:
        print(f"  注意：以下 alpha 子集本次无数据（接口失败/无覆盖），已跳过：{sorted(missing)}")

    # 4) 双模型 walk-forward
    print(f"\n[4/4] 同窗口双模型 walk-forward（A=生产 / B=生产+alpha）...")
    result = run_dual(args, ref_cols, alpha, fwd_excess, labels_panel)

    print_comparison(result, alpha_set)

    # 消融（可选）
    if args.ablation:
        print("\n[消融] leave-one-out：逐个关掉 alpha 子集，重跑双模型...")
        ablation = {}
        for sub in sorted(alpha_set):
            sub_set = alpha_set - {sub}
            a2 = build_alpha_features(
                codes_universe, close_panel.index, args.start_date, end_date, sub_set)
            r2 = run_dual(args, ref_cols, a2, fwd_excess, labels_panel)
            ablation[sub] = r2["delta_B_minus_A"]["rank_ic"]
            print(f"  去掉 {sub:<10} → Δ(RankIC)={ablation[sub]:+.4f}")
        result["ablation"] = ablation

    save_result(result, args, alpha_set)
    return 0


if __name__ == "__main__":
    sys.exit(main())

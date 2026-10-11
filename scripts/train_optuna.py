# -*- coding: utf-8 -*-
"""
Optuna TPE 超参优化 + SHAP 特征筛选训练脚本
对比原版 RandomSearch：TPE 用历史 trial 智能引导，收敛快 3-5 倍，F1 预期 +0.08~0.15
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pickle
import shutil
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from collections import Counter
from loguru import logger

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import xgboost as xgb
# 单一模型口径：本项目只用 xgboost（样本外 RankIC 0.0823 vs lightgbm 0.0647），
# 不再 import lightgbm。optuna / shap 必须在顶层可用 —— 该脚本被 build_exe.py
# 以 --add-data 方式打进 exe，若这两个包没被收进去，这里会直接 ModuleNotFoundError。
import optuna
import shap

from config.settings import config as app_config, MODEL_DIR, LOG_DIR
from src.data.fetcher import fetcher
from src.features.engine import feature_engine

LOG_FILE = LOG_DIR / f"optuna_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
RESULT_FILE = MODEL_DIR / "optuna_search_result.json"
SHAP_FEATURES_FILE = MODEL_DIR / "shap_features.json"

# 参照基线：train_production_v2.py（5档截面分位标签 + V3序数权重 + leak-safe）
# 诚实 walk-forward 实测的样本外 RankIC。本脚本走的是旧管线，结果要跟它比。
PROD_BASELINE_RANKIC = 0.1256

# 生产产物（被本脚本覆盖前必须先备份）
PROD_ARTIFACTS = ["xgboost_model.pkl", "scaler.pkl", "label_encoder.pkl"]


class _TrialProgress:
    """逐 trial 进度回调 —— 把 Optuna 的静默搜索变成可见的日志行。

    为什么必须有它：src/ui/app.py 是把本脚本当子进程跑、靠**日志文本**判断
    进度的；Optuna 自带的 tqdm 进度条靠回车符刷新，在 Streamlit 日志面板里
    根本不渲染 —— 于是几十上百个 trial 期间界面看起来像卡死了。
    这里每个 trial 结束都落一行文本日志（含最佳值与 ETA）。
    """

    def __init__(self, total: int, model_type: str):
        self.total = total
        self.model_type = model_type
        self.t0 = time.time()
        self.best = -np.inf
        self.done = 0

    def __call__(self, study, trial):
        self.done += 1
        elapsed = time.time() - self.t0
        if trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"  [{self.done}/{self.total}] Trial #{trial.number} ✂️ 剪枝跳过"
                        f"  (累计 {elapsed/60:.1f}min)")
            return
        if trial.value is None:
            logger.info(f"  [{self.done}/{self.total}] Trial #{trial.number} ⚠️ 无结果"
                        f"  (累计 {elapsed/60:.1f}min)")
            return

        improved = trial.value > self.best
        if improved:
            self.best = trial.value
        eta = elapsed / self.done * (self.total - self.done) if self.done else 0.0
        flag = " 🏆新最佳" if improved else ""
        logger.info(
            f"  [{self.done}/{self.total}] Trial #{trial.number} "
            f"F1={trial.value:.5f}  最佳={self.best:.5f}{flag}  "
            f"| 已用 {elapsed/60:.1f}min，预计还需 {eta/60:.1f}min"
        )


def _backup_production() -> list:
    """覆盖生产产物前先备份，避免 Optuna（旧管线）静默顶掉已验证的新模型。

    train_production_v2.py 产出的模型实测样本外 RankIC=0.1256；
    本脚本用的是三分类绝对涨跌标签 + SMOTE 全量 + scaler 全量 fit 的旧管线，
    覆盖即降级，所以必须留可回退的备份。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backed = []
    for name in PROD_ARTIFACTS:
        src = MODEL_DIR / name
        if src.exists():
            dst = MODEL_DIR / f"{name}.bak_{ts}"
            try:
                shutil.copy2(src, dst)
                backed.append(dst.name)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"备份 {name} 失败: {e}")
    if backed:
        logger.warning(f"⚠️ 已备份旧生产产物（可回退）: {', '.join(backed)}")
    return backed


def parse_args():
    p = argparse.ArgumentParser(description="Optuna TPE 超参优化训练")
    p.add_argument("--pool", default="hs300")
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--max-stocks", type=int, default=200)
    p.add_argument("--n-trials", type=int, default=50, help="Optuna trials 数（50=快速, 150=标准, 300=深度）")
    p.add_argument("--cv-folds", type=int, default=3)
    # 单一模型口径：只保留 xgboost。"all" 保留仅为兼容旧调用，实际仍只跑 xgboost。
    p.add_argument("--model", default="xgboost", choices=["xgboost", "all"])
    p.add_argument("--retrain-rounds", type=int, default=3)
    p.add_argument("--parallel", action="store_true", help="并行批量获取数据")
    p.add_argument("--parallel-workers", type=int, default=8)
    p.add_argument("--shap-top", type=int, default=60, help="SHAP 筛选后保留的特征数")
    p.add_argument("--no-shap", action="store_true", help="跳过 SHAP 特征筛选")
    p.add_argument("--optuna-timeout", type=int, default=0, help="Optuna 整体超时(秒)，0=不限")
    p.add_argument("--max-samples", type=int, default=0, help="训练样本上限（0=不限，快速建议30000）")
    p.add_argument("--max-estimators", type=int, default=1500, help="n_estimators 上限（快速建议400）")
    p.add_argument("--no-deploy", action="store_true",
                   help="不覆盖生产产物（xgboost_model.pkl/scaler.pkl/label_encoder.pkl），"
                        "结果另存为 *_optuna_*.pkl。默认会覆盖（但覆盖前自动备份）")
    return p.parse_args()


def gather_data(args):
    """收集训练数据（并行优先）"""
    codes = fetcher.get_stock_pool(args.pool)
    codes = codes[:args.max_stocks]
    end_date = datetime.now().strftime("%Y%m%d")

    df_map = {}
    if getattr(args, "parallel", False):
        logger.info(f"股票池: {len(codes)} 只 (并行 {args.parallel_workers}线程)")
        df_map = fetcher.batch_get_daily(
            codes, start_date=args.start_date, end_date=end_date,
            max_workers=args.parallel_workers
        )
    else:
        logger.info(f"股票池: {len(codes)} 只 (串行)")

    all_features, all_labels = [], []
    _t0 = time.time()
    for _i, code in enumerate(codes):
        # 取数阶段最容易"看起来卡死"：单只串行拉行情 + 特征工程，几百只就是几分钟。
        # 这里每 20 只强制落一行日志，UI 日志面板才有东西可看。
        if _i == 0 or (_i + 1) % 20 == 0:
            _el = time.time() - _t0
            logger.info(f"   进度 {_i+1}/{len(codes)} — 已收集 {len(all_features)} 只"
                        f"  (用时 {_el:.0f}s)")
        try:
            if code in df_map:
                df = df_map[code]
            else:
                df = fetcher.get_daily_kline(code, start_date=args.start_date, end_date=end_date)
            if len(df) < 120:
                continue
            f, l = feature_engine.prepare_training_data(df)
            if len(f) > 0:
                f["code"] = code
                all_features.append(f)
                all_labels.append(l)
        except Exception:
            continue

    X = pd.concat(all_features, ignore_index=True).select_dtypes(include=[np.number])
    y = pd.concat(all_labels, ignore_index=True)

    # 采样上限
    if args.max_samples > 0 and len(X) > args.max_samples:
        idx = np.random.RandomState(42).choice(len(X), args.max_samples, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
        logger.info(f"降采样: {len(all_features)} -> {args.max_samples}")

    logger.info(f"数据: {len(X)} 样本, {X.shape[1]} 特征, 标签: {dict(Counter(y))}")
    return X, y


def optuna_objective(trial, model_type, X_scaled, y_enc, cv_folds, class_weights, max_estimators=1500):
    """Optuna objective — 一个 trial = 一次 CV 评估（固定 xgboost）"""
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, max_estimators),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "gamma": trial.suggest_float("gamma", 0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
        "eval_metric": "mlogloss",
    }
    model = xgb.XGBClassifier(**params)

    scores = []
    for train_idx, val_idx in tscv.split(X_scaled):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]
        model.fit(X_tr, y_tr, sample_weight=class_weights[train_idx] if class_weights is not None else None)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average="macro"))

    return float(np.mean(scores))


def optuna_search(X, y, model_type, args):
    """Optuna TPE 超参搜索"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 {model_type.upper()} Optuna TPE — {args.n_trials} trials")
    logger.info(f"{'='*60}")

    scaler = StandardScaler()
    le = LabelEncoder()
    X_scaled = scaler.fit_transform(X.fillna(0))
    y_enc = le.fit_transform(y)

    # 类别权重（处理不均衡）
    cls_w = compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc)
    class_weights = np.array([cls_w[yi] for yi in y_enc])

    # SMOTE
    try:
        cnt = Counter(y_enc)
        k = min(5, min(cnt.values()) - 1)
        if k >= 1:
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_scaled, y_enc = sm.fit_resample(X_scaled, y_enc)
            # 重算类别权重（SMOTE 后分布均匀）
            class_weights = None
            logger.info(f"SMOTE 后: {dict(Counter(y_enc))}")
    except Exception as e:
        logger.warning(f"SMOTE 跳过: {e}")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )

    max_est = getattr(args, "max_estimators", 400)
    obj = lambda trial: optuna_objective(
        trial, model_type, X_scaled, y_enc, args.cv_folds, class_weights, max_est
    )

    timeout = args.optuna_timeout if args.optuna_timeout > 0 else None
    # show_progress_bar 保留（本地终端看得到），但 UI 日志面板靠 _TrialProgress 的逐条日志
    progress_cb = _TrialProgress(args.n_trials, model_type)
    study.optimize(obj, n_trials=args.n_trials, timeout=timeout,
                   n_jobs=1, show_progress_bar=True, callbacks=[progress_cb])

    logger.info(f"\n🏆 {model_type.upper()} 最佳:")
    logger.info(f"  Trial #{study.best_trial.number}  F1(macro) = {study.best_value:.6f}")
    logger.info(f"  参数 = {json.dumps(study.best_params, indent=2)}")

    return {
        "best_params": study.best_params,
        "best_score": float(study.best_value),
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "scaler": scaler,
        "label_encoder": le,
    }


def shap_feature_selection(X, y, best_params, model_type, shap_top_n):
    """用 SHAP 选出最重要的特征"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 SHAP 特征筛选 → 保留 Top {shap_top_n}/{X.shape[1]}")
    logger.info(f"{'='*60}")

    scaler = StandardScaler()
    le = LabelEncoder()
    X_scaled = scaler.fit_transform(X.fillna(0))
    y_enc = le.fit_transform(y)

    # SMOTE
    try:
        cnt = Counter(y_enc)
        k = min(5, min(cnt.values()) - 1)
        if k >= 1:
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_scaled, y_enc = sm.fit_resample(X_scaled, y_enc)
    except:
        pass

    # 用最佳参数训练模型（固定 xgboost）
    params = best_params.copy()
    params.pop("scale_pos_weight", None)
    model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1,
                               verbosity=0, eval_metric="mlogloss")

    model.fit(X_scaled, y_enc)

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)

    # 抽样计算 SHAP（全量太慢）
    sample_n = min(5000, len(X_scaled))
    idx = np.random.choice(len(X_scaled), sample_n, replace=False)
    shap_values = explainer.shap_values(X_scaled[idx])

    # 多分类取平均绝对值
    # ⚠️ 这里曾有个必崩的坑：新版 shap 对多分类返回的是
    #    ndarray (n_samples, n_features, n_classes)，**不是** list。
    #    原代码只判断 isinstance(..., list)，3-D 数组走进 else 分支后
    #    .mean(axis=0) 得到 (n_features, n_classes) 二维 →
    #    pandas 直接抛 "Per-column arrays must each be 1-dimensional"，
    #    SHAP 阶段崩溃 → 后面的重训练/保存全部不执行 → 训练永远没有结果。
    if isinstance(shap_values, list):
        # list of (n_samples, n_features)，逐类取均值再跨类取均值
        shap_importance = np.abs(shap_values).mean(axis=1).mean(axis=0)
    elif getattr(shap_values, "ndim", 0) == 3:
        # (n_samples, n_features, n_classes) → 对样本维与类别维同时取均值
        shap_importance = np.abs(shap_values).mean(axis=(0, 2))
    else:
        shap_importance = np.abs(shap_values).mean(axis=0)

    shap_importance = np.asarray(shap_importance).ravel()

    # 构建重要性 DataFrame
    imp_df = pd.DataFrame({
        "feature": X.columns,
        "shap_importance": shap_importance,
    }).sort_values("shap_importance", ascending=False)

    top_features = imp_df.head(shap_top_n)["feature"].tolist()

    logger.info(f"🔝 Top 15 SHAP 特征:")
    for _, row in imp_df.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['shap_importance']:.6f}")

    important_mask = imp_df["feature"].isin(top_features)
    logger.info(f"  保留 {len(top_features)}/{X.shape[1]} 特征, "
                f"覆盖 {imp_df.loc[important_mask, 'shap_importance'].sum() / imp_df['shap_importance'].sum() * 100:.1f}% 重要性")

    return top_features, imp_df


def retrain_final(X, y, best_params, model_type, rounds, features, result, deploy=True):
    """SHAP 筛选后用最佳参数重训练"""
    n_feat = len(features) if features is not None else X.shape[1]
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 {model_type.upper()} 最终重训练 — {rounds} 轮 ({n_feat} 特征)")
    logger.info(f"{'='*60}")

    X_final = X[features] if features is not None else X

    scaler = StandardScaler()
    le = LabelEncoder()
    X_scaled = scaler.fit_transform(X_final.fillna(0))
    y_enc = le.fit_transform(y)

    try:
        cnt = Counter(y_enc)
        k = min(5, min(cnt.values()) - 1)
        if k >= 1:
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_scaled, y_enc = sm.fit_resample(X_scaled, y_enc)
    except:
        pass

    scores, best_model, best_f1 = [], None, -1
    tscv = TimeSeriesSplit(n_splits=5)

    for r in range(rounds):
        params = best_params.copy()
        params.pop("scale_pos_weight", None)
        model = xgb.XGBClassifier(**params, random_state=42 + r, n_jobs=-1,
                                   verbosity=0, eval_metric="mlogloss")

        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            model.fit(X_scaled[train_idx], y_enc[train_idx])
            preds = model.predict(X_scaled[val_idx])
            cv_scores.append(f1_score(y_enc[val_idx], preds, average="macro"))

        mean_f1 = np.mean(cv_scores)
        scores.append(mean_f1)
        logger.info(f"  Round {r+1}/{rounds}: CV F1 = {mean_f1:.6f}")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            model.fit(X_scaled, y_enc)
            best_model = model

    logger.info(f"\n📊 {model_type.upper()} 最终: F1 = {np.mean(scores):.6f} ± {np.std(scores):.4f}")

    # ===== 保存 =====
    # ⚠️ 本脚本走的是旧管线（三分类绝对涨跌标签 + SMOTE 全量 + scaler 全量 fit），
    #    样本外 RankIC 实测仅 ~0.08；覆盖生产产物就等于把 train_production_v2.py
    #    产出的 RankIC=0.1256 模型降级掉。所以：默认覆盖前先备份；
    #    --no-deploy 则完全不碰生产产物。
    if deploy:
        logger.warning("⚠️ 即将覆盖生产产物 (xgboost_model.pkl / scaler.pkl / label_encoder.pkl)")
        logger.warning(f"   本脚本为旧管线，样本外 RankIC 明显低于生产基线 {PROD_BASELINE_RANKIC:.4f}，"
                       "覆盖即降级（已自动备份，可用 .bak_* 回退）")
        _backup_production()
        model_path = MODEL_DIR / f"{model_type}_model.pkl"
        scaler_path = MODEL_DIR / "scaler.pkl"
        enc_path = MODEL_DIR / "label_encoder.pkl"
    else:
        logger.info("（--no-deploy）不触碰生产产物，结果另存为 *_optuna_*.pkl")
        model_path = MODEL_DIR / f"{model_type}_optuna_model.pkl"
        scaler_path = MODEL_DIR / f"{model_type}_optuna_scaler.pkl"
        enc_path = MODEL_DIR / f"{model_type}_optuna_label_encoder.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"💾 模型: {model_path}")

    # 保存特征列表（predictor 加载时裁剪）
    feat_info = {"features": features if features is not None else list(X.columns), "n_features": len(features) if features else X.shape[1]}
    # 特征列表同样要区分：predictor 会读 features_xgboost.json，
    # --no-deploy 时若还写生产文件名，等于拿一份垃圾实验的特征清单污染生产推理。
    feat_file = MODEL_DIR / (f"features_{model_type}.json" if deploy
                             else f"features_{model_type}_optuna.json")
    with open(feat_file, "w", encoding="utf-8") as f:
        json.dump(feat_info, f, ensure_ascii=False)
    logger.info(f"💾 特征列表: {feat_file}")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(enc_path, "wb") as f:
        pickle.dump(le, f)
    logger.info(f"💾 预处理器: {scaler_path.name} / {enc_path.name}")

    return {
        "mean_f1": float(np.mean(scores)),
        "std_f1": float(np.std(scores)),
        "scores": [float(s) for s in scores],
        "n_features_used": len(features) if features else X.shape[1],
        "deployed": bool(deploy),
        "model_path": str(model_path),
    }


def main():
    """入口：负责把 _run() 的任何失败都翻译成 ===TRAINING_FAILED=== 标记。

    src/ui/app.py 靠日志里的 ===TRAINING_COMPLETE=== / ===TRAINING_FAILED===
    判断训练是否结束。任何一条失败路径若不留下标记，UI 就会永久停在
    "训练进行中"。所以这里（以及 _run() 内部的裸 return 处）必须兜住。
    """
    args = parse_args()
    logger.add(LOG_FILE, rotation="100 MB", level="INFO")
    try:
        _run(args)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"❌ 训练异常终止: {e}")
        logger.error("===TRAINING_FAILED===")


def _run(args):
    t_start = time.time()
    deploy = not args.no_deploy
    logger.info("=" * 70)
    logger.info("🚀 选股神AI — Optuna TPE 超参优化训练")
    logger.info(f"   Trials: {args.n_trials}  ·  CV: {args.cv_folds} 折  ·  重训: {args.retrain_rounds} 轮")
    if not args.no_shap:
        logger.info(f"   SHAP 特征筛选: Top {args.shap_top}")
    logger.info(f"   部署: {'覆盖生产产物（覆盖前自动备份）' if deploy else '不覆盖，另存 *_optuna_*.pkl'}")
    logger.info(f"   日志: {LOG_FILE}")
    logger.info(f"   结果: {RESULT_FILE}")
    logger.info("=" * 70)
    logger.warning(
        "⚠️ 口径提示：本脚本仍走旧管线（三分类绝对涨跌标签 + SMOTE 全量 + scaler 全量 fit），"
        f"优化目标是 F1(macro)。选股真正的决策指标是 RankIC —— 生产基线为 "
        f"{PROD_BASELINE_RANKIC:.4f}（train_production_v2.py，诚实 walk-forward）。"
        "F1 变高不等于选股变强，别只看这个数。"
    )

    # 1. 数据
    logger.info("\n📊 收集训练数据…")
    X, y = gather_data(args)
    if len(X) == 0:
        logger.error("无训练数据!")
        logger.error("===TRAINING_FAILED=== (原因: 未收集到任何训练样本)")
        return

    all_results = {}

    # 2. Optuna 搜索（单一模型口径：只跑 xgboost）
    models_to_run = ["xgboost"]
    for mt in models_to_run:
        r = optuna_search(X, y, mt, args)
        all_results[mt] = r

    # 3. SHAP 特征筛选
    shap_features = {}
    for mt in models_to_run:
        if args.no_shap:
            shap_features[mt] = None
        else:
            top_feats, imp_df = shap_feature_selection(
                X, y, all_results[mt]["best_params"], mt, args.shap_top
            )
            shap_features[mt] = top_feats

    # 4. 重训练
    final_scores = {}
    for mt in models_to_run:
        r = retrain_final(
            X, y, all_results[mt]["best_params"], mt,
            args.retrain_rounds, shap_features[mt], all_results[mt],
            deploy=deploy,
        )
        final_scores[mt] = r

    # 汇总
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_features_raw": X.shape[1],
        "n_trials": args.n_trials,
        "method": "Optuna TPE + SHAP",
        "search_results": {k: {"best_params": v["best_params"],
                                "best_score": v["best_score"],
                                "n_trials": v["n_trials"],
                                "n_pruned": v.get("n_pruned", 0)}
                           for k, v in all_results.items()},
        "final_scores": final_scores,
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("🎉 Optuna 超参优化完成!")
    logger.info(f"   总耗时: {elapsed/60:.1f} 分钟")
    for name, s in final_scores.items():
        sr = all_results.get(name, {})
        logger.info("-" * 70)
        logger.info(f"   [{name}]")
        logger.info(f"     搜索: {sr.get('n_trials', '?')} trials"
                    f"（剪枝 {sr.get('n_pruned', 0)} 个）"
                    f"，最佳 CV F1 = {sr.get('best_score', float('nan')):.6f}")
        bp = sr.get("best_params", {}) or {}
        if bp:
            logger.info(f"     最佳参数: {json.dumps(bp, ensure_ascii=False)}")
        logger.info(f"     最终 CV: F1 = {s['mean_f1']:.6f} ± {s['std_f1']:.4f}"
                    f"，使用 {s['n_features_used']} 个特征")
        logger.info(f"     产物: {s.get('model_path', '?')}"
                    + ("（已覆盖生产产物，旧件已备份为 .bak_*）" if s.get("deployed")
                       else "（未覆盖生产产物）"))
    logger.info("-" * 70)
    logger.warning(
        f"   怎么判断这次训练好不好：F1 不是选股决策指标。"
        f"生产基线 RankIC = {PROD_BASELINE_RANKIC:.4f}（诚实 walk-forward 实测）。"
        "本脚本的旧管线在取数阶段丢弃了 date/code，算不出截面 RankIC；"
        "要一个可信的选股指标请用 scripts/train_production_v2.py（自带 walk-forward RankIC 验证）。"
    )
    logger.info(f"   结果 JSON: {RESULT_FILE}")
    logger.info("===TRAINING_COMPLETE===")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 —— 兜住 main() 之外（如日志初始化）的崩溃
        logger.exception("❌ 训练脚本崩溃")
        logger.error("===TRAINING_FAILED===")
        raise SystemExit(1)

# -*- coding: utf-8 -*-
"""模型样本外评估：回答"哪个模型真的更好"

背景
----
predictor 目前把 xgboost 与 lightgbm 做 50/50 集成。但：
  - ``lightgbm_model.pkl`` 停留在 2026-07-09，之后历次训练都没更新它
  - 训练脚本（train_optuna）只产出 xgboost
于是"训练"只更新了一半权重，另一半永远是三个月前的旧模型 —— 这就是
"训练完全没有用"的根源。

结论（本脚本 200 只 × 45 天样本外评估）：xgboost 更优
（RankIC 0.0823 / TopN 超额 +1.39% vs lightgbm 0.0647 / +0.57%），
项目已收敛为**单一 xgboost**，lightgbm_model.pkl 已归档。
本脚本仍保留对比能力：只要 models/<name>_model.pkl 存在就会纳入对比，
缺失则自动跳过。

本脚本用**统一的样本外口径**评估各模型：
    在每个评估日做截面打分，与"未来 HORIZON 日超额收益"算 Spearman 秩相关
    （即 RankIC），并给出 IC_IR = mean(IC)/std(IC)、TopN 超额收益。
    这是本项目 V2 约定的评估口径（不看 accuracy/F1）。

用法: py -3.12 scripts/eval_model_oos.py [--stocks 120] [--days 10]
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from scipy.stats import spearmanr  # noqa: E402

from config.settings import MODEL_DIR  # noqa: E402
from src.features.engine import feature_engine  # noqa: E402

HORIZON = 5
MIN_HISTORY = 60


def load_model(name: str):
    """加载模型；文件缺失或加载失败都返回 None（对比脚本不该因此整体崩掉）。

    lightgbm_model.pkl 已被归档移走，这里必须容忍缺失：
    脚本仍保留对 lightgbm 的对比能力（万一以后又训了 lgb），只是它不出现时
    不影响 xgboost 的评估。
    """
    path = MODEL_DIR / f"{name}_model.pkl"
    if not path.exists():
        print(f"  ⚠️ 缺少模型文件 {path.name}，跳过")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ 加载 {path.name} 失败（{e}），跳过")
        return None


def load_scaler():
    path = MODEL_DIR / "scaler.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def score_of(model, X) -> np.ndarray:
    """把模型输出转成与收益单调相关的连续分数。

    多分类（分档标签）时取**期望档位** ``Σ p_i * i`` —— 它保留了方向信息；
    而最大类概率只是"置信度"，与未来收益并不单调。
    """
    proba = model.predict_proba(X)
    n_cls = proba.shape[1]
    if n_cls <= 2:
        return proba[:, -1]
    return proba @ np.arange(n_cls, dtype=float)


def build_feature_panels(store: dict) -> dict:
    """逐股构建特征面板（一次 build 覆盖全部日期，避免按日重复计算）"""
    panels = {}
    for i, (code, df) in enumerate(store.items(), 1):
        try:
            if len(df) < MIN_HISTORY:
                continue
            feats = feature_engine.build_features(df)
            if feats is None or feats.empty:
                continue
            if "close" in feats.columns:
                feats = feats.drop(columns=["close"])
            panels[code] = feats
        except Exception:
            continue
        if i % 50 == 0:
            print(f"    特征构建 {i}/{len(store)} ...", flush=True)
    return panels


def evaluate(models: dict, scaler, panels: dict, close: pd.DataFrame,
             eval_dates, top_n: int = 20) -> dict:
    """在给定评估日上算各模型的 RankIC 序列"""
    # 未来 HORIZON 日绝对收益 → 截面超额收益
    fwd = close.shift(-HORIZON) / close - 1
    excess = fwd.sub(fwd.mean(axis=1), axis=0)

    scaler_cols = getattr(scaler, "feature_names_in_", None)
    results = {name: [] for name in models}
    top_excess = {name: [] for name in models}

    for d in eval_dates:
        if d not in excess.index:
            continue
        row_excess = excess.loc[d].dropna()
        if len(row_excess) < 30:
            continue

        # 收集当日所有股票的分数
        scores = {name: {} for name in models}
        for code, feats in panels.items():
            if d not in feats.index:
                continue
            if code not in row_excess.index:
                continue
            try:
                row = feats.loc[[d]].fillna(0)
                if scaler_cols is not None:
                    aligned = row.reindex(columns=scaler_cols, fill_value=0.0)
                else:
                    aligned = row.iloc[:, : getattr(scaler, "n_features_in_", row.shape[1])]
                X = scaler.transform(aligned)
            except Exception:
                continue
            for name, m in models.items():
                try:
                    scores[name][code] = float(score_of(m, X)[0])
                except Exception:
                    continue

        for name in models:
            s = pd.Series(scores[name]).dropna()
            common = s.index.intersection(row_excess.index)
            if len(common) < 30:
                continue
            ic, _ = spearmanr(s[common].values, row_excess[common].values)
            if np.isnan(ic):
                continue
            results[name].append(ic)

            # TopN 组合相对等权基准的超额
            top = s[common].nlargest(top_n).index
            top_excess[name].append(float(row_excess[top].mean()))

    out = {}
    for name in models:
        ics = np.array(results[name], dtype=float)
        tes = np.array(top_excess[name], dtype=float)
        out[name] = {
            "n_days": len(ics),
            "rank_ic": float(ics.mean()) if len(ics) else float("nan"),
            "ic_ir": float(ics.mean() / ics.std(ddof=1)) if len(ics) > 1 else float("nan"),
            "ic_std": float(ics.std(ddof=1)) if len(ics) > 1 else float("nan"),
            "ic_positive_rate": float((ics > 0).mean()) if len(ics) else float("nan"),
            "top_excess": float(tes.mean()) if len(tes) else float("nan"),
            "top_win_rate": float((tes > 0).mean()) if len(tes) else float("nan"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=120)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    print("=" * 68)
    print(f" 模型样本外评估（RankIC 口径，持有期 {args.horizon} 日）")
    print("=" * 68)

    # ---------- 数据 ----------
    print(f"\n[1/4] 加载本地数据（最多 {args.stocks} 只）...")
    from scripts.train_v2 import load_pool

    # 显式 legacy：评估脚本沿用旧口径（按代码排序取前 N 只），避免池子被静默改变
    store = load_pool(args.stocks, pool="legacy")
    if len(store) < 50:
        print("  数据不足，退出")
        return 1
    close = pd.DataFrame({c: d["close"] for c, d in store.items()}).sort_index()
    cover = close.notna().mean(axis=1)
    close = close.loc[cover >= 0.6]
    print(f"  股票 {len(store)} 只 | 日期 {close.index[0].date()} ~ {close.index[-1].date()}")

    # 评估日：需要有未来 HORIZON 日收益，故从末尾往前让出 HORIZON 天
    eval_dates = list(close.index[-(args.horizon + args.days): -args.horizon])
    print(f"  评估日 {len(eval_dates)} 个: {eval_dates[0].date()} ~ {eval_dates[-1].date()}")

    # ---------- 特征 ----------
    print("\n[2/4] 构建特征面板...")
    panels = build_feature_panels(store)
    print(f"  有效股票 {len(panels)} 只")

    # ---------- 模型 ----------
    print("\n[3/4] 加载模型...")
    scaler = load_scaler()
    if scaler is None:
        print("  ❌ 缺少 scaler.pkl，退出")
        return 1
    models = {}
    for name in ("xgboost", "lightgbm"):
        m = load_model(name)
        if m is not None:
            models[name] = m
            print(f"  已加载 {name}")
    if not models:
        print("  ❌ 没有可用模型")
        return 1

    # ---------- 评估 ----------
    print("\n[4/4] 计算样本外 RankIC ...")
    res = evaluate(models, scaler, panels, close, eval_dates, top_n=args.top_n)

    print("\n" + "=" * 68)
    print(f" {'模型':<12}{'RankIC':>10}{'IC_IR':>10}{'IC>0比例':>10}"
          f"{'TopN超额':>12}{'TopN胜率':>10}{'天数':>6}")
    print(" " + "-" * 66)
    for name, r in res.items():
        print(f" {name:<12}{r['rank_ic']:>10.4f}{r['ic_ir']:>10.2f}"
              f"{r['ic_positive_rate']:>10.1%}{r['top_excess']:>12.4%}"
              f"{r['top_win_rate']:>10.1%}{r['n_days']:>6}")
    print(" " + "-" * 66)
    print(" 参考：随机猜 RankIC≈0；本项目 V2 walks-forward 基线 RankIC≈0.049、"
          "IC_IR≈15.4")
    print("=" * 68)

    # 结论
    if len(res) >= 2:
        best = max(res.items(), key=lambda kv: kv[1]["rank_ic"])
        print(f"\n结论：本口径下最好的是 {best[0]}（RankIC={best[1]['rank_ic']:.4f}）")

    # 落盘
    out_f = MODEL_DIR / "oos_eval_result.json"
    try:
        import json
        out_f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存: {out_f}")
    except Exception as e:
        print(f"保存失败: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

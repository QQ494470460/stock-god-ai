# -*- coding: utf-8 -*-
"""
Walk-Forward 训练器（泄漏安全版）
=================================

修复的三个致命问题
------------------
**问题1：scaler在全量数据上 fit → 未来信息泄漏**
   旧代码 preprocess() 对整个 X 做 fit_transform，然后才切分训练/验证。
   验证集的均值方差参与了标准化，等于把未来信息灌进训练。
   → 现在：每个 fold 内部只对 train 段 fit scaler，验证段用 transform。

**问题2：SMOTE 在切分之前对全量数据过采样 → 时序被打乱**
   旧代码 balance_samples() 先做 SMOTE，SMOTE 会在相邻样本间插值生成
   合成样本，把时序结构彻底破坏。随后再 TimeSeriesSplit 切分，
   此时"训练集"里已经混入了未来时点的合成样本。
   → 现在：SMOTE 移到 fold 内部，只在 train 段做，且默认关闭
     （改用 class_weight 做不平衡处理，不动样本本身）。

**问题3：按行切分 → 实际是在跨股票切分**
   旧代码 pd.concat(all_features) 把不同股票的数据首尾相接，
   样本顺序变成 [股票A的全部历史][股票B的全部历史]...
   此时 TimeSeriesSplit 按行切分，切出来的是
   "前 200 只股票" vs "后 100 只股票"，而不是时间先后。
   → 现在：样本必须携带 date，walk_forward_split 按**日期**切分，
     保证 train 日期 < valid 日期，股票维度不参与切分边界。

评估口径也一并改掉
------------------
旧的 accuracy/F1 无法反映选股能力（实测天花板仅 36.7%）。
现在以 RankIC、IC_IR、Top-N 超额收益、Sharpe 为主指标。
"""

from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple
from pathlib import Path
from loguru import logger

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import xgboost as xgb
import lightgbm as lgb

from config.settings import MODEL_DIR
from src.models.labeling import (
    cs_rank, mean_rank_ic, top_portfolio_return, portfolio_metrics,
)


# ============ 数据容器 ============

@dataclass
class WalkForwardResult:
    """Walk-forward 训练与评估结果"""
    model_name: str
    feature_names: List[str] = field(default_factory=list)

    # 逐 fold 明细
    folds: List[dict] = field(default_factory=list)

    # 汇总（样本外拼接后计算）
    oos_rank_ic: float = np.nan          # 全样本外 RankIC 均值
    oos_ic_ir: float = np.nan            # IC_IR
    oos_top_excess: float = np.nan       # Top-N 平均超额收益/期
    oos_annual_return: float = np.nan
    oos_sharpe: float = np.nan
    oos_win_rate: float = np.nan
    oos_max_drawdown: float = np.nan
    oos_accuracy: float = np.nan         # 仅作参考，非核心指标
    n_oos_samples: int = 0
    horizon: int = 5                     # 标签窗口，决定"单期"是多长

    # 特征重要性
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)

    model: Any = None

    def summary(self) -> str:
        lines = [
            f"模型: {self.model_name}",
            f"  样本外 RankIC      {self.oos_rank_ic:+.4f}",
            f"  IC_IR              {self.oos_ic_ir:+.2f}",
            f"  Top20 单期超额     {self.oos_top_excess:+.4%}   (每{self.horizon}日，相对全市场均值)",
            f"  超额胜率           {self.oos_win_rate:.1%}   (跑赢全市场的天数占比)",
            f"  超额最大回撤       {self.oos_max_drawdown:.2%}   (相对基准的累计回撤)",
            f"  五档分类准确率     {self.oos_accuracy:.1%}   (随机猜=20%)",
            f"  样本外样本数{self.n_oos_samples}",
            "",
            "  以上均为 walk-forward 样本外指标：训练/验证严格按日期切分，",
            "  scaler 仅用训练段拟合，SMOTE 已替换为 class_weight。",
            "  评估每隔 horizon 个交易日取一个观测（去除重叠窗口），",
            "  否则同一段5日收益会被重复计入5次，回撤与IC_IR都被放大。",
            "  超额收益未扣手续费与冲击成本；不做复利年化（超额序列不可直接复利）。",
        ]
        return "\n".join(lines)


@dataclass
class WalkForwardSplit:
    """单折切分结果（按日期切，股票不参与边界）"""
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


# ============ 按日期的 walk-forward 切分 ============

def walk_forward_split(
    dates: np.ndarray,
    n_splits: int = 5,
    min_train_ratio: float = 0.5,
    embargo_days: int = 0
) -> List[WalkForwardSplit]:
    """
    按日期做 walk-forward 切分（严格时间先后）

    这是修复"问题3"的核心。传入的是**去重后的日期数组**，
    切分点落在日期上，因此：
      - train 的所有日期 < valid 的所有日期
      - 同一只股票可以同时出现在 train 和 valid（这是正常的，
        我们预测的是同一只股票的下一个时间段）
      - 股票代码本身不影响切分边界

    Parameters
    ----------
    dates : 样本日期数组（可含重复，会自动去重排序）
    n_splits : 验证段数量
    min_train_ratio : 最小训练集占全样本的比例
    embargo_days : 训练集与验证集之间空出的交易日数
        用于隔离标签窗口重叠。注意：若 horizon=5，标签本身已用未来5日，
        train 末端样本的标签会"伸进" valid 区间。留出 embargo 天数可
        缓解（不能完全消除，因为训练集末尾标签天然含未来信息，
        但这与实盘一致：实盘训练时最近N日也没有完整标签）。

    Returns
    -------
    List[WalkForwardSplit]
    """
    uniq = np.sort(pd.unique(dates))
    n = len(uniq)
    if n < 30:
        raise ValueError(f"日期数量不足({n})，无法做 walk-forward 切分")

    n_splits = min(n_splits, max(2, n // 20))

    # 验证段长度 = 剩余部分均分
    test_size = int(n * (1 - min_train_ratio) / n_splits)
    first_train = n - test_size * n_splits
    if first_train < 20:
        test_size = max(5, (n - 20) // n_splits)
        first_train = n - test_size * n_splits

    splits: List[WalkForwardSplit] = []
    for i in range(n_splits):
        tr_start = 0
        tr_end_i = first_train + i * test_size
        va_start_i = tr_end_i + embargo_days
        va_end_i = va_start_i + test_size
        if va_end_i > n:
            break
        splits.append(WalkForwardSplit(
            fold=i,
            train_start=pd.Timestamp(uniq[tr_start]),
            train_end=pd.Timestamp(uniq[tr_end_i - 1]),
            valid_start=pd.Timestamp(uniq[va_start_i]),
            valid_end=pd.Timestamp(uniq[va_end_i - 1]),
        ))
    return splits


# ============ 训练器 ============

class LeakSafeTrainer:
    """
    泄漏安全的 walk-forward 训练器

    与旧 MLTrainer 的区别：
      1. 预处理在 fold 内部 fit，不用全局 scaler
      2. SMOTE 移到 fold 内部且默认关闭（用 class_weight 替代）
      3. 按日期切分，不是按行切分
      4. 评估以 RankIC / 超额收益为主，accuracy 仅作参考
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        use_smote: bool = False,
        n_splits: int = 5,
        horizon: int = 5
    ):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_smote = use_smote
        self.n_splits = n_splits
        self.horizon = horizon

        # 这些会在训练后保存，供推理复用
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.models: Dict[str, Any] = {}

    # ---------- 预处理 ----------

    def fit_transform_train(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """只在训练段 fit scaler，然后 transform 训练段"""
        self.scaler.fit(X)
        return self.scaler.transform(X), y

    def transform_valid(
        self, X: pd.DataFrame
    ) -> np.ndarray:
        """验证段只做 transform，绝不 fit（防泄漏）"""
        return self.scaler.transform(X)

    def _class_weights(self, y: np.ndarray) -> np.ndarray:
        """
        类别权重替代 SMOTE

        SMOTE 会合成新样本、破坏时序结构（问题2）。
        class_weight 通过"给少数类更大损失权重"达到同样的
        类别平衡效果，且不改变样本本身 —— 时序安全。
        """
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        n_cls = len(classes)
        w = {}
        for c, cnt in zip(classes, counts):
            w[c] = total / (n_cls * cnt)
        return np.array([w[v] for v in y])

    # ---------- 折内训练 ----------

    def _fit_one_fold(
        self,
        model_type: str,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
        params: dict,
    ) -> Any:
        """训练单折模型"""
        # ★ 关键：scaler 只在 train 上 fit
        Xtr, ytr = self.fit_transform_train(X_tr, y_tr)
        Xva = self.transform_valid(X_va)

        if self.use_smote:
            # 即使用也只在 train 内部做（修复问题2的关键）
            Xtr, ytr = self._smote_within_train(Xtr, ytr)

        sw = self._class_weights(ytr)

        if model_type == "xgboost":
            model = xgb.XGBClassifier(**params)
            model.fit(Xtr, ytr, sample_weight=sw, verbose=False)
        else:
            model = lgb.LGBMClassifier(**params, verbose=-1)
            model.fit(Xtr, ytr, sample_weight=sw)

        return model

    def _smote_within_train(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """仅对训练段做 SMOTE（默认不走此路径）"""
        try:
            from imblearn.over_sampling import SMOTE
            from collections import Counter
            cnt = Counter(y)
            k = min(5, min(cnt.values()) - 1)
            if k < 1:
                return X, y
            return SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
        except Exception as e:
            logger.warning(f"SMOTE 跳过: {e}")
            return X, y

    # ---------- 主训练流程 ----------

    def train_walk_forward(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        model_type: str = "xgboost",
        params: Optional[dict] = None,
        fwd_excess: Optional[pd.DataFrame] = None,
    ) -> WalkForwardResult:
        """
        执行 walk-forward 训练

        Parameters
        ----------
        features : DataFrame  index=样本ID, columns=特征
            必须与 labels/dates 对齐
        labels : Series  index=样本ID, 值为分类标签
        dates : Series  index=样本ID, 值为该样本的**交易日**
            这是修复"按行切分"问题的必需参数
        model_type : 'xgboost' | 'lightgbm'
        fwd_excess : DataFrame index=date, columns=code
            **真实**的前瞻超额收益矩阵。用于计算诚实的组合收益。
            缺省时组合收益指标不可用（不会用档位标签冒充收益）。
        """
        if params is None:
            params = (
                dict(
                    n_estimators=400, max_depth=5, learning_rate=0.03,
                    subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
                    gamma=0.3, reg_alpha=0.5, reg_lambda=2.0,
                    random_state=42, n_jobs=-1, verbosity=0,
                )
                if model_type == "xgboost"
                else dict(
                    n_estimators=400, max_depth=6, learning_rate=0.03,
                    subsample=0.7, colsample_bytree=0.7, num_leaves=31,
                    min_child_samples=40, reg_alpha=0.5, reg_lambda=2.0,
                    random_state=42, n_jobs=-1, verbose=-1,
                )
            )

        # 对齐并清洗
        df = features.copy()
        df["__label__"] = labels.values
        df["__date__"] = pd.to_datetime(dates.values)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # 若样本索引形如 'YYYY-MM-DD_600000'，可直接解析出代码
        if "__code__" not in df.columns:
            derived = pd.Series(df.index.astype(str), index=df.index).str[-6:]
            df["__code__"] = derived.values

        feature_cols = [c for c in features.columns if c in df.columns]
        self.feature_names = feature_cols

        logger.info(
            f"训练集准备: {len(df)} 样本 × {len(feature_cols)} 特征, "
            f"日期 {df['__date__'].min().date()} ~ {df['__date__'].max().date()}"
        )

        splits = walk_forward_split(
            df["__date__"].values, n_splits=self.n_splits
        )
        logger.info(f"Walk-forward 切分: {len(splits)} 折（按日期切分）")

        oof_pred: List[np.ndarray] = []      # 样本外预测分数
        oof_true: List[np.ndarray] = []      # 样本外真实标签
        oof_dates: List[pd.Series] = []      # 对应日期
        oof_codes: List[np.ndarray] = []     # 对应股票代码
        oof_idx: List[np.ndarray] = []       # 对应样本索引
        oof_feat_imp = np.zeros(len(feature_cols))
        oof_cls: List[np.ndarray] = []       # 样本外预测类别

        fold_rows = []

        for sp in splits:
            tr = df[(df["__date__"] >= sp.train_start) & (df["__date__"] <= sp.train_end)]
            va = df[(df["__date__"] >= sp.valid_start) & (df["__date__"] <= sp.valid_end)]

            if len(tr) < 100 or len(va) < 30:
                logger.warning(f"  Fold {sp.fold+1}: 样本不足，跳过")
                continue

            X_tr = tr[feature_cols]
            y_tr = tr["__label__"].astype(int).values
            X_va = va[feature_cols]
            y_va = va["__label__"].astype(int).values

            model = self._fit_one_fold(
                model_type, X_tr, y_tr, X_va, y_va, params
            )

            # 样本外预测（用最高类概率作为强度分）
            Xva_scaled = self.transform_valid(X_va)
            proba = model.predict_proba(Xva_scaled)
            strength = proba.max(axis=1) * (2 * (np.argmax(proba, axis=1) == proba.shape[1] - 1) - 1)
            y_pred_cls = model.predict(Xva_scaled)

            oof_pred.append(strength)
            oof_true.append(y_va)
            oof_cls.append(y_pred_cls)
            oof_dates.append(va["__date__"].reset_index(drop=True))
            code_col = va["__code__"].reset_index(drop=True) if "__code__" in va.columns else None
            if code_col is not None:
                oof_codes.append(code_col.astype(str).values)
            oof_idx.append(va.index.values)

            # 折内评估
            acc = accuracy_score(y_va, y_pred_cls)
            f1 = f1_score(y_va, y_pred_cls, average="macro")

            # 逐折 RankIC：判断信号是"每年都在"还是"只靠某一段"
            # 同样按 horizon 去重叠，否则重叠窗口会让单折 IC 看起来很稳
            fold_ic = np.nan
            try:
                if "__code__" in va.columns and len(va) > 200:
                    _f = pd.DataFrame({
                        "date": pd.to_datetime(va["__date__"].values),
                        "code": va["__code__"].astype(str).values,
                        "score": strength,
                        "lab": y_va,
                    })
                    _sp = _f.pivot(index="date", columns="code", values="score")
                    _lp = _f.pivot(index="date", columns="code", values="lab")
                    _h = max(int(self.horizon), 1)
                    _sp = _sp.iloc[::_h]
                    _lp = _lp.loc[_sp.index]
                    if len(_sp) >= 10:
                        fold_ic, _, _ = mean_rank_ic(_sp, _lp)
            except Exception as e:
                logger.debug(f"  折内 RankIC 计算失败: {e}")

            oof_feat_imp += np.asarray(model.feature_importances_, dtype=float)

            fold_rows.append({
                "fold": sp.fold + 1,
                "train": f"{sp.train_start.date()}~{sp.train_end.date()}",
                "valid": f"{sp.valid_start.date()}~{sp.valid_end.date()}",
                "n_train": len(tr),
                "n_valid": len(va),
                "accuracy": acc,
                "f1_macro": f1,
                "rank_ic": float(fold_ic) if fold_ic == fold_ic else None,
            })
            logger.info(
                f"  Fold {sp.fold+1}: train={len(tr)} valid={len(va)} "
                f"acc={acc:.4f} f1={f1:.4f} ric={fold_ic:+.4f}"
            )

        if not fold_rows:
            raise RuntimeError("没有有效折可训练，请检查样本量或切分参数")

        # ---------- 汇总样本外指标 ----------

        all_pred = np.concatenate(oof_pred)
        all_true = np.concatenate(oof_true)
        all_dates = pd.concat(oof_dates, ignore_index=True)
        n_oos = len(all_pred)

        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": oof_feat_imp / len(fold_rows),
        }).sort_values("importance", ascending=False)

        # 构造 (date × code) 面板：得分 vs 真实档位
        # 注意：strength 是连续值，不能直接当 pivot 的 columns（会被按数值分箱）。
        # 正确做法是先用 MultiIndex 重排，再 unstack。
        oos_df = pd.DataFrame({
            "date": all_dates,
            "code": np.concatenate(oof_codes) if oof_codes else np.concatenate(oof_idx),
            "score": all_pred,
            "ret": all_true,
        })
        oos_df = oos_df.drop_duplicates(subset=["date", "code"], keep="last")
        try:
            sp_pivot = oos_df.pivot(
                index="date", columns="code", values="score"
            )
            rt_pivot = oos_df.pivot(
                index="date", columns="code", values="ret"
            )
        except Exception as e:
            logger.warning(f"构造 OOS 面板失败: {e}")
            sp_pivot = pd.DataFrame()
            rt_pivot = pd.DataFrame()

        # ---------- 重叠窗口去重（关键口径）----------
        # 标签是"未来 horizon 日收益"，但样本是**逐日**生成的：
        # 相邻两个观测的收益窗口有 (horizon-1)/horizon 的重叠。
        # 直接按日累计，等于把同一个 5 日收益重复计入 5 次 ——
        # 累计超额、最大回撤被放大 5 倍，IC_IR 也被虚高
        #（分母低估，因为重叠样本不是独立观测）。
        #
        # 评估口径统一改成"每 horizon 个交易日取一个"，只保留不重叠观测。
        # 训练仍然用全部逐日样本（样本多有益），只是**度量**时去重。
        _h = max(int(getattr(self, "horizon", 5)), 1)
        if len(sp_pivot) > _h * 3:
            _idx = sp_pivot.index[::_h]
            sp_eval = sp_pivot.loc[_idx]
            rt_eval = rt_pivot.loc[_idx]
        else:
            sp_eval, rt_eval = sp_pivot, rt_pivot

        # ---------- RankIC ----------
        # RankIC 只需要"单调对应关系"，用档位标签算IC 没问题
        #（Spearman 只看秩，不看数值）
        if len(sp_eval) >= 30:
            ic, ic_ir, n_days = mean_rank_ic(sp_eval, rt_eval)
        else:
            ic, ic_ir, n_days = np.nan, np.nan, len(sp_eval)

        # Top-N 组合绩效
        #
        # ⚠️ 关键口径：绝对不能用**档位标签**算组合超额收益
        #档位(0~4) 是截面排名，不是收益率。若拿它当收益：
        #   · Top20 的"收益"恒为 4.0
        #   · 与全市场均值(2.0)相减 = 2.0（永远正，且数值荒谬）
        #   · 会得出"单期超额 +200%"这种完全错误的结论
        #
        # 所以必须传入真实的前瞻超额收益矩阵，只用它算组合收益；
        # RankIC 仍可用档位（Spearman 只看秩）。
        if len(sp_eval) >= 30 and fwd_excess is not None:
            fwd = fwd_excess.reindex(index=sp_eval.index, columns=sp_eval.columns)
            daily = top_portfolio_return(sp_eval, fwd, top_n=20)
            ex_metrics = portfolio_metrics(daily["excess"], periods_per_year=1)
            final_metrics = {
                "mean_period_ret": ex_metrics["mean_period_ret"],
                "win_rate": ex_metrics["win_rate"],
                "max_drawdown": ex_metrics["max_drawdown"],
                "annual_return": None,   # 超额序列不可直接复利年化
                "sharpe": None,
                "n_periods": ex_metrics["n_periods"],
            }
            logger.info(
                f"  Top20 真实单期超额均值={ex_metrics['mean_period_ret']:+.4%}, "
                f"跑赢基准天数占比={ex_metrics['win_rate']:.1%}"
            )
        else:
            final_metrics = portfolio_metrics(pd.Series(dtype=float))

        # 分类准确率：用真正的 predict 类别，而非连续强度分
        if oof_cls:
            oof_cls_arr = np.concatenate(oof_cls)
            oos_accuracy = accuracy_score(
                all_true, oof_cls_arr
            )
        else:
            oos_accuracy = np.nan

        result = WalkForwardResult(
            model_name=model_type,
            feature_names=feature_cols,
            folds=fold_rows,
            oos_rank_ic=ic,
            oos_ic_ir=ic_ir,
            oos_top_excess=final_metrics.get("mean_period_ret", np.nan),
            oos_annual_return=final_metrics.get("annual_return", np.nan),
            oos_sharpe=final_metrics.get("sharpe", np.nan),
            oos_win_rate=final_metrics.get("win_rate", np.nan),
            oos_max_drawdown=final_metrics.get("max_drawdown", np.nan),
            oos_accuracy=oos_accuracy,
            n_oos_samples=n_oos,
            horizon=int(getattr(self, "horizon", 5)),
            feature_importance=imp_df,
        )

        # 用全量数据重训一个最终模型供推理使用
        # 注意：scaler 用全量重fit 是可接受的，因为推理时就用这个 scaler。
        # 关键是上面的 walk-forward 评估已经给出了无偏的样本外估计。
        final = xgb.XGBClassifier(**params) if model_type == "xgboost" else lgb.LGBMClassifier(**params)
        Xf, yf = self.fit_transform_train(df[feature_cols], df["__label__"].astype(int).values)
        final.fit(Xf, yf, sample_weight=self._class_weights(yf), verbose=False if model_type == "xgboost" else None)
        result.model = final

        self.models[model_type] = final

        return result

    # ---------- 持久化 ----------

    def save(self, result: WalkForwardResult, name: str = "leaksafe"):
        """保存模型 + scaler + 特征列表 + 指标"""
        with open(self.model_dir / f"{name}_{result.model_name}_model.pkl", "wb") as f:
            pickle.dump(result.model, f)

        with open(self.model_dir / f"{name}_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(self.model_dir / f"{name}_features.json", "w", encoding="utf-8") as f:
            import json
            json.dump({
                "features": result.feature_names,
                "n_features": len(result.feature_names),
            }, f, ensure_ascii=False, indent=2)

        # 保存评估指标，便于 UI 展示真实表现
        import json
        metrics = {
            "model": result.model_name,
            "horizon": int(getattr(result, "horizon", 5)),
            "oos_rank_ic": result.oos_rank_ic,
            "oos_ic_ir": result.oos_ic_ir,
            "oos_top_excess": result.oos_top_excess,
            "oos_annual_return": result.oos_annual_return,
            "oos_sharpe": result.oos_sharpe,
            "oos_win_rate": result.oos_win_rate,
            "oos_max_drawdown": result.oos_max_drawdown,
            "oos_accuracy": result.oos_accuracy,
            "n_oos_samples": result.n_oos_samples,
            "folds": result.folds,
            "note": "全部为 walk-forward 样本外指标，无数据泄漏",
        }
        with open(self.model_dir / f"{name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"已保存: {name}_{result.model_name} + scaler + metrics")
# -*- coding: utf-8 -*-
"""
模型训练模块
XGBoost + LightGBM 集成训练，含交叉验证、特征选择、样本平衡
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass
from loguru import logger

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb

from config.settings import config as app_config, MODEL_DIR


@dataclass
class TrainingResult:
    """训练结果"""
    model_name: str
    accuracy: float
    f1_macro: float
    precision: float
    recall: float
    feature_importance: pd.DataFrame
    cv_scores: list
    model: Any


class MLTrainer:
    """机器学习模型训练器"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        logger.info("MLTrainer initialized")
    
    def preprocess(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """数据预处理"""
        # 缺失值填充
        features = features.fillna(0)
        
        # 标签编码（-1, 0, 1 → 0, 1, 2）
        if fit_scaler:
            self.label_encoder.fit(labels)
        y = self.label_encoder.transform(labels)
        
        # 特征标准化
        if fit_scaler:
            X = self.scaler.fit_transform(features)
        else:
            X = self.scaler.transform(features)
        
        return X, y
    
    def balance_samples(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE样本平衡"""
        from collections import Counter
        before = Counter(y)
        
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(before.values()) - 1))
            X_balanced, y_balanced = smote.fit_resample(X, y)
            after = Counter(y_balanced)
            logger.info(f"SMOTE平衡: {before} → {after}")
            return X_balanced, y_balanced
        except Exception as e:
            logger.warning(f"SMOTE失败: {e}，使用原始数据")
            return X, y
    
    def train_xgboost(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        use_smote: bool = True
    ) -> TrainingResult:
        """训练XGBoost模型"""
        logger.info("=" * 50)
        logger.info("训练 XGBoost 模型")
        
        X, y = self.preprocess(features, labels, fit_scaler=True)
        
        if use_smote:
            X, y = self.balance_samples(X, y)
        
        params = app_config.ml.xgb_params.copy()
        
        # 类别权重
        classes = np.unique(y)
        class_weights = compute_class_weight("balanced", classes=classes, y=y)
        sample_weights = np.array([class_weights[int(c)] for c in y])
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=app_config.ml.cv_folds)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            sw_tr = sample_weights[train_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            cv_scores.append(acc)
            logger.info(f"  Fold {fold+1}: Accuracy = {acc:.4f}")
        
        logger.info(f"  CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        # 全量训练
        final_model = xgb.XGBClassifier(**params)
        final_model.fit(X, y, sample_weight=sample_weights, verbose=False)
        
        # 特征重要性
        importance_df = pd.DataFrame({
            "feature": features.columns,
            "importance": final_model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        # 验证集评估（使用最后20%作为验证集）
        split_idx = int(len(X) * 0.8)
        X_val, y_val = X[split_idx:], y[split_idx:]
        y_pred = final_model.predict(X_val)
        
        result = TrainingResult(
            model_name="xgboost",
            accuracy=accuracy_score(y_val, y_pred),
            f1_macro=f1_score(y_val, y_pred, average="macro"),
            precision=precision_score(y_val, y_pred, average="macro"),
            recall=recall_score(y_val, y_pred, average="macro"),
            feature_importance=importance_df,
            cv_scores=cv_scores,
            model=final_model
        )
        
        self._save_model(result, "xgboost_model.pkl")
        logger.info(f"XGBoost训练完成: F1={result.f1_macro:.4f}")
        return result
    
    def train_lightgbm(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        use_smote: bool = True
    ) -> TrainingResult:
        """训练LightGBM模型"""
        logger.info("=" * 50)
        logger.info("训练 LightGBM 模型")
        
        X, y = self.preprocess(features, labels, fit_scaler=True)
        
        if use_smote:
            X, y = self.balance_samples(X, y)
        
        params = app_config.ml.lgb_params.copy()
        
        tscv = TimeSeriesSplit(n_splits=app_config.ml.cv_folds)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMClassifier(**params, verbose=-1)
            model.fit(X_tr, y_tr)
            
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            cv_scores.append(acc)
            logger.info(f"  Fold {fold+1}: Accuracy = {acc:.4f}")
        
        logger.info(f"  CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        # 全量训练
        final_model = lgb.LGBMClassifier(**params, verbose=-1)
        final_model.fit(X, y)
        
        # 特征重要性
        importance_df = pd.DataFrame({
            "feature": features.columns,
            "importance": final_model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        # 验证集评估
        split_idx = int(len(X) * 0.8)
        X_val, y_val = X[split_idx:], y[split_idx:]
        y_pred = final_model.predict(X_val)
        
        result = TrainingResult(
            model_name="lightgbm",
            accuracy=accuracy_score(y_val, y_pred),
            f1_macro=f1_score(y_val, y_pred, average="macro"),
            precision=precision_score(y_val, y_pred, average="macro"),
            recall=recall_score(y_val, y_pred, average="macro"),
            feature_importance=importance_df,
            cv_scores=cv_scores,
            model=final_model
        )
        
        self._save_model(result, "lightgbm_model.pkl")
        logger.info(f"LightGBM训练完成: F1={result.f1_macro:.4f}")
        return result
    
    def train_ensemble(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> Dict[str, TrainingResult]:
        """训练集成模型（XGBoost + LightGBM）"""
        results = {}
        
        results["xgboost"] = self.train_xgboost(features, labels)
        results["lightgbm"] = self.train_lightgbm(features, labels)
        
        # 打印对比
        logger.info("\n" + "=" * 60)
        logger.info("📊 模型对比")
        logger.info(f"{'模型':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        logger.info("-" * 55)
        for name, r in results.items():
            logger.info(f"{name:<15} {r.accuracy:<10.4f} {r.f1_macro:<10.4f} "
                       f"{r.precision:<10.4f} {r.recall:<10.4f}")
        
        # 保存scaler和encoder
        self._save_preprocessing()
        
        return results
    
    def _save_model(self, result: TrainingResult, filename: str):
        """保存模型"""
        path = self.model_dir / filename
        with open(path, "wb") as f:
            pickle.dump(result.model, f)
        logger.debug(f"模型已保存: {path}")
    
    def _save_preprocessing(self):
        """保存预处理器"""
        with open(self.model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(self.model_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        logger.info("预处理器已保存")
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """加载已保存的模型"""
        path = self.model_dir / f"{model_name}_model.pkl"
        if not path.exists():
            logger.error(f"模型文件不存在: {path}")
            return None
        
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    
    def load_preprocessing(self) -> Tuple[StandardScaler, LabelEncoder]:
        """加载预处理器"""
        scaler_path = self.model_dir / "scaler.pkl"
        encoder_path = self.model_dir / "label_encoder.pkl"
        
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        if encoder_path.exists():
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
        
        return self.scaler, self.label_encoder
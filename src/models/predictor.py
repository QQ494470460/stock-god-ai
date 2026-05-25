# -*- coding: utf-8 -*-
"""
预测推理模块
加载训练好的模型，对单只或批量股票进行趋势预测和评分
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

import xgboost as xgb
import lightgbm as lgb

from config.settings import config as app_config, MODEL_DIR
from src.features.engine import feature_engine
from src.models.trainer import MLTrainer


@dataclass
class StockPrediction:
    """单只股票预测结果"""
    code: str
    name: str
    # 原始概率
    prob_up: float        # 上涨概率
    prob_neutral: float   # 震荡概率
    prob_down: float      # 下跌概率
    # 综合评分（0-100）
    score: float
    # 模型预测
    xgb_signal: int       # XGBoost预测 (-1/0/1)
    lgb_signal: int       # LightGBM预测
    # 特征值
    features_snapshot: Optional[Dict[str, float]] = None


class Predictor:
    """股票预测器"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.trainer = MLTrainer(self.model_dir)
        
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.scaler = None
        self.label_encoder = None
        
        self._load_models()
    
    def _load_models(self):
        """加载所有已训练模型"""
        try:
            self.xgb_model = self.trainer.load_model("xgboost")
            if self.xgb_model:
                logger.info("✅ XGBoost模型已加载")
            
            self.lgb_model = self.trainer.load_model("lightgbm")
            if self.lgb_model:
                logger.info("✅ LightGBM模型已加载")
            
            self.scaler, self.label_encoder = self.trainer.load_preprocessing()
            if self.scaler:
                logger.info("✅ 预处理器已加载")
        except Exception as e:
            logger.warning(f"模型加载部分失败: {e}")
    
    @property
    def is_ready(self) -> bool:
        return (self.xgb_model is not None or self.lgb_model is not None) and self.scaler is not None
    
    def predict_single(
        self,
        df: pd.DataFrame,
        code: str = "000000",
        name: str = "未知"
    ) -> StockPrediction:
        """对单只股票进行预测"""
        if not self.is_ready:
            raise RuntimeError("模型未加载，请先训练模型")
        
        # 构建特征
        features = feature_engine.build_features(df)
        if "close" in features.columns:
            features = features.drop(columns=["close"])
        
        # 取最新一条
        latest_features = features.iloc[-1:].fillna(0)
        
        # 特征快照
        feature_snapshot = latest_features.iloc[0].to_dict()
        
        # 标准化
        X = self.scaler.transform(latest_features)
        
        # XGBoost预测
        xgb_signal = 0
        xgb_proba = np.array([[0.33, 0.34, 0.33]])
        if self.xgb_model:
            xgb_pred = self.xgb_model.predict(X)[0]
            xgb_signal = int(self.label_encoder.inverse_transform([xgb_pred])[0])
            xgb_proba = self.xgb_model.predict_proba(X)
        
        # LightGBM预测
        lgb_signal = 0
        lgb_proba = np.array([[0.33, 0.34, 0.33]])
        if self.lgb_model:
            lgb_pred = self.lgb_model.predict(X)[0]
            lgb_signal = int(self.label_encoder.inverse_transform([lgb_pred])[0])
            lgb_proba = self.lgb_model.predict_proba(X)
        
        # 集成概率（加权平均）
        weights = []
        probas = []
        if self.xgb_model:
            weights.append(0.5)
            probas.append(xgb_proba)
        if self.lgb_model:
            weights.append(0.5)
            probas.append(lgb_proba)
        
        if len(probas) > 1:
            ensemble_proba = np.average(np.array(probas), axis=0, weights=weights)
        else:
            ensemble_proba = probas[0]
        
        # 确保是3类概率 [down, neutral, up]
        # 映射：label_encoder的classes_是 [-1, 0, 1] → [down, neutral, up]
        try:
            classes = self.label_encoder.classes_
            idx_up = list(classes).index(1)
            idx_neutral = list(classes).index(0)
            idx_down = list(classes).index(-1)
        except (ValueError, AttributeError):
            idx_up, idx_neutral, idx_down = 2, 1, 0
        
        prob_up = float(ensemble_proba[0][idx_up])
        prob_neutral = float(ensemble_proba[0][idx_neutral])
        prob_down = float(ensemble_proba[0][idx_down])
        
        # 综合评分：上涨概率高且下跌概率低的股票得分高
        score = (prob_up * 0.7 - prob_down * 0.3 + 0.3) * 100
        score = max(0, min(100, score))
        
        return StockPrediction(
            code=code,
            name=name,
            prob_up=prob_up,
            prob_neutral=prob_neutral,
            prob_down=prob_down,
            score=score,
            xgb_signal=xgb_signal,
            lgb_signal=lgb_signal,
            features_snapshot=feature_snapshot
        )
    
    def predict_batch(
        self,
        stock_data: Dict[str, pd.DataFrame],
        stock_names: Dict[str, str] = None
    ) -> List[StockPrediction]:
        """批量预测多只股票"""
        predictions = []
        
        for code, df in stock_data.items():
            if len(df) < 60:  # 至少需要60条数据
                continue
            try:
                name = stock_names.get(code, code) if stock_names else code
                pred = self.predict_single(df, code, name)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"预测 {code} 失败: {e}")
        
        # 按评分排序
        predictions.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"批量预测完成: {len(predictions)} 只股票")
        return predictions
    
    def get_top_recommendations(
        self,
        stock_data: Dict[str, pd.DataFrame],
        stock_names: Dict[str, str] = None,
        top_n: int = 20,
        min_score: float = 60.0
    ) -> pd.DataFrame:
        """获取TOP-N推荐股票"""
        predictions = self.predict_batch(stock_data, stock_names)
        
        # 过滤评分过低的
        predictions = [p for p in predictions if p.score >= min_score]
        
        # TOP-N
        top = predictions[:top_n]
        
        # 构建DataFrame
        records = []
        for p in top:
            records.append({
                "排名": len(records) + 1,
                "代码": p.code,
                "名称": p.name,
                "评分": round(p.score, 1),
                "上涨概率": round(p.prob_up * 100, 1),
                "震荡概率": round(p.prob_neutral * 100, 1),
                "下跌概率": round(p.prob_down * 100, 1),
                "XGB信号": "📈买入" if p.xgb_signal == 1 else ("📉卖出" if p.xgb_signal == -1 else "➡️持有"),
                "LGB信号": "📈买入" if p.lgb_signal == 1 else ("📉卖出" if p.lgb_signal == -1 else "➡️持有"),
            })
        
        df = pd.DataFrame(records)
        return df


# 全局单例
predictor = Predictor()
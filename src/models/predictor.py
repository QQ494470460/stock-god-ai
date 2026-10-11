# -*- coding: utf-8 -*-
"""
预测推理模块
加载训练好的模型，对单只或批量股票进行趋势预测和评分
"""

import pickle
import json
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

# 性能优化：xgboost / sklearn（经 MLTrainer 传递）都**不**在顶层 import。
# 实测两者累计约 5.1 秒（xgboost 2.6s + sklearn 2.5s），
# 而启动时只需要知道"模型文件在不在"，真正用到模型是点选股/预测时的事。
# 顶层导入会把这 5 秒白白加在每次启动上（打包后还要额外映射 143MB 的 xgboost.dll）。
# 注：lightgbm 已从本项目移除（单模型 xgboost，见 trainer.py），不再有第三方库的导入开销。
from config.settings import config as app_config, MODEL_DIR
from src.features.engine import feature_engine
from src.models.scoring import compute_score, get_score_interpretation


@dataclass
class StockPrediction:
    """单只股票预测结果"""
    code: str
    name: str
    # 原始概率
    prob_up: float        # 上涨概率
    prob_neutral: float   # 震荡概率
    prob_down: float      # 下跌概率
    # 综合评分（0-100，多重加权）
    score: float
    score_raw: float = 0.0      # 原始评分（未加权）
    confidence: float = 0.0     # 方向确信度 0-1
    agreement: float = 0.0      # 模型一致性 0-1（单模型后恒为 1.0）
    risk: float = 0.0           # 风险指标 0-1
    volatility: float = 0.0     # 近期年化波动率
    # 模型预测
    xgb_signal: int = 0         # XGBoost预测 (-1/0/1)
    # 已废弃：lightgbm 已从项目移除，该字段恒为 0。
    # 保留仅为防御性兼容——scripts/predict.py:77/83、src/trading/*、
    # src/models/tracker.py:169 等仍在读它，避免并行改造漏改导致 AttributeError。
    lgb_signal: int = 0
    # 推荐理由
    reasons: list = None         # top 3 特征名称
    # 特征值
    features_snapshot: Optional[Dict[str, float]] = None
    # 关联板块
    sector: str = ""
    sector_change: float = 0.0


class Predictor:
    """股票预测器"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        # MLTrainer 顶层会 import sklearn（~2.5s），这里只放占位，
        # 真正用到（加载/训练模型）时才通过 trainer property 实例化
        self._trainer = None

        # 懒加载：构造时只读目录，不读模型文件。
        # 之前 __init__ 里直接 _load_models()，导致 import 本模块就要 8-9 秒
        # （xgboost 反序列化），拖慢整个看板首屏。
        self._xgb_model: Optional[Any] = None
        self.xgb_features: Optional[List[str]] = None  # SHAP 筛选后的特征
        self._scaler = None
        self._label_encoder = None
        self._models_loaded = False
        # RLock（可重入）：即便将来 _load_models 里误读了 property 也不会死锁
        self._load_lock = threading.RLock()

        # 单模型（xgboost）权重恒为 1.0；保留 dict 结构以兼容历史读法
        self.model_weights: Dict[str, float] = {"xgboost": 1.0}

    @property
    def trainer(self):
        """按需实例化 MLTrainer（首次访问才会 import sklearn）"""
        if self._trainer is None:
            from src.models.trainer import MLTrainer

            self._trainer = MLTrainer(self.model_dir)
        return self._trainer

    @trainer.setter
    def trainer(self, v):
        self._trainer = v

    # ---------- 懒加载 ----------

    @property
    def xgb_model(self) -> Optional[Any]:
        self._ensure_loaded()
        return self._xgb_model

    @xgb_model.setter
    def xgb_model(self, v):
        self._xgb_model = v

    @property
    def scaler(self):
        self._ensure_loaded()
        return self._scaler

    @scaler.setter
    def scaler(self, v):
        self._scaler = v

    @property
    def label_encoder(self):
        self._ensure_loaded()
        return self._label_encoder

    @label_encoder.setter
    def label_encoder(self, v):
        self._label_encoder = v

    def _ensure_loaded(self) -> None:
        """首次访问模型时才真正加载（线程安全，只加载一次）"""
        if self._models_loaded:
            return
        with self._load_lock:
            if self._models_loaded:  # 双重检查
                return
            self._load_models()
            self._models_loaded = True

    def _load_models(self):
        """加载已训练模型（当前只有 xgboost）。

        注意：这里必须只碰私有属性 _xgb_model / _scaler。
        若写 self.xgb_model 会走 property getter -> _ensure_loaded() -> 再次
        申请同一把锁。普通 Lock 不可重入，会直接死锁。
        """
        try:
            self._xgb_model = self.trainer.load_model("xgboost")
            if self._xgb_model is not None:
                logger.info("✅ XGBoost模型已加载")
                # 加载 SHAP 特征列表（如存在）
                self.xgb_features = self._load_feature_list("xgboost")

            self._scaler, self._label_encoder = self.trainer.load_preprocessing()
            if self._scaler is not None:
                logger.info("✅ 预处理器已加载")
        except Exception as e:
            logger.warning(f"模型加载部分失败: {e}")
    
    def _load_feature_list(self, model_type: str):
        """加载 Optuna 训练时保存的特征列表"""
        feat_file = MODEL_DIR / f"features_{model_type}.json"
        if feat_file.exists():
            try:
                with open(feat_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                feats = data.get("features", [])
                logger.info(f"✅ {model_type} SHAP 特征列表: {len(feats)} 个")
                return feats
            except Exception:
                pass
        return None
    
    def reload(self):
        """热重载模型（训练完成后调用）"""
        logger.info("🔄 热重载模型…")
        self._xgb_model = None
        self.xgb_features = None
        self._scaler = None
        self._label_encoder = None
        self._models_loaded = False
        self._load_models()
        self._models_loaded = True
        if self.is_ready:
            logger.info("✅ 模型热重载完成")
        else:
            logger.warning("⚠️ 模型热重载失败，可能训练尚未完成")
    
    @property
    def is_ready(self) -> bool:
        """模型是否可用。

        性能关键：看板首页一进来就调它。原实现读 `self.xgb_model` → property →
        `_ensure_loaded()` → 反序列化全部模型，等于**每次启动都先付
        xgboost + sklearn 的导入与加载开销（约 5 秒）**，
        而这个调用本来只是想问一句"模型在不在"。
        改成：没加载时只看文件是否存在，绝不在"问状态"时偷偷加载。
        """
        if self._models_loaded:
            return self._xgb_model is not None and self._scaler is not None
        # 未加载状态：只做一次轻量的文件存在性检查（不 import 任何重库）
        try:
            return (self.model_dir / "xgboost_model.pkl").exists()
        except Exception:  # noqa: BLE001
            return False

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
        
        # 特征快照（所有特征，含新增指标）
        feature_snapshot = latest_features.iloc[0].to_dict()
        
        # 对齐到scaler训练时的特征列（新指标不参与模型推理，只影响评分）
        scaler_cols = getattr(self.scaler, "feature_names_in_", None)
        if scaler_cols is not None:
            aligned = latest_features.reindex(columns=scaler_cols, fill_value=0.0)
            aligned.columns = scaler_cols  # 恢复 feature names
        else:
            aligned = latest_features.iloc[:, :self.scaler.n_features_in_]
        
        X = self.scaler.transform(aligned)
        # Wrap with feature names so model predict doesn't warn
        X_input = pd.DataFrame(X, columns=scaler_cols) if scaler_cols is not None else X
        
        # XGBoost预测
        xgb_signal = 0
        xgb_proba = np.array([[0.33, 0.34, 0.33]])
        if self.xgb_model:
            xgb_pred = self.xgb_model.predict(X_input)[0]
            xgb_signal = int(self.label_encoder.inverse_transform([xgb_pred])[0])
            xgb_proba = self.xgb_model.predict_proba(X_input)
        
        # 单模型（xgboost）：只有一份 proba，权重恒为 1.0
        weights = [self.model_weights.get("xgboost", 1.0)]
        probas = [xgb_proba]
        
        # 对齐模型类别数：强制统一为 label_encoder.classes_
        n_classes = len(self.label_encoder.classes_)
        
        # 将各模型 proba 扩充/裁剪至 n_classes
        aligned_probas = []
        for proba in probas:
            n_model = proba.shape[1]
            if n_model == n_classes:
                aligned_probas.append(proba)
            elif n_model < n_classes:
                # 少类补零（二分类补震荡类别）
                padded = np.zeros((1, n_classes), dtype=np.float64)
                if n_model == 2 and n_classes == 3:
                    # 二分类 proba[0]=跌, proba[1]=涨 → 三分类位置 [0]=跌, [1]=震荡(0), [2]=涨
                    padded[0, 0] = proba[0, 0]  # 跌
                    padded[0, 2] = proba[0, 1]  # 涨
                else:
                    padded[0, :n_model] = proba
                aligned_probas.append(padded)
            else:
                # 多类裁剪
                aligned_probas.append(proba[:, :n_classes])
        
        if len(aligned_probas) > 1:
            ensemble_proba = np.average(np.array(aligned_probas), axis=0, weights=weights)
        else:
            ensemble_proba = aligned_probas[0]
        
        # 概率分解（统一从 classes_ 取，兼容 3 类与 5 类）
        # 历史上 predictor 仅面对三分类（classes=[-1,0,1]），用 .index(1)/.index(-1)
        # 硬编档位；新生产模型是 5 档截面分位（classes=[0,1,2,3,4]），
        # 直接 .index(1) 会把「次强档」误当上涨，需分支处理。
        classes = list(self.label_encoder.classes_)
        n_classes = len(classes)
        if n_classes == 5:
            # 5 档截面分位 0..4：上=top2 档(3,4)，下=bottom2 档(0,1)，中=档2
            prob_up = float(ensemble_proba[0][3] + ensemble_proba[0][4])
            prob_down = float(ensemble_proba[0][0] + ensemble_proba[0][1])
            prob_neutral = float(ensemble_proba[0][2])
        else:
            idx_up = classes.index(1)
            idx_down = classes.index(-1)
            prob_up = float(ensemble_proba[0][idx_up])
            prob_down = float(ensemble_proba[0][idx_down])
            prob_neutral = float(ensemble_proba[0][classes.index(0)]) if 0 in classes else 0.0
        
        # 综合评分 V4 — 基于预期收益的可解释评分（2026-07-10）
        score_raw = (prob_up * 0.7 - prob_down * 0.3 + 0.3) * 100
        score_raw = max(0, min(100, score_raw))
        
        confidence = abs(prob_up - prob_down)
        
        # 单模型（xgboost）不存在"两模型分歧"，agreement 恒取 1.0。
        # 保留该变量/字段的理由：它是 compute_score 评分公式的输入，且下游
        # src/trading/executor.py:218 用 require_agreement < 0.8 做硬闸门拒单，
        # 恒取 1.0 才能正常放行；改成 0 或删除字段都会让所有订单被拦下。
        agreement = 1.0
        
        returns = df["close"].pct_change().dropna().tail(20)
        volatility = float(np.nan_to_num(returns.std() * np.sqrt(252), nan=0.3))
        vol_clamp = max(0.1, min(volatility, 1.2))
        risk_pct = (vol_clamp - 0.1) / 1.1
        
        # ★ 新评分公式：预期收益 + 加成分项，每项有明确上限
        score, score_breakdown = compute_score(
            prob_up=prob_up,
            prob_neutral=prob_neutral,
            prob_down=prob_down,
            confidence=confidence,
            agreement=agreement,
            risk_pct=risk_pct,
            volatility=volatility,
            feature_snapshot=feature_snapshot,
            n_classes=n_classes,
            config=dict(
                threshold_up=app_config.ml.threshold_up,
                threshold_down=app_config.ml.threshold_down,
            )
        )
        
        # 推荐理由：评分分解 + 市场状态
        interpretation, note = get_score_interpretation(score)
        reasons = []
        b = score_breakdown
        if b.get("conf_bonus", 0) >= 8:
            reasons.append(f"高确信度 ({confidence:.0%})")
        if b.get("risk_penalty", 0) < -5:
            reasons.append(f"高波动 ({volatility:.1%})")
        if b.get("regime_factor", 1.0) != 1.0:
            reasons.append(b.get("regime_note", ""))
        if abs(b.get("expected_return", 0)) > 0.01:
            reasons.append(f"预期收益 {b['expected_return']:+.2%}")
        if not reasons:
            reasons.append(note)
        
        return StockPrediction(
            code=code,
            name=name,
            prob_up=prob_up,
            prob_neutral=prob_neutral,
            prob_down=prob_down,
            score=score,
            score_raw=score_raw,
            confidence=confidence,
            agreement=agreement,
            risk=risk_pct,
            volatility=volatility,
            xgb_signal=xgb_signal,
            lgb_signal=0,  # 已废弃字段，恒为 0（见 StockPrediction.lgb_signal 注释）
            reasons=reasons,
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
        min_score: float = 50.0
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
            # 一致性标签（agreement 恒为 1.0，需单独处理中性信号 0，
            # 否则会把"观望"误显示成"✅ 一致看跌"）
            if p.agreement >= 1.0:
                if p.xgb_signal == 1:
                    agree_label = "✅ 一致看涨"
                elif p.xgb_signal == -1:
                    agree_label = "✅ 一致看跌"
                else:
                    agree_label = "✅ 一致观望"
            elif p.agreement <= 0.3:
                agree_label = "⚠️ 分歧"
            else:
                agree_label = "—"
            
            reasons_str = "; ".join(p.reasons[:3]) if p.reasons else "—"
            
            records.append({
                "排名": len(records) + 1,
                "代码": p.code,
                "名称": p.name,
                "评分": round(p.score, 1),
                "确信度": round(p.confidence * 100, 1),
                "一致": agree_label,
                "上涨概率": round(p.prob_up * 100, 1),
                "震荡概率": round(p.prob_neutral * 100, 1),
                "下跌概率": round(p.prob_down * 100, 1),
                "波动率": f"{p.volatility*100:.1f}%",
                "XGB信号": "📈买入" if p.xgb_signal == 1 else ("📉卖出" if p.xgb_signal == -1 else "➡️持有"),
                "推荐理由": reasons_str,
            })
        
        df = pd.DataFrame(records)
        return df


# 全局单例
predictor = Predictor()
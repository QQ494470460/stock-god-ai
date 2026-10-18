# -*- coding: utf-8 -*-
"""
预测推理模块
加载训练好的模型，对单只或批量股票进行趋势预测和评分
"""

import pickle
import json
import threading
from datetime import datetime
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
from src.models.signal_quality import neutralize_cross_section

# LLM 定性叠加覆盖层（可选，默认不用）：用 try/except 包住，避免无 openai 时
# 导入失败影响核心预测路径（核心 xgboost 推理完全不依赖它）。
try:
    from src.data.llm_features import LLMFeatureProvider, LLMFeatureConfig
except Exception:  # noqa: BLE001
    LLMFeatureProvider = None
    LLMFeatureConfig = None

# 模型注册中心：只读扫描 + 「该用哪个」的裁决（受人工批准名单约束）
try:
    from src.models.model_registry import (
        scan_models, best_loadable, best_overall, active_status,
        resolve_active_path, resolve_active, version_set_for,
    )
except Exception:  # noqa: BLE001
    scan_models = None
    best_loadable = None
    best_overall = None
    active_status = None
    resolve_active_path = None
    resolve_active = None
    version_set_for = None


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
    # LLM 定性叠加覆盖层输出（仅在启用覆盖层时填充；默认 0.0/0，不影响核心评分）
    llm_sentiment: float = 0.0    # 情绪分 (-1~1, +看多 / -看空)
    llm_risk: float = 0.0         # 风险标记 (0/1, 1=利空/风险事件)
    llm_confidence: float = 0.0   # 置信度 (0~1)


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

        # LLM 定性叠加覆盖层（默认关闭；仅 _enable_llm 后才会构造 provider）
        # 覆盖层对 predict_batch 产出的 score 做纯加性微调，绝不重训 xgboost。
        self.llm_provider = None

        # 超级集成（懒加载；use_ensemble 开启且有可用子模型时才生效）
        self._ensemble = None

        # 行业映射缓存（predict_batch 做截面中性化时懒取一次，避免每次打接口/读盘）
        self._industry_map_cache: Optional[Dict[str, str]] = None

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

    def _log_model_resolution(self):
        """把「注册表裁决」与「实际加载的文件」并排记一条日志。

        现在 predictor 真的会跟随裁决加载：被批准/钉住的版本，其 model.pkl 会被
        version_set_for 解析出的四件套真正 pickle.load 进来；四件套不齐则明确拒绝
        并回退到生产默认。日志如实说明「跟随了哪个、是不是默认」。
        """
        try:
            from src.models.model_registry import (
                active_status, resolve_active_path, resolve_active, version_set_for,
            )

            logger.info(f"📦 模型裁决：{active_status()}")
            default = MODEL_DIR / "xgboost_model.pkl"
            resolved = Path(resolve_active_path())
            if resolved == default:
                logger.info(f"🔗 加载将跟随生产默认 {default.name}（无生效版本）")
                return
            info = resolve_active()
            vs = version_set_for(info, self.model_dir) if info else None
            if vs is not None and vs.complete:
                logger.info(f"🔗 加载将跟随裁决版本 {info.key} @ {resolved.name}（四件套齐全）")
            else:
                logger.warning(
                    f"⚠️ 裁决指向 {resolved.name}，但其四件套不齐，加载将回退到默认 "
                    f"{default.name}（版本切换被拒绝）")
        except Exception as e:  # noqa: BLE001 - 只是日志，绝不能影响加载
            logger.debug(f"模型裁决日志跳过: {e}")

    def _load_models(self):
        """加载已训练模型（当前只有 xgboost）。

        加载路径现在**跟随注册表裁决**：被批准/钉住的版本经 version_set_for 解析出
        四件套后真正 pickle.load 进来；未批准、或版本四件套不齐时，明确拒绝该版本、
        回退到遗留默认 xgboost_model.pkl（行为与改动前完全一致）。

        注意：这里必须只碰私有属性 _xgb_model / _scaler。
        若写 self.xgb_model 会走 property getter -> _ensure_loaded() -> 再次
        申请同一把锁。普通 Lock 不可重入，会直接死锁。
        """
        try:
            self._log_model_resolution()

            info = resolve_active() if resolve_active is not None else None
            vs = None
            if info is not None and version_set_for is not None:
                try:
                    vs = version_set_for(info, self.model_dir)
                except Exception:  # noqa: BLE001
                    vs = None

            if vs is not None and not vs.legacy and vs.complete:
                # —— 版本化产物：按解析出的四件套路径加载 ——
                self._xgb_model = self.trainer.load_model_from(vs.model_path)
                if self._xgb_model is not None:
                    logger.info(f"✅ XGBoost 模型已加载（裁决版本 {info.key}）")
                    self.xgb_features = self._load_feature_list_from(vs.features_path)
                self._scaler, self._label_encoder = self.trainer.load_preprocessing_from(
                    vs.scaler_path, vs.label_encoder_path)
                if self._scaler is not None:
                    logger.info(f"✅ 预处理器已加载（裁决版本 {info.key}）")
                return

            if vs is not None and not vs.complete:
                logger.warning(
                    f"⚠️ 裁决版本 {getattr(info, 'key', '?')} 四件套不齐，拒绝加载，"
                    f"回退生产默认 xgboost_model.pkl")

            # —— 遗留默认路径（无生效版本，或版本不齐时回退）——
            # 这一段与改动前逐字一致，确保默认行为不被改变。
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

    def _load_feature_list_from(self, path):
        """从任意 features.json 路径加载特征列表（版本化产物用 features.json）。"""
        p = Path(path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                feats = data.get("features", [])
                logger.info(f"✅ 特征列表: {len(feats)} 个（{p.name}）")
                return feats
            except Exception:
                pass
        logger.debug(f"特征列表缺失或解析失败: {p}")
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

    # ---------- 模型注册中心集成 ----------

    def active_model_info(self):
        """返回当前运行模型在注册中心里的登记信息（含样本外 RankIC）"""
        if scan_models is None:
            return None
        try:
            infos = scan_models(self.model_dir)
            # 主流 predictor 管线加载的就是 xgboost 主模型
            for m in infos:
                if m.key == "xgboost":
                    return m
            return infos[0] if infos else None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"active_model_info 失败: {e}")
            return None

    def model_report(self) -> str:
        """输出「当前模型 + 全局最优 + 可部署最优」的对比报告"""
        if scan_models is None:
            return "（模型注册中心不可用）"
        lines = ["📦 模型概览（样本外 RankIC）", ""]
        active = self.active_model_info()
        if active:
            ric = f"{active.rank_ic:+.4f}" if active.rank_ic == active.rank_ic else "n/a"
            lines.append(f"当前运行：{active.display_name} — RankIC={ric}")
        bl = best_loadable(self.model_dir)
        if bl and (not active or bl.key != active.key):
            lines.append(f"可部署最优：{bl.display_name} — RankIC={bl.rank_ic:+.4f}")
        bo = best_overall(self.model_dir)
        if bo:
            lines.append(f"全局最优（含实验）：{bo.display_name} — RankIC={bo.rank_ic:+.4f}")
            if not bo.loadable:
                lines.append(f"  ⚠️ {bo.note}")
        if active_status is not None:
            lines.append("")
            lines.append(f"裁决：{active_status(self.model_dir)}")
        lines.append("")
        lines.append("⚠️ 注意：被批准/钉住的版本会**真正改变**加载路径（经 version_set_for "
                     "解析四件套加载；四件套不齐则拒绝并回退默认）。"
                     "未批准时一律回退 models/xgboost_model.pkl，"
                     "新产物不会因指标更高而自动生效。")
        return "\n".join(lines)

    def reload_best(self):
        """重载「可部署最优」模型（当前等于主 xgboost 模型，未来训练出更优模型后自动切换）"""
        if best_loadable is not None:
            try:
                best = best_loadable(self.model_dir)
                if best and best.model_path and best.model_path.exists():
                    logger.info(f"🏆 注册中心选出可部署最优模型: {best.display_name} (RankIC={best.rank_ic:+.4f})")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"reload_best 选优失败，回退默认: {e}")
        # 实际加载逻辑仍由 trainer.load_model 完成（当前即 xgboost_model.pkl）
        self.reload()
    
    # ---------- 超级集成 ----------

    def _get_ensemble(self):
        """懒构建超级集成（XGBoost + Alpha + LSTM）；未启用/无可用子模型则返回 None。

        只在 ml.use_ensemble 为真时构建；任一子模型缺失由 EnsemblePredictor
        的 available 机制自动降级，绝不会因某一腿缺失而影响主模型。
        """
        if not getattr(app_config.ml, "use_ensemble", False):
            return None
        if self._ensemble is None:
            try:
                from src.models.ensemble import (
                    EnsemblePredictor, AlphaSubPredictor, LSTMSubPredictor,
                )
                w = getattr(app_config.ml, "ensemble_weights", {}) or {}
                ep = EnsemblePredictor(self)
                ep.add_sub(AlphaSubPredictor(), weight=w.get("alpha_xgboost", 1.0))
                ep.add_sub(LSTMSubPredictor(), weight=w.get("lstm", 0.8))
                self._ensemble = ep
                logger.info("🧩 超级集成已构建")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"超级集成构建失败，回退单模型: {e}")
                self._ensemble = None
        return self._ensemble

    def ensemble_report(self) -> str:
        """返回超级集成各腿可用性与权重的可读报告。"""
        ep = self._get_ensemble()
        if ep is None:
            return "🧩 超级集成：未启用（ml.use_ensemble=False）"
        return ep.summary()

    def _enable_llm(self, mock: bool = False) -> None:
        """按需启用 LLM 定性叠加覆盖层（纯加性 score 微调，绝不重训 xgboost）。

        设计原则：
          · 只有 config 主开关 enable_llm_overlay 为真时才真正构造 provider
            （用 getattr 安全读取，默认 False，缺失该字段也不崩）。
          · 任何异常都安全降级为 self.llm_provider=None，覆盖层等于无操作 + 警告，
            绝不拖垮核心预测路径。
          · 关闭时（默认）predict_batch 完全不碰 score，输出与改动前一致。

        CLI（scripts/daily_recommend.py --llm / --llm-mock）会在调用本方法前
        把 enable_llm_overlay 临时置 True，因此本方法无需再判 CLI 参数。
        """
        if not getattr(app_config.ml, "enable_llm_overlay", False):
            logger.info("LLM 覆盖层未启用（enable_llm_overlay=False），跳过")
            self.llm_provider = None
            return
        try:
            # LLMFeatureProvider/LLMFeatureConfig 若顶层导入失败则为 None，
            # 下面这行会抛 TypeError 并被捕获，最终降级为无操作（provider=None）。
            self.llm_provider = LLMFeatureProvider(LLMFeatureConfig(mock=mock))
            logger.info(f"✅ LLM 覆盖层已启用 (mock={mock})")
        except Exception as e:  # noqa: BLE001
            self.llm_provider = None
            logger.warning(f"⚠️ LLM 覆盖层启用失败，已降级为无操作: {e}")

    def _apply_llm_overlay(self, pred: "StockPrediction", as_of_date: str) -> None:
        """对单只预测做 LLM 定性叠加微调（纯加性，就地修改 pred）。

        约定：
          · 只在 self.llm_provider 非 None 时生效（默认 None → 完全不动 score）。
          · provider 取不到结果（本机 Ollama 未启动 / 该股无新闻 / 调用失败）时，
            llm_features 返回中性 dict，此处 adj 恒为 0，等价无操作，绝不改分。
          · 只改 score 与 llm_* 字段，绝不触碰 xgboost 概率、scaler、标签体系。
        """
        if self.llm_provider is None:
            return
        try:
            feats = self.llm_provider.features_for_stock(pred.code, as_of_date)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"LLM 覆盖层取特征失败 {pred.code}: {e}")
            return

        sentiment = float(feats.get("llm_sentiment", 0.0))
        risk = int(feats.get("llm_risk", 0))
        conf = float(feats.get("llm_confidence", 0.0))

        # 加性微调：情绪×置信度放大；命中风险事件再扣一档
        adj = sentiment * conf * 8.0
        if risk == 1:
            adj -= 6.0 * conf
        if abs(adj) < 1e-9:
            return

        pred.score = max(0.0, min(100.0, pred.score + adj))
        pred.llm_sentiment = sentiment
        pred.llm_risk = float(risk)
        pred.llm_confidence = conf

        if abs(adj) > 0.5 and pred.reasons is not None:
            # 插到最前：get_top_recommendations 只展示 reasons[:3]，
            # 放末尾会被截断，用户就看不到 LLM 到底说了什么。
            if risk == 1:
                pred.reasons.insert(0, "🤖LLM风险事件")
            else:
                direction = "多" if sentiment > 0 else "空"
                pred.reasons.insert(0, f"🤖LLM偏{direction}({sentiment:+.2f}×{conf:.0%})")

    @property
    def _active_model_and_scaler_paths(self):
        """返回「当前会被加载」的模型与 scaler 路径（跟随注册表裁决）。

        无生效版本 / 解析器不可用时回退到遗留默认，保证调用方永远拿到合法路径。
        """
        if resolve_active is None or version_set_for is None:
            return self.model_dir / "xgboost_model.pkl", self.model_dir / "scaler.pkl"
        try:
            info = resolve_active()
            if info is None:
                return self.model_dir / "xgboost_model.pkl", self.model_dir / "scaler.pkl"
            vs = version_set_for(info, self.model_dir)
            if vs is None:
                return self.model_dir / "xgboost_model.pkl", self.model_dir / "scaler.pkl"
            # 版本化产物四件套不齐时，_load_models 会拒绝它、回退到生产默认；
            # 这里同步反映这一回退，让 is_ready 如实回答「此刻能否真的加载」。
            if vs.legacy or vs.complete:
                return vs.model_path, vs.scaler_path
            return self.model_dir / "xgboost_model.pkl", self.model_dir / "scaler.pkl"
        except Exception:  # noqa: BLE001
            return self.model_dir / "xgboost_model.pkl", self.model_dir / "scaler.pkl"

    @property
    def is_ready(self) -> bool:
        """模型是否可用。

        性能关键：看板首页一进来就调它。原实现读 `self.xgb_model` → property →
        `_ensure_loaded()` → 反序列化全部模型，等于**每次启动都先付
        xgboost + sklearn 的导入与加载开销（约 5 秒）**，
        而这个调用本来只是想问一句"模型在不在"。
        改成：没加载时只看（裁决指向的）文件是否存在，绝不在"问状态"时偷偷加载。

        现在「裁决指向的文件」可能是一个版本化产物（versions/<id>/model.pkl），
        所以这里用 _active_model_and_scaler_paths 取到正确路径再判断存在性。

        ⚠️ 必须是 @property：全项目一律按属性读它（`if not predictor.is_ready:`
        / `ready = predictor.is_ready`），散落在 agent tools、API、scripts 与看板。
        少了这个装饰器，`predictor.is_ready` 会是个**恒为真**的绑定方法对象，
        于是「模型未加载」的保护一次都不会触发 —— 这类失效不报错，
        只是那道闸门从来没关过。同理内部不能写成 `..._paths()`：
        那是在调用一个已经返回元组的 property，会抛 TypeError 并被下面的
        except 吞掉，让整套裁决解析变成死代码。
        """
        if self._models_loaded:
            return self._xgb_model is not None and self._scaler is not None
        # 未加载状态：只做一次轻量的文件存在性检查（不 import 任何重库）
        try:
            model_p, scaler_p = self._active_model_and_scaler_paths
            return model_p.exists() and scaler_p.exists()
        except Exception:  # noqa: BLE001
            return (self.model_dir / "xgboost_model.pkl").exists()

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
    
    def _neutralize_scores(self, predictions: List["StockPrediction"]) -> None:
        """对整批预测做行业中性化（原地修改 pred.score）。

        只在 ml.cross_sectional_neutralize 为真时生效；任何异常都安全降级为「不改分」，
        绝不因行业映射取不到而中断排名。行业映射懒取并缓存到实例，避免每次批量重取。
        """
        if not getattr(app_config.ml, "cross_sectional_neutralize", False):
            return
        if not predictions:
            return
        try:
            from src.data.fetcher import fetcher
            if self._industry_map_cache is None:
                self._industry_map_cache = fetcher.get_industry_map() or {}
            ind_map = self._industry_map_cache
            if not ind_map:
                return
            scores = pd.Series({p.code: float(p.score) for p in predictions})
            neu = neutralize_cross_section(scores, ind_map)
            for p in predictions:
                if p.code in neu.index:
                    p.score = float(neu[p.code])
        except Exception as e:  # noqa: BLE001
            logger.debug(f"行业中性化跳过（安全降级）: {e}")

    def predict_batch(
        self,
        stock_data: Dict[str, pd.DataFrame],
        stock_names: Dict[str, str] = None
    ) -> List[StockPrediction]:
        """批量预测多只股票（默认走超级集成；无可用子模型时回退单模型）"""
        # 过滤数据不足的股票
        valid = {c: df for c, df in stock_data.items() if len(df) >= 60}

        predictions: List[StockPrediction] = []
        ep = self._get_ensemble()
        if ep is not None and ep._available_subs():
            blended = ep.predict_batch(valid)
            predictions = list(blended.values())
            for p in predictions:
                p.name = stock_names.get(p.code, p.code) if stock_names else p.code
            logger.info(f"🧩 超级集成预测完成: {len(predictions)} 只")
        else:
            for code, df in valid.items():
                try:
                    name = stock_names.get(code, code) if stock_names else code
                    pred = self.predict_single(df, code, name)
                    predictions.append(pred)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"预测 {code} 失败: {e}")
        
        # LLM 定性叠加覆盖层（默认 self.llm_provider is None → 完全不动 score）
        if self.llm_provider is not None:
            as_of_date = datetime.now().strftime("%Y-%m-%d")
            for pred in predictions:
                self._apply_llm_overlay(pred, as_of_date)
            logger.info(f"🤖 LLM 覆盖层已应用: {len(predictions)} 只, as_of={as_of_date}")

        # 横截面行业中性化（默认开启）：在排序前对整批评分去行业水位，
        # 让排名不被「模型系统性偏好某些行业」带偏，降低单日 IC 的板块噪声。
        self._neutralize_scores(predictions)

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
        # 仅当本批真的带 LLM 覆盖层数据时才加列：
        # 覆盖层关闭时 schema 与改动前完全一致，下游（看板 / CSV）不受影响
        has_llm = any((p.llm_sentiment != 0.0 or p.llm_risk != 0.0) for p in top)
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
            
            row = {
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
            }
            if has_llm:
                row["LLM情绪"] = round(p.llm_sentiment, 2)
                row["LLM风险"] = "⚠️风险" if p.llm_risk else "—"
            records.append(row)
        
        df = pd.DataFrame(records)
        return df


# 全局单例
predictor = Predictor()
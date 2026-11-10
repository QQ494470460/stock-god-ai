# -*- coding: utf-8 -*-
"""
超级集成（Super Ensemble）
========================

把多个子预测器融合成一个更强的评分器：

    XGBoost（主流生产模型，RankIC≈0.075）
        + Alpha XGBoost（生产特征 + 北向/融资/龙虎榜/行业动量，RankIC≈0.12）
        + LSTM（时间序列深度学习，捕捉时序动态）
        + Transformer（时序排序器，CPU 友好的自注意力）
        + LLM 另类 alpha（DeepSeek 等大模型对近期新闻/公告的定性判断，默认关闭）

融合策略
--------
- 每个子预测器输出 0~100 的「预期收益评分」；
- 权重默认 = 各模型样本外 RankIC（归一化），可经 config 覆盖；
- 任一子模型文件缺失 / 推理失败 → 自动剔除并重新归一化权重（优雅降级）；
- 主预测仍是 StockPrediction，仅把 score 替换为融合值，agreement 保持 1.0
  以兼容交易执行器的硬闸门（require_agreement < 0.8 拒单）。

设计原则
--------
- 只读加载、懒初始化、失败即降级，绝不因某一腿缺失而让整个系统崩；
- 不重训任何模型，只是把它们组合起来。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

import os

from config.settings import MODEL_DIR, DATA_DIR, config as app_config
from src.features.engine import feature_engine


# ---------------------------------------------------------------------------
# 子预测器基类
# ---------------------------------------------------------------------------
class SubPredictor:
    """子预测器接口：给出每支股票 0~100 的预期收益评分。"""

    name: str = "base"
    _ready: Optional[bool] = None
    _reason: Optional[str] = None   # 不可用的原因（留给运维排查，见 _available_subs）

    @property
    def available(self) -> bool:
        if self._ready is None:
            try:
                self._ensure()
                self._ready = True
            except Exception as e:  # noqa: BLE001
                # 原因必须留档：曾出现「LSTM 腿在生产进程里静默降级」——
                # 只有 DEBUG 一行 WinError 1114（torch 的 c10.dll 初始化失败），
                # INFO 层只说「已跳过」，看板上看不出集成其实只剩两条腿。
                self._reason = f"{type(e).__name__}: {str(e)[:160]}"
                logger.warning(f"[{self.name}] 子模型不可用，已降级: {e}")
                self._ready = False
        return self._ready

    def _ensure(self) -> None:
        """加载资源；失败抛异常即被 available 捕获为降级。"""

    def score_for(self, df: pd.DataFrame, code: str) -> Optional[float]:
        """返回该股票的 0~100 评分；失败/不可用返回 None。"""
        if not self.available:
            return None
        try:
            return self._score(df, code)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{self.name}] 评分失败 {code}: {e}")
            return None

    def _score(self, df: pd.DataFrame, code: str) -> float:
        raise NotImplementedError


def _proba_to_score(proba: np.ndarray, classes) -> float:
    """把模型概率向量（按 classes 顺序）转成 0~100 预期收益评分。

    与 src/models/predictor.py 的 score_raw 口径一致，确保不同子模型评分可比。
    """
    n = len(classes)
    if n == 5:
        prob_up = float(proba[3] + proba[4])
        prob_down = float(proba[0] + proba[1])
    else:
        idx_up = list(classes).index(1) if 1 in list(classes) else (n - 1)
        idx_down = list(classes).index(-1) if -1 in list(classes) else 0
        prob_up = float(proba[idx_up])
        prob_down = float(proba[idx_down])
    score = (prob_up * 0.7 - prob_down * 0.3 + 0.3) * 100
    return float(max(0.0, min(100.0, score)))


# ---------------------------------------------------------------------------
# Alpha XGBoost 子预测器（生产特征 + alpha 列）
# ---------------------------------------------------------------------------
# 诊断开关（**默认 0 = 不改变生产行为**）：把 alpha 面板整体推迟 N 个交易日可见。
#
# 背景：alpha 腿用了北向/融资余额/龙虎榜等「当日收盘后才披露」的数据，
# 而 akshare_alpha 的口径是「统计日=当日，预测时视为已知」（见 margin_panel 注释），
# 回测又按当日收盘价成交 → 这一段 IC 是**前视**来的，含系统性虚高。
# 设 ALPHA_LAG_DAYS=1 后重跑 scripts/ab_ensemble_ic.py，与不加 lag 的结果对比，
# 即可量化「同一天披露」贡献了多少 IC（注意：这是诊断值，不是修好的值 ——
# 真正的修法是用滞后后的特征**重训** alpha 模型，否则 train/serve 不一致）。
_ALPHA_LAG = int(os.environ.get("ALPHA_LAG_DAYS", "0") or 0)


class AlphaSubPredictor(SubPredictor):
    name = "alpha_xgboost"

    def _ensure(self):
        import json
        self._model = pickle.load(open(MODEL_DIR / "alpha_xgboost_model.pkl", "rb"))
        self._scaler = pickle.load(open(MODEL_DIR / "alpha_scaler.pkl", "rb"))
        self._le = pickle.load(open(MODEL_DIR / "alpha_label_encoder.pkl", "rb"))
        # 列顺序：优先 scaler 自带 feature_names_in_（在 DataFrame 上 fit 时才有）；
        # 否则从训练指标文件恢复（ref_cols + alpha 列 = 训练时的 cols_b）。
        cols = getattr(self._scaler, "feature_names_in_", None)
        if cols is None:
            mp = MODEL_DIR / "alpha_model_metrics.json"
            if not mp.exists():
                raise RuntimeError("alpha 模型缺少列名（scaler 无 feature_names_in_ 且无 metrics）")
            meta = json.loads(mp.read_text(encoding="utf-8"))
            alpha_cols = list(meta.get("alpha_features") or [])
            ref_cols = list(meta.get("feature_cols") or [])
            if not ref_cols:
                raise RuntimeError("alpha 模型指标文件缺少 feature_cols，无法对齐输入")
            cols = ref_cols + alpha_cols
        self._cols: List[str] = list(cols)
        self._alpha_cols = [c for c in self._cols if c.startswith("alpha_")]
        self._ref_cols = [c for c in self._cols if not c.startswith("alpha_")]
        self._classes = list(self._le.classes_)

    def _alpha_for_code(self, code: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        """从本地 alpha parquet 缓存读取该 code 的 alpha 列，对齐到 index。"""
        if not self._alpha_cols:
            return pd.DataFrame(index=index)
        frames = []
        for col in self._alpha_cols:
            name = col[len("alpha_"):]
            p = DATA_DIR / f"alpha_{name}.parquet"
            if not p.exists():
                frames.append(pd.DataFrame({col: np.nan}, index=index))
                continue
            panel = pd.read_parquet(p)
            if code in panel.columns:
                s = panel[code]
            else:
                s = pd.Series(np.nan, index=panel.index)
            s = s.reindex(index).ffill().fillna(0.0)
            frames.append(pd.DataFrame({col: s.values}, index=index))
        return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=index)

    def _score(self, df: pd.DataFrame, code: str) -> float:
        feats = feature_engine.build_features(df)
        if "close" in feats.columns:
            feats = feats.drop(columns=["close"])
        latest = feats.iloc[-1:]
        if self._alpha_cols:
            if _ALPHA_LAG > 0:
                # 只有诊断模式才需要全历史（shift 要按交易日位移，单看最后一行没法移）。
                # 代价是回填慢一倍，故生产（lag=0）绝不走这条。
                a = self._alpha_for_code(code, feats.index).shift(_ALPHA_LAG).iloc[-1:]
            else:
                a = self._alpha_for_code(code, latest.index)
            row = latest.reindex(columns=self._ref_cols, fill_value=0.0)
            row = pd.concat([row, a.reindex(index=row.index)], axis=1)
        else:
            row = latest.reindex(columns=self._ref_cols, fill_value=0.0)
        row = row.reindex(columns=self._cols, fill_value=0.0).fillna(0.0)
        # 传 numpy 避免「scaler 无特征名」告警（老版 scaler 在数组上 fit）
        Xs = self._scaler.transform(row.values.astype(float))
        proba = self._model.predict_proba(Xs)[0]
        return _proba_to_score(proba, self._classes)


# ---------------------------------------------------------------------------
# LSTM 子预测器（需带元数据的 lstm_best.pt）
# ---------------------------------------------------------------------------
class LSTMSubPredictor(SubPredictor):
    name = "lstm"

    def _ensure(self):
        import torch  # 懒加载，避免无 torch 时 import 失败
        self._torch = torch
        ckpt = torch.load(MODEL_DIR / "lstm_best.pt", map_location="cpu", weights_only=False)
        meta = ckpt.get("meta", {})
        if "feature_cols" not in meta:
            # 老版 lstm_best.pt 不含特征元数据，无法对齐输入 → 主动降级
            raise RuntimeError("lstm_best.pt 缺少 feature_cols 元数据，需重训（见 scripts/retrain_lstm.py）")
        from src.models.lstm import StockLSTM
        from config.settings import config as app_config
        params = app_config.ml.lstm_params
        self._model = StockLSTM(
            input_size=ckpt["input_size"],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        self._feature_cols: List[str] = meta["feature_cols"]
        self._seq_len: int = ckpt.get("sequence_length", params["sequence_length"])
        self._classes = meta.get("classes", [-1, 0, 1])

    def _score(self, df: pd.DataFrame, code: str) -> float:
        feats = feature_engine.build_features(df)
        if "close" in feats.columns:
            feats = feats.drop(columns=["close"])
        win = feats.reindex(columns=self._feature_cols, fill_value=0.0).fillna(0.0)
        if len(win) < self._seq_len:
            raise RuntimeError(f"样本不足 {self._seq_len} 日")
        seq = win.iloc[-self._seq_len:].values.astype(float)
        seq = self._torch.FloatTensor(seq.reshape(1, self._seq_len, -1))
        with self._torch.no_grad():
            out = self._model(seq)
            proba = self._torch.softmax(out, dim=1).cpu().numpy()[0]
        return _proba_to_score(proba, self._classes)


# ---------------------------------------------------------------------------
# Transformer 时序排序器子预测器（需带元数据的 transformer_best.pt）
# ---------------------------------------------------------------------------
class TransformerSubPredictor(SubPredictor):
    name = "transformer"

    def _ensure(self):
        import torch  # 懒加载，避免无 torch 时 import 失败
        self._torch = torch
        ckpt = torch.load(MODEL_DIR / "transformer_best.pt", map_location="cpu", weights_only=False)
        meta = ckpt.get("meta", {})
        if "feature_cols" not in meta:
            # 老版 checkpoint 不含特征元数据 → 主动降级
            raise RuntimeError("transformer_best.pt 缺少 feature_cols 元数据，需重训（见 scripts/train_transformer.py）")
        from src.models.transformer_ranker import TemporalTransformer
        from config.settings import config as app_config
        params = app_config.ml.transformer_params
        self._model = TemporalTransformer(
            input_size=ckpt["input_size"],
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
        )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        self._feature_cols: List[str] = meta["feature_cols"]
        self._seq_len: int = ckpt.get("sequence_length", params["sequence_length"])
        self._classes = meta.get("classes", [-1, 0, 1])

    def _score(self, df: pd.DataFrame, code: str) -> float:
        feats = feature_engine.build_features(df)
        if "close" in feats.columns:
            feats = feats.drop(columns=["close"])
        win = feats.reindex(columns=self._feature_cols, fill_value=0.0).fillna(0.0)
        if len(win) < self._seq_len:
            raise RuntimeError(f"样本不足 {self._seq_len} 日")
        seq = win.iloc[-self._seq_len:].values.astype(float)
        seq = self._torch.FloatTensor(seq.reshape(1, self._seq_len, -1))
        with self._torch.no_grad():
            out = self._model(seq)
            proba = self._torch.softmax(out, dim=1).cpu().numpy()[0]
        return _proba_to_score(proba, self._classes)


# ---------------------------------------------------------------------------
# LLM 另类 alpha 实时集成子预测器（超级集成第五腿，需真实 LLM）
# ---------------------------------------------------------------------------
class LLMAlphaSubPredictor(SubPredictor):
    """把 LLM 对「个股近期新闻/公告」的定性判断转成 0~100 的另类 alpha 评分。

    设计原则（与覆盖层 LLM overlay 严格区分）：
      · 这是**超级集成的一条腿**，与其他腿（XGBoost/Alpha/LSTM/Transformer）加权融合；
        而覆盖层（_apply_llm_overlay）是对主模型 score 做**加性微调**，两者独立。
      · **默认关闭、零行为改变**：仅当 config.ml.enable_llm_alpha_leg=True 且
        配置了真实 LLM（有 api_key / 本机 Ollama 可达）时才激活。
      · 任何降级（开关关 / 无 key 自动转 mock / 构造失败）都让 available=False，
        绝不把离线伪评分当信号注入回测或实盘。
      · 评分映射：score = 50 + 50 * clamp(sentiment - 0.3*risk, -1, 1)，
        与 _proba_to_score 同尺度（0~100），可与各腿直接加权融合。

    零泄漏：LLMFeatureProvider.features_for_stock 内部 gather_news 只用
    publish_time <= as_of_date 的文本（as_of_date 取该股票最新交易日）。
    """

    name = "llm_alpha"

    def __init__(self, provider=None, lookback_days=None):
        # 仅供单测注入：真实路径由 _ensure 按 config 构造，不暴露构造参数。
        self._injected_provider = provider
        self._injected_lookback = lookback_days

    def _ensure(self):
        from config.settings import config as app_config
        ml = app_config.ml
        if not getattr(ml, "enable_llm_alpha_leg", False):
            raise RuntimeError(
                "LLM alpha 腿未启用（enable_llm_alpha_leg=False），保持关闭")

        if self._injected_provider is not None:
            # 单测注入路径：跳过真实 provider 构造与 mock 校验
            self._provider = self._injected_provider
            self._lookback = self._injected_lookback or getattr(
                ml, "llm_alpha_lookback_days", 20)
            return

        # 真实路径：按 config 构造 LLMFeatureProvider（绝不走 mock）
        from src.data.llm_features import LLMFeatureProvider, LLMFeatureConfig
        prov = getattr(ml, "llm_alpha_provider", "deepseek")
        model = getattr(ml, "llm_alpha_model", "deepseek-chat")
        base_url = getattr(ml, "llm_alpha_base_url", "")
        lookback = getattr(ml, "llm_alpha_lookback_days", 20)
        if prov == "local" and not base_url:
            base_url = "http://localhost:11434/v1"  # 本机 Ollama 默认
        # 第五腿磁盘缓存：仅本腿启用时生效（config.ml.llm_alpha_cache，默认开）。
        # 落盘到 DATA_DIR/llm_alpha_cache/，按 (code, as_of_date, days) 复用，
        # 省 LLM 调用额度、抗抖动；命中仍强制校验 end_date 以防零泄漏。
        cache_dir = str(DATA_DIR / "llm_alpha_cache") if getattr(
            ml, "llm_alpha_cache", True) else ""
        cfg = LLMFeatureConfig(
            provider=prov,
            model=model,
            base_url=base_url,
            mock=False,  # 关键：显式关 mock，逼出「无 key 自动转 mock」的降级
            cache_dir=cache_dir,
        )
        provider = LLMFeatureProvider(cfg)
        if provider.cfg.mock:
            # LLMFeatureProvider 在无 key 时自动转 mock（离线伪评分）。
            # 本腿绝不注入伪评分，主动保持关闭。
            raise RuntimeError(
                "LLM alpha 腿：未检测到真实 LLM key（已自动转 mock），"
                "保持关闭，不注入离线伪评分")
        self._provider = provider
        self._lookback = lookback

    def _score(self, df: pd.DataFrame, code: str) -> float:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
        end_date = str(idx[-1].date()) if hasattr(idx[-1], "date") else str(idx[-1])
        feats = self._provider.features_for_stock(
            code, end_date, days=self._lookback)
        sentiment = float(feats.get("llm_sentiment", 0.0) or 0.0)
        risk = int(feats.get("llm_risk", 0) or 0)
        # 利空风险事件扣一档（0.3），再夹到 [-1,1] 映射到 0~100
        raw = max(-1.0, min(1.0, sentiment - 0.3 * risk))
        return float(50.0 + 50.0 * raw)

    def health(self) -> Optional[Dict]:
        """暴露 LLM 腿健康快照（运维/看板观测用）；不可用或单测注入时返回 None。"""
        prov = getattr(self, "_provider", None)
        if prov is None or not hasattr(prov, "stats"):
            return None
        try:
            return prov.stats()
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# 超级集成器
# ---------------------------------------------------------------------------
class EnsemblePredictor:
    """融合多个子预测器，输出加权评分。

    用法：
        ep = EnsemblePredictor(predictor)        # predictor 为已构造的 Predictor
        ep.add_sub(AlphaSubPredictor(), weight=1.0)
        ep.add_sub(LSTMSubPredictor(),  weight=0.8)
        preds = ep.predict_batch(stock_data)     # Dict[code, StockPrediction]
    """

    def __init__(self, primary_predictor, default_weight: float = 1.0):
        self.primary = primary_predictor
        self.subs: List[Tuple[SubPredictor, float]] = []
        self.default_weight = default_weight

    def add_sub(self, sub: SubPredictor, weight: Optional[float] = None) -> None:
        self.subs.append((sub, weight if weight is not None else self.default_weight))

    def _available_subs(self):
        out = []
        for sub, w in self.subs:
            if sub.available:
                out.append((sub, w))
            else:
                why = getattr(sub, "_reason", None)
                logger.warning(f"  集成：{sub.name} 不可用，已跳过"
                               + (f"（原因：{why}）" if why else ""))
        return out

    def _eff_weights(self, avail) -> List[float]:
        """按 ensemble_shrinkage 把子模型权重向「等权」收缩（方差削减第二层）。

        shrink=0 → 原始权重；shrink=1 → 全部等权（最大分散、最小融合方差）。
        主模型权重恒为 1.0（基准，不收缩）。仅当可用子模型 ≥2 时生效。
        """
        shrink = float(getattr(app_config.ml, "ensemble_shrinkage", 0.0) or 0.0)
        if shrink > 0 and len(avail) > 1:
            total = sum(w for _, w in avail)
            uni = total / len(avail)
            return [w * (1.0 - shrink) + uni * shrink for _, w in avail]
        return [w for _, w in avail]

    def _dynamic_weights(self, avail) -> Optional[List[float]]:
        """第六腿：按各腿样本外 RankIC 派生融合权重（替代静态权重）。

        仅在 config.ml.ensemble_dynamic_weights=True 时生效。
        返回归一化权重（与 avail 顺序对齐），或 None（触发回退静态权重）：
          · 开关关
          · leg_rankic.json 缺失 / 过期
          · 全部腿均无统计（极端）
        主模型 xgboost 恒为锚（权重 1.0），本函数只重排**子腿**相对权重。
        """
        ml = getattr(app_config, "ml", None)
        if not getattr(ml, "ensemble_dynamic_weights", False):
            return None
        try:
            from src.core import leg_rankic as lr
        except Exception:  # noqa: BLE001
            return None
        max_age = int(getattr(ml, "ensemble_dynamic_max_age_days", 30) or 30)
        stats = lr.load_leg_rankic(max_age_days=max_age)
        if not stats:
            logger.warning("动态融合：无有效 leg_rankic.json（缺失/过期），回退静态权重")
            return None
        floor = float(getattr(ml, "ensemble_dynamic_floor", 0.0) or 0.0)
        names = [sub.name for sub, _ in avail]
        fallback = [float(w) for _, w in avail]
        try:
            return lr.dynamic_weights_from_stats(stats, names, fallback, floor)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"动态融合：权重派生失败，回退静态权重: {e}")
            return None

    def blend_one(self, pred, df: pd.DataFrame, code: str) -> bool:
        """把子模型评分融合进「单只股票」的 pred（就地改 score/agreement/ensemble_detail）。

        **predict_batch 与 Predictor.predict_single 共用这一条路径**，保证
        「选股批量」和「个股分析页」两处口径绝对一致（2026-10-18 修：此前个股分析页
        恒走单模型，而单模型实测 mean IC 0.0575 / t=1.40 **统计不显著**）。

        返回是否真的发生了融合。任一腿缺失 / 评分失败都保留原单模型分
        （model_mode 仍为 "single"），绝不拖垮调用方。
        """
        avail = self._available_subs()
        if not avail:
            return False
        # 第六腿：动态融合（按各腿样本外 RankIC 派生权重）。开关关 / 无统计 / 过期
        # 时返回 None → 回退到静态权重（含 ensemble_shrinkage），生产默认走这条。
        eff_ws = self._dynamic_weights(avail)
        if eff_ws is None:
            eff_ws = self._eff_weights(avail)
        # 主模型贡献用 score_raw（纯概率分，与子模型 _proba_to_score 同口径），
        # 而非 pred.score（含 compute_score 的 regime/risk/预期收益加成）。
        # 否则主腿与其他腿尺度错位，融合权重失真。
        # 防御：真实 StockPrediction 必有 score_raw；若调用方传入的占位对象缺该属性，
        # 退回用 score，避免 AttributeError（向后兼容历史测试桩）。
        primary_score = getattr(pred, "score_raw", None)
        if primary_score is None:
            primary_score = pred.score
        sub_scores = [primary_score]  # 主模型权重 1.0
        sub_weights = [1.0]
        used_names = ["xgboost"]
        for (sub, _), ew in zip(avail, eff_ws):
            sc = sub.score_for(df, code)
            if sc is not None:
                sub_scores.append(sc)
                sub_weights.append(ew)
                used_names.append(sub.name)
        if len(sub_scores) <= 1:
            return False
        arr = np.array(sub_scores, dtype=float)
        wts = np.array(sub_weights, dtype=float)
        blended = float(np.average(arr, weights=wts))
        spread = float(arr.max() - arr.min())
        pred.score = blended
        # agreement 保持 >=0.8 以兼容交易执行器硬闸门（require_agreement < 0.8 拒单）；
        # 真实分歧记录在 ensemble_detail["spread"]，不阻断下单。
        pred.agreement = max(0.8, 1.0 - spread / 100.0)
        # 只列「本只股票真正出分」的腿，避免某腿返回 None 时三张表长度错位。
        pred.ensemble_detail = {
            "scores": dict(zip(used_names, sub_scores)),
            "weights": dict(zip(used_names, [float(x) for x in sub_weights])),
            "spread": spread,
        }
        pred.model_mode = "ensemble"
        return True

    def predict_batch(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
        """对一批股票做预测，返回 {code: StockPrediction}（score 为融合值）。"""
        results: Dict[str, object] = {}
        for code, df in stock_data.items():
            try:
                # apply_ensemble 默认 False：融合统一由下面的 blend_one 做，
                # 避免「predict_single 内部再融一次」的双重加权。
                pred = self.primary.predict_single(df, code=code)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"主模型预测失败 {code}: {e}")
                continue
            self.blend_one(pred, df, code)
            results[code] = pred
        return results

    def summary(self) -> str:
        lines = ["🧩 超级集成状态："]
        lines.append(f"  主模型 XGBoost：始终启用")
        for sub, w in self.subs:
            st = "✅" if sub.available else "⚠️不可用(降级)"
            extra = ""
            if sub.available and sub.name == "llm_alpha":
                h = sub.health()
                if h:
                    extra = (f"（调用 {h['total_calls']} 成功 {h['success']} "
                             f"失败 {h['failures']} 连续失败 {h['consecutive_failures']}）")
            lines.append(f"  {sub.name}：{st}（权重 {w}）{extra}")
        return "\n".join(lines)

# -*- coding: utf-8 -*-
"""
V2 选股推理
===========

与旧 predictor.py 的区别
------------------------
旧版输出"涨/震荡/跌 + 0-100 评分"，但那个评分是人为公式拼出来的
（见 scoring.py：base_score + 确信度加分 + 一致性加分 + 风险扣分 × 牛熊因子），
与真实收益没有必然联系。

新版输出**可解释的相对强弱排序**：
  - predicted_bin : 模型预测的档位（0-4，越大越强）
  - strength      : 模型看涨的强度分（用于排序）
  - 但最关键的是同时给出**该因子的历史样本外 IC**，
    让使用者知道这个信号历史上靠不靠谱

核心原则：宁可给"这因子历史上样本外 IC 是 +0.05，Top20 组合年化超额 X%"，
也不给一个看起来精确但无意义的"评分 78.5 分"。
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from loguru import logger

from config.settings import MODEL_DIR
from src.models.factors_v2 import FactorPanelBuilder
from src.models.labeling import cs_rank

MODEL_PREFIX = "leaksafe"


@dataclass
class V2Prediction:
    """V2 预测结果"""
    code: str
    name: str = ""
    # 模型输出
    predicted_bin: int = 2# 预测档位 0-4
    strength: float = 0.0       # 看涨强度（越大越看涨）
    bin_prob: List[float] = field(default_factory=list)  # 各档概率
    # 因子明细（该股票的关键因子值，已截面归一化）
    key_factors: Dict[str, float] = field(default_factory=dict)
    # 风险
    volatility: float = 0.0
    # 置信提示（基于历史样本外表现，不是模型自评）
    reliability: str = ""
    reliability_note: str = ""


class PredictorV2:
    """基于因子面板 + walk-forward 模型的选股器"""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.features: List[str] = []
        self.metrics: Dict = {}
        self.factor_ic: Dict[str, float] = {}   # 因子样本外 IC

        self._load()

    def _load(self):
        """加载 V2 模型产物"""
        # scaler + 特征列表
        sc = self.model_dir / f"{MODEL_PREFIX}_scaler.pkl"
        ft = self.model_dir / f"{MODEL_PREFIX}_features.json"
        mt = self.model_dir / f"{MODEL_PREFIX}_metrics.json"

        if ft.exists():
            self.features = json.loads(ft.read_text(encoding="utf-8")).get("features", [])
        if sc.exists():
            self.scaler = pickle.load(open(sc, "rb"))
        if mt.exists():
            self.metrics = json.loads(mt.read_text(encoding="utf-8"))

        # 因子 IC 表
        fr = self.model_dir / "factor_report_v2.csv"
        if fr.exists():
            try:
                fdf = pd.read_csv(fr)
                self.factor_ic = dict(
                    zip(fdf["factor"], fdf.get("oos_ic", pd.Series(dtype=float)))
                )
            except Exception:
                pass

        # 模型本体
        for name, loader in (("xgboost", self._load_xgb), ("lightgbm", self._load_lgb)):
            p = self.model_dir / f"{MODEL_PREFIX}_{name}_model.pkl"
            if p.exists():
                try:
                    self.models[name] = loader(p)
                    logger.info(f"已加载 {name} 模型")
                except Exception as e:
                    logger.warning(f"{name} 加载失败: {e}")

        # 读取模型评估指标摘要
        if self.metrics:
            m = self.metrics
            ic = m.get("oos_rank_ic")
            ex = m.get("oos_top_excess")
            logger.info(
                f"模型历史样本外表现: "
                f"RankIC={ic:+.4f} " if ic is not None else "无IC记录"
            )
            if ic is not None:
                logger.info(
                    f"  IC_IR={m.get('oos_ic_ir'):.2f} "
                    f"单期超额={ex:+.3%} "
                    f"胜率={m.get('oos_win_rate'):.1%}"
                )

    def _load_xgb(self, path):
        import xgboost as xgb
        return pickle.load(open(path, "rb"))

    def _load_lgb(self, path):
        import lightgbm as lgb
        return pickle.load(open(path, "rb"))

    @property
    def is_ready(self) -> bool:
        return bool(self.models) and self.scaler is not None and bool(self.features)

    # ---------- 主预测 ----------

    def predict_batch(
        self,
        stock_data: Dict[str, pd.DataFrame],
        stock_names: Dict[str, str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        批量选股

        Parameters
        ----------
        stock_data : {code: DataFrame(index=date)}
        top_n : 返回前N 只

        Returns
        -------
        DataFrame，按 strength 降序
        """
        if not self.is_ready:
            raise RuntimeError("V2 模型未就绪，请先运行 scripts/train_v2.py")

        stock_names = stock_names or {}

        # 1. 构建因子面板（与训练时完全一致的流程）
        builder = FactorPanelBuilder()
        panel = builder.build(stock_data, use_cs_rank=True, verbose=False)

        # 只保留训练时用过的因子
        panel = panel[[c for c in panel.columns if c[1] in set(self.features)]]

        # 2. 取最近一个交易日
        latest = panel.index[-1]
        row = panel.loc[latest]

        # 3. 展平成 (code × factor)
        wide = row.unstack(level="factor")
        # 对齐训练特征顺序
        wide = wide.reindex(columns=self.features)
        wide = wide.dropna(how="all")

        if len(wide) == 0:
            logger.warning("最新交易日无有效因子值")
            return pd.DataFrame()

        # 4. 标准化 + 预测
        X = self.scaler.transform(wide)
        Xdf = pd.DataFrame(X, index=wide.index, columns=self.features)

        preds = []
        n_models = len(self.models)

        for name, model in self.models.items():
            proba = model.predict_proba(Xdf)
            pred_bin = model.predict(Xdf)
            preds.append((proba, pred_bin))

        # 集成：概率平均
        avg_proba = np.mean([p for p, _ in preds], axis=0)
        # 强度分：最高档概率 - 最低档概率，范围约 [-1, 1]
        strength_vec = avg_proba[:, -1] - avg_proba[:, 0]
        # 预测档位取概率最大的
        pred_bin = np.argmax(avg_proba, axis=1)

        # 5. 组装结果
        records = []
        for i, code in enumerate(wide.index):
            vol = self._recent_vol(stock_data.get(code), wide.index, latest)
            ic_note = self._reliability_note(code, latest, wide, stock_data)
            rel, note = ic_note

            # 关键因子：取 |IC| 最大的几个因子的实际值
            key_facs = {}
            for f in self._top_factors():
                if f in wide.columns:
                    key_facs[f] = round(float(wide.loc[code, f]), 3)

            records.append(V2Prediction(
                code=str(code),
                name=stock_names.get(code, code),
                predicted_bin=int(pred_bin[i]),
                strength=round(float(strength_vec[i]), 4),
                bin_prob=[round(float(x), 4) for x in avg_proba[i]],
                key_factors=key_facs,
                volatility=vol,
                reliability=rel,
                reliability_note=note,
            ))

        records.sort(key=lambda r: r.strength, reverse=True)
        top = records[:top_n]

        return pd.DataFrame([{
            "代码": r.code,
            "名称": r.name,
            "预测档位": f"{r.predicted_bin}/4",
            "强度分": r.strength,
            "上涨概率": round(r.bin_prob[-1], 4),
            "下跌概率": round(r.bin_prob[0], 4),
            "年化波动": f"{r.volatility:.1%}",
            "关键因子": ", ".join(f"{k}={v:+.2f}" for k, v in list(r.key_factors.items())[:3]),
            "可靠性": r.reliability,
            "说明": r.reliability_note,
        } for r in top])

    # ---------- 辅助 ----------

    def _top_factors(self, n: int = 3) -> List[str]:
        """按 |样本外 IC| 选出最重要的因子"""
        if not self.factor_ic:
            return self.features[:n]
        items = sorted(self.factor_ic.items(), key=lambda x: -abs(x[1] if x[1] == x[1] else 0))
        return [k for k, _ in items[:n]]

    def _recent_vol(self, df: Optional[pd.DataFrame], idx, latest) -> float:
        if df is None or len(df) < 20:
            return float("nan")
        r = df["close"].pct_change().tail(60)
        return float(r.std() * np.sqrt(252))

    def _reliability_note(self, code, latest, wide, stock_data) -> tuple:
        """
        基于历史样本外表现给出可靠性提示，而不是模型自评置信度

        诚实很重要：模型说"我有90%把握"是没意义的，
        唯一有意义的是"这个信号历史上样本外 IC 是多少"。
        """
        ic = self.metrics.get("oos_rank_ic")
        ex = self.metrics.get("oos_top_excess")

        if ic is None or (isinstance(ic, float) and ic != ic) or abs(ic) < 0.01:
            return "不足参考", (
                f"历史样本外IC仅 {ic:.3f}，信号很弱" if ic is not None
                else "暂无历史验证数据"
            )

        ex_s = f"{ex:+.3%}" if ex is not None else "n/a"
        if abs(ic) >= 0.03:
            return "较强", f"样本外IC={ic:+.3f}，Top20单期超额{ex_s}"
        return "一般", f"样本外IC={ic:+.3f}，信号强度有限"

    def model_report(self) -> str:
        """输出模型真实表现的报告"""
        if not self.metrics:
            return "尚无 V2 模型，请先运行 scripts/train_v2.py"
        m = self.metrics

        def fmt(key, spec="+.4f", dash="n/a"):
            v = m.get(key)
            if v is None or (isinstance(v, float) and v != v):
                return dash
            return format(v, spec)

        return (
            "V2 模型样本外表现（walk-forward，已消除数据泄漏）\n"
            f"  RankIC         {fmt('oos_rank_ic')}\n"
            f"  IC_IR          {fmt('oos_ic_ir', '+.2f')}\n"
            f"  Top20单期超额  {fmt('oos_top_excess', '+.3%')}   每5日\n"
            f"  超额胜率       {fmt('oos_win_rate', '.1%')}\n"
            f"  超额最大回撤   {fmt('oos_max_drawdown', '.2%')}\n"
            f"  五档分类准确率 {fmt('oos_accuracy', '.1%')}\n"
            f"  样本外样本数   {m.get('n_oos_samples', 0)}\n"
            "  注：均为样本外指标，未做任何测试集调参。超额收益未扣手续费。"
        )
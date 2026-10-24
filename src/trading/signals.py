# -*- coding: utf-8 -*-
"""
量化策略信号适配器
将 QuantEngine / MeanReversion / RiskParity 策略输出
转换为 AutoTrader 可消费的统一信号格式
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

import numpy as np
import pandas as pd


class StrategyMode(str, Enum):
    AI_ONLY = "ai"              # 纯 AI 预测
    QUANT_MULTI = "multi"       # 多因子量化
    QUANT_MR = "mean_rev"       # 均值回归
    HYBRID = "hybrid"           # AI+量化融合
    ENSEMBLE = "ensemble"       # 多策略投票融合（所有策略同时投票）


@dataclass
class QuantSignal:
    """统一量化交易信号"""
    code: str
    name: str
    score: float                # 综合评分 0-100
    agreement: float            # 置信度 0-1
    signal: int                 # 1=买入, -1=卖出, 0=无信号
    prob_up: float = 0.0
    prob_neutral: float = 0.0
    prob_down: float = 0.0
    xgb_signal: int = 0         # 兼容 AI 格式
    source: str = ""
    strategy: str = ""          # 具体策略名
    reasons: List[str] = field(default_factory=list)
    quant_details: dict = field(default_factory=dict)  # 量化细节


class QuantSignalGenerator:
    """量化策略信号生成器
    
    从 watchlist 拉取行情，跑量化策略，输出 AutoTrader 兼容的信号字典
    """
    
    def __init__(self, mode: StrategyMode = StrategyMode.HYBRID):
        self.mode = mode
        self._quant = None       # QuantEngine (lazy)
        self._fetcher = None     # DataFetcher (lazy)
        self._predictor = None   # AI predictor (lazy)
        
        # 策略参数
        self.multi_min_score: float = 55.0     # 多因子最低综合分
        self.mr_lookback: int = 20             # 均值回归回看周期
        self.mr_entry_z: float = 2.0           # 均值回归入场Z-score
        self.hybrid_ai_weight: float = 0.5     # 混合模式 AI 权重
        self.max_candidates: int = 5           # 单次最多候选
        
        # 自选股列表
        self._watchlist: List[str] = []
        self._name_map: Dict[str, str] = {}
    
    @property
    def quant(self):
        if self._quant is None:
            from src.models.quantitative import quant_engine, MeanReversionStrategy
            self._quant = quant_engine
            self._mean_rev = MeanReversionStrategy(
                lookback=self.mr_lookback,
                entry_threshold=self.mr_entry_z
            )
        return self._quant
    
    @property
    def fetcher(self):
        if self._fetcher is None:
            from src.data.fetcher import fetcher
            self._fetcher = fetcher
        return self._fetcher
    
    @property
    def predictor(self):
        if self._predictor is None:
            from src.models.predictor import predictor as ai_pred
            self._predictor = ai_pred
        return self._predictor
    
    # ========== Watchlist 管理 ==========
    
    def load_watchlist(self, extra_codes: List[str] = None) -> List[str]:
        """加载自选股列表"""
        codes = list(extra_codes or [])
        
        try:
            config_file = Path(__file__).parent.parent.parent / "config" / "watchlist.json"
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                wl = cfg.get("watchlist", [])
                codes = list(set(codes + wl))
        except Exception:
            pass
        
        # 构建 code→name 映射
        try:
            stock_list = self.fetcher.get_stock_list()
            self._name_map = dict(zip(stock_list["code"], stock_list["name"]))
            for c in codes:
                if c not in self._name_map:
                    self._name_map[c] = c
        except Exception:
            self._name_map = {c: c for c in codes}
        
        self._watchlist = codes
        logger.info(f"📋 量化信号: 已加载 {len(codes)} 只自选股")
        return codes
    
    def get_name(self, code: str) -> str:
        return self._name_map.get(code, code)
    
    # ========== 信号生成 ==========
    
    def generate(self, code: str) -> Optional[QuantSignal]:
        """为单只股票生成量化交易信号
        
        Returns:
            QuantSignal 或 None (数据不足/异常)
        """
        try:
            df = self.fetcher.get_daily_kline(code)
        except Exception as e:
            logger.debug(f"获取 {code} 行情失败: {e}")
            return None
        
        if len(df) < 60:
            return None
        
        name = self.get_name(code)
        
        if self.mode == StrategyMode.QUANT_MULTI:
            return self._multi_factor_signal(df, code, name)
        elif self.mode == StrategyMode.QUANT_MR:
            return self._mean_rev_signal(df, code, name)
        elif self.mode == StrategyMode.HYBRID:
            return self._hybrid_signal(df, code, name)
        elif self.mode == StrategyMode.ENSEMBLE:
            return self.generate_ensemble(df, code, name)
        else:
            # AI_ONLY — 不走这里，由 AutoTrader 的 ai_predict_fn 处理
            return None

    def generate_ensemble(
        self, df: pd.DataFrame, code: str, name: str
    ) -> Optional[QuantSignal]:
        """多策略投票融合：所有策略同时投票，方向一致才下手。

        为什么不是「选一个策略」
        ----------------------
        单策略在不同市场状态下轮流失效：趋势市多因子强、震荡市均值回归强、
        AI 融合需要量价配合。固定选一个，等于把赌注压在「当前是哪种市」上。
        让它们同时投票、只在方向一致时下手，能天然过滤单一策略的假信号，
        并把一致度作为置信度交给下游调仓。
        """
        legs: List[QuantSignal] = []
        for fn, tag in (
            (self._multi_factor_signal, "multi_factor"),
            (self._mean_rev_signal, "mean_rev"),
            (self._hybrid_signal, "hybrid"),
        ):
            try:
                s = fn(df, code, name)
            except Exception as e:  # noqa: BLE001
                # 单个策略挂掉不该让整只票失去机会（AI 腿尤其容易缺模型/数据）
                logger.debug(f"策略 {tag} 在 {code} 上失败：{e}")
                continue
            if s is None or s.signal == 0:
                continue
            s.strategy = tag
            legs.append(s)

        if not legs:
            return None

        buy = sum(1 for s in legs if s.signal == 1)
        sell = sum(1 for s in legs if s.signal == -1)
        if buy == sell:
            # 完全分歧 → 不下手。分歧本身就是不确定性的信号，
            # 硬选一个方向只是在用掷骰子代替判断。
            return None

        direction = 1 if buy > sell else -1
        aligned = [s for s in legs if s.signal == direction]

        # 按各自置信度加权平均得分，避免弱信号的票靠数量取胜
        weights = [max(float(s.agreement or 0.0), 1e-6) for s in aligned]
        wsum = sum(weights)
        score = sum(s.score * w for s, w in zip(aligned, weights)) / wsum
        agreement = len(aligned) / len(legs)

        reasons: List[str] = []
        for s in aligned:
            reasons.extend(f"[{s.strategy}] {r}" for r in s.reasons[:2])
        reasons.append(f"多策略投票 {len(aligned)}/{len(legs)}（买{buy}/卖{sell}）")

        return QuantSignal(
            code=code,
            name=name,
            score=round(float(score), 2),
            agreement=round(agreement, 3),
            signal=direction,
            prob_up=sum(s.prob_up for s in aligned) / len(aligned),
            prob_neutral=sum(s.prob_neutral for s in aligned) / len(aligned),
            prob_down=sum(s.prob_down for s in aligned) / len(aligned),
            xgb_signal=next((s.xgb_signal for s in aligned if s.xgb_signal), 0),
            source="ensemble",
            strategy="+".join(s.strategy for s in aligned),
            reasons=reasons,
            quant_details={
                "votes": {"buy": buy, "sell": sell, "total": len(legs)},
                "legs": [s.strategy for s in legs],
                "leg_scores": {s.strategy: s.score for s in legs},
            },
        )

    def generate_all(self) -> List[QuantSignal]:
        """批量生成所有自选股的信号"""
        if not self._watchlist:
            self.load_watchlist()
        
        signals = []
        for code in self._watchlist:
            sig = self.generate(code)
            if sig and sig.signal != 0:
                signals.append(sig)
        
        # 按评分排序
        signals.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"📊 量化扫描完成: {len(signals)} 个信号 (共 {len(self._watchlist)} 只)")
        return signals
    
    # ========== 多因子策略 ==========
    
    def _multi_factor_signal(self, df: pd.DataFrame, code: str, name: str) -> Optional[QuantSignal]:
        """多因子量化评分 → 交易信号"""
        qs = self.quant.score_stock(df, code, name)
        
        if qs.total_score <= 0:
            return None
        
        # 信号方向
        if qs.total_score >= self.multi_min_score:
            signal_dir = 1  # 买入
        elif qs.total_score < 30:
            signal_dir = -1  # 卖出
        else:
            signal_dir = 0  # 无信号
        
        # 置信度 = 评分映射
        confidence = min(1.0, qs.total_score / 100.0)
        
        # 计算虚拟概率
        prob_up = qs.total_score / 100.0
        prob_down = (100 - qs.total_score) / 100.0
        prob_neutral = 0.0
        
        reasons = []
        if qs.value_score >= 60:
            reasons.append(f"价值因子 {qs.value_score:.0f}")
        if qs.momentum_score >= 55:
            reasons.append(f"动量因子 {qs.momentum_score:.0f}")
        if qs.quality_score >= 70:
            reasons.append(f"质量因子 {qs.quality_score:.0f}")
        if qs.trend_score >= 40:
            reasons.append(f"趋势强度 {qs.trend_score:.0f}")
        if qs.reversal_score > 0:
            reasons.append(f"反转信号 {qs.reversal_score:.0f}")
        
        return QuantSignal(
            code=code, name=name,
            score=qs.total_score,
            agreement=confidence,
            signal=signal_dir,
            prob_up=round(prob_up * 100, 1),
            prob_neutral=round(prob_neutral * 100, 1),
            prob_down=round(prob_down * 100, 1),
            xgb_signal=signal_dir,
            source="quant_multi",
            strategy="多因子量化",
            reasons=reasons or ["多因子综合评分"],
            quant_details={
                "value": qs.value_score,
                "momentum": qs.momentum_score,
                "quality": qs.quality_score,
                "volatility": qs.volatility_score,
                "trend": qs.trend_score,
                "reversal": qs.reversal_score,
                "sharpe": qs.sharpe,
                "max_dd": qs.max_drawdown,
                "win_rate": qs.win_rate,
                "quant_signal": qs.signal,
            }
        )
    
    # ========== 均值回归策略 ==========
    
    def _mean_rev_signal(self, df: pd.DataFrame, code: str, name: str) -> Optional[QuantSignal]:
        """均值回归策略 → 交易信号"""
        from src.models.quantitative import MeanReversionStrategy
        
        mr = MeanReversionStrategy(
            lookback=self.mr_lookback,
            entry_threshold=self.mr_entry_z
        )
        
        sig_df = mr.generate_signals(df)
        latest_signal = sig_df["signal"].iloc[-1]
        
        # 计算反转强度
        reversal_strength, rev_dir = self.quant.compute_reversal_signal(df)
        
        if latest_signal == 1 and rev_dir == "超卖反弹":
            # 超卖买入信号
            score = min(100, reversal_strength + 40)  # 基线40 + 反转强度
            confidence = min(1.0, score / 100.0)
            
            # 回测验证
            bt = mr.backtest(df, capital=100000)
            bt_alpha = bt.get("alpha", 0)
            
            return QuantSignal(
                code=code, name=name,
                score=round(score, 1),
                agreement=round(confidence, 2),
                signal=1,  # 买入
                prob_up=round(score / 100.0 * 100, 1),
                prob_neutral=round((100 - score) / 2 / 100.0 * 100, 1),
                prob_down=round((100 - score) / 2 / 100.0 * 100, 1),
                xgb_signal=1,
                source="quant_mr",
                strategy="均值回归",
                reasons=[
                    f"布林带超卖 → 回归均值" if reversal_strength > 0 else "RSI超卖反弹",
                    f"反转强度 {reversal_strength:.0f}",
                    f"回测α {bt_alpha:+.1f}%",
                ],
                quant_details={
                    "reversal_strength": reversal_strength,
                    "direction": rev_dir,
                    "z_score": float(
                        (df["close"].iloc[-1] - df["close"].rolling(20).mean().iloc[-1])
                        / (df["close"].rolling(20).std().iloc[-1] + 1e-10)
                    ),
                    "backtest_alpha": bt_alpha,
                    "backtest_trades": bt.get("n_trades", 0),
                    "backtest_maxdd": bt.get("max_drawdown", 0),
                }
            )
        
        elif latest_signal == -1 and rev_dir == "超买回落":
            # 超买卖出信号
            score = min(100, reversal_strength + 40)
            confidence = min(1.0, score / 100.0)
            
            return QuantSignal(
                code=code, name=name,
                score=round(score, 1),
                agreement=round(confidence, 2),
                signal=-1,  # 卖出
                prob_up=round((100 - score) / 2 / 100.0 * 100, 1),
                prob_neutral=round((100 - score) / 2 / 100.0 * 100, 1),
                prob_down=round(score / 100.0 * 100, 1),
                xgb_signal=-1,
                source="quant_mr",
                strategy="均值回归",
                reasons=[
                    "布林带上轨超买 → 回归均值",
                    f"反转强度 {reversal_strength:.0f}",
                ],
                quant_details={
                    "reversal_strength": reversal_strength,
                    "direction": rev_dir,
                    "z_score": float(
                        (df["close"].iloc[-1] - df["close"].rolling(20).mean().iloc[-1])
                        / (df["close"].rolling(20).std().iloc[-1] + 1e-10)
                    ),
                }
            )
        
        return None  # 无显著回归信号
    
    # ========== AI+量化融合策略 ==========
    
    def _hybrid_signal(self, df: pd.DataFrame, code: str, name: str) -> Optional[QuantSignal]:
        """AI+量化融合：两套独立评分综合决策
        
        买入条件：AI评分≥60 AND 量化评分≥50
        卖出条件：AI评分<30 AND 量化评分<30
        """
        # 1. AI 预测
        try:
            ai_pred = self.predictor.predict_single(df, code, name)
            ai_score = ai_pred.score
            ai_agreement = ai_pred.agreement
            ai_xgb = ai_pred.xgb_signal
        except Exception:
            ai_score = 50.0
            ai_agreement = 0.5
            ai_xgb = 0
        
        # 2. 量化评分
        qs = self.quant.score_stock(df, code, name)
        quant_score = qs.total_score
        
        # 3. 融合加权
        w = self.hybrid_ai_weight
        fusion_score = w * ai_score + (1 - w) * quant_score
        
        # 4. 决策逻辑
        if ai_score >= 60 and quant_score >= 50:
            signal_dir = 1  # 双确认买入
        elif ai_score < 30 and quant_score < 30:
            signal_dir = -1  # 双确认卖出
        else:
            signal_dir = 0  # 分歧 → 不交易
        
        confidence = min(1.0, fusion_score / 100.0)
        
        prob_up = fusion_score / 100.0 if signal_dir >= 0 else (100 - fusion_score) / 100.0
        prob_down = (100 - fusion_score) / 100.0 if signal_dir >= 0 else fusion_score / 100.0
        
        return QuantSignal(
            code=code, name=name,
            score=round(fusion_score, 1),
            agreement=round(confidence, 2),
            signal=signal_dir,
            prob_up=round(prob_up * 100, 1),
            prob_neutral=round((1 - abs(prob_up - prob_down) / 100) * 100, 1),
            prob_down=round(prob_down * 100, 1),
            xgb_signal=ai_xgb,
            source="hybrid",
            strategy="AI+量化融合",
            reasons=[
                f"AI评分 {ai_score:.0f} | 量化 {quant_score:.0f}",
                f"融合分 {fusion_score:.0f}",
                "AI+量化双确认" if signal_dir != 0 else "AI/量化分歧",
            ],
            quant_details={
                "ai_score": ai_score,
                "quant_score": quant_score,
                "fusion_score": fusion_score,
                "ai_weight": w,
                "ai_agreement": ai_agreement,
                "quant_signal": qs.signal,
            }
        )
    
    # ========== AI兼容接口（供 AutoTrader 使用）==========
    
    def to_predict_dict(self, sig: QuantSignal) -> dict:
        """将 QuantSignal 转换为 AutoTrader 兼容的 dict 格式"""
        return {
            "code": sig.code,
            "name": sig.name,
            "score": sig.score,
            "agreement": sig.agreement,
            "xgb_signal": sig.xgb_signal,
            "prob_up": sig.prob_up,
            "prob_down": sig.prob_down,
            "source": sig.source,
            "strategy": sig.strategy,
            "reasons": sig.reasons,
            "quant_details": sig.quant_details,
        }
    
    def make_predict_fn(self) -> callable:
        """创建一个兼容 AutoTrader._ai_predict_fn 接口的 callable
        
        Returns:
            predict_fn(code: str) → dict | None
        """
        # 预加载
        if not self._watchlist:
            self.load_watchlist()
        
        def predict(code: str) -> Optional[dict]:
            sig = self.generate(code)
            if sig is None or sig.signal == 0:
                return None
            return self.to_predict_dict(sig)
        
        return predict


# ========== 快捷工厂 ==========

def create_strategy_generator(
    mode: str = "hybrid",
    watchlist_codes: List[str] = None,
    **kwargs
) -> QuantSignalGenerator:
    """快捷创建策略信号生成器
    
    Args:
        mode: "ai" / "multi" / "mean_rev" / "hybrid"
        watchlist_codes: 额外股票代码
        **kwargs: 策略参数覆盖
    
    Returns:
        配置好的 QuantSignalGenerator 实例
    """
    mode_map = {
        "ai": StrategyMode.AI_ONLY,
        "multi": StrategyMode.QUANT_MULTI,
        "mean_rev": StrategyMode.QUANT_MR,
        "hybrid": StrategyMode.HYBRID,
    }
    
    gen = QuantSignalGenerator(mode=mode_map.get(mode, StrategyMode.HYBRID))
    
    # 应用参数覆盖
    for k, v in kwargs.items():
        if hasattr(gen, k):
            setattr(gen, k, v)
    
    # 加载自选股
    gen.load_watchlist(watchlist_codes or [])
    
    return gen

# -*- coding: utf-8 -*-
"""多策略信号集成器（让 Feature #4 的市场状态加权真正作用在信号层）。

背景：现有闭环核心是 ``trade_plan`` 目标组合 + 止损，并非真正的「多策略引擎」；
``signals.py`` 里虽有动量 / 均值回归等独立信号生成器，却没汇入统一的、可被
市场状态加权的决策入口。本模块补上这一环——把多个策略的评分按
「市场状态权重 × 历史 IC 权重」融合成一个综合分与买卖决策，并给出每条策略的
贡献拆解（直接喂给 Feature #2 的绩效归因）。

设计原则：
- 纯逻辑、零 IO、可单测；不依赖实时行情 / 网络。
- 权重 = regime 权重（来自 ``src.trading.regime.RegimeWeighter``）× IC 权重
  （策略历史信息系数，可选，缺省为 1.0）。
- 未列出 / 未知标签按权重 1.0 处理，绝不因「多策略」引入不可控放大。
- 决策阈值可配：综合分 ≥ buy 阈值 → 买入信号；≤ sell 阈值 → 卖出；之间 → 持有。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.trading.regime import RegimeWeighter, DEFAULT_TAG_WEIGHT


def fuse_strategies(
    scores: Dict[str, float],
    regime_weights: Optional[Dict[str, float]] = None,
    ic_weights: Optional[Dict[str, float]] = None,
    decision_buy: float = 55.0,
    decision_sell: float = 45.0,
) -> Dict:
    """把多个策略评分融合成一个综合分 + 决策。

    Args:
        scores: ``{策略标签: AI 评分(0~100)}``。
        regime_weights: ``{标签: 市场状态权重}``（来自 RegimeWeighter）；缺省全 1.0。
        ic_weights: ``{标签: 历史 IC 权重}``（越大表示历史越靠谱）；缺省全 1.0。
        decision_buy / decision_sell: 综合分买卖阈值。

    Returns:
        dict 含：
        - ``fused_score``：加权综合分（0~100）。
        - ``decision``："buy" / "sell" / "hold"。
        - ``weights``：每条策略的最终权重（regime×ic，已归一化）。
        - ``contributions``：``{标签: 该策略对综合分的绝对贡献}``（用于归因）。
        - ``n_strategies``。
    """
    if not scores:
        return {
            "fused_score": 50.0, "decision": "hold", "weights": {},
            "contributions": {}, "n_strategies": 0,
        }

    rw = regime_weights or {}
    iw = ic_weights or {}
    raw_weights: Dict[str, float] = {}
    for tag, score in scores.items():
        w = float(rw.get(tag, DEFAULT_TAG_WEIGHT)) * float(iw.get(tag, 1.0))
        raw_weights[tag] = max(0.0, w)

    total = sum(raw_weights.values()) or 1.0
    norm = {t: w / total for t, w in raw_weights.items()}

    fused = 0.0
    contributions: Dict[str, float] = {}
    for t, w in norm.items():
        s = float(scores[t])
        contrib = w * s
        contributions[t] = contrib
        fused += contrib

    if fused >= decision_buy:
        decision = "buy"
    elif fused <= decision_sell:
        decision = "sell"
    else:
        decision = "hold"

    return {
        "fused_score": round(fused, 4),
        "decision": decision,
        "weights": {t: round(w, 4) for t, w in norm.items()},
        "contributions": {t: round(c, 4) for t, c in contributions.items()},
        "n_strategies": len(scores),
    }


class StrategyEnsemble:
    """面向对象的集成器封装：绑定一个 RegimeWeighter，便于闭环每 tick 复用。"""

    def __init__(
        self,
        regime_weighter: Optional[RegimeWeighter] = None,
        ic_weights: Optional[Dict[str, float]] = None,
        decision_buy: float = 55.0,
        decision_sell: float = 45.0,
    ):
        self.regime = regime_weighter or RegimeWeighter()
        self.ic_weights = dict(ic_weights or {})
        self.decision_buy = decision_buy
        self.decision_sell = decision_sell

    def set_regime(self, regime: Optional[str]) -> None:
        self.regime.update(regime)

    def set_ic_weights(self, ic_weights: Dict[str, float]) -> None:
        self.ic_weights = dict(ic_weights or {})

    def fuse(self, scores: Dict[str, float]) -> Dict:
        """用当前 regime 的权重融合策略评分。"""
        regime_weights = self.regime.describe()  # 当前市场状态下各标签权重
        return fuse_strategies(
            scores,
            regime_weights=regime_weights,
            ic_weights=self.ic_weights,
            decision_buy=self.decision_buy,
            decision_sell=self.decision_sell,
        )

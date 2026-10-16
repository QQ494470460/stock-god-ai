# -*- coding: utf-8 -*-
"""
预测评分公式 — 基于预期收益的可解释评分
替换原版系数连环乘（无经济学意义）
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def compute_score(
    prob_up: float,
    prob_neutral: float,
    prob_down: float,
    confidence: float,
    agreement: float,
    risk_pct: float,
    volatility: float,
    feature_snapshot: Dict[str, float],
    n_classes: int,
    config: dict = None,
) -> Tuple[float, dict]:
    """
    基于预期收益 + 加成分项的可解释评分 (0-100)

    核心思路:
    - base_score = 50 + expected_return × 灵敏度
    - 各加分项都有明确上限，防止单一因子主导
    - 最终乘 regime 因子（牛熊市调整）

    Returns:
        (score, breakdown) — score 0-100, breakdown 各分项说明
    """
    if config is None:
        config = {}

    # 阈值从配置读取（与标签生成一致）
    threshold_up = config.get("threshold_up", 0.03)
    threshold_down = config.get("threshold_down", -0.02)
    weight_up = config.get("weight_up", 0.7)
    weight_down = config.get("weight_down", 0.3)

    # ===== 1. 预期收益基础分 (30-70) =====
    # expected_return 是模型认为的预期对数收益
    # 二分类: prob_up * threshold_up + prob_down * threshold_down
    # 三分类: prob_up * threshold_up + prob_neutral * 0 + prob_down * threshold_down
    expected_return = prob_up * threshold_up + prob_down * threshold_down
    # 映射到 50 ± 20（乘1000：如期望 +1.5% → 65分）
    base_score = 50.0 + expected_return * 1000.0

    # ===== 2. 确信度加分 (0 ~ +12) =====
    # 概率差越大，信号越清晰
    conf_bonus = confidence * 12.0

    # ===== 3. 模型一致性加分 —— 单模型后恒为 0 =====
    # 项目已收敛为单一 XGBoost（2026-10-10），不再存在"两模型是否一致"这回事，
    # predictor 里 agreement 恒为 1.0。一个**对所有股票都相同**的常量对选股排序
    # 毫无区分度，只会把全市场评分整体平移（实测 +8 分），反而让
    # min_ai_score_buy=70 / 卖出 score<40 这类绝对门槛全部够不着
    # （实测 200 只样本：加 +8 时分数落在 45.9~68.3，>=70 与 <40 命中数均为 0）。
    # 因此这里直接取 0，让评分回到"预期收益 + 确信度 + 风险惩罚"的真实水平。
    # 注意：agreement 参数保留（predictor 仍传 1.0），因为
    # src/trading/executor.py 的 require_agreement 闸门仍依赖它放行。
    agree_bonus = 0.0

    # ===== 4. 风险惩罚 (-10 ~ 0) =====
    # 波动率越高扣分越多
    risk_penalty = -risk_pct * 10.0

    # ===== 5. 市场状态因子 (× 0.88 ~ × 1.08) =====
    regime_factor = 1.0
    regime_note = "震荡市"
    is_bull = feature_snapshot.get("regime_bull", 0) > 0.5
    is_bear = feature_snapshot.get("regime_bear", 0) > 0.5

    if is_bull:
        if prob_up > prob_down:
            regime_factor = 1.08  # 牛市看涨信号加8%
            regime_note = "牛市顺风"
        else:
            regime_factor = 0.95  # 牛市但看跌信号？轻扣
            regime_note = "牛市逆风"
    elif is_bear:
        if prob_up > prob_down:
            regime_factor = 0.88  # 熊市逆势看涨，-12%
            regime_note = "熊市逆风"
        else:
            regime_factor = 1.02  # 熊市看跌正确方向，微加
            regime_note = "熊市顺风"

    # ===== 汇总 =====
    intermediate = base_score + conf_bonus + agree_bonus + risk_penalty
    intermediate = max(0.0, min(100.0, intermediate))  # 先 clamp 防止负分溢出
    score = intermediate * regime_factor
    score = max(0.0, min(100.0, score))

    breakdown = {
        "expected_return": round(expected_return, 5),
        "base_score": round(base_score, 1),
        "conf_bonus": round(conf_bonus, 1),
        "agree_bonus": round(agree_bonus, 1),
        "risk_penalty": round(risk_penalty, 1),
        "regime_factor": round(regime_factor, 3),
        "regime_note": regime_note,
        "intermediate": round(intermediate, 1),
        "final_score": round(score, 1),
    }

    return score, breakdown


# 分档基于 2026-10-10 单模型化后实测的 200 只样本分布（mean 47.3 / std 3.6），如评分公式再变需重新校准。
def get_score_interpretation(score: float) -> Tuple[str, str]:
    """评分→人话"""
    if score >= 55:
        return "🟢 强烈看涨", "模型高度确信上涨，建议关注"
    elif score >= 51:
        return "🟢 看涨", "模型看涨信号较明确"
    elif score >= 49:
        return "🟡 偏看涨", "信号偏多但不够强烈"
    elif score >= 45:
        return "⚪ 中性", "方向不明确，建议观望"
    elif score >= 41:
        return "🟠 偏看跌", "信号偏空但不够强烈"
    elif score >= 37:
        return "🔴 看跌", "模型看跌信号较明确"
    else:
        return "🔴 强烈看跌", "模型高度确信下跌，建议回避"

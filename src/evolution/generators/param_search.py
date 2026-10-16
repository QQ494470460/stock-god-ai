# -*- coding: utf-8 -*-
"""自我进化 · 生成器 A：交易结果驱动的组合参数寻优。

跟「随机搜参数」的区别
======================
本生成器不盲搜。它先从反馈快照里**读出症状**，再对症提出改动：

===============  ==========================================  ==================
症状（可量化）    诊断                                         处方
===============  ==========================================  ==================
换手过高          容忍带太窄，来回买卖付手续费                  调高 ``band_pct``
现金长期趴着      保留比例过高，跑不赢基准                      调低 ``cash_reserve_pct``
持仓数长期不满    目标只数超出可选标的数，稀释了仓位            调低 ``max_positions``
回撤逼近熔断线    单票权重上限过宽                              调低 ``weight_cap``
命中率低盈亏比高  等权摊平了高分标的的优势                      权重改 ``score``
碎单过多          最小交易金额太低，交易成本被稀释              调高 ``min_trade_amount``
===============  ==========================================  ==================

每条处方都必须**引用具体数字**作为理由 —— 这样人在批准时能一眼判断对错，
而不是面对一句「模型认为这样更好」。

安全约束
========
* 只在 :data:`src.evolution.space.PARAM_SPACE` 白名单内取值；
* 更激进方向（如调低现金保留、调高单票上限）照提，但 gate 会标记
  「须人工批准」，绝不会被自动采纳 —— 生成器不负责拦截，负责**如实提出**；
* 每个症状最多提一条，避免刷屏；
* 无证据（快照 ``is_useful()`` 为假）时返回空列表，不硬凑。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from src.evolution import space
from src.evolution.feedback import PerformanceSnapshot
from src.evolution.generators.base import atomize, require_evidence
from src.evolution.types import Candidate

NAME = "param_search"

#: 触发阈值（保守取值，宁可不动）
TURNOVER_HIGH = 3.0            # 年换手 > 3 倍 → 交易过频
CASH_DRAG_HIGH = 0.35          # 现金占比均值 > 35% → 拖累
POSITION_GAP_LOW = 0.60        # 平均持仓 / 目标持仓 < 60% → 目标过高
DRAWDOWN_NEAR_HALT = 0.70      # 回撤达到熔断线的 70% → 开始防守
WIN_RATE_LOW = 0.45            # 日胜率 < 45%
PL_RATIO_GOOD = 1.5            # 盈亏比 > 1.5 → 说明选股不差，是权重没给够
MIN_TRADE_BREAK = 0.004        # 单笔最小交易额 / 平均净值 > 0.4% → 可能太碎


# ---------------------------------------------------------------------------
# 症状提取（纯函数，便于单测）
# ---------------------------------------------------------------------------

def diagnose(snapshot: PerformanceSnapshot) -> List[Dict[str, Any]]:
    """把快照翻译成一组「症状」，每条含可引用数字。

    Returns:
        ``[{symptom, detail, evidence: {...}}, ...]``
    """
    out: List[Dict[str, Any]] = []
    m = snapshot.metrics or {}

    if m.get("insufficient"):
        # 净值序列不够 → 只能看交易侧与预测侧
        return _diagnose_without_equity(snapshot)

    turnover = _f(m.get("turnover"))
    if turnover is not None and turnover > TURNOVER_HIGH:
        out.append({
            "symptom": "turnover_high",
            "detail": f"年化换手 {turnover:.1f} 倍（阈值 {TURNOVER_HIGH}）",
            "evidence": {"turnover": turnover},
        })

    # 现金占比：用结算报告里的 cash / equity
    cash_ratio = _avg_cash_ratio(snapshot.equity_series)
    if cash_ratio is not None and cash_ratio > CASH_DRAG_HIGH:
        out.append({
            "symptom": "cash_drag",
            "detail": f"现金占总资产均值 {cash_ratio:.1%}"
                      f"（阈值 {CASH_DRAG_HIGH:.0%}）",
            "evidence": {"cash_ratio": cash_ratio},
        })

    gap = _position_gap(snapshot)
    if gap is not None and gap < POSITION_GAP_LOW:
        out.append({
            "symptom": "positions_underfilled",
            "detail": f"平均持仓仅为目标持仓的 {gap:.0%}"
                      f"（阈值 {POSITION_GAP_LOW:.0%}）",
            "evidence": {"fill_ratio": gap},
        })

    dd = _f(m.get("max_drawdown"))
    if dd is not None:
        limit = _halt_drawdown_limit()
        if limit and abs(dd) >= limit * DRAWDOWN_NEAR_HALT:
            out.append({
                "symptom": "drawdown_near_halt",
                "detail": f"最大回撤 {abs(dd):.1%}，已达熔断线 "
                          f"{limit:.1%} 的 {abs(dd) / limit:.0%}",
                "evidence": {"max_drawdown": abs(dd), "halt_limit": limit},
            })

    win = _f(m.get("daily_win_rate"))
    plr = _f(m.get("profit_loss_ratio"))
    if win is not None and plr is not None and win < WIN_RATE_LOW and plr > PL_RATIO_GOOD:
        out.append({
            "symptom": "weighting_dilutes_winners",
            "detail": f"日胜率仅 {win:.1%} 但盈亏比高达 {plr:.2f}："
                      "赢的时候赚得多，说明选股不差、是等权摊平了优势",
            "evidence": {"daily_win_rate": win, "profit_loss_ratio": plr},
        })

    if (snapshot.trade_stats or {}).get("fill", 0) > 0:
        notional = _f(m.get("notional"))
        avg_eq = _avg_equity(snapshot.equity_series)
        if notional and avg_eq and snapshot.n_settlements:
            per_trade = notional / max(snapshot.trade_stats["fill"], 1)
            if per_trade / avg_eq < MIN_TRADE_BREAK:
                out.append({
                    "symptom": "too_many_small_orders",
                    "detail": f"单笔均额仅 ¥{per_trade:,.0f}"
                              f"（占净值 {per_trade / avg_eq:.2%}）",
                    "evidence": {"per_trade": per_trade, "avg_equity": avg_eq},
                })

    return out


def _diagnose_without_equity(snapshot: PerformanceSnapshot) -> List[Dict[str, Any]]:
    """没有净值序列时的降级诊断：只能看预测侧与持仓数。"""
    out: List[Dict[str, Any]] = []

    # 预测准确率持续偏低 → 说明进池标准太松（用记录里的 t5 序列近似）
    accs = snapshot.accuracy_by_horizon.get("5") or []
    if len(accs) >= 3:
        recent = sum(accs[-3:]) / 3
        if recent < 0.45:
            out.append({
                "symptom": "accuracy_low",
                "detail": f"近 3 批 T+5 准确率均值 {recent:.1%}（阈值 45%）",
                "evidence": {"recent_accuracy": recent},
            })

    gap = _position_gap(snapshot)
    if gap is not None and gap < POSITION_GAP_LOW:
        out.append({
            "symptom": "positions_underfilled",
            "detail": f"平均持仓仅为目标持仓的 {gap:.0%}",
            "evidence": {"fill_ratio": gap},
        })
    return out


# ---------------------------------------------------------------------------
# 处方
# ---------------------------------------------------------------------------

def prescribe(symptom: str, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """症状 → 一条处方。

    Args:
        symptom: 症状名（来自 :func:`diagnose`）。
        ctx: 症状上下文，即 :func:`diagnose` 产出的整条记录
            （含 ``detail`` 说明与 ``evidence`` 数字）。
            理由文案必须引用 ``ctx["detail"]`` 里的真实数字。

    Returns:
        ``{"title":…, "values":{key: new}, "rationale":…}``；无法开方返回 None。
    """
    spec_band = space.get_spec("trading_loop.band_pct")
    spec_cash = space.get_spec("trading_loop.cash_reserve_pct")
    spec_pos = space.get_spec("trading_loop.max_positions")
    spec_cap = space.get_spec("trading_loop.weight_cap")
    spec_min = space.get_spec("trading_loop.min_trade_amount")

    cur_band = _current("trading_loop.band_pct")
    cur_cash = _current("trading_loop.cash_reserve_pct")
    cur_pos = _current("trading_loop.max_positions")
    cur_cap = _current("trading_loop.weight_cap")
    cur_min = _current("trading_loop.min_trade_amount")
    cur_weighting = _current("trading_loop.weighting")

    if symptom == "turnover_high":
        new = spec_band.clamp((cur_band or 0.25) + 0.05)
        return {
            "title": "放宽调仓容忍带，降低交易频率",
            "values": {"trading_loop.band_pct": new},
            "rationale": f"{ctx['detail']}。把 band_pct 由 {cur_band} 提到 {new}，"
                         "让小幅偏离不触发调仓，直接省下双边成本。",
        }

    if symptom == "cash_drag":
        new = spec_cash.clamp((cur_cash or 0.10) - 0.05)
        return {
            "title": "降低现金保留比例，减少拖累",
            "values": {"trading_loop.cash_reserve_pct": new},
            "rationale": f"{ctx['detail']}。把 cash_reserve_pct 由 {cur_cash} "
                         f"降到 {new}。⚠️ 此方向更激进（满仓程度更高），需人工批准。",
        }

    if symptom == "positions_underfilled":
        new = spec_pos.clamp((cur_pos or 10) - 2)
        if new == cur_pos:
            return None
        return {
            "title": "下调目标持仓只数，避免仓位被稀释",
            "values": {"trading_loop.max_positions": new},
            "rationale": f"{ctx['detail']}。目标 {cur_pos} 只但实际买不满，"
                         f"说明可选标的不足；降到 {new} 只让每只仓位更实。",
        }

    if symptom == "drawdown_near_halt":
        new_cap = spec_cap.clamp((cur_cap or 0.30) - 0.05)
        new_cash = spec_cash.clamp((cur_cash or 0.10) + 0.05)
        values = {"trading_loop.weight_cap": new_cap}
        if new_cash != cur_cash:
            values["trading_loop.cash_reserve_pct"] = new_cash
        return {
            "title": "收紧单票权重与提高现金，防守回撤",
            "values": values,
            "rationale": f"{ctx['detail']}。单票上限 {cur_cap} → {new_cap}，"
                         f"现金保留 {cur_cash} → {new_cash}：两者都是**保守方向**，"
                         "优先保住不触及熔断线。",
        }

    if symptom == "weighting_dilutes_winners":
        if cur_weighting == "score":
            return None
        return {
            "title": "权重改为按评分加权，让高分标的占更重",
            "values": {"trading_loop.weighting": "score"},
            "rationale": f"{ctx['detail']}。由 equal 改为 score 加权，"
                         "不改变进池标准，只改变资金分配。",
        }

    if symptom == "too_many_small_orders":
        new = spec_min.clamp((cur_min or 5000.0) + 1000.0)
        return {
            "title": "提高最小交易金额，过滤碎单",
            "values": {"trading_loop.min_trade_amount": new},
            "rationale": f"{ctx['detail']}。min_trade_amount 由 {cur_min} "
                         f"提到 {new}，把不划算的小单挡在门外。",
        }

    if symptom == "accuracy_low":
        # 注意：放宽/收紧选股阈值属于 evolution.* 命名空间，
        # 由 EvolutionEngine 的记录与阈值闸门管理。这里只提示不动手，
        # 避免两套机制同时改一个参数。
        return None

    return None


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def generate(snapshot: PerformanceSnapshot,
             max_candidates: int = 3) -> List[Candidate]:
    """生成候选（纯函数，无副作用，不写盘）。

    Args:
        snapshot: :func:`src.evolution.feedback.collect` 的产出。
        max_candidates: 最多产出几条候选（避免一次刷满待批队列）。
    """
    if not require_evidence(snapshot):
        return []

    if not snapshot.equity_series:
        # 组合参数的症状（换手/现金拖累/持仓不满）全都要净值序列才能看出来。
        # 没有闭环结算数据时硬提改动就是拍脑袋 —— 明确说不做。
        logger.info("🧬 生成器 A：没有任何闭环结算数据（loop_reports 为空），"
                    "组合参数无从判断，本轮不生成候选")
        return []

    symptoms = diagnose(snapshot)
    if not symptoms:
        logger.info("🧬 生成器 A：未发现需要调参的症状，保持现状")
        return []

    pairs = []
    for s in symptoms:
        p = prescribe(s["symptom"], s)
        if p is None:
            continue
        pairs.append((p["title"], p["values"], p["rationale"]))
        if len(pairs) >= max_candidates:
            break

    cands = atomize(NAME, pairs)
    if cands:
        logger.info(f"🧬 生成器 A 产出 {len(cands)} 条候选："
                    + "、".join(c.title for c in cands))
    return cands


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _f(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _current(key: str) -> Any:
    try:
        from src.evolution.overrides import read_effective

        return read_effective(key)
    except Exception:  # noqa: BLE001
        return space.get_default(key)


def _avg_cash_ratio(series: List[Dict[str, Any]]) -> Optional[float]:
    vals = []
    for row in series or []:
        eq = _f(row.get("equity"))
        cash = _f(row.get("cash"))
        if eq and eq > 0 and cash is not None:
            vals.append(cash / eq)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _avg_equity(series: List[Dict[str, Any]]) -> Optional[float]:
    vals = [_f(r.get("equity")) for r in (series or [])]
    vals = [v for v in vals if v and v > 0]
    return sum(vals) / len(vals) if vals else None


def _position_gap(snapshot: PerformanceSnapshot) -> Optional[float]:
    """实际持仓数 / 目标持仓数。"""
    rows = [r for r in (snapshot.equity_series or [])
            if _f(r.get("n_positions")) is not None]
    if not rows:
        return None
    target = _f(_current("trading_loop.max_positions")) or 0
    if target <= 0:
        return None
    avg = sum(_f(r.get("n_positions")) for r in rows) / len(rows)
    return avg / target


def _halt_drawdown_limit() -> Optional[float]:
    """熔断回撤线（绝对值，如 0.08）。读不到就返回 None，不瞎猜。"""
    try:
        from src.trading.guard import GuardConfig

        cfg = GuardConfig()
        v = float(cfg.max_portfolio_drawdown_pct)
        return v / 100.0 if v > 1 else v
    except Exception:  # noqa: BLE001
        return None

# -*- coding: utf-8 -*-
"""执行级 A/B：把实盘的退出规则逐条搬进回测，看谁真的在赚钱、谁在毁灭收益。

用法：
    python scripts/exec_ab.py                        # 默认面板 + 10bp 滑点，跑全部方案
    python scripts/exec_ab.py --slippage-bp 50       # 成本敏感性
    python scripts/exec_ab.py --only live,no_exit    # 只跑指定方案
    python scripts/exec_ab.py --quick                # 只跑最近 400 个交易日（迭代用）
    python scripts/exec_ab.py --phases 15 --only c15,tilt25   # 相位平均（慢 15 倍，但可信）

**为什么要有这个脚本**：实盘 −6.75%/7 天、盈亏比 0.153、10 连亏，而样本外面板
RankIC .061 / 胜率 61.5%。两个数字不可能同时对 —— 差异只可能来自执行层：
止损把波动实现成了亏损。这个脚本就是用来证明/证伪这句话的。

**为什么默认表里带一列 Δ（相对基准）**：11-04 之前每次 A/B 都要人肉拿两个数字相减，
再凭感觉判断差多少算「有效」。把基准和差值放进同一张表，才能让「这个差异
小于相位噪声」这件事变得显眼（见下方 ``PHASE_SPREAD_PP``）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.core.exec_backtest import (  # noqa: E402
    ExecConfig, build_ohlc_panel, run_exec_backtest,
)

# 基准方案自身的调仓相位离散度（1140 天，c15 扫 15 个相位：12.72% ~ 29.35%）。
# 这是「结论可信度」的标尺，不是某个策略的特性：换任何策略量级都类似。
# ⇒ 任何单相位下小于这个数的「A 比 B 好 X pp」都不能当结论，必须看相位均值。
PHASE_SPREAD_PP = 16.64

# 生产默认（真·样本外验证过的那组）
BASE = dict(
    top_n=8, rebalance_days=5, weighting="risk_parity",
    cash_reserve_pct=0.10, band_pct=0.15, weight_cap=0.30,
)

# 方案表：名字 → ExecConfig 覆盖项
# 第一轮结论（1140 天，10bp）：止损单独用有益（回撤 -20.6%→-16.7%），
# 止盈 +15% + 移动止损 -3% 是灾难（年化 -27.6%、胜率 13.9%、持有 1.8 天）。
# 第二轮起围绕「去掉止盈/移动止损 + 降换手 + 择时 + 信号门槛」做组合搜索。
SCENARIOS: dict[str, dict] = {
    # ---- 对照组 ----
    "no_exit":        dict(stop_mode="none"),
    "live_now":       dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           take_profit_pct=15.0, trailing_stop_pct=-3.0),
    "only_stop":      dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0),
    # ---- 换手控制：排名滞后（降成本敏感性的决定性杠杆）----
    "slack15":        dict(stop_mode="none", hold_rank_slack=1.5),
    "slack20":        dict(stop_mode="none", hold_rank_slack=2.0),
    "slack30":        dict(stop_mode="none", hold_rank_slack=3.0),
    "slack20_stop":   dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           hold_rank_slack=2.0),
    # ---- 信号强度门槛（z 分数）----
    "slack20_z15":    dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           hold_rank_slack=2.0, min_score_z=1.5),
    "slack20_gate60": dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           hold_rank_slack=2.0, market_gate="ma60", market_gate_scale=0.5),
    # ---- 最终候选：15 日周期 + 排名滞后 3.0（网格里唯一三档成本都站得住的格子）----
    "c15":            dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0),
    "c15_stop":       dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           rebalance_days=15, hold_rank_slack=3.0),
    "c15_fix20":      dict(stop_mode="fixed", stop_loss_pct=-20.0,
                           rebalance_days=15, hold_rank_slack=3.0),
    "c15_fix12":      dict(stop_mode="fixed", stop_loss_pct=-12.0,
                           rebalance_days=15, hold_rank_slack=3.0),
    "c15_budget":     dict(stop_mode="atr", stop_loss_pct=-5.0, atr_max_pct=15.0,
                           risk_budget_pct=0.005, band_mode="rel", band_pct=0.35,
                           rebalance_days=15, hold_rank_slack=3.0),
    # ---- 风险预算阶梯（不止损 + 15 日 + slack3）：单票上限 = rb / 15% ----
    "c15_rb075":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           risk_budget_pct=0.0075, band_mode="rel", band_pct=0.35),
    "c15_rb100":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           risk_budget_pct=0.010, band_mode="rel", band_pct=0.35),
    "c15_rb150":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           risk_budget_pct=0.015, band_mode="rel", band_pct=0.35),
    # ---- 波动率目标（组合层压回撤）----
    "c15_vol15":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           vol_target_pct=15.0),
    "c15_vol20":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           vol_target_pct=20.0),
    "c15_vol25":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           vol_target_pct=25.0),
    # ---- 回撤闸门（第六处口径漂移：实盘两层「越跌越轻」，回测一个都没有）----
    # 基准 c15 = 不止损 + 15 日周期 + 排名滞后 3.0（唯一三档成本都站得住的格子）。
    # dd_live 复现实盘现状：渐进降仓 floor 0.30 + 8% 熔断冻结整轮调仓。
    "dd_live":        dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           dd_scale_floor=0.30, dd_scale_at_pct=8.0, halt_dd_pct=8.0),
    "dd_scale":       dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           dd_scale_floor=0.30, dd_scale_at_pct=8.0),
    "dd_scale50":     dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           dd_scale_floor=0.50, dd_scale_at_pct=8.0),
    "dd_halt8":       dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=8.0),
    "dd_halt12":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=12.0),
    "dd_halt15":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=15.0),
    "dd_halt20":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=20.0),
    # ---- 熔断自愈的完整生命周期（第 15 处口径漂移）----
    # 生产现状的四件套：回撤熔断 20% + 解冻滞后带 2% + 自愈后 2 日半仓冷静期
    # + 单日亏损熔断 5%。基准 = c15（不止损 + 15 日 + slack3）。
    # ⚠️ halt_dd_pct=20 不是「理论上才有」的闸门：策略最大回撤 -25.5% ⇒
    #    1140 天里必然撞线。以前所有「回测没有这条线」的结论都默认它零成本，未经验证。
    "prod_risk":      dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=20.0, resume_dd_margin_pct=2.0,
                           resume_throttle_days=2, resume_risk_fraction=0.5,
                           daily_halt_pct=5.0),
    "prod_halt20":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=20.0),
    "prod_halt20_m2": dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=20.0, resume_dd_margin_pct=2.0),
    "prod_throttle":  dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=20.0, resume_throttle_days=2,
                           resume_risk_fraction=0.5),
    "prod_thr_1d":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           daily_halt_pct=5.0, resume_throttle_days=1,
                           resume_risk_fraction=0.5),
    "prod_thr_3d":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           daily_halt_pct=5.0, resume_throttle_days=3,
                           resume_risk_fraction=0.5),
    "prod_thr_f75":   dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           daily_halt_pct=5.0, resume_throttle_days=2,
                           resume_risk_fraction=0.75),
    "prod_daily5":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           daily_halt_pct=5.0),
    "prod_halt25":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=25.0, resume_dd_margin_pct=2.0,
                           resume_throttle_days=2, resume_risk_fraction=0.5,
                           daily_halt_pct=5.0),
    "prod_halt30":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           halt_dd_pct=30.0, resume_dd_margin_pct=2.0,
                           resume_throttle_days=2, resume_risk_fraction=0.5,
                           daily_halt_pct=5.0),
    # ---- 相位平均复核用（11-04 发现单相位结论会被 16.6pp 的相位噪声淹没）----
    # 这三组当初都是在 offset=0 上单点选出来的，差值只有 1~4pp，必须用相位均值重判。
    "tilt10":         dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           tilt_strength=0.10),
    "tilt25":         dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           tilt_strength=0.25),
    "cash00":         dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           cash_reserve_pct=0.0),
    "cash05":         dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           cash_reserve_pct=0.05),
    "cash15":         dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           cash_reserve_pct=0.15),
    # ---- 行业集中度上限（第 13 处漂移：30%→50%）----
    # ⚠️ 这组**必须**配 --industry-map 才有意义：引擎在 industry_map 缺失时对
    #    该票 fail-open（豁免，不占额度），映射没传进来 ⇒ max_sector_pct 静默无效，
    #    四个方案会跑出逐位相同的数字。exec_ab 会直接跳过它们而不是给假结论。
    # 30% 是「20 只持仓」时代的遗留；8 只时单票 11.25%，30% 只够 2.67 只。
    # ⚠️ 当初 30%→50% 是**单相位**定的，且档位非单调（记忆里自己写了
    #    「差异是哪只票恰好被拒的随机扰动」）⇒ 必须用相位平均重判。
    "sector30":       dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           max_sector_pct=0.30),
    "sector40":       dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           max_sector_pct=0.40),
    "sector50":       dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           max_sector_pct=0.50),
    "sector70":       dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           max_sector_pct=0.70),
    # ---- 打分后处理（第 16 处漂移，且在打分层：比风控层的 15 处更上游）----
    # 生产排序 = predict_proba → 行业中性化 → EMA(0.85)，面板存的是原分。
    # 记忆里那条「EMA 让年化 23.7%→26.7%」是**单相位 + 无脚本可复现**，
    # 而 4pp 远小于 16.64pp 的相位离散度 ⇒ 必须重判。
    "ema85":          dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           signal_ema_alpha=0.85),
    "ema95":          dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           signal_ema_alpha=0.95),
    "neu":            dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           signal_neutralize=True),
    "prod_signal":    dict(stop_mode="none", rebalance_days=15, hold_rank_slack=3.0,
                           signal_neutralize=True, signal_ema_alpha=0.85),
}


def _fmt(v: float, pct: bool = False, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v*100:+.{digits}f}%" if pct else f"{v:.{digits}f}"


def sweep_phases(scores, ohlc, over: dict, *, slippage_bp: float = 10.0,
                 phases: int = 1, top_n: int = 0) -> list[dict]:
    """在 ``phases`` 个调仓相位上跑同一份配置，返回逐相位的 metrics。

    2026-11-04 的发现：1140 天、同一个策略、同一批数据，只把调仓日整体平移几天，
    基准 c15 的年化就在 12.72% ~ 29.35% 之间跳，离散 **16.64pp**。
    也就是说此前所有「A 比 B 好 2pp」的结论，都可能是 A 恰好抽到了顺手的相位。

    更隐蔽的一层：某条风控线「历史成本为 0」通常不是它安全，而是它冻结的那几天
    **恰好没有调仓日**。实测回撤熔断 20% 在 offset=0 时冻结 2 天、零成本；
    扫到别的相位就会撞上调仓日，最差相位 −2.23pp（memory §5.3）。

    ``phases=1`` 时等价于旧行为（只跑 offset 0），结果逐位相同。
    """
    n = max(int(phases), 1)
    kw = dict(over)
    kw["slippage_bp"] = slippage_bp
    # 相位由本函数统一控制：方案里写 rebalance_offset 一律丢弃。
    # 否则「扫 15 个相位」会静默退化成「同一个相位跑 15 次」，而 summarise
    # 还会把 15 个相同值算出一个漂亮的「方差 0、配对 t=∞」—— 全错但不报错。
    kw.pop("rebalance_offset", None)
    if top_n:
        kw["top_n"] = int(top_n)
    runs: list[dict] = []
    for off in range(n):
        res = run_exec_backtest(scores, ohlc, ExecConfig(rebalance_offset=off, **kw))
        runs.append({"offset": off, "metrics": dict(res.metrics), "result": res})
    return runs


def paired_t(a, b) -> float:
    """配对 t：同一相位下两方案相减，消掉「这个相位本身好坏」这个共同项。

    ⚠️ ``phases`` 个相位是**重叠样本**、不算独立观测，所以这个 t 只用来区分
    「系统性差异」和「方向随机的噪声」，不能当严格显著性检验报给外人。

    边界语义（踩过）：
      · 逐相位差**恒为同一个值**（方差 0）⇒ 这是最系统的结果，返回 ``±inf``，
        不能返回 nan —— nan 在报表里显示成「—」，会被读成「没有信息」。
      · 逐相位差**恒为 0** ⇒ 确实没有可检的差异，返回 nan。
    """
    x = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if x.size < 2:
        return float("nan")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd <= 1e-12:
        if abs(mean) <= 1e-12:
            return float("nan")
        return float("inf") if mean > 0 else float("-inf")
    return float(mean / (sd / np.sqrt(x.size)))


def fmt_t(t: float) -> str:
    """配对 t 的显示：无穷大是「完全系统性」，NaN 是「无可检差异」。"""
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return "—"
    if isinstance(t, float) and np.isinf(t):
        return "∞" if t > 0 else "-∞"
    return f"{t:.2f}"


def summarise(runs: list[dict], base: list[dict] | None = None) -> dict:
    """把逐相位结果压成一行：均值 / 最差 / 与基准的逐相位差。

    三件事一起看（缺一不可）：
      · **均值** = 期望值，真正的决策依据
      · **最差相位** = 尾部风险，策略不能只在好相位上活着
      · **负相位个数** = 稳不稳；若 6/15 负、9/15 恰好为 0，说明是纯尾部成本
    """
    def _col(key: str, scale: float = 1.0) -> list[float]:
        out = []
        for r in runs:
            v = r["metrics"].get(key)
            out.append(float("nan") if v is None else float(v) * scale)
        return out

    def _agg(fn, vals) -> float:
        """全 NaN 时直接给 NaN —— ``np.nanmean`` 会抛 "Mean of empty slice" 警告，
        那种噪音会逐步把真正的告警淹掉。"""
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0 or bool(np.all(np.isnan(arr))):
            return float("nan")
        return float(fn(arr))

    ann = _col("ann_return", 100.0)
    shr = _col("sharpe")
    dd = _col("max_drawdown", 100.0)
    res = {
        "phases": len(runs),
        "ann_mean": _agg(np.nanmean, ann),
        "ann_min": _agg(np.nanmin, ann),
        "ann_max": _agg(np.nanmax, ann),
        "ann_std": _agg(lambda a: np.nanstd(a, ddof=1), ann)
        if len(ann) > 1 else 0.0,
        "sharpe_mean": _agg(np.nanmean, shr),
        "sharpe_min": _agg(np.nanmin, shr),
        "dd_mean": _agg(np.nanmean, dd),
        "d_ann_mean": float("nan"), "d_ann_min": float("nan"),
        "d_ann_max": float("nan"),
        "d_sharpe_mean": float("nan"), "t": float("nan"), "neg_phases": 0,
    }
    for key in ("calmar", "win_rate", "payoff", "profit_factor",
                "avg_hold_days", "turnover_per_year"):
        res[key] = _agg(np.nanmean, _col(key))
    if base and len(base) == len(runs) and runs:
        b_ann = [float(r["metrics"]["ann_return"]) * 100 for r in base]
        b_shr = [float(r["metrics"]["sharpe"]) for r in base]
        d = np.asarray(ann) - np.asarray(b_ann)
        ds = np.asarray(shr) - np.asarray(b_shr)
        res.update({
            "d_ann_mean": _agg(np.nanmean, d),
            "d_ann_min": _agg(np.nanmin, d),
            "d_ann_max": _agg(np.nanmax, d),
            "d_sharpe_mean": _agg(np.nanmean, ds),
            "t": paired_t(ann, b_ann),
            # 差 < -0.01pp 才算「这个相位真的输了」，避免 1e-13 级的浮点差被计数
            "neg_phases": int(np.sum(d < -0.01)),
        })
    return res


def needs_industry_map(over: dict) -> bool:
    """该方案声明了行业上限或行业中性化，却没有拿到 industry_map。

    引擎在缺映射时：行业上限对该票 **fail-open（豁免、不占额度）**，
    中性化则**整层静默跳过**（``apply_signal_pipeline`` 里 ``and cfg.industry_map``）。
    两者都会让参数变成**静默无效**：30% / 50% / 70% 三档跑出逐位相同的数字，
    ``prod_signal`` 退化成「只有 EMA」。读表的人会得出「这条约束零成本」
    ——最坏的一种假结论。
    """
    cap = float(over.get("max_sector_pct", 0.0) or 0.0)
    wants_neutral = bool(over.get("signal_neutralize"))
    return (cap > 0 or wants_neutral) and not over.get("industry_map")


def _run_grid(scores, ohlc, args) -> None:
    """决定性网格：持有周期 × 排名滞后 × 成本三档。

    5 天周期下每期边际收益约 0.5%，一次往返成本在 50bp 假设下约 1.1% ——
    结构上必亏。这个网格回答「周期要拉到多长，edge 才能覆盖成本」。
    """
    print("\n【网格】不止损（no_exit）｜行=持有周期，列=排名滞后，单元格=50bp 下 夏普/年化")
    for bp in (10.0, 30.0, 50.0):
        print(f"\n— 滑点 {bp:.0f}bp/边 " + "—" * 62)
        print(f"{'周期':<8}" + "".join(f"{'slack=' + str(s):>20}" for s in (1.0, 1.5, 2.0, 3.0)))
        for rb in (5, 10, 15, 20, 30):
            cells = []
            for sl in (1.0, 1.5, 2.0, 3.0):
                over = dict(BASE)
                over.update(dict(stop_mode="none", rebalance_days=rb, hold_rank_slack=sl))
                m = run_exec_backtest(scores, ohlc,
                                      ExecConfig(slippage_bp=bp, **over)).metrics
                cells.append(f"{_fmt(m.get('sharpe')):>7}/{_fmt(m.get('ann_return'), True):>9}"
                             f"{'':>4}")
            print(f"{rb:<8}" + "".join(f"{c:>20}" for c in cells))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/evolution/panel_oos")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--slippage-bp", type=float, default=10.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true", help="只跑最近 400 个交易日")
    ap.add_argument("--top-n", type=int, default=0)
    ap.add_argument("--grid", action="store_true", help="跑持有周期×排名滞后网格")
    ap.add_argument("--phases", type=int, default=1,
                    help="调仓相位个数（1=只跑 offset0，与旧结果逐位相同；15=相位平均）")
    ap.add_argument("--baseline", default="c15",
                    help="算 Δ 用的基准方案名，空字符串=不算（单相位下 Δ 只是参考）")
    ap.add_argument("--industry-map", default="",
                    help="行业映射 JSON 路径（如 data/industry_map.json）。"
                         "max_sector_pct 类方案必须给，否则会被跳过而不是给假结论")
    args = ap.parse_args()

    industry_map = None
    if args.industry_map:
        import json
        p = Path(args.industry_map)
        if not p.exists():
            print(f"❌ 行业映射不存在：{p}")
            return 1
        industry_map = json.loads(p.read_text(encoding="utf-8"))
        print(f"🏭 行业映射 {len(industry_map)} 条"
              f"（面板里缺映射的票按 fail-open 处理，与实盘一致）")

    panel = Path(args.panel)
    scores = pd.read_parquet(panel / "score_panel.parquet")
    codes = list(scores.columns)
    start, end = scores.index.min(), scores.index.max()
    print(f"📦 面板 {scores.shape[0]} 天 × {len(codes)} 只  {start.date()} → {end.date()}")

    ohlc = build_ohlc_panel(codes, str(start.date()), str(end.date()), args.data_dir)
    if not ohlc or ohlc["close"].empty:
        print("❌ 没有可用 OHLC 数据")
        return 1

    if args.quick:
        keep = scores.index[-400:]
        scores = scores.loc[keep]
        ohlc = {k: v.loc[v.index.isin(keep)] for k, v in ohlc.items()}
        print(f"⚡ quick 模式：{len(keep)} 个交易日")

    if args.grid:
        _run_grid(scores, ohlc, args)
        return 0

    names = [n for n in (args.only.split(",") if args.only else SCENARIOS) if n]
    base_name = args.baseline if args.baseline in SCENARIOS else ""
    # 基准必须跑，否则整张表没有参照物 —— Δ 才是「值不值得改」的判断依据。
    if base_name and base_name not in names:
        names.insert(0, base_name)
    multi = int(args.phases) > 1

    runs_by_name: dict[str, list[dict]] = {}
    for name in names:
        if name not in SCENARIOS:
            print(f"⚠️ 未知方案 {name}，跳过")
            continue
        over = dict(BASE)
        over.update(SCENARIOS[name])
        if industry_map:
            over["industry_map"] = industry_map
        if needs_industry_map(over):
            print(f"⚠️ 跳过 {name}：声明了 max_sector_pct 但没传 --industry-map，"
                  "跑出来的会是「上限无效」的假结论")
            continue
        runs = sweep_phases(scores, ohlc, over, slippage_bp=args.slippage_bp,
                            phases=args.phases, top_n=args.top_n)
        runs_by_name[name] = runs
        s = summarise(runs)
        print(f"  ✓ {name:<16} 年化 {s['ann_mean']:>7.2f}%  "
              f"夏普 {s['sharpe_mean']:>5.2f}  回撤 {s['dd_mean']:>7.2f}%"
              + (f"   ({s['phases']} 相位)" if multi else ""))

    if not runs_by_name:
        return 1

    base_runs = runs_by_name.get(base_name) if base_name else None
    rows = []
    for name, runs in runs_by_name.items():
        rows.append({**summarise(runs, base_runs), "方案": name,
                     "_result": runs[0]["result"]})

    print("\n" + "=" * 122)
    print(f"成本 {args.slippage_bp:.0f}bp/边 ｜ 佣金 3bp ｜ 印花税 5bp(卖出) ｜ "
          f"相位 {int(args.phases)} 个 ｜ 基准 {base_name or '—'} ｜ "
          f"{'各列为相位均值' if multi else '单相位 offset=0'}")
    print("=" * 122)
    df = pd.DataFrame(rows).sort_values("sharpe_mean", ascending=False, na_position="last")
    # 单相位时「最差」恒等于「年化」（只有一个相位），留着只会让人以为扫过了。
    hdr = (f"{'方案':<16}{'年化':>9}"
           + (f"{'最差':>9}" if multi else "")
           + f"{'Δ基准':>9}{'夏普':>7}{'Δ夏普':>8}"
           + (f"{'配对t':>7}" if multi else "")
           + f"{'回撤':>9}{'Calmar':>8}{'胜率':>8}{'盈亏比':>8}"
             f"{'盈利因子':>9}{'持有日':>7}{'年换手':>8}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        line = (f"{r['方案']:<16}"
                f"{_fmt(r['ann_mean'] / 100, True):>9}")
        if multi:
            line += f"{_fmt(r['ann_min'] / 100, True):>9}"
        line += (f"{_fmt(r['d_ann_mean'] / 100, True):>9}"
                 f"{_fmt(r['sharpe_mean']):>7}"
                 f"{_fmt(r['d_sharpe_mean']):>8}")
        if multi:
            line += f"{fmt_t(r['t']):>7}"
        line += (f"{_fmt(r['dd_mean'] / 100, True):>9}"
                 f"{_fmt(r['calmar']):>8}{_fmt(r['win_rate'], True, 1):>8}"
                 f"{_fmt(r['payoff']):>8}{_fmt(r['profit_factor']):>9}"
                 f"{_fmt(r['avg_hold_days'], False, 1):>7}"
                 f"{_fmt(r['turnover_per_year'], False, 1):>8}")
        print(line)

    # 相位噪声提示：单相位下 Δ 小于离散度的行，一律不能当结论
    if multi:
        if base_runs:
            b = [float(r["metrics"]["ann_return"]) * 100 for r in base_runs]
            print(f"\n📐 基准【{base_name}】自身跨相位年化 {min(b):.2f}% ~ {max(b):.2f}%，"
                  f"离散 {max(b) - min(b):.2f}pp")
        print("   「Δ基准」= 逐相位配对差的均值，「配对t」>2 才算方向稳定"
              "（相位是重叠样本，不当严格显著性检验）。")
    else:
        print(f"\n⚠️ 单相位（offset=0）结果。基准自身换相位时年化可跳 {PHASE_SPREAD_PP:.1f}pp，")
        print("   所以 Δ 小于这个数**不能**当结论 —— 复核用：--phases 15 --only 方案名")

    best = df.iloc[0]
    res = best["_result"]
    if not multi and res is not None and res.exit_stats is not None and len(res.exit_stats):
        print(f"\n【{best['方案']}】退出原因归因（按盈亏排序）")
        print(res.exit_stats.to_string(float_format=lambda x: f"{x:,.2f}"))

    # 成本敏感性：前 3 名在 30/50bp 下是否还成立（唯一判据）
    if not args.only:
        print(f"\n【成本敏感性】前 3 名在更差成本假设下是否仍成立（50bp 崩掉 = 不可用）")
        if multi:
            print("  ⚠️ 这里的成本敏感性沿用 --phases，运行时长 = 9 × 相位个数")
        for _, row in df.head(3).iterrows():
            over = dict(BASE)
            over.update(SCENARIOS[row["方案"]])
            if industry_map:
                over["industry_map"] = industry_map
            line = f"  {row['方案']:<18}"
            for bp in (10.0, 30.0, 50.0):
                runs = sweep_phases(scores, ohlc, over, slippage_bp=bp,
                                    phases=args.phases, top_n=args.top_n)
                s = summarise(runs)
                line += (f"| {bp:>2.0f}bp 年化 {s['ann_mean']:>+7.2f}% "
                         f"夏普 {s['sharpe_mean']:>5.2f} ")
            print(line + "|")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

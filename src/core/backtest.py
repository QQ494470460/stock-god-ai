# -*- coding: utf-8 -*-
"""
严谨回测引擎（Backtest Engine）
==============================

把「AI 打分」落到「可证真的组合收益」上。与看板里单股均值回归回测不同，
本引擎面向**截面选股策略**，特征完备：

  · 成本建模：佣金（含最低佣金）、印花税（卖出单边）、滑点（双边）
  · 多宇宙：全A / HS300 / ZZ500 / ZZ1000（由调用方传入对应 close/score 面板）
  · 调仓与持有：每 rebalance_days 调仓，等权持有 TopN
  · 完整指标：RankIC / IC_IR / IC>0、TopN 超额、年化收益、年化波动、
    夏普、索提诺、最大回撤、胜率、换手率

两种用法
--------
1) 已有样本外（OOS）打分面板 → 直接回测组合：
       bt = WalkForwardBacktester(BacktestConfig(top_n=20))
       res = bt.backtest_scores(score_panel, close_panel, benchmark_close=hs300)

2) 给定预测器 + 行情 → 滚动重打分，得到真实 OOS 组合回测：
       res = bt.backtest_predictor_rolling(predictor, stock_data, rebalance_days=5)

设计原则：只依赖 numpy/pandas；绝不联网；可离线单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# 风险加权估计协方差的历史窗口（交易日）。只用再平衡日 t 及之前的数据，
# 保证 walk-forward 无未来泄露；窗口不足时退化为「t 之前全部可用历史」。
RISK_WINDOW = 60


@dataclass
class BacktestConfig:
    """回测参数（成本默认采用 A 股主流保守值）

    组合构建参数（``weighting`` / ``cash_reserve_pct`` / ``weight_cap`` /
    ``band_pct`` / ``min_trade_amount``）与 ``trading_loop.*`` 一一对应，
    用途是让「自我进化」的样本外闸门能对**真实要调的那些参数**做 A/B
    对比 —— 否则像 ``band_pct`` 这种参数在回测里毫无表达，闸门只能一律拒绝。

    默认值刻意取「不做任何特殊处理」的一侧，保证历史口径可比：
    ``equal`` 等权、零现金保留、不封顶、无容忍带、不过滤碎单。
    """
    top_n: int = 20
    rebalance_days: int = 5          # 调仓/持有周期（交易日）
    commission: float = 0.0003       # 佣金率（单边）
    min_commission: float = 5.0      # 单笔最低佣金（元）
    stamp_tax: float = 0.0005        # 印花税（卖出单边）
    slippage: float = 0.001          # 滑点（单边）
    capital: float = 1_000_000.0     # 初始资金
    risk_free: float = 0.02          # 无风险利率（年化）
    periods_per_year: int = 252
    min_stocks_per_day: int = 30     # 每日最少有效股票数（算 IC 用）

    # ---- 组合构建（与 trading_loop.* 对齐）----
    weighting: str = "equal"         # equal | score
    cash_reserve_pct: float = 0.0    # 现金保留比例（0 = 满仓）
    weight_cap: float = 1.0          # 单票权重上限（1.0 = 不封顶）
    band_pct: float = 0.0            # 调仓容忍带（0 = 每次都调）
    min_trade_amount: float = 0.0    # 单笔最小交易额（元，0 = 不过滤碎单）
    # 持有纪律（与 trading_loop.hold_if_profitable_pct 同义）：持仓跌出目标组合、
    # 但未实现浮盈 ≥ 该百分比时不强制卖出，保留漂移权重继续持有，其余目标仓位
    # 缩到剩余预算。0 = 关闭（默认，行为与历史完全一致）。
    hold_if_profitable_pct: float = 0.0
    tilt_strength: float = 0.0        # alpha 倾斜强度（risk 加权之上叠加，0=纯风险平价；真OOS验证 0.0 全面更优）


@dataclass
class BacktestResult:
    metrics: Dict[str, float] = field(default_factory=dict)
    equity: Optional[pd.Series] = None
    period_returns: Optional[pd.Series] = None
    holdings: Optional[list] = None

    def report(self) -> str:
        m = self.metrics
        lines = ["📊 回测结果"]
        order = [
            ("rank_ic", "RankIC", "{:+.4f}"),
            ("ic_ir", "IC_IR", "{:+.2f}"),
            ("ic_positive_rate", "IC>0 比例", "{:.1%}"),
            ("top_excess", "TopN 超额(单期)", "{:+.2%}"),
            ("ann_return", "年化收益", "{:+.2%}"),
            ("ann_vol", "年化波动", "{:.2%}"),
            ("sharpe", "夏普", "{:.2f}"),
            ("sortino", "索提诺", "{:.2f}"),
            ("max_drawdown", "最大回撤", "{:.2%}"),
            ("win_rate", "胜率", "{:.1%}"),
            ("turnover", "换手率/期", "{:.1%}"),
            ("n_periods", "调仓期数", "{:.0f}"),
        ]
        for k, label, fmt in order:
            if k in m and m[k] is not None and not (isinstance(m[k], float) and np.isnan(m[k])):
                lines.append(f"  {label:<14}{fmt.format(m[k])}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 指标函数
# ---------------------------------------------------------------------------
def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return float("nan")
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def _sortino(returns: pd.Series, rf_period: float, ppy: int) -> float:
    if len(returns) < 2:
        return float("nan")
    downside = returns[returns < 0]
    dd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if dd == 0:
        return float("nan")
    return float((returns.mean() - rf_period) / dd * np.sqrt(ppy))


def _rank_ic_series(score_panel: pd.DataFrame, fwd_ret: pd.DataFrame, min_stocks: int) -> pd.Series:
    common = score_panel.index.intersection(fwd_ret.index)
    ics = {}
    for d in common:
        s = score_panel.loc[d]
        r = fwd_ret.loc[d]
        v = pd.DataFrame({"s": s, "r": r}).dropna()
        if len(v) < min_stocks:
            continue
        ic = v["s"].corr(v["r"], method="spearman")
        if pd.notna(ic):
            ics[d] = float(ic)
    return pd.Series(ics).sort_index()


# ---------------------------------------------------------------------------
# 回测器
# ---------------------------------------------------------------------------
class WalkForwardBacktester:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.cfg = config or BacktestConfig()
        self._rets = None  # 日收益率矩阵缓存（风险加权用，walk-forward）

    # ---- 内部：单期交易成本（按成交额近似） ----
    def _trade_cost_ratio(self, buy_value: float, sell_value: float) -> float:
        """返回本期因买卖产生的成本占总资金比例（近似）。"""
        cfg = self.cfg
        # 买入：佣金 + 滑点
        buy_cost = buy_value * (cfg.commission + cfg.slippage)
        buy_cost = max(buy_cost, cfg.min_commission * (buy_value > 0))
        # 卖出：佣金 + 印花税 + 滑点
        sell_cost = sell_value * (cfg.commission + cfg.stamp_tax + cfg.slippage)
        sell_cost = max(sell_cost, cfg.min_commission * (sell_value > 0))
        return float((buy_cost + sell_cost) / cfg.capital)

    # ---- 内部：目标权重（等权 / 评分加权 + 单票封顶 + 现金保留） ----
    def _target_weights(self, srow: pd.Series, top: List[str], t_date=None) -> Dict[str, float]:
        """算出这一期的目标持仓权重（不含现金，合计 = 1 - 现金保留）。

        * ``weighting="score"``：按评分占比分配。评分需先平移消除负值，
          否则低分股会拿到负权重（等价于做空，A 股普通账户做不到）。
        * ``weighting="equal"``（默认）：等权。
        * ``weighting="risk_parity"`` / ``min_variance`` / ``vol_target``：
          用截至 ``t_date`` 的历史收益率估计协方差/波动率，调
          :func:`src.trading.optimizer.optimize_weights` 求风险平价权重；
          数据不足时 fail-safe 回退等权（与生产 :func:`_risk_weights` 一致）。
        """
        cfg = self.cfg
        invest = max(0.0, 1.0 - cfg.cash_reserve_pct)
        n = len(top)
        if n == 0 or invest <= 0:
            return {}

        if cfg.weighting in ("risk_parity", "min_variance", "vol_target"):
            w = self._risk_target_weights(srow, top, t_date)
            if w is not None:
                return w

        if cfg.weighting == "score":
            s = pd.to_numeric(srow[top], errors="coerce").astype(float)
            s = (s - float(s.min())).fillna(0.0)
            total = float(s.sum())
            w = ({str(c): float(v) / total for c, v in s.items()}
                 if total > 0 else {str(c): 1.0 / n for c in top})
        else:
            w = {str(c): 1.0 / n for c in top}

        cap = float(cfg.weight_cap)
        if 0 < cap < 1.0:
            for _ in range(20):
                over = [c for c, v in w.items() if v > cap + 1e-12]
                if not over:
                    break
                excess = sum(w[c] - cap for c in over)
                for c in over:
                    w[c] = cap
                free = [c for c, v in w.items() if v < cap - 1e-12]
                if not free:
                    break
                add = excess / len(free)
                for c in free:
                    w[c] += add

        return {c: v * invest for c, v in w.items()}

    # ---- 内部：风险平价目标权重（walk-forward 协方差，无未来泄露） ----
    def _risk_target_weights(self, srow: pd.Series, top: List[str], t_date) -> Optional[Dict[str, float]]:
        """风险加权目标权重；失败/数据不足返回 None（调用方回退等权）。

        关键正确性约束：协方差与波动率只用再平衡日 ``t_date`` 及之前的历史估计，
        绝不用未来数据（否则回测会吃到未来信息，夏普虚高）。窗口长度见
        ``RISK_WINDOW``；窗口内若有标的全 NaN 或覆盖不全，直接回退等权——
        与生产 :func:`_risk_weights` 对「缺价的最危险那只」的处理一致。
        """
        if self._rets is None or t_date is None:
            return None
        try:
            rets = self._rets
            if t_date not in rets.index:
                return None
            pos = rets.index.get_loc(t_date)
            start = max(0, pos - RISK_WINDOW + 1)
            window = rets.iloc[start:pos + 1][top]
            sub = window.dropna(axis=1, how="any")
            # 必须覆盖全部标的；否则缺价标的会拿到未经风险调整的权重（最危险），
            # 与生产一致地整体回退等权，而不是带病权重。
            if sub.shape[1] < 2 or sub.shape[1] != len(top):
                return None
            cov = sub.cov().values
            vols = sub.std().values
            if not (np.all(np.isfinite(cov)) and np.all(np.isfinite(vols))
                    and float(np.min(vols)) > 0):
                return None
            from src.trading.optimizer import optimize_weights

            scores = [float(srow.get(c, 0.0) or 0.0) for c in top]
            cap = self.cfg.weight_cap if 0 < self.cfg.weight_cap < 1.0 else None
            w = optimize_weights(
                cov=cov, vols=vols, scores=scores, method=self.cfg.weighting,
                cap=cap, tilt_strength=self.cfg.tilt_strength,
            )
            w = np.asarray(w, dtype=float)
            if w.size != len(top) or not np.all(np.isfinite(w)):
                return None
            total = float(w.sum())
            if total <= 0:
                return None
            invest = max(0.0, 1.0 - self.cfg.cash_reserve_pct)
            # vol_target 允许 sum(w) < 1（剩余留现金），不强行归一；其余方法归一。
            if self.cfg.weighting == "vol_target":
                return {c: float(w[i]) * invest for i, c in enumerate(top)}
            w = w / total
            return {c: float(w[i]) * invest for i, c in enumerate(top)}
        except Exception as e:  # noqa: BLE001
            logger.debug(f"风险加权失败，回退等权: {e}")
            return None

    def backtest_scores(
        self,
        score_panel: pd.DataFrame,
        close_panel: pd.DataFrame,
        benchmark_close: Optional[pd.Series] = None,
        codes_universe: Optional[List[str]] = None,
    ) -> BacktestResult:
        """用样本外打分面板回测 TopN 等权组合。

        score_panel : index=date, columns=code（当日预测分）
        close_panel : index=date, columns=code（后复权/前复权收盘）
        benchmark_close : 基准收盘（用于算超额），可选
        """
        cfg = self.cfg
        score_panel = score_panel.sort_index()
        close_panel = close_panel.sort_index()
        idx = score_panel.index.intersection(close_panel.index)
        score_panel = score_panel.loc[idx]
        close_panel = close_panel.loc[idx]
        # 风险加权所需的「截至当日」历史收益率（walk-forward，无未来泄露）：
        # 协方差只在再平衡日 t 及其之前估计；_target_weights 用 t_date 截窗口。
        # 第 0 行为 NaN（无前日收益），窗口截取会自动跳过。
        self._rets = close_panel.pct_change()
        if codes_universe:
            cols = [c for c in codes_universe if c in close_panel.columns]
            score_panel = score_panel.reindex(columns=cols)
            close_panel = close_panel.reindex(columns=cols)

        dates = list(idx)
        if len(dates) < cfg.rebalance_days + 2:
            logger.warning("回测数据不足")
            return BacktestResult(metrics={"n_periods": 0})

        # 未来 rebalance_days 收益
        fwd_ret = close_panel.shift(-cfg.rebalance_days) / close_panel - 1.0

        # RankIC 序列（用单期未来收益）
        fwd_1 = close_panel.shift(-1) / close_panel - 1.0
        ic_series = _rank_ic_series(score_panel, fwd_1, cfg.min_stocks_per_day)

        # ---- 逐调仓期组合模拟 ----
        # 状态：w_held = 本期期初各票市值权重（由上一期持有漂移而来）；
        #       cash_w = 期初现金权重。首期为「全现金」。
        period_returns, period_dates, holdings_hist, turnovers = [], [], [], []
        excess_list: List[float] = []
        equity = [1.0]
        w_held: Dict[str, float] = {}
        cash_w = 1.0
        # 每只票的建仓成本价（用于「持有纪律」判定浮盈）。买入时记当日价，
        # 持有则保留，卖出则清除 —— 与实盘 paper 账户的 avg_cost 同口径。
        entry_price: Dict[str, float] = {}
        i = 0
        n = len(dates)
        while i + cfg.rebalance_days < n:
            t = dates[i]
            t_next = dates[i + cfg.rebalance_days]
            srow = score_panel.loc[t].dropna()
            if len(srow) == 0:
                i += cfg.rebalance_days
                continue
            top = srow.nlargest(min(cfg.top_n, len(srow))).index.tolist()

            # 目标权重（含现金保留与单票封顶）
            target = self._target_weights(srow, top, t_date=t)

            # 容忍带：偏离不够大就不动，省下双边成本
            trade = True
            if cfg.band_pct > 0 and w_held:
                codes = set(target) | set(w_held)
                dev = max([abs(target.get(c, 0.0) - w_held.get(c, 0.0))
                           for c in codes] or [0.0])
                trade = dev > cfg.band_pct

            if trade:
                w_new = dict(target)
                # 持有纪律：跌出目标组合但浮盈达标 → 保留漂移权重，不强制卖出。
                # 直接「跌出榜单就卖」会在每日换票时甩掉赢家、留下还没跌的输家，
                # 等于做「卖赢家、留输家」的逆向选择（实盘盈亏比 0.15 的根因之一）。
                # 保留仓位从目标预算里扣除，其余目标按比例缩到剩余预算，避免超配。
                if cfg.hold_if_profitable_pct > 0 and w_held:
                    budget = max(0.0, 1.0 - cfg.cash_reserve_pct)
                    retained: Dict[str, float] = {}
                    used = 0.0
                    for c in sorted(w_held, key=lambda k: -w_held[k]):
                        if c in target:
                            continue
                        ep = entry_price.get(c, 0.0)
                        p0 = close_panel.at[t, c] if c in close_panel.columns else np.nan
                        if ep <= 0 or pd.isna(p0) or p0 <= 0:
                            continue
                        if (p0 / ep - 1.0) * 100.0 >= cfg.hold_if_profitable_pct:
                            w = min(w_held[c], budget - used)
                            if w > 0:
                                retained[c] = w
                                used += w
                    if retained:
                        tgt_sum = sum(w_new.values())
                        investable = max(0.0, budget - used)
                        if tgt_sum > 0 and tgt_sum > investable:
                            scale = investable / tgt_sum
                            w_new = {c: v * scale for c, v in w_new.items() if v * scale > 0}
                        w_new.update(retained)
                # 碎单过滤：成交额不足的标的维持原仓位，差额落回现金
                if cfg.min_trade_amount > 0:
                    for c in set(w_new) | set(w_held):
                        delta = w_new.get(c, 0.0) - w_held.get(c, 0.0)
                        if abs(delta) * cfg.capital < cfg.min_trade_amount:
                            if c in w_held:
                                w_new[c] = w_held[c]
                            else:
                                w_new.pop(c, None)
                buys = sum(max(0.0, w_new.get(c, 0.0) - w_held.get(c, 0.0))
                           for c in set(w_new) | set(w_held))
                sells = sum(max(0.0, w_held.get(c, 0.0) - w_new.get(c, 0.0))
                            for c in set(w_new) | set(w_held))
                cost = self._trade_cost_ratio(buys * cfg.capital,
                                              sells * cfg.capital)
            else:
                w_new = dict(w_held)      # 不调仓：维持现有仓位
                buys = sells = 0.0
                cost = 0.0

            # 成本线维护：新买入选当日价，已卖出的清除，持有/保留的保持原成本
            for c in w_new:
                if c not in entry_price:
                    _p0 = close_panel.at[t, c] if c in close_panel.columns else np.nan
                    if pd.notna(_p0) and _p0 > 0:
                        entry_price[c] = float(_p0)
            for c in list(entry_price):
                if c not in w_new:
                    entry_price.pop(c, None)

            cash_new = max(0.0, 1.0 - sum(w_new.values()))

            # 期间收益：期初即已按 w_new 持有
            rets: Dict[str, float] = {}
            for c in w_new:
                p0 = close_panel.at[t, c] if c in close_panel.columns else np.nan
                p1 = close_panel.at[t_next, c] if c in close_panel.columns else np.nan
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    rets[c] = p1 / p0 - 1.0
            if not w_new:
                i += cfg.rebalance_days
                continue
            gross = sum(w_new.get(c, 0.0) * r for c, r in rets.items())

            # 候选池等权基准（同期的全体有效股票）
            univ_rets = []
            for c in srow.index:
                p0 = close_panel.at[t, c] if c in close_panel.columns else np.nan
                p1 = close_panel.at[t_next, c] if c in close_panel.columns else np.nan
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    univ_rets.append(p1 / p0 - 1.0)
            univ = float(np.mean(univ_rets)) if univ_rets else 0.0
            excess_list.append(gross - univ)

            net = gross - cost
            period_returns.append(net)
            period_dates.append(pd.Timestamp(t_next))
            turnovers.append((buys + sells) / 2.0)
            holdings_hist.append({"date": str(pd.Timestamp(t).date()),
                                  "top": list(w_new)})
            equity.append(equity[-1] * (1 + net))

            # 期末漂移 → 下一期期初权重
            denom = 1.0 + gross
            if denom > 0:
                w_held = {c: w_new.get(c, 0.0) * (1 + rets.get(c, 0.0)) / denom
                          for c in w_new}
                cash_w = cash_new / denom
            i += cfg.rebalance_days

        if not period_returns:
            return BacktestResult(metrics={"n_periods": 0})

        pr = pd.Series(period_returns, index=period_dates)
        eq = pd.Series(equity, index=[dates[0]] + period_dates)

        ppy = cfg.periods_per_year
        rf_period = cfg.risk_free * cfg.rebalance_days / ppy
        ann_return = float((1 + pr.mean()) ** (ppy / cfg.rebalance_days) - 1)
        ann_vol = float(pr.std(ddof=1) * np.sqrt(ppy / cfg.rebalance_days)) if len(pr) > 1 else float("nan")
        sharpe = float((ann_return - cfg.risk_free) / ann_vol) if ann_vol and ann_vol > 0 else float("nan")
        sortino = _sortino(pr, rf_period, ppy // cfg.rebalance_days if cfg.rebalance_days else ppy)
        max_dd = _max_drawdown(eq)
        win_rate = float((pr > 0).mean())

        # 用 IC 与组合口径汇总
        metrics = {
            "rank_ic": float(ic_series.mean()) if len(ic_series) else float("nan"),
            # ⚠️ 关于 ic_ir 的口径（复核结论见 tests/test_ic_ir_no_lookahead.py 与
            # docs/IC_IR_复核结论.md）：
            # 本函数这里的 ic_ir = mean(每日IC) / std(每日IC)，即 Grinold-Kahn 信息比率，
            # 反映「日频排序能力的稳定性」。本模型实测约 0.3~1.5（偏低：单日 IC 的 std
            # 约为均值的 3 倍，说明单日排序噪声大，靠组合层/多日累计才显效）。
            # 注意：production_metrics.json 那个 ic_ir=14 是**另一种口径**——来自
            # labeling.py 的 mean_rank_ic，分母是「标准误 SE = std/sqrt(n_days)」（t 统计量），
            # 945 天样本把它放大约 sqrt(945)≈31 倍，所以看着像 14。两者都**不是前视/泄漏**：
            # 负向对照（逐日打乱 score → IC 塌缩到 ~0）已证明指标只响应真实排序信号。
            "ic_ir": float(ic_series.mean() / ic_series.std(ddof=1)) if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else float("nan"),
            "ic_positive_rate": float((ic_series > 0).mean()) if len(ic_series) else float("nan"),
            "top_excess": float(np.mean(excess_list)) if excess_list else float("nan"),
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "turnover": float(np.mean(turnovers)) if turnovers else float("nan"),
            "n_periods": len(pr),
        }
        return BacktestResult(metrics=metrics, equity=eq, period_returns=pr, holdings=holdings_hist)

    # ------------------------------------------------------------------
    # 用真实预测器滚动打分回测（较重，OOS 更真实）
    # ------------------------------------------------------------------
    def backtest_predictor_rolling(
        self,
        predictor,
        stock_data: Dict[str, pd.DataFrame],
        close_panel: Optional[pd.DataFrame] = None,
        benchmark_close: Optional[pd.Series] = None,
        rebalance_days: Optional[int] = None,
    ) -> BacktestResult:
        """在每个调仓日只用 t 及之前数据打分（无未来函数），构建 OOS 组合。

        predictor 需具备 predict_batch(stock_data) -> List[StockPrediction]。
        """
        cfg = self.cfg
        rb = rebalance_days or cfg.rebalance_days
        if close_panel is None:
            close_panel = pd.DataFrame({c: df["close"] for c, df in stock_data.items()}).sort_index()
        dates = list(close_panel.index)
        # 从第 seq 起点开始（保证特征可算）
        start_i = 120
        score_records = []
        i = start_i
        while i < len(dates) - rb:
            t = dates[i]
            sliced = {
                c: df.loc[df.index <= t] for c, df in stock_data.items()
                if df.index[0] <= t
            }
            try:
                preds = predictor.predict_batch(sliced)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"回测打分失败 @{t}: {e}")
                preds = []
            for p in preds:
                score_records.append({"date": pd.Timestamp(t), "code": p.code, "score": p.score})
            i += rb
        if not score_records:
            return BacktestResult(metrics={"n_periods": 0})
        sp = pd.DataFrame(score_records).pivot(index="date", columns="code", values="score")
        return self.backtest_scores(sp, close_panel, benchmark_close=benchmark_close)


# ---------------------------------------------------------------------------
# 便捷函数：对给定宇宙跑一次严谨回测
# ---------------------------------------------------------------------------
def run_backtest(
    score_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    top_n: int = 20,
    rebalance_days: int = 5,
    benchmark_close: Optional[pd.Series] = None,
    **portfolio,
) -> BacktestResult:
    """跑一次严格回测。

    ``**portfolio`` 可传 :class:`BacktestConfig` 里的组合构建参数
    （``weighting`` / ``cash_reserve_pct`` / ``weight_cap`` / ``band_pct`` /
    ``min_trade_amount``），供自我进化的样本外闸门做参数 A/B 对比。
    传了但不在 ``BacktestConfig`` 里的键会被忽略并告警，避免拼错参数名
    导致「对比了两次一样的配置」这种静默假阳性。
    """
    allowed = set(BacktestConfig.__dataclass_fields__)
    extra = {k: v for k, v in portfolio.items() if k in allowed}
    unknown = sorted(set(portfolio) - allowed)
    if unknown:
        logger.warning(f"run_backtest 收到未知参数（已忽略）：{unknown}")
    cfg = BacktestConfig(top_n=top_n, rebalance_days=rebalance_days, **extra)
    bt = WalkForwardBacktester(cfg)
    return bt.backtest_scores(score_panel, close_panel, benchmark_close=benchmark_close)

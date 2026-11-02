# -*- coding: utf-8 -*-
"""
执行级回测（Execution-aware Backtest）
=====================================

**为什么需要它**：`src/core/backtest.py` 只模拟「持有 TopN、按 band 调仓」，
完全没有止损/止盈/移动止损。而实盘跑的是「ATR 自适应止损 −5~−15% + 止盈 +15%
+ 移动止损 −3%」。也就是说，**实盘从来没有交易过被验证过的那个策略** ——
样本外 RankIC 0.061、胜率 61.5%，实盘 7 个交易日却是 −6.75%、盈亏比 0.153、
10 连亏。这道裂缝不量化，任何参数调优都是在错误的目标上前进。

本模块把实盘的退出规则逐条搬进回测，用**日内 OHLC** 判定触发：
  · 次日开盘成交（信号在收盘产生，A 股 T+1 实情），可切换为当日收盘
  · ATR 自适应止损：宽度 = clamp(max(|stop_loss|, k×ATR%), [min,max])，**只增不减**
  · 止盈 / 移动止损（按日内最高价跟踪峰值）
  · 涨跌停/停牌不可成交（一字板、无量）
  · 整手、佣金（含最低）、印花税（卖出单边）、滑点（双边 bp）
  · 单笔风险预算定仓（按止损上限反算，与实盘 guard 同口径）

口径约定（与业绩档案 `performance.py` 一致）：
  · 胜率/盈亏比只算**平仓腿**
  · 净值曲线逐日收盘 mark-to-market
  · 年化按 252 交易日

设计原则：只依赖 numpy/pandas；绝不联网；可离线单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

LOT = 100


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
@dataclass
class ExecConfig:
    """执行级回测参数。默认值 = 「什么都不做」的一侧，便于和历史口径对比。"""

    # ---- 组合构建（与 trading_loop.* 对齐）----
    top_n: int = 8
    rebalance_days: int = 5
    weighting: str = "equal"          # equal | risk_parity
    cash_reserve_pct: float = 0.0
    weight_cap: float = 1.0
    band_pct: float = 0.0
    # 容忍带的度量方式：
    #   abs = 绝对权重偏离（历史口径，8 只×11.25% 时 0.15 已接近「几乎不调仓」）
    #   rel = 相对目标权重的偏离（|Δw| / max(w)，与仓位大小无关）
    # ⚠️ 绝对口径有个致命耦合：一旦单票仓位被压到 3.3%（风险预算按 ATR 上限），
    # 任何股票都凑不出 15% 的绝对偏离 ⇒ 整个组合永不交易（实测年化 0.00%、
    # 换手 0.0%）。要小仓位就必须改用 rel 口径，否则「风控生效」等于「系统停摆」。
    band_mode: str = "abs"

    # ---- 成本（A 股保守值）----
    commission: float = 0.0003
    min_commission: float = 5.0
    stamp_tax: float = 0.0005
    slippage_bp: float = 10.0         # 单边，bp。10/30/50 三档必测
    capital: float = 1_000_000.0

    # ---- 退出规则 ----
    stop_mode: str = "none"           # none | fixed | atr
    stop_loss_pct: float = -5.0       # 负值
    atr_k: float = 2.0
    atr_min_pct: float = 2.0
    atr_max_pct: float = 15.0
    atr_window: int = 14
    take_profit_pct: float = 0.0      # >0 开启
    trailing_stop_pct: float = 0.0    # <0 开启（如 -3.0）
    hold_days_max: int = 0            # >0 时按交易日强制离场（时间止损）

    # ---- 定仓 ----
    risk_budget_pct: float = 0.0      # >0 时单票市值上限 = 权益×pct ÷ 止损宽度
    risk_budget_use_ceiling: bool = True   # True=按 atr_max 上限算（与实盘一致）

    # ---- 择时 / 选股门槛（这两个是「组合层」杠杆，不是参数微调）----
    # 市场状态闸门：用面板内全部标的的等权收益构造市场代理，收盘价在 MA 之下
    # 视为弱市，仓位乘以 market_gate_scale。目的是「弱市少亏」而不是「精准择时」。
    market_gate: str = "off"          # off | ma20 | ma60
    market_gate_scale: float = 0.4    # 弱市仓位系数（0=空仓，1=不降）
    # ---- 换手控制（成本敏感性下的决定性杠杆）----
    # 排名滞后：已持仓标的只要仍排在 top_n × slack 名以内就继续持有，不因
    # 「跌出前 8」被换掉。1.0 = 关闭。这是降低换手最有效的手段 ——
    # 榜单第 9 名和第 8 名的期望收益差异远小于一次往返成本。
    hold_rank_slack: float = 1.0
    # 信号分位门槛：只在截面分位 ≥ q 的标的里选，不够强就留现金（不填满 top_n）
    min_score_quantile: float = 0.0   # 0 = 关闭（永远填满 top_n）
    # 信号强度门槛（真正能「不填坑」的那个）：入选标的的截面 z 分数必须 ≥ 该值。
    # ⚠️ 分位门槛是冗余的 —— top_n=8 / 200 只本身就是前 4%，任何 q<0.96 都不生效。
    min_score_z: float = 0.0          # 0 = 关闭

    # ---- 波动率目标（组合层，压低回撤最直接的手段）----
    # 用组合自身近 vol_window 日的已实现波动率反推仓位：波动高就减仓。
    # 与「个股止损」的根本区别：它不实现亏损，只是把敞口缩到风险预算内，
    # 所以不会像止损那样把噪声变成已实现亏损。
    vol_target_pct: float = 0.0        # 年化目标波动（%），0 = 关闭
    vol_scale_floor: float = 0.30      # 最低仓位系数（绝不清仓）
    vol_window: int = 20               # 已实现波动的回看交易日

    # ---- 回撤闸门（与实盘 GuardState 对齐，2026-10-31 新增）----
    # 实盘有两层「越跌越轻」的机制，回测里一个都没有 —— 这是「实盘跑不过回测」
    # 的第六处口径漂移，而且是最伤的一处。加这两个开关就是为了**量化它**。
    #
    # ① dd_scale_floor：渐进降仓。实盘 ``sizing.portfolio_risk_scale`` 把回撤
    #    线性映射到 [floor, 1.0]，实盘参数 floor=0.30 / 熔断线 8%。于是
    #    回撤 2% 就砍 18% 仓位、4% 砍 35%、8% 砍到 30%。这是纯缩放的顺周期开关。
    #    实测证据：10-27 回撤 7.90% 时日志写着「敞口系数 0.31」。
    # ② halt_dd_pct：回撤熔断。实盘到 8% 就冻结整轮调仓（不建仓也不清仓），
    #    解冻还要等回撤回到 halt - margin 以下。**回测里策略的最大回撤是
    #    -20.5%** ⇒ 8% 熔断必然在正常历史里反复触发，等于换了个策略。
    #
    # 0 = 关闭。要复现实盘现状就设 dd_scale_floor=0.30 + halt_dd_pct=8.0。
    dd_scale_floor: float = 0.0        # 最深回撤时保留的敞口（0=关闭，实盘=0.30）
    dd_scale_at_pct: float = 8.0       # 回撤到该值降到 floor（实盘=熔断线 8%）
    # 非顺周期关键旋钮：回撤到该值才开始降仓（0=从 0 开始，即旧的顺周期行为；
    # 设 12 = 只在回撤≥12% 才介入，回撤 < 12% 满仓照跑）。与 T3 asymmetric_derisk_scale 口径一致。
    dd_start_cut_pct: float = 0.0
    halt_dd_pct: float = 0.0           # >0 时回撤≥该值冻结整轮调仓（0=关闭，实盘=8.0）

    # ---- 撮合 ----
    buy_at: str = "next_open"         # next_open | close
    ppy: int = 252
    risk_free: float = 0.02


@dataclass
class ExecResult:
    metrics: Dict[str, float] = field(default_factory=dict)
    equity: Optional[pd.Series] = None
    trades: Optional[pd.DataFrame] = None
    exit_stats: Optional[pd.DataFrame] = None

    def report(self) -> str:
        m = self.metrics
        order = [
            ("total_return", "累计收益", "{:+.2%}"),
            ("ann_return", "年化收益", "{:+.2%}"),
            ("ann_vol", "年化波动", "{:.2%}"),
            ("sharpe", "夏普", "{:.2f}"),
            ("max_drawdown", "最大回撤", "{:.2%}"),
            ("calmar", "Calmar", "{:.2f}"),
            ("n_trades", "平仓笔数", "{:.0f}"),
            ("win_rate", "胜率", "{:.1%}"),
            ("avg_win_pct", "平均盈利", "{:+.2%}"),
            ("avg_loss_pct", "平均亏损", "{:+.2%}"),
            ("payoff", "盈亏比", "{:.2f}"),
            ("profit_factor", "盈利因子", "{:.2f}"),
            ("avg_hold_days", "平均持有(日)", "{:.1f}"),
            ("turnover_per_year", "年换手(倍)", "{:.1f}"),
            ("fee_pct", "费用占初始本金", "{:.2%}"),
        ]
        lines = ["📊 执行级回测"]
        for k, label, fmt in order:
            v = m.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            lines.append(f"  {label:<16}{fmt.format(v)}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def _metrics_from_equity(eq: pd.Series, cfg: ExecConfig) -> Dict[str, float]:
    if eq is None or len(eq) < 3:
        return {}
    ret = eq.pct_change().dropna()
    n = len(ret)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    ann = float((1 + total) ** (cfg.ppy / max(n, 1)) - 1.0) if total > -1 else -1.0
    vol = float(ret.std(ddof=1) * np.sqrt(cfg.ppy)) if n > 1 else float("nan")
    rf = cfg.risk_free / cfg.ppy
    sharpe = float((ret.mean() - rf) / ret.std(ddof=1) * np.sqrt(cfg.ppy)) if ret.std(ddof=1) > 0 else float("nan")
    run_max = eq.cummax()
    dd = (eq / run_max - 1.0).min()
    maxdd = float(dd)
    calmar = float(ann / abs(maxdd)) if maxdd < 0 else float("nan")
    return {"total_return": total, "ann_return": ann, "ann_vol": vol,
            "sharpe": sharpe, "max_drawdown": maxdd, "calmar": calmar}


# ----------------------------------------------------------------------
# 面板准备
# ----------------------------------------------------------------------
def build_ohlc_panel(codes: List[str], start: str, end: str,
                     data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """把逐股 parquet 拼成 open/high/low/close/volume 五个 date×code 面板。

    只读本地 parquet，绝不联网。缺文件/缺字段的股票直接跳过（记 warning）。
    """
    import os

    acc: Dict[str, Dict[str, pd.Series]] = {
        k: {} for k in ("open", "high", "low", "close", "volume")
    }
    missing = 0
    for code in codes:
        path = os.path.join(data_dir, f"daily_{code}_qfq.parquet")
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            d = pd.read_parquet(path, columns=["date", "open", "high", "low", "close", "volume"])
        except Exception as e:  # noqa: BLE001
            logger.debug(f"读日线失败 {code}: {e}")
            missing += 1
            continue
        d["date"] = pd.to_datetime(d["date"])
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        if len(d) < 20:
            missing += 1
            continue
        d = d.drop_duplicates("date").set_index("date").sort_index()
        for k in acc:
            acc[k][code] = d[k].astype(float)
    if missing:
        logger.warning(f"OHLC 面板：{missing}/{len(codes)} 只缺数据或过短，已跳过")
    out = {k: pd.DataFrame(v).sort_index() for k, v in acc.items()}
    for k in out:
        out[k] = out[k].ffill(limit=5)
    return out


def _atr_pct_panel(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
                   window: int = 14) -> pd.DataFrame:
    """ATR% 面板（True Range 的 window 日均值 ÷ 收盘价 × 100）。

    True Range 用 max(H-L, |H-C_prev|, |L-C_prev|) —— 含隔夜跳空，
    比「日内振幅」更接近实盘 ATR 口径。
    """
    pc = close.shift(1)
    tr = pd.concat([
        (high - low).stack(),
        (high - pc).abs().stack(),
        (low - pc).abs().stack(),
    ], axis=1).max(axis=1).unstack()
    atr = tr.rolling(window, min_periods=max(3, window // 2)).mean()
    return atr / close * 100.0


# ----------------------------------------------------------------------
# 主回测
# ----------------------------------------------------------------------
def _tradable_masks(o: pd.DataFrame, h: pd.DataFrame, low: pd.DataFrame,
                    c: pd.DataFrame, v: pd.DataFrame,
                    limit_pct: float = 9.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """可买/可卖掩码。

    · 无量（volume<=0 或 NaN）→ 停牌，买卖都不行
    · 一字板（high==low）→ 涨跌停封死：涨不许买、跌不许卖
    """
    alive = (v.fillna(0) > 0)
    flat = (h - low).abs() < 1e-9
    pct = c / c.shift(1) - 1.0
    limit_up = flat & (pct > 0)
    limit_dn = flat & (pct < 0)
    can_buy = alive & ~limit_up
    can_sell = alive & ~limit_dn
    return can_buy, can_sell


def run_exec_backtest(
    score_panel: pd.DataFrame,
    ohlc: Dict[str, pd.DataFrame],
    cfg: ExecConfig,
    *,
    verbose: bool = False,
) -> ExecResult:
    """逐日执行级组合回测。

    时序（T 日收盘出信号 → T+1 开盘成交）：
      1. 开盘：执行上一交易日生成的委托（先卖后买，受现金约束）
      2. 日内：对每笔持仓按 OHLC 判定止损/止盈/移动止损（同日多触发按最坏假设）
      3. 收盘：市值重估、更新峰值；若为调仓日 → 生成次日委托
    """
    o, h, low, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
    vol = ohlc.get("volume")
    if vol is None:
        vol = pd.DataFrame(1.0, index=c.index, columns=c.columns)

    idx = score_panel.index.intersection(c.index)
    idx = idx.sort_values()
    codes = [x for x in score_panel.columns if x in c.columns]
    if len(codes) < cfg.top_n:
        logger.warning(f"可选标的 {len(codes)} < top_n {cfg.top_n}，回测跳过")
        return ExecResult(metrics={"n_trades": 0})

    S = score_panel.loc[idx, codes].astype(float)
    O = o.loc[idx, codes].astype(float)
    H = h.loc[idx, codes].astype(float)
    L = low.loc[idx, codes].astype(float)
    C = c.loc[idx, codes].astype(float)
    V = vol.loc[idx, codes].astype(float)

    atr = _atr_pct_panel(H, L, C, cfg.atr_window)
    can_buy, can_sell = _tradable_masks(O, H, L, C, V)

    # 风险平价用的历史波动（只用当日及之前，walk-forward）
    rets = C.pct_change()
    vol60 = rets.rolling(60, min_periods=20).std()

    Oa, Ha, La, Ca = O.to_numpy(), H.to_numpy(), L.to_numpy(), C.to_numpy()
    ATRa = atr.to_numpy()
    canB, canS = can_buy.to_numpy(), can_sell.to_numpy()
    VOL60 = vol60.to_numpy()
    Sa = S.to_numpy()
    cidx = {code: j for j, code in enumerate(codes)}
    dates = list(idx)
    n_days = len(dates)

    # 市场代理：面板内全部标的的等权日收益累乘成指数（不含未来信息）。
    # 用「宽度 + 趋势」判弱市，比单看某只指数更贴近本策略的实际持仓域。
    gate_ok: Optional[np.ndarray] = None
    if cfg.market_gate != "off":
        ew = C.pct_change().mean(axis=1).fillna(0.0)
        mkt = (1.0 + ew).cumprod()
        win = 20 if cfg.market_gate == "ma20" else 60
        ma = mkt.rolling(win, min_periods=max(5, win // 2)).mean()
        gate_ok = (mkt > ma).to_numpy()
        logger.debug(f"市场闸门 {cfg.market_gate}：强势日占比 {np.nanmean(gate_ok):.1%}")

    slip = cfg.slippage_bp / 10000.0
    capital = float(cfg.capital)
    cash = capital
    pos: Dict[str, dict] = {}
    pending: List[Tuple[str, str, float]] = []   # (side, code, value)
    trades: List[dict] = []
    equity_curve = np.full(n_days, np.nan)
    turnover_amt = 0.0
    fees_paid = 0.0
    run_peak = float(capital)      # 峰谷回撤基准（收盘口径）
    n_halt_days = 0                # 被熔断冻结的天数（诊断「换了个策略」用）

    def _fee(amount: float, is_sell: bool) -> float:
        f = max(cfg.min_commission, amount * cfg.commission)
        if is_sell:
            f += amount * cfg.stamp_tax
        return f

    def _close_pos(code: str, price: float, i: int, reason: str) -> None:
        nonlocal cash, fees_paid
        p = pos.pop(code, None)
        if p is None or p["qty"] <= 0:
            return
        qty = p["qty"]
        amount = qty * price
        fee = _fee(amount, True)
        cash += amount - fee
        fees_paid += fee
        cost = p["entry_px"] * qty
        pnl = amount - fee - cost
        trades.append({
            "code": code, "entry_i": p["entry_i"], "exit_i": i,
            "entry_date": dates[p["entry_i"]], "exit_date": dates[i],
            "entry_px": p["entry_px"], "exit_px": price,
            "qty": qty, "pnl": pnl,
            "pnl_pct": pnl / cost if cost else 0.0,
            "hold_days": i - p["entry_i"], "reason": reason,
        })

    def _stop_width(j: int, i: int, entry_width: Optional[float] = None) -> float:
        """止损宽度(%)。atr 模式：max(|静态|, k×ATR) 夹 [min,max]，**只增不减**。"""
        if cfg.stop_mode == "none":
            return 0.0
        base = abs(float(cfg.stop_loss_pct))
        if cfg.stop_mode == "fixed":
            return base
        a = ATRa[i, j]
        if not np.isfinite(a):
            return base
        w = max(base, cfg.atr_k * a)
        w = min(max(w, cfg.atr_min_pct), cfg.atr_max_pct)
        if entry_width is not None:
            w = max(w, entry_width)          # 只增不减
        return w

    def _equity(i: int) -> float:
        mv = 0.0
        for code, p in pos.items():
            j = cidx[code]
            px = Ca[i, j]
            if not np.isfinite(px):
                px = p["last_px"]
            p["last_px"] = px
            mv += p["qty"] * px
        return cash + mv

    for i in range(n_days):
        # ---------- 1) 开盘执行上一日委托 ----------
        if pending:
            opens = Oa[i]
            for side, code, value in pending:
                j = cidx[code]
                px = opens[j]
                if not np.isfinite(px) or px <= 0:
                    continue
                if side == "SELL":
                    if not canS[i, j] or code not in pos:
                        continue
                    _close_pos(code, px * (1 - slip), i, "rebalance")
                else:
                    if not canB[i, j] or code in pos:
                        continue
                    exec_px = px * (1 + slip)
                    budget = min(value, cash)
                    qty = int(budget / exec_px // LOT) * LOT
                    if qty <= 0:
                        continue
                    amount = qty * exec_px
                    fee = _fee(amount, False)
                    if amount + fee > cash:
                        continue
                    cash -= amount + fee
                    fees_paid += fee
                    turnover_amt += amount
                    w0 = _stop_width(j, i)
                    pos[code] = {
                        "qty": qty, "entry_px": exec_px, "entry_i": i,
                        "peak": exec_px, "last_px": exec_px,
                        "width": w0,
                    }
            pending = []

        # ---------- 2) 日内退出 ----------
        for code in list(pos.keys()):
            p = pos[code]
            j = cidx[code]
            oi, hi, li, ci = Oa[i, j], Ha[i, j], La[i, j], Ca[i, j]
            if not np.isfinite(oi) or not np.isfinite(li):
                continue
            entry = p["entry_px"]
            # ATR 只增不减：宽度随当日 ATR 放宽
            if cfg.stop_mode == "atr":
                p["width"] = _stop_width(j, i, p["width"])

            exit_px, reason = None, ""
            # ① 止损（最坏假设优先）：跳空直接用开盘价
            if cfg.stop_mode != "none" and p["width"] > 0:
                stop_px = entry * (1 - p["width"] / 100.0)
                if oi <= stop_px:
                    exit_px, reason = oi, "stop_loss"
                elif li <= stop_px:
                    exit_px, reason = stop_px, "stop_loss"
            # ② 止盈
            if exit_px is None and cfg.take_profit_pct > 0:
                tp_px = entry * (1 + cfg.take_profit_pct / 100.0)
                if oi >= tp_px:
                    exit_px, reason = oi, "stop_profit"
                elif hi >= tp_px:
                    exit_px, reason = tp_px, "stop_profit"
            # ③ 移动止损（仅在曾浮盈后跟踪峰值）
            if exit_px is None and cfg.trailing_stop_pct < 0:
                p["peak"] = max(p["peak"], hi)
                if p["peak"] > entry:
                    trail_px = p["peak"] * (1 + cfg.trailing_stop_pct / 100.0)
                    if li <= trail_px:
                        exit_px = min(oi, trail_px) if oi < trail_px else trail_px
                        reason = "trail_stop"
            # ④ 时间止损
            if exit_px is None and cfg.hold_days_max > 0 and (i - p["entry_i"]) >= cfg.hold_days_max:
                exit_px, reason = ci, "time_exit"

            if exit_px is not None and np.isfinite(exit_px) and canS[i, j]:
                _close_pos(code, exit_px * (1 - slip), i, reason)

        # ---------- 3) 收盘重估 + 调仓信号 ----------
        eq = _equity(i)
        equity_curve[i] = eq
        # 峰谷回撤（收盘口径）。实盘峰值按 tick 采样，通常比收盘价更高 ⇒
        # 实盘回撤只会比这里更深、熔断更早 ⇒ 下面的结论是**下界**。
        run_peak = max(run_peak, eq)
        dd = (1.0 - eq / run_peak) * 100.0 if run_peak > 0 else 0.0
        _halted = bool(cfg.halt_dd_pct > 0 and dd >= cfg.halt_dd_pct)
        if _halted:
            n_halt_days += 1

        if i % cfg.rebalance_days == 0 and i + 1 < n_days and not _halted:
            srow = Sa[i]
            ok = np.isfinite(srow)
            if ok.sum() >= cfg.top_n:
                order = np.argsort(-np.where(ok, srow, -np.inf))[: cfg.top_n]
                top_picks = [codes[j] for j in order]
                # 全截面排名（排名滞后要用完整榜单，不能只看前 top_n）
                full_order = np.argsort(-np.where(ok, srow, -np.inf))
                rank = {codes[j]: r for r, j in enumerate(full_order)}
                picks = top_picks
                # 排名滞后：把仍在前 top_n×slack 内的持仓留在组合里，
                # 腾出名额给新进榜的标的 —— 降低「第 9 名换第 8 名」的无谓换手。
                if cfg.hold_rank_slack > 1.0 and pos:
                    keep_rank = int(round(cfg.top_n * cfg.hold_rank_slack))
                    held = [c_ for c_ in pos if c_ in rank]
                    held.sort(key=lambda c_: rank[c_])
                    kept = [c_ for c_ in held if rank[c_] < keep_rank][: cfg.top_n]
                    merged = list(kept)
                    for c_ in top_picks:
                        if len(merged) >= cfg.top_n:
                            break
                        if c_ not in merged:
                            merged.append(c_)
                    picks = merged
                # 信号分位门槛（注意：top_n 已是前 4%，q 需 >0.96 才会生效）
                if cfg.min_score_quantile > 0 and ok.sum() >= 10:
                    thr = float(np.quantile(srow[ok], cfg.min_score_quantile))
                    picks = [c_ for c_ in picks if srow[cidx[c_]] >= thr]
                # 信号强度门槛（z 分数）：弱信号宁可不买
                if cfg.min_score_z > 0 and ok.sum() >= 10:
                    mu = float(np.nanmean(srow[ok]))
                    sd = float(np.nanstd(srow[ok]))
                    if sd > 1e-12:
                        picks = [c_ for c_ in picks if (srow[cidx[c_]] - mu) / sd >= cfg.min_score_z]
                invest = eq * (1.0 - cfg.cash_reserve_pct)
                # 市场状态闸门：弱市只降仓，不空仓（避免「一次误判错过整段行情」）
                if gate_ok is not None and not gate_ok[i]:
                    invest *= float(cfg.market_gate_scale)
                # 波动率目标：按组合自身已实现波动反推敞口（不实现亏损，只缩敞口）
                if cfg.vol_target_pct > 0 and i >= cfg.vol_window:
                    seg = equity_curve[i - cfg.vol_window:i + 1]
                    seg = seg[np.isfinite(seg) & (seg > 0)]
                    if len(seg) >= cfg.vol_window // 2:
                        r = np.diff(seg) / seg[:-1]
                        rv = float(np.std(r, ddof=1)) * np.sqrt(cfg.ppy) * 100.0
                        if rv > 1e-6:
                            scale = float(cfg.vol_target_pct) / rv
                            scale = min(1.0, max(float(cfg.vol_scale_floor), scale))
                            invest *= scale
                if not picks:
                    continue
                # 渐进降仓（实盘 _portfolio_risk_scale 的等价物）：回撤越深，新仓越小。
                # 注意它缩放的是**整个组合的敞口**，不是「这一笔」—— 与实盘一致。
                if cfg.dd_scale_floor > 0:
                    _start = max(float(getattr(cfg, "dd_start_cut_pct", 0.0)), 0.0)
                    _span = max(float(cfg.dd_scale_at_pct) - _start, 1e-9)
                    _prog = min(max(dd - _start, 0.0) / _span, 1.0)
                    invest *= max(float(cfg.dd_scale_floor),
                                  1.0 - (1.0 - float(cfg.dd_scale_floor)) * _prog)
                # 权重
                if cfg.weighting == "risk_parity":
                    iv = []
                    for code in picks:
                        v = VOL60[i, cidx[code]]
                        iv.append(1.0 / v if np.isfinite(v) and v > 1e-6 else 0.0)
                    s = sum(iv)
                    w = {picks[k]: (iv[k] / s if s > 0 else 1.0 / len(picks))
                         for k in range(len(picks))}
                else:
                    w = {code: 1.0 / len(picks) for code in picks}
                # 单票封顶 + 风险预算
                caps = {}
                for code in picks:
                    j = cidx[code]
                    cap = w[code]
                    if cfg.weight_cap < 1.0:
                        cap = min(cap, cfg.weight_cap)
                    if cfg.risk_budget_pct > 0:
                        width = (_stop_width(j, i) if cfg.stop_mode != "none" else abs(cfg.stop_loss_pct))
                        # 上限口径与止损模式无关：风险预算要防的是「这笔最坏能亏多少」，
                        # 而 ATR 会把宽度放宽到 atr_max_pct。只在 atr 模式下加上限
                        # 会让 no_exit/fixed 模式的预算永不生效（实测 rb 0.75/1.0/1.5%
                        # 三档结果逐位相同 = 根本没约束）。
                        if cfg.risk_budget_use_ceiling:
                            width = max(width, cfg.atr_max_pct)
                        width = max(width, 1e-4)
                        cap = min(cap, cfg.risk_budget_pct / (width / 100.0))
                    caps[code] = cap
                tw = {code: caps[code] for code in picks}

                # 当前权重
                cur = {}
                for code, p in pos.items():
                    j = cidx[code]
                    px = Ca[i, j]
                    if np.isfinite(px) and eq > 0:
                        cur[code] = p["qty"] * px / eq
                # 容忍带
                allc = set(tw) | set(cur)
                if cfg.band_mode == "rel":
                    dev = max([abs(tw.get(x, 0.0) - cur.get(x, 0.0))
                               / max(tw.get(x, 0.0), cur.get(x, 0.0), 1e-9)
                               for x in allc] or [0.0])
                else:
                    dev = max([abs(tw.get(x, 0.0) - cur.get(x, 0.0)) for x in allc] or [0.0])
                # 空仓时必须建仓：容忍带是「已有仓位的小偏离不用管」，
                # 不是「从零开始也不许买」。缺了这个条件，小仓位组合会永久空转。
                if cfg.band_pct <= 0 or not cur or dev > cfg.band_pct:
                    for code in list(cur):
                        if code not in tw:
                            pending.append(("SELL", code, 0.0))
                    for code in picks:
                        tgt_v = tw[code] * invest
                        cur_v = cur.get(code, 0.0) * eq
                        if tgt_v - cur_v > max(1.0, 0.005 * eq):
                            pending.append(("BUY", code, tgt_v - cur_v))

    if verbose:
        logger.info(f"执行回测完成：{n_days} 个交易日，{len(trades)} 笔平仓")

    eq_series = pd.Series(equity_curve, index=pd.DatetimeIndex(dates)).dropna()
    m = _metrics_from_equity(eq_series, cfg)
    tr = pd.DataFrame(trades)
    if len(tr):
        wins = tr[tr["pnl"] > 0]
        losses = tr[tr["pnl"] < 0]
        m.update({
            "n_trades": float(len(tr)),
            "win_rate": float(len(wins) / len(tr)),
            "avg_win_pct": float(wins["pnl_pct"].mean()) if len(wins) else 0.0,
            "avg_loss_pct": float(losses["pnl_pct"].mean()) if len(losses) else 0.0,
            "payoff": float(abs(wins["pnl_pct"].mean() / losses["pnl_pct"].mean()))
                      if len(wins) and len(losses) and losses["pnl_pct"].mean() != 0 else float("nan"),
            "profit_factor": float(wins["pnl"].sum() / abs(losses["pnl"].sum()))
                             if len(losses) and losses["pnl"].sum() != 0 else float("nan"),
            "avg_hold_days": float(tr["hold_days"].mean()),
        })
        by = tr.groupby("reason").agg(
            n=("pnl", "size"), pnl=("pnl", "sum"),
            avg_pct=("pnl_pct", "mean"), win=("pnl", lambda s: float((s > 0).mean())),
        ).sort_values("pnl")
        exit_stats = by
    else:
        m.update({"n_trades": 0.0})
        exit_stats = pd.DataFrame()

    years = max(n_days / cfg.ppy, 1e-9)
    m["turnover_per_year"] = float(turnover_amt / capital / years)
    m["fee_pct"] = float(fees_paid / capital)
    m["halt_days_pct"] = float(n_halt_days / max(n_days, 1))
    return ExecResult(metrics=m, equity=eq_series, trades=tr, exit_stats=exit_stats)

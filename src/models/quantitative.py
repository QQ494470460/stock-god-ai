# -*- coding: utf-8 -*-
"""
量化模型模块
多因子量化选股 + 均值回归 + 趋势跟踪 + 风险平价
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class QuantScore:
    """量化评分结果"""
    code: str
    name: str
    total_score: float          # 综合评分 0-100
    value_score: float          # 价值因子
    momentum_score: float       # 动量因子
    quality_score: float        # 质量因子
    volatility_score: float     # 低波动因子
    trend_score: float          # 趋势强度
    reversal_score: float       # 均值回归信号
    sharpe: float               # 夏普比率
    max_drawdown: float         # 最大回撤
    win_rate: float             # 胜率
    signal: str                 # 做多 / 观望 / 做空
    factors_detail: dict = None # 各因子详细值


class QuantEngine:
    """量化引擎 — 多因子模型 + 策略集"""
    
    FACTOR_WEIGHTS = {
        "value": 0.20,       # 价值
        "momentum": 0.25,    # 动量
        "quality": 0.20,     # 质量
        "volatility": 0.20,  # 低波
        "trend": 0.15,       # 趋势
    }
    
    def __init__(self):
        self._min_bars = 120  # 最少需要120个交易日
        logger.info("QuantEngine initialized")
    
    # ========== 因子计算 ==========
    
    def compute_value_factor(self, df: pd.DataFrame) -> float:
        """价值因子：低估值股票得分高
        - PE/PB 分位数越低越好
        - 用股价相对位置替代(免费数据无基本面)
        """
        close = df["close"]
        n = len(close)
        
        # 当前价相对N日高/低位
        h252 = close.rolling(252).max().iloc[-1] if n >= 252 else close.max()
        l252 = close.rolling(252).min().iloc[-1] if n >= 252 else close.min()
        c = close.iloc[-1]
        
        # 位置越低 = 价值越高
        position = (c - l252) / (h252 - l252 + 1e-10)
        value_score = (1 - position) * 100  # 低位=高分
        
        return np.clip(value_score, 0, 100)
    
    def compute_momentum_factor(self, df: pd.DataFrame) -> float:
        """动量因子：中期趋势强度
        - 多周期收益加权
        """
        close = df["close"]
        c = close.iloc[-1]
        
        rets = {}
        for days in [5, 10, 21, 63, 126]:
            if len(close) > days:
                rets[days] = (c / close.iloc[-(days+1)] - 1) * 100
        
        # 加权 (近期权重大)
        weights = {5: 0.35, 10: 0.25, 21: 0.20, 63: 0.12, 126: 0.08}
        momentum = sum(rets.get(d, 0) * w for d, w in weights.items())
        
        # 映射到 0-100 (正动量=高分)
        momentum_score = 50 + momentum * 1.5
        return np.clip(momentum_score, 0, 100)
    
    def compute_quality_factor(self, df: pd.DataFrame) -> float:
        """质量因子：盈利稳定性、波动结构
        - 免费数据替代方案：价格稳定性 + 趋势一致性
        """
        close = df["close"]
        volume = df["volume"]
        
        rets = close.pct_change().dropna()
        if len(rets) < 20:
            return 50
        
        # 1. 正收益天数占比
        up_days = (rets > 0).sum() / len(rets) * 100
        
        # 2. 收益变异系数 (越低越稳定)
        ret_cv = rets.std() / (rets.mean() + 1e-10)
        cv_score = max(0, 100 - abs(ret_cv) * 0.2)
        
        # 3. 量价关系：缩量调整是好信号
        vol_chg = volume.iloc[-10:].mean() / volume.iloc[-60:].mean()
        vol_score = 100 if 0.8 < vol_chg < 1.5 else 50
        
        quality = up_days * 0.3 + cv_score * 0.4 + vol_score * 0.3
        return np.clip(quality, 0, 100)
    
    def compute_volatility_factor(self, df: pd.DataFrame) -> float:
        """低波动因子：低波动=高分"""
        close = df["close"]
        rets = close.pct_change().dropna()
        
        if len(rets) < 20:
            return 50
        
        # 年化波动率
        annual_vol = rets.iloc[-63:].std() * np.sqrt(252) if len(rets) >= 63 else rets.std() * np.sqrt(252)
        
        # 波动率分位数 (0.15-0.55 映射到 0-100)
        vol_score = max(0, 100 - annual_vol * 200)
        return np.clip(vol_score, 0, 100)
    
    def compute_trend_strength(self, df: pd.DataFrame) -> float:
        """趋势强度 (ADX简化版)
        - 0-100，越高趋势越明显
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # 简版ADX (14周期)
        n = 14
        tr = pd.DataFrame({
            "h_l": high - low,
            "h_pc": abs(high - close.shift(1)),
            "l_pc": abs(low - close.shift(1))
        }).max(axis=1)
        
        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)
        
        atr = tr.ewm(span=n, adjust=False).mean()
        plus_di = (plus_dm.ewm(span=n, adjust=False).mean() / atr) * 100
        minus_di = (minus_dm.ewm(span=n, adjust=False).mean() / atr) * 100
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        adx = dx.ewm(span=n, adjust=False).mean()
        
        return np.clip(float(adx.iloc[-1]), 0, 100)
    
    def compute_reversal_signal(self, df: pd.DataFrame) -> Tuple[float, str]:
        """均值回归信号
        - 基于布林带 + RSI 超买超卖
        Returns: (strength 0-100, direction: "oversold"/"overbought"/"neutral")
        """
        close = df["close"]
        c = close.iloc[-1]
        
        # 布林带
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(span=14, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / (loss + 1e-10))
        
        rsi_val = float(rsi.iloc[-1])
        bb_val = float(bb_lower.iloc[-1]), float(bb_mid.iloc[-1]), float(bb_upper.iloc[-1])
        c_val = float(c)
        
        # 超卖信号
        if rsi_val < 30 and c_val < bb_val[0]:
            bb_strength = (bb_val[0] - c_val) / (bb_val[2] - bb_val[0] + 1e-10) * 100
            return min(bb_strength + (30 - rsi_val) * 1.5, 100), "超卖反弹"
        
        # 超买信号
        if rsi_val > 70 and c_val > bb_val[2]:
            bb_strength = (c_val - bb_val[2]) / (bb_val[2] - bb_val[0] + 1e-10) * 100
            return min(bb_strength + (rsi_val - 70) * 1.5, 100), "超买回落"
        
        return 0, "正常"
    
    def compute_sharpe_ratio(self, df: pd.DataFrame, rf: float = 0.03) -> float:
        """计算滚动夏普比率"""
        rets = df["close"].pct_change().dropna()
        if len(rets) < 63:
            return 0
        
        # 用近63日（一个季度）滚动
        recent = rets.iloc[-63:]
        annual_ret = recent.mean() * 252
        annual_vol = recent.std() * np.sqrt(252)
        
        if annual_vol < 0.001:
            return 0
        return (annual_ret - rf) / annual_vol
    
    def compute_max_drawdown(self, df: pd.DataFrame) -> float:
        """计算最大回撤"""
        close = df["close"]
        rolling_max = close.expanding().max()
        drawdown = (close - rolling_max) / rolling_max
        return float(drawdown.min())
    
    def compute_win_rate(self, df: pd.DataFrame, horizon: int = 5) -> float:
        """计算滚动胜率（未来N日涨跌预测准确率估算）"""
        close = df["close"]
        rets = close.pct_change(horizon).shift(-horizon).dropna()
        if len(rets) < 10:
            return 0
        return float((rets > 0).sum() / len(rets))
    
    # ========== 综合评分 ==========
    
    def score_stock(self, df: pd.DataFrame, code: str, name: str = "") -> QuantScore:
        """
        对单只股票进行多因子量化评分
        """
        if len(df) < self._min_bars:
            return QuantScore(
                code=code, name=name, total_score=0,
                value_score=0, momentum_score=0, quality_score=0,
                volatility_score=0, trend_score=0, reversal_score=0,
                sharpe=0, max_drawdown=0, win_rate=0,
                signal="数据不足"
            )
        
        # 各因子得分
        value = self.compute_value_factor(df)
        momentum = self.compute_momentum_factor(df)
        quality = self.compute_quality_factor(df)
        volatility = self.compute_volatility_factor(df)
        trend = self.compute_trend_strength(df)
        reversal, rev_dir = self.compute_reversal_signal(df)
        
        # 风险指标
        sharpe = self.compute_sharpe_ratio(df)
        max_dd = self.compute_max_drawdown(df)
        win_rate = self.compute_win_rate(df)
        
        # 加权综合
        total = (
            value * self.FACTOR_WEIGHTS["value"]
            + momentum * self.FACTOR_WEIGHTS["momentum"]
            + quality * self.FACTOR_WEIGHTS["quality"]
            + volatility * self.FACTOR_WEIGHTS["volatility"]
            + trend * self.FACTOR_WEIGHTS["trend"]
        )
        
        # 反转信号调整
        if rev_dir == "超卖反弹":
            total += 5
        elif rev_dir == "超买回落":
            total -= 5
        
        # 风险惩罚
        sharpe_adj = max(-10, min(10, (sharpe - 0.5) * 5))
        total += sharpe_adj
        
        # 回撤过大惩罚
        if max_dd < -0.3:
            total -= 8
        
        total = np.clip(total, 0, 100)
        
        # 信号判断
        if total >= 75:
            signal = "强力做多"
        elif total >= 60:
            signal = "建议做多"
        elif total >= 45:
            signal = "谨慎关注"
        elif total >= 30:
            signal = "中性观望"
        else:
            signal = "建议回避"
        
        return QuantScore(
            code=code, name=name, total_score=round(total, 1),
            value_score=round(value, 1), momentum_score=round(momentum, 1),
            quality_score=round(quality, 1), volatility_score=round(volatility, 1),
            trend_score=round(trend, 1), reversal_score=round(reversal, 1),
            sharpe=round(sharpe, 2), max_drawdown=round(max_dd, 3),
            win_rate=round(win_rate, 3),
            signal=signal,
            factors_detail={
                "价值": f"{value:.1f}",
                "动量": f"{momentum:.1f}",
                "质量": f"{quality:.1f}",
                "低波": f"{volatility:.1f}",
                "趋势": f"{trend:.1f}",
                "反转": f"{rev_dir}({reversal:.1f})" if reversal > 0 else "无",
                "夏普": f"{sharpe:.2f}",
                "最大回撤": f"{max_dd:.1%}",
                "胜率": f"{win_rate:.1%}",
            }
        )
    
    def batch_score(self, df_map: Dict[str, Tuple[pd.DataFrame, str]], 
                    top_n: int = 30) -> List[QuantScore]:
        """批量多因子评分"""
        scores = []
        for code, (df, name) in df_map.items():
            try:
                s = self.score_stock(df, code, name)
                if s.signal != "数据不足":
                    scores.append(s)
            except Exception as e:
                logger.warning(f"量化评分 {code} 失败: {e}")
        
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores[:top_n]


# ========== 均值回归策略 ==========

class MeanReversionStrategy:
    """均值回归交易策略"""
    
    def __init__(self, lookback: int = 20, entry_threshold: float = 2.0, 
                 exit_threshold: float = 0.5):
        self.lookback = lookback
        self.entry_threshold = entry_threshold   # 布林带偏离倍数进
        self.exit_threshold = exit_threshold     # 回归到多少σ离场
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成均值回归交易信号
        Returns: DataFrame with 'signal' column (-1=卖, 0=持有, 1=买)
        """
        close = df["close"]
        result = df[["date"]].copy()
        result["close"] = close
        result["signal"] = 0
        result["position"] = 0.0
        
        # 布林带
        mid = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        
        # Z-score
        z = (close - mid) / (std + 1e-10)
        
        pos = 0
        for i in range(self.lookback, len(close)):
            if pos == 0:
                if z.iloc[i] < -self.entry_threshold:
                    result.iloc[i, result.columns.get_loc("signal")] = 1
                    pos = 1
                elif z.iloc[i] > self.entry_threshold:
                    result.iloc[i, result.columns.get_loc("signal")] = -1
                    pos = -1
            elif pos > 0:  # 持有多头
                if z.iloc[i] > -self.exit_threshold:
                    result.iloc[i, result.columns.get_loc("signal")] = -1
                    pos = 0
            else:  # 持有空头
                if z.iloc[i] < self.exit_threshold:
                    result.iloc[i, result.columns.get_loc("signal")] = 1
                    pos = 0
            
            result.iloc[i, result.columns.get_loc("position")] = pos
        
        return result
    
    def backtest(self, df: pd.DataFrame, capital: float = 100000) -> dict:
        """回测均值回归策略"""
        signals = self.generate_signals(df)
        close = df["close"].values
        
        cash = capital
        shares = 0
        equity_curve = []
        trades = []
        
        for i, row in signals.iterrows():
            if i < self.lookback:
                equity_curve.append(cash)
                continue
            
            signal = row["signal"]
            price = close[i]
            
            if signal == 1 and shares == 0:  # 买入
                shares = int(cash * 0.95 / price)
                cost = shares * price
                cash -= cost
                trades.append({"date": row["date"], "action": "buy", "price": price, "shares": shares})
            elif signal == -1 and shares > 0:  # 卖出
                cash += shares * price
                trades.append({"date": row["date"], "action": "sell", "price": price, "shares": shares})
                shares = 0
            
            equity = cash + shares * price
            equity_curve.append(equity)
        
        # 清仓
        if shares > 0:
            cash += shares * close[-1]
        
        final_equity = cash
        total_return = (final_equity / capital - 1) * 100
        buy_hold_return = (close[-1] / close[self.lookback-1] - 1) * 100
        
        return {
            "total_return": round(total_return, 2),
            "buy_hold_return": round(buy_hold_return, 2),
            "alpha": round(total_return - buy_hold_return, 2),
            "n_trades": len(trades) // 2,
            "max_drawdown": self._calc_max_dd(equity_curve),
            "trades": trades[-10:]  # 最近10笔
        }
    
    def _calc_max_dd(self, equity: List[float]) -> float:
        peak = equity[0]
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
        return round(max_dd * 100, 2)


# ========== 风险平价 ==========

class RiskParity:
    """风险平价资产配置"""
    
    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        """
        简单风险平价：权重反比于波动率
        returns: DataFrame (dates × stocks)
        """
        vols = returns.std()
        inv_vols = 1.0 / (vols + 1e-10)
        weights = inv_vols / inv_vols.sum()
        return weights
    
    def equal_risk_contribute(self, returns: pd.DataFrame, max_iter: int = 100) -> pd.Series:
        """风险平价 (等风险贡献) 简化版"""
        cov = returns.cov().values
        n = cov.shape[0]
        w = np.ones(n) / n  # 初始等权
        
        for _ in range(max_iter):
            # 边际风险贡献
            sigma = np.sqrt(w @ cov @ w)
            mrc = cov @ w / sigma
            
            # 风险贡献
            rc = w * mrc
            
            # 目标：所有资产风险贡献相等
            target_rc = sigma / n
            w = w * target_rc / (rc + 1e-10)
            w = w / w.sum()
            
            # 收敛检查
            if np.max(np.abs(rc / sigma - 1.0 / n)) < 0.01:
                break
        
        return pd.Series(w, index=returns.columns)


# 全局实例
quant_engine = QuantEngine()
mean_reversion = MeanReversionStrategy()
risk_parity = RiskParity()
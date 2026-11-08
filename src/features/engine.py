# -*- coding: utf-8 -*-
"""
特征工程模块
从行情数据中提取技术面、基本面、市场面多维特征
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from loguru import logger

from config.settings import config as app_config


class FeatureEngine:
    """特征工程引擎"""
    
    def __init__(self):
        self.feat_config = app_config.features
        logger.info(f"FeatureEngine initialized (technical={self.feat_config.use_technical}, "
                   f"fundamental={self.feat_config.use_fundamental})")
    
    # ============ 技术指标 ============
    
    def _calc_ma_features(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """均线特征"""
        if periods is None:
            periods = self.feat_config.ma_periods
        
        result = pd.DataFrame(index=df.index)
        
        for p in periods:
            ma = df["close"].rolling(window=p).mean()
            result[f"ma_{p}"] = ma
            # 收盘价相对均线的偏离度
            result[f"ma_{p}_bias"] = (df["close"] - ma) / ma * 100
            # 均线斜率
            result[f"ma_{p}_slope"] = ma.diff(p) / p
        
        # 多头/空头排列
        if len(periods) >= 3:
            result["ma_bullish"] = (
                (result[f"ma_{periods[0]}"] > result[f"ma_{periods[1]}"]) &
                (result[f"ma_{periods[1]}"] > result[f"ma_{periods[2]}"])
            ).astype(int)
        
        return result
    
    def _calc_macd_features(self, df: pd.DataFrame,
                           fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD指标特征"""
        result = pd.DataFrame(index=df.index)
        
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd_bar = 2 * (dif - dea)
        
        result["macd_dif"] = dif
        result["macd_dea"] = dea
        result["macd_bar"] = macd_bar
        result["macd_bar_ratio"] = macd_bar / df["close"]  # 标准化
        
        # 金叉/死叉
        result["macd_golden_cross"] = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        result["macd_death_cross"] = (dif < dea) & (dif.shift(1) >= dea.shift(1))
        
        # 背离
        result["macd_divergence"] = (df["close"] - df["close"].shift(20)) * (dif - dif.shift(20))
        
        return result.astype(float)
    
    def _calc_rsi_features(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """RSI指标特征"""
        result = pd.DataFrame(index=df.index)
        
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        result["rsi_14"] = rsi
        result["rsi_overbought"] = (rsi > 70).astype(int)
        result["rsi_oversold"] = (rsi < 30).astype(int)
        # 6日RSI也计算
        delta6 = df["close"].diff()
        avg_gain6 = delta6.clip(lower=0).ewm(span=6, adjust=False).mean()
        avg_loss6 = (-delta6).clip(lower=0).ewm(span=6, adjust=False).mean()
        rs6 = avg_gain6 / (avg_loss6 + 1e-10)
        result["rsi_6"] = 100 - (100 / (1 + rs6))
        
        return result.astype(float)
    
    def _calc_kdj_features(self, df: pd.DataFrame,
                          n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """KDJ指标特征"""
        result = pd.DataFrame(index=df.index)
        
        low_list = df["low"].rolling(window=n, min_periods=1).min()
        high_list = df["high"].rolling(window=n, min_periods=1).max()
        
        rsv = (df["close"] - low_list) / (high_list - low_list + 1e-10) * 100
        
        k = rsv.ewm(span=m1, adjust=False).mean()
        d = k.ewm(span=m2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        result["kdj_k"] = k
        result["kdj_d"] = d
        result["kdj_j"] = j
        # 金叉死叉
        result["kdj_golden_cross"] = (k > d) & (k.shift(1) <= d.shift(1))
        result["kdj_death_cross"] = (k < d) & (k.shift(1) >= d.shift(1))
        
        return result.astype(float)
    
    def _calc_boll_features(self, df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        """布林带特征"""
        result = pd.DataFrame(index=df.index)
        
        mid = df["close"].rolling(window=period).mean()
        std_val = df["close"].rolling(window=period).std()
        
        upper = mid + std * std_val
        lower = mid - std * std_val
        width = (upper - lower) / mid * 100
        
        result["boll_mid"] = mid
        result["boll_upper"] = upper
        result["boll_lower"] = lower
        result["boll_width"] = width
        # 价格在布林带中的位置（0=下轨，1=上轨）
        result["boll_position"] = (df["close"] - lower) / (upper - lower + 1e-10)
        # 带宽变化
        result["boll_width_change"] = width.diff(5)
        
        return result.astype(float)
    
    def _calc_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特征"""
        result = pd.DataFrame(index=df.index)
        
        # 量比（5日均量）
        vol_ma5 = df["volume"].rolling(5).mean()
        result["vol_ratio"] = df["volume"] / (vol_ma5 + 1e-10)
        
        # 成交量MA
        for p in [5, 10, 20]:
            vol_ma = df["volume"].rolling(p).mean()
            result[f"vol_ma_{p}"] = vol_ma
            result[f"vol_ratio_{p}"] = df["volume"] / (vol_ma + 1e-10)
        
        # 换手率特征
        if "turnover" in df.columns:
            result["turnover_ma5"] = df["turnover"].rolling(5).mean()
            result["turnover_ratio"] = df["turnover"] / (result["turnover_ma5"] + 1e-10)
        
        # OBV（能量潮）简化版
        if len(df) > 1:
            direction = np.sign(df["close"].diff().fillna(0))
            obv = (direction * df["volume"]).cumsum()
            result["obv"] = obv
            result["obv_change"] = obv.pct_change(5)
        
        return result.astype(float)
    
    def _calc_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格类特征"""
        result = pd.DataFrame(index=df.index)
        close = df["close"]
        
        # 收益率
        for p in [1, 3, 5, 10, 20]:
            result[f"ret_{p}d"] = close.pct_change(p)
        
        # 波动率
        for p in [5, 10, 20]:
            result[f"volatility_{p}d"] = close.pct_change().rolling(p).std()
        
        # 价格位置（N日高低中的位置）
        for p in [5, 10, 20, 60]:
            high = df["high"].rolling(p).max()
            low = df["low"].rolling(p).min()
            result[f"price_position_{p}d"] = (close - low) / (high - low + 1e-10)
        
        # 振幅
        if "amplitude" in df.columns:
            result["amplitude"] = df["amplitude"]
            result["amplitude_ma5"] = df["amplitude"].rolling(5).mean()
        
        # 连涨/连跌天数
        up_days = (close > close.shift(1)).astype(int)
        result["consecutive_up"] = up_days.groupby((up_days != up_days.shift(1)).cumsum()).cumsum()
        down_days = (close < close.shift(1)).astype(int)
        result["consecutive_down"] = down_days.groupby((down_days != down_days.shift(1)).cumsum()).cumsum()
        
        # 创N日新高/新低
        for p in [20, 60, 120]:
            result[f"new_high_{p}d"] = (close >= close.rolling(p).max().shift(1)).astype(int)
            result[f"new_low_{p}d"] = (close <= close.rolling(p).min().shift(1)).astype(int)
        
        return result.astype(float)
    
    # ============ 新增技术指标 (Task 1) ============
    
    def _calc_adx_features(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ADX趋势强度 + 方向指标（+DI/-DI）"""
        result = pd.DataFrame(index=df.index)
        
        high, low, close = df["high"], df["low"], df["close"]
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        # +DM / -DM
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
        
        # DX → ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        result["adx"] = adx
        result["adx_plus_di"] = plus_di
        result["adx_minus_di"] = minus_di
        result["adx_trend"] = (adx > 25).astype(int)  # ADX>25表示趋势明显
        result["adx_di_cross"] = ((plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))).astype(int)
        
        return result.astype(float)
    
    def _calc_atr_features(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ATR平均真实波幅 + 标准化ATR%"""
        result = pd.DataFrame(index=df.index)
        
        high, low, close = df["high"], df["low"], df["close"]
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        result["atr"] = atr
        result["atr_pct"] = atr / close * 100  # 标准化ATR%
        result["atr_change"] = atr.pct_change(period)  # ATR变化率
        
        return result.astype(float)
    
    def _calc_volume_price_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """VPT (Volume-Price Trend) 量价趋势指标"""
        result = pd.DataFrame(index=df.index)
        
        close = df["close"]
        volume = df["volume"]
        
        # VPT = 累计(价格变动率 × 成交量)
        price_change_pct = close.pct_change().fillna(0)
        vpt = (price_change_pct * volume).cumsum()
        
        result["vpt"] = vpt
        result["vpt_ma5"] = vpt.rolling(5).mean()
        result["vpt_change"] = vpt.diff(5)
        result["vpt_divergence"] = (close - close.shift(20)) * (vpt - vpt.shift(20))  # 量价背离
        
        return result.astype(float)
    
    def _calc_money_flow(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """简化MFI (Money Flow Index) 资金流量指标"""
        result = pd.DataFrame(index=df.index)
        
        high, low, close = df["high"], df["low"], df["close"]
        volume = df["volume"]
        
        # Typical Price
        typical_price = (high + low + close) / 3
        
        # Raw Money Flow
        raw_mf = typical_price * volume
        
        # Positive / Negative Flow
        tp_diff = typical_price.diff()
        pos_flow = raw_mf.where(tp_diff > 0, 0).rolling(period).sum()
        neg_flow = raw_mf.where(tp_diff < 0, 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + pos_flow / (neg_flow + 1e-10)))
        
        result["mfi"] = mfi
        result["mfi_overbought"] = (mfi > 80).astype(int)
        result["mfi_oversold"] = (mfi < 20).astype(int)
        result["mfi_divergence"] = (close - close.shift(period)) * (mfi - mfi.shift(period))  # 与价格背离
        
        return result.astype(float)
    
    # ============ 市场状态检测 (Task 2) ============
    
    def _calc_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """市场状态检测：牛市/熊市/震荡"""
        result = pd.DataFrame(index=df.index)
        
        close = df["close"]
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        
        # MA20斜率（5日变化率）
        ma20_slope = ma20.diff(5)
        
        # 状态判断
        bull = (close > ma60) & (ma20_slope > 0)
        bear = (close < ma60) & (ma20_slope < 0)
        sideways = ~(bull | bear)
        
        result["regime_bull"] = bull.astype(int)
        result["regime_bear"] = bear.astype(int)
        result["regime_sideways"] = sideways.astype(int)
        
        # 连续状态天数
        regime_num = bull.astype(int) * 2 + bear.astype(int) * 1  # 2=牛, 1=熊, 0=震荡
        result["regime_code"] = regime_num
        
        return result.astype(float)
    
    def _calc_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """市场面特征"""
        result = pd.DataFrame(index=df.index)
        
        # 成交额特征
        if "amount" in df.columns:
            result["amount_ma5"] = df["amount"].rolling(5).mean()
            result["amount_ratio"] = df["amount"] / (result["amount_ma5"] + 1e-10)
        
        # 量价关系
        ret_5 = df["close"].pct_change(5)
        vol_change_5 = df["volume"].pct_change(5)
        result["price_vol_corr"] = df["close"].pct_change().rolling(20).corr(
            df["volume"].pct_change()
        )
        
        # 上涨放量 / 下跌缩量 信号
        result["up_with_volume"] = ((ret_5 > 0) & (vol_change_5 > 0)).astype(int)
        result["down_with_shrink"] = ((ret_5 < 0) & (vol_change_5 < 0)).astype(int)
        
        return result.astype(float)
    
    # ============ 基本面特征 ============
    
    def _calc_fundamental_features(self, df: pd.DataFrame, financials: Optional[pd.DataFrame]) -> pd.DataFrame:
        """基本面特征。

        ⚠️ 防前视泄漏：财务数据必须按「报告期(date)」对齐到行情日期，对每个行情日
        取「该日及之前最近一次披露」的财务值（as-of / 前向填充）。绝不允许把最新财报
        值广播到全部历史行 —— 那会在训练 / 回测里用未来信息预测过去（look-ahead）。
        """
        result = pd.DataFrame(index=df.index)

        if financials is None or len(financials) == 0:
            # 无财务数据时，用0填充（避免全NaN导致样本丢弃）
            for col in self._get_fundamental_cols():
                result[col] = 0.0
            return result

        try:
            fin = financials.copy()
            if "date" in fin.columns:
                fin["date"] = pd.to_datetime(fin["date"], errors="coerce")
                fin = fin.dropna(subset=["date"]).sort_values("date").set_index("date")
                if "date" in df.columns:
                    df_dates = pd.to_datetime(df["date"], errors="coerce")
                    # 对每个行情日，取 <= 该日的最近一次财务值（as-of 前向填充）。
                    aligned = fin.reindex(df_dates, method="ffill")
                    for col in fin.columns:
                        if col != "code":
                            result[col] = aligned[col].values
                else:
                    # df 无日期列则无法按报告期对齐：不能泄漏最新值，退化全 0 并告警
                    logger.warning(
                        "基本面特征：行情数据缺少 date 列，无法按报告期对齐，"
                        "退化为 0（避免前视泄漏）")
                    for col in self._get_fundamental_cols():
                        result[col] = 0.0
            else:
                # 财务数据无 date 列：同样不能泄漏最新值，退化全 0 并告警
                logger.warning(
                    "基本面特征：财务数据缺少 date 列，无法按报告期对齐，"
                    "退化为 0（避免前视泄漏）")
                for col in self._get_fundamental_cols():
                    result[col] = 0.0
        except Exception as e:  # noqa: BLE001
            logger.warning(f"基本面特征对齐失败，退化为 0（避免前视泄漏）: {e}")
            for col in self._get_fundamental_cols():
                result[col] = 0.0

        # 补充缺失列
        for col in self._get_fundamental_cols():
            if col not in result.columns:
                result[col] = 0.0

        return result.astype(float)
    
    def _get_fundamental_cols(self) -> List[str]:
        """基本面特征列"""
        return [
            "pe", "pe_ttm", "pb", "ps",
            "roe", "roa", "gross_profit_margin", "net_profit_margin",
            "revenue_growth", "profit_growth",
            "debt_to_asset", "current_ratio",
            "eps", "bps",
        ]
    
    # ============ 特征聚合 ============
    
    def build_features(
        self,
        df: pd.DataFrame,
        financials: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        构建完整特征矩阵
        
        Args:
            df: 日K线DataFrame（需包含 open/close/high/low/volume 等列）
            financials: 财务数据DataFrame（可选）
        
        Returns:
            特征矩阵
        """
        logger.debug(f"构建特征: {len(df)} 条记录")
        
        feature_parts = []
        
        # 价格特征
        feature_parts.append(self._calc_price_features(df))
        
        # 均线特征
        feature_parts.append(self._calc_ma_features(df))
        
        # MACD
        feature_parts.append(self._calc_macd_features(df))
        
        # RSI
        feature_parts.append(self._calc_rsi_features(df))
        
        # KDJ
        feature_parts.append(self._calc_kdj_features(df))
        
        # 布林带
        feature_parts.append(self._calc_boll_features(df))
        
        # 成交量
        feature_parts.append(self._calc_volume_features(df))
        
        # ADX 趋势强度
        feature_parts.append(self._calc_adx_features(df))
        
        # ATR 波动指标
        feature_parts.append(self._calc_atr_features(df))
        
        # VPT 量价趋势
        feature_parts.append(self._calc_volume_price_trend(df))
        
        # MFI 资金流量
        feature_parts.append(self._calc_money_flow(df))
        
        # 市场状态
        feature_parts.append(self._calc_market_regime(df))
        
        # 市场面
        feature_parts.append(self._calc_market_features(df))
        
        # 基本面
        if self.feat_config.use_fundamental:
            feature_parts.append(self._calc_fundamental_features(df, financials))
        
        # 拼接所有特征
        features = pd.concat(feature_parts + [df[["close"]]], axis=1)  # 保留close用于标签生成
        
        # 去除无穷值和极端异常值
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.clip(lower=-1e6, upper=1e6)
        
        logger.debug(f"特征构建完成: {features.shape[1]} 个特征列")
        return features
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        forecast_days: int = None,
        threshold_up: float = None,
        threshold_down: float = None
    ) -> pd.Series:
        """
        生成训练标签
        
        Args:
            df: 包含close列的DataFrame
            forecast_days: 预测天数
            threshold_up: 上涨阈值
            threshold_down: 下跌阈值
        
        Returns:
            标签Series (1=上涨, 0=震荡, -1=下跌)
        """
        if forecast_days is None:
            forecast_days = app_config.ml.forecast_days
        if threshold_up is None:
            threshold_up = app_config.ml.threshold_up
        if threshold_down is None:
            threshold_down = app_config.ml.threshold_down
        
        future_ret = df["close"].shift(-forecast_days) / df["close"] - 1
        
        labels = pd.Series(0, index=df.index, dtype=int)
        labels[future_ret > threshold_up] = 1
        labels[future_ret < threshold_down] = -1
        
        # 二分类模式：去掉震荡样本
        if app_config.ml.binary_mode:
            labels[labels == 0] = np.nan
        
        # 最后N天没有未来数据，标记为NaN
        labels.iloc[-forecast_days:] = np.nan
        
        return labels
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        financials: Optional[pd.DataFrame] = None,
        forecast_days: int = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备完整训练数据：特征 + 标签
        
        Returns:
            (features, labels)
        """
        features = self.build_features(df, financials)
        labels = self.generate_labels(df, forecast_days)
        
        # 移除close列（不放入模型）
        if "close" in features.columns:
            features = features.drop(columns=["close"])
        
        # 去掉NaN
        valid_mask = features.notna().all(axis=1) & labels.notna()
        features = features[valid_mask].copy()
        labels = labels[valid_mask].copy()
        
        logger.info(f"训练数据准备完成: {len(features)} 条样本, {features.shape[1]} 个特征")
        return features, labels


# 全局单例
feature_engine = FeatureEngine()
# -*- coding: utf-8 -*-
"""
数据获取模块 - 支持 AKShare / Tushare
获取A股日线行情、财务数据、实时行情等
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

import akshare as ak

from config.settings import config as app_config, DATA_DIR


@dataclass
class StockInfo:
    """股票基本信息"""
    code: str          # 代码（如 000001）
    name: str          # 名称（如 平安银行）
    market: str        # 市场（sh/sz）
    industry: str      # 行业
    area: str          # 地区
    listed_date: str   # 上市日期
    total_cap: float   # 总市值（亿）
    float_cap: float   # 流通市值（亿）


class DataFetcher:
    """数据获取器"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stock_list_cache: Optional[pd.DataFrame] = None
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        logger.info("DataFetcher initialized")
    
    # ========== 股票列表 ==========
    
    def get_stock_list(self, force_refresh: bool = False) -> pd.DataFrame:
        """获取A股全市场股票列表"""
        if self._stock_list_cache is not None and not force_refresh:
            return self._stock_list_cache
        
        cache_file = self.cache_dir / "stock_list.parquet"
        if cache_file.exists() and not force_refresh:
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < app_config.data.cache_days * 86400:
                logger.info("从缓存加载股票列表")
                self._stock_list_cache = pd.read_parquet(cache_file)
                return self._stock_list_cache
        
        logger.info("正在获取A股全市场股票列表...")
        try:
            # AKShare获取A股列表
            df = ak.stock_info_a_code_name()
            df = df.rename(columns={"code": "code", "name": "name"})
            
            # 过滤：只保留沪深主板、创业板、科创板、北交所
            df = df[df["code"].str.match(r"^\d{6}$")].copy()
            df["market"] = df["code"].apply(lambda x: "sh" if x.startswith(("6", "9")) else "sz")
            
            # 获取行业分类
            try:
                industry_df = ak.stock_board_industry_name_em()
                # 这里简化处理，实际可以通过stock_individual_info_em逐个获取
                df["industry"] = "未知"
                df["area"] = "未知"
            except Exception as e:
                logger.warning(f"获取行业分类失败: {e}")
                df["industry"] = "未知"
                df["area"] = "未知"
            
            df["listed_date"] = ""
            df["total_cap"] = 0.0
            df["float_cap"] = 0.0
            
            df.to_parquet(cache_file, index=False)
            self._stock_list_cache = df
            logger.info(f"获取到 {len(df)} 只A股股票")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            if self._stock_list_cache is not None:
                return self._stock_list_cache
            raise
    
    # ========== 日线行情 ==========
    
    def get_daily_kline(
        self,
        code: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        adjust: str = "qfq"  # 前复权
    ) -> pd.DataFrame:
        """获取单只股票日K线数据"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        # 统一日期格式
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        cache_key = f"{code}_{adjust}_{start}_{end}"
        
        if cache_key in self._daily_cache:
            return self._daily_cache[cache_key].copy()
        
        cache_file = self.cache_dir / f"daily_{code}_{adjust}.parquet"
        
        # 尝试从缓存加载
        if cache_file.exists():
            cached = pd.read_parquet(cache_file)
            if len(cached) > 0:
                cached["date"] = pd.to_datetime(cached["date"])
                cached = cached[(cached["date"] >= start_date) & (cached["date"] <= end_date)]
                if len(cached) > 50:  # 至少50条数据
                    self._daily_cache[cache_key] = cached
                    return cached.copy()
        
        # 从AKShare获取（新浪数据源，稳定可靠）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                wait = app_config.data.rate_limit * (attempt + 1)
                time.sleep(wait)  # 限速
                logger.debug(f"获取 {code} 日线数据... (尝试 {attempt+1}/{max_retries})")
                
                # 判断市场，构建symbol
                market = "sh" if code.startswith(("6", "9")) else "sz"
                symbol = f"{market}{code}"
                
                # 使用新浪数据源（stock_zh_a_daily），比东方财富更稳定
                df = ak.stock_zh_a_daily(symbol=symbol, adjust=adjust)
                
                if df is None or len(df) == 0:
                    logger.warning(f"{code} 无日线数据")
                    return pd.DataFrame()
                
                # 标准化列名（新浪数据源列名不同于东方财富）
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
                
                # 按日期过滤
                df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                
                if len(df) == 0:
                    logger.warning(f"{code} 指定日期范围内无数据")
                    return pd.DataFrame()
                
                # 确保必要列存在并补齐缺失字段
                if "pct_change" not in df.columns:
                    df["pct_change"] = df["close"].pct_change() * 100
                if "amplitude" not in df.columns:
                    df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
                if "change" not in df.columns:
                    df["change"] = df["close"].diff()
                if "turnover" not in df.columns:
                    df["turnover"] = 0.0
                
                # 清理NaN
                df = df.dropna(subset=["open", "close", "high", "low", "volume"])
                
                # 缓存
                df.to_parquet(cache_file, index=False)
                self._daily_cache[cache_key] = df
                
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = (attempt + 1) * 3
                    logger.warning(f"{code} 第{attempt+1}次失败，{backoff}s后重试: {e}")
                    time.sleep(backoff)
                else:
                    logger.error(f"获取 {code} 日线数据失败 (已重试{max_retries}次): {e}")
        
        return pd.DataFrame()
    
    def get_batch_daily_kline(
        self,
        codes: List[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        max_workers: int = 8
    ) -> Dict[str, pd.DataFrame]:
        """批量获取日线数据（多线程）"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_daily_kline, code, start_date, end_date): code
                for code in codes
            }
            
            total = len(futures)
            completed = 0
            for future in as_completed(futures):
                code = futures[future]
                completed += 1
                try:
                    df = future.result()
                    if len(df) > 0:
                        results[code] = df
                except Exception as e:
                    logger.error(f"获取 {code} 失败: {e}")
                
                if completed % 50 == 0:
                    logger.info(f"进度: {completed}/{total}")
        
        logger.info(f"批量获取完成: {len(results)}/{total} 只有效")
        return results
    
    # ========== 基本面数据 ==========
    
    def get_financial_indicators(self, code: str) -> pd.DataFrame:
        """获取财务指标（PE、PB、ROE等）"""
        try:
            time.sleep(app_config.data.rate_limit)
            df = ak.stock_financial_analysis_indicator(symbol=code)
            if df is not None and len(df) > 0:
                df["code"] = code
                return df
        except Exception as e:
            logger.debug(f"获取 {code} 财务指标失败: {e}")
        return pd.DataFrame()
    
    def get_realtime_quote(self, codes: List[str]) -> pd.DataFrame:
        """获取实时行情（盘中）"""
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and len(df) > 0:
                df = df[df["代码"].isin(codes)].copy()
                return df
        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
        return pd.DataFrame()
    
    # ========== 指数/板块 ==========
    
    def get_index_kline(
        self,
        index_code: str = "sh000001",
        start_date: str = "2015-01-01"
    ) -> pd.DataFrame:
        """获取指数K线（上证/深证/创业板等）"""
        try:
            df = ak.stock_zh_index_daily(symbol=index_code)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= start_date].copy()
            return df
        except Exception as e:
            logger.error(f"获取指数K线失败: {e}")
            return pd.DataFrame()
    
    def get_sector_hotspots(self) -> pd.DataFrame:
        """获取板块热点"""
        try:
            df = ak.stock_board_concept_name_em()
            return df.sort_values("涨跌幅", ascending=False)
        except Exception as e:
            logger.error(f"获取板块热点失败: {e}")
            return pd.DataFrame()
    
    # ========== 工具方法 ==========
    
    def get_stock_pool(self, pool: str = "all") -> List[str]:
        """获取指定股票池的代码列表"""
        stock_list = self.get_stock_list()
        
        # 过滤ST、退市等
        stock_list = stock_list[~stock_list["name"].str.contains("ST|退市|\\*ST", na=False)]
        
        if pool == "hs300":
            try:
                hs300 = ak.index_stock_cons("000300")
                codes = hs300["品种代码"].tolist()
                stock_list = stock_list[stock_list["code"].isin(codes)]
            except Exception:
                logger.warning("获取沪深300成分股失败，使用全量")
        
        elif pool == "zz500":
            try:
                zz500 = ak.index_stock_cons("000905")
                codes = zz500["品种代码"].tolist()
                stock_list = stock_list[stock_list["code"].isin(codes)]
            except Exception:
                logger.warning("获取中证500成分股失败，使用全量")
        
        return stock_list["code"].tolist()
    
    def clear_cache(self):
        """清理缓存"""
        self._stock_list_cache = None
        self._daily_cache = {}
        logger.info("缓存已清理")


# 全局单例
fetcher = DataFetcher()
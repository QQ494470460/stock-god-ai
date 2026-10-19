# -*- coding: utf-8 -*-
"""
数据获取模块 - 支持 同花顺 / AKShare
获取A股日线行情、财务数据、实时行情等
主数据源：同花顺 HTTP API (d.10jqka.com.cn)
"""

import time
import re
import json
import threading
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger

# 性能优化：akshare 顶层 import 约 1.9 秒，而本模块的主数据源是同花顺 HTTP，
# akshare 只在少量兜底/补充接口里才用。改成按需加载，启动阶段不再付这 1.9 秒。
_AK = None


def _ak():
    """按需加载并返回 akshare 模块（首次调用时才真正 import）"""
    global _AK
    if _AK is None:
        import akshare

        _AK = akshare
    return _AK


def _parse_cap(val: Any) -> float:
    """把东方财富返回的市值/股本解析成「亿」为单位的浮点数。

    东方财富 ``stock_individual_info_em`` 的 总市值/总股本 等字段通常是「元」量级的数值
    （如 1.6e11），少数情况可能带「亿/万」单位的字符串。统一归一到「亿」。
    """
    if val is None:
        return 0.0
    try:
        if isinstance(val, (int, float)):
            num = float(val)
        else:
            s = str(val).strip()
            if not s:
                return 0.0
            unit = 1.0
            if "亿" in s:
                unit = 1.0
                s = s.replace("亿", "")
            elif "万" in s:
                unit = 1e-4
                s = s.replace("万", "")
            num = float(s.replace(",", ""))
            num *= unit
        # 数值极大（元量级）→ 视为「元」，换算成「亿」
        if num > 1e6:
            num = num / 1e8
        return float(num)
    except Exception:
        return 0.0

from config.settings import config as app_config, DATA_DIR

# 同花顺复权码: 01=不复权, 02=前复权, 03=后复权
_THS_ADJUST_MAP = {"qfq": "02", "hfq": "03", "": "01"}

# 主界面大盘看板展示的指数（顺序即展示顺序）
# code -> (东财 market_id, 腾讯 symbol)
_MAIN_INDICES = {
    "000001": ("1", "sh000001"),   # 上证指数
    "399001": ("0", "sz399001"),   # 深证成指
    "399006": ("0", "sz399006"),   # 创业板指
    "000688": ("1", "sh000688"),   # 科创50
    "000300": ("1", "sh000300"),   # 沪深300
    "000905": ("1", "sh000905"),   # 中证500
    "899050": ("0", "bj899050"),   # 北证50
}


def _normalize_index_symbol(index_code: str) -> str:
    """把各种写法的指数代码归一化成 AKShare 需要的 ``sh000300`` 格式

    支持 ``000300.SH`` / ``sh000300`` / ``000300`` 等写法。
    """
    c = str(index_code).strip().lower()
    # 已经是 sh/sz/bj 前缀格式
    if c[:2] in ("sh", "sz", "bj") and c[2:].isdigit():
        return c
    # 去掉后缀：000300.SH -> 000300
    for suf in (".sh", ".sz", ".bj"):
        if c.endswith(suf):
            c = c[: -len(suf)]
            break
    if not c.isdigit():
        return str(index_code)
    # 000/880 开头是上交所，399/395 开头是深交所，其余按首位粗判
    market = "sh" if c.startswith(("000", "880")) else (
        "sz" if c.startswith(("399", "395", "391")) else (
            "bj" if c.startswith(("899", "920")) else "sh"))
    return f"{market}{c}"


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
        self._pool_cache: Dict[str, tuple] = {}  # pool_name -> (codes_list, timestamp)
        self._pool_cache_ttl = 1800  # 30分钟
        self._industry_map: Optional[Dict[str, str]] = None  # code -> industry
        # 负缓存：确认拉不到数据的代码（部分科创板 CDR、已退市），
        # 避免每次扫描重复走网络 + 重试，白等十几秒
        self._fail_codes: set = set()
        logger.info("DataFetcher initialized")
    
    def _merge_into_cache(self, cache_file: Path, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        将新数据合并进缓存文件（按日期取并集），返回合并后的完整数据

        为什么需要这个方法
        ------------------
        缓存文件名是 daily_{code}_{adjust}.parquet，不含日期区间，
        但写入的如果只是"本次调用区间内的子集"，就会把全量历史
        永久覆盖成短区间数据。典型触发场景：追踪回测只请求最近
        30 天，于是整个历史缓存被压成 30 天，后续训练样本量
        静默变化且无任何告警。

        这里做按日期索引的 upsert：新数据覆盖同日期旧数据，
        其余历史行原样保留。
        """
        try:
            if not cache_file.exists():
                new_df = new_df.sort_values("date")
                new_df.to_parquet(cache_file, index=False)
                return new_df

            old = pd.read_parquet(cache_file)
            if len(old) == 0:
                new_df = new_df.sort_values("date")
                new_df.to_parquet(cache_file, index=False)
                return new_df

            old["date"] = pd.to_datetime(old["date"])
            new_df = new_df.copy()
            new_df["date"] = pd.to_datetime(new_df["date"])

            keep_cols = [c for c in new_df.columns if c in old.columns]
            missing = [c for c in old.columns if c not in keep_cols]
            for c in missing:
                new_df[c] = np.nan

            new_df = new_df[old.columns]

            before = len(old)
            merged = (
                pd.concat([old, new_df], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )

            if len(merged) < before:
                logger.warning(
                    f"{cache_file.name} 合并后行数({len(merged)})少于原缓存({before})，"
                    f"请检查数据源是否异常"
                )
            merged.to_parquet(cache_file, index=False)
            return merged
        except Exception as e:
            logger.warning(f"缓存合并失败({cache_file.name}): {e}，直接写入新数据")
            new_df = new_df.sort_values("date")
            new_df.to_parquet(cache_file, index=False)
            return new_df

    # ========== 股票列表 ==========
    
    def _fetch_stock_list_remote(self) -> pd.DataFrame:
        """真正联网拉取全市场股票列表（耗时约 17 秒，勿在主线程频繁调用）"""
        df = _ak().stock_info_a_code_name()
        df = df.rename(columns={"code": "code", "name": "name"})

        # 过滤：只保留沪深主板、创业板、科创板、北交所
        df = df[df["code"].str.match(r"^\d{6}$")].copy()
        df["market"] = df["code"].apply(
            lambda x: "sh" if x.startswith(("6", "9")) else "sz"
        )
        df["industry"] = "未知"
        df["area"] = "未知"
        df["listed_date"] = ""
        df["total_cap"] = 0.0
        df["float_cap"] = 0.0
        return df

    def _refresh_stock_list_bg(self, cache_file: "Path") -> None:
        """后台线程刷新股票列表缓存（stale-while-revalidate）"""
        if getattr(self, "_stock_list_refreshing", False):
            return
        self._stock_list_refreshing = True
        try:
            logger.info("后台刷新股票列表…")
            df = self._fetch_stock_list_remote()
            if len(df) > 1000:  # 明显异常的结果不写入，避免污染缓存
                df.to_parquet(cache_file, index=False)
                self._stock_list_cache = df
                logger.info(f"股票列表后台刷新完成: {len(df)} 只")
        except Exception as e:
            logger.warning(f"后台刷新股票列表失败（继续使用旧缓存）: {e}")
        finally:
            self._stock_list_refreshing = False

    def get_stock_list(self, force_refresh: bool = False) -> pd.DataFrame:
        """获取A股全市场股票列表。

        采用 stale-while-revalidate：
        只要本地有缓存就立刻返回（毫秒级），过期则在后台线程异步刷新。
        因为全市场列表一天内几乎不变，没必要让用户干等 17 秒。
        """
        if self._stock_list_cache is not None and not force_refresh:
            return self._stock_list_cache

        cache_file = self.cache_dir / "stock_list.parquet"

        if cache_file.exists() and not force_refresh:
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 1000:
                    self._stock_list_cache = df
                    age_h = (time.time() - cache_file.stat().st_mtime) / 3600
                    if age_h > app_config.data.cache_days * 24:
                        # 过期了：先返回旧数据，后台悄悄更新
                        threading.Thread(
                            target=self._refresh_stock_list_bg,
                            args=(cache_file,), daemon=True,
                            name="stock-list-refresh",
                        ).start()
                    return df
            except Exception as e:
                logger.warning(f"读取股票列表缓存失败: {e}")

        logger.info("正在获取A股全市场股票列表（首次约需 10-20 秒）...")
        try:
            df = self._fetch_stock_list_remote()
            df.to_parquet(cache_file, index=False)
            self._stock_list_cache = df
            logger.info(f"获取到 {len(df)} 只A股股票")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            if self._stock_list_cache is not None:
                return self._stock_list_cache
            # 网络挂了但磁盘上还有旧缓存 —— 用过期的也比空表强
            if cache_file.exists():
                try:
                    stale = pd.read_parquet(cache_file)
                    if len(stale) > 0:
                        logger.warning(f"网络失败，回退到本地旧股票列表（{len(stale)} 只）")
                        return stale
                except Exception:
                    pass
            return pd.DataFrame(columns=["code", "name", "market", "industry",
                                          "area", "listed_date", "total_cap", "float_cap"])
            raise
    
    def get_stock_info(self, code: str) -> Dict[str, Any]:
        """获取单只股票的实时基本信息（名称/行业/总市值/流通市值/上市日期/股本）。

        数据源：东方财富 ``stock_individual_info_em``（个股信息最全的接口，命中
        push2.eastmoney.com，部分网络/代理环境会被拦截）。
        返回标准化 dict，市值/股本单位统一为「亿」。
        若调用失败，返回含 ``_error`` 字段的 dict（上层据此显示明确告警，而非静默留空）。
        """
        result: Dict[str, Any] = {"code": code, "name": code}
        try:
            time.sleep(app_config.data.rate_limit)
            raw = _ak().stock_individual_info_em(symbol=code)
            if raw is None or len(raw) == 0:
                result["_error"] = "东方财富返回空数据"
                return result
            kv: Dict[str, Any] = {}
            for _, r in raw.iterrows():
                item = str(r.get("item", "")).strip()
                val = r.get("value")
                if item:
                    kv[item] = val
            result.update({
                "name": str(kv.get("股票简称", code)),
                "industry": str(kv.get("行业", "") or ""),
                "total_cap": _parse_cap(kv.get("总市值")),
                "float_cap": _parse_cap(kv.get("流通市值")),
                "listed_date": str(kv.get("上市时间", "") or ""),
                "total_shares": _parse_cap(kv.get("总股本")),
                "float_shares": _parse_cap(kv.get("流通股")),
            })
            return result
        except Exception as e:
            logger.debug(f"获取 {code} 基本信息失败: {e}")
            result["_error"] = str(e)
            return result

    # ========== 日线行情 ==========
    
    def get_daily_kline(
        self,
        code: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        adjust: str = "qfq"  # 前复权
    ) -> pd.DataFrame:
        """获取单只股票日K线数据"""
        # 统一日期格式：空串/None 一律视为"不限"，否则与 datetime 列比较会得到全 False 空表
        # （空表再取 .max() 就是 NaT，后续做日期减法会 TypeError —— 曾让 AI 实时监控整页崩掉）
        if end_date is None or not str(end_date).strip():
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None or not str(start_date).strip():
            start_date = "2015-01-01"

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
                # 检查数据是否过期：
                # 交易日判断：如果今天是工作日，且缓存最新日期 < 今天，需要刷新
                today_dt = datetime.now()
                today_date = today_dt.date()
                latest_ts = cached["date"].max() if len(cached) > 0 else pd.NaT
                latest_date = latest_ts.date() if hasattr(latest_ts, 'date') else latest_ts
                # 区间内无数据 或 最新日期为 NaT：无法判新鲜度，按过期处理去重新拉取
                if len(cached) == 0 or pd.isna(latest_ts) or pd.isna(latest_date):
                    is_stale = True
                else:
                    # 超过 1 个自然日就算过期（确保每天自动刷新）
                    is_stale = (today_date - latest_date).days > 1
                # 同时判断：如果今天是交易日(周一至周五) + 最新数据不是今天 + 当前已过收盘时间(15:00)，立即刷新
                is_trading_day = today_dt.weekday() < 5
                is_after_close = today_dt.hour >= 15
                latest_ok = not (pd.isna(latest_ts) or pd.isna(latest_date))
                if is_trading_day and is_after_close and latest_ok and latest_date < today_date:
                    is_stale = True
                    logger.info(f"{code} 交易日收盘后缓存过期 (最新: {latest_date} 今天: {today_date})，重新拉取…")
                if len(cached) > 50 and not is_stale:
                    self._daily_cache[cache_key] = cached
                    return cached.copy()
                elif is_stale:
                    logger.info(f"{code} 缓存过期 (最新: {latest_date})，重新拉取…")
        
        # 代码清洗：去掉交易所后缀
        code_clean = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
        # 负缓存：本进程内已确认拉不到数据的代码，不再重复走网络+重试
        #（科创板 CDR 如 689xxx 新浪返回非 JSON，每次扫描白等 20s）
        if code_clean in getattr(self, "_fail_codes", set()):
            logger.debug(f"{code_clean} 命中失败缓存，跳过")
            return pd.DataFrame()

        max_retries = 3
        last_err = None
        for attempt in range(max_retries):
            try:
                wait = app_config.data.rate_limit * (attempt + 1)
                time.sleep(wait)  # 限速
                logger.debug(f"获取 {code} 日线数据... (尝试 {attempt+1}/{max_retries})")

                # 主源新浪，失败回退同花顺。
                # 注意：两个源都要各自 try —— 新浪对部分代码（如 689xxx CDR）
                # 直接抛 JSONDecodeError，若不隔离会跳到外层重试循环，
                # 同花顺兜底永远没机会执行。
                df = pd.DataFrame()
                try:
                    df = self._fetch_from_sina(code_clean, start_date, adjust)
                except Exception as e:
                    logger.warning(f"{code} 新浪源异常: {e}")
                    df = pd.DataFrame()

                if df is None or len(df) == 0:
                    try:
                        df = self._fetch_from_ths(code_clean, start_date, adjust)
                    except Exception as e:
                        logger.warning(f"{code} 同花顺源异常: {e}")
                        df = pd.DataFrame()

                if df is None or len(df) == 0:
                    last_err = "所有数据源均无数据"
                    if attempt < max_retries - 1:
                        backoff = (attempt + 1) * 3
                        logger.warning(f"{code} {last_err}，{backoff}s后重试")
                        time.sleep(backoff)
                        continue
                    logger.error(f"{code} {last_err}，放弃（加入失败缓存）")
                    self._fail_codes.add(code_clean)
                    return pd.DataFrame()
                
                # 按日期过滤
                df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

                if len(df) == 0:
                    logger.warning(f"{code} 指定日期范围内无数据")
                    return pd.DataFrame()

                # ===== 缓存写入：合并而非覆盖 =====
                # 旧实现直接 df.to_parquet(cache_file) 把"按本次调用区间
                # 裁剪后的子集"覆盖写回全量缓存文件。若某次调用只请求了
                # 最近 30 天，缓存文件就永久退化成 30 天数据，
                # 后续训练样本量会静默变化（此即追踪回测污染数据的根因）。
                # 现在改为与已有缓存按日期取并集后再写。
                df = self._merge_into_cache(cache_file, df)
                self._daily_cache[cache_key] = df

                return df
            except Exception as e:
                last_err = str(e)
                if attempt < max_retries - 1:
                    backoff = (attempt + 1) * 3
                    logger.warning(f"{code} 第{attempt+1}次失败，{backoff}s后重试: {e}")
                    time.sleep(backoff)
                else:
                    logger.error(f"获取 {code} 日线数据失败 (已重试{max_retries}次): {e}")
                    self._fail_codes.add(code_clean)

        logger.debug(f"{code} 最终无数据 (最后错误: {last_err})")
        return pd.DataFrame()
    
    def _fetch_from_sina(self, code: str, start_date: str, adjust: str = "qfq") -> pd.DataFrame:
        """
        从新浪数据源（AKShare stock_zh_a_daily）获取个股历史日K线
        优点：返回多年完整历史（2018至今），价格真实复权，稳定不拒连
        """
        market = "sh" if code.startswith(("6", "9")) else "sz"
        symbol = f"{market}{code}"
        start = start_date.replace("-", "")
        end = datetime.now().strftime("%Y%m%d")

        df = _ak().stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust=adjust)
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 标准化列名
        df = df.rename(columns={
            "date": "date", "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume", "amount": "amount",
        })
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # 补齐指标列（与同花顺源对齐）
        if "pct_change" not in df.columns:
            df["pct_change"] = df["close"].pct_change() * 100
        if "turnover" not in df.columns:
            df["turnover"] = df.get("turnover", 0.0)
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
        df["change"] = df["close"].diff()

        return df

    def _fetch_from_ths(self, code: str, start_date: str, adjust: str = "qfq") -> pd.DataFrame:
        """
        从同花顺 HTTP API 获取个股历史日K线
        接口: https://d.10jqka.com.cn/v6/line/hs_XXXXXX/01/last.js
        使用不复权(01)模式：前复权(02)返回编码价(每只倍数不同), 不复权(01)返回真实价格
        """
        # 强制使用不复权(01)获取真实价格，前复权(02)返回未除权编码价
        adjust_code = "01"
        url = f"https://d.10jqka.com.cn/v6/line/hs_{code}/{adjust_code}/last.js"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://stockpage.10jqka.com.cn/",
        }
        
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        
        # 去掉 JSONP 包裹: quotebridge_v6_line_xxx(...)
        m = re.search(r"\((.*)\)", r.text, re.DOTALL)
        if not m:
            raise ValueError(f"同花顺响应解析失败: {r.text[:200]}")
        
        data = json.loads(m.group(1))
        raw = data.get("data", "")
        if not raw:
            logger.warning(f"{code} 同花顺无数据")
            return pd.DataFrame()
        
        # 解析每一行数据
        lines = raw.strip().split(";")
        records = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            records.append({
                "date": parts[0],
                "open": float(parts[1]),
                "high": float(parts[2]),
                "low": float(parts[3]),
                "close": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
                "pct_change": float(parts[7]) if len(parts) > 7 and parts[7] else None,
                "turnover": float(parts[9]) if len(parts) > 9 and parts[9] else None,
            })
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.sort_values("date").reset_index(drop=True)
        
        # 不复权模式下价格为真实值，无需修正。补齐指标
        if df["pct_change"].isna().all():
            df["pct_change"] = df["close"].pct_change() * 100
        if df["turnover"].isna().all():
            df["turnover"] = 0.0
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
        df["change"] = df["close"].diff()
        
        # 过滤起始日期
        df = df[df["date"] >= start_date]
        
        return df
    
    def get_batch_daily_kline(
        self,
        codes: List[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        max_workers: int = 8,
        progress_callback=None,
    ) -> Dict[str, pd.DataFrame]:
        """批量获取日线数据（进程池并发，规避 V8 多线程崩溃）

        ``progress_callback(done, total, ok)`` 可用于向前端回传进度。

        ⚠️ 为什么用进程池而不是线程池：
        akshare 的部分接口（如新浪日线 stock_zh_a_daily）内部走 py_mini_racer / V8
        做 JS 解密，而 V8 的 PartitionAlloc 内存池初始化不是线程安全的。
        多线程并发调用会在某些环境触发原生 abort
        （``Check failed: !IsConfigurablePoolInitialized``），直接杀死整个进程、
        连带拖垮看板，且不可被 try/except 捕获。
        改用 spawn 进程池后，每个 worker 进程各自持有一个 V8 isolate、
        单线程内串行调用，不再有并发初始化竞争 —— 彻底规避该崩溃，
        同时保留真正的并发提速。
        """
        if not codes:
            return {}
        ed = end_date or datetime.now().strftime("%Y-%m-%d")
        if int(max_workers) <= 1:
            # 串行兜底（最稳，沙箱默认）
            return self._batch_serial(codes, start_date, ed, progress_callback, min_rows=1)
        try:
            return _mp_batch_fetch(
                codes, start_date, ed, max_workers, progress_callback, min_rows=1
            )
        except Exception as e:  # noqa: 进程池起不来时回退串行，保证可用
            logger.error(f"进程池批量拉取失败，回退串行: {e}")
            return self._batch_serial(codes, start_date, ed, progress_callback, min_rows=1)

    def _batch_serial(self, codes, start_date, end_date, progress_callback, min_rows=1):
        """串行兜底：单线程依次拉取，绝对不会触发 V8 并发崩溃。"""
        results: Dict[str, pd.DataFrame] = {}
        total = len(codes)
        completed = 0
        for code in codes:
            try:
                df = self.get_daily_kline(code, start_date, end_date)
                if df is not None and len(df) >= min_rows:
                    results[code] = df
            except Exception as e:
                logger.error(f"获取 {code} 失败: {e}")
            completed += 1
            if completed % 50 == 0:
                logger.info(f"进度: {completed}/{total}")
            if progress_callback and (completed % 5 == 0 or completed == total):
                try:
                    progress_callback(completed, total, len(results))
                except Exception:
                    pass  # 回调异常不影响主流程
        logger.info(f"串行批量获取完成: {len(results)}/{total} 只有效")
        return results
    
    # ========== 基本面数据 ==========
    
    def get_financial_indicators(self, code: str) -> pd.DataFrame:
        """获取财务指标（PE、PB、ROE等）"""
        try:
            time.sleep(app_config.data.rate_limit)
            df = _ak().stock_financial_analysis_indicator(symbol=code)
            if df is not None and len(df) > 0:
                df["code"] = code
                return df
        except Exception as e:
            logger.debug(f"获取 {code} 财务指标失败: {e}")
        return pd.DataFrame()
    
    def get_realtime_quote(self, codes: List[str]) -> pd.DataFrame:
        """获取实时行情（盘中）"""
        try:
            df = _ak().stock_zh_a_spot_em()
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
        """获取指数K线（上证/深证/创业板等）

        支持多种代码写法，会自动归一化到 AKShare 需要的前缀格式::

            get_index_kline("000300.SH")
            get_index_kline("sh000300")
            get_index_kline("000300")
        """
        symbol = _normalize_index_symbol(index_code)
        try:
            df = _ak().stock_zh_index_daily(symbol=symbol)
            if df is None or len(df) == 0:
                logger.warning(f"指数 {symbol} 返回空数据")
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= start_date].copy()
            # 补齐与个股日线一致的字段，下游 detect_regime 等无需特判
            if "pct_change" not in df.columns:
                df["pct_change"] = df["close"].pct_change() * 100
            if "change" not in df.columns:
                df["change"] = df["close"].diff()
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.error(f"获取指数K线失败 ({symbol}): {e}")
            return pd.DataFrame()

    def get_benchmark_kline(self, index_code: str = "000300.SH",
                            start_date: str = "2024-01-01") -> pd.DataFrame:
        """获取基准指数（默认沪深300）日线 —— 市场状态判断用

        与 :meth:`get_daily_kline` 的区别：指数不能走个股日线接口
        （新浪/同花顺的个股接口不提供指数，会重试 3 次后返回空）。
        """
        df = self.get_index_kline(index_code, start_date=start_date)
        if len(df) == 0:  # 主接口挂了再试新浪指数源
            df = self._fetch_index_from_tencent(index_code, start_date)
        return df

    def _fetch_index_from_tencent(self, index_code: str,
                                  start_date: str = "2024-01-01") -> pd.DataFrame:
        """腾讯指数日线兜底源（AKShare 接口不可用时使用）"""
        symbol = _normalize_index_symbol(index_code)
        try:
            r = requests.get(f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                             f"?param={symbol},day,2024-01-01,,320",
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            node = (r.json().get("data") or {}).get(symbol) or {}
            bars = node.get("qfqday") or node.get("day") or []
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars, columns=["date", "open", "close", "high",
                                             "low", "volume"][:len(bars[0])])
            df["date"] = pd.to_datetime(df["date"])
            for col in ("open", "close", "high", "low"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
            df = df.dropna(subset=["close"])
            df = df[df["date"] >= start_date]
            df["pct_change"] = df["close"].pct_change() * 100
            df["change"] = df["close"].diff()
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"腾讯指数源失败 ({symbol}): {e}")
            return pd.DataFrame()
    
    def get_market_overview(self) -> Dict:
        """获取大盘实时总览：主要指数行情 + 全市场涨跌家数

        主源：东方财富 ulist.np 批量接口（一次请求拿全部，含涨跌家数）
        备源：腾讯 qt.gtimg.cn（东财挂掉时兜底，无涨跌家数）
        兜底：两者都失败时返回进程内上次成功的结果（最多容忍 10 分钟），
              避免网络抖动导致界面大盘条整块消失。

        返回结构::

            {
              "indices": DataFrame[代码,名称,最新价,涨跌额,涨跌幅,成交额],
              "breadth": {"上涨": n, "下跌": n, "平盘": n},
              "amount_yi": 沪深两市合计成交额（亿）,
              "updated": 数据时间戳字符串,
              "source": "eastmoney" / "tencent" / "stale" / "none"
            }
        """
        try:
            return self._fetch_market_overview()
        except Exception as e:
            logger.error(f"大盘总览获取异常: {e}")
            return self._stale_market_overview()

    def _stale_market_overview(self) -> Dict:
        """返回上次成功的大盘数据（超过 10 分钟则返回空）"""
        cached = getattr(self, "_mkt_cache", None)
        if cached and (time.time() - cached["_ts"]) < 600:
            logger.warning("大盘数据源异常，沿用上次成功结果")
            stale = {k: v for k, v in cached.items() if k != "_ts"}
            stale["source"] = "stale"
            return stale
        return {"indices": pd.DataFrame(), "breadth": {}, "amount_yi": 0.0,
                "updated": "", "source": "none"}

    def _fetch_market_overview(self) -> Dict:
        # ---------- 主源：东方财富（多域名 + 重试，防偶发断连） ----------
        params = {
            "fltt": "2", "invt": "2",
            "fields": "f2,f3,f4,f5,f6,f12,f13,f14,f104,f105,f106,f124",
            "secids": ",".join(f"{mkt}.{code}" for code, (mkt, _) in _MAIN_INDICES.items()),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://quote.eastmoney.com/",
        }
        diff = None
        for host in ("push2.eastmoney.com", "82.push2.eastmoney.com", "push2delay.eastmoney.com"):
            for attempt in range(2):
                try:
                    r = requests.get(
                        f"https://{host}/api/qt/ulist.np/get",
                        params=params, headers=headers, timeout=8,
                    )
                    diff = (r.json().get("data") or {}).get("diff") or []
                    if diff:
                        break
                except Exception as e:
                    logger.debug(f"东财 {host} 第{attempt + 1}次失败: {e}")
                    diff = None
            if diff:
                break

        if diff:
            if isinstance(diff, dict):  # 接口偶尔返回 dict
                diff = list(diff.values())

            rows, up, down, flat, amount = [], 0, 0, 0, 0.0
            latest_ts = 0
            for it in diff:
                code = str(it.get("f12", "")).zfill(6)
                price = it.get("f2")
                if price in (None, "-") or it.get("f14") in (None, "-"):
                    continue
                rows.append({
                    "代码": code,
                    "名称": it["f14"],
                    "最新价": float(price),
                    "涨跌幅": float(it.get("f3") or 0),
                    "涨跌额": float(it.get("f4") or 0),
                    "成交额": float(it.get("f6") or 0),
                })
                # 只用上证+深证统计家数（避免重复计算）
                if code in ("000001", "399001"):
                    up += int(it.get("f104") or 0)
                    down += int(it.get("f105") or 0)
                    flat += int(it.get("f106") or 0)
                latest_ts = max(latest_ts, int(it.get("f124") or 0))

            if rows:
                df = pd.DataFrame(rows)
                df = df.merge(
                    pd.DataFrame(
                        [{"代码": k, "_ord": i} for i, k in enumerate(_MAIN_INDICES)]
                    ),
                    on="代码", how="left",
                ).sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
                # 沪深两市成交额 = 上证 + 深证（指数成交额单位为元）
                amount = float(df[df["代码"].isin(["000001", "399001"])]["成交额"].sum()) / 1e8
                # 东财不返回涨跌停家数，单独补一次乐咕（有则用，无则留空）
                prev_limits = dict((getattr(self, "_mkt_cache", None) or {}).get("limits") or {})
                try:
                    _, prev_limits = self._get_breadth_legu()
                except Exception:
                    pass
                out = {
                    "indices": df,
                    "breadth": {"上涨": up, "下跌": down, "平盘": flat},
                    "limits": prev_limits,
                    "amount_yi": amount,
                    "updated": (
                        datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d %H:%M:%S")
                        if latest_ts else ""
                    ),
                    "source": "eastmoney",
                }
                self._mkt_cache = dict(out, _ts=time.time())
                return out

        logger.warning("东方财富大盘接口不可用，降级到腾讯源")

        # ---------- 备源：腾讯 ----------
        try:
            symbols = [sym for _, (_, sym) in _MAIN_INDICES.items()]
            r = requests.get(f"http://qt.gtimg.cn/q={','.join(symbols)}", timeout=6)
            r.encoding = "gbk"
            rows = []
            for seg in r.text.split(";"):
                if '="' not in seg:
                    continue
                p = seg.split('="', 1)[1].rsplit('"', 1)[0].split("~")
                if len(p) < 33:
                    continue
                rows.append({
                    "代码": p[2],
                    "名称": p[1],
                    "最新价": float(p[3] or 0),
                    "涨跌幅": float(p[32] or 0),
                    "涨跌额": float(p[31] or 0),
                    "成交额": float(p[37] or 0) / 1e4 if len(p) > 37 else 0.0,  # 万 -> 亿
                })
            if rows:
                df = pd.DataFrame(rows)
                # 腾讯源的成交额已换算为亿
                amount = float(df[df["代码"].isin(["000001", "399001"])]["成交额"].sum())
                # 腾讯源没有涨跌家数 → 用乐咕赚钱效应接口补齐（含涨停/跌停）
                prev = getattr(self, "_mkt_cache", None) or {}
                breadth = dict(prev.get("breadth") or {})
                limits = dict(prev.get("limits") or {})
                if not breadth:
                    breadth, limits = self._get_breadth_legu()
                out = {
                    "indices": df,
                    "breadth": breadth,
                    "limits": limits,
                    "amount_yi": amount,
                    "updated": "",
                    "source": "tencent",
                }
                self._mkt_cache = dict(out, _ts=time.time())
                return out
        except Exception as e:
            logger.error(f"腾讯大盘接口失败: {e}")

        return self._stale_market_overview()

    def _get_breadth_legu(self) -> tuple:
        """乐咕乐股赚钱效应：涨跌家数 + 涨停/跌停数（东财不可用时的备源）

        返回 ``(breadth, limits)``，两个 dict；失败返回 ``({}, {})``。
        """
        try:
            df = _ak().stock_market_activity_legu()
            if df is None or len(df) == 0:
                return {}, {}
            vals = dict(zip(df["item"], df["value"]))

            def _n(k):
                try:
                    return int(float(vals.get(k, 0)))
                except (TypeError, ValueError):
                    return 0

            breadth = {"上涨": _n("上涨"), "下跌": _n("下跌"), "平盘": _n("平盘")}
            try:
                active = float(str(vals.get("活跃度", 0)).replace("%", "").strip())
            except (TypeError, ValueError):
                active = 0.0
            limits = {"涨停": _n("真实涨停"), "跌停": _n("真实跌停"), "活跃度": active}
            return breadth, limits
        except Exception as e:
            logger.warning(f"乐咕涨跌家数获取失败: {e}")
            return {}, {}

    def get_sector_hotspots(self) -> pd.DataFrame:
        """获取板块热点（行业涨跌排名）"""
        try:
            df = _ak().stock_board_industry_summary_ths()
            df["板块名称"] = df["板块"]  # 统一字段名
            return df.sort_values("涨跌幅", ascending=False)
        except Exception:
            pass
        # 东方财富备选（经常挂）
        try:
            df = _ak().stock_board_concept_name_em()
            return df.sort_values("涨跌幅", ascending=False)
        except Exception as e:
            logger.error(f"获取板块热点失败: {e}")
            return pd.DataFrame()
    
    # ========== 工具方法 ==========
    
    _VALID_POOLS = frozenset({"all", "hs300", "zz500", "zz1000", "hs300_zz500"})

    def get_stock_pool(self, pool: str = "all") -> List[str]:
        """获取指定股票池的代码列表（带内存缓存）。

        合法池名：all / hs300 / zz500 / zz1000 / hs300_zz500。
        ⚠️ 未知池名**不再静默退回全市场** —— 曾因池名写错触发一次约 5000 只的全市场
        扫描（跑了 9 分钟没完）。现在直接抛 ValueError，逼调用方显式处理，
        避免这种"看着正常、其实扫错宇宙"的静默失效。
        """
        pool = (pool or "all").strip().lower()
        if pool not in self._VALID_POOLS:
            raise ValueError(
                f"未知股票池 {pool!r}；合法值: {sorted(self._VALID_POOLS)}"
            )
        now = time.time()
        if pool in self._pool_cache:
            cached, ts = self._pool_cache[pool]
            if now - ts < self._pool_cache_ttl:
                return cached[:]
        
        stock_list = self.get_stock_list()
        
        # 过滤ST、退市等
        stock_list = stock_list[~stock_list["name"].str.contains("ST|退市|\\*ST", na=False)]
        
        if pool == "hs300":
            try:
                hs300 = _ak().index_stock_cons("000300")
                codes = hs300["品种代码"].tolist()
                stock_list = stock_list[stock_list["code"].isin(codes)]
            except Exception:
                logger.warning("获取沪深300成分股失败，使用全量")
        
        elif pool == "zz500":
            try:
                zz500 = _ak().index_stock_cons("000905")
                codes = zz500["品种代码"].tolist()
                stock_list = stock_list[stock_list["code"].isin(codes)]
            except Exception:
                logger.warning("获取中证500成分股失败，使用全量")

        elif pool == "zz1000":
            try:
                zz1000 = _ak().index_stock_cons("000852")
                codes = zz1000["品种代码"].tolist()
                stock_list = stock_list[stock_list["code"].isin(codes)]
            except Exception:
                logger.warning("获取中证1000成分股失败，使用全量")

        elif pool == "hs300_zz500":
            try:
                hs300 = _ak().index_stock_cons("000300")
                zz500 = _ak().index_stock_cons("000905")
                codes_all = list(set(hs300["品种代码"].tolist() + zz500["品种代码"].tolist()))
                stock_list = stock_list[stock_list["code"].isin(codes_all)]
            except Exception:
                logger.warning("获取沪深300+中证500成分股失败，使用全量")
        
        codes = stock_list["code"].tolist()
        self._pool_cache[pool] = (codes, now)
        return codes[:]
    
    def shuffle_by_prefix(self, codes: List[str]) -> List[str]:
        """按代码前缀轮询排列，确保不同板块均匀分布
        
        例如 000/300/600/688/920 各取第1只 → 各取第2只 → ...
        这样 scan_max=100 时每种板块都能扫到 ~20 只
        """
        prefixes = ["920", "688", "689", "300", "301", "302",
                     "000", "001", "002", "003", "600", "601", "603", "605"]
        buckets = {p: [] for p in prefixes}
        for c in codes:
            pfx = str(c)[:3]
            if pfx in buckets:
                buckets[pfx].append(c)
            else:
                buckets.setdefault(pfx, []).append(c)
        
        result = []
        max_len = max((len(v) for v in buckets.values()), default=0)
        for i in range(max_len):
            for pfx in prefixes:
                if i < len(buckets.get(pfx, [])):
                    result.append(buckets[pfx][i])
        return result
    
    # 少于这个数量就不认为「抓到了行业映射」——多半是接口被限流/代理拦截，
    # 只拿到零散几条。用它来决定要不要落盘，避免空结果把缓存"投毒"。
    _MIN_INDUSTRY_CODES = 20
    # 空缓存的负缓存窗口（小时）：这段时间里不再重试，否则每次预测都要白跑
    # 上百个必然失败的请求，把选股流程拖死。
    _INDUSTRY_NEG_TTL_H = 6.0

    def get_industry_map(self, force_refresh: bool = False) -> Dict[str, str]:
        """获取股票代码→行业映射（内存缓存）。

        两条容易踩的坑，这里都堵上了：
          1) **空结果不许落盘**。历史上接口被代理拦截时会拿到 ``{}``，又被原样
             写进 ``industry_map.json``；此后每次调用都直接返回这个空缓存，
             即使网络恢复也**永远不会自愈**（2026-10-19 实测：东财
             ``17.push2.eastmoney.com`` 直接 ProxyError）。
          2) **空缓存要当负缓存用**。既然抓不到，就得记住"刚失败过"，六小时内
             不再重试；否则一次选股要串行打上百个注定失败的请求。
        """
        import json as _json
        import time as _time

        if self._industry_map is not None and not force_refresh:
            return self._industry_map

        cache_file = self.cache_dir / "industry_map.json"
        # 尝试缓存
        if cache_file.exists() and not force_refresh:
            try:
                cached = _json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                cached = None
            if cached:
                self._industry_map = cached
                return self._industry_map
            # 空缓存 = 上次抓取失败。窗口内直接认命，别再打网络。
            try:
                age_h = (_time.time() - cache_file.stat().st_mtime) / 3600.0
            except OSError:
                age_h = self._INDUSTRY_NEG_TTL_H + 1
            if age_h < self._INDUSTRY_NEG_TTL_H:
                logger.debug(
                    f"行业映射为空缓存（{age_h:.1f}h 前失败），"
                    f"{self._INDUSTRY_NEG_TTL_H:.0f}h 内不再重试"
                )
                self._industry_map = {}
                return {}

        logger.info("从 AKShare 获取行业分类数据…")
        self._industry_map = {}
        try:
            # 按申万行业分类批量获取
            industry_df = _ak().stock_board_industry_name_em()
            if len(industry_df) > 0:
                # 遍历每个行业获取成分股
                industries = industry_df.head(100)["板块名称"].tolist()  # 取前100个行业
                for _i, ind in enumerate(industries):
                    try:
                        cons = _ak().stock_board_industry_cons_em(symbol=ind)
                        if len(cons) > 0:
                            for _, row in cons.iterrows():
                                code = str(row.get("代码", ""))
                                if code:
                                    self._industry_map[code] = ind
                    except Exception:
                        continue
                    # 接口"看起来通、其实全空"时要尽早止损：连试 10 个行业
                    # 一个成分股都没有，再跑下去只是浪费 90 次请求。
                    if _i >= 9 and not self._industry_map:
                        logger.warning("连续 10 个行业板块都拿不到成分股，判定接口不可用，提前放弃")
                        break
        except Exception as e:
            logger.warning(f"行业映射获取失败: {e}")

        # 缓存：只有拿到可信规模才覆盖，否则保留旧值（或仅刷新"失败时间戳"）
        _n = len(self._industry_map)
        try:
            if _n >= self._MIN_INDUSTRY_CODES:
                cache_file.write_text(
                    _json.dumps(self._industry_map, ensure_ascii=False), encoding="utf-8")
            elif not cache_file.exists():
                cache_file.write_text("{}", encoding="utf-8")   # 负缓存：记下"抓不到"
            else:
                cache_file.touch()      # 保留旧内容，只刷新负缓存窗口
        except OSError as e:
            logger.debug(f"行业映射缓存写入失败（已忽略）: {e}")
        logger.info(f"行业映射获取完成: {_n} 只")
        return self._industry_map
    
    def filter_by_board(self, codes: List[str], boards: List[str]) -> List[str]:
        """按板块过滤股票代码
        
        boards: 可选值 — "沪深主板", "创业板", "科创板", "北交所"
        按代码前缀识别：
          沪深主板: 000-003, 600-605
          创业板:   300-302
          科创板:   688-689
          北交所:   920
        """
        if not boards:
            return codes
        
        board_prefixes = {
            "沪深主板": ["000", "001", "002", "003", "600", "601", "603", "605"],
            "创业板":   ["300", "301", "302"],
            "科创板":   ["688", "689"],
            "北交所":   ["920"],
        }
        
        allowed_prefixes = set()
        for b in boards:
            allowed_prefixes.update(board_prefixes.get(b, []))
        
        if not allowed_prefixes:
            return codes
        
        return [c for c in codes if str(c)[:3] in allowed_prefixes]
    
    def get_realtime_quote(self, code: str) -> Optional[Dict]:
        """获取单只股票实时行情（腾讯源，稳定快速 ~50ms）
        
        腾讯接口返回字段（按 ~ 分割）:
          [0]=市场标记, [1]=名称, [2]=代码, [3]=最新价, [4]=昨收, [5]=今开,
          [6]=成交量(手), [7]=外盘, [8]=内盘, [9..28]=买卖五档, [30]=时间戳,
          [31]=涨跌额, [32]=涨跌幅, [33]=最高, [34]=最低, [36]=成交量, [37]=成交额(万),
          [38]=换手率, [39]=市盈率, [41]=最高, [42]=最低, [44]=流通市值, [45]=总市值
        """
        code = str(code).strip()
        if "." in code:
            code = code.split(".")[0]
        if not re.match(r'^\d{6}$', code):
            return None

        # 判断市场前缀
        if code.startswith(("6", "9", "5")):
            sym = f"sh{code}"
        elif code.startswith(("4", "8")):
            sym = f"bj{code}"
        else:
            sym = f"sz{code}"

        return self._tencent_quote(sym)

    def _tencent_quote(self, sym: str) -> Optional[Dict]:
        """按腾讯代码（``sz000001`` / ``sh000001``）取一条快照。

        ⚠️ 单独抽出来的原因：**指数的市场前缀不能按首位数字猜**。
        ``000001`` 对个股是平安银行（``sz000001``），对上证指数是
        ``sh000001`` —— 同一个数字，两个完全不同的标的。历史 bug 正是
        助手问「大盘怎么样」时被解析成平安银行。所以指数必须由调用方
        给出**已带市场前缀**的 symbol，绝不复用个股的前缀推断。
        """
        try:
            r = requests.get(f"http://qt.gtimg.cn/q={sym}", timeout=5)
            r.encoding = "gbk"
            txt = r.text.strip()
            if not txt or "=" not in txt:
                return None
            # 解析 v_sz000001="...";
            payload = txt.split('="', 1)[1].rsplit('"', 1)[0]
            parts = payload.split("~")
            if len(parts) < 40:
                return None

            def _f(v, default=0.0):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return default

            return {
                "代码": parts[2],
                "名称": parts[1],
                "最新价": _f(parts[3]),
                "昨收": _f(parts[4]),
                "今开": _f(parts[5]),
                "成交量": _f(parts[6]) * 100 if _f(parts[6]) > 0 else 0,  # 手->股
                "涨跌额": _f(parts[31]),
                "涨跌幅": _f(parts[32]),
                "最高": _f(parts[33]),
                "最低": _f(parts[34]),
                "成交额": _f(parts[37]),  # 万
                "换手率": _f(parts[38]),
                "市盈率": _f(parts[39]),
                "时间戳": parts[30],
            }
        except Exception as e:
            logger.warning(f"腾讯实时行情获取失败 {sym}: {e}")
            return None

    def get_index_realtime(self, index_code: str = "sh000001") -> Optional[Dict]:
        """指数实时快照（上证/深证/创业板/沪深300/中证500/科创50 等）。"""
        return self._tencent_quote(_normalize_index_symbol(index_code))

    def batch_get_daily(self, codes: List[str], start_date: str = "2018-01-01",
                         end_date: Optional[str] = None, max_workers: int = 8) -> Dict[str, pd.DataFrame]:
        """批量获取日线数据（进程池并发，规避 V8 多线程崩溃；要求 >=60 行）"""
        if not codes:
            return {}
        ed = end_date or datetime.now().strftime("%Y%m%d")
        if int(max_workers) <= 1:
            return self._batch_serial(codes, start_date, ed, None, min_rows=60)
        try:
            return _mp_batch_fetch(codes, start_date, ed, max_workers, None, min_rows=60)
        except Exception as e:  # noqa
            logger.error(f"batch_get_daily 进程池失败，回退串行: {e}")
            return self._batch_serial(codes, start_date, ed, None, min_rows=60)
    
    def clear_cache(self):
        """清理缓存"""
        self._stock_list_cache = None
        self._daily_cache = {}
        self._fail_codes.clear()  # 同时清负缓存，允许重试曾失败的代码
        logger.info("缓存已清理")


# ============================================================================
# 进程池批量拉取（模块级，规避 py_mini_racer / V8 多线程原生崩溃）
# ----------------------------------------------------------------------------
# 根因：akshare 部分接口（如新浪日线 stock_zh_a_daily）内部用 V8(py_mini_racer)
#       做 JS 解密，V8 的 PartitionAlloc 内存池初始化不是线程安全的。
#       多线程并发 -> 原生 abort (Check failed: !IsConfigurablePoolInitialized)
#       -> 杀死整个进程、连带拖垮看板，且不可 try/except 捕获。
# 解法：spawn 进程池，每个 worker 进程各自持有一个 V8 isolate、单线程串行调用，
#       不存在并发初始化竞争；同时保留真正的并发提速。
# 内存上限：akshare 单进程很吃内存，最多 8 个 worker 进程，避免 OOM。
# ============================================================================
_MP_MAX_PROC = 8
_MP_FETCHER = None


def _mp_get_fetcher() -> "DataFetcher":
    """每个 worker 进程内懒加载一个 DataFetcher 单例（复用缓存目录与负缓存）。"""
    global _MP_FETCHER
    if _MP_FETCHER is None:
        _MP_FETCHER = DataFetcher()
    return _MP_FETCHER


def _mp_fetch_one(task):
    """进程池 worker：拉取单只股票日线，异常不影响整体。"""
    code, start_date, end_date, adjust = task
    try:
        df = _mp_get_fetcher().get_daily_kline(code, start_date, end_date, adjust)
        return code, (df if df is not None else pd.DataFrame())
    except Exception as e:  # noqa
        logger.error(f"[mp] {code} 拉取失败: {e}")
        return code, pd.DataFrame()


def _mp_batch_fetch(codes, start_date, end_date, max_workers, progress_callback=None, min_rows=1):
    """用 spawn 进程池并发拉取，返回 {code: df}。"""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    n_proc = max(1, min(int(max_workers), _MP_MAX_PROC))
    tasks = [(c, start_date, end_date, "qfq") for c in codes]

    results: Dict[str, pd.DataFrame] = {}
    total = len(tasks)
    done = 0
    with ctx.Pool(n_proc) as pool:
        for code, df in pool.imap_unordered(_mp_fetch_one, tasks):
            done += 1
            if df is not None and len(df) >= min_rows:
                results[code] = df
            if progress_callback and (done % 5 == 0 or done == total):
                try:
                    progress_callback(done, total, len(results))
                except Exception:
                    pass  # 回调异常不影响主流程

    logger.info(f"进程池批量获取完成: {len(results)}/{total} 只有效 (n_proc={n_proc})")
    return results


# 全局单例
fetcher = DataFetcher()
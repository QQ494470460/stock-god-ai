# -*- coding: utf-8 -*-
"""A 股交易日历与交易时段

为什么必须有这个模块：
    原实现用 ``weekday() < 5`` 冒充交易日，于是国庆、春节、清明等法定假日
    也会被当成交易日 —— 自动交易会在休市日尝试下单，券商拒单/废单堆积，
    还会误触发"止损/止盈"等逻辑。

数据源与降级：
    优先用 akshare ``tool_trade_date_hist_sina``（上交所交易日历），
    结果缓存到 ``data/trade_calendar.json``（一年只需拉一次）。
    拉取失败时回退到"工作日"判断并把 ``is_degraded()`` 置为 True，
    调用方可据此给出提示，**绝不因为日历拿不到就阻断交易**。
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional, Set

from loguru import logger

try:  # 打包环境下 config 可能不在 sys.path 顶层，做兜底
    from config.settings import DATA_DIR
except Exception:  # noqa: BLE001
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

CALENDAR_FILE = DATA_DIR / "trade_calendar.json"

# 交易时段（连续竞价）
AM_OPEN, AM_CLOSE = time(9, 30), time(11, 30)
PM_OPEN, PM_CLOSE = time(13, 0), time(15, 0)
# 开盘集合竞价（可下单，但成交在 9:25 一次性撮合）
AUCTION_OPEN, AUCTION_CLOSE = time(9, 15), time(9, 25)

_LOCK = threading.RLock()
_CAL_CACHE: Set[str] = set()
_CAL_LOADED = False
_DEGRADED = False


# ==================== 日历加载 ====================

def _fetch_from_akshare() -> Set[str]:
    """从 akshare 拉取交易日历（按需 import，不拖慢启动）"""
    import akshare as _ak  # noqa: PLC0415  —— 延迟加载

    df = _ak.tool_trade_date_hist_sina()
    col = "trade_date" if "trade_date" in df.columns else df.columns[0]
    days: Set[str] = set()
    for v in df[col]:
        try:
            d = v.date() if isinstance(v, datetime) else v
            days.add(d.strftime("%Y-%m-%d"))
        except Exception:  # noqa: BLE001
            continue
    return days


def _save_cache(days: Set[str]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CALENDAR_FILE.write_text(
            json.dumps(
                {
                    "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "count": len(days),
                    "trade_days": sorted(days),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"交易日历缓存写入失败: {e}")


def _load_cache() -> Set[str]:
    try:
        if not CALENDAR_FILE.exists():
            return set()
        obj = json.loads(CALENDAR_FILE.read_text(encoding="utf-8"))
        return set(obj.get("trade_days") or [])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"交易日历缓存读取失败: {e}")
        return set()


def _load(force: bool = False) -> Set[str]:
    """加载交易日历（缓存优先，跨年自动刷新）"""
    global _CAL_CACHE, _CAL_LOADED, _DEGRADED

    with _LOCK:
        if _CAL_LOADED and not force:
            return _CAL_CACHE

        days = _load_cache()
        this_year = datetime.now().year

        # 缓存为空、强制刷新、或不含今年 → 重新拉取
        need_fetch = force or not days or not any(
            s.startswith(str(this_year)) for s in days
        )
        if need_fetch:
            try:
                fetched = _fetch_from_akshare()
                if fetched:
                    days = fetched
                    _save_cache(days)
                    _DEGRADED = False
                    logger.info(f"交易日历已更新：{len(days)} 个交易日")
                elif not days:
                    _DEGRADED = True
            except Exception as e:  # noqa: BLE001
                logger.warning(f"交易日历拉取失败（{e}），回退工作日判断")
                if not days:
                    _DEGRADED = True

        _CAL_CACHE = days
        _CAL_LOADED = True
        return _CAL_CACHE


def refresh_calendar() -> int:
    """强制刷新交易日历，返回交易日数量（0 表示拉取失败）"""
    return len(_load(force=True))


def is_degraded() -> bool:
    """日历是否处于降级状态（拿不到真实日历，用工作日兜底）"""
    _load()
    return _DEGRADED or not _CAL_CACHE


# ==================== 查询接口 ====================

def _as_date(d) -> date:
    if d is None:
        return datetime.now().date()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    return d


def is_trading_day(d=None) -> bool:
    """是否为交易日（含法定节假日判断）"""
    day = _as_date(d)
    days = _load()
    if days:
        return day.strftime("%Y-%m-%d") in days
    # 降级：只能用工作日兜底
    return day.weekday() < 5


def is_trading_time(now: Optional[datetime] = None) -> bool:
    """当前是否处于连续竞价时段（9:30-11:30 / 13:00-15:00）"""
    t = (now or datetime.now()).time()
    return (AM_OPEN <= t <= AM_CLOSE) or (PM_OPEN <= t <= PM_CLOSE)


def is_auction_time(now: Optional[datetime] = None) -> bool:
    """是否处于开盘集合竞价时段（9:15-9:25）"""
    t = (now or datetime.now()).time()
    return AUCTION_OPEN <= t <= AUCTION_CLOSE


def can_trade_now(now: Optional[datetime] = None) -> bool:
    """当前能否下单：交易日 + 可下单时段"""
    now = now or datetime.now()
    return is_trading_day(now.date()) and (is_trading_time(now) or is_auction_time(now))


def market_status(now: Optional[datetime] = None) -> str:
    """市场状态（用于 UI 展示与日志）"""
    now = now or datetime.now()
    if not is_trading_day(now.date()):
        return "休市（非交易日）"
    t = now.time()
    if t < AUCTION_OPEN:
        return "盘前"
    if AUCTION_OPEN <= t <= AUCTION_CLOSE:
        return "开盘集合竞价"
    if AUCTION_CLOSE < t < AM_OPEN:
        return "集合竞价结束待开盘"
    if AM_OPEN <= t <= AM_CLOSE:
        return "早盘交易中"
    if AM_CLOSE < t < PM_OPEN:
        return "午间休市"
    if PM_OPEN <= t <= PM_CLOSE:
        return "午盘交易中"
    return "已收盘"


def order_allowed(now: Optional[datetime] = None) -> tuple[bool, str]:
    """能否下单，附带原因（供风控层直接使用）

    返回 ``(是否允许, 原因说明)``
    """
    now = now or datetime.now()
    if not is_trading_day(now.date()):
        return False, f"非交易日（{market_status(now)}）"
    if not (is_trading_time(now) or is_auction_time(now)):
        return False, f"非交易时段（{market_status(now)}）"
    return True, market_status(now)


def next_trading_day(d=None, include_today: bool = False) -> date:
    """下一个交易日"""
    day = _as_date(d)
    if not include_today:
        day = day + timedelta(days=1)
    for _ in range(15):  # 最长假期也不到 15 天
        if is_trading_day(day):
            return day
        day = day + timedelta(days=1)
    return day


def recent_trading_days(n: int = 10, end=None) -> list[date]:
    """最近 n 个交易日（含 end 当天，若它是交易日）"""
    out: list[date] = []
    day = _as_date(end)
    for _ in range(60):
        if is_trading_day(day):
            out.append(day)
            if len(out) >= n:
                break
        day = day - timedelta(days=1)
    return out


def trading_days_between(start, end=None) -> int:
    """``start`` 之**后**到 ``end``（含）之间的交易日数。

    用途：判断某批预测是否已经走满 T+N 持有期。
    **必须按交易日而非日历日算** —— ``today - 5 天`` 经常落在周末，
    据此反推目标日期会永远找不到记录（这是历史遗留的验证失效根因）。

    start 与 end 同日返回 0；end 早于 start 也返回 0。
    """
    s = _as_date(start)
    e = _as_date(end)
    if e <= s:
        return 0
    n = 0
    day = s + timedelta(days=1)
    guard = 0
    while day <= e and guard < 500:      # 500 天上限，防脏数据死循环
        if is_trading_day(day):
            n += 1
        day += timedelta(days=1)
        guard += 1
    return n

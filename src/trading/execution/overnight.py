# -*- coding: utf-8 -*-
"""隔夜预挂与盘前时段判定。

A 股 09:15–09:25 为开盘集合竞价，可报送委托、09:25 一次性撮合。
"隔夜预挂"= 把本应在 09:30 连续竞价发出的首笔，提前到 09:15 盘前报送，
以争取集合竞价的成交（尤其对流动性一般的标的，开盘冲击更大）。
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional


def is_preopen(now: datetime) -> bool:
    """是否处于开盘集合竞价时段（09:15 ≤ t ≤ 09:25）。"""
    t = now.time()
    return time(9, 15) <= t <= time(9, 25)


def preopen_release_dt(trade_date) -> datetime:
    """返回某交易日盘前报送时刻（09:15）。``trade_date`` 为 date 或 datetime。"""
    d = trade_date.date() if isinstance(trade_date, datetime) else trade_date
    return datetime(d.year, d.month, d.day, 9, 15, 0)


def market_open_dt(trade_date) -> datetime:
    d = trade_date.date() if isinstance(trade_date, datetime) else trade_date
    return datetime(d.year, d.month, d.day, 9, 30, 0)


def schedule_window(
    trade_date,
    window_min: int = 300,
    overnight: bool = True,
    close_buffer_min: int = 3,
) -> tuple:
    """返回 (首片时刻, 末片时刻)。

    - overnight=True：首片 = 09:15（盘前报送）；否则 = 09:30。
    - 末片 = 开盘 + window_min，但不晚于 15:00 - close_buffer_min。
    """
    start = preopen_release_dt(trade_date) if overnight else market_open_dt(trade_date)
    end = market_open_dt(trade_date) + timedelta(minutes=window_min)
    cap = datetime(trade_date.year if isinstance(trade_date, datetime) else trade_date.year,
                   trade_date.month if isinstance(trade_date, datetime) else trade_date.month,
                   trade_date.day if isinstance(trade_date, datetime) else trade_date.day,
                   15, 0, 0) - timedelta(minutes=close_buffer_min)
    if end > cap:
        end = cap
    return start, end


def evenly_spaced(start: datetime, end: datetime, n: int) -> list:
    """在 [start, end] 内均匀取 n 个时刻（含首尾）。"""
    if n <= 1:
        return [start]
    step = (end - start) / (n - 1)
    return [start + step * i for i in range(n)]

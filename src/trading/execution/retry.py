# -*- coding: utf-8 -*-
"""限价单重试状态机（LimitRetry）。

真实券商下，限价单可能无法全部成交（或部分成交）。本状态机描述：
- 到达重试时刻后，把委托价向"确保成交"方向微调（买价上抬 / 卖价下压）；
- 重试超过上限后，转为市价单（激进确保成交）；
- 全程不修改目标数量，只改价格策略。

纯状态机、无副作用、可单测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LimitRetry:
    """单笔切片委托的重试策略。

    Attributes:
        max_attempts: 最多重试次数（不含首发的那一次）。0 = 不重试、保持限价。
        step_bps: 每次重试价格向成交方向移动的基点（买 + / 卖 -）。
        timeout_sec: 两次重试的最小间隔（秒）；用于避免同 tick 内死循环。
        escalate_to_market: 超过 max_attempts 后是否转市价单。
        attempts: 已尝试次数（含首发）。
        last_attempt_ts: 上次尝试的时间戳（由调用方写入，单位秒）。
        market_order: 是否已升级为市价单。
    """

    max_attempts: int = 3
    step_bps: float = 8.0
    timeout_sec: float = 60.0
    escalate_to_market: bool = True
    attempts: int = 0
    last_attempt_ts: float = 0.0
    market_order: bool = False

    def reset(self) -> None:
        self.attempts = 0
        self.last_attempt_ts = 0.0
        self.market_order = False

    def next_price(self, price: float, side: str, now_ts: float) -> tuple:
        """给定原价与方向，返回 (新价, 是否市价单)。

        ``side`` 取 'buy' / 'sell'。``now_ts`` 为当前时间戳（秒）；若距上次尝试
        不足 ``timeout_sec`` 则原样返回（避免瞬间连发）。
        """
        if self.market_order:
            return price, True
        if self.attempts == 0:
            self.attempts = 1
            self.last_attempt_ts = now_ts
            return price, False
        if self.attempts >= self.max_attempts:
            if self.escalate_to_market:
                self.market_order = True
                return price, True
            return price, False
        if now_ts - self.last_attempt_ts < self.timeout_sec:
            return price, False
        self.attempts += 1
        self.last_attempt_ts = now_ts
        rate = self.step_bps / 10_000.0
        s = str(getattr(side, "value", side)).lower()
        if s.startswith("buy"):
            return round(price * (1 + rate), 3), False
        return round(price * (1 - rate), 3), False

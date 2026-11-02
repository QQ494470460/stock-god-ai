# -*- coding: utf-8 -*-
"""执行调度器（ExecutionScheduler）—— 把一笔调仓在时间维度摊薄成多片。

核心职责
--------
1. 接收 ``compute_rebalance_orders`` 产出的 ``RebalanceIntent`` 列表；
2. 按 TWAP/VWAP 切成若干片，均匀铺在 [盘前/开盘, 开盘+window] 时间窗；
3. 每 tick 由 ``release_due(now)`` 吐出"此刻该发"的片，调用方经 ``_place`` 下单；
4. 成交后 ``on_fill`` 回报，未全量成交的片按 ``LimitRetry`` 抬价重试（每片独立）；
5. 可选落盘（persist_state），崩溃重启后能续传，不丢单、不重单。

所有逻辑纯状态机，不碰网络；下单仍走 loop 的 ``_place``（必经 guard 裁决）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from src.trading.broker import OrderSide
from src.trading.rebalance import RebalanceIntent

from src.trading.execution.slice import (
    twap_slices, vwap_slices, choose_n_slices, default_vwap_profile,
)
from src.trading.execution.retry import LimitRetry
from src.trading.execution.overnight import schedule_window, evenly_spaced


@dataclass
class Slice:
    slice_id: str
    qty: int
    price: float
    scheduled_dt: datetime
    status: str = "pending"          # pending / sent / filled / cancelled
    sent_ts: float = 0.0
    attempt: int = 0
    retry: LimitRetry = field(default_factory=LimitRetry)

    def is_closed(self) -> bool:
        return self.status in ("filled", "cancelled")


@dataclass
class PendingOrder:
    code: str
    name: str
    side: OrderSide
    price: float
    reason: str
    target_weight: float
    total_qty: int
    slices: List[Slice] = field(default_factory=list)
    filled: int = 0

    def filled_qty(self) -> int:
        return sum(s.qty for s in self.slices if s.status == "filled")

    def is_done(self) -> bool:
        return all(s.is_closed() for s in self.slices)


@dataclass
class ReleaseOrder:
    """release_due 返回给 loop 的下单指令（已从 PendingOrder 扁平化）。"""
    slice_id: str
    code: str
    name: str
    side: OrderSide
    qty: int
    price: float
    reason: str
    target_weight: float


class ExecutionScheduler:
    def __init__(
        self,
        strategy: str = "twap",
        window_min: int = 300,
        max_slices: int = 10,
        overnight: bool = True,
        retry_cfg: Optional[dict] = None,
        persist_path: Optional[str] = None,
        clock=None,
    ):
        self.strategy = strategy
        self.window_min = window_min
        self.max_slices = max_slices
        self.overnight = overnight
        self.retry_cfg = retry_cfg or {}
        self.persist_path = persist_path
        self._clock = clock or datetime.now
        self.pending: List[PendingOrder] = []

    def _new_retry(self) -> LimitRetry:
        rc = self.retry_cfg
        return LimitRetry(
            max_attempts=int(rc.get("max_attempts", 3)),
            step_bps=float(rc.get("step_bps", 8.0)),
            timeout_sec=float(rc.get("timeout_sec", 60.0)),
            escalate_to_market=bool(rc.get("escalate_to_market", True)),
        )

    # ------------------------------------------------------------------
    # 入队
    # ------------------------------------------------------------------
    def enqueue(self, intents: List[RebalanceIntent], trade_date) -> int:
        """把调仓意图切片入队，返回入队的委托数。"""
        if not intents:
            return 0
        # 新一轮调仓覆盖旧挂单（旧片本应在上一周期执行完）
        self.pending = []
        d = trade_date.date() if isinstance(trade_date, datetime) else trade_date
        start, end = schedule_window(d, self.window_min, self.overnight)
        n_total = 0
        for it in intents:
            qty = int(it.quantity)
            if qty <= 0:
                continue
            n = choose_n_slices(qty, self.max_slices)
            if self.strategy == "vwap":
                slice_qtys = vwap_slices(qty, default_vwap_profile(n))
            else:
                slice_qtys = twap_slices(qty, n)
            times = evenly_spaced(start, end, len(slice_qtys))
            po = PendingOrder(
                code=it.code, name=it.name, side=it.side, price=float(it.price),
                reason=it.reason, target_weight=getattr(it, "target_weight", 0.0),
                total_qty=qty,
            )
            for i, (sq, dt) in enumerate(zip(slice_qtys, times)):
                if sq <= 0:
                    continue
                po.slices.append(Slice(
                    slice_id=f"{it.code}#{i}", qty=sq, price=float(it.price),
                    scheduled_dt=dt, retry=self._new_retry(),
                ))
            if po.slices:
                self.pending.append(po)
                n_total += 1
        self._save()
        return n_total

    # ------------------------------------------------------------------
    # 出队（本 tick 该发的片）
    # ------------------------------------------------------------------
    def release_due(self, now: Optional[datetime] = None) -> List[ReleaseOrder]:
        now = now or self._clock()
        now_ts = now.timestamp()
        out: List[ReleaseOrder] = []
        for po in self.pending:
            if po.is_done():
                continue
            for s in po.slices:
                if s.is_closed():
                    continue
                # 已发未全成：超时且仍有重试额度 → 抬价重开
                if s.status == "sent":
                    if (now_ts - (s.sent_ts or 0) >= s.retry.timeout_sec
                            and s.retry.attempts - 1 < s.retry.max_attempts):
                        new_price, _ = s.retry.next_price(s.price, po.side, now_ts)
                        s.price = new_price
                        s.status = "pending"
                    else:
                        continue
                # 到期且待发 → 发出
                if s.status == "pending" and s.scheduled_dt <= now:
                    if s.attempt > s.retry.max_attempts + 3:
                        s.status = "cancelled"     # 多次未成，放弃，防死循环
                        continue
                    if s.retry.attempts == 0:
                        s.retry.next_price(s.price, po.side, now_ts)  # 记首发
                    s.attempt += 1
                    s.status = "sent"
                    s.sent_ts = now_ts
                    out.append(ReleaseOrder(
                        slice_id=s.slice_id, code=po.code, name=po.name,
                        side=po.side, qty=s.qty, price=s.price,
                        reason=po.reason, target_weight=po.target_weight,
                    ))
        return out

    # ------------------------------------------------------------------
    # 成交回报
    # ------------------------------------------------------------------
    def on_fill(self, slice_id: str, filled_qty: int, now: Optional[datetime] = None) -> None:
        now = now or self._clock()
        now_ts = now.timestamp()
        for po in self.pending:
            for s in po.slices:
                if s.slice_id != slice_id:
                    continue
                if filled_qty >= s.qty:
                    s.status = "filled"
                else:
                    # 部分成交：先挂起，等 release_due 在超时后抬价重发
                    s.status = "sent"
                    s.sent_ts = now_ts
                self._save()
                return

    def is_done(self) -> bool:
        return all(po.is_done() for po in self.pending) if self.pending else True

    def pending_count(self) -> int:
        return sum(1 for po in self.pending if not po.is_done())

    # ------------------------------------------------------------------
    # 持久化（崩溃续传）
    # ------------------------------------------------------------------
    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False)
        except Exception:
            pass

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "window_min": self.window_min,
            "max_slices": self.max_slices,
            "overnight": self.overnight,
            "retry_cfg": self.retry_cfg,
            "pending": [
                {
                    "code": po.code, "name": po.name, "side": po.side.value,
                    "price": po.price, "reason": po.reason,
                    "target_weight": po.target_weight, "total_qty": po.total_qty,
                    "slices": [
                        {
                            "slice_id": s.slice_id, "qty": s.qty, "price": s.price,
                            "scheduled_dt": s.scheduled_dt.isoformat(),
                            "status": s.status, "sent_ts": s.sent_ts, "attempt": s.attempt,
                            "retry": {
                                "max_attempts": s.retry.max_attempts,
                                "step_bps": s.retry.step_bps,
                                "timeout_sec": s.retry.timeout_sec,
                                "escalate_to_market": s.retry.escalate_to_market,
                                "attempts": s.retry.attempts,
                                "last_attempt_ts": s.retry.last_attempt_ts,
                                "market_order": s.retry.market_order,
                            },
                        }
                        for s in po.slices
                    ],
                }
                for po in self.pending
            ],
        }

    @classmethod
    def from_dict(cls, d: dict, persist_path: Optional[str] = None) -> "ExecutionScheduler":
        sc = cls(
            strategy=d.get("strategy", "twap"),
            window_min=d.get("window_min", 300),
            max_slices=d.get("max_slices", 10),
            overnight=d.get("overnight", True),
            retry_cfg=d.get("retry_cfg", {}),
            persist_path=persist_path,
        )
        for pd in d.get("pending", []):
            po = PendingOrder(
                code=pd["code"], name=pd.get("name", ""),
                side=OrderSide.BUY if str(pd.get("side", "")).lower().startswith("buy") else OrderSide.SELL,
                price=pd["price"], reason=pd.get("reason", ""),
                target_weight=pd.get("target_weight", 0.0), total_qty=pd.get("total_qty", 0),
            )
            for sd in pd.get("slices", []):
                rc = sd.get("retry", {})
                po.slices.append(Slice(
                    slice_id=sd["slice_id"], qty=sd["qty"], price=sd["price"],
                    scheduled_dt=datetime.fromisoformat(sd["scheduled_dt"]),
                    status=sd.get("status", "pending"), sent_ts=sd.get("sent_ts", 0.0),
                    attempt=sd.get("attempt", 0),
                    retry=LimitRetry(
                        max_attempts=rc.get("max_attempts", 3), step_bps=rc.get("step_bps", 8.0),
                        timeout_sec=rc.get("timeout_sec", 60.0),
                        escalate_to_market=rc.get("escalate_to_market", True),
                        attempts=rc.get("attempts", 0),
                        last_attempt_ts=rc.get("last_attempt_ts", 0.0),
                        market_order=rc.get("market_order", False),
                    ),
                ))
            sc.pending.append(po)
        return sc

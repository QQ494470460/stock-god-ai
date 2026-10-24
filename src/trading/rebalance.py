# -*- coding: utf-8 -*-
"""
调仓计算（Rebalance）—— 目标组合 vs 当前持仓 → 订单意图
========================================================

纯函数、无副作用、不碰网络：吃「交易计划 + 当前持仓 + 实时价」，吐「该卖什么、该买什么」。
把这段逻辑独立出来，是为了它**可以被完整单测**——全自动交易里最容易出错、
也最不能出错的就是"到底该下多少股"。

核心规则
--------
1. **退出**：持仓不在目标组合 → 清仓（跌出榜单就该走，这是量化的纪律）。
2. **减仓/加仓**：目标内但偏离超过容忍带 ±band 才动手，避免每天为 1% 偏差空转成本。
3. **先卖后买**：卖出回款计入可用预算，避免"没钱买"的假性资金不足。
4. **现金底线**：无论怎么买，账户现金不得低于 ``总资产 × cash_reserve_pct``。
5. **A 股细节**：整手（100 股）取整、T+1 可卖量封顶、小于最小交易金额的意图直接丢弃。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

from loguru import logger

from src.trading.broker import OrderSide, Position, AccountInfo
from src.trading.rules import sellable_quantity
from src.trading.trade_plan import TradePlan


@dataclass
class RebalanceIntent:
    """一条调仓意图（尚未下单，仅供决策与预演）"""
    code: str
    name: str
    side: OrderSide
    quantity: int
    price: float
    reason: str = ""
    target_weight: float = 0.0      # 目标权重
    current_value: float = 0.0      # 当前市值
    target_value: float = 0.0       # 目标市值

    @property
    def amount(self) -> float:
        return round(self.quantity * self.price, 2)

    @property
    def delta_value(self) -> float:
        """本次交易带来的目标市值变化（买入为正，卖出为负）"""
        return round(self.amount if self.side == OrderSide.BUY else -self.amount, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["amount"] = self.amount
        return d


def _floor_lot(qty: float, lot_size: int = 100) -> int:
    if qty <= 0 or lot_size <= 0:
        return 0
    return int(math.floor(qty / lot_size) * lot_size)


def compute_rebalance_orders(
    plan: TradePlan,
    positions: Sequence[Position],
    account: Optional[AccountInfo],
    prices: Dict[str, float],
    band_pct: float = 0.25,
    min_trade_amount: float = 5000.0,
    lot_size: int = 100,
    allow_buy: bool = True,
    allow_sell: bool = True,
    risk_scale: float = 1.0,
) -> List[RebalanceIntent]:
    """计算调仓订单意图（卖单在前，买单在后）。

    Args:
        plan: 交易计划（目标组合 + 权重 + 现金保留比例）
        positions: 当前持仓
        account: 账户信息（用于取总资产与可用现金）
        prices: code → 最新价；缺失则回退持仓现价，新标的缺价直接跳过
        band_pct: 偏离容忍带，0.25 表示偏离目标市值 25% 以内不动手
        min_trade_amount: 小于该金额（元）的意图丢弃，避免碎单
        lot_size: 一手股数
        allow_buy / allow_sell: 允许的方向（用于"只减不增"等场景）
        risk_scale: 组合敞口系数 0~1，由组合回撤连续推导（见
            ``src/trading/sizing.py``）。1.0=满仓参与，0.3=只用 30% 建仓、
            其余留现金。**不是清仓**——始终保留一部分敞口在场。

    Returns:
        List[RebalanceIntent]，已排序为「先卖后买」。
    """
    if not plan or not plan.targets:
        # 安全语义：空计划 = 按兵不动，**绝不**清仓。
        # 模型某天没选出票、或数据故障导致 picks 为空，不应该触发全仓卖出 ——
        # 否则一次数据抖动就能把整个组合清掉。要清仓请显式给出空仓计划。
        return []
    if account is None or account.total_asset <= 0:
        logger.debug("调仓跳过：账户总资产不可用")
        return []

    total_asset = float(account.total_asset)
    # 可投敞口：回撤越深，实际动用的总资产越少（连续渐进降仓，非清仓）。
    # 系数由 src.trading.sizing.portfolio_risk_scale 按组合回撤推导。
    deployable = total_asset * min(max(float(risk_scale), 0.0), 1.0)
    reserve = float(min(max(plan.cash_reserve_pct, 0.0), 0.95))
    pos_map: Dict[str, Position] = {
        p.code: p for p in (positions or []) if getattr(p, "quantity", 0) > 0
    }
    target_weight: Dict[str, float] = plan.weight_map
    target_name: Dict[str, str] = {t.code: (t.name or t.code) for t in plan.targets}

    def price_of(code: str) -> float:
        p = prices.get(code)
        if p and p > 0:
            return float(p)
        pos = pos_map.get(code)
        if pos and getattr(pos, "current_price", 0) > 0:
            return float(pos.current_price)
        return 0.0

    sells: List[RebalanceIntent] = []
    buys: List[RebalanceIntent] = []

    # ---------------- A. 退出：持仓不在目标组合 ----------------
    if allow_sell:
        for pos in pos_map.values():
            if pos.code in target_weight:
                continue
            price = price_of(pos.code)
            if price <= 0:
                continue
            qty = sellable_quantity(pos)
            if qty <= 0:
                logger.debug(f"调仓：{pos.code} 无可卖股（T+1 锁仓），本轮跳过")
                continue
            if qty * price < min_trade_amount:
                continue
            sells.append(RebalanceIntent(
                code=pos.code, name=pos.name or pos.code, side=OrderSide.SELL,
                quantity=qty, price=round(price, 3),
                reason="跌出目标组合，清仓",
                target_weight=0.0,
                current_value=float(getattr(pos, "market_value", 0) or qty * price),
                target_value=0.0,
            ))

    # ---------------- B. 目标内持仓：偏离过大则减仓 ----------------
    if allow_sell:
        for t in plan.targets:
            pos = pos_map.get(t.code)
            if not pos:
                continue
            price = price_of(t.code)
            if price <= 0:
                continue
            tgt_val = deployable * t.weight
            cur_val = float(getattr(pos, "market_value", 0) or (pos.quantity * price))
            if tgt_val <= 0:
                continue
            if cur_val <= tgt_val * (1.0 + band_pct):
                continue  # 在容忍带内
            delta = cur_val - tgt_val
            qty = _floor_lot(delta / price, lot_size)
            qty = min(qty, sellable_quantity(pos))
            if qty <= 0 or qty * price < min_trade_amount:
                continue
            sells.append(RebalanceIntent(
                code=t.code, name=t.name or pos.name or t.code, side=OrderSide.SELL,
                quantity=qty, price=round(price, 3),
                reason=f"超出目标仓位 {delta:.0f}元，减仓",
                target_weight=t.weight, current_value=cur_val, target_value=tgt_val,
            ))

    # ---------------- C. 买入：目标缺失或低于目标 ----------------
    if allow_buy:
        sell_proceeds = sum(s.amount for s in sells)
        budget = max(
            0.0,
            float(account.available_cash) + sell_proceeds - total_asset * reserve,
        )

        candidates: List[tuple] = []
        for t in plan.targets:
            pos = pos_map.get(t.code)
            price = price_of(t.code)
            if price <= 0:
                logger.debug(f"调仓：{t.code} 无可用价格，跳过买入")
                continue
            tgt_val = deployable * t.weight
            cur_val = float(getattr(pos, "market_value", 0) or (pos.quantity * price)) if pos else 0.0
            if cur_val >= tgt_val * (1.0 - band_pct):
                continue  # 已达目标（含容忍带）
            gap = tgt_val - cur_val
            candidates.append((gap, t, price, cur_val, tgt_val))

        # 缺口大的优先买（先补齐仓位，再考虑微调）
        candidates.sort(key=lambda x: x[0], reverse=True)

        for gap, t, price, cur_val, tgt_val in candidates:
            if budget <= 0:
                break
            spend = min(gap, budget)
            qty = _floor_lot(spend / price, lot_size)
            if qty <= 0:
                continue
            amount = qty * price
            if amount < min_trade_amount:
                continue
            if amount > budget:
                continue
            budget -= amount
            reason = ("目标组合新增，建仓" if cur_val <= 0
                      else f"低于目标仓位 {gap:.0f}元，补仓")
            buys.append(RebalanceIntent(
                code=t.code, name=t.name or t.code, side=OrderSide.BUY,
                quantity=qty, price=round(price, 3), reason=reason,
                target_weight=t.weight, current_value=cur_val, target_value=tgt_val,
            ))

    return sells + buys


def summarize_intents(intents: Sequence[RebalanceIntent]) -> str:
    """人类可读的调仓预演文本（UI / 推送用）"""
    if not intents:
        return "本轮无需调仓（已与目标组合一致）"
    sells = [i for i in intents if i.side == OrderSide.SELL]
    buys = [i for i in intents if i.side == OrderSide.BUY]
    lines = [f"调仓预演：卖出 {len(sells)} 笔 / 买入 {len(buys)} 笔"]
    for i in intents:
        arrow = "🔴卖出" if i.side == OrderSide.SELL else "🟢买入"
        lines.append(
            f"  {arrow} {i.code} {i.name}  {i.quantity}股 @ {i.price:.2f} "
            f"≈ {i.amount:,.0f}元  —— {i.reason}"
        )
    return "\n".join(lines)


def net_cash_flow(intents: Sequence[RebalanceIntent]) -> float:
    """净现金流：卖出回款 - 买入支出（正数表示回笼现金）"""
    out = 0.0
    for i in intents:
        out += i.amount if i.side == OrderSide.SELL else -i.amount
    return round(out, 2)


if __name__ == "__main__":  # 手动预演
    from src.trading.trade_plan import load_latest_plan
    p = load_latest_plan()
    if p:
        print(p.summary())

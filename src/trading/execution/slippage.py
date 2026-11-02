# -*- coding: utf-8 -*-
"""模拟盘滑点建模（对齐真实成交），opt-in。

默认行为（rules.apply_slippage 现状）：按金额比例上滑点、100% 成交。
开启 ``participation`` 模型后，滑点随「委托额 / 近 20 日成交额(ADV)」放大——
大单冲击成本更高，使模拟盘损益更贴近真实券商撮合，避免回测/纸面收益虚高。

该模块不修改任何默认路径：只有 loop 在 ``execution.smart_routing`` 启用且
``slippage_model='participation'`` 时，调用 ``set_active_slippage_model`` 注册后
才生效；未注册时 ``rules.apply_slippage`` 走原逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

_ACTIVE_MODEL = None  # type: Optional["SimSlippageModel"]


def set_active_slippage_model(model) -> None:
    """注册一个激活的滑点模型（None = 关闭，走 rules.apply_slippage 默认逻辑）。"""
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = model


def get_active_slippage_model():
    return _ACTIVE_MODEL


@dataclass
class SimSlippageModel:
    """参与度加权的滑点 + 可选部分成交模型。

    Attributes:
        base_bps: 基础滑点（基点）。
        participation: 委托额相对 ADV 的比例上限（用于放大冲击）。
        adv_provider: 可选 ``adv_of(code, amount) -> 近20日日均成交额``；给了才按
            参与度放大，否则退化为固定 base_bps（fail-open）。
        partial_fill: 是否模拟部分成交（默认 False = 100% 成交，贴近现状）。
        impact_coef: 冲击系数（参与度→额外滑点的放大倍数）。
        min_tick: 最小价格变动（A股 0.01）。
    """

    base_bps: float = 2.0
    participation: float = 0.01
    adv_provider: Optional[Callable[[str, float], float]] = None
    partial_fill: bool = False
    impact_coef: float = 0.5
    min_tick: float = 0.01

    def _impact_rate(self, code: str, amount: float) -> float:
        rate = self.base_bps / 10_000.0
        if self.adv_provider is not None and amount > 0:
            try:
                adv = self.adv_provider(code, amount)
                if adv and adv > 0:
                    part = min(amount / adv, 1.0)
                    # 参与度越高冲击越大（线性，封顶 1.0）
                    rate += self.impact_coef * part
            except Exception:
                pass
        return rate

    def fill_price(self, price: float, side, amount: float, code: str = "") -> float:
        """返回更接近真实的成交价（买更贵 / 卖更便宜）。"""
        if not price or price <= 0:
            return price
        s = str(getattr(side, "value", side)).lower()
        rate = self._impact_rate(code, amount)
        if s.startswith("buy"):
            adj = max(price * (1 + rate), price + self.min_tick)
        else:
            adj = min(price * (1 - rate), price - self.min_tick)
        return round(adj, 2)

    def fill_qty(self, requested: int) -> int:
        """返回实际成交量（默认 100% 成交）。"""
        if not self.partial_fill:
            return requested
        # 部分成交模拟：此处保持全量，便于上层按 returned<requested 触发重试。
        # 真实概率化部分成交可在 loop 调用处按参与度抽样，本模型仅提供钩子。
        return requested

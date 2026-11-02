# -*- coding: utf-8 -*-
"""非对称、非顺周期的「降风险」系数：只缩新买入，永不强制卖出。

为什么必须非顺周期
------------------
本项目已经用真实面板回测证伪过顺周期降仓：

    基准                 年化 +26.8%  夏普 1.19
    0.30 地板回撤斜坡    年化 +14.7%  夏普 0.99

原因很直白——价格跌 → 系数缩小 → 卖在低点 → 反弹时仓位不足。降风险的动作
本身在制造亏损。所以本模块的硬性设计约束是：

1. **只缩放「新买入」的手数**，从不生成卖出指令。已持仓就留在账户里等反弹。
2. **地板 >= 0.5**。再深的回撤也保留一半买入力，避免「清仓式空仓」。
3. **只在有意义的回撤（>= 12%）才介入**。1~2% 的日常波动不配触发任何调整，
   否则年化换手率会被噪音吃掉。
4. **恢复要滞回**。回落途中不来回抖动：一旦介入，必须修复到
   ``engage - release_hysteresis`` 以下才完全放开。

与 ``src/trading/sizing.py`` 的关系
-----------------------------------
``sizing.py`` 里已有的 ``portfolio_risk_scale`` 是按回撤连续缩放的通用版本
（floor 可到 0.30）。本模块是**更保守的补充件**：更高的介入线、更高的地板、
外加恢复滞回，用于在「不动生产代码」的前提下做 A/B 回测验证。
"""

from __future__ import annotations

import math

__all__ = ["asymmetric_derisk_scale", "DeriskRamp"]

#: 地板硬下限：任何参数下都不许把买入力压到 0.5 以下。
MIN_FLOOR = 0.5


def _clamp_floor(floor) -> float:
    try:
        f = float(floor)
    except (TypeError, ValueError):
        return 0.60
    if not math.isfinite(f):
        return 0.60
    return max(MIN_FLOOR, min(f, 1.0))


def _to_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def asymmetric_derisk_scale(
    drawdown_pct: float,
    engage_pct: float = 12.0,
    max_dd_pct: float = 20.0,
    floor: float = 0.60,
) -> float:
    """按回撤给出**新买入**的缩放系数，取值 ``[floor, 1.0]``。

    Parameters
    ----------
    drawdown_pct
        当前峰谷回撤，**百分数**（12.0 表示 -12%）。
    engage_pct
        介入线。回撤低于此值完全不干预，返回 1.0。
    max_dd_pct
        到达此回撤时压到地板。``<= engage_pct`` 时退化为阶跃。
    floor
        地板系数，**内部强制 >= 0.5**。

    Returns
    -------
    缩放系数：``< engage`` → 1.0；``>= max_dd`` → floor；中间线性斜坡。
    """
    dd = _to_float(drawdown_pct, 0.0)
    fl = _clamp_floor(floor)
    engage = max(_to_float(engage_pct, 12.0), 0.0)
    max_dd = _to_float(max_dd_pct, 20.0)

    if dd <= engage:
        return 1.0
    if max_dd <= engage:
        # 参数退化：不搞除零，直接阶跃到地板。
        return fl
    if dd >= max_dd:
        return fl

    span = max_dd - engage
    t = (dd - engage) / span
    scale = 1.0 - t * (1.0 - fl)
    return max(fl, min(scale, 1.0))


class DeriskRamp:
    """带**恢复滞回**的降风险斜坡（有状态）。

    行为
    ----
    * 未介入时：``dd >= engage_pct`` 才介入，介入后系数 < 1.0。
    * 介入后：斜坡按**介入以来见过的最深回撤**计算（棘轮）。所以从 13% 反弹
      到 11% 时系数不会立刻回到 1.0——它保持 13% 对应的水平，避免「刚减完
      仓就因为小反弹又满仓」的来回摩擦。
    * 只有回撤修复到 ``engage_pct - release_hysteresis_pct`` 或以下，才完全
      放开回 1.0 并重置棘轮。

    再次强调：**本类只用于缩放新买入的规模（手数/金额），任何调用方都不得
    用它去生成卖出指令**。已持仓不因为系数下降而被清算，这是非顺周期的
    核心——降风险不能等于卖在底部。
    """

    def __init__(
        self,
        engage_pct: float = 12.0,
        max_dd_pct: float = 20.0,
        floor: float = 0.60,
        release_hysteresis_pct: float = 2.0,
    ) -> None:
        self.engage_pct = max(_to_float(engage_pct, 12.0), 0.0)
        self.max_dd_pct = _to_float(max_dd_pct, 20.0)
        self.floor = _clamp_floor(floor)
        self.release_hysteresis_pct = max(
            _to_float(release_hysteresis_pct, 2.0), 0.0
        )
        #: 放开线：回撤修复到这条线以下才解除介入。
        self.release_pct = max(self.engage_pct - self.release_hysteresis_pct, 0.0)
        self._scale = 1.0
        self._engaged = False
        self._peak_dd = 0.0

    # -- 查询 ---------------------------------------------------------------
    def current(self) -> float:
        """最近一次 ``update`` 得到的系数（未调用过则为 1.0）。"""
        return self._scale

    @property
    def engaged(self) -> bool:
        """当前是否处于介入状态。"""
        return self._engaged

    @property
    def peak_drawdown_pct(self) -> float:
        """本次介入以来见过的最深回撤（百分数）。"""
        return self._peak_dd

    # -- 状态迁移 -----------------------------------------------------------
    def update(self, drawdown_pct: float) -> float:
        """喂入当前回撤，返回新的新买入缩放系数。"""
        dd = _to_float(drawdown_pct, 0.0)

        if not self._engaged:
            if dd >= self.engage_pct:
                self._engaged = True
                self._peak_dd = dd
                self._scale = asymmetric_derisk_scale(
                    dd, self.engage_pct, self.max_dd_pct, self.floor
                )
            else:
                self._scale = 1.0
            return self._scale

        # 已介入：先看是否修复到放开线以下。
        if dd <= self.release_pct:
            self._engaged = False
            self._peak_dd = 0.0
            self._scale = 1.0
            return self._scale

        # 仍在滞回区间内：棘轮只朝更深的方向走，系数只降不升。
        if dd > self._peak_dd:
            self._peak_dd = dd
        self._scale = asymmetric_derisk_scale(
            self._peak_dd, self.engage_pct, self.max_dd_pct, self.floor
        )
        return self._scale

    def reset(self) -> None:
        """回到初始状态（全部放开、棘轮清零）。"""
        self._scale = 1.0
        self._engaged = False
        self._peak_dd = 0.0

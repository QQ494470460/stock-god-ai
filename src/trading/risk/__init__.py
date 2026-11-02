# -*- coding: utf-8 -*-
"""风险增强模块（默认关闭，未接线到生产）。

所有内容都是**纯函数 + 极小的有状态类**，只为在
``src/core/exec_backtest.py`` 里做参数验证而存在。生产代码不 import 本包。

对外 API::

    ewma_vol_annualized(returns, span=20, periods_per_year=252) -> float
    realized_vol_from_prices(closes, window=20, periods_per_year=252) -> float
    make_vol_provider(price_history_fn, window=20) -> Callable[[str], Optional[float]]

    asymmetric_derisk_scale(drawdown_pct, engage_pct=12.0, max_dd_pct=20.0, floor=0.60) -> float
    DeriskRamp(engage_pct=12.0, max_dd_pct=20.0, floor=0.60, release_hysteresis_pct=2.0)

    portfolio_vol_target_scale(realized_portfolio_vol, target_vol=0.15, floor=0.5, ceil_=1.0) -> float
    blend_scales(a, b, weight_b=0.5) -> float

共同约束：非顺周期。降风险只缩放**新买入**，永不强制卖出；地板 >= 0.5；
仅在回撤 >= 12% 时介入，且恢复带滞回。
"""

from __future__ import annotations

from src.trading.risk.derisk import DeriskRamp, asymmetric_derisk_scale
from src.trading.risk.portfolio_vol import (
    blend_scales,
    portfolio_vol_target_scale,
)
from src.trading.risk.volatility import (
    ewma_vol_annualized,
    make_vol_provider,
    realized_vol_from_prices,
)

__all__ = [
    "ewma_vol_annualized",
    "realized_vol_from_prices",
    "make_vol_provider",
    "asymmetric_derisk_scale",
    "DeriskRamp",
    "portfolio_vol_target_scale",
    "blend_scales",
]

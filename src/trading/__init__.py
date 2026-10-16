# -*- coding: utf-8 -*-
"""选股神AI — 量化自动交易模块"""

from src.trading.broker import (
    BrokerAPI, Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo, TickData
)
from src.trading.yuekai import YuekaiBroker
from src.trading.account import AccountManager
from src.trading.executor import TradeExecutor
from src.trading.autotrader import AutoTrader
from src.trading.signals import (
    QuantSignal, QuantSignalGenerator,
    StrategyMode, create_strategy_generator,
)

__all__ = [
    "BrokerAPI", "Order", "OrderSide", "OrderType", "OrderStatus",
    "Position", "AccountInfo", "TickData",
    "YuekaiBroker", "AccountManager", "TradeExecutor", "AutoTrader",
    "QuantSignal", "QuantSignalGenerator",
    "StrategyMode", "create_strategy_generator",
]

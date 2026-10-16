# -*- coding: utf-8 -*-
"""
券商抽象基类
定义统一的交易接口规范，任何券商实现只需继承 BrokerAPI
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"           # 限价
    MARKET = "market"         # 市价（五档即成剩撤）
    MARKET_BEST5 = "best5"    # 最优五档即成剩撤
    MARKET_OPPONENT = "opponent"  # 对手方最优
    MARKET_OWN = "own"        # 本方最优


class OrderStatus(str, Enum):
    PENDING = "pending"        # 待报
    SUBMITTED = "submitted"    # 已报
    PARTIAL = "partial"        # 部分成交
    FILLED = "filled"          # 全部成交
    CANCELLED = "cancelled"    # 已撤单
    REJECTED = "rejected"      # 拒单
    ERROR = "error"            # 异常


@dataclass
class Order:
    """订单"""
    order_id: str                       # 订单ID
    code: str                           # 股票代码 (000001.SZ)
    name: str = ""                      # 股票名称
    side: OrderSide = OrderSide.BUY     # 买卖方向
    order_type: OrderType = OrderType.LIMIT  # 订单类型
    price: float = 0.0                  # 委托价格
    quantity: int = 0                   # 委托数量（股）
    filled_quantity: int = 0            # 已成交数量
    status: OrderStatus = OrderStatus.PENDING
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    strategy_tag: str = ""              # 策略标签
    ai_score: float = 0.0               # AI评分（用于追踪）
    remark: str = ""                    # 备注
    raw_response: dict = field(default_factory=dict)  # 券商原始回报


@dataclass
class Position:
    """持仓"""
    code: str
    name: str
    quantity: int           # 持仓数量
    available: int          # 可用数量
    avg_cost: float         # 持仓成本
    current_price: float    # 当前价
    market_value: float     # 市值
    pnl: float              # 浮动盈亏
    pnl_pct: float          # 盈亏比例
    today_quantity: int = 0 # 今日买入量


@dataclass
class AccountInfo:
    """账户信息"""
    account_id: str
    total_asset: float          # 总资产
    available_cash: float       # 可用资金
    frozen_cash: float          # 冻结资金
    market_value: float         # 持仓市值
    total_pnl: float            # 累计盈亏
    today_pnl: float            # 当日盈亏
    today_trades: int = 0       # 今日成交笔数
    risk_ratio: float = 0.0     # 风险度


@dataclass
class TickData:
    """实时行情"""
    code: str
    name: str
    last_price: float
    open: float
    high: float
    low: float
    pre_close: float
    volume: int
    amount: float
    bid1: float = 0.0
    bid1_vol: int = 0
    ask1: float = 0.0
    ask1_vol: int = 0
    change_pct: float = 0.0
    time: Optional[datetime] = None


class BrokerCallback(ABC):
    """券商回调基类 — 接收推送消息"""
    
    def on_order_update(self, order: Order):
        """订单状态更新"""
        pass
    
    def on_trade(self, order: Order):
        """成交回报"""
        pass
    
    def on_account_update(self, account: AccountInfo):
        """账户资金变动"""
        pass
    
    def on_error(self, code: int, message: str):
        """错误回调"""
        pass
    
    def on_connected(self):
        """连接成功"""
        pass
    
    def on_disconnected(self):
        """连接断开"""
        pass


class BrokerAPI(ABC):
    """券商API抽象基类"""
    
    def __init__(self):
        self._connected = False
        self._callbacks: List[BrokerCallback] = []
    
    @property
    def connected(self) -> bool:
        return self._connected
    
    def register_callback(self, cb: BrokerCallback):
        """注册回调"""
        self._callbacks.append(cb)
    
    def _notify_order(self, order: Order):
        for cb in self._callbacks:
            try:
                cb.on_order_update(order)
            except Exception:
                pass
    
    def _notify_trade(self, order: Order):
        for cb in self._callbacks:
            try:
                cb.on_trade(order)
            except Exception:
                pass
    
    def _notify_account(self, account: AccountInfo):
        for cb in self._callbacks:
            try:
                cb.on_account_update(account)
            except Exception:
                pass
    
    def _notify_error(self, code: int, msg: str):
        for cb in self._callbacks:
            try:
                cb.on_error(code, msg)
            except Exception:
                pass
    
    def _notify_connected(self):
        for cb in self._callbacks:
            try:
                cb.on_connected()
            except Exception:
                pass
    
    def _notify_disconnected(self):
        for cb in self._callbacks:
            try:
                cb.on_disconnected()
            except Exception:
                pass
    
    # ========== 连接管理 ==========
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """连接券商交易系统"""
        ...
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        ...
    
    # ========== 账户查询 ==========
    @abstractmethod
    def query_account(self) -> Optional[AccountInfo]:
        """查询账户资金"""
        ...
    
    @abstractmethod
    def query_positions(self) -> List[Position]:
        """查询持仓"""
        ...
    
    @abstractmethod
    def query_orders(self, status_filter: str = "all") -> List[Order]:
        """查询订单"""
        ...
    
    # ========== 交易操作 ==========
    @abstractmethod
    def place_order(
        self,
        code: str,
        side: OrderSide,
        order_type: OrderType,
        price: float,
        quantity: int,
        strategy_tag: str = "",
    ) -> Optional[Order]:
        """下单"""
        ...
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        ...
    
    @abstractmethod
    def cancel_all(self) -> int:
        """撤销全部未成交订单，返回撤单数"""
        ...
    
    # ========== 行情 ==========
    @abstractmethod
    def get_tick(self, code: str) -> Optional[TickData]:
        """获取实时行情"""
        ...
    
    # ========== 工具 ==========
    @abstractmethod
    def get_broker_name(self) -> str:
        """券商名称"""
        ...
    
    def is_trading_time(self) -> bool:
        """是否在 A 股可交易时段（交易日 + 9:30-11:30 / 13:00-15:00）

        注意：原来用 ``weekday() < 5`` 判断交易日，会把国庆、春节等法定假日
        也当成交易日 → 自动交易在休市日下单全是废单。
        现改为查询真实交易日历（带本地缓存）。日历不可用时回退旧逻辑，
        **绝不因为日历拿不到就阻断交易**。
        """
        try:
            from src.trading.calendar import can_trade_now
            return can_trade_now()
        except Exception:  # noqa: BLE001
            from datetime import datetime, time
            now = datetime.now()
            t = now.time()
            morning = time(9, 30) <= t <= time(11, 30)
            afternoon = time(13, 0) <= t <= time(15, 0)
            weekday = now.weekday() < 5
            return weekday and (morning or afternoon)

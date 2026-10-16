# -*- coding: utf-8 -*-
"""
粤开证券 — miniQMT 实现
通过 xtquant 包连接 miniQMT 客户端，实现行情 + 交易
"""

import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

from src.trading.broker import (
    BrokerAPI, BrokerCallback,
    Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo, TickData,
)

# xtquant 是可选依赖，运行时检查
try:
    from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
    from xtquant.xttype import StockAccount
    from xtquant import xtconstant
    XT_AVAILABLE = True
except ImportError:
    XT_AVAILABLE = False
    XtQuantTrader = None
    XtQuantTraderCallback = object
    StockAccount = None
    xtconstant = None
    from loguru import logger
    logger.warning("xtquant 未安装，YuekaiBroker 将以模拟模式运行")

# 资金账号类型映射
ACCOUNT_TYPE_MAP = {"stock": StockAccount} if XT_AVAILABLE else {"stock": None}


# order_type / status 映射（xtquant可用时引用常量，否则用int）
if XT_AVAILABLE:
    ORDER_TYPE_MAP = {
        OrderType.LIMIT: xtconstant.FIX_PRICE,
        OrderType.MARKET: xtconstant.MARKET_SH_CONVERT_5_CANCEL,
        OrderType.MARKET_BEST5: xtconstant.MARKET_SH_CONVERT_5_CANCEL,
    }
else:
    ORDER_TYPE_MAP = {OrderType.LIMIT: 11, OrderType.MARKET: 22, OrderType.MARKET_BEST5: 22}

STATUS_MAP = {
    48: OrderStatus.PENDING, 49: OrderStatus.SUBMITTED,
    50: OrderStatus.SUBMITTED, 51: OrderStatus.PARTIAL,
    52: OrderStatus.PARTIAL, 53: OrderStatus.FILLED,
    54: OrderStatus.CANCELLED, 55: OrderStatus.CANCELLED,
    56: OrderStatus.CANCELLED, 57: OrderStatus.REJECTED,
}


class _XtCallback(XtQuantTraderCallback):
    """miniQMT 回调适配器 — 桥接 xtquant 事件到 BrokerCallback"""
    
    def __init__(self, broker: "YuekaiBroker"):
        super().__init__()
        self._broker = broker
    
    def on_disconnected(self):
        logger.warning("⛓️‍💥 miniQMT 连接断开")
        self._broker._connected = False
        self._broker._notify_disconnected()
    
    def on_stock_order(self, order):
        """委托回报"""
        o = _parse_xt_order(order)
        self._broker._update_order_cache(o)
        self._broker._notify_order(o)
    
    def on_stock_trade(self, trade):
        """成交回报"""
        o = _parse_xt_order(trade)
        o.status = OrderStatus.FILLED if trade.m_traded_volume >= trade.m_order_volume else OrderStatus.PARTIAL
        o.filled_quantity = trade.m_traded_volume
        self._broker._update_order_cache(o)
        self._broker._notify_trade(o)
    
    def on_stock_asset(self, asset):
        """资金变动"""
        self._broker._query_asset_cache = {
            "total_asset": asset.m_total_asset,
            "available_cash": asset.m_available_cash,
            "frozen_cash": asset.m_frozen_cash,
            "market_value": asset.m_market_value,
        }
    
    def on_stock_position(self, position):
        """持仓变动"""
        pass
    
    def on_account_error(self, code, msg):
        logger.error(f"miniQMT 错误 [{code}]: {msg}")
        self._broker._notify_error(code, msg)


def _parse_xt_order(raw) -> Order:
    """将 xtquant 委托对象转为统一 Order"""
    side = OrderSide.BUY if raw.m_direction == 1 else OrderSide.SELL
    status = STATUS_MAP.get(raw.m_order_status, OrderStatus.PENDING)
    
    # 拼接委托编号
    order_id = f"{raw.m_order_id}" if hasattr(raw, "m_order_id") else f"{raw.m_strategy_id}_{raw.m_order_remark}"
    
    return Order(
        order_id=order_id,
        code=raw.m_stock_code,
        name=getattr(raw, "m_stock_name", ""),
        side=side,
        order_type=OrderType.LIMIT,  # 默认限价
        price=raw.m_price,
        quantity=raw.m_order_volume,
        filled_quantity=getattr(raw, "m_traded_volume", 0),
        status=status,
        create_time=datetime.now(),
        update_time=datetime.now(),
        raw_response={"order_id": order_id, "status": raw.m_order_status},
    )


class YuekaiBroker(BrokerAPI):
    """粤开证券 miniQMT 实现
    
    需要在启动前：
    1. 开通粤开证券 QMT 权限
    2. 将 xtquant 包复制到 Python site-packages
    3. 启动 XtMiniQmt.exe 极简模式客户端
    """
    
    BROKER_NAME = "粤开证券"
    
    def __init__(self):
        super().__init__()
        self._trader: Optional[XtQuantTrader] = None
        self._callback = _XtCallback(self)
        self._account: Optional = None  # StockAccount
        self._session_id = int(time.time() * 1000) % 1000000000
        self._orders_cache: Dict[str, Order] = {}
        self._query_asset_cache: dict = {}
        self._lock = threading.Lock()
        self._simulated = not XT_AVAILABLE
        self._mode = "auto"          # auto / sim / live（记录本次连接采用的模式）
        self._sim_reason = ""        # 模拟连接的说明（便于 UI 透明展示）
        self._last_error = ""        # 最近一次连接失败原因
    
    # ========== 连接管理 ==========
    
    def connect(
        self,
        qmt_path: str = "",
        account_id: str = "",
        password: str = "",
        mode: str = "auto",
    ) -> bool:
        """连接粤开证券 miniQMT

        Args:
            qmt_path: QMT安装目录下的 userdata_mini 路径
                      例如: D:\\QMT\\userdata_mini
            account_id: 资金账号
            password: 交易密码（部分券商不需要）
            mode: 连接模式
                - "auto"（默认）：装了 xtquant 就尝试实盘连接，否则自动降级模拟
                - "sim" ：强制模拟盘（无需 xtquant / QMT，永远可用）
                - "live"：强制实盘，必须有 xtquant 且 QMT 可连，否则失败
        """
        self._mode = mode
        self._last_error = ""

        # —— 强制模拟：无需任何依赖，直接可用（UI「模拟模式」走这里）——
        if mode == "sim":
            self._simulated = True
            self._sim_reason = "用户选择模拟模式"
            self._connected = True
            self._notify_connected()
            logger.info("🧪 已连接【模拟盘】（无需 xtquant / QMT）")
            return True

        # —— xtquant 缺失 ——
        if not XT_AVAILABLE:
            if mode == "live":
                self._simulated = False
                self._connected = False
                self._last_error = "xtquant 未安装，无法连接实盘"
                logger.error("❌ 实盘模式需要 xtquant（当前未安装）。"
                             "请将 xtquant 放入 site-packages 并启动 QMT 客户端后重试。")
                return False
            # auto：降级模拟
            self._simulated = True
            self._sim_reason = "xtquant 未安装，自动降级为模拟"
            self._connected = True
            self._notify_connected()
            logger.warning("⚠️ xtquant 未安装，已自动降级为【模拟盘】")
            return True

        # —— 有 xtquant：走实盘连接 ——
        self._simulated = False
        if not qmt_path or not Path(qmt_path).exists():
            self._last_error = f"QMT路径不存在: {qmt_path}"
            logger.error(self._last_error)
            return False
        
        try:
            logger.info(f"🔗 连接粤开证券 miniQMT → {qmt_path}")
            
            self._session_id = int(time.time() * 1000) % 1000000000
            self._trader = XtQuantTrader(str(Path(qmt_path)), self._session_id)
            self._trader.register_callback(self._callback)
            
            # 启动交易线程
            self._trader.start()
            
            # 建立连接（等待最多10秒）
            connect_result = self._trader.connect()
            if connect_result != 0:
                logger.error(f"miniQMT 连接失败: {connect_result}")
                return False
            
            # 订阅账号
            if account_id:
                self._account = StockAccount(account_id, "STOCK")
                subscribe_result = self._trader.subscribe(self._account)
                if subscribe_result != 0:
                    logger.error(f"账号订阅失败: {subscribe_result}")
                    return False
                logger.info(f"✅ 账号已订阅: {account_id}")
            
            # 查询一次资产确认连接
            time.sleep(1)
            _ = self._trader.query_stock_asset(self._account)
            
            self._connected = True
            self._notify_connected()
            logger.info("✅ 粤开证券 miniQMT 连接成功")
            return True
            
        except Exception as e:
            logger.error(f"粤开证券连接失败: {e}")
            self._connected = False
            self._last_error = str(e)
            return False
    
    def disconnect(self):
        if self._trader:
            try:
                if self._account:
                    self._trader.unsubscribe(self._account)
            except Exception:
                pass
            try:
                self._trader.stop()
            except Exception:
                pass
            self._trader = None
        
        self._connected = False
        self._notify_disconnected()
        logger.info("🔌 粤开证券已断开")
    
    # ========== 账户查询 ==========
    
    def query_account(self) -> Optional[AccountInfo]:
        if self._simulated:
            return AccountInfo(
                account_id="sim_000000",
                total_asset=1000000.0,
                available_cash=500000.0,
                frozen_cash=0.0,
                market_value=500000.0,
                total_pnl=0.0,
                today_pnl=0.0,
            )
        
        if not self._connected or not self._trader:
            return None
        
        try:
            asset = self._trader.query_stock_asset(self._account)
            if not asset:
                return None
            return AccountInfo(
                account_id=asset.m_account_id,
                total_asset=asset.m_total_asset,
                available_cash=asset.m_available_cash,
                frozen_cash=asset.m_frozen_cash,
                market_value=asset.m_market_value,
                total_pnl=getattr(asset, "m_total_pnl", 0.0),
                today_pnl=getattr(asset, "m_today_pnl", 0.0),
            )
        except Exception as e:
            logger.error(f"查询账户失败: {e}")
            return None
    
    def query_positions(self) -> List[Position]:
        if self._simulated:
            return []
        
        if not self._connected or not self._trader:
            return []
        
        try:
            raw_positions = self._trader.query_stock_positions(self._account)
            result = []
            for p in raw_positions or []:
                mkt_val = p.m_volume * p.m_last_price
                pnl = (p.m_last_price - p.m_avg_price) * p.m_volume
                pnl_pct = (p.m_last_price / p.m_avg_price - 1) * 100 if p.m_avg_price > 0 else 0
                
                result.append(Position(
                    code=p.m_stock_code,
                    name=getattr(p, "m_stock_name", ""),
                    quantity=p.m_volume,
                    available=p.m_can_use_volume,
                    avg_cost=p.m_avg_price,
                    current_price=p.m_last_price,
                    market_value=mkt_val,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    today_quantity=getattr(p, "m_today_volume", 0),
                ))
            return result
        except Exception as e:
            logger.error(f"查询持仓失败: {e}")
            return []
    
    def query_orders(self, status_filter: str = "all") -> List[Order]:
        if self._simulated:
            with self._lock:
                return list(self._orders_cache.values())
        
        if not self._connected or not self._trader:
            return []
        
        try:
            raw_orders = self._trader.query_stock_orders(self._account)
            result = []
            for o in raw_orders or []:
                order = _parse_xt_order(o)
                self._update_order_cache(order)
                result.append(order)
            return result
        except Exception as e:
            logger.error(f"查询订单失败: {e}")
            return list(self._orders_cache.values())
    
    # ========== 交易操作 ==========
    
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
        if self._simulated:
            return self._place_simulated_order(code, side, order_type, price, quantity, strategy_tag)
        
        if not self._connected or not self._trader:
            logger.error("未连接券商，无法下单")
            return None
        
        try:
            price_type = ORDER_TYPE_MAP.get(order_type, xtconstant.FIX_PRICE)
            direction = xtconstant.STOCK_BUY if side == OrderSide.BUY else xtconstant.STOCK_SELL
            
            # quantity 必须是 100 的整数倍（A股）
            if quantity % 100 != 0:
                quantity = (quantity // 100) * 100
                logger.warning(f"⚠️ 下单数量调整为整手: {quantity}股")
            
            if quantity < 100:
                logger.error("下单数量不足1手（100股）")
                return None
            
            order_id = self._trader.order_stock(
                self._account,
                code,
                direction,
                price_type,
                price,
                quantity,
                strategy_tag or "stock_god_ai",
                "",
            )
            
            if order_id < 0:
                logger.error(f"下单失败: 错误码 {order_id}")
                return None
            
            order = Order(
                order_id=str(order_id),
                code=code,
                side=side,
                order_type=order_type,
                price=price,
                quantity=quantity,
                status=OrderStatus.SUBMITTED,
                create_time=datetime.now(),
                update_time=datetime.now(),
                strategy_tag=strategy_tag,
            )
            self._update_order_cache(order)
            
            emoji = "🟢" if side == OrderSide.BUY else "🔴"
            logger.info(f"{emoji} 下单: {code} {side.value.upper()} {quantity}股 @ {price:.2f}")
            return order
            
        except Exception as e:
            logger.error(f"下单异常: {e}")
            return None
    
    def _place_simulated_order(self, code, side, order_type, price, quantity, strategy_tag):
        """模拟下单"""
        import uuid
        oid = f"sim_{uuid.uuid4().hex[:8]}"
        order = Order(
            order_id=oid,
            code=code,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            filled_quantity=quantity,  # 模拟全部成交
            status=OrderStatus.FILLED,
            create_time=datetime.now(),
            update_time=datetime.now(),
            strategy_tag=strategy_tag,
            remark="【模拟交易】",
        )
        self._update_order_cache(order)
        logger.info(f"🧪 模拟下单: {code} {side.value.upper()} {quantity}股 @ {price:.2f}")
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        if self._simulated:
            logger.info(f"🧪 模拟撤单: {order_id}")
            return True
        
        if not self._trader:
            return False
        
        try:
            result = self._trader.cancel_stock_order(self._account, int(order_id))
            logger.info(f"撤单请求: {order_id}, result={result}")
            return result == 0
        except Exception as e:
            logger.error(f"撤单失败: {e}")
            return False
    
    def cancel_all(self) -> int:
        if self._simulated:
            return 0
        
        if not self._trader:
            return 0
        
        try:
            orders = self.query_orders("pending")
            count = 0
            for o in orders:
                if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
                    if self.cancel_order(o.order_id):
                        count += 1
            logger.info(f"批量撤单: {count} 笔")
            return count
        except Exception as e:
            logger.error(f"批量撤单失败: {e}")
            return 0
    
    # ========== 行情 ==========
    
    def get_tick(self, code: str) -> Optional[TickData]:
        """获取实时行情（通过 xtdata）"""
        if self._simulated:
            return None
        
        try:
            from xtquant import xtdata
            
            # 确保代码格式: 000001.SZ / 600519.SH
            if "." not in code:
                code = f"{code}.{'SZ' if code.startswith(('0','3','1')) else 'SH'}"
            
            data = xtdata.get_full_tick([code])
            if not data or code not in data:
                return None
            
            t = data[code]
            if not t:
                return None
            
            change_pct = (t["lastPrice"] / t["lastClose"] - 1) * 100 if t.get("lastClose", 0) > 0 else 0
            
            return TickData(
                code=code,
                name="",
                last_price=t.get("lastPrice", 0),
                open=t.get("open", 0),
                high=t.get("high", 0),
                low=t.get("low", 0),
                pre_close=t.get("lastClose", 0),
                volume=t.get("volume", 0),
                amount=t.get("amount", 0),
                bid1=t.get("bid1", 0),
                bid1_vol=t.get("bid1Vol", 0),
                ask1=t.get("ask1", 0),
                ask1_vol=t.get("ask1Vol", 0),
                change_pct=change_pct,
                time=datetime.now(),
            )
        except Exception as e:
            logger.error(f"获取行情失败 {code}: {e}")
            return None
    
    # ========== 内部工具 ==========
    
    def _update_order_cache(self, order: Order):
        with self._lock:
            self._orders_cache[order.order_id] = order
            # 清理超过100条的旧记录
            if len(self._orders_cache) > 200:
                sorted_orders = sorted(
                    self._orders_cache.values(),
                    key=lambda o: o.create_time or datetime.min,
                    reverse=True
                )
                self._orders_cache = {o.order_id: o for o in sorted_orders[:100]}
    
    def get_broker_name(self) -> str:
        return self.BROKER_NAME
    
    @property
    def is_simulated(self) -> bool:
        return self._simulated

    @property
    def mode(self) -> str:
        """本次连接采用的模式：auto / sim / live"""
        return self._mode

    @property
    def sim_reason(self) -> str:
        """模拟连接的说明（透明展示给用户）"""
        return self._sim_reason

    @property
    def last_error(self) -> str:
        return self._last_error

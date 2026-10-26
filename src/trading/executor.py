# -*- coding: utf-8 -*-
"""
交易执行引擎
风控检查 + 委托生成 + 止盈止损 + 仓位管理
"""

import time
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger

from src.trading.broker import (
    BrokerAPI, Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo, TickData,
)
from src.trading.rules import check_buyable, check_sellable, sellable_quantity
from src.trading.triggers import log_trigger, _classify


@dataclass
class RiskConfig:
    """风控配置"""
    # 仓位限制
    max_position_pct: float = 0.30        # 单票最大仓位（占总资产比例）
    max_total_position_pct: float = 0.80  # 总持仓上限
    min_cash_reserve: float = 10000.0     # 最低现金保留（元）
    
    # 止损止盈
    stop_loss_pct: float = -5.0           # 止损线（%），如 -5%
    stop_profit_pct: float = 15.0         # 止盈线（%），如 +15%
    trailing_stop_pct: float = -3.0       # 移动止损（%，回撤多少触发）
    
    # 日内限制
    max_daily_trades: int = 20            # 日最大成交笔数
    max_daily_loss: float = 5000.0        # 单日最大亏损（元），超限停止交易
    max_daily_loss_pct: float = -3.0      # 单日最大亏损比例（%）
    
    # 交易时段
    trading_hours_only: bool = True       # 仅交易时段可下单（含真实交易日历，排除法定假日）
    
    # 实盘交易规则（停牌/涨跌停/T+1）——关掉则退回旧行为，仅建议调试时用
    enable_market_rules: bool = True
    
    # 下单风控
    max_order_amount: float = 500000.0    # 单笔最大金额
    min_order_interval: float = 2.0       # 最小下单间隔（秒）
    price_deviation_pct: float = 3.0      # 限价单价格偏离预警（%）
    
    # AI 信号过滤
    min_ai_score_buy: float = 55.0        # 买入最低AI评分
    max_ai_score_sell: float = 45.0       # 卖出最高AI评分
    require_agreement: bool = True        # 需要XGB/LGB一致（agreement>=0.8）


@dataclass
class RiskCheckResult:
    """风控检查结果"""
    passed: bool
    reason: str = ""
    adjusted_quantity: int = 0
    details: dict = field(default_factory=dict)


class TradeExecutor:
    """交易执行引擎 — 在 Broker 之上加风控、仓位管理、止损止盈"""
    
    def __init__(
        self,
        broker: BrokerAPI,
        risk_config: Optional[RiskConfig] = None,
    ):
        self.broker = broker
        self.config = risk_config or RiskConfig()
        
        # 日内统计
        self._today_stats = {
            "trade_count": 0,
            "total_pnl": 0.0,
            "start_asset": 0.0,
            "stop_trading": False,
            "stop_reason": "",
        }
        self._last_order_time: Optional[datetime] = None
        # 移动止损用的历史最高价：code → high。
        # 只存数字、绝不写回 Position —— 旧实现直接把高点写进持仓对象的
        # current_price，会污染调用方（市值/盈亏），且清仓后不清除。
        self._high_water: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # 重置每日统计
        self._reset_daily_stats()
        
        logger.info("🛡️ TradeExecutor 已初始化")
    
    # ========== 止盈止损检查 ==========
    
    def check_stop_conditions(
        self,
        positions: Optional[List[Position]] = None,
        prices: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        """检查所有持仓是否需要止损/止盈/移动止损
        
        Args:
            positions: 注入持仓列表（None 则向券商查询）。闭环引擎传入
                已同步的持仓 + 行情，使纸交易/离线环境也能跑同一套止损逻辑。
            prices: 注入 code → 最新价（None 则逐只向券商取 tick）。
        
        Returns:
            [{code, reason, action, pnl_pct}, ...]
        """
        alerts = []
        if positions is None:
            positions = self.broker.query_positions()

        # 只跟踪当前仍持有的标的：已清仓的必须立刻从高点表剔除。
        # 不剔除的话，同一只票卖出后再买回，会拿上一轮的峰值当参照，
        # 建仓当天就被判成「回撤」而直接触发移动止损（真实的废单来源）。
        held = {p.code for p in positions if getattr(p, "quantity", 0) > 0}
        for _code in [c for c in self._high_water if c not in held]:
            self._high_water.pop(_code, None)

        for pos in positions:
            if pos.quantity <= 0:
                continue
            
            # 获取实时行情（优先用注入价，避免重复请求）
            if prices is not None:
                current_price = float(prices.get(pos.code) or 0) or pos.current_price
            else:
                tick = self.broker.get_tick(pos.code)
                current_price = tick.last_price if tick and tick.last_price > 0 else pos.current_price
            
            # 计算盈亏比
            pnl_pct = (current_price / pos.avg_cost - 1) * 100 if pos.avg_cost > 0 else 0
            
            # 跟踪历史最高价（移动止损用）——单独存数字，不碰持仓对象
            key = pos.code
            tracked_high = max(float(self._high_water.get(key, 0.0) or 0.0), current_price)
            self._high_water[key] = tracked_high
            
            drawdown = (current_price / tracked_high - 1) * 100 if tracked_high > 0 else 0

            # 可卖量必须过 T+1：当日买入的份额当日不可卖，止盈止损也不能例外
            # （否则实盘必废单，纸交易会在 sell 里被二次拦截、白跑一轮）。
            qty = sellable_quantity(pos)
            if qty <= 0:
                continue

            # 止损判断
            if pnl_pct <= self.config.stop_loss_pct:
                alerts.append({
                    "code": pos.code, "name": pos.name,
                    "reason": f"止损 {pnl_pct:.1f}%",
                    "action": "sell", "pnl_pct": pnl_pct,
                    "quantity": qty,
                })
            # 止盈判断
            elif pnl_pct >= self.config.stop_profit_pct:
                alerts.append({
                    "code": pos.code, "name": pos.name,
                    "reason": f"止盈 {pnl_pct:.1f}%",
                    "action": "sell", "pnl_pct": pnl_pct,
                    "quantity": qty,
                })
            # 移动止损
            elif pnl_pct > 0 and drawdown <= self.config.trailing_stop_pct:
                alerts.append({
                    "code": pos.code, "name": pos.name,
                    "reason": f"移动止损 回撤{drawdown:.1f}% (最高+{(tracked_high/pos.avg_cost-1)*100:.1f}%)",
                    "action": "sell", "pnl_pct": pnl_pct,
                    "quantity": qty,
                })
        
        return alerts
    
    def execute_stop_orders(self) -> List[Order]:
        """执行所有止盈止损信号"""
        alerts = self.check_stop_conditions()
        orders = []
        for alert in alerts:
            tick = self.broker.get_tick(alert["code"])
            price = tick.ask1 if tick and tick.ask1 > 0 else 0
            if price <= 0:
                continue
            
            order = self.broker.place_order(
                code=alert["code"],
                side=OrderSide.SELL,
                order_type=OrderType.MARKET_BEST5,
                price=price,
                quantity=alert["quantity"],
                strategy_tag=f"stop_{alert['reason'][:10]}",
            )
            if order:
                orders.append(order)
                logger.warning(f"🛑 {alert['reason']}: {alert['code']} {alert['quantity']}股")
                log_trigger(
                    _classify(alert.get("reason", "")),
                    code=alert.get("code", ""),
                    name=alert.get("name", ""),
                    pnl_pct=round(float(alert.get("pnl_pct", 0.0) or 0.0), 2),
                    reason=alert.get("reason", ""),
                    quantity=int(alert.get("quantity", 0) or 0),
                )
        
        return orders
    
    # ========== AI信号下单 ==========
    
    def execute_ai_signal(
        self,
        code: str,
        name: str,
        ai_score: float,
        agreement: float,
        side: OrderSide,
        max_amount: Optional[float] = None,
        risk_exit: bool = False,
        explicit_quantity: Optional[int] = None,
    ) -> Tuple[bool, str, Optional[Order]]:
        """基于AI信号执行交易
        
        Args:
            code: 股票代码
            name: 股票名称
            ai_score: AI综合评分 (0-100)
            agreement: XGB/LGB一致性 (0-1)
            side: 买卖方向
            max_amount: 最大买入金额（元），None则自动计算
            risk_exit: 风控性离场（止盈止损/调仓清仓）。为 True 时卖出不受
                「AI评分仍高不卖」约束，让止损纪律优先于模型观点。
            explicit_quantity: 交易计划给定的目标股数。提供时按计划股数封顶，
                避免「部分止盈」被执行成整仓清仓（旧实现完全丢弃计划数量）。
                硬约束仍然生效：卖出 ≤ T+1 可卖量、买入 ≤ 仓位/现金上限。
        
        Returns:
            (success, message, order)
        """
        # 1. 日内风控检查
        if self._today_stats.get("stop_trading"):
            return False, f"⛔ 今日交易已停止: {self._today_stats['stop_reason']}", None
        
        # 2. 交易时段检查
        if self.config.trading_hours_only and not self.broker.is_trading_time():
            return False, "⏰ 非交易时段，不允许下单", None
        
        # 3. AI信号过滤
        if side == OrderSide.BUY and ai_score < self.config.min_ai_score_buy:
            return False, f"📉 AI评分不足: {ai_score:.0f} < {self.config.min_ai_score_buy}", None
        if side == OrderSide.SELL and not risk_exit and ai_score > self.config.max_ai_score_sell:
            return False, f"📈 AI评分仍高: {ai_score:.0f} > {self.config.max_ai_score_sell}，暂不建议卖出", None
        
        # 4. 模型一致性检查
        if self.config.require_agreement and agreement < 0.8:
            return False, f"🤝 模型分歧较大 (一致性={agreement:.2f})，跳过", None
        
        # 5. 下单间隔检查
        if self._last_order_time:
            elapsed = (datetime.now() - self._last_order_time).total_seconds()
            if elapsed < self.config.min_order_interval:
                time.sleep(self.config.min_order_interval - elapsed)
        
        # 6. 获取账户和行情
        account = self.broker.query_account()
        if not account:
            return False, "❌ 无法获取账户信息", None
        
        tick = self.broker.get_tick(code)
        if not tick or tick.last_price <= 0:
            return False, f"❌ 无法获取 {code} 行情", None
        
        # 6.5 实盘交易规则：停牌 / 涨跌停
        # 涨停时买单、跌停时卖单几乎不可能成交，停牌更是必然废单。
        # 提前拦掉，避免废单堆积占用资金与仓位额度。
        if self.config.enable_market_rules:
            _pre_close = float(getattr(tick, "pre_close", 0) or 0)
            _vol = getattr(tick, "volume", None)
            if side == OrderSide.BUY:
                _ok, _why = check_buyable(code, name, tick.last_price, _pre_close, _vol)
            else:
                _ok, _why = check_sellable(code, name, tick.last_price, _pre_close, _vol)
            if not _ok:
                logger.info(f"[交易规则] 拦截 {name or code} {side.value}: {_why}")
                return False, _why, None
        
        # 7. 计算下单数量和金额
        risk_result = self._check_position_risk(
            account, code, side, tick.last_price, max_amount
        )
        if not risk_result.passed:
            return False, risk_result.reason, None
        
        quantity = risk_result.adjusted_quantity
        # 计划股数封顶：不改硬约束（卖出仍 ≤ T+1 可卖、买入仍 ≤ 仓位/现金），
        # 只把「想下多少」从通用算法收敛到计划值。否则部分止盈会被执行成清仓。
        if explicit_quantity is not None and int(explicit_quantity) > 0:
            quantity = min(quantity, (int(explicit_quantity) // 100) * 100)
        if quantity < 100:
            return False, f"❌ 计算下单量不足1手: {quantity}股", None
        
        # 8. 价格偏离检查
        if side == OrderSide.BUY and tick.ask1 > 0:
            deviation = (tick.ask1 / tick.last_price - 1) * 100
            if abs(deviation) > self.config.price_deviation_pct:
                logger.warning(f"⚠️ 买一价偏离 {deviation:.1f}%，使用限价 {tick.ask1:.2f}")
            price = tick.ask1
        elif side == OrderSide.SELL and tick.bid1 > 0:
            price = tick.bid1
        else:
            price = tick.last_price
        
        # 9. 执行下单
        order = self.broker.place_order(
            code=code,
            side=side,
            order_type=OrderType.MARKET_BEST5,
            price=round(price, 2),
            quantity=quantity,
            strategy_tag=f"ai_{ai_score:.0f}",
        )
        
        if order:
            order.name = name
            order.ai_score = ai_score
            with self._lock:
                self._today_stats["trade_count"] += 1
                self._last_order_time = datetime.now()
            
            # 日内交易次数检查
            if self._today_stats["trade_count"] >= self.config.max_daily_trades:
                was_stopped = self._today_stats.get("stop_trading", False)
                self._today_stats["stop_trading"] = True
                self._today_stats["stop_reason"] = f"日交易次数达上限 {self.config.max_daily_trades}"
                logger.warning(f"⛔ {self._today_stats['stop_reason']}")
                if not was_stopped:
                    log_trigger("daily_trade_cap", count=self._today_stats["trade_count"])
            
            return True, f"✅ 已下单 {name} {side.value.upper()} {quantity}股 @ {price:.2f}", order
        
        return False, "❌ 下单失败", None
    
    def _check_position_risk(
        self,
        account: AccountInfo,
        code: str,
        side: OrderSide,
        price: float,
        max_amount: Optional[float] = None,
    ) -> RiskCheckResult:
        """仓位风控检查"""
        positions = {p.code: p for p in self.broker.query_positions()}
        
        if side == OrderSide.BUY:
            # 检查可用资金
            if account.available_cash < self.config.min_cash_reserve:
                return RiskCheckResult(False, f"💰 可用资金不足，保留 {self.config.min_cash_reserve:.0f}元")
            
            # 计算买入金额
            if max_amount:
                buy_amount = min(max_amount, account.available_cash - self.config.min_cash_reserve)
            else:
                # 默认：单票最大仓位限制
                max_single_amount = account.total_asset * self.config.max_position_pct
                # 如果已有持仓，计算可加仓金额
                if code in positions:
                    current_val = positions[code].market_value
                    buy_amount = min(
                        max_single_amount - current_val,
                        account.available_cash - self.config.min_cash_reserve,
                    )
                else:
                    buy_amount = min(
                        max_single_amount,
                        account.available_cash - self.config.min_cash_reserve,
                    )
            
            buy_amount = max(0, buy_amount)
            
            # 总仓位检查
            total_after_buy = account.market_value + buy_amount
            max_total = account.total_asset * self.config.max_total_position_pct
            if total_after_buy > max_total:
                buy_amount = max(0, max_total - account.market_value)
            
            if buy_amount <= 0:
                return RiskCheckResult(False, "💼 仓位已达上限或资金不足")
            
            # 单笔金额限制
            if buy_amount > self.config.max_order_amount:
                buy_amount = self.config.max_order_amount
            
            quantity = int(buy_amount / price / 100) * 100  # 整手
            return RiskCheckResult(True, "OK", quantity, {"amount": buy_amount})
        
        elif side == OrderSide.SELL:
            if code not in positions:
                return RiskCheckResult(False, f"📭 无 {code} 持仓")
            
            pos = positions[code]
            # T+1：不能只信券商返回的 available（部分接口未扣减当日买入），
            # 用规则层再兜一层：可卖 = min(available, quantity - today_quantity)
            can_sell = sellable_quantity(pos)
            if can_sell <= 0:
                _today = int(getattr(pos, "today_quantity", 0) or 0)
                return RiskCheckResult(
                    False,
                    f"🔒 {code} 无可卖股（T+1 锁仓：持仓 {pos.quantity} 股，"
                    f"其中今日买入 {_today} 股）"
                )
            
            return RiskCheckResult(True, "OK", can_sell, {
                "pnl_pct": pos.pnl_pct,
                "current_quantity": pos.quantity,
                "today_quantity": int(getattr(pos, "today_quantity", 0) or 0),
            })
        
        return RiskCheckResult(False, "未知买卖方向")
    
    # ========== 日内统计 ==========
    
    def _reset_daily_stats(self):
        """重置每日统计（建议每天9:25调用）"""
        self._today_stats = {
            "trade_count": 0,
            "total_pnl": 0.0,
            "start_asset": 0.0,
            "stop_trading": False,
            "stop_reason": "",
        }
        # 记录起始资产
        account = self.broker.query_account()
        if account:
            self._today_stats["start_asset"] = account.total_asset
    
    def update_daily_stats(self):
        """更新日内盈亏统计"""
        account = self.broker.query_account()
        if account and self._today_stats["start_asset"] > 0:
            self._today_stats["total_pnl"] = account.total_asset - self._today_stats["start_asset"]
            
            # 单日最大亏损检查
            pnl_pct = (self._today_stats["total_pnl"] / self._today_stats["start_asset"]) * 100
            if (self._today_stats["total_pnl"] < -self.config.max_daily_loss or
                pnl_pct < self.config.max_daily_loss_pct):
                was_stopped = self._today_stats.get("stop_trading", False)
                self._today_stats["stop_trading"] = True
                self._today_stats["stop_reason"] = (
                    f"单日亏损达 {self._today_stats['total_pnl']:.0f}元 ({pnl_pct:.2f}%)"
                )
                logger.warning(f"⛔ 日内风控触发: {self._today_stats['stop_reason']}")
                # 仅在「由可交易→熔断」的跳变瞬间记录一次，避免每次轮询重复写
                if not was_stopped:
                    log_trigger("daily_loss_circuit",
                                pnl=round(self._today_stats["total_pnl"], 2),
                                pnl_pct=round(pnl_pct, 2))
    
    def get_daily_summary(self) -> dict:
        """获取日内交易摘要"""
        self.update_daily_stats()
        return dict(self._today_stats)
    
    @property
    def trading_allowed(self) -> bool:
        """是否允许交易 = 未被风控熔断 且 处于可交易时段（含真实交易日历）"""
        if self._today_stats.get("stop_trading", False):
            return False
        if self.config.trading_hours_only:
            try:
                from src.trading.calendar import order_allowed
                ok, _why = order_allowed()
                if not ok:
                    return False
            except Exception:  # noqa: BLE001  日历不可用不阻断交易
                pass
        return True

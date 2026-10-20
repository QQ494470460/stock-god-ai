# -*- coding: utf-8 -*-
"""
自动交易编排器
AI信号 → 风控检查 → 下单执行 → 持仓监控 → 止损止盈
"""

import time
import threading
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from loguru import logger

from src.trading.broker import BrokerAPI, OrderSide
from src.trading.executor import TradeExecutor, RiskConfig
from src.trading.yuekai import YuekaiBroker
from src.trading.account import account_manager


class AutoTrader:
    """自动交易编排器
    
    编排整个交易流程：
    1. 定时拉取 AI 预测信号
    2. 风控过滤
    3. 生成买卖指令
    4. 执行下单
    5. 持仓止盈止损监控
    """
    
    def __init__(
        self,
        broker: Optional[BrokerAPI] = None,
        risk_config: Optional[RiskConfig] = None,
    ):
        self.broker = broker or YuekaiBroker()
        self.executor = TradeExecutor(self.broker, risk_config)
        self.config = self.executor.config
        
        # 运行状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # AI 预测回调（外部注入）
        self._ai_predict_fn: Optional[Callable] = None
        self._stop_monitor_fn: Optional[Callable] = None
        
        # 实时事件回调 — UI 可通过此回调获取扫描日志
        self.on_event: Optional[Callable] = None  # Callable[[str, dict], None]
        
        # 纸交易账户引用（独立 paper account）
        self.paper_account = None  # 外部注入 PaperAccount 实例

        # 安全与风控中枢 RiskSupervisor（可选）
        # 接入后：所有下单走安全中枢（状态机+Kill Switch+确认闸门+回撤熔断+高级仓位）。
        # 不接入时完全保留原有行为，做到"默认零风险变更"。
        self.guard = None
        
        # 统计
        self.stats = {
            "start_time": None,
            "signals_received": 0,
            "orders_placed": 0,
            "orders_filled": 0,
            "stops_triggered": 0,
        }
        
        logger.info("🤖 AutoTrader 已初始化")
    
    def _emit_event(self, event_type: str, data: dict):
        """发送事件到 UI（通过 on_event 回调）"""
        if self.on_event:
            try:
                self.on_event(event_type, data)
            except Exception:
                pass
    
    # ========== AI 信号注入 ==========
    
    def set_ai_predictor(self, predict_fn: Callable[[str], Optional[dict]]):
        """设置 AI 预测函数
        
        predict_fn(code: str) → {"score": float, "agreement": float, "signal": int,
                                 "name": str, "xgb_signal": int, ...} | None
        
        注：全项目已统一为单一 XGBoost 模型，agreement 恒为 1.0，
        不再表示两模型之间的一致性。
        """
        self._ai_predict_fn = predict_fn
    
    def use_builtin_predictor(self, watchlist_codes: List[str] = None):
        """使用内置 predictor 作为信号源
        
        watchlist_codes: 股票代码列表，None则自动从 watchlist.json 加载
        """
        try:
            from src.data.fetcher import fetcher
            from src.models.predictor import predictor as ai_predictor
            
            # 自动加载 watchlist（修复空列表bug）
            if not watchlist_codes:
                try:
                    import json as _json
                    config_file = Path(__file__).parent.parent.parent / "config" / "watchlist.json"
                    if config_file.exists():
                        wl_cfg = _json.loads(config_file.read_text(encoding="utf-8"))
                        watchlist_codes = wl_cfg.get("watchlist", [])
                        logger.info(f"📋 从 watchlist.json 加载 {len(watchlist_codes)} 只自选股")
                except Exception:
                    pass
            
            # 缓存日K线，避免同一股票扫描多次拉取
            _kline_cache = {}
            
            def predict_stock(code: str) -> Optional[dict]:
                try:
                    if code not in _kline_cache:
                        df = fetcher.get_daily_kline(code)
                        if len(df) < 60:
                            return None
                        _kline_cache[code] = df
                    else:
                        df = _kline_cache[code]
                    
                    p = ai_predictor.predict_single(df, code)
                    return {
                        "code": code,
                        "name": p.name,
                        "score": p.score,
                        "agreement": p.agreement,
                        "xgb_signal": p.xgb_signal,
                        "prob_up": p.prob_up,
                        "prob_down": p.prob_down,
                    }
                except Exception as e:
                    logger.warning(f"AI预测 {code} 失败: {e}")
                    return None
            
            self._ai_predict_fn = predict_stock
            logger.info(f"✅ 已接入内置AI预测器，监控 {len(watchlist_codes or [])} 只股票")
            
        except Exception as e:
            logger.error(f"接入AI预测器失败: {e}")
    
    # ========== 安全中枢接入（可选 hook，默认不接，保留原行为） ==========

    def set_guard(self, guard) -> None:
        """接入安全与风控中枢 RiskSupervisor。

        接入后，scan_and_trade 内所有买卖执行都改走安全中枢（状态机 + Kill Switch
        + 确认闸门 + 组合级回撤熔断 + 高级仓位），由 guard 统一裁决是否下单。

        设计铁律：未调用本方法（self.guard 为 None）时，行为与接入前完全一致。
        """
        self.guard = guard
        # 共享同一纸账户引用，避免安全中枢与 AutoTrader 各自记账导致重复/错位
        if self.paper_account is not None and getattr(guard, "paper_account", None) is None:
            guard.paper_account = self.paper_account
        if getattr(guard, "paper_account", None) is not None:
            self.paper_account = guard.paper_account
        logger.warning("🛡️ 已接入 RiskSupervisor 安全中枢：后续下单将经过安全裁决")

    def _execute_signal(self, code, name, ai_score, agreement, side, strategy_tag="ai"):
        """执行一笔 AI 信号交易的统一入口。

        已接入 guard → 走安全中枢（propose_trade → approve_and_execute），
        由 guard 依据当前模式(PAPER/LIVE)与熔断状态决定是否真正下单；
        未接入 guard → 走原 TradeExecutor，完全等价于接入前。
        """
        if self.guard is not None:
            prop = self.guard.propose_trade(
                code=code, name=name, side=side, ai_score=ai_score,
                agreement=agreement, strategy_tag=strategy_tag,
            )
            if prop.status == "blocked":
                self._emit_event("risk_block", {
                    "code": code, "name": name, "side": side.value,
                    "reason": prop.reason or "被安全中枢拦截",
                })
                return False, prop.reason or "被安全中枢拦截", None
            ok, msg, order = self.guard.approve_and_execute(
                prop.proposal_id, operator="auto(scan)")
            return ok, msg, order
        return self.executor.execute_ai_signal(
            code=code, name=name, ai_score=ai_score,
            agreement=agreement, side=side,
        )

    # ========== 核心编排 ==========
    
    def scan_and_trade(self) -> List[dict]:
        """单次扫描：AI预测 → 风控 → 下单（券商/纸交易） → 返回结果"""
        if not self._ai_predict_fn:
            self._emit_event("warn", {"msg": "AI预测函数未设置，无法扫描"})
            return []
        
        results = []
        scan_time = datetime.now().strftime("%H:%M:%S")
        
        # 1. 更新日内统计
        self.executor.update_daily_stats()
        if not self.executor.trading_allowed:
            self._emit_event("blocked", {"msg": "日内风控已触发，停止交易", "time": scan_time})
            return results

        # 1.5 安全中枢：周期性回撤 / 单日亏损监控 + Kill Switch 强制熔断
        if self.guard is not None:
            alerts = self.guard.tick_monitor()
            for a in alerts:
                self._emit_event("risk_alert", {"msg": a, "time": scan_time})
            if self.guard.safety.halted:
                self._emit_event("blocked", {
                    "msg": "🛑 安全中枢已熔断(KILL SWITCH)，停止一切交易", "time": scan_time})
                return results
        
        # 2. 止盈止损检查（优先）
        self._emit_event("scan_start", {"time": datetime.now().strftime("%H:%M:%S")})
        
        stop_orders = self.executor.execute_stop_orders()
        if stop_orders:
            self.stats["stops_triggered"] += len(stop_orders)
            for o in stop_orders:
                results.append({"type": "stop", "order": o})
                self._emit_event("stop_triggered", {
                    "code": o.code, "name": getattr(o, "name", ""),
                    "side": "sell", "quantity": o.quantity, "reason": "止盈止损"
                })
        
        # 2.5 纸交易：按最新行情更新持仓市值
        if self.paper_account:
            prices = {}
            for code in self.paper_account.holdings:
                price = 0.0
                # 1) 券商实时行情
                try:
                    tick = self.broker.get_tick(code)
                    if tick and getattr(tick, "last_price", 0) > 0:
                        price = float(tick.last_price)
                except Exception:
                    pass
                # 2) 公开实时行情兜底 —— 与 loop._price_of 保持同源。
                #    缺这一步的话：券商 tick 不可用时（模拟盘实测恒不可用）
                #    这里会**完全跳过 mark**，纸账户 last_price 永远停在成交价，
                #    看板（另一进程读 JSON）看到的就是过期估值。
                if price <= 0:
                    try:
                        from src.data.fetcher import fetcher
                        q = fetcher.get_realtime_quote(code)
                        if isinstance(q, dict):
                            price = float(q.get("最新价") or 0)
                    except Exception:
                        pass
                if price > 0:
                    prices[code] = price
            if prices:
                self.paper_account.mark_to_market(prices)
                # 变更后必须落盘，否则跨进程永远读到旧价（loop 已修，此处对齐）
                if hasattr(self.paper_account, "save"):
                    try:
                        self.paper_account.save()
                    except Exception:  # noqa: BLE001
                        pass
        
        # 3. 持仓股票：AI信号检查是否需要卖出
        positions = self.broker.query_positions()
        self._emit_event("check_positions", {"count": len(positions), "time": scan_time})
        
        sell_count = 0
        sell_candidates = []
        for pos in positions:
            if pos.available <= 0:
                continue
            
            signal = self._ai_predict_fn(pos.code)
            if not signal:
                continue
            
            self.stats["signals_received"] += 1
            
            # AI评分低 + XGBoost 看跌 → 卖出信号
            if (signal["score"] <= self.config.max_ai_score_sell and
                signal["xgb_signal"] == -1):
                sell_candidates.append((pos, signal))
        
        for pos, signal in sell_candidates:
            success, msg, order = self._execute_signal(
                code=pos.code,
                name=pos.name,
                ai_score=signal["score"],
                agreement=signal["agreement"],
                side=OrderSide.SELL,
            )
            if success and order:
                self.stats["orders_placed"] += 1
                sell_count += 1
                # guard 接入时由安全中枢统一记账，避免重复
                if self.paper_account and not self.guard:
                    self.paper_account.sell(
                        code=pos.code, price=order.price,
                        quantity=order.quantity,
                        strategy="AI卖出信号", ai_score=signal["score"],
                    )
                self._emit_event("trade", {
                    "time": scan_time, "side": "sell",
                    "code": pos.code, "name": signal.get("name", pos.code),
                    "price": order.price, "quantity": order.quantity,
                    "ai_score": signal["score"], "order_id": order.order_id,
                })
            results.append({"type": "sell_signal", "code": pos.code, "success": success, "msg": msg})
        
        if sell_count:
            self._emit_event("info", {"msg": f"卖出信号: {sell_count} 只"})
        
        # 4. 获取自选股AI信号，生成买入候选
        candidates = self._scan_buy_candidates()
        self._emit_event("buy_candidates", {"count": len(candidates), "time": scan_time})
        
        buy_count = 0
        for signal in candidates:
            success, msg, order = self._execute_signal(
                code=signal["code"],
                name=signal.get("name", signal["code"]),
                ai_score=signal["score"],
                agreement=signal["agreement"],
                side=OrderSide.BUY,
            )
            if success and order:
                self.stats["orders_placed"] += 1
                buy_count += 1
                # guard 接入时由安全中枢统一记账，避免重复
                if self.paper_account and not self.guard:
                    self.paper_account.buy(
                        code=signal["code"], name=signal.get("name", signal["code"]),
                        price=order.price, quantity=order.quantity,
                        strategy="AI买入信号", ai_score=signal["score"],
                    )
                self._emit_event("trade", {
                    "time": scan_time, "side": "buy",
                    "code": signal["code"], "name": signal.get("name", signal["code"]),
                    "price": order.price, "quantity": order.quantity,
                    "ai_score": signal["score"], "order_id": order.order_id,
                })
            results.append({"type": "buy_signal", "code": signal["code"], "success": success, "msg": msg})
        
        if buy_count:
            self._emit_event("info", {"msg": f"买入信号: {buy_count} 只"})
        
        self._emit_event("scan_done", {
            "time": datetime.now().strftime("%H:%M:%S"),
            "signals": self.stats["signals_received"],
            "orders": self.stats["orders_placed"],
            "stops": self.stats["stops_triggered"],
            "sell_count": sell_count,
            "buy_count": buy_count,
        })
        
        return results
    
    def _scan_buy_candidates(self) -> List[dict]:
        """扫描买入候选（需外部提供股票列表或从config获取）"""
        if not self._ai_predict_fn:
            return []
        
        # 尝试从 watchlist 读自选股（与 watchlist_monitor 共用配置）
        try:
            import json
            from pathlib import Path
            config_file = Path(__file__).parent.parent.parent / "config" / "watchlist.json"
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                stocks = cfg.get("stocks") or cfg.get("watchlist", [])
            else:
                return []
        except Exception:
            return []
        
        candidates = []
        for s in stocks:
            # 兼容两种格式：{"code":"000001"} 或直接 "000001"
            code = s if isinstance(s, str) else s.get("code", "")
            if not code:
                continue
            
            signal = self._ai_predict_fn(code)
            if not signal:
                continue
            
            self.stats["signals_received"] += 1
            
            # 买入条件：AI评分 >= min_ai_score_buy + XGBoost 看涨
            if (signal["score"] >= self.config.min_ai_score_buy and
                signal["xgb_signal"] == 1):
                
                # 检查是否已有持仓
                positions = {p.code: p for p in self.broker.query_positions()}
                if code in positions:
                    if positions[code].pnl_pct < -5:
                        candidates.append({**signal, "remark": "加仓（浮亏补仓）"})
                    else:
                        continue  # 已有盈利持仓，不再加仓
                else:
                    candidates.append(signal)
        
        # 按评分排序，优先买高分
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:5]  # 单次最多5只买入
    
    # ========== 运行模式 ==========
    
    def start(
        self,
        interval_seconds: int = 60,
        trading_hours_only: bool = True,
        on_callback: Optional[Callable] = None,
    ):
        """启动自动交易（后台线程）
        
        Args:
            interval_seconds: 扫描间隔（秒），建议 60-300
            trading_hours_only: 仅交易时段运行
            on_callback: 每次扫描后回调，传入 results list
        """
        if self._running:
            logger.warning("AutoTrader 已在运行中")
            return
        
        self._running = True
        self._stop_event.clear()
        self.stats["start_time"] = datetime.now()
        
        def _loop():
            logger.info(f"🚀 自动交易启动 (间隔 {interval_seconds}s, {'仅交易时段' if trading_hours_only else '24h运行'})")
            
            while not self._stop_event.is_set():
                try:
                    if trading_hours_only and not self.broker.is_trading_time():
                        # 非交易时段每5分钟检查一次
                        self._stop_event.wait(300)
                        continue
                    
                    results = self.scan_and_trade()
                    
                    if on_callback and results:
                        on_callback(results)
                    
                except Exception as e:
                    logger.error(f"AutoTrader 扫描异常: {e}")
                
                self._stop_event.wait(interval_seconds)
            
            logger.info("🛑 自动交易已停止")
        
        self._thread = threading.Thread(target=_loop, daemon=True, name="AutoTrader")
        self._thread.start()
    
    def start_stop_monitor(self, interval_seconds: int = 10):
        """启动止盈止损独立监控（高频检查，仅止盈止损不下主信号）
        
        每10秒检查一次持仓，触发止损/止盈立即下单
        """
        if self._stop_monitor_fn:
            return
        
        self._stop_event.clear()
        
        def _stop_loop():
            logger.info(f"🛡️ 止盈止损监控启动 (间隔 {interval_seconds}s)")
            while not self._stop_event.is_set():
                try:
                    if self.broker.is_trading_time():
                        stop_orders = self.executor.execute_stop_orders()
                        if stop_orders:
                            self.stats["stops_triggered"] += len(stop_orders)
                    else:
                        self._stop_event.wait(300)
                except Exception as e:
                    logger.error(f"止盈止损监控异常: {e}")
                self._stop_event.wait(interval_seconds)
        
        self._stop_monitor_fn = _stop_loop
        threading.Thread(target=_stop_loop, daemon=True, name="StopMonitor").start()
    
    def stop(self):
        """停止自动交易"""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        
        if self.broker.connected:
            self.broker.disconnect()
        
        logger.info("AutoTrader 已完全停止")
    
    def run_once(self) -> List[dict]:
        """手动触发一次扫描（不启动后台线程）"""
        return self.scan_and_trade()
    
    # ========== 状态查询 ==========
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def get_status(self) -> dict:
        """获取运行状态摘要"""
        account = self.broker.query_account()
        positions = self.broker.query_positions()
        
        return {
            "running": self._running,
            "connected": self.broker.connected,
            "broker": self.broker.get_broker_name(),
            "simulated": getattr(self.broker, "is_simulated", True),
            "trading_time": self.broker.is_trading_time(),
            "trading_allowed": self.executor.trading_allowed,
            "account": {
                "total_asset": account.total_asset if account else 0,
                "available_cash": account.available_cash if account else 0,
                "market_value": account.market_value if account else 0,
            } if account else None,
            "positions": len(positions),
            "stats": dict(self.stats),
            "daily": self.executor.get_daily_summary(),
            "guard": self.guard.get_status() if self.guard else None,
        }
    
    def print_status(self):
        """打印状态面板"""
        s = self.get_status()
        print("\n" + "=" * 60)
        print(f"  🤖 选股神AI · 自动交易 | {s['broker']}")
        print(f"  {'🟢 运行中' if s['running'] else '🔴 已停止'} | "
              f"{'🔗 已连接' if s['connected'] else '⛓️‍💥 未连接'} | "
              f"{'🧪 模拟' if s['simulated'] else '💹 实盘'}")
        print("-" * 60)
        
        if s["account"]:
            print(f"  总资产: ¥{s['account']['total_asset']:,.2f}")
            print(f"  可用资金: ¥{s['account']['available_cash']:,.2f}")
            print(f"  持仓市值: ¥{s['account']['market_value']:,.2f}")
        
        print(f"  持仓数: {s['positions']} | 交易时段: {'✅' if s['trading_time'] else '❌'} | 允许交易: {'✅' if s['trading_allowed'] else '⛔'}")
        print(f"  信号: {s['stats'].get('signals_received', 0)} | "
              f"下单: {s['stats'].get('orders_placed', 0)} | "
              f"止盈止损: {s['stats'].get('stops_triggered', 0)}")

        g = s.get("guard")
        if g:
            mode = g.get("mode", "?")
            flag = "🛑 已熔断" if g.get("halted") else "🟢 正常"
            conf = "需逐单确认" if g.get("requires_confirmation") else "自动执行"
            print(f"  🛡️ 安全中枢: mode={mode} | {flag} | {conf}")
        
        daily = s.get("daily", {})
        if daily.get("start_asset", 0) > 0:
            print(f"  今日盈亏: ¥{daily['total_pnl']:,.2f}")
        
        print("=" * 60 + "\n")

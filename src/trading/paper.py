# -*- coding: utf-8 -*-
"""
纸交易引擎 (Paper Trading)
独立的模拟账户 + 虚拟持仓 + 盈亏计算
不依赖券商连接，零风险验证策略
"""

import json
import threading
from pathlib import Path
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from src.trading.broker import (
    BrokerAPI, Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo, TickData, BrokerCallback,
)
from src.trading.rules import apply_slippage, check_t1_sell, sellable_quantity

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PAPER_ACCOUNT_FILE = DATA_DIR / "paper_account.json"
PAPER_HISTORY_FILE = DATA_DIR / "paper_trade_history.json"


@dataclass
class TradeRecord:
    """单笔交易记录"""
    trade_id: str
    code: str
    name: str
    side: str              # "buy" | "sell"
    price: float
    quantity: int
    amount: float          # 成交金额
    commission: float = 0.0
    stamp_tax: float = 0.0
    filled_at: str = ""
    strategy: str = ""     # 策略标签
    ai_score: float = 0.0
    pnl: float = 0.0       # 卖出时才填
    pnl_pct: float = 0.0
    close_id: str = ""     # 平仓对应买入的 trade_id


class PaperAccount:
    """虚拟账户 — 资金 + 持仓"""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        account_file: Optional[Path] = None,
        history_file: Optional[Path] = None,
    ):
        """纸账户。

        Args:
            initial_cash: 初始本金（仅在新账户时生效；已有磁盘状态以磁盘为准）
            account_file: 账户状态文件路径。**测试必须传 tmp 路径**，否则会直写
                生产的 ``data/paper_account.json``（曾经踩过：单元测试把真实
                纸账户的现金/持仓/当日盈亏覆盖掉了）。
            history_file: 成交历史文件路径，同上。
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.frozen_cash = 0.0

        self.account_file: Path = Path(account_file) if account_file else PAPER_ACCOUNT_FILE
        self.history_file: Path = Path(history_file) if history_file else PAPER_HISTORY_FILE
        
        # 持仓: code → {quantity, avg_cost, market_value, ...}
        self.holdings: Dict[str, dict] = {}
        
        # 交易记录
        self.trade_history: List[TradeRecord] = []
        self._history_dirty = False
        
        # 当日统计
        self.today_pnl = 0.0
        self.today_trades = 0
        self._last_reset_date = datetime.now().strftime("%Y-%m-%d")
        
        # 线程安全
        self._lock = threading.Lock()
        
        self._load()
    
    # ========== 持久化 ==========

    def _normalize_holdings(self) -> None:
        """让 ``market_value`` 恒等于 ``quantity × last_price``。

        历史坑：买入时 ``market_value`` 按**成交价**写死，之后 ``last_price``
        被 mark_to_market / 手工修正更新，但 ``market_value`` 没跟着走。实测
        `data/paper_account.json` 出现过 quantity=300、last_price=1501.95、
        market_value=450000（应为 450585）—— 总资产、今日盈亏、调仓的目标市值
        三者同时偏低，而且看不出来。

        统一在载入和落盘两个口子上归一，任何来源的漂移都会被拉正。
        """
        for h in self.holdings.values():
            try:
                qty = int(h.get("quantity", 0) or 0)
            except Exception:  # noqa: BLE001
                qty = 0
            if qty <= 0:
                continue
            try:
                px = float(h.get("last_price") or 0)
            except Exception:  # noqa: BLE001
                px = 0.0
            if px <= 0:
                # 没有最新价就退化成成本价，至少不留下 0 市值
                try:
                    px = float(h.get("avg_cost") or 0)
                except Exception:  # noqa: BLE001
                    px = 0.0
                if px > 0:
                    h["last_price"] = px
            if px > 0:
                h["market_value"] = round(qty * px, 2)

    def _load(self):
        """从磁盘恢复状态"""
        try:
            if self.account_file.exists():
                data = json.loads(self.account_file.read_text(encoding="utf-8"))
                self.cash = data.get("cash", self.initial_cash)
                self.holdings = data.get("holdings", {})
                # 先把 market_value 拉回 quantity×last_price，再做下面的权益对账，
                # 否则陈旧市值会连带把本金基线也算偏。
                self._normalize_holdings()
                # 兼容旧存档：补齐 T+1 相关字段，否则卖出时会 KeyError
                for _h in self.holdings.values():
                    _h.setdefault("today_quantity", 0)
                    _h.setdefault("available", int(_h.get("quantity", 0) or 0))
                self.today_pnl = data.get("today_pnl", 0.0)
                self.today_trades = data.get("today_trades", 0)
                self._last_reset_date = data.get("last_reset_date", self._last_reset_date)
                # ★ 本金基线必须与「实际注入的权益」对齐，否则盈亏/收益率/回撤会被按
                # 默认的 ¥100,000 误算（出现 4890% 这类幽灵收益 —— 即「双账户态不一致」）。
                # 历史存档若未记录 initial_cash（账户被外部注入了资金/持仓），则以载入时的
                # 总资产作为 0 基准，避免把注入资本算成盈利。
                persisted = data.get("initial_cash", None)
                if persisted is not None:
                    self.initial_cash = float(persisted)
                else:
                    loaded_equity = self.cash + sum(
                        float(h.get("market_value", 0.0)) for h in self.holdings.values()
                    )
                    self.initial_cash = loaded_equity
                logger.info(f"📂 纸交易账户已恢复: 现金 ¥{self.cash:,.0f}, {len(self.holdings)} 只持仓"
                            f" · 本金基线 ¥{self.initial_cash:,.0f}")
            else:
                logger.info(f"💰 新建纸交易账户: 初始资金 ¥{self.initial_cash:,.0f}")
        except Exception as e:
            logger.warning(f"加载纸账户失败: {e}")
        
        try:
            if self.history_file.exists():
                raw = json.loads(self.history_file.read_text(encoding="utf-8"))
                self.trade_history = [TradeRecord(**r) for r in raw]
                logger.info(f"📋 已加载 {len(self.trade_history)} 条交易记录")
        except Exception as e:
            logger.warning(f"加载交易记录失败: {e}")
    
    def _save(self):
        """持久化（后台异步写）"""
        try:
            self._normalize_holdings()   # 落盘前再拉正一次，防止外部改 last_price
            data = {
                "cash": self.cash,
                "initial_cash": self.initial_cash,
                "holdings": self.holdings,
                "today_pnl": self.today_pnl,
                "today_trades": self.today_trades,
                "last_reset_date": self._last_reset_date,
            }
            self.account_file.parent.mkdir(parents=True, exist_ok=True)
            self.account_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    
    def _save_history(self):
        try:
            records = [r.__dict__ for r in self.trade_history[-500:]]  # 保留最近500条
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            self._history_dirty = False
        except Exception:
            pass
    
    # ========== 每日重置 ==========
    
    def _check_daily_reset(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            self.today_pnl = 0.0
            self.today_trades = 0
            # T+1 解锁：昨日（及更早）买入的股票，今日起全部转为可卖。
            # 没有这一步的话，持仓会被永久锁死（因为买入时 available 不增加）。
            for _h in self.holdings.values():
                _h["today_quantity"] = 0
                _h["available"] = int(_h.get("quantity", 0) or 0)
            self._last_reset_date = today
            self._save()
    
    # ========== 交易操作 ==========
    
    def buy(self, code: str, name: str, price: float, quantity: int,
            strategy: str = "", ai_score: float = 0.0) -> Tuple[bool, str, Optional[TradeRecord]]:
        """买入（虚拟成交）"""
        with self._lock:
            self._check_daily_reset()
            
            # 最小1手
            quantity = (quantity // 100) * 100
            if quantity < 100:
                return False, "不足1手(100股)", None
            
            # 滑点/冲击成本：真实成交价高于盘口价（买入更贵）
            raw_amount = price * quantity
            fill_price = apply_slippage(price, "buy", raw_amount)

            # 计算费用 — A股费率：佣金万2.5(最低5元)，印花税仅卖出收(千1)
            amount = fill_price * quantity
            commission = max(5.0, amount * 0.00025)
            
            total_cost = amount + commission
            if self.cash - self.frozen_cash < total_cost:
                return False, f"资金不足 (需¥{total_cost:,.0f}，可用¥{self.cash - self.frozen_cash:,.0f})", None
            
            # 扣款
            self.cash -= total_cost
            self.today_trades += 1
            
            # 更新持仓
            if code in self.holdings:
                h = self.holdings[code]
                old_qty = h["quantity"]
                old_cost = h["avg_cost"]
                new_qty = old_qty + quantity
                new_cost = (old_cost * old_qty + amount) / new_qty if new_qty > 0 else 0
                h["quantity"] = new_qty
                # T+1：今日买入的部分当日不可卖 —— available 保持不变，
                # 只累加 today_quantity（次日 _check_daily_reset 时解锁）
                h["today_quantity"] = int(h.get("today_quantity", 0) or 0) + quantity
                h["avg_cost"] = new_cost
                h["market_value"] = new_qty * fill_price
            else:
                self.holdings[code] = {
                    "name": name,
                    "quantity": quantity,
                    "available": 0,            # T+1：当日买入当日不可卖
                    "today_quantity": quantity,
                    "avg_cost": fill_price,
                    "market_value": amount,
                    "last_price": fill_price,
                }
            
            # 记录
            tid = f"PB{datetime.now().strftime('%Y%m%d%H%M%S')}{self.today_trades:03d}"
            record = TradeRecord(
                trade_id=tid,
                code=code, name=name,
                side="buy", price=fill_price,
                quantity=quantity, amount=amount,
                commission=commission,
                filled_at=datetime.now().isoformat(),
                strategy=strategy, ai_score=ai_score,
            )
            self.trade_history.append(record)
            self._history_dirty = True
            
            self._save()
            _slip = (fill_price - price) * quantity
            logger.info(f"📈 纸交易买入: {name}({code}) {quantity}股 @¥{fill_price:.2f} "
                       f"(盘口¥{price:.2f} 滑点¥{_slip:.2f}) 费用¥{commission:.2f} "
                       f"余额¥{self.cash:,.0f}")
            
            return True, f"✅ 买入 {quantity}股 @¥{fill_price:.2f}（滑点¥{_slip:.2f}，T+1 锁定）", record
    
    def sell(self, code: str, price: float, quantity: int,
             strategy: str = "", ai_score: float = 0.0) -> Tuple[bool, str, Optional[TradeRecord]]:
        """卖出"""
        with self._lock:
            self._check_daily_reset()
            
            quantity = (quantity // 100) * 100
            if quantity < 100:
                return False, "不足1手", None
            
            if code not in self.holdings:
                return False, f"无 {code} 持仓", None
            
            h = self.holdings[code]

            # T+1：可卖数量须扣除今日买入部分（不能只看 available，
            # 旧数据或券商返回值可能未扣减，这里统一用规则层兜住）
            _pos = dict(h)
            _pos.setdefault("code", code)
            _pos.setdefault("name", h.get("name", code))
            _pos.setdefault("today_quantity", 0)
            ok, why = check_t1_sell(_pos, quantity)
            if not ok:
                return False, why, None

            # 滑点/冲击成本：真实成交价低于盘口价（卖出更便宜）
            raw_amount = price * quantity
            fill_price = apply_slippage(price, "sell", raw_amount)

            amount = fill_price * quantity
            commission = max(5.0, amount * 0.00025)
            stamp_tax = amount * 0.001  # 印花税（卖出收）
            
            net_amount = amount - commission - stamp_tax
            
            # 计算盈亏
            cost_basis = h["avg_cost"] * quantity
            pnl = amount - cost_basis - commission - stamp_tax
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            # 更新持仓
            h["quantity"] -= quantity
            h["available"] -= quantity
            if h["quantity"] <= 0:
                del self.holdings[code]
            else:
                h["market_value"] = h["quantity"] * price
            
            # 回笼资金
            self.cash += net_amount
            self.today_pnl += pnl
            self.today_trades += 1
            
            # 记录
            tid = f"PS{datetime.now().strftime('%Y%m%d%H%M%S')}{self.today_trades:03d}"
            record = TradeRecord(
                trade_id=tid,
                code=code, name=h.get("name", code),
                side="sell", price=fill_price,
                quantity=quantity, amount=amount,
                commission=commission, stamp_tax=stamp_tax,
                filled_at=datetime.now().isoformat(),
                strategy=strategy, ai_score=ai_score,
                pnl=pnl, pnl_pct=pnl_pct,
            )
            self.trade_history.append(record)
            self._history_dirty = True
            
            self._save()
            _slip = (price - fill_price) * quantity
            logger.info(f"📉 纸交易卖出: {h.get('name', code)}({code}) {quantity}股 @¥{fill_price:.2f} "
                       f"(盘口¥{price:.2f} 滑点¥{_slip:.2f}) "
                       f"盈亏¥{pnl:+,.2f}({pnl_pct:+.2f}%) 余额¥{self.cash:,.0f}")
            
            return True, f"✅ 卖出 {quantity}股 @¥{fill_price:.2f} 盈亏¥{pnl:+,.2f}", record
    
    # ========== 市值更新 ==========
    
    def mark_to_market(self, prices: Dict[str, float]):
        """按最新市价更新持仓市值"""
        with self._lock:
            for code, price in prices.items():
                if code in self.holdings:
                    self.holdings[code]["last_price"] = price
                    self.holdings[code]["market_value"] = self.holdings[code]["quantity"] * price
    
    # ========== 查询 ==========
    
    def get_total_asset(self) -> float:
        """总资产 = 现金 + 持仓市值"""
        mv = sum(h.get("market_value", 0) for h in self.holdings.values())
        return self.cash + mv
    
    def get_total_pnl(self) -> float:
        """累计盈亏"""
        return self.get_total_asset() - self.initial_cash
    
    def get_total_pnl_pct(self) -> float:
        """累计收益率"""
        return (self.get_total_asset() / self.initial_cash - 1) * 100 if self.initial_cash > 0 else 0
    
    def get_positions(self) -> List[dict]:
        """当前持仓列表"""
        return [
            {
                "code": code,
                "name": h.get("name", code),
                "quantity": h["quantity"],
                "available": h["available"],
                # T+1：今日买入量与实际可卖量（UI 要显示"可卖 X 股"而不是总持仓）
                "today_quantity": int(h.get("today_quantity", 0) or 0),
                "sellable": sellable_quantity({
                    "quantity": h["quantity"],
                    "available": h.get("available", h["quantity"]),
                    "today_quantity": h.get("today_quantity", 0),
                }),
                "avg_cost": round(h["avg_cost"], 3),
                "last_price": round(h.get("last_price", h["avg_cost"]), 3),
                "market_value": round(h.get("market_value", 0), 2),
                "pnl": round((h.get("last_price", h["avg_cost"]) - h["avg_cost"]) * h["quantity"], 2),
                "pnl_pct": round((h.get("last_price", h["avg_cost"]) / h["avg_cost"] - 1) * 100, 3) if h["avg_cost"] > 0 else 0,
            }
            for code, h in self.holdings.items()
        ]
    
    def get_recent_trades(self, n: int = 50) -> List[TradeRecord]:
        return self.trade_history[-n:]
    
    def get_performance_stats(self) -> dict:
        """策略绩效统计"""
        sells = [t for t in self.trade_history if t.side == "sell" and t.pnl != 0]
        buys = [t for t in self.trade_history if t.side == "buy"]
        
        wins = [t for t in sells if t.pnl > 0]
        losses = [t for t in sells if t.pnl < 0]
        
        total_trades = len(sells)
        win_count = len(wins)
        lose_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        avg_win = sum(t.pnl for t in wins) / win_count if win_count > 0 else 0
        avg_loss = sum(t.pnl for t in losses) / lose_count if lose_count > 0 else 0
        
        # 利润因子
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        # 回撤计算（简化：从交易序列推）
        equity = self.initial_cash
        peak = equity
        max_dd = 0.0
        for t in self.trade_history:
            if t.side == "buy":
                equity -= t.amount + t.commission
            else:
                equity += t.amount - t.commission - t.stamp_tax
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        # 夏普近似
        if len(sells) >= 5:
            returns = [t.pnl_pct / 100 for t in sells]
            import numpy as np
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(len(sells))
        else:
            sharpe = 0.0
        
        return {
            "total_trades": total_trades,
            "wins": win_count,
            "losses": lose_count,
            "win_rate": round(win_rate, 3),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999,
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe": round(sharpe, 3),
            "total_pnl": round(self.get_total_pnl(), 2),
            "total_return_pct": round(self.get_total_pnl_pct(), 2),
            "total_asset": round(self.get_total_asset(), 2),
            "cash": round(self.cash, 2),
        }
    
    def reset(self, new_cash: float = None):
        """重置账户"""
        with self._lock:
            self.cash = new_cash or self.initial_cash
            self.holdings = {}
            self.trade_history = []
            self.today_pnl = 0.0
            self.today_trades = 0
            self._save()
            self._save_history()
            logger.info(f"🔄 纸交易账户已重置: ¥{self.cash:,.0f}")


# 全局单例
paper_account = PaperAccount()

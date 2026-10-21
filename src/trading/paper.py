# -*- coding: utf-8 -*-
"""
纸交易引擎 (Paper Trading)
独立的模拟账户 + 虚拟持仓 + 盈亏计算
不依赖券商连接，零风险验证策略
"""

import json
import os
import shutil
import threading
import time as _time
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    import msvcrt  # Windows 跨进程文件锁（看板与常驻闭环是两个独立进程，需跨进程串行落盘）
except Exception:  # noqa: BLE001
    msvcrt = None

from src.trading.broker import (
    BrokerAPI, Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo, TickData, BrokerCallback,
)
from src.trading.rules import apply_slippage, check_t1_sell, sellable_quantity

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PAPER_ACCOUNT_FILE = DATA_DIR / "paper_account.json"
PAPER_HISTORY_FILE = DATA_DIR / "paper_trade_history.json"


def _atomic_write_text(path, text):
    """原子写：先写 <name>.tmp.<pid> 再 os.replace（Windows 上为原子重命名）。

    为什么必须原子：纸账户文件由**两个独立进程**（常驻 run_loop 守护 + Streamlit
    看板）轮流向同一份 paper_account.json / paper_trade_history.json 落盘，且我方
    经常用 ``taskkill /F`` 重启守护 —— 若正落在 ``write_text`` 中途被杀，文件会被
    截断成半截，下次加载 JSON 解析失败 → 纸账户被「重置」成空账户、历史全丢（这是
    真实踩过的灾难）。os.replace 要么整体成功要么不动原文件，杜绝半截文件。

    落盘前把上一份**完整**文件留作 .bak（仅当上一份可解析时），损坏时可回滚。
    临时文件名带 pid，保证两进程并发写不会撞同一个临时文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        if path.exists():
            try:
                json.loads(path.read_text(encoding="utf-8"))  # 仅校验可解析
                shutil.copyfile(path, path.with_name(path.name + ".bak"))
            except Exception:  # noqa: BLE001
                pass
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def _xlock(self):
    """跨进程文件锁：让常驻闭环与看板两个进程串行落盘，互不覆盖。

    纸账户的权威态在两个独立进程间共享同一份 JSON 文件。若两个进程同时
    读-改-写（例如看板 reset 的同时常驻进程 buy），后写的会覆盖先写的，
    造成成交/持仓/本金丢失。用 msvcrt 对同一个 ``.paper_account.lock``
    文件加**跨进程**互斥锁，保证同一时刻只有一个进程在改写磁盘。

    锁固定的 0 号字节（双方都 ``seek(0)`` 后加锁），确保锁的是同一段区间，
    真正的全局互斥。非 Windows（msvcrt 不可用）退化为无锁（原子写仍兜底）。

    同进程同实例可重入：嵌套调用只在外层真正加/解 OS 锁，避免
    ``_check_daily_reset`` 内的 ``_save`` 与外层 ``buy`` 的锁嵌套死锁。
    """
    if msvcrt is None:
        yield
        return
    depth = getattr(self, "_xlock_depth", 0)
    if depth > 0:
        self._xlock_depth = depth + 1
        try:
            yield
        finally:
            self._xlock_depth -= 1
        return
    lock_path = self.account_file.parent / ".paper_account.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+b")
    except Exception:  # noqa: BLE001
        yield  # 锁文件拿不到就退化（原子写仍兜底）
        return
    try:
        fh.seek(0)
    except Exception:  # noqa: BLE001
        pass
    deadline = _time.time() + 10.0
    acquired = False
    while True:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            acquired = True
            break
        except OSError:
            if _time.time() >= deadline:
                break
            _time.sleep(0.02)
    if not acquired:
        try:
            fh.close()
        except Exception:  # noqa: BLE001
            pass
        logger.warning("⚠️ 纸账户跨进程锁超时（10s），退化为无锁写入（原子写兜底）")
        yield
        return
    self._xlock_depth = 1
    self._xlock_fh = fh
    try:
        yield
    finally:
        try:
            fh.seek(0)
            msvcrt.locking(self._xlock_fh.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._xlock_fh.close()
        except Exception:  # noqa: BLE001
            pass
        self._xlock_depth = 0
        self._xlock_fh = None


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
        # 当日起始权益（float | None）。None = 当日尚未建仓/尚未定基线，
        # 此时 today_pnl_market() 退化为 today_pnl（已实现口径），不会返回假数。
        self._day_start_equity: Optional[float] = None
        
        # 线程安全
        self._lock = threading.Lock()
        # 跨进程锁状态（让常驻闭环与看板串行落盘，互不覆盖；同实例可重入）
        self._xlock_depth = 0
        self._xlock_fh = None
        
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

    def _read_json(self, path):
        """读 JSON（崩溃安全）：主文件损坏/截断时自动回滚 .bak。

        纸账户文件由常驻闭环 + 看板两个进程并发写，且常被 ``taskkill /F`` 强杀。
        一旦落盘中途被杀，文件会被截成半截 → 下次 JSON 解析失败。若直接抛错，
        账户会被当成"空账户"清空（现金/持仓/历史全丢，这是 2026-10 实测过的灾难）。
        这里：主文件能解析就用主文件；主文件坏则回滚上一份 .bak（原子写每次都会
        留一份"上一份完整文件"当 .bak）；两份都坏才返回 None（调用方退化为新账户）。
        """
        path = Path(path)
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logger.warning(f"⚠️ 纸账户主文件损坏/截断，尝试 .bak 回滚: {path.name}")
        bak = path.with_name(path.name + ".bak")
        try:
            if bak.exists():
                data = json.loads(bak.read_text(encoding="utf-8"))
                # 把 .bak 还原成主文件，避免下一次还从坏文件读
                try:
                    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
                except Exception:  # noqa: BLE001
                    pass
                logger.warning(f"♻️ 已从 .bak 恢复纸账户文件: {path.name}")
                return data
        except Exception as e:  # noqa: BLE001
            logger.warning(f"⚠️ .bak 也损坏，放弃恢复: {path.name} ({e})")
        return None

    def _load(self):
        """从磁盘恢复状态（崩溃安全：主文件坏则回滚 .bak，最坏退化为新账户）"""
        acct = self._read_json(self.account_file)
        if acct is not None:
            self.cash = acct.get("cash", self.initial_cash)
            self.holdings = acct.get("holdings", {})
            # 先把 market_value 拉回 quantity×last_price，再做下面的权益对账，
            # 否则陈旧市值会连带把本金基线也算偏。
            self._normalize_holdings()
            # 兼容旧存档：补齐 T+1 相关字段，否则卖出时会 KeyError
            for _h in self.holdings.values():
                _h.setdefault("today_quantity", 0)
                _h.setdefault("available", int(_h.get("quantity", 0) or 0))
            self.today_pnl = acct.get("today_pnl", 0.0)
            self.today_trades = acct.get("today_trades", 0)
            self._last_reset_date = acct.get("last_reset_date", self._last_reset_date)
            # ★ 本金基线必须与「实际注入的权益」对齐，否则盈亏/收益率/回撤会被按
            # 默认的 ¥100,000 误算（出现 4890% 这类幽灵收益 —— 即「双账户态不一致」）。
            # 历史存档若未记录 initial_cash（账户被外部注入了资金/持仓），则以载入时的
            # 总资产作为 0 基准，避免把注入资本算成盈利。
            persisted = acct.get("initial_cash", None)
            if persisted is not None:
                self.initial_cash = float(persisted)
            else:
                loaded_equity = self.cash + sum(
                    float(h.get("market_value", 0.0)) for h in self.holdings.values()
                )
                self.initial_cash = loaded_equity
            # 当日基线：跨进程重启后当日盈亏不能归零重算，
            # 否则「重启一次，当日收益率就闪回 0」—— 用户会以为系统失忆。
            # 存档里有就用存档的；没有（当日首个进程）才以载入总资产为基线。
            self._day_start_equity = acct.get("day_start_equity", None)
            if self._day_start_equity is None:
                self._day_start_equity = (self.cash + sum(
                    float(h.get("market_value", 0.0)) for h in self.holdings.values()
                )) if self._last_reset_date == datetime.now().strftime("%Y-%m-%d") else None
            logger.info(f"📂 纸交易账户已恢复: 现金 ¥{self.cash:,.0f}, {len(self.holdings)} 只持仓"
                        f" · 本金基线 ¥{self.initial_cash:,.0f}")
        else:
            if self.account_file.exists():
                logger.error("💥 纸账户主文件与 .bak 均损坏，已重置为新账户（历史可能丢失）")
            else:
                logger.info(f"💰 新建纸交易账户: 初始资金 ¥{self.initial_cash:,.0f}")

        hist = self._read_json(self.history_file)
        if hist is not None:
            try:
                self.trade_history = [TradeRecord(**r) for r in hist]
                logger.info(f"📋 已加载 {len(self.trade_history)} 条交易记录")
            except Exception as e:  # noqa: BLE001
                # 个别记录缺字段会导致整份反序列化失败 —— 退化为逐条解析，
                # 能救一条是一条，绝不整体丢弃历史。
                logger.warning(f"交易记录反序列化失败（字段缺失），逐条抢救: {e}")
                kept = []
                for r in hist:
                    try:
                        kept.append(TradeRecord(**r))
                    except Exception:  # noqa: BLE001
                        pass
                self.trade_history = kept
                logger.warning(f"♻️ 已从损坏的历史中抢救出 {len(kept)} 条成交记录")
    
    def _save(self):
        """持久化（带跨进程串行 + 原子写）。

        独立调用方（如 mark_to_market 后的 save）走这里；buy/sell/reset 在已持有
        跨进程锁时改走 ``_save_nolock``，避免重复加锁。
        """
        with _xlock(self):
            self._save_nolock()

    def _save_nolock(self):
        """无锁落盘：调用方必须已持有跨进程锁(_xlock)，否则并发写会互相覆盖。"""
        try:
            self._normalize_holdings()   # 落盘前再拉正一次，防止外部改 last_price
            data = {
                "cash": self.cash,
                "initial_cash": self.initial_cash,
                "holdings": self.holdings,
                "today_pnl": self.today_pnl,
                "today_trades": self.today_trades,
                "last_reset_date": self._last_reset_date,
                "day_start_equity": self._day_start_equity,
            }
            _atomic_write_text(self.account_file, json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            pass
    
    def save(self):
        """对外落盘入口。

        ``mark_to_market()`` 只改内存，历史上从不落盘 —— 于是
        ``data/paper_account.json`` 里的 ``last_price`` 一直停在下一次买入/卖出
        那一刻。看板是**另一个进程**，读到的就是这份陈旧文件，于是持仓浮盈和
        风控口径对不上（2026-10-20：磁盘估值比真实低 ¥4 万，凭空多出 -¥5.5 万
        当日亏损，直接把熔断打爆）。mark 完必须让磁盘跟上。
        """
        self._save()

    def reload(self):
        """从磁盘重载账户与交易历史（看板是独立进程，必须主动重载才能看到
        常驻闭环写入的最新持仓/权益/成交记录）。

        看板的 ``paper_account`` 是模块级单例，只在 Streamlit 进程启动那一刻
        从磁盘载入一次；之后常驻 ``run_loop.py`` 交易产生的变更只落在磁盘上，
        看板若不主动 reload，刷新多少次都看不到新成交、新持仓。本方法让
        「账户仪表盘 / 交易记录」tab 每次渲染前都对齐到磁盘真相。
        """
        with self._lock:
            self._load()

    # ========== 当日盈亏口径 ==========

    def day_start_equity(self) -> Optional[float]:
        """当日起始权益（现金 + 持仓按买入成本计）。

        用于算**真实**当日盈亏：``总资产 − 当日起始权益``。
        ``today_pnl`` 只在**卖出**时累加已实现盈亏，是"已实现口径"，
        绝不能当成"今日赚了多少"展示 —— 一整天只买不卖时它会停在 0，
        而现金已经被建仓抽走，看上去像"没亏"；反过来若把现金变动当成盈亏
        又会把建仓误算成亏损。两种口径混用是 2026-10-20 排障时最大的误导源。

        返回 None = 当日基线未定（新建账户尚未建仓，或跨日刚重置）。
        """
        return (None if self._day_start_equity is None
                else float(self._day_start_equity))

    def today_return_pct(self) -> float:
        """当日收益率（%），基于 day_start_equity；无基线时返回 0。"""
        base = self._day_start_equity
        if not base or base <= 0:
            return 0.0
        return (self.get_total_asset() - base) / base * 100.0

    def today_pnl_market(self) -> float:
        """当日浮动盈亏（已实现 + 持仓浮动），口径与风控一致。"""
        base = self._day_start_equity
        if not base:
            return self.today_pnl
        return self.get_total_asset() - base

    def _save_history(self):
        with _xlock(self):
            self._save_history_nolock()

    def _save_history_nolock(self):
        """无锁落盘历史：调用方必须已持有跨进程锁(_xlock)。"""
        try:
            records = [r.__dict__ for r in self.trade_history[-500:]]  # 保留最近500条
            _atomic_write_text(self.history_file, json.dumps(records, ensure_ascii=False, indent=2))
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
            # 新的一天：当日基线必须跟着重置，否则收益率会拿昨天的起点算，恒为负
            self._day_start_equity = None
            self._save()
    
    # ========== 交易操作 ==========
    
    def buy(self, code: str, name: str, price: float, quantity: int,
            strategy: str = "", ai_score: float = 0.0) -> Tuple[bool, str, Optional[TradeRecord]]:
        """买入（虚拟成交）"""
        with self._lock:
            with _xlock(self):   # 跨进程串行：防止与看板 reset / 另一进程写入互相覆盖
                self._load()     # 落盘前先对齐磁盘最新态（两进程共享同一份文件）
                self._check_daily_reset()
                # 当日第一笔建仓前先定格基线：此刻 total_asset 才是真正的"今日起点"
            if self._day_start_equity is None:
                self._day_start_equity = self.cash + sum(
                    float(h.get("market_value", 0.0) or 0.0)
                    for h in self.holdings.values())
            
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

            # 必须把成交历史落盘：看板是**另一个进程**，只读磁盘上的
            # paper_trade_history.json。旧实现只设了 _history_dirty 却从不调用
            # _save_history()，于是交易记录永远写不进磁盘 —— 看板「交易记录」tab
            # 始终空白，且常驻进程重启后历史全丢。
            self._save_history_nolock()
            self._save_nolock()
            _slip = (fill_price - price) * quantity
            logger.info(f"📈 纸交易买入: {name}({code}) {quantity}股 @¥{fill_price:.2f} "
                       f"(盘口¥{price:.2f} 滑点¥{_slip:.2f}) 费用¥{commission:.2f} "
                       f"余额¥{self.cash:,.0f}")
            
            return True, f"✅ 买入 {quantity}股 @¥{fill_price:.2f}（滑点¥{_slip:.2f}，T+1 锁定）", record
    
    def sell(self, code: str, price: float, quantity: int,
             strategy: str = "", ai_score: float = 0.0) -> Tuple[bool, str, Optional[TradeRecord]]:
        """卖出"""
        with self._lock:
            with _xlock(self):   # 跨进程串行：防止与看板 reset / 另一进程写入互相覆盖
                self._load()     # 落盘前先对齐磁盘最新态（两进程共享同一份文件）
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

            # 配对：把这笔平仓和最早的、尚未被平仓的同标的开仓买入勾在一起（FIFO）。
            # 双向打标：买入记录的 close_id → 平仓卖出的 trade_id；
            # 卖出记录的 close_id → 被平的开仓买入的 trade_id。
            # 看板交易记录据此显示「开仓↔平仓」配对，否则每笔买卖孤立、无法回溯。
            for _tr in self.trade_history:
                if _tr.side == "buy" and _tr.code == code and not _tr.close_id:
                    _tr.close_id = record.trade_id      # 买入 → 卖出
                    record.close_id = _tr.trade_id      # 卖出 → 买入
                    break

            self._history_dirty = True

            # 必须把成交历史落盘：看板是**另一个进程**，只读磁盘上的
            # paper_trade_history.json。旧实现只设了 _history_dirty 却从不调用
            # _save_history()，于是交易记录永远写不进磁盘 —— 看板「交易记录」tab
            # 始终空白，且常驻进程重启后历史全丢。
            self._save_history_nolock()
            self._save_nolock()
            _slip = (price - fill_price) * quantity
            logger.info(f"📉 纸交易卖出: {h.get('name', code)}({code}) {quantity}股 @¥{fill_price:.2f} "
                       f"(盘口¥{price:.2f} 滑点¥{_slip:.2f}) "
                       f"盈亏¥{pnl:+,.2f}({pnl_pct:+.2f}%) 余额¥{self.cash:,.0f}")
            
            return True, f"✅ 卖出 {quantity}股 @¥{fill_price:.2f} 盈亏¥{pnl:+,.2f}", record
    
    # ========== 市值更新 ==========
    
    def mark_to_market(self, prices: Dict[str, float]):
        """按最新市价更新持仓市值"""
        with self._lock:
            # 当日首个有效行情 = 当日基线定格点。必须在 mark 之前取，
            # 否则买入/卖出改变现金后再定基线，会把当天第一笔的损益吞掉。
            if self._day_start_equity is None:
                self._day_start_equity = self.cash + sum(
                    float(h.get("market_value", 0.0) or 0.0)
                    for h in self.holdings.values())
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
    
    def reset(self, new_cash: float = None, keep_history: bool = True):
        """重置账户（清仓 + 重置当日统计，默认保留成交历史）。

        默认 **保留** 成交历史（keep_history=True）—— 之前点「重置账户」会无脑把
        ``paper_trade_history.json`` 一并清空，导致无法回溯某笔买卖的配对/盈亏，
        是个易踩的坑。只有在用户**显式**勾选「同时清空成交历史」时（UI 传
        keep_history=False）才做清盘式重置。

        Args:
            new_cash: 重置后的初始本金；None 则沿用当前 initial_cash。
            keep_history: True=保留历史成交记录；False=连历史一起清空（测试/主动清账）。
        """
        with self._lock:
            with _xlock(self):   # 跨进程串行：防止与常驻进程买入/卖出互相覆盖
                self._load()     # 先对齐磁盘最新态（含另一进程刚写的成交/持仓）
                self.cash = new_cash if new_cash is not None else self.initial_cash
                self.initial_cash = self.cash  # 本金基线跟着新本金走，累计盈亏口径才正确
                self.holdings = {}
                self.today_pnl = 0.0
                self.today_trades = 0
                self._day_start_equity = None
                self._last_reset_date = datetime.now().strftime("%Y-%m-%d")
                if not keep_history:
                    self.trade_history = []
                self._save_nolock()
                self._save_history_nolock()
            logger.info(f"🔄 纸交易账户已重置: ¥{self.cash:,.0f}"
                        f"{'（保留成交历史）' if keep_history else '（已清空成交历史）'}")


# 全局单例
paper_account = PaperAccount()

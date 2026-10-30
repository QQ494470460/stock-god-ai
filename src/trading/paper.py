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
        # 「本账户确实被人重置过」的**证据**：只有 reset() 会写它，高精度时间戳。
        # 常驻闭环拿它判断「initial_cash 变了」到底是人工重置还是状态被写坏 ——
        # 2026-10-30 09:42 空白单例覆盖账户时，边基线跟着变成 10 万，闭环把这次
        # 覆盖当成「用户重置」并自动 rebaseline_peak，回撤归零、熔断被静默解除。
        self.reset_at: str = ""
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

            # T+1 字段不变量：0 <= available <= quantity，0 <= today_quantity <= quantity。
            # 载入与落盘都走这里，所以任何来源（旧存档、手工编辑、外部注入、
            # taskkill 打断的半截写入）造成的越界都会在下一个口子上被拉正。
            #
            # ⚠️ 越界不是「难看」而是「致命」：``sellable_quantity()`` 算的是
            # ``max(0, min(available, quantity - today, quantity))``，available 为负
            # 会被夹成 0 ⇒ 该标的当天 sellable=0 ⇒ 卖出在**预检阶段**就被
            # 「T+1 锁仓」拦下（连止损都发不出去），且只有次日 reset 才自愈。
            # 所以越界时不能简单夹到 0，而要按「quantity 是权威、今日买入量次之」
            # 重建：available = quantity - today_quantity。
            try:
                _tq = int(h.get("today_quantity", 0) or 0)
            except (TypeError, ValueError):
                _tq = 0
            _tq = max(0, min(_tq, qty))
            try:
                _avail = int(h.get("available", qty) or 0)
            except (TypeError, ValueError):
                _avail = qty
            if _avail < 0 or _avail > qty:
                _avail = qty - _tq
            h["today_quantity"] = _tq
            h["available"] = _avail

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
            self.reset_at = str(acct.get("reset_at", "") or "")
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

    def _looks_pristine(self) -> bool:
        """内存里是不是一个「从没被用过」的全新账户（默认 10 万、无持仓）。

        这个形态只有一个合法来源：``PaperAccount()`` 构造后 ``_load`` 没读到
        任何状态。它一旦被 ``_save`` 落盘，磁盘上的真实账户就被抹掉了 ——
        2026-10-30 09:42 正是这么丢的：账户从 ¥4,218,917 + 1 只持仓变成
        ¥100,000 + 0 持仓，风控基线跟着被对齐到 10 万，回撤熔断被静默解除。
        """
        return (not self.holdings
                and abs(float(self.cash) - 100_000.0) < 1e-6
                and abs(float(self.initial_cash) - 100_000.0) < 1e-6)

    def _disk_has_state(self) -> bool:
        """磁盘上是否已有「非全新」账户。**读不出来一律按"有"处理**（保守）。

        判据：文件存在且（有持仓 或 initial_cash ≠ 10 万），或者文件存在但
        解析失败 —— 后者说明状态可能被截断/占用，更不能拿空白账户去覆盖。
        """
        p = Path(self.account_file)
        try:
            if not p.exists():
                return False
        except Exception:  # noqa: BLE001
            return True
        raw = self._read_json(p)
        if raw is None:
            return True          # 存在但读不出来 → 视为有状态，绝不允许覆盖
        try:
            if raw.get("holdings"):
                return True
            return abs(float(raw.get("initial_cash", 0) or 0) - 100_000.0) > 1e-6
        except Exception:  # noqa: BLE001
            return True

    def _save_nolock(self, force: bool = False):
        """无锁落盘：调用方必须已持有跨进程锁(_xlock)，否则并发写会互相覆盖。

        ``force=False`` 时带一道「空白账户不得覆盖已有状态」的闸：全新账户
        （10 万、无持仓）只在磁盘上确实没有状态时才允许落盘。真正的主动重置
        走 ``reset()``，它会显式传 ``force=True``。
        """
        try:
            if not force and self._looks_pristine() and self._disk_has_state():
                logger.error(
                    "🛑 拒绝用空白账户覆盖磁盘上已有的纸账户状态"
                    f"（{Path(self.account_file).name}）—— "
                    "疑似 _load 失败后被误持久化。主动重置请走 PaperAccount.reset()。"
                )
                return
            self._normalize_holdings()   # 落盘前再拉正一次，防止外部改 last_price
            data = {
                "cash": self.cash,
                "initial_cash": self.initial_cash,
                "holdings": self.holdings,
                "today_pnl": self.today_pnl,
                "today_trades": self.today_trades,
                "last_reset_date": self._last_reset_date,
                "reset_at": self.reset_at,
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
    
    def _roll_day_if_needed(self) -> None:
        """跨日滚动：解锁 T+1 + 重置当日基线。**任何读路径都应先调它**。

        为什么读路径也要滚
        ------------------
        解锁原本只写在 ``buy`` / ``sell`` 里。常驻闭环跨日后，如果当天还没有
        任何买入，``guard`` 取持仓 → ``sellable_quantity`` 仍返回「数量 - 昨日
        买入量」→ 卖出在**信号预检阶段**就被「T+1 锁仓」拦下；而拦下就不会调用
        ``sell()``，也就永远不会触发解锁 → 昨日买入的持仓**整天都卖不掉**，
        包括止损。实测隐患：隔夜持仓的止损会静默失效，直到某次买入把它「唤醒」。

        与 ``mark_to_market`` 共用 ``_lock`` 时的注意事项：``_save`` 只拿跨进程
        锁 ``_xlock``（可重入），不碰 ``_lock``，因此这里不会再抢 ``_lock``。
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            if today == self._last_reset_date:
                return   # 绝大多数调用落在这一行，开销仅一次 strftime
            with self._lock:
                self._check_daily_reset()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"跨日滚动失败（继续用旧状态，可能影响 T+1 解锁）: {e}")

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
                # 成本基准把**买入佣金资本化**：否则卖出算 pnl 时只扣了卖出费，
                # 买入佣金凭空消失 → 每笔盈亏系统性高估一个佣金。券商的
                # 「摊薄成本」也是含费口径，这里与之一致。
                new_cost = (
                    (old_cost * old_qty + amount + commission) / new_qty
                    if new_qty > 0 else 0
                )
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
                    # 单位成本含佣金（= 总支出/股数），与卖出端 pnl 口径对齐
                    "avg_cost": total_cost / quantity,
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
        """卖出

        A 股规则：买入必须整手（100 股整数倍），**卖出允许不足一手，但必须
        一次性把该标的的零股全部卖掉**。旧实现把卖出也强行整手取整，导致
        「持有 150 股 → 只能卖 100 → 剩 50 股永远卖不掉」——这类零股来自
        送股/配股/拆分，一旦卡住，连止损都无法执行（且每 tick 刷一条废单日志）。
        """
        with self._lock:
            with _xlock(self):   # 跨进程串行：防止与看板 reset / 另一进程写入互相覆盖
                self._load()     # 落盘前先对齐磁盘最新态（两进程共享同一份文件）
                self._check_daily_reset()

            if code not in self.holdings:
                return False, f"无 {code} 持仓", None

            h = self.holdings[code]
            held = int(h.get("quantity", 0) or 0)

            if held > 0 and int(quantity) >= held:
                # 清仓：零股一次性卖出是合法的，别取整
                quantity = held
            else:
                quantity = (quantity // 100) * 100
                if quantity < 100:
                    return False, "不足1手（非清仓卖出必须整手）", None

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
            # available 必须夹在 [0, quantity]：历史存档 / 外部注入的 available
            # 可能小于「quantity - today_quantity」，直接相减会变负；而
            # ``sellable_quantity()`` 是 ``max(0, min(avail, qty-today, qty))``，
            # 负值会被夹成 0 ⇒ 该标的**当天再也卖不掉**（包括止损），UI 还会显示
            # 「可卖 -50 股」。这是一条静默失效链：不报错、不告警，只是从此不动。
            try:
                _avail = int(h.get("available", 0) or 0) - quantity
            except (TypeError, ValueError):
                _avail = 0
            h["available"] = max(0, min(_avail, int(h["quantity"])))
            if h["quantity"] <= 0:
                del self.holdings[code]
            else:
                # 剩余仓位按**盘口价**重新标价（不是成交价：mark 用的是市场价，
                # 滑点属于已实现的交易成本，已体现在现金里）。同时必须同步
                # ``last_price`` —— ``_normalize_holdings`` 会按 quantity×last_price
                # 重算市值，若只改 market_value，本次标价会在落盘时被
                # 陈旧的 last_price 覆盖掉。看板手动卖出（不经 loop 的
                # mark_to_market）正因此会把「卖出那一刻的标价」丢掉，
                # 导致总资产/收益率继续按隔夜价算。
                h["last_price"] = price
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

    # ========== 除权除息 ==========

    def apply_corporate_action(
        self, code: str, kind: str,
        ratio: float = 0.0, dividend_per_share: float = 0.0,
        ex_date: str = "",
    ) -> Tuple[bool, str]:
        """处理除权除息，保持账本价值连续（避免除权日出现 Phantom 盈亏/止损失效）。

        A 股常见公司行为：

        - ``"split"`` / ``"bonus"`` 送股 / 转增：每 1 股变 ``(1+ratio)`` 股，成本同比例
          摊薄，现金不变。例：10 送 10 → ``ratio=1.0``，持股 1000 变 2000、均价减半。
        - ``"dividend"`` 现金分红：``cash += 持股数 × dividend_per_share``，持仓股数/
          成本不动（除息价由市场自然回落，mark_to_market 会接管）。
        - ``"merge"`` 缩股：每 ``ratio(<1)`` 股并 1 股，成本同比例抬高，现金不变。

        为什么必须做：不复权会让持仓在除权日凭空「少一半市值 + 成本价不变」，
        止损线与盈亏比同时失真 —— 一次 10 送 10 就能把一笔盈利仓算成巨亏，触发
        误斩仓。这里把股数/成本同步调整，使 ``持股数×成本 × 价格`` 在除权前后连续。

        不抛出任何异常：参数非法或持仓缺失只返回 (False, 原因)，主流程照常。
        """
        kind = str(kind or "").strip().lower()
        with self._lock:
            with _xlock(self):
                self._load()
                if code not in self.holdings:
                    return False, f"无 {code} 持仓，无法除权", None
                h = self.holdings[code]
                try:
                    qty = int(h.get("quantity", 0) or 0)
                    avail = int(h.get("available", qty) or 0)
                    today = int(h.get("today_quantity", 0) or 0)
                    cost = float(h.get("avg_cost", 0) or 0)
                    px = float(h.get("last_price", 0) or 0)
                except (TypeError, ValueError):
                    return False, "持仓字段损坏"

                if kind in ("split", "bonus"):
                    factor = 1.0 + float(ratio or 0.0)
                    if factor <= 0:
                        return False, f"送转比例非法: {ratio}"
                    new_qty = int(round(qty * factor))
                    h["quantity"] = new_qty
                    h["available"] = int(round(avail * factor))
                    h["today_quantity"] = int(round(today * factor))
                    h["avg_cost"] = round(cost / factor, 6) if cost > 0 else cost
                    h["market_value"] = round(new_qty * px, 2) if px > 0 else h.get("market_value", 0)
                    note = f"送转 {ratio*100:.0f}%（每 1 股变 {factor:.2f} 股，成本摊薄）"
                elif kind == "dividend":
                    dps = float(dividend_per_share or 0.0)
                    if dps <= 0:
                        return False, f"每股分红非法: {dividend_per_share}"
                    self.cash += round(qty * dps, 2)
                    note = f"现金分红 ¥{dps:.4f}/股 × {qty}股 = ¥{qty*dps:,.2f}"
                elif kind == "merge":
                    factor = float(ratio or 1.0)
                    if factor <= 0 or factor >= 1:
                        return False, f"缩股比例非法（需 0<ratio<1）: {ratio}"
                    new_qty = int(round(qty * factor))
                    if new_qty <= 0:
                        return False, "缩股后股数为 0"
                    h["quantity"] = new_qty
                    h["available"] = int(round(avail * factor))
                    h["today_quantity"] = int(round(today * factor))
                    h["avg_cost"] = round(cost / factor, 6) if cost > 0 else cost
                    h["market_value"] = round(new_qty * px, 2) if px > 0 else h.get("market_value", 0)
                    note = f"缩股（每 {factor:.2f} 股并 1 股，成本抬高）"
                else:
                    return False, f"未知公司行为类型: {kind}", None

                self._normalize_holdings()

                # 审计记录：用独立 side，performance.attribute 只统计 side=="sell"，
                # 不会污染已实现盈亏；看板可据 strategy 单独展示「公司行为」。
                tid = f"CA{datetime.now().strftime('%Y%m%d%H%M%S')}{self.today_trades:03d}"
                rec = TradeRecord(
                    trade_id=tid, code=code, name=h.get("name", code),
                    side="corporate_action", price=0.0, quantity=qty,
                    amount=0.0, filled_at=datetime.now().isoformat(),
                    strategy="corporate_action",
                    ai_score=0.0,
                )
                self.trade_history.append(rec)
                self._history_dirty = True
                self._save_history_nolock()
                self._save_nolock()
                logger.info(f"🪙 公司行为 {code} {kind}: {note}")
                return True, f"✅ {code} 已{('除权' if kind!='dividend' else '分红')}：{note}", rec

    # ========== 市值更新 ==========
    
    def mark_to_market(self, prices: Dict[str, float]):
        """按最新市价更新持仓市值"""
        # 先滚日：常驻进程跨日后，若不在这里解锁，止盈止损会拿「昨日买入量」
        # 当今日锁仓量，导致开盘第一轮止损全部被 T+1 拦掉。
        self._roll_day_if_needed()
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
        """当前持仓列表

        先滚日再返回：guard 的 ``_positions()`` 直接从这条路取持仓，用于判断
        T+1 可卖量。不滚日的话，跨日后「昨日买入」仍算今日锁定，止损会被
        永久拦在预检阶段（详见 ``_roll_day_if_needed`` 的说明）。
        """
        self._roll_day_if_needed()
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
                # 重置证据：高精度时间戳，闭环靠它区分「人工重置」与「状态被写坏」
                self.reset_at = datetime.now().isoformat(timespec="microseconds")
                if not keep_history:
                    self.trade_history = []
                self._save_nolock(force=True)   # 主动重置是明确意图，绕过空白账户保护
                self._save_history_nolock()
            logger.info(f"🔄 纸交易账户已重置: ¥{self.cash:,.0f}"
                        f"{'（保留成交历史）' if keep_history else '（已清空成交历史）'}")


# 全局单例
paper_account = PaperAccount()

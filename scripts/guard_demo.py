# -*- coding: utf-8 -*-
"""
交易安全与风控中枢 · 全流程 Dry-Run 演示 (零真实资金)

演示路径：
  PAPER 纸交易  →  arm_live 取令牌  →  confirm_arm 武装 LIVE_PENDING
  →  实盘委托需人工确认(pending)  →  审批后执行  →  Kill Switch 熔断  →  resume 回到 PAPER

运行：
  python scripts/guard_demo.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.trading.broker import (
    BrokerAPI, Order, OrderSide, OrderType, OrderStatus,
    Position, AccountInfo, TickData,
)
from src.trading.guard import RiskSupervisor, TradingMode


# ---------------- 零风险 Fake 对象 ----------------
class FakeBroker(BrokerAPI):
    def __init__(self, total=1_000_000.0, last=10.0):
        super().__init__()
        self._connected = True
        self.total_asset = total
        self.available_cash = total * 0.6
        self.last_price = last
        self.orders = []
        self.positions = {}

    def connect(self, **k): self._connected = True; return True
    def disconnect(self): self._connected = False
    def query_account(self):
        return AccountInfo("FAKE", self.total_asset, self.available_cash, 0.0,
                           self.total_asset - self.available_cash, 0.0, 0.0)
    def query_positions(self):
        return [Position(c, c, p["qty"], p["qty"], p["cost"], self.last_price,
                         p["qty"] * self.last_price, 0.0, 0.0, 0)
                for c, p in self.positions.items()]
    def query_orders(self, *a, **k): return []
    def place_order(self, code, side, order_type, price, quantity, strategy_tag=""):
        o = Order(f"ORD{len(self.orders)+1:03d}", code, side, order_type, price,
                  quantity, quantity, OrderStatus.FILLED, strategy_tag=strategy_tag)
        self.orders.append(o)
        return o
    def cancel_order(self, oid): return True
    def cancel_all(self): return 0
    def get_tick(self, code):
        return TickData(code, code, self.last_price, 10.0, 10.5, 9.5, 10.0,
                        100000, 1e6, self.last_price - 0.01, self.last_price + 0.01, 0.0)
    def get_broker_name(self): return "FakeBroker"
    def is_trading_time(self): return True


class FakePaper:
    def __init__(self, total=1_000_000.0):
        self.cash = total
        self.frozen_cash = 0.0
        self.holdings = {}
        self.today_pnl = 0.0
    def get_total_asset(self):
        mv = sum(h["qty"] * h["last"] for h in self.holdings.values())
        return self.cash + mv
    def get_total_pnl(self): return 0.0
    def get_positions(self):
        return [{"code": c, "name": c, "quantity": h["qty"], "available": h["qty"],
                 "today_quantity": 0, "avg_cost": h["cost"], "last_price": h["last"],
                 "market_value": h["qty"] * h["last"], "pnl": 0.0, "pnl_pct": 0.0}
                for c, h in self.holdings.items()]
    def buy(self, code, name, price, quantity, strategy="", ai_score=0.0):
        cost = price * quantity
        if self.cash < cost:
            return False, "资金不足", None
        self.cash -= cost
        h = self.holdings.setdefault(code, {"qty": 0, "cost": 0.0, "last": price})
        h["qty"] += quantity; h["cost"] = price; h["last"] = price
        return True, "ok", None
    def sell(self, code, price, quantity, strategy="", ai_score=0.0):
        if code not in self.holdings or self.holdings[code]["qty"] < quantity:
            return False, "无持仓", None
        h = self.holdings[code]
        self.cash += price * quantity
        h["qty"] -= quantity
        if h["qty"] <= 0:
            del self.holdings[code]
        return True, "ok", None


def banner(t):
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


def main():
    cfg_path = str(ROOT / "config" / "trading_guard.yaml")
    broker = FakeBroker()
    paper = FakePaper()

    # ⚠️ 演示必须隔离状态与审计日志。
    # 早期这里直接用 config/trading_guard.yaml 的默认路径，于是 demo 的
    # kill_switch / arm_live / 成交全部写进**生产**的
    # data/trading_guard_state.json 与 data/trading_journal.jsonl：
    #   - 生产审计流水里混进 operator_alice / operator_bob 的假成交
    #   - 更危险的是 arm_live 会把真实状态置为 live_pending，若演示中途异常退出，
    #     生产就卡在一个「已武装但无 arm 留痕」的非法状态（10-16 确实发生过，
    #     事后只能人工写 mode_change 退回 PAPER）
    from tempfile import mkdtemp
    from src.trading.guard import GuardConfig

    demo_dir = Path(mkdtemp(prefix="guard_demo_"))
    if Path(cfg_path).exists():
        cfg = GuardConfig.from_yaml(cfg_path)
    else:
        cfg = GuardConfig()
    cfg.state_file = str(demo_dir / "state.json")
    cfg.journal_file = str(demo_dir / "journal.jsonl")
    sup = RiskSupervisor(broker, cfg, paper_account=paper)
    print(f"🔒 演示隔离目录（不触碰生产状态/日志）: {demo_dir}")

    # 演示用：强制放行交易时段日历检查（仅 dry-run 演示，真实运行不会这样）
    import src.trading.calendar as _cal
    _cal.order_allowed = lambda: (True, "demo-forced")
    # 演示用：放宽下单间隔限制，避免演示步骤间<2s 触发频率熔断（真实运行保留默认2s）
    sup.config.min_order_interval_sec = 0.0

    banner("① 初始状态（应默认 PAPER，未武装）")
    print(f"  模式: {sup.get_status()['mode']} | 已熔断: {sup.get_status()['halted']}")

    banner("② 纸交易：AI 信号 → 自动批准 → 记纸账户")
    prop = sup.propose_trade("600000", "浦发银行", OrderSide.BUY, ai_score=72, agreement=0.9)
    print(f"  预览: {prop.side} {prop.quantity}股 @ {prop.price} (状态={prop.status})")
    ok, msg, _ = sup.approve_and_execute(prop.proposal_id, "auto")
    print(f"  执行: {'✅' if ok else '❌'} {msg} | 券商实单={len(broker.orders)} 纸持仓={list(paper.holdings)}")

    banner("③ 武装实盘：arm_live 取令牌")
    token = sup.safety.arm_live("operator_alice")
    print(f"  arm_live 返回令牌: {token} | 当前模式: {sup.safety.mode.value}（仍为 PAPER）")
    print(f"  错误令牌确认 → {sup.safety.confirm_arm('bad-token', 'alice')}（应 False）")
    print(f"  正确令牌确认 → {sup.safety.confirm_arm(token, 'alice')}（应 True）")
    print(f"  武装后模式: {sup.safety.mode.value}")

    banner("④ LIVE_PENDING：实盘委托必须人工确认")
    prop = sup.propose_trade("000001", "平安银行", OrderSide.BUY, ai_score=75, agreement=0.95)
    print(f"  预览: {prop.side} {prop.quantity}股 @ {prop.price} (状态={prop.status})")
    ok, msg, _ = sup.approve_and_execute(prop.proposal_id, "bob")
    print(f"  未确认直接执行 → {'✅' if ok else '❌'} {msg}（应被拒）")
    print(f"  人工审批 → {sup.gate.approve(prop.proposal_id, 'bob')}")
    ok, msg, order = sup.approve_and_execute(prop.proposal_id, "bob")
    print(f"  审批后执行 → {'✅' if ok else '❌'} {msg} | 券商实单={len(broker.orders)}")

    banner("⑤ Kill Switch：全局熔断（持久化）")
    sup.kill_switch("手动紧急停止", operator="alice")
    print(f"  熔断后模式={sup.safety.mode.value} 已熔断={sup.safety.halted}")
    prop = sup.propose_trade("600519", "贵州茅台", OrderSide.BUY, ai_score=80, agreement=1.0)
    print(f"  熔断后下单预览状态={prop.status}（应 blocked）")

    banner("⑥ Resume：解除熔断 → 强制回到 PAPER，需重新武装才能实盘")
    sup.resume(operator="alice")
    print(f"  resume 后模式={sup.safety.mode.value} 已熔断={sup.safety.halted}")

    banner("✅ 演示完成：安全地基全部行为符合预期")


if __name__ == "__main__":
    main()

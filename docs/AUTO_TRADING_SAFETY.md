# 自动交易 · 安全与风控中枢（分阶段实盘上线指南）

> 本文件说明 `src/trading/guard.py` 的设计与「全自动实盘（分阶段）」的上线流程。
> 目标：在享受全自动选股交易的同时，**默认零风险、武装需显式确认、随时可一键熔断**。

---

## 1. 它解决什么问题

现有 `executor.py` 已有单笔风控（仓位上限、止损止盈、涨跌停/T+1、AI 评分过滤），
但冲着**实盘**还缺三块地基：

| 缺口 | 风险 | 本中枢的补法 |
|------|------|--------------|
| 无武装状态机 | 重启后可能直接实盘 | `OFF→PAPER→LIVE_PENDING→LIVE` 显式切换，**默认 PAPER，持久化** |
| 无人工确认闸门 | 系统自主下真实资金单 | `LIVE_PENDING` 每笔委托 `预览→审批→执行` |
| 无组合级熔断/日志 | 回撤失控、无审计 | 峰谷回撤熔断 + 单日亏损熔断 + append-only 成交日志 |

它是**防御纵深的最外层**：复用 `TradeExecutor` 做逐单风控，再叠加状态机、确认闸门、
组合级熔断。两层都拦，才算通过。

---

## 2. 架构

```
                     ┌─────────────────────────────────────┐
   信号/UI ───────►  │          RiskSupervisor (中枢)         │
                     │  ├─ SafetyController  武装状态机+KillSwitch
                     │  ├─ OrderGate        人工确认闸门
                     │  ├─ PositionSizer    高级仓位管理
                     │  └─ TradeJournal     结构化成交日志
                     └──────────────┬──────────────────────┘
                                    │ 复用内层风控
                                    ▼
                        TradeExecutor (逐单: 仓位/止损/涨跌停/T+1/评分)
                                    │
                          BrokerAPI / PaperAccount
```

四大组件：
- **SafetyController**：四级状态机 + 全局 Kill Switch，状态落盘 `data/trading_guard_state.json`。
- **OrderGate**：委托预览/审批工作流；`PAPER` 自动批准，`LIVE_PENDING` 必须人工 `approve()`。
- **PositionSizer**：`fixed_fractional` / `volatility_target` / `kelly`（半 Kelly 封顶）三种仓位算法。
- **TradeJournal**：append-only JSONL 日志，事件含 `signal/proposal/approve/fill/halt/drawdown_breach` 等，供合规审计。

---

## 3. 四阶段上线路径（务必按顺序）

### 阶段 0 · PAPER（默认，零风险）
- 状态文件为空时默认 `paper`。所有委托走 `PaperAccount`，不碰真实资金。
- 用途：验证信号→筛选→仓位→止损闭环。**先在这里跑足量交易日。**

### 阶段 1 · 模拟盘对接（SIM）
- 接券商 SIM 环境（如可用），确认行情/账户/下单链路通畅。仍不落真实资金。

### 阶段 2 · LIVE_PENDING（逐单确认）← 首次实盘从这里开始
```python
sup = build_supervisor(broker, config_path="config/trading_guard.yaml")
token = sup.safety.arm_live("operator_alice")      # ① 取令牌
assert sup.safety.confirm_arm(token, "alice")      # ② 回传令牌 → 进入 LIVE_PENDING
# 此后每笔实盘委托：
prop = sup.propose_trade("600000", "浦发", OrderSide.BUY, ai_score=72, agreement=0.9)
# 人工在 UI/终端核对 prop 后：
sup.gate.approve(prop.proposal_id, "alice")
ok, msg, order = sup.approve_and_execute(prop.proposal_id, "alice")
```
- 每笔单都需**人工核对预览 → 审批**；超时（默认 300s）自动撤销。
- 错误的令牌、空令牌都会被拒绝，且**保留令牌可重试**。

### 阶段 3 · LIVE（全自动，最后一步）
- 仅在阶段 2 跑稳、且你确认要放手时：再次 `arm_live` + `confirm_arm` 切到 `LIVE`，
  此后系统自主下单（仍需 `require_live_confirmation` 配置与熔断兜底）。
- 建议长期保留 `LIVE_PENDING` + 人工抽检，而非全放开。

---

## 4. Kill Switch（任何时候都能用）

```python
sup.kill_switch("手动紧急停止", operator="alice")   # 全局冻结，持久化
sup.resume(operator="alice")                        # 解除（曾实盘则强制回 PAPER）
```
- 熔断状态**写入磁盘**，进程重启后依然 halt，**不会自动恢复实盘**。
- `resume()` 若此前处于实盘，会**强制回到 PAPER**，必须重新武装才能再上实盘。
- UI 上应常驻一个「紧急停止」按钮，直接调 `kill_switch`。

---

## 5. 熔断与风控参数（`config/trading_guard.yaml`）

| 参数 | 默认 | 含义 |
|------|------|------|
| `max_portfolio_drawdown_pct` | 8.0 | 峰谷最大回撤 ≥ 此值 → 全局熔断 |
| `max_daily_loss_pct` | 3.0 | 单日亏损比例上限 |
| `max_daily_loss_abs` | 5000 | 单日亏损金额上限（元） |
| `max_positions` | 20 | 最大持仓只数 |
| `max_single_position_pct` | 0.30 | 单票仓位上限（占总资产） |
| `max_sector_pct` | 0.50 | 单一行业集中度上限 |
| `min_order_interval_sec` | 2.0 | 两笔委托最小间隔 |
| `max_orders_per_minute` | 10 | 每分钟委托上限 |
| `sizing_method` | fixed_fractional | 仓位算法 |
| `base_risk_fraction` | 0.02 | 每笔风险敞口占净值比例 |

组合级熔断由 `RiskSupervisor.tick_monitor()` 周期性检查；建议 AutoTrader 主循环每轮调用一次。

---

## 6. 上线前 Checklist

- [ ] 阶段 0 纸交易已跑足量交易日，绩效可接受
- [ ] `config/trading_guard.yaml` 参数已按风险承受力复核（尤其回撤/亏损线）
- [ ] 状态文件路径 `data/trading_guard_state.json` 已确认可写
- [ ] Kill Switch 在 UI/终端均可用，且理解「重启仍保持熔断」
- [ ] 操作人标识（operator）已约定，审计日志会记录每次武装/审批
- [ ] 首次实盘仅用 `LIVE_PENDING`，且有人盯盘复核前 N 笔
- [ ] 资金量从**小**开始，验证链路无误再放量

---

## 7. 接入现有 AutoTrader（✅ 已完成）

`RiskSupervisor` 已实际接入 `AutoTrader`（`src/trading/autotrader.py`），默认 `guard=None`
完全保留旧行为；调用 `set_guard(supervisor)` 后，`scan_and_trade()` 内所有买卖执行都改走
安全中枢。核心改动：

```python
class AutoTrader:
    def __init__(self, broker=None, risk_config=None):
        ...
        self.guard = None                 # 默认 None → 旧行为不变

    def set_guard(self, guard) -> None:
        """接入安全中枢。共享同一纸账户引用，由 guard 统一记账。"""
        self.guard = guard
        if self.paper_account is not None and getattr(guard, "paper_account", None) is None:
            guard.paper_account = self.paper_account
        if getattr(guard, "paper_account", None) is not None:
            self.paper_account = guard.paper_account

    def _execute_signal(self, code, name, ai_score, agreement, side, strategy_tag="ai"):
        """统一执行入口：接 guard 走安全中枢，否则走原 executor。"""
        if self.guard is not None:
            prop = self.guard.propose_trade(code=code, name=name, side=side,
                                            ai_score=ai_score, agreement=agreement,
                                            strategy_tag=strategy_tag)
            if prop.status == "blocked":
                return False, prop.reason or "被安全中枢拦截", None
            return self.guard.approve_and_execute(prop.proposal_id, operator="auto(scan)")
        return self.executor.execute_ai_signal(
            code=code, name=name, ai_score=ai_score, agreement=agreement, side=side)
```

`scan_and_trade()` 中还做了：
- 进入扫描即调用 `self.guard.tick_monitor()` 做回撤/单日亏损监控；若 `safety.halted`
  直接 `return`（KILL SWITCH 冻结一切交易）。
- 两处 `execute_ai_signal` 已替换为 `_execute_signal`。
- 当 `guard` 接入时，跳过原 `self.paper_account.buy/sell`，改由 guard 统一记账（共享引用，不重复）。
- `get_status()` / `print_status()` 新增 `guard` 状态（mode / 是否熔断 / 是否需逐单确认）。

最小化启动示例：

```python
from src.trading.autotrader import AutoTrader
from src.trading.guard import build_supervisor
from config.settings import ROOT_DIR

at = AutoTrader(broker=...)                 # 传入你的 broker（纸/实盘）
at.use_builtin_predictor()                  # 或 set_ai_predictor(...)
at.set_guard(build_supervisor(
    at.broker,
    config_path=str(ROOT_DIR / "config" / "trading_guard.yaml"),
    paper_account=at.paper_account,
))
at.start(interval_seconds=60)               # 全自动扫描循环（默认 PAPER 模式，零实盘风险）
```

> 接入后默认仍是 `PAPER` 模式，**不会**自动下单到真实券商。要上实盘，需两步令牌武装
> （`supervisor.safety.arm_live(operator)` → `confirm_arm(token)`），且建议首阶段仅用
> `LIVE_PENDING`（每笔人工确认）。

---

## 8. 验证

- 单元测试：`pytest tests/test_trading_guard.py`（**16 项全绿**，含 4 项 AutoTrader↔Guard 接入集成测试，
  覆盖：接入后走安全中枢/纸账户、Kill Switch 熔断拦截扫描、默认不接入行为不变、状态暴露）
- Dry-run 演示：`python scripts/guard_demo.py`（零资金跑通 纸交易→武装→逐单确认→Kill Switch 全流程）


> ⚠️ 免责声明：本系统仅供研究与学习，不构成投资建议。实盘有风险，
> 任何自动化交易都应在充分回测、小资金验证、且有人盯盘的前提下谨慎使用。

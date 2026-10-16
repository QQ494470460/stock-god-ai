# 自动交易大改 — 2026-07-10

## 四件全部完成

### 1. 独立的纸交易模拟引擎
- `src/trading/paper.py`：`PaperAccount` 类，独立于券商
- 功能：虚拟账户(默认10万本金)、买卖成交、费用（佣金万2.5+印花税千1）、持仓管理、盈亏追踪
- 持久化：`data/paper_account.json` + `data/paper_trade_history.json`
- 绩效统计：总交易/胜率/盈亏比/最大回撤/夏普/累计收益

### 2. 信号流向bug修复
- `autotrader.py` `use_builtin_predictor(None)` 自动从 `watchlist.json` 加载
- K线缓存：同一股票多路调用共享一份 DataFrame，不再重复拉取
- `_scan_buy_candidates` 已兼容 watchlist.json 格式

### 3. 实时交易日志面板
- `autotrader.py` 新增 `on_event` 回调 + `_emit_event()`
- 事件类型：scan_start/scan_done/trade/scan_blocked/check_positions/buy_candidates
- UI Tab5 "实时日志"：统计概览 + 最近扫描摘要 + 交易列表 + 完整事件流
- 调用链：`_make_event_callback()` → `st.session_state.trade_event_log` (deque, max 200)

### 4. 策略绩效追踪
- `PaperAccount.get_performance_stats()` 计算所有指标
- UI Tab3 "纸交易模拟"：账户总览 + 持仓明细 + 绩效统计 + 交易记录
- 纸交易与 AutoTrader 联动：启动纸交易自动建仓（通过 AutoTrader.paper_account）

### UI变化
- trading_ui.py：7个Tab（新增纸交易模拟 + 实时日志，量化分析/订单记录重构）
- "启动纸交易"按钮替代原来的券商连接依赖

### 修改文件
- `src/trading/autotrader.py`：事件系统 + 纸交易集成 + watchlist自动加载 + K线缓存
- `src/ui/trading_ui.py`：完全重写（30906B）
- `src/trading/paper.py`：新建（13589B）

### 验证
- PaperAccount buy/sell/pnl/performance 全链路通过
- Streamlit 运行中：端口8501

# 量化自动交易 Bug 修复 - 2026-07-07

## 用户反馈
"量化自动交易没有用" — 自动交易模块名存实亡。

## 根因：3 个连锁 Bug

### Bug ①（致命）：autotrader._scan_buy_candidates() 读错 JSON key
- `autotrader.py:188` 读 `cfg.get("stocks", [])`
- `watchlist.json` 实际 key 是 `"watchlist"`
- → 永远返回空列表，**永不产生任何买入信号**

**修复：** 兼容双格式 `cfg.get("stocks") or cfg.get("watchlist", [])`，同时兼容 `"000001"` 字符串和 `{"code":"000001"}` 字典两种列表元素格式。

### Bug ②（致命）：模拟模式下 broker 创建后没调 connect()
- `trading_ui.py` 模拟 Tab 创建 `YuekaiBroker()` 后直接使用
- YuekaiBroker 的 `connect()` 才设置 `_connected=True`
- → `connected=False` → `query_positions()` 返回 `[]`，`query_account()` 返回 None
- → 所有风控检查、下单都失败

**修复：** 模拟 Tab 创建 broker 后立即调用 `connect()` (模拟模式也设 connected=True + 100万虚拟账户)

### Bug ③：UI 传入空 watchlist
- `_render_simulate()` 的 AI 信号扫描按钮传入 `wl=[]`
- `_render_ai_trading()` 的启动按钮传入 `trader.use_builtin_predictor([])`
- → 日志显示 "监控 0 只股票"

**修复：** 改为从 `config/watchlist.json` 加载 15 只自选股传入。

## 验证结果
```json
>>> Broker connected=True, account: ¥1,000,000 (模拟)
>>> Manual order: sim_86b574cb, filled, 000001 100股@10.00
>>> AI scan: 15 stocks → found 3 buy signals (601012/002594/300750)
>>> 非交易时段风控正确拦截 (正确行为)
```

## 修改文件
- `src/trading/autotrader.py:188,198` — 兼容 watchlist/stocks 双 key + 字符串/字典格式
- `src/ui/trading_ui.py:165` — 模拟 broker 创建后调 connect()
- `src/ui/trading_ui.py:180` — 手动下单复用已连接的 broker
- `src/ui/trading_ui.py:193-195` — AI信号扫描从 watchlist.json 加载自选股
- `src/ui/trading_ui.py:257` — AI自动交易启动从 watchlist.json 加载自选股

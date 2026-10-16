# AI 智能选股优化 — 2026-07-04

## 修改范围

### 1. `src/features/engine.py` — 新增 5 类指标 + 市场状态
- `_calc_adx_features()`: ADX 趋势强度、+DI/-DI 方向指标、DI 交叉
- `_calc_atr_features()`: ATR 波动率、ATR%、ATR 变化率
- `_calc_volume_price_trend()`: VPT 量价趋势、VPT 发散度
- `_calc_money_flow()`: MFI 资金流量指数、超买超卖标记
- `_calc_market_regime()`: 牛/熊/震荡三态检测 (close vs MA60, MA20 斜率)
- 全部集成至 `build_features()`，特征数 91→111

### 2. `src/models/predictor.py` — 动态权重 + 风险调整评分
- 新增 `model_weights` 字典，从硬编码 0.5/0.5 改为可动态调整
- `set_model_weights(xgb, lgb)`: 手动设权重
- `calibrate_weights(stock_data)`: 基于近期概率优势自动校准
- **风险调整评分公式**:
  - sharpe_like = (score - 50) / (volatility × 100)
  - 最终评分 = 基础分×70% + (基础分 + sharpe_like×10)×30%
  - 牛市看涨 +5%，熊市看涨 -10%
  - 始终 clamp [0, 100]
- 关键修复: features→scaler 列对齐（新 111 特征 → scaler 90 列），传入带列名的 DataFrame 消除 sklearn 警告

### 3. `src/ui/app.py` — 并行扫描 + 板块加成
- AI选股页改用 `fetcher.get_batch_daily_kline(codes, max_workers=8)` 多线程并行拉数据
- 板块热度加成：热点板块股票 +5 分，按 (评分+板块加成) 重排序
- 技术形态预筛选保留，在批量拉取后执行

## 性能对比

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 20只股扫描耗时 | ~60-90s (串行) | ~7s (8线程并行) |
| 特征数量 | 91 列 | 111 列 (+20 新指标) |
| 评分维度 | 概率加权 | 概率+风险+市场状态+板块 |
| 模型权重 | 固定 0.5/0.5 | 可动态校准 |
| sklearn 警告 | 2/股 | 0 |

## 向后兼容

- 新特征通过 reindex 对齐旧的 90 列 scaler
- 新指标不参与模型 predict，仅影响最终评分 (score_raw 来自模型概率，sharpe-like/regime 在评分环节加入)
- 重新训练后 scaler 会自动学习全部 111 列特征

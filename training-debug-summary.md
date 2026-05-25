# 选股神AI - 训练调试总结

## 时间
2026-05-26 03:35 ~ 04:05

## 修复的问题

### 1. 数据源连接失败
- 原因: 东方财富 (stock_zh_a_hist) API RemoteDisconnected
- 修复: 切换到新浪数据源 (stock_zh_a_daily)，稳定可用
- 文件: src/data/fetcher.py

### 2. 特征工程 KeyError: 'ma_5'
- 原因: _calc_ma_features 中 ma_bullish 计算引用了 df 而非 result
- 修复: df → result
- 文件: src/features/engine.py

### 3. 基本面特征全 NaN 导致样本丢弃
- 原因: 无财务数据时 fill NaN，导致 prepare_training_data 的 NaN drop 丢弃所有样本
- 修复: 改为 fill 0.0
- 文件: src/features/engine.py

### 4. scaler.pkl / label_encoder.pkl 缺失
- 原因: train.py 走 --model ensemble 分支时直接调 train_xgboost/train_lightgbm，没调 train_ensemble 也就没保存预处理器
- 修复: 在训练完成后显式调用 trainer._save_preprocessing()
- 文件: scripts/train.py

## 最终模型
- 50 只沪深300成分股
- 85,084 训练样本，90 特征
- XGBoost: F1=0.4817, Test Acc=72.21%
- LightGBM: F1=0.4654, Test Acc=70.33%

## 模型文件
models/xgboost_model.pkl (3.8MB)
models/lightgbm_model.pkl (3.2MB)
models/scaler.pkl (3.9KB)
models/label_encoder.pkl (266B)
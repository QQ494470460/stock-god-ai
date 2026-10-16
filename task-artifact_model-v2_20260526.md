# 选股神 v2 模型训练总结 — 2026-05-26 05:00

## 改动

### 1. 二分类模式（config/settings.py）
- 新增 `binary_mode: True`：去掉 "震荡" 类，只做涨 vs 跌
- 样本更纯净，F1 从 0.48 直升 0.71

### 2. 超参数优化（config/settings.py）
- n_estimators: 300→500
- learning_rate: 0.05→0.03
- subsample/colsample: 0.8→0.75
- 更多正则化（reg_alpha↑, reg_lambda↑）
- LightGBM 新增 num_leaves=63

### 3. 超参数搜索（src/models/trainer.py）
- 新增 `grid_search()` 方法 → RandomizedSearchCV
- 训练脚本支持 `--tune` 参数

### 4. 数据量扩充（scripts/train.py）
- max_stocks 默认值: 100→200

## 训练结果

| 指标 | v1 (3-class, 50stocks) | v2 (binary, 200stocks) | 变化 |
|------|----------------------|----------------------|------|
| XGBoost F1 | 0.48 | 0.68 | +42% |
| LightGBM F1 | 0.48 | **0.71** | +48% |
| 样本数 | 24,834 | 186,214 | 7.5x |
| 特征维度 | 90 | 90 | - |
| LSTM | 未训 | 已完成 | 新 |

## 模型文件
- models/xgboost_model.pkl (1.3MB)
- models/lightgbm_model.pkl (2.8MB)
- models/lstm_best.pt (2.7MB)
- models/scaler.pkl (3.9KB)
- models/label_encoder.pkl (258B)

## 待优化
- CV Fold 5 始终很低 (~0.40)，可能数据有结构性断裂
- F1 0.71 仍中等，可考虑更多特征（财务/情绪）
- Git push 因网络问题未完成，本地已 commit
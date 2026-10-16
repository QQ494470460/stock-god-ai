# 预测修复（二分类/三分类兼容） — 2026-05-26 05:28

## 根因
模型 v2 训练使用 `binary_mode=True`（二分类，label_encoder classes = [-1, 1]），但 `predict_single()` 硬编码了三分类逻辑（总是期望 classes = [-1, 0, 1]），导致：
- `list(classes).index(0)` → ValueError（0 不在 [-1, 1] 中）
- 回退索引 `ensemble_proba[0][2]` → IndexError（proba 只有 2 列）

## 修复
`src/models/predictor.py` 第 130-145 行：根据 `len(classes)` 自动切换二分类/三分类概率映射路径。
- 二分类：涨=proba[:,1], 跌=proba[:,0], 震荡=0
- 三分类：保持原有逻辑

## 验证
5 只沪深300股票预测成功，1 只推荐返回（≥50分阈值），无异常。
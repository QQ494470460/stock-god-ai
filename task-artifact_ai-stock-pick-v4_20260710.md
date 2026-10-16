# AI 选股优化 — 2026-07-10

## 改动

### 1. 二分类 → 三分类
- `config/settings.py`: `binary_mode: True → False`
- 模型现在学习涨(+1)/震荡(0)/跌(-1)，保留震荡信息
- 需重新训练才生效

### 2. 评分公式重写
- 新建 `src/models/scoring.py`: `compute_score()` + `get_score_interpretation()`
- 旧版：7个系数连环乘，无经济学意义
- 新版：`50 + 预期收益×1000 + 确信度(0~12) + 一致加分(-5~8) + 风险惩罚(-10~0)` × 牛熊因子(0.88~1.08)
- 每项有明确上限，不会互相放大
- 推荐理由从 raw feature 值改为人话："两模型一致看涨, 预期收益+1.5%, 高确信度"
- `predictor.py` 已接入新评分

### 3. 预测追踪
- 新建 `src/models/tracker.py`: `save_predictions()` + `verify_predictions()`
- 每次 AI 选股扫描自动保存到 `data/prediction_history.json`
- N 日后可自动验证准确率
- `app.py` 已接入（扫描完成后自动保存）

### 4. SHAP 修复
- `app.py`: Optuna 快速训练移除 `--no-shap`，改为 `--shap-top 60`
- SHAP 特征筛选终于会真正运行

### 待完成
- [ ] 设置页加历史准确率图表
- [ ] 重新训练模型以启用三分类

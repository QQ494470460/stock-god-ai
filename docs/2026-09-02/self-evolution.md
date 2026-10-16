# 自我进化功能（2026-09-02）

## 需求
用户要求「AI选股神有自我进化功能，根据市场进化」。

## 设计思路
系统已有完整反馈闭环：AI选股 → `tracker.py` 记录预测快照 → N日后 `verify_predictions` 回测验证。
在此基础上新增**自我进化引擎**，把验证结果转化为可自动调整的参数。

## 实现（4 个进化维度）

### 1. 模型权重进化
- XGBoost vs LightGBM 谁近期方向判断更准，就加重谁。
- 用加权准确率（accuracy × 样本数）计算目标权重，50% 学习率平滑移动。
- 约束：单模型权重 [0.2, 0.8]。
- 需 ≥10 个模型信号样本才触发。

### 2. 推荐阈值进化
- 准确率 ≥65% → 阈值 -2（放宽，让更多标的入选）
- 准确率 ≤45% → 阈值 +3（收紧，只留高评分）
- 中等 → 轻微回调
- 约束：调整幅度 [-10, +15]，叠加市场状态调整。

### 3. 市场状态适配
- 用沪深300（000300.SH）MA20/MA60 + MA20斜率自动检测牛/熊/震荡。
- 牛市：阈值 -3（激进）；熊市：阈值 +5（保守）；震荡：不变。

### 4. 分窗口性能追踪
- T+5 / T+10 两个回测窗口独立记录准确率。

## 文件改动

### 新建 `src/models/evolution.py`（9.7KB）
`EvolutionEngine` 类：
- `evolve_from_verification(verification)` — 从验证结果学习，返回动作列表
- `_evolve_model_weights()` / `_evolve_threshold()` — 两个进化器
- `detect_regime(benchmark_df)` — 市场状态检测
- `get_effective_min_score(user_min_score)` — 进化后有效阈值
- `get_model_weights()` / `get_summary()` / `reset()` — 查询与重置
- 状态持久化到 `data/evolution_state.json`

### 修改 `src/models/tracker.py`
- `verify_predictions` 新增分模型准确率统计（`xgb_accuracy`/`lgb_accuracy`/`xgb_checked`/`lgb_checked`）
- 每个预测详情记录 `xgb_signal`/`lgb_signal`

### 修改 `src/models/predictor.py`
- 新增 `_get_evolved_weights()` — 惰性导入进化引擎权重，显著偏离 0.5 时采用
- `predict_single` 集成概率改用进化权重

### 修改 `src/ui/app.py`
- import `evolution`
- AI选股页扫描前：检测市场状态 + 计算进化后有效阈值（显示 caption）
- `get_top_recommendations` 用 `effective_min_score`（用户设定 + 阈值调整 + 市场调整）
- 回测验证按钮：验证完成后触发 `evolution.evolve_from_verification()`，展示进化动作
- `save_predictions` 增加 `xgb_signal`/`lgb_signal` 字段
- 设置页：新增「🧬 自我进化引擎」面板（权重/阈值/市场状态/T+5/T+10准确率/进化历史/重置按钮）

## 验证结果
- 4 个文件语法检查通过
- 进化引擎导入正常
- 权重进化验证：XGB 62% vs LGB 45% → 权重 0.5/0.5 → 0.54/0.46 ✅
- predictor 正确读取进化权重 ✅
- Streamlit 重启成功（HTTP 200，PID 13912）

## 使用方式
1. 正常使用 AI选股（自动记录预测快照）
2. 5-10 天后回到 AI选股页，点「🔄 运行回测验证」
3. 系统自动验证 + 触发进化，展示进化动作
4. 后续扫描自动应用进化后的权重和阈值
5. ⚙️ 设置页可查看进化状态、历史、或重置

## 待用户确认/后续方向
- 目前是"在线进化"（仅调整集成权重和阈值，不重训模型）
- 更激进的"离线进化"：累积足够验证数据后，自动用近期数据增量重训模型（数据漂移适应）

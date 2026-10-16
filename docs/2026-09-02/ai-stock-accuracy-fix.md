# AI选股"不准"根因诊断与修复（2026-09-02）

## 用户反馈
"AI选股不太准"（2026-09-02 14:39）

## 诊断过程
1. 查看预测历史 `data/prediction_history.json`，发现几乎所有记录的 `lgb_signal: 0`（LightGBM 永远预测"震荡"），而 XGB 多为 +1（看涨）。
2. 检查模型类别：XGBoost 是三分类 `[0,1,2]`（经 label_encoder 映射 -1/0/1），LightGBM 是**二分类** `[0,1]`。
3. 复现 `predict_single` 的 LGB 信号提取逻辑，确认了核心 bug。

## 根因（核心 bug）

### Bug 1：LightGBM 信号映射错误（严重）
`predictor.py` 中 LGB 预测用**三分类**的 `label_encoder.inverse_transform()` 映射**二分类**模型的输出：
```python
lgb_signal = int(self.label_encoder.inverse_transform([lgb_pred])[0])  # 错误
```
- LGB 二分类 `classes_=[0,1]`：0=跌，1=涨
- 当 LGB 预测"涨"(1) → 被映射成 label_encoder 的 1 → 实际是"震荡"类 ❌
- 当 LGB 预测"跌"(0) → 被映射成 label_encoder 的 0 → "跌"(-1)（纯巧合）

**后果**：LightGBM 模型形同虚设，一致性判断全错，进化引擎分模型评估失效，推荐列表"一致"列误导。

### Bug 2：XGB 三分类极度保守（次生问题）
XGB 三分类模型（forecast_days=5, threshold_up=3%, threshold_down=-2%）对几乎所有股票都预测"震荡"类（概率 0.42~0.55），导致 XGB 信号多为 0。

## 修复内容

### `src/models/predictor.py`（1 处）
LGB 信号提取改为按模型类别数分支处理：
```python
lgb_classes = list(self.lgb_model.classes_)
if len(lgb_classes) == 2:
    # 二分类：class 0 = 跌(-1)，class 1 = 涨(+1)
    lgb_signal = -1 if lgb_pred == lgb_classes[0] else 1
else:
    lgb_signal = int(self.label_encoder.inverse_transform([lgb_pred])[0])
```

## 验证
- 5 只股票实测，LGB 信号从错误的全 `0` 变为正确的 `+1`（看涨）。
- XGB 仍偏保守（三分类震荡类主导），但这是模型本身的阈值设置问题（threshold_up=3% 偏严），非代码 bug。

## 结论
核心 bug（LGB 信号映射）已修复。XGB 保守是标签阈值偏严所致，后续可通过降低 threshold_up / 增加样本多样性改善。

## 后续建议（可选）
1. 重新训练 LightGBM 为三分类，与 XGB 对齐（当前 LGB 是旧二分类模型）
2. 调低 threshold_up（3%→2%），让三分类模型更敢于给出方向信号
3. 用回测系统验证修复后准确率（需累积 5 天数据）

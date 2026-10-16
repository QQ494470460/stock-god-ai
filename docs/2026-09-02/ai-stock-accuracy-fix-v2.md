# AI选股"不准"根因修复 + 模型重训（2026-09-02）

## 用户反馈
"AI选股不太准"，选方案 3 = 两者都做（重训 LGB 为三分类 + 放宽标签阈值）。

## 诊断出的三个问题

### 问题 1：LightGBM 信号映射错误（核心 bug，已修复）
LightGBM 是**二分类**模型（classes_=`[0,1]`），但代码用**三分类**的 `label_encoder.inverse_transform()` 映射它的预测：
- LGB 预测"涨"(1) → 被错误映射成"震荡"(0) ❌
- LGB 预测"跌"(0) → 被映射成"跌"(-1)（巧合）

**修复**：`predictor.py` 按模型类别数分支——二分类直接 0→跌(-1)/1→涨(+1)，三分类才走 encoder。

### 问题 2：同花顺数据源只返回 140 行（训练崩溃根因，已修复）
同花顺 `d.10jqka.com.cn/v6/line/.../last.js` 接口只返回**最近 ~140 个交易日**（约 7 个月），忽略 start_date 参数。导致：
- MA60/MA120 等长周期指标无法计算（57 列 NaN）
- `prepare_training_data` 强制所有列非 NaN → 0 条样本 → `pd.concat` 崩溃

**修复**：`fetcher.py` 新增 `_fetch_from_sina()`（AKShare `stock_zh_a_daily`），能拿**完整历史**（2018 至今 2103 行），真实复权价，稳定不拒连（1.2s/只）。新浪源为主，同花顺兜底。

验证：单只股票 000001 产出 **1859 条训练样本**（之前 0），标签分布合理（震荡848/涨517/跌494）。

### 问题 3：标签阈值偏严（已调）
XGB 三分类模型对几乎所有股票预测"震荡"（概率 0.42~0.55 徘徊），因为 threshold_up=3% 太严。
**修复**：`settings.py` threshold_up 3% → 2%，让模型更敢给方向信号。

## 重训方案
- 两个模型都重训（`--model all`），LightGBM 也训成**三分类**与 XGB 对齐
- 用新浪完整历史（2018-01-01 起）+ 放宽后 2% 阈值
- 配置：Optuna 100 trials / 3折 / 3轮 / hs300+zz500 400只 / shap-top 60 / max_samples 60000

## 执行状态
- 已备份旧模型到 `models/backup_20260902_144903/`
- 训练进程 PID 24864 后台运行中（XGB Optuna 搜索 8/100 trials，每 trial ~6s）
- 已设 cron（每 3 分钟）检查训练完成，完成后自动热加载 + 重启 Streamlit
- Streamlit 8501 已重启（新数据源生效）

## 关键文件改动
- `src/models/predictor.py`：LGB 信号二分类分支映射
- `src/data/fetcher.py`：新增 `_fetch_from_sina()`，新浪源为主、同花顺兜底
- `config/settings.py`：threshold_up 3%→2%

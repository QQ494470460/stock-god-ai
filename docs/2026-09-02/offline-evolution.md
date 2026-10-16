# 离线进化（自动重训）功能（2026-09-02）

## 需求
用户确认要「离线进化」——累积足够验证数据后，自动用近期数据增量重训模型，适应市场风格漂移。

## 与在线进化的区别
| 维度 | 在线进化（已完成） | 离线进化（本次） |
|------|------------------|-----------------|
| 调整对象 | 集成权重、推荐阈值、市场适配 | 模型本体（重训） |
| 数据范围 | 无（只用验证结果） | 近期数据（默认2年） |
| 触发时机 | 每次回测验证 | 近期准确率持续偏低 |
| 成本 | 极低 | 较高（后台重训10-40分钟） |

## 实现

### 1. `evolution.py` 扩展（v1→v2 状态）
- 状态新增 `offline_evolution` 配置块：
  - `enabled` / `retrain_running` / `last_retrain_at`
  - `min_interval_days=7` / `accuracy_threshold=0.45` / `min_batches=3` / `recent_window=3` / `lookback_years=2`
- 新增方法：
  - `should_retrain()` — 判断是否触发（5 条件全满足）
  - `mark_retrain_started()/finished()/failed()` — 重训状态管理
  - `set_offline_config()` — 配置更新
- `get_summary()` 返回 `offline_evolution`

### 2. `scripts/auto_retrain.py`（新建 5KB）
- `--check` 只检查不执行；`--force` 无视条件；默认走 should_retrain
- 满足条件时调用 `train_optuna.py` 快速模式（hs300_zz500 300只、20 trials、3折、2轮重训、SHAP Top60、近期2年数据）
- 成功后：标记完成 + 模型权重重置 0.5/0.5（新模型需重新评估）
- 超时保护 1 小时

### 3. `app.py` UI
- 回测验证后：增加 `should_retrain()` 判断提示
- 设置页「自我进化引擎」新增「🔄 离线进化」区：
  - 启用开关 toggle
  - 5 个参数输入（最少批数/阈值/间隔/窗口/年限）
  - 保存配置按钮
  - 当前状态 + 触发条件检查
  - 「立即离线重训」手动触发按钮（后台 Popen）
  - 「标记重训完成」手动兜底按钮

## 验证结果
- 3 文件语法通过
- should_retrain 逻辑全链路测试通过：
  - 默认未启用 → False ✅
  - 低准确率(0.40/0.38/0.42)+启用 → True ✅
  - 运行中 → False ✅
  - 刚重训完(0天) → False ✅
- auto_retrain.py --check 正常返回"离线进化未启用"
- Streamlit 重启成功（HTTP 200，PID 2196）

## 使用方式
1. 设置页开启「启用离线自动重训」+ 调整参数（默认即可）
2. 正常使用 + 定期回测验证
3. 当近期准确率持续 < 45% 时，系统提示并可自动后台重训
4. 也可手动点「立即离线重训」
5. 重训完成后手动「重载模型」生效

## 后续可选
- 用 cron/心跳自动定期触发 `auto_retrain.py --check`（全自动闭环）
- 重训完成后自动热加载模型（当前需手动点重载，避免重训中模型不稳定）

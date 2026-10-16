# Optuna快速训练崩溃修复 + UI僵死修复

## 问题
Optuna快速训练跑到 18/20 trials 后崩溃，UI一直显示"准备中…"不动。

## 根因
1. **`retrain_final` 崩溃**：UI传了 `--no-shap`，`shap_features[mt]` 为 `None`，`len(None)` → `TypeError`
2. **UI不知道进程死了**：日志停止写入但UI无进程存活检测

## 修复

### train_optuna.py
- `retrain_final`: `len(features)` → 先判空再取值
- `optuna_objective`: `n_jobs=-1` → `n_jobs=1`（Windows防死锁）
- `optuna_objective`: 接受 `max_estimators` 参数控制树数量上限，XGB/LGB 均从 `50` 起步而非 `100`
- `gather_data`: 支持 `--max-samples` 降采样

### app.py (UI)
- Optuna快速: `50 trials/200股/不限样本` → `20 trials/100股/3万样本/400树/跳过SHAP`
- UI阶段检测: 识别tqdm `N/M` 格式 + `Traceback` 崩溃检测
- UI日志: 60秒无写入判定为stale（训练异常终止）

## 结果
- Optuna快速预计 **2-3 分钟** 完成
- 崩溃后UI能自动检测并报错

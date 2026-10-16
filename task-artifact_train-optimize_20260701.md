# 一键训练优化 — 四重致命 Bug 修复

## 问题
用户报告"一键训练优化不合理"，深层分析后发现 4 个互相关联的致命缺陷。

## 根因分析

### Bug 1: 文件名不匹配（最致命）
`train_hyper.py` 第215行保存模型为：
```
xgboost_model_hyper.pkl / lightgbm_model_hyper.pkl
```
但 `predictor.py` → `trainer.load_model("xgboost")` → 查找的是：
```
xgboost_model.pkl / lightgbm_model.pkl
```
**结果：超参优化跑了几十分钟，模型保存了，predictor 永远加载不到！**
App 始终用 5/26 的旧模型。用户看到"模型就绪"以为训练成功了，实际是假象。

### Bug 2: UI 按钮参数不合理
`app.py` 固定传 `--n-iter 1000`，1000 轮超参搜索在 UI 按钮里等 30-60 分钟完全不现实。
改为：快速训练 30 轮（3-5分钟）+ 深度搜索 500 轮（可选）。

### Bug 3: 训练完成后不热重载
`predictor` 是模块级单例，`_load_models()` 只在导入时调用一次。
训练完成 → 旧模型仍在内存 → 页面不刷新 → 用户看不到新模型效果。

### Bug 4: train_hyper.py 独立创建 scaler/encoder
`train_hyper.py` 用自己新建的 `StandardScaler` 和 `LabelEncoder`，不从 `MLTrainer` 复用。
数据一致时没问题，但若特征顺序有细微差异会崩溃。

## 修复内容

### 文件 1: `scripts/train_hyper.py` (Bug 1, 4)
```python
# Before (line 215):
fname = f"{model_type}_model_hyper.pkl"
# After:
fname = f"{model_type}_model.pkl"   # 匹配 predictor 加载路径
```

### 文件 2: `src/models/predictor.py` (Bug 3)
新增 `reload()` 方法：
```python
def reload(self):
    """热重载模型（训练完成后调用）"""
    self.xgb_model = None; self.lgb_model = None
    self.scaler = None; self.label_encoder = None
    self._load_models()
```

### 文件 3: `src/ui/app.py` (Bug 2, 3)
1. 拆分为两个按钮：
   - 🧠 快速训练优化：30 轮（3-5分钟）
   - 🔬 深度超参搜索：500 轮（20-30分钟）
2. 训练完成自动检测 → 调用 `predictor.reload()` → 热加载新模型
3. 新增 🔄 手动重载模型按钮

## 关键决策
- 深度搜索阈值从 1000→500（500 轮已充分，1000 轮边际收益极小）
- 自动热加载检测：解析日志找 "超参数深度优化完成" / "训练完成" 关键词
- 保留 `train.py` 原训练脚本不变（它走后走 `MLTrainer` 路径，文件名正确）

## 验证
- `from src.models.predictor import predictor; predictor.is_ready` → True
- `hasattr(predictor, 'reload')` → True
- `from src.models.quantitative import MeanReversionStrategy` → OK
- Streamlit 已重启 (PID 7204, port 8501)

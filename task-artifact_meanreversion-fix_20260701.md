# MeanReversionStrategy NameError 修复

## 问题
量化策略 - 均值回归回测报错：
```
NameError: name 'MeanReversionStrategy' is not defined
Traceback: app.py line 1698
```

## 根因
`app.py` 第22行只 import 了实例变量，遗漏了类名：
```python
from src.models.quantitative import quant_engine, mean_reversion, risk_parity
#                                   ^^^^^^^^^^^^^ 实例
#                                   缺少 MeanReversionStrategy 类名
```
第1698行直接构造 `MeanReversionStrategy(lookback=..., entry_threshold=...)`，找不到类定义。

## 修复
在 import 行补上类名：
```python
from src.models.quantitative import quant_engine, mean_reversion, risk_parity, MeanReversionStrategy
```

## 影响范围
仅 `src/ui/app.py` 第22行一处修改。Streamlit 已重启（PID 12036, port 8501）。

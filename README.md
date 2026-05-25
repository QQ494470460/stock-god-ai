# 选股神 - Stock God AI
> A股智能选股系统 | AI驱动的多因子量化选股模型

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/AI-Machine_Learning-orange.svg)]()

---

## 📈 简介

**选股神**是一套基于AI/机器学习的A股智能选股系统，综合运用多种机器学习算法，从技术面、基本面、市场面多维度对全市场股票进行量化评分和趋势预测。

### 🎯 核心能力

| 能力 | 说明 |
|------|------|
| 🤖 **多模型集成** | XGBoost + LightGBM + LSTM 深度学习 |
| 📊 **多因子特征** | 技术指标 + 基本面 + 市场面，近百个特征维度 |
| 🎯 **趋势预测** | 预测未来N日涨跌概率，输出买入/持有/卖出信号 |
| 📈 **智能评分** | 综合多模型输出，对全市场股票打分排序 |
| 🌐 **Web UI** | Streamlit 交互式看板，可视化选股结果 |
| 🔌 **REST API** | FastAPI 接口，方便对接其他系统 |
| ⏰ **定时推荐** | 每日收盘后自动更新，推送精选股票池 |

### 🧠 ML 模型架构

```
原始行情数据
    │
    ├──> 技术面特征 (MA/MACD/RSI/KDJ/BOLL/量价...)
    │         │
    ├──> 基本面特征 (PE/PB/ROE/营收增速/利润增速...)
    │         │
    ├──> 市场面特征 (换手率/振幅/资金流向...)
    │         │
    ▼
  特征工程 (标准化/缺失值/时间序列对齐)
    │
    ├──> XGBoost 分类器 ──────┐
    ├──> LightGBM 分类器 ─────┤
    └──> LSTM 序列预测 ──────┘
              │
              ▼
         集成投票/加权评分
              │
              ▼
        股票评分 & 排序
              │
              ▼
     每日精选 TOP-N 推荐
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- Windows / Linux / macOS

### 2. 安装

```bash
git clone https://github.com/yourname/stock-god-ai.git
cd stock-god-ai

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt

# （可选）Windows上安装ta-lib
# 1. 下载wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# 2. pip install TA_Lib-xxx.whl
```

### 3. 训练模型

```bash
# 完整训练流程：获取数据 → 特征工程 → 训练模型
python scripts/train.py

# 仅更新数据
python scripts/train.py --update-data-only

# 指定股票池（沪深300）
python scripts/train.py --pool hs300

# 使用GPU训练LSTM
python scripts/train.py --gpu
```

### 4. 运行选股

```bash
# 一键选股：扫描全市场，输出TOP推荐
python scripts/daily_recommend.py

# 指定模型
python scripts/daily_recommend.py --model xgboost_lgb_ensemble

# 输出到CSV
python scripts/daily_recommend.py --output recommendations.csv
```

### 5. 启动Web UI

```bash
# Streamlit 看板
streamlit run src/ui/app.py

# FastAPI 服务
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📊 API 接口

启动FastAPI后，访问 `http://localhost:8000/docs` 查看Swagger文档。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/recommend` | 获取今日推荐股票 |
| GET | `/api/v1/stock/{code}` | 单只股票分析 |
| POST | `/api/v1/predict` | 批量预测 |
| GET | `/api/v1/hotspots` | 热门概念板块 |
| GET | `/api/v1/model/info` | 模型信息 |

---

## 🗂️ 项目结构

```
stock-god-ai/
├── config/
│   └── settings.py          # 全局配置
├── src/
│   ├── data/
│   │   └── fetcher.py       # 数据获取（AKShare/Tushare）
│   ├── features/
│   │   └── engine.py        # 特征工程
│   ├── models/
│   │   ├── trainer.py       # 模型训练
│   │   ├── predictor.py     # 预测推理
│   │   └── lstm.py          # LSTM深度学习模型
│   ├── api/
│   │   └── app.py           # FastAPI服务
│   └── ui/
│       └── app.py           # Streamlit看板
├── scripts/
│   ├── train.py             # 训练脚本
│   ├── predict.py           # 预测脚本
│   └── daily_recommend.py   # 每日推荐
├── data/                    # 数据缓存目录
├── models/                  # 模型文件目录
├── logs/                    # 日志目录
├── requirements.txt
└── README.md
```

---

## 🧪 技术细节

### 特征维度

| 类别 | 数量 | 典型特征 |
|------|------|----------|
| 价格趋势 | 20+ | 收益率、波动率、动量、均线偏离 |
| 均线指标 | 15+ | MA交叉、MA斜率、MA排列 |
| MACD | 8 | DIF、DEA、柱值、金叉死叉 |
| RSI/KDJ | 10 | 超买超卖、背离、交叉 |
| 布林带 | 6 | 宽度、位置、突破 |
| 成交量 | 12 | VR、OBV、资金流、量比 |
| 基本面 | 10+ | PE、PB、ROE、增速 |
| 市场面 | 8 | 换手率、振幅、市值 |

### 训练策略

- **样本构造**：每只股票每日一条样本，标签为未来N日收益率
- **样本平衡**：使用SMOTE过采样解决类别不平衡
- **验证方式**：时间序列Walk-Forward验证，防止未来信息泄露
- **特征筛选**：基于SHAP/LightGBM importance的特征剪枝
- **超参优化**：Optuna贝叶斯优化

---

## ⚠️ 免责声明

本系统仅供学习和研究使用，**不构成任何投资建议**。股市有风险，投资需谨慎。AI模型基于历史数据训练，过去的模式不一定适用于未来。

---

## 📝 License

MIT License
# -*- coding: utf-8 -*-
"""
选股神 - Stock God AI
A股智能选股系统，基于AI/机器学习的多因子选股模型
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import os
import sys
import yaml

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent


def _resolve_dir(env_key: str, default_name: str) -> Path:
    """解析数据目录：优先环境变量，其次 exe 同级目录，最后源码目录

    打包成 exe 后 PyInstaller 会把代码解包到临时目录（sys._MEIPASS），
    那里的数据目录是只读且随退出销毁 —— 必须改指向 exe 同级的外挂目录。
    """
    env_val = os.environ.get(env_key)
    if env_val:
        return Path(env_val)

    # 打包模式：exe 所在目录优先
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        return exe_dir / default_name

    return ROOT_DIR / default_name


DATA_DIR = _resolve_dir("STOCK_GOD_DATA_DIR", "data")
MODEL_DIR = _resolve_dir("STOCK_GOD_MODEL_DIR", "models")
LOG_DIR = _resolve_dir("STOCK_GOD_LOG_DIR", "logs")

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


@dataclass
class MLConfig:
    """机器学习配置"""
    # XGBoost参数
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "min_child_weight": 5,
        "gamma": 0.2,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    })
    
    # LightGBM参数
    lgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.75,"min_child_samples": 30,
        "num_leaves": 63,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
    })
    
    # LSTM参数
    lstm_params: dict = field(default_factory=lambda: {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 100,
        "sequence_length": 60,  # 用过去60个交易日预测
        "patience": 10,  # early stopping
    })
    
    # 训练配置
    test_size: float = 0.2
    val_size: float = 0.1
    cv_folds: int = 5
    
    # 预测标签
    forecast_days: int = 5  # 预测未来N日涨跌
    threshold_up: float = 0.02  # 涨幅阈值（2%以上为涨，放宽让模型更敢给方向）
    threshold_down: float = -0.02  # 跌幅阈值（-2%以下为跌）
    binary_mode: bool = False  # 三分类（涨/震荡/跌），更精准

    # LLM 定性叠加覆盖层（详见 scripts/daily_recommend.py / src/models/predictor.py）
    # 默认关闭：覆盖层仅对 predict_batch 产出的 score 做纯加性微调，
    # 绝不重训 xgboost，不影响已验证的 walk-forward RankIC=0.1256。
    # CLI（--llm / --llm-mock）会临时把 enable_llm_overlay 置 True。
    enable_llm_overlay: bool = False  # 主开关（默认关闭，零影响）
    llm_mock: bool = False            # 覆盖层是否走离线 mock（无需本地 Ollama）

    # 自进化阈值调整开关（详见 src/models/evolution.py）
    # True  = 有效阈值 = 用户设定 + 准确率调整 + 市场状态调整（熊市 +5、牛市 -3）
    #         —— 这个 ±5 是硬编码启发式，无回测验证，且弱模型下准确率调整项长期≈0
    # False = 有效阈值恒等于用户在 UI 拖的滑块值，不被市场状态静默改写（默认）
    auto_threshold_adjust: bool = False


@dataclass  
class DataConfig:
    """数据配置"""
    # 股票池
    stock_pool: str = "all"  # all / hs300 / zz500 / customized
    # 数据更新
    update_hour: int = 18
    # 数据范围
    start_date: str = "2015-01-01"
    # 缓存
    cache_days: int = 1  # 缓存天数
    # 限速（请求间隔秒数）
    rate_limit: float = 1.5


@dataclass
class FeatureConfig:
    """特征配置"""
    # 技术指标
    use_technical: bool = True
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60, 120])
    macd_params: tuple = (12, 26, 9)
    rsi_period: int = 14
    kdj_params: tuple = (9, 3, 3)
    boll_period: int = 20
    # 基本面
    use_fundamental: bool = True
    # 市场面
    use_market: bool = True


@dataclass
class AppConfig:
    """应用配置"""
    ml: MLConfig = field(default_factory=MLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Streamlit
    streamlit_port: int = 8501
    # 日志级别
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Path):
        from dataclasses import asdict
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, allow_unicode=True, default_flow_style=False)


# 全局配置实例
config = AppConfig()
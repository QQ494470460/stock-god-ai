# -*- coding: utf-8 -*-
"""
选股神 - Stock God AI
A股智能选股系统，基于AI/机器学习的多因子选股模型
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import yaml

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


@dataclass
class MLConfig:
    """机器学习配置"""
    # XGBoost参数
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    })
    
    # LightGBM参数
    lgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
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
    threshold_up: float = 0.03  # 涨幅阈值（3%以上为涨）
    threshold_down: float = -0.02  # 跌幅阈值（-2%以下为跌）


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
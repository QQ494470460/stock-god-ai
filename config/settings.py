# -*- coding: utf-8 -*-
"""
选股神 - Stock God AI
A股智能选股系统，基于AI/机器学习的多因子选股模型
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
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
# 配置目录同样外挂：打包后账号/密钥落在 exe 同级 config/，重建 exe 不丢账号
CONFIG_DIR = _resolve_dir("STOCK_GOD_CONFIG_DIR", "config")

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
try:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:  # noqa: BLE001
    pass


def _seed_external_config() -> None:
    """首次运行时，把包内置的配置模板拷到外挂 CONFIG_DIR。

    打包后 CONFIG_DIR = exe 同级 config/（初始为空），而 templates 在
    _internal/config/ 里。没有这一步，用户就找不到 trading_guard.yaml，
    也就没法在不重新打包的前提下调风控参数。
    只补「缺失的文件」，绝不覆盖用户已改过的配置。
    """
    try:
        src_dir = ROOT_DIR / "config"
        if src_dir.resolve() == CONFIG_DIR.resolve():
            return  # 源码运行：模板目录与外挂目录本就是同一个，无需拷贝
        for name in ("trading_guard.yaml", "watchlist.json"):
            src = src_dir / name
            dst = CONFIG_DIR / name
            if src.exists() and not dst.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:  # noqa: BLE001  配置模板缺失不致命
        pass


_seed_external_config()


def _extend_sys_path_for_external_libs() -> None:
    """把「外挂库目录」加入 sys.path，用于加载刻意不打进 exe 的第三方包。

    当前用途：**xtquant**（券商 miniQMT 的原生交易库）。
    它自带 .pyd/DLL，收进 PyInstaller 包会在隔离子进程里触发访问冲突
    导致打包失败；且它必须配合本机 QMT 客户端使用，打包意义不大。
    因此改为外挂：用户把 xtquant 文件夹放到以下任一位置即可启用实盘：

        <exe 同级>/libs/xtquant/...
        <exe 同级>/config/libs/xtquant/...

    必须在任何 `import xtquant` 之前执行 —— 本模块被所有入口最先导入，
    所以放在这里最稳妥。
    """
    import sys as _sys
    try:
        if getattr(_sys, "frozen", False):
            base = Path(_sys.executable).parent
        else:
            base = ROOT_DIR
        for cand in (base / "libs", CONFIG_DIR / "libs"):
            if cand.is_dir():
                p = str(cand)
                if p not in _sys.path:
                    _sys.path.insert(0, p)
    except Exception:  # noqa: BLE001  外挂库缺失不致命
        pass


_extend_sys_path_for_external_libs()


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
    # True  = 有效阈值 = 固定基线 + 进化准确率调整 + 市场状态调整（熊市 +5、牛市 -3）
    #         进化引擎会依据回测验证准确率自动收紧/放宽阈值（需先攒够验证样本）。
    # False = 有效阈值恒等于用户在 UI 拖的滑块值，不被市场状态静默改写。
    auto_threshold_adjust: bool = True

    # 自我进化「模型重训上线」钉住的模型版本（注册表里的 ModelInfo.key）。
    # 留空 = 由 src/models/model_registry.py 按样本外 RankIC 自动挑选最优。
    # 非空时注册表优先加载该版本；若它不可部署则**自动回退**到自动挑选，
    # 绝不因为一个失效的版本 id 让系统无模型可用。
    # 该键由自我进化的重训生成器提出、经样本外闸门 + 人工批准后才写入覆盖层。
    active_version: str = ""

    # 超级集成开关：以 XGBoost 为主模型，另加 Alpha(生产+北向/融资/龙虎榜/行业)
    # 与 LSTM 两条腿加权融合（predict_batch 会走这条路）。
    #
    # 默认 **True**（已开启，用户明确要求）：三条腿是**独立信号**（实测 600519：
    # 主线 46.4 / Alpha 53.2 / LSTM 43.8），多模型平均可显著削减单模型预测方差，
    # 是提升每日 IC 稳定性（日频 IR）的正路；截面中性化已实测对该模型≈无提升，
    # 故方差削减改走集成平均。想退回单模型：设为 False 即可（行为与改动前一致）。
    # 前提：两条腿的产物都还在 models/ 下（alpha_xgboost_model.pkl / lstm_best.pt），
    # 且 torch 可用；任一腿缺失由 EnsemblePredictor 自动降级，绝不拖垮主模型。
    use_ensemble: bool = True
    # 各腿融合权重（默认按样本外 RankIC 自动归一化；此处作可选覆盖）
    ensemble_weights: dict = field(default_factory=lambda: {
        "xgboost": 1.0, "alpha_xgboost": 1.0, "lstm": 0.8,
    })
    # 预测方差收缩（0~1）：把子模型权重向「等权」收缩，降低对单腿过置信权重的方差。
    # 0 = 用原始权重；1 = 全部等权（最大分散、融合方差最小）；主模型权重恒为 1.0 不收缩。
    # 这是叠加在集成平均之上的第二层方差削减（集成平均削的是模型间误差方差，
    # 这里削的是权重估计带来的方差）。
    ensemble_shrinkage: float = 0.5

    # 横截面行业中性化（默认开启）：在 predict_batch 内对全市场评分做「行业去均值」，
    # 消除「模型在某行业系统性偏高/偏低」造成的排名偏差，降低单日 IC 的板块噪声，
    # 提升每日 IC 稳定性（Grinold-Kahn IR = mean/std）。不改变组内排序，只剔行业水位。
    # 无行业映射（fetcher.get_industry_map 取不到）时自动跳过，零影响。
    cross_sectional_neutralize: bool = True


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
class SchedulerConfig:
    """收盘后自动化调度配置（APScheduler）

    默认全部关闭，需显式 enable。与交易安全中枢（guard）解耦——
    调度只负责「数据更新 + 每日推荐生成 + 进化验证」，不直接下单。
    """
    enabled: bool = False                       # 总开关（默认关，避免意外启动定时任务）
    data_update_cron: str = "30 18 * * mon-fri" # 收盘后更新数据（工作日 18:30）
    daily_pick_cron: str = "45 18 * * mon-fri"  # 生成每日推荐（工作日 18:45）
    evolution_cron: str = "0 19 * * mon-fri"    # 回测验证 + 触发进化（工作日 19:00）
    rolling_ic_cron: str = "15 19 * * mon-fri"  # 每日 RankIC 回填（工作日 19:15，收盘后补当日点）
    # 行业映射补齐：东财在本机是**间歇性**可达（限流时会连着几小时 ProxyError），
    # 而映射一旦拿到就永久落盘、之后不再打网络 —— 所以值得每小时给一次机会。
    # 全 A 只要翻几页列表接口，网络不通时秒级失败，不会拖住其它作业。
    industry_map_cron: str = "0 * * * *"        # 每小时整点
    timezone: str = "Asia/Shanghai"
    # 每日推荐产物
    report_dir: str = "data/daily_reports"
    max_picks: int = 20
    pick_pool: str = "zz500"
    # 多策略复核：量化三策略（多因子/均值回归/融合）对 AI 候选投票后
    # 「集体看空则剔除、一致看多则加权」。默认关，需显式开启；
    # 开关持久化在 config/scheduler_state.json（与 enabled 一起）。
    pick_ensemble: bool = False
    # 推送（可选，默认关闭）
    push_enabled: bool = False
    push_webhook_url: str = ""                  # 留空则不推送
    # 调度自身的状态/日志
    log_file: str = "logs/scheduler.log"


@dataclass
class TradingLoopConfig:
    """闭环全自动交易引擎配置（详见 src/trading/loop.py）

    闭环 = 盘前出计划 → 盘中执行（熔断/止损/调仓）→ 收盘结算 → 推日报。
    安全默认：自动调仓默认开启，但一切下单仍经 guard；
    真跑实盘前建议先 dry_run=True 演练，或把模式留在 PAPER。
    """
    scan_interval_sec: float = 30.0      # 主循环周期（秒）
    auto_build_plan: bool = True         # 自动生成/刷新交易计划
    max_positions: int = 10              # 目标持仓上限
    cash_reserve_pct: float = 0.10       # 现金保留比例
    weight_cap: float = 0.30             # 单票权重上限
    # equal / score / risk_parity / min_variance / vol_target
    # 默认风险平价：等权与按评分加权都只看 alpha，不看波动与相关性，
    # 会让组合风险集中在少数高波动标的上。数据不足时自动退回等权。
    weighting: str = "risk_parity"
    tilt_strength: float = 0.25          # 风险权重上叠加 alpha 倾斜（0=纯风险平价）
    enable_rebalance: bool = True        # 目标组合调仓开关
    enable_stop_orders: bool = True      # 止盈止损开关
    # 调仓容忍带（触发换仓的最小权重偏离）。2026-10-27 由 0.25 降至 0.15：
    # 依据 = 真·样本外面板（data/evolution/panel_oos，4 折逐折重训，1140 天，
    # RankIC 0.061 / IC_IR 12.26 / TopN 超额 0.38% / 胜率 61.5%）上的参数扫描。
    # 0.15 在滑点 10/30/50bp 三档成本下均「夏普不输 + 超额不输」5/5 块，
    # 均夏普 0.96~1.00 vs 基准 0.76~0.78；0.10 更激进（均夏普 1.68）但换手 22%/期，
    # 50bp 下退化为 4/5 不输 —— 成本敏感，故取 0.15 不取 0.10。
    # ⚠️ 该参数改的是「交易强度」，越激进越会吃进样本内幻觉，
    # 任何进一步下调都必须重跑 scripts/sweep_grid.py --panel-dir data/evolution/panel_oos。
    band_pct: float = 0.15
    min_trade_amount: float = 5000.0     # 最小交易金额（元）
    # 持有纪律阈值（0=关闭）。持仓不在目标组合但浮盈≥该比例时不强制清仓，
    # 改由止损/止盈托管。实测「每日换票必卖赢家」是盈亏比 0.15 的根因之一，
    # 调高前须经 walk-forward 验证，否则退回「卖赢家、留输家」。
    hold_if_profitable_pct: float = 0.0
    max_orders_per_tick: int = 10        # 单周期最大下单数
    settle_hour: int = 15                # 收盘结算时刻（时）
    settle_minute: int = 5               # 收盘结算时刻（分）
    notify: bool = True                  # 事件/日报推送
    dry_run: bool = False                # True=只预演不下单


@dataclass
class NotifierConfig:
    """推送通知配置（调度产物推送到企微机器人 / 通用 Webhook / 仅日志）

    默认关闭（enabled=False）。channel 取值：
      - log    ：仅写日志/本地文件（默认，零外部依赖）
      - webhook：向 webhook_url 发 HTTP POST（JSON：{text, markdown, payload}）
      - wecom  ：企业微信机器人，向 qyapi.weixin.qq.com 拼接 wecom_key 发送
    """
    enabled: bool = False
    channel: str = "log"            # log / webhook / wecom
    webhook_url: str = ""
    wecom_key: str = ""
    timeout: float = 5.0


@dataclass
class AgentConfig:
    """AI 助手（agent）配置

    - screen_fetch_workers：智能筛选/研报工具拉取全池行情的并发数。
      默认 1（串行）—— 并行拉取在部分环境会触发 py_mini_racer/V8 原生崩溃
      （Check failed: !IsConfigurablePoolInitialized），直接杀死进程、拖垮看板，
      且不可被 try/except 捕获。在确认并行安全的机器上可调大以提速（上限 16）。
    """
    screen_fetch_workers: int = 1


@dataclass
class AppConfig:
    """应用配置"""
    agent: AgentConfig = field(default_factory=AgentConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trading_loop: TradingLoopConfig = field(default_factory=TradingLoopConfig)
    notifier: NotifierConfig = field(default_factory=NotifierConfig)
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


# --- 收盘后自动化调度开关的持久化 ----------------------------------------
# 没有这一步，scheduler.enabled 永远停在代码默认值 False，看板里「启动调度」
# 按钮会永远在 enabled 守卫处返回 None，显示「未启动」。这里只负责「记住用户
# 上次是否勾选启用」，不碰任何交易/风控参数。
_SCHEDULER_STATE_FILE = CONFIG_DIR / "scheduler_state.json"


def load_scheduler_state() -> None:
    """启动时把持久化的调度开关载入 config.scheduler.enabled / pick_ensemble。"""
    try:
        p = _SCHEDULER_STATE_FILE
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if "enabled" in data:
                    config.scheduler.enabled = bool(data["enabled"])
                if "pick_ensemble" in data:
                    config.scheduler.pick_ensemble = bool(data["pick_ensemble"])
    except Exception:  # noqa: BLE001  持久化文件损坏不应拖垮启动
        pass


def save_scheduler_state(enabled: bool) -> None:
    """持久化调度开关。

    注意：这里是**整份覆盖写**，必须把 pick_ensemble 一起带上。否则用户
    在别处改了复核开关、再点一次「启动调度」，复核就被悄悄关掉了。
    """
    try:
        p = _SCHEDULER_STATE_FILE
        p.write_text(
            json.dumps({"enabled": bool(enabled),
                        "pick_ensemble": bool(config.scheduler.pick_ensemble)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass


def save_pick_ensemble_state(on: bool) -> None:
    """单独持久化「多策略复核」开关（不触碰调度 enabled）。

    必须先 load_scheduler_state()，否则会把 enabled 写成代码默认值 False，
    用户下次打开看板会发现「调度怎么自己关了」。
    """
    try:
        p = _SCHEDULER_STATE_FILE
        cur = {"enabled": bool(config.scheduler.enabled)}
        try:
            if p.exists():
                old = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(old, dict) and "enabled" in old:
                    cur["enabled"] = bool(old["enabled"])
        except Exception:  # noqa: BLE001
            pass
        cur["pick_ensemble"] = bool(on)
        p.write_text(json.dumps(cur, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        config.scheduler.pick_ensemble = bool(on)
    except Exception:  # noqa: BLE001
        pass


# --- 超级集成开关的持久化 --------------------------------------------------
# 同理：ml.use_ensemble 若只能改源码，看板就等于「看得见、摸不着」。
# 这里只记住「用户上次是否选择启用超级集成」，不改任何模型权重/阈值。
_ML_STATE_FILE = CONFIG_DIR / "ml_state.json"


def load_ml_state() -> None:
    """启动时把持久化的超级集成开关载入 config.ml.use_ensemble。"""
    try:
        p = _ML_STATE_FILE
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "use_ensemble" in data:
                config.ml.use_ensemble = bool(data["use_ensemble"])
    except Exception:  # noqa: BLE001  持久化文件损坏不应拖垮启动
        pass


def save_ml_state(use_ensemble: bool) -> None:
    """持久化超级集成开关（仅 use_ensemble 这一个布尔值）。"""
    try:
        p = _ML_STATE_FILE
        p.write_text(
            json.dumps({"use_ensemble": bool(use_ensemble)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass


# --- AI 助手筛选并发数的持久化 --------------------------------------------
_AGENT_STATE_FILE = CONFIG_DIR / "agent_state.json"


def load_agent_state() -> None:
    """启动时把持久化的 AI 助手筛选并发数载入 config.agent.screen_fetch_workers。"""
    try:
        p = _AGENT_STATE_FILE
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "screen_fetch_workers" in data:
                v = int(data["screen_fetch_workers"])
                config.agent.screen_fetch_workers = max(1, min(16, v))
    except Exception:  # noqa: BLE001  持久化文件损坏不应拖垮启动
        pass


def save_agent_state(screen_fetch_workers: int) -> None:
    """持久化 AI 助手筛选并发数（仅 screen_fetch_workers 这一个整数）。"""
    try:
        p = _AGENT_STATE_FILE
        p.write_text(
            json.dumps({"screen_fetch_workers": int(screen_fetch_workers)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass


# --- 策略/选股逻辑覆盖层（供 AI 助手「修改策略」类动作持久化）-----------
# 仅允许一组白名单数值键（min_score / rsi_max / top / vol_ratio_min），
# 写入 config/strategy_overrides.json；screen_stocks / make_report 在未显式传参时读取。
_STRATEGY_OVERRIDE_FILE = CONFIG_DIR / "strategy_overrides.json"
_STRATEGY_OVERRIDES: Dict[str, float] = {}
_STRATEGY_OVERRIDE_KEYS = ("min_score", "rsi_max", "rsi_min", "top", "vol_ratio_min")


def load_strategy_overrides() -> None:
    """启动时载入策略覆盖层（白名单数值键）。"""
    try:
        p = _STRATEGY_OVERRIDE_FILE
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in _STRATEGY_OVERRIDE_KEYS:
                        try:
                            _STRATEGY_OVERRIDES[k] = float(v)
                        except (TypeError, ValueError):
                            pass
    except Exception:  # noqa: BLE001
        pass


def get_strategy_override(key: str, default):
    """读取策略覆盖值，未设置时回退 default。"""
    if key in _STRATEGY_OVERRIDE_KEYS and key in _STRATEGY_OVERRIDES:
        return _STRATEGY_OVERRIDES[key]
    return default


def save_strategy_override(key: str, value) -> float:
    """写回单个策略覆盖键（夹到合理范围），返回实际写入值。"""
    if key not in _STRATEGY_OVERRIDE_KEYS:
        raise KeyError(f"不允许的策略键：{key}")
    v = float(value)
    if key == "min_score":
        v = max(0.0, min(100.0, v))
    elif key in ("rsi_max", "rsi_min"):
        v = max(0.0, min(100.0, v))
    elif key == "top":
        v = max(1, min(200, int(v)))
    elif key == "vol_ratio_min":
        v = max(0.0, min(20.0, v))
    _STRATEGY_OVERRIDES[key] = v
    try:
        _STRATEGY_OVERRIDE_FILE.write_text(
            json.dumps(_STRATEGY_OVERRIDES, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass
    return v


load_scheduler_state()
load_ml_state()
load_agent_state()
load_strategy_overrides()


def load_evolution_overrides(cfg: "AppConfig" = None) -> list:
    """把「自我进化」写下的运行期覆盖应用到配置上。

    背景：本模块的 ``config = AppConfig()`` 是硬编码默认值，没有外部加载，
    所以 `trading_loop.*` 这类参数以前只能改代码。自我进化需要一个
    运行期可写的覆盖层，落在外挂 `CONFIG_DIR/evolution_overrides.json`。

    这里**刻意只用标准库 + 内联校验规则**，不 import `src.evolution`，
    以免与 `src/evolution/overrides.py`（它要 import 本模块）形成循环导入。

    任何非法/越界/风控相关的键一律跳过，绝不因此让配置加载失败。
    """
    target = cfg if cfg is not None else config
    path = Path(CONFIG_DIR) / "evolution_overrides.json"
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        overrides = (data or {}).get("overrides") or {}
        if not isinstance(overrides, dict):
            return []
    except Exception:  # noqa: BLE001  覆盖层坏了不该拖垮启动
        return []

    # 风控硬线黑名单：即使有人手改 JSON 也拒绝生效
    banned_leaves = {
        "max_portfolio_drawdown_pct", "max_daily_loss_pct", "max_daily_loss_abs",
        "max_single_position_pct", "max_orders_per_minute", "max_orders_per_tick",
        "base_risk_fraction", "kelly_cap", "min_ai_score_buy", "min_ai_score_sell",
        "max_ai_score_sell", "require_live_confirmation",
        "require_per_order_confirmation", "default_mode",
    }
    banned_prefixes = ("guard.", "trading_guard.", "risk.", "safety.", "executor.")

    applied = []
    for key, value in overrides.items():
        if not isinstance(key, str) or key.startswith("evolution."):
            continue                       # evolution.* 走 data/evolution_state.json
        leaf = key.rsplit(".", 1)[-1]
        if leaf in banned_leaves or any(key.startswith(p) for p in banned_prefixes):
            continue
        node = target
        try:
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node, dict) else getattr(node, part)
            if isinstance(node, dict):
                node[parts[-1]] = value
            else:
                cur = getattr(node, parts[-1], None)
                if isinstance(cur, bool):
                    value = bool(value)
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    value = int(value)
                elif isinstance(cur, float):
                    value = float(value)
                elif cur is not None and not isinstance(value, type(cur)):
                    continue
                setattr(node, parts[-1], value)
            applied.append(key)
        except Exception:  # noqa: BLE001
            continue
    return applied


load_evolution_overrides()

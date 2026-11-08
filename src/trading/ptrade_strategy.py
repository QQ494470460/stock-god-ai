# -*- coding: utf-8 -*-
"""
选股神AI · 恒生 Ptrade 策略适配器
================================

把「选股神AI」的 **信号 + 组合层风控** 逻辑包成恒生 Ptrade 的策略框架，
上传到券商 Ptrade 终端后，由引擎在**服务端**调用 ``initialize`` / ``handle_data``。

⚠️ 关键约束（Ptrade 服务端沙箱，必须守）
----------------------------------------
- 策略代码运行在**券商机房**，不能访问外部网络、不能 import 本项目本地 ``src/``、
  不能读你本机文件。因此本文件**完全自包含**：信号与风控逻辑全部内联。
- 下单用 Ptrade 全局函数 ``order()`` / ``order_target()``；行情用 ``get_history()`` /
  ``get_snapshot()``。**不依赖**本地 ``predictor`` / ``guard`` 重型对象。
- 这些 API 是**引擎以全局形式注入**的（恒生标准形态，无法 import）。为同时满足
  (a) 运行时拿到真实函数、(b) 静态检查（pyflakes 门禁）看到名字已定义、(c) 绝不覆盖
  引擎全局 —— 本文件在模块顶层用 ``globals().get(...)`` 捕获它们（见下方 ENGINE_API）。
- 实盘开关在 **Ptrade 终端**（模拟/实盘切换），**不在代码里**。任何部署先模拟跑通。

代码分层（便于离线单测）
------------------------
- 纯函数层（不引用任何 Ptrade 全局，可离线 pytest）：
  ``compute_heuristic_signal`` / ``select_targets`` / ``_round_lot`` /
  ``portfolio_drawdown_pct`` / ``is_rebalance_day``
- Ptrade 适配层（仅在引擎内被调用，引用 ENGINE_API 中的全局）：
  ``initialize`` / ``handle_data`` / ``run_strategy`` / ``monitor_risk`` /
  回调 ``on_order_response`` / ``on_trade_response``

风控参数对齐生产默认值（记忆 §5：组合层 ON、个股层 OFF）：
  max_positions=8, 单票≤30%, 组合回撤≤20%, 单日亏损≤5%,
  现金预留10%, 15天再平衡, band 15%, hold_rank_slack 3.0, 等权。
"""

from __future__ import annotations

# ============================================================================
# 一、可调参数（对齐生产默认值；按你的券商/风险偏好改这里即可）
# ============================================================================
MAX_POSITIONS = 8                  # 最大持仓数
MAX_SINGLE_POSITION_PCT = 0.30     # 单票权重上限（组合层）
MAX_SECTOR_PCT = 0.50              # 行业集中度上限（见 SECTOR_MAP，留空则跳过）
MAX_PORTFOLIO_DRAWDOWN_PCT = 20.0  # 组合最大回撤，超过则熔断（只平不开）
MAX_DAILY_LOSS_PCT = 5.0           # 单日最大亏损，超过则当日停止开新仓
CASH_RESERVE_PCT = 0.10            # 现金预留比例
REBALANCE_DAYS = 15                # 硬性再平衡周期（交易日）
BAND_PCT = 0.15                    # 目标权重变动超过该比例才交易（避免频繁调仓）
HOLD_RANK_SLACK = 3.0              # 持仓股排名在阈值+slack 内则保留（不必强行换出）
WEIGHTING = "equal"                # equal / risk_parity（risk_parity 需波动率，简化为等权）

# === PTrade auto-generated config START
# 股票池：由「生产持仓 + 最新计划候选」生成（Ptrade 格式 '600570.SS' / '000001.SZ'）。
# 这是截至生成时刻的候选池；计划每日更新，重跑生成器即可刷新。
UNIVERSE = [
    "603688.SS", "688120.SS", "301377.SZ", "300857.SZ",
    "300475.SZ", "300285.SZ", "300548.SZ", "300570.SZ",
    "301200.SZ", "688313.SS", "601869.SS", "002851.SZ",
    "001389.SZ", "002281.SZ", "603175.SS", "002008.SZ",
    "688629.SS", "603899.SS", "002120.SZ", "688702.SS",
    "003031.SZ", "002409.SZ", "603529.SS", "603083.SS",
    "603658.SS", "688114.SS", "001309.SZ", "688052.SS",
    "605589.SS", "600105.SS", "603156.SS", "688172.SS",
    "300620.SZ", "688002.SS", "603786.SS", "688200.SS",
    "600378.SS", "300390.SZ", "300395.SZ", "300972.SZ",
    "688065.SS",
]

# 行业映射：键为 6 位代码（去后缀），值为行业名；缺映射的票归 "未知"（集中度对其 fail-open）。
SECTOR_MAP = {
    "603688": "非金属材料Ⅱ",
    "688120": "半导体",
    "301377": "通用设备",
    "300857": "消费电子",
    "300475": "其他电子Ⅱ",
    "300285": "电子化学品Ⅱ",
    "300548": "通信设备",
    "300570": "通信设备",
    "301200": "专用设备",
    "688313": "通信设备",
    "601869": "通信设备",
    "002851": "其他电源设备Ⅱ",
    "001389": "元件",
    "002281": "通信设备",
    "603175": "元件",
    "002008": "未知",
    "688629": "军工电子Ⅱ",
    "603899": "文娱用品",
    "002120": "物流",
    "688702": "半导体",
    "003031": "通信设备",
    "002409": "电子化学品Ⅱ",
    "603529": "未知",
    "603083": "通信设备",
    "603658": "医疗器械",
    "688114": "医疗器械",
    "001309": "半导体",
    "688052": "半导体",
    "605589": "未知",
    "600105": "通信设备",
    "603156": "未知",
    "688172": "未知",
    "300620": "通信设备",
    "688002": "军工电子Ⅱ",
    "603786": "未知",
    "688200": "半导体",
    "600378": "未知",
    "300390": "未知",
    "300395": "航空装备Ⅱ",
    "300972": "未知",
    "688065": "化学制品",
}
# === PTrade auto-generated config END ===

# 模型挂载（可选、best-effort）：把训练好的模型 pickle 上传到策略目录后填路径。
# 加载失败自动回退到内置启发式信号，保证策略永远能跑。
MODEL_PATH = ""   # 例如 "model.pkl"


# ============================================================================
# 二、引擎 API 捕获（恒生 Ptrade 以全局注入，无法 import）
# ============================================================================
# 在模块顶层从 globals() 读取引擎注入的全局函数。运行时 globals() 已由引擎预置这些
# 名字 → 拿到真实函数；离线/CI 下为 None（纯函数测试不调用它们，故无影响）。
# 用 globals().get 而非赋值占位，绝不会覆盖引擎全局。
order = globals().get("order")
get_history = globals().get("get_history")
set_universe = globals().get("set_universe")
run_daily = globals().get("run_daily")
get_snapshot = globals().get("get_snapshot")


class _Log:
    """离线/未注入时的日志兜底（引擎注入 log 时自动改用引擎版）。"""
    @staticmethod
    def info(m):
        print(f"[ptrade] {m}")
    @staticmethod
    def warning(m):
        print(f"[ptrade][WARN] {m}")
    @staticmethod
    def error(m):
        print(f"[ptrade][ERR] {m}")


log = globals().get("log") or _Log


def _g():
    """取引擎注入的全局 g（策略状态字典）；离线时为临时 dict。"""
    return globals().get("g") or {}


# ============================================================================
# 三、纯函数层（不引用任何 Ptrade 全局，可离线单测）
# ============================================================================
def _round_lot(qty: float, lot: int = 100) -> int:
    """A股下单数量必须取整百股（1 手=100 股）。"""
    q = int(qty)
    return max(0, (q // lot) * lot)


def portfolio_drawdown_pct(peak: float, current: float) -> float:
    """组合回撤百分比（peak→current，正数表示亏损幅度）。"""
    if peak <= 0:
        return 0.0
    return (peak - current) / peak * 100.0


def is_rebalance_day(day_counter: int, rebalance_days: int = REBALANCE_DAYS) -> bool:
    """按交易日计数判定是否到了硬性再平衡日。"""
    if rebalance_days <= 0:
        return True
    return day_counter % rebalance_days == 0


def compute_heuristic_signal(closes, volumes=None) -> float:
    """内置轻量信号：动量(5/20日均线) + 均值回复(z-score) 的混合打分，范围 0~100。

    Ptrade 服务端无本地模型时作为兜底信号；有模型时见 ``load_model`` 覆盖。
    设计：上行动量给正分，但过度拉伸(z过大)回拉，避免追高接飞刀。
    """
    import numpy as np

    c = np.asarray(list(closes), dtype=float)
    if c.size < 20:
        return 50.0
    ma5 = float(c[-5:].mean())
    ma20 = float(c[-20:].mean())
    mom = (ma5 / ma20 - 1.0) if ma20 > 0 else 0.0
    std = float(c[-20:].std()) + 1e-9
    z = (float(c[-1]) - ma20) / std
    score = 50.0 + 30.0 * float(np.tanh(mom * 5.0)) - 15.0 * float(np.tanh(z))
    return float(min(100.0, max(0.0, score)))


def select_targets(scores: dict, held: set, params: dict) -> dict:
    """根据打分选出目标权重（等权），返回 {code: weight(0~1)}。

    - 先取打分前 ``max_positions`` 名为初选；
    - 持仓股若排名在 ``hold_rank_slack`` 内则保留（不必强行换出）；
    - 可投权重 = (1 - 现金预留) / 选中数，单票封顶 ``max_single_position_pct``。
    """
    max_pos = int(params.get("max_positions", MAX_POSITIONS))
    slack = int(params.get("hold_rank_slack", HOLD_RANK_SLACK))
    cash = float(params.get("cash_reserve_pct", CASH_RESERVE_PCT))
    cap = float(params.get("max_single_position_pct", MAX_SINGLE_POSITION_PCT))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return {}

    selected = [c for c, _ in ranked[:max_pos]]
    if held:
        idx = {c: i for i, (c, _) in enumerate(ranked)}
        cutoff = max_pos - 1 + slack
        for c in held:
            if c in selected:
                continue
            if idx.get(c, 1 << 30) <= cutoff and scores.get(c, 0) > 0:
                selected.append(c)
    selected = selected[: max_pos + slack]

    n_sel = len(selected) or 1
    w = min((1.0 - cash) / n_sel, cap)
    return {c: w for c in selected}


# ============================================================================
# 四、可选模型挂载（best-effort，失败回退启发式）
# ============================================================================
_MODEL = None
_MODEL_LOADED = False


def load_model():
    """尝试加载上传到策略目录的模型 pickle；失败返回 None（用启发式信号）。"""
    global _MODEL, _MODEL_LOADED
    if _MODEL_LOADED:
        return _MODEL
    _MODEL_LOADED = True
    if not MODEL_PATH:
        return None
    try:
        import pickle
        with open(MODEL_PATH, "rb") as fh:
            _MODEL = pickle.load(fh)
        return _MODEL
    except Exception:
        _MODEL = None
        return None


def predict_score(model, features) -> float:
    """用模型对特征向量打分（0~100）。模型接口未知时回退启发式。"""
    if model is None:
        return 50.0
    try:
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba([features])[0][-1])
            return min(100.0, max(0.0, p * 100.0))
        if hasattr(model, "predict"):
            return float(model.predict([features])[0])
    except Exception:
        pass
    return 50.0


# ============================================================================
# 五、Ptrade 适配层（仅在引擎内被调用；引用 ENGINE_API 中的全局）
# ============================================================================
def _last_price(stock: str) -> float:
    """取最新价：优先快照，失败回退到 get_history 末值。"""
    if get_snapshot is not None:
        try:
            snap = get_snapshot([stock])
            if snap and stock in snap and snap[stock].get("last_px", 0) > 0:
                return float(snap[stock]["last_px"])
        except Exception:
            pass
    if get_history is not None:
        try:
            df = get_history(1, "1d", "close", stock, fq="pre", include=True)
            if df is not None and len(df) > 0:
                return float(df["close"].iloc[-1])
        except Exception:
            pass
    return 0.0


def _current_positions(context) -> dict:
    """返回 {code(去后缀): shares} 当前持仓（含可用）。"""
    out = {}
    try:
        for stock, pos in context.portfolio.positions.items():
            out[stock] = int(pos.total_amount)
    except Exception:
        pass
    return out


def compute_signals(context, universe) -> dict:
    """对股票池逐个打分 → {code: score(0~100)}。"""
    model = load_model()
    scores = {}
    for stock in universe:
        try:
            df = get_history(60, "1d", ["close", "volume"], stock, fq="pre", include=False)
            if df is None or len(df) < 20:
                continue
            closes = df["close"].tolist()
            volumes = df["volume"].tolist() if "volume" in df else None
            feat = closes[-20:]  # 简化特征：近20日收盘价序列
            s = predict_score(model, feat) if model is not None else compute_heuristic_signal(closes, volumes)
            scores[stock] = s
        except Exception:
            continue
    return scores


def _sector_ok(targets: dict) -> bool:
    """行业集中度检查：任一行业目标权重不超过 MAX_SECTOR_PCT。"""
    if not SECTOR_MAP:
        return True
    agg = {}
    for code in list(targets.keys()):
        sec = SECTOR_MAP.get(code.replace(".SS", "").replace(".SZ", ""), "未知")
        agg[sec] = agg.get(sec, 0.0) + targets[code]
        if agg[sec] > MAX_SECTOR_PCT:
            return False
    return True


def run_strategy(context, data=None):
    """每日 09:40 触发：选股 + 组合层风控 + 下单。"""
    g = _g()
    universe = list(g.get("universe") or UNIVERSE)
    if not universe:
        return

    equity = float(getattr(context.portfolio, "portfolio_value", 0.0) or 0.0)
    if equity <= 0:
        return

    # —— 组合层熔断：回撤 / 单日亏损 ——
    g["peak_equity"] = max(g.get("peak_equity", equity), equity)
    dd = portfolio_drawdown_pct(g["peak_equity"], equity)
    if dd > MAX_PORTFOLIO_DRAWDOWN_PCT:
        g["halt_drawdown"] = True
        log.info(f"🛑 组合回撤 {dd:.1f}% > {MAX_PORTFOLIO_DRAWDOWN_PCT}%，熔断：只平不开")
    day_start = g.get("day_start_equity") or equity
    daily_pnl_pct = (equity - day_start) / (day_start or 1.0) * 100.0
    if daily_pnl_pct < -MAX_DAILY_LOSS_PCT:
        g["halt_dailyloss"] = True
        log.info(f"🛑 单日亏损 {daily_pnl_pct:.1f}% < -{MAX_DAILY_LOSS_PCT}%，当日停止开新仓")

    g["day_counter"] = g.get("day_counter", 0) + 1
    rebalance = is_rebalance_day(g["day_counter"], REBALANCE_DAYS)
    if not rebalance and g.get("targets"):
        # 非再平衡日：仅在持仓股跌出选择/触发止损时处理，平时不频繁调仓
        _handle_exits_only(context, g, universe)
        return

    scores = compute_signals(context, universe)
    if not scores:
        return
    current_pos = _current_positions(context)
    params = {
        "max_positions": MAX_POSITIONS,
        "hold_rank_slack": HOLD_RANK_SLACK,
        "cash_reserve_pct": CASH_RESERVE_PCT,
        "max_single_position_pct": MAX_SINGLE_POSITION_PCT,
    }
    targets = select_targets(scores, set(current_pos.keys()), params)
    if not _sector_ok(targets):
        log.warning("⚠️ 行业集中度超限，本次不调仓")
        return
    g["targets"] = targets

    investable = equity * (1.0 - CASH_RESERVE_PCT)
    for stock, weight in targets.items():
        price = _last_price(stock)
        if price <= 0:
            continue
        target_value = investable * weight
        target_shares = _round_lot(target_value / price)
        cur = current_pos.get(stock, 0)
        delta = target_shares - cur
        if abs(delta) < 100:
            continue
        # band 过滤：变动幅度过小不交易
        if cur > 0 and abs(delta) / cur < BAND_PCT:
            continue
        if delta > 0 and (g.get("halt_drawdown") or g.get("halt_dailyloss")):
            continue  # 熔断/单日亏：只平不开
        if order is None:
            log.error("⚠️ order() 不可用（非 Ptrade 引擎环境），跳过实盘下单")
            continue
        try:
            order(stock, delta, limit_price=price)
            log.info(f"🟢 调仓 {stock}: {'买' if delta > 0 else '卖'} {abs(delta)}股 @ {price:.2f}")
        except Exception as e:
            log.error(f"下单失败 {stock}: {e}")


def _handle_exits_only(context, g, universe):
    """非再平衡日：只处理「持仓跌出选择」的清仓，不开新仓。"""
    scores = compute_signals(context, universe)
    current_pos = _current_positions(context)
    if not current_pos:
        return
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    idx = {c: i for i, (c, _) in enumerate(ranked)}
    cutoff = MAX_POSITIONS - 1 + int(HOLD_RANK_SLACK)
    for stock, shares in current_pos.items():
        if shares <= 0:
            continue
        rank = idx.get(stock, 1 << 30)
        if rank > cutoff:  # 跌出保留区 → 清仓
            price = _last_price(stock)
            if price > 0 and order is not None:
                try:
                    order(stock, -shares, limit_price=price)
                    log.info(f"🔴 清仓 {stock}: {shares}股（跌出选择）")
                except Exception as e:
                    log.error(f"清仓失败 {stock}: {e}")


def monitor_risk(context, data=None):
    """盘后 14:50 风险巡检：刷新回撤峰值、记录当日权益。"""
    g = _g()
    equity = float(getattr(context.portfolio, "portfolio_value", 0.0) or 0.0)
    g["peak_equity"] = max(g.get("peak_equity", equity), equity)
    if not g.get("day_start_equity"):
        g["day_start_equity"] = equity


def on_order_response(context, order):
    """委托回报回调（Ptrade 引擎调用）。"""
    log.info(f"📨 委托回报: {order.get('security', '')} 状态={order.get('status', '')}")


def on_trade_response(context, trade):
    """成交回报回调（Ptrade 引擎调用）。"""
    log.info(f"✅ 成交: {trade.get('security', '')} {trade.get('amount', '')}股 @ {trade.get('price', '')}")


def initialize(context):
    """Ptrade 引擎启动时调用一次。"""
    g = _g()
    g["universe"] = list(UNIVERSE)
    g["day_counter"] = 0
    g["peak_equity"] = 0.0
    g["day_start_equity"] = 0.0
    g["targets"] = {}
    g["halt_drawdown"] = False
    g["halt_dailyloss"] = False

    universe = list(UNIVERSE)
    if universe and set_universe is not None:
        set_universe(universe)
    if universe and run_daily is not None:
        run_daily(run_strategy, time="09:40")
        run_daily(monitor_risk, time="14:50")
    log.info(f"🤖 选股神AI · Ptrade 策略已初始化，股票池 {len(universe)} 只")


def handle_data(context, data):
    """每个周期触发（日线每天一次 / 分钟线每周期一次）。

    本策略主逻辑放在 run_daily(run_strategy)，此处仅做轻量日内熔断监控，
    避免在分钟线里高频重算信号。
    """
    g = _g()
    equity = float(getattr(context.portfolio, "portfolio_value", 0.0) or 0.0)
    if g.get("day_start_equity"):
        dd = portfolio_drawdown_pct(g["peak_equity"], equity)
        if dd > MAX_PORTFOLIO_DRAWDOWN_PCT and not g.get("halt_drawdown"):
            g["halt_drawdown"] = True
            log.warning(f"🛑 日内触发组合回撤熔断 {dd:.1f}%")

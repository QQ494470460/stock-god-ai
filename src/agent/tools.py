# -*- coding: utf-8 -*-
"""AI 助手的工具集（取数 / 执行层）。

设计约定
--------
1. **工具只负责拿数据、跑计算、读状态**，不负责"说人话"。自然语言叙述统一交给代理
   的最后一次 LLM 调用完成。原因：
   - 对本地 7B 模型更稳（避免工具内再嵌套一次 LLM 推理）；
   - 一次任务里无论调几个工具，叙述只合成一次，省时。
2. **每个工具对异常自包容**：出错时返回一段可读的错误文本，而不是抛异常。
   这样代理有机会在下一步纠正（例如用户/模型把股票代码写错）。
3. 工具入参统一是 `dict`，返回统一是 `str`（给 LLM 的观测）。
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from src.data.fetcher import fetcher
from src.models.llm import deepseek
from src.models.predictor import predictor
from src.ui.common import (
    build_name_map, is_valid_code, lookup_name,
    calc_rsi, calc_macd, calc_ma,
)
from src.core.daily_pick import generate_daily_picks
from config.settings import (
    config as app_config, DATA_DIR, CONFIG_DIR,
    get_strategy_override, save_strategy_override,
    save_agent_state, save_ml_state,
)

# 名称映射（懒加载一次，避免每次查股票都重扫全表）
_NAME_MAP: Optional[dict] = None


def _name_of(code: str) -> str:
    global _NAME_MAP
    if _NAME_MAP is None:
        try:
            _NAME_MAP = build_name_map(fetcher.get_stock_list())
        except Exception:
            _NAME_MAP = {}
    return lookup_name(_NAME_MAP, code, code)


def _resolve_code(raw: str) -> str:
    """把用户输入的 code/name 归一成 6 位代码；支持直接传代码或常见简称。

    简化实现：若已是 6 位数字则直接返回；否则在名称映射里反查；查不到则返回原串
    （交给下游报错，由代理纠错）。
    """
    if raw is None:
        return ""
    raw = str(raw).strip().upper()
    if is_valid_code(raw):
        return raw
    # 反查：名称 -> 代码
    global _NAME_MAP
    if _NAME_MAP is None:
        try:
            _NAME_MAP = build_name_map(fetcher.get_stock_list())
        except Exception:
            _NAME_MAP = {}
    inv = {v: k for k, v in _NAME_MAP.items()}
    if raw in inv:
        return inv[raw]
    # 前缀匹配（如 "宁德"）
    for name, code in _NAME_MAP.items():
        if raw in name:
            return code
    return raw


# ============================ 修改类动作的「确认闸」 ============================
# 设计：修改类工具在 approve=False 时只返回「预览 + 机器可读标记」，绝不落盘/改文件/下单；
# 引擎检测到标记即暂停循环，把提案交还给 UI；用户在 UI 点「确认执行」后，UI 用 approve=True
# 重新调用同一工具，才真正执行。这样「动手改东西」始终经过人确认，杜绝静默篡改。
# 安全铁律：交易类一律仅 PAPER（纸账户）模拟，绝不自动实盘。
PENDING_ACTION_PREFIX = "PENDING_ACTION:"

# 「工具失败」机器标记。引擎据此判定该观测**不可作为结论依据**。
#
# 为什么非要机器标记：本地 7B 拿到一段明确的报错后，仍会把它当「查到了、结果是空」
# 来用。实测「今天大盘怎么样」触发 `ValueError: 未知股票池 'shanghai'`，模型却答
# 「大盘暂无评分 ≥ 55 的股票符合条件」——凭空捏造。靠提示词一句「别用失败结果」
# 拦不住，必须有可判定的标记。
TOOL_FAILED_MARK = "[[TOOL_FAILED]]"


def _failed(msg: str, hint: str = "") -> str:
    """构造一条「工具失败」观测（带机器标记 + 可选的自纠提示）。"""
    out = f"{TOOL_FAILED_MARK}{msg}"
    if hint:
        out += f"\n💡 {hint}"
    return out


# ============================ 参数归一（防模型瞎填） ============================
# 模型经常把「沪深300」写成 shanghai / csi300 / 300，把「大盘」当成一只股票。
# 这里只做**已知别名**的归一；不认识的仍原样下传，由真实 fetcher 报错
# （保持 fail-loud，也避免把注入的测试池名误判成非法）。

_VALID_POOLS = ("all", "hs300", "zz500", "zz1000", "hs300_zz500")
_POOL_ALIASES = {
    "all": "all", "全市场": "all", "全部": "all", "所有": "all", "全a": "all",
    "a股": "all", "沪深全市场": "all",
    "hs300": "hs300", "沪深300": "hs300", "300": "hs300", "csi300": "hs300",
    "沪深三百": "hs300", "sh300": "hs300", "hs": "hs300",
    "zz500": "zz500", "中证500": "zz500", "500": "zz500", "csi500": "zz500",
    "zz1000": "zz1000", "中证1000": "zz1000", "1000": "zz1000", "csi1000": "zz1000",
    "hs300_zz500": "hs300_zz500", "沪深300中证500": "hs300_zz500",
    "300500": "hs300_zz500", "hs300zz500": "hs300_zz500",
}
POOL_HINT = (
    "合法池名：hs300（沪深300）、zz500（中证500）、zz1000（中证1000）、"
    "hs300_zz500（两者合并）、all（全市场）"
)


def _pool_alias(raw) -> Optional[str]:
    """已知别名 → 合法池名；不认识返回 None（原样下传，交给 fetcher 报错）。"""
    s = str(raw or "").strip()
    if not s:
        return None
    key = s.lower()
    for ch in (" ", "\t", "+", "、", "，", ",", "股票池", "池"):
        key = key.replace(ch, "")
    if key in _VALID_POOLS:
        return key
    return _POOL_ALIASES.get(key)


def _resolve_pool(raw) -> str:
    """工具入参的 pool：先归一，认不出就用原值（让 fetcher fail-loud）。"""
    if raw in (None, ""):
        return str(app_config.scheduler.pick_pool or "hs300")
    return _pool_alias(raw) or str(raw).strip()


# 指数别名：**只对含中文/英文的写法生效**，纯数字一律留给个股（000001 是平安银行）。
# 历史上「大盘怎么样」被解析成 sz000001（平安银行），根因就是没有这层区分。
_INDEX_ALIASES = {
    "上证指数": "sh000001", "上证综指": "sh000001", "上证": "sh000001",
    "大盘": "sh000001", "沪指": "sh000001", "沪市": "sh000001", "sh": "sh000001",
    "深证成指": "sz399001", "深成指": "sz399001", "深证": "sz399001", "深市": "sz399001",
    "创业板指": "sz399006", "创业板": "sz399006", "创指": "sz399006",
    "沪深300": "sh000300", "csi300": "sh000300", "hs300指数": "sh000300",
    "中证500": "sh000905", "csi500": "sh000905",
    "中证1000": "sh000852", "csi1000": "sh000852",
    "科创50": "sh000688", "科创板指": "sh000688",
}
INDEX_HINT = "可用指数：上证指数 / 深证成指 / 创业板指 / 沪深300 / 中证500 / 科创50"


def _index_by_word(raw) -> Optional[str]:
    """只认「文字型」指数别名；纯数字返回 None（纯数字是股票代码）。"""
    s = str(raw or "").strip().lower().replace(" ", "")
    if not s or s.isdigit():
        return None
    return _INDEX_ALIASES.get(s)


def _normalize_index(raw) -> tuple:
    """指数入参 → (symbol, err)。纯数字按指数代码处理（000300 → sh000300）。"""
    s = str(raw or "").strip().lower().replace(" ", "")
    if not s:
        return "sh000001", ""
    if s in _INDEX_ALIASES:
        return _INDEX_ALIASES[s], ""
    if s.isdigit() and len(s) == 6:
        from src.data.fetcher import _normalize_index_symbol
        return _normalize_index_symbol(s), ""
    return "", f"无法识别的指数 {raw!r}。{INDEX_HINT}"

ALLOWED_EDIT_ROOTS = (
    str(Path(__file__).resolve().parents[2] / "src"),
    str(Path(__file__).resolve().parents[2] / "config"),
    str(Path(__file__).resolve().parents[2] / "scripts"),
)


def _propose(tool: str, args: dict, desc: str, impact: str, risk: str = "低") -> str:
    """生成待确认提案：人类可读说明 + 机器可读标记（含批准时所需的完整参数）。"""
    spec = json.dumps(
        {"tool": tool, "args": args, "desc": desc, "impact": impact, "risk": risk},
        ensure_ascii=False,
    )
    return (
        f"⚠️ **待确认的操作（风险：{risk}）**\n\n"
        f"{desc}\n\n"
        f"**影响**：{impact}\n\n"
        f"请确认是否执行。\n\n"
        f"{PENDING_ACTION_PREFIX}{spec}"
    )


def _read_config_current(key: str):
    if key == "agent.screen_fetch_workers":
        return getattr(app_config.agent, "screen_fetch_workers", "?")
    if key == "ml.use_ensemble":
        return getattr(app_config.ml, "use_ensemble", "?")
    return "?"


# ============================ 筛选辅助 ============================

# 批量拉行情时 worker 数的「终极兜底」常量。
# ⚠️ 经验证：并行拉取（max_workers>1）会在某些环境触发 py_mini_racer/V8 的
# 原生崩溃（Check failed: !IsConfigurablePoolInitialized），直接杀死进程、
# 连带拖垮看板。这是不可被 try/except 捕获的原生 abort，唯一稳妥的规避方式是
# 串行拉取。故默认 1 —— 现已可通过「设置页 / config.agent.screen_fetch_workers」调整，
# 仅当配置读不到该字段时才用此常量兜底。
SCREEN_FETCH_WORKERS = 1


def _screen_workers() -> int:
    """AI 助手筛选/研报拉行情的并发数：读配置，异常或越界时回落常量并夹到 [1,16]。"""
    try:
        w = int(app_config.agent.screen_fetch_workers)
    except Exception:  # noqa: BLE001
        w = SCREEN_FETCH_WORKERS
    return max(1, min(16, w))


def _opt_float(v):
    """把参数安全转成 float；None/空/非法都返回 None。"""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _truthy(v) -> bool:
    """把参数安全转成布尔（支持中文/英文多种写法）。"""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "是", "对", "t")


def _technicals(df: "pd.DataFrame") -> dict:
    """从日线计算一组常用技术面指标（价格/RSI/均线/MACD/量比）。"""
    close = df["close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    chg = (last / prev - 1) * 100
    rsi = float(calc_rsi(close).iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    dif, dea, _hist = calc_macd(close)
    dif_v = float(dif.iloc[-1])
    dea_v = float(dea.iloc[-1])
    macd_v = dif_v - dea_v
    avg_vol = float(df["volume"].rolling(20).mean().iloc[-1])
    today_vol = float(df["volume"].iloc[-1])
    vol_ratio = (today_vol / avg_vol) if avg_vol else 0.0
    return {
        "last": last, "chg": chg, "rsi": rsi,
        "ma20": ma20, "ma60": ma60,
        "dif": dif_v, "dea": dea_v, "macd": macd_v,
        "vol_ratio": vol_ratio,
    }


def _verdict(t: dict):
    """基于技术面给出偏多/偏空/超买/超卖研判与一句话理由。"""
    above_ma20 = t["last"] > t["ma20"]
    above_ma60 = t["last"] > t["ma60"]
    macd_pos = t["macd"] > 0
    if t["rsi"] >= 70:
        return ("⚠️ 超买", f"RSI {t['rsi']:.0f} 进入超买区，短期注意回落")
    if t["rsi"] <= 30:
        return ("💡 超卖", f"RSI {t['rsi']:.0f} 进入超卖区，存在反弹需求")
    if above_ma20 and above_ma60 and macd_pos and t["vol_ratio"] >= 1.0:
        return ("🟢 偏多", f"价站 MA20/MA60 上方、MACD 金叉且放量(量比{t['vol_ratio']:.1f}x)")
    if (not above_ma20) and (not above_ma60) and (not macd_pos):
        return ("🔴 偏空", "价跌破 MA20/MA60 且 MACD 转弱，趋势偏弱")
    return ("⚪ 中性", f"多空交织（RSI {t['rsi']:.0f}、量比{t['vol_ratio']:.1f}x、"
                      f"{'站上' if above_ma20 else '跌破'}MA20）")


@dataclass
class Tool:
    name: str
    description: str
    # 给 LLM 看的中文参数说明（用于 prompts）
    params_desc: str
    func: Callable[[dict], str] = lambda d: ""


def _safe(fn: Callable[[dict], str]) -> Callable[[dict], str]:
    def wrapper(args: dict) -> str:
        try:
            return fn(args)
        except Exception as e:  # noqa: BLE001
            return _failed(
                f"⚠️ 工具执行失败：{type(e).__name__}: {e}",
                "该结果不可作为数据依据。" + traceback.format_exc(limit=2),
            )
    return wrapper


# ============================ 具体工具 ============================

@_safe
def analyze_stock(args: dict) -> str:
    """个股深度分析：技术面 + AI 评分 + 推荐理由（取数，不含叙述）。"""
    code = _resolve_code(args.get("code", ""))
    if not is_valid_code(code):
        return _failed(
            f"无法识别股票代码/名称：{args.get('code')!r}。",
            "请提供 6 位 A 股代码（如 600519）或准确的股票名称。",
        )
    if not predictor.is_ready:
        return _failed("预测模型未就绪（未训练），无法给出 AI 评分。", "请先训练模型或检查模型文件。")

    df = fetcher.get_daily_kline(code, start_date="2015-01-01", end_date=None)
    if len(df) < 60:
        return _failed(f"{_name_of(code)}({code}) 历史数据不足（{len(df)} 行），无法分析。",
                       "该标的可能上市不足 60 个交易日。")

    close = df["close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    chg = (last / prev - 1) * 100
    rsi = float(calc_rsi(close).iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    macd = calc_macd(close)
    dif, dea, _hist = macd
    macd_val = float(dif.iloc[-1] - dea.iloc[-1])
    dea_val = float(dea.iloc[-1])

    pred = predictor.predict_single(df, code, _name_of(code))

    # 自带技术研判（让弱模型也能给出明确结论）
    avg_vol20 = float(df["volume"].rolling(20).mean().iloc[-1])
    today_vol = float(df["volume"].iloc[-1])
    vol_ratio = (today_vol / avg_vol20) if avg_vol20 else 0.0
    label, reason = _verdict({
        "last": last, "rsi": rsi, "ma20": ma20, "ma60": ma60,
        "macd": macd_val, "vol_ratio": vol_ratio,
    })

    out = (
        f"## {_name_of(code)}（{code}）\n"
        f"- 最新价：{last:.2f}（当日 {chg:+.2f}%）\n"
        f"- MA20：{ma20:.2f} / MA60：{ma60:.2f}（现价相对 MA20 { (last/ma20-1)*100:+.1f}%）\n"
        f"- RSI(14)：{rsi:.1f}\n"
        f"- MACD：DIF-DEA={macd_val:.3f}，DEA={dea_val:.3f}（DIF 上穿 DEA 为金叉）\n"
        f"- 量比：{vol_ratio:.2f}x\n"
        f"- **技术研判：{label}** —— {reason}\n"
        f"- AI 综合评分：{pred.score:.0f}/100（确信度 {pred.confidence*100:.0f}%）\n"
        f"- 上涨概率 {pred.prob_up*100:.0f}% / 震荡 {pred.prob_neutral*100:.0f}% / 下跌 {pred.prob_down*100:.0f}%\n"
        f"- XGBoost 信号：{pred.xgb_signal:+d}\n"
        f"- 主要驱动因子（Top3）：{', '.join(pred.reasons or []) or '—'}\n"
    )
    return out


@_safe
def run_daily_pick(args: dict) -> str:
    """跑每日 AI 选股，返回打分最高的若干只。"""
    pool = _resolve_pool(args.get("pool"))
    top = int(args.get("top", 10))
    try:
        df, report = generate_daily_picks(pool=pool, max_picks=top)
    except Exception as e:  # noqa: BLE001
        return _failed(
            f"选股执行失败（pool={pool!r}）：{type(e).__name__}: {e}",
            f"{POOL_HINT}。请改用上述池名重新调用。",
        )
    # ⚠️ generate_daily_picks 内部会**吞掉**异常并返回 (空表, status="error")。
    # 不看 status 就会把「跑失败了」当成「跑通了但没有达标标的」——实测模型
    # 正是据此捏造出「大盘暂无评分≥55的股票」。这里必须显式区分。
    if str(report.get("status")) == "error":
        return _failed(
            f"选股失败（pool={pool!r}）：{report.get('error')}",
            f"{POOL_HINT}。请改用上述池名重新调用。",
        )
    if df is None or len(df) == 0:
        return f"本轮（{pool} 池）没有评分 ≥ {report.get('min_score')} 的达标标的。"
    cols = [c for c in ["代码", "名称", "评分", "上涨概率", "下跌概率", "推荐理由"] if c in df.columns]
    head = df[cols].head(top)
    lines = [f"## {pool} 池每日选股（前 {len(head)} 只，扫描 {report.get('n_scanned')} 只）"]
    for i, row in enumerate(head.itertuples(index=False), 1):
        rdict = row._asdict()
        lines.append(
            f"{i}. **{rdict.get('名称','?')}（{rdict.get('代码','?')}）** "
            f"评分 {rdict.get('评分','?')}｜涨 {rdict.get('上涨概率','?')}%｜"
            f"理由：{rdict.get('推荐理由','')}"
        )
    return "\n".join(lines)


@_safe
def get_index(args: dict) -> str:
    """大盘/指数行情：实时点位 + 涨跌幅 + 均线位置 + 近期趋势。

    为什么单独一个工具：`get_realtime` 只认 6 位个股代码，且 000001 对个股是
    平安银行、对上证指数是 sh000001。把指数塞给个股工具必然张冠李戴，
    所以指数走独立通道（fetcher.get_index_realtime 自带正确市场前缀）。
    """
    sym, err = _normalize_index(args.get("index") or args.get("code") or args.get("name"))
    if err:
        return _failed(err)
    q = fetcher.get_index_realtime(sym)
    if not q:
        return _failed(
            f"{sym} 指数实时行情获取失败（数据源不可用）。",
            "可稍后重试，或改为查询具体个股。",
        )
    lines = [
        f"## {q.get('名称', sym)}（{sym}）",
        f"- 最新点位：{q.get('最新价')}（{float(q.get('涨跌幅') or 0):+.2f}%）",
        f"- 今开 {q.get('今开')}｜最高 {q.get('最高')}｜最低 {q.get('最低')}｜昨收 {q.get('昨收')}",
        f"- 成交额：{q.get('成交额')} 万元",
    ]
    try:
        df = fetcher.get_index_kline(sym, start_date="2024-01-01")
    except Exception:  # noqa: BLE001
        df = None
    if df is not None and len(df) >= 20:
        close = df["close"]
        last = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        pos = "上方" if last >= ma20 else "下方"
        lines.append(f"- MA20：{ma20:.2f}（现价在 MA20 {pos}）")
        if len(df) >= 60:
            ma60 = float(close.rolling(60).mean().iloc[-1])
            lines.append(f"- MA60：{ma60:.2f}")
        for n in (5, 20, 60):
            if len(close) > n:
                pct = (last / float(close.iloc[-n - 1]) - 1) * 100
                lines.append(f"- 近 {n} 日涨跌：{pct:+.2f}%")
    return "\n".join(lines)


@_safe
def get_realtime(args: dict) -> str:
    """实时行情 / 盯盘诊断（腾讯源）。"""
    raw = args.get("code", "")
    # 指数写法（大盘/上证指数/创业板…）→ 明确指路，别硬当个股查
    idx_word = _index_by_word(raw)
    if idx_word:
        return _failed(
            f"「{raw}」是指数，不是个股；个股实时行情只支持 6 位股票代码。",
            f"请改用 get_index(index=\"{raw}\") 查询指数行情。",
        )
    code = _resolve_code(raw)
    if not is_valid_code(code):
        return _failed(
            f"无法识别股票代码/名称：{raw!r}。",
            "请提供 6 位 A 股代码（如 600519）或准确的股票名称；"
            "若想查大盘，请用 get_index。",
        )
    q = fetcher.get_realtime_quote(code)
    if not q:
        return _failed(f"{_name_of(code)}({code}) 实时行情获取失败（可能非交易时段或源不可用）。")
    name = q.get("名称", _name_of(code))
    price = q.get("最新价", 0)
    chg = q.get("涨跌幅", 0)
    amount = q.get("成交额", 0)
    vol = q.get("成交量", 0)
    turn = q.get("换手率", 0)
    pe = q.get("市盈率", 0)
    out = (
        f"## {name}（{code}）实时行情\n"
        f"- 最新价：{price}（{chg:+.2f}%）\n"
        f"- 成交额：{amount} 万元｜成交量：{vol} 手\n"
        f"- 换手率：{turn}%\n"
        f"- 市盈率(TTM)：{pe}\n"
    )
    # 补充近 20 日均量对比，给"放量/缩量"判断
    df = fetcher.get_daily_kline(code, start_date="2015-01-01", end_date=None)
    if len(df) >= 20:
        avg_vol = float(df["volume"].rolling(20).mean().iloc[-1])
        today_vol = float(df["volume"].iloc[-1])
        ratio = (today_vol / avg_vol) if avg_vol else 0
        out += f"- 近 20 日均量：{avg_vol/10000:.0f} 万手｜今日量比约 {ratio:.2f}x\n"
    return out


@_safe
def generate_report(args: dict) -> str:
    """生成一份综合研报/复盘 Markdown（选股结果 + 进化引擎状态）。"""
    top = int(args.get("top", 10))
    pick_lines = run_daily_pick({"pool": app_config.scheduler.pick_pool, "top": top})
    # 选股失败就不要拼一份「看起来正常」的研报 —— 直接把失败抛给代理去纠正
    if pick_lines.startswith(TOOL_FAILED_MARK):
        return pick_lines
    evo_lines = diagnose_evolution({})
    report = (
        "# 选股神 AI 综合研报\n\n"
        f"_生成时间：{_now()}_\n\n"
        "## 一、今日 AI 选股\n\n" + pick_lines + "\n\n"
        "## 二、进化引擎状态\n\n" + evo_lines + "\n\n"
        "---\n*本报告由 AI 助手自动汇编，仅供研究参考，不构成投资建议。*"
    )
    return report


@_safe
def diagnose_evolution(args: dict) -> str:
    """诊断自我进化引擎：状态、候选处方、覆盖层、是否在线。"""
    import os
    state_file = DATA_DIR / "evolution_state.json"
    override_file = DATA_DIR / "config" / "evolution_overrides.json" if False else None
    override_file = CONFIG_DIR / "evolution_overrides.json"

    lines = ["## 自我进化引擎诊断"]
    # 状态文件
    if state_file.exists():
        try:
            st = json.loads(state_file.read_text(encoding="utf-8"))
            lines.append(
                f"- 状态文件存在：retrain_running={st.get('retrain_running')}，"
                f"last_retrain_at={st.get('last_retrain_at')}，"
                f"candidates={len(st.get('candidates', []))}，"
                f"applied={len(st.get('applied', []))}"
            )
        except Exception as e:
            lines.append(f"- 状态文件解析失败：{e}")
    else:
        lines.append("- 状态文件不存在（进化引擎可能从未运行过）")

    # 覆盖层
    if override_file.exists():
        try:
            ov = json.loads(override_file.read_text(encoding="utf-8"))
            n = len(ov) if isinstance(ov, dict) else 0
            lines.append(f"- 参数覆盖层：{n} 条生效中")
        except Exception as e:
            lines.append(f"- 覆盖层解析失败：{e}")
    else:
        lines.append("- 参数覆盖层：无（全部走默认/进化基线）")

    # 在线/就绪
    try:
        from src.models.evolution import evolution as evo_engine
        lines.append(f"- 引擎在线：{'是' if getattr(evo_engine, 'is_ready', True) else '否'}")
    except Exception as e:
        lines.append(f"- 引擎在线检查失败：{e}")

    lines.append(
        "- 建议：若长期无 `applied` 候选，通常是闸门在「无样本外面板」时整批驳回；"
        "可检查面板缓存是否生成、score 是否稳定跑赢基线。"
    )
    return "\n".join(lines)


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M")


@_safe
def screen_stocks(args: dict) -> str:
    """智能筛选：全池 AI 评分 + 技术面条件过滤，自带研判并排序输出达标清单。"""
    pool = _resolve_pool(args.get("pool"))
    top = int(_opt_float(args.get("top")) if args.get("top") not in (None, "") else get_strategy_override("top", 15))
    min_score = _opt_float(args.get("min_score")) if args.get("min_score") not in (None, "") else get_strategy_override("min_score", 50.0)
    rsi_max = _opt_float(args.get("rsi_max")) if args.get("rsi_max") not in (None, "") else get_strategy_override("rsi_max", None)
    rsi_min = _opt_float(args.get("rsi_min")) if args.get("rsi_min") not in (None, "") else get_strategy_override("rsi_min", None)
    above_ma20 = _truthy(args.get("above_ma20", False))
    above_ma60 = _truthy(args.get("above_ma60", False))
    vol_ratio_min = _opt_float(args.get("vol_ratio_min")) if args.get("vol_ratio_min") not in (None, "") else get_strategy_override("vol_ratio_min", None)

    conds = [f"AI评分≥{min_score:.0f}"]
    if rsi_max is not None:
        conds.append(f"RSI≤{rsi_max:.0f}")
    if rsi_min is not None:
        conds.append(f"RSI≥{rsi_min:.0f}")
    if above_ma20:
        conds.append("站上MA20")
    if above_ma60:
        conds.append("站上MA60")
    if vol_ratio_min is not None:
        conds.append(f"量比≥{vol_ratio_min:.1f}")
    cond_str = "、".join(conds)

    try:
        codes = fetcher.get_stock_pool(pool)
    except Exception as e:  # noqa: BLE001
        return _failed(
            f"无法获取股票池 {pool!r}：{type(e).__name__}: {e}",
            f"{POOL_HINT}。请改用上述池名重新调用。",
        )
    raw = fetcher.get_batch_daily_kline(codes, max_workers=_screen_workers())
    stock_data = {c: df for c, df in raw.items() if df is not None and len(df) >= 60}

    stock_names = {}
    try:
        sl = fetcher.get_stock_list()
        nm = dict(zip(sl["code"], sl["name"]))
        stock_names = {c: nm.get(c, c) for c in stock_data}
    except Exception:
        stock_names = {c: c for c in stock_data}

    ai_ready = predictor.is_ready
    preds = predictor.predict_batch(stock_data, stock_names) if ai_ready else []
    preds = [p for p in preds if p.score >= min_score]
    pred_score = {p.code: p.score for p in preds}

    scan_codes = list(stock_data.keys()) if not ai_ready else [p.code for p in preds]

    rows = []
    for code in scan_codes:
        df = stock_data[code]
        t = _technicals(df)
        if rsi_max is not None and not (t["rsi"] <= rsi_max):
            continue
        if rsi_min is not None and not (t["rsi"] >= rsi_min):
            continue
        if above_ma20 and not (t["last"] > t["ma20"]):
            continue
        if above_ma60 and not (t["last"] > t["ma60"]):
            continue
        if vol_ratio_min is not None and not (t["vol_ratio"] >= vol_ratio_min):
            continue
        label, reason = _verdict(t)
        rows.append({
            "code": code,
            "name": stock_names.get(code, code),
            "score": pred_score.get(code, float("nan")),
            "rsi": t["rsi"],
            "vol_ratio": t["vol_ratio"],
            "verdict": label,
            "reason": reason,
            "chg": t["chg"],
            "last": t["last"],
        })

    if not rows:
        return (
            f"## {pool} 智能筛选（条件：{cond_str}）\n"
            f"扫描 {len(stock_data)} 只，技术+AI 双重过滤后 **0 只达标**"
            f"{'（AI 模型未就绪，仅按技术面筛选）' if not ai_ready else ''}。\n"
            "建议放宽 RSI/量比或降低 AI 评分阈值后再试。"
        )

    rows.sort(key=lambda r: r["score"] if not pd.isna(r["score"]) else -1, reverse=True)
    rows = rows[:top]

    lines = [
        f"## {pool} 智能筛选结果（条件：{cond_str}）",
        f"扫描 {len(stock_data)} 只 → 达标 **{len(rows)}** 只"
        f"{'（AI 未就绪，仅技术面）' if not ai_ready else ''}：",
        "",
        "| # | 代码 | 名称 | AI评分 | 技术研判 | RSI | 量比 | 理由 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(rows, 1):
        sc = f"{r['score']:.0f}" if not pd.isna(r["score"]) else "—"
        lines.append(
            f"| {i} | {r['code']} | {r['name']} | {sc} | {r['verdict']} | "
            f"{r['rsi']:.0f} | {r['vol_ratio']:.1f}x | {r['reason']} |"
        )
    return "\n".join(lines)


@_safe
def make_report(args: dict) -> str:
    """生成「可下载」的筛选研报：跑智能筛选并落盘 Markdown，返回 REPORT_FILE 标记。"""
    top = int(_opt_float(args.get("top")) if args.get("top") not in (None, "") else get_strategy_override("top", 20))
    screen_args = {
        "pool": args.get("pool") or app_config.scheduler.pick_pool,
        "top": top,
        "min_score": _opt_float(args.get("min_score")) if args.get("min_score") not in (None, "") else get_strategy_override("min_score", 55.0),
        "rsi_max": _opt_float(args.get("rsi_max")) if args.get("rsi_max") not in (None, "") else get_strategy_override("rsi_max", 65.0),
        "above_ma20": _truthy(args.get("above_ma20", True)),
        "vol_ratio_min": _opt_float(args.get("vol_ratio_min")) if args.get("vol_ratio_min") not in (None, "") else get_strategy_override("vol_ratio_min", 1.2),
    }
    body = screen_stocks(screen_args)
    report_text = (
        "# 选股神 AI · 智能筛选研报\n\n"
        f"_生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"
        "## 一、筛选条件与结果\n\n" + body + "\n\n"
        "---\n*本报告由 AI 助手自动生成，仅供研究参考，不构成投资建议。*"
    )
    out_dir = DATA_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"screen_report_{ts}.md"
    try:
        path.write_text(report_text, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return body + f"\n\n⚠️ 研报落盘失败：{e}"
    return (
        body + "\n\n"
        f"📄 已生成研报文件：`{path}`\n"
        f"REPORT_FILE:{path}"
    )


# ============================ 修改类工具（确认闸） ============================
# 这些工具都会「先预览、后执行」：未获 approve 时只返回提案（含 PENDING_ACTION 标记），
# 由引擎暂停循环、UI 弹出确认框；用户确认后才带 approve=True 真正落盘/改文件/模拟下单。

# 配置/参数白名单：键 -> (校验, 落盘, 中文标签)
_CONFIG_SETTERS = {
    "agent.screen_fetch_workers": {
        "validate": lambda v: max(1, min(16, int(float(v)))),
        "apply": lambda v: save_agent_state(int(v)),
        "label": "AI 助手筛选并发数",
    },
    "ml.use_ensemble": {
        "validate": lambda v: bool(_truthy(v)),
        "apply": lambda v: save_ml_state(bool(v)),
        "label": "模型融合开关(use_ensemble)",
    },
}


@_safe
def set_config(args: dict) -> str:
    """修改一项配置/参数（白名单）。未确认前只预览，确认后才落盘。"""
    key = str(args.get("key", "")).strip()
    raw = args.get("value")
    if key not in _CONFIG_SETTERS:
        return (f"⚠️ 该配置键不在可改白名单内：{key}。"
                f" 允许：{', '.join(_CONFIG_SETTERS.keys())}")
    spec = _CONFIG_SETTERS[key]
    try:
        new_val = spec["validate"](raw)
    except Exception as e:  # noqa: BLE001
        return f"⚠️ 参数值非法（{key}={raw!r}）：{e}"
    if not _truthy(args.get("approve")):
        cur = _read_config_current(key)
        # 空操作拦截：实测「把筛选并发改成 8」在本来就是 8 的情况下仍弹提案
        # 「从 8 改为 8」，用户点确认后什么都不会变 —— 这正是「点了没用」的来源之一。
        if str(cur) == str(new_val):
            return (f"ℹ️ 当前 `{key}` 已经是 `{new_val}`（{spec['label']}），无需修改。")
        return _propose(
            "set_config",
            {"key": key, "value": new_val, "approve": True},
            desc=f"将 **{spec['label']}**（`{key}`）从 `{cur}` 改为 `{new_val}`",
            impact="立即写盘到对应 state 文件（如 config/agent_state.json），下次相关操作即时生效。",
            risk="低",
        )
    spec["apply"](new_val)
    return f"✅ 已修改 `{key}` = `{new_val}`（{spec['label']}）"


_STRATEGY_LABELS = {
    "min_score": "筛选最低 AI 评分",
    "rsi_max": "筛选 RSI 上限",
    "rsi_min": "筛选 RSI 下限",
    "top": "筛选返回前 N 只",
    "vol_ratio_min": "筛选最小量比",
}


@_safe
def set_strategy_param(args: dict) -> str:
    """修改选股/策略默认参数（白名单键，落盘 strategy_overrides.json）。"""
    key = str(args.get("key", "")).strip()
    raw = args.get("value")
    if key not in _STRATEGY_LABELS:
        return (f"⚠️ 不在可改策略键内：{key}。允许：{', '.join(_STRATEGY_LABELS.keys())}")
    try:
        new_val = float(raw)
    except (TypeError, ValueError):
        return f"⚠️ 参数值非法（{key}={raw!r}），需要数字"
    if not _truthy(args.get("approve")):
        cur = get_strategy_override(key, "(默认)")
        # 同 set_config：值没变就别弹提案（否则「确认了也没变化」）
        try:
            if cur != "(默认)" and float(cur) == float(new_val):
                return f"ℹ️ 当前策略参数 `{key}` 已经是 `{new_val}`（{_STRATEGY_LABELS[key]}），无需修改。"
        except (TypeError, ValueError):
            pass
        return _propose(
            "set_strategy_param",
            {"key": key, "value": new_val, "approve": True},
            desc=f"将选股策略参数 **{_STRATEGY_LABELS[key]}**（`{key}`）从 `{cur}` 改为 `{new_val}`",
            impact="写入 config/strategy_overrides.json，之后「智能筛选/研报」在未显式指定该参数时采用此默认。",
            risk="低",
        )
    saved = save_strategy_override(key, new_val)
    return f"✅ 已修改策略参数 `{key}` = `{saved}`（{_STRATEGY_LABELS[key]}）"


def _make_diff(old: str, new: str) -> str:
    """对单处替换生成简短 diff 预览。"""
    import difflib
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(), lineterm="", fromfile="原", tofile="新",
    ))
    if not lines:
        lines = [f"- {old}", f"+ {new}"]
    return "\n".join(lines[:40])


@_safe
def edit_file(args: dict) -> str:
    """修改项目内的代码/文档文件（路径白名单 + diff 预览 + 确认闸）。"""
    rel = str(args.get("path", "")).strip()
    old = args.get("old_text", "")
    new = args.get("new_text", "")
    if not rel or not old:
        return "⚠️ edit_file 需要 path 与 old_text（要被替换的原文片段）。"
    root = Path(__file__).resolve().parents[2]
    target = (root / rel).resolve()
    allowed = False
    for pref in ALLOWED_EDIT_ROOTS:
        try:
            target.relative_to(Path(pref))
            allowed = True
            break
        except ValueError:
            pass
    if not allowed:
        return (f"⚠️ 出于安全，edit_file 只能修改以下根目录下的文件："
                f"{', '.join(ALLOWED_EDIT_ROOTS)}。已拒绝：{rel}")
    if target.suffix.lower() not in (".py", ".md", ".txt", ".csv"):
        return f"⚠️ 只允许编辑 .py/.md/.txt/.csv 文件，已拒绝后缀：{target.suffix}"
    if not target.exists():
        return f"⚠️ 文件不存在：{rel}"
    text = target.read_text(encoding="utf-8")
    if old not in text:
        return f"⚠️ 在 {rel} 中找不到指定的 old_text（请确认原文完全一致，含缩进/换行）。"
    if text.count(old) > 1:
        return f"⚠️ old_text 在 {rel} 中出现 {text.count(old)} 次，无法唯一定位；请提供更长的上下文片段。"
    if not _truthy(args.get("approve")):
        new_text = text.replace(old, new, 1)
        diff = _make_diff(old, new_text)
        return _propose(
            "edit_file",
            {"path": rel, "old_text": old, "new_text": new, "approve": True},
            desc=f"修改文件 `{rel}`（替换 1 处片段）",
            impact=f"写回磁盘后将影响运行中的代码/文档；若是代码，需要重启看板才能生效。\n\n```diff\n{diff}\n```",
            risk="中",
        )
    new_text = text.replace(old, new, 1)
    target.write_text(new_text, encoding="utf-8")
    return f"✅ 已修改 `{rel}`（替换 1 处片段）。若是代码文件，请重启看板使改动生效。"


@_safe
def propose_trade(args: dict) -> str:
    """提出一笔交易（仅 PAPER 纸账户模拟，确认闸）。绝不自动实盘。"""
    code = _resolve_code(str(args.get("code", "")).strip())
    side = str(args.get("side", "buy")).strip().lower()
    if side not in ("buy", "sell"):
        return "⚠️ side 只能是 buy 或 sell。"
    qty = _opt_float(args.get("quantity")) or _opt_float(args.get("qty"))
    if qty is None or qty <= 0:
        return "⚠️ 需要正的 quantity（股数）。"
    name = _name_of(code)
    if not _truthy(args.get("approve")):
        return _propose(
            "propose_trade",
            {"code": code, "side": side, "quantity": qty, "approve": True},
            desc=f"在 **纸账户(PAPER)** 中拟{'买入' if side == 'buy' else '卖出'} `{code}` {name} {int(qty)} 股",
            impact="仅模拟成交并记入纸账户审计日志，**不会动用任何真实资金**，也绝不会自动切到实盘。",
            risk="中（仅纸账户）",
        )
    # 执行：严格仅 PAPER 模拟，留痕到审计日志，不连接真实券商
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": "PAPER_SIM",
        "by": "ai_assistant",
        "code": code,
        "name": name,
        "side": side,
        "quantity": int(qty),
    }
    jf = DATA_DIR / "agent_trade_journal.jsonl"
    try:
        with open(jf, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass
    return (f"✅ 已在纸账户模拟{'买入' if side == 'buy' else '卖出'} `{code}` {name} {int(qty)} 股，"
            f"并记入审计日志（{jf.name}）。\n仅模拟，未连接真实券商、无任何真实资金变动。")


# ============================ 注册表 ============================

REGISTRY: Dict[str, Tool] = {
    "analyze_stock": Tool(
        name="analyze_stock",
        description="对单只股票做技术面 + AI 评分的深度分析（取数，不含叙述）。",
        params_desc="code: 6位股票代码或股票名称（如 '600519' 或 '贵州茅台'）",
        func=analyze_stock,
    ),
    "run_daily_pick": Tool(
        name="run_daily_pick",
        description="运行每日 AI 选股，返回打分最高的若干只标的。",
        params_desc=(
            f"pool: 股票池，只能是 {', '.join(_VALID_POOLS)}"
            "（中文也可：沪深300/中证500/中证1000/全市场）；top: 返回前几只（默认 10）"
        ),
        func=run_daily_pick,
    ),
    "get_index": Tool(
        name="get_index",
        description="查询大盘/指数行情（点位、涨跌幅、MA20/MA60、近期趋势）。问「大盘/指数怎么样」必须用它。",
        params_desc=(
            "index: 指数名，如 '上证指数'/'大盘'/'深证成指'/'创业板指'/"
            "'沪深300'/'中证500'/'科创50'（也接受 sh000001 这类代码）"
        ),
        func=get_index,
    ),
    "get_realtime": Tool(
        name="get_realtime",
        description="获取单只**个股**的实时行情并做量价诊断（盯盘用）。指数请用 get_index。",
        params_desc="code: 6位股票代码或名称（如 '600519' 或 '贵州茅台'）",
        func=get_realtime,
    ),
    "generate_report": Tool(
        name="generate_report",
        description="生成综合研报/复盘 Markdown（含今日选股 + 进化引擎状态）。",
        params_desc="top: 选股返回前几只（默认 10）",
        func=generate_report,
    ),
    "diagnose_evolution": Tool(
        name="diagnose_evolution",
        description="诊断自我进化引擎：运行状态、候选处方、参数覆盖层、是否在线。",
        params_desc="（无需参数）",
        func=diagnose_evolution,
    ),
    "screen_stocks": Tool(
        name="screen_stocks",
        description="按「AI评分 + 技术面条件」智能筛选全池股票，自带偏多/偏空研判并排序输出达标清单。",
        params_desc=(
            "pool: 股票池(hs300/zz500/zz1000/hs300_zz500/all，也认「沪深300」等中文)；"
            "min_score: AI最低评分(默认50)；"
            "rsi_max/rsi_min: RSI上下限；above_ma20/above_ma60: 是否要求站上均线(bool)；"
            "vol_ratio_min: 最小量比；top: 返回前几只(默认15)"
        ),
        func=screen_stocks,
    ),
    "make_report": Tool(
        name="make_report",
        description="生成一份「可下载」的筛选研报 Markdown（落盘到 data/reports/），适合用户要留档/导出的场景。",
        params_desc=(
            "同 screen_stocks 的参数；默认偏向「偏多」组合(站上MA20、量比≥1.2、AI≥55、RSI≤65)。"
            "调用后会返回 REPORT_FILE 标记指向生成的文件。"
        ),
        func=make_report,
    ),
    "set_config": Tool(
        name="set_config",
        description="（需确认）修改一项配置/参数并落盘；支持白名单键 agent.screen_fetch_workers、ml.use_ensemble。",
        params_desc="key: 配置键名；value: 新值；approve: 是否确认执行(bool，默认false=仅预览)",
        func=set_config,
    ),
    "set_strategy_param": Tool(
        name="set_strategy_param",
        description="（需确认）修改选股/策略默认参数并落盘到 strategy_overrides.json；键：min_score/rsi_max/rsi_min/top/vol_ratio_min。",
        params_desc="key: 策略键名；value: 新数值；approve: 是否确认执行(bool，默认false=仅预览)",
        func=set_strategy_param,
    ),
    "edit_file": Tool(
        name="edit_file",
        description="（需确认）修改项目内的代码/文档文件（src/、config/、scripts/ 下的 .py/.md/.txt/.csv），返回 diff 预览。",
        params_desc="path: 相对项目根的路径；old_text: 要替换的原文片段；new_text: 新内容；approve: 是否确认执行(bool)",
        func=edit_file,
    ),
    "propose_trade": Tool(
        name="propose_trade",
        description="（需确认）提出一笔交易，仅 PAPER 纸账户模拟并留痕，绝不自动实盘。",
        params_desc="code: 股票代码/名称；side: buy/sell；quantity: 股数(正)；approve: 是否确认执行(bool)",
        func=propose_trade,
    ),
}


def get_tools() -> Dict[str, Tool]:
    return REGISTRY


def call_tool(name: str, args: dict) -> str:
    tool = REGISTRY.get(name)
    if tool is None:
        return f"未知工具：{name}。可用工具：{', '.join(REGISTRY.keys())}"
    return tool.func(args or {})

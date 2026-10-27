# -*- coding: utf-8 -*-
"""
Streamlit 可视化看板
选股神 AI - 交互式选股决策界面
"""

import sys
import os
import re
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Windows 打包环境编码兜底（必须放在 import loguru 之前）
# ---------------------------------------------------------------------------
# 冻结运行（PyInstaller）时 sys.stderr 默认按 GBK(cp936) 编码，而看板日志里大量
# 出现 emoji（🤖 🔍 📊 🐞 …），一旦写出就抛：
#   UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f41e'
# 更糟的是 loguru 会用 error_interceptor 打印这个异常 —— 同样是写 stderr ——
# 于是二次崩溃，整个看板挂掉。
#
# 修法关键：用 reconfigure() 【就地】改编码，而不是重新赋值 sys.stderr。
# loguru 在 import 时就已经把 sys.stderr 这个对象抓进默认 sink 里了，
# 只有原地修改才能让它一起生效；整体替换的话 loguru 仍写旧对象，白改。
def _force_utf8_stdio() -> None:
    import io as _io

    for _name in ("stdout", "stderr"):
        _stream = getattr(sys, _name, None)
        if _stream is None:
            continue
        try:
            # 就地改造：loguru / streamlit 持有的还是同一个对象
            _stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except Exception:
            pass
        # 兜底：不是 TextIOWrapper（或被重定向到管道）时整体换一个 UTF-8 包装
        try:
            _buf = getattr(_stream, "buffer", None)
            if _buf is None:
                continue
            setattr(sys, _name, _io.TextIOWrapper(
                _buf, encoding="utf-8", errors="replace", line_buffering=True))
        except Exception:
            pass

    # GUI 模式（console=False / pythonw）下 stderr 可能是 None，
    # 补一个 UTF-8 日志文件流，避免 print/loguru 拿到 None 直接 AttributeError
    if getattr(sys, "stderr", None) is None:
        try:
            _log_dir = Path(os.environ.get(
                "STOCK_GOD_LOG_DIR",
                Path(sys.executable).parent / "logs" if getattr(sys, "frozen", False)
                else Path(__file__).parent.parent.parent / "logs"))
            _log_dir.mkdir(parents=True, exist_ok=True)
            sys.stderr = open(_log_dir / "app_stderr.log", "a",
                              encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdio()
# 子进程（下载/训练脚本等）也走 UTF-8
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from loguru import logger


# ---- 全局图表主题：统一字体 / 网格 / 配色序列 ----
# 只影响**没有显式传 template=** 的图表；显式传了的仍以页面为准。
# 目的：消除"同一份看板里每张图风格都不一样"的廉价感。
try:
    pio.templates.default = "plotly_white"
    _tpl = pio.templates.default.layout
    _tpl.font.family = "Inter, PingFang SC, Microsoft YaHei, sans-serif"
    _tpl.font.size = 12
    _tpl.colorway = ["#5b21b6", "#7c3aed", "#dc2626", "#059669",
                     "#d97706", "#3b82f6", "#64748b", "#a78bfa"]
    _tpl.paper_bgcolor = "rgba(0,0,0,0)"
    _tpl.plot_bgcolor = "rgba(0,0,0,0)"
    _tpl.margin = dict(l=8, r=8, t=8, b=8)
except Exception:  # noqa: BLE001  主题设置失败不影响看板主流程
    pass

from src.data.fetcher import fetcher
from src.models.predictor import predictor
from src.models.quantitative import quant_engine, risk_parity, MeanReversionStrategy
# 追踪器统一走 tracker_v2：旧 tracker.py 用「今天减 N 天」反推日期、基准代码解析错、
# 回撤恒为 0、且 T+5 与 T+10 互相覆盖，已废弃。反馈层（src/evolution/feedback.py）
# 早就在读 v2 的格式，看板若继续用旧版写入，两边 schema 会分叉。
from src.models.tracker_v2 import save_predictions, get_accuracy_summary
from src.models.llm import deepseek
from src.models.evolution import evolution
from config.settings import config as app_config
from config.settings import (save_scheduler_state, save_ml_state, save_agent_state,
                             save_pick_ensemble_state)
from src.scheduler.auto_scheduler import start_scheduler, is_running


def _ensure_scheduler_running():
    """看板进程内拉起「收盘后自动化调度」（若已启用）。幂等，可反复调用。

    关键改进：不再依赖浏览器会话（st.session_state）来触发。只要看板进程在、
    且开关为启用，调度器就在**首个脚本运行（即首次有人打开看板）**时自动拉起，
    并且之后每次脚本重跑都因 ``is_running()`` 短路而立刻返回，零额外开销。
    start_scheduler 内部用锁保证只起一个实例，所以这里无需额外去重。
    """
    try:
        if app_config.scheduler.enabled and not is_running():
            start_scheduler()
    except Exception as e:  # noqa: BLE001  非 Streamlit 上下文或导入失败时静默跳过
        logger.debug(f"调度自动拉起跳过: {e}")


def _loop_stale_threshold_min(now=None) -> float:
    """闭环心跳的「算不算停了」阈值（分钟）。

    必须跟着闭环自己的心跳间隔走：闭环在**非交易日**会把轮询拉长到 10 分钟
    一次（避免周末空转），所以休市时 10 分钟没心跳是正常的；而交易日是
    30 秒一次，用 30 分钟的阈值就等于放任它死了半小时才报警。

    拿不到日历时退回宽松值 30 分钟 —— 宁可漏报，也别在休市日误报「闭环挂了」
    （误报会让人习惯性忽略告警条，比漏报更危险）。
    """
    try:
        from src.trading.calendar import is_trading_day
        if is_trading_day((now or datetime.now()).date()):
            return 5.0
    except Exception:  # noqa: BLE001
        pass
    return 30.0


def _scheduler_daemon_heartbeat():
    """读常驻调度进程的心跳 ``data/scheduler_heartbeat.json``。

    ⚠️ 别用文件 mtime 判断存活 —— 进程被杀时文件会**保留最后的内容**，
    看上去"刚刚还活着"，实际早就死了。必须用文件里的 ``ts`` 算年龄。
    """
    try:
        from config.settings import DATA_DIR
        p = Path(DATA_DIR) / "scheduler_heartbeat.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        try:
            d["age_sec"] = (datetime.now()
                            - datetime.fromisoformat(str(d.get("ts") or ""))
                            ).total_seconds()
        except Exception:  # noqa: BLE001
            d["age_sec"] = None
        return d
    except Exception as e:  # noqa: BLE001
        logger.debug(f"调度心跳读取失败: {e}")
        return None
from src.ui.trading_ui import render_autotrade_page
from src.ui.agent_page import render_agent_page
from src.ui.common import (
    UP_COLOR, DOWN_COLOR, FLAT_COLOR, WARN_COLOR, INFO_COLOR,
    chg_color, chg_arrow, fmt_pct, fmt_num, fmt_yi,
    parse_codes, is_valid_code, build_name_map, lookup_name,
    calc_rsi, calc_macd, calc_kdj, calc_boll, calc_ma, boll_position,
    score_color, score_label, empty_state, require, signal_tag, tag_css,
    metric_card, html_metric,
    # v2 版式组件：统一页头 / KPI 行 / 白卡面板 / 空状态
    page_hero, render_hero, kpi_row, panel, empty_block,
)

# ========== 缓存层 ==========
# 注意：这些函数必须在下方各页面真正调用才有效。
# 之前这里定义了 5 个缓存函数却全是死代码，页面直接裸调网络接口，
# 导致每次组件交互都重新打一次行情 API。


@st.cache_data(ttl=600, show_spinner=False)
def _cached_stock_list():
    """缓存股票列表（10分钟）"""
    return fetcher.get_stock_list()


@st.cache_data(ttl=180, show_spinner=False)
def _cached_hotspots():
    """缓存板块热点（3分钟）"""
    return fetcher.get_sector_hotspots()


@st.cache_data(ttl=60, show_spinner=False)
def _cached_market_overview():
    """缓存大盘实时总览（1分钟）—— 主界面大盘涨跌条"""
    return fetcher.get_market_overview()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_daily_kline(code: str, start: str, end: str):
    """缓存日线数据（10分钟）

    注意：start/end 传空串表示"不限"，必须在这里归一化成真实区间再交给 fetcher。
    直接把 "" 传下去会让日期过滤得到空表（max() → NaT → 日期减法 TypeError）。
    """
    return fetcher.get_daily_kline(
        code,
        start_date=(start or "").strip() or "2015-01-01",
        end_date=(end or "").strip() or None,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_batch_kline(codes: tuple, start: str = "2015-01-01", end: str = ""):
    """缓存批量日线数据（10分钟）。codes 用 tuple 才能 hash。"""
    return fetcher.get_batch_daily_kline(list(codes), start_date=start, end_date=end or None)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_name_map() -> dict:
    """code -> name 映射（5分钟）。各页面共用，避免重复构建全表。"""
    return build_name_map(_cached_stock_list())


@st.cache_data(ttl=600, show_spinner=False)
def _cached_benchmark(index_code: str = "000300.SH", start: str = "2015-01-01"):
    """缓存基准指数（默认沪深300）日线，用于个股相对强弱比较。"""
    return fetcher.get_benchmark_kline(index_code, start_date=start)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_stock_info(code: str):
    """缓存单只股票实时基本信息（1小时）。

    优先用 ``fetcher.get_stock_info``；若当前运行的 fetcher（如打包后的冻结副本）
    缺少该接口，则回退到本文件内置的 ``_fetch_stock_info_local``，避免
    ``AttributeError`` 导致基本面信息永远空白。
    """
    if hasattr(fetcher, "get_stock_info"):
        try:
            r = fetcher.get_stock_info(code)
            if r is not None:
                return r
        except Exception:
            pass
    return _fetch_stock_info_local(code)


def _parse_cap_local(val):
    """把市值/股本解析成「亿」为单位（兼容 元量级数值 / 亿 / 万 三种返回）。"""
    if val is None:
        return 0.0
    try:
        if isinstance(val, (int, float)):
            num = float(val)
        else:
            s = str(val).strip()
            if not s:
                return 0.0
            unit = 1.0
            if "亿" in s:
                s = s.replace("亿", "")
            elif "万" in s:
                unit = 1e-4
                s = s.replace("万", "")
            num = float(s.replace(",", ""))
            num *= unit
        if num > 1e6:
            num = num / 1e8
        return float(num)
    except Exception:
        return 0.0


def _sina_symbol(code: str):
    """把 6 位代码转成新浪要求的带市场前缀格式（sh/sz/bj）。

    注意：akshare 1.18 的 ``stock_zh_a_hist`` 实际走的是东方财富 push2his，
    并非新浪；真正的新浪接口是 ``stock_zh_a_daily``，需要带前缀的 symbol。
    """
    c = str(code).strip().zfill(6)
    if c[0] == "6":
        return "sh" + c
    if c[0] in ("4", "8"):
        return "bj" + c
    return "sz" + c


def _fetch_stock_info_local(code: str):
    """多源兜底获取个股基本信息（不依赖 fetcher 版本）。

    数据源优先级：
      1) 东方财富 ``stock_individual_info_em`` —— 最全（行业/总市值/流通市值/上市日期）。
         但部分网络下 ``push2.eastmoney.com`` 不可达（RemoteDisconnected / ProxyError）。
      2) 备用源（东方财富不可达时启用，尽力补齐，避免基本面整块空白）：
         - 总市值   ← 百度 ``stock_zh_valuation_baidu(indicator="总市值")``（单位：亿）
         - 上市日期 ← 新浪 ``stock_zh_a_daily`` 首行日期（即上市首日）
         - 流通市值 ← 新浪 ``outstanding_share``(流通股本) × 最新收盘价
         - 行业 备用源取不到 → 记入 ``_missing``，由 UI 低调标注。
    """
    # 注意：name 初始必须为 ""，不能回填 code —— UI 会 `if _ri.get("name")` 覆盖
    # base["name"]，若这里塞了代码，会把静态列表里的真实名称（如「平安银行」）冲掉。
    result = {
        "code": code, "name": "", "industry": "", "total_cap": 0.0,
        "float_cap": 0.0, "listed_date": "", "_source": "", "_missing": [],
    }
    em_err = None

    # 1) 东方财富：信息最全，优先
    try:
        import akshare as ak
        raw = ak.stock_individual_info_em(symbol=code)
        if raw is not None and len(raw) > 0:
            kv = {}
            for _, r in raw.iterrows():
                item = str(r.get("item", "")).strip()
                if item:
                    kv[item] = r.get("value")
            result.update({
                "name": str(kv.get("股票简称", "") or ""),
                "industry": str(kv.get("行业", "") or ""),
                "total_cap": _parse_cap_local(kv.get("总市值")),
                "float_cap": _parse_cap_local(kv.get("流通市值")),
                "listed_date": str(kv.get("上市时间", "") or ""),
                "_source": "东方财富",
            })
            return result
    except Exception as e:
        em_err = str(e)

    # 2) 备用源 A：百度估值 → 总市值（返回已是「亿」）
    try:
        import akshare as ak
        b = ak.stock_zh_valuation_baidu(symbol=code, indicator="总市值", period="近一年")
        if b is not None and len(b) > 0:
            v = b["value"].iloc[-1]
            if v is not None and v == v:
                result["total_cap"] = _parse_cap_local(v)
                result["_source"] = "百度(总市值)"
    except Exception:
        pass

    # 3) 备用源 B：新浪日线 stock_zh_a_daily（非东方财富）
    #    - 首行日期                              = 上市日期
    #    - outstanding_share(流通股本) × 收盘价   = 流通市值
    try:
        import akshare as ak
        h = ak.stock_zh_a_daily(symbol=_sina_symbol(code), start_date="19800101",
                                end_date=datetime.now().strftime("%Y%m%d"), adjust="")
        if h is not None and len(h) > 0:
            result["listed_date"] = str(h["date"].iloc[0])[:10]
            try:
                _os = float(h["outstanding_share"].iloc[-1])
                _cl = float(h["close"].iloc[-1])
                if _os > 0 and _cl > 0:
                    result["float_cap"] = round(_os * _cl / 1e8, 2)
            except Exception:
                pass
            result["_source"] = (result["_source"] + "+新浪(上市日/流通市值)").lstrip("+")
    except Exception:
        pass

    # 缺失字段标注（供 UI 低调提示，不再整块空白）
    for k, label in (("industry", "行业"), ("float_cap", "流通市值"),
                     ("total_cap", "总市值"), ("listed_date", "上市日期")):
        if not result.get(k):
            result["_missing"].append(label)

    if result["total_cap"] or result["listed_date"]:
        return result

    result["_error"] = em_err or "东方财富返回空数据"
    return result


def clear_predictor_cache():
    """训练后清缓存让数据重新加载"""
    for fn in (_cached_stock_list, _cached_hotspots, _cached_daily_kline,
               _cached_batch_kline, _cached_market_overview, _cached_name_map):
        try:
            fn.clear()
        except Exception:
            pass


def get_name(code: str) -> str:
    """统一取股票名：查不到就返回代码，绝不抛异常"""
    try:
        return lookup_name(_cached_name_map(), code)
    except Exception:
        return str(code)


# 后台训练子进程的句柄。放模块级全局而不放 st.session_state：
# Popen 对象不可序列化，且模块全局在同会话的多次 rerun 之间是保留的。
_TRAIN_PROC: dict = {}


# ========== 模型自动热重载（与页面无关） ==========
# 背景：后台训练脚本写的是 models/xgboost_model.pkl，但原先只有「⚙️ 设置」页里
# 才会执行 predictor.reload()。用户一旦切到「🔍 AI选股」页（选股主流程在
# predictor.get_top_recommendations，没有任何 reload / 时间戳检查），设置页代码
# 不再执行 → 内存里永远是训练前的旧模型。这正是「训练跑完了但结果没变」的根因。
#
# 这里按模型文件的 mtime 做一次与页面无关的自动热重载。
#
# 性能红线：只用 os / Path 这类轻模块，绝不 import pandas / numpy / xgboost /
# sklearn 等重库 —— 本项目启动时对重库导入极度敏感，历史上出过
# 「import pandas 与 import streamlit 并发 → partially initialized module 'pandas'」
# 导致看板起不来的故障。
_MODEL_PKL_REL = Path("models") / "xgboost_model.pkl"


def _auto_reload_model_if_updated() -> None:
    """按 models/xgboost_model.pkl 的 mtime 判断是否被后台训练改写过，是则热重载。

    与当前页面无关，解决'reload 只写在设置页、切页即失联'的问题。
    首次调用只记录 mtime、不重载（避免每次启动白反序列化一次模型）；
    之后检测到 mtime 变化才调 predictor.reload()。
    整个函数用 try/except 包住，任何异常都静默放过，绝不能因为它导致看板起不来。
    """
    try:
        base = Path(__file__).parent.parent.parent
        pkl = base / _MODEL_PKL_REL
        if not pkl.exists():
            return

        mtime = pkl.stat().st_mtime
        key = "_model_pkl_mtime"
        prev = st.session_state.get(key)

        # 首次只记录基准 mtime，不重载
        if prev is None:
            st.session_state[key] = mtime
            return

        if mtime != prev:
            st.session_state[key] = mtime
            predictor.reload()
            clear_predictor_cache()
            logger.info(f"检测到模型文件更新，已自动热重载: {pkl.name}")
    except Exception as e:  # noqa: BLE001 - 热重载绝不能影响看板启动
        try:
            logger.debug(f"模型自动热重载跳过: {e}")
        except Exception:  # noqa: BLE001
            pass


# 在任何页面路由 / 侧边栏渲染之前执行一次，保证切到任何页面都能感知新模型
_auto_reload_model_if_updated()

# 「收盘后自动化调度」随看板**进程启动**即拉起（不再等到首次有人打开看板再触发）。
# 幂等：is_running() 短路，重跑零开销；受 config.scheduler.enabled 控制。
_ensure_scheduler_running()


# ========== 个股盯盘片段（原地刷新） ==========
# 把盯盘面板的「拉取 + 渲染」包进 st.fragment：自动刷新只重跑该片段、
# 不触发整页 rerun，从而彻底消除刷新间隙的闪动、也不打扰同页其它控件。
_WATCH_POLL_SEC = 5  # 片段轮询周期（秒）；行情/AI 的实际重拉仍由各自 TTL 控制


def _save_last_watched(code: str) -> None:
    """把最近盯盘的代码持久化到 config/watch_last.json，跨会话（重开看板）仍带出。"""
    try:
        import json as _json
        _p = Path(__file__).parent.parent.parent / "config" / "watch_last.json"
        _p.write_text(_json.dumps({"code": code}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_last_watched() -> str:
    try:
        import json as _json
        _p = Path(__file__).parent.parent.parent / "config" / "watch_last.json"
        if _p.exists():
            return _json.loads(_p.read_text(encoding="utf-8")).get("code", "") or ""
    except Exception:
        pass
    return ""


@st.fragment(run_every=_WATCH_POLL_SEC)
def render_watch_fragment() -> None:
    """个股盯盘面板：行情/AI 双层 TTL + 实时价滚动 sparkline，原地刷新。

    读取同页控件（watch_code / watch_interval / watch_ai_interval / watch_auto /
    watch_go / watch_refresh）的 session_state；行情按 5s（盘中自动）或用户间隔刷新，
    AI 按 watch_ai_interval 重算，避免盘中每次刷新都跑模型。
    """
    watch_code = (st.session_state.get("watch_code") or "").strip().upper()
    watch_interval = int(st.session_state.get("watch_interval", 10))
    watch_ai_interval = int(st.session_state.get("watch_ai_interval", 60))
    watch_auto = bool(st.session_state.get("watch_auto", True))
    watch_go = bool(st.session_state.get("watch_go", False))
    watch_refresh = bool(st.session_state.get("watch_refresh", False))

    # 交易时段：片段独立重跑，必须自己算，不能依赖外层捕获
    now = datetime.now()
    market_open = (now.weekday() < 5 and (
        (now.hour == 9 and now.minute >= 30) or now.hour == 10 or
        (now.hour == 11 and now.minute <= 30) or now.hour == 13 or
        now.hour == 14 or (now.hour == 15 and now.minute == 0)))

    cache = st.session_state.get("watch_cache", {}) or {}
    code = watch_code

    # 行情刷新 TTL：盘中自动时按轮询周期拿实时价（行情便宜），其余按用户间隔
    _quote_ttl = _WATCH_POLL_SEC if (market_open and watch_auto) else watch_interval
    _force = watch_go or watch_refresh
    _first = bool(code and "watch_last_ts" not in st.session_state)
    _last = st.session_state.get("watch_last_ts")
    _auto_due = bool(code and watch_auto and _last is not None and (now - _last).total_seconds() > _quote_ttl)
    need_quote = _force or _first or _auto_due

    _prev_pred = cache.get("pred") if cache.get("code") == code else None
    _ai_ts = cache.get("ai_fetched_at")
    _ai_age = (now - _ai_ts).total_seconds() if _ai_ts else 1e9
    need_ai = bool(_force) or (_prev_pred is None) or (_ai_age > watch_ai_interval) or (cache.get("code") != code)

    if need_quote and code:
        if not is_valid_code(code):
            st.warning("请输入6位股票代码")
        else:
            st.session_state.watch_last_ts = now
            stock_name = get_name(code)
            df_k = _cached_daily_kline(code, "", "")
            if len(df_k) < 30:
                if _force:
                    st.warning(f"{code} {stock_name} 数据不足")
            else:
                close = df_k["close"]
                last_close = float(close.iloc[-1])
                prev_close = float(close.iloc[-2])
                day_chg = (last_close / prev_close - 1) * 100

                live_price = last_close
                live_chg = day_chg
                live_vol = int(df_k["volume"].iloc[-1])
                if market_open:
                    spot = fetcher.get_realtime_quote(code)
                    if spot and spot.get("最新价", 0) > 0:
                        live_price = float(spot["最新价"])
                        live_chg = float(spot.get("涨跌幅", day_chg))
                        if spot.get("成交量", 0) > 0:
                            live_vol = int(spot["成交量"])
                    else:
                        try:
                            import akshare as ak
                            df_spot = ak.stock_zh_a_spot_em().set_index("代码")
                            if code in df_spot.index:
                                spot2 = df_spot.loc[code]
                                live_price = float(spot2.get("最新价", last_close))
                                live_chg = float(spot2.get("涨跌幅", day_chg))
                                live_vol = int(float(spot2.get("成交量", 0))) if spot2.get("成交量", 0) else live_vol
                        except Exception:
                            pass

                if need_ai:
                    pred = None
                    try:
                        pred = predictor.predict_single(df_k, code, stock_name)
                    except Exception:
                        pred = None
                    ai_fetched_at = now
                else:
                    pred = _prev_pred
                    ai_fetched_at = _ai_ts

                st.session_state.watch_cache = {
                    "code": code, "stock_name": stock_name,
                    "live_price": live_price, "live_chg": live_chg, "live_vol": live_vol,
                    "df_k": df_k, "pred": pred,
                    "fetched_at": now, "ai_fetched_at": ai_fetched_at, "market_open": market_open,
                }

                # 实时价滚动缓冲（仅盘中），最多保留 60 个点
                if market_open and live_price > 0:
                    _hist = st.session_state.setdefault("watch_prices", {})
                    _buf = _hist.setdefault(code, [])
                    _buf.append((now, live_price))
                    if len(_buf) > 60:
                        _hist[code] = _buf[-60:]
                    for _k in list(_hist.keys()):
                        if _k != code:
                            _hist.pop(_k, None)

                # 跨会话持久化 last-watched（仅代码变化时写盘）
                if st.session_state.get("_watch_persist_code") != code:
                    _save_last_watched(code)
                    st.session_state._watch_persist_code = code

    # ---- 渲染（始终从缓存） ----
    _cache = st.session_state.get("watch_cache", {}) or {}
    if _cache.get("code"):
        code = _cache["code"]
        stock_name = _cache["stock_name"]
        live_price = _cache["live_price"]
        live_chg = _cache["live_chg"]
        live_vol = _cache["live_vol"]
        df_k = _cache["df_k"]
        pred = _cache["pred"]
        fetched_at = _cache.get("fetched_at")

        close = df_k["close"]
        last_close = float(close.iloc[-1])

        # 价格卡片行
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        chg_icon = "📈" if live_chg >= 0 else "📉"
        with pc1:
            metric_card(label=f"{chg_icon} {stock_name}({code})", value=f"{live_price:.2f}", chg=live_chg)
        with pc2:
            ma5 = float(close.rolling(5).mean().iloc[-1])
            metric_card(label="MA5", value=f"{ma5:.2f}",
                        chg=(live_price / ma5 - 1) * 100 if ma5 > 0 else None,
                        chg_text=f"{(live_price/ma5-1)*100:+.2f}% vs 现价" if ma5 > 0 else "")
        with pc3:
            vol_wan = live_vol / 10000
            avg_vol = float(df_k["volume"].rolling(20).mean().iloc[-1]) / 10000
            st.metric(label="成交量(万)", value=f"{vol_wan:.0f}", delta=f"{vol_wan/avg_vol*100:.0f}% vs 20日均量" if avg_vol > 0 else None)
        with pc4:
            rsi_v = float(calc_rsi(close).iloc[-1])
            rsi_label = "超买" if rsi_v > 70 else ("超卖" if rsi_v < 30 else ("偏强" if rsi_v > 50 else "偏弱"))
            st.metric(label=f"RSI(14)·{rsi_label}", value=f"{rsi_v:.0f}")
        with pc5:
            if pred:
                ai_icon = "📈" if pred.prob_up > 0.5 else ("📉" if pred.prob_down > 0.5 else "➡️")
                st.metric(label=f"{ai_icon} AI评分", value=f"{pred.score:.0f}", delta=f"确信{pred.confidence*100:.0f}%")
            else:
                st.metric(label="AI评分", value="—")

        # 盘中实时价滚动 sparkline
        _hist = st.session_state.get("watch_prices", {})
        _buf = _hist.get(code)
        if _buf and len(_buf) >= 2 and _cache.get("market_open"):
            _ts = [t for t, _ in _buf]
            _px = [p for _, p in _buf]
            fig_sp = go.Figure()
            _sp_color = UP_COLOR if _px[-1] >= _px[0] else DOWN_COLOR
            fig_sp.add_trace(go.Scatter(x=_ts, y=_px, mode="lines",
                                        line=dict(color=_sp_color, width=1.5),
                                        fill="tozeroy", fillcolor=_sp_color + "22"))
            fig_sp.update_layout(height=90, margin=dict(l=0, r=0, t=4, b=4),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 template="plotly_white", showlegend=False)
            st.plotly_chart(fig_sp, use_container_width=True, key="watch_spark")

        # 交易信号 + K线
        tc1, tc2 = st.columns([1, 1.5])
        with tc1:
            if pred:
                st.markdown('<div class="section-title">🎯 交易信号</div>', unsafe_allow_html=True)
                reason_html = ""
                for r in getattr(pred, 'reasons', [])[:4]:
                    reason_html += f'<div style="font-size:0.8rem;padding:2px 0;color:#475569;">• {r}</div>'
                if pred.score >= 80:
                    adv = ("强力做多", "#ef4444", "🟢")
                elif pred.score >= 65:
                    adv = ("建议做多", "#f59e0b", "🟡")
                elif pred.score >= 50:
                    adv = ("谨慎关注", "#64748b", "⚪")
                elif pred.score >= 35:
                    adv = ("建议观望", "#64748b", "⚪")
                else:
                    adv = ("建议回避", "#059669", "🔴")
                st.markdown(f"""
                <div style="background:{adv[1]}10; border:1px solid {adv[1]}30; border-radius:12px; padding:0.8rem; margin:0.5rem 0;">
                    <p style="font-weight:700; color:{adv[1]}; font-size:1.1rem; margin:0;">{adv[2]} {adv[0]}</p>
                    <p style="color:#64748b; font-size:0.75rem; margin:0.3rem 0 0 0;">综合评分 {pred.score:.0f} | 置信度 {pred.confidence*100:.0f}% | {'模型看涨' if pred.xgb_signal == 1 else ('模型看跌' if pred.xgb_signal == -1 else '模型观望')}</p>
                </div>
                """, unsafe_allow_html=True)
                st.caption("模型预测概率")
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
                    <span style="font-size:0.75rem; width:30px; color:#ef4444;">📈涨</span>
                    <div style="flex:1; height:18px; background:#e2e8f0; border-radius:9px; overflow:hidden;">
                        <div style="width:{pred.prob_up*100:.0f}%; height:100%; background:linear-gradient(90deg,#ef4444,#f87171); border-radius:9px;"></div>
                    </div>
                    <span style="font-size:0.75rem; font-weight:600; color:#ef4444;">{pred.prob_up*100:.0f}%</span>
                </div>
                <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
                    <span style="font-size:0.75rem; width:30px; color:#059669;">📉跌</span>
                    <div style="flex:1; height:18px; background:#e2e8f0; border-radius:9px; overflow:hidden;">
                        <div style="width:{pred.prob_down*100:.0f}%; height:100%; background:linear-gradient(90deg,#059669,#34d399); border-radius:9px;"></div>
                    </div>
                    <span style="font-size:0.75rem; font-weight:600; color:#059669;">{pred.prob_down*100:.0f}%</span>
                </div>
                """, unsafe_allow_html=True)
                if reason_html:
                    st.markdown(reason_html, unsafe_allow_html=True)
            else:
                st.info("模型未就绪")
        with tc2:
            df_plot = df_k.iloc[-60:].copy()
            fig_mini = go.Figure()
            colors_k = ["#ef4444" if df_plot["close"].iloc[i] >= df_plot["open"].iloc[i] else "#059669" for i in range(len(df_plot))]
            fig_mini.add_trace(go.Candlestick(
                x=df_plot["date"], open=df_plot["open"], high=df_plot["high"],
                low=df_plot["low"], close=df_plot["close"],
                increasing_line_color="#ef4444", decreasing_line_color="#059669",
                name="K线"
            ))
            fig_mini.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["close"].rolling(5).mean(),
                                           mode="lines", line=dict(color="#f59e0b", width=1), name="MA5"))
            fig_mini.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["close"].rolling(20).mean(),
                                           mode="lines", line=dict(color="#8b5cf6", width=1.2), name="MA20"))
            high60 = float(df_plot["high"].max())
            low60 = float(df_plot["low"].min())
            pivot = (float(df_plot["high"].iloc[-1]) + float(df_plot["low"].iloc[-1]) + float(df_plot["close"].iloc[-1])) / 3
            for y, name, color, dash in [(high60, "压力", "#ef4444", "dash"), (low60, "支撑", "#059669", "dash"), (pivot, "枢轴", "#64748b", "dot")]:
                fig_mini.add_hline(y=y, line_dash=dash, line_color=color, opacity=0.5,
                                   annotation_text=f"{name} {y:.2f}", annotation_position="top right")
            fig_mini.update_layout(height=320, template="plotly_white",
                                    title=f"{stock_name} ({code}) · 60日K线",
                                    xaxis_rangeslider_visible=False,
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_mini, use_container_width=True, key="watch_kline")

        # 技术指标速览
        ti1, ti2, ti3, ti4, ti5, ti6 = st.columns(6)
        dif, dea, macd_bar = calc_macd(close)
        macd_val = float(macd_bar.iloc[-1])
        with ti1:
            st.metric(label="MACD", value=f"{macd_val:.3f}", delta="金叉" if macd_val > 0 and float(macd_bar.iloc[-2]) <= 0 else ("死叉" if macd_val < 0 and float(macd_bar.iloc[-2]) >= 0 else None))
        low9 = df_k["low"].rolling(9).min()
        high9 = df_k["high"].rolling(9).max()
        rsv = (close - low9) / (high9 - low9 + 1e-10) * 100
        k_val = rsv.ewm(alpha=1/3, adjust=False).mean()
        d_val = k_val.ewm(alpha=1/3, adjust=False).mean()
        j_val = 3 * k_val - 2 * d_val
        with ti2:
            st.metric(label="KDJ-K", value=f"{float(k_val.iloc[-1]):.0f}")
        with ti3:
            st.metric(label="KDJ-J", value=f"{float(j_val.iloc[-1]):.0f}", delta="超买" if float(j_val.iloc[-1]) > 100 else ("超卖" if float(j_val.iloc[-1]) < 0 else None))
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_pos = (last_close - float(bb_mid.iloc[-1])) / (2 * float(bb_std.iloc[-1]) + 1e-10)
        with ti4:
            st.metric(label="布林位置", value=f"{bb_pos:.2f}σ", delta="上轨附近" if bb_pos > 0.8 else ("下轨附近" if bb_pos < -0.8 else None))
        with ti5:
            chg_5d = (last_close / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0
            st.metric(label="5日涨幅", value=f"{chg_5d:+.2f}%")
        with ti6:
            chg_20d = (last_close / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0
            st.metric(label="20日涨幅", value=f"{chg_20d:+.2f}%")

        # 数据时效提示（行情 vs AI 分开展示）
        if fetched_at:
            _age = int((datetime.now() - fetched_at).total_seconds())
            _ai_ts = _cache.get("ai_fetched_at")
            _ai_age = int((datetime.now() - _ai_ts).total_seconds()) if _ai_ts else None
            if _cache.get("market_open"):
                _src = "盘中实时行情"
            elif watch_auto:
                _src = "非交易时段 · 仅显示上一交易日收盘"
            else:
                _src = "非交易时段 · 自动刷新已关闭"
            _cap = f"🕒 行情 {_age} 秒前 · AI评分" + (f" {_ai_age} 秒前" if _ai_age is not None else " —") + f"（{_src}）"
            st.caption(_cap)
        st.divider()
    else:
        st.caption("👆 输入股票代码，或点选上方自选股，开始盯盘")


# ========== 大盘涨跌条 ==========

def render_market_ticker():
    """主界面顶部大盘实时涨跌条：主要指数 + 涨跌家数 + 成交额

    性能关键：原来这里无条件拉实时行情，首屏必然卡在联网 + akshare 导入上。
    现在改为「先读会话缓存」：没缓存就直接跳过（不渲染、不联网），
    用户在侧边栏点「加载大盘行情」后再刷新即可显示。
    """
    mk = st.session_state.get("_mk_overview")
    if mk is None:
        # 首屏不阻塞：只留一行极轻的提示（长句会变成噪音，压掉视觉重心）
        st.caption("📊 大盘行情未加载 · 侧边栏「大盘速览」可加载")
        return

    idx: pd.DataFrame = mk.get("indices", pd.DataFrame())
    if idx is None or len(idx) == 0:
        st.info("📊 大盘行情暂不可用（数据源连接失败），请稍后刷新重试。")
        return

    _breadth = mk.get("breadth") or {}
    up = int(_breadth.get("上涨", 0))
    down = int(_breadth.get("下跌", 0))
    flat = int(_breadth.get("平盘", 0))
    total = up + down + flat
    amount = float(mk.get("amount_yi") or 0)

    # ---------- 标题行 ----------
    stamp = mk.get("updated") or datetime.now().strftime("%H:%M:%S")
    tcol_l, tcol_r = st.columns([5, 1])
    with tcol_l:
        st.markdown(
            f'<div class="section-title" style="margin-top:.2rem;">📊 大盘实时'
            f'<span style="font-size:.75rem;font-weight:500;color:#94a3b8;margin-left:8px;">'
            f'更新 {stamp}</span></div>',
            unsafe_allow_html=True,
        )
    with tcol_r:
        if st.button("🔄 刷新", key="mkt_refresh", help="清除缓存并重新拉取大盘行情",
                     use_container_width=True):
            _cached_market_overview.clear()
            st.rerun()

    # ---------- 指数卡片 ----------
    cards = list(idx.itertuples(index=False))
    n = len(cards)
    cols = st.columns(n)
    for col, it in zip(cols, cards):
        pct = float(it.涨跌幅)
        chg = float(it.涨跌额)
        price = float(it.最新价)
        # A股习惯：红涨绿跌（色带与文字同步，走 .mini-card 统一卡片观感）
        up_flag = pct >= 0
        clr = "#dc2626" if up_flag else "#059669"
        arrow = "▲" if up_flag else "▼"
        col.markdown(
            f'<div class="mini-card {"up" if up_flag else "down"}" '
            f'style="padding:.75rem .5rem;">'
            f'<div class="label" style="text-transform:none;letter-spacing:0;">{it.名称}</div>'
            f'<div class="val" style="color:{clr};">{price:,.2f}</div>'
            f'<div class="sub" style="color:{clr};font-weight:700;">'
            f'{arrow} {pct:+.2f}%</div>'
            f'<div class="sub">{chg:+.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ---------- 涨跌家数 + 成交额 + 情绪 ----------
    if total > 0:
        up_w = up / total * 100
        flat_w = flat / total * 100
        st.markdown(
            f'<div style="display:flex;height:9px;border-radius:5px;overflow:hidden;'
            f'margin:.6rem 0 .3rem 0;border:1px solid #e5e7eb;">'
            f'<div style="width:{up_w:.1f}%;background:#dc2626;"></div>'
            f'<div style="width:{flat_w:.1f}%;background:#cbd5e1;"></div>'
            f'<div style="width:{100 - up_w - flat_w:.1f}%;background:#059669;"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:.76rem;'
            f'color:#64748b;">'
            f'<span style="color:#dc2626;font-weight:600;">🔴 上涨 {up} 家'
            f'（{up_w:.0f}%）</span>'
            f'<span>⚪ 平盘 {flat} 家</span>'
            f'<span style="color:#059669;font-weight:600;">🟢 下跌 {down} 家'
            f'（{100 - up_w - flat_w:.0f}%）</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        ratio = up / max(down, 1)
        mood = "🔥 情绪亢奋" if ratio >= 2 else (
            "🙂 偏强" if ratio >= 1.2 else (
                "😐 分化" if ratio >= 0.8 else (
                    "😟 偏弱" if ratio >= 0.4 else "🥶 重挫")))
        mood_clr = "#dc2626" if ratio >= 1.2 else (
            "#059669" if ratio < 0.8 else "#d97706")
    else:
        mood, mood_clr = "—", "#94a3b8"
        st.caption("涨跌家数数据源暂不可用")

    lim = mk.get("limits") or {}
    n_up = int(lim.get("涨停", 0) or 0)
    n_dn = int(lim.get("跌停", 0) or 0)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("💰 两市成交额", f"{amount:,.0f} 亿" if amount else "—")
    with b2:
        # 之前 mood_clr 算好了却没传给任何组件，情绪一直是默认黑字
        st.markdown(
            f'<div class="mini-card plain" style="padding:.7rem .8rem;">'
            f'<div class="label" style="text-transform:none;letter-spacing:0;">🌡️ 市场情绪</div>'
            f'<div class="val" style="color:{mood_clr};">{mood}</div>'
            f'</div>', unsafe_allow_html=True)
    b3.metric(
        "🚀 涨停 / 跌停",
        f"{n_up} / {n_dn}" if (n_up or n_dn) else "—",
        delta=None if not n_up else f"{n_up - n_dn:+d}",
    )
    b4.metric("⚖️ 涨跌比", f"{up}/{down}" if total else "—")

    src_label = {
        "eastmoney": "东方财富", "tencent": "腾讯行情",
        "stale": "上次数据（数据源异常）", "none": "不可用",
    }.get(mk.get("source", ""), mk.get("source", ""))
    st.caption(
        f"💡 红涨绿跌 · 数据源：{src_label} · 盘中每 60 秒自动刷新 · "
        "点击右上「🔄 刷新」可强制更新"
    )
    st.markdown('<div style="height:.4rem;"></div>', unsafe_allow_html=True)


# ========== 今日状态条（纯本地文件，零联网） ==========
#
# 为什么要有这块：首屏的「大盘行情」「热点板块」为了性能都改成了手动加载，
# 结果打开看板第一屏几乎没有数字，只有两行"请点击加载" —— 这是"简陋感"的
# 主要来源之一。这里把本来就在磁盘上的运营快照（计划/账户/闭环心跳/预测/IC）
# 直接摊在首屏：不联网、不触发数据源、毫秒级。
#
# 副产品（很重要）：把「交易闭环心跳」放到首屏最显眼处。
# 2026-10-18 出过一整天的静默事故——run_loop 因 GBK 编码崩溃秒退，但心跳只是
# "停在昨天"，页面上完全看不出来。现在停机会变成一条黄色警示条。

_DATE_FMT = "%Y-%m-%d"


def _status_read_json(path: Path):
    """读 JSON，失败返回 None（首屏装饰性数据，绝不能因此抛异常）"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _today_status_data() -> dict:
    """汇总首屏运营快照。全部读本地文件，TTL 60 秒。"""
    root = Path(__file__).parent.parent.parent
    out: dict = {"plan": None, "acct": None, "loop": None,
                 "pred": None, "ic": None}

    # --- 交易计划：取 plan_date 最大的一份（通常是"下一个交易日"） ---
    try:
        best, best_date = None, ""
        files = sorted((root / "data" / "trade_plans").glob("trade_plan_*.json"))[-8:]
        for f in files:
            d = _status_read_json(f) or {}
            dt = str(d.get("plan_date") or "")
            if dt >= best_date:
                best, best_date = d, dt
        if best:
            tg = best.get("targets") or []
            out["plan"] = {
                "date": best_date,
                "n": len(tg),
                "weight": sum(float(t.get("weight") or 0) for t in tg),
                "pool": best.get("pool") or "",
                "regime": best.get("regime") or "",
            }
    except Exception:  # noqa: BLE001
        pass

    # --- 纸账户 ---
    acc = _status_read_json(root / "data" / "paper_account.json")
    if isinstance(acc, dict):
        try:
            hold = acc.get("holdings") or {}
            mv = sum(float(h.get("market_value") or 0) for h in hold.values())
            cash = float(acc.get("cash") or 0)
            init = float(acc.get("initial_cash") or 0) or (cash + mv)
            # 当日盈亏必须用「总资产 − 当日基线」的市值口径。
            # 存档里的 today_pnl 是**只累计卖出**的已实现口径：只买不卖时恒为 0，
            # 一旦拿它当"今日盈亏"展示，看到的就是一个跟真实账户无关的数
            # （2026-10-20 就把 -¥5.5 万的已实现亏损挂在首屏上误导排障）。
            day_base = acc.get("day_start_equity")
            equity = cash + mv
            today_pnl = (equity - float(day_base)) if day_base else float(acc.get("today_pnl") or 0)
            out["acct"] = {
                "equity": equity,
                "n_pos": len(hold),
                "pnl": today_pnl,
                "ret": (equity / init - 1.0) if init else 0.0,
            }
        except Exception:  # noqa: BLE001
            pass

    # --- 闭环心跳（停机在这里暴露） ---
    hb = _status_read_json(root / "data" / "loop_heartbeat.json")
    if isinstance(hb, dict):
        try:
            ts = datetime.fromisoformat(str(hb.get("ts")))
            out["loop"] = {
                "age_min": (datetime.now() - ts).total_seconds() / 60.0,
                "state": hb.get("state") or "",
                "mode": hb.get("guard_mode") or "",
                "n_pos": hb.get("n_positions"),
                "ts": str(hb.get("ts") or "").replace("T", " ")[:16],
            }
        except Exception:  # noqa: BLE001
            pass

    # --- 最近一次扫描批次 ---
    recs = _status_read_json(root / "data" / "prediction_history.json")
    if isinstance(recs, list) and recs:
        try:
            latest = max(recs, key=lambda r: str(r.get("scan_date") or ""))
            out["pred"] = {"date": str(latest.get("scan_date") or ""),
                           "n": int(latest.get("n_predictions") or 0)}
        except Exception:  # noqa: BLE001
            pass

    # --- 滚动 IC ---
    ic = _status_read_json(root / "data" / "rolling_ic_status.json")
    if isinstance(ic, dict):
        out["ic"] = {"mean_ic": ic.get("mean_ic"), "ir": ic.get("ir_mean_std"),
                     "last": ic.get("last_date"), "n": ic.get("total")}

    return out


def _fmt_wan(v: float) -> str:
    """元 → 万元（首屏空间有限）"""
    return f"¥{v / 1e4:,.1f}万"


def render_today_status() -> None:
    """渲染首屏「今日状态」卡片行 + 闭环停机警示"""
    d = _today_status_data()
    cards: list = []
    stale_min = None

    if d.get("plan"):
        p = d["plan"]
        cards.append({
            "label": "📅 交易计划",
            "value": p["date"][5:] if p["date"] else "—",
            "sub": f"{p['n']} 只 · 目标仓位 {p['weight'] * 100:.0f}%"
                   + (f" · {p['pool']}" if p["pool"] else ""),
            "tone": "plain",
        })

    if d.get("acct"):
        a = d["acct"]
        cards.append({
            "label": "💰 纸账户权益",
            "value": _fmt_wan(a["equity"]),
            "sub": f"持仓 {a['n_pos']} 只 · 累计 {a['ret'] * 100:+.2f}%",
            "tone": "up" if a["ret"] >= 0 else "down",
        })

    if d.get("loop"):
        lp = d["loop"]
        stale_min = lp["age_min"]
        thr_min = _loop_stale_threshold_min()
        # value 尽量短（定宽卡片单行显示）：明细放 sub
        if stale_min <= thr_min:
            val, tone = "运行中", "up"
        elif stale_min <= 24 * 60:
            val, tone = f"停 {stale_min / 60:.1f}h", "warn"
        else:
            val, tone = f"停 {stale_min / 60 / 24:.1f}天", "down"
        cards.append({
            "label": "🔁 交易闭环",
            "value": val,
            "sub": f"心跳 {str(lp.get('ts') or '')[5:]} · {lp['mode'] or '—'}",
            "tone": tone,
        })

    if d.get("pred"):
        pr = d["pred"]
        cards.append({
            "label": "🎯 最新预测",
            "value": pr["date"][5:] if pr["date"] else "—",
            "sub": f"{pr['n']} 只已入库",
            "tone": "plain",
        })

    if d.get("ic") and d["ic"].get("mean_ic") is not None:
        ic = d["ic"]
        cards.append({
            "label": "📈 滚动 IC",
            "value": f"{float(ic['mean_ic']):.3f}",
            "sub": f"IR {float(ic['ir'] or 0):.2f} · 截至 {str(ic['last'] or '')[5:]}",
            "tone": "up" if float(ic["mean_ic"]) > 0 else "down",
        })

    if cards:
        kpi_row(cards)

    # 停机警示：把"静默失效"变成看得见的一行
    if stale_min is not None and stale_min > 30:
        st.markdown(
            f'<div class="notice warn">⚠️ <b>交易闭环已停 {stale_min / 60:.1f} 小时</b>'
            f'（最后心跳 {d["loop"].get("ts") or "—"}）—— 期间不会自动下单，'
            f'请确认常驻进程（开机自启或手动 run_loop）是否在跑。</div>',
            unsafe_allow_html=True,
        )


# ========== 页面配置 ==========
st.set_page_config(
    page_title="选股神 AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 全局样式（Design System v2） ==========
# 设计原则：
#   1) 一套色板/圆角/阴影/间距贯穿全部页面，杜绝各页自绘风格。
#   2) 背景色/文字色尽量走 Streamlit 主题变量（--background-color / --text-color），
#      这样用户把看板切到深色主题时卡片不会变成刺眼的白底黑字。
#   3) 品牌色（紫）与涨跌色（红涨绿跌）**硬编码** —— 涨跌配色是 A 股铁律，
#      绝不能跟主题走，也不能用 st.metric 默认的欧美配色。
#   4) 类名全部向后兼容：.main-header / .sub-header / .section-title / .card-row /
#      .mini-card / .tag / .chart-box / .text-sub / .red|.green|.amber|.purple|.gray
#      均保留，老页面不改一行代码也能吃到新样式。
_THEME_CSS = """
<style>
/* ==================== 0. 设计令牌 ==================== */
:root{
    --brand:      #5b21b6;   /* 主品牌：深紫 */
    --brand-2:    #7c3aed;
    --brand-soft: #f5f3ff;
    --brand-line: #ddd6fe;

    --ink-1: var(--text-color, #0f172a);
    --ink-2: #475569;
    --ink-3: #64748b;
    --ink-4: #94a3b8;

    --surface:   var(--background-color, #ffffff);
    --surface-2: var(--secondary-background-color, #f8fafc);
    --line:      #e6e8f0;
    --line-soft: #f1f3f8;

    --up:   #dc2626;   /* A股：涨=红 */
    --down: #059669;   /* A股：跌=绿 */
    --warn: #d97706;
    --info: #3b82f6;

    --r-lg: 18px;
    --r-md: 14px;
    --r-sm: 10px;

    --sh-1: 0 1px 2px rgba(15,23,42,.05);
    --sh-2: 0 4px 16px rgba(15,23,42,.07);
    --sh-3: 0 10px 30px rgba(91,33,182,.14);

    --font: "Inter", -apple-system, "Segoe UI", "PingFang SC",
            "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}

/* ==================== 1. 页面底色与字体 ==================== */
html, body, [data-testid="stAppViewContainer"]{
    font-family: var(--font);
    -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"]{
    /* 两团极淡的氛围光，纯装饰，不干扰任何内容可读性 */
    background:
        radial-gradient(900px 420px at 8% -8%,  rgba(124,58,237,.07), transparent 60%),
        radial-gradient(760px 380px at 102% 4%, rgba(59,130,246,.06), transparent 62%),
        var(--surface-2);
}
/* 顶栏：半透明毛玻璃，避免和氛围底色割裂 */
header[data-testid="stHeader"]{
    background: rgba(255,255,255,.72);
    backdrop-filter: saturate(160%) blur(10px);
    border-bottom: 1px solid var(--line-soft);
}
/* 隐藏 Streamlit 顶部那条默认彩色装饰线 */
div[data-testid="stDecoration"]{ display:none; }
/* 本地单机看板不需要 Deploy / "Made with Streamlit" 这些产品化尾巴 */
[data-testid="stAppDeployButton"]{ display:none !important; }
footer{ display:none !important; }

/* 主内容区：限宽 + 收紧纵向间距
   宽屏（≥1600px）下 Streamlit 默认铺满整屏，卡片会被拉成很长的横条，
   内容又只有两行字 → 满屏留白，这就是"简陋感"的主要来源。 */
div[data-testid="stMainBlockContainer"]{
    max-width: 1360px;
    margin: 0 auto;
    /* ⚠️ 顶部必须给 Streamlit 固定顶栏留位（默认约 6rem）。
       曾把它改成 1.1rem，结果 hero 被半透明顶栏压住、标题上半截被切 ——
       页面一打开就是"坏的"。4rem 是实测能露出完整页头的最小安全值。 */
    padding-top: 4rem;
}
div[data-testid="stVerticalBlock"]{ gap: .8rem; }
/* ⚠️ 这里**刻意不做入场动画**：Streamlit 每次交互都会重跑脚本并重建容器，
   任何 fade/slide 动画都会跟着重放一次 —— 看着像整页在闪。
   动效只保留 :hover 过渡（见 .mini-card / 按钮），那些不会重放。 */
/* 数字统一等宽，避免跳动 */
body, div[data-testid="stMarkdownContainer"]{
    font-variant-numeric: tabular-nums;
}
hr{ border:none; border-top:1px solid var(--line-soft); margin:1.1rem 0; }

/* ==================== 2. 页头 Hero ==================== */
.hero{
    position:relative; display:flex; align-items:center; gap:.9rem;
    flex-wrap:wrap;              /* 徽标多/窗口窄时换行，别溢出卡片 */
    padding:.8rem 1.1rem; margin:.1rem 0 .95rem;
    border-radius:var(--r-lg);
    border:1px solid var(--brand-line);
    background:linear-gradient(115deg,#f6f4ff 0%,#ffffff 46%,#f3f8ff 100%);
    box-shadow:var(--sh-2);
    overflow:hidden;
}
.hero::after{
    content:""; position:absolute; right:-46px; top:-70px;
    width:230px; height:230px; border-radius:50%;
    background:radial-gradient(circle, rgba(124,58,237,.16), transparent 70%);
}
.hero-icon{
    width:46px; height:46px; flex:0 0 46px;
    display:flex; align-items:center; justify-content:center;
    font-size:1.5rem; border-radius:13px;
    background:linear-gradient(135deg,var(--brand),var(--brand-2));
    box-shadow:0 6px 16px rgba(91,33,182,.26);
}
.hero-text{ min-width:0; z-index:1; }
.hero-title{
    font-size:1.3rem; font-weight:800; color:#1e1b4b;
    letter-spacing:-.02em; line-height:1.25;
}
.hero-sub{
    font-size:.8rem; color:var(--ink-3); margin-top:.15rem; line-height:1.4;
}
.hero-badges{
    margin-left:auto; display:flex; gap:.35rem; flex-wrap:wrap;
    justify-content:flex-end; z-index:1;
    max-width:100%;
}
/* 兼容仍用旧写法的页面 */
.main-header{
    font-size:1.5rem !important; font-weight:800; color:#1e1b4b;
    margin:.1rem 0 .15rem; letter-spacing:-.02em;
}
.sub-header{ font-size:.85rem; color:var(--ink-3); margin-bottom:.9rem; }

/* ==================== 3. 胶囊徽标 ==================== */
.pill{
    display:inline-flex; align-items:center; gap:.3rem;
    padding:.3rem .68rem; border-radius:999px;
    font-size:.74rem; font-weight:600; line-height:1.2;
    border:1px solid; background:var(--surface);
    white-space:nowrap;
}
.pill.violet{ color:var(--brand);   border-color:var(--brand-line); background:var(--brand-soft); }
.pill.up    { color:#b91c1c;        border-color:#fecdd3;          background:#fff1f2; }
.pill.down  { color:#047857;        border-color:#a7f3d0;          background:#ecfdf5; }
.pill.gray  { color:var(--ink-2);   border-color:var(--line);       background:var(--surface-2); }
.pill.warn  { color:#b45309;        border-color:#fcd34d;          background:#fffbeb; }
.pill.info  { color:#1d4ed8;        border-color:#bfdbfe;          background:#eff6ff; }

/* ==================== 4. 节标题 ==================== */
.section-title{
    position:relative;
    font-size:1.02rem; font-weight:800; color:#1e293b;
    margin:1.35rem 0 .7rem; padding:0 0 .5rem .8rem;
    border-bottom:1px solid var(--line-soft);
    letter-spacing:-.01em;
}
.section-title::before{
    content:""; position:absolute; left:0; top:.12rem; bottom:.6rem;
    width:4px; border-radius:4px;
    background:linear-gradient(180deg,var(--brand),var(--brand-2));
}

/* ==================== 5. 卡片 ==================== */
/* 卡片组：**定宽网格**，不是 flex 平分。
   之前用 flex:1 让卡片平分整行，1680px 下每张 ~300px 宽、内容只有两行字，
   一片空旷 —— 看着"简陋"的根源就在这。现在限制单卡 172~228px，左对齐铺开。 */
.card-row{
    display:grid;
    grid-template-columns:repeat(auto-fill, minmax(172px, 228px));
    gap:.7rem;
    margin:.8rem 0;
    justify-content:start;
}
.mini-card{
    position:relative; min-width:0;
    background:var(--surface); border:1px solid var(--line);
    border-radius:var(--r-md); padding:.9rem 1rem .85rem;
    text-align:left; box-shadow:var(--sh-1); overflow:hidden;
    transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
/* 左侧 3px 语义色条，比顶部色带更"仪表盘"，也不会和 hover 抬升打架 */
.mini-card::before{
    content:""; position:absolute; left:0; top:.85rem; bottom:.85rem; width:3px;
    border-radius:0 3px 3px 0;
    background:linear-gradient(180deg,var(--brand),var(--brand-2));
}
.mini-card:hover{
    transform:translateY(-2px); box-shadow:var(--sh-2);
    border-color:var(--brand-line);
}
.mini-card.up::before   { background:linear-gradient(180deg,#dc2626,#f87171); }
.mini-card.down::before { background:linear-gradient(180deg,#059669,#34d399); }
.mini-card.warn::before { background:linear-gradient(180deg,#d97706,#fbbf24); }
.mini-card.plain::before{ background:var(--line); }
.mini-card .label{
    font-size:.75rem; font-weight:600; color:var(--ink-3);
    letter-spacing:.02em;
}
.mini-card .val{
    font-size:1.5rem; font-weight:800; color:var(--ink-1);
    margin-top:.2rem; letter-spacing:-.03em; line-height:1.15;
    /* 单行不折行：定宽卡片里长文本折成两行会撑高整排、破坏对齐 */
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.mini-card .sub{ font-size:.73rem; color:var(--ink-3); margin-top:.22rem; line-height:1.3; }

/* 通用面板 */
.panel{
    background:var(--surface); border:1px solid var(--line);
    border-radius:var(--r-md); padding:1rem 1.1rem;
    box-shadow:var(--sh-1); margin:.6rem 0;
}
.panel-title{
    font-size:.9rem; font-weight:700; color:#1e293b;
    margin:0 0 .55rem; display:flex; align-items:center; gap:.4rem;
}
/* st.container(border=True) 生成的容器：套上同一套白卡样式，
   避免"一行控件裸奔在灰底上"的半成品感 */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]){
    background:var(--surface);
    border:1px solid var(--line) !important;
    border-radius:var(--r-md);
    box-shadow:var(--sh-1);
    padding:.6rem .75rem;
}
/* 图表标题：只做标题，不再自己当白卡。
   ⚠️ 历史写法是「st.markdown 开 <div class="chart-box"> + 后面 st.markdown 关 </div>」，
   但两次 markdown 是两个独立 DOM 块，浏览器会把未闭合的 div 当场自动闭合 →
   结果只包住了 <h3>，图本身落在盒子外面，变成「标题一张卡 + 图一张卡」的框中框。
   真正的白卡由 stPlotlyChart 自己提供（见下方规则），这里只负责标题排版。 */
.chart-box{
    background:transparent; border:none; box-shadow:none;
    border-radius:0; padding:0; margin:.9rem 0 .35rem;
}
.chart-box h3{
    font-size:.92rem; font-weight:700; color:#1e293b;
    margin:0; padding-left:.6rem; line-height:1.4;
    border-left:3px solid var(--brand);
}
.text-sub{ font-size:.78rem; color:var(--ink-3); font-weight:500; line-height:1.45; }

/* 状态提示条（比 st.warning 更贴版式，用在本地面板） */
/* 用 block 而非 flex：flex 会把整段文字当一个可压缩的 flex item，
   中文长句会被拆成"停 23.9 小 / 时"这种断行。 */
.notice{
    display:block;
    padding:.62rem .9rem; margin:.45rem 0;
    border-radius:var(--r-sm); border:1px solid;
    font-size:.82rem; font-weight:500; line-height:1.55;
    word-break:normal; overflow-wrap:anywhere;
}
.notice.warn{ background:#fffbeb; border-color:#fde68a; color:#92400e; }
.notice.ok  { background:#ecfdf5; border-color:#a7f3d0; color:#065f46; }
.notice.info{ background:var(--brand-soft); border-color:var(--brand-line); color:#4c1d95; }

/* 空状态 */
.empty{
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:.3rem; padding:1.1rem 1rem; text-align:center;
    background:var(--surface-2); border:1px dashed var(--line);
    border-radius:var(--r-md); color:var(--ink-3); font-size:.85rem;
}
.empty .empty-icon{ font-size:1.8rem; opacity:.85; }

/* ==================== 6. 信号标签 ==================== */
/* A 股习惯：看多=红、看空=绿。类名与 signal_tag() 输出保持一致。 */
.tag-row{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.tag{
    display:inline-flex; align-items:center; gap:.3rem;
    padding:.34rem .72rem; border-radius:999px;
    font-size:.78rem; font-weight:600; white-space:nowrap;
    box-shadow:var(--sh-1);
}
.tag.bullish{ background:#fff1f2; border:1px solid #fecdd3; color:#b91c1c; }
.tag.bearish{ background:#ecfdf5; border:1px solid #a7f3d0; color:#047857; }
.tag.warn   { background:#fffbeb; border:1px solid #fcd34d; color:#b45309; }
.tag.info   { background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; }
/* 兼容历史写法 bull/bear（语义同上） */
.tag.bull{ background:#fff1f2; border:1px solid #fecdd3; color:#b91c1c; }
.tag.bear{ background:#ecfdf5; border:1px solid #a7f3d0; color:#047857; }

/* ==================== 6.5 个股分析仪表盘组件 ==================== */
/* 标的头：名称/代码 + 行业徽标 + 现价涨跌，一整条横向信息带，
   取代原来那行「名称 (代码) · 收盘 xx +0.12」的裸 section-title。 */
.stock-head{
    display:flex; align-items:center; gap:.75rem; flex-wrap:wrap;
    background:linear-gradient(135deg,#faf5ff 0%,#f5f3ff 55%,#ffffff 100%);
    border:1px solid var(--brand-line);
    border-radius:var(--r-md); padding:.8rem 1.05rem; margin:.2rem 0 .7rem;
    box-shadow:var(--sh-1);
}
.stock-head .sh-main{ min-width:0; }
.stock-head .sh-name{
    font-size:1.12rem; font-weight:800; color:var(--ink-1);
    letter-spacing:-.02em; line-height:1.25; white-space:nowrap;
}
.stock-head .sh-code{
    font-size:.73rem; color:var(--ink-3); margin-top:1px;
    font-variant-numeric:tabular-nums; white-space:nowrap;
}
.stock-head .sh-quote{
    margin-left:auto; display:flex; align-items:baseline; gap:.55rem;
    white-space:nowrap;
}
.stock-head .sh-price{
    font-size:1.5rem; font-weight:800; letter-spacing:-.03em;
    font-variant-numeric:tabular-nums;
}
.stock-head .sh-chg{ font-size:.85rem; font-weight:700; font-variant-numeric:tabular-nums; }

/* 评分面板：环形分 + 三条概率条 */
.score{ display:flex; align-items:center; gap:1.05rem; margin:.15rem 0 .1rem; }
.ring{
    --p:50; --c:var(--brand);
    position:relative; flex:0 0 96px; width:96px; height:96px; border-radius:50%;
    background:conic-gradient(var(--c) calc(var(--p) * 1%), #eaedf3 0);
    display:grid; place-items:center;
}
.ring::before{
    content:""; position:absolute; inset:8px; border-radius:50%;
    background:var(--surface);
}
.ring-in{ position:relative; text-align:center; line-height:1; }
.ring .num{
    display:block; font-size:1.65rem; font-weight:800;
    letter-spacing:-.04em; color:var(--c); font-variant-numeric:tabular-nums;
}
.ring .cap{ display:block; font-size:.62rem; color:var(--ink-3); margin-top:3px; }
.bars{ flex:1 1 auto; min-width:0; display:flex; flex-direction:column; gap:.42rem; }
.bar .bar-head{
    display:flex; justify-content:space-between; align-items:baseline;
    font-size:.74rem; color:var(--ink-2); margin-bottom:3px;
}
.bar .bar-track{ height:7px; border-radius:999px; background:#eaedf3; overflow:hidden; }
.bar .bar-fill{ height:100%; border-radius:999px; transition:width .4s ease; }
.score-foot{
    font-size:.74rem; color:var(--ink-3); margin-top:.55rem;
    padding-top:.5rem; border-top:1px dashed var(--line);
    overflow-wrap:anywhere;
}

/* 紧凑 KV 小格网格：基本面/风险指标都用它，取代一行行裸文本。
   最小宽 96px 是实测值：左列（st.columns 半分 + panel 内边距）约 535px，
   用 112px 只会排 4 列，5 个指标被挤成 4+1；96px 正好排到 5 列排满一行。 */
.kv{ display:grid; grid-template-columns:repeat(auto-fill, minmax(96px, 1fr)); gap:.42rem .5rem; }
.kv-i{
    background:var(--surface-2); border:1px solid var(--line);
    border-radius:var(--r-sm); padding:.42rem .55rem; min-width:0;
}
.kv-i .k{ font-size:.67rem; color:var(--ink-3); white-space:nowrap; }
.kv-i .v{
    font-size:.9rem; font-weight:700; color:var(--ink-1); margin-top:1px;
    font-variant-numeric:tabular-nums; white-space:nowrap;
    overflow:hidden; text-overflow:ellipsis;
}

/* 交易建议：纵向行块，比横排 5 个等宽格更好扫 */
.trade-list{ display:flex; flex-direction:column; gap:.32rem; }
.trade-row{
    display:flex; justify-content:space-between; align-items:baseline;
    padding:.4rem .62rem; border-radius:8px; background:var(--surface-2);
    font-size:.8rem;
}
.trade-row .k{ color:var(--ink-3); white-space:nowrap; }
.trade-row .v{ font-weight:700; color:var(--ink-1); font-variant-numeric:tabular-nums; text-align:right; }

/* 紧凑小表：多周期统计 / 相对强弱。同行横向比较靠它，比铺 20 张卡片清楚得多 */
.mini-table{ width:100%; border-collapse:collapse; font-size:.79rem; }
.mini-table th{
    text-align:right; font-weight:600; color:var(--ink-3);
    font-size:.68rem; padding:.3rem .5rem; border-bottom:1px solid var(--line);
    white-space:nowrap;
}
.mini-table th:first-child, .mini-table td:first-child{ text-align:left; }
.mini-table td{
    text-align:right; padding:.4rem .5rem; border-bottom:1px dashed #eef1f6;
    font-variant-numeric:tabular-nums; color:var(--ink-1); white-space:nowrap;
}
.mini-table tbody tr:last-child td{ border-bottom:none; }
.mini-table tbody tr:hover td{ background:#fafaff; }

/* ==================== 7. 按钮 ==================== */
button[data-testid="stBaseButton-primary"],
button[kind="primary"]{
    background:linear-gradient(135deg,var(--brand),var(--brand-2)) !important;
    border:none !important; border-radius:var(--r-sm) !important;
    color:#fff !important; font-weight:600 !important;
    box-shadow:0 4px 14px rgba(91,33,182,.28);
    transition:transform .16s ease, box-shadow .16s ease, filter .16s ease;
}
button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover{
    transform:translateY(-1px); filter:brightness(1.06);
    box-shadow:0 8px 20px rgba(91,33,182,.34);
}
button[data-testid="stBaseButton-secondary"],
button[kind="secondary"]{
    background:var(--surface) !important;
    border:1px solid var(--line) !important;
    border-radius:var(--r-sm) !important;
    color:var(--ink-2) !important; font-weight:500 !important;
    box-shadow:var(--sh-1);
    transition:transform .16s ease, border-color .16s ease, color .16s ease;
}
button[data-testid="stBaseButton-secondary"]:hover,
button[kind="secondary"]:hover{
    transform:translateY(-1px);
    border-color:var(--brand-line) !important;
    color:var(--brand) !important;
    background:var(--brand-soft) !important;
}

/* ==================== 8. 页签 / 指标 / 表格 ==================== */
div[data-testid="stTabs"] button[data-testid="stTab"]{
    border-radius:var(--r-sm) var(--r-sm) 0 0;
    padding:.5rem .9rem; font-weight:600; color:var(--ink-3);
    transition:color .15s ease, background .15s ease;
}
div[data-testid="stTabs"] button[data-testid="stTab"]:hover{ color:var(--brand); background:var(--brand-soft); }
div[data-testid="stTabs"] button[data-testid="stTab"][aria-selected="true"]{ color:var(--brand); }
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
div[data-testid="stTabs"] div[role="tablist"] div[aria-hidden="true"]{
    background-color:var(--brand) !important;
}

div[data-testid="stMetric"]{
    background:var(--surface); border:1px solid var(--line);
    border-radius:var(--r-md); padding:.7rem .9rem; box-shadow:var(--sh-1);
}
div[data-testid="stMetric"] label{
    font-size:.75rem !important; color:var(--ink-3) !important; font-weight:600;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{
    font-size:1.22rem !important; font-weight:800 !important; color:var(--ink-1);
    letter-spacing:-.02em;
}

div[data-testid="stDataFrame"], div[data-testid="stTable"]{
    border:1px solid var(--line); border-radius:var(--r-md);
    overflow:hidden; box-shadow:var(--sh-1); background:var(--surface);
}
table{ border-collapse:separate; border-spacing:0; }
table thead th{
    background:var(--brand-soft) !important; color:#4c1d95 !important;
    font-weight:700 !important; font-size:.78rem !important;
    border-bottom:1px solid var(--brand-line) !important;
}
table tbody tr:nth-child(even){ background:var(--surface-2); }
table tbody tr:hover{ background:var(--brand-soft); }

/* 图表容器统一"白卡"化：只留一层圆角+边框，不加内边距
   （plotly 自己已有边距，再加 padding 会出现"框中框"） */
div[data-testid="stPlotlyChart"], .stPlotlyChart{
    background:var(--surface); border:1px solid var(--line);
    border-radius:var(--r-md); padding:0; box-shadow:var(--sh-1);
    overflow:hidden;
}

/* 提示条：圆角 + 柔和 */
div[data-testid="stAlert"]{
    border-radius:var(--r-sm); border:1px solid rgba(15,23,42,.06);
    box-shadow:var(--sh-1); font-size:.86rem;
}
div[data-testid="stExpander"]{
    border:1px solid var(--line) !important; border-radius:var(--r-md) !important;
    background:var(--surface); box-shadow:var(--sh-1); overflow:hidden;
}

/* 表单控件强调色 */
input[type="radio"], input[type="checkbox"]{ accent-color:var(--brand); }
div[data-baseweb="select"] > div{ border-radius:var(--r-sm) !important; }

/* ==================== 9. 侧边栏 ==================== */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#ffffff 0%,#fbfaff 55%,#f6f4ff 100%);
    border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] > div{ padding-top:.9rem; }
.brand{
    display:flex; align-items:center; gap:.7rem;
    padding:.85rem .95rem; border-radius:var(--r-md);
    background:linear-gradient(135deg,#5b21b6,#7c3aed);
    box-shadow:0 8px 22px rgba(91,33,182,.26);
    margin-bottom:.35rem;
}
.brand-logo{
    width:40px; height:40px; flex:0 0 40px; border-radius:12px;
    display:flex; align-items:center; justify-content:center; font-size:1.35rem;
    background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.3);
}
.brand-name{ color:#fff; font-size:1.08rem; font-weight:800; letter-spacing:-.01em; line-height:1.15; }
.brand-tag{ color:rgba(255,255,255,.78); font-size:.7rem; font-weight:500; margin-top:2px; }
.nav-hint{
    font-size:.68rem; font-weight:700; letter-spacing:.12em;
    color:var(--ink-4); text-transform:uppercase; margin:.85rem 0 .35rem .2rem;
}
/* Streamlit 即便 label_visibility="collapsed" 仍会把「导航」渲染成可见文本，
   跟下面自绘的 .nav-hint 重复成两个"导航" —— 这里直接隐掉控件自带 label。
   （实测 DOM：label[data-testid="stWidgetLabel"] > span > p） */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label[data-testid="stWidgetLabel"]{
    display:none !important;
}
/* 导航项：只命中真正的选项 label（data-baseweb="radio"），
   否则上面的控件 label 也会被当导航项排版。 */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]{
    position:relative;
    display:flex; align-items:center; gap:.5rem;
    padding:.5rem .7rem; margin:2px 0; border-radius:var(--r-sm);
    border:1px solid transparent; cursor:pointer;
    font-size:.88rem; color:var(--ink-2); font-weight:500;
    transition:background .15s ease, color .15s ease, border-color .15s ease;
}
/* 隐掉原生圆点：一列大圆圈很像老式表单，选中态改用底色+左侧色条表达 */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child{
    display:none;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:hover{
    background:var(--brand-soft); border-color:var(--brand-line); color:var(--brand);
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked){
    background:linear-gradient(90deg,#f2eeff,#ffffff 92%);
    border-color:var(--brand-line); color:var(--brand); font-weight:700;
    box-shadow:var(--sh-1);
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked)::before{
    content:""; position:absolute; left:0; top:.35rem; bottom:.35rem; width:3px;
    border-radius:0 3px 3px 0;
    background:linear-gradient(180deg,var(--brand),var(--brand-2));
}
/* 选项文字统一行高，避免 emoji 撑高 */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] p{
    margin:0; line-height:1.4;
}
.side-card{
    background:var(--surface); border:1px solid var(--line);
    border-radius:var(--r-md); padding:.65rem .8rem; box-shadow:var(--sh-1);
    margin:.5rem 0;
}
.side-row{
    display:flex; align-items:center; justify-content:space-between;
    gap:.5rem; font-size:.78rem; padding:2px 0; color:var(--ink-2);
}
.side-foot{
    font-size:.68rem; color:var(--ink-4); text-align:center;
    padding:.5rem 0 .2rem; line-height:1.5;
}

/* ==================== 10. 滚动条 ==================== */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{
    background:#cbd5e1; border-radius:8px;
    border:2px solid transparent; background-clip:padding-box;
}
::-webkit-scrollbar-thumb:hover{ background:#94a3b8; background-clip:padding-box; }

/* ==================== 11. 颜色快捷类 ==================== */
.red   { color:#dc2626; font-weight:600; }
.green { color:#059669; font-weight:600; }
.amber { color:#d97706; font-weight:600; }
.purple{ color:#7c3aed; font-weight:600; }
.gray  { color:#6b7280; }
.mono  { font-variant-numeric: tabular-nums; }
</style>
"""

st.markdown(_THEME_CSS, unsafe_allow_html=True)


# ========== 策略信誉引擎可视化 ==========
def load_credibility_snapshot() -> dict:
    """读策略信誉引擎的持久化快照（fail-safe，看板只读不写）。

    来源：交易闭环 guard 在每个有效 tick 调 ``sync_credibility_from_paper`` 后，
    把 ``StrategyCredibility`` 落盘到 ``data/credibility_state.json``。文件缺失 /
    损坏 / 空都返回 ``{}``，由上层决定展示空状态——绝不能让看板 500。
    """
    try:
        from config.settings import DATA_DIR
        cred_path = Path(DATA_DIR) / "credibility_state.json"
    except Exception:  # noqa: BLE001
        cred_path = Path(__file__).resolve().parent.parent.parent / "data" / "credibility_state.json"
    if not cred_path.exists():
        return {}
    try:
        raw = json.loads(cred_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    scores = raw.get("scores") or {}
    if not isinstance(scores, dict) or not scores:
        return {}
    return {
        "scores": scores,
        "halflife": raw.get("halflife"),
        "floor": raw.get("floor"),
        "cap": raw.get("cap"),
        "updated_at": raw.get("updated_at"),
    }


def _fmt_money_short(v) -> str:
    """把元格式化成「亿 / 万」短标签（看板表格用）。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if x >= 1e8:
        return f"{x / 1e8:.2f}亿"
    if x >= 1e4:
        return f"{x / 1e4:.1f}万"
    return f"{x:,.0f}"


def load_portfolio_risk_view(account_path=None, data_dir=None) -> dict:
    """组合风险视图：1 日 VaR / 平均相关性 / 各标的 ADV 单笔上限。

    口径要点：VaR 的权重是「市值 / 总资产」，不是「市值 / 持仓合计」。
    低仓位本身就是低风险，归一化到持仓合计会把「只用 18% 仓位」误算成满仓风险。

    fail-safe：任何异常返回 ``{}``，看板绝不因风险计算失败而崩；只读，不写状态。
    """
    try:
        from pathlib import Path as _P

        try:
            from config.settings import ROOT_DIR as _ROOT
        except Exception:  # noqa: BLE001
            _ROOT = _P(__file__).resolve().parent.parent.parent
        root = _P(_ROOT)

        acct_path = (_P(account_path) if account_path
                     else root / "data" / "paper_account.json")
        acc = _status_read_json(acct_path)
        if not isinstance(acc, dict):
            return {}
        holdings = acc.get("holdings") or {}
        if not isinstance(holdings, dict) or not holdings:
            return {}

        cash = float(acc.get("cash") or 0)
        codes, values = [], []
        for code, h in holdings.items():
            if not isinstance(h, dict):
                continue
            v = float(h.get("market_value") or 0)
            if v <= 0:
                v = float(h.get("quantity") or 0) * float(h.get("last_price") or 0)
            if v > 0:
                codes.append(str(code))
                values.append(v)
        mv = sum(values)
        equity = cash + mv
        if equity <= 0 or len(codes) == 0:
            return {}

        from src.trading.risk_model import PortfolioRiskMonitor
        from src.trading.liquidity import AdvProvider, PARTICIPATION_DEFAULT

        monitor = (PortfolioRiskMonitor(data_dir=data_dir)
                   if data_dir else PortfolioRiskMonitor())
        advp = AdvProvider(data_dir=data_dir) if data_dir else AdvProvider()

        res = monitor.var_pct(codes, [v / equity for v in values])

        cap = 0.025
        try:
            from src.trading.guard import GuardConfig
            gp = root / "config" / "trading_guard.yaml"
            if gp.exists():
                cap = float(GuardConfig.from_yaml(str(gp)).max_portfolio_var_pct)
        except Exception:  # noqa: BLE001
            pass

        rows = []
        for c, v in zip(codes, values):
            a = advp.adv_amount(c)
            rows.append({
                "code": c,
                "weight": v / equity,
                "adv": a,
                "cap": (advp.max_order_value(c, PARTICIPATION_DEFAULT)
                        if a else None),
            })
        return {
            "equity": equity,
            "invested_pct": mv / equity,
            "var_pct": float(res[0]) if res else None,
            "avg_corr": float(res[1]) if res else None,
            "var_cap": cap,
            "rows": rows,
        }
    except Exception:  # noqa: BLE001
        return {}


def render_portfolio_risk_panel() -> None:
    """🛡️ 组合风险：把「相关性 VaR + 流动性上限」摊在看板上。"""
    st.markdown(
        '<div class="section-title">🛡️ 组合风险 · VaR 与流动性</div>',
        unsafe_allow_html=True,
    )
    view = load_portfolio_risk_view()
    if not view or not view.get("rows"):
        st.caption("暂无持仓或历史数据，组合风险视图不可用。")
        return

    c1, c2, c3 = st.columns(3)
    var_pct = view.get("var_pct")
    c1.metric("组合 1日95% VaR", f"{var_pct * 100:.2f}%" if var_pct is not None else "N/A")
    corr = view.get("avg_corr")
    c2.metric("平均相关性", f"{corr:.2f}" if corr is not None else "N/A")
    c3.metric("仓位 / 总资产", f"{view['invested_pct'] * 100:.0f}%")

    lines = ["| 标的 | 权重 | 近20日ADV | 单笔上限(1%) |", "|---|---|---|---|"]
    for r in view["rows"]:
        lines.append(
            f"| {r['code']} | {r['weight'] * 100:.1f}% | "
            f"{_fmt_money_short(r['adv'])} | {_fmt_money_short(r['cap'])} |"
        )
    st.markdown("\n".join(lines))
    st.caption(
        f"VaR 口径：参数法 95% 单尾、占总资产；上限 {view.get('var_cap', 0) * 100:.1f}%。"
        "超上限时 guard 会按比例降低**新开仓**强度（不清仓）。"
    )


def render_credibility_panel() -> None:
    """🛡️ 策略信誉引擎：把每个策略标签的「方向一致性(EWMA) → 权重」表现可视化。

    数据来自 ``load_credibility_snapshot()``（guard 落盘的只读快照）。
    """
    st.markdown('<div class="section-title">🛡️ 策略信誉引擎 · 权重表现</div>', unsafe_allow_html=True)
    st.caption(
        "信誉引擎用每笔已平仓交易的「信号方向 vs 实际盈亏」一致性，以 EWMA 为每个策略标签"
        "维护一个信誉分，再映射成多策略集成的权重：近期越准 → 权重越高、话语权越大；"
        "未知标签按中性 1.0 处理。"
    )

    snap = load_credibility_snapshot()
    scores = snap.get("scores", {})

    if not scores:
        st.markdown(
            '<div class="empty"><div class="empty-icon">🛡️</div>'
            '<div>策略信誉引擎已启用，但尚无已平仓战绩可展示。</div>'
            '<div style="font-size:.75rem;color:#94a3b8;">'
            '闭环交易产生平仓后，这里会显示各策略标签的方向一致性、权重与样本数。'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    # 数据新鲜度提示（闭环未运行时 = 最近一次落盘）
    up = snap.get("updated_at")
    if up:
        try:
            age = (datetime.now() - datetime.fromisoformat(up)).total_seconds()
            if age < 60:
                age_s = f"{age:.0f} 秒前"
            elif age < 3600:
                age_s = f"{age / 60:.0f} 分钟前"
            else:
                age_s = f"{age / 3600:.1f} 小时前"
            st.caption(f"📌 快照更新于 {age_s}（闭环未运行时为最近一次落盘）")
        except Exception:  # noqa: BLE001
            pass

    # 概览统计
    n_tags = len(scores)
    total_n = sum(int(s.get("n", 0)) for s in scores.values())
    boosted = [t for t, s in scores.items() if float(s.get("weight", 1.0)) > 1.0]
    penalized = [t for t, s in scores.items() if float(s.get("weight", 1.0)) < 1.0]
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("已观测策略", n_tags)
    o2.metric("累计样本", total_n)
    o3.metric("权重↑被信任", len(boosted))
    o4.metric("权重↓被压制", len(penalized))

    # 每标签一张卡（按权重降序）
    tags = sorted(scores.keys(), key=lambda t: float(scores[t].get("weight", 1.0)), reverse=True)
    hl = snap.get("halflife") or 20
    for tag in tags:
        s = scores[tag]
        score = float(s.get("score", 0.5))
        weight = float(s.get("weight", 1.0))
        n = int(s.get("n", 0))
        arrow = "▲" if weight > 1.0 else ("▼" if weight < 1.0 else "＝")
        w_color = "#10b981" if weight > 1.0 else ("#ef4444" if weight < 1.0 else "#64748b")
        score_pct = max(0.0, min(1.0, score)) * 100
        bar_color = "#5b21b6" if score >= 0.5 else "#ef4444"
        st.markdown(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;'
            f'padding:0.8rem 1rem;margin-bottom:0.6rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<b style="color:#1e293b;font-size:1rem;">{tag}</b>'
            f'<span style="color:{w_color};font-weight:700;">权重 {weight:.2f} {arrow}</span></div>'
            f'<div style="font-size:0.72rem;color:#64748b;margin:0.4rem 0 0.2rem;">方向一致性（EWMA）</div>'
            f'<div style="background:#e2e8f0;border-radius:6px;height:8px;width:100%;overflow:hidden;">'
            f'<div style="background:{bar_color};height:8px;width:{score_pct:.0f}%;border-radius:6px;"></div></div>'
            f'<div style="font-size:0.72rem;color:#64748b;margin-top:0.3rem;">'
            f'信誉分 <b style="color:#1e293b;">{score:.2f}</b> · 集成权重 <b style="color:#1e293b;">{weight:.2f}</b>'
            f' · 样本 {n} · 半衰期 {hl} 笔</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ========== 侧边栏 ==========
with st.sidebar:
    # 预测口径 / 调度开关：在侧边栏里算一次，全局所有页面（含页头徽标）复用。
    # 必须**如实反映运行配置**，不写死"集成"，否则又是"开关开着其实没生效"那类静默失效。
    _ens = bool(getattr(getattr(app_config, "ml", None), "use_ensemble", False))
    _sched_on = bool(getattr(getattr(app_config, "scheduler", None), "enabled", False))
    _engine_tag = "XGBoost + Alpha + LSTM 超级集成" if _ens else "XGBoost 单模型引擎"

    st.markdown(f"""
    <div class="brand">
        <div class="brand-logo">📈</div>
        <div>
            <div class="brand-name">选股神 AI</div>
            <div class="brand-tag">{_engine_tag}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "🏠 概览"

    page_options = ["🏠 概览", "🔍 AI选股", "📊 个股分析", "📡 实时监控", "🤖 AI 助手", "💹 自动交易", "📊 业绩归因", "📐 量化模型", "📈 模型稳定性", "⚙️ 设置"]

    st.markdown('<div class="nav-hint">导航</div>', unsafe_allow_html=True)
    page = st.radio(
        "导航",
        page_options,
        label_visibility="collapsed",
        key="_nav_radio",
        index=page_options.index(st.session_state.page)
    )
    st.session_state.page = page

    # ---- 看板启动时自动拉起「收盘后自动化调度」（若已启用）----
    # 幂等调用：首次脚本运行（有人打开看板）即拉起，之后每次重跑因 is_running()
    # 短路而无开销。详见模块级 _ensure_scheduler_running()。
    _ensure_scheduler_running()

    ready = predictor.is_ready
    s_cls = "green" if ready else "red"
    dot_clr = "#10b981" if ready else "#ef4444"
    st.markdown(f"""
    <div class="side-card">
        <div class="side-row">
            <span style="display:flex;align-items:center;gap:6px;">
                <span style="width:8px;height:8px;border-radius:50%;background:{dot_clr};display:inline-block;"></span>
                模型引擎
            </span>
            <b class="{s_cls}">{'就绪' if ready else '未训练'}</b>
        </div>
        <div class="side-row"><span>⚡ 预测口径</span>
            <b>{'超级集成' if _ens else '单模型 XGBoost'}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏大盘速览（任何页面都可见）
    #
    # 性能关键：这里原本无条件调用 _cached_market_overview()。Streamlit 的
    # expander 即便折叠也会执行内部代码，于是**每次首屏都要联网拉行情**
    # （含 akshare 导入 1.9s + 网络往返，网络差时能拖 5–10 秒）。
    # 改成按需：默认不拉，用户点按钮才加载，结果缓存在 session_state。
    with st.expander("📊 大盘速览", expanded=False):
        _mk = st.session_state.get("_mk_overview")
        if _mk is None:
            st.caption("点击加载实时大盘行情（默认不自动请求，避免拖慢首屏）")
            if st.button("🔄 加载大盘行情", key="mk_load_btn", use_container_width=True):
                with st.spinner("获取行情…"):
                    try:
                        _mk_data = _cached_market_overview()
                    except Exception:  # noqa: BLE001
                        _mk_data = None
                    # 失败时保持 None：用户可以直接再点一次重试
                    st.session_state["_mk_overview"] = _mk_data
                st.rerun()
        else:
            try:
                _idx = _mk.get("indices")
                if _idx is not None and len(_idx) > 0:
                    for _it in _idx.head(4).itertuples(index=False):
                        _p = float(_it.涨跌幅)
                        _c = "#dc2626" if _p >= 0 else "#059669"
                        _a = "▲" if _p >= 0 else "▼"
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'align-items:center;font-size:0.78rem;padding:3px 0;">'
                            f'<span style="color:#475569;">{_it.名称}</span>'
                            f'<span style="color:{_c};font-weight:700;">'
                            f'{_a} {_p:+.2f}%</span></div>',
                            unsafe_allow_html=True,
                        )
                    _b = _mk.get("breadth", {})
                    if _b:
                        st.caption(
                            f"🔴{_b.get('上涨', 0)} 🟢{_b.get('下跌', 0)} "
                            f"💰{_mk.get('amount_yi', 0):,.0f}亿"
                        )
                else:
                    st.caption("行情源暂不可用")
            except Exception:
                st.caption("行情源暂不可用")
            if st.button("🔄 刷新", key="mk_refresh_btn"):
                st.session_state.pop("_mk_overview", None)
                st.rerun()

    st.markdown(
        f'<div class="side-foot">⏰ {datetime.now().strftime("%Y-%m-%d %H:%M")}'
        f'<br>选股神 AI · v2 界面</div>',
        unsafe_allow_html=True,
    )

# ========== 🏠 概览页 ==========
if page == "🏠 概览":
    # 徽标如实反映运行时状态（不做假：未就绪就显示未就绪）
    _pool_n = len(_cached_stock_list())
    render_hero(
        "选股神 AI 决策看板",
        "AI 驱动的 A 股智能选股系统 — 选股 · 计划 · 交易 · 进化 全链路闭环",
        icon="📈",
        badges=[
            ("violet", f"股票池 {_pool_n}"),
            # 状态徽标用中性紫/灰，不用红绿 —— 红色 pill 在 A 股语境里容易被读成"涨/看多"
            ("violet" if predictor.is_ready else "gray",
             "模型就绪" if predictor.is_ready else "模型未训练"),
            ("violet", "超级集成" if _ens else "单模型"),
            ("info" if _sched_on else "gray", "调度启用" if _sched_on else "调度停用"),
        ],
    )

    # ===== 今日状态（本地实数据：计划/账户/闭环心跳/预测/IC）=====
    # 这里刻意取代了原来那排"股票池/模型状态/预测口径/决策树/调度"静态卡：
    # 那些信息在页头徽标里已经有了，重复占一屏只会稀释重点；
    # 换成真实运营数字后，首屏才有"仪表盘"的密度。
    render_today_status()

    # ===== 大盘实时涨跌（主界面顶部）=====
    render_market_ticker()

    # 市场热点
    st.markdown('<div class="section-title">🔥 市场热点板块</div>', unsafe_allow_html=True)

    # 性能关键：热点板块同样要联网（含 akshare 导入 ~1.9s）。
    # 首屏先不拉，给一个加载按钮；点过一次后结果缓存在 session_state。
    if st.session_state.get("_hotspots") is None:
        # 空状态也要"有内容"：说清楚为什么空、点了会发生什么，
        # 而不是甩一行灰字 caption（那正是"简陋"的来源）。
        st.markdown(
            '<div class="empty">'
            '<div class="empty-icon">🔥</div>'
            '<div>板块行情需要联网拉取（首次约 2 秒），默认不自动加载以免拖慢首屏</div>'
            '<div style="font-size:.75rem;color:#94a3b8;">'
            '加载一次后本会话内复用缓存，不会重复请求</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("🔥 加载市场热点", key="hotspots_load_btn"):
            with st.spinner("获取板块行情…"):
                try:
                    st.session_state["_hotspots"] = _cached_hotspots()
                except Exception:  # noqa: BLE001
                    st.session_state["_hotspots"] = None
            st.rerun()
    else:
        try:
            hotspots = st.session_state["_hotspots"]
            if len(hotspots) > 0:
                # 排名概览卡片 (前5)
                top5 = hotspots.head(5)
                overview_cols = st.columns(5)
                for i, (_, row) in enumerate(top5.iterrows()):
                    with overview_cols[i]:
                        pct = row.get("涨跌幅", 0)
                        inflow = row.get("净流入", 0)
                        up_n = int(row.get("上涨家数", 0))
                        down_n = int(row.get("下跌家数", 0))
                        c_cls = "red" if pct >= 0 else "green"
                        inflow_cls = "red" if inflow >= 0 else "green"
                        inflow_str = f"{inflow:.1f}亿" if abs(inflow) < 100 else f"{inflow:.0f}亿"
                        sector_name = row.get("板块", row.get("板块名称", "?"))
                        lead_stock = row.get("领涨股", "")
                        lead_pct = row.get("领涨股-涨跌幅", 0)
                        st.markdown(f"""
                        <div style="text-align:center;padding:8px 4px;border-radius:10px;background:#fff;border:1px solid #e2e8f0;">
                            <div style="font-weight:700;color:#1e293b;font-size:0.85rem;">#{i+1} {sector_name[:5]}</div>
                            <div class="{c_cls}" style="font-size:1.3rem;font-weight:700;margin:4px 0;">{pct:+.2f}%</div>
                            <div class="{inflow_cls}" style="font-size:0.7rem;">💰{inflow_str}</div>
                            <div style="font-size:0.7rem;color:#94a3b8;">涨{up_n}跌{down_n} · {lead_stock[:4]} {lead_pct:+.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # 详情表：前15 + 图
                tab1, tab2 = st.tabs(["📋 行业排名表", "📊 资金流向"])

                with tab1:
                    show_n = st.slider("显示前", 5, min(30, len(hotspots)), 15, 5, key="sector_topn")
                    display = hotspots.head(show_n).copy()
                    # 涨跌比
                    display["涨跌比"] = display.apply(
                        lambda r: f"{int(r['上涨家数'])}/{int(r['下跌家数'])}" if r['上涨家数'] + r['下跌家数'] > 0 else "—", axis=1
                    )
                    display["成交额"] = display["总成交额"].apply(lambda x: f"{x/1e8:.1f}亿" if x else "—")
                    display["净流入"] = display["净流入"].apply(lambda x: f"{x:+.1f}亿")
                    display["涨跌幅%"] = display["涨跌幅"].apply(lambda x: f"{x:+.2f}")
                    display["领涨"] = display.apply(
                        lambda r: f"{r['领涨股'][:6]} {r['领涨股-涨跌幅']:+.1f}%", axis=1
                    )

                    show_cols = ["板块", "涨跌幅%", "净流入", "成交额", "涨跌比", "领涨"]
                    show = display[show_cols].copy()
                    show.index = range(1, len(show) + 1)
                    show.index.name = "排名"

                    st.dataframe(
                        show,
                        use_container_width=True,
                        height=(show_n + 1) * 35 + 3,
                        column_config={
                            "涨跌幅%": st.column_config.NumberColumn("涨跌幅%", format="%.2f %%"),
                        }
                    )

                with tab2:
                    inflow_top = hotspots.nlargest(10, "净流入").copy()
                    outflow_top = hotspots.nsmallest(10, "净流入").copy()

                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.markdown('<div style="font-weight:600;color:#dc2626;margin-bottom:6px;">🔴 资金净流入 Top 10（亿）</div>', unsafe_allow_html=True)
                        fig_in = go.Figure(data=[go.Bar(
                            x=inflow_top["净流入"].values,
                            y=inflow_top["板块"].values,
                            orientation="h",
                            marker_color=["#dc2626" if v >= 0 else "#059669" for v in inflow_top["净流入"]],
                            text=[f"{v:+.1f}亿" for v in inflow_top["净流入"]],
                            textposition="outside",
                            textfont=dict(color="#1e293b", size=11)
                        )])
                        fig_in.update_layout(height=300, template="plotly_white", margin=dict(l=10, r=60, t=10, b=10),
                                             xaxis_title="净流入（亿）")
                        st.plotly_chart(fig_in, use_container_width=True)

                    with c_right:
                        st.markdown('<div style="font-weight:600;color:#059669;margin-bottom:6px;">🟢 资金净流出 Top 10（亿）</div>', unsafe_allow_html=True)
                        fig_out = go.Figure(data=[go.Bar(
                            x=[-v for v in outflow_top["净流入"].values],
                            y=outflow_top["板块"].values,
                            orientation="h",
                            marker_color="#059669",
                            text=[f"{v:+.1f}亿" for v in outflow_top["净流入"]],
                            textposition="outside",
                            textfont=dict(color="#1e293b", size=11)
                        )])
                        fig_out.update_layout(height=300, template="plotly_white", margin=dict(l=10, r=60, t=10, b=10),
                                              xaxis_title="净流出（亿）")
                        st.plotly_chart(fig_out, use_container_width=True)

                    # 涨跌幅 vs 资金散点图
                    st.markdown('<div style="font-weight:600;margin:12px 0 6px 0;">📈 涨跌幅 vs 资金净流入（气泡大小 = 成交额）</div>', unsafe_allow_html=True)
                    top30 = hotspots.head(30).copy()
                    fig_scatter = go.Figure(data=[go.Scatter(
                        x=top30["涨跌幅"],
                        y=top30["净流入"],
                        mode="markers+text",
                        text=top30["板块"].values,
                        textposition="top center",
                        textfont=dict(size=9, color="#475569"),
                        marker=dict(
                            size=top30["总成交额"].apply(lambda x: max(8, min(40, x / 5e8))),
                            color=top30["涨跌幅"],
                            colorscale=["#059669", "#fbbf24", "#dc2626"],
                            showscale=True,
                            colorbar=dict(title="涨跌幅%", thickness=10),
                            line=dict(width=1, color="#fff")
                        ),
                        hovertemplate="<b>%{text}</b><br>涨跌: %{x:+.2f}%<br>净流入: %{y:+.1f}亿<extra></extra>"
                    )])
                    fig_scatter.add_hline(y=0, line_dash="dash", line_color="#94a3b8", opacity=0.5)
                    fig_scatter.add_vline(x=0, line_dash="dash", line_color="#94a3b8", opacity=0.5)
                    fig_scatter.update_layout(height=420, template="plotly_white",
                                             xaxis_title="涨跌幅 (%)", yaxis_title="资金净流入（亿）",
                                             margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as e:
            st.info(f"热点数据获取中… ({e})")

    st.markdown("---")
    st.markdown('<div class="section-title">🚀 快速开始</div>', unsafe_allow_html=True)

    # 三张同规格白卡：之前是「一张 HTML 卡 + 两个裸按钮」，高度参差、看着散。
    _ready = predictor.is_ready
    q1, q2, q3 = st.columns(3)
    with q1:
        with st.container(border=True):
            st.markdown(
                f'<div class="panel-title">🧠 模型状态</div>'
                f'<div class="text-sub" style="margin-bottom:.5rem;">'
                f'{"已训练，可直接预测" if _ready else "尚未训练，需先训练模型"}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="pill {"violet" if _ready else "warn"}" '
                f'style="width:100%;justify-content:center;padding:.45rem .6rem;">'
                f'{"✅ 就绪" if _ready else "⚠️ 未就绪"}</div>',
                unsafe_allow_html=True,
            )

    with q2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">🤖 AI 选股</div>', unsafe_allow_html=True)
            st.markdown('<div class="text-sub" style="margin-bottom:.5rem;">'
                        '全市场扫描，按阈值筛出候选池</div>', unsafe_allow_html=True)
            if st.button("开始选股", type="primary", use_container_width=True, key="quick_pick"):
                st.session_state.page = "🔍 AI选股"
                st.session_state.auto_scan = True
                del st.session_state._nav_radio  # 清 radio 缓存，下次用 index
                st.rerun()

    with q3:
        with st.container(border=True):
            st.markdown('<div class="panel-title">📊 个股分析</div>', unsafe_allow_html=True)
            st.markdown('<div class="text-sub" style="margin-bottom:.5rem;">'
                        'K 线 · 技术指标 · 集成分与建议</div>', unsafe_allow_html=True)
            if st.button("开始分析", use_container_width=True, key="quick_analyze"):
                st.session_state.page = "📊 个股分析"
                st.session_state.auto_analyze_code = "000001"
                del st.session_state._nav_radio  # 清 radio 缓存，下次用 index
                st.rerun()

# ========== 🔍 AI选股页 ==========
elif page == "🔍 AI选股":
    render_hero(
        "AI 智能选股",
        "全市场扫描 → 多因子打分 → 按阈值筛出候选池，口径与个股分析页一致",
        icon="🤖",
        badges=[("violet", "超级集成" if _ens else "单模型")],
    )

    # 快速开始：首页按钮触发的自动扫描
    auto_scan = st.session_state.pop("auto_scan", False)
    logger.debug(f"[AUTO_SCAN] auto_scan={auto_scan} page={st.session_state.page}")

    if not predictor.is_ready:
        st.warning("⚠️ 模型尚未训练，无法进行 AI 选股")
        st.markdown("""
<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:1rem 1.2rem;">
  <div style="font-weight:700;color:#92400e;margin-bottom:.4rem;">🔍 请先完成模型训练</div>
  <div style="font-size:.86rem;color:#78350f;line-height:1.7;">
    方式一（推荐）：双击项目根目录的 <b>训练模型.bat</b>，等待完成<br>
    方式二：命令行执行 <code>python scripts/train_optuna.py</code>
  </div>
  <div style="font-size:.8rem;color:#92400e;margin-top:.5rem;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:.5rem .6rem;">
    ⚠️ 请勿使用 <code>scripts/train_v2.py</code>：v2 管线产出的
    <code>models/leaksafe_xgboost_model.pkl</code> <b>看板不读取</b>，照它训练完选股结果不会有任何变化。
    请在界面内训练，或使用 <code>scripts/train_optuna.py</code>。
  </div>
  <div style="font-size:.8rem;color:#92400e;margin-top:.5rem;">
    训练通常需要 10–40 分钟，取决于机器性能。训练完成后本页面会自动解锁。
  </div>
</div>
        """, unsafe_allow_html=True)
        if st.button("🔄 重新检测模型状态", type="primary", key="remodel_check"):
            clear_predictor_cache()
            st.rerun()
    else:
        # 筛选区
        with st.expander("⚙️ 筛选条件", expanded=True):
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                pool = st.selectbox("股票池", ["hs300", "zz500", "all"], index=1, help="hs300=大盘蓝筹, zz500=中盘成长（默认中证500：样本外RankIC更高）")
                top_n = st.slider("推荐数量", 5, 50, 20, 5)
            with rc2:
                # 量程按单模型化后的实际评分区间（约 38~60）收窄，步长 1 便于微调
                min_score = st.slider(
                    "最低评分（手动模式）", 35.0, 65.0, 50.0, 1.0,
                    help="仅当「设置→自我进化引擎→自动阈值调整」关闭时作为阈值；开启时采用自动进化的有效阈值，此滑块不参与筛选。",
                )
                scan_max = st.select_slider("扫描上限", options=[50, 100, 150, 200, 300, "全"], value=100, help="更大的扫描范围 = 更全面的覆盖，但耗时更长。选'全'=不限制")
            with rc3:
                trend_filter = st.radio("技术形态", ["全部", "多头排列", "金叉信号", "超卖反弹"], index=0, horizontal=False)

            # 板块过滤
            board_opts = ["沪深主板", "创业板", "科创板", "北交所"]
            board_filter = st.multiselect(
                "板块过滤", board_opts, default=[],
                help="不选=全部板块。沪深主板=000-003/600-605, 创业板=300-302, 科创板=688-689, 北交所=920"
            )

            # 高级筛选
            adv1, adv2, adv3 = st.columns(3)
            with adv1:
                pe_range = st.slider("PE 范围 (TTM)", 0.0, 200.0, (0.0, 200.0), 5.0, help="基本面具筛选（需有财务数据）")
            with adv2:
                cap_filter = st.selectbox("市值偏好", ["不限", "大盘(>500亿)", "中盘(100-500亿)", "小盘(<100亿)"], index=0)
            with adv3:
                sector_hot = st.checkbox("只选热点板块内股票", value=False, help="仅保留当天板块涨跌幅>0%的成分股")

        if st.button("🚀 开始 AI 选股", type="primary", width="stretch") or auto_scan:
            display_max = scan_max if scan_max != "全" else "全部"

            # ★ 回测验证面板
            with st.expander("📊 历史准确率（回测验证）", expanded=False):
                summary = get_accuracy_summary()
                if summary["n_verified"] == 0:
                    st.info("暂无已验证的预测记录。运行 5-10 天后这里会显示回测准确率。")
                else:
                    # tracker_v2 的汇总键与旧版不同：accuracy / avg_excess，
                    # 且无数据时返回 None（旧版返回 0）。
                    _acc = summary.get("accuracy") or 0.0
                    _exc = summary.get("avg_excess") or 0.0
                    m1, m2, m3 = st.columns(3)
                    m1.metric("已验证批次", f"{summary['n_verified']} 次")
                    m2.metric("平均准确率", f"{_acc*100:.1f}%",
                              delta=f"{_acc-0.5:.1%}" if _acc else None)
                    m3.metric("超额收益", f"{_exc*100:.2f}%",
                              delta=f"{_exc*100:.2f}%" if _exc else None)

                    if st.button("🔄 运行回测验证", help="拉取 5/10 天前推荐的实际表现，并触发自我进化"):
                        with st.spinner("拉取历史数据并验证…"):
                            from src.models.tracker_v2 import verify_all_pending
                            results = verify_all_pending()
                            if results:
                                st.success(f"✅ 验证完成: {len(results)} 个批次")
                                # ★ 触发自我进化
                                evo_actions = []
                                for r in results:
                                    evo_actions.extend(evolution.evolve_from_verification(r))
                                for r in results[-3:]:
                                    # v2 的键是 base_date / hold_days（旧版是 scan_date / lookback_days）
                                    _day = r.get("base_date") or r.get("scan_date")
                                    _h = r.get("hold_days") or r.get("lookback_days")
                                    st.write(
                                        f"📅 **{_day} +{_h}天**: "
                                        f"准确率={(r.get('accuracy') or 0)*100:.1f}% "
                                        f"平均收益={(r.get('avg_return') or 0)*100:+.2f}% "
                                        f"超额={(r.get('excess_return') or 0)*100:+.2f}% "
                                        f"胜率={(r.get('win_rate') or 0)*100:.1f}%"
                                    )
                                if evo_actions:
                                    st.markdown('<div class="section-title">🧬 本次自我进化</div>', unsafe_allow_html=True)
                                    for a in evo_actions:
                                        st.success(a)
                                else:
                                    st.info("🧬 本次数据量不足以触发进化（需≥10个模型信号样本）")

                                # ★ 离线进化判断提示
                                should_retrain, retrain_reason = evolution.should_retrain()
                                if should_retrain:
                                    st.warning(f"🧬 离线进化建议：{retrain_reason}")
                                    st.caption("可在下方「设置 → 自我进化引擎」启用自动重训，或手动点击「立即离线重训」")
                            else:
                                st.info("今日所有历史记录均已验证完毕，或未找到距离今日 ≥5 天的记录")
                        st.rerun()

                # 历史详情表
                if summary.get("history"):
                    st.caption("历史验证详情")
                    hist_df = pd.DataFrame(summary["history"])
                    st.dataframe(hist_df, width="stretch", hide_index=True)

            with st.spinner(f"AI 正在扫描 {pool} 股票池 (上限{display_max}只)…"):
                # ★ 自我进化：检测市场状态 + 应用完全自动的进化阈值（忽略用户滑块，使用固定基线）
                regime, effective_min_score = None, min_score
                bd = evolution.get_effective_breakdown()
                auto_adj = app_config.ml.auto_threshold_adjust
                try:
                    # 指数必须走 get_benchmark_kline，不能用 get_daily_kline
                    bench_df = fetcher.get_benchmark_kline("000300.SH", start_date="2024-01-01")
                    regime = evolution.detect_regime(bench_df)
                    if auto_adj:
                        # 完全自动：固定基线 + 进化调整 + 市场状态调整，滑块不参与
                        effective_min_score = evolution.get_effective_min_score()
                        bd = evolution.get_effective_breakdown()
                    else:
                        # 自动阈值调整已关闭：有效阈值恒等于用户设定，不被市场状态静默改写
                        effective_min_score = min_score
                except Exception as e:
                    logger.warning(f"市场状态检测失败，回退到用户设定阈值: {e}")

                if regime:
                    regime_name = {"bull": "🐂 牛市", "bear": "🐻 熊市", "sideways": "⚖️ 震荡市"}.get(regime, regime)
                    if auto_adj:
                        clamp_note = " · 已截断至阈值上限" if bd["clamped"] else ""
                        st.caption(
                            f"🧬 市场状态: {regime_name} · 自动有效阈值 {effective_min_score:.1f} "
                            f"= 基线{bd['base']:.0f} + 进化{bd['evolution_adj']:+.1f} + 市场{bd['regime_adj']:+.1f}{clamp_note}"
                        )
                    else:
                        st.caption(f"🧬 市场状态: {regime_name} · 自动阈值调整已关闭（有效阈值=用户设定 {min_score:.1f}）")

                # 获取热点板块（如果需要过滤）
                hot_sectors = set()
                if sector_hot:
                    try:
                        hotspots = _cached_hotspots()
                        hot_sectors = set(hotspots[hotspots["涨跌幅"] > 0]["板块"].values)
                        st.caption(f"🔥 当前热点板块: {len(hot_sectors)} 个")
                    except Exception:
                        pass

                codes = fetcher.get_stock_pool(pool)
                
                # 板块过滤（必须在 scan_max 截断之前！）
                if board_filter:
                    codes = fetcher.filter_by_board(codes, board_filter)
                    if not codes:
                        st.warning(f"⚠️ 选中板块（{'/'.join(board_filter)}）在 {pool} 池中无股票")
                        st.stop()
                    st.caption(f"🏷️ 板块过滤后: {len(codes)} 只 → 轮询排列确保各板块均匀覆盖")
                
                # 按板块均匀轮询排列，避免 scan_max 截断时偏向某一板块
                codes = fetcher.shuffle_by_prefix(codes)
                
                if scan_max != "全":
                    codes = codes[:scan_max]

                progress_bar = st.progress(0.1, "并行获取行情数据…")

                def _on_progress(done, total, ok):
                    """把真实下载进度回传到进度条"""
                    pct = 0.1 + 0.4 * (done / max(total, 1))
                    progress_bar.progress(min(pct, 0.5),
                                          f"获取行情 {done}/{total}（成功 {ok}）")

                raw_data = fetcher.get_batch_daily_kline(
                    codes, max_workers=8, progress_callback=_on_progress
                )
                progress_bar.progress(0.5, "技术形态预筛选…")

                if len(raw_data) == 0:
                    progress_bar.empty()
                    st.error(
                        "❌ 全部行情数据获取失败，可能是网络问题或数据源暂时不可用。\n\n"
                        "请检查网络连接后重试，或点击「🔄 刷新」重新加载。"
                    )
                    st.stop()
                elif len(raw_data) < len(codes) * 0.5:
                    st.warning(
                        f"⚠️ {len(codes)} 只中仅 {len(raw_data)} 只拉到数据，"
                        "结果可能不完整。可稍后重试或降低「扫描上限」。"
                    )

                # 技术形态预筛选
                stock_data = {}
                stock_names = {}
                for code, df in raw_data.items():
                    if len(df) < 60:
                        continue
                    # 技术形态预筛选
                    close = df["close"]
                    if trend_filter == "多头排列":
                        ma5 = close.rolling(5).mean().iloc[-1]
                        ma20 = close.rolling(20).mean().iloc[-1]
                        ma60 = close.rolling(60).mean().iloc[-1]
                        if not (ma5 > ma20 > ma60):
                            continue
                    elif trend_filter == "金叉信号":
                        ma5 = close.rolling(5).mean().iloc[-1]
                        ma20 = close.rolling(20).mean().iloc[-1]
                        ma5_prev = close.rolling(5).mean().iloc[-2]
                        ma20_prev = close.rolling(20).mean().iloc[-2]
                        if not (ma5 > ma20 and ma5_prev <= ma20_prev):
                            continue
                    elif trend_filter == "超卖反弹":
                        rsi = float(calc_rsi(close).iloc[-1])
                        if rsi > 30:
                            continue
                    stock_data[code] = df

                # 从全市场 stock_list 构建 code→name 映射
                stock_list = _cached_stock_list()
                name_map = dict(zip(stock_list["code"], stock_list["name"]))
                stock_names = {c: name_map.get(c, c) for c in stock_data.keys()}

                progress_bar.progress(0.95, "AI 模型预测中…")
                result_df = predictor.get_top_recommendations(
                    stock_data, stock_names, top_n=top_n, min_score=effective_min_score
                )
                progress_bar.empty()

                if len(result_df) == 0:
                    st.warning("没有股票达到评分阈值，请降低最低评分或放宽筛选条件")
                    st.stop()

                # 板块热度加成 (Task 6)
                if sector_hot and hot_sectors and len(result_df) > 0:
                    boost_scores = []
                    for _, row in result_df.iterrows():
                        code = row["代码"]
                        score = row["评分"]
                        # 查找该股票所属板块
                        try:
                            stock_info = _cached_stock_list()
                            stock_sector = stock_info[stock_info["code"] == code]
                            if len(stock_sector) > 0:
                                industry = stock_sector["industry"].iloc[0] if "industry" in stock_sector.columns else ""
                                # 板块匹配
                                boost = 0.0
                                for hs in hot_sectors:
                                    if hs and industry and (hs[:2] in industry or industry[:2] in hs):
                                        boost = 5.0
                                        break
                                boost_scores.append(boost)
                            else:
                                boost_scores.append(0.0)
                        except Exception:
                            boost_scores.append(0.0)
                    result_df["板块加成"] = boost_scores
                    result_df["综合评分"] = result_df["评分"] + pd.Series(boost_scores, index=result_df.index)
                    result_df = result_df.sort_values("综合评分", ascending=False).reset_index(drop=True)
                    result_df["排名"] = range(1, len(result_df) + 1)

                st.success(f"✅ 分析完成! 从 {len(stock_data)} 只股票中筛选出 {len(result_df)} 只推荐")

                # ★ 自动保存预测追踪（N日后验证准确率）
                try:
                    pred_records = []
                    signal_map = {"📈买入": 1, "📉卖出": -1, "➡️持有": 0}
                    for _, row in result_df.iterrows():
                        signal = signal_map.get(row.get("XGB信号", ""), 0)
                        pred_records.append({
                            "code": str(row["代码"]),
                            "name": str(row.get("名称", "")),
                            "score": float(row["评分"]),
                            "signal": signal,
                            "xgb_signal": signal_map.get(row.get("XGB信号", ""), 0),
                            "prob_up": float(row["上涨概率"]) / 100,
                            "prob_down": float(row["下跌概率"]) / 100,
                        })
                    # 顺手快照预测日收盘价：v2 验证时直接用快照，不必事后用复权价反推
                    # （前复权数据会被未来的除权事件改写，反推出来的「当时价格」并非当时可见价）。
                    # 价格取自内存里已有的 stock_data，不额外打行情接口。
                    price_snapshot = {}
                    for _code, _df in stock_data.items():
                        try:
                            _close = float(_df["close"].iloc[-1])
                            if _close > 0:
                                price_snapshot[_code] = round(_close, 3)
                        except Exception:
                            continue
                    save_predictions(pred_records, price_snapshot=price_snapshot)
                except Exception:
                    pass

                # 统计概览
                def score_tier(s):
                    if s >= 80: return "强力推荐"
                    if s >= 65: return "推荐"
                    if s >= 50: return "关注"
                    return "观望"
                tiers = result_df["评分"].apply(score_tier)
                st1, st2, st3, st4 = st.columns(4)
                for col, tier, color in [(st1, "强力推荐", "#ef4444"), (st2, "推荐", "#f59e0b"),
                                          (st3, "关注", "#3b82f6"), (st4, "观望", "#6b7280")]:
                    count = (tiers == tier).sum()
                    col.metric(tier, f"{count} 只", delta=f"{count/len(result_df)*100:.0f}%" if len(result_df) > 0 else None)

                # 评分分布图 + 确信度散点
                ch1, ch2 = st.columns([1, 1])
                with ch1:
                    top_plot = result_df.head(15)
                    fig = go.Figure(data=[go.Bar(
                        x=top_plot["评分"], y=top_plot["名称"], orientation="h",
                        marker=dict(
                            color=top_plot["评分"],
                            colorscale=[[0, "#10b981"], [0.4, "#f59e0b"], [0.7, "#ef4444"]],
                            showscale=True, colorbar=dict(title="评分", thickness=10)
                        ),
                        text=top_plot["评分"].round(1), textposition="outside",
                        textfont=dict(color="#1e293b", size=11),
                        hovertemplate="<b>%{y}</b><br>评分: %{x:.1f}<extra></extra>"
                    )])
                    fig.update_layout(height=380, template="plotly_white",
                                      xaxis_title="AI 综合评分", xaxis=dict(range=[0, 100]),
                                      yaxis=dict(autorange="reversed"), margin=dict(l=10, r=30, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                with ch2:
                    # 确信度 vs 评分散点
                    if "确信度" in result_df.columns:
                        top30 = result_df.head(30)
                        fig2 = go.Figure(data=[go.Scatter(
                            x=top30["确信度"], y=top30["评分"],
                            mode="markers+text",
                            text=top30["名称"].str[:4] if "名称" in top30.columns else top30.index.astype(str),
                            textposition="top center", textfont=dict(size=8, color="#475569"),
                            marker=dict(
                                size=top30["评分"].apply(lambda x: max(8, min(30, x / 3))),
                                color=top30["评分"],
                                colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                                showscale=True, colorbar=dict(title="评分", thickness=10),
                                line=dict(width=0.5, color="#fff")
                            ),
                            hovertemplate="<b>%{text}</b><br>评分: %{y:.1f}<br>确信度: %{x:.1f}%<extra></extra>"
                        )])
                        fig2.update_layout(height=380, template="plotly_white",
                                          xaxis_title="确信度 (%)", yaxis_title="AI 评分",
                                          margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig2, use_container_width=True)

                st.markdown('<div class="section-title">📋 推荐详情</div>', unsafe_allow_html=True)
                st.caption("💡 评分 = 预期收益 × 确信度 × 风险惩罚（XGBoost 单模型）| 确信度↑=方向越明确 | 风险惩罚↓=回撤/波动越大扣分越多")
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "评分": st.column_config.ProgressColumn("AI评分", min_value=0, max_value=100, format="%.1f"),
                        "确信度": st.column_config.ProgressColumn("确信度%", min_value=0, max_value=100, format="%.1f"),
                        "上涨概率": st.column_config.NumberColumn("上涨概率%", format="%.1f %%"),
                        "震荡概率": st.column_config.NumberColumn("震荡概率%", format="%.1f %%"),
                        "下跌概率": st.column_config.NumberColumn("下跌概率%", format="%.1f %%"),
                    },
                    height=min(len(result_df) * 35 + 38, 600)
                )

                # ====== 🤖 AI 组合推荐理由 ======
                # 文案按 provider 动态渲染：接了本地模型就别再显示「DeepSeek」，
                # 否则用户会以为还在调云端（而云端可能没余额）。
                _llm_name = "本地大模型" if getattr(deepseek, "is_local", False) else "DeepSeek"
                st.markdown(f'<div class="section-title">🤖 {_llm_name} 组合解读</div>', unsafe_allow_html=True)
                if not deepseek.is_ready:
                    st.info(f"💡 配置{_llm_name}（⚙️ 设置页）后，可生成组合推荐理由。")
                else:
                    ds_combo_btn = st.button("✨ 生成组合推荐理由", type="primary", key="ds_combo_btn")
                    if ds_combo_btn:
                        with st.spinner(f"🤖 {_llm_name} 正在解读组合…"):
                            try:
                                top_stocks = result_df.head(10)[["代码", "名称", "评分", "上涨概率", "下跌概率", "推荐理由"]].to_dict("records")
                                combo_rationale = deepseek.explain_recommendations(top_stocks)
                                st.session_state["ds_combo_rationale"] = combo_rationale
                            except Exception as e:
                                st.error(f"❌ 生成失败: {e}")
                    cached_combo = st.session_state.get("ds_combo_rationale", "")
                    if cached_combo:
                        st.markdown(cached_combo)

# ========== 📊 个股分析页 ==========
elif page == "📊 个股分析":
    render_hero(
        "个股深度分析",
        "K 线 · 技术指标 · 基本面 · AI 评分一屏看全，评分口径与选股页一致",
        icon="📊",
        badges=[("violet", "超级集成" if _ens else "单模型")],
    )

    # 快开始：首页按钮触发的自动分析
    auto_analyze_code = st.session_state.pop("auto_analyze_code", None)

    # 参数行收进白卡：之前四个控件直接浮在灰底上，松散得像半成品
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            stock_code = st.text_input("股票代码", value=auto_analyze_code if auto_analyze_code else "000001", key="input_stock_code", placeholder="6位代码")
        with col2:
            days = st.slider("回看天数", 60, 500, 250, 30)
        with col3:
            use_fund = st.checkbox("基本面", value=True, help="加载PE/PB/ROE等财务数据")
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("🔍 开始分析", type="primary", use_container_width=True) or auto_analyze_code

    # 未分析时给一个有内容的空状态，而不是留半屏白 —— 空白是最"简陋"的观感来源
    if not analyze_btn:
        empty_block(
            "输入 6 位代码，点「开始分析」",
            icon="📊",
            hint="会拉取日线行情 + 基本面，并给出超级集成分、交易建议与相对强弱",
        )

    if analyze_btn:
        with st.spinner(f"正在深度分析 {stock_code}…"):
            df = fetcher.get_daily_kline(stock_code, start_date=(datetime.now() - timedelta(days=days)).strftime("%Y%m%d"))

            if len(df) == 0:
                st.error(f"未找到 {stock_code} 的数据")
                st.stop()

            close = df["close"]
            df_plot = df.tail(180).copy()
            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            returns = close.pct_change().dropna()

            def _err_box(_title: str, _err, _keep: int = 80) -> str:
                """把网络异常压成一行提示条。

                直接 f-string 塞 requests 的原始异常会渲染成一个十几行的巨型红黄块
                （含整条 URL 编码串），把整页视觉都压垮了；这里只留最有辨识度的
                前半段，剩下的让人自己去看日志。
                """
                _t = re.sub(r"\s+", " ", str(_err)).strip()
                for _a, _b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;")):
                    _t = _t.replace(_a, _b)
                if len(_t) > _keep:
                    _t = _t[:_keep] + "…"
                return (f'<div class="notice warn">⚠️ <b>{_title}</b><br>'
                        f'<span style="font-weight:400;">{_t}</span><br>'
                        '<span style="font-weight:400;">不影响行情与 AI 评分；'
                        '行业/市值等字段暂缺。</span></div>')

            # 股票信息：静态列表（code→name 兜底） + 东方财富实时基本信息（行业/市值/上市日）
            stock_info = None
            base = {"name": stock_code, "industry": "", "total_cap": 0.0,
                    "float_cap": 0.0, "listed_date": ""}
            try:
                slist = _cached_stock_list()
                stock_info = slist[slist["code"] == stock_code]
                if len(stock_info) > 0:
                    _r = stock_info.iloc[0]
                    base["name"] = str(_r.get("name", stock_code) or stock_code)
                    base["industry"] = str(_r.get("industry", "") or "")
                    base["total_cap"] = float(_r.get("total_cap") or 0)
                    base["float_cap"] = float(_r.get("float_cap") or 0)
                    base["listed_date"] = str(_r.get("listed_date", "") or "")
            except Exception:
                pass
            # 叠加东方财富实时基本信息（更全：行业/总市值/流通市值/上市日期）
            try:
                _ri = _cached_stock_info(stock_code)
                if _ri and "_error" not in _ri:
                    if _ri.get("name"):
                        base["name"] = _ri["name"]
                    if _ri.get("industry"):
                        base["industry"] = _ri["industry"]
                    if _ri.get("total_cap"):
                        base["total_cap"] = _ri["total_cap"]
                    if _ri.get("float_cap"):
                        base["float_cap"] = _ri["float_cap"]
                    if _ri.get("listed_date"):
                        base["listed_date"] = _ri["listed_date"]
                    # 东方财富不可达但备用源补齐了部分字段 → 低调标注，不再整块空白
                    if _ri.get("_missing"):
                        st.caption(
                            f"ℹ️ 东方财富接口不可达，已用备用源（{_ri.get('_source') or '备用'}）"
                            f"补齐；{('、'.join(_ri['_missing']))} 暂缺"
                        )
                elif _ri and "_error" in _ri:
                    st.markdown(_err_box("个股基本信息获取失败（东方财富接口受限）", _ri["_error"]),
                                unsafe_allow_html=True)
            except Exception as _e:
                st.markdown(_err_box("个股基本信息获取异常", _e), unsafe_allow_html=True)
            stock_name = base["name"]

            # ====== 页面级小工具（评分环 / 概率条 / KV 网格）======
            def _ana_bar(_label: str, _pct: float, _color: str) -> str:
                _w = max(0.0, min(100.0, float(_pct)))
                return (
                    '<div class="bar">'
                    f'<div class="bar-head"><span>{_label}</span>'
                    f'<b style="color:{_color};">{_w:.1f}%</b></div>'
                    f'<div class="bar-track"><div class="bar-fill" '
                    f'style="width:{_w:.1f}%;background:{_color};"></div></div>'
                    '</div>'
                )

            def _ana_kv(_items) -> str:
                """(标签, 数值, 颜色) 列表 → 定宽小格网格，取代一行行裸文本。"""
                _cells = ""
                for _l, _v, _c in _items:
                    _st = f' style="color:{_c};"' if _c else ""
                    _cells += (
                        '<div class="kv-i">'
                        f'<div class="k">{_l}</div>'
                        f'<div class="v"{_st}>{_v}</div>'
                        '</div>'
                    )
                return f'<div class="kv">{_cells}</div>'

            def _ret_n(_s: pd.Series, _n: int) -> float:
                """近 N 个交易日涨跌幅（不足 N 根时退化为全区间）。"""
                if len(_s) < 2:
                    return 0.0
                if len(_s) <= _n:
                    return float(_s.iloc[-1] / _s.iloc[0] - 1) * 100
                return float(_s.iloc[-1] / _s.iloc[-1 - _n] - 1) * 100

            # ====== ① 标的头：名称 / 代码 / 行业 / 现价 / 涨跌 ======
            _chg = last_close - prev_close
            _chg_pct = (last_close / prev_close - 1) * 100 if prev_close else 0.0
            _cls_chg = "red" if _chg > 0 else ("green" if _chg < 0 else "gray")
            _ind_txt = (base.get("industry") or "").strip()
            _ind_html = (
                f'<span class="pill gray">{_ind_txt[:8]}</span>'
                if _ind_txt and _ind_txt not in ("未知", "—") else ""
            )
            st.markdown(
                '<div class="stock-head">'
                '<div class="sh-main">'
                f'<div class="sh-name">{stock_name}</div>'
                f'<div class="sh-code">{stock_code} · 共 {len(df)} 根日线 · 图形近 180 日</div>'
                '</div>'
                f'{_ind_html}'
                '<div class="sh-quote">'
                f'<span class="sh-price {_cls_chg}">{last_close:.2f}</span>'
                f'<span class="sh-chg {_cls_chg}">{_chg:+.2f} ({_chg_pct:+.2f}%)</span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # ====== ② 多周期 KPI 条 ======
            # 技术指标先算一遍：KPI 条要量比，右侧技术信号要 BOLL/KDJ。
            boll_mid, boll_upper, boll_lower = calc_boll(close)
            k, d, j = calc_kdj(df_plot)
            vol_ma5 = df_plot["volume"].rolling(5).mean()
            vol_ratio = float(df_plot["volume"].iloc[-1] / (vol_ma5.iloc[-1] + 1e-10))
            # 用 st.columns(5) 而不是 kpi_row 的 auto-fill 网格：网格按最小宽度 172px
            # 会排到 6 列，5 张卡只填 5 列、右侧留一整列空白，看着像没对齐。
            _kpi_items = [
                (f"{'📈' if _chg_pct > 0 else ('📉' if _chg_pct < 0 else '➡️')} 今日",
                 f"{_chg_pct:+.2f}%", _chg_pct,
                 f"{_chg:+.2f} 元"),
            ]
            for _lb, _vv in [("近 5 日", _ret_n(close, 5)),
                             ("近 20 日", _ret_n(close, 20)),
                             ("近 60 日", _ret_n(close, 60))]:
                _kpi_items.append(
                    (f"🗓️ {_lb}", f"{_vv:+.2f}%", _vv,
                     "短线动能" if _lb == "近 5 日" else ("月度趋势" if _lb == "近 20 日" else "季度趋势"))
                )
            _kpi_items.append(
                ("🔥 量比", f"{vol_ratio:.2f}x", None,
                 "放量" if vol_ratio > 1.5 else ("缩量" if vol_ratio < 0.7 else "量能正常"))
            )
            _kpi_cols = st.columns(len(_kpi_items))
            for _i, (_lb, _val, _sig, _sub) in enumerate(_kpi_items):
                with _kpi_cols[_i]:
                    st.markdown(html_metric(_lb, _val, _sig, _sub), unsafe_allow_html=True)

            # ====== ③ 计算 AI 预测与交易建议（只算不画，供下面两列共用）======
            pred = None
            score = 0.0
            s_cls = "gray"
            _mode = "single"
            _det = {}
            th_red, th_amber, th_caution, th_bear = 55.0, 49.0, 51.0, 45.0
            action, action_cls = "⚪ 观望为主", ""
            entry_range, stop_loss, target, rr_txt = "—", "—", "—", "—"
            agree_str = "➡️ 模型观望"
            if predictor.is_ready:
                # 显式走超级集成，与「选股」批量打分口径一致（共用 blend_one）。
                # 此前本页恒走单模型，而单模型实测 mean IC 0.0575 / t=1.40 统计不显著。
                try:
                    pred = predictor.predict_single(df, stock_code, stock_name, apply_ensemble=True)
                except TypeError:
                    # 冻结包里的 predictor 可能是旧版，没有 apply_ensemble 参数 →
                    # 退回老签名，绝不让页面崩。
                    pred = predictor.predict_single(df, stock_code, stock_name)
                score = pred.score
                # 标注本次评分**实际**走的口径，避免把不显著的单模型分当成集成分来下单
                _mode = getattr(pred, "model_mode", "single")
                _det = getattr(pred, "ensemble_detail", {}) or {}
                # ===== 分档阈值：必须按「实际口径」分别校准，绝不能混用 =====
                # · 单模型（2026-10-10 实测 38~60，mean 47.3）→ 红55 / 黄49 / 谨慎51 / 看空45
                # · 集成分（2026-10-18 实测，scripts/calibrate_ensemble_thresholds.py，
                #   228 只样本且 228/228 全部融合成功）
                #   min 35.2 / p25 44.4 / median 46.5 / p75 49.7 / p90 51.9 / max 58.4
                #   → 红51(约前 11%) / 黄47(约前 42%) / 谨慎48 / 看空43(约后 15%)
                # ⚠️ 千万别拿 prediction_history.json 校准：它只记「上榜票」
                # （先按 min_score=50 过滤再取 TOP-N），是**截断样本**，分布虚高到 55~88。
                if _mode == "ensemble":
                    th_red, th_amber, th_caution, th_bear = 51.0, 47.0, 48.0, 43.0
                else:
                    th_red, th_amber, th_caution, th_bear = 55.0, 49.0, 51.0, 45.0
                s_cls = "red" if score >= th_red else ("amber" if score >= th_amber else "green")
                agree_str = ("📈 模型看多" if pred.xgb_signal == 1
                             else ("📉 模型看空" if pred.xgb_signal == -1 else "➡️ 模型观望"))

                # 交易建议：方向取 XGBoost 主腿，档位阈值随当前口径切换。
                # 风报比改成按入场/止损/目标真实算出，不再写死 1:2/1:1.5（
                # 那两个分支要求 score>=60/70，实际分布 43~58 永远命中不到，恒显示「—」）。
                if score >= th_red and pred.xgb_signal == 1:
                    action, action_cls = "🟢 强烈看多", "red"
                    _lo, _hi, _sl, _tp = last_close * 0.98, last_close, last_close * 0.93, last_close * 1.08
                    _tp2 = last_close * 1.15
                elif score >= th_caution and pred.xgb_signal == 1:
                    action, action_cls = "🟡 谨慎看多", "amber"
                    _lo, _hi, _sl, _tp = last_close * 0.97, last_close, last_close * 0.94, last_close * 1.06
                    _tp2 = None
                elif score < th_bear and pred.xgb_signal == -1:
                    action, action_cls = "🔴 明确看空", "green"
                    _lo = _hi = _sl = _tp = None
                    _tp2 = None
                else:
                    action, action_cls = "⚪ 观望为主", ""
                    _lo = _hi = _sl = _tp = None
                    _tp2 = None

                if _lo is not None:
                    entry_range = f"{_lo:.2f} ~ {_hi:.2f}"
                    stop_loss = f"{_sl:.2f}"
                    target = f"{_tp:.2f}" + (f" / {_tp2:.2f}" if _tp2 else "")
                    _risk = last_close - _sl
                    rr_txt = f"1 : {(_tp - last_close) / _risk:.1f}" if _risk > 1e-9 else "—"

            # ====== ④ 双列：AI 评分 / 交易建议 ======
            _c1, _c2 = st.columns([3, 2])

            with _c1:
                if pred is not None:
                    _ring_p = max(0.0, min(100.0, float(score)))
                    _ring_c = {"red": "#dc2626", "amber": "#d97706",
                               "green": "#059669"}.get(s_cls, "#5b21b6")
                    _level = ("强多头" if score >= th_red else ("偏多" if score >= th_amber
                              else ("中性偏弱" if score >= th_bear else "偏空")))
                    _legs = " + ".join((_det.get("scores") or {}).keys()) or "XGBoost"
                    _spread = float(_det.get("spread", 0.0) or 0.0)
                    _meta = (f'分歧度 {_spread:.1f}｜确信度 {pred.confidence*100:.0f}%'
                             f'｜波动率 {pred.volatility*100:.1f}%') if _mode == "ensemble" else \
                            (f'确信度 {pred.confidence*100:.0f}%｜波动率 {pred.volatility*100:.1f}%')
                    _src = ("🧩 超级集成" if _mode == "ensemble" else "⚠️ 单模型 XGBoost")
                    st.markdown(
                        '<div class="panel"><div class="panel-title">🤖 AI 评分 · '
                        f'{_src}<span class="text-sub" style="font-weight:500;margin-left:.3rem;">'
                        f'（{_legs}）</span></div>'
                        '<div class="score">'
                        f'<div class="ring" style="--p:{_ring_p:.1f};--c:{_ring_c};">'
                        f'<div class="ring-in"><span class="num">{score:.0f}</span>'
                        f'<span class="cap">{_level}</span></div></div>'
                        '<div class="bars">'
                        + _ana_bar("📈 上涨概率", pred.prob_up * 100, "#dc2626")
                        + _ana_bar("➡️ 震荡概率", pred.prob_neutral * 100, "#d97706")
                        + _ana_bar("📉 下跌概率", pred.prob_down * 100, "#059669")
                        + '</div></div>'
                        f'<div class="score-foot">{agree_str} · {_meta}</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(panel(
                        "🤖 AI 评分",
                        '<div class="text-sub">模型未就绪，本页仅展示行情与技术面。'
                        '到「⚙️ 设置」页检查模型文件后即可获得评分。</div>',
                        icon="",
                    ), unsafe_allow_html=True)

            with _c2:
                if pred is not None:
                    _rows = [
                        ("交易建议", f'<b class="{action_cls}">{action}</b>'),
                        ("建议入场", f'<b>{entry_range}</b>'),
                        ("止损位", f'<b class="green">{stop_loss}</b>'),
                        ("目标位", f'<b class="red">{target}</b>'),
                        ("风险收益比", f'<b>{rr_txt}</b>'),
                    ]
                    _body = '<div class="trade-list">' + "".join(
                        f'<div class="trade-row"><span class="k">{_k}</span>'
                        f'<span class="v">{_v}</span></div>' for _k, _v in _rows
                    ) + '</div>'
                    if pred.reasons:
                        _body += ('<div class="tag-row" style="margin-top:.6rem;">'
                                  '<span class="text-sub" style="margin-right:.2rem;">研判</span>'
                                  + "".join(f'<span class="tag info">{_r}</span>'
                                            for _r in pred.reasons[:4])
                                  + '</div>')
                    st.markdown(panel("🎯 交易建议", _body),
                                unsafe_allow_html=True)
                    st.markdown(
                        '<div class="notice info" style="margin-top:-.2rem;">'
                        '⚠️ 建议价由当日收盘价按档位比例推算，非盯盘实时价；'
                        '实盘前请到「量化自动交易」页走风控闸门。</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(panel(
                        "🎯 交易建议",
                        '<div class="text-sub">无模型评分，不给入场/止损建议。'
                        '技术面信号见右下方。</div>',
                    ), unsafe_allow_html=True)

            # ====== ⑤ 双列：基本面 / 技术信号 ======
            _d1, _d2 = st.columns([1, 1])

            with _d1:
                # 基本面：改为小格网格（原来一行行「标签 …… 数值」，十条下来极像半成品）
                fund_items = []
                if base:
                    _ind = base.get("industry", "") or "—"
                    _ind = "—" if _ind in ("未知", "", "—") else _ind[:8]
                    _tc = base.get("total_cap") or 0.0
                    _fc = base.get("float_cap") or 0.0
                    _ld = base.get("listed_date", "") or "—"
                    fund_items.extend([
                        ("行业", _ind, "#7c3aed"),
                        ("总市值", f"{_tc:.0f} 亿" if _tc else "—", ""),
                        ("流通市值", f"{_fc:.0f} 亿" if _fc else "—", ""),
                        ("上市日期", str(_ld)[:10], ""),
                        # 行情起点是本地一定能算出来的事实 —— 补齐第 5 格，
                        # 否则离线时 4 格 + 1 空位，网格看着像塌了一角。
                        ("数据起点", str(df["date"].iloc[0])[:10] if "date" in df.columns else "—", ""),
                    ])
                fin = None
                try:
                    fin = fetcher.get_financial_indicators(stock_code)
                    if fin is not None and len(fin) > 0:
                        latest = fin.iloc[-1]
                        for key, label, unit, fmt in [
                            ("pe_ttm", "PE(TTM)", "", ".1f"),
                            ("pb", "PB", "", ".1f"),
                            ("roe", "ROE", "%", ".1f"),
                            ("profit_growth", "净利增速", "%", ".1f"),
                            ("revenue_growth", "营收增速", "%", ".1f"),
                        ]:
                            val = latest.get(key)
                            if val is not None and val == val:
                                if key == "roe":
                                    c = "#dc2626" if val > 15 else ("#d97706" if val > 8 else "#059669")
                                elif key == "profit_growth":
                                    c = "#dc2626" if val > 20 else ("#d97706" if val > 0 else "#059669")
                                else:
                                    c = ""
                                fund_items.append((label, f"{val:{fmt}}{unit}", c))
                except Exception:
                    pass

                # 行业热度
                try:
                    sectors = _cached_hotspots()
                    ind = base.get("industry", "")
                    if ind and ind != "未知":
                        match = sectors[sectors["板块"].str.contains(ind[:2], na=False)]
                        if len(match) > 0:
                            s = match.iloc[0]
                            spct = s["涨跌幅"]
                            fund_items.append(("板块涨跌", f"{spct:+.2f}%",
                                               "#dc2626" if spct > 0 else "#059669"))
                except Exception:
                    pass

                _miss = sum(1 for _, _v, _ in fund_items if _v == "—")
                _hint = (f'<div class="text-sub" style="margin-top:.45rem;">'
                         f'⚠️ {_miss} 项缺失：本地无离线缓存，联网后重新分析可补齐</div>'
                         if _miss else "")
                st.markdown(panel("📋 基本面概况", _ana_kv(fund_items) + _hint),
                            unsafe_allow_html=True)

            with _d2:
                st.markdown(
                    '<div class="panel"><div class="panel-title">🧠 技术信号 · K线形态</div>',
                    unsafe_allow_html=True,
                )
                ma5_now = float(close.rolling(5).mean().iloc[-1])
                ma20_now = float(close.rolling(20).mean().iloc[-1])
                ma60_now = float(close.rolling(60).mean().iloc[-1])
                rsi14_now = float(calc_rsi(close).iloc[-1])
                k_now, d_now, j_now = float(k.iloc[-1]), float(d.iloc[-1]), float(j.iloc[-1])
                boll_pos = float((last_close - boll_lower.iloc[-1]) / (boll_upper.iloc[-1] - boll_lower.iloc[-1] + 1e-10))
                vol_ratio = float(df_plot["volume"].iloc[-1] / (vol_ma5.iloc[-1] + 1e-10))

                # K线形态识别
                latest = df_plot.iloc[-1]
                prev1 = df_plot.iloc[-2]
                o, c, h, l = float(latest["open"]), float(latest["close"]), float(latest["high"]), float(latest["low"])
                o2, c2, h2, l2 = float(prev1["open"]), float(prev1["close"]), float(prev1["high"]), float(prev1["low"])
                body = abs(c - o)
                upper_shadow = h - max(c, o)
                lower_shadow = min(c, o) - l
                total_range = h - l
                body_ratio = body / (total_range + 1e-10)

                candle_patterns = []
                # 十字星
                if body_ratio < 0.1 and total_range > body * 3:
                    candle_patterns.append(("info", "✝️ 十字星", "多空平衡，即将变盘"))
                # 锤子线
                if body_ratio < 0.35 and lower_shadow > body * 2 and upper_shadow < body * 0.5:
                    candle_patterns.append(("bullish", "🔨 锤子线", "底部反转信号"))
                # 倒锤子
                if body_ratio < 0.35 and upper_shadow > body * 2 and lower_shadow < body * 0.5:
                    candle_patterns.append(("warn", "🔃 倒锤子", "顶部反转警示"))
                # 吞没形态
                if c > o > 0 and c2 < o2:  # 阳包阴
                    if c > o2 and o < c2:
                        candle_patterns.append(("bullish", "📈 看涨吞没", "阳线吞没前阴线"))
                elif c < o < 0 and c2 > o2:  # 阴包阳
                    if c < o2 and o > c2:
                        candle_patterns.append(("bearish", "📉 看跌吞没", "阴线吞没前阳线"))
                # 三连阳/三连阴
                if len(df_plot) >= 4:
                    if all(df_plot["close"].iloc[-3:].values > df_plot["open"].iloc[-3:].values):
                        candle_patterns.append(("bullish", "🔥 三连阳", "强势上攻"))
                    elif all(df_plot["close"].iloc[-3:].values < df_plot["open"].iloc[-3:].values):
                        candle_patterns.append(("bearish", "💧 三连阴", "持续走弱"))

                signals = []
                # 均线排列
                if last_close > ma5_now > ma20_now > ma60_now:
                    signals.append(("bullish", "📈 多头排列", "MA5>MA20>MA60 上升趋势"))
                elif last_close < ma5_now < ma20_now < ma60_now:
                    signals.append(("bearish", "📉 空头排列", "均线全面下行"))
                else:
                    signals.append(("warn", "⚠️ 均线交织", "方向不明"))
                # MA金叉死叉
                ma5_prev = float(close.rolling(5).mean().iloc[-2])
                ma20_prev = float(close.rolling(20).mean().iloc[-2])
                if ma5_now > ma20_now and ma5_prev <= ma20_prev:
                    signals.append(("bullish", "🔺 MA金叉", "MA5上穿MA20"))
                elif ma5_now < ma20_now and ma5_prev >= ma20_prev:
                    signals.append(("bearish", "🔻 MA死叉", "MA5下穿MA20"))
                # RSI
                if rsi14_now > 70:
                    signals.append(("warn", f"⚠️ RSI超买({rsi14_now:.0f})", "注意回调风险"))
                elif rsi14_now < 30:
                    signals.append(("bullish", f"💡 RSI超卖({rsi14_now:.0f})", "可能反弹"))
                else:
                    signals.append(("info", f"✅ RSI({rsi14_now:.0f})", "合理区间"))
                # KDJ
                if j_now < 0:
                    signals.append(("bullish", "💡 KDJ极度超卖", f"J={j_now:.0f}"))
                elif j_now > 100:
                    signals.append(("warn", "⚠️ KDJ极度超买", f"J={j_now:.0f}"))
                elif k_now > d_now:
                    signals.append(("bullish", "🔺 KDJ金叉", f"K({k_now:.0f})>D({d_now:.0f})"))
                else:
                    signals.append(("bearish", "🔻 KDJ死叉", f"K({k_now:.0f})<D({d_now:.0f})"))
                # BOLL
                if boll_pos > 0.95:
                    signals.append(("warn", "⚠️ 触及BOLL上轨", "强压力位"))
                elif boll_pos < 0.05:
                    signals.append(("bullish", "💡 触及BOLL下轨", "强支撑位"))
                else:
                    signals.append(("info", f"✅ BOLL({boll_pos*100:.0f}%)", "通道内运行"))
                # 量能
                if vol_ratio > 2:
                    signals.append(("info", f"🔥 大幅放量({vol_ratio:.1f}x)", "资金活跃"))
                elif vol_ratio < 0.5:
                    signals.append(("warn", f"💤 极度缩量({vol_ratio:.1f}x)", "交投冷清"))

                # 渲染 K线形态 + 技术信号
                # ⚠️ 必须拼成一个 markdown 一次性输出：每个 st.markdown 都会生成独立的
                # block 容器，逐个输出会让每颗 tag 各占一行，右侧一列变成竖排清单。
                _tags_html = ""
                if candle_patterns:
                    _tags_html += ('<div class="tag-row">'
                                   '<span class="text-sub" style="margin-right:.2rem;">🕯️ 形态</span>')
                    for tag_cls, title, desc in candle_patterns:
                        _tags_html += f'<span class="tag {tag_cls}"><b>{title}</b> {desc}</span>'
                    _tags_html += '</div>'
                _tags_html += ('<div class="tag-row">'
                               '<span class="text-sub" style="margin-right:.2rem;">🧠 信号</span>')
                for tag_cls, title, desc in signals:
                    _tags_html += f'<span class="tag {tag_cls}"><b>{title}</b> {desc}</span>'
                _tags_html += '</div>'
                st.markdown(_tags_html, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            _tab_main, _tab_ind, _tab_chip, _tab_fin, _tab_rs, _tab_sig = st.tabs(
                ["📈 主图 · 量能", "🎯 RSI / KDJ", "🧊 筹码分布",
                 "📊 财务趋势", "📡 相对强弱", "🎲 信号回测"]
            )

            with _tab_main:
                # ====== K线 + 支撑压力 ======
                st.markdown('<div class="chart-box"><h3>📈 K线图 · 布林带 · 支撑压力位</h3>', unsafe_allow_html=True)

                # 计算支撑压力
                hl20 = df_plot.iloc[-20:]
                support1 = float(hl20["low"].min())
                resistance1 = float(hl20["high"].max())
                hl60 = df.iloc[-60:]
                support2 = float(hl60["low"].min())
                resistance2 = float(hl60["high"].max())
                pivot = (hl20["high"].max() + hl20["low"].min() + hl20["close"].iloc[-1]) / 3

                fig1 = go.Figure()
                fig1.add_trace(go.Candlestick(
                    x=df_plot["date"], open=df_plot["open"], high=df_plot["high"],
                    low=df_plot["low"], close=df_plot["close"],
                    name="K线", increasing_line_color="#ef4444", decreasing_line_color="#10b981",
                    increasing_fillcolor="#ef4444", decreasing_fillcolor="#10b981"
                ))
                for p, color in [(5, "#3b82f6"), (20, "#f59e0b"), (60, "#8b5cf6")]:
                    ma = close.rolling(p).mean()
                    fig1.add_trace(go.Scatter(x=df_plot["date"], y=ma, name=f"MA{p}",
                                              line=dict(color=color, width=1.5)))
                boll_mid, boll_upper, boll_lower = calc_boll(close)
                boll_fill = pd.DataFrame({
                    "x": list(df_plot["date"]) + list(df_plot["date"])[::-1],
                    "y": list(boll_upper.iloc[-180:]) + list(boll_lower.iloc[-180:])[::-1]
                })
                fig1.add_trace(go.Scatter(x=boll_fill["x"], y=boll_fill["y"], fill="toself",
                                          fillcolor="rgba(100,116,139,0.08)", line=dict(width=0),
                                          name="BOLL通道", showlegend=True))

                # 支撑压力线
                for val, label, color in [
                    (support1, f"支撑1(20日低:{support1:.2f})", "#059669"),
                    (support2, f"支撑2(60日低:{support2:.2f})", "#047857"),
                    (resistance1, f"压力1(20日高:{resistance1:.2f})", "#ef4444"),
                    (resistance2, f"压力2(60日高:{resistance2:.2f})", "#b91c1c"),
                    (pivot, f"枢轴点({pivot:.2f})", "#8b5cf6"),
                ]:
                    # 交替上下摆放：压力1(20日高) 和 压力2(60日高) 常常只差几分钱，
                    # 同侧标注会叠成一团。
                    _pos = ("top right", "bottom right")[_i % 2]
                    fig1.add_hline(y=val, line_dash="dash", line_color=color, opacity=0.5,
                                   annotation_text=label.split("(")[0], annotation_position=_pos)

                # 右侧给支撑/压力标注留位置：annotation_position="right" 是画在
                # 绘图区外侧的 margin 里，r=10 会把「支撑1」切成「支」。
                fig1.update_layout(height=500, template="plotly_white", hovermode="x unified",
                                   xaxis_rangeslider_visible=True, legend=dict(orientation="h", y=1.15),
                                   margin=dict(l=10, r=48, t=20, b=10))
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


                # ====== 成交分布 + MACD ======
                st.markdown('<div class="chart-box"><h3>📊 成交量 · MACD(12,26,9)</h3>', unsafe_allow_html=True)

                fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                     row_heights=[0.3, 0.7])
                vol_colors = ["#ef4444" if c >= o else "#10b981" for c, o in zip(df_plot["close"], df_plot["open"])]
                fig2.add_trace(go.Bar(x=df_plot["date"], y=df_plot["volume"], name="成交量",
                                      marker_color=vol_colors, opacity=0.7), row=1, col=1)
                vol_ma5 = df_plot["volume"].rolling(5).mean()
                fig2.add_trace(go.Scatter(x=df_plot["date"], y=vol_ma5, name="量MA5",
                                          line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
                vol_ma20 = df_plot["volume"].rolling(20).mean()
                fig2.add_trace(go.Scatter(x=df_plot["date"], y=vol_ma20, name="量MA20",
                                          line=dict(color="#94a3b8", width=1, dash="dot")), row=1, col=1)

                dif, dea, macd_bar = calc_macd(close)
                macd_colors = ["#ef4444" if v >= 0 else "#10b981" for v in macd_bar.iloc[-180:]]
                fig2.add_trace(go.Scatter(x=df_plot["date"], y=dif.iloc[-180:], name="DIF",
                                          line=dict(color="#ef4444", width=1.8)), row=2, col=1)
                fig2.add_trace(go.Scatter(x=df_plot["date"], y=dea.iloc[-180:], name="DEA",
                                          line=dict(color="#3b82f6", width=1.8)), row=2, col=1)
                fig2.add_trace(go.Bar(x=df_plot["date"], y=macd_bar.iloc[-180:], name="柱",
                                      marker_color=macd_colors, opacity=0.5), row=2, col=1)
                fig2.add_hline(y=0, line_color="#94a3b8", line_width=0.5, row=2, col=1)
                fig2.update_layout(height=400, template="plotly_white", hovermode="x unified",
                                   legend=dict(orientation="h", y=1.12),
                                   margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


            with _tab_ind:
                # ====== RSI + KDJ ======
                st.markdown('<div class="chart-box"><h3>🎯 RSI(6,14) + KDJ(9,3,3)</h3>', unsafe_allow_html=True)

                fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                     row_heights=[0.5, 0.5])
                for p, color in [(6, "#ef4444"), (14, "#3b82f6")]:
                    rsi = calc_rsi(close, period=p)
                    fig3.add_trace(go.Scatter(x=df_plot["date"], y=rsi.iloc[-180:], name=f"RSI{p}",
                                              line=dict(color=color, width=1.8)), row=1, col=1)
                for y, lbl in [(70, "超买"), (50, "中轴"), (30, "超卖")]:
                    fig3.add_hline(y=y, line_dash="dash", line_color="#94a3b8", opacity=0.4,
                                   annotation_text=lbl, annotation_position="right", row=1, col=1)

                k, d, j = calc_kdj(df_plot)
                fig3.add_trace(go.Scatter(x=df_plot["date"], y=k.iloc[-180:], name="K",
                                          line=dict(color="#ef4444", width=1.8)), row=2, col=1)
                fig3.add_trace(go.Scatter(x=df_plot["date"], y=d.iloc[-180:], name="D",
                                          line=dict(color="#3b82f6", width=1.8)), row=2, col=1)
                fig3.add_trace(go.Scatter(x=df_plot["date"], y=j.iloc[-180:], name="J",
                                          line=dict(color="#8b5cf6", width=1.2, dash="dot")), row=2, col=1)
                for y, lbl in [(80, "超买"), (20, "超卖")]:
                    fig3.add_hline(y=y, line_dash="dash", line_color="#94a3b8", opacity=0.4,
                                   annotation_text=lbl, annotation_position="right", row=2, col=1)
                fig3.update_layout(height=400, template="plotly_white", hovermode="x unified",
                                   legend=dict(orientation="h", y=1.12),
                                   margin=dict(l=10, r=36, t=20, b=10))
                st.plotly_chart(fig3, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


            with _tab_chip:
                # ====== 🧊 筹码分布（成本分布） ======
                try:
                    _cn = min(120, len(df))
                    _cd = df.tail(_cn).copy()
                    _pmin = float(_cd["low"].min())
                    _pmax = float(_cd["high"].max())
                    if _pmax <= _pmin:
                        _pmax = _pmin + 1e-3
                    _nb = 50
                    _edges = np.linspace(_pmin, _pmax, _nb + 1)
                    _centers = (_edges[:-1] + _edges[1:]) / 2
                    _vol = np.zeros(_nb)
                    for _, _row in _cd.iterrows():
                        _lo, _hi, _v = float(_row["low"]), float(_row["high"]), float(_row["volume"])
                        if _v <= 0:
                            continue
                        if _hi <= _lo:
                            _hi = _lo + 1e-3
                        _ia = int(np.searchsorted(_edges, _lo, side="right") - 1)
                        _ib = int(np.searchsorted(_edges, _hi, side="right") - 1)
                        _ia = max(0, min(_nb - 1, _ia))
                        _ib = max(0, min(_nb - 1, _ib))
                        if _ib < _ia:
                            _ib = _ia
                        _seg = np.zeros(_nb)
                        _seg[_ia:_ib + 1] = 1.0
                        _vol += _seg * _v / (_ib - _ia + 1)
                    _tot = _vol.sum()
                    if _tot > 0:
                        _pdf = _vol / _tot
                        _lastc = float(df["close"].iloc[-1])
                        _avg_cost = float((_pdf * _centers).sum())
                        _profit = float(_pdf[_centers < _lastc].sum())
                        _cum = np.cumsum(_pdf)
                        _lo90 = float(_centers[min(_nb - 1, int(np.searchsorted(_cum, 0.05)))])
                        _hi90 = float(_centers[min(_nb - 1, int(np.searchsorted(_cum, 0.95)))])
                        _lo70 = float(_centers[min(_nb - 1, int(np.searchsorted(_cum, 0.15)))])
                        _hi70 = float(_centers[min(_nb - 1, int(np.searchsorted(_cum, 0.85)))])
                        _conc90 = (_hi90 - _lo90) / (_hi90 + _lo90) * 100 if (_hi90 + _lo90) > 0 else 0
                        _conc70 = (_hi70 - _lo70) / (_hi70 + _lo70) * 100 if (_hi70 + _lo70) > 0 else 0
                        st.markdown('<div class="chart-box"><h3>🧊 筹码分布（近 120 日成本）</h3>',
                                    unsafe_allow_html=True)
                        _cols = ["#ef4444" if c < _lastc else "#10b981" for c in _centers]
                        fig_chip = go.Figure()
                        fig_chip.add_trace(go.Bar(y=_centers, x=_pdf * 100, orientation="h",
                                                 marker_color=_cols, opacity=0.75))
                        fig_chip.add_hline(y=_lastc, line_dash="dash", line_color="#1e293b", opacity=0.8,
                                           annotation_text=f"现价 {_lastc:.2f}",
                                           annotation_position="top right")
                        fig_chip.add_hline(y=_avg_cost, line_dash="dot", line_color="#8b5cf6", opacity=0.8,
                                           annotation_text=f"平均成本 {_avg_cost:.2f}",
                                           annotation_position="bottom left")
                        # 标注落在绘图区内会压住筹码柱，加不透明底衬才读得出来
                        for _an in fig_chip.layout.annotations:
                            _an.update(bgcolor="rgba(255,255,255,.9)", borderpad=2)
                        fig_chip.update_layout(height=360, template="plotly_white",
                                               margin=dict(l=10, r=14, t=10, b=10),
                                               xaxis_title="筹码占比(%)", yaxis_title="价格")
                        st.plotly_chart(fig_chip, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        # 四行裸文本 → 定宽小格，和基本面/风险指标同一套观感
                        st.markdown(panel("🎯 筹码摘要", _ana_kv([
                            ("平均成本", f"{_avg_cost:.2f}", "#8b5cf6"),
                            ("获利盘", f"{_profit*100:.1f}%", "#dc2626"),
                            ("90% 成本区间", f"{_lo90:.2f} ~ {_hi90:.2f}", ""),
                            ("90% 集中度", f"{_conc90:.1f}%", ""),
                            ("70% 成本区间", f"{_lo70:.2f} ~ {_hi70:.2f}", ""),
                            ("70% 集中度", f"{_conc70:.1f}%", ""),
                        ])), unsafe_allow_html=True)
                    else:
                        st.caption("⚠️ 成交量数据不足，无法绘制筹码分布")
                except Exception as _e:
                    st.caption(f"⚠️ 筹码分布绘制失败: {_e}")


            with _tab_fin:
                # ====== 📈 财务趋势 ======
                if use_fund and "fin" in dir() and fin is not None and len(fin) > 1:
                    try:
                        _f = fin.copy()
                        if "日期" in _f.columns:
                            _f = _f.sort_values("日期")
                        _periods = _f.tail(8)
                        _x = _periods["日期"].astype(str).tolist() if "日期" in _f.columns else [str(i) for i in range(len(_periods))]

                        def _ser(key):
                            if key in _periods.columns:
                                return pd.to_numeric(_periods[key], errors="coerce")
                            return None

                        _pe = _ser("pe_ttm")
                        _pb = _ser("pb")
                        _roe = _ser("roe")
                        _pg = _ser("profit_growth")
                        _rg = _ser("revenue_growth")

                        st.markdown('<div class="chart-box"><h3>📈 财务趋势（近 8 期）</h3>', unsafe_allow_html=True)
                        fig_fin = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                                                subplot_titles=("估值 PE(TTM) / PB", "盈利 ROE / 增速(%)"),
                                                row_heights=[0.5, 0.5])
                        if _pe is not None and _pb is not None:
                            fig_fin.add_trace(go.Scatter(x=_x, y=_pe.fillna(0), name="PE(TTM)", line=dict(color="#ef4444", width=1.8)), row=1, col=1)
                            fig_fin.add_trace(go.Scatter(x=_x, y=_pb.fillna(0), name="PB", line=dict(color="#3b82f6", width=1.8)), row=1, col=1)
                        if _roe is not None:
                            fig_fin.add_trace(go.Scatter(x=_x, y=_roe.fillna(0), name="ROE%", line=dict(color="#8b5cf6", width=1.8)), row=2, col=1)
                        if _pg is not None:
                            fig_fin.add_trace(go.Scatter(x=_x, y=_pg.fillna(0), name="净利增速%", line=dict(color="#059669", width=1.5)), row=2, col=1)
                        if _rg is not None:
                            fig_fin.add_trace(go.Scatter(x=_x, y=_rg.fillna(0), name="营收增速%", line=dict(color="#f59e0b", width=1.5)), row=2, col=1)
                        fig_fin.update_layout(height=360, template="plotly_white", hovermode="x unified",
                                              legend=dict(orientation="h", y=1.18), margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_fin, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as _e:
                        st.caption(f"⚠️ 财务趋势绘制失败: {_e}")


            with _tab_rs:
                # ====== 个股 vs 大盘 vs 行业 相对强弱 ======
                try:
                    _bench = _cached_benchmark("000300.SH", (datetime.now() - timedelta(days=days)).strftime("%Y%m%d"))
                    if len(_bench) > 5:
                        _bclose = _bench["close"]

                        def _ret(s, n):
                            if len(s) <= n:
                                return float(s.iloc[-1] / s.iloc[0] - 1) * 100
                            return float(s.iloc[-1] / s.iloc[-1 - n] - 1) * 100

                        _stock_ret = {n: _ret(close, n) for n in (5, 20, 60, 120)}
                        _bench_ret = {n: _ret(_bclose, n) for n in (5, 20, 60, 120)}
                        # 行业板块涨跌幅
                        _ind_chg = None
                        try:
                            _secs = _cached_hotspots()
                            _ind = base.get("industry", "")
                            if _ind and _ind != "未知":
                                _m = _secs[_secs["板块"].str.contains(_ind[:2], na=False)]
                                if len(_m) > 0:
                                    _ind_chg = float(_m.iloc[0]["涨跌幅"])
                        except Exception:
                            _ind_chg = None

                        # 原来一行行「个股 xx% · 沪深300 xx% · 相对 xx%」拼字符串，列不对齐、
                        # 数字宽度不统一，读起来很费劲；改成同宽表格，一眼横向比。
                        _rs_rows = ""
                        for n in (5, 20, 60, 120):
                            sr = _stock_ret[n]
                            br = _bench_ret[n]
                            rs = sr - br
                            ic = f"{_ind_chg:+.2f}%" if _ind_chg is not None else "—"
                            _cc = lambda _v: "#dc2626" if _v > 0 else "#059669"
                            _rs_rows += (
                                f'<tr><td><b>近 {n} 日</b></td>'
                                f'<td style="color:{_cc(sr)};font-weight:700;">{sr:+.2f}%</td>'
                                f'<td style="color:{_cc(br)};">{br:+.2f}%</td>'
                                f'<td style="color:{_cc(rs)};font-weight:700;">{rs:+.2f}%</td>'
                                f'<td>{ic}</td></tr>'
                            )
                        st.markdown(
                            '<div class="panel"><div class="panel-title">📡 相对强弱 · 个股 vs 沪深300 vs 行业</div>'
                            '<table class="mini-table"><thead><tr>'
                            '<th>区间</th><th>个股</th><th>沪深300</th><th>超额</th><th>行业</th>'
                            f'</tr></thead><tbody>{_rs_rows}</tbody></table></div>',
                            unsafe_allow_html=True,
                        )

                        # 相对强弱曲线（归一化到 100）
                        try:
                            _bn = min(len(close), len(_bclose))
                            _snorm = (close.iloc[-_bn:].values / close.iloc[-_bn:].iloc[0]) * 100
                            _bnorm = (_bclose.iloc[-_bn:].values / _bclose.iloc[-_bn:].iloc[0]) * 100
                            _bx = list(_bench["date"].iloc[-_bn:])
                            fig_rs = go.Figure()
                            fig_rs.add_trace(go.Scatter(x=_bx, y=_snorm, name=stock_name, line=dict(color="#ef4444", width=1.8)))
                            fig_rs.add_trace(go.Scatter(x=_bx, y=_bnorm, name="沪深300", line=dict(color="#3b82f6", width=1.8)))
                            fig_rs.update_layout(height=300, template="plotly_white", hovermode="x unified",
                                                 legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=20, b=10),
                                                 yaxis_title="归一化(起点=100)")
                            # 裸 plotly 图直接落在灰底上会显得"没装修"，套一层白卡
                            st.markdown('<div class="chart-box"><h3>📡 归一化走势（区间起点 = 100）</h3>',
                                        unsafe_allow_html=True)
                            st.plotly_chart(fig_rs, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception:
                            pass
                    else:
                        st.caption("⚠️ 基准指数数据不足，跳过相对强弱比较")
                except Exception as _e:
                    st.caption(f"⚠️ 相对强弱比较失败: {_e}")


            with _tab_sig:
                # ====== 🎯 相似信号历史回测 ======
                try:
                    _bc = min(len(df), 250)
                    _sub = df.iloc[-_bc:].copy()
                    _c = _sub["close"]
                    _ma5 = _c.rolling(5).mean()
                    _ma20 = _c.rolling(20).mean()
                    _ma60 = _c.rolling(60).mean()
                    _rsi = calc_rsi(_c, 14)
                    _k, _d, _j = calc_kdj(_sub)
                    _bm, _bu, _bl = calc_boll(_c)
                    _bpos = (_c - _bl) / (_bu - _bl + 1e-10)
                    _f5 = (_c.shift(-5) / _c - 1) * 100
                    _f10 = (_c.shift(-10) / _c - 1) * 100
                    _f20 = (_c.shift(-20) / _c - 1) * 100

                    def _stats(mask):
                        m = mask.fillna(False) & _f5.notna() & _f10.notna() & _f20.notna()
                        if int(m.sum()) < 5:
                            return None
                        r5 = _f5[m]
                        r10 = _f10[m]
                        r20 = _f20[m]
                        return dict(n=int(m.sum()),
                                    r5=float(r5.mean()), r10=float(r10.mean()), r20=float(r20.mean()),
                                    w5=float((r5 > 0).mean() * 100))

                    _conds = {
                        "多头排列": (_c > _ma5) & (_ma5 > _ma20) & (_ma20 > _ma60),
                        "空头排列": (_c < _ma5) & (_ma5 < _ma20) & (_ma20 < _ma60),
                        "RSI超卖(<30)": _rsi < 30,
                        "RSI超买(>70)": _rsi > 70,
                        "MA5金叉MA20": (_ma5 > _ma20) & (_ma5.shift(1) <= _ma20.shift(1)),
                        "MA5死叉MA20": (_ma5 < _ma20) & (_ma5.shift(1) >= _ma20.shift(1)),
                        "KDJ金叉": (_k > _d) & (_k.shift(1) <= _d.shift(1)),
                        "触及BOLL下轨": _bpos < 0.05,
                        "触及BOLL上轨": _bpos > 0.95,
                    }
                    _active = {k: bool(v.iloc[-1]) for k, v in _conds.items()}
                    _rows = []
                    for k, v in _conds.items():
                        s = _stats(v)
                        if s is None:
                            continue
                        mark = "★" if _active.get(k) else ""
                        _rows.append((f"{mark}{k}", s["n"], s["r5"], s["r10"], s["r20"], s["w5"]))
                    if _rows:
                        # markdown 管道表在 Streamlit 里是通用样式（无对齐、无 hover），
                        # 换成和其他地方同一套 .mini-table，数字右对齐才看得清。
                        _bt_rows = ""
                        for r in _rows:
                            _cur = str(r[0]).startswith("★")
                            _sig = str(r[0])
                            _c5 = "#dc2626" if r[2] > 0 else "#059669"
                            _c10 = "#dc2626" if r[3] > 0 else "#059669"
                            _c20 = "#dc2626" if r[4] > 0 else "#059669"
                            _cw = "#dc2626" if r[5] >= 50 else "#059669"
                            _bg = ' style="background:#faf5ff;"' if _cur else ""
                            _bt_rows += (
                                f'<tr{_bg}><td>{_sig}</td>'
                                f'<td>{r[1]}</td>'
                                f'<td style="color:{_c5};font-weight:600;">{r[2]:+.2f}%</td>'
                                f'<td style="color:{_c10};">{r[3]:+.2f}%</td>'
                                f'<td style="color:{_c20};">{r[4]:+.2f}%</td>'
                                f'<td style="color:{_cw};font-weight:600;">{r[5]:.0f}%</td></tr>'
                            )
                        st.markdown(
                            '<div class="panel"><div class="panel-title">'
                            '🎯 历史相似技术信号回测（近 250 日）</div>'
                            '<div class="text-sub" style="margin-bottom:.4rem;">'
                            '统计该信号出现后未来 N 日的平均涨跌与胜率；高亮行 = 当前正满足。</div>'
                            '<table class="mini-table"><thead><tr>'
                            '<th>信号</th><th>样本</th><th>未来5日</th><th>未来10日</th>'
                            '<th>未来20日</th><th>5日胜率</th>'
                            f'</tr></thead><tbody>{_bt_rows}</tbody></table></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("⚠️ 样本不足，无法回测")
                except Exception as _e:
                    st.caption(f"⚠️ 相似信号回测失败: {_e}")



            # ====== 风险评估 / 多周期统计（双列卡片）======
            # VaR (95%)
            var95 = float(np.percentile(returns.tail(60), 5)) * 100
            # 最大回撤
            peak = close.expanding().max()
            drawdown = (close - peak) / peak
            max_dd = float(drawdown.min()) * 100
            # 夏普比率
            rf_daily = 0.02 / 252
            excess = returns.tail(60) - rf_daily
            sharpe = float(excess.mean() / (excess.std() + 1e-10) * np.sqrt(252))
            # 波动率
            ann_vol = float(returns.tail(60).std() * np.sqrt(252) * 100)
            # 胜率
            win_rate = float((returns.tail(60) > 0).mean()) * 100

            _rk, _st = st.columns([1, 1])

            with _rk:
                _risk_items = [
                    ("VaR(95%)", f"{var95:.2f}%",
                     "#dc2626" if var95 < -3 else ("#d97706" if var95 < -1.5 else "#059669")),
                    ("最大回撤", f"{max_dd:.1f}%",
                     "#dc2626" if max_dd < -25 else ("#d97706" if max_dd < -10 else "#059669")),
                    ("夏普比率", f"{sharpe:.2f}",
                     "#dc2626" if sharpe > 1 else ("#d97706" if sharpe > 0 else "#059669")),
                    ("年化波动", f"{ann_vol:.1f}%", ""),
                    ("60日胜率", f"{win_rate:.0f}%",
                     "#dc2626" if win_rate > 55 else ("#d97706" if win_rate > 45 else "#059669")),
                ]
                st.markdown(
                    panel("⚠️ 风险评估", _ana_kv(_risk_items)
                          + '<div class="text-sub" style="margin-top:.45rem;">'
                            'VaR/波动/胜率取近 60 个交易日；夏普按无风险利率 2% 年化。</div>'),
                    unsafe_allow_html=True,
                )

            with _st:
                # 多周期统计：原来每个周期平铺 7 张卡（3×7=21 张），满屏色块反而看不清，
                # 改成一张 3 行的紧凑表 —— 同行横向比较才是这里真正要读的东西。
                _head = ("周期", "涨跌幅", "最高", "最低", "振幅", "波动率", "胜率")
                _rows_html = ""
                for label, period_data in [("近 5 日", df.tail(5)), ("近 20 日", df.tail(20)),
                                           ("近 60 日", df.tail(60))]:
                    cd = period_data["close"]
                    ret = (cd.iloc[-1] / cd.iloc[0] - 1) * 100
                    high_p = cd.max()
                    low_p = cd.min()
                    amp = (period_data["high"].max() - period_data["low"].min()) / cd.iloc[0] * 100
                    vol_price = cd.pct_change().std() * 100
                    win = (cd.pct_change().dropna() > 0).mean() * 100
                    _rc = "#dc2626" if ret > 0 else "#059669"
                    _wc = "#dc2626" if win > 50 else "#059669"
                    _rows_html += (
                        f'<tr><td><b>{label}</b></td>'
                        f'<td style="color:{_rc};font-weight:700;">{ret:+.2f}%</td>'
                        f'<td style="color:#dc2626;">{high_p:.2f}</td>'
                        f'<td style="color:#059669;">{low_p:.2f}</td>'
                        f'<td>{amp:.2f}%</td><td>{vol_price:.2f}%</td>'
                        f'<td style="color:{_wc};font-weight:600;">{win:.0f}%</td></tr>'
                    )
                _th = "".join(f"<th>{h}</th>" for h in _head)
                st.markdown(
                    panel("📊 多周期统计",
                          f'<table class="mini-table"><thead><tr>{_th}</tr></thead>'
                          f'<tbody>{_rows_html}</tbody></table>'),
                    unsafe_allow_html=True,
                )

            # ====== AI特征贡献 ======
            if predictor.is_ready and pred.features_snapshot:
                feat = pred.features_snapshot
                meaningful = {k: v for k, v in feat.items() if abs(v) > 1e-6}
                # 特征量纲差异极大（vol_change 可到 90 万，kdj_j 只有几十），
                # 直接画原始值会被单个离群特征压成一根横线，其余 14 根全是 0 长度。
                # 这里按「中位数的 20 倍」截断离群点，并在图下注明剔除了几个。
                _dropped = 0
                if len(meaningful) > 4:
                    _med = float(np.median([abs(v) for v in meaningful.values()]))
                    if _med > 0:
                        _cap = _med * 20.0
                        _kept = {k: v for k, v in meaningful.items() if abs(v) <= _cap}
                        if len(_kept) >= 5:
                            _dropped = len(meaningful) - len(_kept)
                            meaningful = _kept
                st.markdown('<div class="chart-box"><h3>🧬 AI 特征快照 Top 15</h3>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<div class="text-sub" style="margin-bottom:.4rem;">'
                    '红=看涨方向，绿=看跌方向；绝对值越大的特征对本次评分影响越强。'
                    + (f'（已隐藏 {_dropped} 个量纲离群、会压扁坐标轴的特征）' if _dropped else '')
                    + '</div>',
                    unsafe_allow_html=True,
                )
                feat_df = pd.DataFrame(
                    sorted(meaningful.items(), key=lambda x: abs(x[1]), reverse=True)[:15],
                    columns=["特征", "当前值"]
                )
                feat_df["方向"] = feat_df["当前值"].apply(lambda x: "#ef4444" if x > 0 else "#10b981")

                fig4 = go.Figure(data=[go.Bar(
                    x=feat_df["当前值"], y=feat_df["特征"], orientation="h",
                    marker_color=feat_df["方向"],
                    text=feat_df["当前值"].round(3), textposition="outside",
                    textfont=dict(color="#1e293b", size=11),
                    cliponaxis=False,
                )])
                fig4.update_layout(height=420, template="plotly_white",
                                   xaxis_title="特征值 (红=看涨, 绿=看跌)",
                                   xaxis=dict(zeroline=True, zerolinecolor="#94a3b8"),
                                   # 倒序 y 轴：影响最大的特征排在最上面，不用往下找
                                   yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                                   margin=dict(l=10, r=60, t=10, b=10))
                st.plotly_chart(fig4, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ====== 🤖 AI 分析报告 ======
            # 同上：模型来源动态显示，避免本地模式下仍写「DeepSeek」造成误解。
            _llm_name = "本地大模型" if getattr(deepseek, "is_local", False) else "DeepSeek"
            st.markdown(f'<div class="section-title">🤖 {_llm_name} AI 分析报告</div>', unsafe_allow_html=True)
            if not deepseek.is_ready:
                st.info(f"💡 未配置{_llm_name}，请到「⚙️ 设置」页配置后即可生成 AI 分析报告。")
            else:
                ds_btn = st.button(f"✨ 生成 {_llm_name} 分析报告", type="primary", key=f"ds_report_{stock_code}")
                if ds_btn:
                    with st.spinner(f"🤖 {_llm_name} 正在分析…"):
                        try:
                            # 组装指标快照
                            ds_metrics = {
                                "收盘价": f"{last_close:.2f}",
                                "当日涨跌": f"{(last_close-prev_close):+.2f} ({(last_close/prev_close-1)*100:+.2f}%)",
                                "MA5/MA20/MA60": f"{ma5_now:.2f} / {ma20_now:.2f} / {ma60_now:.2f}",
                                "RSI14": f"{rsi14_now:.1f}",
                                "KDJ(K/D/J)": f"{k_now:.0f} / {d_now:.0f} / {j_now:.0f}",
                                "布林带位置": f"{boll_pos*100:.0f}%",
                                "量能比": f"{vol_ratio:.1f}x",
                                "年化波动": f"{ann_vol:.1f}%",
                                "最大回撤": f"{max_dd:.1f}%",
                                "夏普比率": f"{sharpe:.2f}",
                                "60日胜率": f"{win_rate:.0f}%",
                            }
                            if predictor.is_ready:
                                ds_metrics["AI评分"] = f"{pred.score:.1f}"
                                ds_metrics["上涨/震荡/下跌概率"] = f"{pred.prob_up*100:.1f}% / {pred.prob_neutral*100:.1f}% / {pred.prob_down*100:.1f}%"
                                ds_metrics["AI推荐理由"] = "; ".join(pred.reasons[:4]) if pred.reasons else "—"
                            if fund_items:
                                ds_metrics["基本面"] = "; ".join(f"{l}={v}" for l, v, _ in fund_items[:8])
                            if signals:
                                ds_metrics["技术信号"] = "; ".join(t for _, t, _ in signals[:8])

                            report = deepseek.analyze_stock(stock_name, stock_code, ds_metrics)
                            st.session_state[f"ds_report_{stock_code}"] = report
                        except Exception as e:
                            st.error(f"❌ 生成失败: {e}")

                cached_report = st.session_state.get(f"ds_report_{stock_code}", "")
                if cached_report:
                    st.markdown(cached_report)


# ========== 📡 实时监控页 ==========
elif page == "📡 实时监控":
    import time
    render_hero(
        "AI 实时监控",
        "盯盘片段原地刷新：实时价 · 均线 · AI 评分联动，无需整页重跑",
        icon="📡",
        badges=[("info", "盘中每 60 秒")],
    )

    # 交易时间判断
    now = datetime.now()
    market_open = (now.weekday() < 5 and (
        (now.hour == 9 and now.minute >= 30) or
        (now.hour == 10) or
        (now.hour == 11 and now.minute <= 30) or
        (now.hour == 13) or
        (now.hour == 14) or
        (now.hour == 15 and now.minute == 0)
    ))

    # 状态栏
    status1, status2, status3 = st.columns(3)
    with status1:
        if market_open:
            st.success(f"🟢 交易中 · {now.strftime('%H:%M:%S')}")
        else:
            st.info(f"⏸️ 非交易时段 · {now.strftime('%H:%M:%S')}")
    with status2:
        if "mon_last_ts" in st.session_state:
            elapsed_s = int((datetime.now() - st.session_state.mon_last_ts).total_seconds())
            st.caption(f"🕐 上次刷新: {elapsed_s}秒前")
    with status3:
        st.caption(f"📡 数据源: {'东方财富(实时)' if market_open else '日线(收盘价)'}")

    # ========== 数据缓存层(大幅提速) ==========
    if "mon_spot_cache" not in st.session_state:
        st.session_state.mon_spot_cache = {}
    if "mon_daily_cache" not in st.session_state:
        st.session_state.mon_daily_cache = {}
    
    # 个股盯盘 ==========
    st.markdown('<div class="section-title">🎯 个股盯盘</div>', unsafe_allow_html=True)

    # 自选股快速选择：点一下即可盯盘，避免「打开页面空空如也」的观感。
    # 背景：盯盘面板原先只在用户手动输入代码并点「开始盯盘」后才渲染，
    # 打开页面时输入框为空、整块面板空置，看起来就像「没有任何数据」。
    try:
        import json as _json
        _wl_cfg = _json.loads((Path(__file__).parent.parent.parent / "config" / "watchlist.json").read_text(encoding="utf-8"))
        _wl = list(_wl_cfg.get("watchlist", []))
    except Exception:
        _wl = []
    if _wl:
        st.caption("⭐ 自选股快速盯盘：")
        _wl_cols = st.columns(min(len(_wl), 8))
        for i, _c in enumerate(_wl):
            with _wl_cols[i % 8]:
                if st.button(_c, key=f"wlpick_{_c}", use_container_width=True):
                    st.session_state.watch_code = _c
                    st.session_state.pop("watch_last_ts", None)  # 强制立即盯盘该票
                    st.rerun()

    mon1, mon2, mon3, mon4, mon5, mon6 = st.columns([2.0, 1.1, 1.0, 1.0, 0.9, 0.9])
    with mon1:
        # 默认带出上次盯盘的股票；首次打开则带出自选股第一只，避免面板空置
        _default = st.session_state.get("watch_code") or _load_last_watched() or (_wl[0] if _wl else "")
        watch_code = st.text_input("输入股票代码盯盘", value=_default, placeholder="例: 000001, 600519", key="watch_code", label_visibility="collapsed")
    with mon2:
        watch_go = st.button("🔍 开始盯盘", type="primary", use_container_width=True, key="watch_go")
    with mon3:
        watch_interval = st.select_slider("行情刷新(秒)", options=[5, 10, 15, 30, 60], value=10, key="watch_interval")
    with mon4:
        watch_ai_interval = st.select_slider("AI重评(秒)", options=[30, 60, 120, 300], value=60, key="watch_ai_interval",
                                             help="AI 模型推理较重，按更长周期重算；行情价格仍按上方间隔实时刷新")
    with mon5:
        watch_auto = st.checkbox("自动刷新", value=True, key="watch_auto")
    with mon6:
        watch_refresh = st.button("🔄 刷新", use_container_width=True, key="watch_refresh")

    render_watch_fragment()

    # ========== 批量监控 ==========
    st.markdown('<div class="section-title">📋 批量监控</div>', unsafe_allow_html=True)

    # 控制栏
    c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1])
    with c1:
        pool = st.selectbox("股票池", ["watchlist", "hs300", "zz500", "custom"], key="mon_pool",
                           format_func=lambda x: {"watchlist": "⭐ 自选股", "hs300": "沪深300", "zz500": "中证500", "custom": "自定义"}[x],
                           help="⭐ 自选股 = config/watchlist.json 中的股票")
    with c2:
        top_n = st.number_input("数量", 5, 30, value=min(st.session_state.get("mon_topn", 10), 30), key="mon_topn")
    with c3:
        min_ai_score = st.slider("最低AI评分", 0, 100, 0, 5, key="mon_min_score", help="过滤低于此评分的股票，0=不过滤")
    with c4:
        signal_filter = st.selectbox("信号筛选", ["全部", "只看涨", "只看跌", "高分(≥65)"], key="mon_filter")
    with c5:
        col_refresh, col_auto = st.columns(2)
        with col_refresh:
            do_refresh = st.button("🔄 刷新", type="primary", use_container_width=True, key="mon_refresh")
        with col_auto:
            auto_on = st.checkbox("自动", value=True, key="mon_auto")

    # 刷新逻辑
    need_refresh = do_refresh
    if auto_on and "mon_last_ts" in st.session_state:
        elapsed = (datetime.now() - st.session_state.mon_last_ts).total_seconds()
        if elapsed > 30:
            need_refresh = True
    if "mon_last_ts" not in st.session_state:
        need_refresh = True

    if need_refresh or (auto_on and "mon_last_ts" not in st.session_state):
        st.session_state.mon_last_ts = datetime.now()

    # 跨页跳转到个股分析。
    # 注意：个股分析页读的 key 是 auto_analyze_code（见该页 st.session_state.pop），
    # 之前这里写的是 _analyze_code，对不上，跳转永远不生效。
    if st.session_state.get("mon_go_code"):
        st.session_state.page = "📊 个股分析"
        st.session_state.auto_analyze_code = st.session_state.mon_go_code
        st.session_state.mon_go_code = ""
        st.rerun()

    # custom 输入框提前渲染(避免点了按钮才出现)
    if pool == "custom":
        custom_codes = st.text_input("输入股票代码（逗号分隔）", "000001,000002,600519", key="mon_custom")

    # 执行刷新
    with st.spinner("正在获取实时数据 + AI 预测…"):
        try:
            # 获取股票列表
            # 从全市场 stock_list 构建 code→name 映射(所有池通用)
            stock_list = _cached_stock_list()
            name_map = dict(zip(stock_list["code"], stock_list["name"]))
            if pool == "custom":
                codes = parse_codes(custom_codes)
                stock_names = {c: name_map.get(c, c) for c in codes}
            elif pool == "watchlist":
                # 从 config/watchlist.json 加载自选股
                import json
                try:
                    config_file = Path(__file__).parent.parent.parent / "config" / "watchlist.json"
                    wl_cfg = json.loads(config_file.read_text(encoding="utf-8"))
                    wl = wl_cfg.get("watchlist", [])
                    codes = list(wl)
                    stock_names = {c: name_map.get(c, c) for c in codes}
                except Exception:
                    codes = []
                    stock_names = {}
            else:
                codes = fetcher.get_stock_pool(pool)[:top_n]
                stock_names = {c: name_map.get(c, c) for c in codes}

            if not codes:
                empty_state("无监控标的，请在上方选择股票池或输入自定义代码", icon="📭", kind="warning")
            else:
                # 获取实时行情（用缓存加速）
                cache_key = f"spot_{now.strftime('%Y%m%d_%H%M')[:-1]}"  # 按10秒粒度缓存
                if cache_key in st.session_state.mon_spot_cache:
                    spot_data = st.session_state.mon_spot_cache[cache_key]
                else:
                    spot_data = {}
                    if market_open:
                        # 用腾讯单只实时接口（稳定快速 ~50ms/只），避免东方财富全市场接口限流
                        for c in codes:
                            q = fetcher.get_realtime_quote(c)
                            if q and q.get("最新价", 0) > 0:
                                spot_data[q.get("代码", c)] = q
                        if spot_data:
                            st.session_state.mon_spot_cache[cache_key] = spot_data
                            # 只保留最近2个缓存
                            old_keys = sorted(st.session_state.mon_spot_cache.keys())[:-2]
                            for ok in old_keys:
                                del st.session_state.mon_spot_cache[ok]

                # 批量预测
                rows = []
                mon_failed = []
                progress = st.progress(0, text="获取数据中…")
                for i, code in enumerate(codes):
                    progress.progress((i + 1) / len(codes), text=f"{code}…")
                    try:
                        # 走 st.cache_data 统一缓存（10 分钟）。
                        # 之前这里用 session_state 手工缓存，但用了
                        # (now - ts).seconds —— 该属性不含天数，跨天时会误判为未过期。
                        df = _cached_daily_kline(code, "", "")
                        if len(df) < 60:
                            mon_failed.append(code)
                            continue
                        name = stock_names.get(code, code)
                        close_col = df["close"]
                        last_close = float(close_col.iloc[-1])
                        prev_close = float(close_col.iloc[-2])
                        change_pct = (last_close / prev_close - 1) * 100

                        if code in spot_data:
                            spot = spot_data[code]
                            latest_price = float(spot.get("最新价", last_close))
                            chg = float(spot.get("涨跌幅", change_pct))
                            vol = int(float(spot.get("成交量", 0))) if spot.get("成交量", 0) else int(df["volume"].iloc[-1])
                            amount = float(spot.get("成交额", 0))
                        else:
                            latest_price = last_close
                            chg = change_pct
                            vol = int(df["volume"].iloc[-1])
                            amount = round(latest_price * vol / 10000, 2) if vol else 0

                        # AI 预测
                        ai_signal = "观望"
                        ai_score = 0
                        prob_up = 0.0
                        prob_down = 0.0
                        conf = 0.0
                        xgb_signal_val = 0
                        if predictor.is_ready:
                            pred = predictor.predict_single(df, code, name)
                            ai_score = pred.score
                            prob_up = pred.prob_up
                            prob_down = pred.prob_down
                            conf = pred.confidence
                            xgb_signal_val = pred.xgb_signal
                            if pred.prob_up > 0.5:
                                ai_signal = "看涨"
                            elif pred.prob_down > 0.5:
                                ai_signal = "看跌"
                            else:
                                ai_signal = "观望"

                        # 5日趋势
                        last5 = close_col.iloc[-5:].values
                        trend_5d = (last5[-1] / last5[0] - 1) * 100

                        rows.append({
                            "代码": code,
                            "名称": name,
                            "现价": latest_price,
                            "涨跌%": chg,
                            "量(万)": vol / 10000,
                            "额(亿)": amount / 1e8 if amount else 0,
                            "AI信号": ai_signal,
                            "AI评分": ai_score,
                            "确信度%": conf * 100,
                            "模型信号": "买入" if xgb_signal_val == 1 else ("卖出" if xgb_signal_val == -1 else "观望"),
                            "5日趋势%": trend_5d,
                            "上涨%": prob_up * 100,
                            "下跌%": prob_down * 100,
                        })
                    except Exception as e:
                        # 之前是裸 except: pass —— 30 只里挂了 20 只用户也看不出来
                        mon_failed.append(code)
                        logger.debug(f"监控页处理 {code} 失败: {e}")
                progress.empty()

                if mon_failed:
                    st.caption(
                        f"⚠️ {len(mon_failed)}/{len(codes)} 只未纳入："
                        + "、".join(mon_failed[:8]) + ("…" if len(mon_failed) > 8 else "")
                        + "（数据不足 60 根K线或获取失败）"
                    )

                if not rows:
                    empty_state("无有效数据：所选标的数据不足或获取失败", icon="📭")
                else:
                    df_mon = pd.DataFrame(rows)

                    # 过滤
                    if min_ai_score > 0:
                        df_mon = df_mon[df_mon["AI评分"] >= min_ai_score]
                    if signal_filter == "只看涨":
                        df_mon = df_mon[df_mon["AI信号"] == "看涨"]
                    elif signal_filter == "只看跌":
                        df_mon = df_mon[df_mon["AI信号"] == "看跌"]
                    elif signal_filter == "高分(≥65)":
                        df_mon = df_mon[df_mon["AI评分"] >= 65]

                    if len(df_mon) == 0:
                        empty_state("过滤后无股票，请放宽筛选条件", icon="🔍", kind="warning")
                    else:
                        # 摘要栏
                        buys = (df_mon["AI信号"] == "看涨").sum()
                        sells = (df_mon["AI信号"] == "看跌").sum()
                        avg_score = df_mon["AI评分"].mean()
                        avg_chg = df_mon["涨跌%"].mean()
                        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                        mc1.metric("📈 看涨", buys, delta=f"{buys/len(df_mon)*100:.0f}%" if len(df_mon) > 0 else None)
                        mc2.metric("📉 看跌", sells)
                        mc3.metric("➡️ 观望", len(df_mon) - buys - sells)
                        mc4.metric("🧠 均AI评分", f"{avg_score:.0f}")
                        with mc5:
                            metric_card("📊 均涨跌", f"{avg_chg:+.2f}%", chg=avg_chg)

                        # 图表行
                        ch1, ch2 = st.columns([1, 1])
                        with ch1:
                            # 信号分布饼图
                            sig_counts = df_mon["AI信号"].value_counts()
                            pie_data = []
                            for s, color in [("看涨", "#ef4444"), ("看跌", "#059669"), ("观望", "#6b7280")]:
                                if s in sig_counts.index:
                                    pie_data.append(go.Pie(
                                        labels=["看涨", "看跌", "观望"],
                                        values=[sig_counts.get("看涨", 0), sig_counts.get("看跌", 0), sig_counts.get("观望", 0)],
                                        hole=0.5, textinfo="label+value",
                                        marker_colors=["#ef4444", "#059669", "#6b7280"],
                                    ))
                                    break
                            if pie_data:
                                fig_p = go.Figure(data=pie_data)
                                fig_p.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10),
                                                    template="plotly_white", title="AI 信号分布")
                            else:
                                fig_p = go.Figure()
                                fig_p.update_layout(height=280)
                            st.plotly_chart(fig_p, use_container_width=True)

                        with ch2:
                            # 评分分布柱状图
                            top_plot = df_mon.nlargest(12, "AI评分")
                            fig_bar = go.Figure(data=[go.Bar(
                                x=top_plot["AI评分"], y=top_plot["名称"], orientation="h",
                                marker=dict(color=top_plot["AI评分"],
                                            colorscale=[[0, "#6b7280"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                                            showscale=True, colorbar=dict(title="评分", thickness=10)),
                                text=top_plot["AI评分"].round(0).astype(int), textposition="outside",
                                textfont=dict(color="#1e293b", size=11),
                            )])
                            fig_bar.update_layout(height=280, template="plotly_white", title="🏆 评分 Top 12",
                                                  xaxis=dict(range=[0, 100]), yaxis=dict(autorange="reversed"),
                                                  margin=dict(l=10, r=30, t=20, b=10))
                            st.plotly_chart(fig_bar, use_container_width=True)

                        # 涨跌 vs AI评分散点
                        ch3, ch4 = st.columns([1.5, 1])
                        with ch3:
                            fig_scatter = go.Figure(data=[go.Scatter(
                                x=df_mon["涨跌%"], y=df_mon["AI评分"],
                                mode="markers+text",
                                text=df_mon["名称"].str[:4],
                                textposition="top center", textfont=dict(size=8, color="#475569"),
                                marker=dict(
                                    size=df_mon["AI评分"].apply(lambda x: max(8, min(30, x / 3))),
                                    color=df_mon["涨跌%"],
                                    colorscale=[[0, "#059669"], [0.5, "#94a3b8"], [1, "#ef4444"]],
                                    showscale=True, colorbar=dict(title="涨跌%", thickness=10),
                                    line=dict(width=0.5, color="#fff")
                                ),
                                hovertemplate="<b>%{text}</b><br>AI评分: %{y:.0f}<br>涨跌: %{x:+.2f}%<extra></extra>"
                            )])
                            fig_scatter.add_hline(y=65, line_dash="dash", line_color="#f59e0b", opacity=0.5, annotation_text="高分线")
                            fig_scatter.add_vline(x=0, line_dash="solid", line_color="#94a3b8", opacity=0.3)
                            fig_scatter.update_layout(height=300, template="plotly_white",
                                                      title="涨跌 vs AI评分 (气泡=评分大小)",
                                                      xaxis_title="涨跌幅%", yaxis_title="AI 评分",
                                                      margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(fig_scatter, use_container_width=True)

                        with ch4:
                            # Top5 / Bottom5
                            top5 = df_mon.nlargest(5, "涨跌%")
                            bot5 = df_mon.nsmallest(5, "涨跌%")
                            st.markdown('<p style="font-weight:600;font-size:0.85rem;color:#475569;margin:0;">🔥 涨幅 Top 5</p>', unsafe_allow_html=True)
                            for _, r in top5.iterrows():
                                st.markdown(f'<div style="display:flex;justify-content:space-between;padding:3px 8px;font-size:0.78rem;"><span style="color:#1e293b;">{r["名称"][:6]} {r["代码"]}</span><span style="color:#ef4444;font-weight:600;">{r["涨跌%"]:+.2f}%</span></div>', unsafe_allow_html=True)
                            st.markdown('<p style="font-weight:600;font-size:0.85rem;color:#475569;margin:8px 0 0 0;">❄️ 跌幅 Top 5</p>', unsafe_allow_html=True)
                            for _, r in bot5.iterrows():
                                st.markdown(f'<div style="display:flex;justify-content:space-between;padding:3px 8px;font-size:0.78rem;"><span style="color:#1e293b;">{r["名称"][:6]} {r["代码"]}</span><span style="color:#059669;font-weight:600;">{r["涨跌%"]:+.2f}%</span></div>', unsafe_allow_html=True)

                        # 主表格
                        st.markdown('<div class="section-title">📋 监控清单</div>', unsafe_allow_html=True)
                        st.caption("💡 在下方选择任意一只股票，点「🔍 分析该股」即可跳转个股分析 | 按列头可排序")

                        # 真正的跳转入口（此前只有一行「点击名称可跳转」的提示，
                        # 但 dataframe 并不支持点击跳转，属于虚假承诺）
                        _jc1, _jc2 = st.columns([3, 1])
                        with _jc1:
                            _jump_code = st.selectbox(
                                "跳转到个股分析",
                                options=df_mon["代码"].tolist(),
                                format_func=lambda c: f"{c} {dict(zip(df_mon['代码'], df_mon['名称'])).get(c, '')}",
                                key="mon_jump_sel", label_visibility="collapsed",
                            )
                        with _jc2:
                            if st.button("🔍 分析该股", use_container_width=True, key="mon_jump_btn"):
                                st.session_state.mon_go_code = _jump_code
                                st.rerun()

                        df_display = df_mon.sort_values("AI评分", ascending=False).copy()
                        df_display["5日趋势%"] = df_display["5日趋势%"].apply(lambda x: f"{x:+.2f}%")
                        df_display["现价"] = df_display["现价"].apply(lambda x: f"{x:.2f}")
                        df_display["涨跌%"] = df_display["涨跌%"].apply(lambda x: f"{x:+.2f}%")
                        df_display["确信度%"] = df_display["确信度%"].apply(lambda x: f"{x:.0f}")
                        df_display["上涨%"] = df_display["上涨%"].apply(lambda x: f"{x:.1f}")
                        df_display["下跌%"] = df_display["下跌%"].apply(lambda x: f"{x:.1f}")

                        st.dataframe(
                            df_display[["代码", "名称", "现价", "涨跌%", "5日趋势%", "AI信号", "AI评分", "确信度%", "模型信号", "上涨%", "下跌%", "量(万)", "额(亿)"]],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "AI评分": st.column_config.ProgressColumn("AI评分", min_value=0, max_value=100, format="%d"),
                                "确信度%": st.column_config.ProgressColumn("确信度%", min_value=0, max_value=100, format="%d"),
                            },
                            height=min(len(df_mon) * 35 + 38, 600)
                        )

                        st.session_state.mon_last_ts = datetime.now()

        except Exception as e:
            st.error(f"监控异常: {type(e).__name__}: {e}")
            import traceback
            st.code(traceback.format_exc())

    # 自动刷新倒计时
    if auto_on and "mon_last_ts" in st.session_state:
        remaining = 30 - int((datetime.now() - st.session_state.mon_last_ts).total_seconds())
        if remaining > 0:
            st.caption(f"⏱️ {remaining} 秒后自动刷新")
        elif remaining <= 0 and not need_refresh:
            time.sleep(0.5)
            st.rerun()

# ========== 🤖 AI 助手页 ==========
elif page == "🤖 AI 助手":
    render_agent_page()

# ========== 💹 自动交易页 ==========
elif page == "💹 自动交易":
    render_autotrade_page()

# ========== 📐 量化模型页 ==========
elif page == "📐 量化模型":
    render_hero(
        "量化策略模型",
        "多因子量化评分 · 均值回归回测 · 风险平价配置",
        icon="📐",
        badges=[("violet", "Quant Engine")],
    )

    quant_tab1, quant_tab2, quant_tab3, quant_tab4, quant_tab5 = st.tabs(
        ["📊 多因子评分", "🔄 均值回归", "⚖️ 风险平价", "🛡️ 策略信誉", "🧮 组合风险"]
    )

    # ========== Tab 1: 多因子评分 ==========
    with quant_tab1:
        qc1, qc2, qc3 = st.columns([1.5, 1, 1])
        with qc1:
            q_pool = st.selectbox("股票池", ["hs300", "zz500", "custom"], key="quant_pool")
        with qc2:
            q_top = st.slider("推荐数量", 5, 50, 20, 5, key="quant_top")
        with qc3:
            q_run = st.button("🧮 运行量化评分", type="primary", use_container_width=True, key="quant_run")

        # custom 输入框提前在按钮外渲染,避免点了按钮才出现
        if q_pool == "custom":
            custom_codes = st.text_input("输入股票代码(逗号分隔)", "000001,600519,300750", key="quant_custom")

        if q_run:
            with st.spinner("正在计算多因子评分…"):
                if q_pool == "custom":
                    codes = parse_codes(custom_codes)
                else:
                    codes = fetcher.get_stock_pool(q_pool)[:50]

                df_map = {}
                q_progress = st.progress(0, text="获取数据…")
                name_map = _cached_name_map()
                failed = []
                for i, code in enumerate(codes):
                    q_progress.progress((i + 1) / len(codes), text=f"{code}…")
                    try:
                        df = _cached_daily_kline(code, "", "")
                        if len(df) >= 120:
                            df_map[code] = (df, lookup_name(name_map, code))
                        else:
                            failed.append(code)
                    except Exception as e:
                        failed.append(code)
                        logger.debug(f"量化页获取 {code} 失败: {e}")
                q_progress.empty()
                if failed:
                    # 之前这里静默丢弃，用户不知道少了哪几只
                    st.caption(
                        f"⚠️ {len(failed)}/{len(codes)} 只数据不足或获取失败："
                        + "、".join(failed[:8]) + ("…" if len(failed) > 8 else "")
                    )

                if df_map:
                    scores = quant_engine.batch_score(df_map, top_n=q_top)
                    st.success(f"✅ 评分完成：{len(scores)} 只股票")

                    # 摘要
                    if scores:
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("🏆 最高分", f"{scores[0].total_score:.0f}", f"{scores[0].name[:6]}")
                        mc2.metric("📊 均分", f"{sum(s.total_score for s in scores) / len(scores):.0f}")
                        mc3.metric("📈 做多信号", sum(1 for s in scores if "做多" in s.signal))
                        mc4.metric("📉 回避信号", sum(1 for s in scores if "回避" in s.signal))

                        # 因子雷达图 (Top 1)
                        if scores:
                            ch_r1, ch_r2 = st.columns([1, 1.5])
                            with ch_r1:
                                top1 = scores[0]
                                st.markdown('<div class="section-title">🏅 最优标的详情</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:1rem;">
                                    <h3 style="color:#5b21b6; margin:0;">{top1.name} ({top1.code})</h3>
                                    <h2 style="color:#ef4444; margin:0.3rem 0; font-size:2rem;">{top1.total_score:.0f}<span style="font-size:1rem; color:#64748b;">/100</span></h2>
                                    <span style="background:#fef2f2; color:#ef4444; padding:2px 8px; border-radius:4px; font-size:0.8rem;">{top1.signal}</span>
                                </div>
                                """, unsafe_allow_html=True)

                                # 五因子柱状图
                                fig_radar = go.Figure(data=[go.Bar(
                                    x=["价值", "动量", "质量", "低波", "趋势"],
                                    y=[top1.value_score, top1.momentum_score, top1.quality_score,
                                       top1.volatility_score, top1.trend_score],
                                    marker_color=["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"],
                                    text=[f"{v:.0f}" for v in [top1.value_score, top1.momentum_score,
                                            top1.quality_score, top1.volatility_score, top1.trend_score]],
                                    textposition="outside", textfont=dict(size=12, color="#1e293b"),
                                )])
                                fig_radar.update_layout(height=250, template="plotly_white",
                                                        yaxis=dict(range=[0, 100]),
                                                        margin=dict(l=10, r=10, t=10, b=30))
                                st.plotly_chart(fig_radar, use_container_width=True)

                                st.markdown('<p style="font-size:0.8rem;color:#64748b;">📊 风险指标</p>', unsafe_allow_html=True)
                                ri1, ri2, ri3 = st.columns(3)
                                ri1.metric("夏普比率", f"{top1.sharpe:.2f}")
                                ri2.metric("最大回撤", f"{top1.max_drawdown:.1%}")
                                ri3.metric("胜率", f"{top1.win_rate:.1%}")

                            with ch_r2:
                                st.markdown('<div class="section-title">📋 量化评分排序</div>', unsafe_allow_html=True)
                                quant_rows = []
                                for s in scores:
                                    quant_rows.append({
                                        "代码": s.code, "名称": s.name[:8],
                                        "总分": s.total_score, "价值": s.value_score,
                                        "动量": s.momentum_score, "质量": s.quality_score,
                                        "低波": s.volatility_score, "趋势": s.trend_score,
                                        "信号": s.signal
                                    })
                                df_quant = pd.DataFrame(quant_rows)

                                # 热力图样式的表格
                                def color_score(val):
                                    if val >= 75: return "background-color:#fef2f2;color:#ef4444;font-weight:600"
                                    if val >= 60: return "background-color:#fef3c7;color:#d97706"
                                    if val >= 45: return "background-color:#f8fafc;color:#475569"
                                    return "background-color:#f0fdf4;color:#059669"

                                st.dataframe(
                                    df_quant.style.map(lambda x: color_score(x) if isinstance(x, (int, float)) else "", subset=["总分"]),
                                    use_container_width=True, hide_index=True, height=min(len(df_quant) * 37 + 38, 550)
                                )

    # ========== Tab 2: 均值回归 ==========
    with quant_tab2:
        st.markdown('<div class="section-title">🔄 均值回归策略回测</div>', unsafe_allow_html=True)
        st.caption("基于布林带偏离的均值回归信号：超卖买入，回归中线卖出")

        mr1, mr2, mr3, mr4 = st.columns([1.5, 1, 1, 1])
        with mr1:
            mr_code = st.text_input("股票代码", "000001", key="mr_code")
        with mr2:
            mr_lookback = st.slider("布林带周期", 10, 60, 20, 5, key="mr_lb")
        with mr3:
            mr_entry = st.slider("入场阈值(σ)", 1.0, 3.5, 2.0, 0.5, key="mr_entry")
        with mr4:
            mr_run = st.button("📈 运行回测", type="primary", use_container_width=True, key="mr_run")

        if mr_run and mr_code:
            code = mr_code.strip().upper()
            if not re.match(r'^\d{6}$', code):
                st.warning("请输入6位股票代码")
            else:
                with st.spinner(f"回测 {code} 均值回归策略…"):
                    df = fetcher.get_daily_kline(code)
                    if len(df) < 120:
                        st.warning(f"{code} 数据不足 (需≥120日)")
                    else:
                        strategy = MeanReversionStrategy(lookback=mr_lookback, 
                                                         entry_threshold=mr_entry)
                        result = strategy.backtest(df)

                        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                        color_ret = "red" if result["total_return"] > 0 else "green"
                        rc1.metric("策略收益", f"{result['total_return']:+.2f}%")
                        rc2.metric("买入持有", f"{result['buy_hold_return']:+.2f}%")
                        rc3.metric("超额Alpha", f"{result['alpha']:+.2f}%", 
                                   delta=f"{'+优于' if result['alpha'] > 0 else '弱于'}基准")
                        rc4.metric("交易次数", result["n_trades"])
                        rc5.metric("最大回撤", f"{result['max_drawdown']:.2f}%")

                        # 交易记录图
                        signals = strategy.generate_signals(df).iloc[-200:]
                        buy_signals = signals[signals["signal"] == 1]
                        sell_signals = signals[signals["signal"] == -1]

                        fig_mr = go.Figure()
                        fig_mr.add_trace(go.Scatter(x=df["date"].iloc[-200:], y=df["close"].iloc[-200:],
                                                     mode="lines", name="收盘价", line=dict(color="#475569", width=1)))
                        fig_mr.add_trace(go.Scatter(x=buy_signals["date"], y=df.loc[buy_signals.index, "close"],
                                                     mode="markers", name="买入", marker=dict(color="#ef4444", size=8, symbol="triangle-up")))
                        fig_mr.add_trace(go.Scatter(x=sell_signals["date"], y=df.loc[sell_signals.index, "close"],
                                                     mode="markers", name="卖出", marker=dict(color="#059669", size=8, symbol="triangle-down")))
                        fig_mr.update_layout(height=350, template="plotly_white",
                                            title=f"{code} 均值回归交易信号 (最近200日)",
                                            margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_mr, use_container_width=True)

                        if result["trades"]:
                            st.markdown('<div class="section-title">📝 交易记录</div>', unsafe_allow_html=True)
                            df_trades = pd.DataFrame(result["trades"])
                            st.dataframe(df_trades, use_container_width=True, hide_index=True)

    # ========== Tab 3: 风险平价 ==========
    with quant_tab3:
        st.markdown('<div class="section-title">⚖️ 风险平价配置</div>', unsafe_allow_html=True)
        st.caption("等风险贡献组合构建：给低波动资产更高权重，分散风险")

        rp1, rp2 = st.columns([1.5, 1])
        with rp1:
            rp_codes = st.text_input("组合股票(逗号分隔)", "000001,600519,000858,300750,601318", key="rp_codes")
        with rp2:
            rp_run = st.button("⚖️ 计算风险平价", type="primary", use_container_width=True, key="rp_run")

        if rp_run:
            codes = parse_codes(rp_codes)
            if len(codes) < 2:
                st.warning("请至少输入2只股票")
            else:
                with st.spinner("计算风险平价权重…"):
                    rets_dict = {}
                    rp_failed = []
                    for code in codes:
                        try:
                            df = _cached_daily_kline(code, "", "")
                            if len(df) > 20:
                                rets_dict[code] = df["close"].pct_change().dropna()
                            else:
                                rp_failed.append(code)
                        except Exception as e:
                            rp_failed.append(code)
                            logger.debug(f"风险平价：{code} 获取失败 {e}")
                    if rp_failed:
                        st.caption(f"⚠️ {len(rp_failed)} 只无数据已跳过：{'、'.join(rp_failed[:6])}")

                    if len(rets_dict) < 2:
                        st.warning("有效数据不足")
                    else:
                        rets_df = pd.DataFrame(rets_dict).iloc[-252:]  # 近252日

                        # 波动率倒数加权
                        basic_w = risk_parity.allocate(rets_df)
                        # 等风险贡献（失败时降级为波动率倒数加权，但要让用户知道）
                        try:
                            erc_w = risk_parity.equal_risk_contribute(rets_df)
                        except Exception as e:
                            erc_w = basic_w
                            logger.warning(f"等风险贡献计算失败，已降级: {e}")
                            st.caption("ℹ️ 等风险贡献计算失败，已降级为波动率倒数加权")

                        wcol1, wcol2 = st.columns(2)
                        with wcol1:
                            st.markdown('<p style="font-weight:600;">📊 波动率倒数加权</p>', unsafe_allow_html=True)
                            fig_w1 = go.Figure(data=[go.Pie(
                                labels=[f"{c} ({basic_w[c]*100:.1f}%)" for c in basic_w.index],
                                values=basic_w.values, hole=0.4,
                                textinfo="label+percent"
                            )])
                            fig_w1.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig_w1, use_container_width=True)

                        with wcol2:
                            st.markdown('<p style="font-weight:600;">⚖️ 等风险贡献</p>', unsafe_allow_html=True)
                            fig_w2 = go.Figure(data=[go.Pie(
                                labels=[f"{c} ({erc_w[c]*100:.1f}%)" for c in erc_w.index],
                                values=erc_w.values, hole=0.4,
                                textinfo="label+percent"
                            )])
                            fig_w2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig_w2, use_container_width=True)

                        # 收益/风险统计
                        port_basic = (rets_df * basic_w).sum(axis=1)
                        port_erc = (rets_df * erc_w).sum(axis=1)

                        sr1, sr2, sr3, sr4, sr5 = st.columns(5)
                        sr1.metric("组合年化收益(波动率倒数)", f"{port_basic.mean()*252*100:.2f}%")
                        sr2.metric("组合年化波动", f"{port_basic.std()*(252**0.5)*100:.2f}%")
                        sr3.metric("夏普比率", f"{(port_basic.mean()*252-0.03)/(port_basic.std()*(252**0.5)):.2f}")
                        sr4.metric("组合年化收益(ERC)", f"{port_erc.mean()*252*100:.2f}%")
                        sr5.metric("组合年化波动(ERC)", f"{port_erc.std()*(252**0.5)*100:.2f}%")


    # ========== Tab 4: 策略信誉引擎 ==========
    with quant_tab4:
        render_credibility_panel()

    with quant_tab5:
        render_portfolio_risk_panel()


# ========== ⚙️ 设置页 ==========
# ========== 🛡️ 交易安全 / 模型 / 调度 控制面板 ==========
def render_safety_panel():
    """设置页内的「交易安全中枢 + 模型注册中心 + 自动化调度」控制面板。

    注意：本面板只做状态展示与显式人工操作（武装/熔断/调度启停），
    绝不自行下单；下单仍由交易安全中枢 guard 在扫描主循环里裁决。
    """
    # 经 src/ui/paths 解析，避免「看板进程早于源码改动启动」时的 ImportError。
    # 旧进程只重跑 app.py、不重载已导入模块，直连 config.settings 会撞上缓存。
    from src.ui.paths import CONFIG_DIR, STALE_PROCESS
    if STALE_PROCESS:
        st.warning(
            "⚠️ 当前看板进程运行的是**旧版代码**（内存里的 `config.settings` 已过时）。\n\n"
            "本页已按当前磁盘代码自行兼容，但其它新功能仍需重启才能生效 —— "
            "**请关闭看板并重新启动**。Streamlit 只重跑脚本、不会重载已导入的模块，"
            "所以改完源码后必须重启进程。"
        )
    try:
        from src.trading.guard import SafetyController, GuardConfig, TradingMode
        from src.models.model_registry import scan_models, best_loadable, best_overall
        from src.scheduler.auto_scheduler import (
            is_running, start_scheduler, stop_scheduler, run_all_once,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"安全面板依赖加载失败: {e}")
        return

    # ---------- 交易安全中枢 ----------
    st.markdown('<div class="section-title">🛡️ 交易安全中枢 (Safety Controller)</div>', unsafe_allow_html=True)
    st.info("默认 PAPER（纸交易）。进入实盘需两步令牌武装；KILL SWITCH 一键熔断并持久化，重启仍保持。")
    try:
        _gpath = Path(CONFIG_DIR) / "trading_guard.yaml"
        cfg = GuardConfig.from_yaml(str(_gpath)) if _gpath.exists() else GuardConfig()
    except Exception:
        cfg = GuardConfig()
    # 把当前会话里已连接券商带进来：武装实盘前要能确认它是真实盘通道。
    # 没连接时传 None → SafetyController 按模拟处理并拒绝武装（fail-closed）。
    safety = SafetyController(cfg, broker=st.session_state.get("trade_broker"))

    mode = safety.mode
    halted = safety.halted
    mode_color = {
        "off": "#64748b", "paper": "#16a34a",
        "live_pending": "#f59e0b", "live": "#dc2626",
    }.get(mode.value, "#64748b")
    st.markdown(
        f"当前模式：<span style='background:{mode_color};color:#fff;padding:2px 10px;"
        f"border-radius:6px;'>{mode.value.upper()}</span> ｜ "
        f"KILL SWITCH：{'🛑 已熔断' if halted else '🟢 正常'}",
        unsafe_allow_html=True,
    )
    snap = safety.snapshot()
    if snap.get("armed_by"):
        st.caption(f"武装人: {snap['armed_by']} @ {snap.get('armed_at', '')}")
        if snap.get("halt_reason"):
            st.caption(f"熔断原因: {snap['halt_reason']}")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🟢 解除实盘武装→PAPER", key="sec_disarm", use_container_width=True):
            safety.disarm("dashboard")
            st.success("已解除实盘武装，回到纸交易"); st.rerun()
    with c2:
        if st.button("🔴 武装实盘(LIVE_PENDING)", key="sec_arm", use_container_width=True):
            # arm_live 现在会拒绝「券商还是模拟连接」的情况（fail-closed），
            # 必须把拒绝理由原样显示出来，否则按钮点了没反应又不知道为什么。
            try:
                tok = safety.arm_live("dashboard")
            except ValueError as e:
                st.error(f"⛔ 无法武装实盘：{e}")
            else:
                st.session_state["_arm_token"] = tok
                st.rerun()
    with c3:
        if st.button("🛑 KILL SWITCH", key="sec_kill", use_container_width=True):
            safety.halt("手动触发（看板）", "dashboard")
            st.error("🛑 已触发全局熔断，所有下单冻结"); st.rerun()

    # 武装确认（两步令牌）
    if st.session_state.get("_arm_token"):
        st.warning(f"⚠️ 请核对确认令牌（授权人回填完成武装）：`{st.session_state['_arm_token']}`")
        tok_in = st.text_input("输入确认令牌", key="sec_arm_input")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("✅ 确认武装", key="sec_confirm"):
                if safety.confirm_arm(tok_in.strip(), "dashboard"):
                    st.session_state["_arm_token"] = ""
                    st.success("🔴 已武装 LIVE_PENDING，每笔委托需人工确认"); st.rerun()
                else:
                    st.error(
                        "确认失败：令牌不匹配，或券商当前不是实盘连接"
                        "（可运行 `python scripts/check_live_ready.py` 自检）。"
                    )
        with b2:
            if st.button("❌ 取消", key="sec_cancel"):
                st.session_state["_arm_token"] = ""
                st.rerun()

    if halted:
        if st.button("🟢 解除熔断 (resume)", key="sec_resume", use_container_width=True):
            safety.resume("dashboard")
            st.success("已解除熔断（若曾实盘则强制回 PAPER）"); st.rerun()

    # ---------- 模型注册中心 ----------
    st.markdown('<div class="section-title">📦 模型注册中心 (Model Registry)</div>', unsafe_allow_html=True)
    try:
        infos = scan_models()
        rows = []
        for m in infos:
            ric = f"{m.rank_ic:+.4f}" if m.rank_ic == m.rank_ic else "n/a"
            rows.append({
                "模型": m.display_name,
                "RankIC": ric,
                "可部署": "✅" if m.loadable else "⚠️",
                "胜率": f"{m.win_rate:.1%}" if m.win_rate == m.win_rate else "-",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        bl = best_loadable(); bo = best_overall()
        note = ""
        if bo and not bo.loadable:
            note = (f"⚠️ 全局最优 **{bo.display_name}**（RankIC {bo.rank_ic:+.4f}）"
                    "仅是 walk-forward 实验日志，未生成可部署 .pkl；"
                    "需运行 alpha 训练脚本（alpha_set=industry/lhb/margin/northbound）后才会被自动选用。")
        st.caption(
            f"当前可部署最优：{(bl.display_name if bl else '-')} "
            f"(RankIC {bl.rank_ic:+.4f})。" + note
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"模型扫描失败: {e}")

    # ---------- 超级集成 ----------
    st.markdown('<div class="section-title">🧩 超级集成 (XGBoost + Alpha + LSTM)</div>', unsafe_allow_html=True)
    try:
        from src.models.predictor import predictor as _pred
        st.code(_pred.ensemble_report(), language="text")
        st.caption(
            "由 `config.ml.use_ensemble` 控制（默认取值见该配置项）。"
            "任一腿文件缺失或推理失败会自动剔除并重新归一化权重，不影响主模型。")
    except Exception as e:  # noqa: BLE001
        st.caption(f"集成状态不可用: {e}")

    # ---------- 严谨回测 ----------
    st.markdown('<div class="section-title">📊 严谨回测 (成本 + 多宇宙 + Walk-Forward)</div>', unsafe_allow_html=True)
    st.caption("对给定宇宙与 AI 打分做 TopN 等权组合回测：佣金/印花税/滑点建模，"
               "输出 RankIC / 夏普 / 索提诺 / 最大回撤。")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        bt_pool = st.selectbox("回测宇宙", ["zz500", "hs300", "zz1000", "all"], key="bt_pool")
    with bc2:
        bt_topn = st.number_input("TopN", 5, 50, 20, 5, key="bt_topn")
    with bc3:
        bt_rb = st.number_input("调仓天数", 1, 20, 5, 1, key="bt_rb")
    if st.button("📈 运行严谨回测", key="bt_run", use_container_width=True):
        with st.spinner(f"回测 {bt_pool} 池（TopN={bt_topn}, 调仓{bt_rb}日）…"):
            try:
                from src.core.backtest import WalkForwardBacktester, BacktestConfig
                from src.data.fetcher import fetcher as _fet
                from src.models.predictor import predictor as _pred
                codes = _fet.get_stock_pool(bt_pool)[:80]     # 控制回测量
                raw = _fet.get_batch_daily_kline(codes, max_workers=8)
                stock_data = {c: df for c, df in raw.items() if df is not None and len(df) >= 120}
                if not stock_data:
                    st.warning("无可用行情数据")
                else:
                    bt = WalkForwardBacktester(BacktestConfig(top_n=int(bt_topn),
                                                              rebalance_days=int(bt_rb)))
                    res = bt.backtest_predictor_rolling(_pred, stock_data,
                                                        rebalance_days=int(bt_rb))
                    st.code(res.report(), language="text")
                    if res.equity is not None and len(res.equity):
                        st.line_chart(res.equity)
            except Exception as e:  # noqa: BLE001
                st.error(f"回测失败: {e}")

    # ---------- 自动化调度 ----------
    st.markdown('<div class="section-title">⏰ 收盘后自动化调度 (APScheduler)</div>', unsafe_allow_html=True)
    sc = app_config.scheduler
    st.caption(
        f"数据更新 {sc.data_update_cron} ｜ 每日推荐 {sc.daily_pick_cron} ｜ "
        f"进化验证 {sc.evolution_cron}（{sc.timezone}）"
    )
    st.info("调度仅做数据/推荐/进化验证，绝不下单。勾选启用后点击「启动调度」即可拉起后台定时器；"
            "开关会持久化，重启看板后仍记得上次状态。")

    # 启用开关（持久化到 config/scheduler_state.json，跨重启保留）
    _enabled = st.checkbox(
        "启用收盘后自动化调度",
        value=app_config.scheduler.enabled,
        key="sch_enabled",
        help="勾选后，点击「启动调度」才会真正拉起 APScheduler 后台定时器。",
    )
    if _enabled != app_config.scheduler.enabled:
        app_config.scheduler.enabled = _enabled
        save_scheduler_state(_enabled)
        # 勾选即启、取消即停：让开关成为调度的唯一真相来源，避免「已勾选却未运行」
        if _enabled:
            if not is_running():
                start_scheduler()
        else:
            if is_running():
                stop_scheduler()
        st.rerun()

    # 多策略复核开关（持久化，与调度 enabled 同文件但互不覆盖）
    _ens = st.checkbox(
        "选股时启用多策略复核（量化三策略投票）",
        value=bool(getattr(sc, "pick_ensemble", False)),
        key="sch_pick_ensemble",
        help="开启后，每日推荐会先由 AI 打分，再交给量化三策略（多因子／均值回归／"
             "AI+量价融合）投票复核：集体看空则剔除，一致看多则按一致度加权提分（上限 +10%）。"
             "缺数据、无共识、全票否决三种情况一律保留原候选，绝不会把推荐清空。",
    )
    if _ens != bool(getattr(sc, "pick_ensemble", False)):
        save_pick_ensemble_state(_ens)
        st.rerun()

    if not app_config.scheduler.enabled:
        st.warning("⚠️ 调度总开关未启用：勾选上方「启用收盘后自动化调度」后再点「启动调度」。")

    s1, s2, s3 = st.columns(3)
    with s1:
        if st.button("▶️ 启动调度", key="sch_start", use_container_width=True):
            # 兜底启用：即使用户没勾选，点启动也应真的跑起来，而不是静默失败
            if not app_config.scheduler.enabled:
                app_config.scheduler.enabled = True
                save_scheduler_state(True)
            r = start_scheduler()
            if r:
                st.success("✅ 已启动（工作日 18:30 更新数据 / 18:45 生成推荐 / "
                           "19:00 进化验证 / 19:15 RankIC 回填）")
            else:
                st.error("❌ 启动失败：APScheduler 未安装，或 cron 表达式无效。查看日志了解详情。")
                st.info("安装依赖：pip install apscheduler")
            st.rerun()
    with s2:
        if st.button("⏹️ 停止调度", key="sch_stop", use_container_width=True):
            stop_scheduler(); st.success("🛑 已停止"); st.rerun()
    with s3:
        if st.button("🚀 立即跑一轮", key="sch_run", use_container_width=True):
            # 试跑模式：不写报告、不含 RankIC 回填（扫全池拉行情要几分钟，会卡死主线程）
            out = run_all_once(dry_run=True)
            st.json({k: (v.get("status") if isinstance(v, dict) else v)
                     for k, v in out.items() if k != "ran_at"})
            st.caption("试跑不写推荐报告，RankIC 回填按 19:15 的定时作业跑；"
                       "需立刻补点请用 scripts/rolling_ic_backfill.py。")
    st.caption(f"调度状态：{'🟢 运行中' if is_running() else '⚪ 未运行'}")

    # 常驻调度进程存活证据 —— 作业挂了的表现只是「今天没有新报告」，
    # 极易被当成"今天没行情"，所以必须显式把年龄摆出来。
    hb = _scheduler_daemon_heartbeat()
    if hb is None:
        st.warning(
            "⚠️ 未检测到常驻调度进程（无 data/scheduler_heartbeat.json）。\n\n"
            "三个收盘后作业（18:30 数据更新 / 18:45 每日推荐 / 19:00 进化验证）"
            "目前没有自己的常驻载体：看板一关，第二天就没有推荐、交易计划为空、"
            "闭环无事可做。\n\n"
            "治理：开机启动项 `StockGodAI-Scheduler.vbs`（启动文件夹）＋ "
            "`scripts\\tasks\\install_tasks.bat`（10 分钟看门狗）。"
        )
    else:
        age = hb.get("age_sec")
        nxt = {j["id"]: j.get("next_run") for j in (hb.get("jobs") or [])}
        next_pick = nxt.get('daily_pick') or '—'
        if age is None:
            # 时间戳解析不出来时不要谎报"已停 X 秒"，只说读不出来
            st.error("🔴 常驻调度进程心跳时间戳无法解析（下次选股未知）")
        elif age > 180:
            st.error(f"🔴 常驻调度进程心跳已停 {age:.0f} 秒（下次选股 {next_pick}）")
        else:
            st.success(f"🟢 常驻调度进程存活（pid {hb.get('pid')}，心跳 {age:.0f}s 前）"
                       f"｜下次选股 {next_pick}")
        if not hb.get("holding"):
            st.info("ℹ️ 常驻进程尚未接管 APScheduler（当前由看板持有 "
                    "data/.scheduler.lock，或调度未启用）—— 接管最多需 5 分钟。")



# ========== 📈 模型稳定性页 ==========
if page == "📈 模型稳定性":
    render_hero(
        "模型稳定性 · RankIC 滚动监控",
        "逐日横截面 RankIC 稳定性曲线 · 由收盘后自动回填累积",
        icon="📈",
        badges=[("violet", "IC / IR 双轨")],
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("数据源：data/rolling_ic_log.json（收盘后每日自动补点，按评分日去重）")
    with col2:
        window = st.slider("滚动窗口", 5, 60, 20, 1, key="ric_window")

    try:
        from src.core.rolling_ic import render_curve
        svg, stats = render_curve(window=window)
    except Exception as e:  # noqa: BLE001
        svg, stats = None, None
        st.error(f"渲染曲线失败: {e}")

    if svg:
        st.markdown(svg, unsafe_allow_html=True)

    if stats and stats.get("n_points"):
        n = stats["n_points"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("样本点数", n)
        c2.metric("RankIC 均值", f"{stats['mean_ic']:+.4f}")
        c3.metric("日频 IR (mean/std)", f"{stats['ir_mean_std']}")
        c4.metric("t 统计量 (mean/SE)", f"{stats['t_stat']}")
        c5.metric("正 IC 占比", f"{stats['pos_rate']*100:.0f}%")
        st.caption(f"末点日期：{stats.get('last_date')} ｜ 口径说明：production_metrics 里那个 "
                   f"『ic_ir≈14』是 mean/SE（t 统计量，随样本量 √n 放大），并非日频真实 IR。"
                   f"此处日频 IR≈{stats['ir_mean_std']} 才是单日排序稳定性的真实度量。")
    else:
        st.info("暂无 RankIC 数据。点击右侧「立即回填」生成历史序列，或等待收盘后自动任务累积。")

    # 手动回填入口
    with st.expander("🔧 手动回填 RankIC 历史", expanded=False):
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            bf_days = st.number_input("回溯交易日数", 5, 120, 30, 5, key="ric_days")
        with rc2:
            bf_pool = st.selectbox("股票池", ["zz500", "hs300", "zz1000", "all"], key="ric_pool")
        with rc3:
            bf_sample = st.number_input("宇宙截断只数", 30, 500, 150, 10, key="ric_sample")
        if st.button("🚀 立即回填", type="primary", key="ric_run"):
            with st.spinner(f"回填中（{bf_days} 交易日 / {bf_pool} / 样本 {bf_sample}）…后台可能需数分钟"):
                try:
                    from src.core.rolling_ic import run_backfill
                    res = run_backfill(days=int(bf_days), pool=bf_pool, sample=int(bf_sample))
                    if res.get("status") == "error":
                        st.error(f"失败: {res.get('error')}")
                    else:
                        st.success(f"新增 {res.get('added')} 点，日志共 {res.get('total')} 点；"
                                   f"均值 {res.get('mean_ic'):+.4f}，t 统计量 {res.get('t_stat')}")
                        st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"回填异常: {e}")


if page == "📊 业绩归因":
    render_hero(
        "业绩归因 · 净值与风险档案",
        "把分散的每日结算快照汇成一条可判断的净值曲线：赚不赚钱、钱从哪来、风险有没有变大",
        icon="📊",
        badges=[("violet", "净值 / 回撤 / 归因")],
    )

    from src.trading import performance as perf_mod

    rep = perf_mod.build_report()
    per = rep.get("period") or {}
    eqv = rep.get("equity") or {}
    rk = rep.get("risk") or {}
    tr = rep.get("trades") or {}
    n_days = int(per.get("n_days") or 0)

    if n_days == 0:
        st.info("还没有可用的结算快照（data/loop_reports/）。"
                "收盘后闭环会自动生成；也可先跑一次 `python -m src.trading.performance`。")
    else:
        # ---------- KPI ----------
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("期末权益", f"¥{eqv.get('end', 0):,.0f}")
        _tr_pct = eqv.get("total_return_pct")
        c2.metric("累计收益", "—" if _tr_pct is None else f"{_tr_pct:+.2f}%",
                  delta=None if _tr_pct is None else f"{_tr_pct:+.2f}%")
        _ann = eqv.get("annualized_return_pct")
        c3.metric("年化收益", "—" if _ann is None else f"{_ann:+.1f}%")
        c4.metric("夏普", "—" if rk.get("sharpe") is None else f"{rk['sharpe']:.2f}")
        c5.metric("最大回撤", f"{rk.get('max_drawdown_pct', 0):.1f}%",
                  delta=f"{rk.get('max_dd_days', 0)} 日",
                  delta_color="off")
        _wr = tr.get("win_rate")
        c6.metric("平仓胜率", "—" if _wr is None else f"{_wr*100:.0f}%",
                  delta=f"{tr.get('n_closes', 0)} 笔", delta_color="off")

        # 样本太小时口径必须自证，否则用户会把噪声当业绩
        if n_days < 20:
            st.warning(
                f"⚠️ 当前只有 **{n_days}** 个交易日样本：夏普/索提诺/年化一律不显示"
                f"（<20 日无法免疫噪声）。最大回撤与胜率可以看，但不要据此判断策略优劣。"
            )

        # ---------- 净值 + 回撤 ----------
        curve = rep.get("curve") or []
        if curve:
            _df = pd.DataFrame(curve)
            _df["date"] = pd.to_datetime(_df["date"])
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                row_heights=[0.68, 0.32],
                subplot_titles=("净值曲线", "回撤（相对历史峰值）"),
            )
            fig.add_trace(go.Scatter(
                x=_df["date"], y=_df["equity"], mode="lines+markers",
                name="权益", line=dict(color="#5b21b6", width=2.2),
                fill="tozeroy", fillcolor="rgba(91,33,182,0.07)",
                marker=dict(size=4),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=_df["date"], y=_df["dd_pct"], mode="lines", name="回撤%",
                line=dict(color="#dc2626", width=1.6),
                fill="tozeroy", fillcolor="rgba(220,38,38,0.12)",
            ), row=2, col=1)
            fig.update_layout(
                height=430, margin=dict(l=10, r=10, t=38, b=10),
                showlegend=False, hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=11),
            )
            fig.update_yaxes(tickformat=",.0f", row=1, col=1)
            fig.update_yaxes(ticksuffix="%", row=2, col=1)
            # 净值轴不能从 0 起：¥470 万的曲线在 0~500 万坐标里就是一条贴顶的直线，
            # 什么也看不出来。按实际波动区间给 25% 余量，形状才看得出来。
            _lo = float(_df["equity"].min())
            _hi = float(_df["equity"].max())
            _pad = max((_hi - _lo) * 0.25, _hi * 0.004)
            fig.update_yaxes(range=[_lo - _pad, _hi + _pad], row=1, col=1)
            st.plotly_chart(fig, use_container_width=True, key="perf_curve")

        st.caption(
            f"区间 {per.get('start')} ~ {per.get('end')}（{n_days} 个交易日）"
            f" ｜ 最大回撤 {rk.get('max_dd_peak_date') or '—'} → "
            f"{rk.get('max_dd_trough_date') or '—'}"
            f"（{'已恢复' if rk.get('max_dd_recovered') else '未恢复'}）"
            f" ｜ 上涨 {rk.get('up_days', 0)} 天 / 下跌 {rk.get('down_days', 0)} 天"
        )

        # ---------- 月度收益 ----------
        monthly = rep.get("monthly") or {}
        if monthly:
            mdf = pd.DataFrame(
                [{"月份": k, "收益率": v} for k, v in sorted(monthly.items())])
            # A 股口径：正收益红、负收益绿
            colors = ["#dc2626" if v >= 0 else "#059669" for v in mdf["收益率"]]
            mfig = go.Figure(go.Bar(
                x=mdf["月份"], y=mdf["收益率"], marker_color=colors,
                text=[f"{v:+.2f}%" for v in mdf["收益率"]],
                textposition="outside",
            ))
            mfig.update_layout(
                height=250, margin=dict(l=10, r=10, t=22, b=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(ticksuffix="%"), showlegend=False,
            )
            # 「2026-10」会被 Plotly 当日期解析成 23:59:59.999…，轴标签变成时间。
            # 月份是**离散类目**，必须显式 category。
            mfig.update_xaxes(type="category")
            st.markdown("##### 月度收益")
            st.plotly_chart(mfig, use_container_width=True, key="perf_monthly")

        # ---------- 归因 ----------
        attr = rep.get("attribution") or {}
        st.markdown("##### 归因：钱是从哪赚/亏的")
        a1, a2 = st.columns(2)
        with a1:
            st.caption("按行业（缺失映射归入「未知行业」，不会被静默丢掉）")
            bs = attr.get("by_sector") or {}
            if bs:
                df_s = pd.DataFrame(
                    [{"行业": k, "已实现盈亏": v["pnl"], "笔数": v["count"],
                      "胜率%": round((v.get("win_rate") or 0) * 100, 1)}
                     for k, v in sorted(bs.items(), key=lambda kv: kv[1]["pnl"], reverse=True)]
                )
                # 不设固定高度：行业数少时全展开。最大亏损源（如「未知行业」）
                # 若被折进滚动区，等于把最该看的一行藏起来。
                st.dataframe(df_s, use_container_width=True, hide_index=True)
            else:
                st.caption("暂无已平仓交易")
        with a2:
            st.caption("按策略标签")
            bst = attr.get("by_strategy") or {}
            if bst:
                df_g = pd.DataFrame(
                    [{"策略": k, "已实现盈亏": v["pnl"], "笔数": v["count"],
                      "胜率%": round((v.get("win_rate") or 0) * 100, 1)}
                     for k, v in sorted(bst.items(), key=lambda kv: kv[1]["pnl"], reverse=True)]
                )
                st.dataframe(df_g, use_container_width=True, hide_index=True)
            else:
                st.caption("暂无已平仓交易")

        # 退出原因分布：固定桶（含 0 值），让「止盈从未触发」这类事实直接可见。
        # 实测曾因所有风控离场都记成同一个 stop 桶，+15% 止盈线从没走到、
        # 亏损一路跑到 -5% 止损的事实被完全掩盖，归因表上看不出任何异常。
        erd = rep.get("exit_reason_distribution") or []
        if erd:
            st.markdown("##### 退出原因分布（固定桶，含未触发的）")
            df_e = pd.DataFrame([
                {"退出原因": e["label"] + (" ⚠️未触发" if e["count"] == 0 else ""),
                 "已实现盈亏": e["pnl"], "笔数": e["count"],
                 "胜率%": 0 if e["count"] == 0 else round(e["win_rate"] * 100, 1)}
                for e in erd
            ])
            st.dataframe(df_e, use_container_width=True, hide_index=True)
            if any(e["count"] == 0 for e in erd):
                st.warning(
                    "⚠️ 存在「0 笔」的退出桶：说明该退出类型从未触发（如止盈线设得太高、"
                    "或亏损一路跑到止损）。这是盈亏比塌陷的典型信号，优先查止损/止盈参数"
                    "与持有纪律，而不是加仓。"
                )

        bc = attr.get("by_code") or {}
        if bc:
            with st.expander("按个股（前 15）", expanded=False):
                df_c = pd.DataFrame(
                    [{"标的": k, "已实现盈亏": v["pnl"], "笔数": v["count"],
                      "胜率%": round((v.get("win_rate") or 0) * 100, 1),
                      "费用": v["fees"]}
                     for k, v in sorted(bc.items(), key=lambda kv: kv[1]["pnl"], reverse=True)[:15]]
                )
                st.dataframe(df_c, use_container_width=True, hide_index=True)

        # ---------- 交易与成本 ----------
        st.markdown("##### 交易质量与成本")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("盈亏比", "—" if tr.get("payoff_ratio") is None else f"{tr['payoff_ratio']:.2f}")
        t2.metric("期望值", "—" if tr.get("expectancy") is None else f"¥{tr['expectancy']:,.0f}/笔")
        t3.metric("最长连亏", f"{tr.get('max_consecutive_losses', 0)} 笔")
        t4.metric("累计手续费", f"¥{tr.get('fees', 0):,.0f}")
        _fr = tr.get("fee_ratio_pct")
        t5.metric("费用率", "—" if _fr is None else f"{_fr:.3f}%")
        if (tr.get("payoff_ratio") or 9) < 1:
            st.error(
                "🔴 **盈亏比 < 1**：平均每笔亏损大于平均每笔盈利。这不是运气问题，"
                "是止盈/止损不对称——胜率再高也会被少数大亏吃掉。"
                "先查止损是否被放宽过（ATR 自适应），再看是否在亏损仓位上反复补仓。"
            )

        with st.expander("🔧 数据口径与落盘", expanded=False):
            st.markdown(
                f"""
- 数据源：`data/loop_reports/loop_report_*.json`（{n_days} 份）+ `data/paper_trade_history.json`
- **已剔除** `equity_ok=false` 的占位快照：账户取不到时权益是占位值，一条就能把回撤算成 -100%
- 日收益分母用**当日起始权益**（不是前一日收盘），隔夜跳空不会被算进当天
- 胜率/盈亏比只统计**平仓腿**（`side=sell` 且 `pnl≠0`）；买入腿混进来会把胜率稀释成 0
- 夏普/索提诺/年化要求 ≥20 个交易日；零波动时返回「—」而不是 inf
- 生成时间：{rep.get('generated_at')}
                """
            )
            if st.button("💾 落盘为 data/performance.json", key="perf_save"):
                try:
                    p = perf_mod.save_report(rep)
                    st.success(f"已写入 {p}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"落盘失败: {e}")

            if rep.get("rolling_sharpe"):
                rs = pd.DataFrame(rep["rolling_sharpe"]).dropna()
                if len(rs) >= 2:
                    st.caption("滚动 20 日夏普（观察优势是否在衰减）")
                    st.line_chart(rs.set_index("date")["sharpe"], height=180)


if page == "⚙️ 设置":
    render_hero(
        "系统设置",
        "模型开关 · 风控参数 · 调度与数据源配置（改动即时落盘）",
        icon="⚙️",
        badges=[("gray", f"调度 {'启用' if _sched_on else '停用'}")],
    )

    st.markdown('<div class="section-title">🤖 AI 大模型配置</div>', unsafe_allow_html=True)
    st.info("💡 用于生成个股分析报告和选股组合推荐理由。可接云端 DeepSeek，也可接本机推理服务。")

    # provider 选择 —— 云端需要 key 且有余额；本地不需要 key、不联网、不花钱。
    # 此前 llm.py 里 base_url 是个死开关（chat() 用的是硬编码常量），
    # 所以这里以前填什么都不会生效。现在配置真能改到请求地址。
    _is_local_now = bool(getattr(deepseek, "is_local", False))
    ds_provider = st.radio(
        "模型来源",
        ["☁️ 云端 DeepSeek", "🖥️ 本地服务（Ollama / LM Studio / vLLM）"],
        index=1 if _is_local_now else 0,
        horizontal=True,
        key="ds_provider_radio",
        help="本地服务走 OpenAI 兼容接口，无需 API Key，也不产生费用。",
    )
    _want_local = str(ds_provider).startswith("🖥️")

    if _want_local:
        st.caption(
            "💡 本地模式：不联网、不花钱、无需 Key。需先启动服务并拉好模型，"
            "例如 Ollama → `ollama serve` + `ollama pull qwen2.5:7b`。"
            "RTX 4050 Laptop（6GB）建议 7B Q4 或 3B，7B 未量化会爆显存。")
        # 切到本地时，若 base_url 还停在云端地址（如 api.deepseek.com），
        # 预填本机地址而不是把云端地址当默认值显示给用户——否则极易被误导
        # 以为「那个值就是本地的」而忘记改，结果请求仍打到云端拿 402。
        _base_default = (
            "http://localhost:11434/v1"
            if not deepseek._is_loopback(deepseek.config.base_url)
            else deepseek.config.base_url or "http://localhost:11434/v1"
        )
        _r1c1, _r1c2 = st.columns([3, 2])
        with _r1c1:
            ds_base = st.text_input(
                "服务地址（OpenAI 兼容）",
                value=_base_default,
                key="ds_base_url_input",
                placeholder="http://localhost:11434/v1",
                help="必须填你电脑上的服务地址，不是 api.deepseek.com！"
                     "Ollama 默认 11434，LM Studio 默认 1234，vLLM 默认 8000。",
            )
        with _r1c2:
            # 探测到就给下拉，没探测到退回手填 —— 本地模型名又长又带 tag
            _found = st.session_state.get("ds_local_models") or []
            if _found:
                _mi = _found.index(deepseek.config.model) \
                    if deepseek.config.model in _found else 0
                ds_model = st.selectbox("模型", _found, index=_mi,
                                       key="ds_model_sel")
            else:
                ds_model = st.text_input(
                    "模型名", value=deepseek.config.model or "qwen2.5:7b",
                    key="ds_model_txt", placeholder="qwen2.5:7b")
        _r2c1, _r2c2, _r2c3 = st.columns([1, 1, 4])
        with _r2c1:
            ds_probe_btn = st.button("🔍 探测", use_container_width=True,
                                     key="ds_probe_btn")
        with _r2c2:
            ds_save_btn = st.button("💾 保存", use_container_width=True,
                                    key="ds_save_btn")

        if ds_probe_btn:
            with st.spinner("探测中…"):
                names = deepseek.list_models(base_url=ds_base)
            st.session_state["ds_local_models"] = names
            if names:
                st.success(f"✅ 发现 {len(names)} 个模型：" + "、".join(names[:6])
                           + ("…" if len(names) > 6 else ""))
            else:
                st.warning("⚠️ 没取到模型列表 —— 地址填对了吗？服务启动了吗？")

        if ds_save_btn:
            if not deepseek._is_loopback(ds_base):
                st.error(
                    "⚠️ 本地模式的「服务地址」必须指向你本机"
                    "（localhost / 127.0.0.1），不能是 api.deepseek.com。"
                    "请改成 http://localhost:11434/v1，并确保 Ollama 已启动"
                    "（`ollama serve` + `ollama pull qwen2.5:7b`）。")
            else:
                deepseek.configure(provider="local", base_url=ds_base,
                                   model=ds_model)
                st.success(f"✅ 已保存：{deepseek.describe()}")
    else:
        ds_col1, ds_col2, ds_col3 = st.columns([3, 2, 1])
        with ds_col1:
            ds_key = st.text_input(
                "DeepSeek API Key",
                value="",
                type="password",
                key="ds_api_key_input",
                placeholder="sk-..." if not deepseek.config.api_key else "已配置（重新输入可覆盖）",
                help="留空则保持现有配置。申请地址：https://platform.deepseek.com",
            )
        with ds_col2:
            ds_model = st.selectbox(
                "模型",
                ["deepseek-chat", "deepseek-reasoner"],
                index=0 if deepseek.config.model != "deepseek-reasoner" else 1,
                key="ds_model_input",
            )
        with ds_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            ds_save_btn = st.button("💾 保存", use_container_width=True, key="ds_save_btn")

        if ds_save_btn:
            if ds_key.strip():
                deepseek.configure(provider="deepseek", api_key=ds_key,
                                   model=ds_model,
                                   base_url="https://api.deepseek.com")
                st.success("✅ DeepSeek 配置已保存")
            else:
                deepseek.configure(provider="deepseek", model=ds_model)
                st.info("✅ 模型已更新（API Key 未变更）")

    if deepseek.is_ready:
        st.success(f"🟢 {deepseek.describe()}")
        if st.button("🔌 测试连接", key="ds_test_btn"):
            with st.spinner("测试中…"):
                try:
                    msg = deepseek.test_connection()
                    st.success(f"✅ {msg}")
                except Exception as e:
                    st.error(f"❌ {e}")
    else:
        if _want_local:
            st.warning("🟡 请填写本地服务地址（如 http://localhost:11434/v1）")
        else:
            st.warning("🟡 尚未配置 DeepSeek API Key")

    st.markdown('<div class="section-title">🧠 模型配置</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:1rem;">
        <h4 style="color:#ef4444; margin:0 0 0.5rem 0;">XGBoost（主模型）</h4>
    </div>
    """, unsafe_allow_html=True)
    st.json({"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8})
    # 文案如实反映 config.ml.use_ensemble 的当前取值 —— 之前这里写死了
    # 「项目已统一为单模型」，而配置里 use_ensemble 默认为 True，
    # 于是看板一边宣称单模型、一边在后台把 alpha 与 LSTM 两条腿加权融合。
    # 一句写死的话，比不写更糟：它让人据此做出错误判断。
    if getattr(app_config.ml, "use_ensemble", False):
        st.caption(
            "ℹ️ 当前**启用超级集成**：以 XGBoost 为主模型（`models/xgboost_model.pkl`），"
            "并与 Alpha / LSTM 两条腿加权融合。任一腿缺失会自动降级，不影响主模型。")
    else:
        st.caption(
            "ℹ️ 当前**为单模型**：全部训练与预测只使用 XGBoost，"
            "模型文件 `models/xgboost_model.pkl`。")

    # 超级集成开关：之前只能改 config/settings.py 源码，看板上「看得见、摸不着」。
    # 注意这不只是显示口径——开启后 predict_batch 会把 Alpha/LSTM 两条腿
    # 加权融合进评分，**真实改变选股结果**，所以默认沿用你之前拍板的单模型(True/False)。
    _ens_on = st.checkbox(
        "🧩 启用超级集成（XGBoost + Alpha + LSTM 加权融合）",
        value=app_config.ml.use_ensemble,
        key="ml_use_ensemble",
        help="开启后预测 = XGBoost 主模型 + Alpha(权重1.0) + LSTM(权重0.8) 融合；"
             "任一腿缺失会自动降级，不影响主模型。关闭则为纯 XGBoost 单模型。",
    )
    if _ens_on != app_config.ml.use_ensemble:
        app_config.ml.use_ensemble = _ens_on
        save_ml_state(_ens_on)
        st.rerun()
    if _ens_on:
        st.caption("✅ 已启用：选股评分将融合 Alpha 与 LSTM 两条腿。")

    # ====== 🛡️ 交易安全 / 模型 / 调度 控制面板 ======
    render_safety_panel()

    st.markdown('<div class="section-title">📡 数据 & 缓存</div>', unsafe_allow_html=True)
    st.info("🔌 数据源: **AKShare** (新浪财经) · 免费沪深A股行情")

    if st.button("🗑️ 清除数据缓存", use_container_width=True):
        fetcher.clear_cache()
        st.success("✅ 缓存已清除! 下次获取将重新下载数据。")

    # ====== 🤖 AI 助手：筛选并发数（避免并行拉行情炸进程，同时支持真机提速）======
    st.markdown('<div class="section-title">🤖 AI 助手 · 筛选并发</div>', unsafe_allow_html=True)
    _cur_sw = int((getattr(app_config, "agent", None) and app_config.agent.screen_fetch_workers) or 1)
    _cur_sw = max(1, min(16, _cur_sw))
    _new_sw = st.number_input(
        "🔢 AI 助手「智能筛选 / 研报」拉行情的并发数",
        min_value=1, max_value=16, step=1, value=_cur_sw,
        key="agent_screen_workers",
        help="智能筛选与可下载研报在拉取全池行情时的并发线程数。默认 1（串行，最稳）。"
             "若你的环境并行拉取稳定（进程不崩），可调大以显著加速；"
             "部分环境并行会触发原生崩溃，请勿在该环境调高。改动即时对下次筛选生效。",
    )
    if int(_new_sw) != _cur_sw:
        app_config.agent.screen_fetch_workers = int(_new_sw)
        save_agent_state(int(_new_sw))
        st.success(f"✅ 已保存：筛选并发数 = {int(_new_sw)}（下次筛选即时生效）")
        st.rerun()

    st.markdown('<div class="section-title">🚀 一键训练优化</div>', unsafe_allow_html=True)
    st.info("💡 Optuna TPE（智能贝叶斯搜索）+ SHAP 自动筛选最有价值的特征 Top 60 + SMOTE 过采样。"
            "⚠️ 只有「标准/深度」档会覆盖生产模型（训练前自动备份）；「快速」档仅作试验，结果另存 *_optuna_*，绝不覆盖生产模型。")
    
    # 3 个按钮此前是 3 段结构完全相同的复制粘贴，只差 3 个字符串。
    # 改成数据驱动：加/改一个档位只需动这里一行。
    _TRAIN_PRESETS = {
        "optuna": ("TPE 智能引导 · SHAP 特征筛选 Top 60 · SMOTE 过采样", [
            ("⚡ Optuna 快速·试验 (2分钟)", "optuna_fast", True,
             "Optuna 50 trials · SHAP Top 60 · 3折CV · 3轮重训 · hs300/100只 · "
             "仅试验：不部署，结果另存 *_optuna_*，绝不覆盖生产模型"),
            ("🧠 Optuna 标准·部署 (15分钟)", "optuna_mid", False,
             "Optuna 150 trials · SHAP Top 60 · 5折CV · 3轮重训 · zz500 全池 · "
             "完成后覆盖生产模型（覆盖前自动备份 .bak_*）"),
            ("🔬 Optuna 深度·部署 (40分钟)", "optuna_deep", False,
             "Optuna 300 trials · SHAP Top 70 · 5折CV · 5轮重训 · zz500 全池 · "
             "完成后覆盖生产模型（覆盖前自动备份 .bak_*）"),
        ]),
    }

    def _render_train_presets(method: str) -> None:
        caption, presets = _TRAIN_PRESETS[method]
        st.caption(caption)
        cols = st.columns(len(presets))
        for col, (label, mode, primary, help_text) in zip(cols, presets):
            with col:
                if st.button(label, type="primary" if primary else "secondary",
                             use_container_width=True, key=f"train_{mode}_btn",
                             help=help_text):
                    st.session_state.train_queued = True
                    st.session_state.train_mode = mode
                    st.session_state.train_method = method
                    # BUG-1 复位：train_reloaded 全项目没有置回 False 的路径，
                    # 导致同一个会话里第 1 次训练能热加载、第 2 次及以后永远不能，
                    # 还会落进「正在训练」分支 → 4 秒轮询死循环。
                    # 启动新训练时把所有终态标志一起复位。
                    st.session_state.train_reloaded = False
                    st.session_state.train_done = False
                    _TRAIN_PROC.pop("proc", None)
                    st.session_state.train_log_wait = 0
                    st.session_state.pop("train_result", None)

    _render_train_presets("optuna")
    
    if st.session_state.get("train_queued") and not st.session_state.get("train_running"):
        st.session_state.train_running = True
        import subprocess, os
        py = sys.executable
        base = Path(__file__).parent.parent.parent
        log_file = base / "logs" / f"train_ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        # BUG-2：train_log 必须在此处（Popen 之前）就落盘到 session_state。
        # 原先它在 Popen 之后才设置，而 train_running 在上面已经先置 True，
        # 一旦 cmd 拼装抛 NameError / Popen 失败，train_log 永远为空 →
        # 下面的轮询分支永久卡在「等待日志…」的 2 秒 rerun 循环里。
        st.session_state.train_log = str(log_file)
        mode = st.session_state.get("train_mode", "optuna_fast")
        method = st.session_state.get("train_method", "optuna")
        mode_label = mode

        # Optuna TPE + SHAP 训练
        script = base / "scripts" / "train_optuna.py"
        # 部署档（标准/深度）必须与生产 serve 池一致：本文件 :802 股票池默认 zz500，
        # 训练 Universe ≠ serve Universe 时实测 RankIC 不反映实盘（2026-10-12 三池实测结论）。
        # 快速档 = 纯试验：加 --no-deploy，绝不覆盖生产模型。
        # 事故记录（2026-10-12 23:36）：快速档曾以 hs300/100只/SHAP Top60 静默部署，
        # OOS RankIC 仅 0.046（生产 zz500 模型 0.0747），且未留下任何 metrics 记录。
        if mode == "optuna_deep":
            cmd = [py, str(script), "--n-trials", "300", "--cv-folds", "5", "--retrain-rounds", "5",
                   "--pool", "zz500", "--max-stocks", "0", "--parallel", "--model", "xgboost", "--shap-top", "70"]
            mode_label = "Optuna 深度"
        elif mode == "optuna_mid":
            cmd = [py, str(script), "--n-trials", "150", "--cv-folds", "5", "--retrain-rounds", "3",
                   "--pool", "zz500", "--max-stocks", "0", "--parallel", "--model", "xgboost"]
            mode_label = "Optuna 标准"
        else:  # optuna_fast
            cmd = [py, str(script), "--n-trials", "20", "--cv-folds", "3", "--retrain-rounds", "2",
                   "--pool", "hs300", "--max-stocks", "100", "--parallel", "--model", "xgboost",
                   "--max-samples", "30000", "--max-estimators", "400", "--shap-top", "60", "--no-deploy"]
            mode_label = "Optuna 快速"

        # 记住档位名称：终态提示在后续 rerun 里也要用，而那时启动块不会执行
        st.session_state.train_mode_label = mode_label
        
        # 打包后 sys.executable 就是本程序(=启动器)，必须带 --run-script 让它去
        # 执行脚本；否则启动器会把自己当主程序再跑一遍 → 额外弹出一个应用窗口
        if getattr(sys, "frozen", False):
            cmd = [cmd[0], "--run-script"] + list(cmd[1:])

        try:
            _proc = subprocess.Popen(
                cmd,
                cwd=str(base),
                env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
                stdout=open(log_file, "w", encoding="utf-8", errors="replace"),
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            # 存下子进程对象：轮询时可用 poll() 判断进程是否已退出，
            # 兜住「原生崩溃（无 Python traceback）」这类既不 finished 也不 failed 的情况
            _TRAIN_PROC["proc"] = _proc
            st.success(f"✅ {mode_label}训练已在后台启动! 日志: `{Path(log_file).name}`")
        except Exception as e:  # noqa: BLE001 - 启动失败必须退出轮询，不能卡死
            st.session_state.train_running = False
            st.session_state.train_queued = False
            st.session_state.train_done = True
            st.session_state.train_result = ("fail", mode_label, f"训练进程启动失败: {e}")
            st.error(f"❌ 训练进程启动失败: {e}")
    
    # train_done 是独立的终态标志：一旦进入（完成/失败/异常终止）就不再轮询，
    # 避免 train_reloaded 一个标志兼做「是否已重载」和「是否已停止轮询」两件事。
    if st.session_state.get("train_running") and not st.session_state.get("train_done"):
        train_log_file = st.session_state.get("train_log", "")
        log_path = Path(train_log_file) if train_log_file else None

        # ========= 1. 进程状态栏（始终可见）=========
        _psproc = None
        if os.name == "nt":
            import subprocess as _sp
            try:
                r = _sp.run(["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine", "/format:csv"],
                          capture_output=True, text=True, timeout=5)
                for line in r.stdout.splitlines():
                    if "train_optuna" in line:
                        parts = line.strip().split(",")
                        _psproc = {"pid": parts[-1] if len(parts) > 1 else "?", "cmd": "train_optuna"}
                        break
            except Exception as e:
                # wmic 在部分 Windows 11 上已被移除，失败属预期
                logger.debug(f"进程检测不可用: {e}")

        if _psproc:
            log_size = f"{log_path.stat().st_size//1024}KB" if log_path and log_path.exists() else "-"
            st.success(f"🟢 训练进程运行中 | PID: {_psproc['pid']} | {_psproc['cmd']} | 日志: {log_size}")
        elif log_path and log_path.exists():
            mtime = time.time() - log_path.stat().st_mtime
            if mtime < 120:
                st.info(f"🟡 进程已退出 | 日志最后更新 {mtime:.0f}s 前 — 检查下方结果")
            else:
                st.warning(f"🔴 日志已 {mtime:.0f}s 未更新 — 训练可能异常终止")
        else:
            st.warning("🔴 未检测到进程或日志文件")

        # ========= 2. 日志内容分析 =========
        if log_path and log_path.exists():
            try:
                log_lines = Path(train_log_file).read_text(encoding="utf-8", errors="replace").splitlines()
                full_text = "\n".join(log_lines)

                finished = "===TRAINING_COMPLETE===" in full_text
                crashed = "Traceback (most recent call last):" in full_text[-3000:]

                # 兜底判定：子进程已退出，但日志里既没有完成标记也没有失败标记，
                # 说明是异常终止。历史日志 logs/train_ui_20261006_234556.log 显示
                # akshare 依赖的 py_mini_racer.dll 硬崩时是原生栈回溯、没有 Python
                # traceback —— 既不 finished 也不 failed，会永久 4 秒轮询。
                aborted = False
                _proc = _TRAIN_PROC.get("proc")
                if _proc is not None:
                    try:
                        aborted = _proc.poll() is not None
                    except Exception as _pe:  # noqa: BLE001
                        logger.debug(f"训练子进程状态检测失败: {_pe}")
                        aborted = False

                failed = ("===TRAINING_FAILED===" in full_text
                          or crashed
                          or (aborted and not finished))

                # ===== 完成 =====
                if finished:
                    # 启动块本轮未执行时 mode_label 可能不存在，统一走 session_state
                    mode_label = st.session_state.get("train_mode_label", "训练")
                    if not st.session_state.get("train_reloaded"):
                        predictor.reload()
                        clear_predictor_cache()
                        st.session_state.train_reloaded = True
                    st.session_state.train_running = False
                    st.session_state.train_queued = False
                    st.session_state.train_done = True
                    st.session_state.train_result = ("ok", mode_label, "")
                    st.balloons()
                    st.success("🎉 训练完成！模型已热加载。")
                    final_f1 = [l for l in log_lines if "F1 =" in l or "F1(macro)" in l]
                    if final_f1:
                        st.code("\n".join(final_f1[-6:]), language=None)
                    # 终态：只刷新一次让 UI 更新，之后不再进入轮询
                    st.rerun()

                # ===== 失败 / 异常终止 =====
                elif failed:
                    mode_label = st.session_state.get("train_mode_label", "训练")
                    err_start = full_text.rfind("Traceback")
                    if err_start > 0:
                        err_detail = full_text[err_start:err_start+800]
                    else:
                        err_detail = ("训练异常终止（无 Python traceback，"
                                      "可能是原生崩溃，如 py_mini_racer.dll）")
                    st.session_state.train_running = False
                    st.session_state.train_queued = False
                    st.session_state.train_done = True
                    st.session_state.train_result = ("fail", mode_label, err_detail)
                    st.rerun()

                else:
                    # ===== 正在训练 =====
                    is_optuna = "Optuna" in full_text or "optuna" in full_text.lower()

                    # 阶段检测
                    phase = "⏳ 准备中…"
                    pct = 0
                    if "gather_data" in full_text[-3000:] or "\n📊 收集训练数据" in full_text:
                        phase = "📊 收集数据"
                        pct = 0.05
                    elif is_optuna and ("Optuna TPE" in full_text or "trial" in full_text[-3000:].lower()):
                        m = re.search(r'(\d+)/(\d+)\s*\[', full_text[-3000:])
                        if m:
                            done, total = int(m.group(1)), int(m.group(2))
                            phase = f"🔍 Optuna搜索 {done}/{total} trials"
                            pct = 0.1 + 0.5 * done / max(total, 1)
                        else:
                            phase = "🔍 Optuna超参搜索"
                            pct = 0.1
                    elif "超参数搜索" in full_text or "CV " in full_text[-3000:]:
                        cv_done = full_text.count("END ")
                        phase = f"🔍 超参搜索 ({cv_done} fold)" if cv_done else "🔍 超参搜索"
                        pct = 0.2
                    elif is_optuna and "SHAP" in full_text:
                        phase = "🧩 SHAP特征筛选"
                        pct = 0.6
                    elif "重训练" in full_text:
                        rd = full_text.count("Round ") // 2
                        phase = f"🔄 重训练 ({rd} 轮)" if rd else "🔄 重训练"
                        pct = 0.7 + min(rd / 10, 0.28)

                    st.progress(pct, text=phase)

                    # ===== 实时日志（可展开折叠面板）=====
                    with st.expander(f"📋 实时日志 — 最新 {len(log_lines)} 行", expanded=True):
                        tail = log_lines[-25:] if len(log_lines) > 25 else log_lines
                        st.code("\n".join(tail), language=None)

                    time.sleep(4)
                    st.rerun()

            except Exception:
                st.caption(f"⏳ 训练进行中… ({datetime.now().strftime('%H:%M:%S')})")
        else:
            # 日志尚未生成：限定轮询轮数（约 30 秒），超时判定为启动失败并进入终态，
            # 避免 Popen 失败时永久卡在「等待日志…」
            _wait = int(st.session_state.get("train_log_wait", 0)) + 1
            st.session_state.train_log_wait = _wait
            if _wait > 15:
                st.session_state.train_running = False
                st.session_state.train_queued = False
                st.session_state.train_done = True
                st.session_state.train_result = (
                    "fail", "训练",
                    "训练日志始终未生成，训练可能未成功启动，请检查 logs/ 目录。")
                st.error("❌ 训练日志始终未生成，训练可能未成功启动，请检查 logs/ 目录。")
            else:
                st.caption(f"⏳ 等待日志… ({datetime.now().strftime('%H:%M:%S')})")
                time.sleep(2)
                st.rerun()

    # ===== 终态结果提示（不再轮询、不再 rerun）=====
    if st.session_state.get("train_done") and not st.session_state.get("train_running"):
        _res = st.session_state.get("train_result") or ("fail", "当前", "")
        _label = _res[1] if len(_res) > 1 else "当前"
        _detail = _res[2] if len(_res) > 2 else ""
        if _res[0] == "ok":
            st.success(f"🎉 {_label}训练已完成，模型已热加载。")
        else:
            st.error(f"❌ {_label}训练已结束，但失败或异常终止。")
            if _detail:
                st.code(_detail, language=None)
            if st.button("🔁 清除训练结果提示", key="clear_train_result_btn", use_container_width=True):
                st.session_state.train_done = False
                st.session_state.pop("train_result", None)
                st.rerun()

    # 手动热重载模型按钮
    if st.button("🔄 手动重载模型", use_container_width=True, key="reload_model_btn"):
        try:
            predictor.reload()
            clear_predictor_cache()
            if predictor.is_ready:
                st.success("✅ 模型热重载成功！")
            else:
                st.warning("⚠️ 模型文件缺失，请先训练。")
        except Exception as e:
            st.error(f"重载失败: {e}")

    # ====== 🧬 自我进化面板 ======
    st.markdown('<div class="section-title">🧬 自我进化引擎</div>', unsafe_allow_html=True)
    st.info(
        "🧬 系统会根据回测验证结果自动调整模型权重、推荐阈值和市场状态适配，形成「预测→验证→进化→更准」的闭环。"
        "每次在「AI选股」页点击「运行回测验证」时触发进化。"
    )

    evo_summary = evolution.get_summary()

    _ens_on = getattr(app_config.ml, "use_ensemble", False)
    e1, e2, e3 = st.columns(3)
    with e1:
        st.metric(
            "当前模型",
            "XGBoost + 集成" if _ens_on else "XGBoost（单模型）",
            help=("主模型固定为 XGBoost（models/xgboost_model.pkl）。"
                  + ("当前另有 alpha / LSTM 两条腿参与加权融合，"
                     "可由 config.ml.use_ensemble 关闭。"
                     if _ens_on else
                     "config.ml.use_ensemble 已关闭，不做多模型加权。")),
        )
    with e2:
        st.metric(
            "推荐阈值调整",
            f"{evo_summary['threshold_adjustment']:+.1f}",
            help="正数=收紧（更保守），负数=放宽（更激进）",
        )
    with e3:
        st.metric(
            "市场状态",
            evo_summary["current_regime"],
            help="基于沪深300 MA20/MA60 自动检测",
        )

    e4, e5 = st.columns(2)
    with e4:
        st.metric(
            "T+5 平均准确率",
            f"{evo_summary['t5_accuracy']*100:.1f}%" if evo_summary["t5_accuracy"] else "暂无数据",
            delta=f"{evo_summary['t5_batches']} 批次",
        )
    with e5:
        st.metric(
            "T+10 平均准确率",
            f"{evo_summary['t10_accuracy']*100:.1f}%" if evo_summary["t10_accuracy"] else "暂无数据",
            delta=f"{evo_summary['t10_batches']} 批次",
        )

    # 进化历史日志
    if evo_summary["evolution_log"]:
        with st.expander("📜 进化历史（最近 10 条）", expanded=False):
            for entry in reversed(evo_summary["evolution_log"]):
                st.caption(f"`{entry['time'][:16]}` · {entry['action']} → {entry['detail']}")

    # ====== 🧬 离线进化（自动重训）======
    st.markdown('<div class="section-title">🔄 离线进化（自动重训）</div>', unsafe_allow_html=True)
    st.info(
        "当市场风格漂移导致近期预测准确率持续偏低时，自动用**近期数据**重新训练模型。"
        "这是比「在线进化」（只调权重/阈值）更深层的一步——让模型本身适应新市场。"
    )

    off_cfg = evo_summary.get("offline_evolution", {})

    # 启用开关
    off_enabled = st.toggle(
        "启用离线自动重训",
        value=off_cfg.get("enabled", False),
        key="offline_evo_toggle",
        help="开启后，回测验证若发现近期准确率持续偏低，会自动后台重训模型",
    )

    o1, o2, o3 = st.columns(3)
    with o1:
        min_batches = st.number_input(
            "最少验证批数",
            min_value=2, max_value=20, value=int(off_cfg.get("min_batches", 3)),
            key="offline_min_batches",
            help="至少积累这么多批验证数据才考虑重训",
        )
    with o2:
        acc_threshold = st.slider(
            "触发准确率阈值",
            min_value=0.30, max_value=0.60, value=float(off_cfg.get("accuracy_threshold", 0.45)),
            step=0.01, key="offline_acc_threshold",
            help="近期平均准确率低于此值触发重训",
        )
    with o3:
        min_interval = st.number_input(
            "重训最小间隔（天）",
            min_value=1, max_value=60, value=int(off_cfg.get("min_interval_days", 7)),
            key="offline_min_interval",
            help="两次重训之间的最小天数",
        )

    o4, o5 = st.columns(2)
    with o4:
        recent_window = st.number_input(
            "近期窗口（批）",
            min_value=1, max_value=10, value=int(off_cfg.get("recent_window", 3)),
            key="offline_recent_window",
            help="取最近 N 批的平均准确率判断",
        )
    with o5:
        lookback_years = st.number_input(
            "重训数据年限",
            min_value=1, max_value=5, value=int(off_cfg.get("lookback_years", 2)),
            key="offline_lookback_years",
            help="重训用最近 N 年的数据（越小越贴近当前市场）",
        )

    # 保存配置按钮
    if st.button("💾 保存离线进化配置", key="save_offline_cfg_btn", use_container_width=True):
        evolution.set_offline_config(
            enabled=off_enabled,
            min_batches=min_batches,
            accuracy_threshold=acc_threshold,
            min_interval_days=min_interval,
            recent_window=recent_window,
            lookback_years=lookback_years,
        )
        st.success("✅ 离线进化配置已保存")
        st.rerun()

    # 当前状态
    last_retrain = off_cfg.get("last_retrain_at")
    retrain_running = off_cfg.get("retrain_running", False)
    status_text = "🟢 运行中" if retrain_running else (f"上次重训: {last_retrain[:16]}" if last_retrain else "从未重训")
    st.caption(f"状态: {status_text}")

    # 手动触发按钮
    should_rt, rt_reason = evolution.should_retrain()
    if should_rt:
        st.warning(f"🧬 当前满足重训条件：{rt_reason}")
    else:
        st.caption(f"🧬 检查: {rt_reason}")

    col_rt1, col_rt2 = st.columns(2)
    with col_rt1:
        if st.button("🔬 立即离线重训", use_container_width=True, key="manual_offline_retrain_btn",
                     help="用近期数据立即后台重训模型（无视条件）", disabled=retrain_running):
            evolution.mark_retrain_started()
            import subprocess as _sp
            _base = Path(__file__).parent.parent.parent
            _log = _base / "logs" / f"auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            _cmd = [sys.executable, str(_base / "scripts" / "auto_retrain.py"), "--force"]
            if getattr(sys, "frozen", False):
                _cmd = [_cmd[0], "--run-script"] + _cmd[1:]
            _sp.Popen(
                _cmd, cwd=str(_base),
                env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
                stdout=open(_log, "w", encoding="utf-8", errors="replace"),
                stderr=_sp.STDOUT,
                creationflags=_sp.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            st.success(f"✅ 离线重训已在后台启动，日志: `{_log.name}`")
            st.caption("ℹ️ 模型文件更新后看板会自动热重载；如需立即生效也可点下方「手动重载模型」。")
            st.rerun()
    with col_rt2:
        if st.button("✅ 标记重训完成", use_container_width=True, key="mark_retrain_done_btn",
                     help="若后台重训已结束但状态未更新，可手动标记", disabled=not retrain_running):
            evolution.mark_retrain_finished()
            evolution.state["model_weights"] = {"xgboost": 1.0}
            evolution.save()
            st.success("✅ 已标记重训完成（单模型：XGBoost 权重 1.0）")
            st.rerun()

    # 重置按钮（需二次确认）
    reset_confirm = st.checkbox("⚠️ 我确认要重置进化状态", value=False, key="evo_reset_confirm")
    if st.button("🔴 重置进化引擎", disabled=not reset_confirm, key="evo_reset_btn"):
        evolution.reset()
        st.success("✅ 进化状态已重置为初始值（单模型 XGBoost，权重 1.0，阈值 0）")
        st.rerun()
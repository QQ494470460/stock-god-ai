# -*- coding: utf-8 -*-
"""
Streamlit 可视化看板
选股神 AI - 交互式选股决策界面
"""

import sys
import os
import re
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
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from loguru import logger

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
from config.settings import save_scheduler_state, save_ml_state, save_agent_state
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
from src.ui.trading_ui import render_autotrade_page
from src.ui.agent_page import render_agent_page
from src.ui.common import (
    UP_COLOR, DOWN_COLOR, FLAT_COLOR, WARN_COLOR, INFO_COLOR,
    chg_color, chg_arrow, fmt_pct, fmt_num, fmt_yi,
    parse_codes, is_valid_code, build_name_map, lookup_name,
    calc_rsi, calc_macd, calc_kdj, calc_boll, calc_ma, boll_position,
    score_color, score_label, empty_state, require, signal_tag, tag_css,
    metric_card, html_metric,
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


# ========== 大盘涨跌条 ==========

def render_market_ticker():
    """主界面顶部大盘实时涨跌条：主要指数 + 涨跌家数 + 成交额

    性能关键：原来这里无条件拉实时行情，首屏必然卡在联网 + akshare 导入上。
    现在改为「先读会话缓存」：没缓存就直接跳过（不渲染、不联网），
    用户在侧边栏点「加载大盘行情」后再刷新即可显示。
    """
    mk = st.session_state.get("_mk_overview")
    if mk is None:
        # 首屏不阻塞：留一行轻量提示，用户按需加载
        st.caption("📊 大盘行情：侧边栏「大盘速览」→ 点击加载（默认不自动请求，加快首屏）")
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
        # A股习惯：红涨绿跌
        up_flag = pct >= 0
        bg = "#fff1f2" if up_flag else "#ecfdf5"
        bd = "#fecdd3" if up_flag else "#a7f3d0"
        clr = "#dc2626" if up_flag else "#059669"
        arrow = "▲" if up_flag else "▼"
        col.markdown(
            f'<div style="text-align:center;padding:.55rem .3rem;border-radius:10px;'
            f'background:{bg};border:1px solid {bd};min-width:0;">'
            f'<div style="font-size:.78rem;color:#475569;font-weight:600;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis;">{it.名称}</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:{clr};margin:.15rem 0;'
            f'line-height:1.2;">{price:,.2f}</div>'
            f'<div style="font-size:.82rem;font-weight:700;color:{clr};">'
            f'{arrow} {pct:+.2f}%</div>'
            f'<div style="font-size:.68rem;color:#94a3b8;">{chg:+.2f}</div>'
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
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
            f'padding:.55rem .7rem;">'
            f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">🌡️ 市场情绪</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:{mood_clr};">{mood}</div>'
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


# ========== 页面配置 ==========
st.set_page_config(
    page_title="选股神 AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 全局样式 ==========
st.markdown("""
<style>
    /* ====== 纯色大标题（不用渐变偷懒，避免透明字） ====== */
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #5b21b6;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    /* ====== 节标题 ====== */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f2937;
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #e5e7eb;
    }

    /* ====== 卡片组通用 ====== */
    .card-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin: 0.8rem 0;
    }
    .mini-card {
        flex: 1;
        min-width: 120px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .mini-card .label {
        font-size: 0.72rem;
        font-weight: 500;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .mini-card .val {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 0.2rem;
    }

    /* ====== 信号标签 ====== */
    /* A 股习惯：看多=红、看空=绿。类名与 signal_tag() 输出保持一致。 */
    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.5rem 0;
    }
    .tag {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.35rem 0.7rem;
        border-radius: 18px;
        font-size: 0.8rem;
        font-weight: 500;
        white-space: nowrap;
    }
    .tag.bullish { background: #fee2e2; border: 1px solid #fca5a5; color: #b91c1c; }
    .tag.bearish { background: #d1fae5; border: 1px solid #6ee7b7; color: #047857; }
    .tag.warn    { background: #fef3c7; border: 1px solid #fcd34d; color: #b45309; }
    .tag.info    { background: #dbeafe; border: 1px solid #93c5fd; color: #1d4ed8; }
    /* 兼容历史写法 bull/bear（语义同上） */
    .tag.bull { background: #fee2e2; border: 1px solid #fca5a5; color: #b91c1c; }
    .tag.bear { background: #d1fae5; border: 1px solid #6ee7b7; color: #047857; }

    /* ====== 侧边栏 ====== */
    section[data-testid="stSidebar"] {
        background: #f9fafb;
    }

    /* ====== 颜色快捷类 ====== */
    .red   { color: #dc2626; font-weight: 600; }
    .green { color: #059669; font-weight: 600; }
    .amber { color: #d97706; font-weight: 600; }
    .purple{ color: #7c3aed; font-weight: 600; }
    .gray  { color: #6b7280; }

    /* ====== 补充此前被引用但从未定义的类 ====== */
    /* 代码里一直在用 .text-sub / .chart-box，但全局样式里没有它们，
       等于 no-op，样式全靠同标签的 inline style 撑着。这里补齐。 */
    .text-sub {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
        line-height: 1.4;
    }
    .chart-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.6rem 0;
    }
    .chart-box h3 {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e293b;
        margin: 0 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;">
        <span style="font-size:2.2rem;">📈</span>
        <span style="font-size:1.3rem; font-weight:700; color:#1e293b;">选股神 AI</span>
    </div>
    <p style="color:#94a3b8; font-size:0.78rem; margin-top:-0.3rem;">XGBoost 单模型引擎</p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state.page = "🏠 概览"

    page_options = ["🏠 概览", "🔍 AI选股", "📊 个股分析", "📡 实时监控", "🤖 AI 助手", "💹 自动交易", "📐 量化模型", "📈 模型稳定性", "⚙️ 设置"]

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

    st.markdown("---")
    ready = predictor.is_ready
    s_cls = "green" if ready else "red"
    dot_clr = "#10b981" if ready else "#ef4444"
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
        <span style="width:8px; height:8px; border-radius:50%; background:{dot_clr}; display:inline-block;"></span>
        <span style="font-size:0.82rem; color:#64748b;">模型: <b class="{s_cls}">{'就绪' if ready else '未训练'}</b></span>
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

    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ========== 🏠 概览页 ==========
if page == "🏠 概览":
    st.markdown('<p class="main-header">📈 选股神 AI Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI驱动的A股智能选股系统 — XGBoost 单模型</p>', unsafe_allow_html=True)

    # ===== 大盘实时涨跌（主界面顶部）=====
    render_market_ticker()

    # KPI卡片
    st.markdown("""
    <div class="card-row">
        <div class="mini-card">
            <div class="label">📊 股票池</div>
            <div class="val">{pool} 只</div>
        </div>
        <div class="mini-card">
            <div class="label">🧠 模型状态</div>
            <div class="val">{ready_txt}</div>
        </div>
        <div class="mini-card">
            <div class="label">⚡ 预测引擎</div>
            <div class="val">XGBoost</div>
        </div>
        <div class="mini-card">
            <div class="label">🎯 决策树数量</div>
            <div class="val">{trees}</div>
        </div>
    </div>
    """.format(
        pool=len(_cached_stock_list()),
        ready_txt="✅ 就绪" if predictor.is_ready else "❌ 未训练",
        # 注意：这里必须用内部字段 _xgb_model，不能用 xgb_model property ——
        # property 会触发 _ensure_loaded()，等于首屏就把模型反序列化一遍
        # （涉及 xgboost / sklearn 等重库，耗时明显）。未加载时显示 "—" 即可。
        trees=predictor._xgb_model.n_estimators if predictor._xgb_model else "—",
    ), unsafe_allow_html=True)

    # 市场热点
    st.markdown('<div class="section-title">🔥 市场热点板块</div>', unsafe_allow_html=True)

    # 性能关键：热点板块同样要联网（含 akshare 导入 ~1.9s）。
    # 首屏先不拉，给一个加载按钮；点过一次后结果缓存在 session_state。
    if st.session_state.get("_hotspots") is None:
        st.caption("点击加载市场热点板块（默认不自动请求，加快首屏）")
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

    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown('<div class="text-sub" style="text-align:center; margin-bottom:4px;">① 模型就绪</div>', unsafe_allow_html=True)
        ready = predictor.is_ready
        st.markdown(f'''<div style="text-align:center; font-size:2rem; margin-bottom:4px;">{'✅' if ready else '⚠️'}</div>''', unsafe_allow_html=True)
        st.markdown(f'<div class="{'green' if ready else 'red'}" style="text-align:center; font-size:0.8rem;">{'已训练，可直接预测' if ready else '需要先训练'}</div>', unsafe_allow_html=True)

    with q2:
        if st.button("🤖 AI 选股", type="primary", use_container_width=True, key="quick_pick"):
            st.session_state.page = "🔍 AI选股"
            st.session_state.auto_scan = True
            del st.session_state._nav_radio  # 清 radio 缓存，下次用 index
            st.rerun()
        st.markdown('<div class="text-sub" style="text-align:center; cursor:pointer;">获取智能推荐</div>', unsafe_allow_html=True)

    with q3:
        if st.button("📊 个股分析", use_container_width=True, key="quick_analyze"):
            st.session_state.page = "📊 个股分析"
            st.session_state.auto_analyze_code = "000001"
            del st.session_state._nav_radio  # 清 radio 缓存，下次用 index
            st.rerun()
        st.markdown('<div class="text-sub" style="text-align:center; cursor:pointer;">深度研判走势</div>', unsafe_allow_html=True)

# ========== 🔍 AI选股页 ==========
elif page == "🔍 AI选股":
    st.markdown('<p class="main-header">🤖 AI 智能选股</p>', unsafe_allow_html=True)

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
    st.markdown('<p class="main-header">📊 个股深度分析</p>', unsafe_allow_html=True)

    # 快开始：首页按钮触发的自动分析
    auto_analyze_code = st.session_state.pop("auto_analyze_code", None)

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

            # 股票信息
            stock_info = None
            try:
                slist = _cached_stock_list()
                stock_info = slist[slist["code"] == stock_code]
                stock_name = stock_info["name"].iloc[0] if len(stock_info) > 0 else stock_code
            except Exception:
                stock_name = stock_code

            st.markdown(f'<div class="section-title">{stock_name} ({stock_code}) · 收盘 {last_close:.2f} <span class="{("red" if last_close >= prev_close else "green")}">{(last_close-prev_close):+.2f}</span></div>', unsafe_allow_html=True)

            # ====== Row 1: AI 预测 + 基本面 ======
            ai_col, fund_col = st.columns([3, 2])

            with ai_col:
                if predictor.is_ready:
                    pred = predictor.predict_single(df, stock_code, stock_name)
                    score = pred.score
                    # 分档按 2026-10-10 单模型化后实测分布（38~60，mean 47.3）校准；
                    # 旧值 80/65 在新分布下恒为 green，高分股永远不高亮
                    s_cls = "red" if score >= 55 else ("amber" if score >= 49 else "green")
                    # 单模型化后 agreement 恒为 1.0，再显示「一致/分歧」没有信息量，
                    # 改为展示 XGBoost 的方向信号
                    agree_str = "📈 模型看多" if pred.xgb_signal == 1 else ("📉 模型看空" if pred.xgb_signal == -1 else "➡️ 模型观望")
                    xgb_sig = "📈 买入" if pred.xgb_signal == 1 else ("📉 卖出" if pred.xgb_signal == -1 else "➡️ 持有")

                    # 交易建议（单模型：仅依据 XGBoost 信号）
                    if score >= 55 and pred.xgb_signal == 1:
                        action = "🟢 强烈看多"
                        action_cls = "red"
                        entry_range = f"{last_close*0.98:.2f}~{last_close:.2f}"
                        stop_loss = f"{last_close*0.93:.2f}"
                        target = f"{last_close*1.08:.2f} ({last_close*1.15:.2f})"
                    elif score >= 51 and pred.xgb_signal == 1:
                        action = "🟡 谨慎看多"
                        action_cls = "amber"
                        entry_range = f"{last_close*0.97:.2f}~{last_close:.2f}"
                        stop_loss = f"{last_close*0.94:.2f}"
                        target = f"{last_close*1.06:.2f}"
                    elif score < 45 and pred.xgb_signal == -1:
                        action = "🔴 明确看空"
                        action_cls = "green"
                        entry_range = "—"
                        stop_loss = "—"
                        target = "—"
                    else:
                        action = "⚪ 观望为主"
                        action_cls = ""
                        entry_range = "—"
                        stop_loss = "—"
                        target = "—"

                    st.markdown(f"""
                    <div class="card-row">
                        <div class="mini-card" style="background:#f5f3ff;border-color:#ddd6fe;">
                            <div class="label">🤖 AI评分</div>
                            <div class="val {s_cls}">{score:.0f}</div>
                            <div style="font-size:0.65rem;color:#94a3b8;">{agree_str}</div>
                        </div>
                        <div class="mini-card" style="background:#fef2f2;border-color:#fecaca;">
                            <div class="label">📈 上涨概率</div>
                            <div class="val red">{pred.prob_up*100:.1f}%</div>
                        </div>
                        <div class="mini-card" style="background:#fffbeb;border-color:#fde68a;">
                            <div class="label">➡️ 震荡概率</div>
                            <div class="val amber">{pred.prob_neutral*100:.1f}%</div>
                        </div>
                        <div class="mini-card" style="background:#ecfdf5;border-color:#a7f3d0;">
                            <div class="label">📉 下跌概率</div>
                            <div class="val green">{pred.prob_down*100:.1f}%</div>
                        </div>
                        <div class="mini-card">
                            <div class="label">🔮 模型信号</div>
                            <div class="val" style="font-size:0.8rem;line-height:1.5;">
                                确信度: {pred.confidence*100:.0f}%<br>波动率: {pred.volatility*100:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 交易建议卡
                    st.markdown(f"""
                    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;">
                            <div style="min-width:100px;"><span style="font-size:0.7rem;color:#64748b;">交易建议</span><br><b class="{action_cls}" style="font-size:1.1rem;">{action}</b></div>
                            <div style="min-width:100px;"><span style="font-size:0.7rem;color:#64748b;">建议入场</span><br><b style="color:#1e293b;">{entry_range}</b></div>
                            <div style="min-width:100px;"><span style="font-size:0.7rem;color:#64748b;">止损位</span><br><b style="color:#ef4444;">{stop_loss}</b></div>
                            <div style="min-width:100px;"><span style="font-size:0.7rem;color:#64748b;">目标位</span><br><b style="color:#059669;">{target}</b></div>
                            <div style="min-width:80px;"><span style="font-size:0.7rem;color:#64748b;">风险收益比</span><br><b style="color:#1e293b;">{('1:2' if score>=70 else ('1:1.5' if score>=60 else '—'))}</b></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # 基本面
            with fund_col:
                st.markdown('<div style="font-weight:600;color:#475569;font-size:0.85rem;margin-bottom:6px;">📋 基本面概况</div>', unsafe_allow_html=True)
                fund_items = []
                if stock_info is not None and len(stock_info) > 0:
                    info = stock_info.iloc[0]
                    fund_items.extend([
                        ("行业", info.get("industry", "—")[:8], "purple"),
                        ("总市值", f"{info.get('total_cap',0):.0f}亿", ""),
                        ("流通市值", f"{info.get('float_cap',0):.0f}亿", ""),
                        ("上市日期", str(info.get("listed_date","—"))[:10], ""),
                    ])
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
                                    c = "red" if val > 15 else ("amber" if val > 8 else "green")
                                elif key == "profit_growth":
                                    c = "red" if val > 20 else ("amber" if val > 0 else "green")
                                else:
                                    c = ""
                                fund_items.append((label, f"{val:{fmt}}{unit}", c))
                except Exception:
                    pass

                # 行业热度
                try:
                    sectors = _cached_hotspots()
                    if stock_info is not None and len(stock_info) > 0:
                        ind = stock_info.iloc[0].get("industry", "")
                        if ind:
                            match = sectors[sectors["板块"].str.contains(ind[:2], na=False)]
                            if len(match) > 0:
                                s = match.iloc[0]
                                spct = s["涨跌幅"]
                                fund_items.append(("板块涨跌", f"{spct:+.2f}%", "red" if spct>0 else "green"))
                except Exception:
                    pass

                for label, val, cls in fund_items:
                    c_style = f"color:#ef4444" if cls == "red" else (f"color:#059669" if cls == "green" else (f"color:#8b5cf6" if cls == "purple" else "color:#1e293b"))
                    st.markdown(f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #f1f5f9;font-size:0.8rem;"><span style="color:#64748b;">{label}</span><span style="{c_style};font-weight:600;">{val}</span></div>', unsafe_allow_html=True)

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
                fig1.add_hline(y=val, line_dash="dash", line_color=color, opacity=0.5,
                               annotation_text=label.split("(")[0], annotation_position="right")

            fig1.update_layout(height=500, template="plotly_white", hovermode="x unified",
                               xaxis_rangeslider_visible=True, legend=dict(orientation="h", y=1.15),
                               margin=dict(l=10, r=10, t=20, b=10))
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
                               margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ====== 技术信号 + K线形态 ======
            st.markdown('<div class="section-title">🧠 技术信号 + K线形态</div>', unsafe_allow_html=True)

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

            # 渲染 K线形态
            if candle_patterns:
                st.markdown('<div class="tag-row" style="margin-bottom:8px;"><span style="color:#64748b;font-size:0.75rem;margin-right:8px;">🕯️ K线形态:</span>', unsafe_allow_html=True)
                for tag_cls, title, desc in candle_patterns:
                    st.markdown(f'<span class="tag {tag_cls}"><b>{title}</b> {desc}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 渲染技术信号
            st.markdown('<div class="tag-row">', unsafe_allow_html=True)
            for tag_cls, title, desc in signals:
                st.markdown(f'<span class="tag {tag_cls}"><b>{title}</b> {desc}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ====== 风险评估 ======
            risk_col, stats_col = st.columns([1, 1])

            with risk_col:
                st.markdown('<div class="section-title">⚠️ 风险评估</div>', unsafe_allow_html=True)
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

                risk_metrics = [
                    ("VaR(95%)", f"{var95:.2f}%", "当日最大亏损估计", "red" if var95 < -3 else ("amber" if var95 < -1.5 else "green")),
                    ("最大回撤", f"{max_dd:.1f}%", "历史最大跌幅", "red" if max_dd < -25 else ("amber" if max_dd < -10 else "green")),
                    ("夏普比率", f"{sharpe:.2f}", "风险调整后收益", "red" if sharpe > 1 else ("amber" if sharpe > 0 else "green")),
                    ("年化波动", f"{ann_vol:.1f}%", "价格波动程度", ""),
                    ("60日胜率", f"{win_rate:.0f}%", "阳线占比", "red" if win_rate > 55 else ("amber" if win_rate > 45 else "green")),
                ]
                for label, val, desc, cls in risk_metrics:
                    c = "color:#ef4444" if cls == "red" else ("color:#059669" if cls == "green" else ("color:#f59e0b" if cls == "amber" else "color:#1e293b"))
                    st.markdown(f'<div style="display:flex;justify-content:space-between;padding:4px 12px;border-bottom:1px solid #f1f5f9;font-size:0.8rem;"><span style="color:#64748b;">{label}</span><div style="text-align:right;"><span style="{c};font-weight:600;">{val}</span><br><span style="color:#94a3b8;font-size:0.65rem;">{desc}</span></div></div>', unsafe_allow_html=True)

            with stats_col:
                st.markdown('<div class="section-title">📊 多周期统计</div>', unsafe_allow_html=True)
                for label, period_data in [("近 5 日", df.tail(5)), ("近 20 日", df.tail(20)), ("近 60 日", df.tail(60))]:
                    cd = period_data["close"]
                    ret = (cd.iloc[-1] / cd.iloc[0] - 1) * 100
                    high_p = cd.max()
                    low_p = cd.min()
                    amp = (period_data["high"].max() - period_data["low"].min()) / cd.iloc[0] * 100
                    vol_price = cd.pct_change().std() * 100
                    win = (cd.pct_change().dropna() > 0).mean() * 100
                    rc_cls = "red" if ret > 0 else "green"

                    st.markdown(f"""
                    <div class="card-row">
                        <div class="mini-card" style="padding:6px 8px;"><div class="label" style="font-size:0.6rem;">周期</div><div class="val purple" style="font-size:0.85rem;">{label}</div></div>
                        <div class="mini-card" style="padding:6px 8px;"><div class="label" style="font-size:0.6rem;">涨跌幅</div><div class="val {rc_cls}" style="font-size:0.85rem;">{ret:+.2f}%</div></div>
                        <div class="mini-card" style="padding:6px 8px;"><div class="label" style="font-size:0.6rem;">最高</div><div class="val red" style="font-size:0.85rem;">{high_p:.2f}</div></div>
                        <div class="mini-card" style="padding:6px 8px;"><div class="label" style="font-size:0.6rem;">最低</div><div class="val green" style="font-size:0.85rem;">{low_p:.2f}</div></div>
                        <div class="mini-card" style="padding:6px 8px;"><div class="label" style="font-size:0.6rem;">波动率</div><div class="val" style="font-size:0.85rem;">{vol_price:.2f}%</div></div>
                        <div class="mini-card" style="padding:6px 8px;"><div class="label" style="font-size:0.6rem;">振幅</div><div class="val" style="font-size:0.85rem;">{amp:.2f}%</div></div>
                        <div class="mini-card" style="padding:6px 8px;"><div class="label" style="font-size:0.6rem;">胜率</div><div class="val {('red' if win>50 else 'green')}" style="font-size:0.85rem;">{win:.0f}%</div></div>
                    </div>
                    """, unsafe_allow_html=True)

            # ====== AI特征贡献 ======
            if predictor.is_ready and pred.features_snapshot:
                st.markdown('<div class="section-title">🧬 AI 特征快照 Top 15</div>', unsafe_allow_html=True)
                st.caption("💡 正值(红)=看涨贡献 负值(绿)=看跌贡献 | 绝对值越大对AI决策影响越大")

                feat = pred.features_snapshot
                meaningful = {kv: v for kv, v in feat.items() if abs(v) > 1e-6}
                feat_df = pd.DataFrame(
                    sorted(meaningful.items(), key=lambda x: abs(x[1]), reverse=True)[:15],
                    columns=["特征", "当前值"]
                )
                feat_df["方向"] = feat_df["当前值"].apply(lambda x: "#ef4444" if x > 0 else "#10b981")

                fig4 = go.Figure(data=[go.Bar(
                    x=feat_df["当前值"], y=feat_df["特征"], orientation="h",
                    marker_color=feat_df["方向"],
                    text=feat_df["当前值"].round(3), textposition="outside",
                    textfont=dict(color="#1e293b", size=11)
                )])
                fig4.update_layout(height=420, template="plotly_white",
                                   xaxis_title="特征值 (红=看涨, 绿=看跌)",
                                   xaxis=dict(zeroline=True, zerolinecolor="#94a3b8"),
                                   margin=dict(l=10, r=50, t=10, b=10))
                st.plotly_chart(fig4, use_container_width=True)

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
    st.markdown('<p class="main-header">📡 AI 实时监控</p>', unsafe_allow_html=True)

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
        _default = st.session_state.get("watch_code") or (_wl[0] if _wl else "")
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

    need_watch = watch_go or watch_refresh
    # 已带出代码但本次会话还没盯过 → 首屏立即跑一次（解决「打开即空」）
    if watch_code and "watch_last_ts" not in st.session_state:
        need_watch = True
    elif watch_code and watch_auto and "watch_last_ts" in st.session_state:
        elapsed = (datetime.now() - st.session_state.watch_last_ts).total_seconds()
        if elapsed > watch_interval:
            need_watch = True

    # 盯盘结果缓存：面板统一从缓存渲染，避免交互/刷新间隙面板闪空
    if "watch_cache" not in st.session_state:
        st.session_state.watch_cache = {}

    if need_watch and watch_code:
        st.session_state.watch_last_ts = datetime.now()
        code = watch_code.strip().upper()
        if not is_valid_code(code):
            st.warning("请输入6位股票代码")
        else:
            # 股票名：走统一的 code/name 映射（列表列名是英文，不能用中文列取）
            stock_name = get_name(code)

            # 日线数据
            df_k = _cached_daily_kline(code, "", "")
            if len(df_k) < 30:
                st.warning(f"{code} {stock_name} 数据不足")
            else:
                close = df_k["close"]
                last_close = float(close.iloc[-1])
                prev_close = float(close.iloc[-2])
                day_chg = (last_close / prev_close - 1) * 100

                # 获取实时行情（交易时段）
                live_price = last_close
                live_chg = day_chg
                live_vol = int(df_k["volume"].iloc[-1])
                if market_open:
                    # 优先腾讯源（稳定快速），失败回退东方财富
                    spot = fetcher.get_realtime_quote(code)
                    if spot and spot.get("最新价", 0) > 0:
                        live_price = float(spot["最新价"])
                        live_chg = float(spot.get("涨跌幅", day_chg))
                        if spot.get("成交量", 0) > 0:
                            live_vol = int(spot["成交量"])
                    else:
                        try:
                            import akshare as ak
                            df_spot = ak.stock_zh_a_spot_em()
                            df_spot = df_spot.set_index("代码")
                            if code in df_spot.index:
                                spot2 = df_spot.loc[code]
                                live_price = float(spot2.get("最新价", last_close))
                                live_chg = float(spot2.get("涨跌幅", day_chg))
                                live_vol = int(float(spot2.get("成交量", 0))) if spot2.get("成交量", 0) else live_vol
                        except Exception:
                            pass

                # AI 预测（双层 TTL）：行情每次都刷新，AI 模型按更长周期重算，避免盘中卡模型
                _prev = st.session_state.watch_cache
                _cache_pred = _prev.get("pred") if _prev.get("code") == code else None
                _ai_ts = _prev.get("ai_fetched_at")
                _ai_age = (datetime.now() - _ai_ts).total_seconds() if _ai_ts else 1e9
                _force_ai = bool(watch_go or watch_refresh) or (_cache_pred is None) \
                    or (_ai_age > watch_ai_interval) or (_prev.get("code") != code)
                if _force_ai:
                    pred = None
                    try:
                        pred = predictor.predict_single(df_k, code, stock_name)
                    except Exception:
                        pred = None
                    ai_fetched_at = datetime.now()
                else:
                    pred = _cache_pred
                    ai_fetched_at = _ai_ts

                # 写入缓存：面板渲染统一从缓存读，不依赖本次是否刚拉取
                st.session_state.watch_cache = {
                    "code": code,
                    "stock_name": stock_name,
                    "live_price": live_price,
                    "live_chg": live_chg,
                    "live_vol": live_vol,
                    "df_k": df_k,
                    "pred": pred,
                    "fetched_at": datetime.now(),
                    "ai_fetched_at": ai_fetched_at,
                    "market_open": market_open,
                }

    # ---- 渲染盯盘面板：始终从缓存渲染，保证面板不闪烁/不空置 ----
    _cache = st.session_state.get("watch_cache") or {}
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
            # 用自绘卡片而非 st.metric：metric 的 delta_color="normal"
            # 是欧美配色（涨绿跌红），在 A 股界面里语义是反的
            metric_card(
                label=f"{chg_icon} {stock_name}({code})",
                value=f"{live_price:.2f}",
                chg=live_chg,
            )
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
            # RSI
            rsi_v = float(calc_rsi(close).iloc[-1])
            rsi_label = "超买" if rsi_v > 70 else ("超卖" if rsi_v < 30 else ("偏强" if rsi_v > 50 else "偏弱"))
            st.metric(label=f"RSI(14)·{rsi_label}", value=f"{rsi_v:.0f}")
        with pc5:
            if pred:
                ai_icon = "📈" if pred.prob_up > 0.5 else ("📉" if pred.prob_down > 0.5 else "➡️")
                st.metric(label=f"{ai_icon} AI评分", value=f"{pred.score:.0f}", delta=f"确信{pred.confidence*100:.0f}%")
            else:
                st.metric(label="AI评分", value="—")

        # 交易信号 + K线
        tc1, tc2 = st.columns([1, 1.5])
        with tc1:
            if pred:
                st.markdown('<div class="section-title">🎯 交易信号</div>', unsafe_allow_html=True)
                reason_html = ""
                for r in getattr(pred, 'reasons', [])[:4]:
                    reason_html += f'<div style="font-size:0.8rem;padding:2px 0;color:#475569;">• {r}</div>'

                # 建议
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

                # 概率条
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
            # 迷你K线 (最近60天)
            df_plot = df_k.iloc[-60:].copy()
            fig_mini = go.Figure()
            colors_k = ["#ef4444" if df_plot["close"].iloc[i] >= df_plot["open"].iloc[i] else "#059669" for i in range(len(df_plot))]
            fig_mini.add_trace(go.Candlestick(
                x=df_plot["date"], open=df_plot["open"], high=df_plot["high"],
                low=df_plot["low"], close=df_plot["close"],
                increasing_line_color="#ef4444", decreasing_line_color="#059669",
                name="K线"
            ))
            # MA5, MA20
            fig_mini.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["close"].rolling(5).mean(),
                                           mode="lines", line=dict(color="#f59e0b", width=1), name="MA5"))
            fig_mini.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["close"].rolling(20).mean(),
                                           mode="lines", line=dict(color="#8b5cf6", width=1.2), name="MA20"))
            # 支撑/压力线
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
            st.plotly_chart(fig_mini, use_container_width=True)

        # 技术指标速览
        ti1, ti2, ti3, ti4, ti5, ti6 = st.columns(6)
        # MACD
        dif, dea, macd_bar = calc_macd(close)
        macd_val = float(macd_bar.iloc[-1])
        with ti1:
            st.metric(label="MACD", value=f"{macd_val:.3f}", delta="金叉" if macd_val > 0 and float(macd_bar.iloc[-2]) <= 0 else ("死叉" if macd_val < 0 and float(macd_bar.iloc[-2]) >= 0 else None))
        # KDJ
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
        # 布林带位置
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_pos = (last_close - float(bb_mid.iloc[-1])) / (2 * float(bb_std.iloc[-1]) + 1e-10)
        with ti4:
            st.metric(label="布林位置", value=f"{bb_pos:.2f}σ", delta="上轨附近" if bb_pos > 0.8 else ("下轨附近" if bb_pos < -0.8 else None))
        # 涨幅统计
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
            else:
                _src = "非交易时段 · 仅显示上一交易日收盘"
            _cap = f"🕒 行情 {_age} 秒前 · AI评分" + (f" {_ai_age} 秒前" if _ai_age is not None else " —") + f"（{_src}）"
            st.caption(_cap)
        st.divider()
    else:
        st.caption("👆 输入股票代码，或点选上方自选股，开始盯盘")

    # 自动刷新倒计时
    if watch_code and watch_auto and "watch_last_ts" in st.session_state:
        remaining = watch_interval - int((datetime.now() - st.session_state.watch_last_ts).total_seconds())
        if remaining > 0:
            st.caption(f"⏱️ 盯盘 {remaining} 秒后自动刷新")
        else:
            st.rerun()

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
    st.markdown('<p class="main-header">📐 量化策略模型</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">多因子量化评分 · 均值回归回测 · 风险平价配置</p>', unsafe_allow_html=True)

    quant_tab1, quant_tab2, quant_tab3 = st.tabs(["📊 多因子评分", "🔄 均值回归", "⚖️ 风险平价"])

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



# ========== 📈 模型稳定性页 ==========
if page == "📈 模型稳定性":
    st.markdown('<p class="main-header">📈 模型稳定性（RankIC 滚动监控）</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">逐日横截面 RankIC 稳定性曲线 · 由收盘后自动回填累积</p>', unsafe_allow_html=True)

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


if page == "⚙️ 设置":
    st.markdown('<p class="main-header">⚙️ 系统设置</p>', unsafe_allow_html=True)

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
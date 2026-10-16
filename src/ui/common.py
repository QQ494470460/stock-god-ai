# -*- coding: utf-8 -*-
"""UI 公共工具：配色、格式化、技术指标、解析。

这个模块存在的意义是消除看板各页面之间的"同物不同实现"：
同一只股票的 MACD 在两个页面算出两个值、涨跌色一会红一会绿、
6 位代码校验抄了三遍 —— 全部收敛到这里，只留一份。
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

# ==================== 配色（A 股习惯：红涨绿跌） ====================
# 注意与欧美相反。Streamlit 的 st.metric(delta_color="normal") 是
# 涨=绿/跌=红，绝不能用在中文 A 股场景。
UP_COLOR = "#dc2626"      # 涨 / 看多 / 利好
DOWN_COLOR = "#059669"    # 跌 / 看空 / 利空
FLAT_COLOR = "#64748b"    # 平 / 中性
WARN_COLOR = "#d97706"    # 警示
INFO_COLOR = "#3b82f6"    # 提示


def chg_color(v: float) -> str:
    """按涨跌返回颜色：正红、负绿、零灰"""
    if v > 0:
        return UP_COLOR
    if v < 0:
        return DOWN_COLOR
    return FLAT_COLOR


def chg_arrow(v: float) -> str:
    """涨跌箭头"""
    if v > 0:
        return "▲"
    if v < 0:
        return "▼"
    return "—"


# ==================== 格式化 ====================

def fmt_pct(v: Optional[float], digits: int = 2, sign: bool = True) -> str:
    """格式化百分比。0.0123 -> '+1.23%'"""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    fmt = f"{{:+.{digits}f}}%" if sign else f"{{:.{digits}f}}%"
    return fmt.format(v)


def fmt_num(v: Optional[float], digits: int = 2) -> str:
    """格式化数字，带千分位"""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:,.{digits}f}"


def fmt_yi(v: Optional[float], digits: int = 0) -> str:
    """格式化成交额（元 -> 亿元）"""
    if not v:
        return "—"
    return f"{v / 1e8:,.{digits}f}"


def fmt_color_span(text: str, v: float) -> str:
    """把文本包成带涨跌色的 HTML span"""
    return f'<span style="color:{chg_color(v)};font-weight:600;">{text}</span>'


# ==================== 解析 / 校验 ====================

def parse_codes(text: str) -> List[str]:
    """解析逗号分隔的股票代码串，兼容中文逗号与空格"""
    if not text:
        return []
    parts = re.split(r"[,，\s]+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def is_valid_code(code: str) -> bool:
    """6 位纯数字股票代码"""
    return bool(re.match(r"^\d{6}$", str(code).strip()))


def build_name_map(stock_list: Optional[pd.DataFrame]) -> dict:
    """从股票列表构建 code -> name 映射。

    列表列名在不同数据源下可能是英文(code/name)或中文(代码/名称)，
    这里统一兼容，避免调用方 KeyError。
    """
    if stock_list is None or len(stock_list) == 0:
        return {}
    c_col = "code" if "code" in stock_list.columns else ("代码" if "代码" in stock_list.columns else None)
    n_col = "name" if "name" in stock_list.columns else ("名称" if "名称" in stock_list.columns else None)
    if not c_col or not n_col:
        return {}
    return dict(zip(stock_list[c_col].astype(str), stock_list[n_col].astype(str)))


def lookup_name(name_map: dict, code: str, fallback: str = "") -> str:
    """查股票名，查不到回退到代码本身（而不是抛异常）"""
    return name_map.get(str(code).strip()) or fallback or str(code)


# ==================== 技术指标（统一实现，避免各页面算得不一样） ====================

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI —— Wilder 平滑（alpha=1/period），与通达信/同花顺一致"""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD —— 递推式 EWM(adjust=False)，返回 (dif, dea, hist)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2  # A股习惯：MACD 柱 = 2*(DIF-DEA)
    return dif, dea, hist


def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3):
    """KDJ —— 标准 9,3,3，返回 (k, d, j)"""
    low_n = df["low"].rolling(n, min_periods=1).min()
    high_n = df["high"].rolling(n, min_periods=1).max()
    rng = (high_n - low_n).replace(0, np.nan)
    rsv = ((df["close"] - low_n) / rng * 100).fillna(50.0)
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_boll(close: pd.Series, n: int = 20, k: float = 2.0):
    """布林带，返回 (mid, upper, lower)"""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return mid, mid + k * std, mid - k * std


def boll_position(close: float, mid: float, std: float) -> float:
    """价格在布林带中的相对位置：0=下轨，1=上轨"""
    if not std or np.isnan(std):
        return 0.5
    return (close - mid) / (2 * std)


def calc_ma(close: pd.Series, windows: Sequence[int] = (5, 10, 20, 60)) -> dict:
    """批量均线"""
    return {w: close.rolling(w).mean() for w in windows}


# ==================== 评分配色（统一阈值） ====================

def score_color(score: float, high: float = 75.0, mid: float = 60.0, low: float = 45.0) -> str:
    """AI 评分 -> 颜色。全站统一阈值，避免同一分数不同页面不同色"""
    if score >= high:
        return UP_COLOR
    if score >= mid:
        return WARN_COLOR
    if score >= low:
        return "#475569"
    return DOWN_COLOR


def score_label(score: float, high: float = 80.0, mid: float = 65.0, low: float = 50.0) -> str:
    """AI 评分 -> 文字档位"""
    if score >= high:
        return "强力推荐"
    if score >= mid:
        return "推荐"
    if score >= low:
        return "关注"
    return "观望"


# ==================== 空状态 / 提示 ====================

def empty_state(msg: str, icon: str = "📭", kind: str = "info") -> None:
    """统一的空状态提示"""
    text = f"{icon} {msg}"
    if kind == "warning":
        st.warning(text)
    elif kind == "error":
        st.error(text)
    else:
        st.info(text)


def require(cond: bool, msg: str, icon: str = "⚠️") -> bool:
    """条件不满足时提示并 st.stop()。返回 True 表示已终止。"""
    if cond:
        return False
    st.warning(f"{icon} {msg}")
    st.stop()
    return True


# ==================== 技术信号标签 ====================

def signal_tag(kind: str, title: str, desc: str = "") -> str:
    """生成技术信号标签 HTML。

    kind: bullish(看多/红) / bearish(看空/绿) / warn(警示) / info(提示)
    """
    kind = (kind or "info").lower()
    alias = {
        "bull": "bullish", "bullish": "bullish", "up": "bullish", "red": "bullish",
        "bear": "bearish", "bearish": "bearish", "down": "bearish", "green": "bearish",
        "neutral": "info", "info": "info", "blue": "info",
        "amber": "warn", "warn": "warn", "orange": "warn",
    }
    cls = alias.get(kind, "info")
    txt = f"<b>{title}</b>"
    if desc:
        txt += f" {desc}"
    return f'<span class="tag {cls}">{txt}</span>'


def tag_css() -> str:
    """技术信号标签的 CSS。A股：看多=红，看空=绿。"""
    return """
    .tag { display:inline-block; padding:3px 10px; margin:2px 4px 2px 0;
           border-radius:6px; font-size:0.78rem; line-height:1.5; }
    .tag.bullish { background:#fee2e2; border:1px solid #fca5a5; color:#b91c1c; }
    .tag.bearish { background:#d1fae5; border:1px solid #6ee7b7; color:#047857; }
    .tag.warn    { background:#fef3c7; border:1px solid #fcd34d; color:#b45309; }
    .tag.info    { background:#dbeafe; border:1px solid #93c5fd; color:#1d4ed8; }
    """


# ==================== 带涨跌色的指标卡 ====================

def html_metric(label: str, value: str, chg: Optional[float] = None,
                chg_text: str = "", bg: str = "#f8fafc") -> str:
    """自绘指标卡 HTML。

    st.metric 的 delta_color 只有 "normal"(涨绿跌红，欧美习惯) 和 "off",
    都无法表达 A 股的红涨绿跌 —— 涨跌类指标一律用这个。
    """
    if chg is None:
        tail = f'<span style="color:#64748b;font-size:0.78rem;">{chg_text}</span>' if chg_text else ""
    else:
        tail = (f'<span style="color:{chg_color(chg)};font-weight:600;font-size:0.82rem;">'
                f'{chg_arrow(chg)} {chg_text or fmt_pct(chg)}</span>')
    return (
        f'<div style="background:{bg};border:1px solid #e2e8f0;border-radius:10px;'
        f'padding:.55rem .7rem;">'
        f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;">{label}</div>'
        f'<div style="font-size:1.05rem;font-weight:700;color:#0f172a;">{value}</div>'
        f'<div style="margin-top:2px;">{tail}</div>'
        f'</div>'
    )


def metric_card(label: str, value: str, chg: Optional[float] = None,
                chg_text: str = "") -> None:
    """直接渲染一个带涨跌色的指标卡"""
    st.markdown(html_metric(label, value, chg, chg_text), unsafe_allow_html=True)


# ==================== 表格渲染 ====================

def style_chg_column(df: pd.DataFrame, col: str) -> "pd.io.formats.style.Styler":
    """给涨跌列上色（红涨绿跌）"""
    def _c(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        return f"color: {chg_color(v)}; font-weight: 600;"

    return df.style.applymap(_c, subset=[col])

# -*- coding: utf-8 -*-
"""
Streamlit 可视化看板
选股神 AI - 交互式选股决策界面
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.data.fetcher import fetcher
from src.models.predictor import predictor
from src.features.engine import feature_engine

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* 基础 */
    * { font-family: 'Inter', -apple-system, sans-serif; }

    /* 🚨 核弹：所有自定义卡片子元素强制深色文字 */
    .kpi-card, .kpi-card *, .predict-card, .predict-card *,
    .signal-tag, .signal-tag *, .stat-mini, .stat-mini *,
    .chart-box, .chart-box h3 {
        color: #1e293b !important;
    }
    /* 辅助色标签覆盖 */
    .kpi-label, .predict-card .label, .stat-mini .label, .signal-tag .desc {
        color: #94a3b8 !important;
    }

    /* ----- 渐变大标题 ----- */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b !important;
        margin-bottom: 1.5rem;
    }

    /* ----- 指标卡片（概览顶栏）----- */
    .kpi-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .kpi-card {
        flex: 1;
        min-width: 160px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 12px rgba(102,126,234,0.12);
        transform: translateY(-1px);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b !important;
    }

    /* ----- AI 预测卡片（个股分析）----- */
    .predict-row {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin: 1.2rem 0;
    }
    .predict-card {
        flex: 1;
        min-width: 140px;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.15s;
        border: 1px solid #f1f5f9;
    }
    .predict-card:hover { transform: translateY(-2px); }
    .predict-card .label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .predict-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .predict-card.score  { background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); border-color: #e9d5ff; }
    .predict-card.up     { background: linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%); border-color: #fecaca; }
    .predict-card.neutral{ background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border-color: #fde68a; }
    .predict-card.down   { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-color: #bbf7d0; }
    .predict-card.signal { background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-color: #e2e8f0; }

    /* ----- 图表区容器 ----- */
    .chart-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.5rem 1rem 0.2rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .chart-box h3 {
        color: #1e293b !important;
        font-weight: 600;
        font-size: 1rem;
        margin: 0.6rem 0 0.3rem 0;
    }

    /* ----- 信号卡片 ----- */
    .signal-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.6rem 0;
    }
    .signal-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.45rem 0.8rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        color: #1e293b !important;
        border: 1px solid transparent;
        white-space: nowrap;
    }
    .signal-tag.bullish { background: #dcfce7; border-color: #bbf7d0; }
    .signal-tag.bearish { background: #fee2e2; border-color: #fecaca; }
    .signal-tag.warning { background: #fef3c7; border-color: #fde68a; }
    .signal-tag.info    { background: #dbeafe; border-color: #bfdbfe; }

    /* ----- 统计卡片 ----- */
    .stat-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin: 0.4rem 0;
    }
    .stat-mini {
        flex: 1;
        min-width: 90px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .stat-mini .label {
        font-size: 0.7rem;
        font-weight: 500;
        color: #94a3b8 !important;
        text-transform: uppercase;
    }
    .stat-mini .val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b !important;
        margin-top: 0.15rem;
    }

    /* ----- 节标题 ----- */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b !important;
        margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .section-title span {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ----- 侧边栏美化 ----- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }

    /* ----- 通用颜色类 ----- */
    .c-red    { color: #ef4444 !important; }
    .c-green  { color: #10b981 !important; }
    .c-amber  { color: #f59e0b !important; }
    .c-purple { color: #8b5cf6 !important; }
    .c-slate  { color: #64748b !important; }
    .c-dark   { color: #1e293b !important; }

    /* plotly 容器去边距 */
    .js-plotly-plot, .plot-container { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;">
        <span style="font-size:2.2rem;">📈</span>
        <span style="font-size:1.3rem; font-weight:700; color:#1e293b;">选股神 AI</span>
    </div>
    <p style="color:#94a3b8; font-size:0.78rem; margin-top:-0.3rem;">XGBoost + LightGBM 集成</p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "导航",
        ["🏠 概览", "🔍 AI选股", "📊 个股分析", "⚙️ 设置"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    ready = predictor.is_ready
    s_cls = "c-green" if ready else "c-red"
    dot_clr = "#10b981" if ready else "#ef4444"
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
        <span style="width:8px; height:8px; border-radius:50%; background:{dot_clr}; display:inline-block;"></span>
        <span style="font-size:0.82rem; color:#64748b;">模型: <b class="{s_cls}">{'就绪' if ready else '未训练'}</b></span>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ========== 🏠 概览页 ==========
if page == "🏠 概览":
    st.markdown('<p class="main-header">📈 选股神 AI Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI驱动的A股智能选股系统 — XGBoost × LightGBM 双模型集成</p>', unsafe_allow_html=True)

    # KPI卡片
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">📊 股票池</div>
            <div class="kpi-value">{len(fetcher.get_stock_list())} 只</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">🧠 模型状态</div>
            <div class="kpi-value">{'✅ 就绪' if predictor.is_ready else '❌ 未训练'}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">⚡ 集成引擎</div>
            <div class="kpi-value">XGB + LGBM</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">🎯 决策树数量</div>
            <div class="kpi-value">{predictor.xgb_model.n_estimators if predictor.xgb_model else '—'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 市场热点
    st.markdown('<div class="section-title"><span>🔥</span> 市场热点板块</div>', unsafe_allow_html=True)

    try:
        hotspots = fetcher.get_sector_hotspots()
        if len(hotspots) > 0:
            cols = st.columns(min(len(hotspots), 5))
            for i, (_, row) in enumerate(hotspots.head(5).iterrows()):
                with cols[i]:
                    pct = row.get("涨跌幅", 0)
                    c_cls = "c-red" if pct > 0 else "c-green"
                    st.markdown(f"""
                    <div style="text-align:center; padding:14px 8px; border-radius:12px;
                                background:linear-gradient(135deg, #ffffff, #f8fafc);
                                border:1px solid #e2e8f0; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                        <div style="font-weight:600; color:#1e293b; margin-bottom:6px;">{row.get('板块名称', '未知')[:6]}</div>
                        <div class="{c_cls}" style="font-size:1.3rem; font-weight:700;">{pct:+.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.info(f"热点数据获取中… ({e})")

    st.markdown("---")
    st.markdown("""
    <div style="background:linear-gradient(135deg, #faf5ff, #f3e8ff); border:1px solid #e9d5ff;
                border-radius:14px; padding:1.2rem 1.5rem; margin-top:1rem;">
        <h4 style="color:#7c3aed; margin-bottom:0.5rem;">🚀 快速开始</h4>
        <p style="color:#64748b; margin:0; font-size:0.9rem;">
            ① 训练模型 → ② <b>AI选股</b>获取推荐 → ③ <b>个股分析</b>深度研判
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========== 🔍 AI选股页 ==========
elif page == "🔍 AI选股":
    st.markdown('<p class="main-header">🤖 AI 智能选股</p>', unsafe_allow_html=True)

    if not predictor.is_ready:
        st.warning("⚠️ 模型尚未训练！请先运行: `python scripts/train.py`")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            pool = st.selectbox("股票池", ["hs300", "zz500", "all"], index=0)
        with col2:
            top_n = st.slider("推荐数量", 5, 50, 20, 5)
        with col3:
            min_score = st.slider("最低评分", 0.0, 100.0, 60.0, 5.0)

        if st.button("🚀 开始 AI 选股", type="primary", use_container_width=True):
            with st.spinner(f"AI 正在扫描 {pool} 股票池…"):
                codes = fetcher.get_stock_pool(pool)[:50]
                stock_data = {}
                stock_names = {}

                progress_bar = st.progress(0, "正在获取行情数据…")
                for i, code in enumerate(codes):
                    df = fetcher.get_daily_kline(code)
                    if len(df) >= 60:
                        stock_data[code] = df
                        stock_names[code] = code
                    progress_bar.progress((i + 1) / len(codes))

                progress_bar.progress(1.0, "AI模型预测中…")
                result_df = predictor.get_top_recommendations(
                    stock_data, stock_names, top_n=top_n, min_score=min_score
                )
                progress_bar.empty()

                st.success(f"✅ 分析完成! 从 {len(stock_data)} 只股票中筛选出 {len(result_df)} 只推荐")

                col1, col2 = st.columns([1, 2])
                with col1:
                    buy_count = len(result_df[result_df["评分"] >= 80])
                    hold_count = len(result_df[(result_df["评分"] >= 60) & (result_df["评分"] < 80)])
                    fig = go.Figure(data=[go.Pie(
                        labels=["强力推荐(≥80)", "一般推荐(60-80)"],
                        values=[buy_count, hold_count],
                        marker_colors=["#ef4444", "#f59e0b"],
                        hole=0.5,
                        textinfo="label+value"
                    )])
                    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                      template="plotly_white", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    top_df = result_df.head(15)
                    fig = go.Figure(data=[go.Bar(
                        x=top_df["评分"], y=top_df["名称"], orientation="h",
                        marker=dict(
                            color=top_df["评分"],
                            colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                            showscale=True,
                            colorbar=dict(title="评分", thickness=12)
                        ),
                        text=top_df["评分"].round(1), textposition="outside",
                        textfont=dict(color="#1e293b")
                    )])
                    fig.update_layout(height=400, template="plotly_white",
                                      xaxis_title="AI评分", xaxis=dict(range=[0, 100]),
                                      yaxis=dict(autorange="reversed"), margin=dict(l=10, r=30, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="section-title"><span>📋</span> 推荐详情</div>', unsafe_allow_html=True)
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "评分": st.column_config.ProgressColumn("评分", min_value=0, max_value=100, format="%.1f"),
                        "上涨概率": st.column_config.NumberColumn("上涨概率", format="%.1f%%"),
                    }
                )

# ========== 📊 个股分析页 ==========
elif page == "📊 个股分析":
    st.markdown('<p class="main-header">📊 个股深度分析</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        stock_code = st.text_input("股票代码", "000001", placeholder="6位代码")
    with col2:
        days = st.slider("回看天数", 60, 500, 250, 30)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 开始分析", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner(f"正在深度分析 {stock_code}…"):
            df = fetcher.get_daily_kline(stock_code, start_date=(datetime.now() - timedelta(days=days)).strftime("%Y%m%d"))

            if len(df) == 0:
                st.error(f"未找到 {stock_code} 的数据")
                st.stop()

            close = df["close"]
            df_plot = df.tail(180).copy()

            # ====== AI 预测 ======
            if predictor.is_ready:
                pred = predictor.predict_single(df, stock_code, stock_code)
                score = pred.score
                s_cls = "c-red" if score >= 80 else ("c-amber" if score >= 60 else "c-green")
                xgb_sig = "📈 买入" if pred.xgb_signal == 1 else ("📉 卖出" if pred.xgb_signal == -1 else "➡️ 持有")
                lgb_sig = "📈 买入" if pred.lgb_signal == 1 else ("📉 卖出" if pred.lgb_signal == -1 else "➡️ 持有")

                st.markdown(f"""
                <div class="predict-row">
                    <div class="predict-card score">
                        <div class="label">🤖 AI 综合评分</div>
                        <div class="value {s_cls}">{score:.0f}</div>
                    </div>
                    <div class="predict-card up">
                        <div class="label">📈 上涨概率</div>
                        <div class="value c-red">{pred.prob_up*100:.1f}%</div>
                    </div>
                    <div class="predict-card neutral">
                        <div class="label">➡️ 震荡概率</div>
                        <div class="value c-amber">{pred.prob_neutral*100:.1f}%</div>
                    </div>
                    <div class="predict-card down">
                        <div class="label">📉 下跌概率</div>
                        <div class="value c-green">{pred.prob_down*100:.1f}%</div>
                    </div>
                    <div class="predict-card signal">
                        <div class="label">🔮 模型信号</div>
                        <div class="value" style="font-size:0.9rem; line-height:1.5;">
                            <span class="c-red">XGB</span> {xgb_sig}<br>
                            <span class="c-purple">LGB</span> {lgb_sig}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ====== K线 + 布林带 ======
            st.markdown('<div class="chart-box"><h3>📈 K线图 · 布林带(20,2) · MA(5,20,60)</h3>', unsafe_allow_html=True)

            fig1 = go.Figure()
            fig1.add_trace(go.Candlestick(
                x=df_plot["date"], open=df_plot["open"], high=df_plot["high"],
                low=df_plot["low"], close=df_plot["close"],
                name="K线", increasing_line_color="#ef4444", decreasing_line_color="#10b981",
                increasing_fillcolor="#ef4444", decreasing_fillcolor="#10b981"
            ))
            for p, color in [(5, "#3b82f6"), (20, "#f59e0b"), (60, "#8b5cf6")]:
                ma = df_plot["close"].rolling(p).mean()
                fig1.add_trace(go.Scatter(x=df_plot["date"], y=ma, name=f"MA{p}",
                                          line=dict(color=color, width=1.8)))
            boll_mid = close.rolling(20).mean()
            boll_std = close.rolling(20).std()
            boll_upper = boll_mid + 2 * boll_std
            boll_lower = boll_mid - 2 * boll_std
            boll_fill = pd.DataFrame({
                "x": list(df_plot["date"]) + list(df_plot["date"])[::-1],
                "y": list(boll_upper.iloc[-180:]) + list(boll_lower.iloc[-180:])[::-1]
            })
            fig1.add_trace(go.Scatter(x=boll_fill["x"], y=boll_fill["y"], fill="toself",
                                      fillcolor="rgba(100,116,139,0.08)", line=dict(width=0),
                                      name="BOLL通道", showlegend=True))
            fig1.add_trace(go.Scatter(x=df_plot["date"], y=boll_mid.iloc[-180:],
                                      name="BOLL中轨", line=dict(color="#94a3b8", width=1.2, dash="dash")))
            fig1.update_layout(height=480, template="plotly_white", hovermode="x unified",
                               xaxis_rangeslider_visible=True, legend=dict(orientation="h", y=1.12),
                               margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ====== 成交量 + MACD ======
            st.markdown('<div class="chart-box"><h3>📊 成交量 + MACD(12,26,9)</h3>', unsafe_allow_html=True)

            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                 row_heights=[0.3, 0.7])
            vol_colors = ["#ef4444" if c >= o else "#10b981" for c, o in zip(df_plot["close"], df_plot["open"])]
            fig2.add_trace(go.Bar(x=df_plot["date"], y=df_plot["volume"], name="成交量",
                                  marker_color=vol_colors, opacity=0.7), row=1, col=1)
            fig2.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["volume"].rolling(5).mean(),
                                      name="量MA5", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)

            ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
            dif, dea = ema12 - ema26, (ema12 - ema26).ewm(span=9).mean()
            macd_bar = 2 * (dif - dea)
            fig2.add_trace(go.Scatter(x=df_plot["date"], y=dif.iloc[-180:], name="DIF",
                                      line=dict(color="#ef4444", width=1.8)), row=2, col=1)
            fig2.add_trace(go.Scatter(x=df_plot["date"], y=dea.iloc[-180:], name="DEA",
                                      line=dict(color="#3b82f6", width=1.8)), row=2, col=1)
            macd_colors = ["#ef4444" if v >= 0 else "#10b981" for v in macd_bar.iloc[-180:]]
            fig2.add_trace(go.Bar(x=df_plot["date"], y=macd_bar.iloc[-180:], name="柱",
                                  marker_color=macd_colors, opacity=0.5), row=2, col=1)
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
                delta = close.diff()
                gain = delta.clip(lower=0).ewm(span=p, adjust=False).mean()
                loss = (-delta).clip(lower=0).ewm(span=p, adjust=False).mean()
                rsi = 100 - 100 / (1 + gain / (loss + 1e-10))
                fig3.add_trace(go.Scatter(x=df_plot["date"], y=rsi.iloc[-180:], name=f"RSI{p}",
                                          line=dict(color=color, width=1.8)), row=1, col=1)
            for y, lbl in [(70, "超买"), (50, "中轴"), (30, "超卖")]:
                fig3.add_hline(y=y, line_dash="dash", line_color="#94a3b8", opacity=0.4,
                               annotation_text=lbl, annotation_position="right", row=1, col=1)

            low_n = df["low"].rolling(9, min_periods=1).min()
            high_n = df["high"].rolling(9, min_periods=1).max()
            rsv = (close - low_n) / (high_n - low_n + 1e-10) * 100
            k = rsv.ewm(span=3, adjust=False).mean()
            d = k.ewm(span=3, adjust=False).mean()
            j = 3 * k - 2 * d
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

            # ====== 技术信号 ======
            st.markdown('<div class="section-title"><span>🧠</span> 技术信号总结</div>', unsafe_allow_html=True)

            last_close = float(close.iloc[-1])
            ma5_now = float(close.rolling(5).mean().iloc[-1])
            ma20_now = float(close.rolling(20).mean().iloc[-1])
            ma60_now = float(close.rolling(60).mean().iloc[-1])
            rsi14_now = float(100 - 100 / (1 + (
                close.diff().clip(lower=0).ewm(span=14, adjust=False).mean().iloc[-1]
            ) / ((-close.diff()).clip(lower=0).ewm(span=14, adjust=False).mean().iloc[-1] + 1e-10)))
            k_now, d_now, j_now = float(k.iloc[-1]), float(d.iloc[-1]), float(j.iloc[-1])
            boll_pos = float((last_close - boll_lower.iloc[-1]) / (boll_upper.iloc[-1] - boll_lower.iloc[-1] + 1e-10))
            vol_ratio = float(df_plot["volume"].iloc[-1] / (df_plot["volume"].rolling(5).mean().iloc[-1] + 1e-10))

            # 生成所有信号
            signals = []

            if last_close > ma5_now > ma20_now > ma60_now:
                signals.append(("bullish", "📈 多头排列", "MA5>MA20>MA60 强势上升"))
            elif last_close < ma5_now < ma20_now < ma60_now:
                signals.append(("bearish", "📉 空头排列", "均线全面下行"))
            else:
                signals.append(("warning", "⚠️ 均线交织", "方向不明，震荡整理"))

            ma5_prev = float(close.rolling(5).mean().iloc[-2])
            ma20_prev = float(close.rolling(20).mean().iloc[-2])
            if ma5_now > ma20_now and ma5_prev <= ma20_prev:
                signals.append(("bullish", "🔺 MA金叉", "MA5上穿MA20"))
            elif ma5_now < ma20_now and ma5_prev >= ma20_prev:
                signals.append(("bearish", "🔻 MA死叉", "MA5下穿MA20"))

            if rsi14_now > 70:
                signals.append(("warning", f"⚠️ RSI超买({rsi14_now:.0f})", "注意回调"))
            elif rsi14_now < 30:
                signals.append(("bullish", f"💡 RSI超卖({rsi14_now:.0f})", "可能反弹"))
            else:
                signals.append(("info", f"✅ RSI正常({rsi14_now:.0f})", "合理区间"))

            if k_now > d_now and k_now < 80:
                signals.append(("bullish", "🔺 KDJ金叉", f"K({k_now:.0f})>D({d_now:.0f})"))
            elif k_now < d_now and k_now > 20:
                signals.append(("bearish", "🔻 KDJ死叉", f"K({k_now:.0f})<D({d_now:.0f})"))
            elif j_now < 0:
                signals.append(("bullish", "💡 KDJ超卖", "J<0 极度超卖"))
            elif j_now > 100:
                signals.append(("warning", "⚠️ KDJ超买", "J>100 极度超买"))

            if boll_pos > 0.95:
                signals.append(("warning", "⚠️ 触及BOLL上轨", "压力位"))
            elif boll_pos < 0.05:
                signals.append(("bullish", "💡 触及BOLL下轨", "支撑位"))
            else:
                signals.append(("info", f"✅ BOLL通道({boll_pos*100:.0f}%)", "正常运行"))

            if vol_ratio > 2:
                signals.append(("info", f"🔥 大幅放量", f"量比 {vol_ratio:.1f}x"))
            elif vol_ratio < 0.5:
                signals.append(("warning", "💤 极度缩量", "交投冷清"))

            st.markdown('<div class="signal-row">', unsafe_allow_html=True)
            for tag_cls, title, desc in signals:
                st.markdown(f'<span class="signal-tag {tag_cls}"><b>{title}</b> {desc}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ====== 多周期统计 ======
            st.markdown('<div class="section-title"><span>📊</span> 多周期统计</div>', unsafe_allow_html=True)

            for label, period_data in [("近 5 日", df.tail(5)), ("近 20 日", df.tail(20)), ("近 60 日", df.tail(60))]:
                c = period_data["close"]
                ret = (c.iloc[-1] / c.iloc[0] - 1) * 100
                high_p = c.max()
                low_p = c.min()
                amp = (period_data["high"].max() - period_data["low"].min()) / c.iloc[0] * 100
                vol_price = c.pct_change().std() * 100
                rc_cls = "c-red" if ret > 0 else "c-green"

                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-mini"><div class="label">周期</div><div class="val c-purple">{label}</div></div>
                    <div class="stat-mini"><div class="label">涨跌幅</div><div class="val {rc_cls}">{ret:+.2f}%</div></div>
                    <div class="stat-mini"><div class="label">最高</div><div class="val c-red">{high_p:.2f}</div></div>
                    <div class="stat-mini"><div class="label">最低</div><div class="val c-green">{low_p:.2f}</div></div>
                    <div class="stat-mini"><div class="label">波动率</div><div class="val">{vol_price:.2f}%</div></div>
                    <div class="stat-mini"><div class="label">振幅</div><div class="val">{amp:.2f}%</div></div>
                </div>
                """, unsafe_allow_html=True)

            # ====== AI特征贡献 ======
            if predictor.is_ready and pred.features_snapshot:
                st.markdown('<div class="section-title"><span>🧬</span> AI 特征快照 Top 15</div>', unsafe_allow_html=True)

                feat = pred.features_snapshot
                meaningful = {k: v for k, v in feat.items() if abs(v) > 1e-6}
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
                                   xaxis_title="特征值", xaxis=dict(zeroline=True, zerolinecolor="#94a3b8"),
                                   margin=dict(l=10, r=50, t=10, b=10))
                st.plotly_chart(fig4, use_container_width=True)

# ========== ⚙️ 设置页 ==========
elif page == "⚙️ 设置":
    st.markdown('<p class="main-header">⚙️ 系统设置</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><span>🧠</span> 模型配置</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:1rem;">
            <h4 style="color:#ef4444; margin:0 0 0.5rem 0;">XGBoost</h4>
        </div>
        """, unsafe_allow_html=True)
        st.json({"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8})

    with col2:
        st.markdown("""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:1rem;">
            <h4 style="color:#8b5cf6; margin:0 0 0.5rem 0;">LightGBM</h4>
        </div>
        """, unsafe_allow_html=True)
        st.json({"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05, "subsample": 0.8})

    st.markdown('<div class="section-title"><span>📡</span> 数据 & 缓存</div>', unsafe_allow_html=True)
    st.info("🔌 数据源: **AKShare** (新浪财经) · 免费沪深A股行情")

    if st.button("🗑️ 清除数据缓存", use_container_width=True):
        fetcher.clear_cache()
        st.success("✅ 缓存已清除! 下次获取将重新下载数据。")
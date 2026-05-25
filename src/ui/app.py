# -*- coding: utf-8 -*-
"""
Streamlit 可视化看板
选股神 AI - 交互式选股决策界面
"""

import sys
from pathlib import Path

# 添加项目根目录
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

# ========== 样式 ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .stock-up { color: #ff4b4b; font-weight: bold; }
    .stock-down { color: #00b894; font-weight: bold; }
    .score-high { color: #ff4b4b; font-size: 1.5rem; font-weight: bold; }
    .score-mid { color: #fdcb6e; font-size: 1.5rem; font-weight: bold; }
    .score-low { color: #00b894; font-size: 1.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ========== 侧边栏 ==========
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=60)
    st.markdown("## 📈 选股神 AI")
    st.markdown("---")
    
    page = st.radio(
        "导航",
        ["🏠 概览", "🔍 AI选股", "📊 个股分析", "⚙️ 设置"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption(f"模型状态: {'✅ 就绪' if predictor.is_ready else '❌ 未训练'}")
    st.caption(f"更新时间: {datetime.now().strftime('%H:%M:%S')}")

# ========== 🏠 概览页 ==========
if page == "🏠 概览":
    st.markdown('<p class="main-header">📈 选股神 AI Dashboard</p>', unsafe_allow_html=True)
    st.markdown("*AI驱动的A股智能选股系统 — XGBoost + LightGBM + LSTM 深度学习*")
    
    # 状态卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("股票池", f"{len(fetcher.get_stock_list())}" + " 只")
    with col2:
        st.metric("模型状态", "✅ 就绪" if predictor.is_ready else "❌ 未训练")
    with col3:
        st.metric("集成模型", "XGBoost + LightGBM")
    with col4:
        st.metric("预测周期", f"{predictor.xgb_model.n_estimators if predictor.xgb_model else 'N/A'} tress")
    
    # 市场热点
    st.markdown("---")
    st.subheader("🔥 市场热点板块")
    
    try:
        hotspots = fetcher.get_sector_hotspots()
        if len(hotspots) > 0:
            cols = st.columns(5)
            for i, (_, row) in enumerate(hotspots.head(5).iterrows()):
                with cols[i]:
                    pct = row.get("涨跌幅", 0)
                    color = "#ff4b4b" if pct > 0 else "#00b894"
                    st.markdown(f"""
                    <div style="text-align:center; padding:10px; border-radius:8px; background:#f8f9fa;">
                        <strong>{row.get('板块名称', '未知')[:6]}</strong><br>
                        <span style="color:{color}; font-size:1.2rem;">{pct:+.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"热点数据获取失败: {e}")
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 快速开始
    
    1. **训练模型**: 运行 `python scripts/train.py`
    2. **AI选股**: 点击左侧「AI选股」查看推荐
    3. **个股分析**: 点击「个股分析」深入查看
    """)

# ========== 🔍 AI选股页 ==========
elif page == "🔍 AI选股":
    st.markdown('<p class="main-header">🤖 AI 智能选股</p>', unsafe_allow_html=True)
    
    if not predictor.is_ready:
        st.error("⚠️ 模型尚未训练！请先运行: `python scripts/train.py`")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            pool = st.selectbox("股票池", ["hs300", "zz500", "all"], index=0)
        with col2:
            top_n = st.slider("推荐数量", 5, 50, 20, 5)
        with col3:
            min_score = st.slider("最低评分", 0.0, 100.0, 60.0, 5.0)
        
        if st.button("🚀 开始AI选股", type="primary", use_container_width=True):
            with st.spinner(f"AI正在扫描{pool}股票池..."):
                codes = fetcher.get_stock_pool(pool)[:50]  # 演示限制
                stock_data = {}
                stock_names = {}
                
                progress_bar = st.progress(0, "正在获取行情数据...")
                for i, code in enumerate(codes):
                    df = fetcher.get_daily_kline(code)
                    if len(df) >= 60:
                        stock_data[code] = df
                        stock_names[code] = code
                    progress_bar.progress((i + 1) / len(codes))
                
                progress_bar.progress(1.0, "AI模型预测中...")
                result_df = predictor.get_top_recommendations(
                    stock_data, stock_names, top_n=top_n, min_score=min_score
                )
                progress_bar.empty()
                
                # 展示结果
                st.success(f"✅ 分析完成! 从 {len(stock_data)} 只股票中筛选出 {len(result_df)} 只推荐")
                
                # 评分分布
                col1, col2 = st.columns([1, 2])
                with col1:
                    # 信号分布图
                    buy_count = len(result_df[result_df["评分"] >= 80])
                    hold_count = len(result_df[(result_df["评分"] >= 60) & (result_df["评分"] < 80)])
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=["强力推荐(≥80)", "一般推荐(60-80)"],
                            values=[buy_count, hold_count],
                            marker_colors=["#ff4b4b", "#fdcb6e"],
                            hole=0.4
                        )
                    ])
                    fig.update_layout(
                        title="推荐分布",
                        height=250,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 评分排行榜
                    fig = go.Figure(data=[
                        go.Bar(
                            x=result_df["评分"].head(15),
                            y=result_df["名称"].head(15),
                            orientation="h",
                            marker=dict(
                                color=result_df["评分"].head(15),
                                colorscale="RdYlGn",
                                showscale=True
                            ),
                            text=result_df["评分"].head(15).round(1),
                            textposition="outside",
                        )
                    ])
                    fig.update_layout(
                        title="TOP 15 评分排行",
                        height=400,
                        xaxis_title="AI评分",
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 详细表格
                st.markdown("### 📋 推荐详情")
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "评分": st.column_config.ProgressColumn(
                            "评分", min_value=0, max_value=100, format="%.1f"
                        ),
                        "上涨概率": st.column_config.NumberColumn("上涨概率", format="%.1f%%"),
                    }
                )

# ========== 📊 个股分析页 ==========
elif page == "📊 个股分析":
    st.markdown('<p class="main-header">📊 个股深度分析</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        stock_code = st.text_input("股票代码", "000001", placeholder="请输入6位代码")
    
    with col2:
        days = st.slider("分析周期（天）", 60, 500, 250, 30)
    
    if st.button("🔍 分析", type="primary"):
        with st.spinner(f"分析 {stock_code}..."):
            df = fetcher.get_daily_kline(stock_code, start_date=(datetime.now() - timedelta(days=days)).strftime("%Y%m%d"))
            
            if len(df) == 0:
                st.error(f"未找到 {stock_code} 的数据")
            else:
                # AI预测结果
                if predictor.is_ready:
                    pred = predictor.predict_single(df, stock_code, stock_code)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        score_color = "score-high" if pred.score >= 80 else ("score-mid" if pred.score >= 60 else "score-low")
                        st.markdown(f'<div class="metric-card"><p>AI综合评分</p><p class="{score_color}">{pred.score:.0f}</p></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-card"><p>上涨概率</p><p class="stock-up">{pred.prob_up*100:.1f}%</p></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="metric-card"><p>XGBoost信号</p><p>{"📈买入" if pred.xgb_signal==1 else ("📉卖出" if pred.xgb_signal==-1 else "➡️持有")}</p></div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown(f'<div class="metric-card"><p>LightGBM信号</p><p>{"📈买入" if pred.lgb_signal==1 else ("📉卖出" if pred.lgb_signal==-1 else "➡️持有")}</p></div>', unsafe_allow_html=True)
                
                # K线图
                st.markdown("### 📈 K线图 + 技术指标")
                
                # 计算技术指标
                df_plot = df.tail(120).copy()
                
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.5, 0.25, 0.25],
                    subplot_titles=("K线 + MA", "成交量", "MACD")
                )
                
                # K线
                fig.add_trace(
                    go.Candlestick(
                        x=df_plot["date"],
                        open=df_plot["open"],
                        high=df_plot["high"],
                        low=df_plot["low"],
                        close=df_plot["close"],
                        name="K线"
                    ),
                    row=1, col=1
                )
                
                # 均线
                for p, color in [(5, "blue"), (20, "orange"), (60, "purple")]:
                    ma = df_plot["close"].rolling(p).mean()
                    fig.add_trace(
                        go.Scatter(x=df_plot["date"], y=ma, name=f"MA{p}", line=dict(color=color, width=1)),
                        row=1, col=1
                    )
                
                # 成交量
                colors = ["#ff4b4b" if close >= open_ else "#00b894" for close, open_ in zip(df_plot["close"], df_plot["open"])]
                fig.add_trace(
                    go.Bar(x=df_plot["date"], y=df_plot["volume"], name="成交量", marker_color=colors),
                    row=2, col=1
                )
                
                # MACD
                ema12 = df_plot["close"].ewm(span=12).mean()
                ema26 = df_plot["close"].ewm(span=26).mean()
                dif = ema12 - ema26
                dea = dif.ewm(span=9).mean()
                macd_bar = 2 * (dif - dea)
                
                fig.add_trace(go.Scatter(x=df_plot["date"], y=dif, name="DIF", line=dict(color="white", width=1)), row=3, col=1)
                fig.add_trace(go.Scatter(x=df_plot["date"], y=dea, name="DEA", line=dict(color="yellow", width=1)), row=3, col=1)
                
                bar_colors = ["#ff4b4b" if v >= 0 else "#00b894" for v in macd_bar]
                fig.add_trace(go.Bar(x=df_plot["date"], y=macd_bar, name="MACD柱", marker_color=bar_colors), row=3, col=1)
                
                fig.update_layout(
                    height=700,
                    showlegend=False,
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 统计信息
                st.markdown("### 📊 近期统计")
                recent = df.tail(20)
                cols = st.columns(5)
                cols[0].metric("最新收盘价", f"{recent['close'].iloc[-1]:.2f}")
                cols[1].metric("5日涨幅", f"{recent['close'].pct_change(5).iloc[-1]*100:.2f}%")
                cols[2].metric("20日波动率", f"{recent['close'].pct_change().std()*100:.2f}%")
                cols[3].metric("20日最高", f"{recent['high'].max():.2f}")
                cols[4].metric("20日最低", f"{recent['low'].min():.2f}")

# ========== ⚙️ 设置页 ==========
elif page == "⚙️ 设置":
    st.markdown('<p class="main-header">⚙️ 系统设置</p>', unsafe_allow_html=True)
    
    st.markdown("### 模型配置")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### XGBoost参数")
        st.json({
            "n_estimators": 300,
            "max深度": 6,
            "学习率": 0.05,
            "子采样": 0.8,
        })
    
    with col2:
        st.markdown("#### LightGBM参数")
        st.json({
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
        })
    
    st.markdown("### 数据源")
    st.info("当前使用: **AKShare** (免费A股数据源)")
    
    st.markdown("### 缓存管理")
    if st.button("🗑️ 清除数据缓存"):
        fetcher.clear_cache()
        st.success("缓存已清除!")
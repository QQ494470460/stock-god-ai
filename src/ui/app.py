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
                st.stop()
            
            close = df["close"]
            recent = df.tail(60)
            df_plot = df.tail(180).copy()
            
            # ====== AI预测 ======
            if predictor.is_ready:
                pred = predictor.predict_single(df, stock_code, stock_code)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                score_color = "score-high" if pred.score >= 80 else ("score-mid" if pred.score >= 60 else "score-low")
                col1.markdown(f'<div class="metric-card"><p>🤖 AI评分</p><p class="{score_color}">{pred.score:.0f}</p></div>', unsafe_allow_html=True)
                col2.markdown(f'<div class="metric-card"><p>📈 上涨</p><p class="stock-up">{pred.prob_up*100:.1f}%</p></div>', unsafe_allow_html=True)
                col3.markdown(f'<div class="metric-card"><p>➡️ 震荡</p><p style="color:#fdcb6e; font-size:1.3rem; font-weight:bold;">{pred.prob_neutral*100:.1f}%</p></div>', unsafe_allow_html=True)
                col4.markdown(f'<div class="metric-card"><p>📉 下跌</p><p class="stock-down">{pred.prob_down*100:.1f}%</p></div>', unsafe_allow_html=True)
                xgb_sig = "📈 买入" if pred.xgb_signal == 1 else ("📉 卖出" if pred.xgb_signal == -1 else "➡️ 持有")
                lgb_sig = "📈 买入" if pred.lgb_signal == 1 else ("📉 卖出" if pred.lgb_signal == -1 else "➡️ 持有")
                col5.markdown(f'<div class="metric-card"><p>🔮 信号</p><p style="font-size:1rem;">XGB: {xgb_sig}<br>LGB: {lgb_sig}</p></div>', unsafe_allow_html=True)
            
            # ====== K线 + MA + 布林带 ======
            st.markdown("### 📈 K线图 + 布林带 + 均线")
            
            fig1 = make_subplots(
                rows=1, cols=1,
                subplot_titles=("K线 + MA(5,20,60) + 布林带(20,2)",)
            )
            
            # K线
            fig1.add_trace(
                go.Candlestick(
                    x=df_plot["date"], open=df_plot["open"], high=df_plot["high"],
                    low=df_plot["low"], close=df_plot["close"],
                    name="K线", increasing_line_color="#ef5350", decreasing_line_color="#26a69a"
                ), row=1, col=1
            )
            # 均线
            for p, color in [(5, "#42a5f5"), (20, "#ffa726"), (60, "#ab47bc")]:
                ma = df_plot["close"].rolling(p).mean()
                fig1.add_trace(go.Scatter(x=df_plot["date"], y=ma, name=f"MA{p}", line=dict(color=color, width=1.5)), row=1, col=1)
            # 布林带
            boll_mid = close.rolling(20).mean()
            boll_std = close.rolling(20).std()
            boll_upper = boll_mid + 2 * boll_std
            boll_lower = boll_mid - 2 * boll_std
            fig1.add_trace(go.Scatter(x=df_plot["date"], y=boll_upper.iloc[-180:], name="BOLL上轨", line=dict(color="#9e9e9e", width=1, dash="dash")), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df_plot["date"], y=boll_mid.iloc[-180:], name="BOLL中轨", line=dict(color="#9e9e9e", width=1)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df_plot["date"], y=boll_lower.iloc[-180:], name="BOLL下轨", line=dict(color="#9e9e9e", width=1, dash="dash")), row=1, col=1)
            fig1.update_layout(height=450, showlegend=True, xaxis_rangeslider_visible=True, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig1, use_container_width=True)
            
            # ====== 成交量 + MACD ======
            st.markdown("### 📊 成交量 + MACD")
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.35, 0.65],
                                subplot_titles=("成交量 + OBV", "MACD (12,26,9)"))
            
            # 成交量（红涨绿跌）
            vol_colors = ["#ef5350" if c >= o else "#26a69a" for c, o in zip(df_plot["close"], df_plot["open"])]
            fig2.add_trace(go.Bar(x=df_plot["date"], y=df_plot["volume"], name="成交量", marker_color=vol_colors, opacity=0.7), row=1, col=1)
            # 5日均量
            vol_ma5 = df_plot["volume"].rolling(5).mean()
            fig2.add_trace(go.Scatter(x=df_plot["date"], y=vol_ma5, name="量MA5", line=dict(color="#ffa726", width=1)), row=1, col=1)
            
            # MACD
            ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
            dif, dea = ema12 - ema26, (ema12 - ema26).ewm(span=9).mean()
            macd_bar = 2 * (dif - dea)
            fig2.add_trace(go.Scatter(x=df_plot["date"], y=dif.iloc[-180:], name="DIF", line=dict(color="#ef5350", width=1.5)), row=2, col=1)
            fig2.add_trace(go.Scatter(x=df_plot["date"], y=dea.iloc[-180:], name="DEA", line=dict(color="#42a5f5", width=1.5)), row=2, col=1)
            bar_colors = ["#ef5350" if v >= 0 else "#26a69a" for v in macd_bar.iloc[-180:]]
            fig2.add_trace(go.Bar(x=df_plot["date"], y=macd_bar.iloc[-180:], name="MACD柱", marker_color=bar_colors, opacity=0.6), row=2, col=1)
            fig2.update_layout(height=400, showlegend=True, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)
            
            # ====== RSI + KDJ ======
            st.markdown("### 🎯 RSI + KDJ 指标")
            fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.5],
                                subplot_titles=("RSI (6, 14)", "KDJ (9,3,3)"))
            
            # RSI
            for p, color in [(6, "#ef5350"), (14, "#42a5f5")]:
                delta = close.diff()
                gain = delta.clip(lower=0).ewm(span=p, adjust=False).mean()
                loss = (-delta).clip(lower=0).ewm(span=p, adjust=False).mean()
                rsi = 100 - 100 / (1 + gain / (loss + 1e-10))
                fig3.add_trace(go.Scatter(x=df_plot["date"], y=rsi.iloc[-180:], name=f"RSI{p}", line=dict(color=color, width=1.5)), row=1, col=1)
            fig3.add_hline(y=70, line_dash="dash", line_color="#ef5350", opacity=0.5, row=1, col=1)
            fig3.add_hline(y=30, line_dash="dash", line_color="#26a69a", opacity=0.5, row=1, col=1)
            fig3.add_hline(y=50, line_dash="dot", line_color="#9e9e9e", opacity=0.5, row=1, col=1)
            
            # KDJ
            low_n = df["low"].rolling(9, min_periods=1).min()
            high_n = df["high"].rolling(9, min_periods=1).max()
            rsv = (close - low_n) / (high_n - low_n + 1e-10) * 100
            k = rsv.ewm(span=3, adjust=False).mean()
            d = k.ewm(span=3, adjust=False).mean()
            j = 3 * k - 2 * d
            fig3.add_trace(go.Scatter(x=df_plot["date"], y=k.iloc[-180:], name="K值", line=dict(color="#ef5350", width=1.5)), row=2, col=1)
            fig3.add_trace(go.Scatter(x=df_plot["date"], y=d.iloc[-180:], name="D值", line=dict(color="#42a5f5", width=1.5)), row=2, col=1)
            fig3.add_trace(go.Scatter(x=df_plot["date"], y=j.iloc[-180:], name="J值", line=dict(color="#ab47bc", width=1, dash="dot")), row=2, col=1)
            fig3.add_hline(y=80, line_dash="dash", line_color="#ef5350", opacity=0.5, row=2, col=1)
            fig3.add_hline(y=20, line_dash="dash", line_color="#26a69a", opacity=0.5, row=2, col=1)
            fig3.update_layout(height=400, showlegend=True, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig3, use_container_width=True)
            
            # ====== 技术信号总结 ======
            st.markdown("### 🧠 技术信号总结")
            
            # 计算关键信号
            last_close = float(close.iloc[-1])
            ma5_now = float(close.rolling(5).mean().iloc[-1])
            ma20_now = float(close.rolling(20).mean().iloc[-1])
            ma60_now = float(close.rolling(60).mean().iloc[-1])
            rsi14_now = float(100 - 100 / (1 + (close.diff().clip(lower=0).ewm(span=14, adjust=False).mean().iloc[-1]) / 
                                          ((-close.diff()).clip(lower=0).ewm(span=14, adjust=False).mean().iloc[-1] + 1e-10)))
            k_now = float(k.iloc[-1])
            d_now = float(d.iloc[-1])
            j_now = float(j.iloc[-1])
            boll_pos = float((last_close - boll_lower.iloc[-1]) / (boll_upper.iloc[-1] - boll_lower.iloc[-1] + 1e-10))
            vol_ratio = float(df_plot["volume"].iloc[-1] / (df_plot["volume"].rolling(5).mean().iloc[-1] + 1e-10))
            
            signals = []
            # 均线信号
            if last_close > ma5_now > ma20_now > ma60_now:
                signals.append(("✅", "均线多头排列", "MA5>MA20>MA60，强势上涨趋势"))
            elif last_close < ma5_now < ma20_now < ma60_now:
                signals.append(("❌", "均线空头排列", "MA5<MA20<MA60，下跌趋势中"))
            else:
                signals.append(("⚠️", "均线交织", "短期均线方向不明，震荡整理"))
            
            # MA金叉/死叉
            ma5_prev = float(close.rolling(5).mean().iloc[-2])
            ma20_prev = float(close.rolling(20).mean().iloc[-2])
            if ma5_now > ma20_now and ma5_prev <= ma20_prev:
                signals.append(("📈", "MA金叉", "MA5上穿MA20，短线看涨信号"))
            elif ma5_now < ma20_now and ma5_prev >= ma20_prev:
                signals.append(("📉", "MA死叉", "MA5下穿MA20，短线看跌信号"))
            
            # RSI信号
            if rsi14_now > 70:
                signals.append(("⚠️", f"RSI超买 ({rsi14_now:.0f})", "RSI(14)高于70，注意回调风险"))
            elif rsi14_now < 30:
                signals.append(("💡", f"RSI超卖 ({rsi14_now:.0f})", "RSI(14)低于30，可能超跌反弹"))
            else:
                signals.append(("✅", f"RSI正常 ({rsi14_now:.0f})", "RSI(14)在合理区间"))
            
            # KDJ信号
            if k_now > d_now and k_now < 80:
                signals.append(("📈", "KDJ金叉", f"K({k_now:.0f})>D({d_now:.0f})，短线看多"))
            elif k_now < d_now and k_now > 20:
                signals.append(("📉", "KDJ死叉", f"K({k_now:.0f})<D({d_now:.0f})，短线看空"))
            elif j_now < 0:
                signals.append(("💡", "KDJ超卖", "J值<0，极度超卖"))
            elif j_now > 100:
                signals.append(("⚠️", "KDJ超买", "J值>100，极度超买"))
            
            # 布林带
            if boll_pos > 0.95:
                signals.append(("⚠️", "触及BOLL上轨", "价格接近布林带上轨，注意压力"))
            elif boll_pos < 0.05:
                signals.append(("💡", "触及BOLL下轨", "价格接近布林带下轨，可能支撑"))
            else:
                signals.append(("✅", "BOLL通道内", f"价格在布林带中间运行 ({boll_pos*100:.0f}%)"))
            
            # 量能
            if vol_ratio > 2:
                signals.append(("🔥", "大幅放量", f"成交量是5日均量的{vol_ratio:.1f}倍"))
            elif vol_ratio < 0.5:
                signals.append(("💤", "极度缩量", "成交量低迷"))
            
            for icon, title, desc in signals:
                bg = "#e8f5e9" if "✅" in icon or "📈" in icon or "💡" in icon else ("#fff3e0" if "⚠️" in icon else "#ffebee")
                st.markdown(f'<div style="background:{bg}; padding:8px 12px; border-radius:6px; margin:4px 0;"><b>{icon} {title}</b>：{desc}</div>', unsafe_allow_html=True)
            
            # ====== 统计信息 ======
            st.markdown("### 📊 多周期统计")
            
            for label, period_data in [("近5日", df.tail(5)), ("近20日", df.tail(20)), ("近60日", df.tail(60))]:
                c = period_data["close"]
                ret = (c.iloc[-1] / c.iloc[0] - 1) * 100
                high_p = c.max()
                low_p = c.min()
                vol_avg = period_data["volume"].mean() if "volume" in period_data.columns else 0
                amp = (period_data["high"].max() - period_data["low"].min()) / c.iloc[0] * 100
                vol_price = c.pct_change().std() * 100
                ret_color = "#ef5350" if ret > 0 else "#26a69a"
                
                st.markdown(f'''
                <div style="display:flex; gap:16px; flex-wrap:wrap; margin:8px 0;">
                    <div style="flex:1; min-width:100px; background:#f5f5f5; border-radius:8px; padding:10px; text-align:center;">
                        <span style="color:#666; font-size:0.8rem;">{label}</span><br>
                        <span style="color:{ret_color}; font-size:1.2rem; font-weight:bold;">{ret:+.2f}%</span>
                    </div>
                    <div style="flex:1; min-width:100px; background:#f5f5f5; border-radius:8px; padding:10px; text-align:center;">
                        <span style="color:#666; font-size:0.8rem;">最高价</span><br>
                        <span style="font-size:1.2rem; font-weight:bold;">{high_p:.2f}</span>
                    </div>
                    <div style="flex:1; min-width:100px; background:#f5f5f5; border-radius:8px; padding:10px; text-align:center;">
                        <span style="color:#666; font-size:0.8rem;">最低价</span><br>
                        <span style="font-size:1.2rem; font-weight:bold;">{low_p:.2f}</span>
                    </div>
                    <div style="flex:1; min-width:100px; background:#f5f5f5; border-radius:8px; padding:10px; text-align:center;">
                        <span style="color:#666; font-size:0.8rem;">波动率</span><br>
                        <span style="font-size:1.2rem; font-weight:bold;">{vol_price:.2f}%</span>
                    </div>
                    <div style="flex:1; min-width:100px; background:#f5f5f5; border-radius:8px; padding:10px; text-align:center;">
                        <span style="color:#666; font-size:0.8rem;">振幅</span><br>
                        <span style="font-size:1.2rem; font-weight:bold;">{amp:.2f}%</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            # ====== AI 特征重要性 ======
            if predictor.is_ready and pred.features_snapshot:
                st.markdown("### 🧬 AI 特征快照 (Top 15)")
                feat = pred.features_snapshot
                # 只展示有意义的特征（排除fundamental全0的列）
                meaningful = {k: v for k, v in feat.items() if abs(v) > 1e-6}
                feat_df = pd.DataFrame(sorted(meaningful.items(), key=lambda x: abs(x[1]), reverse=True)[:15],
                                      columns=["特征", "当前值"])
                fig4 = go.Figure(data=[go.Bar(x=feat_df["当前值"], y=feat_df["特征"], orientation="h",
                                              marker=dict(color=["#ef5350" if v > 0 else "#26a69a" for v in feat_df["当前值"]],
                                                         colorscale=None))])
                fig4.update_layout(height=400, template="plotly_white", xaxis_title="特征值", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig4, use_container_width=True)

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
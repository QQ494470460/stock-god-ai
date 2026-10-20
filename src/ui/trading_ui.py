# -*- coding: utf-8 -*-
"""
选股神AI — 自动交易 Streamlit UI 组件 (v2)
改造要点：
  1. 独立的纸交易模拟引擎 — 零风险验证策略
  2. 实时交易日志面板 — 每次扫描、信号、下单都可见
  3. 信号流向修复 — watchlist 自动加载 + 缓存防重复取数
  4. 策略绩效追踪 — 胜率/夏普/回撤/权益曲线
"""

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

from src.trading import YuekaiBroker, AutoTrader, OrderSide, OrderType
from src.trading.account import account_manager
from src.trading.executor import RiskConfig
from src.trading.paper import paper_account
# 走本包的 paths 模块而非直接 `from config.settings import CONFIG_DIR`：
# 后者在「看板进程早于源码改动启动」时会撞上缓存里的旧模块并抛 ImportError。
from src.ui.paths import CONFIG_DIR
from src.ui.common import render_hero


def _guard_config_path() -> str | None:
    """安全配置路径：优先外挂 CONFIG_DIR（打包后 exe 同级，可改可持久），
    其次回退到源码目录 config/。都不存在则返回 None（用内置保守默认值）。"""
    candidates = [
        Path(CONFIG_DIR) / "trading_guard.yaml",
        Path(__file__).parent.parent.parent / "config" / "trading_guard.yaml",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


# =====================================================================
# 页面入口
# =====================================================================

def render_autotrade_page():
    """渲染自动交易页面"""
    render_hero(
        "量化自动交易",
        "纸交易零风险验证 · 实时交易日志 · 策略绩效追踪 —— 默认 PAPER，绝不自动实盘",
        icon="💹",
        badges=[("warn", "默认模拟盘")],
    )

    # Session state init
    defaults = {
        "trade_broker": None,
        "trade_autotrader": None,
        "trade_guard": None,        # 连接券商后构建的安全中枢（RiskSupervisor）
        "trade_running": False,
        "trade_connected": False,
        "trade_mode": "sim",
        "trade_strategy": "hybrid",
        "trade_event_log": deque(maxlen=200),  # 实时事件日志
        "trade_last_scan": None,
        "trade_scan_count": 0,
        "trade_loop": None,          # 闭环引擎（TradingLoop）实例
        "trade_loop_last": None,     # 最近一次 tick 报告
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔐 登录管理", "🔁 闭环全自动", "📊 账户仪表盘", "🧪 纸交易模拟",
        "🎯 AI信号交易", "📋 实时日志", "📈 量化分析", "📝 订单记录"
    ])

    with tab1:
        _render_login()
    with tab2:
        _render_loop()
    with tab3:
        _render_dashboard()
    with tab4:
        _render_paper_trading()
    with tab5:
        _render_ai_trading()
    with tab6:
        _render_live_log()
    with tab7:
        _render_quant()
    with tab8:
        _render_orders()


# =====================================================================
# Tab1: 登录管理
# =====================================================================

def _ensure_guard(broker):
    """连接券商后构建安全中枢（RiskSupervisor），与自动交易下单路径打通。"""
    try:
        from src.trading.guard import build_supervisor
        return build_supervisor(
            broker,
            config_path=_guard_config_path(),
            paper_account=paper_account,
        )
    except Exception as e:  # noqa: BLE001
        st.warning(f"⚠️ 安全中枢构建失败（不影响连接）：{e}")
        return None


def _render_login():
    st.markdown('<div class="section-title">🔐 券商账号登录</div>', unsafe_allow_html=True)

    # ===== 当前连接状态（透明展示，避免"登录了却不知道有没有用"）=====
    if st.session_state.trade_connected:
        b = st.session_state.trade_broker
        kind = "🧪 模拟盘（零真实资金）" if (b and b.is_simulated) else "💹 实盘"
        reason = f" · {b.sim_reason}" if (b and b.is_simulated and b.sim_reason) else ""
        g = st.session_state.get("trade_guard")
        gtxt = g.safety.mode.value if g else "未接入"
        st.success(f"✅ 已连接：{(b.get_broker_name() if b else '-')} · {kind}{reason}"
                   f"　｜　安全中枢：{gtxt}")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        # 模式选择（第一步，与是否保存账号无关）
        mode = st.radio(
            "交易模式", ["🧪 模拟模式", "💹 实盘模式"],
            index=0 if st.session_state.trade_mode == "sim" else 1,
            horizontal=True, key="trade_mode_radio"
        )
        st.session_state.trade_mode = "sim" if "模拟" in mode else "live"
        if st.session_state.trade_mode == "sim":
            st.caption("🧪 模拟盘：无需 xtquant / QMT，登录后即可用于纸交易与 AI 自动交易（零真实资金）。")
        else:
            st.caption("💹 实盘：需安装 xtquant 并启动粤开 QMT 客户端，否则会明确报错。")

        # 账号（模拟模式可留空）
        accounts = account_manager.accounts
        acct = None
        if accounts:
            labels = [f"{'🟢' if a.active else '⚪'} {a.name} ({a.broker})" for a in accounts]
            selected_idx = st.selectbox(
                "已保存账号", range(len(accounts)),
                format_func=lambda i: labels[i] if i < len(labels) else "",
                key="trade_acct_select"
            )
            if selected_idx is not None and selected_idx < len(accounts):
                acct = accounts[selected_idx]
                if st.session_state.trade_mode == "live":
                    if acct.qmt_path:
                        st.info(f"📋 QMT路径: `{acct.qmt_path}`")
                    else:
                        st.warning("⚠️ 实盘需要配置 QMT userdata_mini 路径（在下方展开填写）")

                # 直接补填 / 修改该账号的 QMT 路径（不影响已存密码等字段）
                with st.expander(
                    "✏️ 设置 / 修改该账号的 QMT 路径",
                    expanded=(st.session_state.trade_mode == "live" and not acct.qmt_path),
                ):
                    st.caption("填 QMT「极简模式」客户端安装目录下的 `…\\userdata_mini`，"
                               "例如 `D:\\QMT\\userdata_mini`。找不到？点下面的自动检测。")
                    new_path = st.text_input("QMT userdata_mini 路径",
                                             value=acct.qmt_path, key="edit_qmt_path")

                    b_save, b_detect = st.columns([1, 1])
                    with b_save:
                        save_clicked = st.button("💾 保存路径", key="btn_save_qmt_path",
                                                 use_container_width=True)
                    with b_detect:
                        detect_clicked = st.button("🔍 自动检测", key="btn_detect_qmt_path",
                                                   use_container_width=True)

                    if detect_clicked:
                        with st.spinner("正在本机扫描 userdata_mini（约 30 秒）…"):
                            try:
                                from src.trading.qmt_env import detect_qmt_userdata_mini
                                found, why = detect_qmt_userdata_mini()
                            except Exception as e:  # noqa: BLE001
                                found, why = None, f"{type(e).__name__}: {e}"
                        if found:
                            try:
                                account_manager.add_or_update(
                                    broker=acct.broker, account_id=acct.account_id,
                                    qmt_path=found, set_active=acct.active,
                                )
                                st.success(f"✅ 已检测并保存：`{found}`（{why}）")
                                st.rerun()
                            except Exception as e:  # noqa: BLE001
                                st.error(f"检测到 {found} 但保存失败：{e}")
                        else:
                            st.warning(f"未检测到 QMT：{why}。请确认 QMT 已安装，"
                                       "或手动填写 userdata_mini 路径。")

                    if save_clicked:
                        account_manager.add_or_update(
                            broker=acct.broker, account_id=acct.account_id,
                            qmt_path=new_path.strip(), set_active=acct.active,
                        )
                        st.success("✅ 已更新 QMT 路径（密码等其它字段保留）")
                        st.rerun()
        else:
            st.caption("（模拟模式无需保存账号；如需实盘，请在右侧添加账号）")

        col_conn, col_disc = st.columns(2)
        with col_conn:
            if st.button("🔗 连接", type="primary", use_container_width=True,
                         disabled=st.session_state.trade_connected, key="btn_connect"):
                broker = YuekaiBroker()
                pwd = account_manager.get_decrypted_password(acct.account_id) if acct else ""
                ok = broker.connect(
                    qmt_path=(acct.qmt_path if acct else ""),
                    account_id=(acct.qmt_account if acct else ""),
                    password=pwd,
                    mode=st.session_state.trade_mode,
                )
                if ok:
                    st.session_state.trade_broker = broker
                    st.session_state.trade_connected = True
                    st.session_state.trade_guard = _ensure_guard(broker)
                    kind = "模拟盘" if broker.is_simulated else "实盘"
                    st.success(f"✅ 已连接 {broker.get_broker_name()} · {kind}")
                    st.rerun()
                else:
                    st.error(f"❌ 连接失败：{broker.last_error or '请检查 QMT 客户端是否启动'}")

        with col_disc:
            if st.button("🔌 断开", use_container_width=True,
                         disabled=not st.session_state.trade_connected, key="btn_disconnect"):
                if st.session_state.trade_autotrader:
                    st.session_state.trade_autotrader.stop()
                    st.session_state.trade_autotrader = None
                    st.session_state.trade_running = False
                if st.session_state.trade_broker:
                    st.session_state.trade_broker.disconnect()
                    st.session_state.trade_broker = None
                st.session_state.trade_connected = False
                st.session_state.trade_guard = None
                st.success("🔌 已断开")
                st.rerun()

    with col_b:
        st.markdown('<div class="section-title">➕ 添加账号</div>', unsafe_allow_html=True)
        with st.form("add_broker_form"):
            acct_id = st.text_input("资金账号", placeholder="1234567890")
            acct_name = st.text_input("别名（可选）", placeholder="我的粤开账号")
            qmt_path = st.text_input("QMT userdata_mini 路径", placeholder="D:\\QMT\\userdata_mini")
            qmt_acct = st.text_input("QMT登录账号", placeholder="默认同资金账号")
            pwd = st.text_input("交易密码（加密存储）", type="password")
            if st.form_submit_button("💾 保存账号", use_container_width=True, type="primary"):
                if acct_id:
                    account_manager.add_or_update(
                        broker="yuekai", account_id=acct_id, name=acct_name,
                        qmt_path=qmt_path, qmt_account=qmt_acct or acct_id,
                        qmt_password=pwd, set_active=True,
                    )
                    st.success(f"✅ 账号 {acct_id} 已保存")
                    st.rerun()
                else:
                    st.error("请输入资金账号")


# =====================================================================
# Tab2: 账户仪表盘
# =====================================================================

# =====================================================================
# Tab2: 闭环全自动交易
# =====================================================================

_STATE_LABEL = {
    "stopped": "⏹️ 已停止",
    "preopen": "🌅 盘前待命",
    "running": "🟢 盘中执行中",
    "idle": "💤 非交易时段待命",
    "halted": "🛑 已熔断冻结",
    "settled": "✅ 今日已结算",
}


def _data_dir() -> Path:
    """数据目录（与 CONFIG_DIR 同级）。闭环单实例锁就放在这里。"""
    return Path(CONFIG_DIR).parent / "data"


def _loop_lock_path() -> str:
    """全机唯一的闭环单实例锁。

    必须与常驻 ``scripts/run_loop.py`` 用的是**同一个文件**
    （``data/.run_loop.lock``），否则「看板点启动」+「开机自启拉一份」
    会同时跑两个实例，对同一份计划各下一遍单（2026-10-19 实测踩到）。
    """
    return str(_data_dir() / ".run_loop.lock")


def _ensure_loop(broker, guard):
    """构建/复用闭环引擎（TradingLoop）。串接安全中枢、纸账户与推送。"""
    from src.trading.loop import TradingLoop, LoopConfig

    loop = st.session_state.get("trade_loop")
    if loop is not None:
        return loop

    cfg = LoopConfig.from_app_config()
    try:
        from src.core.notifier import build_notifier
        notifier = build_notifier()
    except Exception:  # noqa: BLE001
        notifier = None

    def _on_event(evt, data):
        try:
            st.session_state.trade_event_log.append(
                {"t": datetime.now().strftime("%H:%M:%S"), "evt": evt, "data": data})
        except Exception:  # noqa: BLE001
            pass

    loop = TradingLoop(
        broker, guard=guard, paper_account=paper_account,
        config=cfg, notifier=notifier, event_callback=_on_event,
        lock_path=_loop_lock_path(),
    )
    st.session_state.trade_loop = loop
    return loop


def _render_loop():
    st.markdown('<div class="section-title">🔁 闭环全自动交易</div>', unsafe_allow_html=True)
    st.caption(
        "盘前出计划 → 盘中执行（熔断 → 止盈止损 → 目标调仓）→ 收盘结算 → 推日报。"
        "**一切下单必经安全中枢**；默认纸交易，绝不自动进实盘。"
    )

    # ---------- 前置条件 ----------
    if not st.session_state.get("trade_connected") or not st.session_state.get("trade_broker"):
        st.warning("⚠️ 请先到「🔐 登录管理」连接券商（模拟模式无需账号，连接后即可跑纸交易闭环）。")
        return
    broker = st.session_state.trade_broker
    guard = st.session_state.get("trade_guard") or _ensure_guard(broker)
    if guard is None:
        st.error("❌ 安全中枢构建失败，闭环无法启动（不会裸奔下单）。")
        return

    loop = _ensure_loop(broker, guard)

    # 看板 in-process 线程之外的「常驻 run_loop.py」是否也在跑？
    # 两者抢同一把锁，所以看板点启动会被拒；这里先探明真相，避免给用户一个
    # 永远点不动的「启动闭环」按钮（实为防双开保护，不是故障）。
    from src.trading.loop import is_resident_loop_running
    resident = is_resident_loop_running()
    # 真正"在跑" = 看板 in-process 线程 或 常驻进程
    loop_alive = bool(loop.is_running) or bool(resident["running"])

    # ---------- 启停控制 ----------
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if not loop_alive:
            if st.button("▶️ 启动闭环", type="primary", use_container_width=True,
                         key="btn_loop_start"):
                if loop.start():
                    st.success("✅ 闭环已启动（后台线程按周期运行）")
                else:
                    st.error(
                        "⛔ **已有另一个闭环实例在运行**（单实例锁 "
                        f"`{_loop_lock_path()}` 被占用），已拒绝启动第二个实例 —— "
                        "两个实例会对同一份计划各下一遍单。"
                        "请先在原实例（常驻 `run_loop.py` 或另一个看板标签页）"
                        "停止闭环后再启动。"
                    )
                st.rerun()
        else:
            # 看板线程在跑 → 可在此停止；常驻在跑 → 提示去常驻进程停
            if loop.is_running:
                if st.button("⏹️ 停止闭环", use_container_width=True, key="btn_loop_stop"):
                    loop.stop()
                    st.success("⏹️ 闭环已停止")
                    st.rerun()
            else:
                st.markdown(
                    f"🟢 **常驻进程运行中**（PID `{resident.get('pid')}`，"
                    f"心跳 `{resident.get('age_sec'):.0f}s` 前）\n\n"
                    "闭环由独立的 `run_loop.py` 常驻进程驱动，看板仅用于监控。"
                    "如需从看板接管，请先结束该常驻进程（或在它已停时再点启动）。",
                )
    with c2:
        # 常驻在跑时禁止并发手动触发：会和常驻抢同一份计划/账户，造成重复下单。
        _disabled_once = bool(resident["running"])
        if st.button("🔄 手动跑一周期", use_container_width=True, key="btn_loop_once",
                     disabled=_disabled_once):
            with st.spinner("执行一个闭环周期…"):
                rep = loop.run_once()
            st.session_state.trade_loop_last = rep.to_dict()
            st.success("✅ " + rep.summary())
        if _disabled_once:
            st.caption("常驻进程运行中，已禁用以防重复下单")
    with c3:
        st.caption(f"扫描间隔 {loop.config.scan_interval_sec:.0f}s · "
                   f"状态机 {_STATE_LABEL.get(loop.state.value, loop.state.value)}")

    # ---------- 参数 ----------
    with st.expander("⚙️ 闭环参数", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            loop.config.max_positions = st.number_input(
                "目标持仓数", 1, 50, int(loop.config.max_positions), key="loop_maxpos")
        with p2:
            loop.config.cash_reserve_pct = st.slider(
                "现金保留 %", 0, 80, int(loop.config.cash_reserve_pct * 100),
                key="loop_reserve") / 100.0
        with p3:
            loop.config.weighting = "score" if st.selectbox(
                "权重方式", ["equal", "score"], index=0 if loop.config.weighting == "equal" else 1,
                format_func=lambda x: "等权（稳健）" if x == "equal" else "评分加权",
                key="loop_weighting") == "score" else "equal"
        with p4:
            loop.config.scan_interval_sec = float(st.number_input(
                "周期(秒)", 5, 600, int(loop.config.scan_interval_sec), key="loop_interval"))

        q1, q2, q3 = st.columns(3)
        with q1:
            loop.config.enable_rebalance = st.checkbox(
                "启用目标调仓", value=loop.config.enable_rebalance, key="loop_reb")
        with q2:
            loop.config.enable_stop_orders = st.checkbox(
                "启用止盈止损", value=loop.config.enable_stop_orders, key="loop_stop")
        with q3:
            loop.config.dry_run = st.checkbox(
                "🧪 演练模式（只预演不下单）", value=loop.config.dry_run, key="loop_dry")
        st.caption("修改后对新周期即时生效（无需重启闭环）。")

    st.divider()

    # ---------- 状态 ----------
    status = loop.get_status()
    g = status.get("guard") or {}
    # 看板 in-process 线程没跑、但常驻进程在跑时，状态以常驻心跳为准，避免显示"已停止"
    _live_state = status["state"]
    _is_resident = bool(resident.get("running")) and not loop.is_running
    if _is_resident and resident.get("state"):
        _live_state = resident["state"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("闭环状态",
              _STATE_LABEL.get(_live_state, _live_state)
              + ("（常驻）" if _is_resident else ""))
    m2.metric("安全中枢", f"{g.get('mode', '-')}"
              + ("（已熔断）" if g.get("halted") else ""))
    m3.metric("交易计划", status["plan_date"] or "无")
    m4.metric("目标持仓", f"{status['plan_targets']} 只")

    if loop.config.dry_run:
        st.info("🧪 演练模式已开启：只计算「本该下什么单」，不产生任何真实委托。")
    if g.get("halted"):
        st.error(f"🛑 安全中枢已熔断：{g.get('halt_reason') or '-'}"
                 "（到「⚙️ 设置 → 🛡️ 交易安全中枢」解除）")

    # ---------- 计划 + 调仓预演 ----------
    st.markdown("##### 📋 交易计划与调仓预演")
    try:
        pv = loop.preview()
    except Exception as e:  # noqa: BLE001
        st.warning(f"预演失败：{e}")
        pv = None

    if pv:
        pl = pv.get("plan")
        colL, colR = st.columns(2)
        with colL:
            st.markdown("**目标组合**")
            targets = (pl or {}).get("targets") or []
            if targets:
                import pandas as _pd
                st.dataframe(_pd.DataFrame([{
                    "代码": t["code"], "名称": t.get("name", ""),
                    "评分": t.get("score", 0),
                    "目标权重": f"{t.get('weight', 0)*100:.2f}%",
                } for t in targets]), use_container_width=True, hide_index=True)
                st.caption(f"计划日 {pl.get('plan_date')} · 现金保留 "
                           f"{pl.get('cash_reserve_pct', 0)*100:.0f}% · "
                           f"权重方式 {pl.get('weighting')}")
            else:
                st.info("当前无交易计划。可到「⚙️ 设置」开启调度器于收盘后自动生成，"
                        "或点下方按钮立即用最近一次选股结果生成。")
                if st.button("🛠️ 立即用最近选股结果生成计划", key="btn_make_plan"):
                    try:
                        from src.trading.trade_plan import build_trade_plan
                        np_ = build_trade_plan(
                            max_positions=loop.config.max_positions,
                            cash_reserve_pct=loop.config.cash_reserve_pct,
                            weighting=loop.config.weighting,
                            weight_cap=min(loop.config.weight_cap,
                                           float(getattr(guard.config,
                                                         "max_single_position_pct", 0.30))),
                            write=True)
                        st.success(f"✅ 已生成 {np_.plan_date} 计划（{len(np_.targets)} 只）")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"生成失败：{e}")
        with colR:
            st.markdown("**当前持仓与预演**")
            pos = pv.get("positions") or []
            st.caption(f"总资产 ¥{pv.get('equity', 0):,.0f} · 现金 ¥{pv.get('cash', 0):,.0f}"
                       f" · 持仓 {len(pos)} 只")
            if pv.get("stops"):
                st.warning("触发止盈止损：" + "、".join(
                    f"{s.get('code')}（{s.get('reason')}）" for s in pv["stops"]))
            st.code(pv.get("intent_summary") or "（无调仓意图）", language=None)

    # ---------- 最近一轮报告 ----------
    last = st.session_state.get("trade_loop_last")
    if last:
        st.markdown("##### 🧾 最近一轮执行报告")
        st.code(f"{last.get('at')} · {last.get('market_status')} · 状态={last.get('state')}\n"
                f"净值 ¥{last.get('equity', 0):,.0f} | 现金 ¥{last.get('cash', 0):,.0f} | "
                f"持仓 {last.get('n_positions')} 只\n"
                f"止损 {last.get('n_stops')} | 意图 {last.get('n_intents')} | "
                f"成交 {last.get('n_executed')} | 拦截 {last.get('n_blocked')}\n"
                f"{last.get('note', '')}", language=None)
        acts = last.get("actions") or []
        if acts:
            import pandas as _pd2
            st.dataframe(_pd2.DataFrame(acts), use_container_width=True, hide_index=True)

    # ---------- 事件流 ----------
    logs = list(st.session_state.get("trade_event_log", []))
    logs = [e for e in logs if str(e.get("evt", "")).startswith("loop")
            or e.get("evt") in ("fill", "risk_block", "halt", "loop_settled")]
    if logs:
        with st.expander(f"📜 闭环事件流（最近 {min(len(logs), 30)} 条）", expanded=False):
            for e in logs[-30:][::-1]:
                st.text(f"[{e['t']}] {e['evt']}  {json.dumps(e['data'], ensure_ascii=False)[:160]}")


def _render_dashboard():
    st.markdown('<div class="section-title">📊 账户仪表盘</div>', unsafe_allow_html=True)
    if not st.session_state.trade_connected:
        st.info("👆 请先在「登录管理」中连接券商，或直接用「纸交易模拟」")
        return

    broker = st.session_state.trade_broker
    if st.button("🔄 刷新数据", key="refresh_acct"):
        st.rerun()

    account = broker.query_account()
    if account:
        # 「今日盈亏」用纸账户的市值口径（已实现+浮动），不用 account.today_pnl ——
        # 后者是券商侧的当日盈亏，模拟盘下与纸账户对不上，两个数字同屏会打架。
        _pa = globals().get("paper_account", None)
        _today_pnl = getattr(account, "today_pnl", 0.0) or 0.0
        if _pa is not None and hasattr(_pa, "today_pnl_market"):
            _today_pnl = _pa.today_pnl_market()
        cols = st.columns(4)
        cols[0].metric("💰 总资产", f"¥{account.total_asset:,.0f}")
        cols[1].metric("🏦 可用资金", f"¥{account.available_cash:,.0f}")
        cols[2].metric("📦 持仓市值", f"¥{account.market_value:,.0f}")
        cols[3].metric("📈 今日盈亏", f"¥{_today_pnl:+,.0f}")

    st.markdown("---")
    positions = broker.query_positions()
    if positions:
        st.markdown(f"**📦 持仓明细 ({len(positions)} 只)**")
        pos_data = [{
            "代码": p.code, "名称": p.name,
            "持仓": f"{p.quantity}股", "可用": f"{p.available}股",
            "成本": f"{p.avg_cost:.2f}", "现价": f"{p.current_price:.2f}",
            "市值": f"¥{p.market_value:,.0f}",
            "盈亏": f"¥{p.pnl:+,.0f}", "盈亏%": f"{p.pnl_pct:+.2f}%",
        } for p in positions]
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
    else:
        st.info("📭 当前无持仓")


# =====================================================================
# Tab3: 纸交易模拟 ★ 新核心
# =====================================================================

def _render_paper_trading():
    st.markdown('<div class="section-title">🧪 纸交易模拟</div>', unsafe_allow_html=True)
    st.caption("独立虚拟账户 · 零风险验证策略 · 自动记录盈亏")

    pa = paper_account

    # === Row1: 账户总览 ===
    col1, col2, col3, col4 = st.columns(4)
    total_asset = pa.get_total_asset()
    col1.metric("💰 总资产", f"¥{total_asset:,.0f}",
                f"{pa.get_total_pnl_pct():+.2f}%" if pa.get_total_pnl_pct() != 0 else None)
    col2.metric("🏦 现金", f"¥{pa.cash:,.0f}")
    col3.metric("📦 持仓数", len(pa.holdings))
    stats = pa.get_performance_stats()
    col4.metric("📈 胜率", f"{stats['win_rate']:.1%}" if stats['total_trades'] > 0 else "—")

    # === Row2: 账户操作 ===
    col_op1, col_op2, col_op3 = st.columns(3)
    with col_op1:
        if st.button("🔄 刷新", use_container_width=True, key="paper_refresh"):
            st.rerun()
    with col_op2:
        new_cash = st.number_input("重置本金", 10000.0, 1000000.0, 100000.0, 10000.0, key="paper_reset_cash")
    with col_op3:
        if st.button("🔁 重置账户", use_container_width=True, key="paper_reset_btn"):
            pa.reset(new_cash)
            st.success(f"✅ 已重置，初始资金 ¥{new_cash:,.0f}")
            st.rerun()

    # === Row3: 持仓明细 ===
    # 市场状态（真实交易日历 + 时段），休市时明确提示不会成交
    try:
        from src.trading.calendar import market_status, is_degraded
        _ms = market_status()
        _deg = "（⚠️ 交易日历降级，按工作日估算）" if is_degraded() else ""
        _closed = ("休市" in _ms) or (_ms == "已收盘")
        st.caption(
            f"🕐 当前：{_ms}{_deg}"
            + (" — 非交易时段，下单不会真实成交" if _closed else "")
        )
    except Exception:  # noqa: BLE001
        pass

    st.markdown("---")
    st.markdown("### 📦 虚拟持仓")
    positions = pa.get_positions()
    if positions:
        df_pos = pd.DataFrame(positions)
        # 按盈亏排序
        df_pos = df_pos.sort_values("pnl", ascending=False)

        # 把 T+1 相关列排到前面（可卖 / 今日买入），其余保持原顺序
        _lead = ["code", "name", "quantity", "sellable", "today_quantity",
                 "available", "avg_cost", "last_price", "market_value",
                 "pnl", "pnl_pct"]
        _lead = [c for c in _lead if c in df_pos.columns]
        df_pos = df_pos[_lead + [c for c in df_pos.columns if c not in _lead]]

        st.dataframe(
            df_pos,
            column_config={
                "code": "代码",
                "name": "名称",
                "quantity": st.column_config.NumberColumn("持仓", format="%d 股"),
                "sellable": st.column_config.NumberColumn(
                    "可卖(T+1)", format="%d 股",
                    help="A股 T+1 规则下真正可卖出的数量（已扣除今日买入部分）"),
                "today_quantity": st.column_config.NumberColumn(
                    "今日买入", format="%d 股",
                    help="今日买入的股票当日不可卖，次日自动解锁"),
                "available": st.column_config.NumberColumn("可用", format="%d 股"),
                "avg_cost": st.column_config.NumberColumn("成本价", format="¥%.3f"),
                "last_price": st.column_config.NumberColumn("现价", format="¥%.3f"),
                "market_value": st.column_config.NumberColumn("市值", format="¥%.2f"),
                "pnl": st.column_config.NumberColumn("浮动盈亏", format="¥%.2f"),
                "pnl_pct": st.column_config.NumberColumn("盈亏%", format="%.2f%%"),
            },
            use_container_width=True, hide_index=True
        )
    else:
        st.info("📭 暂无持仓 — 去「AI信号交易」启动扫描后自动建仓")

    # === 绩效统计 ===
    if stats["total_trades"] > 0:
        st.markdown("---")
        st.markdown("### 📊 绩效统计")
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("总交易", stats["total_trades"])
        pc2.metric("胜", stats["wins"]); pc3.metric("负", stats["losses"])
        pc4.metric("盈亏比", f"{stats['profit_factor']:.2f}")
        pc5.metric("最大回撤", f"{stats['max_drawdown_pct']:.2f}%")

        pc6, pc7, pc8 = st.columns(3)
        pc6.metric("夏普", f"{stats['sharpe']:.3f}")
        pc7.metric("总盈亏", f"¥{stats['total_pnl']:+,.0f}")
        pc8.metric("累计收益", f"{stats['total_return_pct']:+.2f}%")

    # === 最近交易记录 ===
    trades = pa.get_recent_trades(30)
    if trades:
        st.markdown("---")
        st.markdown("### 📝 最近交易")
        trade_rows = []
        for t in reversed(trades):
            emoji = "🟢" if t.side == "buy" else "🔴"
            trade_rows.append({
                "": emoji, "代码": t.code, "名称": t.name[:6],
                "方向": "买" if t.side == "buy" else "卖",
                "价格": f"{t.price:.2f}", "数量": t.quantity,
                "金额": f"¥{t.amount:,.0f}",
                "盈亏": f"¥{t.pnl:+,.2f}" if t.side == "sell" else "—",
                "盈亏%": f"{t.pnl_pct:+.2f}%" if t.side == "sell" and t.pnl_pct else "—",
                "AI评分": f"{t.ai_score:.0f}" if t.ai_score else "—",
                "时间": t.filled_at[:16] if t.filled_at else "",
            })
        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)


# =====================================================================
# Tab4: AI信号交易
# =====================================================================

def _render_ai_trading():
    st.markdown('<div class="section-title">🎯 AI信号自动交易</div>', unsafe_allow_html=True)

    # ===== 策略选择 =====
    strat_col1, strat_col2 = st.columns([1, 3])
    with strat_col1:
        strategy_mode = st.selectbox(
            "📐 交易策略",
            ["AI+量化融合", "纯AI预测", "多因子量化", "均值回归"],
            index=0, key="strat_mode_select",
        )
    with strat_col2:
        mode_map = {
            "AI+量化融合": ("hybrid", "AI + 量化双重验证，两套模型一致时才下单"),
            "纯AI预测": ("ai", "XGBoost 单模型预测信号"),
            "多因子量化": ("multi", "价值/动量/质量/低波/趋势五因子综合评分"),
            "均值回归": ("mean_rev", "布林带 + RSI 超买超卖回归策略"),
        }
        info = mode_map[strategy_mode]
        st.session_state.trade_strategy = info[0]
        st.caption(f"💡 {info[1]}")

    if strategy_mode == "AI+量化融合":
        ai_weight = st.slider("AI权重 (vs 量化)", 0.1, 0.9, 0.5, 0.1, key="hybrid_ai_w")

    # ===== 风控参数 =====
    with st.expander("🛡️ 风控参数", expanded=False):
        ca, cb, cc = st.columns(3)
        with ca:
            max_pos = st.slider("单票最大仓位", 0.05, 1.0, 0.30, 0.05, format="%.0f%%", key="rk_maxpos")
            stop_loss = st.slider("止损线", -20.0, -1.0, -5.0, 0.5, format="%.0f%%", key="rk_sl")
        with cb:
            stop_profit = st.slider("止盈线", 5.0, 50.0, 15.0, 1.0, format="%.0f%%", key="rk_sp")
            max_daily_loss = st.number_input("日最大亏损(元)", 1000.0, 1000000.0, 5000.0, 1000.0, key="rk_mdl")
        with cc:
            min_ai = st.slider("买入最低评分", 35.0, 65.0, 55.0, 5.0, key="rk_mas")
            scan_iv = st.number_input("扫描间隔(秒)", 10, 600, 60, 10, key="rk_si",
                                       help="AI交易后台扫描间隔")
        st.checkbox("仅交易时段运行", value=True, key="rk_tho")
        st.checkbox("需要双模型一致", value=True, key="rk_agr")

    # ===== 控制按钮 =====
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🚀 启动纸交易", type="primary", use_container_width=True,
                     disabled=st.session_state.trade_running, key="btn_start_paper"):
            _start_paper_trader(min_ai, scan_iv, max_pos, stop_loss, stop_profit,
                                max_daily_loss, strategy_mode, ai_weight)
    with c2:
        if st.button("⏹️ 停止", use_container_width=True,
                     disabled=not st.session_state.trade_running, key="btn_stop"):
            _stop_trader()
    with c3:
        if st.button("🔍 单次扫描", use_container_width=True, key="btn_scan_once"):
            _scan_once(min_ai, strategy_mode, ai_weight)
    with c4:
        if st.button("🔄 刷新状态", use_container_width=True, key="btn_refresh_status"):
            st.rerun()

    # ===== 运行状态 =====
    if st.session_state.trade_running and st.session_state.trade_autotrader:
        at = st.session_state.trade_autotrader
        st.markdown("---")
        st.markdown(f"### {'🟢 运行中' if at.is_running else '🔴 已停止'}  "
                    f"<small>扫描间隔 {scan_iv}s | 上次: {st.session_state.trade_last_scan or '—'}</small>",
                    unsafe_allow_html=True)
        cols = st.columns(5)
        cols[0].metric("扫描次数", st.session_state.get("trade_scan_count", 0))
        cols[1].metric("信号", at.stats.get("signals_received", 0))
        cols[2].metric("下单", at.stats.get("orders_placed", 0))
        cols[3].metric("止损", at.stats.get("stops_triggered", 0))
        cols[4].metric("纸交易盈亏", f"¥{paper_account.get_total_pnl():+,.0f}")

    # ===== 快速手动下单 =====
    st.markdown("---")
    with st.expander("✋ 手动下单（纸交易）", expanded=False):
        with st.form("paper_manual_order"):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                order_code = st.text_input("代码", "000001.SZ", key="pm_code")
            with c2:
                order_side = st.selectbox("方向", ["买入", "卖出"], key="pm_side")
            with c3:
                order_qty = st.number_input("数量", 100, 1000000, 1000, 100, key="pm_qty")
            with c4:
                order_price = st.number_input("价格", 0.01, 10000.0, 10.0, 0.01, key="pm_px")
            
            pm_name = st.text_input("股票名称（可选）", "", key="pm_name")
            
            if st.form_submit_button("✋ 纸交易下单", type="primary", use_container_width=True):
                price = order_price
                if price <= 0:
                    try:
                        from src.data.fetcher import fetcher
                        df = fetcher.get_daily_kline(order_code)
                        price = float(df["close"].iloc[-1]) if len(df) > 0 else 0
                    except Exception:
                        price = 10.0
                
                name = pm_name or order_code
                if order_side == "买入":
                    ok, msg, _ = paper_account.buy(order_code, name, price, order_qty, strategy="手动")
                else:
                    ok, msg, _ = paper_account.sell(order_code, price, order_qty, strategy="手动")
                
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


def _make_event_callback():
    """创建一个事件回调，把事件推到 st.session_state 的日志队列"""
    def on_event(event_type: str, data: dict):
        log = st.session_state.get("trade_event_log")
        if log is None:
            return
        log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": event_type,
            "data": data,
        })
        if event_type == "scan_done":
            st.session_state.trade_scan_count = st.session_state.get("trade_scan_count", 0) + 1
            st.session_state.trade_last_scan = data.get("time", "")
    return on_event


def _start_paper_trader(min_ai, scan_iv, max_pos, stop_loss, stop_profit, max_daily_loss, strategy_mode, ai_weight):
    """启动纸交易后台"""
    risk = RiskConfig(
        max_position_pct=max_pos, stop_loss_pct=stop_loss,
        stop_profit_pct=stop_profit, max_daily_loss=max_daily_loss,
        min_ai_score_buy=min_ai,
    )
    
    # 复用「登录管理」里已连接的券商；未登录则按当前模式新建
    broker = st.session_state.get("trade_broker")
    if broker is None:
        broker = YuekaiBroker()
        broker.connect(mode=st.session_state.get("trade_mode", "sim"))

    trader = AutoTrader(broker, risk)
    trader.paper_account = paper_account  # 注入纸账户
    trader.on_event = _make_event_callback()  # 注入日志回调
    # 接入安全中枢（登录后构建）；默认 PAPER 模式，零实盘风险
    guard = st.session_state.get("trade_guard")
    if guard is not None:
        trader.set_guard(guard)
    
    # 加载自选股 + 注入预测器（watchlist=None → 自动从文件加载）
    strategy = st.session_state.get("trade_strategy", "ai")
    if strategy == "ai":
        trader.use_builtin_predictor(None)  # 自动加载
    else:
        from src.trading.signals import create_strategy_generator
        gen = create_strategy_generator(
            mode=strategy,
            multi_min_score=min_ai,
            hybrid_ai_weight=st.session_state.get("hybrid_ai_w", 0.5),
        )
        trader._ai_predict_fn = gen.make_predict_fn()
        trader.config.min_ai_score_buy = 40.0 if strategy == "mean_rev" else min_ai
        trader.config.require_agreement = (strategy == "hybrid")
    
    # 启动策略 + 止盈止损独立监控
    trader.start_stop_monitor(interval_seconds=10)
    trader.start(interval_seconds=scan_iv,
                 trading_hours_only=st.session_state.get("rk_tho", True))
    
    st.session_state.trade_autotrader = trader
    st.session_state.trade_running = True
    st.session_state.trade_scan_count = 0
    
    st.rerun()


def _stop_trader():
    if st.session_state.trade_autotrader:
        st.session_state.trade_autotrader.stop()
    st.session_state.trade_running = False
    st.info("⏹️ 已停止")


def _scan_once(min_ai, strategy_mode, ai_weight):
    """手动单次扫描"""
    broker = st.session_state.get("trade_broker")
    if broker is None:
        broker = YuekaiBroker()
        broker.connect(mode=st.session_state.get("trade_mode", "sim"))

    risk = RiskConfig(min_ai_score_buy=min_ai)
    trader = AutoTrader(broker, risk)
    trader.paper_account = paper_account
    trader.on_event = _make_event_callback()
    guard = st.session_state.get("trade_guard")
    if guard is not None:
        trader.set_guard(guard)
    
    strategy = st.session_state.get("trade_strategy", "ai")
    if strategy == "ai":
        trader.use_builtin_predictor(None)
    else:
        from src.trading.signals import create_strategy_generator
        gen = create_strategy_generator(
            mode=strategy, multi_min_score=min_ai,
            hybrid_ai_weight=st.session_state.get("hybrid_ai_w", 0.5),
        )
        trader._ai_predict_fn = gen.make_predict_fn()
        trader.config.min_ai_score_buy = 40.0 if strategy == "mean_rev" else min_ai
    
    results = trader.run_once()
    if results:
        st.success(f"✅ 扫描完成: {len(results)} 个信号 (详情见实时日志)")
    else:
        st.info("📭 当前无符合条件的交易信号")
    st.rerun()


# =====================================================================
# Tab5: 实时日志 ★ 新核心
# =====================================================================

def _render_live_log():
    st.markdown('<div class="section-title">📋 实时交易日志</div>', unsafe_allow_html=True)
    st.caption("纸交易/实盘交易的所有事件实时展示")

    log = st.session_state.get("trade_event_log")
    if not log:
        st.info("📭 暂无事件 — 启动「AI信号交易」或执行单次扫描后日志会出现在这里")
        return

    # 统计概览
    events = list(log)
    scan_events = [e for e in events if e["type"] == "scan_done"]
    trade_events = [e for e in events if e["type"] == "trade"]
    info_events = [e for e in events if e["type"] == "info"]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("扫描完成", len(scan_events))
    col2.metric("交易事件", len(trade_events))
    col3.metric("信息事件", len(info_events) + sum(1 for e in events if e["type"] == "warn"))
    
    st.markdown("---")
    
    # 最近扫描概览
    if scan_events:
        latest_scan = scan_events[-1]["data"]
        st.markdown(f"**📡 最近扫描** — {latest_scan.get('time', '?')}")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("买入", latest_scan.get("buy_count", 0))
        sc2.metric("卖出", latest_scan.get("sell_count", 0))
        sc3.metric("信号", latest_scan.get("signals", 0))
        sc4.metric("止损", latest_scan.get("stops", 0))

    # 交易事件列表
    if trade_events:
        st.markdown("---")
        st.markdown("**📝 最近交易**")
        rows = []
        for e in reversed(trade_events[-30:]):
            d = e["data"]
            emo = "📈" if d.get("side") == "buy" else "📉"
            rows.append({
                "时间": e["time"],
                "": emo,
                "代码": d.get("code", ""),
                "名称": d.get("name", "")[:6],
                "方向": "买入" if d.get("side") == "buy" else "卖出",
                "价格": f"{d.get('price', 0):.2f}",
                "数量": d.get("quantity", 0),
                "AI评分": f"{d.get('ai_score', 0):.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 完整日志流（可折叠）
    with st.expander("📜 完整事件流", expanded=False):
        log_lines = []
        for e in reversed(events[-100:]):
            d = e["data"]
            prefix = {"scan_start": "🔍", "scan_done": "✅", "trade": "💱",
                      "warn": "⚠️", "blocked": "⛔", "info": "ℹ️",
                      "check_positions": "🔎", "buy_candidates": "📊",
                      "stop_triggered": "🛑"}.get(e["type"], "•")
            msg = d.get("msg", "") if isinstance(d, dict) else str(d)
            log_lines.append(f"`{e['time']}` {prefix} **{e['type']}** {msg}")
        st.markdown("\n\n".join(log_lines) if log_lines else "_等待事件…_")


# =====================================================================
# Tab6: 量化策略分析
# =====================================================================

def _render_quant():
    """量化策略分析（精简版，完整版在交易信号扫描中）"""
    st.markdown('<div class="section-title">📈 量化策略分析</div>', unsafe_allow_html=True)

    wl = _load_watchlist()
    if not wl:
        st.warning("⚠️ 未找到自选股配置 (config/watchlist.json)")
        return

    tab_q1, tab_q2, tab_q3 = st.tabs(["📊 多因子评分", "🔄 均值回归", "🧬 AI+量化融合"])

    with tab_q1:
        _render_multi_factor(wl)
    with tab_q2:
        _render_mean_rev(wl)
    with tab_q3:
        _render_hybrid_fusion(wl)


def _load_watchlist():
    try:
        cfg_path = Path(__file__).parent.parent.parent / "config" / "watchlist.json"
        return json.loads(cfg_path.read_text(encoding="utf-8")).get("watchlist", [])
    except Exception:
        return []


def _render_multi_factor(wl):
    st.caption("价值 / 动量 / 质量 / 低波 / 趋势 — 五因子综合评分")
    if st.button("🔍 执行多因子扫描", use_container_width=True, key="qf_scan"):
        with st.spinner(f"扫描 {len(wl)} 只自选股…"):
            from src.models.quantitative import quant_engine
            from src.data.fetcher import fetcher

            rows = []
            name_map = {}
            try:
                sl = fetcher.get_stock_list()
                name_map = dict(zip(sl["code"], sl["name"]))
            except Exception:
                name_map = {c: c for c in wl}

            for code in wl:
                try:
                    df = fetcher.get_daily_kline(code)
                    qs = quant_engine.score_stock(df, code, name_map.get(code, code))
                    if qs.total_score > 0:
                        rows.append({
                            "代码": qs.code, "名称": qs.name[:6],
                            "综合评分": f"{qs.total_score:.1f}", "信号": qs.signal,
                            "价值": f"{qs.value_score:.0f}", "动量": f"{qs.momentum_score:.0f}",
                            "质量": f"{qs.quality_score:.0f}", "低波": f"{qs.volatility_score:.0f}",
                            "趋势": f"{qs.trend_score:.0f}",
                            "夏普": f"{qs.sharpe:.2f}", "回撤": f"{qs.max_drawdown:.1%}",
                        })
                except Exception:
                    pass

            if rows:
                df_all = pd.DataFrame(rows)
                df_all["_sort"] = df_all["综合评分"].astype(float)
                df_all = df_all.sort_values("_sort", ascending=False).drop(columns=["_sort"])
                st.success(f"✅ 扫描完成: {len(rows)}/{len(wl)} 只")
                st.dataframe(df_all, use_container_width=True, hide_index=True)


def _render_mean_rev(wl):
    st.caption("布林带 + RSI 超买超卖回归策略")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mr_lookback = st.slider("布林带周期", 10, 60, 20, 5, key="mr_lb")
        mr_entry = st.slider("入场Z-score阈值", 1.0, 3.0, 2.0, 0.2, key="mr_ez")
    with col_m2:
        mr_capital = st.number_input("回测本金(万)", 1, 100, 10, 1, key="mr_cap")

    if st.button("🔍 均值回归扫描", use_container_width=True, key="mr_scan"):
        with st.spinner(f"扫描 {len(wl)} 只…"):
            from src.models.quantitative import MeanReversionStrategy
            from src.data.fetcher import fetcher

            mr = MeanReversionStrategy(lookback=mr_lookback, entry_threshold=mr_entry)
            name_map = {}
            try:
                sl = fetcher.get_stock_list()
                name_map = dict(zip(sl["code"], sl["name"]))
            except Exception:
                name_map = {c: c for c in wl}

            rows = []
            for code in wl:
                try:
                    df = fetcher.get_daily_kline(code)
                    bt = mr.backtest(df, capital=mr_capital * 10000)
                    z = (df["close"].iloc[-1] - df["close"].rolling(20).mean().iloc[-1]) / (
                        df["close"].rolling(20).std().iloc[-1] + 1e-10
                    )
                    rows.append({
                        "代码": code, "名称": name_map.get(code, code)[:6],
                        "策略收益": f"{bt['total_return']:+.1f}%",
                        "买入持有": f"{bt['buy_hold_return']:+.1f}%",
                        "Alpha": f"{bt['alpha']:+.1f}%",
                        "交易次数": bt["n_trades"],
                        "Z-Score": f"{z:.2f}",
                    })
                except Exception:
                    pass

            if rows:
                df_all = pd.DataFrame(rows)
                df_all["_alpha"] = df_all["Alpha"].str.replace("%", "").astype(float)
                df_all = df_all.sort_values("_alpha", ascending=False).drop(columns=["_alpha"])
                st.success(f"✅ 扫描完成: {len(rows)}/{len(wl)} 只")
                st.dataframe(df_all, use_container_width=True, hide_index=True)
            else:
                st.info("📭 无有效数据")


def _render_hybrid_fusion(wl):
    st.caption("AI模型 + 多因子量化 双引擎融合评分")
    h_ai_weight = st.slider("AI权重", 0.1, 0.9, 0.5, 0.1, key="hb_ai_w2")
    h_min_score = st.slider("买入下限", 35.0, 65.0, 52.0, 1.0, key="hb_min2")

    if st.button("🧬 AI+量化融合扫描", use_container_width=True, key="hb_scan2"):
        with st.spinner(f"融合扫描 {len(wl)} 只…"):
            from src.trading.signals import create_strategy_generator
            gen = create_strategy_generator(
                mode="hybrid", watchlist_codes=wl,
                hybrid_ai_weight=h_ai_weight, multi_min_score=h_min_score,
            )
            signals = gen.generate_all()

            if signals:
                rows = []
                for sig in signals:
                    d = sig.quant_details
                    ai_s = d.get("ai_score", 50)
                    qt_s = d.get("quant_score", 50)
                    rows.append({
                        "代码": sig.code, "名称": sig.name[:6],
                        "融合评分": f"{sig.score:.1f}",
                        "信号": "📈" if sig.signal == 1 else ("📉" if sig.signal == -1 else "—"),
                        "AI评分": f"{ai_s:.0f}", "量化评分": f"{qt_s:.0f}",
                        "置信度": f"{sig.agreement:.0%}",
                        "理由": "; ".join(sig.reasons[:2]),
                    })
                st.success(f"✅ 扫描完成: {len(signals)} 个有效信号")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("📭 当前无符合条件的融合信号")


# =====================================================================
# Tab7: 订单记录
# =====================================================================

def _render_orders():
    st.markdown('<div class="section-title">📝 订单记录</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🏦 券商订单", "🧪 纸交易记录"])
    
    with tabs[0]:
        if not st.session_state.trade_connected:
            st.info("👆 请先连接券商")
        else:
            broker = st.session_state.trade_broker
            if st.button("🔄 查询", key="refresh_orders"):
                st.rerun()
            orders = broker.query_orders()
            if orders:
                ord_data = []
                for o in orders:
                    emoji = "🟢" if o.side == OrderSide.BUY else "🔴"
                    t_str = o.create_time.strftime("%H:%M:%S") if o.create_time else ""
                    ord_data.append({
                        "": emoji, "订单ID": o.order_id, "代码": o.code,
                        "方向": "买" if o.side == OrderSide.BUY else "卖",
                        "价格": o.price, "数量": o.quantity,
                        "成交": f"{o.filled_quantity}/{o.quantity}",
                        "状态": o.status.value,
                        "AI评分": o.ai_score if o.ai_score > 0 else "—",
                        "时间": t_str,
                    })
                st.dataframe(pd.DataFrame(ord_data), use_container_width=True, hide_index=True)
            else:
                st.info("📭 今日无订单记录")
    
    with tabs[1]:
        trades = paper_account.get_recent_trades(50)
        if trades:
            rows = []
            for t in reversed(trades):
                emo = "🟢" if t.side == "buy" else "🔴"
                rows.append({
                    "": emo, "ID": t.trade_id, "代码": t.code, "名称": t.name[:6],
                    "方向": "买" if t.side == "buy" else "卖",
                    "价格": f"{t.price:.2f}", "数量": t.quantity,
                    "金额": f"¥{t.amount:,.0f}",
                    "盈亏": f"¥{t.pnl:+,.2f}" if t.side == "sell" else "—",
                    "策略": t.strategy, "AI评分": f"{t.ai_score:.0f}" if t.ai_score else "—",
                    "时间": t.filled_at[:16] if t.filled_at else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("📭 暂无纸交易记录")

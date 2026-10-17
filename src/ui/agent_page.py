# -*- coding: utf-8 -*-
"""🤖 AI 助手页：自然语言驱动的工具调用代理（ReAct）。"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.models.llm import deepseek
from src.agent.engine import run_agent
from src.agent.tools import call_tool, TOOL_FAILED_MARK


def render_agent_page():
    st.markdown('<p class="main-header">🤖 AI 助手</p>', unsafe_allow_html=True)
    st.markdown(
        "用自然语言驱动选股神 AI：分析个股、跑每日选股、查实时行情、生成研报、诊断进化引擎。"
        "助手会**自主调用工具**拿数据，再综合成回答。"
    )

    if not deepseek.is_ready:
        st.error(
            "⚠️ LLM 未就绪。请到「⚙️ 设置」页确认已配置 DeepSeek（或切到本地 Ollama/LM Studio）"
            "并连通。AI 助手需要可用的模型才能工作。"
        )
        return

    # ---- 会话状态 ----
    # 注意：这里**不再**有「顶部决策守卫」。旧实现把决策写进 session_state
    # 却不在点击处 st.rerun()，于是「点击产生的那次重跑」里，顶部守卫早就以
    # 空值判定过了 —— 决策被存下来却永远没人处理，按钮点了没反应。
    # 现在改为在按钮点击处**就地处理**（见 _apply_decision），不依赖下一次重跑。
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # ---- 快捷指令 ----
    st.markdown("**试试这些指令：**")
    chips = [
        "智能筛选偏多的股票",
        "生成一份可下载的筛选研报",
        "分析一下宁德时代",
        "600519 现在实时行情怎么样",
        "诊断一下进化引擎状态",
    ]
    cols = st.columns(len(chips))
    for i, c in enumerate(chips):
        if cols[i].button(c, key=f"chip_{i}", use_container_width=True):
            st.session_state.agent_prompt = c

    # ---- 输入框 ----
    prompt = st.text_input(
        "给 AI 助手下达指令",
        value=st.session_state.get("agent_prompt", ""),
        key="agent_input",
        placeholder="例如：帮我分析 002416，并和今天的选股结果对照一下",
    )
    # 快捷指令写入后清空，避免重复触发
    if st.session_state.get("agent_prompt"):
        st.session_state.agent_prompt = ""

    submitted = st.button("🚀 发送", type="primary", use_container_width=True)
    st.caption(
        "⏱️ 个股行情 / 大盘指数：数秒；"
        "「智能筛选 / 生成研报」要拉整个股票池的日线，约 1 分钟，请耐心等待（会显示进度）。"
    )

    if submitted and prompt.strip():
        _handle_send(prompt.strip())

    # ---- 渲染历史 ----
    for msg in st.session_state.agent_messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            _render_assistant(msg["content"], msg.get("result"))


def _handle_send(prompt: str):
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    # 保留历史（最多最近 6 轮，避免 prompt 过长）
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.agent_messages[-12:-1]
    ]
    with st.chat_message("assistant"):
        with st.spinner("🤖 唤醒本地模型…" if deepseek.is_local else "🤖 助手正在思考…"):
            # 本地模型闲置 5 分钟会被 Ollama 卸载，下次提问要先重载 4.6GB 权重
            # （实测冷启动 30~45s、热态 1~2s）。先预热并把 keep_alive 拉长，
            # 顺便让 spinner 文案说明白「在唤醒模型」而不是干等。
            if deepseek.is_local:
                deepseek.warmup()
        with st.spinner("🤖 助手正在思考并调用工具…"):
            result = run_agent(prompt, history=history, max_steps=5)
        _render_assistant_inline(result)
    st.session_state.agent_messages.append(
        {"role": "assistant", "content": result.final_text, "result": result}
    )
    # 标记该消息为「待确认」，供后续就地改写为执行/取消结果
    st.session_state.agent_messages[-1]["pending"] = bool(getattr(result, "pending_action", None))
    st.rerun()


def _render_assistant(content: str, result=None):
    container = st.chat_message("assistant")
    _render_result(container, result, content)


def _render_assistant_inline(result):
    container = st
    _render_result(container, result, result.final_text)


def _pending_widget_key(pa: dict, kind: str) -> str:
    """为「确认/取消」这对按钮生成**跨重跑稳定**的 widget key。

    ⚠️ 不能用 ``id(pa)``：那个值随每次重跑重建的对象而变（实测同一次点击
    前后是 confirm_2072315531840 → Confirm_2072315532000），key 一变，
    「这次点击」就对不上「这次渲染的按钮」，点击会被丢进已消失的旧 widget 里，
    于是按钮**点了没反应**——这正是本页那个 bug 的成因之一。
    改用提案内容（tool+参数）做 key：同一次待确认里它恒定不变。
    """
    key = f"{pa.get('tool', '')}|{json.dumps(pa.get('args', {}), sort_keys=True, ensure_ascii=False)}"
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return f"pa_{kind}_{h}"


def _render_result(container, result, content: str):
    # 待确认动作：渲染确认框；用户决策后由 _handle_pending_decision 就地改写该消息
    if result is not None and getattr(result, "pending_action", None):
        pa = result.pending_action
        with container.expander("⚠️ 待确认的操作", expanded=True):
            container.markdown(pa.get("desc", ""))
            container.markdown(f"**影响**：{pa.get('impact', '')}")
            container.markdown(f"**风险等级**：{pa.get('risk', '低')}")
            c1, c2 = container.columns(2)
            # 关键：点击处**就地处理**再 st.rerun()。
            # st.session_state 在同一轮内写入后读得到，所以这里能直接把决策落地，
            # 不必等下一次重跑（旧实现正是败在这一步：只写标记不处理、也不重跑）。
            if c1.button("✅ 确认执行", key=_pending_widget_key(pa, "confirm"), type="primary"):
                st.session_state["_agent_pending_action"] = pa
                st.session_state["_agent_pending_decision"] = "confirm"
                _handle_pending_decision()
            if c2.button("❌ 取消", key=_pending_widget_key(pa, "cancel")):
                st.session_state["_agent_pending_action"] = pa
                st.session_state["_agent_pending_decision"] = "cancel"
                _handle_pending_decision()
        return

    # 逐步展示工具调用（若有）
    if result is not None and getattr(result, "steps", None):
        for i, step in enumerate(result.steps, 1):
            raw_obs = str(step.observation or "")
            failed = raw_obs.startswith(TOOL_FAILED_MARK)
            obs = raw_obs.replace(TOOL_FAILED_MARK, "")
            icon = "❌" if failed else "🔧"
            with container.expander(f"{icon} 步骤 {i}：{step.action}", expanded=failed):
                container.markdown(
                    f"**参数**：`{json.dumps(step.action_input, ensure_ascii=False)}`"
                )
                container.markdown("**失败原因**：" if failed else "**观测结果**：")
                container.markdown(obs if obs else "（无返回）", unsafe_allow_html=False)
    # 工具产出的可下载文件（如筛选研报）
    if result is not None:
        arts = getattr(result, "artifacts", None) or []
        for art in arts:
            p = Path(art)
            if p.exists():
                try:
                    data = p.read_text(encoding="utf-8")
                    container.download_button(
                        label=f"📄 下载研报：{p.name}",
                        data=data,
                        file_name=p.name,
                        mime="text/markdown",
                    )
                except Exception:
                    pass
    if result is not None and getattr(result, "error", None):
        container.warning(result.error)
    container.markdown(content or "（无回答）", unsafe_allow_html=False)


def _handle_pending_decision():
    """处理用户在待确认操作上的「确认/取消」：就地改写最后一条待确认消息。"""
    decision = st.session_state.get("_agent_pending_decision")
    spec = st.session_state.get("_agent_pending_action")
    st.session_state["_agent_pending_decision"] = None
    st.session_state["_agent_pending_action"] = None
    if not spec:
        st.rerun()
        return

    msgs = st.session_state.agent_messages
    last = msgs[-1] if msgs else None

    if decision == "cancel":
        if last is not None and last.get("pending"):
            last["content"] = f"🚫 已取消操作：{spec.get('desc', '')}"
            last["result"] = None
            last["pending"] = False
        st.rerun()
        return

    # 确认执行：用 approve=True 重新调用同一工具真正落盘/改文件/模拟下单
    args = dict(spec.get("args") or {})
    # 防御：无条件置 approve=True。提案里的 args 本就带 True（四个变更工具都在
    # 预览分支里写死），但如果哪天工具改了签名、或提案被别处拼出来少了这个键，
    # 这次调用只会再产出一次预览 —— 表现为「点了确认却什么都没发生」，
    # 正是本项目最忌的静默失效。宁可多设一次，也不能让确认变成空转。
    args["approve"] = True
    try:
        out = call_tool(spec["tool"], args)
    except Exception as e:  # noqa: BLE001
        out = f"⚠️ 执行失败：{type(e).__name__}: {e}"
    if last is not None and last.get("pending"):
        last["content"] = f"✅ 已确认执行：{spec.get('desc', '')}\n\n{out}"
        last["result"] = None
        last["pending"] = False
    st.rerun()

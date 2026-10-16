# -*- coding: utf-8 -*-
"""🤖 AI 助手页：自然语言驱动的工具调用代理（ReAct）。"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.models.llm import deepseek
from src.agent.engine import run_agent
from src.agent.tools import call_tool


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

    # 处理待确认动作的「确认/取消」（在渲染消息前先处理，内部会 st.rerun）
    if st.session_state.get("_agent_pending_decision"):
        _handle_pending_decision()
        return

    # ---- 会话状态 ----
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


def _render_result(container, result, content: str):
    # 待确认动作：渲染确认框；用户决策后由 _handle_pending_decision 就地改写该消息
    if result is not None and getattr(result, "pending_action", None):
        pa = result.pending_action
        with container.expander("⚠️ 待确认的操作", expanded=True):
            container.markdown(pa.get("desc", ""))
            container.markdown(f"**影响**：{pa.get('impact', '')}")
            container.markdown(f"**风险等级**：{pa.get('risk', '低')}")
            c1, c2 = container.columns(2)
            if c1.button("✅ 确认执行", key=f"confirm_{id(pa)}", type="primary"):
                st.session_state["_agent_pending_action"] = pa
                st.session_state["_agent_pending_decision"] = "confirm"
            if c2.button("❌ 取消", key=f"cancel_{id(pa)}"):
                st.session_state["_agent_pending_action"] = pa
                st.session_state["_agent_pending_decision"] = "cancel"
        return

    # 逐步展示工具调用（若有）
    if result is not None and getattr(result, "steps", None):
        for i, step in enumerate(result.steps, 1):
            with container.expander(f"🔧 步骤 {i}：{step.action}", expanded=False):
                container.markdown(
                    f"**参数**：`{json.dumps(step.action_input, ensure_ascii=False)}`"
                )
                container.markdown("**观测结果**：")
                container.markdown(
                    step.observation if step.observation else "（无返回）",
                    unsafe_allow_html=False,
                )
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
    try:
        out = call_tool(spec["tool"], spec["args"])
    except Exception as e:  # noqa: BLE001
        out = f"⚠️ 执行失败：{type(e).__name__}: {e}"
    if last is not None and last.get("pending"):
        last["content"] = f"✅ 已确认执行：{spec.get('desc', '')}\n\n{out}"
        last["result"] = None
        last["pending"] = False
    st.rerun()

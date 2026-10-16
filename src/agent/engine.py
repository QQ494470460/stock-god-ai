# -*- coding: utf-8 -*-
"""AI 助手代理引擎（ReAct 风格，提示词式工具调用）。

为什么不用原生 function calling
------------------------------
`src/models/llm.DeepSeekClient.chat` 走的是裸 requests，且要兼容云端 DeepSeek 与
本地 Ollama/LM Studio（7B 模型对 function calling 支持参差不齐）。用一个**文本协议**
（ACTION / ACTION_INPUT / OBSERVATION / FINAL）对 provider 完全无关，本地 7B 也能跑。

协议（在 system prompt 里告诉模型）
----------------------------------
- 需要调用工具时，回复形如：
      ACTION: 工具名
      ACTION_INPUT: {"code": "600519"}
- 直接回答时，回复以 `FINAL:` 开头。
- 工具执行后，引擎把结果以 `RESULT: ...` 回灌，模型继续决策或给出 FINAL。

容错：本地 7B 常把 JSON 写成单引号 / 缺引号 / 包在 ```代码块 里。解析器逐层兜底。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.llm import deepseek
from src.agent.tools import get_tools, call_tool

# 工具可能在返回中附带 REPORT_FILE:<绝对路径> 标记，引擎负责收集成可下载产物
_REPORT_FILE_RE = re.compile(r"REPORT_FILE:(\S+)")
# 修改类工具在未确认时返回 PENDING_ACTION:{json} 标记，引擎据此暂停循环交还 UI
_PENDING_ACTION_RE = re.compile(r"PENDING_ACTION:(\{.*\})")


SYSTEM_PROMPT = """你是「选股神 AI」内置的智能助手，可以帮助用户做 A 股研究与决策辅助。
你可以通过调用工具来获取实时数据、跑选股、诊断系统，然后基于数据用简洁专业的中文回答用户。

# 可用工具
{tools}

# 工作流程
1. 如果用户的问题可以直接回答（例如闲聊、解释概念），直接用 `FINAL:` 回答。
2. 如果需要数据/计算，优先调用上面的工具。先回复：
   ACTION: <工具名>
   ACTION_INPUT: <JSON 参数，键名参照工具说明>
   然后用一句话说明你接下来要做什么。
3. 工具结果会以 `RESULT:` 形式返回给你，继续分析，必要时再调工具，直到能给出结论，
   最后用 `FINAL:` 开头给出给用户的正式回答（可用 Markdown，含要点、风险提示）。

# 严格要求
- 只使用上面列出的工具；不要编造工具名。
- ACTION_INPUT 必须是合法 JSON（双引号）。
- 所有结论必须基于工具返回的真实数据，不要编造数字。
- 你做的是研究辅助，任何输出都要注明「不构成投资建议」。
- 语言：简体中文。
"""


@dataclass
class AgentStep:
    action: str = ""
    action_input: Dict = field(default_factory=dict)
    observation: str = ""
    thought: str = ""


@dataclass
class AgentResult:
    final_text: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    error: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)  # 工具产出的文件路径（如 REPORT_FILE）
    pending_action: Optional[dict] = None  # 待用户确认的修改类动作提案


def _build_tool_block() -> str:
    lines = []
    for name, tool in get_tools().items():
        lines.append(f"- `{name}`：{tool.description}\n   参数：{tool.params_desc}")
    return "\n".join(lines)


def _parse_reply(text: str):
    """解析模型回复，返回 ('final', content) 或 ('action', name, args_dict) 或 ('raw', text)。"""
    raw = text.strip()
    # 去掉可能存在的 markdown 代码围栏
    fence = re.match(r"^```(?:json|text)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()

    if raw.startswith("FINAL:"):
        return ("final", raw[len("FINAL:"):].strip())

    m = re.search(r"ACTION:\s*([A-Za-z_][\w]*)", raw)
    if m:
        name = m.group(1).strip()
        # 取 ACTION_INPUT 行之后到末尾（或到下一个标记）的内容
        im = re.search(r"ACTION_INPUT:\s*(.*)", raw, re.DOTALL)
        arg_raw = im.group(1).strip() if im else "{}"
        args = _parse_args(arg_raw)
        return ("action", name, args)

    # 没有 FINAL 也没有 ACTION：当作最终回答
    return ("raw", raw)


def _parse_args(arg_raw: str) -> Dict:
    arg_raw = arg_raw.strip()
    # 1) 尝试整段 JSON
    try:
        return json.loads(arg_raw)
    except Exception:
        pass
    # 2) 尝试抽取 { } 块（非贪婪，只取第一个不含嵌套花括号的对象，
    #    避免本地 7B 在 ACTION_INPUT 之后又续写 RESULT/角色扮演内容时被吞进 JSON）
    brace = re.search(r"\{[^{}]*\}", arg_raw, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            pass
    # 3) key: value 逐行解析
    d: Dict[str, str] = {}
    for line in arg_raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip().strip('"\'')
            d[k.strip()] = v
    if d:
        return d
    # 4) 兜底：整体作为一个 value
    return {"value": arg_raw}


def run_agent(
    user_message: str,
    history: Optional[List[Dict]] = None,
    max_steps: int = 5,
    temperature: float = 0.2,
) -> AgentResult:
    """运行一轮代理。

    history: 可选的对话历史（[{role:'user'/'assistant', content:str}]），用于多轮。
    返回 AgentResult，含最终回答与逐步记录（供 UI 渲染）。
    """
    if not deepseek.is_ready:
        return AgentResult(
            error="LLM 未就绪（未配置或本地服务未启动）。请到「设置」页确认 DeepSeek/本地模型已连通。",
            final_text="⚠️ LLM 未就绪，无法启动 AI 助手。请先在设置页配置并连通模型。",
        )

    tool_block = _build_tool_block()
    system = SYSTEM_PROMPT.format(tools=tool_block)

    # 构造对话 transcript（文本协议，方便本地模型遵循）
    transcript_parts: List[str] = []
    for h in (history or []):
        role = "用户" if h.get("role") == "user" else "助手"
        transcript_parts.append(f"{role}：{h.get('content','')}")
    transcript_parts.append(f"用户：{user_message}")
    transcript = "\n".join(transcript_parts)

    steps: List[AgentStep] = []
    result = AgentResult()

    for _ in range(max_steps):
        try:
            reply = deepseek.chat(system, transcript, temperature=temperature, max_tokens=900)
        except Exception as e:  # noqa: BLE001
            result.error = f"LLM 调用失败：{type(e).__name__}: {e}"
            result.final_text = f"⚠️ LLM 调用失败：{e}"
            return result

        kind, *payload = _parse_reply(reply)

        if kind in ("final", "raw"):
            result.final_text = payload[0] if payload else ""
            result.steps = steps
            _ingest_artifacts(result, result.final_text)
            return result

        # action
        _, name, args = kind, payload[0], payload[1]
        observation = call_tool(name, args)

        # 修改类动作：检测到待确认标记即暂停循环，把提案交还 UI 等用户确认
        _pa = _PENDING_ACTION_RE.search(observation)
        if _pa:
            try:
                result.pending_action = json.loads(_pa.group(1))
            except Exception:  # noqa: BLE001
                result.pending_action = None
            result.final_text = observation
            result.steps = steps
            return result

        step = AgentStep(action=name, action_input=args, observation=observation)
        steps.append(step)
        _ingest_artifacts(result, observation)

        transcript += (
            f"\n助手：ACTION: {name}\nACTION_INPUT: {json.dumps(args, ensure_ascii=False)}\n"
            f"RESULT: {observation}\n"
        )

    # 超过步数仍无 FINAL：让模型基于已有观测做总结
    try:
        final_prompt = transcript + "\n助手：请基于以上信息，用 FINAL: 开头给出最终回答。"
        final = deepseek.chat(system, final_prompt, temperature=temperature, max_tokens=900)
        _, *fp = _parse_reply(final)
        result.final_text = fp[0] if fp else final
    except Exception as e:  # noqa: BLE001
        result.final_text = "（已达到最大工具调用步数）" + "\n".join(
            s.observation for s in steps
        )
        result.error = f"最终总结失败：{e}"
    result.steps = steps
    _ingest_artifacts(result, result.final_text)
    return result


def _ingest_artifacts(result: AgentResult, text: str):
    """从工具返回/最终回答中抽取 REPORT_FILE 标记，收集为可下载产物路径。"""
    if not text:
        return
    for m in _REPORT_FILE_RE.findall(text):
        if m not in result.artifacts:
            result.artifacts.append(m)

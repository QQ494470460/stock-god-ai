# -*- coding: utf-8 -*-
"""AI 助手「改前先确认」闸门的 UI 层回归测试（Streamlit AppTest，无浏览器）。

背景（真踩过的坑）：
`agent_page.py` 旧实现里，点「✅ 确认执行 / ❌ 取消」只是把决策**写进
session_state**，然后靠「下一次重跑」里的顶部守卫去处理。但 Streamlit 的语义是
「点击本身就会触发一次重跑」——那次重跑里，页面顶部的决策守卫早就在按钮回调
**之前**以空值判定过了。于是决策被存下来却**永远没人处理**：按钮点了没反应，
配置纹丝不动。

这个 bug **只有真的点一下才会暴露**：pytest 里没有 UI 点击，所以长期无人发现。
本测试用 streamlit.testing.v1.AppTest 真正“点”按钮，把它钉死。

要证明的：
1. 点「确认执行」→ 配置真的落盘为新值；
2. 点「取消」→ 配置纹丝不动（闸门的另一半）；
3. 未点任何按钮（仅预览）→ 配置不动。
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit.testing.v1")


# ---------------------------------------------------------------------------
# 极简宿主脚本：只渲染 agent 页 + 注入一个待确认提案，避开 app.py 的整页重量级依赖。
# ---------------------------------------------------------------------------
_HOST = r'''
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT))

from src.ui.agent_page import _render_result

# 强制 LLM 就绪 + 清空历史，让 render 只走「渲染待确认提案」这一条路
import src.ui.agent_page as ap
ap.deepseek = type("D", (), {{"is_ready": True}})()

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

PA = {{
    "tool": "set_config",
    "args": {{"key": "agent.screen_fetch_workers", "value": {target}{approve_kv}}},
    "desc": "测试：把筛选并发改为 {target}",
    "impact": "测试用",
    "risk": "低",
}}


class R:
    pending_action = PA
    steps = None
    artifacts = None
    error = None
    final_text = "（测试）"


_render_result(st, R(), "（测试）")
'''


def _write_host(tmp_path, target, with_approve=True):
    """with_approve=False 用于负向对照：提案里**故意不带** approve 键。"""
    p = tmp_path / "host_app.py"
    approve_kv = ', "approve": True' if with_approve else ""
    p.write_text(
        _HOST.format(root=str(ROOT), target=target, approve_kv=approve_kv),
        encoding="utf-8",
    )
    return str(p)


def _isolate_state(monkeypatch, tmp_path, start_value):
    """把 set_config 的落盘动作指到临时 state 文件；真实 config/agent_state.json 绝不碰。

    set_config 经 `_CONFIG_SETTERS['agent.screen_fetch_workers']['apply']`
    调 `save_agent_state`（由 config.settings 导入到 tools 命名空间）。
    这里把 tools 里的 save_agent_state 换成写 tmp 的实现即可。
    """
    from src.agent import tools as tools_mod

    state_file = tmp_path / "agent_state.json"
    state_file.write_text(json.dumps({"screen_fetch_workers": start_value}), encoding="utf-8")

    def fake_save(value):
        state_file.write_text(json.dumps({"screen_fetch_workers": int(value)}), encoding="utf-8")
        return int(value)

    monkeypatch.setattr(tools_mod, "save_agent_state", fake_save, raising=True)
    return state_file


def _read(state_file):
    return json.loads(state_file.read_text(encoding="utf-8"))["screen_fetch_workers"]


def _run_app(tmp_path, target, with_approve=True):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(
        _write_host(tmp_path, target, with_approve=with_approve), default_timeout=60)
    at.run()
    return at


def test_confirm_executes_and_writes(monkeypatch, tmp_path):
    """点「确认执行」→ 配置真的变为新值。"""
    start, target = 8, 6
    state_file = _isolate_state(monkeypatch, tmp_path, start)

    at = _run_app(tmp_path, target)
    assert not at.exception, f"渲染即异常：{at.exception}"

    confirm = [b for b in at.button if b.label == "✅ 确认执行"]
    assert confirm, f"未渲染出确认按钮；按钮={[b.label for b in at.button]}"

    # 关键：真正“点”它（Streamlit 会重跑脚本，触发点击处内的就地处理）
    confirm[0].click()
    at.run()
    assert not at.exception, f"点击后异常：{at.exception}"

    got = _read(state_file)
    assert got == target, (
        f"点确认后配置应变为 {target}，实际 {got} —— "
        "说明确认按钮没落地（正是那个「点了没反应」的 bug 回来了）")


def test_cancel_does_not_execute(monkeypatch, tmp_path):
    """点「取消」→ 配置纹丝不动（闸门的另一半）。"""
    start, target = 8, 4
    state_file = _isolate_state(monkeypatch, tmp_path, start)

    at = _run_app(tmp_path, target)
    assert not at.exception

    cancel = [b for b in at.button if b.label == "❌ 取消"]
    assert cancel, f"未渲染出取消按钮；按钮={[b.label for b in at.button]}"
    cancel[0].click()
    at.run()
    assert not at.exception, f"点击后异常：{at.exception}"

    got = _read(state_file)
    assert got == start, (
        f"点取消后配置不该变（应仍是 {start}），实际 {got} —— 取消按钮竟然执行了！")


def test_preview_does_not_write(monkeypatch, tmp_path):
    """仅渲染预览、不点任何按钮 → 配置绝不落盘。"""
    start, target = 8, 5
    state_file = _isolate_state(monkeypatch, tmp_path, start)

    at = _run_app(tmp_path, target)
    assert not at.exception

    got = _read(state_file)
    assert got == start, f"仅预览就落盘了（{start} -> {got}），预览闸门失效"


def test_confirm_forces_approve_even_if_proposal_omits_it(monkeypatch, tmp_path):
    """负向对照：提案 args 里**没有** approve 键时，确认仍必须真执行。

    用户点「确认执行」这件事本身就是批准。若实现只是原样转发 spec["args"]，
    则工具会再次走进「未获 approve → 只返回预览」分支，表现为
    **点了确认却什么都没发生**（静默失效，本项目最忌的那类）。
    所以 `_handle_pending_decision` 必须无条件置 approve=True。
    """
    start, target = 8, 7
    state_file = _isolate_state(monkeypatch, tmp_path, start)

    at = _run_app(tmp_path, target, with_approve=False)
    assert not at.exception, f"渲染即异常：{at.exception}"

    confirm = [b for b in at.button if b.label == "✅ 确认执行"]
    assert confirm, "未渲染出确认按钮"
    confirm[0].click()
    at.run()
    assert not at.exception, f"点击后异常：{at.exception}"

    got = _read(state_file)
    assert got == target, (
        f"提案缺 approve 时，点确认仍应落到 {target}，实际 {got} —— "
        "说明确认被当成又一次预览吞掉了")

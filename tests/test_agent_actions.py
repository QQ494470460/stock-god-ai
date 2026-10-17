# -*- coding: utf-8 -*-
"""AI 助手「改前先确认」变更能力（确认闸 + 四个修改类工具）的单元测试。

覆盖：
- 确认闸机制：未确认只返回 ``PENDING_ACTION`` 预览、绝不落盘/改文件/下单；确认后才真正执行。
- set_config / set_strategy_param 白名单 + 数值校验 + 落盘。
- edit_file 路径白名单 + 后缀白名单 + diff 预览 + 确认后写盘。
- propose_trade 仅 PAPER 纸账户模拟、留痕、绝不实盘。
- 引擎的 pending_action 检测正则。

含负向对照：白名单拒绝、非法参数、确认前不落盘，证明「闸」真的有牙齿。
所有落盘目标都重定向到 tmp_path，绝不触碰生产状态文件。
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config.settings as settings_mod
from src.agent import tools as agent_tools
from src.agent.engine import _PENDING_ACTION_RE, SYSTEM_PROMPT, _build_tool_block


PENDING = agent_tools.PENDING_ACTION_PREFIX


@pytest.fixture(autouse=True)
def _isolated_config_files(tmp_path, monkeypatch):
    # 把落盘目标重定向到临时目录，绝不碰生产五大件/state 文件
    monkeypatch.setattr(settings_mod, "_AGENT_STATE_FILE", tmp_path / "agent_state.json")
    monkeypatch.setattr(settings_mod, "_ML_STATE_FILE", tmp_path / "ml_state.json")
    monkeypatch.setattr(settings_mod, "_STRATEGY_OVERRIDE_FILE", tmp_path / "strategy_overrides.json")
    monkeypatch.setattr(agent_tools, "DATA_DIR", tmp_path)
    # 避免 _name_of 触发网络（名称映射）
    monkeypatch.setattr(agent_tools, "_NAME_MAP", {})
    yield


def _parse_pending(text: str) -> dict:
    m = re.search(re.escape(PENDING) + r"(\{.*\})", text, re.S)
    assert m, f"期望包含 PENDING_ACTION 标记，实际：{text[:200]}"
    return json.loads(m.group(1))


# ---------------- 注册表 / 分发 ----------------

def test_registry_contains_new_tools():
    for name in ("set_config", "set_strategy_param", "edit_file", "propose_trade"):
        assert name in agent_tools.REGISTRY
        assert agent_tools.call_tool(name, {}) is not None  # 不应抛异常


# ---------------- 确认闸机制 ----------------

def test_set_config_whitelist_rejects_unknown_key():
    out = agent_tools.set_config({"key": "evil.key", "value": "1"})
    assert PENDING not in out
    assert "不在可改白名单" in out


def test_set_config_preview_does_not_write():
    # 用与当前值**不同**的值，否则会命中「空操作」分支（见下一条用例）
    out = agent_tools.set_config({"key": "agent.screen_fetch_workers", "value": "5"})
    assert PENDING in out
    spec = _parse_pending(out)
    assert spec["tool"] == "set_config"
    assert spec["args"]["approve"] is True
    # 确认前不应落盘：临时目录里没有任何 state 文件
    assert not (settings_mod._AGENT_STATE_FILE).exists()


def test_set_config_noop_when_value_unchanged():
    """值没变就不要弹提案。

    回归背景：实测对助手说「把筛选并发改成 8」，而当前本来就是 8，它照样弹出
    「从 8 改为 8」的确认框；用户点「确认」后什么都不会变 —— 这是「AI 助手
    点了没用／啥也干不了」最直接的体感来源之一。
    """
    cur = agent_tools._read_config_current("agent.screen_fetch_workers")
    out = agent_tools.set_config(
        {"key": "agent.screen_fetch_workers", "value": str(cur)}
    )
    assert PENDING not in out, "值没变仍然弹了确认提案"
    assert "无需修改" in out
    assert not (settings_mod._AGENT_STATE_FILE).exists()


def test_set_config_approve_writes_and_clamps():
    out = agent_tools.set_config({"key": "agent.screen_fetch_workers", "value": "99", "approve": True})
    assert "已修改" in out
    data = json.loads((settings_mod._AGENT_STATE_FILE).read_text(encoding="utf-8"))
    # 上限夹到 16
    assert data["screen_fetch_workers"] == 16


def test_set_config_approve_bool_ml():
    out = agent_tools.set_config({"key": "ml.use_ensemble", "value": True, "approve": True})
    assert "已修改" in out
    data = json.loads((settings_mod._ML_STATE_FILE).read_text(encoding="utf-8"))
    assert data["use_ensemble"] is True


# ---------------- set_strategy_param ----------------

def test_set_strategy_param_whitelist_rejects():
    out = agent_tools.set_strategy_param({"key": "not_allowed", "value": 1})
    assert PENDING not in out
    assert "不在可改策略键" in out


def test_set_strategy_param_preview_no_write():
    out = agent_tools.set_strategy_param({"key": "min_score", "value": 60})
    assert PENDING in out
    assert not (settings_mod._STRATEGY_OVERRIDE_FILE).exists()


def test_set_strategy_param_approve_writes():
    out = agent_tools.set_strategy_param({"key": "min_score", "value": 60, "approve": True})
    assert "已修改" in out
    data = json.loads((settings_mod._STRATEGY_OVERRIDE_FILE).read_text(encoding="utf-8"))
    assert data["min_score"] == 60.0


# ---------------- edit_file ----------------

def test_edit_file_whitelist_rejects_outside_roots(tmp_path, monkeypatch):
    # 把白名单改成临时目录，但传入临时目录之外的路径 → 应被拒
    monkeypatch.setattr(agent_tools, "ALLOWED_EDIT_ROOTS", (str(tmp_path),))
    outside = str(Path(tmp_path).parent / "escape.py")
    out = agent_tools.edit_file({"path": outside, "old_text": "x", "new_text": "y"})
    assert PENDING not in out
    assert "出于安全" in out


def test_edit_file_suffix_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_tools, "ALLOWED_EDIT_ROOTS", (str(tmp_path),))
    f = tmp_path / "bad.exe"
    f.write_text("hello", encoding="utf-8")
    out = agent_tools.edit_file({"path": str(f), "old_text": "hello", "new_text": "bye"})
    assert PENDING not in out
    assert "只允许编辑" in out


def test_edit_file_preview_and_approve(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_tools, "ALLOWED_EDIT_ROOTS", (str(tmp_path),))
    f = tmp_path / "mod.py"
    f.write_text("X = OLD\n", encoding="utf-8")
    # 预览：不改文件，只返回 diff 预览
    out = agent_tools.edit_file({"path": str(f), "old_text": "OLD", "new_text": "NEW"})
    assert PENDING in out
    assert f.read_text(encoding="utf-8") == "X = OLD\n"
    # 确认：真正写盘
    out2 = agent_tools.edit_file({"path": str(f), "old_text": "OLD", "new_text": "NEW", "approve": True})
    assert "已修改" in out2
    assert f.read_text(encoding="utf-8") == "X = NEW\n"


def test_edit_file_old_text_not_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_tools, "ALLOWED_EDIT_ROOTS", (str(tmp_path),))
    f = tmp_path / "dup.py"
    f.write_text("A\nA\n", encoding="utf-8")
    out = agent_tools.edit_file({"path": str(f), "old_text": "A", "new_text": "B"})
    assert PENDING not in out
    assert "无法唯一定位" in out


# ---------------- propose_trade（仅 PAPER） ----------------

def test_propose_trade_validation():
    out = agent_tools.propose_trade({"code": "600519", "side": "hold", "quantity": 100})
    assert PENDING not in out
    assert "buy 或 sell" in out

    out2 = agent_tools.propose_trade({"code": "600519", "side": "buy", "quantity": 0})
    assert PENDING not in out2
    assert "正的" in out2


def test_propose_trade_preview_no_write():
    out = agent_tools.propose_trade({"code": "600519", "side": "buy", "quantity": 100})
    assert PENDING in out
    spec = _parse_pending(out)
    assert spec["tool"] == "propose_trade"
    assert spec["args"]["approve"] is True
    assert not (agent_tools.DATA_DIR / "agent_trade_journal.jsonl").exists()


def test_propose_trade_approve_writes_paper_only():
    out = agent_tools.propose_trade({"code": "600519", "side": "buy", "quantity": 100, "approve": True})
    assert "已" in out and "纸账户" in out
    jf = agent_tools.DATA_DIR / "agent_trade_journal.jsonl"
    assert jf.exists()
    lines = jf.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    # 铁证：只为 PAPER 模拟，绝不实盘
    assert rec["mode"] == "PAPER_SIM"
    assert rec["code"] == "600519"
    assert rec["side"] == "buy"
    assert rec["quantity"] == 100
    assert rec["by"] == "ai_assistant"


# ---------------- 引擎 pending_action 检测 ----------------

def test_engine_detects_pending_action_marker():
    obs = "一些文字\n" + PENDING + json.dumps({"tool": "edit_file", "args": {"approve": True}}, ensure_ascii=False)
    m = _PENDING_ACTION_RE.search(obs)
    assert m is not None
    spec = json.loads(m.group(1))
    assert spec["tool"] == "edit_file"


def test_engine_no_false_positive():
    assert _PENDING_ACTION_RE.search("正常文本没有标记") is None


# ---------------- SYSTEM_PROMPT 模板合法性（回归） ----------------

def test_system_prompt_formats_without_keyerror():
    """回归：示例中含字面 JSON 花括号，SYSTEM_PROMPT.format(tools=...) 必须不抛 KeyError。

    曾因示例里的 ``{"key": ...}`` 被 .format 当作占位符 → KeyError: '"key"'，
    导致 run_agent 在第一步就崩溃。所有字面花括号必须转义为 {{ }}。

    直接复用引擎生产路径 `_build_tool_block()` + `.format()`，确保与运行时一致。
    """
    tool_block = _build_tool_block()
    # 不应抛出 KeyError / IndexError
    rendered = SYSTEM_PROMPT.format(tools=tool_block)
    assert "set_config" in rendered
    assert "FINAL:" in rendered
    # 转义后的字面花括号应正常还原（示例里出现真实 JSON 示例）
    assert "agent.screen_fetch_workers" in rendered



"""AI 助手（agent）回归测试。

覆盖：
1. 协议解析器 `_parse_reply` / `_parse_args` 的多种模型输出形态（含本地 7B 的
   代码块包裹、单引号 JSON、续写 RESULT 等不规范输出）。
2. 代理引擎 `run_agent` 的主链路（用打桩的 LLM 模拟 ReAct 循环，
   避免依赖本地 Ollama、也不打真网络）。
3. 工具分派：已知/未知工具、错误参数自包容。

负向对照：把 LLM 桩成只会胡说 / 永不 FINAL，断言引擎**不崩**、能兜底返回。
"""

from src.agent import engine
from src.agent.engine import run_agent, _parse_reply, _parse_args


def test_parse_reply_action_normal():
    rep = "我先查一下。\nACTION: analyze_stock\nACTION_INPUT: {\"code\": \"600519\"}"
    assert _parse_reply(rep) == ("action", "analyze_stock", {"code": "600519"})


def test_parse_reply_action_in_code_fence():
    rep = "```\nACTION: get_realtime\nACTION_INPUT: {\"code\":\"000001\"}\n```"
    assert _parse_reply(rep) == ("action", "get_realtime", {"code": "000001"})


def test_parse_reply_final():
    assert _parse_reply("FINAL: 结论如下") == ("final", "结论如下")


def test_parse_reply_raw_fallback():
    # 没有 FINAL / ACTION：当作最终回答，不崩
    assert _parse_reply("你好，我可以帮你分析股票。")[0] == "raw"


def test_parse_args_json():
    assert _parse_args('{"code": "600519"}') == {"code": "600519"}


def test_parse_args_single_quote_json():
    # 本地模型常输出单引号 JSON：尽力解析（至少不抛异常）
    out = _parse_args("{'code': '600519'}")
    assert isinstance(out, dict)


def test_parse_args_leading_text_then_brace():
    # 模型在 ACTION_INPUT 之后又续写 RESULT 等脏内容时，只取第一个 JSON 对象
    raw = '{"code": "600519"}\nRESULT: {"code": "600519", "name": "茅台"}'
    assert _parse_args(raw) == {"code": "600519"}


def test_parse_args_key_value_fallback():
    assert _parse_args("code: 600519\npool: hs300") == {"code": "600519", "pool": "hs300"}


def test_call_tool_dispatch_and_unknown():
    # 已知工具（只读本地文件，无网络）
    out = engine.call_tool("diagnose_evolution", {})
    assert "自我进化引擎诊断" in out
    # 未知工具：自包容报错，不抛异常
    out2 = engine.call_tool("no_such_tool", {})
    assert "未知工具" in out2


def _force_ready(monkeypatch, value: bool):
    """is_ready 是只读 property，用覆盖类属性 property 的方式强制设置。"""
    monkeypatch.setattr(
        type(engine.deepseek), "is_ready", property(lambda self: value)
    )


def test_run_agent_full_loop_with_stub(monkeypatch):
    """LLM 桩：第一次给 ACTION，第二次给 FINAL → 跑完一轮 ReAct。"""
    calls = {"n": 0}

    def fake_chat(system, user, temperature=None, max_tokens=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return "ACTION: diagnose_evolution\nACTION_INPUT: {}"
        return "FINAL: 诊断完成，引擎在线。"

    monkeypatch.setattr(engine.deepseek, "chat", fake_chat)
    _force_ready(monkeypatch, True)

    res = run_agent("诊断进化引擎", max_steps=5)
    assert calls["n"] == 2, calls
    assert len(res.steps) == 1
    assert res.steps[0].action == "diagnose_evolution"
    assert "自我进化引擎诊断" in res.steps[0].observation
    assert res.final_text == "诊断完成，引擎在线。"


def test_run_agent_garbage_reply_no_crash(monkeypatch):
    """负向：LLM 永远胡说 → 引擎兜底返回，不抛异常。"""
    monkeypatch.setattr(engine.deepseek, "chat",
                        lambda *a, **k: "今天的天气真不错呀")
    _force_ready(monkeypatch, True)
    res = run_agent("分析 600519", max_steps=3)
    assert res.error is None or "失败" in (res.error or "")
    assert res.final_text  # 有返回


def test_run_agent_never_final_falls_back(monkeypatch):
    """负向：LLM 永远只给 ACTION（且参数会让工具报错）→ 达步数后兜底总结。"""
    calls = {"n": 0}

    def fake_chat(system, user, temperature=None, max_tokens=None):
        calls["n"] += 1
        return 'ACTION: analyze_stock\nACTION_INPUT: {"code": "不存在的999"}'

    monkeypatch.setattr(engine.deepseek, "chat", fake_chat)
    _force_ready(monkeypatch, True)
    res = run_agent("分析 999", max_steps=2)
    # 2 次 ACTION 循环 + 1 次兜底总结 = 3 次 chat；工具步骤 2 个
    assert calls["n"] == 3
    assert len(res.steps) == 2
    assert res.final_text  # 兜底总结非空


def test_run_agent_llm_not_ready(monkeypatch):
    """LLM 未就绪时直接返回可读提示，不抛异常。"""
    _force_ready(monkeypatch, False)
    res = run_agent("随便问", max_steps=2)
    assert res.error is not None
    assert "未就绪" in res.final_text

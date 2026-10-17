# -*- coding: utf-8 -*-
"""AI 助手「护栏」回归：参数归一 / 失败观测 / 不许编造。

全部来自实测故障，不是假想：

1. 问「今天大盘怎么样」→ 模型调 ``get_realtime('sh')``、``get_realtime('上证指数')``
   全失败，再调 ``run_daily_pick(pool='shanghai')`` 触发
   ``ValueError: 未知股票池``，最后答成「大盘暂无评分 ≥ 55 的股票符合条件」
   —— 把报错当成「查到了、结果是空」，凭空捏造。
2. 说「把筛选并发改成 8」，而当前本来就是 8 → 仍弹「从 8 改为 8」的提案，
   用户点确认后什么都没变。
3. 指数 ``000001`` 对个股是平安银行、对上证指数是 ``sh000001`` ——
   同一个数字两个标的，按首位猜市场前缀必然张冠李戴。
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.agent import tools as agent_tools
from src.agent import engine as agent_engine
from src.agent.tools import TOOL_FAILED_MARK, _pool_alias, _normalize_index


# ---------------------------------------------------------------------------
# 参数归一
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expect", [
    ("hs300", "hs300"),
    ("沪深300", "hs300"),
    ("300", "hs300"),
    ("CSI300", "hs300"),
    ("zz500", "zz500"),
    ("中证500", "zz500"),
    ("中证1000", "zz1000"),
    ("全市场", "all"),
    ("hs300_zz500", "hs300_zz500"),
    ("沪深300+中证500", "hs300_zz500"),
])
def test_pool_alias_normalizes(raw, expect):
    assert _pool_alias(raw) == expect


def test_pool_alias_unknown_returns_none():
    """不认识的池名不硬猜（保持 fail-loud），但也不能把它当成合法池。"""
    assert _pool_alias("shanghai") is None
    assert _pool_alias("") is None


def test_index_symbol_prefix_not_guessed_by_digit():
    """000001 作指数必须是 sh000001（上证），不能沿用个股的 sz000001（平安银行）。"""
    sym, err = _normalize_index("000001")
    assert err == ""
    assert sym == "sh000001"
    assert _normalize_index("大盘")[0] == "sh000001"
    assert _normalize_index("创业板")[0] == "sz399006"
    assert _normalize_index("沪深300")[0] == "sh000300"


# ---------------------------------------------------------------------------
# 工具层：失败必须带机器标记
# ---------------------------------------------------------------------------

def test_get_realtime_rejects_index_and_points_to_get_index():
    out = agent_tools.call_tool("get_realtime", {"code": "上证指数"})
    assert out.startswith(TOOL_FAILED_MARK)
    assert "get_index" in out, "失败时必须告诉模型该用哪个工具，否则它会反复瞎试"


def test_registry_has_get_index():
    assert "get_index" in agent_tools.get_tools()


def test_get_index_outputs_points_and_ma(monkeypatch):
    monkeypatch.setattr(
        agent_tools.fetcher, "get_index_realtime",
        lambda sym: {"名称": "上证指数", "最新价": 3716.28, "涨跌幅": 0.24,
                     "今开": 3723.8, "最高": 3750.77, "最低": 3681.28,
                     "昨收": 3707.48, "成交额": 73247519.0},
    )
    closes = [3600 + i for i in range(80)]
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=80),
        "close": closes, "open": closes, "high": closes,
        "low": closes, "volume": [1e6] * 80,
    })
    monkeypatch.setattr(agent_tools.fetcher, "get_index_kline", lambda *a, **k: df)

    out = agent_tools.call_tool("get_index", {"index": "大盘"})
    assert not out.startswith(TOOL_FAILED_MARK)
    assert "上证指数" in out and "3716.28" in out and "MA20" in out


def test_run_daily_pick_maps_error_status_to_failure(monkeypatch):
    """generate_daily_picks 吞掉异常返回 status=error —— 工具不能当成「空结果」。"""
    monkeypatch.setattr(
        agent_tools, "generate_daily_picks",
        lambda pool, max_picks: (pd.DataFrame(), {
            "status": "error", "error": "未知股票池 'shanghai'", "min_score": 55.0,
        }),
    )
    out = agent_tools.call_tool("run_daily_pick", {"pool": "zz500"})
    assert out.startswith(TOOL_FAILED_MARK), "失败被伪装成了「没有达标标的」"
    assert "未知股票池" in out


# ---------------------------------------------------------------------------
# 引擎层：一次成功观测都没有 → 不许给结论
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    """按脚本依次返回多条回复的 LLM 替身。"""

    is_ready = True

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, *a, **k):
        self.calls += 1
        return self.replies.pop(0)


def test_all_failed_calls_yield_honest_failure_not_fabrication(monkeypatch):
    """回归：工具全失败时，最终回答必须是「没取到数据」，不能是编的结论。"""
    llm = _ScriptedLLM([
        'ACTION: run_daily_pick\nACTION_INPUT: {"pool": "shanghai"}',
        # 真实模型在这里会把报错读成「没有符合条件的股票」
        "FINAL: 今天大盘暂无评分 ≥ 55.0 的股票符合条件，建议观望。",
    ])
    monkeypatch.setattr(agent_engine, "deepseek", llm)
    monkeypatch.setattr(
        agent_engine, "call_tool",
        lambda name, args: f"{TOOL_FAILED_MARK}选股失败：未知股票池 'shanghai'",
    )

    r = agent_engine.run_agent("今天大盘怎么样？", max_steps=3)
    assert "没能取到数据" in r.final_text
    assert "暂无评分" not in r.final_text, "失败观测被当成结论了"
    assert r.error


def test_successful_observation_keeps_model_answer(monkeypatch):
    """有成功观测时不得被失败兜底逻辑误伤。"""
    llm = _ScriptedLLM([
        'ACTION: get_index\nACTION_INPUT: {"index": "上证指数"}',
        "FINAL: 上证指数 3716.28，涨 0.24%。不构成投资建议。",
    ])
    monkeypatch.setattr(agent_engine, "deepseek", llm)
    monkeypatch.setattr(agent_engine, "call_tool",
                        lambda name, args: "## 上证指数（sh000001）\n- 最新点位：3716.28")

    r = agent_engine.run_agent("大盘怎么样", max_steps=3)
    assert "3716.28" in r.final_text
    assert r.error is None

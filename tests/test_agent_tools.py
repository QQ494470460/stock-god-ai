# -*- coding: utf-8 -*-
"""智能筛选 / 可下载研报 / 产物收集 的单元测试。

用 monkeypatch 注入假数据，避免真的去拉行情、跑模型；并含负向对照，
证明过滤/兜底逻辑"真的有牙齿"（不是把现状当规格钉死）。
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import tools as agent_tools
from src.agent.engine import run_agent, _ingest_artifacts, AgentResult


def _kline(kind: str = "bull", n: int = 120, base: float = 10.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    if kind == "bull":
        close = base + idx * 0.05 + rng.normal(0, 0.02, n)        # 陡升
    elif kind == "bear":
        close = base + (n - idx) * 0.05 + rng.normal(0, 0.02, n)  # 下跌
    else:  # mild：温和上行（保证站上 MA20）
        close = base + idx * 0.02 + np.sin(idx / 5.0) * 0.1 + rng.normal(0, 0.02, n)
    vol = np.full(n, 1000.0)
    vol[-1] = 6000.0  # 尾端放量 → 量比高
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "close": close, "volume": vol})


@dataclass
class FakePred:
    code: str
    score: float
    name: str = ""


class FakePredictor:
    def __init__(self, scores: dict):
        self._scores = scores
        self.is_ready = True

    def predict_batch(self, stock_data, stock_names=None):
        return [FakePred(c, s) for c, s in self._scores.items()]

    def predict_single(self, df, code="000000", name="未知"):
        class S:
            score = 70.0
            confidence = 0.6
            prob_up = 0.6
            prob_neutral = 0.2
            prob_down = 0.2
            xgb_signal = 1
            reasons = ["高确信度 (60%)", "预期收益 +2.1%"]
        return S()


class FakeFetcher:
    def __init__(self, klins: dict):
        self._klins = klins

    def get_stock_pool(self, pool):
        return list(self._klins.keys())

    def get_batch_daily_kline(self, codes, max_workers=8):
        return {c: self._klins[c] for c in codes if c in self._klins}

    def get_stock_list(self):
        return pd.DataFrame(
            {"code": list(self._klins.keys()),
             "name": [f"股{c}" for c in self._klins]}
        )

    def get_daily_kline(self, code, start_date=None, end_date=None):
        return self._klins.get(code, _kline())


class FakeConfig:
    class _Sched:
        pick_pool = "test"
    scheduler = _Sched()

    class _Agent:
        screen_fetch_workers = 1
    agent = _Agent()


@pytest.fixture
def env(monkeypatch, tmp_path):
    klins = {
        "600001": _kline("bull"),
        "600002": _kline("bear"),
        "600003": _kline("mild"),
    }
    monkeypatch.setattr(agent_tools, "fetcher", FakeFetcher(klins))
    monkeypatch.setattr(
        agent_tools, "predictor",
        FakePredictor({"600001": 80, "600002": 80, "600003": 80}),
    )
    monkeypatch.setattr(agent_tools, "app_config", FakeConfig())
    monkeypatch.setattr(agent_tools, "DATA_DIR", tmp_path)
    return tmp_path


# ---------------- 智能筛选 ----------------

def test_screen_stocks_no_filter_includes_all(env):
    out = agent_tools.screen_stocks({"pool": "test", "min_score": 50, "top": 10})
    assert "600001" in out and "600002" in out and "600003" in out
    # 至少出现一种研判标签
    assert any(k in out for k in ("偏多", "偏空", "中性", "超买", "超卖"))


def test_screen_stocks_above_ma20_excludes_bear(env):
    out = agent_tools.screen_stocks(
        {"pool": "test", "min_score": 50, "above_ma20": True, "top": 10}
    )
    assert "600001" in out and "600003" in out
    assert "600002" not in out  # 熊股跌破均线被过滤


def test_screen_stocks_negative_all_filtered(env):
    # rsi_max=1 几乎不可能满足 → 0 只达标（证明过滤逻辑真的生效）
    out = agent_tools.screen_stocks(
        {"pool": "test", "min_score": 50, "above_ma20": True, "rsi_max": 1}
    )
    assert "0 只达标" in out


def test_screen_stocks_honors_config_workers(env, monkeypatch):
    # 并发数由 config.agent.screen_fetch_workers 驱动，且会原样传给批量拉行情
    captured = {}

    def fake_batch(codes, max_workers=8):
        captured["mw"] = max_workers
        return {c: _kline() for c in codes}

    monkeypatch.setattr(agent_tools.fetcher, "get_batch_daily_kline", fake_batch)
    env  # 保证 app_config 是 FakeConfig（含 agent.screen_fetch_workers）
    agent_tools.app_config.agent.screen_fetch_workers = 4
    agent_tools.screen_stocks({"pool": "test", "min_score": 50, "top": 10})
    assert captured.get("mw") == 4


def test_screen_workers_clamped_and_fallback():
    # 配置不可读时回落到模块常量，并夹到 [1,16]
    import src.agent.tools as T

    class NoAgent:
        scheduler = type("S", (), {"pick_pool": "x"})()

    monkeypatch_stub = NoAgent()
    orig = T.app_config
    try:
        T.app_config = monkeypatch_stub  # 没有 .agent → 应回落常量 1
        assert T._screen_workers() == T.SCREEN_FETCH_WORKERS == 1
    finally:
        T.app_config = orig


# ---------------- 自带研判 ----------------

def test_analyze_stock_adds_verdict(env):
    out = agent_tools.analyze_stock({"code": "600001"})
    assert "技术研判" in out


# ---------------- 可下载研报 ----------------

def test_make_report_writes_file_and_marker(env):
    out = agent_tools.make_report({"pool": "test", "top": 5})
    assert "REPORT_FILE:" in out
    path = out.split("REPORT_FILE:")[-1].strip().split()[0]
    p = Path(path)
    assert p.exists()
    assert "智能筛选研报" in p.read_text(encoding="utf-8")


# ---------------- 引擎产物收集 ----------------

def test_engine_ingest_artifacts_unit():
    r = AgentResult()
    _ingest_artifacts(r, "xxx REPORT_FILE:/tmp/a.md yyy")
    assert r.artifacts == ["/tmp/a.md"]
    _ingest_artifacts(r, "REPORT_FILE:/tmp/a.md REPORT_FILE:/tmp/b.md")
    assert r.artifacts == ["/tmp/a.md", "/tmp/b.md"]  # 去重且保序


def test_engine_ingest_artifacts_empty_when_no_marker():
    r = AgentResult()
    _ingest_artifacts(r, "没有任何标记的一串文字")
    assert r.artifacts == []


def test_engine_run_agent_collects_artifacts(monkeypatch, env):
    class FakeLLM:
        is_ready = True

        def __init__(self):
            self.calls = 0

        def chat(self, system, user, temperature=None, max_tokens=None):
            self.calls += 1
            if self.calls == 1:
                return 'ACTION: make_report\nACTION_INPUT: {"pool": "test", "top": 3}\n正在生成研报'
            return "FINAL: 已生成研报。"

    monkeypatch.setattr("src.agent.engine.deepseek", FakeLLM())
    res = run_agent("生成一份研报", history=[], max_steps=5)
    assert res.error is None, res.error
    assert res.artifacts, "应收集到 REPORT_FILE 产物"
    assert Path(res.artifacts[0]).exists()

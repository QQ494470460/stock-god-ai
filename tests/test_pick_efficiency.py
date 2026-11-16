"""限速器（``DataFetcher._throttle``）与「宁缺毋滥」选股闸门的契约测试。

两件事都是"效率/质量"的直接落点：
  · 限速：旧实现无条件 ``sleep(1.5)`` 再发请求，425 只白等 ~80 秒；
  · 闸门：默认关（保持与回测一致的分散度），开启后只买达标票。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.daily_pick import _apply_score_threshold
from src.data.fetcher import DataFetcher


# --------------------------------------------------------------- 限速器

def test_throttle_skips_wait_when_interval_already_elapsed(monkeypatch):
    """两次调用之间已过足够久（等价于"请求本身很慢"）→ 完全不再睡。"""
    f = DataFetcher.__new__(DataFetcher)      # 不碰磁盘/日志，只测算法
    slept = []
    monkeypatch.setattr("src.data.fetcher.time.sleep", lambda s: slept.append(s))
    clock = {"t": 100.0}
    monkeypatch.setattr("src.data.fetcher.time.monotonic", lambda: clock["t"])

    f._last_net_ts = 0.0
    assert f._throttle(1.5) == 0.0            # 首次：不需要等
    clock["t"] = 101.8                        # 1.8s 后（网络本身耗时 > 间隔）
    assert f._throttle(1.5) == 0.0
    assert slept == []


def test_throttle_only_waits_the_shortfall(monkeypatch):
    """间隔没走满 → 只补差额，而不是固定睡满 interval。"""
    f = DataFetcher.__new__(DataFetcher)
    slept = []
    monkeypatch.setattr("src.data.fetcher.time.sleep", lambda s: slept.append(s))
    clock = {"t": 200.0}
    monkeypatch.setattr("src.data.fetcher.time.monotonic", lambda: clock["t"])

    f._last_net_ts = 0.0
    f._throttle(1.5)
    clock["t"] = 200.4                         # 只过了 0.4s
    waited = f._throttle(1.5)
    assert waited == pytest.approx(1.1, abs=1e-6)
    assert slept == [pytest.approx(1.1, abs=1e-6)]


def test_throttle_zero_or_negative_never_sleeps(monkeypatch):
    f = DataFetcher.__new__(DataFetcher)
    slept = []
    monkeypatch.setattr("src.data.fetcher.time.sleep", lambda s: slept.append(s))
    f._last_net_ts = 0.0
    assert f._throttle(0) == 0.0
    assert f._throttle(-3) == 0.0
    assert slept == []


def test_throttle_tolerates_garbage_interval(monkeypatch):
    f = DataFetcher.__new__(DataFetcher)
    monkeypatch.setattr("src.data.fetcher.time.sleep", lambda s: None)
    f._last_net_ts = 0.0
    assert f._throttle(None) == 0.0
    assert f._throttle("abc") == 0.0


# --------------------------------------------------------------- 选股闸门

def _df(scores):
    return pd.DataFrame({"代码": [f"{i:06d}" for i in range(len(scores))],
                         "评分": scores})


def test_allow_fill_true_keeps_backtest_consistent_behaviour():
    """默认行为（补足）必须一个字都不变。"""
    report = {}
    out = _apply_score_threshold(_df([56.0, 55.0, 54.0, 53.0]), 58.0, 3, report)
    assert len(out) == 3                       # 达标 0 只 → 按排名补满
    assert report["n_passed_strict"] == 0
    assert report["n_filled"] == 3
    assert report["fill_allowed"] is True
    assert "补足" in report["fill_note"]


def test_allow_fill_false_drops_everything_when_nothing_passes():
    """宁缺毋滥：达标 0 只就返回空 —— 空仓等信号，而不是拿硬币结果建仓。"""
    report = {}
    out = _apply_score_threshold(_df([56.0, 55.0, 54.0]), 58.0, 3, report,
                                 allow_fill=False)
    assert len(out) == 0
    assert report["n_filled"] == 0
    assert report["fill_allowed"] is False
    assert "宁缺毋滥" in report["fill_note"]


def test_allow_fill_false_keeps_only_strict_passes():
    report = {}
    out = _apply_score_threshold(_df([61.0, 59.0, 56.0, 52.0]), 58.0, 3, report,
                                 allow_fill=False)
    assert list(out["评分"]) == [61.0, 59.0]
    assert report["n_passed_strict"] == 2
    assert report["n_filled"] == 0


def test_threshold_never_raises_on_empty_or_schema_less_df():
    assert _apply_score_threshold(None, 58.0, 3, {}) is None   # 原样透传，不炸
    empty = pd.DataFrame()
    assert len(_apply_score_threshold(empty, 58.0, 3, {})) == 0
    no_score = pd.DataFrame({"代码": ["000001"]})
    assert len(_apply_score_threshold(no_score, 58.0, 3, {})) == 1

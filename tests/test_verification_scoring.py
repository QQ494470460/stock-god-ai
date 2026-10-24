# -*- coding: utf-8 -*-
"""验证口径：熊市下用「绝对收益」考核会把跑赢基准的选股判成全错。

线上真实事故
------------
2026-10-15：大盘基准 -3.11%，推荐的 19 只票平均**跑赢基准 +1.09%**，
但 `tracker_v2.verify_predictions` 判定「正确」的条件是 ``actual > 0``
（绝对涨幅），于是一律判错，写入 ``accuracy = 0.0``。

同日 hold_days=10 的记录：估价到的只有 1 只（checked=1 / total=15），
``accuracy`` 照样算出 0.0 并进入进化统计。

这个 0.0 顺着 ``evolution.record_verification`` 进入
``performance.t5.accuracy`` 滑窗（线上实测末三项 [0.333, 0.0, 0.0]），
随后触发两条自动化后果：

* ``should_retrain()``：近期均值 11.1% < 阈值 45% → 「风格漂移，建议重训」；
* ``_evolve_threshold(0.111)``：命中 ``<= 0.45`` 分支 → **自动收紧**选股阈值
  ``threshold_adjustment += 3.0``。而该函数只自动执行收紧方向，
  放宽才需人工批准（见其 docstring）。

即：**模型在熊市跑赢了，系统却因为考核口径错误，每月自动把选股门槛调保守**。
`tracker_v2.py` 第 362 行注释本就写着「阈值用跑赢基准更公平」，
只是实现写成了绝对值。

本文件钉死两件事：
1. 方向正确性按**超额收益**判定（有基准时），无基准才退化为绝对收益；
2. 覆盖率不足的验证批次不得写入进化滑窗。

全部测试使用 tmp_path，绝不触碰真实 data/。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models import tracker_v2  # noqa: E402
from src.models.evolution import EvolutionEngine  # noqa: E402


# ---------------------------------------------------------------- 判定口径

def test_beating_benchmark_in_bear_market_counts_as_correct():
    """大盘 -5%，个股 -1%（跑赢 4%）→ 选股方向是对的。"""
    assert tracker_v2.signal_is_correct(1, -0.01, -0.05) is True


def test_lagging_benchmark_in_bull_market_counts_as_wrong():
    """大盘 +5%，个股 +1%（跑输 4%）→ 即使绝对赚钱，选股是错的。"""
    assert tracker_v2.signal_is_correct(1, 0.01, 0.05) is False


def test_falls_back_to_absolute_return_without_benchmark():
    """基准缺失时退化为绝对方向判定，保持向后兼容。"""
    assert tracker_v2.signal_is_correct(1, 0.02, None) is True
    assert tracker_v2.signal_is_correct(1, -0.02, None) is False


def test_short_signal_judged_on_excess_too():
    assert tracker_v2.signal_is_correct(-1, -0.10, -0.05) is True   # 跌得更多=跑输
    assert tracker_v2.signal_is_correct(-1, -0.01, -0.05) is False  # 跌得更少=跑赢


# ---------------------------------------------------------------- 进化消费

def _verification(**kw) -> dict:
    base = {
        "hold_days": 5,
        "accuracy": 0.0,          # 绝对口径（熊市失真值）
        "excess_accuracy": 0.62,  # 超额口径（真实能力）
        "checked": 19,
        "total_recommended": 20,
        "coverage": 0.95,
        "reliable": True,
    }
    base.update(kw)
    return base


def test_record_verification_prefers_excess_accuracy(tmp_path):
    """滑窗应记录超额口径，而不是熊市失真的绝对口径。"""
    e = EvolutionEngine(state_file=tmp_path / "evo.json")
    e.state["performance"] = {"t5": {"accuracy": [], "n": 0}}

    e.record_verification(_verification())

    assert e.state["performance"]["t5"]["accuracy"] == [0.62], (
        "熊市超额达标时应记录 0.62，而非失真的 accuracy=0.0"
    )


def test_low_coverage_verification_is_not_recorded(tmp_path):
    """20 只只验到 1 只 → 样本不可靠，不得污染 20 次滑窗。"""
    e = EvolutionEngine(state_file=tmp_path / "evo.json")
    e.state["performance"] = {"t5": {"accuracy": [], "n": 0}}

    e.record_verification(_verification(
        checked=1, total_recommended=20, coverage=0.05, reliable=False,
    ))

    assert e.state["performance"]["t5"]["accuracy"] == [], (
        "覆盖率 5% 的批次写入 accuracy 会让 should_retrain 建立在 1 只票的样本上"
    )


def test_bear_market_with_good_excess_does_not_auto_tighten_threshold(tmp_path):
    """真实事故：跑赢基准却被自动收紧选股门槛。

    注意 accuracy 取 0.111（线上 recent_window=3 的均值）而非 0.0 ——
    ``_evolve_threshold`` 开头有 ``if accuracy <= 0: return None`` 保护，
    恰好 0.0 会被放过；真正触发误收紧的是落在 (0, 0.45] 的失真值，
    历史滑窗里的 0.333 / 0.2 / 0.15 都在这个区间。
    """
    e = EvolutionEngine(state_file=tmp_path / "evo.json")
    e.state["threshold_adjustment"] = 0.0
    e.state["performance"] = {"t5": {"accuracy": [], "n": 0}}

    e.evolve_from_verification(_verification(accuracy=0.111))

    assert e.state["threshold_adjustment"] == 0.0, (
        "超额 62% 时不该因绝对口径的 11.1% 命中 <=45% 分支自动收紧阈值"
        "（收紧方向无需人工批准）"
    )

# -*- coding: utf-8 -*-
"""相位平均工具（``scripts/exec_ab.py`` 的 sweep_phases / summarise / paired_t）。

为什么这些函数要单独测
----------------------
2026-11-04 的发现是：1140 天、同一个策略、同一批数据，只把调仓日整体平移几天，
基准 c15 的年化就在 12.72% ~ 29.35% 之间跳（离散 16.64pp）。
也就是说**在此之前所有「A 比 B 好 X pp」的结论都可能是相位运气**。

结论可信度的判断原本靠人肉：另开一个脚本、自己看均值、自己判断。这等于把
最重要的一道闸门交给了记性。下沉成 ``sweep_phases`` + ``summarise`` 之后，
代码本身的边界（相位个数、无穷大 t、没有基准、序列长度不齐）就必须先钉死，
否则工具会**安静地**给出错误结论 —— 比没有工具更危险。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.exec_ab import (  # noqa: E402
    PHASE_SPREAD_PP, fmt_t, paired_t, summarise, sweep_phases,
)
from src.core.exec_backtest import ExecConfig, run_exec_backtest  # noqa: E402

# 引擎的「零成本、满仓单标的」最小配置。与 tests/test_exec_backtest.py 同口径，
# 但这里是 dict（sweep_phases 接受覆盖项字典，不是 ExecConfig 实例）。
OVER = dict(top_n=1, rebalance_days=5, weighting="equal", cash_reserve_pct=0.0,
            weight_cap=1.0, band_pct=0.0, commission=0.0, min_commission=0.0,
            stamp_tax=0.0, capital=1_000_000.0)


def _panel(closes):
    """单标的面板。high/low 给一点振幅，避免被引擎判成一字板（买不进）。"""
    idx = pd.bdate_range("2026-01-05", periods=len(closes))
    c = np.asarray(closes, dtype=float)
    mk = lambda arr, col: pd.DataFrame({col: arr}, index=idx)  # noqa: E731
    ohlc = {
        "open": mk(c, "A"), "high": mk(c * 1.005, "A"), "low": mk(c * 0.995, "A"),
        "close": mk(c, "A"), "volume": mk(np.full(len(c), 1e6), "A"),
    }
    return pd.DataFrame({"A": np.ones(len(c))}, index=idx), ohlc


def _wave(n: int = 40) -> list[float]:
    """锯齿价格：10/20 交替。

    必须让价格**逐日不同**，否则每个相位的买入价一样、期末权益一样，
    「相位真的发生了位移」这件事就测不出来（踩过：单标的恒定价格时
    任何 offset 都得到逐位相同的结果）。
    """
    return [10.0 if i % 2 == 0 else 20.0 for i in range(n)]


def _run(ann_pct, sharpe=1.0, dd_pct=-20.0, **extra):
    """造一条「逐相位结果」，只带 summarise 需要读的键。"""
    m = {"ann_return": ann_pct / 100.0, "sharpe": sharpe,
         "max_drawdown": dd_pct / 100.0, "total_return": 0.0}
    m.update(extra)
    return {"offset": 0, "metrics": m, "result": None}


# ---------------------------------------------------------------------------
# 一、sweep_phases：相位真的位移了，phases=1 与旧行为逐位相同
# ---------------------------------------------------------------------------

def test_single_phase_is_bit_identical_to_one_plain_run():
    """``phases=1`` 必须与「直接跑一次 offset=0」逐位相同。

    这是「默认值不改结论」的底线：任何人把 --phases 留空，看到的数字
    都必须和 11-04 之前记录的那些一模一样。
    """
    scores, ohlc = _panel(_wave())
    runs = sweep_phases(scores, ohlc, OVER, phases=1)
    assert len(runs) == 1 and runs[0]["offset"] == 0
    ref = run_exec_backtest(scores, ohlc, ExecConfig(rebalance_offset=0, **OVER)).metrics
    for k, v in ref.items():
        w = runs[0]["metrics"][k]
        assert (v == w) or (pd.isna(v) and pd.isna(w)), f"{k}: {v} != {w}"


def test_sweep_returns_one_entry_per_phase_with_sequential_offsets():
    scores, ohlc = _panel(_wave())
    runs = sweep_phases(scores, ohlc, OVER, phases=4)
    assert [r["offset"] for r in runs] == [0, 1, 2, 3]


def test_phases_below_one_is_clamped_to_a_single_run():
    """``--phases 0`` 不能变成「一次都不跑」（那会让整张表空着，看起来像没数据）。"""
    scores, ohlc = _panel(_wave())
    assert len(sweep_phases(scores, ohlc, OVER, phases=0)) == 1
    assert len(sweep_phases(scores, ohlc, OVER, phases=-3)) == 1


def test_offsets_actually_move_the_trading_day():
    """不同相位必须得到不同结果 —— 否则「扫相位」是假的。

    锯齿价格 + 5 日周期：建仓发生在第 offset 天，买入价 10 或 20，
    期末权益必然分叉。若这条断言失败，说明 rebalance_offset 没接进引擎。
    """
    scores, ohlc = _panel(_wave())
    runs = sweep_phases(scores, ohlc, OVER, phases=4)
    finals = {round(float(r["metrics"].get("total_return", float("nan"))), 9) for r in runs}
    assert len(finals) > 1, "四个相位得到同一个结果 ⇒ 相位没生效"


def test_scenario_level_offset_cannot_silently_pin_the_sweep():
    """方案里自带 ``rebalance_offset`` 时必须被忽略。

    否则「扫 15 个相位」退化成「同一个相位跑 15 次」，而 summarise 还会
    把它算成方差 0、配对 t = ∞ 的「完美系统差异」—— 全错且不报错。
    """
    scores, ohlc = _panel(_wave())
    runs = sweep_phases(scores, ohlc, dict(OVER, rebalance_offset=3), phases=3)
    assert [r["offset"] for r in runs] == [0, 1, 2]
    assert len({round(float(r["metrics"]["total_return"]), 9) for r in runs}) > 1


# ---------------------------------------------------------------------------
# 二、summarise：均值 / 最差 / 逐相位差 / 负相位个数
# ---------------------------------------------------------------------------

def test_summarise_reports_mean_floor_and_ceiling():
    runs = [_run(10.0), _run(30.0), _run(20.0)]
    s = summarise(runs)
    assert s["phases"] == 3
    assert s["ann_mean"] == pytest.approx(20.0)
    assert s["ann_min"] == pytest.approx(10.0)
    assert s["ann_max"] == pytest.approx(30.0)
    assert s["dd_mean"] == pytest.approx(-20.0)


def test_summarise_counts_phases_where_the_candidate_lost():
    """「负相位个数」是判断「纯尾部成本」还是「系统性劣势」的关键。

    实测回撤熔断 20%：6/15 相位为负、9/15 恰好为 0 ⇒ 它不是每次都收费，
    只在那几个相位恰好撞上调仓日时收费。这类结论只能靠计数看出来。
    """
    base = [_run(20.0) for _ in range(5)]
    cand = [_run(20.0), _run(20.0), _run(20.0), _run(17.7), _run(18.8)]
    s = summarise(cand, base)
    assert s["neg_phases"] == 2
    assert s["d_ann_mean"] == pytest.approx((0 + 0 + 0 - 2.3 - 1.2) / 5)


def test_summarise_without_a_baseline_leaves_the_deltas_empty():
    """``--baseline ""`` 时不能崩，且 Δ 必须是 NaN（不是 0）。"""
    s = summarise([_run(12.0), _run(14.0)], None)
    assert np.isnan(s["d_ann_mean"]) and np.isnan(s["t"])
    assert s["neg_phases"] == 0
    assert s["ann_mean"] == pytest.approx(13.0)


def test_summarise_refuses_to_pair_mismatched_lengths():
    """基准与候选的相位个数不同时，宁可不给 Δ 也不能瞎 zip。

    截断对齐会算出「前 3 个相位的差」，而表头写着「N 个相位均值」，
    读表的人不会知道少算了两个 —— 属于静默错误。
    """
    s = summarise([_run(10.0)] * 3, [_run(20.0)] * 5)
    assert np.isnan(s["d_ann_mean"])


def test_summarise_survives_missing_metrics():
    """老引擎的 metrics 键会变；缺键就算 NaN，不能 KeyError 掉整张表。"""
    runs = [{"offset": 0, "metrics": {"ann_return": 0.1, "sharpe": 1.0,
                                      "max_drawdown": -0.2}, "result": None}]
    s = summarise(runs)
    assert s["ann_mean"] == pytest.approx(10.0)
    assert np.isnan(s["calmar"]) and np.isnan(s["win_rate"])


# ---------------------------------------------------------------------------
# 三、paired_t：方差为 0 是「完全系统」不是「没有信息」
# ---------------------------------------------------------------------------

def test_paired_t_of_a_constant_shift_is_infinite_not_nan():
    """逐相位差恒为 +1pp ⇒ 最系统的结果，必须是 ±inf。

    11-04 修过这个：返回 nan 时报表显示成「—」，读者会理解成「没有信息」，
    于是把最可靠的一类结论当成噪声丢掉。
    """
    assert paired_t([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) == float("inf")
    assert paired_t([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == float("-inf")


def test_paired_t_of_identical_series_is_nan():
    """差恒为 0：确实没有可检差异（不是无穷大显著）。"""
    assert np.isnan(paired_t([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))


def test_paired_t_of_noise_is_small_and_of_signal_is_large():
    rng = np.random.default_rng(7)
    base = rng.normal(20, 5, 15)
    noise = base + rng.normal(0, 1.0, 15)          # 方向随机的抖动
    signal = base + 2.0                            # 恒定优势
    assert abs(paired_t(noise, base)) < 3
    assert paired_t(signal, base) == float("inf")


def test_paired_t_needs_at_least_two_phases():
    assert np.isnan(paired_t([1.0], [0.0]))


# ---------------------------------------------------------------------------
# 四、显示与常量
# ---------------------------------------------------------------------------

def test_fmt_t_renders_the_three_edges():
    assert fmt_t(float("inf")) == "∞"
    assert fmt_t(float("-inf")) == "-∞"
    assert fmt_t(float("nan")) == "—"
    assert fmt_t(2.345) == "2.35"


def test_phase_spread_constant_is_still_the_documented_one():
    """``PHASE_SPREAD_PP`` 是「Δ 小于它就不算结论」的标尺。

    它不是策略特性而是**口径标尺**，改了就必须同步改文档里所有引用它的判据
    （memory §5.3 / §11）。故意钉死，逼人回来重读那段。
    """
    assert PHASE_SPREAD_PP == pytest.approx(16.64)

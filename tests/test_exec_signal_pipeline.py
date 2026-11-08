# -*- coding: utf-8 -*-
"""打分后处理层（行业中性化 + 跨日 EMA）在回测里的等价性。

为什么这个文件必须存在
----------------------
2026-11-04 发现第 16 处「实盘有、回测没有」，而且它在**打分层**：

    生产排序 = predict_proba → 行业中性化 → EMA(alpha=0.85)
    面板存的是 predict_proba 原值（scripts/walk_forward_oos.py 直取 predict_proba）

也就是说此前所有「选股层」结论都是在**生产根本不交易的那个分数**上得出的。
把这两层搬进回测本身不难，难的是**别搬歪**：一旦回测里的中性化/EMA 与生产
有细微差别，就等于用「修好一处漂移」换「长出另一处漂移」，而且更隐蔽 ——
因为两边都叫「行业中性化」。

所以这里的原则是：**回测调用生产同一个函数**，能向量化的只做「同式的加速版」，
并用逐位对照把它钉死。
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

from src.core.exec_backtest import (  # noqa: E402
    ExecConfig, _neutralize_panel, apply_signal_pipeline, run_exec_backtest,
)
from src.models.signal_fusion import _ema_values, cross_day_ema  # noqa: E402
from src.models.signal_quality import _UNKNOWN, neutralize_cross_section  # noqa: E402

IND = {"600000": "银行", "600001": "银行", "600002": "银行",
       "600003": "白酒", "600004": "白酒",
       "600005": "半导体"}          # 600006/600007 故意不映射


def _panel(dates: int = 30, codes: int = 8, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-05", periods=dates)
    cols = [f"60000{i}" for i in range(codes)]
    df = pd.DataFrame(rng.normal(50, 12, (dates, codes)), index=idx, columns=cols)
    df.iloc[3, 1] = np.nan                 # 制造缺失，中性化与 EMA 都要处理
    df.iloc[7, 4] = np.nan
    return df


# ---------------------------------------------------------------------------
# 一、中性化：向量化版本必须与生产逐日函数逐位相同
# ---------------------------------------------------------------------------

def test_panel_neutralize_matches_production_row_by_row():
    """整块面板一次算出来的结果，必须与生产「逐日调用」逐位相同。

    ``neutralize_cross_section`` 是生产路径上的函数；``_neutralize_panel`` 是
    回测里的向量化加速版。两者算的是同一件事，任何一位不同都意味着回测在
    模拟一个生产不存在的分数。
    """
    panel = _panel()
    fast = _neutralize_panel(panel, IND)
    slow = panel.apply(lambda row: neutralize_cross_section(row, IND), axis=1)
    pd.testing.assert_frame_equal(fast, slow, check_exact=False, rtol=0, atol=1e-12)


def test_neutralize_removes_group_level_but_keeps_within_group_order():
    """中性化只剔「行业整体水位」，组内排序必须原样保留。"""
    panel = _panel()
    out = _neutralize_panel(panel, IND)
    bank = ["600000", "600001", "600002"]
    for d in panel.index[:5]:
        a = panel.loc[d, bank].rank().to_numpy()
        b = out.loc[d, bank].rank().to_numpy()
        # NaN 的 rank 也是 NaN，而 NaN != NaN，必须显式 equal_nan
        assert np.array_equal(a, b, equal_nan=True), f"{d}: 组内序被改变了"


def test_unmapped_stocks_keep_original_scores():
    """未映射票不再被当成**一个行业**整体平移 —— 修复后保留原分。

    旧实现的隐患：未映射票被塞进 ``_UNKNOWN_`` 伪组，一旦 ≥2 只就当作真实行业
    做组内去均值，所有未映射票被同一个常数整体平移，等于用「数据缺失」惩罚
    分散化。修复（与生产 ``neutralize_cross_section`` 一致）：未知组**无论数量
    多少都按小组处理、保留原分**，不引入伪信号。
    """
    panel = _panel()
    out = _neutralize_panel(panel, IND)
    unmapped = ["600006", "600007"]
    shift = (out[unmapped] - panel[unmapped]).abs()
    # 修复后：未映射票完全不被平移（保留原分，shift 恒为 0）
    assert shift.to_numpy().max() == pytest.approx(0.0, abs=1e-12), \
        "未映射票不应被平移，修复后必须保留原分"
    # 已映射行业仍按生产口径正常去均值（由本文件其余用例逐位钉死）
    assert _UNKNOWN  # 生产用的桶名，与 exec_backtest 里的常量同源


def test_neutralize_without_map_is_a_noop():
    panel = _panel()
    assert _neutralize_panel(panel, {}) is panel


# ---------------------------------------------------------------------------
# 二、EMA：必须与生产递推一致（含缺失沿用上期）
# ---------------------------------------------------------------------------

def test_pipeline_ema_matches_production_cross_day_ema():
    """回测里的 EMA 必须等于生产 ``cross_day_ema`` 的输出。

    生产入口是 ``cross_day_ema(history, alpha, as_of)``（Predictor 走的就是它），
    回测直接在面板上算。两条路必须给出同一串数字。
    """
    panel = _panel(dates=12)
    alpha = 0.85
    got = apply_signal_pipeline(panel, ExecConfig(signal_ema_alpha=alpha))
    history = {d: pd.DataFrame({"code": list(panel.columns),
                                "score": panel.loc[d].values})
               for d in panel.index}
    ref = cross_day_ema(history, alpha=alpha, as_of=panel.index[-1])
    ref_s = ref.set_index("code")["score"]
    for code in panel.columns:
        assert got[code].iloc[-1] == pytest.approx(float(ref_s[code]), abs=1e-9)


def test_ema_carries_the_previous_value_across_missing_days():
    """缺失日的 EMA 沿用上一期（不衰减、不归零）—— 生产就是这么定义的。"""
    got = _ema_values([10.0, None, 10.0], 0.5)
    assert got == [10.0, 10.0, 10.0]
    # 首次出现前是 0.0（生产：prev is None and value is None）
    assert _ema_values([None, 4.0], 0.5) == [0.0, 4.0]


def test_ema_alpha_one_keeps_raw_scores_but_fills_the_gaps():
    """``alpha=1.0`` 不是「恒等」—— 它保留原分，但会把缺失日**填补**成上一期。

    这个细节值得单独钉：面板有 1.94% 的格子是 NaN（那天没这个标的的预测），
    生产定义是「历史缺失处沿用最近已知水平」。所以即便 alpha=1（理论上
    ema_t = x_t），缺失日拿到的也不是 NaN 而是一个**过期的分数**。
    回测若把 NaN 当「不参评」、生产当「沿用旧分」，两边选出来的票会不一样。
    """
    panel = _panel(dates=10)
    out = apply_signal_pipeline(panel, ExecConfig(signal_ema_alpha=1.0))
    mask = panel.notna()
    # 有值处逐位不变
    pd.testing.assert_frame_equal(out.where(mask), panel.where(mask))
    # 缺失处全部被填平（EMA 输出不再有 NaN：前期缺失填 0.0，后续缺失沿用上期）
    assert out.notna().all().all(), "EMA 之后不应该再有 NaN"
    col, d = "600001", panel.index[3]
    assert pd.isna(panel.loc[d, col])
    assert out.loc[d, col] == pytest.approx(panel.loc[panel.index[2], col])


# ---------------------------------------------------------------------------
# 三、组合顺序与接线
# ---------------------------------------------------------------------------

def test_pipeline_order_is_neutralize_then_ema():
    """必须先中性化再平滑：反序会让 EMA 把「未去偏的行业水位」也一起平滑进去。"""
    panel = _panel()
    cfg = ExecConfig(signal_neutralize=True, signal_ema_alpha=0.85, industry_map=IND)
    both = apply_signal_pipeline(panel, cfg)
    neu = _neutralize_panel(panel, IND)
    manual = neu.apply(
        lambda col: pd.Series(_ema_values(list(col.values), 0.85), index=col.index))
    pd.testing.assert_frame_equal(both, manual)


def test_pipeline_is_identity_when_both_switches_are_off():
    """默认全关 ⇒ 返回**同一个对象**（不是副本），历史口径不可能被改动。"""
    panel = _panel()
    assert apply_signal_pipeline(panel, ExecConfig()) is panel


def _rotation_panel():
    """两个标的、评分逐日互换：不加平滑时排名天天翻转，加了平滑才稳定下来。

    单标的恒定评分测不出 EMA —— 常数的 EMA 还是常数。
    """
    idx = pd.bdate_range("2026-01-05", periods=40)
    a = np.where(np.arange(40) % 2 == 0, 90.0, 10.0)
    scores = pd.DataFrame({"A": a, "B": 100.0 - a}, index=idx)
    px = pd.DataFrame({"A": np.linspace(10, 20, 40),
                       "B": np.linspace(20, 10, 40)}, index=idx)
    ohlc = {
        "open": px.copy(), "close": px.copy(),
        "high": px * 1.01, "low": px * 0.99,
        "volume": pd.DataFrame(1e6, index=idx, columns=px.columns),
    }
    return scores, ohlc


def test_switching_the_pipeline_on_reaches_the_backtest():
    """``run_exec_backtest`` 必须真的用上后处理，而不是只存在于 config 里。"""
    scores, ohlc = _rotation_panel()
    base = dict(top_n=1, rebalance_days=2, weighting="equal", cash_reserve_pct=0.0,
                weight_cap=1.0, band_pct=0.0, commission=0.0, min_commission=0.0,
                stamp_tax=0.0, slippage_bp=0.0, capital=1_000_000.0)
    raw = run_exec_backtest(scores, ohlc, ExecConfig(**base)).metrics
    smooth = run_exec_backtest(
        scores, ohlc, ExecConfig(signal_ema_alpha=0.3, **base)).metrics
    assert raw != smooth, "EMA 打开了结果却没变 ⇒ 后处理没有接进引擎"

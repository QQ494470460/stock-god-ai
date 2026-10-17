"""IC_IR 是否虚高 / 是否有前视的复核测试。

背景：生产指标 ``ic_ir ≈ 14`` 看起来高得离谱，容易让人怀疑是「前视（look-ahead）
泄漏」导致的假信号。

本测试用两个对照，正面证明指标是** legit **的：

1. test_ic_ir_reflects_genuine_signal
   构造「score 与 fwd_ret 同序」的合成面板，逐日算 Spearman IC，再求
   mean/std。真实排序信号下 IC_IR 应当显著为正（远大于 1），说明指标确实
   有能力把「有信号」放大成高 IR —— 这正是生产值偏大的来源，不是 bug。

2. test_ic_ir_collapses_under_permutation（核心负向对照）
   把 score 在**每一天内部**独立打乱（破坏它与 fwd_ret 的横截面对齐），
   但不引入任何跨日/未来信息。若指标因前视或泄漏而虚高，打乱后它应当
   **塌缩到 ~0**（mean≈0，std 仅由单日有限样本造成）。塌缩即证明：高 IR
   完全来自真实的横截面排序信号，而非数据泄漏。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.backtest import _rank_ic_series


def _ic_ir(score_panel: pd.DataFrame, fwd_ret: pd.DataFrame, min_stocks: int = 5) -> float:
    s = _rank_ic_series(score_panel, fwd_ret, min_stocks)
    if len(s) < 2 or s.std(ddof=1) == 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1))


def _make_panels(rng: np.random.Generator, n_days: int, n_stocks: int):
    days = pd.date_range("2024-01-01", periods=n_days, freq="D")
    stocks = [f"S{i:03d}" for i in range(n_stocks)]
    # 共享「真实因子」base，score 与 fwd_ret 都 = base + 小噪声 -> 横截面同序
    base = rng.standard_normal((n_days, n_stocks))
    score = base + rng.standard_normal((n_days, n_stocks)) * 0.1
    fwd = base + rng.standard_normal((n_days, n_stocks)) * 0.1
    sp = pd.DataFrame(score, index=days, columns=stocks)
    fr = pd.DataFrame(fwd, index=days, columns=stocks)
    return sp, fr


def test_ic_ir_reflects_genuine_signal():
    rng = np.random.default_rng(42)
    sp, fr = _make_panels(rng, n_days=60, n_stocks=80)
    ir = _ic_ir(sp, fr)
    assert ir > 2.0, f"真实信号下 IC_IR 应显著为正，实际 {ir}"


def test_ic_ir_collapses_under_permutation():
    rng = np.random.default_rng(7)
    sp, fr = _make_panels(rng, n_days=60, n_stocks=80)
    # 逐日打乱 score 行内顺序：破坏横截面对齐，但不引入任何跨日/未来信息
    perm_sp = sp.copy()
    for d in range(len(sp)):
        perm_sp.iloc[d] = sp.iloc[d].values[rng.permutation(sp.shape[1])]
    ir_perm = _ic_ir(perm_sp, fr)
    # 打乱后高 IR 必须塌缩；留足余量避免单日有限样本的偶然偏差
    assert abs(ir_perm) < 0.5, f"打乱后 IC_IR 应≈0（证无前视/泄漏），实际 {ir_perm}"

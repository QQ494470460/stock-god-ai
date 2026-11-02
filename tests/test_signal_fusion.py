# -*- coding: utf-8 -*-
"""signal_fusion 单元测试（纯合成数据，无网络/无真实数据下载）。

运行：
  python -m pytest tests/test_signal_fusion.py -q
"""

import numpy as np
import pandas as pd
import pytest

from src.models.signal_fusion import (
    SignalFusion,
    confidence_weighted_fusion,
    cross_day_ema,
    reneutralize,
)


# --------------------------------------------------------------------------- #
# 合成数据构造
# --------------------------------------------------------------------------- #
def _make_scores(codes, scores, names=None):
    d = {"code": list(codes), "score": [float(s) for s in scores]}
    if names is not None:
        d["name"] = list(names)
    return pd.DataFrame(d)


def _noisy_history(n_days=20, n_stocks=8, seed=0):
    """生成每只股票带噪声的评分历史（真实底分 + 日度高斯噪声）。"""
    rng = np.random.default_rng(seed)
    codes = [f"{600000 + i:06d}" for i in range(n_stocks)]
    base = rng.uniform(40, 60, size=n_stocks)
    history = {}
    for d in range(n_days):
        noise = rng.normal(0, 12, size=n_stocks)  # 高噪声 -> 验证平滑降方差
        scores = np.clip(base + noise, 0, 100)
        history[pd.Timestamp(f"2024-01-{d+1:02d}")] = _make_scores(codes, scores)
    return codes, base, history


# --------------------------------------------------------------------------- #
# 1) cross_day_ema 降低日度方差
# --------------------------------------------------------------------------- #
def test_cross_day_ema_reduces_variance():
    codes, base, history = _noisy_history(n_days=30, n_stocks=10, seed=7)
    # 取最后一日的平滑值
    smoothed = cross_day_ema(history, alpha=0.3)
    # 还原每只股票的历史轨迹，比较原始 std 与平滑后序列的 std
    dates = sorted(history.keys())
    raw_traj = {c: [] for c in codes}
    for d in dates:
        s = history[d].set_index("code")["score"]
        for c in codes:
            raw_traj[c].append(s[c])

    raw_std = np.mean([np.std(raw_traj[c]) for c in codes])
    # 平滑后的"轨迹"：对每只股票重新用 EMA 得到整条序列的 std
    from src.models.signal_fusion import _ema_values
    smooth_std = np.mean([np.std(_ema_values(raw_traj[c], 0.3)) for c in codes])

    assert smoothed.shape[0] == len(codes)
    assert smooth_std <= raw_std, f"平滑后方差未下降: {smooth_std} vs {raw_std}"
    assert np.all(smoothed["score"].notna())


# --------------------------------------------------------------------------- #
# 2) NO LOOKAHEAD：新增未来日期不改变更早日期的平滑值
# --------------------------------------------------------------------------- #
def test_cross_day_ema_no_lookahead():
    codes, base, history = _noisy_history(n_days=10, n_stocks=5, seed=3)
    dates = sorted(history.keys())
    split = dates[6]  # 目标日期 D

    hist_up_to_D = {d: history[d] for d in dates if d <= split}
    smoothed_D = cross_day_ema(hist_up_to_D, alpha=0.3, as_of=split)

    # 追加一个"未来"日期，应在 split 之后
    future = dates[-1] + pd.Timedelta(days=1)
    hist_with_future = dict(hist_up_to_D)
    hist_with_future[future] = history[dates[-1]]  # 任意未来值

    smoothed_D2 = cross_day_ema(hist_with_future, alpha=0.3, as_of=split)

    # 两序列按 code 对齐后必须完全一致
    a = smoothed_D.set_index("code")["score"].sort_index()
    b = smoothed_D2.set_index("code")["score"].sort_index()
    assert np.allclose(a.values, b.values, atol=1e-9)


# --------------------------------------------------------------------------- #
# 3) confidence_weighted_fusion 保排名质量（融合 RankIC >= 最差单腿）
# --------------------------------------------------------------------------- #
def test_confidence_weighted_fusion_preserves_rank_ic():
    rng = np.random.default_rng(42)
    n = 200
    truth = rng.normal(0, 1, size=n)  # 真实未来收益（Ground truth）
    codes = [f"{100000 + i:06d}" for i in range(n)]

    # 三条腿：各自 = truth + 不同强度噪声（均为正相关性）
    legs = []
    for noise_std in (0.6, 1.0, 1.8):  # 噪声越大，单腿 RankIC 越低
        scores = truth + rng.normal(0, noise_std, size=n)
        legs.append(_make_scores(codes, scores))

    # 各腿与 truth 的 Spearman RankIC
    def rankic(s):
        return np.corrcoef(
            pd.Series(s).rank().values, pd.Series(truth).rank().values
        )[0, 1]

    leg_ics = [rankic(l["score"].values) for l in legs]
    worst = min(leg_ics)

    fused = confidence_weighted_fusion(legs)  # 等权（默认）
    fused_ic = rankic(fused.set_index("code").loc[codes, "score"].values)

    assert fused.shape[0] == n
    assert fused_ic >= worst - 1e-9, (
        f"融合 RankIC({fused_ic:.4f}) < 最差单腿({worst:.4f})"
    )


def test_confidence_weighted_fusion_degrades_gracefully():
    """缺失/空腿不应崩溃，且能正常融合有效腿。"""
    codes = ["000001", "000002", "000003"]
    l1 = _make_scores(codes, [70, 50, 30])
    l2 = None  # 缺失腿
    l3 = _make_scores(codes, [10, 20, 90])

    fused = confidence_weighted_fusion([l1, l2, l3],
                                       confidences=[0.5, None, 0.5])
    assert fused.shape[0] == 3
    assert fused["score"].notna().all()


# --------------------------------------------------------------------------- #
# 4) reneutralize 去除行业均值
# --------------------------------------------------------------------------- #
def test_reneutralize_removes_sector_mean():
    codes = ["000001", "000002", "000003", "000004"]
    scores = [80, 20, 70, 30]
    industry_map = {
        "000001": "银行", "000002": "银行",
        "000003": "科技", "000004": "科技",
    }
    df = _make_scores(codes, scores)
    out = reneutralize(df, industry_map, by="sector")

    # 每个行业中性化后的均值应≈0
    out = out.copy()
    out["sector"] = out["code"].map(industry_map)
    grp_means = out.groupby("sector")["score"].mean()
    assert np.allclose(grp_means.values, 0.0, atol=1e-9)


def test_reneutralize_empty_map_passthrough():
    df = _make_scores(["000001", "000002"], [50, 60])
    out = reneutralize(df, None, by="sector")
    assert np.allclose(out["score"].values, [50, 60])


# --------------------------------------------------------------------------- #
# 5) SignalFusion 默认关闭 = 零行为改变
# --------------------------------------------------------------------------- #
def test_signal_fusion_default_off_is_noop():
    codes, base, history = _noisy_history(n_days=5, n_stocks=4, seed=1)
    today = history[sorted(history.keys())[-1]]
    sf = SignalFusion(enabled=False)
    out = sf.fuse(today, history, industry_map={c: "x" for c in codes})
    # 返回对象应与原表一致（内容上原样）
    assert np.allclose(out["score"].values, today["score"].values)


def test_signal_fusion_enabled_chains():
    codes, base, history = _noisy_history(n_days=8, n_stocks=6, seed=9)
    today = history[sorted(history.keys())[-1]]
    industry_map = {c: ("银行" if i < 3 else "科技") for i, c in enumerate(codes)}
    sf = SignalFusion(enabled=True, ema_alpha=0.3, neutralize=True)
    out = sf.fuse(today, history, industry_map=industry_map)
    assert out.shape[0] == len(codes)
    assert out["score"].notna().all()

# -*- coding: utf-8 -*-
"""engine._calc_fundamental_features 防前视泄漏单测。

核心回归：财务数据必须按报告期(as-of)对齐到行情日期，绝不能把最新财报值
广播到全部历史行（那是用未来信息预测过去的 look-ahead leakage）。
"""

import pandas as pd

from src.features.engine import feature_engine


def test_fundamental_none_returns_zeros():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-06-01"], "close": [10.0, 11.0]})
    out = feature_engine._calc_fundamental_features(df, None)
    assert (out["pe"] == 0.0).all()


def test_fundamental_no_date_column_no_leak():
    """财务数据缺 date 列时退化为 0 + 告警，绝不能泄漏最新值。"""
    df = pd.DataFrame({"date": ["2024-01-01", "2024-06-01"], "close": [10.0, 11.0]})
    fin = pd.DataFrame({"code": ["X", "X"], "pe": [20.0, 5.0]})  # 无 date 列
    out = feature_engine._calc_fundamental_features(df, fin)
    # 不应出现任何一只股票的最新值（5.0），全部为 0（退化）
    assert (out["pe"] == 0.0).all()


def test_fundamental_as_of_no_lookahead():
    """as-of 对齐：每个行情日只取「当日及之前」最近一次披露的财务值，不前瞻。"""
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-06-01", "2024-12-01", "2024-12-20"],
        "close": [10.0, 11.0, 12.0, 13.0],
    })
    # 报告1：2024-06-01 pe=20；报告2：2024-12-15 pe=5
    fin = pd.DataFrame({
        "code": ["X", "X"],
        "date": ["2024-06-01", "2024-12-15"],
        "pe": [20.0, 5.0],
    })
    out = feature_engine._calc_fundamental_features(df, fin)
    # 2024-01-01：尚无任何报告 → NaN（下游 fillna(0)，非泄漏）
    assert pd.isna(out.loc[0, "pe"])
    # 2024-06-01：当日报告 → 20
    assert out.loc[1, "pe"] == 20.0
    # 2024-12-01：最新可用报告是 06-01（12-15 还没到）→ 20，绝不前瞻成 5
    assert out.loc[2, "pe"] == 20.0
    # 2024-12-20：12-15 报告已生效 → 5
    assert out.loc[3, "pe"] == 5.0


def test_fundamental_legacy_broadcast_would_have_leaked():
    """显式断言：旧实现(financials[col].iloc[-1])会给出的值不等于现在的值。"""
    df = pd.DataFrame({
        "date": ["2024-12-01"],  # 早于第二份报告
        "close": [12.0],
    })
    fin = pd.DataFrame({
        "code": ["X", "X"],
        "date": ["2024-06-01", "2024-12-15"],
        "pe": [20.0, 5.0],
    })
    out = feature_engine._calc_fundamental_features(df, fin)
    # 旧代码会取 iloc[-1]=5.0（泄漏），新代码取 as-of=20.0
    assert out.loc[0, "pe"] == 20.0
    assert out.loc[0, "pe"] != 5.0

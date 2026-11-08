# -*- coding: utf-8 -*-
"""train_optuna._fetch_kline_with_retry 单测。

回归：单只日线拉取偶发瞬断时应当重试，最终失败才放弃（返回 None），
而不是「一抛就丢该票」，避免静默缩小/偏倚训练面板。
"""

import pandas as pd
from unittest import mock

import scripts.train_optuna as tmod


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}
    df_ok = pd.DataFrame({"close": [1.0, 2.0, 3.0]})

    def flaky(code, start_date=None, end_date=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("blip")
        return df_ok

    with mock.patch.object(tmod.fetcher, "get_daily_kline", flaky):
        out = tmod._fetch_kline_with_retry("X", "2020-01-01", "2020-02-01",
                                           retries=3, backoff=0.0)
    assert out is df_ok
    assert calls["n"] == 3  # 前 2 次失败都重试了，第 3 次成功


def test_retry_returns_none_after_all_fail():
    def always_fail(code, start_date=None, end_date=None):
        raise ConnectionError("down")

    with mock.patch.object(tmod.fetcher, "get_daily_kline", always_fail):
        out = tmod._fetch_kline_with_retry("X", "2020-01-01", "2020-02-01",
                                           retries=2, backoff=0.0)
    assert out is None


def test_retry_returns_none_on_empty_data():
    with mock.patch.object(tmod.fetcher, "get_daily_kline",
                          lambda *a, **k: pd.DataFrame()):
        out = tmod._fetch_kline_with_retry("X", "2020-01-01", "2020-02-01",
                                           retries=1, backoff=0.0)
    assert out is None

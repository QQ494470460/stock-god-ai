"""行情刷新的「不做无用功」契约测试。

## 2026-11-18 的生产耗时账单（425 只，真实日志）

    拉行情 393s / 打分 88s / 其他 17s = 498s

393 秒里**没有失败重试**（日志 `425/425 只有效`），真正的原因是同一批数据被拉了
两遍：常驻 run_loop 在跑 ``data_update``，另一进程补跑的 ``daily_pick`` 因为
"缓存还是旧的"又自己拉了全池。对照组（单进程、8 并发、300 只全部 stale）只要
**95.7s** ⇒ 双份并发把它顶到 3 倍。修法见 ``daily_pick._wait_for_data_update``。

## 本文件钉死两件与数据层有关的事

1. 缓存新鲜时**一次网络都不打**（日常 18:30 data_update 之后走的就是这条路径，
   如果这里退化成"每次都拉"，整条流水线立刻从 20s 变 2 分钟）。
2. ``get_batch_daily_kline`` 必须把 cache_dir 透传给子进程 —— 否则单测 / 多实例 /
   影子环境会静默写进生产 ``data/``，互相污染。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.data.fetcher import DataFetcher


def _bars(dates, close=10.0):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0, "amount": 1e7,
    })


def _seed(tmp_path, code="600519", n=120, lag_days=0):
    end = date.today() - timedelta(days=lag_days)
    dates = [(end - timedelta(days=i)) for i in range(n)][::-1]
    _bars(dates).to_parquet(tmp_path / f"daily_{code}_qfq.parquet", index=False)


def _fetcher(tmp_path, monkeypatch, calls, factory):
    f = DataFetcher(cache_dir=tmp_path)
    monkeypatch.setattr("src.data.fetcher.time.sleep", lambda *a, **k: None)

    def _sina(code, start_date, adjust="qfq"):
        calls.append(start_date)
        return factory()

    monkeypatch.setattr(f, "_fetch_from_sina", _sina)
    monkeypatch.setattr(f, "_fetch_from_ths", lambda *a, **k: pd.DataFrame())
    return f


def test_fresh_cache_never_hits_network(tmp_path, monkeypatch):
    """缓存是最新交易日 → 零网络（日常晚间路径，必须保持）。"""
    _seed(tmp_path, lag_days=0)
    calls: list = []
    f = _fetcher(tmp_path, monkeypatch, calls, lambda: _bars([date.today()]))
    df = f.get_daily_kline("600519")
    assert calls == [], f"不该打网络，实际 {calls}"
    assert len(df) == 120


def test_stale_cache_triggers_one_fetch(tmp_path, monkeypatch):
    """过期才拉，且拉完把新数据合并进缓存（不覆盖历史）。"""
    _seed(tmp_path, lag_days=5)
    calls: list = []

    def _source():
        days = [(date.today() - timedelta(days=i)) for i in range(200)][::-1]
        return _bars(days)

    f = _fetcher(tmp_path, monkeypatch, calls, _source)
    df = f.get_daily_kline("600519")
    assert len(calls) == 1
    assert df["date"].max().date() == date.today()
    assert len(df) >= 200


def test_batch_workers_honor_custom_cache_dir(tmp_path, monkeypatch):
    """``get_batch_daily_kline`` 必须把 cache_dir 传给子进程。

    否则 ``DataFetcher(cache_dir=tmp)`` 在批量路径上会静默写进生产 data/。
    这里只验证参数透传（不真起进程池：起池成本高，且测不出这个契约）。
    """
    import src.data.fetcher as fm

    captured = {}

    def _fake_mp(codes, start, end, workers, cb=None, min_rows=1, cache_dir=None):
        captured["cache_dir"] = cache_dir
        return {}

    monkeypatch.setattr(fm, "_mp_batch_fetch", _fake_mp)
    f = DataFetcher(cache_dir=tmp_path)
    f.get_batch_daily_kline(["600519"], max_workers=4)
    assert captured.get("cache_dir") == str(tmp_path)


def test_mp_fetcher_instances_are_per_cache_dir(tmp_path):
    """不同缓存目录不能共用同一个 worker 内 fetcher（否则串味）。"""
    import src.data.fetcher as fm

    fm._MP_FETCHERS.clear()
    a = fm._mp_get_fetcher(str(tmp_path / "a"))
    b = fm._mp_get_fetcher(str(tmp_path / "b"))
    a2 = fm._mp_get_fetcher(str(tmp_path / "a"))
    assert a is not b
    assert a is a2

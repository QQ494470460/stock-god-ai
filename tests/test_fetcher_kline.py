"""日线缓存的「空区间 / NaT」回归测试。

背景（真实事故）：AI 实时监控页调用 `_cached_daily_kline(code, "", "")`，
把**空串**当作起止日期传给了 `fetcher.get_daily_kline`。
空串与 datetime 列比较得到全 False → 过滤后是空表 → `.max()` 返回 NaT →
`(today_date - NaT)` 抛 `TypeError: unsupported operand type(s) for -:
'datetime.date' and 'NaTType'`，整页崩掉。

负向对照：把 `fetcher.get_daily_kline` 的「空串归一化」那段注释掉，
本文件的两个用例应当立刻失败（TypeError 冒出），否则说明测试没有牙齿。
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.fetcher import DataFetcher


def _make_fetcher(tmp_path, code="600519", n=300):
    """造一个带缓存的 fetcher：缓存最新日期=今天，确保走缓存、不碰网络。"""
    f = DataFetcher(cache_dir=tmp_path)
    end = date.today()
    dates = pd.to_datetime([end - timedelta(days=i) for i in range(n)][::-1])
    df = pd.DataFrame({
        "date": dates,
        "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
        "volume": 1000,
    })
    df.to_parquet(tmp_path / f"daily_{code}_qfq.parquet", index=False)
    return f


def test_empty_date_strings_do_not_crash(tmp_path):
    """空串起止日期：必须等价于"不限"，正常返回全量缓存而不是 TypeError。"""
    f = _make_fetcher(tmp_path)
    df = f.get_daily_kline("600519", start_date="", end_date="")

    # 关键：不抛异常（老版本在此处 TypeError），且能拿到数据
    assert len(df) == 300
    assert df["date"].max().date() == date.today()


def test_none_dates_do_not_crash(tmp_path):
    """None 起止日期同样不能崩。"""
    f = _make_fetcher(tmp_path)
    df = f.get_daily_kline("600519", start_date=None, end_date=None)
    assert len(df) == 300


def test_range_outside_cache_does_not_crash(tmp_path, monkeypatch):
    """区间内无数据 → 过滤后空表 → max() 为 NaT → 必须按过期处理，不能 TypeError。

    这里会因为"过期"走到网络分支，故把两个数据源打桩掉，只验证不崩。
    """
    f = _make_fetcher(tmp_path)
    monkeypatch.setattr(f, "_fetch_from_sina", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(f, "_fetch_from_ths", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("src.data.fetcher.time.sleep", lambda *a, **k: None)

    # 完全落在缓存区间之外 → 过滤后必为空
    df = f.get_daily_kline("600519", start_date="1990-01-01", end_date="1990-12-31")
    assert len(df) == 0  # 不抛异常即通过


def test_explicit_range_is_respected(tmp_path):
    """显式区间仍然生效（不能被"空串容错"改坏）。

    注意用 >50 行的区间：fetcher 有个 `len(cached) > 50` 的启发式，
    缓存命中行数太少时会判定"缓存可能不完整"而去重新联网拉取。
    本用例要验证的是日期过滤本身，所以取 100 天以稳定命中缓存、不碰网络。
    """
    f = _make_fetcher(tmp_path)
    start = (date.today() - timedelta(days=99)).strftime("%Y-%m-%d")
    end = date.today().strftime("%Y-%m-%d")
    df = f.get_daily_kline("600519", start_date=start, end_date=end)
    assert len(df) == 100
    assert df["date"].min().date() == date.today() - timedelta(days=99)
    assert df["date"].max().date() == date.today()

# -*- coding: utf-8 -*-
"""parquet 缓存的原子写入与损坏留证

两个真实故障（2026-11-05 排查）
-------------------------------
1. 日志里出现过 ``magic bytes not found in footer``（600126 / 603606）。
   parquet 的 footer 是**最后**写的，写入过程被中断就必然缺 footer ——
   这台机器经常被关机/休眠打断（``launcher.log`` 的 ``1073807364`` 连击）。
   直接 ``to_parquet(目标路径)`` = 「先毁掉旧文件，再尝试写新文件」。

2. 更严重的是**读失败后的错误路径**：旧实现在 ``except`` 里照常
   ``to_parquet(cache_file)``，写进去的只是本次拉到的区间（往往 30 天）。
   缓存文件名不含区间 ⇒ 下一次读到的「历史」只剩 30 天，训练样本量悄悄缩水，
   且**不报任何错** —— 正是 ``_merge_into_cache`` 文档开头警告的那个灾难，
   而错误路径自己复现了它。

本文件守四件事：原子写、不留 ``.tmp``、损坏必须留证、合并失败保住旧缓存。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import pytest

import src.data.fetcher as fetcher


def _df(dates, close=None):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "close": close if close is not None else [10.0] * len(dates),
    })


@pytest.fixture
def fx(tmp_path):
    return fetcher.DataFetcher(cache_dir=tmp_path), tmp_path


def _tmps(d: Path):
    return [p.name for p in d.iterdir() if p.name.endswith(".tmp")]


# ======================================================================
# ① 原子写
# ======================================================================

def test_atomic_write_produces_readable_file_without_tmp(tmp_path):
    p = tmp_path / "daily_600000_qfq.parquet"
    fetcher._atomic_to_parquet(_df(["2026-11-03", "2026-11-04"]), p)

    assert p.exists()
    assert _tmps(tmp_path) == [], "临时文件必须被 rename 掉，不能留残骸"
    assert len(pd.read_parquet(p)) == 2


def test_atomic_write_creates_missing_directory(tmp_path):
    p = tmp_path / "nested" / "deep" / "x.parquet"
    fetcher._atomic_to_parquet(_df(["2026-11-04"]), p)
    assert p.exists() and len(pd.read_parquet(p)) == 1


def test_failed_write_leaves_old_file_intact(tmp_path, monkeypatch):
    """写盘中途失败（关机/休眠/磁盘满）⇒ 旧文件必须**一字未改**。

    这是「先毁掉旧文件再写新的」与原子写的唯一差别，也是本组用例的核心。
    """
    p = tmp_path / "daily_600000_qfq.parquet"
    fetcher._atomic_to_parquet(_df(["2026-11-03", "2026-11-04"]), p)
    before = p.read_bytes()

    def _boom(self, *a, **kw):
        raise OSError("模拟写入中断")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
    with pytest.raises(OSError):
        fetcher._atomic_to_parquet(_df(["2026-11-05"]), p)

    assert p.read_bytes() == before, "旧缓存被改动了 —— 原子写没生效"
    assert _tmps(tmp_path) == [], "失败路径也必须清掉半截临时文件"
    assert len(pd.read_parquet(p)) == 2


# ======================================================================
# ② 损坏缓存：留证，而不是覆盖
# ======================================================================

def test_corrupt_cache_is_quarantined_not_silently_overwritten(fx):
    f, d = fx
    cache = d / "daily_600126_qfq.parquet"
    cache.write_bytes(b"PAR1" + b"\x00" * 40)     # 缺 footer 的典型形态

    out = f._merge_into_cache(cache, _df(["2026-11-05"]))

    baks = list(d.glob("daily_600126_qfq.parquet.corrupt-*"))
    assert len(baks) == 1, "损坏的缓存必须改名留证，不能直接覆盖掉"
    assert baks[0].read_bytes().startswith(b"PAR1"), "留证文件内容必须原样"
    # 新缓存可读，且只含本次拉到的区间（历史确实丢了 —— 所以要留证 + 报警）
    assert len(pd.read_parquet(cache)) == 1
    assert len(out) == 1


def test_quarantine_returns_empty_when_file_missing(tmp_path):
    assert fetcher._quarantine(tmp_path / "nope.parquet") == ""


# ======================================================================
# ③ 正常合并
# ======================================================================

def test_merge_preserves_history_and_upserts_by_date(fx):
    f, d = fx
    cache = d / "daily_000001_qfq.parquet"
    fetcher._atomic_to_parquet(
        _df(["2026-11-03", "2026-11-04"], [10.0, 11.0]), cache)

    # 只请求最近一天（追踪回测的典型形态）—— 历史绝不能被压成一天
    out = f._merge_into_cache(cache, _df(["2026-11-04"], [11.5]))

    assert len(out) == 2, "按区间请求把历史压短了"
    assert float(out.loc[out["date"] == pd.Timestamp("2026-11-04"), "close"].iloc[0]) == 11.5
    assert len(pd.read_parquet(cache)) == 2


def test_new_dates_are_appended(fx):
    f, d = fx
    cache = d / "daily_000001_qfq.parquet"
    fetcher._atomic_to_parquet(_df(["2026-11-03"]), cache)
    out = f._merge_into_cache(cache, _df(["2026-11-04"]))
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2026-11-03", "2026-11-04"]


def test_empty_existing_cache_is_replaced(fx):
    f, d = fx
    cache = d / "daily_000001_qfq.parquet"
    fetcher._atomic_to_parquet(_df([], []).astype({"date": "datetime64[ns]"}), cache)
    out = f._merge_into_cache(cache, _df(["2026-11-04"]))
    assert len(out) == 1


# ======================================================================
# ④ 合并失败：保住旧缓存，别拿 30 天换全量
# ======================================================================

def test_merge_failure_keeps_old_cache_and_returns_it(fx):
    """字段变了导致合并不了 ⇒ **不写盘**，返回完整的旧历史。

    缺最近几行是可用性问题；把全量历史换成 30 天是数据事故。
    """
    f, d = fx
    cache = d / "daily_000001_qfq.parquet"
    old = pd.DataFrame({"close": [10.0, 11.0]})     # 没有 date 列 → 合并必失败
    old.to_parquet(cache, index=False)
    before = cache.read_bytes()

    out = f._merge_into_cache(cache, _df(["2026-11-04"]))

    assert cache.read_bytes() == before, "合并失败却写了盘"
    assert list(out.columns) == ["close"] and len(out) == 2


def test_corrupt_then_merge_does_not_raise(fx):
    """损坏缓存这条路径不能把整个 get_daily 打挂（异常要继续可重试）。"""
    f, d = fx
    cache = d / "daily_603606_qfq.parquet"
    cache.write_bytes(b"\x00" * 16)
    out = f._merge_into_cache(cache, _df(["2026-11-04"]))
    assert len(out) == 1


# ======================================================================
# ⑤ 常态化体检：损坏必须被*发现*，不能靠谁偶然翻日志
# ======================================================================

def test_scan_finds_only_broken_files(tmp_path):
    good = tmp_path / "daily_600000_qfq.parquet"
    fetcher._atomic_to_parquet(_df(["2026-11-04"]), good)
    (tmp_path / "daily_600126_qfq.parquet").write_bytes(b"PAR1" + b"\x00" * 40)
    (tmp_path / "alpha_margin.parquet").write_bytes(b"")

    bad = fetcher.scan_corrupt_caches(tmp_path)
    names = {b["file"] for b in bad}
    assert names == {"daily_600126_qfq.parquet", "alpha_margin.parquet"}
    assert all(b["error"] for b in bad)


def test_scan_is_silent_on_clean_dir(tmp_path):
    fetcher._atomic_to_parquet(_df(["2026-11-04"]), tmp_path / "daily_000001_qfq.parquet")
    assert fetcher.scan_corrupt_caches(tmp_path) == []


def test_scan_survives_missing_dir(tmp_path):
    assert fetcher.scan_corrupt_caches(tmp_path / "nope") == []


def test_job_data_update_reports_corrupt_caches(monkeypatch, tmp_path):
    """数据更新作业顺手体检 —— 且**体检失败绝不能把作业打挂**。"""
    import src.scheduler.auto_scheduler as sch

    sentinel = {"file": "daily_600126_qfq.parquet", "size": 1, "error": "ArrowInvalid"}
    monkeypatch.setattr(fetcher, "scan_corrupt_caches", lambda *a, **kw: [sentinel])
    monkeypatch.setattr(fetcher, "update_data", lambda: None, raising=False)
    info = sch._job_data_update_impl()
    assert info["n_corrupt_caches"] == 1
    assert info["corrupt_caches"][0]["file"] == "daily_600126_qfq.parquet"


def test_job_data_update_survives_scan_explosion(monkeypatch):
    import src.scheduler.auto_scheduler as sch

    def _boom(*a, **kw):
        raise RuntimeError("体检炸了")

    monkeypatch.setattr(fetcher, "scan_corrupt_caches", _boom)
    monkeypatch.setattr(fetcher, "update_data", lambda: None, raising=False)
    info = sch._job_data_update_impl()
    assert info["status"] == "ok", "体检异常不该改变作业状态"
    assert "corrupt_caches" not in info

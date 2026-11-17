"""「等 data_update 写完」的契约测试（``daily_pick._wait_for_data_update``）。

背景（2026-11-18 实测）：凌晨补跑时出现过双份全池抓取 —— 常驻 run_loop 在跑
``data_update``，另一进程补跑的 ``daily_pick`` 因为缓存还是旧的又自己拉了 425 只。
两拨 8 进程并发抢同一条网络，拉行情从正常 130s 涨到 **393s**（占整次 79%），
而那批数据 daily_pick 一行都不需要：等它写完就能直接命中缓存。

本文件钉死三件事：
  1. 锁在跑 → 等它消失（而不是重复拉）
  2. 锁陈旧（进程已死）→ 立刻放弃等待，绝不卡死选股
  3. 超时 → 放弃等待继续跑（可用性优先）
"""

from __future__ import annotations

import os
import time

from src.core import daily_pick as dp


def _lock(tmp_path, monkeypatch, age_sec: float = 0.0):
    p = tmp_path / ".job_data_update.lock"
    p.write_text("12345", encoding="utf-8")
    if age_sec:
        old = time.time() - age_sec
        os.utime(p, (old, old))
    monkeypatch.setattr(dp, "_DATA_UPDATE_LOCK", str(p))
    return p


def test_no_lock_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "_DATA_UPDATE_LOCK", str(tmp_path / "nope.lock"))
    assert dp._wait_for_data_update(max_wait=5) == 0.0


def test_disabled_by_zero(tmp_path, monkeypatch):
    _lock(tmp_path, monkeypatch)
    assert dp._wait_for_data_update(max_wait=0) == 0.0


def test_stale_lock_does_not_wait(tmp_path, monkeypatch):
    """锁已 2000s（> 1800s 陈旧阈值）→ 持锁进程应已死，不等。"""
    _lock(tmp_path, monkeypatch, age_sec=2000)
    t0 = time.time()
    assert dp._wait_for_data_update(max_wait=30, poll=0.05) == 0.0
    assert time.time() - t0 < 1.0


def test_waits_until_lock_released(tmp_path, monkeypatch):
    """锁在跑 → 等它消失；消失后立刻返回，不再多等。"""
    p = _lock(tmp_path, monkeypatch)
    import threading
    threading.Timer(0.3, lambda: p.unlink(missing_ok=True)).start()
    waited = dp._wait_for_data_update(max_wait=10, poll=0.05)
    assert 0.2 < waited < 5.0, f"实际 waited={waited}"
    assert not p.exists()


def test_times_out_and_keeps_going(tmp_path, monkeypatch):
    """超时必须返回（可用性优先），不能把选股挂死。"""
    _lock(tmp_path, monkeypatch)
    waited = dp._wait_for_data_update(max_wait=0.5, poll=0.1)
    assert 0.4 <= waited <= 3.0, f"实际 waited={waited}"


def test_lock_vanishing_mid_wait_is_handled(tmp_path, monkeypatch):
    """等待过程中锁被别的进程清掉（stat 失败）不能抛异常。"""
    p = _lock(tmp_path, monkeypatch)
    p.unlink()
    assert dp._wait_for_data_update(max_wait=2, poll=0.05) == 0.0

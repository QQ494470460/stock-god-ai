# -*- coding: utf-8 -*-
"""收盘后调度器的「跨进程单实例」 + 每日推荐理由口径融合的回归测试。

背景
====
调度器会被**两个进程**拉起：看板（app.py）和常驻 ``scripts/run_loop.py``。
两者的 APScheduler 互不知情，18:45 的 daily_pick 会被起两份 —— 用户称之为
"双调度"。修法是给「调度器本体」加一把进程级文件锁 ``data/.scheduler.lock``：
谁抢到谁才真正创建 APScheduler，另一个进程主动退让。

同时 ``is_running()`` 不能只报本进程状态 —— 调度若由常驻进程持有，看板会
显示"未启动"，而作业其实照常在跑，属于误导。

理由口径：交易计划顶层 ``regime`` 是**指数基准**的大盘口径，而个股
``推荐理由`` 来自 scoring 的 ``regime_note``（个股口径），二者可以合法不一致；
但 ``regime: bear`` + ``reason: 牛市顺风`` 看着像 bug，这里把两口径融合进文案。
"""

from pathlib import Path

import pytest

from src.core.daily_pick import _annotate_reason
from src.core.single_instance import SingleInstanceLock
from src.scheduler import auto_scheduler as sch


# ---------------------------------------------------------------------------
# 理由口径融合
# ---------------------------------------------------------------------------

def test_reason_bear_market_bull_stock_is_called_out():
    """大盘熊市 + 个股走强 → 必须是「逆势」，不能还写「顺风」。"""
    out = _annotate_reason("个股牛市顺风", "bear")
    assert "逆势" in out
    assert "大盘熊市" in out
    assert "顺风" not in out, "熊市里还写顺风会把人看懵"


def test_reason_bull_market_resonance():
    """大盘牛市 + 个股走强 → 共振。"""
    out = _annotate_reason("个股牛市顺风", "bull")
    assert "共振" in out
    assert "大盘牛市" in out


def test_reason_without_market_regime_passes_through():
    """拿不到大盘 regime（检测失败/None）时不应凭空造语境。"""
    assert _annotate_reason("个股震荡", None) == "个股震荡"
    assert _annotate_reason("个股震荡", "") == "个股震荡"


def test_reason_empty_is_safe():
    assert _annotate_reason("", "bear") == ""
    assert _annotate_reason(None, "bear") == ""


# ---------------------------------------------------------------------------
# 调度器跨进程单实例
# ---------------------------------------------------------------------------

def _patch_lock_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(sch, "SCHED_LOCK_DIR", Path(tmp_path), raising=False)
    monkeypatch.setattr(sch, "_scheduler", None, raising=False)
    monkeypatch.setattr(sch, "_sched_holder", None, raising=False)
    return Path(tmp_path) / ".scheduler.lock"


def test_start_scheduler_refuses_when_another_process_holds_lock(monkeypatch, tmp_path):
    """别人持有调度锁时，本进程必须退让，不再起第二份 APScheduler。"""
    lockp = _patch_lock_dir(monkeypatch, tmp_path)
    holder = SingleInstanceLock(lockp)
    assert holder.acquire() is True
    try:
        monkeypatch.setattr(sch.app_config.scheduler, "enabled", True, raising=False)
        assert sch.start_scheduler() is None, "锁被占用时不该再拉起第二份调度"
        assert sch._scheduler is None
    finally:
        holder.release()


def test_is_running_true_when_lock_held_by_other_process(monkeypatch, tmp_path):
    """调度由常驻进程持有时，is_running() 也要报 True（否则看板误显示未启动）。"""
    lockp = _patch_lock_dir(monkeypatch, tmp_path)
    holder = SingleInstanceLock(lockp)
    assert holder.acquire() is True
    try:
        assert sch.is_running() is True
    finally:
        holder.release()


def test_is_running_false_when_nobody_holds(monkeypatch, tmp_path):
    _patch_lock_dir(monkeypatch, tmp_path)
    assert sch.is_running() is False


def test_stop_scheduler_releases_the_process_lock(monkeypatch, tmp_path):
    """停止后必须释放锁，否则别的进程再也接不了手（重启后调度永久失效）。"""
    lockp = _patch_lock_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(sch.app_config.scheduler, "enabled", True, raising=False)

    sched = sch.start_scheduler()
    assert sched is not None, "空闲时应当能抢到锁并起调度"
    try:
        assert sch.is_running() is True
    finally:
        assert sch.stop_scheduler() is True

    # 释放后别人能抢到
    assert SingleInstanceLock(lockp).acquire() is True
    assert sch.is_running() is False

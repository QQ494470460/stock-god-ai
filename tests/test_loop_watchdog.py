# -*- coding: utf-8 -*-
"""闭环存活看门狗：只认心跳年龄，超时告警 + 冷却后拉起启动器。

背景见 ``src/scheduler/loop_watchdog.py`` 顶部：2026-10-29 机器 11:50 关机，
整个下午没有任何止损监控，罗博特科穿到 -16.65% 无人处置，最后按收盘价补刀。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.scheduler import loop_watchdog as wd


# 2026-10-30 是周五，10:00 落在交易时段内
IN_WINDOW = datetime(2026, 10, 30, 10, 0, 0)
OUT_WINDOW = datetime(2026, 10, 30, 20, 0, 0)   # 收盘后
WEEKEND = datetime(2026, 10, 31, 10, 0, 0)      # 周六


def _hb(tmp_path: Path, age_sec: float, now: datetime) -> Path:
    p = tmp_path / "loop_heartbeat.json"
    p.write_text(json.dumps({
        "ts": (now - timedelta(seconds=age_sec)).isoformat(timespec="seconds"),
        "pid": 1234, "state": "halted", "equity": 1.0,
    }, ensure_ascii=False), encoding="utf-8")
    return p


def _paths(tmp_path: Path):
    return dict(heartbeat_path=tmp_path / "loop_heartbeat.json",
                state_path=tmp_path / ".watchdog_last_attempt",
                log_path=tmp_path / "watchdog.log")


class _Spawn:
    def __init__(self, ok=True):
        self.calls = 0
        self.ok = ok

    def __call__(self):
        self.calls += 1
        return self.ok


# ---------------------------------------------------------------- 时段
def test_in_monitoring_window_rules():
    assert wd.in_monitoring_window(IN_WINDOW) is True
    assert wd.in_monitoring_window(OUT_WINDOW) is False
    assert wd.in_monitoring_window(WEEKEND) is False
    # 边界：09:15 开始、15:10 结束（收盘后留 10 分钟给结算 tick）
    assert wd.in_monitoring_window(datetime(2026, 10, 30, 9, 14, 59)) is False
    assert wd.in_monitoring_window(datetime(2026, 10, 30, 9, 15, 0)) is True
    assert wd.in_monitoring_window(datetime(2026, 10, 30, 15, 10, 0)) is True
    assert wd.in_monitoring_window(datetime(2026, 10, 30, 15, 10, 1)) is False


# ---------------------------------------------------------------- 存活
def test_fresh_heartbeat_is_ok_and_does_not_spawn(tmp_path):
    sp = _Spawn()
    p = _paths(tmp_path)
    _hb(tmp_path, age_sec=30, now=IN_WINDOW)
    res = wd.check(now=IN_WINDOW, spawn=sp, **p)
    assert res["ok"] is True
    assert res["action"] == "none"
    assert res["age_sec"] == pytest.approx(30.0, abs=1.5)
    assert sp.calls == 0


def test_stale_heartbeat_alerts_and_spawns(tmp_path):
    sp = _Spawn()
    p = _paths(tmp_path)
    _hb(tmp_path, age_sec=900, now=IN_WINDOW)
    res = wd.check(now=IN_WINDOW, spawn=sp, **p)
    assert res["ok"] is False
    assert res["action"] == "restart"
    assert sp.calls == 1
    assert "停" in res["detail"]
    assert p["log_path"].exists(), "告警必须落盘，否则事后无从追溯"
    assert "ALERT" in p["log_path"].read_text(encoding="utf-8")


def test_stale_outside_window_only_reports(tmp_path):
    """收盘后心跳停了不该拉起 —— 夜里重启毫无意义，只会污染日志。"""
    sp = _Spawn()
    p = _paths(tmp_path)
    _hb(tmp_path, age_sec=900, now=OUT_WINDOW)
    res = wd.check(now=OUT_WINDOW, spawn=sp, **p)
    assert res["ok"] is False
    assert res["in_window"] is False
    assert res["action"] == "none"
    assert sp.calls == 0


def test_missing_heartbeat_counts_as_dead(tmp_path):
    sp = _Spawn()
    p = _paths(tmp_path)
    res = wd.check(now=IN_WINDOW, spawn=sp, **p)   # 文件不存在
    assert res["age_sec"] is None
    assert res["ok"] is False
    assert res["action"] == "restart"
    assert sp.calls == 1


def test_restart_false_only_alerts(tmp_path):
    sp = _Spawn()
    p = _paths(tmp_path)
    _hb(tmp_path, age_sec=900, now=IN_WINDOW)
    res = wd.check(now=IN_WINDOW, restart=False, spawn=sp, **p)
    assert res["action"] == "alert"
    assert sp.calls == 0


def test_spawn_failure_keeps_alert_action(tmp_path):
    sp = _Spawn(ok=False)
    p = _paths(tmp_path)
    _hb(tmp_path, age_sec=900, now=IN_WINDOW)
    res = wd.check(now=IN_WINDOW, spawn=sp, **p)
    assert res["action"] == "alert"
    assert "拉起失败" in res["detail"]


# ---------------------------------------------------------------- 冷却
def test_cooldown_blocks_repeated_spawn(tmp_path):
    sp = _Spawn()
    p = _paths(tmp_path)
    _hb(tmp_path, age_sec=900, now=IN_WINDOW)
    wd.check(now=IN_WINDOW, spawn=sp, **p)
    wd.check(now=IN_WINDOW + timedelta(seconds=120), spawn=sp, **p)
    assert sp.calls == 1, "10 分钟冷却内不得反复拉起启动器"
    # 冷却过后可以再试
    wd.check(now=IN_WINDOW + timedelta(seconds=700), spawn=sp, **p)
    assert sp.calls == 2


def test_recovery_clears_cooldown_state(tmp_path):
    """恢复后要清冷却记录，否则下一次事故会被"上次的 10 分钟"挡住。"""
    sp = _Spawn()
    p = _paths(tmp_path)
    _hb(tmp_path, age_sec=900, now=IN_WINDOW)
    wd.check(now=IN_WINDOW, spawn=sp, **p)
    assert p["state_path"].exists()

    _hb(tmp_path, age_sec=10, now=IN_WINDOW + timedelta(seconds=60))
    wd.check(now=IN_WINDOW + timedelta(seconds=60), spawn=sp, **p)
    assert not p["state_path"].exists()

    # 立刻再出事 → 应该能立即拉起（不受上一次冷却影响）
    _hb(tmp_path, age_sec=900, now=IN_WINDOW + timedelta(seconds=120))
    res = wd.check(now=IN_WINDOW + timedelta(seconds=120), spawn=sp, **p)
    assert res["action"] == "restart"
    assert sp.calls == 2


# ---------------------------------------------------------------- 节流
def test_watchdog_tick_throttles_to_interval(tmp_path):
    sp = _Spawn()
    p = _paths(tmp_path)
    _hb(tmp_path, age_sec=30, now=IN_WINDOW)
    dog = wd.LoopWatchdog(interval=60.0, now=IN_WINDOW, spawn=sp, **p)
    assert dog.tick(0.0) is not None          # 首次跑
    assert dog.tick(30.0) is None             # 未到间隔
    assert dog.tick(60.0) is not None         # 到点再跑


def test_watchdog_tick_swallows_errors(tmp_path):
    """看门狗自己崩了不能把常驻调度进程带崩，也不能把日志写脏。"""
    def boom(**_kw):
        raise RuntimeError("故意炸")

    log = tmp_path / "watchdog.log"
    dog = wd.LoopWatchdog(interval=0.0, now=IN_WINDOW, log_path=log)
    import src.scheduler.loop_watchdog as mod
    orig = mod.check
    mod.check = boom
    try:
        assert dog.tick(1.0) is None
    finally:
        mod.check = orig
    assert log.exists() and "故意炸" in log.read_text(encoding="utf-8")


# ---------------------------------------------------------------- 超级集成降级（进程存活但腿静默降级）
def _hb_full(tmp_path: Path, age_sec: float, now: datetime, **extra) -> Path:
    p = tmp_path / "loop_heartbeat.json"
    d = {"ts": (now - timedelta(seconds=age_sec)).isoformat(timespec="seconds"),
         "pid": 1234, "state": "running", "equity": 1.0}
    d.update(extra)
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return p


def test_alive_but_ensemble_degraded_surfaces_in_result(tmp_path):
    sp = _Spawn()
    p = _paths(tmp_path)
    _hb_full(tmp_path, age_sec=30, now=IN_WINDOW,
             ensemble_degraded=True,
             ensemble_alarm="⚠️ 超级集成有腿静默降级：\n  · lstm：RuntimeError: lstm_best.pt 缺失")
    res = wd.check(now=IN_WINDOW, spawn=sp, **p)
    assert res["ok"] is True
    assert res["action"] == "none"          # 进程活着 → 绝不重启
    assert sp.calls == 0
    assert res["ensemble_degraded"] is True
    assert "降级" in res["detail"]
    assert "lstm" in (res["ensemble_alarm"] or "")


def test_alive_degraded_writes_warn_log(tmp_path):
    sp = _Spawn()
    p = _paths(tmp_path)
    _hb_full(tmp_path, age_sec=30, now=IN_WINDOW, ensemble_degraded=True,
             ensemble_alarm="⚠️ 超级集成有腿静默降级：\n  · lstm：x")
    wd.check(now=IN_WINDOW, spawn=sp, **p)
    log = p["log_path"].read_text(encoding="utf-8")
    assert "WARN 超级集成有腿静默降级" in log


def test_alive_no_ensemble_field_defaults_not_degraded(tmp_path):
    """旧心跳文件（无 ensemble_degraded 字段）不应误报降级 —— 默认关闭、零噪声。"""
    sp = _Spawn()
    p = _paths(tmp_path)
    _hb(tmp_path, age_sec=30, now=IN_WINDOW)     # 旧格式，无 ensemble 字段
    res = wd.check(now=IN_WINDOW, spawn=sp, **p)
    assert res["ensemble_degraded"] is False
    assert "降级" not in res["detail"]
    assert res["ensemble_alarm"] is None

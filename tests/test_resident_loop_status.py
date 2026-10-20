# -*- coding: utf-8 -*-
"""is_resident_loop_running 的纯逻辑测试（不依赖浏览器/Streamlit）。"""
import importlib

import src.trading.loop as loop_mod


def _reload():
    importlib.reload(loop_mod)
    return loop_mod


def test_no_heartbeat_means_not_running(monkeypatch):
    mod = _reload()
    monkeypatch.setattr(mod, "read_heartbeat", lambda: {})
    r = mod.is_resident_loop_running()
    assert r["running"] is False
    assert r["pid"] is None


def test_fresh_heartbeat_means_running(monkeypatch):
    mod = _reload()
    monkeypatch.setattr(mod, "read_heartbeat", lambda: {
        "pid": 5204, "age_sec": 12.5, "state": "preopen",
        "ts": "2026-10-20T09:11:59",
    })
    r = mod.is_resident_loop_running(stale_sec=120.0)
    assert r["running"] is True
    assert r["pid"] == 5204
    assert r["state"] == "preopen"
    assert r["age_sec"] == 12.5


def test_stale_heartbeat_means_not_running(monkeypatch):
    mod = _reload()
    monkeypatch.setattr(mod, "read_heartbeat", lambda: {
        "pid": 5204, "age_sec": 3600.0, "state": "idle",
        "ts": "2026-10-19T23:38:06",
    })
    r = mod.is_resident_loop_running(stale_sec=120.0)
    assert r["running"] is False
    assert r["pid"] == 5204  # 仍回传 pid，但 running=False（心跳过期）


def test_custom_stale_threshold(monkeypatch):
    mod = _reload()
    monkeypatch.setattr(mod, "read_heartbeat", lambda: {
        "pid": 1, "age_sec": 90.0, "state": "running",
    })
    # 阈值设 30s → 90s 视为过期
    assert mod.is_resident_loop_running(stale_sec=30.0)["running"] is False
    # 阈值设 120s → 90s 视为存活
    assert mod.is_resident_loop_running(stale_sec=120.0)["running"] is True

# -*- coding: utf-8 -*-
"""调度开关持久化：pick_ensemble 与 enabled 共用一个文件，不许互相覆盖。

为什么值得单独测
----------------
两个开关写同一个 JSON，最典型的翻车方式是「整份覆盖写」：
用户在看板里先开了多策略复核，某天又点了一下「启用调度」，
结果 enabled 存了、pick_ensemble 被写成默认值 —— 复核静默关闭，
且没有任何报错。同样的坑反向也成立（改复核时把调度关掉）。

全部用 tmp_path 重定向状态文件，不碰 config/scheduler_state.json。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config.settings as st  # noqa: E402


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    """把状态文件重定向到 tmp，并复位内存里的两个开关。"""
    f = tmp_path / "scheduler_state.json"
    monkeypatch.setattr(st, "_SCHEDULER_STATE_FILE", f)
    monkeypatch.setattr(st.config.scheduler, "enabled", False)
    monkeypatch.setattr(st.config.scheduler, "pick_ensemble", False)
    return f


def _read(f):
    return json.loads(f.read_text(encoding="utf-8"))


def test_saving_enabled_preserves_ensemble(state_file):
    st.config.scheduler.pick_ensemble = True
    st.save_scheduler_state(True)

    assert _read(state_file)["pick_ensemble"] is True, "保存调度开关时把复核开关冲掉了"


def test_saving_ensemble_preserves_enabled(state_file):
    st.config.scheduler.enabled = True
    st.save_pick_ensemble_state(True)

    read = _read(state_file)
    assert read["pick_ensemble"] is True
    assert read["enabled"] is True, "保存复核开关时把调度开关冲掉了"


def test_ensemble_state_updates_live_config(state_file):
    st.save_pick_ensemble_state(True)

    assert st.config.scheduler.pick_ensemble is True, "存了盘但没生效，等于白开"


def test_load_restores_both_flags(state_file):
    state_file.write_text(json.dumps({"enabled": True, "pick_ensemble": True}),
                          encoding="utf-8")
    st.load_scheduler_state()

    assert st.config.scheduler.enabled is True
    assert st.config.scheduler.pick_ensemble is True


def test_load_tolerates_legacy_file_without_ensemble_key(state_file):
    """旧文件只有 enabled 一个键 —— 不能因为缺键就把复核默许打开。"""
    state_file.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    st.load_scheduler_state()

    assert st.config.scheduler.enabled is True
    assert st.config.scheduler.pick_ensemble is False


def test_load_survives_corrupted_file(state_file):
    state_file.write_text("{ not json", encoding="utf-8")
    st.load_scheduler_state()   # 不应抛异常
    assert st.config.scheduler.enabled is False

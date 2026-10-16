# -*- coding: utf-8 -*-
"""evolve CLI 的纯函数部分：参数透传与前置条件自检。

``--doctor`` 是这套回路的「体检入口」：它的价值全在于**把静默失败说出来**。
所以测试重点不是「它能不能跑」，而是「它有没有如实报出阻塞项、给对命令」。
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.evolve as ev  # noqa: E402


# ---------------------------------------------------------------------------
# --ga-* → 生成器 kwargs
# ---------------------------------------------------------------------------

def _ga_ns(**kw):
    base = dict(ga_population=None, ga_generations=None,
                ga_elite=None, ga_seed=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_generator_kwargs_only_passes_explicit_values():
    """只收集显式给过值的参数 —— 默认值始终由生成器自己说了算。"""
    got = ev._generator_kwargs(_ga_ns(ga_population=16, ga_elite=3))
    assert got == {"genetic": {"population": 16, "elite": 3}}


def test_generator_kwargs_empty_when_nothing_given():
    assert ev._generator_kwargs(_ga_ns()) == {}


def test_generator_kwargs_keeps_zero_and_negative():
    """0 / 负数也要原样透传（由生成器负责收敛），不能在 CLI 层被当成「没给」。"""
    got = ev._generator_kwargs(_ga_ns(ga_generations=0, ga_seed=-1))
    assert got == {"genetic": {"generations": 0, "seed": -1}}


# ---------------------------------------------------------------------------
# --doctor
# ---------------------------------------------------------------------------

def _isolate(monkeypatch, tmp_path):
    import config.settings as cs
    import src.evolution.feedback as fb
    import src.evolution.panels as panels
    from src.models import model_registry as mr

    monkeypatch.setattr(panels, "panel_available", lambda: False)
    monkeypatch.setattr(panels, "panel_meta", lambda: {})
    monkeypatch.setattr(fb, "LOOP_REPORTS_DIR", tmp_path / "loop_reports")
    monkeypatch.setattr(fb, "collect_prediction_feedback",
                        lambda *a, **k: {"n_verified": 0, "pending_verify": 0,
                                         "missing_snapshot": 0})
    monkeypatch.setattr(mr, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(mr, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cs, "DATA_DIR", tmp_path / "data")
    return panels, fb, mr


def test_doctor_exits_2_and_names_the_blockers(monkeypatch, tmp_path, capsys):
    panels, fb, mr = _isolate(monkeypatch, tmp_path)

    rc = ev._doctor()
    out = capsys.readouterr().out

    assert rc == 2, "缺面板 + 闭环从未结算 = 两项阻塞，退出码必须是 2"
    assert "样本外面板" in out and "闭环结算" in out
    assert "--build-panel" in out, "阻塞项必须连同「该跑什么命令」一起给出"
    assert "结论" in out


def test_doctor_exits_0_when_no_blockers(monkeypatch, tmp_path, capsys):
    panels, fb, mr = _isolate(monkeypatch, tmp_path)

    # 面板就绪
    monkeypatch.setattr(panels, "panel_available", lambda: True)
    monkeypatch.setattr(panels, "panel_meta",
                        lambda: {"n_days": 120, "n_codes": 200,
                                 "start": "2024-01-01", "end": "2024-12-31"})
    # 可作主模型的产物就绪
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    for f in ("xgboost_model.pkl", "scaler.pkl", "features_xgboost.json"):
        (models / f).write_bytes(b"x")
    (models / "production_metrics.json").write_text(json.dumps({
        "walk_forward": {"rank_ic": 0.075, "n_days": 200},
        "model_path": "xgboost_model.pkl",
    }), encoding="utf-8")
    # 闭环已结算过
    (tmp_path / "loop_reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "loop_reports" / "loop_report_20261013.json").write_text(
        "{}", encoding="utf-8")

    rc = ev._doctor()
    out = capsys.readouterr().out

    assert rc == 0, f"无阻塞项时退出码必须为 0。输出：\n{out}"
    assert "全部就绪" in out or "待办" in out


def test_doctor_is_read_only(monkeypatch, tmp_path):
    """自检绝不能产生副作用 —— 它被设计成「可以随时跑一下」。"""
    _isolate(monkeypatch, tmp_path)
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    guard = data / "trading_guard_state.json"
    guard.write_text('{"halted": false}', encoding="utf-8")
    before = guard.read_bytes()

    ev._doctor()

    assert guard.read_bytes() == before
    assert not any(data.glob("**/*.tmp")), "自检不该留下临时文件"

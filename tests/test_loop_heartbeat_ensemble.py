# -*- coding: utf-8 -*-
"""闭环心跳透出超级集成健康（集成单测，不触网/不加载真实模型）。

验证 loop.py 的 _write_heartbeat 在超级集成启用时，把 ensemble_health /
ensemble_alarm / ensemble_degraded 一并写进 loop_heartbeat.json，使看门狗与
get_status()/read_heartbeat 能在盘中 tick 级别实时发现「腿静默降级」；
未启用时则绝不写入（默认关闭、零行为改变）。

predictor.ensemble_report / ensemble_degradation_alert 用 monkeypatch 固定返回值，
把「心跳并入」与「审计本身」解耦 —— 审计逻辑由 test_ensemble_health.py 覆盖。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config.settings import config as _app_cfg
from src.trading.loop import TradingLoop, LoopConfig, TickReport, read_heartbeat
from src.models.predictor import predictor as _predictor
from tests.test_loop import mk_loop


def _mk_report():
    return TickReport(
        at="2026-11-12T10:00:00",
        state="running",
        market_status="交易中",
        plan_date="2026-11-12",
        n_positions=3,
        equity=101000.0,
        n_intents=2,
        n_executed=2,
        n_blocked=0,
    )


def _mk_loop(tmp_path):
    # 复用 test_loop 的组装：FakeBroker + FakePaper + 自动构建 PAPER guard，
    # 仅把心跳文件指向 tmp；不触网、不加载真实模型。
    return mk_loop(cfg=LoopConfig(
        auto_build_plan=False,
        heartbeat_file=str(tmp_path / "hb.json")))


def test_heartbeat_includes_ensemble_fields_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(_app_cfg.ml, "use_ensemble", True)
    monkeypatch.setattr(_predictor, "ensemble_report",
                        lambda: "🧩 超级集成健康审计：\n  · alpha_xgboost：✅ 可用")
    monkeypatch.setattr(_predictor, "ensemble_degradation_alert",
                        lambda: None)

    loop = _mk_loop(tmp_path)
    loop._write_heartbeat(_mk_report())

    hb = read_heartbeat(loop.heartbeat_file)
    assert "ensemble_health" in hb
    assert "ensemble_alarm" in hb
    assert "ensemble_degraded" in hb
    assert hb["ensemble_health"].startswith("🧩 超级集成健康审计")
    assert hb["ensemble_alarm"] is None
    assert hb["ensemble_degraded"] is False


def test_heartbeat_flags_degraded_when_alarm_present(tmp_path, monkeypatch):
    monkeypatch.setattr(_app_cfg.ml, "use_ensemble", True)
    monkeypatch.setattr(_predictor, "ensemble_report",
                        lambda: "🧩 超级集成健康审计：\n  · lstm：⚠️ 降级")
    monkeypatch.setattr(
        _predictor, "ensemble_degradation_alert",
        lambda: "⚠️ 超级集成有腿静默降级：\n  · lstm：RuntimeError: lstm_best.pt 缺失")

    loop = _mk_loop(tmp_path)
    loop._write_heartbeat(_mk_report())

    hb = read_heartbeat(loop.heartbeat_file)
    assert hb["ensemble_degraded"] is True
    assert "lstm" in hb["ensemble_alarm"]


def test_heartbeat_omits_ensemble_fields_when_disabled(tmp_path, monkeypatch):
    # 确保默认关闭：即便 predictor 真的有数据也不应写入
    monkeypatch.setattr(_app_cfg.ml, "use_ensemble", False)

    loop = _mk_loop(tmp_path)
    loop._write_heartbeat(_mk_report())

    hb = read_heartbeat(loop.heartbeat_file)
    assert "ensemble_health" not in hb
    assert "ensemble_alarm" not in hb
    assert "ensemble_degraded" not in hb
    # 基础心跳字段仍在
    assert hb.get("state") == "running"
    assert "pid" in hb


def test_heartbeat_omits_ensemble_fields_when_audit_raises(tmp_path, monkeypatch):
    """审计本身抛异常（如模型未加载）绝不波及心跳落盘 —— 交易主流程优先级最高。"""
    monkeypatch.setattr(_app_cfg.ml, "use_ensemble", True)
    monkeypatch.setattr(_predictor, "ensemble_report",
                        lambda: (_ for _ in ()).throw(RuntimeError("模型未就绪")))

    loop = _mk_loop(tmp_path)
    loop._write_heartbeat(_mk_report())  # 不应抛

    hb = read_heartbeat(loop.heartbeat_file)
    assert "ensemble_health" not in hb
    assert "ensemble_degraded" not in hb
    assert hb.get("state") == "running"

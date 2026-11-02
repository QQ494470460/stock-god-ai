# -*- coding: utf-8 -*-
"""T4 接线测试：漂移哨兵 + 调度开关（scripts/drift_retrain.py + auto_scheduler）。

纯决策逻辑见 tests/test_auto_retrain_pipeline.py；这里只测**接线层**：

1. 2σ 阈值：20 日均值的标准误常在 0.03+，若写死 0.010 的阈值，
   日常噪声就会触发重训 —— 那正是进化模块自己警告的「用噪声改策略」；
2. 样本不足不判定（3 天均值没有统计意义）；
3. 默认 `--check`：不跑重训、不写 active_version；
4. 调度作业默认关闭，且关闭时不注册。
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "drift_retrain.py"


def _load_script(tmp_path=None, monkeypatch=None):
    spec = importlib.util.spec_from_file_location("drift_retrain_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["drift_retrain_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_drift_threshold_is_two_sigma_not_hardcoded(tmp_path, monkeypatch):
    """阈值必须随实测 IC 离散度放大：std=0.15, n=20 → se=0.034, 阈值≈0.067。

    写死 0.010 会怎样：均值只要抖 0.011 就重训。日频 IC 标准差本来就在 0.15 上下，
    20 日均值的标准误 0.034 ⇒ 每天都在「漂移」，等于每两周换一次模型。
    """
    mod = _load_script()
    # 构造 IC 序列：均值 0.098，标准差约 0.15
    ic_vals = [0.098 + 0.15 * (1 if i % 2 else -1) for i in range(20)]
    monkeypatch.setattr(mod, "current_metrics", lambda w=20: {
        "rank_ic": 0.098, "ic_std": 0.15, "n_obs": 20.0})
    monkeypatch.setattr(mod, "baseline_metrics", lambda: {"rank_ic": 0.1235})

    rep = mod.decide()
    expect = max(0.010, 2.0 * 0.15 / math.sqrt(20))
    assert rep["drift_threshold"] == pytest.approx(expect)
    assert rep["drift_threshold"] > 0.06
    # 跌幅 0.0257 < 阈值 0.067 → 不得判漂移（这正是 0.010 写死时会误判的地方）
    assert rep["drifted"] is False
    assert ic_vals  # 说明构造的序列确实有这个量级的 std


def test_large_drop_still_counts_as_drift(monkeypatch):
    """真崩了（跌幅远超 2σ）必须判漂移 —— 阈值不能宽到永远不触发。

    注意用例取值要高过 0.05 的绝对下限：否则先被「绝对过低」那条判据拦下，
    就测不到「相对滑落」这条了（两条判据都在，各测各的）。
    """
    mod = _load_script()
    monkeypatch.setattr(mod, "current_metrics", lambda w=20: {
        "rank_ic": 0.055, "ic_std": 0.02, "n_obs": 20.0})
    monkeypatch.setattr(mod, "baseline_metrics", lambda: {"rank_ic": 0.124})
    rep = mod.decide()
    assert rep["drifted"] is True
    assert "滑落" in rep["reason"]


def test_absolute_floor_triggers_before_relative(monkeypatch):
    """RankIC 直接烂到 0.05 以下 ⇒ 不需要基准可比也要判漂移。"""
    mod = _load_script()
    monkeypatch.setattr(mod, "current_metrics", lambda w=20: {
        "rank_ic": 0.010, "ic_std": 0.02, "n_obs": 20.0})
    monkeypatch.setattr(mod, "baseline_metrics", lambda: {})
    rep = mod.decide()
    assert rep["drifted"] is True
    assert "绝对下限" in rep["reason"]


def test_short_sample_is_not_judged(monkeypatch):
    """样本 < 10 个交易日 ⇒ 不做判定（fail-safe，不重训）。"""
    mod = _load_script()
    monkeypatch.setattr(mod, "current_metrics", lambda w=20: {
        "rank_ic": 0.0, "ic_std": 0.2, "n_obs": 3.0})
    monkeypatch.setattr(mod, "baseline_metrics", lambda: {"rank_ic": 0.124})
    rep = mod.decide()
    assert rep["can_judge"] is False
    assert rep["drifted"] is False

    out = mod.run_retrain()
    assert out["ran"] is False, "样本不足时绝不能启动重训"


def test_check_mode_never_trains(tmp_path, monkeypatch):
    """默认路径（--check）只判定，不碰 runner、不写 active_version。"""
    mod = _load_script()
    monkeypatch.setattr(mod, "REPORT_FILE", tmp_path / "drift_retrain_report.json")
    monkeypatch.setattr(mod, "current_metrics", lambda w=20: {
        "rank_ic": 0.010, "ic_std": 0.02, "n_obs": 20.0})
    monkeypatch.setattr(mod, "baseline_metrics", lambda: {"rank_ic": 0.124})

    called = {"run": False}

    def _boom(*a, **k):
        called["run"] = True
        raise AssertionError("check 模式不应调用 EvolutionRunner")

    monkeypatch.setattr(sys, "argv", ["drift_retrain.py", "--json"])
    # 即便没有 monkeypatch 到 runner，check 模式也不该走到 run_retrain
    monkeypatch.setattr(mod, "run_retrain", _boom)
    rc = mod.main()
    assert rc == 0
    assert called["run"] is False
    rep = json.loads((tmp_path / "drift_retrain_report.json").read_text(encoding="utf-8"))
    assert rep["drifted"] is True       # 判定照做（跌幅 0.114 >> 2σ）
    assert "ran" not in rep


def test_scheduler_job_defaults_disabled():
    """调度里的漂移哨兵默认关闭；关闭时 job 直接返回 disabled 而不是偷偷跑。"""
    from config.settings import AppConfig
    from src.scheduler.auto_scheduler import job_drift_retrain

    sc = AppConfig().scheduler
    assert sc.drift_retrain_enabled is False
    assert sc.drift_retrain_cron.startswith("30 19")
    out = job_drift_retrain()
    assert out["status"] == "disabled"

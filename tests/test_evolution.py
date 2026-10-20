# -*- coding: utf-8 -*-
"""自我进化引擎测试 —— 重点覆盖「自动重训被僵尸标记卡死」。

线上真实故障
------------
`data/evolution_state.json` 里 `retrain_running: true`，而 `last_retrain_at`
停在 2026-09-02 —— 重训进程中途崩溃，`mark_retrain_finished()` 从未执行，
标记永久留在 True。`should_retrain()` 从此恒返回「已有重训任务在运行中」，
**自动进化彻底瘫痪**（实测卡了 1019 小时 / 约 6 周）。

修复：引入 `retrain_started_at` + `retrain_timeout_hours`（默认 6h），
超时即判为僵尸并自动清理，同时写进化日志。

全部测试使用 tmp_path 状态文件，绝不触碰真实 data/。
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.evolution import EvolutionEngine  # noqa: E402


def _engine(tmp_path, **offline) -> EvolutionEngine:
    """构造一个使用独立状态文件的引擎。"""
    e = EvolutionEngine(state_file=tmp_path / "evo.json")
    cfg = dict(e.state.get("offline_evolution", {}))
    cfg.update(offline)
    e.state["offline_evolution"] = cfg
    e.save()
    return e


def _iso(hours_ago: float) -> str:
    return (datetime.now() - timedelta(hours=hours_ago)).isoformat()


# ---------------------------------------------------------------- 僵尸标记

def test_zombie_retrain_flag_is_cleared(tmp_path):
    """卡死的 retrain_running 必须被自动清理，恢复进化能力。"""
    e = _engine(tmp_path, enabled=True, retrain_running=True,
                retrain_started_at=_iso(48), retrain_timeout_hours=6)

    should, reason = e.should_retrain()

    assert e.state["offline_evolution"]["retrain_running"] is False
    assert e.state["offline_evolution"]["retrain_started_at"] is None
    assert "运行中" not in reason, "僵尸标记清理后不应再报「运行中」"
    # 且清理动作要留痕
    logs = [x for x in e.state["evolution_log"] if "卡死" in x["detail"]]
    assert logs, "清理僵尸标记必须写进化日志"


def test_zombie_legacy_state_without_started_at(tmp_path):
    """老状态文件没有 retrain_started_at，应回退用 last_retrain_at 判定。"""
    e = _engine(tmp_path, enabled=True, retrain_running=True,
                last_retrain_at=_iso(1019), retrain_timeout_hours=6)
    e.state["offline_evolution"].pop("retrain_started_at", None)
    e.save()

    e.should_retrain()

    assert e.state["offline_evolution"]["retrain_running"] is False


def test_fresh_retrain_still_blocks(tmp_path):
    """真正在跑的重训（未超时）必须继续拦住，不能误判成僵尸。"""
    e = _engine(tmp_path, enabled=True, retrain_running=True,
                retrain_started_at=_iso(0.5), retrain_timeout_hours=6,
                last_retrain_at=_iso(200))

    should, reason = e.should_retrain()

    assert should is False
    assert "运行中" in reason
    assert e.state["offline_evolution"]["retrain_running"] is True


def test_no_started_at_and_no_last_retrain_is_zombie(tmp_path):
    """两个时间都没有 = 无法证明在跑，判为僵尸（否则会永久卡死）。"""
    e = _engine(tmp_path, enabled=True, retrain_running=True)
    e.state["offline_evolution"]["retrain_started_at"] = None
    e.state["offline_evolution"]["last_retrain_at"] = None
    e.save()

    e.should_retrain()

    assert e.state["offline_evolution"]["retrain_running"] is False


# ---------------------------------------------------------------- 生命周期

def test_mark_started_records_time_and_finished_clears(tmp_path):
    """mark_retrain_started 记开始时间；finished 清标记并更新 last_retrain_at。"""
    e = _engine(tmp_path, enabled=True)

    e.mark_retrain_started()
    assert e.state["offline_evolution"]["retrain_running"] is True
    assert e.state["offline_evolution"]["retrain_started_at"] is not None

    e.mark_retrain_finished(f1=0.62)
    cfg = e.state["offline_evolution"]
    assert cfg["retrain_running"] is False
    assert cfg["retrain_started_at"] is None
    assert cfg["last_retrain_at"] is not None


def test_mark_failed_clears_flag(tmp_path):
    """失败也必须清标记，否则同样会卡死。"""
    e = _engine(tmp_path, enabled=True)
    e.mark_retrain_started()
    e.mark_retrain_failed("数据缺失")
    assert e.state["offline_evolution"]["retrain_running"] is False
    assert e.state["offline_evolution"]["retrain_started_at"] is None


# ---------------------------------------------------------------- 触发条件

def test_disabled_never_triggers(tmp_path):
    e = _engine(tmp_path, enabled=False)
    should, reason = e.should_retrain()
    assert should is False
    assert "未启用" in reason


def test_triggers_when_recent_accuracy_low(tmp_path):
    """数据够 + 近期准确率低于阈值 → 应触发重训。"""
    e = _engine(tmp_path, enabled=True, retrain_running=False,
                min_batches=3, recent_window=3, accuracy_threshold=0.45,
                min_interval_days=7, last_retrain_at=_iso(24 * 30))
    e.state["performance"]["t5"] = {"accuracy": [0.30, 0.32, 0.28], "n": 3}
    e.save()

    should, reason = e.should_retrain()

    assert should is True
    assert "风格漂移" in reason


def test_no_trigger_when_accuracy_healthy(tmp_path):
    e = _engine(tmp_path, enabled=True, retrain_running=False,
                min_batches=3, recent_window=3, accuracy_threshold=0.45,
                last_retrain_at=_iso(24 * 30))
    e.state["performance"]["t5"] = {"accuracy": [0.60, 0.62, 0.58], "n": 3}
    e.save()

    should, reason = e.should_retrain()

    assert should is False
    assert "无需重训" in reason


def test_no_trigger_when_interval_too_short(tmp_path):
    """距上次重训不足最小间隔 → 不触发（防止频繁重训）。"""
    e = _engine(tmp_path, enabled=True, retrain_running=False,
                min_batches=3, recent_window=3, accuracy_threshold=0.45,
                min_interval_days=7, last_retrain_at=_iso(24 * 2))
    e.state["performance"]["t5"] = {"accuracy": [0.20, 0.20, 0.20], "n": 3}
    e.save()

    should, reason = e.should_retrain()

    assert should is False
    assert "最小间隔" in reason


def test_state_file_isolation(tmp_path):
    """回归保护：引擎只写自己的 state_file，不碰真实 data/。"""
    real = ROOT / "data" / "evolution_state.json"
    before = real.read_bytes() if real.exists() else None

    e = _engine(tmp_path, enabled=True)
    e.mark_retrain_started()
    e.mark_retrain_finished()

    after = real.read_bytes() if real.exists() else None
    assert after == before, "测试污染了真实 evolution_state.json"
    assert json.loads((tmp_path / "evo.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 分窗口记账

def test_record_verification_uses_hold_days(tmp_path):
    """回归：``tracker_v2`` 返回的是 ``hold_days``，不是 ``lookback_days``。

    若只读 ``lookback_days``，T+10 的结果会被错记到 t5 上，
    直接污染「近 3 批准确率 < 0.45 则重训」的判定。
    """
    e = _engine(tmp_path, enabled=True)

    key = e.record_verification({"hold_days": 10, "accuracy": 0.30})

    assert key == "t10"
    assert e.state["performance"]["t10"]["accuracy"] == [0.30]
    assert e.state["performance"]["t5"]["accuracy"] == []


def test_record_verification_falls_back_to_lookback_days(tmp_path):
    """旧 tracker 的字段仍要能读。"""
    e = _engine(tmp_path, enabled=True)
    key = e.record_verification({"lookback_days": 5, "accuracy": 0.62})
    assert key == "t5"
    assert e.state["performance"]["t5"]["accuracy"] == [0.62]


def test_record_verification_ignores_missing_accuracy(tmp_path):
    e = _engine(tmp_path, enabled=True)
    key = e.record_verification({"hold_days": 5, "accuracy": None})
    assert key == "t5"
    assert e.state["performance"]["t5"]["n"] == 0


def test_record_verification_keeps_last_20(tmp_path):
    e = _engine(tmp_path, enabled=True)
    for i in range(25):
        e.record_verification({"hold_days": 5, "accuracy": 0.5})
    assert len(e.state["performance"]["t5"]["accuracy"]) == 20


def test_evolve_from_verification_records_t10(tmp_path):
    """端到端：v2 格式的验证结果经 evolve_from_verification 也要进 t10。"""
    e = _engine(tmp_path, enabled=True)
    e.evolve_from_verification({"hold_days": 10, "accuracy": 0.55})
    assert e.state["performance"]["t10"]["accuracy"] == [0.55]
    assert e.state["performance"]["t5"]["accuracy"] == []


# ---------------------------------------------------------------- 阈值安全边界

def test_evolve_threshold_tightens_automatically(tmp_path):
    """准确率低 → 收紧阈值（保守方向）可以自动执行。"""
    e = _engine(tmp_path, enabled=True)
    e.state["threshold_adjustment"] = 0.0

    msg = e._evolve_threshold(0.20)

    assert e.state["threshold_adjustment"] == 3.0
    assert msg is not None and "收紧" in msg


def test_evolve_threshold_refuses_to_loosen_automatically(tmp_path):
    """**安全红线**：准确率高想放宽阈值 = 朝更激进方向 → 绝不静默生效。"""
    e = _engine(tmp_path, enabled=True)
    e.state["threshold_adjustment"] = 5.0

    msg = e._evolve_threshold(0.80)

    assert e.state["threshold_adjustment"] == 5.0, "禁止自动放宽选股阈值"
    assert msg is not None
    assert "建议" in msg and "人工批准" in msg
    # 建议也要留痕，方便事后审计
    logs = [x for x in e.state["evolution_log"] if x["action"] == "threshold_suggest"]
    assert logs


def test_evolve_threshold_neutral_branch_never_loosens(tmp_path):
    """中等准确率走 ``old_adj * 0.8``：老值为正时会变宽松，必须被拦住。"""
    e = _engine(tmp_path, enabled=True)
    e.state["threshold_adjustment"] = 6.0

    e._evolve_threshold(0.55)          # 6.0 * 0.8 = 4.8 < 6.0 → 激进

    assert e.state["threshold_adjustment"] == 6.0


def test_evolve_threshold_neutral_branch_can_pull_back_toward_zero(tmp_path):
    """老值为负（曾放宽过）时，0.8 倍是往回收，属保守方向，可自动。"""
    e = _engine(tmp_path, enabled=True)
    e.state["threshold_adjustment"] = -8.0

    e._evolve_threshold(0.55)          # -8.0 * 0.8 = -6.4 > -8.0 → 收紧

    assert e.state["threshold_adjustment"] == -6.4


def test_evolve_threshold_ignores_zero_accuracy(tmp_path):
    e = _engine(tmp_path, enabled=True)
    e.state["threshold_adjustment"] = 2.0
    assert e._evolve_threshold(0.0) is None
    assert e.state["threshold_adjustment"] == 2.0


# ---------------------------------------------------------------------------
# last_evolved_at 不被「过程/失败事件」污染（2026-10-19 事故）
# ---------------------------------------------------------------------------
# 线上真实误导：`last_evolved_at` 停在 2026-10-19T17:58，看起来「刚进化过」，
# 但对应的日志其实是「自动重训失败: 超时」——一次成功进化都没有。
# 根因：`_log_evolution()` 对**任何**日志条目都刷新 last_evolved_at，
# 包括重训启动、重训失败、清理僵尸标记这类过程/故障事件。
# 修复：这类事件传 evolved=False，只有真的改变了状态才算进化。

def test_retrain_start_does_not_count_as_evolved(tmp_path):
    """重训「启动」只是过程事件，不算进化。"""
    e = _engine(tmp_path, enabled=True)
    assert e.state.get("last_evolved_at") is None

    e.mark_retrain_started()
    assert e.state.get("last_evolved_at") is None, "启动重训不该刷新 last_evolved_at"


def test_retrain_failure_does_not_count_as_evolved(tmp_path):
    """重训失败（超时）绝不能刷新 last_evolved_at —— 这正是事故的掩盖点。"""
    e = _engine(tmp_path, enabled=True)
    e.mark_retrain_started()
    e.mark_retrain_failed("超时")

    assert e.state.get("last_evolved_at") is None, (
        "重训失败刷新了 last_evolved_at —— 会把「失败」伪装成「已进化」")
    assert e.state["offline_evolution"]["retrain_running"] is False


def test_retrain_finished_does_count_as_evolved(tmp_path):
    """重训真正完成（模型更新）才算一次进化。"""
    e = _engine(tmp_path, enabled=True)
    e.mark_retrain_started()
    e.mark_retrain_finished(0.63)

    assert e.state.get("last_evolved_at") is not None, "重训完成应当记为一次进化"
    assert e.state["offline_evolution"]["retrain_running"] is False
    assert e.state["offline_evolution"]["last_retrain_at"] is not None


def test_zombie_cleanup_does_not_count_as_evolved(tmp_path):
    """清理卡死标记只是清障，不是进化。"""
    e = _engine(tmp_path, enabled=True)
    e.mark_retrain_started()
    # 把开始时间推到超时之外，制造僵尸标记
    old = (datetime.now() - timedelta(hours=10)).isoformat()
    e.state["offline_evolution"]["retrain_started_at"] = old
    e.save()

    ok, _why = e.should_retrain()
    assert e.state.get("last_evolved_at") is None, "清障不该刷新 last_evolved_at"

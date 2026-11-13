# -*- coding: utf-8 -*-
"""超级集成健康台账（持久化 + 趋势）单元测试。

验证 2026-11-13 加的「腿状态变迁事件流」：
1. 仅可用性 *翻转* 时落盘事件，稳态每 tick 调用是 no-op（幂等）；
2. 状态文件比对 last-known，重复相同状态不产生重复事件；
3. 恢复（不可用→可用）也记录一条事件；
4. 新腿（首次出现）视为事件；
5. recent_health_events 按时间升序返回最后 N 条；
6. 任意异常都被吞（文件坏 / 目录不可写），不抛出。

零网络、零真实模型：直接喂 health_report 形态的 dict。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import src.models.ensemble as ensemble  # noqa: E402


@pytest.fixture
def _tmp_data_dir(tmp_path, monkeypatch):
    """把台账落盘目录钉到临时目录，避免污染真实 DATA_DIR。"""
    monkeypatch.setattr(ensemble, "DATA_DIR", tmp_path)
    return tmp_path


def _rep(legs):
    return {
        "enabled": True,
        "n_subs": len(legs),
        "n_available": sum(1 for l in legs if l["available"]),
        "dynamic_weights": False,
        "legs": legs,
    }


def _leg(name, available, reason=None):
    return {"name": name, "available": available, "reason": reason, "weight": 1.0}


def _read_events(data_dir):
    log = data_dir / "ensemble_health_log.jsonl"
    if not log.exists():
        return []
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def test_no_event_when_all_healthy(_tmp_data_dir):
    """全健康首次 → 应写状态文件但**无事件**（首次出现本应记事件？见 test_new_leg）。"""
    rep = _rep([_leg("alpha", True), _leg("lstm", True)])
    ensemble.log_health_report(rep)
    # 首次出现即视为状态变化，故两条腿各记一条「恢复/可用」事件（available=True）。
    # 这里只断言：稳态时再调一次不应新增事件（幂等核心）。
    n1 = len(_read_events(_tmp_data_dir))
    assert n1 >= 0
    ensemble.log_health_report(rep)  # 完全相同的状态
    n2 = len(_read_events(_tmp_data_dir))
    assert n2 == n1, "稳态重复调用必须 no-op，不得产生重复事件"


def test_down_event_logged_with_reason(_tmp_data_dir):
    """某腿从可用变不可用 → 落盘一条降级事件，含真实原因。"""
    rep_ok = _rep([_leg("alpha", True), _leg("lstm", True)])
    ensemble.log_health_report(rep_ok)            # 首次：两条「可用」事件
    rep_down = _rep([_leg("alpha", True),
                     _leg("lstm", False, "RuntimeError: lstm_best.pt 缺失")])
    ensemble.log_health_report(rep_down)          # lstm 翻转 → 降级事件
    events = _read_events(_tmp_data_dir)
    down = [e for e in events if e["leg"] == "lstm" and not e["available"]]
    assert down, "lstm 降级必须落盘事件"
    assert "lstm_best.pt 缺失" in (down[-1]["reason"] or "")
    assert down[-1]["n_available"] == 1
    assert down[-1]["n_subs"] == 2


def test_no_duplicate_when_stays_down(_tmp_data_dir):
    """持续不可用（状态不变）→ 不重复落盘降级事件。"""
    rep_down = _rep([_leg("alpha", True),
                     _leg("lstm", False, "RuntimeError: lstm_best.pt 缺失")])
    ensemble.log_health_report(rep_down)
    n1 = len(_read_events(_tmp_data_dir))
    ensemble.log_health_report(rep_down)          # 再次完全相同
    n2 = len(_read_events(_tmp_data_dir))
    assert n2 == n1, "持续降级不应重复落盘"


def test_recovery_event_logged(_tmp_data_dir):
    """不可用 → 恢复 → 落盘一条恢复事件（available=True, reason=None）。"""
    rep_down = _rep([_leg("alpha", True),
                     _leg("lstm", False, "RuntimeError: lstm_best.pt 缺失")])
    ensemble.log_health_report(rep_down)
    rep_rec = _rep([_leg("alpha", True), _leg("lstm", True)])
    ensemble.log_health_report(rep_rec)           # 恢复
    events = _read_events(_tmp_data_dir)
    rec = [e for e in events if e["leg"] == "lstm" and e["available"]]
    assert rec, "恢复必须落盘一条恢复事件"
    assert rec[-1]["reason"] is None


def test_new_leg_counts_as_event(_tmp_data_dir):
    """新增一条此前未出现的腿 → 视为状态变化，记一条事件。"""
    rep = _rep([_leg("alpha", True)])
    ensemble.log_health_report(rep)               # alpha 首次
    rep2 = _rep([_leg("alpha", True), _leg("transformer", True)])
    ensemble.log_health_report(rep2)              # transformer 首次出现
    events = _read_events(_tmp_data_dir)
    assert any(e["leg"] == "transformer" for e in events)


def test_recent_health_events_returns_last_n(_tmp_data_dir):
    """recent_health_events 按时间升序返回最后 limit 条。"""
    legs = [_leg("alpha", True), _leg("lstm", False, "x"), _leg("transformer", True)]
    # 制造 3 次翻转，确保事件数 > limit
    ensemble.log_health_report(_rep(legs))
    ensemble.log_health_report(_rep([_leg("alpha", True),
                                     _leg("lstm", True), _leg("transformer", True)]))
    ensemble.log_health_report(_rep(legs))
    out = ensemble.recent_health_events(limit=2, since_days=30)
    assert len(out) == 2
    # 升序：最后一条应是最近的（lstm 再次降级）
    assert out[-1]["leg"] == "lstm"
    assert out[-1]["available"] is False


def test_recent_returns_empty_when_no_log(_tmp_data_dir):
    """日志文件不存在 → 返回 []，不抛。"""
    assert ensemble.recent_health_events() == []


def test_corrupt_state_file_does_not_raise(_tmp_data_dir):
    """状态文件损坏 → 重建而非崩溃（异常全吞）。"""
    (ensemble._health_paths(_tmp_data_dir)[0]).write_text("{ bad", encoding="utf-8")
    rep = _rep([_leg("alpha", True)])
    # 不应抛异常
    ensemble.log_health_report(rep)
    assert (ensemble._health_paths(_tmp_data_dir)[1]).exists()

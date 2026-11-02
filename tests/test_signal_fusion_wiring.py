# -*- coding: utf-8 -*-
"""T1 跨日评分平滑接线测试（Predictor._apply_signal_fusion）。

这里测的是**接线**而不是纯函数（纯函数见 tests/test_signal_fusion.py）。
接线最容易错的三处，正好各一条用例：

1. 历史里存的是原始分还是平滑分 —— 存平滑分会把 alpha 逐日连乘，
   第三天等效 alpha 掉到 0.61，第五天 0.44，直接落进回测证伪的区间；
2. 首日 / 新出现的股票不能把分数打成 0（EMA 单点必须等于原值）；
3. 历史文件损坏必须退化为「不平滑」，不能把整批选股打挂。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import src.models.predictor as predictor_mod
from src.models.predictor import Predictor


def _preds(scores):
    return [SimpleNamespace(code=c, score=float(s)) for c, s in scores.items()]


def _mk(tmp_path, monkeypatch):
    monkeypatch.setattr(predictor_mod, "DATA_DIR", str(tmp_path), raising=False)
    p = Predictor()
    p._score_history = None
    return p


def test_first_day_scores_unchanged_but_history_written(tmp_path, monkeypatch):
    """首日只有当天一个点 ⇒ EMA 等于原值；同时必须落盘原始分供明天用。"""
    p = _mk(tmp_path, monkeypatch)
    preds = _preds({"600000": 80.0, "000001": 60.0})
    p._apply_signal_fusion(preds, as_of_date="2026-11-03")

    assert [x.score for x in preds] == pytest.approx([80.0, 60.0])

    f = tmp_path / "signal_score_history.json"
    assert f.exists()
    hist = json.loads(f.read_text(encoding="utf-8"))
    assert hist["2026-11-03"] == {"600000": 80.0, "000001": 60.0}


def test_second_day_uses_alpha_formula(tmp_path, monkeypatch):
    """第二天：score = alpha*raw + (1-alpha)*前值，alpha 取配置值 0.85。"""
    from config.settings import config as app_config

    alpha = float(app_config.ml.signal_fusion_alpha)
    p = _mk(tmp_path, monkeypatch)
    p._apply_signal_fusion(_preds({"600000": 80.0}), as_of_date="2026-11-02")

    preds = _preds({"600000": 90.0})
    p._apply_signal_fusion(preds, as_of_date="2026-11-03")
    assert preds[0].score == pytest.approx(alpha * 90.0 + (1 - alpha) * 80.0)


def test_history_keeps_raw_scores_not_smoothed(tmp_path, monkeypatch):
    """关键回归：历史必须存原始分。

    若存平滑分，连续三天的等效 alpha 会变成 alpha³ 量级（0.85→约 0.61），
    这正是回测里年化从 23.7% 掉到 13.9% 的那一档。
    """
    p = _mk(tmp_path, monkeypatch)
    raws = {"2026-11-02": 80.0, "2026-11-03": 90.0, "2026-11-04": 70.0}
    for d, raw in raws.items():
        p._apply_signal_fusion(_preds({"600000": raw}), as_of_date=d)

    hist = json.loads((tmp_path / "signal_score_history.json").read_text(encoding="utf-8"))
    assert hist["2026-11-04"]["600000"] == pytest.approx(70.0), "存了平滑值就会漂移"
    assert hist["2026-11-02"]["600000"] == pytest.approx(80.0)


def test_new_code_gets_its_own_score_not_zero(tmp_path, monkeypatch):
    """中途新出现的股票没有历史，EMA 单点必须等于它自己的分（不能补 0）。"""
    p = _mk(tmp_path, monkeypatch)
    p._apply_signal_fusion(_preds({"600000": 80.0}), as_of_date="2026-11-02")

    preds = _preds({"600000": 80.0, "301377": 55.0})
    p._apply_signal_fusion(preds, as_of_date="2026-11-03")
    assert preds[1].score == pytest.approx(55.0)


def test_corrupt_history_degrades_to_no_smoothing(tmp_path, monkeypatch):
    """历史文件损坏 ⇒ 记 WARNING 并按原始分继续，绝不中断选股。"""
    p = _mk(tmp_path, monkeypatch)
    (tmp_path / "signal_score_history.json").write_text("{not json", encoding="utf-8")
    preds = _preds({"600000": 77.0})
    p._apply_signal_fusion(preds, as_of_date="2026-11-03")
    assert preds[0].score == pytest.approx(77.0)


def test_disabled_flag_is_a_true_no_op(tmp_path, monkeypatch):
    """关掉开关必须「一个字节都不动」：不改分、也不写历史文件。"""
    from config.settings import config as app_config

    monkeypatch.setattr(app_config.ml, "signal_fusion_enabled", False)
    p = _mk(tmp_path, monkeypatch)
    preds = _preds({"600000": 80.0})
    p._apply_signal_fusion(preds, as_of_date="2026-11-03")
    assert preds[0].score == pytest.approx(80.0)
    assert not (tmp_path / "signal_score_history.json").exists()


def test_history_pruned_to_configured_days(tmp_path, monkeypatch):
    """滚动窗口按配置天数截断，文件不会无限膨胀。"""
    from config.settings import config as app_config

    p = _mk(tmp_path, monkeypatch)
    keep = int(app_config.ml.signal_fusion_days)
    for i in range(keep + 6):
        p._apply_signal_fusion(_preds({"600000": 50.0 + i}),
                               as_of_date="2026-01-%02d" % (i + 1))
    hist = json.loads((tmp_path / "signal_score_history.json").read_text(encoding="utf-8"))
    assert len(hist) == keep

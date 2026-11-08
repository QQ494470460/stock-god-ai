# -*- coding: utf-8 -*-
"""SignalFusion 跨日平滑的「原始分落盘」契约测试。

memory §8 铁律：signal_score_history.json 必须存**平滑前的原始分**。
若误存平滑后的值，明天会在平滑值上再平滑（alpha 逐日连乘、越滑越钝），
最终退化成被回测证伪的「过度平滑」。本测试直接验证 predictor._apply_signal_fusion
落盘的今日记录是原始分，而非 EMA 平滑后的分。
"""

import json

import pytest

import src.models.predictor as pmod
from src.models.predictor import Predictor, StockPrediction
from config.settings import config as app_config


def _pred(code, score):
    return StockPrediction(code=code, name=code, prob_up=0.0, prob_neutral=0.0,
                           prob_down=0.0, score=float(score), score_raw=float(score))


def test_signal_fusion_persists_raw_scores(tmp_path, monkeypatch):
    # 把评分历史落盘点指向 tmp，避免污染生产 DATA_DIR
    monkeypatch.setattr(pmod, "DATA_DIR", tmp_path / "data")

    old_enabled = app_config.ml.signal_fusion_enabled
    try:
        app_config.ml.signal_fusion_enabled = True

        pred = Predictor(model_dir=tmp_path / "models")

        # 预置前一天的「原始分」历史：A 偏高、B 偏低
        hist_file = tmp_path / "data" / "signal_score_history.json"
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        hist_file.write_text(
            json.dumps({"2026-06-01": {"A": 100.0, "B": 0.0}}, ensure_ascii=False),
            encoding="utf-8",
        )

        # 今日原始分（与历史差异极大，EMA(alpha=0.85) 必会改变 score）
        preds = [_pred("A", 10.0), _pred("B", 90.0)]
        pred._apply_signal_fusion(preds, as_of_date="2026-06-02")

        # 平滑后 p.score 已被 EMA 拉向历史：A 上抬、B 下压
        by_code = {p.code: p.score for p in preds}
        assert by_code["A"] != 10.0, "EMA 未生效？"
        assert by_code["B"] != 90.0, "EMA 未生效？"

        # 落盘的今日记录必须是**原始分**，绝不是被平滑的值
        saved = json.loads(hist_file.read_text(encoding="utf-8"))
        assert saved["2026-06-02"] == {"A": 10.0, "B": 90.0}, saved
    finally:
        app_config.ml.signal_fusion_enabled = old_enabled


def test_signal_fusion_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(pmod, "DATA_DIR", tmp_path / "data")
    old_enabled = app_config.ml.signal_fusion_enabled
    try:
        app_config.ml.signal_fusion_enabled = False
        pred = Predictor(model_dir=tmp_path / "models")
        preds = [_pred("A", 42.0)]
        pred._apply_signal_fusion(preds, as_of_date="2026-06-02")
        # 关闭时：不做任何平滑，score 原样保留
        assert preds[0].score == 42.0
        # 关闭时提前 return，不写历史文件（无平滑需求，无需为未来平滑留基准）
        assert not (tmp_path / "data" / "signal_score_history.json").exists()
    finally:
        app_config.ml.signal_fusion_enabled = old_enabled

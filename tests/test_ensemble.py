# -*- coding: utf-8 -*-
"""超级集成单元测试（离线，用 Fake 子预测器，零网络/零真实模型）"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models.ensemble import EnsemblePredictor, SubPredictor, _proba_to_score  # noqa: E402
from config.settings import config as app_config  # noqa: E402


@pytest.fixture(autouse=True)
def _pin_no_shrinkage():
    """把「权重收缩」钉为 0，让本文件的权重测试只验证纯加权逻辑。

    收缩是独立旋钮（生产默认 0.5），若不钉住会让下面的期望值随全局配置漂移。
    收缩本身的效果由 test_ensemble_shrinkage_pulls_weights_to_uniform 单独验证。
    """
    old = getattr(app_config.ml, "ensemble_shrinkage", 0.0)
    app_config.ml.ensemble_shrinkage = 0.0
    try:
        yield
    finally:
        app_config.ml.ensemble_shrinkage = old


class FakePred:
    def __init__(self, score):
        self.score = score
        self.code = ""
        self.name = ""
        self.xgb_signal = 1
        self.agreement = 1.0


class FakePrimary:
    def __init__(self, score=60.0):
        self.score = score

    def predict_single(self, df, code="x", name="x"):
        p = FakePred(self.score)
        p.code = code
        p.name = name
        return p


class FakeSub(SubPredictor):
    name = "fake"

    def __init__(self, score):
        self._val = score

    def _ensure(self):
        pass

    def _score(self, df, code):
        return self._val


class UnavailableSub(SubPredictor):
    name = "bad"

    def _ensure(self):
        raise RuntimeError("model file missing")


class NoneSub(SubPredictor):
    """可用，但对某些标的出分为 None（特征不足 / 该腿模型没覆盖这只票）。"""

    name = "none"

    def _ensure(self):
        pass

    def _score(self, df, code):
        return None


def _df():
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]})


def test_blend_weighted_average():
    ep = EnsemblePredictor(FakePrimary(60.0))
    ep.add_sub(FakeSub(90.0), weight=1.0)
    out = ep.predict_batch({"000001": _df()})
    p = out["000001"]
    assert abs(p.score - 75.0) < 1e-6
    assert abs(p.ensemble_detail["spread"] - 30.0) < 1e-6
    assert p.agreement >= 0.8  # 不破坏交易执行器硬闸门


def test_weight_respected():
    ep = EnsemblePredictor(FakePrimary(0.0))
    ep.add_sub(FakeSub(100.0), weight=3.0)  # 权重 3:1 → 75
    out = ep.predict_batch({"a": _df()})
    assert abs(out["a"].score - 75.0) < 1e-6


def test_graceful_degradation_no_subs():
    ep = EnsemblePredictor(FakePrimary(42.0))
    out = ep.predict_batch({"a": _df()})
    assert out["a"].score == 42.0
    assert not hasattr(out["a"], "ensemble_detail") or out["a"].ensemble_detail is None


def test_unavailable_sub_skipped():
    ep = EnsemblePredictor(FakePrimary(60.0))
    ep.add_sub(UnavailableSub(), weight=1.0)
    out = ep.predict_batch({"a": _df()})
    # 唯一子模型不可用 → 回退主模型分
    assert out["a"].score == 60.0


def test_ensemble_shrinkage_pulls_weights_to_uniform():
    """权重收缩：把子模型权重向等权拉，降低单腿过置信权重的方差。"""
    def _blend(shrink):
        app_config.ml.ensemble_shrinkage = shrink
        ep = EnsemblePredictor(FakePrimary(0.0))
        ep.add_sub(FakeSub(100.0), weight=1.0)
        ep.add_sub(FakeSub(60.0), weight=3.0)   # 权重 3:1，偏置信第二腿
        return ep.predict_batch({"a": _df()})["a"].score

    # shrink=0：原始权重 [1,3] → (0*1 + 100*1 + 60*3)/5 = 56.0
    assert abs(_blend(0.0) - 56.0) < 1e-6
    # shrink=1：子腿全部等权（总权重 4 → 各 2）→ (0*1 + 100*2 + 60*2)/5 = 64.0
    assert abs(_blend(1.0) - 64.0) < 1e-6
    # 单调性：收缩越强，越偏向「两腿平均」（100 与 60 的平均=80，即权重对等时更靠近高分腿）
    assert _blend(1.0) > _blend(0.0)


def test_detail_tables_stay_aligned_when_a_leg_returns_none():
    """某条腿对某只股票出分 None 时，ensemble_detail 三张表必须同长度。

    回归背景（2026-10-18）：加权重收缩后 eff_ws 是「全部可用腿」的权重表，
    而 scores 只收「真正出分」的腿。一旦某腿返回 None，weights 就比 scores 长，
    看板按腿名对不上号，诊断信息失真（而融合分本身碰巧还算对，问题很隐蔽）。
    """
    app_config.ml.ensemble_shrinkage = 0.5
    ep = EnsemblePredictor(FakePrimary(50.0))
    ep.add_sub(NoneSub(), weight=1.0)
    ep.add_sub(FakeSub(70.0), weight=1.0)
    p = ep.predict_batch({"a": _df()})["a"]
    d = p.ensemble_detail
    assert set(d["scores"]) == set(d["weights"]) == {"xgboost", "fake"}
    assert len(d["scores"]) == len(d["weights"])
    # 只有两条腿参与（主 50 / fake 70，等权）→ 60
    assert abs(p.score - 60.0) < 1e-6


def test_torch_preloaded_before_xtquant():
    """回归：xtquant 必须**惰性**加载，且加载前先把 torch 预载进来。

    1) 惰性不变式：模拟/纸交易默认路径不得 eager import 原生 xtquant（冷启动
       DLL 初始化异常会以原生退出码无声崩溃）。
    2) 顺序不变式：真正走实盘调用 ensure_xtquant() 时，必须先 preload torch 再
       import xtquant，否则 torch 的 c10.dll 初始化失败（WinError 1114），LSTM 腿
       静默降级、集成从三条腿变两条腿而看板上看不出来。
    import 顺序在本进程里无法回退，所以必须用子进程验证。
    """
    import subprocess
    code = (
        "import sys; sys.path.insert(0, r'{root}');"
        " import src.trading.yuekai as yk;"
        " assert 'xtquant' not in sys.modules, 'xtquant must stay lazy until ensure_xtquant()';"
        " ok = yk.ensure_xtquant();"
        " assert ok is True, 'ensure_xtquant should succeed when xtquant is installed';"
        " mods = list(sys.modules);"
        " assert 'torch' in sys.modules, 'torch must be preloaded before xtquant';"
        " assert mods.index('torch') < mods.index('xtquant'), 'torch must load before xtquant';"
        " from src.models.ensemble import LSTMSubPredictor;"
        " print('LSTM', LSTMSubPredictor().available)"
    ).format(root=ROOT)
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=300)
    assert "LSTM True" in r.stdout, r.stdout + r.stderr


def test_alpha_lag_defaults_to_zero():
    """诊断开关默认必须关闭：不能因为环境变量就悄悄改掉生产的 alpha 口径。"""
    from src.models import ensemble as ens
    assert ens._ALPHA_LAG == 0


def test_alpha_lag_shifts_disclosure_by_one_day(monkeypatch, tmp_path):
    """ALPHA_LAG_DAYS=1 时，当日披露的 alpha 值必须推迟一天才可见。

    只验证位移/对齐机制本身（不加载真实模型）。这个开关用来量化「披露滞后
    虚高了多少 IC」，机制若错位，结论就全错。
    """
    import src.models.ensemble as ens
    monkeypatch.setattr(ens, "DATA_DIR", tmp_path)
    idx = pd.to_datetime(["2026-10-13", "2026-10-14", "2026-10-15"])
    pd.DataFrame({"000001": [1.0, 2.0, 3.0]}, index=idx).to_parquet(
        tmp_path / "alpha_margin.parquet")

    sub = object.__new__(ens.AlphaSubPredictor)
    sub._alpha_cols = ["alpha_margin"]

    def _last(lag):
        monkeypatch.setattr(ens, "_ALPHA_LAG", lag)
        a = sub._alpha_for_code("000001", idx)
        if lag:
            a = a.shift(lag)
        return float(a.iloc[-1]["alpha_margin"])

    assert _last(0) == 3.0   # 现口径：当天披露当天可见（含前视）
    assert _last(1) == 2.0   # 滞后一天：只能用昨天已披露的


def test_proba_to_score_5class():
    s = _proba_to_score(np.array([0.1, 0.1, 0.2, 0.3, 0.3]), [0, 1, 2, 3, 4])
    # prob_up=0.6, prob_down=0.2 -> (0.6*0.7 - 0.2*0.3 + 0.3)*100 = 66.0
    assert abs(s - 66.0) < 1e-6


def test_proba_to_score_3class():
    s = _proba_to_score(np.array([0.2, 0.3, 0.5]), [-1, 0, 1])
    # prob_up=0.5, prob_down=0.2 -> (0.35 - 0.06 + 0.3)*100 = 59.0
    assert abs(s - 59.0) < 1e-6

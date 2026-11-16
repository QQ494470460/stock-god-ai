"""超级集成「共享特征矩阵」的契约测试。

背景（2026-11-16 实测）：alpha / lstm / transformer 三条腿原本各自调
``feature_engine.build_features(df)``，同一只股票的同一份 df 被重复计算 2~3 遍，
单次 86ms、425 只 ⇒ 白烧 37~73 秒。改成批量路径算一次、各腿共用。

**必须守住的底线**：共用之后评分必须与"各腿各自算"逐位相同 ——
否则这不是优化，是偷偷换了模型输入。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import ensemble as ens_mod
from src.models.ensemble import EnsemblePredictor, SubPredictor


def _kline(n=120, seed=3):
    rnd = np.random.default_rng(seed)
    close = 10 + np.cumsum(rnd.normal(0, 0.1, n))
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": close * 0.995, "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": rnd.integers(1e5, 1e6, n).astype(float),
        "amount": rnd.integers(1e6, 1e7, n).astype(float),
    })


class _FakePrimary:
    """主模型替身：不加载任何模型文件。"""

    def predict_single(self, df, code=None, name=None):
        class _P:
            pass
        p = _P()
        p.code, p.name = code, name or code
        p.score = 50.0
        p.score_raw = 50.0
        p.agreement = 1.0
        p.ensemble_detail = None
        p.model_mode = "single"
        return p


class _FeatLeg(SubPredictor):
    """吃特征矩阵的假腿：只记调用次数，评分为末行某列的函数。"""

    name = "fake_feat"
    needs_features = True

    def __init__(self, tag: str, calls: list):
        self._tag = tag
        self._calls = calls

    def _ensure(self):
        self._model = True

    def _score(self, df, code, feats=None):
        self._calls.append(self._tag)
        f = self._resolve_feats(df, feats)
        return float(50.0 + f.iloc[-1].get("rsi_14", 0.0) * 0.0)


@pytest.fixture()
def build_counter(monkeypatch):
    """包一层 build_features，统计真实调用次数（并保留原行为）。"""
    real = ens_mod.feature_engine.build_features
    hits = {"n": 0}

    def _wrapped(df, *a, **kw):
        hits["n"] += 1
        return real(df, *a, **kw)

    monkeypatch.setattr(ens_mod.feature_engine, "build_features", _wrapped)
    return hits


def test_features_built_once_per_stock_not_once_per_leg(build_counter):
    calls: list = []
    ep = EnsemblePredictor(_FakePrimary())
    ep.add_sub(_FeatLeg("a", calls), 1.0)
    ep.add_sub(_FeatLeg("b", calls), 1.0)
    ep.add_sub(_FeatLeg("c", calls), 1.0)

    data = {c: _kline(seed=i) for i, c in enumerate(["000001", "000002", "600000"])}
    ep.predict_batch(data)

    # 3 只股票 × 3 条腿 = 9 次腿调用，但特征只需建 3 次
    assert len(calls) == 9
    assert build_counter["n"] == 3, f"build_features 被调了 {build_counter['n']} 次，应为 3"


def test_shared_features_scores_identical_to_per_leg(build_counter):
    """共用特征 vs 各腿各自现算 → 评分必须完全一致。"""
    df = _kline(seed=11)

    class _RealLeg(SubPredictor):
        name = "real_leg"
        needs_features = True

        def _ensure(self):
            self._model = True

        def _score(self, df, code, feats=None):
            f = self._resolve_feats(df, feats)
            last = f.iloc[-1]
            return float(round(float(last.sum()) % 97 + 1, 6))

    shared = _RealLeg().score_with(df, "000001", feats=SubPredictor._resolve_feats(df, None))
    alone = _RealLeg().score_with(df, "000001")
    assert shared == pytest.approx(alone, abs=1e-9)


def test_leg_without_feats_support_still_works():
    """老签名（只吃 df, code）的鸭子类型腿不能被新参数搞挂。"""

    class _OldLeg:
        name = "old_leg"
        available = True

        def score_for(self, df, code):
            return 66.0

    ep = EnsemblePredictor(_FakePrimary())
    ep.add_sub(_OldLeg(), 1.0)
    pred = _FakePrimary().predict_single(_kline(), "000001")
    assert ep.blend_one(pred, _kline(), "000001") is True
    # 老签名腿确实参与了融合（分值原样进入 detail），没有被新参数搞缺席
    assert pred.ensemble_detail["scores"]["old_leg"] == pytest.approx(66.0)


def test_unavailable_legs_still_build_only_once_for_primary(build_counter):
    """全部腿都不可用时，特征只为**主模型**建一次，不会因腿而重复。"""

    class _Down(SubPredictor):
        name = "down"

        def _ensure(self):
            raise RuntimeError("模型缺失")

    ep = EnsemblePredictor(_FakePrimary())
    ep.add_sub(_Down(), 1.0)
    ep.predict_batch({"000001": _kline()})
    assert build_counter["n"] == 1


def test_llm_only_legs_do_not_add_extra_feature_build(build_counter):
    """只装配了不读特征的腿（LLM）时，只为「主模型」建一次，腿不额外加算。"""

    class _LLMLike(SubPredictor):
        name = "llm_like"
        needs_features = False

        def _ensure(self):
            self._model = True

        def _score(self, df, code, feats=None):
            return 55.0

    ep = EnsemblePredictor(_FakePrimary())
    ep.add_sub(_LLMLike(), 1.0)
    ep.predict_batch({"000001": _kline()})
    assert build_counter["n"] == 1


# ------------------------------------------------- alpha 面板缓存（效率核心）

def test_alpha_panel_read_once_per_process(tmp_path, monkeypatch):
    """3 个 alpha 面板 × 425 只股票不能让 pd.read_parquet 跑 1275 次。

    实测基线：不缓存时 366ms/只、单次选股 205 秒（面板总共才 2.9MB）。
    """
    from src.models import ensemble as em

    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    for name in ("northbound", "margin", "lhb"):
        pd.DataFrame({"000001": range(50), "000002": range(50)}, index=dates).to_parquet(
            tmp_path / f"alpha_{name}.parquet")

    monkeypatch.setattr(em, "DATA_DIR", tmp_path)
    reads = {"n": 0}
    real = pd.read_parquet

    def _counting(path, *a, **kw):
        reads["n"] += 1
        return real(path, *a, **kw)

    monkeypatch.setattr(em.pd, "read_parquet", _counting)

    sub = em.AlphaSubPredictor()
    sub._alpha_cols = ["alpha_northbound", "alpha_margin", "alpha_lhb"]
    for code in ("000001", "000002"):
        for _ in range(5):
            sub._alpha_for_code(code, dates[-1:])
    assert reads["n"] == 3, f"面板被读了 {reads['n']} 次，应只读 3 次（每面板 1 次）"


def test_alpha_panel_cache_invalidated_on_rewrite(tmp_path, monkeypatch):
    """面板被重新回填（mtime 变）后必须重新读，不能喂旧数据。"""
    from src.models import ensemble as em

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    p = tmp_path / "alpha_margin.parquet"
    pd.DataFrame({"000001": [1.0] * 10}, index=dates).to_parquet(p)
    monkeypatch.setattr(em, "DATA_DIR", tmp_path)

    sub = em.AlphaSubPredictor()
    sub._alpha_cols = ["alpha_margin"]
    first = sub._alpha_for_code("000001", dates[-1:])

    import os, time
    pd.DataFrame({"000001": [9.0] * 10}, index=dates).to_parquet(p)
    os.utime(p, (time.time() + 5, time.time() + 5))   # 确保 mtime 变化
    second = sub._alpha_for_code("000001", dates[-1:])
    assert first is not None and second is not None
    assert sub._panel_cache[str(p)][1]["000001"].iloc[-1] == 9.0

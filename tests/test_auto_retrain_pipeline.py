# -*- coding: utf-8 -*-
"""``src/evolution/auto_retrain_pipeline.py`` 的单元测试。

全部用合成指标 + 假依赖，**不训练、不联网、不写盘**，秒级跑完。

重点验证的不是「能跑」，而是三条不能被破坏的性质：

1. 没漂移时**绝不**调用 ``train_fn``（否则每次进化回合都会白训一次）；
2. 输入的指标字典**不会被就地改写**（调用方可能还要拿它做别的判断）；
3. 任何「信息不足」都落到 ``pending`` 而不是偷偷放行或崩掉。
"""

from __future__ import annotations

import copy

import pytest

from src.evolution.auto_retrain_pipeline import (
    ACTION_NO_DRIFT, ACTION_PENDING, ACTION_PROMOTE, ACTION_REJECT,
    AutoRetrainPipeline, RetrainDecision, format_report,
)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _metrics(rank_ic: float, sharpe: float = 0.0) -> dict:
    return {"rank_ic": rank_ic, "sharpe": sharpe, "n_days": 240}


def _boom(*_a, **_kw):
    raise AssertionError("这个依赖不该被调用")


# ---------------------------------------------------------------------------
# ① 无漂移
# ---------------------------------------------------------------------------

def test_no_drift_when_current_equals_baseline():
    """现任 ≈ 基准 → 不重训，且**一次都不**调用 train_fn。"""
    pipe = AutoRetrainPipeline(train_fn=_boom, eval_fn=_boom)
    dec = pipe.run(_metrics(0.070), _metrics(0.070))

    assert isinstance(dec, RetrainDecision)
    assert dec.action == ACTION_NO_DRIFT
    assert dec.candidate_metrics == {}
    assert dec.delta == {}
    assert dec.current_metrics["rank_ic"] == pytest.approx(0.070)
    assert "不重训" in dec.reason


def test_no_drift_is_default_off():
    """只给配置、不给任何依赖也能构造（测试/--check 场景）。"""
    pipe = AutoRetrainPipeline()
    assert pipe.detect_drift(_metrics(0.070), _metrics(0.070)) is False
    assert pipe.run(_metrics(0.070), _metrics(0.070)).action == ACTION_NO_DRIFT


def test_small_drop_below_threshold_is_not_drift():
    """掉 0.005 < drift_drop(0.010) → 不算漂移（噪声不触发重训）。"""
    pipe = AutoRetrainPipeline(train_fn=_boom, eval_fn=_boom)
    assert pipe.detect_drift(_metrics(0.075), _metrics(0.080)) is False


# ---------------------------------------------------------------------------
# ② 漂移 → 晋级
# ---------------------------------------------------------------------------

def test_promote_when_drift_and_candidate_materially_better():
    """漂移已发生，候选 RankIC 大幅提升且过绝对下限 → promote，delta 正确。"""
    calls = {"train": 0, "eval": 0}

    def train_fn():
        calls["train"] += 1
        return {"path": "models/cand.pkl"}          # 候选的形态交由 eval_fn 解释

    def eval_fn(cand):
        calls["eval"] += 1
        assert cand == {"path": "models/cand.pkl"}
        return _metrics(0.082, sharpe=1.15)

    pipe = AutoRetrainPipeline(
        train_fn=train_fn, eval_fn=eval_fn,
        min_rankic_delta=0.005, min_abs_rankic=0.05, drift_drop=0.010)

    dec = pipe.run(_metrics(0.068, sharpe=1.00), _metrics(0.080, sharpe=1.00))

    assert dec.action == ACTION_PROMOTE
    assert dec.promoted is True
    assert calls == {"train": 1, "eval": 1}
    assert dec.delta["rank_ic"] == pytest.approx(0.082 - 0.068, abs=1e-6)
    assert dec.delta["sharpe"] == pytest.approx(0.15, abs=1e-6)
    assert dec.candidate_metrics["rank_ic"] == pytest.approx(0.082)
    assert "建议晋升" in dec.reason


def test_run_level_train_eval_override_constructor():
    """run(train_fn=..., eval_fn=...) 优先于构造参数。"""
    pipe = AutoRetrainPipeline(train_fn=_boom, eval_fn=_boom)
    dec = pipe.run(
        _metrics(0.060), _metrics(0.080),
        train_fn=lambda: "cand",
        eval_fn=lambda c: _metrics(0.085),
    )
    assert dec.action == ACTION_PROMOTE


# ---------------------------------------------------------------------------
# ③ 漂移 → 拒绝
# ---------------------------------------------------------------------------

def test_reject_when_candidate_worse():
    """候选比现任还差 → reject，delta 为负。"""
    pipe = AutoRetrainPipeline(train_fn=lambda: "cand",
                               eval_fn=lambda c: _metrics(0.055))
    dec = pipe.run(_metrics(0.060), _metrics(0.080))
    assert dec.action == ACTION_REJECT
    assert dec.delta["rank_ic"] == pytest.approx(-0.005, abs=1e-6)
    assert "拒绝" in dec.reason


def test_reject_when_delta_below_threshold():
    """提升 0.003 < min_rankic_delta(0.005) → reject（哪怕方向是对的）。"""
    pipe = AutoRetrainPipeline(min_rankic_delta=0.005, min_abs_rankic=0.05,
                               train_fn=lambda: "cand",
                               eval_fn=lambda c: _metrics(0.063))
    dec = pipe.run(_metrics(0.060), _metrics(0.080))
    assert dec.action == ACTION_REJECT
    assert "提升不足" in dec.reason


def test_reject_when_candidate_below_absolute_floor():
    """现任很烂，候选略好但仍低于绝对下限 → reject（垃圾里挑冠军没有意义）。"""
    pipe = AutoRetrainPipeline(min_abs_rankic=0.05, drift_drop=0.010,
                               train_fn=lambda: "cand",
                               eval_fn=lambda c: _metrics(0.035))
    dec = pipe.run(_metrics(0.020), _metrics(0.060))
    assert dec.action == ACTION_REJECT
    assert "绝对下限" in dec.reason


# ---------------------------------------------------------------------------
# ④ 漂移 → 待定
# ---------------------------------------------------------------------------

def test_pending_when_drift_but_no_train_fn():
    """检测到漂移但没有 train_fn → pending，且理由说清楚为什么。"""
    pipe = AutoRetrainPipeline(eval_fn=_boom)
    dec = pipe.run(_metrics(0.040), _metrics(0.080))
    assert dec.action == ACTION_PENDING
    assert "train_fn" in dec.reason
    assert dec.candidate_metrics == {}


def test_pending_when_train_returns_none():
    pipe = AutoRetrainPipeline(train_fn=lambda: None, eval_fn=_boom)
    assert pipe.run(_metrics(0.040), _metrics(0.080)).action == ACTION_PENDING


def test_pending_when_train_raises():
    """训练异常不能冒泡 —— 进化回路的失败模式必须是「不动」。"""
    def bad_train():
        raise RuntimeError("磁盘满了")

    pipe = AutoRetrainPipeline(train_fn=bad_train, eval_fn=_boom)
    dec = pipe.run(_metrics(0.040), _metrics(0.080))
    assert dec.action == ACTION_PENDING
    assert "磁盘满了" in dec.reason


def test_pending_when_eval_missing_or_empty():
    pipe_no_eval = AutoRetrainPipeline(train_fn=lambda: "cand")
    assert pipe_no_eval.run(_metrics(0.040), _metrics(0.080)).action == ACTION_PENDING

    pipe_bad_eval = AutoRetrainPipeline(train_fn=lambda: "cand",
                                        eval_fn=lambda c: {})
    assert pipe_bad_eval.run(_metrics(0.040), _metrics(0.080)).action == ACTION_PENDING


def test_pending_when_candidate_rank_ic_uncomputable():
    """候选指标里没有 rank_ic → pending（不可证明，不是「变差」）。"""
    pipe = AutoRetrainPipeline(train_fn=lambda: "cand",
                               eval_fn=lambda c: {"sharpe": 1.2})
    dec = pipe.run(_metrics(0.040), _metrics(0.080))
    assert dec.action == ACTION_PENDING


# ---------------------------------------------------------------------------
# ⑤ 不改输入（防御性拷贝）
# ---------------------------------------------------------------------------

def test_inputs_are_not_mutated():
    """跑完后调用方手里的字典必须逐字节不变。"""
    cur = {"walk_forward": {"rank_ic": 0.060, "sharpe": 1.0, "n_days": 240}}
    base = {"walk_forward": {"rank_ic": 0.080, "sharpe": 1.0}}
    cur_snapshot = copy.deepcopy(cur)
    base_snapshot = copy.deepcopy(base)

    cand_raw = {"rank_ic": 0.090}

    pipe = AutoRetrainPipeline(train_fn=lambda: "cand",
                               eval_fn=lambda c: cand_raw)
    dec = pipe.run(cur, base, eval_fn=lambda c: cand_raw)

    assert cur == cur_snapshot
    assert base == base_snapshot
    assert cand_raw == {"rank_ic": 0.090}          # eval 返回值也没被改
    assert dec.action == ACTION_PROMOTE

    # 决策里的指标是副本：改它不该回写到输入
    dec.current_metrics["rank_ic"] = -999.0
    assert cur == cur_snapshot


# ---------------------------------------------------------------------------
# ⑥ evaluate_only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cand,cur,expected", [
    ({"rank_ic": 0.090}, {"rank_ic": 0.070}, ACTION_PROMOTE),
    ({"rank_ic": 0.072}, {"rank_ic": 0.070}, ACTION_REJECT),   # 提升不足
    ({"rank_ic": 0.060}, {"rank_ic": 0.070}, ACTION_REJECT),   # 倒退
    ({"rank_ic": 0.040}, {"rank_ic": 0.010}, ACTION_REJECT),   # 低于绝对下限
    ({"sharpe": 1.0}, {"rank_ic": 0.070}, ACTION_PENDING),     # 候选无指标
    ({"rank_ic": 0.090}, {}, ACTION_PENDING),                  # 无基准
])
def test_evaluate_only(cand, cur, expected):
    pipe = AutoRetrainPipeline()
    assert pipe.evaluate_only(cand, cur) == expected


# ---------------------------------------------------------------------------
# ⑦ 其它
# ---------------------------------------------------------------------------

def test_missing_metric_is_not_drift():
    """现任指标缺失 ≠ 漂移：一条坏数据不该触发重训。"""
    pipe = AutoRetrainPipeline(train_fn=_boom, eval_fn=_boom)
    assert pipe.detect_drift({}, _metrics(0.080)) is False
    assert pipe.detect_drift({"rank_ic": None}, _metrics(0.080)) is False
    assert pipe.detect_drift({"rank_ic": float("nan")}, _metrics(0.080)) is False


def test_nested_metric_shapes_are_understood():
    """兼容 production_metrics.json / walk_forward_baseline.json 的嵌套形状。"""
    pipe = AutoRetrainPipeline(train_fn=_boom, eval_fn=_boom)
    cur = {"walk_forward": {"summary": {"rank_ic": 0.070}}}
    base = {"walk_forward": {"summary": {"rank_ic": 0.071}}}
    assert pipe.detect_drift(cur, base) is False
    assert pipe.detect_drift({"walk_forward": {"rank_ic": 0.050}}, base) is True


def test_format_report_and_to_dict():
    dec = RetrainDecision(
        action=ACTION_PROMOTE, reason="候选更好",
        current_metrics={"rank_ic": 0.060}, candidate_metrics={"rank_ic": 0.080},
        delta={"rank_ic": 0.020})
    txt = format_report(dec)
    assert "promote" in txt and "0.0200" in txt and "人工批准" in txt
    d = dec.to_dict()
    assert d["action"] == ACTION_PROMOTE and d["delta"]["rank_ic"] == 0.020


def test_unknown_action_rejected():
    with pytest.raises(ValueError):
        RetrainDecision(action="auto_apply", reason="不该有这种动作")

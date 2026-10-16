# -*- coding: utf-8 -*-
"""自我进化 · 生成器 C（模型重训上线）测试。

重点证明三件事：

1. **生成器从不训练模型** —— 它只读注册表、只提候选，训练由运维显式触发；
2. **模型候选走「自带证据」通道** —— 因为换模型会改变评分面板本身，
   没法用同一块面板做 A/B 回测；但证据必须过
   :func:`src.evolution.gate.verify_supplied_evaluation` 的形态与来源校验；
3. **``ml.active_version`` 能冻结裁决** —— 钉住后，后续丢进来的
   「分数更好但没被评估过」的产物不会静默顶掉它。
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evolution import gate, overrides as ov_mod, space  # noqa: E402
from src.evolution.feedback import PerformanceSnapshot  # noqa: E402
from src.evolution.generators import REGISTRY  # noqa: E402
from src.evolution.generators import retrain as rt  # noqa: E402
from src.evolution.types import Candidate, Change, Evaluation  # noqa: E402


# ------------------------------------------------------------------ 夹具

def _info(tmp_path, key, ric, mtime, *, loadable=True, ir=0.10, n=120):
    """造一个带真实文件（可控 mtime）的假 ModelInfo。"""
    p = tmp_path / f"{key}.pkl"
    p.write_bytes(b"x")
    os.utime(p, (mtime, mtime))
    return SimpleNamespace(
        key=key, display_name=key, model_path=p, prefix="",
        rank_ic=ric, ic_ir=ir, win_rate=0.55, top_excess=0.01,
        n_samples=n, loadable=loadable, note="")


def _snap(**kw) -> PerformanceSnapshot:
    base = dict(
        generated_at="2026-10-14T13:00:00",
        n_batches=25, n_verified=17, pending_verify=5,
        accuracy_by_horizon={"5": [0.60, 0.62, 0.58, 0.55, 0.20, 0.15, 0.15]},
        n_settlements=20,
        equity_series=[{"date": f"2026-09-{i:02d}", "equity": 100000.0,
                        "cash": 10000.0, "n_positions": 10}
                       for i in range(1, 21)],
        trade_stats={"fill": 40},
        guard={"halted": False},
        metrics={"insufficient": False, "turnover": 1.0},
    )
    base.update(kw)
    return PerformanceSnapshot(**base)


def _snap_healthy(**kw) -> PerformanceSnapshot:
    return _snap(accuracy_by_horizon={"5": [0.60, 0.62, 0.58, 0.61, 0.59, 0.60]}, **kw)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ov_mod, "OVERRIDES_FILE", tmp_path / "ov.json")
    monkeypatch.setattr(ov_mod, "read_effective", lambda k: CURRENT.get(k))
    return tmp_path


CURRENT = {"ml.active_version": ""}


# ------------------------------------------------------------------ ① 该不该重训

def test_detect_drift_no_data():
    assert rt.detect_drift(_snap(accuracy_by_horizon={})) is None


def test_detect_drift_too_few_points():
    assert rt.detect_drift(_snap(accuracy_by_horizon={"5": [0.5, 0.5]})) is None


def test_detect_drift_flags_low_absolute():
    d = rt.detect_drift(_snap())
    assert d is not None
    assert d["horizon"] == "5"
    assert "16.7%" in d["reason"]                # 近 3 批均值 (0.20+0.15+0.15)/3


def test_detect_drift_flags_drop():
    snap = _snap(accuracy_by_horizon={"5": [0.70, 0.72, 0.71, 0.68, 0.55, 0.54, 0.56]})
    d = rt.detect_drift(snap)
    assert d is not None
    assert "下滑" in d["reason"]


def test_detect_drift_healthy_returns_none():
    assert rt.detect_drift(_snap_healthy()) is None


# ------------------------------------------------------------------ ② 现任 vs 挑战者

def test_plan_refuses_without_pin(env):
    """没有基准钉子 → 按兵不动，并说明原因（注册表会自动采用最优产物）。"""
    models = [_info(env, "a", 0.10, 1000), _info(env, "b", 0.20, 2000)]
    p = rt.plan(models, "")
    assert p["mode"] is None
    assert "--pin" in p["reason"]


def test_plan_promotes_better_artifact(env):
    models = [_info(env, "inc", 0.10, 1000), _info(env, "chl", 0.16, 2000)]
    p = rt.plan(models, "inc")
    assert p["mode"] == "promote"
    assert p["incumbent"].key == "inc"
    assert p["challenger"].key == "chl"


def test_plan_no_op_when_pin_is_already_best(env):
    models = [_info(env, "inc", 0.20, 1000), _info(env, "old", 0.10, 500)]
    p = rt.plan(models, "inc")
    assert p["mode"] is None
    assert "已是样本外最优" in p["reason"]


def test_plan_no_op_when_gain_too_small(env):
    models = [_info(env, "inc", 0.100, 1000), _info(env, "chl", 0.102, 2000)]
    p = rt.plan(models, "inc")
    assert p["mode"] is None
    assert "未超出" in p["reason"]


def test_plan_skips_non_loadable(env):
    models = [_info(env, "inc", 0.10, 1000), _info(env, "exp", 0.99, 2000, loadable=False)]
    p = rt.plan(models, "inc")
    assert p["mode"] is None                     # 实验产物不可部署，不予晋升


def test_plan_reports_dead_pin(env):
    models = [_info(env, "a", 0.10, 1000)]
    p = rt.plan(models, "已消失的版本")
    assert p["mode"] is None
    assert "基准失效" in p["reason"]


def test_plan_no_op_on_empty_registry(env):
    p = rt.plan([], "inc")
    assert p["mode"] is None
    assert "没有可部署模型" in p["reason"]


def test_plan_skips_models_without_metric(env):
    import math
    models = [_info(env, "inc", 0.10, 1000),
              _info(env, "nan", math.nan, 2000)]
    p = rt.plan(models, "inc")
    assert p["mode"] is None


# ------------------------------------------------------------------ ③ 证据构造

def test_build_evidence_passes_when_real_gain(env):
    inc = _info(env, "inc", 0.100, 1000)
    chl = _info(env, "chl", 0.130, 2000)
    ev = rt.build_evidence(inc, chl, {"reason": "准确率下滑"})
    assert ev.passed is True
    assert ev.evaluator == "retrain"
    assert ev.delta["rank_ic"] == pytest.approx(0.03)
    assert ev.window
    assert any("RankIC" in r for r in ev.reasons)


def test_build_evidence_fails_on_tiny_gain(env):
    inc = _info(env, "inc", 0.100, 1000)
    chl = _info(env, "chl", 0.101, 2000)
    ev = rt.build_evidence(inc, chl, {"reason": "x"})
    assert ev.passed is False
    assert any("未胜出" in r for r in ev.reasons)


def test_build_evidence_fails_on_too_few_oos_days(env):
    inc = _info(env, "inc", 0.100, 1000)
    chl = _info(env, "chl", 0.200, 2000, n=5)
    ev = rt.build_evidence(inc, chl, {"reason": "x"})
    assert ev.passed is False
    assert any("覆盖 5.0 天" in r or "覆盖 5 天" in r for r in ev.reasons)


def test_build_evidence_includes_ic_ir_when_present(env):
    inc = _info(env, "inc", 0.10, 1000, ir=0.4)
    chl = _info(env, "chl", 0.20, 2000, ir=0.9)
    ev = rt.build_evidence(inc, chl, {"reason": "x"})
    assert ev.delta["ic_ir"] == pytest.approx(0.5)


@pytest.fixture
def pinned(monkeypatch):
    """把基准钉在 ``inc`` 上（模型晋升必须先有基准）。"""
    def _read(k):
        return "inc" if k == "ml.active_version" else CURRENT.get(k)
    monkeypatch.setattr(ov_mod, "read_effective", _read)
    return "inc"


# ------------------------------------------------------------------ ④ generate

def test_generate_skips_when_no_drift(env, pinned):
    assert rt.generate(_snap_healthy(), models_fn=lambda: []) == []


def test_generate_skips_without_pin(env):
    """没有基准钉子 → 不提候选（并在日志里说清原因）。"""
    models = [_info(env, "inc", 0.10, 1000), _info(env, "chl", 0.16, 2000)]
    assert rt.generate(_snap(), models_fn=lambda: models) == []


def test_generate_skips_when_challenger_loses(env, pinned):
    models = [_info(env, "inc", 0.10, 1000), _info(env, "chl", 0.102, 2000)]
    assert rt.generate(_snap(), models_fn=lambda: models) == []


def test_generate_produces_candidate_when_challenger_wins(env, pinned):
    models = [_info(env, "inc", 0.10, 1000), _info(env, "chl", 0.16, 2000)]
    cands = rt.generate(_snap(), models_fn=lambda: models)

    assert len(cands) == 1
    c = cands[0]
    assert c.source == "retrain"
    assert c.changes[0].key == "ml.active_version"
    assert c.changes[0].old == "inc"               # 基准
    assert c.changes[0].new == "chl"               # 晋升目标
    assert c.meta["incumbent"] == "inc"
    assert c.meta["challenger"] == "chl"
    assert "signature" in c.meta


def test_generate_attaches_evidence_to_meta_not_evaluation(env, pinned):
    """生成器**不自盖章**：证据只进 meta，evaluation 留给闸门写。"""
    models = [_info(env, "inc", 0.10, 1000), _info(env, "chl", 0.16, 2000)]
    c = rt.generate(_snap(), models_fn=lambda: models)[0]
    assert c.evaluation is None
    assert c.meta["evidence"]["evaluator"] == "retrain"
    assert c.meta["evidence"]["passed"] is True
    assert c.status == "pending"


def test_generate_rationale_cites_numbers(env, pinned):
    models = [_info(env, "inc", 0.10, 1000), _info(env, "chl", 0.16, 2000)]
    c = rt.generate(_snap(), models_fn=lambda: models)[0]
    assert "+0.1000" in c.rationale and "+0.1600" in c.rationale
    assert "人工批准" in c.rationale
    assert "绝不自动落地" in c.rationale


def test_generate_no_op_when_already_pinned_to_challenger(env, monkeypatch):
    monkeypatch.setattr(
        ov_mod, "read_effective",
        lambda k: "chl" if k == "ml.active_version" else None)
    models = [_info(env, "inc", 0.10, 1000), _info(env, "chl", 0.16, 2000)]
    assert rt.generate(_snap(), models_fn=lambda: models) == []


def test_generate_survives_registry_failure(env, pinned):
    def boom():
        raise RuntimeError("models 目录读不了")

    assert rt.generate(_snap(), models_fn=boom) == []


def test_generate_returns_empty_without_evidence(env, pinned):
    models = [_info(env, "inc", 0.10, 1000), _info(env, "chl", 0.16, 2000)]
    assert rt.generate(_snap(n_verified=0, n_settlements=0),
                       models_fn=lambda: models) == []


# ------------------------------------------------------------------ ⑤ 闸门：自带证据通道

def _cand_with_evidence(ev: Evaluation) -> Candidate:
    return Candidate.create(
        source="retrain", title="t",
        changes=[Change("ml.active_version", "", "chl")],
        meta={"evidence": ev.to_dict(), "signature": "s"})


def test_evaluate_candidate_routes_to_supplied_evidence():
    ev = Evaluation(passed=True, at="now", evaluator="retrain",
                    baseline={"rank_ic": 0.1}, candidate={"rank_ic": 0.2},
                    delta={"rank_ic": 0.1}, reasons=["ok"], window="OOS 120 天")
    out = gate.evaluate_candidate(_cand_with_evidence(ev), provider=lambda: None)
    assert out.passed is True
    assert out.evaluator == "gate:supplied"


def test_evaluate_candidate_rejects_unpassed_evidence():
    ev = Evaluation(passed=False, at="now", evaluator="retrain",
                    delta={"rank_ic": -0.01}, reasons=["没过"], window="OOS 120 天")
    out = gate.evaluate_candidate(_cand_with_evidence(ev), provider=lambda: None)
    assert out.passed is False
    assert any("自评未通过" in r for r in out.reasons)


def test_evaluate_candidate_rejects_forged_evaluator():
    """生成器自称 ``gate:backtest`` 想冒充闸门判定 → 拒。"""
    ev = Evaluation(passed=True, at="now", evaluator="gate:backtest",
                    delta={"rank_ic": 0.1}, reasons=["我说过了"], window="x")
    out = gate.evaluate_candidate(_cand_with_evidence(ev), provider=lambda: None)
    assert out.passed is False
    assert any("白名单" in r for r in out.reasons)


def test_evaluate_candidate_rejects_evidence_without_delta():
    ev = Evaluation(passed=True, at="now", evaluator="retrain",
                    reasons=["ok"], window="x")
    out = gate.evaluate_candidate(_cand_with_evidence(ev), provider=lambda: None)
    assert out.passed is False
    assert any("delta" in r for r in out.reasons)


def test_evaluate_candidate_without_evidence_uses_backtest():
    c = Candidate.create(source="param_search", title="t",
                         changes=[Change("trading_loop.band_pct", 0.25, 0.30)])
    out = gate.evaluate_candidate(c, provider=lambda: None)
    assert out.evaluator == "gate:no-data"       # 无面板 → 走回测路径判负


def test_supplied_evidence_cannot_touch_protected_params(env):
    """自带证据也不能给风控改动背书 —— ``evaluate_candidate`` 先走回测路径前
    的风险预检由 ``evaluate`` 负责；这里确认带证据的受保护候选仍被拦。"""
    ev = Evaluation(passed=True, at="now", evaluator="retrain",
                    delta={"rank_ic": 0.1}, reasons=["ok"], window="x")
    c = Candidate.create(source="retrain", title="t",
                         changes=[Change("guard.max_portfolio_drawdown_pct", 8.0, 50.0)],
                         meta={"evidence": ev.to_dict(), "signature": "s"})
    # 走的是自带证据通道 → 形态合规会放行；因此落地前必须由 runner 二次风控预检拦下
    out = gate.evaluate_candidate(c, provider=lambda: None)
    assert out.passed is True
    blockers, _ = gate.risk_precheck(c.changes)
    assert blockers


# ------------------------------------------------------------------ ⑥ space：KIND_STR 与 ml.active_version

def test_active_version_registered_and_locked():
    spec = space.get_spec(space_kind_key := "ml.active_version")
    assert spec.kind == space.KIND_STR
    assert spec.default == ""
    assert spec.auto_adopt is False          # 换模型绝不自动落地
    assert space_kind_key in space.PARAM_SPACE


def test_str_cast_and_validate():
    assert space.validate("ml.active_version", "abc") == "abc"
    assert space.validate("ml.active_version", None) == ""
    assert space.validate("ml.active_version", 123) == "123"


def test_str_param_is_never_risk_increasing():
    spec = space.get_spec("ml.active_version")
    assert spec.is_risk_increasing("a", "b") is False
    assert spec.is_risk_increasing("b", "a") is False


def test_str_param_not_protected():
    assert not space.is_protected("ml.active_version")


def test_check_changes_adoptable_blocks_model_swap():
    ok, why = space.check_changes_adoptable([("ml.active_version", "", "chl")])
    assert ok is False
    assert "人工批准" in why or "自动" in why


# ------------------------------------------------------------------ ⑦ 注册表：钉住即冻结

def _write_metric(model_dir: Path, stem: str, ric: float, mt: float):
    (model_dir / "xgboost_model.pkl").write_bytes(b"m")
    (model_dir / "scaler.pkl").write_bytes(b"s")
    (model_dir / "features_xgboost.json").write_text("[]", encoding="utf-8")
    f = model_dir / f"{stem}.json"
    f.write_text(json.dumps({
        "walk_forward": {"rank_ic": ric, "ic_ir": 0.4, "top_win_rate": 0.55,
                         "n_days": 120},
        "model_path": "xgboost_model.pkl",
    }), encoding="utf-8")
    os.utime(f, (mt, mt))
    return f


@pytest.fixture
def registry(tmp_path, monkeypatch):
    from src.models import model_registry as mr
    from config.settings import config

    monkeypatch.setattr(mr, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config.ml, "active_version", "", raising=False)
    return mr, config, tmp_path


def test_registry_no_pin_picks_best(registry):
    mr, cfg, d = registry
    _write_metric(d, "m_low", 0.10, 1000)
    _write_metric(d, "m_high", 0.20, 2000)
    assert mr.pinned_version() == ""
    assert mr.best_loadable().key == "m_high"


def test_registry_pin_freezes_decision(registry):
    """钉住低分模型后，高分的新产物**不会**被自动启用。"""
    mr, cfg, d = registry
    _write_metric(d, "m_low", 0.10, 1000)
    _write_metric(d, "m_high", 0.20, 2000)
    cfg.ml.active_version = "m_low"
    assert mr.best_loadable().key == "m_low"     # 冻结生效
    assert mr.best_model_path().name == "xgboost_model.pkl"


def test_registry_pin_falls_back_when_invalid(registry):
    mr, cfg, d = registry
    _write_metric(d, "m_high", 0.20, 2000)
    cfg.ml.active_version = "不存在的版本"
    assert mr.pinned_model() is None
    assert mr.best_loadable().key == "m_high"    # 优雅回退，绝不无模型可用


# ------------------------------------------------------------------ ⑧ 注册与回路集成

def test_registered_in_registry():
    assert "retrain" in REGISTRY
    assert REGISTRY["retrain"] is rt.generate


def test_generate_never_trains(env, monkeypatch, pinned):
    """生成器不得碰训练/写盘：把训练入口设成「一调就炸」来证明它没被调用。"""
    import scripts.auto_retrain as ar

    def boom(*a, **k):
        raise AssertionError("生成器竟然调用了训练！")

    monkeypatch.setattr(ar, "run_retrain", boom, raising=False)
    models = [_info(env, "inc", 0.10, 1000), _info(env, "chl", 0.16, 2000)]
    assert len(rt.generate(_snap(), models_fn=lambda: models)) == 1


# ------------------------------------------------------------------ ⑨ CLI：--models / --pin

@pytest.fixture
def cli(tmp_path, monkeypatch):
    """把 CLI 的落盘目标与模型目录都指向 tmp。"""
    from src.models import model_registry as mr
    from config.settings import config
    import scripts.evolve as ev

    monkeypatch.setattr(ov_mod, "OVERRIDES_FILE", tmp_path / "ov.json")
    monkeypatch.setattr(mr, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config.ml, "active_version", "", raising=False)
    return ev, mr, config, tmp_path


def test_cli_print_models_warns_when_unpinned(cli, capsys):
    ev, mr, cfg, d = cli
    _write_metric(d, "m_high", 0.20, 2000)

    assert ev._print_models() == 0
    out = capsys.readouterr().out
    assert "未钉住" in out
    # 未钉住的两个真实后果都要说清楚：生成器 C 不工作、且「排序≠生效」
    assert "生成器 C" in out
    assert "人工批准" in out
    assert "m_high" in out


def test_cli_print_models_marks_pinned(cli, capsys):
    ev, mr, cfg, d = cli
    _write_metric(d, "m_high", 0.20, 2000)
    cfg.ml.active_version = "m_high"

    assert ev._print_models() == 0
    assert "📌已钉住" in capsys.readouterr().out


def test_cli_pin_establishes_baseline(cli, capsys):
    ev, mr, cfg, d = cli
    _write_metric(d, "m_high", 0.20, 2000)

    assert ev._pin_model("m_high", "alice") == 0
    assert cfg.ml.active_version == "m_high"          # 已生效到配置单例
    assert mr.best_loadable().key == "m_high"
    assert "📌" in capsys.readouterr().out


def test_cli_pin_refuses_unloadable_key(cli, capsys):
    ev, mr, cfg, d = cli
    _write_metric(d, "m_high", 0.20, 2000)

    assert ev._pin_model("不存在的键", "alice") == 2
    err = capsys.readouterr().err
    assert "不可部署" in err
    assert cfg.ml.active_version == ""                # 没被改坏


def test_cli_unpin_returns_to_auto(cli, capsys):
    ev, mr, cfg, d = cli
    _write_metric(d, "m_low", 0.10, 1000)
    _write_metric(d, "m_high", 0.20, 2000)
    cfg.ml.active_version = "m_low"
    assert mr.best_loadable().key == "m_low"

    assert ev._pin_model("", "alice") == 0
    assert cfg.ml.active_version == ""
    assert mr.best_loadable().key == "m_high"         # 回到自动挑选
    assert "取消钉住" in capsys.readouterr().out


def test_cli_pin_is_idempotent(cli, capsys):
    ev, mr, cfg, d = cli
    _write_metric(d, "m_high", 0.20, 2000)
    cfg.ml.active_version = "m_high"
    ev._pin_model("m_high", "alice")
    # 再钉一次：值没变，apply_changes 无实际操作
    assert ev._pin_model("m_high", "alice") == 0

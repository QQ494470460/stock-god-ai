# -*- coding: utf-8 -*-
"""模型注册中心测试。

覆盖两件事，都是真踩过的坑：

1. **「可部署」必须等于「能作为主模型被加载」**。
   历史上这个判断只检查了主模型三件套是否存在，于是 alpha 模型
   （RankIC 0.1204，比生产模型 0.0747 高一截）被判为「可部署最优」，
   注册表于是报出一个**永远不会被加载**的冠军 —— 用户会以为系统跑的是
   0.1204 的模型。alpha 需要自己那套 alpha_* 特征列，塞进主管线直接崩。

2. **指标高 ≠ 会生效**。生效必须来自显式的人工批准名单；
   没有任何批准时一律回退到生产默认 `xgboost_model.pkl`。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.models.model_registry as mr  # noqa: E402


def _touch(p: Path, content=b"x"):
    p.write_bytes(content)


def _main_family(d: Path):
    for f in ("xgboost_model.pkl", "scaler.pkl", "features_xgboost.json"):
        _touch(d / f)
    (d / "production_metrics.json").write_text(json.dumps({
        "walk_forward": {"rank_ic": 0.075, "ic_ir": 6.0, "top_win_rate": 0.6, "n_days": 200},
        "model_path": "xgboost_model.pkl",
    }), encoding="utf-8")


def _alpha_family(d: Path):
    for f in ("alpha_xgboost_model.pkl", "alpha_scaler.pkl", "alpha_label_encoder.pkl"):
        _touch(d / f)
    (d / "alpha_model_metrics.json").write_text(json.dumps({
        "model_type": "alpha_xgboost",
        "rank_ic": 0.123, "ic_ir": 8.1, "top_win_rate": 0.66, "n_days": 180,
        "model_path": "alpha_xgboost_model.pkl",
        "alpha_features": ["alpha_northbound", "alpha_margin"],
    }), encoding="utf-8")


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    """隔离 MODEL_DIR、批准名单目录**与钉子** —— 绝不读写真 config/ 与 models/。

    `ml.active_version` 必须一起清空：运维在真实环境里钉过基准之后，
    若测试仍读真实配置，这些用例就会因为「外面有钉子」而集体失败 ——
    一个取决于环境状态的测试，等于没有测试。
    """
    from config.settings import config

    monkeypatch.setattr(mr, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(mr, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config.ml, "active_version", "", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# ① alpha 不是主模型候选
# ---------------------------------------------------------------------------

def test_alpha_registered_but_not_main_loadable(reg):
    _main_family(reg)
    _alpha_family(reg)

    infos = mr.scan_models(reg)
    keys = {m.key for m in infos}
    assert "alpha_xgboost" in keys

    alpha = next(m for m in infos if m.key == "alpha_xgboost")
    assert abs(alpha.rank_ic - 0.123) < 1e-9
    assert alpha.loadable is False, "alpha 需要 alpha_* 特征列，不能当主模型"
    assert alpha.role == "ensemble_leg"

    # 关键：哪怕 alpha 的 RankIC 更高，它也不该出现在「可作主模型的最优」里
    best = mr.best_loadable(reg)
    assert best is not None and best.key == "production_metrics", \
        "可作主模型的最优被 alpha 顶掉了 —— 这就是「注册表报出一个不会被加载的冠军」"


def test_alpha_missing_files_not_loadable(reg):
    (reg / "alpha_model_metrics.json").write_text(json.dumps({
        "model_type": "alpha_xgboost", "rank_ic": 0.2,
    }), encoding="utf-8")
    infos = mr.scan_models(reg)
    alpha = next((m for m in infos if m.key == "alpha_xgboost"), None)
    assert alpha is not None and alpha.loadable is False


def test_experiment_log_still_registered(reg):
    (reg / "walk_forward_alpha.json").write_text(json.dumps({
        "model_A": {"summary": {"rank_ic": 0.08, "n_days": 100}},
        "model_B": {"summary": {"rank_ic": 0.123, "n_days": 100}},
    }), encoding="utf-8")
    infos = mr.scan_models(reg)
    keys = {m.key for m in infos}
    assert {"alpha_A", "alpha_B"} <= keys
    assert all(not m.loadable for m in infos)


# ---------------------------------------------------------------------------
# ② 同一模型被多个指标文件描述时按 key 去重
# ---------------------------------------------------------------------------

def test_scan_dedupes_by_key_keeping_higher_rank_ic(reg):
    _main_family(reg)
    # 第二个文件描述同一个 model_A，但分数更低
    (reg / "walk_forward_alpha.json").write_text(json.dumps({
        "model_A": {"summary": {"rank_ic": 0.09, "n_days": 100}},
    }), encoding="utf-8")
    (reg / "walk_forward_alpha_ablation.json").write_text(json.dumps({
        "model_A": {"summary": {"rank_ic": 0.07, "n_days": 100}},
    }), encoding="utf-8")

    infos = mr.scan_models(reg)
    keys = [m.key for m in infos]
    assert keys.count("alpha_A") == 1, "同名条目必须只出现一次，否则用户不知道信哪一行"
    a = next(m for m in infos if m.key == "alpha_A")
    assert abs(a.rank_ic - 0.09) < 1e-9, "应保留 RankIC 较高的那一份"
    assert "来源" in a.note, "note 里要记下来自哪个文件，便于回溯"


# ---------------------------------------------------------------------------
# ③ 人工批准名单 —— 封堵「丢个文件就静默生效」
# ---------------------------------------------------------------------------

def test_no_approvals_means_nothing_gets_promoted(reg):
    _main_family(reg)
    # 连更高分的新产物都造出来
    (reg / "shiny_metrics.json").write_text(json.dumps({
        "walk_forward": {"rank_ic": 0.99, "n_days": 200},
        "model_path": "xgboost_model.pkl",
    }), encoding="utf-8")

    assert mr.load_approvals() == {}
    assert mr.approved_keys() == []
    assert mr.resolve_active(reg) is None, \
        "没有批准名单时不得擅自挑一个「指标最高的」生效"
    assert mr.resolve_active_path(reg).name == "xgboost_model.pkl"


def test_approve_then_resolve(reg):
    _main_family(reg)
    mr.approve_version("production_metrics", operator="alice", note="人工核对过")

    assert mr.approved_keys() == ["production_metrics"]
    got = mr.resolve_active(reg)
    assert got is not None and got.key == "production_metrics"
    assert mr.resolve_active_path(reg).name == "xgboost_model.pkl"

    # 落盘可读
    raw = json.loads(mr.approvals_file().read_text(encoding="utf-8"))
    assert raw["approved"][0]["approved_by"] == "alice"


def test_approve_ignores_unloadable_key(reg):
    """批准名单不接受「不能作主模型」的键 —— 否则名单会变成一堆无效记录。"""
    _main_family(reg)
    _alpha_family(reg)
    mr.approve_version("alpha_xgboost", operator="alice")
    # 名单里有它，但它不是可部署候选 → 裁决依然不给它
    assert mr.resolve_active(reg) is None


def test_revoke(reg):
    _main_family(reg)
    mr.approve_version("production_metrics", operator="alice")
    assert mr.revoke_version("production_metrics") is True
    assert mr.approved_keys() == []
    assert mr.revoke_version("production_metrics") is False


def test_pin_beats_approvals(reg):
    """钉子优先于批准名单：钉住的是运维明确指定的那一个。"""
    _main_family(reg)
    (reg / "other_metrics.json").write_text(json.dumps({
        "walk_forward": {"rank_ic": 0.99, "n_days": 200},
        "model_path": "xgboost_model.pkl",
    }), encoding="utf-8")
    mr.approve_version("other_metrics", operator="alice")

    from config.settings import config
    old = config.ml.active_version
    try:
        config.ml.active_version = "production_metrics"
        got = mr.resolve_active(reg)
        assert got is not None and got.key == "production_metrics"
    finally:
        config.ml.active_version = old


def test_corrupt_approvals_fails_closed(reg):
    """名单文件损坏 → 按空名单处理（更保守），绝不「解析不了就全放行」。"""
    mr.approvals_file().write_text("{ 这不是 JSON", encoding="utf-8")
    assert mr.load_approvals() == {}
    _main_family(reg)
    assert mr.resolve_active(reg) is None


def test_approve_requires_key(reg):
    with pytest.raises(ValueError):
        mr.approve_version("   ")


def test_active_status_explains_when_nothing_approved(reg):
    _main_family(reg)
    s = mr.active_status(reg)
    assert "生产默认" in s or "批准" in s

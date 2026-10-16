# -*- coding: utf-8 -*-
"""predictor._load_models 接通注册表裁决的端到端验证。

核心要证明的事（之前「以为上线了其实什么都没变」的病根）：
1. 有被批准/钉住的版本时，_load_models 真正走 load_model_from(版本路径)，
   而不是死守硬编码的 xgboost_model.pkl。
2. 没有生效版本 / 版本三件套不齐时，明确拒绝版本、回退到遗留默认路径
   （行为与改动前逐字一致，绝不改变当前选股结果）。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models import predictor as predictor_mod          # noqa: E402
from src.models.model_registry import ModelInfo, version_set_for, write_version_bundle  # noqa: E402


def _touch(p: Path, content=b"x"):
    p.write_bytes(content)


def _legacy_family(d: Path):
    _touch(d / "xgboost_model.pkl")
    _touch(d / "scaler.pkl")
    _touch(d / "label_encoder.pkl")
    (d / "features_xgboost.json").write_text(
        json.dumps({"features": ["a", "b"]}), encoding="utf-8")


def _make_predictor(tmp_path, monkeypatch):
    _legacy_family(tmp_path)
    # 隔离 predictor 模块级的 MODEL_DIR（_load_feature_list 用它拼 features 路径）
    monkeypatch.setattr(predictor_mod, "MODEL_DIR", tmp_path)
    p = predictor_mod.Predictor(model_dir=tmp_path)
    return p


class FakeTrainer:
    """记录「实际走了哪条加载路径」，用 sentinel 代替真实反序列化。"""

    def __init__(self):
        self.calls = []

    def load_model(self, name):
        self.calls.append(("load_model", name))
        return {"legacy": name}

    def load_model_from(self, path):
        self.calls.append(("load_model_from", str(path)))
        return {"versioned": str(path)}

    def load_preprocessing(self):
        self.calls.append(("load_preprocessing", None))
        return "scaler", "le"

    def load_preprocessing_from(self, sp, ep):
        self.calls.append(("load_preprocessing_from", str(sp), str(ep)))
        return "scaler", "le"


def _route(predictor, tmp_path, monkeypatch, info_factory):
    ft = FakeTrainer()
    predictor.trainer = ft
    monkeypatch.setattr(predictor_mod, "resolve_active", info_factory)
    # version_set_for 用真实逻辑，但指定 model_dir=tmp_path 以读临时目录。
    # 签名必须收 (info, model_dir)：生产代码调用时会把 self.model_dir 一并传进来，
    # 少收一个参数会抛 TypeError 并被 _load_models 的 except 吞掉，
    # 表现为「静默走回遗留分支」——测试会假通过。
    monkeypatch.setattr(
        predictor_mod, "version_set_for",
        lambda i, model_dir=None: version_set_for(i, tmp_path),
    )
    predictor._load_models()
    return ft


def test_loads_legacy_when_no_approval(monkeypatch, tmp_path):
    p = _make_predictor(tmp_path, monkeypatch)
    ft = _route(p, tmp_path, monkeypatch, lambda: None)

    assert ("load_model", "xgboost") in ft.calls
    assert p._xgb_model == {"legacy": "xgboost"}
    # 版本路径绝不能被走
    assert not any(c[0] == "load_model_from" for c in ft.calls)


def test_loads_versioned_when_approved(monkeypatch, tmp_path):
    p = _make_predictor(tmp_path, monkeypatch)
    vdir = tmp_path / "versions" / "v2"
    vdir.mkdir(parents=True)
    _touch(vdir / "model.pkl")
    _touch(vdir / "scaler.pkl")
    _touch(vdir / "label_encoder.pkl")
    (vdir / "features.json").write_text(json.dumps({"features": ["a", "b"]}), encoding="utf-8")

    def _info():
        return ModelInfo(key="v2", model_path=vdir / "model.pkl", prefix="v2", loadable=True)

    ft = _route(p, tmp_path, monkeypatch, _info)

    assert ("load_model_from", str(vdir / "model.pkl")) in ft.calls
    assert p._xgb_model == {"versioned": str(vdir / "model.pkl")}
    # 遗留路径绝不能被走
    assert ("load_model", "xgboost") not in [c for c in ft.calls]


def test_refuses_incomplete_version_falls_back(monkeypatch, tmp_path):
    p = _make_predictor(tmp_path, monkeypatch)
    vdir = tmp_path / "versions" / "v3"
    vdir.mkdir(parents=True)
    _touch(vdir / "model.pkl")  # 缺 scaler/le/features → 三件套不齐

    def _info():
        return ModelInfo(key="v3", model_path=vdir / "model.pkl", prefix="v3", loadable=False)

    ft = _route(p, tmp_path, monkeypatch, _info)

    # 版本路径被拒，回退遗留默认
    assert not any(c[0] == "load_model_from" for c in ft.calls)
    assert ("load_model", "xgboost") in ft.calls


def test_legacy_approved_still_uses_legacy_path(monkeypatch, tmp_path):
    """批准的是遗留生产模型本身（production_metrics）时，仍走遗留路径、行为不变。"""
    p = _make_predictor(tmp_path, monkeypatch)

    def _info():
        # 与 test_model_registry 里 _main_family 的 production_metrics 同形
        return ModelInfo(key="production_metrics",
                        model_path=tmp_path / "xgboost_model.pkl",
                        prefix="", loadable=True)

    ft = _route(p, tmp_path, monkeypatch, _info)

    assert ("load_model", "xgboost") in ft.calls
    assert not any(c[0] == "load_model_from" for c in ft.calls)


def test_is_ready_follows_resolved_path(monkeypatch, tmp_path):
    p = _make_predictor(tmp_path, monkeypatch)
    vdir = tmp_path / "versions" / "v4"
    vdir.mkdir(parents=True)
    _touch(vdir / "model.pkl")
    _touch(vdir / "scaler.pkl")
    _touch(vdir / "label_encoder.pkl")
    (vdir / "features.json").write_text(json.dumps({"features": ["a"]}), encoding="utf-8")

    def _info():
        return ModelInfo(key="v4", model_path=vdir / "model.pkl", prefix="v4", loadable=True)

    # 批准版本 → is_ready 应检查版本路径（都存在 → True）
    monkeypatch.setattr(predictor_mod, "resolve_active", _info)
    assert p.is_ready is True

    # 版本三件套缺失 → is_ready 反映拒绝回退后的默认（默认文件都在 → 仍 True，
    # 但路径解析不应抛错；这里验证「解析 + 存在性」不崩且指向正确文件）
    bad = tmp_path / "versions" / "vbad"
    bad.mkdir(parents=True)
    _touch(bad / "model.pkl")  # 缺 scaler/features

    def _info_bad():
        return ModelInfo(key="vbad", model_path=bad / "model.pkl", prefix="vbad", loadable=False)

    monkeypatch.setattr(predictor_mod, "resolve_active", _info_bad)
    # 版本不齐 → 回退默认；默认文件都在 → ready
    assert p.is_ready is True


def test_is_ready_is_a_real_boolean_not_a_truthy_bound_method(monkeypatch, tmp_path):
    """负向对照：证明「模型未加载」这道闸门真的关得上。

    少了 @property 时，`predictor.is_ready` 是个**恒为真**的绑定方法对象，
    于是散落在 agent tools / API / scripts / 看板的
    `if not predictor.is_ready:` 一次都不会触发 —— 不报错，只是那道门没关过。
    这里用「空目录 → 必须是 False」证明它有牙齿。

    resolve_active 必须钉成 None：否则会去读真实 config 里的钉子/批准名单，
    测试就成了「取决于外部环境有没有钉过模型」的假测试。
    """
    monkeypatch.setattr(predictor_mod, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predictor_mod, "resolve_active", lambda: None)
    p = predictor_mod.Predictor(model_dir=tmp_path)  # 空目录：没有任何模型文件

    assert isinstance(p.is_ready, bool), (
        "is_ready 必须是真正的布尔值；拿到绑定方法说明 @property 掉了，"
        "`if not predictor.is_ready:` 将永远为真")
    assert p.is_ready is False, (
        "空目录却报「模型就绪」——正是 is_ready 失效时保护不触发的症状")


def test_loads_versioned_is_a_real_switch_not_silent_fallback(monkeypatch, tmp_path):
    """正向对照：证明「批准版本」真的换了加载路径，而不是静默走回默认。

    这条存在的理由：若 version_set_for 的签名少收 model_dir（或调用方没传），
    会抛 TypeError 并被 _load_models 的 except 吞掉 → vs=None → 走遗留分支。
    那种情况下 test_loads_versioned_when_approved 会**假通过**（它只断言
    遗留路径被调用过），必须有一条会因此失败的断言钉住它。
    """
    p = _make_predictor(tmp_path, monkeypatch)
    vdir = tmp_path / "versions" / "v5"
    vdir.mkdir(parents=True)
    for f in ("model.pkl", "scaler.pkl", "label_encoder.pkl"):
        _touch(vdir / f)
    (vdir / "features.json").write_text(json.dumps({"features": ["a"]}), encoding="utf-8")

    def _info():
        return ModelInfo(key="v5", model_path=vdir / "model.pkl", prefix="v5", loadable=True)

    ft = _route(p, tmp_path, monkeypatch, _info)

    # 版本路径必须真的被调用；且加载到的对象是版本那份（不是遗留那份）
    assert ("load_model_from", str(vdir / "model.pkl")) in ft.calls
    assert p._xgb_model == {"versioned": str(vdir / "model.pkl")}
    # 预处理也必须来自版本目录，而不是遗留的 scaler.pkl
    assert ("load_preprocessing_from",
            str(vdir / "scaler.pkl"), str(vdir / "label_encoder.pkl")) in ft.calls

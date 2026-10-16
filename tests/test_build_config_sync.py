# -*- coding: utf-8 -*-
"""构建脚本「运行期配置同步」测试。

背景：打包后 CONFIG_DIR = <exe 同级>/config/，与源码 config/ 是两套独立目录。
`settings.py` 在包内，但 `.salt` / `broker_accounts.enc` / `deepseek.yaml` 等
**用户数据**不在 —— 不搬过去，用户切到 exe 后会发现「账号没了 / AI 功能没了」。
（曾真实踩坑：exe 自检报「未保存任何券商账号」，而源码目录里明明有。）

覆盖点：
  1. 空目录 → 清单内文件全部补齐
  2. 绝不拷贝 settings.py（外挂 config/ 放一份可能遮蔽包内模块）
  3. 已存在的文件不被覆盖（保留用户在 exe 里的改动）
  4. 部分缺失 → 只补缺的那些
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def build_exe():
    """按文件路径加载 build_exe（它不是包，且模块级无副作用）。"""
    spec = importlib.util.spec_from_file_location(
        "_build_exe_under_test", ROOT / "build_exe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_copies_all_missing(build_exe, tmp_path):
    """空目录：清单内文件应全部补齐。"""
    tgt = tmp_path / "dist" / "选股神AI"
    got = build_exe._sync_runtime_config(tgt)
    assert set(got) == set(build_exe.RUNTIME_CFG_FILES)
    for name in build_exe.RUNTIME_CFG_FILES:
        assert (tgt / "config" / name).is_file()


def test_never_copies_settings_py(build_exe, tmp_path):
    """settings.py 必须留在包里，外挂 config/ 不得出现副本。"""
    assert "settings.py" not in build_exe.RUNTIME_CFG_FILES
    tgt = tmp_path / "dist" / "选股神AI"
    build_exe._sync_runtime_config(tgt)
    assert not (tgt / "config" / "settings.py").exists()


def test_does_not_overwrite_existing(build_exe, tmp_path):
    """用户改过的文件不得被覆盖。"""
    tgt = tmp_path / "dist" / "选股神AI"
    (tgt / "config").mkdir(parents=True)
    edited = tgt / "config" / "deepseek.yaml"
    edited.write_text("USER_EDITED", encoding="utf-8")

    got = build_exe._sync_runtime_config(tgt)

    assert "deepseek.yaml" not in got
    assert edited.read_text(encoding="utf-8") == "USER_EDITED"


def test_only_fills_the_missing_one(build_exe, tmp_path):
    """部分缺失时，只补缺的那些。"""
    tgt = tmp_path / "dist" / "选股神AI"
    build_exe._sync_runtime_config(tgt)          # 先铺满
    (tgt / "config" / ".salt").unlink()          # 删掉一个

    got = build_exe._sync_runtime_config(tgt)

    assert got == [".salt"]
    assert (tgt / "config" / ".salt").is_file()


def test_creates_config_dir_when_absent(build_exe, tmp_path):
    """目标 config/ 不存在时应自动创建。"""
    tgt = tmp_path / "brand_new"
    assert not (tgt / "config").exists()
    build_exe._sync_runtime_config(tgt)
    assert (tgt / "config").is_dir()

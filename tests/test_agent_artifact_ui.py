# -*- coding: utf-8 -*-
"""AI 助手「产物下载」链路的 UI 层回归测试。

背景：下载按钮是 `make_report` 这条链路的最后一公里 ——
工具写盘 → 引擎 `_ingest_artifacts` 收进 `AgentResult.artifacts` → UI 渲染
`st.download_button`。中间任何一环断了，用户看到的就是「助手说写好了，
但页面上根本没有下载按钮」。pytest 里不渲染页面、也不点按钮，所以这条
链路此前零覆盖。

两层验证：
- 单元层：用假容器捕获 `download_button` 的真实调用参数（文件名/内容/类型/文案）；
- 运行层：用 Streamlit AppTest 真正跑一遍渲染，确认在真 Streamlit 里不崩。

要证明的：
1. 产物文件存在 → 渲染出下载按钮，且**文件名与内容就是那个文件**；
2. 产物路径不存在（写了 marker 但没落盘 / 路径拼错）→ **一个按钮都不渲染**，
   不能给用户点了报错的空按钮。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit.testing.v1")

# 极简宿主脚本：只渲染 agent 页的产物段，避开 app.py 的整页重量级依赖。
_HOST = r'''
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT))

from src.ui.agent_page import _render_result


class R:
    pending_action = None
    steps = None
    artifacts = [{artifacts_expr}]
    error = None
    final_text = "（测试）"


_render_result(st, R(), "（测试）")
'''


class _FakeContainer:
    """捕获渲染调用，用来对 download_button 的参数做精确断言。"""

    def __init__(self):
        self.downloads = []

    def markdown(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def download_button(self, **kwargs):
        self.downloads.append(kwargs)


def _result(artifacts):
    class R:
        pending_action = None
        steps = None
        error = None
        final_text = ""

    R.artifacts = artifacts
    return R()


def test_download_button_carries_filename_and_content(tmp_path):
    """产物存在 → 渲染 1 个按钮，文件名/内容/mime/文案都对得上。"""
    from src.ui import agent_page

    art = tmp_path / "screen_report_20261016.md"
    body = "# 选股研报\n\n测试内容 123\n"
    art.write_text(body, encoding="utf-8")

    box = _FakeContainer()
    agent_page._render_result(box, _result([str(art)]), "正文")

    assert len(box.downloads) == 1, f"应渲染 1 个下载按钮，实际 {len(box.downloads)}：{box.downloads}"
    kw = box.downloads[0]
    assert kw["file_name"] == art.name, f"文件名不对：{kw.get('file_name')}"
    assert kw["data"] == body, "按钮携带的内容与文件内容不一致（用户会下到错文件）"
    assert kw["mime"] == "text/markdown", f"mime 不对：{kw.get('mime')}"
    assert art.name in kw["label"], f"按钮文案未含文件名：{kw.get('label')}"


def test_missing_artifact_yields_no_button(tmp_path):
    """产物不存在 → 一个按钮都不渲染（负向对照）。"""
    from src.ui import agent_page

    missing = tmp_path / "does_not_exist.md"
    assert not missing.exists()

    box = _FakeContainer()
    agent_page._render_result(box, _result([str(missing)]), "正文")
    assert box.downloads == [], f"文件不存在却渲染了按钮：{box.downloads}"


def test_render_does_not_crash_in_real_streamlit(tmp_path):
    """真 Streamlit 里跑一遍渲染（AppTest），确认产物段不抛异常。"""
    from streamlit.testing.v1 import AppTest

    art = tmp_path / "screen_report_20261016.md"
    art.write_text("# 研报\n", encoding="utf-8")

    p = tmp_path / "host_app.py"
    p.write_text(_HOST.format(root=str(ROOT), artifacts_expr=f"r'{art}'"), encoding="utf-8")

    at = AppTest.from_file(str(p), default_timeout=60)
    at.run()
    assert not at.exception, f"渲染产物段时抛异常：{at.exception}"

# -*- coding: utf-8 -*-
"""单股推荐页「超级集成腿静默降级」横幅单元测试。

为什么用 AST 抽取而不是直接 import
----------------------------------
``src/ui/app.py`` 是 Streamlit 顶层脚本，模块级会拉起调度器等重依赖，
直接 import 有副作用。而 ``_render_ensemble_degradation_banner`` 必须留在
app.py 内（移出去 pyinstaller 冻结包会 ModuleNotFoundError）。
所以只对这一个函数做 AST 切片 + exec，并用 Fake st 捕获 warning 调用。

测什么
------
单股推荐页横幅的**门控逻辑**（这是「健康观测闭环」在单股页的最后一道触达）：
1. 有降级告警（alarm 非空）→ 必须渲染 ⚠️ 横幅，且文案含「残缺集成」+ 原始告警；
2. alarm 为 None / 空串 → 不渲染，保持零噪声（与首屏/面板「健康态不刷屏」一致）。
"""

import ast
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


class _FakeSt:
    """捕获 st.warning 调用，不依赖真实 streamlit 运行时。"""

    def __init__(self):
        self.warnings = []

    def warning(self, msg):
        self.warnings.append(msg)


def _load_banner():
    src = (ROOT / "src" / "ui" / "app.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "_render_ensemble_degradation_banner":
            break
    else:
        raise AssertionError(
            "_render_ensemble_degradation_banner 在 app.py 找不到（改名了？同步更新本测试）"
        )
    st = _FakeSt()
    mod = types.ModuleType("app_banner_frag")
    mod.__dict__["st"] = st
    exec(compile(ast.Module(body=[n], type_ignores=[]), "<banner-frag>", "exec"),
         mod.__dict__)
    return mod, st


def test_banner_renders_when_leg_down():
    """有腿降级（alarm 非空）→ 必须渲染 ⚠️ 横幅，且文案含「残缺集成」+ 原始告警。"""
    mod, st = _load_banner()
    alarm = ("⚠️ 超级集成有腿静默降级：\n"
             "  · lstm：RuntimeError: lstm_best.pt 缺失")
    mod._render_ensemble_degradation_banner(alarm)
    assert len(st.warnings) == 1
    msg = st.warnings[0]
    assert "残缺集成" in msg, "横幅必须点明「本次推荐基于残缺集成」"
    assert alarm in msg, "横幅必须透出原始降级告警（哪条腿、为什么）"


def test_banner_silent_when_alarm_none():
    """alarm 为 None（全部腿健康 / 未启用）→ 不渲染，零噪声。"""
    mod, st = _load_banner()
    mod._render_ensemble_degradation_banner(None)
    assert st.warnings == []


def test_banner_silent_when_alarm_empty():
    """alarm 空串（降级集合为空）→ 同样不渲染。"""
    mod, st = _load_banner()
    mod._render_ensemble_degradation_banner("")
    assert st.warnings == []

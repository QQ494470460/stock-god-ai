"""看板「选股信号强度」横幅测试。

为什么单独测
------------
横幅文案是用户判断"今天该不该按这 8 只推荐建仓"的唯一依据。它埋在
``render_today_status`` 里就只能靠肉眼验证，所以渲染逻辑抽成了纯函数
``_signal_strength_banner``；这里用 AST 切片执行（``src/ui/app.py`` 是
Streamlit 顶层脚本，import 会拉起调度器等重依赖，本仓其它 UI 用例也这么做）。
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

_WANT = ("_signal_strength_banner",)


def _load():
    src = (ROOT / "src" / "ui" / "app.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _WANT]
    missing = set(_WANT) - {n.name for n in nodes}
    assert not missing, f"app.py 里找不到 {sorted(missing)}（改名了？同步更新本测试）"
    frag = types.ModuleType("app_signal_fragment")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<app-frag>", "exec"),
         frag.__dict__)
    return frag


def test_banner_none_for_healthy_signal():
    f = _load()
    assert f._signal_strength_banner({"level": "strong", "strength": 100}) is None
    assert f._signal_strength_banner({"level": "ok", "strength": 60}) is None
    assert f._signal_strength_banner(None) is None
    assert f._signal_strength_banner({}) is None


def test_banner_shows_for_weak_signal():
    f = _load()
    html = f._signal_strength_banner({
        "level": "weak", "level_cn": "弱", "dispersion": 1.38, "strict_pass": 0,
        "note": "达标 0 只（阈值 58）,区分度仅 1.38", "generated_at": "2026-11-13 18:53",
    })
    assert html and "信号偏弱" in html
    assert "σ1.38" in html and "达标 0 只" in html
    assert "require_strict_pass" in html      # 必须给出可执行的下一步


def test_banner_stronger_wording_when_none():
    f = _load()
    html = f._signal_strength_banner({"level": "none", "dispersion": 0.0,
                                     "strict_pass": 0, "note": "接近随机"})
    assert "基本无信号" in html


def test_banner_includes_decay_trend():
    f = _load()
    html = f._signal_strength_banner({
        "level": "weak", "dispersion": 1.2, "strict_pass": 0, "note": "x",
        "trend": "decaying", "trend_note": "信号正在衰减：强度 -40.0",
    })
    assert "信号正在衰减" in html


def test_banner_survives_missing_fields():
    """字段缺失时只允许显示 "None"，绝不允许抛异常把首屏搞崩。"""
    f = _load()
    html = f._signal_strength_banner({"level": "weak"})
    assert html and "<div" in html

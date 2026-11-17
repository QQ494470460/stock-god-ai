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

_WANT = ("_signal_strength_banner", "_pick_quality_banner")


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


# ---------------------------------------------------------------------------
# 选股预测力横幅（比信号强度更硬的一条）
# ---------------------------------------------------------------------------

def test_quality_banner_none_when_effective_and_monitor_ok():
    f = _load()
    assert f._pick_quality_banner(
        {"verdict": "effective", "monitor": {"stalled": False}}) is None
    assert f._pick_quality_banner(None) is None
    assert f._pick_quality_banner({}) is None
    assert f._pick_quality_banner({"verdict": "no_data"}) is None


def test_quality_banner_flags_false_strong_signal():
    """σ 很分散（信号判"良"）但 RankIC 已衰减 —— 这是最会骗人加仓的组合。"""
    f = _load()
    html = f._pick_quality_banner(
        {"verdict": "decayed", "verdict_cn": "已衰减",
         "note": "近 20 日均值 -0.0099", "monitor": {"stalled": False}},
        {"level": "ok", "level_cn": "良", "dispersion": 3.27},
    )
    assert html and "假强信号" in html
    assert "σ3.27" in html and "已衰减" in html
    assert "调阈值无效" in html, "必须点明真正的改进方向，否则用户只会去调阈值"


def test_quality_banner_without_signal_context_still_warns():
    f = _load()
    html = f._pick_quality_banner(
        {"verdict": "decayed", "verdict_cn": "已衰减",
         "note": "均值为负", "monitor": {"stalled": False}}, None)
    assert html and "预测力已衰减" in html
    assert "假强信号" not in html


def test_quality_banner_surfaces_stalled_monitor():
    """监控死了必须单独说一句 —— 否则"质量看起来还行"其实是陈旧数据。"""
    f = _load()
    html = f._pick_quality_banner(
        {"verdict": "decayed", "verdict_cn": "已衰减", "note": "x",
         "monitor": {"stalled": True, "note": "状态停在 running 已 103.2h"}},
        {"level": "weak", "level_cn": "弱"},
    )
    assert html and "质量监控自身出问题" in html
    assert "103.2h" in html


def test_quality_banner_survives_minimal_dict():
    f = _load()
    html = f._pick_quality_banner({"verdict": "marginal"})
    assert html and "<div" in html


def test_quality_banner_leads_with_rank_no_edge():
    """名次无信息比"全池 IC 样本不足"更贴近体感，应当占据标题位。"""
    f = _load()
    html = f._pick_quality_banner(
        {"verdict": "insufficient", "note": "全池仅 7 日",
         "monitor": {"stalled": False},
         "list_ic": {"verdict": "no_edge", "verdict_cn": "名次无信息",
                     "n_periods": 21, "horizon": "5", "note": "名次没有预测力"}},
        None,
    )
    assert html and "名单内名次无预测力" in html
    assert "名次没有预测力" in html and "21 期" in html



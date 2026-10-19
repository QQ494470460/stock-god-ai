# -*- coding: utf-8 -*-
"""UI 版式组件的回归测试（不 import app.py —— 它有模块级副作用）。

重点不是"能跑通"，而是钉死两件事：
  1) 组件吐出的 HTML 结构/类名符合设计系统；
  2) **组件用到的每个类名，全局 CSS 里必须真的有定义** ——
     历史上出过"代码一直在用 .text-sub / .chart-box，样式表里压根没定义"
     的静默失效（等于 no-op），这里用测试堵住同类问题复发。
"""

import ast
from pathlib import Path

from src.ui.common import page_hero, kpi_row, panel, html_metric, empty_block


APP_PY = Path(__file__).parent.parent / "src" / "ui" / "app.py"


def _theme_css() -> str:
    """从 app.py 里抽出 _THEME_CSS 常量（用 ast，避免执行模块级代码）"""
    tree = ast.parse(APP_PY.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_THEME_CSS":
                    return node.value.value
    raise AssertionError("app.py 里找不到 _THEME_CSS 常量")


# 组件产出的类名 → 必须在 CSS 中定义（任一条缺失都会让某处样式静默失效）
_REQUIRED_CLASSES = [
    "hero", "hero-icon", "hero-title", "hero-sub", "hero-badges",
    "pill", "mini-card", "card-row", "panel", "panel-title",
    "empty", "tag", "section-title", "text-sub",
    # 个股分析仪表盘（2026-10-19 改版新增）
    "stock-head", "sh-main", "sh-name", "sh-code", "sh-quote", "sh-price", "sh-chg",
    "score", "ring", "bar", "bars", "score-foot",
    "kv", "kv-i", "trade-list", "trade-row", "mini-table",
]


def test_theme_css_defines_all_component_classes():
    css = _theme_css()
    missing = [c for c in _REQUIRED_CLASSES if f".{c}" not in css]
    assert not missing, f"CSS 缺少这些类的定义（会导致样式静默失效）: {missing}"


def test_theme_css_keeps_backward_compatible_classes():
    """老页面不改一行也要能吃到新样式：历史类名一个都不能删"""
    css = _theme_css()
    for legacy in ["main-header", "sub-header", "chart-box",
                   "tag.bullish", "tag.bearish", "red", "green", "amber"]:
        assert legacy in css, f"历史类名 .{legacy} 被误删"


def test_page_hero_structure():
    html = page_hero("标题", "副标题", icon="📈",
                     badges=[("violet", "集成"), ("up", "IC 0.23"), "普通"])
    assert 'class="hero"' in html
    assert 'class="hero-icon">📈<' in html
    assert 'class="hero-title">标题<' in html
    assert 'class="hero-sub">副标题<' in html
    assert html.count('class="pill') == 3
    assert 'class="pill violet"' in html and 'class="pill up"' in html
    assert 'class="pill gray"' in html  # 只传字符串时回退灰底


def test_page_hero_without_optional_parts():
    html = page_hero("只有标题")
    assert 'class="hero-title"' in html
    assert "hero-sub" not in html
    assert "hero-badges" not in html


def test_kpi_row_emits_toned_cards():
    html_calls = []
    import src.ui.common as common
    orig = common.st.markdown
    try:
        common.st.markdown = lambda body, **kw: html_calls.append(body)
        kpi_row([
            {"label": "股票池", "value": "5577", "sub": "只"},
            {"label": "模型", "value": "就绪", "tone": "up"},
            {"label": "口径", "value": "单模型", "tone": "warn"},
        ])
    finally:
        common.st.markdown = orig

    assert len(html_calls) == 1
    html = html_calls[0]
    assert html.count('class="mini-card') == 3
    assert 'class="mini-card up"' in html
    assert 'class="mini-card warn"' in html
    assert "只" in html


def test_kpi_row_empty_is_noop():
    import src.ui.common as common
    calls = []
    orig = common.st.markdown
    try:
        common.st.markdown = lambda body, **kw: calls.append(body)
        kpi_row([])
    finally:
        common.st.markdown = orig
    assert calls == []


def test_panel_wraps_title_and_body():
    html = panel("交易建议", "<b>看多</b>", icon="🎯")
    assert html.startswith('<div class="panel">')
    assert 'class="panel-title">🎯 交易建议<' in html
    assert "<b>看多</b>" in html
    assert html.endswith("</div>")


def test_html_metric_tone_follows_change():
    """涨跌色带必须跟 chg 走（A股：涨红跌绿），不能恒为同一色"""
    up_html = html_metric("涨", "10.0", chg=1.2)
    down_html = html_metric("跌", "9.0", chg=-1.2)
    flat_html = html_metric("平", "10.0")
    assert 'mini-card up' in up_html and "#dc2626" in up_html
    assert 'mini-card down' in down_html and "#059669" in down_html
    assert 'mini-card plain' in flat_html


def test_empty_block_renders_container():
    import src.ui.common as common
    calls = []
    orig = common.st.markdown
    try:
        common.st.markdown = lambda body, **kw: calls.append(body)
        empty_block("暂无数据", icon="📭", hint="点上方按钮加载")
    finally:
        common.st.markdown = orig
    assert len(calls) == 1
    assert 'class="empty"' in calls[0] and "暂无数据" in calls[0]


# ==================== 个股分析页结构回归 ====================

def test_chart_box_is_heading_not_card():
    """钉死「框中框」bug：.chart-box 只做标题，白卡由 stPlotlyChart 提供。

    历史写法是 st.markdown 开 <div class="chart-box"> + 另一次 st.markdown 关
    </div>，但两次 markdown 是两个独立 DOM 块，浏览器会把未闭合的 div 就地闭合 →
    只有 <h3> 被包住，图表落到盒子外面，最终呈现「标题一张卡 + 图一张卡」。
    """
    css = _theme_css()
    start = css.index(".chart-box{")
    block = css[start:css.index("}", start)]
    assert "background:transparent" in block.replace(" ", ""), \
        ".chart-box 又开始自己画白卡了 —— 会和 stPlotlyChart 的白卡叠成框中框"
    # 图表真正的白卡必须还在
    assert 'div[data-testid="stPlotlyChart"]' in css
    assert "background:var(--surface)" in css


def test_analysis_page_uses_dashboard_components():
    """个股分析页必须真的用上新版式组件，防止重构时退回裸文本。"""
    src = APP_PY.read_text(encoding="utf-8-sig")
    for token in ['class="stock-head"', 'class="score"', 'class="ring"',
                  'class="kv"', "_ana_bar(", "_ana_kv(", "st.tabs(", "mini-table"]:
        assert token in src, f"个股分析页缺少新版式组件: {token}"


def test_benchmark_call_matches_signature():
    """_cached_benchmark 的形参是 start，调用侧曾传 start_date → 相对强弱整段静默报错。"""
    tree = ast.parse(APP_PY.read_text(encoding="utf-8-sig"))
    bad = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_cached_benchmark"):
            for kw in node.keywords:
                if kw.arg not in ("index_code", "start"):
                    bad.append((node.lineno, kw.arg))
    assert not bad, f"_cached_benchmark 收到未知关键字参数: {bad}"

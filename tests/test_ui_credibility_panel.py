r"""看板「策略信誉快照加载」逻辑测试。

为什么用 AST 抽取而不是直接 import
----------------------------------
``src/ui/app.py`` 是 Streamlit 顶层脚本，模块级就会拉起调度器等重依赖，直接
import 有副作用（本仓其它 UI 用例也是这么处理的）。而 ``load_credibility_snapshot``
必须留在 app.py 内（移出去会因冻结包构建时没打包该模块而 ModuleNotFoundError），
所以只对它做 AST 切片 + exec。

为什么值得测
------------
信誉引擎的权重表现看板此前从未展示。落盘文件 ``data/credibility_state.json`` 由
交易闭环 guard 在每 tick 写，看板只读。这个函数负责把文件读成看板能用的结构，
任何失败都必须退化成 ``{}``（让上层显示空状态），绝不能让整个页面 500。
"""

from __future__ import annotations

import ast
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

_WANT = ("load_credibility_snapshot",)


def _load_fragment():
    src = (ROOT / "src" / "ui" / "app.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _WANT]
    missing = set(_WANT) - {n.name for n in nodes}
    assert not missing, f"app.py 里找不到 {sorted(missing)}（改名了？同步更新本测试）"
    ns = {"Path": Path, "json": json, "datetime": datetime, "logger": None}
    frag = types.ModuleType("app_cred_fragment")
    frag.__dict__.update(ns)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<app-frag>", "exec"),
         frag.__dict__)
    return frag


frag = _load_fragment()


# ---------------------------------------------------------------------------
# 用假 st 做「渲染不抛、结构正确」的确定性校验（不依赖浏览器）
# ---------------------------------------------------------------------------
class _FakeCol:
    def __init__(self):
        self.metrics = []

    def metric(self, *a, **k):
        self.metrics.append(a)


class _FakeSt:
    def __init__(self):
        self.markdowns = []
        self.captions = []
        self.cols = []

    def markdown(self, s, *a, **k):
        self.markdowns.append(s)

    def caption(self, s, *a, **k):
        self.captions.append(s)

    def columns(self, n):
        cols = [_FakeCol() for _ in range(n)]
        self.cols.append(cols)
        return cols


def _render_fragment(tmp_path, monkeypatch):
    """抽取 load + render 两个函数，注入假 st / 假 settings，返回 (frag, fake_st)。"""
    _fake_settings(tmp_path, monkeypatch)
    src = (ROOT / "src" / "ui" / "app.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in
             ("load_credibility_snapshot", "render_credibility_panel")]
    fake_st = _FakeSt()
    ns = {"Path": Path, "json": json, "datetime": datetime, "st": fake_st, "logger": None}
    mod = types.ModuleType("app_cred_render_fragment")
    mod.__dict__.update(ns)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<app-render-frag>", "exec"),
         mod.__dict__)
    return mod, fake_st


def test_render_empty_state_shows_no_data(tmp_path, monkeypatch):
    mod, fake_st = _render_fragment(tmp_path, monkeypatch)
    mod.render_credibility_panel()  # DATA_DIR 下无文件
    joined = "\n".join(fake_st.markdowns)
    assert "策略信誉引擎" in joined
    assert "尚无已平仓战绩" in joined


def test_render_populated_shows_tags_and_metrics(tmp_path, monkeypatch):
    mod, fake_st = _render_fragment(tmp_path, monkeypatch)
    _write_snapshot(tmp_path / "credibility_state.json", {
        "rebalance": {"score": 0.72, "weight": 1.22, "n": 13},
        "stop": {"score": 0.41, "weight": 0.82, "n": 8},
    })
    mod.render_credibility_panel()
    joined = "\n".join(fake_st.markdowns)
    # 概览统计走 st.columns().metric，不在 markdown 里——单独收集
    metric_text = "\n".join(
        str(m) for group in fake_st.cols for col in group for m in col.metrics
    )
    # 概览统计 + 每标签卡片
    assert "已观测策略" in metric_text
    assert "权重↑被信任" in metric_text
    assert "rebalance" in joined
    assert "stop" in joined
    # 概览统计：4 列一组，至少一组含 4 个 metric
    assert any(len(group) == 4 for group in fake_st.cols)
    # 新鲜度提示应出现（文件带 updated_at）
    assert any("快照更新于" in c for c in fake_st.captions)


def _fake_settings(data_dir, monkeypatch):
    mod = types.ModuleType("config.settings")
    mod.DATA_DIR = str(data_dir)
    monkeypatch.setitem(sys.modules, "config.settings", mod)


def _write_snapshot(path: Path, scores: dict):
    path.write_text(json.dumps({
        "halflife": 20, "floor": 0.5, "cap": 1.5,
        "updated_at": datetime(2026, 10, 26, 12, 0, 0).isoformat(timespec="seconds"),
        "scores": scores,
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# 空 / 缺失 / 损坏
# ---------------------------------------------------------------------------

def test_missing_file_returns_empty(tmp_path, monkeypatch):
    _fake_settings(tmp_path, monkeypatch)
    assert frag.load_credibility_snapshot() == {}


def test_corrupt_file_returns_empty(tmp_path, monkeypatch):
    _fake_settings(tmp_path, monkeypatch)
    (tmp_path / "credibility_state.json").write_text("{ bad json", encoding="utf-8")
    assert frag.load_credibility_snapshot() == {}


def test_scores_not_dict_returns_empty(tmp_path, monkeypatch):
    _fake_settings(tmp_path, monkeypatch)
    (tmp_path / "credibility_state.json").write_text(
        json.dumps({"scores": "not-a-dict"}), encoding="utf-8")
    assert frag.load_credibility_snapshot() == {}


def test_empty_scores_returns_empty(tmp_path, monkeypatch):
    _fake_settings(tmp_path, monkeypatch)
    _write_snapshot(tmp_path / "credibility_state.json", {})
    assert frag.load_credibility_snapshot() == {}


# ---------------------------------------------------------------------------
# 正常解析
# ---------------------------------------------------------------------------

def test_parses_scores_and_meta(tmp_path, monkeypatch):
    _fake_settings(tmp_path, monkeypatch)
    _write_snapshot(tmp_path / "credibility_state.json", {
        "momentum": {"score": 0.72, "weight": 1.22, "n": 13},
        "defensive": {"score": 0.41, "weight": 0.82, "n": 8},
    })
    snap = frag.load_credibility_snapshot()
    assert len(snap["scores"]) == 2
    assert snap["scores"]["momentum"]["weight"] == pytest.approx(1.22)
    assert snap["scores"]["defensive"]["n"] == 8
    assert snap["halflife"] == 20
    assert snap["updated_at"].startswith("2026-10-26T12:00:00")


def test_path_resolution_falls_back_when_settings_missing(tmp_path, monkeypatch):
    """config.settings 拿不到时退回「<__file__ 项目根>/data」，找不到文件返回 {} 且不抛。"""
    monkeypatch.setitem(sys.modules, "config.settings", None)
    # 模拟 app.py 的 __file__，让退回路径落在 tmp/data 下（该文件不存在）
    frag.__dict__["__file__"] = str(tmp_path / "src" / "ui" / "app.py")
    snap = frag.load_credibility_snapshot()
    assert snap == {}

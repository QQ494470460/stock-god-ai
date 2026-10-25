r"""看板「心跳判读」逻辑测试。

为什么用 AST 抽取而不是直接 import
----------------------------------
``src/ui/app.py`` 是 Streamlit 顶层脚本，模块级就会拉起调度器等重依赖，
直接 import 有副作用（本仓其它用例也是这么处理的）。而这些函数**必须留在
app.py 内** —— 移出去会因冻结包（pyinstaller 构建时没打包该模块）而
ModuleNotFoundError。所以只对这两个函数做 AST 切片 + exec。

为什么值得测
------------
1. 闭环心跳阈值曾经**固定 30 分钟**：交易日心跳是 30 秒一次，死了要半小时
   才发现；而周末闭环本就降到 10 分钟一次，用 5 分钟阈值又会天天误报。
   误报比漏报更危险 —— 会让人习惯性忽略告警条。
2. 常驻调度进程的心跳必须用文件里的 ``ts`` 算年龄，**不能用文件 mtime**：
   进程被杀时文件会保留最后内容，mtime 也停在那一刻，看上去"刚刚还活着"。
"""

from __future__ import annotations

import ast
import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

_WANT = ("_loop_stale_threshold_min", "_scheduler_daemon_heartbeat")


class _Log:
    def debug(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


def _load_fragment():
    src = (ROOT / "src" / "ui" / "app.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _WANT]
    missing = set(_WANT) - {n.name for n in nodes}
    assert not missing, f"app.py 里找不到 {sorted(missing)}（改名了？同步更新本测试）"
    ns = {"Path": Path, "datetime": datetime, "json": json, "logger": _Log()}
    frag = types.ModuleType("app_heartbeat_fragment")
    frag.__dict__.update(ns)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<app-frag>", "exec"),
         frag.__dict__)
    return frag


frag = _load_fragment()


def _fake_calendar(trading: bool, monkeypatch):
    mod = types.ModuleType("src.trading.calendar")
    mod.is_trading_day = lambda _d: trading
    monkeypatch.setitem(sys.modules, "src.trading.calendar", mod)


def _fake_settings(data_dir, monkeypatch):
    mod = types.ModuleType("config.settings")
    mod.DATA_DIR = str(data_dir)
    monkeypatch.setitem(sys.modules, "config.settings", mod)


# ---------------------------------------------------------------------------
# 闭环心跳阈值：跟随日历
# ---------------------------------------------------------------------------

def test_threshold_is_tight_on_trading_day(monkeypatch):
    _fake_calendar(True, monkeypatch)
    assert frag._loop_stale_threshold_min() == 5.0, \
        "交易日心跳 30s 一次，5 分钟没动静就必须是异常"


def test_threshold_is_loose_on_non_trading_day(monkeypatch):
    """周末闭环降到 10 分钟一次，别用 5 分钟阈值去误报。"""
    _fake_calendar(False, monkeypatch)
    assert frag._loop_stale_threshold_min() == 30.0


def test_threshold_falls_back_to_loose_when_calendar_unavailable(monkeypatch):
    """日历拿不到时宁可漏报，不可误报。"""
    mod = types.ModuleType("src.trading.calendar")
    mod.is_trading_day = lambda _d: (_ for _ in ()).throw(RuntimeError("网络挂了"))
    monkeypatch.setitem(sys.modules, "src.trading.calendar", mod)
    assert frag._loop_stale_threshold_min() == 30.0


def test_threshold_accepts_injected_now(monkeypatch):
    """可注入时间，避免测试依赖真实日期（否则周一和周日跑出来不一样）。"""
    _fake_calendar(True, monkeypatch)
    assert frag._loop_stale_threshold_min(datetime(2026, 10, 26, 10, 0)) == 5.0


# ---------------------------------------------------------------------------
# 常驻调度进程心跳：用 ts 算年龄，不是 mtime
# ---------------------------------------------------------------------------

def test_missing_heartbeat_file_returns_none(tmp_path, monkeypatch):
    _fake_settings(tmp_path, monkeypatch)
    assert frag._scheduler_daemon_heartbeat() is None


def test_fresh_heartbeat_reports_small_age(tmp_path, monkeypatch):
    _fake_settings(tmp_path, monkeypatch)
    (tmp_path / "scheduler_heartbeat.json").write_text(
        json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                    "pid": 1, "holding": True, "jobs": []}),
        encoding="utf-8")
    hb = frag._scheduler_daemon_heartbeat()
    assert hb is not None
    assert 0 <= hb["age_sec"] < 10


def test_stale_heartbeat_uses_ts_not_mtime(tmp_path, monkeypatch):
    """关键：mtime 会撒谎，ts 不会。

    进程被杀时文件保留最后内容、mtime 停在死亡那一刻 —— 看上去"刚刚还活着"。
    这里故意把 mtime 刷成现在，只让内容里的 ts 是 1 小时前，
    必须判出 3600 秒。
    """
    _fake_settings(tmp_path, monkeypatch)
    p = tmp_path / "scheduler_heartbeat.json"
    old = datetime.now() - timedelta(seconds=3600)
    p.write_text(json.dumps({"ts": old.isoformat(timespec="seconds"),
                             "pid": 1, "holding": True, "jobs": []}),
                 encoding="utf-8")
    import os
    os.utime(p, None)                      # mtime 刷成"刚刚"

    hb = frag._scheduler_daemon_heartbeat()
    assert hb["age_sec"] > 3500, "拿 mtime 判断的话这里会误判成还活着"


def test_unparsable_ts_yields_none_age(tmp_path, monkeypatch):
    """时间戳坏了要说"读不出来"，不能谎报"已停 X 秒"。"""
    _fake_settings(tmp_path, monkeypatch)
    (tmp_path / "scheduler_heartbeat.json").write_text(
        json.dumps({"ts": "not-a-time", "pid": 1, "holding": True, "jobs": []}),
        encoding="utf-8")
    assert frag._scheduler_daemon_heartbeat()["age_sec"] is None


def test_broken_file_returns_none(tmp_path, monkeypatch):
    """心跳文件写坏了也不能让整个看板页面 500。"""
    _fake_settings(tmp_path, monkeypatch)
    (tmp_path / "scheduler_heartbeat.json").write_text("{ bad json", encoding="utf-8")
    assert frag._scheduler_daemon_heartbeat() is None


@pytest.mark.parametrize("jobs", [[], [{"id": "daily_pick", "next_run": None}]])
def test_heartbeat_keeps_job_list(tmp_path, monkeypatch, jobs):
    _fake_settings(tmp_path, monkeypatch)
    (tmp_path / "scheduler_heartbeat.json").write_text(
        json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                    "pid": 7, "holding": False, "jobs": jobs}),
        encoding="utf-8")
    hb = frag._scheduler_daemon_heartbeat()
    assert hb["jobs"] == jobs
    assert hb["holding"] is False

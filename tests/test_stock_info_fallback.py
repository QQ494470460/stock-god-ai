# -*- coding: utf-8 -*-
"""个股基本信息「多源兜底」回归测试。

背景（连续两轮真实事故）：
1. 基本面概况恒显示 行业=未知 / 市值=0亿 / 上市日期空 —— 静态股票列表写死桩值。
2. 加了东方财富 `stock_individual_info_em` 后，用户环境该接口不可达
   （`RemoteDisconnected`），而旧实现失败即返回 `_error` → 基本面**整块空白**。
3. 更隐蔽：`stock_zh_a_hist` 名字像新浪，实际是东方财富 push2his —— 拿它当
   「备用源」会在东财不可达时**一起挂掉**，而且被 `except: pass` 静默吞掉。

本测试要钉死的是「**备用源真的能顶上**」这条正向能力，而不只是「不报错」：
- 东财可用   → 行业/总市值/流通市值/上市日期 全有；
- 东财不可用 → 百度给总市值、新浪给上市日期+流通市值，**只有行业缺失**；
- 全挂       → 才置 `_error`。
另外钉死 `_sina_symbol` 前缀映射，以及 `name` 不得回填代码（会冲掉真实名称）。

为什么用 ast 抽取而不是 `from src.ui.app import ...`：
`app.py` 是 Streamlit 顶层脚本，模块级就会拉起调度器等重依赖，
直接 import 有副作用；且这些兜底函数**必须留在 app.py 内**——移到新模块会因
冻结包（pyinstaller 构建时打包）里没有该模块而 ModuleNotFoundError。
"""
import ast
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_PY = ROOT / "src" / "ui" / "app.py"

_WANT = {"_parse_cap_local", "_sina_symbol", "_fetch_stock_info_local"}


def _load_helpers():
    """从 app.py 源码里抽取这三个纯函数（不执行 app.py 其余部分）。"""
    src = APP_PY.read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    segs = [
        ast.get_source_segment(src, n)
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in _WANT
    ]
    assert len(segs) == len(_WANT), f"未抽全兜底函数：{len(segs)}/{len(_WANT)}"
    ns = {"datetime": datetime}
    exec("\n\n".join(segs), ns)
    return ns


@pytest.fixture(scope="module")
def h():
    return _load_helpers()


# --------------------------------------------------------------------------
# 1) 新浪 symbol 前缀映射
# --------------------------------------------------------------------------
def test_sina_symbol_prefix(h):
    f = h["_sina_symbol"]
    assert f("000001") == "sz000001"      # 深主板
    assert f("300750") == "sz300750"      # 创业板
    assert f("600519") == "sh600519"      # 沪主板
    assert f("688001") == "sh688001"      # 科创板（6 开头）
    assert f("830799") == "bj830799"      # 北交所
    assert f("1") == "sz000001"           # 短代码补零


# --------------------------------------------------------------------------
# 2) 市值单位归一（元 / 亿 / 万 / 空）
# --------------------------------------------------------------------------
def test_parse_cap_local(h):
    f = h["_parse_cap_local"]
    assert f(2231.68) == 2231.68               # 已是「亿」
    assert f(2.23168e11) == pytest.approx(2231.68)   # 元量级 → 亿
    assert f("100亿") == 100.0
    assert f("15000万") == pytest.approx(1.5)
    assert f(None) == 0.0
    assert f("") == 0.0
    assert f("abc") == 0.0                     # 非法不炸


# --------------------------------------------------------------------------
# 3) 东财可用 → 字段最全（含行业）
# --------------------------------------------------------------------------
def test_eastmoney_ok_gives_all_fields(h, monkeypatch):
    em = pd.DataFrame([
        {"item": "股票简称", "value": "平安银行"},
        {"item": "行业", "value": "银行"},
        {"item": "总市值", "value": 2.23168e11},
        {"item": "流通市值", "value": 2.23165e11},
        {"item": "上市时间", "value": "1991-04-03"},
    ])

    def fake_em(symbol):
        return em

    def boom(*a, **k):  # 备用源不应被调用
        raise AssertionError("东财成功时不应再走备用源")

    ak = types.ModuleType("akshare")
    ak.stock_individual_info_em = fake_em
    ak.stock_zh_valuation_baidu = boom
    ak.stock_zh_a_daily = boom
    monkeypatch.setitem(sys.modules, "akshare", ak)

    r = h["_fetch_stock_info_local"]("000001")
    assert "_error" not in r
    assert r["name"] == "平安银行"
    assert r["industry"] == "银行"
    assert r["total_cap"] == pytest.approx(2231.68)
    assert r["float_cap"] == pytest.approx(2231.65)
    assert r["listed_date"] == "1991-04-03"
    assert r["_source"] == "东方财富"
    assert r["_missing"] == []


# --------------------------------------------------------------------------
# 4) 关键：东财不可达 → 备用源必须顶上（只有行业缺失）
# --------------------------------------------------------------------------
def test_backup_sources_fill_caps_and_listed_date(h, monkeypatch):
    def em_boom(symbol):
        raise OSError("('Connection aborted.', RemoteDisconnected(...))")

    baidu = pd.DataFrame([{"date": "2026-10-18", "value": 2231.68}])
    sina = pd.DataFrame([
        {"date": "1991-04-03", "close": 49.0, "outstanding_share": 3.71e7},
        {"date": "2026-10-18", "close": 11.5, "outstanding_share": 1.9406e10},
    ])

    ak = types.ModuleType("akshare")
    ak.stock_individual_info_em = em_boom
    ak.stock_zh_valuation_baidu = lambda **k: baidu
    ak.stock_zh_a_daily = lambda **k: sina
    monkeypatch.setitem(sys.modules, "akshare", ak)

    r = h["_fetch_stock_info_local"]("000001")
    assert "_error" not in r, "备用源可用时不应报整体失败"
    # 正向断言：三个字段真的被补上了
    assert r["total_cap"] == pytest.approx(2231.68)              # 百度
    assert r["listed_date"] == "1991-04-03"                      # 新浪首日
    assert r["float_cap"] == pytest.approx(2231.65, abs=1.0)     # 股本 × 收盘
    # 仅行业缺失（离线确实取不到），且被标注出来
    assert r["_missing"] == ["行业"]
    assert "新浪" in r["_source"] and "百度" in r["_source"]
    # name 不能回填代码，否则会冲掉静态列表里的真实名称
    assert r["name"] == ""


# --------------------------------------------------------------------------
# 5) 全部数据源都挂 → 才置 _error（UI 才会告警）
# --------------------------------------------------------------------------
def test_all_sources_down_sets_error(h, monkeypatch):
    def boom(*a, **k):
        raise OSError("network down")

    ak = types.ModuleType("akshare")
    ak.stock_individual_info_em = boom
    ak.stock_zh_valuation_baidu = boom
    ak.stock_zh_a_daily = boom
    monkeypatch.setitem(sys.modules, "akshare", ak)

    r = h["_fetch_stock_info_local"]("000001")
    assert "_error" in r
    assert "network down" in r["_error"]


# --------------------------------------------------------------------------
# 6) 防复发：不得再把 stock_zh_a_hist 当「新浪备用源」
#    （它在 akshare 1.18 实为东方财富 push2his，东财挂时它必挂）
# --------------------------------------------------------------------------
def test_must_not_use_stock_zh_a_hist_as_sina_fallback():
    """只拦「真实调用」，允许注释/docstring 里解释性提及该函数名。"""
    src = APP_PY.read_text(encoding="utf-8-sig")
    assert "stock_zh_a_hist(" not in src, (
        "stock_zh_a_hist 不是新浪接口（实为东方财富 push2his），"
        "东财不可达时不能当备用源；请改用 stock_zh_a_daily"
    )
    assert "stock_zh_a_daily(" in src

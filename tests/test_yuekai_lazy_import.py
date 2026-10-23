# -*- coding: utf-8 -*-
"""回归测试：xtquant 必须惰性、容错加载，绝不在模拟/纸交易模式下被 eager import。

背景（2026-10 冷启动事故）：
    yuekai.py 曾在模块顶层无条件 import 原生 xtquant DLL。纸交易（默认）根本用不到它，
    却仍会在冷启动时加载该原生扩展；一旦 QMT 未就绪导致 DLL 初始化异常，进程以原生退出码
    (-1 / 0x40010004) 无声崩溃，没有任何 Python traceback，闭环整天没跑。

这些测试锁定「模拟模式不碰 xtquant」+「加载失败降级为模拟而非崩溃」两个不变量。
"""
import builtins
import sys
import unittest.mock as mock

import pytest

import src.trading.yuekai as yk
from src.trading.yuekai import YuekaiBroker


@pytest.fixture(autouse=True)
def _reset_xt_state():
    """每个用例前后复位 xtquant 全局状态，避免用例间串扰。"""
    saved = (yk.XT_AVAILABLE, yk._xt_import_error, yk.XtQuantTrader,
             yk.XtQuantTraderCallback, yk.StockAccount, yk.xtconstant)
    yk.XT_AVAILABLE = False
    yk._xt_import_error = None
    yk.XtQuantTrader = None
    yk.XtQuantTraderCallback = None
    yk.StockAccount = None
    yk.xtconstant = None
    yield
    (yk.XT_AVAILABLE, yk._xt_import_error, yk.XtQuantTrader,
     yk.XtQuantTraderCallback, yk.StockAccount, yk.xtconstant) = saved


def test_xtquant_not_eagerly_loaded_in_paper_mode():
    """导入 yuekai 后，默认（纸交易）模式下不得已经加载 xtquant。"""
    assert yk.XT_AVAILABLE is False
    assert yk.XtQuantTrader is None


def test_sim_connect_never_touches_xtquant():
    """sim 模式连接成功，且全程不设置任何 xtquant 句柄。"""
    broker = YuekaiBroker()
    ok = broker.connect(mode="sim")
    assert ok is True
    assert broker._simulated is True
    assert yk.XT_AVAILABLE is False  # 仍未加载


def test_ensure_xtquant_success_when_installed():
    """本机已装 xtquant 时，ensure_xtquant() 成功并刷新真实常量映射。"""
    # 若本机未安装 xtquant，该用例直接跳过（不在 CI 强依赖）。
    try:
        import xtquant  # noqa: F401
    except ImportError:
        pytest.skip("xtquant 未安装，跳过实装路径测试")
    ok = yk.ensure_xtquant()
    assert ok is True
    assert yk.XT_AVAILABLE is True
    # 真实 xtconstant 值：FIX_PRICE=11, MARKET_SH_CONVERT_5_CANCEL=42
    assert yk.ORDER_TYPE_MAP[yk.OrderType.LIMIT] == 11
    assert yk.ORDER_TYPE_MAP[yk.OrderType.MARKET] == 42


def test_connect_degrades_to_sim_when_xt_unavailable_auto():
    """auto 模式且 xtquant 不可用 → 自动降级为模拟，不崩溃、不返回 None。"""
    with mock.patch.object(yk, "ensure_xtquant", return_value=False) as m:
        yk._xt_import_error = "forced unavailable"
        broker = YuekaiBroker()
        ok = broker.connect(mode="auto")
    assert m.called
    assert ok is True
    assert broker._simulated is True


def test_connect_refuses_live_when_xt_unavailable():
    """live 模式且 xtquant 不可用 → 干净返回 False（不崩溃、不降级）。"""
    with mock.patch.object(yk, "ensure_xtquant", return_value=False) as m:
        yk._xt_import_error = "forced unavailable"
        broker = YuekaiBroker()
        ok = broker.connect(mode="live")
    assert m.called
    assert ok is False
    assert broker._simulated is False


def test_ensure_xtquant_survives_import_error():
    """xtquant import 抛 ImportError 时，ensure_xtquant 必须捕获并返回 False，
    绝不把原生/导入异常冒泡到调用方（这正是冷启动崩溃的根因防护）。"""
    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("xtquant"):
            raise ImportError(f"No module named {name}")
        return orig_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=_fake_import):
        result = yk.ensure_xtquant()
    assert result is False
    assert yk.XT_AVAILABLE is False
    assert yk._xt_import_error is not None


def test_ensure_xtquant_survives_native_like_failure():
    """即使 import 抛非 ImportError 的异常（模拟 DLL 初始化失败类错误），
    ensure_xtquant 内部的 `except Exception` 也必须捕获并返回 False，
    绝不把异常冒泡成原生崩溃。"""
    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("xtquant"):
            raise RuntimeError("simulated DLL init failure")
        return orig_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=_fake_import):
        result = yk.ensure_xtquant()
    assert result is False
    assert yk.XT_AVAILABLE is False
    assert yk._xt_import_error is not None

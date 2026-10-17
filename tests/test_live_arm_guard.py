# -*- coding: utf-8 -*-
"""实盘武装前的券商真伪校验（fail-closed）。

背景（静默失效）：``is_simulated`` 一直只在看板上显示，guard / loop /
executor 谁都没读它。于是「武装成 LIVE、界面写着实盘、单子却全进了模拟通道」
是可能的 —— 用户以为在拿真钱跑，实际什么都没发生，而且没有任何报错。

修法：武装实盘（arm_live / confirm_arm）前必须确认券商不是模拟连接；
属性缺失一律按模拟处理（宁可拒绝，也不给假象）。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.trading.guard import GuardConfig, SafetyController, TradingMode


def _cfg(tmp_path):
    return GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
        require_live_confirmation=True,
    )


class _SimBroker:
    is_simulated = True
    sim_reason = "用户选择模拟模式"


class _LiveBroker:
    is_simulated = False


class _NoFlagBroker:
    """什么标记都没有的券商实现（老代码 / 第三方实现）。"""


def test_arm_live_refuses_simulated_broker(tmp_path):
    ctl = SafetyController(_cfg(tmp_path), broker=_SimBroker())
    with pytest.raises(ValueError) as ei:
        ctl.arm_live("alice")
    assert "拒绝武装实盘" in str(ei.value)
    assert "模拟" in str(ei.value)
    assert ctl.mode == TradingMode.PAPER
    assert not ctl._state.pending_token, "被拒绝时不该发出令牌"


def test_arm_live_refuses_when_flag_missing(tmp_path):
    """缺 is_simulated 属性 → 按模拟处理（fail-closed），不给「看起来像实盘」的假象。"""
    ctl = SafetyController(_cfg(tmp_path), broker=_NoFlagBroker())
    with pytest.raises(ValueError):
        ctl.arm_live("alice")
    assert ctl.mode == TradingMode.PAPER


def test_arm_live_refuses_without_any_broker(tmp_path):
    ctl = SafetyController(_cfg(tmp_path), broker=None)
    with pytest.raises(ValueError):
        ctl.arm_live("alice")


def test_arm_live_allows_real_broker(tmp_path):
    ctl = SafetyController(_cfg(tmp_path), broker=_LiveBroker())
    token = ctl.arm_live("alice")
    assert len(token) == 8
    assert ctl.mode == TradingMode.PAPER          # 第一步仍未切换
    assert ctl.confirm_arm(token, "alice") is True
    assert ctl.mode == TradingMode.LIVE_PENDING


def test_confirm_arm_rechecks_broker(tmp_path):
    """令牌有效期里券商可能被重连成模拟 —— 第二步必须再校验一次。"""
    broker = _LiveBroker()
    ctl = SafetyController(_cfg(tmp_path), broker=broker)
    token = ctl.arm_live("alice")

    broker.is_simulated = True                    # 中途断线降级成模拟
    assert ctl.confirm_arm(token, "alice") is False
    assert ctl.mode == TradingMode.PAPER, "券商是模拟时绝不能进实盘模式"

# -*- coding: utf-8 -*-
"""熔断净值基线随「策略形状」重置。

背景（真实死锁，2026-10-31）：

    回撤熔断按「峰值 → 当前权益」算，而峰值是**旧策略**打出来的成绩。
    10-31 用执行级回测推翻了整套个股退出规则（止损/止盈/移动止损全关、
    调仓周期 5→15 天）。改造完成后账户仍是 10.86% 回撤、解冻线 6%
    要求回撤降到 6% 以下，而 92.7% 是现金、只剩 1 只 7.3% 的持仓 ——
    靠自己根本回不到解冻线，新规则一行代码都执行不到。

修法：把「策略形状指纹」记进风控状态；指纹变了就把净值基线对齐当前权益。
三条 fail-safe 必须都在（否则等于给自己发免死金牌）：

    1. 指纹为空 / 权益非法 → 不动；
    2. 旧状态文件没有该字段（空串）→ **只登记不重置**。首次引入本机制
       就静默放宽一次熔断是不可接受的，那一次必须由操作者显式
       ``--rebaseline-peak``；
    3. 指纹相同 → 不动。

另外指纹**只含改变收益分布的参数**，不含风控阈值 —— 调 ``max_portfolio_drawdown_pct``
不该触发基线重置，否则「调风控」本身就变成了「放宽风控」。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.trading.guard import GuardConfig, SafetyController
from src.trading.loop import LoopConfig, TradingLoop


def _cfg(tmp_path):
    return GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
    )


# ----------------------------------------------------------------------
# SafetyController.sync_strategy_version
# ----------------------------------------------------------------------
def test_first_sight_only_registers_and_never_resets_peak(tmp_path):
    """旧状态文件没有版本字段 ⇒ 只登记，绝不动峰值。"""
    ctl = SafetyController(_cfg(tmp_path))
    ctl.update_equity(1_000_000.0)

    changed, why = ctl.sync_strategy_version("abc123", 900_000.0)

    assert changed is False
    assert "首见" in why
    assert ctl._state.peak_equity == 1_000_000.0, "首见不得重置峰值"
    assert ctl._state.strategy_version == "abc123", "但必须把版本登记下来"


def test_version_change_rebaselines_peak_and_clears_drawdown(tmp_path):
    ctl = SafetyController(_cfg(tmp_path))
    ctl.update_equity(1_000_000.0)
    ctl.sync_strategy_version("v-old", 900_000.0)      # 首见：只登记
    assert ctl.current_drawdown_pct(900_000.0) > 9.9

    changed, why = ctl.sync_strategy_version("v-new", 900_000.0)

    assert changed is True
    assert "v-old" in why and "v-new" in why
    assert ctl._state.peak_equity == 900_000.0
    assert ctl.current_drawdown_pct(900_000.0) == 0.0, "基线对齐后回撤必须归零"
    assert ctl._state.strategy_version == "v-new"


def test_same_version_is_noop(tmp_path):
    ctl = SafetyController(_cfg(tmp_path))
    ctl.update_equity(1_000_000.0)
    ctl.sync_strategy_version("v1", 900_000.0)
    ctl.sync_strategy_version("v1", 900_000.0)

    changed, why = ctl.sync_strategy_version("v1", 900_000.0)

    assert changed is False and "未变" in why
    assert ctl._state.peak_equity == 1_000_000.0


def test_empty_version_or_bad_equity_is_refused(tmp_path):
    ctl = SafetyController(_cfg(tmp_path))
    ctl.update_equity(1_000_000.0)

    assert ctl.sync_strategy_version("", 900_000.0)[0] is False
    assert ctl.sync_strategy_version("  ", 900_000.0)[0] is False
    assert ctl.sync_strategy_version("v1", 0)[0] is False
    assert ctl.sync_strategy_version("v1", -1)[0] is False
    assert ctl.sync_strategy_version("v1", "not-a-number")[0] is False

    assert ctl._state.peak_equity == 1_000_000.0
    assert ctl._state.strategy_version == "", "被拒绝时不该留下版本"


def test_rebaseline_survives_reload(tmp_path):
    """版本与峰值都要落盘，否则常驻进程重启后又会拿旧峰值判熔断。"""
    cfg = _cfg(tmp_path)
    ctl = SafetyController(cfg)
    ctl.update_equity(1_000_000.0)
    ctl.sync_strategy_version("v-old", 900_000.0)
    ctl.sync_strategy_version("v-new", 900_000.0)

    again = SafetyController(cfg)

    assert again._state.strategy_version == "v-new"
    assert again._state.peak_equity == 900_000.0


# ----------------------------------------------------------------------
# TradingLoop._strategy_fingerprint
# ----------------------------------------------------------------------
def _fp(loop_overrides=None, guard_overrides=None) -> str:
    """不构造完整 TradingLoop（要 broker/账户），只喂两个配置对象。"""
    obj = object.__new__(TradingLoop)
    obj.config = LoopConfig(**(loop_overrides or {}))
    if guard_overrides is None:
        obj.guard = None
    else:
        obj.guard = SimpleNamespace(config=SimpleNamespace(**guard_overrides))
    return obj._strategy_fingerprint()


def test_fingerprint_reacts_to_strategy_shape():
    base = _fp({"rebalance_days": 15})
    assert base != _fp({"rebalance_days": 5}), "周期是决定成本敏感性的参数，必须进指纹"
    assert base != _fp({"rebalance_days": 15, "band_pct": 0.10})
    assert base != _fp({"rebalance_days": 15, "max_positions": 5})
    assert base != _fp({"rebalance_days": 15, "weighting": "equal"})
    assert base == _fp({"rebalance_days": 15}), "同参数必须稳定"


def test_fingerprint_reacts_to_exit_rules():
    """止损/止盈/移动止损/ATR 开关必须进指纹 —— 10-31 改的就是这几条。"""
    base = _fp({"rebalance_days": 15},
               {"stop_loss_pct": 0.0, "stop_profit_pct": 0.0,
                "trailing_stop_pct": 0.0, "atr_stop_enabled": False})
    assert base != _fp({"rebalance_days": 15},
                       {"stop_loss_pct": -5.0, "stop_profit_pct": 0.0,
                        "trailing_stop_pct": 0.0, "atr_stop_enabled": False})
    assert base != _fp({"rebalance_days": 15},
                       {"stop_loss_pct": 0.0, "stop_profit_pct": 15.0,
                        "trailing_stop_pct": 0.0, "atr_stop_enabled": False})
    assert base != _fp({"rebalance_days": 15},
                       {"stop_loss_pct": 0.0, "stop_profit_pct": 0.0,
                        "trailing_stop_pct": -3.0, "atr_stop_enabled": False})
    assert base != _fp({"rebalance_days": 15},
                       {"stop_loss_pct": 0.0, "stop_profit_pct": 0.0,
                        "trailing_stop_pct": 0.0, "atr_stop_enabled": True})


def test_fingerprint_ignores_risk_limits():
    """风控阈值不进指纹：否则「调熔断线」就等于「重置熔断基线」。"""
    a = _fp({"rebalance_days": 15},
            {"stop_loss_pct": 0.0, "stop_profit_pct": 0.0, "trailing_stop_pct": 0.0,
             "atr_stop_enabled": False, "max_portfolio_drawdown_pct": 8.0})
    b = _fp({"rebalance_days": 15},
            {"stop_loss_pct": 0.0, "stop_profit_pct": 0.0, "trailing_stop_pct": 0.0,
             "atr_stop_enabled": False, "max_portfolio_drawdown_pct": 5.0})
    assert a == b

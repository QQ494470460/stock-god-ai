# -*- coding: utf-8 -*-
"""止损参数从配置贯通到执行器 + 单日亏损两条线的耦合告警。

两个真实缺陷：

1. **止损参数改了配置不生效**：``RiskSupervisor`` 一直是
   ``TradeExecutor(broker, risk_config)``，而 ``risk_config`` 传的是 None，
   于是永远用 ``RiskConfig()`` 的硬编码默认值 —— ``trading_guard.yaml`` 里
   写什么止损线都没用。现在 None 时会按 GuardConfig 的止损段构造。

2. **单日亏损「金额线」比「比例线」更紧，把比例线变成死配置**：
   ``max_daily_loss_abs_pct: 1.0`` 配 ``max_daily_loss_pct: 3.0``，
   两条线是「或」，实际阈值就是 1%。2026-10-26 实测每天 -1% 就熔断，
   当天 -2.39% 触发后整个下午不再交易。这类配置冲突必须**显式告警**，
   否则永远靠人肉发现（日志里只会说「金额线超限」）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from loguru import logger

from src.trading.broker import AccountInfo
from src.trading.guard import GuardConfig, RiskSupervisor


class _Broker:
    """RiskSupervisor 只需要 broker 能查账户（构造 TradeExecutor 时会调用）。"""

    is_simulated = False

    def query_account(self):
        return AccountInfo(account_id="T", total_asset=1_000_000.0,
                           available_cash=1_000_000.0, frozen_cash=0.0,
                           market_value=0.0, total_pnl=0.0, today_pnl=0.0)


# ----------------------------------------------------------------------
# 配置 → 执行器
# ----------------------------------------------------------------------
def test_yaml_stop_section_is_parsed(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(
        "stop_loss_pct: -7.5\n"
        "stop_profit_pct: 22.0\n"
        "trailing_stop_pct: -4.0\n"
        "atr_stop_enabled: true\n"
        "atr_stop_k: 2.5\n"
        "atr_stop_max_pct: 12.0\n",
        encoding="utf-8",
    )
    cfg = GuardConfig.from_yaml(str(p))
    assert cfg.stop_loss_pct == -7.5
    assert cfg.stop_profit_pct == 22.0
    assert cfg.trailing_stop_pct == -4.0
    assert cfg.atr_stop_enabled is True
    assert cfg.atr_stop_k == 2.5
    assert cfg.atr_stop_max_pct == 12.0


def test_supervisor_builds_risk_config_from_guard_config(tmp_path):
    """risk_config=None 时必须用配置里的止损，而不是 RiskConfig 的默认值。"""
    cfg = GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
        stop_loss_pct=-9.0,
        stop_profit_pct=25.0,
        trailing_stop_pct=-4.5,
        atr_stop_enabled=True,
        atr_stop_k=3.0,
    )
    sup = RiskSupervisor(_Broker(), cfg, industry_map={})
    rc = sup.executor.config
    assert rc.stop_loss_pct == -9.0
    assert rc.stop_profit_pct == 25.0
    assert rc.trailing_stop_pct == -4.5
    assert rc.atr_stop_enabled is True
    assert rc.atr_stop_k == 3.0
    # 与风控主管自身的配置对象保持同一个引用（改一处两处一致）
    assert sup.risk_config is rc


def test_explicit_risk_config_wins(tmp_path):
    """显式传入的 RiskConfig 不被覆盖（测试/多实例场景要用自己的口径）。"""
    from src.trading.executor import RiskConfig

    cfg = GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
        stop_loss_pct=-9.0,
    )
    mine = RiskConfig(stop_loss_pct=-2.0)
    sup = RiskSupervisor(_Broker(), cfg, industry_map={}, risk_config=mine)
    assert sup.executor.config.stop_loss_pct == -2.0


# ----------------------------------------------------------------------
# 单日亏损两条线
# ----------------------------------------------------------------------
def _cfg(tmp_path, **kw):
    return GuardConfig(
        state_file=str(tmp_path / "s.json"),
        journal_file=str(tmp_path / "j.jsonl"),
        **kw,
    )


def test_abs_line_derives_from_equity(tmp_path):
    cfg = _cfg(tmp_path, max_daily_loss_abs=0.0, max_daily_loss_abs_pct=3.0)
    assert cfg.daily_loss_abs_limit(1_000_000.0) == pytest.approx(30_000.0)


def test_abs_line_uses_fixed_as_floor(tmp_path):
    cfg = _cfg(tmp_path, max_daily_loss_abs=50_000.0, max_daily_loss_abs_pct=1.0)
    # 1% × 100 万 = 1 万 < 固定 5 万 → 取固定值当下限
    assert cfg.daily_loss_abs_limit(1_000_000.0) == pytest.approx(50_000.0)


def test_no_lines_means_infinite(tmp_path):
    cfg = _cfg(tmp_path, max_daily_loss_abs=0.0, max_daily_loss_abs_pct=0.0)
    assert cfg.daily_loss_abs_limit(1_000_000.0) == float("inf")


def _capture_loguru():
    msgs: list = []
    sink_id = logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
    return msgs, sink_id


def test_warns_when_abs_line_tighter_than_pct_line(tmp_path):
    """loguru 不走标准 logging，caplog 抓不到 —— 直接挂一个 sink。"""
    cfg = _cfg(tmp_path, max_daily_loss_pct=3.0, max_daily_loss_abs=0.0,
               max_daily_loss_abs_pct=1.0)
    msgs, sink_id = _capture_loguru()
    try:
        cfg.daily_loss_abs_limit(1_000_000.0)
    finally:
        logger.remove(sink_id)
    assert any("更紧" in m for m in msgs)

    # 只告警一次，不刷屏（每 tick 都调这个方法）
    msgs2, sink2 = _capture_loguru()
    try:
        cfg.daily_loss_abs_limit(1_000_000.0)
    finally:
        logger.remove(sink2)
    assert not any("更紧" in m for m in msgs2)


def test_no_warning_when_aligned(tmp_path):
    cfg = _cfg(tmp_path, max_daily_loss_pct=3.0, max_daily_loss_abs=0.0,
               max_daily_loss_abs_pct=3.0)
    msgs, sink_id = _capture_loguru()
    try:
        cfg.daily_loss_abs_limit(1_000_000.0)
    finally:
        logger.remove(sink_id)
    assert not any("更紧" in m for m in msgs)

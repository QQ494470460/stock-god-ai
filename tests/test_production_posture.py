# -*- coding: utf-8 -*-
"""生产风控姿态快照 —— 一处改错就让整套「实盘对齐回测」的姿态破功。

为什么要单独一个文件
--------------------
2026-10-31 起，实盘的风控姿态被系统地改造过三轮（14 处「实盘有、回测没有」），
改造方向是一致的：**把「越跌越轻」的顺周期闸门全部关掉，只留组合层的硬边界**。
这些结论散落在 yaml 注释、memory 和各模块的测试里。任何一处被无声改回去，
都不会报错 —— 只会表现为「实盘又跑不过回测了」，而那时已经很难定位。

所以这里把**整套姿态**连同**推导出来的不变量**一起钉住。重点不是记住数字，
而是记住数字之间的关系：很多坑不是某个值错了，而是两个值互相矛盾
（例：风险预算上限 < 权重上限 ⇒ 每笔买入都被悄悄砍小，计划永远建不满）。

改这里之前，请先回答：证据在哪？相位平均过没有？三档成本过没有？
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import AppConfig  # noqa: E402
from src.trading.guard import GuardConfig  # noqa: E402

GUARD_YAML = ROOT / "config" / "trading_guard.yaml"


@pytest.fixture(scope="module")
def g() -> dict:
    return yaml.safe_load(GUARD_YAML.read_text(encoding="utf-8")) or {}


@pytest.fixture(scope="module")
def loop():
    return AppConfig().trading_loop


# ---------------------------------------------------------------------------
# 一、个股层全关（15 日周期下退出规则无正期望，风险交组合层）
# ---------------------------------------------------------------------------

def test_stock_level_exits_are_all_disabled(g):
    """止损/止盈/移动止损/ATR 全关。

    1140 天执行级回测（生产口径）结论：止损-3% 移动止损低于日内振幅、每天触发，
    把噪声变成已实现亏损 —— 实盘原状 −27.6%/夏普 −5.33/回撤 −77.2%。
    15 天周期后连 ATR 止损也转负 ⇒ 个股层退出无正期望。
    """
    assert float(g["stop_loss_pct"]) == 0.0
    assert float(g["stop_profit_pct"]) == 0.0
    assert float(g["trailing_stop_pct"]) == 0.0
    assert g["atr_stop_enabled"] is False


def test_loop_exit_hook_is_inert_because_every_line_is_zero(g, loop):
    """交易回路的止损钩子仍开着，但四条退出线全是 0 ⇒ 钩子空转。

    钉住一个容易误判的认知：「enable_stop_orders=True」不等于「会止损交易」，
    真正决定的是各条线是否为 0（0 = 关闭的三守卫见
    ``test_guard_stop_config.test_zero_threshold_disables_exit_rules``）。
    别因为这个开关是 True 就以为个股止损还活着。
    """
    assert loop.enable_stop_orders is True
    for key in ("stop_loss_pct", "stop_profit_pct", "trailing_stop_pct"):
        assert float(g[key]) == 0.0, f"{key} 非 0 ⇒ 个股层退出被重新打开"
    assert g["atr_stop_enabled"] is False


# ---------------------------------------------------------------------------
# 二、组合层硬边界（保留的只有这些）
# ---------------------------------------------------------------------------

def test_portfolio_level_limits(g):
    assert float(g["max_portfolio_drawdown_pct"]) == 20.0, "20% 是策略自身回撤带的上沿"
    assert float(g["max_daily_loss_pct"]) == 5.0, "3% 在噪声带内：年化 −2.7pp 且回撤更深"
    assert float(g["max_sector_pct"]) == 0.50
    assert float(g["max_single_position_pct"]) == 0.30


def test_daily_loss_line_matches_abs_line(g):
    """比例线与金额线必须同位置，否则紧的那条会遮蔽另一条。

    2026-10-26 原状：max_daily_loss_abs=1% 而比例线 3% ⇒ 每天 −1% 就熔断，
    3% 那条一次都没机会生效，闭环下午全部静默。
    """
    assert float(g["max_daily_loss_abs_pct"]) == float(g["max_daily_loss_pct"]) or \
        float(g["max_daily_loss_abs"]) == 0.0


# ---------------------------------------------------------------------------
# 三、四类运行期缩放全关（都不在策略指纹里，关掉不触发熔断基线重置）
# ---------------------------------------------------------------------------

def test_runtime_scalers_are_all_off(g):
    """regime 加权 / Vol-Control / VaR 缩放 / credibility 在线调权。

    它们全是「市场不好就少买」的顺周期砍仓，回测引擎里没有对应物。
    """
    assert g["regime_weighting_enabled"] is False
    assert g["vol_control_enabled"] is False
    assert g["var_monitor_enabled"] is False
    assert g["credibility_enabled"] is False


def test_t3_overlay_is_off(loop):
    """T3 覆盖层（回撤斜坡 + 波动率目标）在 slack3 下三档成本全输 off。"""
    assert loop.derisk_mode == "off"
    assert float(loop.vol_target_pct) == 0.0
    assert loop.enable_risk_scaling is False
    assert float(loop.tilt_strength) == 0.0, "倾斜抬高换手，50bp 下优势消失（相位平均）"


# ---------------------------------------------------------------------------
# 四、熔断自愈（第 15 处漂移：冷静期已关）
# ---------------------------------------------------------------------------

def test_heal_throttle_is_off(g):
    """半仓冷静期是按构造在谷底减仓：熔断由深回撤触发、解冻在回撤刚修复时。"""
    assert float(g["auto_resume_risk_fraction"]) == 1.0
    assert float(g["auto_resume_throttle_days"]) == 0.0
    assert float(g["auto_resume_throttle_sec"]) == 0.0


def test_heal_hysteresis_is_kept(g):
    """滞后带保留：没有它账户会在熔断线上下反复横跳（10-26/27/28 实测）。"""
    assert float(g["auto_resume_drawdown_margin_pct"]) == 2.0


# ---------------------------------------------------------------------------
# 五、推导式不变量（数字之间的关系才是真坑）
# ---------------------------------------------------------------------------

def test_risk_budget_is_not_the_binding_constraint(g):
    """风险预算反算出的单票上限，必须**宽于**权重上限。

    ``_apply_execution_limits`` 对每笔买入强制:
        单票市值 ≤ 总资产 × max_risk_per_trade_pct ÷ 止损距离
    个股止损关掉后，止损距离只剩 ``risk_budget_stop_pct`` 这个静态下限。
    若把 max_risk_per_trade_pct 设成 0.5%（dataclass 默认值），反算就是
        total × 0.005 / 0.05 = 总资产 10%
    而计划单票是 90%/8 = 11.25% ⇒ **每一笔买入都被悄悄砍到 10%**，
    8 只合计 80%、永久多留 10pp 现金，回测的收益永远兑现不了。
    这正是「仓位建不满」的一种，且完全静默。
    """
    per_trade_cap = float(g["max_risk_per_trade_pct"]) / float(g["risk_budget_stop_pct"])
    assert per_trade_cap >= float(g["max_single_position_pct"]) - 1e-9, (
        f"风险预算反算的单票上限 {per_trade_cap:.1%} 窄于 max_single_position_pct "
        f"{float(g['max_single_position_pct']):.0%} ⇒ 它会变成主约束，"
        "每笔买入都被砍小")


def test_daily_turnover_allows_building_a_full_book(g, loop):
    """当日买入上限必须够「从现金一次建满」。

    原值 0.30 时每次调仓只能建 1/3 仓位，而周期 15 个交易日 ⇒ 两个多月才建满。
    """
    need = 1.0 - float(loop.cash_reserve_pct)
    assert float(g["max_daily_turnover_pct"]) >= need - 1e-9, (
        f"max_daily_turnover_pct {g['max_daily_turnover_pct']} < 1-cash_reserve {need:.2f}"
        " ⇒ 一次调仓建不满")


def test_order_rate_limit_can_serve_one_full_rebalance(loop):
    """单 tick 委托上限必须够覆盖一次完整调仓（买+卖），否则截断只执行一半。"""
    assert loop.max_orders_per_tick >= 2 * int(loop.max_positions)


def test_sector_cap_leaves_room_for_at_least_two_names(g, loop):
    """行业上限必须容得下至少 2 只 —— 否则 8 只组合实际只能买 1 只/行业。

    30% 是「20 只持仓」时代的遗留：8 只时只够 2.67 只，撞线就会拒单。
    """
    per_name = (1.0 - float(loop.cash_reserve_pct)) / int(loop.max_positions)
    assert float(g["max_sector_pct"]) >= 2 * per_name - 1e-9


def test_weight_cap_is_consistent_with_position_count(loop):
    """单票上限不能低于等权权重，否则组合天生建不满。"""
    per_name = (1.0 - float(loop.cash_reserve_pct)) / int(loop.max_positions)
    assert float(loop.weight_cap) >= per_name - 1e-9


# ---------------------------------------------------------------------------
# 六、默认值与生产值不许分叉
# ---------------------------------------------------------------------------

def test_dataclass_defaults_do_not_reintroduce_procyclical_rules():
    """dataclass 默认值就是「配错 yaml 时的兜底」，它本身不能是顺周期规则。"""
    c = GuardConfig()
    assert c.regime_weighting_enabled is False
    assert c.vol_control_enabled is False
    assert c.var_monitor_enabled is False
    assert c.credibility_enabled is False
    assert c.auto_resume_risk_fraction == 1.0
    assert c.auto_resume_throttle_days == 0.0
    assert c.auto_resume_throttle_sec == 0

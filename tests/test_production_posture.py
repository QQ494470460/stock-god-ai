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


# ---------------------------------------------------------------------------
# 七、策略形状参数（已用相位平均验过，动它等于让全部历史结论作废）
# ---------------------------------------------------------------------------

def test_strategy_shape_params_are_phase_vetted(loop):
    """这七个值不是「调出来的」，是**扫过相位**之后唯一站得住的形状。

    证据（1140 天 × 200 只，生产口径）：
      · ``rebalance_days`` 5→15：5 天周期在 50bp 下结构必亏（每期边际 ~0.5%，
        一次往返成本 ~1.1%）；15 天是「三档成本 + 相位平均」唯一同时为正的格子。
      · ``hold_rank_slack`` 3.0：老仓只要还在保留线内就不换出，是降换手的第一杠杆。
        与 ``rebalance_days`` 是同一件事的两个旋钮（周期管节奏、滞后管名次边界）。
      · ``tilt_strength`` 0.0：倾斜 = 在波动最大的票上押最重，与风险平价反向；
        实证 0 最优（1.341），任何 >0 单调劣化（1.043~1.238，回撤 −25.5%→−33.4%）。
      · ``cash_reserve_pct`` 0.10：0% 在 10bp 下确实多 2.65pp（配对 t 3.73），
        但 50bp 衰减到 1.53pp 且回撤更深 ⇒ 保留。
      · ``max_positions`` 8 / ``band_pct`` 0.15 / ``weighting``：真 OOS 面板
        RankIC .061、IC_IR 12.26、胜率 61.5% 那一组。

    ⚠️ 这些值是**联合**成立的：改 ``max_positions`` 必须重算
    ``max_sector_pct``（见 ``test_sector_cap_leaves_room_for_at_least_two_names``），
    改 ``rebalance_days`` 必须重跑 ``scripts/exec_phase.py``。
    """
    assert int(loop.rebalance_days) == 15
    assert float(loop.hold_rank_slack) == 3.0
    assert int(loop.max_positions) == 8
    assert float(loop.cash_reserve_pct) == 0.10
    assert float(loop.band_pct) == 0.15
    assert loop.weighting == "risk_parity"
    assert float(loop.weight_cap) == 0.30
    assert float(loop.tilt_strength) == 0.0


def test_ab_baseline_c15_still_equals_the_production_caliber(loop):
    """``scripts/exec_ab.py`` 里那个叫 ``c15`` 的方案必须**就是**生产口径。

    所有「A 比 B 好 X pp」的历史结论都是相对 ``c15`` 记录的。生产一旦改了
    周期/排名滞后/持仓数，"c15" 这个名字还在、数字却代表另一个策略 ——
    此时引用旧结论就是引用别人家的回测。这条断言让那件事立刻响。
    """
    from scripts.exec_ab import BASE, SCENARIOS

    c15 = dict(BASE)
    c15.update(SCENARIOS["c15"])
    assert int(c15["rebalance_days"]) == int(loop.rebalance_days)
    assert float(c15["hold_rank_slack"]) == float(loop.hold_rank_slack)
    assert int(c15["top_n"]) == int(loop.max_positions)
    assert float(c15["cash_reserve_pct"]) == float(loop.cash_reserve_pct)
    assert float(c15["band_pct"]) == float(loop.band_pct)
    assert float(c15["weight_cap"]) == float(loop.weight_cap)
    assert c15["weighting"] == loop.weighting
    # 基准必须等于「个股层退出全关」的生产姿态
    assert c15["stop_mode"] == "none"


def test_phase_script_prod_base_matches_the_production_caliber(loop):
    """``exec_phase.py`` 的 ``PROD_BASE`` 是硬写死的口径兜底。

    它刻意不复用 ``SCENARIOS``：万一某个方案忘记覆盖，仍能保证「15 日 + slack3」。
    代价是它自己会漂 —— 所以单独钉一遍。
    """
    from scripts.exec_phase import PROD_BASE

    assert int(PROD_BASE["rebalance_days"]) == int(loop.rebalance_days)
    assert float(PROD_BASE["hold_rank_slack"]) == float(loop.hold_rank_slack)
    assert PROD_BASE["stop_mode"] == "none"


# ---------------------------------------------------------------------------
# 八、打分后处理层（第 16 处漂移 —— 比风控层那 15 处更上游）
# ---------------------------------------------------------------------------

def test_cross_sectional_neutralize_is_off():
    """行业中性化 2026-11-04 关闭：它是全链条最贵的单项。

    1140 天 × 200 只 × 15 相位 × 三档成本，逐相位配对差
    −4.71 / −5.11 / −5.96pp，配对 t −2.78 / −2.67 / −3.68，
    负相位 11 / 11 / 12（共 15）。回撤每一档都更深，年换手 10.5 → 12.0。
    **成本越贵差得越多 ⇒ 逆成本单调，不是噪声。**

    为什么会长这样：模型对「行业水位」的预测本身就是 alpha，事后按行业去均值
    等于把它剔掉。而「不要全押一个行业」在持仓层用 ``max_sector_pct`` 只需
    −0.45pp —— 同一个目的，工具放错层级代价差一个数量级。

    ⚠️ 重开之前必须回答：为什么 15 个相位 × 三档成本都指向负？只跑过单相位
    的「有效」不构成理由（相位离散度 16.64pp）。复核命令：
    ``python scripts/exec_phase.py --only neu --industry-map data/industry_map.json``
    """
    assert AppConfig().ml.cross_sectional_neutralize is False, (
        "中性化被重新打开了 —— 先跑三档成本的相位平均，确认 Δ 不再为负")


def test_signal_ema_stays_on_but_its_evidence_is_weaker_than_documented():
    """跨日 EMA 保持开启，但**证据强度弱于 settings 里的旧注释**。

    本轮回测（15 相位 × 10bp）：EMA(0.85) 相对原分 −1.11pp、配对 t −1.36、
    11/15 相位为负。方向偏负但不显著 ⇒ 不动它（没有「明确有害」的证据）。

    要点是别再把旧注释当成硬证据：那张表是**单相位**、且**含 T3 覆盖层**
    （T3 现已关闭）。在 16.64pp 的相位离散度面前，4pp 的「提升」不构成结论。
    """
    ml = AppConfig().ml
    assert ml.signal_fusion_enabled is True
    assert float(ml.signal_fusion_alpha) == 0.85


def test_backtest_signal_pipeline_defaults_are_off():
    """回测引擎的打分后处理默认必须全关。

    否则「历史结论逐位可复现」这条底线就断了：2026-11-04 之前的全部结论
    都是在**原始 predict_proba** 上得出的，默认打开等于让所有旧数字失效。
    """
    from src.core.exec_backtest import ExecConfig

    c = ExecConfig()
    assert c.signal_neutralize is False
    assert float(c.signal_ema_alpha) == 0.0



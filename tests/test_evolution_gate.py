# -*- coding: utf-8 -*-
"""自我进化 · 样本外验证闸门测试。

闸门是「进化」这件事唯一的刹车。它的失效方式有两种，都很致命：

* **假阳性**：参数对回测毫无影响（没映射上），差异为 0，
  却因为「没有变差」被判为通过 → 系统以为改进了，其实什么都没发生；
* **假阴性**：基准夏普缺失时顺势放行 → 没有可比对象也敢说胜出。

因此本文件的断言重点在「拒绝的理由是否成立」，
而不只是「拒绝/通过」这个布尔值。

回测部分用**合成面板**（可解析的确定性数据）验证 A/B 逻辑，
不依赖本机行情数据；面板缓存到磁盘的读写由 ``src/evolution/panels.py``
负责，``default_panel_provider`` 从这里读缓存。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evolution import gate  # noqa: E402
from src.evolution.gate import GateCriteria  # noqa: E402
from src.evolution.store import CandidateStore  # noqa: E402
from src.evolution.types import (  # noqa: E402
    Candidate, Change, Evaluation, ST_PENDING, ST_REJECTED,
)


def _cand(changes, source="param_search") -> Candidate:
    return Candidate.create(source=source, title="测试候选", changes=changes)


# ------------------------------------------------------------------ 合成面板

def _panels(n_days: int = 80, n_codes: int = 20, seed: int = 7):
    """构造确定性面板：代码按「未来收益」单调排序。

    这样 top_n 越小（只买最好的几只）→ 组合收益越高、波动越低，
    于是「把 max_positions 调小」应当明显胜出，可用于验证 A/B 方向正确。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{600000 + i}" for i in range(n_codes)]

    # 每只股票一个固定质量分位，收益与之成正比 + 少量噪声
    quality = np.linspace(1.0, 0.0, n_codes)
    base_ret = 0.004 * quality          # 日收益

    close = pd.DataFrame(1.0, index=dates, columns=codes)
    for j, c in enumerate(codes):
        r = base_ret[j] + rng.normal(0, 0.0008, len(dates))
        close[c] = (1 + r).cumprod()

    score = pd.DataFrame(
        np.tile(quality * 100.0, (n_days, 1)), index=dates, columns=codes)

    return score, close, None


def _provider(panels):
    return lambda: panels


# ------------------------------------------------------------------ 第一层：风控硬线

def test_risk_precheck_blocks_protected_namespace():
    blockers, warnings = gate.risk_precheck([
        Change("guard.max_portfolio_drawdown_pct", 8.0, 20.0)])
    assert blockers and "风控硬线" in blockers[0]
    assert warnings == []


def test_risk_precheck_blocks_protected_leaf():
    blockers, _ = gate.risk_precheck([
        Change("trading_loop.max_orders_per_tick", 10, 99)])
    assert blockers and "风控硬线" in blockers[0]


def test_risk_precheck_blocks_unregistered():
    blockers, _ = gate.risk_precheck([
        Change("trading_loop.不存在的参数", 1, 2)])
    assert blockers


def test_risk_precheck_blocks_out_of_range():
    blockers, _ = gate.risk_precheck([
        Change("trading_loop.band_pct", 0.25, 99.0)])
    assert blockers and "越界" in blockers[0]


def test_risk_precheck_warns_on_aggressive_change():
    """更激进方向不判负，但必须留下提醒（人工批准的依据）。"""
    blockers, warnings = gate.risk_precheck([
        Change("trading_loop.weight_cap", 0.20, 0.30)])
    assert blockers == []
    assert warnings and "更激进" in warnings[0]


def test_risk_precheck_empty_candidate():
    blockers, _ = gate.risk_precheck([])
    assert blockers


def test_hardline_is_checked_before_any_backtest():
    """风控硬线必须**前置**判负 —— 连回测都不该跑。"""
    called = {"n": 0}

    def provider():
        called["n"] += 1
        return _panels()

    ev = gate.evaluate(
        _cand([Change("guard.max_single_position_pct", 0.10, 0.80)]),
        provider=provider)

    assert ev.passed is False
    assert ev.evaluator == "gate:hardline"
    assert called["n"] == 0, "触碰风控硬线时不得执行回测"


# ------------------------------------------------------------------ 第二层：可评估性

def test_unmappable_change_is_rejected_not_passed():
    """命脉：不改评分面板、回测也表达不了的参数，**绝不等于通过**。

    ``scan_interval_sec``（盘中扫描间隔）只影响实盘轮询频率，
    回测里没有任何对应物 —— 所以只能拒。
    """
    ev = gate.evaluate(
        _cand([Change("trading_loop.scan_interval_sec", 30.0, 60.0)]),
        provider=_provider(_panels()))

    assert ev.passed is False
    assert ev.evaluator == "gate:unmappable"
    assert any("无法用回测评估" in r for r in ev.reasons)


def test_mixed_changes_with_any_unmappable_is_rejected():
    """命脉：只要有一项无法评估，整批都不许通过。

    否则「能测的那条胜出」会给「没测过的那条」背书 —— 搭便车放行。
    """
    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 10, 6),
               Change("trading_loop.scan_interval_sec", 30.0, 60.0)]),
        provider=_provider(_panels()))

    assert ev.passed is False
    assert ev.evaluator == "gate:unmappable"
    assert any("不允许部分放行" in r for r in ev.reasons)
    # 且必须真的没有跑回测（baseline 为空）
    assert ev.baseline == {}


def test_no_panel_means_rejected_with_reason():
    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 10, 6)]),
        provider=lambda: None)

    assert ev.passed is False
    assert ev.evaluator == "gate:no-data"
    assert any("没有可用的样本外面板" in r for r in ev.reasons)


def test_gate_rejects_with_no_data_when_there_is_no_panel(monkeypatch):
    """没有面板 → 判 ``gate:no-data``（宁可不过，也不编数据）。

    这条用例原本叫 ``test_default_provider_returns_none``，直接断言
    「默认 provider 永远返回 None」—— 它把当时的**实现细节**当成了规格，
    于是 ``panels.py`` 做出缓存之后，这条断言反而成了阻止修复的枷锁：
    谁把默认 provider 接上缓存，它就会红。

    现在改成断言**意图**：缓存里没有面板时，provider 给 None、
    闸门判 ``gate:no-data``。并且**显式**把缓存说成「没有」，
    而不是依赖「本机恰好没缓存」—— 否则装了面板的机器上这条用例不成立。
    """
    from src.evolution import panels

    monkeypatch.setattr(panels, "panel_available", lambda: False)

    assert gate.default_panel_provider() is None
    ev = gate.evaluate(_cand([Change("trading_loop.max_positions", 10, 6)]))
    assert ev.passed is False and ev.evaluator == "gate:no-data"


def test_map_changes_reports_unmappable():
    """组合构建类参数都能映射；纯运行期参数（扫描间隔）不能。"""
    overrides, unmappable = gate.map_changes([
        Change("trading_loop.max_positions", 10, 8),
        Change("trading_loop.cash_reserve_pct", 0.10, 0.15),
        Change("trading_loop.band_pct", 0.25, 0.30),
        Change("trading_loop.weighting", "equal", "score"),
        Change("trading_loop.scan_interval_sec", 30.0, 60.0),
    ])
    assert overrides == {
        "top_n": 8, "cash_reserve_pct": 0.15, "band_pct": 0.30,
        "weighting": "score",
    }
    assert unmappable == ["trading_loop.scan_interval_sec"]


def test_gate_backtest_keys_match_map():
    """映射表的值域必须与回测引擎能接受的键一致（防拼错参数名）。"""
    from src.core.backtest import BacktestConfig

    for key in gate.gate_backtest_keys():
        assert key in BacktestConfig.__dataclass_fields__, \
            f"{key} 不是 BacktestConfig 的字段，映射写错了"


# ------------------------------------------------------------------ 第三层：指标对比

def _m(**kw) -> dict:
    base = {"n_periods": 30, "sharpe": 1.0, "ann_return": 0.10,
            "ann_vol": 0.15, "max_drawdown": -0.10}
    base.update(kw)
    return base


def test_compare_passes_on_clear_win():
    ok, reasons, delta = gate.compare_metrics(
        _m(), _m(sharpe=1.4, ann_return=0.15, max_drawdown=-0.08))

    assert ok is True
    assert delta["sharpe"] == pytest.approx(0.4)
    assert any("夏普提升" in r for r in reasons)


def test_compare_fails_when_sharpe_gain_too_small():
    ok, reasons, _ = gate.compare_metrics(_m(), _m(sharpe=1.02))
    assert ok is False
    assert any("夏普未达门槛" in r for r in reasons)


def test_compare_fails_on_return_regression():
    ok, reasons, _ = gate.compare_metrics(
        _m(), _m(sharpe=1.5, ann_return=0.05))
    assert ok is False
    assert any("年化收益倒退" in r for r in reasons)


def test_compare_fails_on_drawdown_worsening():
    ok, reasons, _ = gate.compare_metrics(
        _m(), _m(sharpe=1.5, max_drawdown=-0.20))
    assert ok is False
    assert any("最大回撤恶化" in r for r in reasons)


def test_compare_fails_when_periods_insufficient():
    ok, reasons, delta = gate.compare_metrics(
        _m(n_periods=3), _m(n_periods=4, sharpe=2.0))
    assert ok is False
    assert delta == {}
    assert any("调仓期数不足" in r for r in reasons)


def test_compare_refuses_when_base_sharpe_missing():
    """没有可比基准时不许放行（假阴性防线）。"""
    ok, reasons, _ = gate.compare_metrics(
        _m(sharpe=None), _m(sharpe=2.0))
    assert ok is False
    assert any("基准夏普缺失" in r for r in reasons)


def test_compare_refuses_when_candidate_sharpe_missing():
    ok, reasons, _ = gate.compare_metrics(_m(), _m(sharpe=None))
    assert ok is False
    assert any("候选夏普无法计算" in r for r in reasons)


def test_compare_handles_nan():
    ok, reasons, _ = gate.compare_metrics(_m(sharpe=float("nan")), _m(sharpe=1.2))
    assert ok is False


def test_compare_optional_vol_constraint():
    crit = GateCriteria(require_no_vol_increase=True)
    ok, reasons, _ = gate.compare_metrics(
        _m(), _m(sharpe=1.5, ann_vol=0.25), crit)
    assert ok is False
    assert any("年化波动上升" in r for r in reasons)

    ok, _, _ = gate.compare_metrics(
        _m(), _m(sharpe=1.5, ann_vol=0.10), crit)
    assert ok is True


# ------------------------------------------------------------------ A/B 回测端到端

def test_evaluate_passes_when_candidate_genuinely_better(monkeypatch):
    """面板里质量分位单调 → 减少持仓数应当真的提升夏普。"""
    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 20})

    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 20, 5)]),
        provider=_provider(_panels()))

    assert ev.evaluator == "gate:backtest"
    assert ev.baseline and ev.candidate
    assert ev.candidate["sharpe"] > ev.baseline["sharpe"]
    assert ev.passed is True, ev.reasons
    assert ev.window


def test_evaluate_rejects_when_candidate_worse(monkeypatch):
    """反向：把持仓数从 5 拉到 20（买进一堆劣质标的）必须被拒。"""
    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 5})

    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 5, 20)]),
        provider=_provider(_panels()))

    assert ev.evaluator == "gate:backtest"
    assert ev.candidate["sharpe"] < ev.baseline["sharpe"]
    assert ev.passed is False
    assert any("夏普未达门槛" in r for r in ev.reasons)


def test_evaluate_rejects_noop_candidate(monkeypatch):
    """候选值与当前值相同 → 无对比可言，不许判通过。"""
    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 10})
    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 10, 10)]),
        provider=_provider(_panels()))
    assert ev.passed is False
    assert ev.evaluator == "gate:unmappable"


def test_evaluate_reports_aggressive_change_as_warning(monkeypatch):
    """胜出但方向更激进 → 仍标 pending，且原因里带上提醒。"""
    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 3})
    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 3, 20)]),
        provider=_provider(_panels()))
    # 无论通过与否，提醒都必须在
    assert any("更激进" in r for r in ev.reasons)


def test_evaluate_survives_provider_exception():
    def boom():
        raise RuntimeError("面板加载崩了")

    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 10, 6)]), provider=boom)
    assert ev.passed is False and ev.evaluator == "gate:no-data"


def test_evaluate_handles_tiny_panel(monkeypatch):
    """面板过小 → 回测拿不到调仓期 → 判负而不是崩。"""
    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 10})
    ev = gate.evaluate(
        _cand([Change("trading_loop.max_positions", 10, 5)]),
        provider=_provider(_panels(n_days=4, n_codes=5)))
    assert ev.passed is False


# ------------------------------------------------------------------ 自带证据校验

def test_supplied_evaluation_missing():
    ev = gate.verify_supplied_evaluation(None)
    assert ev.passed is False
    assert any("未提供样本外证据" in r for r in ev.reasons)


def test_supplied_evaluation_self_stamped_is_rejected():
    """生成器不许自称通过 —— evaluator 必须在白名单内。"""
    fake = Evaluation(passed=True, evaluator="me", reasons=["我说好"],
                      delta={"sharpe": 1.0}, window="2024~2025")
    ev = gate.verify_supplied_evaluation(fake)
    assert ev.passed is False
    assert any("不在可信白名单" in r for r in ev.reasons)


def test_supplied_evaluation_without_evidence_is_rejected():
    fake = Evaluation(passed=True, evaluator="walk_forward", window="2024~2025")
    ev = gate.verify_supplied_evaluation(fake)
    assert ev.passed is False
    assert any("reasons" in r or "delta" in r for r in ev.reasons)


def test_supplied_evaluation_without_window_is_rejected():
    fake = Evaluation(passed=True, evaluator="walk_forward",
                      reasons=["样本外胜出"], delta={"sharpe": 0.3})
    ev = gate.verify_supplied_evaluation(fake)
    assert ev.passed is False
    assert any("window" in r for r in ev.reasons)


def test_supplied_evaluation_accepts_credible_evidence():
    good = Evaluation(passed=True, evaluator="walk_forward",
                      baseline={"sharpe": 0.8}, candidate={"sharpe": 1.2},
                      reasons=["5 折 walk-forward 平均夏普提升"],
                      delta={"sharpe": 0.4}, window="2023-01~2025-06")
    ev = gate.verify_supplied_evaluation(good)
    assert ev.passed is True
    assert ev.evaluator == "gate:supplied"
    assert any("校验通过" in r for r in ev.reasons)


def test_supplied_evaluation_rejects_failed_self_report():
    bad = Evaluation(passed=False, evaluator="walk_forward",
                     reasons=["没赢"], delta={"sharpe": -0.1}, window="w")
    ev = gate.verify_supplied_evaluation(bad)
    assert ev.passed is False


# ------------------------------------------------------------------ 落地到队列

def test_apply_verdict_pending_on_pass(tmp_path):
    st = CandidateStore(tmp_path / "c.json")
    c = _cand([Change("trading_loop.max_positions", 20, 5)])
    ev = Evaluation(passed=True, evaluator="gate:backtest",
                    reasons=["✅ 全部指标达标"])

    out = gate.apply_verdict(c, ev, st)

    assert out.status == ST_PENDING
    assert out.evaluation is ev
    assert st.summary()["pending"] == 1


def test_apply_verdict_rejected_on_fail(tmp_path):
    st = CandidateStore(tmp_path / "c.json")
    c = _cand([Change("guard.max_single_position_pct", 0.1, 0.8)])
    ev = Evaluation(passed=False, evaluator="gate:hardline",
                    reasons=["触碰风控硬线"])

    out = gate.apply_verdict(c, ev, st)

    assert out.status == ST_REJECTED
    assert "风控硬线" in out.decision_reason


def test_apply_verdict_is_idempotent(tmp_path):
    st = CandidateStore(tmp_path / "c.json")
    c = _cand([Change("trading_loop.max_positions", 20, 5)])
    ev = Evaluation(passed=True, evaluator="gate:backtest", reasons=["ok"])

    gate.apply_verdict(c, ev, st)
    gate.apply_verdict(c, ev, st)

    assert st.summary()["total"] == 1


# ---------------------------------------------------------------------------
# 默认面板提供者：读缓存，而不是「假装没有面板」
# ---------------------------------------------------------------------------

def test_default_panel_provider_reads_the_cache(monkeypatch):
    """有缓存时必须真的把面板交出来。

    回归背景：``default_panel_provider()`` 曾经**硬编码 ``return None``**
    （注释是「本机没有随取随用的面板缓存」—— 那句话在 ``panels.py``
    落地缓存之后就过期了）。后果是凡是没有被显式注入 provider 的调用方
    （典型是**生成器 D / GA**）一律被判「没有样本外面板」而跳过，
    尽管面板就躺在 ``data/evolution/panel/``。
    实跑证据：同一个进化回合里，闸门拿真实夏普否决了生成器 A 的候选，
    生成器 D 却说「没有样本外面板」—— 同一块磁盘，两个相反的答案。
    """
    from src.evolution import panels

    sentinel = (object(), object(), None)
    monkeypatch.setattr(panels, "panel_available", lambda: True)
    monkeypatch.setattr(panels, "load_panel", lambda: sentinel)

    assert gate.default_panel_provider() is sentinel


def test_default_panel_provider_is_none_without_cache(monkeypatch):
    """没缓存就返回 None —— 让闸门判负并说明原因，而不是编个空面板糊过去。"""
    from src.evolution import panels

    monkeypatch.setattr(panels, "panel_available", lambda: False)

    assert gate.default_panel_provider() is None


def test_default_panel_provider_survives_a_broken_cache(monkeypatch):
    """缓存文件损坏时降级为 None，不能把整个回合炸掉。"""
    from src.evolution import panels

    def _boom():
        raise RuntimeError("parquet 损坏")

    monkeypatch.setattr(panels, "panel_available", lambda: True)
    monkeypatch.setattr(panels, "load_panel", _boom)

    assert gate.default_panel_provider() is None



# ------------------------------------------------------------------ 口径一致性（2026-11-04）

def test_backtest_map_covers_the_two_caliber_params():
    """周期与排名滞后必须进映射表 —— 它们是**口径**，不是可调参数。

    缺 ``rebalance_days`` 时闸门用引擎默认 5 天周期，而生产是 15 天：
    「5 天周期在结构上必亏」（边际收益 0.5% vs 往返成本 1.1%），
    等于拿一个必亏的策略去判断「改 max_positions 好不好」。
    缺 ``hold_rank_slack`` 时回测没有排名滞后概念，而生产是 3.0；
    §memory 实测同一批数据 slack 1→3 会把结论正反颠倒。
    """
    assert gate.BACKTEST_MAP["trading_loop.rebalance_days"] == ("rebalance_days", int)
    assert gate.BACKTEST_MAP["trading_loop.hold_rank_slack"] == ("hold_rank_slack", float)


def test_current_backtest_params_matches_production_caliber():
    """基准侧必须落到生产值，而不是回测引擎的隐式默认。

    ``rebalance_days`` / ``hold_rank_slack`` 都**不在**可进化参数空间里，
    所以取不到 overrides、也取不到 space 默认值 —— 必须由生产配置兜底。
    没有这层兜底，两者会静默退到 5 / 1.0，基准就成了另一套策略。
    """
    from config.settings import AppConfig

    p = gate._current_backtest_params()
    loop = AppConfig().trading_loop
    assert p["rebalance_days"] == int(loop.rebalance_days) == 15
    assert p["hold_rank_slack"] == pytest.approx(float(loop.hold_rank_slack)) == 3.0
    assert p["top_n"] == int(loop.max_positions)


def test_phase_samples_default_is_more_than_one():
    """默认必须相位平均：单相位的配对差方向会随相位翻转（实测最差 −1.24pp、
    最好 +8.87pp），闸门会按 10~20% 的概率把噪声批成改进。"""
    assert GateCriteria().phase_samples >= 3


def test_ab_over_phases_sweeps_offsets_and_averages(monkeypatch):
    """相位平均必须真的扫 0..k-1 并把指标取均值。"""
    seen = []

    # 用 top_n 是否被改成 1 来区分「候选」与「基准」
    def fake(score, close, bench, params):
        seen.append(params["rebalance_offset"])
        cand = params.get("top_n") == 1
        base_s = 1.0
        cand_s = 1.0 + (0.4 if params["rebalance_offset"] % 2 == 0 else -0.1)
        return {"n_periods": 30, "sharpe": cand_s if cand else base_s,
                "ann_return": 0.1, "ann_vol": 0.15, "max_drawdown": -0.1}

    monkeypatch.setattr(gate, "_run", fake)
    b, c, diffs = gate._ab_over_phases(None, None, None,
                                       {"top_n": 8}, {"top_n": 1}, 4)
    assert seen == [0, 0, 1, 1, 2, 2, 3, 3], "每个相位都要跑基准与候选各一次"
    assert b["sharpe"] == pytest.approx(1.0)
    assert c["sharpe"] == pytest.approx(1.0 + 0.15)
    assert diffs == pytest.approx([0.4, -0.1, 0.4, -0.1])


def test_evaluate_reports_phase_agreement(monkeypatch):
    """判定理由里必须写明相位均值与「几个相位方向相反」。

    方向随相位翻转却只报一个漂亮的单点，是这套闸门最隐蔽的失效方式。
    """
    monkeypatch.setattr(gate, "_current_backtest_params",
                        lambda: {"top_n": 20})

    def fake(score, close, bench, params):
        cand = params.get("top_n") == 5
        off = params.get("rebalance_offset", 0)
        base_s = 1.0
        cand_s = 1.5 if off == 0 else (0.8 if off == 1 else 1.4)
        return {"n_periods": 30, "sharpe": cand_s if cand else base_s,
                "ann_return": 0.10 + (0.05 if cand else 0.0),
                "ann_vol": 0.15, "max_drawdown": -0.10}

    monkeypatch.setattr(gate, "_run", fake)
    ev = gate.evaluate(_cand([Change("trading_loop.max_positions", 20, 5)]),
                      provider=_provider(_panels()))
    txt = " ".join(ev.reasons)
    assert "相位平均" in txt
    assert "1/3" in txt, "必须报出方向相反的相位个数"
    assert "方向不一致" in txt

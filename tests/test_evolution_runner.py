# -*- coding: utf-8 -*-
"""自我进化 · 回路编排测试。

编排层唯一的职责是**不许越权**，所以断言集中在四条红线：

1. 熔断期间不生成、不落地任何东西；
2. 默认**不**自动落地；即使开了 ``auto_apply``，也绝不放行更激进的改动；
3. ``candidates.json`` 被手改成风控参数后，落地必须被拦下（二次校验）；
4. 回滚只能对「已落地」的候选做，且必须把旧值写回去。
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evolution import gate, overrides as ov_mod  # noqa: E402
from src.evolution.feedback import PerformanceSnapshot  # noqa: E402
from src.evolution.runner import EvolutionRunner  # noqa: E402
from src.evolution.store import CandidateStore  # noqa: E402
from src.evolution.types import (  # noqa: E402
    Candidate, Change, Evaluation, ST_APPLIED, ST_APPROVED, ST_PENDING,
    ST_REJECTED, ST_ROLLED_BACK,
)

CURRENT = {
    "trading_loop.band_pct": 0.25,
    "trading_loop.cash_reserve_pct": 0.10,
    "trading_loop.max_positions": 10,
    "trading_loop.weight_cap": 0.30,
    "trading_loop.min_trade_amount": 5000.0,
    "trading_loop.weighting": "equal",
}


# ------------------------------------------------------------------ 夹具

def _panels(n_days: int = 120, n_codes: int = 20, seed: int = 11):
    """合成面板：**质量分位单调，且尾部为负 alpha**。

    质量从 +1.0 线性降到 -1.0，即排名最后的那些票是**亏钱**的。
    这样「持仓越少越优」是**双向成立**的 —— 减少持仓既抬高了均值收益，
    又剔除了负 alpha 的拖累，两处都改善夏普，判定不受噪声实现左右。

    （早期版本用 ``linspace(1.0, 0.0, n)``，尾部只是「低收益但不亏」，
    换成 8 只后波动上升的幅度会盖过收益上升的幅度，夏普反而可能下降 ——
    实测 20 个种子里有 4 个这样。改成负 alpha 尾部后 19/20 稳健胜出。）
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{600000 + i}" for i in range(n_codes)]
    quality = np.linspace(1.0, -1.0, n_codes)
    close = pd.DataFrame(1.0, index=dates, columns=codes)
    for j, c in enumerate(codes):
        r = 0.004 * quality[j] + rng.normal(0, 0.008, len(dates))
        close[c] = (1 + r).cumprod()
    score = pd.DataFrame(np.tile(quality * 100.0, (n_days, 1)),
                         index=dates, columns=codes)
    return score, close, None


def _snap(halted: bool = False, **kw) -> PerformanceSnapshot:
    base = dict(
        generated_at="2026-10-14T13:00:00",
        n_verified=10,
        n_settlements=20,
        equity_series=[
            {"date": f"2026-09-{i:02d}", "equity": 100000.0,
             "cash": 10000.0, "n_positions": 10}
            for i in range(1, 21)],
        trade_stats={"fill": 40},
        guard={"halted": halted, "halt_reason": "回撤熔断" if halted else ""},
        metrics={"insufficient": False, "turnover": 5.2},
    )
    base.update(kw)
    return PerformanceSnapshot(**base)


def _underfilled_snap() -> PerformanceSnapshot:
    """持仓长期不满 → 生成器 A 会提 max_positions（**唯一能过真实闸门**的处方）。"""
    return _snap( metrics={"insufficient": False},
                 equity_series=[
                     {"date": f"2026-09-{i:02d}", "equity": 100000.0,
                      "cash": 10000.0, "n_positions": 3}
                     for i in range(1, 21)])


@pytest.fixture
def env(tmp_path, monkeypatch):
    """隔离：候选队列、覆盖层、进化引擎、当前参数全部指向 tmp。"""
    from src.models.evolution import EvolutionEngine

    monkeypatch.setattr(ov_mod, "OVERRIDES_FILE", tmp_path / "ov.json")
    monkeypatch.setattr(ov_mod, "read_effective", lambda k: CURRENT.get(k))
    evo = EvolutionEngine(state_file=tmp_path / "evo.json")
    monkeypatch.setattr(ov_mod, "_evolution_engine", lambda: evo)

    # 引擎写日志也走 tmp
    import src.models.evolution as mev
    monkeypatch.setattr(mev, "EVOLUTION_FILE", tmp_path / "evo.json")

    monkeypatch.setattr(gate, "_current_backtest_params", lambda: {"top_n": 10})

    # LLM 生成器默认关掉：本机 config/deepseek.yaml 有真实 Key，
    # runner 默认跑 REGISTRY 全量，不隔离就会在单测里发起真实网络请求。
    # 需要测它时由用例自己把 REGISTRY["llm_review"] 换成注入了假 chat 的版本。
    import src.evolution.generators.llm_review as lr
    monkeypatch.setattr(lr, "_llm_ready", lambda: False)

    return {"store": CandidateStore(tmp_path / "cands.json"),
            "evo": evo, "evo_path": tmp_path / "evo.json"}


@pytest.fixture
def patched_gate(monkeypatch):
    """把闸门换成确定性判定。

    本文件测的是**编排逻辑**（权责边界），不是回测统计 ——
    统计部分由 test_evolution_gate.py 覆盖。用确定性判定可以让
    「升序/降序/拒绝」三种走向都被稳定复现，不受合成面板噪声影响。
    """
    state = {"pass": True}

    def fake_evaluate(candidate, provider=None, criteria=None):
        return Evaluation(
            passed=state["pass"], at="2026-10-14T13:00:00",
            evaluator="gate:backtest",
            baseline={"sharpe": 1.0}, candidate={"sharpe": 1.2},
            delta={"sharpe": 0.2},
            reasons=["✅ 全部指标达标"] if state["pass"] else ["❌ 夏普未达门槛"],
            window="2024-01-01~2024-12-31")

    monkeypatch.setattr(gate, "evaluate", fake_evaluate)
    return state


@pytest.fixture
def restore_app_config():
    from config.settings import config as app_config

    snap = copy.deepcopy(app_config)
    yield app_config
    for name in snap.__dataclass_fields__:
        setattr(app_config, name, copy.deepcopy(getattr(snap, name)))


def _runner(env, snapshot=None, panels=True):
    return EvolutionRunner(
        store=env["store"],
        provider=(lambda: _panels()) if panels else (lambda: None),
        snapshot_fn=(lambda: snapshot) if snapshot is not None else None,
    )


# ------------------------------------------------------------------ 红线 1：熔断

def test_halted_guard_blocks_the_whole_round(env):
    rep = _runner(env, _snap(halted=True)).run()

    assert rep.n_candidates == 0
    assert rep.pending == []
    assert any("熔断" in n for n in rep.notes)
    assert env["store"].summary()["total"] == 0


# ------------------------------------------------------------------ 正常回合

def test_run_produces_pending_candidate(env, patched_gate, restore_app_config):
    """换手过高 → 生成器 A 提 band_pct → 闸门通过 → 进待批准队列。

    只跑 ``param_search`` 是**刻意**的：这条用例要钉的是「A 的候选怎么走完整条路」，
    不是「全注册表一共能产出几条」。原先它不指定生成器，于是当生成器 D（GA）
    因为「找不到面板」而永远跳过时，``n_candidates == 1`` 恰好成立 ——
    修复 GA 之后这条断言就红了。一个依赖「某个部件坏着」才成立的测试，
    本身就是个隐患：它会让修复看起来像是回归。
    """
    rep = _runner(env, _snap()).run(generators=["param_search"])

    assert rep.n_candidates == 1
    assert len(rep.pending) == 1
    assert rep.rejected == []

    c = env["store"].get(rep.pending[0])
    assert c.status == ST_PENDING
    assert c.changes[0].key == "trading_loop.band_pct"
    assert c.evaluation.passed is True


def test_run_does_not_write_overrides(env, patched_gate, restore_app_config):
    """回合本身**不落地** —— 落地必须人工触发。"""
    _runner(env, _snap()).run()
    assert ov_mod.load_overrides() == {}


def test_run_skips_duplicate_signature(env, patched_gate, restore_app_config):
    """同一症状重复扫描 → 第二次跳过，不刷屏。"""
    r = _runner(env, _snap())
    r.run()
    rep2 = r.run()

    assert rep2.n_candidates == 0            # 同类候选已在队列里 → 不再产出
    assert rep2.pending == []
    assert any("已有同类候选" in s["reason"] for s in rep2.skipped)


def test_run_reports_unknown_generator(env, patched_gate):
    rep = _runner(env, _snap()).run(generators=["不存在的生成器"])
    assert any("未注册的生成器" in e for e in rep.errors)


def test_run_survives_generator_exception(env, patched_gate, monkeypatch):
    from src.evolution.generators import REGISTRY

    def boom(snapshot, **kw):
        raise RuntimeError("生成器崩了")

    monkeypatch.setitem(REGISTRY, "boom", boom)
    rep = _runner(env, _snap()).run(generators=["boom"])
    assert any("异常" in e for e in rep.errors)


def test_run_marks_rejected_when_no_panel(env):
    """没有样本外面板 → 闸门判负 → 候选进 rejected，不进 pending。"""
    rep = _runner(env, _snap(), panels=False).run()

    assert rep.n_candidates == 1
    assert rep.pending == []
    assert len(rep.rejected) == 1
    c = env["store"].get(rep.rejected[0])
    assert c.status == ST_REJECTED
    assert "面板" in c.decision_reason or "面板" in \
        " ".join(c.evaluation.reasons)


def test_verify_is_off_by_default(env, patched_gate, monkeypatch):
    """默认不补验（补验会写 prediction_history.json）。"""
    called = {"n": 0}

    def fake_verify(**kw):
        called["n"] += 1
        return []

    import src.evolution.feedback as fb
    monkeypatch.setattr(fb, "verify_pending", fake_verify)

    _runner(env, _snap()).run()
    assert called["n"] == 0

    _runner(env, _snap()).run(verify=True)
    assert called["n"] == 1


# ------------------------------------------------------------------ 红线 2：自动落地

def test_auto_apply_off_by_default(env, patched_gate, restore_app_config):
    rep = _runner(env, _snap()).run(auto_apply=False)
    assert rep.auto_applied == []
    assert ov_mod.load_overrides() == {}


def test_auto_apply_lands_safe_direction_candidate(env, patched_gate, restore_app_config):
    """band_pct 是中性且标记可自动采纳 → 允许自动落地。"""
    rep = _runner(env, _snap()).run(auto_apply=True)

    assert len(rep.auto_applied) == 1
    assert ov_mod.load_overrides()["trading_loop.band_pct"] == 0.30
    assert env["store"].get(rep.auto_applied[0]).status == "auto_applied"


def test_auto_apply_refuses_aggressive_change(env, patched_gate,
                                             restore_app_config, monkeypatch):
    """现金拖累 → 调低现金保留 = 更激进 → 绝不允许自动落地。

    同 ``test_run_produces_pending_candidate``：只跑 A，把断言钉在
    「这条候选为什么不能自动落地」上，而不是「全注册表产出几条」。
    """
    snap = _snap(metrics={"insufficient": False})
    snap.equity_series = [
        {"date": f"2026-09-{i:02d}", "equity": 100000.0, "cash": 45000.0,
         "n_positions": 10} for i in range(1, 21)]

    rep = _runner(env, snap).run(generators=["param_search"], auto_apply=True)

    assert rep.auto_applied == []
    assert ov_mod.load_overrides() == {}
    assert len(rep.pending) == 1            # 仍可人工批准


# ------------------------------------------------------------------ 人工裁决

def test_approve_then_apply(env, patched_gate, restore_app_config):
    rep = _runner(env, _snap()).run()
    cid = rep.pending[0]

    r = _runner(env, _snap())
    c = r.approve(cid, by="alice", reason="同意")
    assert c.status == ST_APPROVED
    assert c.decided_by == "alice"

    applied = r.apply(cid, by="alice")
    assert applied.status == ST_APPLIED
    assert ov_mod.load_overrides()["trading_loop.band_pct"] == 0.30


def test_apply_pending_without_approve_is_allowed(env, patched_gate, restore_app_config):
    """--apply 直接落地 pending 也允许（CLI 一把过），但必须留痕操作人。"""
    rep = _runner(env, _snap()).run()
    c = _runner(env, _snap()).apply(rep.pending[0], by="bob")
    assert c.status == ST_APPLIED
    assert c.decided_by == "bob"


def test_cannot_apply_rejected(env, restore_app_config):
    """被否决的候选不许落地 —— 否决必须是终局。"""
    st = env["store"]
    c = Candidate.create(source="param_search", title="被拒的候选",
                         changes=[Change("trading_loop.band_pct", 0.25, 0.30)])
    st.add(c)
    st.set_status(c.id, ST_REJECTED, by="gate", reason="样本外不显著")

    with pytest.raises(ValueError, match="不能落地"):
        _runner(env, _snap()).apply(c.id)


def test_cannot_apply_twice(env, patched_gate, restore_app_config):
    rep = _runner(env, _snap()).run()
    r = _runner(env, _snap())
    r.apply(rep.pending[0])
    with pytest.raises(ValueError):
        r.apply(rep.pending[0])


def test_reject_pending(env, patched_gate):
    rep = _runner(env, _snap()).run()
    c = _runner(env, _snap()).reject(rep.pending[0], by="alice",
                                     reason="样本外不显著")
    assert c.status == ST_REJECTED
    assert c.decision_reason == "样本外不显著"


def test_show_unknown_id(env):
    assert _runner(env, _snap()).show("不存在") is None


# ------------------------------------------------------------------ 红线 3：落地二次校验

def test_apply_blocks_hand_edited_protected_change(env, restore_app_config,
                                                   tmp_path):
    """有人手改 candidates.json 塞进风控参数 → 落地必须被拦。"""
    st = env["store"]
    c = Candidate.create(
        source="param_search", title="手改的候选",
        changes=[Change("guard.max_portfolio_drawdown_pct", 8.0, 50.0)])
    st.add(c)
    st.set_status(c.id, ST_APPROVED, by="attacker")

    with pytest.raises(ValueError, match="风控硬线"):
        _runner(env, _snap()).apply(c.id)

    assert ov_mod.load_overrides() == {}


def test_apply_blocks_out_of_range_hand_edit(env, restore_app_config):
    st = env["store"]
    c = Candidate.create(
        source="param_search", title="越界候选",
        changes=[Change("trading_loop.band_pct", 0.25, 9.9)])
    st.add(c)

    with pytest.raises(ValueError):
        _runner(env, _snap()).apply(c.id)


# ------------------------------------------------------------------ 红线 4：回滚

def test_rollback_restores_old_value(env, patched_gate, restore_app_config):
    rep = _runner(env, _snap()).run()
    cid = rep.pending[0]
    r = _runner(env, _snap())
    r.apply(cid)
    assert ov_mod.load_overrides()["trading_loop.band_pct"] == 0.30

    out = r.rollback(cid, by="alice")

    assert out.status == ST_ROLLED_BACK
    assert ov_mod.load_overrides()["trading_loop.band_pct"] == 0.25
    assert r.rollbackable() == []


def test_cannot_rollback_pending(env, patched_gate):
    rep = _runner(env, _snap()).run()
    with pytest.raises(ValueError, match="不能回滚"):
        _runner(env, _snap()).rollback(rep.pending[0])


def test_cannot_rollback_twice(env, patched_gate, restore_app_config):
    rep = _runner(env, _snap()).run()
    cid = rep.pending[0]
    r = _runner(env, _snap())
    r.apply(cid)
    r.rollback(cid)
    with pytest.raises(ValueError):
        r.rollback(cid)


# ------------------------------------------------------------------ 状态视图

def test_status_shape(env, patched_gate):
    st = _runner(env, _snap()).status()
    for k in ("candidates", "pending", "applied", "overrides", "feedback",
              "guard_halted"):
        assert k in st
    assert st["guard_halted"] is False


def test_status_exposes_risk_note(env, patched_gate, restore_app_config):
    rep = _runner(env, _snap()).run()
    st = _runner(env, _snap()).status()
    item = next(x for x in st["pending"] if x["id"] == rep.pending[0])
    assert "方向" in item["risk_note"]


def test_report_render_is_readable(env, patched_gate, restore_app_config):
    rep = _runner(env, _snap()).run()
    text = rep.render()
    assert "进化回合" in text
    assert "待批准" in text


def test_audit_written_to_evolution_log(env, patched_gate, restore_app_config):
    """回合结论要留痕，但**不能**写进交易审计流水。"""
    journal = ROOT / "data" / "trading_journal.jsonl"
    before = journal.read_bytes() if journal.exists() else None

    _runner(env, _snap()).run()

    after = journal.read_bytes() if journal.exists() else None
    assert after == before, "进化回合污染了交易审计日志"

    # 另起一个引擎从磁盘重读：log 是写在文件里的，不是改内存对象
    from src.models.evolution import EvolutionEngine

    logs = EvolutionEngine(state_file=env["evo_path"]).state.get("evolution_log") or []
    assert any(x.get("action") == "evolution_round" for x in logs)


# ------------------------------------------------------------------ 集成：真实闸门

def test_real_gate_end_to_end_with_mappable_change(env, restore_app_config):
    """唯一走**真实闸门 + 真实回测**的用例。

    场景：持仓长期只有 3 只而目标是 10 只 → 生成器 A 提「降到 8 只」。
    ``max_positions`` 是能映射到回测 ``top_n`` 的参数，因此在
    「质量分位单调」的合成面板上（持仓越少越优）应当真的胜出。

    只跑 A：这条用例要证明的是「A 的候选能穿过真实回测闸门」，
    掺进生成器 D（它也用同一块面板搜索）只会让断言变得依赖
    「这个回合一共有几条候选」，与本用例的意图无关。
    """
    r = _runner(env, _underfilled_snap())
    rep = r.run(generators=["param_search"])

    assert rep.n_candidates == 1
    assert rep.rejected == [], f"未通过：{rep.errors}"
    assert len(rep.pending) == 1

    c = env["store"].get(rep.pending[0])
    assert c.changes[0].key == "trading_loop.max_positions"
    assert c.evaluation.evaluator == "gate:backtest"
    assert c.evaluation.candidate["sharpe"] > c.evaluation.baseline["sharpe"]


def test_full_registry_round_includes_the_genetic_generator(env, restore_app_config):
    """全注册表跑一回合 → A 与 D **都**应该产出候选。

    这是修复「生成器 D 永远跳过」之后的**正向**覆盖：面板在磁盘上时，
    GA 必须真的参与进来。修 bug 时最容易漏的一步是「把坏掉的东西接上后，
    没有任何断言能证明它真的动起来了」—— 这条用例就是那个断言。
    """
    rep = _runner(env, _underfilled_snap()).run()

    sources = {env["store"].get(cid).source for cid in rep.pending + rep.rejected}
    assert "genetic" in sources, (
        "生成器 D 没有产出任何候选 —— 它又被静默跳过了？"
        f"（本回合来源：{sources or '无'}）"
    )
    assert "param_search" in sources, f"生成器 A 没动静（来源：{sources or '无'}）"


def test_real_gate_rejects_when_no_panel(env):
    """没有面板 → 真实闸门判 gate:no-data，候选进 rejected。"""
    rep = _runner(env, _underfilled_snap(), panels=False).run()
    assert rep.pending == []
    assert env["store"].get(rep.rejected[0]).evaluation.evaluator == "gate:no-data"


# ------------------------------------------------------------------ 集成：LLM 生成器接入回路

def _with_llm(monkeypatch, payload):
    """把 REGISTRY 里的 llm_review 换成注入了假 LLM 的版本。"""
    import json as _json

    from src.evolution.generators import REGISTRY
    from src.evolution.generators import llm_review as lr

    text = payload if isinstance(payload, str) else _json.dumps(
        payload, ensure_ascii=False)
    monkeypatch.setitem(
        REGISTRY, "llm_review",
        lambda snap: lr.generate(snap, chat_fn=lambda s, u: text))


def test_llm_generator_flows_through_the_round(env, patched_gate,
                                              restore_app_config, monkeypatch):
    """LLM 复盘产出的候选，和生成器 A 走**同一条**闸门 → 队列流水线。"""
    _with_llm(monkeypatch, {"verdict": "换手偏高", "actions": [
        {"action": "raise_min_trade", "confidence": 0.9,
         "reason": "碎单偏多", "evidence": "notional"}]})

    rep = _runner(env, _snap()).run(generators=["llm_review"])

    assert rep.n_candidates == 1
    assert len(rep.pending) == 1
    c = env["store"].get(rep.pending[0])
    assert c.source == "llm_review"
    assert c.changes[0].key == "trading_loop.min_trade_amount"
    assert c.changes[0].new == pytest.approx(6000.0)     # 5000 + step 1000


def test_cross_generator_dedupe_keeps_one(env, patched_gate,
                                         restore_app_config, monkeypatch):
    """两个生成器提同一件事（同为 band_pct 0.30）→ 只留一条。

    这正是统一架构的价值：不用在每个生成器里各写一遍去重。
    """
    from src.evolution.generators import REGISTRY
    import src.evolution.generators.param_search as ps
    import src.evolution.generators.llm_review as lr

    # 生成器 A 的「换手过高」处方也是 band_pct 0.25 → 0.30
    _with_llm(monkeypatch, {"actions": [
        {"action": "widen_band", "confidence": 0.9, "reason": "换手 5.2 倍"}]})

    rep = _runner(env, _snap()).run(generators=["param_search", "llm_review"])

    assert rep.n_candidates == 1                      # 第二条被指纹去重
    assert any("已有同类候选" in s["reason"] for s in rep.skipped)
    assert ps.NAME in REGISTRY and lr.NAME in REGISTRY


def test_llm_generator_failure_does_not_break_other_generators(
        env, patched_gate, restore_app_config, monkeypatch):
    """LLM 抛异常 → 记进 errors，但生成器 A 照常产出。"""
    from src.evolution.generators import REGISTRY

    def boom(snap):
        raise RuntimeError("LLM 服务 503")

    monkeypatch.setitem(REGISTRY, "llm_review", boom)

    rep = _runner(env, _snap()).run(generators=["param_search", "llm_review"])

    assert rep.n_candidates == 1                      # A 的候选仍在
    assert any("llm_review" in e for e in rep.errors)


# ------------------------------------------------------------------ 集成：重训候选（自带证据）

def _retrain_cand(passed: bool = True, evaluator: str = "retrain") -> Candidate:
    ev = Evaluation(
        passed=passed, at="2026-10-14T13:00:00", evaluator=evaluator,
        baseline={"rank_ic": 0.10}, candidate={"rank_ic": 0.16},
        delta={"rank_ic": 0.06},
        reasons=["触发重训：准确率下滑", "样本外 RankIC 提升"],
        window="walk-forward 样本外 120 天",
    )
    return Candidate.create(
        source="retrain", title="上线重训模型 chl",
        changes=[Change("ml.active_version", "inc", "chl")],
        rationale="证据见 meta",
        meta={"signature": "ml.active_version='chl'",
              "evidence": ev.to_dict(), "incumbent": "inc",
              "challenger": "chl"})


def test_retrain_candidate_goes_pending_with_valid_evidence(env, restore_app_config,
                                                           monkeypatch):
    """模型候选走自带证据通道 → 校验通过 → 进待批准队列（等人工批准）。"""
    from src.evolution.generators import REGISTRY

    monkeypatch.setitem(REGISTRY, "retrain", lambda snap: [_retrain_cand()])

    rep = _runner(env, _snap()).run(generators=["retrain"])

    assert rep.n_candidates == 1
    assert len(rep.pending) == 1
    c = env["store"].get(rep.pending[0])
    assert c.evaluation.evaluator == "gate:supplied"
    assert c.evaluation.passed is True
    assert c.status == ST_PENDING


def test_retrain_candidate_rejected_on_forged_evidence(env, monkeypatch):
    """生成器自称闸门判定（evaluator=gate:backtest）→ 自带证据校验必须拦下。"""
    from src.evolution.generators import REGISTRY

    monkeypatch.setitem(
        REGISTRY, "retrain",
        lambda snap: [_retrain_cand(passed=True, evaluator="gate:backtest")])

    rep = _runner(env, _snap()).run(generators=["retrain"])

    assert rep.pending == []
    assert len(rep.rejected) == 1
    c = env["store"].get(rep.rejected[0])
    assert c.status == ST_REJECTED
    assert any("白名单" in r for r in c.evaluation.reasons)


def test_retrain_model_swap_never_auto_applies(env, restore_app_config, monkeypatch):
    """即便开了 --auto-apply，换模型也绝不自动落地（auto_adopt=False）。"""
    from src.evolution.generators import REGISTRY

    monkeypatch.setitem(REGISTRY, "retrain", lambda snap: [_retrain_cand()])

    rep = _runner(env, _snap()).run(generators=["retrain"], auto_apply=True)

    assert rep.auto_applied == []
    assert len(rep.pending) == 1                      # 仍可人工批准
    assert ov_mod.load_overrides() == {}


# ---------------------------------------------------------------------------
# 生成器参数透传（--ga-* 之类）
# ---------------------------------------------------------------------------

def test_generator_kwargs_are_forwarded(env, monkeypatch):
    """透传的参数必须真的落到生成器手上。"""
    from src.evolution.generators import REGISTRY

    seen = {}

    def fake_gen(snapshot, *, population=10):
        seen["population"] = population
        return []

    monkeypatch.setitem(REGISTRY, "fake", fake_gen)
    rep = _runner(env, _snap()).run(
        generators=["fake"],
        generator_kwargs={"fake": {"population": 42}})

    assert seen == {"population": 42}
    assert not rep.errors


def test_unknown_generator_kwarg_is_reported_not_swallowed(env, monkeypatch):
    """参数名写错必须当场报出来，而不是被静默忽略。

    静默忽略的代价：用户以为自己把搜索预算调大了 4 倍，实际什么都没变，
    然后因为「结果没差别」而怀疑进化引擎根本没起作用。
    """
    from src.evolution.generators import REGISTRY

    called = {"n": 0}

    def fake_gen(snapshot, *, population=10):
        called["n"] += 1
        return []

    monkeypatch.setitem(REGISTRY, "fake", fake_gen)
    rep = _runner(env, _snap()).run(
        generators=["fake"],
        generator_kwargs={"fake": {"populatoin": 42}})   # 故意拼错

    assert called["n"] == 0, "参数名错了就不该照常把生成器跑一遍"
    assert any("populatoin" in e for e in rep.errors), \
        f"错误没有被报出来：{rep.errors}"


def test_generator_kwargs_default_to_nothing(env, monkeypatch):
    """不传 kwargs 时生成器按自己的默认值跑（行为与从前一致）。"""
    from src.evolution.generators import REGISTRY

    seen = {}

    def fake_gen(snapshot, *, population=10, generations=4):
        seen.update(population=population, generations=generations)
        return []

    monkeypatch.setitem(REGISTRY, "fake", fake_gen)
    rep = _runner(env, _snap()).run(generators=["fake"])

    assert seen == {"population": 10, "generations": 4}
    assert not rep.errors


# ---------------------------------------------------------------------------
# runner 持有的 provider 必须也交给生成器
# ---------------------------------------------------------------------------

def test_runner_hands_its_provider_to_generators_that_accept_it(env, monkeypatch):
    """接受 ``provider`` 的生成器，必须拿到 runner 手里那一份。

    回归背景（实跑发现）：面板明明在磁盘上，闸门（走 ``self.provider``）
    拿真实夏普否决了生成器 A 的候选，**同一个回合**里生成器 D 却说
    「没有样本外面板」而跳过 —— 因为 evolve.py 只把 provider 注入到 runner，
    没有注入生成器。同一块面板必须贯穿全链路，否则两个部件会各说各话。
    """
    from src.evolution.generators import REGISTRY

    seen = {}

    def fake_gen(snapshot, *, provider=None):
        seen["provider"] = provider
        return []

    monkeypatch.setitem(REGISTRY, "fake", fake_gen)
    _runner(env, _snap()).run(generators=["fake"])

    assert seen["provider"] is not None, \
        "生成器没拿到 runner 的 provider —— 又会各自去猜面板在不在"
    assert seen["provider"](), "塞过去的那份 provider 应该是可用的"


def test_runner_does_not_force_provider_on_generators_without_it(env, monkeypatch):
    """不接受 ``provider`` 的生成器照旧跑，不能因为多塞参数而崩。"""
    from src.evolution.generators import REGISTRY

    called = {"n": 0}

    def fake_gen(snapshot):
        called["n"] += 1
        return []

    monkeypatch.setitem(REGISTRY, "fake", fake_gen)
    rep = _runner(env, _snap()).run(generators=["fake"])

    assert called["n"] == 1
    assert not rep.errors, f"多塞参数把生成器打崩了：{rep.errors}"


def test_runner_without_provider_leaves_the_generator_default_alone(env, monkeypatch):
    """runner 本身没有 provider 时，不要硬塞一个 ``None`` 进去。

    区别很重要：显式传 ``provider=None`` 与「压根不传」在生成器那边
    可能是两种语义（前者等于「我知道没面板」，后者等于「用你的默认」）。
    生成器 D 的 ``provider or gate.default_panel_provider`` 恰好把两者
    当成一回事，但不该指望所有生成器都这么宽容。
    """
    from src.evolution.generators import REGISTRY

    seen = {}

    def fake_gen(snapshot, *, provider="UNSET"):
        seen["provider"] = provider
        return []

    monkeypatch.setitem(REGISTRY, "fake", fake_gen)
    EvolutionRunner(store=env["store"], provider=None,
                    snapshot_fn=lambda: _snap()).run(generators=["fake"])

    assert seen["provider"] == "UNSET", "runner 没有 provider 时不该塞一个 None"


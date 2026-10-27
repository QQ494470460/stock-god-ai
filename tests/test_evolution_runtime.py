# -*- coding: utf-8 -*-
"""自我进化 · 运行期覆盖层与候选队列测试。

这一层负责「**真正把改动写进系统**」和「**一键回滚**」，
风险最高，因此测试重点在：

1. **先全校验、后统一写** —— 一批改动里只要有一条非法，
   磁盘上必须分毫未动（不能出现"改了一半"的半成品状态）。
2. **风控硬线写不进去** —— 即便有人手改 JSON 塞进 guard 参数，
   `load_overrides` / `refresh_app_config` 也必须把它拦掉。
3. **回滚幂等与状态守卫** —— 未落地的候选不许回滚；
   已回滚的不许再回滚（防止 old 被反复套用）。
4. **零污染** —— 全程不碰真实 ``config/evolution_overrides.json``
   与 ``data/evolution/candidates.json``。
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evolution import overrides as ov_mod  # noqa: E402
from src.evolution.store import CandidateStore  # noqa: E402
from src.evolution.types import (  # noqa: E402
    Candidate, Change, Evaluation, ST_APPLIED, ST_AUTO_APPLIED, ST_PENDING,
    ST_ROLLED_BACK,
)

# 真实文件（用于零污染断言）
REAL_OVERRIDES = ROOT / "config" / "evolution_overrides.json"
REAL_CANDIDATES = ROOT / "data" / "evolution" / "candidates.json"


def _default_band() -> float:
    """``band_pct`` 的出厂默认值 —— 从配置里取，不在测试里硬编码。

    ⚠️ 曾经这里写死 ``0.25``，结果 2026-10-27 把它调到 0.15 后，
    「重置回默认」这类测试全线假失败（被测逻辑没错，是断言过期了）。
    默认值属于会随参数扫描结论演进的东西，测试只该断言「回到了默认」，
    不该断言「默认是多少」。
    """
    from config.settings import AppConfig

    return AppConfig().trading_loop.band_pct


# ------------------------------------------------------------------ 夹具

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """把覆盖层文件、候选队列、进化引擎全部重定向到 tmp_path。"""
    from src.models.evolution import EvolutionEngine

    ov_file = tmp_path / "evolution_overrides.json"
    monkeypatch.setattr(ov_mod, "OVERRIDES_FILE", ov_file)

    evo = EvolutionEngine(state_file=tmp_path / "evolution_state.json")
    monkeypatch.setattr(ov_mod, "_evolution_engine", lambda: evo)

    return {
        "ov_file": ov_file,
        "evo": evo,
        "store_path": tmp_path / "candidates.json",
    }


@pytest.fixture
def restore_app_config():
    """快照并复原全局 ``config`` 单例 —— 覆盖层测试会改它。

    没有这一层，某个测试把 ``band_pct`` 改成 0.3 就会污染
    后面所有读 ``config`` 的测试（同一 pytest 进程共享单例）。
    """
    from config.settings import config as app_config

    snapshot = copy.deepcopy(app_config)
    yield app_config
    for name in snapshot.__dataclass_fields__:
        setattr(app_config, name, copy.deepcopy(getattr(snapshot, name)))


def _no_pollution():
    return (
        REAL_OVERRIDES.read_bytes() if REAL_OVERRIDES.exists() else None,
        REAL_CANDIDATES.read_bytes() if REAL_CANDIDATES.exists() else None,
    )


# ------------------------------------------------------------------ 覆盖层读写

def test_save_and_load_roundtrip(sandbox):
    ov_mod.save_overrides({"trading_loop.band_pct": 0.3})
    assert ov_mod.load_overrides() == {"trading_loop.band_pct": 0.3}


def test_load_returns_inner_overrides_not_metadata(sandbox):
    """回归：曾经返回顶层字典（含 _comment），导致「写了读不回」。"""
    ov_mod.save_overrides({"trading_loop.band_pct": 0.3})
    raw = json.loads(sandbox["ov_file"].read_text(encoding="utf-8"))

    assert raw["overrides"] == {"trading_loop.band_pct": 0.3}
    assert "_comment" in raw                     # 文件里仍有元数据
    loaded = ov_mod.load_overrides()
    assert "_comment" not in loaded
    assert "_updated_at" not in loaded


def test_load_tolerates_legacy_flat_format(sandbox):
    """兼容早期平铺写法（无 overrides 键）。"""
    sandbox["ov_file"].write_text(
        json.dumps({"_comment": "old", "trading_loop.band_pct": 0.4}),
        encoding="utf-8")
    assert ov_mod.load_overrides() == {"trading_loop.band_pct": 0.4}


def test_load_tolerates_corrupt_file(sandbox):
    """文件损坏只能当空处理，绝不能让整个系统起不来。"""
    sandbox["ov_file"].write_text("{ 这不是 JSON", encoding="utf-8")
    assert ov_mod.load_overrides() == {}


def test_load_missing_file_is_empty(sandbox):
    assert not sandbox["ov_file"].exists()
    assert ov_mod.load_overrides() == {}


def test_overrides_summary_reports_state(sandbox, restore_app_config):
    s0 = ov_mod.overrides_summary()
    assert s0["exists"] is False

    ov_mod.save_overrides({"trading_loop.band_pct": 0.3})
    s1 = ov_mod.overrides_summary()
    assert s1["exists"] is True
    assert s1["overrides"] == {"trading_loop.band_pct": 0.3}
    assert s1["updated_at"]


# ------------------------------------------------------------------ 落地

def test_apply_changes_writes_and_refreshes(sandbox, restore_app_config):
    applied = ov_mod.apply_changes([Change("trading_loop.band_pct", 0.25, 0.30)])

    assert applied == ["trading_loop.band_pct"]
    assert ov_mod.load_overrides()["trading_loop.band_pct"] == 0.30
    # 立即生效，无需重启
    assert restore_app_config.trading_loop.band_pct == 0.30


def test_apply_changes_is_atomic_on_invalid_param(sandbox, restore_app_config):
    """先全校验后统一写：第二批非法 → 第一批也不许落盘。"""
    ov_mod.save_overrides({"trading_loop.band_pct": 0.25})

    with pytest.raises(Exception):
        ov_mod.apply_changes([
            Change("trading_loop.min_trade_amount", 5000.0, 8000.0),   # 合法
            Change("guard.max_portfolio_drawdown_pct", 0.08, 0.50),    # 风控硬线
        ])

    assert ov_mod.load_overrides() == {"trading_loop.band_pct": 0.25}
    assert restore_app_config.trading_loop.min_trade_amount == 5000.0


def test_apply_changes_rejects_out_of_range(sandbox, restore_app_config):
    with pytest.raises(Exception):
        ov_mod.apply_changes([Change("trading_loop.band_pct", 0.25, 9.9)])
    assert ov_mod.load_overrides() == {}


def test_refresh_skips_handwritten_protected_entries(sandbox, restore_app_config):
    """有人手改 JSON 塞风控参数 → 刷新时必须拦下，合法参数照常生效。"""
    sandbox["ov_file"].write_text(json.dumps({
        "_comment": "手改",
        "overrides": {
            "trading_loop.band_pct": 0.35,
            "guard.max_portfolio_drawdown_pct": 0.99,
            "trading_loop.max_orders_per_tick": 9999,      # 风控叶子名
            "trading_loop.不存在": 1,
        },
    }, ensure_ascii=False), encoding="utf-8")

    applied = ov_mod.refresh_app_config()

    assert applied == ["trading_loop.band_pct"]
    assert restore_app_config.trading_loop.band_pct == 0.35
    assert restore_app_config.trading_loop.max_orders_per_tick != 9999


def test_apply_changes_cannot_bypass_guard_via_load_path(sandbox, restore_app_config):
    """核心断言：覆盖层**不是**绕过风控的后门。"""
    with pytest.raises(Exception, match="风控硬线"):
        ov_mod.apply_changes([
            Change("guard.max_single_position_pct", 0.10, 0.80),
        ])
    assert ov_mod.load_overrides() == {}


# ------------------------------------------------------------------ evolution.* 分流

def test_evolution_params_go_to_state_not_overrides(sandbox, restore_app_config):
    """``evolution.*`` 归 EvolutionEngine 管，避免两套真相。"""
    applied = ov_mod.apply_changes([
        Change("evolution.min_score_base", 50.0, 55.0),
    ])

    assert applied == ["evolution.min_score_base"]
    assert sandbox["evo"].state["min_score_base"] == 55.0
    assert "evolution.min_score_base" not in ov_mod.load_overrides()


def test_read_effective_sees_both_namespaces(sandbox, restore_app_config):
    ov_mod.apply_changes([
        Change("trading_loop.band_pct", 0.25, 0.30),
        Change("evolution.min_score_base", 50.0, 52.0),
    ])
    assert ov_mod.read_effective("trading_loop.band_pct") == 0.30
    assert ov_mod.read_effective("evolution.min_score_base") == 52.0


def test_read_effective_rejects_protected(sandbox, restore_app_config):
    with pytest.raises(Exception, match="风控硬线"):
        ov_mod.read_effective("guard.max_portfolio_drawdown_pct")


def test_reset_all_restores_evolution_ns_default(sandbox, restore_app_config):
    """回归：``evolution.*`` 走 state，重置也必须回写默认值。"""
    ov_mod.apply_changes([Change("evolution.min_score_base", 50.0, 58.0)])
    assert sandbox["evo"].state["min_score_base"] == 58.0

    ov_mod.reset_all()

    assert sandbox["evo"].state["min_score_base"] == 50.0


def test_refresh_alone_does_not_restore_defaults(sandbox, restore_app_config):
    """明确语义：refresh 只做「加」，撤销覆盖是 reset_all 的职责。

    这条不是缺陷而是契约 —— 免得有人误以为删 key 就能退回默认值。
    """
    ov_mod.apply_changes([Change("trading_loop.band_pct", 0.25, 0.40)])
    ov_mod.save_overrides({})

    assert ov_mod.refresh_app_config() == []
    assert restore_app_config.trading_loop.band_pct == 0.40      # 仍在覆盖态

    ov_mod._restore_defaults(["trading_loop.band_pct"])
    assert restore_app_config.trading_loop.band_pct == _default_band()


def test_reset_all_clears_overrides(sandbox, restore_app_config):
    """重置 = 清空文件 + 把内存里的值写回默认（缺了后半步就是假重置）。"""
    ov_mod.apply_changes([Change("trading_loop.band_pct", 0.25, 0.40)])
    assert restore_app_config.trading_loop.band_pct == 0.40

    cleared = ov_mod.reset_all()

    assert cleared == ["trading_loop.band_pct"]
    assert ov_mod.load_overrides() == {}
    assert restore_app_config.trading_loop.band_pct == _default_band()   # 恢复默认


# ------------------------------------------------------------------ 候选队列

def _cand(**kw) -> Candidate:
    base = dict(
        source="param_search", title="测试候选",
        changes=[Change("trading_loop.band_pct", 0.25, 0.30)],
    )
    base.update(kw)
    return Candidate.create(**base)


def test_store_add_get_all(sandbox):
    st = CandidateStore(sandbox["store_path"])
    c = st.add(_cand())

    assert st.get(c.id) is c
    assert [x.id for x in st.all()] == [c.id]
    assert st.summary()["total"] == 1


def test_store_persistence_across_instances(sandbox):
    st = CandidateStore(sandbox["store_path"])
    c = st.add(_cand())

    st2 = CandidateStore(sandbox["store_path"])
    reloaded = st2.get(c.id)
    assert reloaded is not None
    assert reloaded.changes[0].key == "trading_loop.band_pct"
    assert reloaded.changes[0].old == 0.25


def test_store_roundtrip_keeps_evaluation(sandbox):
    st = CandidateStore(sandbox["store_path"])
    c = _cand()
    c.evaluation = Evaluation(passed=True, evaluator="backtest",
                              delta={"sharpe": 0.3},
                              reasons=["样本外胜出"])
    st.add(c)

    reloaded = CandidateStore(sandbox["store_path"]).get(c.id)
    assert reloaded.evaluation.passed is True
    assert reloaded.evaluation.evaluator == "backtest"
    assert reloaded.evaluation.delta["sharpe"] == 0.3


def test_store_status_filters(sandbox):
    st = CandidateStore(sandbox["store_path"])
    a = st.add(_cand(title="a", source="param_search"))
    b = st.add(_cand(title="b", source="genetic"))
    c = st.add(_cand(title="c", source="retrain"))

    st.mark_applied(a.id)
    st.mark_applied(b.id, auto=True)
    st.reject(c.id, by="tester", reason="样本外不显著")

    assert {x.id for x in st.pending()} == set()
    assert {x.id for x in st.applied()} == {a.id, b.id}
    assert a.status == ST_APPLIED and b.status == ST_AUTO_APPLIED
    assert st.summary()["pending"] == 0
    assert st.summary()["counts"][ST_APPLIED] == 1
    assert st.summary()["counts"]["rejected"] == 1


def test_has_pending_for_dedupes_generator(sandbox):
    """同一生成器已有一条待批候选时，不再重复堆叠。"""
    st = CandidateStore(sandbox["store_path"])
    assert st.has_pending_for("genetic") is False

    st.add(_cand(source="genetic"))
    assert st.has_pending_for("genetic") is True

    st.mark_applied(st.all()[0].id)
    assert st.has_pending_for("genetic") is False


def test_rollback_uses_recorded_old_values(sandbox, restore_app_config):
    """回滚依据是 Change.old，不是快照文件 —— 落地后回滚必须还原原值。"""
    st = CandidateStore(sandbox["store_path"])
    before = ov_mod.read_effective("trading_loop.band_pct")

    c = st.add(_cand(changes=[Change("trading_loop.band_pct", before, 0.45)]))
    ov_mod.apply_changes(c.changes)
    st.mark_applied(c.id)
    assert ov_mod.read_effective("trading_loop.band_pct") == 0.45

    st.mark_rolled_back(c.id, by="tester")
    ov_mod.apply_changes([Change(ch.key, ch.new, ch.old) for ch in c.changes])

    assert ov_mod.read_effective("trading_loop.band_pct") == before
    assert st.get(c.id).status == ST_ROLLED_BACK
    assert st.get(c.id).rolled_back_at


def test_rollbackable_only_contains_applied(sandbox):
    st = CandidateStore(sandbox["store_path"])
    p = st.add(_cand(title="待批准"))
    a = st.add(_cand(title="已落地"))
    st.mark_applied(a.id)

    assert [x.id for x in st.rollbackable()] == [a.id]
    assert p.id not in [x.id for x in st.rollbackable()]


def test_cannot_rollback_pending_candidate(sandbox):
    """未落地的候选没有可回滚的东西 —— 必须报错而不是静默通过。"""
    st = CandidateStore(sandbox["store_path"])
    c = st.add(_cand())
    assert c.status == ST_PENDING

    with pytest.raises(ValueError, match="不能回滚"):
        st.mark_rolled_back(c.id)


def test_cannot_rollback_twice(sandbox):
    """二次回滚会把 old 反复套用 —— 必须拦住。"""
    st = CandidateStore(sandbox["store_path"])
    c = st.add(_cand())
    st.mark_applied(c.id)
    st.mark_rolled_back(c.id)

    with pytest.raises(ValueError):
        st.mark_rolled_back(c.id)


def test_set_status_unknown_id_raises(sandbox):
    st = CandidateStore(sandbox["store_path"])
    with pytest.raises(KeyError):
        st.set_status("不存在", ST_APPLIED)


# ------------------------------------------------------------------ 零污染

def test_no_pollution_of_real_files(sandbox, restore_app_config):
    """全程快照真实文件，跑完必须一字未改。"""
    before = _no_pollution()

    st = CandidateStore(sandbox["store_path"])
    c = st.add(_cand())
    ov_mod.apply_changes(c.changes)
    st.mark_applied(c.id)
    ov_mod.reset_all()

    assert _no_pollution() == before, "测试污染了真实的配置/候选文件"


def test_store_default_path_uses_data_dir_constant():
    """默认路径必须挂在 DATA_DIR 下（打包后可外挂到 exe 同级）。"""
    from config.settings import DATA_DIR
    from src.evolution import store as store_mod

    assert str(store_mod.CANDIDATES_FILE).startswith(str(DATA_DIR))
    assert Path(DATA_DIR) / "evolution" / "candidates.json" == store_mod.CANDIDATES_FILE


def test_overrides_default_path_uses_config_dir_constant():
    """默认路径必须在 CONFIG_DIR 下（重建 exe 不丢）。"""
    from config.settings import CONFIG_DIR
    from src.evolution import overrides as m

    assert Path(CONFIG_DIR) / "evolution_overrides.json" == m.OVERRIDES_FILE

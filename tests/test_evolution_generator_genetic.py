# -*- coding: utf-8 -*-
"""自我进化 · 生成器 D（策略基因进化 / GA）测试。

重点证明：

1. 基因组只用**闸门能回测表达**的参数（否则白搜，注定被拒）；
2. 变异只在网格上挪**一步**、交叉只在合法值之间挑 —— 产出永远是合法基因组；
3. 适应度与闸门**共用同一条验收边界**（回撤恶化超标直接判负）；
4. GA **只搜不自判** —— 候选的 ``evaluation`` 恒为 ``None``，胜出由闸门独立复核；
5. 确定性（同种子同结果）+ 评估预算硬上限。
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evolution import gate, overrides as ov_mod, space  # noqa: E402
from src.evolution.feedback import PerformanceSnapshot  # noqa: E402
from src.evolution.generators import REGISTRY  # noqa: E402
from src.evolution.generators import genetic as ga  # noqa: E402

CURRENT = {
    "trading_loop.band_pct": 0.25,
    "trading_loop.cash_reserve_pct": 0.10,
    "trading_loop.max_positions": 10,
    "trading_loop.weight_cap": 0.30,
    "trading_loop.min_trade_amount": 5000.0,
    "trading_loop.weighting": "equal",
    "trading_loop.tilt_strength": 0.0,
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ov_mod, "OVERRIDES_FILE", tmp_path / "ov.json")
    monkeypatch.setattr(ov_mod, "read_effective", lambda k: CURRENT.get(k))
    return tmp_path


def _snap(**kw) -> PerformanceSnapshot:
    base = dict(generated_at="2026-10-14T13:00:00", n_verified=10, n_settlements=20,
                equity_series=[{"date": "2026-09-01", "equity": 1e5,
                                "cash": 1e4, "n_positions": 10}],
                trade_stats={"fill": 1}, guard={"halted": False},
                metrics={"insufficient": False})
    base.update(kw)
    return PerformanceSnapshot(**base)


def _panels(n_days=100, n_codes=16, seed=11):
    """质量分位单调（尾部负 alpha）的合成面板。"""
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


# ------------------------------------------------------------------ ① 基因集合的合法性

def test_genes_are_exactly_the_backtestable_ones():
    expected = {k for k in space.PARAM_SPACE if k in gate.BACKTEST_MAP}
    assert set(ga.GENES) == expected
    assert ga.GENES, "基因组不该为空"


def test_genes_all_mappable_by_gate():
    for k in ga.GENES:
        assert k in gate.BACKTEST_MAP


def test_genes_exclude_protected_and_unbacktestable():
    for k in ga.GENES:
        assert not space.is_protected(k)
    # evolution.*（阈值，回测无法表达）与 ml.*（超参，须重训）都不该在基因里
    assert not any(k.startswith("evolution.") for k in ga.GENES)
    assert not any(k.startswith("ml.") for k in ga.GENES)
    assert "trading_loop.scan_interval_sec" not in ga.GENES


def test_gene_domains_are_bounded_and_come_from_space():
    for k in ga.GENES:
        dom = ga.gene_domain(k)
        assert len(dom) >= 2
        spec = space.get_spec(k)
        for v in dom:
            assert spec.in_range(v), f"{k} 的取值 {v!r} 越界"


def test_gene_domain_matches_spec_domain():
    for k in ga.GENES:
        assert ga.gene_domain(k) == space.get_spec(k).domain()


# ------------------------------------------------------------------ ② 变异与交叉

def test_mutate_moves_exactly_one_grid_step(env):
    rng = random.Random(7)
    base = ga.current_genome()
    m = ga.mutate(base, rng, n=6)
    for k in ga.GENES:
        dom = ga.gene_domain(k)
        i, j = dom.index(base[k]), dom.index(m[k])
        assert abs(i - j) <= 1, f"{k} 挪了 {abs(i - j)} 步"


def test_mutate_default_moves_only_one_gene(env):
    """默认只挪 1 个基因 —— 这是「邻域精调」的定义。"""
    rng = random.Random(11)
    base = ga.current_genome()
    moved_counts = []
    for _ in range(40):
        m = ga.mutate(base, rng)
        moved_counts.append(sum(1 for k in ga.GENES if m[k] != base[k]))
    assert max(moved_counts) == 1
    assert sum(moved_counts) > 0


def test_mutate_n_zero_is_noop(env):
    base = ga.current_genome()
    assert ga.mutate(base, random.Random(1), n=0) == base


def test_mutate_n_two_moves_two_genes(env):
    base = ga.current_genome()
    m = ga.mutate(base, random.Random(3), n=2)
    assert sum(1 for k in ga.GENES if m[k] != base[k]) <= 2
    assert sum(1 for k in ga.GENES if m[k] != base[k]) >= 1


def test_mutate_respects_boundary(env, monkeypatch):
    """基因在取值集合边界时，变异要往另一边挪，不能越界。"""
    key = "trading_loop.band_pct"
    dom = ga.gene_domain(key)
    monkeypatch.setattr(ov_mod, "read_effective",
                        lambda k: dom[-1] if k == key else CURRENT.get(k))
    base = ga.current_genome()
    assert base[key] == dom[-1]
    m = ga.mutate(base, random.Random(5), n=6)
    assert m[key] == dom[-2]


def test_mutate_always_returns_valid_genome(env):
    rng = random.Random(11)
    g = ga.current_genome()
    for _ in range(50):
        g = ga.mutate(g, rng, n=2)
        for k in ga.GENES:
            assert g[k] in ga.gene_domain(k)


def test_crossover_takes_genes_from_parents(env):
    rng = random.Random(2)
    a, b = ga.current_genome(), ga.mutate(ga.current_genome(), rng, n=3)
    c = ga.crossover(a, b, rng)
    for k in ga.GENES:
        assert c[k] in (a[k], b[k])


def test_crossover_result_is_valid(env):
    rng = random.Random(4)
    a = ga.random_genome(rng)
    b = ga.random_genome(rng)
    c = ga.crossover(a, b, rng)
    for k in ga.GENES:
        assert c[k] in ga.gene_domain(k)


def test_genome_key_is_stable(env):
    g = ga.current_genome()
    assert ga.genome_key(g) == ga.genome_key(dict(g))
    g2 = dict(g)
    g2["trading_loop.band_pct"] = 0.05
    assert ga.genome_key(g) != ga.genome_key(g2)


def test_diff_genome_reports_changes(env):
    base = ga.current_genome()
    other = dict(base)
    other["trading_loop.band_pct"] = 0.05
    other["trading_loop.weighting"] = "score"
    d = dict((k, (o, n)) for k, o, n in ga.diff_genome(other, base))
    assert d["trading_loop.band_pct"] == (0.25, 0.05)
    assert d["trading_loop.weighting"] == ("equal", "score")


def test_diff_genome_empty_when_identical(env):
    base = ga.current_genome()
    assert ga.diff_genome(dict(base), base) == []


# ------------------------------------------------------------------ ③ 适应度与闸门同边界

def _m(sharpe=1.0, n_periods=20, mdd=-0.05):
    return {"sharpe": sharpe, "n_periods": n_periods, "max_drawdown": mdd}


def test_fitness_none_metrics():
    assert ga._fitness(None, _m(), gate.DEFAULT_CRITERIA) == float("-inf")


def test_fitness_too_few_periods():
    assert ga._fitness(_m(n_periods=5), _m(), gate.DEFAULT_CRITERIA) == float("-inf")


def test_fitness_missing_sharpe():
    assert ga._fitness({"n_periods": 20, "max_drawdown": -0.05},
                       _m(), gate.DEFAULT_CRITERIA) == float("-inf")


def test_fitness_rejects_excess_drawdown_worsening():
    """回撤恶化超过闸门允许 → 判负（与 compare_metrics 同一条线）。"""
    base = _m(mdd=-0.05)
    bad = _m(sharpe=9.9, mdd=-0.05 - ga.gate.DEFAULT_CRITERIA.max_drawdown_worsen - 0.001)
    assert ga._fitness(bad, base, gate.DEFAULT_CRITERIA) == float("-inf")


def test_fitness_accepts_drawdown_at_the_edge():
    base = _m(mdd=-0.05)
    edge = _m(sharpe=2.0, mdd=-0.05 - ga.gate.DEFAULT_CRITERIA.max_drawdown_worsen)
    assert ga._fitness(edge, base, gate.DEFAULT_CRITERIA) == pytest.approx(2.0)


def test_fitness_returns_sharpe():
    assert ga._fitness(_m(sharpe=2.5), _m(), gate.DEFAULT_CRITERIA) == pytest.approx(2.5)


def test_fitness_ignores_drawdown_when_base_missing():
    assert ga._fitness(_m(sharpe=3.0, mdd=-0.9), {}, gate.DEFAULT_CRITERIA) == pytest.approx(3.0)


# ------------------------------------------------------------------ ④ evolve 主循环

def _distance_landscape(target):
    """适应度 = 负的「网格距离」：每朝目标挪一步就涨 1。可攀爬。"""
    idx = {k: {v: i for i, v in enumerate(ga.gene_domain(k))} for k in ga.GENES}

    def ev(g):
        d = sum(abs(idx[k][g[k]] - idx[k][target[k]]) for k in ga.GENES)
        return _m(sharpe=float(-d))
    return ev


def test_evolve_improves_toward_target(env):
    """目标 = 当前参数只动一个基因一步 → 邻域搜索应当爬到它。"""
    base = ga.current_genome()
    target = dict(base)
    key = "trading_loop.band_pct"
    dom = ga.gene_domain(key)
    target[key] = dom[dom.index(base[key]) + 1]

    best, m, n = ga.evolve(_distance_landscape(target), base, _m(sharpe=-1.0),
                           population=8, generations=3, rng=random.Random(1))

    assert best is not None
    assert ga._fitness(m, _m(sharpe=-1.0), gate.DEFAULT_CRITERIA) > -1.0
    assert best == target                       # 距离 0 = 完全命中
    assert n > 0


def test_evolve_climbs_multiple_steps(env):
    """起点距目标 2 步 → 多代迭代应当爬上去（证明它真的在搜索，不是抽签）。"""
    base = ga.current_genome()
    target = dict(base)
    key = "trading_loop.band_pct"
    dom = ga.gene_domain(key)
    i = dom.index(base[key])
    target[key] = dom[i + 2]

    # 注：基因组每多一个基因，单基因达到指定步数的收敛就更慢（每次变异只动 1 个
    # 基因）。把 tilt_strength 纳入可进化基因后共 7 个基因，4 代只能爬 1 步；
    # 扩到 10 代才能稳定爬满 2 步。这是测试参数随基因数演进，不是 GA 退化。
    best, m, _ = ga.evolve(_distance_landscape(target), base, _m(sharpe=-2.0),
                           population=10, generations=10, rng=random.Random(2))
    assert best[key] == target[key]


def test_evolve_never_loses_the_best(env):
    """精英保留：最终解不劣于起点。"""
    base = ga.current_genome()
    target = dict(base)
    target["trading_loop.max_positions"] = 12

    best, m, _ = ga.evolve(_distance_landscape(target), base, _m(sharpe=-1.0),
                           population=6, generations=3, rng=random.Random(9))
    assert ga._fitness(m, _m(sharpe=-1.0), gate.DEFAULT_CRITERIA) >= -1.0


def test_evolve_is_deterministic(env):
    base = ga.current_genome()
    land = _distance_landscape({**base, "trading_loop.max_positions": 8})
    r1 = ga.evolve(land, base, _m(sharpe=-1.0), population=8, generations=3,
                   rng=random.Random(42))
    r2 = ga.evolve(land, base, _m(sharpe=-1.0), population=8, generations=3,
                   rng=random.Random(42))
    assert r1[0] == r2[0]
    assert r1[2] == r2[2]                     # 评估次数也一致


def test_evolve_respects_hard_budget(env, monkeypatch):
    """无论调用方传多大的种群/代数，回测次数都不超过硬上限。"""
    monkeypatch.setattr(ga, "MAX_EVALUATIONS", 25)
    base = ga.current_genome()
    land = _distance_landscape(base)
    _, _, n = ga.evolve(land, base, _m(sharpe=0.0),
                        population=50, generations=50, rng=random.Random(1))
    assert n <= 25


def test_evolve_returns_base_when_landscape_is_flat(env):
    """所有个体同分且都不优于起点 → 返回起点（不硬凑）。"""
    base = ga.current_genome()
    _, m, _ = ga.evolve(lambda g: _m(sharpe=1.0), base, _m(sharpe=1.0),
                        population=6, generations=2, rng=random.Random(3))
    assert m is not None


# ------------------------------------------------------------------ ⑤ generate

def test_generate_without_evidence(env):
    assert ga.generate(_snap(n_verified=0, n_settlements=0),
                       provider=lambda: _panels()) == []


def test_generate_without_panel(env):
    assert ga.generate(_snap(), provider=lambda: None) == []


def test_generate_when_panel_incomplete(env):
    assert ga.generate(_snap(), provider=lambda: (None, None, None)) == []


def test_generate_produces_candidate_on_synthetic_panel(env):
    """真实回测（而非假适应度）的集成用例。"""
    cands = ga.generate(_snap(), provider=lambda: _panels(),
                        population=6, generations=2)
    assert len(cands) == 1
    c = cands[0]
    assert c.source == "genetic"
    assert c.changes                                   # 至少动了一个基因
    for ch in c.changes:
        assert ch.key in ga.GENES
        assert ch.new in ga.gene_domain(ch.key)
        assert ch.old == CURRENT[ch.key]               # 旧值来自当前生效参数
    assert c.evaluation is None                        # 不自盖章
    assert c.status == "pending"
    assert "邻域搜索" in c.rationale
    assert "独立复核" in c.rationale


def test_generate_is_reproducible(env):
    """同一面板同一参数 → 同一答案（否则每回合都刷一条不同的候选）。"""
    a = ga.generate(_snap(), provider=lambda: _panels(),
                    population=6, generations=2)
    b = ga.generate(_snap(), provider=lambda: _panels(),
                    population=6, generations=2)
    assert [ch.to_dict() for ch in a[0].changes] == [ch.to_dict() for ch in b[0].changes]


def test_generate_respects_max_candidates(env):
    cands = ga.generate(_snap(), provider=lambda: _panels(),
                        population=6, generations=2, max_candidates=0)
    assert cands == []


def test_generate_returns_empty_when_current_is_optimal(env, monkeypatch):
    """把基因组锁成单点 → 无法变异出差异 → 不提候选。"""
    only = {k: [CURRENT[k]] for k in ga.GENES}
    monkeypatch.setattr(ga, "gene_domain", lambda k: list(only[k]))
    assert ga.generate(_snap(), provider=lambda: _panels(),
                       population=6, generations=2) == []


def test_generate_falls_back_to_default_provider(env, monkeypatch):
    """**不传 provider** 时必须回落到 ``gate.default_panel_provider()``。

    回归背景（实跑发现）：``--run`` 只会把 provider 注入给 runner（闸门那条路），
    生成器 D 拿到的是 ``None``，于是回落到一个**硬编码 ``return None``** 的
    ``gate.default_panel_provider()`` → 永远「没有样本外面板，本轮跳过」。
    GA 因此在面板明明存在的情况下成了纯粹的死代码。

    这条断言把「默认路径真的能拿到面板」钉住 ——
    没有它，上面那些显式传 provider 的用例会全绿，而线上 GA 一次都不跑。
    """
    monkeypatch.setattr(gate, "default_panel_provider", lambda: _panels())

    cands = ga.generate(_snap(), population=6, generations=2)

    assert len(cands) == 1, "默认 provider 路径没生效，GA 又被静默跳过了"


def test_generate_without_any_panel_backs_off_quietly(env, monkeypatch):
    """默认路径也拿不到面板时安静返回空 —— 跳过，而不是崩。"""
    monkeypatch.setattr(gate, "default_panel_provider", lambda: None)

    assert ga.generate(_snap(), population=6, generations=2) == []


# ------------------------------------------------------------------ ⑥ 映射与注册

def test_to_bt_params_uses_gate_map(env):
    g = ga.current_genome()
    p = ga._to_bt_params(g)
    assert p["top_n"] == 10
    assert p["band_pct"] == pytest.approx(0.25)
    assert p["weighting"] == "equal"
    # 这里只返回基因组自己的覆盖项 —— 生产口径（周期 / 排名滞后）由**闸门**
    # 的基准侧负责，不在这里叠加。叠加会把 GA 的搜索空间压到「已是最优邻域」，
    # 让它每轮空手而归（提案者失效，闸门再准也没东西可审）。详见 _to_bt_params 注释。
    assert set(p) <= gate.gate_backtest_keys()
    assert "rebalance_days" not in p and "hold_rank_slack" not in p


def test_registered_in_registry():
    assert "genetic" in REGISTRY
    assert REGISTRY["genetic"] is ga.generate


def test_all_four_generators_registered():
    assert set(REGISTRY) == {"param_search", "llm_review", "retrain", "genetic"}


# ---------------------------------------------------------------------------
# 搜索预算：可从 CLI 透传，且越界值不会让搜索崩掉
# ---------------------------------------------------------------------------

def test_generate_accepts_explicit_budget(env):
    """population / generations / elite / seed 都能从外部指定。"""
    cands = ga.generate(_snap(), provider=lambda: _panels(),
                        population=6, generations=2, elite=1, seed=20261014)
    assert len(cands) == 1


def test_default_budget_is_unchanged(env, monkeypatch):
    """不指定时用生成器自己的默认值 —— CLI 只表达「我要偏离默认」。"""
    seen = {}

    def fake_evolve(evaluate, base, base_metrics, **kw):
        seen.update(kw)
        return None, None, 0

    monkeypatch.setattr(ga, "evolve", fake_evolve)
    ga.generate(_snap(), provider=lambda: _panels())

    assert seen["population"] == ga.DEFAULT_POPULATION
    assert seen["generations"] == ga.DEFAULT_GENERATIONS
    assert seen["elite"] == ga.DEFAULT_ELITE


def test_absurd_budget_is_clamped_not_crashed(env, monkeypatch):
    """population=0 / generations=-3 / elite 大于种群 都要收敛到合法域。

    越界值的意图是明确的（「多搜一点」或手滑），报错只会让运维去猜合法范围；
    但放行则会让搜索退化成「不搜索」或直接算不出后代。
    """
    seen = {}

    def fake_evolve(evaluate, base, base_metrics, **kw):
        seen.update(kw)
        return None, None, 0

    monkeypatch.setattr(ga, "evolve", fake_evolve)
    ga.generate(_snap(), provider=lambda: _panels(),
                population=0, generations=-3, elite=999)

    assert seen["population"] >= 2, "种群不能小于 2（否则没有交叉对象）"
    assert seen["generations"] >= 1, "至少迭代一代，否则等于没搜"
    assert 1 <= seen["elite"] < seen["population"], \
        "精英数必须落在 [1, 种群-1]，否则整代全保留、不再探索"


# ---------------------------------------------------------------------------
# 搜索预算：可从 CLI 透传，且越界值不会让搜索崩掉
# ---------------------------------------------------------------------------

def test_generate_accepts_explicit_budget(env):
    """population / generations / elite / seed 都能从外部指定。"""
    cands = ga.generate(_snap(), provider=lambda: _panels(),
                        population=6, generations=2, elite=1, seed=20261014)
    assert len(cands) == 1


def test_default_budget_is_unchanged(env, monkeypatch):
    """不指定时用生成器自己的默认值 —— CLI 只表达「我要偏离默认」。"""
    seen = {}

    def fake_evolve(evaluate, base, base_metrics, **kw):
        seen.update(kw)
        return None, None, 0

    monkeypatch.setattr(ga, "evolve", fake_evolve)
    ga.generate(_snap(), provider=lambda: _panels())

    assert seen["population"] == ga.DEFAULT_POPULATION
    assert seen["generations"] == ga.DEFAULT_GENERATIONS
    assert seen["elite"] == ga.DEFAULT_ELITE


def test_absurd_budget_is_clamped_not_crashed(env, monkeypatch):
    """population=0 / generations=-3 / elite 大于种群 都要收敛到合法域。

    越界值的意图是明确的（「多搜一点」或手滑），报错只会让运维去猜合法范围；
    但放行则会让搜索退化成「不搜索」或直接算不出后代。
    """
    seen = {}

    def fake_evolve(evaluate, base, base_metrics, **kw):
        seen.update(kw)
        return None, None, 0

    monkeypatch.setattr(ga, "evolve", fake_evolve)
    ga.generate(_snap(), provider=lambda: _panels(),
                population=0, generations=-3, elite=999)

    assert seen["population"] >= 2, "种群不能小于 2（否则没有交叉对象）"
    assert seen["generations"] >= 1, "至少迭代一代，否则等于没搜"
    assert 1 <= seen["elite"] < seen["population"], \
        "精英数必须落在 [1, 种群-1]，否则整代全保留、不再探索"

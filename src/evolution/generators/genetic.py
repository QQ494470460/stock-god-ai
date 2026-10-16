# -*- coding: utf-8 -*-
"""自我进化 · 生成器 D：策略基因进化（遗传算法）。

它和生成器 A 的分工
====================
生成器 A 是「单个症状 → 单个处方」：换手高就动 ``band_pct``，一次只动一个基因。
它做不了**组合**优化 —— 而参数之间是有交互的：放宽容忍带（省手续费）会改变
最优持仓只数；降低现金保留（提高收益）又会让回撤逼近熔断线。
需要同时调几个基因才能找到更好的组合，这时就该 GA 上场。

三条设计约束（都为了「别乱搜」）
================================

**① 基因组只用闸门能回测表达的参数。**
:data:`GENES` 直接从 ``space.PARAM_SPACE ∩ gate.BACKTEST_MAP`` 推导。
不这么做的后果很具体：GA 辛苦搜出来的组合一旦含 ``scan_interval_sec``
这种回测无法表达的参数，闸门会按「整批不可评估」直接驳回（见
``gate.evaluate`` 的 ``unmappable`` 分支）—— 白搜。

**② 种群从当前参数出发做邻域搜索，不做全空间随机搜索。**
搜索空间是 ``11×6×10×50×20×2 ≈ 132 万`` 种组合；用几十次回测做全空间搜索
等于随机抽样，结果不可复现也不可解释。所以第 0 号个体**就是当前生效参数**，
其余个体是它的邻域变异，交叉只在精英之间发生 —— 于是产出永远是
「在当前配置附近的一次精调」：每个基因与起点最多相差约
``generations`` 个步长，人一眼能判断幅度是否合理。

变异算子刻意是「每次挪定 1 个基因（偶尔 2 个）」而不是「每个基因按概率挪」：
后者在 6 个基因上按 0.5 概率会每次挪 3 个，后代离父代的平均距离恒定 ——
那是随机漂移，朝更优方向**爬不了坡**（这是实测踩到的坑，不是理论担心）。

**③ 适应度与闸门共用同一条验收边界。**
:func:`_fitness` 直接把「回撤比基准恶化超过 ``max_drawdown_worsen``」判为
不可接受（返回 ``-inf``），和 ``gate.compare_metrics`` 用的是同一个阈值。
这样搜出来的冠军天然落在闸门的可接受域内，而不是搜完再被闸门驳回。

**④ 确定性。**
默认随机种子由面板区间推导 —— 同一块面板、同一份当前参数，重跑得到同一个
答案。否则每个回合都会提出一个不同的基因组，把待批队列刷成噪声。

职责边界：GA **只搜、只提候选**。胜出与否仍由 ``gate.evaluate`` 独立做 A/B
回测判定 —— 搜索者不能给自己当裁判。
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from src.evolution import gate, space
from src.evolution.feedback import PerformanceSnapshot
from src.evolution.generators.base import atomize, require_evidence
from src.evolution.types import Candidate

NAME = "genetic"

#: 基因组：可被回测表达的可进化参数（由参数空间与闸门映射求交推导）。
GENES: Tuple[str, ...] = tuple(sorted(
    k for k in space.PARAM_SPACE if k in gate.BACKTEST_MAP))

#: 默认搜索预算（每次迭代 = 一次全样本回测，所以刻意取小）。
DEFAULT_POPULATION = 10
DEFAULT_GENERATIONS = 4
DEFAULT_ELITE = 2
#: 后代相对父代挪几个基因（1 = 标准邻域搜索；偶尔 2 增加一点探索）。
DEFAULT_MUTATION_N = 1
EXPLORE_N = 2
#: 有多大概率做 2 基因跳跃（其余情况挪 1 个）。
EXPLORE_P = 0.25

#: 硬上限：无论调用方传什么，单个回合的回测次数不得超过它。
MAX_EVALUATIONS = 240


# ---------------------------------------------------------------------------
# 基因：随机、交叉、变异（全部离线可测）
# ---------------------------------------------------------------------------

def gene_domain(key: str) -> List[Any]:
    """某个基因的合法取值集合（来自参数空间，天然有界）。"""
    return space.get_spec(key).domain()


def current_genome() -> Dict[str, Any]:
    """当前生效参数构成的基因组（第 0 号个体）。"""
    from src.evolution.overrides import read_effective

    out: Dict[str, Any] = {}
    for key in GENES:
        spec = space.get_spec(key)
        try:
            out[key] = spec.clamp(read_effective(key))
        except Exception:  # noqa: BLE001
            out[key] = spec.clamp(spec.default)
    return out


def random_genome(rng: random.Random) -> Dict[str, Any]:
    """在参数空间里均匀采一个基因组（仅用于兜底初始化）。"""
    out: Dict[str, Any] = {}
    for key in GENES:
        dom = gene_domain(key)
        out[key] = rng.choice(dom) if dom else space.get_default(key)
    return out


def mutate(genome: Dict[str, Any], rng: random.Random, n: int = 1,
           ) -> Dict[str, Any]:
    """把 ``n`` 个随机基因各挪**一格**（邻域搜索的标准算子）。

    为什么是「固定挪几个」而不是「每个基因按概率挪」：
    后者在 6 个基因上按 0.5 概率会每次挪 3 个 —— 等于随机漂移，
    后代离父代的平均距离恒定，「朝更优方向爬坡」根本发生不了
    （实测：单基因之差的目标，几十次评估都爬不上去）。
    固定挪 1~2 格才能让每一代**收敛**，也才配得上「从当前配置出发的邻域精调」。

    基因值只会在取值集合（网格）上移动，且不越界。
    """
    out = dict(genome)
    movable = [k for k in GENES if len(gene_domain(k)) > 1]
    if not movable or n <= 0:
        return out
    for key in rng.sample(movable, min(int(n), len(movable))):
        dom = gene_domain(key)
        try:
            i = dom.index(out.get(key))
        except ValueError:
            out[key] = rng.choice(dom)
            continue
        step = rng.choice([-1, 1])
        j = min(max(i + step, 0), len(dom) - 1)
        if j == i:                      # 已在边界 → 往另一边
            j = min(max(i - step, 0), len(dom) - 1)
        out[key] = dom[j]
    return out


def crossover(a: Dict[str, Any], b: Dict[str, Any],
              rng: random.Random) -> Dict[str, Any]:
    """均匀交叉：每个基因独立地从两个父代里挑一个。

    不做算术混合（取平均）是因为参数**离散且带步长**：两个合法值的中点
    未必落在网格上，还得再夹一次。均匀交叉的结果天然合法。
    """
    return {k: (a[k] if rng.random() < 0.5 else b.get(k, a[k])) for k in GENES}


def genome_key(g: Dict[str, Any]) -> Tuple:
    """基因组的可哈希指纹（用于去重与评估缓存）。"""
    return tuple((k, g.get(k)) for k in GENES)


def diff_genome(genome: Dict[str, Any], base: Dict[str, Any],
                ) -> List[Tuple[str, Any, Any]]:
    """基因组相对基准的差异 → ``[(key, old, new), ...]``。"""
    out = []
    for key in GENES:
        old, new = base.get(key), genome.get(key)
        if old != new:
            out.append((key, old, new))
    return out


# ---------------------------------------------------------------------------
# 适应度
# ---------------------------------------------------------------------------

def _num(m: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not m:
        return None
    try:
        v = float(m.get(key))
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _fitness(metrics: Optional[Dict[str, Any]],
             base_metrics: Optional[Dict[str, Any]],
             criteria: gate.GateCriteria) -> float:
    """适应度 = 夏普；但**先过两道与闸门一致的否决线**。

    否决线（回去看 ``gate.compare_metrics``，用的是同一组阈值）：

    1. 调仓期数不足 → ``-inf``（样本太短，指标不可信）；
    2. 回撤比基准恶化超过 ``max_drawdown_worsen`` → ``-inf``
       （收益再好也不接受 —— 这是本系统的安全取向）。

    两条都不满足才返回夏普。这样 GA 的搜索方向与闸门的验收方向一致。
    """
    if not metrics:
        return float("-inf")
    if (_num(metrics, "n_periods") or 0) < criteria.min_periods:
        return float("-inf")
    sharpe = _num(metrics, "sharpe")
    if sharpe is None:
        return float("-inf")

    mdd = _num(metrics, "max_drawdown")
    base_mdd = _num(base_metrics, "max_drawdown")
    if mdd is not None and base_mdd is not None:
        # 回撤是负数；恶化 = 更负
        if mdd < base_mdd - criteria.max_drawdown_worsen - 1e-12:
            return float("-inf")
    return sharpe


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def evolve(
    evaluate_genome: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    base_genome: Dict[str, Any],
    base_metrics: Optional[Dict[str, Any]] = None,
    *,
    population: int = DEFAULT_POPULATION,
    generations: int = DEFAULT_GENERATIONS,
    elite: int = DEFAULT_ELITE,
    rng: Optional[random.Random] = None,
    criteria: gate.GateCriteria = gate.DEFAULT_CRITERIA,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], int]:
    """跑一轮遗传算法（从 ``base_genome`` 出发的邻域搜索）。

    Args:
        evaluate_genome: ``genome -> metrics``。测试注入假函数即可完全离线。
        base_genome: 当前生效参数（第 0 号个体，同时也是适应度基准）。
        base_metrics: 当前参数的指标（用于回撤否决线）。缺省则先评估一次。

    Returns:
        ``(best_genome, best_metrics, n_evaluations)``；都不可行时 best 为 None。
    """
    rng = rng or random.Random(0)
    population = max(2, int(population))
    generations = max(1, int(generations))
    elite = max(1, min(int(elite), population - 1))

    cache: Dict[Tuple, Tuple[float, Optional[Dict[str, Any]]]] = {}
    budget = {"n": 0}

    def fit(g: Dict[str, Any]) -> float:
        k = genome_key(g)
        if k in cache:
            return cache[k][0]
        if budget["n"] >= MAX_EVALUATIONS:
            return float("-inf")           # 预算耗尽 → 之后一律判负
        m = evaluate_genome(g)
        budget["n"] += 1
        f = _fitness(m, base_metrics, criteria)
        cache[k] = (f, m)
        return f

    def offspring(parent: Dict[str, Any]) -> Dict[str, Any]:
        n = EXPLORE_N if rng.random() < EXPLORE_P else DEFAULT_MUTATION_N
        return mutate(parent, rng, n=n)

    # ---- 初始化：当前参数 + 其一步邻域 ----
    if base_metrics is None:
        base_metrics = evaluate_genome(base_genome)
        budget["n"] += 1
    cache[genome_key(base_genome)] = (
        _fitness(base_metrics, base_metrics, criteria), base_metrics)

    pop: List[Dict[str, Any]] = [dict(base_genome)]
    seen = {genome_key(base_genome)}
    guard = 0
    while len(pop) < population and guard < population * 30:
        cand = offspring(base_genome)
        guard += 1
        k = genome_key(cand)
        if k in seen:
            continue
        seen.add(k)
        pop.append(cand)
    while len(pop) < population:           # 邻域已取尽 → 随机补
        cand = random_genome(rng)
        k = genome_key(cand)
        if k in seen:
            break
        seen.add(k)
        pop.append(cand)

    best_g, best_f = dict(base_genome), fit(base_genome)

    for gen in range(generations):
        scored = sorted(((fit(g), g) for g in pop), key=lambda x: -x[0])
        if scored and scored[0][0] > best_f:
            best_f, best_g = scored[0][0], dict(scored[0][1])

        elites = [dict(g) for f, g in scored[:elite] if _finite(f)]
        if not elites:                     # 全体不可行 → 用当前参数续命
            elites = [dict(base_genome)]

        nxt: List[Dict[str, Any]] = list(elites)      # 精英保留
        seen = {genome_key(g) for g in nxt}
        guard = 0
        while len(nxt) < population and guard < population * 40:
            guard += 1
            if len(elites) >= 2 and rng.random() < 0.6:
                child = crossover(rng.choice(elites), rng.choice(elites), rng)
                child = mutate(child, rng,
                               n=EXPLORE_N if rng.random() < EXPLORE_P
                               else DEFAULT_MUTATION_N)
            else:
                child = offspring(rng.choice(elites))
            k = genome_key(child)
            if k in seen:
                continue
            seen.add(k)
            nxt.append(child)
        pop = nxt
        if budget["n"] >= MAX_EVALUATIONS:
            logger.info(f"🧬 生成器 D：评估预算用尽（{budget['n']} 次），停止迭代")
            break

    # 末代再收一次
    for g in pop:
        f = fit(g)
        if f > best_f:
            best_f, best_g = f, dict(g)

    return best_g, cache.get(genome_key(best_g), (None, None))[1], budget["n"]


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and not math.isinf(x) and not math.isnan(x)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def generate(
    snapshot: PerformanceSnapshot,
    provider: Optional[gate.PanelProvider] = None,
    *,
    population: int = DEFAULT_POPULATION,
    generations: int = DEFAULT_GENERATIONS,
    elite: int = DEFAULT_ELITE,
    seed: Optional[int] = None,
    max_candidates: int = 1,
) -> List[Candidate]:
    """GA 搜索 → 候选。

    Args:
        provider: 样本外面板提供者；没有面板就没法评估，直接返回空。
        population / generations / elite: 搜索预算。默认值刻意很小
            （约 35–45 次回测）—— 每个个体都是一次完整回测，调大前请先确认
            面板规模与机器性能。这三个值可从 ``evolve.py`` 的 ``--ga-*`` 传入。
        seed: 随机种子。缺省由面板区间派生 → 同一块面板结果可复现。
    """
    if not require_evidence(snapshot):
        return []

    # 参数卫生：CLI 传进来的 0 / 负数不能让搜索直接崩或退化成「不搜索」。
    # 这里收敛到合法域而不报错 —— 越界值的意图是明确的（「多搜一点」），
    # 报错只会让运维去猜合法范围。
    population = max(2, int(population))
    generations = max(1, int(generations))
    elite = max(1, min(int(elite), population - 1)) if population > 2 else 1

    prov = provider or gate.default_panel_provider
    panel = prov() if callable(prov) else None
    if not panel:
        logger.info("🧬 生成器 D：没有样本外面板，无法评估适应度，本轮跳过")
        return []

    score_panel, close_panel, bench = (list(panel) + [None, None, None])[:3]
    if score_panel is None or close_panel is None:
        logger.info("🧬 生成器 D：面板内容不完整，本轮跳过")
        return []

    if seed is None:
        seed = int(hashlib.md5(
            str(gate._window(score_panel)).encode("utf-8")).hexdigest()[:8], 16)

    base_genome = current_genome()
    base_params = _to_bt_params(base_genome)
    base_metrics = gate._run(score_panel, close_panel, bench, base_params)
    if not base_metrics:
        logger.info("🧬 生成器 D：基准参数跑不出回测结果，本轮跳过")
        return []

    def evaluate_genome(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return gate._run(score_panel, close_panel, bench, _to_bt_params(g))

    best, best_metrics, n_eval = evolve(
        evaluate_genome, base_genome, base_metrics,
        population=population, generations=generations, elite=elite,
        rng=random.Random(seed))

    logger.info(f"🧬 生成器 D：搜索完成（{n_eval} 次回测，seed={seed}），"
                f"基准夏普 {_f(base_metrics, 'sharpe')}，"
                f"最优夏普 {_f(best_metrics, 'sharpe')}")

    if not best or not best_metrics:
        return []

    diffs = diff_genome(best, base_genome)
    if not diffs:
        logger.info("🧬 生成器 D：最优个体与当前参数一致 —— 当前配置已是最优邻域，"
                    "按兵不动")
        return []

    passed, reasons, _delta = gate.compare_metrics(base_metrics, best_metrics)
    if not passed:
        logger.info(f"🧬 生成器 D：最优个体未达闸门门槛"
                    f"（{(reasons or ['?'])[0]}），不提候选 —— "
                    "避免制造注定被拒的噪声")
        return []

    sb, sc = _f(base_metrics, "sharpe"), _f(best_metrics, "sharpe")
    why = "；".join(reasons[:2])
    title = f"GA 精调组合参数（{'、'.join(k.rsplit('.', 1)[-1] for k, _, _ in diffs)}）"
    rationale = (
        f"遗传算法在样本外面板上从当前参数出发做了邻域搜索（{n_eval} 次回测，"
        f"seed={seed}）：\n"
        f"　夏普 {sb} → {sc}；{why}\n"
        f"　改动 {len(diffs)} 个基因："
        + "；".join(f"{k.rsplit('.', 1)[-1]} {o!r}→{n!r}" for k, o, n in diffs)
        + "\n　注意：GA 只负责**搜索**，胜出与否由闸门用同一条验收边界独立复核。"
    )
    return atomize(NAME, [(title, {k: n for k, _, n in diffs}, rationale)])[:max_candidates]


def _to_bt_params(genome: Dict[str, Any]) -> Dict[str, Any]:
    """基因组 → ``run_backtest`` 形参（复用闸门的映射表，保持单一来源）。"""
    from src.evolution.types import Change

    overrides, unmappable = gate.map_changes(
        [Change(key=k, old=None, new=v) for k, v in genome.items()])
    if unmappable:
        logger.warning(f"🧬 生成器 D：以下基因无法映射到回测（已忽略）：{unmappable}")
    return overrides


def _f(metrics: Optional[Dict[str, Any]], key: str) -> str:
    v = _num(metrics, key)
    return "n/a" if v is None else f"{v:.3f}"

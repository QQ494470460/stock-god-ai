# -*- coding: utf-8 -*-
"""自我进化 · 样本外验证闸门。

为什么必须有闸门
================
没有闸门的「自我进化」= 用噪声改策略。单批验证准确率涨到 0.65 就放宽阈值，
下一批跌到 0.40 再收紧 —— 参数在噪声里来回抖，手续费倒是实实在在付出去了。

所以任何候选**必须先在样本外区间上与「当前参数」正面对比并胜出**，
才允许进入待批准队列。

三层判定，逐层收紧
==================

1. **风控硬线（前置，一票否决）**
   候选里只要有任意一项命中黑名单（熔断线 / 单票上限 / 下单频率 / 止盈止损…），
   立即判负，**连回测都不跑** —— 不给「跑一跑也许能过」留口子。
2. **可评估性（宁可不通过，也不放水）**
   回测引擎吃的是「评分面板」（date × code → score）。
   像 ``band_pct`` / ``min_trade_amount`` 这类**不改评分**的参数，
   回测根本看不出差别。此时闸门必须**明确判负并说明原因**，
   而不是因为「差异为 0」就误判为通过 —— 那是最危险的假阳性。
3. **指标对比**
   夏普须有实质提升、收益不得倒退、回撤不得恶化、调仓期数须够多。
4. **walk-forward 多折样本外**（可选，``require_walk_forward``）
   按时间切折，候选须在多数折上同时不输基准 —— 防「全样本好看、实盘拉胯」。
5. **回测-实盘一致性**（可选，``require_live_consistency``）
   拿 ``data/performance.json`` 里的真实夏普校验回测夏普。前四层全在模拟里
   打转，这一层是唯一的外部参照：背离过大即说明撮合假设失真，此时 A/B 出来的
   「好 0.05 夏普」是模拟的产物而不是策略的改进，整批拒绝放行。
   真实样本不足 ``min_live_days`` 时只报告不拦截 —— 用几天业绩否决候选，
   等于用噪声管噪声。业绩摘要**无论开关如何**都会写进判定依据。

生成器的分工
============
* 生成器 A（参数寻优）/ D（遗传算法）：改的是组合参数 → 走本闸门的 A/B 回测。
* 生成器 B（LLM 复盘）：往往产出定性假设 → 必须落成具体数值改动才可评估。
* 生成器 C（模型重训）：换模型会**改变评分面板本身** → 无法用同一套分数 A/B，
  必须自带 walk-forward 样本外证据。本模块提供
  :func:`verify_supplied_evaluation` 来校验「自带证据」是否达标，
  防止生成器自己给自己盖章。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from src.evolution import space
from src.evolution.types import Candidate, Change, Evaluation

__all__ = [
    "GateCriteria", "DEFAULT_CRITERIA", "BACKTEST_MAP", "PanelProvider",
    "risk_precheck", "map_changes", "compare_metrics", "evaluate",
    "evaluate_candidate", "verify_supplied_evaluation", "apply_verdict",
    "load_criteria", "load_live_performance",
]

#: 闸门配置文件的默认位置（项目根 config/）。可被 load_criteria(path=...) 覆盖（测试用）。
_GATE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "evolution_gate.json"

#: 真实业绩报告（由 src/trading/performance.py 的 build_report 落盘）。
#: 进化闸门消费它是为了让「是否采纳候选」不再只看回测 —— 回测是模拟，
#: 业绩是真金白银（纸盘也是同一套撮合与风控口径）。
_LIVE_PERF_PATH = Path(__file__).resolve().parents[2] / "data" / "performance.json"


# ---------------------------------------------------------------------------
# 判定标准
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateCriteria:
    """候选要「胜出」必须满足的量化门槛。

    默认值刻意保守：宁可漏掉真改进，也不接受噪声。
    """
    #: 样本外夏普相对当前参数的**最小提升**（绝对差）
    min_sharpe_gain: float = 0.05
    #: 年化收益的最小提升（可为负，表示允许小幅倒退换取夏普上升）
    min_return_gain: float = 0.0
    #: 最大回撤允许恶化的幅度（绝对值，与回撤同量纲；回撤以负数表示）
    max_drawdown_worsen: float = 0.02
    #: 样本外至少覆盖多少个调仓期（太少则统计上不可信）
    min_periods: int = 12
    #: 是否要求年化波动不得上升
    require_no_vol_increase: bool = False
    #: 是否启用 walk-forward 多折样本外验证（防过拟合）。默认关闭 ——
    #   开启后候选必须在「按时间切分的多个 train/test 折」上整体胜过当前参数，
    #   才允许进入待批准队列。需要可用的评分面板；面板缺失/过短则退化为不拦。
    require_walk_forward: bool = False
    #: walk-forward 至少要在多少折上胜过基准（样本外胜率门槛）
    min_wf_folds: int = 3

    # ---- 第五层：回测-实盘一致性（防「回测很好看、实盘一直亏」的模拟失真）----
    #: 是否启用一致性校验。默认关闭 —— 真实业绩样本不足时它本来就会自动跳过，
    #   但「开」仍必须显式，避免误以为系统已经在校验。
    require_live_consistency: bool = False
    #: 允许的夏普背离上限（回测夏普 − 真实夏普）。超过即认为模拟严重失真，
    #   A/B 回测的差异不再可信 —— 此时任何候选都不该被放行。
    max_live_gap_sharpe: float = 2.0
    #: 真实业绩最少交易日。不足则只报告、不拦截（fail-safe）：
    #   用 4 天的业绩去否决候选，等于用噪声管噪声。
    min_live_days: int = 20

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CRITERIA = GateCriteria()

#: 面板提供者：返回 (score_panel, close_panel, benchmark_close 或 None)
PanelProvider = Callable[[], Optional[Tuple[Any, Any, Optional[Any]]]]


def load_criteria(path=None) -> GateCriteria:
    """从 ``config/evolution_gate.json`` 读取闸门门槛（可选）。

    这是把 Walk-Forward 门禁从「写了代码但永远关着」变成「可一键开关」的接线点：
    生产回路（``scripts/evolve.py``）默认用 :func:`load_criteria` 构造
    :class:`GateCriteria`，而不是硬编码 ``DEFAULT_CRITERIA``。

    ⚠️ 刻意保守的默认：**只要文件里没有显式 ``require_walk_forward: true``，
    就保持关闭**。理由——开启 WF 需要面板足够长（≥ ``max(min_periods*3, 36)`` 天），
    否则 ``_walk_forward_verdict`` 会静默退化为「不拦」，等于开关开了却没用；
    更糟的是如果哪天误把开关打开却忘了先 ``--build-panel``，所有候选会被
    ``gate:no-data`` 静默拒掉。所以「开」必须是显式的、有意为之的动作。

    Args:
        path: 配置文件路径；缺省用 :data:`_GATE_CONFIG_PATH`。

    Returns:
        合并后的 :class:`GateCriteria`。文件缺失 / 损坏 / 没有任何已知字段时，
        回退到 ``DEFAULT_CRITERIA``（WF 关闭）。
    """
    p = Path(path) if path else _GATE_CONFIG_PATH
    if not p.exists():
        return DEFAULT_CRITERIA
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"闸门配置读取失败（用默认门槛）: {e}")
        return DEFAULT_CRITERIA
    if not isinstance(data, dict):
        return DEFAULT_CRITERIA
    known = {k: v for k, v in data.items()
             if k in GateCriteria.__dataclass_fields__}
    if not known:
        return DEFAULT_CRITERIA
    return replace(DEFAULT_CRITERIA, **known)


# ---------------------------------------------------------------------------
# 候选 → 回测参数 的映射
# ---------------------------------------------------------------------------

#: 可被回测引擎直接表达的参数：进化参数 key → (run_backtest 形参名, 类型)
BACKTEST_MAP: Dict[str, Tuple[str, type]] = {
    "trading_loop.max_positions": ("top_n", int),
    "trading_loop.cash_reserve_pct": ("cash_reserve_pct", float),
    "trading_loop.weight_cap": ("weight_cap", float),
    "trading_loop.band_pct": ("band_pct", float),
    "trading_loop.min_trade_amount": ("min_trade_amount", float),
    "trading_loop.weighting": ("weighting", str),
    "trading_loop.tilt_strength": ("tilt_strength", float),
}


def map_changes(changes: List[Change]) -> Tuple[Dict[str, Any], List[str]]:
    """把候选变更翻译成回测参数覆盖。

    Returns:
        (回测参数覆盖字典, 无法映射的 key 列表)
    """
    overrides: Dict[str, Any] = {}
    unmappable: List[str] = []
    for c in changes:
        pair = BACKTEST_MAP.get(c.key)
        if pair is None:
            unmappable.append(c.key)
            continue
        bt_key, caster = pair
        try:
            overrides[bt_key] = caster(c.new)
        except (TypeError, ValueError):
            unmappable.append(c.key)
    return overrides, unmappable


def _current_backtest_params() -> Dict[str, Any]:
    """当前生效参数 → 回测参数（A/B 的基准侧）。

    取不到生效值时**回退到参数空间默认值**，绝不留给 ``run_backtest``
    的隐式默认（``top_n=20``）—— 那会让「基准」变成另一套参数，
    A/B 对比就不再是 A/B。
    """
    from src.evolution.overrides import read_effective

    out: Dict[str, Any] = {}
    for key, (bt_key, caster) in BACKTEST_MAP.items():
        try:
            out[bt_key] = caster(read_effective(key))
        except Exception:  # noqa: BLE001
            try:
                out[bt_key] = caster(space.get_default(key))
            except Exception:  # noqa: BLE001
                logger.warning(f"无法确定 {key} 的基准值，A/B 对比将不可靠")
    return out


# ---------------------------------------------------------------------------
# 第一层：风控硬线（一票否决）
# ---------------------------------------------------------------------------

def risk_precheck(changes: List[Change]) -> Tuple[List[str], List[str]]:
    """检查候选是否触碰风控硬线。

    Returns:
        ``(阻断项, 提醒项)``

        * 阻断项：命中黑名单 / 非法取值 → 候选直接判负。
        * 提醒项：合法但朝更激进方向 → 不判负，但落地前必须人工批准
          （由 :func:`space.check_changes_adoptable` 决定是否自动采纳）。
    """
    blockers: List[str] = []
    warnings: List[str] = []

    if not changes:
        return ["候选不含任何改动"], []

    for c in changes:
        if space.is_protected(c.key):
            blockers.append(f"{c.key} 属风控硬线，永久不可进化")
            continue
        try:
            spec = space.get_spec(c.key)
        except space.ParamError as e:
            blockers.append(str(e))
            continue
        try:
            space.validate(c.key, c.new)
        except space.ParamError as e:
            blockers.append(str(e))
            continue
        if spec.is_risk_increasing(c.old, c.new):
            warnings.append(
                f"{c.key} 的改动朝更激进方向（{c.old} → {c.new}），"
                "即便胜出也需人工批准")

    return blockers, warnings


# ---------------------------------------------------------------------------
# 第三层：指标对比
# ---------------------------------------------------------------------------

#: 值越小越好的指标（回撤以负数表示，故单独按绝对值处理）
def _num(m: Dict[str, Any], key: str) -> Optional[float]:
    v = (m or {}).get(key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:          # NaN
        return None
    return f


def compare_metrics(base: Dict[str, Any], cand: Dict[str, Any],
                    criteria: GateCriteria = DEFAULT_CRITERIA,
                    ) -> Tuple[bool, List[str], Dict[str, float]]:
    """对比基准与候选的回测指标。

    Returns:
        ``(是否通过, 原因列表, 差值字典)``

    通过当且仅当**存在至少一项达标项，且无任何未达标项**。
    用显式的 ok / fail 两个列表判定，而不是靠原因文案的前缀去猜
    （靠文案猜会在改文案时静默改变判定结果）。
    """
    oks: List[str] = []
    fails: List[str] = []
    delta: Dict[str, float] = {}

    np_base = _num(base, "n_periods")
    np_cand = _num(cand, "n_periods")
    if max(np_base or 0.0, np_cand or 0.0) < criteria.min_periods:
        return False, [
            f"样本外调仓期数不足（基准 {np_base or 0:.0f} / "
            f"候选 {np_cand or 0:.0f}，要求 ≥ {criteria.min_periods}）"
            "—— 统计上不可信"
        ], {}

    base_sharpe, cand_sharpe = _num(base, "sharpe"), _num(cand, "sharpe")
    if cand_sharpe is None:
        return False, ["候选夏普无法计算（样本不足或收益恒定）"], {}
    if base_sharpe is None:
        # 没有可比对象时**不判定胜出**：缺基准比「基准差」更该拒
        return False, ["基准夏普缺失，无法对比 —— 拒绝放行"], {}

    delta["sharpe"] = round(cand_sharpe - base_sharpe, 6)
    if delta["sharpe"] < criteria.min_sharpe_gain:
        fails.append(f"夏普未达门槛：{base_sharpe:.3f} → {cand_sharpe:.3f}"
                     f"（需 ≥ {criteria.min_sharpe_gain:+.3f} 提升）")
    else:
        oks.append(f"夏普提升 {base_sharpe:.3f} → {cand_sharpe:.3f}"
                   f"（{delta['sharpe']:+.3f}）")

    base_ret, cand_ret = _num(base, "ann_return"), _num(cand, "ann_return")
    if base_ret is not None and cand_ret is not None:
        delta["ann_return"] = round(cand_ret - base_ret, 6)
        if delta["ann_return"] < criteria.min_return_gain:
            fails.append(f"年化收益倒退：{base_ret:+.2%} → {cand_ret:+.2%}"
                         f"（要求 ≥ {criteria.min_return_gain:+.2%}）")
        else:
            oks.append(f"年化收益 {base_ret:+.2%} → {cand_ret:+.2%}")

    base_dd, cand_dd = _num(base, "max_drawdown"), _num(cand, "max_drawdown")
    if base_dd is not None and cand_dd is not None:
        worsen = abs(cand_dd) - abs(base_dd)      # 回撤为负值，取绝对值比大小
        delta["max_drawdown"] = round(cand_dd - base_dd, 6)
        if worsen > criteria.max_drawdown_worsen:
            fails.append(f"最大回撤恶化 {abs(base_dd):.2%} → {abs(cand_dd):.2%}"
                         f"（超出允许 {criteria.max_drawdown_worsen:.2%}）")
        else:
            oks.append(f"最大回撤 {abs(base_dd):.2%} → {abs(cand_dd):.2%}")

    if criteria.require_no_vol_increase:
        base_vol, cand_vol = _num(base, "ann_vol"), _num(cand, "ann_vol")
        if base_vol is not None and cand_vol is not None:
            delta["ann_vol"] = round(cand_vol - base_vol, 6)
            if cand_vol > base_vol + 1e-9:
                fails.append(f"年化波动上升 {base_vol:.2%} → {cand_vol:.2%}")
            else:
                oks.append(f"年化波动未上升 {base_vol:.2%} → {cand_vol:.2%}")

    passed = bool(oks) and not fails
    reasons = [f"❌ {r}" for r in fails] + [f"✅ {r}" for r in oks]
    if not fails:
        reasons.insert(0, "✅ 全部指标达标")
    return passed, reasons, delta


# ---------------------------------------------------------------------------
# 第二层 + 总入口
# ---------------------------------------------------------------------------

def evaluate(
    candidate: Candidate,
    provider: Optional[PanelProvider] = None,
    criteria: GateCriteria = DEFAULT_CRITERIA,
    live_perf_path=None,
) -> Evaluation:
    """对一个候选做样本外验证。

    Args:
        candidate: 待验证候选。
        provider: 面板提供者，返回 ``(score_panel, close_panel, benchmark)``。
            缺省尝试 :func:`default_panel_provider`（当前返回 None，
            即「本机没有现成面板」→ 判负而非放行）。
        criteria: 判定门槛。
        live_perf_path: 真实业绩报告路径（缺省 ``data/performance.json``）。
            仅供测试注入，生产无需传。

    Returns:
        :class:`Evaluation`。``evaluator`` 取值为
        ``"gate:hardline"`` / ``"gate:unmappable"`` / ``"gate:backtest"`` /
        ``"gate:no-data"``，便于事后区分「为什么没通过」。
    """
    def _fail(ev: str, why: str, **kw) -> Evaluation:
        return Evaluation(passed=False, at=_now(), evaluator=ev,
                          reasons=[why], **kw)

    # ---- 第一层：风控硬线，一票否决（连回测都不跑）----
    blockers, warnings = risk_precheck(candidate.changes)
    if blockers:
        return _fail("gate:hardline",
                     "触碰风控硬线：" + "；".join(blockers))

    # ---- 第二层：可评估性 ----
    # 只要**有一项**无法评估，整批驳回。理由：不能拿「能测的那条胜出」
    # 去给「没测过的那条」背书 —— 这就是典型的搭便车放行。
    overrides, unmappable = map_changes(candidate.changes)
    if unmappable:
        return Evaluation(
            passed=False, at=_now(), evaluator="gate:unmappable",
            reasons=[
                f"含有无法用回测评估的改动（{'、'.join(unmappable)}）："
                "这些参数不改变评分面板，回测对比无意义。"
                "不允许部分放行 —— 整批候选必须整体可评估。"
                "请拆成独立候选，或由生成器提供自带的样本外证据"
                "（walk-forward 等）。"
            ])

    # ---- 取面板 ----
    prov = provider or default_panel_provider
    try:
        panels = prov()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"样本外面板获取失败: {e}")
        panels = None
    if not panels:
        return Evaluation(
            passed=False, at=_now(), evaluator="gate:no-data",
            reasons=["本机没有可用的样本外面板，无法完成 A/B 对比 → 拒绝放行。"
                     "请先构建评分面板（见 scripts/walk_forward_oos.py）"],
            window="")

    score_panel, close_panel, bench = panels

    # ---- 第三层：A/B 回测 ----
    base_params = _current_backtest_params()
    cand_params = dict(base_params)
    cand_params.update(overrides)

    base_metrics = _run(score_panel, close_panel, bench, base_params)
    if base_metrics is None:
        return _fail("gate:no-data", "基准参数回测失败（面板数据不足）")

    cand_metrics = _run(score_panel, close_panel, bench, cand_params)
    if cand_metrics is None:
        return _fail("gate:no-data", "候选参数回测失败（面板数据不足）")

    # 防「假阳性」：参数若对回测毫无影响，说明映射没生效
    touched = {k: (base_params.get(k), cand_params.get(k)) for k in overrides}
    if all(a == b for a, b in touched.values()):
        return _fail("gate:unmappable",
                     f"候选参数与当前值相同（{touched}），无对比可言")

    passed, reasons, delta = compare_metrics(base_metrics, cand_metrics, criteria)
    if warnings:
        reasons = reasons + ["（提醒）" + w for w in warnings]

    # ---- 第四层（可选）：walk-forward 多折样本外验证（防过拟合）----
    if passed and criteria.require_walk_forward:
        wf_ok, wf_msg = _walk_forward_verdict(
            score_panel, close_panel, bench, base_params, cand_params, criteria)
        # 通过/未通过都写进 reasons：否则「开了 WF 且过关」在审计记录里完全不可见，
        # 事后无法区分「真过了 WF」和「根本没跑 WF」。wf_msg 本身已带
        # 「walk-forward 通过/未通过」前缀，直接留痕即可。
        reasons = reasons + [wf_msg]
        if not wf_ok:
            passed = False

    # ---- 第五层：回测-实盘一致性（消费真实业绩）----
    # 无论通过与否都把依据写进 reasons：真实业绩是这条判定链上唯一的
    # 「非模拟」证据，漏记会让事后复盘看不到当时的现实处境。
    if passed:
        live_ok, live_reasons = _live_consistency_verdict(
            base_metrics, criteria, path=live_perf_path)
        reasons = reasons + live_reasons
        if not live_ok:
            passed = False

    ev = Evaluation(
        passed=passed,
        at=_now(),
        baseline=_clean(base_metrics),
        candidate=_clean(cand_metrics),
        delta=delta,
        reasons=reasons,
        evaluator="gate:backtest",
        window=_window(score_panel),
    )
    logger.info(f"闸门判定 {candidate.id}: {'通过' if passed else '未通过'}"
                f"（{len(reasons)} 条依据）")
    return ev


def _run(score_panel, close_panel, bench, params: Dict[str, Any]) -> Optional[Dict]:
    """跑一次回测。``params`` 里的键直接作为 ``run_backtest`` 的组合参数。"""
    try:
        from src.core.backtest import run_backtest

        kwargs = {k: v for k, v in params.items() if k in gate_backtest_keys()}
        res = run_backtest(score_panel, close_panel, benchmark_close=bench, **kwargs)
        m = dict(res.metrics or {})
        return m if m.get("n_periods") else None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"回测执行失败: {e}")
        return None


def _walk_forward_verdict(
    score_panel, close_panel, bench, base_params, cand_params,
    criteria: GateCriteria, _runner=None,
) -> Tuple[bool, str]:
    """walk-forward 多折样本外验证：防「全样本回测好看、实盘拉胯」的过拟合。

    把时间轴切成若干 ``(train, test)`` 紧邻折，在**每一折的样本外 test 区间**上
    比较「候选参数」与「当前参数」的回测表现；候选须在大多数折上同时不输于
    基准（夏普与超额收益），才认为它真有泛化能力。

    Args:
        _runner: 可注入的回测函数（测试时可 monkeypatch ``gate._run``）；
            默认用模块内的 :func:`_run`。

    Returns:
        ``(通过?, 说明文案)``。面板太短无法切折、或某折回测失败 → 退化为
        ``(True, "")``（不拦），以确保「无面板/面板不足」时行为与旧逻辑一致。
    """
    runner = _runner or _run
    try:
        from src.trading.analytics import walk_forward_splits
        import numpy as np
    except Exception:  # noqa: BLE001
        return True, ""
    n = len(score_panel)
    # 面板至少要够切出「1 个完整的 train+test 折」才值得做
    _min_len = max(criteria.min_periods * 3, 36)
    if n < _min_len:
        return True, ""  # 面板太短，退化为不拦（与旧行为一致）

    folds = walk_forward_splits(
        n_folds=5,
        train_size=max(criteria.min_periods, 12),
        test_size=max(criteria.min_periods // 2, 5),
    )
    folds = [(t, e) for (t, e) in folds if e[1] < n]
    if not folds:
        return True, ""

    wins = 0
    usable = 0
    for (t0, t1), (e0, e1) in folds:
        try:
            sp_te = score_panel.iloc[e0:e1 + 1]
            cp_te = close_panel.iloc[e0:e1 + 1]
            bm_te = bench.iloc[e0:e1 + 1] if bench is not None else None
            base_te = runner(sp_te, cp_te, bm_te, base_params)
            cand_te = runner(sp_te, cp_te, bm_te, cand_params)
        except Exception:  # noqa: BLE001
            continue
        if base_te is None or cand_te is None:
            continue
        usable += 1
        b_sharpe = base_te.get("sharpe") or 0.0
        c_sharpe = cand_te.get("sharpe") or 0.0
        b_excess = base_te.get("top_excess") or 0.0
        c_excess = cand_te.get("top_excess") or 0.0
        if c_sharpe >= b_sharpe and c_excess >= b_excess:
            wins += 1

    if usable == 0:
        return True, ""  # 折都跑不出结果 → 不拦（退化）
    needed = max(1, int(criteria.min_wf_folds))
    ok = wins >= needed
    verdict = "通过" if ok else "未通过"
    # ⚠️ 即使通过也要把结果写进 reasons：否则「WF 门禁开启且候选过关」这件事
    # 在审计记录里完全不可见，事后无法区分「真过了 WF」和「压根没跑 WF」。
    return ok, (
        f"walk-forward {verdict}：样本外胜率 {wins}/{usable}"
        f"（需≥{needed} 折胜过基准）"
    )


def gate_backtest_keys() -> set:
    """``run_backtest`` 能接受的组合参数键（与 BACKTEST_MAP 的值域保持同步）。"""
    return {bt_key for bt_key, _ in BACKTEST_MAP.values()}


# ---------------------------------------------------------------------------
# 第五层：回测-实盘一致性（消费真实业绩）
# ---------------------------------------------------------------------------

def load_live_performance(path=None) -> Optional[Dict[str, Any]]:
    """读取 ``data/performance.json``（业绩归因引擎的产物）。

    缺失 / 损坏 / 不是 dict 一律返回 None —— 门禁据此「只报告不拦截」。
    绝不因为盘上有半截 JSON 就在进化回路上抛异常：进化的失败模式必须是
    「不动」而不是「崩掉」。
    """
    p = Path(path) if path else _LIVE_PERF_PATH
    try:
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"真实业绩报告读取失败（一致性校验跳过）: {e}")
        return None


def _live_summary(perf: Dict[str, Any]) -> str:
    """一行人话摘要，写进每条判定依据里 —— 让「进化决定」天生带真实业绩上下文。"""
    period = perf.get("period") or {}
    eq = perf.get("equity") or {}
    risk = perf.get("risk") or {}
    trades = perf.get("trades") or {}

    def _f(v, unit=""):
        return "—" if v is None else f"{float(v):.2f}{unit}"

    dd = risk.get("max_drawdown_pct")
    dd_s = "—" if dd is None else f"{float(dd):.2f}%"
    return (
        f"（实盘）{period.get('start', '?')}~{period.get('end', '?')} "
        f"共 {period.get('n_days', 0)} 个交易日："
        f"累计 {_f(eq.get('total_return_pct'), '%')}，"
        f"最大回撤 {dd_s}，"
        f"盈亏比 {_f(trades.get('payoff_ratio'))}，"
        f"胜率 {_f((trades.get('win_rate') or 0) * 100, '%') if trades.get('win_rate') is not None else '—'}，"
        f"最长连亏 {trades.get('max_consecutive_losses', '—')} 笔"
    )


def _live_consistency_verdict(
    backtest_metrics: Dict[str, Any],
    criteria: GateCriteria,
    path=None,
) -> Tuple[bool, List[str]]:
    """回测夏普 vs 真实夏普的背离校验。

    为什么需要这一层
    ----------------
    前四层全在**回测**里打转。回测是模拟：滑点按固定值估、成交按收盘价给、
    信号按 T 日收盘计算 T+1 开盘执行。任何一处失真，A/B 对比出来的
    「候选比基准好 0.05 夏普」就可能是模拟假设的产物，而不是策略的改进。
    真实业绩恰恰是这套假设的**校验器**：回测夏普 1.5 而真实夏普 -2，说明
    模拟与现实差了两个数量级，此时拿回测差异去改参数等于在错的坐标系里调参。

    保守边界（三条都不拦）
    ----------------------
    · 报告缺失 / 损坏
    · 真实样本 < ``min_live_days``（4 天的业绩否定不了任何东西）
    · 任一夏普为 None（样本不足 20 日 / 零波动时夏普无定义）
    宁可漏拦，也不让一条「数据还没攒够」就永久冻结进化。

    Returns:
        ``(是否放行, 依据文案列表)``。依据**总会**返回（含仅报告的情况），
        这样即便开关关闭，进化日志里也能看到当时的真实业绩。
    """
    perf = load_live_performance(path)
    if not perf:
        return True, ["（实盘）无业绩报告，一致性校验跳过"]

    reasons = [_live_summary(perf)]

    if not criteria.require_live_consistency:
        return True, reasons

    n_days = int((perf.get("period") or {}).get("n_days") or 0)
    if n_days < int(criteria.min_live_days):
        reasons.append(
            f"一致性校验跳过：真实样本 {n_days} 交易日 < {criteria.min_live_days}，"
            "不足以判定模拟失真")
        return True, reasons

    bt = _num(backtest_metrics, "sharpe")
    live = (perf.get("risk") or {}).get("sharpe")
    if bt is None or live is None:
        reasons.append("一致性校验跳过：回测或真实夏普缺失（样本不足/零波动）")
        return True, reasons

    gap = float(bt) - float(live)
    cap = float(criteria.max_live_gap_sharpe)
    if gap > cap:
        reasons.append(
            f"回测-实盘背离过大：回测夏普 {float(bt):.2f} 比真实夏普 "
            f"{float(live):.2f} 高 {gap:.2f}，超出容忍 {cap:.2f} → "
            "模拟逼真度存疑（滑点/成交价/信号延迟），本次 A/B 差异不可信，拒绝放行。"
            "请先校准撮合假设或排查实盘执行偏差。")
        return False, reasons

    reasons.append(
        f"回测-实盘一致性通过：回测夏普 {float(bt):.2f} vs 真实 {float(live):.2f}"
        f"（背离 {gap:+.2f}，容忍 ≤{cap:.2f}）")
    return True, reasons


def _clean(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ("rank_ic", "ic_ir", "top_excess", "ann_return", "ann_vol",
              "sharpe", "sortino", "max_drawdown", "win_rate", "turnover",
              "n_periods"):
        v = _num(metrics, k)
        if v is not None:
            out[k] = round(v, 6)
    return out


def _window(score_panel) -> str:
    try:
        idx = score_panel.index
        return f"{idx[0]}~{idx[-1]}"
    except Exception:  # noqa: BLE001
        return ""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def default_panel_provider() -> Optional[Tuple[Any, Any, Optional[Any]]]:
    """默认面板提供者：读**已缓存**的样本外面板。

    面板由 ``scripts/evolve.py --build-panel`` 构建并缓存在
    ``data/evolution/panel/``（``panels.save_panel`` / ``panels.load_panel``）。
    缓存缺失时返回 ``None`` —— 效果是闸门判负并说明原因，
    而不是编一个空面板糊过去。

    ⚠️ 这里曾经**硬编码 ``return None``**，注释是「本机没有随取随用的面板缓存，
    构造它需要特征面板 + 已训练模型」。那句话在写下的当时是对的，
    但 ``panels.py`` 把缓存做出来之后它就成了一句谎话。后果有两处：

    1. 直接调用 ``gate.evaluate()``（不经 runner 注入 provider）永远判
       ``gate:no-data``；
    2. **生成器 D（GA）永远「没有样本外面板，本轮跳过」** ——
       ``scripts/evolve.py`` 只把 provider 注入到 runner（闸门那条路），
       没注入生成器，于是 GA 成了纯粹的死代码，尽管面板就躺在磁盘上。

    实跑证据（2026-10-14）：同一个进化回合里，闸门拿真实夏普
    （0.763 → 0.463）否决了生成器 A 的候选，生成器 D 却报「没有样本外面板」。
    同一个回合、同一块磁盘，两个部件对「面板在不在」给出了相反答案 ——
    这正是最该被消除的那类不一致。

    惰性导入 ``panels`` 是为了保持依赖单向（``panels`` 本身不依赖
    ``gate``，但惰性导入能让这个方向在未来也不容易被无意打破）。
    """
    try:
        from src.evolution import panels
        if not panels.panel_available():
            return None
        return panels.load_panel()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"读取样本外面板缓存失败: {e}")
        return None


# ---------------------------------------------------------------------------
# 校验「自带证据」的候选（生成器 C 等）
# ---------------------------------------------------------------------------

def evaluate_candidate(
    candidate: Candidate,
    provider: Optional[PanelProvider] = None,
    criteria: GateCriteria = DEFAULT_CRITERIA,
) -> Evaluation:
    """统一的候选评估入口：**自带证据优先，否则走回测**。

    两条路的区别与理由：

    * **参数类候选**（``trading_loop.*``）走 :func:`evaluate` ——
      用样本外面板做 A/B 回测，闸门自己算，不假手他人。
    * **模型类候选**（``ml.active_version``）走
      :func:`verify_supplied_evaluation` —— 换模型会**改变评分面板本身**，
      没法用「同一块面板回测两个配置」来比；只能采信生成器随候选附上的
      walk-forward 证据，并由本函数校验其形态与来源白名单。

    证据放在 ``candidate.meta["evidence"]``（而不是 ``candidate.evaluation``），
    是为了守住「只有闸门能写 ``evaluation``」这条不变量 ——
    生成器交上来的是**待核验的呈堂证供**，不是判决书。
    """
    ev = (candidate.meta or {}).get("evidence")
    if ev:
        supplied = Evaluation.from_dict(ev if isinstance(ev, dict) else None)
        return verify_supplied_evaluation(supplied, criteria)
    return evaluate(candidate, provider=provider, criteria=criteria)


def verify_supplied_evaluation(
    evaluation: Optional[Evaluation],
    criteria: GateCriteria = DEFAULT_CRITERIA,
    *,
    allowed_evaluators: Tuple[str, ...] = (
        "walk_forward", "retrain", "oos", "genetic:oos"),
) -> Evaluation:
    """校验生成器自带的样本外证据是否可信。

    存在的理由：模型重训会改变评分面板本身，无法用本模块的 A/B 回测。
    若不校验生成器交上来的 ``Evaluation``，它完全可以自己写
    ``passed=True`` 蒙混过关 —— 这等于没有闸门。

    规则：
      * 必须真的 ``passed``；
      * ``evaluator`` 必须在白名单内（不许自称 ``"gate"`` 等）；
      * 必须给出 ``reasons`` 与 ``delta``（留痕，便于审计）。
    """
    if evaluation is None:
        return Evaluation(passed=False, at=_now(), evaluator="gate:supplied",
                          reasons=["生成器未提供样本外证据 → 拒绝放行"])

    problems: List[str] = []
    if not evaluation.passed:
        problems.append("生成器自评未通过")
    if evaluation.evaluator not in allowed_evaluators:
        problems.append(
            f"证据来源 {evaluation.evaluator!r} 不在可信白名单 "
            f"{allowed_evaluators} 内")
    if not evaluation.reasons:
        problems.append("未给出判定依据（reasons 为空）")
    if not evaluation.delta:
        problems.append("未给出与基准的差异（delta 为空）")
    if not evaluation.window:
        problems.append("未标注评估区间（window 为空）")

    if problems:
        return Evaluation(passed=False, at=_now(), evaluator="gate:supplied",
                          baseline=evaluation.baseline,
                          candidate=evaluation.candidate,
                          delta=evaluation.delta,
                          window=evaluation.window,
                          reasons=[f"自带证据不可信：{p}" for p in problems])

    return Evaluation(passed=True, at=_now(), evaluator="gate:supplied",
                      baseline=evaluation.baseline,
                      candidate=evaluation.candidate,
                      delta=evaluation.delta, window=evaluation.window,
                      reasons=["✅ 自带样本外证据校验通过"] + list(evaluation.reasons))


# ---------------------------------------------------------------------------
# 把判定结果落到候选队列
# ---------------------------------------------------------------------------

def apply_verdict(candidate: Candidate, evaluation: Evaluation, store) -> Candidate:
    """把闸门结论写进候选队列。

    * 通过 → 保持 ``pending``（等人工批准）
    * 未通过 → ``rejected``，并记录原因

    Returns:
        更新后的候选对象。
    """
    from src.evolution.types import ST_PENDING, ST_REJECTED

    candidate.evaluation = evaluation
    store.add(candidate)          # 按 id 覆盖写入，天然幂等
    store.set_status(
        candidate.id,
        ST_PENDING if evaluation.passed else ST_REJECTED,
        by="gate",
        reason=("；".join(evaluation.reasons[:3]) if evaluation.reasons
                else evaluation.evaluator),
    )
    return store.get(candidate.id) or candidate

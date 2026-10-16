# -*- coding: utf-8 -*-
"""自我进化 · 生成器 C：模型重训 → 样本外评估 → 胜出才上线。

它为什么不能复用生成器 A/B 那一套
==================================
换模型会**改变评分面板本身**。生成器 A/B 的候选（``trading_loop.*``）可以用
「同一块样本外面板回测两套配置」来决胜负；但模型候选不行 ——
换了模型，面板的每一列分数都变了，A/B 回测比的就不再是同一件事。

所以模型候选走的是**自带证据**通道：生成器把重训跑出来的 walk-forward
结果作为证供附上，由 :func:`src.evolution.gate.verify_supplied_evaluation`
校验形态与来源白名单，再进同一套「待批准 → 落地 → 回滚」流程。

职责边界（重要）
================
**本生成器绝不训练模型。** 它只做三件事：

1. 从反馈快照判断**该不该重训**（准确率漂移）；
2. 从模型注册表里读**现任**与**挑战者**的样本外指标；
3. 挑战者确实胜出时，提出「把 ``ml.active_version`` 钉到它」的候选。

训练是**运维/调度显式触发**的动作（``scripts/auto_retrain.py``），
因为它要写 ``models/`` 目录、耗时数十分钟 —— 那不属于「生成器只提候选」
这个契约。把它塞进生成器会让「跑一次进化回合」变成不可预期的重活。

``ml.active_version``：基准钉子
==============================
注册表默认「谁的样本外 RankIC 高就用谁」。这意味着只要往 ``models/`` 目录
丢一个新产物的指标文件，**它就会立即生效 —— 绕过闸门与人工批准**。
（这是本系统既有行为，不是本模块引入的。）

钉住 ``ml.active_version`` 就是为了让模型上线变成一个**显式决定**：

* 第一次由运维用 ``scripts/evolve.py --pin <模型键>`` 建立基准 ——
  「把哪个已上线模型当成起点」该由人拍板；
* 之后本生成器在「有基准」的前提下，对样本外确实更优的产物提出晋升；
* 晋升候选必须过闸门（自带证据校验）**并且**经人工批准才写入覆盖层。

一旦钉住，后续再出现「分数更好但没被评估过」的产物也不会顶掉它。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.evolution import space
from src.evolution.feedback import PerformanceSnapshot
from src.evolution.generators.base import change_signature, require_evidence
from src.evolution.types import Candidate, Change, Evaluation

NAME = "retrain"

# ---- 该不该重训（准确率漂移）----
ACCURACY_LOW = 0.45        # 近期 T+5 准确率低于此值 → 模型可能已失效
ACCURACY_DROP = 0.10       # 近期比早期掉超过 10 个百分点 → 漂移
RECENT_N = 3               # 「近期」取最近几批

# ---- 挑战者要胜出必须具备 ----
MIN_RANK_IC_GAIN = 0.005   # 样本外 RankIC 必须超过现任这么多
MIN_OOS_DAYS = 40          # 样本外覆盖天数下限（太少不足为据）

PIN_KEY = "ml.active_version"


# ---------------------------------------------------------------------------
# ① 该不该重训
# ---------------------------------------------------------------------------

def detect_drift(snapshot: PerformanceSnapshot) -> Optional[Dict[str, Any]]:
    """准确率是否漂移到该重训了。返回证据字典；不需要则 None。

    判据（两者满足其一）：
      * 近期 T+5 准确率绝对值过低（``< ACCURACY_LOW``）；
      * 近期比早期明显下滑（``> ACCURACY_DROP``）。
    """
    for h in ("5", "10"):
        vals = [v for v in (snapshot.accuracy_by_horizon.get(h) or [])
                if v is not None]
        if len(vals) < RECENT_N + 1:
            continue
        recent = sum(vals[-RECENT_N:]) / RECENT_N
        early = sum(vals[:-RECENT_N]) / max(len(vals) - RECENT_N, 1)
        if recent < ACCURACY_LOW:
            return {
                "horizon": h, "recent": recent, "early": early,
                "reason": f"近 {RECENT_N} 批 T+{h} 准确率仅 {recent:.1%}"
                          f"（阈值 {ACCURACY_LOW:.0%}）",
            }
        if early - recent > ACCURACY_DROP:
            return {
                "horizon": h, "recent": recent, "early": early,
                "reason": f"近 {RECENT_N} 批 T+{h} 准确率 {recent:.1%}，"
                          f"较早期 {early:.1%} 下滑 {early - recent:.1%}"
                          f"（阈值 {ACCURACY_DROP:.0%}）",
            }
    return None


# ---------------------------------------------------------------------------
# ② 现任 vs 挑战者
# ---------------------------------------------------------------------------

def _mtime(info: Any) -> float:
    p = getattr(info, "model_path", None)
    try:
        return p.stat().st_mtime if p else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def plan(models: List[Any], incumbent_key: str = "") -> Dict[str, Any]:
    """决定这一轮「该不该动模型」。返回 ``{"mode", "incumbent", "challenger", "reason"}``。

    ``mode`` 只有两种取值：

    * ``"promote"`` —— 现任（由 ``ml.active_version`` 钉住）之外存在一个
      **样本外 RankIC 更高**的可部署产物 → 提出把钉子移到它身上。
    * ``None`` —— 按兵不动（没有钉子 / 没有更优产物 / 样本外覆盖不足）。
      此时 ``reason`` 说明为什么。

    为什么**必须**先有钉子
    ----------------------
    没有钉子时，注册表的行为是「谁的样本外 RankIC 高就用谁」—— 也就是说
    只要往 ``models/`` 里丢一个新产物的指标文件，它就**已经生效了**，
    根本不等闸门与人工批准。这种情形下「更优」没有意义（新产物已是现任），
    而且真正的毛病是**静默自动上线**，不是「该不该晋升」。

    所以第一次建立基准必须由运维显式做（``scripts/evolve.py --pin <key>``）：
    「把哪个已上线模型当成起点」本身就是一个该由人拍板的决定。
    钉住之后，本生成器才在「有基准可比」的前提下做证据驱动的晋升。
    """
    loadable = [m for m in (models or []) if getattr(m, "loadable", False)
                and getattr(m, "model_path", None)]
    if not loadable:
        return {"mode": None, "incumbent": None, "challenger": None,
                "reason": "注册表里没有可部署模型"}

    key = str(incumbent_key or "").strip()
    if not key:
        return {"mode": None, "incumbent": None, "challenger": None,
                "reason": "尚未钉住基准版本（ml.active_version 为空）——"
                          "没有基准就无从谈「更优」；"
                          "注册表当前会自动采用样本外最优产物，"
                          "请先用 `evolve.py --pin <模型键>` 明确基准"}

    incumbent = next((m for m in loadable if m.key == key), None)
    if incumbent is None:
        return {"mode": None, "incumbent": None, "challenger": None,
                "reason": f"钉住的版本 {key!r} 不在可部署产物里，基准失效"}

    ranked = [m for m in loadable
              if _rank_ic(m) != float("-inf")]
    if not ranked:
        return {"mode": None, "incumbent": incumbent, "challenger": None,
                "reason": "所有产物都缺样本外 RankIC，无法比较"}
    challenger = max(ranked, key=lambda m: (_rank_ic(m), _mtime(m)))

    if challenger.key == incumbent.key:
        return {"mode": None, "incumbent": incumbent, "challenger": None,
                "reason": f"现任 {incumbent.key} 已是样本外最优，无需晋升"}

    gain = _rank_ic(challenger) - _rank_ic(incumbent)
    if gain < MIN_RANK_IC_GAIN:
        return {"mode": None, "incumbent": incumbent, "challenger": challenger,
                "reason": f"最优产物 {challenger.key} 的样本外 RankIC "
                          f"{_rank_ic(challenger):+.4f} 未超出现任 "
                          f"{_rank_ic(incumbent):+.4f} 达 {MIN_RANK_IC_GAIN}，不晋升"}
    return {"mode": "promote", "incumbent": incumbent,
            "challenger": challenger, "reason": ""}


def _rank_ic(info: Any) -> float:
    v = getattr(info, "rank_ic", None)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("-inf")
    return float("-inf") if f != f else f


def _num(info: Any, field: str) -> Optional[float]:
    v = getattr(info, field, None)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


# ---------------------------------------------------------------------------
# ③ 造证据 + 提候选
# ---------------------------------------------------------------------------

def build_evidence(incumbent: Any, challenger: Any,
                   drift: Dict[str, Any]) -> Evaluation:
    """把「现任 vs 挑战者」的样本外对比整理成一份自带证据。"""
    inc_ric = _rank_ic(incumbent)
    chl_ric = _rank_ic(challenger)
    gain = chl_ric - inc_ric
    n_days = _num(challenger, "n_samples")

    oks: List[str] = []
    fails: List[str] = []
    (oks if gain >= MIN_RANK_IC_GAIN else fails).append(
        f"样本外 RankIC {inc_ric:+.4f} → {chl_ric:+.4f}"
        f"（需 ≥ +{MIN_RANK_IC_GAIN}）")
    (oks if (n_days is not None and n_days >= MIN_OOS_DAYS)
     else fails).append(
        f"样本外覆盖 {n_days if n_days is not None else '未知'} 天"
        f"（需 ≥ {MIN_OOS_DAYS}）")

    inc_ir, chl_ir = _num(incumbent, "ic_ir"), _num(challenger, "ic_ir")
    baseline: Dict[str, float] = {"rank_ic": inc_ric}
    candidate: Dict[str, float] = {"rank_ic": chl_ric}
    delta: Dict[str, float] = {"rank_ic": gain}
    if inc_ir is not None and chl_ir is not None:
        baseline["ic_ir"], candidate["ic_ir"] = inc_ir, chl_ir
        delta["ic_ir"] = chl_ir - inc_ir

    reasons = [f"触发重训：{drift['reason']}"] + oks + fails
    if fails:
        reasons.append("❌ 挑战者未胜出 → 拒绝上线")

    return Evaluation(
        passed=not fails,
        at=_now(),
        baseline=baseline, candidate=candidate, delta=delta,
        reasons=reasons,
        evaluator="retrain",
        window=(f"walk-forward 样本外 {n_days} 天"
                if n_days is not None else "walk-forward（未标注天数）"),
    )


def generate(
    snapshot: PerformanceSnapshot,
    models_fn: Optional[Callable[[], List[Any]]] = None,
    max_candidates: int = 1,
) -> List[Candidate]:
    """重训上线候选。

    Args:
        models_fn: 返回模型列表（``model_registry.ModelInfo``）。
            默认读本机模型注册表；测试注入假列表即可完全离线。
        max_candidates: 最多几条（重训一次只该有一条结论）。

    Returns:
        候选列表。**不需要重训 / 没有挑战者 / 挑战者没胜出** 时返回空列表 ——
        前两种属于「按兵不动」，第三种只写日志不进队列（避免制造噪声候选）。
    """
    drift = detect_drift(snapshot)
    if drift is None:
        logger.info("🧬 生成器 C：准确率未出现漂移，本轮不重训")
        return []

    if not require_evidence(snapshot):
        return []

    if models_fn is None:
        models_fn = _default_models

    try:
        models = models_fn() or []
    except Exception as e:  # noqa: BLE001
        logger.warning(f"🧬 生成器 C：读取模型注册表失败（{e}），本轮跳过")
        return []

    incumbent_key = _current(PIN_KEY) or ""
    p = plan(models, str(incumbent_key))
    if p["mode"] != "promote":
        logger.info(f"🧬 生成器 C：不动模型 —— {p.get('reason') or '无晋升理由'}")
        return []

    incumbent, challenger = p["incumbent"], p["challenger"]
    ev = build_evidence(incumbent, challenger, drift)
    if not ev.passed:
        logger.info(f"🧬 生成器 C：挑战者 {challenger.key} 未胜出，不上线。"
                    + "；".join(r for r in ev.reasons if r.startswith("❌")))
        return []

    try:
        old = space.validate(PIN_KEY, incumbent_key)
        new = space.validate(PIN_KEY, challenger.key)
    except space.ParamError as e:
        logger.warning(f"🧬 生成器 C：{PIN_KEY} 取值非法（{e}），本轮跳过")
        return []
    if old == new:
        return []

    ch = Change(key=PIN_KEY, old=old, new=new)
    rationale = (
        f"{drift['reason']}，且重训产物 {challenger.key} 的样本外表现优于现任 "
        f"{incumbent.key}：\n"
        f"　RankIC {ev.baseline['rank_ic']:+.4f} → {ev.candidate['rank_ic']:+.4f}"
        f"（{ev.delta['rank_ic']:+.4f}），覆盖 {ev.window}。\n"
        f"　把 {PIN_KEY} 由 {incumbent.key} 钉到 {challenger.key}：这是让新模型"
        "**经过闸门**才生效的唯一方式，也顺带冻结裁决 —— 之后再丢进来的未评估"
        "产物不会静默顶掉它。"
        "⚠️ 换模型改变所有后续选股，须人工批准，绝不自动落地。"
    )
    cand = Candidate.create(
        source=NAME, title=f"上线重训模型 {challenger.key}（样本外 RankIC 更优）",
        changes=[ch], rationale=rationale,
        meta={"signature": change_signature([ch]),
              "evidence": ev.to_dict(),
              "incumbent": incumbent.key, "challenger": challenger.key},
    )
    logger.info(f"🧬 生成器 C 产出 1 条候选：{cand.title}")
    return [cand][:max_candidates]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _default_models() -> List[Any]:
    from src.models import model_registry

    return model_registry.scan_models()


def _current(key: str) -> Any:
    try:
        from src.evolution.overrides import read_effective

        return read_effective(key)
    except Exception:  # noqa: BLE001
        return space.get_default(key)


def _now() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")

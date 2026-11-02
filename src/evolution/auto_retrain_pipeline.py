# -*- coding: utf-8 -*-
"""自我进化 · 自动重训流水线（**纯决策，不落地**）。

它解决什么问题
==============
「模型该不该重训」这件事，现在散落在三处：

* ``src/evolution/generators/retrain.py`` 只用**准确率**判漂移（``ACCURACY_LOW``
  / ``ACCURACY_DROP``），且只看已存在的产物，**不会真的重训**；
* ``scripts/auto_retrain.py`` 会真的重训，但判据是 ``evolution.should_retrain()``
  （状态机口径），训完**不做样本外对比**就重置权重；
* ``src/evolution/gate.py`` 有完整的样本外对比逻辑，却只认「回测面板 A/B」与
  「生成器自带证据」两条路，自己没有「先重训再评估」的能力。

三者口径不一致的后果：重训触发了，但没人回答「新模型到底有没有更好」；
或者答了，却答在另一套指标口径上。本模块把这条链**收敛成一个决策函数**：

    检测漂移 → （注入的）训练出候选 → （注入的）样本外评估 → 晋级 / 拒绝 / 待定

设计红线（很重要）
==================
**本模块绝不落地。** 它不训练、不写文件、不改配置、不碰模型目录、不写候选队列。
``train_fn`` / ``eval_fn`` 全部由调用方注入，本模块只负责判断与解释。
理由：把「训练 + 晋升」塞进一个能被随手 import 的模块，等于给「静默自动上线」
开了一道后门 —— 而 ``retrain.py`` 的模块注释已经把这条边界写死：
训练是**运维显式触发**的动作，晋升必须**过闸门 + 人工批准**。

因此：

* 无 import 期副作用（下面全是纯函数与一个无状态的类）；
* 输入字典**防御性拷贝**，调用方传进来的 ``current_metrics`` 不会被就地改写；
* 默认关闭：生产代码不 import 本模块（由 ``AUTO_RETRAIN_ENABLED`` 之类的开关
  在**上层**接线，那是后续步骤的事）。

指标字段口径
============
兼容现有落盘形态，取值时依次尝试：

1. 扁平键：``{"rank_ic": 0.07, "sharpe": 1.2}``（``gate._clean`` 的形状）；
2. 嵌套：``{"walk_forward": {...}}``（``models/production_metrics.json`` 的形状）；
3. 嵌套再一层 ``summary``：``{"walk_forward": {"summary": {...}}}``
   （``models/walk_forward_baseline.json`` 的形状）。

缺失 / ``None`` / ``NaN`` / 非数值一律视为「取不到」→ 返回 ``None``，
由调用方按「无法证明即不放行」处理，而不是当成 0 参与比较。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

__all__ = [
    "RetrainDecision", "AutoRetrainPipeline",
    "ACTION_NO_DRIFT", "ACTION_PROMOTE", "ACTION_REJECT", "ACTION_PENDING",
]

#: 无需重训（当前指标没掉出容忍范围）
ACTION_NO_DRIFT = "no_drift"
#: 候选确实更好，**建议**晋升（是否落地由上层决定）
ACTION_PROMOTE = "promote"
#: 候选不达标 → 拒绝
ACTION_REJECT = "reject"
#: 信息不足（没注入 train_fn / 评估失败 / 指标不可算）→ 待定
ACTION_PENDING = "pending"

_ACTIONS = (ACTION_NO_DRIFT, ACTION_PROMOTE, ACTION_REJECT, ACTION_PENDING)

#: 取值时依次尝试的嵌套容器键（对应现有 json 的落盘形状）
_NESTED_KEYS = ("walk_forward", "summary", "metrics", "oos")

#: 参与 delta 计算的指标（按重要性排序，仅用于报告展示）
_DELTA_KEYS = ("rank_ic", "sharpe")


# ---------------------------------------------------------------------------
# 指标取值（容忍多种落盘形状）
# ---------------------------------------------------------------------------

def _num(metrics: Any, key: str, _depth: int = 0) -> Optional[float]:
    """从指标字典里取一个浮点数；取不到返回 ``None``。

    容忍：扁平 / ``walk_forward`` 嵌套 / 再套一层 ``summary``。
    ``NaN`` 视同缺失 —— 让 NaN 参与 ``>=`` 比较会永远为 False，
    看起来像「安全」，实际是让判定悄悄变成「一律拒绝」，事后极难排查。
    """
    if not isinstance(metrics, dict) or _depth > 3:
        return None
    v = metrics.get(key)
    if v is None:
        for nest in _NESTED_KEYS:
            sub = metrics.get(nest)
            if isinstance(sub, dict):
                got = _num(sub, key, _depth + 1)
                if got is not None:
                    return got
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):   # NaN / inf
        return None
    return f


def _fmt(v: Optional[float], digits: int = 4) -> str:
    return "—" if v is None else f"{v:+.{digits}f}"


# ---------------------------------------------------------------------------
# 决策对象
# ---------------------------------------------------------------------------

@dataclass
class RetrainDecision:
    """一次「要不要换模型」的判断结果。

    这是**建议**，不是命令：``action == "promote"`` 只表示「候选在样本外
    确实更好」，落地仍需过闸门与人工批准（见 ``gate.verify_supplied_evaluation``）。
    """

    #: ``"no_drift"`` / ``"promote"`` / ``"reject"`` / ``"pending"``
    action: str
    #: 人话理由（一句话，可含具体阈值）
    reason: str
    #: 现任指标（只含能算出来的项）
    current_metrics: Dict[str, float] = field(default_factory=dict)
    #: 候选指标（同上；``no_drift`` / ``pending`` 时可能为空）
    candidate_metrics: Dict[str, float] = field(default_factory=dict)
    #: 候选 − 现任（如 ``{"rank_ic": +0.008, "sharpe": +0.12}``）
    delta: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise ValueError(
                f"未知 action {self.action!r}，只允许 {_ACTIONS}")

    @property
    def promoted(self) -> bool:
        """是否建议晋升。仅用于可读性，不构成任何动作。"""
        return self.action == ACTION_PROMOTE

    def to_dict(self) -> Dict[str, Any]:
        """转成可 JSON 序列化的字典（便于上层写审计日志）。"""
        return {
            "action": self.action,
            "reason": self.reason,
            "current_metrics": dict(self.current_metrics),
            "candidate_metrics": dict(self.candidate_metrics),
            "delta": dict(self.delta),
        }


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------

class AutoRetrainPipeline:
    """漂移检测 → 训练 → 评估 → 决策。**不训练、不落地**。

    所有重活都由调用方注入：

    * ``train_fn() -> Any``：返回候选模型（或产物路径 / 指标字典 / 随便什么），
      本模块**原样**转交给 ``eval_fn``，不解释其类型；
    * ``eval_fn(candidate) -> dict``：返回候选的样本外指标字典
      （``{"rank_ic": ..., "sharpe": ...}``）。

    两个都可以是 ``None``：此时 :meth:`run` 只做漂移检测并返回
    ``pending``，这让「只想看该不该重训」的调用方（例如 ``scripts/auto_retrain.py
    --check``）不必造假的依赖。

    Args:
        train_fn: 见上。
        eval_fn: 见上。
        min_rankic_delta: 候选 RankIC 相对现任的最小提升（绝对差）。
        min_abs_rankic: 候选 RankIC 的绝对下限 —— 现任再烂，候选也不能是垃圾。
        drift_drop: 现任 RankIC 相对基准的下滑达到此值 → 判为漂移。

    ⚠️ 三个阈值的默认值刻意保守（与 ``gate.DEFAULT_CRITERIA``、
    ``generators/retrain.MIN_RANK_IC_GAIN`` 同量级）：宁可少换一次模型，
    也不在噪声里反复换 —— 换模型的代价是全部后续选股都变，且不可逆。
    """

    def __init__(
        self,
        train_fn: Optional[Callable[[], Any]] = None,
        eval_fn: Optional[Callable[[Any], Optional[Dict[str, Any]]]] = None,
        min_rankic_delta: float = 0.005,
        min_abs_rankic: float = 0.05,
        drift_drop: float = 0.010,
    ) -> None:
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.min_rankic_delta = float(min_rankic_delta)
        self.min_abs_rankic = float(min_abs_rankic)
        self.drift_drop = float(drift_drop)

    # -- ① 漂移检测 --------------------------------------------------------

    def drift_reason(self, current_metrics: dict,
                     baseline_metrics: dict) -> Optional[str]:
        """返回漂移的具体理由；未漂移返回 ``None``。

        与 :meth:`detect_drift` 是同一判据，只是多给一句可读解释 ——
        「检测到漂移」而不说「掉了多少」，等于让运维无从复核阈值是否合理。

        两条判据（满足其一即漂移）：

        1. **相对下滑**：现任 RankIC 比基准低 ``drift_drop`` 以上；
        2. **绝对过低**：现任 RankIC 本身低于 ``min_abs_rankic``
           ——市场变了、模型钝了，都会掉到这里，但后者（钝）更常见，
           所以即便没有基准可比也要能判。

        缺失容忍：现任 RankIC 取不到 → ``None``（不判漂移）；
        基准缺失 → 只按绝对下限判。**取不到不等于漂移**，
        否则一条坏数据就能触发一次重训。
        """
        cur = _num(current_metrics, "rank_ic")
        if cur is None:
            return None
        if cur < self.min_abs_rankic:
            return (f"现任 RankIC {cur:+.4f} 低于绝对下限 "
                    f"{self.min_abs_rankic:.4f}")
        base = _num(baseline_metrics, "rank_ic")
        if base is None:
            return None
        drop = base - cur
        if drop >= self.drift_drop:
            return (f"现任 RankIC 由基准 {base:+.4f} 滑落至 {cur:+.4f}"
                    f"（-{drop:.4f}，阈值 {self.drift_drop:.4f}）")
        return None

    def detect_drift(self, current_metrics: dict,
                     baseline_metrics: dict) -> bool:
        """是否已漂移到该重训（见 :meth:`drift_reason` 的判据）。"""
        return self.drift_reason(current_metrics, baseline_metrics) is not None

    # -- ② 单点评估（无漂移/无训练时也能用）--------------------------------

    def evaluate_only(self, candidate_metrics: dict,
                      current_metrics: dict) -> str:
        """只比「候选 vs 现任」，不做漂移检测、不训练。返回 action 字符串。

        用途：上层已经拿到候选指标（例如注册表里现成的产物）时，
        复用同一套门槛判断，避免每个调用方各写一遍阈值。

        判定顺序（先排除「无法证明」，再判优劣）：

        1. 候选 RankIC 取不到 → ``pending``（不可证明 ≠ 变差）；
        2. 候选 RankIC < ``min_abs_rankic`` → ``reject``（绝对水平不合格）；
        3. 现任 RankIC 取不到 → ``pending``（没有比较基准）；
        4. 候选 − 现任 < ``min_rankic_delta`` → ``reject``；
        5. 否则 → ``promote``。
        """
        cand = _num(candidate_metrics, "rank_ic")
        if cand is None:
            return ACTION_PENDING
        if cand < self.min_abs_rankic:
            return ACTION_REJECT
        cur = _num(current_metrics, "rank_ic")
        if cur is None:
            return ACTION_PENDING
        if (cand - cur) < self.min_rankic_delta:
            return ACTION_REJECT
        return ACTION_PROMOTE

    def _compare(self, candidate_metrics: dict,
                 current_metrics: dict) -> RetrainDecision:
        """把 :meth:`evaluate_only` 的结论包装成带理由与 delta 的决策。"""
        cur_all = _pick(current_metrics)
        cand_all = _pick(candidate_metrics)
        delta = _delta(cur_all, cand_all)

        cand = _num(candidate_metrics, "rank_ic")
        cur = _num(current_metrics, "rank_ic")
        action = self.evaluate_only(candidate_metrics, current_metrics)

        if action == ACTION_PENDING:
            if cand is None:
                reason = "候选 RankIC 无法计算（样本不足或指标缺失）→ 待定"
            else:
                reason = "现任 RankIC 缺失，无从对比 → 待定"
        elif action == ACTION_REJECT:
            if cand < self.min_abs_rankic:
                reason = (f"候选 RankIC {cand:+.4f} 低于绝对下限 "
                          f"{self.min_abs_rankic:.4f} → 拒绝")
            else:
                reason = (f"候选 RankIC 提升不足：{_fmt(cur)} → {_fmt(cand)}"
                          f"（{_fmt(delta.get('rank_ic'), 4)}，"
                          f"需 ≥ {self.min_rankic_delta:+.4f}）→ 拒绝")
        else:
            reason = (f"候选 RankIC {_fmt(cur)} → {_fmt(cand)}"
                      f"（{_fmt(delta.get('rank_ic'), 4)}），"
                      f"达到 {self.min_rankic_delta:+.4f} 提升门槛"
                      f"且高于绝对下限 {self.min_abs_rankic:.4f} → 建议晋升")

        return RetrainDecision(
            action=action,
            reason=reason,
            current_metrics=cur_all,
            candidate_metrics=cand_all,
            delta=delta,
        )

    # -- ③ 主流程 ----------------------------------------------------------

    def run(
        self,
        current_metrics: dict,
        baseline_metrics: dict,
        train_fn: Optional[Callable[[], Any]] = None,
        eval_fn: Optional[Callable[[Any], Optional[Dict[str, Any]]]] = None,
    ) -> RetrainDecision:
        """跑一遍决策链。**只返回决策，不做任何写入**。

        Args:
            current_metrics: 现任模型的样本外指标（不会被修改）。
            baseline_metrics: 漂移参照（通常是上线时的基线）。
            train_fn: 本次覆盖构造参数里的 ``train_fn``。
            eval_fn: 同上。

        异常处理策略：训练/评估抛异常一律降级为 ``pending`` 并写明原因，
        **不向上抛**。理由：这条链会挂在进化回路上，进化的失败模式必须是
        「不动」而不是「崩掉」——一次网络抖动或一次 OOM 不该让整轮进化停摆。
        """
        cur_copy = copy.deepcopy(current_metrics or {})
        base_copy = copy.deepcopy(baseline_metrics or {})

        why = self.drift_reason(cur_copy, base_copy)
        if why is None:
            cur = _pick(cur_copy)
            return RetrainDecision(
                action=ACTION_NO_DRIFT,
                reason="指标未出现漂移，本轮不重训（未调用 train_fn）",
                current_metrics=cur,
                candidate_metrics={},
                delta={},
            )

        train = train_fn or self.train_fn
        if train is None:
            return RetrainDecision(
                action=ACTION_PENDING,
                reason=f"检测到漂移（{why}），但未提供 train_fn → 待定",
                current_metrics=_pick(cur_copy),
                candidate_metrics={},
                delta={},
            )

        try:
            candidate = train()
        except Exception as e:  # noqa: BLE001
            return RetrainDecision(
                action=ACTION_PENDING,
                reason=f"漂移已确认（{why}），但训练失败：{type(e).__name__}: {e}",
                current_metrics=_pick(cur_copy),
                candidate_metrics={},
                delta={},
            )
        if candidate is None:
            return RetrainDecision(
                action=ACTION_PENDING,
                reason=f"漂移已确认（{why}），但 train_fn 未产出候选 → 待定",
                current_metrics=_pick(cur_copy),
                candidate_metrics={},
                delta={},
            )

        ev = eval_fn or self.eval_fn
        if ev is None:
            return RetrainDecision(
                action=ACTION_PENDING,
                reason=f"漂移已确认（{why}），候选已产出，但未提供 eval_fn → 待定",
                current_metrics=_pick(cur_copy),
                candidate_metrics={},
                delta={},
            )

        try:
            raw = ev(candidate)
        except Exception as e:  # noqa: BLE001
            return RetrainDecision(
                action=ACTION_PENDING,
                reason=f"漂移已确认（{why}），但候选评估失败："
                       f"{type(e).__name__}: {e}",
                current_metrics=_pick(cur_copy),
                candidate_metrics={},
                delta={},
            )
        if not isinstance(raw, dict) or not raw:
            return RetrainDecision(
                action=ACTION_PENDING,
                reason=f"漂移已确认（{why}），但 eval_fn 未返回可用指标 → 待定",
                current_metrics=_pick(cur_copy),
                candidate_metrics={},
                delta={},
            )

        dec = self._compare(copy.deepcopy(raw), cur_copy)
        # 把「为什么重训」拼进理由：只写「候选更好」会丢掉触发上下文，
        # 事后无法解释「既然现任不行，为什么不直接换一个更保守的配置」。
        dec.reason = f"{why} → {dec.reason}"
        return dec


# ---------------------------------------------------------------------------
# 纯函数小工具
# ---------------------------------------------------------------------------

def _pick(metrics: Any) -> Dict[str, float]:
    """抽取可比较的指标子集（只保留算得出来的）。"""
    out: Dict[str, float] = {}
    for k in ("rank_ic", "ic_ir", "top_excess", "sharpe", "n_days", "n_periods"):
        v = _num(metrics, k)
        if v is not None:
            out[k] = round(v, 6)
    return out


def _delta(current: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    """候选 − 现任。任一侧缺失则该指标不进 delta（宁缺勿假）。"""
    out: Dict[str, float] = {}
    for k in _DELTA_KEYS:
        a, b = current.get(k), candidate.get(k)
        if a is None or b is None:
            continue
        out[k] = round(b - a, 6)
    return out


def format_report(dec: RetrainDecision) -> str:
    """把决策渲染成一段人话（供日志 / CLI / 审计留痕）。

    刻意不含 emoji 之外的装饰性排版：这段文本会被写进日志文件，
    多行对齐在窄终端里会散架。
    """
    icon = {ACTION_PROMOTE: "✅", ACTION_REJECT: "❌",
            ACTION_PENDING: "⏳", ACTION_NO_DRIFT: "💤"}.get(dec.action, "•")
    lines = [
        f"{icon} 自动重训决策：{dec.action}",
        f"   理由：{dec.reason}",
    ]
    for label, m in (("现任", dec.current_metrics), ("候选", dec.candidate_metrics)):
        body = ", ".join(f"{k}={v:+.4f}" for k, v in m.items()) or "—"
        lines.append(f"   {label}：{body}")
    if dec.delta:
        lines.append(
            "   差值：" + ", ".join(f"{k}={v:+.4f}" for k, v in dec.delta.items()))
    if dec.action == ACTION_PROMOTE:
        lines.append("   注意：这里只是**建议**，落地须过闸门并人工批准。")
    return "\n".join(lines)

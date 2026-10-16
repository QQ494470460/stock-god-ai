# -*- coding: utf-8 -*-
"""自我进化 · 回路编排。

把五个零件串成一条闭环，并守住「谁有权改什么」：

    ①反馈收集   ②生成候选      ③样本外闸门     ④待批准队列    ⑤落地/回滚
    feedback → generators  →  gate        →  store      →  overrides
    （只读）     （纯函数）      （判胜出）      （记账）       （唯一写盘口）

权责边界（本模块唯一的职责就是「不许越权」）
============================================

* **生成器不许自我批准**：它们的 ``Evaluation`` 不作数，
  统一由 :func:`gate.evaluate` 重新判定。
* **闸门不许改配置**：它只产出结论，写盘在 :mod:`overrides`。
* **默认不自动落地**：``auto_apply=False``。即便打开，也只允许
  ``space.check_changes_adoptable`` 判定为「安全方向且已标记可自动采纳」
  的改动落地；任何「朝更激进方向」的改动**永远**要人工点一次。
* **熔断期间停止进化**：风控处于 halt 状态时直接返回，
  连候选都不生成 —— 系统正在自保，不该同时改参数。
* **落地前二次校验**：``candidates.json`` 是普通 JSON 文件，人可以直接改。
  落地时重跑一遍 :func:`gate.risk_precheck`，防止手改文件塞进风控参数。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.evolution import feedback, gate, space
from src.evolution.generators import REGISTRY
from src.evolution.store import APPLIED_STATES, CandidateStore
from src.evolution.types import (
    Candidate, Change, Evaluation, ST_APPLIED, ST_APPROVED, ST_AUTO_APPLIED,
    ST_PENDING, ST_REJECTED, ST_ROLLED_BACK,
)

__all__ = ["RunReport", "EvolutionRunner"]


@dataclass
class RunReport:
    """一次进化回合的完整留痕（给 CLI / 看板展示）。"""

    started_at: str = ""
    finished_at: str = ""
    generators: List[str] = field(default_factory=list)
    verified: int = 0
    n_candidates: int = 0
    pending: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    auto_applied: List[str] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "generators": self.generators,
            "verified": self.verified,
            "n_candidates": self.n_candidates,
            "pending": self.pending,
            "rejected": self.rejected,
            "auto_applied": self.auto_applied,
            "skipped": self.skipped,
            "errors": self.errors,
            "notes": self.notes,
        }

    def render(self) -> str:
        L = [f"🧬 进化回合 {self.started_at} → {self.finished_at}"]
        L.append(f"  生成器：{'、'.join(self.generators) or '（无）'}"
                 f" | 补验：{self.verified} 条")
        L.append(f"  候选：{self.n_candidates} 条"
                 f"（待批准 {len(self.pending)} / 被拒 {len(self.rejected)}"
                 f" / 自动落地 {len(self.auto_applied)}）")
        for cid in self.pending:
            L.append(f"    ⏳ 待批准 {cid}")
        for cid in self.rejected:
            L.append(f"    ❌ 已否决 {cid}")
        for cid in self.auto_applied:
            L.append(f"    ✅ 已自动落地 {cid}")
        for s in self.skipped:
            L.append(f"    ⏭️ 跳过 {s.get('signature', '')}：{s.get('reason', '')}")
        for e in self.errors:
            L.append(f"    ⚠️ {e}")
        for n in self.notes:
            L.append(f"  ℹ️ {n}")
        return "\n".join(L)


class EvolutionRunner:
    """自我进化编排器。"""

    def __init__(
        self,
        store: Optional[CandidateStore] = None,
        criteria: Optional[gate.GateCriteria] = None,
        provider: Optional[Callable] = None,
        snapshot_fn: Optional[Callable[[], feedback.PerformanceSnapshot]] = None,
    ):
        self.store = store or CandidateStore()
        self.criteria = criteria or gate.DEFAULT_CRITERIA
        # provider=None → 走面板缓存（缺失时闸门会判负并提示构建）
        self.provider = provider
        self._snapshot_fn = snapshot_fn

    # ------------------------------------------------------------------ 回合

    def snapshot(self) -> feedback.PerformanceSnapshot:
        return (self._snapshot_fn or feedback.collect)()

    def run(
        self,
        generators: Optional[List[str]] = None,
        *,
        verify: bool = False,
        auto_apply: bool = False,
        max_candidates: int = 3,
        max_per_generator: int = 1,
        generator_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> RunReport:
        """跑一个进化回合。

        Args:
            generators: 要跑的生成器名（缺省：全部已注册的）。
            verify: 是否先补验到期预测批次。**会写盘**，故默认关闭，
                由 ``scripts/evolve.py --verify`` 显式开启。
            auto_apply: 是否允许**安全方向**的候选自动落地。默认关闭。
            max_candidates: 单回合送闸门的候选上限（闸门跑回测，很贵）。
            max_per_generator: 每个生成器最多采纳几条候选（防单个生成器刷屏）。
            generator_kwargs: 按生成器名分组的额外关键字参数，例如
                ``{"genetic": {"population": 16, "generations": 6}}``。
                只透传目标生成器**签名里真的存在**的参数，名字写错会明确报错
                而不是被静默忽略 —— 一个拼错的 ``--ga-population`` 应该让人
                看见，而不是让人以为自己调大了预算却什么都没变。
        """
        rep = RunReport(started_at=_now(),
                        generators=list(generators or REGISTRY))
        try:
            return self._run(rep, generators, verify, auto_apply,
                            max_candidates, max_per_generator,
                            generator_kwargs)
        finally:
            rep.finished_at = _now()

    def _run(self, rep, generators, verify, auto_apply,
             max_candidates, max_per_generator, generator_kwargs=None) -> RunReport:
        # ---- 0. 熔断期不动 ----
        snap = self.snapshot()
        if (snap.guard or {}).get("halted"):
            rep.notes.append(
                f"风控处于熔断状态（{snap.guard.get('halt_reason')}），"
                "本轮不进任何改动 —— 系统自保期间不该同时改策略")
            return rep

        # ---- 1. 补验（可选，会写盘）----
        if verify:
            try:
                rep.verified = len(feedback.verify_pending())
            except Exception as e:  # noqa: BLE001
                rep.errors.append(f"补验失败：{e}")
            snap = self.snapshot()

        if not snap.is_useful():
            rep.notes.append(
                f"证据不足（已验证 {snap.n_verified} 批 / 结算 "
                f"{snap.n_settlements} 天），本轮仅观察、不生成候选")

        # ---- 2. 生成候选 ----
        names = generators or list(REGISTRY)
        seen_signatures = {c.meta.get("signature") for c in self.store.all()
                           if c.meta.get("signature")}
        produced: List[Candidate] = []

        for name in names:
            fn = REGISTRY.get(name)
            if fn is None:
                rep.errors.append(f"未注册的生成器：{name}（可用："
                                  f"{'、'.join(sorted(REGISTRY))}）")
                continue
            kwargs = dict((generator_kwargs or {}).get(name) or {})
            # 生成器若接受 provider，就把 runner 手里那份一并给它。
            # 理由：闸门走的是 self.provider，生成器若自行取面板，两者可能
            # 取到**不同的面板**（或一个有一个没有）。实跑踩过——
            # 面板明明在磁盘上，闸门拿真实夏普否决了候选，同一个回合里
            # 生成器 D 却说「没有样本外面板」而跳过。同一块面板必须贯穿全链路。
            # 只在 runner 确实持有 provider 时注入；否则留给生成器自己的默认
            # （``gate.default_panel_provider``，它同样会读缓存）。
            if (self.provider is not None and "provider" not in kwargs
                    and _accepts_kwarg(fn, "provider")):
                kwargs["provider"] = self.provider
            if kwargs:
                try:
                    _check_generator_kwargs(name, fn, kwargs)
                except TypeError as e:
                    rep.errors.append(str(e))
                    continue
            try:
                cands = fn(snap, **kwargs) or []
            except Exception as e:  # noqa: BLE001
                rep.errors.append(f"生成器 {name} 异常：{e}")
                logger.exception(f"生成器 {name} 异常")
                continue

            taken = 0
            for c in cands:
                if taken >= max_per_generator:
                    break
                sig = c.meta.get("signature")
                if sig and sig in seen_signatures:
                    rep.skipped.append({"signature": sig,
                                        "reason": "队列中已有同类候选"})
                    continue
                seen_signatures.add(sig)
                produced.append(c)
                taken += 1

        rep.n_candidates = len(produced)

        # ---- 3. 过闸门 + 入队 ----
        for c in produced[:max_candidates]:
            try:
                ev = self._evaluate(c)
            except Exception as e:  # noqa: BLE001
                rep.errors.append(f"候选 {c.id} 评估异常：{e}")
                continue

            if not ev.passed:
                gate.apply_verdict(c, ev, self.store)
                rep.rejected.append(c.id)
                logger.info(f"❌ 候选 {c.id} 未通过闸门："
                            f"{(ev.reasons or ['?'])[0]}")
                continue

            gate.apply_verdict(c, ev, self.store)
            if auto_apply and self._is_safe_to_auto(c):
                try:
                    self.apply(c.id, by="auto(evolution)", auto=True)
                    rep.auto_applied.append(c.id)
                    continue
                except Exception as e:  # noqa: BLE001
                    rep.errors.append(f"自动落地 {c.id} 失败：{e}")
            rep.pending.append(c.id)

        if rep.n_candidates > max_candidates:
            rep.notes.append(
                f"另有 {rep.n_candidates - max_candidates} 条候选未评估"
                "（单回合上限），下回合继续")

        self._audit(rep)
        return rep

    # ------------------------------------------------------------------ 内部

    def _evaluate(self, c: Candidate) -> Evaluation:
        """统一评估入口：自带证据优先，否则走样本外回测。

        走 :func:`gate.evaluate_candidate` 而不是直接 ``gate.evaluate``，
        是为了让「模型类候选」（携带 walk-forward 证据）也能被同一套
        队列/批准/回滚流程接住，而不必在 runner 里长出一堆 if。
        """
        return gate.evaluate_candidate(c, provider=self.provider,
                                      criteria=self.criteria)

    def _is_safe_to_auto(self, c: Candidate) -> bool:
        """是否允许自动落地：必须 *全部* 是安全方向且已标记可自动采纳。"""
        ok, why = space.check_changes_adoptable(c.change_tuples)
        if not ok:
            logger.info(f"候选 {c.id} 不自动落地：{why}")
        return ok

    def _audit(self, rep: RunReport) -> None:
        """把这一回合的结论写进进化日志（``evolution_state.json``）。

        刻意**不写** ``data/trading_journal.jsonl``：那是交易审计流水，
        混入策略参数变更会污染合规追溯。
        """
        try:
            from src.models.evolution import EvolutionEngine

            eng = EvolutionEngine()
            eng._log_evolution(  # noqa: SLF001
                "evolution_round",
                f"生成器 {'、'.join(rep.generators)}：候选 {rep.n_candidates}，"
                f"待批准 {len(rep.pending)}，否决 {len(rep.rejected)}，"
                f"自动落地 {len(rep.auto_applied)}")
            eng.save()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"进化日志写入失败（不影响回合结果）: {e}")

    # ------------------------------------------------------------------ 队列

    def pending(self) -> List[Candidate]:
        return self.store.pending()

    def rollbackable(self) -> List[Candidate]:
        return self.store.rollbackable()

    def show(self, cand_id: str) -> Optional[Candidate]:
        return self.store.get(cand_id)

    def approve(self, cand_id: str, by: str = "user",
                reason: str = "") -> Candidate:
        """人工批准（尚未落地）。"""
        c = self._require(cand_id)
        if c.status not in (ST_PENDING,):
            raise ValueError(f"候选 {cand_id} 状态为 {c.status}，无法批准")
        return self.store.set_status(cand_id, ST_APPROVED, by=by, reason=reason)

    def reject(self, cand_id: str, by: str = "user",
               reason: str = "") -> Candidate:
        c = self._require(cand_id)
        if c.status in (ST_APPLIED, ST_ROLLED_BACK):
            raise ValueError(f"候选 {cand_id} 已落地/已回滚，不能拒绝")
        return self.store.reject(cand_id, by=by, reason=reason or "人工拒绝")

    def apply(self, cand_id: str, by: str = "user",
              auto: bool = False) -> Candidate:
        """把候选写进运行期配置（唯一会改参数的动作）。

        前置：状态必须是 ``pending``（人工批准会被自动跳过）或 ``approved``。

        Args:
            auto: 是否由回路的「安全方向自动落地」触发。
                它只是**留痕口径**不同（``auto_applied`` vs ``applied``），
                不改变任何校验 —— 越权判定仍在 ``_is_safe_to_auto`` 里。
        """
        from src.evolution import overrides

        c = self._require(cand_id)
        if c.status not in (ST_PENDING, ST_APPROVED):
            raise ValueError(f"候选 {cand_id} 状态为 {c.status}，不能落地")

        # 二次校验：candidates.json 可被手改，落地前再拦一次风控硬线
        blockers, _ = gate.risk_precheck(c.changes)
        if blockers:
            raise ValueError("落地被拒（触碰风控硬线）：" + "；".join(blockers))

        applied = overrides.apply_changes(c.changes)
        target = ST_AUTO_APPLIED if auto else ST_APPLIED
        out = self.store.set_status(
            cand_id, target, by=by, reason="自动落地（安全方向）" if auto else "已落地")
        logger.info(f"{'🤖' if auto else '✅'} 候选 {cand_id} 已落地："
                    f"{'、'.join(applied)}")
        return out

    def rollback(self, cand_id: str, by: str = "user") -> Candidate:
        """一键回滚：把候选里记录的 ``old`` 写回去。"""
        from src.evolution import overrides

        c = self._require(cand_id)
        if c.status not in APPLIED_STATES:
            raise ValueError(f"候选 {cand_id} 状态为 {c.status}，不能回滚")

        inverse = [Change(key=ch.key, old=ch.new, new=ch.old)
                   for ch in c.changes]
        blockers, _ = gate.risk_precheck(inverse)
        if blockers:
            raise ValueError("回滚被拒：" + "；".join(blockers))

        applied = overrides.apply_changes(inverse)
        logger.info(f"↩️ 候选 {cand_id} 已回滚：{'、'.join(applied)}")
        return self.store.mark_rolled_back(cand_id, by=by)

    def _require(self, cand_id: str) -> Candidate:
        c = self.store.get(cand_id)
        if c is None:
            raise KeyError(f"候选不存在：{cand_id}")
        return c

    # ------------------------------------------------------------------ 概览

    def status(self) -> Dict[str, Any]:
        from src.evolution import overrides

        snap = self.snapshot()
        return {
            "candidates": self.store.summary(),
            "pending": [
                {"id": c.id, "title": c.title, "source": c.source,
                 "changes": [ch.describe() for ch in c.changes],
                 "risk_note": self._risk_note(c)}
                for c in self.store.pending()
            ],
            "applied": [
                {"id": c.id, "title": c.title,
                 "applied_at": c.applied_at} for c in self.store.rollbackable()
            ],
            "overrides": overrides.overrides_summary(),
            "feedback": snap.summary(),
            "guard_halted": (snap.guard or {}).get("halted"),
        }

    def _risk_note(self, c: Candidate) -> str:
        """给人工批准时看的风险提示（一句话说清「批了会怎样」）。"""
        parts = []
        for ch in c.changes:
            try:
                spec = space.get_spec(ch.key)
            except space.ParamError:
                parts.append("⛔ 风控硬线（不该出现在这里）")
                continue
            if spec.is_risk_increasing(ch.old, ch.new):
                parts.append(f"⚠️ 更激进（{spec.doc or ch.key}）")
            else:
                parts.append("✅ 保守/中性方向")
        return "；".join(parts)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _accepts_kwarg(fn: Callable, name: str) -> bool:
    """生成器签名里是否接受 ``name``（显式声明或 ``**kwargs`` 兜底都算）。

    只用来决定「要不要把 provider 递过去」，因此不可内省时保守返回 ``False``
    —— 宁可不注入，也不要因为猜错签名把生成器调崩。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def _check_generator_kwargs(name: str, fn: Callable, kwargs: Dict[str, Any]) -> None:
    """校验透传的关键字参数确实是这个生成器接受的。

    为什么必须校验而不是「不认识的丢掉」：``--ga-population`` 这类参数只在
    少数生成器上存在。若静默丢弃，用户会以为自己把搜索预算调大了 4 倍，
    实际什么都没变，还会因为结果没差别而怀疑「进化引擎没起作用」。
    参数名写错必须当场炸出来。

    参数带 ``**kwargs`` 兜底的生成器一律放行（它自己承接）。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # 内置/不可内省的调用对象，放行
        return
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return
    unknown = sorted(k for k in kwargs if k not in sig.parameters)
    if unknown:
        accepted = sorted(
            k for k, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        )
        raise TypeError(
            f"生成器 {name} 不接受参数 {'、'.join(unknown)}；"
            f"它接受：{'、'.join(accepted) or '（无）'}"
        )

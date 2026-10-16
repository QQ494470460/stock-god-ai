# -*- coding: utf-8 -*-
"""自我进化 · 候选生成器公共契约。

四个生成器（参数寻优 / LLM 复盘 / 模型重训 / 遗传算法）都遵守同一份约定：

**生成器只「提候选」，不做三件事：**

1. **不写盘** —— 不碰 ``config/`` 、不碰 ``data/`` 。产出纯内存对象。
2. **不做决策** —— 是否胜出由 :mod:`src.evolution.gate` 说了算，
   生成器不许自己给自己盖章（``passed=True``）。
3. **不落地** —— 落地只能走 ``overrides.apply_changes``，
   且必须经过人工批准。

这样每个生成器都可以独立单元测试：输入一份 :class:`PerformanceSnapshot`，
断言「产出哪些改动、理由里有没有引用数据」。

候选的「有效期」与去重
======================
同一症状会被反复观察到（换手一直偏高）。若每次扫描都提一遍，
待批准队列会被同一件事塞满。因此每个候选都带一个
:func:`change_signature` 指纹，由 runner 用它跳过重复项。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from src.evolution import space
from src.evolution.feedback import PerformanceSnapshot
from src.evolution.types import Candidate, Change

__all__ = [
    "GeneratorError", "change_signature", "draft", "require_evidence", "atomize",
]


class GeneratorError(RuntimeError):
    """生成器参数非法/前置条件不满足。"""


# ---------------------------------------------------------------------------
# 指纹与去重
# ---------------------------------------------------------------------------

def change_signature(changes: Sequence[Change]) -> str:
    """候选指纹：同样的目标值 = 同一件事（与旧值无关）。

    刻意**只用新值**做指纹：从 10 调到 6 和从 12 调到 6 是同一件事，
    不该各占一条待批候选。
    """
    return "|".join(f"{c.key}={c.new!r}" for c in sorted(
        changes, key=lambda x: x.key))


# ---------------------------------------------------------------------------
# 起草候选
# ---------------------------------------------------------------------------

def draft(
    source: str,
    title: str,
    rationale: str,
    new_values: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Candidate]:
    """按「目标值」起草一个候选，自动补上旧值。

    三件事自动完成，避免每个生成器各写一遍、各错一遍：

    * **取旧值**：``overrides.read_effective`` 读当前生效值（含已有覆盖），
      不是读参数空间的默认值 —— 否则第二次进化会用错基线。
    * **丢弃空改动**：新值等于旧值的项直接剔除；若全被剔除则返回 ``None``，
      不产出「改了 0 项」的空候选。
    * **越界即报错**：任何一项非法（越界 / 受保护 / 未登记）立刻抛
      :class:`space.ParamError`，不静默夹紧 —— 生成器提出的东西必须自己成立。

    Returns:
        候选对象；无有效改动时返回 ``None``。
    """
    from src.evolution.overrides import read_effective

    if not new_values:
        return None

    changes: List[Change] = []
    for key, new in new_values.items():
        space.validate(key, new)                    # 风控硬线在这里被拒
        try:
            old = read_effective(key)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"读取 {key} 当前值失败: {e}")
            old = space.get_default(key)
        if old == new:
            continue
        changes.append(Change(key=key, old=old, new=new))

    if not changes:
        return None

    m = dict(meta or {})
    m.setdefault("signature", change_signature(changes))
    return Candidate.create(source=source, title=title, changes=changes,
                            rationale=rationale, meta=m)


# ---------------------------------------------------------------------------
# 前置：证据
# ---------------------------------------------------------------------------

def require_evidence(snapshot: PerformanceSnapshot,
                     minimum_verified: int = 3,
                     minimum_settlements: int = 5) -> bool:
    """是否有足够证据支撑「生成候选」这件事。

    没证据就提改动 = 拍脑袋。宁可这一轮什么都不做（空交易计划 = 按兵不动），
    也不要为了「看起来在进化」而制造噪声候选。
    """
    if snapshot.n_verified >= minimum_verified:
        return True
    if snapshot.n_settlements >= minimum_settlements:
        return True
    logger.info(
        f"🧬 证据不足（已验证 {snapshot.n_verified} 批 / 结算 {snapshot.n_settlements} 天，"
        f"需 ≥{minimum_verified} 或 ≥{minimum_settlements}），本轮不生成候选")
    return False


# ---------------------------------------------------------------------------
# 批量起草
# ---------------------------------------------------------------------------

def atomize(source: str, pairs: Sequence[Tuple[str, Dict[str, Any], str]],
            ) -> List[Candidate]:
    """把多组「标题 + 目标值 + 理由」批量起草成候选，自动丢弃空改动。

    Args:
        pairs: ``[(title, new_values, rationale), ...]``
    """
    out: List[Candidate] = []
    for title, values, why in pairs:
        c = draft(source, title, why, values)
        if c is not None:
            out.append(c)
    return out

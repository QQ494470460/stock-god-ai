# -*- coding: utf-8 -*-
"""自我进化 · 数据结构。"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

__all__ = [
    "Change", "Evaluation", "Candidate",
    "ST_PENDING", "ST_APPROVED", "ST_REJECTED", "ST_APPLIED",
    "ST_AUTO_APPLIED", "ST_ROLLED_BACK", "new_id",
]

# 候选状态机
ST_PENDING = "pending"            # 已通过样本外闸门，等待人工批准
ST_APPROVED = "approved"          # 人工批准（尚未落地）
ST_REJECTED = "rejected"          # 人工拒绝
ST_APPLIED = "applied"            # 已落地
ST_AUTO_APPLIED = "auto_applied"  # 自动采纳（仅限安全方向的改动）
ST_ROLLED_BACK = "rolled_back"    # 已回滚

_TERMINAL = (ST_REJECTED, ST_APPLIED, ST_AUTO_APPLIED, ST_ROLLED_BACK)


def new_id(prefix: str = "cand") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass
class Change:
    """一项参数变更。`old` 同时充当回滚依据，故必须如实记录。"""
    key: str
    old: Any
    new: Any

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Change":
        return cls(key=d["key"], old=d.get("old"), new=d.get("new"))

    def describe(self) -> str:
        return f"{self.key}: {self.old!r} → {self.new!r}"


@dataclass
class Evaluation:
    """样本外验证结果（当前参数 vs 候选参数）。"""
    passed: bool = False
    at: str = ""
    baseline: Dict[str, float] = field(default_factory=dict)
    candidate: Dict[str, float] = field(default_factory=dict)
    delta: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    evaluator: str = ""
    window: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> Optional["Evaluation"]:
        if not d:
            return None
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Candidate:
    """一个待评估/待批准的进化候选。"""
    id: str
    created_at: str
    source: str                     # 生成器名：param_search / llm_review / retrain / genetic
    title: str
    rationale: str = ""
    changes: List[Change] = field(default_factory=list)
    evaluation: Optional[Evaluation] = None
    status: str = ST_PENDING
    decided_by: str = ""
    decided_at: str = ""
    decision_reason: str = ""
    applied_at: str = ""
    rolled_back_at: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- 便捷 ----

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    @property
    def change_tuples(self) -> List[tuple]:
        return [(c.key, c.old, c.new) for c in self.changes]

    def describe(self) -> str:
        head = f"[{self.id}] {self.title} （{self.source}）"
        body = "\n".join(f"    · {c.describe()}" for c in self.changes)
        ev = ""
        if self.evaluation:
            ev = (f"\n    评估: {'✅ 通过' if self.evaluation.passed else '❌ 未通过'}"
                  f"（{self.evaluation.evaluator}）")
            if self.evaluation.reasons:
                ev += "\n      " + "；".join(self.evaluation.reasons[:3])
        return f"{head}\n  状态: {self.status}{ev}\n{body}"

    # ---- 序列化 ----

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["changes"] = [c.to_dict() for c in self.changes]
        d["evaluation"] = self.evaluation.to_dict() if self.evaluation else None
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Candidate":
        return cls(
            id=d["id"],
            created_at=d.get("created_at", ""),
            source=d.get("source", ""),
            title=d.get("title", ""),
            rationale=d.get("rationale", ""),
            changes=[Change.from_dict(c) for c in d.get("changes", [])],
            evaluation=Evaluation.from_dict(d.get("evaluation")),
            status=d.get("status", ST_PENDING),
            decided_by=d.get("decided_by", ""),
            decided_at=d.get("decided_at", ""),
            decision_reason=d.get("decision_reason", ""),
            applied_at=d.get("applied_at", ""),
            rolled_back_at=d.get("rolled_back_at", ""),
            meta=d.get("meta", {}) or {},
        )

    @classmethod
    def create(cls, source: str, title: str, changes: List[Change],
               rationale: str = "", meta: Optional[Dict] = None) -> "Candidate":
        return cls(
            id=new_id(source[:6]),
            created_at=datetime.now().isoformat(timespec="seconds"),
            source=source, title=title, rationale=rationale,
            changes=changes, meta=meta or {},
        )

# -*- coding: utf-8 -*-
"""自我进化 · 候选队列与回滚台账。

设计要点
========
* **只记账，不改配置**。真正的写盘由 `src/evolution/overrides.py` 负责，
  这样存储层可以独立测试。
* **回滚靠 `Change.old`**：每个变更都如实记录旧值，回滚就是把旧值写回去。
  不依赖快照文件，也就不存在"快照与配置不一致"的问题。
* 一旦某项候选被落地，它的 `old` 就是唯一的回滚依据，故落地前后
  都不允许修改 `changes`（`add()` 之后为只读约定）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from config.settings import DATA_DIR
from src.evolution.types import (
    Candidate, ST_APPLIED, ST_APPROVED, ST_AUTO_APPLIED, ST_PENDING,
    ST_REJECTED, ST_ROLLED_BACK,
)

__all__ = ["CANDIDATES_FILE", "CandidateStore", "APPLIED_STATES"]

CANDIDATES_FILE = Path(DATA_DIR) / "evolution" / "candidates.json"

#: 这些状态下的候选已经落过盘，回滚有意义
APPLIED_STATES = (ST_APPLIED, ST_AUTO_APPLIED)


class CandidateStore:
    """候选队列的持久化（JSON 文件，容量小、便于人工查看）。"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else CANDIDATES_FILE
        self._candidates: Dict[str, Candidate] = {}
        self._load()

    # ------------------------------------------------------------------ IO

    def _load(self) -> None:
        self._candidates = {}
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                items = raw.get("candidates", raw) if isinstance(raw, dict) else raw
                for d in items or []:
                    c = Candidate.from_dict(d)
                    self._candidates[c.id] = c
        except Exception as e:  # noqa: BLE001
            logger.warning(f"进化候选队列读取失败，按空处理: {e}")

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "_comment": "自我进化候选队列。status: pending/approved/rejected/applied/auto_applied/rolled_back",
                "_updated_at": datetime.now().isoformat(timespec="seconds"),
                "candidates": [c.to_dict() for c in self._candidates.values()],
            }
            self.path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.error(f"进化候选队列写入失败: {e}")

    # ------------------------------------------------------------------ 查询

    def add(self, cand: Candidate) -> Candidate:
        self._candidates[cand.id] = cand
        self._save()
        return cand

    def get(self, cand_id: str) -> Optional[Candidate]:
        return self._candidates.get(cand_id)

    def all(self) -> List[Candidate]:
        return sorted(self._candidates.values(), key=lambda c: c.created_at, reverse=True)

    def by_status(self, *status: str) -> List[Candidate]:
        want = set(status)
        return [c for c in self.all() if c.status in want]

    def pending(self) -> List[Candidate]:
        return self.by_status(ST_PENDING, ST_APPROVED)

    def applied(self) -> List[Candidate]:
        return self.by_status(*APPLIED_STATES)

    def rollbackable(self) -> List[Candidate]:
        """还能回滚的（已落地且未回滚）。"""
        return [c for c in self.applied() if not c.rolled_back_at]

    def has_pending_for(self, source: str) -> bool:
        return any(c.source == source for c in self.pending())

    # ------------------------------------------------------------------ 状态流转

    def set_status(self, cand_id: str, status: str, *,
                   by: str = "", reason: str = "") -> Candidate:
        c = self._candidates.get(cand_id)
        if c is None:
            raise KeyError(f"候选不存在: {cand_id}")
        c.status = status
        if by:
            c.decided_by = by
            c.decided_at = datetime.now().isoformat(timespec="seconds")
        if reason:
            c.decision_reason = reason
        if status in APPLIED_STATES:
            c.applied_at = datetime.now().isoformat(timespec="seconds")
        elif status == ST_ROLLED_BACK:
            c.rolled_back_at = datetime.now().isoformat(timespec="seconds")
        self._save()
        return c

    def reject(self, cand_id: str, by: str = "user", reason: str = "") -> Candidate:
        return self.set_status(cand_id, ST_REJECTED, by=by, reason=reason)

    def mark_applied(self, cand_id: str, auto: bool = False) -> Candidate:
        return self.set_status(cand_id, ST_AUTO_APPLIED if auto else ST_APPLIED)

    def mark_rolled_back(self, cand_id: str, by: str = "") -> Candidate:
        c = self._candidates.get(cand_id)
        if c is None:
            raise KeyError(f"候选不存在: {cand_id}")
        if c.status not in APPLIED_STATES:
            raise ValueError(f"候选 {cand_id} 状态为 {c.status}，不能回滚")
        return self.set_status(cand_id, ST_ROLLED_BACK, by=by, reason="已回滚")

    # ------------------------------------------------------------------ 概览

    def summary(self) -> Dict:
        counts: Dict[str, int] = {}
        for c in self._candidates.values():
            counts[c.status] = counts.get(c.status, 0) + 1
        return {
            "file": str(self.path),
            "total": len(self._candidates),
            "counts": counts,
            "pending": len(self.pending()),
        }

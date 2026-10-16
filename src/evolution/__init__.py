# -*- coding: utf-8 -*-
"""``src.evolution`` —— 自我进化框架。

四个「候选生成器」共用一条回路：

    反馈收集 → 生成候选 → 样本外闸门 → 待批准队列 → 落地 → 可回滚
    feedback    generators      gate         store

安全边界见 `src/evolution/space.py`：只有登记在 :data:`PARAM_SPACE`
里的**策略参数**可被进化触碰；风控硬线（熔断线、单票上限、下单频率等）
永久受保护，且任何「朝更激进方向」的改动一律需要人工批准。
"""

from src.evolution.feedback import (  # noqa: F401
    PerformanceSnapshot, collect, collect_prediction_feedback,
    collect_trade_feedback, compute_metrics, render_text, verify_pending,
)
from src.evolution.gate import (  # noqa: F401
    DEFAULT_CRITERIA, GateCriteria, apply_verdict, compare_metrics, evaluate,
    risk_precheck, verify_supplied_evaluation,
)
from src.evolution.runner import EvolutionRunner, RunReport  # noqa: F401
from src.evolution.space import (  # noqa: F401
    PARAM_SPACE, ParamError, ParamSpec, check_changes_adoptable,
    describe_space, is_protected, validate,
)
from src.evolution.store import APPLIED_STATES, CandidateStore  # noqa: F401
from src.evolution.types import (  # noqa: F401
    Candidate, Change, Evaluation,
    ST_APPLIED, ST_APPROVED, ST_AUTO_APPLIED, ST_PENDING, ST_REJECTED,
    ST_ROLLED_BACK,
)

__all__ = [
    "PARAM_SPACE", "ParamError", "ParamSpec", "check_changes_adoptable",
    "describe_space", "is_protected", "validate",
    "CandidateStore", "APPLIED_STATES",
    "Candidate", "Change", "Evaluation",
    "ST_PENDING", "ST_APPROVED", "ST_REJECTED", "ST_APPLIED",
    "ST_AUTO_APPLIED", "ST_ROLLED_BACK",
    "PerformanceSnapshot", "collect", "collect_prediction_feedback",
    "collect_trade_feedback", "compute_metrics", "render_text", "verify_pending",
    "GateCriteria", "DEFAULT_CRITERIA", "evaluate", "compare_metrics",
    "risk_precheck", "verify_supplied_evaluation", "apply_verdict",
    "EvolutionRunner", "RunReport",
]

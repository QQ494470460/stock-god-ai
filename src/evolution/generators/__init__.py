# -*- coding: utf-8 -*-
"""``src.evolution.generators`` —— 候选生成器集合。

所有生成器共用同一份契约（见 :mod:`src.evolution.generators.base`）：

* **只提候选**：纯函数、不写盘、不自我批准、不落地；
* 产出统一是 :class:`src.evolution.types.Candidate`（含旧值与理由）；
* 由 ``src.evolution.runner`` 统一送闸门 → 队列 → 人工批准。

======================  ====================================================
生成器                   驱动信号
======================  ====================================================
``param_search``        组合绩效症状（换手 / 现金拖累 / 回撤 / 持仓不满）
``llm_review``          交易日志与验证结果 → LLM 复盘出的改进假设
``retrain``             预测准确率漂移 → 重训 → 样本外胜出才上线
``genetic``             参数适应度 → 选择 / 交叉 / 变异搜索
======================  ====================================================

``REGISTRY`` 只登记**已实现**的生成器；未实现的名字不会出现在里面，
调用方（``--generators`` 参数）据此报错而不是静默跳过。
"""

from src.evolution.generators import (  # noqa: F401
    base, genetic, llm_review, param_search, retrain,
)
from src.evolution.generators.base import (  # noqa: F401
    GeneratorError, atomize, change_signature, draft, require_evidence,
)

#: 生成器名 → ``generate(snapshot, **kwargs) -> List[Candidate]``
REGISTRY = {
    param_search.NAME: param_search.generate,
    llm_review.NAME: llm_review.generate,
    retrain.NAME: retrain.generate,
    genetic.NAME: genetic.generate,
}


def register(name: str) -> None:
    """把新生成器登记进 REGISTRY（实现方在模块底部调用）。"""
    from importlib import import_module

    mod = import_module(f"src.evolution.generators.{name}")
    REGISTRY[mod.NAME] = mod.generate


def all_names() -> list:
    return sorted(REGISTRY)


__all__ = [
    "base", "param_search", "llm_review", "retrain", "genetic",
    "GeneratorError", "atomize", "change_signature", "draft",
    "require_evidence", "REGISTRY", "register", "all_names",
]

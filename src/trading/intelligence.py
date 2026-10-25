# -*- coding: utf-8 -*-
"""闭环智能层（Batch 3）：让风控/加权从「静态规则」进化为「会学习」。

包含两个零 IO、可单测、fail-safe 的组件：

1. ``StrategyCredibility`` —— 在线策略信誉。
   用每笔已平仓交易的「信号方向 vs 实际盈亏」一致性（类 IC 的 EWMA 代理）
   为每个策略标签维护一个信誉分，映射成 ``ic_weights`` 喂给
   ``ensemble.fuse_strategies``。这样 Feature #7 的多策略集成**不再用静态权重**，
   而是随真实战绩自我调优——好策略权重涨、差策略权重跌。零 IO、纯数学。

2. （波动自适应熔断线逻辑放在 ``guard.RiskSupervisor._effective_daily_loss_pct``，
   因为它需要直接读 guard 的配置与实时市场波动；这里不重复实现。）

设计原则：纯逻辑、可单测、绝不抛异常、未知标签安全兜底。
"""

from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional


class StrategyCredibility:
    """在线策略信誉：EWMA 跟踪每个策略标签的「方向一致性」，输出 IC 风格权重。

    语义（贴合本系统「多头为主」的实况）：
    - 入场意图视为看多（``entry_strength >= 0`` → 方向 +1）；空头/回避场景
      ``entry_strength < 0`` → 方向 -1。若入场强度缺省为 0，按看多处理。
    - 平仓实际收益 ``realized_return >= 0`` → 方向 +1，否则 -1。
    - 方向一致记 1 分、不一致记 0 分，按半衰期做指数滑动平均（EWMA）。
      分值∈[0,1]，0.5 为中性。

    权重映射：``weight = floor + (cap - floor) * score``，中性(0.5)→ 1.0，
    满分→ cap，零分→ floor。未知/尚未观测的标签按中性权重 1.0 返回（不放大、不压制）。
    """

    def __init__(
        self,
        halflife: int = 20,
        floor: float = 0.5,
        cap: float = 1.5,
        neutral: float = 0.5,
    ):
        self.halflife = max(1, int(halflife))
        # EWMA 系数：alpha = 1 - 0.5^(1/halflife)
        self.alpha = 1.0 - math.pow(0.5, 1.0 / self.halflife)
        self.floor = float(floor)
        self.cap = float(cap)
        self.neutral = float(neutral)
        # tag -> EWMA 分（∈[0,1]）
        self._score: Dict[str, float] = {}
        # tag -> 已观测样本数（供看板展示「近期战绩」样本量）
        self._n: Dict[str, int] = {}
        # 脏标记：仅在 observe 后才需要落盘（避免每 tick 空写）
        self._dirty: bool = False

    def observe(
        self,
        tag: str,
        entry_strength: float,
        realized_return: float,
    ) -> None:
        """喂入一笔已平仓结果，更新该标签信誉。

        Args:
            tag: 策略标签（如 "momentum" / "defensive"）。
            entry_strength: 入场时的信号强度（>=0 视为看多）。
            realized_return: 该笔实际收益率（如 pnl_pct/100）。
        """
        if not tag:
            return
        try:
            intended = 1.0 if float(entry_strength) >= 0 else -1.0
            actual = 1.0 if float(realized_return) >= 0 else -1.0
            hit = 1.0 if intended == actual else 0.0
        except (TypeError, ValueError):
            return
        prev = self._score.get(tag, self.neutral)
        self._score[tag] = prev + self.alpha * (hit - prev)
        self._n[tag] = int(self._n.get(tag, 0)) + 1
        self._dirty = True

    def weight(self, tag: str) -> float:
        s = self._score.get(tag, self.neutral)
        return self.floor + (self.cap - self.floor) * s

    def score(self, tag: str) -> float:
        """返回该标签的原始 EWMA 信誉分（∈[0,1]，中性 0.5），供 UI 直接展示。"""
        return float(self._score.get(tag, self.neutral))

    def weights(self) -> Dict[str, float]:
        """返回所有已观测标签的权重表（供 ensemble.fuse_strategies 的 ic_weights）。"""
        return {t: self.weight(t) for t in self._score}

    def count(self, tag: str) -> int:
        """返回该标签已观测的样本数（供看板展示「近期战绩」样本量）。"""
        return int(self._n.get(tag, 0))

    def snapshot(self) -> Dict[str, Any]:
        """导出当前所有已观测标签的信誉快照（score/weight/n + 元参数）。"""
        return {
            "halflife": self.halflife,
            "floor": self.floor,
            "cap": self.cap,
            "neutral": self.neutral,
            "scores": {
                t: {
                    "score": round(float(self._score.get(t, self.neutral)), 6),
                    "weight": round(float(self.weight(t)), 6),
                    "n": int(self._n.get(t, 0)),
                }
                for t in self._score
            },
        }

    def is_dirty(self) -> bool:
        """是否有自上次落盘以来新观测到的样本（guard 用它做写盘节流）。"""
        return self._dirty

    def save_state(self, path) -> None:
        """把信誉快照原子落盘（.tmp + os.replace），任何异常静默吞掉。

        仅当 ``is_dirty()`` 为真才应调用（见 guard 的落盘节流）；调用成功后清除
        dirty 标记。
        """
        try:
            p = Path(path)
            data = self.snapshot()
            data["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, p)
            self._dirty = False
        except Exception:  # noqa: BLE001  // 落盘失败绝不能影响主流程
            pass

    def load_state(self, path) -> None:
        """从落盘快照恢复信誉（score/n），失败静默忽略。

        注意：只恢复 score/n，不覆盖本实例的 halflife/floor/cap（权重仍按构造参数算）。
        看板侧只读加载、从不写回，因此不会与 guard 的落盘互相覆盖。
        """
        try:
            p = Path(path)
            if not p.exists():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            scores = raw.get("scores") or {}
            if not isinstance(scores, dict):
                return
            for tag, v in scores.items():
                if not isinstance(v, dict):
                    continue
                try:
                    self._score[tag] = float(v.get("score", self.neutral))
                    self._n[tag] = int(v.get("n", 0))
                except (TypeError, ValueError):
                    continue
        except Exception:  # noqa: BLE001
            pass

    def reset(self) -> None:
        self._score.clear()
        self._n.clear()
        self._dirty = False

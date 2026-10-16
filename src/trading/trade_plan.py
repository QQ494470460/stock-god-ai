# -*- coding: utf-8 -*-
"""
交易计划（Trade Plan）—— 选股与交易的桥梁
=========================================

问题：选股在收盘后跑（daily_pick），交易在盘中跑（AutoTrader），
但两者之间**没有桥**——盘中买入候选来自人手维护的 watchlist.json，
与模型当天选出的标的完全脱节，所谓"全自动"其实只有半截。

本模块把「收盘后的 AI 选股结果」固化成一份**可执行的交易计划**：

    picks（模型选出的 TopN）
        ↓ build_trade_plan()
    trade_plan_YYYYMMDD.json  {
        targets: [{code, name, score, weight, price}],
        cash_reserve_pct, max_positions, ...
    }
        ↓ 盘中由 rebalance.py 消费
    目标持仓差异 → 买单/卖单

权重分配支持：
  · equal  —— TopN 等权（默认，稳健、换手低）
  · score  —— 按评分加权（强者多买，换手略高）

安全约束（写在生成端，避免下游重复判断）：
  · 目标只数 ≤ max_positions（与风控 guard 的上限保持一致，防止计划本身超限）
  · 权重合计 ≤ 1 - cash_reserve_pct（始终留底仓现金）
  · 计划自带 plan_date，只对「下一个交易日」有效，过期自动作废
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger

from config.settings import DATA_DIR


# ----------------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------------

@dataclass
class PlanTarget:
    """计划中的一只目标持仓"""
    code: str
    name: str = ""
    score: float = 0.0
    weight: float = 0.0          # 目标权重（占总资产比例，0~1）
    price: float = 0.0           # 计划生成时参考价（盘中会以实时价重算）
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlanTarget":
        return cls(
            code=str(d.get("code", "")),
            name=str(d.get("name", "")),
            score=float(d.get("score", 0) or 0),
            weight=float(d.get("weight", 0) or 0),
            price=float(d.get("price", 0) or 0),
            reason=str(d.get("reason", "")),
        )


@dataclass
class TradePlan:
    """一份交易计划 = 目标组合"""
    plan_date: str                       # 计划适用的交易日 YYYY-MM-DD
    created_at: str = ""
    source: str = "daily_pick"           # daily_pick / watchlist / manual
    pool: str = ""
    min_score: float = 0.0
    regime: str = ""
    targets: List[PlanTarget] = field(default_factory=list)
    cash_reserve_pct: float = 0.10       # 始终保留的现金比例
    max_positions: int = 10              # 目标持仓上限
    weighting: str = "equal"             # equal / score
    notes: str = ""

    # ---------- 便捷视图 ----------
    @property
    def codes(self) -> List[str]:
        return [t.code for t in self.targets]

    @property
    def weight_map(self) -> Dict[str, float]:
        return {t.code: t.weight for t in self.targets}

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TradePlan":
        return cls(
            plan_date=str(d.get("plan_date", "")),
            created_at=str(d.get("created_at", "")),
            source=str(d.get("source", "daily_pick")),
            pool=str(d.get("pool", "")),
            min_score=float(d.get("min_score", 0) or 0),
            regime=str(d.get("regime", "") or ""),
            targets=[PlanTarget.from_dict(t) for t in (d.get("targets") or [])],
            cash_reserve_pct=float(d.get("cash_reserve_pct", 0.10) or 0),
            max_positions=int(d.get("max_positions", 10) or 10),
            weighting=str(d.get("weighting", "equal")),
            notes=str(d.get("notes", "")),
        )

    # ---------- 有效性 ----------
    def is_valid_for(self, day: Optional[date] = None) -> bool:
        """计划是否**就是为这一天**准备的（``plan_date == day``）。

        时点约定：计划是收盘后为**下一个交易日**生成的，所以 D 日盘中该用的
        是「D-1 收盘后写的、``plan_date == D``」那一份。本方法因此做严格相等
        判断 —— 它回答的是「这份计划是不是今天的」，不是「这份计划还有效吗」。

        两个容易踩的坑，都写在这里免得下次再推理一遍：

        1. **不要放宽成 ``plan_date >= day``。** 那会让「为明天准备的计划」
          在今天就通过校验，进而被拿去交易今天。真正的处理在
           ``TradingLoop._ensure_plan``：过期时先回磁盘重读「今天的」计划，
           读不到才重建；手上若是待生效的次日计划则**原样保留**。
        2. **本方法的 docstring 曾经声称**「plan_date 是下一个交易日且今天还没到
           也算有效」——**实现从来没有那样做过**。文档许诺一个更宽松的行为、
           代码给一个更严格的行为，比两者都错更危险：读文档的人会据此写出
           错误的调用假设。现在文档与实现对齐（实现是对的那一方）。
        """
        day = day or date.today()
        if not self.plan_date:
            return False
        try:
            pd_ = datetime.strptime(self.plan_date, "%Y-%m-%d").date()
        except Exception:  # noqa: BLE001
            return False
        return pd_ == day

    def summary(self) -> str:
        lines = [
            f"交易计划 {self.plan_date}（{self.source}，池={self.pool or '-'}，"
            f"权重={self.weighting}）",
            f"  目标 {len(self.targets)} 只 · 现金保留 {self.cash_reserve_pct*100:.0f}% · "
            f"仓位上限 {self.max_positions} 只",
        ]
        for i, t in enumerate(self.targets, 1):
            lines.append(
                f"  {i:>2}. {t.code} {t.name}  评分 {t.score:>5.1f}  "
                f"目标权重 {t.weight*100:>5.2f}%"
            )
        if not self.targets:
            lines.append("  （无目标标的）")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# 权重分配
# ----------------------------------------------------------------------------

def _assign_weights(
    targets: List[PlanTarget],
    investable_pct: float,
    weighting: str = "equal",
    weight_cap: Optional[float] = None,
) -> None:
    """就地填充 target.weight。

    investable_pct：可投资比例 = 1 - cash_reserve_pct。
    weight_cap ：单票权重上限。设了它就与风控 guard 的 max_single_position_pct
                 对齐，避免「计划要 45%、风控只准 30%」的错配导致执行端缩量。
                 被截断后剩余额度**留在现金**（更保守），不强行摊给别人。
    """
    n = len(targets)
    if n == 0:
        return
    if weighting == "score":
        raw = [max(0.0, t.score) for t in targets]
        total = sum(raw)
        if total <= 0:
            raw = [1.0] * n
            total = float(n)
        for t, r in zip(targets, raw):
            t.weight = investable_pct * (r / total)
    else:  # equal
        w = investable_pct / n
        for t in targets:
            t.weight = w

    if weight_cap is not None and weight_cap > 0:
        for t in targets:
            t.weight = min(t.weight, float(weight_cap))


# ----------------------------------------------------------------------------
# 输入归一化
# ----------------------------------------------------------------------------

def _normalize_picks(picks: Any) -> List[dict]:
    """把「每日推荐」的产物归一化成 [{code,name,score,reason}]。

    兼容三种输入：
      · DataFrame（predictor.get_top_recommendations 的产物）
      · list[dict]（daily_pick report 里的 picks 字段，中文键）
      · list[dict]（英文键 code/name/score，通用）
    """
    rows: List[dict] = []

    # pandas DataFrame
    try:
        import pandas as pd
        if isinstance(picks, pd.DataFrame):
            picks = picks.to_dict(orient="records")
    except Exception:  # noqa: BLE001
        pass

    if not isinstance(picks, Iterable):
        return rows

    for r in picks:
        if not isinstance(r, dict):
            continue
        code = r.get("code") or r.get("代码") or ""
        if not code:
            continue
        name = r.get("name") or r.get("名称") or code
        try:
            score = float(r.get("score") or r.get("评分") or 0)
        except Exception:  # noqa: BLE001
            score = 0.0
        reason = r.get("reason") or r.get("推荐理由") or ""
        rows.append({"code": str(code), "name": str(name),
                     "score": score, "reason": str(reason)})

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


# ----------------------------------------------------------------------------
# 构建 / 持久化
# ----------------------------------------------------------------------------

def default_plan_dir() -> Path:
    return Path(DATA_DIR) / "trade_plans"


def plan_path_for(plan_date: str, plan_dir: Optional[Path] = None) -> Path:
    d = Path(plan_dir) if plan_dir else default_plan_dir()
    return d / f"trade_plan_{plan_date.replace('-', '')}.json"


def resolve_target_date(ref: Optional[date] = None) -> str:
    """计划的目标执行日 = 下一个交易日（若今天就交易且还没收盘，则含今天）。"""
    try:
        from src.trading.calendar import next_trading_day, is_trading_day, is_trading_time
        ref = ref or date.today()
        if is_trading_day(ref) and is_trading_time():
            return ref.strftime("%Y-%m-%d")
        return next_trading_day(ref, include_today=False).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        ref = ref or date.today()
        # 退化：默认下一自然日
        return (ref + timedelta(days=1)).strftime("%Y-%m-%d")


def build_trade_plan(
    picks: Any = None,
    plan_date: Optional[str] = None,
    source: str = "daily_pick",
    pool: str = "",
    min_score: float = 0.0,
    regime: str = "",
    top_n: Optional[int] = None,
    max_positions: int = 10,
    cash_reserve_pct: float = 0.10,
    weighting: str = "equal",
    weight_cap: Optional[float] = None,
    write: bool = True,
    plan_dir: Optional[Path] = None,
) -> TradePlan:
    """从选股结果构建交易计划。

    picks 为 None 时，自动读取最近一份 daily_reports 里的推荐（离线可用）。
    weight_cap 建议传风控 guard 的 max_single_position_pct（默认 0.30），
    使计划与风控口径一致。
    """
    # 1. 拿 picks
    if picks is None:
        picks, meta = load_latest_picks()
        pool = pool or meta.get("pool", "")
        min_score = min_score or float(meta.get("min_score", 0) or 0)
        regime = regime or str(meta.get("regime") or "")
        source = source or meta.get("source", "daily_pick")

    rows = _normalize_picks(picks)

    # 2. 目标只数收口（计划本身不得超限）
    limit = max_positions if top_n is None else min(top_n, max_positions)
    rows = rows[: max(0, limit)]

    targets = [
        PlanTarget(code=r["code"], name=r["name"], score=r["score"], reason=r["reason"])
        for r in rows
    ]

    # 3. 分配权重
    cash_reserve_pct = float(min(max(cash_reserve_pct, 0.0), 0.95))
    _assign_weights(targets, 1.0 - cash_reserve_pct, weighting, weight_cap=weight_cap)

    plan = TradePlan(
        plan_date=plan_date or resolve_target_date(),
        created_at=datetime.now().isoformat(timespec="seconds"),
        source=source,
        pool=pool,
        min_score=round(float(min_score), 1),
        regime=regime,
        targets=targets,
        cash_reserve_pct=cash_reserve_pct,
        max_positions=max_positions,
        weighting=weighting,
    )

    if write:
        save_trade_plan(plan, plan_dir=plan_dir)
    logger.info(
        f"📋 交易计划已生成：{plan.plan_date} · {len(targets)} 只 · "
        f"现金保留 {cash_reserve_pct*100:.0f}% · 权重={weighting}"
    )
    return plan


def save_trade_plan(plan: TradePlan, plan_dir: Optional[Path] = None) -> Path:
    p = plan_path_for(plan.plan_date, plan_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return p


def load_trade_plan(
    plan_date: Optional[str] = None,
    plan_dir: Optional[Path] = None,
) -> Optional[TradePlan]:
    """读取指定日期的计划；plan_date 为空时取最近一份。"""
    if plan_date:
        p = plan_path_for(plan_date, plan_dir)
        if not p.exists():
            return None
        try:
            return TradePlan.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"交易计划解析失败 {p}: {e}")
            return None
    return load_latest_plan(plan_dir)


def load_latest_plan(plan_dir: Optional[Path] = None) -> Optional[TradePlan]:
    d = Path(plan_dir) if plan_dir else default_plan_dir()
    if not d.exists():
        return None
    files = sorted(d.glob("trade_plan_*.json"))
    for f in reversed(files):
        try:
            return TradePlan.from_dict(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"交易计划解析失败 {f}: {e}")
            continue
    return None


def load_latest_picks(report_dir: Optional[Path] = None) -> tuple[List[dict], dict]:
    """读取最近一份每日推荐报告的 picks（离线兜底数据源）。"""
    from config.settings import config as app_config
    d = Path(report_dir) if report_dir else Path(app_config.scheduler.report_dir)
    if not d.exists():
        return [], {}
    files = sorted(d.glob("daily_picks_*.json"))
    for f in reversed(files):
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
            picks = rep.get("picks") or []
            meta = {
                "pool": rep.get("pool", ""),
                "min_score": rep.get("min_score", 0),
                "regime": rep.get("regime") or "",
                "source": "daily_pick",
                "generated_at": rep.get("generated_at", ""),
            }
            if picks:
                return picks, meta
        except Exception:  # noqa: BLE001
            continue
    return [], {}


def plan_diff(plan: TradePlan, held_codes: Sequence[str]) -> Dict[str, List[str]]:
    """计划与当前持仓的差异（供 UI 展示与调仓预演）。"""
    target = set(plan.codes)
    held = set(held_codes or [])
    return {
        "to_buy": sorted(target - held),
        "to_hold": sorted(target & held),
        "to_sell": sorted(held - target),
    }


if __name__ == "__main__":  # 手动生成一次
    p = build_trade_plan(write=True)
    print(p.summary())

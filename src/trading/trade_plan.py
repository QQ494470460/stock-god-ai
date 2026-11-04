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
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger

from config.settings import DATA_DIR


# ----------------------------------------------------------------------------
# 市场状态 → 组合参数覆盖（指数级 regime 自适应）
# ----------------------------------------------------------------------------
# 仅在调用方「未显式指定」max_positions / cash_reserve_pct 时生效。
# 设计：全市场走熊时收敛仓位（更少的标的 + 更多现金），走牛时略放开。
# 约定：plan.regime 是**指数级**（000300.SH）口径，与 picks 里逐股的 reason
# （个股口径）是两回事，二者可合法不一致。这里用指数级 regime 做组合层面的
# 防御性调整（全市场走熊 → 少而精 + 多留现金），与个股判断互不干扰。
# 显式传入的参数永远优先（手动/测试可覆盖）。
REGIME_POSITION_OVERRIDES: dict = {}
# 市场状态 → 组合参数覆盖。**已停用（故意保持为空）**。
#
# 历史：bear 压缩到 6 只 / 20% 现金，bull 放宽到 12 只 / 5% 现金。
# 停用理由（2026-10-31）：
#   1. 它是一套**顺周期择时** —— 越跌仓位越轻，反弹时恰好最轻。而执行级
#      回测（1140 天，含 2022 熊市与 2024 年初急跌）用的始终是固定
#      「8 只 / 10% 现金」，年化 +25%。用一个回测里不存在的机制去改造
#      被验证过的配置，等于把验证作废（同类问题：绝对分数门槛、ma60 闸门）。
#   2. 它制造入口不一致：手动 `--build-plan` 命中覆盖（6 只 / 20% 现金 /
#      equal 权重），闭环重建时显式传参（8 只 / 10% / risk_parity）——
#      同一份选股产出两个不同的组合。
# 恢复方式：把 `{"bear": {...}, "bull": {...}}` 加回来。恢复前请先跑
# `scripts/exec_ab.py` 验证这套覆盖到底有没有正贡献，别凭直觉。


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
    # code → 当日候选池内的评分名次（0 起）。给「排名滞后」用：
    # 老持仓只要还在 ``top_n × hold_rank_slack`` 名以内就保留，不因为
    # 「今天没进前 8」被换掉。没有它，实盘只能按「跌出目标组合即清仓」，
    # 换手率是回测最优配置的 1.7 倍（50bp 成本下年化差 6.4pp）。
    raw_rank: Dict[str, int] = field(default_factory=dict)
    # 这份计划吃的是哪一天的选股报告，以及它相对 plan_date 滞后了几个**交易日**。
    # 存在的理由：``load_latest_picks()`` 会安静地拿「最近一份」报告，
    # 选股作业漏跑时它就变成几天前的旧票，而计划本身看起来完全正常。
    # 落在计划文件里，UI / 审计工具才能把「这批票是旧的」摆到屏幕上。
    # 旧计划文件没有这两个字段 → from_dict 回落到 ""/0（= 未知，不误报）。
    picks_date: str = ""
    picks_lag_days: int = 0

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
            raw_rank={
                str(k): int(v)
                for k, v in (d.get("raw_rank") or {}).items()
                if str(k).strip()
            },
            picks_date=str(d.get("picks_date", "") or ""),
            picks_lag_days=int(d.get("picks_lag_days", 0) or 0),
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
        if self.picks_date:
            lag = f"，滞后 {self.picks_lag_days} 个交易日" if self.picks_lag_days else ""
            flag = "  ⚠️ 选股结果陈旧" if self.picks_lag_days >= 2 else ""
            lines.append(f"  选股报告 {self.picks_date}{lag}{flag}")
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

def _load_risk_stats(codes: List[str], data_dir: Optional[Path] = None) -> Optional[dict]:
    """给风险加权收集协方差 / 波动率 / 目标波动。

    全部 fail-safe：任何环节不可用都返回 None → 调用方回退等权。
    **绝不**因为读不到数据就编造权重——那会让「风险平价」变成「随机加权」，
    比等权更糟且更难排查。
    """
    codes = [str(c) for c in (codes or []) if c]
    if len(codes) < 2:
        return None
    try:
        from src.trading.risk_model import PortfolioRiskMonitor

        mon = PortfolioRiskMonitor(data_dir=data_dir) if data_dir else PortfolioRiskMonitor()
        got = mon.cov_and_vols(codes)
        if got is None:
            return None
        usable, cov, vols = got

        target_vol = None
        try:
            from config.settings import ROOT_DIR

            gp = Path(ROOT_DIR) / "config" / "trading_guard.yaml"
            if gp.exists():
                from src.trading.guard import GuardConfig

                target_vol = float(GuardConfig.from_yaml(str(gp)).volatility_target_pct)
        except Exception:  # noqa: BLE001
            target_vol = None

        return {
            "codes": list(usable),
            "cov": cov,
            "vols": vols,
            "target_vol": target_vol,
        }
    except Exception as e:  # noqa: BLE001
        logger.debug(f"风险统计加载失败，权重回退等权: {e}")
        return None


def _assign_weights(
    targets: List[PlanTarget],
    investable_pct: float,
    weighting: str = "equal",
    weight_cap: Optional[float] = None,
    risk_stats: Optional[dict] = None,
    tilt_strength: float = 0.0,
) -> None:
    """就地填充 target.weight。

    investable_pct：可投资比例 = 1 - cash_reserve_pct。
    weight_cap ：单票权重上限。设了它就与风控 guard 的 max_single_position_pct
                 对齐，避免「计划要 45%、风控只准 30%」的错配导致执行端缩量。
                 被截断后剩余额度**留在现金**（更保守），不强行摊给别人。
    risk_stats ：风险统计（``{"codes", "cov", "vols"}``），``risk_parity`` /
                 ``min_variance`` / ``vol_target`` 三种加权方式必需；缺失时
                 优雅退回等权（不报错、不猜）。
    tilt_strength：风险权重之上叠加 alpha 倾斜的力度（0=纯风险平价）。

    为什么要有风险加权
    ------------------
    ``equal`` 与 ``score`` 都只看 alpha，不看波动与相关性：等权会让一只 70%
    年化波动的题材股与一只 25% 的银行股拿一样多的钱（8 成风险来自前者）；
    按评分加权更糟——模型偏爱动量，高分往往就是高波动，等于在波动最大的
    地方下最重的注。风险平价让每只票贡献相同风险，组合的 VaR 与风控预算同源。
    """
    n = len(targets)
    if n == 0:
        return

    method = (weighting or "equal").lower()
    if method in ("risk_parity", "min_variance", "vol_target"):
        raw = _risk_weights(targets, method, risk_stats, weight_cap, tilt_strength)
        if raw is not None:
            for t, w in zip(targets, raw):
                t.weight = investable_pct * float(w)
            return  # 风险权重已在内部处理 cap，直接返回
        method = "equal"  # 数据不足 → 退回等权

    if method == "score":
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


def _risk_weights(
    targets: List[PlanTarget],
    method: str,
    risk_stats: Optional[dict],
    weight_cap: Optional[float],
    tilt_strength: float,
) -> Optional[List[float]]:
    """按风险统计求权重；数据不可用时返回 None（调用方回退等权）。"""
    if not risk_stats or not targets:
        return None
    codes = [str(c) for c in (risk_stats.get("codes") or [])]
    cov = risk_stats.get("cov")
    vols = risk_stats.get("vols")
    if not codes:
        return None

    # 风险统计的标的集合必须覆盖计划全部标的，否则「缺价的那只」会拿到
    # 未经风险调整的权重 —— 那正是最危险的一只（数据缺失往往意味着流动性差）。
    pos = {c: i for i, c in enumerate(codes)}
    if any(t.code not in pos for t in targets):
        logger.debug("风险统计未覆盖全部标的，权重回退等权")
        return None

    idx = [pos[t.code] for t in targets]
    scores = [float(t.score or 0.0) for t in targets]

    try:
        import numpy as np

        from src.trading.optimizer import optimize_weights

        sub_cov = None
        if cov is not None and method in ("risk_parity", "min_variance"):
            arr = np.asarray(cov, dtype=float)
            if arr.ndim != 2 or arr.shape[0] < len(codes):
                return None
            sub_cov = arr[np.ix_(idx, idx)]

        sub_vols = None
        if vols is not None:
            varr = np.asarray(vols, dtype=float)
            if varr.size >= len(codes):
                sub_vols = varr[idx]

        target_vol = risk_stats.get("target_vol")
        w = optimize_weights(
            cov=sub_cov,
            vols=sub_vols,
            scores=scores,
            method=method,
            cap=weight_cap,
            target_vol=float(target_vol) if target_vol else None,
            tilt_strength=float(tilt_strength),
        )
        w = np.asarray(w, dtype=float)
        if w.size != len(targets) or not np.all(np.isfinite(w)):
            return None

        # vol_target 允许 sum(w) < 1（波动高就少持仓，剩下的留现金）；
        # 其余方法强制归一，避免因数值误差漏掉一小截资金。
        total = float(w.sum())
        if total <= 0:
            return None
        if method != "vol_target" and abs(total - 1.0) > 1e-6:
            w = w / total
        # 注：换手封顶需要「上一版计划权重」作为基准，计划层拿不到持仓口径的
        # 旧权重，因此换手控制放在 rebalance / 执行侧做（那里的 band_pct 与
        # max_daily_turnover_pct 才是正确的着力点），此处不做。
        return [float(x) for x in w]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"风险加权失败，回退等权: {e}")
        return None



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
        # 参考价：优先用推荐时点的最新价/收盘价。早期版本从没透传过价格，
        # 于是计划里 10 只目标全是 price=0.0，字段名存实亡 —— 执行侧读不到
        # 任何参考价，没法做「价格跳空太多就别追」的校验。
        try:
            price = float(
                r.get("price") or r.get("最新价") or r.get("收盘价") or r.get("close") or 0
            )
        except Exception:  # noqa: BLE001
            price = 0.0
        rows.append({"code": str(code), "name": str(name),
                     "score": score, "reason": str(reason), "price": price})

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def _fetch_close_prices(codes: List[str]) -> Dict[str, float]:
    """为计划目标补充参考收盘价（盘中重算前的锚点）。

    daily_pick 本应在 report.picks 里透传 price，但实测某些行情源下该字段没接上
    （stock_data 的 key 格式与 picks 里的代码对不上），导致计划 price 恒为 0.0、
    「跳空不追」这类校验无从做起。这里在计划构建层做一次兜底：直接按代码拉日线，
    取最后一根收盘价。失败一律返回空 dict —— 行情取不到绝不该中断计划生成。
    """
    codes = [str(c) for c in codes if c]
    if not codes:
        return {}
    try:
        from src.data.fetcher import fetcher

        raw = fetcher.get_batch_daily_kline(codes, max_workers=4)
        out: Dict[str, float] = {}
        for c, df in raw.items():
            try:
                out[str(c)] = round(float(df["close"].iloc[-1]), 2)
            except Exception:  # noqa: BLE001
                continue
        return out
    except Exception:  # noqa: BLE001
        return {}


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
    max_positions: Optional[int] = None,
    cash_reserve_pct: Optional[float] = None,
    weighting: str = "equal",
    weight_cap: Optional[float] = None,
    risk_stats: Optional[dict] = None,
    tilt_strength: float = 0.0,
    picks_date: Optional[str] = None,
    write: bool = True,
    plan_dir: Optional[Path] = None,
) -> TradePlan:
    """从选股结果构建交易计划。

    picks 为 None 时，自动读取最近一份 daily_reports 里的推荐（离线可用）。
    weight_cap 建议传风控 guard 的 max_single_position_pct（默认 0.30），
    使计划与风控口径一致。

    ``picks_date``：调用方**自己知道**这批 picks 是哪天的报告时显式传入
    （例如 ``job_daily_pick`` 刚跑出报告就建计划）。显式传 picks 时 meta 为空，
    不传这个就永远记不到 ``picks_date``，下游的新鲜度审计会一直是「未知」——
    而夜间那次正常生成恰恰是唯一每天都走的路径。

    市场状态自适应：当 ``regime`` 命中 ``REGIME_POSITION_OVERRIDES`` 且调用方
    未显式传入 ``max_positions`` / ``cash_reserve_pct`` 时，自动套用该状态的
    防御性组合参数（熊市→少而精+多现金，牛市→略放开）。显式传参优先。
    """
    # 1. 拿 picks
    meta: Dict[str, Any] = {}
    if picks is None:
        picks, meta = load_latest_picks()
        pool = pool or meta.get("pool", "")
        min_score = min_score or float(meta.get("min_score", 0) or 0)
        regime = regime or str(meta.get("regime") or "")
        source = source or meta.get("source", "daily_pick")

    # 1b. 新鲜度裁决。**只在这一处**算 plan_date：下面构造 TradePlan 也用它。
    #     两处各调一次 resolve_target_date() 会在跨午夜时得到两个不同的日期，
    #     而「计划的有效日」和「新鲜度的基准日」必须是同一天。
    _plan_date_str = plan_date or resolve_target_date()
    fresh = picks_staleness(
        {"report_date": picks_date} if picks_date else meta, _plan_date_str
    )
    if fresh.get("stale"):
        # 这是「安静地不做事」里最贵的一种：选股作业漏跑 → 计划换了批几天前的
        # 旧票，而计划文件、界面、审计看板全都显示正常。
        # 实跑证据（2026-11-05）：11-04 傍晚 loop 崩溃 → 18:45 选股作业没触发 →
        # 11-05 补建计划时吃到 11-03 的 picks，8 只里 4 只缺行业映射、合计 43% 权重，
        # 与 11-04 那份目标**逐位相同**，没有任何一处报错。
        logger.warning(f"⚠️ 交易计划用的是旧选股结果：{fresh['reason']}")

    rows = _normalize_picks(picks)

    # ---- 市场状态（指数级）自适应：熊市更保守、牛市更激进 ----
    # 仅当调用方未显式传入对应参数时才套用覆盖（REGIME_POSITION_OVERRIDES），
    # 保证手动/测试显式指定的值永远优先、可复现。
    if regime in REGIME_POSITION_OVERRIDES:
        ov = REGIME_POSITION_OVERRIDES[regime]
        if max_positions is None and "max_positions" in ov:
            max_positions = int(ov["max_positions"])
        if cash_reserve_pct is None and "cash_reserve_pct" in ov:
            cash_reserve_pct = float(ov["cash_reserve_pct"])
    # 兜底默认（调用方未传、且 regime 无覆盖时）
    if max_positions is None:
        max_positions = 10
    if cash_reserve_pct is None:
        cash_reserve_pct = 0.10

    # 注意：price 只来自选股报告（daily_pick 的「最新价」）。报告没给价就保持 0.0，
    # 这里**不**去行情兜底——「缺价=未知」本身就是有用信号，自动编造会让执行侧的
    # 「跳空不追」校验误以为拿到了真实锚点。历史计划用 backfill_prices() 手动补。

    # 2. 目标只数收口（计划本身不得超限）
    # ⚠️ 排名表必须在截断**之前**生成：截断后只剩 top_n 只，「排名滞后」
    # 需要的 9~(top_n×slack) 名信息就永久丢失了，届时 slack 只能退化成 1.0。
    raw_rank: Dict[str, int] = {
        str(r["code"]): i for i, r in enumerate(rows) if r.get("code")
    }
    limit = max_positions if top_n is None else min(top_n, max_positions)
    rows = rows[: max(0, limit)]

    targets = [
        PlanTarget(code=r["code"], name=r["name"], score=r["score"],
                   reason=r["reason"], price=float(r.get("price") or 0.0))
        for r in rows
    ]

    # 3. 分配权重
    cash_reserve_pct = float(min(max(cash_reserve_pct, 0.0), 0.95))
    if weighting in ("risk_parity", "min_variance", "vol_target") and risk_stats is None:
        risk_stats = _load_risk_stats([t.code for t in targets])
    _assign_weights(targets, 1.0 - cash_reserve_pct, weighting, weight_cap=weight_cap,
                    risk_stats=risk_stats, tilt_strength=tilt_strength)

    plan = TradePlan(
        plan_date=_plan_date_str,
        created_at=datetime.now().isoformat(timespec="seconds"),
        source=source,
        pool=pool,
        min_score=round(float(min_score), 1),
        regime=regime,
        targets=targets,
        cash_reserve_pct=cash_reserve_pct,
        max_positions=max_positions,
        weighting=weighting,
        raw_rank=raw_rank,
        picks_date=str(fresh.get("report_date") or ""),
        picks_lag_days=int(fresh.get("lag_days") or 0),
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


def backfill_prices(plan_date: Optional[str] = None, plan_dir: Optional[Path] = None) -> Optional[TradePlan]:
    """给已有计划补上参考价（不改动标的与权重）。返回更新后的计划，无计划返回 None。

    场景：历史计划生成时 report 没透传价格，导致 10 只目标 price 全 0.0。
    这里按现有标的代码拉收盘价回填，让「跳空不追」校验重新可用。
    """
    plan = load_trade_plan(plan_date, plan_dir)
    if plan is None:
        logger.warning("回填价格失败：找不到计划 %s", plan_date or "(最新)")
        return None
    closes = _fetch_close_prices([t.code for t in plan.targets])
    n = 0
    for t in plan.targets:
        if float(t.price or 0) <= 0 and t.code in closes:
            t.price = closes[t.code]
            n += 1
    if n:
        save_trade_plan(plan, plan_dir=plan_dir)
        logger.info(f"📌 计划 {plan.plan_date} 回填参考价 {n} 只")
    else:
        logger.info(f"计划 {plan.plan_date} 无需回填（{len(plan.targets)} 只均已有价格或取价失败）")
    return plan


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


_PICKS_NAME_RE = re.compile(r"daily_picks_(\d{4})(\d{2})(\d{2})(?:_(\d{4}))?\.json$")


def _report_date_of(path: Path) -> str:
    """从 ``daily_picks_YYYYMMDD_HHMM.json`` 取出报告日期（``YYYY-MM-DD``）。

    刻意用**文件名**而不是 ``generated_at`` 或 mtime：文件名是生成端写死的
    业务日期，而 mtime 会被拷贝/同步改写，``generated_at`` 在部分历史报告里
    缺失。取不到就返回空串（调用方据此判定「未知」，不猜）。
    """
    m = _PICKS_NAME_RE.search(Path(path).name)
    if not m:
        return ""
    y, mo, d = m.group(1), m.group(2), m.group(3)
    try:
        return date(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def picks_staleness(meta: dict, plan_date: Optional[str] = None) -> dict:
    """评估这批 picks 相对计划日陈旧到什么程度（按**交易日**，不是日历日）。

    时点约定（与 ``is_valid_for`` 同一套）：D-1 收盘后跑选股 → 生成 D 日的计划。
    所以 ``报告日 = D-1、计划日 = D`` 时滞后 **1 个交易日**，这是正常值。
    滞后 ≥2 说明中间至少有一个交易日没跑成选股作业，计划吃的仍是旧票 ——
    而计划文件本身看起来毫无异常，这是「安静地不做事」最典型的形态。

    Returns::

        {"report_date": "2026-11-03", "plan_date": "2026-11-06",
         "lag_days": 3, "stale": True,
         "skipped_reports": 0, "reason": "选股报告 2026-11-03 距今 3 个交易日…"}

    ``stale`` 只在**能证明**滞后 ≥2 时为 True；拿不到报告日期时保持 False
    并把原因写进 ``reason``——宁可少报，也不要每次重建计划都叫一次。
    """
    rep = str((meta or {}).get("report_date") or "")
    pd_ = str(plan_date or "")
    out = {
        "report_date": rep, "plan_date": pd_, "lag_days": 0,
        "stale": False, "skipped_reports": int((meta or {}).get("skipped_reports", 0) or 0),
        "reason": "",
    }
    if not rep:
        out["reason"] = "选股报告日期未知（无报告或文件名不合规），无法判断陈旧度"
        return out
    if not pd_:
        pd_ = resolve_target_date()
        out["plan_date"] = pd_
    try:
        r_d = datetime.strptime(rep, "%Y-%m-%d").date()
        p_d = datetime.strptime(pd_, "%Y-%m-%d").date()
    except Exception:  # noqa: BLE001
        out["reason"] = f"日期解析失败（report={rep!r}, plan={pd_!r}）"
        return out

    try:
        from src.trading.calendar import trading_days_between
        lag = int(trading_days_between(r_d, p_d))
    except Exception:  # noqa: BLE001
        # 日历不可用时退化成自然日差（会偏大，只会更容易告警，不会漏报）
        lag = max(0, (p_d - r_d).days)
    out["lag_days"] = lag

    if lag >= 2:
        out["stale"] = True
        out["reason"] = (
            f"选股报告 {rep} 距今 {lag} 个交易日（正常应为 1）—— "
            "中间有交易日的选股作业没跑成，计划用的是旧票"
        )
        if out["skipped_reports"]:
            out["reason"] += (
                f"；另有 {out['skipped_reports']} 份更晚的报告是空的（选股失败），已被跳过"
            )
    return out


def load_latest_picks(report_dir: Optional[Path] = None) -> tuple[List[dict], dict]:
    """读取最近一份**有 picks 的**每日推荐报告（离线兜底数据源）。

    ⚠️ 两个容易忽略的行为，都在 meta 里显式交代，别让下游蒙在鼓里：

    1. **会跳过空报告**。最新那份报告若是选股失败（``picks=[]``），这里会
       继续往前找 —— 于是「今天选不出来」被悄悄替换成「用前天的票」。
       ``meta["skipped_reports"]`` 记录跳过了几份，供告警使用。
    2. **不做新鲜度判断**。取哪份是「最近有内容的」，是否陈旧由调用方用
       :func:`picks_staleness` 裁决（``build_trade_plan`` 已经这么做了）。
    """
    from config.settings import config as app_config
    d = Path(report_dir) if report_dir else Path(app_config.scheduler.report_dir)
    if not d.exists():
        return [], {}
    files = sorted(d.glob("daily_picks_*.json"))
    skipped = 0
    for f in reversed(files):
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            skipped += 1          # 解析失败也算「这份没内容」
            continue
        picks = rep.get("picks") or []
        if not picks:
            skipped += 1
            continue
        return picks, {
            "pool": rep.get("pool", ""),
            "min_score": rep.get("min_score", 0),
            "regime": rep.get("regime") or "",
            "source": "daily_pick",
            "generated_at": rep.get("generated_at", ""),
            "report_date": _report_date_of(f),
            "report_file": f.name,
            "skipped_reports": skipped,
        }
    return [], {"skipped_reports": skipped}


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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", nargs="?", const="", default=None,
                   help="回填已有计划的参考价（默认最新，可指定 plan_date），不改动标的/权重")
    ap.add_argument("--plan-date", dest="plan_date", default=None)
    a = ap.parse_args()

    if a.backfill is not None:
        plan = backfill_prices(a.backfill or a.plan_date)
        print(plan.summary() if plan else "未找到计划")
    else:
        # 与闭环 / 调度器同源：显式吃生产配置，不吃函数默认值（equal 权重）。
        from config.settings import config as _app_cfg

        _kw = {}
        _tl = getattr(_app_cfg, "trading_loop", None)
        if _tl is not None:
            _kw = dict(
                max_positions=int(_tl.max_positions),
                cash_reserve_pct=float(_tl.cash_reserve_pct),
                weighting=str(_tl.weighting),
                tilt_strength=float(_tl.tilt_strength),
                weight_cap=float(_tl.weight_cap),
            )
        p = build_trade_plan(write=True, **_kw)
        print(p.summary())

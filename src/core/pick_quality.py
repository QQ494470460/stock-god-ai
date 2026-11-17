"""选股质量审计：回答「这批票到底还有没有 alpha」，而不是「分数看起来散不散」。

## 为什么必须有这个模块（2026-11-18 实测，这条是本项目的核心诊断）

``signal_strength``（见 ``src/core/signal_strength.py``）用**截面区分度 σ** 和
**达标只数** 判断"今天信号好不好"。它是个**代理指标**，而代理指标会骗人：

| 指标 | 2026-11-18 实测 |
|------|-----------------|
| 截面 σ | 3.27（满分基准 2.0 → 封顶，看起来"极其分散"）|
| 严格达标 | 0 只 |
| **全池 RankIC（48 日）** | **+0.0237，t=0.89（不显著）** |
| **全池 RankIC（近 20 日）** | **−0.0099（已转负）** |
| 月度走势 | 8 月 +0.127 → 9 月 +0.021 → **10 月 −0.035** |

**分数很分散 ≠ 排序有预测力。** 一个把票随机打散到 30~90 分的模型，σ 可以很大，
但分数与未来收益的秩相关依然是 0 —— 用户体感"选股质量太差"，说的正是这件事：
榜单内部顺序没有信息量，从中挑 8 只等于随机挑。

更直接的证据（用 ``data/prediction_history.json`` 里 21 期、每期 20 只的历史推荐
回算名单内 RankIC）：

- 5 日：Rank IC 均值 **−0.005**，为正仅 38% 的期数；按分数分三组，
  头部（top）平均 **−0.09%** < 中段（mid）**+0.74%** —— 组间完全没有单调性。
- 但**名单整体**对基准的超额是 **+1.15%（5 日）/ +1.56%（10 日）**，
  为正期数 71% / 67%。

结论：**「谁进名单」这一步有效，「名单里谁排第一」是噪声。** 所以质量改进的
着力点不在阈值、不在排序，而在模型本身的横截面预测力 —— 而预测力是可以被
持续度量的，本模块就是那把尺子。

## 它同时修掉的一个真故障

2026-11-18 排查发现 ``data/rolling_ic_status.json`` 停在
``{"status": "running", "started_at": "2026-11-13T19:15:00"}``，而
``rolling_ic_log.json`` 最后一条是 **2026-11-04** —— 质量监控已经死了 7 个交易日，
状态文件却一直显示"正在运行"。原因：``run_backfill`` 开头写 ``running``，进程被
``taskkill /F`` 杀掉时没有任何 ``finally`` 能执行；而 ``catch_up_missed()`` 只补
``data_update``/``daily_pick``，晚间 19:15 的 ``rolling_ic`` 错过即永久丢失。
于是「监控死了」这件事本身也没人知道。:func:`monitor_freshness` 专门判这个。

## 纪律

本模块**只读观测、只写一个诊断文件**，绝不参与选股与下单，任何异常都吞掉。
默认不改变任何生产行为（与全局铁律一致）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.rolling_ic import (
    IC_LOG,
    STATUS_FILE,
    STATUS_RUNNING_STALE_HOURS,
    _stats_from_entries,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = ROOT / "data" / "pick_quality.json"

# ---- 判定阈值（每一项都有出处，改这里必须同步改本文件文档） ----
IC_FLOOR = 0.02            # 日频横截面 RankIC 均值「可用」下限。业界经验 0.02~0.05
                           # 为可用区间；本项目 48 日均值 0.0237 恰好压在线上。
T_SIG = 2.0                # 双侧 95% 显著性（t 分布大样本近似）
MIN_DAYS = 20              # 少于 20 个评分日不做判定（否则 t 值毫无意义）
RECENT_WINDOW = 20         # 「近期」= 最近 20 个评分日（约一个月）
# 全池口径的下限（zz500 生产实测 423~430 只）。低于这个数就是**抽样口径**，
# 与生产口径不可混算 —— 见 :func:`ic_stats` 的 universe 拆分。
FULL_UNIVERSE_MIN = 300
# 台账落后多少个交易日算「监控停摆」。
# ⚠️ 不能只看"落后几天"：前向收益需要 ``horizon`` 个交易日的未来数据，
# 所以**健康的台账天生就落后 horizon 天**（horizon=5 时，今天最多只能算到
# 5 个交易日前的评分日）。2026-11-18 第一版把阈值设成 2 天，结果回填刚刚成功
# 也照样报"停摆 6 个交易日"—— 一个永久假告警。真正口径是 horizon + 余量。
STALE_SLACK_TRADING_DAYS = 2
DEFAULT_HORIZON = 5        # 与 run_backfill / job_rolling_ic 的默认一致
# 「running 卡多久算进程已死」的唯一口径在 rolling_ic 里（它才是状态文件的写入方），
# 这里只做别名，绝不另立一套阈值 —— 两处阈值不一致会导致"一个说死一个说活"。
RUNNING_STALE_HOURS = STATUS_RUNNING_STALE_HOURS

_VERDICT_CN = {
    "effective": "有效",
    "marginal": "偏弱（有方向但不显著）",
    "decayed": "已衰减",
    "insufficient": "样本不足",
    "no_data": "无数据",
}
_VERDICT_TONE = {
    "effective": "up",
    "marginal": "warn",
    "decayed": "down",
    "insufficient": "plain",
    "no_data": "plain",
}


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (v != v or v in (float("inf"), float("-inf"))) else v


def _load_json(path: Path):
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return None


def ic_verdict(stats: Dict[str, Any]) -> str:
    """把整体 + 近期统计判成一个标签（纯函数，可离线单测）。

    判定顺序刻意如此：
    1. 样本不够 → ``insufficient``（先别下结论）
    2. 整体均值为负 → ``decayed``（方向已经反了）
    3. 整体为正但近期转负 → ``decayed``（**曾经可用、现在失效** —— 本项目当前状态）
    4. 显著且过线 → ``effective``
    5. 其余 → ``marginal``
    """
    n = int(stats.get("n") or 0)
    mean = _f(stats.get("mean_ic"))
    t = _f(stats.get("t_stat"))
    recent = _f(stats.get("recent_mean_ic"))
    if not n or mean is None:
        return "no_data"
    if n < MIN_DAYS:
        return "insufficient"
    if mean <= 0:
        return "decayed"
    if recent is not None and recent <= 0:
        return "decayed"
    if t is not None and t >= T_SIG and mean >= IC_FLOOR:
        return "effective"
    if t is None and mean >= IC_FLOOR:
        # 零方差：每期 IC 完全相同 → t 不可算，但一致性已拉满，按均值判
        return "effective"
    return "marginal"


def universe_of(entry: Dict[str, Any]) -> str:
    """一条台账属于哪个股票池口径：``full``（全池，≈生产）还是 ``sample``（抽样）。"""
    try:
        n = int(entry.get("n") or 0)
    except (TypeError, ValueError):
        n = 0
    return "full" if n >= FULL_UNIVERSE_MIN else "sample"


_UNIVERSE_CN = {"full": "全池", "sample": "抽样"}


def _universe_stats(rows: List[Dict[str, Any]], recent_window: int) -> Dict[str, Any]:
    """同一口径内的统计（整体 + 近期 + 月度）。行不得为空。"""
    base = _stats_from_entries(rows)
    w = max(1, int(recent_window))
    recent_ics = [float(_f(r.get("ic"))) for r in rows[-w:]]
    recent_mean = float(np.mean(recent_ics)) if recent_ics else None
    recent_pos = (sum(1 for v in recent_ics if v > 0) / len(recent_ics)) if recent_ics else None

    monthly: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = str(r.get("date"))[:7]
        bucket = monthly.setdefault(key, {"n": 0, "sum": 0.0, "pos": 0})
        v = float(_f(r.get("ic")))
        bucket["n"] += 1
        bucket["sum"] += v
        bucket["pos"] += 1 if v > 0 else 0

    return {
        "n": int(base.get("total") or len(rows)),
        "mean_ic": _f(base.get("mean_ic")),
        "ir": _f(base.get("ir_mean_std")),
        "t_stat": _f(base.get("t_stat")),
        "pos_rate": _f(base.get("pos_rate")),
        "last_date": base.get("last_date") or rows[-1].get("date"),
        "first_date": str(rows[0].get("date")),
        "universe_n": int(rows[-1].get("n") or 0),
        "recent_window": len(recent_ics),
        "recent_mean_ic": None if recent_mean is None else round(recent_mean, 4),
        "recent_pos_rate": None if recent_pos is None else round(recent_pos, 3),
        "monthly": {
            k: {"n": v["n"], "mean_ic": round(v["sum"] / v["n"], 4),
                "pos_rate": round(v["pos"] / v["n"], 3)}
            for k, v in sorted(monthly.items())
        },
    }


def ic_stats(
    entries: Optional[List[Dict[str, Any]]],
    recent_window: int = RECENT_WINDOW,
) -> Dict[str, Any]:
    """从 IC 台账条目算出「预测力」全套统计（按股票池口径拆分 + 判定）。

    ## 为什么要按口径拆分（2026-11-18 实测，这个坑差点让结论反向）

    台账里有**两个生产者、两种股票池**：``run_backfill`` 默认 ``sample=150``，
    而生产选股扫的是全池 ~425 只。实测两套口径的结论完全相反：

    | 口径 | 天数 | 均值 RankIC | 跨度 |
    |------|------|-------------|------|
    | 抽样 ~150 | 41 | **+0.0010** | 08-21 ~ 11-04 |
    | 全池 ~425 | 7 | **+0.1569** | 09-07 ~ 10-23 |

    把两者混在一根序列里求平均（原实现就是这么做的，得 +0.0237）是
    **Simpson 悖论**：均值既不描述抽样口径，也不描述生产口径。更糟的是"近 20 日
    转负"这样的结论会完全由"最近几天的口径恰好不同"造成 —— 看起来像模型衰减，
    其实只是换了把尺子。

    所以这里**只认一个口径**：优先 ``full``（≈生产）；``full`` 不足则退回
    ``sample``，但明确标注 ``confidence="low"`` 且 ``verdict`` 只会是
    ``insufficient`` —— **宁可不判定，也不拿错口径下结论**。

    Returns:
        结构化 dict；无数据时 ``verdict="no_data"``，绝不抛异常。
    """
    rows_all = [r for r in (entries or [])
                if isinstance(r, dict) and r.get("date") and _f(r.get("ic")) is not None]
    rows_all.sort(key=lambda r: str(r.get("date")))
    if not rows_all:
        return {"verdict": "no_data", "verdict_cn": _VERDICT_CN["no_data"],
                "tone": "plain", "n": 0, "mean_ic": None, "t_stat": None,
                "universe": "none", "universes": {}, "mixed": False,
                "note": "没有可用的 RankIC 台账（data/rolling_ic_log.json 缺失或为空）"}

    groups: Dict[str, List[Dict[str, Any]]] = {"full": [], "sample": []}
    for r in rows_all:
        groups[universe_of(r)].append(r)
    # 先过滤空分组再统计：空组没有"最后一条"，进去就会越界
    per = {k: _universe_stats(g, recent_window)
           for k, g in groups.items() if g}

    # 主口径：全池优先；全池样本够就不看抽样（口径不可混）
    if per.get("full", {}).get("n", 0) > 0:
        primary = "full"
    else:
        primary = "sample"
    out: Dict[str, Any] = dict(per[primary])
    out.update({
        "universe": primary,
        "universe_cn": _UNIVERSE_CN[primary],
        "universes": per,
        "mixed": len(per) > 1,
        "ic_floor": IC_FLOOR,
        "t_sig": T_SIG,
    })
    # 只有「口径与生产一致 + 天数够」才允许给出有效结论
    verdict = ic_verdict(out) if primary == "full" else "insufficient"
    out["confidence"] = "ok" if (primary == "full" and out["n"] >= MIN_DAYS) else "low"
    out["verdict"] = verdict
    out["verdict_cn"] = _VERDICT_CN[verdict]
    out["tone"] = _VERDICT_TONE[verdict]
    out["note"] = _verdict_note(out)
    return out


def _verdict_note(s: Dict[str, Any]) -> str:
    n, mean, t = s.get("n"), s.get("mean_ic"), s.get("t_stat")
    recent, rw = s.get("recent_mean_ic"), s.get("recent_window") or RECENT_WINDOW
    uni = s.get("universe_cn") or "—"
    mix = ""
    per = s.get("universes") or {}
    if s.get("mixed"):
        bits = []
        for k in ("full", "sample"):
            v = per.get(k) or {}
            if v.get("n"):
                bits.append(f"{_UNIVERSE_CN[k]} {v['n']} 日均值 "
                            f"{v['mean_ic']:+.4f}")
        if bits:
            mix = ("。⚠️ 台账混有两种口径：" + " / ".join(bits)
                   + " —— 混合平均没有意义，这里只用一种口径下结论")
    mean_s = f"{mean:+.4f}" if mean is not None else "—"
    t_s = f"{t:.2f}" if t is not None else "—"
    if s.get("verdict") == "effective":
        return (f"{uni}口径 {n} 个评分日 RankIC 均值 {mean_s}（t={t_s}），达到可用线 "
                f"{IC_FLOOR}——排序有预测力")
    if s.get("verdict") == "decayed":
        if mean is not None and mean <= 0:
            return (f"{uni}口径 {n} 个评分日 RankIC 均值 {mean_s}（t={t_s}）已经为负——"
                    f"分数与实际收益反向，排序不但没信息，还可能有害{mix}")
        return (f"{uni}口径全期均值 {mean_s} 尚为正，但近 {rw} 个评分日均值 "
                f"{recent:+.4f}——**曾经可用、现在已失效**，需重训或降仓{mix}")
    if s.get("verdict") == "insufficient":
        if s.get("universe") != "full":
            return (f"台账里只有抽样口径（{n} 个评分日），**与生产口径（全池 ~425 只）"
                    f"不是一把尺子**（实测前缀抽样均值 +0.0010 vs 全池 +0.1569），"
                    f"不能据此判断生产预测力{mix}。"
                    f"每日 19:15 的 rolling_ic 作业已改为全池口径，攒够 "
                    f"{MIN_DAYS} 个交易日即可判定；要立刻补可跑 "
                    f"scripts/rolling_ic_backfill.py --days 30 --sample 0")
        return (f"{uni}口径仅 {n} 个评分日（<{MIN_DAYS}），均值 {mean_s}、t={t_s} —— "
                f"方向可供参考，但样本不够，不能判定{mix}")
    return (f"{uni}口径 {n} 个评分日 RankIC 均值 {mean_s}（t={t_s} < {T_SIG}）——"
            f"方向为正但不显著，不能据此认为排序有效{mix}")


LIST_MIN_PERIODS = 10      # 名单内排序 IC 的最少期数（实测可用 21 期）


def list_ic_stats(
    records: Optional[List[Dict[str, Any]]],
    horizon: str = "5",
    min_periods: int = LIST_MIN_PERIODS,
) -> Dict[str, Any]:
    """**名单内部**的排序 IC：从候选里挑前 N 只时，名次到底有没有用。

    为什么单独算这一层（这是「选股质量太差」体感的直接来源）
    ------------------------------------------------------
    全池 RankIC 回答的是"模型能不能区分好坏股票"；但用户实际买到的是
    **名单里的前 8 名**。这是两个不同的问题：实测 2026-11-18 用
    ``data/prediction_history.json`` 里 21 期真实生产名单回算，结果是

    - 5 日：名单内 RankIC 均值 **−0.005**，为正仅 38% 的期数；
      按分数三分组，头部 −0.09% < 中段 **+0.74%**，完全无单调性。
    - 但名单整体对基准超额 **+1.15%（5 日）/ +1.56%（10 日）**，为正 71% / 67%。

    即 **「谁进名单」有效，「名单里谁排第一」是噪声**。这条结论直接决定了下单
    方式：名次不可信时，"取前 8 名集中下注"等于随机取，应改按风险分散建仓。

    数据源是 ``prediction_history.json`` 的 ``verifications[h]["details"]``
    （每条含 ``score`` 与 ``actual_return``），是**生产真实推荐**，
    口径天然一致 —— 不像 ``rolling_ic_log`` 那样混着抽样/全池两套尺子。

    Returns:
        ``{n_periods, mean_ic, t_stat, pos_rate, quantiles, verdict, ...}``
    """
    from src.core.rolling_ic import _spearman

    per: List[Dict[str, Any]] = []
    buckets: Dict[str, List[float]] = {"top": [], "mid": [], "bot": []}
    for rec in (records or []):
        if not isinstance(rec, dict):
            continue
        ver = (rec.get("verifications") or {}).get(str(horizon))
        if not isinstance(ver, dict):
            continue
        det = [d for d in (ver.get("details") or [])
               if isinstance(d, dict) and _f(d.get("score")) is not None
               and _f(d.get("actual_return")) is not None]
        if len(det) < 5:
            continue
        scores = [float(_f(d["score"])) for d in det]
        rets = [float(_f(d["actual_return"])) for d in det]
        ic = _spearman(scores, rets)
        if ic == ic:
            per.append({"date": str(rec.get("scan_date") or ""),
                        "ic": round(float(ic), 4), "n": len(det)})
        # 三分位：按分数降序（并列时保持原序，避免抖动）
        order = sorted(range(len(det)), key=lambda i: -scores[i])
        k = max(1, len(order) // 3)
        segs = {"top": order[:k], "mid": order[k:2 * k], "bot": order[2 * k:]}
        for lab, idx in segs.items():
            buckets[lab].extend(rets[i] for i in idx)

    if not per:
        return {"n_periods": 0, "mean_ic": None, "t_stat": None, "pos_rate": None,
                "quantiles": {}, "verdict": "no_data",
                "verdict_cn": _VERDICT_CN["no_data"], "tone": "plain",
                "horizon": str(horizon), "note": "没有可用的历史推荐验证明细"}

    ics = np.asarray([p["ic"] for p in per], dtype=float)
    n = int(ics.size)
    mean = float(ics.mean())
    sd = float(ics.std(ddof=1)) if n > 1 else float("nan")
    se = sd / (n ** 0.5) if n > 1 and sd == sd and sd > 0 else float("nan")
    tstat = mean / se if se == se and se > 0 else None
    pos = float((ics > 0).mean())
    quantiles = {k: (round(float(np.mean(v)), 4) if v else None)
                 for k, v in buckets.items()}

    if n < min_periods:
        verdict = "insufficient"
    elif mean <= 0:
        verdict = "no_edge"        # 名次无信息甚至反向
    elif tstat is None:
        # IC 序列零方差（每期完全一样）→ t 不可算，但一致性已拉满。
        # 这时按均值是否过线判，别把"最稳"误判成"没信号"。
        verdict = "effective" if mean >= IC_FLOOR else "no_edge"
    elif tstat >= T_SIG and mean >= IC_FLOOR:
        verdict = "effective"
    else:
        verdict = "no_edge"        # 正但远不显著 → 视同无排序力（宁严不松）
    note = _list_note(verdict, n, mean, tstat, quantiles, horizon)
    return {
        "n_periods": n,
        "mean_ic": round(mean, 4),
        "t_stat": None if tstat is None else round(float(tstat), 2),
        "pos_rate": round(pos, 3),
        "quantiles": quantiles,
        "horizon": str(horizon),
        "verdict": verdict,
        "verdict_cn": {"effective": "名次有效", "no_edge": "名次无信息",
                       "insufficient": "样本不足"}[verdict],
        "tone": {"effective": "up", "no_edge": "down",
                 "insufficient": "plain"}[verdict],
        "note": note,
    }


def _list_note(verdict: str, n: int, mean: float, tstat, quantiles: Dict[str, Any],
               horizon: str) -> str:
    top, mid, bot = (quantiles.get("top"), quantiles.get("mid"), quantiles.get("bot"))
    q_txt = ""
    if None not in (top, mid, bot):
        q_txt = (f"；按分数三分组，头部 {top * 100:+.2f}% / 中段 {mid * 100:+.2f}%"
                 f" / 尾部 {bot * 100:+.2f}%")
        if top <= mid:
            q_txt += " —— 头部不如中段，组间没有单调性"
    t_s = f"{tstat:.2f}" if tstat is not None else "—"
    if verdict == "effective":
        return (f"{n} 期、持有 {horizon} 日：名单内 RankIC {mean:+.4f}（t={t_s}）"
                f"—— 名次有预测力，可以按排名挑前 N 只{q_txt}")
    if verdict == "insufficient":
        return f"仅 {n} 期验证明细（<{LIST_MIN_PERIODS}），不足以判断名次有没有预测力"
    return (f"{n} 期、持有 {horizon} 日：名单内 RankIC {mean:+.4f}（t={t_s}）"
            f"—— **名次没有预测力**，从候选里挑前 N 只约等于随机挑{q_txt}"
            f"。这正是『名单给了 8 只、却挑不准』的量化解释")


def monitor_freshness(
    entries: Optional[List[Dict[str, Any]]] = None,
    status: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    horizon: int = DEFAULT_HORIZON,
) -> Dict[str, Any]:
    """判断质量监控**本身**是否还活着（这是本项目真踩过的坑）。

    两个独立信号：
    1. 台账落后：``last_date`` 距今天数（优先用交易日，日历不可用时退化用自然日）。
       门槛是 ``horizon + STALE_SLACK_TRADING_DAYS``，不是固定 2 天 —— 前向收益
       本来就要等 ``horizon`` 个交易日，健康的台账也必然落后那么多。
    2. 状态卡死：``status == "running"`` 但 ``started_at`` 已过去很久 —— 进程被
       ``taskkill /F`` 杀掉时不会有 ``finally``，状态就永久停在 running。

    Returns:
        ``{"state": "ok"|"stale"|"stuck"|"aborted"|"empty", "stalled": bool, ...}``
    """
    now = now or datetime.now()
    rows = [r for r in (entries or []) if isinstance(r, dict) and r.get("date")]
    rows.sort(key=lambda r: str(r.get("date")))
    last_date = str(rows[-1]["date"]) if rows else None
    status = status if isinstance(status, dict) else {}
    st = str(status.get("status") or "")

    # 卡死判定：running 但 started_at 太久
    stuck = False
    running_hours = None
    if st == "running":
        started = status.get("started_at")
        try:
            started_dt = datetime.fromisoformat(str(started))
            running_hours = round((now - started_dt).total_seconds() / 3600.0, 1)
            stuck = running_hours >= RUNNING_STALE_HOURS
        except Exception:  # noqa: BLE001
            stuck = True          # 连 started_at 都没有，等同于卡死

    # 落后判定：优先交易日
    lag_days = None
    lag_trading = None
    if last_date:
        try:
            d_last = datetime.strptime(last_date, "%Y-%m-%d").date()
            lag_days = (now.date() - d_last).days
            try:
                from src.trading import calendar as trade_cal
                lag_trading = int(trade_cal.trading_days_between(d_last, now.date()))
            except Exception:  # noqa: BLE001
                lag_trading = None
        except Exception:  # noqa: BLE001
            pass

    # 健康台账天生落后 horizon 个交易日（前向收益还没走完），所以门槛必须加余量
    try:
        expect_lag = max(0, int(horizon)) + STALE_SLACK_TRADING_DAYS
    except (TypeError, ValueError):
        expect_lag = DEFAULT_HORIZON + STALE_SLACK_TRADING_DAYS

    if not rows and st not in ("running", "aborted"):
        state = "empty"
    elif stuck:
        state = "stuck"
    elif st == "aborted":
        state = "aborted"
    elif lag_trading is not None and lag_trading > expect_lag:
        state = "stale"
    elif lag_trading is None and lag_days is not None and lag_days > expect_lag * 2:
        state = "stale"
    else:
        state = "ok"

    note = {
        "ok": (f"质量监控正常（最后评分日 {last_date}"
               + (f"，落后 {lag_trading} 个交易日，在 horizon={horizon} 的合理范围内"
                  if lag_trading is not None else "") + "）"),
        "stale": (f"质量监控停摆：台账最后一条是 {last_date}"
                  + (f"，已落后 {lag_trading} 个交易日" if lag_trading is not None else "")
                  + f"（超过 horizon {horizon} + 余量 {STALE_SLACK_TRADING_DAYS} 天）"
                  + "。晚间 rolling_ic 作业错过即永久丢失（catch_up 只补 data_update/daily_pick）"),
        "stuck": (f"质量监控卡死：状态停在 running 已 {running_hours}h"
                  f"（最后评分日 {last_date}）——回填进程多半被强杀，"
                  f"状态没人改回"),
        "aborted": (f"上次质量回填被强制终止（{status.get('reason') or '原因未记录'}），"
                    f"台账最后一条是 {last_date}"
                    + (f"，已落后 {lag_trading} 个交易日" if lag_trading is not None else "")
                    + "——需要手动跑一次 scripts/rolling_ic_backfill.py"),
        "empty": "从未成功产出过 RankIC 台账",
    }[state]
    return {
        "state": state,
        "stalled": state in ("stale", "stuck", "aborted", "empty"),
        "status": st or None,
        "last_date": last_date,
        "lag_days": lag_days,
        "lag_trading_days": lag_trading,
        "running_hours": running_hours,
        "expect_lag_trading_days": expect_lag,
        "horizon": horizon,
        "note": note,
    }


def quality_report(
    log_path: Optional[Path] = None,
    status_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    now: Optional[datetime] = None,
    history_path: Optional[Path] = None,
    list_horizon: str = "5",
    ic_horizon: int = DEFAULT_HORIZON,
) -> Dict[str, Any]:
    """一次算全：全池预测力 + 名单内名次预测力 + 监控存活状态。

    落 ``data/pick_quality.json``。任何异常都吞掉并返回 ``verdict="no_data"``
    —— 观测代码不许拖垮选股。
    """
    try:
        entries = _load_json(Path(log_path) if log_path else IC_LOG) or []
        if not isinstance(entries, list):
            entries = []
        status = _load_json(Path(status_path) if status_path else STATUS_FILE)
        stats = ic_stats(entries)
        fresh = monitor_freshness(entries, status, now=now, horizon=ic_horizon)
        hist = _load_json(Path(history_path) if history_path
                          else ROOT / "data" / "prediction_history.json")
        out = dict(stats)
        out.update({
            "list_ic": list_ic_stats(hist if isinstance(hist, list) else [],
                                     horizon=list_horizon),
            "monitor": fresh,
            "computed_at": (now or datetime.now()).isoformat(timespec="seconds"),
        })
        target = Path(out_path) if out_path else OUT_FILE
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            import os
            os.replace(tmp, target)      # 原子替换：写一半被关机也不会留下坏文件
        except Exception as e:  # noqa: BLE001
            logger.debug(f"选股质量诊断写入失败（不影响选股）: {e}")
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug(f"选股质量诊断失败（不影响选股）: {e}")
        return {"verdict": "no_data", "verdict_cn": _VERDICT_CN["no_data"],
                "tone": "plain", "n": 0, "note": f"诊断异常: {e}"}


def format_pick_quality(q: Optional[Dict[str, Any]]) -> str:
    """多行文本（CLI / 日报 / 看板共用一套口径）。"""
    if not isinstance(q, dict) or q.get("verdict") in (None, "no_data"):
        return "🎯 选股预测力：无 RankIC 台账，无法评估"
    mean = q.get("mean_ic")
    t = q.get("t_stat")
    uni = q.get("universe_cn") or "—"
    lines = [
        f"🎯 选股预测力：{q.get('verdict_cn')}"
        f"（{uni}口径 RankIC {mean:+.4f}，t={t:.2f}，{q.get('n')} 日）"
        if mean is not None and t is not None else
        f"🎯 选股预测力：{q.get('verdict_cn')}",
        f"  · 为正占比 {_pct(q.get('pos_rate'))} · IR {_num(q.get('ir'), 3)}"
        f" · 可用线 {IC_FLOOR} / 显著线 t={T_SIG:.1f}",
        f"  · 近 {q.get('recent_window')} 日均值 {_num(q.get('recent_mean_ic'), 4, '+')}"
        f"（为正 {_pct(q.get('recent_pos_rate'))}）",
    ]
    per = q.get("universes") or {}
    if len(per) > 1:
        lines.append("  · 口径拆分：" + " / ".join(
            f"{_UNIVERSE_CN[k]} {v['n']} 日 {_num(v.get('mean_ic'), 4, '+')}"
            for k, v in per.items() if v.get("n")))
    monthly = q.get("monthly") or {}
    if monthly:
        tail = list(monthly.items())[-3:]
        lines.append("  · 月度：" + " / ".join(
            f"{k} {v['mean_ic']:+.3f}" for k, v in tail))
    if q.get("note"):
        lines.append(f"  · {q['note']}")
    # 名单内排序 IC（直接对应"从候选里挑前 N 只"这一步）
    li = q.get("list_ic") or {}
    if li.get("n_periods"):
        lines.append(
            f"  · 名单内名次（{li.get('horizon')} 日 / {li.get('n_periods')} 期）："
            f"RankIC {_num(li.get('mean_ic'), 4, '+')}（t={_num(li.get('t_stat'), 2)}）"
            f" → {li.get('verdict_cn')}")
        if li.get("note"):
            lines.append(f"    {li['note']}")
    mon = q.get("monitor") or {}
    if mon.get("stalled"):
        lines.append(f"  · ⚠️ {mon.get('note')}")
    return "\n".join(lines)


def quality_alert(
    q: Optional[Dict[str, Any]],
    signal: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """需要提示时返回聚焦文案；健康返回 ``None``。

    最有价值的一条是**组合判断**：``signal_strength`` 说"分散度很好"，
    而历史预测力已失效 —— 这就是"假强信号"，必须明确点出来，
    否则用户会按"信号不错"加仓。
    """
    if not isinstance(q, dict):
        return None
    verdict = q.get("verdict")
    mon = q.get("monitor") or {}
    li = q.get("list_ic") or {}
    parts: List[str] = []

    if mon.get("stalled"):
        parts.append(f"· {mon.get('note')}")
    # 名单内名次：这条最贴近用户体感（"从候选里挑前 8 只"就是这一步）
    if li.get("verdict") == "no_edge":
        parts.append(f"· {li.get('note')}")
        parts.append("· 建议：既然名次没有预测力，就别『取前 N 名集中下注』——"
                     "把名单当候选池，按行业/相关性分散建仓；"
                     "要真提升质量只能重训模型（scripts/train_transformer.py / "
                     "compute_leg_rankic.py），调阈值或改排名权重都无效")
    elif li.get("verdict") == "effective":
        parts.append(f"· {li.get('note')}")
    if verdict in ("decayed", "marginal", "insufficient"):
        # insufficient 常不是"运气不好"，而是口径不一致或监控停摆造成的 ——
        # 必须说出来，否则用户会把"测不出来"误读成"应该还行"。
        parts.append(f"· {q.get('note')}")
    if not parts:
        return None

    # 假强信号：分数分布看着漂亮，但名次/排序没有预测力
    sig_level = str((signal or {}).get("level") or "")
    _weak_rank = (li.get("verdict") == "no_edge" or verdict in ("decayed", "marginal"))
    if _weak_rank and sig_level in ("strong", "ok"):
        parts.insert(0, (f"· ⚠️ 假强信号：今日截面分散度 σ={(signal or {}).get('dispersion')}"
                         f"（{((signal or {}).get('level_cn'))}），但历史排序预测力判定为"
                         f"「{li.get('verdict_cn') or q.get('verdict_cn')}」"
                         f"—— 分散不等于有预测力"))
    return "🎯 选股质量诊断：\n" + "\n".join(parts)


def _num(v, nd: int = 3, sign: str = "") -> str:
    f = _f(v)
    return f"{f:{sign}.{nd}f}" if f is not None else "—"


def _pct(v) -> str:
    f = _f(v)
    return f"{f * 100:.0f}%" if f is not None else "—"


__all__ = [
    "ic_stats",
    "ic_verdict",
    "list_ic_stats",
    "universe_of",
    "monitor_freshness",
    "quality_report",
    "format_pick_quality",
    "quality_alert",
    "IC_FLOOR",
    "T_SIG",
    "RECENT_WINDOW",
    "STALE_SLACK_TRADING_DAYS",
    "DEFAULT_HORIZON",
    "FULL_UNIVERSE_MIN",
]

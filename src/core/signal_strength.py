"""选股「信号强度」诊断：把"今天这套推荐到底还有没有边际"变成可算、可看、可追溯的数。

## 为什么要有这个模块（2026-11-16 实测证据）

同一套生产参数、同一个 zz500 池，隔 6 个交易日，榜单质量判若两池：

| 日期       | Top40 最高分 | Top40 标准差 | 严格达标 | TOP 平均上涨概率 |
|------------|--------------|--------------|----------|------------------|
| 2026-11-07 | 62.4         | 2.43         | 9 只     | 67.0%            |
| 2026-11-10 | 56.7         | 1.54         | 0 只     | 55.1%            |
| 2026-11-13 | 56.3         | 1.38         | 0 只     | 49.8%            |

`_apply_score_threshold` 的设计是「过关优先 + 排名补足」——达标 0 只时仍按排名补满
8 只（为了保持与回测一致的分散度）。于是用户看到的是「AI 给了 8 只票」，
看不到「这 8 只票的截面区分度已经塌了一半、胜率掉到抛硬币水平」。
"选股质量太差"的体感，一大半就是这么来的：**系统没有说谎，但也没有告诉你真话。**

本模块只做观测与陈述，绝不改动任何选股结果：

1. :func:`compute_signal_strength` —— 结构化指标 + 四档分级（强/良/弱/无）
2. :func:`record_signal_history` —— 每日一行落 ``data/signal_strength_history.jsonl``
3. :func:`signal_trend` —— 与近期历史比，判断是在衰减还是在恢复
4. :func:`format_signal_strength` / :func:`signal_strength_alert` —— 给人看的文本 / 横幅

## 分级口径（阈值全部有据，见上表）

- ``strength``（0~100）= 50% × 达标进度 + 50% × 区分度进度
  - 达标进度 = min(1, 严格达标数 / 目标持仓数) —— "够不够建仓"这个问法
  - 区分度进度 = min(1, 截面标准差 / 2.0) —— 以 11-07 的 σ=2.43 为封顶基准
- ``level``：``strong`` ≥ 75 / ``ok`` ≥ 50 / ``weak`` ≥ 25 / ``none`` < 25
  - 11-07（达标 9、σ2.43）→ 100 强；11-13（达标 0、σ1.38）→ 35.5 弱。
  - 阈值刻意远离这两个实测点（35.5 vs 边界 25 / 50），不搞"卡在刀口上"的分级。

> 铁律：本模块**永不**参与买卖决策。是否"信号弱就不建仓"由
> ``scheduler.require_strict_pass``（默认 False）在选股侧显式决定。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---- 分级基准（与模块文档里的实测表一一对应，改这里必须同步改文档） ----
DISPERSION_FULL_MARK = 2.0     # 截面标准差满分基准（11-07 实测 2.43 → 封顶）
LEVEL_STRONG = 75.0
LEVEL_OK = 50.0
LEVEL_WEAK = 25.0

_HISTORY_NAME = "signal_strength_history.jsonl"
_MAX_HISTORY_LINES = 400       # 约一年半交易日，够算趋势也够小


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (v != v or v in (float("inf"), float("-inf"))) else v


def _stats(xs: Sequence[float]) -> Dict[str, float]:
    """安全的描述统计（空列表返回全 0，绝不抛异常）。"""
    arr = np.asarray([v for v in xs if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def _level_of(strength: float) -> str:
    if strength >= LEVEL_STRONG:
        return "strong"
    if strength >= LEVEL_OK:
        return "ok"
    if strength >= LEVEL_WEAK:
        return "weak"
    return "none"


_LEVEL_CN = {"strong": "强", "ok": "良", "weak": "弱", "none": "无"}
_LEVEL_TONE = {"strong": "up", "ok": "plain", "weak": "warn", "none": "down"}


def compute_signal_strength(
    batch: Optional[Dict[str, Any]],
    min_score: float,
    target_n: int,
) -> Dict[str, Any]:
    """从一整批预测算出「今日信号强度」。

    Args:
        batch: ``{"scores": [...], "prob_up": [...], "prob_down": [...]}``，
            **全截面**（同一天所有被打分的股票），不是 Top40 榜单。
            通常是 ``Predictor.last_batch_scores``。
        min_score: 当日生效阈值（基线 + 进化 + 市场状态）。
        target_n: 目标持仓只数（``trading_loop.max_positions``）。

    Returns:
        结构化 dict（可直接 JSON 进报告）：``level`` / ``strength`` / ``dispersion``
        / ``strict_pass`` / ``te`` 等。数据不足时返回 ``level="unknown"``，
        绝不让诊断本身影响选股。
    """
    out: Dict[str, Any] = {
        "level": "unknown",
        "level_cn": "未知",
        "tone": "plain",
        "strength": None,
        "n": 0,
        "target_n": int(target_n or 0),
        "min_score": round(float(min_score or 0.0), 1),
        "dispersion": None,
        "strict_pass": 0,
        "strict_ratio": None,
        "score_max": None,
        "score_median": None,
        "top_mean": None,
        "top_edge_pp": None,
        "note": "无可用预测明细（batch 为空或结构不符）",
    }
    if not isinstance(batch, dict):
        return out
    scores = [v for v in (batch.get("scores") or []) if _f(v) is not None]
    if len(scores) < 10:
        out["note"] = f"样本不足（{len(scores)} 只 < 10 只），不做信号强度判定"
        out["n"] = len(scores)
        return out

    scores_sorted = sorted((float(v) for v in scores), reverse=True)
    st = _stats(scores_sorted)
    n = st["n"]
    tn = max(1, int(target_n or 1))
    top = scores_sorted[:tn]
    top_mean = float(np.mean(top)) if top else st["mean"]

    strict_pass = int(sum(1 for s in scores_sorted if s >= float(min_score)))
    strict_ratio = strict_pass / n

    strict_progress = min(1.0, strict_pass / float(max(1, tn)))
    disp_progress = min(1.0, st["std"] / DISPERSION_FULL_MARK) if DISPERSION_FULL_MARK > 0 else 0.0
    strength = round(100.0 * (0.5 * strict_progress + 0.5 * disp_progress), 1)

    # 截面边际（可选信息，用于解释"TOP 比全池强多少"，不参与分级）
    up = [v for v in (batch.get("prob_up") or []) if _f(v) is not None]
    top_up = None
    top_edge_pp = None
    if len(up) == len(scores) and up:
        up_sorted_by_score = [float(u) for _, u in sorted(zip(scores, up), key=lambda t: -t[0])]
        top_up = float(np.mean(up_sorted_by_score[:tn]))
        top_edge_pp = round((top_up - float(np.mean(up))) * 100.0, 1)

    level = _level_of(strength)
    # 「一只都没达标」是硬信号：不论区分度多漂亮，今天的推荐都只能靠排名补足，
    # 说成"良/强"会误导用户按常规仓位建仓。2026-11-16 实测就撞上这一档：
    # σ=3.16（很好）但达标 0 只（阈值 58 全池无人达到）→ 必须判弱。
    strict_zero = (strict_pass == 0)
    if strict_zero and level in ("strong", "ok"):
        level = "weak"
    out.update({
        "level": level,
        "level_cn": _LEVEL_CN[level],
        "tone": _LEVEL_TONE[level],
        "strength": strength,
        "n": n,
        "dispersion": round(st["std"], 2),
        "strict_pass": strict_pass,
        "strict_ratio": round(strict_ratio, 4),
        "score_max": round(st["max"], 1),
        "score_median": round(st["median"], 1),
        "top_mean": round(top_mean, 1),
        "top_up_prob": round(top_up * 100.0, 1) if top_up is not None else None,
        "top_edge_pp": top_edge_pp,
        "note": _auto_note(level, strength, strict_pass, n, st["std"], min_score,
                           strict_zero),
    })
    return out


def _auto_note(level: str, strength: float, strict_pass: int, n: int,
               dispersion: float, min_score: float, strict_zero: bool = False) -> str:
    """一句话说明"今天的信号到底什么样"。不带「信号偏弱」这类前缀 ——
    前缀由 level/横幅负责，重复一遍会变成"信号偏弱：信号偏弱：…"。"""
    if level == "strong":
        return (f"信号健康：{n} 只里 {strict_pass} 只达到阈值 {min_score:.0f}，"
                f"截面区分度 {dispersion:.2f}（健康基准 {DISPERSION_FULL_MARK:.1f}）")
    if level == "ok":
        return (f"信号一般：达标 {strict_pass} 只，区分度 {dispersion:.2f}，"
                f"推荐可执行但不宜重仓")
    if level == "weak":
        if strict_zero:
            return (f"全池 {n} 只没有一只达到阈值 {min_score:.0f}，区分度 "
                    f"{dispersion:.2f}（健康基准 {DISPERSION_FULL_MARK:.1f}）—— "
                    f"推荐全部由排名补足，不是达标信号")
        return (f"达标 {strict_pass} 只（阈值 {min_score:.0f}），区分度仅 "
                f"{dispersion:.2f}（健康基准 {DISPERSION_FULL_MARK:.1f}）—— "
                f"推荐由排名补足，不是达标信号")
    return (f"达标 {strict_pass} 只，区分度 {dispersion:.2f}（健康基准 "
            f"{DISPERSION_FULL_MARK:.1f}）—— 当前排名接近随机，"
            f"选出来的票没有统计边际")


# ---------------------------------------------------------------------------
# 历史台账（事件溯源：让"模型什么时候开始失效"可回看）
# ---------------------------------------------------------------------------

def _history_path(data_dir: Optional[Path] = None) -> Path:
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data"
    return root / _HISTORY_NAME


def record_signal_history(
    stats: Dict[str, Any],
    data_dir: Optional[Path] = None,
    as_of: Optional[str] = None,
) -> Optional[Path]:
    """把当天的信号强度追加到台账（同一天重复调用只保留最后一条）。

    幂等：先读尾部，若最后一行 ``date`` 与本次相同则整体替换掉那一行。
    任何异常都吞掉——台账不能拖垮选股（观测代码的第一纪律）。
    """
    if not isinstance(stats, dict) or stats.get("level") in (None, "unknown"):
        return None
    path = _history_path(data_dir)
    rec = {
        "date": as_of or datetime.now().strftime("%Y-%m-%d"),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": stats.get("level"),
        "strength": stats.get("strength"),
        "dispersion": stats.get("dispersion"),
        "strict_pass": stats.get("strict_pass"),
        "n": stats.get("n"),
        "min_score": stats.get("min_score"),
        "top_up_prob": stats.get("top_up_prob"),
        "top_edge_pp": stats.get("top_edge_pp"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        if path.exists():
            try:
                lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except Exception:  # noqa: BLE001
                lines = []
            keep: List[str] = []
            for ln in lines:
                try:
                    if json.loads(ln).get("date") == rec["date"]:
                        continue        # 覆盖同日旧记录
                except Exception:  # noqa: BLE001
                    pass
                keep.append(ln)
            lines = keep
        lines.append(json.dumps(rec, ensure_ascii=False))
        if len(lines) > _MAX_HISTORY_LINES:
            lines = lines[-_MAX_HISTORY_LINES:]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)          # 原子替换，避免写一半被关机腰斩
        return path
    except Exception as e:  # noqa: BLE001
        logger.debug(f"信号强度台账写入失败（不影响选股）: {e}")
        return None


def recent_signal_history(
    limit: int = 30,
    data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """读取最近的信号强度记录（按日期升序）。坏行跳过，绝不抛异常。"""
    path = _history_path(data_dir)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(rec, dict) and rec.get("date"):
                out.append(rec)
    except Exception:  # noqa: BLE001
        return out
    out.sort(key=lambda r: str(r.get("date") or ""))
    return out[-max(1, int(limit)):]


def signal_trend(
    current: Optional[Dict[str, Any]] = None,
    lookback: int = 5,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """与近 ``lookback`` 个交易日对比，判断信号在衰减还是恢复。

    对比的两个量都是无量纲的：``strength``（0~100）与 ``dispersion``（标准差）。
    样本不足 2 天返回 ``state="unknown"``。
    """
    hist = recent_signal_history(limit=max(lookback + 1, 6), data_dir=data_dir)
    cur_date = str((current or {}).get("date") or "")
    prev = [r for r in hist if str(r.get("date") or "") != cur_date]
    prev = prev[-lookback:]
    if len(prev) < 2:
        return {"state": "unknown", "samples": len(prev), "strength_delta": None,
                "dispersion_delta": None, "note": "历史样本不足 2 天，暂不判断趋势"}
    cur = current or {}
    try:
        cur_strength = float(cur.get("strength"))
        cur_disp = float(cur.get("dispersion"))
    except (TypeError, ValueError):
        return {"state": "unknown", "samples": len(prev), "strength_delta": None,
                "dispersion_delta": None, "note": "当前记录缺字段，暂不判断趋势"}
    base_strength = float(np.mean([r.get("strength") or 0.0 for r in prev]))
    base_disp = float(np.mean([r.get("dispersion") or 0.0 for r in prev]))
    ds = round(cur_strength - base_strength, 1)
    dd = round(cur_disp - base_disp, 2)
    if ds <= -15.0:
        state = "decaying"
        note = f"信号正在衰减：强度 {ds:+.1f}、区分度 {dd:+.2f}（对比前 {len(prev)} 日）"
    elif ds >= 15.0:
        state = "improving"
        note = f"信号在恢复：强度 {ds:+.1f}、区分度 {dd:+.2f}（对比前 {len(prev)} 日）"
    else:
        state = "stable"
        note = f"信号持平：强度 {ds:+.1f}、区分度 {dd:+.2f}（对比前 {len(prev)} 日）"
    return {"state": state, "samples": len(prev), "strength_delta": ds,
            "dispersion_delta": dd, "base_strength": round(base_strength, 1),
            "base_dispersion": round(base_disp, 2), "note": note}


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def format_signal_strength(
    stats: Optional[Dict[str, Any]],
    trend: Optional[Dict[str, Any]] = None,
) -> str:
    """把信号强度渲染成多行文本（CLI / 日报 / 看板共用一套口径）。"""
    if not isinstance(stats, dict) or stats.get("level") in (None, "unknown"):
        return "📶 信号强度：数据不足，未判定"
    lines = [
        f"📶 信号强度：{stats.get('level_cn')}（{stats.get('strength')}/100）",
        f"  · 全截面 {stats.get('n')} 只 · 截面区分度 σ={stats.get('dispersion')}"
        f"（健康基准 {DISPERSION_FULL_MARK:.1f}）",
        f"  · 严格达标 {stats.get('strict_pass')} 只 / 阈值 {stats.get('min_score')}"
        f"（目标 {stats.get('target_n')} 只）",
    ]
    if stats.get("top_up_prob") is not None:
        edge = stats.get("top_edge_pp")
        edge_s = f"{edge:+.1f}pp" if isinstance(edge, (int, float)) else "—"
        lines.append(f"  · TOP{stats.get('target_n')} 平均上涨概率 "
                     f"{stats.get('top_up_prob')}%，领先全池 {edge_s}")
    if stats.get("note"):
        lines.append(f"  · {stats['note']}")
    if isinstance(trend, dict) and trend.get("state") not in (None, "unknown"):
        lines.append(f"  · 趋势：{trend.get('note')}")
    return "\n".join(lines)


def signal_strength_alert(
    stats: Optional[Dict[str, Any]],
    trend: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """需要提示用户时返回聚焦文案；信号健康（strong/ok 且不衰减）返回 ``None``。

    「弱」才提示、「无」重提示——避免天天挂横幅导致告警疲劳。
    """
    if not isinstance(stats, dict):
        return None
    level = stats.get("level")
    decaying = isinstance(trend, dict) and trend.get("state") == "decaying"
    if level in ("strong", "ok") and not decaying:
        return None
    if level == "unknown":
        return None
    head = ("⚠️ 今日选股基本无信号" if level == "none" else "⚠️ 今日选股信号偏弱")
    body = stats.get("note") or ""
    extra = ""
    if decaying:
        extra = f"\n· {trend.get('note')}"
    advice = ("\n· 建议：降低仓位或持币观望；如需强制只买达标票，把 "
              "scheduler.require_strict_pass 设为 true")
    return f"{head}：{body}{extra}{advice}"


__all__ = [
    "compute_signal_strength",
    "record_signal_history",
    "recent_signal_history",
    "signal_trend",
    "format_signal_strength",
    "signal_strength_alert",
    "DISPERSION_FULL_MARK",
]

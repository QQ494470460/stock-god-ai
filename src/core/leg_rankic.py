# -*- coding: utf-8 -*-
"""各腿样本外 RankIC 计算 + 动态融合权重派生
==========================================

为「超级集成第六腿：RankIC 动态融合」提供两层能力：

1. compute_leg_rankic()
   - 复用 rolling_ic 的 OOS 切片模式（评分日 d 只用 d 及之前数据，前向收益用
     d+horizon 交易日），逐日算**每条腿**原始 score 与当日横截面超额收益的
     Spearman（= 该腿当日 RankIC），汇总成 data/leg_rankic.json。
   - 每条腿的「原始 score」取自 predictor 融合时顺手记录的
     pred.ensemble_detail["scores"][leg]（blend_one 已逐腿存下），不重训、不重复推理。
   - 需联网拉行情，生产应在收盘后 / 耐久上下文跑（见 scripts/compute_leg_rankic.py）。

2. 权重派生（纯函数，可离线单测）：
   - aggregate_leg_rankic(per_day)        ：逐日 IC → 每腿均值 RankIC / IR / 天数
   - dynamic_weights_from_stats(...)      ：把各腿 RankIC 转成归一化融合权重
   - load_leg_rankic(...)                 ：读 json + 过期判定

设计原则（与全局铁律一致）
--------------------------
- 默认不启用：ensemle_dynamic_weights=False 时 EnsemblePredictor 完全按静态权重融合，
  本模块不被调用，生产行为零改变。
- 启用后若 leg_rankic.json 缺失 / 过期 / 某腿无统计 → 自动回退静态权重，绝不拖垮主模型。
- 主模型（xgboost）恒为锚（权重 1.0），动态权重只重排**子腿**之间的相对贡献。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
LEG_RANKIC_FILE = DATA_DIR / "leg_rankic.json"


# ---------------------------------------------------------------------------
# 纯函数层（可离线单测）
# ---------------------------------------------------------------------------
def _spearman(x, y) -> float:
    """横截面 Spearman 秩相关；退化（样本<3 / 常量）返回 nan。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def aggregate_leg_rankic(
    per_day: Dict[str, List[Tuple[List[float], List[float]]]]
) -> Dict[str, Dict]:
    """把逐日 (scores, fwd_excess) 列表聚合成每腿统计。

    per_day[leg] = [ (scores_day1, fwd_day1), (scores_day2, fwd_day2), ... ]
    每对是某评分日该腿在所有有效股票上的分数与当日超额收益。

    Returns：{leg: {"rankic": mean_daily_ic, "ic_ir": mean/std, "n_days": int}}
    某腿没有任何有效日 IC 时不出现在结果里。
    """
    out: Dict[str, Dict] = {}
    for leg, days in per_day.items():
        ics: List[float] = []
        for scores, fwd in days:
            ic = _spearman(scores, fwd)
            if ic == ic and len(scores) >= 3:  # 有限值 + 足够样本
                ics.append(ic)
        if not ics:
            continue
        m = float(np.mean(ics))
        sd = float(np.std(ics, ddof=1)) if len(ics) > 1 else float("nan")
        ir = m / sd if (sd and sd > 0) else float("nan")
        out[leg] = {
            "rankic": round(m, 4),
            "ic_ir": round(ir, 3) if ir == ir else None,
            "n_days": len(ics),
        }
    return out


def dynamic_weights_from_stats(
    stats: Dict[str, Dict],
    leg_names: List[str],
    fallback_weights: List[float],
    floor: float = 0.0,
) -> List[float]:
    """把各腿 RankIC 转成归一化融合权重（与 leg_names 顺序对齐）。

    - 腿有统计：raw = max(rankic, floor)
    - 腿无统计：raw = 其静态 fallback 权重（与静态模式平滑衔接）
    - 归一化 raw → 权重；若全 ≤0（极端，所有腿负 IC 且无 fallback）退化为等权。
    """
    if len(leg_names) != len(fallback_weights):
        raise ValueError("leg_names 与 fallback_weights 长度必须一致")
    raw = []
    for name, fb in zip(leg_names, fallback_weights):
        s = stats.get(name)
        if isinstance(s, dict) and s.get("rankic") is not None:
            raw.append(max(float(s["rankic"]), float(floor)))
        else:
            raw.append(float(fb))
    total = float(sum(raw))
    if total <= 0:
        return [1.0 / len(raw)] * len(raw)
    return [r / total for r in raw]


def load_leg_rankic(
    path: Optional[Path] = None,
    max_age_days: int = 30,
) -> Optional[Dict[str, Dict]]:
    """读 data/leg_rankic.json 的 legs 段；缺失 / 格式错 / 过期 → None（触发回退）。"""
    p = Path(path) if path else LEG_RANKIC_FILE
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or "legs" not in data:
        return None
    legs = data.get("legs", {})
    if not isinstance(legs, dict) or not legs:
        return None
    last = data.get("last_date")
    if last:
        try:
            d = datetime.strptime(str(last), "%Y-%m-%d")
            if (datetime.now() - d).days > int(max_age_days):
                return None  # 过期：让调用方回退静态权重
        except Exception:
            pass
    return legs


# ---------------------------------------------------------------------------
# 纯函数层：增量回填 / 每日 IC 聚合（可离线单测）
# ---------------------------------------------------------------------------
def _aggregate_day(
    by_code: Dict[str, Dict[str, float]],
    valid: List[str],
    excess: "np.ndarray",
) -> Dict[str, Tuple[List[float], List[float]]]:
    """把「当日 {code:{leg:score}}」聚合成 {leg:(scores_all, fwd_all)}（横截面，供 _spearman）。

    修复：原实现把每只股票单独 append 成 ( [单点], [单点] )，样本<3 → IC 恒为 nan。
    这里按腿把当日全部有效股票拼成一个横截面数组，才能算横截面秩相关。
    """
    day: Dict[str, Tuple[List[float], List[float]]] = {}
    ex_map = {c: float(excess[i]) for i, c in enumerate(valid)}
    for c in valid:
        ex = ex_map[c]
        for leg, s in by_code[c].items():
            bucket = day.setdefault(leg, ([], []))
            bucket[0].append(float(s))
            bucket[1].append(ex)
    return day


def aggregate_from_daily(daily_ic: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Dict]:
    """由「已算好的每日 IC 记录」聚合出每腿统计（增量回填用）。

    daily_ic[leg] = [(date_str, ic), ...]。等价于 aggregate_leg_rankic，但输入是
    已聚合的每日 IC（而非原始 scores/fwd），避免落盘/重算时保存庞大的逐票数组。
    """
    out: Dict[str, Dict] = {}
    for leg, recs in daily_ic.items():
        ics = [ic for _, ic in recs if ic == ic]
        if not ics:
            continue
        m = float(np.mean(ics))
        sd = float(np.std(ics, ddof=1)) if len(ics) > 1 else float("nan")
        ir = m / sd if (sd and sd > 0) else float("nan")
        out[leg] = {
            "rankic": round(m, 4),
            "ic_ir": round(ir, 3) if ir == ir else None,
            "n_days": len(ics),
        }
    return out


def merge_daily_records(
    existing: Dict[str, List[Dict]],
    new: Dict[str, List[Dict]],
    max_days: int,
) -> Dict[str, List[Dict]]:
    """合并新旧每日 IC 记录（同 date 以 new 为准），按 date 升序裁剪到最近 max_days。

    existing / new 形如 {leg: [{date, ic}, ...]}（即可落盘格式）。返回同格式。
    """
    merged: Dict[str, List[Dict]] = {}
    for leg in set(existing) | set(new):
        recs: Dict[str, float] = {r["date"]: float(r["ic"]) for r in existing.get(leg, [])}
        for r in new.get(leg, []):
            recs[r["date"]] = float(r["ic"])  # new 覆盖旧
        ordered = sorted(recs.items())  # (date, ic)
        if max_days and len(ordered) > max_days:
            ordered = ordered[-max_days:]
        merged[leg] = [{"date": d, "ic": ic} for d, ic in ordered]
    return merged


# ---------------------------------------------------------------------------
# 回填计算（需联网，生产跑于耐久上下文）
# ---------------------------------------------------------------------------
def compute_leg_rankic(
    days: int = 30,
    sample: int = 150,
    horizon: int = 5,
    min_stocks: int = 30,
    pool: str = "zz500",
    legs: Optional[List[str]] = None,
    out: Optional[Path] = None,
    dry_run: bool = False,
    progress_cb=None,
    incremental: bool = True,
) -> Dict:
    """逐日算每条腿的 RankIC，写 leg_rankic.json（支持增量回填）。

    Returns：{status, mode, legs:{...}, daily:{...}, last_date, generated_at}
      - mode="full"      ：全量重算（覆盖写）
      - mode="incremental"：复用已有 leg_rankic.json 的每日 IC，只算缺失的交易日并合并
    需联网；会强制临时开启集成（确保 ensemble_detail 含每腿分数），结束后还原。
    """
    from config.settings import config as app_config
    from src.data.fetcher import fetcher
    from src.models.predictor import predictor
    from src.core.rolling_ic import _next_trading_day

    out = Path(out) if out else LEG_RANKIC_FILE
    old_use = app_config.ml.use_ensemble
    app_config.ml.use_ensemble = True  # 必须，否则不融合、无 ensemble_detail
    try:
        codes = list(fetcher.get_stock_pool(pool))
        if sample and sample < len(codes):
            codes = codes[:sample]

        raw: Dict[str, pd.DataFrame] = {}
        for i, code in enumerate(codes, 1):
            try:
                df = fetcher.get_daily_kline(code)
                if df is None or len(df) < 120:
                    continue
                df = df.copy()
                df.index = pd.to_datetime(df["date"])
                df = df.sort_index()
                raw[code] = df
            except Exception:  # noqa: BLE001
                pass
            if progress_cb and i % 50 == 0:
                progress_cb(i, len(codes), 0)

        if not raw:
            return {"status": "error", "error": "no_usable_klines"}

        close = pd.DataFrame({c: df["close"] for c, df in raw.items()}).sort_index()
        all_dates = list(close.index)
        # 候选日 = 最近 days 个「有 horizon 日后的前向收益」的交易日
        candidate_pool = []
        i = len(all_dates) - 1
        while i >= 0 and len(candidate_pool) < days:
            d = all_dates[i]
            if _next_trading_day(d, horizon) in close.index:
                candidate_pool.append(d)
            i -= 1
        candidate_pool = list(reversed(candidate_pool))  # 升序

        if dry_run:
            return {"status": "dry_run", "candidates": len(candidate_pool)}

        # 已有记录的日期集合（增量跳过）
        existing_daily: Dict[str, List[Dict]] = {}
        if incremental and out.exists():
            try:
                prev = json.loads(out.read_text(encoding="utf-8"))
                existing_daily = prev.get("daily", {}) or {}
            except Exception:  # noqa: BLE001
                existing_daily = {}
        existing_dates = set()
        for recs in existing_daily.values():
            for r in recs:
                existing_dates.add(r["date"])

        if incremental and existing_daily:
            to_compute = [d for d in candidate_pool
                          if d.strftime("%Y-%m-%d") not in existing_dates]
            mode = "incremental"
        else:
            to_compute = candidate_pool
            mode = "full"

        # 新算出的每日 IC：leg -> [(date, ic), ...]
        new_daily_ic: Dict[str, List[Tuple[str, float]]] = {}
        for k, d in enumerate(to_compute, 1):
            d_fwd = _next_trading_day(d, horizon)
            sliced = {c: df.loc[df.index <= d] for c, df in raw.items() if df.index[0] <= d}
            try:
                preds = predictor.predict_batch(sliced)
            except Exception:  # noqa: BLE001
                preds = {}
            # 修复：predict_batch 返回 {code: StockPrediction}，要 .items() 迭代
            by_code: Dict[str, Dict[str, float]] = {}
            if isinstance(preds, dict):
                for code, p in preds.items():
                    det = getattr(p, "ensemble_detail", None)
                    if isinstance(det, dict) and "scores" in det:
                        by_code[code] = det["scores"]
            c0 = close.loc[d]
            c1 = close.loc[d_fwd]
            valid = [c for c in raw if c in by_code and c in c0 and c in c1
                     and pd.notna(c0[c]) and pd.notna(c1[c]) and c0[c] > 0]
            if len(valid) < min_stocks:
                continue
            rets = np.array([c1[c] / c0[c] - 1.0 for c in valid], dtype=float)
            excess = rets - rets.mean()
            # 按腿聚合当日横截面（修复：整段横截面喂给 _spearman）
            day = _aggregate_day(by_code, valid, excess)
            date_str = d.strftime("%Y-%m-%d")
            for leg, (scs, fwd) in day.items():
                ic = _spearman(scs, fwd)
                if ic == ic and len(scs) >= 3:
                    new_daily_ic.setdefault(leg, []).append((date_str, ic))
            if progress_cb:
                progress_cb(len(to_compute), len(to_compute), k)

        if not new_daily_ic and mode == "full":
            return {"status": "error", "error": "no_valid_scores"}

        # 合并每日 IC 记录
        if mode == "incremental" and existing_daily:
            new_as_records = {
                leg: [{"date": d, "ic": ic} for d, ic in recs]
                for leg, recs in new_daily_ic.items()
            }
            merged = merge_daily_records(existing_daily, new_as_records, days)
        else:
            merged = {
                leg: [{"date": d, "ic": ic} for d, ic in recs]
                for leg, recs in new_daily_ic.items()
            }

        daily_ic = {leg: [(r["date"], r["ic"]) for r in recs] for leg, recs in merged.items()}
        legs_stats = aggregate_from_daily(daily_ic)
        if legs:
            for want in legs:
                legs_stats.setdefault(want, {"rankic": None, "ic_ir": None, "n_days": 0})

        last_date = None
        for recs in merged.values():
            for r in recs:
                if last_date is None or r["date"] > last_date:
                    last_date = r["date"]

        result = {
            "status": "ok",
            "mode": mode,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "last_date": last_date,
            "params": {"days": days, "sample": sample, "horizon": horizon, "pool": pool},
            "legs": legs_stats,
            "daily": merged,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        app_config.ml.use_ensemble = old_use

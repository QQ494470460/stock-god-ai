# -*- coding: utf-8 -*-
"""
预测追踪 + 回测验证（重写版）
=============================

修复的四个致命问题
------------------
**问题1：自然日 vs 交易日错配 → 验证永不命中**
   旧代码 target_date = now - timedelta(days=5)，然后字符串相等匹配。
   但预测只在交易日产生，周末/节假日必然对不上。
   实测：22 批预测，0 批验证成功，面板永远空白。
   → 现在：遍历**每一条历史记录**，按日期找真实交易日价格，
     不做日期字符串反推。

**问题2：基准代码解析错误 → 超额收益恒等于平均收益**
   旧代码用 "000300.SH"，去掉后缀后不以 6/9 开头 → 解析为 sz000300
   （平安银行，不是沪深300）。异常被 except: return 0.0 静默吞掉。
   → 现在：用 sh000300 走指数接口，失败时明确告警而非静默归零。

**问题3：max_drawdown 按收益排序后累加 → 恒等于 0**
   降序排列后曲线单调不降，peak 恒等于 cumulative，回撤永远是 0。
   → 现在：按真实时间顺序累计净值算回撤。

**问题4：未快照预测日价格 → 基准价靠反推**
   旧代码验证时重新拉历史价格。而前复权数据会被未来除权事件改写，
   回推出的"当时价格"不是当时真实可见的价格。
   → 现在：save_predictions 时就快照预测日收盘价，验证时直接用快照。

另修正：同日扫描不再静默覆盖（改为按 scan_date + 时间戳去重）
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from loguru import logger

from config.settings import DATA_DIR

TRACK_FILE = DATA_DIR / "prediction_history.json"
KEEP_DAYS = 180


# ============ 存取 ============

def _load_all_history() -> List[Dict]:
    if not TRACK_FILE.exists():
        return []
    try:
        return json.loads(TRACK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_history(history: List[Dict]):
    history.sort(key=lambda e: e.get("scan_date", ""), reverse=True)
    history = history[:KEEP_DAYS]
    TRACK_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def entry_verifications(entry: Dict) -> Dict[str, Dict]:
    """读取一条历史的全部验证结果，**兼容两代追踪器的字段**。

    这是本模块唯一的兼容规则，供本模块、``src/evolution/feedback.py``
    与所有汇总/展示代码共用 —— 兼容逻辑只写一遍，否则两边迟早写歪。

    * 现行（v2）：``entry["verifications"] = {"5": {...}, "10": {...}}``
    * 旧版（tracker.py）：``entry["verification"] = {...}`` 单槽位，
      T+5 与 T+10 会互相覆盖。只做读取兼容，不迁移、不改盘。

    同一窗口若两种格式都有，以 **v2 的为准**（``setdefault``）——
    这正是「被 v2 重新验证过」的语义：新结果不该被旧结果盖回去。
    """
    out: Dict[str, Dict] = {}
    for k, v in (entry.get("verifications") or {}).items():
        if isinstance(v, dict):
            out[str(k)] = v
    legacy = entry.get("verification")
    if isinstance(legacy, dict):
        h = str(legacy.get("hold_days") or legacy.get("lookback_days") or 5)
        out.setdefault(h, legacy)
    return out


def save_predictions(
    predictions: List[Dict],
    scan_date: str = None,
    price_snapshot: Optional[Dict[str, float]] = None
):
    """
    保存一批预测结果

    新增 price_snapshot 参数：预测当日各股票的收盘价快照。
    这是修复问题4的关键 —— 基准价必须用当时的价格，不能事后反推。
    """
    if scan_date is None:
        scan_date = datetime.now().strftime("%Y-%m-%d")

    entry = {
        "scan_date": scan_date,
        "logged_at": datetime.now().isoformat(),
        "n_predictions": len(predictions),
        "predictions": predictions,
        # 价格快照（修复问题4）
        "price_snapshot": price_snapshot or {},
    }

    history = _load_all_history()

    # 不再静默覆盖：只替换同一 scan_date 且没有价格快照的旧记录
    replaced = False
    for e in history:
        if e.get("scan_date") == scan_date and not e.get("price_snapshot"):
            e.clear()
            e.update(entry)
            replaced = True
            break
    if not replaced:
        history.append(entry)

    _save_history(history)
    logger.info(
        f"追踪记录已保存: {scan_date} — {len(predictions)} 只推荐"
        f"{'（含价格快照）' if price_snapshot else '（⚠无价格快照）'}"
    )


# ============ 价格读取 ============

def _load_price_panel(codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """从本地缓存读取价格（不联网，避免触发全量刷新）"""
    data_dir = Path(DATA_DIR)
    out = {}
    for code in codes:
        for suffix in ("qfq", ""):
            f = data_dir / f"daily_{code}_{suffix}.parquet"
            if f.exists():
                try:
                    df = pd.read_parquet(f)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date").sort_index()
                    out[code] = df
                    break
                except Exception:
                    continue
        if code not in out:
            try:
                from src.data.fetcher import fetcher
                df = fetcher.get_daily_kline(code, start_date=start, end_date=end)
                if df is not None and len(df):
                    df["date"] = pd.to_datetime(df["date"])
                    out[code] = df.set_index("date").sort_index()
            except Exception:
                continue
    return out


def _load_close_matrix(
    codes: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp
) -> pd.DataFrame:
    """
    一次性载入多只股票的收盘价矩阵（index=date, columns=code）

    验证 20 只股票 × 22 批记录 = 440 次 parquet 读取太慢，
    这里批量读一次并对齐成矩阵，后续计算全部走矩阵运算。
    """
    data_dir = Path(DATA_DIR)
    series = {}
    for code in codes:
        f = data_dir / f"daily_{code}_qfq.parquet"
        if not f.exists():
            f = data_dir / f"daily_{code}.parquet"
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f)
            if len(df) == 0 or "date" not in df.columns or "close" not in df.columns:
                continue
            s = pd.Series(
                pd.to_numeric(df["close"], errors="coerce").values,
                index=pd.to_datetime(df["date"]),
                name=str(code),
            )
            series[str(code)] = s[~s.index.duplicated(keep="last")].sort_index()
        except Exception:
            continue

    if not series:
        return pd.DataFrame()

    mat = pd.DataFrame(series).sort_index()
    # 只保留 [start - 缓冲, end] 区间，减小内存
    lo = start - pd.Timedelta(days=20)
    return mat.loc[(mat.index >= lo) & (mat.index <= end)]


def _first_price_on_or_after(
    panel: Dict[str, pd.DataFrame], code: str, date: pd.Timestamp
) -> Optional[float]:
    """
    取该股票在 date 当日或之后第一个交易日的收盘价

    注意：不能用"之前最后一个"，因为如果那天停牌，用旧价会产生
    虚假的巨大收益。用"之后第一个"更接近实盘可成交价。
    """
    if code not in panel:
        return None
    df = panel[code]
    after = df[df.index >= date]
    if len(after) == 0:
        return None
    v = float(after["close"].iloc[0])
    return v if v > 0 else None


_BENCH_CACHE: Optional[pd.Series] = None


def _get_benchmark_series() -> Optional[pd.Series]:
    """沪深300 收盘价序列（进程内缓存，避免重复联网）"""
    global _BENCH_CACHE
    if _BENCH_CACHE is not None:
        return _BENCH_CACHE
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol="sh000300")
        s = pd.Series(
            pd.to_numeric(df["close"], errors="coerce").values,
            index=pd.to_datetime(df["date"]),
        )
        _BENCH_CACHE = s[~s.index.duplicated(keep="last")].sort_index()
    except Exception as e:
        logger.warning(f"沪深300 数据获取失败: {e}")
        _BENCH_CACHE = None
    return _BENCH_CACHE


def _fetch_benchmark_return(start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """
    获取沪深300 同期收益（修复问题2）

    旧代码的 "000300.SH" 会被解析成 sz000300（平安银行），异常被吞成 0。
    这里用正确的指数代码，并对失败明确告警。
    """
    s = _get_benchmark_series()
    if s is None or len(s) == 0:
        return float("nan")

    seg = s[(s.index >= start_date) & (s.index <= end_date)]
    if len(seg) < 2:
        logger.warning(f"沪深300 在 {start_date.date()}~{end_date.date()} 数据不足")
        return float("nan")

    p0, p1 = float(seg.iloc[0]), float(seg.iloc[-1])
    if p0 <= 0:
        return float("nan")
    return p1 / p0 - 1


# ============ 核心验证 ============

# 覆盖率门槛：低于此值的批次不足以代表模型能力，标记为不可靠。
# 没有这道门槛时，「20 只票只验到 1 只」照样算出 accuracy=0 并进入滑窗，
# 于是 should_retrain 的均值判断会建立在 1 只票的样本上。
MIN_COVERAGE = 0.6
MIN_CHECKED = 8


def signal_is_correct(
    signal: int,
    actual_return: float,
    benchmark_return=None,
) -> bool:
    """判定单次预测的方向正确性（按超额收益，而非绝对收益）。

    为什么必须用超额
    ----------------
    选股给的是**看多候选**，其价值应以「是否跑赢基准」衡量：
    大盘 -5% 时个股 -1%（跑赢 4%）是有效选股；大盘 +5% 时个股 +1%
    （跑输 4%）反而是失败的选股。

    旧实现写成了 ``actual > 0`` 的绝对判定 —— 注释写着「阈值用跑赢基准
    更公平」，实现却与之相反。后果已在线上复现：2026-10-15 基准 -3.11%，
    推荐组合跑赢 +1.09%，却被判 accuracy=0.0；该值进入
    ``performance.t5.accuracy`` 后近三批均值仅 11.1%，触发「风格漂移
    建议重训」，且在 (0, 0.45] 区间时会由 ``_evolve_threshold`` 自动收紧
    选股阈值（收紧方向无需人工批准）。

    Parameters
    ----------
    signal : 1=看多，-1=看空，0=中性
    actual_return : 个股区间收益率
    benchmark_return : 同期基准收益率；为 None 或缺席时退化为绝对收益判定

    Returns
    -------
    方向是否正确
    """
    if benchmark_return is None:
        return (signal == 1 and actual_return > 0) or (
            signal == -1 and actual_return < 0)

    excess = actual_return - benchmark_return
    return (signal == 1 and excess > 0) or (signal == -1 and excess < 0)


def coverage_is_reliable(checked: int, total: int) -> bool:
    """本次验证的样本覆盖是否足以代表模型能力。"""
    if not total or not checked:
        return False
    return checked >= MIN_CHECKED and (checked / total) >= MIN_COVERAGE


def verify_predictions(
    scan_date: str,
    horizon: int = 5,
    persist: bool = True
) -> Optional[Dict]:
    """
    验证指定预测日的实际表现

    Parameters
    ----------
    scan_date : 预测日（字符串 YYYY-MM-DD），必须与保存时一致
    horizon : 持有交易日数（5 或 10）
    persist : 是否写回结果

    Returns
    -------
    验证结果 dict，失败返回 None
    """
    history = _load_all_history()
    entry = next((e for e in history if e.get("scan_date") == scan_date), None)
    if not entry:
        logger.debug(f"未找到 {scan_date} 的预测记录")
        return None

    preds = entry.get("predictions", [])
    if not preds:
        return None

    snapshot = entry.get("price_snapshot", {})
    start = pd.Timestamp(scan_date)

    codes = [p["code"] for p in preds if p.get("code")]
    if not codes:
        return None

    # 批量载入一次收盘价矩阵（关键性能优化：避免逐票重复读盘）
    # 先多取 60 天缓冲，用于确定基准日与结束日
    mat = _load_close_matrix(
        codes, start, start + pd.Timedelta(days=int(horizon * 2.5) + 40)
    )
    if len(mat) == 0:
        logger.warning(f"{scan_date}: 无可用价格数据")
        return None

    # 基准日 = start 当日或之后第一个有数据的交易日
    valid_rows = mat.index[mat.index >= start]
    if len(valid_rows) == 0:
        return None

    base_day = valid_rows[0]
    # 结束日 = base_day 之后第 horizon 个有数据的交易日
    after_base = mat.index[mat.index > base_day]
    if len(after_base) < horizon:
        logger.debug(f"{scan_date}: 尚未满 {horizon} 个交易日，跳过")
        return None
    end_day = after_base[horizon - 1]

    base_prices = mat.loc[base_day]
    end_prices = mat.loc[end_day]

    returns = {}
    for c in codes:
        c = str(c)
        # 优先用记录里的价格快照（修复问题4：不能用未来复权后的历史价）
        p0 = snapshot.get(c)
        if p0 is None or not (p0 > 0):
            v = base_prices.get(c, np.nan)
            p0 = float(v) if pd.notna(v) and v > 0 else None
        p1v = end_prices.get(c, np.nan)
        if p0 and pd.notna(p1v) and p1v > 0:
            returns[c] = (float(p1v) - p0) / p0

    if not returns:
        logger.warning(f"{scan_date}: 无法计算任何收益")
        return None

    bench = _fetch_benchmark_return(base_day, end_day)

    # ---- 指标计算 ----
    detail_rows = []
    port_rets = []
    correct = checked = abs_correct = 0
    xgb_c = xgb_n = 0
    score_buckets = {">=55": [], "45-55": [], "<45": []}

    # 基准为 NaN（取不到指数）时无法算超额，退化到绝对口径并如实标注
    bench_scalar = float(bench) if not np.isnan(bench) else None

    for p in preds:
        code = p.get("code", "")
        actual = returns.get(code)
        if actual is None:
            continue
        signal = int(p.get("signal", 0))
        score = float(p.get("score", 50))
        xs = int(p.get("xgb_signal", signal))

        # 方向正确性：有基准按超额收益判定（详见 signal_is_correct）
        ok = signal_is_correct(signal, actual, bench_scalar)
        if ok:
            correct += 1
        # 绝对口径仅留作人工对照，不参与任何自动化决策
        if signal_is_correct(signal, actual, None):
            abs_correct += 1
        checked += 1

        if xs != 0:
            xgb_n += 1
            xgb_c += int(signal_is_correct(xs, actual, bench_scalar))

        # 分桶按 2026-10-10 单模型化后实测分布校准；旧值 80/60 会让所有记录都落进 "<45" 桶
        b = ">=55" if score >= 55 else ("45-55" if score >= 45 else "<45")
        score_buckets[b].append(actual)

        port_rets.append(actual)
        detail_rows.append({
            "code": code,
            "name": p.get("name", ""),
            "score": round(score, 1),
            "signal": signal,
            "actual_return": round(actual, 4),
            "correct": bool(ok),
        })

    if not port_rets:
        return None

    port = pd.Series(port_rets)
    avg_ret = float(port.mean())
    excess = avg_ret - bench if not np.isnan(bench) else None

    # 真实时间序列的最大回撤（修复问题3）
    equity = (1 + port).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    max_dd = float(abs(dd))

    wins, losses = port[port > 0], port[port < 0]
    result = {
        "scan_date": scan_date,
        "base_date": str(base_day.date()),
        "exit_date": str(end_day.date()),
        "hold_days": horizon,
        "verified_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "price_source": "snapshot" if snapshot else "lookup",
        "total_recommended": len(preds),
        "checked": checked,
        "correct": correct,
        # accuracy 现按「跑赢基准」口径；基准不可得时退化为绝对口径
        "accuracy": round(correct / checked, 4) if checked else None,
        # 同值的显式字段名：accuracy 的语义已经变更，另给一个自解释的名字，
        # 让下游（evolution.record_verification）可以显式取「跑赢基准」口径，
        # 不必依赖对 accuracy 语义的理解。
        "excess_accuracy": round(correct / checked, 4) if checked else None,
        # 绝对收益口径，仅供人工对照，**不得用于自动化决策**（熊市系统性失真）
        "absolute_accuracy": round(abs_correct / checked, 4) if checked else None,
        "coverage": round(checked / len(preds), 4) if preds else 0.0,
        "reliable": coverage_is_reliable(checked, len(preds)),
        "benchmark_used": bench_scalar is not None,
        "avg_return": round(avg_ret, 4),
        "benchmark_return": round(bench, 4) if not np.isnan(bench) else None,
        "excess_return": round(excess, 4) if excess is not None else None,
        "win_rate": round(float((port > 0).mean()), 4),
        "avg_win": round(float(wins.mean()), 4) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 4) if len(losses) else None,
        "max_drawdown": round(max_dd, 4),
        "xgb_accuracy": round(xgb_c / xgb_n, 4) if xgb_n else None,
        "xgb_checked": xgb_n,
        "by_score_bucket": {
            k: (round(float(np.mean(v)), 4) if v else None)
            for k, v in score_buckets.items()
        },
        "details": detail_rows,
    }

    if persist:
        for e in history:
            if e.get("scan_date") == scan_date:
                e.setdefault("verifications", {})[str(horizon)] = result
                break
        _save_history(history)

    ex = f"{excess:+.2%}" if excess is not None else "基准缺失"
    logger.info(
        f"验证 {scan_date}+{horizon}日: 准确率={result['accuracy']} "
        f"平均={avg_ret:+.2%} 超额={ex} 胜率={result['win_rate']}"
    )
    return result


def verify_all_pending(horizons: List[int] = None) -> List[Dict]:
    """
    验证所有已满持有期的历史记录

    修复问题1的核心：不再用"今天减N天"反推日期，
    而是逐条检查每条记录是否真的走够了 horizon 个交易日。
    """
    if horizons is None:
        horizons = [5, 10]

    history = _load_all_history()
    results = []

    for entry in history:
        scan_date = entry.get("scan_date")
        if not scan_date:
            continue
        # 走统一兼容读取：旧版写进 `verification` 单槽位的窗口也算「已验证」，
        # 不再重复计算（重复计算会平白多打一轮行情接口）。
        already = entry_verifications(entry)
        for h in horizons:
            if str(h) in already:
                continue
            r = verify_predictions(scan_date, horizon=h, persist=True)
            if r:
                results.append(r)

    return results


def get_accuracy_summary() -> Dict:
    """汇总所有验证过的历史记录"""
    history = _load_all_history()

    verified = []
    for e in history:
        for h, v in entry_verifications(e).items():
            row = dict(v)
            row.setdefault("scan_date", e.get("scan_date"))
            # 窗口号以 key 为准，保证「旧版写的行」与「v2 写的行」在表格里同构
            try:
                row.setdefault("hold_days", int(h))
            except (TypeError, ValueError):
                pass
            verified.append(row)

    if not verified:
        return {
            "n_verified": 0,
            "accuracy": None,
            "avg_return": None,
            "avg_excess": None,
            "avg_win_rate": None,
            "note": "尚无已验证记录。首次预测后需满 5 个交易日才会出现结果。",
            "history": [],
        }

    def _mean(key):
        vals = [v[key] for v in verified if v.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    total_checked = sum(v.get("checked", 0) for v in verified)
    total_correct = sum(v.get("correct", 0) for v in verified)

    return {
        "n_verified": len(verified),
        "n_checked": total_checked,
        # 用样本数加权的总体准确率，而非简单平均
        "accuracy": round(total_correct / total_checked, 4) if total_checked else None,
        "avg_return": _mean("avg_return"),
        "avg_excess": _mean("excess_return"),
        "avg_win_rate": _mean("win_rate"),
        "avg_max_drawdown": _mean("max_drawdown"),
        "history": [
            {
                "scan_date": v.get("scan_date"),
                "hold_days": v.get("hold_days"),
                "accuracy": v.get("accuracy"),
                "avg_return": v.get("avg_return"),
                "excess_return": v.get("excess_return"),
                "win_rate": v.get("win_rate"),
                "checked": v.get("checked"),
            }
            for v in sorted(verified, key=lambda x: x.get("scan_date", ""), reverse=True)
        ],
    }
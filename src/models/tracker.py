# -*- coding: utf-8 -*-
"""
预测追踪 + 回测验证模块

每次 AI 选股扫描时记录推荐快照 → N 日后批量回测验证：
- 准确率（信号方向是否正确）
- 组合收益（等权持有 N 日的实际涨跌）
- 超额收益（相对沪深 300）
- 胜率、盈亏比、最大回撤

数据存储: data/prediction_history.json
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from loguru import logger

from config.settings import DATA_DIR

TRACK_FILE = DATA_DIR / "prediction_history.json"


def save_predictions(predictions: List[Dict], scan_date: str = None):
    """保存一批预测结果至追踪文件（每日一次覆盖）"""
    if scan_date is None:
        scan_date = datetime.now().strftime("%Y-%m-%d")

    entry = {
        "scan_date": scan_date,
        "n_predictions": len(predictions),
        "predictions": predictions,
        "logged_at": datetime.now().isoformat(),
    }

    existing = []
    if TRACK_FILE.exists():
        try:
            existing = json.loads(TRACK_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    existing = [e for e in existing if e.get("scan_date") != scan_date]
    existing.append(entry)
    existing.sort(key=lambda e: e.get("scan_date", ""), reverse=True)
    existing = existing[:90]  # 保留最近 90 天

    TRACK_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"📝 追踪记录已保存: {scan_date} — {len(predictions)} 只推荐")


def _load_all_history() -> List[Dict]:
    if not TRACK_FILE.exists():
        return []
    try:
        return json.loads(TRACK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _fetch_returns_batch(codes: List[str], lookback_days: int) -> Dict[str, float]:
    """
    批量获取一组股票的"预测日后 N 日实际收益"

    返回: {code: actual_return}
    """
    from src.data.fetcher import fetcher

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y%m%d")

    returns = {}
    # 复用 fetcher 缓存的最近数据，但这里需要历史日期，回退到直接拉取
    for code in codes:
        try:
            df = fetcher.get_daily_kline(code, start_date=start, end_date=end)
            if df is None or len(df) < lookback_days + 2:
                continue
            # 第 -lookback_days-1 行 = 预测日的收盘价
            idx = len(df) - lookback_days - 1
            if idx < 0:
                continue
            pred_close = float(df["close"].iloc[idx])
            now_close = float(df["close"].iloc[-1])
            if pred_close <= 0:
                continue
            returns[code] = (now_close - pred_close) / pred_close
        except Exception as e:
            logger.debug(f"获取 {code} 历史收益失败: {e}")
            continue
    return returns


def _fetch_benchmark_return(lookback_days: int) -> float:
    """获取沪深 300 同期的实际收益（基准）

    注意：指数必须走 ``get_benchmark_kline``。原先用的 ``get_daily_kline``
    会走个股日线接口，指数取不到数据（静默返回空）→ 基准收益恒为 0，
    导致回测面板里的「超额收益」永远是 0，失去参考意义。
    """
    from src.data.fetcher import fetcher

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y%m%d")

    try:
        df = fetcher.get_benchmark_kline("000300.SH", start_date=start)
        if df is None or len(df) < lookback_days + 2:
            return 0.0
        idx = len(df) - lookback_days - 1
        if idx < 0:
            return 0.0
        pred = float(df["close"].iloc[idx])
        now = float(df["close"].iloc[-1])
        return (now - pred) / pred if pred > 0 else 0.0
    except Exception as e:
        logger.debug(f"获取基准收益失败: {e}")
        return 0.0


def verify_predictions(lookback_days: int = 5) -> Optional[Dict]:
    """
    验证 N 天前的预测是否准确（一次性算完，不重复拉数据）
    """
    history = _load_all_history()
    if not history:
        return None

    target_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    target_entry = None
    for e in history:
        if e.get("scan_date") == target_date:
            target_entry = e
            break

    if not target_entry:
        logger.debug(f"未找到 {target_date} 的预测记录")
        return None

    preds = target_entry.get("predictions", [])
    if not preds:
        return None

    # 一次性批量拉取所有股票收益
    codes = [p.get("code", "") for p in preds if p.get("code")]
    returns_map = _fetch_returns_batch(codes, lookback_days)
    bench_return = _fetch_benchmark_return(lookback_days)

    correct = 0
    checked = 0
    details = []
    portfolio_returns = []

    # 分模型统计（自我进化核心反馈）
    xgb_correct = 0
    xgb_checked = 0

    for p in preds:
        code = p.get("code", "")
        signal = int(p.get("signal", 0))
        score = float(p.get("score", 50))
        xgb_signal = int(p.get("xgb_signal", signal))
        actual = returns_map.get(code)

        if actual is None:
            details.append({
                "code": code,
                "name": p.get("name", ""),
                "signal": signal,
                "score": score,
                "actual_return": None,
                "correct": None,
                "skip_reason": "数据缺失",
            })
            continue

        # 信号方向正确性
        is_correct = False
        if signal == 1 and actual > 0.01:
            is_correct = True
        elif signal == -1 and actual < -0.01:
            is_correct = True
        elif signal == 0 and abs(actual) <= 0.015:
            is_correct = True

        if is_correct:
            correct += 1
        checked += 1

        # 分模型方向正确性（用同一套判定阈值）
        def _sig_correct(sig):
            if sig == 1 and actual > 0.01:
                return True
            if sig == -1 and actual < -0.01:
                return True
            if sig == 0 and abs(actual) <= 0.015:
                return True
            return False

        if xgb_signal != 0:  # 只有明确信号才计入模型准确率
            xgb_checked += 1
            if _sig_correct(xgb_signal):
                xgb_correct += 1

        details.append({
            "code": code,
            "name": p.get("name", ""),
            "signal": signal,
            "score": score,
            "xgb_signal": xgb_signal,
            "actual_return": round(actual, 4),
            "correct": is_correct,
        })

        # 组合收益：等权持有
        portfolio_returns.append(actual)

    accuracy = correct / checked if checked > 0 else 0
    avg_return = sum(portfolio_returns) / len(portfolio_returns) if portfolio_returns else 0
    excess = avg_return - bench_return

    # 胜率（推荐列表中实际收益>0 的占比）
    winners = [r for r in portfolio_returns if r > 0]
    losers = [r for r in portfolio_returns if r < 0]
    win_rate = len(winners) / len(portfolio_returns) if portfolio_returns else 0

    # 盈亏比
    avg_win = sum(winners) / len(winners) if winners else 0
    avg_loss = abs(sum(losers) / len(losers)) if losers else 1e-9
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 最大回撤（按个股收益排序后累计）
    sorted_returns = sorted(portfolio_returns, reverse=True)
    cumulative = 0
    peak = 0
    max_dd = 0
    for r in sorted_returns:
        cumulative += r
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    result = {
        "scan_date": target_date,
        "verified_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lookback_days": lookback_days,
        "total_recommended": len(preds),
        "checked": checked,
        "correct": correct,
        "accuracy": round(accuracy, 3),
        "avg_return": round(avg_return, 4),
        "benchmark_return": round(bench_return, 4),
        "excess_return": round(excess, 4),
        "win_rate": round(win_rate, 3),
        "profit_loss_ratio": round(profit_loss_ratio, 2),
        "max_drawdown": round(max_dd, 4),
        "xgb_accuracy": round(xgb_correct / xgb_checked, 3) if xgb_checked > 0 else None,
        "xgb_checked": xgb_checked,
        "details": details,
    }

    # 把验证结果回写到对应记录里
    for e in history:
        if e.get("scan_date") == target_date:
            e["verification"] = result
            break
    TRACK_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    logger.info(
        f"📊 {target_date}+{lookback_days}天验证: "
        f"准确率={accuracy:.1%} 平均收益={avg_return:.2%} "
        f"超额={excess:.2%} 胜率={win_rate:.1%}"
    )
    return result


def verify_all_pending() -> List[Dict]:
    """
    一次性验证所有未验证的历史记录（5 日 / 10 日两个回测窗口）

    返回: 验证结果列表
    """
    history = _load_all_history()
    today = datetime.now().strftime("%Y-%m-%d")
    results = []

    for window in [5, 10]:
        # 找出该窗口内、尚未验证的日期
        for entry in history:
            scan_date = entry.get("scan_date")
            if not scan_date:
                continue
            if entry.get("verification", {}).get("lookback_days") == window:
                continue  # 已验证
            # 检查该日期距今是否够 window 天
            try:
                scan_dt = datetime.strptime(scan_date, "%Y-%m-%d")
                days_elapsed = (datetime.now() - scan_dt).days
                if days_elapsed < window:
                    continue
            except Exception:
                continue
            # 回测
            r = verify_predictions(lookback_days=window)
            if r:
                results.append(r)

    return results


def get_accuracy_summary() -> Dict:
    """
    汇总所有验证过的历史记录
    """
    history = _load_all_history()
    verified = [e for e in history if e.get("verification")]

    if not verified:
        return {
            "n_verified": 0,
            "avg_accuracy": 0,
            "avg_excess_return": 0,
            "avg_win_rate": 0,
            "history": [],
        }

    accs = [v["verification"]["accuracy"] for v in verified]
    excesses = [v["verification"]["excess_return"] for v in verified]
    win_rates = [v["verification"]["win_rate"] for v in verified]

    return {
        "n_verified": len(verified),
        "avg_accuracy": round(sum(accs) / len(accs), 3),
        "avg_excess_return": round(sum(excesses) / len(excesses), 4),
        "avg_win_rate": round(sum(win_rates) / len(win_rates), 3),
        "history": [
            {
                "scan_date": v["scan_date"],
                "lookback_days": v["verification"]["lookback_days"],
                "accuracy": v["verification"]["accuracy"],
                "avg_return": v["verification"]["avg_return"],
                "excess_return": v["verification"]["excess_return"],
                "win_rate": v["verification"]["win_rate"],
                "checked": v["verification"]["checked"],
            }
            for v in verified
        ],
    }
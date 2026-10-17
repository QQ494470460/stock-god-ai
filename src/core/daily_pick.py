# -*- coding: utf-8 -*-
"""
每日推荐生成（收盘后自动化核心）
================================

把看板「开始 AI 选股」的核心流程抽成可复用函数，供：
  · 调度器（APScheduler）收盘后自动生成每日推荐
  · CLI / 脚本手动触发
  · 单元测试注入 Fake 依赖

设计原则：
  · 与看板选股逻辑保持一致（同一 predictor / fetcher / evolution 单例）。
  · 默认用「完全自动」有效阈值（基线 + 进化调整 + 市场状态调整）。
  · 所有网络/数据异常都被捕获，绝不因单只失败而中断整批。
  · 产物同时写 JSON（机器可读）与 Markdown（人可读）到 report_dir。
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from loguru import logger

from config.settings import config as app_config
from src.data.fetcher import fetcher
from src.models.predictor import predictor
from src.models.evolution import evolution


def generate_daily_picks(
    pool: Optional[str] = None,
    max_picks: Optional[int] = None,
    min_score: Optional[float] = None,
    fetcher_inst=None,
    predictor_inst=None,
    evolution_inst=None,
    write_report: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    收盘后自动生成每日推荐。

    Returns
    -------
    (df, report) ：推荐结果 DataFrame 与结构化报告字典。
    df 可能为空（无达标标的），report['status'] 记录 'ok' / 'empty' / 'error'。
    """
    fet = fetcher_inst or fetcher
    pred = predictor_inst or predictor
    evo = evolution_inst or evolution

    pool = pool or app_config.scheduler.pick_pool
    max_picks = max_picks or app_config.scheduler.max_picks

    # 有效阈值：自动模式优先用进化引擎的完全自动阈值
    if min_score is None:
        try:
            if getattr(app_config.ml, "auto_threshold_adjust", False):
                min_score = evo.get_effective_min_score()
            else:
                min_score = 50.0
        except Exception:
            min_score = 50.0

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pool": pool,
        "max_picks": max_picks,
        "min_score": round(float(min_score), 1),
        "regime": None,
        "n_scanned": 0,
        "n_passed": 0,
        "picks": [],
        "status": "ok",
        "error": None,
    }

    try:
        codes = fet.get_stock_pool(pool)
        report["n_scanned"] = len(codes)

        # 拉取行情（会顺带刷新本地缓存）
        raw = fet.get_batch_daily_kline(codes, max_workers=8)
        stock_data = {
            c: df for c, df in raw.items()
            if df is not None and len(df) >= 60
        }

        # code → name 映射
        stock_names: Dict[str, str] = {}
        try:
            sl = fet.get_stock_list()
            name_map = dict(zip(sl["code"], sl["name"]))
            stock_names = {c: name_map.get(c, c) for c in stock_data}
        except Exception as e:
            logger.debug(f"股票列表获取失败，使用代码作名称: {e}")

        df = pred.get_top_recommendations(
            stock_data, stock_names, top_n=max_picks, min_score=min_score
        )

        # 参考价：把推荐时点的最新收盘价带进 picks，供交易计划 / 执行侧做
        # 「跳空太多就不追」这类校验。早期没有这一步，计划里 price 恒为 0.0，
        # 字段名存实亡。取不到就填 0，不影响推荐本身。
        try:
            if len(df):
                col = "代码" if "代码" in df.columns else ("code" if "code" in df.columns else None)
                if col is not None:
                    def _last_close(c):
                        d = stock_data.get(str(c)) or stock_data.get(str(c).zfill(6))
                        try:
                            return round(float(d["close"].iloc[-1]), 2)
                        except Exception:  # noqa: BLE001
                            return 0.0
                    df = df.copy()
                    df["最新价"] = df[col].map(_last_close)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"参考价补充失败（不影响推荐）: {e}")

        # 市场状态刷新（写入进化引擎，供阈值自适应）
        try:
            bd = fet.get_benchmark_kline("000300.SH", start_date="2024-01-01")
            report["regime"] = evo.detect_regime(bd)
        except Exception as e:
            logger.warning(f"regime 检测失败: {e}")

        report["n_passed"] = len(df)
        report["picks"] = df.to_dict(orient="records") if len(df) else []
        if len(df) == 0:
            report["status"] = "empty"
            logger.info(f"每日推荐为空（阈值 {min_score:.1f} 过高或行情缺失）")
        else:
            logger.info(f"✅ 每日推荐生成: {len(df)} 只（阈值 {min_score:.1f}）")

    except Exception as e:
        report["status"] = "error"
        report["error"] = str(e)
        logger.error(f"每日推荐生成失败: {e}\n{traceback.format_exc()}")
        df = pd.DataFrame()

    if write_report and report["status"] != "error":
        try:
            out_dir = Path(app_config.scheduler.report_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            (out_dir / f"daily_picks_{ts}.json").write_text(
                json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8"
            )
            (out_dir / f"daily_picks_{ts}.md").write_text(
                _to_markdown(report), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"报告写入失败: {e}")

    return df, report


def _to_markdown(report: Dict[str, Any]) -> str:
    """把结构化报告渲染成 Markdown"""
    lines = [
        f"# 📈 每日 AI 选股推荐",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 股票池：{report['pool']}",
        f"- 有效阈值：{report['min_score']:.1f}（完全自动：基线+进化+市场状态）",
        f"- 市场状态：{report.get('regime') or '未知'}",
        f"- 扫描 {report['n_scanned']} 只 → 达标 {report['n_passed']} 只",
        "",
    ]
    picks = report.get("picks") or []
    if not picks:
        lines.append("_（本次无达标标的）_")
        return "\n".join(lines)

    lines.append("| 排名 | 代码 | 名称 | 评分 | 上涨概率 | 信号 | 理由 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in picks:
        lines.append(
            f"| {row.get('排名', '')} | {row.get('代码', '')} | {row.get('名称', '')} | "
            f"{row.get('评分', '')} | {row.get('上涨概率', '')}% | {row.get('XGB信号', '')} | "
            f"{row.get('推荐理由', '')} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    df, rep = generate_daily_picks(write_report=True)
    print(f"status={rep['status']} n_passed={rep['n_passed']}")

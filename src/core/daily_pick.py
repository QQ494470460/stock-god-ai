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
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from loguru import logger

from config.settings import config as app_config, DATA_DIR
from src.data.fetcher import fetcher
from src.models.predictor import predictor
from src.models.evolution import evolution
from src.models.evolution import THRESHOLD_MIN as _SCORE_FLOOR
from src.core.signal_strength import (
    compute_signal_strength,
    format_signal_strength,
    record_signal_history,
    signal_strength_alert,
    signal_trend,
)


# 大盘状态 → 中文语境（与 evolution.detect_regime 返回值对齐）
_MARKET_REGIME_CTX = {
    "bull": "大盘牛市",
    "bear": "大盘熊市",
    "sideways": "大盘震荡",
}


def _annotate_reason(reason: str, market_regime: Optional[str]) -> str:
    """把「个股口吻」的推荐理由叠加「大盘口吻」，消除两份计划里
    ``regime: bear`` + ``reason: 牛市顺风`` 这种看着像矛盾、实则两个口径的困惑。

    scoring.compute_score 的 ``regime_note`` 只反映**个股**特征牛熊
    （熊市里也有走牛个股），而交易计划顶层的 ``regime`` 是**指数基准**的大盘口径，
    两者可以不一致。这里只做文案融合，**不动评分**，让用户一眼看出
    「这是大盘熊市下个股逆势走强」，而不是以为系统把牛熊搞反了。
    """
    r = (reason or "").strip()
    if not r:
        return r
    # 把个股口吻的「顺风/逆风」转成语义更清楚的「逆势/顺势」，再带大盘语境
    if "个股牛市顺风" in r:
        base = "个股逆势走强" if market_regime == "bear" else "个股共振走强"
    elif "个股牛市逆风" in r:
        base = "个股牛市却走弱" if market_regime == "bull" else "个股逆势走弱"
    elif "个股熊市顺风" in r:
        base = "顺势防御"
    elif "个股熊市逆风" in r:
        base = "逆风抗跌"
    else:
        base = r
    ctx = _MARKET_REGIME_CTX.get(market_regime or "")
    return f"{base}（{ctx}）" if ctx else base


def _apply_ensemble_vote(
    df: pd.DataFrame,
    stock_data: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """用多策略投票复核 AI 候选。

    返回 ``(候选, 统计)``。统计里 ``fallback=True`` 表示「全部被否决 → 原样返回」，
    必须与「一票都没否决」区分开：否则报表显示 20→20、否决 0 只，
    值班的人会以为复核很干净，实际是它整个失效了。


    为什么是「复核」而不是「替换」
    ----------------------------
    AI 打分是本系统的主航道，量化三策略（多因子 / 均值回归 / AI+量价融合）
    只做两件事：集体看空则**否决**，一致看多则按一致度**加权**。

    三条不误伤的兜底（缺一不可）：
    * 行情不足 60 根 K 线 → 保留（拿不到数据不惩罚候选）；
    * 策略无共识（分歧或全无信号）→ 保留；
    * **全部被否决则直接返回原始候选** —— 绝不让一次数据抖动把推荐清空，
      与 rebalance「空计划=按兵不动、绝不误会成清仓」是同一条原则。
    """
    from src.trading.signals import QuantSignalGenerator, StrategyMode

    code_col = ("代码" if "代码" in df.columns
                else ("code" if "code" in df.columns else None))
    score_col = ("评分" if "评分" in df.columns
                 else ("score" if "score" in df.columns else None))
    empty_stats = {"before": 0, "after": 0, "vetoed": 0, "boosted": 0,
                   "skipped": 0, "fallback": False}
    if code_col is None or score_col is None or len(df) == 0:
        return df, empty_stats

    gen = QuantSignalGenerator(mode=StrategyMode.ENSEMBLE)
    kept = []
    n_vetoed = n_boosted = n_skipped = 0
    for _, row in df.iterrows():
        code = str(row[code_col])
        # 注意：不能用 `a or b` 取 DataFrame 兜底 —— 会触发真值歧义
        data = stock_data.get(code)
        if data is None:
            data = stock_data.get(code.zfill(6))
        if data is None or len(data) < 60:
            n_skipped += 1
            kept.append(row)
            continue
        try:
            sig = gen.generate_ensemble(data, code, str(row.get("名称", code)))
        except Exception:  # noqa: BLE001
            n_skipped += 1
            kept.append(row)
            continue
        if sig is None or sig.signal == 0:
            kept.append(row)          # 无共识 → 留着，不否决也不提分
            continue
        if sig.signal < 0:
            n_vetoed += 1             # 三策略集体看空 → 否决
            continue
        new_row = row.copy()
        # 一致度越高提分越多，上限 +10%（复核是辅助，不喧宾夺主）
        new_row[score_col] = round(
            float(row[score_col]) * (0.9 + 0.2 * min(max(sig.agreement, 0.0), 1.0)),
            1,
        )
        n_boosted += 1
        kept.append(new_row)

    if not kept:
        # 全票否决 → 原样返回，并把 fallback 标出来，别假装「什么问题都没有」
        return df, {"before": len(df), "after": len(df), "vetoed": n_vetoed,
                    "boosted": 0, "skipped": n_skipped, "fallback": True}
    out = pd.DataFrame(kept).reset_index(drop=True)
    try:
        out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    except Exception:  # noqa: BLE001
        pass
    return out, {"before": len(df), "after": len(out), "vetoed": n_vetoed,
                 "boosted": n_boosted, "skipped": n_skipped, "fallback": False}


def _apply_score_threshold(
    df: pd.DataFrame,
    min_score: float,
    target_n: int,
    report: Optional[Dict[str, Any]] = None,
    allow_fill: bool = True,
) -> pd.DataFrame:
    """按「过关优先 + 排名补足」筛选候选，而不是用绝对门槛一票否决。

    **为什么改**：熊市里全市场评分整体下移，而 ``regime_adj`` 还会把阈值
    再加 5 分（50→55），于是通过数掉到 4~6 只 —— 实测 2026-10-26 只有 4 只、
    10-27 只有 6 只。后果有两层：

    1. 组合被动集中：4 只票 vs 回测假设的 8 只，单票权重翻倍，波动与回撤
       都被放大。回测的 +25%/年、-25% 回撤是按 8 只算的，实盘根本不是同一个
       组合，两个数字天然不可比。
    2. 它其实是个**顺周期开关**：越跌→分数越低→通过越少→仓位越轻，反弹时
       恰好仓位最轻。回测里没有这个概念（永远按排名取前 N 只）。

    所以改成：达标的照常优先，不足目标只数时按评分从高到低补足，但补位票
    不得低于硬底线 ``THRESHOLD_MIN``（30 分），避免把明显不合格的标的塞进来。
    这样「选择性」由排名承担，「门槛」只做极端兜底 —— 与回测口径一致。

    Args:
        df: 按评分降序的完整榜单（应按 ``max(max_picks, target_n)`` 宽取）
        min_score: 当前生效阈值（基线 + 进化 + 市场状态）
        target_n: 目标持仓只数（``trading_loop.max_positions``）
        report: 结构化报告（就地写入 n_passed_strict / n_filled / fill_note）
        allow_fill: 是否允许「排名补足」。默认 True（与回测口径一致）。
            设为 False 即「宁缺毋滥」：达标几只就是几只，0 只就空仓 —— 由
            ``scheduler.require_strict_pass`` 显式开启，绝不悄悄改默认行为。

    Returns:
        过滤后的 DataFrame，仍为评分降序；不足目标数时返回尽可能多的票。
    """
    if df is None or len(df) == 0:
        return df
    col = "评分" if "评分" in df.columns else None
    if col is None:
        return df

    scores = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    passed = df[scores >= float(min_score)]
    n_pass = len(passed)
    if report is not None:
        report["n_passed_strict"] = int(n_pass)
        report["fill_allowed"] = bool(allow_fill)

    if not allow_fill:
        if report is not None:
            report["n_filled"] = 0
            report["fill_note"] = (
                f"宁缺毋滥（require_strict_pass=True）：仅保留达标 {n_pass} 只，"
                f"不做排名补足（目标 {int(target_n)} 只）"
            )
            if n_pass < int(target_n):
                logger.warning(report["fill_note"])
        return passed

    need = int(target_n) - n_pass
    if need <= 0:
        if report is not None:
            report["n_filled"] = 0
        return passed

    fillable = df[(scores < float(min_score)) & (scores >= float(_SCORE_FLOOR))]
    fill = fillable.head(need)
    if report is not None:
        report["n_filled"] = int(len(fill))
        if len(fill):
            report["fill_note"] = (
                f"达标仅 {n_pass} 只（阈值 {min_score:.1f}），按评分补足 "
                f"{len(fill)} 只到目标 {int(target_n)} 只 —— 保持与回测一致的"
                f"分散度，避免熊市里组合被动集中"
            )
            logger.warning(report["fill_note"])
    if len(fill) == 0:
        return passed
    return pd.concat([passed, fill]) if len(passed) else fill


def _ranking_of(df: Optional[pd.DataFrame]) -> list:
    """把「按评分降序的完整榜单」压成 ``[{代码, 名称, 评分}]``。

    交易计划的 ``raw_rank`` 吃这个，用来判断「老持仓还在不在保留线内」。
    刻意**不**含 picks 的门槛筛选：门槛决定「今天买什么」，排名决定「老仓留不留」。
    混为一谈的代价见 ``generate_daily_picks`` 里的注释。
    """
    if df is None or len(df) == 0:
        return []
    ccol = "代码" if "代码" in df.columns else ("code" if "code" in df.columns else None)
    if ccol is None:
        return []
    ncol = "名称" if "名称" in df.columns else ("name" if "name" in df.columns else None)
    scol = "评分" if "评分" in df.columns else ("score" if "score" in df.columns else None)
    out = []
    for r in df.to_dict(orient="records"):
        code = str(r.get(ccol) or "").strip()
        if not code:
            continue
        item: Dict[str, Any] = {"代码": code}
        if ncol:
            item["名称"] = str(r.get(ncol) or "")
        if scol:
            try:
                item["评分"] = round(float(r.get(scol) or 0), 2)
            except Exception:  # noqa: BLE001
                item["评分"] = 0.0
        out.append(item)
    return out


# data_update 的跨进程锁（与 auto_scheduler._job_lease 同一位置：data/.job_<id>.lock）
_DATA_UPDATE_LOCK = str(Path(DATA_DIR) / ".job_data_update.lock")


def _wait_for_data_update(max_wait: Optional[float] = None, poll: float = 5.0) -> float:
    """若**另一个进程**正在跑 ``data_update``（拉全池行情），先等它写完再读数。

    为什么值得等（2026-11-18 实测）
    ------------------------------
    凌晨补跑时出现过这样的现场：常驻 run_loop 正在执行 data_update（拉全池），
    而另一进程补跑的 daily_pick 因为"缓存还是旧的"又自己拉了 425 只全量 ——
    两拨 8 进程并发抢同一条网络，双方都变慢：daily_pick 的拉行情从正常 130s
    涨到 **393s**（占整次选股 79%），而这批数据它**一行都不需要**：
    data_update 写完缓存后，daily_pick 本来就能直接命中。

    等锁释放（最多 ``max_wait`` 秒）通常比"重复拉一遍全池"快 3 倍以上；
    超时则照常自己拉，绝不把选股卡死。

    Returns:
        实际等待秒数（0 表示无需等待）。
    """
    if max_wait is None:
        try:
            max_wait = float(getattr(app_config.scheduler, "wait_for_data_update_sec", 900) or 0)
        except Exception:  # noqa: BLE001
            max_wait = 900.0
    if max_wait <= 0:
        return 0.0
    path = Path(_DATA_UPDATE_LOCK)
    if not path.exists():
        return 0.0
    # 与 auto_scheduler._job_lease 同一口径：锁超过 1800s 视为陈旧（持锁进程已死）
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return 0.0
    if age >= 1800:
        return 0.0
    logger.info(
        f"⏳ 另一个进程正在更新行情（锁 {path.name}，已 {age:.0f}s），"
        f"先等它写完，最多等 {max_wait:.0f}s —— 避免重复拉全池")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(poll)
        if not path.exists():
            waited = time.time() - t0
            logger.info(f"✅ 行情更新已完成（等待 {waited:.0f}s），直接使用缓存")
            return waited
        try:
            if time.time() - path.stat().st_mtime >= 1800:
                logger.warning("⚠️ 行情更新锁已陈旧（>1800s），不再等待，自行刷新")
                return time.time() - t0
        except OSError:
            return time.time() - t0
    logger.warning(f"⚠️ 等待行情更新超时（{max_wait:.0f}s），改为自行拉取")
    return time.time() - t0


def generate_daily_picks(
    pool: Optional[str] = None,
    max_picks: Optional[int] = None,
    min_score: Optional[float] = None,
    fetcher_inst=None,
    predictor_inst=None,
    evolution_inst=None,
    write_report: bool = True,
    use_ensemble: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    收盘后自动生成每日推荐。

    use_ensemble: 用多策略投票对 AI 候选做一次复核（否决 + 加权）。
        默认关闭，启用需显式传入（或 CLI ``--ensemble``）。

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
        "ranking": [],
        "status": "ok",
        "error": None,
    }
    # 阶段耗时（秒）：把"选股为什么慢"从猜测变成可核对的数字
    timing: Dict[str, float] = {}
    _t_all = time.perf_counter()

    try:
        codes = fet.get_stock_pool(pool)
        report["n_scanned"] = len(codes)

        # 拉取行情（会顺带刷新本地缓存）
        # 若另一进程正在跑 data_update，先等它（见 _wait_for_data_update）：
        # 等 1 分钟比重复拉 425 只全量（实测 393s）划算得多。
        timing["wait_data_update"] = round(_wait_for_data_update(), 1)
        _t0 = time.perf_counter()
        raw = fet.get_batch_daily_kline(codes, max_workers=8)
        timing["fetch_kline"] = round(time.perf_counter() - _t0, 1)
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

        # 宽取榜单（先不做绝对门槛），再由 _apply_score_threshold 做
        # 「过关优先 + 排名补足」—— 只取到 max_picks 会让补位无从下手。
        #
        # ⚠️ 宽度还必须覆盖**排名滞后**的保留窗口：交易计划的 ``raw_rank``
        # 就是从这张榜单来的，而持仓保留线是 ``max_positions × hold_rank_slack``
        # （生产 8×3=24）。只取 max_picks=40 时看着够，但只要 max_positions 上调、
        # 或 slack 调到 6，榜单就比保留线还窄 —— 那时「排名在保留线内」这件事
        # 根本无法判断，slack 静默退化成 1.0（等于每个调仓日清掉所有老持仓）。
        # 2026-11-05 实测：报告只有 9 只通过门槛，计划 raw_rank 就只有 9 条，
        # 而保留线是 24 ⇒ 5 只老持仓全部被判「跌出目标组合，清仓」。
        _target_n = int(getattr(app_config.trading_loop, "max_positions", 8) or 8)
        _slack = float(getattr(app_config.trading_loop, "hold_rank_slack", 1.0) or 1.0)
        _wide_n = max(int(max_picks), _target_n, int(round(_target_n * max(_slack, 1.0))))
        _t0 = time.perf_counter()
        df_wide = pred.get_top_recommendations(
            stock_data, stock_names, top_n=_wide_n, min_score=0.0
        )
        timing["score_all"] = round(time.perf_counter() - _t0, 1)
        # 完整榜单落进报告：交易计划的排名表用它，而不是用「筛过门槛的 picks」。
        # 两者不是一回事 —— picks 是**今天要买什么**，ranking 是**模型怎么看全市场**，
        # 「老持仓该不该留」问的是后者。
        report["ranking"] = _ranking_of(df_wide)

        # 信号强度诊断（纯观测）：回答"今天这套推荐到底还有没有统计边际"。
        # 用**全截面**评分算区分度与达标率，而不是只用 Top40 —— 否则
        # "模型整体塌陷"会被误读成"今天没选出好票"。失败不影响选股。
        _signal = None
        try:
            _signal = compute_signal_strength(
                getattr(pred, "last_batch_scores", None), float(min_score), _target_n)
            _as_of = datetime.now().strftime("%Y-%m-%d")
            _signal["date"] = _as_of       # 供趋势比对时排除"今天这条自己"
            _trend = signal_trend(_signal)
            _signal["trend"] = _trend.get("state")
            _signal["trend_note"] = _trend.get("note")
            report["signal_strength"] = _signal
            report["signal_trend"] = _trend
            record_signal_history(_signal, as_of=_as_of)
            logger.info(format_signal_strength(_signal, _trend))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"信号强度诊断失败（不影响推荐）: {e}")

        # 「宁缺毋滥」闸门（默认关）：开启后不再按排名补足，只买严格达标的票。
        # 11-13 那种「达标 0 只仍补满 8 只」的局面下，它会让组合空仓等信号 ——
        # 是"质量优先"的显式选择，而不是偷偷改默认行为。
        _allow_fill = not bool(getattr(app_config.scheduler, "require_strict_pass", False))
        if not _allow_fill:
            logger.info("选股闸门：require_strict_pass=True，本次不做排名补足（只买达标票）")
        df = _apply_score_threshold(df_wide, float(min_score), _target_n, report,
                                    allow_fill=_allow_fill)

        # 参考价：把推荐时点的最新收盘价带进 picks，供交易计划 / 执行侧做
        # 「跳空太多就不追」这类校验。早期没有这一步，计划里 price 恒为 0.0，
        # 字段名存实亡。取不到就填 0，不影响推荐本身。
        try:
            if len(df):
                col = "代码" if "代码" in df.columns else ("code" if "code" in df.columns else None)
                if col is not None:
                    def _last_close(c):
                        # ⚠️ 不能用 `df or fallback`：pandas DataFrame 的布尔值
                        # 是「ambiguous」，直接 `bool(df)` 会抛异常（曾被外层
                        # except 静默吞掉，导致最新价从未真正注入）。改用显式
                        # None 判断。
                        d = stock_data.get(str(c))
                        if d is None:
                            d = stock_data.get(str(c).zfill(6))
                        if d is None or len(d) == 0:
                            return 0.0
                        try:
                            return round(float(d["close"].iloc[-1]), 2)
                        except Exception:  # noqa: BLE001
                            return 0.0
                    df = df.copy()
                    df["最新价"] = df[col].map(_last_close)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"参考价补充失败（不影响推荐）: {e}")

        # 多策略复核（默认关）。AI 打分是主航道，量化三策略只做「否决 + 加权」：
        # 集体看空才剔除、一致看多才提分；拿不到数据的一律保留，
        # 绝不因为数据缺失而误杀整批候选。
        if use_ensemble and len(df):
            try:
                df, ens = _apply_ensemble_vote(df, stock_data)
                report["ensemble"] = ens
            except Exception as e:  # noqa: BLE001
                logger.debug(f"多策略复核失败（保留原候选）：{e}")

        # 市场状态刷新（写入进化引擎，供阈值自适应）
        try:
            bd = fet.get_benchmark_kline("000300.SH", start_date="2024-01-01")
            report["regime"] = evo.detect_regime(bd)
        except Exception as e:
            logger.warning(f"regime 检测失败: {e}")

        report["n_passed"] = len(df)
        # 把「个股口吻」理由叠加「大盘口吻」（regime 是指数基准，两者口径不同），
        # 避免 plan 顶层 regime=bear 却显示「牛市顺风」这种看着像矛盾、实则合法的文案。
        if len(df) and "推荐理由" in df.columns:
            mkt = report.get("regime")
            if mkt:
                df["推荐理由"] = df["推荐理由"].map(lambda r: _annotate_reason(str(r), mkt))
        report["picks"] = df.to_dict(orient="records") if len(df) else []
        if len(df) == 0:
            report["status"] = "empty"
            logger.info(f"每日推荐为空（阈值 {min_score:.1f} 过高或行情缺失）")
        else:
            logger.info(f"✅ 每日推荐生成: {len(df)} 只（阈值 {min_score:.1f}）")

        # 超级集成健康审计：把各腿可用状态/降级原因透出到报告
        # （看板/日报/运维可直接读 ensemble_health；ensemble_alarm 为聚焦告警文本）。
        # 默认 ml.use_ensemble=False 时 ensemble_report 返回「未启用」、alarm 返回 None，
        # 零行为改变、零额外开销（若已启用，子模型在上文 get_top_recommendations
        # 里已被 ensure，不会二次加载）。
        try:
            report["ensemble_health"] = pred.ensemble_report()
            report["ensemble_alarm"] = pred.ensemble_degradation_alert()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"超级集成健康审计失败（不影响推荐）: {e}")

        # 耗时拆解：直接回答"选股为什么慢"，也让优化前后的收益可核对
        timing["total"] = round(time.perf_counter() - _t_all, 1)
        timing["other"] = round(max(0.0, timing["total"]
                                    - timing.get("fetch_kline", 0.0)
                                    - timing.get("score_all", 0.0)
                                    - timing.get("wait_data_update", 0.0)), 1)
        report["timing"] = timing
        _wait_s = timing.get("wait_data_update", 0.0)
        logger.info(
            f"⏱️ 选股耗时：合计 {timing['total']}s（拉行情 {timing.get('fetch_kline', 0)}s"
            + (f" / 等行情更新 {_wait_s}s" if _wait_s else "")
            + f" / 打分 {timing.get('score_all', 0)}s / 其他 {timing['other']}s）"
            f" · {report.get('n_scanned', 0)} 只")

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
    ss = report.get("signal_strength")
    if isinstance(ss, dict) and ss.get("level") not in (None, "unknown"):
        trend = report.get("signal_trend") if isinstance(report.get("signal_trend"), dict) else None
        lines.append("```")
        lines.append(format_signal_strength(ss, trend))
        lines.append("```")
        alarm = signal_strength_alert(ss, trend)
        if alarm:
            lines.append("")
            lines.append("> " + alarm.replace("\n", "\n> "))
        lines.append("")
    tm = report.get("timing")
    if isinstance(tm, dict) and tm.get("total") is not None:
        lines.append(
            f"- 耗时：合计 {tm.get('total')}s（拉行情 {tm.get('fetch_kline', 0)}s"
            f" / 打分 {tm.get('score_all', 0)}s / 其他 {tm.get('other', 0)}s）"
        )
        lines.append("")
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

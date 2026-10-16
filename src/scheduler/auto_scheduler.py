# -*- coding: utf-8 -*-
"""
收盘后自动化调度器（APScheduler）
================================

让系统「无人值守」自动运转：

  · 工作日 18:30  更新数据（刷新本地行情缓存）
  · 工作日 18:45  生成每日 AI 推荐（调用 core.daily_pick）
  · 工作日 19:00  回测验证 + 触发自我进化（复用 tracker_v2.verify_all_pending）

安全约束：
  · 默认 config.scheduler.enabled = False，不启用绝不启动定时任务。
  · 调度器只做「数据 + 推荐 + 进化验证」，绝不直接下单（下单由交易安全中枢 guard 管辖）。
  · APScheduler 若未安装，start_scheduler 安全返回 None，不影响其他功能。
  · 每个 job 都包了 try/except，单任务失败不影响其他任务与整个进程。

用法：
  from src.scheduler.auto_scheduler import start_scheduler, stop_scheduler, run_all_once
  start_scheduler()          # 受 config.scheduler.enabled 控制
  run_all_once(dry_run=True) # 立即跑一轮（测试/手动触发），dry_run 不写报告
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    _HAS_APS = True
except Exception:  # noqa: BLE001
    BackgroundScheduler = None
    CronTrigger = None
    _HAS_APS = False

from config.settings import config as app_config
from src.data.fetcher import fetcher
from src.models.predictor import predictor
from src.models.evolution import evolution
from src.core.daily_pick import generate_daily_picks


_scheduler: Optional["BackgroundScheduler"] = None
_lock = threading.Lock()


# ----------------------------------------------------------------------------
# 单个任务（独立可测，不依赖 APScheduler）
# ----------------------------------------------------------------------------

def job_data_update() -> Dict:
    """刷新本地行情缓存。无显式 update 方法时，退化为拉取当前池最新行情。"""
    info: Dict = {"job": "data_update", "at": datetime.now().isoformat(timespec="seconds"),
                  "status": "ok", "refreshed": 0, "error": None}
    try:
        if hasattr(fetcher, "update_data"):
            fetcher.update_data()
            info["refreshed"] = -1  # 未知数量
            logger.info("🔄 数据更新完成（update_data）")
        else:
            pool = app_config.scheduler.pick_pool
            codes = fetcher.get_stock_pool(pool)
            fetcher.get_batch_daily_kline(codes, max_workers=8)
            info["refreshed"] = len(codes)
            logger.info(f"🔄 数据缓存刷新完成：{pool} 池 {len(codes)} 只")
    except Exception as e:  # noqa: BLE001
        info["status"] = "error"
        info["error"] = str(e)
        logger.error(f"数据更新失败: {e}\n{traceback.format_exc()}")
    return info


def job_daily_pick(dry_run: bool = False) -> Dict:
    """生成每日 AI 推荐并写报告；同步产出「交易计划」；可选推送到企微/Webhook。

    闭环关键一环：推荐只写报告的话，盘中交易拿不到标的。这里顺手把它转成
    交易计划（trade_plan），盘中的闭环引擎（src/trading/loop.py）就能直接消费。
    """
    info: Dict = {"job": "daily_pick", "at": datetime.now().isoformat(timespec="seconds"),
                  "status": "ok", "n_passed": 0, "pushed": False,
                  "plan_date": "", "plan_targets": 0, "error": None}
    try:
        _, report = generate_daily_picks(write_report=not dry_run)
        info["n_passed"] = report.get("n_passed", 0)
        info["status"] = report.get("status", "ok")
        info["error"] = report.get("error")
        info["report"] = report
        logger.info(f"📈 每日推荐完成：{info['n_passed']} 只（status={info['status']}）")

        # 推荐 → 交易计划（闭环的桥）
        if report.get("status") in ("ok", "empty"):
            try:
                from src.trading.trade_plan import build_trade_plan
                tc = app_config.trading_loop
                # 与风控口径对齐：单票权重不得超过 guard 的单票上限
                cap = tc.weight_cap
                try:
                    from src.trading.guard import GuardConfig as _GC
                    from config.settings import CONFIG_DIR
                    _g = Path(CONFIG_DIR) / "trading_guard.yaml"
                    _cfg = _GC.from_yaml(str(_g)) if _g.exists() else _GC()
                    cap = min(cap, float(_cfg.max_single_position_pct))
                except Exception:  # noqa: BLE001
                    pass

                plan = build_trade_plan(
                    picks=report.get("picks") or [],
                    pool=report.get("pool", ""),
                    min_score=float(report.get("min_score", 0) or 0),
                    regime=str(report.get("regime") or ""),
                    max_positions=tc.max_positions,
                    cash_reserve_pct=tc.cash_reserve_pct,
                    weighting=tc.weighting,
                    weight_cap=cap,
                    write=not dry_run,
                )
                info["plan_date"] = plan.plan_date
                info["plan_targets"] = len(plan.targets)
                logger.info(f"📋 已生成交易计划 {plan.plan_date}：{len(plan.targets)} 只")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"交易计划生成失败（已忽略）: {e}")

        # 推送闭环：推荐生成后推送到 config.notifier 配置的渠道（默认关闭）
        if not dry_run and report.get("status") in ("ok", "empty"):
            try:
                from src.core.notifier import build_notifier
                from src.core.daily_pick import _to_markdown
                notifier = build_notifier()
                md = _to_markdown(report)
                if info["plan_date"]:
                    md += (f"\n\n---\n📋 已据此生成 {info['plan_date']} 交易计划"
                           f"（{info['plan_targets']} 只），盘中闭环将按计划自动调仓。")
                info["pushed"] = notifier.send(
                    f"每日选股推荐 ({report.get('n_passed', 0)}只)", md,
                    payload={"picks": report.get("picks", []), "regime": report.get("regime")},
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"推送失败（已忽略）: {e}")
    except Exception as e:  # noqa: BLE001
        info["status"] = "error"
        info["error"] = str(e)
        logger.error(f"每日推荐失败: {e}\n{traceback.format_exc()}")
    return info


def job_evolution() -> Dict:
    """
    回测验证 + 触发自我进化（与看板「运行回测验证」完全一致的逻辑）。
    复用 src.models.tracker_v2.verify_all_pending()。

    为什么必须走 v2：旧 tracker.py 用「今天减 N 天」反推日期做字符串相等匹配，
    周末/节假日必然对不上，且 T+5 与 T+10 共用一个 `verification` 槽位互相覆盖。
    返回的字典键也与 v2 一致（``hold_days`` 而非 ``lookback_days``）——
    ``evolution.record_verification`` 两种键都认，不会把 T+10 记到 T+5 上。
    """
    info: Dict = {"job": "evolution", "at": datetime.now().isoformat(timespec="seconds"),
                  "status": "ok", "n_verified": 0, "n_actions": 0, "error": None}
    try:
        from src.models.tracker_v2 import verify_all_pending
        results = verify_all_pending() or []
        info["n_verified"] = len(results)
        actions = []
        for r in results:
            actions.extend(evolution.evolve_from_verification(r))
        info["n_actions"] = len(actions)
        info["actions"] = actions

        # 市场状态刷新（写入进化引擎，驱动阈值自适应）
        try:
            bd = fetcher.get_benchmark_kline("000300.SH", start_date="2024-01-01")
            regime = evolution.detect_regime(bd)
            info["regime"] = regime
        except Exception as e:
            logger.warning(f"regime 刷新失败: {e}")

        # 离线进化监测
        try:
            should, reason = evolution.should_retrain()
            info["should_retrain"] = should
            info["retrain_reason"] = reason
        except Exception:
            pass

        # 观测心跳：作业跑完即刷新 last_check_at（即使本批无可验证样本），
        # 用于区分「链路断了」与「只是样本未到」。见 evolution.mark_checked 说明。
        try:
            evolution.mark_checked()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"mark_checked 失败（已忽略）: {e}")

        logger.info(
            f"🧬 进化任务完成：验证 {len(results)} 批，进化动作 {len(actions)} 条"
            + (f"，建议重训：{reason}" if info.get("should_retrain") else "")
        )
    except Exception as e:  # noqa: BLE001
        info["status"] = "error"
        info["error"] = str(e)
        logger.error(f"进化任务失败: {e}\n{traceback.format_exc()}")
    return info


# ----------------------------------------------------------------------------
# 调度器控制
# ----------------------------------------------------------------------------

def _make_trigger(cron: str):
    if CronTrigger is None:
        return None
    try:
        return CronTrigger.from_crontab(cron, timezone=app_config.scheduler.timezone)
    except Exception as e:  # noqa: BLE001
        logger.error(f"cron 解析失败 '{cron}': {e}")
        return None


def start_scheduler() -> Optional["BackgroundScheduler"]:
    """启动调度器。受 config.scheduler.enabled 控制；未启用则返回 None。"""
    global _scheduler
    with _lock:
        if _scheduler is not None and getattr(_scheduler, "running", False):
            logger.info("调度器已在运行")
            return _scheduler
        if not app_config.scheduler.enabled:
            logger.warning("⚠️ 调度器未启用（config.scheduler.enabled=False），不启动定时任务。"
                           "请在配置中开启，或调用 run_all_once() 手动跑一轮。")
            return None
        if not _HAS_APS:
            logger.error("❌ APScheduler 未安装，无法启动调度。pip install apscheduler 后重试。")
            return None

        sched = BackgroundScheduler(timezone=app_config.scheduler.timezone)
        jobs = [
            ("data_update", app_config.scheduler.data_update_cron, job_data_update),
            ("daily_pick", app_config.scheduler.daily_pick_cron, job_daily_pick),
            ("evolution", app_config.scheduler.evolution_cron, job_evolution),
        ]
        for jid, cron, fn in jobs:
            trig = _make_trigger(cron)
            if trig is None:
                logger.error(f"任务 {jid} 的 cron '{cron}' 无效，跳过")
                continue
            sched.add_job(fn, trig, id=jid, replace_existing=True,
                          misfire_grace_time=3600)
        sched.start()
        _scheduler = sched
        logger.info("✅ 收盘后自动化调度已启动（APScheduler 后台运行）")
        return sched


def stop_scheduler() -> bool:
    """停止调度器。"""
    global _scheduler
    with _lock:
        if _scheduler is None:
            return False
        try:
            _scheduler.shutdown(wait=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"停止调度器时异常: {e}")
        _scheduler = None
        logger.info("🛑 调度器已停止")
        return True


def is_running() -> bool:
    return _scheduler is not None and getattr(_scheduler, "running", False)


def run_all_once(dry_run: bool = False) -> Dict:
    """
    立即顺序跑一轮全部任务（手动触发 / 测试用）。
    dry_run=True 时每日推荐不写报告文件。
    """
    logger.info(f"🚀 手动跑一轮自动化任务（dry_run={dry_run}）")
    out = {
        "data_update": job_data_update(),
        "daily_pick": job_daily_pick(dry_run=dry_run),
        "evolution": job_evolution(),
    }
    out["ran_at"] = datetime.now().isoformat(timespec="seconds")
    # 简要落盘（便于追溯）
    try:
        log_dir = Path(app_config.scheduler.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        hist = log_dir / f"scheduler_runs_{stamp}.jsonl"
        with open(hist, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"运行历史写入失败: {e}")
    return out


if __name__ == "__main__":
    print(json.dumps(run_all_once(dry_run=True), ensure_ascii=False, indent=2, default=str))

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
import os
import threading
import time
import traceback
from contextlib import contextmanager
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

from config.settings import DATA_DIR
from config.settings import config as app_config
from src.core.single_instance import SingleInstanceLock
from src.data.fetcher import fetcher
from src.models.predictor import predictor
from src.models.evolution import evolution
from src.core.daily_pick import generate_daily_picks


_scheduler: Optional["BackgroundScheduler"] = None
_lock = threading.Lock()

# 「调度器本体」的进程级持有者：谁抢到 data/.scheduler.lock，谁才真正起
# APScheduler。看板与常驻 run_loop 都会调 start_scheduler()，靠这把锁收敛到
# 唯一进程，避免出现两份互不知情的 APScheduler（用户称之为"双调度"）。
_sched_holder = None          # type: Optional["SingleInstanceLock"]

# 作业租约锁文件目录（测试可 monkeypatch 到 tmp_path，避免在 data/ 留痕）
LEASE_DIR: Path = Path(DATA_DIR)
# 调度器本体锁目录（测试同样可 monkeypatch）
SCHED_LOCK_DIR: Path = Path(DATA_DIR)


# ----------------------------------------------------------------------------
# 跨进程作业租约（防止看板与 run_loop 两份调度器重复触发同一个 cron）
# ----------------------------------------------------------------------------
@contextmanager
def _job_lease(job_id: str, ttl_sec: float = 1800.0):
    """给一个作业加跨进程互斥锁，yield True 表示拿到租约。

    背景：调度器会随「看板进程」和「常驻 run_loop 进程」各自拉起一份。
    两个进程的 APScheduler 互不知情，18:45 的 daily_pick 会被跑两遍 ——
    重复扫全池、重复写报告与交易计划，进化作业甚至可能重复应用同一批参数变更。

    用 ``O_CREAT|O_EXCL`` 建一个锁文件做跨进程互斥；进程崩溃留下的陈旧锁
    按 ``ttl_sec`` 接管（所以 ttl 必须大于单个作业的最长耗时）。
    """
    path = Path(LEASE_DIR) / f".job_{job_id}.lock"
    acquired = False

    def _try_acquire() -> bool:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except Exception:  # noqa: BLE001
            return False
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
        return True

    if _try_acquire():
        acquired = True
    else:
        # 已有进程持锁：判断是否陈旧
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            age = 0.0
        if age < ttl_sec:
            logger.info(f"⏭️  作业 {job_id} 正由其他进程执行中（锁 {path.name}，{age:.0f}s），本次跳过")
        else:
            logger.warning(f"⚠️  作业 {job_id} 的锁已陈旧（{age:.0f}s），接管执行")
            try:
                path.unlink()
            except OSError:
                pass
            acquired = _try_acquire()

    try:
        yield acquired
    finally:
        if acquired:
            try:
                path.unlink()
            except OSError:
                pass


# ----------------------------------------------------------------------------
# 单个任务（独立可测，不依赖 APScheduler）
# ----------------------------------------------------------------------------

def _job_data_update_impl() -> Dict:
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


def _job_daily_pick_impl(dry_run: bool = False) -> Dict:
    """生成每日 AI 推荐并写报告；同步产出「交易计划」；可选推送到企微/Webhook。

    闭环关键一环：推荐只写报告的话，盘中交易拿不到标的。这里顺手把它转成
    交易计划（trade_plan），盘中的闭环引擎（src/trading/loop.py）就能直接消费。
    """
    info: Dict = {"job": "daily_pick", "at": datetime.now().isoformat(timespec="seconds"),
                  "status": "ok", "n_passed": 0, "pushed": False,
                  "plan_date": "", "plan_targets": 0, "error": None}
    try:
        _, report = generate_daily_picks(
            write_report=not dry_run,
            use_ensemble=bool(app_config.scheduler.pick_ensemble),
        )
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
                    tilt_strength=tc.tilt_strength,
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


def _job_evolution_impl() -> Dict:
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
# 对外入口：带跨进程租约的作业（看板与 run_loop 各自起一份调度器也不会重复跑）
# ----------------------------------------------------------------------------

def _skipped(job: str, ttl: float) -> Dict:
    return {"job": job, "status": "skipped",
            "reason": "另一个进程正在执行同一作业（租约被占用）",
            "at": datetime.now().isoformat(timespec="seconds")}


def _job_industry_map_impl() -> Dict:
    """机会主义地补齐「代码 → 所属行业」。

    为什么要单开一个高频作业：本机到东财是**间歇性**可达（2026-10-19 实测
    30 分钟内 40 次尝试全败，但此前某两分钟里能正常取到 496 个行业板块）。
    行业映射一旦补齐就永久落盘、之后不再打网络，所以**多试几次就有意义**；
    而每日一次（18:45 选股时顺带尝试）的曝光率太低了。

    已经有了就不再折腾（直接返回 already_ok），没拿到也不算失败 ——
    中性化会退到板级分组，交易照常，只是粒度粗。
    """
    info: Dict = {"job": "industry_map", "at": datetime.now().isoformat(timespec="seconds"),
                  "status": "ok", "n": 0}
    try:
        from src.data.fetcher import DataFetcher, fetcher
        need = DataFetcher._INDUSTRY_BULK_MIN_CODES
        cache = Path(fetcher.cache_dir) / "industry_map.json"
        try:
            have = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        except Exception:  # noqa: BLE001
            have = {}
        if len(have) >= need:
            info["status"] = "already_ok"
            info["n"] = len(have)
            return info

        got = fetcher.get_industry_map(force_refresh=True) or {}
        info["n"] = len(got)
        info["status"] = "ok" if len(got) >= need else "empty"
        if info["status"] == "ok":
            logger.info(f"🏷️  行业映射已补齐：{len(got)} 只 / "
                        f"{len(set(got.values()))} 个行业 → 中性化切回行业粒度")
        else:
            logger.debug(f"行业映射仍未拿到（{len(got)}/{need}），继续走板级中性化")
    except Exception as e:  # noqa: BLE001
        info["status"] = "error"
        info["error"] = str(e)
        logger.warning(f"行业映射补齐失败（已忽略）: {e}")
    return info


def job_industry_map() -> Dict:
    """行业映射补齐（带跨进程租约，每小时一次）。"""
    with _job_lease("industry_map", ttl_sec=1800) as ok:
        if not ok:
            return _skipped("industry_map", 1800)
        return _job_industry_map_impl()


def job_data_update() -> Dict:
    """数据更新（带跨进程租约）。"""
    with _job_lease("data_update", ttl_sec=1800) as ok:
        if not ok:
            return _skipped("data_update", 1800)
        return _job_data_update_impl()


def job_daily_pick(dry_run: bool = False) -> Dict:
    """每日推荐 + 交易计划（带跨进程租约）。dry_run 不写报告。"""
    with _job_lease("daily_pick", ttl_sec=3600) as ok:
        if not ok:
            return _skipped("daily_pick", 3600)
        return _job_daily_pick_impl(dry_run=dry_run)


def job_evolution() -> Dict:
    """回测验证 + 自我进化（带跨进程租约）。"""
    with _job_lease("evolution", ttl_sec=3600) as ok:
        if not ok:
            return _skipped("evolution", 3600)
        return _job_evolution_impl()


def job_rolling_ic(days: int = 5, dry_run: bool = False) -> Dict:
    """每日 RankIC 回填（带跨进程租约）。

    默认只补最近 ``days`` 个交易日的缺失评分日（收盘后每天跑一次，只增量补当天
    及最近遗漏的点），避免长时间回填阻塞调度；大批量历史补齐请走
    ``scripts/rolling_ic_backfill.py --days 30``。
    """
    with _job_lease("rolling_ic", ttl_sec=3600) as ok:
        if not ok:
            return _skipped("rolling_ic", 3600)
        return _job_rolling_ic_impl(days=days, dry_run=dry_run)


def _job_rolling_ic_impl(days: int = 5, dry_run: bool = False) -> Dict:
    try:
        from src.core.rolling_ic import run_backfill
        res = run_backfill(days=days, dry_run=dry_run)
    except Exception as e:  # noqa: BLE001
        logger.exception("rolling_ic 回填失败")
        return {"job": "rolling_ic", "status": "error", "error": str(e),
                "at": datetime.now().isoformat(timespec="seconds")}
    return {"job": "rolling_ic", **res, "at": datetime.now().isoformat(timespec="seconds")}


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
    global _scheduler, _sched_holder
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

        # 跨进程单实例：只有抢到 data/.scheduler.lock 的进程才真正起 APScheduler。
        # 看板（app.py）与常驻 run_loop 都会调 start_scheduler()，以前两者的
        # APScheduler 互不知情、各跑一遍 18:45 的 daily_pick（"双调度"）。
        # 现在收敛为唯一持有者：常驻进程先抢到，看板就自动退让不再起第二份。
        lock = SingleInstanceLock(Path(SCHED_LOCK_DIR) / ".scheduler.lock")
        if not lock.acquire():
            logger.info(
                "调度器已被另一个进程持有（通常是常驻 run_loop），"
                "本进程不再重复拉起第二份 APScheduler —— 作业由持有者跑，效果一致"
            )
            return None
        _sched_holder = lock

        sched = BackgroundScheduler(timezone=app_config.scheduler.timezone)
        jobs = [
            ("data_update", app_config.scheduler.data_update_cron, job_data_update),
            ("daily_pick", app_config.scheduler.daily_pick_cron, job_daily_pick),
            ("evolution", app_config.scheduler.evolution_cron, job_evolution),
            ("rolling_ic", app_config.scheduler.rolling_ic_cron, job_rolling_ic),
            # 机会型作业：东财间歇性可达，多给几次机会就能把行业映射补齐
            ("industry_map",
             getattr(app_config.scheduler, "industry_map_cron", "0 * * * *"),
             job_industry_map),
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
    """停止调度器，并释放「调度器本体」的进程锁（让别的进程可以接手）。"""
    global _scheduler, _sched_holder
    with _lock:
        if _scheduler is None:
            return False
        try:
            _scheduler.shutdown(wait=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"停止调度器时异常: {e}")
        _scheduler = None
        if _sched_holder is not None:
            _sched_holder.release()
            _sched_holder = None
        logger.info("🛑 调度器已停止（已释放调度进程锁）")
        return True


def is_running() -> bool:
    """调度是否**在生效**。

    不只是「本进程有没有起 APScheduler」—— 调度可能由常驻 ``run_loop`` 进程
    持有（看板因抢不到 .scheduler.lock 而主动退让）。若这里只报本进程状态，
    看板会显示「未启动」，而实际上作业照常在跑，属于误导。
    """
    if _scheduler is not None and getattr(_scheduler, "running", False):
        return True
    # 探测锁：抢得到=没人持有（空闲），抢不到=有进程持有（可能是自己，也可能是常驻进程）
    probe = SingleInstanceLock(Path(SCHED_LOCK_DIR) / ".scheduler.lock")
    got = probe.acquire()
    if got:
        probe.release()
        return False
    return True


def run_all_once(dry_run: bool = False, heavy: bool = True) -> Dict:
    """立即顺序跑一轮全部任务（手动触发 / 测试用）。

    dry_run=True 时每日推荐不写报告文件。

    rolling_ic 是**重量级**作业：要扫全池拉行情再逐日横截面打分，实测数分钟
    （开启集成融合后更长）。它只在收盘后由 cron 触发（19:15），不应出现在
    「点一下看看通不通」的同步路径上 —— 看板「🚀 立即跑一轮」按钮和测试都走
    dry_run=True，一旦带上它整个 Streamlit 主线程会被卡死几十分钟。

    因此默认规则：``dry_run=True`` 或 ``heavy=False`` 时跳过回填，只留一条 skipped
    记录；要真补点请显式 ``run_all_once(dry_run=False)`` 或直接跑
    ``scripts/rolling_ic_backfill.py``。
    """
    logger.info(f"🚀 手动跑一轮自动化任务（dry_run={dry_run}）")
    out = {
        "data_update": job_data_update(),
        "daily_pick": job_daily_pick(dry_run=dry_run),
        "evolution": job_evolution(),
    }
    if heavy and not dry_run:
        out["rolling_ic"] = job_rolling_ic(dry_run=False)
    else:
        out["rolling_ic"] = {
            "job": "rolling_ic",
            "status": "skipped",
            "reason": "试跑模式跳过重量级回填（扫全池拉行情耗时数分钟）；"
                      "需补点请 dry_run=False 或跑 scripts/rolling_ic_backfill.py",
            "at": datetime.now().isoformat(timespec="seconds"),
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

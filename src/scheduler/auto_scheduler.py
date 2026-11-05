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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

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
                    # 显式传 picks 时没有 meta，不传这一项计划里就永远没有
                    # picks_date，新鲜度审计会一直显示「未知」。而夜间这次
                    # 生成**恰恰是唯一每天都走的路径**。
                    picks_date=(str(report.get("generated_at") or "")[:10]
                                or datetime.now().strftime("%Y-%m-%d")),
                    # 完整榜单同理：显式传 picks 就绕开了 load_latest_picks 里的
                    # meta.ranking，排名表会退化成「只有今天这 8 只」——
                    # 而保留线是 max_positions×slack（24 名），老持仓一律被清。
                    ranking=report.get("ranking") or None,
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


# ----------------------------------------------------------------------------
# 作业记账 + 「错过的窗口」补跑
# ----------------------------------------------------------------------------
# 为什么需要这一层
# ----------------
# APScheduler 用的是**内存 jobstore**：进程死掉 / 机器休眠期间的时间窗，它
# 根本不知道发生过，恢复后只从「现在」往后算下一次 —— 错过的就永久错过。
# 而 ``misfire_grace_time`` 救不了：它只覆盖「进程活着、但触发晚了」的窄窗口，
# 进程不在的那段压根不会进 APScheduler 的视野。
#
# 2026-11-04 实测：常驻 loop 傍晚崩溃（调度器与它同进程），18:30 数据更新 /
# 18:45 选股 / 19:00 进化三个作业全没触发，第二天**整个交易日没有交易计划**，
# 而界面、心跳、日志一切正常 —— 这是「安静地不做事」最贵的形态。
#
# 这里只做一件事：每个作业跑完记一笔，再对比「按 cron 本该最近一次触发的时间」，
# 该跑没跑就补跑一次。刻意**只补 data_update / daily_pick** —— 它们决定
# 「明天有没有计划」；而 evolution 会写参数状态，绝不该被自动补跑。

JOB_RUNS_PATH: Path = Path(DATA_DIR) / "job_runs.json"

# 补跑范围：只补决定「明天有没有计划」的两个。**顺序即执行顺序** ——
# 行情没更新就去选股，等于用旧数据选今天的票。
CATCH_UP_JOBS: tuple = ("data_update", "daily_pick")

# 超过这么久就不再补。隔了一整天还去补跑一个「收盘后」作业，得到的计划属于
# 后天，对今天毫无意义，只会白占 CPU —— 那时该由 TradingLoop._ensure_plan 兜底。
CATCH_UP_MAX_LAG_SEC: int = 12 * 3600

# 反推 cron 触发时刻的回溯上限。只给枚举一个起点，
# 真正的时效判断由 CATCH_UP_MAX_LAG_SEC 负责。
_CATCH_UP_LOOKBACK_DAYS: int = 7


def _job_runs_load() -> Dict[str, dict]:
    """读作业记账。任何异常都当空账 —— 记账坏了不该阻断作业本身。"""
    try:
        data = json.loads(Path(JOB_RUNS_PATH).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _job_runs_mark(job_id: str, at: Optional[datetime] = None) -> None:
    """记一笔「``job_id`` 在 at 跑完了」。原子替换落盘，避免读到半截 JSON。"""
    at = at or datetime.now()
    try:
        data = _job_runs_load()
        data[job_id] = {"at": at.isoformat(timespec="seconds")}
        p = Path(JOB_RUNS_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"作业记账写入失败（已忽略）: {e}")


def _parse_iso(s) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s))
    except Exception:  # noqa: BLE001
        return None


def _tracked(job_id: str):
    """给作业加记账：跑完（且不是「被别人占着而跳过」）就记下时间。

    为什么必须记在**作业函数里**而不是注册处：``run_all_once`` 和补跑本身都
    直接调作业函数，绕开了 APScheduler 的注册包装。记在函数里才覆盖全部路径。
    """

    def deco(fn):
        def wrapper(*a, **k):
            res = fn(*a, **k)
            try:
                # 「被别的进程占着」说明它正在跑，由那个进程自己记账；
                # 这里记了等于把「跳过」当成「跑过」，会让真正的漏跑被掩盖。
                if not (isinstance(res, dict) and res.get("status") == "skipped"):
                    _job_runs_mark(job_id)
            except Exception:  # noqa: BLE001
                pass
            return res

        wrapper.__name__ = getattr(fn, "__name__", f"job_{job_id}")
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return deco


def last_fire_time(cron: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """cron 在 ``now``（含）之前**最近一次应当触发**的本地时间。

    刻意借 APScheduler 自己的 ``CronTrigger`` 反推，而不是另写一个 cron 解析器 ——
    两套解析器迟早分叉，那时「补跑判断」和「真实调度」说的就不是同一件事了。
    """
    trig = _make_trigger(cron)
    if trig is None:
        return None
    naive_now = now or datetime.now()
    aware_now = naive_now if naive_now.tzinfo else naive_now.astimezone()
    cursor = aware_now - timedelta(days=_CATCH_UP_LOOKBACK_DAYS)
    prev = None
    for _ in range(4000):          # 上限只为防脏 cron 死循环
        nxt = trig.get_next_fire_time(prev, cursor)
        if nxt is None or nxt > aware_now:
            break
        prev, cursor = nxt, nxt
    return prev.astimezone().replace(tzinfo=None) if prev is not None else None


_CRON_ATTR = {"data_update": "data_update_cron", "daily_pick": "daily_pick_cron"}

# 所有带记账的作业（与 @_tracked 的 id 一一对应）。只用于首次建立基准。
_ALL_TRACKED = ("data_update", "daily_pick", "evolution",
                "rolling_ic", "industry_map", "drift_retrain")


def _job_runs_bootstrap(now: Optional[datetime] = None) -> bool:
    """首次记账：``job_runs.json`` 还不存在时，把**现在**当作基准。

    不做这一步的话，新部署的机器永远「无记账记录」→ 永远不补跑，
    等于机制从没生效过；而如果反过来把历史全当漏跑，部署当天就会白跑一遍
    全池选股。

    只在文件**不存在**时动手。已有文件绝不动 —— 否则每次启动都等于
    「宣布所有作业刚跑过」，会把真正的漏跑抹掉，那正是这个机制要防的事。
    """
    p = Path(JOB_RUNS_PATH)
    try:
        if p.exists():
            return False
    except Exception:  # noqa: BLE001
        return False
    at = now or datetime.now()
    stamp = at.isoformat(timespec="seconds")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({jid: {"at": stamp, "baseline": True} for jid in _ALL_TRACKED},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, p)
        logger.info(f"🕒 作业记账首次建立（基准时间 {stamp}）：从此刻起算，不补历史窗口")
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug(f"作业记账初始化失败（已忽略）: {e}")
        return False


def _catch_up_crons() -> Dict[str, str]:
    s = app_config.scheduler
    return {j: str(getattr(s, _CRON_ATTR[j], "") or "") for j in CATCH_UP_JOBS}


def _market_window(now: Optional[datetime] = None) -> bool:
    """现在是否处于「别来抢 CPU」的时段（交易日 09:00–15:30）。

    用整个**盘中时段**而不只是连续竞价：补跑一次要扫全池、占满 CPU 好几分钟，
    09:20 把机器打满和 09:35 打满对开盘那几笔单的伤害是一样的。
    """
    now = now or datetime.now()
    try:
        from src.trading.calendar import is_trading_day

        if not is_trading_day(now):
            return False
    except Exception:  # noqa: BLE001
        return False
    return (9, 0) <= (now.hour, now.minute) <= (15, 30)


def missed_jobs(now: Optional[datetime] = None,
                respect_trading_hours: bool = True) -> List[dict]:
    """哪些作业「本该跑、但没跑」。**纯判断，不执行任何作业。**

    ``respect_trading_hours``：盘中不补跑。补跑要扫全池、占满 CPU，跟盘中闭环
    抢资源；而且盘中的计划已由 ``TradingLoop._ensure_plan`` 兜底补建。
    判断用的是**传入的 now**，不是真实时钟 —— 否则这个函数在测试里是薛定谔的。
    """
    now = now or datetime.now()
    if respect_trading_hours and _market_window(now):
        logger.info("盘中不补跑收盘后作业（避免与闭环抢 CPU），收盘后再补")
        return []

    runs = _job_runs_load()
    out: List[dict] = []
    for jid, cron in _catch_up_crons().items():
        if not cron:
            continue
        fire = last_fire_time(cron, now)
        if fire is None:
            continue
        lag = (now - fire).total_seconds()
        if lag < 0 or lag > CATCH_UP_MAX_LAG_SEC:
            continue
        rec = runs.get(jid) or {}
        at = _parse_iso(rec.get("at"))
        if at is None:
            # 没有记账 = 无法判断是否真漏（也可能只是刚部署、从没记过账）。
            # 宁可少补，也不要每次启动都白跑一遍全池选股。
            logger.info(f"作业 {jid} 无记账记录，跳过补跑（无法判断是否真的漏了）")
            continue
        if at >= fire:
            continue
        out.append({"job": jid, "missed_at": fire.isoformat(timespec="seconds"),
                    "lag_sec": int(lag), "last_run": rec.get("at")})
    return out


def catch_up_missed(now: Optional[datetime] = None, dry_run: bool = False,
                    runner=None) -> List[dict]:
    """把错过的作业补跑一遍（顺序见 ``CATCH_UP_JOBS``）。返回每次补跑的结果。

    ``runner``：``runner(job_id) -> dict``，供测试注入；默认按名字取 ``job_<id>``。
    ``dry_run``：只报告会补哪些，不真跑。
    """
    if _job_runs_bootstrap(now):
        return []                  # 刚建立基准，没有「错过」可言
    todo = missed_jobs(now)
    if not todo:
        return []
    results: List[dict] = []
    for item in todo:
        jid = item["job"]
        logger.warning(
            f"⏰ 补跑错过的作业 {jid}：本该在 {item['missed_at']} 触发，"
            f"已晚 {item['lag_sec'] // 60} 分钟（上次成功 {item['last_run']}）"
        )
        if dry_run:
            results.append({**item, "status": "would_run"})
            continue
        try:
            fn = runner(jid) if runner is not None else globals().get(f"job_{jid}")
            if fn is None:
                results.append({**item, "status": "no_runner"})
                continue
            res = fn()
            results.append({**item, "status": (res or {}).get("status", "ok"),
                            "result": res})
        except Exception as e:  # noqa: BLE001
            logger.error(f"补跑 {jid} 失败: {e}\n{traceback.format_exc()}")
            results.append({**item, "status": "error", "error": str(e)})
    return results


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


@_tracked("industry_map")
def job_industry_map() -> Dict:
    """行业映射补齐（带跨进程租约，每小时一次）。"""
    with _job_lease("industry_map", ttl_sec=1800) as ok:
        if not ok:
            return _skipped("industry_map", 1800)
        return _job_industry_map_impl()


@_tracked("data_update")
def job_data_update() -> Dict:
    """数据更新（带跨进程租约）。"""
    with _job_lease("data_update", ttl_sec=1800) as ok:
        if not ok:
            return _skipped("data_update", 1800)
        return _job_data_update_impl()


@_tracked("daily_pick")
def job_daily_pick(dry_run: bool = False) -> Dict:
    """每日推荐 + 交易计划（带跨进程租约）。dry_run 不写报告。"""
    with _job_lease("daily_pick", ttl_sec=3600) as ok:
        if not ok:
            return _skipped("daily_pick", 3600)
        return _job_daily_pick_impl(dry_run=dry_run)


@_tracked("evolution")
def job_evolution() -> Dict:
    """回测验证 + 自我进化（带跨进程租约）。"""
    with _job_lease("evolution", ttl_sec=3600) as ok:
        if not ok:
            return _skipped("evolution", 3600)
        return _job_evolution_impl()


@_tracked("rolling_ic")
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


@_tracked("drift_retrain")
def job_drift_retrain() -> Dict:
    """T4 漂移触发重训（**默认关闭**，受 ``scheduler.drift_retrain_enabled`` 控制）。

    与 ``job_evolution`` 的分工：进化作业是「定期搜参数/重训」的常客，
    本作业是「**只在检测到显著漂移时**才动模型」的哨兵。它不做任何落地——
    产出永远是待批准候选，人工 ``evolve.py --approve/--apply`` 后才生效。
    """
    if not getattr(app_config.scheduler, "drift_retrain_enabled", False):
        return {"job": "drift_retrain", "status": "disabled",
                "reason": "scheduler.drift_retrain_enabled=False（默认关闭）",
                "at": datetime.now().isoformat(timespec="seconds")}
    with _job_lease("drift_retrain", ttl_sec=7200) as ok:
        if not ok:
            return _skipped("drift_retrain", 7200)
        return _job_drift_retrain_impl()


def _job_drift_retrain_impl() -> Dict:
    """跑 ``scripts/drift_retrain.py --run``（子进程）。

    为什么用子进程而不是在调度进程内直接调：重训是会吃掉几十分钟 CPU/内存的
    长任务，放在 APScheduler 线程里会把整个调度进程一起拖住（其他作业到点不触发）。
    子进程还能给出硬超时，超时即杀，不留僵尸。
    """
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "drift_retrain.py"
    if not script.exists():
        return {"job": "drift_retrain", "status": "error",
                "error": f"脚本不存在: {script}",
                "at": datetime.now().isoformat(timespec="seconds")}
    timeout = int(getattr(app_config.scheduler, "drift_retrain_timeout_min", 150)) * 60
    try:
        cp = subprocess.run(
            [sys.executable, str(script), "--run", "--json"],
            cwd=str(root), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        out = (cp.stdout or "").strip()
        res: Dict = {"job": "drift_retrain", "status": "ok" if cp.returncode == 0 else "error",
                     "returncode": cp.returncode,
                     "at": datetime.now().isoformat(timespec="seconds")}
        try:
            # loguru 走 stderr，stdout 只有 --json 的那一个 JSON 对象
            res.update(json.loads(out) if out.startswith("{") else {"raw": out[-2000:]})
        except Exception:  # noqa: BLE001
            res["raw"] = out[-2000:]
        if cp.returncode != 0:
            res["stderr"] = (cp.stderr or "")[-2000:]
        return res
    except subprocess.TimeoutExpired:
        logger.error(f"漂移重训超时（{timeout // 60} 分钟）")
        return {"job": "drift_retrain", "status": "timeout", "timeout_min": timeout // 60,
                "at": datetime.now().isoformat(timespec="seconds")}
    except Exception as e:  # noqa: BLE001
        logger.exception("漂移重训作业失败")
        return {"job": "drift_retrain", "status": "error", "error": str(e),
                "at": datetime.now().isoformat(timespec="seconds")}


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
        # T4 漂移哨兵：默认关闭，只有显式打开 config.scheduler.drift_retrain_enabled
        # 才注册（「没打开却挂着作业」会让人误以为它在保护模型）。
        if getattr(app_config.scheduler, "drift_retrain_enabled", False):
            jobs.append(("drift_retrain",
                         getattr(app_config.scheduler, "drift_retrain_cron",
                                 "30 19 * * mon-fri"),
                         job_drift_retrain))
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

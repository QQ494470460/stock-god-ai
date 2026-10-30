# -*- coding: utf-8 -*-
"""收盘后自动化作业常驻进程 —— 让选股/进化不再依赖「看板开着」。

背景
----
三个收盘后作业（数据更新 18:30 / 每日推荐 18:45 / 进化验证 19:00）
原本只活在 **Streamlit 看板进程**里：``app.py`` 在脚本顶层调
``_ensure_scheduler_running()``，而那段代码只有在**有人打开看板**时才执行。
后果是：看板一关，第二天就没有推荐 → 交易计划为空 → 交易闭环无事可做，
整条链静默断掉，且日志里看不出任何异常（只是没有新报告而已）。

本进程把它们搬到独立的常驻进程里，配 Windows 计划任务做到真正无人值守。

如何与看板共存
--------------
``auto_scheduler`` 内部有跨进程锁 ``data/.scheduler.lock``：
本进程先抢到，看板再调 ``start_scheduler()`` 时会主动退让，看板状态页
照常显示「🟢 运行中」（``is_running()`` 会探锁，不只报本进程），互不打架。

用法
----
    py -3.12 scripts/scheduler_daemon.py            # 常驻（配计划任务用这个）
    py -3.12 scripts/scheduler_daemon.py --once     # 起好调度器并打印下次作业时间后退出

退出码：
    0  优雅退出（收到 Ctrl+C / SIGTERM）
    2  前置校验失败（调度未启用 / APScheduler 未安装）
    3  已有另一个常驻调度进程在跑（单实例锁被占用）
    1  运行中异常
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _force_utf8_stdio() -> None:
    """把 stdout/stderr 就地改成 UTF-8 —— 无头启动时**必须先跑这个**。

    与 ``run_loop.py`` 同一个坑：启动器用 ``pythonw ... >> log 2>&1`` 起常驻
    进程，重定向后的 stdout 默认按 OEM 码页（简中 cp936/GBK）编码，日志里
    一个 emoji 就抛 ``UnicodeEncodeError`` → 进程秒退 → 启动器重试若干次后
    放弃 → **当天的选股与进化全部没跑**，而表面上只是"没有新报告"。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001  老版本/已被替换的流，忽略
            pass


_force_utf8_stdio()

try:  # 与 run_loop.py 一致：先导入 config.settings 以注入外部 libs 路径
    import config.settings  # noqa: F401
except Exception:  # noqa: BLE001
    pass

from loguru import logger  # noqa: E402

_HEARTBEAT_INTERVAL = 30.0     # 心跳写入间隔（秒）
_REACQUIRE_INTERVAL = 300.0    # 没抢到调度器锁时的重试间隔（秒）
_TICK = 5.0                    # 主循环步长（秒）

_stop = False


def _on_signal(signum, _frame):  # pragma: no cover - 信号路径
    global _stop
    _stop = True
    try:
        logger.info(f"🛑 收到信号 {signum}，准备优雅退出…")
    except Exception:  # noqa: BLE001
        pass


def _heartbeat_path() -> Path:
    from config.settings import DATA_DIR
    return Path(DATA_DIR) / "scheduler_heartbeat.json"


def _write_heartbeat(sched, holding: bool, lock_path: Path) -> None:
    """把「常驻调度进程还活着 + 下次作业时间」落盘。

    为什么值得单独一个心跳：闭环有 ``loop_heartbeat.json``，看板据此判断
    「闭环是不是死了」。但**收盘后作业**此前没有任何存活证据 ——
    它挂了的表现只是「今天没有新报告」，极其容易被当成"今天没行情"。
    """
    jobs = []
    if holding and sched is not None:
        try:
            for j in sched.get_jobs():
                nxt = getattr(j, "next_run_time", None)
                jobs.append({
                    "id": j.id,
                    "next_run": nxt.strftime("%Y-%m-%d %H:%M:%S") if nxt else None,
                })
        except Exception:  # noqa: BLE001
            pass
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "holding": bool(holding),
        "enabled": bool(_cfg_enabled()),
        "jobs": jobs,
        "lock": str(lock_path),
    }
    try:
        p = _heartbeat_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(_dump(payload), encoding="utf-8")
        os.replace(tmp, p)          # 原子替换：看板不会读到写了一半的 JSON
    except Exception as e:  # noqa: BLE001
        logger.debug(f"调度心跳写入失败: {e}")


def _dump(payload) -> str:
    import json
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _cfg_enabled() -> bool:
    try:
        from config.settings import config
        return bool(config.scheduler.enabled)
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="收盘后自动化作业常驻进程")
    ap.add_argument("--once", action="store_true",
                    help="起好调度器、打印下次作业时间后立刻退出（自检用）")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    try:
        signal.signal(signal.SIGTERM, _on_signal)
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    # ---- 单实例闸门（必须在起 APScheduler 之前）----
    # 两个常驻调度进程会把 18:45 的选股跑两遍，且各自写一份 daily_reports。
    from config.settings import DATA_DIR
    from src.core.singleton import LOCK_DISABLED, acquire_lock, release_lock
    lock_path = Path(DATA_DIR) / ".scheduler_daemon.lock"
    fd = acquire_lock(lock_path)
    if fd is None:
        print("❌ 已有另一个常驻调度进程在运行（单实例锁被占用），本次退出。",
              file=sys.stderr)
        return 3
    own_fd = fd if fd != LOCK_DISABLED else None

    from src.scheduler.auto_scheduler import (
        is_running, start_scheduler, stop_scheduler,
    )
    # 闭环存活看门狗：本进程是唯一"开机就在"的常驻体，顺手负责盯 loop 心跳。
    # 2026-10-29 的教训 —— 闭环死了/没起来，唯一表现是"今天没成交"，与"今天
    # 没信号"完全同形，人工根本发现不了，一下午的止损监控就这么丢了。
    from src.scheduler.loop_watchdog import LoopWatchdog

    sched = None
    holding = False
    last_try = 0.0
    last_hb = 0.0
    watchdog = LoopWatchdog(interval=60.0)
    logger.info(f"🕒 常驻调度进程启动（pid={os.getpid()}）")

    try:
        while not _stop:
            now = time.monotonic()

            # 还没抢到调度器所有权（可能看板正持有）：每 5 分钟再试一次。
            # 这样看板一关，本进程 5 分钟内自动接手，作业不会漏。
            if not holding and (last_try == 0.0 or now - last_try >= _REACQUIRE_INTERVAL):
                last_try = now
                if _cfg_enabled():
                    got = start_scheduler()
                    if got is not None:
                        sched, holding = got, True
                        logger.info("✅ 已接管收盘后自动化调度")
                    elif args.once:
                        # 明确区分两种「没起成」：配置没开 vs 锁被别人拿着，
                        # 否则值班的人只会看到一句「未运行」，无从下手。
                        print("⚠️ 未能接管调度器：配置未启用，或 data/.scheduler.lock "
                              "已被其他进程持有（通常是看板）。")
                        return 2
                    else:
                        logger.info("⏳ 调度器暂由其他进程持有，5 分钟后再试接管")
                elif args.once:
                    print("⚠️ 调度器未启用（config.scheduler.enabled=False）")
                    return 2

            if now - last_hb >= _HEARTBEAT_INTERVAL:
                last_hb = now
                _write_heartbeat(sched, holding, lock_path)

            # 闭环存活检查（自己节流到 60s 一次，且只在交易时段内动作）。
            # 放在心跳之后：调度心跳先落盘，再去看别人的心跳，顺序颠倒会让
            # 看板在检查瞬间读到旧的调度心跳。
            try:
                res = watchdog.tick(now)
                if res and res.get("action") in ("alert", "restart"):
                    logger.warning(f"🩺 闭环看门狗：{res.get('detail')}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"闭环看门狗异常（已忽略）: {e}")

            if args.once:
                break
            time.sleep(_TICK)

        if args.once:
            _write_heartbeat(sched, holding, lock_path)
            if holding and sched is not None:
                for j in sched.get_jobs():
                    nxt = getattr(j, "next_run_time", None)
                    print(f"  {j.id:<14} 下次 {nxt.strftime('%Y-%m-%d %H:%M:%S') if nxt else '—'}")
            print(f"  holding={holding} running={is_running()}")
    except Exception as e:  # noqa: BLE001
        logger.error(f"常驻调度进程异常: {e}")
        return 1
    finally:
        # 「我是主动退出的」留痕：退出码 0 说明不了意图 —— 被任务管理器结束、
        # 会话被回收、系统关机，进程照样以 0 收场。而启动器一看到 0 就
        # 永久放弃，于是这个进程死到下次开机，表面上却看不出异常。
        # 只有收到 Ctrl+C / SIGTERM（_stop）才算主动，才落这个标记。
        if _stop:
            try:
                from config.settings import DATA_DIR as _dd
                (Path(_dd) / ".scheduler_daemon.stopped").write_text(
                    f"{datetime.now().isoformat(timespec='seconds')}\n",
                    encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        if holding:
            try:
                stop_scheduler()
            except Exception:  # noqa: BLE001
                pass
        release_lock(own_fd)
        logger.info("👋 常驻调度进程已退出")

    return 0


if __name__ == "__main__":
    sys.exit(main())

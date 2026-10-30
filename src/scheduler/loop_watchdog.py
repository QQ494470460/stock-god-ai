# -*- coding: utf-8 -*-
"""闭环存活看门狗 —— 交易时段内发现常驻闭环没跑就告警 / 重新拉起。

为什么需要
----------
之前不存在任何"闭环是否活着"的自动检查：进程死了的唯一表现是**没有新成交**，
而这跟"今天没信号"长得一模一样。2026-10-29 就吃过一次：机器 11:50 关机，
整个下午（13:00-15:00）没有任何止损监控，罗博特科从 -13% 一路穿到 -16.65%
无人处置，直到 20:20 开机后第一个 tick 才按收盘价补了一刀，比止损价多亏约
¥5,400。

判据为什么是心跳而不是进程
--------------------------
进程存在 ≠ 在干活（可能卡在等行情、卡在锁、或者睡在某个死循环里）。
``loop_heartbeat.json`` 由闭环每个 tick 写一次（30s），是唯一能证明"它真的
在转"的证据。所以这里只认心跳年龄，不认 pid 是否在 tasklist 里。

为什么不杀进程
--------------
"进程还在但心跳停了"可能是短暂的行情卡顿，也可能真的挂死 —— 两者从外面
分不出来。此时**直接杀掉重启**会把正在跑的止损状态、当日换手预算、待确认
委托一起丢掉，代价远大于收益。所以本模块只做两件事：**告警** + **在确认
没有任何闭环进程时尝试拉起启动器**。判定是否"没有进程"交给启动器自己的
单实例锁（``data/.run_loop.lock``）—— 重复拉起会被它以退出码 3 挡掉，安全。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------- 常量
STALE_SEC = 240.0          # 心跳超过 4 分钟没更新 = 闭环没在转（它 30s 写一次）
COOLDOWN_SEC = 600.0       # 两次"拉起启动器"之间至少间隔 10 分钟
WINDOW_START = dtime(9, 15)
WINDOW_END = dtime(15, 10)  # 收盘 15:00 后留 10 分钟给结算 tick

_DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    try:
        from config.settings import DATA_DIR
        return Path(DATA_DIR)
    except Exception:  # noqa: BLE001
        return _root() / "data"


def _hb_path() -> Path:
    return _data_dir() / "loop_heartbeat.json"


def _state_path() -> Path:
    return _data_dir() / ".watchdog_last_attempt"


def _log_path() -> Path:
    return _root() / "logs" / "watchdog.log"


# ---------------------------------------------------------------- 基础读
def heartbeat_age_sec(path: Optional[Path] = None,
                      now: Optional[datetime] = None) -> Optional[float]:
    """返回心跳年龄（秒）；文件缺失或格式坏了返回 None。"""
    p = Path(path) if path else _hb_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(str(raw["ts"]))
    except Exception:  # noqa: BLE001
        return None
    cur = now or datetime.now()
    return max(0.0, (cur - ts).total_seconds())


def in_monitoring_window(now: Optional[datetime] = None) -> bool:
    """只在工作日的交易时段内行动 —— 夜里没人交易，重启毫无意义还扰日志。"""
    cur = now or datetime.now()
    if cur.weekday() >= 5:          # 周六周日
        return False
    return WINDOW_START <= cur.time() <= WINDOW_END


def _read_last_attempt(path: Path) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(path.read_text(encoding="utf-8").strip())
    except Exception:  # noqa: BLE001
        return None


def _write_last_attempt(path: Path, when: datetime) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(when.isoformat(timespec="seconds") + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _clear_last_attempt(path: Path) -> None:
    try:
        path.unlink()
    except Exception:  # noqa: BLE001
        pass


def _append_log(msg: str, path: Optional[Path] = None) -> None:
    try:
        p = Path(path) if path else _log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- 拉起
def spawn_launcher(root: Optional[Path] = None) -> bool:
    """后台拉起 ``scripts/run_loop_daemon.bat``。

    独立进程组 + DETACHED，避免它跟着常驻调度进程一起退出；启动器自身有
    「永不放弃」的重试阶梯，并且闭环有单实例锁，重复拉起会被挡（退出码 3），
    所以这里不需要先探测进程是否存在 —— 探测反而容易误判。
    """
    base = Path(root) if root else _root()
    bat = base / "scripts" / "run_loop_daemon.bat"
    if not bat.exists():
        _append_log(f"ERROR 启动器不存在: {bat}")
        return False
    if os.name != "nt":
        _append_log("WARN 非 Windows 环境，跳过拉起启动器")
        return False
    try:
        subprocess.Popen(
            ["cmd.exe", "/c", str(bat)],
            cwd=str(base),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_DETACHED,
            close_fds=True,
        )
        return True
    except Exception as e:  # noqa: BLE001
        _append_log(f"ERROR 拉起启动器失败: {e}")
        return False


# ---------------------------------------------------------------- 主检查
def check(
    *,
    heartbeat_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    now: Optional[datetime] = None,
    stale_sec: float = STALE_SEC,
    cooldown_sec: float = COOLDOWN_SEC,
    restart: bool = True,
    spawn: Optional[Callable[[], bool]] = None,
    window: Optional[bool] = None,
) -> dict:
    """做一次存活检查。返回可序列化的结果字典（便于日志/自检/测试断言）。

    ``window`` 显式给定时不看时间（测试用）；否则自动判定是否在监控时段内。
    """
    cur = now or datetime.now()
    hb = Path(heartbeat_path) if heartbeat_path else _hb_path()
    st = Path(state_path) if state_path else _state_path()
    lg = Path(log_path) if log_path else None

    age = heartbeat_age_sec(hb, cur)
    ok = age is not None and age <= stale_sec
    active = in_monitoring_window(cur) if window is None else bool(window)

    result = {
        "at": cur.isoformat(timespec="seconds"),
        "age_sec": None if age is None else round(age, 1),
        "ok": bool(ok),
        "in_window": bool(active),
        "action": "none",
        "detail": "",
    }

    if ok:
        # 恢复了就把"上次拉起"的冷却记录清掉，下一次事故可以立刻处置。
        _clear_last_attempt(st)
        result["detail"] = f"闭环存活（心跳 {age:.0f}s）"
        return result

    if not active:
        result["detail"] = ("非监控时段" if age is not None else "心跳文件缺失")
        return result

    reason = "心跳文件缺失/无法解析" if age is None else f"心跳已停 {age:.0f}s"
    result["action"] = "alert"
    result["detail"] = f"⚠️ 闭环疑似停摆：{reason}"
    _append_log(f"ALERT {result['detail']}（阈值 {stale_sec:.0f}s）", lg)
    try:
        from loguru import logger
        logger.error(f"🩺 闭环看门狗：{result['detail']}")
    except Exception:  # noqa: BLE001
        pass

    if not restart:
        return result

    last = _read_last_attempt(st)
    if last is not None and (cur - last).total_seconds() < cooldown_sec:
        result["detail"] += "｜冷却中，暂不重复拉起"
        return result

    _write_last_attempt(st, cur)
    spawner = spawn or spawn_launcher
    started = False
    try:
        started = bool(spawner())
    except Exception as e:  # noqa: BLE001
        _append_log(f"ERROR 拉起异常: {e}", lg)
    if started:
        result["action"] = "restart"
        result["detail"] += "｜已拉起启动器"
        _append_log("ACTION 已拉起 scripts/run_loop_daemon.bat", lg)
    else:
        result["detail"] += "｜拉起失败（见 logs/watchdog.log）"
    return result


# ---------------------------------------------------------------- 常驻接入
class LoopWatchdog:
    """给 ``scheduler_daemon`` 主循环用的节流包装（每 interval 秒才真检查一次）。"""

    def __init__(self, interval: float = 60.0, **kwargs):
        self.interval = float(interval)
        self._kwargs = kwargs
        # 用 -inf 而不是 0：第一次调用必须无条件跑一次（0 会让"启动即到点"
        # 这件事依赖 monotonic 的绝对值，换个起点就不成立了）。
        self._last = float("-inf")

    def due(self, monotonic_now: float) -> bool:
        return (monotonic_now - self._last) >= self.interval

    def tick(self, monotonic_now: float) -> Optional[dict]:
        if not self.due(monotonic_now):
            return None
        self._last = monotonic_now
        try:
            return check(**self._kwargs)
        except Exception as e:  # noqa: BLE001
            _append_log(f"ERROR 看门狗自身异常（已忽略）: {e}")
            return None


def main(argv: Optional[list] = None) -> int:
    """命令行自检：``python -m src.scheduler.loop_watchdog [--no-restart]``。"""
    import argparse

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(description="闭环存活看门狗（单次检查）")
    ap.add_argument("--no-restart", action="store_true", help="只告警，不拉起启动器")
    ap.add_argument("--force-window", action="store_true", help="忽略时段限制（手工排查用）")
    args = ap.parse_args(argv)

    res = check(restart=not args.no_restart,
                window=True if args.force_window else None)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

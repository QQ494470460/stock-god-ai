r"""收盘后调度常驻进程的启动器回归测试。

与 ``test_run_loop_launcher.py`` 同一套纪律，别因为是新文件就松掉：

1. 纯 ASCII + 全 CRLF（cmd.exe 用 GBK 解码 + 对换行敏感）；
2. 用探针的**退出码**判重，而不是 ERRORLEVEL（重定向失败不改 ERRORLEVEL）；
3. 退出码 0 必须配 stop marker 才允许放弃自启（被外部杀掉的进程同样以 0 收场）。

还有一条本文件特有的：启动器必须用**自己的**锁（.scheduler_daemon.lock），
不能复用 APScheduler 的归属锁 .scheduler.lock —— 否则看板的 is_running()
会因为常驻进程占着归属锁而永远显示"运行中"，哪怕 APScheduler 根本没起来。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "scripts" / "scheduler_daemon.bat"


def _text():
    return BAT.read_text(encoding="ascii")


def test_bat_exists():
    assert BAT.is_file(), f"缺少启动器 {BAT}"


def test_bat_is_pure_ascii():
    bad = [(i, b) for i, b in enumerate(BAT.read_bytes()) if b > 127]
    assert not bad, (
        f"scheduler_daemon.bat 含非 ASCII 字节（首个偏移 {bad[0][0]}）："
        "cmd.exe 会用 GBK 解码导致行被切碎、if 块错乱。"
    )


def test_bat_uses_crlf_only():
    raw = BAT.read_bytes()
    lone_lf = raw.count(b"\n") - raw.count(b"\r\n")
    assert lone_lf == 0, f"有 {lone_lf} 个裸 LF 换行，cmd 解析器会失效"
    assert raw.count(b"\r\n") > 10


def test_bat_has_no_bom():
    assert not BAT.read_bytes().startswith(b"\xef\xbb\xbf")


def test_bat_uses_its_own_lock_probe():
    text = _text()
    assert "loop_lock_probe.py --name .scheduler_daemon.lock" in text, \
        "必须用 .scheduler_daemon.lock 自己的探针，别和闭环共用一把锁"
    idx = text.index("loop_lock_probe.py")
    assert "errorlevel 1" in text[idx:], "探针之后必须按退出码分支"


def test_bat_exits_on_rc3():
    text = _text()
    assert '"!rc!"=="3"' in text
    assert "exit /b 0" in text


def test_bat_restarts_forever_on_abnormal_exit():
    """绝不能像早期版本那样「5 次失败就放弃」—— 那会让进程死到下次开机。"""
    text = _text()
    assert "giving up" not in text.lower(), "启动器不允许放弃自愈"
    assert "ping -n" in text and "goto loop" in text


def test_bat_clears_stop_marker_before_loop():
    text = _text()
    assert 'if exist "%MARK%" del' in text
    assert text.index('if exist "%MARK%" del') < text.index(":loop")


def test_bat_requires_marker_before_giving_up():
    text = _text()
    block = text[text.index('if "!rc!"=="0"'):]
    block = block[:block.index('if "!rc!"=="3"')]
    assert 'if exist "%MARK%"' in block
    assert block.index('if exist "%MARK%"') < block.index("exit /b 0")


def test_bat_uses_separate_logs_from_the_loop():
    """两个常驻进程的日志必须分开。

    共用一个文件时，先跑的那个会一直持有句柄，后一个的 ``>>`` 重定向直接失败，
    而**重定向失败不会反映到 ERRORLEVEL** —— 于是它连启动都启动不了，
    还看不到任何报错。
    """
    text = _text()
    assert "scheduler_launcher.log" in text
    assert "scheduler_daemon.log" in text
    assert "run_loop.log" not in text, "别和闭环共用日志文件"


def test_daemon_writes_marker_only_on_signal_stop():
    """被动终止绝不能留标记，否则启动器会以为它自己想停。"""
    src = (ROOT / "scripts" / "scheduler_daemon.py").read_text(encoding="utf-8")
    lines = [ln for ln in src.splitlines() if ".scheduler_daemon.stopped" in ln]
    assert len(lines) == 1, f"期望只有一处写 stop marker，实际 {len(lines)}"
    idx = src.index(".scheduler_daemon.stopped")
    assert "if _stop:" in src[max(0, idx - 300):idx], \
        "写标记必须被「收到退出信号」条件包住"

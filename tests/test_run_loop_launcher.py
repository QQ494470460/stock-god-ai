"""启动器与单实例探针的回归测试。

这些用例存在的理由都是「真出过事」：

1. ``run_loop_daemon.bat`` 曾经是 **UTF-8 中文 + LF 换行**。cmd.exe 用 OEM
   码页（简中 936/GBK）解码批处理且解析器对换行敏感，于是 if 块和 goto 标签
   被切碎，启动器直接死于 ``'M' 不是内部或外部命令`` —— 而「开机自启」这条
   链路因此从未真正跑通过。用例把「纯 ASCII + 全 CRLF」钉成硬约束。

2. 启动器曾经靠 ``>> "%LOG%" echo.`` 之后看 ``ERRORLEVEL`` 判断「是否已有
   实例」。实测**重定向失败不会改 ERRORLEVEL**（始终 0），探测形同虚设，
   于是走进 5 次重启空转。现在改用 ``scripts/loop_lock_probe.py`` 的退出码。

3. 探针曾经用「``os.open(O_RDWR)`` 是否成功」判占用，但 Windows 上字节区间
   锁**不拦 open**、只拦读写，结果永远判成「空闲」。必须真的去
   ``msvcrt.locking`` 抢一下。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "scripts" / "run_loop_daemon.bat"
PROBE_PY = ROOT / "scripts" / "loop_lock_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("_loop_lock_probe", PROBE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# 1) 批处理编码/换行：纯 ASCII + 全 CRLF
# --------------------------------------------------------------------------


def test_daemon_bat_exists():
    assert BAT.is_file(), f"缺少启动器 {BAT}"


def test_daemon_bat_is_pure_ascii():
    raw = BAT.read_bytes()
    bad = [(i, b) for i, b in enumerate(raw) if b > 127]
    assert not bad, (
        f"run_loop_daemon.bat 含非 ASCII 字节（首个偏移 {bad[0][0]}）："
        "cmd.exe 会用 GBK 解码导致行被切碎、if 块错乱。批处理必须是纯 ASCII。"
    )


def test_daemon_bat_uses_crlf_only():
    raw = BAT.read_bytes()
    lone_lf = raw.count(b"\n") - raw.count(b"\r\n")
    assert lone_lf == 0, (
        f"run_loop_daemon.bat 有 {lone_lf} 个裸 LF 换行；"
        "cmd.exe 解析器要求 CRLF，否则 goto 标签/多行 if 块会失效。"
    )
    assert raw.count(b"\r\n") > 10, "批处理行数异常少，可能被写坏了"


def test_daemon_bat_has_no_bom():
    assert not BAT.read_bytes().startswith(b"\xef\xbb\xbf"), "批处理不该有 UTF-8 BOM"


# --------------------------------------------------------------------------
# 2) 启动器要真的用探针 + 尊重 rc=3
# --------------------------------------------------------------------------


def test_daemon_bat_uses_lock_probe_exit_code():
    text = BAT.read_text(encoding="ascii")
    assert "loop_lock_probe.py" in text, "启动器没有调用锁探针"
    # 探针之后必须紧跟 errorlevel 分支
    idx = text.index("loop_lock_probe.py")
    tail = text[idx:]
    assert "errorlevel 1" in tail, "调用探针后没有按 errorlevel 分支处理"


def test_daemon_bat_exits_on_rc3():
    text = BAT.read_text(encoding="ascii")
    assert '"!rc!"=="3"' in text, "启动器没有识别 run_loop.py 的退出码 3（已有实例）"
    assert "exit /b 0" in text


def test_daemon_bat_does_not_use_timeout_for_delay():
    # timeout 在 stdin 被重定向（无头启动）时会直接报错退出，改用 ping 延时
    text = BAT.read_text(encoding="ascii")
    assert "timeout /t" not in text, "无头环境不能用 timeout 做延时，请用 ping"
    assert "ping -n" in text


# --------------------------------------------------------------------------
# 3) 探针语义：空闲 = 0，被占用 = 1
# --------------------------------------------------------------------------


def test_probe_reports_free_lock(tmp_path):
    mod = _load_probe()
    lock = tmp_path / ".run_loop.lock"
    assert mod.probe(lock) == 0
    assert lock.exists(), "探针应当（在空闲时）创建锁文件"


def test_probe_reports_busy_when_lock_held(tmp_path):
    """真的去抢锁：占用期间探针必须返回 1。"""
    mod = _load_probe()
    lock = tmp_path / ".run_loop.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o644)
    f = os.fdopen(fd, "r+", encoding="utf-8")
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        pytest.skip("本平台不允许同进程二次加锁，OS 语义需跨进程验证")

    try:
        assert mod.probe(lock) == 1, "有人持锁时探针必须报 1（fail-closed）"
    finally:
        f.close()

    assert mod.probe(lock) == 0, "锁释放后探针必须回到 0"


def test_probe_unusable_path_fails_open(tmp_path):
    """锁文件都建不出来时宁可能跑（与 run_loop.py 的 fail-open 分支一致）。"""
    mod = _load_probe()
    blocked = tmp_path / "nope" / ".run_loop.lock"
    blocked.parent.write_text("我是文件，不是目录", encoding="utf-8")
    assert mod.probe(blocked) == 0


def test_probe_matches_run_loop_semantics(tmp_path):
    """探针与 run_loop.acquire_single_instance 必须互补：探针 =1 ⟺ 主程序拒绝启动。"""
    run_loop = pytest.importorskip("scripts.run_loop")
    probe_mod = _load_probe()

    lock = tmp_path / ".run_loop.lock"
    assert probe_mod.probe(lock) == 0
    assert run_loop.acquire_single_instance(lock) is True, "空闲锁应当能被主程序拿到"

    # 主程序持锁期间，探针必须看到「被占用」
    assert probe_mod.probe(lock) == 1, "主程序持锁后探针仍报空闲 —— 探测通道失效了"

    run_loop.release_single_instance()
    assert probe_mod.probe(lock) == 0

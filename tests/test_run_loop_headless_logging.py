"""常驻闭环「无控制台 = 日志裸奔」回归测试。

2026-10-19 真事故：用 ``pythonw scripts/run_loop.py --no-notify`` 拉常驻进程时没做
重定向，pythonw 没有控制台 → ``sys.stdout`` / ``sys.stderr`` 都是 ``None`` →
``print()`` 变成**静默空操作**（不抛异常）、loguru 默认 sink 也指向 None。

结果是最难查的一种"假活"：心跳文件照常每 30 秒刷新，但日志文件一个字节都不增长，
闭环里发生什么完全查不到。

这里用子进程真实加载 ``scripts/run_loop.py``（模块级就会跑兜底逻辑），
断言两种启动方式下的行为。
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_loop.py"

# 用 importlib 手工装载，避免触发 __main__ 里的 main()（那会真的去连券商）
_LOADER = """
import importlib.util
spec = importlib.util.spec_from_file_location("run_loop_uut", r"{script}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
"""


def _run(child_code: str, log_path: Path) -> subprocess.CompletedProcess:
    """在子进程里真实装载 run_loop.py，日志路径强制指向 tmp（绝不碰生产日志）。"""
    env = dict(os.environ)
    env["RUN_LOOP_LOG"] = str(log_path)
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


def test_takes_over_logfile_when_stdout_is_none(tmp_path):
    """stdout/stderr 都是 None（pythonw 无控制台）→ 必须接管到日志文件。"""
    log = tmp_path / "run_loop.log"
    code = (
        "import sys\n"
        "sys.stdout = None\n"
        "sys.stderr = None\n"
        + _LOADER.format(script=SCRIPT)
        + "\nprint('TAIL-MARK')\n"
        "sys.stdout.flush()\n"
    )
    r = _run(code, log)
    assert r.returncode == 0, f"子进程失败: {r.stderr[-800:]}"

    text = log.read_text(encoding="utf-8", errors="replace")
    assert "日志已接管" in text, f"没有接管日志，文件内容={text!r}"
    # 关键：接管之后的 print 必须真的落盘（print 在 stdout=None 时是静默空操作）
    assert "TAIL-MARK" in text, f"接管后的 print 仍未落盘，文件内容={text!r}"


def test_no_takeover_when_stdout_is_valid(tmp_path):
    """stdout 有效（.bat 已 >> 重定向）→ 不能重复挂文件 sink，否则日志双写。"""
    log = tmp_path / "run_loop.log"
    code = (
        _LOADER.format(script=SCRIPT)
        + "\nprint('TOOK' if mod._headless_logfile is not None else 'NOTOOK')\n"
    )
    r = _run(code, log)
    assert r.returncode == 0, f"子进程失败: {r.stderr[-800:]}"
    assert "NOTOOK" in r.stdout, f"有效 stdout 下不该接管: {r.stdout!r}"
    assert not log.exists(), "有效 stdout 下不应创建日志文件（会双写重复行）"


def test_takeover_survives_emoji(tmp_path):
    """接管后的流必须是 UTF-8 + errors=replace —— emoji 不能把进程干掉。

    2026-10-18 的另一起事故就是 GBK 编码 emoji 抛 UnicodeEncodeError 导致
    闭环整天没跑，这里钉死：无控制台接管路径也不能死于编码。
    """
    log = tmp_path / "run_loop.log"
    code = (
        "import sys\n"
        "sys.stdout = None\n"
        "sys.stderr = None\n"
        + _LOADER.format(script=SCRIPT)
        + "\nprint('✅ 已连接 🛡️ 安全中枢 🧪 演练')\n"
        "sys.stdout.flush()\n"
    )
    r = _run(code, log)
    assert r.returncode == 0, f"emoji 让子进程挂了: {r.stderr[-800:]}"
    assert "✅ 已连接" in log.read_text(encoding="utf-8", errors="replace")


@pytest.mark.parametrize("drop", ["stdout", "stderr"])
def test_takes_over_when_only_one_stream_is_missing(tmp_path, drop):
    """只缺一个流（stdout 有 / stderr 没有）时也要接管缺失的那个。"""
    log = tmp_path / "run_loop.log"
    code = (
        "import sys\n"
        f"sys.{drop} = None\n"
        + _LOADER.format(script=SCRIPT)
        + "\nprint('TAKEOVER' if mod._headless_logfile is not None else 'NOTOOK')\n"
        f"sys.{drop}.write('VIA-{drop.upper()}\\n')\n"
        f"sys.{drop}.flush()\n"
    )
    r = _run(code, log)
    assert r.returncode == 0, f"子进程失败: {r.stderr[-800:]}"
    text = log.read_text(encoding="utf-8", errors="replace")
    # 缺 stdout 时 print 落文件，缺 stderr 时 print 仍走真 stdout —— 两处都要找
    assert "TAKEOVER" in (text + r.stdout), f"未接管: file={text!r} out={r.stdout!r}"
    # 关键：缺失的那个流被接管后，写它必须落盘
    assert f"VIA-{drop.upper()}" in text, f"sys.{drop} 未被接管: {text!r}"

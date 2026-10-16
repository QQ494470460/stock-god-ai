"""静态拦截「未定义名字」——NameError 类崩溃的防复发闸门。

背景（真实事故，同一天连炸两次）：
1. `app.py` 用了 `metric_card(...)` 渲染指标卡，但**从未从 `src.ui.common` 导入**，
   个股盯盘页每次渲染都抛 `NameError: name 'metric_card' is not defined`。
2. 顺带查出 `src/trading/autotrader.py:104` 用了 `Path` 却没导入，
   而该文件**没有** `from __future__ import annotations`，且这句被包在
   `except Exception: pass` 里 → NameError 被静默吞掉 →
   `watchlist.json` 的 15 只自选股从来没被加载，日志还打印「监控 0 只股票」。
   这正是本项目最典型的「静默失效」：不报错、不告警，只是那颗螺丝没在转。

为什么需要这道闸门：`app.py` 是**顶层脚本**（UI 代码直接写在模块级），
只有真正渲染到那一行才会炸，单元测试通常覆盖不到；而 py_compile 也抓不到
NameError。所以只能用静态分析兜底。

负向对照：把 `app.py` 里 `metric_card` 的导入删掉，
`test_ui_app_has_no_undefined_names` 必须立刻失败。
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _undefined_names(*targets: str) -> list:
    """返回 pyflakes 报出的所有 'undefined name' 条目。"""
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", *targets],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return [ln for ln in out.splitlines() if "undefined name" in ln]


@pytest.fixture(scope="module")
def pyflakes_available():
    pytest.importorskip("pyflakes", reason="pyflakes 未安装，跳过静态检查")


def test_ui_app_has_no_undefined_names(pyflakes_available):
    """app.py 不得有任何未定义名字（metric_card 就是这么漏的）。"""
    bad = _undefined_names(str(REPO_ROOT / "src" / "ui" / "app.py"))
    assert not bad, "app.py 存在未定义名字，页面渲染会 NameError：\n" + "\n".join(bad)


def test_trading_autotrader_has_no_undefined_names(pyflakes_available):
    """autotrader.py 不得有未定义名字（Path 曾被 except 静默吞掉）。"""
    bad = _undefined_names(str(REPO_ROOT / "src" / "trading" / "autotrader.py"))
    assert not bad, "autotrader.py 存在未定义名字：\n" + "\n".join(bad)


def test_whole_src_has_no_undefined_names(pyflakes_available):
    """整个 src/ 与 scripts/ 都不允许有未定义名字。"""
    bad = _undefined_names(str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts"))
    assert not bad, "存在未定义名字：\n" + "\n".join(bad)


def test_ui_helpers_used_by_app_are_importable():
    """页面用到的 UI 辅助函数必须真的能从 common 导入（不依赖静态分析）。"""
    from src.ui.common import metric_card, html_metric, calc_rsi

    assert callable(metric_card)
    assert callable(html_metric)
    html = html_metric("标签", "1.00", chg=1.5, chg_text="+1.50%")
    assert isinstance(html, str) and len(html) > 0

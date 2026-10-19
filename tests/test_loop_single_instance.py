# -*- coding: utf-8 -*-
"""闭环「全机唯一实例」的回归测试。

背景（2026-10-19 真实事故）
==========================
``data/loop_heartbeat.json`` 里的 pid 在 ``20508``（常驻
``scripts/run_loop.py``）和 ``21204``（看板进程里内嵌的闭环）之间来回跳 ——
两个实例在**交替 tick**。两个实例 = 对同一份交易计划各下一遍单，而且风控与
纸账户各有一份内存副本，对账必乱。

常驻脚本在启动时本来就抢了 OS 级文件锁，能挡住第二个**进程**；但内嵌闭环跑在
看板进程里、根本不走这段启动逻辑，于是形成了「看板点一次启动 + 开机自启又拉
一份」的双实例。修法是把同一把锁（``data/.run_loop.lock``）下沉到
``TradingLoop.start()`` 里，让两种启动方式共用。

这里钉死三件事：
  1) 锁本身的抢/放语义正确（占用时第二个实例必须被拒）；
  2) ``TradingLoop.start()`` 抢不到锁时**返回 False 而不是照常启动**；
  3) 看板内嵌闭环传的锁路径与常驻脚本是**同一个文件**。
"""

from pathlib import Path

from src.trading.loop import SingleInstanceLock

ROOT = Path(__file__).parent.parent


def test_acquire_release_roundtrip(tmp_path):
    """抢锁 → 放锁 → 别人能再抢到。"""
    p = tmp_path / ".run_loop.lock"
    a = SingleInstanceLock(p)
    assert a.acquire() is True, "空闲时应当能抢到锁"
    assert a.acquired is True
    assert p.exists(), "抢到锁后锁文件必须存在"

    a.release()
    assert a.acquired is False

    b = SingleInstanceLock(p)
    assert b.acquire() is True, "释放后另一个实例必须能抢到"
    b.release()


def test_second_instance_is_refused_while_held(tmp_path):
    """别人持锁时，第二个实例必须被拒 —— 这是防重复下单的唯一防线。"""
    p = tmp_path / ".run_loop.lock"
    holder = SingleInstanceLock(p)
    assert holder.acquire() is True
    try:
        other = SingleInstanceLock(p)
        assert other.acquire() is False, (
            "锁被占用时第二个实例居然抢到了 —— 双实例会重复下单"
        )
        assert other.acquired is False
    finally:
        holder.release()


def test_acquire_is_idempotent_for_same_instance(tmp_path):
    """同一实例重复 acquire 应当成功且不会把自己锁死。"""
    p = tmp_path / ".run_loop.lock"
    a = SingleInstanceLock(p)
    assert a.acquire() is True
    assert a.acquire() is True, "自己已持锁时重复 acquire 应当直接返回 True"
    a.release()


def test_start_returns_false_when_lock_is_held(tmp_path, monkeypatch):
    """``TradingLoop.start()`` 抢不到锁时必须返回 False 且不启线程。

    用一个``__new__``出来的空壳实例：构造完整的 TradingLoop 需要券商/安全中枢，
    而这里要验的只有「启停与锁的交互」这一条逻辑。
    """
    from src.trading import loop as loop_mod

    holder = SingleInstanceLock(tmp_path / ".run_loop.lock")
    assert holder.acquire() is True
    try:
        obj = loop_mod.TradingLoop.__new__(loop_mod.TradingLoop)
        obj._instance_lock = SingleInstanceLock(tmp_path / ".run_loop.lock")
        obj._stop_event = loop_mod.threading.Event()
        obj._thread = None          # is_running 是只读 property，由 _thread 推导
        obj.config = loop_mod.LoopConfig()
        obj._stop_event = loop_mod.threading.Event()
        assert obj.is_running is False

        assert obj.start() is False, "抢不到锁时必须拒绝启动"
        assert obj._thread is None, "被拒绝时绝不能留下后台线程"
    finally:
        holder.release()


def test_dashboard_loop_shares_the_same_lock_file():
    """看板内嵌闭环必须与常驻 run_loop.py 共用同一个锁文件。"""
    ui = (ROOT / "src" / "ui" / "trading_ui.py").read_text(encoding="utf-8")
    assert ".run_loop.lock" in ui, "看板没有指定单实例锁文件"
    assert "lock_path=_loop_lock_path()" in ui, "构建 TradingLoop 时没有传入锁路径"

    engine = (ROOT / "src" / "trading" / "loop.py").read_text(encoding="utf-8")
    assert "class SingleInstanceLock" in engine
    assert "def start(self, interval_seconds: Optional[float] = None) -> bool:" in engine, (
        "start() 必须返回 bool，好让 UI 能区分「已启动」和「被拒绝」"
    )

    launcher = (ROOT / "scripts" / "run_loop.py").read_text(encoding="utf-8")
    assert ".run_loop.lock" in launcher, "常驻脚本不再用同一把锁了"

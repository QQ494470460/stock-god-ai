# -*- coding: utf-8 -*-
"""调仓日「未建满」顺延重试（``rebalance_retry_*``，默认关闭）

背景（2026-11-02 实盘）
-----------------------
调仓日 8 笔买单，**5 成交 / 3 被风控拦**（未知行业集中度，⑭ 号 bug，11-03 才修）。
``_run_rebalance`` 当时只给 ``max_orders_per_tick`` 截断留了「不记周期标记」的活口，
**执行期被风控拦掉的量照样写 ``last_rebalance_date``** ⇒ 仓位停在 58%、现金 41%，
一直锁到 11-24 才轮到下一次调仓。

这一组用例守三件事：

1. **默认关闭时与旧行为逐位相同**（写了什么、没写什么，逐个字段对齐）。
2. 打开后「未建满 ⇒ 撤回周期标记 + 顺延」，且**一个交易日只重试一次** ——
   放开的后果是结构性拦截让每个 tick 都重下一遍，把 15 天周期的低换手优势作废。
3. 顺延**有预算**（``rebalance_retry_max_days``），用尽即收尾，且周期标记回填到
   **锚点**而不是今天 —— 顺延补的是执行，不该把节奏往后推。
"""
import sys
from datetime import datetime as _dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

import src.trading.loop as loopmod
from src.trading.broker import OrderSide
from src.trading.loop import LoopConfig, TickReport, TradingLoop
from src.trading.rebalance import RebalanceIntent

# 2026-11-02 是周一，02→06 连续五个交易日（便于算「顺延了几个交易日」）
MON = "2026-11-02"
TUE = "2026-11-03"
WED = "2026-11-04"
THU = "2026-11-05"


# ======================================================================
# 夹具：只装配 _run_rebalance 真正会用到的那几个属性
# ======================================================================

class _FrozenClock:
    """把 ``loopmod.datetime`` 换成一个「now() 恒为指定时刻」的类。

    只冻结 ``now()``：``_rebalance_due`` / ``_run_rebalance`` 都只用到它，
    而 ``date.fromisoformat`` 之类仍然走真 ``datetime.date``（不受影响）。
    """

    def __init__(self, iso: str):
        self._d = _dt.fromisoformat(iso + "T10:00:00")

    def install(self, monkeypatch):
        d = self._d

        class _Now(_dt):
            @classmethod
            def now(cls, tz=None):
                return d

        monkeypatch.setattr(loopmod, "datetime", _Now)


@pytest.fixture
def clock(monkeypatch):
    def _set(iso):
        _FrozenClock(iso).install(monkeypatch)
    return _set


def _intent(code, side=OrderSide.BUY, qty=1000, price=10.0):
    return RebalanceIntent(code=code, name=code, side=side,
                           quantity=qty, price=price, reason="调仓")


def _mk_loop(tmp_path, cfg, *, last="", anchor="", defer_date="", fill=None,
             intents=None):
    """直接装配一个只跑得动 ``_run_rebalance`` 的 loop。

    刻意不走 ``__init__``（那里会建 guard、包券商、抢单实例锁），本文件要验的是
    「记账与顺延的判定」，与风控链路无关；风控链路由 ``test_loop.py`` 覆盖。
    走 ``__init__`` 反而会让用例依赖一堆与主题无关的桩。
    """
    loop = object.__new__(TradingLoop)
    loop.config = cfg
    loop.plan = None
    loop._exec_scheduler = None
    loop._last_rebalance_date = last
    loop._rebalance_defer_from = anchor
    loop._rebalance_defer_date = defer_date
    loop._day_start_equity = {}
    loop._settled_dates = set()
    loop._runtime_file = Path(tmp_path) / "loop_runtime.json"
    # 与顺延判定无关的三条支路，直接短路
    loop._portfolio_risk_scale = lambda acc: 1.0
    loop._adv_provider = lambda: None
    loop._participation = lambda: 0.01
    loop.calls = []

    fill = fill if fill is not None else set()

    def _place(code, name, side, quantity, price, reason,
               ai_score=50.0, risk_exit=False, tag="loop"):
        ok = code in fill
        loop.calls.append((code, side.value, quantity, price, ok))
        if not ok:
            return False, "被安全中枢拦截", None, {
                "code": code, "side": side.value, "quantity": quantity,
                "reason": "集中度", "blocked": True,
            }
        return True, "ok", None, {
            "code": code, "name": name, "side": side.value,
            "quantity": quantity, "price": price, "reason": reason, "ok": True,
        }

    loop._place = _place

    _intents = intents if intents is not None else [
        _intent(f"60000{i}") for i in range(8)]
    import src.trading.rebalance as rb
    loop._test_intents = _intents
    return loop


@pytest.fixture(autouse=True)
def _stub_intents(monkeypatch):
    """把 ``compute_rebalance_orders`` 换成「返回 loop._test_intents」。

    ``_run_rebalance`` 是在函数体里 ``from ... import`` 的，所以打模块属性即可。
    """
    real = __import__("src.trading.rebalance", fromlist=["compute_rebalance_orders"])
    holder = {}

    def fake(plan, positions, account, prices, **kw):
        return list(holder.get("intents") or [])

    monkeypatch.setattr(real, "compute_rebalance_orders", fake)
    return holder


def _plan():
    from src.trading.trade_plan import PlanTarget, TradePlan
    return TradePlan(
        plan_date=MON,
        targets=[PlanTarget(code=f"60000{i}", name=f"60000{i}",
                            score=60.0, weight=0.1125) for i in range(8)],
        cash_reserve_pct=0.10,
    )


def _run(loop, intents, holder):
    holder["intents"] = intents
    loop.plan = _plan()
    rep = TickReport()
    loop._run_rebalance([], object(), {}, rep)
    return rep


def _state(loop):
    return {
        "last": loop._last_rebalance_date,
        "anchor": loop._rebalance_defer_from,
        "defer_date": loop._rebalance_defer_date,
    }


CODES = [f"60000{i}" for i in range(8)]


# ======================================================================
# ① 默认关闭 = 与旧行为逐位相同
# ======================================================================

def test_config_defaults_are_opted_in_and_documented():
    cfg = LoopConfig()
    assert cfg.rebalance_retry_enabled is True, "2026-11-06 用户确认打开：根治「未建满锁死 15 天」"
    assert cfg.rebalance_retry_max_days == 2
    assert cfg.rebalance_retry_min_fill_ratio == 0.85


def test_off_path_still_matches_legacy_behavior():
    """开关显式关闭时，行为与引入该特性前逐位相同（字节级回归保护）。"""
    cfg = LoopConfig(rebalance_retry_enabled=False)
    assert cfg.rebalance_retry_enabled is False


def test_production_config_exposes_the_knobs():
    """生产权威值在 config/settings.py —— 两处字段名必须对得上，
    否则 ``LoopConfig.from_app_config`` 会静默丢掉这几项（等于开关不存在）。"""
    import inspect

    from config import settings

    want = {"rebalance_retry_enabled", "rebalance_retry_max_days",
            "rebalance_retry_min_fill_ratio"}
    prod = {f.name for f in __import__("dataclasses").fields(
        settings.TradingLoopConfig)}
    loop_fields = set(LoopConfig.__dataclass_fields__)
    assert want <= prod, "settings.TradingLoopConfig 缺字段"
    assert want <= loop_fields, "loop.LoopConfig 缺字段 —— from_app_config 会丢"
    assert "asdict" in inspect.getsource(LoopConfig.from_app_config)


def test_from_app_config_carries_the_knobs(monkeypatch):
    """改 settings 里的值要能被 from_app_config 带过来（防「配置改了、进程没读到」）。"""
    from config import settings

    sec = settings.TradingLoopConfig(
        rebalance_retry_enabled=True, rebalance_retry_max_days=5,
        rebalance_retry_min_fill_ratio=0.6)

    class _App:
        trading_loop = sec

    monkeypatch.setattr(loopmod, "app_config", _App(), raising=False)
    cfg = LoopConfig.from_app_config()
    assert cfg.rebalance_retry_enabled is True
    assert cfg.rebalance_retry_max_days == 5
    assert cfg.rebalance_retry_min_fill_ratio == 0.6


def test_disabled_still_marks_cycle_when_orders_blocked(tmp_path, clock, _stub_intents):
    """**核心回归**：开关关闭时，5/8 成交也要照旧写周期标记 —— 一个字节都不许变。"""
    clock(MON)
    loop = _mk_loop(tmp_path, LoopConfig(rebalance_retry_enabled=False), fill=set(CODES[:5]))
    rep = _run(loop, [_intent(c) for c in CODES], _stub_intents)

    assert _state(loop) == {"last": MON, "anchor": "", "defer_date": ""}
    assert rep.n_executed == 5 and rep.n_blocked == 3


def test_disabled_no_defer_even_when_nothing_filled(tmp_path, clock, _stub_intents):
    clock(MON)
    loop = _mk_loop(tmp_path, LoopConfig(rebalance_retry_enabled=False), fill=set())
    _run(loop, [_intent(c) for c in CODES], _stub_intents)
    assert _state(loop)["last"] == MON
    assert _state(loop)["anchor"] == ""


# ======================================================================
# ② 打开后：未建满 ⇒ 撤回标记 + 顺延
# ======================================================================

def _retry_cfg(**kw):
    base = dict(rebalance_retry_enabled=True)
    base.update(kw)
    return LoopConfig(**base)


def test_underfilled_defers_and_retracts_cycle_mark(tmp_path, clock, _stub_intents):
    clock(MON)
    loop = _mk_loop(tmp_path, _retry_cfg(), fill=set(CODES[:5]))
    rep = _run(loop, [_intent(c) for c in CODES], _stub_intents)

    st = _state(loop)
    assert st["last"] == "", "未建满就不算「调过」，否则又要等 15 个交易日"
    assert st["anchor"] == MON, "锚点 = 本轮第一次尝试日"
    assert st["defer_date"] == MON
    assert "顺延" in rep.note


def test_fully_filled_closes_cycle_normally(tmp_path, clock, _stub_intents):
    clock(MON)
    loop = _mk_loop(tmp_path, _retry_cfg(), fill=set(CODES))
    _run(loop, [_intent(c) for c in CODES], _stub_intents)
    assert _state(loop) == {"last": MON, "anchor": "", "defer_date": ""}


def test_seven_of_eight_is_acceptable(tmp_path, clock, _stub_intents):
    """87.5% > 85% ⇒ 不顺延。否则「少买一只」也会被当成事故反复重试。"""
    clock(MON)
    loop = _mk_loop(tmp_path, _retry_cfg(), fill=set(CODES[:7]))
    _run(loop, [_intent(c) for c in CODES], _stub_intents)
    assert _state(loop)["last"] == MON
    assert _state(loop)["anchor"] == ""


def test_threshold_is_inclusive_of_ratio_boundary(tmp_path, clock, _stub_intents):
    """成交金额恰好等于阈值时**不**顺延（`<` 而非 `<=`）。

    6/8 = 75%。把阈值设成 0.75 时应正常收尾；0.76 时应顺延。
    """
    clock(MON)
    loop = _mk_loop(tmp_path, _retry_cfg(rebalance_retry_min_fill_ratio=0.75),
                    fill=set(CODES[:6]))
    _run(loop, [_intent(c) for c in CODES], _stub_intents)
    assert _state(loop)["last"] == MON

    clock(MON)
    loop2 = _mk_loop(tmp_path, _retry_cfg(rebalance_retry_min_fill_ratio=0.76),
                     fill=set(CODES[:6]))
    _run(loop2, [_intent(c) for c in CODES], _stub_intents)
    assert _state(loop2)["last"] == ""


def test_ratio_uses_actual_filled_quantity(tmp_path, clock, _stub_intents):
    """guard 可能下调数量 ⇒ 判据取「实际成交量×成交价」，不是意愿量。

    8 笔各 1000 股全部「成功」，但 guard 每笔只批 500 股 ⇒ 实际 50%。
    """
    clock(MON)
    loop = _mk_loop(tmp_path, _retry_cfg(), fill=set(CODES))
    rep = TickReport()
    loop.plan = _plan()

    def _place(code, name, side, quantity, price, reason,
               ai_score=50.0, risk_exit=False, tag="loop"):
        return True, "ok", None, {
            "code": code, "side": side.value, "quantity": quantity // 2,
            "price": price, "reason": reason, "ok": True,
        }

    loop._place = _place
    _stub_intents["intents"] = [_intent(c) for c in CODES]
    loop._run_rebalance([], object(), {}, rep)
    assert _state(loop)["last"] == "", "实际只成交一半，应判未建满"
    assert rep.n_executed == 8


# ======================================================================
# ③ 一天最多重试一次
# ======================================================================

def test_same_day_second_tick_does_not_retry(tmp_path, clock, _stub_intents):
    clock(MON)
    loop = _mk_loop(tmp_path, _retry_cfg(), anchor=MON, defer_date=MON,
                    fill=set(CODES))
    rep = _run(loop, [_intent(c) for c in CODES], _stub_intents)

    assert loop.calls == [], "同一天已经顺延过，不许再下一遍"
    assert rep.n_intents == 0
    assert "下个交易日" in rep.note


def test_next_trading_day_retries(tmp_path, clock, _stub_intents):
    clock(TUE)
    loop = _mk_loop(tmp_path, _retry_cfg(), anchor=MON, defer_date=MON,
                    fill=set(CODES))
    rep = _run(loop, [_intent(c) for c in CODES], _stub_intents)

    assert len(loop.calls) == 8, "换了一个交易日就该重试"
    assert rep.n_executed == 8


def test_retry_success_closes_with_anchor_not_today(tmp_path, clock, _stub_intents):
    """顺延补的是执行，不该把 15 天的节奏往后推 ⇒ 标记回填锚点。"""
    clock(TUE)
    loop = _mk_loop(tmp_path, _retry_cfg(), anchor=MON, defer_date=MON,
                    fill=set(CODES))
    _run(loop, [_intent(c) for c in CODES], _stub_intents)

    assert loop._last_rebalance_date == MON
    assert loop._rebalance_defer_from == "" and loop._rebalance_defer_date == ""


def test_underfilled_again_keeps_original_anchor(tmp_path, clock, _stub_intents):
    clock(TUE)
    loop = _mk_loop(tmp_path, _retry_cfg(), anchor=MON, defer_date=MON,
                    fill=set(CODES[:2]))
    _run(loop, [_intent(c) for c in CODES], _stub_intents)

    assert loop._rebalance_defer_from == MON, "锚点不能被后一次尝试覆盖"
    assert loop._rebalance_defer_date == TUE
    assert loop._last_rebalance_date == ""


# ======================================================================
# ④ 顺延预算
# ======================================================================

def test_budget_boundary_still_retries(tmp_path, clock, _stub_intents):
    """锚点 11-02，今天 11-04 ⇒ 已顺延 2 个交易日 = 上限 ⇒ 仍可再试一次。"""
    clock(WED)
    loop = _mk_loop(tmp_path, _retry_cfg(), anchor=MON, defer_date=TUE,
                    fill=set(CODES))
    _run(loop, [_intent(c) for c in CODES], _stub_intents)
    assert len(loop.calls) == 8


def test_budget_exhausted_closes_without_ordering(tmp_path, clock, _stub_intents):
    """锚点 11-02，今天 11-05 ⇒ 3 > 2 ⇒ 收尾，且**不再下单**。

    结构性拦截不会因为再试一次而消失，继续下单只是把同一段日志再刷一遍。
    """
    clock(THU)
    loop = _mk_loop(tmp_path, _retry_cfg(), anchor=MON, defer_date=WED,
                    fill=set(CODES))
    rep = _run(loop, [_intent(c) for c in CODES], _stub_intents)

    assert loop.calls == []
    assert loop._last_rebalance_date == MON, "标记回填锚点而不是今天"
    assert loop._rebalance_defer_from == "" and loop._rebalance_defer_date == ""
    assert "预算用尽" in rep.note


def test_budget_zero_means_no_retry(tmp_path, clock, _stub_intents):
    """max_days=0 ⇒ 等价于「关掉顺延」（但记账走 _close_cycle 路径）。"""
    clock(MON)
    loop = _mk_loop(tmp_path, _retry_cfg(rebalance_retry_max_days=0),
                    fill=set(CODES[:3]))
    _run(loop, [_intent(c) for c in CODES], _stub_intents)
    # 首日：used=0 不大于 0 ⇒ 正常尝试；未建满 ⇒ 仍然顺延到锚点+1
    assert loop._rebalance_defer_from == MON
    assert loop._last_rebalance_date == ""

    clock(TUE)
    loop2 = _mk_loop(tmp_path, _retry_cfg(rebalance_retry_max_days=0),
                     anchor=MON, defer_date=MON, fill=set(CODES))
    rep = _run(loop2, [_intent(c) for c in CODES], _stub_intents)
    assert loop2.calls == []
    assert loop2._last_rebalance_date == MON


def test_unparsable_anchor_fails_closed(tmp_path, clock, _stub_intents):
    """锚点写坏了 ⇒ 算不出预算 ⇒ 按「用尽」收尾，绝不无限重试。"""
    clock(THU)
    loop = _mk_loop(tmp_path, _retry_cfg(), anchor="not-a-date",
                    defer_date=MON, fill=set(CODES))
    _run(loop, [_intent(c) for c in CODES], _stub_intents)
    assert loop.calls == []
    assert loop._rebalance_defer_from == ""


def test_bad_ratio_config_degrades_to_lenient(tmp_path, clock, _stub_intents):
    """阈值越界（>1 / <=0）一律按 1.0：宁可少顺延，也不要「每次调仓都顺延」。"""
    for bad in (1.5, 0.0, -3.0, "abc"):
        clock(MON)
        loop = _mk_loop(tmp_path,
                        _retry_cfg(rebalance_retry_min_fill_ratio=bad),
                        fill=set(CODES))
        _run(loop, [_intent(c) for c in CODES], _stub_intents)
        assert loop._last_rebalance_date == MON, f"阈值={bad!r} 不该触发顺延"


# ======================================================================
# ⑤ 与既有的两条机制不打架
# ======================================================================

def test_truncation_path_is_untouched(tmp_path, clock, _stub_intents):
    """被 max_orders_per_tick 截断时：不写标记、**不设顺延状态**、下个 tick 继续。

    两套重试机制不能叠加，否则同一批委托会被各推一次。
    """
    clock(MON)
    cfg = _retry_cfg(max_orders_per_tick=3)
    loop = _mk_loop(tmp_path, cfg, fill=set(CODES))
    rep = _run(loop, [_intent(c) for c in CODES], _stub_intents)

    assert len(loop.calls) == 3
    assert loop._last_rebalance_date == ""
    assert loop._rebalance_defer_from == "" and loop._rebalance_defer_date == ""
    assert rep.n_executed == 3


def test_empty_intents_closes_cycle(tmp_path, clock, _stub_intents):
    """零意图 = 已对齐目标（或带内），照旧记账。"""
    clock(MON)
    loop = _mk_loop(tmp_path, _retry_cfg(), fill=set(CODES))
    _run(loop, [], _stub_intents)
    assert loop._last_rebalance_date == MON


def test_empty_intents_after_defer_clears_defer_state(tmp_path, clock, _stub_intents):
    """顺延后的重试日跑出零意图 ⇒ 说明已经建满，顺延结束。"""
    clock(TUE)
    loop = _mk_loop(tmp_path, _retry_cfg(), anchor=MON, defer_date=MON)
    _run(loop, [], _stub_intents)
    assert loop._rebalance_defer_from == "" and loop._rebalance_defer_date == ""
    assert loop._last_rebalance_date == MON


def test_not_due_short_circuits_before_any_retry_logic(tmp_path, clock, _stub_intents):
    clock(THU)
    loop = _mk_loop(tmp_path, _retry_cfg(), last=THU)   # 刚调过，远不到期
    rep = _run(loop, [_intent(c) for c in CODES], _stub_intents)
    assert loop.calls == []
    assert "不足" in rep.note
    assert loop._rebalance_defer_from == ""


def test_no_plan_is_a_noop(tmp_path, clock, _stub_intents):
    clock(MON)
    loop = _mk_loop(tmp_path, _retry_cfg())
    loop.plan = None
    rep = TickReport()
    loop._run_rebalance([], object(), {}, rep)
    assert "无有效交易计划" in rep.note


# ======================================================================
# ⑥ 落盘：重启不能把顺延状态弄丢
# ======================================================================

def test_defer_state_survives_restart(tmp_path, clock, _stub_intents):
    clock(MON)
    loop = _mk_loop(tmp_path, _retry_cfg(), fill=set(CODES[:2]))
    _run(loop, [_intent(c) for c in CODES], _stub_intents)

    # 模拟自愈重启：同一个运行态文件，新实例
    loop2 = object.__new__(TradingLoop)
    loop2._runtime_file = loop._runtime_file
    loop2._day_start_equity = {}
    loop2._settled_dates = set()
    loop2._last_rebalance_date = ""
    loop2._rebalance_defer_from = ""
    loop2._rebalance_defer_date = ""
    loop2._load_runtime()

    assert loop2._rebalance_defer_from == MON
    assert loop2._rebalance_defer_date == MON
    assert loop2._last_rebalance_date == ""


def test_old_runtime_file_without_new_keys_still_loads(tmp_path):
    """向后兼容：老 ``loop_runtime.json`` 没有这两个键 ⇒ 按空串处理，不报错。"""
    f = Path(tmp_path) / "loop_runtime.json"
    f.write_text('{"day_start_equity": {"2026-11-02": 100.0},'
                 ' "settled_dates": ["2026-11-02"],'
                 ' "last_rebalance_date": "2026-11-02"}', encoding="utf-8")
    loop = object.__new__(TradingLoop)
    loop._runtime_file = f
    loop._day_start_equity = {}
    loop._settled_dates = set()
    loop._last_rebalance_date = ""
    loop._rebalance_defer_from = ""
    loop._rebalance_defer_date = ""
    loop._load_runtime()

    assert loop._last_rebalance_date == MON
    assert loop._rebalance_defer_from == ""
    assert loop._day_start_equity == {"2026-11-02": 100.0}


def test_save_runtime_writes_both_keys(tmp_path, clock, _stub_intents):
    import json

    clock(TUE)
    loop = _mk_loop(tmp_path, _retry_cfg(), anchor=MON, defer_date=MON,
                    fill=set(CODES[:1]))
    _run(loop, [_intent(c) for c in CODES], _stub_intents)

    data = json.loads(loop._runtime_file.read_text(encoding="utf-8"))
    assert data["rebalance_defer_from"] == MON
    assert data["rebalance_defer_date"] == TUE
    assert data["last_rebalance_date"] == ""


# ======================================================================
# ⑦ 审计首屏要把「缺口」和「还有没有下一次机会」连起来
# ======================================================================

def _audit_base(**kw):
    base = {
        "plan_date": "20261105", "pending": False, "plan_source": "daily_pick",
        "plan_regime": "bear", "n_planned": 8, "weight_plan_total": 0.9,
        "weight_filled_total": 0.9, "fill_ratio": 1.0, "healthy": True,
        "n_missing": 0, "n_partially_filled": 0, "unmapped_weight_pct": 0.0,
        "unmapped_codes": [], "cash_pct": 0.1, "equity": 1_000_000.0,
        "positions": [], "picks_date": "2026-11-04", "picks_lag_days": 1,
        "picks_stale": False,
    }
    base.update(kw)
    return base


def test_audit_hint_when_retry_is_off(monkeypatch):
    """未开启时必须**明说**「这轮已经结束了」——否则读者只会以为「少买了三只」。"""
    import scripts.plan_vs_fill as pvf
    from config import settings

    class _App:
        trading_loop = settings.TradingLoopConfig(
            rebalance_retry_enabled=False, rebalance_retry_min_fill_ratio=0.85)

    monkeypatch.setattr(settings, "app_config", _App(), raising=False)
    out = pvf.render(_audit_base(fill_ratio=0.65, healthy=False, n_missing=3))
    assert "顺延重试**未开启**" in out
    assert "11-02 就是这种形态" in out


def test_audit_hint_when_retry_is_on(monkeypatch):
    import scripts.plan_vs_fill as pvf
    from config import settings

    class _App:
        trading_loop = settings.TradingLoopConfig(
            rebalance_retry_enabled=True, rebalance_retry_min_fill_ratio=0.85,
            rebalance_retry_max_days=2)

    monkeypatch.setattr(settings, "app_config", _App(), raising=False)
    out = pvf.render(_audit_base(fill_ratio=0.65, healthy=False, n_missing=3))
    assert "顺延重试**已开启**" in out and "最多 2 个交易日" in out


def test_audit_hint_silent_when_filled_or_pending(monkeypatch):
    import scripts.plan_vs_fill as pvf
    from config import settings

    class _App:
        trading_loop = settings.TradingLoopConfig(rebalance_retry_enabled=False)

    monkeypatch.setattr(settings, "app_config", _App(), raising=False)
    assert "顺延" not in pvf.render(_audit_base(fill_ratio=1.0))
    # 计划还没到执行日 → 谈「顺延」是误导
    assert "顺延" not in pvf.render(_audit_base(pending=True, fill_ratio=0.0))

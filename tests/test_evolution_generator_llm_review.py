# -*- coding: utf-8 -*-
"""自我进化 · 生成器 B（LLM 自我复盘）测试。

本文件的断言集中在**安全边界**上，因为引入 LLM 等于引入了一个
「会一本正经胡说八道的外部依赖」，必须证明：

1. LLM **无法**产出任何数值 —— 它只能从封闭动作目录里选 id；
2. LLM **无法**触碰风控硬线 —— 目录 + ``space.validate`` 两道锁；
3. LLM **无法**给自己盖章 —— 候选的 ``evaluation`` 恒为 ``None``；
4. LLM **挂了/胡说** 时系统按兵不动，而不是崩掉或乱改参数。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evolution import overrides as ov_mod, space  # noqa: E402
from src.evolution.feedback import PerformanceSnapshot  # noqa: E402
from src.evolution.generators import REGISTRY  # noqa: E402
from src.evolution.generators import llm_review as lr  # noqa: E402

CURRENT = {
    "trading_loop.band_pct": 0.25,
    "trading_loop.cash_reserve_pct": 0.10,
    "trading_loop.max_positions": 10,
    "trading_loop.weight_cap": 0.30,
    "trading_loop.min_trade_amount": 5000.0,
    "trading_loop.weighting": "equal",
}


# ------------------------------------------------------------------ 夹具

def _snap(**kw) -> PerformanceSnapshot:
    base = dict(
        generated_at="2026-10-14T13:00:00",
        n_batches=25, n_verified=17, pending_verify=31,
        accuracy_by_horizon={"5": [0.20, 0.15, 0.15, 0.333, 0.20]},
        excess_by_horizon={"5": [-0.01, 0.02, -0.005]},
        win_rate_by_horizon={"5": [0.40, 0.55, 0.42]},
        n_settlements=20,
        equity_series=[
            {"date": f"2026-09-{i:02d}", "equity": 100000.0,
             "cash": 10000.0, "n_positions": 10}
            for i in range(1, 21)],
        trade_stats={"fill": 40, "risk_block": 3},
        guard={"halted": False, "halt_reason": ""},
        metrics={"insufficient": False, "total_return": 0.08,
                 "max_drawdown": -0.03, "daily_win_rate": 0.42,
                 "profit_loss_ratio": 1.8, "turnover": 5.2, "sharpe": 1.1},
    )
    base.update(kw)
    return PerformanceSnapshot(**base)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """参数取值全部指向确定性字典，避免读真实配置。"""
    monkeypatch.setattr(ov_mod, "OVERRIDES_FILE", tmp_path / "ov.json")
    monkeypatch.setattr(ov_mod, "read_effective", lambda k: CURRENT.get(k))
    return tmp_path


def _chat(payload) -> callable:
    """造一个返回固定内容的假 LLM。payload 可以是 dict / str。"""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)

    def fn(system, user):
        return text

    return fn


# ------------------------------------------------------------------ ① 目录不得越界

def test_catalog_is_clean():
    assert lr.catalog_problems() == []


def test_catalog_only_targets_whitelisted_params():
    for name, spec in lr.ACTIONS.items():
        key = spec["key"]
        assert not space.is_protected(key), f"{name} 指向了受保护参数 {key}"
        assert key in space.PARAM_SPACE, f"{name} 指向了未登记参数 {key}"


def test_catalog_problems_detects_protected_target(monkeypatch):
    """有人往目录里塞风控参数 → 自检必须报出来。"""
    monkeypatch.setitem(lr.ACTIONS, "evil", {
        "key": "guard.max_portfolio_drawdown_pct", "op": "bump", "sign": 1,
        "title": "调熔断线", "meaning": "…",
    })
    problems = lr.catalog_problems()
    assert problems and any("guard." in p for p in problems)


def test_catalog_problems_detects_unknown_param(monkeypatch):
    monkeypatch.setitem(lr.ACTIONS, "typo", {
        "key": "trading_loop.bandpct", "op": "bump", "sign": 1,
        "title": "拼错的参数", "meaning": "…",
    })
    assert any("typo" in p for p in lr.catalog_problems())


def test_catalog_problems_detects_bad_op(monkeypatch):
    monkeypatch.setitem(lr.ACTIONS, "weird", {
        "key": "trading_loop.band_pct", "op": "multiply", "sign": 1,
        "title": "未知算子", "meaning": "…",
    })
    assert any("weird" in p for p in lr.catalog_problems())


# ------------------------------------------------------------------ ② 事实底稿

def test_factsheet_carries_real_numbers(env):
    text = lr.build_factsheet(_snap())

    assert "T+5 准确率" in text
    assert "20.0%" in text                      # 引自 accuracy_by_horizon
    assert "5.2 倍" in text                     # 换手
    assert "25 批，已验证 17 批" in text         # 验证进度
    assert "31" in text                         # 到期未验
    assert "-3.00%" in text                     # 最大回撤
    assert "trading_loop.band_pct" in text      # 可调参数现状
    assert "widen_band" in text                 # 动作目录


def test_factsheet_never_lists_protected_params(env):
    """底稿里不能出现风控硬线参数名 —— 免得 LLM 以为能调。"""
    text = lr.build_factsheet(_snap())
    for ns in space.PROTECTED_NAMESPACES:
        assert ns not in text


def test_factsheet_flags_insufficient_metrics(env):
    text = lr.build_factsheet(_snap(metrics={"insufficient": True}))
    assert "不可用" in text


def test_factsheet_includes_notes(env):
    text = lr.build_factsheet(_snap(notes=["闭环尚未收盘结算过任何一天"]))
    assert "闭环尚未收盘结算过任何一天" in text


def test_factsheet_without_equity(env):
    text = lr.build_factsheet(_snap(equity_series=[]))
    assert "持仓结构不可得" in text


# ------------------------------------------------------------------ ③ 解析

def test_parse_plain_json():
    out = lr.parse_review(json.dumps({
        "verdict": "换手偏高",
        "actions": [{"action": "widen_band", "confidence": 0.9,
                     "reason": "换手 5.2 倍", "evidence": "turnover"}],
    }, ensure_ascii=False))
    assert out["verdict"] == "换手偏高"
    assert out["actions"][0]["action"] == "widen_band"
    assert out["actions"][0]["confidence"] == pytest.approx(0.9)


def test_parse_fenced_json():
    text = "好的，我的分析如下：\n```json\n" + json.dumps({
        "verdict": "x", "actions": [{"action": "widen_band", "reason": "r"}]}
    ) + "\n```\n希望有帮助。"
    out = lr.parse_review(text)
    assert len(out["actions"]) == 1
    assert out["actions"][0]["confidence"] == pytest.approx(0.5)   # 缺省 0.5


def test_parse_drops_unknown_action():
    out = lr.parse_review(json.dumps({
        "actions": [{"action": "set_drawdown_limit_to_50", "reason": "r"}],
    }))
    assert out["actions"] == []


def test_parse_drops_action_without_reason():
    out = lr.parse_review(json.dumps({
        "actions": [{"action": "widen_band", "reason": "   "}],
    }))
    assert out["actions"] == []


def test_parse_keeps_good_drops_bad():
    """坏动作不该连坐好动作。"""
    out = lr.parse_review(json.dumps({
        "actions": [
            {"action": "widen_band", "reason": "换手 5.2 倍", "confidence": 0.8},
            {"action": "nonsense_action", "reason": "r"},
            {"action": "raise_cash", "reason": "回撤 -3%", "confidence": 0.7},
        ],
    }, ensure_ascii=False))
    assert [a["action"] for a in out["actions"]] == ["widen_band", "raise_cash"]


def test_parse_clamps_confidence():
    out = lr.parse_review(json.dumps({
        "actions": [{"action": "widen_band", "reason": "r", "confidence": 7}],
    }))
    assert out["actions"][0]["confidence"] == 1.0


def test_parse_handles_non_dict_top_level():
    assert lr.parse_review("[1,2,3]")["actions"] == []


def test_parse_handles_no_json():
    assert lr.parse_review("我不知道该怎么复盘。")["actions"] == []


def test_parse_handles_single_action_object():
    out = lr.parse_review(json.dumps({
        "actions": {"action": "widen_band", "reason": "r"}}))
    assert len(out["actions"]) == 1


# ------------------------------------------------------------------ ④ 动作 → 数值

def test_resolve_bump_uses_param_step(env):
    r = lr.resolve("widen_band")
    assert r["key"] == "trading_loop.band_pct"
    assert r["value"] == pytest.approx(0.30)        # 0.25 + step 0.05


def test_resolve_bump_direction(env):
    assert lr.resolve("lower_cash")["value"] == pytest.approx(0.05)
    assert lr.resolve("raise_cash")["value"] == pytest.approx(0.15)
    assert lr.resolve("cut_positions")["value"] == 8      # 10 - 2*step1
    assert lr.resolve("add_positions")["value"] == 12


def test_resolve_set_category(env):
    r = lr.resolve("use_score_weighting")
    assert r["key"] == "trading_loop.weighting"
    assert r["value"] == "score"


def test_resolve_returns_none_on_unknown_action(env):
    assert lr.resolve("rm_rf_slash") is None


def test_resolve_returns_none_at_boundary(env, monkeypatch):
    """band_pct 已在区间上界时，再放宽就是无效动作。"""
    monkeypatch.setattr(ov_mod, "read_effective",
                        lambda k: 0.50 if k == "trading_loop.band_pct" else CURRENT.get(k))
    assert lr.resolve("widen_band") is None
    assert lr.resolve("tighten_band") is not None


def test_resolve_falls_back_to_default_when_current_unreadable(env, monkeypatch):
    """读不到当前值时用参数空间默认值算，不抛错。"""
    monkeypatch.setattr(ov_mod, "read_effective", lambda k: None)
    r = lr.resolve("widen_band")

    # 期望值由「参数空间默认 + 一个 step」推出，不硬编码 ——
    # 默认值会随参数扫描结论演进（2026-10-27 由 0.25 → 0.15）。
    spec = space.get_spec("trading_loop.band_pct")
    assert r["value"] == pytest.approx(spec.clamp(spec.default + spec.step))


# ------------------------------------------------------------------ ⑤ generate 的安全边界

def test_generate_uses_code_computed_values_not_llm_numbers(env):
    """LLM 就算塞了数值字段，也必须被无视。"""
    chat = _chat({
        "verdict": "放宽容忍带",
        "actions": [{"action": "widen_band", "confidence": 0.9,
                     "reason": "换手 5.2 倍", "value": 999, "new_value": 0.99}],
    })
    cands = lr.generate(_snap(), chat_fn=chat)
    assert len(cands) == 1
    ch = cands[0].changes[0]
    assert ch.key == "trading_loop.band_pct"
    assert ch.new == pytest.approx(0.30)            # 代码按步长算，不是 999


def test_generate_never_self_approves(env):
    chat = _chat({"actions": [{"action": "widen_band", "confidence": 1.0,
                               "reason": "换手 5.2 倍", "safe_to_auto_apply": True}]})
    c = lr.generate(_snap(), chat_fn=chat)[0]
    assert c.evaluation is None          # 没有任何自盖章
    assert c.status == "pending"


def test_generate_records_confidence_in_meta_only(env):
    chat = _chat({"verdict": "总体偏差", "actions": [
        {"action": "widen_band", "confidence": 0.88, "reason": "换手 5.2 倍",
         "evidence": "turnover"}]})
    c = lr.generate(_snap(), chat_fn=chat)[0]
    assert c.meta["llm"]["confidence"] == pytest.approx(0.88)
    assert c.meta["llm"]["verdict"] == "总体偏差"
    assert c.meta["llm"]["evidence"] == "turnover"
    assert "signature" in c.meta


def test_generate_rationale_keeps_llm_reason(env):
    chat = _chat({"actions": [{"action": "widen_band", "confidence": 0.9,
                               "reason": "换手 5.2 倍，手续费吃掉收益"}]})
    c = lr.generate(_snap(), chat_fn=chat)[0]
    assert "换手 5.2 倍" in c.rationale
    assert "widen_band" in c.rationale
    assert "数值由参数步长算出" in c.rationale


def test_generate_dedupes_same_key_keeping_higher_confidence(env):
    """同时对同一参数给出两个方向 → 只留置信度高的。"""
    chat = _chat({"actions": [
        {"action": "widen_band", "confidence": 0.6, "reason": "少交易"},
        {"action": "tighten_band", "confidence": 0.9, "reason": "贴近计划"},
    ]})
    cands = lr.generate(_snap(), chat_fn=chat)
    assert len(cands) == 1
    assert cands[0].changes[0].new == pytest.approx(0.20)   # tighten 胜出


def test_generate_filters_low_confidence(env):
    chat = _chat({"actions": [{"action": "widen_band", "confidence": 0.2,
                               "reason": "有点想改"}]})
    assert lr.generate(_snap(), chat_fn=chat, min_confidence=0.5) == []
    assert len(lr.generate(_snap(), chat_fn=chat, min_confidence=0.1)) == 1


def test_generate_respects_max_candidates(env):
    chat = _chat({"actions": [
        {"action": "widen_band", "confidence": 0.9, "reason": "a"},
        {"action": "raise_min_trade", "confidence": 0.8, "reason": "b"},
        {"action": "lower_weight_cap", "confidence": 0.7, "reason": "c"},
    ]})
    assert len(lr.generate(_snap(), chat_fn=chat, max_candidates=2)) == 2


def test_generate_drops_invalid_action_at_boundary(env, monkeypatch):
    monkeypatch.setattr(ov_mod, "read_effective",
                        lambda k: 0.50 if k == "trading_loop.band_pct" else CURRENT.get(k))
    chat = _chat({"actions": [{"action": "widen_band", "confidence": 0.9,
                               "reason": "换手高"}]})
    assert lr.generate(_snap(), chat_fn=chat) == []


def test_generate_returns_empty_when_llm_says_nothing(env):
    assert lr.generate(_snap(), chat_fn=_chat({"verdict": "一切正常", "actions": []})) == []


def test_generate_returns_empty_without_evidence(env):
    chat = _chat({"actions": [{"action": "widen_band", "confidence": 0.9, "reason": "r"}]})
    # 证据不足：0 批验证 + 0 结算
    assert lr.generate(_snap(n_verified=0, n_settlements=0), chat_fn=chat) == []


# ------------------------------------------------------------------ ⑥ 外部依赖故障

def test_generate_survives_llm_exception(env):
    """LLM 网络挂了 → 空列表，不抛异常（不该让回合看起来像 bug）。"""
    def boom(system, user):
        raise RuntimeError("DeepSeek 请求超时")

    assert lr.generate(_snap(), chat_fn=boom) == []


def test_generate_survives_garbage_reply(env):
    assert lr.generate(_snap(), chat_fn=_chat("我今天不想复盘 🌚")) == []


def test_generate_skips_without_api_key(env, monkeypatch):
    """未配置 Key → 直接跳过，且**不**发起调用。"""
    called = {"n": 0}

    def fake_default_chat(system, user):
        called["n"] += 1
        return "{}"

    monkeypatch.setattr(lr, "_llm_ready", lambda: False)
    monkeypatch.setattr(lr, "_default_chat", fake_default_chat)

    assert lr.generate(_snap()) == []
    assert called["n"] == 0


def test_generate_refuses_when_catalog_broken(env, monkeypatch):
    """目录越界 → 停手，一条候选都不出。"""
    monkeypatch.setitem(lr.ACTIONS, "evil", {
        "key": "guard.max_portfolio_drawdown_pct", "op": "set", "value": 50.0,
        "title": "调熔断线", "meaning": "…",
    })
    chat = _chat({"actions": [{"action": "widen_band", "confidence": 0.9, "reason": "r"}]})
    assert lr.generate(_snap(), chat_fn=chat) == []


# ------------------------------------------------------------------ ⑦ 注册与集成

def test_registered_in_registry():
    assert "llm_review" in REGISTRY
    assert REGISTRY["llm_review"] is lr.generate


def test_prompt_forbids_numbers_and_requires_catalog_ids():
    s = lr.SYSTEM_PROMPT
    assert "只能输出一个 JSON" in s
    assert "不能" in s and "给出任何数值" in s
    assert "动作清单里的 id" in s
    assert "按兵不动是合法答案" in s

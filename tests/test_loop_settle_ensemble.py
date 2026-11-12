# -*- coding: utf-8 -*-
"""闭环结算日报渲染超级集成健康审计（纯渲染单测，不触网/不建 TradingLoop 实例）。

验证 loop.py 的 _settle_markdown 在 snapshot 携带 ensemble_health / ensemble_alarm
时，把超级集成健康审计渲染进日报（降级显眼），未携带时则干净不渲染。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.trading.loop import TradingLoop  # noqa: E402


def _base_snapshot():
    return {
        "date": "2026-11-12",
        "settled_at": "2026-11-12T15:00:00",
        "state": "running",
        "guard_mode": "paper",
        "plan_date": "2026-11-12",
        "equity": 100000.0,
        "day_start_equity": 99000.0,
        "equity_ok": True,
        "day_pnl": 1000.0,
        "day_pnl_pct": 1.01,
        "cash": 10000.0,
        "market_value": 90000.0,
        "n_positions": 0,
        "positions": [],
        "trades": [],
        "stats": {"stops": 0, "intents": 0, "executed": 0, "blocked": 0},
    }


HEALTH = (
    "🧩 超级集成健康审计：\n"
    "  启用腿 2/3（动态权重=关）\n"
    "  · alpha_xgboost：✅ 可用（权重 1.0）\n"
    "  · lstm：⚠️ 降级（原因：RuntimeError: lstm_best.pt 缺失）（权重 0.8）"
)
ALARM = (
    "⚠️ 超级集成有腿静默降级：\n"
    "  · lstm：RuntimeError: lstm_best.pt 缺失"
)


def test_settle_markdown_includes_ensemble_health_section():
    snap = _base_snapshot()
    snap["ensemble_health"] = HEALTH
    snap["ensemble_alarm"] = ALARM
    md = TradingLoop._settle_markdown(snap)
    # 健康审计章节
    assert "## 🧩 超级集成健康" in md
    assert "alpha_xgboost：✅ 可用" in md
    assert "lstm：⚠️ 降级" in md
    # 降级告警渲染成块引用（呼应「用户唯一主动翻的产物」要显眼）
    assert "> ⚠️ **超级集成有腿静默降级**：" in md
    assert ">   · lstm：RuntimeError: lstm_best.pt 缺失" in md


def test_settle_markdown_omits_section_when_no_ensemble_fields():
    snap = _base_snapshot()  # 无 ensemble_health / ensemble_alarm
    md = TradingLoop._settle_markdown(snap)
    assert "超级集成健康" not in md
    assert "超级集成有腿静默降级" not in md

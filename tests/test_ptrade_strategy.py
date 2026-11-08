# -*- coding: utf-8 -*-
"""离线单测：Ptrade 策略适配器的纯函数层（不依赖 Ptrade 引擎/全局）。

这些函数不引用任何 Ptrade 全局（get_history/order/set_universe 等），
因此无需券商环境即可验证核心数学正确性。适配层（run_strategy 等）只在
Ptrade 终端内被引擎调用，无法离线测试，靠部署后模拟盘验证。
"""
import sys
from pathlib import Path

import pytest

# 确保仓库根在 sys.path（pytest.ini 已配 pythonpath=.，这里双保险）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading.ptrade_strategy import (  # noqa: E402
    _round_lot,
    portfolio_drawdown_pct,
    is_rebalance_day,
    compute_heuristic_signal,
    select_targets,
)


def test_round_lot():
    assert _round_lot(150) == 100
    assert _round_lot(250) == 200
    assert _round_lot(100) == 100
    assert _round_lot(99) == 0
    assert _round_lot(0) == 0
    assert _round_lot(-50) == 0


def test_portfolio_drawdown_pct():
    assert portfolio_drawdown_pct(100.0, 80.0) == pytest.approx(20.0)
    assert portfolio_drawdown_pct(100.0, 100.0) == pytest.approx(0.0)
    # peak<=0 守卫：不应出现除零
    assert portfolio_drawdown_pct(0.0, 50.0) == 0.0
    assert portfolio_drawdown_pct(200.0, 260.0) == pytest.approx(-30.0)


def test_is_rebalance_day():
    assert is_rebalance_day(15, 15) is True
    assert is_rebalance_day(30, 15) is True
    assert is_rebalance_day(14, 15) is False
    assert is_rebalance_day(1, 15) is False
    assert is_rebalance_day(0, 15) is True  # 0 % 15 == 0（边界）


def test_heuristic_signal_direction():
    # 强上升 → >50
    up = list(range(1, 31))
    assert compute_heuristic_signal(up) > 50.0
    # 强下降 → <50
    down = list(range(30, 0, -1))
    assert compute_heuristic_signal(down) < 50.0
    # 持平 → ≈50
    flat = [10.0] * 25
    assert compute_heuristic_signal(flat) == pytest.approx(50.0, abs=1e-6)
    # 数据不足 → 兜底 50
    assert compute_heuristic_signal([1.0, 2.0]) == 50.0
    # 分数恒在 [0,100]
    s = compute_heuristic_signal(up)
    assert 0.0 <= s <= 100.0


def test_select_targets_top_n_and_equal_weight():
    scores = {f"c{i}": 100 - 10 * i for i in range(10)}  # c0=100 ... c9=10
    targets = select_targets(scores, set(), {})
    assert len(targets) == 8
    assert set(targets.keys()) == {f"c{i}" for i in range(8)}
    # 等权：(1-0.10)/8 = 0.1125
    assert all(abs(w - 0.1125) < 1e-9 for w in targets.values())


def test_select_targets_single_position_cap():
    scores = {"A": 90.0, "B": 80.0}
    targets = select_targets(scores, set(), {})
    # 仅 2 只：原始 (0.9)/2=0.45 超过单票上限 0.30 → 封顶
    assert targets == {"A": 0.30, "B": 0.30}


def test_select_targets_hold_rank_slack_retains_held():
    scores = {f"c{i}": 100 - 10 * i for i in range(10)}  # c9=10 排名最末
    # 无持仓：c9 不应入选
    t_empty = select_targets(scores, set(), {})
    assert "c9" not in t_empty
    # 持仓 c9：排名在 slack 内（cutoff=8-1+3=10, idx(c9)=9 <=10）→ 保留
    t_held = select_targets(scores, {"c9"}, {})
    assert "c9" in t_held
    assert len(t_held) == 9  # 8 初选 + c9


def test_select_targets_empty_scores():
    assert select_targets({}, set(), {}) == {}

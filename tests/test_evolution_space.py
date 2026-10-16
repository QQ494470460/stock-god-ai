# -*- coding: utf-8 -*-
"""自我进化 · 参数空间与安全边界测试。

这是整套进化能力的**刹车片**。如果这里的断言松了，
系统就能自己把熔断线调高、把单票上限拉满 —— 相当于自己拆掉刹车。

覆盖四件事
----------
1. 风控黑名单：``guard.*`` / ``risk.*`` / 熔断线 / 单票上限 / 下单频率
   一律拒绝，不看白名单。
2. 白名单自洽：每条 :data:`PARAM_SPACE` 记录本身不得落在黑名单里，
   默认值必须落在自己的取值域内。
3. 取值域：类型转换、越界、枚举、``domain()``。
4. 采纳策略：朝更激进方向 / 标记不可自动采纳 → 必须人工批准。

全部测试只操作内存对象，不写任何文件。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evolution import space  # noqa: E402
from src.evolution.space import ParamError, ParamSpec  # noqa: E402


# --------------------------------------------------------------- 风控黑名单

@pytest.mark.parametrize("key", [
    "guard.max_portfolio_drawdown_pct",
    "guard.max_single_position_pct",
    "trading_guard.max_orders_per_minute",
    "risk.max_daily_loss_pct",
    "safety.kill_switch",
    "executor.max_orders_per_tick",
])
def test_protected_namespaces_are_rejected(key):
    """风控命名空间下的任何参数都不可进化（前缀即拒绝）。"""
    assert space.is_protected(key) is True
    with pytest.raises(ParamError, match="风控硬线"):
        space.get_spec(key)


@pytest.mark.parametrize("key", [
    # 裸叶子名（真实存在于 config.settings，但不许动）
    "trading_loop.max_orders_per_tick",
    "max_portfolio_drawdown_pct",
    "stop_loss_pct",
    "take_profit_pct",
    "trailing_stop_pct",
    "max_single_position_pct",
    "position_sizing_method",
    "require_live_confirmation",
    "default_mode",
])
def test_protected_leaves_are_rejected(key):
    """风控叶子名即便不带命名空间前缀，也必须被拦下。"""
    assert space.is_protected(key) is True
    with pytest.raises(ParamError):
        space.get_spec(key)


def test_ambiguous_bare_name_is_rejected():
    """裸 ``max_positions`` 语义不明（组合持仓数？单票持仓数？）→ 拒绝。

    带命名空间的 ``trading_loop.max_positions`` 是允许的，
    这正是「显式优于隐式」的落点。
    """
    assert space.is_protected("max_positions") is True
    with pytest.raises(ParamError):
        space.get_spec("max_positions")

    assert space.is_protected("trading_loop.max_positions") is False
    assert space.get_spec("trading_loop.max_positions").kind == "int"


def test_empty_key_is_protected():
    """空 key 视为非法而非「通配」。"""
    assert space.is_protected("") is True
    assert space.is_protected("   ") is True
    assert space.is_protected(None) is True


def test_unregistered_key_raises():
    """不在白名单里的参数同样拒绝（白名单是闭集）。"""
    with pytest.raises(ParamError, match="未登记"):
        space.get_spec("trading_loop.not_a_real_param")


# --------------------------------------------------------------- 白名单自洽

def test_space_is_never_self_contradictory():
    """白名单里的每一条都不得命中黑名单 —— 否则就是自相矛盾的配置。"""
    bad = [k for k in space.PARAM_SPACE if space.is_protected(k)]
    assert not bad, f"参数空间登记了受保护参数: {bad}"


def test_every_default_is_within_its_own_range():
    """默认值必须在自己的取值域内，否则第一次 clamp 就会改掉用户配置。"""
    for key, spec in space.PARAM_SPACE.items():
        assert spec.in_range(spec.default), f"{key} 默认值 {spec.default} 越界"
        assert spec.clamp(spec.default) == spec.default, f"{key} 默认值被 clamp 改写"


def test_specs_match_real_config_defaults():
    """参数空间里的默认值必须与 ``config.settings`` 的真实默认一致。

    否则「进化前的基线」本身就是错的，评估结果全部失真。
    """
    from config.settings import config as app_config

    for key, spec in space.PARAM_SPACE.items():
        if key.startswith("evolution.") or key.startswith("ml."):
            continue            # 这两个命名空间不在 AppConfig 上
        live = space.get_value(app_config, key)
        assert live == spec.default, (
            f"{key}: 参数空间默认 {spec.default!r} != 配置实际 {live!r}")


def test_protected_leaves_actually_exist_in_config():
    """黑名单不是摆设 —— 这些名字确实存在于真实配置里才值得保护。"""
    from config.settings import config as app_config

    assert hasattr(app_config.trading_loop, "max_orders_per_tick")
    assert hasattr(app_config.trading_loop, "max_positions")
    assert space.is_protected("trading_loop.max_orders_per_tick") is True


def test_risky_params_are_not_auto_adoptable():
    """凡是标了风险方向的参数，都不允许自动落地（双保险）。"""
    offenders = [k for k, s in space.PARAM_SPACE.items()
                 if s.risk != "neutral" and s.auto_adopt]
    assert not offenders, f"风险参数不允许自动采纳: {offenders}"


# --------------------------------------------------------------- 取值域

def test_cast_and_clamp():
    spec = space.get_spec("trading_loop.max_positions")   # int, 1~20
    assert spec.cast("12") == 12
    assert spec.clamp(999) == 20
    assert spec.clamp(-5) == 1
    assert spec.clamp(7.6) == 8          # int 取整

    fspec = space.get_spec("trading_loop.cash_reserve_pct")   # float, 0.0~0.5
    assert fspec.clamp(0.9) == 0.5
    assert fspec.clamp(-1) == 0.0


def test_cast_bool_from_string():
    spec = space.get_spec("trading_loop.enable_rebalance")
    assert spec.cast("true") is True
    assert spec.cast("0") is False
    assert spec.cast("yes") is True
    assert spec.cast(0) is False


def test_cast_bad_type_raises_param_error():
    with pytest.raises(ParamError):
        space.get_spec("trading_loop.max_positions").cast("十只")


def test_int_cast_rounds_instead_of_truncating():
    """回归：截断会把 7.999999999999999（浮点误差常见长相）打成 7。"""
    spec = space.get_spec("trading_loop.max_positions")
    assert spec.cast(7.999999999999999) == 8
    assert spec.cast(7.6) == 8
    assert spec.cast(19.5) == 20
    assert spec.cast("12") == 12
    # 与 clamp 的取整语义一致，不出现「同一路径两种取整」
    assert spec.clamp(7.6) == spec.cast(7.6)


def test_category_only_accepts_choices():
    spec = space.get_spec("trading_loop.weighting")     # equal | score
    assert spec.cast("score") == "score"
    with pytest.raises(ParamError, match="不在允许集合"):
        spec.cast("momentum")


def test_validate_rejects_out_of_range():
    with pytest.raises(ParamError, match="越界"):
        space.validate("trading_loop.band_pct", 5.0)
    assert space.validate("trading_loop.band_pct", 0.3) == 0.3


def test_coerce_clamps_instead_of_raising():
    """coerce 用于生成器纠偏：越界夹紧，不抛异常。"""
    assert space.coerce("trading_loop.max_positions", 100) == 20


def test_domain_generation():
    ints = space.sample_domain("trading_loop.max_positions")
    assert ints[0] == 1 and ints[-1] == 20
    assert all(isinstance(v, int) for v in ints)

    cats = space.sample_domain("trading_loop.weighting")
    assert set(cats) == {"equal", "score"}

    bools = space.sample_domain("trading_loop.enable_rebalance")
    assert set(bools) == {True, False}

    floats = space.sample_domain("trading_loop.band_pct")
    assert floats[0] == 0.05 and floats[-1] == 0.5


# --------------------------------------------------------------- 风险方向

def test_risk_direction_detection():
    up = space.get_spec("trading_loop.weight_cap")          # risky_up
    assert up.is_risk_increasing(0.20, 0.30) is True
    assert up.is_risk_increasing(0.30, 0.20) is False

    down = space.get_spec("trading_loop.cash_reserve_pct")  # risky_down
    assert down.is_risk_increasing(0.20, 0.05) is True
    assert down.is_risk_increasing(0.10, 0.20) is False

    neutral = space.get_spec("trading_loop.band_pct")
    assert neutral.is_risk_increasing(0.1, 0.5) is False
    assert neutral.is_risk_increasing(0.5, 0.1) is False


def test_risk_direction_ok_helper():
    assert space.risk_direction_ok("trading_loop.weight_cap", 0.30, 0.20) is True
    assert space.risk_direction_ok("trading_loop.weight_cap", 0.20, 0.30) is False


def test_unparseable_change_is_treated_as_risky():
    """说不清方向的改动按「有风险」处理 —— 保守优先。"""
    spec = space.get_spec("trading_loop.max_positions")
    assert spec.is_risk_increasing("abc", "def") is True


# --------------------------------------------------------------- 采纳策略

def test_neutral_autoadopt_change_is_adoptable():
    ok, why = space.check_changes_adoptable([
        ("trading_loop.band_pct", 0.25, 0.30),
    ])
    assert ok is True
    assert "可自动采纳" in why


def test_risk_increasing_change_requires_human():
    """朝更激进方向的改动必须人工批准 —— 这是本模块存在的理由。"""
    ok, why = space.check_changes_adoptable([
        ("trading_loop.weight_cap", 0.20, 0.30),
    ])
    assert ok is False
    assert "更激进" in why and "人工批准" in why


def test_auto_adopt_false_blocks_even_safe_direction():
    """``enable_stop_orders`` 是中性方向，但关掉止盈止损=放弃纪律 → 不许自动。"""
    ok, why = space.check_changes_adoptable([
        ("trading_loop.enable_stop_orders", True, False),
    ])
    assert ok is False
    assert "不可自动采纳" in why


def test_protected_change_is_rejected_by_adopt_check():
    ok, why = space.check_changes_adoptable([
        ("guard.max_portfolio_drawdown_pct", 0.08, 0.20),
    ])
    assert ok is False
    assert "风控硬线" in why


def test_empty_and_noop_changes_are_not_adoptable():
    ok, why = space.check_changes_adoptable([])
    assert ok is False and "无改动" in why

    ok, why = space.check_changes_adoptable([
        ("trading_loop.band_pct", 0.25, 0.25),
    ])
    assert ok is False and "相同" in why


def test_one_bad_apple_blocks_whole_batch():
    """一组改动里只要有一条越线，整批都不许自动落地（不部分采纳）。"""
    ok, _ = space.check_changes_adoptable([
        ("trading_loop.band_pct", 0.25, 0.30),        # 安全
        ("trading_loop.weight_cap", 0.20, 0.30),      # 激进
    ])
    assert ok is False


# --------------------------------------------------------------- 描述输出

def test_describe_space_lists_params_and_protections():
    text = space.describe_space()
    assert "可进化参数" in text
    assert "trading_loop.band_pct" in text
    assert "永久受保护" in text
    assert "max_portfolio_drawdown_pct" in text


# --------------------------------------------------------------- ParamSpec 边界

def test_bool_and_category_clamp_is_identity():
    bspec = ParamSpec("x", "bool", True)
    assert bspec.clamp(True) is True
    cspec = ParamSpec("y", "category", "a", choices=("a", "b"))
    assert cspec.clamp("b") == "b"


def test_frozen_spec_cannot_be_mutated():
    """规格是常量，不允许运行期被改写（防「顺手改一下范围」）。"""
    spec = space.get_spec("trading_loop.band_pct")
    with pytest.raises(Exception):
        spec.high = 99.0        # frozen dataclass

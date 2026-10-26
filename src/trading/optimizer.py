# -*- coding: utf-8 -*-
"""组合权重优化器 —— 把「风险预算」从约束变成决策变量。

为什么需要它
------------
在这之前，计划层只有两种权重：``equal``（等权）和 ``score``（按评分加权）。
两者都只看「alpha」，完全无视各标的的波动与相关性，结果是：

- 等权：一只年化波动 25% 的银行股和一只 70% 的题材股拿一样多的钱，
  组合的风险贡献 8 成来自后者 —— **名义分散、实质单押**。
- 按评分加权：高分票往往就是高波动票（模型偏爱动量），再给它更大权重，
  等于**在波动最大的地方下最重的注**。

本模块补上另一半：给定协方差矩阵，求「每只票贡献相同风险」的权重
（风险平价 ERC）、或「组合方差最小」的权重（最小方差），再可选地用
alpha 评分做一次温和倾斜。这样计划层的权重与风控层的 VaR 预算就同源了，
不会出现「计划要 45%、VaR 一算超预算、执行端再砍一半」的来回打架。

约定
----
- 全部纯函数、只依赖 numpy，不读盘不联网；输入非法一律返回等权（fail-safe）。
- 权重恒为 ``sum=1`` 的「可投部分」口径，由调用方乘 ``investable_pct``。
- 长仓、无杠杆：所有权重 >= 0。
- ``cap`` 单票上限是**软约束**：先求无约束解再迭代截断+再归一，
  直到满足或达到迭代上限；不可行（n*cap < 1）时直接放弃 cap 并记警告。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_CAP = 1.0


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _cov_array(cov) -> Optional[np.ndarray]:
    """把输入安全转成 (n,n) 对称方阵；非法返回 None。"""
    try:
        c = np.asarray(cov, dtype=float)
    except (TypeError, ValueError):
        return None
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        return None
    if not np.all(np.isfinite(c)):
        return None
    # 数值上强制对称，避免外部喂进来的半正定矩阵因浮点误差出负方差
    return (c + c.T) / 2.0


def _normalize(w: np.ndarray) -> np.ndarray:
    """非负归一；全零或非法退化为等权。"""
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = float(w.sum())
    if s <= 0:
        n = w.size or 1
        return np.full(n, 1.0 / n)
    return w / s


def portfolio_variance(weights: Sequence[float], cov) -> float:
    """组合方差 ``wᵀΣw``（非法输入返回 0）。"""
    c = _cov_array(cov)
    if c is None:
        return 0.0
    w = np.asarray(weights, dtype=float)
    if w.size != c.shape[0]:
        return 0.0
    return float(w @ c @ w)


def risk_contributions(weights: Sequence[float], cov) -> np.ndarray:
    """各标的风险贡献 ``w_i · (Σw)_i``（绝对贡献，单位=方差）。

    风险平价的目标是让它全部相等；UI 里也用它画「谁在贡献风险」。
    """
    c = _cov_array(cov)
    w = np.asarray(weights, dtype=float)
    if c is None or w.size != c.shape[0]:
        return np.zeros(w.size if w.ndim else 0)
    return w * (c @ w)


def _apply_cap(w: np.ndarray, cap: Optional[float], max_iter: int = 12) -> np.ndarray:
    """把权重压到 ``w_i <= cap`` 并保持 sum=1（迭代「截顶 → 剩下的按比例重分」）。"""
    if cap is None or not np.isfinite(cap) or cap <= 0 or cap >= 1.0:
        return w
    n = w.size
    if n * cap < 1.0 - 1e-12:
        # 不可行：n 只票都顶格也凑不满 100%，放弃 cap（宁可超限也不留无谓现金）
        return w
    out = w.copy()
    for _ in range(max_iter):
        over = out > cap + 1e-15
        if not over.any():
            break
        out[over] = cap
        free = ~over
        if not free.any():
            break
        rest = 1.0 - cap * int(over.sum())
        base = float(out[free].sum())
        if base <= 0:
            out[free] = rest / int(free.sum())
        else:
            out[free] *= rest / base
    return _normalize(out)


# ---------------------------------------------------------------------------
# 风险平价（等风险贡献 ERC）
# ---------------------------------------------------------------------------
def risk_parity_weights(
    cov,
    cap: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """等风险贡献权重（Equal Risk Contribution / 风险平价）。

    用经典乘性不动点迭代：``w_i ← w_i · (rc_target / rc_i)^{1/2}``。
    对冲标的（负相关）也能收敛，且不需要任何 QP 求解器。

    Returns:
        长度 n 的权重向量，sum=1；协方差非法时返回等权。
    """
    c = _cov_array(cov)
    if c is None:
        return np.full(1, 1.0)
    n = c.shape[0]
    diag = np.clip(np.diag(c), 1e-16, None)
    w = _normalize(1.0 / np.sqrt(diag))  # 逆波动率初始化：天然很接近解

    for _ in range(int(max_iter)):
        rc = w * (c @ w)
        total = float(rc.sum())
        if total <= 0:
            w = np.full(n, 1.0 / n)
            break
        target = total / n
        rc_safe = np.clip(rc, 1e-18, None)
        ratio = target / rc_safe
        w = w * np.sqrt(np.clip(ratio, 1e-6, 1e6))
        w = _normalize(w)
        # 收敛判据：各标的风险贡献相对目标的偏离都进了 tol
        if float(np.max(np.abs(rc_safe / target - 1.0))) < tol:
            break

    return _apply_cap(_normalize(w), cap)


# ---------------------------------------------------------------------------
# 最小方差 / 波动率目标
# ---------------------------------------------------------------------------
def min_variance_weights(
    cov,
    cap: Optional[float] = None,
    max_iter: int = 800,
    lr: float = 0.5,
) -> np.ndarray:
    """长仓最小方差权重（投影梯度下降，无需 QP 求解器）。

    梯度为 ``2Σw``；步长用 ``1/(2·λmax)`` 的上界再打折，保证不发散。
    """
    c = _cov_array(cov)
    if c is None:
        return np.full(1, 1.0)
    n = c.shape[0]
    # 逆方差初始化（最小方差在 Σ 对角时的解析解）
    w = _normalize(1.0 / np.clip(np.diag(c), 1e-16, None))

    try:
        eig_max = float(np.max(np.linalg.eigvalsh(c)))
    except np.linalg.LinAlgError:
        eig_max = float(np.trace(c)) or 1.0
    step = lr / max(2.0 * max(eig_max, 1e-12), 1e-12)

    prev = None
    for _ in range(int(max_iter)):
        grad = 2.0 * (c @ w)
        w = w - step * grad
        w = np.where(w > 0, w, 0.0)      # 长仓约束
        w = _normalize(w)
        if prev is not None and float(np.max(np.abs(w - prev))) < 1e-12:
            break
        prev = w.copy()

    return _apply_cap(w, cap)


def vol_target_weights(
    vols: Sequence[float],
    target_vol: Optional[float] = None,
    cap: Optional[float] = None,
    periods_per_year: int = 242,
) -> np.ndarray:
    """逆波动率权重；给了 ``target_vol`` 就整体缩放到该年化波动目标。

    逆波动率是「忽略相关性」的风险平价近似，但只需要个股波动、不需要
    协方差矩阵（协方差在标的少/历史短时估计噪声极大）。
    缩放后 ``sum(w)`` 可以 < 1 —— 剩下的是现金，这正是波动率目标法的本意：
    **波动率高的时候自动少持仓**，而不是永远满仓。

    Args:
        vols: 各标的**日**波动率（非年化）；<=0 的按缺失处理
        target_vol: 组合年化波动率目标（如 0.15）；None 则不做缩放
        cap: 单票上限（缩放前先截断，避免靠缩放掩盖集中度）
    """
    v = np.asarray(vols, dtype=float)
    if v.size == 0:
        return np.zeros(0)
    ok = np.isfinite(v) & (v > 0)
    if not ok.any():
        return np.full(v.size, 1.0 / v.size)
    inv = np.where(ok, 1.0 / np.where(ok, v, 1.0), 0.0)
    w = _normalize(inv)

    if target_vol is not None and np.isfinite(target_vol) and target_vol > 0:
        # 组合年化波动 ≈ sqrt(wᵀ diag(v²) w) × sqrt(242)
        port_daily = float(np.sqrt(np.sum((w * np.where(ok, v, 0.0)) ** 2)))
        port_annual = port_daily * np.sqrt(max(int(periods_per_year), 1))
        if port_annual > 0:
            scale = min(1.0, float(target_vol) / port_annual)
            # 先按 cap 截断可投比例，再缩放；两步都做，避免「靠缩小总仓位
            # 让单票名义上不超限」的假合规。
            w = _apply_cap(w * scale, cap) if scale >= 1.0 else w * scale
            if scale < 1.0 and cap is not None:
                w = np.minimum(w, float(cap))
    return np.asarray(w, dtype=float)


# ---------------------------------------------------------------------------
# alpha 倾斜 / 换手封顶
# ---------------------------------------------------------------------------
def alpha_tilt(
    weights: Sequence[float],
    scores: Sequence[float],
    strength: float = 0.25,
    cap: Optional[float] = None,
) -> np.ndarray:
    """在风险平价权重上叠一层温和的 alpha 倾斜。

    用**秩**而不是原始评分做倾斜系数，是为了抗极值：模型偶尔给出一个
    99 分而其余都是 60 分时，按原始值加权会把 9 成资金压在一只票上；
    按秩则只是「最好的那只比中位数多拿 strength 比例」。

    ``strength=0.25`` 表示最好的一只权重 ×1.25、最差的一只 ×0.75。

    为什么默认只到 0.25（而不是 0.5）：秩空间里 2 只票就是首尾两端，
    strength=0.5 会给出 1.5/0.5 的 3 倍乘子，足以把刚算好的风险平价完全
    抹平 —— 「风险平价 + 倾斜」退化成「按评分加权」，白算一遍协方差。
    """
    w = _normalize(np.asarray(weights, dtype=float))
    s = np.asarray(scores, dtype=float)
    n = w.size
    if n == 0 or s.size != n:
        return w
    s = np.where(np.isfinite(s), s, np.nanmin(s) if np.isfinite(s).any() else 0.0)

    # 秩 → [-1, 1]，中点 0
    order = np.argsort(np.argsort(s))
    denom = max(n - 1, 1)
    rank = (order / denom) * 2.0 - 1.0 if n > 1 else np.zeros(n)
    mult = 1.0 + max(min(float(strength), 1.0), 0.0) * rank

    return _apply_cap(_normalize(w * np.clip(mult, 0.05, None)), cap)


def cap_turnover(
    old_weights: Sequence[float],
    new_weights: Sequence[float],
    max_turnover: Optional[float] = 0.5,
) -> np.ndarray:
    """限制单次调仓的名义换手率。

    换手率定义为 ``0.5·Σ|w_new - w_old|``（单边口径，与行业惯例一致）。
    超限时不是「拒绝调仓」，而是把新权重向旧权重**线性插值**——
    渐进逼近目标，代价是慢一点，收益是省掉一大笔双边成本。
    """
    o = _normalize(np.asarray(old_weights, dtype=float))
    n_w = np.asarray(new_weights, dtype=float)
    if o.size != n_w.size:
        return _normalize(n_w)
    n_w = _normalize(n_w)
    if max_turnover is None or not np.isfinite(max_turnover) or max_turnover <= 0:
        return n_w

    turnover = 0.5 * float(np.sum(np.abs(n_w - o)))
    if turnover <= max_turnover:
        return n_w
    # δ 使插值后的换手恰好等于上限
    alpha = max_turnover / turnover
    return _normalize(o + alpha * (n_w - o))


# ---------------------------------------------------------------------------
# 一站式入口（给 trade_plan / 脚本用）
# ---------------------------------------------------------------------------
def optimize_weights(
    cov=None,
    vols: Optional[Sequence[float]] = None,
    scores: Optional[Sequence[float]] = None,
    method: str = "risk_parity",
    cap: Optional[float] = None,
    target_vol: Optional[float] = None,
    tilt_strength: float = 0.25,
    old_weights: Optional[Sequence[float]] = None,
    max_turnover: Optional[float] = None,
) -> np.ndarray:
    """统一的权重求解入口（fail-safe：出任何问题都退化为等权）。

    method:
      - ``risk_parity``  : 等风险贡献（需要 cov）
      - ``min_variance`` : 组合方差最小（需要 cov）
      - ``vol_target``   : 逆波动率 + 目标波动缩放（需要 vols）
      - ``equal``        : 等权
    """
    n = 0
    if cov is not None:
        c = _cov_array(cov)
        n = 0 if c is None else c.shape[0]
    elif vols is not None:
        n = len(vols)

    try:
        if method == "risk_parity" and cov is not None:
            w = risk_parity_weights(cov, cap=cap)
        elif method == "min_variance" and cov is not None:
            w = min_variance_weights(cov, cap=cap)
        elif method == "vol_target" and vols is not None:
            w = vol_target_weights(vols, target_vol=target_vol, cap=cap)
        else:
            w = np.full(n or 1, 1.0 / (n or 1))

        total_before = float(np.sum(w))
        if scores is not None and len(scores) == len(w) and tilt_strength > 0:
            w = alpha_tilt(w, scores, strength=tilt_strength, cap=cap)
            # alpha_tilt 会把权重归一化到 1；但 vol_target 的意义就在于
            # 「总仓位可以 < 1（剩下的留现金）」。倾斜只能在已投部分内部重分配，
            # 不能顺手把仓位补满 —— 否则波动率目标法被倾斜悄悄废掉。
            if method == "vol_target" and total_before > 0:
                cur = float(np.sum(w))
                if cur > 0:
                    w = w * (total_before / cur)
        if old_weights is not None and len(old_weights) == len(w):
            w = cap_turnover(old_weights, w, max_turnover)
        return np.asarray(w, dtype=float)
    except Exception:  # noqa: BLE001
        return np.full(n or 1, 1.0 / (n or 1))


def explain(weights: Sequence[float], cov=None, codes: Optional[Sequence[str]] = None) -> List[Dict]:
    """把权重与风险贡献整理成可展示的行（UI / 日志用）。"""
    w = np.asarray(weights, dtype=float)
    codes = list(codes or [f"#{i}" for i in range(w.size)])
    rc = risk_contributions(w, cov) if cov is not None else np.zeros(w.size)
    total_rc = float(np.sum(rc))
    rows: List[Dict] = []
    for i, code in enumerate(codes[: w.size]):
        rows.append({
            "code": code,
            "weight": round(float(w[i]), 4),
            "risk_contrib": round(float(rc[i]), 6),
            "risk_pct": round(float(rc[i]) / total_rc, 4) if total_rc > 0 else 0.0,
        })
    rows.sort(key=lambda r: r["risk_pct"], reverse=True)
    return rows


def weights_to_dict(weights: Sequence[float], codes: Sequence[str]) -> Dict[str, float]:
    """weights + codes → {code: weight}（截断到权重长度）。"""
    return {str(c): round(float(w), 6) for c, w in zip(codes, weights)}

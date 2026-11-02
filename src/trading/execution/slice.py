# -*- coding: utf-8 -*-
"""订单切片（TWAP / VWAP）与切片数选择。

纯函数、无副作用、不碰网络，便于单测。所有数量按整手（lot=100）取整。
"""
from __future__ import annotations

import math
from typing import List, Sequence

LOT_SIZE = 100


def choose_n_slices(total_qty: int, max_slices: int, lot_size: int = LOT_SIZE) -> int:
    """根据总量与单手上限，选出既不超过 max_slices、又保证每片 ≥ 1 手的切片数。

    逻辑：先按 max_slices 切，若每片不足 1 手则逐档下调，最少 1 片。
    """
    if total_qty <= 0 or max_slices <= 1:
        return 1
    n = max(1, int(max_slices))
    while n > 1 and (total_qty // n) < lot_size:
        n -= 1
    return n


def _lot_round_down(qty: int, lot_size: int) -> int:
    if qty <= 0 or lot_size <= 0:
        return 0
    return int(math.floor(qty / lot_size) * lot_size)


def twap_slices(total_qty: int, n_slices: int, lot_size: int = LOT_SIZE) -> List[int]:
    """把 ``total_qty`` 均分为 ``n_slices`` 片，逐片整手取整，余数补到前若干片。

    返回长度 = n_slices，各项为整手股数，求和 = ``total_qty``（向下取整到整手后的最大值）。
    """
    if n_slices <= 1:
        return [_lot_round_down(total_qty, lot_size)]
    base = _lot_round_down(total_qty // n_slices, lot_size)
    out = [base] * n_slices
    # 余数（无法整除的零散股）忽略（A股整手约束），保证每片合法
    assigned = base * n_slices
    rem = _lot_round_down(total_qty, lot_size) - assigned
    i = 0
    while rem >= lot_size and i < n_slices:
        out[i] += lot_size
        rem -= lot_size
        i += 1
    return out


def vwap_slices(
    total_qty: int,
    profile: Sequence[float],
    lot_size: int = LOT_SIZE,
) -> List[int]:
    """按成交量分布 ``profile``（各片权重，无需归一）分配总量。

    典型 intraday 呈 U 形（开盘/收盘放量、午间缩量）。返回整手股数，求和≈总量。
    """
    n = len(profile)
    if n <= 1:
        return [_lot_round_down(total_qty, lot_size)]
    s = sum(p for p in profile if p > 0) or 1.0
    out = []
    for w in profile:
        q = int(round(total_qty * max(w, 0.0) / s))
        out.append(_lot_round_down(q, lot_size))
    # 校准：把总量与各片之和的差额补到最大的一片（仍在整手约束内）
    diff = _lot_round_down(total_qty, lot_size) - sum(out)
    if diff != 0 and out:
        idx = max(range(n), key=lambda i: out[i])
        out[idx] = max(0, out[idx] + diff)
    return out


def default_vwap_profile(n_slices: int) -> List[float]:
    """生成 U 形日内量分布（开盘/收盘重、中间轻），用于 VWAP 切片。"""
    if n_slices <= 1:
        return [1.0]
    prof = []
    for i in range(n_slices):
        # 到两端的距离越近权重越大
        x = i / (n_slices - 1)
        prof.append(0.5 + 0.5 * (2 * abs(x - 0.5)) ** 1.5)
    return prof

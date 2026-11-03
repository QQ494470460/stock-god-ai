# -*- coding: utf-8 -*-
"""行业映射补齐脚本的合并语义。

为什么这两条值得单测
--------------------
``data/industry_map.json`` 只覆盖 2621 只，而 A 股约 5400 只 —— 也就是说
**一半的票没有行业**。而这张表同时喂三个地方：

  · ``guard`` 的行业集中度检查（未映射票 fail-open ⇒ 集中度上限对它免检）
  · ``predictor`` 的行业中性化（未映射票被归入一个桶，实测被整体平移 +0.020 分）
  · 回测里的 ``max_sector_pct``（同样 fail-open）

2026-11-04 查到两个让这张表**永远补不满**的原因：

1) 脚本用 ``len(have) >= 2500`` 判断「已完成」。2500 本是「别把抓了一半的空表
   落盘毒化缓存」的下限，却被当成了完工信号 —— 于是系统停在半个市场就宣布 ok，
   再也不会重试。这就是 ``--target``（默认 4800）与 ``--min``（落盘门槛）分开的原因。
2) 落盘是**整表覆盖**：只要某次抓到的 2500 只与现有 2621 只不重叠，就会丢掉
   已经攒下的映射，而且过程看起来完全成功。东财快照按页返回，部分重叠确实会发生。

所以这里钉的是「只增不减」和「目标与门槛分离」这两条语义 —— 它们不联网也能测。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from scripts.build_industry_map import merge_industry_maps  # noqa: E402


def test_merge_keeps_everything_already_known():
    """新抓到的部分结果绝不能替换掉已有的映射。"""
    have = {"600000": "银行", "600001": "白酒", "600002": "半导体"}
    fresh = {"600003": "元件"}                    # 只有一只新的
    merged = merge_industry_maps(have, fresh)
    assert merged == {**have, **fresh}


def test_merge_never_shrinks_even_when_the_fetch_is_disjoint():
    """抓到的是完全不同的一批票，也必须变成并集（这就是「半张表换半张表」的修正）。"""
    have = {"600000": "银行", "600001": "白酒"}
    fresh = {"300001": "元件", "300002": "软件"}
    merged = merge_industry_maps(have, fresh)
    assert len(merged) == 4
    assert all(k in merged for k in (*have, *fresh))


def test_merge_ignores_empty_values_so_a_bad_page_cannot_erase_data():
    """空字符串/None 不能覆盖已知行业 —— 那是「静默丢数据」最典型的形态。"""
    have = {"600000": "银行", "600001": "白酒"}
    fresh = {"600000": "", "600001": None, "600002": "   "}
    assert merge_industry_maps(have, fresh) == have


def test_merge_strips_whitespace_and_stringifies_keys():
    merged = merge_industry_maps({}, {600000: " 银行 ", "600001": "白酒"})
    assert merged == {"600000": "银行", "600001": "白酒"}


def test_merge_tolerates_empty_inputs():
    assert merge_industry_maps({}, {}) == {}
    assert merge_industry_maps(None, {"600000": "银行"}) == {"600000": "银行"}
    assert merge_industry_maps({"600000": "银行"}, None) == {"600000": "银行"}


def test_target_default_is_an_honest_full_market_number():
    """「够齐了」的目标必须接近全 A 股，不能退回到落盘门槛 2500。

    这条断言防的是一个很容易「顺手改回去」的诱惑：把 target 调小能让脚本
    立刻报成功。但 2500 ≈ 半个市场，等于宣称「一半的票不需要行业」——
    而那正是集中度上限变成半瞎的原因。
    """
    import inspect

    import scripts.build_industry_map as mod

    src = inspect.getsource(mod.main)
    assert "--target" in src
    assert "default=4800" in src, "target 默认值被改小了：先读本文件头注"
    assert "default=2500" not in src, "min 与 target 又被合并成一个数了"

# -*- coding: utf-8 -*-
"""get_stock_pool 未知池名「失败要响亮」测试。

历史坑：池名写错（如 'test'）时旧实现静默退回全市场 ≈5000 只，触发一次 9 分钟
的扫盘。现改为直接抛 ValueError。这里把该纪律钉死。
"""

import pandas as pd
import pytest

from src.data import fetcher as fmod


def _fake_stock_list():
    return pd.DataFrame({
        "code": ["000001", "600519", "300750", "688001"],
        "name": ["平安银行", "贵州茅台", "宁德时代", "华兴源创"],
    })


def test_unknown_pool_raises_fail_loud():
    """未知池名必须抛 ValueError（不得静默退回全市场）。"""
    f = fmod.fetcher
    with pytest.raises(ValueError) as ei:
        f.get_stock_pool("nosuchpool")
    assert "未知股票池" in str(ei.value)


def test_none_empty_defaults_to_all(monkeypatch):
    """None / 空串 / 大小写 / 空格 都应归一化为 all（向后兼容）。"""
    f = fmod.fetcher
    monkeypatch.setattr(f, "get_stock_list", lambda *a, **k: _fake_stock_list())
    for p in (None, "", "ALL", " all "):
        f._pool_cache.pop("all", None)
        codes = f.get_stock_pool(p)
        assert set(codes) == {"000001", "600519", "300750", "688001"}


def test_st_filter_still_applied(monkeypatch):
    """正常路径不回归：ST/退市仍被过滤。"""
    f = fmod.fetcher
    df = pd.DataFrame({"code": ["000001", "600519"], "name": ["ST平安", "贵州茅台"]})
    monkeypatch.setattr(f, "get_stock_list", lambda *a, **k: df)
    f._pool_cache.pop("all", None)
    assert f.get_stock_pool("all") == ["600519"]

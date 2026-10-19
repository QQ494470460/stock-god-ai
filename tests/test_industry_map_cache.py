# -*- coding: utf-8 -*-
"""``fetcher.get_industry_map`` 的缓存回归测试（针对"空结果投毒"）。

真实事故（2026-10-19）
=====================
本机访问东财 ``17.push2.eastmoney.com`` 直接 ProxyError，行业映射一条都抓不到。
旧实现把拿到的 ``{}`` **原样写进** ``data/industry_map.json``，此后每次调用都
命中这个空缓存直接返回 —— 即使哪天网络恢复，也**永远不会自愈**，
``ml.cross_sectional_neutralize`` 只能一直降级到板级分组。

这里钉死两条修复：
  1) 抓不到（或只抓到零散几条）时**不许覆盖**已有的好缓存；
  2) 失败要记进负缓存，TTL 窗口内不再重试 —— 否则一次选股要串行打上百个
     注定失败的请求。
"""

import json
import time
from pathlib import Path

import src.data.fetcher as fm

MIN_OK = fm.DataFetcher._MIN_INDUSTRY_CODES


def _setup(tmp_path, monkeypatch, initial=None):
    """把单例的缓存目录指向 tmp_path，并清空内存缓存。"""
    monkeypatch.setattr(fm.fetcher, "cache_dir", Path(tmp_path), raising=False)
    monkeypatch.setattr(fm.fetcher, "_industry_map", None, raising=False)
    if initial is not None:
        (Path(tmp_path) / "industry_map.json").write_text(
            json.dumps(initial, ensure_ascii=False), encoding="utf-8")
    return Path(tmp_path) / "industry_map.json"


def _block_network(monkeypatch):
    """模拟东财接口不通（代理拦截）。"""
    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        raise RuntimeError("ProxyError: 17.push2.eastmoney.com")

    monkeypatch.setattr(fm, "_ak", _boom, raising=False)
    return calls


def test_empty_result_does_not_poison_existing_cache(tmp_path, monkeypatch):
    """接口挂了也必须保住已有的好缓存，绝不能拿 {} 覆盖。"""
    good = {f"60{i:04d}": "银行" for i in range(MIN_OK + 10)}
    cache = _setup(tmp_path, monkeypatch, initial=good)
    _block_network(monkeypatch)

    got = fm.fetcher.get_industry_map(force_refresh=True)

    assert got == {}, "接口不通时返回空是正常的"
    assert json.loads(cache.read_text(encoding="utf-8")) == good, (
        "空结果把已有缓存覆盖了 —— 网络恢复后永远不会自愈"
    )


def test_failure_is_negative_cached_no_retry_within_ttl(tmp_path, monkeypatch):
    """失败要落负缓存：TTL 窗口内不再重复打网络。"""
    cache = _setup(tmp_path, monkeypatch)
    calls = _block_network(monkeypatch)

    fm.fetcher.get_industry_map(force_refresh=True)
    assert cache.exists(), "首次失败应当留下负缓存文件"
    assert calls["n"] == 1

    monkeypatch.setattr(fm.fetcher, "_industry_map", None, raising=False)
    fm.fetcher.get_industry_map(force_refresh=False)
    assert calls["n"] == 1, "负缓存窗口内又去打网络了 —— 会把选股流程拖死"


def test_negative_cache_expires_and_retries(tmp_path, monkeypatch):
    """过了 TTL 允许重试（否则网络恢复后仍然永不相见）。"""
    cache = _setup(tmp_path, monkeypatch)
    calls = _block_network(monkeypatch)

    fm.fetcher.get_industry_map(force_refresh=True)
    # 把负缓存的 mtime 推到 TTL 之外
    old = time.time() - (fm.DataFetcher._INDUSTRY_NEG_TTL_H + 1) * 3600
    import os
    os.utime(cache, (old, old))

    monkeypatch.setattr(fm.fetcher, "_industry_map", None, raising=False)
    fm.fetcher.get_industry_map(force_refresh=False)
    assert calls["n"] == 2, "负缓存过期后应当重试一次"


def test_good_cache_is_used_without_network(tmp_path, monkeypatch):
    """有好缓存时直接命中，一次网络都不打。"""
    good = {f"00{i:04d}": "医药" for i in range(MIN_OK + 5)}
    _setup(tmp_path, monkeypatch, initial=good)
    calls = _block_network(monkeypatch)

    got = fm.fetcher.get_industry_map(force_refresh=False)
    assert got == good
    assert calls["n"] == 0

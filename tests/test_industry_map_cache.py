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
    """模拟东财接口不通（代理拦截）。

    ⚠️ 两条路都要堵：逐板块循环走 ``_ak()``，全市场快照走 ``requests.get``。
    只堵前者的话，新加的快照路径会在测试里真去打网络（12 页 × 12s 超时）。
    """
    calls = {"n": 0}

    def _boom(*a, **kw):
        calls["n"] += 1          # 只统计逐板块路线（历史用例按这个口径断言）
        raise RuntimeError("ProxyError: 17.push2.eastmoney.com")

    def _http_boom(*a, **kw):
        raise RuntimeError("ProxyError: 17.push2.eastmoney.com")

    monkeypatch.setattr(fm, "_ak", _boom, raising=False)
    monkeypatch.setattr("requests.get", _http_boom, raising=True)
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


# ---------------------------------------------------------------------------
# 全市场快照路线（2026-10-19 新增）
# ---------------------------------------------------------------------------
# 背景：本机到东财是**间歇性**可达 —— 实测某两分钟里能拿 200 OK（496 个行业板块），
# 之后连续 ProxyError / ConnectionReset。原「列板块 + 逐板块拉成分股」要上百次请求，
# 几乎不可能全部撞在可用窗口内；改成翻几页列表接口（自带 f100 所属行业）后，
# 只要撞到一个窗口就能补齐全市场，并永久落盘。

def _fake_pages(pages, total):
    """伪造 requests.get：按页码返回对应行，页码超出后返回空表。"""

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    def _get(url, params=None, timeout=None, **kwargs):
        pn = int((params or {}).get("pn", 1))
        rows = pages.get(pn, [])
        return _Resp({"data": {"total": total, "diff": rows}})

    return _get


def _rows(n, start=0, industry="银行"):
    return [{"f12": f"{600000 + start + i}", "f14": f"股票{i}", "f100": industry}
            for i in range(n)]


def test_bulk_snapshot_is_accepted_and_persisted(tmp_path, monkeypatch):
    """覆盖够了就采用并落盘，而且一次都不用去打板块循环。"""
    cache = _setup(tmp_path, monkeypatch)
    akshare_calls = _block_network(monkeypatch)          # 先全堵
    need = fm.DataFetcher._INDUSTRY_BULK_MIN_CODES
    pages = {1: _rows(1000), 2: _rows(1000, 1000), 3: _rows(need - 2000 + 10, 2000)}
    monkeypatch.setattr("requests.get", _fake_pages(pages, 5400), raising=True)

    got = fm.fetcher.get_industry_map(force_refresh=True)

    assert len(got) >= need, f"覆盖够的快照应当被采用，实际 {len(got)}"
    assert got["600000"] == "银行"
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert len(saved) == len(got), "采用后必须落盘（否则下次还要赌网络）"
    assert akshare_calls["n"] == 0, "快照够用时不该再走上百次请求的板块循环"


def test_partial_snapshot_is_rejected(tmp_path, monkeypatch):
    """覆盖不足的半张表**不许**采用 —— 中性化过的分和原始分混排名比板级更糟。"""
    cache = _setup(tmp_path, monkeypatch)
    _block_network(monkeypatch)
    monkeypatch.setattr("requests.get", _fake_pages({1: _rows(300)}, 5400), raising=True)

    got = fm.fetcher.get_industry_map(force_refresh=True)

    assert got == {}, "半张表（300/5400）必须判为失败，不能采用"
    assert json.loads(cache.read_text(encoding="utf-8")) == {}, "失败的快照不许污染缓存"


def test_snapshot_failure_falls_back_to_board_loop(tmp_path, monkeypatch):
    """快照这条路彻底不通时，仍要退回原来的逐板块路线（不抛异常）。"""
    _setup(tmp_path, monkeypatch)
    calls = _block_network(monkeypatch)                   # requests 也会炸
    import pandas as pd

    class _FakeAk:
        @staticmethod
        def stock_board_industry_name_em():
            return pd.DataFrame({"板块名称": ["银行"]})

        @staticmethod
        def stock_board_industry_cons_em(symbol):
            return pd.DataFrame({"代码": [f"60{i:04d}" for i in range(30)]})

    monkeypatch.setattr(fm, "_ak", lambda: _FakeAk, raising=False)

    got = fm.fetcher.get_industry_map(force_refresh=True)
    assert len(got) == 30, f"快照失败后应回退板块循环，实际 {len(got)}"


def test_bulk_snapshot_respects_overall_deadline(tmp_path, monkeypatch):
    """网络全挂时不能在选股作业里干等超时 —— 总时限一到立刻收工。

    12 页 × 2 次（直连+代理）× 12s 超时 ≈ 5 分钟，18:45 的 daily_pick 会被拖死。
    """
    import time as _t

    pages_seen = {"n": 0}

    def _slow_get(url, params=None, timeout=None, **kwargs):
        pages_seen["n"] += 1
        _t.sleep(0.05)                      # 每页都要花点时间
        pn = int((params or {}).get("pn", 1))

        class _Resp:
            @staticmethod
            def json():
                # total 永远够大，逼它一直翻页，只能靠时限停下来
                return {"data": {"total": 5400,
                                 "diff": _rows(1000, start=pn * 1000)}}

        return _Resp()

    monkeypatch.setattr("requests.get", _slow_get, raising=True)
    got = fm.fetcher._fetch_industry_map_bulk(max_seconds=0.2, timeout=5)

    assert 0 < pages_seen["n"] < 12, f"总时限没生效，翻了 {pages_seen['n']} 页"
    assert len(got) > 0, "到时限也要把已取到的部分交出去"

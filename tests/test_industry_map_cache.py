# -*- coding: utf-8 -*-
"""``fetcher.get_industry_map`` 增量自愈 / 显式降级 回归测试。

真实事故（2026-10-19 ~ 10-20）
=============================
本机到东财 ``push2.eastmoney.com`` 被代理**彻底封锁**（直连/代理均 ``ProxyError``，
已实测），东财 f100 全市场快照只能在极短暂的「窗口」里抓到，THS/csindex 成分股接口
在本机 AKShare 版本不可达/需鉴权。

由此带来两套必须钉死的行为：

旧坑（已修）：抓不到时把 ``{}`` 原样写进 ``industry_map.json``，此后每次命中空缓存
直接返回 —— 即使网络恢复也**永远不会自愈**；且覆盖不足时**静默**降级到板级中性化，
出问题时无从排查。

新契约（2026-10-20 重写）：
  1) **增量自愈**：已有缓存永远当基底，每轮成功就 update+落盘，多轮抢窗口、跨小时/跨天
     累计，绝不再「一次全失败就清零」；
  2) **非空缓存永远复用**，失败只落负缓存并在 TTL 窗口内不再重试（否则一次选股要串行
     打上百个注定失败的请求，把流程拖死）；
  3) **显式状态**：无论成败都写 ``industry_map_status.json``（coverage/source/note），
     覆盖不足时打 WARNING 说明根因与恢复方式，替代静默跳过；
  4) **THS 行业 taxonomy 仅作诊断缓存**到 ``industry_boards_ths.json``，绝不掺进
     ``industry_map.json``，否则会污染中性化。
"""

import json
import time as _real_time
from pathlib import Path

import src.data.fetcher as fm

MIN_OK = fm.DataFetcher._MIN_INDUSTRY_CODES
NEED = fm.DataFetcher._INDUSTRY_BULK_MIN_CODES
FLOOR = fm.DataFetcher._INDUSTRY_ACCEPT_FLOOR


def _setup(tmp_path, monkeypatch, initial=None):
    """把单例的缓存目录指向 tmp_path，并清空内存缓存；stub 掉 time.sleep 加速。"""
    monkeypatch.setattr(fm.fetcher, "cache_dir", Path(tmp_path), raising=False)
    monkeypatch.setattr(fm.fetcher, "_industry_map", None, raising=False)
    monkeypatch.setattr(_real_time, "sleep", lambda *a, **k: None, raising=True)
    if initial is not None:
        (Path(tmp_path) / "industry_map.json").write_text(
            json.dumps(initial, ensure_ascii=False), encoding="utf-8")
    return Path(tmp_path) / "industry_map.json"


def _block_network(monkeypatch):
    """模拟东财接口不通（代理拦截）。

    ⚠️ 两条路都要堵：快照走 ``requests.get``，THS 诊断走 ``_ak()``。
    只堵前者的话，快照路径会在测试里真去打网络（12 页 × 12s 超时）。
    """
    calls = {"n": 0}

    def _ak_boom():
        calls["n"] += 1          # 只统计 THS 诊断路线（按这个口径断言）
        raise RuntimeError("ProxyError: 17.push2.eastmoney.com")

    def _http_boom(*a, **kw):
        raise RuntimeError("ProxyError: 17.push2.eastmoney.com")

    monkeypatch.setattr(fm, "_ak", _ak_boom, raising=False)
    monkeypatch.setattr("requests.get", _http_boom, raising=True)
    return calls


# ---------------------------------------------------------------------------
# 全市场快照路线（伪造 requests.get）
# ---------------------------------------------------------------------------

def _fake_pages(pages, total=5400, industry="银行"):
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


# ---------------------------------------------------------------------------
# 旧坑已修：非空缓存永不丢、失败落负缓存
# ---------------------------------------------------------------------------

def test_existing_good_cache_preserved_and_returned(tmp_path, monkeypatch):
    """接口挂了也必须保住已有的好缓存；force_refresh 也以它为增量基底返回，不能丢。"""
    good = {f"60{i:04d}": "银行" for i in range(MIN_OK + 10)}
    cache = _setup(tmp_path, monkeypatch, initial=good)
    _block_network(monkeypatch)

    got = fm.fetcher.get_industry_map(force_refresh=True)

    assert got == good, "force_refresh 不通时仍以已有缓存为增量基底返回，不能丢数据"
    assert json.loads(cache.read_text(encoding="utf-8")) == good, (
        "空结果把已有缓存覆盖了 —— 网络恢复后永远不会自愈"
    )


def test_failure_is_negative_cached_no_retry_within_ttl(tmp_path, monkeypatch):
    """失败要落负缓存：TTL 窗口内不再重复打网络。"""
    cache = _setup(tmp_path, monkeypatch)
    calls = _block_network(monkeypatch)

    fm.fetcher.get_industry_map(force_refresh=True)
    assert cache.exists(), "首次失败应当留下负缓存文件"
    assert calls["n"] == 1, "force_refresh 应当触发一次 THS 诊断"

    monkeypatch.setattr(fm.fetcher, "_industry_map", None, raising=False)
    fm.fetcher.get_industry_map(force_refresh=False)
    assert calls["n"] == 1, "负缓存窗口内又去打网络了 —— 会把选股流程拖死"


def test_negative_cache_expires_and_retries(tmp_path, monkeypatch):
    """过了 TTL 允许重试（否则网络恢复后仍然永不相见）。"""
    import os

    cache = _setup(tmp_path, monkeypatch)
    calls = _block_network(monkeypatch)

    fm.fetcher.get_industry_map(force_refresh=True)
    old = _real_time.time() - (fm.DataFetcher._INDUSTRY_NEG_TTL_H + 1) * 3600
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
# 新契约：增量自愈 / 半张表保留 / 显式降级
# ---------------------------------------------------------------------------

def test_bulk_snapshot_is_accepted_and_persisted(tmp_path, monkeypatch):
    """覆盖够了就采用并落盘，而且一次都不用去打 THS 诊断。"""
    cache = _setup(tmp_path, monkeypatch)
    akshare_calls = _block_network(monkeypatch)          # 先全堵
    pages = {1: _rows(1000), 2: _rows(1000, 1000), 3: _rows(NEED - 2000 + 10, 2000)}
    monkeypatch.setattr("requests.get", _fake_pages(pages, 5400), raising=True)

    got = fm.fetcher.get_industry_map(force_refresh=True)

    assert len(got) >= NEED, f"覆盖够的快照应当被采用，实际 {len(got)}"
    assert got["600000"] == "银行"
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert len(saved) == len(got), "采用后必须落盘（否则下次还要赌网络）"
    assert akshare_calls["n"] == 0, "快照够用时不该再走 THS 诊断"


def test_partial_snapshot_is_kept_as_incremental_base(tmp_path, monkeypatch):
    """半张表不再「判失败清空」，而是作为增量基底保留并落盘，等后续窗口补齐。"""
    cache = _setup(tmp_path, monkeypatch)
    _block_network(monkeypatch)
    monkeypatch.setattr("requests.get", _fake_pages({1: _rows(300)}, 5400), raising=True)

    got = fm.fetcher.get_industry_map(force_refresh=True)

    assert len(got) == 300, f"半张表应作为增量基底保留，实际 {len(got)}"
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert len(saved) == 300, "半张表也应落盘，等下次窗口补齐"
    status = json.loads((Path(tmp_path) / "industry_map_status.json").read_text(encoding="utf-8"))
    assert status["coverage"] == 300
    assert status["floor"] == NEED
    assert status["source"] == "none"


def test_incremental_accumulates_across_calls(tmp_path, monkeypatch):
    """两次窗口各拿一部分，应累加而非互相覆盖/清零。"""
    cache = _setup(tmp_path, monkeypatch)
    _block_network(monkeypatch)
    monkeypatch.setattr("requests.get",
                        _fake_pages({1: _rows(300, 0, "银行")}, 5400), raising=True)
    g1 = fm.fetcher.get_industry_map(force_refresh=True)
    assert len(g1) == 300

    # 第二个窗口拿到另一批（不同 code 区间）
    monkeypatch.setattr("requests.get",
                        _fake_pages({1: _rows(300, 300, "医药")}, 5400), raising=True)
    fm.fetcher._industry_map = None  # 模拟新进程/重新载入
    g2 = fm.fetcher.get_industry_map(force_refresh=True)
    assert len(g2) == 600, f"两次窗口应累加，实际 {len(g2)}"
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert len(saved) == 600


def test_ths_diagnostic_cached_but_not_polluting_map(tmp_path, monkeypatch):
    """coverage < 接受阈值时，THS 行业 taxonomy 作为诊断落盘（industry_boards_ths.json），
    但绝不掺进 industry_map.json。"""
    import pandas as pd

    cache = _setup(tmp_path, monkeypatch)
    _block_network(monkeypatch)
    monkeypatch.setattr("requests.get", _fake_pages({1: _rows(300)}, 5400), raising=True)

    class _FakeAk:
        @staticmethod
        def stock_board_industry_name_ths():
            return pd.DataFrame({"name": ["银行", "医药"], "code": ["881155", "881156"]})

    monkeypatch.setattr(fm, "_ak", lambda: _FakeAk, raising=False)

    got = fm.fetcher.get_industry_map(force_refresh=True)

    boards_file = Path(tmp_path) / "industry_boards_ths.json"
    assert boards_file.exists(), "THS taxonomy 应落盘作诊断"
    boards = json.loads(boards_file.read_text(encoding="utf-8"))
    assert boards.get("银行") == "881155"
    # 映射本身只含 300 只个股（来自东财快照），不被 THS taxonomy 污染
    assert len(got) == 300
    assert "银行" not in got, "THS 的 name→code 没混进个股映射"


def test_status_file_records_explicit_degradation(tmp_path, monkeypatch):
    """彻底不通时，写 industry_map_status.json，note 显式说明根因（不再静默）。"""
    _setup(tmp_path, monkeypatch)
    _block_network(monkeypatch)

    fm.fetcher.get_industry_map(force_refresh=True)

    status = json.loads((Path(tmp_path) / "industry_map_status.json").read_text(encoding="utf-8"))
    assert status["coverage"] == 0
    assert "skipped" in status["note"]
    assert ("代理" in status["note"]) or ("eastmoney" in status["note"].lower())


def test_read_industry_map_status_note(tmp_path, monkeypatch):
    """predictor 中性化降级时读 status 根因的契约：无文件返回空，有文件返回 note。"""
    from src.models.predictor import Predictor as _SP

    monkeypatch.setattr(fm.fetcher, "cache_dir", Path(tmp_path), raising=False)
    assert _SP._read_industry_map_status_note() == ""

    (Path(tmp_path) / "industry_map_status.json").write_text(
        json.dumps({"note": "skipped：代理封锁 push2.eastmoney.com", "coverage": 0}),
        encoding="utf-8")
    assert "代理封锁" in _SP._read_industry_map_status_note()


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
                return {"data": {"total": 5400,
                                 "diff": _rows(1000, start=pn * 1000)}}

        return _Resp()

    monkeypatch.setattr("requests.get", _slow_get, raising=True)
    got = fm.fetcher._fetch_industry_map_bulk(max_seconds=0.2, timeout=5)

    assert 0 < pages_seen["n"] < 12, f"总时限没生效，翻了 {pages_seen['n']} 页"
    assert len(got) > 0, "到时限也要把已取到的部分交出去"

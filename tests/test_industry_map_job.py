# -*- coding: utf-8 -*-
"""``auto_scheduler.job_industry_map``（机会型行业映射补齐作业）回归测试。

为什么要有这个作业：本机到东财是**间歇性**可达 —— 2026-10-19 实测 22:47 还能
正常取到 496 个行业板块，之后 30 分钟内 40 次尝试**全部** ProxyError /
ConnectionReset。行业映射一旦补齐就永久落盘、之后不再打网络，所以"多试几次"
是有实际收益的；而仅靠 18:45 选股时顺带尝试一次，曝光率太低。

这里钉死三条：
  1) 已经补齐了就别再折腾网络（每天每小时跑，不能每次都打接口）；
  2) 撞到可用窗口要真的落盘；
  3) 网络不通不算"作业失败"——中性化只是退化到板级分组，不是故障。
"""

import json
from pathlib import Path

import src.data.fetcher as fm
import src.scheduler.auto_scheduler as sch


def _isolate(tmp_path, monkeypatch, initial=None):
    """把缓存目录、作业租约目录都指向 tmp（绝不碰生产 data/）。"""
    monkeypatch.setattr(fm.fetcher, "cache_dir", Path(tmp_path), raising=False)
    monkeypatch.setattr(fm.fetcher, "_industry_map", None, raising=False)
    monkeypatch.setattr(sch, "LEASE_DIR", Path(tmp_path), raising=False)
    if initial is not None:
        (Path(tmp_path) / "industry_map.json").write_text(
            json.dumps(initial, ensure_ascii=False), encoding="utf-8")
    return Path(tmp_path) / "industry_map.json"


def _pages(total_rows, total=5400, industry="银行"):
    """伪造 requests.get：按页返回行，取完即空。"""

    class _Resp:
        def __init__(self, p):
            self._p = p

        def json(self):
            return self._p

    def _get(url, params=None, timeout=None, **kwargs):
        pn = int((params or {}).get("pn", 1))
        start = (pn - 1) * 1000
        rows = [{"f12": f"{600000 + start + i}", "f14": f"股票{i}", "f100": industry}
                for i in range(min(1000, max(0, total_rows - start)))]
        return _Resp({"data": {"total": total, "diff": rows}})

    return _get


def test_already_ok_does_not_hit_network(tmp_path, monkeypatch):
    """已经补齐 → 直接 already_ok，一次网络都不打。"""
    need = fm.DataFetcher._INDUSTRY_BULK_MIN_CODES
    _isolate(tmp_path, monkeypatch, initial={f"60{i:05d}": "银行" for i in range(need + 10)})

    calls = {"n": 0}

    def _boom(*a, **kw):
        calls["n"] += 1
        raise AssertionError("已经补齐了还去打网络")

    monkeypatch.setattr("requests.get", _boom, raising=True)
    monkeypatch.setattr(fm, "_ak", _boom, raising=False)

    info = sch.job_industry_map()
    assert info["status"] == "already_ok", info
    assert info["n"] >= need
    assert calls["n"] == 0


def test_persists_when_window_is_available(tmp_path, monkeypatch):
    """撞到可用窗口 → 落盘，并把条数报出来。"""
    cache = _isolate(tmp_path, monkeypatch)
    need = fm.DataFetcher._INDUSTRY_BULK_MIN_CODES

    def _ak_boom():
        raise RuntimeError("ProxyError")

    monkeypatch.setattr(fm, "_ak", _ak_boom, raising=False)
    monkeypatch.setattr("requests.get", _pages(need + 50), raising=True)

    info = sch.job_industry_map()

    assert info["status"] == "ok", info
    assert info["n"] >= need
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert len(saved) >= need, "拿到就必须落盘（否则每小时都要重赌一次）"


def test_network_down_is_empty_not_error(tmp_path, monkeypatch):
    """网络不通是常态，不能报 error —— 那会淹没真正的作业故障。"""
    _isolate(tmp_path, monkeypatch)

    def _boom(*a, **kw):
        raise RuntimeError("ProxyError: 17.push2.eastmoney.com")

    monkeypatch.setattr("requests.get", _boom, raising=True)
    monkeypatch.setattr(fm, "_ak", _boom, raising=False)

    info = sch.job_industry_map()
    assert info["status"] == "empty", f"网络不通应报 empty：{info}"
    assert info["n"] == 0


def test_job_is_registered_hourly():
    """作业要真的挂进 start_scheduler 的作业表（每小时一次），否则等于没做。

    用源码断言而不是真的起 APScheduler：起调度器要抢跨进程锁、拉后台线程，
    在测试里既慢又会污染 ``data/.scheduler.lock``（同 rolling_ic 那条的做法）。
    """
    import inspect

    src = inspect.getsource(sch.start_scheduler)
    assert '"industry_map"' in src, "作业表缺 industry_map"
    assert "job_industry_map" in src
    assert "industry_map_cron" in src

    from config.settings import SchedulerConfig
    assert getattr(SchedulerConfig, "industry_map_cron", "0 * * * *") == "0 * * * *"

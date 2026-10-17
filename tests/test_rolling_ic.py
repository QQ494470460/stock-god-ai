# -*- coding: utf-8 -*-
"""rolling_ic 核心单元测试（全离线：不联网、不写生产 data/）。

覆盖：
  · render_curve / write_curve_files 对真实/空日志的渲染与统计口径
  · run_backfill 离线全链路（patch fetcher/predictor）+ 可重入去重
  · 调度器 job_rolling_ic 已注册并接入 cron 配置
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core import rolling_ic


def _fake_klines(n_codes=3, n_days=150, start="2026-01-02"):
    """生成连续日历日的合成日线（close 列），返回 {code: DataFrame(date, close)}。"""
    dates = pd.date_range(start, periods=n_days, freq="D")
    raw = {}
    rng = np.random.default_rng(42)
    for i in range(n_codes):
        # 让「后几只」相对收益系统性偏高，使评分与超额收益正相关（IC>0 可复现）
        base = 10.0 + i
        drift = (np.arange(n_days) * (0.002 * (i + 1)))
        noise = rng.normal(0, 0.01, n_days).cumsum()
        close = base + drift + noise
        df = pd.DataFrame({"date": dates, "close": close})
        df.index = pd.to_datetime(df["date"])
        raw[f"00000{i}.SZ"] = df.sort_index()
    return raw


def _install_offline(monkeypatch, n_codes=3, n_days=80):
    raw = _fake_klines(n_codes=n_codes, n_days=n_days)

    class _Code:
        def __init__(self, code, score):
            self.code = code
            self.score = score

    from src.data.fetcher import fetcher as F
    from src.models.predictor import predictor as P

    monkeypatch.setattr(F, "get_stock_pool", lambda pool: list(raw.keys()))
    monkeypatch.setattr(F, "get_daily_kline", lambda code: raw[code].copy())

    def _fake_predict(sliced):
        # 评分 = 代码序号（与合成收益正相关）
        out = []
        for k, code in enumerate(sorted(sliced.keys())):
            out.append(_Code(code, float(k)))
        return out

    monkeypatch.setattr(P, "predict_batch", _fake_predict)
    # 用日历日平移代替真实交易日历（避免导入 heavy 的 calendar 模块）
    monkeypatch.setattr(rolling_ic, "_next_trading_day",
                        lambda d, n: pd.Timestamp(d) + pd.Timedelta(days=n))
    # 状态文件不落盘（测试隔离）
    monkeypatch.setattr(rolling_ic, "_write_status", lambda d, *a, **k: None)
    return raw


# ---------------------------------------------------------------------------
# render_curve / write_curve_files
# ---------------------------------------------------------------------------

def test_render_curve_basic_stats(tmp_path):
    log = [
        {"date": "2026-09-01", "ic": 0.10, "n": 50},
        {"date": "2026-09-02", "ic": -0.05, "n": 50},
        {"date": "2026-09-03", "ic": 0.20, "n": 50},
        {"date": "2026-09-04", "ic": 0.05, "n": 50},
    ]
    logp = tmp_path / "log.json"
    logp.write_text(json.dumps(log), encoding="utf-8")

    svg, stats = rolling_ic.render_curve(log_path=logp, metrics_path=tmp_path / "no.json",
                                         window=3)
    assert "<svg" in svg
    assert stats["n_points"] == 4
    assert abs(stats["mean_ic"] - np.mean([0.10, -0.05, 0.20, 0.05])) < 1e-6
    # 正 IC 占比 3/4
    assert stats["pos_rate"] == pytest.approx(0.75)
    # 日频 IR = mean/std（G-K 口径），与 t 统计量(mean/SE) 不同
    sd = float(np.std([0.10, -0.05, 0.20, 0.05], ddof=1))
    assert stats["ir_mean_std"] == pytest.approx(0.075 / sd, abs=1e-3)
    assert stats["last_date"] == "2026-09-04"


def test_render_curve_empty_placeholder(tmp_path):
    logp = tmp_path / "empty.json"
    logp.write_text("[]", encoding="utf-8")
    svg, stats = rolling_ic.render_curve(log_path=logp)
    assert stats["n_points"] == 0
    assert "暂无 RankIC" in svg


def test_write_curve_files_writes_html_svg(tmp_path):
    log = [{"date": "2026-09-01", "ic": 0.12, "n": 40},
           {"date": "2026-09-02", "ic": 0.08, "n": 40}]
    logp = tmp_path / "log.json"
    logp.write_text(json.dumps(log), encoding="utf-8")
    stats = rolling_ic.write_curve_files(
        out_html=tmp_path / "c.html", out_svg=tmp_path / "c.svg",
        window=2, log_path=logp, metrics_path=tmp_path / "no.json")
    assert stats["n_points"] == 2
    assert (tmp_path / "c.svg").exists()
    assert (tmp_path / "c.html").exists()
    assert "<svg" in (tmp_path / "c.html").read_text(encoding="utf-8")


def test_spearman_monotonic():
    x = np.arange(10)
    y = 2 * x + 1
    assert rolling_ic._spearman(x, y) == pytest.approx(1.0)
    # 完全逆序 -> -1
    assert rolling_ic._spearman(x, -y) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# run_backfill 离线全链路 + 可重入去重
# ---------------------------------------------------------------------------

def test_run_backfill_offline_and_idempotent(monkeypatch, tmp_path):
    _install_offline(monkeypatch, n_codes=4, n_days=150)

    out = tmp_path / "ric_log.json"
    res1 = rolling_ic.run_backfill(days=10, sample=4, horizon=3, min_stocks=3,
                                   pool="zz500", out=out)
    assert res1["status"] == "ok"
    assert res1["added"] > 0
    assert res1["total"] == res1["added"]            # 全新日志
    assert res1["mean_ic"] == pytest.approx(float(res1["mean_ic"]))  # 有限数
    assert out.exists()
    first_total = res1["total"]

    # 第二次跑（相同数据/路径）：可重入去重，应 noop
    res2 = rolling_ic.run_backfill(days=10, sample=4, horizon=3, min_stocks=3,
                                   pool="zz500", out=out)
    assert res2["status"] == "noop"
    assert res2["added"] == 0
    assert res2["total"] == first_total


def test_run_backfill_dry_run_shape(monkeypatch, tmp_path):
    _install_offline(monkeypatch, n_codes=4, n_days=150)
    out = tmp_path / "ric_log_dry.json"
    res = rolling_ic.run_backfill(days=8, sample=4, horizon=3, min_stocks=3,
                                  pool="zz500", out=out, dry_run=True)
    assert res["status"] == "dry_run"
    assert res["candidates"] >= 1
    assert res["todo"] >= 1
    # dry_run 不应写日志
    assert not out.exists()


# ---------------------------------------------------------------------------
# 调度器接线
# ---------------------------------------------------------------------------

def test_scheduler_wires_rolling_ic():
    from config.settings import config as app_config
    assert hasattr(app_config.scheduler, "rolling_ic_cron")
    assert app_config.scheduler.rolling_ic_cron == "15 19 * * mon-fri"

    import src.scheduler.auto_scheduler as sch
    # job 函数存在且可被导入
    assert hasattr(sch, "job_rolling_ic")
    # start_scheduler 的 jobs 列表里包含 rolling_ic（构造但不真正启动）
    # 通过反射确认注册名出现在源码装配处
    import inspect
    src_txt = inspect.getsource(sch.start_scheduler)
    assert '("rolling_ic"' in src_txt
    assert "rolling_ic_cron" in src_txt

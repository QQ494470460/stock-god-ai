# -*- coding: utf-8 -*-
"""收盘后自动化调度 + 模型注册中心 单元测试（零外部依赖 / 零真实资金）"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.models.model_registry import scan_models, best_loadable, best_overall
from src.core import daily_pick
from src.scheduler import auto_scheduler


# ---------------------------------------------------------------------------
# Fake 依赖
# ---------------------------------------------------------------------------

def _make_df(n=80):
    return pd.DataFrame({"close": list(range(100, 100 + n))})


class FakeFetcher:
    def get_stock_pool(self, pool="zz500"):
        return ["000001", "600519"]

    def get_batch_daily_kline(self, codes, max_workers=8, progress_callback=None):
        return {c: _make_df() for c in codes}

    def get_stock_list(self):
        return pd.DataFrame({"code": ["000001", "600519"], "name": ["平安银行", "贵州茅台"]})

    def get_benchmark_kline(self, index_code="000300.SH", start_date=None):
        return _make_df(120)

    def update_data(self):
        return True


class FakePredictor:
    def get_top_recommendations(self, stock_data, stock_names=None, top_n=20, min_score=50.0):
        return pd.DataFrame([
            {"排名": 1, "代码": "600519", "名称": "贵州茅台", "评分": 82.0,
             "上涨概率": 70.0, "XGB信号": "📈买入", "推荐理由": "动量"},
            {"排名": 2, "代码": "000001", "名称": "平安银行", "评分": 61.0,
             "上涨概率": 55.0, "XGB信号": "📈买入", "推荐理由": "价值"},
        ])


class FakeEvolution:
    def __init__(self):
        self.checked = 0

    def get_effective_min_score(self, *a, **k):
        return 55.0

    def detect_regime(self, df):
        return "bear"

    def evolve_from_verification(self, r):
        return []

    def should_retrain(self):
        return False, ""

    def mark_checked(self):
        # 观测心跳：job_evolution 跑完必调一次（无论有无可验证样本）
        self.checked += 1


@pytest.fixture
def patched(monkeypatch, tmp_path):
    ff, fp, fe = FakeFetcher(), FakePredictor(), FakeEvolution()
    # daily_pick 模块级单例
    monkeypatch.setattr(daily_pick, "fetcher", ff)
    monkeypatch.setattr(daily_pick, "predictor", fp)
    monkeypatch.setattr(daily_pick, "evolution", fe)
    # auto_scheduler 模块级单例
    monkeypatch.setattr(auto_scheduler, "fetcher", ff)
    monkeypatch.setattr(auto_scheduler, "predictor", fp)
    monkeypatch.setattr(auto_scheduler, "evolution", fe)
    # 让调度器报告写入临时目录
    monkeypatch.setattr("src.scheduler.auto_scheduler.app_config", _cfg())
    monkeypatch.setattr("src.core.daily_pick.app_config", _cfg())
    # 命脉：job_evolution() 内部会调 tracker_v2.verify_all_pending()，它会写
    # data/prediction_history.json —— **那是生产状态**，而且 dry_run 对它无效
    # （job_evolution 不接受 dry_run 参数）。这里的隔离缺失曾导致每次跑全量
    # 测试都真实改写预测历史（文件里出现「跑测当天」的 verified_at 时间戳）。
    import src.models.tracker_v2 as _tracker
    monkeypatch.setattr(_tracker, "TRACK_FILE",
                        tmp_path / "prediction_history.json")
    return ff, fp, fe


def _cfg():
    """构造一个把 report_dir 指向 tmp 的轻量配置替身"""
    class S:
        class scheduler:
            pick_pool = "zz500"
            max_picks = 20
            report_dir = str(Path(__file__).resolve().parent / "_sched_tmp")
            enabled = False
        class ml:
            auto_threshold_adjust = True
        class trading_loop:
            max_positions = 10
            cash_reserve_pct = 0.10
            weight_cap = 0.30
            weighting = "equal"
    return S()


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_registry_scans_models():
    infos = scan_models()
    assert len(infos) >= 1
    # 全局最优应为 alpha（不可部署）或已部署生产模型
    bo = best_overall()
    assert bo is not None
    # 可部署最优应当是生产 xgboost（rank_ic 0.0747）
    bl = best_loadable()
    assert bl is not None
    assert bl.loadable is True
    assert bl.rank_ic > 0.07


def test_daily_pick_generates_report(patched, tmp_path, monkeypatch):
    # 让报告写到 tmp_path
    monkeypatch.setattr("src.core.daily_pick.app_config.scheduler.report_dir", str(tmp_path))
    df, rep = daily_pick.generate_daily_picks(pool="zz500", write_report=True)
    assert rep["status"] in ("ok", "empty")
    assert rep["n_passed"] == 2
    assert rep["regime"] == "bear"
    files = list(tmp_path.glob("daily_picks_*.md"))
    assert len(files) == 1
    assert "贵州茅台" in files[0].read_text(encoding="utf-8")


def test_scheduler_jobs_run(patched, monkeypatch, tmp_path):
    ff, fp, fe = patched
    # job_data_update
    info1 = auto_scheduler.job_data_update()
    assert info1["status"] == "ok"
    # job_daily_pick（dry_run 不写文件）
    monkeypatch.setattr("src.core.daily_pick.app_config.scheduler.report_dir", str(tmp_path))
    info2 = auto_scheduler.job_daily_pick(dry_run=True)
    assert info2["status"] in ("ok", "empty")
    assert info2["n_passed"] == 2
    # job_evolution 不崩溃（tracker 可能无数据，返回 dict 即可）
    info3 = auto_scheduler.job_evolution()
    assert isinstance(info3, dict) and info3["job"] == "evolution"
    # 观测心跳必须被触发（即使 0 可验证样本，也要能看出「作业跑过」）
    assert fe.checked == 1, "job_evolution 未调用 evolution.mark_checked()"


def test_scheduler_disabled_by_default(patched):
    # config.scheduler.enabled=False → 不启动
    r = auto_scheduler.start_scheduler()
    assert r is None
    assert auto_scheduler.is_running() is False


# ---------------------------------------------------------------------------
# 零污染回归
# ---------------------------------------------------------------------------

def test_job_evolution_never_touches_real_prediction_history(patched, monkeypatch):
    """命脉：调度里的进化作业**不得**改写真实预测历史。

    回归背景：``job_evolution()`` 复用 ``tracker_v2.verify_all_pending()``，
    后者把验证结果写进 ``TRACK_FILE``（真实为 ``data/prediction_history.json``）。
    夹具若漏了隔离，跑一次全量测试就会把生产文件里的 ``verifications``
    填上「今天」的 ``verified_at`` —— 污染了真实数据，还让「零污染核验」
    本身失去意义。本用例把这条纪律固化成断言。
    """
    import src.models.tracker_v2 as tracker

    real = ROOT / "data" / "prediction_history.json"
    before = real.read_bytes() if real.exists() else None

    # 夹具已把 TRACK_FILE 指向 tmp；确认它确实不在真实路径上
    assert Path(tracker.TRACK_FILE) != real

    auto_scheduler.job_evolution()

    after = real.read_bytes() if real.exists() else None
    assert after == before, "job_evolution 污染了真实 prediction_history.json"


def test_job_evolution_never_touches_real_evolution_state(patched):
    """零污染：job_evolution 末尾的观测心跳不得改写真实 evolution_state.json。

    新增 mark_checked() 写盘点后，若夹具没把 evolution 换成替身，每次跑测都会把
    last_check_at 刷成「跑测当天」，污染生产进化状态。这里把该纪律钉死。
    """
    real = ROOT / "data" / "evolution_state.json"
    before = real.read_bytes() if real.exists() else None

    auto_scheduler.job_evolution()

    after = real.read_bytes() if real.exists() else None
    assert after == before, "job_evolution 污染了真实 evolution_state.json"


def test_job_evolution_writes_into_tmp(patched, tmp_path):
    """反证：隔离生效后，写入落在 tmp 里（而不是「什么都没写」）。"""
    import src.models.tracker_v2 as tracker

    auto_scheduler.job_evolution()
    assert Path(tracker.TRACK_FILE).parent == tmp_path


def test_run_all_once(patched, monkeypatch, tmp_path):
    monkeypatch.setattr("src.core.daily_pick.app_config.scheduler.report_dir", str(tmp_path))
    out = auto_scheduler.run_all_once(dry_run=True)
    assert "data_update" in out and "daily_pick" in out and "evolution" in out
    assert out["daily_pick"]["n_passed"] == 2


def test_daily_pick_also_produces_trade_plan(patched, monkeypatch, tmp_path):
    """闭环的桥：收盘选股后必须顺手产出交易计划，盘中才拿得到标的。"""
    import src.trading.trade_plan as tp

    monkeypatch.setattr("src.core.daily_pick.app_config.scheduler.report_dir", str(tmp_path))
    monkeypatch.setattr(tp, "DATA_DIR", tmp_path)              # 计划落到 tmp
    monkeypatch.setattr(tp, "resolve_target_date", lambda ref=None: "2026-10-15")

    info = auto_scheduler.job_daily_pick(dry_run=False)
    assert info["status"] == "ok"
    assert info["plan_date"] == "2026-10-15"
    assert info["plan_targets"] == 2                            # FakePredictor 给 2 只

    p = tmp_path / "trade_plans" / "trade_plan_20261015.json"
    assert p.exists(), "交易计划文件未生成"

    import json
    plan = json.loads(p.read_text(encoding="utf-8"))
    assert [t["code"] for t in plan["targets"]] == ["600519", "000001"]
    # 2 只等权本应各 45%，但被单票权重上限 30% 截断 → 各 30%，合计 60%，
    # 剩余额度留在现金（更保守）。这验证了「计划与风控口径对齐」。
    assert all(abs(t["weight"] - 0.30) < 1e-6 for t in plan["targets"])
    assert abs(sum(t["weight"] for t in plan["targets"]) - 0.60) < 1e-6


def test_daily_pick_plan_dry_run_writes_nothing(patched, monkeypatch, tmp_path):
    """dry_run 时不落盘计划文件。"""
    import src.trading.trade_plan as tp
    monkeypatch.setattr("src.core.daily_pick.app_config.scheduler.report_dir", str(tmp_path))
    monkeypatch.setattr(tp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(tp, "resolve_target_date", lambda ref=None: "2026-10-15")

    info = auto_scheduler.job_daily_pick(dry_run=True)
    assert info["plan_date"] == "2026-10-15"        # 计划对象仍生成
    assert not (tmp_path / "trade_plans").exists()  # 但不落盘

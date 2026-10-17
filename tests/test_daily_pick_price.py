"""daily_pick 必须把最新收盘价透传进 picks（根因修复验证）。

历史 bug：报告里 20 只 picks 全没有「最新价」键，导致 build_trade_plan 拿不到
任何参考价、计划 price 恒为 0.0，「跳空不追」校验无从做起。

本测试用 DI 注入 Fake fetcher / predictor / evolution，确认 generate_daily_picks
在产出 report["picks"] 时，每只 pick 都带上了「最新价」键、且值等于行情里该标的
的最后一笔收盘价（四舍五入 2 位）。
"""

from __future__ import annotations

import pandas as pd
import pytest

DP = pytest.importorskip("src.core.daily_pick")
generate_daily_picks = DP.generate_daily_picks


class _FakeKline:
    """构造一个 len>=60、带 close 列的 DataFrame，末根收盘价 = last_close。"""

    def __init__(self, last_close: float):
        self.last_close = float(last_close)

    def to_df(self) -> pd.DataFrame:
        n = 60
        closes = [self.last_close - (n - 1 - i) * 0.01 for i in range(n)]
        return pd.DataFrame({"close": closes, "open": closes, "vol": [100] * n})


class FakeFetcher:
    CLOSES = {"600177": 8.45, "600901": 7.09, "002966": 9.0}

    def get_stock_pool(self, pool="zz500"):
        return list(self.CLOSES.keys())

    def get_batch_daily_kline(self, codes, max_workers=8):
        out = {}
        for c in codes:
            if c in self.CLOSES:
                out[c] = _FakeKline(self.CLOSES[c]).to_df()
        return out

    def get_stock_list(self):
        return pd.DataFrame(
            {"code": list(self.CLOSES.keys()),
             "name": ["雅戈尔", "江苏金租", "苏州银行"]}
        )

    def get_benchmark_kline(self, index_code, start_date=None):
        # regime 检测失败被上层吞掉，不影响参考价透传，这里直接抛省事
        raise RuntimeError("benchmark unavailable in test")


class FakePredictor:
    def get_top_recommendations(self, stock_data, stock_names, top_n=10, min_score=0.0):
        rows = []
        for c, name in stock_names.items():
            rows.append({
                "代码": c, "名称": name, "评分": 60.0,
                "推荐理由": "个股牛市顺风", "确信度": 25.0,
                "一致": "一致", "上涨概率": 0.6, "震荡概率": 0.1,
                "下跌概率": 0.3, "波动率": "12%", "XGB信号": "持有",
            })
        return pd.DataFrame(rows)


class FakeEvolution:
    def detect_regime(self, kline):
        return "bull"

    def get_effective_min_score(self):
        return 50.0


def test_daily_pick_injects_latest_price():
    df, report = generate_daily_picks(
        pool="zz500", max_picks=10, min_score=50.0,
        fetcher_inst=FakeFetcher(),
        predictor_inst=FakePredictor(),
        evolution_inst=FakeEvolution(),
        write_report=False,
    )
    picks = report.get("picks") or []
    assert len(picks) == 3, f"应有 3 只 pick，实际 {len(picks)}"

    for p in picks:
        assert "最新价" in p, f"pick {p.get('代码')} 缺少「最新价」键"
        expected = FakeFetcher.CLOSES[p["代码"]]
        assert p["最新价"] == pytest.approx(expected, abs=1e-6), (
            f"{p['代码']} 最新价应为 {expected}，实际 {p['最新价']}"
        )

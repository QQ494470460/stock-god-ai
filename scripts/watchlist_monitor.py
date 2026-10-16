# -*- coding: utf-8 -*-
"""
自选股 AI 实时监测 —— 独立监控脚本
脱离 Streamlit，可被 OpenClaw / cron 直接调用。
输出纯文本报告，适合推送到聊天窗口。

用法:
  python scripts/watchlist_monitor.py              # 静默模式（仅异常告警输出）
  python scripts/watchlist_monitor.py --verbose    # 完整报告
  python scripts/watchlist_monitor.py --dry-run    # 不调用AI，仅行情快照
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import config as app_config
from src.data.fetcher import fetcher
from src.models.predictor import predictor


class WatchlistMonitor:
    """自选股 AI 实时监测器"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or (PROJECT_ROOT / "config" / "watchlist.json")
        self.watch_config = self._load_config()
        self._spot_cache = None

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            logger.warning(f"自选股配置不存在: {self.config_path}，使用默认值")
            return {
                "watchlist": ["000001", "000002", "600519"],
                "alert": {"min_ai_score": 50, "alert_on_buy_signal": True,
                          "alert_on_sell_signal": True, "alert_on_score_change": 15,
                          "price_move_threshold": 3.0, "max_alerts_per_run": 5},
                "schedule": {"enabled": True, "interval_minutes": 5, "trading_hours_only": True}
            }
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def watchlist(self) -> list:
        return self.watch_config.get("watchlist", [])

    @property
    def alert_config(self) -> dict:
        return self.watch_config.get("alert", {})

    def is_trading_time(self) -> bool:
        """判断当前是否在 A 股交易时段"""
        now = datetime.now()
        if now.weekday() >= 5:  # 周末
            return False
        t = now.hour * 60 + now.minute
        return (9 * 60 + 30 <= t <= 11 * 60 + 30) or (13 * 60 <= t <= 15 * 60 + 0)

    def fetch_spot_data(self) -> dict:
        """获取全市场实时行情（带缓存10秒）"""
        now = datetime.now()
        cache_key = now.strftime("%Y%m%d_%H%M")[:-1]  # 10秒粒度
        if self._spot_cache and self._spot_cache.get("key") == cache_key:
            return self._spot_cache.get("data", {})

        spot_data = {}
        try:
            import akshare as ak
            df_spot = ak.stock_zh_a_spot_em()
            df_spot = df_spot.set_index("代码")
            spot_data = df_spot.to_dict("index")
            self._spot_cache = {"key": cache_key, "data": spot_data}
            logger.info(f"✅ 实时行情已获取，{len(spot_data)} 只")
        except Exception as e:
            logger.warning(f"获取实时行情失败: {e}")
        return spot_data

    def get_stock_name(self, code: str) -> str:
        """获取股票名称"""
        try:
            df_name = fetcher.get_stock_list()
            matches = df_name[df_name["代码"] == code]
            if len(matches) > 0:
                return str(matches["名称"].values[0])
        except Exception:
            pass
        return code

    def run_single(self, code: str, spot_data: dict, names: dict) -> dict:
        """对单只股票运行监控"""
        result = {
            "code": code,
            "name": names.get(code, code),
            "price": 0, "change_pct": 0, "volume_wan": 0,
            "ai_score": 0, "ai_signal": "—", "prob_up": 0, "prob_down": 0,
            "confidence": 0, "agreement": 0, "reasons": [],
            "alert": False, "alert_reason": "", "error": None
        }

        try:
            df = fetcher.get_daily_kline(code)
            if len(df) < 60:
                result["error"] = "数据不足"
                return result

            close_col = df["close"]
            last_close = float(close_col.iloc[-1])
            prev_close = float(close_col.iloc[-2])
            change_pct = (last_close / prev_close - 1) * 100

            # 实时行情 vs 日线收盘价
            live_price = last_close
            live_chg = change_pct
            live_vol = int(df["volume"].iloc[-1])

            if code in spot_data:
                spot = spot_data[code]
                live_price = float(spot.get("最新价", last_close))
                live_chg = float(spot.get("涨跌幅", change_pct))
                live_vol = int(float(spot.get("成交量", 0))) if spot.get("成交量", 0) else live_vol

            result["price"] = round(live_price, 2)
            result["change_pct"] = round(live_chg, 2)
            result["volume_wan"] = round(live_vol / 10000, 0)

            # AI 预测
            if predictor.is_ready:
                pred = predictor.predict_single(df, code, result["name"])
                result["ai_score"] = round(pred.score, 1)
                result["prob_up"] = round(pred.prob_up * 100, 1)
                result["prob_down"] = round(pred.prob_down * 100, 1)
                result["confidence"] = round(pred.confidence * 100, 1)
                result["agreement"] = pred.agreement
                result["reasons"] = pred.reasons[:3] if pred.reasons else []

                if pred.prob_up > 0.5:
                    result["ai_signal"] = "📈看涨"
                elif pred.prob_down > 0.5:
                    result["ai_signal"] = "📉看跌"
                else:
                    result["ai_signal"] = "➡️观望"
        except Exception as e:
            result["error"] = str(e)

        return result

    def check_alerts(self, results: list) -> list:
        """检查触发告警的股票"""
        alerts = []
        alert_cfg = self.alert_config
        min_score = alert_cfg.get("min_ai_score", 50)
        threshold = alert_cfg.get("price_move_threshold", 3.0)
        score_delta = alert_cfg.get("alert_on_score_change", 15)

        for r in results:
            if r["error"]:
                continue
            reason = ""
            is_alert = False

            # AI 买入信号 + 高分
            if alert_cfg.get("alert_on_buy_signal") and r["ai_signal"] == "📈看涨":
                if r["ai_score"] >= 65:
                    reason = f"🟢 AI强力看涨 (评分{r['ai_score']:.0f})"
                    is_alert = True
                elif r["ai_score"] >= 50:
                    reason = f"🟡 AI看涨 (评分{r['ai_score']:.0f})"
                    is_alert = True

            # AI 卖出信号
            if alert_cfg.get("alert_on_sell_signal") and r["ai_signal"] == "📉看跌":
                if r["ai_score"] >= 50:
                    reason = f"🔴 AI看跌 (评分{r['ai_score']:.0f})"
                    is_alert = True

            # 价格剧烈波动
            if abs(r["change_pct"]) >= threshold:
                direction = "📈" if r["change_pct"] > 0 else "📉"
                reason = f"{direction} 涨跌{r['change_pct']:+.2f}%超过阈值{threshold}%"
                is_alert = True

            if is_alert:
                r["alert"] = True
                r["alert_reason"] = reason
                alerts.append(r)

        return alerts

    def run(self, verbose: bool = False, dry_run: bool = False) -> str:
        """主入口：运行一轮监控，返回文本报告"""
        now = datetime.now()
        trading = self.is_trading_time()
        lines = []

        # 头
        lines.append(f"📡 **选股神 AI · 自选股监控**")
        lines.append(f"⏰ {now.strftime('%Y-%m-%d %H:%M:%S')} | {'🟢 交易中' if trading else '⏸️ 非交易时段'}")
        lines.append("")

        codes = self.watchlist
        if not codes:
            return "⚠️ 自选股列表为空，请在 config/watchlist.json 中添加股票代码"

        # 获取实时数据
        spot_data = {}
        if trading:
            spot_data = self.fetch_spot_data()
            lines.append(f"📊 实时行情: {len(spot_data)} 只 | 数据源: 东方财富")
        else:
            lines.append("📊 数据源: 日线收盘价（非交易时段）")

        # 取股票名称
        names = {}
        try:
            df_all = fetcher.get_stock_list()
            for c in codes:
                m = df_all[df_all["代码"] == c]
                if len(m) > 0:
                    names[c] = str(m["名称"].values[0])
        except Exception:
            pass

        lines.append(f"🎯 监控标的: {len(codes)} 只")
        lines.append("")

        # 逐只分析
        results = []
        for i, code in enumerate(codes):
            r = self.run_single(code, spot_data if not dry_run else {}, names)
            results.append(r)

        # 排序：AI评分降序
        results.sort(key=lambda x: x["ai_score"], reverse=True)

        # ---- 告警检查 ----
        if not dry_run:
            alerts = self.check_alerts(results)
            max_alerts = self.alert_config.get("max_alerts_per_run", 5)
            if alerts:
                lines.append("### 🚨 告警信号")
                lines.append("")
                for a in alerts[:max_alerts]:
                    sig_icon = a["ai_signal"]
                    lines.append(f"**{a['name']}({a['code']})** {sig_icon}")
                    lines.append(f"  现价 {a['price']:.2f} | 涨跌 {a['change_pct']:+.2f}% | AI评分 {a['ai_score']:.0f}")
                    lines.append(f"  原因: {a['alert_reason']}")
                    lines.append(f"  置信度 {a['confidence']:.0f}% | {'✅一致' if a['agreement'] >= 1.0 else '⚠️分歧'}")
                    lines.append("")
                if len(alerts) > max_alerts:
                    lines.append(f"… 另有 {len(alerts) - max_alerts} 条告警已省略")
                    lines.append("")

        # ---- 完整报告 (verbose) ----
        if verbose:
            lines.append("### 📋 完整监控报告")
            lines.append("")
            # 表头
            header = f"{'代码':<8} {'名称':<8} {'现价':>8} {'涨跌%':>7} {'AI评分':>7} {'信号':<8} {'确信%':>6} {'一致':<6}"
            lines.append(f"```")
            lines.append(header)
            lines.append("-" * len(header))

            for r in results:
                if r["error"]:
                    lines.append(f"{r['code']:<8} {'ERR':<8} {'—':>8} {'—':>7} {'—':>7} {'—':<8} {'—':>6} {'—':<6}")
                    continue
                agree_str = "✅" if r["agreement"] >= 1.0 else ("⚠️" if r["agreement"] <= 0.3 else "—")
                lines.append(
                    f"{r['code']:<8} {r['name'][:8]:<8} {r['price']:>8.2f} {r['change_pct']:>+6.2f}% {r['ai_score']:>6.0f}  {r['ai_signal']:<5} {r['confidence']:>5.0f}% {agree_str:<5}"
                )
            lines.append("```")
            lines.append("")

            # 统计摘要
            buys = sum(1 for r in results if r["ai_signal"] == "📈看涨")
            sells = sum(1 for r in results if r["ai_signal"] == "📉看跌")
            avg_score = np.mean([r["ai_score"] for r in results if not r["error"]]) if results else 0
            avg_chg = np.mean([r["change_pct"] for r in results if not r["error"]]) if results else 0

            lines.append(f"📈 看涨 {buys} | 📉 看跌 {sells} | ➡️ 观望 {len(results) - buys - sells}")
            lines.append(f"🧠 均AI评分 {avg_score:.0f} | 📊 均涨跌 {avg_chg:+.2f}%")
            lines.append("")

            # Top 变化
            top_up = sorted([r for r in results if not r["error"]], key=lambda x: x["change_pct"], reverse=True)[:3]
            top_down = sorted([r for r in results if not r["error"]], key=lambda x: x["change_pct"])[:3]
            top_ai = sorted([r for r in results if not r["error"]], key=lambda x: x["ai_score"], reverse=True)[:3]

            if top_up:
                lines.append("🔥 涨幅 Top 3:")
                for r in top_up:
                    lines.append(f"  {r['name']}({r['code']}) {r['change_pct']:+.2f}%  {r['ai_signal']} 评分{r['ai_score']:.0f}")
            if top_down:
                lines.append("❄️ 跌幅 Top 3:")
                for r in top_down:
                    lines.append(f"  {r['name']}({r['code']}) {r['change_pct']:+.2f}%  {r['ai_signal']} 评分{r['ai_score']:.0f}")
            if top_ai:
                lines.append("🧠 AI 高分 Top 3:")
                for r in top_ai:
                    reason_str = "; ".join(r.get("reasons", [])[:2]) or "—"
                    lines.append(f"  {r['name']}({r['code']}) 评分{r['ai_score']:.0f} {r['ai_signal']} | {reason_str}")

        elif not alerts:
            # 静默模式 + 无告警 → 简短摘要
            buys = sum(1 for r in results if r["ai_signal"] == "📈看涨")
            avg_score = np.mean([r["ai_score"] for r in results if not r["error"]]) if results else 0
            lines.append(f"✅ 一切平静 | 看涨{buys}/{len(results)} | 均评分{avg_score:.0f}")
            return ""  # 无告警时不输出

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="选股神AI - 自选股实时监测")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出完整报告")
    parser.add_argument("--dry-run", action="store_true", help="仅行情快照，不调用AI")
    parser.add_argument("--config", type=str, help="配置文件路径")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    monitor = WatchlistMonitor(config_path)

    # 检查交易时段（cron场景提前退出），但 dry-run/verbose 不受限
    if not args.dry_run and not args.verbose:
        if monitor.watch_config.get("schedule", {}).get("trading_hours_only", True):
            if not monitor.is_trading_time():
                logger.info("非交易时段，跳过")
                return

    report = monitor.run(verbose=args.verbose, dry_run=args.dry_run)
    if report:
        print(report)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""RankIC 历史回填 —— 把 rolling_ic_log.json 从「偶尔 1 点」补成「逐交易日序列」。

为什么需要它
------------
paper_shadow_audit.py --live-ic 只在「跑那一下」产出 1 个点，所以日志长期只有
零星 2 个点，画不出稳定性曲线。本脚本用**真实预测器**对历史上每个交易日打分，
算「评分日 → 后 horizon 个交易日」的超额收益（与训练标签口径一致：超额 =
个股收益 - 等权宇宙均值），再逐日算横截面 Spearman IC，累积进 rolling_ic_log.json。

前视（look-ahead）防护（与回测引擎同一套纪律）：
  · 评分日 d 的打分只用 d 及之前的数据（predict_batch 喂的是 df.loc[df.index <= d]）。
  · 前向收益用 d+horizon（交易日），绝不回看未来之外的信息。
  · 因此算出的每日 IC 是真实的 OOS 排序能力，可直接拿去画稳定性曲线 / 判稳定性。

性能与边界
----------
  · 成本 ≈ (评分日数) × (宇宙只数) 次单股打分。默认 --days 30 --sample 150，
    约 4500 次，后台跑几分钟到十几分钟。机器够快可调大。
  · --sample 截断宇宙（取前 N 只），只为控制成本，不影响 IC 口径。
  · 可重入：已存在于日志的评分日自动跳过；日志只保留最近 250 点。
  · 纯离线无需券商；但需要联网拿行情（fetcher），这是本机看板已具备的能力。

用法
----
    python scripts/rolling_ic_backfill.py                 # 默认 zz500，近 30 交易日，150 只
    python scripts/rolling_ic_backfill.py --days 60 --sample 300
    python scripts/rolling_ic_backfill.py --pool hs300 --horizon 5
    python scripts/rolling_ic_backfill.py --dry-run       # 只报告规模，不跑模型
    python scripts/rolling_ic_backfill.py --out data/rolling_ic_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IC_LOG = ROOT / "data" / "rolling_ic_log.json"


def _load_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _next_trading_day(d: pd.Timestamp, n: int) -> pd.Timestamp:
    from src.trading.calendar import next_trading_day
    cur = d
    for _ in range(n):
        cur = pd.Timestamp(next_trading_day(cur, include_today=False))
    return cur


def main() -> int:
    ap = argparse.ArgumentParser(description="RankIC 历史回填（累积 rolling_ic_log.json）")
    ap.add_argument("--pool", default="zz500", help="股票池 hs300/zz500/zz1000/all")
    ap.add_argument("--days", type=int, default=30, help="回溯的评分日数量（交易日）")
    ap.add_argument("--sample", type=int, default=150, help="宇宙截断只数（取前 N 只，控制成本）")
    ap.add_argument("--horizon", type=int, default=5, help="前向收益跨度（交易日）")
    ap.add_argument("--min-stocks", type=int, default=30, help="单日有效样本下限，低于则跳过该日")
    ap.add_argument("--out", type=Path, default=IC_LOG, help="输出日志 JSON")
    ap.add_argument("--dry-run", action="store_true", help="只报告规模不跑模型")
    args = ap.parse_args()

    from src.data.fetcher import fetcher
    from src.models.predictor import predictor

    print(f"📡 拉取股票池 {args.pool} ...")
    codes = list(fetcher.get_stock_pool(args.pool))
    if args.sample and args.sample < len(codes):
        codes = codes[: args.sample]
    print(f"   宇宙 {len(codes)} 只（截断自 {args.pool}）")

    print("📥 拉取历史日线 ...")
    raw = {}
    for i, code in enumerate(codes, 1):
        try:
            df = fetcher.get_daily_kline(code)
            if df is None or len(df) < 120:
                continue
            df = df.copy()
            df.index = pd.to_datetime(df["date"])
            df = df.sort_index()
            raw[code] = df
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ {code} 行情失败: {e}")
        if i % 50 == 0:
            print(f"   ...{i}/{len(codes)}")
    if not raw:
        print("❌ 无任何可用行情，退出")
        return 1
    print(f"   可用 {len(raw)} 只")

    close = pd.DataFrame({c: df["close"] for c, df in raw.items()}).sort_index()
    all_dates = list(close.index)
    # 候选评分日：从末尾往前取 --days 个「足够靠后、且 d+horizon 仍在数据内」的交易日
    candidates = []
    i = len(all_dates) - 1
    while i >= 0 and len(candidates) < args.days:
        d = all_dates[i]
        d_fwd = _next_trading_day(d, args.horizon)
        if d_fwd in close.index:
            candidates.append(d)
        i -= 1
    candidates = list(reversed(candidates))  # 升序

    # 已存在日志的日期 -> 跳过（可重入）
    log = _load_json(args.out) or []
    done = {r.get("date") for r in log if isinstance(r, dict)}
    todo = [d for d in candidates if d.strftime("%Y-%m-%d") not in done]
    print(f"🗓️ 候选评分日 {len(candidates)} 个，待算 {len(todo)} 个"
          f"（已存在 {len(candidates) - len(todo)} 个）")

    if args.dry_run:
        print("🛑 dry-run：不跑模型。")
        return 0
    if not todo:
        print("✅ 全部已存在，无需回填。")
        return 0

    score_rows = []  # (date, code, score)
    fwd_rows = []    # (date, code, excess)
    t0 = datetime.now()
    for k, d in enumerate(todo, 1):
        d_fwd = _next_trading_day(d, args.horizon)
        sliced = {c: df.loc[df.index <= d] for c, df in raw.items() if df.index[0] <= d}
        try:
            preds = predictor.predict_batch(sliced)
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ 打分失败 @{d.date()}: {e}")
            preds = []
        sc = {p.code: float(p.score) for p in preds}
        c0 = close.loc[d]
        c1 = close.loc[d_fwd]
        valid = [c for c in raw if c in sc and c in c0 and c in c1
                 and pd.notna(c0[c]) and pd.notna(c1[c]) and c0[c] > 0]
        if len(valid) < args.min_stocks:
            continue
        rets = np.array([c1[c] / c0[c] - 1.0 for c in valid], dtype=float)
        excess = rets - rets.mean()  # 相对等权宇宙的超额（与训练标签口径一致）
        for c, s, e in zip(valid, [sc[c] for c in valid], excess):
            score_rows.append({"date": d, "code": c, "score": s})
            fwd_rows.append({"date": d, "code": c, "excess": e})
        if k % 5 == 0 or k == len(todo):
            print(f"   ...{k}/{len(todo)} 评分日，累计 {len(score_rows)} 条 "
                  f"({ (datetime.now()-t0).total_seconds():.0f}s)")

    if not score_rows:
        print("❌ 未产出任何有效评分，退出")
        return 1

    sp = pd.DataFrame(score_rows).pivot(index="date", columns="code", values="score")
    fr = pd.DataFrame(fwd_rows).pivot(index="date", columns="code", values="excess")

    new_entries = []
    for d in sp.index:
        s = sp.loc[d]
        r = fr.loc[d]
        v = pd.DataFrame({"s": s, "r": r}).dropna()
        if len(v) < args.min_stocks:
            continue
        ic = _spearman(v["s"].values, v["r"].values)
        if ic == ic:
            new_entries.append({"date": d.strftime("%Y-%m-%d"),
                                "ic": round(float(ic), 4),
                                "n": int(len(v)),
                                "offline": False})

    # 合并去重（按 date），保留最近 250
    merged = {r.get("date"): r for r in log if isinstance(r, dict)}
    for e in new_entries:
        merged[e["date"]] = e
    merged_list = sorted(merged.values(), key=lambda r: r.get("date", ""))
    merged_list = merged_list[-250:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged_list, ensure_ascii=False, indent=2), encoding="utf-8")

    ics = [e["ic"] for e in merged_list]
    m = float(np.mean(ics))
    sd = float(np.std(ics, ddof=1)) if len(ics) > 1 else float("nan")
    se = sd / (len(ics) ** 0.5) if len(ics) > 1 and sd == sd else float("nan")
    ir = m / sd if sd and sd > 0 else float("nan")
    tstat = m / se if se and se > 0 else float("nan")
    print(f"\n✅ 回填完成：新增 {len(new_entries)} 个评分日，日志共 {len(merged_list)} 点")
    print(f"   全样本 RankIC 均值   {m:+.4f}")
    print(f"   日频 IR (mean/std)   {ir:.2f}   ← 真正反映「单日排序稳不稳」")
    print(f"   t 统计量 (mean/SE)   {tstat:.2f}   ← production_metrics 里那个 'ic_ir=14' 就是这个口径")
    print(f"   正 IC 占比           {sum(1 for v in ics if v > 0)/len(ics)*100:.0f}%")
    print(f"   下一步：python scripts/rolling_ic_curve.py  # 画稳定性曲线")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
增量补数据：把本地日线缓存补齐到最近一个已收盘的交易日。

为什么需要单独一个脚本
--------------------
每日循环只刷新被扫描到的股票，全市场 5000+ 只里大部分会停留在
几天甚至几周前的日期。训练集因此会在时间轴末尾出现大面积缺失，
模型看到的"最近状态"是过时的。

本脚本解决三件事
----------------
1. **增量拉取**：只请求上次日期之后的部分，不全量重下（快 4 倍以上）
2. **复权连续性校验**：前复权数据在除权日会整体改写历史价格。
   若重叠日价格比值偏离 1 超过阈值，说明该股在这段时间发生了
   除权/分红，新旧数据拼接会出现跳空 —— 此时改用全量重拉，
   保证单只股票内部的复权序列自洽。
3. **断点续跑**：已到目标日期的文件直接跳过，中断后重跑不重复劳动。

为什么用多进程而不是多线程
------------------------
akshare 的新浪接口用 py_mini_racer 解码 JS，该引擎非线程安全。
线程池并发会让进程直接段错误崩溃（无 Python 异常可捕）。改用
进程池后每个 worker 独立持有 JS 引擎实例，互不干扰。

用法
----
    py -3.12 scripts/backfill_data.py                    # 补到最近已收盘交易日
    py -3.12 scripts/backfill_data.py --target 2026-11-02
    py -3.12 scripts/backfill_data.py --limit 50 --dry-run
    py -3.12 scripts/backfill_data.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
CALENDAR = DATA_DIR / "trade_calendar.json"

# 除权判定阈值：重叠日价格比值的中位数偏离 1 超过该值，视为发生了除权。
# 取 0.5%：复权因子变动通常远大于此，而两位小数舍入噪声（低价股
# 0.005/2.0 ≈ 0.25%）经过多日中位数后不会达到 0.5%
ADJUST_TOLERANCE = 0.005

# 全量重拉的最小历史覆盖率：新数据行数不足旧数据 98% 时拒绝覆盖，
# 防止数据源单方面截断历史把本地缓存写坏
MIN_COVERAGE = 0.98

_FETCHER = None


def latest_closed_trading_day() -> str:
    """最近一个已收盘的交易日（15:00 之后当天才算完整）"""
    now = datetime.now()
    days = []
    if CALENDAR.exists():
        try:
            days = json.loads(CALENDAR.read_text(encoding="utf-8")).get("trade_days", [])
        except Exception:
            days = []
    today = now.strftime("%Y-%m-%d")
    if days:
        passed = [d for d in days if d <= today]
        if passed and passed[-1] == today and now.hour < 15:
            passed = passed[:-1]
        if passed:
            return passed[-1]
    d = now if now.hour >= 15 else now - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _read_cached(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
        if len(df) == 0:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def _atomic_write(path: Path, df: pd.DataFrame) -> None:
    """先写临时文件再替换，避免中断留下半个 parquet"""
    tmp = path.parent / f".{path.name}.tmp{os.getpid()}"
    df.sort_values("date").reset_index(drop=True).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _needs_full_refetch(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """
    判断是否因除权导致历史被整体改写，需要全量重拉

    做法：取重叠交易日的 close 比值，若中位数偏离 1 超过容差，
    说明复权基准变了（dividend/split），增量的新数据与旧历史
    不在同一复权口径上，直接拼接会造出虚假跳空。
    """
    if new is None or len(new) == 0:
        return False
    m = old[["date", "close"]].merge(
        new[["date", "close"]], on="date", suffixes=("_old", "_new")
    )
    m = m[(m["close_old"] > 0) & (m["close_new"] > 0)]
    if len(m) < 2:
        return False
    ratio = (m["close_new"] / m["close_old"]).to_numpy()
    return bool(abs(float(np.median(ratio)) - 1.0) > ADJUST_TOLERANCE)


def process_one(code: str, target: str, dry_run: bool = False) -> dict:
    """处理单只股票，返回状态字典"""
    cache = DATA_DIR / f"daily_{code}_qfq.parquet"
    old = _read_cached(cache) if cache.exists() else None

    if old is not None:
        old_last = old["date"].max()
        if old_last >= pd.Timestamp(target):
            return {"code": code, "status": "fresh"}
        # 增量窗口：从上次日期回退 20 个自然日，确保有足够重叠日做除权校验
        start = (old_last - timedelta(days=20)).strftime("%Y-%m-%d")
    else:
        old_last = None
        start = "2015-01-01"

    new = _FETCHER._fetch_from_sina(code, start)
    if new is None or len(new) == 0:
        return {"code": code, "status": "no_data"}

    new["date"] = pd.to_datetime(new["date"])
    if new["date"].max() < pd.Timestamp(target):
        # 停牌 / 退市，数据源没给到目标日期，保留现状
        return {"code": code, "status": "still_stale"}

    if old is None:
        if dry_run:
            return {"code": code, "status": "would_create", "rows": len(new)}
        _atomic_write(cache, new)
        return {"code": code, "status": "created", "rows": len(new)}

    if _needs_full_refetch(old, new):
        full = _FETCHER._fetch_from_sina(code, "2015-01-01")
        if full is None or len(full) == 0:
            return {"code": code, "status": "refetch_failed"}
        full["date"] = pd.to_datetime(full["date"])
        if len(full) < len(old) * MIN_COVERAGE:
            return {"code": code, "status": "skip_truncated",
                    "old_rows": len(old), "new_rows": len(full)}
        if dry_run:
            return {"code": code, "status": "would_refetch", "rows": len(full)}
        _atomic_write(cache, full)
        return {"code": code, "status": "refetched_full", "rows": len(full)}

    if dry_run:
        return {"code": code, "status": "would_update", "rows": len(new)}

    merged = _FETCHER._merge_into_cache(cache, new)
    return {"code": code, "status": "updated", "rows": len(merged)}


def _init_worker() -> None:
    """子进程初始化：每个进程独立持有 DataFetcher 与 JS 引擎"""
    global _FETCHER
    logging.disable(logging.WARNING)  # 子进程日志会淹没进度条
    from src.data.fetcher import DataFetcher

    _FETCHER = DataFetcher()


def _run_chunk(codes: list[str], target: str, dry_run: bool,
               workers: int, retries: int) -> list[dict]:
    """跑一批代码；进程池崩溃时自动降级为更小批次重试"""
    out: list[dict] = []
    if not codes:
        return out
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futs = {ex.submit(_worker_task, c, target, dry_run, retries): c for c in codes}
            for fut in as_completed(futs):
                try:
                    out.append(fut.result())
                except Exception as e:
                    out.append({"code": futs[fut], "status": "error",
                                "error": f"{type(e).__name__}: {e}"})
    except BrokenProcessPool:
        # 某个 worker 被底层 C 库带崩，把本批拆小重试；单只再崩就标记失败
        if len(codes) > 1:
            mid = len(codes) // 2
            sub_w = max(1, workers // 2)
            out.extend(_run_chunk(codes[:mid], target, dry_run, sub_w, retries))
            out.extend(_run_chunk(codes[mid:], target, dry_run, sub_w, retries))
        else:
            out.append({"code": codes[0], "status": "error",
                        "error": "BrokenProcessPool"})
    return out


def _worker_task(code: str, target: str, dry_run: bool, retries: int) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            return process_one(code, target, dry_run)
        except Exception as e:  # 网络抖动重试
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.8 * (attempt + 1))
    return {"code": code, "status": "error", "error": last_err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None, help="目标交易日，默认取最近已收盘交易日")
    ap.add_argument("--workers", type=int, default=8, help="并发进程数（4-10 较安全）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只，用于试跑")
    ap.add_argument("--chunk", type=int, default=400, help="每批代码数，崩溃时只损失一批")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="只检查不写盘")
    ap.add_argument("--only-existing", action="store_true",
                    help="只更新已有文件，不拉本地尚无数据的股票")
    args = ap.parse_args()

    target = args.target or latest_closed_trading_day()
    print(f"目标交易日: {target}")
    print(f"并发进程: {args.workers}   dry-run: {args.dry_run}")

    existing = {f.name.split("_")[1] for f in DATA_DIR.glob("daily_*_qfq.parquet")}
    codes = set(existing)
    if not args.only_existing:
        sl_path = DATA_DIR / "stock_list.parquet"
        if sl_path.exists():
            sl = pd.read_parquet(sl_path)
            codes |= set(sl["code"].astype(str).str.zfill(6))
    codes = sorted(codes)
    if args.limit:
        codes = codes[: args.limit]

    print(f"待扫描: {len(codes)} 只（本地已有 {len(existing)} 只）")

    # 预扫描：先剔掉已是最新的，避免白跑网络
    t0 = time.time()
    stale: list[str] = []
    for code in codes:
        p = DATA_DIR / f"daily_{code}_qfq.parquet"
        if p.exists():
            try:
                d = pd.read_parquet(p, columns=["date"])
                if pd.to_datetime(d["date"]).max() >= pd.Timestamp(target):
                    continue
            except Exception:
                pass
        stale.append(code)
    print(f"需要更新: {len(stale)} 只 / {len(codes)} 只"
          f"（预扫描 {time.time() - t0:.1f}s）")

    if not stale:
        print("全部已是最新，无需补数")
        return 0

    results: list[dict] = []
    t_start = time.time()
    chunks = [stale[i:i + args.chunk] for i in range(0, len(stale), args.chunk)]
    for ci, ch in enumerate(chunks, 1):
        results.extend(_run_chunk(ch, target, args.dry_run, args.workers, args.retries))
        done = len(results)
        el = time.time() - t_start
        rate = done / el if el > 0 else 0
        eta = (len(stale) - done) / rate / 60 if rate > 0 else 0
        bad = sum(1 for r in results if r["status"] in
                  ("error", "no_data", "refetch_failed", "skip_truncated"))
        print(f"  批次 {ci}/{len(chunks)}  {done}/{len(stale)}  "
              f"用时 {el/60:.1f}min  均 {1/rate:.2f}s/只  "
              f"剩余约 {eta:.0f}min  异常 {bad}", flush=True)

    df = pd.DataFrame(results)
    print("\n=== 结果统计 ===")
    for status, cnt in df["status"].value_counts().items():
        print(f"  {status:<16} {cnt}")

    fails = df[df["status"].isin(["error", "no_data", "refetch_failed", "skip_truncated"])]
    if len(fails):
        print(f"\n未完成 {len(fails)} 只（前 20）:")
        for _, r in fails.head(20).iterrows():
            print(f"  {r['code']}  {r['status']}")

    report = {
        "target": target,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round((time.time() - t_start) / 60, 1),
        "workers": args.workers,
        "dry_run": args.dry_run,
        "counts": {k: int(v) for k, v in df["status"].value_counts().items()},
        "failed": [{"code": r["code"], "status": r["status"]}
                   for _, r in fails.iterrows()],
    }
    rp = DATA_DIR / "backfill_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {rp}")
    return 0 if len(fails) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

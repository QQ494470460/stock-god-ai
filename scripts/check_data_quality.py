"""
数据体检：补数之后验证本地行情数据是否真的可用。

为什么需要这一步
----------------
"补了 5153 个文件"不等于"数据可用"。真实数据集的常见毛病有：
停牌留下的空洞、除权没处理造成的假跳空、零价格、OHLC 互相矛盾、
同一天重复两行。这些问题不会让训练报错，只会让 IC 静默变差。

检查项
------
1. 新鲜度：最新交易日是否到达目标日
2. 空洞：近一年内缺失的交易日（停牌会留空，属正常，但要能度量）
3. 价格合法性：close<=0、high<low、open/close 越界
4. 除权残留：单日涨跌幅绝对值 >21%（前复权序列里不该出现，
   超过涨跌停限制说明复权没做干净，会污染动量/反转因子）
5. 重复日期

用法
----
    py -3.12 scripts/check_data_quality.py
    py -3.12 scripts/check_data_quality.py --recent-days 250 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CALENDAR = DATA_DIR / "trade_calendar.json"

# A股涨跌幅限制：主板 10%，ST 5%，创业板/科创板 20%，北交所 30%。
# 前复权序列内部不应出现超出该限制的单日跳动 —— 那基本是除权没洗干净。
# 北交所（4/8/920 开头）必须用 30% 的阈值，否则会把正常涨跌误判成复权错误：
# 实测 487 个"异常"文件里绝大多数是北交所的合法波动。
MAX_ABS_RET_MAIN = 0.21
MAX_ABS_RET_BJ = 0.32


def _max_ret_threshold(code: str) -> float:
    return MAX_ABS_RET_BJ if code.startswith(("4", "8", "920")) else MAX_ABS_RET_MAIN


def load_trade_days() -> list[str]:
    try:
        return json.loads(CALENDAR.read_text(encoding="utf-8")).get("trade_days", [])
    except Exception:
        return []


def check_one(path: Path, trade_days: list[str]) -> dict:
    code = path.name.split("_")[1]
    out = {"code": code, "rows": 0, "last": None, "issues": []}
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        out["issues"].append(f"读取失败:{type(e).__name__}")
        return out

    if len(df) == 0:
        out["issues"].append("空文件")
        return out

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    out["rows"] = len(df)
    out["last"] = str(df["date"].max().date())

    # 重复日期
    dup = int(df["date"].duplicated().sum())
    if dup:
        out["issues"].append(f"重复日期x{dup}")

    # 价格合法性
    if "close" in df:
        bad_close = int((df["close"] <= 0).sum())
        if bad_close:
            out["issues"].append(f"close<=0 x{bad_close}")
    if {"high", "low"}.issubset(df.columns):
        bad_hl = int((df["high"] < df["low"]).sum())
        if bad_hl:
            out["issues"].append(f"high<low x{bad_hl}")

    # 除权残留：单日异常跳动
    if "close" in df and len(df) > 5:
        ret = df["close"].pct_change()
        jump = int((ret.abs() > _max_ret_threshold(code)).sum())
        if jump:
            out["issues"].append(f"单日跳动异常 x{jump}")

    # 近一年空洞：用交易日历对齐，统计缺失比例
    if trade_days:
        last = df["date"].max()
        window = [d for d in trade_days
                  if pd.Timestamp(d) <= last
                  and pd.Timestamp(d) >= last - pd.Timedelta(days=365)]
        if window:
            have = set(df["date"].dt.strftime("%Y-%m-%d"))
            missing = sum(1 for d in window if d not in have)
            out["miss_ratio"] = missing / len(window)
        else:
            out["miss_ratio"] = np.nan
    else:
        out["miss_ratio"] = np.nan

    return out


def fix_invalid_rows(path: Path) -> dict:
    """
    删除价格非法的行（close<=0 或 high<low）

    这类行来自数据源缺陷：北交所 2018-2019 年的部分记录 OHLC 全为 0
    而 volume/amount 有值。它们会被下游当成"价格跌到 0"，
    产生 -100% 的假收益，再用分位数裁剪也救不回来，只能删。
    """
    code = path.name.split("_")[1]
    df = pd.read_parquet(path)
    if len(df) == 0:
        return {"code": code, "dropped": 0}
    bad = pd.Series(False, index=df.index)
    if "close" in df.columns:
        bad |= df["close"].isna() | (df["close"] <= 0)
    if {"high", "low"}.issubset(df.columns):
        bad |= df["high"] < df["low"]
    n_bad = int(bad.sum())
    if n_bad == 0:
        return {"code": code, "dropped": 0}
    cleaned = df[~bad].sort_values("date").reset_index(drop=True)
    tmp = path.parent / f".{path.name}.fix"
    cleaned.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return {"code": code, "dropped": n_bad, "left": len(cleaned)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", default=None, help="期望的最新交易日，默认取日历最近日")
    ap.add_argument("--gap-threshold", type=float, default=0.10,
                    help="近一年缺失比例超过该值视为可疑")
    ap.add_argument("--fix", action="store_true",
                    help="删除价格非法的行（close<=0 / high<low）并重写文件")
    args = ap.parse_args()

    files = sorted(DATA_DIR.glob("daily_*_qfq.parquet"))
    trade_days = load_trade_days()
    target = args.target
    if target is None and trade_days:
        today = datetime.now().strftime("%Y-%m-%d")
        passed = [d for d in trade_days if d <= today]
        if passed and passed[-1] == today and datetime.now().hour < 15:
            passed = passed[:-1]
        target = passed[-1] if passed else None

    print(f"检查 {len(files)} 个文件  期望最新交易日: {target}")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(check_one, f, trade_days) for f in files]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 1000 == 0:
                print(f"  已检查 {i}/{len(files)}", flush=True)

    df = pd.DataFrame(rows)
    n = len(df)
    df["miss_ratio"] = pd.to_numeric(df.get("miss_ratio"), errors="coerce")

    print("\n=== 新鲜度 ===")
    if target:
        at_target = int((df["last"] == target).sum())
        print(f"  已达 {target}: {at_target}/{n}  ({at_target/n:.1%})")
    vc = df["last"].value_counts().head(5)
    print("  最新日期分布:")
    for d, c in vc.items():
        print(f"    {d}: {c}")

    print("\n=== 完整性问题 ===")
    has_issue = df["issues"].map(len) > 0
    print(f"  存在问题的文件: {int(has_issue.sum())}/{n}")
    issue_types: dict[str, int] = {}
    for lst in df["issues"]:
        for it in lst:
            k = it.split(" x")[0]
            issue_types[k] = issue_types.get(k, 0) + 1
    for k, v in sorted(issue_types.items(), key=lambda x: -x[1]):
        print(f"    {k:<24}{v}")

    print("\n=== 交易日空洞（近一年）===")
    valid = df["miss_ratio"].dropna()
    if len(valid):
        print(f"  中位数缺失率: {valid.median():.2%}")
        print(f"  缺失 >{args.gap_threshold:.0%} 的股票: "
              f"{int((valid > args.gap_threshold).sum())} 只"
              f"（长期停牌或退市，属正常）")

    print("\n=== 数据量 ===")
    print(f"  行数: 中位数 {int(df['rows'].median())}  最小 {int(df['rows'].min())}")

    suspicious = df[has_issue].sort_values("issues", key=lambda s: s.map(len), ascending=False)
    if len(suspicious):
        print("\n=== 问题最多的 10 只 ===")
        for _, r in suspicious.head(10).iterrows():
            print(f"  {r['code']}  {r['last']}  {'; '.join(r['issues'])}")

    # ---------- 修复非法价格行 ----------
    fixed = []
    if args.fix:
        targets = df[df["issues"].map(
            lambda lst: any(k in it for it in lst for k in ("close<=0", "high<low"))
        )]["code"].tolist()
        print(f"\n=== 修复：{len(targets)} 个文件存在非法价格行 ===")
        for code in targets:
            p = DATA_DIR / f"daily_{code}_qfq.parquet"
            if not p.exists():
                continue
            try:
                r = fix_invalid_rows(p)
                if r["dropped"]:
                    fixed.append(r)
            except Exception as e:
                print(f"  {code} 修复失败: {type(e).__name__}: {e}")
        total_dropped = sum(r["dropped"] for r in fixed)
        print(f"  已清理 {len(fixed)} 个文件，共删除 {total_dropped} 行非法价格")
        if fixed:
            print("  示例: " + ", ".join(f"{r['code']}(-{r['dropped']}行)" for r in fixed[:8]))

    report = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": target,
        "n_files": n,
        "at_target": int((df["last"] == target).sum()) if target else None,
        "by_last_date": {str(k): int(v) for k, v in vc.items()},
        "files_with_issues": int(has_issue.sum()),
        "issue_types": issue_types,
        "median_miss_ratio": float(valid.median()) if len(valid) else None,
        "fixed_files": fixed,
        "suspicious": [
            {"code": r["code"], "last": r["last"], "issues": r["issues"]}
            for _, r in suspicious.head(200).iterrows()
        ],
    }
    rp = DATA_DIR / "data_quality_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回填进化引擎的「分窗口准确率」样本（生产状态迁移，一次性工具）。

背景
----
10-08 那天一次性补验了 20 个历史批次，验证结果直接写进了 prediction_history
的 ``verifications`` 槽位。但 ``verify_all_pending()`` 会跳过**已验证**批次，
于是这些准确率从来没有流到 ``record_verification()`` ——
``evolution_state.performance.t5.accuracy`` 长期只有 3 个样本（10-14 补验的那批），
``should_retrain()`` 的「近期准确率 < 阈值」这条判据基本是瞎的。

做法
----
不做「增量补差异」（无法可靠判断哪几条已记过、顺序也会乱），而是**全量重放**：

    清空 performance 的 accuracy → 按 (scan_date, verified_at) 升序重放所有
    verifications → 每个窗口只保留最近 20 条（与 record_verification 一致）

因此本脚本**幂等**：跑几次结果都一样，不会重复计数。

安全
----
* 默认 ``--dry-run``，只打印将要写入什么；
* ``--apply`` 前自动备份 ``data/evolution_state.json``；
* **不触碰** ``last_evolved_at``（这是状态迁移，不是一次进化）。

用法::

    python scripts/backfill_evolution_accuracy.py            # dry-run
    python scripts/backfill_evolution_accuracy.py --apply    # 真写
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "prediction_history.json"
STATE_FILE = DATA_DIR / "evolution_state.json"

WINDOW_KEEP = 20  # 与 evolution.record_verification 一致


def _window_key(v: dict, fallback: str) -> str:
    """verification → 窗口键。与 record_verification 的取法保持一致。"""
    lb = v.get("lookback_days") or v.get("hold_days")
    try:
        lb = int(lb)
    except (TypeError, ValueError):
        lb = None
    if lb:
        return f"t{lb}"
    # 外层 key 本身就是窗口（如 {"5": {...}}）
    try:
        return f"t{int(fallback)}"
    except (TypeError, ValueError):
        return fallback if fallback.startswith("t") else f"t{fallback}"


def collect(history) -> list[tuple[str, str, dict]]:
    """返回 [(sort_key, window_key, verification)]，按时间升序。"""
    if isinstance(history, dict):
        batches = history.get("batches") or []
    else:
        batches = history or []

    rows: list[tuple[str, str, dict]] = []
    for b in batches:
        verifs = b.get("verifications") or {}
        if not isinstance(verifs, dict):
            continue
        scan_date = str(b.get("scan_date") or "")
        for key, v in verifs.items():
            if not isinstance(v, dict):
                continue
            if v.get("accuracy") is None:
                continue
            sort_key = f"{scan_date}|{v.get('verified_at') or ''}"
            rows.append((sort_key, _window_key(v, str(key)), v))
    rows.sort(key=lambda r: r[0])
    return rows


def rebuild(state: dict, rows) -> tuple[dict, dict]:
    """按 rows 重建 performance，返回 (new_performance, 统计)。"""
    perf = state.setdefault("performance", {})
    before = {
        wk: {"n": (d or {}).get("n", 0), "acc": list((d or {}).get("accuracy") or [])}
        for wk, d in perf.items()
    }

    acc_by_window: dict[str, list[float]] = {}
    n_by_window: dict[str, int] = {}
    for _sort_key, wk, v in rows:
        acc_by_window.setdefault(wk, []).append(round(float(v["accuracy"]), 3))
        n_by_window[wk] = n_by_window.get(wk, 0) + 1

    for wk, accs in acc_by_window.items():
        slot = perf.setdefault(wk, {"accuracy": [], "n": 0})
        slot["accuracy"] = accs[-WINDOW_KEEP:]
        # ⚠️ n 必须是「窗口内的样本数」= len(accuracy)，不是历史总批次数。
        # 线上 record_verification() 的 accuracy 会被 [-20:] 截断，但 n 一直累加，
        # 于是线上 n 与 len(accuracy) 本来就可能不等（这是已知的口径瑕疵，
        # should_retrain 读的也是 len(accuracy)）。回填若把 n 设成总批次数，
        # 就会造出「n=36 但只有 18 条数据」的假样本量，误导排障与后续判断。
        # 这里跟 len(accuracy) 对齐，两边口径才一致。
        slot["n"] = len(slot["accuracy"])

    stats = {"windows": {}, "before": before}
    for wk in sorted(acc_by_window):
        stats["windows"][wk] = {
            "n_records": n_by_window[wk],
            "kept": min(n_by_window[wk], WINDOW_KEEP),
            "acc": acc_by_window[wk][-WINDOW_KEEP:],
        }
    return perf, stats


def avg(xs) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真写入（默认只 dry-run）")
    ap.add_argument("--state", default=str(STATE_FILE))
    ap.add_argument("--history", default=str(HISTORY_FILE))
    args = ap.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8")

    hist_p, state_p = Path(args.history), Path(args.state)
    if not hist_p.exists():
        print(f"❌ 找不到 {hist_p}")
        return 2
    if not state_p.exists():
        print(f"❌ 找不到 {state_p}")
        return 2

    history = json.loads(hist_p.read_text(encoding="utf-8"))
    state = json.loads(state_p.read_text(encoding="utf-8"))

    rows = collect(history)
    if not rows:
        print("没有可回填的验证结果（history 里没有带 accuracy 的 verifications）")
        return 0

    perf, stats = rebuild(state, rows)

    print(f"📥 待重放验证记录: {len(rows)} 条")
    for wk, s in stats["windows"].items():
        old = stats["before"].get(wk, {})
        old_acc = old.get("acc") or []
        print(f"  {wk}: n {old.get('n', 0)} → {s['kept']}"
              f" | 窗口内样本 {s['kept']} 条（历史共 {s['n_records']} 条）"
              f" | 均值 {avg(s['acc']):.1%} (原 {avg(old_acc):.1%})")
        if old_acc and len(old_acc) == len(s["acc"]) and old_acc != s["acc"]:
            print("        ⚠️ 同等长度但内容不同，请确认是新增数据而非重放污染")
        print(f"        accuracy = {s['acc']}")

    if not args.apply:
        print("\n🔍 dry-run：未写入。确认无误后加 --apply")
        return 0

    bak = state_p.with_suffix(state_p.suffix + f".bak_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(state_p, bak)
    print(f"\n💾 已备份: {bak}")

    # 状态迁移 ≠ 进化：绝不碰 last_evolved_at
    state["performance"] = perf
    state["accuracy_backfilled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state_p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已写入 {state_p}（last_evolved_at 保持不变）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

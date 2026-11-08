#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_ptrade_config.py — 把「生产持仓 + 最新计划候选」烘焙成 Ptrade 策略的
UNIVERSE / SECTOR_MAP 字面量，写回 src/trading/ptrade_strategy.py 的标记区间。

为什么需要它：
- Ptrade 是券商机房里的服务端策略引擎，**不能 import 本地 src/，也不能读本地文件**。
- 因此 UNIVERSE / SECTOR_MAP 必须是「写死在策略文件里的字面量」，而不是运行时读取。
- 本脚本在你本地（项目根）跑，从真实生产数据生成这些字面量，再贴回策略文件。
- 计划每天更新、持仓每天变动 —— 重跑本脚本即可刷新候选池，无需手改。

用法：
    python scripts/gen_ptrade_config.py            # 仅打印将要写入的内容 + 报告
    python scripts/gen_ptrade_config.py --write    # 直接写回 ptrade_strategy.py

来源（全部只读）：
    data/paper_account.json         当前真实持仓（"当前生产持仓"）
    data/daily_reports/daily_picks_*.json  (按 mtime 取最新) 计划 picks + ranking 候选
    data/industry_map.json          6 位代码 -> 行业名
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_PATH = os.path.join(ROOT, "src", "trading", "ptrade_strategy.py")
MARK_START = "# === PTrade auto-generated config START"
MARK_END = "# === PTrade auto-generated config END"
UNKNOWN = "未知"


def to_ptrade_code(code: str) -> str:
    """6 位 A 股代码 -> Ptrade 格式 '600570.SS' / '000001.SZ' / '8xxxxx.BJ'。"""
    c = str(code).strip()
    if c.endswith((".SS", ".SZ", ".BJ", ".SH")):
        return c
    if c.startswith(("60", "68", "90", "91", "92")):
        return c + ".SS"          # 上海主板 / 科创板 / B 股
    if c.startswith(("00", "30", "20", "03")):
        return c + ".SZ"          # 深圳主板 / 创业板 / 中小板
    if c[0] in "48":
        return c + ".BJ"          # 北交所
    return c + ".SS"              # 兜底上海


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def latest_plan_path() -> str | None:
    plans = sorted(
        glob.glob(os.path.join(ROOT, "data", "daily_reports", "daily_picks_*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    return plans[0] if plans else None


def gather_codes() -> tuple[list[str], str]:
    """返回 (候选 6 位代码去重列表, 计划来源说明)。"""
    codes: list[str] = []
    note = ""

    # 1) 当前真实持仓
    pa_path = os.path.join(ROOT, "data", "paper_account.json")
    if os.path.exists(pa_path):
        pa = load_json(pa_path)
        holds = list(pa.get("holdings", {}).keys())
        codes.extend(holds)

    # 2) 最新计划的 picks + ranking
    plan_path = latest_plan_path()
    if plan_path:
        plan = load_json(plan_path)
        picks = [p["代码"] for p in plan.get("picks", []) if "代码" in p]
        ranking = [r["代码"] for r in plan.get("ranking", []) if "代码" in r]
        codes.extend(picks)
        codes.extend(ranking)
        note = f"{os.path.basename(plan_path)} (pool={plan.get('pool')})"

    # 去重保序
    seen, uniq = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq, note


def build_block(codes: list[str], industry: dict) -> tuple[str, list[str]]:
    universe = [to_ptrade_code(c) for c in codes]
    sector_map = {}
    unmapped = []
    for c in codes:
        s = industry.get(c)
        if not s:
            s = UNKNOWN
            unmapped.append(c)
        sector_map[c] = s

    lines = []
    lines.append("# 股票池：由「生产持仓 + 最新计划候选」生成（Ptrade 格式 '600570.SS' / '000001.SZ'）。")
    lines.append("# 这是截至生成时刻的候选池；计划每日更新，重跑生成器即可刷新。")
    lines.append("UNIVERSE = [")
    # 每行 4 个，便于审阅
    for i in range(0, len(universe), 4):
        chunk = universe[i:i + 4]
        lines.append("    " + ", ".join(f'"{u}"' for u in chunk) + ",")
    lines.append("]")
    lines.append("")
    lines.append("# 行业映射：键为 6 位代码（去后缀），值为行业名；缺映射的票归 \"未知\"（集中度对其 fail-open）。")
    lines.append("SECTOR_MAP = {")
    for c in codes:
        lines.append(f'    "{c}": "{sector_map[c]}",')
    lines.append("}")
    return "\n".join(lines), unmapped


def regenerate(text: str, block: str) -> str:
    """把 MARK_START/END 之间的内容替换为 block。"""
    s = text.index(MARK_START)
    e = text.index(MARK_END) + len(MARK_END)
    return text[:s] + MARK_START + "\n" + block + "\n" + MARK_END + text[e:]


def main() -> int:
    write = "--write" in sys.argv[1:]

    industry_path = os.path.join(ROOT, "data", "industry_map.json")
    industry = load_json(industry_path) if os.path.exists(industry_path) else {}

    codes, note = gather_codes()
    if not codes:
        print("[gen_ptrade_config] 未找到任何候选代码（无持仓也无计划），退出。", file=sys.stderr)
        return 1

    block, unmapped = build_block(codes, industry)

    print(f"[gen_ptrade_config] 候选池来源: {note or '(无计划文件)'}")
    print(f"[gen_ptrade_config] UNIVERSE 数量: {len(codes)}")
    print(f"[gen_ptrade_config] 行业未映射(归'{UNKNOWN}'): {len(unmapped)} -> {unmapped}")
    print("-" * 60)
    print(block)
    print("-" * 60)

    if not write:
        print("[gen_ptrade_config] 未加 --write，仅预览。不会修改任何文件。")
        return 0

    if not os.path.exists(STRATEGY_PATH):
        print(f"[gen_ptrade_config] 找不到策略文件: {STRATEGY_PATH}", file=sys.stderr)
        return 1

    text = open(STRATEGY_PATH, "r", encoding="utf-8").read()
    if MARK_START not in text or MARK_END not in text:
        print("[gen_ptrade_config] 策略文件缺少标记区间，无法写入。", file=sys.stderr)
        return 1

    new_text = regenerate(text, block)
    with open(STRATEGY_PATH, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    print(f"[gen_ptrade_config] 已写回 {STRATEGY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

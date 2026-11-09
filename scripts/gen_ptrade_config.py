#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_ptrade_config.py — 把「生产持仓 + 最新计划候选」烘焙成 Ptrade 策略的
UNIVERSE / SECTOR_MAP 字面量，写回 src/trading/ptrade_strategy.py。

为什么需要它：
- Ptrade 是券商机房里的服务端策略引擎，**不能 import 本地 src/，也不能读本地文件**。
- 因此 UNIVERSE / SECTOR_MAP 必须是「写死在策略文件里的字面量」，而不是运行时读取。
- 本脚本在你本地（项目根）跑，从真实生产数据生成这些字面量，再贴回策略文件。
- 计划每天更新、持仓每天变动 —— 重跑本脚本即可刷新候选池，无需手改。

用法：
    python scripts/gen_ptrade_config.py            # 仅打印将要写入的内容 + 报告
    python scripts/gen_ptrade_config.py --write    # 直接写回 ptrade_strategy.py

来源（全部只读）：
    data/paper_account.json         当前真实持仓
    data/daily_reports/daily_picks_*.json  (按 mtime 取最新) 计划 picks + ranking 候选
    data/industry_map.json          6 位代码 -> 行业名

写回方式：
    不依赖任何标记注释块，改用正则精准替换策略文件中的
    ``UNIVERSE = [ ... ]`` 与 ``SECTOR_MAP = { ... }`` 两个字面量块（均为扁平字面量，
    内部无嵌套括号），因此策略文件本身保持「纯净、可直接粘贴进 Ptrade 终端」的形态。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_PATH = os.path.join(ROOT, "src", "trading", "ptrade_strategy.py")
UNKNOWN = "未知"

# 扁平字面量：UNIVERSE 为字符串列表、SECTOR_MAP 为 字符串->字符串 字典，
# 内部均不含嵌套括号，故可用「到首个收尾括号」的方式整体替换。
UNIVERSE_RE = re.compile(r"UNIVERSE\s*=\s*\[[^\]]*\]", re.DOTALL)
SECTOR_RE = re.compile(r"SECTOR_MAP\s*=\s*\{[^}]*\}", re.DOTALL)


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


def build_universe_block(codes: list[str]) -> str:
    """生成 `UNIVERSE = [ ... ]` 字面量（不含任何注释）。"""
    universe = [to_ptrade_code(c) for c in codes]
    lines = ["UNIVERSE = ["]
    # 每行 4 个，便于审阅
    for i in range(0, len(universe), 4):
        chunk = universe[i:i + 4]
        lines.append("    " + ", ".join(f'"{u}"' for u in chunk) + ",")
    lines.append("]")
    return "\n".join(lines)


def build_sector_block(codes: list[str], industry: dict) -> tuple[str, list[str]]:
    """生成 `SECTOR_MAP = { ... }` 字面量（不含任何注释），返回 (块文本, 未映射列表)。"""
    sector_map = {}
    unmapped = []
    for c in codes:
        s = industry.get(c)
        if not s:
            s = UNKNOWN
            unmapped.append(c)
        sector_map[c] = s
    lines = ["SECTOR_MAP = {"]
    for c in codes:
        lines.append(f'    "{c}": "{sector_map[c]}",')
    lines.append("}")
    return "\n".join(lines), unmapped


def regenerate(text: str, uni_block: str, sec_block: str) -> str:
    """用正则精准替换 UNIVERSE / SECTOR_MAP 两个字面量块。"""
    new_text = UNIVERSE_RE.sub(uni_block, text)
    new_text = SECTOR_RE.sub(sec_block, new_text)
    return new_text


def main() -> int:
    write = "--write" in sys.argv[1:]

    industry_path = os.path.join(ROOT, "data", "industry_map.json")
    industry = load_json(industry_path) if os.path.exists(industry_path) else {}

    codes, note = gather_codes()
    if not codes:
        print("[gen_ptrade_config] 未找到任何候选代码（无持仓也无计划），退出。", file=sys.stderr)
        return 1

    uni_block = build_universe_block(codes)
    sec_block, unmapped = build_sector_block(codes, industry)

    print(f"[gen_ptrade_config] 候选池来源: {note or '(无计划文件)'}")
    print(f"[gen_ptrade_config] UNIVERSE 数量: {len(codes)}")
    print(f"[gen_ptrade_config] 行业未映射(归'{UNKNOWN}'): {len(unmapped)} -> {unmapped}")
    print("-" * 60)
    print(uni_block)
    print("")
    print(sec_block)
    print("-" * 60)

    if not write:
        print("[gen_ptrade_config] 未加 --write，仅预览。不会修改任何文件。")
        return 0

    if not os.path.exists(STRATEGY_PATH):
        print(f"[gen_ptrade_config] 找不到策略文件: {STRATEGY_PATH}", file=sys.stderr)
        return 1

    text = open(STRATEGY_PATH, "r", encoding="utf-8").read()
    if not UNIVERSE_RE.search(text) or not SECTOR_RE.search(text):
        print("[gen_ptrade_config] 策略文件中未找到 UNIVERSE / SECTOR_MAP 字面量块，无法写入。", file=sys.stderr)
        return 1

    new_text = regenerate(text, uni_block, sec_block)
    if new_text == text:
        print("[gen_ptrade_config] 内容无变化，无需写回。")
        return 0
    with open(STRATEGY_PATH, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    print(f"[gen_ptrade_config] 已写回 {STRATEGY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

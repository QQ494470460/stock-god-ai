# -*- coding: utf-8 -*-
"""超级集成「真正生效」耐久运行器（第四 / 五 / 六腿）

在**耐久上下文**跑（本地双击 enable_super_ensemble.bat，或注册每日 15:10 定时任务）：
  1. 训练 Transformer 第四腿 checkpoint -> models/transformer_best.pt
     （该腿无需开关：checkpoint 存在即由 EnsemblePredictor 自动启用）
  2. 生成各腿样本外 RankIC 基线 -> data/leg_rankic.json
  3. 产物齐备后翻开关（仅本项目 / 已存在的 dist 配置副本同步改）：
     - ensemble_dynamic_weights = True   （第六腿，依赖 leg_rankic.json）
     - enable_llm_alpha_leg = True       仅当检测到 DEEPSEEK_API_KEY / LLM_API_KEY（第五腿）
     任一步失败都不影响其余步骤；缺产物则对应腿保持关闭（优雅降级）。

⚠️ 这是长任务：Transformer 训练在 CPU 上可能数小时，leg_rankic 需联网拉行情。
agent 会话会回收长任务，必须在耐久上下文（终端 / 定时任务）启动。

用法：
  python scripts/enable_super_ensemble.py [--dry-run]
      [--epochs 40] [--max-stocks 150] [--days 30] [--sample 150]
      [--force-retrain] [--force-regen]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
CFG = ROOT / "config" / "settings.py"
DIST_CFG = ROOT / "dist" / "选股神AI" / "_internal" / "config" / "settings.py"


def _run(cmd: list) -> int:
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call(cmd)


def _set_flag(path: Path, name: str, value: bool) -> str:
    """把 config/settings.py 里 `name: bool = ...` 的默认值改成 value。返回状态串。"""
    if not path.exists():
        return f"(跳过 {path.name}: 不存在)"
    text = path.read_text(encoding="utf-8")
    pat = re.compile(rf"(\n[ \t]*{name}: bool = )(True|False)")
    new, n = pat.subn(lambda m: m.group(1) + ("True" if value else "False"), text)
    if n == 0:
        return f"(跳过 {path.name}: 未找到 {name})"
    if new == text:
        return f"({path.name}: {name} 已是 {value})"
    path.write_text(new, encoding="utf-8")
    return f"({path.name}: {name} -> {value})"


def _flip_flags(dynamic: bool, llm: bool) -> list:
    msgs = []
    for p in (CFG, DIST_CFG):
        msgs.append(_set_flag(p, "ensemble_dynamic_weights", dynamic))
        msgs.append(_set_flag(p, "enable_llm_alpha_leg", llm))
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser(description="超级集成第四/五/六腿耐久生效运行器")
    ap.add_argument("--dry-run", action="store_true", help="只打印会做什么，不跑长任务、不改配置")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-stocks", type=int, default=150)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--force-retrain", action="store_true", help="即使 transformer_best.pt 已存在也重训")
    ap.add_argument("--force-regen", action="store_true", help="即使 leg_rankic.json 已存在也重算")
    args = ap.parse_args()

    py = sys.executable
    ckpt = MODEL_DIR / "transformer_best.pt"
    leg = DATA_DIR / "leg_rankic.json"

    print("=" * 64)
    print("超级集成耐久生效运行器", "(DRY-RUN)" if args.dry_run else "")
    print(f"项目根: {ROOT}")
    print(f"Transformer checkpoint: {ckpt}")
    print(f"Leg RankIC 统计:        {leg}")
    print("=" * 64)

    # ---- 步骤 1：训练 Transformer 第四腿 ----
    if ckpt.exists() and not args.force_retrain:
        print("[1/3] Transformer checkpoint 已存在，跳过训练（--force-retrain 可强制）。")
    else:
        if args.dry_run:
            print(f"[1/3] DRY-RUN: 将跑 `train_transformer.py --epochs {args.epochs} "
                  f"--max-stocks {args.max_stocks} --seq-len {args.seq_len}`")
        else:
            print("[1/3] 训练 Transformer 第四腿（CPU，可能数小时）...")
            rc = _run([py, str(ROOT / "scripts" / "train_transformer.py"),
                       "--epochs", str(args.epochs),
                       "--max-stocks", str(args.max_stocks),
                       "--seq-len", str(args.seq_len)])
            print(f"      train_transformer 退出码: {rc}")

    # ---- 步骤 2：生成各腿 RankIC 基线 ----
    if leg.exists() and not args.force_regen:
        print("[2/3] leg_rankic.json 已存在，跳过生成（--force-regen 可强制）。")
    else:
        if args.dry_run:
            print(f"[2/3] DRY-RUN: 将跑 `compute_leg_rankic.py --days {args.days} "
                  f"--sample {args.sample}`")
        else:
            print("[2/3] 生成各腿样本外 RankIC 基线（需联网拉行情）...")
            rc = _run([py, str(ROOT / "scripts" / "compute_leg_rankic.py"),
                       "--days", str(args.days),
                       "--sample", str(args.sample)])
            print(f"      compute_leg_rankic 退出码: {rc}")

    # ---- 步骤 3：产物齐备后翻开关 ----
    has_ckpt = ckpt.exists()
    has_leg = leg.exists()
    llm_key = bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY"))

    print("[3/3] 判定启用状态：")
    print(f"      第四腿(Transformer): {'✅ 已启用' if has_ckpt else '⚠️ 未启用（缺 transformer_best.pt）'}")
    print(f"      第六腿(动态融合):     {'✅ 就绪（将开启）' if has_leg else '⚠️ 未就绪（缺 leg_rankic.json）'}")
    print(f"      第五腿(LLM alpha):    {'✅ 有 key，将开启' if llm_key else '⚠️ 无 DEEPSEEK_API_KEY/LLM_API_KEY，保持关闭（不注入伪评分）'}")

    if args.dry_run:
        print("[3/3] DRY-RUN: 不修改任何配置。")
        return 0

    flip_dynamic = has_leg
    flip_llm = has_leg and llm_key  # 第五腿还需 leg_rankic 存在才合理开启
    # 注：第五腿独立依赖 key，不强制要求 leg；但为稳妥，仅当有 key 即开（无 key 会自动降级）
    flip_llm = llm_key
    msgs = _flip_flags(dynamic=flip_dynamic, llm=flip_llm)
    print("      配置变更:")
    for m in msgs:
        print("        " + m)

    print("\n完成。下一次 loop 重启（约 01:00）或手动重启后生效。")
    print("若 key / checkpoint 缺失，对应腿会自动降级，不影响主模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""T4 自主进化闭环 · 漂移触发重训。

它在整条链路里的位置
====================
    每日 RankIC 回填（rolling_ic）
        → 本脚本判断「现任模型是否已漂移」
        → 漂移才跑生成器 C（重训）
        → 进化闸门（A/B + walk-forward + 回测-实盘一致性）
        → **待批准候选**（绝不自动上线）
        → 人工 `scripts/evolve.py --approve <id> --apply <id>`

为什么「绝不自动上线」是设计而不是偷懒
======================================
换模型会让**之后所有选股**都变，而且不可逆（旧模型的预测已经发出去了）。
在只有 20~60 个交易日真实样本的情况下，「自动上线」等于用噪声改策略——
这正是项目里 `gate.py` 第五层（回测-实盘一致性）存在的理由。
所以本脚本的产出永远是「一条待批准候选 + 一份可复核的判定依据」。

三条硬约束
==========
1. **不训练、不评估、不写 active_version** —— 全部委托给
   :class:`src.evolution.auto_retrain_pipeline.AutoRetrainPipeline`（决策）与
   :class:`src.evolution.runner.EvolutionRunner`（执行 + 闸门）。
2. **取不到指标就什么都不做**（fail-safe）：宁可这周不重训，
   也不因为一条坏数据触发一次重训。
3. **默认 `--check`**：只判定并落盘报告，不跑重训（重训很贵、很慢）。

用法
====
    python scripts/drift_retrain.py                 # 只判定（默认）
    python scripts/drift_retrain.py --run           # 漂移则跑重训（产出待批候选）
    python scripts/drift_retrain.py --force-run      # 无视漂移，强制跑一轮重训
    python scripts/drift_retrain.py --json          # 输出机器可读结果
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.evolution.auto_retrain_pipeline import AutoRetrainPipeline  # noqa: E402

DATA = ROOT / "data"
REPORT_FILE = DATA / "evolution" / "drift_retrain_report.json"


# ----------------------------------------------------------------------------
# 指标采集（全部 fail-safe：取不到就返回空 dict，让判定层判「无法判定」）
# ----------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"读取 {path.name} 失败：{e}")
    return None


def current_metrics(window: int = 20) -> Dict[str, float]:
    """现任模型的近期 RankIC：取 ``rolling_ic_log.json`` 最近 ``window`` 个交易日的均值。

    为什么用均值而不是最后一天：单日 IC 的信噪比极低（日频 IC 标准差常在 0.15 上下），
    拿一天的数字判漂移等于抛硬币。同时也把**标准差**带出来 —— 判定阈值要用它算
    标准误（见 :func:`decide` 里的 2σ 规则），否则「均值抖了 0.02」就会被当成漂移。
    """
    log = _load_json(DATA / "rolling_ic_log.json")
    rows = log if isinstance(log, list) else (log or {}).get("records") or []
    vals = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        v = r.get("rank_ic", r.get("ic", r.get("value")))
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:                       # 非 NaN
            vals.append(f)
    if not vals:
        return {}
    tail = vals[-max(int(window), 1):]
    n = len(tail)
    mean = sum(tail) / n
    if n >= 2:
        var = sum((x - mean) ** 2 for x in tail) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0
    return {"rank_ic": mean, "ic_std": std, "n_obs": float(n)}


def baseline_metrics() -> Dict[str, float]:
    """基准 RankIC：优先进化状态里记录的「上线时」水平，其次模型注册表。"""
    st = _load_json(DATA / "evolution_state.json") or {}
    for key in ("baseline_rank_ic", "best_rank_ic", "model_rank_ic"):
        v = st.get(key) if isinstance(st, dict) else None
        try:
            if v is not None:
                return {"rank_ic": float(v)}
        except (TypeError, ValueError):
            pass
    # 退一步：模型注册表里 rank_ic 最高的那个（= 当前基准线）
    try:
        from src.models import model_registry as mr
        best = mr.best_overall()
        v = float(getattr(best, "rank_ic", 0) or 0) if best is not None else 0.0
        if v > 0:
            return {"rank_ic": v}
    except Exception as e:  # noqa: BLE001
        logger.debug(f"模型注册表不可用（基准取不到）：{e}")
    return {}


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

MIN_OBS = 10          # 少于这么多个交易日不做判定（几天均值 = 纯噪声）
SIGMA = 2.0           # 判定用的显著性倍数
DEFAULT_DELTA = 0.010 # 兜底最小跌幅（与 AutoRetrainPipeline 默认值一致）


def decide(window: int = 20) -> Dict[str, Any]:
    """只判定，不训练。返回可落盘的报告字典。

    ⚠️ 阈值用 **2σ 规则**动态算，而不是写死 0.010：
    20 日 IC 均值的标准误 ``se = std/√n`` 常在 0.03 上下，写死 0.01 的阈值
    意味着「均值抖了 0.02 就重训」—— 那是用噪声改模型，正好是进化模块
    自己一直在警告的事。所以有效阈值 = ``max(min_delta, 2·se)``。
    """
    cur = current_metrics(window)
    base = baseline_metrics()

    se = 0.0
    drift_drop = DEFAULT_DELTA
    if cur:
        n = int(cur.get("n_obs", 0) or 0)
        se = (float(cur.get("ic_std", 0.0) or 0.0) / (n ** 0.5)) if n > 0 else 0.0
        drift_drop = max(DEFAULT_DELTA, SIGMA * se)

    pipe = AutoRetrainPipeline(drift_drop=drift_drop)
    reason = None
    note = ""
    if not cur:
        note = "取不到现任 RankIC（rolling_ic 未回填？）"
    elif int(cur.get("n_obs", 0)) < MIN_OBS:
        note = (f"样本仅 {int(cur.get('n_obs', 0))} 个交易日 < {MIN_OBS}，"
                f"不做漂移判定（几天的均值是噪声）")
    else:
        reason = pipe.drift_reason(cur, base)

    return {
        "at": datetime.now().isoformat(timespec="seconds"),
        "window": int(window),
        "current": cur,
        "baseline": base,
        "se": se,
        "drift_threshold": drift_drop,
        "can_judge": bool(cur) and int(cur.get("n_obs", 0)) >= MIN_OBS,
        "drifted": bool(reason),
        "reason": reason or note,
        "action_hint": ("跑 `python scripts/evolve.py --run --generators retrain --verify`"
                        if reason else "无需动作"),
    }


def run_retrain(force: bool = False, window: int = 20) -> Dict[str, Any]:
    """漂移（或 --force-run）时跑一轮**带闸门**的重训，产出待批候选。"""
    rep = decide(window)
    if not rep["drifted"] and not force:
        rep["ran"] = False
        rep["note"] = "未检测到（显著）漂移，不重训（用 --force-run 可强制）"
        return rep
    if not rep["can_judge"]:
        # 连现任 IC 都取不到、或样本太短 → 无法判定，也不该贸然重训
        rep["ran"] = False
        rep["note"] = f"无法判定（{rep['reason']}），按 fail-safe 不重训"
        return rep

    try:
        from src.evolution.runner import EvolutionRunner
        runner = EvolutionRunner()
        report = runner.run(generators=["retrain"], verify=True, auto_apply=False)
        pending = list(getattr(report, "pending", []) or [])
        rep["ran"] = True
        rep["candidate_ids"] = pending
        rep["n_candidates"] = int(getattr(report, "n_candidates", 0) or 0)
        rep["rejected"] = list(getattr(report, "rejected", []) or [])
        rep["next_step"] = (
            "候选待人工批准：" + ", ".join(f"`python scripts/evolve.py --approve {c} "
                                            f"--apply {c}`" for c in pending)
            if pending else "本回合未产出待批候选（闸门判负或未达门槛）"
        )
        rep["runner_summary"] = report.render() if hasattr(report, "render") else str(report)
    except Exception as e:  # noqa: BLE001
        logger.exception("重训回合失败")
        rep["ran"] = False
        rep["error"] = str(e)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="T4 漂移触发重训（决策 + 待批候选）")
    ap.add_argument("--run", action="store_true", help="检测到漂移时跑重训")
    ap.add_argument("--force-run", action="store_true", help="无视漂移强制重训")
    ap.add_argument("--window", type=int, default=20, help="现任 RankIC 均值窗口（交易日）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    rep = run_retrain(force=args.force_run, window=args.window) if (
        args.run or args.force_run) else decide(args.window)

    try:
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"报告落盘失败（不影响判定）：{e}")

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    print("=" * 68)
    print("T4 漂移判定")
    print("=" * 68)
    print(f"现任 RankIC（近 {rep['window']} 日均值）: "
          f"{rep['current'].get('rank_ic', float('nan')):+.4f}"
          if rep["current"] else "现任 RankIC: 取不到（rolling_ic 未回填？）")
    print(f"基准 RankIC: {rep['baseline'].get('rank_ic', '未知')}")
    print(f"均值标准误 {rep.get('se', 0):.4f} → 有效阈值 "
          f"{rep.get('drift_threshold', 0):.4f}（max(0.010, 2σ)）")
    print(f"判定: {'⚠️ 已漂移 — ' + rep['reason'] if rep['drifted'] else ('✅ 未漂移' if rep.get('can_judge') else '⏸ 无法判定 — ' + rep['reason'])}")
    if rep.get("ran") is not None:
        print(f"重训: {'已执行' if rep['ran'] else '未执行'} —— {rep.get('note', '')}")
    for k in ("candidate_ids", "next_step", "error"):
        if rep.get(k):
            print(f"{k}: {rep[k]}")
    print(f"\n报告: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

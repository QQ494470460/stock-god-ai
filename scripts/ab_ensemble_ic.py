# -*- coding: utf-8 -*-
"""集成融合 A/B 对比：同日期窗口、同参数，只切换 ml.use_ensemble（及 ensemble_shrinkage），
量化「多模型平均 + 权重收缩」对每日 RankIC 稳定性（Grinold-Kahn IR = mean/std）的影响。

输出：
  data/ab_ens_off.json      单模型（use_ensemble=False）回填日志
  data/ab_ens_on.json       集成（use_ensemble=True）回填日志
  data/ab_ens_curve_off.{svg,html}
  data/ab_ens_curve_on.{svg,html}
  data/ab_ens_report.json   对比结论

用法：
  python scripts/ab_ensemble_ic.py [--days 20] [--sample 90] [--pool zz500] [--horizon 5] [--shrinkage 0.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _ic_stats(ics: list) -> dict:
    if not ics:
        return {}
    arr = np.asarray(ics, dtype=float)
    m = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    se = sd / (len(arr) ** 0.5) if len(arr) > 1 and sd == sd else float("nan")
    ir = m / sd if sd and sd > 0 else float("nan")
    t = m / se if se and se > 0 else float("nan")
    return {
        "n": int(len(arr)),
        "mean_ic": round(m, 4),
        "std_daily": round(sd, 4),
        "ir_mean_std": round(ir, 3),
        "t_stat": round(t, 2),
        "pos_rate": round(float(np.mean(arr > 0)), 3),
    }


def _run(variant: str, use_ensemble: bool, shrinkage: float,
         days: int, sample: int, pool: str, horizon: int,
         drop_leg: str = "") -> Path:
    from config.settings import config as app_config
    from src.core.rolling_ic import run_backfill
    from src.models.predictor import predictor

    app_config.ml.use_ensemble = bool(use_ensemble)
    app_config.ml.ensemble_shrinkage = float(shrinkage)
    # 剔腿实验：用于验证收益不是来自某条腿的「前视」泄漏。
    # 典型用法 --drop-leg alpha_xgboost —— alpha 腿用了融资融券/龙虎榜等
    # 「当日收盘后才披露」的特征，回测却按当日收盘价成交，IC 会被系统性抬高。
    if drop_leg:
        ep = predictor._get_ensemble()
        if ep is not None:
            ep.subs = [(s, w) for s, w in ep.subs if s.name != drop_leg]
    out = DATA / f"ab_ens_{variant}.json"
    if out.exists():
        out.unlink()  # 确保全量重算，不被旧文件幂等跳过
    res = run_backfill(days=days, sample=sample, pool=pool, horizon=horizon, out=out)

    # 诊断：本跑实际启用到几条腿
    legs = []
    try:
        ep = getattr(predictor, "_ensemble", None)
        if ep is not None:
            legs = [(s.name, ("✅" if s.available else "⚠️")) for s, _ in ep.subs]
    except Exception:  # noqa: BLE001
        pass
    print(f"[{variant}] use_ensemble={use_ensemble} shrinkage={shrinkage} legs={legs} "
          f"status={res.get('status')} added={res.get('added')} "
          f"mean_ic={res.get('mean_ic')} ir={res.get('ir_mean_std')} "
          f"t={res.get('t_stat')} pos={res.get('pos_rate')}")
    return out


def _render(variant: str, log_path: Path, window: int):
    from src.core.rolling_ic import write_curve_files
    out_svg = DATA / f"ab_ens_curve_{variant}.svg"
    out_html = DATA / f"ab_ens_curve_{variant}.html"
    write_curve_files(out_html=out_html, out_svg=out_svg, window=window,
                      log_path=log_path, metrics_path=None)
    print(f"[{variant}] curve -> {out_svg.name} / {out_html.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--sample", type=int, default=90)
    ap.add_argument("--pool", type=str, default="zz500")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--shrinkage", type=float, default=0.5)
    ap.add_argument("--only", type=str, default="both", choices=["off", "on", "both"])
    ap.add_argument("--drop-leg", type=str, default="", choices=["", "alpha_xgboost", "lstm"],
                    help="从集成里剔除某条腿（用于隔离「该腿贡献」或排查前视泄漏）")
    args = ap.parse_args()

    # 剔腿时结果落独立文件，避免与全腿结果混在一起
    tag = "on" if not args.drop_leg else f"on_no{args.drop_leg.split('_')[0]}"

    off_path = on_path = None
    if args.only in ("off", "both"):
        off_path = _run("off", False, 0.0, args.days, args.sample, args.pool, args.horizon)
    else:
        p = DATA / "ab_ens_off.json"
        off_path = p if p.exists() else None
    if args.only in ("on", "both"):
        on_path = _run(tag, True, args.shrinkage, args.days, args.sample, args.pool,
                       args.horizon, drop_leg=args.drop_leg)

    off = json.loads(off_path.read_text(encoding="utf-8")) if off_path else []
    on = json.loads(on_path.read_text(encoding="utf-8")) if on_path else []
    off_d = {r["date"]: r["ic"] for r in off if isinstance(r, dict) and "ic" in r}
    on_d = {r["date"]: r["ic"] for r in on if isinstance(r, dict) and "ic" in r}

    off_stats = _ic_stats(list(off_d.values()))
    on_stats = _ic_stats(list(on_d.values()))

    common = sorted(set(off_d) & set(on_d))
    diff = [on_d[d] - off_d[d] for d in common]
    paired = {
        "n_paired": len(common),
        "mean_ic_delta": round(float(np.mean(diff)), 4) if diff else None,
        "paired_t": round(float(np.mean(diff) / (np.std(diff, ddof=1) / (len(diff) ** 0.5))), 2)
        if len(diff) > 1 and np.std(diff, ddof=1) > 0 else None,
        "ir_delta": round(on_stats.get("ir_mean_std", float("nan"))
                          - off_stats.get("ir_mean_std", float("nan")), 3)
        if off_stats and on_stats else None,
    }

    report = {
        "params": {"days": args.days, "sample": args.sample, "pool": args.pool,
                   "horizon": args.horizon, "shrinkage": args.shrinkage,
                   "drop_leg": args.drop_leg or None},
        "off_single": off_stats,
        "on_ensemble": on_stats,
        "paired": paired,
        "conclusion": None,
    }
    if off_stats and on_stats:
        ir_o, ir_n = off_stats["ir_mean_std"], on_stats["ir_mean_std"]
        if ir_n > ir_o:
            verdict = (f"集成融合提升每日 IC 稳定性：日频 IR {ir_o} → {ir_n} "
                       f"（+{round((ir_n - ir_o), 3)}），均值 IC {off_stats['mean_ic']} → "
                       f"{on_stats['mean_ic']}，正 IC 占比 {off_stats['pos_rate']} → {on_stats['pos_rate']}。")
        else:
            verdict = (f"本窗口内集成融合未提升日频 IR：{ir_o} → {ir_n}。"
                       f"可能子模型误差与主模型相关（非独立），平均化收益有限（n={off_stats['n']}）。")
        report["conclusion"] = verdict
        print("\n=== 结论 ===")
        print(verdict)

    (DATA / "ab_ens_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOFF(单模型): {off_stats}")
    print(f"ON (集成)  : {on_stats}")
    print(f"paired: {paired}")

    if off_path:
        _render("off", off_path, args.window)
    if on_path:
        _render("on", on_path, args.window)


if __name__ == "__main__":
    main()

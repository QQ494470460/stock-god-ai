# -*- coding: utf-8 -*-
"""行业中性化 A/B 对比：同日期窗口、同模型，只切换 cross_sectional_neutralize，
量化「中性化」对每日 RankIC 稳定性（Grinold-Kahn IR = mean/std）的影响。

输出：
  data/ab_no_neutral.json    OFF 基线回填日志
  data/ab_neutral.json       ON  回填日志
  data/ab_curve_no_neutral.{svg,html}
  data/ab_curve_neutral.{svg,html}
  data/ab_neutralize_report.json   对比结论

用法：
  python scripts/ab_neutralize_ic.py [--days 30] [--sample 120] [--pool zz500] [--horizon 5]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
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
    pos = float(np.mean(arr > 0))
    return {
        "n": int(len(arr)),
        "mean_ic": round(m, 4),
        "std_daily": round(sd, 4),
        "ir_mean_std": round(ir, 3),
        "t_stat": round(t, 2),
        "pos_rate": round(pos, 3),
    }


def _run(variant: str, neutralize: bool, days: int, sample: int, pool: str, horizon: int) -> tuple:
    from config.settings import config as app_config
    from src.core.rolling_ic import run_backfill
    from src.models.predictor import predictor

    app_config.ml.cross_sectional_neutralize = neutralize
    out = DATA / f"ab_{variant}.json"
    if out.exists():
        out.unlink()  # 确保全量重算，不被旧文件幂等跳过
    res = run_backfill(days=days, sample=sample, pool=pool, horizon=horizon, out=out)
    kind = getattr(predictor, "_last_neutral_kind", None)
    print(f"[{variant}] status={res.get('status')} added={res.get('added')} "
          f"mean_ic={res.get('mean_ic')} ir={res.get('ir_mean_std')} t={res.get('t_stat')} "
          f"pos={res.get('pos_rate')} grouping={kind}")
    return out, kind


def _render(variant: str, log_path: Path, window: int):
    from src.core.rolling_ic import render_curve, write_curve_files
    svg, _ = render_curve(log_path=log_path, metrics_path=None, window=window)
    out_svg = DATA / f"ab_curve_{variant}.svg"
    out_html = DATA / f"ab_curve_{variant}.html"
    svg_data = write_curve_files(out_html=out_html, out_svg=out_svg, window=window,
                                 log_path=log_path, metrics_path=None)
    print(f"[{variant}] curve -> {out_svg.name} / {out_html.name}")
    return svg_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--pool", type=str, default="zz500")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--only", type=str, default="both", choices=["off", "neutral", "both"])
    args = ap.parse_args()

    # 预热行业映射（ON 跑优先用行业；失败/缺失则回退板级分组）
    try:
        from src.data.fetcher import fetcher
        im = fetcher.get_industry_map() or {}
        print(f"[warmup] industry_map size = {len(im)}")
    except Exception as e:  # noqa: BLE001
        print(f"[warmup] industry_map 获取失败（将回退板级分组）: {e}")

    off_path, off_kind = (None, None)
    on_path, on_kind = (None, None)
    if args.only in ("off", "both"):
        off_path, off_kind = _run("no_neutral", False, args.days, args.sample, args.pool, args.horizon)
    else:
        # 复用既有 OFF 基线（config=False 时恒为 no-op，与行业映射无关）
        p = DATA / "ab_no_neutral.json"
        off_path = p if p.exists() else None
        off_kind = "none(off)"
    if args.only in ("neutral", "both"):
        on_path, on_kind = _run("neutral", True, args.days, args.sample, args.pool, args.horizon)

    off = json.loads(off_path.read_text(encoding="utf-8")) if off_path else []
    on = json.loads(on_path.read_text(encoding="utf-8")) if on_path else []
    off_d = {r["date"]: r["ic"] for r in off if isinstance(r, dict) and "ic" in r}
    on_d = {r["date"]: r["ic"] for r in on if isinstance(r, dict) and "ic" in r}

    off_stats = _ic_stats(list(off_d.values()))
    on_stats = _ic_stats(list(on_d.values()))

    # 配对对比：取两跑都有的日期
    common = sorted(set(off_d) & set(on_d))
    paired_diff = [on_d[d] - off_d[d] for d in common]
    paired = {
        "n_paired": len(common),
        "mean_ic_delta": round(float(np.mean(paired_diff)), 4) if paired_diff else None,
        "paired_t": round(float(np.mean(paired_diff) / (np.std(paired_diff, ddof=1) / (len(paired_diff) ** 0.5))), 2)
        if len(paired_diff) > 1 and np.std(paired_diff, ddof=1) > 0 else None,
        "ir_delta": round(on_stats.get("ir_mean_std", float("nan")) - off_stats.get("ir_mean_std", float("nan")), 3)
        if off_stats and on_stats else None,
    }

    report = {
        "params": {"days": args.days, "sample": args.sample, "pool": args.pool, "horizon": args.horizon},
        "off": off_stats,
        "on": on_stats,
        "off_grouping": off_kind,
        "on_grouping": on_kind,
        "paired": paired,
        "conclusion": None,
    }
    if off_stats and on_stats:
        ir_o = off_stats["ir_mean_std"]; ir_n = on_stats["ir_mean_std"]
        grouping_note = (f"（OFF 分组={off_kind}，ON 分组={on_kind}）")
        if ir_n > ir_o:
            verdict = (f"截面中性化提升每日 IC 稳定性：日频 IR {ir_o} → {ir_n} "
                       f"（+{round((ir_n-ir_o),3)}），均值 IC {off_stats['mean_ic']} → {on_stats['mean_ic']}，"
                       f"正 IC 占比 {off_stats['pos_rate']} → {on_stats['pos_rate']}。{grouping_note}")
        else:
            verdict = (f"本窗口内截面中性化未提升日频 IR：{ir_o} → {ir_n}。"
                       f"可能该池/窗口截面偏差本就不显著，或样本噪声较大（n={off_stats['n']}）。{grouping_note}")
        report["conclusion"] = verdict
        print("\n=== 结论 ===")
        print(verdict)

    (DATA / "ab_neutralize_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写 data/ab_neutralize_report.json")
    print(f"OFF: {off_stats}")
    print(f"ON : {on_stats}")
    print(f"paired: {paired}")

    _render("no_neutral", off_path, args.window)
    _render("neutral", on_path, args.window)


if __name__ == "__main__":
    main()

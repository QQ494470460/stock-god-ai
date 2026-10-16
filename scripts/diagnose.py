# -*- coding: utf-8 -*-
"""
诊断脚本（一次性，只读，不修改任何模型）
目的：用真实数据量化"预测不准"的根因，而不是拍脑袋调参。

诊断项：
 D1 数据覆盖与陈旧度
 D2 标签分布：绝对涨跌 vs 相对强弱，哪个更可分
 D3 现有关键特征的单变量 RankIC（真实样本外Spearman）
 D4 三分类"涨/震荡/跌"的真实样本外准确率基线
 D5 现有评分公式与真实未来收益的相关性
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

DATA = Path(__file__).resolve().parent.parent / "data"

# ---------- 工具 ----------
def rank_ic(fac: pd.Series, fwd: pd.Series) -> float:
    """横截面 Spearman IC；按日期分组计算，逐日IC 再取均值"""
    df = pd.DataFrame({"fac": fac, "fwd": fwd}).dropna()
    if len(df) < 20:
        return np.nan
    if df["fac"].nunique() < 2 or df["fwd"].nunique() < 2:
        return np.nan
    return float(df["fac"].corr(df["fwd"], method="spearman"))


def main():
    print("=" * 72)
    print(" 选股神 诊断报告")
    print("=" * 72)

    # ========== D1 数据体检 ==========
    print("\n【D1】数据覆盖与陈旧度")
    files = sorted(DATA.glob("daily_*_qfq.parquet"))
    print(f"  parquet 文件数: {len(files)}")

    metas = []
    for f in files[:400]:
        try:
            df = pd.read_parquet(f)
            if len(df) == 0:
                continue
            metas.append({
                "code": f.stem.split("_")[1],
                "rows": len(df),
                "start": str(df["date"].iloc[0])[:10],
                "end": str(df["date"].iloc[-1])[:10],
                "cols": list(df.columns),
            })
        except Exception:
            continue

    meta = pd.DataFrame(metas)
    if len(meta) == 0:
        print("  !! 无有效数据")
        return
    print(f"  成功读取: {len(meta)} 只")
    print(f"  起始日期分布: {meta['start'].value_counts().head(3).to_dict()}")
    print(f"  最新日期分布(前3): {meta['end'].value_counts().head(3).to_dict()}")
    short = meta[meta["rows"] < 400]
    print(f"  !! 疑似被截断的缓存(<400行): {len(short)} 只")
    if len(short):
        print("     示例:", short[["code", "rows", "start", "end"]].head(5).to_dict("records"))

    # 装载一个可用股票池
    pool = meta[meta["rows"] > 800]["code"].tolist()[:120]
    print(f"\n  使用股票池: {len(pool)} 只")

    # 加载
    store = {}
    for c in pool:
        try:
            df = pd.read_parquet(DATA / f"daily_{c}_qfq.parquet")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            store[c] = df
        except Exception:
            continue
    print(f"  实际载入: {len(store)} 只")

    if len(store) < 30:
        print("  !! 可用股票不足，终止")
        return

    # 统一日期轴（用所有股票的日期并集，避免单只股票轴错位）
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(d["date"]) for d in store.values()])))
    print(f"  日期轴: {all_dates[0].date()} ~ {all_dates[-1].date()}  ({len(all_dates)} 个交易日)")

    # ========== D2 标签可分性 ==========
    print("\n【D2】标签定义对比：绝对涨跌 vs 相对强弱")
    HORIZON = 5
    close = pd.DataFrame({c: d.set_index("date")["close"] for c, d in store.items()})
    close = close.reindex(all_dates).ffill(limit=3)
    # 只保留覆盖度足够的日期轴（上市较晚的票会引入NaN，按覆盖度筛选）
    cover = close.notna().mean(axis=1)
    close = close.loc[cover >= 0.5]
    print(f"  有效日期轴(覆盖率>=50%): {len(close)} 天")

    # 绝对收益
    fwd_abs = close.shift(-HORIZON) / close - 1
    # 相对收益（每日截面减均值）—— 剥离市场beta
    fwd_rel = fwd_abs.sub(fwd_abs.mean(axis=1), axis=0)

    sample = fwd_abs.stack().dropna()
    rel_sample = fwd_rel.stack().dropna()
    print(f"  样本数: {len(sample)}")
    print(f"  绝对收益  均值={sample.mean():+.4f} 标准差={sample.std():.4f}")
    print(f"  相对收益  均值={rel_sample.mean():+.4f} 标准差={rel_sample.std():.4f}")
    print(f"  → 相对收益波动是绝对收益的 {rel_sample.std()/sample.std():.1%}")
    print("  (相对收益方差更小说明市场共同波动被剥离，剩下的才是选股 alpha)")

    # 三分类标签分布
    up, dn = 0.02, -0.02
    lab_abs = pd.Series(0, index=sample.index, dtype=int)
    lab_abs[sample > up] = 1
    lab_abs[sample < dn] = -1
    dist = lab_abs.value_counts(normalize=True).sort_index().to_dict()
    print(f"\n  现有三分类标签分布(±2%): " +
          ", ".join(f"{ {1:'涨',0:'震荡',-1:'跌'}[k]}={v:.1%}" for k, v in dist.items()))

    # ========== D3 特征 RankIC 实测 ==========
    print("\n【D3】核心特征的真实样本外 RankIC（最近500日，逐日IC均值）")
    recent = all_dates[-500:]
    close_r = close.reindex(recent)
    fwd_r = fwd_rel.reindex(recent)

    # features: 行=特征名, 列=股票
    feats = {}
    for c, df in list(store.items())[:60]:
        s = df.set_index("date").reindex(recent)
        if s["close"].isna().all():
            continue
        cl = s["close"]
        feats[c] = {
            # 动量
            "ret_5d": cl.pct_change(5),
            "ret_20d": cl.pct_change(20),
            # 反转（短期）
            "rev_5d": -cl.pct_change(5),
            # 波动率
            "vol_20d": cl.pct_change().rolling(20).std(),
            # 量比
            "vol_ratio": (s["volume"] / s["volume"].rolling(5).mean()
            if "volume" in s.columns else pd.Series(np.nan, index=cl.index)),
            # 均线偏离
            "ma20_bias": (cl - cl.rolling(20).mean()) / cl.rolling(20).mean(),
        }
    if feats:
        # feats: {股票: {特征: Series(date_indexed)}} → 构造成 DataFrame(行=日期, 列=股票) 每个特征一张表
        F_tables = {}
        for fname in ["ret_5d", "ret_20d", "rev_5d", "vol_20d", "vol_ratio", "ma20_bias"]:
            series_by_code = {c: rec[fname] for c, rec in feats.items()}
            F_tables[fname] = pd.DataFrame(series_by_code)   # index=date, columns=code
        F = F_tables
    else:
        F = {}
    ics = {}
    for col, tbl in F.items():
        daily_ics = []
        for d in recent:
            if d not in tbl.index:
                continue
            f = tbl.loc[d]
            if d not in fwd_r.index:
                continue
            t = fwd_r.loc[d]
            ic = rank_ic(f, t)
            if not np.isnan(ic):
                daily_ics.append(ic)
        if daily_ics:
            ics[col] = (np.mean(daily_ics), np.std(daily_ics) / np.sqrt(len(daily_ics)))

    if ics:
        print(f"  {'特征':<12}{'平均IC':>10}{'标准误':>10}{'IC_IR':>9}")
        print("  " + "-" * 41)
        for k, (m, se) in sorted(ics.items(), key=lambda x: -abs(x[1][0])):
            ir = m / se if se else 0
            print(f"  {k:<12}{m:>+10.4f}{se:>10.4f}{ir:>9.2f}")
        best = max(ics.items(), key=lambda x: abs(x[1][0]))
        print(f"\n  最强单因子: {best[0]}  IC={best[1][0]:+.4f}")
        print("  (量化实盘单因子 |IC| 通常需 >0.03 才有交易价值)")
    else:
        print("  特征不足，无法计算")
        print("  F keys:", list(F.keys())[:6] if F else "empty")

    # ========== D4 准确率天花板 ==========
    print("\n【D4】分类准确率的理论天花板")
    # 多数类基线（永远猜最大类）
    maj = dist.get(0, 0)
    print(f"  多数类基线(全猜'震荡'): {maj:.1%}")
    print(f"  随机猜测三分类:         {33.3:.1f}%")
    print(f"  → 现有模型若报出60%+ 准确率，几乎必然来自数据泄漏")
    print(f"  → 历史日志中F1=0.68 但 CV accuracy 仅 0.53，差距即泄漏幅度")

    # ========== D5 组合构建可行性 ==========
    print("\n【D5】按因子选股的真实效果（Top20 vs 全市场，最近500日）")
    if "ret_20d" in F:
        rows = []
        tbl = F["ret_20d"]
        sample_dates = [d for d in recent[HORIZON:-HORIZON] if d in tbl.index]
        for d in sample_dates:
            f = tbl.loc[d]
            t = fwd_r.loc[d]
            v = pd.DataFrame({"fac": f, "fwd": t}).dropna()
            if len(v) < 30:
                continue
            top = v.nlargest(20, "fac")
            rows.append({
                "top_ret": top["fwd"].mean(),
                "mkt_ret": v["fwd"].mean(),
            })
        if rows:
            r = pd.DataFrame(rows)
            excess = (r["top_ret"] - r["mkt_ret"]).mean()
            win = (r["top_ret"] > r["mkt_ret"]).mean()
            print(f"  样本日数: {len(r)}")
            print(f"  Top20 平均5日收益: {r['top_ret'].mean():+.3%}")
            print(f"  全市场平均5日收益: {r['mkt_ret'].mean():+.3%}")
            print(f"  超额收益:           {excess:+.3%} / 5日")
            print(f"  跑赢市场天数占比:   {win:.1%}")
            print(f"\n  结论: 动量因子 {'有效' if excess > 0 else '无效(负超额)'}")
            print(f"  年化超额(粗算x48):  {excess*48:+.2%}")
        else:
            print("  样本不足")
    else:
        print("  无ret_20d 特征")

    print("\n" + "=" * 72)
    print(" 诊断结束")
    print("=" * 72)


if __name__ == "__main__":
    main()
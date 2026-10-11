# -*- coding: utf-8 -*-
"""
核心 alpha 特征（akshare 免费接口）
=================================

用 akshare 免费接口补一批「核心 alpha 特征」：
  - 北向资金净额      northbound  （市场级 regime 因子，同日对所有股票相同）
  - 融资融券余额      margin      （逐股融资余额时间序列）
  - 龙虎榜            lhb         （逐股上榜强度 / 上榜买入额）
  - 行业估值动量      industry    （行业动量映射回个股，可选，做不出返回空）

零泄漏原则（见各函数内注释 ★）
--------------------------------
  - 所有 alpha 特征只用「发布 / 统计日期 <= 预测日」的数据
  - 按日期对齐，未来信息绝不进入训练
  - 逐日（或逐行业）取值后只做「前向填充」（ffill，绝不超过该 date），
    因此某一 (date, code) 上看到的值，一定是 date 当天或之前最近一次公布的

akshare 采用**懒加载**（函数内 import），避免无网络时 import 失败拖垮整个模块。

缓存
----
  - 所有拉取缓存到 DATA_DIR/alpha_{name}.parquet（index=date）
  - 采用「读旧 -> concat -> drop_duplicates(subset=['date']) -> sort -> 写回」的 upsert
  - refresh=False 默认走缓存；融资余额逐日即时落盘，断点可续
  - 每个接口独立 try/except，失败返回空 DataFrame 并 logger.warning，
    绝不让单个接口失败阻断其它接口
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# 路径 & 配置
# ---------------------------------------------------------------------------
# ★ akshare 不在此处 import（懒加载）。仅导入轻量的本地配置。
# 本文件位于 src/data/ 下（两层），故需 parent.parent.parent 才能到达项目根。
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR  # noqa: E402

# alpha 列前缀 & 可选子集
PREFIX = "alpha_"
ALPHA_SET_ALL = ("northbound", "margin", "lhb", "industry")


# ---------------------------------------------------------------------------
# 缓存工具
# ---------------------------------------------------------------------------
def _cache_path(name: str) -> Path:
    return DATA_DIR / f"alpha_{name}.parquet"


def _load_cache(name: str) -> Optional[pd.DataFrame]:
    """读取该 name 的缓存面板（index=date）。失败返回 None。"""
    p = _cache_path(name)
    if p.exists():
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[alpha] 读取缓存 {name} 失败: {e}")
    return None


def _save_cache(name: str, df: pd.DataFrame) -> None:
    """upsert：读旧 -> concat -> 按日期去重(保留最新) -> 排序 -> 写回。

    ★ 只按 index(date) 去重，避免重复拉取日覆盖已有数据。
    """
    if df is None or len(df) == 0:
        return
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    old = _load_cache(name)
    if old is not None and len(old):
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    try:
        df.to_parquet(_cache_path(name))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[alpha] 写缓存 {name} 失败: {e}")


def _trade_dates(start: str, end: str) -> List[pd.Timestamp]:
    """交易日历（来自新浪，作为融资余额逐日拉取的循环依据）。"""
    try:
        import akshare as ak

        t = ak.tool_trade_date_hist_sina()
        ds = pd.to_datetime(t.iloc[:, 0])
        ds = ds[(ds >= pd.Timestamp(start)) & (ds <= pd.Timestamp(end))]
        return list(ds)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[alpha] 交易日历获取失败: {e}")
        return []


def _empty(cols) -> pd.DataFrame:
    return pd.DataFrame(columns=list(cols) if cols is not None else [])


# ---------------------------------------------------------------------------
# 1) 北向资金净额（市场级 regime 因子）
# ---------------------------------------------------------------------------
def northbound_panel(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """市场级北向资金净买入（历史），返回 index=date, columns=['northbound_net']。

    ★ 泄漏防护：只保留 start<=date<=end 的行。北向净额是当日收盘后公布，
      对齐到当日预测时视为「已知」，不会引入未来信息。

    实现说明：本环境 akshare 版本对应的是 ``stock_hsgt_hist_em(symbol='北向资金')``
    （原文档 stock_hsgt_north_net_flow_em 在本版本不存在），取「当日成交净买额」列。
    注：北向实时净买额披露在 2024-08 后暂停，之后为 NaN（训练时 fillna(0)）。
    """
    try:
        import akshare as ak
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[alpha] 无法 import akshare（北向跳过）: {e}")
        return _empty(["northbound_net"])

    name = "northbound"
    if not refresh:
        cached = _load_cache(name)
        if cached is not None and len(cached) and "northbound_net" in cached.columns:
            return cached.loc[(cached.index >= pd.Timestamp(start)) &
                             (cached.index <= pd.Timestamp(end))]

    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        s = df.set_index(pd.to_datetime(df["日期"]))["当日成交净买额"]
        out = pd.DataFrame({"northbound_net": s[~s.index.duplicated(keep="last")]}).sort_index()
        _save_cache(name, out)
        return out.loc[(out.index >= pd.Timestamp(start)) &
                       (out.index <= pd.Timestamp(end))]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[alpha] 北向资金拉取失败（跳过）: {e}")
        return _empty(["northbound_net"])


# ---------------------------------------------------------------------------
# 2) 融资融券余额（逐股融资余额时间序列）
# ---------------------------------------------------------------------------
def _margin_one_day(ak, ds: str) -> Dict[str, float]:
    """拉取单个交易日全市场融资余额，返回 {code: 融资余额}。"""
    rec: Dict[str, float] = {}
    specs = [
        ("stock_margin_detail_sse", "标的证券代码", "融资余额"),
        ("stock_margin_detail_szse", "证券代码", "融资余额"),
    ]
    for fn_name, code_col, bal_col in specs:
        try:
            d = getattr(ak, fn_name)(date=ds)
        except Exception:  # noqa: BLE001
            continue
        if d is None or not len(d):
            continue
        bal = bal_col if bal_col in d.columns else next(
            (c for c in d.columns if "融资余额" in c), None)
        codec = code_col if code_col in d.columns else next(
            (c for c in d.columns if ("证券代码" in c or "代码" in c)), None)
        if bal is None or codec is None:
            continue
        for _, row in d.iterrows():
            c = str(row[codec]).strip()
            try:
                v = float(row[bal])
            except Exception:  # noqa: BLE001
                continue
            rec[c] = v
    return rec


def margin_panel(
    start: str,
    end: str,
    codes: Optional[Sequence[str]] = None,
    trade_dates: Optional[Sequence] = None,
    refresh: bool = False,
    max_calls: int = 200,
) -> pd.DataFrame:
    """逐股融资余额时间序列，返回 index=date, columns=code（仅保留 codes 内个股）。

    ★ 泄漏防护：每交易日只取「该交易日收盘后」公布的融资余额（date=统计日），
      对齐到当日预测时视为已知；之后仅做前向填充，亦不超过该 date。

    实现说明：本环境对应 ``stock_margin_detail_sse / stock_margin_detail_szse``
    （原文档 stock_margin_detail_em(date=...) 在本版本不存在，且按沪/深拆分）。

    性能 / 断点：
      - 先 load_pool 拿到候选 code（由调用方通过 codes 传入），只保留这些 code；
      - 循环区间内交易日逐日拉取，并**即时写入缓存**（断点可续）；
      - 若区间太大太慢（交易日数 > max_calls），自动降采样（每 N 个交易日取一次）
        并在日志注明，避免一次性打爆接口。
    """
    try:
        import akshare as ak
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[alpha] 无法 import akshare（融资余额跳过）: {e}")
        return _empty(codes)

    name = "margin"
    if not refresh:
        cached = _load_cache(name)
        if cached is not None and len(cached):
            sub = cached.loc[(cached.index >= pd.Timestamp(start)) &
                             (cached.index <= pd.Timestamp(end))]
            if codes is not None:
                sub = sub[[c for c in codes if c in sub.columns]]
            if len(sub):
                return sub

    # 交易日历
    if trade_dates is None:
        trade_dates = _trade_dates(start, end)
    else:
        trade_dates = [pd.Timestamp(t) for t in trade_dates]

    if not trade_dates:
        logger.warning("[alpha] 融资余额：无可用交易日历，跳过")
        return _empty(codes)

    # 断点续传：已有缓存的日期集合
    existing = set()
    cached = _load_cache(name)
    if cached is not None and len(cached):
        existing = set(pd.to_datetime(cached.index))

    # 降采样步长（区间太大则每 step 个交易日取一次，约等于每 5 个交易日取一次）
    step = 1
    if len(trade_dates) > max_calls:
        step = int(np.ceil(len(trade_dates) / max_calls))
        logger.warning(
            f"[alpha] 融资余额区间交易日 {len(trade_dates)} 个 > {max_calls} 上限，"
            f"降采样为每 {step} 个交易日取一次（与文档示例一致；中间日将缺失，已注明）"
        )

    fetched = 0
    for d in trade_dates[::step]:
        if d in existing:
            continue
        ds = d.strftime("%Y%m%d")
        try:
            part = _margin_one_day(ak, ds)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[alpha] 融资余额 {ds} 失败: {e}")
            part = None
        if part:
            # 即时落盘：单日 (date, code) 面板，断点可续
            _save_cache(
                name,
                pd.DataFrame(
                    {c: [v] for c, v in part.items()}, index=[d]
                ).rename_axis("date"),
            )
            fetched += 1
        time.sleep(0.05)  # 礼貌限速

    if fetched:
        logger.info(f"[alpha] 融资余额新拉取 {fetched} 个交易日")

    cached = _load_cache(name)
    if cached is None or not len(cached):
        return _empty(codes)
    out = cached.loc[(cached.index >= pd.Timestamp(start)) &
                     (cached.index <= pd.Timestamp(end))]
    if codes is not None:
        out = out[[c for c in codes if c in out.columns]]
    return out.sort_index()


# ---------------------------------------------------------------------------
# 3) 龙虎榜（逐股上榜强度 / 上榜买入额）
# ---------------------------------------------------------------------------
def lhb_panel(
    start: str,
    end: str,
    codes: Optional[Sequence[str]] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """逐股龙虎榜「上榜买入额」面板，返回 index=date, columns=code。

    ★ 泄漏防护：上榜日为公布日，对齐到当日预测时视为已知（非未来）。
      未上榜的 (date, code) 值为 0（caller 端 ffill+fillna(0) 后再用 >0 派生「是否上榜」）。

    实现说明：``stock_lhb_detail_em(start_date, end_date)`` 返回逐股龙虎榜明细，
    取「龙虎榜买入额」按 (上榜日, 代码) 聚合。
    """
    try:
        import akshare as ak
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[alpha] 无法 import akshare（龙虎榜跳过）: {e}")
        return _empty(codes)

    name = "lhb"
    if not refresh:
        cached = _load_cache(name)
        if cached is not None and len(cached):
            sub = cached.loc[(cached.index >= pd.Timestamp(start)) &
                             (cached.index <= pd.Timestamp(end))]
            if codes is not None:
                sub = sub[[c for c in codes if c in sub.columns]]
            if len(sub):
                return sub

    s = start.replace("-", "")
    e = end.replace("-", "")
    try:
        df = ak.stock_lhb_detail_em(start_date=s, end_date=e)
        df["上榜日"] = pd.to_datetime(df["上榜日"])
        df["代码"] = df["代码"].astype(str).str.strip()
        buy = (df.groupby(["上榜日", "代码"])["龙虎榜买入额"]
               .sum().unstack(fill_value=0.0))
        out = buy.sort_index()
        _save_cache(name, out)
        out = out.loc[(out.index >= pd.Timestamp(start)) &
                      (out.index <= pd.Timestamp(end))]
        if codes is not None:
            out = out[[c for c in codes if c in out.columns]]
        return out.sort_index()
    except Exception as ex:  # noqa: BLE001
        logger.warning(f"[alpha] 龙虎榜拉取失败（跳过）: {ex}")
        return _empty(codes)


# ---------------------------------------------------------------------------
# 4) 行业估值动量（可选，做不出返回空并说明）
# ---------------------------------------------------------------------------
def industry_panel(
    start: str,
    end: str,
    codes: Optional[Sequence[str]] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """行业动量映射回个股，返回 index=date, columns=code（值为行业当日收益率=动量）。

    ★ 泄漏防护：行业动量 = 行业指数当日收益率（由历史收盘计算，仅依赖过去，
      无泄漏）；缺失 (date, code) 前向填充后仍只含 <=date 的已知动量。

    实现说明（本环境 eastmoney 行业接口可能受限，失败即返回空并说明）：
      - stock_board_industry_name_em 取行业列表
      - stock_board_industry_cons_em 取行业成分，建立 code->行业 映射
        （只为包含我们 codes 的行业拉成分，提前 break 省调用）
      - stock_board_industry_hist_em 取行业日线，算每日收益率作为行业动量
    """
    try:
        import akshare as ak
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[alpha] 无法 import akshare（行业动量跳过）: {e}")
        return _empty(codes)

    name = "industry"
    if not refresh:
        cached = _load_cache(name)
        if cached is not None and len(cached):
            sub = cached.loc[(cached.index >= pd.Timestamp(start)) &
                             (cached.index <= pd.Timestamp(end))]
            if codes is not None:
                sub = sub[[c for c in codes if c in sub.columns]]
            if len(sub):
                return sub

    s, e = start.replace("-", ""), end.replace("-", "")
    try:
        ind = ak.stock_board_industry_name_em()
    except Exception as ex:  # noqa: BLE001
        logger.warning(f"[alpha] 行业列表获取失败（industry 跳过）: {ex}")
        return _empty(codes)

    name_col = "板块名称" if "板块名称" in ind.columns else ind.columns[0]
    names = [str(x) for x in ind[name_col].dropna().unique()]

    # code -> 行业 映射（只为含我们 codes 的行业拉成分）
    code2ind: Dict[str, str] = {}
    for nm in names:
        try:
            cons = ak.stock_board_industry_cons_em(symbol=nm)
        except Exception:  # noqa: BLE001
            continue
        if cons is None or not len(cons):
            continue
        ccol = next((c for c in cons.columns if "代码" in c), None)
        if ccol is None:
            continue
        for c in cons[ccol].astype(str).str.strip():
            if codes is None or c in set(codes):
                code2ind[c] = nm
        if codes is not None and set(code2ind.keys()) >= set(codes):
            break  # 已覆盖全部目标 code，提前结束

    if not code2ind:
        logger.warning("[alpha] 未建立 code->行业 映射（industry 跳过）")
        return _empty(codes)

    # 拉取相关行业的日线，算行业动量（降采样省调用）
    ind_ret: Dict[str, pd.Series] = {}
    for nm in set(code2ind.values()):
        try:
            h = ak.stock_board_industry_hist_em(
                symbol=nm, period="day", start_date=s, end_date=e, adjust="")
        except Exception as ex:  # noqa: BLE001
            logger.warning(f"[alpha] 行业 {nm} 日线失败: {ex}")
            continue
        if h is None or not len(h):
            continue
        dcol = "日期" if "日期" in h.columns else h.columns[0]
        ccol = "收盘" if "收盘" in h.columns else next(
            (c for c in h.columns if "收盘" in c), None)
        if ccol is None:
            continue
        h[dcol] = pd.to_datetime(h[dcol])
        h = h.set_index(dcol)[ccol].astype(float)
        h = h[~h.index.duplicated(keep="last")].sort_index()
        # ★ 动量 = 当日收益率（已知，无泄漏）
        ind_ret[nm] = h.pct_change()

    if not ind_ret:
        logger.warning("[alpha] 行业动量计算失败（industry 跳过）")
        return _empty(codes)

    # 映射回 code：每个 code 取其行业当日动量
    rows: Dict[str, pd.Series] = {}
    for c, nm in code2ind.items():
        if nm in ind_ret:
            rows[c] = ind_ret[nm]
    if not rows:
        return _empty(codes)

    panel = pd.DataFrame(rows).sort_index()
    _save_cache(name, panel)
    out = panel.loc[(panel.index >= pd.Timestamp(start)) &
                    (panel.index <= pd.Timestamp(end))]
    if codes is not None:
        out = out[[c for c in codes if c in out.columns]]
    return out.sort_index()


# ---------------------------------------------------------------------------
# 聚合：把各 panel 拼成 index=(date, code) 的 alpha 特征表
# ---------------------------------------------------------------------------
def _cs_rank_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """横截面 rank -> (-1, 1)，按行（每日期）独立。

    与项目 src/models/labeling.cs_rank 口径一致；每日期内所有值均为
    <=date 的已知值，因此无泄漏。
    """
    return panel.rank(pct=True, axis=1).mul(2).sub(1)


def build_alpha_features(
    codes: Sequence[str],
    date_index: Sequence,
    start: str,
    end: str,
    alpha_set: Sequence[str] = ALPHA_SET_ALL,
) -> pd.DataFrame:
    """返回 index=(date, code) 的 alpha 特征表。

    对每个 (date, code)：取该 date 及之前最近可用的
      - 融资余额（前向填充，但绝不超过 date）
      - 当日北向净买入（市场级，同日所有股票相同）
      - 当日龙虎榜强度（上榜买入额，未上榜=0）
      - 当日行业动量
    缺失 -> NaN（下游训练时 fillna(0)）。

    ★ 泄漏防护：所有取值只来自 date<=预测日 的历史缓存。
      前向填充（ffill）只向后 carry 最近已知值，永不引入未来信息。
    """
    di = pd.DatetimeIndex(pd.to_datetime(sorted(set(pd.to_datetime(list(date_index))))))
    codes = list(codes)
    frames: Dict[str, pd.DataFrame] = {}

    if "northbound" in alpha_set:
        nb = northbound_panel(start, end)
        if len(nb):
            s = nb["northbound_net"]
            # ★ 前向填充但绝不超过 date
            aligned = s.reindex(di).ffill()
            frames["alpha_northbound"] = pd.DataFrame({c: aligned for c in codes})

    if "margin" in alpha_set:
        mg = margin_panel(start, end, codes=codes, trade_dates=di)
        if len(mg):
            ff = mg.reindex(di).ffill()
            frames["alpha_margin"] = _cs_rank_panel(ff)

    if "lhb" in alpha_set:
        lh = lhb_panel(start, end, codes=codes)
        if len(lh):
            ff = lh.reindex(di).ffill().fillna(0.0)
            frames["alpha_lhb"] = _cs_rank_panel(ff)

    if "industry" in alpha_set:
        ind = industry_panel(start, end, codes=codes)
        if len(ind):
            ff = ind.reindex(di).ffill().fillna(0.0)
            frames["alpha_industry_mom"] = _cs_rank_panel(ff)

    if not frames:
        idx = pd.MultiIndex.from_product([di, codes], names=["date", "code"])
        return pd.DataFrame(index=idx)

    parts = [panel.stack().rename(name) for name, panel in frames.items()]
    alpha = pd.concat(parts, axis=1)
    alpha.index = alpha.index.set_names(["date", "code"])
    # 仅保留真正出现过的 alpha 列
    alpha = alpha[[c for c in alpha.columns if not alpha[c].isna().all()]]
    return alpha.sort_index()


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # 限制到最近一段时间，避免长拉取
    end_d = pd.Timestamp.today().normalize()
    start_d = end_d - pd.Timedelta(days=40)
    s_str, e_str = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")

    from scripts.train_v2 import load_pool

    # 显式 legacy：仅做接口自检，股票池无关紧要，固定旧口径保证可复现
    store = load_pool(10, pool="legacy")
    codes = list(store.keys())
    print(f"[self-check] codes({len(codes)}): {codes[:10]}  range {s_str}~{e_str}")

    for fn, kw in [
        ("northbound", dict(start=s_str, end=e_str)),
        ("margin", dict(start=s_str, end=e_str, codes=codes,
                        trade_dates=list(pd.date_range(s_str, e_str, freq="B")[-15:]))),
        ("lhb", dict(start=s_str, end=e_str, codes=codes)),
        ("industry", dict(start=s_str, end=e_str, codes=codes)),
    ]:
        try:
            df = (northbound_panel if fn == "northbound" else
                  margin_panel if fn == "margin" else
                  lhb_panel if fn == "lhb" else industry_panel)(**kw)
            miss = "（接口失败/无数据->空）" if df.shape[1] == 0 else ""
            print(f"  {fn:10s} shape={df.shape} cols={list(df.columns)[:5]}{miss}")
        except Exception as ex:  # noqa: BLE001
            print(f"  {fn:10s} EXCEPTION: {ex!r}")

    try:
        af = build_alpha_features(
            codes, pd.date_range(s_str, e_str, freq="B")[-15:],
            s_str, e_str, set(ALPHA_SET_ALL))
        print(f"[self-check] alpha_features shape={af.shape} cols={list(af.columns)}")
    except Exception as ex:  # noqa: BLE001
        print(f"[self-check] build_alpha_features EXCEPTION: {ex!r}")

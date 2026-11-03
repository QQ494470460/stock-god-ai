#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""耐心地把「代码 → 所属行业」补齐并落盘（供截面中性化使用）。

为什么要单独一个脚本
====================
``ml.cross_sectional_neutralize=True`` 开启后，``predict_batch`` 会按**行业**对
评分做截面中性化（去行业水位）。行业映射来自 ``data/industry_map.json``：
它是空的时，中性化只能退化成「板级分组」（按代码前缀，粗粒度，仅去板块水位）。

2026-10-19 实测：本机到东财 ``17.push2.eastmoney.com`` 是**间歇性**可达 ——
某两分钟里能拿到 200 OK（496 个行业板块），之后连续 ProxyError /
ConnectionReset（走代理和直连都一样）。原来的取数路线要「列板块 + 逐板块拉
成分股」上百次请求，几乎不可能全部撞在可用窗口内。

``fetcher._fetch_industry_map_bulk()`` 改成翻几页行情列表接口（自带 f100
所属行业字段），一次成功就能覆盖全市场。本脚本负责**反复试**，撞到一个可用
窗口就把成果永久落盘（之后 ``get_industry_map`` 直接命中缓存，不再打网络）。

用法::

    python scripts/build_industry_map.py                 # 默认试 30 次、每次隔 40s
    python scripts/build_industry_map.py --attempts 60 --interval 20
    python scripts/build_industry_map.py --once          # 只试一次（可挂定时任务）

退出码：0 = 已补齐（或本来就有）；1 = 窗口期结束仍未拿到。
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 无头启动时 stdout 可能是 GBK / 甚至 None（pythonw），必须先兜住编码
for _name in ("stdout", "stderr"):
    _s = getattr(sys, _name, None)
    if _s is None:
        continue
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def merge_industry_maps(existing: dict, fresh: dict) -> dict:
    """把新抓到的映射**合并**进已有表，绝不用「半张表换半张表」。

    2026-11-04 发现的两个坑：

    1) 原实现是 ``cache_file.write_text(json.dumps(got))`` —— 整表覆盖。
       只要某次抓到的 2500 只与现有的 2621 只不重叠，就会**丢掉已经攒下的
       映射**，而且看起来还「成功」了。东财快照是按页返回的，这种部分重叠
       完全可能发生，所以必须合并而不是替换。
    2) 空值不许覆盖已有值：``{"600000": ""}`` 打进来会把本来知道的行业抹掉。
       只接受非空字符串。
    """
    out = dict(existing or {})
    for code, name in (fresh or {}).items():
        key = str(code)
        val = str(name or "").strip()
        if not val:
            continue
        out[key] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="补齐行业映射（耐心重试版）")
    ap.add_argument("--attempts", type=int, default=30, help="最多试几次（默认 30）")
    ap.add_argument("--interval", type=float, default=40.0, help="每次间隔秒数（默认 40）")
    ap.add_argument("--min", type=int, default=0,
                    help="**落盘**门槛（默认取 fetcher 阈值 2500，防止空结果投毒）")
    ap.add_argument("--target", type=int, default=4800,
                    help="「够齐了」的目标覆盖率（A 股约 5400 只）。低于它继续重试。"
                         "⚠️ 别把它调回 2500：那就是「半个市场就算完工」，"
                         "集中度上限对近一半仓位会一直 fail-open")
    ap.add_argument("--timeout", type=float, default=12.0, help="单次请求超时秒数")
    ap.add_argument("--once", action="store_true", help="只试一次就退出")
    args = ap.parse_args()

    try:
        import config.settings  # noqa: F401  （打包版需要先注入 libs 路径）
    except Exception:  # noqa: BLE001
        pass

    from src.data.fetcher import DataFetcher, fetcher

    floor = args.min or DataFetcher._INDUSTRY_BULK_MIN_CODES
    target = max(int(args.target), floor)
    cache_file = Path(fetcher.cache_dir) / "industry_map.json"

    have: dict = {}
    if cache_file.exists():
        try:
            have = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            have = {}
        if len(have) >= target:
            print(f"[{_ts()}] ✅ 行业映射已覆盖 {len(have)} 只（≥{target}），无需重建")
            return 0
        print(f"[{_ts()}] 现有缓存 {len(have)} 只 / 目标 {target}（覆盖 "
              f"{len(have) / max(target, 1):.1%}），继续补齐…（只增不减）")
    else:
        print(f"[{_ts()}] 尚无行业映射缓存，开始获取（目标 ≥{target} 只）…")

    attempts = 1 if args.once else args.attempts
    for i in range(1, attempts + 1):
        try:
            got = fetcher._fetch_industry_map_bulk(timeout=args.timeout)
        except Exception as e:  # noqa: BLE001
            got = {}
            print(f"[{_ts()}] 第 {i}/{attempts} 次异常: {type(e).__name__}: {str(e)[:70]}")

        merged = merge_industry_maps(have, got)
        if len(merged) > len(have) and len(merged) >= floor:
            # 落盘条件：**累积后**达到门槛。原实现用单次结果判定，
            # 于是「攒了三轮的 2600 只」会被一次 200 只的新抓取判成失败。
            gained = len(merged) - len(have)
            have = merged
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(have, ensure_ascii=False), encoding="utf-8")
            print(f"[{_ts()}] 已写入 {len(have)} 只 / {len(set(have.values()))} 个行业"
                  f"（本轮新增 {gained}）")

        if len(have) >= target:
            from collections import Counter

            top = Counter(have.values()).most_common(5)
            print(f"[{_ts()}] ✅ 第 {i} 次达到目标：{len(have)} 只 → {cache_file}")
            print(f"[{_ts()}]    行业 Top5: {top}")
            return 0

        print(f"[{_ts()}] 第 {i}/{attempts} 次后覆盖 {len(have)} 只（需 ≥{target}），"
              f"{args.interval:.0f}s 后重试…")
        if i < attempts:
            time.sleep(args.interval)

    print(f"[{_ts()}] ❌ {attempts} 次后覆盖仍只有 {len(have)} 只（目标 {target}）—— "
          f"东财当前不可达/限流。已有映射保留不丢；集中度上限对未映射票继续 fail-open，"
          f"`data/industry_map_status.json` 会写明根因")
    return 1


if __name__ == "__main__":
    sys.exit(main())

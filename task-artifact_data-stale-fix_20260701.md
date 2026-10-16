# 选股神AI 数据过期修复 & 自选股监控增强

## 问题
用户反馈个股分析、实时监控使用的都是旧数据（最新数据停留在 2026-05-25），数据已过期超过一个月。

## 根因
`src/data/fetcher.py` 的 `get_daily_kline()` 缓存判断只检查行数（>50条即返回），不检查数据日期是否覆盖到最近。缓存的 `.parquet` 文件一旦存在就能永远吃冷饭。

## 修复
在 `fetcher.py` 第119-131行缓存加载处新增日期过期检查：

```
latest_ts = cached["date"].max()
is_stale = (datetime.now() - latest_ts).days > 2
if len(cached) > 50 and not is_stale:
    return cached  # 2天内可用
else:
    logger.info("缓存过期，重新拉取…")  # → 走AKShare实时下载
```

决策：过期阈值设 **2天**（覆盖周末/节假日），而非1天。周一凌晨也能正常反馈上周五数据。

## 相关文件
- `src/data/fetcher.py` — 核心修复（第119-131行）
- `config/watchlist.json` — 自选股配置（新增）
- `scripts/watchlist_monitor.py` — 自选股AI实时监测脚本（新增）
- `scripts/watchlist_manager.py` — 自选股命令行管理工具（新增）

## 验证
测试10只代表股，全部刷新到 2026-06-30（当前最新日线）。Streamlit UI 中所有页面（个股分析、实时监控、AI选股）共用一个 `fetcher` 实例，修复后自动生效。

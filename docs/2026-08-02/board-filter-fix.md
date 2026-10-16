# AI 选股板块过滤修复 + get_stock_pool 缓存优化

**时间**: 2026-08-02 01:46-01:57
**反馈**: "AI选股不好用。板块过滤不行。"

## 修复内容

### 问题 1：板块过滤后 scan_max 截断顺序反了 🔴
**根因**: app.py L495 `scan_max` 截断在 `filter_by_board` 之前 → 对原始池取前 100 只再过滤 → hs300 前 100 只全是 000xxx，选创业板筛完 0 只
**修复**: 先 filter_by_board 再 scan_max 截断

### 问题 2：scan_max 偏向一个板块 🔴
**根因**: 股票列表按代码前缀排序（000→002→300→600→688…），scan_max=50 取前 50 只 = 全是 000xxx
**修复**: 新增 `fetcher.shuffle_by_prefix()` — 按板块前缀轮询排列（920/688/689/300/301/302/000/001/002/003/600/601/603/605 各取 1 只循环），保证任意 scan_max 都能覆盖所有板块

### 问题 3：get_stock_pool 每次调 AKShare 4 秒+ 🟡
**修复**: 内存缓存 30 分钟 (`_pool_cache + _pool_cache_ttl`)，重复点击秒出

### 问题 4：行业"未知" → 热点板块过滤形同虚设 🟡
**修复**: 新增 `fetcher.get_industry_map()` — 从 AKShare 批量加载申万行业分类，缓存在 `data/industry_map.json`

## 验证结果

| 场景 | 结果 |
|------|------|
| hs300+创业板+scan_max=50 | ✅ 33 只全覆盖（300/301/302 均匀分布） |
| all+科创板+scan_max=100 | ✅ 100 只（688×99 + 689×1） |
| hs300 无过滤 scan_max=50 | ✅ 所有板块均匀覆盖（000×6, 002×5, 300×6, 600×5, 688×6…） |

## 修改文件
- `src/data/fetcher.py`: `__init__` 新增 shuffle_by_prefix/get_industry_map + `get_stock_pool` 内存缓存
- `src/ui/app.py`: 板块过滤顺序修复 + 轮询排列调用

## 状态
- Streamlit: 端口 8501, HTTP 200 ✅

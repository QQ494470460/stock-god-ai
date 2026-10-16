# 市场热点板块修复 — 2026-05-26 05:13

## 问题
Streamlit 看板"市场热点板块"区域空白，什么都不显示。

## 根因
`ak.stock_board_concept_name_em()` 调用的东方财富 API 挂掉了（RemoteDisconnected）。

## 修复
1. 切换到同花顺数据源 `ak.stock_board_industry_summary_ths()`（90个行业，含涨跌幅+资金流向+领涨股）
2. 东方财富接口作为备选（fallback）
3. 修复列名映射：THS API 列名是「板块」而非「名称」
4. UI 增强：每张卡片新增资金净流入和领涨股信息

## 效果
当前热点：半导体 +4.52%、自动化设备 +4.03%、煤炭 +3.27%
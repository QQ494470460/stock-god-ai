# 粤开证券量化自动交易模块 — 实现记录

**日期**: 2026-07-04  
**状态**: ✅ 完成

## 概述

为选股神AI创建了完整的量化自动交易模块，基于粤开证券 miniQMT (xtquant) 接口，支持模拟/实盘双模式。

## 新增文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/trading/__init__.py` | - | 模块入口，导出所有公共API |
| `src/trading/broker.py` | 257 | 券商抽象基类 (BrokerAPI), Order/Position/TickData dataclasses, BrokerCallback |
| `src/trading/account.py` | 212 | 账号加密管理器 (XOR+PBKDF2+base64)，支持多账号配置 |
| `src/trading/yuekai.py` | 420+ | 粤开证券 miniQMT 实现，含模拟模式 fallback |
| `src/trading/executor.py` | 350+ | 交易执行引擎：风控、仓位管理、止盈止损、AI信号过滤 |
| `src/trading/autotrader.py` | 300+ | 自动交易编排器：AI信号→风控→下单→持仓监控 |
| `scripts/autotrade.py` | 380+ | CLI入口：连接/启停/下单/查询/模拟/账号管理 |
| `config/broker.yaml.template` | 48 | 券商配置模板 |
| `src/ui/trading_ui.py` | 260+ | Streamlit UI组件：登录/仪表盘/模拟/AI交易/订单5个Tab |

## 修改文件

- `src/ui/app.py`: 导入 trading_ui, 新增 "💹 自动交易" 导航页, 页面路由

## 架构设计

```
                    ┌──────────────┐
                    │  Streamlit   │  ← app.py + trading_ui.py
                    │   UI面板      │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  AutoTrader  │  ← 编排层：信号→风控→下单→监控
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼───┐ ┌─────▼─────┐ ┌───▼──────────┐
     │ predictor  │ │ Executor  │ │ AccountMgr   │
     │ (AI模型)   │ │ (风控引擎) │ │ (账号加密)    │
     └────────────┘ └─────┬─────┘ └──────────────┘
                          │
                   ┌──────▼──────┐
                   │  BrokerAPI  │  ← 抽象层
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │ YuekaiBroker│  ← 粤开 miniQMT
                   │  (xtquant)  │
                   └─────────────┘
```

## 核心功能

1. **账号管理**: 加密存储券商凭证，多账号支持
2. **连接管理**: miniQMT 连接/断开/断线检测
3. **账户查询**: 总资产、可用资金、持仓、订单
4. **AI信号交易**: 接入现有 predictor，评分≥70 + XGB/LGB一致 → 自动买入
5. **风控系统**:
   - 单票仓位上限 (默认30%)
   - 止损线 (-5%) / 止盈线 (+15%) / 移动止损 (-3%)
   - 单日最大亏损/最大交易笔数
   - 交易时段限制
6. **模拟模式**: xtquant未安装时自动降级为模拟交易
7. **Streamlit UI**: 5个Tab完整交互面板
8. **CLI工具**: `python scripts/autotrade.py <command>` 完整命令行

## 验证

- ✅ 全部导入测试通过 (Python 3.12)
- ✅ 模拟模式正常工作 (xtquant 未安装时自动 fallback)
- ✅ 账号管理器初始化正常
- ⚠️ 实盘交易需: 开通粤开QMT权限 + 安装xtquant + 启动XtMiniQmt.exe

## 使用方式

```bash
# 添加账号
python scripts/autotrade.py add-account --account-id 1234567890 --qmt-path D:\QMT\userdata_mini

# 模拟测试
python scripts/autotrade.py simulate

# 模拟持续运行
python scripts/autotrade.py start-simulate

# 实盘启动 (需QMT)
python scripts/autotrade.py start

# 查询
python scripts/autotrade.py query --type positions
```

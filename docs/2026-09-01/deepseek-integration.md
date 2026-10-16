# DeepSeek AI 接入（2026-09-01）

## 目标
用户要求"让AI选股神接入DeepSeek"，确认方向：**用 DeepSeek 生成选股推荐理由/分析报告**（已有 key）。

## 实现

### 新建 `src/models/llm.py`（DeepSeekClient）
- 基于 `requests` 直连 OpenAI 兼容接口 `https://api.deepseek.com/chat/completions`（无需 openai 包）
- 配置从 `config/deepseek.yaml` 读取，环境变量 `DEEPSEEK_API_KEY` 优先
- 方法：
  - `chat(system, user)` — 通用聊天
  - `analyze_stock(name, code, metrics)` — 个股分析报告（Markdown，5段式：核心结论/技术面/基本面/风险提示/操作参考）
  - `explain_recommendations(stocks)` — 选股组合推荐理由（提炼共性+板块逻辑+风险）
  - `test_connection()` — 测试连接
  - `set_api_key()` / `save()` — 配置管理
- 模型：`deepseek-chat`（V3，默认）/ `deepseek-reasoner`（R1）

### 修改 `src/ui/app.py`
1. **import**：`from src.models.llm import deepseek`
2. **设置页（⚙️）**：新增 "🤖 DeepSeek AI 配置" 区块 — API Key 输入（password 型）+ 模型下拉 + 保存按钮 + 测试连接
3. **个股分析页（📊）**：末尾新增 "🤖 DeepSeek AI 分析报告" — 生成按钮 + 组装指标快照（收盘价/涨跌/MA/RSI/KDJ/BOLL/量能/波动/回撤/夏普/胜率 + AI评分/概率/理由 + 基本面 + 技术信号）
4. **AI选股页（📈）**：推荐详情表后新增 "🤖 DeepSeek 组合解读" — 生成组合推荐理由按钮（取 Top10 传给 LLM）

## 验证
- 语法检查通过（app.py + llm.py）
- llm 模块导入 OK（is_ready=False，未配置 key）
- 请求路径验证：用假 key 调用真实接口，返回 401 `Authentication Fails, Your api key: ****-123 is invalid`，确认接口可达 + 错误处理正确
- Streamlit 重启成功（HTTP 200，PID 16272）

## 待用户操作
1. 打开 http://localhost:8501 → ⚙️ 设置页
2. 填入 DeepSeek API Key → 保存
3. 在 📊 个股分析 或 📈 AI选股 页点击生成按钮

## 文件清单
- 新建：`src/models/llm.py`
- 修改：`src/ui/app.py`（import + 设置页 + 个股分析页 + AI选股页）

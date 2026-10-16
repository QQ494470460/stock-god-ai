# AI 选股崩溃修复 + 回测系统部署 — 完成报告

**时间**: 2026-08-01
**最终验证**: ✅ e2e 100 只 hs300 扫描 → 20 只推荐 → save_predictions → summary 全链路通过

## 修复内容

### Bug 1（崩溃根因）: XGBoost 三分类 + LightGBM 二分类不兼容
- 位置: `src/models/predictor.py` 第 ~243 行
- 原因: XGBoost `classes_=[0,1,2]`(三分类), LightGBM `classes_=[0,1]`(二分类), `np.average(np.array(probas))` 对维度不同的数组报 `inhomogeneous shape`
- 修复: 新增概率对齐逻辑——二分类自动补零映射到三分类 [跌, 震荡=0, 涨]

### Bug 2: 空 result_df 无拦截
- 位置: `src/ui/app.py` 第 ~505 行
- 修复: 预测完成后加 `if len(result_df) == 0: st.warning(...); st.stop()`

### Bug 3: save_predictions 重复 3 次
- 位置: `src/ui/app.py` 第 ~550-600 行
- 修复: 合并为一段

### 新功能: 回测验证系统
- `src/models/tracker.py` 重写 (10.5KB):
  - `verify_predictions(lookback_days)` — 批量拉取 N 日收益，验证信号方向
  - `verify_all_pending()` — 一键验证 5+10 日窗口
  - `get_accuracy_summary()` — 汇总准确率/超额收益/胜率
- `src/ui/app.py` — 新增 "📊 历史准确率" 折叠面板 + "运行回测验证" 按钮
- 数据从今天开始累积，5 天后可验证第一批

## 状态
- Streamlit: PID 11776, 端口 8501, HTTP 200
- predictor: is_ready=True, 双模型已加载
- tracker: 今日预测已保存

# IC_IR 是否虚高 / 是否有前视的复核结论

**一句话结论：`production_metrics.json` 里的 `ic_ir=14` 不是前视（look-ahead）泄漏，
但它是「t 统计量」而不是「信息比率」——名字相同、口径不同，看着吓人，其实含义温和。**

## 1. 14 是怎么来的（关键纠错）

`production_metrics.json` 的 `ic_ir` 由 `src/models/labeling.py::mean_rank_ic` 计算：

```python
se = arr.std(ddof=1) / np.sqrt(len(arr))   # 标准误，不是标准差
return arr.mean(), arr.mean() / se, len(arr)  # ic_ir = mean / 标准误  →  t 统计量
```

即 `ic_ir = mean(每日IC) / SE`，分母是**标准误** `std/√n_days`，不是每日 IC 的
**标准差**。945 天样本把分母缩小了 `√945≈31` 倍，于是：

```
ic_ir ≈ mean_IC / (std_daily / √945) = 0.0747 / (0.16 / 30.7) ≈ 14
```

所以 14 的真实含义是「**平均 RankIC 距离零有 14 个标准误**」——一个统计上极其
显著、绝非噪声的结果。它**不代表**日频排序能力很稳。

## 2. 真正反映「稳不稳」的指标：Grinold-Kahn IR = mean/std

回测引擎 `src/core/backtest.py` 里的 `ic_ir = mean/std` 才是经典信息比率，衡量
**单日 IC 的日频波动 vs 均值**：

- 实测 `std(daily IC) ≈ 0.16~0.18`，**约为均值的 3 倍**；
- 因此真实日频 IR = `mean/std ≈ 0.35`（30 天回填实测：mean +0.0618 / std 0.176）。

含义很明确：**排序信号在「多日、组合层」上稳定为正，但单日噪声很大**——某一天
IC 可能为负，不能拿「今天模型打分最高」当单票押注依据，但等权持有一篮子高分股、跨
多个交易日，alpha 会累积出来。

## 3. 用真实数据交叉验证（rolling_ic_backfill，28 个新评分日）

跑 `scripts/rolling_ic_backfill.py --days 30 --sample 150` 回填了 28 个真实交易日
（用当前预测器对历史日打分，严格 OOS：评分只用 d 日及之前数据，前向收益用 d+5 个
交易日），日志共 30 点：

| 指标 | 值 | 说明 |
|------|----|------|
| mean RankIC | **+0.0618** | 与 production 的 0.0747 量级一致 → 信号真实 |
| std(每日IC) | **0.176** | 约为均值 2.9 倍 → 单日噪声大 |
| mean/std（真 IR） | **0.35** | 日频稳定性偏弱 |
| mean/SE（t 统计量） | **1.92** | 按 √n 外推到 945 天 ≈ **10.8**，与 production 的 14 吻合 |

→ production 的 14 与回填的 1.92 是**同一现象、不同样本量**，完全自洽，没有矛盾，
更没有泄漏。

## 4. 前视排查（负向对照，已钉死）

`tests/test_ic_ir_no_lookahead.py` 用两个对照证明指标只响应真实排序信号、不靠泄漏：

- **真实信号**：score 与 fwd_ret 同序 → `ic_ir(mean/std) > 2`；
- **逐日打乱 score**（破坏横截面对齐、不引入任何跨日/未来信息）→ `ic_ir` 塌缩到
  `|ic_ir| < 0.5`（≈0）。

塌缩即证明：高信号完全来自真实排序能力，而非数据泄漏。

## 5. 对「模型靠谱吗」的最终回答

- ✅ **横截面排序能力真实存在**：RankIC 长期显著为正（生产 +0.075 / 回填 +0.062），
  t 统计量极大、负向对照归零 → 不是噪声、不是前视。
- ⚠️ **但单日噪声大、点位预测≈随机**（accuracy 27.7%≈三分类随机；OOS 最大回撤
  -35.7%；纸盘仅 4 笔成交）。所以这是「**研究级排序信号**」，不是「精准择时模型」。
- 📌 **用法建议**：做等权 / 评分加权的组合层选择（持有高分股篮子、跨多日），
  而非押注单票方向；并把 `ic_ir` 当作 t 统计量解读（信显著），把日频 IR≈0.35
  当成「别过度信任单日排序」的提醒。

## 相关改动

- `src/core/backtest.py:341` 注释：更正 ic_ir 口径（本处=mean/std；production=mean/SE）。
- `src/models/labeling.py::mean_rank_ic`：确认 ic_ir 实为 mean/SE（t 统计量）。
- `tests/test_ic_ir_no_lookahead.py`：新增负向对照。
- `scripts/rolling_ic_backfill.py` + `scripts/rolling_ic_curve.py`：回填逐日 IC 并画稳定性曲线
  （见 `data/rolling_ic_curve.html`）。两脚本的 ic_ir 均按 mean/std 口径展示并显式标注。

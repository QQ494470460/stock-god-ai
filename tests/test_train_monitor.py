# -*- coding: utf-8 -*-
"""训练可观测性：``src/ui/train_monitor.py`` 的日志解析 + UI 侧的回归守卫。

## 为什么这些断言值得单独写

用户的原话是「一键训练优化训练怎么老是不提示训练结束」。查下来**提示逻辑本身
是完整的**（``===TRAINING_COMPLETE===`` → reload → balloons → success），真正
坏掉的是让它永远走不到那一步的四条链：

1. ``wmic`` 在 Windows 11 已被移除 → 进程探测恒失败 → 状态栏**永远**显示
   「进程已退出」。训练明明在跑，界面说它退出了。
2. 阶段判断用全文 ``in`` 匹配 → 日志里一旦出现 ``Optuna TPE`` 就再也退不出该
   分支，Optuna 结束后的最终训练阶段被显示成「超参搜索 10%」，进度条从 60%
   **倒退**回 10% 并永久卡住。
3. 单 trial 要几分钟到十几分钟，而「日志 120s 未更新即报异常终止」写死 120s。
4. 训练是数小时作业，浏览器标签空闲后 Streamlit 回收会话，``session_state``
   清空 → 那句 ``st.success`` 再也不会被渲染。日志成对出现（一条 COMPLETE、
   几分钟后又起一条）就是使用者「没看到提示所以又点了一次」的痕迹。

所以这里的断言不只是「解析函数算得对」，还包括**不允许退回旧实现**。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src.ui import train_monitor as tm  # noqa: E402

APP_PY = ROOT / "src" / "ui" / "app.py"
TRAIN_PY = ROOT / "scripts" / "train_optuna.py"
LOGS = ROOT / "logs"


# ======================================================================
# 阶段判定：不许倒退、不许卡死
# ======================================================================

def _log(*parts: str) -> str:
    return "\n".join(parts)


def test_stage_advances_past_optuna_into_final_training():
    """Optuna 结束后必须真的进入 final 阶段。

    旧实现：``elif is_optuna and ("Optuna TPE" in full_text ...)`` —— ``Optuna TPE``
    一旦出现就永远命中，于是最终训练阶段仍显示成「超参搜索」，pct 掉回 0.1。
    """
    text = _log(
        "🔍 XGBOOST Optuna TPE — 50 trials   目标: 样本外 RankIC",
        "  [1/50] Trial #0 RankIC=+0.04100  最佳=+0.04100  | 已用 3.0min，预计还需 147.0min",
        "🏆 最佳: Trial #7  验证集 RankIC = +0.051234",
        "🔄 最终模型：全量训练 + 5 折 walk-forward 诚实验证",
    )
    p = tm.parse_progress(text)
    assert p.stage == "final"
    assert p.pct > 0.9, f"最终训练阶段不该还停在低位：{p.pct}"
    assert p.total == 5  # 5 折


def test_pct_never_goes_backwards_as_the_log_grows():
    """把日志逐段喂进去，进度只许单调不减 —— 倒退就是「卡死」的观感来源。"""
    stages = [
        "📊 收集面板数据（保留 date/code，供截面 RankIC）…",
        "   进度 425/425 — 已收集 400 只  (用时 90s)",
        "🔍 XGBOOST Optuna TPE — 50 trials   目标: 样本外 RankIC",
        "  [10/50] Trial #9 RankIC=+0.04000  最佳=+0.05000  | 已用 30.0min，预计还需 120.0min",
        "  [50/50] Trial #49 RankIC=+0.04600  最佳=+0.05100  | 已用 150.0min，预计还需 0.0min",
        "🏆 最佳: Trial #7  验证集 RankIC = +0.051234",
        "🔄 最终模型：全量训练 + 5 折 walk-forward 诚实验证",
        "   折: 2019-01-02~2020-06-30 → 2020-07-01~2021-12-31  RankIC=+0.0400 (IC_IR=1.2, 300天)",
        "💾 模型: models/xgboost_model.pkl",
        "===TRAINING_COMPLETE===",
    ]
    seen: list[float] = []
    for i in range(1, len(stages) + 1):
        seen.append(tm.parse_progress(_log(*stages[:i])).pct)
    assert seen == sorted(seen), f"进度出现倒退：{seen}"
    assert seen[-1] == pytest.approx(1.0)


def test_search_stage_uses_the_last_trial_line_not_the_first():
    """ETA/单 trial 耗时必须取最后一次匹配 —— 取第一次会显示半小时前的状态。"""
    text = _log(
        "🔍 XGBOOST Optuna TPE — 50 trials   目标: 样本外 RankIC",
        "  [2/50] Trial #1 RankIC=+0.04000  最佳=+0.05000  | 已用 6.0min，预计还需 144.0min",
        "  [30/50] Trial #29 RankIC=+0.04500  最佳=+0.05100  | 已用 240.0min，预计还需 160.0min",
    )
    p = tm.parse_progress(text)
    assert (p.done, p.total) == (30, 50)
    assert p.per_trial_sec == pytest.approx(240.0 * 60 / 30)   # 480s，不是 180s
    assert p.eta_sec == pytest.approx(160.0 * 60)


def test_first_trial_is_reported_as_running_not_as_idle():
    """0/50 时也要显示「首个 trial 运行中」，而不是停在「启动中」。

    TPE 每跑完一个 trial 才刷新一次，首个 trial 在 zz500 全池上要好几分钟 ——
    这段时间界面必须解释清楚，否则看起来就是卡死。
    """
    text = _log(
        "📊 收集面板数据（保留 date/code，供截面 RankIC）…",
        "🔍 XGBOOST Optuna TPE — 150 trials   目标: 样本外 RankIC",
        "  0%|          | 0/150 [00:00<?, ?it/s]",
    )
    p = tm.parse_progress(text)
    assert p.stage == "search"
    assert (p.done, p.total) == (0, 150)
    assert "首个 trial" in p.detail
    assert p.eta_sec is None          # 还没有任何参照，不许编一个数出来
    assert p.per_trial_sec is None


def test_gather_stage_reports_stock_count():
    text = _log("📊 收集面板数据（保留 date/code，供截面 RankIC）…",
                "   进度 200/425 — 已收集 198 只  (用时 42s)")
    p = tm.parse_progress(text)
    assert p.stage == "gather"
    assert (p.done, p.total) == (200, 425)
    assert "425" in p.detail
    assert 0.0 < p.pct < 0.1


def test_empty_log_is_prepare_and_not_a_crash():
    p = tm.parse_progress("")
    assert p.stage == "prepare"
    assert p.pct == 0.0


# ======================================================================
# 终态判定
# ======================================================================

def test_complete_marker_wins_even_if_earlier_phases_mentioned_failure():
    text = _log("Traceback (most recent call last):", "...", "===TRAINING_COMPLETE===")
    assert tm.detect_state(text) == "complete"

    text2 = _log("===TRAINING_FAILED===")
    assert tm.detect_state(text2) == "failed"


def test_a_long_crash_traceback_far_from_the_tail_is_ignored():
    """只认尾部 4KB 内的 traceback。

    中段出现过的 traceback 可能来自某个可恢复的可选步骤（例如 SHAP/版本化保存
    失败只 warn 不致命），拿它判失败会把一次成功的训练报成失败。
    """
    text = ("Traceback (most recent call last):\n" + "x" * 5000
            + "\n🔍 XGBOOST Optuna TPE — 50 trials\n  [3/50] Trial #2 RankIC=+0.04")
    assert tm.detect_state(text) == "running"


# ======================================================================
# stall 阈值：不许再用写死的 120s
# ======================================================================

def test_stall_limit_is_never_as_tight_as_the_old_120s():
    """旧实现 120s 没新日志就报「🔴 可能异常终止」，而单 trial 就要几分钟。"""
    cases = [
        tm.parse_progress("📊 收集面板数据…"),
        tm.parse_progress("🔍 XGBOOST Optuna TPE — 50 trials"),
        tm.parse_progress("🔄 最终模型：全量训练 + 5 折 walk-forward 诚实验证"),
    ]
    for p in cases:
        assert tm.stall_limit_sec(p) > 120.0


def test_stall_limit_scales_with_measured_per_trial_time():
    """实测单 trial 600s 时，容差必须随之放大到 3 倍以上。"""
    text = _log("🔍 XGBOOST Optuna TPE — 50 trials",
                "  [5/50] Trial #4 RankIC=+0.04  最佳=+0.05  | 已用 50.0min，预计还需 450.0min")
    p = tm.parse_progress(text)
    assert p.per_trial_sec == pytest.approx(600.0)
    assert tm.stall_limit_sec(p) >= 1800.0


# ======================================================================
# 小工具
# ======================================================================

@pytest.mark.parametrize("sec,expected", [
    (None, "—"),
    (float("nan"), "—"),
    (-1, "—"),
    (45, "45 秒"),
    (120, "2 分"),
    (150, "2 分 30 秒"),
    (3600, "1 小时"),
    (5000, "1 小时 23 分"),
])
def test_fmt_duration(sec, expected):
    assert tm.fmt_duration(sec) == expected


def test_read_log_tail_survives_a_missing_file(tmp_path):
    assert tm.read_log_tail(tmp_path / "nope.log") == ""


def test_read_log_tail_keeps_the_end_of_a_large_file(tmp_path):
    p = tmp_path / "big.log"
    p.write_text("A" * 300_000 + "\n===TRAINING_COMPLETE===\n", encoding="utf-8")
    tail = tm.read_log_tail(p, max_bytes=64 * 1024)
    assert "===TRAINING_COMPLETE===" in tail
    assert len(tail) <= 64 * 1024 + 64


# ======================================================================
# 真实历史日志（logs/ 下的取证样本）
# ======================================================================

def test_real_killed_run_reports_its_partial_progress():
    """logs/train_ui_20261018_214504.log 是跑到 12/20 被掐断的一次。

    它的尾部是 tqdm 与 loguru 挤在同一行的字符汤，旧实现靠
    ``re.search(r'(\\d+)/(\\d+)\\s*\\[')`` 去抠，拿到的数字并不可靠。
    """
    f = LOGS / "train_ui_20261018_214504.log"
    if not f.exists():
        pytest.skip("历史日志已清理")
    p = tm.parse_progress(tm.read_log_tail(f))
    assert p.stage == "search"
    assert p.total == 20
    assert 1 <= p.done <= 20
    assert p.per_trial_sec and p.per_trial_sec > 30


def test_real_completed_run_is_recognised():
    f = LOGS / "train_ui_20261029_204914.log"
    if not f.exists():
        pytest.skip("历史日志已清理")
    text = tm.read_log_tail(f)
    assert tm.detect_state(text) == "complete"
    assert tm.parse_progress(text).pct == pytest.approx(1.0)


# ======================================================================
# UI 回归守卫：不许退回旧实现
# ======================================================================

def test_ui_no_longer_shells_out_to_wmic():
    """wmic 在 Windows 11 已被移除，用它探测进程会恒失败 → 永远显示「已退出」。

    说明性注释里提到 wmic 是允许的（不然后人又要踩一遍），所以只禁代码形态。
    """
    src = APP_PY.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert '"wmic"' not in code and "'wmic'" not in code


def test_ui_process_liveness_uses_the_popen_handle_or_psutil():
    src = APP_PY.read_text(encoding="utf-8")
    assert "def _pid_alive" in src
    assert "poll() is None" in src          # Popen 句柄优先
    assert "psutil" in src                  # 句柄丢失时用 pid 兜底


def test_ui_persists_the_run_so_a_refresh_cannot_swallow_the_notice():
    """训练是数小时作业，会话被回收后必须还能补上完成提示。"""
    src = APP_PY.read_text(encoding="utf-8")
    assert "_train_run_save" in src and "_train_run_load" in src
    assert "def _resume_train_run" in src
    assert '"notified"' in src               # 只提示一次，不重复弹


@pytest.mark.parametrize("lie", ["(2分钟)", "(15分钟)", "(40分钟)"])
def test_ui_duration_labels_are_not_the_old_lies(lie):
    """三档实测分别是 36~56 分钟 / 从未跑完 / 从未跑过，旧标签差了 25~77 倍。

    使用者按「15 分钟」的预期等十几小时，必然认为卡死并反复重开。
    """
    assert lie not in APP_PY.read_text(encoding="utf-8")


def test_no_deploy_run_cannot_clobber_the_deployed_search_record():
    """``optuna_search_result.json`` 记录「当前已部署模型用的哪组超参」。

    ``--no-deploy`` 的语义是不碰生产产物（模型/scaler/features 都已另存
    ``*_optuna_*``），但它**仍然**会写 RESULT_FILE —— 于是一次 2 个 trial 的
    冒烟跑就把生产模型的搜索记录换成了试验记录；而 ``production_metrics.json``
    只在 deploy=True 时更新，两份记录直接互相矛盾。2026-11-04 实际踩到过：
    一个 2-trial 验证跑覆盖了 10-29 那次 20-trial 的结果，只能从日志逐项复原。
    """
    src = TRAIN_PY.read_text(encoding="utf-8")
    assert src.count("result_file = RESULT_FILE if deploy else") == 2, \
        "两条管线（rankic / f1）都必须区分部署与试验的写入口"
    assert "optuna_search_result_trial.json" in src
    # 不许再有无条件的 RESULT_FILE 写入
    assert "with open(RESULT_FILE" not in src


def test_ui_duration_labels_track_the_measured_per_trial_cost():
    """三档标注必须与实测同量级，否则使用者会按错的预期等。

    2026-11-04 实测：小跑 148538 样本 / mean 75 棵树 → 14.6s/trial（全核），
    折到 zz500 全池 803603 样本 / mean 775 棵树 ≈ 13.5 分钟 / trial。
    """
    src = APP_PY.read_text(encoding="utf-8")
    assert "(实测≈12小时)" in src     # 50 trials
    assert "(实测≈40分钟)" in src     # 20 trials / 100 只
    assert "≈3天" in src              # 300 trials


def test_ui_mid_preset_uses_50_trials_and_passes_full_threads():
    src = APP_PY.read_text(encoding="utf-8")
    mid = src.split('elif mode == "optuna_mid":')[1].split('mode_label = "Optuna 标准"')[0]
    assert '"--n-trials", "50"' in mid
    assert '"--xgb-threads", "-1"' in mid


def test_ui_fast_preset_no_longer_passes_a_silently_ignored_max_samples():
    """--max-samples 只在旧口径 gather_data 生效，RankIC 口径下静默无效。"""
    src = APP_PY.read_text(encoding="utf-8")
    fast = src.split("# optuna_fast")[1].split('mode_label = "Optuna 快速"')[0]
    args_only = "\n".join(
        line for line in fast.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--max-samples" not in args_only


def test_ui_imports_the_shared_monitor_module_instead_of_reparsing_tqdm():
    src = APP_PY.read_text(encoding="utf-8")
    assert "from src.ui import train_monitor as tm" in src
    # 旧的 tqdm 抠字法（N/M [）不许回来
    assert r"(\d+)/(\d+)\s*\[" not in src


# ======================================================================
# 训练脚本：搜索期线程数
# ======================================================================

@pytest.fixture(scope="module")
def train_mod():
    spec = importlib.util.spec_from_file_location("_train_optuna_under_test", TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeTrial:
    def suggest_int(self, name, lo, hi):  # noqa: D102
        return (lo + hi) // 2

    def suggest_float(self, name, lo, hi, log=False):  # noqa: D102
        return (lo + hi) / 2


def test_search_threads_reach_xgboost(train_mod):
    """历史上 ``_suggest_params`` 写死 n_jobs=1，而 optuna 也恒为 n_jobs=1 ——
    整轮搜索只用一个核。12 核机实测单 trial 61.5s vs 26.4s。"""
    assert train_mod._suggest_params(_FakeTrial(), 1500)["n_jobs"] == 1
    assert train_mod._suggest_params(_FakeTrial(), 1500, -1)["n_jobs"] == -1
    assert train_mod._suggest_params(_FakeTrial(), 1500, 4)["n_jobs"] == 4


def test_changing_threads_does_not_touch_the_search_space(train_mod):
    """线程数是资源参数，绝不能顺手改了搜索空间 —— 那等于换了一个搜索问题。"""
    a = train_mod._suggest_params(_FakeTrial(), 1500, 1)
    b = train_mod._suggest_params(_FakeTrial(), 1500, -1)
    assert set(a) == set(b)
    assert {k: v for k, v in a.items() if k != "n_jobs"} == \
           {k: v for k, v in b.items() if k != "n_jobs"}


def test_rankic_path_warns_that_max_samples_is_ignored():
    """与其静默忽略，不如明确告警 —— 这正是「参数在但没用」那类最难查的坑。"""
    src = TRAIN_PY.read_text(encoding="utf-8")
    body = src.split("def _run_rankic(")[1].split("def main(")[0]
    assert "max_samples" in body
    assert "无效" in body and "忽略" in body


def test_too_few_stocks_is_rejected_before_fetching_anything(train_mod):
    """--max-stocks 小于每日截面下限时，必然 0 样本。

    ``make_rank_labels`` 会把「当日有效股票数 < MIN_STOCKS_PER_DAY」的交易日
    整行作废；池子比它还小 ⇒ 每个交易日都被作废 ⇒ 面板 0 行，最后只报一句
    「未收集到任何训练样本」，完全看不出是 --max-stocks 太小。
    实测：--max-stocks 20 时白跑了一分多钟取数才发现。
    """
    import argparse

    from loguru import logger as _lg

    args = argparse.Namespace(max_samples=0, max_stocks=20)
    msgs: list[str] = []
    sink = _lg.add(lambda m: msgs.append(m), level="ERROR", format="{message}")
    try:
        train_mod._run_rankic(args, 0.0, deploy=False)
    finally:
        _lg.remove(sink)

    text = "".join(msgs)
    assert "低于每日截面下限" in text
    assert "===TRAINING_FAILED===" in text   # 必须留下终态标记，否则 UI 会永远轮询

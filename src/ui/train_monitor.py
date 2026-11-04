"""训练进度解析（纯函数，不 import streamlit，便于单测）。

## 为什么单独成模块

`src/ui/app.py` 里原先那套阶段判断有三个结构性错误，合起来造成了
「训练跑完了却不提示」的观感：

1. **用全文 `in` 匹配判阶段**。一旦日志里出现过 ``Optuna TPE``，
   ``elif is_optuna and ("Optuna TPE" in full_text ...)`` 此后**恒真** ——
   Optuna 结束后的「全量训练 + K 折 walk-forward 验证」阶段仍被显示成
   「🔍 Optuna超参搜索」，而那时 ``full_text[-3000:]`` 里已经没有 tqdm 的
   ``N/M [`` 了，于是 pct 掉回 0.1 并永久卡住。进度条从 60% 倒退到 10%，
   看起来就像卡死。
2. **去 tqdm 的 ``N/M [`` 里抠数字**。tqdm 直接写 stdout（``\\r`` 刷新、
   不换行），loguru 又往同一个文件写带换行的整行，两者会挤进同一行，
   尾部是不可靠的字符汤。见 logs/train_ui_20261018_214504.log。
3. **固定百分比**（数据 5% / Optuna 10%~60% / SHAP 60% / 重训 70%+）。
   实际 Optuna 占整轮 95% 以上的时间，把 50% 的条宽分给 95% 的时长，
   每个 trial 只有 0.33% 的位移，视觉上等于不动。

脚本其实每个 trial 都主动打了一行**结构化、带 ETA** 的日志
（``scripts/train_optuna.py::_TrialProgress``）::

    [12/150] Trial #11 RankIC=+0.04839  最佳=+0.05070  | 已用 15.2min，预计还需 12.4min

本模块只认这行，以及脚本里的阶段横幅。阶段判定改为**取出现位置最靠后
的那个标记**，因此不会像旧逻辑那样「进去就出不来」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ==================== 阶段标记（与 scripts/train_optuna.py 的日志字符串一一对应）====================

M_GATHER = "收集面板数据"
M_SEARCH = "Optuna TPE —"
M_BEST = "🏆 最佳: Trial"
M_FINAL = "最终模型：全量训练"
M_SAVE = "💾 模型:"
M_DONE = "===TRAINING_COMPLETE==="
M_FAILED = "===TRAINING_FAILED==="
M_TRACEBACK = "Traceback (most recent call last):"

# 每个标记把日志推进到下一个阶段
_STAGE_MARKERS: tuple[tuple[str, str], ...] = (
    (M_GATHER, "gather"),
    (M_SEARCH, "search"),
    (M_BEST, "final"),
    (M_FINAL, "final"),
    (M_SAVE, "save"),
    (M_DONE, "done"),
)

# ==================== 正则 ====================

_TRIAL_RE = re.compile(r"\[(\d+)/(\d+)\]\s*Trial\s*#(\d+)")
_ETA_RE = re.compile(r"已用\s*([\d.]+)\s*min[，,]\s*预计还需\s*([\d.]+)\s*min")
_SEARCH_TOTAL_RE = re.compile(r"Optuna TPE\s*[—–-]\s*(\d+)\s*trials")
_GATHER_PROG_RE = re.compile(r"进度\s*(\d+)/(\d+)\s*[—–-]")
_GATHER_SECS_RE = re.compile(r"已收集\s*\d+\s*只\s*\(用时\s*(\d+)s\)")
_CV_FOLDS_RE = re.compile(r"(\d+)\s*折\s*walk-forward")
_FOLD_LINE_RE = re.compile(r"折:\s*\d{4}-\d{2}-\d{2}~")
_BEST_RE = re.compile(r"最佳=([+-][\d.]+)")

# ==================== 进度条份额 ====================
# 按「各阶段实际耗时占比」分配条宽，而不是按「有几个阶段」均分。
# Optuna 搜索在 50 trials 档占整轮 90%+ 的时间，所以给它 8%~92%。
_GATHER_LO, _GATHER_HI = 0.01, 0.08
_SEARCH_LO, _SEARCH_HI = 0.08, 0.92
_FINAL_LO, _FINAL_HI = 0.92, 0.99

_STAGE_TITLE = {
    "prepare": "⏳ 启动中，等待日志…",
    "gather": "📊 收集面板数据",
    "search": "🔍 Optuna 超参搜索",
    "final": "🔄 全量训练 + walk-forward 验证",
    "save": "💾 保存模型产物",
    "done": "✅ 训练完成",
}


@dataclass(frozen=True)
class TrainProgress:
    """一轮训练的可视化状态。全部字段都可能缺失（日志只写了一半时）。"""

    stage: str = "prepare"
    pct: float = 0.0
    title: str = _STAGE_TITLE["prepare"]
    detail: str = ""
    done: int = 0
    total: int = 0
    eta_sec: float | None = None
    per_trial_sec: float | None = None
    best: float | None = None

    @property
    def in_search(self) -> bool:
        return self.stage == "search"


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """取**最后**一次匹配 —— 进度类信息必须取最后一次，取第一次会显示很久以前的状态。"""
    last = None
    for m in pattern.finditer(text):
        last = m
    return last


def _stage_of(text: str) -> str:
    """按「标记出现的最后位置」判阶段，避免旧逻辑「进去不出」的死锁。"""
    stage = "prepare"
    best_pos = -1
    for marker, mapped in _STAGE_MARKERS:
        pos = text.rfind(marker)
        if pos > best_pos:
            best_pos = pos
            stage = mapped
    return stage


def detect_state(text: str) -> str:
    """``complete`` / ``failed`` / ``running``。

    只看日志文本，不看进程 —— 进程存活由调用方用 psutil 判，两者互补：
    原生崩溃（如 py_mini_racer.dll 硬崩）会留下 traceback；
    被 taskkill 则什么标记都不留，只能靠进程判。
    """
    if M_DONE in text:
        return "complete"
    if M_FAILED in text:
        return "failed"
    if M_TRACEBACK in text[-4000:]:
        return "failed"
    return "running"


def parse_progress(text: str) -> TrainProgress:
    """日志文本 → 阶段 / 百分比 / 已完成的 trial 数 / ETA。"""
    stage = _stage_of(text)

    if stage == "done":
        return TrainProgress(stage="done", pct=1.0, title=_STAGE_TITLE["done"],
                             detail="模型已写入并热加载")

    # ---- 阶段一：收集面板 ----
    if stage == "gather":
        got = _last_match(_GATHER_PROG_RE, text)
        if got:
            i, n = int(got.group(1)), int(got.group(2))
            frac = min(max(i / max(n, 1), 0.0), 1.0)
            detail = f"{i}/{n} 只"
            secs = _last_match(_GATHER_SECS_RE, text)
            if secs:
                detail += f"（已用 {int(secs.group(1))}s）"
            return TrainProgress(
                stage="gather", pct=_GATHER_LO + (_GATHER_HI - _GATHER_LO) * frac,
                title=_STAGE_TITLE["gather"], detail=detail, done=i, total=n,
            )
        return TrainProgress(stage="gather", pct=_GATHER_LO,
                             title=_STAGE_TITLE["gather"], detail="开始取数…")

    # ---- 阶段二：Optuna 搜索 ----
    if stage == "search":
        total = 0
        t = _last_match(_SEARCH_TOTAL_RE, text)
        if t:
            total = int(t.group(1))
        done, last_trial = 0, 0
        tr = _last_match(_TRIAL_RE, text)
        if tr:
            done, total, last_trial = int(tr.group(1)), int(tr.group(2)) or total, int(tr.group(3))

        eta_sec = per_trial = None
        eta = _last_match(_ETA_RE, text)
        if eta:
            elapsed_min, remain_min = float(eta.group(1)), float(eta.group(2))
            eta_sec = remain_min * 60.0
            if done > 0:
                per_trial = elapsed_min * 60.0 / done
        best = _last_match(_BEST_RE, text)

        frac = done / total if total else 0.0
        detail = f"第 {done}/{total} 个 trial" if total else "等待首个 trial"
        if done:
            detail += f"（Trial #{last_trial}）"
        else:
            detail = "首个 trial 运行中（TPE 每跑完一个才刷新）"
        return TrainProgress(
            stage="search", pct=_SEARCH_LO + (_SEARCH_HI - _SEARCH_LO) * min(frac, 1.0),
            title=_STAGE_TITLE["search"], detail=detail,
            done=done, total=total, eta_sec=eta_sec, per_trial_sec=per_trial,
            best=float(best.group(1)) if best else None,
        )

    # ---- 阶段三：全量训练 + 折验证 ----
    if stage == "final":
        tr = _last_match(_TRIAL_RE, text)
        last_trial = int(tr.group(3)) if tr else 0
        folds_done = len(_FOLD_LINE_RE.findall(text))
        cv = _last_match(_CV_FOLDS_RE, text)
        folds_total = int(cv.group(1)) if cv else 0
        frac = folds_done / folds_total if folds_total else 0.0
        detail = f"{folds_done}/{folds_total} 折已验" if folds_total else "训练最终模型…"
        best = _last_match(_BEST_RE, text)
        return TrainProgress(
            stage="final", pct=_FINAL_LO + (_FINAL_HI - _FINAL_LO) * min(frac, 1.0),
            title=_STAGE_TITLE["final"], detail=detail,
            done=folds_done, total=folds_total,
            best=float(best.group(1)) if best else None,
            # 单折验证在 80 万样本上要几分钟，给一个可用的粗估
            eta_sec=(4.0 * 60.0 * (folds_total - folds_done)) if folds_total else None,
        )

    if stage == "save":
        return TrainProgress(stage="save", pct=0.99, title=_STAGE_TITLE["save"], detail="写入 .pkl")

    return TrainProgress(stage="prepare", pct=0.0, title=_STAGE_TITLE["prepare"])


def stall_limit_sec(prog: TrainProgress) -> float:
    """超过这么久没有新的日志行，才值得提示「可能异常终止」。

    旧实现写死 120s，而单个 trial 在 zz500 全池（80 万样本 × 110 特征）上
    要几分钟到十几分钟 —— 正常训练 2 分钟就被判成「🔴 可能异常终止」。
    这里按实测单 trial 耗时动态放宽，且**只作为提示**，绝不用它做终态判定。
    """
    if prog.stage == "search":
        if prog.per_trial_sec:
            return max(600.0, 3.0 * prog.per_trial_sec)
        # 首个 trial 结束前没有任何参照，而它恰恰是最慢的一个（TPE 冷启动）
        return 1800.0
    if prog.stage == "final":
        return 1800.0
    return 600.0


def fmt_duration(sec: float | None) -> str:
    """秒 → 「1 小时 23 分」。None/非法 → 「—」。"""
    if sec is None or sec != sec or sec < 0:  # NaN 或负数
        return "—"
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} 秒"
    if sec < 3600:
        return f"{sec // 60} 分 {sec % 60} 秒" if sec % 60 else f"{sec // 60} 分"
    h, m = divmod(sec // 60, 60)
    return f"{h} 小时 {m} 分" if m else f"{h} 小时"


def read_log_tail(path: str | Path, max_bytes: int = 512 * 1024) -> str:
    """只读日志尾部。

    训练日志会随 tqdm 的 ``\\r`` 刷新不断膨胀（单次运行可达数 MB），
    而 UI 每 4 秒读一次，整文件读会随着时间推移越来越慢。
    阶段标记和完成标记总在尾部，读尾部不丢信息。
    """
    p = Path(path)
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                # 丢掉可能被截断的半行
                f.readline()
            raw = f.read()
        # tqdm 用 \r 刷新，splitlines 会把它当行分隔 —— 恰好能把进度条拆成可读的逐步状态
        return raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 日志读不到不该让整页崩掉
        return ""

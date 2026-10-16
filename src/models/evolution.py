# -*- coding: utf-8 -*-
"""
自我进化引擎 (Self-Evolution Engine)

让"选股神 AI"根据市场实际表现自我调整，形成闭环：
  预测 → 记录 → N日后验证 → 反馈调整参数 → 更准的预测

核心机制（4 个可自动调整的维度）：
  1. 模型权重进化   — 已下线：全项目统一为单一 XGBoost 模型，权重恒为 {"xgboost": 1.0}
  2. 推荐阈值进化   — 根据历史准确率动态调整 min_score（准→放宽，不准→收紧）
  3. 市场状态适配   — 识别牛/熊/震荡，切换对应参数组合（regime profile）
  4. 预测时段适配   — 区分 T+5 / T+10 窗口，各自独立校准

数据存储: data/evolution_state.json
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger

from config.settings import DATA_DIR

EVOLUTION_FILE = DATA_DIR / "evolution_state.json"

# 权重调整参数
WEIGHT_STEP = 0.05           # 每次调整步长
WEIGHT_MIN = 0.2             # 单模型权重下限
WEIGHT_MAX = 0.8             # 单模型权重上限
CONFIDENCE_WINDOW = 3        # 至少需要多少批验证数据才启动进化

# 阈值调整参数
THRESHOLD_BASE = 50.0        # 基准最低评分
THRESHOLD_MIN = 30.0         # 有效阈值下限
THRESHOLD_MAX = 60.0         # 有效阈值上限

# 市场状态
REGIME_PROFILES = {
    "bull": {   # 牛市：激进，阈值略降、权重偏向高确信度模型
        "min_score_adjust": -3.0,
        "name": "🐂 牛市",
    },
    "bear": {   # 熊市：保守，阈值提高
        "min_score_adjust": +5.0,
        "name": "🐻 熊市",
    },
    "sideways": {  # 震荡：中性
        "min_score_adjust": 0.0,
        "name": "⚖️ 震荡市",
    },
}


class EvolutionEngine:
    """自我进化引擎"""

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = Path(state_file) if state_file else EVOLUTION_FILE
        self.state = self._load()
        self.last_effective_min_score_raw: Optional[float] = None

    # ========== 状态持久化 ==========

    def _default_state(self) -> Dict:
        return {
            "version": 2,
            "initialized_at": datetime.now().isoformat(),
            # 模型权重（单模型：仅 xgboost）
            "model_weights": {"xgboost": 1.0},
            # 阈值校准
            "min_score_base": THRESHOLD_BASE,
            "threshold_adjustment": 0.0,
            # 市场状态
            "current_regime": "sideways",
            "regime_detected_at": None,
            # 分窗口性能记录（用于阈值进化）
            "performance": {
                "t5": {"accuracy": [], "n": 0},
                "t10": {"accuracy": [], "n": 0},
            },
            # ★ 离线进化（自动重训）配置
            "offline_evolution": {
                "enabled": False,                 # 是否启用自动重训
                "last_retrain_at": None,          # 上次重训**结束**时间
                "retrain_started_at": None,       # 本次重训**开始**时间（判断僵尸标记用）
                "retrain_running": False,         # 是否有重训任务在跑
                "retrain_timeout_hours": 6,       # 超过此时长未结束 → 判为卡死，自动清理
                "min_interval_days": 7,           # 两次重训最小间隔（天）
                "accuracy_threshold": 0.45,       # 近期准确率低于此值才触发
                "min_batches": 3,                 # 至少需要多少批验证数据
                "recent_window": 3,               # 取最近 N 批平均准确率
                "lookback_years": 2,              # 重训用近期数据年数
            },
            # 进化历史
            "evolution_log": [],
            "last_evolved_at": None,
            # ★ 观测心跳：每次进化作业被调度并跑完（无论是否有可验证样本）都会刷新。
            # 用于区分「作业没跑」与「跑了但还没有可验证样本」——否则 last_evolved_at
            # 长期不动会让人误以为调度链路断了。
            "last_check_at": None,
        }

    def _load(self) -> Dict:
        state = None
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"进化状态加载失败，使用默认值: {e}")
                state = None

        if not isinstance(state, dict):
            return self._default_state()

        # 迁移：全项目已统一为单一 XGBoost 模型。老状态文件的 model_weights
        # 里可能仍残留 lightgbm 键（会污染权重），发现即整体重置并落盘。
        weights = state.get("model_weights")
        if not isinstance(weights, dict) or weights != {"xgboost": 1.0}:
            self.state = state
            state["model_weights"] = {"xgboost": 1.0}
            self.save()
            logger.info("🧬 进化状态已迁移: model_weights → {'xgboost': 1.0}（单模型）")

        return state

    def save(self):
        self.state_file.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def mark_checked(self):
        """观测心跳：记录「进化作业被调度并跑完」的时间戳。

        与 ``last_evolved_at`` 的区别：后者只在真正发生进化回合时更新；
        前者**每次作业跑完就更新**（即使当前没有可验证样本、进化动作为 0）。

        用途：``last_evolved_at`` 长期不动无法区分「调度链路断了」还是
        「只是还没积累到可验证样本」。监控时两者对照即可一眼判明——
        ``last_check_at`` 在动而 ``last_evolved_at`` 不动 = 正常（样本未到）；
        两者都停 = 作业真的没被调度。
        """
        self.state["last_check_at"] = datetime.now().isoformat()
        self.save()

    def _log_evolution(self, action: str, detail: str):
        entry = {
            "time": datetime.now().isoformat(),
            "action": action,
            "detail": detail,
        }
        self.state.setdefault("evolution_log", []).append(entry)
        # 只保留最近 100 条
        self.state["evolution_log"] = self.state["evolution_log"][-100:]
        self.state["last_evolved_at"] = entry["time"]
        logger.info(f"🧬 自我进化 → {action}: {detail}")

    # ========== 核心：从验证结果中学习 ==========

    def evolve_from_verification(self, verification: Dict) -> List[str]:
        """
        根据一次回测验证结果，执行自我进化。

        参数 verification: tracker.verify_predictions() 的返回值

        返回: 本次进化执行的动作列表（供 UI 展示）
        """
        actions = []

        # 记录分窗口性能（副作用），本函数不再负责「记录」这件事
        self.record_verification(verification)
        accuracy = verification.get("accuracy") or 0

        # ---- 推荐阈值进化 ----
        # （模型权重进化已随单模型改造下线，不再有两模型权重可调）
        t_action = self._evolve_threshold(accuracy)
        if t_action:
            actions.append(t_action)

        self.save()
        return actions

    def record_verification(self, verification: Dict) -> str:
        """**只记录**一次验证结果的分窗口准确率，不触发任何参数变更。

        为什么单独拆出来：新进化回路（`src/evolution/`）需要「先收集事实、
        再由带闸门的生成器提候选」；把「记录」与「改参数」混在一个函数里，
        就会出现「读了数据就顺手改了参数」的越权路径。

        窗口键取 ``lookback_days``（旧 tracker）或 ``hold_days``（tracker_v2）。
        后者缺失会导致 T+10 的结果被错记到 t5 上，**污染重训触发判定**。

        返回: 窗口键，如 ``"t5"``
        """
        lookback = (verification.get("lookback_days")
                    or verification.get("hold_days") or 5)
        try:
            lookback = int(lookback)
        except (TypeError, ValueError):
            lookback = 5
        accuracy = verification.get("accuracy")
        if accuracy is None:
            return f"t{lookback}"

        window_key = f"t{lookback}"
        perf = self.state["performance"].setdefault(
            window_key, {"accuracy": [], "n": 0})
        perf["accuracy"].append(round(float(accuracy), 3))
        perf["accuracy"] = perf["accuracy"][-20:]      # 只保留最近 20 次
        perf["n"] += 1
        return window_key

    # ========== 离线进化（自动重训）==========

    def _retrain_stale(self, cfg: Dict) -> Tuple[bool, str]:
        """判断 `retrain_running=True` 是否是个「僵尸标记」。

        重训是长任务，若进程中途被杀/崩溃，`mark_retrain_finished()`
        永远不会执行，标记就永久留在 True —— `should_retrain()` 从此恒返回
        「已有重训任务在运行中」，**自动进化彻底瘫痪**（实测线上卡了 6 周）。

        故引入超时：超过 `retrain_timeout_hours` 未结束即判为卡死。
        取不到开始时间时回退到 `last_retrain_at`；两者都不可用则判为异常。

        返回: (是否已卡死, 原因)
        """
        if not cfg.get("retrain_running"):
            return False, ""

        started = cfg.get("retrain_started_at") or cfg.get("last_retrain_at")
        timeout_h = float(cfg.get("retrain_timeout_hours", 6) or 6)
        if not started:
            return True, "无任何开始时间记录"
        try:
            elapsed_h = (datetime.now() - datetime.fromisoformat(started)).total_seconds() / 3600
        except Exception:  # noqa: BLE001
            return True, f"开始时间无法解析（{started!r}）"
        if elapsed_h > timeout_h:
            return True, f"已持续 {elapsed_h:.1f} 小时 > 超时 {timeout_h:.0f} 小时"
        return False, ""

    def should_retrain(self) -> Tuple[bool, str]:
        """
        判断是否应该触发离线重训（适应市场风格漂移）。

        触发条件（需全部满足）：
          1. 用户已启用离线进化（offline_evolution.enabled）
          2. 当前无重训任务在跑（僵尸标记会被自动清理）
          3. 距上次重训 ≥ min_interval_days 天
          4. 已积累 ≥ min_batches 批验证数据
          5. 最近 recent_window 批平均准确率 < accuracy_threshold

        返回: (是否触发, 原因说明)
        """
        cfg = self.state.get("offline_evolution", {})

        if not cfg.get("enabled", False):
            return False, "离线进化未启用"
        if cfg.get("retrain_running", False):
            stale, why = self._retrain_stale(cfg)
            if not stale:
                return False, "已有重训任务在运行中"
            # 清理僵尸标记：否则自动进化会被永久卡住
            cfg["retrain_running"] = False
            cfg["retrain_started_at"] = None
            self._log_evolution("offline_retrain",
                                f"清理卡死的重训标记（{why}），恢复自动进化")
            self.save()
            logger.warning(f"🧬 检测到卡死的重训标记（{why}），已自动清理")

        # 距离上次重训的时间
        last = cfg.get("last_retrain_at")
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                elapsed_days = (datetime.now() - last_dt).days
                min_interval = cfg.get("min_interval_days", 7)
                if elapsed_days < min_interval:
                    return False, f"距上次重训仅 {elapsed_days} 天（最小间隔 {min_interval} 天）"
            except Exception:
                pass

        # 数据量检查
        min_batches = cfg.get("min_batches", 3)
        recent_window = cfg.get("recent_window", 3)
        # 用 T+5 窗口作为主判断依据（样本更充足）
        t5_accs = self.state.get("performance", {}).get("t5", {}).get("accuracy", [])
        if len(t5_accs) < min_batches:
            return False, f"验证数据不足（{len(t5_accs)}/{min_batches} 批）"

        # 近期准确率
        recent = t5_accs[-recent_window:]
        recent_avg = sum(recent) / len(recent)
        threshold = cfg.get("accuracy_threshold", 0.45)
        if recent_avg >= threshold:
            return False, f"近期准确率 {recent_avg:.1%} ≥ 阈值 {threshold:.1%}，无需重训"

        return True, (f"近期准确率 {recent_avg:.1%} < 阈值 {threshold:.1%}，"
                      f"市场可能发生风格漂移，建议用近期数据重训")

    def mark_retrain_started(self):
        """标记重训任务开始（同时记录开始时间，供超时判定）"""
        cfg = self.state.setdefault("offline_evolution", {})
        cfg["retrain_running"] = True
        cfg["retrain_started_at"] = datetime.now().isoformat()
        self._log_evolution("offline_retrain", "自动重训任务已启动")
        self.save()

    def mark_retrain_finished(self, f1: Optional[float] = None):
        """标记重训任务完成"""
        cfg = self.state.setdefault("offline_evolution", {})
        cfg["retrain_running"] = False
        cfg["retrain_started_at"] = None
        cfg["last_retrain_at"] = datetime.now().isoformat()
        detail = "自动重训完成" + (f" (F1={f1:.4f})" if f1 is not None else "")
        self._log_evolution("offline_retrain", detail)
        self.save()

    def mark_retrain_failed(self, reason: str = ""):
        """标记重训任务失败"""
        cfg = self.state.setdefault("offline_evolution", {})
        cfg["retrain_running"] = False
        cfg["retrain_started_at"] = None
        self._log_evolution("offline_retrain", f"自动重训失败: {reason}")
        self.save()

    def set_offline_config(self, **kwargs):
        """更新离线进化配置"""
        cfg = self.state.setdefault("offline_evolution", {})
        cfg.update(kwargs)
        self.save()
        logger.info(f"离线进化配置已更新: {kwargs}")

    def _evolve_threshold(self, accuracy: float) -> Optional[str]:
        """根据准确率动态调整推荐阈值。

        安全边界
        --------
        ``threshold_adjustment`` 变小 = 「放宽进池标准」= **朝更激进方向**。
        静默放宽是安全红线：模型一旦不准，放宽阈值只会让组合承接更多劣质标的。
        因此本函数**只自动执行收紧方向**（new_adj >= old_adj）；
        放宽方向仅返回一条建议文案，不写入任何状态，需人工批准。

        这与 ``src/evolution/space.py`` 的登记一致：
        ``evolution.threshold_adjustment`` 的 ``risk="risky_down"``、
        ``auto_adopt=False``。
        """
        if accuracy <= 0:
            return None

        old_adj = self.state["threshold_adjustment"]

        if accuracy >= 0.65:
            # 很准 → 想放宽阈值，让更多标的进入推荐（激进方向）
            new_adj = old_adj - 2.0
        elif accuracy <= 0.45:
            # 不准 → 收紧阈值，只保留高评分标的（保守方向，可自动）
            new_adj = old_adj + 3.0
        else:
            # 中等 → 轻微回调
            new_adj = old_adj * 0.8

        # 约束
        new_adj = max(-10.0, min(15.0, new_adj))
        new_adj = round(new_adj, 1)

        if abs(new_adj - old_adj) < 0.5:
            return None

        # ---- 激进方向：只建议，不落地 ----
        if new_adj < old_adj:
            self._log_evolution(
                "threshold_suggest",
                f"准确率{accuracy:.1%} → 建议放宽阈值 {old_adj:+.1f} → {new_adj:+.1f}"
                "（更激进方向，需人工批准，未自动生效）",
            )
            return (f"💡 建议放宽推荐阈值 {old_adj:+.1f} → {new_adj:+.1f}"
                    "（更激进方向，须人工批准，未自动生效）")

        self.state["threshold_adjustment"] = new_adj
        self._log_evolution(
            "threshold",
            f"准确率{accuracy:.1%} → 阈值收紧 {old_adj:+.1f} → {new_adj:+.1f}",
        )
        return f"🎯 推荐阈值收紧 {old_adj:+.1f} → {new_adj:+.1f}"

    # ========== 市场状态适配 ==========

    def detect_regime(self, benchmark_df) -> Optional[str]:
        """
        检测当前市场状态（用基准指数，如沪深300）

        参数 benchmark_df: 含 close 列的日线 DataFrame
        返回: 'bull' / 'bear' / 'sideways'
        """
        if benchmark_df is None or len(benchmark_df) < 60:
            return None

        close = benchmark_df["close"]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        ma20_slope = close.rolling(20).mean().diff(5).iloc[-1]
        last_close = close.iloc[-1]

        if last_close > ma60 and ma20_slope > 0:
            regime = "bull"
        elif last_close < ma60 and ma20_slope < 0:
            regime = "bear"
        else:
            regime = "sideways"

        if regime != self.state.get("current_regime"):
            self._log_evolution(
                "regime",
                f"市场状态切换: {self.state.get('current_regime', '?')} → {regime}",
            )
            self.state["current_regime"] = regime
            self.state["regime_detected_at"] = datetime.now().isoformat()
            self.save()

        return regime

    # ========== 对外查询接口 ==========

    def get_model_weights(self) -> Dict[str, float]:
        """获取模型权重（单模型：恒为 {"xgboost": 1.0}）"""
        return {"xgboost": 1.0}

    def get_effective_min_score(self, user_min_score: float = None) -> float:
        """
        获取进化后的"有效最低评分"（完全自动）

        固定基线(基线 + 阈值调整(进化) + 市场状态调整)，不再读取用户滑块设定。
        用户滑块仅作为"自动阈值调整关闭/市场检测失败"时的兜底（见 app.py）。
        """
        return self.get_effective_breakdown()["effective"]

    def get_effective_breakdown(self) -> Dict:
        """
        返回有效阈值的自动拆解，供 UI 展示：
            effective = 基线(min_score_base) + 进化调整(threshold_adjustment) + 市场调整(regime_adj)
        """
        base = self.state.get("min_score_base", THRESHOLD_BASE)
        adj = self.state.get("threshold_adjustment", 0.0)
        regime = self.state.get("current_regime", "sideways")
        regime_adj = REGIME_PROFILES.get(regime, REGIME_PROFILES["sideways"])["min_score_adjust"]

        raw = base + adj + regime_adj
        effective = round(max(THRESHOLD_MIN, min(THRESHOLD_MAX, raw)), 1)
        return {
            "base": round(base, 1),
            "evolution_adj": round(adj, 1),
            "regime_adj": round(regime_adj, 1),
            "raw": round(raw, 1),
            "effective": effective,
            "clamped": round(raw, 1) != effective,
        }

    def get_summary(self) -> Dict:
        """获取进化状态摘要（供 UI 展示）"""
        perf = self.state.get("performance", {})
        t5 = perf.get("t5", {})
        t10 = perf.get("t10", {})

        def _avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0.0

        return {
            "model_weights": self.get_model_weights(),
            "current_regime": REGIME_PROFILES.get(
                self.state.get("current_regime", "sideways"),
                REGIME_PROFILES["sideways"],
            )["name"],
            "threshold_adjustment": self.state.get("threshold_adjustment", 0.0),
            "min_score_base": self.state.get("min_score_base", THRESHOLD_BASE),
            "t5_accuracy": _avg(t5.get("accuracy", [])),
            "t5_batches": t5.get("n", 0),
            "t10_accuracy": _avg(t10.get("accuracy", [])),
            "t10_batches": t10.get("n", 0),
            "last_evolved_at": self.state.get("last_evolved_at"),
            "evolution_log": self.state.get("evolution_log", [])[-10:],
            "offline_evolution": self.state.get("offline_evolution", {}),
        }

    def reset(self):
        """重置进化状态（危险操作，需用户确认）"""
        self.state = self._default_state()
        self.save()
        logger.warning("🧬 自我进化状态已重置")


# 全局单例
evolution = EvolutionEngine()

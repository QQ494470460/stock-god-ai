# -*- coding: utf-8 -*-
"""
离线进化 — 自动重训脚本

当市场风格漂移导致近期预测准确率持续偏低时，用【近期数据】重新训练模型，
让模型适应新的市场风格。

用法：
  python scripts/auto_retrain.py              # 检查 should_retrain，触发则重训
  python scripts/auto_retrain.py --force      # 无视条件，立即重训
  python scripts/auto_retrain.py --check      # 只检查，不执行

设计：
  1. 从 evolution_state.json 读取离线进化配置
  2. 判断是否满足重训条件（should_retrain）
  3. 满足则用近期数据（默认 2 年）调用 train_optuna.py 快速模式重训
  4. 成功后重置模型权重为 0.5/0.5（新模型需重新评估），更新状态
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import ROOT_DIR, MODEL_DIR, LOG_DIR
from src.models.evolution import evolution


def parse_args():
    p = argparse.ArgumentParser(description="离线进化 — 自动重训")
    p.add_argument("--force", action="store_true", help="无视条件立即重训")
    p.add_argument("--check", action="store_true", help="只检查不执行")
    p.add_argument("--lookback-years", type=int, default=0,
                   help="重训用近期数据年数（0=用配置值）")
    p.add_argument("--n-trials", type=int, default=0,
                   help="Optuna trials（0=用快速模式默认 12）")
    p.add_argument("--timeout-min", type=int, default=90,
                   help="重训子进程硬超时（分钟，默认 90）。超时即视为失败并带回日志尾部便于诊断。")
    return p.parse_args()


def get_start_date(lookback_years: int) -> str:
    """计算近期数据的起始日期"""
    if lookback_years <= 0:
        cfg = evolution.state.get("offline_evolution", {})
        lookback_years = cfg.get("lookback_years", 2)
    start = datetime.now() - timedelta(days=int(lookback_years * 365))
    return start.strftime("%Y-%m-%d")


def run_retrain(args) -> dict:
    """执行重训，调用 train_optuna.py 快速模式"""
    start_date = get_start_date(args.lookback_years)
    n_trials = args.n_trials if args.n_trials > 0 else 12
    log_file = LOG_DIR / f"auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger.info(f"🚀 离线重训启动：start_date={start_date}, trials={n_trials}")
    logger.info(f"   日志: {log_file}")

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "train_optuna.py"),
        # 与生产 serve 池保持一致（app.py 股票池默认 zz500）；max-stocks 0 = 全池。
        # 历史：曾用 hs300_zz500/300 只，与部署池不一致，实测 RankIC 不反映实盘。
        "--pool", "zz500",
        "--max-stocks", "0",
        "--start-date", start_date,
        "--n-trials", str(n_trials),
        "--cv-folds", "3",
        "--model", "all",
        "--retrain-rounds", "2",
        "--parallel",
        "--parallel-workers", "8",
        "--shap-top", "60",
        "--max-samples", "40000",
        "--max-estimators", "500",
    ]

    try:
        # 先把命令写进日志文件头部，这样即使 1 小时后超时子进程被 kill、
        # stdout 从未完整落盘，log 里也至少留下可复现的命令与参数。
        log_file.write_text(
            "CMD: " + " ".join(str(c) for c in cmd) + "\n",
            encoding="utf-8", errors="replace",
        )
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(args.timeout_min) * 60,  # 默认 90 分钟，可配
        )
        # 同时写日志文件
        log_file.write_text(result.stdout, encoding="utf-8", errors="replace")

        if "===TRAINING_COMPLETE===" in result.stdout:
            # 提取 F1
            f1 = None
            for line in result.stdout.splitlines():
                if "F1 =" in line or "F1(macro)" in line:
                    # 尝试提取数字
                    import re
                    m = re.search(r"F1(?:\(macro\))?\s*=\s*([\d.]+)", line)
                    if m:
                        try:
                            f1 = float(m.group(1))
                            break
                        except ValueError:
                            pass
            logger.info(f"✅ 离线重训成功 (F1={f1})")
            return {"success": True, "f1": f1, "log": str(log_file)}
        else:
            logger.error("❌ 离线重训失败：训练未完成")
            return {"success": False, "error": "训练未完成", "log": str(log_file)}
    except subprocess.TimeoutExpired:
        # 超时 = 子进程被 kill，stdout 没落盘。这里把 log 里已有的 CMD 尾部带上，
        # 让人一眼看到"是拿什么参数跑挂的"，而不是只看到一句「超时」。
        tail = ""
        try:
            if log_file.exists():
                tail = log_file.read_text(encoding="utf-8", errors="replace")[-500:]
        except Exception:  # noqa: BLE001
            pass
        logger.error(f"❌ 离线重训超时（>{args.timeout_min}分钟）。可 --timeout-min 调大，"
                     f"或降 --n-trials / --lookback-years 缩小规模。\n日志尾部:\n{tail}")
        return {"success": False, "error": f"超时(>{args.timeout_min}分钟)",
                "log": str(log_file), "log_tail": tail}
    except Exception as e:
        logger.error(f"❌ 离线重训异常: {e}")
        return {"success": False, "error": str(e), "log": str(log_file)}


def main():
    args = parse_args()

    # 只检查模式
    if args.check:
        should, reason = evolution.should_retrain()
        print(f"should_retrain: {should}")
        print(f"reason: {reason}")
        return

    # 判断是否触发
    should, reason = evolution.should_retrain()
    if not should and not args.force:
        logger.info(f"未触发重训：{reason}")
        print(f"SKIP: {reason}")
        return

    if should:
        logger.info(f"触发重训：{reason}")
    else:
        logger.info(f"强制重训（--force）")

    # 标记开始
    evolution.mark_retrain_started()

    # 执行重训
    result = run_retrain(args)

    if result.get("success"):
        evolution.mark_retrain_finished(f1=result.get("f1"))
        # ★ 单一模型口径：只有 xgboost，权重恒为 1.0
        # （旧逻辑会把权重重置为 xgboost/lightgbm 各 0.5，而 lightgbm 已不再训练，
        #   等于让一个 2026-07-09 的旧模型永久占一半权重 —— 正是"训练没用"的根源）
        evolution.state["model_weights"] = {"xgboost": 1.0}
        evolution.save()
        logger.info("模型权重已设为 xgboost=1.0（单一模型口径）")
    else:
        evolution.mark_retrain_failed(reason=result.get("error", "未知错误"))

    print(f"DONE: {'success' if result.get('success') else 'failed'}")
    if result.get("f1") is not None:
        print(f"F1={result['f1']}")


if __name__ == "__main__":
    main()

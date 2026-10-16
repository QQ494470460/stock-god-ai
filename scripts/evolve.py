# -*- coding: utf-8 -*-
"""自我进化 · 命令行入口。

用法
----
    # 看现状（只读）：待批准候选 + 参数覆盖 + 反馈底稿
    py -3.12 scripts/evolve.py --status

    # 跑一个进化回合（默认不自动落地、不补验）
    py -3.12 scripts/evolve.py --run

    # 跑回合并先补验到期的预测批次（会写 prediction_history.json）
    py -3.12 scripts/evolve.py --run --verify

    # 允许「安全方向」的改动自动落地（默认关闭）
    py -3.12 scripts/evolve.py --run --auto-apply

    # 指定生成器 / 上限
    py -3.12 scripts/evolve.py --run --generators param_search --max-candidates 2

    # 构建样本外面板（重，首次必做，否则闸门无法判定 → 候选一律被拒）
    py -3.12 scripts/evolve.py --build-panel --max-stocks 200 --pool hs300

    # 人工裁决
    py -3.12 scripts/evolve.py --show <候选ID>
    py -3.12 scripts/evolve.py --approve <候选ID>
    py -3.12 scripts/evolve.py --apply <候选ID>          # 落地（会改运行期参数）
    py -3.12 scripts/evolve.py --reject <候选ID> --reason "样本外不显著"
    py -3.12 scripts/evolve.py --rollback <候选ID>       # 一键回滚

    # 模型重训上线（生成器 C）
    py -3.12 scripts/evolve.py --models                  # 列出模型注册表
    py -3.12 scripts/evolve.py --pin production_metrics   # 建立基准（必须先做）
    py -3.12 scripts/evolve.py --retrain                 # 离线重训（重）
    py -3.12 scripts/evolve.py --run --generators retrain  # 生成晋升候选

    # 清空全部参数覆盖（恢复默认）
    py -3.12 scripts/evolve.py --reset-overrides

安全
----
* **默认不自动落地**；``--auto-apply`` 也只对「安全方向且标记可自动采纳」的改动生效。
* 任何触碰风控硬线（熔断线 / 单票上限 / 下单频率 / 止盈止损）的候选，
  在生成、闸门、落地三个环节各拦一次。
* 风控处于熔断状态时，``--run`` 直接返回，不改任何东西。
* ``--apply`` / ``--rollback`` / ``--pin`` 是会改运行期参数的动作，都会留痕。
* ``--retrain`` 只训练、不上线；上线必须经 ``--run --generators retrain``
  生成候选 → 闸门 → ``--approve`` → ``--apply``。
* ``ml.active_version`` 永不自动落地（``auto_adopt=False``）：
  换模型改变所有后续选股，必须人工批准。

退出码：0 = 正常；1 = 参数/执行错误；2 = 前置条件不满足。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须先导入 config.settings：它会注入外挂 libs 目录（打包版需要）
try:  # noqa: SIM105
    import config.settings  # noqa: F401
except Exception:  # noqa: BLE001
    pass


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="自我进化回路（反馈 → 候选 → 样本外闸门 → 待批准 → 落地/回滚）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="查看现状（默认动作）")
    ap.add_argument("--run", action="store_true", help="跑一个进化回合")
    ap.add_argument("--verify", action="store_true",
                    help="回合前先补验到期预测批次（会写 prediction_history.json）")
    ap.add_argument("--auto-apply", action="store_true",
                    help="允许安全方向的候选自动落地（默认关闭）")
    ap.add_argument("--generators", default="",
                    help="逗号分隔的生成器名（默认全部已注册）")
    ap.add_argument("--max-candidates", type=int, default=3,
                    help="单回合送闸门的候选上限（默认 3）")
    ap.add_argument("--show", metavar="ID", help="查看某条候选详情")
    ap.add_argument("--approve", metavar="ID", help="人工批准某条候选")
    ap.add_argument("--apply", metavar="ID", help="落地某条候选（会改运行期参数）")
    ap.add_argument("--reject", metavar="ID", help="拒绝某条候选")
    ap.add_argument("--rollback", metavar="ID", help="回滚某条已落地候选")
    ap.add_argument("--reason", default="", help="批准/拒绝时备注原因")
    ap.add_argument("--operator", default="cli", help="操作人（留痕用）")
    ap.add_argument("--build-panel", action="store_true",
                    help="构建样本外面板（重，首次必做）")
    ap.add_argument("--max-stocks", type=int, default=200, help="面板股票数上限")
    ap.add_argument("--pool", default="hs300", help="股票池（hs300/zz500/all…）")
    ap.add_argument("--reset-overrides", action="store_true",
                    help="清空全部参数覆盖（恢复默认值）")
    ap.add_argument("--models", action="store_true",
                    help="列出模型注册表（含「已钉住」标记）")
    ap.add_argument("--pin", metavar="KEY",
                    help="把基准模型钉在 KEY（建立/移动基准）。"
                         "重训晋升必须先有基准，否则无从谈「更优」")
    ap.add_argument("--unpin", action="store_true",
                    help="取消钉住基准，回到「按样本外 RankIC 自动挑选」")
    ap.add_argument("--retrain", action="store_true",
                    help="先跑离线重训（重，数十分钟）。与 --run 同用则重训后接着跑回合")
    ap.add_argument("--lookback-years", type=int, default=0,
                    help="重训用近期数据年数（0=用配置值）")
    ap.add_argument("--n-trials", type=int, default=0,
                    help="重训 Optuna trials（0=快速模式默认 20）")
    ap.add_argument("--space", action="store_true", help="打印可进化参数空间")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出（供脚本消费）")

    # ---- 模型版本的人工批准（封堵「丢个文件就静默生效」）----
    ap.add_argument("--approve-model", metavar="KEY",
                    help="把某个模型版本加入人工批准名单（否则新产品一律不生效，"
                         "系统固定加载生产默认 models/xgboost_model.pkl）")
    ap.add_argument("--revoke-model", metavar="KEY",
                    help="从人工批准名单里撤销某个模型版本")

    # ---- 前置条件自检 ----
    ap.add_argument("--doctor", action="store_true",
                    help="一次性检查全部前置条件（面板/基准/闭环结算/LLM/可部署模型），"
                         "逐条给出「已就绪」或「该跑什么命令」")

    # ---- 生成器参数透传（当前只有 D 需要）----
    ap.add_argument("--ga-population", type=int, default=None,
                    help="遗传算法种群规模（默认 10）")
    ap.add_argument("--ga-generations", type=int, default=None,
                    help="遗传算法迭代代数（默认 4）")
    ap.add_argument("--ga-elite", type=int, default=None,
                    help="每代保留的精英数（默认 2）")
    ap.add_argument("--ga-seed", type=int, default=None,
                    help="遗传算法随机种子（默认由面板区间派生，保证可复现）")
    return ap


def _print_models() -> int:
    """列出模型注册表，标出钉住 / 已批准 / 集成腿，并给出当前裁决。"""
    from src.models import model_registry as mr

    pinned = mr.pinned_version()
    approvals = mr.load_approvals()
    infos = mr.scan_models()
    print("=" * 72)
    print("模型注册表（按样本外 RankIC 降序）")
    print("=" * 72)
    if pinned:
        print(f"当前钉住基准：{pinned}")
    else:
        print("当前**未钉住**基准。两个后果，都请知悉：")
        print("  1) 生成器 C（模型重训晋升）无法工作 —— 没有基准就无从谈「更优」。")
        print("  2) 注册表排序只用于展示。真正生效需要人工批准（--approve-model）；")
        print("     未批准时一律回退到生产默认 models/xgboost_model.pkl。")
        print("     建议：scripts/evolve.py --pin <模型键>")
    print()
    if not infos:
        print("（未找到任何带指标的模型产物）")
        return 0
    for m in infos:
        def _n(v):
            try:
                f = float(v)
                return "n/a" if f != f else f"{f:+.4f}"
            except (TypeError, ValueError):
                return "n/a"
        if m.key == pinned:
            mark = "📌已钉住"
        elif m.loadable:
            mark = "✅可作主模型"
        elif m.role == "ensemble_leg":
            mark = "🧩集成腿"
        else:
            mark = "⚠️不可部署"
        if m.key in approvals:
            mark += "｜已批准"
        print(f"  {mark:<16} {m.key:<28} RankIC {_n(m.rank_ic):>9} "
              f"IC_IR {_n(m.ic_ir):>7} n={m.n_samples}")
        if m.note:
            print(f"            {m.note}")
    print()
    print(f"🧭 裁决：{mr.active_status()}")
    if approvals:
        print(f"   已批准名单：{'、'.join(sorted(approvals))}")
    print()
    print("⚠️ 加载路径目前硬编码为 models/xgboost_model.pkl —— "
          "上面的排序与裁决只决定「该不该换」，")
    print("   真正接通版本切换还需要产物按版本命名 + 加载器读裁决（尚未实现）。")
    return 0


def _approve_model(key: str, operator: str, note: str) -> int:
    """把某个模型版本加入人工批准名单（解除「未批准一律不生效」）。"""
    from src.models import model_registry as mr

    key = (key or "").strip()
    if not key:
        print("❌ 必须给出模型键", file=sys.stderr)
        return 2
    loadable = {m.key: m for m in mr.scan_models() if m.loadable}
    if key not in loadable:
        print(f"❌ 模型键 {key!r} 不能作为主模型加载（未登记 / 不是主模型候选）。"
              f"可选：{'、'.join(sorted(loadable)) or '（无）'}", file=sys.stderr)
        return 2
    mr.approve_version(key, operator=operator, note=note)
    print(f"✅ 已批准模型版本 {key!r}（{operator}）")
    print(f"   {mr.active_status()}")
    return 0


def _revoke_model(key: str) -> int:
    """从人工批准名单里撤销某个版本。"""
    from src.models import model_registry as mr

    key = (key or "").strip()
    if not key:
        print("❌ 必须给出模型键", file=sys.stderr)
        return 2
    if not mr.revoke_version(key):
        print(f"⚠️ 名单里本来就没有 {key!r}", file=sys.stderr)
        return 0
    print(f"🗑️ 已撤销 {key!r}")
    print(f"   {mr.active_status()}")
    return 0


def _pin_model(key: str, operator: str) -> int:
    """钉住 / 取消钉住基准模型（写覆盖层，留痕）。"""
    from src.evolution import overrides, space
    from src.evolution.types import Change

    key = (key or "").strip()
    if key:
        from src.models import model_registry as mr

        loadable = {m.key: m for m in mr.scan_models() if m.loadable}
        if key not in loadable:
            print(f"❌ 模型键 {key!r} 不可部署或不存在。可选："
                  f"{'、'.join(sorted(loadable)) or '（无）'}", file=sys.stderr)
            return 2

    old = space.get_default("ml.active_version")
    try:
        from src.evolution.overrides import read_effective

        old = read_effective("ml.active_version")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 读取当前钉住值失败（{e}），按默认值处理")

    try:
        applied = overrides.apply_changes(
            [Change(key="ml.active_version", old=old, new=key)])
    except space.ParamError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    if not applied:
        print(f"ℹ️ 基准已是 {key or '（未钉住）'}，无需改动")
        return 0
    if key:
        print(f"📌 已把基准模型钉在 {key}（操作人 {operator}）")
        print("   注册表今后优先加载它；即便出现分数更高的新产物，"
              "也不会静默顶掉它。")
        print("   取消钉住：scripts/evolve.py --unpin")
    else:
        print(f"📌 已取消钉住（原 {old}）→ 回到按样本外 RankIC 自动挑选")
    print(f"   落盘：config/evolution_overrides.json（ml.active_version）")
    return 0


def _run_retrain(args) -> int:
    """跑离线重训（复用 scripts/auto_retrain.py 的既有流程）。"""
    from types import SimpleNamespace

    print("🚀 开始离线重训（数十分钟，日志见 logs/auto_retrain_*.log）")
    try:
        import scripts.auto_retrain as ar
    except Exception as e:  # noqa: BLE001
        print(f"❌ 无法加载重训模块：{e}", file=sys.stderr)
        return 1

    ns = SimpleNamespace(force=True, check=False,
                         lookback_years=args.lookback_years,
                         n_trials=args.n_trials)
    try:
        result = ar.run_retrain(ns) or {}
    except Exception as e:  # noqa: BLE001
        print(f"❌ 重训执行失败：{e}", file=sys.stderr)
        return 1

    ok = result.get("success", result.get("ok"))
    print(f"{'✅ 重训完成' if ok else '⚠️ 重训结束（见返回结果）'}：{result}")
    print("   新产物已写入 models/。要让它生效：")
    print("   1) scripts/evolve.py --models            # 找到新产物键")
    print("   2) scripts/evolve.py --run --generators retrain   # 生成晋升候选")
    print("   3) scripts/evolve.py --approve <ID> && --apply <ID>")
    return 0 if ok else 1


def _generator_kwargs(args) -> Dict[str, Dict]:
    """把 CLI 上的 ``--ga-*`` 收敛成「按生成器名分组」的透传参数。

    只收集**显式给过值**的参数（缺省 None 的一律不传），这样生成器的默认值
    始终由生成器自己说了算 —— CLI 只表达「我要偏离默认」。
    """
    ga: Dict[str, Any] = {}
    if args.ga_population is not None:
        ga["population"] = args.ga_population
    if args.ga_generations is not None:
        ga["generations"] = args.ga_generations
    if args.ga_elite is not None:
        ga["elite"] = args.ga_elite
    if args.ga_seed is not None:
        ga["seed"] = args.ga_seed
    return {"genetic": ga} if ga else {}


def _doctor() -> int:
    """前置条件自检：把「没弄好的」变成一份可执行的清单。

    全只读。每一项给出 ✅已就绪 / ⚠️可选未完成 / ❌阻塞，以及**下一步该跑什么**。
    退出码：0 = 没有阻塞项；2 = 存在阻塞项。

    为什么要这个东西：这套回路的失败模式几乎都是「静默的」——
    没有面板时闸门安静地拒掉每个候选、没有基准时生成器 C 安静地返回空、
    闭环没结算过时生成器 A 安静地不开处方。用户看到的只有「什么都没发生」。
    与其让每个失败模式各写一段文档，不如让 CLI 自己把话说清楚。
    """
    from src.evolution import feedback, panels
    from src.models import model_registry as mr

    blockers = 0
    warns = 0

    print("=" * 72)
    print("自我进化 · 前置条件自检")
    print("=" * 72)

    def ok(title: str, detail: str = "") -> None:
        print(f"✅ {title}" + (f"\n     {detail}" if detail else ""))

    def warn(title: str, todo: str) -> None:
        nonlocal warns
        warns += 1
        print(f"⚠️ {title}\n     → {todo}")

    def bad(title: str, todo: str) -> None:
        nonlocal blockers
        blockers += 1
        print(f"❌ {title}\n     → {todo}")

    # ---------------- 1 样本外面板 ----------------
    print("\n【1/6 样本外面板】闸门的评测素材")
    try:
        if panels.panel_available():
            meta = panels.panel_meta() or {}
            ok("面板已缓存",
               f"{meta.get('n_days')} 天 × {meta.get('n_codes')} 只"
               f"（{meta.get('start')} ~ {meta.get('end')}）")
        else:
            bad("没有样本外面板 —— 闸门无法判定，**所有候选都会被拒**",
                "scripts/evolve.py --build-panel --max-stocks 200 --pool hs300")
    except Exception as e:  # noqa: BLE001
        bad(f"面板状态无法判断：{e}", "检查 data/evolution/panel/")

    # ---------------- 2 模型注册表与基准 ----------------
    print("\n【2/6 模型注册表与基准】生成器 C 的前提")
    try:
        infos = mr.scan_models()
        loadable = [m for m in infos if m.loadable]
        if loadable:
            ok("可作主模型的产物",
               "、".join(f"{m.key}(RankIC {m.rank_ic:+.4f})" for m in loadable))
        else:
            bad("没有任何可作主模型的产物（主模型三件套不全）",
                "scripts/evolve.py --retrain")

        pin = mr.pinned_version()
        if pin and mr.pinned_model() is not None:
            ok(f"基准已钉住：{pin}")
        elif pin:
            bad(f"ml.active_version={pin!r} 不可部署或不存在",
                "改钉一个可部署的键：scripts/evolve.py --pin <键>，或 --unpin")
        else:
            warn("未钉住基准 —— 生成器 C 不工作（没有基准就无从谈「更优」）",
                 "scripts/evolve.py --models 挑一个可部署键，再 --pin <键>")
        print(f"     🧭 裁决：{mr.active_status()}")
        approved = mr.approved_keys()
        if approved:
            ok("人工批准名单", "、".join(approved))
        else:
            print("     ℹ️ 人工批准名单为空 → 新产物一律不生效，固定加载生产默认。"
                  "这是刻意的保守默认。")
    except Exception as e:  # noqa: BLE001
        bad(f"模型注册表不可用：{e}", "检查 src/models/model_registry.py")

    # ---------------- 3 预测验证 ----------------
    print("\n【3/6 预测验证】反馈底稿的一半")
    try:
        pf = feedback.collect_prediction_feedback()
        n_ver = pf.get("n_verified", 0)
        n_pending = pf.get("pending_verify", 0)
        if n_ver:
            ok(f"已验证 {n_ver} 个（批次×窗口）")
        else:
            warn("一条验证结果都没有 —— 生成器 C 的漂移检测无从判断",
                 "scripts/evolve.py --run --verify（会写 prediction_history.json）")
        if n_pending:
            warn(f"另有 {n_pending} 个已走满持有期但未验证",
                 "补验：scripts/evolve.py --run --verify")
        if pf.get("missing_snapshot"):
            print(f"     ℹ️ {pf['missing_snapshot']} 批缺 price_snapshot："
                  "验证会回退用复权价反推，收益可能失真。看板新版已开始落快照。")
    except Exception as e:  # noqa: BLE001
        warn(f"预测验证状态无法判断：{e}", "检查 data/prediction_history.json")

    # ---------------- 4 闭环结算 ----------------
    print("\n【4/6 闭环结算】生成器 A 的处方依据")
    try:
        d = feedback.LOOP_REPORTS_DIR
        n = len(list(d.glob("loop_report_*.json"))) if d.exists() else 0
        if n:
            ok(f"已有 {n} 个收盘结算报告", f"{d}")
        else:
            bad("闭环从未收盘结算过 —— 拿不到净值序列，"
                "收益/回撤/换手全部不可用",
                "启动闭环：交易循环需在交易时段实际运行并收盘结算"
                "（data/loop_reports/ 为空即此项未达成）")
    except Exception as e:  # noqa: BLE001
        warn(f"闭环结算状态无法判断：{e}", "检查 src/trading/loop.py")

    # ---------------- 5 LLM ----------------
    print("\n【5/6 LLM 复盘】生成器 B")
    try:
        import src.evolution.generators.llm_review as lr

        if lr._llm_ready():
            ok("LLM 已配置（Key 可读）",
               "注意：Key 有效 ≠ 余额充足。余额不足时接口返回 402，"
               "生成器会记一条警告后优雅跳过，不会中断回合")
        else:
            warn("LLM 未就绪（读不到 API Key）—— 生成器 B 恒为空",
                 "配置 config/deepseek.yaml；看板的「AI 个股分析」同源，也会一起失效")
    except Exception as e:  # noqa: BLE001
        warn(f"LLM 状态无法判断：{e}", "检查 config/deepseek.yaml")

    # ---------------- 6 状态文件 ----------------
    print("\n【6/6 生产状态文件】确认它们都在、且可写")
    from config.settings import DATA_DIR

    for p, what in [
        (Path(DATA_DIR) / "trading_guard_state.json", "风控状态"),
        (Path(DATA_DIR) / "trading_journal.jsonl", "交易审计"),
        (Path(DATA_DIR) / "evolution_state.json", "进化引擎状态"),
        (Path(DATA_DIR) / "prediction_history.json", "预测验证历史"),
    ]:
        if p.exists():
            ok(f"{what}：{p.name}", f"{p.stat().st_size} 字节")
        else:
            print(f"     ℹ️ {what}：{p.name} 尚未生成（首次运行时创建）")

    print("\n" + "=" * 72)
    if blockers:
        print(f"结论：{blockers} 项阻塞，{warns} 项待办。"
              f"阻塞项不解决，进化回合会「安静地什么都不做」。")
    elif warns:
        print(f"结论：无阻塞项，{warns} 项待办。回路可用，但效果会打折。")
    else:
        print("结论：全部就绪。")
    print("=" * 72)
    return 2 if blockers else 0


def main() -> int:
    args = _build_parser().parse_args()

    from src.evolution import gate, panels, space
    from src.evolution.runner import EvolutionRunner

    # ---- 参数空间速查（只读）----
    if args.space:
        print(space.describe_space())
        if not (args.run or args.status or args.build_panel):
            return 0

    # ---- 构建面板（重）----
    if args.build_panel:
        print(f"🏗️ 构建样本外面板：pool={args.pool} max_stocks={args.max_stocks}")
        out = panels.build_panel(max_stocks=args.max_stocks, pool=args.pool)
        if out is None:
            print("❌ 面板构建失败（详见日志）", file=sys.stderr)
            return 2
        print(f"✅ 面板已缓存：{out}")
        meta = panels.panel_meta()
        if meta:
            print(f"   {meta.get('n_days')} 天 × {meta.get('n_codes')} 只 "
                  f"（{meta.get('start')} ~ {meta.get('end')}）")
        return 0

    # ---- 模型注册表（只读）----
    if args.models:
        return _print_models()

    # ---- 模型版本人工批准 / 撤销（写 config/model_approvals.json）----
    if args.approve_model:
        return _approve_model(args.approve_model, args.operator, args.reason)
    if args.revoke_model:
        return _revoke_model(args.revoke_model)

    # ---- 前置条件自检（只读）----
    if args.doctor:
        return _doctor()

    # ---- 钉住 / 取消钉住基准模型 ----
    if args.pin:
        return _pin_model(args.pin, args.operator)
    if args.unpin:
        return _pin_model("", args.operator)

    # ---- 离线重训（重；由运维/调度显式触发）----
    if args.retrain:
        rc = _run_retrain(args)
        if rc != 0 or not (args.run or args.status):
            return rc

    # ---- 清空覆盖 ----
    if args.reset_overrides:
        from src.evolution import overrides

        cleared = overrides.reset_all()
        print(f"✅ 已清除 {len(cleared)} 项参数覆盖，恢复默认值")
        for k in cleared:
            print(f"   · {k}")
        return 0

    # provider：有面板缓存就用，没有就交给闸门判负并提示
    if not panels.panel_available():
        print("⚠️ 未找到样本外面板缓存（闸门无法判定，候选会被拒）。"
              "请先执行：scripts/evolve.py --build-panel", file=sys.stderr)
    runner = EvolutionRunner(provider=panels.provider_from_cache())

    # ---- 裁决动作 ----
    try:
        if args.show:
            c = runner.show(args.show)
            if c is None:
                print(f"❌ 找不到候选 {args.show}", file=sys.stderr)
                return 2
            print(c.describe())
            print(f"  理由：{c.rationale}")
            print(f"  风险：{runner._risk_note(c)}")
            if c.evaluation:
                print(f"  评估器：{c.evaluation.evaluator}"
                      f" | 区间：{c.evaluation.window}")
            if c.decision_reason:
                print(f"  裁决：{c.decision_reason}")
            return 0

        if args.approve:
            c = runner.approve(args.approve, by=args.operator,
                               reason=args.reason or "人工批准")
            print(f"✅ 已批准 {c.id}（状态 {c.status}）。"
                  f"落地请执行：--apply {c.id}")
            return 0

        if args.reject:
            c = runner.reject(args.reject, by=args.operator,
                              reason=args.reason)
            print(f"🚫 已拒绝 {c.id}")
            return 0

        if args.apply:
            c = runner.apply(args.apply, by=args.operator)
            print(f"✅ 已落地 {c.id}")
            for ch in c.changes:
                print(f"   · {ch.describe()}")
            return 0

        if args.rollback:
            c = runner.rollback(args.rollback, by=args.operator)
            print(f"↩️ 已回滚 {c.id}")
            for ch in c.changes:
                print(f"   · {ch.key}: {ch.new} → {ch.old}（已恢复旧值）")
            return 0
    except (KeyError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    # ---- 跑回合 ----
    if args.run:
        names = [x.strip() for x in args.generators.split(",") if x.strip()]
        rep = runner.run(generators=names or None,
                         verify=args.verify,
                         auto_apply=args.auto_apply,
                         max_candidates=args.max_candidates,
                         generator_kwargs=_generator_kwargs(args))
        if args.json:
            print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(rep.render())
        return 0 if not rep.errors else 1

    # ---- 默认：看现状 ----
    st = runner.status()
    if args.json:
        print(json.dumps(st, ensure_ascii=False, indent=2, default=str))
        return 0

    print("=" * 72)
    print("自我进化 · 现状")
    print("=" * 72)
    print("\n【反馈底稿】")
    print(st["feedback"])
    print(f"\n【候选队列】总计 {st['candidates']['total']} 条，"
          f"待批准 {st['candidates']['pending']} 条")
    for item in st["pending"]:
        print(f"  ⏳ {item['id']}  {item['title']}（{item['source']}）")
        for ch in item["changes"]:
            print(f"       · {ch}")
        print(f"       风险：{item['risk_note']}")
    if st["applied"]:
        print("\n【已落地（可回滚）】")
        for item in st["applied"]:
            print(f"  ✅ {item['id']}  {item['title']}  ({item['applied_at']})")

    ov = st["overrides"]
    print(f"\n【参数覆盖】{ov['file']}")
    if ov["overrides"]:
        for k, v in ov["overrides"].items():
            print(f"   · {k} = {v}")
        print(f"   （更新时间 {ov['updated_at']}）")
    else:
        print("   （无覆盖，全部为默认参数）")

    if st["guard_halted"]:
        print("\n⚠️ 风控处于熔断状态，进化回合将不做任何改动")

    print("\n【判定标准】")
    c = gate.DEFAULT_CRITERIA
    print(f"   夏普提升 ≥ {c.min_sharpe_gain:+.3f}，"
          f"收益提升 ≥ {c.min_return_gain:+.2%}，"
          f"回撤恶化 ≤ {c.max_drawdown_worsen:.2%}，"
          f"调仓期数 ≥ {c.min_periods}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        print(f"❌ 执行失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

# -*- coding: utf-8 -*-
"""无人值守闭环交易进程 —— 让「闭环全自动」不再绑定看板窗口。

背景
----
看板里的闭环跑在 Streamlit 的 daemon 线程上（`trading_ui._render_loop` 的
「▶️ 启动闭环」按钮），一旦关掉应用就停。本脚本提供**常驻入口**：
同样的 `TradingLoop`、同样的 `RiskSupervisor` 安全中枢，只是没有界面，
可以放进计划任务里，真正做到无人值守。

用法
----
    py -3.12 scripts/run_loop.py --preview        # 只预演「本该下什么单」，然后退出
    py -3.12 scripts/run_loop.py --once           # 只跑一个周期
    py -3.12 scripts/run_loop.py --max-ticks 5    # 跑 5 个周期后退出
    py -3.12 scripts/run_loop.py                  # 常驻（默认纸交易）
    py -3.12 scripts/run_loop.py --until-close    # 常驻，收盘结算完自动退出
    py -3.12 scripts/run_loop.py --dry-run        # 只演练：连纸单都不下，只记录「本该下」
    py -3.12 scripts/run_loop.py --live           # 实盘（必须先两步令牌武装，否则拒绝启动）

    # 换了账号 / 出入金 / 纸账户重置后，旧净值峰值会失真导致误熔断，用这个重置基线：
    py -3.12 scripts/run_loop.py --rebaseline-peak --resume

安全
----
* **默认纸交易**。`--live` 会校验风控模式：未武装 / 已熔断一律拒绝启动。
* 所有下单仍走 `guard.propose_trade → approve_and_execute`，不存在绕过风控的路径。
* `--dry-run` / `--preview` 会把风控状态文件**隔离到 data/loop_rehearsal/**，
  演练绝不写生产状态 —— 曾经演练把 halted=true 写进真实文件，把实盘冻住。
* `Ctrl+C` 优雅退出：停循环 → 打印最终状态 → 写出当日结算报告（若已到结算点）。

退出码：0 = 正常结束；2 = 前置校验失败（连接失败 / 实盘未武装）；1 = 运行中异常。
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ⚠️ 必须在使用 xtquant 之前导入 config.settings ——
# 它的模块级 _extend_sys_path_for_external_libs() 会把 <exe 同级>/libs
# 注入 sys.path，否则打包版 exe 里 import xtquant 会失败。
try:  # noqa: SIM105
    import config.settings  # noqa: F401
except Exception:  # noqa: BLE001
    pass


_stop_requested = False
# 单实例锁的文件句柄：进程存活期间一直持有，退出（含崩溃）由 OS 自动释放
_instance_lock_handle = None


def acquire_single_instance(lock_path: Path) -> bool:
    """拿「全机唯一闭环实例」的进程级文件锁，拿不到返回 False。

    为什么必须有：闭环一旦多开一个实例，就会对同一份计划**再下一遍单**。
    加了开机自启之后，最容易出现「计划任务拉了一份 + 你手动又开一份」，
    而且两边的风控/纸账户各有一份内存副本，对账也会乱。

    用 OS 级文件锁（Windows msvcrt / POSIX flock）而不是 PID 文件：
    进程无论正常退出还是崩溃，操作系统都会释放锁，不会留下"假死锁"。
    注意不能用 ``os.kill(pid, 0)`` 探活 —— Windows 上它会真的
    TerminateProcess，把正在跑的进程杀掉。
    """
    global _instance_lock_handle
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    existed = lock_path.exists()
    try:
        # 必须可写打开（POSIX 的 flock 同样要求可写句柄）。
        # ⚠️ Windows 的语义在这里很关键：只要别的进程锁住了文件里的字节区间，
        # 任何 *可写* 打开都会直接 Permission denied —— 连 open 都进不去。
        # 所以「打不开」本身就是「已有实例在跑」的确凿信号，不能当成异常放行。
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    except PermissionError:
        print(f"⚠️  {lock_path} 正被另一个实例锁定（写打开被拒）", file=sys.stderr)
        return False
    except OSError as e:
        if existed:
            print(f"⚠️  无法打开单实例锁 {lock_path}: {e}", file=sys.stderr)
            return False
        print(f"⚠️  无法创建单实例锁 {lock_path}: {e}（锁不可用，继续运行）",
              file=sys.stderr)
        return True          # 连锁文件都建不出来：宁可能跑，不可不跑
    f = os.fdopen(fd, "r+", encoding="utf-8")
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:                # pragma: no cover - POSIX
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return False

    try:
        f.seek(0)
        f.truncate()
        f.write(f"pid={os.getpid()}\nsince={datetime.now().isoformat(timespec='seconds')}\n")
        f.flush()
    except OSError:
        pass
    _instance_lock_handle = f
    return True


def _on_signal(signum, _frame):  # pragma: no cover - 信号路径
    global _stop_requested
    _stop_requested = True
    print(f"\n⏹️  收到信号 {signum}，正在优雅退出…", flush=True)


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def build_loop(args):
    """装配 broker / guard / TradingLoop（与看板同一套接线）。"""
    from config.settings import CONFIG_DIR, DATA_DIR
    from src.trading.guard import GuardConfig, RiskSupervisor, build_supervisor
    from src.trading.loop import LoopConfig, TradingLoop
    from src.trading.paper import paper_account
    from src.trading.yuekai import YuekaiBroker

    # ---- 券商连接 ----
    broker = YuekaiBroker()
    qmt_path = account_id = password = ""
    if args.live or args.account:
        try:
            from src.trading.account import account_manager
            acct = None
            for a in account_manager.accounts:
                if not args.account or a.account_id == args.account:
                    acct = a
                    break
            if acct is None:
                print(f"❌ 找不到账号 {args.account!r}，请先在界面「添加账号」里保存",
                      file=sys.stderr)
                return None, None, None
            qmt_path = acct.qmt_path or ""
            account_id = acct.qmt_account or ""
            password = account_manager.get_decrypted_password(acct.account_id) or ""
        except Exception as e:  # noqa: BLE001
            print(f"❌ 读取账号失败：{e}", file=sys.stderr)
            return None, None, None

    mode = "live" if args.live else "sim"
    if not broker.connect(qmt_path=qmt_path, account_id=account_id,
                          password=password, mode=mode):
        print(f"❌ 连接券商失败：{broker.last_error or '请检查 QMT 客户端是否已启动并登录'}",
              file=sys.stderr)
        return None, None, None

    kind = "模拟盘" if broker.is_simulated else "实盘"
    print(f"✅ 已连接 {broker.get_broker_name()} · {kind}"
          + (f"（{broker.sim_reason}）" if broker.is_simulated and broker.sim_reason else ""))

    # ---- 安全中枢（绝不裸奔） ----
    cfg_path = Path(CONFIG_DIR) / "trading_guard.yaml"
    gcfg = GuardConfig.from_yaml(str(cfg_path)) if cfg_path.exists() else GuardConfig()

    # 演练/预演 → 隔离风控持久化文件。
    # 教训：演练曾把 halted=true 写进真实 data/trading_guard_state.json，
    # 把用户的实盘风控冻住。演练就该零副作用，绝不碰生产状态。
    rehearsal = bool(args.dry_run or args.preview)
    if rehearsal:
        scratch = Path(DATA_DIR) / "loop_rehearsal"
        scratch.mkdir(parents=True, exist_ok=True)
        gcfg.state_file = str(scratch / "trading_guard_state.json")
        gcfg.journal_file = str(scratch / "trading_journal.jsonl")
        print(f"🧪 演练模式：风控状态隔离在 {scratch}（不触碰生产状态文件）")

    guard = RiskSupervisor(broker, gcfg, paper_account=paper_account)
    st = guard.safety
    print(f"🛡️  安全中枢：模式={st.mode.value} 熔断={st.halted}")
    if not rehearsal:
        snap = st.snapshot()
        print(f"    净值基线：峰值 ¥{snap.get('peak_equity', 0):,.0f}"
              f" · 当日起始 ¥{snap.get('today_start_equity', 0):,.0f}"
              f" · 账号 {snap.get('baseline_account') or '未记录'}")
        # 基线明显失真时给出提示（不自动放宽熔断，交由操作者决定）
        acct = guard.account()
        if acct and snap.get("peak_equity", 0) > 0:
            dd = st.current_drawdown_pct(acct.total_asset)
            if dd >= float(gcfg.max_portfolio_drawdown_pct):
                print(f"    ⚠️  当前权益 ¥{acct.total_asset:,.0f} 相对峰值已回撤 {dd:.2f}%，"
                      f"已达熔断线 —— 若这是「换了账号/出入金/纸账户重置」导致的，"
                      f"请用 --rebaseline-peak 重置基线，而不是直接改 JSON。")

    # ---- 显式重置净值基线 ----
    if args.rebaseline_peak:
        acct = guard.account()
        if not acct:
            print("❌ 拿不到账户权益，无法重置基线", file=sys.stderr)
            broker.disconnect()
            return None, None, None
        st.rebaseline_peak(acct.total_asset, operator=args.operator or "cli",
                           reason=args.rebaseline_reason or "手动执行 --rebaseline-peak",
                           account_id=str(getattr(acct, "account_id", "") or ""))
        if st.halted and args.resume:
            st.resume(operator=args.operator or "cli")
            print("🟢 已同时解除熔断")
        print(f"✅ 净值基线已重置为 ¥{acct.total_asset:,.0f}")

    # ---- 实盘二次闸门：本脚本自己再拦一道 ----
    if args.live:
        if not st.is_live():
            print(f"❌ 拒绝启动实盘：风控模式为 {st.mode.value}，未武装。\n"
                  "   请先在界面「交易安全」里两步武装（arm_live → confirm_arm），"
                  "或改用纸交易运行。", file=sys.stderr)
            broker.disconnect()
            return None, None, None
        if st.halted:
            print("❌ 拒绝启动实盘：风控已熔断（halted=True）。\n"
                  "   先排查熔断原因；若属基线失真，用 --rebaseline-peak --resume 处理。",
                  file=sys.stderr)
            broker.disconnect()
            return None, None, None
        print("⚠️  **实盘模式**：所有下单将使用真实资金，请确认风控参数符合预期。")

    # ---- 闭环引擎 ----
    cfg = LoopConfig.from_app_config()
    if args.interval:
        cfg.scan_interval_sec = float(args.interval)
    if args.dry_run:
        cfg.dry_run = True
    if args.no_notify:
        cfg.notify = False

    loop = TradingLoop(broker, guard=guard, paper_account=paper_account, config=cfg)
    return loop, broker, cfg


def _print_preview(loop) -> None:
    """打印预演结果（对应 loop.preview() 的真实字段）。"""
    pv = loop.preview()
    print("\n🔍 预演结果（未下任何单）")
    print("-" * 64)
    print(f"  风控模式    {pv.get('guard_mode', '?')}"
          + ("（演练，不下单）" if pv.get("dry_run") else ""))
    # plan_summary 是多行文本，压成一行避免打乱排版
    _ps = " ".join(str(pv.get("plan_summary", "（无计划）")).split())
    print(f"  交易计划    {_ps}")
    print(f"  账户权益    ¥{float(pv.get('equity') or 0):,.0f}"
          f"    可用现金 ¥{float(pv.get('cash') or 0):,.0f}")

    pos = pv.get("positions") or []
    if pos:
        print(f"\n  当前持仓 {len(pos)} 只：")
        for p in pos:
            print(f"    {p.get('code','?'):8} {str(p.get('name',''))[:6]:<8}"
                  f" {p.get('quantity', 0):>7} 股   盈亏 {p.get('pnl_pct', 0):+.2f}%")

    stops = pv.get("stops") or []
    if stops:
        print(f"\n  触发止盈止损 {len(stops)} 只：")
        for s in stops:
            if isinstance(s, dict):
                print(f"    {s.get('code','?')} {s.get('reason') or s.get('action','')}")
            else:
                print(f"    {s}")

    intents = pv.get("intents") or []
    if intents:
        print(f"\n  本该下 {len(intents)} 单：")
        for it in intents:
            print(f"    {str(it.get('side','?')):4} {str(it.get('code','?')):8}"
                  f" {int(it.get('quantity') or 0):>7} 股"
                  f" @ {float(it.get('price') or 0):>9.2f}"
                  f"  ≈ ¥{float(it.get('amount') or 0):>12,.0f}")
        print(f"    {pv.get('intent_summary', '')}")
    else:
        print("\n  无需调仓（当前持仓已在容忍带内）")
    print("-" * 64)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="无人值守闭环交易进程（默认纸交易）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--preview", action="store_true", help="只预演本该下什么单，然后退出")
    ap.add_argument("--once", action="store_true", help="只跑一个周期")
    ap.add_argument("--max-ticks", type=int, default=0, help="跑够 N 个周期后退出（0=不限）")
    ap.add_argument("--until-close", action="store_true", help="收盘结算完成后自动退出")
    ap.add_argument("--interval", type=float, default=0.0, help="tick 间隔秒数（默认取配置）")
    ap.add_argument("--dry-run", action="store_true", help="只演练：连纸单都不下")
    ap.add_argument("--live", action="store_true", help="实盘（需风控已两步武装，否则拒绝启动）")
    ap.add_argument("--account", default="", help="指定资金账号（默认取第一个已存账号）")
    ap.add_argument("--no-notify", action="store_true", help="关闭事件/日报推送")
    ap.add_argument("--rebaseline-peak", action="store_true",
                    help="把净值基线（峰值+当日起始）重置为当前权益，然后继续。"
                         "用于「换了账号 / 出入金 / 纸账户重置」导致旧峰值失真时")
    ap.add_argument("--rebaseline-reason", default="", help="重置基线的原因（写入 journal）")
    ap.add_argument("--resume", action="store_true",
                    help="配合 --rebaseline-peak：重置基线后顺便解除熔断")
    ap.add_argument("--operator", default="cli", help="操作者标识（写入 journal，默认 cli）")
    args = ap.parse_args()

    # ---- 单实例闸门（必须在建 broker/guard 之前，尽早失败）----
    # 两个闭环实例会对同一份计划重复下单，且各自的纸账户/风控是独立内存副本，
    # 对账一定会乱。--preview 只读，不受限制。
    if not args.preview:
        from config.settings import DATA_DIR as _DATA_DIR
        if not acquire_single_instance(Path(_DATA_DIR) / ".run_loop.lock"):
            print("❌ 已有一个闭环实例在运行（单实例锁被占用），本次退出。\n"
                  "   两个实例会对同一份计划重复下单。\n"
                  "   确认确实没有实例在跑后，删除 data/.run_loop.lock 再重试；\n"
                  "   想看状态请读 data/loop_heartbeat.json 的 age_sec。",
                  file=sys.stderr)
            return 3

    loop, broker, cfg = build_loop(args)
    if loop is None:
        return 2

    signal.signal(signal.SIGINT, _on_signal)
    try:
        signal.signal(signal.SIGTERM, _on_signal)
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    # ---- 无头模式也拉起「收盘后自动化调度」（与看板同一套接线，幂等）----
    # 之前调度只在看板首次被打开（浏览器连接）时才启动，导致无人值守的 run_loop
    # 进程里 18:30 数据更新 / 18:45 选股 / 19:00 进化 三个 cron 永远不触发。
    # 现在常驻进程里也自动拉起；仍受 config.scheduler.enabled 控制。
    # 演练 / 预演（--dry-run / --preview）不启动，保持零副作用（与风控隔离纪律一致）。
    if not (args.dry_run or args.preview):
        try:
            from src.scheduler.auto_scheduler import start_scheduler, is_running
            if not is_running():
                sched = start_scheduler()
                if sched is not None:
                    print("✅ 无头调度已拉起（数据更新 18:30 / 选股 18:45 / 进化 19:00，受 scheduler.enabled 控制）")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 无头调度自动拉起失败（已忽略，闭环继续）: {e}", file=sys.stderr)

    try:
        if args.preview:
            _print_preview(loop)
            return 0

        interval = float(args.interval or cfg.scan_interval_sec)
        limit = 1 if args.once else max(0, args.max_ticks)
        print(f"\n🚀 闭环启动｜模式={'演练' if cfg.dry_run else ('实盘' if args.live else '纸交易')}"
              f"｜间隔={interval:.0f}s"
              + (f"｜周期上限={limit}" if limit else "｜常驻")
              + ("｜收盘即退" if args.until_close else ""))
        print("   Ctrl+C 可随时优雅退出\n")

        n = 0
        exit_code = 0
        while not _stop_requested:
            try:
                rep = loop.tick()
                n += 1
                print(f"[{_stamp()}] {rep.state:>8} · {rep.market_status}"
                      f" · 计划 {rep.plan_date or '无'}"
                      f" · 持仓 {rep.n_positions} · 权益 ¥{rep.equity:,.0f}"
                      f" · 意图 {rep.n_intents}/下单 {rep.n_executed}/拦截 {rep.n_blocked}"
                      f" · 止损 {rep.n_stops}", flush=True)
                if rep.alerts:
                    for a in rep.alerts:
                        print(f"           ⚠️ {a}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{_stamp()}] ❌ 周期异常：{type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                exit_code = 1

            if limit and n >= limit:
                print(f"\n✅ 已达周期上限（{n}），退出")
                break

            if args.until_close:
                st = loop.get_status()
                if st.get("settled_today"):
                    print("\n✅ 今日已结算，退出")
                    break

            # 可中断的等待
            slept = 0.0
            while slept < interval and not _stop_requested:
                time.sleep(min(0.5, interval - slept))
                slept += 0.5
    finally:
        try:
            loop.stop(timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            st = loop.get_status()
            last = st.get("last_report") or {}
            if last:
                print(f"\n最后状态：{last.get('state')} · {last.get('market_status')}"
                      f" · 权益 ¥{float(last.get('equity') or 0):,.0f}")
        except Exception:  # noqa: BLE001
            pass
        try:
            broker.disconnect()
        except Exception:  # noqa: BLE001
            pass
        print("👋 已退出")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

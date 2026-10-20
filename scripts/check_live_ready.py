# -*- coding: utf-8 -*-
"""
实盘就绪自检 —— 一键告诉你「还差哪几步才能开实盘」
==================================================

依次检查 7 项实盘前置条件，输出 ✅/⚠️/❌ 清单与下一步操作：

  1. Python 位数          miniQMT 的 xtquant 只支持 64 位
  2. xtquant 是否可导入   （实盘行情+交易的唯一依赖）
  3. QMT 客户端是否在运行  （进程探测，尽力而为）
  4. 账号是否已配置        （broker_accounts.enc，落在外挂 CONFIG_DIR）
  5. QMT userdata_mini 路径是否填写且真实存在（可用 --autofill 自动探测回填）
  6. 交易密码是否可解密    （加密存储，能取到才算「可登录」）
  7. 风控中枢当前模式      （默认 PAPER；进实盘必须两步令牌）

用法
----
    py -3.12 scripts/check_live_ready.py
    py -3.12 scripts/check_live_ready.py --account 260000125091
    py -3.12 scripts/check_live_ready.py --autofill   # 自动探测 QMT 路径并回填账号

退出码：0 = 实盘就绪；1 = 仍有阻断项（❌）。
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ⚠️ 必须在任何 `import xtquant` 之前导入 config.settings ——
# 它的模块级 _extend_sys_path_for_external_libs() 会把 <exe 同级>/libs
# 注入 sys.path，从而让打包版 exe 能外挂加载 xtquant。
# （曾踩坑：xtquant 检查排在第 2 项、config.settings 首次导入在第 4 项之后，
#   导致 exe 里永远报「xtquant 未安装」，即便 libs/ 已就位。）
try:  # noqa: SIM105
    import config.settings  # noqa: F401
except Exception:  # noqa: BLE001
    pass

OK = "✅"
WARN = "⚠️ "
BAD = "❌"
INFO = "ℹ️ "

_results: list[tuple[str, str, str, str]] = []  # (level, title, detail, action)
# 解释器不匹配时置 True：此时下面的 ❌ 都可能是假象，结论必须标注「不可信」
_wrong_interpreter = False


def add(level: str, title: str, detail: str = "", action: str = "") -> None:
    _results.append((level, title, detail, action))


# ---------------------------------------------------------------- 0. 解释器
def check_interpreter() -> None:
    """第 0 项：先确认「当前解释器就是平时跑项目的那个」。

    曾踩坑（2026-10-20）：用没装项目依赖的 Python 跑本脚本（如沙箱自带 3.13），
    loguru/pandas 全部导入失败 → 连带刷出「xtquant 未安装 / 账号模块加载失败 /
    风控中枢状态读取失败」一串 ❌，看起来像「实盘缺一大堆东西」，
    实际只是**解释器选错了**，一度还被当成结论记进了项目记忆。

    这里先探两个项目核心依赖（loguru + pandas，正确环境必有、裸解释器必无），
    不匹配就明确告警，并让最终结论标注「不可信」，避免照着假清单白折腾。
    """
    global _wrong_interpreter
    missing = []
    for mod in ("loguru", "pandas"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(mod)
    if not missing:
        return
    _wrong_interpreter = True
    add(BAD, "当前 Python 解释器缺少项目依赖 —— 本清单结果不可信",
        f"缺少 {'、'.join(missing)}；解释器 {sys.executable}",
        "换用装了项目依赖的解释器重跑（本机为系统 Python 3.12）："
        "py -3.12 scripts/check_live_ready.py")


# ---------------------------------------------------------------- 1. 位数
def check_arch() -> None:
    bits = 64 if sys.maxsize > 2**32 else 32
    if bits == 64:
        add(OK, f"Python 位数：{bits} 位", f"解释器 {sys.executable}")
    else:
        add(BAD, f"Python 位数：{bits} 位（32 位）",
            "xtquant 的 miniQMT 接口只提供 64 位版本",
            "换用 64 位 Python 运行本系统")


# ---------------------------------------------------------------- 2. xtquant
def check_xtquant() -> None:
    try:
        import xtquant  # noqa: F401
        ver = getattr(xtquant, "__version__", "")
        add(OK, "xtquant 已安装", f"{xtquant.__file__} {ver}".strip())
    except Exception as e:  # noqa: BLE001
        if getattr(sys, "frozen", False):
            base = Path(sys.executable).parent
            hint = (f"把 xtquant 文件夹放到 {base / 'libs'}\\xtquant 下"
                    "（打包版走外挂加载，系统会自动注入 sys.path）")
        else:
            hint = ("py -3.12 -m pip install xtquant"
                    "（或从 QMT 安装目录复制 xtquant 到 site-packages）")
        add(BAD, "xtquant 未安装（实盘阻断项）",
            f"{type(e).__name__}: {e}", hint)


# ---------------------------------------------------------------- 3. QMT 进程
def check_qmt_process() -> None:
    """尽力探测 QMT 客户端进程（不依赖 psutil，避免额外依赖）。"""
    found: list[str] = []
    try:
        import subprocess
        # Windows：tasklist 过滤可能的 QMT 进程名
        out = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        keywords = ("qmt", "xtmini", "xtquant", "极速", "国金", "粤开", "策略交易")
        for line in out.splitlines():
            low = line.lower()
            if any(k.lower() in low for k in keywords):
                name = line.split(",")[0].strip('"')
                if name and name not in found:
                    found.append(name)
    except Exception as e:  # noqa: BLE001
        add(WARN, "QMT 客户端进程探测失败", f"{type(e).__name__}: {e}",
            "可忽略，手动确认 QMT「极简模式」已启动并登录即可")
        return

    if found:
        add(OK, "检测到疑似 QMT 客户端进程", "、".join(found[:5]))
    else:
        add(WARN, "未检测到 QMT 客户端进程",
            "xtquant 连接的是本机已登录的 miniQMT 客户端",
            "先启动 QMT → 选择「极简模式」→ 登录交易账号，再点界面「连接」")


# ---------------------------------------------------------------- 4-6. 账号
def check_accounts(target: str | None, autofill: bool = False) -> None:
    try:
        from src.trading.account import account_manager
    except Exception as e:  # noqa: BLE001
        add(BAD, "账号模块加载失败", f"{type(e).__name__}: {e}")
        return

    try:
        from config.settings import CONFIG_DIR
        acct_file = Path(CONFIG_DIR) / "broker_accounts.enc"
    except Exception:  # noqa: BLE001
        acct_file = Path("?")

    accts = account_manager.accounts
    if not accts:
        add(BAD, "未保存任何券商账号", f"账号文件：{acct_file}",
            "在「自动交易 → 添加账号」里填资金账号/QMT 路径/密码并保存")
        return
    add(OK, f"已保存账号 {len(accts)} 个", f"账号文件：{acct_file}")

    chosen = None
    if target:
        chosen = next((a for a in accts if a.account_id == target), None)
        if chosen is None:
            add(WARN, f"未找到指定账号 {target}", "回退检查第一个账号")
    if chosen is None:
        chosen = next((a for a in accts if a.active), accts[0])

    # 4. 账号字段
    add(INFO, f"检查账号：{chosen.name} ({chosen.account_id})",
        f"券商={chosen.broker} · QMT登录账号={chosen.qmt_account or '(空)'}")

    # 5. QMT 路径
    qp = (chosen.qmt_path or "").strip()
    if not qp and autofill:
        # 自动探测并回填
        add(INFO, "QMT 路径为空 → 启动自动探测", "在本机常见位置扫描 userdata_mini（约 30 秒）")
        try:
            from src.trading.qmt_env import detect_qmt_userdata_mini
            found, why = detect_qmt_userdata_mini()
        except Exception as e:  # noqa: BLE001
            found, why = None, f"探测模块加载失败：{type(e).__name__}: {e}"
        if found:
            try:
                account_manager.add_or_update(
                    broker=chosen.broker, account_id=chosen.account_id,
                    qmt_path=found, set_active=chosen.active,
                )
                qp = found
                add(OK, f"已自动探测并回填 QMT 路径：{found}", why)
            except Exception as e:  # noqa: BLE001
                add(WARN, f"探测到路径 {found} 但回填失败", f"{type(e).__name__}: {e}")
                qp = ""
        else:
            add(WARN, "自动探测未找到 QMT",
                why,
                "确认 QMT 是否已安装；或手动在界面填写 userdata_mini 路径")

    if not qp:
        add(BAD, "QMT userdata_mini 路径未填写",
            "实盘连接必须要它（例如 D:\\QMT\\userdata_mini）",
            "优先试：py -3.12 scripts/check_live_ready.py --autofill ；"
            "或界面「券商账号登录 → ✏️ 设置/修改该账号的 QMT 路径」填写")
    elif not Path(qp).exists():
        add(BAD, f"QMT 路径不存在：{qp}",
            "路径写错，或 QMT 未安装在该位置",
            "确认 QMT「极简模式」安装目录下的 userdata_mini 实际路径后重新填写")
    else:
        add(OK, f"QMT 路径有效：{qp}")

    # 6. 密码可解密
    pwd = account_manager.get_decrypted_password(chosen.account_id)
    if pwd:
        add(OK, "交易密码可解密", "（加密存储，长度 %d）" % len(pwd))
    else:
        add(BAD, "交易密码为空或无法解密",
            "实盘登录需要交易密码",
            "在该账号处重新填写交易密码并保存（加密后落盘）")


# ---------------------------------------------------------------- 7. 风控模式
def check_guard_mode() -> None:
    try:
        from src.trading.guard import GuardConfig, SafetyController
        from config.settings import CONFIG_DIR
        gp = Path(CONFIG_DIR) / "trading_guard.yaml"
        cfg = GuardConfig.from_yaml(str(gp)) if gp.exists() else GuardConfig()
        safety = SafetyController(cfg)
        mode = safety.mode.value
        halted = safety.halted
        detail = f"配置文件：{gp if gp.exists() else '(缺失→内置保守默认)'} · 熔断={halted}"
        if halted:
            add(WARN, f"风控中枢当前：{mode}（已熔断 halted）", detail,
                "如需恢复交易，在安全面板显式 resume（解除熔断）")
        elif mode in ("live", "live_pending"):
            add(OK, f"风控中枢当前：{mode}", detail)
        else:
            add(INFO, f"风控中枢当前：{mode}", detail,
                "实盘前需两步令牌武装：arm_live() → confirm_arm(token)，"
                "首笔建议留在 LIVE_PENDING（逐单人工确认）")
    except Exception as e:  # noqa: BLE001
        add(WARN, "风控中枢状态读取失败", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 汇总
def main() -> int:
    ap = argparse.ArgumentParser(description="实盘就绪自检")
    ap.add_argument("--account", default=None, help="指定检查的资金账号（默认取当前激活账号）")
    ap.add_argument("--autofill", action="store_true",
                    help="QMT 路径为空时自动探测并回填到账号（约 30 秒）")
    args = ap.parse_args()

    print("=" * 68)
    print("  选股神AI · 实盘就绪自检 (Live Readiness)")
    print("=" * 68)

    check_interpreter()          # 第 0 项：先确认没选错解释器
    check_arch()
    check_xtquant()
    check_qmt_process()
    check_accounts(args.account, autofill=args.autofill)
    check_guard_mode()

    blockers = [r for r in _results if r[0] == BAD]
    warns = [r for r in _results if r[0] == WARN]

    print()
    for level, title, detail, action in _results:
        line = f" {level} {title}"
        print(line)
        if detail:
            print(f"      {detail}")
        if action:
            print(f"      → {action}")

    print()
    print("-" * 68)
    if _wrong_interpreter:
        # 解释器都选错了，下面的 ❌ 全是假象 —— 别让人照着假清单去装 QMT/xtquant
        print(" 结论：❌ 解释器选错，本清单不可信（并非实盘真的缺失这些条件）")
        print(" 请用系统 Python 3.12 重跑：py -3.12 scripts/check_live_ready.py")
        return 1
    if blockers:
        print(f" 结论：实盘未就绪 —— {len(blockers)} 项阻断（❌）、{len(warns)} 项提醒（⚠️）")
        print(" 按上面标注了 → 的步骤逐条处理后重跑本脚本。")
        print(" 完整清单：docs/实盘接入清单.md")
        return 1
    print(f" 结论：实盘前置条件已满足（{len(warns)} 项提醒）")
    print(" 下一步：界面「💹 实盘模式」→ 连接 → 安全面板两步令牌进入 LIVE_PENDING。")
    print(" 首笔务必留在 LIVE_PENDING 逐单人工确认，确认无误再提升到 LIVE。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

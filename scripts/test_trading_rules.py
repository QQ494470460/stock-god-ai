# -*- coding: utf-8 -*-
"""自动交易规则验证脚本

覆盖本次补齐的四项实盘规则：
  1. 交易日历 + 交易时段（法定节假日不能当交易日、非交易时段不能下单）
  2. 涨跌停 / 停牌过滤
  3. T+1 锁仓（当日买入当日不可卖，次日自动解锁）
  4. 滑点 / 冲击成本（买入更贵、卖出更便宜）

⚠️ 纸交易账户是真实存档，脚本会在开头备份、结尾恢复（try/finally 保证）。

用法: py -3.12 scripts/test_trading_rules.py
"""

import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.trading.paper import PAPER_ACCOUNT_FILE, PAPER_HISTORY_FILE  # noqa: E402
from src.trading import calendar as cal  # noqa: E402
from src.trading.rules import (  # noqa: E402
    board_limit_pct, limit_prices, is_limit_up, is_limit_down,
    is_suspended, check_buyable, check_sellable,
    sellable_quantity, apply_slippage, slippage_rate,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  ✓ {name}" + (f"  — {detail}" if detail else ""))
    else:
        FAIL.append(name)
        print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))


def test_calendar():
    print("\n【1】交易日历与时段")
    days = cal.refresh_calendar()
    check("交易日历可获取", days > 0, f"{days} 个交易日")
    check("日历未降级", not cal.is_degraded())

    # 周末必然非交易日
    sat = datetime(2026, 10, 10).date()      # 周六
    check("周六非交易日", not cal.is_trading_day(sat), str(sat))

    # 国庆假期内的工作日也不应是交易日（旧逻辑 weekday<5 会误判）
    holidays = [d for d in
                (datetime(2026, 10, d).date() for d in range(1, 8))
                if d.weekday() < 5]
    wrong = [str(d) for d in holidays if cal.is_trading_day(d)]
    check("国庆假期工作日被排除", not wrong,
          f"检查 {len(holidays)} 天，误判 {wrong or '无'}")

    # 时段
    check("10:00 属于交易时段",
          cal.is_trading_time(datetime(2026, 10, 9, 10, 0)))
    check("12:00 午休不属于交易时段",
          not cal.is_trading_time(datetime(2026, 10, 9, 12, 0)))
    check("20:00 不属于交易时段",
          not cal.is_trading_time(datetime(2026, 10, 9, 20, 0)))

    ok, why = cal.order_allowed(datetime(2026, 10, 10, 10, 0))  # 周六 10 点
    check("周六 10 点不允许下单", not ok, why)
    print(f"    market_status(周六10点) = {cal.market_status(datetime(2026, 10, 10, 10, 0))}")


def test_limit_rules():
    print("\n【2】涨跌停 / 停牌")
    check("主板 ±10%", board_limit_pct("600000") == 0.10)
    check("创业板 ±20%", board_limit_pct("300750") == 0.20)
    check("科创板 ±20%", board_limit_pct("688981") == 0.20)
    check("主板 ST ±5%", board_limit_pct("600123", "ST某某") == 0.05)
    check("北交所 ±30%", board_limit_pct("830799") == 0.30)

    dn, up = limit_prices(10.0, "600000")
    check("主板涨跌停价", (dn, up) == (9.0, 11.0), f"{dn} / {up}")
    _, up20 = limit_prices(10.0, "300750")
    check("创业板涨停价 20%", up20 == 12.0, str(up20))

    check("涨停识别", is_limit_up(11.0, 10.0, "600000"))
    check("未涨停不误判", not is_limit_up(10.5, 10.0, "600000"))
    check("跌停识别", is_limit_down(9.0, 10.0, "600000"))

    check("无价无量为停牌", is_suspended(0.0, 0))
    check("有价无量为停牌", is_suspended(10.0, 0))
    check("有价有量非停牌", not is_suspended(10.0, 100))

    ok, why = check_buyable("600000", "某某", 11.0, 10.0, 10000)
    check("涨停禁止买入", not ok, why)
    ok, why = check_sellable("600000", "某某", 9.0, 10.0, 10000)
    check("跌停禁止卖出", not ok, why)
    ok, _ = check_buyable("600000", "某某", 10.2, 10.0, 10000)
    check("正常价允许买入", ok)


def test_slippage():
    print("\n【3】滑点 / 冲击成本")
    buy = apply_slippage(10.0, "buy", 10.0 * 1000)
    sell = apply_slippage(10.0, "sell", 10.0 * 1000)
    check("买入成交价更高", buy > 10.0, f"{buy}")
    check("卖出成交价更低", sell < 10.0, f"{sell}")
    r_small = slippage_rate(50_000)
    r_large = slippage_rate(5_000_000)
    check("大额冲击成本更高", r_large > r_small,
          f"5万={r_small*100:.3f}%  500万={r_large*100:.3f}%")


def test_t1_and_paper():
    print("\n【4】T+1 锁仓（纸交易账户）")
    from src.trading.paper import PaperAccount

    acc = PaperAccount(initial_cash=1_000_000)
    acc.reset(1_000_000)

    ok, msg, rec = acc.buy("600000", "测试股", 10.00, 1000)
    check("买入成功", ok, msg)
    h = acc.holdings.get("600000", {})
    check("当日买入部分不可卖", h.get("available", -1) == 0,
          f"available={h.get('available')} today_quantity={h.get('today_quantity')}")
    check("滑点已计入成本", rec is not None and rec.price > 10.0,
          f"成交价 {rec.price if rec else '?'}（盘口 10.00）")

    ok2, msg2, _ = acc.sell("600000", 10.50, 1000)
    check("当日卖出被 T+1 拒绝", not ok2, msg2)

    # 模拟次日：解锁
    acc._last_reset_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    acc._check_daily_reset()
    h2 = acc.holdings.get("600000", {})
    check("次日自动解锁", h2.get("available") == 1000 and h2.get("today_quantity") == 0,
          f"available={h2.get('available')} today_quantity={h2.get('today_quantity')}")

    ok3, msg3, rec3 = acc.sell("600000", 10.50, 1000)
    check("次日可正常卖出", ok3, msg3)
    check("卖出成交价含滑点", rec3 is not None and rec3.price < 10.50,
          f"成交价 {rec3.price if rec3 else '?'}（盘口 10.50）")

    # 部分 T+1：持仓 1000（其中今日买 300）→ 只能卖 700
    # 部分 T+1：先买 700 股并跨日解锁，再买 300 股 → 可卖应为 700。
    # 注意初始资金要够：1500×700=105 万，给 500 万避免首笔因资金不足失败。
    acc.reset(5_000_000)
    acc.buy("600519", "贵州茅台", 1500.0, 700)
    acc._last_reset_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    acc._check_daily_reset()
    acc.buy("600519", "贵州茅台", 1500.0, 300)
    pos = acc.get_positions()[0]
    check("部分 T+1 可卖量正确", pos["sellable"] == 700,
          f"总持仓 {pos['quantity']}，今日买 {pos['today_quantity']}，可卖 {pos['sellable']}")
    ok4, msg4, _ = acc.sell("600519", 1500.0, 1000)
    check("超额卖出被拒", not ok4, msg4)
    ok5, msg5, _ = acc.sell("600519", 1500.0, 700)
    check("可卖额度内卖出成功", ok5, msg5)


def main() -> int:
    print("=" * 66)
    print("自动交易实盘规则验证")
    print("=" * 66)

    # 备份真实纸交易存档
    backup_dir = Path(tempfile.mkdtemp(prefix="paper_bak_"))
    backups = []
    for f in (PAPER_ACCOUNT_FILE, PAPER_HISTORY_FILE):
        if f.exists():
            dst = backup_dir / f.name
            shutil.copy2(f, dst)
            backups.append((f, dst))

    try:
        test_calendar()
        test_limit_rules()
        test_slippage()
        test_t1_and_paper()
    finally:
        # 恢复真实存档
        for orig, bak in backups:
            try:
                shutil.copy2(bak, orig)
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ 恢复 {orig.name} 失败: {e}")
        shutil.rmtree(backup_dir, ignore_errors=True)
        if backups:
            print(f"\n（已恢复 {len(backups)} 个纸交易存档文件）")

    print("\n" + "=" * 66)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 66)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

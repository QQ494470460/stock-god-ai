# -*- coding: utf-8 -*-
"""
自选股管理器 —— 命令行管理自选股池

用法:
  python scripts/watchlist_manager.py --list          # 查看自选
  python scripts/watchlist_manager.py --add CODES     # 添加 (逗号分隔)
  python scripts/watchlist_manager.py --remove CODES  # 删除 (逗号分隔)
  python scripts/watchlist_manager.py --set-alert KEY VALUE  # 设置告警参数
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetcher import fetcher

CONFIG_PATH = PROJECT_ROOT / "config" / "watchlist.json"
DEFAULT_CONFIG = {
    "watchlist": [],
    "alert": {
        "min_ai_score": 50,
        "alert_on_buy_signal": True,
        "alert_on_sell_signal": True,
        "alert_on_score_change": 15,
        "price_move_threshold": 3.0,
        "max_alerts_per_run": 5
    },
    "schedule": {
        "enabled": True,
        "interval_minutes": 5,
        "trading_hours_only": True
    }
}


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def get_stock_name(code: str) -> str:
    try:
        df = fetcher.get_stock_list()
        m = df[df["代码"] == code]
        if len(m) > 0:
            return str(m["名称"].values[0])
    except Exception:
        pass
    return code


def cmd_list(cfg):
    """列出当前自选股"""
    watchlist = cfg.get("watchlist", [])
    if not watchlist:
        print("⚠️ 自选股列表为空，使用 --add 添加股票")
        return

    print(f"\n📋 当前自选股 ({len(watchlist)} 只)")
    print("-" * 50)
    print(f"{'代码':<10} {'名称':<12} {'市场'}")
    print("-" * 50)

    try:
        df_all = fetcher.get_stock_list()
        name_map = {}
        for _, row in df_all.iterrows():
            name_map[str(row["代码"])] = str(row["名称"])
    except Exception:
        name_map = {}

    for code in watchlist:
        name = name_map.get(code, code)
        market = "沪市" if code.startswith(("6", "9")) else "深市"
        print(f"{code:<10} {name[:12]:<12} {market}")

    print("-" * 50)

    # 告警设置
    alert = cfg.get("alert", {})
    print(f"\n⚙️ 告警设置:")
    print(f"  AI评分阈值: ≥{alert.get('min_ai_score', 50)}")
    print(f"  买入告警: {'✅ 开' if alert.get('alert_on_buy_signal') else '❌ 关'}")
    print(f"  卖出告警: {'✅ 开' if alert.get('alert_on_sell_signal') else '❌ 关'}")
    print(f"  波动阈值: ±{alert.get('price_move_threshold', 3.0)}%")
    print(f"  每次最多告警: {alert.get('max_alerts_per_run', 5)} 条")

    schedule = cfg.get("schedule", {})
    print(f"\n⏰ 定时监控:")
    print(f"  状态: {'✅ 开' if schedule.get('enabled') else '❌ 关'}")
    print(f"  间隔: {schedule.get('interval_minutes', 5)} 分钟")
    print(f"  仅交易时段: {'✅' if schedule.get('trading_hours_only') else '❌'}")
    print()


def cmd_add(cfg, codes_str):
    """添加股票到自选"""
    codes = [c.strip() for c in codes_str.replace("，", ",").split(",") if c.strip()]
    valid = []
    invalid = []

    try:
        df_all = fetcher.get_stock_list()
        all_codes = set(str(c) for c in df_all["代码"].values)
        for c in codes:
            if c in all_codes:
                valid.append(c)
            else:
                invalid.append(c)
    except Exception:
        valid = codes  # 无法验证时全接受

    current = set(cfg.get("watchlist", []))
    added = []
    existed = []
    for c in valid:
        if c not in current:
            current.add(c)
            added.append(c)
        else:
            existed.append(c)

    cfg["watchlist"] = sorted(current)
    save_config(cfg)

    for c in added:
        name = get_stock_name(c)
        print(f"✅ 已添加: {c} {name}")
    for c in existed:
        print(f"⏭️ 已存在: {c}")
    for c in invalid:
        print(f"❌ 未找到: {c}")

    print(f"\n📋 当前自选: {len(cfg['watchlist'])} 只")
    print(f"   文件: {CONFIG_PATH}")


def cmd_remove(cfg, codes_str):
    """从自选中删除"""
    codes = [c.strip() for c in codes_str.replace("，", ",").split(",") if c.strip()]
    current = cfg.get("watchlist", [])
    removed = []
    not_found = []

    for c in codes:
        if c in current:
            current.remove(c)
            removed.append(c)
        else:
            not_found.append(c)

    cfg["watchlist"] = current
    save_config(cfg)

    for c in removed:
        print(f"🗑️ 已删除: {c}")
    for c in not_found:
        print(f"⚠️ 不在列表中: {c}")

    print(f"\n📋 当前自选: {len(cfg['watchlist'])} 只")


def cmd_set_alert(cfg, key, value):
    """设置告警参数"""
    alert = cfg.get("alert", {})
    if key in alert:
        # 类型转换
        if isinstance(alert[key], bool):
            value = value.lower() in ("true", "1", "yes", "on")
        elif isinstance(alert[key], int):
            value = int(value)
        elif isinstance(alert[key], float):
            value = float(value)
        alert[key] = value
        save_config(cfg)
        print(f"✅ 告警参数已更新: {key} = {value}")
    else:
        print(f"❌ 未知告警参数: {key}")
        print(f"   可用参数: {list(alert.keys())}")


def main():
    parser = argparse.ArgumentParser(description="选股神AI - 自选股管理器")
    parser.add_argument("--list", action="store_true", help="列出当前自选")
    parser.add_argument("--add", type=str, help="添加股票（逗号分隔）")
    parser.add_argument("--remove", type=str, help="删除股票（逗号分隔）")
    parser.add_argument("--set-alert", nargs=2, metavar=("KEY", "VALUE"),
                        help="设置告警参数")
    args = parser.parse_args()

    cfg = load_config()

    if args.list:
        cmd_list(cfg)
    elif args.add:
        cmd_add(cfg, args.add)
    elif args.remove:
        cmd_remove(cfg, args.remove)
    elif args.set_alert:
        cmd_set_alert(cfg, args.set_alert[0], args.set_alert[1])
    else:
        cmd_list(cfg)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
自动交易CLI入口
启动/停止自动交易、手动下单、账户查询、模拟测试
"""

import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import LOG_DIR

from src.trading import (
    YuekaiBroker, AccountManager, TradeExecutor, AutoTrader,
    OrderSide, OrderType,
)
from src.trading.account import account_manager
from src.trading.executor import RiskConfig


def setup_logging():
    """配置日志"""
    log_file = LOG_DIR / f"autotrade_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, rotation="100 MB", level="INFO")
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


def cmd_connect(args):
    """连接到券商"""
    broker = YuekaiBroker()
    
    acct = None
    if args.account_id:
        acct = account_manager.get(args.account_id)
    else:
        acct = account_manager.get_active()
    
    if not acct:
        print("❌ 未找到账号，请先添加账号: python scripts/autotrade.py add-account")
        return
    
    password = ""
    if acct.qmt_password_enc:
        password = account_manager.get_decrypted_password(acct.account_id) or ""
    
    success = broker.connect(
        qmt_path=acct.qmt_path,
        account_id=acct.qmt_account,
        password=password,
    )
    
    if success:
        print(f"✅ 已连接 {broker.get_broker_name()} | 模拟: {broker.is_simulated}")
        
        # 显示账户
        account = broker.query_account()
        if account:
            print(f"\n💰 账户信息:")
            print(f"   总资产: ¥{account.total_asset:,.2f}")
            print(f"   可用资金: ¥{account.available_cash:,.2f}")
            print(f"   持仓市值: ¥{account.market_value:,.2f}")
        
        # 显示持仓
        positions = broker.query_positions()
        if positions:
            print(f"\n📊 持仓 ({len(positions)} 只):")
            for p in positions:
                emoji = "🟢" if p.pnl_pct > 0 else "🔴"
                print(f"   {emoji} {p.code} {p.name} | {p.quantity}股 | 成本{p.avg_cost:.2f} "
                      f"| 现价{p.current_price:.2f} | 盈亏{p.pnl_pct:+.2f}% | 市值¥{p.market_value:,.0f}")
        
        broker.disconnect()
    else:
        print("❌ 连接失败，请检查QMT路径和账号配置")


def cmd_start(args):
    """启动自动交易"""
    setup_logging()
    
    broker = YuekaiBroker()
    
    # 连接
    acct = account_manager.get_active()
    if not acct:
        print("❌ 无激活账号，请先: python scripts/autotrade.py add-account")
        return
    
    password = account_manager.get_decrypted_password(acct.account_id) or ""
    
    success = broker.connect(
        qmt_path=acct.qmt_path,
        account_id=acct.qmt_account,
        password=password,
    )
    
    if not success:
        print("❌ 券商连接失败")
        return
    
    # 风控配置
    risk = RiskConfig(
        max_position_pct=args.max_position_pct,
        stop_loss_pct=args.stop_loss_pct,
        stop_profit_pct=args.stop_profit_pct,
        min_ai_score_buy=args.min_ai_score,
        max_daily_loss=args.max_daily_loss,
    )
    
    # 创建 AutoTrader
    trader = AutoTrader(broker, risk)
    
    # 接入 AI 预测器
    if not args.no_ai:
        trader.use_builtin_predictor([])  # 将从 watchlist.json 自动读取
        print("✅ AI 预测器已接入")
    
    # 启动止盈止损监控（高频）
    trader.start_stop_monitor(interval_seconds=args.stop_interval)
    
    # 启动自动交易（低频：AI信号扫描）
    trader.start(
        interval_seconds=args.scan_interval,
        trading_hours_only=not args.all_hours,
        on_callback=lambda results: print_results(results),
    )
    
    print(f"\n🚀 自动交易已启动!")
    print(f"   AI信号扫描间隔: {args.scan_interval}s")
    print(f"   止盈止损监控间隔: {args.stop_interval}s")
    print(f"   买入最低AI评分: {args.min_ai_score}")
    print(f"   止损: {args.stop_loss_pct}% | 止盈: {args.stop_profit_pct}%")
    print(f"   按 Ctrl+C 停止\n")
    
    try:
        while True:
            time.sleep(30)
            trader.print_status()
    except KeyboardInterrupt:
        print("\n⏹️ 收到停止信号...")
        trader.stop()
        print("👋 自动交易已停止")


def print_results(results):
    """打印交易结果"""
    for r in results:
        if r.get("type") == "stop":
            o = r["order"]
            logger.warning(f"🛑 止盈止损: {o.code} {o.side.value.upper()} {o.quantity}股")
        elif r.get("success"):
            logger.info(f"✅ {r.get('msg', '')}")
        else:
            logger.debug(f"⏭️ {r.get('code', '')} {r.get('msg', '')}")


def cmd_place_order(args):
    """手动下单"""
    broker = YuekaiBroker()
    
    acct = account_manager.get_active()
    if not acct:
        print("❌ 无激活账号")
        return
    
    password = account_manager.get_decrypted_password(acct.account_id) or ""
    broker.connect(qmt_path=acct.qmt_path, account_id=acct.qmt_account, password=password)
    
    side = OrderSide.BUY if args.side.lower() == "buy" else OrderSide.SELL
    order_type = OrderType.LIMIT if args.limit else OrderType.MARKET_BEST5
    
    order = broker.place_order(
        code=args.code,
        side=side,
        order_type=order_type,
        price=args.price if args.limit else 0,
        quantity=args.quantity,
    )
    
    if order:
        print(f"✅ 下单成功: {order.order_id}")
        print(f"   {order.code} {'买入' if side == OrderSide.BUY else '卖出'} "
              f"{order.quantity}股 @ {order.price:.2f}")
    else:
        print("❌ 下单失败")
    
    broker.disconnect()


def cmd_query(args):
    """查询账户/持仓/订单"""
    broker = YuekaiBroker()
    
    acct = account_manager.get_active()
    if not acct:
        print("❌ 无激活账号")
        return
    
    password = account_manager.get_decrypted_password(acct.account_id) or ""
    broker.connect(qmt_path=acct.qmt_path, account_id=acct.qmt_account, password=password)
    
    if args.type == "account":
        info = broker.query_account()
        if info:
            print(f"\n💰 账户: {info.account_id}")
            print(f"   总资产: ¥{info.total_asset:,.2f}")
            print(f"   可用: ¥{info.available_cash:,.2f}")
            print(f"   持仓: ¥{info.market_value:,.2f}")
            print(f"   冻结: ¥{info.frozen_cash:,.2f}")
    
    elif args.type == "positions":
        positions = broker.query_positions()
        if not positions:
            print("📭 当前无持仓")
        else:
            print(f"\n📊 持仓 ({len(positions)} 只):")
            print(f"{'代码':<12} {'名称':<10} {'数量':>6} {'可用':>6} {'成本':>8} {'现价':>8} {'盈亏%':>8} {'市值':>12}")
            print("-" * 80)
            for p in positions:
                print(f"{p.code:<12} {p.name:<10} {p.quantity:>6} {p.available:>6} "
                      f"{p.avg_cost:>8.2f} {p.current_price:>8.2f} "
                      f"{p.pnl_pct:>+7.2f}% ¥{p.market_value:>10,.0f}")
    
    elif args.type == "orders":
        orders = broker.query_orders()
        if not orders:
            print("📭 今日无订单")
        else:
            print(f"\n📋 今日订单 ({len(orders)} 笔):")
            for o in orders:
                emoji = "🟢" if o.side == OrderSide.BUY else "🔴"
                print(f"   {emoji} {o.order_id} | {o.code} | {o.side.value.upper()} "
                      f"{o.quantity}股 @ {o.price:.2f} | {o.status.value} | "
                      f"{o.filled_quantity}/{o.quantity} 成交")
    
    broker.disconnect()


def cmd_add_account(args):
    """添加/更新券商账号"""
    acct = account_manager.add_or_update(
        broker="yuekai",
        account_id=args.account_id,
        name=args.name or f"粤开_{args.account_id[-4:]}",
        qmt_path=args.qmt_path,
        qmt_account=args.qmt_account or args.account_id,
        qmt_password=args.password,
        set_active=True,
    )
    print(f"✅ 账号已保存: {acct.name} ({acct.account_id})")
    print(f"   QMT路径: {acct.qmt_path}")
    print(f"   (密码已加密存储)")


def cmd_list_accounts(args):
    """列出所有已保存的账号"""
    accounts = account_manager.accounts
    if not accounts:
        print("📭 暂无保存的券商账号")
        print("   添加账号: python scripts/autotrade.py add-account")
        return
    
    print(f"\n📋 已保存账号 ({len(accounts)} 个):")
    for a in accounts:
        active_mark = " ✅ 激活" if a.active else ""
        print(f"   {a.broker} | {a.name} | {a.account_id}{active_mark}")
        print(f"      QMT路径: {a.qmt_path}")


def cmd_simulate(args):
    """模拟交易测试 — 不连券商，用模拟模式跑一遍流程"""
    setup_logging()
    
    broker = YuekaiBroker()  # 自动进入模拟模式（xtquant未安装）
    
    risk = RiskConfig()
    trader = AutoTrader(broker, risk)
    
    if not args.no_ai:
        trader.use_builtin_predictor([])
    
    print("🧪 模拟交易测试模式")
    print(f"   买入最低AI评分: {risk.min_ai_score_buy}")
    print(f"   止损: {risk.stop_loss_pct}% | 止盈: {risk.stop_profit_pct}%")
    print()
    
    # 单次扫描
    print("📡 执行AI信号扫描...")
    results = trader.run_once()
    
    if results:
        print(f"\n✅ 扫描结果: {len(results)} 个信号")
        for r in results:
            print(f"   {r}")
    else:
        print("📭 无交易信号")
    
    # 显示状态
    trader.print_status()
    
    print("💡 提示: 启动持续监控模式: python scripts/autotrade.py start --simulate")


def cmd_start_simulate(args):
    """启动模拟模式持续运行（不连实盘）"""
    setup_logging()
    
    broker = YuekaiBroker()  # 模拟模式
    
    risk = RiskConfig(
        max_position_pct=args.max_position_pct,
        stop_loss_pct=args.stop_loss_pct,
        stop_profit_pct=args.stop_profit_pct,
        min_ai_score_buy=args.min_ai_score,
    )
    
    trader = AutoTrader(broker, risk)
    
    if not args.no_ai:
        trader.use_builtin_predictor([])
    
    trader.start_stop_monitor(interval_seconds=args.stop_interval)
    trader.start(
        interval_seconds=args.scan_interval,
        trading_hours_only=not args.all_hours,
        on_callback=lambda results: print_results(results),
    )
    
    print(f"\n🧪 模拟交易已启动!")
    print(f"   扫描间隔: {args.scan_interval}s | 止损监控: {args.stop_interval}s")
    print(f"   按 Ctrl+C 停止\n")
    
    try:
        while True:
            time.sleep(60)
            trader.print_status()
    except KeyboardInterrupt:
        print("\n⏹️ 停止...")
        trader.stop()
        print("👋 模拟交易已停止")


def main():
    parser = argparse.ArgumentParser(
        description="选股神AI · 量化自动交易",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 添加粤开证券账号
  python scripts/autotrade.py add-account --account-id 1234567890 --qmt-path D:\\QMT\\userdata_mini
  
  # 模拟测试（不连实盘）
  python scripts/autotrade.py simulate
  
  # 启动模拟持续运行
  python scripts/autotrade.py start-simulate
  
  # 启动实盘自动交易
  python scripts/autotrade.py start
  
  # 查询持仓
  python scripts/autotrade.py query --type positions
  
  # 手动下单
  python scripts/autotrade.py order --code 000001.SZ --side buy --quantity 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # connect
    p = subparsers.add_parser("connect", help="测试券商连接")
    p.add_argument("--account-id", help="账号ID（默认使用激活账号）")
    
    # start
    p = subparsers.add_parser("start", help="启动实盘自动交易")
    p.add_argument("--scan-interval", type=int, default=120, help="AI信号扫描间隔（秒，默认120）")
    p.add_argument("--stop-interval", type=int, default=10, help="止盈止损监控间隔（秒，默认10）")
    p.add_argument("--max-position-pct", type=float, default=0.30, help="单票最大仓位比例（默认0.30）")
    p.add_argument("--stop-loss-pct", type=float, default=-5.0, help="止损线（%，默认-5）")
    p.add_argument("--stop-profit-pct", type=float, default=15.0, help="止盈线（%，默认15）")
    p.add_argument("--min-ai-score", type=float, default=55.0, help="买入最低AI评分（默认55）")
    p.add_argument("--max-daily-loss", type=float, default=5000.0, help="单日最大亏损（元，默认5000）")
    p.add_argument("--no-ai", action="store_true", help="不使用AI预测（纯手动）")
    p.add_argument("--all-hours", action="store_true", help="24小时运行（不限于交易时段）")
    
    # simulate
    p = subparsers.add_parser("simulate", help="模拟交易单次测试")
    p.add_argument("--no-ai", action="store_true", help="不使用AI预测")
    
    # start-simulate
    p = subparsers.add_parser("start-simulate", help="启动模拟持续运行")
    p.add_argument("--scan-interval", type=int, default=120, help="AI信号扫描间隔（秒）")
    p.add_argument("--stop-interval", type=int, default=10, help="止盈止损监控间隔（秒）")
    p.add_argument("--max-position-pct", type=float, default=0.30)
    p.add_argument("--stop-loss-pct", type=float, default=-5.0)
    p.add_argument("--stop-profit-pct", type=float, default=15.0)
    p.add_argument("--min-ai-score", type=float, default=55.0)
    p.add_argument("--no-ai", action="store_true")
    p.add_argument("--all-hours", action="store_true")
    
    # order
    p = subparsers.add_parser("order", help="手动下单")
    p.add_argument("--code", required=True, help="股票代码 (如 000001.SZ)")
    p.add_argument("--side", required=True, choices=["buy", "sell"], help="买卖方向")
    p.add_argument("--quantity", type=int, required=True, help="数量（股，必须是100的整数倍）")
    p.add_argument("--price", type=float, default=0, help="限价（不指定则用市价）")
    p.add_argument("--limit", action="store_true", help="使用限价单")
    
    # query
    p = subparsers.add_parser("query", help="查询账户/持仓/订单")
    p.add_argument("--type", default="account", choices=["account", "positions", "orders"])
    
    # add-account
    p = subparsers.add_parser("add-account", help="添加/更新券商账号")
    p.add_argument("--account-id", required=True, help="资金账号")
    p.add_argument("--qmt-path", required=True, help="QMT安装目录下的userdata_mini路径")
    p.add_argument("--qmt-account", help="QMT登录账号（默认同资金账号）")
    p.add_argument("--password", help="交易密码（将加密存储）")
    p.add_argument("--name", help="账号别名")
    
    # list-accounts
    subparsers.add_parser("list-accounts", help="列出已保存的券商账号")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    cmds = {
        "connect": cmd_connect,
        "start": cmd_start,
        "simulate": cmd_simulate,
        "start-simulate": cmd_start_simulate,
        "order": cmd_place_order,
        "query": cmd_query,
        "add-account": cmd_add_account,
        "list-accounts": cmd_list_accounts,
    }
    
    fn = cmds.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

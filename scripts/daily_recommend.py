# -*- coding: utf-8 -*-
"""
每日推荐脚本
收盘后运行，AI自动扫描全市场并生成精选股票池
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from config.settings import config as app_config, LOG_DIR
from src.data.fetcher import fetcher
from src.models.predictor import predictor


def parse_args():
    parser = argparse.ArgumentParser(description="选股神AI - 每日推荐")
    parser.add_argument("--pool", type=str, default="hs300", choices=["all", "hs300", "zz500"],
                       help="股票池 (默认: hs300)")
    parser.add_argument("--top", type=int, default=20, help="推荐数量 (默认: 20)")
    parser.add_argument("--min-score", type=float, default=60.0, help="最低评分 (默认: 60)")
    parser.add_argument("--output", type=str, help="输出CSV文件路径")
    parser.add_argument("--no-log", action="store_true", help="不输出日志文件")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.no_log:
        logger.add(LOG_DIR / f"recommend_{datetime.now().strftime('%Y%m%d')}.log", level="INFO")
    
    logger.info("=" * 60)
    logger.info(f"🚀 选股神AI - 每日推荐 ({datetime.now().strftime('%Y-%m-%d')})")
    logger.info(f"   股票池: {args.pool} | TOP: {args.top} | 最低评分: {args.min_score}")
    logger.info("=" * 60)
    
    # 检查模型
    if not predictor.is_ready:
        logger.error("❌ 模型未训练! 请先运行: python scripts/train.py")
        return
    
    # 获取股票池
    codes = fetcher.get_stock_pool(args.pool)
    logger.info(f"📋 股票池: {len(codes)} 只")
    
    # 获取行情数据
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
    
    logger.info(f"📊 获取行情数据 ({start_date} ~ {end_date})...")
    
    stock_data = {}
    stock_names = {}
    failed = 0
    
    for i, code in enumerate(codes):
        try:
            df = fetcher.get_daily_kline(code, start_date=start_date, end_date=end_date)
            if len(df) >= 60:
                stock_data[code] = df
                stock_names[code] = code
            else:
                failed += 1
        except Exception:
            failed += 1
        
        if (i + 1) % 50 == 0:
            logger.info(f"   进度: {i+1}/{len(codes)}")
    
    logger.info(f"   获取完成: {len(stock_data)} 只有效数据, {failed} 只失败")
    
    # AI预测
    logger.info("🤖 AI模型预测中...")
    result_df = predictor.get_top_recommendations(
        stock_data, stock_names,
        top_n=args.top, min_score=args.min_score
    )
    
    # 展示结果
    print("\n" + "=" * 80)
    print(f"  📈 选股神AI - 每日精选 TOP {len(result_df)}")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print(f"\n{'排名':<5} {'代码':<8} {'名称':<10} {'评分':<8} {'上涨概率':<10} {'XGB信号':<10} {'LGB信号'}")
    print("-" * 80)
    
    for _, row in result_df.iterrows():
        print(f"  {row['排名']:<5} {row['代码']:<8} {row['名称']:<10} "
              f"{row['评分']:<8.1f} {row['上涨概率']:<10.1f}% "
              f"{row['XGB信号']:<10} {row['LGB信号']}")
    
    print("\n" + "=" * 80)
    
    buy_signals = sum(1 for _, r in result_df.iterrows() if "买入" in r["XGB信号"] and "买入" in r["LGB信号"])
    logger.info(f"\n📊 统计: {buy_signals} 只双模型一致推荐买入")
    logger.info(f"🔥 最高评分: {result_df['评分'].iloc[0]:.1f} ({result_df['代码'].iloc[0]} {result_df['名称'].iloc[0]})")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"💾 结果已保存到: {output_path}")
    else:
        # 默认保存
        default_path = LOG_DIR / f"recommendations_{datetime.now().strftime('%Y%m%d')}.csv"
        result_df.to_csv(default_path, index=False, encoding="utf-8-sig")
        logger.info(f"💾 结果已保存到: {default_path}")


if __name__ == "__main__":
    main()
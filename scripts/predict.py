# -*- coding: utf-8 -*-
"""
快速预测脚本
对指定股票列表进行批量AI预测
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from config.settings import config as app_config
from src.data.fetcher import fetcher
from src.models.predictor import predictor


def parse_args():
    parser = argparse.ArgumentParser(description="选股神AI - 批量预测")
    parser.add_argument("--codes", type=str, nargs="+", required=True,
                       help="股票代码列表，如: 000001 600519")
    parser.add_argument("--from-file", type=str,
                       help="从文件读取股票代码 (每行一个)")
    parser.add_argument("--json", action="store_true",
                       help="JSON格式输出")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not predictor.is_ready:
        logger.error("❌ 模型未训练! 请先运行: python scripts/train.py")
        return
    
    # 获取代码列表
    codes = args.codes or []
    if args.from_file:
        with open(args.from_file, "r") as f:
            codes.extend([line.strip() for line in f if line.strip()])
    
    codes = list(set(codes))  # 去重
    logger.info(f"预测 {len(codes)} 只股票: {codes}")
    
    # 获取数据
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    stock_data = {}
    stock_names = {}
    
    for code in codes:
        df = fetcher.get_daily_kline(code, start_date=start_date)
        if len(df) >= 60:
            stock_data[code] = df
            stock_names[code] = code
        else:
            logger.warning(f"{code} 数据不足 (仅{len(df)}条)")
    
    if not stock_data:
        logger.error("没有有效数据")
        return
    
    # 预测
    predictions = predictor.predict_batch(stock_data, stock_names)
    
    if args.json:
        import json
        output = []
        for p in predictions:
            output.append({
                "code": p.code,
                "score": round(p.score, 1),
                "prob_up": round(p.prob_up, 3),
                "xgb_signal": p.xgb_signal,
                "lgb_signal": p.lgb_signal,
            })
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        for p in predictions:
            xgb_emoji = "📈" if p.xgb_signal == 1 else ("📉" if p.xgb_signal == -1 else "➡️")
            lgb_emoji = "📈" if p.lgb_signal == 1 else ("📉" if p.lgb_signal == -1 else "➡️")
            print(f"{p.code:<8} 评分:{p.score:6.1f}  "
                  f"涨:{p.prob_up*100:5.1f}%  "
                  f"平:{p.prob_neutral*100:5.1f}%  "
                  f"跌:{p.prob_down*100:5.1f}%  "
                  f"XGB:{xgb_emoji} LGB:{lgb_emoji}")


if __name__ == "__main__":
    main()
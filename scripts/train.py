# -*- coding: utf-8 -*-
"""
训练脚本
完整的模型训练流程：数据获取 → 特征工程 → 模型训练
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from config.settings import config as app_config, MODEL_DIR, LOG_DIR
from src.data.fetcher import fetcher
from src.features.engine import feature_engine
from src.models.trainer import MLTrainer
from src.models.lstm import LSTMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="选股神AI - 模型训练")
    parser.add_argument("--pool", type=str, default="hs300", choices=["all", "hs300", "zz500"],
                       help="股票池 (默认: hs300)")
    parser.add_argument("--start-date", type=str, default="2018-01-01",
                       help="起始日期 (默认: 2018-01-01)")
    parser.add_argument("--days", type=int, default=200,
                       help="获取最近多少天的数据 (默认: 200)")
    parser.add_argument("--model", type=str, default="ensemble",
                       choices=["xgboost", "lightgbm", "ensemble", "all"],
                       help="训练模型 (默认: ensemble)")
    parser.add_argument("--no-lstm", action="store_true",
                       help="跳过LSTM训练 (Windows可能缺少PyTorch)")
    parser.add_argument("--gpu", action="store_true",
                       help="使用GPU训练LSTM")
    parser.add_argument("--update-data-only", action="store_true",
                       help="仅更新数据，不训练")
    parser.add_argument("--max-stocks", type=int, default=100,
                       help="最大训练股票数 (默认: 100)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 配置日志
    logger.add(
        LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="100 MB",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("🚀 选股神AI - 开始模型训练")
    logger.info(f"   股票池: {args.pool}")
    logger.info(f"   日期范围: {args.start_date} ~ {datetime.now().strftime('%Y-%m-%d')}")
    logger.info(f"   最大股票数: {args.max_stocks}")
    logger.info("=" * 60)
    
    # ============ Step 1: 获取股票列表 ============
    logger.info("\n📋 Step 1/4: 获取股票列表...")
    codes = fetcher.get_stock_pool(args.pool)
    logger.info(f"   获取到 {len(codes)} 只股票")
    
    # 限制数量（训练阶段）
    if args.max_stocks > 0:
        codes = codes[:args.max_stocks]
        logger.info(f"   限制训练数为 {len(codes)} 只")
    
    if args.update_data_only:
        logger.info("仅更新数据模式，开始获取日线...")
        end_date = datetime.now().strftime("%Y%m%d")
        fetcher.get_batch_daily_kline(codes, start_date=args.start_date, end_date=end_date)
        logger.info("✅ 数据更新完成")
        return
    
    # ============ Step 2: 获取行情数据 & 构建特征 ============
    logger.info("\n📊 Step 2/4: 获取行情数据 & 构建特征...")
    
    all_features = []
    all_labels = []
    
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = args.start_date
    
    for i, code in enumerate(codes):
        try:
            df = fetcher.get_daily_kline(code, start_date=start_date, end_date=end_date)
            if len(df) < 120:  # 至少120个交易日
                continue
            
            # 构建特征和标签
            features, labels = feature_engine.prepare_training_data(df)
            
            if len(features) > 0:
                features["code"] = code
                all_features.append(features)
                all_labels.append(labels)
            
            if (i + 1) % 20 == 0:
                logger.info(f"   进度: {i+1}/{len(codes)} - 已收集 {len(all_features)} 只有效股票")
        
        except Exception as e:
            logger.warning(f"   处理 {code} 失败: {e}")
    
    if len(all_features) == 0:
        logger.error("❌ 没有收集到任何有效训练数据!")
        return
    
    # 合并所有股票的特征
    X_all = pd.concat(all_features, ignore_index=True)
    y_all = pd.concat(all_labels, ignore_index=True)
    
    # 去除非数值列
    X_all = X_all.select_dtypes(include=[np.number])
    
    logger.info(f"\n   训练数据汇总:")
    logger.info(f"   样本总数: {len(X_all)}")
    logger.info(f"   特征维度: {X_all.shape[1]}")
    logger.info(f"   标签分布: 涨={len(y_all[y_all==1])}, 震荡={len(y_all[y_all==0])}, 跌={len(y_all[y_all==-1])}")
    
    # ============ Step 3: 训练ML模型 ============
    logger.info("\n🧠 Step 3/4: 训练机器学习模型...")
    
    trainer = MLTrainer()
    results = {}
    
    if args.model in ("xgboost", "ensemble", "all"):
        xgb_result = trainer.train_xgboost(X_all, y_all)
        results["xgboost"] = xgb_result
    
    if args.model in ("lightgbm", "ensemble", "all"):
        lgb_result = trainer.train_lightgbm(X_all, y_all)
        results["lightgbm"] = lgb_result
    
    # ============ Step 4: 训练LSTM（可选）============
    if not args.no_lstm:
        try:
            logger.info("\n🧬 Step 4/4: 训练LSTM深度学习模型...")
            
            # 标准化数据
            X_scaled = trainer.scaler.transform(X_all.fillna(0))
            y_encoded = trainer.label_encoder.transform(y_all)
            
            lstm_trainer = LSTMTrainer(
                input_size=X_all.shape[1],
                device="cuda" if args.gpu else "cpu"
            )
            
            history = lstm_trainer.train(X_scaled, y_encoded.astype(float))
            
            logger.info(f"   LSTM训练完成: 最佳验证精度 = {max(history['val_acc']):.4f}")
        except ImportError as e:
            logger.warning(f"   LSTM训练跳过: {e} (可能未安装PyTorch)")
        except Exception as e:
            logger.error(f"   LSTM训练失败: {e}")
    
    # ============ 完成 ============
    logger.info("\n" + "=" * 60)
    logger.info("🎉 模型训练完成!")
    logger.info(f"   模型保存位置: {MODEL_DIR}")
    logger.info("\n📊 最终结果:")
    
    for name, result in results.items():
        logger.info(f"   {name}: Acc={result.accuracy:.4f}, F1={result.f1_macro:.4f}")
        logger.info(f"   Top-5 重要特征:")
        for _, row in result.feature_importance.head(5).iterrows():
            logger.info(f"      {row['feature']}: {row['importance']:.4f}")
    
    logger.info("\n💡 下一步:")
    logger.info("   python scripts/daily_recommend.py  # AI选股")
    logger.info("   streamlit run src/ui/app.py         # 启动看板")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
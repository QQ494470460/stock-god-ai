# -*- coding: utf-8 -*-
"""
FastAPI 服务 - 选股神 REST API
提供股票推荐、单股分析、批量预测等接口
"""

from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from config.settings import config as app_config
from src.data.fetcher import fetcher
from src.models.predictor import predictor

# FastAPI应用
app = FastAPI(
    title="选股神 API - Stock God AI",
    description="A股AI智能选股系统 REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 数据模型 ==========

class StockItem(BaseModel):
    code: str = Field(..., description="股票代码")
    name: str = Field(..., description="股票名称")


class PredictionRequest(BaseModel):
    codes: List[str] = Field(..., description="股票代码列表")
    start_date: Optional[str] = Field("2020-01-01", description="起始日期")


class PredictionResult(BaseModel):
    code: str
    name: str
    score: float
    prob_up: float
    prob_neutral: float
    prob_down: float
    xgb_signal: str
    lgb_signal: str


class RecommendationResponse(BaseModel):
    date: str
    model_status: bool
    count: int
    recommendations: List[PredictionResult]


class StockAnalysis(BaseModel):
    code: str
    name: str
    prediction: Optional[PredictionResult]
    recent_performance: dict
    technical_summary: dict


class HotSector(BaseModel):
    name: str
    pct_change: float
    lead_stock: str
    stock_count: int


# ========== API 接口 ==========

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "选股神 API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/api/v1/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    top_n: int = Query(20, ge=1, le=100, description="推荐数量"),
    min_score: float = Query(60.0, ge=0, le=100, description="最低评分"),
    pool: str = Query("hs300", description="股票池: all/hs300/zz500")
):
    """
    获取今日推荐股票
    
    - 基于多模型集成预测，输出TOP-N推荐
    - 评分越高表示AI越看好
    """
    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="模型尚未训练，请先运行 python scripts/train.py"
        )
    
    try:
        # 获取股票池
        codes = fetcher.get_stock_pool(pool)
        
        # 获取行情数据
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        
        logger.info(f"开始扫描 {len(codes)} 只股票...")
        
        stock_data = {}
        stock_names = {}
        
        # 批量获取（限制数量以提高响应速度）
        sample_codes = codes[:100]  # 先取前100只做演示，生产环境可全量
        for code in sample_codes:
            df = fetcher.get_daily_kline(code, start_date=start_date)
            if len(df) >= 60:
                stock_data[code] = df
                stock_names[code] = code  # 实际使用时从stock_list获取名称
        
        # 预测
        df_result = predictor.get_top_recommendations(
            stock_data, stock_names, top_n=top_n, min_score=min_score
        )
        
        recommendations = []
        for _, row in df_result.iterrows():
            recommendations.append(PredictionResult(
                code=row["代码"],
                name=row["名称"],
                score=row["评分"],
                prob_up=row["上涨概率"],
                prob_neutral=row["震荡概率"],
                prob_down=row["下跌概率"],
                xgb_signal=row["XGB信号"],
                lgb_signal=row["LGB信号"],
            ))
        
        return RecommendationResponse(
            date=datetime.now().strftime("%Y-%m-%d"),
            model_status=predictor.is_ready,
            count=len(recommendations),
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"获取推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stock/{code}", response_model=StockAnalysis)
async def analyze_stock(
    code: str,
    days: int = Query(120, ge=30, le=365, description="分析天数")
):
    """获取单只股票的AI分析"""
    try:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = fetcher.get_daily_kline(code, start_date=start_date)
        
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"股票 {code} 无数据")
        
        # AI预测
        prediction = None
        if predictor.is_ready and len(df) >= 60:
            pred = predictor.predict_single(df, code, code)
            prediction = PredictionResult(
                code=pred.code,
                name=pred.name,
                score=pred.score,
                prob_up=pred.prob_up,
                prob_neutral=pred.prob_neutral,
                prob_down=pred.prob_down,
                xgb_signal="📈买入" if pred.xgb_signal == 1 else ("📉卖出" if pred.xgb_signal == -1 else "➡️持有"),
                lgb_signal="📈买入" if pred.lgb_signal == 1 else ("📉卖出" if pred.lgb_signal == -1 else "➡️持有"),
            )
        
        # 近期表现
        recent = df.tail(20)
        recent_perf = {
            "latest_close": float(recent["close"].iloc[-1]),
            "return_5d": float(recent["close"].pct_change(5).iloc[-1] * 100) if len(recent) >= 5 else 0,
            "return_20d": float(recent["close"].pct_change(20).iloc[-1] * 100) if len(recent) >= 20 else 0,
            "volatility_20d": float(recent["close"].pct_change().std() * 100),
            "avg_volume": float(recent["volume"].mean()),
        }
        
        # 技术摘要
        tech_summary = {
            "ma_5_20_bullish": bool(recent["close"].tail(5).mean() > recent["close"].tail(20).mean()) if len(recent) >= 20 else False,
            "in_uptrend": bool(recent["close"].iloc[-1] > recent["close"].tail(10).mean()) if len(recent) >= 10 else False,
            "data_points": len(df),
        }
        
        return StockAnalysis(
            code=code,
            name=code,
            prediction=prediction,
            recent_performance=recent_perf,
            technical_summary=tech_summary
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析 {code} 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict")
async def batch_predict(request: PredictionRequest):
    """批量预测多只股票"""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="模型尚未训练")
    
    try:
        stock_data = {}
        for code in request.codes:
            df = fetcher.get_daily_kline(code, start_date=request.start_date)
            if len(df) >= 60:
                stock_data[code] = df
        
        df_result = predictor.get_top_recommendations(
            stock_data,
            top_n=len(request.codes)
        )
        
        return {"predictions": df_result.to_dict("records")}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/hotspots")
async def get_hotspots():
    """获取热门概念板块"""
    try:
        df = fetcher.get_sector_hotspots()
        if len(df) == 0:
            return {"hotspots": []}
        
        hotspots = df.head(10).to_dict("records")
        return {"hotspots": hotspots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/info")
async def model_info():
    """获取模型信息"""
    return {
        "model_ready": predictor.is_ready,
        "models": {
            "xgboost": predictor.xgb_model is not None,
            "lightgbm": predictor.lgb_model is not None,
        },
        "config": {
            "forecast_days": app_config.ml.forecast_days,
            "threshold_up": app_config.ml.threshold_up,
            "threshold_down": app_config.ml.threshold_down,
        }
    }


@app.get("/api/v1/stock_list")
async def stock_list(
    pool: str = Query("all", description="all/hs300/zz500")
):
    """获取股票列表"""
    try:
        codes = fetcher.get_stock_pool(pool)
        return {"pool": pool, "count": len(codes), "codes": codes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 启动入口 ==========

if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动选股神 API 服务...")
    uvicorn.run(
        app,
        host=app_config.api_host,
        port=app_config.api_port,
        log_level=app_config.log_level.lower()
    )
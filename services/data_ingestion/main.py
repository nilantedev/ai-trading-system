#!/usr/bin/env python3
"""
AI Trading System - Data Ingestion Service
Handles real-time market data, news, and social sentiment collection.
"""

import asyncio
import sys
from pathlib import Path

# Add shared library to path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

from trading_common import get_logger, get_settings, MarketData, NewsItem, SocialSentiment
from trading_common.cache import get_trading_cache
from trading_common.database import get_redis_client

# Import new services
from market_data_service import get_market_data_service
from news_service import get_news_service
from reference_data_service import get_reference_data_service
from data_validation_service import get_data_validation_service

logger = get_logger(__name__)
settings = get_settings()

# Global state
cache_client = None
redis_client = None
market_data_svc = None
news_svc = None
reference_svc = None
validation_svc = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global cache_client, redis_client, market_data_svc, news_svc, reference_svc, validation_svc
    
    logger.info("Starting Data Ingestion Service")
    
    # Initialize connections
    try:
        cache_client = get_trading_cache()
        redis_client = get_redis_client()
        logger.info("Connected to cache and database services")
        
        # Initialize services
        market_data_svc = await get_market_data_service()
        news_svc = await get_news_service()
        reference_svc = await get_reference_data_service()
        validation_svc = await get_data_validation_service()
        
        logger.info("All data ingestion services initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup
    if cache_client:
        await cache_client.close()
    if redis_client:
        await redis_client.close()
    
    # Stop services
    if market_data_svc:
        await market_data_svc.stop()
    if news_svc:
        await news_svc.stop()
    if reference_svc:
        await reference_svc.stop()
    if validation_svc:
        await validation_svc.stop()
        
    logger.info("Data Ingestion Service stopped")


app = FastAPI(
    title="AI Trading System - Data Ingestion Service",
    description="Handles real-time market data, news, and social sentiment collection",
    version="1.0.0-dev",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "data-ingestion",
        "timestamp": datetime.utcnow().isoformat(),
        "connections": {
            "cache": cache_client is not None,
            "redis": redis_client is not None
        }
    }


@app.get("/status")
async def get_status():
    """Get service status and statistics."""
    try:
        # Get health from all services
        service_health = {}
        
        if market_data_svc:
            service_health["market_data"] = await market_data_svc.get_service_health()
        if news_svc:
            service_health["news"] = await news_svc.get_service_health()
        if reference_svc:
            service_health["reference_data"] = await reference_svc.get_service_health()
        if validation_svc:
            service_health["validation"] = await validation_svc.get_service_health()
            
        return {
            "service": "data-ingestion",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "services": service_health,
            "data_sources": {
                "alpaca": bool(os.getenv("ALPACA_API_KEY")),
                "polygon": bool(os.getenv("POLYGON_API_KEY")),
                "news_api": bool(os.getenv("NEWS_API_KEY")),
                "reddit": bool(os.getenv("REDDIT_CLIENT_ID")),
                "finnhub": bool(os.getenv("FINNHUB_API_KEY")),
                "alpha_vantage": bool(os.getenv("ALPHA_VANTAGE_API_KEY"))
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service status")


@app.post("/ingest/market-data")
async def ingest_market_data(symbol: str):
    """Trigger market data ingestion for a symbol."""
    try:
        logger.info(f"Starting market data ingestion for {symbol}")
        
        # For Phase 2, this is a placeholder that would:
        # 1. Connect to Alpaca/Polygon API
        # 2. Fetch real-time data for symbol
        # 3. Store in QuestDB via cache layer
        # 4. Publish to message queue for downstream processing
        
        sample_data = MarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            open=100.0,
            high=102.5,
            low=99.8,
            close=101.2,
            volume=1000000,
            timeframe="1min",
            data_source="sample"
        )
        
        # Store in cache (would be real implementation)
        if cache_client:
            await cache_client.set_market_data(sample_data)
            
        logger.info(f"Successfully ingested market data for {symbol}")
        return {
            "status": "success",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Market data ingestion completed"
        }
        
    except Exception as e:
        logger.error(f"Market data ingestion failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/news")
async def ingest_news(query: str = "SPY TSLA AAPL"):
    """Trigger news ingestion for given query."""
    try:
        logger.info(f"Starting news ingestion for query: {query}")
        
        # Placeholder for Phase 2 - would implement:
        # 1. Connect to NewsAPI/other news sources
        # 2. Fetch relevant financial news
        # 3. Run sentiment analysis
        # 4. Store processed news with sentiment scores
        
        sample_news = NewsItem(
            title="Market Update: Tech Stocks Rally",
            content="Technology stocks showed strong performance...",
            source="Financial Times",
            published_at=datetime.utcnow(),
            url="https://example.com/news/123",
            sentiment_score=0.7,
            relevance_score=0.9,
            symbols=["SPY", "TSLA", "AAPL"]
        )
        
        if cache_client:
            await cache_client.set_news_item(sample_news)
            
        logger.info(f"Successfully ingested news for query: {query}")
        return {
            "status": "success", 
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "News ingestion completed"
        }
        
    except Exception as e:
        logger.error(f"News ingestion failed for query {query}: {e}")
        raise HTTPException(status_code=500, detail=f"News ingestion failed: {str(e)}")


@app.get("/data/recent/{symbol}")
async def get_recent_data(symbol: str, hours: int = 1):
    """Get recent market data for a symbol."""
    try:
        if not cache_client:
            raise HTTPException(status_code=503, detail="Cache service unavailable")
            
        # Placeholder - would query actual cached data
        logger.info(f"Retrieving recent data for {symbol} (last {hours} hours)")
        
        return {
            "symbol": symbol,
            "timeframe": f"last_{hours}_hours",
            "data_points": 0,  # Would return actual data
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"No data available yet - service in development mode"
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Data retrieval failed: {str(e)}")


# New comprehensive endpoints

@app.post("/market-data/quote/{symbol}")
async def get_real_time_quote(symbol: str):
    """Get real-time quote for a symbol."""
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    
    try:
        quote = await market_data_svc.get_real_time_quote(symbol.upper())
        if quote:
            # Validate the data
            if validation_svc:
                validation_results = await validation_svc.validate_market_data(quote)
                has_errors = any(r.severity.value == "error" for r in validation_results)
                
                return {
                    "data": {
                        "symbol": quote.symbol,
                        "timestamp": quote.timestamp.isoformat(),
                        "open": quote.open,
                        "high": quote.high,
                        "low": quote.low,
                        "close": quote.close,
                        "volume": quote.volume,
                        "source": quote.data_source
                    },
                    "validation": {
                        "valid": not has_errors,
                        "issues": len(validation_results),
                        "details": [{"severity": r.severity.value, "message": r.message} for r in validation_results]
                    }
                }
            else:
                return {"data": quote, "validation": {"valid": True, "issues": 0}}
        else:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/market-data/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    timeframe: str = "1min",
    hours_back: int = 24,
    limit: int = 1000
):
    """Get historical market data."""
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        data = await market_data_svc.get_historical_data(
            symbol.upper(), timeframe, start_time, end_time, limit
        )
        
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "count": len(data),
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "data": [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume
                } for bar in data
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/news/collect")
async def collect_financial_news(
    symbols: Optional[List[str]] = None,
    hours_back: int = 1,
    max_articles: int = 50
):
    """Collect financial news."""
    if not news_svc:
        raise HTTPException(status_code=503, detail="News service not available")
    
    try:
        news_items = await news_svc.collect_financial_news(symbols, hours_back, max_articles)
        
        return {
            "status": "success",
            "symbols": symbols,
            "articles_collected": len(news_items),
            "hours_back": hours_back,
            "timestamp": datetime.utcnow().isoformat(),
            "articles": [
                {
                    "title": item.title,
                    "source": item.source,
                    "published_at": item.published_at.isoformat(),
                    "sentiment_score": item.sentiment_score,
                    "relevance_score": item.relevance_score,
                    "symbols": item.symbols,
                    "url": item.url
                } for item in news_items
            ]
        }
    except Exception as e:
        logger.error(f"Failed to collect news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reference/security/{symbol}")
async def get_security_info(symbol: str, refresh: bool = False):
    """Get security reference information."""
    if not reference_svc:
        raise HTTPException(status_code=503, detail="Reference data service not available")
    
    try:
        info = await reference_svc.get_security_info(symbol.upper(), refresh)
        if info:
            return {
                "symbol": info.symbol,
                "name": info.name,
                "exchange": info.exchange,
                "sector": info.sector,
                "industry": info.industry,
                "market_cap": info.market_cap,
                "currency": info.currency,
                "country": info.country,
                "active": info.active,
                "last_updated": info.last_updated.isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"No information found for {symbol}")
    except Exception as e:
        logger.error(f"Failed to get security info for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reference/watchlist")
async def get_watchlist():
    """Get current watchlist symbols."""
    if not reference_svc:
        raise HTTPException(status_code=503, detail="Reference data service not available")
    
    try:
        symbols = await reference_svc.get_watchlist_symbols()
        return {
            "symbols": symbols,
            "count": len(symbols),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reference/watchlist/add")
async def add_to_watchlist(symbols: List[str]):
    """Add symbols to watchlist."""
    if not reference_svc:
        raise HTTPException(status_code=503, detail="Reference data service not available")
    
    try:
        symbols_upper = [s.upper() for s in symbols]
        success = await reference_svc.add_to_watchlist(symbols_upper)
        if success:
            return {
                "status": "success",
                "added_symbols": symbols_upper,
                "count": len(symbols_upper),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update watchlist")
    except Exception as e:
        logger.error(f"Failed to add symbols to watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validation/quality-metrics/{symbol}")
async def get_data_quality_metrics(symbol: str, hours_back: int = 24):
    """Get data quality metrics for a symbol."""
    if not validation_svc:
        raise HTTPException(status_code=503, detail="Validation service not available")
    
    try:
        metrics = await validation_svc.calculate_data_quality_metrics(symbol.upper(), hours_back)
        
        return {
            "symbol": symbol.upper(),
            "period_hours": hours_back,
            "metrics": {
                "total_records": metrics.total_records,
                "valid_records": metrics.valid_records,
                "invalid_records": metrics.invalid_records,
                "completeness_score": round(metrics.completeness_score, 3),
                "accuracy_score": round(metrics.accuracy_score, 3),
                "timeliness_score": round(metrics.timeliness_score, 3),
                "consistency_score": round(metrics.consistency_score, 3),
                "overall_score": round(metrics.overall_score, 3)
            },
            "issues": [
                {
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "field": issue.field,
                    "suggested_fix": issue.suggested_fix
                } for issue in metrics.issues[-10:]  # Last 10 issues
            ],
            "timestamp": metrics.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get quality metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream/start")
async def start_data_stream(background_tasks: BackgroundTasks, symbols: List[str]):
    """Start real-time data streaming for symbols."""
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    
    try:
        symbols_upper = [s.upper() for s in symbols]
        
        # Start background streaming task
        background_tasks.add_task(stream_market_data, symbols_upper)
        
        return {
            "status": "started",
            "symbols": symbols_upper,
            "message": "Real-time data streaming started",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to start data stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_market_data(symbols: List[str]):
    """Background task for streaming market data."""
    try:
        if market_data_svc:
            logger.info(f"Starting data stream for {len(symbols)} symbols")
            async for quote in market_data_svc.stream_real_time_data(symbols):
                logger.info(f"Streamed data for {quote.symbol}: ${quote.close}")
                # Data is automatically cached and published to message system
    except Exception as e:
        logger.error(f"Data streaming error: {e}")


if __name__ == "__main__":
    logger.info("Starting Data Ingestion Service...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
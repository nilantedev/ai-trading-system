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

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

from trading_common import get_logger, get_settings, MarketData, NewsItem, SocialSentiment
from trading_common.cache import get_trading_cache
from trading_common.database import get_redis_client

logger = get_logger(__name__)
settings = get_settings()

# Global state
cache_client = None
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global cache_client, redis_client
    
    logger.info("Starting Data Ingestion Service")
    
    # Initialize connections
    try:
        cache_client = get_trading_cache()
        redis_client = get_redis_client()
        logger.info("Connected to cache and database services")
    except Exception as e:
        logger.error(f"Failed to initialize connections: {e}")
        raise
    
    yield
    
    # Cleanup
    if cache_client:
        await cache_client.close()
    if redis_client:
        await redis_client.close()
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
        # Get some basic stats from cache
        market_data_count = 0
        news_count = 0
        
        if cache_client:
            # This would normally query actual cached data
            market_data_count = await cache_client.count("market_data:*") or 0
            news_count = await cache_client.count("news:*") or 0
            
        return {
            "service": "data-ingestion",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": {
                "market_data_cached": market_data_count,
                "news_items_cached": news_count,
                "uptime_seconds": 0  # Would track actual uptime
            },
            "data_sources": {
                "alpaca": bool(os.getenv("ALPACA_API_KEY")),
                "polygon": bool(os.getenv("POLYGON_API_KEY")),
                "news_api": bool(os.getenv("NEWS_API_KEY")),
                "reddit": bool(os.getenv("REDDIT_CLIENT_ID"))
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


if __name__ == "__main__":
    logger.info("Starting Data Ingestion Service...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
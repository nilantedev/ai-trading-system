"""
Production Dashboard Data API
Provides comprehensive real-time data for business and admin dashboards
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import logging
import os

router = APIRouter(prefix="/api/dashboard", tags=["dashboard-data"])
logger = logging.getLogger(__name__)

# Import dependencies
try:
    from trading_common.database_manager import get_database_manager
    from api.auth import get_current_user_cookie_or_bearer
except ImportError as e:
    logger.warning(f"Import error: {e}")
    get_database_manager = None
    get_current_user_cookie_or_bearer = None


@router.get("/watchlist/all")
async def get_full_watchlist():
    """Get complete watchlist with real-time status"""
    try:
        import redis.asyncio as aioredis
        
        # Use REDIS_URL from environment (same pattern as auth.py)
        redis_url = os.getenv("REDIS_URL", "redis://trading-redis:6379/0")
        redis_password = os.getenv("REDIS_PASSWORD", "")
        
        # Connect to Redis
        redis_client = await aioredis.from_url(
            redis_url,
            password=redis_password,
            decode_responses=True
        )
        
        # Get all symbols from watchlist
        symbols = await redis_client.smembers("watchlist")
        symbol_list = sorted(list(symbols))
        await redis_client.close()
        
        return {
            "total": len(symbol_list),
            "symbols": symbol_list,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_status": "continuous"
        }
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}", exc_info=True)
        return {"total": 0, "symbols": [], "error": str(e)}


@router.get("/market/summary")
async def get_market_summary():
    """Get real-time market summary across all symbols"""
    try:
        import aiohttp
        
        # Query QuestDB via HTTP
        questdb_host = "http://trading-questdb:9000"
        async with aiohttp.ClientSession() as session:
            # Count total bars
            async with session.get(f"{questdb_host}/exec?query=SELECT count(*) as cnt FROM market_data") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bar_count = data['dataset'][0][0] if data.get('dataset') and len(data['dataset']) > 0 else 0
                else:
                    bar_count = 0
            
            # Get latest timestamp
            async with session.get(f"{questdb_host}/exec?query=SELECT max(timestamp) as latest FROM market_data") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    latest_ts = data['dataset'][0][0] if data.get('dataset') and len(data['dataset']) > 0 else None
                else:
                    latest_ts = None
            
            # Get unique symbols count
            async with session.get(f"{questdb_host}/exec?query=SELECT count(*) FROM (SELECT DISTINCT symbol FROM market_data)") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    unique_symbols = data['dataset'][0][0] if data.get('dataset') and len(data['dataset']) > 0 else 0
                else:
                    unique_symbols = 0
            
            return {
                "total_bars": bar_count,
                "unique_symbols": unique_symbols,
                "latest_timestamp": latest_ts,
                "bars_today": bar_count,  # Simplified for now
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching market summary: {e}", exc_info=True)
        return {"error": str(e), "total_bars": 0}


@router.get("/symbol/{symbol}/latest")
async def get_symbol_latest_data(symbol: str):
    """Get latest data for a specific symbol"""
    try:
        db = await get_database_manager()
        
        # Get latest bar data from QuestDB
        async with db.get_questdb() as conn:
            latest = await conn.fetchrow(f"""
                SELECT * FROM market_data 
                WHERE symbol = '{symbol.upper()}' 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            if latest:
                return {
                    "symbol": symbol.upper(),
                    "timestamp": str(latest['timestamp']),
                    "open": float(latest['open']) if latest['open'] else None,
                    "high": float(latest['high']) if latest['high'] else None,
                    "low": float(latest['low']) if latest['low'] else None,
                    "close": float(latest['close']) if latest['close'] else None,
                    "volume": int(latest['volume']) if latest['volume'] else None,
                }
            else:
                return {"symbol": symbol.upper(), "error": "No data available"}
                
    except Exception as e:
        logger.error(f"Error fetching symbol data for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


@router.get("/symbol/{symbol}/history")
async def get_symbol_history(symbol: str, days: int = 30):
    """Get historical data for charting"""
    try:
        db = await get_database_manager()
        
        async with db.get_questdb() as conn:
            history = await conn.fetch(f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM market_data 
                WHERE symbol = '{symbol.upper()}' 
                AND timestamp > dateadd('d', -{days}, now()) 
                ORDER BY timestamp DESC
            """)
            
            if history:
                data = []
                for row in history:
                    data.append({
                        "timestamp": str(row['timestamp']),
                        "open": float(row['open']) if row['open'] else None,
                        "high": float(row['high']) if row['high'] else None,
                        "low": float(row['low']) if row['low'] else None,
                        "close": float(row['close']) if row['close'] else None,
                        "volume": int(row['volume']) if row['volume'] else None,
                    })
                
                return {
                    "symbol": symbol.upper(),
                    "days": days,
                    "bars": len(data),
                    "data": data
                }
            else:
                return {"symbol": symbol.upper(), "bars": 0, "data": []}
                
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e), "data": []}


@router.get("/services/health")
async def get_all_services_health():
    """Get health status of all microservices"""
    import aiohttp
    
    services = {
        "ml": "http://trading-ml:8001/health",
        "data-ingestion": "http://trading-data-ingestion:8002/health",
        "signal-generator": "http://trading-signal-generator:8003/health",
        "execution": "http://trading-execution:8004/health",
        "risk-monitor": "http://trading-risk-monitor:8005/health",
        "strategy-engine": "http://trading-strategy-engine:8006/health",
        "backtesting": "http://trading-backtesting:8007/health",
    }
    
    results = {}
    
    async with aiohttp.ClientSession() as session:
        for name, url in services.items():
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results[name] = {
                            "status": data.get("status", "unknown"),
                            "healthy": True
                        }
                    else:
                        results[name] = {"status": "error", "healthy": False}
            except Exception as e:
                results[name] = {"status": "unreachable", "healthy": False, "error": str(e)}
    
    # Count healthy services
    healthy_count = sum(1 for v in results.values() if v.get("healthy"))
    total_count = len(results)
    
    return {
        "services": results,
        "healthy": healthy_count,
        "total": total_count,
        "all_healthy": healthy_count == total_count,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/system/metrics")
async def get_system_metrics():
    """Get real-time system metrics"""
    try:
        # Get system metrics using docker stats
        import subprocess
        
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        containers = []
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) == 3:
                        containers.append({
                            "name": parts[0],
                            "cpu": parts[1],
                            "memory": parts[2]
                        })
        
        return {
            "containers": containers,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching system metrics: {e}")
        return {"error": str(e), "containers": []}


@router.get("/news/recent")
async def get_recent_news(limit: int = 50):
    """Get recent news articles"""
    try:
        db = await get_database_manager()
        
        async with db.get_questdb() as conn:
            news = await conn.fetch(f"""
                SELECT timestamp, symbol, title, summary, sentiment_score
                FROM news
                ORDER BY timestamp DESC
                LIMIT {limit}
            """)
            
            articles = []
            for row in news:
                articles.append({
                    "timestamp": str(row['timestamp']),
                    "symbol": row['symbol'],
                    "title": row['title'],
                    "summary": row['summary'],
                    "sentiment": float(row['sentiment_score']) if row['sentiment_score'] else 0
                })
            
            return {
                "total": len(articles),
                "articles": articles,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return {"total": 0, "articles": [], "error": str(e)}


@router.get("/strategies/performance")
async def get_strategy_performance():
    """Get performance metrics for all active strategies"""
    try:
        # This will be populated by the strategy engine
        # For now, return structure
        return {
            "strategies": [
                {"name": "momentum", "status": "active", "signals_today": 0},
                {"name": "mean_reversion", "status": "active", "signals_today": 0},
                {"name": "stat_arb", "status": "active", "signals_today": 0},
                {"name": "market_making", "status": "active", "signals_today": 0},
                {"name": "vol_arb", "status": "active", "signals_today": 0},
                {"name": "index_arb", "status": "active", "signals_today": 0},
                {"name": "trend_following", "status": "active", "signals_today": 0},
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching strategy performance: {e}")
        return {"strategies": [], "error": str(e)}


@router.get("/processing/stats")
async def get_processing_stats():
    """Get real-time data processing statistics"""
    try:
        import redis.asyncio as aioredis
        import aiohttp
        import os
        
        # Direct Redis connection
        redis_password = os.getenv("REDIS_PASSWORD", "")
        redis_host = os.getenv("REDIS_HOST", "trading-redis")
        redis_url = f"redis://{redis_host}:6379/0"
        
        redis_client = await aioredis.from_url(
            redis_url, 
            decode_responses=True,
            password=redis_password
        )
        
        # Get watchlist size
        watchlist_size = await redis_client.scard("watchlist")
        await redis_client.close()
        
        # Get recent activity from QuestDB via HTTP
        questdb_host = "http://trading-questdb:9000"
        async with aiohttp.ClientSession() as session:
            # Bars in last hour
            try:
                async with session.get(f"{questdb_host}/exec?query=SELECT count(*) as cnt FROM market_data WHERE timestamp > now() - 3600000000L", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        recent_bars = data.get('dataset', [[0]])[0][0] if data.get('dataset') else 0
                    else:
                        recent_bars = 0
            except:
                recent_bars = 0
            
            return {
                "watchlist_symbols": watchlist_size,
                "bars_last_hour": recent_bars,
                "news_last_hour": 0,  # Simplified for now
                "processing_mode": "continuous",
                "interval_seconds": 30,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching processing stats: {e}", exc_info=True)
        return {"error": str(e)}


@router.get("/data/comprehensive")
async def get_comprehensive_data_inventory():
    """Get complete data inventory across ALL systems (QuestDB, PostgreSQL, Weaviate, Redis)"""
    try:
        import aiohttp
        
        inventory = {}
        
        # QuestDB Data
        questdb_host = "http://trading-questdb:9000"
        async with aiohttp.ClientSession() as session:
            questdb_tables = {
                "market_data": "Market OHLCV bars (20 years)",
                "daily_bars": "Daily aggregated bars",
                "social_signals": "Social media sentiment",
                "options_data": "Options chain data",
                "news_events": "News articles"
            }
            
            questdb_data = {}
            for table, desc in questdb_tables.items():
                try:
                    async with session.get(f"{questdb_host}/exec?query=SELECT count(*) FROM {table}", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            count = data['dataset'][0][0] if data.get('dataset') else 0
                            questdb_data[table] = {"count": count, "description": desc}
                except:
                    questdb_data[table] = {"count": 0, "description": desc, "error": "timeout"}
            
            inventory["questdb"] = questdb_data
        
        # PostgreSQL Data (using credentials from env)
        try:
            db = await get_database_manager()
            async with db.get_postgres() as conn:
                from sqlalchemy import text
                result = await conn.execute(text("""
                    SELECT relname, n_live_tup 
                    FROM pg_stat_user_tables 
                    WHERE n_live_tup > 0 
                    ORDER BY n_live_tup DESC
                """))
                pg_data = {row[0]: {"count": row[1], "description": f"PostgreSQL table"} for row in result}
                inventory["postgres"] = pg_data
        except Exception as e:
            inventory["postgres"] = {"error": str(e)}
        
        # Redis Data
        try:
            import redis.asyncio as aioredis
            redis_url = os.getenv("REDIS_URL", "redis://trading-redis:6379/0")
            redis_password = os.getenv("REDIS_PASSWORD", "")
            redis_client = await aioredis.from_url(redis_url, password=redis_password, decode_responses=True)
            
            watchlist_size = await redis_client.scard("watchlist")
            dbsize = await redis_client.dbsize()
            
            inventory["redis"] = {
                "watchlist": {"count": watchlist_size, "description": "Active trading symbols"},
                "total_keys": {"count": dbsize, "description": "Total Redis keys"}
            }
            await redis_client.close()
        except Exception as e:
            inventory["redis"] = {"error": str(e)}
        
        # Weaviate Vector Database
        try:
            weaviate_classes = ["EquityBar", "NewsArticle", "SocialSentiment", "OptionContract"]
            async with aiohttp.ClientSession() as session:
                weaviate_data = {}
                for cls in weaviate_classes:
                    try:
                        async with session.get(f"http://trading-weaviate:8080/v1/objects?class={cls}&limit=1", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                count = data.get('totalResults', 0)
                                weaviate_data[cls] = {"count": count, "description": f"Vector embeddings for {cls}"}
                    except:
                        weaviate_data[cls] = {"count": 0, "description": f"Vector embeddings for {cls}"}
                
                inventory["weaviate"] = weaviate_data
        except Exception as e:
            inventory["weaviate"] = {"error": str(e)}
        
        # Calculate totals
        total_records = sum(
            item.get("count", 0) 
            for source in inventory.values() 
            if isinstance(source, dict)
            for item in source.values()
            if isinstance(item, dict) and "count" in item
        )
        
        return {
            "inventory": inventory,
            "total_records": total_records,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Error fetching comprehensive data: {e}", exc_info=True)
        return {"error": str(e), "inventory": {}}


@router.get("/social/recent")
async def get_recent_social_signals(limit: int = 100):
    """Get recent social media signals"""
    try:
        import aiohttp
        questdb_host = "http://trading-questdb:9000"
        
        async with aiohttp.ClientSession() as session:
            query = f"SELECT symbol, source, ts, sentiment, engagement, content FROM social_signals ORDER BY ts DESC LIMIT {limit}"
            async with session.get(f"{questdb_host}/exec?query={query.replace(' ', '+')}", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    signals = []
                    if data.get('dataset'):
                        for row in data['dataset']:
                            signals.append({
                                "symbol": row[0],
                                "source": row[1],
                                "timestamp": row[2],
                                "sentiment": row[3],
                                "engagement": row[4],
                                "content": row[5][:200] if row[5] else ""
                            })
                    
                    return {
                        "total": len(signals),
                        "signals": signals,
                        "timestamp": datetime.utcnow().isoformat()
                    }
        
        return {"total": 0, "signals": []}
    except Exception as e:
        logger.error(f"Error fetching social signals: {e}", exc_info=True)
        return {"total": 0, "signals": [], "error": str(e)}


@router.get("/options/flow")
async def get_options_flow(limit: int = 50):
    """Get recent options flow data"""
    try:
        import aiohttp
        questdb_host = "http://trading-questdb:9000"
        
        async with aiohttp.ClientSession() as session:
            query = f"SELECT symbol, timestamp, strike, expiry, option_type, volume, open_interest, implied_volatility FROM options_data ORDER BY timestamp DESC LIMIT {limit}"
            async with session.get(f"{questdb_host}/exec?query={query.replace(' ', '+')}", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    options = []
                    if data.get('dataset'):
                        for row in data['dataset']:
                            options.append({
                                "symbol": row[0],
                                "timestamp": row[1],
                                "strike": row[2],
                                "expiry": row[3],
                                "type": row[4],
                                "volume": row[5],
                                "open_interest": row[6],
                                "iv": row[7]
                            })
                    
                    return {
                        "total": len(options),
                        "options": options,
                        "timestamp": datetime.utcnow().isoformat()
                    }
        
        return {"total": 0, "options": []}
    except Exception as e:
        logger.error(f"Error fetching options flow: {e}", exc_info=True)
        return {"total": 0, "options": [], "error": str(e)}

"""Redis caching utilities and data structures for trading system."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from decimal import Decimal
import hashlib

import redis.asyncio as redis
from pydantic import BaseModel

from .database import get_redis_client
from .models import MarketData, TradingSignal, PortfolioSnapshot, Position
from .logging import get_logger

logger = get_logger(__name__)


class CacheKeyGenerator:
    """Generate consistent cache keys for different data types."""
    
    @staticmethod
    def market_data(symbol: str, timeframe: str = "1m") -> str:
        """Generate market data cache key."""
        return f"market_data:{symbol.upper()}:{timeframe}"
    
    @staticmethod
    def market_data_latest(symbol: str) -> str:
        """Generate latest market data cache key."""
        return f"market_data:{symbol.upper()}:latest"
    
    @staticmethod
    def portfolio_snapshot(portfolio_id: str) -> str:
        """Generate portfolio snapshot cache key."""
        return f"portfolio:{portfolio_id}:snapshot"
    
    @staticmethod
    def position(portfolio_id: str, symbol: str) -> str:
        """Generate position cache key."""
        return f"position:{portfolio_id}:{symbol.upper()}"
    
    @staticmethod
    def trading_signal(signal_id: str) -> str:
        """Generate trading signal cache key."""
        return f"signal:{signal_id}"
    
    @staticmethod
    def active_signals(symbol: Optional[str] = None) -> str:
        """Generate active signals cache key."""
        if symbol:
            return f"signals:active:{symbol.upper()}"
        return "signals:active"
    
    @staticmethod
    def technical_indicator(symbol: str, indicator: str, timeframe: str) -> str:
        """Generate technical indicator cache key."""
        return f"indicator:{symbol.upper()}:{indicator}:{timeframe}"
    
    @staticmethod
    def system_status(component: str) -> str:
        """Generate system status cache key."""
        return f"system_status:{component}"
    
    @staticmethod
    def rate_limit(api_name: str, identifier: str) -> str:
        """Generate rate limiting cache key."""
        return f"rate_limit:{api_name}:{identifier}"


class TradingCache:
    """High-level trading data cache with intelligent expiration."""
    
    def __init__(self):
        self.db_manager = None
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize cache connection."""
        self.db_manager = await get_database_manager()
        logger.info("Trading cache initialized")
    
    async def _get_redis(self):
        """Get Redis connection."""
        if not self.db_manager:
            await self.initialize()
        return self.db_manager.get_redis()
    
    # Market Data Caching
    async def cache_market_data(self, data: MarketData, timeframe: str = "1m", ttl: int = 300):
        """Cache market data with automatic key generation."""
        try:
            async with await self._get_redis() as r:
                # Cache latest data
                latest_key = CacheKeyGenerator.market_data_latest(data.symbol)
                await r.setex(latest_key, ttl, data.json())
                
                # Cache in time-series structure
                ts_key = CacheKeyGenerator.market_data(data.symbol, timeframe)
                score = data.timestamp.timestamp()
                await r.zadd(ts_key, {data.json(): score})
                await r.expire(ts_key, ttl * 2)  # Longer TTL for time series
                
                self._cache_stats['sets'] += 1
                logger.debug(f"Cached market data for {data.symbol}")
                
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to cache market data: {e}")
    
    async def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest cached market data."""
        try:
            async with await self._get_redis() as r:
                key = CacheKeyGenerator.market_data_latest(symbol)
                data = await r.get(key)
                
                if data:
                    self._cache_stats['hits'] += 1
                    return MarketData.parse_raw(data)
                else:
                    self._cache_stats['misses'] += 1
                    return None
                    
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to get cached market data: {e}")
            return None
    
    async def get_market_data_series(self, symbol: str, timeframe: str = "1m", 
                                   limit: int = 100) -> List[MarketData]:
        """Get cached market data time series."""
        try:
            async with await self._get_redis() as r:
                key = CacheKeyGenerator.market_data(symbol, timeframe)
                # Get recent entries (highest scores = most recent)
                data_list = await r.zrevrange(key, 0, limit - 1)
                
                if data_list:
                    self._cache_stats['hits'] += 1
                    return [MarketData.parse_raw(item) for item in data_list]
                else:
                    self._cache_stats['misses'] += 1
                    return []
                    
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to get market data series: {e}")
            return []
    
    # Portfolio Caching
    async def cache_portfolio_snapshot(self, snapshot: PortfolioSnapshot, ttl: int = 60):
        """Cache portfolio snapshot."""
        try:
            async with await self._get_redis() as r:
                key = CacheKeyGenerator.portfolio_snapshot(snapshot.portfolio_id)
                await r.setex(key, ttl, snapshot.json())
                
                # Also cache individual positions
                for position in snapshot.positions:
                    pos_key = CacheKeyGenerator.position(snapshot.portfolio_id, position.symbol)
                    await r.setex(pos_key, ttl, position.json())
                
                self._cache_stats['sets'] += 1
                logger.debug(f"Cached portfolio snapshot for {snapshot.portfolio_id}")
                
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to cache portfolio snapshot: {e}")
    
    async def get_portfolio_snapshot(self, portfolio_id: str) -> Optional[PortfolioSnapshot]:
        """Get cached portfolio snapshot."""
        try:
            async with await self._get_redis() as r:
                key = CacheKeyGenerator.portfolio_snapshot(portfolio_id)
                data = await r.get(key)
                
                if data:
                    self._cache_stats['hits'] += 1
                    return PortfolioSnapshot.parse_raw(data)
                else:
                    self._cache_stats['misses'] += 1
                    return None
                    
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to get cached portfolio: {e}")
            return None
    
    async def get_position(self, portfolio_id: str, symbol: str) -> Optional[Position]:
        """Get cached position data."""
        try:
            async with await self._get_redis() as r:
                key = CacheKeyGenerator.position(portfolio_id, symbol)
                data = await r.get(key)
                
                if data:
                    self._cache_stats['hits'] += 1
                    return Position.parse_raw(data)
                else:
                    self._cache_stats['misses'] += 1
                    return None
                    
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to get cached position: {e}")
            return None
    
    # Trading Signal Caching
    async def cache_trading_signal(self, signal: TradingSignal, ttl: int = 3600):
        """Cache trading signal."""
        try:
            async with await self._get_redis() as r:
                # Individual signal cache
                key = CacheKeyGenerator.trading_signal(signal.id)
                await r.setex(key, ttl, signal.json())
                
                # Add to active signals set if active
                if signal.status.value == "active":
                    active_key = CacheKeyGenerator.active_signals(signal.symbol)
                    await r.sadd(active_key, signal.id)
                    await r.expire(active_key, ttl)
                    
                    # Global active signals
                    global_active = CacheKeyGenerator.active_signals()
                    await r.sadd(global_active, signal.id)
                    await r.expire(global_active, ttl)
                
                self._cache_stats['sets'] += 1
                logger.debug(f"Cached trading signal {signal.id}")
                
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to cache trading signal: {e}")
    
    async def get_active_signals(self, symbol: Optional[str] = None) -> List[TradingSignal]:
        """Get active trading signals."""
        try:
            async with await self._get_redis() as r:
                if symbol:
                    active_key = CacheKeyGenerator.active_signals(symbol)
                else:
                    active_key = CacheKeyGenerator.active_signals()
                
                signal_ids = await r.smembers(active_key)
                
                if not signal_ids:
                    self._cache_stats['misses'] += 1
                    return []
                
                # Get full signal data
                signals = []
                for signal_id in signal_ids:
                    key = CacheKeyGenerator.trading_signal(signal_id)
                    data = await r.get(key)
                    if data:
                        signals.append(TradingSignal.parse_raw(data))
                
                self._cache_stats['hits'] += 1
                return signals
                
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to get active signals: {e}")
            return []
    
    # Technical Indicators
    async def cache_technical_indicator(self, symbol: str, indicator: str, 
                                      timeframe: str, value: float, 
                                      metadata: Optional[Dict] = None, ttl: int = 300):
        """Cache technical indicator value."""
        try:
            async with await self._get_redis() as r:
                key = CacheKeyGenerator.technical_indicator(symbol, indicator, timeframe)
                
                indicator_data = {
                    'value': value,
                    'timestamp': datetime.utcnow().isoformat(),
                    'metadata': metadata or {}
                }
                
                await r.setex(key, ttl, json.dumps(indicator_data))
                self._cache_stats['sets'] += 1
                
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to cache technical indicator: {e}")
    
    async def get_technical_indicator(self, symbol: str, indicator: str, 
                                    timeframe: str) -> Optional[Dict[str, Any]]:
        """Get cached technical indicator."""
        try:
            async with await self._get_redis() as r:
                key = CacheKeyGenerator.technical_indicator(symbol, indicator, timeframe)
                data = await r.get(key)
                
                if data:
                    self._cache_stats['hits'] += 1
                    return json.loads(data)
                else:
                    self._cache_stats['misses'] += 1
                    return None
                    
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to get technical indicator: {e}")
            return None
    
    # System Status Caching
    async def set_system_status(self, component: str, status: Dict[str, Any], ttl: int = 300):
        """Set system component status."""
        try:
            async with await self._get_redis() as r:
                key = CacheKeyGenerator.system_status(component)
                status_data = {
                    **status,
                    'timestamp': datetime.utcnow().isoformat(),
                    'component': component
                }
                await r.setex(key, ttl, json.dumps(status_data))
                self._cache_stats['sets'] += 1
                
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to set system status: {e}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        try:
            async with await self._get_redis() as r:
                pattern = "system_status:*"
                keys = await r.keys(pattern)
                
                health = {}
                for key in keys:
                    component = key.split(":", 1)[1]
                    data = await r.get(key)
                    if data:
                        health[component] = json.loads(data)
                
                self._cache_stats['hits'] += 1
                return health
                
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Failed to get system health: {e}")
            return {}
    
    # Rate Limiting
    async def check_rate_limit(self, api_name: str, identifier: str, 
                             limit: int, window: int) -> Dict[str, Any]:
        """Check and update rate limit."""
        try:
            async with await self._get_redis() as r:
                key = CacheKeyGenerator.rate_limit(api_name, identifier)
                
                # Use sliding window rate limiting
                now = datetime.utcnow().timestamp()
                
                # Remove old entries
                await r.zremrangebyscore(key, 0, now - window)
                
                # Get current count
                current_count = await r.zcard(key)
                
                if current_count >= limit:
                    # Rate limited
                    ttl = await r.ttl(key)
                    return {
                        'allowed': False,
                        'current_count': current_count,
                        'limit': limit,
                        'reset_in': ttl
                    }
                
                # Add current request
                await r.zadd(key, {str(now): now})
                await r.expire(key, window)
                
                return {
                    'allowed': True,
                    'current_count': current_count + 1,
                    'limit': limit,
                    'reset_in': window
                }
                
        except Exception as e:
            self._cache_stats['errors'] += 1
            logger.error(f"Rate limit check failed: {e}")
            return {'allowed': True, 'current_count': 0, 'limit': limit, 'reset_in': window}
    
    # Cache Management
    async def clear_cache_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern."""
        try:
            async with await self._get_redis() as r:
                keys = await r.keys(pattern)
                if keys:
                    deleted = await r.delete(*keys)
                    logger.info(f"Cleared {deleted} cache entries matching {pattern}")
                    return deleted
                return 0
                
        except Exception as e:
            logger.error(f"Failed to clear cache pattern {pattern}: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self._cache_stats,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }
    
    async def flush_all_cache(self):
        """Flush all cache data (use with caution)."""
        try:
            async with await self._get_redis() as r:
                await r.flushdb()
                logger.warning("Flushed all cache data")
                
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")


# Global cache instance
_trading_cache: Optional[TradingCache] = None

async def get_trading_cache() -> TradingCache:
    """Get or create global trading cache."""
    global _trading_cache
    if _trading_cache is None:
        _trading_cache = TradingCache()
        await _trading_cache.initialize()
    return _trading_cache
"""Redis client and utilities."""

import json
import time
from typing import Optional, Any, Dict, List, Union
from functools import lru_cache

import redis.asyncio as aioredis
from redis.asyncio import Redis

from ..config import get_settings
from ..logging import get_logger
from ..exceptions import DatabaseError

logger = get_logger(__name__)


class RedisClient:
    """Async Redis client wrapper with trading-specific utilities."""
    
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client: Optional[Redis] = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_error=[ConnectionError],
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            # Test connection
            await self._client.ping()
            logger.info("Connected to Redis", url=self.redis_url)
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise DatabaseError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Disconnected from Redis")
    
    @property
    def client(self) -> Redis:
        """Get the Redis client instance."""
        if not self._client:
            raise DatabaseError("Redis client not connected")
        return self._client
    
    async def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a JSON value in Redis."""
        try:
            json_data = json.dumps(value, default=str)
            result = await self.client.set(key, json_data, ex=ttl)
            return bool(result)
        except Exception as e:
            logger.error("Failed to set JSON in Redis", key=key, error=str(e))
            raise DatabaseError(f"Redis set_json failed: {e}")
    
    async def get_json(self, key: str, default: Any = None) -> Any:
        """Get a JSON value from Redis."""
        try:
            data = await self.client.get(key)
            if data is None:
                return default
            return json.loads(data)
        except Exception as e:
            logger.error("Failed to get JSON from Redis", key=key, error=str(e))
            raise DatabaseError(f"Redis get_json failed: {e}")
    
    async def cache_market_data(
        self,
        symbol: str,
        data_type: str,
        data: Dict[str, Any],
        ttl: int = 60,
    ) -> None:
        """Cache market data with structured key."""
        key = f"market:{symbol}:{data_type}"
        await self.set_json(key, data, ttl)
    
    async def get_market_data(
        self,
        symbol: str,
        data_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cached market data."""
        key = f"market:{symbol}:{data_type}"
        return await self.get_json(key)
    
    async def cache_trading_signal(
        self,
        signal_id: str,
        signal_data: Dict[str, Any],
        ttl: int = 300,
    ) -> None:
        """Cache a trading signal."""
        key = f"signal:{signal_id}"
        await self.set_json(key, signal_data, ttl)
    
    async def get_trading_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get a cached trading signal."""
        key = f"signal:{signal_id}"
        return await self.get_json(key)
    
    async def set_circuit_breaker(self, service: str, duration: int = 300) -> None:
        """Set a circuit breaker for a service."""
        key = f"circuit_breaker:{service}"
        await self.client.setex(key, duration, "active")
    
    async def is_circuit_breaker_active(self, service: str) -> bool:
        """Check if circuit breaker is active for a service."""
        key = f"circuit_breaker:{service}"
        return bool(await self.client.exists(key))
    
    async def increment_error_count(self, service: str, window: int = 60) -> int:
        """Increment error count for a service with sliding window."""
        key = f"errors:{service}"
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        results = await pipe.execute()
        return int(results[0])
    
    async def get_error_count(self, service: str) -> int:
        """Get current error count for a service."""
        key = f"errors:{service}"
        count = await self.client.get(key)
        return int(count) if count else 0
    
    async def store_session(self, session_id: str, user_data: Dict[str, Any], ttl: int = 3600) -> None:
        """Store user session data."""
        key = f"session:{session_id}"
        await self.set_json(key, user_data, ttl)
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get user session data."""
        key = f"session:{session_id}"
        return await self.get_json(key)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete user session."""
        key = f"session:{session_id}"
        return bool(await self.client.delete(key))
    
    async def add_to_list(self, key: str, value: Any, max_length: Optional[int] = None) -> int:
        """Add item to a Redis list with optional max length."""
        json_value = json.dumps(value, default=str)
        pipe = self.client.pipeline()
        pipe.lpush(key, json_value)
        if max_length:
            pipe.ltrim(key, 0, max_length - 1)
        results = await pipe.execute()
        return int(results[0])
    
    async def get_list(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get items from a Redis list."""
        data = await self.client.lrange(key, start, end)
        return [json.loads(item) for item in data]
    
    async def get(self, key: str) -> Optional[str]:
        """Get a string value from Redis."""
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error("Failed to get from Redis", key=key, error=str(e))
            raise DatabaseError(f"Redis get failed: {e}")
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set a string value in Redis."""
        try:
            result = await self.client.set(key, value, ex=ttl)
            return bool(result)
        except Exception as e:
            logger.error("Failed to set in Redis", key=key, error=str(e))
            raise DatabaseError(f"Redis set failed: {e}")
    
    async def setex(self, key: str, ttl: int, value: str) -> bool:
        """Set a value with expiration time."""
        try:
            result = await self.client.setex(key, ttl, value)
            return bool(result)
        except Exception as e:
            logger.error("Failed to setex in Redis", key=key, error=str(e))
            raise DatabaseError(f"Redis setex failed: {e}")
    
    async def delete(self, key: str) -> int:
        """Delete a key from Redis."""
        try:
            return await self.client.delete(key)
        except Exception as e:
            logger.error("Failed to delete from Redis", key=key, error=str(e))
            raise DatabaseError(f"Redis delete failed: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return bool(await self.client.exists(key))
        except Exception as e:
            logger.error("Failed to check exists in Redis", key=key, error=str(e))
            raise DatabaseError(f"Redis exists failed: {e}")
    
    async def sadd(self, key: str, *values: str) -> int:
        """Add members to a set."""
        try:
            return await self.client.sadd(key, *values)
        except Exception as e:
            logger.error("Failed to sadd in Redis", key=key, error=str(e))
            raise DatabaseError(f"Redis sadd failed: {e}")
    
    async def srem(self, key: str, *values: str) -> int:
        """Remove members from a set."""
        try:
            return await self.client.srem(key, *values)
        except Exception as e:
            logger.error("Failed to srem in Redis", key=key, error=str(e))
            raise DatabaseError(f"Redis srem failed: {e}")
    
    async def smembers(self, key: str) -> set:
        """Get all members of a set."""
        try:
            return await self.client.smembers(key)
        except Exception as e:
            logger.error("Failed to smembers in Redis", key=key, error=str(e))
            raise DatabaseError(f"Redis smembers failed: {e}")
    
    async def close(self) -> None:
        """Close the Redis connection (alias for disconnect)."""
        await self.disconnect()

    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check."""
        try:
            start_time = time.time()
            await self.client.ping()
            latency = (time.time() - start_time) * 1000
            
            info = await self.client.info()
            
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "redis_version": info.get("redis_version", "unknown"),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


@lru_cache()
def get_redis_client() -> RedisClient:
    """Get cached Redis client instance."""
    settings = get_settings()
    return RedisClient(settings.database.redis_url)
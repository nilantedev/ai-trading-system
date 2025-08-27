#!/usr/bin/env python3
"""
Redis-based Rate Limiter for AI Trading System API
Provides distributed rate limiting with sliding window algorithm.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request
import aioredis
import os
import json

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """Redis-based distributed rate limiter using sliding window algorithm."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.redis: Optional[aioredis.Redis] = None
        self.connected = False
        self.degraded_mode = False  # True when operating with relaxed safeguards due to backend outage
        self._degraded_last_log: float = 0.0
        
        # Default rate limit settings (can be overridden)
        self.default_requests_per_minute = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "100"))
        self.default_burst = int(os.getenv("RATE_LIMIT_BURST", "20"))
        
        # Rate limit configurations by endpoint type
        self.rate_limits = {
            "default": {"requests": self.default_requests_per_minute, "window": 60, "burst": self.default_burst},
            "auth": {"requests": 10, "window": 60, "burst": 5},  # Stricter for auth endpoints
            "websocket": {"requests": 50, "window": 60, "burst": 10},  # WebSocket connections
            "market_data": {"requests": 200, "window": 60, "burst": 50},  # Higher for market data
            "trading": {"requests": 30, "window": 60, "burst": 5},  # Conservative for trading
            "admin": {"requests": 500, "window": 60, "burst": 100},  # Higher for admin users
        }
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            if self.redis_password:
                self.redis = aioredis.from_url(
                    self.redis_url, 
                    password=self.redis_password,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            else:
                self.redis = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8", 
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            
            # Test connection with timeout
            await asyncio.wait_for(self.redis.ping(), timeout=5.0)
            self.connected = True
            logger.info("Redis rate limiter connected successfully")
            # Clear degraded mode if previously set
            self.degraded_mode = False
            try:
                from api.metrics import rate_limiter_degraded
                rate_limiter_degraded.set(0)
            except Exception:
                pass
            
        except asyncio.TimeoutError:
            logger.error("Redis connection timeout - rate limiting will fail closed in production")
            self.connected = False
            environment = os.getenv("ENVIRONMENT", "development")
            if environment in ["production", "staging", "prod"]:
                raise RuntimeError("Redis required for production - cannot start without rate limiting")
            self._memory_cache = {}
            self._enter_degraded_mode(reason="timeout during initialize")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for rate limiting: {e}")
            self.connected = False
            environment = os.getenv("ENVIRONMENT", "development")
            if environment in ["production", "staging", "prod"]:
                raise RuntimeError(f"Redis required for production: {e}")
            # Development only fallback
            self._memory_cache = {}
            self._enter_degraded_mode(reason=str(e))

    def _enter_degraded_mode(self, reason: str):
        """Activate degraded mode (limited permissive behavior) with throttled logging."""
        self.degraded_mode = True
        now = time.time()
        if now - self._degraded_last_log > 30:  # throttle logs
            logger.warning(f"Rate limiter entering degraded mode: {reason}")
            self._degraded_last_log = now
        try:
            from api.metrics import rate_limiter_degraded
            rate_limiter_degraded.set(1)
        except Exception:
            pass
    
    async def check_rate_limit(
        self, 
        identifier: str, 
        limit_type: str = "default", 
        custom_limit: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: Unique identifier (IP, user_id, API key, etc.)
            limit_type: Type of rate limit to apply
            custom_limit: Custom rate limit override
            
        Returns:
            Dict with rate limit status and metadata
        """
        if not self.connected:
            environment = os.getenv("ENVIRONMENT", "development")
            fail_closed = os.getenv("RATE_LIMIT_FAIL_CLOSED", "true").lower() == "true"
            
            # In production, ALWAYS fail closed when Redis is unavailable
            if environment in ["production", "staging", "prod"] and fail_closed:
                logger.error("Redis unavailable - denying ALL requests (fail-closed mode)")
                return self._create_rate_limit_response(
                    allowed=False,
                    current_requests=0,
                    limit=0,
                    window=60,
                    reset_time=time.time() + 60,
                    error="Rate limiter unavailable - failing closed for security"
                )
            
            # Development only: allow with memory fallback (NOT FOR PRODUCTION)
            if environment not in ["production", "staging", "prod"]:
                if not self.degraded_mode:
                    self._enter_degraded_mode("redis disconnected - DEV ONLY memory fallback")
                result = await self._check_memory_fallback(identifier, limit_type)
                try:
                    from api.metrics import rate_limiter_fallback_requests_total
                    rate_limiter_fallback_requests_total.labels(status='allowed' if result['allowed'] else 'blocked').inc()
                except Exception:
                    pass
                return result
            
            # Should never reach here in production
            logger.critical("Rate limiter in undefined state - denying request")
            return self._create_rate_limit_response(
                allowed=False,
                current_requests=0,
                limit=0,
                window=60,
                reset_time=time.time() + 60,
                error="Rate limiter error - request denied"
            )
        
        # Get rate limit configuration
        if custom_limit:
            config = custom_limit
        else:
            config = self.rate_limits.get(limit_type, self.rate_limits["default"])
        
        requests_allowed = config["requests"]
        window_seconds = config["window"]
        burst_allowed = config.get("burst", requests_allowed // 5)
        
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Redis key for this identifier and limit type
        key = f"rate_limit:{limit_type}:{identifier}"
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Remove old entries outside the sliding window
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Execute pipeline
            results = await pipe.execute()
            current_requests = results[1]
            
            # Check burst limit first (stricter)
            if current_requests >= burst_allowed:
                # Check last request time for burst limit
                recent_requests = await self.redis.zrevrange(key, 0, burst_allowed-1, withscores=True)
                if recent_requests:
                    oldest_recent = recent_requests[-1][1]  # Score is timestamp
                    if current_time - oldest_recent < 60:  # Within 1 minute
                        return self._create_rate_limit_response(
                            allowed=False,
                            current_requests=current_requests,
                            limit=burst_allowed,
                            window=60,
                            reset_time=oldest_recent + 60,
                            limit_type="burst"
                        )
            
            # Check main rate limit
            if current_requests >= requests_allowed:
                # Get oldest request in window to calculate reset time
                oldest = await self.redis.zrange(key, 0, 0, withscores=True)
                reset_time = oldest[0][1] + window_seconds if oldest else current_time + window_seconds
                
                return self._create_rate_limit_response(
                    allowed=False,
                    current_requests=current_requests,
                    limit=requests_allowed,
                    window=window_seconds,
                    reset_time=reset_time
                )
            
            # Add current request to sliding window
            await self.redis.zadd(key, {str(current_time): current_time})
            
            # Set expiry on key (cleanup)
            await self.redis.expire(key, window_seconds + 60)  # Extra buffer for cleanup
            
            return self._create_rate_limit_response(
                allowed=True,
                current_requests=current_requests + 1,
                limit=requests_allowed,
                window=window_seconds,
                reset_time=current_time + window_seconds
            )
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Enter degraded mode and memory fallback while allowing request (safer than full deny mid-flight)
            self._enter_degraded_mode(reason=f"redis ops error: {e}")
            fb = await self._check_memory_fallback(identifier, limit_type)
            try:
                from api.metrics import rate_limiter_fallback_requests_total
                rate_limiter_fallback_requests_total.labels(status='allowed' if fb['allowed'] else 'blocked').inc()
            except Exception:
                pass
            return fb
    
    async def _check_memory_fallback(self, identifier: str, limit_type: str) -> Dict[str, Any]:
        """Fallback in-memory rate limiting (not recommended for production)."""
        if not self.degraded_mode:
            logger.warning("Using in-memory rate limiting fallback - not recommended for production!")
        
        config = self.rate_limits.get(limit_type, self.rate_limits["default"])
        requests_allowed = config["requests"]
        window_seconds = config["window"]
        
        current_time = time.time()
        
        # Clean old entries
        if not hasattr(self, '_memory_cache'):
            self._memory_cache = {}
            
        key = f"{limit_type}:{identifier}"
        if key not in self._memory_cache:
            self._memory_cache[key] = []
        
        # Remove old timestamps
        self._memory_cache[key] = [
            ts for ts in self._memory_cache[key] 
            if current_time - ts < window_seconds
        ]
        
        current_requests = len(self._memory_cache[key])
        
        if current_requests >= requests_allowed:
            oldest_request = min(self._memory_cache[key])
            return self._create_rate_limit_response(
                allowed=False,
                current_requests=current_requests,
                limit=requests_allowed,
                window=window_seconds,
                reset_time=oldest_request + window_seconds,
                error="Using memory fallback"
            )
        
        # Add current request
        self._memory_cache[key].append(current_time)
        
        return self._create_rate_limit_response(
            allowed=True,
            current_requests=current_requests + 1,
            limit=requests_allowed,
            window=window_seconds,
            reset_time=current_time + window_seconds,
            error="degraded-memory-fallback"
        )
    
    def _create_rate_limit_response(
        self, 
        allowed: bool, 
        current_requests: int, 
        limit: int, 
        window: int, 
        reset_time: float,
        limit_type: str = "window",
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized rate limit response."""
        return {
            "allowed": allowed,
            "current_requests": current_requests,
            "limit": limit,
            "window_seconds": window,
            "reset_time": reset_time,
            "reset_in_seconds": max(0, int(reset_time - time.time())),
            "limit_type": limit_type,
            "error": error
        }
    
    async def get_user_rate_limit_type(self, request: Request) -> str:
        """Determine rate limit type based on request context."""
        path = request.url.path.lower()
        
        # Check for authenticated user
        user = getattr(request.state, 'user', None)
        if user and 'admin' in getattr(user, 'roles', []):
            return "admin"
        
        # Endpoint-based rate limiting
        if '/auth/' in path:
            return "auth"
        elif '/ws' in path or 'websocket' in path:
            return "websocket"
        elif '/market' in path or '/data' in path:
            return "market_data"
        elif '/trading' in path or '/orders' in path or '/portfolio' in path:
            return "trading"
        else:
            return "default"
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self.connected = False


# Global rate limiter instance
_rate_limiter: Optional[RedisRateLimiter] = None


async def get_rate_limiter() -> RedisRateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RedisRateLimiter()
        await _rate_limiter.initialize()
    return _rate_limiter


async def create_rate_limit_middleware():
    """Create rate limiting middleware function."""
    limiter = await get_rate_limiter()
    
    async def rate_limit_middleware(request: Request, call_next):
        """Rate limiting middleware."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/v1/health"]:
            return await call_next(request)
        
        # Get client identifier (prefer user ID over IP)
        client_ip = request.client.host if request.client else "unknown"
        user = getattr(request.state, 'user', None)
        identifier = getattr(user, 'user_id', client_ip) if user else client_ip
        
        # Get rate limit type
        limit_type = await limiter.get_user_rate_limit_type(request)
        
        # Check rate limit
        result = await limiter.check_rate_limit(identifier, limit_type)
        
        # Metrics integration (best-effort; metrics module may not be available early)
        try:
            from api.metrics import metrics
            metrics.record_rate_limit_check(limit_type, result["allowed"], identifier_type='user' if user else 'ip')
            if not result["allowed"]:
                from api.metrics import metrics as _m
                from api.metrics import rate_limit_blocks_total  # noqa: F401 (ensures import side effects)
        except Exception:
            pass

        if not result["allowed"]:
            # Add rate limit headers
            headers = {
                "X-RateLimit-Limit": str(result["limit"]),
                "X-RateLimit-Remaining": str(max(0, result["limit"] - result["current_requests"])),
                "X-RateLimit-Reset": str(int(result["reset_time"])),
                "X-RateLimit-Window": str(result["window_seconds"]),
                "Retry-After": str(result["reset_in_seconds"])
            }
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {result['limit']} per {result['window_seconds']}s",
                    "reset_in_seconds": result["reset_in_seconds"],
                    "limit_type": result.get("limit_type", "window")
                },
                headers=headers
            )
        
        # Add rate limit info headers to successful requests
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(result["limit"])
        response.headers["X-RateLimit-Remaining"] = str(max(0, result["limit"] - result["current_requests"]))
        response.headers["X-RateLimit-Reset"] = str(int(result["reset_time"]))
        
        return response
    
    return rate_limit_middleware
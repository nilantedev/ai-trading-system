#!/usr/bin/env python3
"""
Enhanced Rate Limiter with Strict Fail-Closed Enforcement
Production-grade rate limiting with security-first approach
"""

import asyncio
import time
import logging
import hashlib
import hmac
from typing import Optional, Dict, Any, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from fastapi import HTTPException, Request, Response
import redis.asyncio as redis
import os
import json

logger = logging.getLogger(__name__)


class RateLimitMode(Enum):
    """Rate limiter operational modes"""
    NORMAL = "normal"           # Redis available, normal operation
    FAIL_CLOSED = "fail_closed" # Redis unavailable, deny all
    DEGRADED = "degraded"        # Redis intermittent, strict limits
    MAINTENANCE = "maintenance"  # Admin-controlled restrictive mode


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests: int
    window: int
    burst: int
    priority: int = 0  # Higher priority gets more resources
    enforce_strict: bool = True
    allow_override: bool = False


class EnhancedRateLimiter:
    """Enhanced rate limiter with strict security enforcement"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.redis: Optional[aioredis.Redis] = None
        self.mode = RateLimitMode.NORMAL
        self.connected = False
        
        # Security settings
        self.enforce_fail_closed = os.getenv("ENFORCE_FAIL_CLOSED", "true").lower() == "true"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.is_production = self.environment in ["production", "staging", "prod"]
        
        # Connection health tracking
        self.last_successful_ping = time.time()
        self.consecutive_failures = 0
        self.max_failures_before_fail_closed = 3
        
        # Trusted IPs that bypass rate limiting (carefully managed)
        self.trusted_ips: Set[str] = set()
        self._load_trusted_ips()
        
        # Rate limit configurations with security-first defaults
        self.rate_limits = {
            "auth": RateLimitConfig(5, 60, 2, priority=0, enforce_strict=True),
            "auth_failed": RateLimitConfig(3, 300, 1, priority=0, enforce_strict=True),  # Failed auth attempts
            "password_reset": RateLimitConfig(3, 3600, 1, priority=0, enforce_strict=True),
            "api_key_generation": RateLimitConfig(5, 3600, 2, priority=0, enforce_strict=True),
            "websocket": RateLimitConfig(20, 60, 5, priority=1),
            "market_data": RateLimitConfig(100, 60, 30, priority=2),
            "trading": RateLimitConfig(20, 60, 5, priority=1, enforce_strict=True),
            "admin": RateLimitConfig(200, 60, 50, priority=3),
            "health": RateLimitConfig(100, 60, 20, priority=3, enforce_strict=False),
            "default": RateLimitConfig(60, 60, 15, priority=1),
            "strict": RateLimitConfig(10, 60, 3, priority=0, enforce_strict=True),  # Fallback strict mode
        }
        
        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "fail_closed_denials": 0,
            "redis_failures": 0,
            "mode_changes": []
        }
    
    def _load_trusted_ips(self):
        """Load trusted IPs from secure configuration"""
        trusted_ips_env = os.getenv("RATE_LIMIT_TRUSTED_IPS", "")
        if trusted_ips_env:
            # Verify HMAC signature for trusted IPs list
            parts = trusted_ips_env.split(":")
            if len(parts) == 2:
                ips_data, signature = parts
                expected_sig = hmac.new(
                    os.getenv("TRUSTED_IP_SECRET", "").encode(),
                    ips_data.encode(),
                    hashlib.sha256
                ).hexdigest()
                if hmac.compare_digest(signature, expected_sig):
                    self.trusted_ips = set(ips_data.split(","))
                    logger.info(f"Loaded {len(self.trusted_ips)} trusted IPs")
                else:
                    logger.error("Trusted IPs signature verification failed")
    
    async def initialize(self):
        """Initialize Redis connection with strict fail-closed enforcement"""
        try:
            # Create Redis connection
            self.redis = aioredis.from_url(
                self.redis_url,
                password=self.redis_password,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=3,  # Shorter timeout for faster failure detection
                socket_timeout=3,
                retry_on_timeout=False,  # Don't retry on timeout
                health_check_interval=10  # Regular health checks
            )
            
            # Test connection with strict timeout
            await asyncio.wait_for(self.redis.ping(), timeout=2.0)
            
            # Verify Redis version and features
            info = await self.redis.info()
            redis_version = info.get('redis_version', '0.0.0')
            logger.info(f"Redis rate limiter connected (v{redis_version})")
            
            # Set operational mode
            self.connected = True
            self.mode = RateLimitMode.NORMAL
            self.last_successful_ping = time.time()
            self.consecutive_failures = 0
            
            # Start background health monitor
            asyncio.create_task(self._health_monitor())
            
            # Update metrics
            await self._update_metrics("initialize_success")
            
        except asyncio.TimeoutError:
            logger.error("Redis connection timeout - entering fail-closed mode")
            await self._enter_fail_closed_mode("connection_timeout")
            
            if self.is_production and self.enforce_fail_closed:
                raise RuntimeError(
                    "SECURITY: Redis required for production rate limiting. "
                    "Cannot start without functional rate limiter."
                )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            await self._enter_fail_closed_mode(f"connection_error: {e}")
            
            if self.is_production and self.enforce_fail_closed:
                raise RuntimeError(
                    f"SECURITY: Redis required for production. Error: {e}"
                )
    
    async def _health_monitor(self):
        """Background task to monitor Redis health"""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if self.redis:
                    try:
                        await asyncio.wait_for(self.redis.ping(), timeout=1.0)
                        self.last_successful_ping = time.time()
                        
                        # Recover from fail-closed if Redis is back
                        if self.mode == RateLimitMode.FAIL_CLOSED:
                            await self._recover_from_fail_closed()
                        
                        self.consecutive_failures = 0
                        
                    except (asyncio.TimeoutError, Exception) as e:
                        self.consecutive_failures += 1
                        logger.warning(f"Redis health check failed ({self.consecutive_failures}): {e}")
                        
                        # Enter fail-closed mode after threshold
                        if self.consecutive_failures >= self.max_failures_before_fail_closed:
                            await self._enter_fail_closed_mode("health_check_failures")
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _enter_fail_closed_mode(self, reason: str):
        """Enter fail-closed mode - deny all requests"""
        previous_mode = self.mode
        self.mode = RateLimitMode.FAIL_CLOSED
        self.connected = False
        
        # Log mode change
        self.metrics["mode_changes"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "from": previous_mode.value,
            "to": self.mode.value,
            "reason": reason
        })
        
        logger.critical(f"SECURITY: Rate limiter entering FAIL-CLOSED mode. Reason: {reason}")
        
        # Alert monitoring systems
        await self._send_security_alert(
            "RATE_LIMITER_FAIL_CLOSED",
            f"Rate limiter failing closed due to: {reason}"
        )
        
        # Update metrics
        await self._update_metrics("fail_closed_activated")
    
    async def _recover_from_fail_closed(self):
        """Recover from fail-closed mode when Redis is available"""
        if self.mode != RateLimitMode.FAIL_CLOSED:
            return
        
        self.mode = RateLimitMode.DEGRADED  # Start in degraded mode
        self.connected = True
        
        logger.warning("Rate limiter recovering from fail-closed mode (entering degraded mode)")
        
        # Gradually return to normal after verification period
        asyncio.create_task(self._gradual_recovery())
    
    async def _gradual_recovery(self):
        """Gradually recover from degraded to normal mode"""
        await asyncio.sleep(30)  # Observe for 30 seconds
        
        if self.consecutive_failures == 0 and self.connected:
            self.mode = RateLimitMode.NORMAL
            logger.info("Rate limiter fully recovered to normal mode")
            await self._update_metrics("recovered_to_normal")
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit_type: str = "default",
        request: Optional[Request] = None,
        custom_limit: Optional[RateLimitConfig] = None
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limit with strict enforcement
        """
        self.metrics["total_requests"] += 1
        
        # Check trusted IPs (with caution)
        if request and request.client and request.client.host in self.trusted_ips:
            return self._create_rate_limit_response(
                allowed=True,
                current_requests=0,
                limit=999999,
                window=60,
                reset_time=time.time() + 60,
                trusted=True
            )
        
        # FAIL-CLOSED MODE: Deny everything except health checks
        if self.mode == RateLimitMode.FAIL_CLOSED:
            self.metrics["fail_closed_denials"] += 1
            
            # Allow only critical health checks
            if limit_type == "health" and request and "/health" in request.url.path:
                return self._create_rate_limit_response(
                    allowed=True,
                    current_requests=0,
                    limit=1,
                    window=60,
                    reset_time=time.time() + 60,
                    mode=self.mode.value,
                    warning="Rate limiter in fail-closed mode"
                )
            
            # Deny all other requests
            logger.warning(f"FAIL-CLOSED: Denying request from {identifier} ({limit_type})")
            return self._create_rate_limit_response(
                allowed=False,
                current_requests=0,
                limit=0,
                window=60,
                reset_time=time.time() + 60,
                mode=self.mode.value,
                error="Service temporarily unavailable - rate limiter in fail-closed mode"
            )
        
        # DEGRADED MODE: Use strict limits
        if self.mode == RateLimitMode.DEGRADED:
            config = self.rate_limits.get("strict")
        else:
            config = custom_limit or self.rate_limits.get(limit_type, self.rate_limits["default"])
        
        # MAINTENANCE MODE: Very restrictive
        if self.mode == RateLimitMode.MAINTENANCE:
            config = RateLimitConfig(
                requests=max(1, config.requests // 10),
                window=config.window,
                burst=max(1, config.burst // 10),
                enforce_strict=True
            )
        
        # Try Redis-based rate limiting
        if self.connected and self.redis:
            try:
                result = await self._check_redis_rate_limit(identifier, config)
                if not result["allowed"]:
                    self.metrics["blocked_requests"] += 1
                return result
            except Exception as e:
                logger.error(f"Redis rate limit check failed: {e}")
                self.metrics["redis_failures"] += 1
                self.consecutive_failures += 1
                
                # In production, fail closed on Redis errors
                if self.is_production and config.enforce_strict:
                    return self._create_rate_limit_response(
                        allowed=False,
                        current_requests=0,
                        limit=0,
                        window=60,
                        reset_time=time.time() + 60,
                        error="Rate limiting error - request denied for security"
                    )
        
        # No Redis connection in production = deny
        if self.is_production and self.enforce_fail_closed:
            self.metrics["blocked_requests"] += 1
            return self._create_rate_limit_response(
                allowed=False,
                current_requests=0,
                limit=0,
                window=60,
                reset_time=time.time() + 60,
                error="Rate limiter unavailable - failing closed"
            )
        
        # Development only: Allow with warning
        logger.warning("DEVELOPMENT: Allowing request without rate limiting")
        return self._create_rate_limit_response(
            allowed=True,
            current_requests=0,
            limit=config.requests,
            window=config.window,
            reset_time=time.time() + config.window,
            warning="Rate limiting bypassed (development only)"
        )
    
    async def _check_redis_rate_limit(
        self,
        identifier: str,
        config: RateLimitConfig
    ) -> Dict[str, Any]:
        """Check rate limit using Redis with sliding window algorithm"""
        current_time = time.time()
        window_start = current_time - config.window
        
        # Create unique key with hash for security
        key_hash = hashlib.sha256(f"{identifier}:{config.window}".encode()).hexdigest()[:16]
        key = f"rl:{key_hash}:{identifier}"
        
        # Atomic Redis operations
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)  # Remove old entries
        pipe.zcard(key)  # Count current requests
        pipe.execute()
        
        current_requests = await self.redis.zcard(key)
        
        # Check burst limit
        if current_requests >= config.burst:
            recent = await self.redis.zrevrange(key, 0, config.burst - 1, withscores=True)
            if recent and (current_time - recent[-1][1] < 10):  # Burst window
                return self._create_rate_limit_response(
                    allowed=False,
                    current_requests=current_requests,
                    limit=config.burst,
                    window=10,
                    reset_time=recent[-1][1] + 10,
                    limit_type="burst"
                )
        
        # Check main limit
        if current_requests >= config.requests:
            oldest = await self.redis.zrange(key, 0, 0, withscores=True)
            reset_time = oldest[0][1] + config.window if oldest else current_time + config.window
            
            return self._create_rate_limit_response(
                allowed=False,
                current_requests=current_requests,
                limit=config.requests,
                window=config.window,
                reset_time=reset_time
            )
        
        # Add current request
        await self.redis.zadd(key, {str(current_time): current_time})
        await self.redis.expire(key, config.window + 60)
        
        return self._create_rate_limit_response(
            allowed=True,
            current_requests=current_requests + 1,
            limit=config.requests,
            window=config.window,
            reset_time=current_time + config.window
        )
    
    def _create_rate_limit_response(
        self,
        allowed: bool,
        current_requests: int,
        limit: int,
        window: int,
        reset_time: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Create standardized rate limit response"""
        response = {
            "allowed": allowed,
            "current_requests": current_requests,
            "limit": limit,
            "window_seconds": window,
            "reset_time": reset_time,
            "reset_in_seconds": max(0, int(reset_time - time.time())),
            "mode": self.mode.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        response.update(kwargs)
        return response
    
    async def _send_security_alert(self, alert_type: str, message: str):
        """Send security alerts to monitoring systems"""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "environment": self.environment,
            "mode": self.mode.value,
            "metrics": self.metrics
        }
        
        # Send to monitoring systems (implement based on your infrastructure)
        logger.critical(f"SECURITY ALERT: {json.dumps(alert)}")
        
        # Could integrate with PagerDuty, Slack, etc.
        # await send_to_pagerduty(alert)
        # await send_to_slack(alert)
    
    async def _update_metrics(self, event: str):
        """Update internal metrics"""
        # This would integrate with Prometheus
        try:
            from api.metrics import (
                rate_limiter_mode_gauge,
                rate_limiter_events_total
            )
            
            mode_values = {
                RateLimitMode.NORMAL: 0,
                RateLimitMode.DEGRADED: 1,
                RateLimitMode.FAIL_CLOSED: 2,
                RateLimitMode.MAINTENANCE: 3
            }
            
            rate_limiter_mode_gauge.set(mode_values[self.mode])
            rate_limiter_events_total.labels(event=event).inc()
        except ImportError:
            pass
    
    async def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status for monitoring"""
        return {
            "mode": self.mode.value,
            "connected": self.connected,
            "environment": self.environment,
            "enforce_fail_closed": self.enforce_fail_closed,
            "last_successful_ping": datetime.fromtimestamp(self.last_successful_ping).isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "metrics": self.metrics,
            "trusted_ips_count": len(self.trusted_ips)
        }
    
    async def set_maintenance_mode(self, enabled: bool, admin_token: str) -> bool:
        """Enable/disable maintenance mode (admin only)"""
        # Verify admin token
        expected_token = hashlib.sha256(
            f"maintenance_{os.getenv('ADMIN_SECRET', '')}".encode()
        ).hexdigest()
        
        if not hmac.compare_digest(admin_token, expected_token):
            logger.warning("Invalid admin token for maintenance mode change")
            return False
        
        if enabled:
            self.mode = RateLimitMode.MAINTENANCE
            logger.warning("Rate limiter entering MAINTENANCE mode")
        else:
            self.mode = RateLimitMode.NORMAL if self.connected else RateLimitMode.FAIL_CLOSED
            logger.info(f"Rate limiter exiting maintenance mode to {self.mode.value}")
        
        await self._update_metrics(f"maintenance_{'enabled' if enabled else 'disabled'}")
        return True
    
    async def close(self):
        """Clean shutdown"""
        if self.redis:
            await self.redis.close()
            self.connected = False
            logger.info("Rate limiter connection closed")


# Global limiter instance
_rate_limiter: Optional[EnhancedRateLimiter] = None


async def get_rate_limiter() -> EnhancedRateLimiter:
    """Get or create the global rate limiter instance (compatibility function)"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = EnhancedRateLimiter()
        await _rate_limiter.initialize()
    return _rate_limiter


# Enhanced middleware with security headers
async def create_enhanced_rate_limit_middleware():
    """Create enhanced rate limiting middleware"""
    limiter = EnhancedRateLimiter()
    await limiter.initialize()
    
    async def enhanced_middleware(request: Request, call_next):
        """Enhanced rate limiting middleware with security headers"""
        # Always add security headers
        security_headers = {
            "X-RateLimit-Mode": limiter.mode.value,
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY"
        }
        
        # Skip rate limiting for health checks in fail-closed mode
        if request.url.path in ["/health", "/api/v1/health", "/ready"]:
            response = await call_next(request)
            for header, value in security_headers.items():
                response.headers[header] = value
            return response
        
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        user = getattr(request.state, 'user', None)
        identifier = getattr(user, 'user_id', client_ip) if user else client_ip
        
        # Determine limit type
        path = request.url.path.lower()
        if '/auth/login' in path:
            limit_type = "auth"
        elif '/auth/' in path and 'fail' in str(request.url):
            limit_type = "auth_failed"
        elif '/password' in path:
            limit_type = "password_reset"
        elif '/api/keys' in path:
            limit_type = "api_key_generation"
        elif user and 'admin' in getattr(user, 'roles', []):
            limit_type = "admin"
        else:
            limit_type = "default"
        
        # Check rate limit
        result = await limiter.check_rate_limit(identifier, limit_type, request)
        
        if not result["allowed"]:
            # Add rate limit headers to error response
            headers = {
                "X-RateLimit-Limit": str(result["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(result["reset_time"])),
                "Retry-After": str(result["reset_in_seconds"]),
                **security_headers
            }
            
            # Different status codes for different scenarios
            if limiter.mode == RateLimitMode.FAIL_CLOSED:
                status_code = 503  # Service Unavailable
                detail = "Service temporarily unavailable"
            else:
                status_code = 429  # Too Many Requests
                detail = f"Rate limit exceeded. Retry after {result['reset_in_seconds']} seconds"
            
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": detail,
                    "mode": result.get("mode"),
                    "reset_in_seconds": result["reset_in_seconds"]
                },
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit info to successful responses
        response.headers["X-RateLimit-Limit"] = str(result["limit"])
        response.headers["X-RateLimit-Remaining"] = str(max(0, result["limit"] - result["current_requests"]))
        response.headers["X-RateLimit-Reset"] = str(int(result["reset_time"]))
        
        # Add security headers
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    return enhanced_middleware


# Compatibility aliases for old names
create_rate_limit_middleware = create_enhanced_rate_limit_middleware
RedisRateLimiter = EnhancedRateLimiter  # Compatibility for tests
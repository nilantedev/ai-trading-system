#!/usr/bin/env python3
"""
Comprehensive tests for rate limiter module.
Tests Redis-based rate limiting, fallback behavior, and production safety.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import os

# Add parent directories to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.rate_limiter import RedisRateLimiter, get_rate_limiter


class TestRateLimiterInitialization:
    """Test rate limiter initialization and Redis connection."""
    
    @pytest.mark.asyncio
    async def test_redis_connection_success(self):
        """Test successful Redis connection."""
        limiter = RedisRateLimiter("redis://localhost:6379")
        
        # Mock successful Redis connection
        with patch('aioredis.from_url') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.ping = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            await limiter.initialize()
            
            assert limiter.connected is True
            mock_redis_instance.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure_development(self):
        """Test Redis connection failure in development environment."""
        limiter = RedisRateLimiter("redis://nonexistent:6379")
        
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            with patch('aioredis.from_url') as mock_redis:
                mock_redis.side_effect = Exception("Connection failed")
                
                # Should not raise in development
                await limiter.initialize()
                
                assert limiter.connected is False
                assert hasattr(limiter, '_memory_cache')
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure_production(self):
        """Test Redis connection failure in production environment."""
        limiter = RedisRateLimiter("redis://nonexistent:6379")
        
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with patch('aioredis.from_url') as mock_redis:
                mock_redis.side_effect = Exception("Connection failed")
                
                # Should raise in production
                with pytest.raises(RuntimeError):
                    await limiter.initialize()
    
    @pytest.mark.asyncio
    async def test_redis_timeout_production(self):
        """Test Redis timeout in production environment."""
        limiter = RedisRateLimiter("redis://localhost:6379")
        
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with patch('aioredis.from_url') as mock_redis:
                mock_redis_instance = AsyncMock()
                mock_redis_instance.ping = AsyncMock(side_effect=asyncio.TimeoutError())
                mock_redis.return_value = mock_redis_instance
                
                # Should raise in production
                with pytest.raises(RuntimeError):
                    await limiter.initialize()


class TestRateLimitChecking:
    """Test rate limit checking functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.limiter = RedisRateLimiter()
    
    @pytest.mark.asyncio
    async def test_rate_limit_allowed_within_limit(self):
        """Test that requests within limit are allowed."""
        # Mock connected Redis
        self.limiter.connected = True
        self.limiter.redis = AsyncMock()
        
        # Mock Redis pipeline operations
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[None, 1])  # 1 current request
        self.limiter.redis.pipeline.return_value = mock_pipeline
        self.limiter.redis.zrevrange = AsyncMock(return_value=[])
        self.limiter.redis.zrange = AsyncMock(return_value=[])
        self.limiter.redis.zadd = AsyncMock()
        self.limiter.redis.expire = AsyncMock()
        
        result = await self.limiter.check_rate_limit("test_user", "default")
        
        assert result["allowed"] is True
        assert result["current_requests"] == 2  # 1 + 1 for this request
        assert result["limit"] == 100  # Default limit
    
    @pytest.mark.asyncio
    async def test_rate_limit_denied_over_limit(self):
        """Test that requests over limit are denied."""
        self.limiter.connected = True
        self.limiter.redis = AsyncMock()
        
        # Mock Redis pipeline showing too many requests
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[None, 150])  # Over limit
        self.limiter.redis.pipeline.return_value = mock_pipeline
        self.limiter.redis.zrange = AsyncMock(return_value=[(str(time.time()), time.time())])
        
        result = await self.limiter.check_rate_limit("test_user", "default")
        
        assert result["allowed"] is False
        assert result["current_requests"] == 150
        assert "reset_time" in result
    
    @pytest.mark.asyncio
    async def test_burst_limit_protection(self):
        """Test burst limit protection."""
        self.limiter.connected = True
        self.limiter.redis = AsyncMock()
        
        # Mock pipeline showing burst limit exceeded
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[None, 25])  # Over burst limit
        self.limiter.redis.pipeline.return_value = mock_pipeline
        
        # Mock recent requests within burst window
        recent_time = time.time() - 30  # 30 seconds ago
        self.limiter.redis.zrevrange = AsyncMock(
            return_value=[("req1", recent_time)] * 20  # 20 recent requests
        )
        
        result = await self.limiter.check_rate_limit("test_user", "default")
        
        assert result["allowed"] is False
        assert result.get("limit_type") == "burst"
    
    @pytest.mark.asyncio
    async def test_different_limit_types(self):
        """Test different rate limit types."""
        self.limiter.connected = True
        self.limiter.redis = AsyncMock()
        
        # Set up mock for auth endpoint (stricter limits)
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[None, 1])
        self.limiter.redis.pipeline.return_value = mock_pipeline
        self.limiter.redis.zrevrange = AsyncMock(return_value=[])
        self.limiter.redis.zrange = AsyncMock(return_value=[])
        self.limiter.redis.zadd = AsyncMock()
        self.limiter.redis.expire = AsyncMock()
        
        result = await self.limiter.check_rate_limit("test_user", "auth")
        
        assert result["allowed"] is True
        assert result["limit"] == 10  # Auth limit is stricter
    
    @pytest.mark.asyncio
    async def test_custom_limit_override(self):
        """Test custom limit override."""
        self.limiter.connected = True
        self.limiter.redis = AsyncMock()
        
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[None, 1])
        self.limiter.redis.pipeline.return_value = mock_pipeline
        self.limiter.redis.zrevrange = AsyncMock(return_value=[])
        self.limiter.redis.zrange = AsyncMock(return_value=[])
        self.limiter.redis.zadd = AsyncMock()
        self.limiter.redis.expire = AsyncMock()
        
        custom_limit = {"requests": 50, "window": 60, "burst": 10}
        result = await self.limiter.check_rate_limit("test_user", "default", custom_limit)
        
        assert result["allowed"] is True
        assert result["limit"] == 50  # Custom limit applied


class TestFallbackBehavior:
    """Test fallback behavior when Redis is unavailable."""
    
    def setup_method(self):
        """Set up test environment."""
        self.limiter = RedisRateLimiter()
    
    @pytest.mark.asyncio
    async def test_production_fail_closed(self):
        """Test that production fails closed when Redis unavailable."""
        self.limiter.connected = False
        
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            result = await self.limiter.check_rate_limit("test_user", "default")
            
            assert result["allowed"] is False
            assert "Rate limiter unavailable" in result["error"]
    
    @pytest.mark.asyncio
    async def test_development_memory_fallback(self):
        """Test memory fallback in development."""
        self.limiter.connected = False
        self.limiter._memory_cache = {}
        
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            # First request should be allowed
            result = await self.limiter.check_rate_limit("test_user", "default")
            assert result["allowed"] is True
            
            # Fill up to limit
            for _ in range(99):
                await self.limiter.check_rate_limit("test_user", "default")
            
            # This should be denied
            result = await self.limiter.check_rate_limit("test_user", "default")
            assert result["allowed"] is False
    
    @pytest.mark.asyncio
    async def test_unknown_environment_fail_closed(self):
        """Test that unknown environment fails closed."""
        self.limiter.connected = False
        
        with patch.dict(os.environ, {"ENVIRONMENT": "unknown"}):
            result = await self.limiter.check_rate_limit("test_user", "default")
            
            assert result["allowed"] is False
            assert "Invalid environment configuration" in result["error"]
    
    @pytest.mark.asyncio
    async def test_redis_error_fallback(self):
        """Test fallback when Redis operations fail."""
        self.limiter.connected = True
        self.limiter.redis = AsyncMock()
        
        # Mock Redis pipeline to raise exception
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(side_effect=Exception("Redis error"))
        self.limiter.redis.pipeline.return_value = mock_pipeline
        
        result = await self.limiter.check_rate_limit("test_user", "default")
        
        # Should fallback to allowing request with error note
        assert result["allowed"] is True
        assert "Redis unavailable" in result["error"]


class TestRateLimitTypes:
    """Test different rate limit types and configurations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.limiter = RedisRateLimiter()
    
    @pytest.mark.asyncio
    async def test_get_user_rate_limit_type(self):
        """Test determination of rate limit type from request."""
        from fastapi import Request
        
        # Mock request for different endpoints
        auth_request = MagicMock()
        auth_request.url.path = "/auth/login"
        
        result = await self.limiter.get_user_rate_limit_type(auth_request)
        assert result == "auth"
        
        # Test WebSocket endpoint
        ws_request = MagicMock()
        ws_request.url.path = "/ws/market-data"
        
        result = await self.limiter.get_user_rate_limit_type(ws_request)
        assert result == "websocket"
        
        # Test admin user
        admin_request = MagicMock()
        admin_request.url.path = "/api/data"
        admin_user = MagicMock()
        admin_user.roles = ["admin"]
        admin_request.state.user = admin_user
        
        result = await self.limiter.get_user_rate_limit_type(admin_request)
        assert result == "admin"
    
    def test_rate_limit_configurations(self):
        """Test that rate limit configurations are properly set."""
        limiter = RedisRateLimiter()
        
        # Check that different endpoint types have appropriate limits
        assert limiter.rate_limits["auth"]["requests"] < limiter.rate_limits["default"]["requests"]
        assert limiter.rate_limits["admin"]["requests"] > limiter.rate_limits["default"]["requests"]
        assert limiter.rate_limits["trading"]["requests"] < limiter.rate_limits["market_data"]["requests"]
    
    def test_rate_limit_response_format(self):
        """Test rate limit response format."""
        limiter = RedisRateLimiter()
        
        response = limiter._create_rate_limit_response(
            allowed=True,
            current_requests=10,
            limit=100,
            window=60,
            reset_time=time.time() + 60
        )
        
        required_fields = ["allowed", "current_requests", "limit", "window_seconds", 
                          "reset_time", "reset_in_seconds", "limit_type"]
        
        for field in required_fields:
            assert field in response


class TestRateLimiterCleanup:
    """Test rate limiter cleanup and resource management."""
    
    @pytest.mark.asyncio
    async def test_close_connection(self):
        """Test closing Redis connection."""
        limiter = RedisRateLimiter()
        limiter.redis = AsyncMock()
        limiter.connected = True
        
        await limiter.close()
        
        limiter.redis.close.assert_called_once()
        assert limiter.connected is False
    
    @pytest.mark.asyncio
    async def test_global_limiter_instance(self):
        """Test global rate limiter instance management."""
        # Reset global instance
        import api.rate_limiter
        api.rate_limiter._rate_limiter = None
        
        with patch.object(RedisRateLimiter, 'initialize', AsyncMock()):
            limiter1 = await get_rate_limiter()
            limiter2 = await get_rate_limiter()
            
            # Should return same instance
            assert limiter1 is limiter2


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_rate_limiter.py -v
    pytest.main([__file__, "-v"])
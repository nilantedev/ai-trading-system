"""
Integration tests for API security features with comprehensive coverage.
"""

import pytest
import asyncio
import jwt
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any
import httpx
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import redis

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    verify_password,
    hash_password,
    RateLimiter,
    SecurityMiddleware,
    JWTRevocationStore,
    BruteForceProtection
)
from api.app import create_app
from config.secure_config import SecureConfig


class TestJWTAuthentication:
    """Test JWT authentication flow."""
    
    @pytest.fixture
    def secret_key(self):
        return "test_secret_key_for_testing_only_32_chars_long"
    
    @pytest.fixture
    def test_user(self):
        return {
            "user_id": "123",
            "username": "testuser",
            "roles": ["user", "trader"]
        }
    
    def test_create_access_token(self, secret_key, test_user):
        """Test creating an access token."""
        # Act
        token = create_access_token(
            data=test_user,
            secret_key=secret_key,
            expires_delta=timedelta(minutes=15)
        )
        
        # Assert
        assert token is not None
        decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
        assert decoded["user_id"] == test_user["user_id"]
        assert decoded["username"] == test_user["username"]
        assert "exp" in decoded
        assert "iat" in decoded
        assert "jti" in decoded  # JWT ID for revocation
    
    def test_create_refresh_token(self, secret_key, test_user):
        """Test creating a refresh token."""
        # Act
        token = create_refresh_token(
            data=test_user,
            secret_key=secret_key,
            expires_delta=timedelta(days=7)
        )
        
        # Assert
        assert token is not None
        decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
        assert decoded["user_id"] == test_user["user_id"]
        assert decoded["type"] == "refresh"
        
        # Refresh token should have longer expiry
        exp_time = datetime.fromtimestamp(decoded["exp"])
        iat_time = datetime.fromtimestamp(decoded["iat"])
        assert (exp_time - iat_time).days >= 6
    
    def test_verify_valid_token(self, secret_key, test_user):
        """Test verifying a valid token."""
        # Arrange
        token = create_access_token(test_user, secret_key)
        
        # Act
        payload = verify_token(token, secret_key)
        
        # Assert
        assert payload is not None
        assert payload["user_id"] == test_user["user_id"]
    
    def test_verify_expired_token(self, secret_key, test_user):
        """Test verifying an expired token."""
        # Arrange
        token = create_access_token(
            test_user,
            secret_key,
            expires_delta=timedelta(seconds=-1)
        )
        
        # Act
        payload = verify_token(token, secret_key)
        
        # Assert
        assert payload is None
    
    def test_verify_invalid_token(self, secret_key):
        """Test verifying an invalid token."""
        # Act
        payload = verify_token("invalid.token.here", secret_key)
        
        # Assert
        assert payload is None
    
    def test_token_with_invalid_signature(self, secret_key, test_user):
        """Test token with wrong signature."""
        # Arrange
        token = create_access_token(test_user, secret_key)
        
        # Act
        payload = verify_token(token, "wrong_secret_key")
        
        # Assert
        assert payload is None


class TestPasswordHashing:
    """Test password hashing and verification."""
    
    def test_hash_password(self):
        """Test password hashing."""
        # Arrange
        password = "SecurePassword123!"
        
        # Act
        hashed = hash_password(password)
        
        # Assert
        assert hashed != password
        assert len(hashed) > 0
        assert "$2b$" in hashed  # bcrypt identifier
    
    def test_verify_correct_password(self):
        """Test verifying correct password."""
        # Arrange
        password = "SecurePassword123!"
        hashed = hash_password(password)
        
        # Act
        result = verify_password(password, hashed)
        
        # Assert
        assert result is True
    
    def test_verify_incorrect_password(self):
        """Test verifying incorrect password."""
        # Arrange
        password = "SecurePassword123!"
        hashed = hash_password(password)
        
        # Act
        result = verify_password("WrongPassword", hashed)
        
        # Assert
        assert result is False
    
    def test_different_hashes_for_same_password(self):
        """Test that same password produces different hashes."""
        # Arrange
        password = "TestPassword"
        
        # Act
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        # Assert
        assert hash1 != hash2  # Different salts
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    @pytest.fixture
    def redis_mock(self):
        """Create Redis mock."""
        mock = Mock()
        mock.pipeline.return_value.__enter__ = Mock(return_value=mock)
        mock.pipeline.return_value.__exit__ = Mock(return_value=None)
        mock.zadd.return_value = None
        mock.zremrangebyscore.return_value = None
        mock.zcard.return_value = 5
        mock.execute.return_value = [None, None, 5]
        return mock
    
    def test_rate_limit_not_exceeded(self, redis_mock):
        """Test when rate limit is not exceeded."""
        # Arrange
        limiter = RateLimiter(redis_mock, max_requests=10, window_seconds=60)
        
        # Act
        allowed = limiter.is_allowed("test_key")
        
        # Assert
        assert allowed is True
    
    def test_rate_limit_exceeded(self, redis_mock):
        """Test when rate limit is exceeded."""
        # Arrange
        redis_mock.execute.return_value = [None, None, 15]
        limiter = RateLimiter(redis_mock, max_requests=10, window_seconds=60)
        
        # Act
        allowed = limiter.is_allowed("test_key")
        
        # Assert
        assert allowed is False
    
    def test_rate_limit_with_redis_error(self, redis_mock):
        """Test rate limiting with Redis error."""
        # Arrange
        redis_mock.pipeline.side_effect = redis.RedisError("Connection failed")
        limiter = RateLimiter(
            redis_mock,
            max_requests=10,
            window_seconds=60,
            fail_open=True
        )
        
        # Act
        allowed = limiter.is_allowed("test_key")
        
        # Assert (fail open)
        assert allowed is True
    
    def test_rate_limit_fail_closed(self, redis_mock):
        """Test rate limiting failing closed on error."""
        # Arrange
        redis_mock.pipeline.side_effect = redis.RedisError("Connection failed")
        limiter = RateLimiter(
            redis_mock,
            max_requests=10,
            window_seconds=60,
            fail_open=False
        )
        
        # Act
        allowed = limiter.is_allowed("test_key")
        
        # Assert (fail closed)
        assert allowed is False
    
    def test_sliding_window_cleanup(self, redis_mock):
        """Test that old entries are cleaned up."""
        # Arrange
        limiter = RateLimiter(redis_mock, max_requests=10, window_seconds=60)
        current_time = time.time()
        
        # Act
        limiter.is_allowed("test_key")
        
        # Assert
        redis_mock.zremrangebyscore.assert_called()
        args = redis_mock.zremrangebyscore.call_args[0]
        assert args[0] == "rate_limit:test_key"
        assert args[1] == 0
        assert abs(args[2] - (current_time - 60)) < 1


class TestJWTRevocationStore:
    """Test JWT revocation functionality."""
    
    @pytest.fixture
    def revocation_store(self):
        """Create revocation store with mock Redis."""
        redis_mock = Mock()
        redis_mock.setex.return_value = True
        redis_mock.exists.return_value = False
        return JWTRevocationStore(redis_mock)
    
    def test_revoke_token(self, revocation_store):
        """Test revoking a token."""
        # Arrange
        jti = "token_id_123"
        exp_timestamp = int(time.time()) + 3600
        
        # Act
        result = revocation_store.revoke_token(jti, exp_timestamp)
        
        # Assert
        assert result is True
        revocation_store.redis_client.setex.assert_called_once()
    
    def test_check_revoked_token(self, revocation_store):
        """Test checking if token is revoked."""
        # Arrange
        jti = "revoked_token"
        revocation_store.redis_client.exists.return_value = True
        
        # Act
        is_revoked = revocation_store.is_revoked(jti)
        
        # Assert
        assert is_revoked is True
    
    def test_check_non_revoked_token(self, revocation_store):
        """Test checking non-revoked token."""
        # Arrange
        jti = "valid_token"
        revocation_store.redis_client.exists.return_value = False
        
        # Act
        is_revoked = revocation_store.is_revoked(jti)
        
        # Assert
        assert is_revoked is False


class TestBruteForceProtection:
    """Test brute force protection."""
    
    @pytest.fixture
    def protection(self):
        """Create brute force protection with mock Redis."""
        redis_mock = Mock()
        redis_mock.incr.return_value = 1
        redis_mock.expire.return_value = True
        redis_mock.get.return_value = None
        redis_mock.ttl.return_value = -1
        return BruteForceProtection(redis_mock)
    
    def test_track_failed_attempt(self, protection):
        """Test tracking failed login attempts."""
        # Arrange
        identifier = "user@example.com"
        
        # Act
        protection.track_failed_attempt(identifier)
        
        # Assert
        protection.redis_client.incr.assert_called_with(f"brute_force:{identifier}")
    
    def test_check_not_blocked(self, protection):
        """Test checking non-blocked identifier."""
        # Arrange
        protection.redis_client.get.return_value = b"2"  # Below threshold
        
        # Act
        is_blocked = protection.is_blocked("user@example.com")
        
        # Assert
        assert is_blocked is False
    
    def test_check_blocked(self, protection):
        """Test checking blocked identifier."""
        # Arrange
        protection.redis_client.get.return_value = b"6"  # Above threshold
        
        # Act
        is_blocked = protection.is_blocked("user@example.com")
        
        # Assert
        assert is_blocked is True
    
    def test_reset_attempts(self, protection):
        """Test resetting failed attempts."""
        # Arrange
        identifier = "user@example.com"
        
        # Act
        protection.reset_attempts(identifier)
        
        # Assert
        protection.redis_client.delete.assert_called_with(f"brute_force:{identifier}")


class TestSecurityMiddleware:
    """Test security middleware."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        @app.get("/admin")
        async def admin_endpoint():
            return {"status": "admin"}
        
        return app
    
    @pytest.fixture
    def client_with_middleware(self, app):
        """Create test client with security middleware."""
        middleware = SecurityMiddleware(app)
        return TestClient(app)
    
    def test_security_headers(self, client_with_middleware):
        """Test that security headers are added."""
        # Act
        response = client_with_middleware.get("/test")
        
        # Assert
        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
    
    def test_cors_headers(self, client_with_middleware):
        """Test CORS headers."""
        # Act
        response = client_with_middleware.options(
            "/test",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Assert
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
    
    def test_request_id_generation(self, client_with_middleware):
        """Test that request IDs are generated."""
        # Act
        response = client_with_middleware.get("/test")
        
        # Assert
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0


class TestAPISecurityIntegration:
    """Integration tests for complete API security flow."""
    
    @pytest.fixture
    async def app(self):
        """Create full test application."""
        return create_app(testing=True)
    
    @pytest.fixture
    async def client(self, app):
        """Create async test client."""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token."""
        # Act
        response = await client.get("/api/v1/protected")
        
        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_with_valid_token(self, client):
        """Test accessing protected endpoint with valid token."""
        # Arrange
        token = create_access_token(
            {"user_id": "123", "username": "test"},
            "test_secret_key"
        )
        
        # Act
        response = await client.get(
            "/api/v1/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
    
    @pytest.mark.asyncio
    async def test_login_flow(self, client):
        """Test complete login flow."""
        # Act
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "testuser", "password": "testpass"}
        )
        
        # Assert
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "access_token" in data
            assert "refresh_token" in data
            assert "token_type" in data
            assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_token_refresh_flow(self, client):
        """Test token refresh flow."""
        # Arrange - login first
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"username": "testuser", "password": "testpass"}
        )
        
        if login_response.status_code == status.HTTP_200_OK:
            refresh_token = login_response.json()["refresh_token"]
            
            # Act - refresh token
            refresh_response = await client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": refresh_token}
            )
            
            # Assert
            assert refresh_response.status_code == status.HTTP_200_OK
            data = refresh_response.json()
            assert "access_token" in data
    
    @pytest.mark.asyncio
    async def test_logout_flow(self, client):
        """Test logout flow with token revocation."""
        # Arrange - login first
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"username": "testuser", "password": "testpass"}
        )
        
        if login_response.status_code == status.HTTP_200_OK:
            access_token = login_response.json()["access_token"]
            
            # Act - logout
            logout_response = await client.post(
                "/api/v1/auth/logout",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            # Assert
            assert logout_response.status_code == status.HTTP_200_OK
            
            # Token should now be revoked
            protected_response = await client.get(
                "/api/v1/protected",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            assert protected_response.status_code == status.HTTP_401_UNAUTHORIZED


class TestSecurityPerformance:
    """Performance tests for security features."""
    
    def test_password_hashing_performance(self):
        """Test password hashing performance."""
        import time
        
        # Arrange
        password = "TestPassword123!"
        iterations = 10
        
        # Act
        start_time = time.time()
        for _ in range(iterations):
            hash_password(password)
        elapsed = time.time() - start_time
        
        # Assert
        avg_time = elapsed / iterations
        assert avg_time < 0.5  # Should hash in under 500ms
    
    def test_token_generation_performance(self):
        """Test JWT generation performance."""
        import time
        
        # Arrange
        user_data = {"user_id": "123", "username": "test"}
        secret = "test_secret_key"
        iterations = 100
        
        # Act
        start_time = time.time()
        for _ in range(iterations):
            create_access_token(user_data, secret)
        elapsed = time.time() - start_time
        
        # Assert
        avg_time = elapsed / iterations
        assert avg_time < 0.01  # Should generate in under 10ms
    
    def test_rate_limiting_performance(self):
        """Test rate limiting performance."""
        import time
        
        # Arrange
        redis_mock = Mock()
        redis_mock.pipeline.return_value.__enter__ = Mock(return_value=redis_mock)
        redis_mock.pipeline.return_value.__exit__ = Mock(return_value=None)
        redis_mock.execute.return_value = [None, None, 5]
        
        limiter = RateLimiter(redis_mock, max_requests=100)
        iterations = 1000
        
        # Act
        start_time = time.time()
        for i in range(iterations):
            limiter.is_allowed(f"key_{i % 10}")
        elapsed = time.time() - start_time
        
        # Assert
        avg_time = elapsed / iterations
        assert avg_time < 0.001  # Should check in under 1ms
#!/usr/bin/env python3
"""
Comprehensive tests for updated authentication with Redis-based persistent security store.
Tests async token revocation, refresh token management, and security event logging.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.auth import (
    authenticate_user, create_access_token, create_refresh_token,
    verify_access_token, refresh_access_token, revoke_token,
    get_current_user, get_current_active_user, get_optional_user,
    User, UserRole, TokenData, SYSTEM_USERS
)
from trading_common.security_store import PersistentSecurityStore, SecurityEventType
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


class TestAsyncTokenManagement:
    """Test suite for async token management with Redis persistence."""
    
    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(
            user_id="test_user_001",
            username="test_user",
            roles=[UserRole.TRADER],
            permissions=["read:market_data", "write:orders"],
            is_active=True
        )
    
    @pytest.fixture
    async def mock_security_store(self):
        """Create mock persistent security store."""
        store = AsyncMock(spec=PersistentSecurityStore)
        store.revoke_token.return_value = True
        store.is_token_revoked.return_value = False
        store.store_refresh_token.return_value = True
        store.get_refresh_token.return_value = {
            "user_id": "test_user_001",
            "username": "test_user",
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=30)
        }
        store.revoke_refresh_token.return_value = True
        return store
    
    @pytest.mark.asyncio
    async def test_async_token_revocation(self, test_user, mock_security_store):
        """Test async token revocation with persistent store."""
        # Create access token
        token = create_access_token(test_user)
        
        # Mock security store
        with patch('api.auth.get_security_store', return_value=mock_security_store):
            with patch('api.auth.log_security_event') as mock_log:
                # Revoke token
                await revoke_token(token)
                
                # Verify security store was called
                mock_security_store.revoke_token.assert_called_once()
                
                # Verify security event was logged
                mock_log.assert_called_once()
                call_args = mock_log.call_args
                assert call_args[1]['event_type'] == SecurityEventType.TOKEN_REVOKED
    
    @pytest.mark.asyncio
    async def test_async_token_verification_not_revoked(self, test_user, mock_security_store):
        """Test async token verification for valid token."""
        token = create_access_token(test_user)
        
        # Mock security store to return token not revoked
        mock_security_store.is_token_revoked.return_value = False
        
        with patch('api.auth.get_security_store', return_value=mock_security_store):
            token_data = await verify_access_token(token)
            
            assert token_data is not None
            assert token_data.username == test_user.username
            mock_security_store.is_token_revoked.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_token_verification_revoked(self, test_user, mock_security_store):
        """Test async token verification for revoked token."""
        token = create_access_token(test_user)
        
        # Mock security store to return token is revoked
        mock_security_store.is_token_revoked.return_value = True
        
        with patch('api.auth.get_security_store', return_value=mock_security_store):
            with pytest.raises(HTTPException) as exc_info:
                await verify_access_token(token)
            
            assert exc_info.value.status_code == 401
            assert "revoked" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_async_refresh_token_creation(self, test_user, mock_security_store):
        """Test async refresh token creation with persistent store."""
        with patch('api.auth.get_security_store', return_value=mock_security_store):
            refresh_token = await create_refresh_token(test_user)
            
            assert isinstance(refresh_token, str)
            assert len(refresh_token) > 0
            mock_security_store.store_refresh_token.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_refresh_token_usage(self, test_user, mock_security_store):
        """Test async refresh token usage and rotation."""
        # Create initial refresh token
        with patch('api.auth.get_security_store', return_value=mock_security_store):
            refresh_token = await create_refresh_token(test_user)
            
            # Mock successful refresh token retrieval
            mock_security_store.get_refresh_token.return_value = {
                "user_id": test_user.user_id,
                "username": test_user.username,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(days=30)
            }
            
            # Mock security event logging
            with patch('api.auth.log_security_event') as mock_log:
                # Use refresh token to get new access token
                new_access_token, new_refresh_token = await refresh_access_token(refresh_token)
                
                assert new_access_token is not None
                assert new_refresh_token is not None
                assert new_refresh_token != refresh_token  # Token should be rotated
                
                # Verify old refresh token was revoked and new one stored
                mock_security_store.revoke_refresh_token.assert_called_once_with(refresh_token)
                assert mock_security_store.store_refresh_token.call_count >= 2  # Initial + rotation
                
                # Verify security event was logged
                mock_log.assert_called_once()
                call_args = mock_log.call_args
                assert call_args[1]['event_type'] == SecurityEventType.TOKEN_REFRESH
    
    @pytest.mark.asyncio
    async def test_async_refresh_token_invalid(self, mock_security_store):
        """Test async refresh token usage with invalid token."""
        invalid_refresh_token = "invalid_token_123"
        
        # Mock security store to return None for invalid token
        mock_security_store.get_refresh_token.return_value = None
        
        with patch('api.auth.get_security_store', return_value=mock_security_store):
            result = await refresh_access_token(invalid_refresh_token)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_get_current_user_async(self, test_user, mock_security_store):
        """Test async current user retrieval."""
        token = create_access_token(test_user)
        credentials = MagicMock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = token
        
        # Mock security store to return token not revoked
        mock_security_store.is_token_revoked.return_value = False
        
        with patch('api.auth.get_security_store', return_value=mock_security_store):
            current_user = await get_current_user(credentials)
            
            assert current_user is not None
            assert current_user.username == test_user.username
            assert current_user.user_id == test_user.user_id
    
    @pytest.mark.asyncio
    async def test_get_optional_user_async(self, test_user, mock_security_store):
        """Test async optional user retrieval."""
        token = create_access_token(test_user)
        credentials = MagicMock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = token
        
        # Mock security store to return token not revoked
        mock_security_store.is_token_revoked.return_value = False
        
        with patch('api.auth.get_security_store', return_value=mock_security_store):
            optional_user = await get_optional_user(credentials)
            
            assert optional_user is not None
            assert optional_user.username == test_user.username
    
    @pytest.mark.asyncio
    async def test_get_optional_user_none(self):
        """Test optional user with no credentials."""
        optional_user = await get_optional_user(None)
        assert optional_user is None
    
    @pytest.mark.asyncio
    async def test_get_optional_user_invalid_token(self, mock_security_store):
        """Test optional user with invalid/revoked token."""
        credentials = MagicMock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = "invalid_token"
        
        # Mock security store to indicate token is revoked
        mock_security_store.is_token_revoked.return_value = True
        
        with patch('api.auth.get_security_store', return_value=mock_security_store):
            optional_user = await get_optional_user(credentials)
            
            assert optional_user is None


class TestSecurityEventLogging:
    """Test suite for security event logging integration."""
    
    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(
            user_id="security_test_user",
            username="security_user",
            roles=[UserRole.ADMIN],
            permissions=["read:system", "write:system"],
            is_active=True
        )
    
    @pytest.mark.asyncio
    async def test_security_event_on_token_creation(self, test_user):
        """Test security event logging on token creation."""
        with patch('api.auth.log_security_event') as mock_log:
            token = create_access_token(test_user)
            
            # Token creation itself doesn't log (handled by login endpoint)
            # But verify token structure includes necessary audit fields
            import jwt
            from api.auth import JWT_SECRET_KEY, JWT_ALGORITHM
            
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            assert 'jti' in payload  # JWT ID for revocation tracking
            assert 'user_id' in payload
            assert 'username' in payload
    
    @pytest.mark.asyncio
    async def test_concurrent_token_operations(self, test_user):
        """Test concurrent token operations with async safety."""
        mock_security_store = AsyncMock(spec=PersistentSecurityStore)
        mock_security_store.revoke_token.return_value = True
        mock_security_store.is_token_revoked.return_value = False
        
        # Create multiple tokens
        tokens = [create_access_token(test_user) for _ in range(5)]
        
        with patch('api.auth.get_security_store', return_value=mock_security_store):
            with patch('api.auth.log_security_event'):
                # Concurrently revoke all tokens
                tasks = [revoke_token(token) for token in tokens]
                await asyncio.gather(*tasks)
                
                # Verify all revocations were processed
                assert mock_security_store.revoke_token.call_count == 5


class TestBruteForceProtectionIntegration:
    """Test brute force protection with Redis persistence."""
    
    @pytest.mark.asyncio
    async def test_brute_force_with_persistent_store(self):
        """Test brute force protection integration with persistent store."""
        username = "brute_force_test"
        
        # This would require implementing Redis-based brute force tracking
        # For now, test the existing in-memory implementation
        from api.auth import check_brute_force, record_failed_login
        
        # Record multiple failed attempts
        for _ in range(5):
            record_failed_login(username)
        
        # Should be locked out
        is_locked = check_brute_force(username)
        assert is_locked is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
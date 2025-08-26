#!/usr/bin/env python3
"""
Basic authentication tests for the updated persistent user system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
import os
import sys

# Add parent directories to path for imports  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.auth import (
    create_access_token, verify_access_token, 
    User, UserRole, TokenData
)
from trading_common.user_management import UserRole as UserMgmtRole, UserStatus


class TestTokenOperations:
    """Test JWT token creation and verification."""
    
    def test_create_access_token(self):
        """Test access token creation."""
        user = User(
            user_id="test-user-123",
            username="testuser", 
            email="test@example.com",
            role=UserRole.TRADER,
            permissions=["read:data", "write:orders"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        token = create_access_token(user)
        
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are typically long
        assert "." in token  # JWT format has dots
        
    @pytest.mark.asyncio
    async def test_verify_access_token_valid(self):
        """Test verification of valid access token."""
        user = User(
            user_id="test-user-123",
            username="testuser",
            email="test@example.com", 
            role=UserRole.TRADER,
            permissions=["read:data", "write:orders"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Create token with longer expiry for testing
        token = create_access_token(user, expires_delta=timedelta(hours=1))
        
        # Mock the security store to avoid actual Redis calls
        with patch('api.auth.get_security_store') as mock_store:
            mock_security_store = AsyncMock()
            mock_security_store.is_token_revoked.return_value = False
            mock_store.return_value = mock_security_store
            
            token_data = await verify_access_token(token)
            
            assert token_data.username == "testuser"
            assert token_data.user_id == "test-user-123"
            assert token_data.role == UserRole.TRADER.value
            assert "read:data" in token_data.permissions
            
    @pytest.mark.asyncio
    async def test_verify_access_token_expired(self):
        """Test verification of expired token."""
        user = User(
            user_id="test-user-123",
            username="testuser",
            email="test@example.com",
            role=UserRole.TRADER, 
            permissions=["read:data"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Create token that expires immediately
        token = create_access_token(user, expires_delta=timedelta(seconds=-1))
        
        with patch('api.auth.get_security_store') as mock_store:
            mock_security_store = AsyncMock()
            mock_security_store.is_token_revoked.return_value = False
            mock_store.return_value = mock_security_store
            
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await verify_access_token(token)
            
            assert exc_info.value.status_code == 401
            assert "expired" in exc_info.value.detail.lower()


class TestUserRoles:
    """Test user role enumeration."""
    
    def test_user_roles_defined(self):
        """Test that all expected user roles are defined."""
        expected_roles = ["super_admin", "admin", "trader", "analyst", "api_user", "viewer"]
        
        for role_value in expected_roles:
            assert any(role.value == role_value for role in UserRole)
            
    def test_user_role_values(self):
        """Test specific user role values."""
        assert UserRole.SUPER_ADMIN.value == "super_admin"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.TRADER.value == "trader"
        assert UserRole.VIEWER.value == "viewer"


class TestUserModel:
    """Test User model validation."""
    
    def test_user_creation(self):
        """Test creating a User instance."""
        user = User(
            user_id="test-123",
            username="testuser",
            email="test@example.com",
            role=UserRole.ADMIN,
            permissions=["admin:all"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        assert user.user_id == "test-123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.ADMIN
        assert "admin:all" in user.permissions
        assert user.is_active is True  # Default value


class TestTokenDataModel:
    """Test TokenData model validation."""
    
    def test_token_data_creation(self):
        """Test creating a TokenData instance."""
        exp_time = datetime.utcnow() + timedelta(hours=1)
        iat_time = datetime.utcnow()
        
        token_data = TokenData(
            user_id="test-123",
            username="testuser", 
            role="trader",
            permissions=["read:data", "write:orders"],
            exp=exp_time,
            iat=iat_time,
            jti="unique-token-id"
        )
        
        assert token_data.user_id == "test-123"
        assert token_data.username == "testuser"
        assert token_data.role == "trader"
        assert len(token_data.permissions) == 2
        assert token_data.exp == exp_time
        assert token_data.iat == iat_time
        assert token_data.iss == "ai-trading-system"  # Default value
        assert token_data.aud == "trading-api"  # Default value
        assert token_data.jti == "unique-token-id"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
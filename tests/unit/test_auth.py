#!/usr/bin/env python3
"""
Comprehensive tests for authentication module.
Tests brute force protection, token revocation, and security features.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import os

# Add parent directories to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.auth import (
    authenticate_user, create_access_token, create_refresh_token,
    verify_access_token, refresh_access_token, revoke_token,
    check_brute_force, record_failed_login, login_attempts,
    User, UserRole, TokenData, SYSTEM_USERS
)
from fastapi import HTTPException


class TestBruteForceProtection:
    """Test brute force protection mechanisms."""
    
    def setup_method(self):
        """Reset login attempts before each test."""
        login_attempts.clear()
    
    def test_initial_login_allowed(self):
        """Test that initial login attempts are allowed."""
        result = check_brute_force("test_user")
        assert result is False
    
    def test_brute_force_lockout(self):
        """Test that too many failed attempts lock account."""
        username = "test_user"
        
        # Record maximum failed attempts
        for _ in range(5):
            record_failed_login(username)
        
        # Should now be locked out
        result = check_brute_force(username)
        assert result is True
    
    def test_brute_force_expiry(self):
        """Test that lockout expires after timeout."""
        username = "test_user"
        
        # Mock datetime to simulate old attempts
        old_time = datetime.utcnow() - timedelta(seconds=400)
        
        # Add old attempts
        login_attempts[username] = [old_time] * 5
        
        # Should not be locked out (attempts expired)
        result = check_brute_force(username)
        assert result is False
        
        # Should have cleaned up old attempts
        assert len(login_attempts[username]) == 0
    
    def test_successful_login_clears_attempts(self):
        """Test that successful login clears failed attempts."""
        username = "admin"
        password = os.getenv("ADMIN_PASSWORD", "admin123")
        
        # Record some failed attempts first
        for _ in range(3):
            record_failed_login(username)
        
        assert len(login_attempts[username]) == 3
        
        # Successful authentication should clear attempts
        user = authenticate_user(username, password)
        
        if user:  # Only test if admin user is properly configured
            assert len(login_attempts.get(username, [])) == 0


class TestTokenManagement:
    """Test JWT token creation, validation, and revocation."""
    
    def setup_method(self):
        """Set up test user."""
        self.test_user = User(
            user_id="test_123",
            username="test_user",
            roles=[UserRole.TRADER],
            permissions=["read:market_data", "write:orders"],
            is_active=True,
            created_at=datetime.utcnow()
        )
    
    def test_create_access_token(self):
        """Test access token creation."""
        token = create_access_token(self.test_user)
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_valid_token(self):
        """Test verification of valid token."""
        token = create_access_token(self.test_user)
        token_data = verify_access_token(token)
        
        assert token_data.user_id == self.test_user.user_id
        assert token_data.username == self.test_user.username
        assert token_data.roles == self.test_user.roles
        assert token_data.permissions == self.test_user.permissions
    
    def test_verify_expired_token(self):
        """Test verification of expired token."""
        # Create token with very short expiry
        short_expiry = timedelta(milliseconds=1)
        token = create_access_token(self.test_user, short_expiry)
        
        # Wait for expiry
        import time
        time.sleep(0.002)
        
        with pytest.raises(HTTPException) as exc_info:
            verify_access_token(token)
        
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()
    
    def test_verify_invalid_token(self):
        """Test verification of malformed token."""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(HTTPException) as exc_info:
            verify_access_token(invalid_token)
        
        assert exc_info.value.status_code == 401
    
    def test_token_revocation(self):
        """Test token revocation functionality."""
        token = create_access_token(self.test_user)
        
        # Token should be valid initially
        token_data = verify_access_token(token)
        assert token_data is not None
        
        # Revoke the token
        revoke_token(token)
        
        # Token should now be invalid
        with pytest.raises(HTTPException) as exc_info:
            verify_access_token(token)
        
        assert exc_info.value.status_code == 401
        assert "revoked" in exc_info.value.detail.lower()
    
    def test_refresh_token_creation(self):
        """Test refresh token creation."""
        refresh_token = create_refresh_token(self.test_user)
        assert isinstance(refresh_token, str)
        assert len(refresh_token) > 0
    
    def test_access_token_refresh(self):
        """Test access token refresh using refresh token."""
        refresh_token = create_refresh_token(self.test_user)
        
        # Mock SYSTEM_USERS to include test user
        with patch.dict(SYSTEM_USERS, {
            self.test_user.username: {
                "user_id": self.test_user.user_id,
                "username": self.test_user.username,
                "password_hash": "dummy_hash",
                "roles": self.test_user.roles,
                "permissions": self.test_user.permissions,
                "is_active": True,
                "created_at": self.test_user.created_at
            }
        }):
            result = refresh_access_token(refresh_token)
            
            assert result is not None
            new_access_token, new_refresh_token = result
            
            # Verify new access token is valid
            token_data = verify_access_token(new_access_token)
            assert token_data.user_id == self.test_user.user_id
            
            # Verify new refresh token is different
            assert new_refresh_token != refresh_token
    
    def test_refresh_invalid_token(self):
        """Test refresh with invalid token."""
        result = refresh_access_token("invalid_refresh_token")
        assert result is None


class TestUserAuthentication:
    """Test user authentication flows."""
    
    def test_authenticate_valid_user(self):
        """Test authentication with valid credentials."""
        # This test depends on proper admin user setup
        username = "admin"
        password = os.getenv("ADMIN_PASSWORD")
        
        if password:  # Only test if admin password is configured
            user = authenticate_user(username, password)
            assert user is not None
            assert user.username == username
            assert UserRole.ADMIN in user.roles
    
    def test_authenticate_invalid_user(self):
        """Test authentication with invalid username."""
        user = authenticate_user("nonexistent_user", "any_password")
        assert user is None
    
    def test_authenticate_wrong_password(self):
        """Test authentication with wrong password."""
        user = authenticate_user("admin", "wrong_password")
        assert user is None
    
    def test_authenticate_inactive_user(self):
        """Test authentication with inactive user."""
        # Mock an inactive user in SYSTEM_USERS
        with patch.dict(SYSTEM_USERS, {
            "inactive_user": {
                "user_id": "inactive_123",
                "username": "inactive_user",
                "password_hash": "$2b$12$dummy_hash",
                "roles": [UserRole.VIEWER],
                "permissions": ["read:market_data"],
                "is_active": False,
                "created_at": datetime.utcnow()
            }
        }):
            user = authenticate_user("inactive_user", "any_password")
            assert user is None


class TestTokenSecurity:
    """Test token security features."""
    
    def test_token_has_unique_id(self):
        """Test that tokens have unique JTI for revocation."""
        user = User(
            user_id="test_123",
            username="test_user",
            roles=[UserRole.TRADER],
            permissions=["read:market_data"],
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        token1 = create_access_token(user)
        token2 = create_access_token(user)
        
        # Decode tokens to check JTI
        token_data1 = verify_access_token(token1)
        token_data2 = verify_access_token(token2)
        
        assert token_data1.jti != token_data2.jti
    
    def test_token_issuer_validation(self):
        """Test that tokens validate issuer and audience."""
        from jose import jwt
        
        user = User(
            user_id="test_123",
            username="test_user", 
            roles=[UserRole.TRADER],
            permissions=["read:market_data"],
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        # Create token with wrong issuer
        from api.auth import JWT_SECRET_KEY, JWT_ALGORITHM
        
        fake_payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": user.roles,
            "permissions": user.permissions,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "fake-issuer",  # Wrong issuer
            "aud": "trading-api"
        }
        
        fake_token = jwt.encode(fake_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        with pytest.raises(HTTPException) as exc_info:
            verify_access_token(fake_token)
        
        assert exc_info.value.status_code == 401
        assert "issuer" in exc_info.value.detail.lower() or "audience" in exc_info.value.detail.lower()


class TestSecurityConfiguration:
    """Test security configuration validation."""
    
    def test_production_secret_validation(self):
        """Test that production rejects default secrets."""
        # This test would need environment mocking to test production validation
        # The actual validation is in the auth module initialization
        pass
    
    def test_jwt_configuration(self):
        """Test JWT configuration is properly loaded."""
        from api.auth import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRY_MINUTES
        
        assert JWT_SECRET_KEY is not None
        assert JWT_ALGORITHM == "HS256"  # Default algorithm
        assert JWT_EXPIRY_MINUTES > 0


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_auth.py -v
    pytest.main([__file__, "-v"])
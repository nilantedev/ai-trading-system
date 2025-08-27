#!/usr/bin/env python3
"""
JWT Authentication Module for AI Trading System API
Provides secure JWT token generation, validation, and user management.
"""

import os
import secrets
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from enum import Enum
import logging
import hashlib
import hmac
from collections import defaultdict
import asyncio

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_common import get_settings
from shared.logging_config import get_logger
from trading_common.security_store import (
    get_security_store, log_security_event, SecurityEventType,
    UserSession, RefreshToken
)
from trading_common.user_management import get_user_manager, UserRole as UserMgmtRole, UserStatus as UserMgmtStatus

logger = get_logger(__name__)
settings = get_settings()

# Security configuration
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Brute force protection - now using persistent store
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 300  # 5 minutes in seconds

# JWT configuration - use unified settings
JWT_SECRET_KEY = settings.security.secret_key
JWT_ALGORITHM = settings.security.jwt_algorithm
JWT_EXPIRY_MINUTES = settings.security.jwt_expire_minutes

# Legacy environment variable support with deprecation warning
if os.getenv("JWT_SECRET_KEY") and os.getenv("JWT_SECRET_KEY") != settings.security.secret_key:
    logger.warning("Legacy JWT_SECRET_KEY detected. Please use SECURITY_SECRET_KEY instead.")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

# Production validation
if settings.is_production and len(JWT_SECRET_KEY) < 32:
    logger.error("CRITICAL: JWT secret key must be at least 32 characters in production!")
    raise ValueError("Production deployment requires secure secret key")

# Initialize user manager for persistent authentication
user_manager = get_user_manager()


class UserRole(str, Enum):
    """User roles for permission management."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    API_USER = "api_user"
    VIEWER = "viewer"


class User(BaseModel):
    """User model with permissions."""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[str]
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None


class TokenData(BaseModel):
    """JWT token payload data."""
    user_id: str
    username: str
    role: str
    permissions: List[str]
    exp: datetime
    iat: datetime
    iss: str = "ai-trading-system"
    aud: str = "trading-api"
    jti: Optional[str] = None  # JWT ID for token revocation


# Token storage now handled by persistent security store
# revoked_tokens and refresh_tokens replaced with Redis-backed store

# User authentication now handled by persistent user management system
# No more in-memory SYSTEM_USERS dictionary


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


async def check_brute_force(username: str) -> bool:
    """Check if account is locked due to brute force attempts using persistent store."""
    try:
        security_store = await get_security_store()
        is_locked = await security_store.is_account_locked(username, MAX_LOGIN_ATTEMPTS)
        
        if is_locked:
            logger.warning(f"Account {username} is locked due to excessive failed login attempts")
            
            # Log security event for account lockout
            await log_security_event(
                event_type=SecurityEventType.ACCOUNT_LOCKED,
                success=False,
                username=username,
                details={"max_attempts": MAX_LOGIN_ATTEMPTS, "lockout_duration": LOCKOUT_DURATION},
                severity="WARNING"
            )
        
        return is_locked
    except Exception as e:
        logger.error(f"Error checking brute force protection: {e}")
        # Fail open to avoid blocking legitimate users due to store issues
        return False

async def record_failed_login(username: str, ip_address: Optional[str] = None):
    """Record a failed login attempt using persistent store."""
    try:
        security_store = await get_security_store()
        attempt_count = await security_store.record_login_attempt(username, False, ip_address or "unknown")
        
        logger.warning(f"Failed login attempt for {username}. Total recent attempts: {attempt_count}")
        
        # Log security event for failed login
        await log_security_event(
            event_type=SecurityEventType.LOGIN_FAILURE,
            success=False,
            username=username,
            ip_address=ip_address,
            details={"attempt_count": attempt_count},
            severity="WARNING" if attempt_count >= MAX_LOGIN_ATTEMPTS - 1 else "INFO"
        )
        
    except Exception as e:
        logger.error(f"Error recording failed login attempt: {e}")

async def record_successful_login(username: str, ip_address: Optional[str] = None):
    """Record a successful login attempt using persistent store."""
    try:
        security_store = await get_security_store()
        await security_store.record_login_attempt(username, True, ip_address or "unknown")
        
        # Log security event for successful login
        await log_security_event(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            success=True,
            username=username,
            ip_address=ip_address,
            details={},
            severity="INFO"
        )
        
    except Exception as e:
        logger.error(f"Error recording successful login attempt: {e}")

async def authenticate_user(username: str, password: str, ip_address: Optional[str] = None) -> Optional[User]:
    """Authenticate user with username and password using persistent user management."""
    try:
        # Use the user manager for authentication
        user_mgmt_user = await user_manager.authenticate_user(username, password, ip_address)
        
        if not user_mgmt_user:
            return None
        
        # Convert user management User to API User model
        api_user = User(
            user_id=user_mgmt_user.user_id,
            username=user_mgmt_user.username,
            email=user_mgmt_user.email,
            role=UserRole(user_mgmt_user.role.value),
            permissions=list(user_mgmt_user.permissions),
            is_active=user_mgmt_user.status.value == "active",
            created_at=user_mgmt_user.created_at,
            updated_at=user_mgmt_user.updated_at,
            last_login=user_mgmt_user.last_login
        )
        
        return api_user
        
    except Exception as e:
        logger.error(f"Authentication error for user {username}: {e}")
        return None


def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token for user."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    
    # Add unique token ID for revocation
    token_id = secrets.token_urlsafe(32)
    
    iat_time = datetime.utcnow()
    
    # Create payload with Unix timestamps for exp and iat
    payload = {
        "user_id": user.user_id,
        "username": user.username, 
        "role": user.role.value,
        "permissions": user.permissions,
        "exp": int(expire.timestamp()),  # Convert to Unix timestamp
        "iat": int(iat_time.timestamp()),  # Convert to Unix timestamp
        "iss": "ai-trading-system",
        "aud": "trading-api",
        "jti": token_id
    }
    
    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )

async def create_refresh_token(user: User) -> str:
    """Create refresh token for user using persistent store."""
    return await store_refresh_token(user)

async def store_refresh_token(user: User) -> str:
    """Store refresh token in persistent security store."""
    token = secrets.token_urlsafe(64)
    security_store = await get_security_store()
    
    token_data = RefreshToken(
        token_hash="",  # Will be generated by security store
        user_id=user.user_id,
        username=user.username,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(days=30)
    )
    
    await security_store.store_refresh_token(token, token_data)
    return token

async def refresh_access_token(refresh_token: str) -> Optional[Tuple[str, str]]:
    """Refresh access token using persistent store."""
    try:
        security_store = await get_security_store()
        token_data = await security_store.get_refresh_token(refresh_token)
        
        if not token_data or token_data.is_revoked:
            return None
        
        # Check if token is expired
        if token_data.expires_at < datetime.utcnow():
            return None
        
        # Get user from user manager
        user_mgmt_user = await user_manager._get_user_by_username(token_data.username)
        if not user_mgmt_user or user_mgmt_user.status.value != "active":
            return None
        
        # Convert to API User model
        api_user = User(
            user_id=user_mgmt_user.user_id,
            username=user_mgmt_user.username,
            email=user_mgmt_user.email,
            role=UserRole(user_mgmt_user.role.value),
            permissions=list(user_mgmt_user.permissions),
            is_active=True,
            created_at=user_mgmt_user.created_at,
            updated_at=user_mgmt_user.updated_at,
            last_login=user_mgmt_user.last_login
        )
        
        # Create new access token
        new_access_token = create_access_token(api_user)
        
        # Rotate refresh token (remove old, create new)
        await security_store.revoke_refresh_token(refresh_token)
        new_refresh_token = await store_refresh_token(api_user)
        
        # Log security event
        await log_security_event(
            event_type=SecurityEventType.TOKEN_REFRESH,
            success=True,
            user_id=api_user.user_id,
            username=api_user.username
        )
        
        return new_access_token, new_refresh_token
        
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        return None

async def revoke_token(token: str):
    """Revoke an access token using persistent store."""
    try:
        payload = jwt.decode(
            token, 
            JWT_SECRET_KEY, 
            algorithms=[JWT_ALGORITHM],
            audience="trading-api",
            issuer="ai-trading-system"
        )
        jti = payload.get("jti")
        exp = payload.get("exp")
        
        if jti and exp:
            expires_at = datetime.fromtimestamp(exp)
            security_store = await get_security_store()
            await security_store.revoke_token(jti, expires_at)
            
            # Log security event
            await log_security_event(
                event_type=SecurityEventType.TOKEN_REVOKED,
                success=True,
                user_id=payload.get("user_id"),
                username=payload.get("username"),
                details={"token_jti": jti}
            )
            
            logger.info(f"Token {jti} revoked")
    except JWTError as e:
        logger.warning(f"Failed to revoke token: {e}")


async def verify_access_token(token: str) -> TokenData:
    """Verify and decode JWT access token using persistent store."""
    try:
        payload = jwt.decode(
            token, 
            JWT_SECRET_KEY, 
            algorithms=[JWT_ALGORITHM],
            audience="trading-api",
            issuer="ai-trading-system"
        )
        
        # Check if token is revoked using persistent store
        jti = payload.get("jti")
        if jti:
            security_store = await get_security_store()
            if await security_store.is_token_revoked(jti):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
        
        # Validate required fields
        user_id = payload.get("user_id")
        username = payload.get("username")
        
        if not user_id or not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Check expiry - exp is in seconds since epoch
        exp = payload.get("exp")
        if exp:
            # Convert exp to datetime for comparison
            exp_datetime = datetime.fromtimestamp(exp)
            current_time = datetime.utcnow()
            if exp_datetime < current_time:
                logger.debug(f"Token expired: exp={exp_datetime}, current={current_time}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
        
        # Validate issuer/audience
        iss = payload.get("iss", "")
        aud = payload.get("aud", "")
        if iss != "ai-trading-system" or aud != "trading-api":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token issuer or audience"
            )
        
        return TokenData(**payload)
        
    except JWTError as e:
        logger.warning(f"JWT validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token."""
    try:
        token_data = await verify_access_token(credentials.credentials)
        
        # Get user from user manager
        user_mgmt_user = await user_manager._get_user_by_username(token_data.username)
        if not user_mgmt_user or user_mgmt_user.status.value != "active":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Convert to API User model
        api_user = User(
            user_id=user_mgmt_user.user_id,
            username=user_mgmt_user.username,
            email=user_mgmt_user.email,
            role=UserRole(user_mgmt_user.role.value),
            permissions=list(user_mgmt_user.permissions),
            is_active=True,
            created_at=user_mgmt_user.created_at,
            updated_at=user_mgmt_user.updated_at,
            last_login=user_mgmt_user.last_login
        )
        
        return api_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[User]:
    """Get optional user from JWT token - returns None if no token or invalid."""
    if not credentials:
        return None
    
    try:
        token_data = await verify_access_token(credentials.credentials)
        
        # Get user from user manager
        user_mgmt_user = await user_manager._get_user_by_username(token_data.username)
        if not user_mgmt_user or user_mgmt_user.status.value != "active":
            return None
        
        # Convert to API User model
        api_user = User(
            user_id=user_mgmt_user.user_id,
            username=user_mgmt_user.username,
            email=user_mgmt_user.email,
            role=UserRole(user_mgmt_user.role.value),
            permissions=list(user_mgmt_user.permissions),
            is_active=True,
            created_at=user_mgmt_user.created_at,
            updated_at=user_mgmt_user.updated_at,
            last_login=user_mgmt_user.last_login
        )
        
        return api_user
        
    except (JWTError, HTTPException, ValueError, Exception):
        return None


def require_permission(permission: str):
    """Dependency to require specific permission."""
    async def permission_checker(current_user: User = Depends(get_current_active_user)):
        if permission not in current_user.permissions and "admin:all" not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        return current_user
    return permission_checker


def require_role(role: UserRole):
    """Dependency to require specific role."""
    async def role_checker(current_user: User = Depends(get_current_active_user)):
        # Allow super admin and admin to access everything
        if current_user.role in [UserRole.SUPER_ADMIN, UserRole.ADMIN]:
            return current_user
        
        # Check specific role match
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {role}, current: {current_user.role}"
            )
        return current_user
    return role_checker




# Health check for auth system
async def get_auth_health() -> Dict[str, Any]:
    """Get authentication system health status."""
    try:
        # Check user manager availability
        user_manager_available = user_manager is not None and user_manager._session_factory is not None
        
        return {
            "status": "healthy",
            "jwt_configured": bool(JWT_SECRET_KEY and len(JWT_SECRET_KEY) >= 16),
            "user_manager_available": user_manager_available,
            "database_connected": user_manager_available,
            "settings_loaded": settings is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
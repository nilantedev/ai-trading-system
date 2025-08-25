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

from trading_common import get_settings, get_logger
from trading_common.security_store import (
    get_security_store, log_security_event, SecurityEventType,
    UserSession, RefreshToken
)

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
if settings.environment == "production" and JWT_SECRET_KEY == "dev-secret-change-in-production":
    logger.error("CRITICAL: Default secret key detected in production! Set SECURITY_SECRET_KEY immediately.")
    raise ValueError("Production deployment requires secure secret key")

# Admin configuration
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")  # Pre-hashed bcrypt password

# Enforce password hash requirement in non-development environments
if not ADMIN_PASSWORD_HASH:
    if settings.environment != "development":
        logger.critical("ADMIN_PASSWORD_HASH must be set in non-development environments!")
        raise ValueError("Security Error: ADMIN_PASSWORD_HASH is required for this environment")
    else:
        # Development only: generate hash from ADMIN_PASSWORD env var
        dev_password = os.getenv("ADMIN_PASSWORD")
        if not dev_password:
            logger.critical("Either ADMIN_PASSWORD_HASH or ADMIN_PASSWORD must be set")
            raise ValueError("Security Error: Admin credentials not configured")
        ADMIN_PASSWORD_HASH = pwd_context.hash(dev_password)
        logger.warning("Development mode: Using ADMIN_PASSWORD env var (not for production)")


class UserRole(str, Enum):
    """User roles for permission management."""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"


class User(BaseModel):
    """User model with permissions."""
    user_id: str
    username: str
    roles: List[UserRole]
    permissions: List[str]
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class TokenData(BaseModel):
    """JWT token payload data."""
    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]
    exp: datetime
    iat: datetime
    iss: str = "ai-trading-system"
    aud: str = "trading-api"
    jti: Optional[str] = None  # JWT ID for token revocation


# Token storage now handled by persistent security store
# revoked_tokens and refresh_tokens replaced with Redis-backed store

# Predefined users (in production, use database)
SYSTEM_USERS = {
    ADMIN_USERNAME: {
        "user_id": "admin_001",
        "username": ADMIN_USERNAME,
        "password_hash": ADMIN_PASSWORD_HASH,
        "roles": [UserRole.ADMIN],
        "permissions": [
            "read:market_data",
            "write:orders",
            "read:portfolio", 
            "write:portfolio",
            "read:system",
            "write:system",
            "admin:all"
        ],
        "is_active": True,
        "created_at": datetime.utcnow()
    }
}


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
    """Authenticate user with username and password using persistent security store."""
    # Check brute force protection
    if await check_brute_force(username):
        return None
    
    user_data = SYSTEM_USERS.get(username)
    if not user_data:
        await record_failed_login(username, ip_address)
        return None
    
    if not verify_password(password, user_data["password_hash"]):
        await record_failed_login(username, ip_address)
        return None
    
    if not user_data["is_active"]:
        await record_failed_login(username, ip_address)
        return None
    
    # Record successful login (clears failed attempts automatically)
    await record_successful_login(username, ip_address)
    
    # Update last login
    user_data["last_login"] = datetime.utcnow()
    
    return User(**{k: v for k, v in user_data.items() if k != "password_hash"})


def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token for user."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    
    # Add unique token ID for revocation
    token_id = secrets.token_urlsafe(32)
    
    token_data = TokenData(
        user_id=user.user_id,
        username=user.username,
        roles=user.roles,
        permissions=user.permissions,
        exp=expire,
        iat=datetime.utcnow(),
        jti=token_id  # JWT ID for revocation
    )
    
    return jwt.encode(
        token_data.model_dump(),
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
    security_store = await get_security_store()
    token_data = await security_store.get_refresh_token(refresh_token)
    
    if not token_data or token_data.is_revoked:
        return None
    
    # Check if token is expired
    if token_data.expires_at < datetime.utcnow():
        return None
    
    # Get user
    user_data = SYSTEM_USERS.get(token_data.username)
    if not user_data or not user_data["is_active"]:
        return None
    
    user = User(**{k: v for k, v in user_data.items() if k != "password_hash"})
    
    # Create new access token
    new_access_token = create_access_token(user)
    
    # Rotate refresh token (remove old, create new)
    await security_store.revoke_refresh_token(refresh_token)
    new_refresh_token = await store_refresh_token(user)
    
    # Log security event
    await log_security_event(
        event_type=SecurityEventType.TOKEN_REFRESH,
        success=True,
        user_id=user.user_id,
        username=user.username
    )
    
    return new_access_token, new_refresh_token

async def revoke_token(token: str):
    """Revoke an access token using persistent store."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
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
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
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
        
        # Check expiry
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
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
    token_data = await verify_access_token(credentials.credentials)
    
    # Get user from system users
    user_data = SYSTEM_USERS.get(token_data.username)
    if not user_data or not user_data["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return User(**{k: v for k, v in user_data.items() if k != "password_hash"})


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
        
        # Get user from system users
        user_data = SYSTEM_USERS.get(token_data.username)
        if not user_data or not user_data["is_active"]:
            return None
        
        return User(**{k: v for k, v in user_data.items() if k != "password_hash"})
    except (JWTError, HTTPException, ValueError):
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
        if role not in current_user.roles and UserRole.ADMIN not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {role}"
            )
        return current_user
    return role_checker




# Health check for auth system
def get_auth_health() -> Dict[str, Any]:
    """Get authentication system health status."""
    return {
        "status": "healthy",
        "jwt_configured": bool(JWT_SECRET_KEY),
        "admin_configured": ADMIN_USERNAME in SYSTEM_USERS,
        "total_users": len(SYSTEM_USERS),
        "active_users": sum(1 for u in SYSTEM_USERS.values() if u["is_active"]),
        "timestamp": datetime.utcnow().isoformat()
    }
#!/usr/bin/env python3
"""
JWT Authentication Module for AI Trading System API
Provides secure JWT token generation, validation, and user management.
"""

import os
import secrets
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from enum import Enum
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_common import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Security configuration
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

# If no admin password hash is provided, create one for demo (MUST be changed in production)
if not ADMIN_PASSWORD_HASH:
    # Default password: "TradingSystem2024!" - CHANGE THIS IN PRODUCTION
    default_password = os.getenv("ADMIN_PASSWORD", "TradingSystem2024!")
    ADMIN_PASSWORD_HASH = pwd_context.hash(default_password)
    logger.warning("Using default admin password - CHANGE THIS IN PRODUCTION!")


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


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password."""
    user_data = SYSTEM_USERS.get(username)
    if not user_data:
        return None
    
    if not verify_password(password, user_data["password_hash"]):
        return None
    
    if not user_data["is_active"]:
        return None
    
    # Update last login
    user_data["last_login"] = datetime.utcnow()
    
    return User(**{k: v for k, v in user_data.items() if k != "password_hash"})


def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token for user."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    
    token_data = TokenData(
        user_id=user.user_id,
        username=user.username,
        roles=user.roles,
        permissions=user.permissions,
        exp=expire,
        iat=datetime.utcnow()
    )
    
    return jwt.encode(
        token_data.model_dump(),
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )


def verify_access_token(token: str) -> TokenData:
    """Verify and decode JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
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
    token_data = verify_access_token(credentials.credentials)
    
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
        token_data = verify_access_token(credentials.credentials)
        
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


async def get_optional_user(
    authorization: Optional[str] = Depends(lambda: None)
) -> Optional[User]:
    """Optional authentication for public endpoints."""
    if not authorization:
        return None
    
    try:
        if authorization.startswith("Bearer "):
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=authorization[7:]
            )
            return await get_current_user(credentials)
    except HTTPException:
        # Invalid token - return None for optional auth
        pass
    
    return None


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
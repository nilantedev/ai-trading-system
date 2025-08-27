#!/usr/bin/env python3
"""
Consolidated JWT Authentication Module for AI Trading System
Combines JWT rotation, revocation, and security features from multiple modules
"""

import os
import jwt
import json
import time
import uuid
import secrets
import hashlib
import hmac
import asyncio
import aioredis
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
security = HTTPBearer()


class UserRole(str, Enum):
    """User roles for the trading system"""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"
    SERVICE = "service"


class TokenType(Enum):
    """Token types"""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    SERVICE = "service"


@dataclass
class JWTKey:
    """JWT signing key with metadata"""
    kid: str                # Key ID
    secret: str             # Signing secret
    algorithm: str          # Signing algorithm
    created_at: datetime    # Creation timestamp
    expires_at: datetime    # Expiration timestamp
    is_active: bool        # Whether key can sign new tokens
    is_valid: bool         # Whether key can verify tokens
    rotation_version: int   # Rotation generation


class User(BaseModel):
    """User model"""
    user_id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    is_active: bool = True  # Added for auth router
    roles: List[str] = []
    permissions: List[str] = []  # Added for auth router
    api_key_id: Optional[str] = None
    last_login: Optional[datetime] = None
    created_at: datetime = datetime.now(timezone.utc)  # Added for auth router
    mfa_enabled: bool = False


class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[str] = []
    jti: Optional[str] = None
    token_type: str = "access"


class JWTAuthManager:
    """
    Unified JWT authentication manager with key rotation and revocation
    """
    
    def __init__(self):
        # Redis connection
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.redis: Optional[aioredis.Redis] = None
        
        # JWT configuration
        self.issuer = os.getenv("JWT_ISSUER", "ai-trading-system")
        self.audience = os.getenv("JWT_AUDIENCE", "trading-api")
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        
        # Key rotation settings
        self.key_rotation_days = int(os.getenv("JWT_KEY_ROTATION_DAYS", "30"))
        self.key_overlap_hours = int(os.getenv("JWT_KEY_OVERLAP_HOURS", "24"))
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15
        self.require_mfa_for_sensitive = os.getenv("REQUIRE_MFA_SENSITIVE", "true").lower() == "true"
        
        # Key storage
        self.keys: Dict[str, JWTKey] = {}
        self.active_key: Optional[JWTKey] = None
        
        # Revocation tracking
        self.revoked_tokens: set = set()
        
        # Brute force protection
        self.failed_attempts: Dict[str, List[datetime]] = {}
        
        # Initialize default key if needed
        self._init_default_key()
    
    def _init_default_key(self):
        """Initialize a default key if none exists"""
        if not self.active_key:
            kid = f"key-{uuid.uuid4().hex[:8]}-{int(time.time())}"
            secret = os.getenv("JWT_SECRET", secrets.token_urlsafe(64))
            
            self.active_key = JWTKey(
                kid=kid,
                secret=secret,
                algorithm="HS256",
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(days=self.key_rotation_days),
                is_active=True,
                is_valid=True,
                rotation_version=1
            )
            self.keys[kid] = self.active_key
    
    async def initialize(self):
        """Initialize Redis connection and load keys"""
        try:
            self.redis = aioredis.from_url(
                self.redis_url,
                password=self.redis_password,
                decode_responses=True
            )
            await self.redis.ping()
            logger.info("JWT Auth Manager initialized with Redis")
            
            # Load existing keys from Redis
            await self._load_keys_from_redis()
            
            # Start key rotation scheduler
            asyncio.create_task(self._key_rotation_scheduler())
            
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
    
    async def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create an access token"""
        if not self.active_key:
            raise RuntimeError("No active JWT key available")
        
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "iss": self.issuer,
            "aud": self.audience,
            "jti": str(uuid.uuid4()),
            "type": TokenType.ACCESS.value,
            "kid": self.active_key.kid
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.active_key.secret,
            algorithm=self.active_key.algorithm,
            headers={"kid": self.active_key.kid}
        )
        
        return encoded_jwt
    
    async def create_refresh_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a refresh token"""
        if not self.active_key:
            raise RuntimeError("No active JWT key available")
        
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "iss": self.issuer,
            "aud": self.audience,
            "jti": str(uuid.uuid4()),
            "type": TokenType.REFRESH.value,
            "kid": self.active_key.kid
        })
        
        # Store refresh token in Redis if available
        if self.redis:
            jti = to_encode["jti"]
            user_id = data.get("sub", "unknown")
            await self.redis.setex(
                f"refresh_token:{jti}",
                int((expire - datetime.now(timezone.utc)).total_seconds()),
                user_id
            )
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.active_key.secret,
            algorithm=self.active_key.algorithm,
            headers={"kid": self.active_key.kid}
        )
        
        return encoded_jwt
    
    async def verify_token(
        self,
        token: str,
        token_type: Optional[TokenType] = None
    ) -> Dict[str, Any]:
        """Verify a JWT token"""
        try:
            # Get kid from header
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            
            # Find appropriate key
            if kid and kid in self.keys:
                key = self.keys[kid]
                if not key.is_valid:
                    raise jwt.InvalidKeyError(f"Key {kid} is no longer valid")
            else:
                key = self.active_key
                if not key:
                    raise jwt.InvalidKeyError("No active key available")
            
            # Decode token
            payload = jwt.decode(
                token,
                key.secret,
                algorithms=[key.algorithm],
                audience=self.audience,
                issuer=self.issuer
            )
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti and await self.is_token_revoked(jti):
                raise jwt.InvalidTokenError("Token has been revoked")
            
            # Verify token type
            if token_type and payload.get("type") != token_type.value:
                raise jwt.InvalidTokenError(f"Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        try:
            # Decode without verification to get JTI
            payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            
            jti = payload.get("jti")
            if not jti:
                return False
            
            # Add to revoked set
            self.revoked_tokens.add(jti)
            
            # Store in Redis
            if self.redis:
                exp = payload.get("exp", 0)
                ttl = max(0, exp - time.time()) if exp else 3600
                await self.redis.setex(
                    f"revoked_token:{jti}",
                    int(ttl),
                    "1"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    async def is_token_revoked(self, jti: str) -> bool:
        """Check if a token is revoked"""
        if jti in self.revoked_tokens:
            return True
        
        if self.redis:
            try:
                result = await self.redis.get(f"revoked_token:{jti}")
                return result == "1"
            except Exception:
                pass
        
        return False
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    async def check_brute_force(self, identifier: str) -> bool:
        """Check if an account is locked due to brute force attempts"""
        if self.redis:
            try:
                attempts = await self.redis.get(f"failed_attempts:{identifier}")
                if attempts and int(attempts) >= self.max_failed_attempts:
                    return True
            except Exception:
                pass
        else:
            # In-memory fallback
            if identifier in self.failed_attempts:
                recent_attempts = [
                    dt for dt in self.failed_attempts[identifier]
                    if (datetime.now(timezone.utc) - dt).total_seconds() < self.lockout_duration_minutes * 60
                ]
                if len(recent_attempts) >= self.max_failed_attempts:
                    return True
        
        return False
    
    async def record_failed_attempt(self, identifier: str):
        """Record a failed login attempt"""
        if self.redis:
            try:
                key = f"failed_attempts:{identifier}"
                await self.redis.incr(key)
                await self.redis.expire(key, self.lockout_duration_minutes * 60)
            except Exception:
                pass
        else:
            # In-memory fallback
            if identifier not in self.failed_attempts:
                self.failed_attempts[identifier] = []
            self.failed_attempts[identifier].append(datetime.now(timezone.utc))
    
    async def clear_failed_attempts(self, identifier: str):
        """Clear failed attempts after successful login"""
        if self.redis:
            try:
                await self.redis.delete(f"failed_attempts:{identifier}")
            except Exception:
                pass
        else:
            if identifier in self.failed_attempts:
                del self.failed_attempts[identifier]
    
    async def rotate_keys(self) -> JWTKey:
        """Rotate JWT signing keys"""
        # Generate new key
        new_kid = f"key-{uuid.uuid4().hex[:8]}-{int(time.time())}"
        new_secret = secrets.token_urlsafe(64)
        
        new_key = JWTKey(
            kid=new_kid,
            secret=new_secret,
            algorithm="HS256",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=self.key_rotation_days),
            is_active=True,
            is_valid=True,
            rotation_version=len(self.keys) + 1
        )
        
        # Deactivate old key
        if self.active_key:
            self.active_key.is_active = False
            # Schedule invalidation
            asyncio.create_task(
                self._schedule_key_invalidation(
                    self.active_key.kid,
                    self.key_overlap_hours
                )
            )
        
        # Set new active key
        self.keys[new_kid] = new_key
        self.active_key = new_key
        
        # Save to Redis
        if self.redis:
            await self._save_key_to_redis(new_key)
        
        logger.info(f"JWT key rotated: {new_kid}")
        return new_key
    
    async def _schedule_key_invalidation(self, kid: str, hours: int):
        """Schedule key invalidation after overlap period"""
        await asyncio.sleep(hours * 3600)
        if kid in self.keys:
            self.keys[kid].is_valid = False
            logger.info(f"JWT key {kid} invalidated")
    
    async def _key_rotation_scheduler(self):
        """Background task for key rotation"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                if self.active_key:
                    time_until_expiry = (
                        self.active_key.expires_at - datetime.now(timezone.utc)
                    ).total_seconds()
                    
                    # Rotate if less than 25% lifetime remaining
                    if time_until_expiry < (self.key_rotation_days * 86400 * 0.25):
                        await self.rotate_keys()
                        
            except Exception as e:
                logger.error(f"Key rotation scheduler error: {e}")
    
    async def _load_keys_from_redis(self):
        """Load JWT keys from Redis"""
        if not self.redis:
            return
        
        try:
            keys = await self.redis.keys("jwt:key:*")
            for key_name in keys:
                key_data = await self.redis.get(key_name)
                if key_data:
                    data = json.loads(key_data)
                    kid = data["kid"]
                    
                    self.keys[kid] = JWTKey(
                        kid=kid,
                        secret=data["secret"],
                        algorithm=data["algorithm"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        expires_at=datetime.fromisoformat(data["expires_at"]),
                        is_active=data["is_active"],
                        is_valid=data["is_valid"],
                        rotation_version=data["rotation_version"]
                    )
                    
                    if data["is_active"]:
                        self.active_key = self.keys[kid]
                        
        except Exception as e:
            logger.error(f"Failed to load keys from Redis: {e}")
    
    async def _save_key_to_redis(self, key: JWTKey):
        """Save JWT key to Redis"""
        if not self.redis:
            return
        
        try:
            key_data = {
                "kid": key.kid,
                "secret": key.secret,
                "algorithm": key.algorithm,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat(),
                "is_active": key.is_active,
                "is_valid": key.is_valid,
                "rotation_version": key.rotation_version
            }
            
            ttl = int((key.expires_at - datetime.now(timezone.utc)).total_seconds())
            await self.redis.setex(
                f"jwt:key:{key.kid}",
                max(ttl, 86400),
                json.dumps(key_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to save key to Redis: {e}")


# Global instance
_auth_manager: Optional[JWTAuthManager] = None


async def get_auth_manager() -> JWTAuthManager:
    """Get or create auth manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = JWTAuthManager()
        await _auth_manager.initialize()
    return _auth_manager


# FastAPI Dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current user from token"""
    auth_manager = await get_auth_manager()
    
    try:
        payload = await auth_manager.verify_token(
            credentials.credentials,
            token_type=TokenType.ACCESS
        )
        
        # Create user object from token
        roles = payload.get("roles", [])
        permissions = []
        for role in roles:
            permissions.extend(get_role_permissions(role))
        
        user = User(
            user_id=payload.get("sub", ""),
            username=payload.get("username", ""),
            email=payload.get("email"),
            roles=roles,
            permissions=list(set(permissions)),  # Remove duplicates
            is_active=True  # If token is valid, user is active
        )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user"""
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get optional user (for endpoints that work with or without auth)"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_roles(*allowed_roles: str):
    """Dependency to require specific roles"""
    async def role_checker(current_user: User = Depends(get_current_active_user)):
        if not any(role in current_user.roles for role in allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker


# Additional functions required by auth router
JWT_EXPIRY_MINUTES = 15  # Default token expiry


def get_role_permissions(role: str) -> List[str]:
    """
    Get permissions for a given role
    """
    role_permissions = {
        "admin": [
            "admin:*",
            "trading:*",
            "analytics:*",
            "user:*",
            "system:*"
        ],
        "trader": [
            "trading:read",
            "trading:write",
            "trading:execute",
            "analytics:read",
            "portfolio:read",
            "portfolio:write"
        ],
        "analyst": [
            "trading:read",
            "analytics:*",
            "portfolio:read",
            "reports:*"
        ],
        "viewer": [
            "trading:read",
            "analytics:read",
            "portfolio:read"
        ],
        "service": [
            "system:read",
            "system:write",
            "metrics:write"
        ]
    }
    
    return role_permissions.get(role, [])


async def authenticate_user(username: str, password: str, client_ip: str = "unknown") -> Optional[User]:
    """
    Authenticate a user with username and password
    Includes brute force protection
    """
    auth_manager = await get_auth_manager()
    
    # Check brute force protection
    if await auth_manager.check_brute_force(username):
        logger.warning(f"Account {username} locked due to too many failed attempts from {client_ip}")
        return None
    
    # Import here to avoid circular dependency
    from trading_common.user_repository import UserRepository
    
    # Get user from database
    user_repo = UserRepository()
    user_data = await user_repo.get_user_by_username(username)
    
    if not user_data:
        await auth_manager.record_failed_attempt(username)
        return None
    
    # Verify password
    if not auth_manager.verify_password(password, user_data.hashed_password):
        await auth_manager.record_failed_attempt(username)
        return None
    
    # Clear failed attempts on successful auth
    await auth_manager.clear_failed_attempts(username)
    
    # Update last login
    await user_repo.update_last_login(user_data.id)
    
    # Create User object
    user = User(
        user_id=str(user_data.id),
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        disabled=not user_data.is_active,
        is_active=user_data.is_active,
        roles=[user_data.role],
        permissions=get_role_permissions(user_data.role),  # Get permissions based on role
        created_at=user_data.created_at if hasattr(user_data, 'created_at') else datetime.now(timezone.utc),
        last_login=datetime.now(timezone.utc)
    )
    
    logger.info(f"User {username} authenticated successfully from {client_ip}")
    return user


def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token for a user (synchronous wrapper)
    """
    import asyncio
    
    async def _create():
        auth_manager = await get_auth_manager()
        
        data = {
            "sub": user.user_id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "type": "access"
        }
        
        return await auth_manager.create_access_token(data, expires_delta)
    
    # Run async function in sync context
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If loop is already running, create task
        task = asyncio.create_task(_create())
        return asyncio.run_until_complete(task)
    else:
        return asyncio.run(_create())


def get_auth_health() -> dict:
    """
    Get authentication system health status
    """
    try:
        import asyncio
        
        async def _get_health():
            auth_manager = await get_auth_manager()
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "jwt_keys": {
                        "active": auth_manager.active_key is not None,
                        "total_keys": len(auth_manager.keys),
                        "rotation_days": auth_manager.key_rotation_days
                    },
                    "redis": {
                        "connected": auth_manager.redis is not None
                    },
                    "security": {
                        "max_failed_attempts": auth_manager.max_failed_attempts,
                        "lockout_duration_minutes": auth_manager.lockout_duration_minutes,
                        "mfa_required_for_sensitive": auth_manager.require_mfa_for_sensitive
                    },
                    "tokens": {
                        "access_token_expire_minutes": auth_manager.access_token_expire_minutes,
                        "refresh_token_expire_days": auth_manager.refresh_token_expire_days,
                        "revoked_count": len(auth_manager.revoked_tokens)
                    }
                }
            }
            
            # Check Redis connectivity
            if auth_manager.redis:
                try:
                    await auth_manager.redis.ping()
                    health_status["components"]["redis"]["status"] = "connected"
                except Exception as e:
                    health_status["components"]["redis"]["status"] = "disconnected"
                    health_status["components"]["redis"]["error"] = str(e)
                    health_status["status"] = "degraded"
            
            return health_status
        
        # Run async function
        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = asyncio.create_task(_get_health())
            return asyncio.run_until_complete(task)
        else:
            return asyncio.run(_get_health())
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
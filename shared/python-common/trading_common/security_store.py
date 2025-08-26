#!/usr/bin/env python3
"""
Persistent Security Store for JWT tokens, sessions, and audit logs.
Addresses critical security persistence issues identified in production readiness review.
"""

import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .logging import get_logger
from .config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class SecurityEventType(str, Enum):
    """Types of security events for audit logging."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    TOKEN_CREATED = "token_created"
    TOKEN_REVOKED = "token_revoked"
    TOKEN_REFRESH = "token_refresh"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    PASSWORD_CHANGE = "password_change"
    ACCOUNT_LOCKED = "account_locked"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    success: bool
    details: Dict[str, Any]
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL
    
    def __post_init__(self):
        """Generate event ID if not provided."""
        if not self.event_id:
            import uuid
            self.event_id = str(uuid.uuid4())


@dataclass
class UserSession:
    """Persistent user session data."""
    session_id: str
    user_id: str
    username: str
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str]
    user_agent: Optional[str]
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSession':
        """Create from dictionary."""
        # Convert datetime strings back to datetime objects
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('last_activity'), str):
            data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        return cls(**data)


@dataclass
class RefreshToken:
    """Persistent refresh token data."""
    token_hash: str  # Hashed token value for security
    user_id: str
    username: str
    created_at: datetime
    expires_at: datetime
    last_used: Optional[datetime] = None
    is_revoked: bool = False
    device_info: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefreshToken':
        """Create from dictionary."""
        # Convert datetime strings back to datetime objects
        for field in ['created_at', 'expires_at', 'last_used']:
            if isinstance(data.get(field), str):
                data[field] = datetime.fromisoformat(data[field])
            elif data.get(field) is None and field == 'last_used':
                data[field] = None
        return cls(**data)


class PersistentSecurityStore:
    """Persistent store for security data using Redis backend."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize security store."""
        self.redis_url = redis_url or settings.database.redis_url or "redis://localhost:6379"
        self.redis: Optional[aioredis.Redis] = None
        self.connected = False
        
        # Key prefixes for different data types
        self.REVOKED_TOKEN_PREFIX = "revoked_token:"
        self.REFRESH_TOKEN_PREFIX = "refresh_token:"
        self.SESSION_PREFIX = "session:"
        self.AUDIT_LOG_PREFIX = "audit_log:"
        self.LOGIN_ATTEMPTS_PREFIX = "login_attempts:"
        self.RATE_LIMIT_PREFIX = "rate_limit:"
        
        # Expiration times
        self.REVOKED_TOKEN_TTL = 86400 * 30  # 30 days
        self.REFRESH_TOKEN_TTL = 86400 * 30  # 30 days  
        self.SESSION_TTL = 86400 * 7  # 7 days
        self.AUDIT_LOG_TTL = 86400 * 90  # 90 days retention
        self.LOGIN_ATTEMPTS_TTL = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.error("Redis not available - security store requires redis package")
            raise RuntimeError("Redis required for persistent security store")
        
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await asyncio.wait_for(self.redis.ping(), timeout=5.0)
            self.connected = True
            logger.info("Persistent security store connected to Redis")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis for security store: {e}")
            self.connected = False
            raise RuntimeError(f"Security store connection failed: {e}")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self.connected = False
    
    # Token Revocation Management
    async def revoke_token(self, jti: str, expires_at: datetime) -> bool:
        """Revoke a JWT token by JTI."""
        if not self.connected:
            raise RuntimeError("Security store not connected")
        
        key = f"{self.REVOKED_TOKEN_PREFIX}{jti}"
        
        # Store revocation with expiration matching token expiry
        ttl = int((expires_at - datetime.utcnow()).total_seconds())
        if ttl > 0:
            await self.redis.setex(key, ttl, "revoked")
            logger.info(f"Token {jti} revoked and will expire at {expires_at}")
            return True
        
        return False
    
    async def is_token_revoked(self, jti: str) -> bool:
        """Check if a token is revoked."""
        if not self.connected:
            return False  # Fail open in disconnected state for availability
        
        try:
            key = f"{self.REVOKED_TOKEN_PREFIX}{jti}"
            result = await self.redis.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Error checking token revocation: {e}")
            return False  # Fail open
    
    async def cleanup_expired_revoked_tokens(self):
        """Cleanup expired revoked tokens (handled by Redis TTL)."""
        # Redis handles TTL automatically, but we can add monitoring
        if not self.connected:
            return
        
        try:
            # Count current revoked tokens for monitoring
            pattern = f"{self.REVOKED_TOKEN_PREFIX}*"
            keys = await self.redis.keys(pattern)
            logger.info(f"Currently tracking {len(keys)} revoked tokens")
        except Exception as e:
            logger.error(f"Error during revoked token cleanup: {e}")
    
    # Refresh Token Management
    def _hash_refresh_token(self, token: str) -> str:
        """Hash refresh token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    async def store_refresh_token(self, token: str, refresh_token_data: RefreshToken) -> bool:
        """Store refresh token data."""
        if not self.connected:
            raise RuntimeError("Security store not connected")
        
        token_hash = self._hash_refresh_token(token)
        key = f"{self.REFRESH_TOKEN_PREFIX}{token_hash}"
        
        # Store with TTL
        data = refresh_token_data.to_dict()
        await self.redis.setex(
            key, 
            self.REFRESH_TOKEN_TTL,
            json.dumps(data, default=str)
        )
        
        return True
    
    async def get_refresh_token(self, token: str) -> Optional[RefreshToken]:
        """Get refresh token data."""
        if not self.connected:
            return None
        
        try:
            token_hash = self._hash_refresh_token(token)
            key = f"{self.REFRESH_TOKEN_PREFIX}{token_hash}"
            
            data = await self.redis.get(key)
            if data:
                token_dict = json.loads(data)
                return RefreshToken.from_dict(token_dict)
        except Exception as e:
            logger.error(f"Error retrieving refresh token: {e}")
        
        return None
    
    async def revoke_refresh_token(self, token: str) -> bool:
        """Revoke a refresh token."""
        if not self.connected:
            return False
        
        try:
            token_hash = self._hash_refresh_token(token)
            key = f"{self.REFRESH_TOKEN_PREFIX}{token_hash}"
            
            # Update revocation status
            data = await self.redis.get(key)
            if data:
                token_dict = json.loads(data)
                token_dict['is_revoked'] = True
                await self.redis.setex(
                    key,
                    self.REFRESH_TOKEN_TTL, 
                    json.dumps(token_dict, default=str)
                )
                return True
        except Exception as e:
            logger.error(f"Error revoking refresh token: {e}")
        
        return False
    
    # Session Management  
    async def create_session(self, session: UserSession) -> bool:
        """Create a new user session."""
        if not self.connected:
            raise RuntimeError("Security store not connected")
        
        key = f"{self.SESSION_PREFIX}{session.session_id}"
        data = session.to_dict()
        
        await self.redis.setex(
            key,
            self.SESSION_TTL,
            json.dumps(data, default=str)
        )
        
        return True
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session data."""
        if not self.connected:
            return None
        
        try:
            key = f"{self.SESSION_PREFIX}{session_id}"
            data = await self.redis.get(key)
            if data:
                session_dict = json.loads(data)
                return UserSession.from_dict(session_dict)
        except Exception as e:
            logger.error(f"Error retrieving session: {e}")
        
        return None
    
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity."""
        if not self.connected:
            return False
        
        try:
            session = await self.get_session(session_id)
            if session and session.is_active:
                session.last_activity = datetime.utcnow()
                await self.create_session(session)  # Update with new TTL
                return True
        except Exception as e:
            logger.error(f"Error updating session activity: {e}")
        
        return False
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        if not self.connected:
            return False
        
        try:
            key = f"{self.SESSION_PREFIX}{session_id}"
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error invalidating session: {e}")
        
        return False
    
    # Audit Logging
    async def log_security_event(self, event: SecurityEvent) -> bool:
        """Log security event for auditing."""
        if not self.connected:
            logger.warning("Security store disconnected - audit event not persisted")
            return False
        
        try:
            # Use time-based key for chronological ordering
            timestamp_key = event.timestamp.strftime("%Y%m%d_%H%M%S")
            key = f"{self.AUDIT_LOG_PREFIX}{timestamp_key}_{event.event_id}"
            
            data = asdict(event)
            await self.redis.setex(
                key,
                self.AUDIT_LOG_TTL,
                json.dumps(data, default=str)
            )
            
            # Also log to regular logger for immediate visibility
            log_msg = f"SECURITY_EVENT: {event.event_type.value} | User: {event.username} | IP: {event.ip_address} | Success: {event.success}"
            if event.severity == "CRITICAL":
                logger.critical(log_msg)
            elif event.severity == "ERROR":
                logger.error(log_msg)
            elif event.severity == "WARNING":
                logger.warning(log_msg)
            else:
                logger.info(log_msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
            return False
    
    async def get_security_events(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[SecurityEventType]] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Retrieve security events with filtering."""
        if not self.connected:
            return []
        
        try:
            pattern = f"{self.AUDIT_LOG_PREFIX}*"
            keys = await self.redis.keys(pattern)
            
            events = []
            for key in sorted(keys, reverse=True)[:limit * 2]:  # Get more to allow filtering
                try:
                    data = await self.redis.get(key)
                    if data:
                        event_dict = json.loads(data)
                        event = SecurityEvent(**event_dict)
                        
                        # Apply filters
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue
                        if event_types and event.event_type not in event_types:
                            continue
                        if user_id and event.user_id != user_id:
                            continue
                        
                        events.append(event)
                        
                        if len(events) >= limit:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error parsing security event from key {key}: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving security events: {e}")
            return []
    
    # Login Attempts Management  
    async def record_login_attempt(self, username: str, success: bool, ip_address: str) -> int:
        """Record login attempt and return current failure count."""
        if not self.connected:
            return 0
        
        try:
            key = f"{self.LOGIN_ATTEMPTS_PREFIX}{username}"
            
            if success:
                # Clear attempts on successful login
                await self.redis.delete(key)
                return 0
            else:
                # Increment failure count
                count = await self.redis.incr(key)
                if count == 1:
                    # Set expiration on first failure
                    await self.redis.expire(key, self.LOGIN_ATTEMPTS_TTL)
                
                return count
                
        except Exception as e:
            logger.error(f"Error recording login attempt: {e}")
            return 0
    
    async def get_login_attempt_count(self, username: str) -> int:
        """Get current login attempt count."""
        if not self.connected:
            return 0
        
        try:
            key = f"{self.LOGIN_ATTEMPTS_PREFIX}{username}"
            count = await self.redis.get(key)
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"Error getting login attempt count: {e}")
            return 0
    
    async def is_account_locked(self, username: str, max_attempts: int = 5) -> bool:
        """Check if account is locked due to failed attempts."""
        count = await self.get_login_attempt_count(username)
        return count >= max_attempts
    
    # Health and Maintenance
    async def get_store_health(self) -> Dict[str, Any]:
        """Get security store health information."""
        if not self.connected:
            return {"status": "disconnected", "error": "Not connected to Redis"}
        
        try:
            # Get Redis info
            info = await self.redis.info()
            
            # Count different types of keys
            stats = {}
            for prefix in [self.REVOKED_TOKEN_PREFIX, self.REFRESH_TOKEN_PREFIX, 
                          self.SESSION_PREFIX, self.AUDIT_LOG_PREFIX, self.LOGIN_ATTEMPTS_PREFIX]:
                pattern = f"{prefix}*"
                keys = await self.redis.keys(pattern)
                stats[prefix.rstrip(':')] = len(keys)
            
            return {
                "status": "healthy",
                "connected": True,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "key_counts": stats,
                "uptime": info.get("uptime_in_seconds")
            }
            
        except Exception as e:
            logger.error(f"Error getting security store health: {e}")
            return {"status": "error", "error": str(e)}


# Global security store instance
_security_store: Optional[PersistentSecurityStore] = None


async def get_security_store() -> PersistentSecurityStore:
    """Get or create global security store instance."""
    global _security_store
    if _security_store is None:
        _security_store = PersistentSecurityStore()
        await _security_store.initialize()
    return _security_store


async def log_security_event(
    event_type: SecurityEventType,
    success: bool,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    severity: str = "INFO"
) -> bool:
    """Convenience function to log security events."""
    try:
        store = await get_security_store()
        event = SecurityEvent(
            event_id="",  # Will be auto-generated
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {},
            severity=severity
        )
        return await store.log_security_event(event)
    except Exception as e:
        logger.error(f"Failed to log security event: {e}")
        return False
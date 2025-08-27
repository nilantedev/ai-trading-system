#!/usr/bin/env python3
"""
Production-ready JWT authentication with token revocation and security features.
"""

import jwt
import redis
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class JWTConfig:
    """JWT configuration settings."""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    issuer: str = "ai-trading-system"
    redis_client: Optional[redis.Redis] = None
    max_refresh_count: int = 5


class JWTAuth:
    """
    JWT authentication handler with security features.
    - Token revocation via Redis blacklist
    - Refresh token rotation
    - Rate limiting for login attempts
    - Secure token generation
    """
    
    def __init__(self, config: JWTConfig):
        self.config = config
        self.redis_client = config.redis_client
        
        # Key prefixes for Redis
        self.BLACKLIST_PREFIX = "jwt:blacklist:"
        self.REFRESH_PREFIX = "jwt:refresh:"
        self.RATE_LIMIT_PREFIX = "auth:ratelimit:"
        self.SESSION_PREFIX = "jwt:session:"
    
    def generate_jti(self) -> str:
        """Generate unique JWT ID."""
        return secrets.token_urlsafe(32)
    
    def create_access_token(
        self,
        user_id: str,
        roles: List[str] = None,
        permissions: List[str] = None,
        additional_claims: Dict[str, Any] = None
    ) -> str:
        """
        Create JWT access token.
        
        Args:
            user_id: User identifier
            roles: User roles
            permissions: User permissions
            additional_claims: Additional JWT claims
            
        Returns:
            Encoded JWT token
        """
        now = datetime.now(timezone.utc)
        jti = self.generate_jti()
        
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(minutes=self.config.access_token_expire_minutes),
            "jti": jti,
            "type": "access",
            "iss": self.config.issuer,
            "roles": roles or [],
            "permissions": permissions or []
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        # Store session info in Redis
        if self.redis_client:
            session_key = f"{self.SESSION_PREFIX}{user_id}:{jti}"
            self.redis_client.setex(
                session_key,
                self.config.access_token_expire_minutes * 60,
                "active"
            )
        
        return jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
    
    def create_refresh_token(
        self,
        user_id: str,
        device_id: Optional[str] = None
    ) -> str:
        """
        Create JWT refresh token with rotation tracking.
        
        Args:
            user_id: User identifier
            device_id: Device identifier for multi-device support
            
        Returns:
            Encoded refresh token
        """
        now = datetime.now(timezone.utc)
        jti = self.generate_jti()
        
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(days=self.config.refresh_token_expire_days),
            "jti": jti,
            "type": "refresh",
            "iss": self.config.issuer,
            "device_id": device_id or "default",
            "refresh_count": 0
        }
        
        # Track refresh token in Redis
        if self.redis_client:
            refresh_key = f"{self.REFRESH_PREFIX}{user_id}:{jti}"
            self.redis_client.setex(
                refresh_key,
                self.config.refresh_token_expire_days * 86400,
                "0"  # Refresh count
            )
        
        return jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            token_type: Expected token type (access/refresh)
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                issuer=self.config.issuer
            )
            
            # Check token type
            if payload.get("type") != token_type:
                logger.warning(f"Invalid token type: expected {token_type}, got {payload.get('type')}")
                return None
            
            # Check if token is blacklisted
            if self.is_token_revoked(payload.get("jti")):
                logger.warning(f"Token {payload.get('jti')} is revoked")
                return None
            
            # For refresh tokens, check refresh count
            if token_type == "refresh" and self.redis_client:
                refresh_key = f"{self.REFRESH_PREFIX}{payload['sub']}:{payload['jti']}"
                refresh_count = self.redis_client.get(refresh_key)
                if refresh_count and int(refresh_count) >= self.config.max_refresh_count:
                    logger.warning(f"Refresh token exceeded max refresh count")
                    return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.debug("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[tuple[str, str]]:
        """
        Generate new access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Tuple of (new_access_token, new_refresh_token) or None
        """
        payload = self.verify_token(refresh_token, token_type="refresh")
        if not payload:
            return None
        
        user_id = payload["sub"]
        device_id = payload.get("device_id")
        old_jti = payload["jti"]
        
        # Increment refresh count
        if self.redis_client:
            refresh_key = f"{self.REFRESH_PREFIX}{user_id}:{old_jti}"
            current_count = self.redis_client.get(refresh_key)
            if current_count:
                new_count = int(current_count) + 1
                if new_count >= self.config.max_refresh_count:
                    # Revoke the refresh token
                    self.revoke_token(old_jti)
                    return None
                self.redis_client.setex(
                    refresh_key,
                    self.config.refresh_token_expire_days * 86400,
                    str(new_count)
                )
        
        # Create new tokens
        new_access_token = self.create_access_token(
            user_id=user_id,
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", [])
        )
        
        # Rotate refresh token
        new_refresh_token = self.create_refresh_token(user_id, device_id)
        
        # Revoke old refresh token
        self.revoke_token(old_jti)
        
        return new_access_token, new_refresh_token
    
    def revoke_token(self, jti: str, expiry: int = 86400) -> bool:
        """
        Revoke a token by adding to blacklist.
        
        Args:
            jti: JWT ID to revoke
            expiry: Blacklist expiry in seconds
            
        Returns:
            True if revoked successfully
        """
        if not self.redis_client:
            logger.warning("Redis not configured, cannot revoke token")
            return False
        
        blacklist_key = f"{self.BLACKLIST_PREFIX}{jti}"
        self.redis_client.setex(blacklist_key, expiry, "revoked")
        logger.info(f"Token {jti} revoked")
        return True
    
    def is_token_revoked(self, jti: str) -> bool:
        """
        Check if token is revoked.
        
        Args:
            jti: JWT ID to check
            
        Returns:
            True if token is revoked
        """
        if not self.redis_client:
            return False
        
        blacklist_key = f"{self.BLACKLIST_PREFIX}{jti}"
        return self.redis_client.exists(blacklist_key) > 0
    
    def revoke_all_user_tokens(self, user_id: str) -> int:
        """
        Revoke all tokens for a user (logout from all devices).
        
        Args:
            user_id: User to logout
            
        Returns:
            Number of tokens revoked
        """
        if not self.redis_client:
            return 0
        
        count = 0
        # Find all session keys for user
        pattern = f"{self.SESSION_PREFIX}{user_id}:*"
        for key in self.redis_client.scan_iter(match=pattern):
            # Extract JTI from key
            jti = key.decode().split(":")[-1]
            if self.revoke_token(jti):
                count += 1
            self.redis_client.delete(key)
        
        # Revoke all refresh tokens
        pattern = f"{self.REFRESH_PREFIX}{user_id}:*"
        for key in self.redis_client.scan_iter(match=pattern):
            jti = key.decode().split(":")[-1]
            if self.revoke_token(jti):
                count += 1
            self.redis_client.delete(key)
        
        logger.info(f"Revoked {count} tokens for user {user_id}")
        return count
    
    def check_rate_limit(self, identifier: str, max_attempts: int = 5, window: int = 300) -> bool:
        """
        Check if identifier has exceeded rate limit.
        
        Args:
            identifier: IP address or user ID
            max_attempts: Maximum attempts allowed
            window: Time window in seconds
            
        Returns:
            True if rate limit NOT exceeded
        """
        if not self.redis_client:
            return True
        
        rate_key = f"{self.RATE_LIMIT_PREFIX}{identifier}"
        
        # Increment attempt counter
        attempts = self.redis_client.incr(rate_key)
        
        # Set expiry on first attempt
        if attempts == 1:
            self.redis_client.expire(rate_key, window)
        
        if attempts > max_attempts:
            ttl = self.redis_client.ttl(rate_key)
            logger.warning(f"Rate limit exceeded for {identifier}. Retry in {ttl} seconds")
            return False
        
        return True
    
    def clear_rate_limit(self, identifier: str):
        """Clear rate limit for identifier (after successful login)."""
        if self.redis_client:
            rate_key = f"{self.RATE_LIMIT_PREFIX}{identifier}"
            self.redis_client.delete(rate_key)
    
    def get_active_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of active session information
        """
        if not self.redis_client:
            return []
        
        sessions = []
        pattern = f"{self.SESSION_PREFIX}{user_id}:*"
        
        for key in self.redis_client.scan_iter(match=pattern):
            jti = key.decode().split(":")[-1]
            ttl = self.redis_client.ttl(key)
            sessions.append({
                "jti": jti,
                "remaining_seconds": ttl
            })
        
        return sessions


# FastAPI dependency for JWT verification
def get_jwt_auth(redis_url: Optional[str] = None) -> JWTAuth:
    """
    Get JWT auth instance for dependency injection.
    
    Args:
        redis_url: Redis connection URL
        
    Returns:
        JWTAuth instance
    """
    import os
    
    redis_client = None
    if redis_url or os.getenv("REDIS_URL"):
        redis_client = redis.from_url(redis_url or os.getenv("REDIS_URL"))
    
    config = JWTConfig(
        secret_key=os.getenv("JWT_SECRET", ""),
        redis_client=redis_client
    )
    
    return JWTAuth(config)
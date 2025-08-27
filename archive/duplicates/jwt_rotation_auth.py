#!/usr/bin/env python3
"""
JWT Authentication with Key Rotation Support
Implements kid (Key ID) header support for seamless key rotation
"""

import os
import jwt
import json
import time
import uuid
import hashlib
import hmac
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aioredis
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)


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
    algorithm: str          # Signing algorithm (HS256, RS256, etc.)
    created_at: datetime    # Creation timestamp
    expires_at: datetime    # Expiration timestamp
    is_active: bool        # Whether key can sign new tokens
    is_valid: bool         # Whether key can verify tokens
    rotation_version: int   # Rotation generation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kid": self.kid,
            "algorithm": self.algorithm,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "is_active": self.is_active,
            "is_valid": self.is_valid,
            "rotation_version": self.rotation_version
        }


class JWTRotationManager:
    """Manages JWT key rotation with zero-downtime"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.redis: Optional[aioredis.Redis] = None
        
        # Key rotation settings
        self.key_rotation_days = int(os.getenv("JWT_KEY_ROTATION_DAYS", "30"))
        self.key_overlap_hours = int(os.getenv("JWT_KEY_OVERLAP_HOURS", "24"))
        self.max_keys = int(os.getenv("JWT_MAX_KEYS", "5"))
        
        # Token settings
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        self.api_key_expire_days = int(os.getenv("API_KEY_EXPIRE_DAYS", "365"))
        
        # Security settings
        self.issuer = os.getenv("JWT_ISSUER", "ai-trading-system")
        self.audience = os.getenv("JWT_AUDIENCE", "trading-api")
        self.require_kid = os.getenv("JWT_REQUIRE_KID", "true").lower() == "true"
        
        # Key storage
        self.keys: Dict[str, JWTKey] = {}
        self.active_key: Optional[JWTKey] = None
        
        # Revocation tracking
        self.revoked_tokens: set = set()
        self.revocation_cache_ttl = 3600  # 1 hour cache
        
        # Metrics
        self.metrics = {
            "tokens_issued": 0,
            "tokens_verified": 0,
            "tokens_revoked": 0,
            "key_rotations": 0,
            "verification_failures": 0
        }
    
    async def initialize(self):
        """Initialize the JWT rotation manager"""
        try:
            # Connect to Redis for distributed key storage
            self.redis = aioredis.from_url(
                self.redis_url,
                password=self.redis_password,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Load existing keys from Redis
            await self._load_keys_from_redis()
            
            # Generate initial key if none exist
            if not self.keys:
                await self.rotate_keys()
            else:
                # Find active key
                for key in self.keys.values():
                    if key.is_active:
                        self.active_key = key
                        break
            
            # Start key rotation scheduler
            asyncio.create_task(self._key_rotation_scheduler())
            
            # Start revocation cache cleanup
            asyncio.create_task(self._revocation_cleanup_scheduler())
            
            logger.info(f"JWT Rotation Manager initialized with {len(self.keys)} keys")
            
        except Exception as e:
            logger.error(f"Failed to initialize JWT rotation manager: {e}")
            # Generate local key as fallback
            self._generate_fallback_key()
    
    def _generate_fallback_key(self):
        """Generate a fallback key when Redis is unavailable"""
        kid = f"fallback-{uuid.uuid4().hex[:8]}"
        secret = os.getenv("JWT_SECRET", os.urandom(64).hex())
        
        self.active_key = JWTKey(
            kid=kid,
            secret=secret,
            algorithm="HS256",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            is_active=True,
            is_valid=True,
            rotation_version=0
        )
        self.keys[kid] = self.active_key
        
        logger.warning("Using fallback JWT key (Redis unavailable)")
    
    async def rotate_keys(self) -> JWTKey:
        """Rotate JWT signing keys with overlap period"""
        try:
            # Generate new key
            new_kid = f"key-{uuid.uuid4().hex[:8]}-{int(time.time())}"
            new_secret = os.urandom(64).hex()
            
            current_time = datetime.now(timezone.utc)
            new_key = JWTKey(
                kid=new_kid,
                secret=new_secret,
                algorithm="HS256",
                created_at=current_time,
                expires_at=current_time + timedelta(days=self.key_rotation_days),
                is_active=True,
                is_valid=True,
                rotation_version=len(self.keys) + 1
            )
            
            # Deactivate old active key (but keep it valid for overlap period)
            if self.active_key:
                self.active_key.is_active = False
                # Schedule old key invalidation after overlap period
                asyncio.create_task(
                    self._schedule_key_invalidation(
                        self.active_key.kid,
                        self.key_overlap_hours
                    )
                )
            
            # Store new key
            self.keys[new_kid] = new_key
            self.active_key = new_key
            
            # Persist to Redis
            if self.redis:
                await self._save_key_to_redis(new_key)
            
            # Clean up old keys
            await self._cleanup_old_keys()
            
            # Update metrics
            self.metrics["key_rotations"] += 1
            
            # Log rotation event
            logger.info(
                f"JWT key rotated: {new_kid} (version {new_key.rotation_version})"
            )
            
            # Send notification about key rotation
            await self._notify_key_rotation(new_key)
            
            return new_key
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise
    
    async def _schedule_key_invalidation(self, kid: str, hours: int):
        """Schedule a key to be invalidated after specified hours"""
        await asyncio.sleep(hours * 3600)
        
        if kid in self.keys:
            self.keys[kid].is_valid = False
            logger.info(f"JWT key {kid} invalidated after overlap period")
            
            if self.redis:
                await self._save_key_to_redis(self.keys[kid])
    
    async def _cleanup_old_keys(self):
        """Remove expired and invalid keys beyond max limit"""
        if len(self.keys) <= self.max_keys:
            return
        
        # Sort keys by creation date
        sorted_keys = sorted(
            self.keys.items(),
            key=lambda x: x[1].created_at
        )
        
        # Remove oldest invalid keys
        for kid, key in sorted_keys:
            if not key.is_valid and len(self.keys) > self.max_keys:
                del self.keys[kid]
                
                if self.redis:
                    await self.redis.delete(f"jwt:key:{kid}")
                
                logger.info(f"Removed old JWT key: {kid}")
    
    async def create_token(
        self,
        subject: str,
        token_type: TokenType = TokenType.ACCESS,
        additional_claims: Optional[Dict[str, Any]] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new JWT token with current active key"""
        if not self.active_key:
            raise RuntimeError("No active JWT key available")
        
        # Determine expiration
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        elif token_type == TokenType.ACCESS:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.access_token_expire_minutes
            )
        elif token_type == TokenType.REFRESH:
            expire = datetime.now(timezone.utc) + timedelta(
                days=self.refresh_token_expire_days
            )
        elif token_type == TokenType.API_KEY:
            expire = datetime.now(timezone.utc) + timedelta(
                days=self.api_key_expire_days
            )
        else:
            expire = datetime.now(timezone.utc) + timedelta(hours=1)
        
        # Build claims
        claims = {
            "sub": subject,
            "iss": self.issuer,
            "aud": self.audience,
            "iat": datetime.now(timezone.utc),
            "exp": expire,
            "jti": str(uuid.uuid4()),  # Unique token ID for revocation
            "type": token_type.value,
            "kid": self.active_key.kid  # Include key ID in claims
        }
        
        if additional_claims:
            claims.update(additional_claims)
        
        # Sign token with kid in header
        token = jwt.encode(
            claims,
            self.active_key.secret,
            algorithm=self.active_key.algorithm,
            headers={"kid": self.active_key.kid}
        )
        
        # Update metrics
        self.metrics["tokens_issued"] += 1
        
        return token
    
    async def verify_token(
        self,
        token: str,
        token_type: Optional[TokenType] = None,
        verify_exp: bool = True
    ) -> Dict[str, Any]:
        """Verify a JWT token using appropriate key"""
        try:
            # Decode header to get kid
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            
            if self.require_kid and not kid:
                raise jwt.InvalidTokenError("Token missing kid header")
            
            # Find appropriate key
            if kid:
                if kid not in self.keys:
                    raise jwt.InvalidKeyError(f"Unknown key ID: {kid}")
                
                key = self.keys[kid]
                
                if not key.is_valid:
                    raise jwt.InvalidKeyError(f"Key {kid} is no longer valid")
            else:
                # Fallback to active key if no kid
                key = self.active_key
                if not key:
                    raise jwt.InvalidKeyError("No active key available")
            
            # Verify token
            claims = jwt.decode(
                token,
                key.secret,
                algorithms=[key.algorithm],
                audience=self.audience,
                issuer=self.issuer,
                options={"verify_exp": verify_exp}
            )
            
            # Check if token is revoked
            jti = claims.get("jti")
            if jti and await self.is_token_revoked(jti):
                raise jwt.InvalidTokenError("Token has been revoked")
            
            # Verify token type if specified
            if token_type and claims.get("type") != token_type.value:
                raise jwt.InvalidTokenError(
                    f"Invalid token type: expected {token_type.value}, "
                    f"got {claims.get('type')}"
                )
            
            # Update metrics
            self.metrics["tokens_verified"] += 1
            
            return claims
            
        except jwt.ExpiredSignatureError:
            self.metrics["verification_failures"] += 1
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            self.metrics["verification_failures"] += 1
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            self.metrics["verification_failures"] += 1
            logger.error(f"Token verification error: {e}")
            raise HTTPException(status_code=401, detail="Token verification failed")
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke a token by its JTI"""
        try:
            # Decode token to get JTI (without verification for revocation)
            claims = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False}
            )
            
            jti = claims.get("jti")
            if not jti:
                return False
            
            # Add to revocation list
            self.revoked_tokens.add(jti)
            
            # Store in Redis with TTL
            if self.redis:
                exp = claims.get("exp", 0)
                ttl = max(0, exp - time.time()) if exp else self.revocation_cache_ttl
                await self.redis.setex(
                    f"jwt:revoked:{jti}",
                    int(ttl),
                    "1"
                )
            
            # Update metrics
            self.metrics["tokens_revoked"] += 1
            
            logger.info(f"Token revoked: {jti}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    async def is_token_revoked(self, jti: str) -> bool:
        """Check if a token is revoked"""
        # Check memory cache
        if jti in self.revoked_tokens:
            return True
        
        # Check Redis
        if self.redis:
            try:
                result = await self.redis.get(f"jwt:revoked:{jti}")
                return result == "1"
            except Exception:
                pass
        
        return False
    
    async def _load_keys_from_redis(self):
        """Load JWT keys from Redis"""
        if not self.redis:
            return
        
        try:
            # Get all key IDs
            key_pattern = "jwt:key:*"
            keys = await self.redis.keys(key_pattern)
            
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
            
            logger.info(f"Loaded {len(self.keys)} JWT keys from Redis")
            
        except Exception as e:
            logger.error(f"Failed to load keys from Redis: {e}")
    
    async def _save_key_to_redis(self, key: JWTKey):
        """Save a JWT key to Redis"""
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
            
            # Save with expiration
            ttl = int((key.expires_at - datetime.now(timezone.utc)).total_seconds())
            await self.redis.setex(
                f"jwt:key:{key.kid}",
                max(ttl, 86400),  # At least 1 day
                json.dumps(key_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to save key to Redis: {e}")
    
    async def _key_rotation_scheduler(self):
        """Background task to rotate keys on schedule"""
        while True:
            try:
                # Check every hour
                await asyncio.sleep(3600)
                
                if self.active_key:
                    # Check if active key is near expiration
                    time_until_expiry = (
                        self.active_key.expires_at - datetime.now(timezone.utc)
                    ).total_seconds()
                    
                    # Rotate if less than 25% of lifetime remaining
                    if time_until_expiry < (self.key_rotation_days * 86400 * 0.25):
                        logger.info("Scheduled key rotation triggered")
                        await self.rotate_keys()
                
            except Exception as e:
                logger.error(f"Key rotation scheduler error: {e}")
    
    async def _revocation_cleanup_scheduler(self):
        """Clean up expired revocations from memory"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run hourly
                
                # This is simplified - in production you'd track expiration times
                if len(self.revoked_tokens) > 10000:  # Arbitrary limit
                    self.revoked_tokens.clear()
                    logger.info("Cleared revocation cache")
                
            except Exception as e:
                logger.error(f"Revocation cleanup error: {e}")
    
    async def _notify_key_rotation(self, new_key: JWTKey):
        """Notify systems about key rotation"""
        notification = {
            "event": "jwt_key_rotation",
            "new_kid": new_key.kid,
            "algorithm": new_key.algorithm,
            "expires_at": new_key.expires_at.isoformat(),
            "rotation_version": new_key.rotation_version,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Log the event
        logger.info(f"Key rotation notification: {json.dumps(notification)}")
        
        # Could send to webhook, message queue, etc.
        # await send_to_webhook(notification)
    
    async def get_public_keys(self) -> List[Dict[str, Any]]:
        """Get public key information (for JWKS endpoint)"""
        public_keys = []
        
        for key in self.keys.values():
            if key.is_valid:
                # For symmetric keys, don't expose the secret
                public_keys.append({
                    "kid": key.kid,
                    "alg": key.algorithm,
                    "use": "sig",
                    "created_at": key.created_at.isoformat(),
                    "expires_at": key.expires_at.isoformat()
                })
        
        return public_keys
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get JWT manager metrics"""
        return {
            **self.metrics,
            "active_keys": len([k for k in self.keys.values() if k.is_valid]),
            "active_key_kid": self.active_key.kid if self.active_key else None,
            "oldest_key_age_days": self._get_oldest_key_age(),
            "newest_key_age_hours": self._get_newest_key_age()
        }
    
    def _get_oldest_key_age(self) -> int:
        """Get age of oldest valid key in days"""
        if not self.keys:
            return 0
        
        oldest = min(
            self.keys.values(),
            key=lambda k: k.created_at
        )
        
        age = datetime.now(timezone.utc) - oldest.created_at
        return age.days
    
    def _get_newest_key_age(self) -> float:
        """Get age of newest key in hours"""
        if not self.active_key:
            return 0
        
        age = datetime.now(timezone.utc) - self.active_key.created_at
        return age.total_seconds() / 3600


# FastAPI dependencies
security = HTTPBearer()


async def get_jwt_manager() -> JWTRotationManager:
    """Get or create JWT rotation manager instance"""
    # This would be initialized at app startup
    # For now, create a new instance
    manager = JWTRotationManager()
    await manager.initialize()
    return manager


async def verify_access_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """FastAPI dependency to verify access tokens"""
    manager = await get_jwt_manager()
    
    claims = await manager.verify_token(
        credentials.credentials,
        token_type=TokenType.ACCESS
    )
    
    return claims


async def verify_refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """FastAPI dependency to verify refresh tokens"""
    manager = await get_jwt_manager()
    
    claims = await manager.verify_token(
        credentials.credentials,
        token_type=TokenType.REFRESH
    )
    
    return claims
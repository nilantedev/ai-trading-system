"""Database repository for user management."""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from .user_models import (
    Users, UserSessions, RefreshTokens, UserAuditLogs,
    LoginAttempts, ApiKeys, UserRole, UserStatus
)
from .logging import get_logger

logger = get_logger(__name__)


class UserRepository:
    """Repository for user database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_user(self, user_data: Dict[str, Any]) -> Users:
        """Create a new user in the database."""
        try:
            user = Users(**user_data)
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            return user
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create user: {e}")
            raise
    
    async def get_user_by_id(self, user_id: str) -> Optional[Users]:
        """Get user by ID."""
        try:
            result = await self.session.execute(
                select(Users).where(
                    and_(
                        Users.user_id == user_id,
                        Users.deleted_at.is_(None)
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[Users]:
        """Get user by username."""
        try:
            result = await self.session.execute(
                select(Users).where(
                    and_(
                        Users.username == username,
                        Users.deleted_at.is_(None)
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[Users]:
        """Get user by email."""
        try:
            result = await self.session.execute(
                select(Users).where(
                    and_(
                        Users.email == email,
                        Users.deleted_at.is_(None)
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            return None
    
    async def get_user_by_api_key(self, api_key_hash: str) -> Optional[Users]:
        """Get user by API key hash."""
        try:
            result = await self.session.execute(
                select(Users).join(ApiKeys).where(
                    and_(
                        ApiKeys.key_hash == api_key_hash,
                        ApiKeys.revoked == False,
                        or_(
                            ApiKeys.expires_at.is_(None),
                            ApiKeys.expires_at > datetime.utcnow()
                        ),
                        Users.deleted_at.is_(None)
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user by API key: {e}")
            return None
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user fields."""
        try:
            updates['updated_at'] = datetime.utcnow()
            result = await self.session.execute(
                update(Users).where(
                    and_(
                        Users.user_id == user_id,
                        Users.deleted_at.is_(None)
                    )
                ).values(**updates)
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update user {user_id}: {e}")
            return False
    
    async def delete_user(self, user_id: str, soft_delete: bool = True) -> bool:
        """Delete or soft-delete a user."""
        try:
            if soft_delete:
                result = await self.session.execute(
                    update(Users).where(
                        Users.user_id == user_id
                    ).values(
                        deleted_at=datetime.utcnow(),
                        status=UserStatus.INACTIVE
                    )
                )
            else:
                result = await self.session.execute(
                    delete(Users).where(Users.user_id == user_id)
                )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
        role: Optional[UserRole] = None,
        status: Optional[UserStatus] = None
    ) -> List[Users]:
        """List users with optional filtering."""
        try:
            query = select(Users).where(Users.deleted_at.is_(None))
            
            if role:
                query = query.where(Users.role == role)
            if status:
                query = query.where(Users.status == status)
            
            query = query.offset(skip).limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []
    
    async def create_session(self, session_data: Dict[str, Any]) -> UserSessions:
        """Create a new user session."""
        try:
            session = UserSessions(**session_data)
            self.session.add(session)
            await self.session.commit()
            await self.session.refresh(session)
            return session
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[UserSessions]:
        """Get session by ID."""
        try:
            result = await self.session.execute(
                select(UserSessions).where(
                    and_(
                        UserSessions.session_id == session_id,
                        UserSessions.revoked == False,
                        UserSessions.expires_at > datetime.utcnow()
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last accessed time."""
        try:
            result = await self.session.execute(
                update(UserSessions).where(
                    UserSessions.session_id == session_id
                ).values(last_accessed=datetime.utcnow())
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update session activity {session_id}: {e}")
            return False
    
    async def revoke_session(self, session_id: str, reason: str = "") -> bool:
        """Revoke a user session."""
        try:
            result = await self.session.execute(
                update(UserSessions).where(
                    UserSessions.session_id == session_id
                ).values(
                    revoked=True,
                    revoked_at=datetime.utcnow(),
                    revoke_reason=reason
                )
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to revoke session {session_id}: {e}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Delete expired sessions."""
        try:
            result = await self.session.execute(
                delete(UserSessions).where(
                    UserSessions.expires_at < datetime.utcnow()
                )
            )
            await self.session.commit()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    async def create_refresh_token(self, token_data: Dict[str, Any]) -> RefreshTokens:
        """Store a new refresh token."""
        try:
            token = RefreshTokens(**token_data)
            self.session.add(token)
            await self.session.commit()
            await self.session.refresh(token)
            return token
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create refresh token: {e}")
            raise
    
    async def get_refresh_token(self, jti: str) -> Optional[RefreshTokens]:
        """Get refresh token by JTI."""
        try:
            result = await self.session.execute(
                select(RefreshTokens).where(
                    and_(
                        RefreshTokens.jti == jti,
                        RefreshTokens.revoked == False,
                        RefreshTokens.expires_at > datetime.utcnow()
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get refresh token {jti}: {e}")
            return None
    
    async def revoke_refresh_token(self, jti: str, reason: str = "") -> bool:
        """Revoke a refresh token."""
        try:
            result = await self.session.execute(
                update(RefreshTokens).where(
                    RefreshTokens.jti == jti
                ).values(
                    revoked=True,
                    revoked_at=datetime.utcnow(),
                    revoke_reason=reason
                )
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to revoke refresh token {jti}: {e}")
            return False
    
    async def revoke_token_family(self, family_id: str, reason: str = "family_compromise") -> int:
        """Revoke all tokens in a family (for refresh token rotation)."""
        try:
            result = await self.session.execute(
                update(RefreshTokens).where(
                    RefreshTokens.family_id == family_id
                ).values(
                    revoked=True,
                    revoked_at=datetime.utcnow(),
                    revoke_reason=reason
                )
            )
            await self.session.commit()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to revoke token family {family_id}: {e}")
            return 0
    
    async def log_audit_event(self, event_data: Dict[str, Any]) -> UserAuditLogs:
        """Log an audit event."""
        try:
            event_data['audit_id'] = str(uuid.uuid4())
            event = UserAuditLogs(**event_data)
            self.session.add(event)
            await self.session.commit()
            return event
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to log audit event: {e}")
            raise
    
    async def log_login_attempt(self, attempt_data: Dict[str, Any]) -> LoginAttempts:
        """Log a login attempt."""
        try:
            attempt_data['attempt_id'] = str(uuid.uuid4())
            attempt = LoginAttempts(**attempt_data)
            self.session.add(attempt)
            await self.session.commit()
            return attempt
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to log login attempt: {e}")
            raise
    
    async def get_recent_login_attempts(
        self,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        minutes: int = 30
    ) -> List[LoginAttempts]:
        """Get recent login attempts."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            query = select(LoginAttempts).where(
                LoginAttempts.attempt_timestamp > cutoff_time
            )
            
            if username:
                query = query.where(LoginAttempts.username == username)
            if ip_address:
                query = query.where(LoginAttempts.ip_address == ip_address)
            
            query = query.order_by(LoginAttempts.attempt_timestamp.desc())
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Failed to get login attempts: {e}")
            return []
    
    async def create_api_key(self, key_data: Dict[str, Any]) -> ApiKeys:
        """Create a new API key."""
        try:
            key = ApiKeys(**key_data)
            self.session.add(key)
            await self.session.commit()
            await self.session.refresh(key)
            return key
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create API key: {e}")
            raise
    
    async def get_api_key(self, key_hash: str) -> Optional[ApiKeys]:
        """Get API key by hash."""
        try:
            result = await self.session.execute(
                select(ApiKeys).where(
                    and_(
                        ApiKeys.key_hash == key_hash,
                        ApiKeys.revoked == False,
                        or_(
                            ApiKeys.expires_at.is_(None),
                            ApiKeys.expires_at > datetime.utcnow()
                        )
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get API key: {e}")
            return None
    
    async def update_api_key_usage(self, key_id: str) -> bool:
        """Update API key last used timestamp."""
        try:
            result = await self.session.execute(
                update(ApiKeys).where(
                    ApiKeys.key_id == key_id
                ).values(last_used_at=datetime.utcnow())
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update API key usage {key_id}: {e}")
            return False
    
    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        try:
            result = await self.session.execute(
                update(ApiKeys).where(
                    ApiKeys.key_id == key_id
                ).values(
                    revoked=True,
                    revoked_at=datetime.utcnow()
                )
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to revoke API key {key_id}: {e}")
            return False
    
    async def cleanup_expired_tokens(self) -> Dict[str, int]:
        """Cleanup expired tokens and sessions."""
        try:
            # Clean refresh tokens
            refresh_result = await self.session.execute(
                delete(RefreshTokens).where(
                    RefreshTokens.expires_at < datetime.utcnow()
                )
            )
            
            # Clean sessions
            session_result = await self.session.execute(
                delete(UserSessions).where(
                    UserSessions.expires_at < datetime.utcnow()
                )
            )
            
            await self.session.commit()
            
            return {
                'refresh_tokens': refresh_result.rowcount,
                'sessions': session_result.rowcount
            }
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to cleanup expired tokens: {e}")
            return {'refresh_tokens': 0, 'sessions': 0}
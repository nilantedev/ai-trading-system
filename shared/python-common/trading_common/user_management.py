#!/usr/bin/env python3
"""
User Management System for AI Trading System.
Provides multi-user authentication, role-based access control, and user lifecycle management.
"""

import asyncio
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging

try:
    import asyncpg
    import bcrypt
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    from sqlalchemy import (
        Column, String, DateTime, Boolean, Integer, Text, 
        JSON, create_engine, MetaData, Table
    )
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import declarative_base
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .config import get_settings
from .logging import get_logger
from .security_store import get_security_store, log_security_event, SecurityEventType
from .database import get_redis_client
from .user_repository import UserRepository
from .user_models import Users, UserSessions, UserRole as DBUserRole, UserStatus as DBUserStatus
import functools

logger = get_logger(__name__)


class UserRole(str, Enum):
    """User roles with hierarchical permissions."""
    SUPER_ADMIN = "super_admin"       # Full system access
    ADMIN = "admin"                   # User and system management
    TRADER = "trader"                 # Trading operations
    ANALYST = "analyst"               # Read-only data access
    API_USER = "api_user"             # Programmatic access
    VIEWER = "viewer"                 # Limited read-only access


class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    PENDING_VERIFICATION = "pending_verification"


@dataclass
class User:
    """User account representation."""
    user_id: str
    username: str
    email: str
    role: UserRole
    status: UserStatus
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    password_hash: Optional[str] = None
    salt: Optional[str] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    password_expires_at: Optional[datetime] = None
    two_factor_enabled: bool = False
    api_key: Optional[str] = None
    permissions: Set[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.permissions is None:
            self.permissions = set()
        if self.metadata is None:
            self.metadata = {}


@dataclass  
class Session:
    """User session representation."""
    session_id: str
    user_id: str
    username: str
    role: UserRole
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_api_session: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


class PermissionManager:
    """Manages role-based permissions."""
    
    # Define role permissions
    ROLE_PERMISSIONS = {
        UserRole.SUPER_ADMIN: {
            'user:create', 'user:read', 'user:update', 'user:delete',
            'system:admin', 'system:config', 'system:metrics',
            'trading:execute', 'trading:read', 'trading:manage',
            'data:read', 'data:write', 'models:deploy', 'models:manage'
        },
        UserRole.ADMIN: {
            'user:create', 'user:read', 'user:update',
            'system:metrics', 'trading:read', 'trading:manage',
            'data:read', 'data:write', 'models:manage'
        },
        UserRole.TRADER: {
            'trading:execute', 'trading:read', 'data:read', 'models:read'
        },
        UserRole.ANALYST: {
            'data:read', 'models:read', 'system:metrics'
        },
        UserRole.API_USER: {
            'trading:execute', 'data:read', 'models:read'
        },
        UserRole.VIEWER: {
            'data:read'
        }
    }
    
    @classmethod
    def get_role_permissions(cls, role: UserRole) -> Set[str]:
        """Get permissions for a role."""
        return cls.ROLE_PERMISSIONS.get(role, set())
    
    @classmethod
    def has_permission(cls, role: UserRole, permission: str) -> bool:
        """Check if role has specific permission."""
        return permission in cls.get_role_permissions(role)
    
    @classmethod
    def can_access_endpoint(cls, role: UserRole, endpoint: str) -> bool:
        """Check if role can access specific endpoint."""
        # Map endpoints to required permissions
        endpoint_permissions = {
            '/admin': 'system:admin',
            '/users': 'user:read',
            '/trading': 'trading:read',
            '/orders': 'trading:execute',
            '/data': 'data:read',
            '/models': 'models:read',
            '/metrics': 'system:metrics'
        }
        
        for pattern, permission in endpoint_permissions.items():
            if endpoint.startswith(pattern):
                return cls.has_permission(role, permission)
        
        # Default: allow basic read access
        return cls.has_permission(role, 'data:read')


class UserManager:
    """Manages user accounts and authentication."""
    
    def __init__(self):
        """Initialize user manager."""
        self.settings = get_settings()
        self.security_store = get_security_store()
        self.redis = get_redis_client()
        
        # Password policy
        self.min_password_length = 12
        self.password_history_count = 5
        self.password_expiry_days = 90
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30
        
        # Session management
        self.session_timeout_hours = 8
        self.api_session_timeout_hours = 24
        
        # Database connection
        self._engine = None
        self._session_factory = None
        self._db_manager = None
        
        if SQLALCHEMY_AVAILABLE:
            self._init_database()
    
    def _init_database(self):
        """Initialize database connection and tables."""
        try:
            # Get database URL from settings
            db_url = self.settings.postgres_url or self.settings.database_url
            if not db_url:
                db_url = "postgresql://trading_user:trading_password@localhost:5432/trading_db"
            
            # Ensure async URL format
            if "postgresql://" in db_url:
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
            
            self._engine = create_async_engine(db_url, echo=False)
            self._session_factory = async_sessionmaker(self._engine, class_=AsyncSession)
            
            # Database manager not needed for basic functionality
            
            logger.info("Database connection initialized for user management")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Don't raise - allow system to work with degraded functionality
            self._engine = None
            self._session_factory = None
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.VIEWER,
        created_by: Optional[str] = None
    ) -> User:
        """Create a new user account."""
        # Validate inputs
        if len(username) < 3 or len(username) > 50:
            raise ValueError("Username must be 3-50 characters")
        
        if not self._is_valid_email(email):
            raise ValueError("Invalid email format")
        
        if not self._is_valid_password(password):
            raise ValueError(f"Password must be at least {self.min_password_length} characters")
        
        # Check if user exists
        if await self._user_exists(username, email):
            raise ValueError("User already exists")
        
        try:
            # Generate secure password hash
            salt = secrets.token_hex(32)
            password_hash = self._hash_password(password, salt)
            
            # Create user object
            user = User(
                user_id=str(uuid.uuid4()),
                username=username,
                email=email,
                role=role,
                status=UserStatus.ACTIVE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                password_hash=password_hash,
                salt=salt,
                password_expires_at=datetime.utcnow() + timedelta(days=self.password_expiry_days),
                permissions=PermissionManager.get_role_permissions(role)
            )
            
            # Store user in database
            await self._store_user(user)
            
            # Log security event
            await log_security_event(
                event_type=SecurityEventType.USER_CREATED,
                user_id=user.user_id,
                details={
                    'username': username,
                    'role': role.value,
                    'created_by': created_by
                }
            )
            
            logger.info(f"Created user {username} with role {role.value}")
            return user
            
        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            raise
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> Optional[User]:
        """Authenticate user credentials."""
        try:
            # Get user from database
            user = await self._get_user_by_username(username)
            if not user:
                await self._log_failed_login(username, ip_address, "user_not_found")
                return None
            
            # Check account status
            if user.status != UserStatus.ACTIVE:
                await self._log_failed_login(username, ip_address, f"account_{user.status.value}")
                return None
            
            # Check if account is locked
            if user.account_locked_until and user.account_locked_until > datetime.utcnow():
                await self._log_failed_login(username, ip_address, "account_locked")
                return None
            
            # Verify password
            if not self._verify_password(password, user.password_hash, user.salt):
                await self._handle_failed_login(user, ip_address)
                return None
            
            # Check password expiry
            if user.password_expires_at and user.password_expires_at < datetime.utcnow():
                await self._log_failed_login(username, ip_address, "password_expired")
                return None
            
            # Successful authentication
            await self._handle_successful_login(user, ip_address)
            
            return user
            
        except Exception as e:
            logger.error(f"Authentication failed for {username}: {e}")
            return None
    
    async def create_session(
        self,
        user: User,
        is_api: bool = False,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Session:
        """Create a new user session."""
        try:
            session_timeout = (
                self.api_session_timeout_hours if is_api 
                else self.session_timeout_hours
            )
            
            session = Session(
                session_id=str(uuid.uuid4()),
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=session_timeout),
                ip_address=ip_address,
                user_agent=user_agent,
                is_api_session=is_api
            )
            
            # Store session in Redis
            session_key = f"session:{session.session_id}"
            session_data = asdict(session)
            # Convert datetime objects to ISO format
            for key, value in session_data.items():
                if isinstance(value, datetime):
                    session_data[key] = value.isoformat()
                elif isinstance(value, UserRole):
                    session_data[key] = value.value
            
            await self.redis.setex(
                session_key,
                int(timedelta(hours=session_timeout).total_seconds()),
                json.dumps(session_data, default=str)
            )
            
            # Log session creation
            await log_security_event(
                event_type=SecurityEventType.SESSION_CREATED,
                user_id=user.user_id,
                session_id=session.session_id,
                details={
                    'username': user.username,
                    'is_api': is_api,
                    'ip_address': ip_address
                }
            )
            
            logger.info(f"Created session for user {user.username}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session for user {user.username}: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve and validate a session."""
        try:
            session_key = f"session:{session_id}"
            session_data = await self.redis.get(session_key)
            
            if not session_data:
                return None
            
            session_dict = json.loads(session_data)
            
            # Convert ISO datetime strings back to datetime objects
            for key in ['created_at', 'last_accessed', 'expires_at']:
                if session_dict[key]:
                    session_dict[key] = datetime.fromisoformat(session_dict[key])
            
            session_dict['role'] = UserRole(session_dict['role'])
            
            session = Session(**session_dict)
            
            # Check if session has expired
            if session.expires_at < datetime.utcnow():
                await self.invalidate_session(session_id)
                return None
            
            # Update last accessed time
            session.last_accessed = datetime.utcnow()
            await self._update_session(session)
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def invalidate_session(self, session_id: str):
        """Invalidate a user session."""
        try:
            session_key = f"session:{session_id}"
            await self.redis.delete(session_key)
            
            await log_security_event(
                event_type=SecurityEventType.SESSION_INVALIDATED,
                session_id=session_id,
                details={'reason': 'explicit_invalidation'}
            )
            
        except Exception as e:
            logger.error(f"Failed to invalidate session {session_id}: {e}")
    
    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str
    ) -> bool:
        """Change user password."""
        try:
            user = await self._get_user_by_id(user_id)
            if not user:
                return False
            
            # Verify current password
            if not self._verify_password(current_password, user.password_hash, user.salt):
                return False
            
            # Validate new password
            if not self._is_valid_password(new_password):
                raise ValueError(f"Password must be at least {self.min_password_length} characters")
            
            # Generate new hash
            salt = secrets.token_hex(32)
            password_hash = self._hash_password(new_password, salt)
            
            # Update user
            user.password_hash = password_hash
            user.salt = salt
            user.password_expires_at = datetime.utcnow() + timedelta(days=self.password_expiry_days)
            user.updated_at = datetime.utcnow()
            
            await self._update_user(user)
            
            # Log security event
            await log_security_event(
                event_type=SecurityEventType.PASSWORD_CHANGED,
                user_id=user_id,
                details={'username': user.username}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to change password for user {user_id}: {e}")
            return False
    
    async def update_user_role(self, user_id: str, new_role: UserRole, updated_by: str) -> bool:
        """Update user role and permissions."""
        try:
            user = await self._get_user_by_id(user_id)
            if not user:
                return False
            
            old_role = user.role
            user.role = new_role
            user.permissions = PermissionManager.get_role_permissions(new_role)
            user.updated_at = datetime.utcnow()
            
            await self._update_user(user)
            
            # Log security event
            await log_security_event(
                event_type=SecurityEventType.ROLE_CHANGED,
                user_id=user_id,
                details={
                    'username': user.username,
                    'old_role': old_role.value,
                    'new_role': new_role.value,
                    'updated_by': updated_by
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update role for user {user_id}: {e}")
            return False
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using bcrypt only."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "bcrypt is required for password hashing. "
                "Install it with: pip install bcrypt"
            )
        return bcrypt.hashpw((password + salt).encode(), bcrypt.gensalt()).decode()
    
    def _verify_password(self, password: str, hash: str, salt: str) -> bool:
        """Verify password against hash using bcrypt only."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "bcrypt is required for password verification. "
                "Install it with: pip install bcrypt"
            )
        return bcrypt.checkpw((password + salt).encode(), hash.encode())
    
    def _is_valid_email(self, email: str) -> bool:
        """Basic email validation."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _is_valid_password(self, password: str) -> bool:
        """Validate password strength."""
        if len(password) < self.min_password_length:
            return False
        
        # Check for complexity (at least one letter, number, special char)
        has_letter = any(c.isalpha() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_letter and has_digit and has_special
    
    async def _user_exists(self, username: str, email: str) -> bool:
        """Check if user already exists."""
        if not self._session_factory:
            return False
        
        try:
            async with self._session_factory() as session:
                repo = UserRepository(session)
                user_by_username = await repo.get_user_by_username(username)
                user_by_email = await repo.get_user_by_email(email)
                return user_by_username is not None or user_by_email is not None
        except Exception as e:
            logger.error(f"Error checking if user exists: {e}")
            return False
    
    async def _store_user(self, user: User):
        """Store user in database."""
        if not self._session_factory:
            logger.warning("Database not available, user not persisted")
            return
        
        try:
            async with self._session_factory() as session:
                repo = UserRepository(session)
                
                # Convert User dataclass to database model
                user_data = {
                    'user_id': user.user_id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'status': user.status,
                    'password_hash': user.password_hash,
                    'salt': user.salt,
                    'password_expires_at': user.password_expires_at,
                    'failed_login_attempts': user.failed_login_attempts,
                    'account_locked_until': user.account_locked_until,
                    'two_factor_enabled': user.two_factor_enabled,
                    'permissions': list(user.permissions) if user.permissions else [],
                    'user_metadata': user.metadata or {},
                    'created_at': user.created_at,
                    'updated_at': user.updated_at
                }
                
                await repo.create_user(user_data)
                logger.info(f"User {user.username} stored in database")
                
        except Exception as e:
            logger.error(f"Failed to store user in database: {e}")
    
    async def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username from database."""
        if not self._session_factory:
            return None
        
        try:
            async with self._session_factory() as session:
                repo = UserRepository(session)
                db_user = await repo.get_user_by_username(username)
                
                if not db_user:
                    return None
                
                # Convert database model to User dataclass
                return User(
                    user_id=db_user.user_id,
                    username=db_user.username,
                    email=db_user.email,
                    role=UserRole(db_user.role.value),
                    status=UserStatus(db_user.status.value),
                    created_at=db_user.created_at,
                    updated_at=db_user.updated_at,
                    last_login=db_user.last_login,
                    password_hash=db_user.password_hash,
                    salt=db_user.salt,
                    failed_login_attempts=db_user.failed_login_attempts,
                    account_locked_until=db_user.account_locked_until,
                    password_expires_at=db_user.password_expires_at,
                    two_factor_enabled=db_user.two_factor_enabled,
                    api_key=db_user.api_key,
                    permissions=set(db_user.permissions) if db_user.permissions else set(),
                    metadata=db_user.user_metadata or {}
                )
                
        except Exception as e:
            logger.error(f"Failed to get user by username: {e}")
            return None
    
    async def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID from database."""
        if not self._session_factory:
            return None
        
        try:
            async with self._session_factory() as session:
                repo = UserRepository(session)
                db_user = await repo.get_user_by_id(user_id)
                
                if not db_user:
                    return None
                
                # Convert database model to User dataclass
                return User(
                    user_id=db_user.user_id,
                    username=db_user.username,
                    email=db_user.email,
                    role=UserRole(db_user.role.value),
                    status=UserStatus(db_user.status.value),
                    created_at=db_user.created_at,
                    updated_at=db_user.updated_at,
                    last_login=db_user.last_login,
                    password_hash=db_user.password_hash,
                    salt=db_user.salt,
                    failed_login_attempts=db_user.failed_login_attempts,
                    account_locked_until=db_user.account_locked_until,
                    password_expires_at=db_user.password_expires_at,
                    two_factor_enabled=db_user.two_factor_enabled,
                    api_key=db_user.api_key,
                    permissions=set(db_user.permissions) if db_user.permissions else set(),
                    metadata=db_user.user_metadata or {}
                )
                
        except Exception as e:
            logger.error(f"Failed to get user by ID: {e}")
            return None
    
    async def _update_user(self, user: User):
        """Update user in database."""
        if not self._session_factory:
            logger.warning("Database not available, user update not persisted")
            return
        
        try:
            async with self._session_factory() as session:
                repo = UserRepository(session)
                
                updates = {
                    'status': user.status.value,
                    'failed_login_attempts': user.failed_login_attempts,
                    'account_locked_until': user.account_locked_until,
                    'last_login': user.last_login,
                    'password_hash': user.password_hash,
                    'salt': user.salt,
                    'password_expires_at': user.password_expires_at,
                    'updated_at': user.updated_at
                }
                
                await repo.update_user(user.user_id, updates)
                logger.info(f"User {user.username} updated in database")
                
        except Exception as e:
            logger.error(f"Failed to update user in database: {e}")
    
    async def _update_session(self, session: Session):
        """Update session in Redis."""
        session_key = f"session:{session.session_id}"
        session_data = asdict(session)
        
        # Convert datetime objects to ISO format
        for key, value in session_data.items():
            if isinstance(value, datetime):
                session_data[key] = value.isoformat()
            elif isinstance(value, UserRole):
                session_data[key] = value.value
        
        timeout = int((session.expires_at - datetime.utcnow()).total_seconds())
        await self.redis.setex(session_key, timeout, json.dumps(session_data, default=str))
    
    async def _handle_failed_login(self, user: User, ip_address: Optional[str]):
        """Handle failed login attempt."""
        user.failed_login_attempts += 1
        
        if user.failed_login_attempts >= self.max_failed_attempts:
            user.account_locked_until = datetime.utcnow() + timedelta(minutes=self.lockout_duration_minutes)
            user.status = UserStatus.LOCKED
        
        user.updated_at = datetime.utcnow()
        await self._update_user(user)
        
        await log_security_event(
            event_type=SecurityEventType.LOGIN_FAILED,
            user_id=user.user_id,
            details={
                'username': user.username,
                'ip_address': ip_address,
                'failed_attempts': user.failed_login_attempts,
                'locked': user.account_locked_until is not None
            }
        )
    
    async def _handle_successful_login(self, user: User, ip_address: Optional[str]):
        """Handle successful login."""
        user.failed_login_attempts = 0
        user.account_locked_until = None
        user.last_login = datetime.utcnow()
        user.updated_at = datetime.utcnow()
        
        if user.status == UserStatus.LOCKED:
            user.status = UserStatus.ACTIVE
        
        await self._update_user(user)
        
        await log_security_event(
            event_type=SecurityEventType.LOGIN_SUCCESSFUL,
            user_id=user.user_id,
            details={
                'username': user.username,
                'ip_address': ip_address
            }
        )
    
    async def _log_failed_login(self, username: str, ip_address: Optional[str], reason: str):
        """Log failed login attempt."""
        await log_security_event(
            event_type=SecurityEventType.LOGIN_FAILED,
            details={
                'username': username,
                'ip_address': ip_address,
                'reason': reason
            }
        )


# Global user manager instance
_user_manager: Optional[UserManager] = None


def get_user_manager() -> UserManager:
    """Get or create global user manager instance."""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager


async def create_default_admin_user():
    """Create default admin user on first run."""
    try:
        user_manager = get_user_manager()
        settings = get_settings()
        
        # Check if admin user already exists
        admin_exists = await user_manager._user_exists("admin", settings.admin_email)
        if admin_exists:
            return
        
        # Create default admin user
        admin_password = settings.admin_password or secrets.token_urlsafe(16)
        
        admin_user = await user_manager.create_user(
            username="admin",
            email=settings.admin_email,
            password=admin_password,
            role=UserRole.SUPER_ADMIN,
            created_by="system"
        )
        
        # Save credentials to secure file instead of logging
        import os
        cred_file = os.path.join("/tmp", ".admin_init_credentials")
        with open(cred_file, 'w') as f:
            f.write(f"Admin user created\n")
            f.write(f"Username: admin\n")
            f.write(f"Email: {admin_email}\n")
            f.write(f"Password: {admin_password}\n")
            f.write("\n⚠️  CHANGE THIS PASSWORD IMMEDIATELY!\n")
        os.chmod(cred_file, 0o600)
        
        logger.warning(f"Created default admin user - credentials saved to: {cred_file}")
        logger.warning("Please change this password immediately!")
        
        return admin_user
        
    except Exception as e:
        logger.error(f"Failed to create default admin user: {e}")


def require_permission(permission: str):
    """Decorator to require specific permission for endpoint access."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract session/user from request context
            # This would be implemented based on your FastAPI setup
            # For now, this is a placeholder
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(min_role: UserRole):
    """Decorator to require minimum role for endpoint access."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract and validate user role from request context
            # This would be implemented based on your FastAPI setup
            return await func(*args, **kwargs)
        return wrapper
    return decorator
"""SQLAlchemy models for user management system."""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, Text, JSON,
    Index, UniqueConstraint, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    """User roles with hierarchical permissions."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin" 
    TRADER = "trader"
    ANALYST = "analyst"
    API_USER = "api_user"
    VIEWER = "viewer"


class UserStatus(str, enum.Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    PENDING_VERIFICATION = "pending_verification"


class Users(Base):
    """User account database model."""
    __tablename__ = 'users'
    
    # Primary fields
    user_id = Column(String(36), primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    status = Column(SQLEnum(UserStatus), nullable=False, default=UserStatus.ACTIVE)
    
    # Authentication
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(64), nullable=False)
    password_expires_at = Column(DateTime)
    password_history = Column(JSON)  # Store last N password hashes
    
    # Security
    failed_login_attempts = Column(Integer, default=0)
    account_locked_until = Column(DateTime)
    last_login = Column(DateTime)
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(255))
    
    # API access
    api_key = Column(String(64), unique=True, index=True)
    api_key_expires_at = Column(DateTime)
    
    # Permissions and metadata
    permissions = Column(JSON)  # Set of permission strings
    metadata = Column(JSON)  # Additional user data
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime)  # Soft delete
    
    # Relationships
    sessions = relationship("UserSessions", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("UserAuditLogs", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshTokens", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_users_status_role', 'status', 'role'),
        Index('idx_users_created_at', 'created_at'),
        Index('idx_users_last_login', 'last_login'),
    )


class UserSessions(Base):
    """User session tracking."""
    __tablename__ = 'user_sessions'
    
    session_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    
    # Session details
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_accessed = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False, index=True)
    
    # Context
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(Text)
    is_api_session = Column(Boolean, default=False)
    
    # Security
    revoked = Column(Boolean, default=False)
    revoked_at = Column(DateTime)
    revoke_reason = Column(String(255))
    
    # Metadata
    metadata = Column(JSON)
    
    # Relationships
    user = relationship("Users", back_populates="sessions")
    
    # Indexes
    __table_args__ = (
        Index('idx_sessions_expires', 'expires_at'),
        Index('idx_sessions_user_created', 'user_id', 'created_at'),
    )


class RefreshTokens(Base):
    """JWT refresh token storage for revocation support."""
    __tablename__ = 'refresh_tokens'
    
    jti = Column(String(36), primary_key=True)  # JWT ID
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    
    # Token details
    token_hash = Column(String(64), unique=True, nullable=False)  # SHA-256 of token
    issued_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False, index=True)
    
    # Security
    revoked = Column(Boolean, default=False, index=True)
    revoked_at = Column(DateTime)
    revoke_reason = Column(String(255))
    
    # Context
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Token family for rotation
    family_id = Column(String(36), index=True)  # Track refresh token families
    
    # Relationships
    user = relationship("Users", back_populates="refresh_tokens")
    
    # Indexes
    __table_args__ = (
        Index('idx_refresh_tokens_family', 'family_id', 'revoked'),
        Index('idx_refresh_tokens_expires', 'expires_at', 'revoked'),
    )


class UserAuditLogs(Base):
    """Comprehensive audit logging for user actions."""
    __tablename__ = 'user_audit_logs'
    
    audit_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.user_id'), index=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)
    event_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Context
    session_id = Column(String(36), index=True)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Event data
    details = Column(JSON)
    old_values = Column(JSON)
    new_values = Column(JSON)
    
    # Security
    severity = Column(String(20), index=True)  # INFO, WARNING, ERROR, CRITICAL
    
    # Relationships
    user = relationship("Users", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_timestamp_type', 'event_timestamp', 'event_type'),
        Index('idx_audit_user_timestamp', 'user_id', 'event_timestamp'),
        Index('idx_audit_severity_timestamp', 'severity', 'event_timestamp'),
    )


class UserPermissions(Base):
    """Fine-grained permission definitions."""
    __tablename__ = 'user_permissions'
    
    permission_id = Column(String(36), primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    resource = Column(String(100), index=True)  # e.g., 'trading', 'models', 'users'
    action = Column(String(50), index=True)  # e.g., 'read', 'write', 'execute', 'delete'
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('resource', 'action', name='uq_resource_action'),
        Index('idx_permissions_resource', 'resource'),
    )


class RolePermissions(Base):
    """Role to permission mapping."""
    __tablename__ = 'role_permissions'
    
    role = Column(SQLEnum(UserRole), primary_key=True)
    permission_id = Column(String(36), ForeignKey('user_permissions.permission_id'), primary_key=True)
    
    # Metadata
    granted_at = Column(DateTime, default=datetime.utcnow)
    granted_by = Column(String(36))  # User ID who granted
    
    # Indexes
    __table_args__ = (
        Index('idx_role_permissions', 'role', 'permission_id'),
    )


class LoginAttempts(Base):
    """Track login attempts for security monitoring."""
    __tablename__ = 'login_attempts'
    
    attempt_id = Column(String(36), primary_key=True)
    username = Column(String(50), index=True)
    ip_address = Column(String(45), index=True)
    
    # Attempt details
    attempt_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    success = Column(Boolean, nullable=False)
    failure_reason = Column(String(100))
    
    # Context
    user_agent = Column(Text)
    
    # Indexes
    __table_args__ = (
        Index('idx_attempts_username_timestamp', 'username', 'attempt_timestamp'),
        Index('idx_attempts_ip_timestamp', 'ip_address', 'attempt_timestamp'),
        Index('idx_attempts_success_timestamp', 'success', 'attempt_timestamp'),
    )


class ApiKeys(Base):
    """API key management for programmatic access."""
    __tablename__ = 'api_keys'
    
    key_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    
    # Key details
    key_hash = Column(String(64), unique=True, nullable=False)  # SHA-256 of key
    key_prefix = Column(String(8), index=True)  # First 8 chars for identification
    name = Column(String(100))
    description = Column(Text)
    
    # Security
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, index=True)
    last_used_at = Column(DateTime)
    revoked = Column(Boolean, default=False)
    revoked_at = Column(DateTime)
    
    # Permissions
    scopes = Column(JSON)  # List of allowed scopes
    rate_limit = Column(Integer)  # Requests per hour
    
    # Indexes
    __table_args__ = (
        Index('idx_api_keys_user', 'user_id', 'revoked'),
        Index('idx_api_keys_expires', 'expires_at', 'revoked'),
    )
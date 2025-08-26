"""Create user management tables

Revision ID: 001
Revises: 
Create Date: 2025-01-27 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create user management tables."""
    
    # Create users table
    op.create_table('users',
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('role', sa.Enum('super_admin', 'admin', 'trader', 'analyst', 'api_user', 'viewer', name='userrole'), nullable=False),
        sa.Column('status', sa.Enum('active', 'inactive', 'suspended', 'locked', 'pending_verification', name='userstatus'), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('salt', sa.String(64), nullable=False),
        sa.Column('password_expires_at', sa.DateTime(), nullable=True),
        sa.Column('password_history', sa.JSON(), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=True, default=0),
        sa.Column('account_locked_until', sa.DateTime(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('two_factor_enabled', sa.Boolean(), nullable=True, default=False),
        sa.Column('two_factor_secret', sa.String(255), nullable=True),
        sa.Column('api_key', sa.String(64), nullable=True),
        sa.Column('api_key_expires_at', sa.DateTime(), nullable=True),
        sa.Column('permissions', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('user_id')
    )
    
    # Create indexes for users table
    op.create_index('idx_users_username', 'users', ['username'], unique=True)
    op.create_index('idx_users_email', 'users', ['email'], unique=True)
    op.create_index('idx_users_api_key', 'users', ['api_key'], unique=True)
    op.create_index('idx_users_status_role', 'users', ['status', 'role'])
    op.create_index('idx_users_created_at', 'users', ['created_at'])
    op.create_index('idx_users_last_login', 'users', ['last_login'])
    
    # Create user_sessions table
    op.create_table('user_sessions',
        sa.Column('session_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_accessed', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('is_api_session', sa.Boolean(), nullable=True, default=False),
        sa.Column('revoked', sa.Boolean(), nullable=True, default=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('revoke_reason', sa.String(255), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('session_id')
    )
    
    # Create indexes for user_sessions table
    op.create_index('idx_sessions_user_id', 'user_sessions', ['user_id'])
    op.create_index('idx_sessions_expires', 'user_sessions', ['expires_at'])
    op.create_index('idx_sessions_user_created', 'user_sessions', ['user_id', 'created_at'])
    
    # Create refresh_tokens table
    op.create_table('refresh_tokens',
        sa.Column('jti', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('token_hash', sa.String(64), nullable=False),
        sa.Column('issued_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('revoked', sa.Boolean(), nullable=True, default=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('revoke_reason', sa.String(255), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('family_id', sa.String(36), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('jti')
    )
    
    # Create indexes for refresh_tokens table
    op.create_index('idx_refresh_tokens_user_id', 'refresh_tokens', ['user_id'])
    op.create_index('idx_refresh_tokens_token_hash', 'refresh_tokens', ['token_hash'], unique=True)
    op.create_index('idx_refresh_tokens_revoked', 'refresh_tokens', ['revoked'])
    op.create_index('idx_refresh_tokens_family', 'refresh_tokens', ['family_id', 'revoked'])
    op.create_index('idx_refresh_tokens_expires', 'refresh_tokens', ['expires_at', 'revoked'])
    
    # Create user_audit_logs table
    op.create_table('user_audit_logs',
        sa.Column('audit_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('event_timestamp', sa.DateTime(), nullable=False),
        sa.Column('session_id', sa.String(36), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('old_values', sa.JSON(), nullable=True),
        sa.Column('new_values', sa.JSON(), nullable=True),
        sa.Column('severity', sa.String(20), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('audit_id')
    )
    
    # Create indexes for user_audit_logs table
    op.create_index('idx_audit_user_id', 'user_audit_logs', ['user_id'])
    op.create_index('idx_audit_event_type', 'user_audit_logs', ['event_type'])
    op.create_index('idx_audit_timestamp_type', 'user_audit_logs', ['event_timestamp', 'event_type'])
    op.create_index('idx_audit_user_timestamp', 'user_audit_logs', ['user_id', 'event_timestamp'])
    op.create_index('idx_audit_severity_timestamp', 'user_audit_logs', ['severity', 'event_timestamp'])
    op.create_index('idx_audit_session_id', 'user_audit_logs', ['session_id'])
    
    # Create user_permissions table
    op.create_table('user_permissions',
        sa.Column('permission_id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('resource', sa.String(100), nullable=True),
        sa.Column('action', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('permission_id')
    )
    
    # Create indexes for user_permissions table
    op.create_index('idx_permissions_name', 'user_permissions', ['name'], unique=True)
    op.create_index('idx_permissions_resource', 'user_permissions', ['resource'])
    op.create_unique_constraint('uq_resource_action', 'user_permissions', ['resource', 'action'])
    
    # Create role_permissions table
    op.create_table('role_permissions',
        sa.Column('role', sa.Enum('super_admin', 'admin', 'trader', 'analyst', 'api_user', 'viewer', name='userrole'), nullable=False),
        sa.Column('permission_id', sa.String(36), nullable=False),
        sa.Column('granted_at', sa.DateTime(), nullable=True),
        sa.Column('granted_by', sa.String(36), nullable=True),
        sa.ForeignKeyConstraint(['permission_id'], ['user_permissions.permission_id'], ),
        sa.PrimaryKeyConstraint('role', 'permission_id')
    )
    
    # Create indexes for role_permissions table
    op.create_index('idx_role_permissions', 'role_permissions', ['role', 'permission_id'])
    
    # Create login_attempts table
    op.create_table('login_attempts',
        sa.Column('attempt_id', sa.String(36), nullable=False),
        sa.Column('username', sa.String(50), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('attempt_timestamp', sa.DateTime(), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('failure_reason', sa.String(100), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('attempt_id')
    )
    
    # Create indexes for login_attempts table
    op.create_index('idx_attempts_username', 'login_attempts', ['username'])
    op.create_index('idx_attempts_ip_address', 'login_attempts', ['ip_address'])
    op.create_index('idx_attempts_timestamp', 'login_attempts', ['attempt_timestamp'])
    op.create_index('idx_attempts_username_timestamp', 'login_attempts', ['username', 'attempt_timestamp'])
    op.create_index('idx_attempts_ip_timestamp', 'login_attempts', ['ip_address', 'attempt_timestamp'])
    op.create_index('idx_attempts_success_timestamp', 'login_attempts', ['success', 'attempt_timestamp'])
    
    # Create api_keys table
    op.create_table('api_keys',
        sa.Column('key_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('key_hash', sa.String(64), nullable=False),
        sa.Column('key_prefix', sa.String(8), nullable=True),
        sa.Column('name', sa.String(100), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('revoked', sa.Boolean(), nullable=True, default=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('scopes', sa.JSON(), nullable=True),
        sa.Column('rate_limit', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('key_id')
    )
    
    # Create indexes for api_keys table
    op.create_index('idx_api_keys_user', 'api_keys', ['user_id', 'revoked'])
    op.create_index('idx_api_keys_key_hash', 'api_keys', ['key_hash'], unique=True)
    op.create_index('idx_api_keys_key_prefix', 'api_keys', ['key_prefix'])
    op.create_index('idx_api_keys_expires', 'api_keys', ['expires_at', 'revoked'])


def downgrade() -> None:
    """Drop user management tables."""
    op.drop_table('api_keys')
    op.drop_table('login_attempts')
    op.drop_table('role_permissions')
    op.drop_table('user_permissions')
    op.drop_table('user_audit_logs')
    op.drop_table('refresh_tokens')
    op.drop_table('user_sessions')
    op.drop_table('users')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS userrole')
    op.execute('DROP TYPE IF EXISTS userstatus')
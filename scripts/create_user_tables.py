#!/usr/bin/env python3
"""
Simple script to create user management tables.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get database URL from environment or use default."""
    return (
        os.getenv('DATABASE_URL') or
        os.getenv('POSTGRES_URL') or 
        'postgresql://trading_user:trading_password@localhost:5432/trading_db'
    )


def create_user_tables():
    """Create user management tables directly with SQL."""
    db_url = get_database_url()
    logger.info(f"Connecting to database: {db_url.split('@')[1] if '@' in db_url else db_url}")
    
    try:
        engine = create_engine(db_url, echo=False)
        
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            
            try:
                # Create custom enums first
                logger.info("Creating custom enums...")
                conn.execute(text("""
                    DO $$ BEGIN
                        CREATE TYPE userrole AS ENUM (
                            'super_admin', 'admin', 'trader', 'analyst', 'api_user', 'viewer'
                        );
                    EXCEPTION
                        WHEN duplicate_object THEN null;
                    END $$;
                """))
                
                conn.execute(text("""
                    DO $$ BEGIN
                        CREATE TYPE userstatus AS ENUM (
                            'active', 'inactive', 'suspended', 'locked', 'pending_verification'
                        );
                    EXCEPTION
                        WHEN duplicate_object THEN null;
                    END $$;
                """))
                
                # Create users table
                logger.info("Creating users table...")
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR(36) PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        role userrole NOT NULL DEFAULT 'viewer',
                        status userstatus NOT NULL DEFAULT 'active',
                        password_hash VARCHAR(255) NOT NULL,
                        salt VARCHAR(64) NOT NULL,
                        password_expires_at TIMESTAMP,
                        password_history JSON,
                        failed_login_attempts INTEGER DEFAULT 0,
                        account_locked_until TIMESTAMP,
                        last_login TIMESTAMP,
                        two_factor_enabled BOOLEAN DEFAULT FALSE,
                        two_factor_secret VARCHAR(255),
                        api_key VARCHAR(64) UNIQUE,
                        api_key_expires_at TIMESTAMP,
                        permissions JSON,
                        metadata JSON,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        deleted_at TIMESTAMP
                    )
                """))
                
                # Create indexes for users table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_status_role ON users(status, role)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login)"))
                
                # Create user_sessions table
                logger.info("Creating user_sessions table...")
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id VARCHAR(36) PRIMARY KEY,
                        user_id VARCHAR(36) NOT NULL REFERENCES users(user_id),
                        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        last_accessed TIMESTAMP NOT NULL DEFAULT NOW(),
                        expires_at TIMESTAMP NOT NULL,
                        ip_address VARCHAR(45),
                        user_agent TEXT,
                        is_api_session BOOLEAN DEFAULT FALSE,
                        revoked BOOLEAN DEFAULT FALSE,
                        revoked_at TIMESTAMP,
                        revoke_reason VARCHAR(255),
                        metadata JSON
                    )
                """))
                
                # Create indexes for sessions
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at)"))
                
                # Create refresh_tokens table
                logger.info("Creating refresh_tokens table...")
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS refresh_tokens (
                        jti VARCHAR(36) PRIMARY KEY,
                        user_id VARCHAR(36) NOT NULL REFERENCES users(user_id),
                        token_hash VARCHAR(64) UNIQUE NOT NULL,
                        issued_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        expires_at TIMESTAMP NOT NULL,
                        revoked BOOLEAN DEFAULT FALSE,
                        revoked_at TIMESTAMP,
                        revoke_reason VARCHAR(255),
                        ip_address VARCHAR(45),
                        user_agent TEXT,
                        family_id VARCHAR(36)
                    )
                """))
                
                # Create indexes for refresh tokens
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_revoked ON refresh_tokens(revoked)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires ON refresh_tokens(expires_at)"))
                
                # Create user_audit_logs table
                logger.info("Creating user_audit_logs table...")
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_audit_logs (
                        audit_id VARCHAR(36) PRIMARY KEY,
                        user_id VARCHAR(36) REFERENCES users(user_id),
                        event_type VARCHAR(50) NOT NULL,
                        event_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                        session_id VARCHAR(36),
                        ip_address VARCHAR(45),
                        user_agent TEXT,
                        details JSON,
                        old_values JSON,
                        new_values JSON,
                        severity VARCHAR(20)
                    )
                """))
                
                # Create indexes for audit logs
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_user_id ON user_audit_logs(user_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_event_type ON user_audit_logs(event_type)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON user_audit_logs(event_timestamp)"))
                
                # Create login_attempts table
                logger.info("Creating login_attempts table...")
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS login_attempts (
                        attempt_id VARCHAR(36) PRIMARY KEY,
                        username VARCHAR(50),
                        ip_address VARCHAR(45),
                        attempt_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                        success BOOLEAN NOT NULL,
                        failure_reason VARCHAR(100),
                        user_agent TEXT
                    )
                """))
                
                # Create indexes for login attempts
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_attempts_username ON login_attempts(username)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_attempts_ip_address ON login_attempts(ip_address)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_attempts_timestamp ON login_attempts(attempt_timestamp)"))
                
                # Commit transaction
                trans.commit()
                logger.info("✅ All user management tables created successfully!")
                
                # Verify tables were created
                result = conn.execute(text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('users', 'user_sessions', 'refresh_tokens', 'user_audit_logs', 'login_attempts')
                    ORDER BY table_name
                """))
                
                tables = [row[0] for row in result]
                logger.info(f"Created tables: {', '.join(tables)}")
                
                return True
                
            except Exception as e:
                trans.rollback()
                logger.error(f"Failed to create tables: {e}")
                return False
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
    finally:
        engine.dispose()


def create_default_admin_user():
    """Create a default admin user for initial access."""
    db_url = get_database_url()
    
    try:
        engine = create_engine(db_url, echo=False)
        
        with engine.connect() as conn:
            # Check if any users exist
            result = conn.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.fetchone()[0]
            
            if user_count > 0:
                logger.info("Users already exist, skipping default admin creation")
                return
            
            # Create default admin user
            import uuid
            import hashlib
            import secrets
            
            user_id = str(uuid.uuid4())
            username = "admin"
            email = "admin@trading-system.local"
            password = "TradingAdmin2024!"  # Change this in production!
            salt = secrets.token_hex(32)
            
            # Simple password hashing (in production use bcrypt)
            password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            
            conn.execute(text("""
                INSERT INTO users (
                    user_id, username, email, role, status, 
                    password_hash, salt, created_at, updated_at
                ) VALUES (
                    :user_id, :username, :email, 'super_admin', 'active',
                    :password_hash, :salt, NOW(), NOW()
                )
            """), {
                "user_id": user_id,
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "salt": salt
            })
            
            logger.info(f"✅ Default admin user created:")
            logger.info(f"   Username: {username}")
            logger.info(f"   Password: {password}")
            logger.info("   ⚠️  CHANGE THIS PASSWORD IMMEDIATELY!")
            
    except Exception as e:
        logger.error(f"Failed to create default admin user: {e}")
    finally:
        engine.dispose()


def main():
    """Main function."""
    logger.info("🗃️ Creating user management tables...")
    
    if create_user_tables():
        logger.info("🎉 Database tables created successfully!")
        
        # Create default admin user
        create_default_admin_user()
        
        logger.info("✅ User management system ready!")
        return True
    else:
        logger.error("❌ Failed to create database tables!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
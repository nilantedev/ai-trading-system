#!/usr/bin/env python3
"""
Database migration runner script.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared" / "python-common"))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import ProgrammingError
import asyncpg

# Import our models to register them
from trading_common.user_models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get database URL from environment or use default."""
    return (
        os.getenv('DATABASE_URL') or
        os.getenv('POSTGRES_URL') or
        'postgresql://trading_user:trading_password@localhost:5432/trading_db'
    )


def create_database_if_not_exists():
    """Create database if it doesn't exist."""
    db_url = get_database_url()
    
    # Parse the URL to get database name
    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    db_name = parsed.path.lstrip('/')
    
    # Create connection to postgres database
    postgres_url = db_url.replace(f'/{db_name}', '/postgres')
    
    try:
        engine = create_engine(postgres_url, isolation_level='AUTOCOMMIT')
        
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(text(
                "SELECT 1 FROM pg_database WHERE datname = :db_name"
            ), {"db_name": db_name})
            
            if not result.fetchone():
                logger.info(f"Creating database: {db_name}")
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                logger.info(f"Database {db_name} created successfully")
            else:
                logger.info(f"Database {db_name} already exists")
        
        engine.dispose()
        
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        # Continue anyway - the database might already exist


def run_migrations():
    """Run database migrations."""
    try:
        # First ensure database exists
        create_database_if_not_exists()
        
        # Get database URL
        db_url = get_database_url()
        logger.info(f"Running migrations on database: {db_url.split('@')[1] if '@' in db_url else db_url}")
        
        # Create engine
        engine = create_engine(db_url, echo=False)
        
        # Check connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"Connected to PostgreSQL: {version[:50]}...")
        
        # Create all tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(engine)
        
        # Verify tables were created
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        expected_tables = [
            'users', 'user_sessions', 'refresh_tokens', 'user_audit_logs',
            'user_permissions', 'role_permissions', 'login_attempts', 'api_keys'
        ]
        
        created_tables = [table for table in expected_tables if table in tables]
        logger.info(f"Created tables: {', '.join(created_tables)}")
        
        if len(created_tables) == len(expected_tables):
            logger.info("✅ All user management tables created successfully!")
        else:
            missing = set(expected_tables) - set(created_tables)
            logger.warning(f"⚠️ Missing tables: {', '.join(missing)}")
        
        # Create default permissions
        create_default_permissions(engine)
        
        engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_default_permissions(engine):
    """Create default permissions and role assignments."""
    try:
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            
            try:
                # Check if permissions already exist
                result = conn.execute(text("SELECT COUNT(*) FROM user_permissions"))
                if result.fetchone()[0] > 0:
                    logger.info("Permissions already exist, skipping default creation")
                    trans.rollback()
                    return
                
                # Insert default permissions
                permissions = [
                    ('perm_001', 'user:create', 'Create new users', 'users', 'create'),
                    ('perm_002', 'user:read', 'View user information', 'users', 'read'),
                    ('perm_003', 'user:update', 'Update user information', 'users', 'update'),
                    ('perm_004', 'user:delete', 'Delete users', 'users', 'delete'),
                    ('perm_005', 'system:admin', 'System administration', 'system', 'admin'),
                    ('perm_006', 'system:config', 'System configuration', 'system', 'config'),
                    ('perm_007', 'system:metrics', 'View system metrics', 'system', 'metrics'),
                    ('perm_008', 'trading:execute', 'Execute trades', 'trading', 'execute'),
                    ('perm_009', 'trading:read', 'View trading data', 'trading', 'read'),
                    ('perm_010', 'trading:manage', 'Manage trading strategies', 'trading', 'manage'),
                    ('perm_011', 'data:read', 'Read market data', 'data', 'read'),
                    ('perm_012', 'data:write', 'Write market data', 'data', 'write'),
                    ('perm_013', 'models:deploy', 'Deploy ML models', 'models', 'deploy'),
                    ('perm_014', 'models:manage', 'Manage ML models', 'models', 'manage'),
                    ('perm_015', 'models:read', 'View ML models', 'models', 'read'),
                ]
                
                for perm in permissions:
                    conn.execute(text("""
                        INSERT INTO user_permissions 
                        (permission_id, name, description, resource, action, created_at)
                        VALUES (:id, :name, :desc, :resource, :action, NOW())
                    """), {
                        "id": perm[0],
                        "name": perm[1], 
                        "desc": perm[2],
                        "resource": perm[3],
                        "action": perm[4]
                    })
                
                # Assign permissions to roles
                role_permissions = [
                    # SUPER_ADMIN gets all permissions
                    ('super_admin', ['perm_001', 'perm_002', 'perm_003', 'perm_004', 'perm_005', 
                                   'perm_006', 'perm_007', 'perm_008', 'perm_009', 'perm_010', 
                                   'perm_011', 'perm_012', 'perm_013', 'perm_014', 'perm_015']),
                    # ADMIN gets most permissions
                    ('admin', ['perm_001', 'perm_002', 'perm_003', 'perm_007', 'perm_009', 
                             'perm_010', 'perm_011', 'perm_012', 'perm_014', 'perm_015']),
                    # TRADER gets trading and data permissions
                    ('trader', ['perm_008', 'perm_009', 'perm_011', 'perm_015']),
                    # ANALYST gets read permissions
                    ('analyst', ['perm_011', 'perm_015', 'perm_007']),
                    # API_USER gets programmatic access
                    ('api_user', ['perm_008', 'perm_011', 'perm_015']),
                    # VIEWER gets minimal read access
                    ('viewer', ['perm_011']),
                ]
                
                for role, perms in role_permissions:
                    for perm_id in perms:
                        conn.execute(text("""
                            INSERT INTO role_permissions (role, permission_id, granted_at)
                            VALUES (:role, :perm_id, NOW())
                        """), {"role": role, "perm_id": perm_id})
                
                trans.commit()
                logger.info("✅ Default permissions and role assignments created")
                
            except Exception as e:
                trans.rollback()
                raise e
                
    except Exception as e:
        logger.error(f"Failed to create default permissions: {e}")


async def test_user_creation():
    """Test creating a user after migration."""
    try:
        # Import user management after tables are created
        from trading_common.user_management import get_user_manager, UserRole
        import secrets
        
        user_manager = get_user_manager()
        
        # Test creating a default admin user
        try:
            admin_user = await user_manager.create_user(
                username="admin",
                email="admin@trading-system.local", 
                password=secrets.token_urlsafe(16),
                role=UserRole.SUPER_ADMIN
            )
            logger.info(f"✅ Test admin user created: {admin_user.username}")
            
        except Exception as e:
            if "already exists" in str(e):
                logger.info("Admin user already exists")
            else:
                logger.error(f"Failed to create test user: {e}")
                
    except ImportError as e:
        logger.warning(f"Could not test user creation: {e}")
    except Exception as e:
        logger.error(f"Error testing user creation: {e}")


def main():
    """Main migration runner."""
    logger.info("🗃️ Starting database migration process...")
    
    if run_migrations():
        logger.info("🎉 Database migrations completed successfully!")
        
        # Test user creation
        try:
            asyncio.run(test_user_creation())
        except Exception as e:
            logger.warning(f"User creation test failed: {e}")
            
        return True
    else:
        logger.error("❌ Database migrations failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
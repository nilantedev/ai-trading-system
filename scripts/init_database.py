#!/usr/bin/env python3
"""
Database initialization script for AI Trading System.
Creates database tables and default admin user.
"""

import asyncio
import os
import sys
import secrets
from pathlib import Path

# Add paths for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "shared" / "python-common"))

from trading_common.user_management import get_user_manager, UserRole
from trading_common.user_models import Users
from trading_common import get_settings
from shared.logging_config import get_logger

logger = get_logger(__name__)


async def create_tables():
    """Create database tables using Alembic or direct SQLAlchemy."""
    try:
        from sqlalchemy import MetaData
        from sqlalchemy.ext.asyncio import create_async_engine
        from trading_common.user_models import Base
        
        settings = get_settings()
        
        # Build database URL
        db_url = f"postgresql+asyncpg://{settings.database.postgres_user}:{settings.database.postgres_password}@{settings.database.postgres_host}:{settings.database.postgres_port}/{settings.database.postgres_db}"
        
        engine = create_async_engine(db_url, echo=False)
        
        logger.info("Creating database tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        await engine.dispose()
        
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        # Don't raise - let the user manager handle it gracefully


async def create_default_admin():
    """Create default admin user if not exists."""
    try:
        user_manager = get_user_manager()
        settings = get_settings()
        
        # Check if admin user already exists
        existing_admin = await user_manager._get_user_by_username("admin")
        if existing_admin:
            logger.info("Default admin user already exists, skipping creation")
            return existing_admin
        
        # Generate secure password
        admin_password = os.getenv("ADMIN_PASSWORD") or secrets.token_urlsafe(16)
        admin_email = os.getenv("ADMIN_EMAIL", "admin@trading-system.local")
        
        # Create admin user
        admin_user = await user_manager.create_user(
            username="admin",
            email=admin_email,
            password=admin_password,
            role=UserRole.SUPER_ADMIN,
            created_by="system_init"
        )
        
        # Save password to secure file instead of logging
        import os
        password_file = os.path.join(os.path.dirname(__file__), ".admin_credentials")
        with open(password_file, 'w') as f:
            f.write(f"Username: admin\n")
            f.write(f"Email: {admin_email}\n")
            f.write(f"Password: {admin_password}\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write("\n⚠️  CHANGE THIS PASSWORD IMMEDIATELY AFTER FIRST LOGIN!\n")
        
        # Set restrictive permissions
        os.chmod(password_file, 0o600)
        
        logger.info(f"Created default admin user:")
        logger.info(f"  Username: admin")
        logger.info(f"  Email: {admin_email}")
        logger.info(f"  Credentials saved to: {password_file}")
        logger.warning("IMPORTANT: Change the password immediately after first login!")
        
        return admin_user
        
    except Exception as e:
        logger.error(f"Failed to create default admin user: {e}")
        raise


async def initialize_database():
    """Initialize database with tables and default data."""
    logger.info("Starting database initialization...")
    
    try:
        # Create tables first
        await create_tables()
        
        # Create default admin user
        admin_user = await create_default_admin()
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def main():
    """Main initialization function."""
    print("AI Trading System - Database Initialization")
    print("=" * 50)
    
    # Check environment variables
    settings = get_settings()
    if not settings.database.postgres_password:
        print("ERROR: POSTGRES_PASSWORD environment variable is required")
        print("Please set the database password in your .env file")
        return 1
    
    # Run initialization
    try:
        success = asyncio.run(initialize_database())
        if success:
            print("\nDatabase initialization completed successfully!")
            print("You can now start the API server.")
            return 0
        else:
            print("\nDatabase initialization failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\nInitialization cancelled by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
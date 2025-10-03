#!/usr/bin/env python3
"""Database migration system for AI Trading System."""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add the shared library to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "python-common"))

from trading_common.database import get_database_manager, QuestDBOperations
from trading_common.logging import get_logger

logger = get_logger(__name__)


class MigrationManager:
    """Manage database migrations for the trading system."""
    
    def __init__(self):
        self.db_manager = None
        self.migrations_dir = Path(__file__).parent.parent / "shared" / "migrations"
        self.schemas_dir = Path(__file__).parent.parent / "shared" / "schemas"
        
    async def initialize(self):
        """Initialize database connections."""
        self.db_manager = await get_database_manager()
        logger.info("Migration manager initialized")
    
    async def create_migration_table(self):
        """Create migration tracking table."""
        query = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id STRING,
            migration_name STRING,
            executed_at TIMESTAMP,
            execution_time_ms LONG,
            success BOOLEAN,
            error_message STRING
        ) TIMESTAMP(executed_at) PARTITION BY DAY
        """
        
        try:
            async with self.db_manager.get_questdb() as conn:
                await conn.execute(query)
                logger.info("Created schema_migrations table")
        except Exception as e:
            logger.error(f"Failed to create migration table: {e}")
            raise
    
    async def get_executed_migrations(self) -> List[str]:
        """Get list of already executed migrations."""
        query = "SELECT migration_name FROM schema_migrations WHERE success = true ORDER BY executed_at"
        
        try:
            async with self.db_manager.get_questdb() as conn:
                rows = await conn.fetch(query)
                return [row['migration_name'] for row in rows]
        except Exception as e:
            # Migration table might not exist yet
            logger.warning(f"Could not get executed migrations: {e}")
            return []
    
    async def execute_migration(self, migration_name: str, sql_content: str) -> bool:
        """Execute a single migration."""
        migration_id = f"migration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.utcnow()
        
        try:
            # Split SQL content into individual statements
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            async with self.db_manager.get_questdb() as conn:
                # Execute migration statements
                for statement in statements:
                    if statement:
                        await conn.execute(statement)
                
                # Record successful migration
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                record_query = """
                INSERT INTO schema_migrations 
                (id, migration_name, executed_at, execution_time_ms, success, error_message)
                VALUES ($1, $2, $3, $4, $5, $6)
                """
                
                await conn.execute(record_query, 
                    migration_id, migration_name, start_time, 
                    execution_time, True, None
                )
                
                logger.info(f"Successfully executed migration: {migration_name}")
                return True
                
        except Exception as e:
            # Record failed migration
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            error_msg = str(e)
            
            try:
                async with self.db_manager.get_questdb() as conn:
                    record_query = """
                    INSERT INTO schema_migrations 
                    (id, migration_name, executed_at, execution_time_ms, success, error_message)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """
                    
                    await conn.execute(record_query,
                        migration_id, migration_name, start_time,
                        execution_time, False, error_msg
                    )
            except:
                pass  # If we can't record the failure, just log it
            
            logger.error(f"Migration {migration_name} failed: {e}")
            return False
    
    async def run_migrations(self, specific_migration: Optional[str] = None):
        """Run all pending migrations or a specific migration."""
        await self.create_migration_table()
        
        # Get executed migrations
        executed = await self.get_executed_migrations()
        logger.info(f"Found {len(executed)} executed migrations")
        
        # Find migration files
        migration_files = []
        
        # Check schemas directory for .sql files
        if self.schemas_dir.exists():
            for file_path in sorted(self.schemas_dir.glob("*.sql")):
                migration_files.append(file_path)
        
        # Check migrations directory for .sql files  
        if self.migrations_dir.exists():
            for file_path in sorted(self.migrations_dir.glob("*.sql")):
                migration_files.append(file_path)
        
        if not migration_files:
            logger.warning("No migration files found")
            return
        
        # Execute migrations
        success_count = 0
        failure_count = 0
        
        for file_path in migration_files:
            migration_name = file_path.name
            
            # Skip if specific migration requested and this isn't it
            if specific_migration and migration_name != specific_migration:
                continue
            
            # Skip if already executed (unless specific migration requested)
            if migration_name in executed and not specific_migration:
                logger.info(f"Skipping already executed migration: {migration_name}")
                continue
            
            logger.info(f"Executing migration: {migration_name}")
            
            # Read migration content
            try:
                with open(file_path, 'r') as f:
                    sql_content = f.read()
            except Exception as e:
                logger.error(f"Could not read migration file {file_path}: {e}")
                failure_count += 1
                continue
            
            # Execute migration
            if await self.execute_migration(migration_name, sql_content):
                success_count += 1
            else:
                failure_count += 1
                
                # Stop on failure unless specific migration
                if not specific_migration:
                    logger.error("Stopping migrations due to failure")
                    break
        
        logger.info(f"Migration summary: {success_count} successful, {failure_count} failed")
        
        if failure_count > 0:
            sys.exit(1)
    
    async def rollback_migration(self, migration_name: str):
        """Rollback a specific migration (if rollback script exists)."""
        rollback_file = self.migrations_dir / f"rollback_{migration_name}"
        
        if not rollback_file.exists():
            logger.error(f"No rollback script found for {migration_name}")
            return False
        
        try:
            with open(rollback_file, 'r') as f:
                sql_content = f.read()
            
            # Execute rollback
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            async with self.db_manager.get_questdb() as conn:
                for statement in statements:
                    if statement:
                        await conn.execute(statement)
                
                # Update migration record
                update_query = """
                UPDATE schema_migrations 
                SET success = false, error_message = 'Rolled back manually'
                WHERE migration_name = $1
                """
                await conn.execute(update_query, migration_name)
            
            logger.info(f"Successfully rolled back migration: {migration_name}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed for {migration_name}: {e}")
            return False
    
    async def migration_status(self):
        """Show migration status."""
        try:
            # Get all migrations
            executed = []
            try:
                query = """
                SELECT migration_name, executed_at, success, execution_time_ms, error_message
                FROM schema_migrations 
                ORDER BY executed_at DESC
                LIMIT 50
                """
                
                async with self.db_manager.get_questdb() as conn:
                    rows = await conn.fetch(query)
                    executed = [dict(row) for row in rows]
                    
            except Exception:
                logger.warning("Could not fetch migration history")
            
            print("\n=== MIGRATION STATUS ===")
            print(f"Executed migrations: {len(executed)}")
            
            if executed:
                print("\nRecent migrations:")
                print("Name                          | Status    | Executed At         | Time (ms)")
                print("-" * 80)
                
                for migration in executed:
                    status = "SUCCESS" if migration['success'] else "FAILED"
                    name = migration['migration_name'][:28]
                    executed_at = migration['executed_at'].strftime('%Y-%m-%d %H:%M:%S')
                    time_ms = migration['execution_time_ms'] or 0
                    
                    print(f"{name:<30} | {status:<9} | {executed_at} | {time_ms:>8}")
                    
                    if not migration['success'] and migration['error_message']:
                        print(f"  Error: {migration['error_message']}")
            
            # Check for pending migrations
            migration_files = []
            if self.schemas_dir.exists():
                migration_files.extend(sorted(self.schemas_dir.glob("*.sql")))
            if self.migrations_dir.exists():
                migration_files.extend(sorted(self.migrations_dir.glob("*.sql")))
            
            executed_names = {m['migration_name'] for m in executed if m['success']}
            pending = [f.name for f in migration_files if f.name not in executed_names]
            
            print(f"\nPending migrations: {len(pending)}")
            if pending:
                for p in pending:
                    print(f"  - {p}")
            
            print()
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
    
    async def close(self):
        """Close database connections."""
        if self.db_manager:
            await self.db_manager.close()


async def main():
    """Main migration CLI."""
    parser = argparse.ArgumentParser(description="AI Trading System Database Migrations")
    parser.add_argument("command", choices=["migrate", "rollback", "status"], 
                       help="Migration command")
    parser.add_argument("--migration", "-m", help="Specific migration name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create migration manager
    manager = MigrationManager()
    
    try:
        await manager.initialize()
        
        if args.command == "migrate":
            await manager.run_migrations(args.migration)
        elif args.command == "rollback":
            if not args.migration:
                print("Error: --migration required for rollback command")
                sys.exit(1)
            await manager.rollback_migration(args.migration)
        elif args.command == "status":
            await manager.migration_status()
            
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
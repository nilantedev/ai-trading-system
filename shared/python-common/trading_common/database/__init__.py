"""Database utilities and connection management."""

from .redis_client import get_redis_client, RedisClient
# from .questdb_client import get_questdb_client, QuestDBClient
# from .postgres_client import get_postgres_client

def get_database():
    """Get default database connection (Redis)."""
    return get_redis_client()

def get_database_manager():
    """Get database manager (alias for compatibility)."""
    return get_database()

__all__ = [
    "get_redis_client",
    "RedisClient",
    "get_database",
    "get_database_manager",
    # "get_questdb_client",
    # "QuestDBClient",
    # "get_postgres_client",
]
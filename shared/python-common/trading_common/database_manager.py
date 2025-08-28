"""Database connection and utility functions for trading system."""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from .config import get_settings, DatabaseSettings
from .models import MarketData, TradingSignal, PortfolioSnapshot
from .logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Centralized database connection management."""
    
    def __init__(self, settings: Optional[DatabaseSettings] = None):
        self.settings = settings or get_settings().database
        self._redis_pool: Optional[redis.ConnectionPool] = None
        self._questdb_pool: Optional[asyncpg.Pool] = None
        self._postgres_engine = None
        self._postgres_session_factory = None
        
    async def initialize(self):
        """Initialize all database connections."""
        await self._init_redis()
        await self._init_questdb()
        await self._init_postgres()
        logger.info("Database connections initialized")
    
    async def close(self):
        """Close all database connections."""
        if self._redis_pool:
            await self._redis_pool.disconnect()
        if self._questdb_pool:
            await self._questdb_pool.close()
        if self._postgres_engine:
            await self._postgres_engine.dispose()
        logger.info("Database connections closed")
    
    async def _init_redis(self):
        """Initialize Redis connection pool."""
        try:
            self._redis_pool = redis.ConnectionPool.from_url(
                self.settings.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                decode_responses=True
            )
            # Test connection
            async with redis.Redis(connection_pool=self._redis_pool) as r:
                await r.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def _init_questdb(self):
        """Initialize QuestDB connection pool."""
        try:
            questdb_url = f"postgresql://{self.settings.questdb_user or 'questdb'}:{self.settings.questdb_password or 'quest'}@{self.settings.questdb_host}:{self.settings.questdb_port}/qdb"
            
            self._questdb_pool = await asyncpg.create_pool(
                questdb_url,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            # Test connection
            async with self._questdb_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.info("QuestDB connection established")
        except Exception as e:
            logger.warning(f"QuestDB connection failed (may still be starting): {e}")
            self._questdb_pool = None
    
    async def _init_postgres(self):
        """Initialize PostgreSQL connection (if configured)."""
        if not self.settings.postgres_url:
            logger.info("PostgreSQL not configured, skipping")
            return
            
        try:
            self._postgres_engine = create_async_engine(
                self.settings.postgres_url,
                echo=False,
                pool_size=5,
                max_overflow=10
            )
            self._postgres_session_factory = sessionmaker(
                self._postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            # Test connection
            async with self._postgres_engine.begin() as conn:
                await conn.execute("SELECT 1")
            logger.info("PostgreSQL connection established")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    @asynccontextmanager
    async def get_redis(self):
        """Get Redis connection."""
        if not self._redis_pool:
            raise RuntimeError("Redis not initialized")
        
        async with redis.Redis(connection_pool=self._redis_pool) as r:
            yield r
    
    @asynccontextmanager
    async def get_questdb(self):
        """Get QuestDB connection."""
        if not self._questdb_pool:
            raise RuntimeError("QuestDB not available")
        
        async with self._questdb_pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager
    async def get_postgres(self):
        """Get PostgreSQL session."""
        if not self._postgres_session_factory:
            raise RuntimeError("PostgreSQL not initialized")
        
        async with self._postgres_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


class QuestDBOperations:
    """QuestDB-specific operations for time-series data."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def create_tables(self):
        """Create all trading-related tables in QuestDB."""
        with open('shared/schemas/market_data.sql', 'r') as f:
            schema_sql = f.read()
        
        # Split into individual statements
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        try:
            async with self.db_manager.get_questdb() as conn:
                for statement in statements:
                    if statement:
                        await conn.execute(statement)
            logger.info(f"Created {len(statements)} database objects")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def insert_market_data(self, data: Union[MarketData, List[MarketData]]) -> int:
        """Insert market data records."""
        if isinstance(data, MarketData):
            data = [data]
        
        query = """
        INSERT INTO market_data (
            symbol, timestamp, open, high, low, close, volume, 
            vwap, trade_count, data_source
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        records = []
        for item in data:
            records.append((
                item.symbol, item.timestamp, item.open, item.high, item.low,
                item.close, item.volume, item.vwap, item.trade_count, item.data_source
            ))
        
        try:
            async with self.db_manager.get_questdb() as conn:
                await conn.executemany(query, records)
            logger.debug(f"Inserted {len(records)} market data records")
            return len(records)
        except Exception as e:
            logger.error(f"Failed to insert market data: {e}")
            raise
    
    async def insert_trading_signal(self, signal: TradingSignal) -> bool:
        """Insert a trading signal."""
        query = """
        INSERT INTO trading_signals (
            id, timestamp, symbol, signal_type, confidence, target_price,
            stop_loss, take_profit, timeframe, strategy_name, agent_id,
            reasoning, market_conditions, risk_assessment, expires_at, status
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        """
        
        try:
            async with self.db_manager.get_questdb() as conn:
                await conn.execute(query, 
                    signal.id, signal.timestamp, signal.symbol, signal.signal_type.value,
                    signal.confidence, signal.target_price, signal.stop_loss, signal.take_profit,
                    signal.timeframe.value, signal.strategy_name, signal.agent_id,
                    signal.reasoning, 
                    signal.market_conditions or '{}',
                    signal.risk_assessment or '{}',
                    signal.expires_at, signal.status.value
                )
            logger.debug(f"Inserted trading signal {signal.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert trading signal: {e}")
            raise
    
    async def get_latest_market_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get latest market data for a symbol."""
        query = """
        SELECT * FROM market_data 
        WHERE symbol = $1 
        ORDER BY timestamp DESC 
        LIMIT $2
        """
        
        try:
            async with self.db_manager.get_questdb() as conn:
                rows = await conn.fetch(query, symbol, limit)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return []
    
    async def get_active_signals(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active trading signals."""
        if symbol:
            query = "SELECT * FROM trading_signals WHERE symbol = $1 AND status = 'active'"
            params = [symbol]
        else:
            query = "SELECT * FROM trading_signals WHERE status = 'active'"
            params = []
        
        try:
            async with self.db_manager.get_questdb() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get active signals: {e}")
            return []


class RedisOperations:
    """Redis operations for caching and real-time data."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def cache_market_data(self, symbol: str, data: MarketData, ttl: int = 300):
        """Cache latest market data."""
        key = f"market_data:{symbol}:latest"
        value = data.json()
        
        async with self.db_manager.get_redis() as r:
            await r.setex(key, ttl, value)
    
    async def get_cached_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get cached market data."""
        key = f"market_data:{symbol}:latest"
        
        async with self.db_manager.get_redis() as r:
            data = await r.get(key)
            if data:
                return MarketData.parse_raw(data)
            return None
    
    async def cache_portfolio_snapshot(self, portfolio_id: str, snapshot: PortfolioSnapshot, ttl: int = 60):
        """Cache portfolio snapshot."""
        key = f"portfolio:{portfolio_id}:snapshot"
        value = snapshot.json()
        
        async with self.db_manager.get_redis() as r:
            await r.setex(key, ttl, value)
    
    async def get_cached_portfolio(self, portfolio_id: str) -> Optional[PortfolioSnapshot]:
        """Get cached portfolio snapshot."""
        key = f"portfolio:{portfolio_id}:snapshot"
        
        async with self.db_manager.get_redis() as r:
            data = await r.get(key)
            if data:
                return PortfolioSnapshot.parse_raw(data)
            return None
    
    async def set_system_status(self, component: str, status: Dict[str, Any], ttl: int = 300):
        """Set system component status."""
        key = f"system_status:{component}"
        value = str(status)
        
        async with self.db_manager.get_redis() as r:
            await r.setex(key, ttl, value)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        async with self.db_manager.get_redis() as r:
            keys = await r.keys("system_status:*")
            health = {}
            for key in keys:
                component = key.split(":", 1)[1]
                status = await r.get(key)
                if status:
                    try:
                        import json
                        health[component] = json.loads(status) if isinstance(status, str) else status
                    except (json.JSONDecodeError, TypeError):
                        health[component] = {"status": "unknown", "raw": str(status)}
            return health


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """Get or create global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    return _db_manager

async def close_database_connections():
    """Close global database connections."""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None
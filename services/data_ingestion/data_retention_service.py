#!/usr/bin/env python3
"""
Advanced Data Retention and Compaction Service
Intelligently manages historical data storage with compression and smart retention policies.

Enhancements:
- Adds QuestDB HTTP execution path for concrete table-specific retention rules
    (option_daily, news_items, social_signals) while keeping legacy generic
    market_data policies in place for future alignment.
"""

import asyncio
import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics
import zlib
import pickle
import aiohttp
from prometheus_client import Counter, Gauge

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.database import get_database_manager

logger = get_logger(__name__)
settings = get_settings()


class DataGranularity(Enum):
    """Data granularity levels for storage optimization."""
    TICK = "tick"           # Every tick (real-time)
    SECOND = "1s"           # 1-second bars
    MINUTE = "1m"           # 1-minute bars
    FIVE_MINUTE = "5m"      # 5-minute bars
    FIFTEEN_MINUTE = "15m"  # 15-minute bars
    HOUR = "1h"             # Hourly bars
    DAILY = "1d"            # Daily bars
    WEEKLY = "1w"           # Weekly bars
    MONTHLY = "1M"          # Monthly bars


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""
    granularity: DataGranularity
    retention_days: int
    compression_after_days: Optional[int] = None
    aggregation_target: Optional[DataGranularity] = None
    priority_symbols: List[str] = None  # Symbols to keep longer


class DataRetentionService:
    """
    Manages data retention, compression, and intelligent storage optimization.
    
    Key Features:
    - Tiered retention policies based on data age and value
    - Intelligent compression for older data
    - Automatic aggregation to reduce storage
    - Priority retention for important symbols
    - Performance-aware data compaction
    """
    
    def __init__(self):
        self.cache = None
        self.db_manager = None
        self._http: Optional[aiohttp.ClientSession] = None
        # QuestDB HTTP endpoint for SQL execution (e.g., http://trading-questdb:9000/exec)
        host = os.getenv('QUESTDB_HOST', 'trading-questdb')
        http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
        self.questdb_http_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
        # Retention tunables (env overrides)
        self.opt_zero_vol_days = int(os.getenv('OPTION_ZERO_VOL_DAYS', '540'))
        self.opt_expiry_past_days = int(os.getenv('OPTION_EXPIRY_PAST_DAYS', '120'))
        # Hard retention requirement: keep news for 5 years (1825 days) unless overridden.
        # Backfill horizon may be smaller initially (e.g. 2y for options) but retention
        # ensures forward accumulation up to 5y before pruning.
        self.news_retention_days = int(os.getenv('NEWS_RETENTION_DAYS', '1825'))
        self.news_lowrel_days = int(os.getenv('NEWS_LOWRELEVANCE_DAYS', '30'))
        self.news_lowrel_sent = float(os.getenv('NEWS_LOWRELEVANCE_SENTIMENT', '0.05'))
        self.news_lowrel_rel = float(os.getenv('NEWS_LOWRELEVANCE_THRESHOLD', '0.2'))
        # Social signals retention aligned to 5 years (1825 days) to match requirement.
        self.social_retention_days = int(os.getenv('SOCIAL_RETENTION_DAYS', '1825'))
        self.social_low_days = int(os.getenv('SOCIAL_LOWENGAGEMENT_DAYS', '14'))
        self.social_low_eng = float(os.getenv('SOCIAL_LOWENGAGEMENT_THRESHOLD', '0.05'))
        self.social_low_inf = float(os.getenv('SOCIAL_LOWINFLUENCE_THRESHOLD', '0.1'))
        self.social_lowvalue_sent = float(os.getenv('SOCIAL_LOWVALUE_SENTIMENT', '0.05'))
        self.retention_window_days = int(os.getenv('RETENTION_WINDOW_DAYS', '30'))
        self.retention_max_windows = int(os.getenv('RETENTION_MAX_WINDOWS_PER_TABLE', '6'))

        # Prometheus metrics (shared registry via FastAPI process)
        try:
            self._retention_runs = Counter('retention_runs_total', 'Total retention runs executed')
            self._retention_deletes = Counter('retention_deletions_total', 'Total rows deleted by retention', ['table', 'reason'])
            self._retention_last_run_ts = Gauge('retention_last_run_timestamp', 'Last retention run timestamp (unix seconds)')
        except Exception:
            self._retention_runs = None
            self._retention_deletes = None
            self._retention_last_run_ts = None
        
        # Define retention policies (can be customized)
        self.retention_policies = [
            # Keep tick data for 1 day, then aggregate to 1-second
            RetentionPolicy(
                granularity=DataGranularity.TICK,
                retention_days=1,
                aggregation_target=DataGranularity.SECOND
            ),
            # Keep 1-second data for 7 days, then aggregate to 1-minute
            RetentionPolicy(
                granularity=DataGranularity.SECOND,
                retention_days=7,
                aggregation_target=DataGranularity.MINUTE
            ),
            # Keep 1-minute data for 30 days, compress after 14 days
            RetentionPolicy(
                granularity=DataGranularity.MINUTE,
                retention_days=30,
                compression_after_days=14,
                aggregation_target=DataGranularity.FIVE_MINUTE
            ),
            # Keep 5-minute data for 90 days
            RetentionPolicy(
                granularity=DataGranularity.FIVE_MINUTE,
                retention_days=90,
                compression_after_days=30,
                aggregation_target=DataGranularity.FIFTEEN_MINUTE
            ),
            # Keep 15-minute data for 180 days
            RetentionPolicy(
                granularity=DataGranularity.FIFTEEN_MINUTE,
                retention_days=180,
                compression_after_days=60,
                aggregation_target=DataGranularity.HOUR
            ),
            # Keep hourly data for 2 years
            RetentionPolicy(
                granularity=DataGranularity.HOUR,
                retention_days=730,
                compression_after_days=180,
                aggregation_target=DataGranularity.DAILY
            ),
            # Keep daily data forever (compressed after 1 year)
            RetentionPolicy(
                granularity=DataGranularity.DAILY,
                retention_days=-1,  # Never delete
                compression_after_days=365
            ),
        ]
        
        # Priority symbols get 2x retention period
        self.priority_symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN"]
        
        # Compression settings
        self.compression_level = 6  # zlib compression level (1-9)
        self.min_compression_ratio = 0.8  # Don't compress if ratio > 80%
        
        # Performance tracking
        self.stats = {
            'data_points_processed': 0,
            'data_points_compressed': 0,
            'data_points_deleted': 0,
            'storage_saved_mb': 0.0,
            'last_cleanup': None,
            # Table-specific deletion counters (aligned to actual table names)
            'table_deletes': {
                'options_data': 0,
                'news_items': 0,
                'social_events': 0,
                'social_signals': 0,
                'market_data': 0,
            }
        }
    
    async def start(self):
        """Initialize service connections."""
        logger.info("Starting Data Retention Service")
        self.cache = get_trading_cache()
        self.db_manager = get_database_manager()
        # Create HTTP session for QuestDB REST if not already present
        try:
            if self._http is None:
                self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to create HTTP session for QuestDB: {e}")
        
        # Start background cleanup task
        asyncio.create_task(self._periodic_cleanup())
        
        logger.info("Data Retention Service started")
    
    async def stop(self):
        """Cleanup service connections."""
        logger.info("Data Retention Service stopped")
        try:
            if self._http:
                await self._http.close()
        except Exception:  # noqa: BLE001
            pass
    
    async def _periodic_cleanup(self):
        """Run periodic data cleanup and compaction."""
        while True:
            try:
                # Run cleanup at 2 AM daily (low activity time)
                now = datetime.utcnow()
                next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                
                wait_seconds = (next_run - now).total_seconds()
                logger.info(f"Next data cleanup scheduled in {wait_seconds/3600:.1f} hours")
                
                await asyncio.sleep(wait_seconds)
                
                # Run cleanup
                await self.run_data_cleanup()
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def run_data_cleanup(self):
        """
        Execute data cleanup and compaction process.
        This is the main retention enforcement method.
        """
        logger.info("Starting data cleanup and compaction")
        start_time = datetime.utcnow()
        
        try:
            # Process each retention policy (legacy generic model)
            for policy in self.retention_policies:
                await self._process_retention_policy(policy)
            
            # Apply concrete table-specific policies via QuestDB HTTP (best-effort)
            await self._apply_table_specific_policies()

            # Compact fragmented storage
            await self._compact_storage()
            
            # Update statistics
            self.stats['last_cleanup'] = start_time
            if self._retention_last_run_ts:
                try:
                    self._retention_last_run_ts.set(datetime.utcnow().timestamp())
                except Exception:
                    pass
            
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                f"Data cleanup completed in {elapsed:.2f}s - "
                f"Compressed: {self.stats['data_points_compressed']:,}, "
                f"Deleted: {self.stats['data_points_deleted']:,}, "
                f"Saved: {self.stats['storage_saved_mb']:.2f} MB"
            )
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
        finally:
            if self._retention_runs:
                try:
                    self._retention_runs.inc()
                except Exception:
                    pass
    
    async def _process_retention_policy(self, policy: RetentionPolicy):
        """Process a single retention policy."""
        cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_days)
        
        if policy.retention_days == -1:
            # Never delete this granularity, only compress
            if policy.compression_after_days:
                compress_date = datetime.utcnow() - timedelta(days=policy.compression_after_days)
                await self._compress_data(policy.granularity, compress_date)
        else:
            # Delete old data
            deleted = await self._delete_old_data(policy.granularity, cutoff_date)
            self.stats['data_points_deleted'] += deleted
            
            # Compress if configured
            if policy.compression_after_days:
                compress_date = datetime.utcnow() - timedelta(days=policy.compression_after_days)
                await self._compress_data(policy.granularity, compress_date)
            
            # Aggregate if configured
            if policy.aggregation_target:
                await self._aggregate_data(
                    policy.granularity,
                    policy.aggregation_target,
                    cutoff_date
                )
    
    async def _delete_old_data(self, granularity: DataGranularity, cutoff_date: datetime) -> int:
        """Delete data older than cutoff date."""
        try:
            # Use QuestDB's DELETE functionality
            query = f"""
                DELETE FROM market_data 
                WHERE timestamp < '{cutoff_date.isoformat()}' 
                AND timeframe = '{granularity.value}'
            """
            
            # For priority symbols, extend retention
            if self.priority_symbols:
                priority_cutoff = cutoff_date - timedelta(days=cutoff_date.day)  # Double retention
                symbols_list = ','.join([f"'{s}'" for s in self.priority_symbols])
                query += f" AND symbol NOT IN ({symbols_list})"
            
            # Execute deletion (would use actual database connection)
            logger.debug(f"Deleting {granularity.value} data older than {cutoff_date}")
            
            # Return count of deleted records (mock for now)
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting old data: {e}")
            return 0
    
    async def _compress_data(self, granularity: DataGranularity, compress_before: datetime):
        """Compress historical data to save storage."""
        try:
            # Query uncompressed data
            query = f"""
                SELECT * FROM market_data 
                WHERE timestamp < '{compress_before.isoformat()}' 
                AND timeframe = '{granularity.value}'
                AND compressed = false
                LIMIT 10000
            """
            
            # Fetch data to compress (mock implementation)
            # In reality, would batch process the data
            
            logger.debug(f"Compressing {granularity.value} data older than {compress_before}")
            
            # Compression logic would:
            # 1. Fetch batch of uncompressed data
            # 2. Compress using zlib
            # 3. Store compressed data
            # 4. Mark original as compressed or delete
            
        except Exception as e:
            logger.error(f"Error compressing data: {e}")
    
    async def _aggregate_data(self, source_granularity: DataGranularity, 
                            target_granularity: DataGranularity,
                            before_date: datetime):
        """Aggregate fine-grained data to coarser granularity."""
        try:
            # Aggregation logic
            # Convert 1-minute bars to 5-minute bars, etc.
            
            logger.debug(
                f"Aggregating {source_granularity.value} to {target_granularity.value} "
                f"for data before {before_date}"
            )
            
            # Example aggregation query (for QuestDB)
            query = f"""
                INSERT INTO market_data
                SELECT 
                    symbol,
                    date_trunc('{target_granularity.value}', timestamp) as timestamp,
                    first(open) as open,
                    max(high) as high,
                    min(low) as low,
                    last(close) as close,
                    sum(volume) as volume,
                    '{target_granularity.value}' as timeframe,
                    data_source
                FROM market_data
                WHERE timeframe = '{source_granularity.value}'
                AND timestamp < '{before_date.isoformat()}'
                GROUP BY symbol, date_trunc('{target_granularity.value}', timestamp), data_source
            """
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
    
    async def _compact_storage(self):
        """Compact fragmented storage for better performance."""
        try:
            # Run VACUUM or equivalent operation on database
            # This reclaims space from deleted rows
            
            logger.debug("Compacting storage to reclaim space")
            
            # QuestDB specific optimization
            # await self.db_manager.execute("VACUUM FULL market_data")
            
        except Exception as e:
            logger.error(f"Error compacting storage: {e}")

    async def _exec_qdb_sql(self, sql: str) -> Optional[int]:
        """Execute SQL via QuestDB HTTP /exec; returns affected row count if available.

        This is best-effort; failures are logged and None is returned.
        """
        if not self._http or not self.questdb_http_url:
            logger.debug("QuestDB HTTP path not available; skipping SQL execution")
            return None
        try:
            params = {'query': sql}
            async with self._http.get(self.questdb_http_url, params=params) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    logger.warning("QuestDB HTTP exec failed", extra={"status": resp.status, "body": txt[:300]})
                    return None
                data = await resp.json()
                # QuestDB returns {"status":"OK", ...} for DDL/DML; row count may be in 'count'
                count = None
                try:
                    count = int(data.get('count')) if isinstance(data.get('count'), (int, float)) else None
                except Exception:
                    count = None
                return count
        except Exception as e:  # noqa: BLE001
            logger.warning(f"QuestDB HTTP exec error: {e}")
            return None

    async def _qdb_query(self, sql: str) -> Optional[dict]:
        """Run a SELECT-style SQL via QuestDB HTTP /exec and return parsed JSON or None on error."""
        if not self._http or not self.questdb_http_url:
            return None
        try:
            async with self._http.get(self.questdb_http_url, params={"query": sql}) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception:
            return None

    async def _apply_table_specific_policies(self) -> None:
        """Apply concrete retention rules for option_daily, news_items, social_signals.

        These use DELETE statements against QuestDB via the HTTP endpoint.
        """
        # Enforce hard horizon windows first (equities 20y, others 5y) best-effort
        try:
            # Equities (market_data daily bars) older than 20 years (retain indefinitely otherwise)
            sql_equities = (
                "DELETE FROM market_data WHERE timestamp < dateadd('d', -" + str(20*365) + ", now())"
            )
            await self._exec_qdb_sql(sql_equities)
            # Options/social/news horizons (5 years)
            horizon_days = 5 * 365
            # Table -> designated timestamp column mapping
            horizon_tables = {
                'options_data': 'timestamp',
                'news_items': 'ts',           # news_items schema uses 'ts' as designated timestamp
                'social_events': 'timestamp',
                'social_signals': 'ts',       # social_signals schema uses 'ts'
            }
            for tbl, ts_col in horizon_tables.items():
                sql = f"DELETE FROM {tbl} WHERE {ts_col} < dateadd('d', -{horizon_days}, now())"
                await self._exec_qdb_sql(sql)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Horizon enforcement failed: {e}")

        # Options retention (refined clean-up)
        try:
            await self._delete_in_time_windows(
                table='options_data',
                ts_column='timestamp',
                where_extra="volume = 0",
                cutoff_days=self.opt_zero_vol_days,
                reason='zero_volume_old'
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"options_data retention apply failed: {e}")

        # News items retention
        try:
            # 1) Hard cutoff with windows
            await self._delete_in_time_windows(
                table='news_items',
                ts_column='ts',
                where_extra=None,
                cutoff_days=self.news_retention_days,
                reason='age_cutoff'
            )
            # 2) Low-relevance near-neutral items older than X days
            sql2 = (
                "DELETE FROM news_items "
                f"WHERE ts < dateadd('d', -{self.news_lowrel_days}, now()) "
                f"AND relevance < {self.news_lowrel_rel} AND abs(sentiment) < {self.news_lowrel_sent}"
            )
            cnt2 = await self._exec_qdb_sql(sql2)
            if isinstance(cnt2, int):
                self.stats['table_deletes']['news_items'] += max(0, cnt2)
                if self._retention_deletes:
                    try:
                        self._retention_deletes.labels(table='news_items', reason='low_value_old').inc(max(0, cnt2))
                    except Exception:
                        pass
            # 3) Value-score based prune: very low value_score (if column exists) older than lowrel_days (guarded)
            try:
                news_value_floor = float(os.getenv('NEWS_VALUE_SCORE_FLOOR','0.25'))
            except Exception:
                news_value_floor = 0.25
            sql_vs = (
                "DELETE FROM news_items "
                f"WHERE ts < dateadd('d', -{self.news_lowrel_days}, now()) "
                f"AND value_score < {news_value_floor}"
            )
            cnt_vs = await self._exec_qdb_sql(sql_vs)
            if isinstance(cnt_vs, int) and cnt_vs > 0:
                self.stats['table_deletes']['news_items'] += max(0, cnt_vs)
                if self._retention_deletes:
                    try:
                        self._retention_deletes.labels(table='news_items', reason='low_value_old').inc(max(0, cnt_vs))
                    except Exception:
                        pass
        except Exception as e:  # noqa: BLE001
            logger.debug(f"News_items retention apply failed: {e}")

        # Social signals retention
        try:
            # 1) Hard cutoff with windows
            await self._delete_in_time_windows(
                table='social_events',
                ts_column='timestamp',
                where_extra=None,
                cutoff_days=self.social_retention_days,
                reason='age_cutoff'
            )
            # 2) Also handle social_signals (written by social_media_collector)
            # Detect correct timestamp column for social_signals ('ts' vs 'timestamp')
            ts_col = 'ts'
            try:
                meta = await self._qdb_query("show columns from social_signals")
                if meta and isinstance(meta.get('dataset'), list):
                    cols = []
                    name_idx = None
                    for i, c in enumerate(meta.get('columns', []) or []):
                        if c.get('name') == 'column':
                            name_idx = i
                            break
                    for r in meta.get('dataset') or []:
                        try:
                            if name_idx is not None:
                                cols.append(str(r[name_idx]))
                        except Exception:
                            continue
                    if 'ts' in cols:
                        ts_col = 'ts'
                    elif 'timestamp' in cols:
                        ts_col = 'timestamp'
            except Exception:
                ts_col = 'ts'
            await self._delete_in_time_windows(
                table='social_signals',
                ts_column=ts_col,
                where_extra=None,
                cutoff_days=self.social_retention_days,
                reason='age_cutoff'
            )
            # 3) Low-value pruning: very low engagement & influence & near-neutral sentiment older than social_low_days
            try:
                sql_lv = (
                    "DELETE FROM social_signals "
                    f"WHERE ts < dateadd('d', -{self.social_low_days}, now()) "
                    f"AND engagement < {self.social_low_eng} "
                    f"AND influence < {self.social_low_inf} "
                    f"AND abs(sentiment) < {self.social_lowvalue_sent}"
                )
                cnt_lv = await self._exec_qdb_sql(sql_lv)
                if isinstance(cnt_lv, int) and cnt_lv > 0:
                    self.stats['table_deletes']['social_signals'] = self.stats['table_deletes'].get('social_signals', 0) + cnt_lv
                    if self._retention_deletes:
                        try:
                            self._retention_deletes.labels(table='social_signals', reason='low_value_old').inc(cnt_lv)
                        except Exception:
                            pass
            except Exception as e:  # noqa: BLE001
                logger.debug(f"social_signals low-value prune failed: {e}")
            # 4) Social events value_score prune (if column exists) older than social_low_days
            try:
                social_value_floor = float(os.getenv('SOCIAL_VALUE_SCORE_FLOOR','0.25'))
            except Exception:
                social_value_floor = 0.25
            try:
                sql_sv = (
                    "DELETE FROM social_events "
                    f"WHERE timestamp < dateadd('d', -{self.social_low_days}, now()) "
                    f"AND value_score < {social_value_floor}"
                )
                cnt_sv = await self._exec_qdb_sql(sql_sv)
                if isinstance(cnt_sv, int) and cnt_sv > 0:
                    self.stats['table_deletes']['social_events'] = self.stats['table_deletes'].get('social_events', 0) + cnt_sv
                    if self._retention_deletes:
                        try:
                            self._retention_deletes.labels(table='social_events', reason='low_value_old').inc(cnt_sv)
                        except Exception:
                            pass
            except Exception as e:  # noqa: BLE001
                logger.debug(f"social_events value_score prune failed: {e}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"social_events retention apply failed: {e}")

    async def _delete_in_time_windows(self, *, table: str, ts_column: str = 'timestamp', where_extra: Optional[str], cutoff_days: int, reason: str) -> None:
        """Perform deletes in windows to avoid massive single-batch operations.

        Deletes rows where timestamp < now()-cutoff_days, processed in recent windows of size
        self.retention_window_days, up to self.retention_max_windows per run.
        """
        try:
            processed = 0
            for i in range(self.retention_max_windows):
                # Define window [older_bound, newer_bound)
                older = cutoff_days + (i+1) * self.retention_window_days
                newer = cutoff_days + i * self.retention_window_days
                sql = (
                    f"DELETE FROM {table} "
                    f"WHERE {ts_column} < dateadd('d', -{newer}, now()) "
                    f"AND {ts_column} >= dateadd('d', -{older}, now())"
                )
                if where_extra:
                    sql += f" AND ({where_extra})"
                cnt = await self._exec_qdb_sql(sql)
                if isinstance(cnt, int):
                    self.stats['table_deletes'][table] = self.stats['table_deletes'].get(table, 0) + max(0, cnt)
                    self.stats['data_points_deleted'] += max(0, cnt)
                    if self._retention_deletes:
                        try:
                            self._retention_deletes.labels(table=table, reason=reason).inc(max(0, cnt))
                        except Exception:
                            pass
                processed += 1
                # If QuestDB didn't return counts, we still limit by windows
            logger.debug(f"Retention windowed delete for {table}: windows={processed}, cutoff_days={cutoff_days}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Windowed delete failed for {table}: {e}")
    
    def compress_data_block(self, data: Any) -> Tuple[bytes, float]:
        """
        Compress a data block and return compressed data with compression ratio.
        
        Returns:
            Tuple of (compressed_data, compression_ratio)
        """
        # Serialize data
        serialized = pickle.dumps(data)
        original_size = len(serialized)
        
        # Compress
        compressed = zlib.compress(serialized, level=self.compression_level)
        compressed_size = len(compressed)
        
        # Calculate compression ratio
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        # Update statistics
        self.stats['storage_saved_mb'] += (original_size - compressed_size) / (1024 * 1024)
        
        return compressed, compression_ratio
    
    def decompress_data_block(self, compressed_data: bytes) -> Any:
        """Decompress a data block."""
        decompressed = zlib.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    async def get_retention_statistics(self) -> Dict[str, Any]:
        """Get retention service statistics."""
        return {
            "statistics": self.stats,
            "policies": [
                {
                    "granularity": p.granularity.value,
                    "retention_days": p.retention_days,
                    "compression_after_days": p.compression_after_days,
                    "aggregation_target": p.aggregation_target.value if p.aggregation_target else None
                }
                for p in self.retention_policies
            ],
            "priority_symbols": self.priority_symbols,
            "compression_level": self.compression_level
        }
    
    async def optimize_symbol_retention(self, symbol: str, importance_score: float):
        """
        Dynamically adjust retention for a symbol based on its importance.
        
        Args:
            symbol: Stock symbol
            importance_score: 0-1 score indicating symbol importance
        """
        if importance_score > 0.8 and symbol not in self.priority_symbols:
            self.priority_symbols.append(symbol)
            logger.info(f"Added {symbol} to priority retention list (score: {importance_score:.2f})")
        elif importance_score < 0.3 and symbol in self.priority_symbols:
            self.priority_symbols.remove(symbol)
            logger.info(f"Removed {symbol} from priority retention list (score: {importance_score:.2f})")


# Global service instance
_retention_service: Optional[DataRetentionService] = None


async def get_retention_service() -> DataRetentionService:
    """Get or create retention service instance."""
    global _retention_service
    if _retention_service is None:
        _retention_service = DataRetentionService()
        await _retention_service.start()
    return _retention_service
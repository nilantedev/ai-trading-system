"""Historical Collector (Phase 1.5)

Extended from initial skeleton to add Redis-backed persistence for progress &
dedup tracking (daily bars only). Still intentionally minimal on write side –
no storage to MinIO / QuestDB yet (future phases) but now resilient across
process restarts so pilot runs can pause/resume safely.

Capabilities:
    * Daily bar backfill via `MarketDataService.get_bulk_daily_historical`
    * Feature gated by ENABLE_HISTORICAL_BACKFILL (default: false)
    * Redis persistence (if available) for:
            - Last ingested date per symbol (progress)
            - Dedup set of ingested YYYY-MM-DD dates per symbol
    * In-memory fallback dedup if Redis unavailable
    * Idempotent: re-running will skip already ingested dates

Key Design Choices:
    * Keep provider logic separate (provider abstraction) – collector focuses
        on orchestration, dedup & progress semantics.
    * Redis chosen for early phase (fast, already deployed). Migration to
        Postgres progress table can layer on later without changing public API.
    * No implicit scheduling: caller triggers backfill explicitly.

SAFE FOR PRODUCTION: Only reads external data and writes lightweight metadata
to Redis when enabled by feature flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Set
import os
import logging
from prometheus_client import Gauge  # Prometheus metric for progress timestamp

logger = logging.getLogger(__name__)

# Gauge storing UNIX epoch seconds of last successful historical backfill progress update (any symbol)
_BACKFILL_PROGRESS_TS = Gauge(
    "historical_backfill_progress_timestamp_seconds",
    "Unix timestamp of the most recent successful historical daily backfill progress update across all symbols"
)


@dataclass
class BackfillResult:
    symbol: str
    requested_start: datetime
    requested_end: datetime
    bars_retrieved: int
    status: str
    error: Optional[str] = None


class HistoricalCollector:
    """Historical daily collector with Redis-backed progress & dedup.

    Public Methods:
      backfill_daily(symbol, start_date, end_date) -> BackfillResult

    Persistence (Redis key schema):
      hist:progress:daily:<symbol> -> last ingested date (YYYY-MM-DD)
      hist:dedup:daily:<symbol>    -> SET of date strings already ingested

    If Redis not available, falls back to per-run in-memory dedup.

    NOTE: Redis connection is NOT established in __init__ (cannot await). Call
    await ensure_redis() once after construction (idempotent, safe) before invoking
    backfill_daily. If ensure_redis() is never called or fails, the collector
    will transparently fall back to in-memory dedup only.
    """

    def __init__(self, market_data_service, redis_client: Optional[object] = None):
        self.market_data_service = market_data_service
        self.enabled = os.getenv("ENABLE_HISTORICAL_BACKFILL", "false").lower() == "true"
        self._dedup_seen: Dict[str, Set[str]] = {}
        self._redis = redis_client  # May be a RedisClient wrapper or None
        self._redis_ready = False
        if self._redis is None:
            try:
                from trading_common.database import get_redis_client  # type: ignore
                self._redis = get_redis_client()
            except Exception:  # noqa: BLE001
                self._redis = None

    async def ensure_redis(self) -> None:
        """Idempotently establish Redis connection if possible.

        Swallows all exceptions (logs at debug) so ingestion never hard-fails.
        """
        if self._redis is None or self._redis_ready:
            return
        try:
            # Only connect if underlying client not yet created
            if getattr(self._redis, '_client', None) is None:
                connect_fn = getattr(self._redis, 'connect', None)
                if connect_fn:
                    await connect_fn()  # type: ignore[misc]
            # Mark ready if a client was established
            if getattr(self._redis, '_client', None) is not None:
                self._redis_ready = True
                logger.info("HistoricalCollector Redis ready")
        except Exception as e:  # noqa: BLE001
            logger.debug("HistoricalCollector Redis init failed: %s", e)

    # ---------------- Persistence Helpers ---------------- #
    def _progress_key(self, symbol: str) -> str:
        return f"hist:progress:daily:{symbol.upper()}"

    def _dedup_key(self, symbol: str) -> str:
        return f"hist:dedup:daily:{symbol.upper()}"

    async def _get_last_date(self, symbol: str) -> Optional[datetime]:
        if not self._redis:
            return None
        try:
            raw = await self._redis.get(self._progress_key(symbol))  # type: ignore[attr-defined]
            if raw:
                return datetime.strptime(raw, "%Y-%m-%d")
        except Exception:  # noqa: BLE001
            pass
        return None

    async def _set_last_date(self, symbol: str, dt: datetime) -> None:
        if not self._redis:
            return
        try:
            await self._redis.set(self._progress_key(symbol), dt.strftime("%Y-%m-%d"))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    async def _is_deduped(self, symbol: str, date_str: str) -> bool:
        if self._redis:
            try:
                # Using SISMEMBER requires direct redis client; wrapper exposes exists helpers; try sadd pattern
                # We'll optimistically add via SADD and check return (0 => already present)
                added = await self._redis.sadd(self._dedup_key(symbol), date_str)  # type: ignore[attr-defined]
                if added == 0:
                    return True  # already existed
                # Optionally set TTL on first addition
                await self._redis.client.expire(self._dedup_key(symbol), 60 * 60 * 24 * 30)  # 30d TTL
                return False
            except Exception:  # noqa: BLE001
                pass
        # Fallback in-memory
        seen = self._dedup_seen.setdefault(symbol, set())
        if date_str in seen:
            return True
        seen.add(date_str)
        return False

    async def backfill_daily(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> BackfillResult:
        """Backfill daily bars for symbol between start & end (inclusive start, exclusive end).

        Adjusts start_date forward if Redis progress indicates partial completion.
        """
        if not self.enabled:
            logger.info("Historical backfill disabled via ENABLE_HISTORICAL_BACKFILL flag")
            return BackfillResult(symbol, start_date, end_date, 0, status="disabled")

        if start_date >= end_date:
            return BackfillResult(symbol, start_date, end_date, 0, status="error", error="start_date >= end_date")

        # Resume support: advance start to the day after last ingested (if within range)
        try:
            last_dt = await self._get_last_date(symbol)
            if last_dt and last_dt >= start_date:
                resume_start = last_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                if resume_start < end_date:
                    start_date = resume_start
        except Exception:  # noqa: BLE001
            pass

        try:
            bars = await self.market_data_service.get_bulk_daily_historical(symbol, start_date, end_date)
            logger.info("Retrieved %d raw daily bars for %s", len(bars), symbol)

            unique: List = []
            newest: Optional[datetime] = None
            for b in bars:
                date_str = b.timestamp.strftime("%Y-%m-%d")
                if await self._is_deduped(symbol, date_str):
                    continue
                unique.append(b)
                if newest is None or b.timestamp > newest:
                    newest = b.timestamp

            if newest:
                await self._set_last_date(symbol, newest)
                # Emit progress timestamp (coarse-grained) so alert rules can detect stalls.
                try:
                    _BACKFILL_PROGRESS_TS.set(datetime.utcnow().timestamp())
                except Exception:  # noqa: BLE001
                    pass

            return BackfillResult(symbol, start_date, end_date, len(unique), status="success")
        except Exception as e:  # noqa: BLE001
            logger.error("Backfill failed for %s: %s", symbol, e)
            return BackfillResult(symbol, start_date, end_date, 0, status="error", error=str(e))


__all__ = ["HistoricalCollector", "BackfillResult"]

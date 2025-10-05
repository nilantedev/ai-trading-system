#!/usr/bin/env python3
"""
AI Trading System - Data Ingestion Service
Handles real-time market data, news, and social sentiment collection.
"""

import asyncio
import sys
from pathlib import Path

# Robust path setup: compose overrides PYTHONPATH to /app, so we defensively
# add both shared python-common and the service directory itself for relative
# imports without relying on container-level env. Idempotent (duplicates ignored).
_THIS_DIR = Path(__file__).parent
_SHARED = (_THIS_DIR / "../../shared/python-common").resolve()
_SHARED_ROOT = (_THIS_DIR / "../../shared").resolve()
_SERVICE_DIR = _THIS_DIR.resolve()
for _p in (_SHARED, _SHARED_ROOT, _SERVICE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime, timedelta, date, time as _time
from typing import Dict, List, Optional, Any, Union
import random
import os
import socket
from zoneinfo import ZoneInfo
import json
import math
import aiohttp
import glob
import base64
import pathlib

try:
    # Run blocking work in a thread to avoid event-loop quirks with C extensions
    from starlette.concurrency import run_in_threadpool  # type: ignore
except Exception:  # noqa: BLE001
    run_in_threadpool = None  # type: ignore

from trading_common import get_logger, get_settings, MarketData, NewsItem, SocialSentiment
from trading_common.ingestion import get_ingestion_manager  # new metrics summary inclusion
from trading_common.cache import get_trading_cache
from trading_common.database import get_redis_client
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Gauge, Counter, Histogram
import time
from provider_metrics import record_provider_request, set_backfill_progress
from provider_metrics import PROVIDER_HTTP_RESPONSES_TOTAL, PROVIDER_RATE_LIMIT_TOTAL  # type: ignore
from pydantic import BaseModel, Field
# Shared observability (canonical app_http_* metrics & concurrency control)
try:  # Import defensively so service can still start in degraded mode
    from observability import install_observability  # type: ignore
except Exception:  # noqa: BLE001
    install_observability = None  # type: ignore

# Import new services
from market_data_service import get_market_data_service
from historical_collector import HistoricalCollector
# Additional service imports used later in this module (ensure names exist at runtime)
from news_service import get_news_service
from reference_data_service import get_reference_data_service
from data_validation_service import get_data_validation_service
from data_retention_service import get_retention_service
from smart_data_filter import SmartDataFilter
from social_media_collector import get_social_media_collector
from calendar_service import get_calendar_service
try:
    # Optional import for vector fallback indexing
    from shared.vector.indexing import index_options_fallback  # type: ignore
except Exception:
    try:
        from ..shared.vector.indexing import index_options_fallback  # type: ignore
    except Exception:
        index_options_fallback = None  # type: ignore

# Module logger and settings
logger = get_logger(__name__)
settings = get_settings()

# ---- Module-level runtime state & helpers (declared early) ----
# Start time for uptime metrics
_START_TIME = datetime.utcnow()

# Dependency handles initialized during lifespan
cache_client = None
redis_client = None
market_data_svc = None
news_svc = None
reference_svc = None
validation_svc = None
retention_svc = None
social_collector = None
historical_collector = None
smart_filter = None
calendar_svc = None
QUOTE_LOOP_TASK = None  # asyncio.Task | None for quote stream loop

# Admin/runtime overrides
QUOTE_STREAM_GATING_OVERRIDE = None  # None => use env default; bool overrides
QUOTE_STREAM_SYMBOLS_OVERRIDE = None  # Optional[List[str]]

# Prometheus counter alias used with defensive re-registration guards
_PCounter = Counter

# Default window size for news backfill batching (days)
NEWS_BACKFILL_WINDOW_DAYS = 14

# Options history default knobs
OPTIONS_HISTORY_MAX_CONTRACTS = 1000
OPTIONS_HISTORY_PACING_SECONDS = 0.2

# News backfill pacing between windows (seconds)
NEWS_BACKFILL_PACING_SECONDS = 0.2

# Historical lookback horizons (years) for planning default ranges
# Tunable via environment; conservative defaults to respect provider limits
try:
    NEWS_BACKFILL_YEARS = int(os.getenv("NEWS_BACKFILL_YEARS", "3"))
except Exception:
    NEWS_BACKFILL_YEARS = 3
try:
    SOCIAL_BACKFILL_YEARS = int(os.getenv("SOCIAL_BACKFILL_YEARS", "1"))
except Exception:
    SOCIAL_BACKFILL_YEARS = 1

# Lightweight loop status structure used by status endpoints/loops
LOOP_STATUS: Dict[str, Dict[str, Optional[str]]] = {
    'quote_stream': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    'social_stream': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    'news_stream': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    'daily_delta': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    # Additional loops initialized during lifespan
    'daily_options': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    'options_coverage': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    'artifact_upload': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    'equities_backfill': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    'equities_coverage': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    # Vector store reconciliation (news -> Weaviate) loop status
    'vector_reconcile': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    # Housekeeping loop (archive/prune old JSON artifacts to MinIO)
    'housekeeping': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
    # Automated watchlist refresh (discover new optionable symbols daily)
    'watchlist_refresh': {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None},
}

# Gauge for loop last-run timestamps (best-effort; safe if registry already has it)
try:
    LOOP_LAST_RUN_UNIX = Gauge('loop_last_run_unix_seconds', 'Last loop run time (unix seconds)', ['loop'])
    # Pre-warm label sets we reference to avoid missing series in dashboards
    for _loop in list(LOOP_STATUS.keys()):
        try:
            LOOP_LAST_RUN_UNIX.labels(loop=_loop).set(0)
        except Exception:
            pass
except Exception:  # noqa: BLE001
    LOOP_LAST_RUN_UNIX = None  # type: ignore

# Vector backfill/indexing metrics (best-effort)
try:
    VECTOR_NEWS_INDEXED = Counter('vector_news_indexed_total', 'Total news items indexed into vector store', ['path'])
except Exception:  # noqa: BLE001
    VECTOR_NEWS_INDEXED = None
try:
    VECTOR_NEWS_LAST_TS = Gauge('vector_news_last_indexed_timestamp_seconds', 'Unix ts of last news item indexed to vector store')
except Exception:  # noqa: BLE001
    VECTOR_NEWS_LAST_TS = None


# ---------------------- One-off Vector Reconcile Helper ---------------------- #
async def _vector_reconcile_once(rec_days: int = 3, rec_limit: int = 2000) -> dict:
    """One-off reconciliation: index recent QuestDB news into vector store (Weaviate).

    Mirrors the background loop logic but runs once on demand.
    Returns a small report: indexed count and last timestamp observed.
    """
    # Discover available columns
    try:
        meta = await _qdb_exec("show columns from news_items")
        name_idx = next((i for i, c in enumerate(meta.get('columns', []) or []) if c.get('name') == 'column'), None)
        cols_available: list[str] = []
        for r in meta.get('dataset') or []:
            try:
                if name_idx is not None:
                    cols_available.append(str(r[name_idx]))
            except Exception:
                continue
    except Exception:
        cols_available = []

    # Minimal required fields are title and ts; add optional when present
    proj_parts = ["title", "ts"]
    for c in ("source", "url", "symbol", "sentiment", "relevance", "provider", "value_score"):
        if c in cols_available and c not in proj_parts:
            proj_parts.append(c)
    select_list = ", ".join(proj_parts)
    look_sql = (
        f"select {select_list} from news_items "
        f"where ts >= dateadd('d', -{max(1, int(rec_days))}, now()) "
        "and title is not null and title != '' "
        "order by ts desc limit " + str(max(1, int(rec_limit)))
    )
    data = await _qdb_exec(look_sql, timeout=20.0)
    cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
    items = []
    last_ts_val = None
    for r in (data.get('dataset') or [])[:rec_limit]:
        try:
            title = str(r[cols['title']]) if 'title' in cols else ''
            content = ''
            source = str(r[cols.get('source')]) if 'source' in cols and r[cols.get('source')] is not None else 'news_items'
            ts = r[cols.get('ts')] if 'ts' in cols else None
            symbol = r[cols.get('symbol')] if 'symbol' in cols else None
            if ts is not None:
                try:
                    last_ts_val = ts
                except Exception:
                    pass
            items.append({
                'title': title,
                'content': content,
                'source': source,
                'published_at': str(ts) if ts is not None else datetime.utcnow().isoformat(),
                'symbols': [str(symbol).upper()] if symbol else []
            })
        except Exception:
            continue
    indexed = 0
    if items:
        try:
            try:
                from shared.vector.indexing import index_news_fallback  # type: ignore
            except Exception:
                from ..shared.vector.indexing import index_news_fallback  # type: ignore
            indexed = await index_news_fallback(items)
        except Exception:
            indexed = 0
    try:
        if indexed and VECTOR_NEWS_INDEXED is not None:
            VECTOR_NEWS_INDEXED.labels(path='reconcile_once').inc(indexed)
    except Exception:
        pass
    try:
        if last_ts_val and VECTOR_NEWS_LAST_TS is not None:
            from datetime import datetime as _dt
            if isinstance(last_ts_val, str):
                _dtv = _dt.fromisoformat(str(last_ts_val).replace('Z','+00:00'))
                VECTOR_NEWS_LAST_TS.set(_dtv.timestamp())
            elif hasattr(last_ts_val, 'timestamp'):
                VECTOR_NEWS_LAST_TS.set(last_ts_val.timestamp())
    except Exception:
        pass
    return {"indexed": int(indexed), "last_ts": str(last_ts_val) if last_ts_val is not None else None}


def _parse_hhmm(s: str) -> tuple[int, int]:
    try:
        hh, mm = s.strip().split(":", 1)
        return int(hh), int(mm)
    except Exception:  # noqa: BLE001
        return 0, 0


def _parse_date_yyyy_mm_dd(s: Optional[str]) -> Optional[datetime]:
    """Parse a YYYY-MM-DD string into a datetime at 00:00:00 UTC.

    Returns None if input is falsy or malformed.
    """
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def is_trading_hours() -> bool:
    """Return True if current local time is within configured trading hours on a weekday.

    Env:
      - TRADING_TIMEZONE (default America/New_York)
      - TRADING_HOURS_OPEN (default 09:30)
      - TRADING_HOURS_CLOSE (default 16:00)
    """
    try:
        tz = os.getenv("TRADING_TIMEZONE", "America/New_York")
        now_local = datetime.now(ZoneInfo(tz))
        if now_local.weekday() >= 5:  # Sat/Sun
            return False
        open_s = os.getenv("TRADING_HOURS_OPEN", "09:30")
        close_s = os.getenv("TRADING_HOURS_CLOSE", "16:00")
        oh, om = _parse_hhmm(open_s)
        ch, cm = _parse_hhmm(close_s)
        open_t = _time(hour=oh, minute=om)
        close_t = _time(hour=ch, minute=cm)
        now_t = now_local.time()
        return open_t <= now_t <= close_t
    except Exception:
        return True  # fail-open to avoid stalling loops if tz parsing fails


# ---------------------- QuestDB Helper (HTTP /exec via GET) ---------------------- #
async def _qdb_exec(sql: str, timeout: float = 12.0) -> dict:
    """Execute a QuestDB SQL via HTTP GET /exec and return JSON.

    Always uses GET to avoid "bad method" errors on the QuestDB HTTP endpoint.
    Environment variables honored (fall back to service DNS defaults):
      - QUESTDB_HTTP_URL or QUESTDB_HOST/QUESTDB_HTTP_PORT
    """
    host = os.getenv('QUESTDB_HOST', 'trading-questdb')
    http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
    qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
    # Use a per-call client to keep the helper self-contained and low-risk
    try:
        timeout_cfg = aiohttp.ClientTimeout(total=max(1.0, float(timeout)))
    except Exception:
        timeout_cfg = aiohttp.ClientTimeout(total=12.0)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.get(qdb_url, params={"query": sql}) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"QuestDB HTTP {resp.status}: {txt[:180]}")
            try:
                return await resp.json()
            except Exception:
                # Normalize to an empty payload if non-JSON (older versions)
                return {"columns": [], "dataset": []}


# ---------------------- Lightweight Health & Coverage Endpoints ---------------------- #
# NOTE: Earlier definitions of /health/extended and /calendar/coverage were moved below
# after the FastAPI app is created to avoid referencing `app` before it exists. The
# canonical implementations now live later in this module.

# ---------------------- Social & News Backfill Endpoints (below app init) ---------------------- #

# Retention deletions counter (referenced by Grafana panels) — ensure at least zero series exist
try:
    _RETENTION_DELETES = _PCounter('retention_deletions_total', 'Total rows deleted by retention', ['table','reason'])
    # Pre-warm expected label sets (best-effort)
    # Pre-warm for both legacy and current table names to keep panels stable
    for t in ('option_daily','options_data','news_items','social_signals'):
        for r in ('zero_volume_old','expiry_old','age_cutoff','low_value_old'):
            try:
                _RETENTION_DELETES.labels(table=t, reason=r).inc(0)
            except Exception:
                pass
except Exception:  # noqa: BLE001
    _RETENTION_DELETES = None

# Historical backfill status gauge (single active series w/ status label)
try:  # Guard against duplicate registration on reload
    HISTORICAL_BACKFILL_STATUS = Gauge(
        'historical_backfill_status',
        'Current historical backfill lifecycle status (value always 1 for the active status)',
        ['status']
    )
except Exception:  # noqa: BLE001
    HISTORICAL_BACKFILL_STATUS = None  # Already registered

# (duplicate definition removed) LOOP_LAST_RUN_UNIX is defined earlier with pre-warmed labels

# Provider metrics now centralized in provider_metrics.py

# Options backfill metrics (guarded against duplicate registration)
try:
    OPTIONS_BACKFILL_CONTRACTS = _PCounter(
        'options_backfill_contracts_total',
        'Total option contracts processed by ingestion path',
        ['path']
    )
except Exception:
    OPTIONS_BACKFILL_CONTRACTS = None
try:
    OPTIONS_BACKFILL_BARS = _PCounter(
        'options_backfill_bars_total',
        'Total option daily bars ingested by path',
        ['path']
    )
except Exception:
    OPTIONS_BACKFILL_BARS = None
try:
    OPTIONS_BACKFILL_ERRORS = _PCounter(
        'options_backfill_errors_total',
        'Errors encountered during options backfill by path',
        ['path']
    )
except Exception:  # noqa: BLE001
    OPTIONS_BACKFILL_ERRORS = None

# Option aggregate fetch instrumentation (empty vs rows vs error + latency)
try:
    OPTIONS_AGGS_FETCHES = _PCounter(
        'options_aggs_fetch_total',
        'Total option aggregate fetch attempts',
        ['result']  # result label: empty|rows|error
    )
except Exception:  # noqa: BLE001
    OPTIONS_AGGS_FETCHES = None
try:
    OPTIONS_AGGS_ROWS = _PCounter(
        'options_aggs_rows_total',
        'Total option aggregate rows (per contract call)'
    )
except Exception:  # noqa: BLE001
    OPTIONS_AGGS_ROWS = None
try:
    OPTIONS_AGGS_LATENCY = Histogram(
        'options_aggs_fetch_latency_seconds',
        'Latency of option aggregate fetch calls',
        buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    )
except Exception:  # noqa: BLE001
    OPTIONS_AGGS_LATENCY = None

# Equities coverage metrics (aggregate only – avoid per-symbol cardinality)
try:
    EQUITIES_COVERAGE_RATIO_20Y = Gauge(
        'equities_coverage_ratio_20y',
        'Ratio (0-1) of equity symbols with >= ~20 years (>=19.5y) of daily bar history'
    )
except Exception:  # noqa: BLE001
    EQUITIES_COVERAGE_RATIO_20Y = None
try:
    EQUITIES_COVERAGE_SYMBOLS_TOTAL = Gauge(
        'equities_coverage_symbols_total',
        'Total equity symbols evaluated for coverage (full scan only)'
    )
except Exception:  # noqa: BLE001
    EQUITIES_COVERAGE_SYMBOLS_TOTAL = None
try:
    EQUITIES_COVERAGE_SYMBOLS_20Y = Gauge(
        'equities_coverage_symbols_20y',
        'Equity symbols meeting >= ~20 year span (full scan only)'
    )
except Exception:  # noqa: BLE001
    EQUITIES_COVERAGE_SYMBOLS_20Y = None
try:
    EQUITIES_REMEDIATED_SYMBOLS = Counter(
        'equities_remediated_symbols_total',
        'Symbols remediated (historical backfill performed)'
    )
except Exception:  # noqa: BLE001
    EQUITIES_REMEDIATED_SYMBOLS = None
try:
    EQUITIES_REMEDIATED_BARS = Counter(
        'equities_remediated_bars_total',
        'Daily bars ingested via remediation' 
    )
except Exception:  # noqa: BLE001
    EQUITIES_REMEDIATED_BARS = None
try:
    EQUITIES_REMEDIATION_RUNS = Counter(
        'equities_remediation_runs_total',
        'Remediation endpoint executions',
        ['result']
    )
except Exception:  # noqa: BLE001
    EQUITIES_REMEDIATION_RUNS = None

# ----------------------- Exclusion & IPO helpers ----------------------- #
_EQUITY_EXCLUSIONS_CACHE: set[str] | None = None

def _load_equity_exclusions() -> set[str]:
    global _EQUITY_EXCLUSIONS_CACHE
    if _EQUITY_EXCLUSIONS_CACHE is not None:
        return _EQUITY_EXCLUSIONS_CACHE
    exclusions: set[str] = set()
    # Env variable list
    env_list = os.getenv('EQUITY_COVERAGE_EXCLUSIONS', '').strip()
    if env_list:
        for s in env_list.split(','):
            s2 = s.strip().upper()
            if s2:
                exclusions.add(s2)
    # File path
    path = os.getenv('EQUITY_COVERAGE_EXCLUSIONS_FILE', '').strip()
    if path and os.path.isfile(path):
        try:
            with open(path, 'r') as f:
                for line in f:
                    s = line.strip().upper()
                    if s and not s.startswith('#'):
                        exclusions.add(s)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed reading exclusions file", path=path, error=str(e))
    _EQUITY_EXCLUSIONS_CACHE = exclusions
    return exclusions

async def _get_listing_date(symbol: str) -> datetime | None:
    """Attempt to retrieve listing/IPO date from reference service (best-effort)."""
    if not reference_svc:
        return None
    getter = getattr(reference_svc, 'get_security_info', None)
    if not callable(getter):
        return None
    try:
        info = await getter(symbol, False)
        if not info:
            return None
        # Attempt multiple attribute names gracefully
        for attr in ('listing_date','listed_date','ipo_date','list_date'):
            val = getattr(info, attr, None)
            if val:
                if isinstance(val, datetime):
                    return val
                try:
                    return datetime.fromisoformat(str(val))
                except Exception:  # noqa: BLE001
                    continue
    except Exception:
        return None
    return None

async def _connect_with_retry(name: str, factory, attempts: int = 5, base_delay: float = 0.5):
    """Attempt to create a dependency with simple exponential backoff.
    Returns the instance or None (degraded mode) without raising to keep liveness healthy.
    """
    for attempt in range(1, attempts + 1):
        try:
            inst = factory()
            if inst:
                logger.info(f"{name} connection established", attempt=attempt)
                return inst
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{name} connection attempt {attempt}/{attempts} failed: {e}")
        await asyncio.sleep(min(base_delay * (2 ** (attempt - 1)), 8))
    logger.error(f"{name} connection failed after {attempts} attempts; continuing in degraded mode")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management with resilient startup."""
    global cache_client, redis_client, market_data_svc, news_svc, reference_svc, validation_svc, retention_svc, smart_filter, historical_collector

    logger.info("Starting Data Ingestion Service")

    try:
        # Resilient connections
        cache_client = await _connect_with_retry("cache", get_trading_cache)
        redis_client = await _connect_with_retry("redis", get_redis_client)

        # --- Optional component initialization with timeout so port binds fast ---
        async def _init_optional_components():
            # Using module-level globals (declared in lifespan scope) for component references.
            # IMPORTANT: declare globals so assignments below update module-level state.
            global market_data_svc, news_svc, reference_svc, validation_svc, retention_svc, historical_collector, smart_filter

            async def init_component(coro_factory, name: str):
                try:
                    comp = await coro_factory()
                    logger.info(f"{name} initialized")
                    return comp
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"{name} failed to initialize: {e}")
                    return None

            market_data_svc = await init_component(get_market_data_service, "market_data_service")
            news_svc = await init_component(get_news_service, "news_service")
            reference_svc = await init_component(get_reference_data_service, "reference_data_service")
            validation_svc = await init_component(get_data_validation_service, "data_validation_service")
            retention_svc = await init_component(get_retention_service, "retention_service")
            # Calendar service (provider: Alpha Vantage or EODHD)
            try:
                global calendar_svc
                calendar_svc = await get_calendar_service()
                if calendar_svc and calendar_svc.enabled:
                    logger.info("Calendar service initialized", provider=getattr(calendar_svc, 'provider', 'unknown'))
                else:
                    logger.info("Calendar service disabled (missing key or flag)")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Calendar service init failed: {e}")
            # Social media collector (optional)
            try:
                global social_collector
                social_collector = await get_social_media_collector()
                logger.info("Social media collector initialized")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Social media collector init failed: {e}")

            # Historical collector (feature-gated) – safe even if redis unavailable
            try:
                historical_collector = HistoricalCollector(market_data_svc, redis_client)
                if historical_collector.enabled:
                    logger.info("HistoricalCollector enabled (feature flag ON)")
                else:
                    logger.info("HistoricalCollector disabled (ENABLE_HISTORICAL_BACKFILL not true)")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"HistoricalCollector init failed: {e}")

            # Smart filter (non-critical)
            try:
                smart_filter = SmartDataFilter(
                    min_quality_score=0.6,
                    volume_surge_threshold=2.0,
                    volatility_threshold=0.02,
                    enable_adaptive_filtering=True,
                    enable_anomaly_detection=True
                )
                await smart_filter.initialize()
                logger.info("Smart data filter initialized")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Smart data filter init failed: {e}")

            # Ensure Weaviate schema (best-effort) so fallback/direct indexing has classes available
            try:
                if os.getenv("ENABLE_WEAVIATE_SCHEMA_ENSURE", "true").lower() in ("1","true","yes"):
                    try:
                        from shared.vector.weaviate_schema import get_weaviate_client, ensure_desired_schema  # type: ignore
                    except Exception:
                        from ..shared.vector.weaviate_schema import get_weaviate_client, ensure_desired_schema  # type: ignore
                    client = get_weaviate_client()
                    ensure_desired_schema(client)
                    logger.info("Weaviate schema ensured (best-effort)")
            except Exception as e:  # noqa: BLE001
                logger.warning("Weaviate schema ensure skipped/failed: %s", e)

            # Lightweight daily delta scheduler (feature gated)
            if os.getenv("ENABLE_DAILY_DELTA", "false").lower() in ("1", "true", "yes"):
                # Configurable parameters
                try:
                    delta_interval = int(os.getenv("DAILY_DELTA_INTERVAL_SECONDS", "3600"))
                except Exception:
                    delta_interval = 3600
                try:
                    delta_lookback_days = int(os.getenv("DAILY_DELTA_LOOKBACK_DAYS", "1"))
                except Exception:
                    delta_lookback_days = 1
                try:
                    delta_max_symbols = int(os.getenv("DAILY_DELTA_MAX_SYMBOLS", "0"))
                    if delta_max_symbols <= 0:
                        delta_max_symbols = None  # No limit - process all
                except Exception:
                    delta_max_symbols = None  # No limit
                try:
                    delta_pacing_seconds = float(os.getenv("DAILY_DELTA_PACING_SECONDS", "0.15"))
                except Exception:
                    delta_pacing_seconds = 0.15
                async def _daily_delta_loop():
                    while True:
                        try:
                            LOOP_STATUS['daily_delta']["enabled"] = True
                            LOOP_STATUS['daily_delta']["interval_seconds"] = max(60, delta_interval)
                            if reference_svc and market_data_svc:
                                symbols = await reference_svc.get_watchlist_symbols()
                                if symbols:
                                    end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                                    # Use configurable lookback
                                    start_dt = end_dt - timedelta(days=max(1, delta_lookback_days))
                                    # Process all symbols if no limit, otherwise apply limit
                                    process_symbols = symbols if delta_max_symbols is None else symbols[:max(1, delta_max_symbols)]
                                    for sym in process_symbols:
                                        try:
                                            rows = await market_data_svc.get_bulk_daily_historical(sym, start_dt, end_dt)
                                            logger.info("Daily delta fetched", symbol=sym, bars=len(rows))
                                        except Exception as e:  # noqa: BLE001
                                            logger.warning("Daily delta fetch failed", symbol=sym, error=str(e))
                                        await asyncio.sleep(delta_pacing_seconds)
                            LOOP_STATUS['daily_delta']["last_run"] = datetime.utcnow().isoformat()
                            try:
                                if LOOP_LAST_RUN_UNIX is not None:
                                    LOOP_LAST_RUN_UNIX.labels(loop='daily_delta').set(time.time())
                            except Exception:
                                pass
                            await asyncio.sleep(max(60, delta_interval))
                        except Exception as e:  # noqa: BLE001
                            logger.warning(f"Daily delta scheduler error: {e}")
                            LOOP_STATUS['daily_delta']["last_error"] = str(e)
                            await asyncio.sleep(300)
                asyncio.create_task(_daily_delta_loop())
                logger.info("Daily delta scheduler started (feature flag ENABLE_DAILY_DELTA)")

            # Live quote streaming loop (feature gated)
            # Start if env enabled OR runtime overrides are present (admin-triggered)
            if os.getenv("ENABLE_QUOTE_STREAM", "false").lower() in ("1","true","yes") or QUOTE_STREAM_GATING_OVERRIDE is not None or QUOTE_STREAM_SYMBOLS_OVERRIDE:
                try:
                    max_syms = int(os.getenv("QUOTE_STREAM_MAX_SYMBOLS", "200"))
                except Exception:
                    max_syms = 200
                # Per-cycle sampler to respect vendor rate limits (e.g., Polygon ~100/min)
                try:
                    sample_size = int(os.getenv("QUOTE_STREAM_SAMPLE_SIZE", "2"))
                except Exception:
                    sample_size = 2
                symbols_env = os.getenv("QUOTE_STREAM_SYMBOLS", "").strip()
                async def _quote_stream_loop():
                    while True:
                        try:
                            LOOP_STATUS['quote_stream']["enabled"] = True
                            # Advertise nominal loop pacing (per-symbol loop is ~1s baseline)
                            LOOP_STATUS['quote_stream']["interval_seconds"] = 1
                            # Optional trading-hours gate (default ON for production safety)
                            _env_gate = os.getenv("QUOTE_STREAM_TRADING_HOURS_ONLY", "true").lower() in ("1","true","yes")
                            _gate = QUOTE_STREAM_GATING_OVERRIDE if QUOTE_STREAM_GATING_OVERRIDE is not None else _env_gate
                            if _gate:
                                if not is_trading_hours():
                                    # Idle politely outside trading hours
                                    await asyncio.sleep(30)
                                    continue
                            symbols: List[str] = []
                            # Highest precedence: runtime override
                            if QUOTE_STREAM_SYMBOLS_OVERRIDE:
                                symbols = [s.strip().upper() for s in QUOTE_STREAM_SYMBOLS_OVERRIDE if s and s.strip()]
                            elif symbols_env:
                                symbols = [s.strip().upper() for s in symbols_env.split(',') if s.strip()]
                            elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                                symbols = (await reference_svc.get_watchlist_symbols()) or []
                            if not symbols:
                                await asyncio.sleep(30)
                                continue
                            # Randomize order and pick a small sample to avoid hammering vendors
                            try:
                                random.shuffle(symbols)
                            except Exception:
                                pass
                            stream_syms = symbols[:max(1, min(max_syms, sample_size))]
                            if market_data_svc:
                                # Update last_run on each item received from the stream to reflect activity
                                async for _item in market_data_svc.stream_real_time_data(stream_syms):
                                    LOOP_STATUS['quote_stream']["last_run"] = datetime.utcnow().isoformat()
                                    try:
                                        if LOOP_LAST_RUN_UNIX is not None:
                                            LOOP_LAST_RUN_UNIX.labels(loop='quote_stream').set(time.time())
                                    except Exception:
                                        pass
                                    # Stream generator handles publish; yield pacing keeps event loop responsive
                                    await asyncio.sleep(0)
                            else:
                                await asyncio.sleep(5)
                        except Exception as e:  # noqa: BLE001
                            logger.warning("Quote stream loop error: %s", e)
                            LOOP_STATUS['quote_stream']["last_error"] = str(e)
                            await asyncio.sleep(5)
                # Start the loop only once; reuse single task handle
                try:
                    global QUOTE_LOOP_TASK
                    if QUOTE_LOOP_TASK is None or QUOTE_LOOP_TASK.done():
                        QUOTE_LOOP_TASK = asyncio.create_task(_quote_stream_loop())
                        logger.info("Quote stream loop started", mode="env-or-override")
                    else:
                        logger.info("Quote stream loop already running")
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed starting quote stream loop: %s", e)

            # Social streaming loop (feature gated)
            if os.getenv("ENABLE_SOCIAL_STREAM", "false").lower() in ("1","true","yes"):
                try:
                    social_interval = int(os.getenv("SOCIAL_STREAM_INTERVAL_SECONDS", "60"))
                except Exception:
                    social_interval = 60
                # Adaptive discovery knobs from recent news in QuestDB
                enable_news_driven = os.getenv("SOCIAL_SYMBOLS_FROM_NEWS_ENABLED", "true").lower() in ("1","true","yes")
                try:
                    news_hours = int(os.getenv("SOCIAL_SYMBOLS_FROM_NEWS_HOURS", "6"))
                except Exception:
                    news_hours = 6
                try:
                    news_limit = int(os.getenv("SOCIAL_SYMBOLS_FROM_NEWS_LIMIT", "20"))
                except Exception:
                    news_limit = 20
                try:
                    news_min_count = int(os.getenv("SOCIAL_SYMBOLS_FROM_NEWS_MIN_COUNT", "1"))
                except Exception:
                    news_min_count = 1
                symbols_env = os.getenv("SOCIAL_STREAM_SYMBOLS", "").strip()
                async def _social_stream_loop():
                    while True:
                        try:
                            LOOP_STATUS['social_stream']["enabled"] = True
                            LOOP_STATUS['social_stream']["interval_seconds"] = max(10, social_interval)
                            symbols = []
                            if symbols_env:
                                symbols = [s.strip().upper() for s in symbols_env.split(',') if s.strip()]
                            elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                                symbols = (await reference_svc.get_watchlist_symbols()) or []
                            # Augment with top symbols from recent news_items in QuestDB (best-effort)
                            if enable_news_driven:
                                try:
                                    import aiohttp
                                    host = os.getenv('QUESTDB_HOST', 'trading-questdb')
                                    http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
                                    qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
                                    sql = (
                                        "select symbol, count() c from news_items "
                                        f"where ts >= dateadd('h', -{max(1, news_hours)}, now()) "
                                        "and symbol is not null and symbol != '' group by symbol order by c desc limit "
                                        f"{max(1, news_limit)}"
                                    )
                                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
                                        async with session.get(qdb_url, params={"query": sql}) as resp:
                                            if resp.status == 200:
                                                data = await resp.json()
                                                cols = {c['name']: i for i, c in enumerate(data.get('columns', []))}
                                                for r in data.get('dataset', []) or []:
                                                    try:
                                                        sym = str(r[cols['symbol']]).upper()
                                                        cnt = int(r[cols.get('c', -1)]) if 'c' in cols else 0
                                                        if sym and cnt >= max(1, news_min_count) and sym not in symbols:
                                                            symbols.append(sym)
                                                    except Exception:
                                                        continue
                                except Exception:
                                    # degrade silently if QuestDB not available or query fails
                                    pass
                            symbols = symbols[:100]  # cap to keep API usage reasonable
                            # Fallback: ensure loop has work even when feeds quiet
                            if not symbols:
                                fb_raw = os.getenv('SOCIAL_FALLBACK_SYMBOLS', 'AAPL,MSFT,TSLA,NVDA,SPY')
                                symbols = [s.strip().upper() for s in fb_raw.split(',') if s.strip()]
                            if symbols and social_collector:
                                await social_collector.collect_social_data(symbols, hours_back=1)
                            LOOP_STATUS['social_stream']["last_run"] = datetime.utcnow().isoformat()
                            try:
                                if LOOP_LAST_RUN_UNIX is not None:
                                    LOOP_LAST_RUN_UNIX.labels(loop='social_stream').set(time.time())
                            except Exception:
                                pass
                            await asyncio.sleep(max(10, social_interval))
                        except Exception as e:  # noqa: BLE001
                            logger.warning("Social stream loop error: %s", e)
                            LOOP_STATUS['social_stream']["last_error"] = str(e)
                            await asyncio.sleep(30)
                asyncio.create_task(_social_stream_loop())
                logger.info("Social stream loop started (feature flag ENABLE_SOCIAL_STREAM)")

            # One-time social historical backfill on startup (feature gated)
            if os.getenv("ENABLE_SOCIAL_HISTORICAL_BACKFILL", "false").lower() in ("1","true","yes"):
                try:
                    try:
                        social_years = int(os.getenv("SOCIAL_BACKFILL_YEARS", str(SOCIAL_BACKFILL_YEARS)))
                    except Exception:
                        social_years = SOCIAL_BACKFILL_YEARS
                    try:
                        social_batch = int(os.getenv("SOCIAL_BACKFILL_SYMBOLS", "200"))
                        if social_batch <= 0:
                            social_batch = None  # No limit - process all watchlist symbols
                    except Exception:
                        social_batch = 200
                    async def _social_backfill_once():
                        if not social_collector:
                            return
                        syms: list[str] = []
                        if reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                            try:
                                syms = (await reference_svc.get_watchlist_symbols()) or []
                            except Exception:
                                syms = []
                        if not syms:
                            fb = os.getenv('SOCIAL_FALLBACK_SYMBOLS', 'AAPL,MSFT,TSLA,NVDA,SPY')
                            syms = [s.strip().upper() for s in fb.split(',') if s.strip()]
                        # Process all symbols if social_batch is None, otherwise limit
                        if social_batch is not None:
                            syms = syms[:max(1, social_batch)]
                        hours = int(social_years * 365.25 * 24)
                        # Chunk symbols to keep API load reasonable
                        chunk = int(os.getenv('SOCIAL_BACKFILL_CHUNK', '25') or '25')
                        for i in range(0, len(syms), chunk):
                            batch = syms[i:i+chunk]
                            try:
                                await social_collector.collect_social_data(batch, hours_back=hours)
                            except Exception:
                                pass
                            await asyncio.sleep(1.0)
                        logger.info("Social historical backfill completed", symbols=len(syms), years=social_years)
                    asyncio.create_task(_social_backfill_once())
                    logger.info("Social historical backfill task scheduled (ENABLE_SOCIAL_HISTORICAL_BACKFILL)")
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to schedule social historical backfill: %s", e)

            # News streaming loop (feature gated)
            if os.getenv("ENABLE_NEWS_STREAM", "false").lower() in ("1","true","yes"):
                try:
                    # Increase default interval to reduce provider 429s (especially NewsAPI)
                    news_interval = int(os.getenv("NEWS_STREAM_INTERVAL_SECONDS", "300"))
                except Exception:
                    news_interval = 300
                # Intensity knobs
                try:
                    news_stream_hours = int(os.getenv("NEWS_STREAM_HOURS_BACK", "1"))
                except Exception:
                    news_stream_hours = 1
                try:
                    news_stream_max = int(os.getenv("NEWS_STREAM_MAX_ARTICLES", "100"))
                except Exception:
                    news_stream_max = 100
                symbols_env_news = os.getenv("NEWS_STREAM_SYMBOLS", "").strip()
                async def _news_stream_loop():
                    while True:
                        try:
                            if not news_svc:
                                await asyncio.sleep(10)
                                continue
                            LOOP_STATUS['news_stream']["enabled"] = True
                            LOOP_STATUS['news_stream']["interval_seconds"] = max(15, news_interval)
                            symbols = []
                            if symbols_env_news:
                                symbols = [s.strip().upper() for s in symbols_env_news.split(',') if s.strip()]
                            elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                                symbols = (await reference_svc.get_watchlist_symbols()) or []
                            symbols = symbols[:100]
                            if not symbols:
                                fb_news_raw = os.getenv('NEWS_FALLBACK_SYMBOLS', 'AAPL,MSFT,TSLA,NVDA,SPY')
                                symbols = [s.strip().upper() for s in fb_news_raw.split(',') if s.strip()]
                            if symbols:
                                # Collect recent news; hours_back kept small for streaming
                                try:
                                    await news_svc.collect_financial_news(symbols, hours_back=max(1, news_stream_hours), max_articles=max(1, news_stream_max))
                                except Exception as e:  # noqa: BLE001
                                    logger.warning("News stream collect failed", error=str(e))
                            LOOP_STATUS['news_stream']["last_run"] = datetime.utcnow().isoformat()
                            try:
                                if LOOP_LAST_RUN_UNIX is not None:
                                    LOOP_LAST_RUN_UNIX.labels(loop='news_stream').set(time.time())
                            except Exception:
                                pass
                            await asyncio.sleep(max(15, news_interval))
                        except Exception as e:  # noqa: BLE001
                            logger.warning("News stream loop error: %s", e)
                            LOOP_STATUS['news_stream']["last_error"] = str(e)
                            await asyncio.sleep(30)
                asyncio.create_task(_news_stream_loop())
                logger.info("News stream loop started (feature flag ENABLE_NEWS_STREAM)")

            # Options daily refresh loop (feature gated)
            if os.getenv("ENABLE_DAILY_OPTIONS", "false").lower() in ("1","true","yes"):
                try:
                    opt_interval = int(os.getenv("DAILY_OPTIONS_INTERVAL_SECONDS", "14400"))  # default 4h
                except Exception:
                    opt_interval = 14400
                try:
                    opt_max_underlyings = int(os.getenv("DAILY_OPTIONS_MAX_UNDERLYINGS", "50"))
                    if opt_max_underlyings <= 0:
                        opt_max_underlyings = None  # No limit - process all watchlist symbols
                except Exception:
                    opt_max_underlyings = 50
                try:
                    opt_pacing = float(os.getenv("DAILY_OPTIONS_PACING_SECONDS", "0.2"))
                except Exception:
                    opt_pacing = 0.2
                try:
                    opt_expiry_ahead_days = int(os.getenv("DAILY_OPTIONS_EXPIRY_AHEAD_DAYS", "45"))
                except Exception:
                    opt_expiry_ahead_days = 45
                try:
                    opt_expiry_back_days = int(os.getenv("DAILY_OPTIONS_EXPIRY_BACK_DAYS", "7"))
                except Exception:
                    opt_expiry_back_days = 7

                async def _daily_options_loop():
                    while True:
                        try:
                            LOOP_STATUS['daily_options']["enabled"] = True
                            LOOP_STATUS['daily_options']["interval_seconds"] = max(300, opt_interval)
                            if reference_svc and market_data_svc and getattr(market_data_svc, 'enable_options_ingest', False):
                                underlyings = await reference_svc.get_watchlist_symbols()
                                # Process all symbols if opt_max_underlyings is None, otherwise limit
                                if opt_max_underlyings is not None:
                                    underlyings = (underlyings or [])[:max(1, opt_max_underlyings)]
                                else:
                                    underlyings = underlyings or []
                                # Derive expiry window near-term to control load
                                now_d = datetime.utcnow().date()
                                start_expiry = now_d - timedelta(days=max(0, opt_expiry_back_days))
                                end_expiry = now_d + timedelta(days=max(1, opt_expiry_ahead_days))
                                # Historical window small for daily refresh
                                start_hist = now_d - timedelta(days=5)
                                end_hist = now_d
                                for u in underlyings:
                                    try:
                                        await market_data_svc.backfill_options_chain(
                                            u.upper(),
                                            datetime.combine(start_expiry, datetime.min.time()),
                                            datetime.combine(end_expiry, datetime.min.time()),
                                            start_date=datetime.combine(start_hist, datetime.min.time()),
                                            end_date=datetime.combine(end_hist, datetime.min.time()),
                                            max_contracts=500,
                                            pacing_seconds=opt_pacing,
                                        )
                                    except Exception as e:  # noqa: BLE001
                                        logger.warning("Daily options loop for %s failed: %s", u, e)
                                    await asyncio.sleep(opt_pacing)
                            LOOP_STATUS['daily_options']["last_run"] = datetime.utcnow().isoformat()
                            try:
                                if LOOP_LAST_RUN_UNIX is not None:
                                    LOOP_LAST_RUN_UNIX.labels(loop='daily_options').set(time.time())
                            except Exception:
                                pass
                            await asyncio.sleep(max(300, opt_interval))
                        except Exception as e:  # noqa: BLE001
                            logger.warning("Daily options scheduler error: %s", e)
                            LOOP_STATUS['daily_options']["last_error"] = str(e)
                            await asyncio.sleep(600)
                asyncio.create_task(_daily_options_loop())
                logger.info("Daily options scheduler started (feature flag ENABLE_DAILY_OPTIONS)")

            # Options coverage report loop (feature gated)
            if os.getenv("ENABLE_OPTIONS_COVERAGE_REPORT", "false").lower() in ("1","true","yes"):
                try:
                    cov_interval = int(os.getenv("OPTIONS_COVERAGE_INTERVAL_SECONDS", "86400"))  # default daily
                except Exception:
                    cov_interval = 86400
                out_dir = os.getenv("OPTIONS_COVERAGE_OUTPUT_DIR", "/app/export/grafana-csv").rstrip("/")
                max_underlyings = int(os.getenv("OPTIONS_COVERAGE_MAX_UNDERLYINGS", "200"))
                symbols_env_cov = os.getenv("OPTIONS_COVERAGE_UNDERLYINGS", "").strip()

                async def _coverage_loop():
                    # local helper mirrors /coverage/options endpoint logic
                    import aiohttp
                    host = os.getenv('QUESTDB_HOST', 'trading-questdb')
                    http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
                    qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")

                    async def _q(session: aiohttp.ClientSession, sql: str) -> dict:
                        async with session.get(qdb_url, params={"query": sql}) as resp:
                            if resp.status != 200:
                                txt = await resp.text()
                                raise RuntimeError(f"QuestDB HTTP {resp.status}: {txt[:160]}")
                            return await resp.json()

                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                        while True:
                            try:
                                LOOP_STATUS['options_coverage']["enabled"] = True
                                LOOP_STATUS['options_coverage']["interval_seconds"] = max(300, cov_interval)
                                # resolve underlyings
                                syms: List[str] = []
                                if symbols_env_cov:
                                    syms = [s.strip().upper() for s in symbols_env_cov.split(',') if s.strip()]
                                elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                                    try:
                                        syms = (await reference_svc.get_watchlist_symbols()) or []
                                    except Exception:
                                        syms = []
                                if not syms:
                                    syms = ['AAPL','MSFT','TSLA','NVDA','SPY']
                                syms = syms[:max(1, max_underlyings)]

                                # compute coverage
                                out = []
                                for u in syms:
                                    try:
                                        sql_summary = (
                                            "select count() as rows, count_distinct(option_symbol) as contracts, "
                                            "min(timestamp) as first_ts, "
                                            "max(timestamp) as last_ts "
                                            f"from options_data where underlying = '{u}'"
                                        )
                                        data = await _q(session, sql_summary)
                                        if not data.get('dataset'):
                                            out.append({"underlying": u, "rows": 0, "contracts": 0, "first_day": None, "last_day": None, "recent_gap_days_30d": None})
                                            continue
                                        r = data['dataset'][0]
                                        cols = {c['name']: i for i, c in enumerate(data.get('columns', []))}
                                        rows = int(r[cols['rows']]) if 'rows' in cols else 0
                                        contracts = int(r[cols['contracts']]) if 'contracts' in cols else 0
                                        # Timestamps are ISO strings; format to YYYY-MM-DD
                                        def _fmt_iso_day(v):
                                            try:
                                                return str(v)[:10]
                                            except Exception:
                                                return None
                                        first_day = _fmt_iso_day(r[cols['first_ts']]) if 'first_ts' in cols else None
                                        last_day = _fmt_iso_day(r[cols['last_ts']]) if 'last_ts' in cols else None
                                        sql_recent = (
                                            "select count_distinct(cast(timestamp as LONG)/86400000000) as have_days "
                                            f"from options_data where underlying = '{u}' and timestamp >= dateadd('d', -30, now())"
                                        )
                                        d2 = await _q(session, sql_recent)
                                        have_days = 0
                                        if d2.get('dataset'):
                                            c2 = {c['name']: i for i, c in enumerate(d2.get('columns', []))}
                                            try:
                                                have_days = int(d2['dataset'][0][c2['have_days']])
                                            except Exception:
                                                have_days = 0
                                        out.append({
                                            "underlying": u,
                                            "rows": rows,
                                            "contracts": contracts,
                                            "first_day": first_day,
                                            "last_day": last_day,
                                            "recent_gap_days_30d": max(0, 30 - have_days),
                                        })
                                    except Exception as e:  # noqa: BLE001
                                        out.append({"underlying": u, "error": str(e)})

                                # write JSON artifact
                                try:
                                    os.makedirs(out_dir, exist_ok=True)
                                    payload = {"generated_at": datetime.utcnow().isoformat(), "questdb": qdb_url, "coverage": out}
                                    # Stable filename for dashboard
                                    stable_path = os.path.join(out_dir, 'options_coverage.json')
                                    # Dated filename for audit/trend
                                    date_path = os.path.join(out_dir, f"options_coverage_{datetime.utcnow().strftime('%Y%m%d')}.json")
                                    import json
                                    with open(stable_path, 'w') as f:
                                        json.dump(payload, f, indent=2)
                                    with open(date_path, 'w') as f:
                                        json.dump(payload, f, indent=2)
                                    logger.info("Options coverage report written", path=stable_path, items=len(out))
                                except Exception as e:  # noqa: BLE001
                                    logger.warning("Failed writing options coverage report: %s", e)
                                # Update last_run for coverage loop
                                try:
                                    LOOP_STATUS['options_coverage']["last_run"] = datetime.utcnow().isoformat()
                                    if LOOP_LAST_RUN_UNIX is not None:
                                        LOOP_LAST_RUN_UNIX.labels(loop='options_coverage').set(time.time())
                                except Exception:
                                    pass
                            except Exception as e:  # noqa: BLE001
                                logger.warning("Options coverage report loop error: %s", e)
                                LOOP_STATUS['options_coverage']["last_error"] = str(e)

                asyncio.create_task(_coverage_loop())
                logger.info("Options coverage report loop started (feature flag ENABLE_OPTIONS_COVERAGE_REPORT)")

            # MinIO artifact upload loop (feature gated)
            if os.getenv("ENABLE_MINIO_ARTIFACT_UPLOAD", "false").lower() in ("1","true","yes"):
                try:
                    upload_interval = int(os.getenv("ARTIFACT_UPLOAD_INTERVAL_SECONDS", "21600"))  # 6h default
                except Exception:
                    upload_interval = 21600

                async def _artifact_upload_loop():
                    while True:
                        try:
                            out_dir = os.getenv("GRAFANA_EXPORT_DIR", "/app/export/grafana-csv").rstrip("/")
                            bucket = os.getenv("MINIO_ARTIFACTS_BUCKET", "trading")
                            prefix = os.getenv("MINIO_ARTIFACTS_PREFIX", "dashboards")
                            try:
                                # Use internal helper to avoid code duplication
                                await minio_upload_artifacts(directory=out_dir, bucket=bucket, prefix=prefix, pattern="*.json")
                                logger.info("Artifacts uploaded to MinIO", bucket=bucket, prefix=prefix)
                            except Exception as e:  # noqa: BLE001
                                logger.warning("Artifact upload failed: %s", e)
                            LOOP_STATUS.setdefault('artifact_upload', {"enabled": True, "interval_seconds": None, "last_run": None, "last_error": None})
                            LOOP_STATUS['artifact_upload']["enabled"] = True
                            LOOP_STATUS['artifact_upload']["interval_seconds"] = max(300, upload_interval)
                            LOOP_STATUS['artifact_upload']["last_run"] = datetime.utcnow().isoformat()
                            try:
                                if LOOP_LAST_RUN_UNIX is not None:
                                    LOOP_LAST_RUN_UNIX.labels(loop='artifact_upload').set(time.time())
                            except Exception:
                                pass
                            await asyncio.sleep(max(300, upload_interval))
                        except Exception as e:  # noqa: BLE001
                            LOOP_STATUS['artifact_upload']["last_error"] = str(e)
                            await asyncio.sleep(600)

                asyncio.create_task(_artifact_upload_loop())
                logger.info("MinIO artifact upload loop started (feature flag ENABLE_MINIO_ARTIFACT_UPLOAD)")

            # Analytics persistence loop (system/trading KPIs -> QuestDB)
            if os.getenv("ENABLE_ANALYTICS_PERSIST", "false").lower() in ("1","true","yes"):
                try:
                    analytics_interval = int(os.getenv("ANALYTICS_PERSIST_INTERVAL_SECONDS", "300"))
                except Exception:
                    analytics_interval = 300

                async def _analytics_persist_loop():
                    while True:
                        try:
                            LOOP_STATUS.setdefault('analytics_persist', {"enabled": True, "interval_seconds": None, "last_run": None, "last_error": None})
                            LOOP_STATUS['analytics_persist']["enabled"] = True
                            LOOP_STATUS['analytics_persist']["interval_seconds"] = max(60, analytics_interval)
                            ts_now = datetime.utcnow().isoformat()

                            # Build system metrics from loop status and provider flags
                            sys_rows: list[dict] = []
                            try:
                                for lname, meta in (LOOP_STATUS or {}).items():
                                    try:
                                        sys_rows.append({
                                            'component': f'loop:{lname}',
                                            'metric': 'enabled',
                                            'value': 1.0 if meta.get('enabled') else 0.0,
                                            'str_value': None,
                                        })
                                        last = meta.get('last_run')
                                        if last:
                                            try:
                                                age = (datetime.utcnow() - datetime.fromisoformat(str(last).replace('Z','+00:00'))).total_seconds()
                                                sys_rows.append({'component': f'loop:{lname}', 'metric': 'last_run_age_seconds', 'value': float(max(0.0, age)), 'str_value': None})
                                            except Exception:
                                                pass
                                    except Exception:
                                        continue
                                prov_flags = {
                                    'eodhd_key': 1.0 if os.getenv('EODHD_API_KEY') else 0.0,
                                    'alpha_vantage_key': 1.0 if (os.getenv('ALPHAVANTAGE_API_KEY') or os.getenv('ALPHA_VANTAGE_API_KEY')) else 0.0,
                                    'polygon_key': 1.0 if os.getenv('POLYGON_API_KEY') else 0.0,
                                    'alpaca_key': 1.0 if os.getenv('ALPACA_API_KEY') else 0.0,
                                }
                                for k, v in prov_flags.items():
                                    sys_rows.append({'component': 'providers', 'metric': k, 'value': v, 'str_value': None})
                            except Exception:
                                pass

                            # Key dataset coverage counts (rows) from QuestDB
                            trade_rows: list[dict] = []
                            try:
                                async def _count(table: str) -> int:
                                    try:
                                        data = await _qdb_exec(f"select count() as c from {table}", timeout=8.0)
                                        cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
                                        if data.get('dataset'):
                                            return int(data['dataset'][0][cols.get('c', 0)])
                                    except Exception:
                                        return 0
                                    return 0
                                md = await _count('market_data')
                                od = await _count('options_data')
                                nw = await _count('news_items')
                                so = await _count('social_signals')
                                for metric, val in (
                                    ('market_data_rows', md),
                                    ('options_data_rows', od),
                                    ('news_items_rows', nw),
                                    ('social_signals_rows', so),
                                ):
                                    trade_rows.append({'symbol': '*', 'metric': metric, 'value': float(val), 'str_value': None})
                            except Exception:
                                pass

                            # Persist via QuestDB /exec INSERTs (best-effort)
                            try:
                                for r in sys_rows:
                                    sval = 'null' if r.get('str_value') is None else "'" + str(r['str_value']).replace("'", "''") + "'"
                                    q = (
                                        "INSERT INTO system_analytics (ts, component, metric, value, str_value) VALUES ("
                                        f"now(), '{r['component']}', '{r['metric']}', {r.get('value') or 0.0}, {sval})"
                                    )
                                    try:
                                        await _qdb_exec(q, timeout=5.0)
                                    except Exception:
                                        pass
                                for r in trade_rows:
                                    sval = 'null' if r.get('str_value') is None else "'" + str(r['str_value']).replace("'", "''") + "'"
                                    q = (
                                        "INSERT INTO trading_analytics (ts, symbol, metric, value, str_value) VALUES ("
                                        f"now(), '{r.get('symbol') or '*'}', '{r['metric']}', {r.get('value') or 0.0}, {sval})"
                                    )
                                    try:
                                        await _qdb_exec(q, timeout=5.0)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            LOOP_STATUS['analytics_persist']["last_run"] = datetime.utcnow().isoformat()
                            LOOP_STATUS['analytics_persist']["last_error"] = None
                            try:
                                if LOOP_LAST_RUN_UNIX is not None:
                                    LOOP_LAST_RUN_UNIX.labels(loop='analytics_persist').set(time.time())
                            except Exception:
                                pass
                            await asyncio.sleep(max(60, analytics_interval))
                        except Exception as e:  # noqa: BLE001
                            LOOP_STATUS['analytics_persist']["last_error"] = str(e)[:200]
                            await asyncio.sleep(max(60, analytics_interval))

                asyncio.create_task(_analytics_persist_loop())
                logger.info("Analytics persistence loop started (feature flag ENABLE_ANALYTICS_PERSIST)")
            else:
                logger.info("Analytics persistence loop disabled (ENABLE_ANALYTICS_PERSIST=false)")

            # Housekeeping loop: archive and prune old JSON artifacts to MinIO (feature gated)
            if os.getenv("ENABLE_HOUSEKEEPING", "false").lower() in ("1","true","yes"):
                try:
                    hk_interval = int(os.getenv("HOUSEKEEPING_INTERVAL_SECONDS", "21600"))  # 6h default
                except Exception:
                    hk_interval = 21600
                try:
                    hk_age_days = int(os.getenv("HOUSEKEEPING_AGE_DAYS", "30"))
                except Exception:
                    hk_age_days = 30
                hk_bucket = os.getenv("HOUSEKEEPING_BUCKET", "trading").strip()
                hk_prefix = os.getenv("HOUSEKEEPING_PREFIX", "archives/ops-json").strip().strip("/")
                # Comma-separated absolute paths to scan
                hk_paths = [p.strip() for p in os.getenv("HOUSEKEEPING_PATHS", "/srv,/srv/ai-trading-system").split(",") if p.strip()]
                # Comma-separated glob patterns to match
                hk_patterns = [p.strip() for p in os.getenv("HOUSEKEEPING_PATTERNS", "*.json").split(",") if p.strip()]

                async def _housekeeping_loop():
                    # Archive matching files older than N days to MinIO, then remove locally
                    import time as _t
                    from datetime import datetime as _dt
                    while True:
                        try:
                            LOOP_STATUS['housekeeping']["enabled"] = True
                            LOOP_STATUS['housekeeping']["interval_seconds"] = max(300, hk_interval)
                            cutoff_ts = _t.time() - hk_age_days * 86400
                            attempted = 0
                            uploaded = 0
                            removed = 0
                            errors = 0
                            keys = []
                            # Build file list
                            files: list[str] = []
                            try:
                                import glob as _glob
                                for base in hk_paths:
                                    for pat in hk_patterns:
                                        try:
                                            files.extend(_glob.glob(os.path.join(base, pat)))
                                            # include subdirs conservatively
                                            files.extend(_glob.glob(os.path.join(base, "**", pat), recursive=True))
                                        except Exception:
                                            continue
                            except Exception:
                                files = []
                            # Filter by age
                            cand = []
                            for f in sorted(set(files)):
                                try:
                                    st = os.stat(f)
                                    if st and st.st_mtime < cutoff_ts and os.path.isfile(f):
                                        cand.append((f, int(st.st_mtime)))
                                except Exception:
                                    continue
                            # Upload to MinIO using existing helper
                            if cand:
                                try:
                                    os.makedirs("/tmp", exist_ok=True)
                                except Exception:
                                    pass
                                # Upload one-by-one to preserve original filenames
                                for (path, mtime) in cand:
                                    attempted += 1
                                    # Prefix dated subfolder e.g., archives/ops-json/2025/09/
                                    try:
                                        dt = _dt.utcfromtimestamp(mtime)
                                        dated_prefix = f"{hk_prefix}/{dt.year:04d}/{dt.month:02d}"
                                    except Exception:
                                        dated_prefix = hk_prefix
                                    try:
                                        res = await minio_upload_artifacts(directory=os.path.dirname(path), bucket=hk_bucket, prefix=dated_prefix, pattern=os.path.basename(path))
                                        if res.get('uploaded'):
                                            uploaded += res.get('uploaded', 0)
                                            if res.get('keys'):
                                                keys.extend(res['keys'])
                                            # Safe remove after upload
                                            try:
                                                os.remove(path)
                                                removed += 1
                                            except Exception:
                                                pass
                                        else:
                                            errors += 1
                                    except Exception:
                                        errors += 1
                            # Update loop status
                            LOOP_STATUS['housekeeping']["last_run"] = _dt.utcnow().isoformat()
                            try:
                                if LOOP_LAST_RUN_UNIX is not None:
                                    LOOP_LAST_RUN_UNIX.labels(loop='housekeeping').set(_t.time())
                            except Exception:
                                pass
                            logger.info("Housekeeping run", attempted=attempted, uploaded=uploaded, removed=removed, errors=errors, examples=keys[:3])
                            await asyncio.sleep(max(300, hk_interval))
                        except Exception as e:  # noqa: BLE001
                            LOOP_STATUS['housekeeping']["last_error"] = str(e)
                            await asyncio.sleep(900)

                asyncio.create_task(_housekeeping_loop())
                logger.info("Housekeeping loop started (feature flag ENABLE_HOUSEKEEPING)")

            # Housekeeping loop: archive and prune old JSON artifacts to MinIO (best-effort)
            if os.getenv("ENABLE_HOUSEKEEPING", "false").lower() in ("1","true","yes"):
                try:
                    hk_interval = int(os.getenv("HOUSEKEEPING_INTERVAL_SECONDS", "21600"))
                except Exception:
                    hk_interval = 21600
                try:
                    hk_age_days = int(os.getenv("HOUSEKEEPING_AGE_DAYS", "30"))
                except Exception:
                    hk_age_days = 30
                hk_bucket = os.getenv("HOUSEKEEPING_BUCKET", "trading").strip()
                hk_prefix = os.getenv("HOUSEKEEPING_PREFIX", "archives/ops-json").strip().strip("/")
                # Comma-separated roots and patterns
                hk_paths = [p.strip() for p in (os.getenv("HOUSEKEEPING_PATHS", "/srv,/srv/ai-trading-system") or "").split(",") if p.strip()]
                hk_patterns = [p.strip() for p in (os.getenv("HOUSEKEEPING_PATTERNS", "coverage_snapshot*.json,coverage_snapshot_full*.json,coverage_run_latest.json,coverage_summary_consolidated.json,retention_*_questdb.json,*_seed_report*.json,*_seed_checkpoint*.json,backfill_progress.json,storage_projection_*.json") or "").split(",") if p.strip()]

                async def _housekeeping_loop():
                    while True:
                        try:
                            cutoff = datetime.utcnow() - timedelta(days=max(1, hk_age_days))
                            total_found = 0
                            archived = 0
                            deleted = 0
                            # Find files matching patterns older than cutoff
                            for root in hk_paths:
                                for pat in hk_patterns:
                                    try:
                                        for path in glob.glob(os.path.join(root, pat)):
                                            try:
                                                st = os.stat(path)
                                                mtime = datetime.utcfromtimestamp(st.st_mtime)
                                                if mtime <= cutoff:
                                                    total_found += 1
                                                    # Upload to MinIO under prefix/YYYYMM/filename
                                                    ym = datetime.utcnow().strftime('%Y%m')
                                                    key_prefix = f"{hk_prefix}/{ym}" if hk_prefix else ym
                                                    # Reuse uploader for a single file by passing directory and filtering later
                                                    # Here we perform direct upload via uploader after copying to tmp list
                                                    try:
                                                        # Minimal inline uploader call: symlink by copying file into a temp dir is heavy; instead, call uploader on dir and filter would upload too much.
                                                        # Implement a tiny direct path using boto3/minio if available.
                                                        res = await minio_upload_artifacts(directory=os.path.dirname(path), bucket=hk_bucket, prefix=key_prefix, pattern=os.path.basename(path))
                                                        if res.get('uploaded', 0) >= 1:
                                                            archived += 1
                                                            # Remove the local file after successful upload
                                                            try:
                                                                os.remove(path)
                                                                deleted += 1
                                                            except Exception:
                                                                pass
                                                    except Exception:
                                                        continue
                                            except Exception:
                                                continue
                                    except Exception:
                                        continue
                            logger.info("Housekeeping run completed", found=total_found, archived=archived, deleted=deleted)
                        except Exception as e:
                            logger.warning("Housekeeping loop error: %s", e)
                        # Sleep until next interval
                        await asyncio.sleep(max(300, hk_interval))

                asyncio.create_task(_housekeeping_loop())
                logger.info("Housekeeping loop started (feature flag ENABLE_HOUSEKEEPING)")

            # Optional one-time equities historical backfill on startup (feature gated)
            if os.getenv("ENABLE_EQUITY_BACKFILL_ON_START", "false").lower() in ("1","true","yes"):
                try:
                    years = int(os.getenv("EQUITY_BACKFILL_YEARS", "20"))
                except Exception:
                    years = 20
                try:
                    max_symbols = int(os.getenv("EQUITY_BACKFILL_MAX_SYMBOLS", "0"))
                    if max_symbols <= 0:
                        max_symbols = None  # No limit - process all watchlist symbols
                except Exception:
                    max_symbols = None  # No limit on error
                try:
                    pacing = float(os.getenv("EQUITY_BACKFILL_PACING_SECONDS", "0.2"))
                except Exception:
                    pacing = 0.2
                symbols_env = os.getenv("EQUITY_BACKFILL_SYMBOLS", "").strip()

                async def _equities_backfill_once():
                    # Run once per process start to seed historical data
                    if not market_data_svc:
                        return
                    LOOP_STATUS['equities_backfill']["enabled"] = True
                    LOOP_STATUS['equities_backfill']["interval_seconds"] = None
                    try:
                        # Resolve symbols
                        syms: list[str] = []
                        if symbols_env:
                            syms = [s.strip().upper() for s in symbols_env.split(',') if s.strip()]
                        elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                            syms = (await reference_svc.get_watchlist_symbols()) or []
                        if not syms:
                            syms = ['AAPL','MSFT','TSLA','NVDA','SPY']
                        # Process all symbols if max_symbols is None, otherwise limit
                        if max_symbols is not None:
                            syms = syms[:max(1, max_symbols)]

                        end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                        start_dt = end_dt - timedelta(days=int(years * 365.25))
                        total = len(syms)
                        done = 0
                        bars_total = 0
                        errs = 0
                        for sym in syms:
                            try:
                                rows = await market_data_svc.get_bulk_daily_historical(sym, start_dt, end_dt)
                                try:
                                    bars_total += int(len(rows or []))
                                except Exception:
                                    pass
                            except Exception as e:  # noqa: BLE001
                                errs += 1
                                logger.warning("Equities backfill failed", symbol=sym, error=str(e))
                            done += 1
                            # Update progress gauge (percent of symbols processed)
                            try:
                                pct = (done / total) * 100.0 if total else 0.0
                                set_backfill_progress(pct)
                            except Exception:
                                pass
                            await asyncio.sleep(max(0.0, pacing))
                        LOOP_STATUS['equities_backfill']["last_run"] = datetime.utcnow().isoformat()
                        try:
                            if LOOP_LAST_RUN_UNIX is not None:
                                LOOP_LAST_RUN_UNIX.labels(loop='equities_backfill').set(time.time())
                        except Exception:
                            pass
                        logger.info(
                            "Equities historical backfill completed",
                            total_symbols=total,
                            errors=errs,
                            bars_ingested=bars_total,
                            years=years,
                        )
                    except Exception as e:  # noqa: BLE001
                        LOOP_STATUS['equities_backfill']["last_error"] = str(e)
                        logger.warning("Equities backfill loop error: %s", e)

                asyncio.create_task(_equities_backfill_once())
                logger.info("Equities backfill task scheduled (ENABLE_EQUITY_BACKFILL_ON_START)")

            # Equities coverage scheduled loop (feature gated)
            if os.getenv("ENABLE_EQUITIES_COVERAGE_REPORT", "false").lower() in ("1","true","yes"):
                try:
                    eq_interval = int(os.getenv("EQUITIES_COVERAGE_INTERVAL_SECONDS", "86400"))
                except Exception:
                    eq_interval = 86400
                out_dir = os.getenv("EQUITIES_COVERAGE_OUTPUT_DIR", "/app/export/grafana-csv").rstrip("/")

                async def _equities_coverage_loop():
                    while True:
                        try:
                            LOOP_STATUS['equities_coverage']["enabled"] = True
                            LOOP_STATUS['equities_coverage']["interval_seconds"] = max(300, eq_interval)
                            # Full scan (sample=false) to ensure metrics refresh
                            try:
                                # Reuse endpoint helper for equities coverage computation (module-level function)
                                # Note: equities_coverage returns symbols_evaluated and ratio; items only when sample=True.
                                report = await equities_coverage(sample=False, min_years=19.5)
                                # Clear any prior error on success
                                LOOP_STATUS['equities_coverage']["last_error"] = None
                            except Exception as e:  # noqa: BLE001
                                LOOP_STATUS['equities_coverage']["last_error"] = str(e)
                                await asyncio.sleep(max(300, eq_interval))
                                continue
                            # Persist stable JSON artifact
                            try:
                                os.makedirs(out_dir, exist_ok=True)
                                import json
                                stable = os.path.join(out_dir, 'equities_coverage.json')
                                dated = os.path.join(out_dir, f"equities_coverage_{datetime.utcnow().strftime('%Y%m%d')}.json")
                                with open(stable, 'w') as f:
                                    json.dump(report, f, indent=2)
                                with open(dated, 'w') as f:
                                    json.dump(report, f, indent=2)
                                logger.info("Equities coverage report written", path=stable, symbols=report.get('symbols_evaluated'))
                            except Exception as e:  # noqa: BLE001
                                logger.warning("Failed writing equities coverage report: %s", e)
                            LOOP_STATUS['equities_coverage']["last_run"] = datetime.utcnow().isoformat()
                            try:
                                if LOOP_LAST_RUN_UNIX is not None:
                                    LOOP_LAST_RUN_UNIX.labels(loop='equities_coverage').set(time.time())
                            except Exception:
                                pass
                            await asyncio.sleep(max(300, eq_interval))
                        except Exception as e:  # noqa: BLE001
                            logger.warning("Equities coverage loop error: %s", e)
                            LOOP_STATUS['equities_coverage']["last_error"] = str(e)
                            await asyncio.sleep(600)
                asyncio.create_task(_equities_coverage_loop())
                logger.info("Equities coverage report loop started (feature flag ENABLE_EQUITIES_COVERAGE_REPORT)")

            # Weaviate reconciliation loop: periodically index recent QuestDB news into vector store
            if os.getenv("ENABLE_WEAVIATE_RECONCILE", "false").lower() in ("1","true","yes"):
                try:
                    rec_interval = int(os.getenv("WEAVIATE_RECONCILE_INTERVAL_SECONDS", "900"))  # default 15 min
                except Exception:
                    rec_interval = 900
                try:
                    rec_days = int(os.getenv("WEAVIATE_RECONCILE_LOOKBACK_DAYS", "3"))
                except Exception:
                    rec_days = 3
                try:
                    rec_limit = int(os.getenv("WEAVIATE_RECONCILE_LIMIT", "2000"))
                except Exception:
                    rec_limit = 2000

                async def _vector_reconcile_loop():
                    while True:
                        try:
                            LOOP_STATUS['vector_reconcile']["enabled"] = True
                            LOOP_STATUS['vector_reconcile']["interval_seconds"] = max(120, rec_interval)
                            # Query recent news from QuestDB and index via direct fallback
                            try:
                                # Discover available columns to build a compatible projection
                                try:
                                    meta = await _qdb_exec("show columns from news_items")
                                    name_idx = next((i for i, c in enumerate(meta.get('columns', []) or []) if c.get('name') == 'column'), None)
                                    cols_available: list[str] = []
                                    for r in meta.get('dataset') or []:
                                        try:
                                            if name_idx is not None:
                                                cols_available.append(str(r[name_idx]))
                                        except Exception:
                                            continue
                                except Exception:
                                    cols_available = []

                                # Minimal required fields are title and ts; add optional ones when present
                                proj_parts = ["title", "ts"]
                                for c in ("source", "url", "symbol", "sentiment", "relevance", "provider", "value_score"):
                                    if c in cols_available and c not in proj_parts:
                                        proj_parts.append(c)
                                select_list = ", ".join(proj_parts)
                                look_sql = (
                                    f"select {select_list} from news_items "
                                    f"where ts >= dateadd('d', -{max(1, rec_days)}, now()) "
                                    "and title is not null and title != '' "
                                    "order by ts desc limit " + str(max(1, rec_limit))
                                )
                                data = await _qdb_exec(look_sql, timeout=20.0)
                            except Exception as e:  # noqa: BLE001
                                LOOP_STATUS['vector_reconcile']["last_error"] = str(e)
                                await asyncio.sleep(max(120, rec_interval))
                                continue
                            cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
                            items = []
                            last_ts_val = None
                            for r in (data.get('dataset') or [])[:rec_limit]:
                                try:
                                    title = str(r[cols['title']]) if 'title' in cols else ''
                                    # content is not stored in current schema; leave blank
                                    content = ''
                                    source = str(r[cols.get('source')]) if 'source' in cols and r[cols.get('source')] is not None else 'news_items'
                                    ts = r[cols.get('ts')] if 'ts' in cols else None
                                    symbol = r[cols.get('symbol')] if 'symbol' in cols else None
                                    if ts is not None:
                                        try:
                                            last_ts_val = ts
                                        except Exception:
                                            pass
                                    items.append({
                                        'title': title,
                                        'content': content,
                                        'source': source,
                                        'published_at': str(ts) if ts is not None else datetime.utcnow().isoformat(),
                                        'symbols': [str(symbol).upper()] if symbol else []
                                    })
                                except Exception:
                                    continue
                            indexed = 0
                            if items:
                                try:
                                    # Lazy import fallback indexer
                                    try:
                                        from shared.vector.indexing import index_news_fallback  # type: ignore
                                    except Exception:
                                        from ..shared.vector.indexing import index_news_fallback  # type: ignore
                                    indexed = await index_news_fallback(items)
                                except Exception:
                                    indexed = 0
                            try:
                                if indexed and VECTOR_NEWS_INDEXED is not None:
                                    VECTOR_NEWS_INDEXED.labels(path='reconcile_loop').inc(indexed)
                            except Exception:
                                pass
                            try:
                                if last_ts_val and VECTOR_NEWS_LAST_TS is not None:
                                    # best-effort parse to epoch seconds
                                    import time as _t
                                    try:
                                        # If ISO string
                                        from datetime import datetime as _dt
                                        if isinstance(last_ts_val, str):
                                            _dtv = _dt.fromisoformat(last_ts_val.replace('Z','+00:00'))
                                            VECTOR_NEWS_LAST_TS.set(_dtv.timestamp())
                                        elif hasattr(last_ts_val, 'timestamp'):
                                            VECTOR_NEWS_LAST_TS.set(last_ts_val.timestamp())
                                    except Exception:
                                        VECTOR_NEWS_LAST_TS.set(_t.time())
                            except Exception:
                                pass
                            LOOP_STATUS['vector_reconcile']["last_run"] = datetime.utcnow().isoformat()
                            try:
                                if LOOP_LAST_RUN_UNIX is not None:
                                    LOOP_LAST_RUN_UNIX.labels(loop='vector_reconcile').set(time.time())
                            except Exception:
                                pass
                            await asyncio.sleep(max(120, rec_interval))
                        except Exception as e:  # noqa: BLE001
                            LOOP_STATUS['vector_reconcile']["last_error"] = str(e)
                            await asyncio.sleep(max(300, rec_interval))

                asyncio.create_task(_vector_reconcile_loop())
                logger.info("Weaviate reconciliation loop started (feature flag ENABLE_WEAVIATE_RECONCILE)")

            # Auto-refresh watchlist from Polygon (daily discovery of new optionable symbols)
            # Watchlist auto-refresh disabled - use manual discovery script instead
            # The full options discovery can take 20+ minutes due to Polygon rate limits
            # Run: docker exec trading-data-ingestion python services/data_ingestion/options_symbol_discovery.py --sync-watchlist
            if os.getenv("ENABLE_WATCHLIST_AUTO_REFRESH", "false").lower() in ("1","true","yes"):
                async def _watchlist_refresh_loop():
                    """Daily refresh using lightweight method - full discovery should be run offline."""
                    refresh_interval = int(os.getenv("WATCHLIST_REFRESH_INTERVAL_SECONDS", str(24*3600)))  # Default: daily
                    LOOP_STATUS.setdefault('watchlist_refresh', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": refresh_interval})
                    LOOP_STATUS['watchlist_refresh']["enabled"] = True
                    LOOP_STATUS['watchlist_refresh']["interval_seconds"] = refresh_interval
                    
                    while True:
                        try:
                            if not reference_svc:
                                await asyncio.sleep(60)
                                continue
                            
                            logger.info("Starting lightweight watchlist refresh (top liquid symbols only)")
                            # Use lightweight method for automated refresh
                            count = await reference_svc.populate_watchlist_from_polygon(
                                locale='us',
                                market='stocks',
                                active=True,
                                types='CS',  # Common stock only
                                page_limit=1000,
                                max_pages=2  # Limit to ~2000 symbols for performance
                            )
                            logger.info(f"Watchlist refreshed (lightweight): {count} symbols", source='polygon')
                            LOOP_STATUS['watchlist_refresh']["last_run"] = datetime.utcnow().isoformat()
                            LOOP_STATUS['watchlist_refresh']["last_error"] = None
                            LOOP_STATUS['watchlist_refresh']["last_count"] = count
                            try:
                                if LOOP_LAST_RUN_UNIX:
                                    LOOP_LAST_RUN_UNIX.labels(loop='watchlist_refresh').set(time.time())
                            except Exception:
                                pass
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Watchlist refresh failed: {e}")
                            LOOP_STATUS['watchlist_refresh']["last_error"] = str(e)
                        
                        await asyncio.sleep(refresh_interval)
                
                asyncio.create_task(_watchlist_refresh_loop())
                logger.info("Automated lightweight watchlist refresh enabled (ENABLE_WATCHLIST_AUTO_REFRESH - set to false by default)")

        # Fire-and-forget optional component initialization so startup isn't blocked
        asyncio.create_task(_init_optional_components())
        logger.info(
            "Optional component initialization running in background; service starting immediately",
            timeout_strategy="fire-and-forget"
        )

        # Ensure market_data_svc begins initialization as early as possible to populate vendor metrics
        try:
            asyncio.create_task(get_market_data_service())
            logger.info("Scheduled early market_data_service initialization task")
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to schedule early market_data_service init", error=str(e))

    except Exception as e:  # noqa: BLE001
        logger.error(f"Unexpected startup error: {e}")

    yield

    # Cleanup (best-effort)
    try:
        if cache_client and hasattr(cache_client, "close"):
            await cache_client.close()
        if redis_client and hasattr(redis_client, "close"):
            await redis_client.close()
        for svc in [market_data_svc, news_svc, reference_svc, validation_svc, retention_svc]:
            if svc and hasattr(svc, "stop"):
                try:
                    await svc.stop()
                except Exception:  # noqa: BLE001
                    pass
        logger.info("Data Ingestion Service stopped")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Error during shutdown cleanup: {e}")


app = FastAPI(
    title="AI Trading System - Data Ingestion Service",
    description="Handles real-time market data, news, and social sentiment collection",
    version="1.0.0-dev",
    lifespan=lifespan
)

# Ensure loop enablement flags are visible in extended health even before first run
@app.on_event("startup")
async def _mark_enabled_loops():
    try:
        if str(os.getenv('ENABLE_WEAVIATE_RECONCILE', 'false')).lower() in ('1','true','yes','on'):
            LOOP_STATUS.setdefault('vector_reconcile', {"enabled": False, "last_run": None, "last_error": None, "interval_seconds": None})
            LOOP_STATUS['vector_reconcile']["enabled"] = True
            try:
                LOOP_STATUS['vector_reconcile']["interval_seconds"] = int(os.getenv('WEAVIATE_RECONCILE_INTERVAL_SECONDS', '900'))
            except Exception:
                LOOP_STATUS['vector_reconcile']["interval_seconds"] = None
    except Exception:
        pass

# ---------------------- Liveness/Readiness Endpoints ---------------------- #
@app.get("/health")
async def health():
    """Simple liveness endpoint for container healthcheck."""
    # Basic signals: service started and logger available
    try:
        started = _START_TIME.isoformat()
    except Exception:
        started = None  # best-effort
    return {"status": "ok", "service": "data-ingestion", "started": started}


@app.get("/healthz")
async def healthz():
    """Alias for liveness (some probes prefer /healthz)."""
    return {"status": "ok", "service": "data-ingestion"}


@app.get("/ready")
async def ready():
    """Lightweight readiness check with core dependencies only.

    Criteria: cache and redis clients are initialized. Optional components are
    not required for readiness to avoid flapping during long initializations.
    """
    components = {
        "cache": cache_client is not None,
        "redis": redis_client is not None,
    }
    degraded = [k + "_unavailable" for k, v in components.items() if not v]
    ok = not degraded
    return JSONResponse(status_code=200 if ok else 503, content={
        "service": "data-ingestion",
        "status": "ready" if ok else "degraded",
        "components": components,
        "degraded_reasons": degraded or None,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.get("/health/extended")
async def health_extended():
    """Extended health with pipeline and provider snapshots used by API aggregator.

    Returns a subset that is cheap to compute to avoid timeouts while the service is busy.
    """
    try:
        # Pipelines best-effort snapshot from LOOP_STATUS
        pipelines = {}
        now = time.time()
        for name, info in (LOOP_STATUS or {}).items():
            last_run_iso = info.get("last_run")
            # Attempt to compute seconds since last run if ISO present
            age = None
            try:
                if last_run_iso:
                    from datetime import datetime as _dt
                    age = max(0, int(now - _dt.fromisoformat(last_run_iso.replace('Z','+00:00')).timestamp()))
            except Exception:
                age = None
            pipelines[name] = {
                "enabled": bool(info.get("enabled")),
                "last_run": last_run_iso,
                "last_error": info.get("last_error"),
                "interval_seconds": info.get("interval_seconds"),
                "seconds_since_last": age,
                # Placeholder counters; populated by loops where available
                "success_total": info.get("success_total"),
                "error_total": info.get("error_total"),
            }

        provider_metrics = {}
        try:
            # Expose minimal provider rate-limit counters if available
            if PROVIDER_HTTP_RESPONSES_TOTAL is not None:
                provider_metrics["http_responses_metric"] = True
            if PROVIDER_RATE_LIMIT_TOTAL is not None:
                provider_metrics["rate_limit_metric"] = True
        except Exception:
            provider_metrics = {}

        return JSONResponse({
            "status": "ok",
            "ingestion_pipelines": pipelines,
            "ingestion_errors_aggregated": {},  # placeholder; detailed aggregation done in loops
            "provider_metrics": provider_metrics,
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)[:200]})

# ---------------------- Stream Controls and Status ---------------------- #
@app.get("/streams/status")
async def streams_status():
    """Returns lightweight status for primary ingestion loops (including quotes)."""
    try:
        out = {}
        for name in ("quote_stream","news_stream","social_stream","daily_delta","daily_options","options_coverage","vector_reconcile","equities_backfill","equities_coverage","artifact_upload","housekeeping"):
            if name in LOOP_STATUS:
                out[name] = {
                    "enabled": bool(LOOP_STATUS[name].get("enabled")),
                    "last_run": LOOP_STATUS[name].get("last_run"),
                    "last_error": LOOP_STATUS[name].get("last_error"),
                    "interval_seconds": LOOP_STATUS[name].get("interval_seconds"),
                }
        # Include current gating/symbols overrides for quotes
        out["quote_stream_overrides"] = {
            "gating_override": QUOTE_STREAM_GATING_OVERRIDE,
            "symbols_override": QUOTE_STREAM_SYMBOLS_OVERRIDE,
            "env_enabled": os.getenv("ENABLE_QUOTE_STREAM", "false").lower() in ("1","true","yes"),
            "env_hours_only": os.getenv("QUOTE_STREAM_TRADING_HOURS_ONLY", "true").lower() in ("1","true","yes"),
        }
        return out
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

class QuoteStreamOverrideRequest(BaseModel):
    gating: Optional[bool] = None  # when True, run only during trading hours; when False, run now regardless of hours
    symbols: Optional[List[str]] = None  # if provided, restrict streaming to these symbols

@app.post("/streams/quotes/override")
async def override_quote_stream(req: QuoteStreamOverrideRequest):
    """Set runtime overrides for the quote stream (hours gating and/or symbol list).

    Notes:
      - The quote stream loop starts only when ENABLE_QUOTE_STREAM=true at startup.
      - This endpoint lets you disable trading-hours gating (to run now) and/or set a specific symbols list.
    """
    global QUOTE_STREAM_GATING_OVERRIDE, QUOTE_STREAM_SYMBOLS_OVERRIDE
    try:
        if req.gating is not None:
            QUOTE_STREAM_GATING_OVERRIDE = bool(req.gating)
        if req.symbols is not None:
            symbols = [str(s).strip().upper() for s in req.symbols if str(s).strip()]
            QUOTE_STREAM_SYMBOLS_OVERRIDE = symbols if symbols else None
        # Ensure loop is running even if env flag was false
        try:
            global QUOTE_LOOP_TASK
            if QUOTE_LOOP_TASK is None or QUOTE_LOOP_TASK.done():
                # replicate the startup condition to spawn the loop task
                async def _starter():
                    # small delay to allow overrides to take effect before first iteration
                    await asyncio.sleep(0.1)
                    # Reuse the same loop factory as at startup by calling lifespan initializer block indirectly
                    # Here we inline minimal start of the loop to avoid re-entry to lifespan
                    try:
                        # Local re-definition mirrors the startup loop behavior
                        async def _quote_stream_loop_once_started():
                            while True:
                                try:
                                    LOOP_STATUS['quote_stream']["enabled"] = True
                                    LOOP_STATUS['quote_stream']["interval_seconds"] = 1
                                    _env_gate = os.getenv("QUOTE_STREAM_TRADING_HOURS_ONLY", "true").lower() in ("1","true","yes")
                                    _gate = QUOTE_STREAM_GATING_OVERRIDE if QUOTE_STREAM_GATING_OVERRIDE is not None else _env_gate
                                    if _gate and not is_trading_hours():
                                        await asyncio.sleep(30)
                                        continue
                                    symbols: List[str] = []
                                    if QUOTE_STREAM_SYMBOLS_OVERRIDE:
                                        symbols = [s.strip().upper() for s in QUOTE_STREAM_SYMBOLS_OVERRIDE if s and s.strip()]
                                    else:
                                        env_syms = os.getenv("QUOTE_STREAM_SYMBOLS", "").strip()
                                        if env_syms:
                                            symbols = [s.strip().upper() for s in env_syms.split(',') if s.strip()]
                                        elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                                            symbols = (await reference_svc.get_watchlist_symbols()) or []
                                    if not symbols:
                                        await asyncio.sleep(10)
                                        continue
                                    try:
                                        random.shuffle(symbols)
                                    except Exception:
                                        pass
                                    sample_size = int(os.getenv("QUOTE_STREAM_SAMPLE_SIZE", "2") or '2')
                                    max_syms = int(os.getenv("QUOTE_STREAM_MAX_SYMBOLS", "200") or '200')
                                    stream_syms = symbols[:max(1, min(max_syms, sample_size))]
                                    if market_data_svc:
                                        async for _item in market_data_svc.stream_real_time_data(stream_syms):
                                            LOOP_STATUS['quote_stream']["last_run"] = datetime.utcnow().isoformat()
                                            try:
                                                if LOOP_LAST_RUN_UNIX is not None:
                                                    LOOP_LAST_RUN_UNIX.labels(loop='quote_stream').set(time.time())
                                            except Exception:
                                                pass
                                            await asyncio.sleep(0)
                                    else:
                                        await asyncio.sleep(5)
                                except Exception as e:  # noqa: BLE001
                                    LOOP_STATUS['quote_stream']["last_error"] = str(e)
                                    await asyncio.sleep(5)
                        # Assign global task
                        global QUOTE_LOOP_TASK
                        QUOTE_LOOP_TASK = asyncio.create_task(_quote_stream_loop_once_started())
                    except Exception:
                        pass
                QUOTE_LOOP_TASK = asyncio.create_task(_starter())
        except Exception:
            pass
        # Return consolidated status including last_run for quick verification
        return {
            "status": "ok",
            "overrides": {
                "gating_override": QUOTE_STREAM_GATING_OVERRIDE,
                "symbols_override": QUOTE_STREAM_SYMBOLS_OVERRIDE,
            },
            "loop": LOOP_STATUS.get('quote_stream', {}),
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------- Vector store health (Weaviate) ---------------------- #
@app.get("/vector/health")
async def vector_health(timeout_seconds: float | None = None):
    """Best-effort Weaviate health report with class object counts.

    - Uses shared weaviate client factory (supports API key or anonymous).
    - Applies a short timeout per class with a single retry to reduce '?' results.
    - Never fails the endpoint; returns counts or '?' strings.
    """
    classes = ["NewsArticle", "SocialSentiment", "OptionContract", "EquityBar"]
    timeout_s = 0.0
    if timeout_seconds is not None:
        try:
            timeout_s = float(timeout_seconds)
        except Exception:
            timeout_s = 0.0
    if not timeout_s:
        try:
            timeout_s = float(os.getenv("WEAVIATE_HEALTH_TIMEOUT_SECONDS", "2.0"))
        except Exception:
            timeout_s = 2.0

    # Lazy import of schema helpers to avoid import errors when vector is not configured
    try:
        try:
            from shared.vector.weaviate_schema import get_weaviate_client  # type: ignore
        except Exception:
            from ..shared.vector.weaviate_schema import get_weaviate_client  # type: ignore
    except Exception:
        return {"status": "degraded", "error": "weaviate client unavailable", "classes": {c: "?" for c in classes}}

    # Obtain client (may raise); degrade to '?' on failure
    try:
        client = await asyncio.to_thread(get_weaviate_client)
    except Exception as e:  # noqa: BLE001
        return {"status": "degraded", "error": str(e)[:200], "classes": {c: "?" for c in classes}}

    async def _count(cls_name: str) -> str:
        # Run synchronous aggregate in a thread with timeout and one retry
        async def _once() -> str:
            def _agg() -> str:
                try:
                    coll = client.collections.get(cls_name)
                except Exception:
                    return "?"  # class missing or API error
                try:
                    # v4 aggregate API: over_all(total_count=True).total_count
                    res = coll.aggregate.over_all(total_count=True)
                    # Some client versions return an object; others dict-like
                    cnt = getattr(res, "total_count", None)
                    if cnt is None:
                        try:
                            cnt = int(res.get("totalCount"))  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    return str(cnt if cnt is not None else "?")
                except Exception:
                    return "?"
            try:
                return await asyncio.wait_for(asyncio.to_thread(_agg), timeout=max(0.2, timeout_s))
            except Exception:
                return "?"

        out = await _once()
        if out == "?":
            # brief backoff and retry once
            try:
                await asyncio.sleep(0.25)
            except Exception:
                pass
            out = await _once()
        return out

    results = {}
    for name in classes:
        try:
            results[name] = await _count(name)
        except Exception:
            results[name] = "?"

    status = "ok" if any(v.isdigit() for v in results.values()) else "degraded"
    return {"status": status, "classes": results}

# ---------------------- Calendar coverage & backfill endpoints ---------------------- #
@app.get("/calendar/coverage")
async def calendar_coverage():
    """Return row counts and first/last days for earnings/IPO/splits/dividends calendars.

    Also includes provider flags resolved by CalendarService (when initialized).
    """
    tables = [
        ("earnings_calendar", "earnings_calendar"),
        ("ipo_calendar", "ipo_calendar"),
        ("splits_calendar", "splits_calendar"),
        ("dividends_calendar", "dividends_calendar"),
    ]
    out: dict[str, dict] = {}
    for key, table in tables:
        try:
            res = await _qdb_exec(f"select count() c, min(timestamp) mn, max(timestamp) mx from {table}", timeout=10.0)
            cols = {c['name']: i for i, c in enumerate(res.get('columns', []) or [])}
            if res.get('dataset'):
                r = res['dataset'][0]
                cnt = int(r[cols.get('c', 0)]) if 'c' in cols else 0
                mn = r[cols.get('mn')] if 'mn' in cols else None
                mx = r[cols.get('mx')] if 'mx' in cols else None
                to_day = lambda v: (str(v)[:10] if v is not None else None)
                out[key] = {"rows": cnt, "first_day": to_day(mn), "last_day": to_day(mx)}
            else:
                out[key] = {"rows": 0, "first_day": None, "last_day": None}
        except Exception:
            out[key] = {"rows": 0, "first_day": None, "last_day": None}
    # Provider flags (best-effort)
    prov = {}
    try:
        global calendar_svc
        if calendar_svc is not None:
            prov = {
                "provider": getattr(calendar_svc, 'provider', None),
                "alpha_vantage_enabled": bool(getattr(calendar_svc, 'av_enabled', False)),
                "eodhd_enabled": bool(getattr(calendar_svc, 'eodhd_enabled', False)),
            }
        else:
            prov = {
                "provider": os.getenv('CALENDAR_PROVIDER'),
                "alpha_vantage_enabled": bool(os.getenv('ALPHAVANTAGE_API_KEY') or os.getenv('ALPHA_VANTAGE_API_KEY')),
                "eodhd_enabled": bool(os.getenv('EODHD_API_KEY')),
            }
    except Exception:
        prov = {}
    return {**out, **prov}


class CalendarBackfillRequest(BaseModel):
    years: int | None = 3
    include_earnings: bool | None = True
    include_ipo: bool | None = True
    include_splits: bool | None = True
    include_dividends: bool | None = True


@app.post("/backfill/calendar/eodhd")
async def backfill_calendar_eodhd(req: CalendarBackfillRequest = Body(default=CalendarBackfillRequest())):
    """Run EODHD calendar backfill for the given lookback window in background tasks.

    Returns immediately with a started status; progress is visible in QuestDB via row counts.
    """
    global calendar_svc
    if calendar_svc is None:
        # Try to lazily initialize if not yet available
        try:
            from calendar_service import get_calendar_service as _get_cal
        except Exception:
            _get_cal = None  # type: ignore
        if _get_cal is not None:
            try:
                calendar_svc = await _get_cal()  # type: ignore[func-returns-value]
            except Exception:
                calendar_svc = None
    if calendar_svc is None or not getattr(calendar_svc, 'eodhd_enabled', False):
        raise HTTPException(status_code=503, detail="eodhd_calendar_unavailable")

    years = max(1, int(req.years or 3))
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=int(years * 365.25))
    tasks: list[asyncio.Task] = []
    try:
        if req.include_earnings:
            tasks.append(asyncio.create_task(calendar_svc.collect_earnings_range(start_dt, end_dt)))
        if req.include_ipo:
            tasks.append(asyncio.create_task(calendar_svc.collect_ipo_range(start_dt, end_dt)))
        if req.include_splits:
            tasks.append(asyncio.create_task(calendar_svc.collect_splits_range(start_dt, end_dt)))
        if req.include_dividends:
            tasks.append(asyncio.create_task(calendar_svc.collect_dividends_range(start_dt, end_dt)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    # Fire-and-forget; do not await
    return {"status": "started", "years": years, "tasks": len(tasks)}


class SplitDivBackfillRequest(BaseModel):
    symbols: list[str] | None = None
    years: int | None = 10
    max_symbols: int | None = 50
    concurrency: int | None = 5


@app.post("/backfill/eodhd/splits-dividends")
async def backfill_splits_dividends(req: SplitDivBackfillRequest = Body(default=SplitDivBackfillRequest())):
    """Per-symbol splits and dividends backfill using EODHD symbol feeds (non-calendar).

    If symbols not provided, resolves from reference service watchlist.
    """
    global calendar_svc, reference_svc
    if calendar_svc is None:
        try:
            from calendar_service import get_calendar_service as _get_cal  # type: ignore
        except Exception:
            _get_cal = None  # type: ignore
        if _get_cal is not None:
            try:
                calendar_svc = await _get_cal()  # type: ignore[func-returns-value]
            except Exception:
                calendar_svc = None
    if calendar_svc is None or not getattr(calendar_svc, 'eodhd_enabled', False):
        raise HTTPException(status_code=503, detail="eodhd_calendar_unavailable")

    # Resolve symbols
    symbols: list[str] = [s.strip().upper() for s in (req.symbols or []) if s and str(s).strip()]
    if not symbols and reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            symbols = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            symbols = []
    if not symbols:
        # Safe fallback
        symbols = [s.strip().upper() for s in (os.getenv('SOCIAL_FALLBACK_SYMBOLS', 'AAPL,MSFT,TSLA,NVDA,SPY') or '').split(',') if s.strip()]
    max_syms = max(1, int(req.max_symbols or 50))
    symbols = symbols[:max_syms]
    years = max(1, int(req.years or 10))
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=int(years * 365.25))
    sem = asyncio.Semaphore(max(1, int(req.concurrency or 5)))

    splits_ins = 0
    divs_ins = 0

    async def _one(sym: str):
        nonlocal splits_ins, divs_ins
        async with sem:
            try:
                s_cnt = await calendar_svc.collect_eodhd_splits_symbol(sym, start_dt, end_dt)
                d_cnt = await calendar_svc.collect_eodhd_dividends_symbol(sym, start_dt, end_dt)
                splits_ins += int(s_cnt or 0)
                divs_ins += int(d_cnt or 0)
            except Exception:
                pass

    tasks = [asyncio.create_task(_one(sym)) for sym in symbols]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return {"symbols": len(symbols), "splits_inserted": int(splits_ins), "dividends_inserted": int(divs_ins)}

# One-shot reconcile trigger for Weaviate indexing of recent QuestDB news
@app.post("/vector/reconcile/once")
async def vector_reconcile_once(days: int = 3, limit: int = 2000):
    try:
        d = max(1, int(days))
        l = max(1, int(limit))
    except Exception:
        d = 3; l = 2000
    try:
        res = await _vector_reconcile_once(rec_days=d, rec_limit=l)
        return {"status": "ok", **res}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)[:200]})

# ---------------------- One-off housekeeping archive to MinIO ---------------------- #
class HousekeepingRequest(BaseModel):
    roots: Optional[List[str]] = Field(default=None, description="Root folders to scan (default ['/srv', '/srv/ai-trading-system'])")
    patterns: Optional[List[str]] = Field(default=None, description="Glob patterns to match (default from HOUSEKEEPING_PATTERNS)")
    age_days: int = Field(default=30, ge=1, le=365)
    bucket: Optional[str] = Field(default=None, description="MinIO bucket (default MINIO_ARTIFACTS_BUCKET or 'trading')")
    prefix: Optional[str] = Field(default=None, description="Prefix in bucket (default MINIO_ARTIFACTS_PREFIX or 'archives/ops-json')")


@app.post("/housekeeping/archive-now")
async def housekeeping_archive_now(req: HousekeepingRequest = Body(default=HousekeepingRequest())):
    """Archive stale JSON artifacts to MinIO and prune locally.

    Credentials:
      - Uses MINIO_ACCESS_KEY_ID / MINIO_SECRET_ACCESS_KEY when provided.
      - Falls back to MINIO_ROOT_USER / MINIO_ROOT_PASSWORD if set.
      - If neither provided, falls back to 'minioadmin' for both (local dev default).
    Endpoint:
      - endpoint_url http://minio:9000 (intra-docker) unless MINIO_ENDPOINT_URL set.
    Safety:
      - Only uploads files older than age_days matching provided patterns.
      - Deletes file locally only after successful upload.
    """
    try:
        import boto3  # type: ignore
        from botocore.config import Config as _BotoCfg  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"boto3 not available: {e}")

    roots = req.roots or [p for p in (os.getenv("HOUSEKEEPING_PATHS", "/srv,/srv/ai-trading-system") or "").split(",") if p.strip()]
    if not roots:
        roots = ["/srv", "/srv/ai-trading-system"]
    pat_env = os.getenv("HOUSEKEEPING_PATTERNS", "coverage_snapshot*.json,coverage_snapshot_full*.json,coverage_run_latest.json,coverage_summary_consolidated.json,retention_*_questdb.json")
    patterns = req.patterns or [p.strip() for p in pat_env.split(",") if p.strip()]
    cutoff = datetime.utcnow() - timedelta(days=max(1, int(req.age_days)))
    bucket = (req.bucket or os.getenv("MINIO_ARTIFACTS_BUCKET") or "trading").strip()
    prefix = (req.prefix or os.getenv("MINIO_ARTIFACTS_PREFIX") or "archives/ops-json").strip().strip("/")
    endpoint = os.getenv("MINIO_ENDPOINT_URL", "http://minio:9000").strip()
    access = os.getenv("MINIO_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER") or "minioadmin"
    secret = os.getenv("MINIO_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD") or "minioadmin"

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        config=_BotoCfg(signature_version='s3v4', retries={'max_attempts': 2, 'mode': 'standard'})
    )

    attempted = 0
    uploaded = 0
    deleted = 0
    errors: List[str] = []
    examples: List[str] = []
    for root in roots:
        for pat in patterns:
            try:
                for path in glob.glob(os.path.join(root, pat)):
                    try:
                        st = os.stat(path)
                    except Exception:
                        continue
                    mtime = datetime.utcfromtimestamp(st.st_mtime)
                    if mtime > cutoff or not os.path.isfile(path):
                        continue
                    attempted += 1
                    # Build dated key under prefix (YYYY/MM/filename)
                    dt = mtime
                    key_prefix = f"{prefix}/{dt.year:04d}/{dt.month:02d}"
                    fname = pathlib.Path(path).name
                    key = f"{key_prefix}/{fname}"
                    try:
                        s3.upload_file(path, bucket, key)
                        uploaded += 1
                        if len(examples) < 3:
                            examples.append(key)
                        try:
                            os.remove(path)
                            deleted += 1
                        except Exception:
                            pass
                    except Exception as e:  # noqa: BLE001
                        errors.append(str(e)[:120])
            except Exception as e:  # noqa: BLE001
                errors.append(str(e)[:120])

    return {
        "status": "ok",
        "attempted": attempted,
        "uploaded": uploaded,
        "deleted": deleted,
        "errors": errors[:5],
        "examples": examples,
        "bucket": bucket,
        "prefix": prefix,
        "endpoint": endpoint,
    }

# ---------------------- On-demand backfill endpoints ---------------------- #

class BackfillRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to backfill; default uses NEWS_STREAM_SYMBOLS or reference service")
    days: int = Field(default=60, ge=1, le=365*5, description="Lookback window in days")
    batch_days: int = Field(default=14, ge=1, le=60, description="Batch size (days) per provider call window")
    max_articles_per_batch: int = Field(default=80, ge=10, le=1000)


@app.post("/backfill/news/eodhd-60d")
async def trigger_news_backfill(req: BackfillRequest = Body(default=BackfillRequest()), background_tasks: BackgroundTasks = None):
    """Trigger a bounded news backfill (EODHD-first) over the past N days.

    Runs asynchronously in the background. Uses EODHD as primary via the existing
    NewsService.collect_financial_news_range implementation and persists to QuestDB/Postgres when enabled.
    """
    if not news_svc:
        raise HTTPException(status_code=503, detail="news service not initialized")
    # Resolve symbols: env NEWS_STREAM_SYMBOLS -> request.symbols -> reference service -> fallback
    symbols: List[str] = []
    try:
        if req.symbols:
            symbols = [s.strip().upper() for s in req.symbols if s and s.strip()]
    except Exception:
        symbols = []
    if not symbols:
        env_syms = os.getenv("NEWS_STREAM_SYMBOLS", "").strip()
        if env_syms:
            symbols = [s.strip().upper() for s in env_syms.split(',') if s.strip()]
    if not symbols and reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            symbols = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            symbols = []
    if not symbols:
        symbols = ['AAPL','MSFT','TSLA','NVDA','SPY']
    # Limit symbol count to keep provider calls reasonable
    symbols = symbols[:100]

    # Date window
    end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=max(1, int(req.days)))

    async def _run_backfill():
        try:
            total, batches = await news_svc.collect_financial_news_range(
                symbols=symbols,
                start_date=start_dt,
                end_date=end_dt,
                batch_days=int(req.batch_days),
                max_articles_per_batch=int(req.max_articles_per_batch),
                backfill_mode=True,
            )
            logger.info("News backfill complete", total_articles=total, batches=len(batches))
        except Exception as e:  # noqa: BLE001
            logger.warning("News backfill failed", error=str(e))

    # schedule in background
    if background_tasks is not None:
        background_tasks.add_task(_run_backfill)
    else:
        # Fallback if BackgroundTasks not provided (e.g., internal call)
        asyncio.create_task(_run_backfill())
    return {"status": "scheduled", "symbols": len(symbols), "start": start_dt.date().isoformat(), "end": end_dt.date().isoformat()}


class CalendarBackfillRequest(BaseModel):
    years: int = Field(default=5, ge=1, le=10, description="Lookback horizon in years")
    include_dividends: bool = Field(default=True)
    pacing_seconds: float = Field(default=0.1, ge=0.0, le=5.0)


@app.post("/backfill/calendar/eodhd")
async def trigger_calendar_backfill(req: CalendarBackfillRequest = Body(default=CalendarBackfillRequest()), background_tasks: BackgroundTasks = None):
    """Trigger EODHD calendar backfill (earnings, IPOs, splits, optional dividends) for the last N years."""
    # Allow running if the EODHD path is available even when global provider isn't eodhd
    if not calendar_svc:
        raise HTTPException(status_code=503, detail="calendar service not initialized")
    # Require EODHD API key
    if not getattr(calendar_svc, 'eodhd_api_key', ''):
        raise HTTPException(status_code=503, detail="calendar service disabled or not initialized")
    end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(req.years * 365.25))

    async def _run():
        try:
            # Process by month windows to keep calls reasonable
            cur = start_dt
            total_e = total_i = total_s = total_d = 0
            while cur <= end_dt:
                wnd_end = min(end_dt, cur + timedelta(days=29))
                try:
                    total_e += await calendar_svc.collect_earnings_range(cur, wnd_end)
                except Exception:
                    pass
                try:
                    total_i += await calendar_svc.collect_ipo_range(cur, wnd_end)
                except Exception:
                    pass
                try:
                    total_s += await calendar_svc.collect_splits_range(cur, wnd_end)
                except Exception:
                    pass
                if req.include_dividends:
                    try:
                        total_d += await calendar_svc.collect_dividends_range(cur, wnd_end)
                    except Exception:
                        pass
                LOOP_STATUS.setdefault('calendar_backfill', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
                LOOP_STATUS['calendar_backfill']["last_run"] = datetime.utcnow().isoformat()
                try:
                    if LOOP_LAST_RUN_UNIX is not None:
                        LOOP_LAST_RUN_UNIX.labels(loop='calendar_backfill').set(time.time())
                except Exception:
                    pass
                await asyncio.sleep(max(0.0, float(req.pacing_seconds)))
                cur = wnd_end + timedelta(days=1)
            logger.info("Calendar backfill complete", earnings=total_e, ipos=total_i, splits=total_s, dividends=total_d)
        except Exception as e:  # noqa: BLE001
            LOOP_STATUS.setdefault('calendar_backfill', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
            LOOP_STATUS['calendar_backfill']["last_error"] = str(e)
            logger.warning("Calendar backfill failed: %s", e)

    if background_tasks is not None:
        background_tasks.add_task(_run)
    else:
        asyncio.create_task(_run())
    return {"status": "scheduled", "start": start_dt.date().isoformat(), "end": end_dt.date().isoformat(), "include_dividends": req.include_dividends}


# ---------------------- Alpha Vantage calendar endpoints ---------------------- #
class AvEarningsRequest(BaseModel):
    symbol: Optional[str] = Field(default=None, description="Optional symbol; default collects full upcoming list")
    horizon: Optional[str] = Field(default=None, description="3month|6month|12month; default uses AV_EARNINGS_CALENDAR_HORIZON")


@app.post("/calendar/av/earnings")
async def run_av_earnings(req: AvEarningsRequest = Body(default=AvEarningsRequest())):
    if not calendar_svc or not getattr(calendar_svc, 'enabled', False) or getattr(calendar_svc, 'provider', '') != 'alphavantage':
        raise HTTPException(status_code=503, detail="Alpha Vantage calendar provider not enabled")
    try:
        added = await calendar_svc.collect_av_earnings_upcoming(symbol=req.symbol, horizon=req.horizon)
        LOOP_STATUS.setdefault('calendar_av_earnings', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
        LOOP_STATUS['calendar_av_earnings']["last_run"] = datetime.utcnow().isoformat()
        try:
            if LOOP_LAST_RUN_UNIX is not None:
                LOOP_LAST_RUN_UNIX.labels(loop='calendar_av_earnings').set(time.time())
        except Exception:
            pass
        return {"status": "ok", "inserted": int(added)}
    except Exception as e:
        LOOP_STATUS.setdefault('calendar_av_earnings', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
        LOOP_STATUS['calendar_av_earnings']["last_error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------- EODHD per-symbol splits/dividends backfill ---------------------- #
class EodhdSplitsDividendsRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to process; defaults to reference watchlist or fallback set")
    years: float = Field(default=10.0, ge=0.5, le=30.0, description="Lookback window in years")
    include_dividends: bool = Field(default=True)
    pacing_seconds: float = Field(default=0.05, ge=0.0, le=2.0)


@app.post("/backfill/eodhd/splits-dividends")
async def backfill_eodhd_splits_dividends(req: EodhdSplitsDividendsRequest = Body(default=EodhdSplitsDividendsRequest())):
    if not calendar_svc or not getattr(calendar_svc, 'enabled', False):
        raise HTTPException(status_code=503, detail="calendar service disabled or not initialized")
    # Resolve symbols
    syms: List[str] = []
    try:
        if req.symbols:
            syms = [s.strip().upper() for s in req.symbols if s and s.strip()]
    except Exception:
        syms = []
    if not syms and reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            syms = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            syms = []
    if not syms:
        syms = ['AAPL','MSFT','TSLA','NVDA','SPY']
    # Window
    end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(float(req.years) * 365.25))
    total_splits = 0
    total_divs = 0
    errors = 0
    for sym in syms:
        try:
            try:
                total_splits += await calendar_svc.collect_eodhd_splits_symbol(sym, start_dt, end_dt)  # type: ignore[attr-defined]
            except Exception:
                pass
            if req.include_dividends:
                try:
                    total_divs += await calendar_svc.collect_eodhd_dividends_symbol(sym, start_dt, end_dt)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            errors += 1
        # pacing
        try:
            await asyncio.sleep(float(req.pacing_seconds))
        except Exception:
            pass
    LOOP_STATUS.setdefault('calendar_eodhd_symbol_backfill', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
    LOOP_STATUS['calendar_eodhd_symbol_backfill']["last_run"] = datetime.utcnow().isoformat()
    try:
        if LOOP_LAST_RUN_UNIX is not None:
            LOOP_LAST_RUN_UNIX.labels(loop='calendar_eodhd_symbol_backfill').set(time.time())
    except Exception:
        pass
    return {"status": "ok", "symbols": len(syms), "splits_inserted": int(total_splits), "dividends_inserted": int(total_divs), "errors": int(errors)}


@app.post("/calendar/av/ipos")
async def run_av_ipos():
    if not calendar_svc or not getattr(calendar_svc, 'enabled', False) or getattr(calendar_svc, 'provider', '') != 'alphavantage':
        raise HTTPException(status_code=503, detail="Alpha Vantage calendar provider not enabled")
    try:
        added = await calendar_svc.collect_av_ipo_upcoming()
        LOOP_STATUS.setdefault('calendar_av_ipos', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
        LOOP_STATUS['calendar_av_ipos']["last_run"] = datetime.utcnow().isoformat()
        try:
            if LOOP_LAST_RUN_UNIX is not None:
                LOOP_LAST_RUN_UNIX.labels(loop='calendar_av_ipos').set(time.time())
        except Exception:
            pass
        return {"status": "ok", "inserted": int(added)}
    except Exception as e:
        LOOP_STATUS.setdefault('calendar_av_ipos', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
        LOOP_STATUS['calendar_av_ipos']["last_error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@app.get("/calendar/coverage")
async def calendar_coverage():
    """Return quick row counts and last dates for calendar tables (QuestDB)."""
    try:
        out: Dict[str, Dict[str, Optional[str] | int]] = {}
        for table, date_col in (
            ('earnings_calendar','date'),
            ('ipo_calendar','date'),
            ('splits_calendar','date'),
            ('dividends_calendar','ex_date'),
        ):
            try:
                data = await _qdb_exec(
                    f"select count() as rows, min({date_col}) as first_day, max({date_col}) as last_day from {table}"
                )
                if data.get('dataset'):
                    r = data['dataset'][0]
                    cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
                    rows = int(r[cols.get('rows')]) if 'rows' in cols else 0
                    first_day = str(r[cols.get('first_day')])[:10] if 'first_day' in cols and r[cols.get('first_day')] is not None else None
                    last_day = str(r[cols.get('last_day')])[:10] if 'last_day' in cols and r[cols.get('last_day')] is not None else None
                    out[table] = {"rows": rows, "first_day": first_day, "last_day": last_day}
                else:
                    out[table] = {"rows": 0, "first_day": None, "last_day": None}
            except Exception:
                out[table] = {"rows": 0, "first_day": None, "last_day": None}
        # Provider and availability snapshot for operators
        if calendar_svc:
            out['provider'] = getattr(calendar_svc, 'provider', None)
            out['alpha_vantage_enabled'] = bool(getattr(calendar_svc, 'av_enabled', False))
            out['eodhd_enabled'] = bool(getattr(calendar_svc, 'eodhd_enabled', False))
        else:
            out['provider'] = None
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


@app.get("/coverage/summary")
async def coverage_summary():
    """Lightweight coverage counts and first/last timestamps for core datasets.

    This endpoint is used by production health scripts to sanity-check data tables quickly.
    """
    try:
        tables = [
            {"table": "market_data", "ts": "timestamp", "name": "equities"},
            {"table": "options_data", "ts": "timestamp", "name": "options"},
            {"table": "news_items", "ts": "ts", "name": "news"},
            {"table": "social_signals", "ts": "ts", "name": "social"},
        ]
        out: Dict[str, Dict[str, Any]] = {}
        for spec in tables:
            tname = spec["table"]; tcol = spec["ts"]; name = spec["name"]
            try:
                data = await _qdb_exec(
                    f"select count() as rows, min({tcol}) as first_ts, max({tcol}) as last_ts from {tname}")
                rows = 0; first = None; last = None
                if data.get('dataset'):
                    r = data['dataset'][0]
                    cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
                    try:
                        rows = int(r[cols.get('rows', 0)])
                    except Exception:
                        rows = 0
                    try:
                        first = str(r[cols.get('first_ts')]) if r[cols.get('first_ts')] is not None else None
                    except Exception:
                        first = None
                    try:
                        last = str(r[cols.get('last_ts')]) if r[cols.get('last_ts')] is not None else None
                    except Exception:
                        last = None
                out[name] = {"rows": rows, "first": first, "last": last}
            except Exception:
                out[name] = {"rows": 0, "first": None, "last": None}
        return {"status": "ok", **out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# --------- On-demand Vector Reconcile (Weaviate news indexing) ---------
class VectorReconcileRequest(BaseModel):
    days: int = Field(default=3, ge=1, le=30)
    limit: int = Field(default=2000, ge=1, le=10000)


@app.post("/vector/reconcile/run")
async def vector_reconcile_run(req: VectorReconcileRequest = Body(default=VectorReconcileRequest())):
    """Run a one-off news vector reconcile into Weaviate (no env flags required)."""
    try:
        report = await _vector_reconcile_once(rec_days=int(req.days), rec_limit=int(req.limit))
        # Update loop status metric for observability
        LOOP_STATUS.setdefault('vector_reconcile', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
        LOOP_STATUS['vector_reconcile']["last_run"] = datetime.utcnow().isoformat()
        try:
            if LOOP_LAST_RUN_UNIX is not None:
                LOOP_LAST_RUN_UNIX.labels(loop='vector_reconcile').set(time.time())
        except Exception:
            pass
        return {"status": "ok", **report}
    except Exception as e:
        LOOP_STATUS.setdefault('vector_reconcile', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
        LOOP_STATUS['vector_reconcile']["last_error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# --------- Options-wide News Backfill Batch Scheduler ---------
class OptionsNewsBackfillBatchRequest(BaseModel):
    underlyings: Optional[List[str]] = Field(default=None, description="Explicit underlying symbols; default is QuestDB distinct underlyings")
    max_underlyings: int = Field(default=500, ge=1, le=5000, description="Cap number of underlyings resolved from QuestDB")
    chunk_size: int = Field(default=25, ge=1, le=200, description="Symbols per batch window")
    years: float = Field(default=float(NEWS_BACKFILL_YEARS), ge=0.1, le=10.0, description="Historical lookback in years")
    batch_days: int = Field(default=NEWS_BACKFILL_WINDOW_DAYS, ge=1, le=60, description="Days per provider call window")
    max_articles_per_batch: int = Field(default=80, ge=10, le=1000)
    inter_batch_delay_seconds: float = Field(default=5.0, ge=0.0, le=60.0)


@app.post("/backfill/news/options-batch/run")
async def trigger_options_news_batch_scheduler(req: OptionsNewsBackfillBatchRequest = Body(default=OptionsNewsBackfillBatchRequest())):
    """Schedule a server-side batch backfill of news for option underlyings.

    Behavior:
      - Resolve underlyings from QuestDB options_data (distinct) unless provided
      - Slice into chunks of N symbols and call NewsService.collect_financial_news_range per chunk
      - Honor provider pacing via inter-batch delay and per-window batch_days/article caps

    Returns an immediate schedule acknowledgment; work runs asynchronously.
    """
    if not news_svc:
        raise HTTPException(status_code=503, detail="news service not initialized")

    # Resolve underlyings list
    underlyings: List[str] = []
    try:
        if req.underlyings:
            underlyings = [s.strip().upper() for s in req.underlyings if s and s.strip()]
    except Exception:
        underlyings = []
    if not underlyings:
        try:
            data = await _qdb_exec(
                "select distinct upper(underlying) as u from options_data where underlying is not null and underlying != '' order by u limit "
                + str(max(1, int(req.max_underlyings)))
            )
            cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
            u_idx = cols.get('u')
            if u_idx is not None:
                for r in data.get('dataset') or []:
                    try:
                        val = str(r[u_idx]).strip().upper()
                        if val:
                            underlyings.append(val)
                    except Exception:
                        continue
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"QuestDB underlyings query failed: {e}")
    # Fallback safety
    if not underlyings:
        underlyings = ['AAPL','MSFT','TSLA','NVDA','SPY']

    # Compute overall date window
    end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        lookback_days = int(float(req.years) * 365.25)
    except Exception:
        lookback_days = int(float(NEWS_BACKFILL_YEARS) * 365.25)
    start_dt = end_dt - timedelta(days=max(1, lookback_days))

    total_syms = len(underlyings)
    chunk = max(1, int(req.chunk_size))
    chunks = math.ceil(total_syms / chunk)

    async def _run_batches():
        LOOP_STATUS.setdefault('news_options_batch', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
        LOOP_STATUS['news_options_batch']["enabled"] = True
        LOOP_STATUS['news_options_batch']["interval_seconds"] = None
        err_count = 0
        completed = 0
        for i in range(0, total_syms, chunk):
            batch = underlyings[i:i+chunk]
            try:
                await news_svc.collect_financial_news_range(
                    symbols=batch,
                    start_date=start_dt,
                    end_date=end_dt,
                    batch_days=int(req.batch_days),
                    max_articles_per_batch=int(req.max_articles_per_batch),
                    backfill_mode=True,
                )
            except Exception as e:  # noqa: BLE001
                err_count += 1
                LOOP_STATUS['news_options_batch']["last_error"] = str(e)
            completed += 1
            LOOP_STATUS['news_options_batch']["last_run"] = datetime.utcnow().isoformat()
            try:
                if LOOP_LAST_RUN_UNIX is not None:
                    LOOP_LAST_RUN_UNIX.labels(loop='news_options_batch').set(time.time())
            except Exception:
                pass
            await asyncio.sleep(max(0.0, float(req.inter_batch_delay_seconds)))
        logger.info(
            "Options-wide news batch backfill completed",
            batches=completed,
            errors=err_count,
            total_symbols=total_syms,
            chunk_size=chunk,
            years=float(req.years),
        )

    asyncio.create_task(_run_batches())
    return {
        "status": "scheduled",
        "underlyings": total_syms,
        "chunks": chunks,
        "chunk_size": chunk,
        "start": start_dt.date().isoformat(),
        "end": end_dt.date().isoformat(),
        "years": float(req.years),
    }

# --------- Full Equities & Options Backfill Endpoints (on-demand) ---------
class EquitiesBackfillRequest(BaseModel):
    years: float = Field(default=20.0, ge=1.0, le=30.0)
    max_symbols: int = Field(default=1000, ge=1, le=5000)
    pacing_seconds: float = Field(default=0.2, ge=0.0, le=5.0)
    symbols: Optional[List[str]] = None


@app.post("/backfill/equities/run")
async def trigger_equities_full_backfill(req: EquitiesBackfillRequest = Body(default=EquitiesBackfillRequest())):
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="market data service not initialized")
    # Resolve symbols
    syms: List[str] = []
    try:
        if req.symbols:
            syms = [s.strip().upper() for s in req.symbols if s and s.strip()]
    except Exception:
        syms = []
    if not syms and reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            syms = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            syms = []
    if not syms:
        syms = ['AAPL','MSFT','TSLA','NVDA','SPY']
    syms = syms[:max(1, int(req.max_symbols))]

    end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(float(req.years) * 365.25))

    async def _run():
        try:
            total = len(syms)
            done = 0
            errs = 0
            bars_total = 0
            LOOP_STATUS.setdefault('equities_backfill', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
            LOOP_STATUS['equities_backfill']["enabled"] = True
            LOOP_STATUS['equities_backfill']["interval_seconds"] = None
            for sym in syms:
                try:
                    rows = await market_data_svc.get_bulk_daily_historical(sym, start_dt, end_dt)
                    bars_total += int(len(rows or []))
                except Exception as e:  # noqa: BLE001
                    errs += 1
                    LOOP_STATUS['equities_backfill']["last_error"] = str(e)
                done += 1
                try:
                    pct = (done / total) * 100.0 if total else 0.0
                    set_backfill_progress(pct)
                except Exception:
                    pass
                await asyncio.sleep(float(req.pacing_seconds))
            LOOP_STATUS['equities_backfill']["last_run"] = datetime.utcnow().isoformat()
            try:
                if LOOP_LAST_RUN_UNIX is not None:
                    LOOP_LAST_RUN_UNIX.labels(loop='equities_backfill').set(time.time())
            except Exception:
                pass
            logger.info("On-demand equities backfill completed", total_symbols=total, errors=errs, bars=bars_total, years=float(req.years))
        except Exception as e:  # noqa: BLE001
            LOOP_STATUS.setdefault('equities_backfill', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
            LOOP_STATUS['equities_backfill']["last_error"] = str(e)
            logger.warning("On-demand equities backfill error: %s", e)

    asyncio.create_task(_run())
    return {"status": "scheduled", "symbols": len(syms), "start": start_dt.date().isoformat(), "end": end_dt.date().isoformat(), "years": float(req.years)}


class OptionsBackfillRequest(BaseModel):
    underlyings: Optional[List[str]] = None
    max_underlyings: int = Field(default=200, ge=1, le=1000)
    pacing_seconds: float = Field(default=0.2, ge=0.0, le=5.0)
    expiry_back_days: int = Field(default=365*2, ge=7, le=365*10)
    expiry_ahead_days: int = Field(default=90, ge=7, le=365)
    hist_lookback_days: int = Field(default=365*5, ge=7, le=365*10)


@app.post("/backfill/options/run")
async def trigger_options_full_backfill(req: OptionsBackfillRequest = Body(default=OptionsBackfillRequest())):
    if not market_data_svc or not getattr(market_data_svc, 'enable_options_ingest', False):
        raise HTTPException(status_code=503, detail="options ingest not enabled")
    u: List[str] = []
    try:
        if req.underlyings:
            u = [s.strip().upper() for s in req.underlyings if s and s.strip()]
    except Exception:
        u = []
    if not u and reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            u = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            u = []
    if not u:
        u = ['AAPL','MSFT','TSLA','NVDA','SPY']
    u = u[:max(1, int(req.max_underlyings))]

    now_d = datetime.utcnow().date()
    start_expiry = now_d - timedelta(days=max(7, int(req.expiry_back_days)))
    end_expiry = now_d + timedelta(days=max(7, int(req.expiry_ahead_days)))
    start_hist = now_d - timedelta(days=max(7, int(req.hist_lookback_days)))
    end_hist = now_d

    async def _run():
        try:
            LOOP_STATUS.setdefault('daily_options', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
            LOOP_STATUS['daily_options']["enabled"] = True
            LOOP_STATUS['daily_options']["interval_seconds"] = None
            for sym in u:
                try:
                    await market_data_svc.backfill_options_chain(
                        sym,
                        datetime.combine(start_expiry, datetime.min.time()),
                        datetime.combine(end_expiry, datetime.min.time()),
                        start_date=datetime.combine(start_hist, datetime.min.time()),
                        end_date=datetime.combine(end_hist, datetime.min.time()),
                        max_contracts=1000,
                        pacing_seconds=float(req.pacing_seconds),
                    )
                except Exception as e:  # noqa: BLE001
                    LOOP_STATUS['daily_options']["last_error"] = str(e)
                await asyncio.sleep(float(req.pacing_seconds))
            LOOP_STATUS['daily_options']["last_run"] = datetime.utcnow().isoformat()
            try:
                if LOOP_LAST_RUN_UNIX is not None:
                    LOOP_LAST_RUN_UNIX.labels(loop='daily_options').set(time.time())
            except Exception:
                pass
            logger.info("On-demand options backfill completed", underlyings=len(u))
        except Exception as e:  # noqa: BLE001
            LOOP_STATUS.setdefault('daily_options', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
            LOOP_STATUS['daily_options']["last_error"] = str(e)
            logger.warning("On-demand options backfill error: %s", e)

    asyncio.create_task(_run())
    return {"status": "scheduled", "underlyings": len(u), "expiry_window": {"start": str(start_expiry), "end": str(end_expiry)}, "hist_days": int(req.hist_lookback_days)}

# --------- EODHD per-symbol splits/dividends backfill ---------
class SplitsDividendsBackfillRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None)
    years: float = Field(default=5.0, ge=0.5, le=30.0)
    pacing_seconds: float = Field(default=0.1, ge=0.0, le=2.0)


@app.post('/backfill/eodhd/splits-dividends')
async def backfill_eodhd_splits_dividends(req: SplitsDividendsBackfillRequest = Body(default=SplitsDividendsBackfillRequest())):
    """Backfill EODHD splits and dividends for a list of symbols over a lookback window.

    This uses EODHD non-calendar feeds (which your plan includes) while keeping Alpha Vantage
    as the calendar provider for earnings and IPOs.
    """
    if not calendar_svc:
        raise HTTPException(status_code=503, detail="calendar service not initialized")
    # Determine symbol list
    syms: List[str] = []
    if req.symbols:
        syms = [s.strip().upper() for s in req.symbols if s and s.strip()]
    if not syms and reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            syms = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            syms = []
    if not syms:
        syms = ['AAPL','MSFT','TSLA','NVDA','SPY']
    # Time window
    end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(float(req.years) * 365.25))

    async def _run():
        ok_splits = 0; ok_divs = 0; errs = 0
        for sym in syms:
            try:
                if hasattr(calendar_svc, 'collect_eodhd_splits_symbol'):
                    ok_splits += int(await calendar_svc.collect_eodhd_splits_symbol(sym, start_dt, end_dt))
            except Exception:
                errs += 1
            try:
                if hasattr(calendar_svc, 'collect_eodhd_dividends_symbol'):
                    ok_divs += int(await calendar_svc.collect_eodhd_dividends_symbol(sym, start_dt, end_dt))
            except Exception:
                errs += 1
            await asyncio.sleep(max(0.0, float(req.pacing_seconds)))
        LOOP_STATUS.setdefault('calendar_backfill', {"enabled": True, "last_run": None, "last_error": None, "interval_seconds": None})
        LOOP_STATUS['calendar_backfill']["last_run"] = datetime.utcnow().isoformat()
        try:
            if LOOP_LAST_RUN_UNIX is not None:
                LOOP_LAST_RUN_UNIX.labels(loop='calendar_backfill').set(time.time())
        except Exception:
            pass
        logger.info("EODHD splits/dividends backfill completed", symbols=len(syms), splits=ok_splits, dividends=ok_divs, errors=errs)

    asyncio.create_task(_run())
    return {"status": "scheduled", "symbols": len(syms), "start": start_dt.date().isoformat(), "end": end_dt.date().isoformat()}

# ---------------------------------------------------------------------------
# Install shared observability middleware early (canonical metrics & concurrency)
# ---------------------------------------------------------------------------
_INGEST_CONCURRENCY_LIMIT: int | None = None
_raw_limit = os.getenv("INGEST_CONCURRENCY_LIMIT", "").strip()
if _raw_limit:
    try:
        _INGEST_CONCURRENCY_LIMIT = int(_raw_limit)
    except Exception:  # noqa: BLE001
        logger.warning("Invalid INGEST_CONCURRENCY_LIMIT; ignoring", value=_raw_limit)

if install_observability:
    try:
        # NOTE: service label normalized to hyphen form (data-ingestion) to match Prometheus rules & other services.
        install_observability(app, service_name="data-ingestion", concurrency_limit=_INGEST_CONCURRENCY_LIMIT)
        logger.info(
            "Observability installed",
            concurrency_limit=_INGEST_CONCURRENCY_LIMIT,
            service="data-ingestion"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to install observability middleware", error=str(e), service="data-ingestion")
else:
    logger.warning("observability module not available; running without canonical metrics", service="data-ingestion")


# ---------------------- QuestDB HTTP helper ---------------------- #
async def _qdb_exec(sql: str, timeout: float = 15.0) -> dict:
    """Execute SQL via QuestDB HTTP /exec and return parsed JSON.

    Uses QUESTDB_HTTP_URL if provided, else constructs from host/port envs.
    Raises HTTPException on transport/response errors.
    """
    host = os.getenv('QUESTDB_HOST', 'trading-questdb')
    http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
    qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(qdb_url, params={"query": sql}) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    raise HTTPException(status_code=502, detail=f"QuestDB HTTP {resp.status}: {txt[:200]}")
                return await resp.json()
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"QuestDB exec error: {e}")


# ---------------------- Streams status endpoint ---------------------- #
@app.get("/streams/status")
async def streams_status():
    """Return current background loop status with last_run timestamps and error notes."""
    try:
        # Shallow copy to avoid accidental mutation by callers
        return {"timestamp": datetime.utcnow().isoformat(), "loops": {k: dict(v) for k, v in LOOP_STATUS.items()}}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})

    # ---------------------- Coverage Export (On-demand) ---------------------- #
    class OptionsCoverageRequest(BaseModel):
        underlyings: Optional[List[str]] = None
        max_underlyings: int = Field(default=200, ge=1, le=5000)

    async def _compute_options_coverage(underlyings: Optional[List[str]] = None, max_underlyings: int = 200) -> dict:
        import aiohttp
        host = os.getenv('QUESTDB_HOST', 'trading-questdb')
        http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
        qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
        symbols_env_cov = os.getenv("OPTIONS_COVERAGE_UNDERLYINGS", "").strip()
        syms: List[str] = []
        if underlyings:
            syms = [s.strip().upper() for s in underlyings if s and s.strip()]
        elif symbols_env_cov:
            syms = [s.strip().upper() for s in symbols_env_cov.split(',') if s.strip()]
        elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
            try:
                syms = (await reference_svc.get_watchlist_symbols()) or []
            except Exception:
                syms = []
        if not syms:
            syms = ['AAPL','MSFT','TSLA','NVDA','SPY']
        syms = syms[:max(1, max_underlyings)]

        async def _q(session: aiohttp.ClientSession, sql: str) -> dict:
            async with session.get(qdb_url, params={"query": sql}) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    raise RuntimeError(f"QuestDB HTTP {resp.status}: {txt[:160]}")
                return await resp.json()

        out = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for u in syms:
                try:
                    sql_summary = (
                        "select count() as rows, count_distinct(option_symbol) as contracts, "
                        "min(timestamp) as first_ts, "
                        "max(timestamp) as last_ts "
                        f"from options_data where underlying = '{u}'"
                    )
                    data = await _q(session, sql_summary)
                    if not data.get('dataset'):
                        out.append({"underlying": u, "rows": 0, "contracts": 0, "first_day": None, "last_day": None, "recent_gap_days_30d": None})
                        continue
                    r = data['dataset'][0]
                    cols = {c['name']: i for i, c in enumerate(data.get('columns', []))}
                    rows = int(r[cols['rows']]) if 'rows' in cols else 0
                    contracts = int(r[cols['contracts']]) if 'contracts' in cols else 0
                    def _fmt_iso_day(v):
                        try:
                            return str(v)[:10]
                        except Exception:
                            return None
                    first_day = _fmt_iso_day(r[cols['first_ts']]) if 'first_ts' in cols else None
                    last_day = _fmt_iso_day(r[cols['last_ts']]) if 'last_ts' in cols else None
                    sql_recent = (
                        "select count_distinct(cast(timestamp as LONG)/86400000000) as have_days "
                        f"from options_data where underlying = '{u}' and timestamp >= dateadd('d', -30, now())"
                    )
                    d2 = await _q(session, sql_recent)
                    have_days = 0
                    if d2.get('dataset'):
                        c2 = {c['name']: i for i, c in enumerate(d2.get('columns', []))}
                        try:
                            have_days = int(d2['dataset'][0][c2['have_days']])
                        except Exception:
                            have_days = 0
                    out.append({
                        "underlying": u,
                        "rows": rows,
                        "contracts": contracts,
                        "first_day": first_day,
                        "last_day": last_day,
                        "recent_gap_days_30d": max(0, 30 - have_days),
                    })
                except Exception as e:
                    out.append({"underlying": u, "error": str(e)})
        return {"generated_at": datetime.utcnow().isoformat(), "coverage": out}

    @app.post('/coverage/options/export')
    async def coverage_options_export(req: OptionsCoverageRequest = Body(default=OptionsCoverageRequest())):
        try:
            report = await _compute_options_coverage(underlyings=req.underlyings, max_underlyings=int(req.max_underlyings))
            # Persist stable and dated JSON under export dir like scheduled loop
            out_dir = os.getenv("OPTIONS_COVERAGE_OUTPUT_DIR", "/app/export/grafana-csv").rstrip("/")
            os.makedirs(out_dir, exist_ok=True)
            stable_path = os.path.join(out_dir, 'options_coverage.json')
            dated = os.path.join(out_dir, f"options_coverage_{datetime.utcnow().strftime('%Y%m%d')}.json")
            with open(stable_path, 'w') as f:
                json.dump(report, f, indent=2)
            with open(dated, 'w') as f:
                json.dump(report, f, indent=2)
            return {"status": "ok", "items": len(report.get('coverage', [])), "path": stable_path}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    class EquitiesCoverageRequest(BaseModel):
        sample: bool = Field(default=False)
        min_years: float = Field(default=19.5, ge=1.0, le=30.0)

    async def _compute_equities_coverage(sample: bool = False, min_years: float = 19.5) -> dict:
        """Compute equities coverage summary across symbols from QuestDB daily bars.
        Returns aggregate counts and ratios for dashboard JSON.
        """
        try:
            # Discover distinct symbols (sampled or all depending on flag)
            data = await _qdb_exec("select distinct symbol from market_data where symbol is not null and symbol != ''")
            cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
            syms: List[str] = []
            for r in data.get('dataset', []) or []:
                try:
                    s = str(r[cols['symbol']]).strip().upper()
                    if s:
                        syms.append(s)
                except Exception:
                    continue
            if sample and len(syms) > 500:
                try:
                    import random as _rnd
                    _rnd.shuffle(syms)
                except Exception:
                    pass
                syms = syms[:500]
            total = len(syms)
            min_days = int(min_years * 365.25 * 0.975)  # ~2.5% tolerance on calendar
            have_20y = 0
            # Count coverage per symbol (cheap count of distinct days via cast to LONG/86400e9)
            for i in range(0, total, 200):
                batch = syms[i:i+200]
                in_list = ",".join(["'" + s.replace("'","''") + "'" for s in batch])
                sql = (
                    "select symbol, count_distinct(cast(timestamp as LONG)/86400000000) as d from market_data "
                    f"where symbol in ({in_list}) group by symbol"
                )
                part = await _qdb_exec(sql)
                c = {c['name']: i for i, c in enumerate(part.get('columns', []) or [])}
                for r in part.get('dataset', []) or []:
                    try:
                        days = int(r[c.get('d')])
                        if days >= min_days:
                            have_20y += 1
                    except Exception:
                        continue
            ratio = (have_20y / total) if total else 0.0
            # Update Prometheus gauges if available
            try:
                if EQUITIES_COVERAGE_RATIO_20Y is not None:
                    EQUITIES_COVERAGE_RATIO_20Y.set(ratio)
                if EQUITIES_COVERAGE_SYMBOLS_TOTAL is not None:
                    EQUITIES_COVERAGE_SYMBOLS_TOTAL.set(total)
                if EQUITIES_COVERAGE_SYMBOLS_20Y is not None:
                    EQUITIES_COVERAGE_SYMBOLS_20Y.set(have_20y)
            except Exception:
                pass
            return {
                'generated_at': datetime.utcnow().isoformat(),
                'symbols_evaluated': total,
                'symbols_with_min_years': have_20y,
                'min_years': min_years,
                'coverage_ratio': ratio,
            }
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f'equities_coverage_compute_failed:{e}')

    @app.post('/coverage/equities/export')
    async def coverage_equities_export(req: EquitiesCoverageRequest = Body(default=EquitiesCoverageRequest())):
        try:
            report = await _compute_equities_coverage(sample=bool(req.sample), min_years=float(req.min_years))
            out_dir = os.getenv("EQUITIES_COVERAGE_OUTPUT_DIR", "/app/export/grafana-csv").rstrip("/")
            os.makedirs(out_dir, exist_ok=True)
            stable = os.path.join(out_dir, 'equities_coverage.json')
            dated = os.path.join(out_dir, f"equities_coverage_{datetime.utcnow().strftime('%Y%m%d')}.json")
            with open(stable, 'w') as f:
                json.dump(report, f, indent=2)
            with open(dated, 'w') as f:
                json.dump(report, f, indent=2)
            return {"status": "ok", "path": stable, "symbols": report.get('symbols_evaluated')}
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Extended health (detailed overview) ---------------------- #
# NOTE: A lightweight /health/extended endpoint is defined earlier for fast aggregation.
# This detailed variant is exposed under /health/overview to avoid overriding the fast path.
@app.get("/health/overview")
async def extended_health():
    """Aggregate ingestion health details used by API /health/full.

    Includes:
      - provider HTTP response summaries (last scrape window)
      - provider rate-limit counts by endpoint
      - background loop statuses (enabled, last_run, last_error)
    """
    out: dict[str, Any] = {"status": "ok"}
    # Provider metrics snapshots (best-effort; Prom client registry is process-local)
    providers: dict[str, Any] = {"http_codes": {}, "rate_limits": {}}
    try:
        # HTTP response codes
        if PROVIDER_HTTP_RESPONSES_TOTAL is not None:
            # Each child has labels: provider, endpoint, code
            for sample in getattr(PROVIDER_HTTP_RESPONSES_TOTAL, '_samples', []):  # type: ignore[attr-defined]
                try:
                    labels = sample.labels  # type: ignore[attr-defined]
                    p = labels.get('provider'); ep = labels.get('endpoint'); code = labels.get('code')
                    if not p or not ep or code is None:
                        continue
                    bucket = providers["http_codes"].setdefault(p, {}).setdefault(ep, {})
                    # sample.value is cumulative; surface current value
                    bucket[code] = sample.value  # type: ignore[attr-defined]
                except Exception:
                    continue
        # Rate limits
        if PROVIDER_RATE_LIMIT_TOTAL is not None:
            for sample in getattr(PROVIDER_RATE_LIMIT_TOTAL, '_samples', []):  # type: ignore[attr-defined]
                try:
                    labels = sample.labels  # type: ignore[attr-defined]
                    p = labels.get('provider'); ep = labels.get('endpoint')
                    if not p or not ep:
                        continue
                    cur = providers["rate_limits"].setdefault(p, {})
                    cur[ep] = sample.value  # type: ignore[attr-defined]
                except Exception:
                    continue
    except Exception:
        pass
    out['provider_metrics'] = providers
    # Loop status snapshot
    try:
        out['ingestion_pipelines'] = {k: dict(v) for k, v in LOOP_STATUS.items()}
        # Aggregate recent errors across loops
        out['ingestion_errors_aggregated'] = {k: v.get('last_error') for k, v in LOOP_STATUS.items() if v.get('last_error')}
    except Exception:
        out['ingestion_pipelines'] = {}
    return out


# ---------------------- MinIO artifact upload helper (consolidated) ---------------------- #
async def minio_upload_artifacts(*, directory: str, bucket: str, prefix: str = "", pattern: str = "*.json") -> dict:
    """Upload artifact files from a directory to MinIO (S3-compatible).

    Best-effort behavior: if dependencies/credentials are missing, return a summary without raising.
    Returns: {attempted, uploaded, skipped, errors, keys?: [first few keys], error?: str}
    """
    attempted = 0
    uploaded = 0
    errors = 0
    keys: list[str] = []
    try:
        files = sorted(glob.glob(os.path.join(directory, pattern)))
    except Exception:
        files = []
    if not files:
        return {"attempted": 0, "uploaded": 0, "skipped": 0, "errors": 0, "keys": []}
    endpoint = os.getenv("MINIO_ENDPOINT", "").strip() or os.getenv("MINIO_URL", "").strip() or "trading-minio:9000"
    access_key = os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "")).strip()
    secret_key = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "")).strip()
    if not secret_key:
        b64 = os.getenv("MINIO_SECRET_KEY_B64", "").strip()
        if b64:
            try:
                secret_key = base64.b64decode(b64).decode("utf-8", errors="ignore")
            except Exception:
                secret_key = ""
    if not (endpoint and access_key and secret_key and bucket):
        return {"attempted": 0, "uploaded": 0, "skipped": len(files), "errors": 0, "keys": [], "error": "missing_minio_configuration"}
    # Prefer boto3; fallback to minio SDK
    pref = prefix.strip().strip("/")
    try:
        import boto3  # type: ignore
        from botocore.config import Config as _BotoCfg  # type: ignore
        endpoint_url = endpoint if endpoint.startswith("http://") or endpoint.startswith("https://") else f"http://{endpoint}"
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=_BotoCfg(signature_version="s3v4", retries={"max_attempts": 2, "mode": "standard"}),
        )
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:
            try:
                s3.create_bucket(Bucket=bucket)
            except Exception:
                pass
        for path in files:
            attempted += 1
            key = os.path.basename(path)
            if pref:
                key = f"{pref}/{key}"
            try:
                s3.upload_file(path, bucket, key)
                uploaded += 1
                if len(keys) < 5:
                    keys.append(key)
            except Exception as e:  # noqa: BLE001
                errors += 1
                logger.warning("MinIO upload failed", file=path, key=key, error=str(e))
        return {"attempted": attempted, "uploaded": uploaded, "skipped": attempted - uploaded, "errors": errors, "keys": keys}
    except Exception as e:  # noqa: BLE001
        # Fallback to minio SDK
        try:
            from minio import Minio  # type: ignore
            secure = endpoint.startswith("https://") or os.getenv("MINIO_SECURE", "false").lower() in ("1","true","yes")
            ep = endpoint.replace("https://", "").replace("http://", "")
            client = Minio(ep, access_key=access_key, secret_key=secret_key, secure=secure)
            try:
                if not client.bucket_exists(bucket):
                    client.make_bucket(bucket)
            except Exception:
                pass
            for path in files:
                attempted += 1
                key = os.path.basename(path)
                if pref:
                    key = f"{pref}/{key}"
                try:
                    client.fput_object(bucket, key, path)
                    uploaded += 1
                    if len(keys) < 5:
                        keys.append(key)
                except Exception as ie:  # noqa: BLE001
                    errors += 1
                    logger.warning("MinIO upload failed (minio SDK)", file=path, key=key, error=str(ie))
            return {"attempted": attempted, "uploaded": uploaded, "skipped": attempted - uploaded, "errors": errors, "keys": keys}
        except Exception:
            return {"attempted": 0, "uploaded": 0, "skipped": len(files), "errors": 1, "keys": [], "error": str(e)[:200]}


# ---------------------- Coverage Endpoints ---------------------- #
def _fmt_iso_day(val) -> Optional[str]:
    try:
        s = str(val)
        return s[:10]
    except Exception:
        return None


# ---------------------- Basic health endpoints ---------------------- #
@app.get("/healthz")
async def healthz():
    """Lightweight health probe for liveness checks."""
    try:
        return {"status": "ok", "service": "data-ingestion", "time": datetime.utcnow().isoformat()}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)[:200]})


@app.get("/coverage/equities")
async def equities_coverage(sample: bool = Query(default=False), min_years: float = Query(default=19.5)):
    """Compute equities daily coverage span across symbols in market_data.

    Returns total symbols scanned, how many meet >= min_years coverage, and ratio.
    """
    # Select a sample of symbols to keep load light when requested
    limit_clause = " limit 200" if sample else ""
    # Derive span in days per symbol (based on designated timestamp column 'timestamp')
    sql = (
        "select symbol, min(timestamp) first_ts, max(timestamp) last_ts, "
        "datediff('d', min(timestamp), max(timestamp)) as span_days "
        "from market_data group by symbol" + limit_clause
    )
    try:
        data = await _qdb_exec(sql)
        cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
        rows = data.get('dataset') or []
        evaluated = len(rows)
        meet = 0
        items: list[dict] = []
        for r in rows:
            try:
                span_days = float(r[cols['span_days']]) if 'span_days' in cols else 0.0
                years = span_days / 365.25
                if years >= float(min_years):
                    meet += 1
                items.append({
                    'symbol': str(r[cols['symbol']]),
                    'first_day': _fmt_iso_day(r[cols['first_ts']]) if 'first_ts' in cols else None,
                    'last_day': _fmt_iso_day(r[cols['last_ts']]) if 'last_ts' in cols else None,
                    'span_years': round(years, 2),
                })
            except Exception:
                continue
        ratio = (meet / evaluated) if evaluated else 0.0
        try:
            if EQUITIES_COVERAGE_RATIO_20Y is not None and not sample:
                EQUITIES_COVERAGE_RATIO_20Y.set(ratio)
            if EQUITIES_COVERAGE_SYMBOLS_TOTAL is not None and not sample:
                EQUITIES_COVERAGE_SYMBOLS_TOTAL.set(evaluated)
            if EQUITIES_COVERAGE_SYMBOLS_20Y is not None and not sample:
                EQUITIES_COVERAGE_SYMBOLS_20Y.set(meet)
        except Exception:
            pass
        return {
            'symbols_evaluated': evaluated,
            'symbols_meet_threshold': meet,
            'ratio': ratio,
            'min_years': float(min_years),
            'items': items if sample else None,
        }
    except HTTPException as e:
        # Fallback to an overall summary without grouping to avoid transient/compat errors
        try:
            summary = await _qdb_exec(
                "select count() as rows, count_distinct(symbol) as symbols, min(timestamp) as first_ts, max(timestamp) as last_ts from market_data"
            )
            cols = {c['name']: i for i, c in enumerate(summary.get('columns', []) or [])}
            r = (summary.get('dataset') or [None])[0]
            if not r:
                return {'error': 'equities_coverage_failed', 'detail': str(e)}
            span_years = None
            try:
                # Use datediff in a separate call to get span days reliably
                dd = await _qdb_exec("select datediff('d', (select min(timestamp) from market_data), (select max(timestamp) from market_data)) as d")
                c2 = {c['name']: i for i, c in enumerate(dd.get('columns', []) or [])}
                drow = (dd.get('dataset') or [[0]])[0]
                dval = float(drow[c2.get('d', 0)]) if isinstance(drow, list) else float(drow)
                span_years = round(dval / 365.25, 2)
            except Exception:
                span_years = None
            return {
                'symbols_evaluated': int(r[cols.get('symbols', -1)] or 0),
                'symbols_meet_threshold': None,
                'ratio': None,
                'min_years': float(min_years),
                'overall': {
                    'rows': int(r[cols.get('rows', -1)] or 0),
                    'symbols': int(r[cols.get('symbols', -1)] or 0),
                    'first_day': _fmt_iso_day(r[cols.get('first_ts')]) if 'first_ts' in cols else None,
                    'last_day': _fmt_iso_day(r[cols.get('last_ts')]) if 'last_ts' in cols else None,
                    'span_years': span_years,
                },
                'items': None,
                'note': 'returned fallback summary due to QuestDB HTTP error',
            }
        except Exception:
            # If fallback also fails, bubble a concise error
            raise e


@app.get("/coverage/options")
async def options_coverage(underlying: Optional[str] = None):
    """Coverage summary for options_data table.

    If underlying is provided, returns per-underlying stats; otherwise overall totals.
    """
    if underlying:
        sql = (
            "select count() as rows, count_distinct(option_symbol) as contracts, "
            "min(timestamp) as first_ts, max(timestamp) as last_ts "
            f"from options_data where underlying = '{underlying.upper()}'"
        )
        data = await _qdb_exec(sql)
        cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
        r = (data.get('dataset') or [None])[0]
        if not r:
            return {"underlying": underlying.upper(), "rows": 0, "contracts": 0}
        return {
            "underlying": underlying.upper(),
            "rows": int(r[cols.get('rows', -1)] or 0),
            "contracts": int(r[cols.get('contracts', -1)] or 0),
            "first_day": _fmt_iso_day(r[cols.get('first_ts')]) if 'first_ts' in cols else None,
            "last_day": _fmt_iso_day(r[cols.get('last_ts')]) if 'last_ts' in cols else None,
        }
    # overall
    sql = (
        "select count() as rows, count_distinct(option_symbol) as contracts, count_distinct(underlying) as underlyings, "
        "min(timestamp) as first_ts, max(timestamp) as last_ts from options_data"
    )
    data = await _qdb_exec(sql)
    cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
    r = (data.get('dataset') or [None])[0]
    if not r:
        return {"rows": 0, "contracts": 0, "underlyings": 0}
    return {
        "rows": int(r[cols.get('rows', -1)] or 0),
        "contracts": int(r[cols.get('contracts', -1)] or 0),
        "underlyings": int(r[cols.get('underlyings', -1)] or 0),
        "first_day": _fmt_iso_day(r[cols.get('first_ts')]) if 'first_ts' in cols else None,
        "last_day": _fmt_iso_day(r[cols.get('last_ts')]) if 'last_ts' in cols else None,
    }


@app.get("/coverage/news")
async def news_coverage():
    """Coverage summary for news_items table."""
    sql = (
        "select count() as rows, count_distinct(symbol) as symbols, min(ts) as first_ts, max(ts) as last_ts from news_items"
    )
    data = await _qdb_exec(sql)
    cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
    r = (data.get('dataset') or [None])[0]
    if not r:
        return {"rows": 0, "symbols": 0}
    return {
        "rows": int(r[cols.get('rows', -1)] or 0),
        "symbols": int(r[cols.get('symbols', -1)] or 0),
        "first_day": _fmt_iso_day(r[cols.get('first_ts')]) if 'first_ts' in cols else None,
        "last_day": _fmt_iso_day(r[cols.get('last_ts')]) if 'last_ts' in cols else None,
    }


@app.get("/coverage/social")
async def social_coverage():
    """Coverage summary for social_signals (QuestDB).

    Detects whether the table exists and which timestamp column to use (ts or timestamp).
    Returns rows, count of distinct symbols, and first/last day.
    """
    # Try direct query; if table missing, ensure schema and retry once.
    # Additionally, verify table existence via 'show tables' to avoid false negatives.
    async def _table_exists() -> bool:
        try:
            meta = await _qdb_exec("show tables")
            cols = {c.get('name'): i for i, c in enumerate(meta.get('columns', []) or [])}
            name_idx = cols.get('table') if 'table' in cols else 0
            for r in meta.get('dataset') or []:
                try:
                    if str(r[name_idx]).strip().lower() == 'social_signals':
                        return True
                except Exception:
                    continue
        except Exception:
            # If metadata call fails, don't block – treat as unknown/False
            return False
        return False
    async def _pick_ts_col() -> str:
        try:
            cols_meta = await _qdb_exec("show columns from social_signals")
            name_idx = next((i for i, c in enumerate(cols_meta.get('columns', []) or []) if c.get('name') == 'column'), None)
            available_cols: list[str] = []
            for r in cols_meta.get('dataset') or []:
                try:
                    if name_idx is not None:
                        available_cols.append(str(r[name_idx]))
                except Exception:
                    continue
            if 'ts' in available_cols:
                return 'ts'
            if 'timestamp' in available_cols:
                return 'timestamp'
        except Exception:
            pass
        return 'ts'

    async def _run_query(ts_col: str) -> dict:
        sql = (
            "select count() as rows, count_distinct(symbol) as symbols, "
            f"min({ts_col}) as first_ts, max({ts_col}) as last_ts from social_signals"
        )
        return await _qdb_exec(sql)

    try:
        ts_col = await _pick_ts_col()
        data = await _run_query(ts_col)
    except HTTPException as e:
        detail = str(e.detail)
        # Confirm existence via metadata to avoid misleading message
        exists = await _table_exists()
        if 'table' in detail.lower() and 'not found' in detail.lower():
            # If it truly doesn't exist, ensure schema then retry once
            if not exists:
                try:
                    await _qdb_exec(
                        "create table if not exists social_signals ("
                        "symbol symbol, source symbol, sentiment double, engagement double, influence double, "
                        "author string, url string, content string, ts timestamp) timestamp(ts) PARTITION BY DAY"
                    )
                    ts_col = await _pick_ts_col()
                    data = await _run_query(ts_col)
                except Exception:
                    return {"rows": 0, "symbols": 0, "note": "social_signals table not found", "diagnostic": {"exists": False}}
            else:
                # Table exists but query still failed – return explicit error to surface mismatch
                return JSONResponse(status_code=502, content={
                    "error": "questdb_mismatch",
                    "detail": e.detail,
                    "diagnostic": {"exists": True, "hint": "Verify timestamp column and permissions"}
                })
        else:
            return JSONResponse(status_code=e.status_code, content={"error": "questdb_error", "detail": e.detail})
    cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
    r = (data.get('dataset') or [None])[0]
    if not r:
        return {"rows": 0, "symbols": 0}
    return {
        "rows": int(r[cols.get('rows', -1)] or 0),
        "symbols": int(r[cols.get('symbols', -1)] or 0),
        "first_day": _fmt_iso_day(r[cols.get('first_ts')]) if 'first_ts' in cols else None,
        "last_day": _fmt_iso_day(r[cols.get('last_ts')]) if 'last_ts' in cols else None,
    }


@app.get("/coverage/summary")
async def coverage_summary(sample: bool = Query(default=True)):
    """Consolidated coverage snapshot across equities, options, and news.

    Robust against partial failures; returns available sections with error notes instead of 500.
    """
    out: dict = {"timestamp": datetime.utcnow().isoformat()}
    try:
        # When calling endpoint functions internally, avoid passing FastAPI's Query defaults.
        # Explicitly provide a numeric min_years to prevent float(Query) errors.
        out["equities"] = await equities_coverage(sample=sample, min_years=19.5)
    except Exception as e:  # noqa: BLE001
        out["equities"] = {"error": str(e)[:200]}
    try:
        out["options"] = await options_coverage()
    except Exception as e:  # noqa: BLE001
        out["options"] = {"error": str(e)[:200]}
    try:
        out["news"] = await news_coverage()
    except Exception as e:  # noqa: BLE001
        out["news"] = {"error": str(e)[:200]}
    # Social coverage is optional; return error note instead of failing the entire summary
    try:
        out["social"] = await social_coverage()
    except Exception as e:  # noqa: BLE001
        out["social"] = {"error": str(e)[:200]}
    return out


@app.get("/coverage/postgres")
async def postgres_coverage():
    """Summarize key relational tables in Postgres (best-effort).

    This function now:
    - Detects whether a target table exists before querying it.
    - Introspects columns to pick a reasonable timestamp/date column for min/max when schema varies.
    - Returns structured errors instead of raising, so coverage remains robust under partial availability.
    """
    engine = None
    db = None
    # Try shared manager first
    try:
        from trading_common.database import get_database_manager  # type: ignore
        db = get_database_manager()
        if db and hasattr(db, "get_async_engine"):
            engine = db.get_async_engine()  # type: ignore[attr-defined]
    except Exception:
        engine = None
    # Fallback to direct engine
    if engine is None:
        try:
            import sqlalchemy as sa  # type: ignore
            from sqlalchemy.ext.asyncio import create_async_engine  # type: ignore
            url = os.getenv("DB_POSTGRES_URL") or os.getenv("DATABASE_URL")
            if not url:
                raise HTTPException(status_code=503, detail="postgres url not configured")
            if "asyncpg" not in url:
                if url.startswith("postgresql://"):
                    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
                elif url.startswith("postgres://"):
                    url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            engine = create_async_engine(url, pool_pre_ping=True)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"postgres engine init failed: {e}")
    import sqlalchemy as sa  # type: ignore

    async def _q(text_sql: str):
        try:
            async with engine.connect() as conn:  # type: ignore[arg-type]
                res = await conn.execute(sa.text(text_sql))
                rows = res.fetchall()
                cols = list(res.keys())
                return cols, rows
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    out: dict = {"timestamp": datetime.utcnow().isoformat(), "tables": {}}

    async def _table_exists(table: str) -> bool:
        cols, rows = await _q(
            "SELECT 1 FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog','information_schema') AND table_name = :t LIMIT 1".replace(":t", f"'{table}'")
        )
        return isinstance(cols, list) and bool(rows)

    async def _pick_time_column(table: str) -> str | None:
        # Prefer timestamptz/timestamp columns, then date, then any column named like time/ts/date
        cols, rows = await _q(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :t".replace(":t", f"'{table}'")
        )
        if not isinstance(cols, list) or not rows:
            return None
        # Build list of (name, type)
        headers = {k: i for i, k in enumerate(cols)}
        candidates_ts = []
        candidates_date = []
        name_hints = []
        for r in rows:
            name = r[headers.get("column_name", 0)]
            dtype = (r[headers.get("data_type", 1)] or "").lower()
            if "timestamp" in dtype:
                candidates_ts.append(name)
            elif dtype == "date":
                candidates_date.append(name)
            if any(h in str(name).lower() for h in ["timestamp", "ts", "time", "date", "dt", "at"]):
                name_hints.append(name)
        for lst in (candidates_ts, candidates_date, name_hints):
            if lst:
                return lst[0]
        return None

    # Include common table names used by this stack; we check existence before querying
    targets = [
        ("historical_daily_bars", None),
        ("news_events", "published_at"),
        ("news_items", None),
        ("options_daily", None),
        ("options_data", None),
    ]
    for name, fixed_time_col in targets:
        try:
            if not await _table_exists(name):
                out["tables"][name] = {"error": "table_not_found"}
                continue
            time_col = fixed_time_col or (await _pick_time_column(name)) or "timestamp"
            # Build safe SQL using inferred column
            sql = f"SELECT count(*) AS rows, min({time_col}) AS first_ts, max({time_col}) AS last_ts FROM {name}"
            cols, rows = await _q(sql)
            if isinstance(cols, list) and rows:
                r = rows[0]
                def _g(k):
                    try:
                        idx = cols.index(k)
                        return r[idx]
                    except Exception:
                        return None
                out["tables"][name] = {
                    "rows": int(_g("rows") or 0),
                    "first_day": _fmt_iso_day(_g("first_ts")),
                    "last_day": _fmt_iso_day(_g("last_ts")),
                }
            else:
                out["tables"][name] = {"error": rows if isinstance(rows, str) else "unavailable"}
        except Exception as e:  # noqa: BLE001
            out["tables"][name] = {"error": str(e)}
    return out


# ---------------------- Calendar backfill admin endpoints (provider-routed) ---------------------- #
@app.post("/calendar/backfill")
async def calendar_backfill(
    years: int = Query(default=5, ge=1, le=10, description="Lookback horizon in years"),
    include: str = Query(default="earnings,ipo,splits", description="Comma list: earnings,ipo,splits,dividends"),
):
    """Run bounded backfill for calendar datasets with provider routing (default 5 years).

    - Earnings & IPO come from Alpha Vantage when available; otherwise EODHD range collectors.
    - Splits & Dividends come from EODHD range collectors.
    """
    if not calendar_svc or not getattr(calendar_svc, 'enabled', False):
        return JSONResponse(status_code=503, content={"status": "error", "detail": "calendar service not available"})
    end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(years * 365.25))
    parts = [p.strip().lower() for p in include.split(',') if p.strip()]
    results: dict = {}
    has_av = bool(getattr(calendar_svc, 'av_api_key', None))
    try:
        if 'earnings' in parts:
            try:
                if has_av:
                    results['earnings'] = await calendar_svc.collect_av_earnings_upcoming(horizon='12month')
                    results['earnings_provider'] = 'alphavantage'
                else:
                    results['earnings'] = await calendar_svc.collect_earnings_range(start_dt, end_dt)
                    results['earnings_provider'] = 'eodhd'
            except Exception as e:  # noqa: BLE001
                results['earnings_error'] = str(e)[:200]
        if 'ipo' in parts:
            try:
                if has_av:
                    results['ipo'] = await calendar_svc.collect_av_ipo_upcoming()
                    results['ipo_provider'] = 'alphavantage'
                else:
                    results['ipo'] = await calendar_svc.collect_ipo_range(start_dt, end_dt)
                    results['ipo_provider'] = 'eodhd'
            except Exception as e:  # noqa: BLE001
                results['ipo_error'] = str(e)[:200]
        if 'splits' in parts:
            try:
                results['splits'] = await calendar_svc.collect_splits_range(start_dt, end_dt)
            except Exception as e:  # noqa: BLE001
                results['splits_error'] = str(e)[:200]
        if 'dividends' in parts:
            try:
                results['dividends'] = await calendar_svc.collect_dividends_range(start_dt, end_dt)
            except Exception as e:  # noqa: BLE001
                results['dividends_error'] = str(e)[:200]
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)[:200]})
    return {"status": "ok", "start": start_dt.strftime('%Y-%m-%d'), "end": end_dt.strftime('%Y-%m-%d'), **results}

@app.get("/calendar/backfill/run")
async def calendar_backfill_run(
    years: int = Query(default=1, ge=1, le=10, description="Lookback horizon in years"),
    include: str = Query(default="earnings,ipo,splits,dividends", description="Comma list: earnings,ipo,splits,dividends"),
):
    """Schedule a non-blocking calendar backfill job and return immediately.

    This endpoint triggers the same collectors as /calendar/backfill but runs them in the background
    to avoid client timeouts for long ranges. Returns a job id and the requested parameters.
    """
    if not calendar_svc or not getattr(calendar_svc, 'enabled', False):
        return JSONResponse(status_code=503, content={"status": "error", "detail": "calendar service not available"})
    end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(years * 365.25))
    parts = [p.strip().lower() for p in include.split(',') if p.strip()]
    job_id = f"calendar:{int(time.time())}"

    async def _run():
        summary: dict[str, int | str] = {"start": start_dt.strftime('%Y-%m-%d'), "end": end_dt.strftime('%Y-%m-%d')}
        has_av = bool(getattr(calendar_svc, 'av_api_key', None))
        try:
            if 'earnings' in parts:
                try:
                    if has_av:
                        summary['earnings'] = await calendar_svc.collect_av_earnings_upcoming(horizon='12month')  # type: ignore[index]
                        summary['earnings_provider'] = 'alphavantage'
                    else:
                        summary['earnings'] = await calendar_svc.collect_earnings_range(start_dt, end_dt)  # type: ignore[index]
                        summary['earnings_provider'] = 'eodhd'
                except Exception as e:  # noqa: BLE001
                    summary['earnings_error'] = str(e)[:200]
            if 'ipo' in parts:
                try:
                    if has_av:
                        summary['ipo'] = await calendar_svc.collect_av_ipo_upcoming()  # type: ignore[index]
                        summary['ipo_provider'] = 'alphavantage'
                    else:
                        summary['ipo'] = await calendar_svc.collect_ipo_range(start_dt, end_dt)  # type: ignore[index]
                        summary['ipo_provider'] = 'eodhd'
                except Exception as e:  # noqa: BLE001
                    summary['ipo_error'] = str(e)[:200]
            if 'splits' in parts:
                try:
                    summary['splits'] = await calendar_svc.collect_splits_range(start_dt, end_dt)  # type: ignore[index]
                except Exception as e:  # noqa: BLE001
                    summary['splits_error'] = str(e)[:200]
            if 'dividends' in parts:
                try:
                    summary['dividends'] = await calendar_svc.collect_dividends_range(start_dt, end_dt)  # type: ignore[index]
                except Exception as e:  # noqa: BLE001
                    summary['dividends_error'] = str(e)[:200]
            logger.info("Calendar backfill job complete", job_id=job_id, years=years, include=parts, summary=summary)
        except Exception as e:  # noqa: BLE001
            logger.warning("Calendar backfill job failed", job_id=job_id, error=str(e))

    # Schedule fire-and-forget task on the running event loop and return immediately.
    # Avoid BackgroundTasks here because it executes in a threadpool without an event loop.
    try:
        asyncio.create_task(_run())
    except Exception:
        # As a fallback, run in detached task via loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_run())
        except Exception:
            # If scheduling fails, report accepted but with a warning hint in logs
            logger.warning("Failed to schedule calendar backfill background task")
    return {"status": "accepted", "job_id": job_id, "years": int(years), "include": parts}

@app.get("/calendar/coverage")
async def calendar_coverage():
    """Return basic counts for calendar tables (best-effort)."""
    out: dict = {}
    try:
        data = await _qdb_exec("select count() as rows, min(timestamp) as first_ts, max(timestamp) as last_ts from earnings_calendar")
        if data.get('dataset'):
            c = {c['name']: i for i, c in enumerate(data.get('columns', []))}
            r = data['dataset'][0]
            out['earnings'] = {"rows": int(r[c.get('rows',0)] or 0), "first_day": _fmt_iso_day(r[c.get('first_ts')]), "last_day": _fmt_iso_day(r[c.get('last_ts')])}
    except Exception:
        out['earnings'] = {"rows": 0}
    try:
        data = await _qdb_exec("select count() as rows, min(timestamp) as first_ts, max(timestamp) as last_ts from ipo_calendar")
        if data.get('dataset'):
            c = {c['name']: i for i, c in enumerate(data.get('columns', []))}
            r = data['dataset'][0]
            out['ipo'] = {"rows": int(r[c.get('rows',0)] or 0), "first_day": _fmt_iso_day(r[c.get('first_ts')]), "last_day": _fmt_iso_day(r[c.get('last_ts')])}
    except Exception:
        out['ipo'] = {"rows": 0}
    try:
        data = await _qdb_exec("select count() as rows, min(timestamp) as first_ts, max(timestamp) as last_ts from splits_calendar")
        if data.get('dataset'):
            c = {c['name']: i for i, c in enumerate(data.get('columns', []))}
            r = data['dataset'][0]
            out['splits'] = {"rows": int(r[c.get('rows',0)] or 0), "first_day": _fmt_iso_day(r[c.get('first_ts')]), "last_day": _fmt_iso_day(r[c.get('last_ts')])}
    except Exception:
        out['splits'] = {"rows": 0}
    try:
        data = await _qdb_exec("select count() as rows, min(timestamp) as first_ts, max(timestamp) as last_ts from dividends_calendar")
        if data.get('dataset'):
            c = {c['name']: i for i, c in enumerate(data.get('columns', []))}
            r = data['dataset'][0]
            out['dividends'] = {"rows": int(r[c.get('rows',0)] or 0), "first_day": _fmt_iso_day(r[c.get('first_ts')]), "last_day": _fmt_iso_day(r[c.get('last_ts')])}
    except Exception:
        out['dividends'] = {"rows": 0}
    return out


# ---------------------- Vector admin endpoint: options index ---------------------- #
@app.post("/admin/vector/options/index")
async def admin_vector_options_index(
    underlyings: Optional[str] = Query(default=None, description="Comma-separated underlyings (default: AAPL,MSFT,SPY)"),
    limit: int = Query(default=200, ge=1, le=2000),
    days: int = Query(default=7, ge=1, le=90),
):
    """Index a bounded set of option contracts from QuestDB into Weaviate.

    Pulls recent rows from options_data and builds OptionContract objects. Uses fallback direct indexer
    so it doesn't depend on ML service routes. Adapts to available columns.
    """
    if not index_options_fallback:
        return JSONResponse(status_code=503, content={"status": "error", "detail": "index_options_fallback not available"})
    # Resolve underlyings
    syms: list[str] = []
    if underlyings:
        syms = [s.strip().upper() for s in underlyings.split(',') if s and s.strip()]
    elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            syms = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            syms = []
    if not syms:
        syms = ['AAPL','MSFT','SPY']
    # Discover available columns on options_data to build compatible projection
    try:
        meta = await _qdb_exec("show columns from options_data")
        name_idx = next((i for i, c in enumerate(meta.get('columns', []) or []) if c.get('name') == 'column'), None)
        cols_available: list[str] = []
        for r in meta.get('dataset') or []:
            try:
                if name_idx is not None:
                    cols_available.append(str(r[name_idx]))
            except Exception:
                continue
    except Exception:
        cols_available = []
    # Build select list using available columns; map to output keys
    select_parts = []
    mapping = {}
    def add(col: str, as_name: str):
        nonlocal select_parts, mapping
        if col in cols_available:
            select_parts.append(f"{col} as {as_name}")
            mapping[as_name] = as_name
    # Common schema variants
    add('underlying', 'underlying')
    add('option_symbol', 'option_symbol')
    add('expiry', 'expiry')
    # Right/option_type variants
    if 'right' in cols_available:
        select_parts.append("right as option_right")
        mapping['option_right'] = 'option_right'
    if 'option_type' in cols_available and 'option_right' not in mapping:
        select_parts.append("option_type as option_right")
        mapping['option_right'] = 'option_right'
    # Strike typed/text variants
    if 'strike' in cols_available:
        select_parts.append("strike as strike_val")
        mapping['strike_val'] = 'strike_val'
    # Timestamp/date variants
    if 'timestamp' in cols_available:
        select_parts.append("timestamp as ts_val")
        mapping['ts_val'] = 'ts_val'
    elif 'ts' in cols_available:
        select_parts.append("ts as ts_val")
        mapping['ts_val'] = 'ts_val'
    if not select_parts:
        return JSONResponse(status_code=500, content={"status": "error", "detail": "options_data has incompatible schema"})
    select_list = ", ".join(select_parts)
    # Construct SQL filtered by underlyings and recent days, limit overall rows
    sym_list = ", ".join([f"'{s}'" for s in syms])
    sql = (
        f"select {select_list} from options_data where underlying in ({sym_list}) "
        f"and ({'timestamp' if 'timestamp' in cols_available else 'ts'}) >= dateadd('d', -{int(days)}, now()) "
        f"order by {'timestamp' if 'timestamp' in cols_available else 'ts'} desc limit {int(limit)}"
    )
    try:
        data = await _qdb_exec(sql, timeout=20.0)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"status": "error", "detail": e.detail})
    cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
    items: list[dict] = []
    for r in data.get('dataset') or []:
        try:
            underlying = str(r[cols.get('underlying')]) if 'underlying' in cols else ''
            option_symbol = str(r[cols.get('option_symbol')]) if 'option_symbol' in cols else ''
            expiry = r[cols.get('expiry')] if 'expiry' in cols else None
            option_right = r[cols.get('option_right')] if 'option_right' in cols else None
            strike_val = r[cols.get('strike_val')] if 'strike_val' in cols else None
            ts_val = r[cols.get('ts_val')] if 'ts_val' in cols else None
            items.append({
                'underlying': underlying,
                'option_symbol': option_symbol,
                'expiry': str(expiry) if expiry is not None else None,
                'right': str(option_right) if option_right is not None else None,
                'strike': strike_val,
                'timestamp': str(ts_val) if ts_val is not None else None,
            })
        except Exception:
            continue
    indexed = 0
    if items:
        try:
            # Optionally pass redis for cross-run dedup if available
            redis = None
            try:
                if cache_client:
                    redis = getattr(cache_client, 'redis', None) or cache_client
            except Exception:
                redis = None
            indexed = await index_options_fallback(items, redis=redis)  # type: ignore[arg-type]
        except Exception:
            indexed = 0
    return {"status": "ok", "requested": len(items), "indexed": int(indexed), "underlyings": syms, "days": int(days), "limit": int(limit)}


@app.get("/coverage/source-usage")
async def coverage_source_usage(days: int = Query(default=30, ge=1, le=365)):
    """Summarize last-N-days market_data rows by data_source.

    Returns list of {data_source, rows} for the given lookback window.
    Useful for verifying value from providers (e.g., EODHD vs fallbacks).
    """
    # Introspect available columns to build a compatible query across schema variants
    available_columns: list[str] = []
    try:
        meta = await _qdb_exec("show columns from market_data")
        # The result has columns like ['column','type',...] with rows describing table columns
        name_idx = None
        for i, c in enumerate(meta.get('columns', []) or []):
            if c.get('name') == 'column':
                name_idx = i
                break
        for r in meta.get('dataset') or []:
            try:
                if name_idx is not None:
                    available_columns.append(str(r[name_idx]))
            except Exception:
                continue
    except Exception:
        # If introspection fails, continue with empty list and fallbacks
        available_columns = []

    # Prefer data_source; fallback to timeframe; else return total rows only
    group_col = None
    for cand in ('data_source', 'provider', 'source'):
        if cand in available_columns:
            group_col = cand
            break
    if group_col is None and 'timeframe' in available_columns:
        group_col = 'timeframe'

    if not group_col:
        # No suitable grouping column available – return total rows for the window
        total_sql = (
            "select count() as rows from market_data "
            f"where timestamp >= dateadd('d', -{int(days)}, now())"
        )
        try:
            d = await _qdb_exec(total_sql)
            cols = {c['name']: i for i, c in enumerate(d.get('columns', []) or [])}
            r = (d.get('dataset') or [None])[0]
            total = int(r[cols.get('rows', 0)]) if r else 0
            return {
                'days': days,
                'sources': [],
                'total_rows': total,
                'note': 'data_source column not found; returned total rows',
                'available_columns': available_columns,
            }
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={'error': 'questdb_error', 'detail': e.detail, 'available_columns': available_columns})

    sql = (
        f"select {group_col} as grp, count() as rows from market_data "
        f"where timestamp >= dateadd('d', -{int(days)}, now()) group by {group_col} order by rows desc"
    )
    try:
        data = await _qdb_exec(sql)
        cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
        out = []
        for r in data.get('dataset') or []:
            try:
                out.append({
                    'group': str(r[cols.get('grp', 0)]),
                    'rows': int(r[cols.get('rows', 1)]),
                })
            except Exception:
                continue
        result = {'days': days, 'group_by': group_col, 'sources': out}
        if group_col != 'data_source':
            result['note'] = 'grouped by fallback column due to missing data_source'
            result['available_columns'] = available_columns
        return result
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={'error': 'questdb_error', 'detail': e.detail, 'available_columns': available_columns})


# ---------------------- Vector admin endpoints (schema ensure + news reconcile) ---------------------- #
@app.post("/admin/vector/schema/ensure")
async def admin_vector_schema_ensure():
    """Ensure Weaviate schema classes exist (best-effort)."""
    try:
        try:
            from shared.vector.weaviate_schema import get_weaviate_client, ensure_desired_schema  # type: ignore
        except Exception:
            from ..shared.vector.weaviate_schema import get_weaviate_client, ensure_desired_schema  # type: ignore
        client = get_weaviate_client()
        result = ensure_desired_schema(client)
        return {"status": "ok", "result": result}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)[:200]})


@app.post("/admin/vector/news/reconcile")
async def admin_vector_news_reconcile(days: int = Query(default=3, ge=1, le=30), limit: int = Query(default=1000, ge=1, le=5000)):
    """Backfill recent news from QuestDB into vector store using fallback indexer.

    Adapts to available columns in news_items table. Expects at least title and ts; optional columns: source, url, symbol, sentiment, relevance, provider, value_score.
    """
    # Discover available columns
    try:
        meta = await _qdb_exec("show columns from news_items")
        name_idx = next((i for i, c in enumerate(meta.get('columns', []) or []) if c.get('name') == 'column'), None)
        cols_available = []
        for r in meta.get('dataset') or []:
            try:
                if name_idx is not None:
                    cols_available.append(str(r[name_idx]))
            except Exception:
                continue
    except Exception:
        cols_available = []
    # Build projection selecting only existing columns
    proj_parts = ["title", "ts"]
    optional = ["source", "url", "symbol", "sentiment", "relevance", "provider", "value_score"]
    for c in optional:
        if c in cols_available and c not in proj_parts:
            proj_parts.append(c)
    select_list = ", ".join(proj_parts)
    # Note: some deployments store 'symbol' as a tag of the item; keep optional
    sql = (
        f"select {select_list} from news_items "
        f"where ts >= dateadd('d', -{int(days)}, now()) and title is not null and title != '' "
        f"order by ts desc limit {int(limit)}"
    )
    try:
        data = await _qdb_exec(sql, timeout=25.0)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"status": "error", "detail": e.detail})
    cols_idx = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
    items = []
    for r in data.get('dataset') or []:
        try:
            title = str(r[cols_idx['title']]) if 'title' in cols_idx else ''
            if not title:
                continue
            published = str(r[cols_idx['ts']]) if 'ts' in cols_idx else datetime.utcnow().isoformat()
            content = ""  # not stored in current schema; leave blank
            source = str(r[cols_idx.get('source')]) if 'source' in cols_idx and r[cols_idx.get('source')] is not None else 'news_items'
            url = str(r[cols_idx.get('url')]) if 'url' in cols_idx and r[cols_idx.get('url')] is not None else ''
            symbol = r[cols_idx.get('symbol')] if 'symbol' in cols_idx else None
            syms = [str(symbol).upper()] if symbol else []
            items.append({
                'title': title,
                'content': content,
                'source': source,
                'published_at': published,
                'url': url,
                'symbols': syms,
            })
        except Exception:
            continue
    indexed = 0
    try:
        try:
            from shared.vector.indexing import index_news_fallback  # type: ignore
        except Exception:
            from ..shared.vector.indexing import index_news_fallback  # type: ignore
        if items:
            indexed = await index_news_fallback(items)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)[:200]})
    if indexed and VECTOR_NEWS_INDEXED is not None:
        try:
            VECTOR_NEWS_INDEXED.labels(path='admin_reconcile').inc(indexed)
        except Exception:
            pass
    if items and VECTOR_NEWS_LAST_TS is not None:
        try:
            from datetime import datetime as _dt
            last_ts = items[0]['published_at']
            _dtv = _dt.fromisoformat(str(last_ts).replace('Z','+00:00'))
            VECTOR_NEWS_LAST_TS.set(_dtv.timestamp())
        except Exception:
            pass
    return {"status": "ok", "requested": len(items), "indexed": indexed, "days": days, "limit": limit}


@app.post("/admin/social/test-insert")
async def admin_social_test_insert(symbol: str = Query(default="AAPL"), source: str = Query(default="test")):
    """Create social_signals table if missing and insert a single test row to validate QuestDB path."""
    try:
        # Create table if not exists
        await _qdb_exec(
            "create table if not exists social_signals ("
            "symbol symbol, source symbol, sentiment double, engagement double, influence double, "
            "author string, url string, content string, ts timestamp) timestamp(ts) PARTITION BY DAY"
        )
        # Insert a single row with current UTC
        from datetime import datetime as _dt
        ts_iso = _dt.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000000Z')
        sql = (
            "insert into social_signals(symbol,source,sentiment,engagement,influence,author,url,content,ts) values "
            f"('{symbol.upper()}','{source}',0.0,0.0,0.0,'tester','','ingestion test',to_timestamp('{ts_iso}','yyyy-MM-ddTHH:mm:ss.SSSSSSZ'))"
        )
        await _qdb_exec(sql)
        return {"status": "ok", "inserted": 1, "symbol": symbol.upper(), "source": source}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"status": "error", "detail": e.detail})
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)[:200]})


@app.post("/admin/options/schema/ensure")
async def admin_options_schema_ensure():
    """Ensure options_data table exists in QuestDB with expected column types.

    Schema: underlying symbol, option_symbol symbol, right string, strike string, expiry string,
            open double, high double, low double, close double, volume long, timestamp timestamp
            TIMESTAMP(timestamp) PARTITION BY DAY
    """
    try:
        await _qdb_exec(
            "create table if not exists options_data ("
            "underlying symbol, option_symbol symbol, right string, strike string, expiry string, "
            "open double, high double, low double, close double, volume long, timestamp timestamp) "
            "timestamp(timestamp) PARTITION BY DAY"
        )
        return {"status": "ok"}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"status": "error", "detail": e.detail})
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)[:200]})


class BackfillRequest(BaseModel):
    equities_years: int = Field(default=20, ge=1, le=25)
    options_years: int = Field(default=5, ge=1, le=10)
    news_years: int = Field(default=5, ge=1, le=10)
    social_years: int = Field(default=5, ge=1, le=10)
    pacing_seconds: float = Field(default=0.25, ge=0.0, le=5.0)
    max_symbols: int = Field(default=500, ge=1, le=5000)


class SocialBackfillRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to backfill; defaults to watchlist or fallback")
    years: float = Field(default=1.0, ge=0.1, le=10.0)


class NewsBackfillJobRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to backfill; defaults to watchlist or fallback")
    start: str = Field(..., description="Historical start date (YYYY-MM-DD)")
    end: Optional[str] = Field(default=None, description="Historical end date (YYYY-MM-DD); default = today")
    batch_days: int = Field(default=NEWS_BACKFILL_WINDOW_DAYS, ge=1, le=60)
    max_articles_per_batch: int = Field(default=80, ge=1, le=200)


@app.post("/backfill/social")
async def backfill_social(req: SocialBackfillRequest, background: BackgroundTasks):
    """Trigger best-effort social backfill for given symbols and horizon (years).

    Uses SocialMediaCollector under the hood. Returns counts by symbol.
    """
    if not social_collector:
        raise HTTPException(status_code=503, detail="social collector not initialized")
    syms: list[str] = []
    if req.symbols:
        syms = [s.strip().upper() for s in req.symbols if s and s.strip()]
    elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            syms = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            syms = []
    if not syms:
        fb = os.getenv('SOCIAL_FALLBACK_SYMBOLS', 'AAPL,MSFT,TSLA,NVDA,SPY')
        syms = [s.strip().upper() for s in fb.split(',') if s.strip()]
    hours = int(float(req.years) * 365.25 * 24)
    # Run collection in background to avoid blocking the request
    step = int(os.getenv('SOCIAL_BACKFILL_CHUNK', '25') or '25')
    job_id = f"social:{int(time.time())}"
    async def _run():
        out: dict[str, int] = {}
        for i in range(0, len(syms), step):
            batch = syms[i:i+step]
            try:
                res = await social_collector.collect_social_data(batch, hours_back=hours)
                for k, v in (res or {}).items():
                    try:
                        out[k] = int(len(v or []))
                    except Exception:
                        out[k] = 0
            except Exception as e:  # noqa: BLE001
                logger.warning("social backfill batch failed: %s", e)
            await asyncio.sleep(1.0)
        logger.info("Social backfill job complete", job_id=job_id, symbols=len(syms), years=req.years, collected=sum(out.values()))
    try:
        background.add_task(asyncio.create_task, _run())
    except Exception:
        # Fallback: fire-and-forget task without BackgroundTasks if unavailable
        asyncio.create_task(_run())
    return {"status": "accepted", "job_id": job_id, "symbols_scheduled": len(syms), "years": req.years, "batch_size": step}


@app.get("/backfill/social/run")
async def backfill_social_run(years: float = Query(default=1.0, ge=0.1, le=10.0), symbols: Optional[str] = Query(default=None, description="Comma-separated symbols"), background: BackgroundTasks = None):
    syms: list[str] = []
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(',') if s and s.strip()]
    req = SocialBackfillRequest(symbols=syms or None, years=years)
    # Delegate to POST handler
    return await backfill_social(req, background or BackgroundTasks())


@app.post("/backfill/news")
async def backfill_news(req: NewsBackfillJobRequest, background: BackgroundTasks):
    """Schedule a non-blocking, chunked news backfill job.

    Delegates to news_svc.collect_financial_news_range in a background task, which persists/indexes
    according to feature flags. Returns immediately with an accepted job id.
    """
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    # Resolve symbols
    syms: list[str] = []
    if req.symbols:
        syms = [s.strip().upper() for s in req.symbols if s and s.strip()]
    elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            syms = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            syms = []
    if not syms:
        fb = os.getenv('NEWS_FALLBACK_SYMBOLS', 'AAPL,MSFT,TSLA,NVDA,SPY')
        syms = [s.strip().upper() for s in fb.split(',') if s.strip()]
    # Parse dates
    try:
        start_dt = datetime.strptime(req.start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
    if req.end:
        try:
            end_dt = datetime.strptime(req.end, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end format, expected YYYY-MM-DD")
    else:
        end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")

    batch_days = max(1, int(req.batch_days or NEWS_BACKFILL_WINDOW_DAYS))
    max_articles = max(1, int(req.max_articles_per_batch or 80))
    job_id = f"news:{int(time.time())}"

    async def _run():
        try:
            total, batches = await news_svc.collect_financial_news_range(
                syms, start_dt, end_dt,
                batch_days=batch_days,
                max_articles_per_batch=max_articles,
                backfill_mode=True,
            )
            logger.info("News backfill job complete", job_id=job_id, symbols=len(syms), total_articles=int(total))
        except Exception as e:  # noqa: BLE001
            logger.warning("News backfill job failed", job_id=job_id, error=str(e))

    try:
        background.add_task(asyncio.create_task, _run())
    except Exception:
        asyncio.create_task(_run())
    return {
        "status": "accepted",
        "job_id": job_id,
        "symbols_scheduled": len(syms),
        "start": start_dt.strftime('%Y-%m-%d'),
        "end": end_dt.strftime('%Y-%m-%d'),
        "batch_days": batch_days,
        "max_articles_per_batch": max_articles,
    }


@app.get("/backfill/news/run")
async def backfill_news_run(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="YYYY-MM-DD (default=today)"),
    symbols: Optional[str] = Query(default=None, description="Comma-separated symbols; default watchlist"),
    batch_days: int = Query(default=NEWS_BACKFILL_WINDOW_DAYS, ge=1, le=60),
    max_articles_per_batch: int = Query(default=80, ge=1, le=200),
    background: BackgroundTasks = None,
):
    """Schedule a non-blocking, chunked news backfill job (inline scheduler).

    This avoids coupling to the POST handler signature and directly queues the work here.
    """
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")

    # Resolve symbols from query or watchlist fallback
    syms: list[str] = []
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(',') if s and s.strip()]
    elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            syms = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            syms = []
    if not syms:
        fb = os.getenv('NEWS_FALLBACK_SYMBOLS', 'AAPL,MSFT,TSLA,NVDA,SPY')
        syms = [s.strip().upper() for s in fb.split(',') if s.strip()]

    # Parse dates
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
    if end:
        try:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end format, expected YYYY-MM-DD")
    else:
        end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")

    bdays = max(1, int(batch_days or 14))
    max_articles = max(1, int(max_articles_per_batch or 80))
    job_id = f"news:{int(time.time())}"

    async def _run():
        try:
            total, batches = await news_svc.collect_financial_news_range(
                syms, start_dt, end_dt,
                batch_days=bdays,
                max_articles_per_batch=max_articles,
                backfill_mode=True,
            )
            logger.info("News backfill job complete", job_id=job_id, symbols=len(syms), total_articles=int(total))
        except Exception as e:  # noqa: BLE001
            logger.warning("News backfill job failed", job_id=job_id, error=str(e))

    try:
        (background or BackgroundTasks()).add_task(asyncio.create_task, _run())
    except Exception:
        asyncio.create_task(_run())
    return {
        "status": "accepted",
        "job_id": job_id,
        "symbols_scheduled": len(syms),
        "start": start_dt.strftime('%Y-%m-%d'),
        "end": end_dt.strftime('%Y-%m-%d'),
        "batch_days": bdays,
        "max_articles_per_batch": max_articles,
    }


@app.post("/backfill/run")
async def backfill_run(req: BackfillRequest):
    """Start unified backfill: equities(20y), options(5y), news(5y), social(5y) with safe pacing.

    Runs as background tasks using existing service loops/APIs.
    """
    # Equities
    if market_data_svc and hasattr(market_data_svc, 'get_bulk_daily_historical'):
        try:
            years = int(req.equities_years)
            symbols: list[str] = []
            if reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                symbols = (await reference_svc.get_watchlist_symbols()) or []
            if not symbols:
                symbols = ['AAPL','MSFT','TSLA','NVDA','SPY']
            symbols = symbols[:max(1, int(req.max_symbols))]
            end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            start_dt = end_dt - timedelta(days=int(years * 365.25))
            async def _equities_job():
                for sym in symbols:
                    try:
                        await market_data_svc.get_bulk_daily_historical(sym, start_dt, end_dt)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Equities backfill symbol failed", symbol=sym, error=str(e))
                    await asyncio.sleep(max(0.0, float(req.pacing_seconds)))
            asyncio.create_task(_equities_job())
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed scheduling equities backfill", error=str(e))
    # Options (use coverage loop’s per-underlying approach lightly)
    if market_data_svc and getattr(market_data_svc, 'enable_options_ingest', False):
        try:
            underlyings: list[str] = []
            if reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                underlyings = (await reference_svc.get_watchlist_symbols()) or []
            if not underlyings:
                underlyings = ['AAPL','MSFT','TSLA','NVDA','SPY']
            underlyings = underlyings[:min(len(underlyings), 100)]
            end_d = datetime.utcnow().date()
            start_d = (end_d - timedelta(days=365 * int(req.options_years)))
            async def _options_job():
                for u in underlyings:
                    # Near-term expiries to limit load, historical small window
                    start_expiry = end_d - timedelta(days=7)
                    end_expiry = end_d + timedelta(days=45)
                    try:
                        await market_data_svc.backfill_options_chain(
                            u, datetime.combine(start_expiry, datetime.min.time()),
                            datetime.combine(end_expiry, datetime.min.time()),
                            start_date=datetime.combine(max(start_d, end_d - timedelta(days=30)), datetime.min.time()),
                            end_date=datetime.combine(end_d, datetime.min.time()),
                            max_contracts=500, pacing_seconds=max(0.1, float(req.pacing_seconds))
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Options backfill underlying failed", underlying=u, error=str(e))
                    await asyncio.sleep(max(0.0, float(req.pacing_seconds)))
            asyncio.create_task(_options_job())
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed scheduling options backfill", error=str(e))
    # News (use symbols + date ranges)
    if news_svc and hasattr(news_svc, 'collect_financial_news_range'):
        try:
            years = int(req.news_years)
            end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            start_dt = end_dt - timedelta(days=int(years * 365.25))
            symbols: list[str] = []
            if reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                symbols = (await reference_svc.get_watchlist_symbols()) or []
            if not symbols:
                symbols = ['AAPL','MSFT','TSLA','NVDA','SPY']
            symbols = symbols[:max(1, min(int(req.max_symbols), 200))]
            async def _news_job():
                try:
                    await news_svc.collect_financial_news_range(symbols, start_dt, end_dt, batch_days=14, max_articles_per_batch=80, backfill_mode=True)
                except Exception as e:  # noqa: BLE001
                    logger.warning("News backfill failed", error=str(e))
            asyncio.create_task(_news_job())
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed scheduling news backfill", error=str(e))
    # Social (best-effort)
    if social_collector and hasattr(social_collector, 'backfill'):
        try:
            years = int(req.social_years)
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=int(years * 365.25))
            symbols: list[str] = []
            if reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                symbols = (await reference_svc.get_watchlist_symbols()) or []
            if not symbols:
                symbols = ['AAPL','MSFT','TSLA','NVDA','SPY']
            async def _social_job():
                try:
                    await social_collector.backfill(symbols, start_dt, end_dt)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Social backfill failed", error=str(e))
            asyncio.create_task(_social_job())
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed scheduling social backfill", error=str(e))
    return {"status": "scheduled", "timestamp": datetime.utcnow().isoformat()}


def _dependency_states() -> Dict[str, bool]:
    return {
        "cache": cache_client is not None,
        "redis": redis_client is not None,
        "market_data": market_data_svc is not None,
        "news": news_svc is not None,
        "reference_data": reference_svc is not None,
        "validation": validation_svc is not None,
        "retention": retention_svc is not None,
    }

def _degraded_reasons(states: Dict[str, bool]) -> List[str]:
    return [k for k, v in states.items() if not v and k in ("cache", "redis")]

# ---------------------- Debug endpoints (S3 news probing) ---------------------- #
@app.get("/debug/polygon-s3-keys")
async def debug_polygon_s3_keys(prefix: str | None = Query(None), max_keys: int = Query(50, ge=1, le=1000)):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    try:
        keys = await news_svc.debug_list_polygon_news_keys(prefix=prefix, max_keys=max_keys)
        return {"count": len(keys), "prefix": prefix or "news/", "keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")

@app.get("/debug/polygon-s3-probe")
async def debug_polygon_s3_probe(date: str = Query(..., description="YYYY-MM-DD")):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid date format; expected YYYY-MM-DD")
    result = await news_svc.debug_probe_polygon_news_keys_for_date(dt)
    return result


def _get_vendor_summary_safe():
    """Return vendor summary or (None, reason)."""
    if not market_data_svc:
        return None, 'market_data_service_not_initialized'
    try:
        if hasattr(market_data_svc, 'summary_metrics'):
            return market_data_svc.summary_metrics(), None
        return None, 'summary_metrics_missing'
    except Exception as e:  # noqa: BLE001
        logger.warning('vendor summary failed', error=str(e))
        return None, 'summary_metrics_error'


async def _compute_backfill_status(redis_client) -> Dict[str, object]:
    """Derive a richer backfill status snapshot.

    States:
      disabled              -> Feature flag off
      running               -> Collector enabled & is_running True
      enabled_no_progress   -> Enabled but no progress keys found yet
      idle                  -> Enabled, progress keys exist, not currently running
    """
    status = 'disabled'
    progress_keys = 0
    last_progress_date: str | None = None
    if historical_collector and historical_collector.enabled:
        # Determine running state (non-fatal if attribute missing)
        is_running = bool(getattr(historical_collector, 'is_running', False))
        # Attempt to discover progress keys (best-effort)
        if redis_client:
            try:
                # Try common methods (async). Prefer scan to avoid blocking; fallback to keys.
                pattern = 'hist:progress:daily:*'
                keys = []
                scan = getattr(redis_client, 'scan', None)
                if callable(scan):  # type: ignore[truthy-bool]
                    cursor = '0'
                    # One scan cycle (limit breadth) – production safe
                    cursor, batch = await scan(cursor=0, match=pattern, count=50)  # type: ignore[arg-type]
                    keys.extend(batch or [])
                else:
                    kfn = getattr(redis_client, 'keys', None)
                    if callable(kfn):
                        keys = await kfn(pattern)  # type: ignore[misc]
                progress_keys = len(keys)
                if keys:
                    # Derive last progress date from Redis values (stop after few to bound cost)
                    for k in keys[:5]:  # safety cap
                        try:
                            val = await redis_client.get(k)  # type: ignore[attr-defined]
                            if val:
                                last_progress_date = val.decode() if hasattr(val, 'decode') else str(val)
                        except Exception:  # noqa: BLE001
                            continue
            except Exception:  # noqa: BLE001
                pass
        if is_running:
            status = 'running'
        else:
            status = 'enabled_no_progress' if progress_keys == 0 else 'idle'
    # Update gauge (single active label) – set only the current status
    if HISTORICAL_BACKFILL_STATUS:
        try:
            HISTORICAL_BACKFILL_STATUS.labels(status=status).set(1)
        except Exception:  # noqa: BLE001
            pass
    # Attempt derived progress percent (rough heuristic): if we have a reference symbol list
    # available via reference_svc, compare number of progress keys to number of symbols.
    try:
        progress_percent = None
        if reference_svc and progress_keys > 0:
            get_syms = getattr(reference_svc, 'get_watchlist_symbols', None)
            if callable(get_syms):
                symbols = await get_syms()
                total = len(symbols or [])
                if total > 0:
                    progress_percent = (progress_keys / total) * 100.0
                    set_backfill_progress(progress_percent)
        elif progress_keys == 0:
            set_backfill_progress(0.0)
    except Exception:  # noqa: BLE001
        pass

    return {
        'enabled': bool(historical_collector and historical_collector.enabled),
        'status': status,
        'progress_keys': progress_keys,
        'last_progress_date': last_progress_date,
        'progress_percent': progress_percent
    }

@app.get("/health")
async def health_check():
    """Liveness style health check (does not require all dependencies)."""
    states = _dependency_states()
    now = datetime.utcnow()
    return {
        "status": "healthy",  # Process alive if this handler is reached
        "service": "data-ingestion",
        "timestamp": now.isoformat(),
        "uptime_seconds": (now - _START_TIME).total_seconds(),
        "dependencies": states,
        "historical_backfill": {
            "enabled": bool(historical_collector and historical_collector.enabled),
            "in_progress": bool(historical_collector and getattr(historical_collector, 'is_running', False)),
        }
    }

@app.get("/healthz")
async def healthz():
    """Basic fast health endpoint (kept for backward compatibility)."""
    return {"status": "ok", "service": "data-ingestion", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
async def readiness():
    """Readiness probe - core dependencies must be available (cache + redis)."""
    states = _dependency_states()
    core_ready = states["cache"] and states["redis"]
    degraded = _degraded_reasons(states)
    status_code = 200 if core_ready else 503
    return JSONResponse(status_code=status_code, content={
        "status": "ready" if core_ready else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "core_dependencies": {"cache": states["cache"], "redis": states["redis"]},
        "optional_dependencies": {k: v for k, v in states.items() if k not in ("cache", "redis")},
        "missing_core": degraded,
        "historical_backfill": {
            "enabled": bool(historical_collector and historical_collector.enabled),
            "in_progress": bool(historical_collector and getattr(historical_collector, 'is_running', False)),
        }
    })

@app.get("/health/extended")
async def extended_health():
    """Extended operational snapshot (heavier; not for liveness probes)."""
    states = _dependency_states()
    now = datetime.utcnow()
    vendor_summary, vendor_reason = _get_vendor_summary_safe()
    # Ingestion pipeline metrics (best-effort; does not fail health if unavailable)
    ingestion_pipelines = None
    ingestion_errors_agg = None
    try:
        mgr = get_ingestion_manager()
        metrics_map = await mgr.get_all_metrics() if mgr else {}
        ingestion_pipelines = metrics_map
        # Aggregate error types across pipelines for quick Prometheus alert context
        agg: dict[str, int] = {}
        for m in (metrics_map or {}).values():
            for et, cnt in (m.get('error_types') or {}).items():
                agg[et] = agg.get(et, 0) + int(cnt)
        ingestion_errors_agg = agg
    except Exception as e:  # noqa: BLE001
        ingestion_pipelines = {'error': f'ingestion_metrics_unavailable: {e}'}
    backfill_status = await _maybe_async(_compute_backfill_status)
    return {
        'service': 'data-ingestion',
        'timestamp': now.isoformat(),
        'uptime_seconds': (now - _START_TIME).total_seconds(),
        'dependencies': states,
        'core_ready': states['cache'] and states['redis'],
        'historical_backfill': {
            **backfill_status,
            'reason': 'feature_flag_disabled' if backfill_status.get('status') == 'disabled' else None
        },
        'feature_flags': {
            'historical_backfill': bool(historical_collector and historical_collector.enabled),
            'questdb_persist': bool(market_data_svc and getattr(market_data_svc, 'enable_questdb_persist', False)),
        },
        'vendor_metrics': vendor_summary,
        'vendor_metrics_unavailable_reason': vendor_reason,
        'loops': LOOP_STATUS,
        'ingestion_pipelines': ingestion_pipelines,
        'ingestion_errors_aggregated': ingestion_errors_agg,
    }

# Helper to allow both sync & async functions in extended health composition
async def _maybe_async(func):  # pragma: no cover - minimal utility
    try:
        result = func(redis_client)
        if asyncio.iscoroutine(result):
            return await result
        return result
    except Exception:  # noqa: BLE001
        return {'enabled': False, 'status': 'error', 'progress_keys': 0, 'last_progress_date': None}

@app.get("/streams/status")
async def streams_status():
    """Summarize background loop status with staleness hints.

    Returns per-loop: enabled, last_run, last_error, interval_seconds, stale_seconds.
    """
    out: dict[str, dict] = {}
    now = datetime.utcnow()
    for name, info in LOOP_STATUS.items():
        last_run = info.get("last_run")
        stale = None
        if last_run:
            try:
                dt = datetime.fromisoformat(str(last_run))
                stale = max(0, int((now - dt).total_seconds()))
            except Exception:
                stale = None
        out[name] = {
            "enabled": bool(info.get("enabled")),
            "last_run": last_run,
            "last_error": info.get("last_error"),
            "interval_seconds": info.get("interval_seconds"),
            "stale_seconds": stale,
        }
    return {"timestamp": now.isoformat(), "loops": out}


@app.get("/metrics.json")
async def metrics_json():
    """Lightweight metrics snapshot (internal counters).

    Note: Not Prometheus exposition format yet; quick JSON for operational validation.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    try:
        m = market_data_svc.metrics
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "quotes": m.get('quotes'),
            "historical": m.get('historical'),
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to gather metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics collection failed")


@app.get('/metrics')
async def prometheus_metrics():
    """Prometheus metrics exposition."""
    try:
        data = generate_latest()
        return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Prometheus metrics generation failed: {e}")
        raise HTTPException(status_code=500, detail='Metrics exposition failed')


# Warm-up task to ensure at least one request emitted so canonical series are present early
@app.on_event("startup")
async def _metrics_warmup():  # pragma: no cover - side-effect for observability
    if os.getenv("DISABLE_METRICS_WARMUP", "false").lower() in ("1", "true", "yes"):
        return
    # Avoid circular import / optional dependency issues by importing here
    try:
        import httpx  # type: ignore
    except Exception:  # noqa: BLE001
        logger.warning("httpx not available; skipping metrics warm-up")
        return
    # Use 127.0.0.1 and disable proxy env to avoid misrouting local requests
    url = os.getenv("INGEST_WARMUP_URL", "http://127.0.0.1:8002/healthz")
    try:
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            await client.get(url)
        logger.info("Ingestion metrics warm-up request executed", url=url)
    except Exception as e:  # noqa: BLE001
        logger.warning("Ingestion metrics warm-up failed", error=str(e), url=url)


@app.get("/status")
async def get_status():
    """Get service status and statistics."""
    try:
        # Get health from all services
        service_health = {}
        
        if market_data_svc:
            if hasattr(market_data_svc, 'get_service_health'):
                service_health["market_data"] = await market_data_svc.get_service_health()
            else:
                service_health["market_data"] = {"status": "unknown", "reason": "get_service_health missing"}
        if news_svc:
            if hasattr(news_svc, 'get_service_health'):
                service_health["news"] = await news_svc.get_service_health()
            else:
                service_health["news"] = {"status": "unknown", "reason": "get_service_health missing"}
        if reference_svc:
            if hasattr(reference_svc, 'get_service_health'):
                service_health["reference_data"] = await reference_svc.get_service_health()
            else:
                service_health["reference_data"] = {"status": "unknown", "reason": "get_service_health missing"}
        if validation_svc:
            if hasattr(validation_svc, 'get_service_health'):
                service_health["validation"] = await validation_svc.get_service_health()
            else:
                service_health["validation"] = {"status": "unknown", "reason": "get_service_health missing"}
            
        return {
            "service": "data-ingestion",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "services": service_health,
            "data_sources": {
                "alpaca": bool(os.getenv("ALPACA_API_KEY")),
                "polygon": bool(os.getenv("POLYGON_API_KEY")),
                "news_api": bool(os.getenv("NEWS_API_KEY")),
                "reddit": bool(os.getenv("REDDIT_CLIENT_ID")),
                "finnhub": bool(os.getenv("FINNHUB_API_KEY")),
                "alpha_vantage": bool(os.getenv("ALPHA_VANTAGE_API_KEY"))
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service status")


# ---------------------- Runtime controls (admin) ---------------------- #

class QuoteGateState(BaseModel):
    enabled: Optional[bool] = None  # None => use env default; True/False => override


@app.get("/admin/quote-gate")
async def get_quote_gate():
    """Return current effective state of trading-hours gating for quote stream.

    effective = override if set, else env default. Also returns raw env and override values.
    """
    env_default = os.getenv("QUOTE_STREAM_TRADING_HOURS_ONLY", "true").lower() in ("1","true","yes")
    effective = QUOTE_STREAM_GATING_OVERRIDE if QUOTE_STREAM_GATING_OVERRIDE is not None else env_default
    return {
        "env_default": env_default,
        "override": QUOTE_STREAM_GATING_OVERRIDE,
        "effective": effective,
        "trading_hours_now": is_trading_hours(),
    }


@app.post("/admin/vector/schema/ensure")
async def admin_vector_schema_ensure():
    """On-demand ensure desired Weaviate schema (best-effort)."""
    try:
        try:
            from shared.vector.weaviate_schema import get_weaviate_client, ensure_desired_schema  # type: ignore
        except Exception:
            from ..shared.vector.weaviate_schema import get_weaviate_client, ensure_desired_schema  # type: ignore
        client = get_weaviate_client()
        result = ensure_desired_schema(client)
        return {"status": "ok", "result": result}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)[:300]})


@app.post("/admin/vector/news/reconcile")
async def admin_vector_news_reconcile(limit: int = Query(500, ge=1, le=5000), days: int = Query(3, ge=1, le=90)):
    """On-demand reconciliation: pull recent news from QuestDB and index into Weaviate using fallback path."""
    try:
        sql = (
            "select title, content, source, ts, symbol from news_items "
            f"where ts >= dateadd('d', -{max(1, days)}, now()) "
            "and title is not null and title != '' order by ts desc limit " + str(max(1, limit))
        )
        data = await _qdb_exec(sql, timeout=20.0)
        cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
        items = []
        for r in (data.get('dataset') or [])[:limit]:
            try:
                items.append({
                    'title': str(r[cols['title']]) if 'title' in cols else '',
                    'content': str(r[cols.get('content')]) if 'content' in cols and r[cols.get('content')] is not None else '',
                    'source': str(r[cols.get('source')]) if 'source' in cols and r[cols.get('source')] is not None else 'news_items',
                    'published_at': str(r[cols.get('ts')]) if 'ts' in cols else datetime.utcnow().isoformat(),
                    'symbols': [str(r[cols.get('symbol')]).upper()] if 'symbol' in cols and r[cols.get('symbol')] else []
                })
            except Exception:
                continue
        indexed = 0
        if items:
            try:
                try:
                    from shared.vector.indexing import index_news_fallback  # type: ignore
                except Exception:
                    from ..shared.vector.indexing import index_news_fallback  # type: ignore
                indexed = await index_news_fallback(items)
                if indexed and VECTOR_NEWS_INDEXED is not None:
                    VECTOR_NEWS_INDEXED.labels(path='admin').inc(indexed)
            except Exception:
                indexed = 0
        return {"attempted": len(items), "indexed": indexed}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)[:400]})


@app.post("/admin/quote-gate")
async def set_quote_gate(state: QuoteGateState):
    """Set or clear the runtime override for quote-stream trading-hours gating.

    Pass {"enabled": true} to force gating ON, {"enabled": false} to force OFF,
    or {"enabled": null} (or omit) to clear override and use env default.
    """
    global QUOTE_STREAM_GATING_OVERRIDE
    if state.enabled is None:
        QUOTE_STREAM_GATING_OVERRIDE = None
    else:
        QUOTE_STREAM_GATING_OVERRIDE = bool(state.enabled)
    # Return current effective state
    env_default = os.getenv("QUOTE_STREAM_TRADING_HOURS_ONLY", "true").lower() in ("1","true","yes")
    effective = QUOTE_STREAM_GATING_OVERRIDE if QUOTE_STREAM_GATING_OVERRIDE is not None else env_default
    return {
        "override": QUOTE_STREAM_GATING_OVERRIDE,
        "effective": effective,
        "trading_hours_now": is_trading_hours(),
    }


@app.post("/admin/retention/run")
async def run_retention_now():
    """Trigger retention cleanup immediately (production admin).

    Runs DataRetentionService.run_data_cleanup() once and returns a brief summary.
    """
    try:
        svc = await get_retention_service()
        started = datetime.utcnow()
        await svc.run_data_cleanup()
        stats = await svc.get_retention_statistics()
        return {
            "status": "completed",
            "started_at": started.isoformat(),
            "ended_at": datetime.utcnow().isoformat(),
            "stats": stats.get("statistics"),
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Retention run failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class QuoteSymbolsState(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="Override symbols for quote stream; null or empty to clear override")
    max_symbols: Optional[int] = Field(default=None, description="Optional max symbols to keep from provided list")


@app.get("/admin/quote-symbols")
async def get_quote_symbols_override():
    """Return current symbol selection for quote stream.

    Includes runtime override (if set), env-provided list, and a brief summary of watchlist availability.
    """
    env_syms_raw = os.getenv("QUOTE_STREAM_SYMBOLS", "").strip()
    env_syms = [s.strip().upper() for s in env_syms_raw.split(',') if s.strip()]
    override_syms = QUOTE_STREAM_SYMBOLS_OVERRIDE[:] if QUOTE_STREAM_SYMBOLS_OVERRIDE else None
    watchlist_summary = None
    try:
        if reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
            syms = await reference_svc.get_watchlist_symbols()
            watchlist_summary = {"available": True, "count": len(syms or [])}
        else:
            watchlist_summary = {"available": False}
    except Exception:
        watchlist_summary = {"available": False}
    return {
        "override": override_syms,
        "env_symbols": env_syms,
        "watchlist": watchlist_summary,
    }


@app.post("/admin/quote-symbols")
async def set_quote_symbols_override(state: QuoteSymbolsState):
    """Set or clear runtime override list for quote stream symbols.

    Pass {"symbols": ["AAPL","MSFT"], "max_symbols": 50} to set; pass {"symbols": null} or an empty list to clear.
    """
    global QUOTE_STREAM_SYMBOLS_OVERRIDE
    syms = state.symbols
    if not syms:
        QUOTE_STREAM_SYMBOLS_OVERRIDE = None
    else:
        # Normalize and cap to a reasonable size to avoid hot loops
        normalized = []
        for s in syms:
            if not s:
                continue
            s2 = str(s).strip().upper()
            if s2:
                normalized.append(s2)
        # De-duplicate preserving order
        seen = set()
        deduped: List[str] = []
        for s in normalized:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        limit = state.max_symbols if isinstance(state.max_symbols, int) and state.max_symbols and state.max_symbols > 0 else None
        if limit is not None:
            deduped = deduped[:limit]
        # Hard upper bound safety valve
        QUOTE_STREAM_SYMBOLS_OVERRIDE = deduped[:1000]
    return await get_quote_symbols_override()


@app.get("/streams/status")
async def streams_status():
    """Operational snapshot focused on streaming loops and gating/symbols.

    Exposes effective trading-hours gating, symbol selection inputs, and staleness of loops.
    """
    # Effective gate
    env_default = os.getenv("QUOTE_STREAM_TRADING_HOURS_ONLY", "true").lower() in ("1","true","yes")
    effective_gate = QUOTE_STREAM_GATING_OVERRIDE if QUOTE_STREAM_GATING_OVERRIDE is not None else env_default
    # Symbols inputs
    env_syms_raw = os.getenv("QUOTE_STREAM_SYMBOLS", "").strip()
    env_syms = [s.strip().upper() for s in env_syms_raw.split(',') if s.strip()]
    override_syms = QUOTE_STREAM_SYMBOLS_OVERRIDE[:] if QUOTE_STREAM_SYMBOLS_OVERRIDE else None
    # Loop staleness based on LOOP_STATUS last_run (best-effort)
    now = datetime.utcnow()
    loops = {}
    for name, info in LOOP_STATUS.items():
        last_run_iso = info.get('last_run') if isinstance(info, dict) else None
        stale_seconds = None
        if isinstance(last_run_iso, str):
            try:
                last_dt = datetime.fromisoformat(last_run_iso)
                stale_seconds = (now - last_dt).total_seconds()
            except Exception:
                stale_seconds = None
        loops[name] = {
            'enabled': bool(info.get('enabled')) if isinstance(info, dict) else False,
            'interval_seconds': info.get('interval_seconds') if isinstance(info, dict) else None,
            'last_run': last_run_iso,
            'last_error': info.get('last_error') if isinstance(info, dict) else None,
            'stale_seconds': stale_seconds,
        }
    # Sampling knobs
    try:
        sample_size = int(os.getenv("QUOTE_STREAM_SAMPLE_SIZE", "2"))
    except Exception:
        sample_size = 2
    try:
        max_syms = int(os.getenv("QUOTE_STREAM_MAX_SYMBOLS", "200"))
    except Exception:
        max_syms = 200
    return {
        'service': 'data-ingestion',
        'timestamp': now.isoformat(),
        'quote_stream': {
            'gating': {
                'env_default': env_default,
                'override': QUOTE_STREAM_GATING_OVERRIDE,
                'effective': effective_gate,
                'trading_hours_now': is_trading_hours(),
            },
            'symbols': {
                'override': override_syms,
                'env_symbols': env_syms,
            },
            'sampling': {
                'sample_size': sample_size,
                'max_symbols': max_syms,
            }
        },
        'loops': loops,
    }

@app.get("/debug/polygon-s3-keys")
async def debug_polygon_s3_keys(prefix: str | None = Query(None), max_keys: int = Query(50, ge=1, le=1000)):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    try:
        keys = await news_svc.debug_list_polygon_news_keys(prefix=prefix, max_keys=max_keys)
        return {"count": len(keys), "prefix": prefix or "news/", "keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")

@app.get("/debug/polygon-s3-probe")
async def debug_polygon_s3_probe(date: str = Query(..., description="YYYY-MM-DD")):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    try:
        day = datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid date format; expected YYYY-MM-DD")
    result = await news_svc.debug_probe_polygon_news_keys_for_date(day)
    return result

@app.get("/debug/options-precheck")
async def debug_options_precheck():
    svc = market_data_svc
    if not svc:
        raise HTTPException(status_code=503, detail="market_data_service unavailable")
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "enable_options_ingest": bool(getattr(svc, 'enable_options_ingest', False)),
        "enable_questdb_persist": bool(getattr(svc, 'enable_questdb_persist', False)),
        "enable_hist_dry_run": bool(getattr(svc, 'enable_hist_dry_run', False)),
        "questdb_conf": getattr(svc, '_qdb_conf', None),
        "sender_available": bool(getattr(svc, '_qdb_Sender', None) and getattr(svc, '_qdb_conf', None)),
        "polygon_key_present": bool(svc.polygon_config.get('api_key') if svc else False),
    }

@app.get("/debug/news-precheck")
async def debug_news_precheck():
    svc = news_svc
    if not svc:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "available": False,
            "reason": "news_service_not_running"
        }
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "available": True,
        "enable_news_questdb_persist": bool(getattr(svc, 'enable_questdb_persist', False)),
        "questdb_conf": getattr(svc, '_qdb_conf', None),
        "ilp_sender_ready": bool(getattr(svc, '_qdb_sender', None) and getattr(svc, '_qdb_conf', None)),
        "news_api_key": bool(svc.news_api_config.get('api_key')),
        "alpha_vantage_key": bool(svc.alpha_vantage_config.get('api_key')),
        "finnhub_key": bool(svc.finnhub_config.get('api_key')),
    }


@app.post("/ingest/market-data")
async def ingest_market_data(symbol: str):
    """Trigger market data ingestion for a symbol."""
    try:
        logger.info(f"Starting market data ingestion for {symbol}")
        
        # For Phase 2, this is a placeholder that would:
        # 1. Connect to Alpaca/Polygon API
        # 2. Fetch real-time data for symbol
        # 3. Store in QuestDB via cache layer
        # 4. Publish to message queue for downstream processing
        
        sample_data = MarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            open=100.0,
            high=102.5,
            low=99.8,
            close=101.2,
            volume=1000000,
            timeframe="1min",
            data_source="sample"
        )
        
        # Store in cache (would be real implementation)
        if cache_client:
            await cache_client.set_market_data(sample_data)
            
        logger.info(f"Successfully ingested market data for {symbol}")
        return {
            "status": "success",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Market data ingestion completed"
        }
        
    except Exception as e:
        logger.error(f"Market data ingestion failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/news")
async def ingest_news(query: str = "SPY TSLA AAPL"):
    """Trigger news ingestion for given query."""
    try:
        logger.info(f"Starting news ingestion for query: {query}")
        
        # Placeholder for Phase 2 - would implement:
        # 1. Connect to NewsAPI/other news sources
        # 2. Fetch relevant financial news
        # 3. Run sentiment analysis
        # 4. Store processed news with sentiment scores
        
        sample_news = NewsItem(
            title="Market Update: Tech Stocks Rally",
            content="Technology stocks showed strong performance...",
            source="Financial Times",
            published_at=datetime.utcnow(),
            url="https://example.com/news/123",
            sentiment_score=0.7,
            relevance_score=0.9,
            symbols=["SPY", "TSLA", "AAPL"]
        )
        
        if cache_client:
            await cache_client.set_news_item(sample_news)
            
        logger.info(f"Successfully ingested news for query: {query}")
        return {
            "status": "success", 
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "News ingestion completed"
        }
        
    except Exception as e:
        logger.error(f"News ingestion failed for query {query}: {e}")
        raise HTTPException(status_code=500, detail=f"News ingestion failed: {str(e)}")


@app.get("/data/recent/{symbol}")
async def get_recent_data(symbol: str, hours: int = 1):
    """Get recent market data for a symbol."""
    try:
        if not cache_client:
            raise HTTPException(status_code=503, detail="Cache service unavailable")
            
        # Placeholder - would query actual cached data
        logger.info(f"Retrieving recent data for {symbol} (last {hours} hours)")
        
        return {
            "symbol": symbol,
            "timeframe": f"last_{hours}_hours",
            "data_points": 0,  # Would return actual data
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"No data available yet - service in development mode"
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Data retrieval failed: {str(e)}")


# New comprehensive endpoints

@app.post("/market-data/quote/{symbol}")
async def get_real_time_quote(symbol: str):
    """Get real-time quote for a symbol."""
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    
    try:
        quote = await market_data_svc.get_real_time_quote(symbol.upper())
        if quote:
            # Validate the data
            if validation_svc:
                validation_results = await validation_svc.validate_market_data(quote)
                has_errors = any(r.severity.value == "error" for r in validation_results)
                
                return {
                    "data": {
                        "symbol": quote.symbol,
                        "timestamp": quote.timestamp.isoformat(),
                        "open": quote.open,
                        "high": quote.high,
                        "low": quote.low,
                        "close": quote.close,
                        "volume": quote.volume,
                        "source": quote.data_source
                    },
                    "validation": {
                        "valid": not has_errors,
                        "issues": len(validation_results),
                        "details": [{"severity": r.severity.value, "message": r.message} for r in validation_results]
                    }
                }
            else:
                return {"data": quote, "validation": {"valid": True, "issues": 0}}
        else:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Artifact Upload to MinIO ---------------------- #

def _load_minio_client():
    """Lazy import and construct MinIO client using shared helper.

    Returns (client, cfg) or (None, None) if unavailable.
    """
    try:
        from shared.storage.minio_storage import get_minio_client, MinIOConfig  # type: ignore
        # Support base64-encoded secret to avoid issues with $ in docker-compose interpolation
        secret_b64 = os.getenv("MINIO_SECRET_KEY_B64", "").strip()
        if secret_b64:
            import base64
            try:
                decoded = base64.b64decode(secret_b64).decode("utf-8")
                secret = decoded
            except Exception:
                secret = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", ""))
        else:
            secret = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", ""))
        cfg = MinIOConfig(
            endpoint=os.getenv("MINIO_ENDPOINT", os.getenv("MINIO_URL", "localhost:9000")),
            access_key=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "")),
            secret_key=secret,
            secure=os.getenv("MINIO_SECURE", "false").lower() in ("1","true","yes"),
            region=os.getenv("MINIO_REGION"),
        )
        client = get_minio_client(cfg)
        return client, cfg
    except Exception:
        return None, None


@app.post("/artifacts/upload")
async def upload_artifacts_to_minio(
    directory: str = Query(default=os.getenv("GRAFANA_EXPORT_DIR", "/app/export/grafana-csv"), description="Directory containing JSON artifacts"),
    bucket: str = Query(default=os.getenv("MINIO_ARTIFACTS_BUCKET", "trading"), description="MinIO bucket name"),
    prefix: str = Query(default=os.getenv("MINIO_ARTIFACTS_PREFIX", "dashboards"), description="Key prefix inside bucket"),
    pattern: str = Query(default="*.json", description="Glob pattern of files to upload"),
):
    """Upload local JSON artifacts (e.g., coverage reports) to MinIO.

    Walks the given directory and uploads matching files to s3://{bucket}/{prefix}/filename.
    Returns a summary with successes and failures.
    """
    client, cfg = _load_minio_client()
    if not client:
        raise HTTPException(status_code=503, detail="MinIO client unavailable or credentials not configured")
    try:
        from shared.storage.minio_storage import ensure_bucket  # type: ignore
    except Exception:
        ensure_bucket = None  # type: ignore

    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    import glob, json
    files = list(dir_path.glob(pattern))
    uploaded: list[dict] = []
    errors: list[dict] = []

    # Ensure bucket exists (best-effort)
    if ensure_bucket:
        try:
            ensure_bucket(bucket, client=client)  # type: ignore
        except Exception as e:  # noqa: BLE001
            # Continue; upload attempts may still create lazily
            logger.warning("ensure_bucket failed", bucket=bucket, error=str(e))

    for f in files:
        try:
            data = f.read_bytes()
            key = f"{prefix.strip('/')}" if prefix else ""
            if key:
                key += "/" + f.name
            else:
                key = f.name
            # Use client directly to avoid extra deps
            from io import BytesIO
            stream = BytesIO(data)
            client.put_object(bucket, key, stream, length=len(data), content_type="application/json")  # type: ignore[attr-defined]
            uploaded.append({"file": str(f), "bucket": bucket, "key": key, "bytes": len(data)})
        except Exception as e:  # noqa: BLE001
            errors.append({"file": str(f), "error": str(e)})

    return {
        "status": "completed",
        "directory": str(dir_path),
        "bucket": bucket,
        "prefix": prefix,
        "uploaded": uploaded,
        "uploaded_count": len(uploaded),
        "errors": errors,
        "error_count": len(errors),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/market-data/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    timeframe: str = "1min",
    hours_back: int = 24,
    limit: int = 1000,
    start: Optional[str] = Query(None, description="Optional start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="Optional end date YYYY-MM-DD"),
):
    """Get historical market data.

    Modes:
      1. Intraday/legacy (default): timeframe != '1d' OR no start supplied.
         Uses hours_back window ending now and underlying provider specific intraday retrieval.
      2. Bulk daily (extended history): timeframe == '1d' AND start supplied (YYYY-MM-DD).
         Optional end (YYYY-MM-DD, default = today). Delegates to bulk daily provider chain
         (EODHD primary) for 20y scale retrieval.

    Backward compatibility: existing callers that only pass timeframe/hours_back continue working.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")

    try:
        symbol_u = symbol.upper()

        # ---------------- Bulk Daily Path (EODHD) ----------------
        if timeframe == '1d' and start:
            try:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
            if end:
                try:
                    end_dt = datetime.strptime(end, "%Y-%m-%d")
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid end format, expected YYYY-MM-DD")
            else:
                # End date defaults to today (UTC date boundary)
                end_dt = datetime.utcnow()

            if start_dt > end_dt:
                raise HTTPException(status_code=400, detail="start must be <= end")

            # Hard guard: limit to 25 years (same as backfill endpoint)
            if (end_dt - start_dt).days > 365 * 25:
                raise HTTPException(status_code=400, detail="Range too large; limit to 25 years")

            rows = await market_data_svc.get_bulk_daily_historical(symbol_u, start_dt, end_dt)

            return {
                "symbol": symbol_u,
                "timeframe": timeframe,
                "mode": "bulk_daily",
                "count": len(rows),
                "start": start_dt.strftime("%Y-%m-%d"),
                "end": end_dt.strftime("%Y-%m-%d"),
                # Best-effort primary source indicator without touching internal attributes
                "data_source_primary": ("EODHD" if os.getenv("EODHD_API_KEY") else None),
                "data": [
                    {
                        "timestamp": bar.timestamp.isoformat(),
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "source": "eodhd",
                    } for bar in rows
                ],
            }

        # ---------------- Intraday / Legacy Path ----------------
        # Fallback: intraday path not implemented in MarketDataService; map to daily bars over window
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        # Normalize to date boundaries
        start_dt = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
        rows = await market_data_svc.get_bulk_daily_historical(symbol_u, start_dt, end_dt)

        return {
            "symbol": symbol_u,
            "timeframe": timeframe,
            "mode": "intraday_window_mapped_to_daily",
            "count": len(rows),
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "data": [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "source": "eodhd",
                } for bar in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/news/collect")
async def collect_financial_news(
    symbols: Optional[List[str]] = Query(default=None),
    hours_back: int = 1,
    max_articles: int = 50,
    body: Optional[dict] = Body(default=None, description="Optional JSON body: { symbols: string[], hours_back?: number, max_articles?: number }")
):
    """Collect financial news."""
    if not news_svc:
        raise HTTPException(status_code=503, detail="News service not available")
    
    try:
        # Accept either query params or JSON body for convenience
        eff_symbols = symbols
        eff_hours_back = hours_back
        eff_max_articles = max_articles
        try:
            if body and isinstance(body, dict):
                if eff_symbols is None and isinstance(body.get('symbols'), list):
                    eff_symbols = [str(s).upper() for s in body.get('symbols') if isinstance(s, (str,)) and s.strip()]
                if 'hours_back' in body and isinstance(body.get('hours_back'), (int, float)):
                    eff_hours_back = int(body.get('hours_back'))
                if 'max_articles' in body and isinstance(body.get('max_articles'), (int, float)):
                    eff_max_articles = int(body.get('max_articles'))
        except Exception:
            # Ignore malformed body and fall back to query params
            pass

        news_items = await news_svc.collect_financial_news(eff_symbols, eff_hours_back, eff_max_articles)
        
        return {
            "status": "success",
            "symbols": eff_symbols,
            "articles_collected": len(news_items),
            "hours_back": eff_hours_back,
            "timestamp": datetime.utcnow().isoformat(),
            "articles": [
                {
                    "title": item.title,
                    "source": item.source,
                    "published_at": item.published_at.isoformat(),
                    "sentiment_score": item.sentiment_score,
                    "relevance_score": item.relevance_score,
                    "symbols": item.symbols,
                    "url": item.url
                } for item in news_items
            ]
        }
    except Exception as e:
        logger.error(f"Failed to collect news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Historical/Backfill Endpoints ---------------------- #

class NewsBackfillRequest(BaseModel):
    symbols: List[str]
    start: str
    end: str
    batch_days: int | None = Field(default=NEWS_BACKFILL_WINDOW_DAYS)
    max_articles_per_batch: int | None = Field(default=80)


@app.post("/news/backfill")
async def news_backfill(req: NewsBackfillRequest):
    """Backfill news over a historical date range (batched provider calls).

    Delegates to NewsService.collect_financial_news_range which persists/indexes
    according to feature flags (QuestDB, Weaviate, Pulsar).
    """
    if not news_svc:
        raise HTTPException(status_code=503, detail="News service not available")
    try:
        try:
            start_dt = datetime.strptime(req.start, "%Y-%m-%d")
            end_dt = datetime.strptime(req.end, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")
        if start_dt > end_dt:
            raise HTTPException(status_code=400, detail="start must be <= end")
        batch_days = max(1, int(req.batch_days or NEWS_BACKFILL_WINDOW_DAYS))
        max_articles = max(1, int(req.max_articles_per_batch or 80))
        total, batches = await news_svc.collect_financial_news_range(
            [s.strip().upper() for s in (req.symbols or []) if s and s.strip()],
            start_dt,
            end_dt,
            batch_days=batch_days,
            max_articles_per_batch=max_articles,
            backfill_mode=True,
        )
        return {
            "status": "completed",
            "symbols": [s.strip().upper() for s in (req.symbols or []) if s and s.strip()],
            "start": req.start,
            "end": req.end,
            "batch_days": batch_days,
            "max_articles_per_batch": max_articles,
            "articles_collected": int(total),
            "persisted_count": int(sum(int(b.get('persisted', 0) or 0) for b in batches)),
            "batches": batches,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("News backfill failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ----- Social backfill alias to match orchestrator path (/social/backfill) ----- #
class SocialBackfillWindowRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None)
    start: str
    end: str
    batch_hours: int = Field(default=6, ge=1, le=24)


@app.post("/social/backfill")
async def social_backfill_compat(
    req: Optional[SocialBackfillWindowRequest] = Body(default=None),
    symbols: Optional[str] = Query(default=None, description="Comma-separated symbols (optional)"),
    start: Optional[str] = Query(default=None, description="Start date YYYY-MM-DD (optional)"),
    end: Optional[str] = Query(default=None, description="End date YYYY-MM-DD (optional)"),
    years: Optional[float] = Query(default=None, description="Years lookback (optional alternative to start/end)"),
    batch_hours: int = Query(default=6, ge=1, le=24),
    background: BackgroundTasks = None,
):
    """Compatibility endpoint for orchestrator expecting /social/backfill.

    Accepts either a windowed payload {symbols, start, end, batch_hours} or a simple
    SocialBackfillRequest {symbols, years}. Schedules work in the background and returns 202-style ack.
    """
    if not social_collector:
        raise HTTPException(status_code=503, detail="social collector not initialized")
    # Resolve symbols via request -> reference -> fallback
    syms: list[str] = []
    try:
        body_syms = (req.symbols if (req and hasattr(req, 'symbols')) else None)  # type: ignore[attr-defined]
        if body_syms:
            syms = [s.strip().upper() for s in body_syms if s and s.strip()]
        elif symbols:
            syms = [s.strip().upper() for s in symbols.split(',') if s and s.strip()]
    except Exception:
        syms = []
    if not syms and reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            syms = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            syms = []
    if not syms:
        fb = os.getenv('SOCIAL_FALLBACK_SYMBOLS', 'AAPL,MSFT,TSLA,NVDA,SPY')
        syms = [s.strip().upper() for s in fb.split(',') if s.strip()]

    # Compute hours_back from either start/end or years
    job_id = f"social:{int(time.time())}"
    # Determine mode: windowed if start/end provided (either in body or query); else years-based
    body_start = getattr(req, 'start', None) if req else None
    body_end = getattr(req, 'end', None) if req else None
    use_window = bool((body_start and body_end) or (start and end))
    if use_window:
        # Parse dates and compute total hours; chunk by batch_hours to keep API load reasonable
        s_val = (body_start or start)
        e_val = (body_end or end)
        try:
            start_dt = datetime.strptime(str(s_val), "%Y-%m-%d")
            end_dt = datetime.strptime(str(e_val), "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD for start/end")
        if start_dt > end_dt:
            raise HTTPException(status_code=400, detail="start must be <= end")
        total_hours = max(1, int((end_dt - start_dt).total_seconds() // 3600) + 24)
        bh = max(1, int(batch_hours))

        async def _run():
            try:
                # Walk in windows of batch_hours from end backwards to start
                remaining = total_hours
                collected_total = 0
                while remaining > 0:
                    hours_back = min(bh, remaining)
                    try:
                        res = await social_collector.collect_social_data(syms, hours_back=hours_back)
                        if isinstance(res, dict):
                            collected_total += sum(len(v or []) for v in res.values())
                    except Exception:
                        pass
                    remaining -= hours_back
                    await asyncio.sleep(1.0)
                logger.info("Social backfill window job complete", job_id=job_id, symbols=len(syms), hours=total_hours, batches=max(1, total_hours // bh))
            except Exception as e:  # noqa: BLE001
                logger.warning("Social backfill window failed", job_id=job_id, error=str(e))

        try:
            (background or BackgroundTasks()).add_task(asyncio.create_task, _run())
        except Exception:
            asyncio.create_task(_run())
        return {"status": "accepted", "job_id": job_id, "symbols_scheduled": len(syms), "hours_total": total_hours, "batch_hours": bh}

    # Years-based: use years from body if provided else query param
    yrs = None
    try:
        yrs = float(getattr(req, 'years', None)) if req and hasattr(req, 'years') else None  # type: ignore[attr-defined]
    except Exception:
        yrs = None
    if yrs is None:
        yrs = float(years) if years is not None else 1.0
    return await backfill_social(SocialBackfillRequest(symbols=syms, years=yrs), background or BackgroundTasks())


@app.post("/market-data/options-contract")
async def backfill_single_option_contract(
    option_ticker: str = Query(..., description="Polygon option ticker e.g., O:SPY251219C00600000"),
    start: str = Query(..., description="Historical start date (YYYY-MM-DD) for aggregates window"),
    end: Optional[str] = Query(default=None, description="Historical end date (YYYY-MM-DD) inclusive for aggregates window"),
):
    """Validate single-contract ingest path by fetching aggregates for a specific option ticker and persisting.

    Useful for diagnosing ILP persistence and coverage movement without full-chain load.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="market_data_service unavailable")
    if not getattr(market_data_svc, 'enable_options_ingest', False):
        raise HTTPException(status_code=403, detail="Options ingestion disabled")
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.utcnow()
    try:
        rows = await market_data_svc._polygon_fetch_option_aggs(option_ticker, start_dt, end_dt)
        meta = market_data_svc._parse_option_symbol(option_ticker)
        # Attach metadata minimally for persistence
        for r in rows:
            r.update({
                'underlying': option_ticker.split(':',1)[1][:option_ticker.split(':',1)[1].find(meta.get('right','') or 'C')].rstrip('0123456789') if ':' in option_ticker else '',
                'option_symbol': option_ticker,
                'right': meta.get('right'),
                'strike': meta.get('strike'),
                'expiry': meta.get('expiry')
            })
        await market_data_svc._persist_option_bars_qdb(rows)
        try:
            if OPTIONS_BACKFILL_CONTRACTS:
                OPTIONS_BACKFILL_CONTRACTS.labels(path='single-contract').inc(1)
            if OPTIONS_BACKFILL_BARS:
                OPTIONS_BACKFILL_BARS.labels(path='single-contract').inc(len(rows))
        except Exception:
            pass
        return {"option_ticker": option_ticker, "rows": len(rows), "start": start_dt.strftime('%Y-%m-%d'), "end": end_dt.strftime('%Y-%m-%d')}
    except Exception as e:  # noqa: BLE001
        try:
            if OPTIONS_BACKFILL_ERRORS:
                OPTIONS_BACKFILL_ERRORS.labels(path='single-contract').inc()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/market-data/options-chain/{underlying}")
async def backfill_options_chain_endpoint(
    underlying: str,
    start: str = Query(..., description="Historical bars start YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="Historical bars end YYYY-MM-DD (default: today)"),
    start_expiry: Optional[str] = Query(default=None, description="Expiry start YYYY-MM-DD (optional)"),
    end_expiry: Optional[str] = Query(default=None, description="Expiry end YYYY-MM-DD (optional)"),
    max_contracts: int = Query(default=OPTIONS_HISTORY_MAX_CONTRACTS, ge=1, le=10000),
    pacing_seconds: float = Query(default=OPTIONS_HISTORY_PACING_SECONDS, ge=0.0, le=5.0),
):
    """Trigger options-chain backfill for a single underlying.

    Aligns with orchestrator expectations at /market-data/options-chain/{symbol} and delegates to
    market_data_svc.backfill_options_chain with a bounded expiry window.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    if not getattr(market_data_svc, 'enable_options_ingest', False):
        raise HTTPException(status_code=403, detail="Options ingestion disabled")
    # Parse dates
    try:
        start_hist = datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
    end_hist = datetime.utcnow() if not end else datetime.strptime(end, "%Y-%m-%d")
    # Expiry window defaults near-term if not provided
    se = _parse_date_yyyy_mm_dd(start_expiry) if start_expiry else None
    ee = _parse_date_yyyy_mm_dd(end_expiry) if end_expiry else None
    if se is None or ee is None:
        today = datetime.utcnow().date()
        se = datetime.combine(today - timedelta(days=7), datetime.min.time()) if se is None else se
        ee = datetime.combine(today + timedelta(days=45), datetime.min.time()) if ee is None else ee
    # Delegate
    try:
        summary = await market_data_svc.backfill_options_chain(
            underlying.strip().upper(),
            se,
            ee,
            start_date=start_hist,
            end_date=end_hist,
            max_contracts=max_contracts,
            pacing_seconds=pacing_seconds,
        )
        return {"status": "completed", **summary, "start": start_hist.date().isoformat(), "end": end_hist.date().isoformat(), "start_expiry": se.date().isoformat(), "end_expiry": ee.date().isoformat()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backfill/options")
async def backfill_options(
    underlyings: Optional[List[str]] = Query(default=None, description="Underlyings to process; defaults to watchlist sample"),
    start_expiry: Optional[str] = Query(default=None, description="Expiry start YYYY-MM-DD (optional)"),
    end_expiry: Optional[str] = Query(default=None, description="Expiry end YYYY-MM-DD (optional)"),
    start: str = Query(..., description="Historical bars start YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="Historical bars end YYYY-MM-DD (default: today)"),
    max_contracts: int = Query(default=OPTIONS_HISTORY_MAX_CONTRACTS, ge=1, le=10000),
    pacing_seconds: float = Query(default=OPTIONS_HISTORY_PACING_SECONDS, ge=0.0, le=5.0),
    limit_underlyings: int = Query(default=100, ge=1, le=2000),
    dry_run: bool = Query(default=False),
    persist_questdb: Optional[bool] = Query(default=None, description="Override QuestDB persistence flag for this run"),
    enable_ingest: Optional[bool] = Query(default=None, description="Override ENABLE_OPTIONS_INGEST for this run"),
):
    """Orchestrate historical options backfill across a set of underlyings.

    Delegates to market_data_svc.backfill_options_chain per underlying with pacing.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    if not getattr(market_data_svc, 'enable_options_ingest', False):
        raise HTTPException(status_code=403, detail="Options ingestion disabled")

    # Resolve underlyings
    eff_und: List[str] = []
    if underlyings:
        eff_und = [u.strip().upper() for u in underlyings if u and u.strip()]
    elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            eff_und = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            eff_und = []
    if not eff_und:
        eff_und = ['AAPL','MSFT','TSLA','NVDA','SPY']
    eff_und = eff_und[:max(1, limit_underlyings)]

    # Parse dates
    try:
        start_hist = datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
    end_hist = datetime.utcnow() if not end else datetime.strptime(end, "%Y-%m-%d")
    start_exp_dt = _parse_date_yyyy_mm_dd(start_expiry) if start_expiry else None
    end_exp_dt = _parse_date_yyyy_mm_dd(end_expiry) if end_expiry else None
    # Default expiry window to near-term if not provided
    if start_exp_dt is None or end_exp_dt is None:
        today = datetime.utcnow().date()
        start_exp_dt = datetime.combine(today - timedelta(days=7), datetime.min.time()) if start_exp_dt is None else start_exp_dt
        end_exp_dt = datetime.combine(today + timedelta(days=45), datetime.min.time()) if end_exp_dt is None else end_exp_dt

    if dry_run:
        return {
            "status": "planned",
            "underlyings": eff_und,
            "start": start_hist.date().isoformat(),
            "end": end_hist.date().isoformat(),
            "start_expiry": start_exp_dt.date().isoformat() if start_exp_dt else None,
            "end_expiry": end_exp_dt.date().isoformat() if end_exp_dt else None,
            "max_contracts": max_contracts,
            "pacing_seconds": pacing_seconds,
        }

    totals = {"underlyings": len(eff_und), "contracts_processed": 0, "bars_ingested": 0, "errors": 0, "per_underlying": []}
    started = datetime.utcnow()
    # Temporary flag overrides
    orig_persist = getattr(market_data_svc, 'enable_questdb_persist', None)
    orig_ingest = getattr(market_data_svc, 'enable_options_ingest', None)
    try:
      if persist_questdb is not None:
          try:
              market_data_svc.enable_questdb_persist = bool(persist_questdb)
          except Exception:
              pass
      if enable_ingest is not None:
          try:
              market_data_svc.enable_options_ingest = bool(enable_ingest)
          except Exception:
              pass
      for u in eff_und:
        try:
            summary = await market_data_svc.backfill_options_chain(
                u,
                start_exp_dt,
                end_exp_dt,
                start_date=start_hist,
                end_date=end_hist,
                max_contracts=max_contracts,
                pacing_seconds=pacing_seconds,
            )
            # MarketDataService returns {'underlying','contracts','bars','enabled'}
            totals["contracts_processed"] += int(summary.get('contracts', 0) or 0)
            totals["bars_ingested"] += int(summary.get('bars', 0) or 0)
            totals["per_underlying"].append({"underlying": u, **summary})
        except Exception as e:  # noqa: BLE001
            totals["errors"] += 1
            totals["per_underlying"].append({"underlying": u, "error": str(e)})
        await asyncio.sleep(max(0.0, float(pacing_seconds)))
    finally:
      try:
          if orig_persist is not None:
              market_data_svc.enable_questdb_persist = orig_persist
      except Exception:
          pass
      try:
          if orig_ingest is not None:
              market_data_svc.enable_options_ingest = orig_ingest
      except Exception:
          pass

    return {
        "status": "completed",
        "started_at": started.isoformat(),
        "ended_at": datetime.utcnow().isoformat(),
        **totals,
        "start": start_hist.date().isoformat(),
        "end": end_hist.date().isoformat(),
    }


@app.get("/coverage/equities_full")
async def equities_coverage_full(limit: int = 500, sample: bool = True, include_adjusted: bool = True, mature_years: float = 7.0):
    """Compute equities historical coverage from QuestDB and optionally update gauges.

    - Reads earliest/latest TIMESTAMP per symbol from table (env EQUITY_DAILY_TABLE, default 'market_data').
    - Returns list entries: {symbol, first_date, last_date, years_span, meets_20y, listing_age_years?, ipo_adjusted_meets?}.
    - When sample is False, updates Prometheus gauges for ratios and totals.
    """
    table_name = os.getenv('EQUITY_DAILY_TABLE', 'market_data')
    host = os.getenv('QUESTDB_HOST', 'trading-questdb')
    http_port = os.getenv('QUESTDB_HTTP_PORT', '9000')
    url = f"http://{host}:{http_port}/exec"
    query = (
        f"select symbol, min(timestamp) first_ts, max(timestamp) last_ts "
        f"from {table_name} where timestamp is not null group by symbol"
    )
    if sample:
        query += f" limit {max(1, min(limit, 5000))}"
    import aiohttp
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=25)) as session:
            async with session.get(url, params={'query': query, 'limit': 'max'}) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    raise HTTPException(status_code=502, detail=f"QuestDB HTTP {resp.status}: {txt[:160]}")
                data = await resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Equities coverage query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        dataset = data.get('dataset', [])
        # Parse results
        exclusions = _load_equity_exclusions()
        out: list[dict] = []
        adjusted_meets = 0
        for row in dataset:
            try:
                sym = str(row[0]).upper()
            except Exception:
                continue
            if sym in exclusions:
                continue
            first_raw = row[1] if len(row) > 1 else None
            last_raw = row[2] if len(row) > 2 else None
            try:
                first_dt = datetime.fromisoformat(str(first_raw).replace('Z','+00:00')) if first_raw else None
                last_dt = datetime.fromisoformat(str(last_raw).replace('Z','+00:00')) if last_raw else None
            except Exception:
                continue
            if not (first_dt and last_dt):
                continue
            years_span = (last_dt - first_dt).days / 365.25
            meets_20y = years_span >= 19.5
            listing_dt = await _get_listing_date(sym)
            listing_age_years = None
            ipo_adjusted_meets = None
            if include_adjusted:
                if listing_dt and listing_dt < last_dt:
                    listing_age_years = (last_dt - listing_dt).days / 365.25
                else:
                    # Fallback: treat earliest bar as listing date (may overestimate if data missing)
                    listing_age_years = years_span
                # IPO-adjusted rule: if listing age < 20y then treat as satisfied if coverage spans >= 90% of listing age
                # plus always count genuine 20y span.
                if listing_age_years < 19.5:
                    ipo_adjusted_meets = (years_span / max(listing_age_years, 0.01)) >= 0.9
                else:
                    ipo_adjusted_meets = meets_20y
                if ipo_adjusted_meets:
                    adjusted_meets += 1
            out.append({
                'symbol': sym,
                'first_date': first_dt.date().isoformat(),
                'last_date': last_dt.date().isoformat(),
                'years_span': round(years_span, 2),
                'meets_20y': meets_20y,
                'listing_age_years': round(listing_age_years, 2) if listing_age_years is not None else None,
                'ipo_adjusted_meets': ipo_adjusted_meets,
            })

        coverage_ratio = (sum(1 for r in out if r['meets_20y']) / len(out)) if out else 0.0
        adjusted_ratio = (adjusted_meets / len(out) if len(out) > 0 else None) if include_adjusted else None

        # Update gauges only on full scan
        if not sample:
            try:
                if EQUITIES_COVERAGE_RATIO_20Y is not None:
                    EQUITIES_COVERAGE_RATIO_20Y.set(coverage_ratio)
                if EQUITIES_COVERAGE_SYMBOLS_TOTAL is not None:
                    EQUITIES_COVERAGE_SYMBOLS_TOTAL.set(len(out))
                if EQUITIES_COVERAGE_SYMBOLS_20Y is not None:
                    EQUITIES_COVERAGE_SYMBOLS_20Y.set(sum(1 for r in out if r['meets_20y']))
            except Exception:
                pass

        return {
            'status': 'success',
            'symbols_evaluated': len(out),
            'coverage_20y_ratio': round(coverage_ratio, 3),
            'coverage_ipo_adjusted_ratio': round(adjusted_ratio, 3) if adjusted_ratio is not None else None,
            'table': table_name,
            'sampled': sample,
            'data': out[:limit]
        }
    except Exception as e:
        logger.error(f"Failed to process equities coverage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/coverage/equities/remediate")
async def equities_coverage_remediate(
    target_years: int = 20,
    max_symbols: int = 25,
    dry_run: bool = True,
    pacing_seconds: float = 0.15,
    min_mature_years: float = 7.0,
):
    """Attempt remediation (historical backfill) for deficient mature symbols.

    Strategy:
      * Full scan (sample=false) to obtain earliest bar per symbol (exclusions respected).
      * Identify symbols with years_span < target_years - 0.5 AND listing_age_years >= min_mature_years.
      * For each, compute missing range start = (last_ts - target_years years) and ingest missing early window
        prior to current earliest date (bounded by 25y safety).
      * Dry-run mode reports planned actions without ingestion.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    # Acquire coverage (full scan) by calling internal logic (reuse query code rather than HTTP self-call for efficiency)
    request = await equities_coverage_full(sample=False, limit=10**9, include_adjusted=True)  # reuse function
    data = request['data']
    now = datetime.utcnow()
    # Define target earliest date
    target_delta_days = int(target_years * 365.25)
    plan: list[dict] = []
    exclusions = _load_equity_exclusions()
    for entry in data:
        sym = entry['symbol']
        if sym in exclusions:
            continue
        years_span = entry['years_span']
        listing_age = entry.get('listing_age_years') or years_span
        if years_span >= target_years - 0.5:
            continue  # already adequate
        if listing_age < min_mature_years:
            continue  # young listing, skip remediation
        earliest = datetime.strptime(entry['first_date'], '%Y-%m-%d')
        latest = datetime.strptime(entry['last_date'], '%Y-%m-%d')
        target_start = latest - timedelta(days=target_delta_days)
        # If earliest <= target_start -> missing data *might* be small or none; skip
        if earliest <= target_start + timedelta(days=30):  # tolerance 30d
            continue
        fetch_start = target_start
        fetch_end = earliest - timedelta(days=1)
        plan.append({
            'symbol': sym,
            'missing_days_estimate': (earliest - fetch_start).days,
            'fetch_start': fetch_start.date().isoformat(),
            'fetch_end': fetch_end.date().isoformat(),
        })
        if len(plan) >= max_symbols:
            break
    performed = []
    total_bars = 0
    if not dry_run:
        for item in plan:
            sym = item['symbol']
            try:
                start_dt = datetime.strptime(item['fetch_start'], '%Y-%m-%d')
                end_dt = datetime.strptime(item['fetch_end'], '%Y-%m-%d')
                rows = await market_data_svc.get_bulk_daily_historical(sym, start_dt, end_dt)
                bars = len(rows)
                total_bars += bars
                performed.append({'symbol': sym, 'bars_ingested': bars})
                if EQUITIES_REMEDIATED_SYMBOLS:
                    EQUITIES_REMEDIATED_SYMBOLS.inc()
                if EQUITIES_REMEDIATED_BARS:
                    EQUITIES_REMEDIATED_BARS.inc(bars)
                await asyncio.sleep(pacing_seconds)
            except Exception as e:  # noqa: BLE001
                performed.append({'symbol': sym, 'error': str(e)})
                if EQUITIES_REMEDIATION_RUNS:
                    try:
                        EQUITIES_REMEDIATION_RUNS.labels(result='error').inc()
                    except Exception:
                        pass
        if EQUITIES_REMEDIATION_RUNS:
            try:
                EQUITIES_REMEDIATION_RUNS.labels(result='success').inc()
            except Exception:
                pass
    else:
        if EQUITIES_REMEDIATION_RUNS:
            try:
                EQUITIES_REMEDIATION_RUNS.labels(result='dry_run').inc()
            except Exception:
                pass
    return {
        'status': 'dry_run' if dry_run else 'executed',
        'generated_at': datetime.utcnow().isoformat(),
        'target_years': target_years,
        'symbols_considered': len(data),
        'plan_count': len(plan),
        'executed': performed if not dry_run else [],
        'plan': plan,
        'total_bars_ingested': total_bars if not dry_run else 0,
        'exclusions': sorted(list(exclusions)),
    }

@app.post("/coverage/equities/export")
async def equities_coverage_export(limit: int = 5000, include_adjusted: bool = True):
    """Compute equities coverage and write a stable JSON artifact used by Grafana."""
    try:
        report = await equities_coverage_full(limit=limit, sample=False, include_adjusted=include_adjusted)
        out_dir = os.getenv("EQUITIES_COVERAGE_OUTPUT_DIR", "/app/export/grafana-csv").rstrip("/")
        os.makedirs(out_dir, exist_ok=True)
        import json as _json
        stable = os.path.join(out_dir, 'equities_coverage.json')
        dated = os.path.join(out_dir, f"equities_coverage_{datetime.utcnow().strftime('%Y%m%d')}.json")
        with open(stable, 'w') as f:
            _json.dump(report, f, indent=2)
        with open(dated, 'w') as f:
            _json.dump(report, f, indent=2)
        logger.info("Equities coverage report written", path=stable, symbols=report.get('symbols_evaluated'))
        return {"status": "success", "path": stable, "symbols": report.get('symbols_evaluated', 0)}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Equities coverage export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coverage/options/export")
async def export_options_coverage(underlyings: Optional[List[str]] = Query(default=None)):
    """Compute options coverage summary and write a stable JSON artifact (mirrors scheduled loop)."""
    import aiohttp
    host = os.getenv('QUESTDB_HOST', 'trading-questdb')
    http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
    qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
    out_dir = os.getenv("OPTIONS_COVERAGE_OUTPUT_DIR", "/app/export/grafana-csv").rstrip("/")

    async def _q(session: aiohttp.ClientSession, sql: str) -> dict:
        async with session.get(qdb_url, params={"query": sql}) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"QuestDB HTTP {resp.status}: {txt[:160]}")
            return await resp.json()

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            # resolve underlyings
            syms: List[str] = []
            if underlyings:
                syms = [s.strip().upper() for s in underlyings.split(',') if s.strip()]
            elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                try:
                    syms = (await reference_svc.get_watchlist_symbols()) or []
                except Exception:
                    syms = []
            if not syms:
                syms = ['AAPL','MSFT','TSLA','NVDA','SPY']
            syms = syms[:max(1, max_underlyings)]

            # compute coverage
            out = []
            for u in syms:
                try:
                    sql_summary = (
                        "select count() as rows, count_distinct(option_symbol) as contracts, "
                        "min(timestamp) as first_ts, max(timestamp) as last_ts "
                        f"from options_data where underlying = '{u}'"
                    )
                    data = await _q(session, sql_summary)
                    if not data.get('dataset'):
                        out.append({"underlying": u, "rows": 0, "contracts": 0, "first_day": None, "last_day": None, "recent_gap_days_30d": None})
                        continue
                    r = data['dataset'][0]
                    cols = {c['name']: i for i, c in enumerate(data.get('columns', []))}
                    rows = int(r[cols['rows']]) if 'rows' in cols else 0
                    contracts = int(r[cols['contracts']]) if 'contracts' in cols else 0
                    # Timestamps are ISO strings; format to YYYY-MM-DD
                    def _fmt_iso_day(v):
                        try:
                            return str(v)[:10]
                        except Exception:
                            return None
                    first_day = _fmt_iso_day(r[cols['first_ts']]) if 'first_ts' in cols else None
                    last_day = _fmt_iso_day(r[cols['last_ts']]) if 'last_ts' in cols else None
                    sql_recent = (
                        "select count_distinct(cast(timestamp as LONG)/86400000000) as have_days "
                        f"from options_data where underlying = '{u}' and timestamp >= dateadd('d', -30, now())"
                    )
                    d2 = await _q(session, sql_recent)
                    have_days = 0
                    if d2.get('dataset'):
                        c2 = {c['name']: i for i, c in enumerate(d2.get('columns', []))}
                        try:
                            have_days = int(d2['dataset'][0][c2['have_days']])
                        except Exception:
                            have_days = 0
                    out.append({
                        "underlying": u,
                        "rows": rows,
                        "contracts": contracts,
                        "first_day": first_day,
                        "last_day": last_day,
                        "recent_gap_days_30d": max(0, 30 - have_days),
                    })
                except Exception as e:  # noqa: BLE001
                    out.append({"underlying": u, "error": str(e)})

            # write JSON artifact
            os.makedirs(out_dir, exist_ok=True)
            payload = {"generated_at": datetime.utcnow().isoformat(), "questdb": qdb_url, "coverage": out}
            import json as _json
            stable_path = os.path.join(out_dir, 'options_coverage.json')
            date_path = os.path.join(out_dir, f"options_coverage_{datetime.utcnow().strftime('%Y%m%d')}.json")
            with open(stable_path, 'w') as f:
                _json.dump(payload, f, indent=2)
            with open(date_path, 'w') as f:
                _json.dump(payload, f, indent=2)
            logger.info("Options coverage report written", path=stable_path, items=len(out))
        return {"status": "success", "path": stable_path, "items": len(out)}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Options coverage export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Calendar (EODHD) Endpoints ---------------------- #
@app.get("/calendar/coverage")
async def calendar_coverage():
    """Summarize QuestDB calendar tables (earnings, IPO, splits, dividends)."""
    try:
        out: dict = {"timestamp": datetime.utcnow().isoformat(), "tables": {}}
        for table, date_col in (
            ("earnings_calendar", "date"),
            ("ipo_calendar", "date"),
            ("splits_calendar", "date"),
            ("dividends_calendar", "ex_date"),
        ):
            try:
                data = await _qdb_exec(
                    f"select count() as rows, min({date_col}) as first_day, max({date_col}) as last_day from {table}"
                )
                cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
                r = (data.get('dataset') or [None])[0]
                if not r:
                    out["tables"][table] = {"rows": 0, "first_day": None, "last_day": None}
                    continue
                out["tables"][table] = {
                    "rows": int(r[cols.get('rows', -1)] or 0),
                    "first_day": _fmt_iso_day(r[cols.get('first_day')]) if 'first_day' in cols else None,
                    "last_day": _fmt_iso_day(r[cols.get('last_day')]) if 'last_day' in cols else None,
                }
            except Exception as e:  # noqa: BLE001
                out["tables"][table] = {"error": str(e)[:200]}
        return out
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": "calendar_coverage_failed", "detail": str(e)[:400]})


@app.post("/calendar/backfill")
async def calendar_backfill(
    years: int = Query(default=int(os.getenv("CALENDAR_BACKFILL_YEARS", "5"))),
    include: str = Query(default="earnings,ipo,splits,dividends"),
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD optional start date"),
    end: Optional[str] = Query(default=None, description="YYYY-MM-DD optional end date"),
):
    """Calendar backfill with provider routing.

    Behavior:
      - Earnings & IPO are collected from Alpha Vantage when ALPHAVANTAGE_API_KEY is available
        (upcoming windows via AV CSV). Falls back to EODHD range collectors when AV is not configured.
      - Splits & Dividends are collected from EODHD range collectors.

    Query params:
      - years: integer window size (default from CALENDAR_BACKFILL_YEARS env, fallback 5)
      - include: comma list of sections to include: earnings, ipo, splits, dividends
      - start/end: optional explicit YYYY-MM-DD range for EODHD collectors (AV remains upcoming-only)
    """
    try:
        global calendar_svc
        if not calendar_svc or not getattr(calendar_svc, 'enabled', False):
            return JSONResponse(status_code=503, content={"error": "calendar_service_disabled"})
        if start and end:
            try:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
                end_dt = datetime.strptime(end, "%Y-%m-%d")
            except Exception:
                return JSONResponse(status_code=400, content={"error": "invalid_date_format", "detail": "Use YYYY-MM-DD for start/end"})
        else:
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=int(max(1, years) * 365.25))
        parts = [p.strip().lower() for p in (include or '').split(',') if p.strip()]
        results: dict = {
            "start": start_dt.strftime('%Y-%m-%d'),
            "end": end_dt.strftime('%Y-%m-%d'),
            "routing": "earnings/ipo=alphavantage(if available), splits/dividends=eodhd",
        }
        # Alpha Vantage availability
        has_av = bool(getattr(calendar_svc, 'av_api_key', None))

        # Earnings: prefer AV upcoming, else EODHD range
        if 'earnings' in parts:
            try:
                if has_av:
                    # Map years to AV horizon; years is int so use 12 months for broader coverage
                    horizon = '12month'
                    results['earnings'] = await calendar_svc.collect_av_earnings_upcoming(horizon=horizon)
                    results['earnings_provider'] = 'alphavantage'
                else:
                    results['earnings'] = await calendar_svc.collect_earnings_range(start_dt, end_dt)
                    results['earnings_provider'] = 'eodhd'
            except Exception as e:  # noqa: BLE001
                results['earnings_error'] = str(e)[:200]

        # IPO: prefer AV upcoming, else EODHD range
        if 'ipo' in parts or 'ipos' in parts:
            try:
                if has_av:
                    results['ipo'] = await calendar_svc.collect_av_ipo_upcoming()
                    results['ipo_provider'] = 'alphavantage'
                else:
                    results['ipo'] = await calendar_svc.collect_ipo_range(start_dt, end_dt)
                    results['ipo_provider'] = 'eodhd'
            except Exception as e:  # noqa: BLE001
                results['ipo_error'] = str(e)[:200]
        if 'splits' in parts:
            try:
                results['splits'] = await calendar_svc.collect_splits_range(start_dt, end_dt)
            except Exception as e:  # noqa: BLE001
                results['splits_error'] = str(e)[:200]
        if 'dividends' in parts:
            try:
                results['dividends'] = await calendar_svc.collect_dividends_range(start_dt, end_dt)
            except Exception as e:  # noqa: BLE001
                results['dividends_error'] = str(e)[:200]
        return results
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": "calendar_backfill_failed", "detail": str(e)[:400]})


# ---------------------- Admin: Vector Options Index ---------------------- #
@app.post("/admin/vector/options/index")
async def admin_vector_options_index(
    underlyings: str = Query(default="AAPL,MSFT,SPY"),
    limit: int = Query(default=200),
    days: int = Query(default=14),
):
    """Populate Weaviate OptionContract objects from QuestDB options_data.

    - underlyings: comma-separated list of tickers
    - limit: max contracts to index
    - days: lookback window for recent option rows to project (by timestamp)
    """
    try:
        if index_options_fallback is None:
            return JSONResponse(status_code=503, content={"error": "fallback_indexer_unavailable"})
        syms = [s.strip().upper() for s in (underlyings or '').split(',') if s.strip()]
        if not syms:
            return {"indexed": 0, "items": 0}
        # Discover available columns to build compatible projection
        try:
            meta = await _qdb_exec("show columns from options_data")
            name_idx = next((i for i, c in enumerate(meta.get('columns', []) or []) if c.get('name') == 'column'), None)
            cols_available: list[str] = []
            for r in meta.get('dataset') or []:
                try:
                    if name_idx is not None:
                        cols_available.append(str(r[name_idx]))
                except Exception:
                    continue
        except Exception:
            cols_available = []
        # Build select list with graceful fallbacks
        parts = ["underlying", "option_symbol", "timestamp"]
        for c in ("expiry", "option_type", "strike", "implied_vol"):
            if c in cols_available and c not in parts:
                parts.append(c)
        select_list = ", ".join(parts)
        # Construct SQL
        look_days = max(1, int(days))
        in_list = ",".join([f"'{s}'" for s in syms])
        sql = (
            f"select {select_list} from options_data where underlying in ({in_list}) "
            f"and timestamp >= dateadd('d', -{look_days}, now()) order by timestamp desc limit {max(1, int(limit))}"
        )
        data = await _qdb_exec(sql, timeout=20.0)
        cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
        items = []
        for r in data.get('dataset') or []:
            try:
                itm = {
                    'underlying': str(r[cols['underlying']]).upper() if 'underlying' in cols else '',
                    'option_symbol': str(r[cols.get('option_symbol')]) if 'option_symbol' in cols else '',
                    'timestamp': str(r[cols.get('timestamp')]) if 'timestamp' in cols else '',
                    'expiry': str(r[cols.get('expiry')]) if 'expiry' in cols else '',
                    'right': str(r[cols.get('option_type')]) if 'option_type' in cols else '',
                    'strike': r[cols.get('strike')] if 'strike' in cols else None,
                    'implied_vol': r[cols.get('implied_vol')] if 'implied_vol' in cols else None,
                }
            except Exception:
                continue
            items.append(itm)
        # Index via fallback
        inserted = 0
        if items:
            try:
                inserted = await index_options_fallback(items)
            except Exception:
                inserted = 0
        return {"requested": len(syms), "items": len(items), "indexed": inserted}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": "options_index_failed", "detail": str(e)[:400]})


@app.post("/admin/vector/options/index/auto")
async def admin_vector_options_index_auto(
    days: int = Query(default=14),
    limit: int = Query(default=25),
    per_underlying: int = Query(default=250),
):
    """Auto-discover top underlyings by recent options rows and index into Weaviate.

    - days: lookback window to rank underlyings
    - limit: number of underlyings to index
    - per_underlying: per-underlying row cap to project
    """
    try:
        if index_options_fallback is None:
            return JSONResponse(status_code=503, content={"error": "fallback_indexer_unavailable"})
        look_days = max(1, int(days))
        rank_sql = (
            "select underlying, count() as c from options_data "
            f"where timestamp >= dateadd('d', -{look_days}, now()) and underlying is not null "
            f"group by underlying order by c desc limit {max(1, int(limit))}"
        )
        ranked = await _qdb_exec(rank_sql, timeout=15.0)
        cols_r = {c['name']: i for i, c in enumerate(ranked.get('columns', []) or [])}
        underlyings: list[str] = []
        for r in ranked.get('dataset') or []:
            try:
                u = str(r[cols_r['underlying']]).upper()
                if u:
                    underlyings.append(u)
            except Exception:
                continue
        if not underlyings:
            return {"underlyings": [], "items": 0, "indexed": 0}
        # Build select and fetch
        meta = await _qdb_exec("show columns from options_data")
        name_idx = next((i for i, c in enumerate(meta.get('columns', []) or []) if c.get('name') == 'column'), None)
        cols_avail: list[str] = []
        for r in meta.get('dataset') or []:
            if name_idx is not None:
                cols_avail.append(str(r[name_idx]))
        parts = ["underlying", "option_symbol", "timestamp"]
        for c in ("expiry", "option_type", "strike", "implied_vol"):
            if c in cols_avail and c not in parts:
                parts.append(c)
        select_list = ", ".join(parts)
        in_list = ",".join([f"'{u}'" for u in underlyings])
        sql = (
            f"select {select_list} from options_data where underlying in ({in_list}) "
            f"and timestamp >= dateadd('d', -{look_days}, now()) order by timestamp desc limit {max(1, int(per_underlying)) * len(underlyings)}"
        )
        data = await _qdb_exec(sql, timeout=25.0)
        cols = {c['name']: i for i, c in enumerate(data.get('columns', []) or [])}
        items: list[dict] = []
        for r in data.get('dataset') or []:
            try:
                items.append({
                    'underlying': str(r[cols['underlying']]).upper() if 'underlying' in cols else '',
                    'option_symbol': str(r[cols.get('option_symbol')]) if 'option_symbol' in cols else '',
                    'timestamp': str(r[cols.get('timestamp')]) if 'timestamp' in cols else '',
                    'expiry': str(r[cols.get('expiry')]) if 'expiry' in cols else '',
                    'right': str(r[cols.get('option_type')]) if 'option_type' in cols else '',
                    'strike': r[cols.get('strike')] if 'strike' in cols else None,
                    'implied_vol': r[cols.get('implied_vol')] if 'implied_vol' in cols else None,
                })
            except Exception:
                continue
        inserted = 0
        if items:
            try:
                inserted = await index_options_fallback(items)
            except Exception:
                inserted = 0
        return {"underlyings": underlyings, "items": len(items), "indexed": inserted}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "options_index_auto_failed", "detail": str(e)[:400]})


@app.get("/news/backfill")
async def news_backfill(
    symbols: Optional[str] = Query(default=None, description="Comma-separated symbols; defaults to NEWS_STREAM_SYMBOLS"),
    days: int = Query(default=45, description="Lookback days for backfill"),
    batch_days: int = Query(default=7),
    max_articles: int = Query(default=60),
):
    """Run a bounded news backfill over the recent window using configured providers, with vectorization enabled.

    Returns a summary with total articles and per-batch counts.
    """
    try:
        if not news_svc:
            return JSONResponse(status_code=503, content={"error": "news_service_unavailable"})
        # Resolve symbols
        if symbols:
            syms = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        else:
            env_list = os.getenv("NEWS_STREAM_SYMBOLS", "AAPL,MSFT,SPY,TSLA,NVDA").strip()
            syms = [s.strip().upper() for s in env_list.split(',') if s.strip()]
        if not syms:
            return JSONResponse(status_code=400, content={"error": "no_symbols"})
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=max(1, int(days)))
        total, batches = await news_svc.collect_financial_news_range(
            syms, start_dt, end_dt, batch_days=max(1, int(batch_days)), max_articles_per_batch=max(1, int(max_articles)), backfill_mode=True
        )
        return {"symbols": syms[:10] + (["…"] if len(syms) > 10 else []), "start": start_dt.strftime('%Y-%m-%d'), "end": end_dt.strftime('%Y-%m-%d'), "total": total, "batches": batches}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "news_backfill_failed", "detail": str(e)[:400]})


@app.post("/news/backfill")
async def news_backfill_post(
    symbols: Optional[str] = Query(default=None),
    days: int = Query(default=45),
    batch_days: int = Query(default=7),
    max_articles: int = Query(default=60),
):
    return await news_backfill(symbols=symbols, days=days, batch_days=batch_days, max_articles=max_articles)

@app.get("/debug/polygon-s3-keys")
async def debug_polygon_s3_keys(prefix: str | None = Query(None), max_keys: int = Query(50, ge=1, le=1000)):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    try:
        keys = await news_svc.debug_list_polygon_news_keys(prefix=prefix, max_keys=max_keys)
        return {"count": len(keys), "prefix": prefix or "news/", "keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")

@app.get("/debug/polygon-s3-probe")
async def debug_polygon_s3_probe(date: str = Query(..., description="YYYY-MM-DD")):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    try:
        day = datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid date format; expected YYYY-MM-DD")
    result = await news_svc.debug_probe_polygon_news_keys_for_date(day)
    return result

@app.get("/debug/options-precheck")
async def debug_options_precheck():
    svc = market_data_svc
    if not svc:
        raise HTTPException(status_code=503, detail="market_data_service unavailable")
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "enable_options_ingest": bool(getattr(svc, 'enable_options_ingest', False)),
        "enable_questdb_persist": bool(getattr(svc, 'enable_questdb_persist', False)),
        "enable_hist_dry_run": bool(getattr(svc, 'enable_hist_dry_run', False)),
        "questdb_conf": getattr(svc, 'questdb_conf', None),
        "sender_available": bool(getattr(svc, 'questdb_conf', None) and 'Sender' in str(type(getattr(svc, 'questdb_conf', '')))),
        "polygon_key_present": bool(svc.polygon_config.get('api_key') if svc else False),
    }

@app.get("/debug/news-precheck")
async def debug_news_precheck():
    svc = news_svc
    if not svc:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "available": False,
            "reason": "news_service_not_running"
        }
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "available": True,
        "enable_news_questdb_persist": bool(getattr(svc, 'enable_questdb_persist', False)),
        "questdb_conf": getattr(svc, '_qdb_conf', None),
        "ilp_sender_ready": bool(getattr(svc, '_qdb_sender', None) and getattr(svc, '_qdb_conf', None)),
        "news_api_key": bool(svc.news_api_config.get('api_key')),
        "alpha_vantage_key": bool(svc.alpha_vantage_config.get('api_key')),
        "finnhub_key": bool(svc.finnhub_config.get('api_key')),
    }


@app.post("/ingest/market-data")
async def ingest_market_data(symbol: str):
    """Trigger market data ingestion for a symbol."""
    try:
        logger.info(f"Starting market data ingestion for {symbol}")
        
        # For Phase 2, this is a placeholder that would:
        # 1. Connect to Alpaca/Polygon API
        # 2. Fetch real-time data for symbol
        # 3. Store in QuestDB via cache layer
        # 4. Publish to message queue for downstream processing
        
        sample_data = MarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            open=100.0,
            high=102.5,
            low=99.8,
            close=101.2,
            volume=1000000,
            timeframe="1min",
            data_source="sample"
        )
        
        # Store in cache (would be real implementation)
        if cache_client:
            await cache_client.set_market_data(sample_data)
            
        logger.info(f"Successfully ingested market data for {symbol}")
        return {
            "status": "success",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Market data ingestion completed"
        }
        
    except Exception as e:
        logger.error(f"Market data ingestion failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/news")
async def ingest_news(query: str = "SPY TSLA AAPL"):
    """Trigger news ingestion for given query."""
    try:
        logger.info(f"Starting news ingestion for query: {query}")
        
        # Placeholder for Phase 2 - would implement:
        # 1. Connect to NewsAPI/other news sources
        # 2. Fetch relevant financial news
        # 3. Run sentiment analysis
        # 4. Store processed news with sentiment scores
        
        sample_news = NewsItem(
            title="Market Update: Tech Stocks Rally",
            content="Technology stocks showed strong performance...",
            source="Financial Times",
            published_at=datetime.utcnow(),
            url="https://example.com/news/123",
            sentiment_score=0.7,
            relevance_score=0.9,
            symbols=["SPY", "TSLA", "AAPL"]
        )
        
        if cache_client:
            await cache_client.set_news_item(sample_news)
            
        logger.info(f"Successfully ingested news for query: {query}")
        return {
            "status": "success", 
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "News ingestion completed"
        }
        
    except Exception as e:
        logger.error(f"News ingestion failed for query {query}: {e}")
        raise HTTPException(status_code=500, detail=f"News ingestion failed: {str(e)}")


@app.get("/data/recent/{symbol}")
async def get_recent_data(symbol: str, hours: int = 1):
    """Get recent market data for a symbol."""
    try:
        if not cache_client:
            raise HTTPException(status_code=503, detail="Cache service unavailable")
            
        # Placeholder - would query actual cached data
        logger.info(f"Retrieving recent data for {symbol} (last {hours} hours)")
        
        return {
            "symbol": symbol,
            "timeframe": f"last_{hours}_hours",
            "data_points": 0,  # Would return actual data
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"No data available yet - service in development mode"
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Data retrieval failed: {str(e)}")


# New comprehensive endpoints

@app.post("/market-data/quote/{symbol}")
async def get_real_time_quote(symbol: str):
    """Get real-time quote for a symbol."""
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    
    try:
        quote = await market_data_svc.get_real_time_quote(symbol.upper())
        if quote:
            # Validate the data
            if validation_svc:
                validation_results = await validation_svc.validate_market_data(quote)
                has_errors = any(r.severity.value == "error" for r in validation_results)
                
                return {
                    "data": {
                        "symbol": quote.symbol,
                        "timestamp": quote.timestamp.isoformat(),
                        "open": quote.open,
                        "high": quote.high,
                        "low": quote.low,
                        "close": quote.close,
                        "volume": quote.volume,
                        "source": quote.data_source
                    },
                    "validation": {
                        "valid": not has_errors,
                        "issues": len(validation_results),
                        "details": [{"severity": r.severity.value, "message": r.message} for r in validation_results]
                    }
                }
            else:
                return {"data": quote, "validation": {"valid": True, "issues": 0}}
        else:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Artifact Upload to MinIO ---------------------- #

def _load_minio_client():
    """Lazy import and construct MinIO client using shared helper.

    Returns (client, cfg) or (None, None) if unavailable.
    """
    try:
        from shared.storage.minio_storage import get_minio_client, MinIOConfig  # type: ignore
        # Support base64-encoded secret to avoid issues with $ in docker-compose interpolation
        secret_b64 = os.getenv("MINIO_SECRET_KEY_B64", "").strip()
        if secret_b64:
            import base64
            try:
                decoded = base64.b64decode(secret_b64).decode("utf-8")
                secret = decoded
            except Exception:
                secret = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", ""))
        else:
            secret = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", ""))
        cfg = MinIOConfig(
            endpoint=os.getenv("MINIO_ENDPOINT", os.getenv("MINIO_URL", "localhost:9000")),
            access_key=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "")),
            secret_key=secret,
            secure=os.getenv("MINIO_SECURE", "false").lower() in ("1","true","yes"),
            region=os.getenv("MINIO_REGION"),
        )
        client = get_minio_client(cfg)
        return client, cfg
    except Exception:
        return None, None


@app.post("/artifacts/upload")
async def upload_artifacts_to_minio(
    directory: str = Query(default=os.getenv("GRAFANA_EXPORT_DIR", "/app/export/grafana-csv"), description="Directory containing JSON artifacts"),
    bucket: str = Query(default=os.getenv("MINIO_ARTIFACTS_BUCKET", "trading"), description="MinIO bucket name"),
    prefix: str = Query(default=os.getenv("MINIO_ARTIFACTS_PREFIX", "dashboards"), description="Key prefix inside bucket"),
    pattern: str = Query(default="*.json", description="Glob pattern of files to upload"),
):
    """Upload local JSON artifacts (e.g., coverage reports) to MinIO.

    Walks the given directory and uploads matching files to s3://{bucket}/{prefix}/filename.
    Returns a summary with successes and failures.
    """
    client, cfg = _load_minio_client()
    if not client:
        raise HTTPException(status_code=503, detail="MinIO client unavailable or credentials not configured")
    try:
        from shared.storage.minio_storage import ensure_bucket  # type: ignore
    except Exception:
        ensure_bucket = None  # type: ignore

    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    import glob, json
    files = list(dir_path.glob(pattern))
    uploaded: list[dict] = []
    errors: list[dict] = []

    # Ensure bucket exists (best-effort)
    if ensure_bucket:
        try:
            ensure_bucket(bucket, client=client)  # type: ignore
        except Exception as e:  # noqa: BLE001
            # Continue; upload attempts may still create lazily
            logger.warning("ensure_bucket failed", bucket=bucket, error=str(e))

    for f in files:
        try:
            data = f.read_bytes()
            key = f"{prefix.strip('/')}" if prefix else ""
            if key:
                key += "/" + f.name
            else:
                key = f.name
            # Use client directly to avoid extra deps
            from io import BytesIO
            stream = BytesIO(data)
            client.put_object(bucket, key, stream, length=len(data), content_type="application/json")  # type: ignore[attr-defined]
            uploaded.append({"file": str(f), "bucket": bucket, "key": key, "bytes": len(data)})
        except Exception as e:  # noqa: BLE001
            errors.append({"file": str(f), "error": str(e)})

    return {
        "status": "completed",
        "directory": str(dir_path),
        "bucket": bucket,
        "prefix": prefix,
        "uploaded": uploaded,
        "uploaded_count": len(uploaded),
        "errors": errors,
        "error_count": len(errors),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/market-data/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    timeframe: str = "1min",
    hours_back: int = 24,
    limit: int = 1000,
    start: Optional[str] = Query(None, description="Optional start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="Optional end date YYYY-MM-DD"),
):
    """Get historical market data.

    Modes:
      1. Intraday/legacy (default): timeframe != '1d' OR no start supplied.
         Uses hours_back window ending now and underlying provider specific intraday retrieval.
      2. Bulk daily (extended history): timeframe == '1d' AND start supplied (YYYY-MM-DD).
         Optional end (YYYY-MM-DD, default = today). Delegates to bulk daily provider chain
         (EODHD primary) for 20y scale retrieval.

    Backward compatibility: existing callers that only pass timeframe/hours_back continue working.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")

    try:
        symbol_u = symbol.upper()

        # ---------------- Bulk Daily Path (EODHD) ----------------
        if timeframe == '1d' and start:
            try:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
            if end:
                try:
                    end_dt = datetime.strptime(end, "%Y-%m-%d")
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid end format, expected YYYY-MM-DD")
            else:
                # End date defaults to today (UTC date boundary)
                end_dt = datetime.utcnow()

            if start_dt > end_dt:
                raise HTTPException(status_code=400, detail="start must be <= end")

            # Hard guard: limit to 25 years (same as backfill endpoint)
            if (end_dt - start_dt).days > 365 * 25:
                raise HTTPException(status_code=400, detail="Range too large; limit to 25 years")

            rows = await market_data_svc.get_bulk_daily_historical(symbol_u, start_dt, end_dt)

            return {
                "symbol": symbol_u,
                "timeframe": timeframe,
                "mode": "bulk_daily",
                "count": len(rows),
                "start": start_dt.strftime("%Y-%m-%d"),
                "end": end_dt.strftime("%Y-%m-%d"),
                "data_source_primary": ("EODHD" if os.getenv("EODHD_API_KEY") else None),
                "data": [
                    {
                        "timestamp": bar.timestamp.isoformat(),
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "source": "eodhd",
                    } for bar in rows
                ],
            }

        # ---------------- Intraday / Legacy Path ----------------
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)

        # Map to daily bars range when intraday not available
        start_dt = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
        rows = await market_data_svc.get_bulk_daily_historical(symbol_u, start_dt, end_dt)
        return {
            "symbol": symbol_u,
            "timeframe": timeframe,
            "mode": "intraday_window_mapped_to_daily",
            "count": len(rows),
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "data": [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "source": "eodhd",
                } for bar in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/news/collect")
async def collect_financial_news(
    symbols: Optional[List[str]] = Query(default=None),
    hours_back: int = 1,
    max_articles: int = 50,
    body: Optional[dict] = Body(default=None, description="Optional JSON body: { symbols: string[], hours_back?: number, max_articles?: number }")
):
    """Collect financial news."""
    if not news_svc:
        raise HTTPException(status_code=503, detail="News service not available")
    
    try:
        # Accept either query params or JSON body for convenience
        eff_symbols = symbols
        eff_hours_back = hours_back
        eff_max_articles = max_articles
        try:
            if body and isinstance(body, dict):
                if eff_symbols is None and isinstance(body.get('symbols'), list):
                    eff_symbols = [str(s).upper() for s in body.get('symbols') if isinstance(s, (str,)) and s.strip()]
                if 'hours_back' in body and isinstance(body.get('hours_back'), (int, float)):
                    eff_hours_back = int(body.get('hours_back'))
                if 'max_articles' in body and isinstance(body.get('max_articles'), (int, float)):
                    eff_max_articles = int(body.get('max_articles'))
        except Exception:
            # Ignore malformed body and fall back to query params
            pass

        news_items = await news_svc.collect_financial_news(eff_symbols, eff_hours_back, eff_max_articles)
        
        return {
            "status": "success",
            "symbols": eff_symbols,
            "articles_collected": len(news_items),
            "hours_back": eff_hours_back,
            "timestamp": datetime.utcnow().isoformat(),
            "articles": [
                {
                    "title": item.title,
                    "source": item.source,
                    "published_at": item.published_at.isoformat(),
                    "sentiment_score": item.sentiment_score,
                    "relevance_score": item.relevance_score,
                    "symbols": item.symbols,
                    "url": item.url
                } for item in news_items
            ]
        }
    except Exception as e:
        logger.error(f"Failed to collect news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Historical/Backfill Endpoints ---------------------- #

class NewsBackfillRequest(BaseModel):
    symbols: List[str]
    start: str
    end: str
    batch_days: int | None = Field(default=NEWS_BACKFILL_WINDOW_DAYS)
    max_articles_per_batch: int | None = Field(default=80)


@app.post("/news/backfill")
async def news_backfill(req: NewsBackfillRequest):
    """Backfill news over a historical date range (batched provider calls).

    Delegates to NewsService.collect_financial_news_range which persists/indexes
    according to feature flags (QuestDB, Weaviate, Pulsar).
    """
    if not news_svc:
        raise HTTPException(status_code=503, detail="News service not available")
    try:
        try:
            start_dt = datetime.strptime(req.start, "%Y-%m-%d")
            end_dt = datetime.strptime(req.end, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")
        if start_dt > end_dt:
            raise HTTPException(status_code=400, detail="start must be <= end")
        batch_days = max(1, int(req.batch_days or NEWS_BACKFILL_WINDOW_DAYS))
        max_articles = max(1, int(req.max_articles_per_batch or 80))
        total, batches = await news_svc.collect_financial_news_range(
            [s.strip().upper() for s in (req.symbols or []) if s and s.strip()],
            start_dt,
            end_dt,
            batch_days=batch_days,
            max_articles_per_batch=max_articles,
            backfill_mode=True,
        )
        return {
            "status": "completed",
            "symbols": [s.strip().upper() for s in (req.symbols or []) if s and s.strip()],
            "start": req.start,
            "end": req.end,
            "batch_days": batch_days,
            "max_articles_per_batch": max_articles,
            "articles_collected": int(total),
            "persisted_count": int(sum(int(b.get('persisted', 0) or 0) for b in batches)),
            "batches": batches,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("News backfill failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/market-data/options-chain/{underlying}")
async def backfill_options_chain_endpoint(
    underlying: str,
    start: str = Query(..., description="Historical start date (YYYY-MM-DD) for aggregates window"),
    end: str = Query(..., description="Historical end date (YYYY-MM-DD) inclusive for aggregates window"),
    max_contracts: int = Query(default=OPTIONS_HISTORY_MAX_CONTRACTS, ge=1, le=2000),
    pacing_seconds: float = Query(default=OPTIONS_HISTORY_PACING_SECONDS, ge=0.0, description="Sleep between contracts"),
    expired: bool = Query(default=False, description="Include only expired contracts (advanced)"),
    include_recent_expired: bool = Query(default=True, description="Augment active set with recently expired contracts"),
    recent_expired_days: int = Query(default=7, ge=0, le=30),
    start_expiry: Optional[str] = Query(default=None, description="Optional expiry window start (YYYY-MM-DD)"),
    end_expiry: Optional[str] = Query(default=None, description="Optional expiry window end (YYYY-MM-DD)"),
    persist_questdb: Optional[bool] = Query(default=None, description="Override QuestDB persistence flag for this run"),
    enable_ingest: Optional[bool] = Query(default=None, description="Override ENABLE_OPTIONS_INGEST for this run"),
):
    """Backfill daily option aggregates for a chain of an underlying over a window.

    Persists rows to QuestDB/Postgres when those feature flags are enabled.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    if not getattr(market_data_svc, 'enable_options_ingest', False):
        raise HTTPException(status_code=403, detail="Options ingestion disabled")
    try:
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid expiry format, expected YYYY-MM-DD")
    today_utc = datetime.utcnow().date()
    is_expired_actual = expiry_dt.date() < today_utc
    # Validation: if client asserts expired but it's not, or vice versa, provide clear error/help
    if expired and not is_expired_actual:
        raise HTTPException(status_code=400, detail="Parameter expired=true but expiry date is not in the past")
    # If not marked expired but actually expired, we continue (read-only) and annotate response
    response_notes: list[str] = []
    if (not expired) and is_expired_actual:
        response_notes.append("expiry is in the past; treat as expired contract (informational)")
    if is_expired_actual and include_recent_expired and recent_expired_days > 0:
        # Provide an informational note if contract is within the 'recent expired' window
        if (today_utc - expiry_dt.date()).days <= recent_expired_days:
            response_notes.append(f"contract expired within last {recent_expired_days} days")
    if right.upper() not in ("C","P"):
        raise HTTPException(status_code=400, detail="right must be C or P")
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
    if end:
        try:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end format, expected YYYY-MM-DD")
    else:
        end_dt = datetime.utcnow()
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")
    try:
        rows = await market_data_svc.get_option_daily_aggregates(
            underlying.upper(),
            expiry_dt,
            right.upper(),
            strike,
            start_dt,
            end_dt,
            option_ticker=option_ticker or None,
        )
        # Metrics: single-contract path
        try:
            if OPTIONS_BACKFILL_CONTRACTS:
                OPTIONS_BACKFILL_CONTRACTS.labels(path='single-contract').inc(1)
            if OPTIONS_BACKFILL_BARS:
                OPTIONS_BACKFILL_BARS.labels(path='single-contract').inc(len(rows))
        except Exception:
            pass
        return {
            "underlying": underlying.upper(),
            "expiry": expiry_dt.strftime('%Y-%m-%d'),
            "right": right.upper(),
            "strike": strike,
            "count": len(rows),
            "start": start_dt.strftime('%Y-%m-%d'),
            "end": end_dt.strftime('%Y-%m-%d'),
            "expired_param": expired,
            "expired_actual": is_expired_actual,
            "recent_expired_window_days": recent_expired_days,
            "notes": response_notes,
        }
    except Exception as e:
        try:
            if OPTIONS_BACKFILL_ERRORS:
                OPTIONS_BACKFILL_ERRORS.labels(path='single-contract').inc()
        except Exception:
            pass
        logger.error(f"Options ingestion failed for {underlying} {expiry} {right} {strike}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backfill/options")
async def backfill_options(
    underlyings: Optional[List[str]] = Query(default=None, description="Underlyings to process; defaults to watchlist sample"),
    start_expiry: Optional[str] = Query(default=None, description="Expiry start YYYY-MM-DD (optional)"),
    end_expiry: Optional[str] = Query(default=None, description="Expiry end YYYY-MM-DD (optional)"),
    start: str = Query(..., description="Historical bars start YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="Historical bars end YYYY-MM-DD (default: today)"),
    max_contracts: int = Query(default=OPTIONS_HISTORY_MAX_CONTRACTS, ge=1, le=10000),
    pacing_seconds: float = Query(default=OPTIONS_HISTORY_PACING_SECONDS, ge=0.0, le=5.0),
    limit_underlyings: int = Query(default=100, ge=1, le=2000),
    dry_run: bool = Query(default=False),
    persist_questdb: Optional[bool] = Query(default=None, description="Override QuestDB persistence flag for this run"),
    enable_ingest: Optional[bool] = Query(default=None, description="Override ENABLE_OPTIONS_INGEST for this run"),
):
    """Orchestrate historical options backfill across a set of underlyings.

    Delegates to market_data_svc.backfill_options_chain per underlying with pacing.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    if not getattr(market_data_svc, 'enable_options_ingest', False):
        raise HTTPException(status_code=403, detail="Options ingestion disabled")

    # Resolve underlyings
    eff_und: List[str] = []
    if underlyings:
        eff_und = [u.strip().upper() for u in underlyings if u and u.strip()]
    elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            eff_und = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            eff_und = []
    if not eff_und:
        eff_und = ['AAPL','MSFT','TSLA','NVDA','SPY']
    eff_und = eff_und[:max(1, limit_underlyings)]

    # Parse dates
    try:
        start_hist = datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
    end_hist = datetime.utcnow() if not end else datetime.strptime(end, "%Y-%m-%d")
    start_exp_dt = _parse_date_yyyy_mm_dd(start_expiry) if start_expiry else None
    end_exp_dt = _parse_date_yyyy_mm_dd(end_expiry) if end_expiry else None

    if dry_run:
        return {
            "status": "planned",
            "underlyings": eff_und,
            "start": start_hist.date().isoformat(),
            "end": end_hist.date().isoformat(),
            "start_expiry": start_exp_dt.date().isoformat() if start_exp_dt else None,
            "end_expiry": end_exp_dt.date().isoformat() if end_exp_dt else None,
            "max_contracts": max_contracts,
            "pacing_seconds": pacing_seconds,
        }

    totals = {"underlyings": len(eff_und), "contracts_processed": 0, "bars_ingested": 0, "errors": 0, "per_underlying": []}
    started = datetime.utcnow()
    # Temporary flag overrides
    orig_persist = getattr(market_data_svc, 'enable_questdb_persist', None)
    orig_ingest = getattr(market_data_svc, 'enable_options_ingest', None)
    try:
        if persist_questdb is not None:
            try:
                market_data_svc.enable_questdb_persist = bool(persist_questdb)
            except Exception:
                pass
        if enable_ingest is not None:
            try:
                market_data_svc.enable_options_ingest = bool(enable_ingest)
            except Exception:
                pass
        for u in eff_und:
            try:
                summary = await market_data_svc.backfill_options_chain(
                    u,
                    start_exp_dt,
                    end_exp_dt,
                    start_date=start_hist,
                    end_date=end_hist,
                    max_contracts=max_contracts,
                    pacing_seconds=pacing_seconds,
                )
                totals["contracts_processed"] += int(summary.get('contracts_processed', 0) or 0)
                totals["bars_ingested"] += int(summary.get('bars_ingested', 0) or 0)
                totals["per_underlying"].append({"underlying": u, **summary})
            except Exception as e:  # noqa: BLE001
                totals["per_underlying"].append({"underlying": u, "error": str(e)})
            await asyncio.sleep(max(0.0, pacing_seconds))
    finally:
        try:
            if orig_persist is not None:
                market_data_svc.enable_questdb_persist = orig_persist
        except Exception:
            pass
        try:
            if orig_ingest is not None:
                market_data_svc.enable_options_ingest = orig_ingest
        except Exception:
            pass

    return {
        "status": "completed",
        "started_at": started.isoformat(),
        "ended_at": datetime.utcnow().isoformat(),
        **totals,
        "start": start_hist.date().isoformat(),
        "end": end_hist.date().isoformat(),
    }


# ---------------------- News backfill endpoint ---------------------- #
from pydantic import BaseModel as _BaseModel

class NewsBackfillRequest(_BaseModel):
    symbols: list[str] | None = None
    start: str
    end: str
    batch_days: int | None = 14
    max_articles_per_batch: int | None = 80
    backfill_mode: bool | None = True
    persist_postgres: bool | None = True
    providers: list[str] | None = None  # e.g., ["polygon_s3","gdelt","newsapi","polygon_rest","alpha_vantage"]


@app.post("/backfill/news")
async def backfill_news(req: NewsBackfillRequest):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    # Parse dates
    try:
        start_dt = datetime.strptime(req.start, "%Y-%m-%d")
        end_dt = datetime.strptime(req.end, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format; expected YYYY-MM-DD")

    symbols = [s.strip().upper() for s in (req.symbols or []) if s and s.strip()]
    # Temporary provider gating by mutating configs, then restoring
    want = set((req.providers or []))
    restore = {
        'newsapi_key': news_svc.news_api_config.get('api_key'),
        'polygon_key': news_svc.polygon_config.get('api_key'),
        'alpha_key': news_svc.alpha_vantage_config.get('api_key'),
        'poly_s3_access': news_svc.polygon_flatfiles.get('access_key'),
        'poly_s3_secret': news_svc.polygon_flatfiles.get('secret_key'),
    }
    def _enable(name: str) -> bool:
        return not want or (name in want)
    # Gate providers
    if not _enable('newsapi'):
        news_svc.news_api_config['api_key'] = None
    if not _enable('polygon_rest'):
        news_svc.polygon_config['api_key'] = None
    if not _enable('alpha_vantage'):
        news_svc.alpha_vantage_config['api_key'] = None
    if not _enable('polygon_s3'):
        news_svc.polygon_flatfiles['access_key'] = None
        news_svc.polygon_flatfiles['secret_key'] = None
    # Postgres persistence toggle
    original_pg = getattr(news_svc, 'enable_postgres_persist', False)
    try:
        if req.persist_postgres is not None:
            news_svc.enable_postgres_persist = bool(req.persist_postgres)
        total, batches = await news_svc.collect_financial_news_range(
            symbols or ['AAPL','MSFT','NVDA','SPY'],
            start_dt,
            end_dt,
            batch_days=int(req.batch_days or 14),
            max_articles_per_batch=int(req.max_articles_per_batch or 80),
            backfill_mode=bool(req.backfill_mode or False),
        )
    finally:
        # Restore provider configs
        news_svc.news_api_config['api_key'] = restore['newsapi_key']
        news_svc.polygon_config['api_key'] = restore['polygon_key']
        news_svc.alpha_vantage_config['api_key'] = restore['alpha_key']
        news_svc.polygon_flatfiles['access_key'] = restore['poly_s3_access']
        news_svc.polygon_flatfiles['secret_key'] = restore['poly_s3_secret']
        news_svc.enable_postgres_persist = original_pg

    return {
        "status": "completed",
        "symbols": symbols,
        "articles_total": total,
        "batches": batches,
    }

@app.get("/coverage/equities_full")
async def equities_coverage_full(limit: int = 500, sample: bool = True, include_adjusted: bool = True, mature_years: float = 7.0):
    """Compute equities historical coverage from QuestDB and optionally update gauges.

    - Reads earliest/latest TIMESTAMP per symbol from table (env EQUITY_DAILY_TABLE, default 'market_data').
    - Returns list entries: {symbol, first_date, last_date, years_span, meets_20y, listing_age_years?, ipo_adjusted_meets?}.
    - When sample is False, updates Prometheus gauges for ratios and totals.
    """
    table_name = os.getenv('EQUITY_DAILY_TABLE', 'market_data')
    host = os.getenv('QUESTDB_HOST', 'trading-questdb')
    http_port = os.getenv('QUESTDB_HTTP_PORT', '9000')
    url = f"http://{host}:{http_port}/exec"
    query = (
        f"select symbol, min(timestamp) first_ts, max(timestamp) last_ts "
        f"from {table_name} where timestamp is not null group by symbol"
    )
    if sample:
        query += f" limit {max(1, min(limit, 5000))}"
    import aiohttp
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=25)) as session:
            async with session.get(url, params={'query': query, 'limit': 'max'}) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    raise HTTPException(status_code=502, detail=f"QuestDB HTTP {resp.status}: {txt[:160]}")
                data = await resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Equities coverage query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        dataset = data.get('dataset', [])
        # Parse results
        exclusions = _load_equity_exclusions()
        out: list[dict] = []
        adjusted_meets = 0
        for row in dataset:
            try:
                sym = str(row[0]).upper()
            except Exception:
                continue
            if sym in exclusions:
                continue
            first_raw = row[1] if len(row) > 1 else None
            last_raw = row[2] if len(row) > 2 else None
            try:
                first_dt = datetime.fromisoformat(str(first_raw).replace('Z','+00:00')) if first_raw else None
                last_dt = datetime.fromisoformat(str(last_raw).replace('Z','+00:00')) if last_raw else None
            except Exception:
                continue
            if not (first_dt and last_dt):
                continue
            years_span = (last_dt - first_dt).days / 365.25
            meets_20y = years_span >= 19.5
            listing_dt = await _get_listing_date(sym)
            listing_age_years = None
            ipo_adjusted_meets = None
            if include_adjusted:
                if listing_dt and listing_dt < last_dt:
                    listing_age_years = (last_dt - listing_dt).days / 365.25
                else:
                    # Fallback: treat earliest bar as listing date (may overestimate if data missing)
                    listing_age_years = years_span
                # IPO-adjusted rule: if listing age < 20y then treat as satisfied if coverage spans >= 90% of listing age
                # plus always count genuine 20y span.
                if listing_age_years < 19.5:
                    ipo_adjusted_meets = (years_span / max(listing_age_years, 0.01)) >= 0.9
                else:
                    ipo_adjusted_meets = meets_20y
                if ipo_adjusted_meets:
                    adjusted_meets += 1
            out.append({
                'symbol': sym,
                'first_date': first_dt.date().isoformat(),
                'last_date': last_dt.date().isoformat(),
                'years_span': round(years_span, 2),
                'meets_20y': meets_20y,
                'listing_age_years': round(listing_age_years, 2) if listing_age_years is not None else None,
                'ipo_adjusted_meets': ipo_adjusted_meets,
            })

        coverage_ratio = (sum(1 for r in out if r['meets_20y']) / len(out)) if out else 0.0
        adjusted_ratio = (adjusted_meets / len(out) if len(out) > 0 else None) if include_adjusted else None

        # Update gauges only on full scan
        if not sample:
            try:
                if EQUITIES_COVERAGE_RATIO_20Y is not None:
                    EQUITIES_COVERAGE_RATIO_20Y.set(coverage_ratio)
                if EQUITIES_COVERAGE_SYMBOLS_TOTAL is not None:
                    EQUITIES_COVERAGE_SYMBOLS_TOTAL.set(len(out))
                if EQUITIES_COVERAGE_SYMBOLS_20Y is not None:
                    EQUITIES_COVERAGE_SYMBOLS_20Y.set(sum(1 for r in out if r['meets_20y']))
            except Exception:
                pass

        return {
            'status': 'success',
            'symbols_evaluated': len(out),
            'coverage_20y_ratio': round(coverage_ratio, 3),
            'coverage_ipo_adjusted_ratio': round(adjusted_ratio, 3) if adjusted_ratio is not None else None,
            'table': table_name,
            'sampled': sample,
            'data': out[:limit]
        }
    except Exception as e:
        logger.error(f"Failed to process equities coverage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/coverage/equities/remediate")
async def equities_coverage_remediate(
    target_years: int = 20,
    max_symbols: int = 25,
    dry_run: bool = True,
    pacing_seconds: float = 0.15,
    min_mature_years: float = 7.0,
):
    """Attempt remediation (historical backfill) for deficient mature symbols.

    Strategy:
      * Full scan (sample=false) to obtain earliest bar per symbol (exclusions respected).
      * Identify symbols with years_span < target_years - 0.5 AND listing_age_years >= min_mature_years.
      * For each, compute missing range start = (last_ts - target_years years) and ingest missing early window
        prior to current earliest date (bounded by 25y safety).
      * Dry-run mode reports planned actions without ingestion.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    # Acquire coverage (full scan) by calling internal logic (reuse query code rather than HTTP self-call for efficiency)
    request = await equities_coverage_full(sample=False, limit=10**9, include_adjusted=True)  # reuse function
    data = request['data']
    now = datetime.utcnow()
    # Define target earliest date
    target_delta_days = int(target_years * 365.25)
    plan: list[dict] = []
    exclusions = _load_equity_exclusions()
    for entry in data:
        sym = entry['symbol']
        if sym in exclusions:
            continue
        years_span = entry['years_span']
        listing_age = entry.get('listing_age_years') or years_span
        if years_span >= target_years - 0.5:
            continue  # already adequate
        if listing_age < min_mature_years:
            continue  # young listing, skip remediation
        earliest = datetime.strptime(entry['first_date'], '%Y-%m-%d')
        latest = datetime.strptime(entry['last_date'], '%Y-%m-%d')
        target_start = latest - timedelta(days=target_delta_days)
        # If earliest <= target_start -> missing data *might* be small or none; skip
        if earliest <= target_start + timedelta(days=30):  # tolerance 30d
            continue
        fetch_start = target_start
        fetch_end = earliest - timedelta(days=1)
        plan.append({
            'symbol': sym,
            'missing_days_estimate': (earliest - fetch_start).days,
            'fetch_start': fetch_start.date().isoformat(),
            'fetch_end': fetch_end.date().isoformat(),
        })
        if len(plan) >= max_symbols:
            break
    performed = []
    total_bars = 0
    if not dry_run:
        for item in plan:
            sym = item['symbol']
            try:
                start_dt = datetime.strptime(item['fetch_start'], '%Y-%m-%d')
                end_dt = datetime.strptime(item['fetch_end'], '%Y-%m-%d')
                rows = await market_data_svc.get_bulk_daily_historical(sym, start_dt, end_dt)
                bars = len(rows)
                total_bars += bars
                performed.append({'symbol': sym, 'bars_ingested': bars})
                if EQUITIES_REMEDIATED_SYMBOLS:
                    EQUITIES_REMEDIATED_SYMBOLS.inc()
                if EQUITIES_REMEDIATED_BARS:
                    EQUITIES_REMEDIATED_BARS.inc(bars)
                await asyncio.sleep(pacing_seconds)
            except Exception as e:  # noqa: BLE001
                performed.append({'symbol': sym, 'error': str(e)})
                if EQUITIES_REMEDIATION_RUNS:
                    try:
                        EQUITIES_REMEDIATION_RUNS.labels(result='error').inc()
                    except Exception:
                        pass
        if EQUITIES_REMEDIATION_RUNS:
            try:
                EQUITIES_REMEDIATION_RUNS.labels(result='success').inc()
            except Exception:
                pass
    else:
        if EQUITIES_REMEDIATION_RUNS:
            try:
                EQUITIES_REMEDIATION_RUNS.labels(result='dry_run').inc()
            except Exception:
                pass
    return {
        'status': 'dry_run' if dry_run else 'executed',
        'generated_at': datetime.utcnow().isoformat(),
        'target_years': target_years,
        'symbols_considered': len(data),
        'plan_count': len(plan),
        'executed': performed if not dry_run else [],
        'plan': plan,
        'total_bars_ingested': total_bars if not dry_run else 0,
        'exclusions': sorted(list(exclusions)),
    }

@app.post("/coverage/equities/export")
async def equities_coverage_export(limit: int = 5000, include_adjusted: bool = True):
    """Compute equities coverage and write a stable JSON artifact used by Grafana."""
    try:
        report = await equities_coverage_full(limit=limit, sample=False, include_adjusted=include_adjusted)
        out_dir = os.getenv("EQUITIES_COVERAGE_OUTPUT_DIR", "/app/export/grafana-csv").rstrip("/")
        os.makedirs(out_dir, exist_ok=True)
        import json as _json
        stable = os.path.join(out_dir, 'equities_coverage.json')
        dated = os.path.join(out_dir, f"equities_coverage_{datetime.utcnow().strftime('%Y%m%d')}.json")
        with open(stable, 'w') as f:
            _json.dump(report, f, indent=2)
        with open(dated, 'w') as f:
            _json.dump(report, f, indent=2)
        logger.info("Equities coverage report written", path=stable, symbols=report.get('symbols_evaluated'))
        return {"status": "success", "path": stable, "symbols": report.get('symbols_evaluated', 0)}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Equities coverage export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coverage/options/export")
async def export_options_coverage(underlyings: Optional[List[str]] = Query(default=None)):
    """Compute options coverage summary and write a stable JSON artifact (mirrors scheduled loop)."""
    import aiohttp
    host = os.getenv('QUESTDB_HOST', 'trading-questdb')
    http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
    qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
    out_dir = os.getenv("OPTIONS_COVERAGE_OUTPUT_DIR", "/app/export/grafana-csv").rstrip("/")

    async def _q(session: aiohttp.ClientSession, sql: str) -> dict:
        async with session.get(qdb_url, params={"query": sql}) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"QuestDB HTTP {resp.status}: {txt[:160]}")
            return await resp.json()

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            # resolve underlyings
            syms: List[str] = []
            if underlyings:
                syms = [s.strip().upper() for s in underlyings.split(',') if s.strip()]
            elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                try:
                    syms = (await reference_svc.get_watchlist_symbols()) or []
                except Exception:
                    syms = []
            if not syms:
                syms = ['AAPL','MSFT','TSLA','NVDA','SPY']
            syms = syms[:max(1, max_underlyings)]

            # compute coverage
            out = []
            for u in syms:
                try:
                    sql_summary = (
                        "select count() as rows, count_distinct(option_symbol) as contracts, "
                        "min(timestamp) as first_ts, max(timestamp) as last_ts "
                        f"from options_data where underlying = '{u}'"
                    )
                    data = await _q(session, sql_summary)
                    if not data.get('dataset'):
                        out.append({"underlying": u, "rows": 0, "contracts": 0, "first_day": None, "last_day": None, "recent_gap_days_30d": None})
                        continue
                    r = data['dataset'][0]
                    cols = {c['name']: i for i, c in enumerate(data.get('columns', []))}
                    rows = int(r[cols['rows']]) if 'rows' in cols else 0
                    contracts = int(r[cols['contracts']]) if 'contracts' in cols else 0
                    # Timestamps are ISO strings; format to YYYY-MM-DD
                    def _fmt_iso_day(v):
                        try:
                            return str(v)[:10]
                        except Exception:
                            return None
                    first_day = _fmt_iso_day(r[cols['first_ts']]) if 'first_ts' in cols else None
                    last_day = _fmt_iso_day(r[cols['last_ts']]) if 'last_ts' in cols else None
                    sql_recent = (
                        "select count_distinct(cast(timestamp as LONG)/86400000000) as have_days "
                        f"from options_data where underlying = '{u}' and timestamp >= dateadd('d', -30, now())"
                    )
                    d2 = await _q(session, sql_recent)
                    have_days = 0
                    if d2.get('dataset'):
                        c2 = {c['name']: i for i, c in enumerate(d2.get('columns', []))}
                        try:
                            have_days = int(d2['dataset'][0][c2['have_days']])
                        except Exception:
                            have_days = 0
                    out.append({
                        "underlying": u,
                        "rows": rows,
                        "contracts": contracts,
                        "first_day": first_day,
                        "last_day": last_day,
                        "recent_gap_days_30d": max(0, 30 - have_days),
                    })
                except Exception as e:  # noqa: BLE001
                    out.append({"underlying": u, "error": str(e)})

            # write JSON artifact
            os.makedirs(out_dir, exist_ok=True)
            payload = {"generated_at": datetime.utcnow().isoformat(), "questdb": qdb_url, "coverage": out}
            import json as _json
            stable_path = os.path.join(out_dir, 'options_coverage.json')
            date_path = os.path.join(out_dir, f"options_coverage_{datetime.utcnow().strftime('%Y%m%d')}.json")
            with open(stable_path, 'w') as f:
                _json.dump(payload, f, indent=2)
            with open(date_path, 'w') as f:
                _json.dump(payload, f, indent=2)
            logger.info("Options coverage report written", path=stable_path, items=len(out))
        return {"status": "success", "path": stable_path, "items": len(out)}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Options coverage export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/polygon-s3-keys")
async def debug_polygon_s3_keys(prefix: str | None = Query(None), max_keys: int = Query(50, ge=1, le=1000)):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    try:
        keys = await news_svc.debug_list_polygon_news_keys(prefix=prefix, max_keys=max_keys)
        return {"count": len(keys), "prefix": prefix or "news/", "keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")

@app.get("/debug/polygon-s3-probe")
async def debug_polygon_s3_probe(date: str = Query(..., description="YYYY-MM-DD")):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    try:
        day = datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid date format; expected YYYY-MM-DD")
    result = await news_svc.debug_probe_polygon_news_keys_for_date(day)
    return result

@app.get("/debug/options-precheck")
async def debug_options_precheck():
    svc = market_data_svc
    if not svc:
        raise HTTPException(status_code=503, detail="market_data_service unavailable")
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "enable_options_ingest": bool(getattr(svc, 'enable_options_ingest', False)),
        "enable_questdb_persist": bool(getattr(svc, 'enable_questdb_persist', False)),
        "enable_hist_dry_run": bool(getattr(svc, 'enable_hist_dry_run', False)),
        "questdb_conf": getattr(svc, 'questdb_conf', None),
        "sender_available": bool(getattr(svc, 'questdb_conf', None) and 'Sender' in str(type(getattr(svc, 'questdb_conf', '')))),
        "polygon_key_present": bool(svc.polygon_config.get('api_key') if svc else False),
    }

@app.get("/debug/news-precheck")
async def debug_news_precheck():
    svc = news_svc
    if not svc:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "available": False,
            "reason": "news_service_not_running"
        }
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "available": True,
        "enable_news_questdb_persist": bool(getattr(svc, 'enable_questdb_persist', False)),
        "questdb_conf": getattr(svc, '_qdb_conf', None),
        "ilp_sender_ready": bool(getattr(svc, '_qdb_sender', None) and getattr(svc, '_qdb_conf', None)),
        "news_api_key": bool(svc.news_api_config.get('api_key')),
        "alpha_vantage_key": bool(svc.alpha_vantage_config.get('api_key')),
        "finnhub_key": bool(svc.finnhub_config.get('api_key')),
    }


@app.post("/ingest/market-data")
async def ingest_market_data(symbol: str):
    """Trigger market data ingestion for a symbol."""
    try:
        logger.info(f"Starting market data ingestion for {symbol}")
        
        # For Phase 2, this is a placeholder that would:
        # 1. Connect to Alpaca/Polygon API
        # 2. Fetch real-time data for symbol
        # 3. Store in QuestDB via cache layer
        # 4. Publish to message queue for downstream processing
        
        sample_data = MarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            open=100.0,
            high=102.5,
            low=99.8,
            close=101.2,
            volume=1000000,
            timeframe="1min",
            data_source="sample"
        )
        
        # Store in cache (would be real implementation)
        if cache_client:
            await cache_client.set_market_data(sample_data)
            
        logger.info(f"Successfully ingested market data for {symbol}")
        return {
            "status": "success",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Market data ingestion completed"
        }
        
    except Exception as e:
        logger.error(f"Market data ingestion failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/news")
async def ingest_news(query: str = "SPY TSLA AAPL"):
    """Trigger news ingestion for given query."""
    try:
        logger.info(f"Starting news ingestion for query: {query}")
        
        # Placeholder for Phase 2 - would implement:
        # 1. Connect to NewsAPI/other news sources
        # 2. Fetch relevant financial news
        # 3. Run sentiment analysis
        # 4. Store processed news with sentiment scores
        
        sample_news = NewsItem(
            title="Market Update: Tech Stocks Rally",
            content="Technology stocks showed strong performance...",
            source="Financial Times",
            published_at=datetime.utcnow(),
            url="https://example.com/news/123",
            sentiment_score=0.7,
            relevance_score=0.9,
            symbols=["SPY", "TSLA", "AAPL"]
        )
        
        if cache_client:
            await cache_client.set_news_item(sample_news)
            
        logger.info(f"Successfully ingested news for query: {query}")
        return {
            "status": "success", 
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "News ingestion completed"
        }
        
    except Exception as e:
        logger.error(f"News ingestion failed for query {query}: {e}")
        raise HTTPException(status_code=500, detail=f"News ingestion failed: {str(e)}")


@app.get("/data/recent/{symbol}")
async def get_recent_data(symbol: str, hours: int = 1):
    """Get recent market data for a symbol."""
    try:
        if not cache_client:
            raise HTTPException(status_code=503, detail="Cache service unavailable")
            
        # Placeholder - would query actual cached data
        logger.info(f"Retrieving recent data for {symbol} (last {hours} hours)")
        
        return {
            "symbol": symbol,
            "timeframe": f"last_{hours}_hours",
            "data_points": 0,  # Would return actual data
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"No data available yet - service in development mode"
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Data retrieval failed: {str(e)}")


# New comprehensive endpoints

@app.post("/market-data/quote/{symbol}")
async def get_real_time_quote(symbol: str):
    """Get real-time quote for a symbol."""
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    
    try:
        quote = await market_data_svc.get_real_time_quote(symbol.upper())
        if quote:
            # Validate the data
            if validation_svc:
                validation_results = await validation_svc.validate_market_data(quote)
                has_errors = any(r.severity.value == "error" for r in validation_results)
                
                return {
                    "data": {
                        "symbol": quote.symbol,
                        "timestamp": quote.timestamp.isoformat(),
                        "open": quote.open,
                        "high": quote.high,
                        "low": quote.low,
                        "close": quote.close,
                        "volume": quote.volume,
                        "source": quote.data_source
                    },
                    "validation": {
                        "valid": not has_errors,
                        "issues": len(validation_results),
                        "details": [{"severity": r.severity.value, "message": r.message} for r in validation_results]
                    }
                }
            else:
                return {"data": quote, "validation": {"valid": True, "issues": 0}}
        else:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Artifact Upload to MinIO ---------------------- #

def _load_minio_client():
    """Lazy import and construct MinIO client using shared helper.

    Returns (client, cfg) or (None, None) if unavailable.
    """
    try:
        from shared.storage.minio_storage import get_minio_client, MinIOConfig  # type: ignore
        # Support base64-encoded secret to avoid issues with $ in docker-compose interpolation
        secret_b64 = os.getenv("MINIO_SECRET_KEY_B64", "").strip()
        if secret_b64:
            import base64
            try:
                decoded = base64.b64decode(secret_b64).decode("utf-8")
                secret = decoded
            except Exception:
                secret = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", ""))
        else:
            secret = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", ""))
        cfg = MinIOConfig(
            endpoint=os.getenv("MINIO_ENDPOINT", os.getenv("MINIO_URL", "localhost:9000")),
            access_key=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "")),
            secret_key=secret,
            secure=os.getenv("MINIO_SECURE", "false").lower() in ("1","true","yes"),
            region=os.getenv("MINIO_REGION"),
        )
        client = get_minio_client(cfg)
        return client, cfg
    except Exception:
        return None, None


@app.post("/artifacts/upload")
async def upload_artifacts_to_minio(
    directory: str = Query(default=os.getenv("GRAFANA_EXPORT_DIR", "/app/export/grafana-csv"), description="Directory containing JSON artifacts"),
    bucket: str = Query(default=os.getenv("MINIO_ARTIFACTS_BUCKET", "trading"), description="MinIO bucket name"),
    prefix: str = Query(default=os.getenv("MINIO_ARTIFACTS_PREFIX", "dashboards"), description="Key prefix inside bucket"),
    pattern: str = Query(default="*.json", description="Glob pattern of files to upload"),
):
    """Upload local JSON artifacts (e.g., coverage reports) to MinIO.

    Walks the given directory and uploads matching files to s3://{bucket}/{prefix}/filename.
    Returns a summary with successes and failures.
    """
    client, cfg = _load_minio_client()
    if not client:
        raise HTTPException(status_code=503, detail="MinIO client unavailable or credentials not configured")
    try:
        from shared.storage.minio_storage import ensure_bucket  # type: ignore
    except Exception:
        ensure_bucket = None  # type: ignore

    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    import glob, json
    files = list(dir_path.glob(pattern))
    uploaded: list[dict] = []
    errors: list[dict] = []

    # Ensure bucket exists (best-effort)
    if ensure_bucket:
        try:
            ensure_bucket(bucket, client=client)  # type: ignore
        except Exception as e:  # noqa: BLE001
            # Continue; upload attempts may still create lazily
            logger.warning("ensure_bucket failed", bucket=bucket, error=str(e))

    for f in files:
        try:
            data = f.read_bytes()
            key = f"{prefix.strip('/')}" if prefix else ""
            if key:
                key += "/" + f.name
            else:
                key = f.name
            # Use client directly to avoid extra deps
            from io import BytesIO
            stream = BytesIO(data)
            client.put_object(bucket, key, stream, length=len(data), content_type="application/json")  # type: ignore[attr-defined]
            uploaded.append({"file": str(f), "bucket": bucket, "key": key, "bytes": len(data)})
        except Exception as e:  # noqa: BLE001
            errors.append({"file": str(f), "error": str(e)})

    return {
        "status": "completed",
        "directory": str(dir_path),
        "bucket": bucket,
        "prefix": prefix,
        "uploaded": uploaded,
        "uploaded_count": len(uploaded),
        "errors": errors,
        "error_count": len(errors),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/market-data/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    timeframe: str = "1min",
    hours_back: int = 24,
    limit: int = 1000,
    start: Optional[str] = Query(None, description="Optional start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="Optional end date YYYY-MM-DD"),
):
    """Get historical market data.

    Modes:
      1. Intraday/legacy (default): timeframe != '1d' OR no start supplied.
         Uses hours_back window ending now and underlying provider specific intraday retrieval.
      2. Bulk daily (extended history): timeframe == '1d' AND start supplied (YYYY-MM-DD).
         Optional end (YYYY-MM-DD, default = today). Delegates to bulk daily provider chain
         (EODHD primary) for 20y scale retrieval.

    Backward compatibility: existing callers that only pass timeframe/hours_back continue working.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")

    try:
        symbol_u = symbol.upper()

        # ---------------- Bulk Daily Path (EODHD) ----------------
        if timeframe == '1d' and start:
            try:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
            if end:
                try:
                    end_dt = datetime.strptime(end, "%Y-%m-%d")
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid end format, expected YYYY-MM-DD")
            else:
                # End date defaults to today (UTC date boundary)
                end_dt = datetime.utcnow()

            if start_dt > end_dt:
                raise HTTPException(status_code=400, detail="start must be <= end")

            # Hard guard: limit to 25 years (same as backfill endpoint)
            if (end_dt - start_dt).days > 365 * 25:
                raise HTTPException(status_code=400, detail="Range too large; limit to 25 years")

            rows = await market_data_svc.get_bulk_daily_historical(symbol_u, start_dt, end_dt)

            return {
                "symbol": symbol_u,
                "timeframe": timeframe,
                "mode": "bulk_daily",
                "count": len(rows),
                "start": start_dt.strftime("%Y-%m-%d"),
                "end": end_dt.strftime("%Y-%m-%d"),
                # Best-effort indicator without touching internal attributes; relies on env presence
                "data_source_primary": ("EODHD" if os.getenv("EODHD_API_KEY") else None),
                "data": [
                    {
                        "timestamp": bar.timestamp.isoformat(),
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        # Bulk daily path currently uses EODHD primary provider
                        "source": "eodhd",
                    } for bar in rows
                ],
            }

        # ---------------- Intraday / Legacy Path ----------------
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)

        data = await market_data_svc.get_historical_data(
            symbol_u, timeframe, start_time, end_time, limit
        )

        return {
            "symbol": symbol_u,
            "timeframe": timeframe,
            "mode": "intraday_window",
            "count": len(data),
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "data": [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "source": bar.data_source,
                } for bar in data
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/news/collect")
async def collect_financial_news(
    symbols: Optional[List[str]] = Query(default=None),
    hours_back: int = 1,
    max_articles: int = 50,
    body: Optional[dict] = Body(default=None, description="Optional JSON body: { symbols: string[], hours_back?: number, max_articles?: number }")
):
    """Collect financial news."""
    if not news_svc:
        raise HTTPException(status_code=503, detail="News service not available")
    
    try:
        # Accept either query params or JSON body for convenience
        eff_symbols = symbols
        eff_hours_back = hours_back
        eff_max_articles = max_articles
        try:
            if body and isinstance(body, dict):
                if eff_symbols is None and isinstance(body.get('symbols'), list):
                    eff_symbols = [str(s).upper() for s in body.get('symbols') if isinstance(s, (str,)) and s.strip()]
                if 'hours_back' in body and isinstance(body.get('hours_back'), (int, float)):
                    eff_hours_back = int(body.get('hours_back'))
                if 'max_articles' in body and isinstance(body.get('max_articles'), (int, float)):
                    eff_max_articles = int(body.get('max_articles'))
        except Exception:
            # Ignore malformed body and fall back to query params
            pass

        news_items = await news_svc.collect_financial_news(eff_symbols, eff_hours_back, eff_max_articles)
        
        return {
            "status": "success",
            "symbols": eff_symbols,
            "articles_collected": len(news_items),
            "hours_back": eff_hours_back,
            "timestamp": datetime.utcnow().isoformat(),
            "articles": [
                {
                    "title": item.title,
                    "source": item.source,
                    "published_at": item.published_at.isoformat(),
                    "sentiment_score": item.sentiment_score,
                    "relevance_score": item.relevance_score,
                    "symbols": item.symbols,
                    "url": item.url
                } for item in news_items
            ]
        }
    except Exception as e:
        logger.error(f"Failed to collect news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Historical/Backfill Endpoints ---------------------- #

class NewsBackfillRequest(BaseModel):
    symbols: List[str]
    start: str
    end: str
    batch_days: int | None = Field(default=NEWS_BACKFILL_WINDOW_DAYS)
    max_articles_per_batch: int | None = Field(default=80)


@app.post("/news/backfill")
async def news_backfill(req: NewsBackfillRequest):
    """Backfill news over a historical date range (batched provider calls).

    Delegates to NewsService.collect_financial_news_range which persists/indexes
    according to feature flags (QuestDB, Weaviate, Pulsar).
    """
    if not news_svc:
        raise HTTPException(status_code=503, detail="News service not available")
    try:
        try:
            start_dt = datetime.strptime(req.start, "%Y-%m-%d")
            end_dt = datetime.strptime(req.end, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")
        if start_dt > end_dt:
            raise HTTPException(status_code=400, detail="start must be <= end")
        batch_days = max(1, int(req.batch_days or NEWS_BACKFILL_WINDOW_DAYS))
        max_articles = max(1, int(req.max_articles_per_batch or 80))
        total, batches = await news_svc.collect_financial_news_range(
            [s.strip().upper() for s in (req.symbols or []) if s and s.strip()],
            start_dt,
            end_dt,
            batch_days=batch_days,
            max_articles_per_batch=max_articles,
            backfill_mode=True,
        )
        return {
            "status": "completed",
            "symbols": [s.strip().upper() for s in (req.symbols or []) if s and s.strip()],
            "start": req.start,
            "end": req.end,
            "batch_days": batch_days,
            "max_articles_per_batch": max_articles,
            "articles_collected": int(total),
            "persisted_count": int(sum(int(b.get('persisted', 0) or 0) for b in batches)),
            "batches": batches,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("News backfill failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/market-data/options-chain/{underlying}")
async def backfill_options_chain_endpoint(
    underlying: str,
    start: str = Query(..., description="Historical start date (YYYY-MM-DD) for aggregates window"),
    end: str = Query(..., description="Historical end date (YYYY-MM-DD) inclusive for aggregates window"),
    max_contracts: int = Query(default=OPTIONS_HISTORY_MAX_CONTRACTS, ge=1, le=2000),
    pacing_seconds: float = Query(default=OPTIONS_HISTORY_PACING_SECONDS, ge=0.0, description="Sleep between contracts"),
    expired: bool = Query(default=False, description="Include only expired contracts (advanced)"),
    include_recent_expired: bool = Query(default=True, description="Augment active set with recently expired contracts"),
    recent_expired_days: int = Query(default=7, ge=0, le=30),
    start_expiry: Optional[str] = Query(default=None, description="Optional expiry window start (YYYY-MM-DD)"),
    end_expiry: Optional[str] = Query(default=None, description="Optional expiry window end (YYYY-MM-DD)"),
    persist_questdb: Optional[bool] = Query(default=None, description="Override QuestDB persistence flag for this run"),
    enable_ingest: Optional[bool] = Query(default=None, description="Override ENABLE_OPTIONS_INGEST for this run"),
):
    """Backfill daily option aggregates for a chain of an underlying over a window.

    Persists rows to QuestDB/Postgres when those feature flags are enabled.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    if not getattr(market_data_svc, 'enable_options_ingest', False):
        raise HTTPException(status_code=403, detail="Options ingestion disabled")
    try:
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid expiry format, expected YYYY-MM-DD")
    today_utc = datetime.utcnow().date()
    is_expired_actual = expiry_dt.date() < today_utc
    # Validation: if client asserts expired but it's not, or vice versa, provide clear error/help
    if expired and not is_expired_actual:
        raise HTTPException(status_code=400, detail="Parameter expired=true but expiry date is not in the past")
    # If not marked expired but actually expired, we continue (read-only) and annotate response
    response_notes: list[str] = []
    if (not expired) and is_expired_actual:
        response_notes.append("expiry is in the past; treat as expired contract (informational)")
    if is_expired_actual and include_recent_expired and recent_expired_days > 0:
        # Provide an informational note if contract is within the 'recent expired' window
        if (today_utc - expiry_dt.date()).days <= recent_expired_days:
            response_notes.append(f"contract expired within last {recent_expired_days} days")
    if right.upper() not in ("C","P"):
        raise HTTPException(status_code=400, detail="right must be C or P")
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
    if end:
        try:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end format, expected YYYY-MM-DD")
    else:
        end_dt = datetime.utcnow()
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")
    try:
        rows = await market_data_svc.get_option_daily_aggregates(
            underlying.upper(),
            expiry_dt,
            right.upper(),
            strike,
            start_dt,
            end_dt,
            option_ticker=option_ticker or None,
        )
        # Metrics: single-contract path
        try:
            if OPTIONS_BACKFILL_CONTRACTS:
                OPTIONS_BACKFILL_CONTRACTS.labels(path='single-contract').inc(1)
            if OPTIONS_BACKFILL_BARS:
                OPTIONS_BACKFILL_BARS.labels(path='single-contract').inc(len(rows))
        except Exception:
            pass
        return {
            "underlying": underlying.upper(),
            "expiry": expiry_dt.strftime('%Y-%m-%d'),
            "right": right.upper(),
            "strike": strike,
            "count": len(rows),
            "start": start_dt.strftime('%Y-%m-%d'),
            "end": end_dt.strftime('%Y-%m-%d'),
            "expired_param": expired,
            "expired_actual": is_expired_actual,
            "recent_expired_window_days": recent_expired_days,
            "notes": response_notes,
        }
    except Exception as e:
        try:
            if OPTIONS_BACKFILL_ERRORS:
                OPTIONS_BACKFILL_ERRORS.labels(path='single-contract').inc()
        except Exception:
            pass
        logger.error(f"Options ingestion failed for {underlying} {expiry} {right} {strike}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backfill/options")
async def backfill_options(
    underlyings: Optional[List[str]] = Query(default=None, description="Underlyings to process; defaults to watchlist sample"),
    start_expiry: Optional[str] = Query(default=None, description="Expiry start YYYY-MM-DD (optional)"),
    end_expiry: Optional[str] = Query(default=None, description="Expiry end YYYY-MM-DD (optional)"),
    start: str = Query(..., description="Historical bars start YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="Historical bars end YYYY-MM-DD (default: today)"),
    max_contracts: int = Query(default=OPTIONS_HISTORY_MAX_CONTRACTS, ge=1, le=10000),
    pacing_seconds: float = Query(default=OPTIONS_HISTORY_PACING_SECONDS, ge=0.0, le=5.0),
    limit_underlyings: int = Query(default=100, ge=1, le=2000),
    dry_run: bool = Query(default=False),
    persist_questdb: Optional[bool] = Query(default=None, description="Override QuestDB persistence flag for this run"),
    enable_ingest: Optional[bool] = Query(default=None, description="Override ENABLE_OPTIONS_INGEST for this run"),
):
    """Orchestrate historical options backfill across a set of underlyings.

    Delegates to market_data_svc.backfill_options_chain per underlying with pacing.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    if not getattr(market_data_svc, 'enable_options_ingest', False):
        raise HTTPException(status_code=403, detail="Options ingestion disabled")

    # Resolve underlyings
    eff_und: List[str] = []
    if underlyings:
        eff_und = [u.strip().upper() for u in underlyings if u and u.strip()]
    elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
        try:
            eff_und = (await reference_svc.get_watchlist_symbols()) or []
        except Exception:
            eff_und = []
    if not eff_und:
        eff_und = ['AAPL','MSFT','TSLA','NVDA','SPY']
    eff_und = eff_und[:max(1, limit_underlyings)]

    # Parse dates
    try:
        start_hist = datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start format, expected YYYY-MM-DD")
    end_hist = datetime.utcnow() if not end else datetime.strptime(end, "%Y-%m-%d")
    start_exp_dt = _parse_date_yyyy_mm_dd(start_expiry) if start_expiry else None
    end_exp_dt = _parse_date_yyyy_mm_dd(end_expiry) if end_expiry else None

    if dry_run:
        return {
            "status": "planned",
            "underlyings": eff_und,
            "start": start_hist.date().isoformat(),
            "end": end_hist.date().isoformat(),
            "start_expiry": start_exp_dt.date().isoformat() if start_exp_dt else None,
            "end_expiry": end_exp_dt.date().isoformat() if end_exp_dt else None,
            "max_contracts": max_contracts,
            "pacing_seconds": pacing_seconds,
        }

    totals = {"underlyings": len(eff_und), "contracts_processed": 0, "bars_ingested": 0, "errors": 0}
    started = datetime.utcnow()
    # Temporary flag overrides
    orig_persist = getattr(market_data_svc, 'enable_questdb_persist', None)
    orig_ingest = getattr(market_data_svc, 'enable_options_ingest', None)
    try:
      if persist_questdb is not None:
          try:
              market_data_svc.enable_questdb_persist = bool(persist_questdb)
          except Exception:
              pass
      if enable_ingest is not None:
          try:
              market_data_svc.enable_options_ingest = bool(enable_ingest)
          except Exception:
              pass
      for u in eff_und:
        try:
            summary = await market_data_svc.backfill_options_chain(
                u,
                start_exp_dt,
                end_exp_dt,
                start_date=start_hist,
                end_date=end_hist,
                max_contracts=max_contracts,
                pacing_seconds=pacing_seconds,
            )
            totals["contracts_processed"] += int(summary.get('contracts_processed', 0) or 0)
            totals["bars_ingested"] += int(summary.get('bars_ingested', 0) or 0)
            totals["per_underlying"].append({"underlying": u, **summary})
        except Exception as e:  # noqa: BLE001
            totals["per_underlying"].append({"underlying": u, "error": str(e)})
        await asyncio.sleep(req.pacing_seconds)
    finally:
      try:
          if orig_persist is not None:
              market_data_svc.enable_questdb_persist = orig_persist
      except Exception:
          pass
      try:
          if orig_ingest is not None:
              market_data_svc.enable_options_ingest = orig_ingest
      except Exception:
          pass

    return {
        "status": "completed",
        "started_at": started.isoformat(),
        "ended_at": datetime.utcnow().isoformat(),
        **totals,
        "start": start_hist.date().isoformat(),
        "end": end_hist.date().isoformat(),
    }


@app.get("/coverage/equities_full")
async def equities_coverage_full(limit: int = 500, sample: bool = True, include_adjusted: bool = True, mature_years: float = 7.0):
    """Compute equities historical coverage from QuestDB and optionally update gauges.

    - Reads earliest/latest TIMESTAMP per symbol from table (env EQUITY_DAILY_TABLE, default 'market_data').
    - Returns list entries: {symbol, first_date, last_date, years_span, meets_20y, listing_age_years?, ipo_adjusted_meets?}.
    - When sample is False, updates Prometheus gauges for ratios and totals.
    """
    table_name = os.getenv('EQUITY_DAILY_TABLE', 'market_data')
    host = os.getenv('QUESTDB_HOST', 'trading-questdb')
    http_port = os.getenv('QUESTDB_HTTP_PORT', '9000')
    url = f"http://{host}:{http_port}/exec"
    query = (
        f"select symbol, min(timestamp) first_ts, max(timestamp) last_ts "
        f"from {table_name} where timestamp is not null group by symbol"
    )
    if sample:
        query += f" limit {max(1, min(limit, 5000))}"
    import aiohttp
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=25)) as session:
            async with session.get(url, params={'query': query, 'limit': 'max'}) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    raise HTTPException(status_code=502, detail=f"QuestDB HTTP {resp.status}: {txt[:160]}")
                data = await resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Equities coverage query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        dataset = data.get('dataset', [])
        # Parse results
        exclusions = _load_equity_exclusions()
        out: list[dict] = []
        adjusted_meets = 0
        for row in dataset:
            try:
                sym = str(row[0]).upper()
            except Exception:
                continue
            if sym in exclusions:
                continue
            first_raw = row[1] if len(row) > 1 else None
            last_raw = row[2] if len(row) > 2 else None
            try:
                first_dt = datetime.fromisoformat(str(first_raw).replace('Z','+00:00')) if first_raw else None
                last_dt = datetime.fromisoformat(str(last_raw).replace('Z','+00:00')) if last_raw else None
            except Exception:
                continue
            if not (first_dt and last_dt):
                continue
            years_span = (last_dt - first_dt).days / 365.25
            meets_20y = years_span >= 19.5
            listing_dt = await _get_listing_date(sym)
            listing_age_years = None
            ipo_adjusted_meets = None
            if include_adjusted:
                if listing_dt and listing_dt < last_dt:
                    listing_age_years = (last_dt - listing_dt).days / 365.25
                else:
                    # Fallback: treat earliest bar as listing date (may overestimate if data missing)
                    listing_age_years = years_span
                # IPO-adjusted rule: if listing age < 20y then treat as satisfied if coverage spans >= 90% of listing age
                # plus always count genuine 20y span.
                if listing_age_years < 19.5:
                    ipo_adjusted_meets = (years_span / max(listing_age_years, 0.01)) >= 0.9
                else:
                    ipo_adjusted_meets = meets_20y
                if ipo_adjusted_meets:
                    adjusted_meets += 1
            out.append({
                'symbol': sym,
                'first_date': first_dt.date().isoformat(),
                'last_date': last_dt.date().isoformat(),
                'years_span': round(years_span, 2),
                'meets_20y': meets_20y,
                'listing_age_years': round(listing_age_years, 2) if listing_age_years is not None else None,
                'ipo_adjusted_meets': ipo_adjusted_meets,
            })

        coverage_ratio = (sum(1 for r in out if r['meets_20y']) / len(out)) if out else 0.0
        adjusted_ratio = (adjusted_meets / len(out) if len(out) > 0 else None) if include_adjusted else None

        # Update gauges only on full scan
        if not sample:
            try:
                if EQUITIES_COVERAGE_RATIO_20Y is not None:
                    EQUITIES_COVERAGE_RATIO_20Y.set(coverage_ratio)
                if EQUITIES_COVERAGE_SYMBOLS_TOTAL is not None:
                    EQUITIES_COVERAGE_SYMBOLS_TOTAL.set(len(out))
                if EQUITIES_COVERAGE_SYMBOLS_20Y is not None:
                    EQUITIES_COVERAGE_SYMBOLS_20Y.set(sum(1 for r in out if r['meets_20y']))
            except Exception:
                pass

        return {
            'status': 'success',
            'symbols_evaluated': len(out),
            'coverage_20y_ratio': round(coverage_ratio, 3),
            'coverage_ipo_adjusted_ratio': round(adjusted_ratio, 3) if adjusted_ratio is not None else None,
            'table': table_name,
            'sampled': sample,
            'data': out[:limit]
        }
    except Exception as e:
        logger.error(f"Failed to process equities coverage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/coverage/equities/remediate")
async def equities_coverage_remediate(
    target_years: int = 20,
    max_symbols: int = 25,
    dry_run: bool = True,
    pacing_seconds: float = 0.15,
    min_mature_years: float = 7.0,
):
    """Attempt remediation (historical backfill) for deficient mature symbols.

    Strategy:
      * Full scan (sample=false) to obtain earliest bar per symbol (exclusions respected).
      * Identify symbols with years_span < target_years - 0.5 AND listing_age_years >= min_mature_years.
      * For each, compute missing range start = (last_ts - target_years years) and ingest missing early window
        prior to current earliest date (bounded by 25y safety).
      * Dry-run mode reports planned actions without ingestion.
    """
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    # Acquire coverage (full scan) by calling internal logic (reuse query code rather than HTTP self-call for efficiency)
    request = await equities_coverage_full(sample=False, limit=10**9, include_adjusted=True)  # reuse function
    data = request['data']
    now = datetime.utcnow()
    # Define target earliest date
    target_delta_days = int(target_years * 365.25)
    plan: list[dict] = []
    exclusions = _load_equity_exclusions()
    for entry in data:
        sym = entry['symbol']
        if sym in exclusions:
            continue
        years_span = entry['years_span']
        listing_age = entry.get('listing_age_years') or years_span
        if years_span >= target_years - 0.5:
            continue  # already adequate
        if listing_age < min_mature_years:
            continue  # young listing, skip remediation
        earliest = datetime.strptime(entry['first_date'], '%Y-%m-%d')
        latest = datetime.strptime(entry['last_date'], '%Y-%m-%d')
        target_start = latest - timedelta(days=target_delta_days)
        # If earliest <= target_start -> missing data *might* be small or none; skip
        if earliest <= target_start + timedelta(days=30):  # tolerance 30d
            continue
        fetch_start = target_start
        fetch_end = earliest - timedelta(days=1)
        plan.append({
            'symbol': sym,
            'missing_days_estimate': (earliest - fetch_start).days,
            'fetch_start': fetch_start.date().isoformat(),
            'fetch_end': fetch_end.date().isoformat(),
        })
        if len(plan) >= max_symbols:
            break
    performed = []
    total_bars = 0
    if not dry_run:
        for item in plan:
            sym = item['symbol']
            try:
                start_dt = datetime.strptime(item['fetch_start'], '%Y-%m-%d')
                end_dt = datetime.strptime(item['fetch_end'], '%Y-%m-%d')
                rows = await market_data_svc.get_bulk_daily_historical(sym, start_dt, end_dt)
                bars = len(rows)
                total_bars += bars
                performed.append({'symbol': sym, 'bars_ingested': bars})
                if EQUITIES_REMEDIATED_SYMBOLS:
                    EQUITIES_REMEDIATED_SYMBOLS.inc()
                if EQUITIES_REMEDIATED_BARS:
                    EQUITIES_REMEDIATED_BARS.inc(bars)
                await asyncio.sleep(pacing_seconds)
            except Exception as e:  # noqa: BLE001
                performed.append({'symbol': sym, 'error': str(e)})
                if EQUITIES_REMEDIATION_RUNS:
                    try:
                        EQUITIES_REMEDIATION_RUNS.labels(result='error').inc()
                    except Exception:
                        pass
        if EQUITIES_REMEDIATION_RUNS:
            try:
                EQUITIES_REMEDIATION_RUNS.labels(result='success').inc()
            except Exception:
                pass
    else:
        if EQUITIES_REMEDIATION_RUNS:
            try:
                EQUITIES_REMEDIATION_RUNS.labels(result='dry_run').inc()
            except Exception:
                pass
    return {
        'status': 'dry_run' if dry_run else 'executed',
        'generated_at': datetime.utcnow().isoformat(),
        'target_years': target_years,
        'symbols_considered': len(data),
        'plan_count': len(plan),
        'executed': performed if not dry_run else [],
        'plan': plan,
        'total_bars_ingested': total_bars if not dry_run else 0,
        'exclusions': sorted(list(exclusions)),
    }

@app.post("/coverage/equities/export")
async def equities_coverage_export(limit: int = 5000, include_adjusted: bool = True):
    """Compute equities coverage and write a stable JSON artifact used by Grafana."""
    try:
        report = await equities_coverage_full(limit=limit, sample=False, include_adjusted=include_adjusted)
        out_dir = os.getenv("EQUITIES_COVERAGE_OUTPUT_DIR", "/app/export/grafana-csv").rstrip("/")
        os.makedirs(out_dir, exist_ok=True)
        import json as _json
        stable = os.path.join(out_dir, 'equities_coverage.json')
        dated = os.path.join(out_dir, f"equities_coverage_{datetime.utcnow().strftime('%Y%m%d')}.json")
        with open(stable, 'w') as f:
            _json.dump(report, f, indent=2)
        with open(dated, 'w') as f:
            _json.dump(report, f, indent=2)
        logger.info("Equities coverage report written", path=stable, symbols=report.get('symbols_evaluated'))
        return {"status": "success", "path": stable, "symbols": report.get('symbols_evaluated', 0)}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Equities coverage export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coverage/options/export")
async def export_options_coverage(underlyings: Optional[List[str]] = Query(default=None)):
    """Compute options coverage summary and write a stable JSON artifact (mirrors scheduled loop)."""
    import aiohttp
    host = os.getenv('QUESTDB_HOST', 'trading-questdb')
    http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
    qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
    out_dir = os.getenv("OPTIONS_COVERAGE_OUTPUT_DIR", "/app/export/grafana-csv").rstrip("/")

    async def _q(session: aiohttp.ClientSession, sql: str) -> dict:
        async with session.get(qdb_url, params={"query": sql}) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"QuestDB HTTP {resp.status}: {txt[:160]}")
            return await resp.json()

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            # resolve underlyings
            syms: List[str] = []
            if underlyings:
                syms = [s.strip().upper() for s in underlyings.split(',') if s.strip()]
            elif reference_svc and hasattr(reference_svc, 'get_watchlist_symbols'):
                try:
                    syms = (await reference_svc.get_watchlist_symbols()) or []
                except Exception:
                    syms = []
            if not syms:
                syms = ['AAPL','MSFT','TSLA','NVDA','SPY']
            syms = syms[:max(1, max_underlyings)]

            # compute coverage
            out = []
            for u in syms:
                try:
                    sql_summary = (
                        "select count() as rows, count_distinct(option_symbol) as contracts, "
                        "min(timestamp) as first_ts, max(timestamp) as last_ts "
                        f"from options_data where underlying = '{u}'"
                    )
                    data = await _q(session, sql_summary)
                    if not data.get('dataset'):
                        out.append({"underlying": u, "rows": 0, "contracts": 0, "first_day": None, "last_day": None, "recent_gap_days_30d": None})
                        continue
                    r = data['dataset'][0]
                    cols = {c['name']: i for i, c in enumerate(data.get('columns', []))}
                    rows = int(r[cols['rows']]) if 'rows' in cols else 0
                    contracts = int(r[cols['contracts']]) if 'contracts' in cols else 0
                    # Timestamps are ISO strings; format to YYYY-MM-DD
                    def _fmt_iso_day(v):
                        try:
                            return str(v)[:10]
                        except Exception:
                            return None
                    first_day = _fmt_iso_day(r[cols['first_ts']]) if 'first_ts' in cols else None
                    last_day = _fmt_iso_day(r[cols['last_ts']]) if 'last_ts' in cols else None
                    sql_recent = (
                        "select count_distinct(cast(timestamp as LONG)/86400000000) as have_days "
                        f"from options_data where underlying = '{u}' and timestamp >= dateadd('d', -30, now())"
                    )
                    d2 = await _q(session, sql_recent)
                    have_days = 0
                    if d2.get('dataset'):
                        c2 = {c['name']: i for i, c in enumerate(d2.get('columns', []))}
                        try:
                            have_days = int(d2['dataset'][0][c2['have_days']])
                        except Exception:
                            have_days = 0
                    out.append({
                        "underlying": u,
                        "rows": rows,
                        "contracts": contracts,
                        "first_day": first_day,
                        "last_day": last_day,
                        "recent_gap_days_30d": max(0, 30 - have_days),
                    })
                except Exception as e:  # noqa: BLE001
                    out.append({"underlying": u, "error": str(e)})

            # write JSON artifact
            os.makedirs(out_dir, exist_ok=True)
            payload = {"generated_at": datetime.utcnow().isoformat(), "questdb": qdb_url, "coverage": out}
            import json as _json
            stable_path = os.path.join(out_dir, 'options_coverage.json')
            date_path = os.path.join(out_dir, f"options_coverage_{datetime.utcnow().strftime('%Y%m%d')}.json")
            with open(stable_path, 'w') as f:
                _json.dump(payload, f, indent=2)
            with open(date_path, 'w') as f:
                _json.dump(payload, f, indent=2)
            logger.info("Options coverage report written", path=stable_path, items=len(out))
        return {"status": "success", "path": stable_path, "items": len(out)}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Options coverage export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/polygon-s3-keys")
async def debug_polygon_s3_keys(prefix: str | None = Query(None), max_keys: int = Query(50, ge=1, le=1000)):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    try:
        keys = await news_svc.debug_list_polygon_news_keys(prefix=prefix, max_keys=max_keys)
        return {"count": len(keys), "prefix": prefix or "news/", "keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")

@app.get("/debug/polygon-s3-probe")
async def debug_polygon_s3_probe(date: str = Query(..., description="YYYY-MM-DD")):
    if not news_svc:
        raise HTTPException(status_code=503, detail="news_service not initialized")
    try:
        day = datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid date format; expected YYYY-MM-DD")
    result = await news_svc.debug_probe_polygon_news_keys_for_date(day)
    return result

@app.get("/debug/options-precheck")
async def debug_options_precheck():
    svc = market_data_svc
    if not svc:
        raise HTTPException(status_code=503, detail="market_data_service unavailable")
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "enable_options_ingest": bool(getattr(svc, 'enable_options_ingest', False)),
        "enable_questdb_persist": bool(getattr(svc, 'enable_questdb_persist', False)),
        "enable_hist_dry_run": bool(getattr(svc, 'enable_hist_dry_run', False)),
        "questdb_conf": getattr(svc, 'questdb_conf', None),
        "sender_available": bool(getattr(svc, 'questdb_conf', None) and 'Sender' in str(type(getattr(svc, 'questdb_conf', '')))),
        "polygon_key_present": bool(svc.polygon_config.get('api_key') if svc else False),
    }

@app.get("/debug/news-precheck")
async def debug_news_precheck():
    svc = news_svc
    if not svc:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "available": False,
            "reason": "news_service_not_running"
        }
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "available": True,
        "enable_news_questdb_persist": bool(getattr(svc, 'enable_questdb_persist', False)),
        "questdb_conf": getattr(svc, '_qdb_conf', None),
        "ilp_sender_ready": bool(getattr(svc, '_qdb_sender', None) and getattr(svc, '_qdb_conf', None)),
        "news_api_key": bool(svc.news_api_config.get('api_key')),
        "alpha_vantage_key": bool(svc.alpha_vantage_config.get('api_key')),
        "finnhub_key": bool(svc.finnhub_config.get('api_key')),
    }


@app.post("/ingest/market-data")
async def ingest_market_data(symbol: str):
    """Trigger market data ingestion for a symbol."""
    try:
        logger.info(f"Starting market data ingestion for {symbol}")
        
        # For Phase 2, this is a placeholder that would:
        # 1. Connect to Alpaca/Polygon API
        # 2. Fetch real-time data for symbol
        # 3. Store in QuestDB via cache layer
        # 4. Publish to message queue for downstream processing
        
        sample_data = MarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            open=100.0,
            high=102.5,
            low=99.8,
            close=101.2,
            volume=1000000,
            timeframe="1min",
            data_source="sample"
        )
        
        # Store in cache (would be real implementation)
        if cache_client:
            await cache_client.set_market_data(sample_data)
            
        logger.info(f"Successfully ingested market data for {symbol}")
        return {
            "status": "success",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Market data ingestion completed"
        }
        
    except Exception as e:
        logger.error(f"Market data ingestion failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/news")
async def ingest_news(query: str = "SPY TSLA AAPL"):
    """Trigger news ingestion for given query."""
    try:
        logger.info(f"Starting news ingestion for query: {query}")
        
        # Placeholder for Phase 2 - would implement:
        # 1. Connect to NewsAPI/other news sources
        # 2. Fetch relevant financial news
        # 3. Run sentiment analysis
        # 4. Store processed news with sentiment scores
        
        sample_news = NewsItem(
            title="Market Update: Tech Stocks Rally",
            content="Technology stocks showed strong performance...",
            source="Financial Times",
            published_at=datetime.utcnow(),
            url="https://example.com/news/123",
            sentiment_score=0.7,
            relevance_score=0.9,
            symbols=["SPY", "TSLA", "AAPL"]
        )
        
        if cache_client:
            await cache_client.set_news_item(sample_news)
            
        logger.info(f"Successfully ingested news for query: {query}")
        return {
            "status": "success", 
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "News ingestion completed"
        }
        
    except Exception as e:
        logger.error(f"News ingestion failed for query {query}: {e}")
        raise HTTPException(status_code=500, detail=f"News ingestion failed: {str(e)}")


@app.get("/data/recent/{symbol}")
async def get_recent_data(symbol: str, hours: int = 1):
    """Get recent market data for a symbol."""
    try:
        if not cache_client:
            raise HTTPException(status_code=503, detail="Cache service unavailable")
            
        # Placeholder - would query actual cached data
        logger.info(f"Retrieving recent data for {symbol} (last {hours} hours)")
        
        return {
            "symbol": symbol,
            "timeframe": f"last_{hours}_hours",
            "data_points": 0,  # Would return actual data
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"No data available yet - service in development mode"
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Data retrieval failed: {str(e)}")


# New comprehensive endpoints

@app.post("/market-data/quote/{symbol}")
async def get_real_time_quote(symbol: str):
    """Get real-time quote for a symbol."""
    if not market_data_svc:
        raise HTTPException(status_code=503, detail="Market data service not available")
    
    try:
        quote = await market_data_svc.get_real_time_quote(symbol.upper())
        if quote:
            # Validate the data
            if validation_svc:
                validation_results = await validation_svc.validate_market_data(quote)
                has_errors = any(r.severity.value == "error" for r in validation_results)
                
                return {
                    "data": {
                        "symbol": quote.symbol,
                        "timestamp": quote.timestamp.isoformat(),
                        "open": quote.open,
                        "high": quote.high,
                        "low": quote.low,
                        "close": quote.close,
                        "volume": quote.volume,
                        "source": quote.data_source
                    },
                    "validation": {
                        "valid": not has_errors,
                        "issues": len(validation_results),
                        "details": [{"severity": r.severity.value, "message": r.message} for r in validation_results]
                    }
                }
            else:
                return {"data": quote, "validation": {"valid": True, "issues": 0}}
        else:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Artifact
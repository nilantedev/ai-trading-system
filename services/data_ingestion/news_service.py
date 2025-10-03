#!/usr/bin/env python3
"""News Service - Financial news and sentiment data collection."""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime, timedelta
from dataclasses import asdict
import os
import re
from urllib.parse import quote

from trading_common import NewsItem, SocialSentiment, get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_pulsar_client
from prometheus_client import Counter, Gauge
from trading_common.ai_models import generate_response, ModelType
from enum import Enum
import time
from contextlib import asynccontextmanager
from dateutil import parser as date_parser

# Minimal internal resilience helpers (to avoid missing external dependency)
class BreakerState(Enum):
    CLOSED = 'closed'
    HALF_OPEN = 'half_open'
    OPEN = 'open'


class AsyncCircuitBreaker:
    """Lightweight async circuit breaker with context manager semantics."""
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 30.0, success_threshold: int = 1):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self._state = BreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_ts: Optional[float] = None
        self._listeners: list = []

    def add_state_listener(self, cb):
        self._listeners.append(cb)

    def _transition(self, new_state: BreakerState):
        old = self._state
        if old is new_state:
            return
        self._state = new_state
        for fn in list(self._listeners):
            try:
                fn(old, new_state)
            except Exception:  # noqa: BLE001
                continue

    def get_state(self):
        return {
            'state': self._state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
        }

    @asynccontextmanager
    async def context(self):
        now = time.monotonic()
        if self._state is BreakerState.OPEN:
            if self._last_failure_ts is None or (now - self._last_failure_ts) >= self.recovery_timeout:
                self._transition(BreakerState.HALF_OPEN)
                self._success_count = 0
            else:
                raise RuntimeError(f"Circuit breaker {self.name} is OPEN")
        try:
            yield
        except Exception:
            self._failure_count += 1
            self._last_failure_ts = time.monotonic()
            if self._state is BreakerState.HALF_OPEN:
                self._transition(BreakerState.OPEN)
                self._failure_count = 0
                self._success_count = 0
            elif self._state is BreakerState.CLOSED and self._failure_count >= self.failure_threshold:
                self._transition(BreakerState.OPEN)
                self._failure_count = 0
            raise
        else:
            if self._state is BreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition(BreakerState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state is BreakerState.CLOSED:
                self._failure_count = 0


class AdaptiveTokenBucket:
    """Simple adaptive token bucket for send rate control."""
    def __init__(self, rate: float, capacity: float):
        self._rate = float(rate)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()

    async def acquire(self, wait: bool = False) -> bool:
        now = time.monotonic()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        if wait:
            deficit = 1.0 - self._tokens
            await asyncio.sleep(deficit / max(self._rate, 1e-6))
            self._tokens = max(0.0, self._tokens - 1.0)
            return True
        return False

    def record_result(self, success: bool, latency: float):
        if success:
            self._rate = min(self._capacity, self._rate * (1.02 if latency < 0.2 else 1.0))
        else:
            self._rate = max(1.0, self._rate * 0.85)

    def metrics_snapshot(self):
        return {'rate': self._rate, 'tokens': self._tokens}

# Import provider metrics helper & decorator
try:
    from services.data_ingestion.provider_metrics import (
        record_provider_request,
        provider_instrumentation,
        record_http_response,
        record_rate_limit,
    )
except Exception:  # noqa: BLE001
    def record_provider_request(*args, **kwargs):  # fallback no-op
        pass
    def provider_instrumentation(provider, endpoint):
        def _d(f):
            return f
        return _d
    def record_http_response(*args, **kwargs):
        pass
    def record_rate_limit(*args, **kwargs):
        pass

logger = get_logger(__name__)
settings = get_settings()


class NewsService:
    """Handles financial news ingestion and sentiment analysis."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = None
        self.pulsar_client = None
        self.producer = None
        self._pulsar_error_tokens = 15
        self._pulsar_error_last_refill = datetime.utcnow()
        try:
            self._pulsar_error_counter = Counter('pulsar_persistence_errors_total', 'Total Pulsar persistence or producer send errors')
        except Exception:  # noqa: BLE001
            self._pulsar_error_counter = None
        # Resilience primitives for Pulsar producer
        self._pulsar_breaker = AsyncCircuitBreaker('pulsar_producer')
        self._pulsar_bucket = AdaptiveTokenBucket(rate=30, capacity=90)  # News volume lower than market data
        # Metrics (lazy creation to avoid duplicate registration)
        try:
            self._cb_state = Counter  # sentinel to avoid linter unused warnings
            from prometheus_client import Gauge, Counter as PCounter
            try:
                self._cb_transitions = PCounter('circuit_breaker_transitions_total','Circuit breaker transitions',['service','breaker','from_state','to_state'])
            except Exception:
                self._cb_transitions = None
            try:
                self._cb_state_gauge = Gauge('circuit_breaker_state','Circuit breaker state (0=closed,1=half-open,2=open)',['service','breaker'])
            except Exception:
                self._cb_state_gauge = None
            try:
                self._bucket_rate_g = Gauge('adaptive_token_bucket_rate','Current adaptive token bucket rate',['service','bucket'])
                self._bucket_tokens_g = Gauge('adaptive_token_bucket_tokens','Current available tokens in bucket',['service','bucket'])
                self._bucket_rate_g.labels(service='news', bucket='pulsar_producer').set(30)
                self._bucket_tokens_g.labels(service='news', bucket='pulsar_producer').set(90)
            except Exception:
                self._bucket_rate_g = None
                self._bucket_tokens_g = None
        except Exception:
            self._cb_transitions = None
            self._cb_state_gauge = None
            self._bucket_rate_g = None
            self._bucket_tokens_g = None
        # Breaker state listener
        def _listener(old, new):
            mapping = {BreakerState.CLOSED:0, BreakerState.HALF_OPEN:1, BreakerState.OPEN:2}
            if self._cb_state_gauge:
                try:
                    self._cb_state_gauge.labels(service='news', breaker='pulsar_producer').set(mapping.get(new,0))
                except Exception:
                    pass
            if self._cb_transitions:
                try:
                    self._cb_transitions.labels(service='news', breaker='pulsar_producer', from_state=old.value, to_state=new.value).inc()
                except Exception:
                    pass
        try:
            self._pulsar_breaker.add_state_listener(_listener)
        except Exception:
            pass
        # Backpressure skip counter
        try:
            from prometheus_client import Counter as _PC  # type: ignore
            try:
                self._publish_skipped = _PC('producer_publish_skipped_total','Messages skipped before publish due to backpressure',['service','reason'])
            except Exception:
                self._publish_skipped = None
        except Exception:
            self._publish_skipped = None
        
        # API configurations
        self.news_api_config = {
            'api_key': os.getenv('NEWS_API_KEY'),
            'base_url': 'https://newsapi.org/v2'
        }
        
        self.finnhub_config = {
            'api_key': os.getenv('FINNHUB_API_KEY'),
            'base_url': 'https://finnhub.io/api/v1'
        }
        # Polygon.io News (good historical coverage with date filters)
        self.polygon_config = {
            'api_key': os.getenv('POLYGON_API_KEY'),
            'base_url': 'https://api.polygon.io'
        }
        # Alpha Vantage news sentiment API (historical capable)
        self.alpha_vantage_config = {
            'api_key': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'base_url': 'https://www.alphavantage.co'
        }
        # EODHD Financial News API (preferred over NewsAPI)
        self.eodhd_config = {
            'api_key': os.getenv('EODHD_API_KEY'),
            'base_url': os.getenv('EODHD_BASE_URL', 'https://eodhd.com/api'),
            'default_exchange': os.getenv('EODHD_DEFAULT_EXCHANGE', 'US'),
            'max_limit': int(os.getenv('EODHD_NEWS_MAX_LIMIT', '1000')),
        }
        # Polygon Flat Files (S3) access for historical news (bulk, cost-effective)
        self.polygon_flatfiles = {
            'endpoint_url': os.getenv('POLYGON_S3_ENDPOINT', 'https://files.polygon.io'),
            'bucket': os.getenv('POLYGON_S3_BUCKET', 'flatfiles'),
            'access_key': os.getenv('POLYGON_S3_ACCESS_KEY_ID'),
            'secret_key': os.getenv('POLYGON_S3_SECRET_ACCESS_KEY'),
            'region': os.getenv('POLYGON_S3_REGION', 'us-east-1'),
        }
        
        self.reddit_config = {
            'client_id': os.getenv('REDDIT_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'user_agent': 'AI-Trading-System/1.0'
        }

        # Provider gating and priorities (env-driven)
        # Defaults: prefer paid/premium providers first to reduce 429s: Polygon (API + Flat Files), Alpha Vantage, Finnhub (free-tier limited), NewsAPI, GDELT
        def _flag(name: str, default: str) -> bool:
            return os.getenv(name, default).strip().lower() in ("1","true","yes","on")
        # Deprioritize NewsAPI by default; keep disabled unless explicitly enabled
        self.enable_newsapi = _flag('NEWSAPI_ENABLED', 'false') and bool(self.news_api_config['api_key'])
        self.enable_finnhub = _flag('FINNHUB_NEWS_ENABLED', 'true') and bool(self.finnhub_config['api_key'])
        self.enable_polygon_news = _flag('POLYGON_NEWS_ENABLED', 'true') and bool(self.polygon_config['api_key'])
        self.enable_polygon_flatfiles = _flag('POLYGON_FLATFILES_ENABLED', 'true') and bool(self.polygon_flatfiles.get('access_key'))
        self.enable_alpha_vantage_news = _flag('ALPHAVANTAGE_NEWS_ENABLED', 'true') and bool(self.alpha_vantage_config['api_key'])
        self.enable_gdelt = _flag('GDELT_ENABLED', 'true')
        self.enable_eodhd_news = _flag('EODHD_NEWS_ENABLED', 'true') and bool(self.eodhd_config['api_key'])
        # Finnhub free-tier caps
        try:
            self.finnhub_max_symbols_per_window = int(os.getenv('FINNHUB_SYMBOLS_PER_WINDOW','5'))
        except Exception:
            self.finnhub_max_symbols_per_window = 5
        try:
            self.finnhub_symbol_sleep = float(os.getenv('FINNHUB_SYMBOL_PACING_SECONDS','0.25'))
        except Exception:
            self.finnhub_symbol_sleep = 0.25
        
        # Financial keywords for filtering
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'dividend', 'merger', 'acquisition',
            'IPO', 'stock', 'shares', 'market', 'trading', 'investor', 'analyst',
            'upgrade', 'downgrade', 'rating', 'SEC', 'FDA', 'federal reserve',
            'inflation', 'interest rate', 'GDP', 'unemployment', 'CPI'
        ]
        # Persistence flags
        self.enable_questdb_persist = os.getenv('ENABLE_QUESTDB_NEWS_PERSIST', 'false').lower() in ('1','true','yes')
        self.enable_weaviate_persist = os.getenv('ENABLE_WEAVIATE_PERSIST', 'false').lower() in ('1','true','yes')
        # Postgres sink can be toggled at runtime as well as via env
        self.enable_postgres_persist = os.getenv('ENABLE_POSTGRES_NEWS_PERSIST','false').lower() in ('1','true','yes')
        self.ml_service_url = os.getenv('ML_SERVICE_URL', 'http://trading-ml:8001')
        try:
            from questdb.ingress import Sender, TimestampNanos  # type: ignore
            self._qdb_sender = Sender
            self._qdb_ts = TimestampNanos
        except Exception:  # noqa: BLE001
            self._qdb_sender = None
            self._qdb_ts = None
        self._qdb_conf = None
        # Structured log flag (indentation fix)
        self.enable_structured_logs = os.getenv('ENABLE_STRUCTURED_INGEST_LOGS','false').lower() in ('1','true','yes')
        # ---------------- Persistence Metrics (idempotent) ---------------- #
        try:  # guard duplicate registration
            self._news_rows_total = Counter('news_persist_rows_total','Total news rows persisted (QuestDB + Postgres)',['sink'])
        except Exception:  # noqa: BLE001
            self._news_rows_total = None
        try:
            self._news_persist_failures = Counter('news_persist_failures_total','Total news persistence failures',['path'])
        except Exception:  # noqa: BLE001
            self._news_persist_failures = None
        try:
            self._news_last_persist_ts = Gauge('news_last_persist_timestamp_seconds','Unix timestamp of last successful news persist')
        except Exception:  # noqa: BLE001
            self._news_last_persist_ts = None
        # Value score threshold (used by caller logic / retention heuristics, not enforced here)
        try:
            self._news_value_score_min = float(os.getenv('NEWS_VALUE_SCORE_MIN','0.0'))
        except Exception:
            self._news_value_score_min = 0.0
        # In-memory novelty tracking (bounded) to avoid async redis calls from sync scoring function
        self._seen_title_hashes: list[int] = []  # maintain insertion order; treat as ring buffer
        self._seen_title_max = 5000
        # Ingestion-time filtering threshold (pre-persist) separate from retention pruning.
        try:
            self._news_ingest_value_min = float(os.getenv('NEWS_INGEST_VALUE_MIN','0.0'))
        except Exception:
            self._news_ingest_value_min = 0.0
        # Metrics for ingestion filtering (created lazily to avoid duplicate registration)
        self._ingest_filter_metrics_ready = False
        # Backlog reindex + historical backfill knobs
        try:
            self._news_reindex_batch = int(os.getenv('NEWS_REINDEX_BATCH', '200') or '200')
        except Exception:
            self._news_reindex_batch = 200
        try:
            self._news_backfill_years = int(os.getenv('NEWS_BACKFILL_YEARS', '3') or '3')
        except Exception:
            self._news_backfill_years = 3

    def _rate_limited_pulsar_error(self, msg: str) -> None:
        """Log Pulsar errors with a simple token-bucket rate limiter to avoid log spam."""
        try:
            now = datetime.utcnow()
            # Refill tokens roughly once per 60s window
            if (now - self._pulsar_error_last_refill).total_seconds() >= 60:
                self._pulsar_error_tokens = 15
                self._pulsar_error_last_refill = now
            if self._pulsar_error_tokens > 0:
                self._pulsar_error_tokens -= 1
                logger.warning(msg)
        except Exception:
            # Fallback to direct log on any failure
            try:
                logger.warning(msg)
            except Exception:
                pass
    
    async def collect_financial_news_range(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        *,
        batch_days: int = 14,
        max_articles_per_batch: int = 80,
        backfill_mode: bool = False,
    ) -> tuple[int, list[dict]]:
        """Collect financial news over a date range by batching provider calls.

        Returns a tuple of (total_articles, batch_summaries). Providers without
        historical range support will attempt best-effort using from/to params.
        """
        if not symbols:
            return 0, []
        # Ensure session & cache are initialized
        if self.session is None:
            await self.start()

        total = 0
        # In backfill mode, temporarily relax ingestion threshold
        _ingest_min_backup: Optional[float] = None
        if backfill_mode:
            try:
                _ingest_min_backup = self._news_ingest_value_min
                self._news_ingest_value_min = 0.0
            except Exception:
                _ingest_min_backup = None

        batches: list[dict] = []
        cur = start_date
        while cur <= end_date:
            wnd_end = min(end_date, cur + timedelta(days=max(1, batch_days) - 1))
            collected = 0
            persisted_count = 0
            batch_items: list[NewsItem] = []

            # Prefer high-quota providers with explicit window and relaxed mode in backfills
            try:
                # EODHD provider (PRIMARY, paid)
                if getattr(self, 'enable_eodhd_news', False) and self.eodhd_config.get('api_key') and symbols:
                    try:
                        eodhd_items = await self._collect_from_eodhd_range(symbols, cur, wnd_end, max_articles_per_batch)
                    except Exception:
                        eodhd_items = []
                    for ni in eodhd_items:
                        try:
                            if ni.sentiment_score is None:
                                ni.sentiment_score = await self._analyze_sentiment((ni.title or '') + ' ' + (ni.content or ''))
                        except Exception:
                            try:
                                ni.sentiment_score = 0.0
                            except Exception:
                                pass
                        # Compute value score locally; do not attach undeclared fields to Pydantic model
                        try:
                            _ = self._compute_value_score(ni)
                        except Exception:
                            _ = 0.0
                        await self._publish_news(ni)
                        try:
                            await self._maybe_persist_news(ni)
                            persisted_count += 1
                        except Exception:
                            pass
                        collected += 1
                        total += 1
                        batch_items.append(ni)

                # Polygon Flat Files (S3) bulk news ingestion within window (secondary)
                try:
                    poly_s3_items = await self._collect_from_polygon_flatfiles(symbols, cur, wnd_end, max_articles_per_batch)
                except Exception:
                    poly_s3_items = []
                for ni in poly_s3_items:
                    try:
                        if ni.sentiment_score is None:
                            ni.sentiment_score = await self._analyze_sentiment((ni.title or '') + ' ' + (ni.content or ''))
                    except Exception:
                        try:
                            ni.sentiment_score = 0.0
                        except Exception:
                            pass
                    try:
                        _ = self._compute_value_score(ni)
                    except Exception:
                        _ = 0.0
                    await self._publish_news(ni)
                    try:
                        await self._maybe_persist_news(ni)
                        persisted_count += 1
                    except Exception:
                        pass
                    collected += 1
                    total += 1
                    batch_items.append(ni)

                # Polygon.io news per-symbol fanout within window
                if self.polygon_config.get('api_key') and symbols:
                    try:
                        poly_items = await self._collect_from_polygon_news(symbols, cur, wnd_end, max_articles_per_batch, relaxed=backfill_mode)
                    except Exception:
                        poly_items = []
                    for ni in poly_items:
                        try:
                            if ni.sentiment_score is None:
                                ni.sentiment_score = await self._analyze_sentiment((ni.title or '') + ' ' + (ni.content or ''))
                        except Exception:
                            try:
                                ni.sentiment_score = 0.0
                            except Exception:
                                pass
                        try:
                            _ = self._compute_value_score(ni)
                        except Exception:
                            _ = 0.0
                        await self._publish_news(ni)
                        try:
                            await self._maybe_persist_news(ni)
                            persisted_count += 1
                        except Exception:
                            pass
                        collected += 1
                        total += 1
                        batch_items.append(ni)

                # Alpha Vantage (additional historical coverage)
                if self.alpha_vantage_config.get('api_key'):
                    try:
                        av_items = await self._collect_from_alpha_vantage(symbols, max_articles_per_batch, start_date=cur, end_date=wnd_end)
                    except Exception:
                        av_items = []
                    for ni in av_items:
                        try:
                            if ni.sentiment_score is None:
                                ni.sentiment_score = await self._analyze_sentiment((ni.title or '') + ' ' + (ni.content or ''))
                        except Exception:
                            try:
                                ni.sentiment_score = 0.0
                            except Exception:
                                pass
                        try:
                            _ = self._compute_value_score(ni)
                        except Exception:
                            _ = 0.0
                        await self._publish_news(ni)
                        try:
                            await self._maybe_persist_news(ni)
                            persisted_count += 1
                        except Exception:
                            pass
                        collected += 1
                        total += 1
                        batch_items.append(ni)

                # Finnhub (free-tier fallback)
                if self.finnhub_config.get('api_key') and symbols:
                    try:
                        finnhub_items = await self._collect_from_finnhub_range(symbols, cur, wnd_end, max_articles_per_batch)
                    except Exception:
                        finnhub_items = []
                    for ni in finnhub_items:
                        try:
                            if ni.sentiment_score is None:
                                ni.sentiment_score = await self._analyze_sentiment((ni.title or '') + ' ' + (ni.content or ''))
                        except Exception:
                            try:
                                ni.sentiment_score = 0.0
                            except Exception:
                                pass
                        try:
                            _ = self._compute_value_score(ni)
                        except Exception:
                            _ = 0.0
                        await self._publish_news(ni)
                        try:
                            await self._maybe_persist_news(ni)
                            persisted_count += 1
                        except Exception:
                            pass
                        collected += 1
                        total += 1
                        batch_items.append(ni)

                # NewsAPI only if EODHD disabled/unavailable (opt-in)
                if self.news_api_config.get('api_key') and not getattr(self, 'enable_eodhd_news', False):
                    try:
                        newsapi_items = await self._collect_from_newsapi(
                            symbols,
                            hours_back=1,  # ignored when start/end provided
                            max_articles=max_articles_per_batch,
                            start_date=cur,
                            end_date=wnd_end,
                            relaxed=backfill_mode,
                        )
                    except Exception:
                        newsapi_items = []
                    for ni in newsapi_items:
                        try:
                            if ni.sentiment_score is None:
                                ni.sentiment_score = await self._analyze_sentiment((ni.title or '') + ' ' + (ni.content or ''))
                        except Exception:
                            try:
                                ni.sentiment_score = 0.0
                            except Exception:
                                pass
                        try:
                            setattr(ni, 'value_score', self._compute_value_score(ni))
                        except Exception:
                            setattr(ni, 'value_score', 0.0)
                        await self._publish_news(ni)
                        try:
                            await self._maybe_persist_news(ni)
                            persisted_count += 1
                        except Exception:
                            pass
                        collected += 1
                        total += 1
                        batch_items.append(ni)

                # GDELT as lowest-priority supplemental source
                try:
                    gdelt_items = await self._collect_from_gdelt(symbols, cur, wnd_end, max_articles_per_batch, relaxed=backfill_mode)
                except Exception:
                    gdelt_items = []
                for ni in gdelt_items:
                    try:
                        if ni.sentiment_score is None:
                            ni.sentiment_score = await self._analyze_sentiment((ni.title or '') + ' ' + (ni.content or ''))
                    except Exception:
                        try:
                            ni.sentiment_score = 0.0
                        except Exception:
                            pass
                    try:
                        _ = self._compute_value_score(ni)
                    except Exception:
                        _ = 0.0
                    await self._publish_news(ni)
                    try:
                        await self._maybe_persist_news(ni)
                        persisted_count += 1
                    except Exception:
                        pass
                    collected += 1
                    total += 1
                    batch_items.append(ni)

            except Exception as e:  # noqa: BLE001
                logger.debug("News range batch error: %s", e)

            # After finishing providers for this window, index the batch into Weaviate if enabled
            try:
                if self.enable_weaviate_persist and batch_items:
                    await self._index_news_to_weaviate(batch_items)
            except Exception as e:
                try:
                    logger.warning("Weaviate indexing (range batch) failed: %s", e)
                except Exception:
                    pass

            # If still no items collected for the window, try AV topic-based without symbols
            if collected == 0 and self.alpha_vantage_config.get('api_key'):
                try:
                    av_topic_items = await self._collect_from_alpha_vantage([], max_articles_per_batch, start_date=cur, end_date=wnd_end)
                except Exception:
                    av_topic_items = []
                for ni in av_topic_items:
                    try:
                        if ni.sentiment_score is None:
                            ni.sentiment_score = await self._analyze_sentiment((ni.title or '') + ' ' + (ni.content or ''))
                    except Exception:
                        try:
                            ni.sentiment_score = 0.0
                        except Exception:
                            pass
                    try:
                        _ = self._compute_value_score(ni)
                    except Exception:
                        _ = 0.0
                    # Ensure 'ALL' tag when no symbols provided so downstream counts can see it
                    if not getattr(ni, 'symbols', None):
                        try:
                            ni.symbols = ['ALL']
                        except Exception:
                            pass
                    await self._publish_news(ni)
                    try:
                        await self._maybe_persist_news(ni)
                        persisted_count += 1
                    except Exception:
                        pass
                    collected += 1
                    total += 1
                    batch_items.append(ni)

            batches.append({
                'start': cur.strftime('%Y-%m-%d'),
                'end': wnd_end.strftime('%Y-%m-%d'),
                'articles': collected,
                'persisted': persisted_count
            })

            # Pacing between windows with safe default and env override
            try:
                pacing = float(os.getenv('NEWS_BACKFILL_PACING_SECONDS', '0.2'))
            except Exception:
                pacing = 0.2
            await asyncio.sleep(max(0.0, pacing))
            cur = wnd_end + timedelta(days=1)

        # Restore threshold
        if backfill_mode and _ingest_min_backup is not None:
            try:
                self._news_ingest_value_min = _ingest_min_backup
            except Exception:
                pass

        return total, batches

    # ---------------- EODHD Provider ---------------- #
    def _eodhd_symbol(self, sym: str) -> str:
        s = (sym or '').strip().upper()
        if not s:
            return s
        return s if '.' in s else f"{s}.{self.eodhd_config.get('default_exchange','US').upper()}"

    def _normalize_eodhd_symbols(self, arr: list[str] | None, fallback: str | None = None) -> list[str]:
        out: list[str] = []
        for s in (arr or []):
            try:
                s2 = str(s).strip().upper()
            except Exception:
                continue
            if not s2:
                continue
            # drop .EX suffix for internal uniformity
            out.append(s2.split('.')[0])
        if not out and fallback:
            try:
                out.append(fallback.split('.')[0])
            except Exception:
                pass
        return out

    @provider_instrumentation('eodhd', 'news-range')
    async def _collect_from_eodhd_range(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        max_articles: int,
    ) -> List[NewsItem]:
        if not (self.session and self.eodhd_config.get('api_key')):
            return []
        base = f"{self.eodhd_config['base_url'].rstrip('/')}/news"
        out: List[NewsItem] = []
        syms = symbols[: max(1, min(len(symbols), int(os.getenv('EODHD_MAX_SYMBOLS_PER_WINDOW','25'))))]
        per_sym = max(1, max_articles // max(1, len(syms)))
        for sym in syms:
            mapped = self._eodhd_symbol(sym)
            offset = 0
            collected = 0
            while collected < per_sym and len(out) < max_articles:
                params = {
                    's': mapped,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'limit': str(min(per_sym - collected, self.eodhd_config.get('max_limit', 1000))),
                    'offset': str(offset),
                    'api_token': self.eodhd_config['api_key'],
                    'fmt': 'json',
                }
                try:
                    async with self.session.get(base, params=params, timeout=30) as resp:
                        if resp.status != 200:
                            break
                        data = await resp.json()
                except Exception:
                    break
                items = data if isinstance(data, list) else []
                if not items:
                    break
                for a in items:
                    if len(out) >= max_articles or collected >= per_sym:
                        break
                    try:
                        title = a.get('title') or ''
                        url = a.get('link') or ''
                        content = a.get('content') or ''
                        pub_raw = a.get('date') or a.get('published_at')
                        published = None
                        if pub_raw:
                            try:
                                published = date_parser.parse(str(pub_raw))
                            except Exception:
                                published = None
                        syms_norm = self._normalize_eodhd_symbols(a.get('symbols'), fallback=mapped)
                        sent_obj = a.get('sentiment') or {}
                        sent = None
                        try:
                            if isinstance(sent_obj, dict):
                                if 'polarity' in sent_obj:
                                    sent = float(sent_obj.get('polarity') or 0.0)
                                elif 'pos' in sent_obj or 'neg' in sent_obj:
                                    sent = float(sent_obj.get('pos', 0.0)) - float(sent_obj.get('neg', 0.0))
                        except Exception:
                            sent = None
                        ni = NewsItem(
                            title=title,
                            content=content,
                            source='EODHD',
                            published_at=published or datetime.utcnow(),
                            url=url,
                            sentiment_score=float(sent or 0.0),
                            relevance_score=0.9,
                            symbols=syms_norm or [sym]
                        )
                        out.append(ni)
                        collected += 1
                    except Exception:
                        continue
                offset += len(items)
                await asyncio.sleep(0.1)
        return out

    @provider_instrumentation('eodhd', 'news-recent')
    async def _collect_from_eodhd_recent(
        self,
        symbols: Optional[List[str]],
        hours_back: int,
        max_articles: int,
    ) -> List[NewsItem]:
        if not (self.session and self.eodhd_config.get('api_key') and symbols):
            return []
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(hours=max(1, hours_back))
        return await self._collect_from_eodhd_range(symbols, start_dt, end_dt, max_articles)

    @provider_instrumentation('finnhub', 'company-news-range')
    async def _collect_from_finnhub_range(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        max_articles: int,
    ) -> List[NewsItem]:
        """Collect company news per symbol from Finnhub within a date range.

        Uses /company-news with from/to for each symbol; caps total per batch to avoid free-tier exhaustion.
        """
        if not (self.finnhub_config.get('api_key') and self.session):
            return []
        out: List[NewsItem] = []
        base = f"{self.finnhub_config['base_url'].rstrip('/')}/company-news"
        per_sym = max(1, max_articles // max(1, len(symbols[:10])))
        frm = start_date.strftime('%Y-%m-%d')
        to = end_date.strftime('%Y-%m-%d')
        for sym in symbols[:10]:  # cap to 10 symbols per window for free tier
            params = {
                'symbol': sym,
                'from': frm,
                'to': to,
                'token': self.finnhub_config['api_key']
            }
            try:
                async with self.session.get(base, params=params, timeout=30) as resp:
                    if resp.status != 200:
                        continue
                    arts = await resp.json()
            except Exception:
                continue
            # Finnhub returns list of dicts
            for a in (arts or [])[:per_sym]:
                try:
                    title = a.get('headline', '')
                    summary = a.get('summary', '')
                    url = a.get('url', '')
                    src = a.get('source', 'Finnhub')
                    ts = a.get('datetime')
                    try:
                        published = datetime.fromtimestamp(int(ts)) if ts is not None else end_date
                    except Exception:
                        published = end_date
                    ni = NewsItem(
                        title=title,
                        content=summary,
                        source=src,
                        published_at=published,
                        url=url,
                        sentiment_score=None,
                        relevance_score=0.7,
                        symbols=[sym]
                    )
                    out.append(ni)
                except Exception:
                    continue
            try:
                await asyncio.sleep(0.2)
            except Exception:
                pass
        return out

    async def start(self):
        """Initialize service connections."""
        logger.info("Starting News Service")
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Trading-System/1.0'}
        )
        
        # Initialize cache
        try:
            self.cache = await get_trading_cache()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to initialize trading cache (continuing without cache): %s", e)
            self.cache = None
        
        # Initialize message producer
        try:
            self.pulsar_client = get_pulsar_client()
            self.producer = self.pulsar_client.create_producer(
                topic='persistent://trading/production/news-data',
                producer_name='news-service'
            )
            logger.info("Connected to message system")
        except Exception as e:
            logger.warning(f"Failed to connect to message system: {e}")
        # Configure QuestDB sender conf string (reuse env from data-ingestion service)
        try:
            if self.enable_questdb_persist and self._qdb_sender:
                host = os.getenv('QUESTDB_HOST', 'trading-questdb')
                proto = os.getenv('QUESTDB_INGEST_PROTOCOL', 'tcp').strip().lower()
                if proto == 'http':
                    port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
                    self._qdb_conf = f"http::addr={host}:{port};"
                else:
                    port = int(os.getenv('QUESTDB_LINE_TCP_PORT', '9009'))
                    self._qdb_conf = f"tcp::addr={host}:{port};"
        except Exception:  # noqa: BLE001
            self._qdb_conf = None
        
        # Best-effort Polygon Flat Files (S3) configuration/health check
        try:
            cfg = self.polygon_flatfiles or {}
            access = (cfg.get('access_key') or '').strip()
            secret = (cfg.get('secret_key') or '').strip()
            bucket = (cfg.get('bucket') or '').strip()
            endpoint = (cfg.get('endpoint_url') or '').strip()
            region = (cfg.get('region') or 'us-east-1').strip()
            if not (access and secret and bucket):
                logger.info(
                    "Polygon Flat Files disabled: missing POLYGON_S3_* envs",
                    extra={
                        "event": "polygon_flatfiles_disabled",
                        "have_access": bool(access),
                        "have_secret": bool(secret),
                        "have_bucket": bool(bucket),
                    }
                )
            else:
                try:
                    import boto3  # type: ignore
                    from botocore.config import Config as _BotoCfg  # type: ignore
                    s3 = boto3.client(
                        's3',
                        endpoint_url=endpoint or None,
                        aws_access_key_id=access,
                        aws_secret_access_key=secret,
                        region_name=region or 'us-east-1',
                        config=_BotoCfg(signature_version='s3v4', retries={'max_attempts': 2, 'mode': 'standard'})
                    )
                    # Lightweight probe: list up to 1 key under news/ prefix
                    s3.list_objects_v2(Bucket=bucket, Prefix='news/', MaxKeys=1)
                    logger.info(
                        "Polygon Flat Files S3 connectivity OK",
                        extra={
                            "event": "polygon_flatfiles_s3_ok",
                            "bucket": bucket,
                            "endpoint": endpoint or 'https://files.polygon.io',
                            "region": region or 'us-east-1',
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Polygon Flat Files S3 connectivity failed: %s",
                        e,
                        extra={
                            "event": "polygon_flatfiles_s3_failed",
                            "bucket": bucket,
                            "endpoint": endpoint or 'https://files.polygon.io',
                            "region": region or 'us-east-1',
                        }
                    )
        except Exception:
            # Never block service start on this optional capability
            pass
        # Optional: schedule backlog reindexer (hourly) and historical backfill (one-shot)
        try:
            if os.getenv('ENABLE_NEWS_BACKLOG_REINDEX', 'false').lower() in ('1','true','yes','on'):
                async def _reindex_loop():
                    try:
                        await self.reindex_news_backlog()
                    except Exception:
                        pass
                    while True:
                        try:
                            await asyncio.sleep(3600)
                            await self.reindex_news_backlog()
                        except Exception:
                            await asyncio.sleep(60)
                asyncio.create_task(_reindex_loop())
                logger.info("News backlog reindexer scheduled")
        except Exception:
            pass
        try:
            if os.getenv('ENABLE_NEWS_HISTORICAL_BACKFILL', 'false').lower() in ('1','true','yes','on'):
                asyncio.create_task(self._run_news_backfill_task())
                logger.info("News historical backfill scheduled")
        except Exception:
            pass

    async def debug_list_polygon_news_keys(self, prefix: str | None = None, max_keys: int = 50) -> list[str]:
        """List sample keys from Polygon Flat Files (S3) for debugging key structure.

        Returns up to max_keys keys under the given prefix (default 'news/').
        """
        try:
            cfg = self.polygon_flatfiles or {}
            access = (cfg.get('access_key') or '').strip()
            secret = (cfg.get('secret_key') or '').strip()
            bucket = (cfg.get('bucket') or '').strip()
            endpoint = (cfg.get('endpoint_url') or '').strip()
            region = (cfg.get('region') or 'us-east-1').strip()
            if not (access and secret and bucket):
                return []
            import boto3  # type: ignore
            from botocore.config import Config as _BotoCfg  # type: ignore
            s3 = boto3.client(
                's3',
                endpoint_url=endpoint or None,
                aws_access_key_id=access,
                aws_secret_access_key=secret,
                region_name=region or 'us-east-1',
                config=_BotoCfg(signature_version='s3v4', retries={'max_attempts': 2, 'mode': 'standard'})
            )
            pref = (prefix or 'news/').lstrip('/')
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=pref, MaxKeys=max(1, min(max_keys, 1000)))
            contents = resp.get('Contents') or []
            keys: list[str] = []
            for obj in contents:
                try:
                    k = obj.get('Key')
                    if k:
                        keys.append(str(k))
                except Exception:
                    continue
            return keys
        except Exception:
            return []

    async def debug_probe_polygon_news_keys_for_date(self, day: datetime) -> dict:
        """Probe common Polygon news key patterns for a specific date using HEAD requests.

        Returns a dict with keys: {"date": YYYY-MM-DD, "found": [keys], "tried": n}
        """
        try:
            cfg = self.polygon_flatfiles or {}
            access = (cfg.get('access_key') or '').strip()
            secret = (cfg.get('secret_key') or '').strip()
            bucket = (cfg.get('bucket') or '').strip()
            endpoint = (cfg.get('endpoint_url') or '').strip()
            region = (cfg.get('region') or 'us-east-1').strip()
            if not (access and secret and bucket):
                return {"date": day.date().isoformat(), "found": [], "tried": 0}
            import boto3  # type: ignore
            from botocore.config import Config as _BotoCfg  # type: ignore
            s3 = boto3.client(
                's3',
                endpoint_url=endpoint or None,
                aws_access_key_id=access,
                aws_secret_access_key=secret,
                region_name=region or 'us-east-1',
                config=_BotoCfg(signature_version='s3v4', retries={'max_attempts': 2, 'mode': 'standard'})
            )
            d = day.date()
            y = f"{d.year:04d}"; m = f"{d.month:02d}"; dd = f"{d.day:02d}"
            base_names = [
                f"news-{y}-{m}-{dd}", f"news_{y}_{m}_{dd}", f"{y}-{m}-{dd}", f"{y}{m}{dd}", "news"
            ]
            exts = [".ndjson", ".jsonl", ".json", ".ndjson.gz", ".jsonl.gz", ".json.gz"]
            prefixes = [
                f"news/{y}/{m}/{dd}/", f"news/{y}/{m}/", f"news/{y}-{m}-{dd}/", "news/", f"flatfiles/news/{y}/{m}/{dd}/"
            ]
            tried = 0
            found: list[str] = []
            # Try HEAD on combinations
            for pref in prefixes:
                for b in base_names:
                    for ext in exts:
                        key = f"{pref}{b}{ext}".replace('//','/')
                        tried += 1
                        try:
                            s3.head_object(Bucket=bucket, Key=key)
                            found.append(key)
                        except Exception:
                            continue
            return {"date": d.isoformat(), "found": found, "tried": tried}
        except Exception:
            return {"date": day.date().isoformat(), "found": [], "tried": 0}
    
    async def stop(self):
        """Cleanup service connections."""
        if self.session:
            await self.session.close()
        if self.producer:
            self.producer.close()
        if self.pulsar_client:
            self.pulsar_client.close()
        logger.info("News Service stopped")
    
    async def collect_financial_news(
        self,
        symbols: Optional[List[str]] = None,
        hours_back: int = 1,
        max_articles: int = 50,
        *,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        backfill_mode: bool = False,
    ) -> List[NewsItem]:
        """Collect financial news from multiple sources."""
        if self.enable_structured_logs:
            try:
                logger.info(
                    "news_collect_start",
                    extra={
                        "event": "news_collect_start",
                        "symbols": symbols or [],
                        "hours_back": hours_back,
                        "max_articles": max_articles,
                        "start_date": start_date.isoformat() if start_date else None,
                        "end_date": end_date.isoformat() if end_date else None,
                    },
                )
            except Exception:
                pass
        else:
            logger.info(f"Collecting financial news for symbols: {symbols}")

        all_news: List[NewsItem] = []

        # Provider priority: EODHD (preferred, paid), Polygon (Flat Files/API), Alpha Vantage, Finnhub (free), NewsAPI (off by default), GDELT
        # 1) EODHD first (paid primary)
        if getattr(self, "enable_eodhd_news", False) and self.eodhd_config.get("api_key") and symbols:
            try:
                if start_date is not None and end_date is not None:
                    eodhd_items = await self._collect_from_eodhd_range(
                        symbols, start_date, end_date, max(max_articles // 2, 20)
                    )
                else:
                    eodhd_items = await self._collect_from_eodhd_recent(
                        symbols, hours_back, max(max_articles // 2, 20)
                    )
                all_news.extend(eodhd_items)
            except Exception as e:
                logger.debug("EODHD news collection failed: %s", e)

        # 2) Polygon Flat Files (bulk, cheap)
        if self.enable_polygon_flatfiles and (start_date is not None or end_date is not None):
            try:
                p_items = await self._collect_from_polygon_flatfiles(
                    symbols,
                    start_date or datetime.utcnow() - timedelta(days=1),
                    end_date or datetime.utcnow(),
                    max_articles,
                )
                all_news.extend(p_items)
            except Exception as e:
                logger.debug("Polygon Flat Files collection failed: %s", e)

        # 2) Polygon API
        if self.enable_polygon_news and symbols and (start_date is not None and end_date is not None):
            try:
                p_api = await self._collect_from_polygon_news(
                    symbols, start_date, end_date, max_articles // 2, relaxed=backfill_mode
                )
                all_news.extend(p_api)
            except Exception as e:
                logger.debug("Polygon API news failed: %s", e)

        # 3) Alpha Vantage
        if self.enable_alpha_vantage_news:
            try:
                av_articles = await self._collect_from_alpha_vantage(
                    symbols or [], max_articles // 3, start_date=start_date, end_date=end_date
                )
                all_news.extend(av_articles)
            except Exception as e:
                logger.debug("Alpha Vantage news collection failed: %s", e)

        # 4) Finnhub (free-tier limited)
        if self.enable_finnhub and symbols:
            try:
                lim_syms = symbols[: self.finnhub_max_symbols_per_window]
                finnhub_articles = await self._collect_from_finnhub(lim_syms, max_articles // 3)
                all_news.extend(finnhub_articles)
            except Exception as e:
                logger.warning("Finnhub news collection failed (continuing): %s", e)

        # 5) NewsAPI (explicitly opt-in; only when EODHD disabled/unavailable)
        if self.enable_newsapi and not getattr(self, "enable_eodhd_news", False):
            try:
                news_api_articles = await self._collect_from_newsapi(
                    symbols, hours_back, max_articles // 3, start_date=start_date, end_date=end_date, relaxed=backfill_mode
                )
                all_news.extend(news_api_articles)
            except Exception as e:
                # Gracefully continue if NewsAPI rate limited or errors
                logger.warning("NewsAPI collection failed (continuing with other providers): %s", e)

        # 6) GDELT (broad public)
        if self.enable_gdelt and (start_date is not None and end_date is not None):
            try:
                gd = await self._collect_from_gdelt(symbols, start_date, end_date, max_articles // 4, relaxed=backfill_mode)
                all_news.extend(gd)
            except Exception as e:
                logger.debug("GDELT collection failed: %s", e)

        # Remove duplicates based on URL or title similarity
        unique_news = self._deduplicate_news(all_news)

        # Adaptive retry: if no articles found and hours_back < 72 and no explicit start_date, widen and retry once (NewsAPI only)
        if not unique_news and (start_date is None) and hours_back < 72 and self.news_api_config.get("api_key"):
            try:
                wider_hours = min(72, max(2 * max(1, hours_back), hours_back + 12))
                logger.info("No news found; widening hours_back to %sh and retrying NewsAPI once", wider_hours)
                more_news = await self._collect_from_newsapi(symbols, wider_hours, max_articles)
                unique_news = self._deduplicate_news(more_news)
            except Exception:
                pass

        # Historical-range fallback: if explicit start/end provided and still no items, try Alpha Vantage topic-based
        if not unique_news and (start_date is not None) and (end_date is not None) and self.alpha_vantage_config.get("api_key"):
            try:
                av_items = await self._collect_from_alpha_vantage([], max_articles, start_date=start_date, end_date=end_date)
                # Tag with 'ALL' if symbols list is empty for persistence visibility
                for it in av_items:
                    if not getattr(it, "symbols", None):
                        try:
                            it.symbols = ["ALL"]
                        except Exception:
                            pass
                unique_news = self._deduplicate_news(av_items)
            except Exception:
                unique_news = []

        # Analyze sentiment for each article
        kept: List[NewsItem] = []
        filtered_count = 0

        # Lazy metric registration block for ingest filtering
        if not self._ingest_filter_metrics_ready:
            try:
                from prometheus_client import Counter as _PC  # type: ignore
                try:
                    self._news_ingest_filtered = _PC(
                        "news_ingest_filtered_total",
                        "News articles dropped at ingestion before persistence",
                        ["reason"],
                    )
                except Exception:
                    self._news_ingest_filtered = None
                try:
                    self._news_ingest_kept = _PC(
                        "news_ingest_kept_total", "News articles kept after ingestion filters"
                    )
                except Exception:
                    self._news_ingest_kept = None
            except Exception:
                self._news_ingest_filtered = None  # type: ignore
                self._news_ingest_kept = None  # type: ignore
            self._ingest_filter_metrics_ready = True

        for article in unique_news:
            try:
                if article.sentiment_score is None or article.sentiment_score == 0.0:
                    sentiment = await self._analyze_sentiment(article.title + " " + article.content)
                    article.sentiment_score = float(sentiment)
            except Exception:
                try:
                    article.sentiment_score = 0.0
                except Exception:
                    pass

            # Compute value score locally
            try:
                _val = float(self._compute_value_score(article))
            except Exception:
                _val = 0.0

            # Apply ingestion-time value threshold (distinct from retention prune which may use different floor)
            try:
                if _val < self._news_ingest_value_min:
                    filtered_count += 1
                    if getattr(self, "_news_ingest_filtered", None):
                        try:
                            self._news_ingest_filtered.labels(reason="value_score").inc()  # type: ignore[union-attr]
                        except Exception:
                            pass
                    continue
            except Exception:
                pass

            kept.append(article)

        unique_news = kept
        if filtered_count and getattr(self, "_news_ingest_filtered", None) is None:
            pass  # metric not available; silently ignore
        if kept and getattr(self, "_news_ingest_kept", None):
            try:
                self._news_ingest_kept.inc(len(kept))  # type: ignore[union-attr]
            except Exception:
                pass

        # Cache and publish news
        for article in unique_news:
            # Optional: caching disabled (no dedicated cache method for news items)
            if self.producer:
                try:
                    await self._publish_news(article)
                except Exception as e:
                    logger.warning(f"Failed to publish news: {e}")
            # Persist to QuestDB (optional)
            try:
                await self._maybe_persist_news(article)
            except Exception:
                pass

        if self.enable_structured_logs:
            try:
                logger.info(
                    "news_collect_complete",
                    extra={
                        "event": "news_collect_complete",
                        "symbols": symbols or [],
                        "article_count": len(unique_news),
                        "filtered_low_value": filtered_count,
                        "ingest_value_min": self._news_ingest_value_min,
                    },
                )
            except Exception:
                pass
        else:
            logger.info(
                f"Collected {len(unique_news)} news articles (filtered={filtered_count} value_min={self._news_ingest_value_min})"
            )

        # Optional: index into vector store via ML service
        try:
            if self.enable_weaviate_persist and unique_news:
                await self._index_news_to_weaviate(unique_news)
        except Exception as e:
            logger.warning(f"Weaviate indexing request failed: {e}")

        return unique_news
    
    async def collect_social_sentiment(
        self, 
        symbols: List[str],
        hours_back: int = 1
    ) -> List[SocialSentiment]:
        """Collect social media sentiment for symbols."""
        logger.info(f"Collecting social sentiment for: {symbols}")
        
        sentiment_data = []
        
        # Collect from Reddit (if configured)
        if self.reddit_config['client_id']:
            reddit_sentiment = await self._collect_reddit_sentiment(symbols, hours_back)
            sentiment_data.extend(reddit_sentiment)
        
        return sentiment_data
    
    @provider_instrumentation('newsapi', 'everything')
    async def _collect_from_newsapi(
        self,
        symbols: Optional[List[str]],
        hours_back: int,
        max_articles: int,
        *,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        relaxed: bool = False,
    ) -> List[NewsItem]:
        """Collect news from NewsAPI."""
        # ---------------- Query Construction with Chunking ---------------- #
        # Large symbol sets can produce a 400 from NewsAPI due to query length.
        # We chunk symbol lists and merge results.
        max_syms_per_query =  int(os.getenv('NEWSAPI_MAX_SYMBOLS_PER_QUERY','20') or '20')
        max_syms_per_query = max(1, min(max_syms_per_query, 40))
        symbols_clean: List[str] = []
        if symbols:
            seen = set()
            for s in symbols:
                s2 = (s or '').strip().upper()
                if s2 and s2 not in seen:
                    seen.add(s2)
                    symbols_clean.append(s2)
        symbol_chunks: List[List[str]] = []
        if symbols_clean:
            for i in range(0, len(symbols_clean), max_syms_per_query):
                symbol_chunks.append(symbols_clean[i:i+max_syms_per_query])
        if not symbol_chunks:
            symbol_chunks = [[]]  # single generic query path

        # Metrics (lazy) for chunking behavior
        try:
            from prometheus_client import Counter as _PC  # type: ignore
            if not hasattr(self, '_newsapi_batches_total'):
                try:
                    self._newsapi_batches_total = _PC('newsapi_batches_total','Total NewsAPI batch queries executed')
                except Exception:
                    self._newsapi_batches_total = None
            if not hasattr(self, '_newsapi_http_400_total'):
                try:
                    self._newsapi_http_400_total = _PC('newsapi_http_400_total','NewsAPI HTTP 400 responses (likely query too large)')
                except Exception:
                    self._newsapi_http_400_total = None
        except Exception:
            pass

        # Determine window limits
        if start_date is not None:
            from_date = start_date.isoformat()
        else:
            from_date = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
        to_date: Optional[str] = None
        if end_date is not None:
            to_date = end_date.isoformat()
        base_url = f"{self.news_api_config['base_url']}/everything"
        aggregated: List[NewsItem] = []
        # Per-chunk allocate article budget
        per_chunk_budget = max(1, max_articles // max(1, len(symbol_chunks)))
        for chunk in symbol_chunks:
            if len(aggregated) >= max_articles:
                break
            # Build query for this chunk
            if chunk:
                query = (
                    f"({' OR '.join(chunk)}) AND (stock OR shares OR trading OR market)"
                    if not relaxed else f"({' OR '.join(chunk)})"
                )
            else:
                query = (
                    "stock market OR trading OR finance OR earnings"
                    if not relaxed else "technology OR ai OR finance OR markets"
                )
            page = 1
            per_page = min((max(per_chunk_budget, 25) if relaxed else per_chunk_budget), 100)
            # Retry logic for 400 reduction (halve chunk until success or single symbol)
            current_chunk = chunk[:]
            while True:
                params_base = {
                    'q': query,
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': per_page,
                    'apiKey': self.news_api_config['api_key']
                }
                if to_date:
                    params_base['to'] = to_date
                # Paginate up to 2 pages (3 if relaxed) per chunk to cap load
                local_items: List[NewsItem] = []
                error_400 = False
                max_pages = 3 if relaxed else 2
                while len(local_items) < per_chunk_budget and page <= max_pages:
                    params = dict(params_base)
                    params['page'] = page
                    async with self.session.get(base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            arts = data.get('articles', [])
                            for art in arts:
                                if len(local_items) + len(aggregated) >= max_articles:
                                    break
                                if relaxed or self._is_financial_news(art.get('title','') + ' ' + art.get('description','')):
                                    try:
                                        published = datetime.fromisoformat(art.get('publishedAt','').replace('Z','+00:00'))
                                    except Exception:
                                        published = datetime.utcnow()
                                    ni = NewsItem(
                                        title=art.get('title',''),
                                        content=art.get('description',''),
                                        source=art.get('source',{}).get('name','NewsAPI'),
                                        published_at=published,
                                        url=art.get('url',''),
                                        sentiment_score=None,
                                        relevance_score=self._calculate_relevance(art, symbols),
                                        symbols=self._extract_symbols(art.get('title','') + ' ' + art.get('description',''), symbols)
                                    )
                                    local_items.append(ni)
                        elif response.status == 400:
                            try:
                                from services.data_ingestion.provider_metrics import record_http_response  # type: ignore
                                record_http_response('newsapi','everything',400)
                            except Exception:
                                pass
                            error_400 = True
                            if getattr(self, '_newsapi_http_400_total', None):
                                try: self._newsapi_http_400_total.inc()  # type: ignore[union-attr]
                                except Exception: pass
                            try:
                                # Often indicates query too long or invalid date range
                                body = (await response.text())[:400]
                                logger.warning("NewsAPI HTTP 400 for chunk(%s syms) page=%s: %s", len(current_chunk), page, body)
                            except Exception:
                                pass
                            break
                        elif response.status == 429:
                            try:
                                from services.data_ingestion.provider_metrics import record_http_response, record_rate_limit  # type: ignore
                                record_http_response('newsapi','everything',429)
                                record_rate_limit('newsapi','everything')
                            except Exception:
                                pass
                            try:
                                body = (await response.text())[:400]
                                logger.warning("NewsAPI rate limited (429): %s", body)
                            except Exception:
                                pass
                            break
                        elif response.status in (401,403):
                            # Auth/plan issue – abort early for this chunk
                            error_400 = False
                            try:
                                from services.data_ingestion.provider_metrics import record_http_response  # type: ignore
                                record_http_response('newsapi','everything',response.status)
                            except Exception:
                                pass
                            try:
                                body = (await response.text())[:400]
                                logger.warning("NewsAPI auth/plan error HTTP %s: %s", response.status, body)
                            except Exception:
                                pass
                            break
                        else:
                            # Other errors: break out of this chunk
                            try:
                                from services.data_ingestion.provider_metrics import record_http_response  # type: ignore
                                record_http_response('newsapi','everything',response.status)
                            except Exception:
                                pass
                            try:
                                body = (await response.text())[:400]
                                logger.warning("NewsAPI unexpected HTTP %s: %s", response.status, body)
                            except Exception:
                                pass
                            break
                    page += 1
                    if error_400:
                        break
                if error_400 and len(current_chunk) > 1:
                    # Halve symbols and retry (adaptive shrinking)
                    half = max(1, len(current_chunk)//2)
                    current_chunk = current_chunk[:half]
                    query = f"({' OR '.join(current_chunk)}) AND (stock OR shares OR trading OR market)"
                    page = 1
                    continue
                # Merge accepted items
                aggregated.extend(local_items)
                if getattr(self, '_newsapi_batches_total', None):
                    try: self._newsapi_batches_total.inc()  # type: ignore[union-attr]
                    except Exception: pass
                break  # exit chunk loop
        return aggregated

    @provider_instrumentation('gdelt', 'doc')
    async def _collect_from_gdelt(
        self,
        symbols: Optional[List[str]],
        start_date: datetime,
        end_date: datetime,
        max_articles: int,
        relaxed: bool = False,
    ) -> List[NewsItem]:
        """Collect news from GDELT 2.1 Doc API (no API key required).

        Reference: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
        Endpoint:  https://api.gdeltproject.org/api/v2/doc/doc?query=...&mode=ArtList&format=json
        """
        if not self.session:
            return []
        base = 'https://api.gdeltproject.org/api/v2/doc/doc'
        # Build query. Prefer symbol terms combined with finance context when not relaxed.
        syms: List[str] = []
        if symbols:
            for s in symbols[:20]:
                s2 = (s or '').strip().upper()
                if s2:
                    syms.append(s2)
        if syms:
            if relaxed:
                q = ' OR '.join([f'"{s}"' for s in syms])
            else:
                q = '(' + ' OR '.join([f'"{s}"' for s in syms]) + ') AND (stock OR shares OR market OR earnings OR finance)'
        else:
            q = 'stock OR shares OR market OR earnings OR finance'
        # Time bounds (UTC) in YYYYMMDDHHMMSS
        def _fmt(dt: datetime) -> str:
            try:
                return dt.strftime('%Y%m%d%H%M%S')
            except Exception:
                from datetime import datetime as _dt
                return _dt.utcnow().strftime('%Y%m%d%H%M%S')
        start_s = _fmt(start_date)
        end_s = _fmt(end_date + timedelta(hours=23, minutes=59, seconds=59))
        per = max(1, min(max_articles, 100))
        params = {
            'query': q,
            'mode': 'ArtList',
            'format': 'json',
            'maxrecords': str(per),
            'startdatetime': start_s,
            'enddatetime': end_s,
        }
        out: List[NewsItem] = []
        try:
            async with self.session.get(base, params=params, timeout=30) as resp:
                if resp.status != 200:
                    try:
                        from services.data_ingestion.provider_metrics import record_http_response  # type: ignore
                        record_http_response('gdelt','doc',resp.status)
                    except Exception:
                        pass
                    try:
                        body = (await resp.text())[:400]
                        logger.warning("GDELT HTTP %s: %s", resp.status, body)
                    except Exception:
                        pass
                    return out
                data = await resp.json()
        except Exception:
            return out
        # Parse articles (GDELT returns 'articles') best-effort
        arts = []
        try:
            arts = data.get('articles') or data.get('Articles') or []
        except Exception:
            arts = []
        for a in arts[:per]:
            try:
                title = a.get('title') or a.get('Title') or ''
                url = a.get('url') or a.get('URL') or ''
                source = a.get('domain') or a.get('SourceCommonName') or 'GDELT'
                # published time keys vary: 'seendate' (YYYYMMDDHHMMSS), 'publication_datetime'
                published = datetime.utcnow()
                for k in ('seendate','publishDate','pubDate','publication_datetime'):
                    v = a.get(k)
                    if not v:
                        continue
                    try:
                        if isinstance(v, str) and len(v) >= 14 and v.isdigit():
                            published = datetime.strptime(v[:14], '%Y%m%d%H%M%S')
                            break
                        # Try ISO parse as fallback
                        published = datetime.fromisoformat(str(v).replace('Z','+00:00'))
                        break
                    except Exception:
                        continue
                descr = a.get('snippet') or a.get('excerpt') or a.get('summary') or ''
                # Best-effort symbol extraction from title+snippet
                text = f"{title} {descr}"
                syms_out: List[str] = []
                for s in syms:
                    try:
                        if s in text.upper():
                            syms_out.append(s)
                    except Exception:
                        continue
                ni = NewsItem(
                    title=title,
                    content=descr,
                    source=str(source or 'GDELT'),
                    published_at=published,
                    url=url,
                    sentiment_score=None,
                    relevance_score=0.6 if syms_out else 0.4,
                    symbols=syms_out or (syms[:1] if syms else ['ALL'])
                )
                # Optional filter by financial keywords when not relaxed
                if relaxed or self._is_financial_news(title + ' ' + descr):
                    out.append(ni)
            except Exception:
                continue
        return out

    @provider_instrumentation('polygon', 'flatfiles')
    async def _collect_from_polygon_flatfiles(
        self,
        symbols: Optional[List[str]],
        start_date: datetime,
        end_date: datetime,
        max_articles: int,
    ) -> List[NewsItem]:
        """Load Polygon News from Flat Files (S3) archive between dates.

        Expects creds via env POLYGON_S3_ACCESS_KEY_ID / POLYGON_S3_SECRET_ACCESS_KEY
        and endpoint/bucket from POLYGON_S3_ENDPOINT / POLYGON_S3_BUCKET.

        Flat file structure is typically grouped by date. We list and read small JSON/NDJSON files
        for days overlapping [start_date, end_date], filter by symbols and cap to max_articles.
        """
        cfg = self.polygon_flatfiles
        if not (cfg.get('access_key') and cfg.get('secret_key') and cfg.get('bucket')):
            return []
        import boto3
        from botocore.config import Config as _BotoCfg  # type: ignore
        s3 = boto3.client(
            's3',
            endpoint_url=cfg.get('endpoint_url'),
            aws_access_key_id=cfg.get('access_key'),
            aws_secret_access_key=cfg.get('secret_key'),
            region_name=cfg.get('region') or 'us-east-1',
            config=_BotoCfg(signature_version='s3v4', retries={'max_attempts': 3, 'mode': 'standard'})
        )
        bucket = cfg['bucket']
        # Date iteration
        day = start_date.date()
        end_day = end_date.date()
        syms_set = set((s or '').upper() for s in (symbols or []) if s)
        out: List[NewsItem] = []
        # Heuristic paths; Polygon docs vary. Prefer direct key probes (works even if ListObjects is disallowed),
        # with a fallback to listing a few lightweight prefixes when permitted.
        def _candidate_prefixes(d):
            y = f"{d.year:04d}"
            m = f"{d.month:02d}"
            dd = f"{d.day:02d}"
            return [
                f"news/{y}/{m}/{dd}/",
                f"news/{y}-{m}-{dd}/",
                f"news/{y}-{m}-{dd}",
                f"news/{y}/{m}/",
                f"news/",
                f"reference/news/{y}/{m}/{dd}/",
                f"reference/news/{y}/{m}/",
                f"reference/news/",
                f"v2/reference/news/{y}/{m}/{dd}/",
                f"v2/reference/news/{y}/{m}/",
                f"v2/reference/news/",
            ]
        # Direct key candidates per day (try uncompressed and gzip variants)
        def _candidate_keys(d):
            y = f"{d.year:04d}"; m = f"{d.month:02d}"; dd = f"{d.day:02d}"
            base_names = [
                f"news-{y}-{m}-{dd}",
                f"{y}-{m}-{dd}",
                "news",
                f"reference-news-{y}-{m}-{dd}",
                f"reference-news",
                f"v2-reference-news-{y}-{m}-{dd}",
                f"v2-reference-news",
            ]
            exts = [".ndjson", ".jsonl", ".json", ".ndjson.gz", ".jsonl.gz", ".json.gz"]
            keys: list[str] = []
            # Nested directories and flat daily files
            for b in base_names:
                for ext in exts:
                    keys.append(f"news/{y}/{m}/{dd}/{b}{ext}")
                    keys.append(f"news/{y}/{m}/{b}{ext}")
                    keys.append(f"news/{y}-{m}-{dd}/{b}{ext}")
                    keys.append(f"news/{y}-{m}-{dd}{ext}")
                    keys.append(f"news/{b}-{y}-{m}-{dd}{ext}")
                    # reference path variants
                    keys.append(f"reference/news/{y}/{m}/{dd}/{b}{ext}")
                    keys.append(f"reference/news/{y}/{m}/{b}{ext}")
                    keys.append(f"reference/news/{y}-{m}-{dd}/{b}{ext}")
                    keys.append(f"reference/news/{y}-{m}-{dd}{ext}")
                    keys.append(f"reference/news/{b}-{y}-{m}-{dd}{ext}")
                    # v2 reference path variants
                    keys.append(f"v2/reference/news/{y}/{m}/{dd}/{b}{ext}")
                    keys.append(f"v2/reference/news/{y}/{m}/{b}{ext}")
                    keys.append(f"v2/reference/news/{y}-{m}-{dd}/{b}{ext}")
                    keys.append(f"v2/reference/news/{y}-{m}-{dd}{ext}")
                    keys.append(f"v2/reference/news/{b}-{y}-{m}-{dd}{ext}")
            # Deduplicate while preserving order
            seen: set[str] = set()
            uniq: list[str] = []
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    uniq.append(k)
            return uniq
        def _read_records_from_text(text: str) -> list:
            # Try NDJSON first
            try:
                if text.count('\n') > 0:
                    recs = [json.loads(line) for line in text.splitlines() if line.strip()]
                    if recs:
                        return recs
            except Exception:
                pass
            # Fallback to JSON array or {results:[...]}
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get('results') or data.get('articles') or []
            except Exception:
                return []
            return []
        def _try_get_object(key: str) -> list:
            # Returns parsed records from the key or empty list
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                body = obj['Body'].read()
                text: str
                if key.endswith('.gz'):
                    import gzip
                    try:
                        text = gzip.decompress(body).decode('utf-8', errors='ignore')
                    except Exception:
                        # Some providers set ContentEncoding=gzip but no .gz suffix
                        try:
                            text = gzip.decompress(body).decode('utf-8', errors='ignore')
                        except Exception:
                            text = body.decode('utf-8', errors='ignore')
                else:
                    # Handle optional ContentEncoding
                    enc = obj.get('ContentEncoding') or ''
                    if isinstance(enc, str) and 'gzip' in enc.lower():
                        import gzip
                        try:
                            text = gzip.decompress(body).decode('utf-8', errors='ignore')
                        except Exception:
                            text = body.decode('utf-8', errors='ignore')
                    else:
                        text = body.decode('utf-8', errors='ignore')
                return _read_records_from_text(text)
            except Exception:
                return []
        # List keys per day with a bounded number to avoid scanning entire bucket
        while day <= end_day and len(out) < max_articles:
            # 1) Try direct key probes first (works when ListObjects is restricted)
            for key in _candidate_keys(day):
                if len(out) >= max_articles:
                    break
                recs = _try_get_object(key)
                for a in recs:
                    if len(out) >= max_articles:
                        break
                    try:
                        title = a.get('title') or ''
                        url = a.get('article_url') or a.get('url') or ''
                        src = (a.get('publisher') or {}).get('name') or a.get('source') or 'Polygon'
                        pub_raw = a.get('published_utc') or a.get('timestamp') or a.get('time_published')
                        published = None
                        if pub_raw:
                            try:
                                published = date_parser.parse(str(pub_raw))
                            except Exception:
                                published = None
                        published = published or datetime.combine(day, datetime.min.time())
                        tickers = a.get('tickers') or a.get('ticker_symbols') or a.get('symbols') or []
                        tickers = [str(t).upper() for t in tickers if isinstance(t, (str,))]
                        # Symbol filter if provided
                        if syms_set:
                            if not (syms_set.intersection(tickers) or any(s in (title.upper()) for s in syms_set)):
                                continue
                        ni = NewsItem(
                            title=title,
                            content=a.get('description') or a.get('excerpt') or a.get('summary') or '',
                            source=src,
                            published_at=published,
                            url=url,
                            sentiment_score=0.0,
                            relevance_score=0.8 if tickers else 0.5,
                            symbols=(list(syms_set.intersection(tickers)) or (tickers[:3] if tickers else (list(syms_set)[:1] if syms_set else ['ALL'])))
                        )
                        out.append(ni)
                    except Exception:
                        continue
            # 2) If still nothing for the day, try a conservative list on a couple of prefixes
            if len(out) < max_articles:
                for pref in _candidate_prefixes(day):
                    try:
                        resp = s3.list_objects_v2(Bucket=bucket, Prefix=pref, MaxKeys=30)
                        contents = resp.get('Contents') or []
                        if not contents:
                            continue
                        for obj in contents[:30]:
                            key = obj.get('Key') or ''
                            if not key or not any(k in key.lower() for k in ('.json', '.ndjson', '.jsonl', '.gz')):
                                continue
                            recs = _try_get_object(key)
                            for a in recs:
                                if len(out) >= max_articles:
                                    break
                                try:
                                    title = a.get('title') or ''
                                    url = a.get('article_url') or a.get('url') or ''
                                    src = (a.get('publisher') or {}).get('name') or a.get('source') or 'Polygon'
                                    pub_raw = a.get('published_utc') or a.get('timestamp') or a.get('time_published')
                                    published = None
                                    if pub_raw:
                                        try:
                                            published = date_parser.parse(str(pub_raw))
                                        except Exception:
                                            published = None
                                    published = published or datetime.combine(day, datetime.min.time())
                                    tickers = a.get('tickers') or a.get('ticker_symbols') or a.get('symbols') or []
                                    tickers = [str(t).upper() for t in tickers if isinstance(t, (str,))]
                                    if syms_set:
                                        if not (syms_set.intersection(tickers) or any(s in (title.upper()) for s in syms_set)):
                                            continue
                                    ni = NewsItem(
                                        title=title,
                                        content=a.get('description') or a.get('excerpt') or a.get('summary') or '',
                                        source=src,
                                        published_at=published,
                                        url=url,
                                        sentiment_score=0.0,
                                        relevance_score=0.8 if tickers else 0.5,
                                        symbols=(list(syms_set.intersection(tickers)) or (tickers[:3] if tickers else (list(syms_set)[:1] if syms_set else ['ALL'])))
                                    )
                                    out.append(ni)
                                except Exception:
                                    continue
                        if len(out) >= max_articles:
                            break
                    except Exception:
                        continue
            day = day + timedelta(days=1)
        return out
    
    @provider_instrumentation('finnhub', 'company-news')
    async def _collect_from_finnhub(self, symbols: List[str], max_articles: int) -> List[NewsItem]:
        """Collect company-specific news from Finnhub."""
        all_news = []
        for symbol in symbols[: self.finnhub_max_symbols_per_window]:  # Limit to avoid rate limits
            params = {
                'symbol': symbol,
                'from': (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'to': datetime.utcnow().strftime('%Y-%m-%d'),
                'token': self.finnhub_config['api_key']
            }
            url = f"{self.finnhub_config['base_url']}/company-news"
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    articles = await response.json()
                    for article in articles[:max_articles // len(symbols)]:
                        news_item = NewsItem(
                            title=article.get('headline', ''),
                            content=article.get('summary', ''),
                            source='Finnhub',
                            published_at=datetime.fromtimestamp(article.get('datetime', 0)),
                            url=article.get('url', ''),
                            sentiment_score=None,
                            relevance_score=0.9,
                            symbols=[symbol]
                        )
                        all_news.append(news_item)
                else:
                    raise RuntimeError(f"Finnhub error HTTP {response.status} for {symbol}")
            await asyncio.sleep(self.finnhub_symbol_sleep)  # rate limiting
        return all_news
    
    async def _collect_reddit_sentiment(
        self, 
        symbols: List[str], 
        hours_back: int
    ) -> List[SocialSentiment]:
        """Collect sentiment from Reddit financial subreddits."""
        try:
            # This would implement Reddit API calls to collect posts/comments
            # from subreddits like r/stocks, r/investing, r/SecurityAnalysis
            # For now, return mock data structure
            
            sentiment_data = []
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'wallstreetbets']
            
            for symbol in symbols:
                # Mock sentiment calculation aligned to SocialSentiment model
                sentiment_data.append(SocialSentiment(
                    platform='reddit',
                    content=f'Mentions summary for {symbol} (last {hours_back}h)',
                    author='aggregate',
                    timestamp=datetime.utcnow(),
                    sentiment_score=0.5,  # Placeholder
                    engagement_score=100.0,  # Mentions count as engagement proxy
                    symbols=[symbol]
                ))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Reddit sentiment collection error: {e}")
        
        return []

    @provider_instrumentation('alpha_vantage', 'NEWS_SENTIMENT')
    async def _collect_from_alpha_vantage(
        self,
        symbols: List[str],
        max_articles: int,
        *,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[NewsItem]:
        """Collect news using Alpha Vantage NEWS_SENTIMENT API.

        Alpha Vantage supports historical range via time_from/time_to epoch seconds.
        We fan out per-symbol with small caps to respect rate limits.
        """
        if not self.alpha_vantage_config['api_key']:
            return []
        out: List[NewsItem] = []
        base_url = self.alpha_vantage_config['base_url'].rstrip('/') + '/query'
        # Convert to Alpha Vantage expected time format (YYYYMMDDTHHMM)
        # Docs: NEWS_SENTIMENT supports time_from/time_to in UTC in the form 20200101T0000
        def _fmt_av_time(dt: Optional[datetime]) -> Optional[str]:
            if not dt:
                return None
            try:
                # Ensure UTC naive/aware safe formatting
                return dt.strftime('%Y%m%dT%H%M')
            except Exception:
                return None
        t_from = _fmt_av_time(start_date)
        t_to = _fmt_av_time(end_date)

        async def _append_feed_items(data: dict, symbols_for_item: List[str]):
            feed = data.get('feed', [])
            per = max_articles if not symbols else max(1, max_articles // max(1, len(symbols[:10])))
            for article in feed[:per]:
                try:
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    published_raw = article.get('time_published')
                    try:
                        published = datetime.strptime(published_raw, '%Y%m%dT%H%M%S') if published_raw else datetime.utcnow()
                    except Exception:
                        published = datetime.utcnow()
                    ni = NewsItem(
                        title=title,
                        content=summary,
                        source=article.get('source', 'Alpha Vantage'),
                        published_at=published,
                        url=article.get('url', ''),
                        sentiment_score=0.0,
                        relevance_score=0.8,
                        symbols=symbols_for_item
                    )
                    out.append(ni)
                except Exception:
                    continue

        # If symbols provided, fan-out per symbol
        if symbols:
            for sym in symbols[:10]:
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': sym,
                    'apikey': self.alpha_vantage_config['api_key'],
                    'limit': min(max_articles, 100)
                }
                if t_from:
                    params['time_from'] = t_from
                if t_to:
                    params['time_to'] = t_to
                try:
                    async with self.session.get(base_url, params=params) as resp:
                        # record raw HTTP status
                        try:
                            record_http_response('alpha_vantage', 'NEWS_SENTIMENT', resp.status)
                            if resp.status == 429:
                                record_rate_limit('alpha_vantage', 'NEWS_SENTIMENT')
                        except Exception:
                            pass
                        if resp.status != 200:
                            try:
                                body = (await resp.text())[:400]
                                logger.warning("AlphaVantage NEWS_SENTIMENT HTTP %s for %s: %s", resp.status, sym, body)
                            except Exception:
                                pass
                            continue
                        data = await resp.json()
                        # Alpha Vantage may return informational messages under 'Information' or 'Note'
                        try:
                            info = data.get('Information') or data.get('Note')
                            if info:
                                logger.warning("AlphaVantage info for %s: %s", sym, str(info)[:400])
                                try:
                                    # treat as soft throttle for observability
                                    record_rate_limit('alpha_vantage', 'NEWS_SENTIMENT')
                                    record_http_response('alpha_vantage', 'NEWS_SENTIMENT', 'note')
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        await _append_feed_items(data, [sym])
                    await asyncio.sleep(0.2)
                except Exception:
                    continue
            return out

        # No symbols -> use topic-based aggregation for broad finance news
        params = {
            'function': 'NEWS_SENTIMENT',
            'topics': os.getenv('ALPHA_VANTAGE_TOPICS', 'financial_markets,earnings,mergers_and_acquisitions,technology,ai').strip(),
            'apikey': self.alpha_vantage_config['api_key'],
            'limit': min(max_articles, 100)
        }
        if t_from:
            params['time_from'] = t_from
        if t_to:
            params['time_to'] = t_to
        try:
            async with self.session.get(base_url, params=params) as resp:
                try:
                    record_http_response('alpha_vantage', 'NEWS_SENTIMENT', resp.status)
                    if resp.status == 429:
                        record_rate_limit('alpha_vantage', 'NEWS_SENTIMENT')
                except Exception:
                    pass
                if resp.status != 200:
                    return out
                data = await resp.json()
                # Note/Information on topic path too
                try:
                    info = data.get('Information') or data.get('Note')
                    if info:
                        logger.warning("AlphaVantage info (topics): %s", str(info)[:400])
                        try:
                            record_rate_limit('alpha_vantage', 'NEWS_SENTIMENT')
                            record_http_response('alpha_vantage', 'NEWS_SENTIMENT', 'note')
                        except Exception:
                            pass
                except Exception:
                    pass
                await _append_feed_items(data, [])
        except Exception:
            return out
        return out

    @provider_instrumentation('polygon', 'reference/news')
    async def _collect_from_polygon_news(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        max_articles: int,
        relaxed: bool = False,
    ) -> List[NewsItem]:
        """Collect news from Polygon.io by date range per symbol.

        Uses v2/reference/news with published_utc.gte/lte filters. Caps to one
        page per symbol per window to respect rate limits.
        """
        if not (self.polygon_config.get('api_key') and symbols and self.session):
            return []
        out: List[NewsItem] = []
        base = self.polygon_config['base_url'].rstrip('/') + '/v2/reference/news'
        # Per-symbol budget
        per_sym = max(1, max_articles // max(1, len(symbols)))
        if relaxed:
            per_sym = max(per_sym, 20)
        # Format timestamps to ISO8601
        gte = start_date.strftime('%Y-%m-%dT00:00:00Z')
        lte = end_date.strftime('%Y-%m-%dT23:59:59Z')
        for sym in symbols[:20]:  # cap symbols per window to bound calls
            params = {
                'ticker': sym,
                'order': 'desc',
                'sort': 'published_utc',
                'limit': min(50, per_sym),
                'published_utc.gte': gte,
                'published_utc.lte': lte,
                'apiKey': self.polygon_config['api_key'],
            }
            try:
                async with self.session.get(base, params=params, timeout=30) as resp:
                    try:
                        record_http_response('polygon', 'reference/news', resp.status)
                        if resp.status == 429:
                            record_rate_limit('polygon', 'reference/news')
                    except Exception:
                        pass
                    if resp.status != 200:
                        try:
                            body = (await resp.text())[:400]
                            logger.warning("Polygon news HTTP %s for %s: %s", resp.status, sym, body)
                        except Exception:
                            pass
                        continue
                    data = await resp.json()
            except Exception:
                continue
            results = data.get('results', []) or []
            for art in results[:per_sym]:
                try:
                    title = art.get('title', '')
                    descr = art.get('description', '') or art.get('excerpt', '') or ''
                    src = art.get('publisher', {}).get('name') or art.get('source') or 'Polygon'
                    url = art.get('article_url') or art.get('url', '')
                    pub_raw = art.get('published_utc') or art.get('timestamp')
                    try:
                        # Polygon returns RFC3339
                        published = datetime.fromisoformat(str(pub_raw).replace('Z','+00:00')) if pub_raw else datetime.utcnow()
                    except Exception:
                        published = datetime.utcnow()
                    tickers = art.get('tickers') or art.get('ticker_symbols') or []
                    tickers = [t for t in (tickers or []) if isinstance(t, str)]
                    syms = [sym]
                    # If response includes explicit tickers, intersect/augment
                    if tickers:
                        try:
                            if sym in tickers:
                                syms = [sym]
                            else:
                                syms = [sym] + [t.upper() for t in tickers[:3] if isinstance(t, str)]
                        except Exception:
                            syms = [sym]
                    ni = NewsItem(
                        title=title,
                        content=descr,
                        source=src,
                        published_at=published,
                        url=url,
                        sentiment_score=0.0,
                        relevance_score=0.8,
                        symbols=syms,
                    )
                except Exception:
                    continue
                out.append(ni)
            # brief pacing between symbols
            try:
                await asyncio.sleep(0.15)
            except Exception:
                pass
        return out

    async def _index_news_to_weaviate(self, items: List[NewsItem]) -> None:
        """Index news items (primary ML endpoint then fallback) with rich metrics.

        Metrics (lazy-created, idempotent):
          - news_weaviate_primary_attempts_total
          - news_weaviate_primary_failures_total
          - news_weaviate_primary_latency_seconds (Histogram)
          - news_weaviate_indexed_total (on successful primary path)
          - news_weaviate_fallback_invocations_total
          - news_weaviate_fallback_indexed_total (existing pattern)
          - news_weaviate_last_failure_timestamp_seconds (Gauge)
        """
        if not items or not self.session:
            return
        url = self.ml_service_url.rstrip('/') + '/vector/index/news'
        payload: List[Dict[str, Any]] = []
        for it in items:
            try:
                published = it.published_at.isoformat()
            except Exception:
                published = datetime.utcnow().isoformat()
            payload.append({
                'title': getattr(it, 'title', ''),
                'content': getattr(it, 'content', ''),
                'source': getattr(it, 'source', ''),
                'published_at': published,
                'url': getattr(it, 'url', ''),
                'symbols': getattr(it, 'symbols', []) or []
            })
        data = {'items': payload}
        # Lazy metric registration block
        try:  # noqa: SIM105
            from prometheus_client import Counter as _PC, Histogram as _PH, Gauge as _PG  # type: ignore
            if not hasattr(self, '_news_weaviate_primary_attempts'):
                try:
                    self._news_weaviate_primary_attempts = _PC('news_weaviate_primary_attempts_total','Primary news indexing attempts')
                except Exception: self._news_weaviate_primary_attempts = None
            if not hasattr(self, '_news_weaviate_primary_failures'):
                try:
                    self._news_weaviate_primary_failures = _PC('news_weaviate_primary_failures_total','Primary news indexing failures')
                except Exception: self._news_weaviate_primary_failures = None
            if not hasattr(self, '_news_weaviate_primary_latency'):
                try:
                    self._news_weaviate_primary_latency = _PH('news_weaviate_primary_latency_seconds','Primary news indexing latency (successful attempts)')
                except Exception: self._news_weaviate_primary_latency = None
            if not hasattr(self, '_news_weaviate_counter'):
                try:
                    self._news_weaviate_counter = _PC('news_weaviate_indexed_total','Total news articles indexed into Weaviate')
                except Exception: self._news_weaviate_counter = None
            if not hasattr(self, '_news_weaviate_fallback_invocations'):
                try:
                    self._news_weaviate_fallback_invocations = _PC('news_weaviate_fallback_invocations_total','Fallback indexing invocations')
                except Exception: self._news_weaviate_fallback_invocations = None
            if not hasattr(self, '_news_weaviate_last_failure_ts'):
                try:
                    self._news_weaviate_last_failure_ts = _PG('news_weaviate_last_failure_timestamp_seconds','Epoch seconds of last primary indexing failure')
                except Exception: self._news_weaviate_last_failure_ts = None
        except Exception:
            pass

        for attempt in (0,1):
            start = time.monotonic()
            try:
                if getattr(self, '_news_weaviate_primary_attempts', None):
                    try: self._news_weaviate_primary_attempts.inc()  # type: ignore[union-attr]
                    except Exception: pass
                async with self.session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=25)) as resp:
                    if resp.status == 200:
                        # Latency & count metrics
                        if getattr(self, '_news_weaviate_primary_latency', None):
                            try: self._news_weaviate_primary_latency.observe(max(0.0, time.monotonic()-start))  # type: ignore[union-attr]
                            except Exception: pass
                        if getattr(self, '_news_weaviate_counter', None):
                            try: self._news_weaviate_counter.inc(len(items))  # type: ignore[union-attr]
                            except Exception: pass
                        return
                    raise RuntimeError(f"HTTP {resp.status}")
            except Exception:
                if getattr(self, '_news_weaviate_primary_failures', None):
                    try: self._news_weaviate_primary_failures.inc()  # type: ignore[union-attr]
                    except Exception: pass
                if getattr(self, '_news_weaviate_last_failure_ts', None):
                    try: self._news_weaviate_last_failure_ts.set(time.time())  # type: ignore[union-attr]
                    except Exception: pass
                if attempt == 0:
                    try: await asyncio.sleep(0.5)
                    except Exception: pass
                continue
        # Fallback path
        try:
            from shared.vector.indexing import index_news_fallback  # type: ignore
            redis = None
            try:
                if self.cache:
                    redis = getattr(self.cache, 'redis', None) or self.cache
            except Exception:
                redis = None
            inserted = 0
            try:
                inserted = await index_news_fallback(list(payload), redis=redis)  # type: ignore[arg-type]
            except Exception:
                inserted = 0
            if getattr(self, '_news_weaviate_fallback_invocations', None):
                try: self._news_weaviate_fallback_invocations.inc()  # type: ignore[union-attr]
                except Exception: pass
            if inserted:
                try:
                    from prometheus_client import Counter as _PC2  # type: ignore
                    if not hasattr(self, '_news_weaviate_fallback_counter'):
                        try:
                            self._news_weaviate_fallback_counter = _PC2('news_weaviate_fallback_indexed_total','Total news articles indexed via fallback direct path')
                        except Exception:
                            self._news_weaviate_fallback_counter = None
                    if getattr(self, '_news_weaviate_fallback_counter', None):
                        self._news_weaviate_fallback_counter.inc(inserted)  # type: ignore[union-attr]
                except Exception:
                    pass
        except Exception:
            return
    
    def _is_financial_news(self, text: str) -> bool:
        """Check if text contains financial keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.financial_keywords)
    
    def _calculate_relevance(
        self, 
        article: Dict, 
        symbols: Optional[List[str]] = None
    ) -> float:
        """Calculate relevance score for an article."""
        if not symbols:
            return 0.5
        
        title = article.get('title', '').upper()
        description = article.get('description', '').upper()
        text = title + ' ' + description
        
        # Count symbol mentions
        symbol_mentions = sum(1 for symbol in symbols if symbol in text)
        
        # Base relevance
        relevance = min(symbol_mentions / len(symbols), 1.0)
        
        # Boost for financial keywords
        financial_mentions = sum(1 for keyword in self.financial_keywords 
                               if keyword.upper() in text)
        relevance += min(financial_mentions * 0.1, 0.3)
        
        return min(relevance, 1.0)
    
    def _extract_symbols(
        self, 
        text: str, 
        possible_symbols: Optional[List[str]] = None
    ) -> List[str]:
        """Extract stock symbols mentioned in text."""
        if not possible_symbols:
            return []
        
        text_upper = text.upper()
        mentioned_symbols = []
        
        for symbol in possible_symbols:
            if symbol in text_upper:
                mentioned_symbols.append(symbol)
        
        return mentioned_symbols
    
    def _deduplicate_news(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news articles."""
        seen_urls = set()
        seen_titles = set()
        unique_news = []
        
        for article in news_list:
            # Skip if URL already seen
            if article.url and article.url in seen_urls:
                continue
            
            # Skip if title is very similar (simple check)
            title_words = set(article.title.lower().split())
            is_duplicate = False
            
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if len(title_words.intersection(seen_words)) / max(len(title_words), len(seen_words)) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_news.append(article)
                if article.url:
                    seen_urls.add(article.url)
                seen_titles.add(article.title.lower())
        
        return unique_news
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using AI models."""
        try:
            prompt = f"""
            Analyze the sentiment of this financial news text and provide a sentiment score from -1 (very negative) to +1 (very positive):
            
            Text: {text[:1000]}  # Limit text length
            
            Consider:
            - Positive indicators: growth, profit, bullish, upgrade, beat expectations
            - Negative indicators: loss, decline, bearish, downgrade, miss expectations
            - Neutral indicators: mixed signals, uncertain outlook
            
            Provide only a decimal number between -1 and 1.
            """
            
            response = await generate_response(
                prompt, 
                model_preference=[ModelType.LOCAL_OLLAMA]  # Only local models
            )
            
            # Extract numeric score from response
            score_text = response.content.strip()
            try:
                score = float(score_text)
                return max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
            except ValueError:
                # Fallback: simple keyword-based sentiment
                return self._simple_sentiment_analysis(text)
                
        except Exception as e:
            logger.warning(f"AI sentiment analysis failed: {e}, using fallback")
            return self._simple_sentiment_analysis(text)
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """Simple keyword-based sentiment analysis fallback."""
        text_lower = text.lower()
        
        positive_words = [
            'profit', 'growth', 'increase', 'bullish', 'upgrade', 'beat', 'strong',
            'positive', 'gain', 'rally', 'boost', 'surge', 'soar', 'outperform'
        ]
        
        negative_words = [
            'loss', 'decline', 'bearish', 'downgrade', 'miss', 'weak', 'negative',
            'drop', 'fall', 'crash', 'plunge', 'underperform', 'concern', 'risk'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0
        
        total_words = positive_count + negative_count
        return (positive_count - negative_count) / total_words
    
    async def _publish_news(self, news_item: NewsItem):
        """Publish news item to message stream."""
        if not self.producer:
            return
        allowed = await self._pulsar_bucket.acquire(wait=False)
        if not allowed:
            try:
                self._pulsar_bucket.record_result(False, 0.0)
            except Exception:
                pass
            if getattr(self, '_publish_skipped', None):
                try:
                    self._publish_skipped.labels(service='news', reason='rate_limited').inc()
                except Exception:
                    pass
            return
        start = datetime.utcnow().timestamp()
        try:
            try:
                payload = news_item.model_dump(mode='json')  # pydantic v2
            except Exception:  # noqa: BLE001
                payload = news_item.dict()  # pydantic v1 fallback
            message = json.dumps(payload, default=str)
            async with self._pulsar_breaker.context():
                self.producer.send(message.encode('utf-8'))
        except Exception as e:  # noqa: BLE001
            if self._pulsar_error_counter:
                try:
                    self._pulsar_error_counter.inc()
                except Exception:
                    pass
            try:
                self._pulsar_bucket.record_result(False, max(0.0, datetime.utcnow().timestamp() - start))
            except Exception:
                pass
            self._rate_limited_pulsar_error(f"Failed to publish news item: {e}")
        else:
            try:
                self._pulsar_bucket.record_result(True, max(0.0, datetime.utcnow().timestamp() - start))
                snap = self._pulsar_bucket.metrics_snapshot()
                if self._bucket_rate_g:
                    self._bucket_rate_g.labels(service='news', bucket='pulsar_producer').set(snap['rate'])
                if self._bucket_tokens_g:
                    self._bucket_tokens_g.labels(service='news', bucket='pulsar_producer').set(snap['tokens'])
            except Exception:
                pass

    async def _maybe_persist_news(self, news_item: NewsItem) -> None:
        """Persist news to QuestDB table `news_items` if enabled.

        Table schema (implicitly created by ILP on first write):
          news_items
            SYMBOLS: symbol (string, optional for fan-out)
            COLUMNS: title, source, url, sentiment (float), relevance (float)
                     published (timestamp as column), provider (string)
          designated timestamp: ts (from published_at)
        """
        if not self.enable_questdb_persist:
            return
        ilp_ready = bool(self._qdb_sender and self._qdb_conf and self._qdb_ts)
        # Structured or plain precheck log
        try:
            if self.enable_structured_logs:
                logger.info(
                    "news_persist_precheck",
                    extra={
                        "event": "news_persist_precheck",
                        "ilp_ready": ilp_ready,
                        "conf": self._qdb_conf,
                        "has_ts_class": bool(self._qdb_ts),
                    }
                )
            else:
                logger.debug("news.persist.precheck ilp_ready=%s conf=%s", ilp_ready, self._qdb_conf)
        except Exception:
            pass
        symbols = news_item.symbols or ['ALL']
        # Compute value_score locally (NewsItem model does not declare it)
        try:
            value_score = float(self._compute_value_score(news_item))
        except Exception:
            value_score = 0.0
        persisted_via = None
        # ILP path
        if ilp_ready:
            try:
                with self._qdb_sender.from_conf(self._qdb_conf) as s:  # type: ignore[arg-type]
                    for sym in symbols:
                        at_ts = self._qdb_ts.from_datetime(news_item.published_at)  # type: ignore[union-attr]
                        s.row(
                            'news_items',
                            symbols={'symbol': sym.upper()},
                            columns={
                                'title': str(news_item.title or '')[:300],
                                'source': str(news_item.source or ''),
                                'url': str(news_item.url or ''),
                                'sentiment': float(news_item.sentiment_score or 0.0),
                                'relevance': float(news_item.relevance_score or 0.0),
                                'provider': str(news_item.source or ''),
                                'value_score': float(value_score),
                            },
                            at=at_ts,
                        )
                    s.flush()
                persisted_via = 'ilp'
            except Exception as e:  # noqa: BLE001
                try:
                    logger.warning("News ILP persist failed: %s", e)
                except Exception:
                    pass
        # HTTP fallback if ILP not ready or failed
        if not persisted_via:
            try:
                import urllib.parse as _up, urllib.request as _ur
                http_url = os.getenv('QUESTDB_HTTP_URL', f"http://{os.getenv('QUESTDB_HOST','127.0.0.1')}:9000/exec")
                stmts: list[str] = []
                # Basic escaping for single quotes
                title = (news_item.title or '').replace("'", "''")[:300]
                source = (news_item.source or '').replace("'", "''")
                url = (news_item.url or '').replace("'", "''")
                prov = source
                sent = float(news_item.sentiment_score or 0.0)
                rel = float(news_item.relevance_score or 0.0)
                # Use RFC3339 and wrap with to_timestamp() to avoid implicit cast warnings
                ts_iso = news_item.published_at.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                ts_expr = f"to_timestamp('{ts_iso}','yyyy-MM-ddTHH:mm:ss.SSSZ')"
                for sym in symbols:
                    symu = sym.upper().replace("'", "''")
                    stmt = (
                        "INSERT INTO news_items (symbol, ts, title, source, url, sentiment, relevance, provider, value_score) VALUES ("
                        f"'{symu}', {ts_expr}, '{title}', '{source}', '{url}', {sent}, {rel}, '{prov}', {float(value_score)})"
                    )
                    stmts.append(stmt)
                batch_sql = ';'.join(stmts)
                params = {'query': batch_sql}
                q = _up.urlencode(params)
                url_full = http_url + ("&" if "?" in http_url else "?") + q
                try:
                    with _ur.urlopen(url_full, timeout=15) as resp:  # nosec - internal
                        _ = resp.read()
                    persisted_via = 'http'
                except Exception as e1:  # retry without value_score if column missing
                    try:
                        # Rebuild statements without value_score column
                        stmts2: list[str] = []
                        for sym in symbols:
                            symu2 = sym.upper().replace("'", "''")
                            stmt2 = (
                                "INSERT INTO news_items (symbol, ts, title, source, url, sentiment, relevance, provider) VALUES ("
                                f"'{symu2}', {ts_expr}, '{title}', '{source}', '{url}', {sent}, {rel}, '{prov}')"
                            )
                            stmts2.append(stmt2)
                        batch_sql2 = ';'.join(stmts2)
                        q2 = _up.urlencode({'query': batch_sql2})
                        url_full2 = http_url + ("&" if "?" in http_url else "?") + q2
                        with _ur.urlopen(url_full2, timeout=15) as resp2:  # nosec - internal
                            _ = resp2.read()
                        persisted_via = 'http'
                    except Exception as e2:
                        raise e2 from e1
            except Exception as e:  # noqa: BLE001
                try:
                    logger.warning("News HTTP fallback persist failed: %s", e)
                except Exception:
                    pass
        if persisted_via:
            try:
                if self.enable_structured_logs:
                    logger.info("news_persist_success", extra={"event":"news_persist_success","via":persisted_via})
                else:
                    logger.debug("news.persist.success via=%s", persisted_via)
                if self._news_rows_total:
                    try:
                        self._news_rows_total.labels(sink=persisted_via).inc(len(symbols))
                    except Exception:
                        pass
                if self._news_last_persist_ts:
                    try:
                        self._news_last_persist_ts.set(time.time())
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            try:
                logger.warning("news.persist.failed all_paths")
                if self._news_persist_failures:
                    try:
                        self._news_persist_failures.labels(path='all').inc()
                    except Exception:
                        pass
            except Exception:
                pass
        # Optional Postgres sink for news (multi-sink) with idempotent upsert
        try:
            if self.enable_postgres_persist:
                from trading_common.database_manager import get_database_manager  # type: ignore
                # Use SQLAlchemy text() with AsyncSession (provided by DatabaseManager)
                try:
                    from sqlalchemy import text as sa_text  # type: ignore
                except Exception:
                    sa_text = None  # type: ignore
                dbm = await get_database_manager()
                async with dbm.get_postgres() as pg:  # pg is AsyncSession
                    # CREATE TABLE via exec_driver_sql/text depending on availability
                    create_sql = (
                        """
                        CREATE TABLE IF NOT EXISTS news_events (
                            id SERIAL PRIMARY KEY,
                            symbol TEXT NOT NULL,
                            published_at TIMESTAMPTZ NOT NULL,
                            title TEXT,
                            source TEXT,
                            url TEXT,
                            sentiment DOUBLE PRECISION,
                            relevance DOUBLE PRECISION,
                            provider TEXT,
                            value_score DOUBLE PRECISION DEFAULT 0,
                            UNIQUE(symbol, published_at, url)
                        )
                        """
                    )
                    try:
                        if sa_text is not None:
                            await pg.execute(sa_text(create_sql))
                        else:
                            # Fallback – most drivers accept exec_driver_sql on connection
                            await pg.connection().exec_driver_sql(create_sql)  # type: ignore[attr-defined]
                    except Exception:
                        # Best-effort: ignore DDL failure (table may exist)
                        pass

                    # Prepare row parameters for executemany-style execute
                    rows: list[dict] = []
                    for sym in symbols:
                        rows.append({
                            'symbol': sym.upper(),
                            'published_at': news_item.published_at,
                            'title': news_item.title,
                            'source': news_item.source,
                            'url': news_item.url,
                            'sentiment': float(news_item.sentiment_score or 0.0),
                            'relevance': float(news_item.relevance_score or 0.0),
                            'provider': str(news_item.source or ''),
                            'value_score': float(value_score),
                        })

                    if rows:
                        insert_sql = (
                            """
                            INSERT INTO news_events(
                                symbol, published_at, title, source, url, sentiment, relevance, provider, value_score
                            ) VALUES (
                                :symbol, :published_at, :title, :source, :url, :sentiment, :relevance, :provider, :value_score
                            )
                            ON CONFLICT(symbol, published_at, url) DO NOTHING
                            """
                        )
                        try:
                            if sa_text is not None:
                                await pg.execute(sa_text(insert_sql), rows)
                            else:
                                # Fallback to per-row insertion if text() is unavailable
                                for r in rows:
                                    await pg.execute(
                                        """
                                        INSERT INTO news_events(symbol, published_at, title, source, url, sentiment, relevance, provider, value_score)
                                        VALUES(:symbol,:published_at,:title,:source,:url,:sentiment,:relevance,:provider,:value_score)
                                        ON CONFLICT(symbol, published_at, url) DO NOTHING
                                        """,
                                        r,
                                    )
                        except Exception as e:
                            # As ultimate fallback, try driver-level exec_driver_sql if accessible
                            try:
                                conn = await pg.connection()  # type: ignore[attr-defined]
                                for r in rows:
                                    await conn.exec_driver_sql(
                                        """
                                        INSERT INTO news_events(symbol, published_at, title, source, url, sentiment, relevance, provider, value_score)
                                        VALUES(%(symbol)s,%(published_at)s,%(title)s,%(source)s,%(url)s,%(sentiment)s,%(relevance)s,%(provider)s,%(value_score)s)
                                        ON CONFLICT(symbol, published_at, url) DO NOTHING
                                        """,
                                        r,
                                    )
                            except Exception:
                                raise e

                    # Metrics counter
                    from prometheus_client import Counter as _PC  # type: ignore
                    if not hasattr(self, '_news_pg_counter'):
                        try:
                            self._news_pg_counter = _PC('news_postgres_rows_persisted_total','Total news rows persisted to Postgres')
                        except Exception:
                            self._news_pg_counter = None
                    if getattr(self, '_news_pg_counter', None):
                        try:
                            self._news_pg_counter.inc(len(rows))
                        except Exception:
                            pass
        except Exception as e:  # noqa: BLE001
            try:
                logger.warning(f"Postgres news persistence failed: {e}")
            except Exception:
                pass

    # ---------------- Value Scoring ---------------- #
    def _compute_value_score(self, article: NewsItem) -> float:
        """Compute a heuristic value score (0-1) for a news article.

        Components:
          - Relevance (weight 0.35)
          - Absolute sentiment magnitude (0.20)
          - Source credibility (0.20)
          - Symbol coverage breadth (0.15)
          - Novelty (0.10) – penalize highly similar titles already seen in session/cache (best-effort)
        Falls back gracefully; clamps to [0,1].
        """
        try:
            rel = float(getattr(article, 'relevance_score', 0.0) or 0.0)
        except Exception:
            rel = 0.0
        try:
            sent = float(abs(getattr(article, 'sentiment_score', 0.0) or 0.0))
        except Exception:
            sent = 0.0
        # Source credibility heuristic map; unknown default 0.4
        source = (getattr(article, 'source', '') or '').lower()
        credibility_map = {
            'reuters': 0.95,
            'bloomberg': 0.95,
            'wsj': 0.9,
            'wall street journal': 0.9,
            'financial times': 0.9,
            'cnbc': 0.8,
            'marketwatch': 0.75,
            'seekingalpha': 0.7,
        }
        cred = 0.4
        for k,v in credibility_map.items():
            if k in source:
                cred = v
                break
        # Symbol breadth (cap at 3)
        symbols = getattr(article, 'symbols', []) or []
        breadth = min(len(symbols), 3) / 3.0
        # Novelty: in-memory approximate duplicate detection (no async ops inside sync method)
        try:
            h = hash((article.title or '')[:160].lower())
            exists = h in self._seen_title_hashes
            if not exists:
                self._seen_title_hashes.append(h)
                # Bounded size
                if len(self._seen_title_hashes) > self._seen_title_max:
                    # Drop oldest ~10% to keep amortized O(1)
                    drop = max(1, self._seen_title_max // 10)
                    self._seen_title_hashes = self._seen_title_hashes[drop:]
            novelty = 0.5 if exists else 1.0
        except Exception:
            novelty = 1.0
        score = (
            0.35*rel +
            0.20*sent +
            0.20*cred +
            0.15*breadth +
            0.10*novelty
        )
        if score < 0:
            score = 0.0
        if score > 1:
            score = 1.0
        return score
    
    async def get_service_health(self) -> Dict:
        """Get service health status."""
        return {
            'service': 'news',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'data_sources': {
                'news_api': bool(self.news_api_config['api_key']),
                'finnhub': bool(self.finnhub_config['api_key']),
                'reddit': bool(self.reddit_config['client_id'])
            },
            'connections': {
                'http_session': self.session is not None,
                'cache': self.cache is not None,
                'message_producer': self.producer is not None
            }
        }

    # ---------------- Backlog reindexer & backfill helpers ---------------- #
    async def reindex_news_backlog(self, *, hours_back: int | None = None) -> int:
        """Scan Postgres news_events and reindex into Weaviate via ML service.

        Uses Redis (via trading_cache) to checkpoint last indexed published_at to avoid duplicates.
        Returns number of items sent to the indexer in this pass.
        """
        if not (self.enable_weaviate_persist and self.session):
            return 0
        try:
            from trading_common.database_manager import get_database_manager  # type: ignore
        except Exception:
            return 0
        last_ts_key = 'vector:news:last_indexed_ts'
        redis = None
        try:
            if self.cache:
                redis = getattr(self.cache, 'redis', None) or self.cache
        except Exception:
            redis = None
        since: datetime | None = None
        if hours_back is not None:
            since = datetime.utcnow() - timedelta(hours=max(1, hours_back))
        elif redis:
            try:
                val = await redis.get(last_ts_key)
                if val:
                    since = datetime.fromisoformat(val.decode() if isinstance(val, (bytes, bytearray)) else str(val))
            except Exception:
                since = None
        total_sent = 0
        try:
            dbm = await get_database_manager()
            async with dbm.get_postgres() as pg:  # AsyncSession
                # Pull in ascending published_at order; limit batch size using SQLAlchemy text()
                try:
                    from sqlalchemy import text as sa_text  # type: ignore
                except Exception:
                    sa_text = None  # type: ignore
                limit_n = max(1, self._news_reindex_batch)
                if sa_text is not None:
                    if since:
                        stmt = sa_text(
                            """
                            SELECT symbol, published_at, title, source, url, sentiment, relevance
                            FROM news_events
                            WHERE published_at > :since
                            ORDER BY published_at ASC
                            LIMIT :lim
                            """
                        )
                        result = await pg.execute(stmt, {"since": since, "lim": limit_n})
                    else:
                        stmt = sa_text(
                            """
                            SELECT symbol, published_at, title, source, url, sentiment, relevance
                            FROM news_events
                            ORDER BY published_at ASC
                            LIMIT :lim
                            """
                        )
                        result = await pg.execute(stmt, {"lim": limit_n})
                    rows = result.mappings().all()
                else:
                    # Fallback: try driver-level exec_driver_sql
                    conn = await pg.connection()  # type: ignore[attr-defined]
                    if since:
                        result = await conn.exec_driver_sql(
                            """
                            SELECT symbol, published_at, title, source, url, sentiment, relevance
                            FROM news_events
                            WHERE published_at > %(since)s
                            ORDER BY published_at ASC
                            LIMIT %(lim)s
                            """,
                            {"since": since, "lim": limit_n},
                        )
                    else:
                        result = await conn.exec_driver_sql(
                            """
                            SELECT symbol, published_at, title, source, url, sentiment, relevance
                            FROM news_events
                            ORDER BY published_at ASC
                            LIMIT %(lim)s
                            """,
                            {"lim": limit_n},
                        )
                    rows = [dict(r._mapping) for r in result]  # type: ignore[attr-defined]

                if not rows:
                    return 0
                # Group rows by url+ts to dedupe
                seen: set[tuple[str, str]] = set()
                items: list[NewsItem] = []
                last_pub: datetime | None = None
                for r in rows:
                    try:
                        url = r.get('url') or ''
                        ts = r.get('published_at')
                        if not ts:
                            continue
                        key = (str(url), ts.isoformat())
                        if key in seen:
                            continue
                        seen.add(key)
                        last_pub = ts
                        items.append(NewsItem(
                            title=r.get('title') or '',
                            content='',
                            source=r.get('source') or 'news',
                            published_at=ts,
                            url=url,
                            sentiment_score=float(r.get('sentiment') or 0.0),
                            relevance_score=float(r.get('relevance') or 0.5),
                            symbols=[r.get('symbol') or 'ALL']
                        ))
                    except Exception:
                        continue
                if not items:
                    return 0
                await self._index_news_to_weaviate(items)
                total_sent = len(items)
                # Advance checkpoint
                if redis and last_pub:
                    try:
                        await redis.set(last_ts_key, last_pub.isoformat())
                    except Exception:
                        pass
        except Exception as e:
            try:
                logger.warning("Backlog reindex execution failed: %s", e)
            except Exception:
                pass
        return total_sent

    async def _run_news_backfill_task(self) -> None:
        """Background task: historical backfill for last N years and index as we go."""
        try:
            # Choose symbols
            syms_raw = os.getenv('NEWS_BACKFILL_SYMBOLS', '')
            if syms_raw.strip():
                symbols = [s.strip().upper() for s in syms_raw.split(',') if s.strip()]
            else:
                fb = os.getenv('NEWS_FALLBACK_SYMBOLS', 'AAPL,MSFT,TSLA,NVDA,SPY')
                symbols = [s.strip().upper() for s in fb.split(',') if s.strip()]
            years = max(1, self._news_backfill_years)
            end = datetime.utcnow().date()
            start = datetime(end.year - years, end.month, end.day)
            total, batches = await self.collect_financial_news_range(symbols, datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()), batch_days=14, max_articles_per_batch=120, backfill_mode=True)
            try:
                logger.info("News historical backfill completed", total_articles=total, batches=len(batches))
            except Exception:
                pass
        except Exception as e:
            try:
                logger.warning("News historical backfill task failed: %s", e)
            except Exception:
                pass


# Global service instance
news_service: Optional[NewsService] = None


async def get_news_service() -> NewsService:
    """Get or create news service instance."""
    global news_service
    if news_service is None:
        news_service = NewsService()
        await news_service.start()
    return news_service
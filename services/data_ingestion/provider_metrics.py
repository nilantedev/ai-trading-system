#!/usr/bin/env python3
"""Shared provider metrics instrumentation for data ingestion services.

Centralizes Prometheus metric objects and helper functions to avoid duplicate
registrations and circular imports.
"""
from prometheus_client import Counter, Gauge, Histogram
import time
import functools
import asyncio
from typing import Callable, Any

# Counter: total requests per provider/endpoint with success vs error status
try:
    PROVIDER_REQUESTS_TOTAL = Counter(
        'provider_requests_total',
        'Total requests made to external data providers',
        ['provider', 'endpoint', 'status']  # status=success|error
    )
except Exception:  # noqa: BLE001
    PROVIDER_REQUESTS_TOTAL = None  # Already registered

# Attempts counter (includes successes + failures per logical call)
try:
    PROVIDER_ATTEMPTS_TOTAL = Counter(
        'provider_attempts_total',
        'Total attempt count for provider operations (includes retries)',
        ['provider', 'endpoint']
    )
except Exception:  # noqa: BLE001
    PROVIDER_ATTEMPTS_TOTAL = None

# Latency histogram per provider/endpoint (seconds) includes both success and error paths
try:
    PROVIDER_REQUEST_LATENCY_SECONDS = Histogram(
        'provider_request_latency_seconds',
        'Latency of external provider requests (seconds)',
        ['provider', 'endpoint', 'status'],
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21)
    )
except Exception:  # noqa: BLE001
    PROVIDER_REQUEST_LATENCY_SECONDS = None

# Gauge: backfill progress percent (0-100). Single series per service context.
try:
    BACKFILL_PROGRESS_PERCENT = Gauge(
        'historical_backfill_progress_percent',
        'Historical backfill completion percentage (0-100)',
        []
    )
except Exception:  # noqa: BLE001
    BACKFILL_PROGRESS_PERCENT = None

# Counter: total timeouts per provider/endpoint
try:
    PROVIDER_TIMEOUT_TOTAL = Counter(
        'provider_timeout_total',
        'Total timeouts encountered when calling external providers',
        ['provider', 'endpoint']
    )
except Exception:  # noqa: BLE001
    PROVIDER_TIMEOUT_TOTAL = None

# Counter: raw HTTP response codes observed per provider/endpoint
try:
    PROVIDER_HTTP_RESPONSES_TOTAL = Counter(
        'provider_http_responses_total',
        'HTTP responses by code for external provider requests',
        ['provider', 'endpoint', 'code']
    )
except Exception:  # noqa: BLE001
    PROVIDER_HTTP_RESPONSES_TOTAL = None

# Counter: explicit rate-limit events (e.g., HTTP 429 or vendor-specific throttle notes)
try:
    PROVIDER_RATE_LIMIT_TOTAL = Counter(
        'provider_rate_limit_total',
        'Total rate-limit events encountered (HTTP 429 or vendor-specific)',
        ['provider', 'endpoint']
    )
except Exception:  # noqa: BLE001
    PROVIDER_RATE_LIMIT_TOTAL = None

def record_provider_request(provider: str, endpoint: str, success: bool, start_time: float | None = None):
    """Increment provider request counter safely (and observe latency if start_time).

    Parameters
    ----------
    provider: str
        Provider name (e.g., 'alpaca', 'polygon').
    endpoint: str
        Logical endpoint identifier (e.g., 'quote', 'daily_bars').
    success: bool
        Whether the request succeeded (semantic success, not just HTTP 200 if parsing failed).
    start_time: float | None
        Monotonic start time; if provided, latency will be recorded.
    """
    if PROVIDER_ATTEMPTS_TOTAL:
        try:
            PROVIDER_ATTEMPTS_TOTAL.labels(provider=provider, endpoint=endpoint).inc()
        except Exception:  # noqa: BLE001
            pass
    if PROVIDER_REQUESTS_TOTAL:
        try:
            PROVIDER_REQUESTS_TOTAL.labels(
                provider=provider,
                endpoint=endpoint,
                status='success' if success else 'error'
            ).inc()
        except Exception:  # noqa: BLE001
            pass
    if start_time is not None and PROVIDER_REQUEST_LATENCY_SECONDS:
        try:
            PROVIDER_REQUEST_LATENCY_SECONDS.labels(
                provider=provider,
                endpoint=endpoint,
                status='success' if success else 'error'
            ).observe(max(0.0, time.monotonic() - start_time))
        except Exception:  # noqa: BLE001
            pass

def provider_instrumentation(provider: str, endpoint: str):
    """Decorator to standardize provider call instrumentation.

    Records attempts, success/error counters, and latency for both outcomes.
    """
    def _decorate(func: Callable[..., Any]):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def _async_wrapper(*args, **kwargs):
                start = time.monotonic()
                try:
                    result = await func(*args, **kwargs)
                except Exception:  # noqa: BLE001
                    record_provider_request(provider, endpoint, False, start_time=start)
                    raise
                else:
                    record_provider_request(provider, endpoint, True, start_time=start)
                    return result
            return _async_wrapper
        else:
            @functools.wraps(func)
            def _sync_wrapper(*args, **kwargs):
                start = time.monotonic()
                try:
                    result = func(*args, **kwargs)
                except Exception:  # noqa: BLE001
                    record_provider_request(provider, endpoint, False, start_time=start)
                    raise
                else:
                    record_provider_request(provider, endpoint, True, start_time=start)
                    return result
            return _sync_wrapper
    return _decorate

def set_backfill_progress(percent: float):
    """Set backfill progress percent gauge (clamped 0-100)."""
    if BACKFILL_PROGRESS_PERCENT:
        try:
            if percent < 0:
                percent = 0.0
            elif percent > 100:
                percent = 100.0
            BACKFILL_PROGRESS_PERCENT.set(percent)
        except Exception:  # noqa: BLE001
            pass

def record_provider_timeout(provider: str, endpoint: str):
    """Increment provider timeout counter safely."""
    if PROVIDER_TIMEOUT_TOTAL:
        try:
            PROVIDER_TIMEOUT_TOTAL.labels(provider=provider, endpoint=endpoint).inc()
        except Exception:  # noqa: BLE001
            pass

def record_http_response(provider: str, endpoint: str, code: int | str):
    """Record a raw HTTP response code for diagnostics and alerting."""
    if PROVIDER_HTTP_RESPONSES_TOTAL:
        try:
            PROVIDER_HTTP_RESPONSES_TOTAL.labels(provider=provider, endpoint=endpoint, code=str(code)).inc()
        except Exception:  # noqa: BLE001
            pass

def record_rate_limit(provider: str, endpoint: str):
    """Record a rate-limit event (HTTP 429 or vendor throttle notice)."""
    if PROVIDER_RATE_LIMIT_TOTAL:
        try:
            PROVIDER_RATE_LIMIT_TOTAL.labels(provider=provider, endpoint=endpoint).inc()
        except Exception:  # noqa: BLE001
            pass

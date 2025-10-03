#!/usr/bin/env python3
"""
Prometheus Metrics for AI Trading System API
Provides comprehensive monitoring and observability metrics.
"""

import time
import logging
import os
from typing import Optional, Dict, Any
from fastapi import Request, Response
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    multiprocess
)
try:  # Prefer shared canonical app_* metrics if observability module present
    from observability import (
        HTTP_REQUESTS as APP_HTTP_REQUESTS,
        HTTP_LATENCY as APP_HTTP_LATENCY,
        INFLIGHT as APP_HTTP_INFLIGHT,
        CONCURRENCY_LIMIT as APP_CONCURRENCY_LIMIT,
        QUEUE_DEPTH as APP_QUEUE_DEPTH,
    )  # type: ignore
    _HAS_SHARED_APP_METRICS = True
except Exception:  # noqa: BLE001 - Fallback to local only (will add aliases)
    _HAS_SHARED_APP_METRICS = False
import asyncio
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Create custom registry for multi-process support
if os.environ.get('PROMETHEUS_MULTIPROC_DIR'):
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
else:
    registry = None


"""Local metrics + canonical app_* exposure.

If the shared observability module (shared/python-common/observability.py) is available,
we rely on those canonical metrics (app_http_requests_total, etc.). If not (API running
standalone or path import issue) we create lightweight alias metrics with canonical names
so the health script sees them, emitting zero-valued gauges where meaningful.
"""

# Primary (service-local) HTTP metrics remain for backward compatibility / legacy dashboards
http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

# Canonical app_* metrics (aliases or standalone fallbacks)
if not _HAS_SHARED_APP_METRICS:
    # Define minimal canonical metrics so health check stops warning
    APP_HTTP_REQUESTS = Counter(
        'app_http_requests_total',
        'Canonical total HTTP requests (fallback local definition)',
        ['service', 'method', 'path', 'status'],
        registry=registry
    )
    APP_HTTP_LATENCY = Histogram(
        'app_http_request_latency_seconds',
        'Canonical HTTP request latency (fallback local definition)',
        ['service', 'method', 'path'],
        buckets=[0.005,0.01,0.02,0.05,0.1,0.25,0.5,1,2,5],
        registry=registry
    )
    APP_HTTP_INFLIGHT = Gauge(
        'app_http_inflight_requests',
        'In-flight HTTP requests (fallback local definition)',
        ['service'],
        registry=registry
    )
    APP_CONCURRENCY_LIMIT = Gauge(
        'app_concurrency_limit',
        'Configured concurrency limit (fallback local definition)',
        ['service'],
        registry=registry
    )
    APP_QUEUE_DEPTH = Gauge(
        'app_request_queue_depth',
        'Queued requests awaiting concurrency slot (fallback local definition)',
        ['service'],
        registry=registry
    )
else:
    # Map to names used later for uniform access
    APP_HTTP_INFLIGHT = APP_HTTP_INFLIGHT  # type: ignore
    APP_CONCURRENCY_LIMIT = APP_CONCURRENCY_LIMIT  # type: ignore
    APP_QUEUE_DEPTH = APP_QUEUE_DEPTH  # type: ignore

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

# Authentication Metrics
auth_requests_total = Counter(
    'auth_requests_total',
    'Total number of authentication requests',
    ['endpoint', 'status'],
    registry=registry
)

auth_failures_total = Counter(
    'auth_failures_total',
    'Total number of authentication failures',
    ['reason'],
    registry=registry
)

active_sessions = Gauge(
    'active_sessions_current',
    'Current number of active user sessions',
    registry=registry
)

# Rate Limiting Metrics
rate_limit_requests_total = Counter(
    'rate_limit_requests_total',
    'Total number of requests checked by rate limiter',
    ['limit_type', 'status'],
    registry=registry
)

rate_limit_blocks_total = Counter(
    'rate_limit_blocks_total',
    'Total number of requests blocked by rate limiter',
    ['limit_type', 'identifier_type'],
    registry=registry
)

# Degraded / fallback mode indicators
rate_limiter_degraded = Gauge(
    'rate_limiter_degraded',
    'Rate limiter degraded mode active (1=yes,0=no)',
    registry=registry
)

rate_limiter_fallback_requests_total = Counter(
    'rate_limiter_fallback_requests_total',
    'Total requests processed under degraded/fallback mode',
    ['status'],
    registry=registry
)

# Object Storage (MinIO) Metrics
object_storage_operations_total = Counter(
    'object_storage_operations_total',
    'Total MinIO storage operations',
    ['operation', 'status'],
    registry=registry
)

object_storage_operation_duration_seconds = Histogram(
    'object_storage_operation_duration_seconds',
    'Latency of MinIO storage operations in seconds',
    ['operation'],
    buckets=[0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2,5],
    registry=registry
)

# Vector Store (Weaviate) Metrics
vector_store_operations_total = Counter(
    'vector_store_operations_total',
    'Total Weaviate operations',
    ['operation', 'status'],
    registry=registry
)

vector_store_operation_duration_seconds = Histogram(
    'vector_store_operation_duration_seconds',
    'Latency of Weaviate operations in seconds',
    ['operation'],
    buckets=[0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2,5],
    registry=registry
)

# WebSocket Metrics
websocket_connections_total = Counter(
    'websocket_connections_total',
    'Total number of WebSocket connections',
    ['stream_type', 'status'],
    registry=registry
)

websocket_connections_current = Gauge(
    'websocket_connections_current',
    'Current number of active WebSocket connections',
    ['stream_type'],
    registry=registry
)

websocket_messages_total = Counter(
    'websocket_messages_total',
    'Total number of WebSocket messages sent',
    ['stream_type', 'message_type'],
    registry=registry
)

websocket_errors_total = Counter(
    'websocket_errors_total',
    'Total number of WebSocket errors',
    ['stream_type', 'error_type'],
    registry=registry
)

# WebSocket Broadcast Metrics
websocket_broadcast_total = Counter(
    'websocket_broadcast_total',
    'Total number of WebSocket broadcast operations',
    ['result'],
    registry=registry
)

websocket_broadcast_messages = Histogram(
    'websocket_broadcast_messages',
    'Number of messages (fanout recipients) per broadcast',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
    registry=registry
)

websocket_broadcast_duration_seconds = Histogram(
    'websocket_broadcast_duration_seconds',
    'Duration of WebSocket broadcast operations in seconds',
    buckets=[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry
)

# Trading System Metrics
market_data_points_total = Counter(
    'market_data_points_total',
    'Total number of market data points processed',
    ['symbol', 'source'],
    registry=registry
)

trading_signals_total = Counter(
    'trading_signals_total',
    'Total number of trading signals generated',
    ['symbol', 'signal_type', 'strategy'],
    registry=registry
)

orders_total = Counter(
    'orders_total',
    'Total number of orders',
    ['symbol', 'order_type', 'status'],
    registry=registry
)

portfolio_value_current = Gauge(
    'portfolio_value_current',
    'Current portfolio value',
    registry=registry
)

# System Health Metrics
system_info = Info(
    'system_info',
    'System information',
    registry=registry
)

service_health_status = Gauge(
    'service_health_status',
    'Service health status (1=healthy, 0=unhealthy)',
    ['service_name'],
    registry=registry
)

database_connections_current = Gauge(
    'database_connections_current',
    'Current number of database connections',
    ['database_type'],
    registry=registry
)

external_api_requests_total = Counter(
    'external_api_requests_total',
    'Total number of external API requests',
    ['provider', 'endpoint', 'status'],
    registry=registry
)

external_api_duration_seconds = Histogram(
    'external_api_duration_seconds',
    'External API request duration in seconds',
    ['provider', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    registry=registry
)

# Cache Metrics
cache_operations_total = Counter(
    'cache_operations_total',
    'Total number of cache operations',
    ['operation', 'status'],
    registry=registry
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio',
    registry=registry
)

# Error Metrics
errors_total = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'severity'],
    registry=registry
)

# Background Task Metrics
background_tasks_total = Counter(
    'background_tasks_total',
    'Total number of background tasks executed',
    ['task_type', 'status'],
    registry=registry
)

background_task_duration_seconds = Histogram(
    'background_task_duration_seconds',
    'Background task duration in seconds',
    ['task_type'],
    registry=registry
)

# Historical coverage gauges (populated by admin verification endpoints)
historical_dataset_coverage_ratio = Gauge(
    'historical_dataset_coverage_ratio',
    'Approx distinct trading day coverage ratio relative to target horizon (per dataset)',
    ['dataset'],
    registry=registry
)
historical_dataset_row_count = Gauge(
    'historical_dataset_row_count',
    'Row count per historical dataset',
    ['dataset'],
    registry=registry
)
historical_dataset_last_timestamp_seconds = Gauge(
    'historical_dataset_last_timestamp_seconds',
    'Last observed timestamp (epoch seconds) per dataset',
    ['dataset'],
    registry=registry
)
historical_dataset_span_days = Gauge(
    'historical_dataset_span_days',
    'Span in days between first and last timestamp per dataset',
    ['dataset'],
    registry=registry
)


CANONICAL_FLAG = os.getenv('ENABLE_METRICS_CANONICAL_NAMES', '0') in ('1', 'true', 'True', 'YES', 'yes')

# Helper wrappers for dual-publish
def _dual_counter(primary: Counter, canonical_name: str, documentation: str, labelnames: tuple):
    if not CANONICAL_FLAG:
        return primary, None
    canonical = Counter(canonical_name, documentation + ' (canonical alias)', labelnames, registry=registry)
    return primary, canonical

def _dual_histogram(primary: Histogram, canonical_name: str, documentation: str, labelnames: tuple, **kwargs):
    if not CANONICAL_FLAG:
        return primary, None
    buckets = kwargs.get('buckets') or primary._upper_bounds  # reuse
    canonical = Histogram(canonical_name, documentation + ' (canonical alias)', labelnames, buckets=buckets, registry=registry)
    return primary, canonical

def _observe(counter_pair, labels: Dict[str, str], value=None, histogram=False):
    primary, alias = counter_pair
    if histogram:
        primary.labels(**labels).observe(value)
        if alias:
            alias.labels(**labels).observe(value)
    else:
        primary.labels(**labels).inc(value if value is not None else 1)
        if alias:
            alias.labels(**labels).inc(value if value is not None else 1)

# Keep original metric objects (already defined above) and add canonical alias pairs
try:
    http_requests_pair = (http_requests_total, None)
    http_duration_pair = (http_request_duration_seconds, None)
    if CANONICAL_FLAG:
        # Define canonical alias metrics (prefixed variants)
        http_requests_pair = _dual_counter(http_requests_total, 'api_http_requests_total', 'Total number of HTTP requests', ('method','endpoint','status_code'))
        http_duration_pair = _dual_histogram(http_request_duration_seconds, 'api_http_request_duration_seconds', 'HTTP request duration in seconds', ('method','endpoint'))
except Exception as e:
    logging.getLogger(__name__).warning(f"Canonical metrics setup issue: {e}")


class MetricsCollector:
    """Metrics collection and management."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_start_times = {}
        
        # Initialize system info
        system_info.info({
            'version': '1.0.0',
            'python_version': os.sys.version,
            'started_at': str(int(self.start_time))
        })
    
    async def record_http_request(self, request: Request, response: Response, duration: float):
        """Record HTTP request metrics."""
        method = request.method
        endpoint = self._get_endpoint_label(request.url.path)
        status_code = str(response.status_code)
        # Primary + alias counter
        _observe(http_requests_pair, {'method': method, 'endpoint': endpoint, 'status_code': status_code})
        # Also emit canonical app_* metrics (service label = api) ensuring low cardinality path label
        try:
            path_label = endpoint  # Already normalized above
            APP_HTTP_REQUESTS.labels(service='api', method=method, path=path_label, status=status_code).inc()  # type: ignore[arg-type]
            APP_HTTP_LATENCY.labels(service='api', method=method, path=path_label).observe(duration)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass
        # Primary + alias histogram
        primary, alias = http_duration_pair
        primary.labels(method=method, endpoint=endpoint).observe(duration)
        if alias:
            alias.labels(method=method, endpoint=endpoint).observe(duration)
        
        # Record request size (avoid using protected _body attribute)
        try:
            # Prefer Content-Length header if provided
            cl = request.headers.get('content-length')
            if cl is not None:
                request_size = int(cl)
            else:
                # Fallback: read body once (only for reasonably small payloads)
                # NOTE: Reading the body will consume the stream; in most FastAPI usage the body
                # is already read by downstream (e.g., JSON parsing). We defensively limit.
                BODY_INSPECTION_LIMIT = 64 * 1024  # 64KB safeguard
                body = await request.body()
                if len(body) > BODY_INSPECTION_LIMIT:
                    request_size = BODY_INSPECTION_LIMIT
                else:
                    request_size = len(body)
            http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(request_size)
        except Exception:
            # Swallow any issues calculating size to avoid impacting request flow
            pass
        
        # Record response size
        content_length = response.headers.get('content-length')
        if content_length:
            http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint
            ).observe(int(content_length))
    
    def record_auth_attempt(self, endpoint: str, success: bool, failure_reason: Optional[str] = None):
        """Record authentication attempt."""
        status = 'success' if success else 'failure'
        auth_requests_total.labels(endpoint=endpoint, status=status).inc()
        
        if not success and failure_reason:
            auth_failures_total.labels(reason=failure_reason).inc()
    
    def update_active_sessions(self, count: int):
        """Update active sessions count."""
        active_sessions.set(count)
    
    def record_rate_limit_check(self, limit_type: str, allowed: bool, identifier_type: str = 'ip'):
        """Record rate limit check."""
        status = 'allowed' if allowed else 'blocked'
        rate_limit_requests_total.labels(limit_type=limit_type, status=status).inc()
        
        if not allowed:
            rate_limit_blocks_total.labels(
                limit_type=limit_type,
                identifier_type=identifier_type
            ).inc()
    
    def record_websocket_connection(self, stream_type: str, connected: bool):
        """Record WebSocket connection event."""
        status = 'connected' if connected else 'disconnected'
        websocket_connections_total.labels(stream_type=stream_type, status=status).inc()
    
    def update_websocket_connections(self, stream_type: str, count: int):
        """Update current WebSocket connections count."""
        websocket_connections_current.labels(stream_type=stream_type).set(count)
    
    def record_websocket_message(self, stream_type: str, message_type: str):
        """Record WebSocket message sent."""
        websocket_messages_total.labels(
            stream_type=stream_type,
            message_type=message_type
        ).inc()
    
    def record_websocket_error(self, stream_type: str, error_type: str):
        """Record WebSocket error."""
        websocket_errors_total.labels(
            stream_type=stream_type,
            error_type=error_type
        ).inc()

    # --- WebSocket Broadcast ---
    def record_websocket_broadcast(self, connections_count: int, successful_sends: int, failed_sends: int, broadcast_time: float):
        """Record a broadcast fanout event."""
        result = 'success' if failed_sends == 0 else ('partial' if successful_sends > 0 else 'failed')
        websocket_broadcast_total.labels(result=result).inc()
        websocket_broadcast_messages.observe(connections_count)
        websocket_broadcast_duration_seconds.observe(broadcast_time)
        # Additional derived gauges could be added later (success ratio, etc.)
    
    def record_market_data(self, symbol: str, source: str):
        """Record market data point processed."""
        market_data_points_total.labels(symbol=symbol, source=source).inc()
    
    def record_trading_signal(self, symbol: str, signal_type: str, strategy: str):
        """Record trading signal generated."""
        trading_signals_total.labels(
            symbol=symbol,
            signal_type=signal_type,
            strategy=strategy
        ).inc()
    
    def record_order(self, symbol: str, order_type: str, status: str):
        """Record order event."""
        orders_total.labels(
            symbol=symbol,
            order_type=order_type,
            status=status
        ).inc()
    
    def update_portfolio_value(self, value: float):
        """Update current portfolio value."""
        portfolio_value_current.set(value)
    
    def update_service_health(self, service_name: str, healthy: bool):
        """Update service health status."""
        service_health_status.labels(service_name=service_name).set(1 if healthy else 0)
    
    def update_database_connections(self, database_type: str, count: int):
        """Update database connections count."""
        database_connections_current.labels(database_type=database_type).set(count)
    
    def record_external_api_request(self, provider: str, endpoint: str, duration: float, success: bool):
        """Record external API request."""
        status = 'success' if success else 'error'
        prov_label = self._normalize_provider(provider)
        ep_label = self._normalize_endpoint(endpoint)
        external_api_requests_total.labels(
            provider=prov_label,
            endpoint=ep_label,
            status=status
        ).inc()
        external_api_duration_seconds.labels(
            provider=prov_label,
            endpoint=ep_label
        ).observe(duration)
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation."""
        status = 'hit' if hit else 'miss'
        cache_operations_total.labels(operation=operation, status=status).inc()
    
    def update_cache_hit_ratio(self, ratio: float):
        """Update cache hit ratio."""
        cache_hit_ratio.set(ratio)
    
    def record_error(self, error_type: str, severity: str = 'error'):
        """Record error occurrence."""
        errors_total.labels(error_type=error_type, severity=severity).inc()
    
    def record_background_task(self, task_type: str, duration: float, success: bool):
        """Record background task execution."""
        status = 'success' if success else 'error'
        background_tasks_total.labels(task_type=task_type, status=status).inc()
        background_task_duration_seconds.labels(task_type=task_type).observe(duration)
    
    def _get_endpoint_label(self, path: str) -> str:
        """Get normalized endpoint label for metrics."""
        # Normalize path for metrics (remove dynamic parts)
        if path.startswith('/api/v1/'):
            parts = path.split('/')
            if len(parts) > 3:
                # Keep first 3 parts: ['', 'api', 'v1', endpoint]
                endpoint = parts[3]
                # Handle common dynamic parts
                if len(parts) > 4 and parts[4].replace('-', '').replace('_', '').isalnum():
                    return f"/api/v1/{endpoint}/:id"
                return f"/api/v1/{endpoint}"
        
        # Handle WebSocket paths
        if path.startswith('/ws'):
            return '/ws'
        
        # Handle other special paths
        if path in ['/', '/health', '/docs', '/redoc', '/openapi.json']:
            return path
        
        return '/other'

    def _normalize_provider(self, provider: str) -> str:
        """Normalize provider label to a small, stable set to reduce cardinality.
        Accepts a provider key or URL; returns lowercase provider key.
        """
        try:
            if not provider:
                return 'unknown'
            p = provider.strip().lower()
            # If a URL/domain was passed, extract netloc base
            if p.startswith('http://') or p.startswith('https://'):
                netloc = urlparse(p).netloc
                p = netloc.split(':')[0]
            # Map common domains to canonical providers
            if 'alpaca' in p:
                return 'alpaca'
            if 'polygon' in p:
                return 'polygon'
            if 'newsapi' in p or 'newsapi.org' in p:
                return 'newsapi'
            if 'eodhd' in p:
                return 'eodhd'
            if 'reddit' in p:
                return 'reddit'
            if 'twitter' in p or 'x.com' in p:
                return 'twitter'
            return p
        except Exception:
            return 'unknown'

    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize endpoint path to limit label cardinality.
        - Strip query strings and fragments
        - Keep only the first 2-3 path segments
        - Replace numeric/date-like segments with tokens (:id)
        - Truncate overly long labels
        """
        try:
            if not endpoint:
                return '/'
            # If a full URL, parse and use the path; otherwise treat as path
            path = endpoint
            if endpoint.startswith('http://') or endpoint.startswith('https://'):
                path = urlparse(endpoint).path or '/'
            # Split and keep a few leading segments
            segs = [s for s in path.split('/') if s]
            norm = []
            for s in segs[:3]:
                # Replace numeric or UUID-like segments
                if s.isdigit() or _looks_like_uuid(s) or _looks_like_date(s):
                    norm.append(':id')
                else:
                    # Shorten very long slugs to prevent explosion
                    norm.append(s[:24])
            label = '/' + '/'.join(norm)
            # Final guard: cap to 64 chars
            if len(label) > 64:
                label = label[:61] + '...'
            return label if label else '/'
        except Exception:
            return '/'


def _looks_like_uuid(s: str) -> bool:
    if len(s) in (32, 36):
        hexchars = s.replace('-', '')
        return all(c in '0123456789abcdefABCDEF' for c in hexchars)
    return False


def _looks_like_date(s: str) -> bool:
    # Basic patterns: YYYY, YYYYMMDD, YYYY-MM-DD
    if len(s) == 10 and s[4] == '-' and s[7] == '-' and s[:4].isdigit() and s[5:7].isdigit() and s[8:].isdigit():
        return True
    if len(s) == 8 and s.isdigit():
        return True
    return False


# Global metrics collector
metrics = MetricsCollector()


async def create_metrics_middleware():
    """Create metrics middleware function."""
    
    async def metrics_middleware(request: Request, call_next):
        """Prometheus metrics middleware."""
        start_time = time.time()
        
        # Store start time for request
        request_id = id(request)
        metrics.request_start_times[request_id] = start_time
        
        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record metrics
            await metrics.record_http_request(request, response, duration)
            
            return response
            
        except Exception as e:
            # Record error
            duration = time.time() - start_time
            error_type = type(e).__name__
            metrics.record_error(error_type, 'error')
            
            # Create error response for metrics
            from fastapi.responses import JSONResponse
            error_response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
            
            await metrics.record_http_request(request, error_response, duration)
            
            raise
        finally:
            # Cleanup
            metrics.request_start_times.pop(request_id, None)
    
    return metrics_middleware


def get_metrics_handler():
    """Get Prometheus metrics handler."""
    async def metrics_handler():
        """Prometheus metrics endpoint."""
        if registry:
            content = generate_latest(registry)
        else:
            content = generate_latest()
        
        return Response(content, media_type=CONTENT_TYPE_LATEST)
    
    return metrics_handler


# Export metrics instance for use in other modules
__all__ = ['metrics', 'create_metrics_middleware', 'get_metrics_handler']

# Export coverage gauges explicitly for importers
__all__ += [
    'historical_dataset_coverage_ratio',
    'historical_dataset_row_count',
    'historical_dataset_last_timestamp_seconds',
    'historical_dataset_span_days'
]
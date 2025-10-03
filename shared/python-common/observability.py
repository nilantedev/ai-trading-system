"""Shared observability utilities for AI Trading System.

Centralizes metrics/tracing/middleware helpers to reduce duplication
across microservices. Designed to be incrementally adopted.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable, Awaitable, Optional, Dict
import contextlib

from prometheus_client import Counter, Histogram, Gauge
import os
try:  # Lazy / optional tracing imports
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _OTEL_AVAILABLE = True
except Exception:  # noqa: BLE001
    _OTEL_AVAILABLE = False

# Core shared metrics
HTTP_REQUESTS = Counter(
    "app_http_requests_total",
    "Total HTTP requests",
    ["service", "method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "app_http_request_latency_seconds",
    "HTTP request latency",
    ["service", "method", "path"],
    buckets=(0.005,0.01,0.02,0.05,0.1,0.25,0.5,1,2,5),
)
INFLIGHT = Gauge(
    "app_http_inflight_requests",
    "In-flight HTTP requests",
    ["service"],
)
SHED = Counter(
    "app_requests_shed_total",
    "Requests shed due to concurrency limits",
    ["service", "reason"],
)
CONCURRENCY_LIMIT = Gauge(
    "app_concurrency_limit",
    "Configured concurrency limit",
    ["service"],
)
QUEUE_DEPTH = Gauge(
    "app_request_queue_depth",
    "Queued requests awaiting concurrency slot",
    ["service"],
)

# Inference generic metrics (cross-service optional)
INFERENCE_LATENCY = Histogram(
    "app_inference_latency_seconds",
    "Generic inference latency (for services without own model metrics)",
    ["service", "model"],
    buckets=(0.005,0.01,0.02,0.05,0.1,0.25,0.5,1,2,5,10),
)
INFERENCE_REQUESTS = Counter(
    "app_inference_requests_total",
    "Generic inference request status counts",
    ["service", "model", "status"],
)

# Internal registry for path templates
_PATH_TEMPLATES: Dict[str, str] = {}


def register_path_template(raw: str, templated: str):
    _PATH_TEMPLATES[raw] = templated


def normalize_path(p: str) -> str:
    # Simple first pass; future: implement parameter extraction if needed
    for raw, templ in _PATH_TEMPLATES.items():
        if raw.endswith('*'):
            prefix = raw[:-1]
            if p.startswith(prefix):
                return templ
    return p


class ConcurrencyController:
    def __init__(self, service: str, limit: Optional[int]):
        self.service = service
        self.limit = limit or 0
        self.sem = asyncio.Semaphore(limit) if limit and limit > 0 else None
        self.waiters = 0  # approximate queue depth
        if self.limit:
            CONCURRENCY_LIMIT.labels(service=service).set(self.limit)

    async def run(self, coro_factory: Callable[[], Awaitable]):
        if not self.sem:
            return await coro_factory()
        acquired = False
        try:
            self.waiters += 1
            QUEUE_DEPTH.labels(service=self.service).set(self.waiters - 1 if self.waiters > 0 else 0)
            acquired = await self.sem.acquire()
            self.waiters -= 1
            QUEUE_DEPTH.labels(service=self.service).set(self.waiters)
            return await coro_factory()
        finally:
            if acquired:
                self.sem.release()
            else:
                # If not acquired (exception path), adjust waiter count
                if self.waiters > 0:
                    self.waiters -= 1
                    QUEUE_DEPTH.labels(service=self.service).set(self.waiters)


def middleware_factory(service_name: str, controller: ConcurrencyController | None = None):
    async def _middleware(request, call_next):
        path = normalize_path(request.url.path)
        method = request.method
        start = time.perf_counter()

        # Opportunistic shedding: if concurrency limit set and all permits taken and waiters exceed limit threshold, shed.
        if controller and controller.sem and controller.limit and controller.sem.locked():
            # If too many waiting already (>= limit), shed early
            if controller.waiters >= controller.limit:
                SHED.labels(service=service_name, reason="concurrency_limit").inc()
                status = 503
                HTTP_REQUESTS.labels(service=service_name, method=method, path=path, status=str(status)).inc()
                HTTP_LATENCY.labels(service=service_name, method=method, path=path).observe(0.0)
                return type("_Resp", (), {"status_code": status, "body": b"", "headers": {}})()

        INFLIGHT.labels(service=service_name).inc()
        try:
            if controller and controller.sem:
                # Run under controller to update queue depth properly
                async def _call():
                    return await call_next(request)
                response = await controller.run(_call)
            else:
                response = await call_next(request)
            status = response.status_code
            return response
        except Exception:
            status = 500
            raise
        finally:
            elapsed = time.perf_counter() - start
            HTTP_REQUESTS.labels(service=service_name, method=method, path=path, status=str(status)).inc()
            HTTP_LATENCY.labels(service=service_name, method=method, path=path).observe(elapsed)
            INFLIGHT.labels(service=service_name).dec()
    return _middleware


async def timed_inference(service: str, model: str, awaitable):
    start = time.perf_counter()
    status = 'success'
    try:
        return await awaitable
    except Exception:
        status = 'error'
        raise
    finally:
        INFERENCE_LATENCY.labels(service=service, model=model).observe(time.perf_counter() - start)
        INFERENCE_REQUESTS.labels(service=service, model=model, status=status).inc()


def install_observability(app, service_name: str, concurrency_limit: Optional[int] = None):
    controller = ConcurrencyController(service_name, concurrency_limit)
    app.middleware('http')(middleware_factory(service_name, controller))
    _maybe_init_tracing(service_name, app)
    # Baseline zero-value emission so metrics appear on first scrape (prevents false "missing" warnings)
    try:  # Counters/histograms need at least one label combination emitted
        SHED.labels(service=service_name, reason="init").inc(0)
        INFERENCE_LATENCY.labels(service=service_name, model="baseline").observe(0.0)
        INFERENCE_REQUESTS.labels(service=service_name, model="baseline", status="init").inc(0)
        CONCURRENCY_LIMIT.labels(service=service_name).set(concurrency_limit or 0)
        QUEUE_DEPTH.labels(service=service_name).set(0)
        INFLIGHT.labels(service=service_name).set(0)
    except Exception:  # noqa: BLE001
        pass
    return controller


_TRACING_INITIALIZED = False

def _maybe_init_tracing(service_name: str, app):  # noqa: D401
    """Initialize OpenTelemetry tracing once if OTEL enabled and available.

    Controlled by environment variables:
      OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4318)
      ENABLE_TRACING=true|false
    """
    global _TRACING_INITIALIZED
    if _TRACING_INITIALIZED:
        return
    if not _OTEL_AVAILABLE:
        return
    if os.getenv('ENABLE_TRACING', 'false').lower() not in ('1','true','yes'):  # feature flag
        return
    endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')
    try:
        resource = Resource.create({
            'service.name': service_name,
            'service.namespace': 'ai-trading-system',
            'deployment.environment': os.getenv('ENV', 'production')
        })
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)
        _TRACING_INITIALIZED = True

        # Inject tracing middleware AFTER metrics so span wraps business logic
        @app.middleware('http')
        async def _tracing_middleware(request, call_next):  # type: ignore
            corr_id = request.headers.get('X-Correlation-ID')
            tracer_local = trace.get_tracer(__name__)
            with tracer_local.start_as_current_span(f"HTTP {request.method} {request.url.path}") as span:
                try:
                    if corr_id:
                        span.set_attribute('http.correlation_id', corr_id)
                    span.set_attribute('http.method', request.method)
                    span.set_attribute('http.target', request.url.path)
                    span.set_attribute('service.name', service_name)
                    response = await call_next(request)
                    span.set_attribute('http.status_code', response.status_code)
                    return response
                except Exception as e:  # noqa: BLE001
                    span.record_exception(e)
                    span.set_attribute('error', True)
                    raise
    except Exception:
        # Fail silent to avoid impacting request flow
        return

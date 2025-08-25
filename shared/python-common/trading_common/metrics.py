#!/usr/bin/env python3
"""
Comprehensive metrics and monitoring for AI Trading System.
Provides Prometheus metrics, OpenTelemetry tracing, and structured logging.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta
import psutil
import json
from contextlib import asynccontextmanager, contextmanager

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricConfig:
    """Configuration for metrics collection."""
    enable_prometheus: bool = True
    enable_tracing: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    service_name: str = "ai-trading-system"
    resource_metrics_interval: int = 30  # seconds


class MetricsRegistry:
    """Central registry for all metrics."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        self.config = config or MetricConfig()
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.tracer = None
        
        # Initialize metrics
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        
        if OPENTELEMETRY_AVAILABLE:
            self._init_tracing()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # API Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry,
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Trading Metrics
        self.orders_total = Counter(
            'orders_total',
            'Total orders processed',
            ['symbol', 'side', 'status'],
            registry=self.registry
        )
        
        self.order_fill_latency_seconds = Histogram(
            'order_fill_latency_seconds',
            'Order fill latency in seconds',
            ['symbol'],
            registry=self.registry
        )
        
        self.portfolio_value_usd = Gauge(
            'portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.positions_count = Gauge(
            'positions_count',
            'Number of active positions',
            registry=self.registry
        )
        
        # Market Data Metrics
        self.market_data_messages_total = Counter(
            'market_data_messages_total',
            'Total market data messages received',
            ['source', 'symbol', 'message_type'],
            registry=self.registry
        )
        
        self.market_data_latency_seconds = Histogram(
            'market_data_latency_seconds',
            'Market data latency in seconds',
            ['source'],
            registry=self.registry
        )
        
        # ML Model Metrics
        self.model_predictions_total = Counter(
            'model_predictions_total',
            'Total model predictions made',
            ['model_name', 'symbol'],
            registry=self.registry
        )
        
        self.model_accuracy_ratio = Gauge(
            'model_accuracy_ratio',
            'Model accuracy ratio (0-1)',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_training_duration_seconds = Histogram(
            'model_training_duration_seconds',
            'Model training duration in seconds',
            ['model_name'],
            registry=self.registry
        )
        
        # Circuit Breaker Metrics
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['breaker_name'],
            registry=self.registry
        )
        
        self.circuit_breaker_failures_total = Counter(
            'circuit_breaker_failures_total',
            'Total circuit breaker failures',
            ['breaker_name'],
            registry=self.registry
        )
        
        # Rate Limiting Metrics
        self.rate_limit_requests_total = Counter(
            'rate_limit_requests_total',
            'Total rate limit checks',
            ['limit_type', 'result'],
            registry=self.registry
        )
        
        # System Resource Metrics
        self.system_cpu_usage_ratio = Gauge(
            'system_cpu_usage_ratio',
            'System CPU usage ratio (0-1)',
            registry=self.registry
        )
        
        self.system_memory_usage_bytes = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.redis_connections_active = Gauge(
            'redis_connections_active',
            'Active Redis connections',
            registry=self.registry
        )
        
        logger.info("Prometheus metrics initialized")
    
    def _init_tracing(self):
        """Initialize OpenTelemetry tracing."""
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": "1.0.0"
        })
        
        provider = TracerProvider(resource=resource)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            endpoint=self.config.jaeger_endpoint,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(span_processor)
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)
        
        logger.info("OpenTelemetry tracing initialized")
    
    def get_metrics_content(self) -> str:
        """Get Prometheus metrics in text format."""
        if not PROMETHEUS_AVAILABLE or not self.registry:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry)
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration_seconds.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
    
    def record_order(self, symbol: str, side: str, status: str, fill_latency: Optional[float] = None):
        """Record trading order metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.orders_total.labels(symbol=symbol, side=side, status=status).inc()
        
        if fill_latency is not None:
            self.order_fill_latency_seconds.labels(symbol=symbol).observe(fill_latency)
    
    def update_portfolio_metrics(self, value_usd: float, positions_count: int):
        """Update portfolio metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.portfolio_value_usd.set(value_usd)
        self.positions_count.set(positions_count)
    
    def record_market_data(self, source: str, symbol: str, message_type: str, latency: float):
        """Record market data metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.market_data_messages_total.labels(
            source=source, 
            symbol=symbol, 
            message_type=message_type
        ).inc()
        
        self.market_data_latency_seconds.labels(source=source).observe(latency)
    
    def record_model_prediction(self, model_name: str, symbol: str):
        """Record ML model prediction."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.model_predictions_total.labels(model_name=model_name, symbol=symbol).inc()
    
    def update_model_accuracy(self, model_name: str, accuracy: float):
        """Update model accuracy metric."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.model_accuracy_ratio.labels(model_name=model_name).set(accuracy)
    
    def record_model_training(self, model_name: str, duration: float):
        """Record model training duration."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.model_training_duration_seconds.labels(model_name=model_name).observe(duration)
    
    def update_circuit_breaker(self, breaker_name: str, state: str, failure_count: int = 0):
        """Update circuit breaker metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        state_map = {"closed": 0, "half_open": 1, "open": 2}
        self.circuit_breaker_state.labels(breaker_name=breaker_name).set(
            state_map.get(state, 0)
        )
        
        if failure_count > 0:
            self.circuit_breaker_failures_total.labels(breaker_name=breaker_name).inc(failure_count)
    
    def record_rate_limit_check(self, limit_type: str, allowed: bool):
        """Record rate limit check result."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        result = "allowed" if allowed else "denied"
        self.rate_limit_requests_total.labels(limit_type=limit_type, result=result).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.system_cpu_usage_ratio.set(cpu_percent / 100.0)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.system_memory_usage_bytes.set(memory.used)
    
    @contextmanager
    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a tracing span context manager."""
        if not OPENTELEMETRY_AVAILABLE or not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span
    
    async def start_resource_monitoring(self):
        """Start background task for resource monitoring."""
        while True:
            try:
                self.update_system_metrics()
                await asyncio.sleep(self.config.resource_metrics_interval)
            except Exception as e:
                logger.error(f"Error updating resource metrics: {e}")
                await asyncio.sleep(self.config.resource_metrics_interval)


# Global metrics registry
_metrics_registry: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    """Get or create global metrics registry."""
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


def init_metrics(config: Optional[MetricConfig] = None):
    """Initialize global metrics registry."""
    global _metrics_registry
    _metrics_registry = MetricsRegistry(config)
    return _metrics_registry


# Decorators for automatic metrics collection
def track_api_metrics(endpoint: str = None):
    """Decorator to track API endpoint metrics."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            metrics = get_metrics_registry()
            
            # Extract endpoint from function name if not provided
            endpoint_name = endpoint or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                
                # Record successful request
                duration = time.time() - start_time
                metrics.record_http_request("GET", endpoint_name, 200, duration)
                
                return result
                
            except Exception as e:
                # Record failed request
                duration = time.time() - start_time
                status_code = getattr(e, 'status_code', 500)
                metrics.record_http_request("GET", endpoint_name, status_code, duration)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            metrics = get_metrics_registry()
            
            endpoint_name = endpoint or func.__name__
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_http_request("GET", endpoint_name, 200, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                status_code = getattr(e, 'status_code', 500)
                metrics.record_http_request("GET", endpoint_name, status_code, duration)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def track_model_metrics(model_name: str):
    """Decorator to track ML model metrics."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            metrics = get_metrics_registry()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record model prediction
                if 'symbol' in kwargs:
                    metrics.record_model_prediction(model_name, kwargs['symbol'])
                
                return result
                
            except Exception as e:
                logger.error(f"Model {model_name} prediction failed: {e}")
                raise
            finally:
                # Record training duration if this was a training call
                if 'training' in func.__name__.lower():
                    duration = time.time() - start_time
                    metrics.record_model_training(model_name, duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            metrics = get_metrics_registry()
            
            try:
                result = func(*args, **kwargs)
                
                if 'symbol' in kwargs:
                    metrics.record_model_prediction(model_name, kwargs['symbol'])
                
                return result
                
            except Exception as e:
                logger.error(f"Model {model_name} prediction failed: {e}")
                raise
            finally:
                if 'training' in func.__name__.lower():
                    duration = time.time() - start_time
                    metrics.record_model_training(model_name, duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Structured logging with metrics correlation
class MetricsLogger:
    """Logger with automatic metrics correlation."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.metrics = get_metrics_registry()
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info with metrics context."""
        if extra is None:
            extra = {}
        
        # Add correlation context
        extra.update({
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'INFO'
        })
        extra.update(kwargs)
        
        self.logger.info(message, extra=extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False, **kwargs):
        """Log error with metrics context."""
        if extra is None:
            extra = {}
        
        extra.update({
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'ERROR'
        })
        extra.update(kwargs)
        
        self.logger.error(message, extra=extra, exc_info=exc_info)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning with metrics context."""
        if extra is None:
            extra = {}
        
        extra.update({
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'WARNING'
        })
        extra.update(kwargs)
        
        self.logger.warning(message, extra=extra)


def get_metrics_logger(name: str) -> MetricsLogger:
    """Get a metrics-aware logger."""
    return MetricsLogger(name)
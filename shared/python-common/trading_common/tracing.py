#!/usr/bin/env python3
"""
OpenTelemetry tracing integration for AI Trading System.
Provides comprehensive distributed tracing across all service layers.
"""

import asyncio
import time
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass

try:
    from opentelemetry import trace, context, baggage
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider, Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor  
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace.status import Status, StatusCode
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class TracingConfig:
    """Configuration for OpenTelemetry tracing."""
    service_name: str = "ai-trading-system"
    service_version: str = "1.0.0"
    environment: str = "production"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    otlp_endpoint: str = "http://localhost:4317"
    console_export: bool = False
    sample_rate: float = 1.0
    enable_auto_instrumentation: bool = True


class TracingManager:
    """Central manager for OpenTelemetry tracing."""
    
    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig()
        self.tracer = None
        self.initialized = False
        
        if TRACING_AVAILABLE:
            self._init_tracing()
    
    def _init_tracing(self):
        """Initialize OpenTelemetry tracing."""
        # Create resource with service information
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": self.config.service_version,
            "deployment.environment": self.config.environment,
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Add exporters
        processors = []
        
        # Jaeger exporter
        try:
            jaeger_exporter = JaegerExporter(
                endpoint=self.config.jaeger_endpoint,
            )
            processors.append(BatchSpanProcessor(jaeger_exporter))
            logger.info("Jaeger exporter configured")
        except Exception as e:
            logger.warning(f"Failed to configure Jaeger exporter: {e}")
        
        # OTLP exporter (for services like Datadog, New Relic, etc.)
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
            )
            processors.append(BatchSpanProcessor(otlp_exporter))
            logger.info("OTLP exporter configured")
        except Exception as e:
            logger.warning(f"Failed to configure OTLP exporter: {e}")
        
        # Console exporter for development
        if self.config.console_export:
            processors.append(BatchSpanProcessor(ConsoleSpanExporter()))
        
        # Add processors to provider
        for processor in processors:
            provider.add_span_processor(processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)
        
        # Enable automatic instrumentation
        if self.config.enable_auto_instrumentation:
            self._enable_auto_instrumentation()
        
        self.initialized = True
        logger.info("OpenTelemetry tracing initialized")
    
    def _enable_auto_instrumentation(self):
        """Enable automatic instrumentation for common libraries."""
        try:
            # FastAPI instrumentation
            FastAPIInstrumentor().instrument()
            logger.debug("FastAPI auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"FastAPI instrumentation failed: {e}")
        
        try:
            # HTTP client instrumentation
            AioHttpClientInstrumentor().instrument()
            RequestsInstrumentor().instrument()
            logger.debug("HTTP client auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"HTTP client instrumentation failed: {e}")
        
        try:
            # Redis instrumentation
            RedisInstrumentor().instrument()
            logger.debug("Redis auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"Redis instrumentation failed: {e}")
        
        try:
            # Database instrumentation
            AsyncPGInstrumentor().instrument()
            Psycopg2Instrumentor().instrument()
            logger.debug("Database auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"Database instrumentation failed: {e}")
    
    @contextmanager
    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a tracing span context manager."""
        if not self.initialized or not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(str(key), str(value))
            
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    @asynccontextmanager
    async def async_trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create an async tracing span context manager."""
        if not self.initialized or not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(str(key), str(value))
            
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    def get_trace_context(self) -> Dict[str, str]:
        """Get current trace context for propagation."""
        if not self.initialized:
            return {}
        
        carrier = {}
        inject(carrier)
        return carrier
    
    def set_trace_context(self, context_dict: Dict[str, str]):
        """Set trace context from propagated headers."""
        if not self.initialized:
            return
        
        ctx = extract(context_dict)
        context.attach(ctx)


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> TracingManager:
    """Get or create global tracing manager."""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


def init_tracing(config: Optional[TracingConfig] = None):
    """Initialize global tracing manager."""
    global _tracing_manager
    _tracing_manager = TracingManager(config)
    return _tracing_manager


# Decorators for automatic tracing
def trace_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace function execution."""
    def decorator(func: Callable):
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracing = get_tracing_manager()
                
                func_attributes = attributes or {}
                func_attributes.update({
                    "function.name": func.__name__,
                    "function.module": func.__module__
                })
                
                async with tracing.async_trace_span(span_name, func_attributes) as span:
                    if span:
                        # Add function arguments as attributes (be careful with sensitive data)
                        if kwargs:
                            for key, value in kwargs.items():
                                if not key.startswith('_') and len(str(value)) < 100:
                                    span.set_attribute(f"function.arg.{key}", str(value))
                    
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        
                        if span:
                            span.set_attribute("function.success", True)
                            span.set_attribute("function.duration_ms", 
                                             (time.time() - start_time) * 1000)
                        
                        return result
                    except Exception as e:
                        if span:
                            span.set_attribute("function.success", False)
                            span.set_attribute("function.error", str(e))
                        raise
            
            return async_wrapper
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracing = get_tracing_manager()
                
                func_attributes = attributes or {}
                func_attributes.update({
                    "function.name": func.__name__,
                    "function.module": func.__module__
                })
                
                with tracing.trace_span(span_name, func_attributes) as span:
                    if span:
                        if kwargs:
                            for key, value in kwargs.items():
                                if not key.startswith('_') and len(str(value)) < 100:
                                    span.set_attribute(f"function.arg.{key}", str(value))
                    
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        
                        if span:
                            span.set_attribute("function.success", True)
                            span.set_attribute("function.duration_ms",
                                             (time.time() - start_time) * 1000)
                        
                        return result
                    except Exception as e:
                        if span:
                            span.set_attribute("function.success", False)
                            span.set_attribute("function.error", str(e))
                        raise
            
            return sync_wrapper
    
    return decorator


def trace_trading_operation(operation_type: str, symbol: str = None):
    """Decorator specifically for trading operations."""
    def decorator(func: Callable):
        attributes = {
            "trading.operation_type": operation_type
        }
        if symbol:
            attributes["trading.symbol"] = symbol
        
        return trace_function(f"trading.{operation_type}", attributes)(func)
    
    return decorator


def trace_data_ingestion(source: str, data_type: str):
    """Decorator for data ingestion operations."""
    def decorator(func: Callable):
        attributes = {
            "data.source": source,
            "data.type": data_type
        }
        
        return trace_function(f"data_ingestion.{source}.{data_type}", attributes)(func)
    
    return decorator


def trace_ml_operation(model_name: str, operation: str):
    """Decorator for ML operations."""
    def decorator(func: Callable):
        attributes = {
            "ml.model_name": model_name,
            "ml.operation": operation
        }
        
        return trace_function(f"ml.{model_name}.{operation}", attributes)(func)
    
    return decorator


def trace_risk_check(check_type: str):
    """Decorator for risk management operations."""
    def decorator(func: Callable):
        attributes = {
            "risk.check_type": check_type
        }
        
        return trace_function(f"risk.{check_type}", attributes)(func)
    
    return decorator


# Utility functions for manual tracing
def add_trace_attributes(span, **attributes):
    """Add multiple attributes to current span."""
    if span and TRACING_AVAILABLE:
        for key, value in attributes.items():
            span.set_attribute(str(key), str(value))


def record_trace_event(span, event_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Record an event in the current span."""
    if span and TRACING_AVAILABLE:
        span.add_event(event_name, attributes or {})


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID for correlation."""
    if not TRACING_AVAILABLE:
        return None
    
    span = trace.get_current_span()
    if span and span.get_span_context().trace_id:
        return f"{span.get_span_context().trace_id:032x}"
    return None


def get_current_span_id() -> Optional[str]:
    """Get current span ID for correlation."""
    if not TRACING_AVAILABLE:
        return None
    
    span = trace.get_current_span()
    if span and span.get_span_context().span_id:
        return f"{span.get_span_context().span_id:016x}"
    return None
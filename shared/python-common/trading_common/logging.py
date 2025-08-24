"""Structured logging configuration for trading system."""

import sys
import logging
import contextvars
from typing import Optional, Dict, Any
from functools import lru_cache

import structlog
from structlog.types import Processor


# Context variables for request tracing
request_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_id', default=None
)
user_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'user_id', default=None
)


def add_request_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add request context to log entries."""
    request_id = request_id_context.get()
    user_id = user_id_context.get()
    
    if request_id:
        event_dict['request_id'] = request_id
    if user_id:
        event_dict['user_id'] = user_id
    
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    service_name: str = "trading-service",
    environment: str = "development",
    json_logs: bool = True,
) -> None:
    """Setup structured logging configuration."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Shared processors for all loggers
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        add_request_context,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add service metadata
    structlog.configure_defaults(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Development vs production formatting
    if json_logs and environment.lower() == "production":
        # JSON formatting for production
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable formatting for development
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Add service context to all logs
    structlog.contextvars.bind_contextvars(
        service=service_name,
        environment=environment,
    )


@lru_cache()
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def set_request_context(request_id: str, user_id: Optional[str] = None) -> None:
    """Set request context for logging."""
    request_id_context.set(request_id)
    if user_id:
        user_id_context.set(user_id)


def clear_request_context() -> None:
    """Clear request context."""
    request_id_context.set(None)
    user_id_context.set(None)


class LoggerMixin:
    """Mixin class to add structured logging to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)
    
    def log_method_call(self, method_name: str, **kwargs: Any) -> None:
        """Log a method call with parameters."""
        self.logger.debug(
            "Method called",
            method=method_name,
            parameters=kwargs,
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error with context."""
        self.logger.error(
            "Error occurred",
            error=str(error),
            error_type=type(error).__name__,
            context=context or {},
            exc_info=True,
        )


# Pre-configured loggers for common use cases
audit_logger = get_logger("trading.audit")
performance_logger = get_logger("trading.performance")
security_logger = get_logger("trading.security")
trading_logger = get_logger("trading.decisions")
data_logger = get_logger("trading.data")
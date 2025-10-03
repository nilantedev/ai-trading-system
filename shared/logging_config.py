"""Structured logging configuration using structlog.
Ensures correlation_id and request context (if available) are attached.
"""
from __future__ import annotations
import logging, sys, time, contextvars, os, json
import structlog
from typing import Any, Dict

_DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
try:
    LOG_LEVEL = logging.getLevelName(_DEFAULT_LEVEL)
except Exception:  # noqa: BLE001
    LOG_LEVEL = logging.INFO

ENABLE_PLAIN_LOGS = os.getenv("ENABLE_PLAIN_TEXT_LOGS", "0").lower() in ("1", "true", "yes")

# Context variables
correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("correlation_id", default=None)

class _AddTimestamp:
    def __call__(self, logger: Any, method_name: str, event_dict: Dict[str, Any]):
        event_dict['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        return event_dict

class _RenameEventKey:
    def __call__(self, logger, method_name, event_dict):
        if 'event' in event_dict:
            event_dict['message'] = event_dict.pop('event')
        return event_dict

class _AddCorrelationId:
    def __call__(self, logger, method_name, event_dict):
        cid = correlation_id_var.get()
        if cid:
            event_dict['correlation_id'] = cid
        return event_dict

def configure_logging():
    """Configure structlog with optional plain-text fallback.

    Environment Variables:
      LOG_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL (default INFO)
      ENABLE_PLAIN_TEXT_LOGS: if true emits simplified text logs (useful when tailing locally)
    """
    logging.basicConfig(level=LOG_LEVEL, format="%(message)s", stream=sys.stdout)
    processors = [
        structlog.contextvars.merge_contextvars,
        _AddTimestamp(),
        _AddCorrelationId(),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _RenameEventKey(),
    ]
    if ENABLE_PLAIN_LOGS:
        processors.append(structlog.processors.KeyValueRenderer(key_order=["timestamp","level","message"]))
    else:
        processors.append(structlog.processors.JSONRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    if not structlog.is_configured():  # defensive
        configure_logging()
    return structlog.get_logger(name)

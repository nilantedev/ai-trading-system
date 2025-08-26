"""Structured logging configuration using structlog.
Ensures correlation_id and request context (if available) are attached.
"""
from __future__ import annotations
import logging, sys, time, contextvars
import structlog
from typing import Any, Dict

LOG_LEVEL = logging.getLevelName("INFO")

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
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(message)s",
        stream=sys.stdout,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _AddTimestamp(),
            _AddCorrelationId(),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _RenameEventKey(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    if not structlog.is_configured():  # defensive
        configure_logging()
    return structlog.get_logger(name)

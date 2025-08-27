#!/usr/bin/env python3
"""Unified structured event logging utilities for ML lifecycle events.

Provides a single emit_event() function that formats and logs JSON lines with
standard fields plus dynamic payload, and (optionally) Prometheus counters if
available. Designed to be a soft dependency: failures are swallowed.
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:  # optional metrics integration
    from api.metrics import metrics  # type: ignore
    from prometheus_client import Counter  # type: ignore
    _event_counter = Counter(
        'model_lifecycle_events_total',
        'Total model lifecycle events emitted',
        ['event_type'],
        registry=getattr(metrics, 'registry', None) if hasattr(metrics, 'registry') else None
    )
except Exception:  # pragma: no cover
    _event_counter = None  # type: ignore

STANDARD_FIELDS = [
    'timestamp', 'event_type', 'model_name', 'version'
]


def emit_event(event_type: str, **payload: Any) -> None:
    """Emit a structured lifecycle event.

    Args:
        event_type: dot-namespaced event type (e.g. 'model.drift.scan').
        **payload: Additional event attributes (must be JSON serializable or coercible)
    """
    evt: Dict[str, Any] = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'event_type': event_type,
    }
    evt.update(payload)
    try:
        logger.info("MODEL_EVENT %s", json.dumps(evt, default=str, sort_keys=True))
        if _event_counter:
            try:
                _event_counter.labels(event_type=event_type).inc()
            except Exception:
                pass
    except Exception:  # pragma: no cover
        # Swallow to avoid impacting main flow
        pass

__all__ = ['emit_event']

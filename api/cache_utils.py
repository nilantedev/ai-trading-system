from __future__ import annotations
import time
import threading
from typing import Callable, Any, Dict, Tuple

_lock = threading.RLock()
_cache: Dict[str, Tuple[float, Any]] = {}

def ttl_cache(key: str, ttl_seconds: int, loader: Callable[[], Any]) -> Any:
    """Simple thread-safe TTL cache for infrequently polled dashboard data.

    Not for high cardinality keys. Values expire after ttl_seconds.
    """
    now = time.time()
    with _lock:
        entry = _cache.get(key)
        if entry and entry[0] > now:
            return entry[1]
    # Load outside lock to avoid long critical sections
    value = loader()
    with _lock:
        _cache[key] = (now + ttl_seconds, value)
    return value

def clear_cache(prefix: str | None = None):  # pragma: no cover - maintenance helper
    with _lock:
        if prefix is None:
            _cache.clear()
        else:
            for k in list(_cache.keys()):
                if k.startswith(prefix):
                    _cache.pop(k, None)

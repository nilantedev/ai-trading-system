"""Trading Common Library - Shared utilities for AI trading system.

This module intentionally performs ONLY minimal, low‑risk imports eagerly to
avoid hard crashes in lightweight service containers that do not install the
full dependency set (metrics, tracing, ML registry, etc.).

Several production outages were caused by ImportError exceptions during
container startup because ``trading_common.__init__`` previously imported many
optional subsystems (metrics, tracing, feature store, model registry, SLO
monitoring, etc.). Those modules pull in packages (e.g. prometheus-client,
opentelemetry, torch, heavy ML libs) that are NOT installed in the slimmer
service images (execution, strategy-engine, data-ingestion, etc.), leading the
Python process to exit immediately and Docker to restart the container.

Fix: convert optional imports to lazy/guarded form. Core services only rely on
``get_logger``, ``get_settings`` and shared models/cache/database helpers. All
additional capabilities are exposed on a best‑effort basis if dependencies are
present, otherwise silently skipped (they can be imported directly by modules
that require them).
"""

from .config import get_settings, get_settings_with_vault, Settings  # Core config
from .logging import get_logger, setup_logging  # Core logging
from .exceptions import TradingError, ValidationError, ConfigError  # Core errors
from .models import *  # Data models (pure Pydantic, safe)
from .database import get_redis_client, RedisClient  # Lightweight dependency
from .cache import get_trading_cache, TradingCache  # Lightweight dependency

__version__ = "1.0.0-dev"

# Optional features are imported defensively. Each block sets a flag so callers
# can feature-detect if needed.

_optional_exports = []

def _try_import(name: str, import_block):
    try:
        import_block()
    except Exception:  # noqa: BLE001 - Broad: we deliberately swallow all import issues
        pass


# Metrics & monitoring
_try_import("metrics", lambda: _optional_exports.extend([
    "get_metrics_registry", "init_metrics", "MetricsRegistry", "MetricConfig", "get_metrics_logger"
]))
try:  # Kept separate so we can still reference names if present
    from .metrics import (  # type: ignore
        get_metrics_registry, init_metrics, MetricsRegistry, MetricConfig, get_metrics_logger
    )
except Exception:  # noqa: BLE001
    pass

# Resilience patterns
try:
    from .resilience import RetryStrategy, CircuitBreaker, get_circuit_breaker  # type: ignore
    _optional_exports += ["RetryStrategy", "CircuitBreaker", "get_circuit_breaker"]
except Exception:  # noqa: BLE001
    pass

# Security store
try:
    from .security_store import (  # type: ignore
        get_security_store, log_security_event, SecurityEventType,
        PersistentSecurityStore, SecurityEvent, UserSession, RefreshToken
    )
    _optional_exports += [
        "get_security_store", "log_security_event", "SecurityEventType",
        "PersistentSecurityStore", "SecurityEvent", "UserSession", "RefreshToken"
    ]
except Exception:  # noqa: BLE001
    pass

# Feature store & ML registry (heavy, may pull torch/ML)
try:
    from .feature_store import get_feature_store, FeatureVector, FeatureDefinition  # type: ignore
    _optional_exports += ["get_feature_store", "FeatureVector", "FeatureDefinition"]
except Exception:  # noqa: BLE001
    pass
try:
    from .ml_pipeline import get_ml_pipeline, TrainingConfig, ModelMetrics  # type: ignore
    _optional_exports += ["get_ml_pipeline", "TrainingConfig", "ModelMetrics"]
except Exception:  # noqa: BLE001
    pass
try:
    from .ml_registry import get_model_registry, ModelMetadata, ModelType, ModelStatus  # type: ignore
    _optional_exports += ["get_model_registry", "ModelMetadata", "ModelType", "ModelStatus"]
except Exception:  # noqa: BLE001
    pass

# User management (may depend on database tables / hashing libs)
try:
    from .user_management import (  # type: ignore
        get_user_manager, User, UserRole, UserStatus, Session, PermissionManager,
        create_default_admin_user, require_permission, require_role
    )
    _optional_exports += [
        "get_user_manager", "User", "UserRole", "UserStatus", "Session",
        "PermissionManager", "create_default_admin_user", "require_permission", "require_role"
    ]
except Exception:  # noqa: BLE001
    pass

# Tracing (opentelemetry)
try:
    from .tracing import (  # type: ignore
        get_tracing_manager, init_tracing, TracingConfig, trace_function,
        trace_trading_operation, trace_data_ingestion, trace_ml_operation,
        trace_risk_check, get_current_trace_id
    )
    _optional_exports += [
        "get_tracing_manager", "init_tracing", "TracingConfig", "trace_function",
        "trace_trading_operation", "trace_data_ingestion", "trace_ml_operation",
        "trace_risk_check", "get_current_trace_id"
    ]
except Exception:  # noqa: BLE001
    pass

# SLO monitoring
try:
    from .slo_monitoring import (  # type: ignore
        get_slo_manager, SLOTarget, SLOType, AlertSeverity, Alert, record_slo_metric, init_slo_monitoring
    )
    _optional_exports += [
        "get_slo_manager", "SLOTarget", "SLOType", "AlertSeverity", "Alert",
        "record_slo_metric", "init_slo_monitoring"
    ]
except Exception:  # noqa: BLE001
    pass

# Public exports (core + any optional successfully imported)
__all__ = [
    # Core config/logging/errors
    "get_settings", "get_settings_with_vault", "Settings", "get_logger", "setup_logging",
    "TradingError", "ValidationError", "ConfigError",
    # Core infra
    "get_redis_client", "RedisClient", "get_trading_cache", "TradingCache",
] + _optional_exports  # models are exported via wildcard earlier

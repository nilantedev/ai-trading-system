"""Trading Common Library - Shared utilities for AI trading system."""

from .config import get_settings, get_settings_with_vault, Settings
from .logging import get_logger, setup_logging
from .exceptions import TradingError, ValidationError, ConfigError

# Import all models
from .models import *

# Import database utilities
from .database import get_redis_client, RedisClient

# Import cache utilities  
from .cache import get_trading_cache, TradingCache

# Import metrics and monitoring
from .metrics import get_metrics_registry, init_metrics, MetricsRegistry, MetricConfig, get_metrics_logger

# Import resilience patterns
from .resilience import RetryStrategy, CircuitBreaker, get_circuit_breaker

# Import security store
from .security_store import (
    get_security_store, log_security_event, SecurityEventType,
    PersistentSecurityStore, SecurityEvent, UserSession, RefreshToken
)

# Import ML components
from .feature_store import get_feature_store, FeatureVector, FeatureDefinition
from .ml_pipeline import get_ml_pipeline, TrainingConfig, ModelMetrics
from .ml_registry import get_model_registry, ModelMetadata, ModelType, ModelStatus

# Import user management
from .user_management import (
    get_user_manager, User, UserRole, UserStatus, Session, PermissionManager,
    create_default_admin_user, require_permission, require_role
)

# Import tracing
from .tracing import (
    get_tracing_manager, init_tracing, TracingConfig,
    trace_function, trace_trading_operation, trace_data_ingestion,
    trace_ml_operation, trace_risk_check, get_current_trace_id
)

# Import SLO monitoring
from .slo_monitoring import (
    get_slo_manager, SLOTarget, SLOType, AlertSeverity, Alert,
    record_slo_metric, init_slo_monitoring
)

__version__ = "1.0.0-dev"
__all__ = [
    "get_settings",
    "get_settings_with_vault", 
    "Settings", 
    "get_logger",
    "setup_logging",
    "TradingError",
    "ValidationError",
    "ConfigError",
    "get_redis_client",
    "RedisClient",
    "get_trading_cache",
    "TradingCache",
    "get_metrics_registry",
    "init_metrics",
    "MetricsRegistry",
    "MetricConfig",
    "get_metrics_logger",
    "RetryStrategy",
    "CircuitBreaker",
    "get_circuit_breaker",
    "get_security_store",
    "log_security_event",
    "SecurityEventType",
    "PersistentSecurityStore",
    "SecurityEvent", 
    "UserSession",
    "RefreshToken",
    
    # ML components
    "get_model_registry",
    "ModelMetadata", 
    "ModelType",
    "ModelStatus",
    
    # User management
    "get_user_manager",
    "User",
    "UserRole", 
    "UserStatus",
    "Session",
    "PermissionManager",
    "create_default_admin_user",
    "require_permission",
    "require_role",
    
    # Tracing
    "get_tracing_manager",
    "init_tracing",
    "TracingConfig",
    "trace_function",
    "trace_trading_operation",
    "trace_data_ingestion", 
    "trace_ml_operation",
    "trace_risk_check",
    "get_current_trace_id",
    
    # SLO monitoring
    "get_slo_manager",
    "SLOTarget",
    "SLOType", 
    "AlertSeverity",
    "Alert",
    "record_slo_metric",
    "init_slo_monitoring"
]
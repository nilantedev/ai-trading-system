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
    "RefreshToken"
]
"""Configuration management for trading system."""

import os
from functools import lru_cache
from typing import Optional, List, Dict, Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    redis_url: str = Field(default="redis://localhost:6379/0")
    questdb_host: str = Field(default="localhost")
    questdb_port: int = Field(default=9000)
    questdb_user: Optional[str] = Field(default=None)
    questdb_password: Optional[str] = Field(default=None)
    
    postgres_url: Optional[str] = Field(default=None)
    weaviate_url: str = Field(default="http://localhost:8080")
    arangodb_url: str = Field(default="http://localhost:8529")
    
    class Config:
        env_prefix = "DB_"


class MessageSettings(BaseSettings):
    """Message broker configuration."""
    
    pulsar_url: str = Field(default="pulsar://localhost:6650")
    pulsar_tenant: str = Field(default="trading")
    pulsar_namespace: str = Field(default="default")
    
    class Config:
        env_prefix = "MSG_"


class AISettings(BaseSettings):
    """AI/ML configuration settings."""
    
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    
    local_model_path: str = Field(default="/models")
    model_cache_size: str = Field(default="32GB")
    max_batch_size: int = Field(default=16)
    inference_timeout: int = Field(default=30)
    
    class Config:
        env_prefix = "AI_"


class TradingSettings(BaseSettings):
    """Trading-specific configuration."""
    
    # API keys
    polygon_api_key: Optional[str] = Field(default=None)
    alpaca_api_key: Optional[str] = Field(default=None)
    alpaca_secret_key: Optional[str] = Field(default=None)
    
    # Trading parameters
    paper_trading: bool = Field(default=True)
    max_position_size: float = Field(default=10000.0)
    risk_limit_percent: float = Field(default=2.0)
    
    # Market hours (NYSE)
    market_open_hour: int = Field(default=9)
    market_open_minute: int = Field(default=30)
    market_close_hour: int = Field(default=16)
    market_close_minute: int = Field(default=0)
    
    class Config:
        env_prefix = "TRADING_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    prometheus_port: int = Field(default=8000)
    health_check_interval: int = Field(default=30)
    log_level: str = Field(default="INFO")
    
    # OpenTelemetry
    otel_exporter_endpoint: Optional[str] = Field(default=None)
    otel_service_name: str = Field(default="trading-service")
    
    class Config:
        env_prefix = "MONITORING_"


class SecuritySettings(BaseSettings):
    """Security configuration."""
    
    secret_key: str = Field(default="dev-secret-change-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=30)
    
    # CORS
    cors_origins: List[str] = Field(default=["http://localhost:3000"])
    cors_allow_credentials: bool = Field(default=True)
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    class Config:
        env_prefix = "SECURITY_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Service configuration
    service_name: str = Field(default="trading-service")
    service_version: str = Field(default="1.0.0-dev")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    messaging: MessageSettings = Field(default_factory=MessageSettings)
    ai: AISettings = Field(default_factory=AISettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    
    @validator('debug', pre=True)
    def parse_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_database_url(db_type: str = "redis") -> str:
    """Get database URL for specified database type."""
    settings = get_settings()
    
    urls = {
        "redis": settings.database.redis_url,
        "postgres": settings.database.postgres_url,
        "weaviate": settings.database.weaviate_url,
        "arangodb": settings.database.arangodb_url,
    }
    
    url = urls.get(db_type)
    if not url:
        raise ValueError(f"No URL configured for database type: {db_type}")
    
    return url


def is_feature_enabled(feature: str) -> bool:
    """Check if a feature flag is enabled."""
    settings = get_settings()
    env_var = f"FEATURE_{feature.upper()}_ENABLED"
    return os.getenv(env_var, "false").lower() in ("true", "1", "yes", "on")
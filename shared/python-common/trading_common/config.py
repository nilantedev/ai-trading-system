"""Configuration management for trading system."""

import os
from functools import lru_cache
from typing import Optional, List, Dict, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings with dual environment support."""
    
    # Redis configuration
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)
    
    # QuestDB configuration  
    questdb_host: str = Field(default="localhost")
    questdb_http_port: int = Field(default=9000)
    questdb_pg_port: int = Field(default=8812)
    questdb_influx_port: int = Field(default=9009)
    questdb_user: Optional[str] = Field(default=None)
    questdb_password: Optional[str] = Field(default=None)
    questdb_database: str = Field(default="qdb")
    
    # PostgreSQL (optional)
    postgres_url: Optional[str] = Field(default=None)
    postgres_host: Optional[str] = Field(default=None)
    postgres_port: int = Field(default=5432)
    postgres_user: Optional[str] = Field(default=None)
    postgres_password: Optional[str] = Field(default=None)
    postgres_database: Optional[str] = Field(default=None)
    
    # Vector and Graph databases
    weaviate_url: str = Field(default="http://localhost:8080")
    weaviate_host: str = Field(default="localhost")
    weaviate_port: int = Field(default=8080)
    
    arangodb_url: str = Field(default="http://localhost:8529")
    arangodb_host: str = Field(default="localhost")
    arangodb_port: int = Field(default=8529)
    arangodb_user: str = Field(default="root")
    arangodb_password: Optional[str] = Field(default=None)
    
    # Connection pool settings
    redis_max_connections: int = Field(default=30)
    questdb_min_pool_size: int = Field(default=5)
    questdb_max_pool_size: int = Field(default=25)
    connection_timeout: int = Field(default=30)
    
    @field_validator('redis_url', mode='before')
    @classmethod
    def build_redis_url(cls, v, info):
        """Build Redis URL from components if not provided."""
        if v and v != "redis://localhost:6379/0":
            return v
        
        values = info.data if info else {}
        host = values.get('redis_host', 'localhost')
        port = values.get('redis_port', 6379)
        db = values.get('redis_db', 0)
        password = values.get('redis_password')
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"
    
    @field_validator('weaviate_url', mode='before')
    @classmethod
    def build_weaviate_url(cls, v, info):
        """Build Weaviate URL from components if not provided."""
        if v and v != "http://localhost:8080":
            return v
            
        values = info.data if info else {}
        host = values.get('weaviate_host', 'localhost')
        port = values.get('weaviate_port', 8080)
        return f"http://{host}:{port}"
        
    @field_validator('arangodb_url', mode='before')
    @classmethod
    def build_arangodb_url(cls, v, info):
        """Build ArangoDB URL from components if not provided."""
        if v and v != "http://localhost:8529":
            return v
            
        values = info.data if info else {}
        host = values.get('arangodb_host', 'localhost')
        port = values.get('arangodb_port', 8529)
        return f"http://{host}:{port}"
    
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
    """AI/ML configuration settings - Local models only."""
    
    # Ollama configuration (local models only - no API costs)
    ollama_host: str = Field(default="http://localhost:11434")
    use_local_models_only: bool = Field(default=True)
    
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
    
    # Additional rate limiting (used by rate limiter code)
    rate_limit_requests_per_minute: int = Field(default=100)
    rate_limit_burst: int = Field(default=10)
    
    # Trusted hosts
    trusted_hosts: List[str] = Field(default=["localhost", "127.0.0.1"])
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @field_validator('trusted_hosts', mode='before')
    @classmethod
    def parse_trusted_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(',')]
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
    
    @field_validator('debug', mode='before')
    @classmethod
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
    
    def validate_production_config(self) -> List[str]:
        """Validate production configuration and return list of issues."""
        issues = []
        
        if not self.is_production:
            return issues
            
        # Check critical secrets
        if self.security.secret_key == "dev-secret-change-in-production":
            issues.append("CRITICAL: Using default secret key in production! Set SECURITY_SECRET_KEY")
            
        if len(self.security.secret_key) < 32:
            issues.append("CRITICAL: Secret key too short for production (minimum 32 characters)")
            
        # Check trading API keys
        if not self.trading.alpaca_api_key:
            issues.append("WARNING: No Alpaca API key configured - trading features disabled")
            
        if not self.trading.alpaca_secret_key:
            issues.append("WARNING: No Alpaca secret key configured - trading features disabled")
            
        # Check if still using paper trading in production
        if self.trading.paper_trading:
            issues.append("WARNING: Paper trading enabled in production - consider live trading")
            
        # Check admin password
        admin_password = os.getenv("ADMIN_PASSWORD")
        admin_password_hash = os.getenv("ADMIN_PASSWORD_HASH")
        
        # Strict check for production environments
        if self.environment == "production":
            if not admin_password_hash:
                issues.append("CRITICAL: ADMIN_PASSWORD_HASH must be set in production!")
            if admin_password:
                issues.append("CRITICAL: Plain text ADMIN_PASSWORD should not be used in production!")
            
        # Check for known default passwords
        default_passwords = ["TradingSystem2024!", "admin", "password", "123456", "default"]
        if admin_password and any(pwd in admin_password for pwd in default_passwords):
            if self.environment == "production":
                issues.append("CRITICAL: Default or weak admin password detected in production!")
            else:
                issues.append("WARNING: Using default/weak admin password - change before production")
            
        return issues
    
    def enforce_production_security(self):
        """Enforce production security requirements - raises exception if critical issues found."""
        issues = self.validate_production_config()
        critical_issues = [issue for issue in issues if issue.startswith("CRITICAL")]
        
        if critical_issues:
            error_msg = "Production security validation failed:\n" + "\n".join(critical_issues)
            raise ValueError(error_msg)
            
        # Log warnings
        warnings = [issue for issue in issues if issue.startswith("WARNING")]
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(warning)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        extra = "ignore"  # Allow extra environment variables


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Enhanced settings with secrets vault support (opt-in)
def get_settings_with_vault():
    """Get settings with secrets vault integration."""
    try:
        from .config_secrets import get_enhanced_settings
        return get_enhanced_settings()
    except ImportError:
        # Fall back to base settings if vault integration not available
        return get_settings()


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
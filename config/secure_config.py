"""
Secure configuration management with environment-specific settings.
All secrets are loaded from secure providers, never hardcoded.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseSettings, Field, SecretStr, validator
from enum import Enum
import logging

from shared.security.secrets_manager import get_secrets_manager

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class SecureConfig(BaseSettings):
    """Secure configuration with validation and secret management."""
    
    # Environment
    environment: Environment = Field(
        default=Environment.PRODUCTION,
        description="Application environment"
    )
    
    # Application
    app_name: str = Field(default="AI Trading System", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Security - Loaded from secrets manager
    secret_key: Optional[SecretStr] = Field(default=None, description="Application secret key")
    jwt_secret: Optional[SecretStr] = Field(default=None, description="JWT signing secret")
    encryption_key: Optional[SecretStr] = Field(default=None, description="Data encryption key")
    
    # Database - Loaded from secrets manager
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="trading_system", description="Database name")
    db_user: Optional[str] = Field(default=None, description="Database user")
    db_password: Optional[SecretStr] = Field(default=None, description="Database password")
    db_pool_size: int = Field(default=10, description="Connection pool size")
    db_max_overflow: int = Field(default=20, description="Maximum overflow connections")
    
    # Redis - Loaded from secrets manager
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[SecretStr] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database")
    redis_pool_size: int = Field(default=10, description="Redis connection pool size")
    
    # API Keys - All loaded from secrets manager
    # Removed OpenAI/Anthropic - using local Ollama models only
    alpaca_api_key: Optional[SecretStr] = Field(default=None, description="Alpaca API key")
    alpaca_secret: Optional[SecretStr] = Field(default=None, description="Alpaca secret")
    polygon_api_key: Optional[SecretStr] = Field(default=None, description="Polygon API key")
    binance_api_key: Optional[SecretStr] = Field(default=None, description="Binance API key")
    binance_secret: Optional[SecretStr] = Field(default=None, description="Binance secret")
    
    # Monitoring - Loaded from secrets manager
    grafana_user: str = Field(default="admin", description="Grafana admin user")
    grafana_password: Optional[SecretStr] = Field(default=None, description="Grafana password")
    prometheus_auth_token: Optional[SecretStr] = Field(default=None, description="Prometheus auth token")
    
    # Security Settings
    cors_origins: list = Field(default=["http://localhost:3000"], description="CORS origins")
    trusted_hosts: list = Field(default=["localhost", "127.0.0.1"], description="Trusted hosts")
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    session_timeout: int = Field(default=3600, description="Session timeout in seconds")
    password_min_length: int = Field(default=12, description="Minimum password length")
    mfa_enabled: bool = Field(default=True, description="Enable MFA")
    
    # ML Configuration
    model_path: str = Field(default="/models", description="Model storage path")
    feature_store_path: str = Field(default="/features", description="Feature store path")
    model_registry_url: Optional[str] = Field(default=None, description="Model registry URL")
    mlflow_tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking URI")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")
    log_dir: str = Field(default="/logs", description="Log directory")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        """Initialize config with secure secret loading."""
        super().__init__(**kwargs)
        self._load_secrets()
        self._validate_config()
    
    def _load_secrets(self):
        """Load secrets from secure provider."""
        secrets_manager = get_secrets_manager()
        
        # Map of config field to secret key
        secret_mappings = {
            "secret_key": "app/secret_key",
            "jwt_secret": "app/jwt_secret",
            "encryption_key": "app/encryption_key",
            "db_user": "database/user",
            "db_password": "database/password",
            "redis_password": "redis/password",
            # Removed OpenAI/Anthropic - using local models
            "alpaca_api_key": "api/alpaca_key",
            "alpaca_secret": "api/alpaca_secret",
            "polygon_api_key": "api/polygon_key",
            "binance_api_key": "api/binance_key",
            "binance_secret": "api/binance_secret",
            "grafana_password": "monitoring/grafana_password",
            "prometheus_auth_token": "monitoring/prometheus_token",
        }
        
        for field, secret_key in secret_mappings.items():
            if getattr(self, field) is None:
                value = secrets_manager.get_secret(secret_key)
                if value:
                    if field.endswith("password") or field.endswith("secret") or field.endswith("key"):
                        setattr(self, field, SecretStr(value))
                    else:
                        setattr(self, field, value)
                elif self.environment == Environment.PRODUCTION:
                    # In production, missing secrets are critical
                    if field in ["secret_key", "jwt_secret", "db_password"]:
                        raise ValueError(f"Required secret '{secret_key}' not found in production")
    
    def _validate_config(self):
        """Validate configuration based on environment."""
        
        if self.environment == Environment.PRODUCTION:
            # Production validations
            if not self.secret_key:
                raise ValueError("SECRET_KEY is required in production")
            if not self.db_password:
                raise ValueError("Database password is required in production")
            if self.debug:
                raise ValueError("Debug mode cannot be enabled in production")
            if "localhost" in self.cors_origins:
                logger.warning("Localhost in CORS origins for production")
            if not self.mfa_enabled:
                logger.warning("MFA is disabled in production")
        
        # Ensure critical secrets are strong enough
        if self.secret_key and len(self.secret_key.get_secret_value()) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        
        if self.jwt_secret and len(self.jwt_secret.get_secret_value()) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        """Validate and normalize environment."""
        if isinstance(v, str):
            v = v.lower()
            if v not in [e.value for e in Environment]:
                raise ValueError(f"Invalid environment: {v}")
        return v
    
    @property
    def database_url(self) -> str:
        """Generate database URL."""
        password = self.db_password.get_secret_value() if self.db_password else ""
        return f"postgresql+asyncpg://{self.db_user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def redis_url(self) -> str:
        """Generate Redis URL."""
        auth = f":{self.redis_password.get_secret_value()}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.environment == Environment.TESTING
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        key_map = {
            # Removed OpenAI/Anthropic - using local models
            "alpaca": self.alpaca_api_key,
            "polygon": self.polygon_api_key,
            "binance": self.binance_api_key,
        }
        
        key = key_map.get(provider.lower())
        return key.get_secret_value() if key else None
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convert config to dictionary, optionally excluding secrets."""
        data = self.dict()
        
        if not include_secrets:
            # Remove sensitive fields
            sensitive_fields = [
                "secret_key", "jwt_secret", "encryption_key",
                "db_password", "redis_password",
                # Removed OpenAI/Anthropic
                "alpaca_api_key", "alpaca_secret",
                "polygon_api_key", "binance_api_key", "binance_secret",
                "grafana_password", "prometheus_auth_token"
            ]
            for field in sensitive_fields:
                data.pop(field, None)
        else:
            # Convert SecretStr to actual values
            for key, value in data.items():
                if isinstance(value, SecretStr):
                    data[key] = value.get_secret_value()
        
        return data


# Environment-specific configurations
class DevelopmentConfig(SecureConfig):
    """Development environment configuration."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "DEBUG"
    mfa_enabled: bool = False
    
    def _load_secrets(self):
        """Override to allow local development defaults."""
        super()._load_secrets()
        
        # Set development defaults if secrets not found
        if not self.secret_key:
            self.secret_key = SecretStr("dev-secret-key-only-for-local-development")
        if not self.jwt_secret:
            self.jwt_secret = SecretStr("dev-jwt-secret-only-for-local-development")
        if not self.db_user:
            self.db_user = "dev_user"
        if not self.db_password:
            self.db_password = SecretStr("dev_password")


class TestingConfig(SecureConfig):
    """Testing environment configuration."""
    environment: Environment = Environment.TESTING
    debug: bool = True
    log_level: str = "DEBUG"
    mfa_enabled: bool = False
    
    # Use in-memory database for tests
    db_host: str = "localhost"
    db_name: str = "test_trading_system"
    
    def _load_secrets(self):
        """Override for testing with mock secrets."""
        self.secret_key = SecretStr("test-secret-key")
        self.jwt_secret = SecretStr("test-jwt-secret")
        self.db_user = "test_user"
        self.db_password = SecretStr("test_password")


class ProductionConfig(SecureConfig):
    """Production environment configuration."""
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    log_level: str = "INFO"
    mfa_enabled: bool = True
    
    # Production-specific settings
    rate_limit_requests: int = 50
    session_timeout: int = 1800  # 30 minutes
    password_min_length: int = 16
    
    # Stricter CORS in production
    cors_origins: list = Field(default=[], description="CORS origins")
    trusted_hosts: list = Field(default=[], description="Trusted hosts")


# Configuration factory
def get_config(environment: Optional[str] = None) -> SecureConfig:
    """Get configuration for the specified environment."""
    
    env = environment or os.getenv("ENVIRONMENT", "production")
    env = env.lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "testing": TestingConfig,
        "production": ProductionConfig,
        "staging": ProductionConfig,  # Staging uses production config
    }
    
    config_class = config_map.get(env, ProductionConfig)
    return config_class()


# Global config instance
_config: Optional[SecureConfig] = None


def get_current_config() -> SecureConfig:
    """Get the current configuration instance."""
    global _config
    if not _config:
        _config = get_config()
    return _config


def reload_config():
    """Reload configuration (useful for testing)."""
    global _config
    _config = get_config()
    return _config
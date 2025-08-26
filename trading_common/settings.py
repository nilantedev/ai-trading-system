import os
import secrets
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class SecuritySettings:
    cors_origins: List[str] = field(default_factory=lambda: [])
    cors_allow_credentials: bool = True
    trusted_hosts: List[str] = field(default_factory=lambda: [])
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    jwt_refresh_expire_days: int = 7
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    api_key_length: int = 32

@dataclass
class DatabaseSettings:
    postgres_user: str = ""
    postgres_password: str = ""
    postgres_db: str = ""
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    redis_password: str = ""
    redis_host: str = "localhost"
    redis_port: int = 6379

@dataclass
class TradingApiSettings:
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    polygon_api_key: str = ""
    
@dataclass
class VaultSettings:
    vault_url: str = ""
    vault_token: str = ""
    vault_mount_path: str = "secret"
    aws_secret_manager_region: str = "us-east-1"
    azure_key_vault_url: str = ""

@dataclass
class Settings:
    environment: str = "development"
    is_production: bool = False
    debug: bool = False
    log_level: str = "INFO"
    security: SecuritySettings = field(default_factory=SecuritySettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    trading: TradingApiSettings = field(default_factory=TradingApiSettings)
    vault: VaultSettings = field(default_factory=VaultSettings)

    def enforce_production_security(self) -> None:
        """Enforce production security requirements"""
        if not self.is_production:
            return
            
        errors = []
        
        # Check required secrets
        if not self.security.secret_key or len(self.security.secret_key) < 32:
            errors.append("JWT_SECRET_KEY must be at least 32 characters")
            
        if not self.database.postgres_password or self.database.postgres_password == "secure_postgres_password_123":
            errors.append("POSTGRES_PASSWORD must be set and not use default value")
            
        if not self.database.redis_password or self.database.redis_password == "secure_redis_password_123":
            errors.append("REDIS_PASSWORD must be set and not use default value")
            
        if not self.trading.alpaca_api_key or self.trading.alpaca_api_key == "your_alpaca_paper_key_here":
            errors.append("ALPACA_API_KEY must be set and not use placeholder value")
            
        # Check security settings
        if "*" in self.security.cors_origins:
            errors.append("CORS origins cannot use wildcard (*) in production")
            
        if "*" in self.security.trusted_hosts:
            errors.append("Trusted hosts cannot use wildcard (*) in production")
            
        if self.debug:
            errors.append("DEBUG must be disabled in production")
            
        if errors:
            logger.error("Production security validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError(f"Production security requirements not met: {'; '.join(errors)}")


def get_settings() -> Settings:
    """Load settings from environment variables with secure defaults"""
    
    # Determine environment
    environment = os.getenv("ENVIRONMENT", "development").lower()
    is_production = environment == "production"
    
    # Security settings
    secret_key = os.getenv("JWT_SECRET_KEY", "")
    if not secret_key and not is_production:
        # Generate a secure random key for development
        secret_key = secrets.token_urlsafe(32)
        logger.warning("Generated temporary JWT secret key for development")
    
    # CORS origins
    cors_origins_str = os.getenv("CORS_ORIGINS", "")
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()] if cors_origins_str else []
    if not cors_origins and not is_production:
        cors_origins = ["http://localhost:3000", "http://localhost:8000"]
    
    # Trusted hosts
    trusted_hosts_str = os.getenv("TRUSTED_HOSTS", "")
    trusted_hosts = [host.strip() for host in trusted_hosts_str.split(",") if host.strip()] if trusted_hosts_str else []
    if not trusted_hosts and not is_production:
        trusted_hosts = ["localhost", "127.0.0.1"]
    
    settings = Settings(
        environment=environment,
        is_production=is_production,
        debug=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        
        security=SecuritySettings(
            cors_origins=cors_origins,
            cors_allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
            trusted_hosts=trusted_hosts,
            secret_key=secret_key,
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "30")),
            jwt_refresh_expire_days=int(os.getenv("JWT_REFRESH_EXPIRE_DAYS", "7")),
            max_login_attempts=int(os.getenv("MAX_LOGIN_ATTEMPTS", "5")),
            lockout_duration_minutes=int(os.getenv("LOCKOUT_DURATION_MINUTES", "15")),
            api_key_length=int(os.getenv("API_KEY_LENGTH", "32"))
        ),
        
        database=DatabaseSettings(
            postgres_user=os.getenv("POSTGRES_USER", "trading_user"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
            postgres_db=os.getenv("POSTGRES_DB", "trading_system"),
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            redis_password=os.getenv("REDIS_PASSWORD", ""),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379"))
        ),
        
        trading=TradingApiSettings(
            alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            polygon_api_key=os.getenv("POLYGON_API_KEY", "")
        ),
        
        vault=VaultSettings(
            vault_url=os.getenv("VAULT_URL", ""),
            vault_token=os.getenv("VAULT_TOKEN", ""),
            vault_mount_path=os.getenv("VAULT_MOUNT_PATH", "secret"),
            aws_secret_manager_region=os.getenv("AWS_SECRET_MANAGER_REGION", "us-east-1"),
            azure_key_vault_url=os.getenv("AZURE_KEY_VAULT_URL", "")
        )
    )
    
    # Enforce production security if needed
    settings.enforce_production_security()
    
    return settings

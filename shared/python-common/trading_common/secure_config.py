"""Secure configuration management with vault integration."""

import os
import sys
import hashlib
import logging
from typing import Any, Optional, Dict, Set
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
import asyncio

from .secrets_vault import (
    get_secrets_manager, VaultConfig, VaultType,
    configure_secrets_manager, SecretNotFoundError
)
from .logging import get_logger

logger = get_logger(__name__)

# Track which secrets have been accessed for audit
_accessed_secrets: Set[str] = set()
_redacted_secrets: Dict[str, str] = {}


class SecureSettings(BaseSettings):
    """Secure settings with vault integration and validation."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Vault configuration
    vault_type: str = Field(default="env", env="VAULT_TYPE")
    vault_endpoint: Optional[str] = Field(default=None, env="VAULT_ENDPOINT")
    vault_token: Optional[str] = Field(default=None, env="VAULT_TOKEN")
    vault_namespace: Optional[str] = Field(default=None, env="VAULT_NAMESPACE")
    vault_mount_path: Optional[str] = Field(default="secret", env="VAULT_MOUNT_PATH")
    
    # Security enforcement
    enforce_secrets_vault: bool = Field(default=True, env="ENFORCE_SECRETS_VAULT")
    allow_env_fallback: bool = Field(default=True, env="ALLOW_ENV_FALLBACK")
    fail_on_missing_secret: bool = Field(default=True, env="FAIL_ON_MISSING_SECRET")
    audit_secret_access: bool = Field(default=True, env="AUDIT_SECRET_ACCESS")
    
    # Secret rotation
    secret_rotation_hours: int = Field(default=24, env="SECRET_ROTATION_HOURS")
    api_key_rotation_days: int = Field(default=30, env="API_KEY_ROTATION_DAYS")
    
    # Sensitive fields (will be loaded from vault)
    _database_password: Optional[str] = None
    _redis_password: Optional[str] = None
    _jwt_secret: Optional[str] = None
    _admin_password_hash: Optional[str] = None
    _api_keys: Dict[str, str] = {}
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @validator("environment")
    def validate_environment(cls, v):
        """Ensure environment is valid."""
        valid_envs = ["development", "staging", "production", "test"]
        if v not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v
    
    def __init__(self, **kwargs):
        """Initialize secure settings."""
        super().__init__(**kwargs)
        
        # Initialize vault configuration
        if self.enforce_secrets_vault and self.environment != "development":
            self._init_vault()
    
    def _init_vault(self):
        """Initialize vault configuration."""
        try:
            vault_config = self._get_vault_config()
            
            # Configure fallback if allowed
            fallback_config = None
            if self.allow_env_fallback:
                fallback_config = VaultConfig(VaultType.ENVIRONMENT)
            
            configure_secrets_manager(vault_config, fallback_config)
            logger.info(f"Vault configured: {self.vault_type}")
            
        except Exception as e:
            if self.fail_on_missing_secret:
                logger.error(f"Failed to initialize vault: {e}")
                raise
            else:
                logger.warning(f"Vault initialization failed, using environment: {e}")
    
    def _get_vault_config(self) -> VaultConfig:
        """Get vault configuration based on type."""
        vault_type_map = {
            "hashicorp": VaultType.HASHICORP_VAULT,
            "aws": VaultType.AWS_SECRETS_MANAGER,
            "azure": VaultType.AZURE_KEY_VAULT,
            "env": VaultType.ENVIRONMENT
        }
        
        vault_type = vault_type_map.get(self.vault_type.lower())
        if not vault_type:
            raise ValueError(f"Invalid vault type: {self.vault_type}")
        
        config = VaultConfig(
            vault_type=vault_type,
            endpoint=self.vault_endpoint,
            token=self.vault_token,
            namespace=self.vault_namespace,
            mount_path=self.vault_mount_path
        )
        
        # Add AWS-specific config if needed
        if vault_type == VaultType.AWS_SECRETS_MANAGER:
            config.region = os.getenv("AWS_REGION", "us-east-1")
            config.access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            config.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        # Add Azure-specific config if needed
        elif vault_type == VaultType.AZURE_KEY_VAULT:
            config.tenant_id = os.getenv("AZURE_TENANT_ID")
            config.client_id = os.getenv("AZURE_CLIENT_ID")
            config.client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        return config
    
    async def load_secrets(self):
        """Load all secrets from vault."""
        try:
            manager = await get_secrets_manager()
            
            # Load database secrets
            try:
                db_secrets = await manager.get_secret("database")
                self._database_password = db_secrets.get("postgres_password")
                self._redis_password = db_secrets.get("redis_password")
                self._track_secret_access("database")
            except SecretNotFoundError:
                if self.fail_on_missing_secret and self.environment == "production":
                    raise
                logger.warning("Database secrets not found in vault")
            
            # Load JWT secrets
            try:
                jwt_secrets = await manager.get_secret("jwt")
                self._jwt_secret = jwt_secrets.get("secret_key")
                self._track_secret_access("jwt")
            except SecretNotFoundError:
                if self.fail_on_missing_secret and self.environment == "production":
                    raise
                logger.warning("JWT secrets not found in vault")
            
            # Load API keys
            try:
                api_secrets = await manager.get_secret("api_keys")
                self._api_keys = api_secrets
                self._track_secret_access("api_keys")
            except SecretNotFoundError:
                if self.fail_on_missing_secret and self.environment == "production":
                    raise
                logger.warning("API keys not found in vault")
            
            # Load admin password hash
            try:
                auth_secrets = await manager.get_secret("auth")
                self._admin_password_hash = auth_secrets.get("admin_password_hash")
                self._track_secret_access("auth")
            except SecretNotFoundError:
                if self.fail_on_missing_secret and self.environment == "production":
                    raise
                logger.warning("Auth secrets not found in vault")
            
            logger.info("Secrets loaded successfully from vault")
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            if self.fail_on_missing_secret:
                raise
    
    def _track_secret_access(self, secret_name: str):
        """Track secret access for audit."""
        if self.audit_secret_access:
            _accessed_secrets.add(secret_name)
            logger.info(f"Secret accessed: {secret_name}")
    
    def get_database_password(self) -> str:
        """Get database password securely."""
        self._track_secret_access("database_password")
        
        if self._database_password:
            return self._database_password
        
        # Fallback to environment if allowed
        if self.allow_env_fallback:
            return os.getenv("POSTGRES_PASSWORD", "")
        
        if self.fail_on_missing_secret:
            raise SecretNotFoundError("Database password not available")
        
        return ""
    
    def get_redis_password(self) -> str:
        """Get Redis password securely."""
        self._track_secret_access("redis_password")
        
        if self._redis_password:
            return self._redis_password
        
        if self.allow_env_fallback:
            return os.getenv("REDIS_PASSWORD", "")
        
        if self.fail_on_missing_secret:
            raise SecretNotFoundError("Redis password not available")
        
        return ""
    
    def get_jwt_secret(self) -> str:
        """Get JWT secret securely."""
        self._track_secret_access("jwt_secret")
        
        if self._jwt_secret:
            return self._jwt_secret
        
        if self.allow_env_fallback:
            return os.getenv("JWT_SECRET_KEY", "")
        
        if self.fail_on_missing_secret:
            raise SecretNotFoundError("JWT secret not available")
        
        return ""
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a specific service."""
        self._track_secret_access(f"api_key_{service}")
        
        # Check loaded API keys
        if self._api_keys:
            key = self._api_keys.get(f"{service}_api_key")
            if key:
                return key
        
        # Fallback to environment
        if self.allow_env_fallback:
            env_key = f"{service.upper()}_API_KEY"
            return os.getenv(env_key, "")
        
        if self.fail_on_missing_secret:
            raise SecretNotFoundError(f"API key for {service} not available")
        
        return ""
    
    def get_redacted_value(self, value: str) -> str:
        """Get redacted version of a secret value."""
        if not value:
            return "[EMPTY]"
        
        # Cache redacted values
        if value in _redacted_secrets:
            return _redacted_secrets[value]
        
        # Create redacted version
        if len(value) <= 8:
            redacted = "*" * len(value)
        else:
            # Show first 2 and last 2 characters
            redacted = f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
        
        _redacted_secrets[value] = redacted
        return redacted
    
    def validate_secrets(self) -> Dict[str, bool]:
        """Validate that required secrets are available."""
        validation = {}
        
        # Check database password
        try:
            pwd = self.get_database_password()
            validation["database_password"] = bool(pwd)
        except SecretNotFoundError:
            validation["database_password"] = False
        
        # Check Redis password
        try:
            pwd = self.get_redis_password()
            validation["redis_password"] = bool(pwd)
        except SecretNotFoundError:
            validation["redis_password"] = False
        
        # Check JWT secret
        try:
            secret = self.get_jwt_secret()
            validation["jwt_secret"] = bool(secret)
        except SecretNotFoundError:
            validation["jwt_secret"] = False
        
        # Check critical API keys
        for service in ["alpaca", "polygon"]:
            try:
                key = self.get_api_key(service)
                validation[f"{service}_api_key"] = bool(key)
            except SecretNotFoundError:
                validation[f"{service}_api_key"] = False
        
        return validation
    
    def enforce_production_requirements(self):
        """Enforce security requirements for production."""
        if self.environment != "production":
            return
        
        errors = []
        
        # Must use vault in production
        if self.vault_type == "env":
            errors.append("Production must use a secrets vault, not environment variables")
        
        # Validate all required secrets
        validation = self.validate_secrets()
        missing = [k for k, v in validation.items() if not v]
        
        if missing:
            errors.append(f"Missing required secrets: {', '.join(missing)}")
        
        # Check secret strength
        try:
            jwt_secret = self.get_jwt_secret()
            if jwt_secret and len(jwt_secret) < 32:
                errors.append("JWT secret must be at least 32 characters in production")
        except SecretNotFoundError:
            errors.append("JWT secret is required in production")
        
        if errors:
            logger.error("Production security requirements not met:")
            for error in errors:
                logger.error(f"  - {error}")
            
            if self.fail_on_missing_secret:
                raise ValueError("Production security requirements not met")
    
    def get_accessed_secrets(self) -> Set[str]:
        """Get list of secrets that have been accessed."""
        return _accessed_secrets.copy()
    
    def clear_secret_cache(self):
        """Clear cached secrets (for rotation)."""
        self._database_password = None
        self._redis_password = None
        self._jwt_secret = None
        self._admin_password_hash = None
        self._api_keys = {}
        _redacted_secrets.clear()
        logger.info("Secret cache cleared")


@lru_cache()
def get_secure_settings() -> SecureSettings:
    """Get cached secure settings instance."""
    settings = SecureSettings()
    
    # Load secrets asynchronously
    if settings.enforce_secrets_vault:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule for later if loop is already running
                asyncio.create_task(settings.load_secrets())
            else:
                # Run now if no loop is running
                loop.run_until_complete(settings.load_secrets())
        except RuntimeError:
            # Create new loop if needed
            asyncio.run(settings.load_secrets())
    
    # Enforce production requirements
    settings.enforce_production_requirements()
    
    return settings


def rotate_secrets():
    """Force reload of secrets from vault."""
    get_secure_settings.cache_clear()
    logger.info("Settings cache cleared for secret rotation")


def audit_secret_access() -> Dict[str, Any]:
    """Get audit report of secret access."""
    settings = get_secure_settings()
    
    return {
        "environment": settings.environment,
        "vault_type": settings.vault_type,
        "accessed_secrets": list(_accessed_secrets),
        "total_accesses": len(_accessed_secrets),
        "validation": settings.validate_secrets()
    }
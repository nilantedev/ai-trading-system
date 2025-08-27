#!/usr/bin/env python3
"""
Initialize and manage secrets for the AI Trading System.
This script sets up all required secrets in the chosen provider.
"""

import os
import sys
import secrets
import string
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import getpass
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.security.secrets_manager import (
    SecretsManager,
    HashiCorpVaultProvider,
    LocalEncryptedProvider,
    AWSSecretsManagerProvider,
    AzureKeyVaultProvider,
    GCPSecretManagerProvider
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecretsInitializer:
    """Initialize and manage application secrets."""
    
    def __init__(self, provider_type: str = "local", **provider_kwargs):
        self.provider_type = provider_type
        self.manager = self._create_manager(provider_type, **provider_kwargs)
        self.secrets_dir = Path(__file__).parent.parent / "secrets"
        self.secrets_dir.mkdir(exist_ok=True, mode=0o700)
    
    def _create_manager(self, provider_type: str, **kwargs) -> SecretsManager:
        """Create secrets manager with specified provider."""
        
        if provider_type == "vault":
            provider = HashiCorpVaultProvider(
                vault_url=kwargs.get("vault_url", os.getenv("VAULT_ADDR", "http://localhost:8200")),
                token=kwargs.get("vault_token", os.getenv("VAULT_TOKEN")),
                mount_point=kwargs.get("mount_point", "secret")
            )
        elif provider_type == "aws":
            provider = AWSSecretsManagerProvider(
                region_name=kwargs.get("region", "us-east-1")
            )
        elif provider_type == "azure":
            provider = AzureKeyVaultProvider(
                vault_url=kwargs.get("vault_url", os.getenv("AZURE_KEY_VAULT_URL"))
            )
        elif provider_type == "gcp":
            provider = GCPSecretManagerProvider(
                project_id=kwargs.get("project_id", os.getenv("GOOGLE_CLOUD_PROJECT"))
            )
        else:  # local
            provider = LocalEncryptedProvider(
                storage_path=kwargs.get("storage_path", str(self.secrets_dir / "encrypted")),
                master_key=kwargs.get("master_key")
            )
        
        return SecretsManager(provider=provider)
    
    def generate_password(self, length: int = 32, include_special: bool = True) -> str:
        """Generate a secure password."""
        alphabet = string.ascii_letters + string.digits
        if include_special:
            alphabet += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)
    
    def initialize_required_secrets(self, environment: str = "production", interactive: bool = False):
        """Initialize all required secrets for the application."""
        
        logger.info(f"Initializing secrets for {environment} environment...")
        
        # Define required secrets with their types and defaults
        required_secrets = {
            # Application secrets
            "app/secret_key": {"type": "key", "length": 64},
            "app/jwt_secret": {"type": "key", "length": 64},
            "app/encryption_key": {"type": "key", "length": 44},  # Fernet key
            
            # Database secrets
            "database/user": {"type": "username", "default": f"trading_{environment}"},
            "database/password": {"type": "password", "length": 32},
            "database/root_password": {"type": "password", "length": 32},
            
            # Redis secrets
            "redis/password": {"type": "password", "length": 32},
            
            # Monitoring secrets
            "monitoring/grafana_user": {"type": "username", "default": "admin"},
            "monitoring/grafana_password": {"type": "password", "length": 24},
            "monitoring/prometheus_token": {"type": "key", "length": 32},
            
            # Vault token (if using Vault)
            "vault/root_token": {"type": "key", "length": 32},
        }
        
        # Add API keys if in production
        if environment == "production":
            api_secrets = {
                "api/openai_key": {"type": "api_key", "prompt": "OpenAI API Key"},
                "api/anthropic_key": {"type": "api_key", "prompt": "Anthropic API Key"},
                "api/alpaca_key": {"type": "api_key", "prompt": "Alpaca API Key"},
                "api/alpaca_secret": {"type": "api_key", "prompt": "Alpaca Secret"},
                "api/polygon_key": {"type": "api_key", "prompt": "Polygon API Key"},
                "api/binance_key": {"type": "api_key", "prompt": "Binance API Key"},
                "api/binance_secret": {"type": "api_key", "prompt": "Binance Secret"},
            }
            required_secrets.update(api_secrets)
        
        initialized = []
        skipped = []
        failed = []
        
        for secret_key, config in required_secrets.items():
            # Check if secret already exists
            existing = self.manager.get_secret(secret_key)
            if existing:
                skipped.append(secret_key)
                logger.info(f"Secret '{secret_key}' already exists, skipping...")
                continue
            
            # Generate or prompt for secret value
            if interactive and config.get("prompt"):
                value = getpass.getpass(f"Enter {config['prompt']}: ")
                if not value:
                    logger.warning(f"No value provided for '{secret_key}', skipping...")
                    continue
            else:
                # Auto-generate based on type
                if config["type"] == "password":
                    value = self.generate_password(config.get("length", 32))
                elif config["type"] == "key":
                    if config.get("length") == 44:  # Fernet key
                        from cryptography.fernet import Fernet
                        value = Fernet.generate_key().decode()
                    else:
                        value = secrets.token_urlsafe(config.get("length", 32))[:config.get("length", 32)]
                elif config["type"] == "api_key":
                    if interactive:
                        value = getpass.getpass(f"Enter {config.get('prompt', secret_key)}: ")
                    else:
                        value = None  # Skip API keys in non-interactive mode
                elif config["type"] == "username":
                    value = config.get("default", "admin")
                else:
                    value = self.generate_api_key()
            
            if value:
                # Store the secret
                if self.manager.set_secret(secret_key, value):
                    initialized.append(secret_key)
                    logger.info(f"✓ Initialized secret: {secret_key}")
                    
                    # Also write to local files for Docker secrets
                    if self.provider_type == "local":
                        self._write_docker_secret(secret_key.split("/")[-1], value)
                else:
                    failed.append(secret_key)
                    logger.error(f"✗ Failed to initialize secret: {secret_key}")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info(f"Secrets Initialization Summary:")
        logger.info(f"  Initialized: {len(initialized)}")
        logger.info(f"  Skipped (existing): {len(skipped)}")
        logger.info(f"  Failed: {len(failed)}")
        
        if failed:
            logger.error(f"Failed secrets: {', '.join(failed)}")
            return False
        
        return True
    
    def _write_docker_secret(self, name: str, value: str):
        """Write secret to file for Docker secrets."""
        secret_file = self.secrets_dir / f"{name}.txt"
        secret_file.write_text(value)
        secret_file.chmod(0o600)
    
    def rotate_secrets(self, pattern: Optional[str] = None):
        """Rotate secrets matching the pattern."""
        
        secrets_list = self.manager.list_secrets()
        
        if pattern:
            import re
            regex = re.compile(pattern)
            secrets_list = [s for s in secrets_list if regex.match(s)]
        
        logger.info(f"Rotating {len(secrets_list)} secrets...")
        
        rotated = []
        failed = []
        
        for secret_key in secrets_list:
            try:
                # Skip API keys and usernames
                if any(x in secret_key for x in ["api/", "/user", "username"]):
                    logger.info(f"Skipping rotation for {secret_key}")
                    continue
                
                new_value = self.manager.rotate_secret(secret_key)
                rotated.append(secret_key)
                logger.info(f"✓ Rotated: {secret_key}")
                
                # Update Docker secret file if local
                if self.provider_type == "local":
                    self._write_docker_secret(secret_key.split("/")[-1], new_value)
                    
            except Exception as e:
                failed.append(secret_key)
                logger.error(f"✗ Failed to rotate {secret_key}: {e}")
        
        logger.info(f"\nRotation complete: {len(rotated)} rotated, {len(failed)} failed")
        return len(failed) == 0
    
    def validate_secrets(self, environment: str = "production") -> bool:
        """Validate that all required secrets exist."""
        
        required_keys = [
            "app/secret_key",
            "app/jwt_secret",
            "app/encryption_key",
            "database/user",
            "database/password",
            "redis/password",
            "monitoring/grafana_password",
        ]
        
        if environment == "production":
            required_keys.extend([
                "api/openai_key",
                "api/alpaca_key",
                "api/alpaca_secret",
            ])
        
        validation = self.manager.validate_secrets(required_keys)
        
        missing = [k for k, v in validation.items() if not v]
        
        if missing:
            logger.error(f"Missing required secrets: {', '.join(missing)}")
            return False
        
        logger.info("✓ All required secrets are present")
        return True
    
    def export_env_template(self, output_file: str = ".env.secrets"):
        """Export environment variable template for secrets."""
        
        secrets_list = self.manager.list_secrets()
        
        template = """# Secrets Configuration Template
# DO NOT COMMIT THIS FILE TO VERSION CONTROL
# Generated secrets reference - actual values stored in secure provider

"""
        
        for secret_key in sorted(secrets_list):
            env_var = secret_key.upper().replace("/", "_").replace("-", "_")
            template += f"# {env_var}=<stored in {self.provider_type}>\n"
        
        template += """
# Provider Configuration
SECRETS_PROVIDER={provider}
""".format(provider=self.provider_type)
        
        Path(output_file).write_text(template)
        logger.info(f"Environment template exported to {output_file}")
    
    def backup_secrets(self, backup_file: str):
        """Backup all secrets to an encrypted file."""
        
        from cryptography.fernet import Fernet
        
        # Generate backup key
        backup_key = Fernet.generate_key()
        cipher = Fernet(backup_key)
        
        # Collect all secrets
        backup_data = {}
        for secret_key in self.manager.list_secrets():
            value = self.manager.get_secret(secret_key)
            if value:
                backup_data[secret_key] = value
        
        # Encrypt and save
        encrypted_data = cipher.encrypt(json.dumps(backup_data).encode())
        
        backup_path = Path(backup_file)
        backup_path.write_bytes(encrypted_data)
        backup_path.chmod(0o600)
        
        # Save key separately
        key_path = backup_path.with_suffix(".key")
        key_path.write_bytes(backup_key)
        key_path.chmod(0o600)
        
        logger.info(f"Backup created: {backup_file}")
        logger.info(f"Backup key saved: {key_path}")
        logger.warning("Store the backup key securely and separately from the backup file!")
        
        return True
    
    def restore_secrets(self, backup_file: str, key_file: str):
        """Restore secrets from an encrypted backup."""
        
        from cryptography.fernet import Fernet
        
        # Read backup and key
        backup_data = Path(backup_file).read_bytes()
        backup_key = Path(key_file).read_bytes()
        
        # Decrypt
        cipher = Fernet(backup_key)
        decrypted_data = cipher.decrypt(backup_data)
        secrets_data = json.loads(decrypted_data)
        
        # Restore secrets
        restored = []
        failed = []
        
        for secret_key, value in secrets_data.items():
            if self.manager.set_secret(secret_key, value):
                restored.append(secret_key)
                logger.info(f"✓ Restored: {secret_key}")
            else:
                failed.append(secret_key)
                logger.error(f"✗ Failed to restore: {secret_key}")
        
        logger.info(f"\nRestore complete: {len(restored)} restored, {len(failed)} failed")
        return len(failed) == 0


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Initialize and manage secrets")
    parser.add_argument("--provider", choices=["local", "vault", "aws", "azure", "gcp"],
                       default="local", help="Secrets provider to use")
    parser.add_argument("--environment", choices=["development", "staging", "production"],
                       default="production", help="Environment to initialize")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode for entering API keys")
    parser.add_argument("--rotate", action="store_true",
                       help="Rotate existing secrets")
    parser.add_argument("--rotate-pattern", help="Pattern for secrets to rotate")
    parser.add_argument("--validate", action="store_true",
                       help="Validate required secrets exist")
    parser.add_argument("--export-template", action="store_true",
                       help="Export environment template")
    parser.add_argument("--backup", help="Create encrypted backup of secrets")
    parser.add_argument("--restore", help="Restore secrets from backup")
    parser.add_argument("--restore-key", help="Key file for restore operation")
    
    # Provider-specific arguments
    parser.add_argument("--vault-url", help="HashiCorp Vault URL")
    parser.add_argument("--vault-token", help="HashiCorp Vault token")
    parser.add_argument("--aws-region", help="AWS region")
    parser.add_argument("--azure-vault-url", help="Azure Key Vault URL")
    parser.add_argument("--gcp-project", help="GCP project ID")
    parser.add_argument("--storage-path", help="Local storage path")
    
    args = parser.parse_args()
    
    # Build provider kwargs
    provider_kwargs = {}
    if args.vault_url:
        provider_kwargs["vault_url"] = args.vault_url
    if args.vault_token:
        provider_kwargs["vault_token"] = args.vault_token
    if args.aws_region:
        provider_kwargs["region"] = args.aws_region
    if args.azure_vault_url:
        provider_kwargs["vault_url"] = args.azure_vault_url
    if args.gcp_project:
        provider_kwargs["project_id"] = args.gcp_project
    if args.storage_path:
        provider_kwargs["storage_path"] = args.storage_path
    
    # Initialize secrets manager
    initializer = SecretsInitializer(args.provider, **provider_kwargs)
    
    # Execute requested action
    if args.backup:
        initializer.backup_secrets(args.backup)
    elif args.restore and args.restore_key:
        initializer.restore_secrets(args.restore, args.restore_key)
    elif args.rotate:
        initializer.rotate_secrets(args.rotate_pattern)
    elif args.validate:
        initializer.validate_secrets(args.environment)
    elif args.export_template:
        initializer.export_env_template()
    else:
        # Default: initialize secrets
        success = initializer.initialize_required_secrets(args.environment, args.interactive)
        if success:
            initializer.validate_secrets(args.environment)
            if args.provider == "local":
                logger.info("\nDocker secrets have been created in ./secrets/")
                logger.info("You can now start the secure stack with: docker-compose -f docker-compose.secure.yml up")


if __name__ == "__main__":
    main()
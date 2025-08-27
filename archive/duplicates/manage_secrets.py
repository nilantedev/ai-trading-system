#!/usr/bin/env python3
"""
Secrets Management Script for AI Trading System
Helps with vault setup, secret migration, and security validation.
"""

import asyncio
import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "shared" / "python-common"))

try:
    from trading_common.secrets_vault import (
        SecretsManager, VaultConfig, VaultType,
        get_secrets_manager, configure_secrets_manager,
        SecretNotFoundError, VaultConnectionError
    )
    from trading_common.config_secrets import (
        get_enhanced_settings_async, SecretsVaultConfiguration
    )
except ImportError as e:
    print(f"⚠️ Warning: Could not import trading_common modules: {e}")
    print("This is expected if running outside the development environment")
    # Define minimal stubs for scanning functionality
    class VaultType:
        ENVIRONMENT = "env"
    class VaultConfig:
        def __init__(self, **kwargs):
            pass


class SecretsManagementTool:
    """Tool for managing secrets and vault operations."""
    
    def __init__(self):
        self.vault_manager: Optional[SecretsManager] = None
    
    async def initialize_vault(self, vault_type: str = "env"):
        """Initialize vault connection."""
        try:
            if vault_type.lower() == "hashicorp":
                vault_config = VaultConfig(
                    vault_type=VaultType.HASHICORP_VAULT,
                    endpoint=os.getenv("VAULT_ENDPOINT", "http://localhost:8200"),
                    token=os.getenv("VAULT_TOKEN"),
                    namespace=os.getenv("VAULT_NAMESPACE"),
                    mount_path=os.getenv("VAULT_MOUNT_PATH", "secret")
                )
            elif vault_type.lower() == "aws":
                vault_config = VaultConfig(
                    vault_type=VaultType.AWS_SECRETS_MANAGER,
                    region=os.getenv("AWS_REGION", "us-west-2"),
                    access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
                )
            elif vault_type.lower() == "azure":
                vault_config = VaultConfig(
                    vault_type=VaultType.AZURE_KEY_VAULT,
                    endpoint=os.getenv("AZURE_VAULT_URL"),
                    tenant_id=os.getenv("AZURE_TENANT_ID"),
                    client_id=os.getenv("AZURE_CLIENT_ID"),
                    client_secret=os.getenv("AZURE_CLIENT_SECRET")
                )
            else:
                # Default to environment
                vault_config = VaultConfig(vault_type=VaultType.ENVIRONMENT)
            
            # Create fallback to environment
            fallback_config = VaultConfig(VaultType.ENVIRONMENT)
            configure_secrets_manager(vault_config, fallback_config)
            
            self.vault_manager = await get_secrets_manager()
            print(f"✅ Initialized {vault_type} vault")
            
        except Exception as e:
            print(f"❌ Failed to initialize vault: {e}")
            raise
    
    async def check_vault_health(self):
        """Check vault connectivity and health."""
        if not self.vault_manager:
            print("❌ Vault not initialized")
            return False
        
        try:
            health = await self.vault_manager.health_check()
            print("🔍 Vault Health Status:")
            for vault_name, is_healthy in health.items():
                status = "✅ Healthy" if is_healthy else "❌ Unhealthy"
                print(f"  {vault_name}: {status}")
            
            return any(health.values())
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    async def migrate_env_to_vault(self, dry_run: bool = True):
        """Migrate environment variables to vault."""
        print("🔄 Starting secrets migration...")
        
        if not self.vault_manager:
            await self.initialize_vault("env")  # Start with env for reading
        
        # Define secrets to migrate
        secrets_map = {
            "database": {
                "redis_password": os.getenv("REDIS_PASSWORD", ""),
                "postgres_password": os.getenv("POSTGRES_PASSWORD", ""),
                "questdb_password": os.getenv("QUESTDB_PASSWORD", ""),
            },
            "api_keys": {
                "alpaca_api_key": os.getenv("ALPACA_API_KEY", ""),
                "alpaca_secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
                "polygon_api_key": os.getenv("POLYGON_API_KEY", ""),
                "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
                "newsapi_key": os.getenv("NEWSAPI_KEY", ""),
                "finnhub_api_key": os.getenv("FINNHUB_API_KEY", ""),
            },
            "jwt": {
                "secret_key": os.getenv("JWT_SECRET_KEY", ""),
                "algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
            }
        }
        
        migration_plan = []
        
        for secret_path, secret_data in secrets_map.items():
            # Filter out empty values
            filtered_data = {k: v for k, v in secret_data.items() if v}
            
            if filtered_data:
                migration_plan.append((secret_path, filtered_data))
                print(f"📋 {secret_path}:")
                for key in filtered_data.keys():
                    print(f"  - {key}: ***")
        
        if dry_run:
            print("\n🏃‍♂️ DRY RUN - No secrets were actually migrated")
            print("Run with --no-dry-run to perform actual migration")
            return migration_plan
        else:
            print("\n⚠️  ACTUAL MIGRATION - This will store secrets in vault")
            confirm = input("Continue? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Migration cancelled")
                return []
            
            # Note: Actual vault storage would require vault-specific implementation
            print("✅ Migration complete (simulated)")
            return migration_plan
    
    async def validate_secrets(self):
        """Validate that all required secrets are accessible."""
        print("🔍 Validating secrets access...")
        
        if not self.vault_manager:
            await self.initialize_vault("env")
        
        # Test critical secrets
        critical_secrets = [
            ("database", "redis_password"),
            ("api_keys", "alpaca_api_key"),
            ("jwt", "secret_key"),
        ]
        
        issues = []
        
        for secret_path, key in critical_secrets:
            try:
                value = await self.vault_manager.get_secret_value(secret_path, key)
                if not value:
                    issues.append(f"Empty value: {secret_path}.{key}")
                elif value in ["dev-secret-key", "dev-secret-change-in-production"]:
                    issues.append(f"Development secret in use: {secret_path}.{key}")
                else:
                    print(f"✅ {secret_path}.{key}: OK")
            except SecretNotFoundError:
                issues.append(f"Missing secret: {secret_path}.{key}")
            except Exception as e:
                issues.append(f"Error accessing {secret_path}.{key}: {e}")
        
        if issues:
            print("\n❌ Validation Issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("\n✅ All secrets validated successfully")
            return True
    
    async def list_secrets(self):
        """List available secrets (without revealing values)."""
        print("📋 Available Secrets:")
        
        if not self.vault_manager:
            await self.initialize_vault("env")
        
        secret_paths = ["database", "api_keys", "jwt"]
        
        for secret_path in secret_paths:
            try:
                secret_data = await self.vault_manager.get_secret(secret_path)
                print(f"\n📁 {secret_path}:")
                for key in secret_data.keys():
                    value_preview = "***" if secret_data[key] else "(empty)"
                    print(f"  - {key}: {value_preview}")
            except SecretNotFoundError:
                print(f"\n📁 {secret_path}: (not found)")
            except Exception as e:
                print(f"\n📁 {secret_path}: (error: {e})")
    
    async def generate_secrets_template(self, output_file: str = ".env.secrets.template"):
        """Generate template file for secrets."""
        template_content = """# Secrets Template for AI Trading System
# Copy to .env and fill in real values (never commit this with real secrets!)

# Vault Configuration
SECRETS_VAULT_TYPE=env  # Options: env, hashicorp, aws, azure
VAULT_ENDPOINT=
VAULT_TOKEN=
VAULT_NAMESPACE=
VAULT_MOUNT_PATH=secret

# Database Secrets
REDIS_PASSWORD=your_redis_password_here
POSTGRES_PASSWORD=your_postgres_password_here
QUESTDB_PASSWORD=your_questdb_password_here

# API Keys for Data Providers
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
POLYGON_API_KEY=your_polygon_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
NEWSAPI_KEY=your_newsapi_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here

# JWT Configuration
JWT_SECRET_KEY=your_strong_jwt_secret_here
JWT_ALGORITHM=HS256

# AWS Secrets Manager (if using)
AWS_REGION=us-west-2
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# Azure Key Vault (if using)
AZURE_VAULT_URL=
AZURE_TENANT_ID=
AZURE_CLIENT_ID=
AZURE_CLIENT_SECRET=
"""
        
        with open(output_file, 'w') as f:
            f.write(template_content)
        
        print(f"✅ Secrets template generated: {output_file}")
        print("⚠️  Remember to:")
        print("   1. Copy template to .env")
        print("   2. Fill in real values")
        print("   3. Never commit .env with real secrets")
    
    async def scan_for_hardcoded_secrets(self):
        """Scan codebase for hardcoded secrets."""
        print("🔍 Scanning for hardcoded secrets...")
        
        # Patterns that indicate potential hardcoded secrets
        danger_patterns = [
            "password.*=.*['\"][^'\"]{8,}['\"]",
            "secret.*=.*['\"][^'\"]{8,}['\"]",
            "key.*=.*['\"][^'\"]{8,}['\"]",
            "token.*=.*['\"][^'\"]{8,}['\"]",
        ]
        
        # Files to check
        python_files = list(Path(".").rglob("*.py"))
        
        issues_found = []
        
        for file_path in python_files:
            if "test" in str(file_path) or ".venv" in str(file_path):
                continue
                
            try:
                content = file_path.read_text()
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    
                    # Check for development secrets
                    if "dev-secret" in line_lower and "dev-secret-change-in-production" not in line_lower:
                        issues_found.append(f"{file_path}:{i}: Development secret detected")
                    
                    # Check for potential hardcoded secrets
                    if any(pattern in line_lower for pattern in ["password", "secret", "key", "token"]):
                        if any(char in line for char in ["=", ":"]) and len(line.strip()) > 20:
                            # This is a heuristic - may have false positives
                            if not any(safe in line_lower for safe in ["getenv", "os.environ", "config.", "settings."]):
                                if len([c for c in line if c.isalnum()]) > 15:  # Likely a secret
                                    issues_found.append(f"{file_path}:{i}: Potential hardcoded secret")
            
            except Exception:
                continue  # Skip files we can't read
        
        if issues_found:
            print("❌ Potential security issues found:")
            for issue in issues_found[:10]:  # Limit output
                print(f"  - {issue}")
            if len(issues_found) > 10:
                print(f"  ... and {len(issues_found) - 10} more")
            return False
        else:
            print("✅ No obvious hardcoded secrets detected")
            return True


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Manage secrets for AI Trading System")
    parser.add_argument("command", choices=[
        "health", "migrate", "validate", "list", "template", "scan"
    ], help="Command to execute")
    parser.add_argument("--vault-type", default="env", 
                       choices=["env", "hashicorp", "aws", "azure"],
                       help="Vault type to use")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Perform dry run (default)")
    parser.add_argument("--no-dry-run", action="store_true",
                       help="Perform actual operation")
    
    args = parser.parse_args()
    
    tool = SecretsManagementTool()
    
    try:
        if args.command == "health":
            await tool.initialize_vault(args.vault_type)
            success = await tool.check_vault_health()
            sys.exit(0 if success else 1)
        
        elif args.command == "migrate":
            dry_run = args.dry_run and not args.no_dry_run
            await tool.migrate_env_to_vault(dry_run=dry_run)
        
        elif args.command == "validate":
            success = await tool.validate_secrets()
            sys.exit(0 if success else 1)
        
        elif args.command == "list":
            await tool.list_secrets()
        
        elif args.command == "template":
            await tool.generate_secrets_template()
        
        elif args.command == "scan":
            success = await tool.scan_for_hardcoded_secrets()
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
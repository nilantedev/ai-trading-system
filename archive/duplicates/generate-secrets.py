#!/usr/bin/env python3
"""
Generate secure secrets for AI Trading System production deployment.
This script creates cryptographically secure passwords and keys.
"""

import secrets
import string
import bcrypt
import os
from pathlib import Path
import argparse


def generate_password(length: int = 32, include_special: bool = True) -> str:
    """Generate a cryptographically secure password."""
    alphabet = string.ascii_letters + string.digits
    if include_special:
        alphabet += "!@#$%^&*()-_=+[]{}|;:,.<>?"
    
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    return password


def generate_api_key(length: int = 64) -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(length)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def generate_jwt_secret() -> str:
    """Generate a JWT secret key."""
    return secrets.token_urlsafe(64)


def main():
    parser = argparse.ArgumentParser(description='Generate secrets for AI Trading System')
    parser.add_argument('--output', '-o', default='.env.production.generated', 
                       help='Output file path (default: .env.production.generated)')
    parser.add_argument('--admin-password', help='Custom admin password (default: generated)')
    parser.add_argument('--force', action='store_true', 
                       help='Overwrite existing output file')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        return 1
    
    # Generate secrets
    print("🔐 Generating secure secrets for AI Trading System...")
    
    secrets_data = {
        'JWT_SECRET_KEY': generate_jwt_secret(),
        'ADMIN_PASSWORD': args.admin_password or generate_password(24),
        'GRAFANA_PASSWORD': generate_password(20),
        'MINIO_ROOT_USER': 'admin',
        'MINIO_ROOT_PASSWORD': generate_password(24),
        'REDIS_PASSWORD': generate_password(32),
        'WEAVIATE_AUTHENTICATION_APIKEY_ALLOWED_KEYS': generate_api_key(32),
    }
    
    # Hash the admin password
    admin_password_hash = hash_password(secrets_data['ADMIN_PASSWORD'])
    
    # Generate .env file content
    env_content = f"""# AI Trading System - Generated Production Secrets
# Generated on: {os.popen('date').read().strip()}
# 
# ⚠️  SECURITY WARNING: This file contains sensitive credentials!
# - Store this file securely and never commit to version control
# - Restrict file permissions: chmod 600 {output_path}
# - Use a secrets management system in production (HashiCorp Vault, AWS Secrets Manager, etc.)

# =============================================================================
# JWT Authentication & Security
# =============================================================================
JWT_SECRET_KEY={secrets_data['JWT_SECRET_KEY']}
JWT_EXPIRY_HOURS=24
ADMIN_USERNAME=admin
# Raw password (will be hashed automatically)
ADMIN_PASSWORD={secrets_data['ADMIN_PASSWORD']}
# Or use pre-hashed password (more secure)
# ADMIN_PASSWORD_HASH={admin_password_hash}

# =============================================================================
# CORS & Host Security  
# =============================================================================
CORS_ALLOWED_ORIGINS=https://trading.main-nilante.com,https://www.trading.main-nilante.com
TRUSTED_HOSTS=trading.main-nilante.com,www.trading.main-nilante.com,localhost

# =============================================================================
# Database & Storage Credentials
# =============================================================================
# Grafana Admin Password
GRAFANA_PASSWORD={secrets_data['GRAFANA_PASSWORD']}

# MinIO Object Storage
MINIO_ROOT_USER={secrets_data['MINIO_ROOT_USER']}
MINIO_ROOT_PASSWORD={secrets_data['MINIO_ROOT_PASSWORD']}

# Redis Password
REDIS_PASSWORD={secrets_data['REDIS_PASSWORD']}

# =============================================================================
# Weaviate Security
# =============================================================================
WEAVIATE_AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=false
WEAVIATE_AUTHENTICATION_APIKEY_ENABLED=true
WEAVIATE_AUTHENTICATION_APIKEY_ALLOWED_KEYS={secrets_data['WEAVIATE_AUTHENTICATION_APIKEY_ALLOWED_KEYS']}
WEAVIATE_AUTHENTICATION_APIKEY_USERS=admin

# =============================================================================
# External API Keys (FILL THESE IN MANUALLY)
# =============================================================================
# Market Data APIs
POLYGON_API_KEY=your-polygon-api-key-here
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key-here

# News APIs  
NEWS_API_KEY=your-news-api-key-here

# Broker APIs
ALPACA_API_KEY=your-alpaca-api-key-here
ALPACA_SECRET_KEY=your-alpaca-secret-key-here

# AI/ML Service Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# =============================================================================
# SSL Configuration
# =============================================================================
LETSENCRYPT_EMAIL=admin@main-nilante.com

# =============================================================================
# Performance Configuration
# =============================================================================
REDIS_MAX_MEMORY=16gb
REDIS_MAX_MEMORY_POLICY=allkeys-lru

# =============================================================================
# Rate Limiting
# =============================================================================
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST=20

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL=INFO
"""

    # Write secrets to file
    with open(output_path, 'w') as f:
        f.write(env_content)
    
    # Set secure file permissions
    os.chmod(output_path, 0o600)
    
    print(f"✅ Secrets generated successfully!")
    print(f"📁 Output file: {output_path}")
    print(f"🔒 File permissions set to 600 (owner read/write only)")
    print()
    print("📋 Generated credentials:")
    print(f"   Admin Username: admin")
    print(f"   Admin Password: {secrets_data['ADMIN_PASSWORD']}")
    print(f"   Grafana Password: {secrets_data['GRAFANA_PASSWORD']}")
    print(f"   MinIO User: {secrets_data['MINIO_ROOT_USER']}")
    print(f"   MinIO Password: {secrets_data['MINIO_ROOT_PASSWORD']}")
    print()
    print("⚠️  Next steps:")
    print("   1. Review and customize the generated .env file")
    print("   2. Fill in external API keys manually")
    print(f"   3. Copy {output_path} to your production server")
    print("   4. Deploy with: docker-compose --env-file .env.production.generated up -d")
    print()
    print("🔐 Security reminders:")
    print("   - Never commit this file to version control")
    print("   - Store in a secure location with restricted access")
    print("   - Consider using a secrets management system for production")
    print("   - Rotate credentials regularly")
    
    return 0


if __name__ == '__main__':
    exit(main())
# Configuration Migration Guide

## Overview

This guide helps you migrate from the old unprefixed environment variables to the new prefixed configuration system with enhanced security.

## 🔄 Migration Steps

### 1. Update Environment Variables

The system now uses **prefixed environment variables** for better organization and security. Update your `.env` file with the following mappings:

#### Security Settings (SECURITY_* prefix)
```bash
# OLD → NEW
JWT_SECRET_KEY → SECURITY_SECRET_KEY
JWT_ALGORITHM → SECURITY_JWT_ALGORITHM  
JWT_EXPIRY_HOURS → SECURITY_JWT_EXPIRE_MINUTES (converted to minutes)

# New additions
SECURITY_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
SECURITY_CORS_ALLOW_CREDENTIALS=true
SECURITY_RATE_LIMIT_REQUESTS=100
SECURITY_RATE_LIMIT_WINDOW=60
SECURITY_RATE_LIMIT_REQUESTS_PER_MINUTE=100
SECURITY_RATE_LIMIT_BURST=10
SECURITY_TRUSTED_HOSTS=localhost,127.0.0.1
```

#### Database Settings (DB_* prefix)
```bash
# OLD → NEW
REDIS_URL → DB_REDIS_URL
REDIS_HOST → DB_REDIS_HOST
REDIS_PORT → DB_REDIS_PORT
REDIS_PASSWORD → DB_REDIS_PASSWORD

POSTGRES_USER → DB_POSTGRES_USER
POSTGRES_PASSWORD → DB_POSTGRES_PASSWORD
POSTGRES_DB → DB_POSTGRES_DATABASE

QUESTDB_HOST → DB_QUESTDB_HOST
QUESTDB_PASSWORD → DB_QUESTDB_PASSWORD

# Vector/Graph databases
WEAVIATE_URL → DB_WEAVIATE_URL
ARANGODB_URL → DB_ARANGODB_URL
```

#### Trading Settings (TRADING_* prefix)
```bash
# OLD → NEW
ALPACA_API_KEY → TRADING_ALPACA_API_KEY
ALPACA_SECRET_KEY → TRADING_ALPACA_SECRET_KEY
POLYGON_API_KEY → TRADING_POLYGON_API_KEY

# New trading parameters
TRADING_PAPER_TRADING=true
TRADING_MAX_POSITION_SIZE=10000.0
TRADING_RISK_LIMIT_PERCENT=2.0
```

#### AI/ML Settings (AI_* prefix)
```bash
# OLD → NEW
OPENAI_API_KEY → AI_OPENAI_API_KEY
ANTHROPIC_API_KEY → AI_ANTHROPIC_API_KEY

# New AI settings
AI_LOCAL_MODEL_PATH=/models
AI_MODEL_CACHE_SIZE=32GB
AI_MAX_BATCH_SIZE=16
AI_INFERENCE_TIMEOUT=30
```

#### Message Broker Settings (MSG_* prefix)
```bash
# OLD → NEW
PULSAR_URL → MSG_PULSAR_URL
PULSAR_TENANT → MSG_PULSAR_TENANT
PULSAR_NAMESPACE → MSG_PULSAR_NAMESPACE
```

#### Monitoring Settings (MONITORING_* prefix)
```bash
# OLD → NEW
LOG_LEVEL → MONITORING_LOG_LEVEL
PROMETHEUS_PORT → MONITORING_PROMETHEUS_PORT

# New monitoring settings
MONITORING_HEALTH_CHECK_INTERVAL=30
MONITORING_OTEL_EXPORTER_ENDPOINT=
MONITORING_OTEL_SERVICE_NAME=ai-trading-system
```

### 2. Generate Secure Secrets

For production deployments, generate secure secrets:

#### Generate JWT Secret
```bash
# Generate a secure 32+ character secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env
SECURITY_SECRET_KEY=<generated_secret>
```

#### Hash Admin Password
```bash
# Generate bcrypt hash for admin password
python -c "from passlib.context import CryptContext; print(CryptContext(schemes=['bcrypt']).hash('your_secure_password'))"

# Add to .env
ADMIN_PASSWORD_HASH=<generated_hash>
# Remove any plain ADMIN_PASSWORD entries
```

### 3. Update Feature Flags

Feature flags now use a consistent pattern:

```bash
# Feature flag format: FEATURE_<NAME>_ENABLED
FEATURE_PHD_INTELLIGENCE_ENABLED=true
FEATURE_SOCIAL_MEDIA_ENABLED=true
FEATURE_COMPANY_INTELLIGENCE_ENABLED=true
FEATURE_OFF_HOURS_TRAINING_ENABLED=true
```

### 4. Validate Configuration

The system now validates configuration on startup:

```python
# Development mode - warnings only
ENVIRONMENT=development

# Production mode - fails on critical issues
ENVIRONMENT=production
```

**Production validation checks:**
- JWT secret is not default and ≥32 characters
- Admin password is hashed (not plain text)
- Required trading API keys are configured
- No default values for critical secrets

### 5. Test Migration

After updating your configuration:

```bash
# Test configuration loading
python -c "
from trading_common import get_settings
settings = get_settings()
issues = settings.validate_production_config()
print(f'Configuration issues: {issues}')
"

# Start the system to verify
docker-compose up api
```

## 🔒 Security Improvements

### JWT Security
- **Unified secret management** through SecuritySettings
- **Production validation** prevents weak secrets
- **Automatic expiry** now in minutes (more granular control)

### Authentication Updates
- **get_optional_user()** for public endpoints
- **Compatibility aliases** for legacy code
- **Enhanced error handling** with proper status codes

### Security Headers
All API responses now include:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Content-Security-Policy` restrictions
- `Strict-Transport-Security` (production only)

### Production Safety
- **Fail-fast deployment** with insecure configuration
- **Automatic validation** on startup
- **Clear error messages** for configuration issues

## 📝 Example .env File

```bash
# Environment
ENVIRONMENT=production
DEBUG=false

# Security (REQUIRED - generate secure values)
SECURITY_SECRET_KEY=your-32-character-minimum-secret-key-here
SECURITY_JWT_ALGORITHM=HS256
SECURITY_JWT_EXPIRE_MINUTES=30
SECURITY_CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
SECURITY_CORS_ALLOW_CREDENTIALS=true
SECURITY_RATE_LIMIT_REQUESTS=100
SECURITY_RATE_LIMIT_WINDOW=60
SECURITY_TRUSTED_HOSTS=yourdomain.com,api.yourdomain.com

# Trading (REQUIRED for trading features)
TRADING_ALPACA_API_KEY=your_alpaca_key
TRADING_ALPACA_SECRET_KEY=your_alpaca_secret
TRADING_POLYGON_API_KEY=your_polygon_key
TRADING_PAPER_TRADING=false  # Set to false for live trading
TRADING_MAX_POSITION_SIZE=50000.0
TRADING_RISK_LIMIT_PERCENT=2.0

# Databases
DB_REDIS_URL=redis://:password@redis:6379/0
DB_REDIS_PASSWORD=secure_redis_password
DB_POSTGRES_URL=postgresql://user:pass@postgres:5432/trading
DB_QUESTDB_PASSWORD=secure_questdb_password

# Admin (use hash, not plain password)
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=$2b$12$hashedPasswordHere

# Features
FEATURE_PHD_INTELLIGENCE_ENABLED=true
FEATURE_SOCIAL_MEDIA_ENABLED=true
```

## ⚠️ Breaking Changes

1. **JWT_EXPIRY_HOURS → JWT_EXPIRE_MINUTES**: Time unit changed from hours to minutes
2. **Direct os.getenv() calls removed**: All config through unified Settings
3. **Production validation enforced**: System won't start with insecure defaults
4. **UserRole now Enum**: Proper Pydantic enum for user roles

## 🔧 Rollback Plan

If you need to temporarily use old configuration:

1. The system maintains backward compatibility with deprecation warnings
2. Old unprefixed variables still work but log warnings
3. Plan to migrate within 30 days before support is removed

## 📞 Support

For configuration issues:
1. Check validation errors in logs
2. Run configuration validator script
3. Ensure all prefixed variables are set
4. Verify secure values for production
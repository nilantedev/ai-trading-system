# Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the AI Trading System to production environments.

## Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 LTS or Ubuntu 24.04 LTS
- **CPU**: 16+ cores recommended
- **RAM**: 32GB+ recommended
- **Storage**: 500GB+ SSD
- **Network**: Static IP, ports 80/443 open

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- PostgreSQL 15+
- Redis 7+
- Python 3.11+
- Node.js 18+ (for dashboard)

## Pre-Deployment Checklist

### 1. Environment Configuration
```bash
# Copy and configure environment files
cp .env.production.template .env.production

# Edit with production values
nano .env.production
```

**Required Environment Variables:**
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `SECRET_KEY` - Strong random secret for JWT
- `API_KEYS` - External API credentials (Alpaca, Polygon, etc.)
- `VAULT_URL` - HashiCorp Vault or secrets manager URL

### 2. Security Setup
```bash
# Generate strong secrets
python scripts/generate_secrets.py

# Setup SSL certificates
certbot certonly --webroot -w /var/www/certbot \
  -d yourdomain.com \
  -d api.yourdomain.com

# Configure firewall
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw enable
```

### 3. Database Initialization
```bash
# Create production database
createdb trading_system_production

# Run migrations
alembic upgrade head

# Initialize admin user
python scripts/init_database.py --production
```

## Deployment Steps

### 1. Build Production Images
```bash
# Build all services
make build-production

# Or build individually
docker build -f docker/Dockerfile.api -t trading-api:latest .
docker build -f docker/Dockerfile.worker -t trading-worker:latest .
```

### 2. Deploy Infrastructure
```bash
# Start infrastructure services
docker-compose -f docker-compose.infrastructure.yml up -d

# Verify services
docker-compose -f docker-compose.infrastructure.yml ps
```

### 3. Deploy Application
```bash
# Deploy application services
docker-compose -f docker-compose.production.yml up -d

# Check deployment status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

### 4. Setup Monitoring
```bash
# Deploy monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Configure Grafana dashboards
./scripts/setup_dashboards.sh

# Setup alerting
./scripts/configure_alerts.sh
```

### 5. Configure Backups
```bash
# Setup automated backups
./scripts/setup_backup_cron.sh

# Test backup system
python scripts/disaster_recovery.py --action backup

# Start DR monitoring
./scripts/start_dr_monitoring.sh
```

## Health Verification

### System Health Check
```bash
# Run comprehensive health check
make health-check-production

# Check individual services
curl https://api.yourdomain.com/health
curl https://api.yourdomain.com/api/v1/system/status
```

### Security Verification
```bash
# Run security audit
make security-audit

# Check SSL configuration
nmap --script ssl-enum-ciphers -p 443 yourdomain.com

# Verify authentication
curl -X POST https://api.yourdomain.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"secure_password"}'
```

## Rollback Procedures

### Quick Rollback (< 5 minutes)
```bash
# Stop current deployment
docker-compose -f docker-compose.production.yml down

# Restore previous version
docker-compose -f docker-compose.production.yml up -d --scale api=0
docker tag trading-api:backup trading-api:latest
docker-compose -f docker-compose.production.yml up -d

# Verify rollback
make health-check-production
```

### Database Rollback
```bash
# Stop application
docker-compose -f docker-compose.production.yml stop

# Restore database backup
pg_restore -d trading_system_production /backups/latest.dump

# Restart application
docker-compose -f docker-compose.production.yml start
```

### Emergency Recovery
```bash
# Trigger emergency backup
python scripts/disaster_recovery.py --action backup

# Full system restore
python scripts/disaster_recovery.py --action restore --backup-id <backup_id>

# Verify recovery
make health-check-production
```

## Post-Deployment

### 1. Monitoring Setup
- Access Grafana: https://monitoring.yourdomain.com
- Default credentials: admin/admin (change immediately)
- Import dashboards from `monitoring/dashboards/`

### 2. Log Aggregation
```bash
# View application logs
docker logs trading-api

# View audit logs
docker exec trading-api python -m trading_common.audit_logger --show-recent

# Setup log rotation
cat > /etc/logrotate.d/trading-system << EOF
/var/log/trading-system/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 trading trading
}
EOF
```

### 3. Performance Tuning
```bash
# Database optimization
psql -d trading_system_production -c "ANALYZE;"
psql -d trading_system_production -c "REINDEX DATABASE trading_system_production;"

# Redis optimization
redis-cli CONFIG SET maxmemory 4gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## Maintenance

### Daily Tasks
- Review Grafana dashboards
- Check backup completion
- Review audit logs for anomalies
- Verify external API connectivity

### Weekly Tasks
- Review compliance reports
- Update security patches
- Performance analysis
- Capacity planning review

### Monthly Tasks
- Disaster recovery test
- Security audit
- Dependency updates
- Documentation review

## Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check PostgreSQL status
systemctl status postgresql

# Test connection
psql -h localhost -U trading_user -d trading_system_production

# Check logs
tail -f /var/log/postgresql/postgresql-15-main.log
```

**High Memory Usage**
```bash
# Check memory usage
docker stats

# Restart specific service
docker-compose -f docker-compose.production.yml restart api

# Clear Redis cache if needed
redis-cli FLUSHDB
```

**API Performance Issues**
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s https://api.yourdomain.com/health

# Review slow queries
docker exec trading-api python -m trading_common.performance_monitor

# Check circuit breaker status
curl https://api.yourdomain.com/api/v1/resilience/status
```

## Security Considerations

1. **Secrets Management**
   - Never commit secrets to git
   - Use HashiCorp Vault or AWS Secrets Manager
   - Rotate secrets regularly

2. **Network Security**
   - Use private networks for internal services
   - Implement rate limiting
   - Enable CORS with specific origins

3. **Data Protection**
   - Encrypt data at rest
   - Use TLS 1.3 for transit
   - Implement audit logging

4. **Access Control**
   - Use strong passwords
   - Enable 2FA for admin accounts
   - Regular access reviews

## Support

For production issues:
1. Check logs in `/var/log/trading-system/`
2. Review monitoring dashboards
3. Consult disaster recovery documentation
4. Contact system administrator

## Appendix

### Useful Commands
```bash
# View all running containers
docker ps

# Check disk usage
df -h

# Monitor system resources
htop

# Check network connections
ss -tulpn

# View systemd logs
journalctl -xe
```

### Configuration Files
- Main config: `.env.production`
- Backup config: `config/backup_config.json`
- DR config: `config/disaster_recovery_config.json`
- Docker config: `docker-compose.production.yml`

---

Last Updated: November 26, 2024  
Version: 3.0.0-production
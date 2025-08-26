# 🚀 Production Deployment Checklist

## ✅ Pre-Deployment Verification

### Security
- [ ] All secrets removed from code (NO hardcoded passwords)
- [ ] `.env.production` created with ALL required values
- [ ] No placeholder values in `.env.production` (no "REPLACE_WITH" or "your-key")
- [ ] All passwords are strong (32+ characters)
- [ ] API keys are valid and tested

### Testing
- [ ] Unit tests passing: `pytest tests/unit -v`
- [ ] Integration tests passing: `pytest tests/integration/test_critical_trading_flows.py -v`
- [ ] No security vulnerabilities: `grep -r "password\|secret" *.yml`

### Infrastructure
- [ ] Server accessible: `ssh root@168.119.145.135`
- [ ] Docker installed on server
- [ ] Sufficient disk space on `/mnt/bulkdata` for backups
- [ ] Ports available: 8000 (API), 3000 (Grafana), 9090 (Prometheus)

## 📋 Deployment Steps

### 1. Prepare Environment
```bash
# Copy and fill production environment
cp .env.production.template .env.production
nano .env.production  # Fill ALL required values

# Generate secure secrets
openssl rand -hex 32  # For SECRET_KEY
openssl rand -hex 32  # For JWT_SECRET
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"  # ENCRYPTION_KEY
```

### 2. Run Tests
```bash
# Run all critical tests
pytest tests/unit -v
pytest tests/integration/test_critical_trading_flows.py -v
```

### 3. Deploy to Production
```bash
# Deploy with all checks
./deploy_production.sh

# Or skip tests if already run
./deploy_production.sh --skip-tests
```

### 4. Post-Deployment Setup

#### Enable Automated Backups
```bash
ssh root@168.119.145.135
cd /srv/trading
sudo systemctl enable backup.timer
sudo systemctl start backup.timer
sudo systemctl status backup.timer
```

#### Verify Services
```bash
# Check all services are running
docker-compose ps

# Check API health
curl http://localhost:8000/health

# Check logs
docker-compose logs --tail=50
```

## 🔍 Monitoring

### Access Points
- **API**: http://168.119.145.135:8000
- **Grafana**: http://168.119.145.135:3000 (login with GRAFANA_USER/GRAFANA_PASSWORD)
- **Prometheus**: http://168.119.145.135:9090

### Critical Metrics to Monitor
- [ ] API response time < 1 second
- [ ] Database connections available
- [ ] No error rate spikes
- [ ] Memory usage < 80%
- [ ] Disk usage < 85%

### Log Monitoring
```bash
# Watch real-time logs
ssh root@168.119.145.135 'cd /srv/trading && docker-compose logs -f'

# Check specific service
ssh root@168.119.145.135 'cd /srv/trading && docker-compose logs api --tail=100'
```

## 🔄 Daily Operations

### Backup Verification
```bash
# Check last backup
ssh root@168.119.145.135 'ls -la /mnt/bulkdata/backups/'

# View backup report
ssh root@168.119.145.135 'cat /mnt/bulkdata/backups/latest_backup_report.txt'
```

### Manual Backup (if needed)
```bash
ssh root@168.119.145.135 'cd /srv/trading && ./scripts/automated_backup.sh'
```

## 🚨 Emergency Procedures

### Rollback Deployment
```bash
# On server
cd /srv/trading
./scripts/automated_recovery.sh  # Uses latest backup
```

### Stop Trading Immediately
```bash
ssh root@168.119.145.135 'cd /srv/trading && docker-compose stop api'
```

### Full System Recovery
```bash
# Restore from specific backup
ssh root@168.119.145.135 'cd /srv/trading && ./scripts/automated_recovery.sh 20240826_020000'
```

## ⚠️ Important Reminders

1. **NEVER** commit `.env.production` to git
2. **ALWAYS** test with paper trading first
3. **MONITOR** closely for first 24 hours
4. **BACKUP** before any major changes
5. **DOCUMENT** any manual interventions

## 📞 Support Contacts

- System Admin: [Your contact]
- Database Admin: [Your contact]
- On-call Engineer: [Your contact]

## ✅ Sign-off

- [ ] All checks completed
- [ ] System monitored for 24 hours
- [ ] Backup tested successfully
- [ ] Ready for live trading

---

Deployment Date: _______________
Deployed By: _______________
Version: _______________
# 🚀 Production Deployment Checklist

## 📍 Command Location Guide
- 💻 **LOCAL** = Run on your development machine (your laptop/desktop)
- 🖥️ **SERVER** = Run on production server (168.119.145.135)

---

## ✅ Pre-Deployment Verification

### Security Checks
- [ ] All secrets removed from code (NO hardcoded passwords)
- [ ] `.env.production` created with ALL required values
- [ ] No placeholder values in `.env.production` (no "REPLACE_WITH" or "your-key")
- [ ] All passwords are strong (32+ characters)
- [ ] API keys are valid and tested

### Testing 💻 **LOCAL**
```bash
# Run these on your LOCAL development machine
cd ~/main-nilante-server/ai-trading-system

# Run unit tests
pytest tests/unit -v

# Run integration tests
pytest tests/integration/test_critical_trading_flows.py -v

# Check for hardcoded secrets
grep -r "password\|secret" *.yml --exclude=".env.production.template"
```

### Infrastructure Verification 💻 **LOCAL**
```bash
# Test SSH connection from LOCAL to SERVER
ssh root@168.119.145.135 "echo 'Connection successful'"

# Check server disk space from LOCAL
ssh root@168.119.145.135 "df -h"

# Verify Docker is installed on server
ssh root@168.119.145.135 "docker --version && docker-compose --version"
```

---

## 📋 Deployment Steps

### 1. Prepare Environment 💻 **LOCAL**

```bash
# On your LOCAL development machine
cd ~/main-nilante-server/ai-trading-system

# Copy and fill production environment
cp .env.production.template .env.production
nano .env.production  # Fill ALL required values

# Generate secure secrets (run these locally and copy the output)
openssl rand -hex 32  # For SECRET_KEY
openssl rand -hex 32  # For JWT_SECRET
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"  # ENCRYPTION_KEY
openssl rand -base64 32  # For DB_PASSWORD
openssl rand -base64 32  # For REDIS_PASSWORD
```

### 2. Run Tests 💻 **LOCAL**

```bash
# Still on your LOCAL machine
cd ~/main-nilante-server/ai-trading-system

# Run all critical tests
pytest tests/unit -v
pytest tests/integration/test_critical_trading_flows.py -v

# Verify Docker builds successfully
docker build -t ai-trading-test .
```

### 3. Deploy to Production 💻 **LOCAL**

```bash
# Run deployment script from your LOCAL machine
cd ~/main-nilante-server/ai-trading-system

# Full deployment with all checks
./deploy_production.sh

# OR skip tests if already run
./deploy_production.sh --skip-tests

# OR skip backup if you just backed up
./deploy_production.sh --skip-backup
```

### 4. Post-Deployment Setup 🖥️ **SERVER**

#### Enable Automated Backups 🖥️ **SERVER**
```bash
# SSH into the SERVER
ssh root@168.119.145.135

# Now you're on the SERVER - run these commands there:
cd /srv/trading

# Install backup service
sudo cp scripts/backup.service /etc/systemd/system/
sudo cp scripts/backup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable backup.timer
sudo systemctl start backup.timer

# Verify timer is active
sudo systemctl status backup.timer

# Exit back to LOCAL
exit
```

#### Verify Services 🖥️ **SERVER**
```bash
# SSH into the SERVER
ssh root@168.119.145.135

# Check all services are running (on SERVER)
cd /srv/trading
docker-compose ps

# Check API health (on SERVER)
curl http://localhost:8000/health

# Check logs (on SERVER)
docker-compose logs --tail=50 api

# Exit back to LOCAL
exit
```

---

## 🔍 Monitoring

### Access Points 💻 **LOCAL**
Open these URLs in your browser on your LOCAL machine:
- **API**: http://168.119.145.135:8000
- **API Docs**: http://168.119.145.135:8000/docs
- **Grafana**: http://168.119.145.135:3000 (login with GRAFANA_USER/GRAFANA_PASSWORD)
- **Prometheus**: http://168.119.145.135:9090

### Critical Metrics to Monitor 💻 **LOCAL**
From your LOCAL machine, check these via browser or curl:
```bash
# Check API metrics from LOCAL
curl http://168.119.145.135:8000/metrics

# Check Prometheus targets from LOCAL
curl http://168.119.145.135:9090/api/v1/targets
```

### Log Monitoring 

#### Option 1: From LOCAL Machine 💻 **LOCAL**
```bash
# Watch logs remotely from your LOCAL machine
ssh root@168.119.145.135 'cd /srv/trading && docker-compose logs -f'

# Check specific service from LOCAL
ssh root@168.119.145.135 'cd /srv/trading && docker-compose logs api --tail=100'
```

#### Option 2: Directly on SERVER 🖥️ **SERVER**
```bash
# SSH into server first
ssh root@168.119.145.135

# Then run on SERVER:
cd /srv/trading
docker-compose logs -f

# Check specific service
docker-compose logs api --tail=100
```

---

## 🔄 Daily Operations

### Backup Verification

#### From LOCAL 💻 **LOCAL**
```bash
# Check last backup from your LOCAL machine
ssh root@168.119.145.135 'ls -la /mnt/bulkdata/backups/ | tail -5'

# View backup report from LOCAL
ssh root@168.119.145.135 'cat /mnt/bulkdata/backups/latest_backup_report.txt'
```

#### From SERVER 🖥️ **SERVER**
```bash
# SSH into server first
ssh root@168.119.145.135

# Check backups on SERVER
ls -la /mnt/bulkdata/backups/
cat /mnt/bulkdata/backups/latest_backup_report.txt
```

### Manual Backup

#### Trigger from LOCAL 💻 **LOCAL**
```bash
# Run backup remotely from LOCAL
ssh root@168.119.145.135 'cd /srv/trading && ./scripts/automated_backup.sh'
```

#### Run on SERVER 🖥️ **SERVER**
```bash
# SSH into server first
ssh root@168.119.145.135

# Run backup on SERVER
cd /srv/trading
./scripts/automated_backup.sh
```

---

## 🚨 Emergency Procedures

### Stop Trading Immediately 💻 **LOCAL**
```bash
# Emergency stop from your LOCAL machine
ssh root@168.119.145.135 'cd /srv/trading && docker-compose stop api'
```

### Rollback Deployment

#### From LOCAL 💻 **LOCAL**
```bash
# Trigger recovery remotely from LOCAL
ssh root@168.119.145.135 'cd /srv/trading && ./scripts/automated_recovery.sh'

# Restore specific backup from LOCAL
ssh root@168.119.145.135 'cd /srv/trading && ./scripts/automated_recovery.sh 20240826_020000'
```

#### From SERVER 🖥️ **SERVER**
```bash
# SSH into server first
ssh root@168.119.145.135

# Run recovery on SERVER
cd /srv/trading
./scripts/automated_recovery.sh  # Uses latest backup

# Or restore specific backup
./scripts/automated_recovery.sh 20240826_020000
```

### Debug Failed Deployment 🖥️ **SERVER**
```bash
# SSH into server
ssh root@168.119.145.135

# Check what's wrong (run these on SERVER)
cd /srv/trading
docker-compose ps -a  # See all containers including stopped
docker-compose logs --tail=100  # Check recent logs
df -h  # Check disk space
free -m  # Check memory
```

---

## 📝 Quick Reference

### Where to Run What:

| Task | Location | Command |
|------|----------|---------|
| Deploy | 💻 LOCAL | `./deploy_production.sh` |
| Run Tests | 💻 LOCAL | `pytest tests/` |
| Check Logs | 💻 LOCAL | `ssh root@168.119.145.135 'docker-compose logs'` |
| Check Logs | 🖥️ SERVER | `docker-compose logs` |
| Backup | 💻 LOCAL | `ssh root@168.119.145.135 './scripts/automated_backup.sh'` |
| Backup | 🖥️ SERVER | `./scripts/automated_backup.sh` |
| Recovery | 💻 LOCAL | `ssh root@168.119.145.135 './scripts/automated_recovery.sh'` |
| Recovery | 🖥️ SERVER | `./scripts/automated_recovery.sh` |
| Stop Services | 💻 LOCAL | `ssh root@168.119.145.135 'docker-compose stop'` |
| Stop Services | 🖥️ SERVER | `docker-compose stop` |
| Check Health | 💻 LOCAL | `curl http://168.119.145.135:8000/health` |
| Check Health | 🖥️ SERVER | `curl http://localhost:8000/health` |

---

## ⚠️ Important Reminders

### Development Machine (LOCAL) 💻
- This is where you write code
- This is where you run the deployment script
- This is where you run tests before deploying
- Keep `.env.production` secure here

### Production Server (SERVER) 🖥️
- This is where the application runs (168.119.145.135)
- Only log in here for maintenance/debugging
- Automated backups run here
- Docker containers run here

### Security Rules
1. **NEVER** edit code directly on the SERVER
2. **NEVER** commit `.env.production` to git
3. **ALWAYS** deploy from LOCAL using the script
4. **ALWAYS** test on LOCAL before deploying
5. **MONITOR** from LOCAL (via browser/curl)
6. **DEBUG** on SERVER (when needed)

---

## 📞 Support & Troubleshooting

### Common Issues and Solutions

#### "Permission denied" when running deploy script 💻 **LOCAL**
```bash
# Make script executable on LOCAL
chmod +x deploy_production.sh
```

#### "Cannot connect to server" 💻 **LOCAL**
```bash
# Check SSH from LOCAL
ssh -v root@168.119.145.135  # Verbose mode to see issues
```

#### "No space left on device" 🖥️ **SERVER**
```bash
# Clean up on SERVER
ssh root@168.119.145.135
docker system prune -a  # Remove unused Docker data
rm -rf /tmp/*  # Clear temp files
```

#### "Container won't start" 🖥️ **SERVER**
```bash
# Debug on SERVER
ssh root@168.119.145.135
cd /srv/trading
docker-compose logs [service_name]  # Check specific service logs
docker-compose down  # Stop everything
docker-compose up -d  # Start fresh
```

---

## ✅ Final Sign-off

### Pre-Deployment Checklist
- [ ] All tests pass on LOCAL
- [ ] `.env.production` filled on LOCAL
- [ ] SSH access works from LOCAL to SERVER
- [ ] Sufficient disk space on SERVER

### Post-Deployment Checklist
- [ ] API responds (check from LOCAL)
- [ ] Grafana accessible (check from LOCAL)
- [ ] Backups enabled (verify on SERVER)
- [ ] Logs look clean (check from SERVER or LOCAL)

---

**Remember**: 
- 💻 Deploy FROM your local machine
- 🖥️ Application RUNS ON the server
- 💻 Monitor FROM your local machine
- 🖥️ Debug ON the server (when needed)

Deployment Date: _______________
Deployed By: _______________
Version: _______________
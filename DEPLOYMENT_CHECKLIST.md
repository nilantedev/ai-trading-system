# 🚀 Production Deployment Checklist

## 📍 Command Location Guide
- 💻 **LOCAL** = Run on your development machine (your laptop/desktop)
- 🖥️ **SERVER** = Run on production server (168.119.145.135)

## 👤 User Account for Deployment
**RECOMMENDED: Use `nilante` (sudo user) instead of `root` for better security**

### Why Use `nilante` Instead of `root`?
- ✅ **Security**: Never expose root account unnecessarily
- ✅ **Audit Trail**: All actions are logged to your user
- ✅ **Best Practice**: Follow principle of least privilege
- ✅ **Safer**: Reduces risk of accidental system damage

### Initial Server Setup (One Time Only) 🖥️ **SERVER**
```bash
# If not already done, create the nilante user with sudo privileges
# Run this as root only once
ssh root@168.119.145.135
adduser nilante
usermod -aG sudo nilante
usermod -aG docker nilante  # Important: Add to docker group

# Create deployment directory with correct permissions
sudo mkdir -p /srv/trading
sudo chown nilante:nilante /srv/trading

# Create backup directory
sudo mkdir -p /mnt/bulkdata/backups
sudo chown nilante:nilante /mnt/bulkdata/backups

# Set up SSH key for nilante user
su - nilante
mkdir -p ~/.ssh
# Exit back to your local machine
exit
exit

# From LOCAL, copy your SSH key to nilante user
ssh-copy-id nilante@168.119.145.135
```

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
# Test SSH connection from LOCAL to SERVER (as nilante user)
ssh nilante@168.119.145.135 "echo 'Connection successful'"

# Check server disk space from LOCAL
ssh nilante@168.119.145.135 "df -h"

# Verify Docker is installed and nilante can use it
ssh nilante@168.119.145.135 "docker --version && docker-compose --version"

# Verify nilante has sudo access
ssh nilante@168.119.145.135 "sudo -n true && echo 'Sudo access confirmed'"
```

---

## 📋 Deployment Steps

### 1. Update Deployment Script for nilante User 💻 **LOCAL**

First, update the deployment script to use `nilante` instead of `root`:

```bash
# On your LOCAL machine
cd ~/main-nilante-server/ai-trading-system

# Edit the deployment script
nano deploy_production.sh

# Change these lines:
# FROM: SERVER_USER="${SERVER_USER:-root}"
# TO:   SERVER_USER="${SERVER_USER:-nilante}"
```

### 2. Prepare Environment 💻 **LOCAL**

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

### 3. Run Tests 💻 **LOCAL**

```bash
# Still on your LOCAL machine
cd ~/main-nilante-server/ai-trading-system

# Run all critical tests
pytest tests/unit -v
pytest tests/integration/test_critical_trading_flows.py -v

# Verify Docker builds successfully
docker build -t ai-trading-test .
```

### 4. Deploy to Production 💻 **LOCAL**

```bash
# Run deployment script from your LOCAL machine (now using nilante user)
cd ~/main-nilante-server/ai-trading-system

# Set the server user explicitly
export SERVER_USER=nilante

# Full deployment with all checks
./deploy_production.sh

# OR skip tests if already run
./deploy_production.sh --skip-tests

# OR skip backup if you just backed up
./deploy_production.sh --skip-backup
```

### 5. Post-Deployment Setup 🖥️ **SERVER**

#### Enable Automated Backups 🖥️ **SERVER**
```bash
# SSH into the SERVER as nilante user
ssh nilante@168.119.145.135

# Now you're on the SERVER as nilante - run these commands:
cd /srv/trading

# Install backup service (requires sudo)
sudo cp scripts/backup.service /etc/systemd/system/
sudo cp scripts/backup.timer /etc/systemd/system/

# Update service file to run as nilante user
sudo sed -i 's/User=root/User=nilante/g' /etc/systemd/system/backup.service
sudo sed -i 's/Group=root/Group=nilante/g' /etc/systemd/system/backup.service

# Reload and enable
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
# SSH into the SERVER as nilante
ssh nilante@168.119.145.135

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
# Watch logs remotely from your LOCAL machine (using nilante user)
ssh nilante@168.119.145.135 'cd /srv/trading && docker-compose logs -f'

# Check specific service from LOCAL
ssh nilante@168.119.145.135 'cd /srv/trading && docker-compose logs api --tail=100'
```

#### Option 2: Directly on SERVER 🖥️ **SERVER**
```bash
# SSH into server as nilante
ssh nilante@168.119.145.135

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
ssh nilante@168.119.145.135 'ls -la /mnt/bulkdata/backups/ | tail -5'

# View backup report from LOCAL
ssh nilante@168.119.145.135 'cat /mnt/bulkdata/backups/latest_backup_report.txt'
```

#### From SERVER 🖥️ **SERVER**
```bash
# SSH into server as nilante
ssh nilante@168.119.145.135

# Check backups on SERVER
ls -la /mnt/bulkdata/backups/
cat /mnt/bulkdata/backups/latest_backup_report.txt
```

### Manual Backup

#### Trigger from LOCAL 💻 **LOCAL**
```bash
# Run backup remotely from LOCAL
ssh nilante@168.119.145.135 'cd /srv/trading && ./scripts/automated_backup.sh'
```

#### Run on SERVER 🖥️ **SERVER**
```bash
# SSH into server as nilante
ssh nilante@168.119.145.135

# Run backup on SERVER
cd /srv/trading
./scripts/automated_backup.sh
```

---

## 🚨 Emergency Procedures

### Stop Trading Immediately 💻 **LOCAL**
```bash
# Emergency stop from your LOCAL machine
ssh nilante@168.119.145.135 'cd /srv/trading && docker-compose stop api'
```

### Rollback Deployment

#### From LOCAL 💻 **LOCAL**
```bash
# Trigger recovery remotely from LOCAL
ssh nilante@168.119.145.135 'cd /srv/trading && ./scripts/automated_recovery.sh'

# Restore specific backup from LOCAL
ssh nilante@168.119.145.135 'cd /srv/trading && ./scripts/automated_recovery.sh 20240826_020000'
```

#### From SERVER 🖥️ **SERVER**
```bash
# SSH into server as nilante
ssh nilante@168.119.145.135

# Run recovery on SERVER
cd /srv/trading
./scripts/automated_recovery.sh  # Uses latest backup

# Or restore specific backup
./scripts/automated_recovery.sh 20240826_020000
```

### Debug Failed Deployment 🖥️ **SERVER**
```bash
# SSH into server as nilante
ssh nilante@168.119.145.135

# Check what's wrong (run these on SERVER)
cd /srv/trading
docker-compose ps -a  # See all containers including stopped
docker-compose logs --tail=100  # Check recent logs
df -h  # Check disk space
free -m  # Check memory

# If permission issues with Docker
sudo usermod -aG docker nilante
newgrp docker  # Refresh group membership
```

---

## 🔐 Permission Management

### Common Permission Fixes 🖥️ **SERVER**

#### Docker Permission Issues
```bash
# If nilante can't access Docker
sudo usermod -aG docker nilante
# Log out and back in, or run:
newgrp docker
```

#### File Permission Issues
```bash
# If files are owned by root after deployment
cd /srv/trading
sudo chown -R nilante:nilante .

# Make scripts executable
chmod +x scripts/*.sh
```

#### Systemd Service Permissions
```bash
# For services that need to run as nilante
sudo systemctl edit backup.service

# Add or modify:
[Service]
User=nilante
Group=nilante
```

---

## 📝 Quick Reference

### Where to Run What (Using nilante User):

| Task | Location | Command |
|------|----------|---------|
| Deploy | 💻 LOCAL | `SERVER_USER=nilante ./deploy_production.sh` |
| Run Tests | 💻 LOCAL | `pytest tests/` |
| SSH to Server | 💻 LOCAL | `ssh nilante@168.119.145.135` |
| Check Logs | 💻 LOCAL | `ssh nilante@168.119.145.135 'docker-compose logs'` |
| Check Logs | 🖥️ SERVER | `docker-compose logs` |
| Backup | 💻 LOCAL | `ssh nilante@168.119.145.135 './scripts/automated_backup.sh'` |
| Backup | 🖥️ SERVER | `./scripts/automated_backup.sh` |
| Recovery | 💻 LOCAL | `ssh nilante@168.119.145.135 './scripts/automated_recovery.sh'` |
| Recovery | 🖥️ SERVER | `./scripts/automated_recovery.sh` |
| Stop Services | 💻 LOCAL | `ssh nilante@168.119.145.135 'docker-compose stop'` |
| Stop Services | 🖥️ SERVER | `docker-compose stop` |
| Check Health | 💻 LOCAL | `curl http://168.119.145.135:8000/health` |
| Check Health | 🖥️ SERVER | `curl http://localhost:8000/health` |
| Use Sudo | 🖥️ SERVER | `sudo systemctl restart docker` |

---

## ⚠️ Important Reminders

### Development Machine (LOCAL) 💻
- This is where you write code
- This is where you run the deployment script
- This is where you run tests before deploying
- Keep `.env.production` secure here
- Always use `nilante@168.119.145.135` for SSH

### Production Server (SERVER) 🖥️
- This is where the application runs (168.119.145.135)
- Log in as `nilante` user (NOT root)
- Use `sudo` only when necessary
- Automated backups run as `nilante` user
- Docker containers managed by `nilante` user

### Security Best Practices
1. **NEVER** use root for routine deployments
2. **ALWAYS** use `nilante` user with sudo when needed
3. **NEVER** edit code directly on the SERVER
4. **NEVER** commit `.env.production` to git
5. **ALWAYS** deploy from LOCAL using the script
6. **ALWAYS** test on LOCAL before deploying
7. **MONITOR** from LOCAL (via browser/curl)
8. **DEBUG** on SERVER as `nilante` user

### User Account Summary
- **nilante**: Your main deployment and management user
  - Has sudo privileges for system tasks
  - Member of docker group for container management
  - Owns `/srv/trading` directory
  - Runs automated backups

- **root**: Only use for initial server setup
  - Creating users
  - System-level configuration
  - Emergency recovery (if nilante is locked out)

---

## 📞 Support & Troubleshooting

### Common Issues and Solutions

#### "Permission denied" when running deploy script 💻 **LOCAL**
```bash
# Make script executable on LOCAL
chmod +x deploy_production.sh

# Ensure using correct user
export SERVER_USER=nilante
```

#### "Cannot connect to server" 💻 **LOCAL**
```bash
# Check SSH from LOCAL
ssh -v nilante@168.119.145.135  # Verbose mode to see issues

# If nilante doesn't work, ensure user exists
ssh root@168.119.145.135 "id nilante"
```

#### "Permission denied" for Docker 🖥️ **SERVER**
```bash
# On SERVER as nilante
sudo usermod -aG docker nilante
# Log out and back in, or:
newgrp docker
```

#### "No space left on device" 🖥️ **SERVER**
```bash
# Clean up on SERVER as nilante
docker system prune -a  # Remove unused Docker data
sudo rm -rf /tmp/*  # Clear temp files (needs sudo)
```

#### "Container won't start" 🖥️ **SERVER**
```bash
# Debug on SERVER as nilante
ssh nilante@168.119.145.135
cd /srv/trading
docker-compose logs [service_name]  # Check specific service logs
docker-compose down  # Stop everything
docker-compose up -d  # Start fresh
```

#### "Cannot write to /srv/trading" 🖥️ **SERVER**
```bash
# Fix ownership on SERVER
ssh nilante@168.119.145.135
sudo chown -R nilante:nilante /srv/trading
```

---

## ✅ Final Sign-off

### Pre-Deployment Checklist
- [ ] All tests pass on LOCAL
- [ ] `.env.production` filled on LOCAL
- [ ] SSH access works: `ssh nilante@168.119.145.135`
- [ ] nilante user has docker access on SERVER
- [ ] Sufficient disk space on SERVER

### Post-Deployment Checklist
- [ ] API responds (check from LOCAL)
- [ ] Grafana accessible (check from LOCAL)
- [ ] Backups enabled (verify on SERVER as nilante)
- [ ] Logs look clean (check as nilante user)
- [ ] File ownership correct (`nilante:nilante`)

---

**Remember**: 
- 💻 Deploy FROM your local machine
- 🖥️ Application RUNS ON the server
- 👤 Use `nilante` user, NOT root
- 🔐 Use `sudo` only when necessary
- 💻 Monitor FROM your local machine
- 🖥️ Debug ON the server as `nilante`

Deployment Date: _______________
Deployed By: nilante
Version: _______________
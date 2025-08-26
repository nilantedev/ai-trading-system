# 🚀 AI Trading System - Complete Deployment Guide

## 📦 Current Deployment Method: Script-Based

We use a deployment script that handles everything from your local machine to the production server.

---

## 🔧 ONE-TIME SETUP (Do This First!)

### 1. Setup SSH Access to Your Server
```bash
# Add your SSH key to the server (if not already done)
ssh-copy-id root@168.119.145.135

# Test connection
ssh root@168.119.145.135 "echo 'Connection successful'"
```

### 2. Prepare Server Environment
```bash
# SSH into your server
ssh root@168.119.145.135

# Install Docker and Docker Compose
apt-get update
apt-get install -y docker.io docker-compose git

# Create deployment directory
mkdir -p /srv/trading
mkdir -p /mnt/bulkdata/backups

# Exit server
exit
```

### 3. Create Production Secrets File
```bash
# On your LOCAL machine
cd ai-trading-system

# Copy template
cp .env.production.template .env.production

# Generate secure passwords
echo "Generate these secure values:"
echo "SECRET_KEY=$(openssl rand -hex 32)"
echo "JWT_SECRET=$(openssl rand -hex 32)"
echo "DB_PASSWORD=$(openssl rand -base64 32)"
echo "REDIS_PASSWORD=$(openssl rand -base64 32)"
echo "GRAFANA_PASSWORD=$(openssl rand -base64 24)"
echo "BACKUP_ENCRYPTION_KEY=$(openssl rand -hex 32)"

# Edit .env.production and fill in ALL values
nano .env.production
```

**CRITICAL VALUES TO SET:**
```env
# Database (REQUIRED)
DB_USER=trading_admin
DB_PASSWORD=[generated password from above]
DB_NAME=trading_system

# Security Keys (REQUIRED)
SECRET_KEY=[generated key from above]
JWT_SECRET=[generated key from above]

# Redis (REQUIRED)
REDIS_PASSWORD=[generated password from above]

# Monitoring (REQUIRED)
GRAFANA_USER=admin
GRAFANA_PASSWORD=[generated password from above]

# Trading APIs (REQUIRED for trading)
ALPACA_API_KEY=[your actual API key]
ALPACA_SECRET=[your actual secret]
POLYGON_API_KEY=[your actual API key]
OPENAI_API_KEY=[your actual API key]

# Backup (REQUIRED)
BACKUP_ENCRYPTION_KEY=[generated key from above]

# Server Settings (REQUIRED)
CORS_ORIGINS=https://your-domain.com
TRUSTED_HOSTS=your-domain.com,168.119.145.135
LETSENCRYPT_EMAIL=your-email@domain.com
```

---

## 🎯 DEPLOYMENT STEPS (Every Time You Deploy)

### Method 1: Full Deployment (Recommended)
```bash
# From your LOCAL machine, in the ai-trading-system directory
cd ai-trading-system

# Run the deployment script
./deploy_production.sh

# This will:
# 1. Check for required files
# 2. Verify no placeholder secrets
# 3. Run tests locally
# 4. Build Docker images
# 5. Create backup on server
# 6. Upload files to server
# 7. Deploy on server
# 8. Verify deployment
# 9. Setup monitoring
```

### Method 2: Quick Deployment (After First Time)
```bash
# Skip tests if you've already run them
./deploy_production.sh --skip-tests

# Skip backup if you just backed up
./deploy_production.sh --skip-backup

# Skip both
./deploy_production.sh --skip-tests --skip-backup
```

---

## 🔄 ALTERNATIVE: GitHub Actions CI/CD Setup

### Option A: GitHub Actions Automated Deployment

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:  # Allow manual trigger

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      
      - name: Run tests
        run: |
          pytest tests/unit -v
          pytest tests/integration/test_critical_trading_flows.py -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to server
        env:
          SERVER_HOST: ${{ secrets.SERVER_HOST }}
          SERVER_USER: ${{ secrets.SERVER_USER }}
          SERVER_SSH_KEY: ${{ secrets.SERVER_SSH_KEY }}
        run: |
          # Setup SSH
          mkdir -p ~/.ssh
          echo "$SERVER_SSH_KEY" > ~/.ssh/deploy_key
          chmod 600 ~/.ssh/deploy_key
          
          # Add to known hosts
          ssh-keyscan -H $SERVER_HOST >> ~/.ssh/known_hosts
          
          # Deploy
          scp -i ~/.ssh/deploy_key -r ./* $SERVER_USER@$SERVER_HOST:/srv/trading/
          
          # Restart services
          ssh -i ~/.ssh/deploy_key $SERVER_USER@$SERVER_HOST \
            "cd /srv/trading && docker-compose down && docker-compose up -d"
```

### Setting Up GitHub Actions:

1. **Go to GitHub Repository Settings**:
   - Navigate to: https://github.com/nilantedev/ai-trading-system/settings/secrets/actions

2. **Add Repository Secrets**:
   ```
   SERVER_HOST: 168.119.145.135
   SERVER_USER: root
   SERVER_SSH_KEY: [Your private SSH key content]
   ```

3. **Add Production Secrets**:
   ```
   DB_PASSWORD: [Your secure password]
   JWT_SECRET: [Your JWT secret]
   REDIS_PASSWORD: [Your Redis password]
   # ... add all from .env.production
   ```

---

## 🐳 ALTERNATIVE: Docker Hub Deployment

### Setup Docker Hub:

1. **Build and Push to Docker Hub**:
```bash
# Login to Docker Hub
docker login

# Build image
docker build -t yourusername/ai-trading-system:latest .

# Push to Docker Hub
docker push yourusername/ai-trading-system:latest
```

2. **On Production Server**:
```bash
# Pull latest image
docker pull yourusername/ai-trading-system:latest

# Run with docker-compose
docker-compose up -d
```

---

## 🔍 VERIFY DEPLOYMENT

### 1. Check Services Are Running
```bash
# SSH to server
ssh root@168.119.145.135

# Check Docker containers
docker-compose ps

# Should see:
# trading-api         running   0.0.0.0:8000->8000/tcp
# trading-postgres    running   5432/tcp
# trading-redis       running   6379/tcp
# trading-prometheus  running   0.0.0.0:9090->9090/tcp
# trading-grafana     running   0.0.0.0:3000->3000/tcp
```

### 2. Test API Health
```bash
# From your local machine
curl http://168.119.145.135:8000/health

# Should return:
# {"status":"healthy","timestamp":"..."}
```

### 3. Access Web Interfaces
- **API Docs**: http://168.119.145.135:8000/docs
- **Grafana**: http://168.119.145.135:3000 (login with GRAFANA_USER/PASSWORD)
- **Prometheus**: http://168.119.145.135:9090

### 4. Check Logs
```bash
# On server
cd /srv/trading
docker-compose logs --tail=50 api
```

---

## 🔄 POST-DEPLOYMENT SETUP

### 1. Enable Automated Backups
```bash
ssh root@168.119.145.135

# Install backup service
cd /srv/trading
sudo cp scripts/backup.service /etc/systemd/system/
sudo cp scripts/backup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable backup.timer
sudo systemctl start backup.timer

# Verify timer is active
sudo systemctl status backup.timer
```

### 2. Setup Monitoring Alerts (Optional)
```bash
# Configure Prometheus alerts
cd /srv/trading/infrastructure/docker/prometheus
nano alert.rules.yml

# Restart Prometheus
docker-compose restart prometheus
```

### 3. Test Manual Backup
```bash
# Run backup manually to test
cd /srv/trading
./scripts/automated_backup.sh

# Check backup was created
ls -la /mnt/bulkdata/backups/
```

---

## 🚨 TROUBLESHOOTING

### If Deployment Fails:

1. **Check Logs**:
```bash
# Check deployment script output
# Look for error messages

# On server, check Docker logs
ssh root@168.119.145.135
cd /srv/trading
docker-compose logs
```

2. **Common Issues**:

**Issue: "DB_PASSWORD environment variable is required"**
```bash
# You forgot to set .env.production
# Make sure .env.production exists on server
scp .env.production root@168.119.145.135:/srv/trading/
```

**Issue: "Connection refused"**
```bash
# Firewall might be blocking ports
ssh root@168.119.145.135
ufw allow 8000  # API
ufw allow 3000  # Grafana
ufw allow 9090  # Prometheus
```

**Issue: "No space left on device"**
```bash
# Check disk space
ssh root@168.119.145.135
df -h
# Clean up Docker
docker system prune -a
```

3. **Emergency Rollback**:
```bash
ssh root@168.119.145.135
cd /srv/trading
./scripts/automated_recovery.sh  # Restore from last backup
```

---

## 📊 MONITORING DEPLOYMENT

### Watch Real-Time Logs:
```bash
# From local machine
ssh root@168.119.145.135 'cd /srv/trading && docker-compose logs -f api'
```

### Monitor Resource Usage:
```bash
ssh root@168.119.145.135
docker stats
```

### Check Application Metrics:
1. Open Grafana: http://168.119.145.135:3000
2. Default dashboard shows:
   - API request rate
   - Response times
   - Error rates
   - System resources

---

## ✅ DEPLOYMENT CHECKLIST

Before each deployment:
- [ ] All tests passing locally
- [ ] No hardcoded secrets in code
- [ ] .env.production filled with real values
- [ ] Recent backup exists
- [ ] No active trades running

After deployment:
- [ ] API health check passes
- [ ] Can login to Grafana
- [ ] No errors in logs
- [ ] Automated backup scheduled
- [ ] Monitor for 1 hour

---

## 🔐 SECURITY REMINDERS

1. **NEVER** commit `.env.production` to Git
2. **NEVER** use default passwords
3. **ALWAYS** use HTTPS in production (setup reverse proxy with nginx)
4. **ALWAYS** backup before major updates
5. **ALWAYS** test with paper trading first

---

## 📞 NEED HELP?

If deployment fails:
1. Check this guide's troubleshooting section
2. Review logs: `docker-compose logs`
3. Check server resources: `df -h`, `free -m`
4. Verify all secrets are set correctly
5. Ensure ports are not blocked by firewall

Remember: The deployment script does everything for you. Just run:
```bash
./deploy_production.sh
```

And it handles the entire deployment process!
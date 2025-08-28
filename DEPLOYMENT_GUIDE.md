# 🚀 AI TRADING SYSTEM - COMPLETE DEPLOYMENT GUIDE

**IMPORTANT**: This system is configured for PAPER TRADING ONLY until fully tested.  
**Server**: germ1-ain-nilante.com (168.119.145.135)  
**GitHub**: https://github.com/nilantedev/ai-trading-system  

---

## 📋 BEFORE YOU START

### What You Need Ready:
1. ✅ GitHub repository created (private recommended)
2. ✅ Your Alpaca PAPER trading API keys (get from https://alpaca.markets)
3. ✅ SSH access to server: `ssh -i ~/.ssh/hetzner_admin -p 2222 nilante@germ1-ain-nilante.com`
4. ✅ About 30 minutes for deployment

### Critical Safety Notes:
- 🔴 System starts in PAPER TRADING mode (no real money)
- 🔴 All safety features are ENABLED by default
- 🔴 Auto-trading is DISABLED - you must manually enable
- 🔴 Kill switch is ACTIVE and tested

---

## 🎯 STEP-BY-STEP DEPLOYMENT

### Step 1: Push Code to GitHub
```bash
# ON YOUR LOCAL MACHINE (in VSCode terminal)
cd /home/nilante/main-nilante-server/ai-trading-system

# Initialize git if needed
git init
git add .
git commit -m "Initial deployment - paper trading mode with safety features"

# Add your GitHub repository
git remote add origin https://github.com/nilantedev/ai-trading-system.git
git branch -M main
git push -u origin main
```

### Step 2: Connect to Server
```bash
# From your LOCAL machine
ssh -i ~/.ssh/hetzner_admin -p 2222 nilante@germ1-ain-nilante.com
```

### Step 3: Install Docker (if not already installed)
```bash
# Check if Docker exists
docker --version

# If not installed, install it:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker nilante

# IMPORTANT: Log out and log back in for docker permissions
exit
ssh -i ~/.ssh/hetzner_admin -p 2222 nilante@germ1-ain-nilante.com
```

### Step 4: Create Required Directories
```bash
# Application directory (fast SSD)
sudo mkdir -p /srv/ai-trading-system
sudo chown nilante:nilante /srv/ai-trading-system

# Database storage (fast NVMe)
sudo mkdir -p /mnt/fastdrive/postgres /mnt/fastdrive/redis
sudo chown -R nilante:nilante /mnt/fastdrive/

# Backup storage (large HDD)
sudo mkdir -p /mnt/bulkdata/trading-backups
sudo chown nilante:nilante /mnt/bulkdata/trading-backups

# Logs directory
sudo mkdir -p /var/log/trading-system
sudo chown nilante:nilante /var/log/trading-system
```

### Step 5: Clone Repository from GitHub
```bash
cd /srv
git clone https://github.com/nilantedev/ai-trading-system.git
cd ai-trading-system
```

### Step 6: Setup Python Environment
```bash
# Install Python 3.11 if needed
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install -e shared/python-common/
```

### Step 7: Configure Environment Variables
```bash
# Copy the template
cp .env .env.production

# Edit with your REAL API keys
nano .env.production
```

**Add these keys (replace with your actual keys):**
```
# CRITICAL: These should be your PAPER trading keys!
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Optional data provider keys (can add later)
POLYGON_API_KEY=your_key_here_if_you_have
ALPHA_VANTAGE_API_KEY=your_key_here_if_you_have
NEWS_API_KEY=your_key_here_if_you_have
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### Step 8: Configure Docker Storage Paths
```bash
# Create Docker override for production paths
cat > docker-compose.override.yml << 'EOF'
version: '3.9'
services:
  postgres:
    volumes:
      - /mnt/fastdrive/postgres:/var/lib/postgresql/data
  redis:
    volumes:
      - /mnt/fastdrive/redis:/data
  api:
    volumes:
      - /var/log/trading-system:/app/logs
    env_file:
      - .env.production
EOF
```

### Step 9: Start Database Services
```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Wait for them to start
sleep 30

# Check they're running
docker ps
```

### Step 10: Initialize Database
```bash
# Activate virtual environment
source .venv/bin/activate

# Run database migrations
alembic upgrade head
```

### Step 11: Run Safety Checks
```bash
# Run the safety startup script
python start_safe_trading.py
```

**You should see:**
```
✓ Checking trading mode... Mode: PAPER
✓ Testing kill switch... Kill switch tested successfully
✓ Checking risk limits... Max position: $1000, Max daily loss: $200
✓ Testing data validation... Data validator working correctly
✅ ALL SAFETY CHECKS PASSED
```

### Step 12: Start the Trading System
```bash
# Start in detached mode
docker-compose up -d api

# Or run directly to see output (for testing)
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Step 13: Verify Everything Works
```bash
# Check health endpoint
curl http://localhost:8000/health

# You should see:
# {"status":"healthy","timestamp":"..."}

# Check services are running
docker ps

# You should see 3 containers:
# - trading-postgres
# - trading-redis  
# - trading-api

# Check API docs
# Open in browser: http://germ1-ain-nilante.com:8000/docs
```

---

## 🎮 CONTROL PANEL

### Access the API Documentation
Open in your browser: `http://germ1-ain-nilante.com:8000/docs`

### Emergency Stop (Kill Switch)
```bash
curl -X POST http://localhost:8000/api/v1/governor/emergency-stop \
  -H "Content-Type: application/json" \
  -d '{"reason": "Manual emergency stop"}'
```

### Check System Status
```bash
curl http://localhost:8000/api/v1/governor/state
```

### Enable Auto-Trading (When Ready)
```bash
# ONLY do this after monitoring paper trading!
curl -X POST http://localhost:8000/api/v1/governor/setting \
  -H "Content-Type: application/json" \
  -d '{"key": "auto_trade_enabled", "value": true}'
```

---

## 📊 MONITORING

### View Logs
```bash
# API logs
docker-compose logs -f api

# Database logs
docker-compose logs postgres

# All logs
docker-compose logs -f
```

### Check Resource Usage
```bash
# Memory and CPU
htop

# Disk space
df -h

# Docker stats
docker stats
```

---

## 🚨 TROUBLESHOOTING

### If API Won't Start
```bash
# Check logs
docker-compose logs api | tail -50

# Common fix: restart everything
docker-compose down
docker-compose up -d

# Check Python version
python3.11 --version  # Should be 3.11.x
```

### If Database Connection Fails
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Test connection
docker exec -it trading-postgres psql -U trading_user -d trading_system

# Restart if needed
docker-compose restart postgres
```

### If Kill Switch is Stuck
```bash
# Check Redis
docker exec -it trading-redis redis-cli
> GET trading:kill_switch:active
> DEL trading:kill_switch:active  # Only if you need to clear it
```

---

## ✅ POST-DEPLOYMENT CHECKLIST

### Immediate (First 10 Minutes)
- [ ] API health check working
- [ ] All Docker containers running
- [ ] Can access API docs in browser
- [ ] Logs showing no errors

### First Hour
- [ ] Verify paper trading mode active
- [ ] Test kill switch works
- [ ] Check risk limits are conservative
- [ ] Verify data validation working
- [ ] Review audit trail is logging

### First Day
- [ ] Monitor for any errors
- [ ] Check memory/CPU usage stable
- [ ] Verify no real money at risk
- [ ] Test paper trades executing (if enabled)
- [ ] Review all safety systems active

---

## 🔐 SECURITY REMINDERS

1. **NEVER** commit `.env.production` to git
2. **NEVER** change from paper to live trading without extensive testing
3. **ALWAYS** test the kill switch before enabling auto-trading
4. **MONITOR** the system closely for first 48 hours
5. **BACKUP** your configuration before any changes

---

## 📞 COMMANDS REFERENCE

### Start System
```bash
cd /srv/ai-trading-system
docker-compose up -d
```

### Stop System
```bash
docker-compose down
```

### Restart System
```bash
docker-compose restart
```

### Update from GitHub
```bash
git pull origin main
docker-compose down
docker-compose build
docker-compose up -d
```

### Emergency Stop
```bash
# Via API
curl -X POST http://localhost:8000/api/v1/governor/emergency-stop \
  -d '{"reason": "Emergency"}'

# Via Docker (nuclear option)
docker-compose down
```

---

## 🎯 NEXT STEPS AFTER DEPLOYMENT

1. **Monitor for 24 Hours**: Watch logs, check for errors
2. **Test Paper Trading**: Enable auto-trading in paper mode
3. **Verify Safety Systems**: Test kill switch, risk limits
4. **Tune Parameters**: Adjust position sizes, risk limits
5. **Add Data Sources**: Configure Polygon, Alpha Vantage APIs (optional)

---

## ⚠️ WARNING: TRANSITIONING TO REAL MONEY

**DO NOT** switch to real money trading until:
- ✅ 30+ days of profitable paper trading
- ✅ All safety systems tested under load
- ✅ Risk limits verified working
- ✅ Kill switch tested multiple times
- ✅ Audit trail verified immutable
- ✅ Backup and recovery tested
- ✅ You have 24/7 monitoring in place

To switch to real trading (ONLY when ready):
1. Get LIVE API keys from Alpaca
2. Update `.env.production` with live keys
3. Change `ALPACA_BASE_URL` to `https://api.alpaca.markets`
4. Set conservative risk limits
5. Start with minimal capital ($1000 max)

---

**Remember**: The system is designed to be SAFE FIRST. All dangerous features are disabled by default. Take your time, test thoroughly, and only enable features as you verify they work correctly.

**Support**: If you need help during deployment, I can see your terminal output and guide you through any issues.

Good luck! 🚀
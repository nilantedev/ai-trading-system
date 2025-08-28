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

### Step 1: Push Code to GitHub (ALREADY DONE ✅)
Code is already on GitHub at: https://github.com/nilantedev/ai-trading-system

### Step 2: Connect to Server (ALREADY DONE ✅)
You're already connected via SSH!

### Step 3: Install Docker
```bash
# YOU SHOULD BE IN: /home/nilante
# Check where you are:
pwd
# If not in /home/nilante, run:
cd ~

# Check if Docker exists
docker --version

# If you see "command not found", install Docker:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add yourself to docker group
sudo usermod -aG docker nilante

# Install docker-compose
sudo apt update
sudo apt install -y docker-compose

# IMPORTANT: Log out and log back in for docker permissions
exit
# Then SSH back in:
ssh -i ~/.ssh/hetzner_admin -p 2222 nilante@germ1-ain-nilante.com
```

### Step 4: Create Required Directories
```bash
# YOU SHOULD BE IN: /home/nilante
# Check:
pwd
# If not there:
cd ~

# Now create all directories:

# Application directory (fast SSD)
sudo mkdir -p /srv/ai-trading-system
sudo chown nilante:nilante /srv/ai-trading-system
ls -la /srv/  # Verify it's created

# Database storage (fast NVMe)
sudo mkdir -p /mnt/fastdrive/postgres
sudo mkdir -p /mnt/fastdrive/redis
sudo chown -R nilante:nilante /mnt/fastdrive/
ls -la /mnt/fastdrive/  # Verify they're created

# Backup storage (large HDD)
sudo mkdir -p /mnt/bulkdata/trading-backups
sudo chown nilante:nilante /mnt/bulkdata/trading-backups
ls -la /mnt/bulkdata/  # Verify it's created

# Logs directory
sudo mkdir -p /var/log/trading-system
sudo chown nilante:nilante /var/log/trading-system
ls -la /var/log/ | grep trading  # Verify it's created
```

### Step 5: Clone Repository from GitHub
```bash
# YOU SHOULD BE IN: /home/nilante
pwd

# Go to /srv directory
cd /srv
pwd  # Should show: /srv

# Clone the repository
git clone https://github.com/nilantedev/ai-trading-system.git

# Enter the project directory
cd ai-trading-system
pwd  # Should show: /srv/ai-trading-system

# Verify files are there
ls -la
# You should see: api/, services/, docker-compose.yml, etc.
```

### Step 6: Setup Python Environment
```bash
# YOU MUST BE IN: /srv/ai-trading-system
pwd  # Should show: /srv/ai-trading-system
# If not:
cd /srv/ai-trading-system

# Install Python 3.11 if needed
python3 --version
# If it's not 3.11.x, install it:
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip

# Create virtual environment IN THE PROJECT DIRECTORY
python3.11 -m venv .venv

# Activate it
source .venv/bin/activate

# Your prompt should now show (.venv) at the beginning
# Like: (.venv) nilante@main-nilante:/srv/ai-trading-system$

# Upgrade pip
pip install --upgrade pip

# Install dependencies (this takes 2-3 minutes)
pip install -r requirements.txt

# Install shared libraries
pip install -e shared/python-common/
```

### Step 7: Configure Environment Variables
```bash
# YOU MUST BE IN: /srv/ai-trading-system
pwd  # Should show: /srv/ai-trading-system
# If not:
cd /srv/ai-trading-system

# Copy the template
cp .env .env.production

# Check it was created
ls -la .env*
# You should see both .env and .env.production

# Edit with your REAL API keys
nano .env.production
```

**In the nano editor, add these (replace with YOUR keys):**
```
# CRITICAL: These should be your PAPER trading keys!
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database passwords (keep these as-is for now)
DB_PASSWORD=trading_password_123
REDIS_PASSWORD=redis_password_123

# Security (keep as-is for now)
SECRET_KEY=your-secret-key-here-change-in-production
JWT_SECRET=your-jwt-secret-here-change-in-production

# Optional data provider keys (can add later)
POLYGON_API_KEY=demo_key
ALPHA_VANTAGE_API_KEY=demo_key
NEWS_API_KEY=demo_key
```

**To save in nano:**
1. Press `Ctrl+X`
2. Press `Y` (for yes)
3. Press `Enter`

### Step 8: Configure Docker Storage Paths
```bash
# YOU MUST BE IN: /srv/ai-trading-system
pwd  # Should show: /srv/ai-trading-system
# If not:
cd /srv/ai-trading-system

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

# Verify it was created
cat docker-compose.override.yml
```

### Step 9: Start Database Services
```bash
# YOU MUST BE IN: /srv/ai-trading-system
pwd  # Should show: /srv/ai-trading-system
# If not:
cd /srv/ai-trading-system

# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Wait for them to start (important!)
sleep 30

# Check they're running
docker ps
# You should see 2 containers running (postgres and redis)

# If you don't see them, check logs:
docker-compose logs postgres
docker-compose logs redis
```

### Step 10: Initialize Database
```bash
# YOU MUST BE IN: /srv/ai-trading-system
pwd  # Should show: /srv/ai-trading-system
# If not:
cd /srv/ai-trading-system

# Make sure virtual environment is active
# You should see (.venv) in your prompt
# If not:
source .venv/bin/activate

# Run database migrations
alembic upgrade head

# You should see output like:
# INFO  [alembic.runtime.migration] Running upgrade...
```

### Step 11: Run Safety Checks
```bash
# YOU MUST BE IN: /srv/ai-trading-system
pwd  # Should show: /srv/ai-trading-system
# If not:
cd /srv/ai-trading-system

# Make sure virtual environment is active
source .venv/bin/activate

# Run the safety startup script
python start_safe_trading.py

# You should see:
# ✓ Checking trading mode... Mode: PAPER
# ✓ Testing kill switch... Kill switch tested successfully
# ✅ ALL SAFETY CHECKS PASSED
```

### Step 12: Start the Trading System
```bash
# YOU MUST BE IN: /srv/ai-trading-system
pwd  # Should show: /srv/ai-trading-system
# If not:
cd /srv/ai-trading-system

# Option A: Start with Docker (recommended)
docker-compose up -d api

# Check it's running
docker ps
# You should now see 3 containers (postgres, redis, api)

# OR Option B: Run directly to see output (for testing)
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Press Ctrl+C to stop when done testing
```

### Step 13: Verify Everything Works
```bash
# YOU CAN BE IN ANY DIRECTORY FOR THESE TESTS

# Check health endpoint
curl http://localhost:8000/health

# You should see:
# {"status":"healthy","timestamp":"..."}

# If that works, check from outside (from your local machine)
# Open a NEW terminal on your LOCAL machine and run:
curl http://germ1-ain-nilante.com:8000/health

# Check services are running
docker ps
# You should see 3 containers: postgres, redis, api

# Check logs for errors
docker-compose logs --tail=50 api
```

---

## 🎮 AFTER DEPLOYMENT - CONTROL COMMANDS

### View Logs (from /srv/ai-trading-system)
```bash
cd /srv/ai-trading-system
docker-compose logs -f api  # Follow API logs
docker-compose logs -f      # Follow all logs
```

### Stop System (from /srv/ai-trading-system)
```bash
cd /srv/ai-trading-system
docker-compose down
```

### Restart System (from /srv/ai-trading-system)
```bash
cd /srv/ai-trading-system
docker-compose restart
```

### Emergency Stop (from anywhere)
```bash
# Kill switch
curl -X POST http://localhost:8000/api/v1/governor/emergency-stop \
  -H "Content-Type: application/json" \
  -d '{"reason": "Manual emergency stop"}'

# Nuclear option - stop everything
cd /srv/ai-trading-system
docker-compose down
```

### Check System Status (from anywhere)
```bash
curl http://localhost:8000/api/v1/governor/state
```

### Update from GitHub (from /srv/ai-trading-system)
```bash
cd /srv/ai-trading-system
git pull origin main
docker-compose down
docker-compose build
docker-compose up -d
```

---

## 🚨 TROUBLESHOOTING

### If Docker install fails:
```bash
# Try manual install
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
```

### If "Permission denied" on docker commands:
```bash
# You need to logout and login
exit
ssh -i ~/.ssh/hetzner_admin -p 2222 nilante@germ1-ain-nilante.com
```

### If API won't start:
```bash
cd /srv/ai-trading-system
# Check logs
docker-compose logs api | tail -100
# Common fix: restart everything
docker-compose down
docker-compose up -d
```

### If database connection fails:
```bash
cd /srv/ai-trading-system
# Check PostgreSQL is running
docker ps | grep postgres
# Restart it
docker-compose restart postgres
# Wait 30 seconds
sleep 30
# Try API again
docker-compose restart api
```

### If port 8000 is already in use:
```bash
# Find what's using it
sudo lsof -i :8000
# Kill the process (replace PID with actual number)
sudo kill -9 PID
```

---

## ✅ SUCCESS INDICATORS

You know deployment worked when:
1. ✅ `docker ps` shows 3 containers running
2. ✅ `curl http://localhost:8000/health` returns `{"status":"healthy"...}`
3. ✅ No errors in `docker-compose logs api`
4. ✅ Can access http://germ1-ain-nilante.com:8000/docs in browser

---

## 📞 QUICK COMMAND REFERENCE

**Always run from: `/srv/ai-trading-system`**
```bash
cd /srv/ai-trading-system     # Go to app directory
docker ps                      # Check what's running
docker-compose logs -f         # View logs
docker-compose down            # Stop everything
docker-compose up -d           # Start everything
source .venv/bin/activate      # Activate Python environment
```

---

**Remember**: 
- Every command shows WHERE you should be
- Use `pwd` to check your current directory
- Use `cd /srv/ai-trading-system` to get to the app directory
- The system starts in PAPER TRADING mode (safe!)

**I can see your terminal and will help if anything goes wrong!**
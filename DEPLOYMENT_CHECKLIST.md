# 🚀 AI TRADING SYSTEM - DEPLOYMENT CHECKLIST

## 📍 Server Storage Locations
- **Application:** `/srv/ai-trading-system/` (1.5TB NVMe - FAST)
- **PostgreSQL:** `/mnt/fastdrive/postgres/` (1.8TB NVMe - FAST) 
- **Redis:** `/mnt/fastdrive/redis/` (1.8TB NVMe - FAST)
- **Backups:** `/mnt/bulkdata/trading-backups/` (15TB HDD - LARGE)
- **Logs:** `/var/log/trading-system/`

## ✅ Pre-Deployment (LOCAL MACHINE)
- [ ] All changes committed to git
- [ ] Push to GitHub: `git push origin main`
- [ ] Have API keys ready (don't commit them!)
- [ ] production_keys.txt file ready

## 🖥️ Server Deployment Steps

### 1️⃣ Connect to Server
```bash
ssh -i ~/.ssh/hetzner_admin -p 2222 nilante@germ1-ain-nilante.com
```

### 2️⃣ Clone Repository
```bash
cd /srv
git clone https://github.com/nilantedev/ai-trading-system.git
cd ai-trading-system
```

### 3️⃣ Create Storage Directories
```bash
# Database storage (fast NVMe)
sudo mkdir -p /mnt/fastdrive/postgres /mnt/fastdrive/redis
sudo chown -R nilante:nilante /mnt/fastdrive/

# Backup storage (large HDD)
sudo mkdir -p /mnt/bulkdata/trading-backups
sudo chown nilante:nilante /mnt/bulkdata/trading-backups

# Logs
sudo mkdir -p /var/log/trading-system
sudo chown nilante:nilante /var/log/trading-system
```

### 4️⃣ Setup Environment
```bash
# Copy .env template
cp .env .env.production

# Edit and add your REAL API keys
nano .env.production

# Add these keys:
ALPACA_API_KEY=your_real_key
ALPACA_SECRET_KEY=your_real_secret
POLYGON_API_KEY=your_real_key
ALPHA_VANTAGE_API_KEY=your_real_key
REDDIT_CLIENT_ID=your_real_id
REDDIT_CLIENT_SECRET=your_real_secret
NEWS_API_KEY=your_real_key
```

### 5️⃣ Install Python Dependencies
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e shared/python-common/
```

### 6️⃣ Configure Docker Storage
```bash
# Create override file for production paths
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
EOF
```

### 7️⃣ Start Services
```bash
# Start databases
docker-compose up -d postgres redis

# Wait for startup
sleep 30

# Initialize database
source .venv/bin/activate
alembic upgrade head

# Test API
uvicorn api.main:app --host 0.0.0.0 --port 8000
# (Ctrl+C to stop after testing)

# Start production
docker-compose up -d
```

### 8️⃣ Verify Deployment
```bash
# Check services
docker ps

# Test health endpoint
curl http://localhost:8000/health

# Check logs
docker-compose logs -f api
```

## 🔍 What I Can See in Terminal
YES! When you SSH from VSCode terminal, I can:
- ✅ See all commands you type
- ✅ See command outputs
- ✅ Help debug any issues
- ✅ Guide you step-by-step

## 📝 Important Notes
1. **NEVER** commit .env.production with real keys
2. Use `/srv/` for app (1.5TB fast SSD)
3. Use `/mnt/fastdrive/` for databases (1.8TB fast SSD)
4. Use `/mnt/bulkdata/` for backups (15TB HDD)
5. Check disk space: `df -h`
6. Monitor resources: `htop`

## 🚨 If Something Goes Wrong
```bash
# Check logs
docker-compose logs api
docker-compose logs postgres

# Restart services
docker-compose down
docker-compose up -d

# Check disk space
df -h

# Check memory
free -h
```

## ✅ Post-Deployment
- [ ] API responding on port 8000
- [ ] Database connections working
- [ ] Redis cache operational
- [ ] API keys validated
- [ ] Data feeds connected
- [ ] Monitoring enabled

Ready to deploy! 🚀
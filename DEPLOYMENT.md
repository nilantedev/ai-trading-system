# 🚀 AI Trading System - Production Deployment Guide

## 🎯 DEPLOYMENT INFORMATION
**Time Required**: 45-60 minutes  
**Server**: 168.119.145.135  
**Domain**: trading.main-nilante.com  
**User**: nilante  
**Path**: /srv/trading  

---

## ⚡ QUICK START (If you know what you're doing)
```bash
# 1. SSH to server
ssh nilante@168.119.145.135

# 2. Clone and deploy
git clone https://github.com/nilantedev/ai-trading-system.git /srv/trading
cd /srv/trading
cp .env.example .env.production
# [Edit .env.production with secure values]
./deploy_production.sh --skip-tests

# 3. Verify
curl http://localhost:8000/health
```

---

## 📋 PHASE 1: PRE-DEPLOYMENT (5 min)

### 1.1 Local Preparation
```bash
# Ensure you have SSH access
ssh nilante@168.119.145.135 "echo 'Connected'"

# Get latest code
cd ~/main-nilante-server/ai-trading-system
git pull origin main
```

### 1.2 Server Access
```bash
ssh nilante@168.119.145.135
```

---

## 🔧 PHASE 2: SERVER SETUP (10 min)

### 2.1 Directory Structure
```bash
# Create deployment directories
sudo mkdir -p /srv/trading
sudo mkdir -p /mnt/fastdrive/trading/{questdb,prometheus,grafana,pulsar,weaviate}
sudo mkdir -p /mnt/bulkdata/trading/{minio,backups}
sudo chown -R nilante:nilante /srv/trading /mnt/fastdrive/trading /mnt/bulkdata/trading

# Verify
df -h | grep -E "(srv|fastdrive|bulkdata)"
```

### 2.2 System Check
```bash
# Quick system verification
echo "Python: $(python3 --version)"
echo "Docker: $(docker --version)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk /srv: $(df -h /srv | tail -1 | awk '{print $4}')"
```

---

## 🤖 PHASE 3: AI MODELS (15 min - can run parallel)

### 3.1 Install Ollama
```bash
# Install if needed
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    sudo systemctl enable ollama
    sudo systemctl start ollama
fi

ollama version
```

### 3.2 Download Models (PARALLEL - Open 4 SSH sessions)
```bash
# Terminal 1
ollama pull qwen2.5:72b      # 45GB

# Terminal 2  
ollama pull deepseek-r1:70b  # 42GB

# Terminal 3
ollama pull llama3.1:70b      # 40GB

# Terminal 4
ollama pull mixtral:8x7b      # 26GB
ollama pull phi3:medium       # 7GB

# Verify all models
ollama list
```

---

## 📦 PHASE 4: CODE DEPLOYMENT (5 min)

### 4.1 Clone Repository
```bash
cd /srv
sudo rm -rf trading  # Clean slate
sudo git clone https://github.com/nilantedev/ai-trading-system.git trading
sudo chown -R nilante:nilante trading
cd trading
```

### 4.2 Environment Configuration
```bash
# Copy template
cp .env.example .env.production

# Generate ALL passwords at once
cat > /tmp/passwords.txt << 'EOF'
DB_PASSWORD=$(openssl rand -base64 32)
DB_ROOT_PASSWORD=$(openssl rand -base64 32)  
REDIS_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 64 | tr -d '\n')
SECRET_KEY=$(openssl rand -base64 64)
GRAFANA_PASSWORD=$(openssl rand -base64 24)
MINIO_ROOT_PASSWORD=$(openssl rand -base64 32)
BACKUP_ENCRYPTION_KEY=$(openssl rand -base64 32)
EOF

# Execute to generate
bash /tmp/passwords.txt

# Edit with generated values
nano .env.production
```

### 4.3 Required .env.production Settings
```
ENVIRONMENT=production
DEBUG=false

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=trading_db
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=[GENERATED]

# Redis
REDIS_HOST=redis
REDIS_PASSWORD=[GENERATED]

# Security
JWT_SECRET=[GENERATED]
SECRET_KEY=[GENERATED]

# Monitoring
GRAFANA_PASSWORD=[GENERATED]

# Domain
DOMAIN_NAME=trading.main-nilante.com
LETSENCRYPT_EMAIL=admin@main-nilante.com

# AI (Local)
OLLAMA_HOST=http://host.docker.internal:11434
USE_LOCAL_MODELS_ONLY=true
```

---

## 🔒 PHASE 5: SSL & NETWORKING (5 min)

### 5.1 Quick SSL Setup
```bash
# Self-signed for immediate deployment
sudo ./scripts/setup_ssl_certificates.sh --self-signed --domain trading.main-nilante.com

# Firewall
sudo ufw allow 22,80,443,8000/tcp
sudo ufw --force enable
```

### 5.2 Domain Configuration (if DNS ready)
```bash
./scripts/configure_domain.sh --domain main-nilante.com --subdomain trading --check-dns
```

---

## 🚀 PHASE 6: LAUNCH (10 min)

### 6.1 Deploy
```bash
cd /srv/trading

# Option A: Automated deployment
./deploy_production.sh --skip-tests

# Option B: Manual deployment
docker-compose down -v  # Clean start
docker-compose build
docker-compose up -d postgres redis
sleep 30
docker-compose up -d
```

### 6.2 Quick Verification
```bash
# Check services
docker-compose ps

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/metrics | head -5

# Check logs
docker-compose logs --tail=50 api
```

---

## ✅ PHASE 7: VALIDATION (5 min)

### 7.1 Health Checks
```bash
# Service status
echo "=== Service Health ==="
for service in api postgres redis; do
    STATUS=$(docker-compose ps $service | grep Up && echo "✓" || echo "✗")
    echo "$service: $STATUS"
done

# API endpoints
echo "=== API Health ==="
curl -s http://localhost:8000/health | jq -r '.status'
curl -s http://localhost:8000/ready | jq -r '.ready'
```

### 7.2 Database Check
```bash
# PostgreSQL
docker-compose exec postgres pg_isready

# Redis
docker-compose exec redis redis-cli ping
```

### 7.3 AI Model Test
```bash
# Test Ollama
curl -X POST http://localhost:8000/api/v1/ai/test \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test"}'
```

---

## 🎯 PHASE 8: GO LIVE

### 8.1 Production Configuration
```bash
# Enable production mode
docker-compose -f docker-compose.yml \
  -f infrastructure/docker/docker-compose.production.yml up -d

# Setup backups
sudo ./scripts/setup_backup_cron.sh

# Configure monitoring
./scripts/setup_monitoring_alerts.sh --email admin@main-nilante.com
```

### 8.2 Access Points
- **API**: https://trading.main-nilante.com
- **Docs**: https://trading.main-nilante.com/docs
- **Grafana**: https://trading.main-nilante.com/grafana
- **Health**: https://trading.main-nilante.com/health

---

## 🔥 QUICK FIXES

### Container Issues
```bash
docker-compose restart [service]
docker-compose logs --tail=100 [service]
```

### Database Reset
```bash
docker-compose down -v postgres
docker-compose up -d postgres
```

### Memory Issues
```bash
docker system prune -af
docker-compose down && docker-compose up -d
```

### Ollama Issues
```bash
sudo systemctl restart ollama
ollama list
```

---

## ✅ SUCCESS CRITERIA
- [ ] All containers running: `docker-compose ps`
- [ ] Health endpoint returns 200: `curl http://localhost:8000/health`
- [ ] No errors in last 100 log lines: `docker-compose logs --tail=100`
- [ ] Database connected: `docker-compose exec postgres pg_isready`
- [ ] Redis responding: `docker-compose exec redis redis-cli ping`
- [ ] AI models loaded: `ollama list` (shows 5 models)

---

## 📱 MONITORING
```bash
# Real-time logs
docker-compose logs -f api

# System metrics
docker stats

# Service health
watch -n 5 'docker-compose ps'
```

---

## 🚨 EMERGENCY ROLLBACK
```bash
# Complete rollback
cd /srv/trading
docker-compose down -v
git checkout HEAD~1
docker-compose up -d
```

---

**DEPLOYMENT TIME: 45 minutes** ⏱️

Start with **PHASE 1** now! The system will be live in less than an hour.
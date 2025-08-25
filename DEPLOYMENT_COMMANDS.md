# AI Trading System - Production Deployment Guide

**Server**: main-nilante.com (168.119.145.135)  
**OS**: Ubuntu 24.04 LTS  
**Hardware**: AMD EPYC 64-core, 1TB RAM  
**Status**: Phase 7 Complete - Ready for Production Deployment

---

## Pre-Deployment: GitHub Repository Setup

### Step 1: Create GitHub Repository
```bash
# Go to: https://github.com/new
# Repository name: ai-trading-system
# Owner: nilantedev
# Description: AI-powered algorithmic trading system with real-time market data processing, signal generation, and automated portfolio management
# Make it Public
# Do NOT initialize with README (we already have one)
```

### Step 2: Push Code to Repository
```bash
# From local development machine
cd /home/nilante/main-nilante-server/ai-trading-system
git push -u origin main
```

---

## Phase 8: Production Deployment

### Step 1: Connect to Production Server
```bash
ssh -i ~/.ssh/hetzner_admin -p 2222 nilante@168.119.145.135
```

### Step 2: Clone Repository on Server
```bash
cd /home/nilante/
git clone https://github.com/nilantedev/ai-trading-system.git
cd ai-trading-system
```

### Step 3: Environment Setup
```bash
# Install required packages (if not already installed)
sudo apt update
sudo apt install -y docker.io docker-compose-v2 python3-pip python3-venv git

# Start Docker
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Storage Directory Creation

**Run these commands on the production server as user with sudo access:**

```bash
# Create required storage directories
sudo mkdir -p /srv/trading/redis
sudo mkdir -p /srv/trading/config/{traefik,redis,prometheus,grafana,loki,pulsar,promtail}
sudo mkdir -p /srv/trading/config/letsencrypt
sudo mkdir -p /srv/trading/logs
sudo mkdir -p /mnt/fastdrive/trading/{questdb,prometheus,grafana,pulsar,weaviate}
sudo mkdir -p /mnt/bulkdata/trading/{minio,backups}

# Set proper ownership (replace $USER with actual username if needed)
sudo chown -R $USER:$USER /srv/trading
sudo chown -R $USER:$USER /mnt/fastdrive/trading
sudo chown -R $USER:$USER /mnt/bulkdata/trading

# Set proper permissions for Let's Encrypt
sudo chmod 600 /srv/trading/config/letsencrypt/acme.json

# Verify directories exist
ls -la /srv/trading/
ls -la /mnt/fastdrive/trading/
ls -la /mnt/bulkdata/trading/
```

### Step 5: Environment Configuration
```bash
# Copy production environment file
cp .env.production .env

# Update .env with actual API keys and secrets
nano .env
# Set the following variables:
# - POLYGON_API_KEY=your_polygon_api_key
# - NEWS_API_KEY=your_news_api_key
# - OPENAI_API_KEY=your_openai_api_key
# - ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Step 6: Deploy Infrastructure Services
```bash
# Deploy all infrastructure services
docker compose -f infrastructure/docker/docker-compose.production.yml up -d

# Check service status
docker compose -f infrastructure/docker/docker-compose.production.yml ps

# View logs if needed
docker compose -f infrastructure/docker/docker-compose.production.yml logs
```

### Step 7: Health Check and Verification
```bash
# Check all services are running
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test service endpoints
curl -f https://trading.main-nilante.com/prometheus/api/v1/targets  # Prometheus
curl -f https://trading.main-nilante.com/questdb/status           # QuestDB  
curl -f https://trading.main-nilante.com/grafana/api/health       # Grafana
echo "PING" | redis-cli -h 127.0.0.1 -p 6379                     # Redis
curl -f http://127.0.0.1:8080/v1/.well-known/ready               # Weaviate
curl -f http://127.0.0.1:9001/minio/health/live                  # MinIO

# Run basic system tests
source .venv/bin/activate
python -m pytest tests/unit/test_basic_functionality.py -v
```

---

## Production Service URLs

### Public Access (via Traefik)
- **Main Trading Interface**: https://trading.main-nilante.com/
- **Grafana Dashboard**: https://trading.main-nilante.com/grafana (admin/TradingSystem2024!)
- **Prometheus**: https://trading.main-nilante.com/prometheus
- **QuestDB Console**: https://trading.main-nilante.com/questdb
- **Traefik Dashboard**: https://trading.main-nilante.com/traefik

### Internal Services (localhost only)
- **Redis**: localhost:6379
- **Weaviate API**: localhost:8080
- **MinIO Console**: localhost:9001 (admin/TradingSystem2024!)

---

## Required API Keys and Credentials

**Before deployment, ensure you have:**
1. **Polygon API Key** - For real-time market data
2. **News API Key** - For financial news integration  
3. **OpenAI API Key** - For AI agent orchestration
4. **Anthropic API Key** - For Claude AI integration

**Default Passwords:**
- **Grafana**: admin / TradingSystem2024!
- **MinIO**: admin / TradingSystem2024!

---

## Troubleshooting

### Common Issues

#### 1. Docker Permission Issues
```bash
# If "permission denied" errors occur
sudo usermod -aG docker $USER
# Log out and back in, or run:
newgrp docker
```

#### 2. Storage Directory Permission Issues
```bash
# Fix ownership issues
sudo chown -R $USER:$USER /srv/trading
sudo chown -R $USER:$USER /mnt/fastdrive/trading  
sudo chown -R $USER:$USER /mnt/bulkdata/trading
```

#### 3. Service Not Starting
```bash
# Check service logs
docker compose -f infrastructure/docker/docker-compose.production.yml logs [service_name]

# Restart specific service
docker compose -f infrastructure/docker/docker-compose.production.yml restart [service_name]
```

#### 4. SSL Certificate Issues
```bash
# Check Traefik logs for Let's Encrypt issues
docker logs trading-traefik
```

---

## Deployment Validation Checklist

- [ ] GitHub repository created and code pushed
- [ ] Production server accessible via SSH
- [ ] Docker and Docker Compose installed
- [ ] Storage directories created with correct permissions
- [ ] Environment variables configured (.env file)
- [ ] All infrastructure services running
- [ ] Health checks passing
- [ ] Web interfaces accessible
- [ ] Basic system tests passing

**Deployment Status**: Ready for Phase 8 Production Deployment
docker-compose -f infrastructure/docker/docker-compose.production.yml up -d

# Check service status
docker-compose -f infrastructure/docker/docker-compose.production.yml ps

# View logs if needed
docker-compose -f infrastructure/docker/docker-compose.production.yml logs

# Health check all services (production endpoints)
curl -f https://trading.main-nilante.com/prometheus/api/v1/targets  # Prometheus
curl -f https://trading.main-nilante.com/questdb/status           # QuestDB  
curl -f https://trading.main-nilante.com/grafana/api/health       # Grafana
curl -f http://127.0.0.1:6379 && echo "PING" | redis-cli         # Redis
curl -f http://127.0.0.1:8080/v1/.well-known/ready              # Weaviate
curl -f http://127.0.0.1:9001/minio/health/live                 # MinIO
```

### Expected Services After Phase 1

- **Traefik**: Reverse proxy and load balancer (ports 80, 443, 8081, 8082)
- **Redis**: Cache and session store (port 6379, localhost only)
- **QuestDB**: Time-series database (ports 9000, 8812, 9009, localhost only)
- **Prometheus**: Metrics collection (port 9090, localhost only)
- **Grafana**: Monitoring dashboards (port 3001, localhost only)
- **Pulsar**: Message streaming broker (ports 6650, 8083, localhost only)
- **Weaviate**: Vector database for AI (ports 8080, 50051, localhost only)
- **MinIO**: Object storage (ports 9001, 9002, localhost only)
- **Loki**: Log aggregation (port 3100, localhost only)
- **Node Exporter**: System metrics (port 9100, localhost only)
- **cAdvisor**: Container metrics (port 8084, localhost only)
- **Promtail**: Log collection agent

### Access URLs (Production)

- **Main Trading Interface**: https://trading.main-nilante.com/
- **Grafana Dashboard**: https://trading.main-nilante.com/grafana (admin/TradingSystem2024!)
- **Prometheus**: https://trading.main-nilante.com/prometheus
- **QuestDB Console**: https://trading.main-nilante.com/questdb
- **Traefik Dashboard**: https://trading.main-nilante.com/traefik
- **Weaviate API**: https://trading.main-nilante.com/weaviate
- **MinIO Console**: https://trading.main-nilante.com/minio (admin/TradingSystem2024!)
- **Pulsar Admin**: https://trading.main-nilante.com/pulsar

### Verification Steps

```bash
# Test service connectivity
curl -f http://localhost:9090/api/v1/targets  # Prometheus
curl -f http://localhost:9000/status          # QuestDB
curl -f http://localhost:3001/api/health      # Grafana
redis-cli ping                                # Redis

# Check Docker resource usage
docker stats

# Verify all containers are healthy
docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml ps
```

### Next Steps

After successful Phase 1 deployment:
1. Verify all services are healthy
2. Access Grafana and configure dashboards
3. Proceed to Phase 2: Core Data Layer

---

**⚠️ IMPORTANT**: Phase 1 must be fully operational before proceeding to subsequent phases.
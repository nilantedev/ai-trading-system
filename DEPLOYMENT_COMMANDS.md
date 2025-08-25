# AI Trading System - Deployment Commands

## Phase 1: Infrastructure Deployment

### Pre-Deployment: Storage Directory Creation

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

### Phase 1 Deployment Commands

```bash
# Navigate to project directory
cd /home/nilante/main-nilante-server/ai-trading-system

# For production deployment, use production compose file
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
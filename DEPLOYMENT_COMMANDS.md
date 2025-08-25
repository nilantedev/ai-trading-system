# AI Trading System - Deployment Commands

## Phase 1: Infrastructure Deployment

### Pre-Deployment: Storage Directory Creation

**Run these commands on the production server as user with sudo access:**

```bash
# Create required storage directories
sudo mkdir -p /srv/trading/redis
sudo mkdir -p /srv/trading/questdb
sudo mkdir -p /mnt/fastdrive/trading/prometheus
sudo mkdir -p /mnt/fastdrive/trading/grafana
sudo mkdir -p /mnt/fastdrive/trading/pulsar

# Set proper ownership (replace $USER with actual username if needed)
sudo chown -R $USER:$USER /srv/trading
sudo chown -R $USER:$USER /mnt/fastdrive/trading

# Verify directories exist
ls -la /srv/trading/
ls -la /mnt/fastdrive/trading/
```

### Phase 1 Deployment Commands

```bash
# Navigate to project directory
cd /home/nilante/main-nilante-server/ai-trading-system

# Start infrastructure services
docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml up -d

# Check service status
docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml ps

# View logs if needed
docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml logs

# Health check all services
make health-check-infrastructure
```

### Expected Services After Phase 1

- **Traefik**: Reverse proxy at port 8081
- **Redis**: Cache at port 6379
- **QuestDB**: Time-series DB at port 9000
- **Prometheus**: Metrics at port 9090
- **Grafana**: Dashboards at port 3001
- **Pulsar**: Message broker at port 6650
- **Loki**: Log aggregation at port 3100
- **Node Exporter**: System metrics at port 9100
- **cAdvisor**: Container metrics at port 8082

### Access URLs

- **Grafana Dashboard**: http://localhost:3001 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **QuestDB Console**: http://localhost:9000
- **Traefik Dashboard**: http://localhost:8081

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
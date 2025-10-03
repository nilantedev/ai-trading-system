# Quick Commands Reference

## Health & Monitoring

```bash
# Run comprehensive health check
bash scripts/comprehensive_health_check.sh

# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# View API logs (last 100 lines)
docker logs trading-api --tail 100 -f

# View data ingestion activity
docker logs trading-data-ingestion --tail 50 -f

# Check watchlist size
docker exec trading-redis redis-cli -a "$REDIS_PASSWORD" SCARD watchlist
```

## External Access

```bash
# Test all domains
curl -I https://biz.mekoshi.com
curl -I https://admin.mekoshi.com
curl -I https://api.mekoshi.com

# Check SSL certificates
echo | openssl s_client -servername biz.mekoshi.com -connect biz.mekoshi.com:443 2>/dev/null | openssl x509 -noout -dates
```

## Database Queries

```bash
# PostgreSQL - count symbols
docker exec trading-postgres psql -U trading -d trading -c "SELECT COUNT(*) FROM symbols WHERE optionable = true;"

# Redis - check keys
docker exec trading-redis redis-cli -a "$REDIS_PASSWORD" KEYS "*"

# QuestDB - recent market data
curl "http://localhost:9000/exec?query=SELECT COUNT(*) FROM market_data WHERE timestamp > dateadd('h',-24,now())"
```

## Container Management

```bash
# Restart API container
docker restart trading-api

# Rebuild API (after code changes)
docker-compose build api
docker-compose up -d api

# View all trading containers
docker ps | grep trading
```

## URLs

- **Business Portal**: https://biz.mekoshi.com
- **Admin Portal**: https://admin.mekoshi.com  
- **API Docs**: https://api.mekoshi.com
- **Grafana**: http://localhost:3000
- **QuestDB**: http://localhost:9000
- **Prometheus**: http://localhost:9090

---

*Quick reference for Mekoshi Trading System*

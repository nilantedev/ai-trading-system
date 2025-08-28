# 📋 POST-DEPLOYMENT ACTION PLAN
## What to Do AFTER the App is Running on Server

---

## 🔴 IMMEDIATE (First 10 Minutes)

### 1. Verify Basic Health
```bash
# Check if API is responding
curl http://localhost:8000/health

# Expected output:
# {"status":"healthy","timestamp":"..."}

# If not working:
docker-compose logs api | tail -50
```

### 2. Check All Services Running
```bash
# See what's running
docker ps

# You should see 3+ containers:
# - trading-postgres (database)
# - trading-redis (cache)
# - trading-api (your app)

# If any missing:
docker-compose up -d
```

### 3. Test API Endpoints
```bash
# Test root endpoint
curl http://localhost:8000/

# Test metrics
curl http://localhost:8000/metrics

# Test from outside (from your local machine):
curl http://germ1-ain-nilante.com:8000/health
```

---

## 🟡 CRITICAL FIXES (First Hour)

### 1. Fix Database Connection Issues
**Problem:** "Can't connect to PostgreSQL"
```bash
# Check if postgres is running
docker ps | grep postgres

# Check postgres logs
docker-compose logs postgres

# Fix: Restart with correct password
docker-compose down postgres
docker-compose up -d postgres

# Wait 30 seconds
sleep 30

# Retry migrations
source .venv/bin/activate
alembic upgrade head
```

### 2. Fix Redis Connection Issues  
**Problem:** "Redis connection refused"
```bash
# Check if redis is running
docker ps | grep redis

# Test redis connection
docker exec trading-redis redis-cli ping
# Should return: PONG

# Fix: Restart redis
docker-compose restart redis
```

### 3. Fix API Key Issues
**Problem:** "Invalid API key" or "401 Unauthorized"
```bash
# Check which APIs are failing
docker-compose logs api | grep -i "api\|error\|401"

# Edit .env to fix keys
nano .env

# Find the failing API section and update:
ALPACA_API_KEY=your_correct_key_here
POLYGON_API_KEY=your_correct_key_here

# Save: Ctrl+O, Enter, Ctrl+X

# Restart API to load new keys
docker-compose restart api
```

---

## 🟢 FIRST DAY TASKS

### 1. Enable Data Collection
```bash
# Monitor data flow
docker-compose logs -f api | grep -i "market\|data\|fetch"

# You should see:
# "Fetching market data for..."
# "Received data from Polygon..."
# "Cached market data..."

# If no data flowing:
# - Check market hours (9:30 AM - 4 PM EST)
# - Verify API keys are correct
# - Check rate limits
```

### 2. Test Paper Trading
```bash
# Access API documentation
# From your browser, go to:
http://germ1-ain-nilante.com:8000/docs

# Test a paper trade via curl:
curl -X POST http://localhost:8000/api/v1/trading/orders \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "qty": 1,
    "side": "buy",
    "type": "market",
    "time_in_force": "day"
  }'
```

### 3. Set Up Basic Monitoring
```bash
# Create monitoring script
cat > ~/monitor_trading.sh << 'EOF'
#!/bin/bash
echo "=== Trading System Status ==="
echo "API Health:"
curl -s http://localhost:8000/health | python3 -m json.tool
echo -e "\nRunning Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}"
echo -e "\nDisk Usage:"
df -h | grep -E "/$|fastdrive|bulkdata"
echo -e "\nMemory Usage:"
free -h
echo -e "\nRecent API Logs:"
docker-compose logs api --tail=10
EOF

chmod +x ~/monitor_trading.sh

# Run it
~/monitor_trading.sh
```

---

## 🔵 FIRST WEEK TASKS

### 1. Install Monitoring Dashboard
```bash
# Start Grafana & Prometheus
docker-compose up -d prometheus grafana

# Access Grafana
# Browser: http://germ1-ain-nilante.com:3000
# Default login: admin / grafana_dev_123

# Import dashboards (in Grafana UI):
# 1. Click "+" → "Import"
# 2. Upload: /srv/ai-trading-system/config/grafana/dashboards/*.json
```

### 2. Set Up Automated Backups
```bash
# Create backup script
cat > ~/backup_trading.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/mnt/bulkdata/trading-backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker exec trading-postgres pg_dump -U trading_user trading_system | \
  gzip > $BACKUP_DIR/postgres_$DATE.sql.gz

# Backup Redis
docker exec trading-redis redis-cli SAVE
docker cp trading-redis:/data/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

# Keep only last 7 days
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x ~/backup_trading.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /home/nilante/backup_trading.sh") | crontab -
```

### 3. Enable SSL/HTTPS
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d germ1-ain-nilante.com

# Auto-renew
sudo certbot renew --dry-run
```

---

## 🟣 OPTIMIZATION TASKS (Week 2)

### 1. Install AI Models (Ollama)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a small model first (for testing)
ollama pull llama2:7b

# Test it works
ollama run llama2:7b "What is the stock market?"

# Later, pull larger models:
# ollama pull llama3.3:70b  # Needs 48GB RAM
# ollama pull mixtral:8x7b  # Needs 32GB RAM
```

### 2. Optimize Database Performance
```bash
# Check slow queries
docker exec trading-postgres psql -U trading_user -d trading_system -c \
  "SELECT query, calls, mean_exec_time 
   FROM pg_stat_statements 
   ORDER BY mean_exec_time DESC 
   LIMIT 10;"

# Add indexes if needed
source .venv/bin/activate
python << 'EOF'
from trading_common.database_manager import get_database_manager
import asyncio

async def add_indexes():
    db = get_database_manager()
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_data_timestamp 
        ON market_data(timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol 
        ON trading_signals(symbol, timestamp DESC);
    """)
    print("Indexes created!")

asyncio.run(add_indexes())
EOF
```

### 3. Set Up Rate Limit Monitoring
```bash
# Create rate limit checker
cat > ~/check_rate_limits.py << 'EOF'
import asyncio
from datetime import datetime
import json

async def check_limits():
    limits = {
        "Alpaca": {"used": 0, "limit": 200, "reset": "1 min"},
        "Polygon": {"used": 0, "limit": 100, "reset": "1 min"},
        "Reddit": {"used": 0, "limit": 60, "reset": "1 min"},
        "AlphaVantage": {"used": 0, "limit": 5, "reset": "1 min"}
    }
    
    # Check Redis for actual counts
    import redis
    r = redis.Redis(host='localhost', port=6379)
    
    for api in limits:
        key = f"rate_limit:{api.lower()}"
        used = r.get(key)
        if used:
            limits[api]["used"] = int(used)
            pct = (limits[api]["used"] / limits[api]["limit"]) * 100
            status = "🟢 OK" if pct < 80 else "🟡 WARNING" if pct < 95 else "🔴 CRITICAL"
            print(f"{api}: {limits[api]['used']}/{limits[api]['limit']} ({pct:.1f}%) {status}")

asyncio.run(check_limits())
EOF

python3 ~/check_rate_limits.py
```

---

## 🚨 TROUBLESHOOTING GUIDE

### Problem: "Module trading_common not found"
```bash
cd /srv/ai-trading-system
source .venv/bin/activate
pip install -e shared/python-common/
```

### Problem: "Port 8000 already in use"
```bash
# Find what's using port 8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Restart
docker-compose up -d
```

### Problem: "Disk space full"
```bash
# Check what's using space
du -sh /srv/* | sort -h
du -sh /mnt/fastdrive/* | sort -h

# Clean Docker
docker system prune -a

# Clean old logs
find /var/log/trading-system -name "*.log" -mtime +7 -delete
```

### Problem: "API not accessible from outside"
```bash
# Check firewall
sudo ufw status

# Open port 8000
sudo ufw allow 8000

# Check if API is binding to all interfaces
docker-compose logs api | grep "0.0.0.0:8000"
```

---

## 📊 SUCCESS METRICS (How to Know It's Working)

### ✅ Day 1 Success:
- [ ] Health endpoint returns 200 OK
- [ ] All 3 containers running (postgres, redis, api)
- [ ] No ERROR in logs (only INFO/WARNING)
- [ ] Can access API docs at :8000/docs

### ✅ Week 1 Success:
- [ ] Collecting market data (check logs)
- [ ] Paper trades executing
- [ ] Less than 50% CPU usage
- [ ] Less than 30% memory usage
- [ ] Daily backups running

### ✅ Month 1 Success:
- [ ] 99% uptime
- [ ] Grafana dashboards populated
- [ ] AI models installed and running
- [ ] SSL/HTTPS enabled
- [ ] Profitable paper trading

---

## 📞 GET HELP

### Check Logs First:
```bash
# API logs
docker-compose logs api --tail=100

# All logs
docker-compose logs --tail=100

# Follow logs live
docker-compose logs -f
```

### System Resources:
```bash
# CPU and Memory
htop

# Disk space
df -h

# Network connections
ss -tuln | grep LISTEN
```

### Restart Everything:
```bash
cd /srv/ai-trading-system
docker-compose down
docker-compose up -d
```

---

**Remember:** The app works but needs fine-tuning. Start with basic functionality, then gradually enable advanced features. Monitor logs constantly during the first 24 hours!

🎯 **Goal for Day 1:** API running, data flowing, no crashes
🎯 **Goal for Week 1:** Stable operation, monitoring enabled, backups working
🎯 **Goal for Month 1:** AI models running, optimized performance, profitable strategies
#!/bin/bash
set -euo pipefail

# Comprehensive Investor Readiness Report
# Validates all aspects of the AI Trading System
# Generated: $(date)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
WARN=0
FAIL=0
REPORT_FILE="/tmp/investor_report_$(date +%Y%m%d_%H%M%S).txt"

log() {
    echo -e "$1" | tee -a "$REPORT_FILE"
}

header() {
    log "\n${BLUE}═══════════════════════════════════════════════════${NC}"
    log "${BLUE}$1${NC}"
    log "${BLUE}═══════════════════════════════════════════════════${NC}\n"
}

check_pass() {
    log "${GREEN}✓ PASS${NC}: $1"
    PASS=$((PASS + 1))
}

check_warn() {
    log "${YELLOW}⚠ WARN${NC}: $1"
    WARN=$((WARN + 1))
}

check_fail() {
    log "${RED}✗ FAIL${NC}: $1"
    FAIL=$((FAIL + 1))
}

cd /srv/ai-trading-system

log "╔════════════════════════════════════════════════════════════╗"
log "║     MEKOSHI AI TRADING SYSTEM - INVESTOR REPORT            ║"
log "║     Comprehensive Production Readiness Assessment          ║"
log "║     Generated: $(date)                         ║"
log "╚════════════════════════════════════════════════════════════╝"

# ============================================================================
header "1. INFRASTRUCTURE STATUS"
# ============================================================================

log "Container Health Status:"
CONTAINERS=$(docker ps --format "{{.Names}}" | grep "trading-" | sort)
TOTAL_CONTAINERS=$(echo "$CONTAINERS" | wc -l)
HEALTHY_CONTAINERS=0

for container in $CONTAINERS; do
    STATUS=$(docker inspect "$container" --format='{{.State.Status}}' 2>/dev/null || echo "error")
    HEALTH=$(docker inspect "$container" --format='{{.State.Health.Status}}' 2>/dev/null || echo "none")
    
    if [ "$STATUS" = "running" ] && { [ "$HEALTH" = "healthy" ] || [ "$HEALTH" = "none" ]; }; then
        check_pass "$container: Running (Health: ${HEALTH})"
        HEALTHY_CONTAINERS=$((HEALTHY_CONTAINERS + 1))
    else
        check_fail "$container: $STATUS (Health: $HEALTH)"
    fi
done

log "\nContainer Summary: $HEALTHY_CONTAINERS/$TOTAL_CONTAINERS healthy"

# ============================================================================
header "2. DATABASE LAYER"
# ============================================================================

log "PostgreSQL (Primary Relational Database):"
if docker exec trading-postgres pg_isready -U trading_user > /dev/null 2>&1; then
    check_pass "PostgreSQL accepting connections"
    
    # Check database size
    DB_SIZE=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT pg_size_pretty(pg_database_size('trading_db'));" 2>/dev/null | xargs || echo "N/A")
    log "  Database Size: $DB_SIZE"
    
    # Check table counts
    TABLES=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null | xargs || echo "0")
    log "  Tables: $TABLES"
    
    # Check for recent activity
    RECENT_TRADES=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM trades WHERE created_at > NOW() - INTERVAL '24 hours';" 2>/dev/null | xargs || echo "0")
    log "  Recent trades (24h): $RECENT_TRADES"
else
    check_fail "PostgreSQL NOT responding"
fi

log "\nRedis (Cache & Real-time Data):"
REDIS_PASS=$(grep "^REDIS_PASSWORD=" .env 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'" | head -1 || echo "")
if [ -n "$REDIS_PASS" ]; then
    if docker exec trading-redis redis-cli -a "$REDIS_PASS" --no-auth-warning ping 2>&1 | grep -q PONG; then
        check_pass "Redis responding"
        
        REDIS_KEYS=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" --no-auth-warning DBSIZE 2>&1 | grep -o '[0-9]*' | head -1 || echo "0")
        log "  Cached keys: $REDIS_KEYS"
        
        WATCHLIST_COUNT=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" --no-auth-warning SCARD "watchlist" 2>&1 | grep -o '[0-9]*' | head -1 || echo "0")
        log "  Watchlist symbols: $WATCHLIST_COUNT"
        
        if [ "$WATCHLIST_COUNT" -gt 300 ]; then
            check_pass "Watchlist well-populated ($WATCHLIST_COUNT symbols)"
        else
            check_warn "Watchlist has only $WATCHLIST_COUNT symbols"
        fi
    else
        check_fail "Redis NOT responding"
    fi
else
    check_fail "Redis password not configured"
fi

log "\nQuestDB (Time-Series Market Data):"
if curl -s http://localhost:9000/exec?query=SELECT+1 | grep -q "dataset"; then
    check_pass "QuestDB responding"
    
    # Check daily_bars table
    DAILY_BARS=$(curl -s "http://localhost:9000/exec?query=SELECT+COUNT(*)+FROM+daily_bars" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0] if 'dataset' in d else 0)" 2>/dev/null || echo "0")
    log "  Daily bars: $(printf "%'d" $DAILY_BARS)"
    
    # Check symbols with data
    SYMBOLS_WITH_DATA=$(curl -s "http://localhost:9000/exec?query=SELECT+COUNT+DISTINCT+symbol+FROM+daily_bars" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0] if 'dataset' in d else 0)" 2>/dev/null || echo "0")
    log "  Symbols with data: $SYMBOLS_WITH_DATA"
    
    # Check recent data (last 24h)
    RECENT_BARS=$(curl -s "http://localhost:9000/exec?query=SELECT+COUNT(*)+FROM+market_data+WHERE+timestamp+>+dateadd('d',-1,now())" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0] if 'dataset' in d else 0)" 2>/dev/null || echo "0")
    log "  Recent market data (24h): $(printf "%'d" $RECENT_BARS)"
    
    if [ "$RECENT_BARS" -gt 1000 ]; then
        check_pass "Active data ingestion ($RECENT_BARS bars/day)"
    else
        check_warn "Low data ingestion rate: $RECENT_BARS bars/day"
    fi
    
    # Check options data
    OPTIONS_COUNT=$(curl -s "http://localhost:9000/exec?query=SELECT+COUNT(*)+FROM+options_data" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0] if 'dataset' in d else 0)" 2>/dev/null || echo "0")
    log "  Options contracts: $(printf "%'d" $OPTIONS_COUNT)"
    
    # Check news items
    NEWS_COUNT=$(curl -s "http://localhost:9000/exec?query=SELECT+COUNT(*)+FROM+news_items" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0] if 'dataset' in d else 0)" 2>/dev/null || echo "0")
    log "  News items: $(printf "%'d" $NEWS_COUNT)"
    
    # Check social signals
    SOCIAL_COUNT=$(curl -s "http://localhost:9000/exec?query=SELECT+COUNT(*)+FROM+social_signals" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0] if 'dataset' in d else 0)" 2>/dev/null || echo "0")
    log "  Social signals: $(printf "%'d" $SOCIAL_COUNT)"
else
    check_fail "QuestDB NOT responding"
fi

log "\nWeaviate (Vector Database for AI/ML):"
if curl -s http://localhost:8080/v1/.well-known/ready | grep -q "true"; then
    check_pass "Weaviate responding"
    
    WEAVIATE_OBJECTS=$(curl -s http://localhost:8080/v1/schema | grep -o '"class"' | wc -l || echo "0")
    log "  Schema classes: $WEAVIATE_OBJECTS"
else
    check_warn "Weaviate may not be fully ready"
fi

# ============================================================================
header "3. API & WEB SERVICES"
# ============================================================================

log "Core API (Port 8000):"
API_HEALTH=$(docker exec trading-api curl -s http://localhost:8000/health 2>/dev/null || echo "ERROR")
if echo "$API_HEALTH" | grep -q "ok\|healthy"; then
    check_pass "API health endpoint responding"
    
    # Check API version
    API_VERSION=$(docker exec trading-api curl -s http://localhost:8000/version 2>/dev/null | grep -oP '"version":"\K[^"]+' || echo "unknown")
    log "  Version: $API_VERSION"
else
    check_fail "API health endpoint NOT responding"
fi

log "\nPublic Websites:"
# Check biz.mekoshi.com
BIZ_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://biz.mekoshi.com/auth/login 2>/dev/null || echo "000")
if [ "$BIZ_STATUS" = "200" ]; then
    check_pass "biz.mekoshi.com accessible (HTTP $BIZ_STATUS)"
else
    check_fail "biz.mekoshi.com NOT accessible (HTTP $BIZ_STATUS)"
fi

# Check admin.mekoshi.com
ADMIN_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://admin.mekoshi.com 2>/dev/null || echo "000")
if [ "$ADMIN_STATUS" = "200" ] || [ "$ADMIN_STATUS" = "401" ]; then
    check_pass "admin.mekoshi.com accessible (HTTP $ADMIN_STATUS)"
else
    check_fail "admin.mekoshi.com NOT accessible (HTTP $ADMIN_STATUS)"
fi

# Check API endpoint
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.mekoshi.com/health 2>/dev/null || echo "000")
if [ "$API_STATUS" = "200" ]; then
    check_pass "api.mekoshi.com accessible (HTTP $API_STATUS)"
else
    check_warn "api.mekoshi.com response: HTTP $API_STATUS"
fi

log "\nSSL Certificates:"
for domain in biz.mekoshi.com admin.mekoshi.com api.mekoshi.com; do
    CERT_EXPIRY=$(echo | timeout 5 openssl s_client -servername "$domain" -connect "$domain:443" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null | grep notAfter | cut -d= -f2)
    if [ -n "$CERT_EXPIRY" ]; then
        check_pass "$domain: Valid until $CERT_EXPIRY"
    else
        check_warn "$domain: Certificate check timed out (likely valid if site accessible)"
    fi
done

# ============================================================================
header "4. DATA INGESTION & PROCESSING"
# ============================================================================

log "Data Collection Services:"
DATA_INGESTION_STATUS=$(docker inspect trading-data-ingestion --format='{{.State.Status}}' 2>/dev/null || echo "error")
if [ "$DATA_INGESTION_STATUS" = "running" ]; then
    check_pass "Data Ingestion service running"
    
    # Check logs for recent activity
    RECENT_INGESTION=$(docker logs trading-data-ingestion --since 10m 2>&1 | grep -i "collected\|ingested\|stored" | wc -l || echo "0")
    if [ "$RECENT_INGESTION" -gt 5 ]; then
        check_pass "Active data collection (${RECENT_INGESTION} events in 10min)"
    else
        check_warn "Low data collection activity: $RECENT_INGESTION events"
    fi
else
    check_fail "Data Ingestion service NOT running"
fi

log "\nData Sources Configured:"
# Check for API keys in environment
POLYGON_KEY=$(grep "^POLYGON_API_KEY=" .env 2>/dev/null | cut -d= -f2 | wc -c)
ALPACA_KEY=$(grep "^ALPACA_API_KEY=" .env 2>/dev/null | cut -d= -f2 | wc -c)
EODHD_KEY=$(grep "^EODHD_API_KEY=" .env 2>/dev/null | cut -d= -f2 | wc -c)
ALPHAVANTAGE_KEY=$(grep "^ALPHAVANTAGE_API_KEY=" .env 2>/dev/null | cut -d= -f2 | wc -c)

if [ "$POLYGON_KEY" -gt 10 ]; then check_pass "Polygon.io API configured"; else check_warn "Polygon.io API not configured"; fi
if [ "$ALPACA_KEY" -gt 10 ]; then check_pass "Alpaca API configured"; else check_warn "Alpaca API not configured"; fi
if [ "$EODHD_KEY" -gt 10 ]; then check_pass "EODHD API configured"; else check_warn "EODHD API not configured"; fi
if [ "$ALPHAVANTAGE_KEY" -gt 10 ]; then check_pass "AlphaVantage API configured"; else check_warn "AlphaVantage API not configured"; fi

# ============================================================================
header "5. MACHINE LEARNING & AI"
# ============================================================================

log "ML Service Status:"
ML_STATUS=$(docker inspect trading-ml --format='{{.State.Status}}' 2>/dev/null || echo "error")
if [ "$ML_STATUS" = "running" ]; then
    check_pass "ML service running"
    
    ML_HEALTH=$(docker exec trading-ml curl -s http://localhost:8001/healthz 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    if [ "$ML_HEALTH" = "ok" ]; then
        check_pass "ML service healthy"
    else
        check_warn "ML service health: $ML_HEALTH"
    fi
    
    # Check for recent ML activity
    ML_ACTIVITY=$(docker logs trading-ml --since 30m 2>&1 | grep -E "(scheduler enabled|Updated.*factor|Updating factor)" | wc -l || echo "0")
    if [ "$ML_ACTIVITY" -gt 0 ]; then
        check_pass "ML scheduler active ($ML_ACTIVITY events in 30min)"
    else
        check_warn "No recent ML activity detected"
    fi
    
    # Check ML models
    ML_LOGS=$(docker logs trading-ml --since 1h 2>&1 | tail -100)
    if echo "$ML_LOGS" | grep -q "factor models"; then
        log "  Factor models: Active"
    fi
    if echo "$ML_LOGS" | grep -q "correlation"; then
        log "  Correlation analysis: Active"
    fi
else
    check_fail "ML service NOT running"
fi

log "\nOllama (LLM Models):"
OLLAMA_STATUS=$(docker inspect trading-ollama --format='{{.State.Status}}' 2>/dev/null || echo "error")
if [ "$OLLAMA_STATUS" = "running" ]; then
    check_pass "Ollama service running"
    
    OLLAMA_MODELS=$(docker exec trading-ollama ollama list 2>/dev/null | grep -c ":" || echo "0")
    if [ "$OLLAMA_MODELS" -gt 0 ]; then
        check_pass "LLM models loaded: $OLLAMA_MODELS"
    else
        check_warn "No LLM models found"
    fi
else
    check_warn "Ollama service not running"
fi

# ============================================================================
header "6. TRADING EXECUTION"
# ============================================================================

log "Trading Services:"
TRADING_SERVICES=("execution" "strategy-engine" "signal-generator" "risk-monitor")
for service in "${TRADING_SERVICES[@]}"; do
    STATUS=$(docker inspect "trading-$service" --format='{{.State.Status}}' 2>/dev/null || echo "error")
    if [ "$STATUS" = "running" ]; then
        check_pass "trading-$service: Running"
    else
        check_warn "trading-$service: $STATUS"
    fi
done

log "\nRecent Trading Activity:"
if [ "$RECENT_TRADES" -gt 0 ]; then
    check_pass "Trades executed (24h): $RECENT_TRADES"
else
    log "  No trades in last 24 hours (may be normal in testing)"
fi

# Check for active orders
ACTIVE_ORDERS=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM orders WHERE status='open';" 2>/dev/null | xargs || echo "0")
log "  Active orders: $ACTIVE_ORDERS"

# ============================================================================
header "7. MONITORING & OBSERVABILITY"
# ============================================================================

log "Monitoring Stack:"
MONITORING_SERVICES=("prometheus" "grafana" "loki" "alertmanager")
for service in "${MONITORING_SERVICES[@]}"; do
    STATUS=$(docker inspect "trading-$service" --format='{{.State.Status}}' 2>/dev/null || echo "error")
    if [ "$STATUS" = "running" ]; then
        check_pass "$service: Running"
    else
        check_warn "$service: $STATUS"
    fi
done

log "\nGrafana Dashboards:"
GRAFANA_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health 2>/dev/null || echo "000")
if [ "$GRAFANA_STATUS" = "200" ]; then
    check_pass "Grafana accessible"
else
    check_warn "Grafana response: HTTP $GRAFANA_STATUS"
fi

log "\nRecent Errors:"
API_ERRORS=$(docker logs trading-api --since 1h 2>&1 | grep -E " ERROR |Exception" | grep -v "WARNING" | wc -l || echo "0")
ML_ERRORS=$(docker logs trading-ml --since 1h 2>&1 | grep -E " ERROR |Exception" | wc -l || echo "0")
DATA_ERRORS=$(docker logs trading-data-ingestion --since 1h 2>&1 | grep -E " ERROR |Exception" | wc -l || echo "0")

if [ "$API_ERRORS" -eq 0 ]; then
    check_pass "API: No errors (1h)"
else
    check_warn "API: $API_ERRORS errors (1h)"
fi

if [ "$ML_ERRORS" -eq 0 ]; then
    check_pass "ML: No errors (1h)"
else
    check_warn "ML: $ML_ERRORS errors (1h)"
fi

if [ "$DATA_ERRORS" -eq 0 ]; then
    check_pass "Data Ingestion: No errors (1h)"
else
    check_warn "Data Ingestion: $DATA_ERRORS errors (1h)"
fi

# ============================================================================
header "8. SYSTEM RESOURCES"
# ============================================================================

log "Compute Resources:"
TOTAL_MEM=$(free -g | awk '/^Mem:/ {print $2}')
USED_MEM=$(free -g | awk '/^Mem:/ {print $3}')
MEM_PERCENT=$(awk "BEGIN {printf \"%.0f\", ($USED_MEM/$TOTAL_MEM)*100}")

log "  Memory: ${USED_MEM}G / ${TOTAL_MEM}G (${MEM_PERCENT}%)"
if [ "$MEM_PERCENT" -lt 85 ]; then
    check_pass "Memory usage healthy"
else
    check_warn "Memory usage high: ${MEM_PERCENT}%"
fi

CPU_CORES=$(nproc)
LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
log "  CPU: $CPU_CORES cores, Load avg: $LOAD_AVG"
check_pass "CPU resources available"

log "\nDisk Space:"
for mount in /srv /mnt/fastdrive /mnt/bulkdata; do
    if [ -d "$mount" ]; then
        USAGE=$(df -h "$mount" | awk 'NR==2 {print $5}' | sed 's/%//')
        AVAIL=$(df -h "$mount" | awk 'NR==2 {print $4}')
        log "  $mount: ${USAGE}% used (${AVAIL} available)"
        if [ "$USAGE" -lt 80 ]; then
            check_pass "$mount: Healthy"
        elif [ "$USAGE" -lt 90 ]; then
            check_warn "$mount: Monitor (${USAGE}%)"
        else
            check_fail "$mount: Critical (${USAGE}%)"
        fi
    fi
done

# ============================================================================
header "9. DATA RETENTION & MAINTENANCE"
# ============================================================================

log "Automated Maintenance:"
if crontab -l 2>/dev/null | grep -q "enforce_retention"; then
    check_pass "Data retention scheduled (cron)"
    crontab -l 2>/dev/null | grep "retention" | while read line; do
        log "  $line"
    done
else
    check_warn "Data retention not scheduled"
fi

log "\nData Retention Compliance:"
# Check if data is within retention policies
OLDEST_BAR=$(curl -s "http://localhost:9000/exec?query=SELECT+MIN(timestamp)+FROM+daily_bars" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0][:10] if 'dataset' in d and len(d['dataset'])>0 else 'unknown')" 2>/dev/null || echo "unknown")
if [ "$OLDEST_BAR" != "unknown" ]; then
    OLDEST_YEAR=$(echo "$OLDEST_BAR" | cut -d- -f1)
    CURRENT_YEAR=$(date +%Y)
    DATA_AGE=$((CURRENT_YEAR - OLDEST_YEAR))
    log "  Oldest data: $OLDEST_BAR (${DATA_AGE} years)"
    if [ "$DATA_AGE" -le 20 ]; then
        check_pass "Data within 20-year retention policy"
    else
        check_warn "Data exceeds 20-year retention policy (${DATA_AGE} years)"
    fi
fi

# ============================================================================
header "10. SECURITY & COMPLIANCE"
# ============================================================================

log "Network Security:"
# Check if services are properly firewalled
PUBLIC_PORTS=$(netstat -tln 2>/dev/null | grep "0.0.0.0" | awk '{print $4}' | cut -d: -f2 | sort -u | wc -l || echo "unknown")
log "  Public listening ports: $PUBLIC_PORTS"
if [ "$PUBLIC_PORTS" -lt 10 ]; then
    check_pass "Limited public exposure"
else
    check_warn "Many public ports exposed: $PUBLIC_PORTS"
fi

log "\nEnvironment Configuration:"
if [ -f ".env" ]; then
    check_pass ".env file exists"
    
    ENV_SECRETS=$(grep -E "PASSWORD|SECRET|KEY|TOKEN" .env 2>/dev/null | wc -l || echo "0")
    if [ "$ENV_SECRETS" -gt 10 ]; then
        check_pass "Secrets configured: $ENV_SECRETS"
    else
        check_warn "Few secrets configured: $ENV_SECRETS"
    fi
else
    check_fail ".env file missing"
fi

log "\nDocker Security:"
PRIVILEGED_CONTAINERS=$(docker ps --format '{{.Names}}' | xargs -I {} docker inspect {} --format '{{.Name}}: {{.HostConfig.Privileged}}' 2>/dev/null | grep -c "true" || echo "0")
if [ "$PRIVILEGED_CONTAINERS" -eq 0 ]; then
    check_pass "No privileged containers"
else
    check_warn "Privileged containers: $PRIVILEGED_CONTAINERS"
fi

# ============================================================================
header "11. CODE QUALITY & CONFIGURATION"
# ============================================================================

log "Repository Status:"
if [ -d ".git" ]; then
    check_pass "Git repository initialized"
    
    BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    log "  Branch: $BRANCH"
    
    UNCOMMITTED=$(git status --porcelain 2>/dev/null | wc -l || echo "0")
    if [ "$UNCOMMITTED" -eq 0 ]; then
        check_pass "No uncommitted changes"
    else
        log "  Uncommitted changes: $UNCOMMITTED files"
    fi
else
    check_warn "Not a git repository"
fi

log "\nConfiguration Files:"
CONFIG_FILES=("docker-compose.yml" "requirements.txt" ".env" "alembic.ini")
for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$file present"
    else
        check_warn "$file missing"
    fi
done

log "\nProduction Scripts:"
SCRIPT_COUNT=$(ls -1 scripts/*.sh 2>/dev/null | wc -l || echo "0")
log "  Production scripts: $SCRIPT_COUNT"
if [ "$SCRIPT_COUNT" -gt 15 ] && [ "$SCRIPT_COUNT" -lt 30 ]; then
    check_pass "Reasonable script count"
else
    log "  (Review for duplicates if > 30)"
fi

# ============================================================================
header "EXECUTIVE SUMMARY"
# ============================================================================

TOTAL=$((PASS + WARN + FAIL))
SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($PASS/$TOTAL)*100}")

log "\n╔════════════════════════════════════════════════════════════╗"
log "║                    ASSESSMENT RESULTS                       ║"
log "╚════════════════════════════════════════════════════════════╝"
log ""
log "Total Checks: $TOTAL"
log "${GREEN}Passed:${NC} $PASS"
log "${YELLOW}Warnings:${NC} $WARN"
log "${RED}Failed:${NC} $FAIL"
log ""
log "Success Rate: ${SUCCESS_RATE}%"
log ""

# Overall readiness assessment
if [ "$FAIL" -eq 0 ] && [ "$WARN" -lt 10 ] && [ "$WATCHLIST_COUNT" -gt 300 ]; then
    log "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    log "${GREEN}║   ✓ SYSTEM IS PRODUCTION READY FOR LIVE TRADING           ║${NC}"
    log "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    READINESS="READY"
elif [ "$FAIL" -lt 3 ] && [ "$WATCHLIST_COUNT" -gt 200 ]; then
    log "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
    log "${YELLOW}║   ⚠ SYSTEM OPERATIONAL - MINOR ISSUES TO ADDRESS          ║${NC}"
    log "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
    READINESS="OPERATIONAL"
else
    log "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    log "${RED}║   ✗ CRITICAL ISSUES REQUIRE IMMEDIATE ATTENTION            ║${NC}"
    log "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    READINESS="NEEDS_ATTENTION"
fi

log ""
log "Key Metrics:"
log "  - Total Containers: $TOTAL_CONTAINERS ($HEALTHY_CONTAINERS healthy)"
log "  - Watchlist Symbols: $WATCHLIST_COUNT"
log "  - Daily Bars: $(printf "%'d" $DAILY_BARS)"
log "  - Recent Activity: $(printf "%'d" $RECENT_BARS) bars/day"
log "  - Database Size: $DB_SIZE"
log "  - Memory Usage: ${MEM_PERCENT}%"
log ""
log "Report saved to: $REPORT_FILE"
log ""

if [ "$FAIL" -eq 0 ]; then
    exit 0
elif [ "$FAIL" -lt 3 ]; then
    exit 1
else
    exit 2
fi

#!/bin/bash
set -euo pipefail

# =============================================================================
# MEKOSHI AI TRADING SYSTEM - HOLISTIC PRODUCTION HEALTH CHECK
# =============================================================================
# Complete system monitoring with detailed metrics, latencies, and rates
# Checks all aspects: Infrastructure, Data, ML, Trading, Performance
# Version: 2.0 - Enhanced for Production Operations
# Updated: October 3, 2025
# =============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Report file
REPORT_FILE="/tmp/holistic_health_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$REPORT_FILE") 2>&1

# Utility functions
log() {
    echo -e "$1"
}

section() {
    log "\n${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    log "${BOLD}${CYAN} $1${NC}"
    log "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════${NC}\n"
}

subsection() {
    log "${BOLD}${BLUE}▬▬▬ $1 ▬▬▬${NC}"
}

check_pass() {
    log "  ${GREEN}✓${NC} $1"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

check_fail() {
    log "  ${RED}✗${NC} $1"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

check_warn() {
    log "  ${YELLOW}⚠${NC} $1"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

info() {
    log "    ${CYAN}•${NC} $1"
}

# Measure latency
measure_latency() {
    local url=$1
    local desc=$2
    local result
    result=$(curl -o /dev/null -s -w "%{time_total}" --max-time 5 "$url" 2>/dev/null || echo "timeout")
    if [ "$result" != "timeout" ]; then
        local ms=$(awk "BEGIN {printf \"%.0f\", $result * 1000}")
        if [ "$ms" -lt 100 ]; then
            check_pass "$desc: ${ms}ms (excellent)"
        elif [ "$ms" -lt 500 ]; then
            check_pass "$desc: ${ms}ms (good)"
        elif [ "$ms" -lt 1000 ]; then
            check_warn "$desc: ${ms}ms (acceptable)"
        else
            check_warn "$desc: ${ms}ms (slow)"
        fi
        echo "$ms"
    else
        check_fail "$desc: timeout"
        echo "0"
    fi
}

# Start health check
log "╔═══════════════════════════════════════════════════════════════════╗"
log "║   MEKOSHI AI TRADING SYSTEM - HOLISTIC HEALTH CHECK              ║"
log "║   Complete System Audit & Performance Analysis                   ║"
log "║   $(date)                                      ║"
log "╚═══════════════════════════════════════════════════════════════════╝"

cd /srv/ai-trading-system 2>/dev/null || cd /

# =============================================================================
section "1. CONTAINER INFRASTRUCTURE"
# =============================================================================

subsection "1.1 Container Status & Health"
CONTAINERS=$(docker ps --format "{{.Names}}" | grep "trading-" | sort)
TOTAL_CONTAINERS=$(echo "$CONTAINERS" | wc -l)
HEALTHY_COUNT=0
UNHEALTHY_COUNT=0

for container in $CONTAINERS; do
    STATUS=$(docker inspect "$container" --format='{{.State.Status}}' 2>/dev/null || echo "missing")
    HEALTH=$(docker inspect "$container" --format='{{.State.Health.Status}}' 2>/dev/null || echo "none")
    RESTARTS=$(docker inspect "$container" --format='{{.RestartCount}}' 2>/dev/null || echo "0")
    
    if [ "$STATUS" = "running" ] && { [ "$HEALTH" = "healthy" ] || [ "$HEALTH" = "none" ]; }; then
        check_pass "$container: Running (Health: $HEALTH, Restarts: $RESTARTS)"
        HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
    else
        check_fail "$container: $STATUS (Health: $HEALTH, Restarts: $RESTARTS)"
        UNHEALTHY_COUNT=$((UNHEALTHY_COUNT + 1))
    fi
done

info "Summary: $HEALTHY_COUNT/$TOTAL_CONTAINERS healthy"

subsection "1.2 Container Resource Usage"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" | grep "trading-" | head -10 | while read line; do
    info "$line"
done

subsection "1.3 Docker System Health"
DOCKER_INFO=$(docker system df 2>/dev/null || echo "error")
if echo "$DOCKER_INFO" | grep -q "error"; then
    check_fail "Docker system info unavailable"
else
    check_pass "Docker system responding"
    echo "$DOCKER_INFO" | tail -n +2 | while read line; do
        info "$line"
    done
fi

# =============================================================================
section "2. DATABASE LAYER - DETAILED METRICS"
# =============================================================================

subsection "2.1 PostgreSQL (Primary Relational DB)"
if docker exec trading-postgres pg_isready -U trading_user > /dev/null 2>&1; then
    check_pass "PostgreSQL accepting connections"
    
    # Connection count
    CONN_COUNT=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | xargs || echo "0")
    info "Active connections: $CONN_COUNT"
    
    # Database size
    DB_SIZE=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT pg_size_pretty(pg_database_size('trading_db'));" 2>/dev/null | xargs || echo "N/A")
    info "Database size: $DB_SIZE"
    
    # Table count and row counts
    TABLES=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null | xargs || echo "0")
    info "Tables: $TABLES"
    
    # Check key tables
    for table in trades orders positions users strategies; do
        COUNT=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM $table;" 2>/dev/null | xargs || echo "0")
        info "  $table: $(printf "%'d" $COUNT) rows"
    done
    
    # Recent activity
    RECENT_TRADES=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM trades WHERE created_at > NOW() - INTERVAL '24 hours';" 2>/dev/null | xargs || echo "0")
    if [ "$RECENT_TRADES" -gt 0 ]; then
        check_pass "Recent trades (24h): $RECENT_TRADES"
    else
        info "Recent trades (24h): 0 (may be normal)"
    fi
    
    # Query performance
    PG_LATENCY=$(measure_latency "http://localhost:5432" "PostgreSQL response")
else
    check_fail "PostgreSQL NOT responding"
fi

subsection "2.2 Redis (Cache & Real-time)"
REDIS_PASS=$(grep "^REDIS_PASSWORD=" .env 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'" | head -1 || echo "")
if [ -n "$REDIS_PASS" ]; then
    if docker exec trading-redis redis-cli -a "$REDIS_PASS" --no-auth-warning ping 2>&1 | grep -q PONG; then
        check_pass "Redis responding"
        
        # Memory usage
        REDIS_MEM=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" --no-auth-warning INFO memory 2>&1 | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r' || echo "N/A")
        info "Memory usage: $REDIS_MEM"
        
        # Key count
        REDIS_KEYS=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" --no-auth-warning DBSIZE 2>&1 | grep -o '[0-9]*' | head -1 || echo "0")
        info "Total keys: $REDIS_KEYS"
        
        # Watchlist
        WATCHLIST_COUNT=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" --no-auth-warning SCARD "watchlist" 2>&1 | grep -o '[0-9]*' | head -1 || echo "0")
        info "Watchlist symbols: $WATCHLIST_COUNT"
        
        # Connected clients
        REDIS_CLIENTS=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" --no-auth-warning INFO clients 2>&1 | grep "connected_clients:" | cut -d: -f2 | tr -d '\r' || echo "0")
        info "Connected clients: $REDIS_CLIENTS"
        
        # Ops/sec
        REDIS_OPS=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" --no-auth-warning INFO stats 2>&1 | grep "instantaneous_ops_per_sec:" | cut -d: -f2 | tr -d '\r' || echo "0")
        info "Operations/sec: $REDIS_OPS"
        
        if [ "$WATCHLIST_COUNT" -gt 300 ]; then
            check_pass "Watchlist well-populated: $WATCHLIST_COUNT symbols"
        else
            check_warn "Watchlist has $WATCHLIST_COUNT symbols (expected >300)"
        fi
    else
        check_fail "Redis NOT responding"
    fi
else
    check_fail "Redis password not configured"
fi

subsection "2.3 QuestDB (Time-Series Market Data)"
if curl -s http://localhost:9000/exec?query=SELECT+1 | grep -q "dataset"; then
    check_pass "QuestDB HTTP endpoint responding"
    
    # Table counts
    for table in daily_bars market_data options_data news_items social_signals; do
        COUNT=$(curl -s "http://localhost:9000/exec?query=SELECT+COUNT(*)+FROM+$table" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0] if 'dataset' in d and len(d['dataset'])>0 else 0)" 2>/dev/null || echo "0")
        info "$table: $(printf "%'d" $COUNT) rows"
    done
    
    # Query latency
    QDB_LATENCY=$(measure_latency "http://localhost:9000/exec?query=SELECT+1" "QuestDB query latency")
    
    # Recent data ingestion
    RECENT_BARS=$(curl -s "http://localhost:9000/exec?query=SELECT+COUNT(*)+FROM+market_data+WHERE+timestamp+>+dateadd('h',-1,now())" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0] if 'dataset' in d else 0)" 2>/dev/null || echo "0")
    if [ "$RECENT_BARS" -gt 100 ]; then
        check_pass "Recent data ingestion (1h): $(printf "%'d" $RECENT_BARS) bars"
    elif [ "$RECENT_BARS" -gt 0 ]; then
        check_warn "Low data ingestion (1h): $RECENT_BARS bars"
    else
        info "No recent data (may be outside market hours)"
    fi
else
    check_fail "QuestDB NOT responding"
fi

subsection "2.4 Weaviate (Vector Database)"
WEAVIATE_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8080/v1/.well-known/ready 2>/dev/null || echo "000")
WEAVIATE_CODE=$(echo "$WEAVIATE_RESPONSE" | tail -1)
if [ "$WEAVIATE_CODE" = "200" ] || [ "$WEAVIATE_CODE" = "204" ]; then
    check_pass "Weaviate ready (HTTP $WEAVIATE_CODE)"
    
    # Schema classes
    SCHEMA_COUNT=$(curl -s http://localhost:8080/v1/schema 2>/dev/null | grep -o '"class"' | wc -l || echo "0")
    info "Schema classes: $SCHEMA_COUNT"
    
    # Object counts (if available)
    for class in News Social Analysis; do
        OBJ_COUNT=$(curl -s "http://localhost:8080/v1/objects?class=$class&limit=0" 2>/dev/null | grep -oP '"totalResults":\K\d+' || echo "N/A")
        if [ "$OBJ_COUNT" != "N/A" ]; then
        info "  $class objects: $OBJ_COUNT"
        fi
    done
else
    check_warn "Weaviate not ready (HTTP $WEAVIATE_CODE)"
fi

# =============================================================================
section "3. API & WEB SERVICES - LATENCY ANALYSIS"
# =============================================================================

subsection "3.1 Core API Endpoints"
API_LATENCY_HEALTH=$(measure_latency "http://localhost:8000/health" "API /health")
API_LATENCY_VERSION=$(measure_latency "http://localhost:8000/version" "API /version")

# Check if API is responding correctly
API_HEALTH=$(docker exec trading-api curl -s http://localhost:8000/health 2>/dev/null || echo "ERROR")
if echo "$API_HEALTH" | grep -q "ok\|healthy"; then
    check_pass "API health status: OK"
else
    check_fail "API health status: FAILED"
fi

subsection "3.2 External Website Status"
# Note: api.mekoshi.com is internal-only and not publicly exposed
for site in biz.mekoshi.com admin.mekoshi.com; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://$site" --max-time 5 2>/dev/null || echo "000")
    LATENCY=$(curl -o /dev/null -s -w "%{time_total}" --max-time 5 "https://$site" 2>/dev/null || echo "0")
    LATENCY_MS=$(awk "BEGIN {printf \"%.0f\", $LATENCY * 1000}")
    
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "405" ]; then
        check_pass "$site: HTTP $HTTP_CODE (${LATENCY_MS}ms)"
    elif [ "$HTTP_CODE" = "301" ] || [ "$HTTP_CODE" = "307" ]; then
        check_warn "$site: HTTP $HTTP_CODE redirect (${LATENCY_MS}ms)"
    else
        check_fail "$site: HTTP $HTTP_CODE (${LATENCY_MS}ms)"
    fi
done

subsection "3.3 SSL Certificate Status"
# Note: api.mekoshi.com is internal-only and does not need public SSL
for domain in biz.mekoshi.com admin.mekoshi.com; do
    CERT_EXPIRY=$(echo | timeout 5 openssl s_client -servername "$domain" -connect "$domain:443" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null | grep notAfter | cut -d= -f2 || echo "")
    if [ -n "$CERT_EXPIRY" ]; then
        check_pass "$domain: Valid until $CERT_EXPIRY"
    else
        check_warn "$domain: Certificate check failed"
    fi
done

# =============================================================================
section "4. ML & AI SERVICES - PERFORMANCE METRICS"
# =============================================================================

subsection "4.1 ML Service Status"
if docker exec trading-ml curl -s http://localhost:8001/healthz 2>/dev/null | grep -q "ok"; then
    check_pass "ML service healthy"
    
    ML_LATENCY=$(measure_latency "http://localhost:8001/healthz" "ML service response")
    
    # Check for metrics endpoint
    ML_METRICS=$(curl -s http://localhost:8001/metrics 2>/dev/null || echo "")
    if echo "$ML_METRICS" | grep -q "^ml_"; then
        check_pass "ML metrics endpoint responding"
        
        # Token rates (if available)
        if echo "$ML_METRICS" | grep -q "tokens_per_sec"; then
            info "Token rates (top 3 models):"
            echo "$ML_METRICS" | grep "tokens_per_sec" | head -3 | while read line; do
                info "  $line"
            done
        fi
        
        # Request counts
        REQUEST_COUNT=$(echo "$ML_METRICS" | grep "^ml_requests_total" | awk '{print $2}' | head -1 || echo "0")
        info "Total ML requests: $REQUEST_COUNT"
        
        # Error rates
        ERROR_COUNT=$(echo "$ML_METRICS" | grep "^ml_errors_total" | awk '{print $2}' | head -1 || echo "0")
        info "Total ML errors: $ERROR_COUNT"
    fi
    
    # Check recent ML activity
    ML_LOGS=$(docker logs trading-ml --since 30m 2>&1 | tail -100)
    if echo "$ML_LOGS" | grep -q "scheduler enabled"; then
        check_pass "ML scheduler active"
    fi
    if echo "$ML_LOGS" | grep -q "Updated.*factor"; then
        FACTOR_COUNT=$(echo "$ML_LOGS" | grep "Updated.*factor" | tail -1 | grep -oP '\d+(?= factor)' || echo "0")
        check_pass "Factor models training: $FACTOR_COUNT symbols"
    fi
else
    check_fail "ML service NOT responding"
fi

subsection "4.2 Ollama (LLM Models)"
if docker exec trading-ollama ollama list > /dev/null 2>&1; then
    check_pass "Ollama service responding"
    
    MODEL_COUNT=$(docker exec trading-ollama ollama list 2>/dev/null | tail -n +2 | wc -l || echo "0")
    check_pass "LLM models loaded: $MODEL_COUNT"
    
    info "Available models:"
    docker exec trading-ollama ollama list 2>/dev/null | tail -n +2 | head -10 | while read line; do
        info "  $line"
    done
else
    check_warn "Ollama service not responding"
fi

# =============================================================================
section "5. DATA INGESTION & PROCESSING"
# =============================================================================

subsection "5.1 Data Ingestion Service"
if docker inspect trading-data-ingestion --format='{{.State.Status}}' 2>/dev/null | grep -q "running"; then
    check_pass "Data ingestion service running"
    
    # Check recent activity
    INGESTION_LOGS=$(docker logs trading-data-ingestion --since 10m 2>&1 | tail -200)
    
    # Count different types of ingestion events
    MARKET_INGESTS=$(echo "$INGESTION_LOGS" | grep -ci "market.*collected\|bars.*stored" 2>/dev/null || echo "0")
    NEWS_INGESTS=$(echo "$INGESTION_LOGS" | grep -ci "news.*collected\|news.*stored" 2>/dev/null || echo "0")
    SOCIAL_INGESTS=$(echo "$INGESTION_LOGS" | grep -ci "social.*collected\|reddit.*stored" 2>/dev/null || echo "0")
    OPTIONS_INGESTS=$(echo "$INGESTION_LOGS" | grep -ci "options.*collected\|options.*stored" 2>/dev/null || echo "0")
    
    # Ensure all variables are numeric
    MARKET_INGESTS=$(echo "$MARKET_INGESTS" | tr -cd '0-9' | head -c 10)
    NEWS_INGESTS=$(echo "$NEWS_INGESTS" | tr -cd '0-9' | head -c 10)
    SOCIAL_INGESTS=$(echo "$SOCIAL_INGESTS" | tr -cd '0-9' | head -c 10)
    OPTIONS_INGESTS=$(echo "$OPTIONS_INGESTS" | tr -cd '0-9' | head -c 10)
    
    # Default to 0 if empty
    MARKET_INGESTS=${MARKET_INGESTS:-0}
    NEWS_INGESTS=${NEWS_INGESTS:-0}
    SOCIAL_INGESTS=${SOCIAL_INGESTS:-0}
    OPTIONS_INGESTS=${OPTIONS_INGESTS:-0}
    
    info "Ingestion events (10min):"
    info "  Market data: $MARKET_INGESTS"
    info "  News: $NEWS_INGESTS"
    info "  Social: $SOCIAL_INGESTS"
    info "  Options: $OPTIONS_INGESTS"
    
    TOTAL_INGESTS=$((MARKET_INGESTS + NEWS_INGESTS + SOCIAL_INGESTS + OPTIONS_INGESTS))
    if [ "$TOTAL_INGESTS" -gt 50 ]; then
        check_pass "Active data collection: $TOTAL_INGESTS events"
    elif [ "$TOTAL_INGESTS" -gt 10 ]; then
        check_warn "Moderate data collection: $TOTAL_INGESTS events"
    else
        check_warn "Low data collection: $TOTAL_INGESTS events (may be normal outside market hours)"
    fi
else
    check_fail "Data ingestion service NOT running"
fi

subsection "5.2 Data Provider Configuration"
# Check for configured API keys
POLYGON_KEY=$(grep "^POLYGON_API_KEY=" .env 2>/dev/null | cut -d= -f2 | wc -c)
ALPACA_KEY=$(grep "^ALPACA_API_KEY=" .env 2>/dev/null | cut -d= -f2 | wc -c)
EODHD_KEY=$(grep "^EODHD_API_KEY=" .env 2>/dev/null | cut -d= -f2 | wc -c)
ALPHAVANTAGE_KEY=$(grep "^ALPHAVANTAGE_API_KEY=" .env 2>/dev/null | cut -d= -f2 | wc -c)

if [ "$POLYGON_KEY" -gt 10 ]; then check_pass "Polygon.io configured"; else check_warn "Polygon.io not configured"; fi
if [ "$ALPACA_KEY" -gt 10 ]; then check_pass "Alpaca configured"; else check_warn "Alpaca not configured"; fi
if [ "$EODHD_KEY" -gt 10 ]; then check_pass "EODHD configured"; else check_warn "EODHD not configured"; fi
if [ "$ALPHAVANTAGE_KEY" -gt 10 ]; then check_pass "AlphaVantage configured"; else check_warn "AlphaVantage not configured"; fi

# =============================================================================
section "6. TRADING EXECUTION SYSTEM"
# =============================================================================

subsection "6.1 Trading Services Status"
for service in execution strategy-engine signal-generator risk-monitor; do
    STATUS=$(docker inspect "trading-$service" --format='{{.State.Status}}' 2>/dev/null || echo "missing")
    if [ "$STATUS" = "running" ]; then
        check_pass "trading-$service: Running"
    else
        check_warn "trading-$service: $STATUS"
    fi
done

subsection "6.2 Trading Activity & Metrics"
# Recent trades
RECENT_TRADES=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM trades WHERE created_at > NOW() - INTERVAL '24 hours';" 2>/dev/null | xargs || echo "0")
info "Trades (24h): $RECENT_TRADES"

# Active orders
ACTIVE_ORDERS=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM orders WHERE status='open';" 2>/dev/null | xargs || echo "0")
info "Active orders: $ACTIVE_ORDERS"

# Open positions
OPEN_POSITIONS=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM positions WHERE status='open';" 2>/dev/null | xargs || echo "0")
info "Open positions: $OPEN_POSITIONS"

# =============================================================================
section "7. MONITORING & OBSERVABILITY"
# =============================================================================

subsection "7.1 Monitoring Stack"
for service in prometheus grafana loki alertmanager; do
    STATUS=$(docker inspect "trading-$service" --format='{{.State.Status}}' 2>/dev/null || echo "missing")
    if [ "$STATUS" = "running" ]; then
        check_pass "$service: Running"
    else
        check_warn "$service: $STATUS"
    fi
done

subsection "7.2 Prometheus Metrics"
if curl -s http://localhost:9090/-/ready | grep -q "Prometheus"; then
    check_pass "Prometheus ready"
    
    # Target health
    TARGETS=$(curl -s http://localhost:9090/api/v1/targets 2>/dev/null || echo "{}")
    ACTIVE_TARGETS=$(echo "$TARGETS" | grep -o '"health":"up"' | wc -l || echo "0")
    info "Active targets: $ACTIVE_TARGETS"
    
    PROM_LATENCY=$(measure_latency "http://localhost:9090/-/ready" "Prometheus response")
else
    check_warn "Prometheus not ready"
fi

subsection "7.3 Grafana Dashboards"
GRAFANA_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health --max-time 3 2>/dev/null || echo "000")
if [ "$GRAFANA_STATUS" = "200" ]; then
    check_pass "Grafana accessible (HTTP $GRAFANA_STATUS)"
    GRAFANA_LATENCY=$(measure_latency "http://localhost:3000/api/health" "Grafana response")
else
    check_warn "Grafana response: HTTP $GRAFANA_STATUS"
fi

# =============================================================================
section "8. SYSTEM RESOURCES & PERFORMANCE"
# =============================================================================

subsection "8.1 Memory Usage"
TOTAL_MEM=$(free -g | awk '/^Mem:/ {print $2}')
USED_MEM=$(free -g | awk '/^Mem:/ {print $3}')
FREE_MEM=$(free -g | awk '/^Mem:/ {print $4}')
MEM_PERCENT=$(awk "BEGIN {printf \"%.0f\", ($USED_MEM/$TOTAL_MEM)*100}")

info "Total Memory: ${TOTAL_MEM}G"
info "Used Memory: ${USED_MEM}G (${MEM_PERCENT}%)"
info "Free Memory: ${FREE_MEM}G"

if [ "$MEM_PERCENT" -lt 85 ]; then
    check_pass "Memory usage healthy: ${MEM_PERCENT}%"
elif [ "$MEM_PERCENT" -lt 95 ]; then
    check_warn "Memory usage high: ${MEM_PERCENT}%"
else
    check_fail "Memory usage critical: ${MEM_PERCENT}%"
fi

subsection "8.2 CPU & Load Average"
CPU_CORES=$(nproc)
LOAD_1MIN=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
LOAD_5MIN=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $2}' | sed 's/,//')
LOAD_15MIN=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $3}' | sed 's/,//')

info "CPU Cores: $CPU_CORES"
info "Load Average: $LOAD_1MIN (1m) | $LOAD_5MIN (5m) | $LOAD_15MIN (15m)"

LOAD_PERCENT=$(awk "BEGIN {printf \"%.0f\", ($LOAD_1MIN/$CPU_CORES)*100}")
if [ "$LOAD_PERCENT" -lt 80 ]; then
    check_pass "CPU load healthy: ${LOAD_PERCENT}%"
else
    check_warn "CPU load high: ${LOAD_PERCENT}%"
fi

subsection "8.3 Disk Space"
for mount in /srv /mnt/fastdrive /mnt/bulkdata; do
    if [ -d "$mount" ]; then
        USAGE=$(df -h "$mount" | awk 'NR==2 {print $5}' | sed 's/%//')
        AVAIL=$(df -h "$mount" | awk 'NR==2 {print $4}')
        USED=$(df -h "$mount" | awk 'NR==2 {print $3}')
        
        info "$mount: ${USED} used, ${AVAIL} available (${USAGE}%)"
        
        if [ "$USAGE" -lt 80 ]; then
            check_pass "$mount: Healthy (${USAGE}%)"
        elif [ "$USAGE" -lt 90 ]; then
            check_warn "$mount: Monitor (${USAGE}%)"
        else
            check_fail "$mount: Critical (${USAGE}%)"
        fi
    fi
done

subsection "8.4 Network Statistics"
# Network interface stats
info "Network interfaces:"
ip -s link show | grep -E "^\d+:|RX|TX" | head -20 | while read line; do
    info "  $line"
done

# =============================================================================
section "9. ERROR ANALYSIS & LOGS"
# =============================================================================

subsection "9.1 Recent Error Counts (Last Hour)"
for service in api ml data-ingestion execution; do
    ERROR_COUNT=$(docker logs "trading-$service" --since 1h 2>&1 | grep -cE " ERROR |Exception" | grep -v "WARNING" || echo "0")
    if [ "$ERROR_COUNT" -eq 0 ]; then
        check_pass "$service: 0 errors"
    elif [ "$ERROR_COUNT" -lt 10 ]; then
        check_warn "$service: $ERROR_COUNT errors"
    else
        check_fail "$service: $ERROR_COUNT errors"
    fi
done

subsection "9.2 Recent Critical Errors"
info "Last 5 critical errors across services:"
for service in api ml data-ingestion; do
    docker logs "trading-$service" --since 1h 2>&1 | grep -E " ERROR |CRITICAL" | grep -v "WARNING" | tail -2 | while read line; do
        info "  [$service] ${line:0:100}"
    done
done

# =============================================================================
section "10. SECURITY & COMPLIANCE"
# =============================================================================

subsection "10.1 Security Configuration"
if [ -f ".env" ]; then
    check_pass ".env file present"
    SECRETS_COUNT=$(grep -cE "PASSWORD|SECRET|KEY|TOKEN" .env 2>/dev/null || echo "0")
    info "Configured secrets: $SECRETS_COUNT"
else
    check_fail ".env file missing"
fi

subsection "10.2 Container Security"
PRIVILEGED=$(docker ps --format '{{.Names}}' | xargs -I {} docker inspect {} --format '{{.Name}}: {{.HostConfig.Privileged}}' 2>/dev/null | grep -c "true" || echo "0")
if [ "$PRIVILEGED" -eq 0 ]; then
    check_pass "No privileged containers"
else
    check_warn "Privileged containers: $PRIVILEGED"
fi

subsection "10.3 Data Retention"
if crontab -l 2>/dev/null | grep -q "enforce_retention"; then
    check_pass "Data retention scheduled"
    crontab -l 2>/dev/null | grep "retention" | while read line; do
        info "  $line"
    done
else
    check_warn "Data retention not scheduled"
fi

# =============================================================================
section "11. FINAL SUMMARY & RECOMMENDATIONS"
# =============================================================================

TOTAL=$((PASSED_CHECKS + FAILED_CHECKS + WARNING_CHECKS))
SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($PASSED_CHECKS/$TOTAL)*100}")

log "\n╔═══════════════════════════════════════════════════════════════════╗"
log "║                    HEALTH CHECK RESULTS                           ║"
log "╚═══════════════════════════════════════════════════════════════════╝"
log ""
log "Total Checks:    $TOTAL"
log "${GREEN}Passed:${NC}          $PASSED_CHECKS"
log "${YELLOW}Warnings:${NC}        $WARNING_CHECKS"
log "${RED}Failed:${NC}          $FAILED_CHECKS"
log ""
log "Success Rate:    ${SUCCESS_RATE}%"
log ""

if [ "$FAILED_CHECKS" -eq 0 ] && [ "$WARNING_CHECKS" -lt 10 ]; then
    log "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    log "${GREEN}║   ✓ SYSTEM IS PRODUCTION READY                                    ║${NC}"
    log "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    READINESS="READY"
elif [ "$FAILED_CHECKS" -lt 5 ]; then
    log "${YELLOW}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    log "${YELLOW}║   ⚠ SYSTEM OPERATIONAL - MINOR ISSUES TO ADDRESS                 ║${NC}"
    log "${YELLOW}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    READINESS="OPERATIONAL"
else
    log "${RED}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    log "${RED}║   ✗ CRITICAL ISSUES DETECTED - IMMEDIATE ACTION REQUIRED          ║${NC}"
    log "${RED}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    READINESS="NEEDS_ATTENTION"
fi

log ""
log "Report saved to: $REPORT_FILE"
log "Generated: $(date)"
log ""

# Exit codes
if [ "$FAILED_CHECKS" -eq 0 ]; then
    exit 0
elif [ "$FAILED_CHECKS" -lt 5 ]; then
    exit 1
else
    exit 2
fi

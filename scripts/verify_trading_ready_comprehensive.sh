#!/bin/bash
#
# Comprehensive Trading System Readiness Verification
# Validates all components for live trading operations
#

set +e  # Continue on errors
cd /srv/ai-trading-system

echo "=========================================="
echo "TRADING SYSTEM READINESS CHECK"
echo "$(date)"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0
WARN=0

check() {
    local name="$1"
    local status=$2
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $name"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $name"
        ((FAIL++))
    fi
}

warn() {
    local name="$1"
    echo -e "${YELLOW}⚠${NC} $name"
    ((WARN++))
}

info() {
    local name="$1"
    echo -e "${BLUE}ℹ${NC} $name"
}

# 1. Container Health
echo "1. CONTAINER HEALTH"
echo "-------------------"
CONTAINERS="trading-api trading-data-ingestion trading-ml trading-strategy-engine trading-execution trading-risk-monitor"
for container in $CONTAINERS; do
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        check "$container running" 0
    else
        check "$container running" 1
    fi
done
echo ""

# 2. Infrastructure Health
echo "2. INFRASTRUCTURE HEALTH"
echo "------------------------"
INFRA="postgres redis questdb pulsar weaviate minio traefik"
for service in $INFRA; do
    if docker ps --format '{{.Names}}' | grep -q "$service"; then
        check "$service running" 0
    else
        check "$service running" 1
    fi
done
echo ""

# 3. API Endpoints
echo "3. API ENDPOINTS"
echo "----------------"
# Test API health
if curl -s -f http://localhost:8000/health >/dev/null 2>&1; then
    check "API health endpoint" 0
else
    check "API health endpoint" 1
fi

# Test new intelligence endpoint
if curl -s -f http://localhost:8000/business/api/market/heatmap >/dev/null 2>&1; then
    check "Intelligence API" 0
else
    check "Intelligence API" 1
fi

# Test admin god-mode endpoint
if curl -s -f http://localhost:8000/admin/api/ml/models >/dev/null 2>&1; then
    check "Admin God-Mode API" 0
else
    check "Admin God-Mode API" 1
fi
echo ""

# 4. Dashboard Access
echo "4. DASHBOARD ACCESS"
echo "-------------------"
# Note: Dashboards require authentication
info "Admin dashboard: http://localhost:8000/admin (requires auth)"
info "Business dashboard: http://localhost:8000/business (requires auth)"
echo ""

# 5. Data Coverage
echo "5. DATA COVERAGE"
echo "----------------"
DATA_STATS=$(bash scripts/check_data_coverage.sh 2>/dev/null)

if [ -n "$DATA_STATS" ]; then
    IFS='|' read -r market options news social symbols <<< "$DATA_STATS"
    
    if [ -n "$market" ] && [ "$market" -gt 0 ]; then
        check "Market bars: $(printf "%'d" $market)" 0
    else
        warn "Market bars: 0"
    fi
    
    if [ -n "$options" ] && [ "$options" -gt 0 ]; then
        check "Options bars: $(printf "%'d" $options)" 0
    else
        warn "Options bars: 0"
    fi
    
    if [ -n "$news" ] && [ "$news" -gt 0 ]; then
        check "News items: $(printf "%'d" $news)" 0
    else
        warn "News items: 0"
    fi
    
    if [ -n "$social" ] && [ "$social" -gt 0 ]; then
        check "Social signals: $(printf "%'d" $social)" 0
    else
        warn "Social signals: 0"
    fi
else
    warn "Could not query data coverage"
fi
echo ""

# 6. Watchlist Size
echo "6. WATCHLIST STATUS"
echo "-------------------"
REDIS_PASS=$(grep "^REDIS_PASSWORD=" .env | cut -d= -f2)
WATCHLIST_SIZE=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" SCARD watchlist 2>&1 | grep -v Warning)
if [ -n "$WATCHLIST_SIZE" ] && [ "$WATCHLIST_SIZE" -gt 0 ]; then
    check "Watchlist populated: $WATCHLIST_SIZE symbols" 0
else
    check "Watchlist populated" 1
fi
echo ""

# 7. ML Models
echo "7. ML MODELS"
echo "------------"
MODELS=$(docker exec trading-api curl -s http://ollama:11434/api/tags 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('models', [])))" 2>/dev/null)
if [ -n "$MODELS" ] && [ "$MODELS" -gt 0 ]; then
    check "ML models loaded: $MODELS models" 0
else
    check "ML models loaded" 1
fi
echo ""

# 8. Backfill Progress
echo "8. BACKFILL STATUS"
echo "------------------"
BACKFILL_STATUS=$(docker logs trading-data-ingestion 2>&1 | tail -100 | grep -i "backfill" | tail -3)
if [ -n "$BACKFILL_STATUS" ]; then
    echo "$BACKFILL_STATUS" | while read line; do
        info "$line"
    done
else
    info "No recent backfill activity"
fi
echo ""

# 9. Real-time Streaming
echo "9. REAL-TIME STREAMING"
echo "----------------------"
STREAM_STATUS=$(docker exec trading-data-ingestion curl -s http://localhost:8002/streams/status 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print('\n'.join([f'{k}: {v.get(\"enabled\", False)}' for k,v in data.items()]))" 2>/dev/null)
if [ -n "$STREAM_STATUS" ]; then
    echo "$STREAM_STATUS" | while read line; do
        if echo "$line" | grep -q "True"; then
            check "$line" 0
        else
            warn "$line"
        fi
    done
else
    warn "Could not query stream status"
fi
echo ""

# 10. Trading Readiness
echo "10. TRADING EXECUTION READINESS"
echo "--------------------------------"
# Check if execution service is ready
EXEC_READY=$(docker exec trading-execution curl -s http://localhost:8004/health 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', ''))" 2>/dev/null)
if [ "$EXEC_READY" = "healthy" ]; then
    check "Execution service ready" 0
else
    warn "Execution service status: $EXEC_READY"
fi
echo ""

# Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo -e "${GREEN}Passed:${NC} $PASS"
echo -e "${YELLOW}Warnings:${NC} $WARN"
echo -e "${RED}Failed:${NC} $FAIL"
echo ""

TOTAL=$((PASS + WARN + FAIL))
if [ $TOTAL -gt 0 ]; then
    SUCCESS_RATE=$((PASS * 100 / TOTAL))
    echo "Success Rate: ${SUCCESS_RATE}%"
fi
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ SYSTEM IS TRADING-READY${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Access admin dashboard: http://localhost:8000/admin"
    echo "  2. Access business dashboard: http://localhost:8000/business"
    echo "  3. Monitor live intelligence and options flow"
    echo "  4. Review PhD-level symbol analysis"
    echo "  5. Adjust model weights and factor configurations"
    echo ""
    exit 0
else
    echo -e "${RED}✗ SYSTEM NOT READY - $FAIL critical issues${NC}"
    echo ""
    echo "Fix the failed checks above before trading."
    echo ""
    exit 1
fi

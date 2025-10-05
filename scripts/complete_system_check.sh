#!/bin/bash
#
# Complete System Readiness Check
# Verifies all services, data, and trading system operational status
#

set -euo pipefail

# Source credentials
if [ -f /srv/ai-trading-system/.env ]; then
    export $(grep -v '^#' /srv/ai-trading-system/.env | grep -E '^(REDIS_PASSWORD|DB_PASSWORD|DB_USER)=' | xargs)
fi

echo "========================================"
echo "TRADING SYSTEM READINESS CHECK"
echo "$(date)"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    if [ "$1" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $2"
    elif [ "$1" = "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

ISSUES=0

# 1. Service Health Check
echo "1. MICROSERVICES HEALTH"
echo "   ===================="
SERVICES=("ml" "data-ingestion" "signal-generator" "execution" "risk-monitor" "strategy-engine" "backtesting" "api")
HEALTHY_COUNT=0

for service in "${SERVICES[@]}"; do
    STATUS=$(docker ps --filter "name=trading-$service" --format "{{.Status}}" 2>/dev/null | grep -o "healthy" || echo "")
    if [ "$STATUS" = "healthy" ]; then
        print_status "OK" "trading-$service"
        HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
    else
        print_status "FAIL" "trading-$service (not healthy)"
        ISSUES=$((ISSUES + 1))
    fi
done
echo "   Services: $HEALTHY_COUNT/${#SERVICES[@]} healthy"
echo ""

# 2. Trading Configuration
echo "2. TRADING CONFIGURATION"
echo "   ===================="
if [ -f /srv/ai-trading-system/.env ]; then
    PAPER_TRADING=$(grep "^PAPER_TRADING=" /srv/ai-trading-system/.env | cut -d'=' -f2)
    TRADING_MODE=$(grep "^TRADING_MODE=" /srv/ai-trading-system/.env | cut -d'=' -f2)
    print_status "OK" "Paper Trading: $PAPER_TRADING"
    print_status "OK" "Trading Mode: $TRADING_MODE"
else
    print_status "FAIL" ".env file not found"
    ISSUES=$((ISSUES + 1))
fi
echo ""

# 3. Data Availability
echo "3. DATA INVENTORY"
echo "   =============="
MARKET_DATA=$(timeout 5 curl -sG "http://localhost:9000/exec" --data-urlencode "query=SELECT count(*) FROM market_data" 2>/dev/null | jq -r '.dataset[0][0]' 2>/dev/null || echo "0")
WATCHLIST=$(docker exec trading-redis redis-cli -a "$REDIS_PASSWORD" SCARD watchlist 2>/dev/null || echo "0")
NEWS=$(docker exec trading-postgres psql -U "$DB_USER" -d trading_db -t -c "SELECT COUNT(*) FROM news_events" 2>/dev/null | tr -d ' ' || echo "0")

if [ "$MARKET_DATA" -gt 1000000 ]; then
    print_status "OK" "Market Data: $(printf "%'d" $MARKET_DATA) bars"
else
    print_status "WARN" "Market Data: $MARKET_DATA bars (low)"
    ISSUES=$((ISSUES + 1))
fi

if [ "$WATCHLIST" -gt 100 ]; then
    print_status "OK" "Watchlist: $WATCHLIST symbols"
else
    print_status "WARN" "Watchlist: $WATCHLIST symbols (low)"
fi

if [ "$NEWS" -gt 1000 ]; then
    print_status "OK" "News Events: $(printf "%'d" $NEWS)"
else
    print_status "WARN" "News Events: $NEWS (may need backfill)"
fi
echo ""

# 4. API Endpoints
echo "4. API HEALTH"
echo "   =========="
API_ENDPOINTS=("health" "dashboard/services/health" "dashboard/watchlist/all")
API_OK=0

for endpoint in "${API_ENDPOINTS[@]}"; do
    HTTP_CODE=$(timeout 5 curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/api/$endpoint" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        print_status "OK" "/api/$endpoint"
        API_OK=$((API_OK + 1))
    else
        print_status "FAIL" "/api/$endpoint (HTTP $HTTP_CODE)"
        ISSUES=$((ISSUES + 1))
    fi
done
echo "   Endpoints: $API_OK/${#API_ENDPOINTS[@]} responding"
echo ""

# 5. Discovery Status
echo "5. SYMBOL DISCOVERY"
echo "   ================"
DISCOVERY_RUNNING=$(ps aux | grep "options_symbol_discovery.py" | grep -v grep | wc -l || echo "0")
if [ "$DISCOVERY_RUNNING" -gt 0 ]; then
    print_status "OK" "Fast discovery job running"
    echo "   Check progress: tail -f /tmp/discovery.log"
else
    print_status "OK" "No discovery running (can run on-demand)"
fi
echo ""

# 6. Login Page Update
echo "6. LOGIN PAGE"
echo "   =========="
LOGIN_TEXT=$(timeout 5 curl -s http://localhost:8000/auth/login 2>/dev/null | grep -o "subtitle\">[^<]*" | cut -d'>' -f2 || echo "")
if [[ "$LOGIN_TEXT" == "Advanced Trading Intelligence Platform" ]]; then
    print_status "OK" "Login page updated (PhD removed)"
else
    print_status "WARN" "Login shows: '$LOGIN_TEXT'"
fi
echo ""

# Final Summary
echo "========================================"
echo "SYSTEM STATUS SUMMARY"
echo "========================================"
echo ""

if [ "$ISSUES" -eq 0 ] && [ "$HEALTHY_COUNT" -ge 7 ]; then
    echo -e "${GREEN}✓✓✓ SYSTEM READY FOR TRADING ✓✓✓${NC}"
    echo ""
    echo "Status: OPERATIONAL"
    echo "Services: $HEALTHY_COUNT/${#SERVICES[@]} healthy"
    echo "Data: $(printf "%'d" $MARKET_DATA)+ bars available"
    echo "Watchlist: $WATCHLIST symbols"
    echo "Mode: $TRADING_MODE (Paper: $PAPER_TRADING)"
    echo ""
    echo "Access URLs:"
    echo "  Login: https://biz.mekoshi.com/auth/login"
    echo "  Business: https://biz.mekoshi.com/business"
    echo "  Admin: https://admin.mekoshi.com/admin"
    echo ""
    exit 0
else
    echo -e "${YELLOW}⚠ SYSTEM OPERATIONAL WITH ISSUES${NC}"
    echo ""
    echo "Issues found: $ISSUES"
    echo "Services: $HEALTHY_COUNT/${#SERVICES[@]} healthy"
    echo ""
    echo "Review logs: docker-compose logs [service-name]"
    echo ""
    exit 1
fi

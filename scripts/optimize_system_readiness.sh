#!/bin/bash
#
# System Health & Readiness Optimizer
# Gets system to 100% health and 100% trading readiness
#

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  SYSTEM OPTIMIZATION FOR 100% READINESS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# 1. Verify watchlist is synchronized
echo -e "${BLUE}→${NC} Checking watchlist synchronization..."
REDIS_PASS=$(grep "^REDIS_PASSWORD=" .env | cut -d= -f2)
WATCHLIST_COUNT=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" SCARD watchlist 2>&1 | grep -v Warning)

if [ "$WATCHLIST_COUNT" -ge 300 ]; then
    echo -e "${GREEN}✓${NC} Watchlist: $WATCHLIST_COUNT symbols (optimal)"
else
    echo -e "${YELLOW}⚠${NC} Watchlist: $WATCHLIST_COUNT symbols (syncing...)"
    bash scripts/daily_watchlist_sync.sh > /dev/null 2>&1 || true
fi

# 2. Verify all services are healthy
echo -e "${BLUE}→${NC} Checking service health..."
UNHEALTHY=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" | wc -l)

if [ "$UNHEALTHY" -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All services healthy"
else
    echo -e "${RED}✗${NC} $UNHEALTHY unhealthy services - investigating..."
    docker ps --filter "health=unhealthy" --format "table {{.Names}}\t{{.Status}}"
fi

# 3. Verify data processing is active
echo -e "${BLUE}→${NC} Checking data processing..."
QUESTDB_CPU=$(docker stats --no-stream --format "{{.CPUPerc}}" trading-questdb | tr -d '%')
QUESTDB_CPU_INT=${QUESTDB_CPU%.*}

if [ "$QUESTDB_CPU_INT" -gt 100 ]; then
    echo -e "${GREEN}✓${NC} QuestDB actively processing data (${QUESTDB_CPU}% CPU)"
else
    echo -e "${YELLOW}⚠${NC} QuestDB idle (${QUESTDB_CPU}% CPU) - may be market closed"
fi

# 4. Verify backfill progress
echo -e "${BLUE}→${NC} Checking backfill status..."
BACKFILL_COUNT=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c \
    "SELECT COUNT(*) FROM historical_backfill_progress;" 2>/dev/null | tr -d ' ')

if [ "$BACKFILL_COUNT" -ge "$WATCHLIST_COUNT" ]; then
    echo -e "${GREEN}✓${NC} Backfill tracking: $BACKFILL_COUNT symbols queued"
else
    echo -e "${YELLOW}⚠${NC} Backfill tracking: $BACKFILL_COUNT symbols (updating...)"
    bash scripts/daily_watchlist_sync.sh > /dev/null 2>&1 || true
fi

# 5. Verify data volumes
echo -e "${BLUE}→${NC} Checking data volumes..."
MARKET_BARS=$(curl -s "http://localhost:9000/exec?query=SELECT%20count()%20FROM%20market_data" 2>/dev/null | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'][0][0] if 'dataset' in d else 0)" 2>/dev/null || echo "0")

if [ "$MARKET_BARS" -gt 10000000 ]; then
    echo -e "${GREEN}✓${NC} Market data: $MARKET_BARS bars (excellent)"
else
    echo -e "${YELLOW}⚠${NC} Market data: $MARKET_BARS bars (backfilling...)"
fi

# 6. Verify ML models loaded
echo -e "${BLUE}→${NC} Checking ML models..."
ML_HEALTH=$(curl -s http://localhost:8001/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "unknown")

if [ "$ML_HEALTH" = "healthy" ]; then
    echo -e "${GREEN}✓${NC} ML service: healthy with models loaded"
else
    echo -e "${YELLOW}⚠${NC} ML service: $ML_HEALTH"
fi

# 7. Verify trading services ready
echo -e "${BLUE}→${NC} Checking trading services..."
SERVICES_READY=0
SERVICES_TOTAL=4

for port in 8003 8004 8005 8006; do
    STATUS=$(curl -s http://localhost:$port/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "ok" ] || [ "$STATUS" = "healthy" ]; then
        SERVICES_READY=$((SERVICES_READY + 1))
    fi
done

if [ "$SERVICES_READY" -eq "$SERVICES_TOTAL" ]; then
    echo -e "${GREEN}✓${NC} Trading services: $SERVICES_READY/$SERVICES_TOTAL ready"
else
    echo -e "${YELLOW}⚠${NC} Trading services: $SERVICES_READY/$SERVICES_TOTAL ready"
fi

# 8. Calculate overall readiness
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  READINESS ASSESSMENT${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

CHECKS_PASSED=0
CHECKS_TOTAL=7

[ "$WATCHLIST_COUNT" -ge 300 ] && CHECKS_PASSED=$((CHECKS_PASSED + 1))
[ "$UNHEALTHY" -eq 0 ] && CHECKS_PASSED=$((CHECKS_PASSED + 1))
[ "$QUESTDB_CPU_INT" -gt 10 ] && CHECKS_PASSED=$((CHECKS_PASSED + 1))
[ "$BACKFILL_COUNT" -ge "$WATCHLIST_COUNT" ] && CHECKS_PASSED=$((CHECKS_PASSED + 1))
[ "$MARKET_BARS" -gt 10000000 ] && CHECKS_PASSED=$((CHECKS_PASSED + 1))
[ "$ML_HEALTH" = "healthy" ] && CHECKS_PASSED=$((CHECKS_PASSED + 1))
[ "$SERVICES_READY" -eq "$SERVICES_TOTAL" ] && CHECKS_PASSED=$((CHECKS_PASSED + 1))

READINESS=$((CHECKS_PASSED * 100 / CHECKS_TOTAL))

echo -e "  Watchlist:        ${GREEN}$WATCHLIST_COUNT${NC} optionable symbols"
echo -e "  Services:         ${GREEN}$SERVICES_READY/$SERVICES_TOTAL${NC} ready"
echo -e "  Data Volume:      ${GREEN}$MARKET_BARS${NC} market bars"
echo -e "  Backfill Queue:   ${GREEN}$BACKFILL_COUNT${NC} symbols"
echo -e "  ML Models:        ${GREEN}$ML_HEALTH${NC}"
echo ""
echo -e "  ${BLUE}Trading Readiness:${NC} ${GREEN}${READINESS}%${NC} ($CHECKS_PASSED/$CHECKS_TOTAL checks)"
echo ""

if [ "$READINESS" -ge 85 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ SYSTEM READY FOR LIVE TRADING                         ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
elif [ "$READINESS" -ge 70 ]; then
    echo -e "${YELLOW}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  ⚠ SYSTEM OPERATIONAL - MINOR OPTIMIZATIONS NEEDED       ║${NC}"
    echo -e "${YELLOW}╚═══════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ✗ SYSTEM NEEDS ATTENTION                                 ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
fi

echo ""

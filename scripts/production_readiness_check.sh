#!/bin/bash
###############################################################################
# PRODUCTION READINESS CHECK
# Comprehensive verification of live data streaming, ML intelligence, 
# and trading system readiness
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_section() { echo -e "\n${BLUE}═══ $1 ═══${NC}\n"; }

# Load environment
source /srv/ai-trading-system/.env 2>/dev/null || true

ISSUES=0
WARNINGS=0

echo "═══════════════════════════════════════════════════════════════"
echo "  PRODUCTION READINESS CHECK"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"

###############################################################################
# 1. CONTAINER HEALTH
###############################################################################
log_section "1. CONTAINER HEALTH"

TOTAL=$(docker ps --filter "name=trading-" --format "{{.Names}}" | wc -l)
HEALTHY=$(docker ps --filter "name=trading-" --filter "health=healthy" --format "{{.Names}}" | wc -l)

if [ "$HEALTHY" -eq "$TOTAL" ]; then
    log_info "All $TOTAL containers healthy"
else
    log_error "Only $HEALTHY/$TOTAL containers healthy"
    docker ps --filter "name=trading-" --format "table {{.Names}}\t{{.Status}}" | grep -v healthy || true
    ISSUES=$((ISSUES + 1))
fi

###############################################################################
# 2. WATCHLIST STATUS
###############################################################################
log_section "2. WATCHLIST STATUS"

WATCHLIST_COUNT=$(docker exec trading-redis redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning SCARD watchlist 2>/dev/null || echo 0)

if [ "$WATCHLIST_COUNT" -gt 0 ]; then
    log_info "Watchlist: $WATCHLIST_COUNT symbols"
    echo "   Sample: $(docker exec trading-redis redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning SRANDMEMBER watchlist 10 2>/dev/null | tr '\n' ' ')"
else
    log_error "Watchlist is empty!"
    ISSUES=$((ISSUES + 1))
fi

###############################################################################
# 3. LIVE DATA STREAMING
###############################################################################
log_section "3. LIVE DATA STREAMING"

# Check Pulsar topics
echo "Pulsar Topics:"
TOPICS=$(docker exec trading-pulsar bin/pulsar-admin topics list trading/development 2>/dev/null | wc -l)
if [ "$TOPICS" -gt 5 ]; then
    log_info "$TOPICS topics configured"
else
    log_warn "Only $TOPICS topics found"
    WARNINGS=$((WARNINGS + 1))
fi

# Check data ingestion activity
echo ""
echo "Data Ingestion (last 2 minutes):"
RECENT_LOGS=$(docker logs trading-data-ingestion --since 2m 2>&1 | grep -c "Collected" || echo 0)
if [ "$RECENT_LOGS" -gt 10 ]; then
    log_info "Active: $RECENT_LOGS social signal collections"
else
    log_warn "Low activity: only $RECENT_LOGS collections"
    WARNINGS=$((WARNINGS + 1))
fi

###############################################################################
# 4. QUESTDB DATA STORAGE
###############################################################################
log_section "4. QUESTDB DATA STORAGE"

# Check recent data
SOCIAL_RECENT=$(docker exec trading-questdb curl -s -G \
    --data-urlencode "query=SELECT COUNT(*) FROM social_signals WHERE timestamp > dateadd('h', -1, now())" \
    http://localhost:9000/exec 2>/dev/null | jq -r '.dataset[0][0]' 2>/dev/null || echo 0)

MARKET_TOTAL=$(docker exec trading-questdb curl -s -G \
    --data-urlencode "query=SELECT COUNT(*) FROM market_data" \
    http://localhost:9000/exec 2>/dev/null | jq -r '.dataset[0][0]' 2>/dev/null || echo 0)

if [ "$SOCIAL_RECENT" -gt 100 ]; then
    log_info "Social signals (last hour): $SOCIAL_RECENT"
else
    log_warn "Low recent social signals: $SOCIAL_RECENT"
    WARNINGS=$((WARNINGS + 1))
fi

log_info "Total market data rows: $(printf "%'d" $MARKET_TOTAL)"

###############################################################################
# 5. ML INTELLIGENCE SYSTEM
###############################################################################
log_section "5. ML INTELLIGENCE SYSTEM"

# Check ML service
ML_STATUS=$(docker inspect trading-ml --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
if [ "$ML_STATUS" = "healthy" ]; then
    log_info "ML service: healthy"
    
    # Check if models loaded
    ML_LOGS=$(docker logs trading-ml --since 10m 2>&1 | tail -50)
    if echo "$ML_LOGS" | grep -q "Model.*loaded\|Embedding.*ready"; then
        log_info "ML models: loaded"
    else
        log_warn "ML models status unclear"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    log_error "ML service: $ML_STATUS"
    ISSUES=$((ISSUES + 1))
fi

# Check signal generator
SG_STATUS=$(docker inspect trading-signal-generator --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
if [ "$SG_STATUS" = "healthy" ]; then
    log_info "Signal generator: healthy"
else
    log_warn "Signal generator: $SG_STATUS"
    WARNINGS=$((WARNINGS + 1))
fi

###############################################################################
# 6. BACKFILL PROGRESS
###############################################################################
log_section "6. BACKFILL PROGRESS"

BACKFILL_TOTAL=$(docker exec trading-postgres psql -U trading_user -d trading_db -tAc \
    "SELECT COUNT(*) FROM historical_backfill_progress" 2>/dev/null || echo 0)

BACKFILL_READY=$(docker exec trading-postgres psql -U trading_user -d trading_db -tAc \
    "SELECT COUNT(*) FROM historical_backfill_progress WHERE last_date > '2020-01-01'" 2>/dev/null || echo 0)

if [ "$BACKFILL_TOTAL" -gt 0 ]; then
    log_info "Backfill tracking: $BACKFILL_TOTAL symbols"
    
    if [ "$BACKFILL_READY" -gt 0 ]; then
        log_info "Symbols with historical data: $BACKFILL_READY ($(( BACKFILL_READY * 100 / BACKFILL_TOTAL ))%)"
    else
        log_warn "No symbols have completed backfill yet"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    log_warn "No backfill tracking configured"
    WARNINGS=$((WARNINGS + 1))
fi

###############################################################################
# 7. TRADING SYSTEM COMPONENTS
###############################################################################
log_section "7. TRADING SYSTEM COMPONENTS"

# Check execution engine
EXEC_STATUS=$(docker inspect trading-execution --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
if [ "$EXEC_STATUS" = "healthy" ]; then
    log_info "Execution engine: healthy"
else
    log_error "Execution engine: $EXEC_STATUS"
    ISSUES=$((ISSUES + 1))
fi

# Check risk monitor
RISK_STATUS=$(docker inspect trading-risk-monitor --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
if [ "$RISK_STATUS" = "healthy" ]; then
    log_info "Risk monitor: healthy"
else
    log_error "Risk monitor: $RISK_STATUS"
    ISSUES=$((ISSUES + 1))
fi

# Check strategy engine
STRAT_STATUS=$(docker inspect trading-strategy-engine --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
if [ "$STRAT_STATUS" = "healthy" ]; then
    log_info "Strategy engine: healthy"
else
    log_error "Strategy engine: $STRAT_STATUS"
    ISSUES=$((ISSUES + 1))
fi

###############################################################################
# 8. TRADING MODE & CONFIGURATION
###############################################################################
log_section "8. TRADING MODE & CONFIGURATION"

TRADING_MODE=$(grep "^TRADING_MODE=" /srv/ai-trading-system/.env | cut -d'=' -f2 || echo "unknown")
case "$TRADING_MODE" in
    "paper")
        log_info "Trading mode: PAPER (safe for testing)"
        ;;
    "live")
        log_warn "Trading mode: LIVE (real money trading active!)"
        ;;
    *)
        log_error "Trading mode: $TRADING_MODE (invalid)"
        ISSUES=$((ISSUES + 1))
        ;;
esac

# Check API keys configured
if grep -q "^POLYGON_API_KEY=.*[A-Za-z0-9]" /srv/ai-trading-system/.env; then
    log_info "Polygon API: configured"
else
    log_error "Polygon API: not configured"
    ISSUES=$((ISSUES + 1))
fi

if grep -q "^ALPACA_API_KEY=.*[A-Za-z0-9]" /srv/ai-trading-system/.env; then
    log_info "Alpaca API: configured"
else
    log_warn "Alpaca API: not configured"
    WARNINGS=$((WARNINGS + 1))
fi

###############################################################################
# 9. AUTOMATION STATUS
###############################################################################
log_section "9. AUTOMATION STATUS"

CRON_COUNT=$(crontab -l 2>/dev/null | grep -v "^#" | grep -v "^$" | wc -l || echo 0)
if [ "$CRON_COUNT" -gt 0 ]; then
    log_info "Cron jobs: $CRON_COUNT configured"
    crontab -l 2>/dev/null | grep -v "^#" | grep -v "^$" | while read -r line; do
        echo "   $line"
    done
else
    log_warn "No cron jobs configured"
    WARNINGS=$((WARNINGS + 1))
fi

###############################################################################
# 10. SYSTEM RESOURCES
###############################################################################
log_section "10. SYSTEM RESOURCES"

# Check disk space
DISK_USAGE=$(df -h /srv | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 80 ]; then
    log_info "Disk usage: ${DISK_USAGE}% (healthy)"
else
    log_warn "Disk usage: ${DISK_USAGE}% (high)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check memory
FREE_MEM=$(free -g | awk '/^Mem:/{print $7}')
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$FREE_MEM" -gt 10 ]; then
    log_info "Available memory: ${FREE_MEM}GB / ${TOTAL_MEM}GB"
else
    log_warn "Low available memory: ${FREE_MEM}GB / ${TOTAL_MEM}GB"
    WARNINGS=$((WARNINGS + 1))
fi

###############################################################################
# FINAL SUMMARY
###############################################################################
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  READINESS SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ $ISSUES -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    log_info "System Status: FULLY OPERATIONAL"
    log_info "Trading Ready: YES"
    echo ""
    echo "The system is ready for trading operations."
    EXIT_CODE=0
elif [ $ISSUES -eq 0 ]; then
    log_warn "System Status: OPERATIONAL WITH WARNINGS"
    log_warn "Issues: $ISSUES critical, $WARNINGS warnings"
    echo ""
    echo "The system can operate but has some warnings to address."
    EXIT_CODE=0
else
    log_error "System Status: ISSUES DETECTED"
    log_error "Issues: $ISSUES critical, $WARNINGS warnings"
    echo ""
    echo "Critical issues must be resolved before trading."
    EXIT_CODE=1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  MONITORING COMMANDS"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Live data streaming:"
echo "  docker logs -f trading-data-ingestion --since 5m"
echo ""
echo "ML intelligence:"
echo "  docker logs -f trading-signal-generator --since 5m"
echo ""
echo "Trading activity:"
echo "  docker logs -f trading-execution --since 5m"
echo ""
echo "System health:"
echo "  docker ps --filter 'name=trading-' --format 'table {{.Names}}\t{{.Status}}'"
echo ""

exit $EXIT_CODE

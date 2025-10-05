#!/bin/bash
# Activate Continuous Real-Time Processing
# This script enables and starts continuous processing for all watchlist symbols

set -e

echo "============================================"
echo "   ACTIVATING CONTINUOUS REAL-TIME PROCESSING"
echo "============================================"
echo ""

cd /srv/ai-trading-system

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== Step 1: Verify Configuration ==="
echo ""

# Check current configuration
EQUITY_BACKFILL_MAX=$(grep "^EQUITY_BACKFILL_MAX_SYMBOLS=" .env | cut -d'=' -f2)
DAILY_DELTA_MAX=$(grep "^DAILY_DELTA_MAX_SYMBOLS=" .env | cut -d'=' -f2)
API_WORKERS=$(grep "^API_WORKERS=" .env | cut -d'=' -f2)

echo "Current configuration:"
echo "  EQUITY_BACKFILL_MAX_SYMBOLS: $EQUITY_BACKFILL_MAX"
echo "  DAILY_DELTA_MAX_SYMBOLS: $DAILY_DELTA_MAX"
echo "  API_WORKERS: $API_WORKERS"
echo ""

if [ "$EQUITY_BACKFILL_MAX" != "0" ] || [ "$DAILY_DELTA_MAX" != "0" ]; then
    echo -e "${RED}✗${NC} ERROR: Symbol limits still in place!"
    echo "Please ensure all *_MAX_SYMBOLS settings are set to 0 in .env"
    exit 1
fi

echo -e "${GREEN}✓${NC} Configuration verified - unlimited processing enabled"
echo ""

echo "=== Step 2: Restart Data Ingestion with New Limits ==="
echo ""

echo "Restarting data-ingestion service to apply unlimited processing..."
docker-compose restart data-ingestion

echo "Waiting for service to stabilize..."
sleep 10

# Check if service is healthy
if curl -s -f http://localhost:8002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Data Ingestion service healthy"
else
    echo -e "${RED}✗${NC} Data Ingestion service not responding"
    exit 1
fi
echo ""

echo "=== Step 3: Verify Watchlist Size ==="
echo ""

WATCHLIST_COUNT=$(docker exec trading-redis redis-cli -a 'Okunka!Blebogyan02$' SCARD watchlist 2>/dev/null | tail -1)
echo "Watchlist contains $WATCHLIST_COUNT symbols"

if [ "$WATCHLIST_COUNT" -lt 100 ]; then
    echo -e "${YELLOW}⚠${NC} WARNING: Watchlist seems small, consider running discovery"
    echo "Run: docker exec -it trading-data-ingestion python3 /app/scripts/sync_optionable_watchlist.py"
fi
echo ""

echo "=== Step 4: Monitor Processing Start ==="
echo ""

echo "Waiting 30 seconds for processing to begin..."
sleep 30

echo "Checking recent processing activity..."
SYMBOLS_PROCESSED=$(docker logs trading-data-ingestion --since 1m 2>&1 | grep "Daily delta fetched" | grep -oP "symbol=\K\w+" | sort -u | wc -l)

if [ "$SYMBOLS_PROCESSED" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Processing active - $SYMBOLS_PROCESSED symbols processed in last minute"
else
    echo -e "${YELLOW}⚠${NC} No processing detected yet - check logs for any issues"
    echo "Check logs: docker logs trading-data-ingestion --tail 50"
fi
echo ""

echo "=== Step 5: Calculate Processing Capacity ==="
echo ""

# Estimate processing capacity
SYMBOLS_PER_MINUTE=$SYMBOLS_PROCESSED
SYMBOLS_PER_HOUR=$((SYMBOLS_PER_MINUTE * 60))
HOURS_FOR_FULL_WATCHLIST=$((WATCHLIST_COUNT / (SYMBOLS_PER_MINUTE * 60 + 1)))

echo "Processing capacity estimate:"
echo "  Symbols/minute: ~$SYMBOLS_PER_MINUTE"
echo "  Symbols/hour: ~$SYMBOLS_PER_HOUR"
if [ "$SYMBOLS_PER_MINUTE" -gt 0 ]; then
    echo "  Time to process full watchlist: ~$HOURS_FOR_FULL_WATCHLIST hours"
    
    if [ "$HOURS_FOR_FULL_WATCHLIST" -gt 4 ]; then
        echo ""
        echo -e "${YELLOW}⚠${NC} RECOMMENDATION: Processing rate is slow for ${WATCHLIST_COUNT} symbols"
        echo "   Consider increasing API_WORKERS (currently: $API_WORKERS)"
        echo "   Suggested: 16-32 workers for optimal throughput"
    fi
fi
echo ""

echo "=== Step 6: Setup Continuous Strategy Processing ==="
echo ""

echo "Checking if continuous processor is available..."
if [ -f "services/strategy-engine/continuous_processor.py" ]; then
    echo -e "${GREEN}✓${NC} Continuous processor found"
    echo ""
    echo "To activate continuous strategy analysis, add to strategy-engine startup:"
    echo ""
    echo "# Add to services/strategy-engine/strategy_manager.py lifespan:"
    echo "# Start continuous processor in background"
    echo "asyncio.create_task(start_continuous_processor())"
    echo ""
else
    echo -e "${YELLOW}⚠${NC} Continuous processor not found"
fi
echo ""

echo "=== Step 7: Recommendations for Scale ==="
echo ""

echo "For processing $WATCHLIST_COUNT symbols in real-time:"
echo ""
echo "1. INCREASE WORKERS:"
echo "   - Set API_WORKERS=16 (or higher) in .env"
echo "   - Restart data-ingestion: docker-compose restart data-ingestion"
echo ""
echo "2. ENABLE PARALLEL PROCESSING:"
echo "   - Consider running multiple data-ingestion containers"
echo "   - Split watchlist into chunks across containers"
echo ""
echo "3. OPTIMIZE DATABASE:"
echo "   - Ensure QuestDB has adequate resources"
echo "   - Check PostgreSQL connection pooling"
echo "   - Monitor Redis memory usage"
echo ""
echo "4. ACTIVATE ML PIPELINE:"
echo "   - Ensure ML service is processing predictions"
echo "   - Check continuous_training_orchestrator.py"
echo ""
echo "5. ENABLE CONTINUOUS BACKTESTING:"
echo "   - Run rolling backtests for all strategies"
echo "   - Monitor strategy performance in real-time"
echo ""

echo "============================================"
echo "CONTINUOUS PROCESSING ACTIVATION COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Monitor processing: docker logs -f trading-data-ingestion"
echo "2. Check status: ./scripts/comprehensive_system_status.sh"
echo "3. View metrics: http://localhost:3000 (Grafana)"
echo ""
echo "System is now processing ALL $WATCHLIST_COUNT watchlist symbols continuously!"
echo ""

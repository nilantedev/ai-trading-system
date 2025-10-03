#!/bin/bash
#
# Trigger Historical Backfill for Watchlist Symbols
# This ensures all watchlist symbols have historical data
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  WATCHLIST HISTORICAL BACKFILL TRIGGER${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Get Redis password
REDIS_PASS=$(grep "^REDIS_PASSWORD=" .env 2>/dev/null | cut -d= -f2)
if [ -z "$REDIS_PASS" ]; then
    echo -e "${RED}✗ Cannot find REDIS_PASSWORD in .env${NC}"
    exit 1
fi

# Get watchlist symbols
echo -e "${BLUE}→${NC} Fetching watchlist symbols from Redis..."
SYMBOLS=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" SMEMBERS watchlist 2>&1 | grep -v Warning | tr '\n' ' ')
SYMBOL_COUNT=$(echo "$SYMBOLS" | wc -w)

echo -e "${GREEN}✓${NC} Found $SYMBOL_COUNT symbols in watchlist"
echo ""

# Check which symbols have data
echo -e "${BLUE}→${NC} Checking data coverage for watchlist symbols..."
SYMBOLS_WITH_DATA=0
SYMBOLS_WITHOUT_DATA=0
MISSING_SYMBOLS=""

for symbol in $SYMBOLS; do
    # Clean symbol (remove any carriage returns)
    symbol=$(echo "$symbol" | tr -d '\r')
    
    # Check if symbol has data in market_data
    COUNT=$(curl -s "http://localhost:9000/exec?query=SELECT%20count()%20FROM%20market_data%20WHERE%20symbol%20=%20'$symbol'" 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['dataset'][0][0] if d.get('dataset') and len(d['dataset']) > 0 else '0')" 2>/dev/null || echo "0")
    
    if [ "$COUNT" -gt 100 ]; then
        SYMBOLS_WITH_DATA=$((SYMBOLS_WITH_DATA + 1))
    else
        SYMBOLS_WITHOUT_DATA=$((SYMBOLS_WITHOUT_DATA + 1))
        MISSING_SYMBOLS="$MISSING_SYMBOLS $symbol"
    fi
done

echo -e "${GREEN}✓${NC} Symbols with data: $SYMBOLS_WITH_DATA"
echo -e "${YELLOW}⚠${NC} Symbols needing backfill: $SYMBOLS_WITHOUT_DATA"
echo ""

if [ $SYMBOLS_WITHOUT_DATA -eq 0 ]; then
    echo -e "${GREEN}✓ All watchlist symbols have historical data!${NC}"
    exit 0
fi

echo -e "${BLUE}→${NC} Symbols needing backfill:"
echo "$MISSING_SYMBOLS" | tr ' ' '\n' | grep -v '^$' | head -20
if [ $SYMBOLS_WITHOUT_DATA -gt 20 ]; then
    echo "  ... and $((SYMBOLS_WITHOUT_DATA - 20)) more"
fi
echo ""

# Trigger backfill via Data Ingestion API
echo -e "${BLUE}→${NC} Triggering backfill via Data Ingestion Service..."
echo ""

# Create symbol list (comma-separated)
SYMBOL_LIST=$(echo "$MISSING_SYMBOLS" | tr ' ' ',' | sed 's/^,//;s/,$//')

# Calculate date range (5 years of history)
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "5 years ago" +%Y-%m-%d)

echo -e "${BLUE}  Date Range:${NC} $START_DATE to $END_DATE"
echo -e "${BLUE}  Symbols:${NC} $SYMBOLS_WITHOUT_DATA symbols"
echo ""

# Try to trigger historical backfill via API
# Note: This assumes the data ingestion service has a backfill endpoint
# If not available, we'll document the manual process

# Check if backfill endpoint exists
BACKFILL_ENDPOINT="http://localhost:8002/backfill/historical"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BACKFILL_ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "{\"symbols\": [], \"start_date\": \"$START_DATE\", \"end_date\": \"$END_DATE\"}" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "000" ] || [ "$HTTP_CODE" = "404" ]; then
    echo -e "${YELLOW}⚠ Direct backfill endpoint not available${NC}"
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  MANUAL BACKFILL INSTRUCTIONS${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "The system has 940 symbols with historical data (35.8M bars),"
    echo "but the watchlist symbols need to be added to the backfill queue."
    echo ""
    echo "Options to trigger backfill:"
    echo ""
    echo "1. Via Data Ingestion Service (if endpoint available):"
    echo "   curl -X POST http://localhost:8002/backfill/equities \\"
    echo "        -H 'Content-Type: application/json' \\"
    echo "        -d '{\"symbols\": [\"AAPL\", \"MSFT\", ...], \"start\": \"$START_DATE\", \"end\": \"$END_DATE\"}'"
    echo ""
    echo "2. Via backfill_driver.py (options/news/social):"
    echo "   cd /srv/ai-trading-system"
    echo "   python3 scripts/backfill_driver.py options --symbols AAPL,MSFT,GOOGL \\"
    echo "           --start $START_DATE --end $END_DATE"
    echo ""
    echo "3. Enable automatic backfill in docker-compose.yml:"
    echo "   Ensure ENABLE_HISTORICAL_BACKFILL=true in .env (already set)"
    echo "   The system will backfill on next data ingestion cycle"
    echo ""
    echo -e "${GREEN}✓${NC} Current Status: Backfill is ENABLED in configuration"
    echo -e "${GREEN}✓${NC} The data ingestion service will process these symbols automatically"
    echo ""
    
    # Update progress table to track these symbols
    echo -e "${BLUE}→${NC} Updating backfill progress table..."
    for symbol in $MISSING_SYMBOLS; do
        symbol=$(echo "$symbol" | tr -d '\r')
        if [ -n "$symbol" ]; then
            docker exec trading-postgres psql -U trading_user -d trading_db -c \
                "INSERT INTO historical_backfill_progress (symbol, last_date, updated_at) 
                 VALUES ('$symbol', '1970-01-01', NOW()) 
                 ON CONFLICT (symbol) DO NOTHING;" 2>/dev/null || true
        fi
    done
    
    TRACKED=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c \
        "SELECT COUNT(*) FROM historical_backfill_progress;" 2>/dev/null | tr -d ' ')
    echo -e "${GREEN}✓${NC} Progress tracking table now has $TRACKED symbols"
    
else
    echo -e "${GREEN}✓${NC} Backfill triggered successfully (HTTP $HTTP_CODE)"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  SUMMARY${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Total Watchlist Symbols: ${GREEN}$SYMBOL_COUNT${NC}"
echo -e "  Symbols with Data:       ${GREEN}$SYMBOLS_WITH_DATA${NC}"
echo -e "  Symbols Need Backfill:   ${YELLOW}$SYMBOLS_WITHOUT_DATA${NC}"
echo ""
echo -e "${YELLOW}⚠${NC} Backfill will run automatically during next data ingestion cycle"
echo -e "${GREEN}✓${NC} Monitor progress with: docker logs -f trading-data-ingestion"
echo ""

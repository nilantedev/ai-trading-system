#!/bin/bash
#
# Production Watchlist Manager
# Single source of truth for watchlist management
# Uses optionable_symbols_discovered.json from Polygon options discovery
# No external dependencies - pure bash/docker
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get Redis password
REDIS_PASS=$(grep "^REDIS_PASSWORD=" .env 2>/dev/null | cut -d= -f2)
if [ -z "$REDIS_PASS" ]; then
    echo -e "${RED}✗ Cannot find REDIS_PASSWORD in .env${NC}"
    exit 1
fi

# Create logs directory
mkdir -p logs

LOG_FILE="logs/watchlist_sync_$(date +%Y%m%d_%H%M%S).log"

{
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  PRODUCTION WATCHLIST MANAGER - $(date)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Get current watchlist count
    CURRENT_COUNT=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" SCARD watchlist 2>&1 | grep -v Warning | tr -d ' \r')
    echo -e "${BLUE}→${NC} Current watchlist: $CURRENT_COUNT symbols"
    
    # Extract optionable symbols from data-ingestion container
    echo -e "${BLUE}→${NC} Discovering optionable symbols from coverage data..."
    
    TEMP_SYMBOLS="/tmp/optionable_symbols_$$.txt"
    
    docker exec trading-data-ingestion python3 << 'PYTHON_END' > "$TEMP_SYMBOLS" 2>&1
import json
import sys

try:
    # Use the discovered symbols from Polygon options discovery
    with open('/app/export/grafana-csv/optionable_symbols_discovered.json', 'r') as f:
        data = json.load(f)
    
    symbols = data.get('symbols', [])
    
    # Output sorted unique symbols
    for symbol in sorted(symbols):
        print(symbol)
    
    print(f"TOTAL_COUNT:{len(symbols)}", file=sys.stderr)
    
except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    sys.exit(1)
PYTHON_END
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to extract optionable symbols${NC}"
        rm -f "$TEMP_SYMBOLS"
        exit 1
    fi
    
    # Count discovered symbols
    DISCOVERED_COUNT=$(grep -v "^TOTAL_COUNT:" "$TEMP_SYMBOLS" | grep -v "^ERROR:" | wc -l)
    
    if [ "$DISCOVERED_COUNT" -lt 100 ]; then
        echo -e "${YELLOW}⚠${NC} Warning: Only $DISCOVERED_COUNT symbols found (expected 300+)"
        echo -e "${YELLOW}⚠${NC} Keeping existing watchlist to prevent data loss"
        rm -f "$TEMP_SYMBOLS"
        exit 0
    fi
    
    echo -e "${GREEN}✓${NC} Discovered $DISCOVERED_COUNT optionable symbols"
    
    # Show sample
    echo -e "${BLUE}→${NC} Sample symbols:"
    head -20 "$TEMP_SYMBOLS" | pr -t -4
    echo ""
    
    # Sync to Redis
    echo -e "${BLUE}→${NC} Synchronizing watchlist..."
    docker exec trading-redis redis-cli -a "$REDIS_PASS" DEL watchlist 2>&1 | grep -v Warning > /dev/null
    
    ADDED=0
    while IFS= read -r symbol; do
        [[ -z "$symbol" ]] && continue
        symbol=$(echo "$symbol" | tr -d ' \r\n')
        [[ -z "$symbol" ]] && continue
        
        docker exec trading-redis redis-cli -a "$REDIS_PASS" SADD watchlist "$symbol" 2>&1 | grep -v Warning > /dev/null
        ADDED=$((ADDED + 1))
    done < "$TEMP_SYMBOLS"
    
    FINAL_COUNT=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" SCARD watchlist 2>&1 | grep -v Warning | tr -d ' \r')
    
    # Update backfill queue
    echo -e "${BLUE}→${NC} Updating backfill queue..."
    
    ADDED_TO_BACKFILL=0
    while IFS= read -r symbol; do
        [[ -z "$symbol" ]] && continue
        symbol=$(echo "$symbol" | tr -d ' \r\n')
        [[ -z "$symbol" ]] && continue
        
        docker exec trading-postgres psql -U trading_user -d trading_db -c \
            "INSERT INTO historical_backfill_progress (symbol, last_date, updated_at) 
             VALUES ('$symbol', '1970-01-01', NOW()) 
             ON CONFLICT (symbol) DO NOTHING;" > /dev/null 2>&1 && ADDED_TO_BACKFILL=$((ADDED_TO_BACKFILL + 1))
    done < "$TEMP_SYMBOLS"
    
    BACKFILL_TOTAL=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c \
        "SELECT COUNT(*) FROM historical_backfill_progress;" 2>/dev/null | tr -d ' ')
    
    rm -f "$TEMP_SYMBOLS"
    
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  SYNC COMPLETE${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  Previous:         ${YELLOW}$CURRENT_COUNT${NC} symbols"
    echo -e "  Discovered:       ${GREEN}$DISCOVERED_COUNT${NC} optionable symbols"
    echo -e "  Current:          ${GREEN}$FINAL_COUNT${NC} symbols (unique)"
    echo -e "  Backfill Queue:   ${GREEN}$BACKFILL_TOTAL${NC} symbols"
    echo -e "  New in Backfill:  ${GREEN}$ADDED_TO_BACKFILL${NC} symbols"
    echo ""
    
    if [ "$FINAL_COUNT" -eq "$DISCOVERED_COUNT" ]; then
        echo -e "${GREEN}✓ Watchlist successfully synchronized${NC}"
    else
        echo -e "${YELLOW}⚠${NC} Count mismatch (expected $DISCOVERED_COUNT, got $FINAL_COUNT)"
    fi
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ WATCHLIST MANAGEMENT COMPLETE                          ║${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}║  ${FINAL_COUNT} unique optionable symbols in watchlist${NC}"
    echo -e "${GREEN}║  ${BACKFILL_TOTAL} symbols queued for backfill${NC}"
    echo -e "${GREEN}║  System automatically discovers and maintains list        ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
} | tee "$LOG_FILE"

# Keep only last 30 days of logs
find logs/ -name "watchlist_sync_*.log" -mtime +30 -delete 2>/dev/null || true

echo "Log saved: $LOG_FILE"

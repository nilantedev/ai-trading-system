#!/bin/bash
#
# Update Backfill Queue from Current Watchlist
# Ensures all watchlist symbols are queued for backfill
#

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}→${NC} Syncing backfill queue with watchlist..."

# Get Redis password
REDIS_PASS=$(grep "^REDIS_PASSWORD=" .env | cut -d= -f2)

# Get all watchlist symbols
SYMBOLS=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" SMEMBERS watchlist 2>&1 | grep -v Warning)

ADDED=0
while IFS= read -r symbol; do
    [[ -z "$symbol" ]] && continue
    symbol=$(echo "$symbol" | tr -d ' \r\n')
    [[ -z "$symbol" ]] && continue
    
    docker exec trading-postgres psql -U trading_user -d trading_db -c \
        "INSERT INTO historical_backfill_progress (symbol, last_date, updated_at) 
         VALUES ('$symbol', '1970-01-01', NOW()) 
         ON CONFLICT (symbol) DO NOTHING;" > /dev/null 2>&1 && ADDED=$((ADDED + 1))
done <<< "$SYMBOLS"

TOTAL=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c \
    "SELECT COUNT(*) FROM historical_backfill_progress;" 2>/dev/null | tr -d ' ')

echo -e "${GREEN}✓${NC} Backfill queue updated: $TOTAL symbols ($ADDED new)"

#!/bin/bash
# Quick Sync & Aggressive Backfill

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  QUICK WATCHLIST SYNC & AGGRESSIVE BACKFILL"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Load environment
source /srv/ai-trading-system/.env 2>/dev/null || true

# Get current count
CURRENT=$(docker exec trading-redis redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning SCARD watchlist 2>/dev/null)
echo "Current watchlist: $CURRENT symbols"
echo ""

# Run quick tickers-based discovery with sync
echo "Running quick symbol discovery (tickers endpoint)..."
echo "This uses the fast tickers API instead of contracts..."
echo ""

docker exec trading-data-ingestion python3 << 'EOF'
import os
import redis
import requests
from datetime import datetime

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
REDIS_HOST = os.getenv('REDIS_HOST', 'trading-redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

print("Connecting to Polygon API...")
symbols = set()

url = "https://api.polygon.io/v3/reference/tickers"
params = {
    'market': 'stocks',
    'active': 'true',
    'limit': 1000,
    'apiKey': POLYGON_API_KEY
}

page = 0
max_pages = 15  # About 15k symbols, will get most optionable ones

while page < max_pages:
    page += 1
    response = requests.get(url, params=params, timeout=30)
    
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        break
    
    data = response.json()
    results = data.get('results', [])
    
    if not results:
        break
    
    # Filter for likely optionable stocks
    for ticker in results:
        symbol = ticker.get('ticker', '')
        # Most optionable: 1-5 letter, all uppercase, alpha only
        if symbol and 1 <= len(symbol) <= 5 and symbol.isalpha() and symbol.isupper():
            symbols.add(symbol)
    
    if page % 5 == 0:
        print(f"Page {page}: {len(symbols)} symbols collected")
    
    # Get next page
    next_url = data.get('next_url')
    if not next_url or 'cursor=' not in next_url:
        break
    
    cursor = next_url.split('cursor=')[1].split('&')[0]
    params['cursor'] = cursor

print(f"\nTotal symbols discovered: {len(symbols)}")

# Sync to Redis
print("\nSyncing to Redis...")
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)

if symbols:
    r.delete('watchlist')
    r.sadd('watchlist', *symbols)
    final_count = r.scard('watchlist')
    print(f"Watchlist updated: {final_count} symbols")
else:
    print("No symbols to sync!")

EOF

# Get updated count
UPDATED=$(docker exec trading-redis redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning SCARD watchlist 2>/dev/null)
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Previous: $CURRENT symbols"
echo "  Updated:  $UPDATED symbols"
echo "  Added:    $(($UPDATED - $CURRENT)) symbols"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Trigger aggressive backfill
echo "Triggering aggressive backfill for all symbols..."
echo ""

# Use the existing trigger script
bash /srv/ai-trading-system/scripts/trigger_watchlist_backfill.sh

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SYNC & BACKFILL INITIATED"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Monitor backfill progress:"
echo "  docker logs -f trading-data-ingestion --since 5m"
echo ""
echo "Check backfill status:"
echo "  bash /srv/ai-trading-system/scripts/check_backfill_status.sh"
echo ""

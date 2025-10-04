#!/bin/bash

###############################################################################
# Automated Watchlist Update Script
# Purpose: Discovers optionable symbols, syncs watchlist, triggers backfill
#          for new symbols, and cleans up removed symbols
# Schedule: Run daily via cron or systemd timer
###############################################################################

set -e

# Configuration
REDIS_PASSWORD="${REDIS_PASSWORD:-your_redis_password_here}"
LOG_FILE="/app/logs/watchlist_update_$(date +%Y%m%d_%H%M%S).log"
EXPORT_DIR="/mnt/fastdrive/trading/export/grafana-csv"
BACKUP_DIR="/mnt/bulkdata/trading/backups/watchlist"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

###############################################################################
# Step 1: Backup current watchlist
###############################################################################
log "Step 1: Backing up current watchlist..."
mkdir -p "$BACKUP_DIR"

BACKUP_FILE="$BACKUP_DIR/watchlist_backup_$(date +%Y%m%d_%H%M%S).txt"
docker exec trading-redis redis-cli -a "$REDIS_PASSWORD" SMEMBERS watchlist > "$BACKUP_FILE" 2>/dev/null || true
PREV_COUNT=$(wc -l < "$BACKUP_FILE" 2>/dev/null || echo 0)

log "Previous watchlist: $PREV_COUNT symbols backed up to $BACKUP_FILE"

###############################################################################
# Step 2: Run symbol discovery
###############################################################################
log "Step 2: Running optionable symbol discovery..."

docker exec trading-data-ingestion python3 /app/services/data_ingestion/options_symbol_discovery.py 2>&1 | tee -a "$LOG_FILE"

# Wait for discovery to complete
sleep 5

###############################################################################
# Step 3: Verify new watchlist
###############################################################################
log "Step 3: Verifying new watchlist..."

NEW_COUNT=$(docker exec trading-redis redis-cli -a "$REDIS_PASSWORD" SCARD watchlist 2>/dev/null || echo 0)
log "New watchlist: $NEW_COUNT symbols"

# Check export file
if [ -f "$EXPORT_DIR/optionable_symbols_discovered.json" ]; then
    EXPORT_COUNT=$(cat "$EXPORT_DIR/optionable_symbols_discovered.json" | jq -r '.count // 0' 2>/dev/null || echo 0)
    EXPORT_SOURCE=$(cat "$EXPORT_DIR/optionable_symbols_discovered.json" | jq -r '.source // "unknown"' 2>/dev/null || echo "unknown")
    log "Export file: $EXPORT_COUNT symbols from $EXPORT_SOURCE"
else
    warn "Export file not found at $EXPORT_DIR/optionable_symbols_discovered.json"
fi

###############################################################################
# Step 4: Detect changes (new/removed symbols)
###############################################################################
log "Step 4: Detecting watchlist changes..."

# Get current symbols
CURRENT_FILE="/tmp/watchlist_current_$(date +%s).txt"
docker exec trading-redis redis-cli -a "$REDIS_PASSWORD" SMEMBERS watchlist > "$CURRENT_FILE" 2>/dev/null || true

# Find new symbols (in current but not in backup)
NEW_SYMBOLS_FILE="/tmp/watchlist_new_$(date +%s).txt"
comm -23 <(sort "$CURRENT_FILE") <(sort "$BACKUP_FILE") > "$NEW_SYMBOLS_FILE"
NEW_SYMBOLS_COUNT=$(wc -l < "$NEW_SYMBOLS_FILE")

# Find removed symbols (in backup but not in current)
REMOVED_SYMBOLS_FILE="/tmp/watchlist_removed_$(date +%s).txt"
comm -13 <(sort "$CURRENT_FILE") <(sort "$BACKUP_FILE") > "$REMOVED_SYMBOLS_FILE"
REMOVED_SYMBOLS_COUNT=$(wc -l < "$REMOVED_SYMBOLS_FILE")

log "Changes detected: +$NEW_SYMBOLS_COUNT new, -$REMOVED_SYMBOLS_COUNT removed"

if [ $NEW_SYMBOLS_COUNT -gt 0 ]; then
    log "New symbols:"
    head -20 "$NEW_SYMBOLS_FILE" | tee -a "$LOG_FILE"
    if [ $NEW_SYMBOLS_COUNT -gt 20 ]; then
        log "... and $(($NEW_SYMBOLS_COUNT - 20)) more"
    fi
fi

if [ $REMOVED_SYMBOLS_COUNT -gt 0 ]; then
    warn "Removed symbols:"
    head -20 "$REMOVED_SYMBOLS_FILE" | tee -a "$LOG_FILE"
    if [ $REMOVED_SYMBOLS_COUNT -gt 20 ]; then
        warn "... and $(($REMOVED_SYMBOLS_COUNT - 20)) more"
    fi
fi

###############################################################################
# Step 5: Trigger backfill for new symbols
###############################################################################
if [ $NEW_SYMBOLS_COUNT -gt 0 ]; then
    log "Step 5: Triggering backfill for new symbols..."
    
    # Trigger equity backfill
    log "Triggering equity backfill..."
    docker exec trading-data-ingestion python3 -c "
from services.data_ingestion.equity_backfill import EquityBackfill
import asyncio

async def run():
    backfill = EquityBackfill()
    await backfill.backfill_all_symbols()
    
asyncio.run(run())
" 2>&1 | tee -a "$LOG_FILE" &
    
    # Trigger options backfill
    log "Triggering options backfill..."
    docker exec trading-data-ingestion python3 -c "
from services.data_ingestion.options_backfill import OptionsBackfill
import asyncio

async def run():
    backfill = OptionsBackfill()
    await backfill.backfill_all_symbols()
    
asyncio.run(run())
" 2>&1 | tee -a "$LOG_FILE" &
    
    log "Backfill processes started in background"
    
else
    log "Step 5: No new symbols, skipping backfill"
fi

###############################################################################
# Step 6: Clean up removed symbols (mark for retention cleanup)
###############################################################################
if [ $REMOVED_SYMBOLS_COUNT -gt 0 ]; then
    log "Step 6: Marking removed symbols for cleanup..."
    
    # Add to cleanup list in Redis
    while IFS= read -r symbol; do
        if [ -n "$symbol" ]; then
            docker exec trading-redis redis-cli -a "$REDIS_PASSWORD" SADD symbols_removed "$symbol" >/dev/null 2>&1
        fi
    done < "$REMOVED_SYMBOLS_FILE"
    
    log "Marked $REMOVED_SYMBOLS_COUNT symbols for retention cleanup"
    warn "Run retention policies to clean up historical data for removed symbols"
    
else
    log "Step 6: No removed symbols, skipping cleanup"
fi

###############################################################################
# Step 7: Update ML feature store
###############################################################################
log "Step 7: Triggering ML feature store update..."

docker exec trading-ml python3 -c "
from services.ml.feature_store import FeatureStore
import asyncio

async def run():
    feature_store = FeatureStore()
    await feature_store.sync_watchlist_features()
    
asyncio.run(run())
" 2>&1 | tee -a "$LOG_FILE" || warn "ML feature store update failed (may not be critical)"

###############################################################################
# Step 8: Verify system health
###############################################################################
log "Step 8: Verifying system health..."

# Check QuestDB row counts
QUESTDB_MARKET_DATA=$(docker exec trading-questdb curl -s -G \
    --data-urlencode "query=SELECT count(*) as cnt FROM market_data WHERE symbol IN (SELECT DISTINCT symbol FROM watchlist_symbols)" \
    http://localhost:9000/exec | jq -r '.dataset[0][0]' 2>/dev/null || echo 0)

log "QuestDB market_data: $QUESTDB_MARKET_DATA rows"

# Check PostgreSQL backfill tracking
POSTGRES_BACKFILL=$(docker exec trading-postgres psql -U postgres -d trading -tAc \
    "SELECT COUNT(DISTINCT symbol) FROM backfill_progress WHERE symbol IN (SELECT symbol FROM watchlist_symbols)" 2>/dev/null || echo 0)

log "PostgreSQL backfill tracking: $POSTGRES_BACKFILL symbols"

# Check Redis watchlist
REDIS_WATCHLIST=$(docker exec trading-redis redis-cli -a "$REDIS_PASSWORD" SCARD watchlist 2>/dev/null || echo 0)

log "Redis watchlist: $REDIS_WATCHLIST symbols"

###############################################################################
# Step 9: Generate summary report
###############################################################################
log "Step 9: Generating summary report..."

REPORT_FILE="$EXPORT_DIR/watchlist_update_report_$(date +%Y%m%d_%H%M%S).json"

cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "previous_count": $PREV_COUNT,
  "current_count": $NEW_COUNT,
  "new_symbols": $NEW_SYMBOLS_COUNT,
  "removed_symbols": $REMOVED_SYMBOLS_COUNT,
  "export_count": ${EXPORT_COUNT:-0},
  "export_source": "${EXPORT_SOURCE:-unknown}",
  "verification": {
    "redis_watchlist": $REDIS_WATCHLIST,
    "questdb_market_data": $QUESTDB_MARKET_DATA,
    "postgres_backfill": $POSTGRES_BACKFILL
  },
  "backfill_triggered": $([ $NEW_SYMBOLS_COUNT -gt 0 ] && echo "true" || echo "false"),
  "cleanup_marked": $([ $REMOVED_SYMBOLS_COUNT -gt 0 ] && echo "true" || echo "false"),
  "log_file": "$LOG_FILE",
  "backup_file": "$BACKUP_FILE"
}
EOF

log "Report saved to $REPORT_FILE"
cat "$REPORT_FILE" | jq '.' 2>/dev/null || cat "$REPORT_FILE"

# Cleanup temp files
rm -f "$CURRENT_FILE" "$NEW_SYMBOLS_FILE" "$REMOVED_SYMBOLS_FILE"

log "Watchlist update complete!"
log "Total change: $PREV_COUNT → $NEW_COUNT symbols ($(($NEW_COUNT - $PREV_COUNT)))"

#!/bin/bash
# Wrapper script for cron execution of symbol discovery

LOG_DIR="/mnt/fastdrive/trading/logs/symbol_discovery"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/discovery_$TIMESTAMP.log"

echo "=== Symbol Discovery Started at $(date) ===" | tee -a "$LOG_FILE"

# Run discovery in the data-ingestion container
docker exec trading-data-ingestion python /app/services/data_ingestion/options_symbol_discovery.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Symbol Discovery Completed Successfully at $(date) ===" | tee -a "$LOG_FILE"
    
    # Get symbol count from Redis
    SYMBOL_COUNT=$(docker exec trading-redis redis-cli -a 'quesrdbstrongpassword0910' --no-auth-warning SCARD watchlist 2>/dev/null || echo "N/A")
    echo "Total symbols in watchlist: $SYMBOL_COUNT" | tee -a "$LOG_FILE"
    
    # Keep only last 30 days of logs
    find "$LOG_DIR" -name "discovery_*.log" -mtime +30 -delete
else
    echo "=== Symbol Discovery Failed with exit code $EXIT_CODE at $(date) ===" | tee -a "$LOG_FILE"
fi

# Create symlink to latest log
ln -sf "$LOG_FILE" "$LOG_DIR/discovery_latest.log"

exit $EXIT_CODE

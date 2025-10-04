#!/bin/bash
# Monitor the options symbol discovery process
# Usage: ./monitor_discovery.sh [log_file]

LOG_FILE="${1:-/tmp/options_discovery_fixed.log}"

echo "Monitoring discovery process..."
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo ""

LAST_SIZE=0

while true; do
    if [ -f "$LOG_FILE" ]; then
        CURRENT_SIZE=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
        
        # Get last few lines
        echo "=== Discovery Progress ($(date +%T)) ==="
        tail -n 5 "$LOG_FILE" 2>/dev/null | grep -E "(page|Progress|symbols|Reached)" || echo "Waiting for updates..."
        echo ""
        
        # Check if process is still running
        if docker exec trading-data-ingestion pgrep -f "options_symbol_discovery" >/dev/null 2>&1; then
            echo "✓ Process is running"
        else
            echo "✗ Process completed or not running"
            echo ""
            echo "=== Final Results ==="
            tail -n 10 "$LOG_FILE"
            break
        fi
        
        LAST_SIZE=$CURRENT_SIZE
    else
        echo "Log file not found yet..."
    fi
    
    sleep 10
done

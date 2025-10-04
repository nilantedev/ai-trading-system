#!/bin/bash
# Automated Options Symbol Discovery Scheduler
# This script sets up a cron job or systemd timer to keep the watchlist updated

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Automated Symbol Discovery Scheduler ==="
echo ""

# Discovery script path
DISCOVERY_SCRIPT="$PROJECT_ROOT/services/data_ingestion/options_symbol_discovery.py"
LOG_DIR="/mnt/fastdrive/trading/logs/symbol_discovery"

# Create log directory
mkdir -p "$LOG_DIR"

# Create wrapper script for cron execution
WRAPPER_SCRIPT="$SCRIPT_DIR/run_symbol_discovery_cron.sh"

cat > "$WRAPPER_SCRIPT" << 'WRAPPER_EOF'
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
WRAPPER_EOF

chmod +x "$WRAPPER_SCRIPT"

echo "✓ Created wrapper script: $WRAPPER_SCRIPT"
echo ""

# Create systemd timer (if systemd is available)
if command -v systemctl &> /dev/null; then
    echo "Setting up systemd timer..."
    
    TIMER_FILE="/tmp/symbol-discovery.timer"
    SERVICE_FILE="/tmp/symbol-discovery.service"
    
    # Create service file
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Options Symbol Discovery Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=$WRAPPER_SCRIPT
User=$(whoami)
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Create timer file (runs weekly on Sunday at 2 AM)
    cat > "$TIMER_FILE" << EOF
[Unit]
Description=Weekly Options Symbol Discovery Timer
Requires=symbol-discovery.service

[Timer]
OnCalendar=Sun *-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

    echo ""
    echo "Systemd files created. To install, run:"
    echo "  sudo cp $SERVICE_FILE /etc/systemd/system/"
    echo "  sudo cp $TIMER_FILE /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable symbol-discovery.timer"
    echo "  sudo systemctl start symbol-discovery.timer"
    echo ""
fi

# Create crontab entry
echo "Setting up cron job..."
CRON_ENTRY="0 2 * * 0 $WRAPPER_SCRIPT"

# Check if cron entry already exists
if crontab -l 2>/dev/null | grep -F "$WRAPPER_SCRIPT" > /dev/null; then
    echo "⚠ Cron job already exists"
else
    # Add cron job (runs weekly on Sunday at 2 AM)
    (crontab -l 2>/dev/null; echo "# Options Symbol Discovery - Weekly on Sunday at 2 AM"; echo "$CRON_ENTRY") | crontab -
    echo "✓ Added cron job: $CRON_ENTRY"
fi

echo ""
echo "=== Scheduler Setup Complete ==="
echo ""
echo "Schedule: Weekly on Sunday at 2:00 AM"
echo "Logs: $LOG_DIR"
echo "Latest log: $LOG_DIR/discovery_latest.log"
echo ""
echo "To run discovery manually:"
echo "  $WRAPPER_SCRIPT"
echo ""
echo "To check cron status:"
echo "  crontab -l | grep symbol_discovery"
echo ""

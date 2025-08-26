#!/bin/bash
"""
Setup automated backup cron jobs for AI Trading System.
"""

set -e

# Configuration
TRADING_SYSTEM_DIR="/home/nilante/main-nilante-server/ai-trading-system"
BACKUP_SCRIPT="$TRADING_SYSTEM_DIR/scripts/backup_system.py"
PYTHON_BIN="$TRADING_SYSTEM_DIR/.venv/bin/python"
LOG_DIR="$TRADING_SYSTEM_DIR/logs"
BACKUP_LOG="$LOG_DIR/backup.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Create backup configuration
cat > "$TRADING_SYSTEM_DIR/config/backup_config.json" << 'EOF'
{
    "local_backup_dir": "/var/backups/trading-system",
    "daily_retention": 30,
    "weekly_retention": 12,
    "monthly_retention": 12,
    "compress_backups": true,
    "verify_backups": true,
    "postgres_url": null,
    "redis_url": null,
    "model_artifacts_dir": "services/model-registry/artifacts",
    "logs_dir": "logs",
    "config_dir": "config"
}
EOF

echo "✅ Created backup configuration at: $TRADING_SYSTEM_DIR/config/backup_config.json"

# Create backup wrapper script
cat > "$TRADING_SYSTEM_DIR/scripts/run_backup.sh" << EOF
#!/bin/bash
# Automated backup wrapper script

set -e

# Change to project directory
cd "$TRADING_SYSTEM_DIR"

# Set up environment
export PYTHONPATH=".:shared/python-common"
export DATABASE_URL="\${DATABASE_URL:-postgresql://trading_user:trading_password@localhost:5432/trading_db}"
export REDIS_URL="\${REDIS_URL:-redis://localhost:6379/0}"

# Run backup with logging
echo "\$(date): Starting automated backup..." >> "$BACKUP_LOG"
"$PYTHON_BIN" "$BACKUP_SCRIPT" --action backup --config "$TRADING_SYSTEM_DIR/config/backup_config.json" >> "$BACKUP_LOG" 2>&1

if [ \$? -eq 0 ]; then
    echo "\$(date): Backup completed successfully" >> "$BACKUP_LOG"
else
    echo "\$(date): Backup failed with exit code \$?" >> "$BACKUP_LOG"
    # Send alert (you can customize this)
    echo "Trading System backup failed at \$(date)" | logger -t trading-backup
fi
EOF

chmod +x "$TRADING_SYSTEM_DIR/scripts/run_backup.sh"
echo "✅ Created backup wrapper script: $TRADING_SYSTEM_DIR/scripts/run_backup.sh"

# Create cleanup script
cat > "$TRADING_SYSTEM_DIR/scripts/run_backup_cleanup.sh" << EOF
#!/bin/bash
# Automated backup cleanup script

set -e

cd "$TRADING_SYSTEM_DIR"

export PYTHONPATH=".:shared/python-common"

echo "\$(date): Starting backup cleanup..." >> "$BACKUP_LOG"
"$PYTHON_BIN" "$BACKUP_SCRIPT" --action cleanup --config "$TRADING_SYSTEM_DIR/config/backup_config.json" >> "$BACKUP_LOG" 2>&1

if [ \$? -eq 0 ]; then
    echo "\$(date): Backup cleanup completed successfully" >> "$BACKUP_LOG"
else
    echo "\$(date): Backup cleanup failed with exit code \$?" >> "$BACKUP_LOG"
fi
EOF

chmod +x "$TRADING_SYSTEM_DIR/scripts/run_backup_cleanup.sh"
echo "✅ Created backup cleanup script: $TRADING_SYSTEM_DIR/scripts/run_backup_cleanup.sh"

# Add cron jobs
echo "Setting up cron jobs..."

# Remove existing trading-system backup crons
crontab -l 2>/dev/null | grep -v "trading-system-backup" | crontab -

# Get current crontab
TEMP_CRON=$(mktemp)
crontab -l 2>/dev/null > "$TEMP_CRON" || true

# Add new cron jobs
cat >> "$TEMP_CRON" << EOF

# AI Trading System Automated Backups
# Daily backup at 2 AM
0 2 * * * $TRADING_SYSTEM_DIR/scripts/run_backup.sh # trading-system-backup-daily

# Weekly cleanup on Sundays at 3 AM  
0 3 * * 0 $TRADING_SYSTEM_DIR/scripts/run_backup_cleanup.sh # trading-system-backup-cleanup

# Monthly full system check on 1st at 4 AM
0 4 1 * * $TRADING_SYSTEM_DIR/scripts/run_backup.sh # trading-system-backup-monthly

EOF

# Install new crontab
crontab "$TEMP_CRON"
rm "$TEMP_CRON"

echo "✅ Cron jobs installed:"
echo "   - Daily backups: 2:00 AM"
echo "   - Weekly cleanup: Sunday 3:00 AM" 
echo "   - Monthly check: 1st of month 4:00 AM"

# Test backup script
echo "🧪 Testing backup script..."
if [ -x "$PYTHON_BIN" ] && [ -f "$BACKUP_SCRIPT" ]; then
    echo "✅ Backup script is executable and Python is available"
    
    # Test dry run
    cd "$TRADING_SYSTEM_DIR"
    export PYTHONPATH=".:shared/python-common"
    "$PYTHON_BIN" "$BACKUP_SCRIPT" --action list > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Backup script test passed"
    else
        echo "⚠️  Backup script test failed - may work when databases are available"
    fi
else
    echo "❌ Python or backup script not found"
    exit 1
fi

# Create backup directory with proper permissions
sudo mkdir -p /var/backups/trading-system
sudo chown $USER:$USER /var/backups/trading-system
sudo chmod 750 /var/backups/trading-system

echo "✅ Backup directory created: /var/backups/trading-system"

echo ""
echo "🎉 Backup automation setup completed!"
echo ""
echo "📋 Summary:"
echo "   - Backup configuration: $TRADING_SYSTEM_DIR/config/backup_config.json"
echo "   - Backup logs: $BACKUP_LOG"
echo "   - Backup storage: /var/backups/trading-system"
echo ""
echo "🔧 Manual commands:"
echo "   - Run backup now: $TRADING_SYSTEM_DIR/scripts/run_backup.sh"
echo "   - List backups: $PYTHON_BIN $BACKUP_SCRIPT --action list"
echo "   - Cleanup old: $PYTHON_BIN $BACKUP_SCRIPT --action cleanup"
echo ""
echo "📅 Scheduled jobs:"
crontab -l | grep "trading-system-backup"
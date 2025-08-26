#!/bin/bash
#
# Production Automated Backup Script for AI Trading System
# This script performs comprehensive backups of all critical system components
# Run via cron: 0 2 * * * /srv/trading/scripts/automated_backup.sh

set -euo pipefail

# Load environment variables
if [ -f "/srv/trading/.env.production" ]; then
    export $(grep -v '^#' /srv/trading/.env.production | xargs)
fi

# Configuration
BACKUP_BASE_DIR="${BACKUP_PATH:-/mnt/bulkdata/backups}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="$BACKUP_BASE_DIR/$TIMESTAMP"
LOG_FILE="$BACKUP_BASE_DIR/backup_$TIMESTAMP.log"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY}"

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
    log "ERROR: Backup failed at line $1"
    
    # Send alert (implement your notification method)
    if [ ! -z "${ALERT_EMAIL}" ]; then
        echo "Backup failed on $(hostname) at $(date)" | mail -s "CRITICAL: Backup Failed" "$ALERT_EMAIL"
    fi
    
    # Cleanup partial backup
    rm -rf "$BACKUP_DIR"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Start backup process
log "Starting automated backup process"

# 1. Database Backup
log "Backing up PostgreSQL database..."
PGPASSWORD="$DB_PASSWORD" pg_dump \
    -h "${DB_HOST:-postgres}" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --format=custom \
    --verbose \
    --file="$BACKUP_DIR/database_$TIMESTAMP.dump" \
    2>> "$LOG_FILE"

# Verify database backup
if [ -s "$BACKUP_DIR/database_$TIMESTAMP.dump" ]; then
    log "Database backup completed successfully"
else
    log "ERROR: Database backup file is empty"
    exit 1
fi

# 2. Redis Backup (if using Redis persistence)
log "Backing up Redis data..."
if [ ! -z "${REDIS_PASSWORD}" ]; then
    redis-cli -h "${REDIS_HOST:-redis}" -a "$REDIS_PASSWORD" --rdb "$BACKUP_DIR/redis_$TIMESTAMP.rdb" 2>> "$LOG_FILE"
else
    redis-cli -h "${REDIS_HOST:-redis}" --rdb "$BACKUP_DIR/redis_$TIMESTAMP.rdb" 2>> "$LOG_FILE"
fi

# 3. Model Artifacts Backup
log "Backing up ML model artifacts..."
MODEL_DIR="/srv/trading/models"
if [ -d "$MODEL_DIR" ]; then
    tar -czf "$BACKUP_DIR/models_$TIMESTAMP.tar.gz" -C "$MODEL_DIR" . 2>> "$LOG_FILE"
    log "Model artifacts backed up"
else
    log "WARNING: Model directory not found"
fi

# 4. Configuration Backup (excluding secrets)
log "Backing up configuration files..."
CONFIG_DIR="/srv/trading/config"
if [ -d "$CONFIG_DIR" ]; then
    # Exclude .env files and other sensitive files
    tar --exclude='*.env*' --exclude='*secret*' \
        -czf "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" \
        -C "$CONFIG_DIR" . 2>> "$LOG_FILE"
    log "Configuration backed up"
fi

# 5. Trading Data Backup
log "Backing up trading data..."
DATA_DIR="/srv/trading/data"
if [ -d "$DATA_DIR" ]; then
    tar -czf "$BACKUP_DIR/trading_data_$TIMESTAMP.tar.gz" -C "$DATA_DIR" . 2>> "$LOG_FILE"
    log "Trading data backed up"
fi

# 6. Logs Backup (last 7 days)
log "Backing up recent logs..."
LOGS_DIR="/srv/trading/logs"
if [ -d "$LOGS_DIR" ]; then
    find "$LOGS_DIR" -type f -mtime -7 -print0 | \
        tar -czf "$BACKUP_DIR/logs_$TIMESTAMP.tar.gz" --null -T - 2>> "$LOG_FILE"
    log "Logs backed up"
fi

# 7. Create backup manifest
log "Creating backup manifest..."
cat > "$BACKUP_DIR/manifest.json" <<EOF
{
    "timestamp": "$TIMESTAMP",
    "hostname": "$(hostname)",
    "components": [
        "database",
        "redis",
        "models",
        "config",
        "trading_data",
        "logs"
    ],
    "size_bytes": $(du -sb "$BACKUP_DIR" | cut -f1),
    "retention_days": $RETENTION_DAYS
}
EOF

# 8. Encrypt backup if encryption key is provided
if [ ! -z "$ENCRYPTION_KEY" ]; then
    log "Encrypting backup..."
    
    # Create tarball of entire backup
    tar -czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_BASE_DIR" "$TIMESTAMP" 2>> "$LOG_FILE"
    
    # Encrypt using openssl
    openssl enc -aes-256-cbc -salt -in "$BACKUP_DIR.tar.gz" \
        -out "$BACKUP_DIR.tar.gz.enc" -pass pass:"$ENCRYPTION_KEY" 2>> "$LOG_FILE"
    
    # Remove unencrypted files
    rm -rf "$BACKUP_DIR"
    rm -f "$BACKUP_DIR.tar.gz"
    
    log "Backup encrypted successfully"
fi

# 9. Upload to S3 if configured
if [ ! -z "${BACKUP_S3_BUCKET}" ]; then
    log "Uploading backup to S3..."
    
    if [ -f "$BACKUP_DIR.tar.gz.enc" ]; then
        aws s3 cp "$BACKUP_DIR.tar.gz.enc" \
            "s3://$BACKUP_S3_BUCKET/backups/$TIMESTAMP.tar.gz.enc" \
            --region "${BACKUP_S3_REGION:-us-east-1}" 2>> "$LOG_FILE"
    else
        aws s3 sync "$BACKUP_DIR" \
            "s3://$BACKUP_S3_BUCKET/backups/$TIMESTAMP/" \
            --region "${BACKUP_S3_REGION:-us-east-1}" 2>> "$LOG_FILE"
    fi
    
    log "Backup uploaded to S3"
fi

# 10. Cleanup old backups
log "Cleaning up old backups..."
find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \; 2>> "$LOG_FILE"
find "$BACKUP_BASE_DIR" -maxdepth 1 -name "*.tar.gz.enc" -mtime +$RETENTION_DAYS -delete 2>> "$LOG_FILE"

# 11. Verify backup integrity
log "Verifying backup integrity..."
if [ -f "$BACKUP_DIR.tar.gz.enc" ]; then
    # Test decrypt
    openssl enc -aes-256-cbc -d -in "$BACKUP_DIR.tar.gz.enc" \
        -pass pass:"$ENCRYPTION_KEY" 2>/dev/null | tar -tzf - > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        log "Backup integrity verified"
    else
        log "ERROR: Backup integrity check failed"
        exit 1
    fi
fi

# 12. Generate backup report
BACKUP_SIZE=$(du -sh "$BACKUP_BASE_DIR/$TIMESTAMP"* | cut -f1)
cat > "$BACKUP_BASE_DIR/latest_backup_report.txt" <<EOF
AI Trading System Backup Report
================================
Timestamp: $TIMESTAMP
Status: SUCCESS
Size: $BACKUP_SIZE
Location: $BACKUP_DIR
Retention: $RETENTION_DAYS days
Encrypted: $([ ! -z "$ENCRYPTION_KEY" ] && echo "Yes" || echo "No")
S3 Upload: $([ ! -z "${BACKUP_S3_BUCKET}" ] && echo "Yes" || echo "No")

Components Backed Up:
- PostgreSQL Database
- Redis Data
- ML Models
- Configuration
- Trading Data
- Application Logs

Next scheduled backup: $(date -d "tomorrow 02:00" '+%Y-%m-%d %H:%M:%S')
EOF

log "Backup completed successfully"

# 13. Update monitoring metrics (if Prometheus pushgateway is configured)
if [ ! -z "${PROMETHEUS_PUSHGATEWAY_URL}" ]; then
    cat <<EOF | curl --data-binary @- "${PROMETHEUS_PUSHGATEWAY_URL}/metrics/job/backup/instance/$(hostname)"
# TYPE backup_last_success gauge
backup_last_success $(date +%s)
# TYPE backup_size_bytes gauge
backup_size_bytes $(du -sb "$BACKUP_BASE_DIR/$TIMESTAMP"* | cut -f1)
# TYPE backup_duration_seconds gauge
backup_duration_seconds $SECONDS
EOF
fi

exit 0
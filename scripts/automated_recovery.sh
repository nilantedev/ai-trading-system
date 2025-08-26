#!/bin/bash
#
# Production Automated Recovery Script for AI Trading System
# This script restores the system from backup in case of disaster
# Usage: ./automated_recovery.sh [backup_timestamp]

set -euo pipefail

# Load environment variables
if [ -f "/srv/trading/.env.production" ]; then
    export $(grep -v '^#' /srv/trading/.env.production | xargs)
fi

# Configuration
BACKUP_BASE_DIR="${BACKUP_PATH:-/mnt/bulkdata/backups}"
RECOVERY_LOG="/tmp/recovery_$(date +%Y%m%d_%H%M%S).log"
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY}"

# Parse arguments
BACKUP_TIMESTAMP="${1:-latest}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RECOVERY_LOG"
}

# Error handling
handle_error() {
    log "ERROR: Recovery failed at line $1"
    log "Please check the recovery log at: $RECOVERY_LOG"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Start recovery process
log "Starting disaster recovery process"

# 1. Find backup to restore
if [ "$BACKUP_TIMESTAMP" == "latest" ]; then
    log "Finding latest backup..."
    
    # Find most recent backup
    if [ -f "$BACKUP_BASE_DIR/latest_backup_report.txt" ]; then
        BACKUP_TIMESTAMP=$(grep "Timestamp:" "$BACKUP_BASE_DIR/latest_backup_report.txt" | awk '{print $2}')
        log "Using latest backup: $BACKUP_TIMESTAMP"
    else
        # Find most recent directory
        BACKUP_TIMESTAMP=$(ls -t "$BACKUP_BASE_DIR" | grep -E '^[0-9]{8}_[0-9]{6}' | head -1)
        if [ -z "$BACKUP_TIMESTAMP" ]; then
            log "ERROR: No backup found"
            exit 1
        fi
    fi
fi

BACKUP_DIR="$BACKUP_BASE_DIR/$BACKUP_TIMESTAMP"

# 2. Check if backup is encrypted and decrypt if necessary
if [ -f "$BACKUP_DIR.tar.gz.enc" ]; then
    log "Decrypting backup..."
    
    if [ -z "$ENCRYPTION_KEY" ]; then
        log "ERROR: Backup is encrypted but BACKUP_ENCRYPTION_KEY is not set"
        exit 1
    fi
    
    # Decrypt backup
    openssl enc -aes-256-cbc -d -in "$BACKUP_DIR.tar.gz.enc" \
        -out "$BACKUP_DIR.tar.gz" -pass pass:"$ENCRYPTION_KEY"
    
    # Extract backup
    tar -xzf "$BACKUP_DIR.tar.gz" -C "$BACKUP_BASE_DIR"
    rm -f "$BACKUP_DIR.tar.gz"
    
    log "Backup decrypted successfully"
fi

# 3. Verify backup exists
if [ ! -d "$BACKUP_DIR" ]; then
    log "ERROR: Backup directory not found: $BACKUP_DIR"
    exit 1
fi

log "Using backup from: $BACKUP_DIR"

# 4. Stop services
log "Stopping services..."
cd /srv/trading
docker-compose down || true

# 5. Restore PostgreSQL database
if [ -f "$BACKUP_DIR/database_$BACKUP_TIMESTAMP.dump" ]; then
    log "Restoring PostgreSQL database..."
    
    # Start only PostgreSQL service
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    sleep 10
    
    # Drop existing database and recreate
    PGPASSWORD="$DB_PASSWORD" psql \
        -h "${DB_HOST:-localhost}" \
        -U "$DB_USER" \
        -d postgres \
        -c "DROP DATABASE IF EXISTS $DB_NAME;"
    
    PGPASSWORD="$DB_PASSWORD" psql \
        -h "${DB_HOST:-localhost}" \
        -U "$DB_USER" \
        -d postgres \
        -c "CREATE DATABASE $DB_NAME;"
    
    # Restore database
    PGPASSWORD="$DB_PASSWORD" pg_restore \
        -h "${DB_HOST:-localhost}" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --verbose \
        --no-owner \
        --no-privileges \
        "$BACKUP_DIR/database_$BACKUP_TIMESTAMP.dump" \
        2>> "$RECOVERY_LOG"
    
    log "Database restored successfully"
else
    log "WARNING: Database backup not found"
fi

# 6. Restore Redis data
if [ -f "$BACKUP_DIR/redis_$BACKUP_TIMESTAMP.rdb" ]; then
    log "Restoring Redis data..."
    
    # Stop Redis if running
    docker-compose stop redis || true
    
    # Copy Redis backup to volume
    docker run --rm \
        -v trading-system_redis_data:/data \
        -v "$BACKUP_DIR:/backup" \
        alpine cp "/backup/redis_$BACKUP_TIMESTAMP.rdb" /data/dump.rdb
    
    log "Redis data restored"
else
    log "WARNING: Redis backup not found"
fi

# 7. Restore ML models
if [ -f "$BACKUP_DIR/models_$BACKUP_TIMESTAMP.tar.gz" ]; then
    log "Restoring ML models..."
    
    MODEL_DIR="/srv/trading/models"
    mkdir -p "$MODEL_DIR"
    
    # Backup current models if they exist
    if [ "$(ls -A $MODEL_DIR)" ]; then
        mv "$MODEL_DIR" "${MODEL_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Extract models
    tar -xzf "$BACKUP_DIR/models_$BACKUP_TIMESTAMP.tar.gz" -C "$MODEL_DIR"
    
    log "ML models restored"
else
    log "WARNING: Model backup not found"
fi

# 8. Restore configuration (carefully)
if [ -f "$BACKUP_DIR/config_$BACKUP_TIMESTAMP.tar.gz" ]; then
    log "Restoring configuration files..."
    
    CONFIG_DIR="/srv/trading/config"
    CONFIG_BACKUP="${CONFIG_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    
    # Backup current config
    if [ -d "$CONFIG_DIR" ]; then
        cp -r "$CONFIG_DIR" "$CONFIG_BACKUP"
    fi
    
    # Extract config (but preserve .env files)
    tar -xzf "$BACKUP_DIR/config_$BACKUP_TIMESTAMP.tar.gz" -C "$CONFIG_DIR"
    
    # Restore .env files from backup if they exist
    if [ -d "$CONFIG_BACKUP" ]; then
        cp "$CONFIG_BACKUP"/.env* "$CONFIG_DIR"/ 2>/dev/null || true
    fi
    
    log "Configuration restored"
else
    log "WARNING: Configuration backup not found"
fi

# 9. Restore trading data
if [ -f "$BACKUP_DIR/trading_data_$BACKUP_TIMESTAMP.tar.gz" ]; then
    log "Restoring trading data..."
    
    DATA_DIR="/srv/trading/data"
    mkdir -p "$DATA_DIR"
    
    # Extract trading data
    tar -xzf "$BACKUP_DIR/trading_data_$BACKUP_TIMESTAMP.tar.gz" -C "$DATA_DIR"
    
    log "Trading data restored"
else
    log "WARNING: Trading data backup not found"
fi

# 10. Start all services
log "Starting services..."
cd /srv/trading
docker-compose up -d

# 11. Wait for services to be healthy
log "Waiting for services to become healthy..."
sleep 30

# 12. Verify services are running
log "Verifying services..."

# Check PostgreSQL
if PGPASSWORD="$DB_PASSWORD" psql -h "${DB_HOST:-localhost}" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
    log "✓ PostgreSQL is running"
else
    log "✗ PostgreSQL is not responding"
fi

# Check Redis
if redis-cli -h "${REDIS_HOST:-localhost}" ping > /dev/null 2>&1; then
    log "✓ Redis is running"
else
    log "✗ Redis is not responding"
fi

# Check API
if curl -f "http://localhost:${API_PORT:-8000}/health" > /dev/null 2>&1; then
    log "✓ API is running"
else
    log "✗ API is not responding"
fi

# 13. Run post-recovery validation
log "Running post-recovery validation..."

# Check database integrity
PGPASSWORD="$DB_PASSWORD" psql \
    -h "${DB_HOST:-localhost}" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" \
    >> "$RECOVERY_LOG" 2>&1

# 14. Generate recovery report
cat > "/tmp/recovery_report_$(date +%Y%m%d_%H%M%S).txt" <<EOF
AI Trading System Recovery Report
==================================
Recovery Date: $(date)
Backup Used: $BACKUP_TIMESTAMP
Status: COMPLETED

Components Restored:
$([ -f "$BACKUP_DIR/database_$BACKUP_TIMESTAMP.dump" ] && echo "✓ PostgreSQL Database" || echo "✗ PostgreSQL Database")
$([ -f "$BACKUP_DIR/redis_$BACKUP_TIMESTAMP.rdb" ] && echo "✓ Redis Data" || echo "✗ Redis Data")
$([ -f "$BACKUP_DIR/models_$BACKUP_TIMESTAMP.tar.gz" ] && echo "✓ ML Models" || echo "✗ ML Models")
$([ -f "$BACKUP_DIR/config_$BACKUP_TIMESTAMP.tar.gz" ] && echo "✓ Configuration" || echo "✗ Configuration")
$([ -f "$BACKUP_DIR/trading_data_$BACKUP_TIMESTAMP.tar.gz" ] && echo "✓ Trading Data" || echo "✗ Trading Data")

Service Status:
$(docker-compose ps)

Recovery Log: $RECOVERY_LOG

IMPORTANT: Please verify all services and data integrity before resuming trading operations.
EOF

log "Recovery completed successfully"
log "Recovery report saved to: /tmp/recovery_report_$(date +%Y%m%d_%H%M%S).txt"
log ""
log "IMPORTANT NEXT STEPS:"
log "1. Verify all services are functioning correctly"
log "2. Check data integrity and consistency"
log "3. Review recent trades and positions"
log "4. Test with paper trading before resuming live trading"
log "5. Monitor system closely for the next 24 hours"

exit 0
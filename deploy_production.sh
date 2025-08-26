#!/bin/bash
#
# Production Deployment Script for AI Trading System
# This script deploys the trading system to the production server
# Usage: ./deploy_production.sh [--skip-tests] [--skip-backup]

set -euo pipefail

# Configuration
SERVER_HOST="${SERVER_HOST:-168.119.145.135}"
SERVER_USER="${SERVER_USER:-nilante}"
DEPLOYMENT_PATH="${DEPLOYMENT_PATH:-/srv/trading}"
LOCAL_PATH="$(pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_TESTS=false
SKIP_BACKUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
log_info "Starting production deployment..."

# 1. Check for required files
log_info "Checking required files..."

REQUIRED_FILES=(
    ".env.production"
    "docker-compose.yml"
    "Dockerfile"
    "requirements.txt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        log_error "Required file missing: $file"
        exit 1
    fi
done

# Check that .env.production has no placeholder values
if grep -q "REQUIRED\|REPLACE_WITH\|your-.*-key" .env.production; then
    log_error ".env.production contains placeholder values. Please fill in all required secrets."
    exit 1
fi

log_info "✓ All required files present"

# 2. Run tests (unless skipped)
if [ "$SKIP_TESTS" = false ]; then
    log_info "Running critical tests..."
    
    # Run unit tests
    python -m pytest tests/unit -v --tb=short || {
        log_error "Unit tests failed"
        exit 1
    }
    
    # Run integration tests
    python -m pytest tests/integration/test_critical_trading_flows.py -v --tb=short || {
        log_error "Critical integration tests failed"
        exit 1
    }
    
    log_info "✓ All tests passed"
else
    log_warning "Skipping tests (--skip-tests flag used)"
fi

# 3. Build Docker images
log_info "Building Docker images..."

docker build -t ai-trading-system:$TIMESTAMP . || {
    log_error "Docker build failed"
    exit 1
}

docker tag ai-trading-system:$TIMESTAMP ai-trading-system:latest

log_info "✓ Docker images built successfully"

# 4. Create deployment package
log_info "Creating deployment package..."

DEPLOY_PACKAGE="deploy_${TIMESTAMP}.tar.gz"

tar -czf "$DEPLOY_PACKAGE" \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.env.development' \
    --exclude='tests' \
    --exclude='*.log' \
    --exclude='data/*.csv' \
    docker-compose.yml \
    docker-compose.secure.yml \
    Dockerfile \
    requirements.txt \
    api \
    services \
    shared \
    config \
    scripts \
    infrastructure/docker

log_info "✓ Deployment package created: $DEPLOY_PACKAGE"

# 5. Create pre-deployment backup on server
if [ "$SKIP_BACKUP" = false ]; then
    log_info "Creating pre-deployment backup on server..."
    
    ssh "$SERVER_USER@$SERVER_HOST" "cd $DEPLOYMENT_PATH && ./scripts/automated_backup.sh" || {
        log_warning "Pre-deployment backup failed (non-critical if first deployment)"
    }
else
    log_warning "Skipping backup (--skip-backup flag used)"
fi

# 6. Upload deployment package
log_info "Uploading deployment package to server..."

scp "$DEPLOY_PACKAGE" "$SERVER_USER@$SERVER_HOST:/tmp/" || {
    log_error "Failed to upload deployment package"
    exit 1
}

# Upload .env.production separately (encrypted transfer)
scp .env.production "$SERVER_USER@$SERVER_HOST:/tmp/.env.production.tmp" || {
    log_error "Failed to upload environment file"
    exit 1
}

log_info "✓ Files uploaded to server"

# 7. Deploy on server
log_info "Deploying on server..."

ssh "$SERVER_USER@$SERVER_HOST" << 'REMOTE_SCRIPT'
set -e

DEPLOYMENT_PATH="/srv/trading"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create deployment directory if it doesn't exist
mkdir -p "$DEPLOYMENT_PATH"
cd "$DEPLOYMENT_PATH"

# Stop existing services
if [ -f "docker-compose.yml" ]; then
    echo "Stopping existing services..."
    docker-compose down || true
fi

# Backup current deployment
if [ -d "api" ]; then
    echo "Backing up current deployment..."
    tar -czf "/tmp/backup_pre_deploy_$TIMESTAMP.tar.gz" \
        api services shared config scripts || true
fi

# Extract new deployment
echo "Extracting new deployment..."
tar -xzf "/tmp/deploy_*.tar.gz"

# Move environment file to correct location
mv /tmp/.env.production.tmp .env.production
chmod 600 .env.production

# Set proper permissions
chmod +x scripts/*.sh
chmod 600 .env.production

# Pull latest Docker images
docker-compose pull

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to become healthy..."
sleep 30

# Cleanup
rm -f /tmp/deploy_*.tar.gz
rm -f /tmp/.env.production.tmp

echo "Deployment completed on server"
REMOTE_SCRIPT

log_info "✓ Deployment completed on server"

# 8. Verify deployment
log_info "Verifying deployment..."

# Check API health
API_HEALTH=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s http://localhost:8000/health" || echo "failed")

if [[ "$API_HEALTH" == *"healthy"* ]]; then
    log_info "✓ API is healthy"
else
    log_error "API health check failed"
    log_error "Response: $API_HEALTH"
    
    # Show logs for debugging
    ssh "$SERVER_USER@$SERVER_HOST" "cd $DEPLOYMENT_PATH && docker-compose logs --tail=50"
    exit 1
fi

# Check database connection
DB_CHECK=$(ssh "$SERVER_USER@$SERVER_HOST" "cd $DEPLOYMENT_PATH && docker-compose exec -T postgres pg_isready" || echo "failed")

if [[ "$DB_CHECK" == *"accepting connections"* ]]; then
    log_info "✓ Database is accepting connections"
else
    log_warning "Database check returned: $DB_CHECK"
fi

# 9. Setup monitoring and backups
log_info "Setting up monitoring and automated backups..."

ssh "$SERVER_USER@$SERVER_HOST" << 'SETUP_SCRIPT'
set -e

DEPLOYMENT_PATH="/srv/trading"
cd "$DEPLOYMENT_PATH"

# Install systemd services for backup
if [ -f "scripts/backup.service" ]; then
    sudo cp scripts/backup.service /etc/systemd/system/
    sudo cp scripts/backup.timer /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable backup.timer
    sudo systemctl start backup.timer
    echo "✓ Automated backup scheduled"
fi

# Setup log rotation
cat > /etc/logrotate.d/trading-system << EOF
/srv/trading/logs/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    create 0640 root root
}
EOF

echo "✓ Log rotation configured"
SETUP_SCRIPT

log_info "✓ Monitoring and backups configured"

# 10. Run post-deployment tests
log_info "Running post-deployment smoke tests..."

# Test API endpoints
ENDPOINTS=(
    "http://localhost:8000/health"
    "http://localhost:8000/api/v1/status"
    "http://localhost:9090/api/v1/query?query=up"  # Prometheus
)

for endpoint in "${ENDPOINTS[@]}"; do
    RESPONSE=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s -o /dev/null -w '%{http_code}' $endpoint" || echo "000")
    
    if [[ "$RESPONSE" == "200" ]]; then
        log_info "✓ $endpoint is responding"
    else
        log_warning "$endpoint returned HTTP $RESPONSE"
    fi
done

# 11. Final deployment summary
log_info "========================================="
log_info "DEPLOYMENT COMPLETED SUCCESSFULLY"
log_info "========================================="
log_info "Server: $SERVER_HOST"
log_info "Path: $DEPLOYMENT_PATH"
log_info "Version: $TIMESTAMP"
log_info ""
log_info "Services URLs:"
log_info "- API: http://$SERVER_HOST:8000"
log_info "- Grafana: http://$SERVER_HOST:3000"
log_info "- Prometheus: http://$SERVER_HOST:9090"
log_info ""
log_info "Next steps:"
log_info "1. Monitor logs: ssh $SERVER_USER@$SERVER_HOST 'cd $DEPLOYMENT_PATH && docker-compose logs -f'"
log_info "2. Check metrics in Grafana"
log_info "3. Verify automated backups are running"
log_info "4. Test with paper trading before enabling live trading"
log_info ""
log_warning "IMPORTANT: The system is now LIVE. Monitor closely for the first 24 hours."

# Cleanup local files
rm -f "$DEPLOY_PACKAGE"

exit 0
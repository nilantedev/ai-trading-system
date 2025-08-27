#!/bin/bash
#
# Production Deployment Script for AI Trading System
# This script deploys the trading system to the production server using an atomic versioned release pattern.
#
# Usage: ./deploy_production.sh [flags]
#
# Core Flags:
#   --skip-tests            Skip running local test suites
#   --skip-backup           Skip invoking remote pre-deployment backup script
#   --no-sbom               Disable SBOM generation & transfer
#   --multi-compose         Append additional production compose file if present
#   --no-vuln-scan          Skip vulnerability scanning (trivy/grype)
#   --enforce-vuln          Fail deployment if HIGH/CRITICAL vulns detected
#   --no-slo-guard          Skip Prometheus 5xx SLO query
#   --slo-threshold <float> Maximum allowed 5xx rate over 5m window (default 0.1)
#   --no-canary             Skip canary endpoint checks
#   --dry-run               Execute build/tests/scans only; do not upload/deploy
#   -h / --help             Show this help
#
# Behavior:
#   1. Validates required files & env variables.
#   2. Runs tests unless skipped.
#   3. Builds Docker image (timestamp tag + latest).
#   4. (Dry-run mode returns here after optional scans.)
#   5. Creates tarball excluding transient artifacts.
#   6. Uploads package & env file; extracts into new timestamped release dir.
#   7. Atomic symlink flip to new release with rollback on failure.
#   8. Health, readiness, parity, canary, SLO, vulnerability, and metrics checks.
#   9. Optional SBOM generation + retention + hash & structured deployment event.
#  10. Post-deploy smoke tests; summary output.

set -euo pipefail

# Configuration
SERVER_HOST="${SERVER_HOST:-168.119.145.135}"
SERVER_USER="${SERVER_USER:-nilante}"
DEPLOYMENT_PATH="${DEPLOYMENT_PATH:-/srv/trading}"
LOCAL_PATH="$(pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RELEASES_DIR="releases"
CURRENT_LINK="current"
# Use /var/lock for persistence across reboots
LOCK_DIR="/var/lock/ai-trading-system"
mkdir -p "$LOCK_DIR" 2>/dev/null || true
LOCK_FILE="$LOCK_DIR/deploy.lock"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_TESTS=false
SKIP_BACKUP=false
GENERATE_SBOM=true
MULTI_COMPOSE=false
VULN_SCAN=true
SLO_GUARD=true
CANARY_TEST=true
DRY_RUN=false
ENFORCE_VULN=false
SLO_THRESHOLD="0.1" # default allowable 5xx rate

ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-sbom) GENERATE_SBOM=false ; shift ;;
    --multi-compose) MULTI_COMPOSE=true ; shift ;;
    --no-vuln-scan) VULN_SCAN=false ; shift ;;
    --no-slo-guard) SLO_GUARD=false ; shift ;;
    --no-canary) CANARY_TEST=false ; shift ;;
    --skip-tests) SKIP_TESTS=true ; shift ;;
    --skip-backup) SKIP_BACKUP=true ; shift ;;
    --dry-run) DRY_RUN=true ; shift ;;
    --enforce-vuln) ENFORCE_VULN=true ; shift ;;
    --slo-threshold)
        if [[ -n ${2:-} ]]; then SLO_THRESHOLD="$2"; shift 2; else echo "--slo-threshold requires value"; exit 1; fi ;;
    -h|--help)
        echo "Usage: $0 [options]"
        echo "  --skip-tests          Skip local test execution"
        echo "  --skip-backup         Skip remote pre-deployment backup"
        echo "  --no-sbom             Disable SBOM generation/transfer"
        echo "  --multi-compose       Include extra production compose file if present"
        echo "  --no-vuln-scan        Skip vulnerability scanning"
        echo "  --enforce-vuln        Fail deployment on HIGH/CRITICAL vulns (requires scanner)"
        echo "  --no-slo-guard        Skip Prometheus SLO guard query"
        echo "  --slo-threshold <n>   Max allowed 5xx rate (float, default 0.1)"
        echo "  --no-canary           Skip canary endpoint checks"
        echo "  --dry-run             Run build/tests/scans only; do not upload or deploy"
        exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]}"

if [ "$DRY_RUN" = true ]; then
  log_info "Dry-run mode: remote deployment steps will be skipped"
fi

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
# Concurrency lock to prevent overlapping deployments
exec 9>"$LOCK_FILE" || { log_error "Cannot open lock file"; exit 1; }
if ! flock -n 9; then
    log_error "Another deployment is in progress. Aborting."; exit 1; fi
log_info "Acquired deployment lock"

# 1. Check for required files
log_info "Checking required files..."

REQUIRED_FILES=(
    ".env.production"
    "docker-compose.yml"
    "Dockerfile"
    "requirements.txt"
    "config/logging.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        log_error "Required file missing: $file"
        exit 1
    fi
done

# Allow either legacy or runtime SBOM generator presence when SBOM enabled
if [ "$GENERATE_SBOM" = true ]; then
    if [ ! -f scripts/generate_sbom.py ] && [ ! -f scripts/runtime/generate_sbom.py ]; then
        log_warning "SBOM generation enabled but no generator script found (continuing without SBOM)"
        GENERATE_SBOM=false
    fi
fi

if grep -q "REQUIRED\|REPLACE_WITH\|your-.*-key" .env.production; then
    log_error ".env.production contains placeholder values. Please fill in all required secrets."
    exit 1
fi

# Extended critical env validation (subset, best-effort)
REQUIRED_VARS=(SECRET_KEY JWT_SECRET DB_USER DB_PASSWORD DB_NAME REDIS_PASSWORD GRAFANA_PASSWORD)
OPTIONAL_VARS=(MINIO_ROOT_USER MINIO_ROOT_PASSWORD WEAVIATE_AUTHENTICATION_APIKEY_ALLOWED_KEYS WEAVIATE_AUTHENTICATION_APIKEY_USERS ALPACA_API_KEY ALPACA_SECRET BACKUP_ENCRYPTION_KEY)
MISSING=0
for v in "${REQUIRED_VARS[@]}"; do
    if ! grep -E "^${v}=" .env.production >/dev/null; then
        log_error "Missing required env var: ${v} in .env.production"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    log_error "Environment validation failed"; exit 1; fi
log_info "✓ Environment variable validation passed"

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

# 3. Migration head check (local)
if command -v alembic >/dev/null 2>&1 && [ -f alembic.ini ]; then
    log_info "Checking alembic migration head locally..."
    LOCAL_HEAD=$(alembic heads 2>/dev/null | awk '{print $1}' | head -n1 || true)
    if [ -n "$LOCAL_HEAD" ]; then
        log_info "Local alembic head: $LOCAL_HEAD"
    else
        log_warning "Could not determine local migration head (continuing)"
    fi
fi

# 4. Build Docker images
log_info "Building Docker images..."

docker build -t ai-trading-system:$TIMESTAMP . || {
    log_error "Docker build failed"
    exit 1
}

docker tag ai-trading-system:$TIMESTAMP ai-trading-system:latest

log_info "✓ Docker images built successfully"

# If dry run, stop after build & (optional) vulnerability scan locally
if [ "$DRY_RUN" = true ]; then
    if [ "$VULN_SCAN" = true ]; then
        log_info "(Dry-run) Running local vulnerability scan..."
        if command -v trivy >/dev/null 2>&1; then
            trivy image --quiet --severity HIGH,CRITICAL ai-trading-system:$TIMESTAMP || log_warning "Trivy reported issues"
        elif command -v grype >/dev/null 2>&1; then
            grype ai-trading-system:$TIMESTAMP || log_warning "Grype reported issues"
        else
            log_warning "No scanner available (trivy/grype)"
        fi
    fi
    log_info "Dry-run complete. Exiting before remote deployment."
    exit 0
fi

# 5. Create deployment package
log_info "Creating deployment package (manifest-driven)..."

DEPLOY_PACKAGE="deploy_${TIMESTAMP}.tar.gz"

if [ ! -f deployment.manifest ]; then
        log_error "deployment.manifest missing"; exit 1; fi

MANIFEST_ITEMS=$(grep -v '^#' deployment.manifest | sed '/^$/d')

TMP_LIST_FILE=$(mktemp)
> "$TMP_LIST_FILE"

while IFS= read -r item; do
    if [ -e "$item" ]; then
        echo "$item" >> "$TMP_LIST_FILE"
    else
        log_warning "Manifest entry missing (skipping): $item"
    fi
done <<< "$MANIFEST_ITEMS"

tar -czf "$DEPLOY_PACKAGE" \
        --exclude='.git' \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.env*' \
        --exclude='tests' \
        --exclude='docs' \
        --exclude='research' \
        --exclude='Design Docs - Final' \
        --exclude='Server Docs' \
        --exclude='logs' \
        --exclude='data/backups' \
        --exclude='*.log' \
        --exclude='*.md' \
        --files-from "$TMP_LIST_FILE"

rm -f "$TMP_LIST_FILE"

# Optional multi-compose addition
if [ "$MULTI_COMPOSE" = true ] && [ -f infrastructure/docker/docker-compose.production.yml ]; then
    tar -rzf "$DEPLOY_PACKAGE" infrastructure/docker/docker-compose.production.yml || log_warning "Failed to append production infra compose"
fi

log_info "✓ Deployment package created: $DEPLOY_PACKAGE"

# 6. Create pre-deployment backup on server
if [ "$SKIP_BACKUP" = false ]; then
    log_info "Creating pre-deployment backup on server..."
    
    ssh "$SERVER_USER@$SERVER_HOST" "cd $DEPLOYMENT_PATH && ./scripts/automated_backup.sh" || {
        log_warning "Pre-deployment backup failed (non-critical if first deployment)"
    }
else
    log_warning "Skipping backup (--skip-backup flag used)"
fi

# 7. Upload deployment package
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

# 8. Deploy on server (atomic pattern)
log_info "Deploying on server (atomic)..."

ssh "$SERVER_USER@$SERVER_HOST" << 'REMOTE_SCRIPT'
set -e

DEPLOYMENT_PATH="/srv/trading"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RELEASES_DIR="$DEPLOYMENT_PATH/releases"
CURRENT_LINK="$DEPLOYMENT_PATH/current"
NEW_RELEASE_DIR="$RELEASES_DIR/$TIMESTAMP"

mkdir -p "$RELEASES_DIR"
cd "$DEPLOYMENT_PATH"
echo "Extracting new deployment to versioned directory..."
PKG_FILE=$(ls -1 /tmp/deploy_*.tar.gz | tail -n1)
mkdir -p "$NEW_RELEASE_DIR"
tar -xzf "$PKG_FILE" -C "$NEW_RELEASE_DIR"

# Move environment file to correct location
mv /tmp/.env.production.tmp "$NEW_RELEASE_DIR/.env.production"
chmod 600 "$NEW_RELEASE_DIR/.env.production"

# Set proper permissions
chmod +x "$NEW_RELEASE_DIR"/scripts/*.sh || true

# Atomic symlink switch
PREVIOUS_TARGET=""
if [ -L "$CURRENT_LINK" ]; then PREVIOUS_TARGET=$(readlink "$CURRENT_LINK"); fi
ln -sfn "$NEW_RELEASE_DIR" "$CURRENT_LINK"
cd "$CURRENT_LINK"

docker-compose pull || true
echo "Starting services (atomic)..."
if ! docker-compose up -d; then
    echo "Startup failed; attempting rollback" >&2
    if [ -n "$PREVIOUS_TARGET" ] && [ -d "$PREVIOUS_TARGET" ]; then
        ln -sfn "$PREVIOUS_TARGET" "$CURRENT_LINK"
        cd "$CURRENT_LINK" && docker-compose up -d || true
    fi
    exit 1
fi

# Wait for services to be healthy
echo "Waiting for services to become healthy..."
sleep 30

# Cleanup
rm -f /tmp/deploy_*.tar.gz /tmp/.env.production.tmp || true

echo "Deployment completed on server"
REMOTE_SCRIPT

log_info "✓ Deployment completed on server"

if [ "$GENERATE_SBOM" = true ]; then
    log_info "Generating SBOM locally..."
    if [ -f scripts/runtime/generate_sbom.py ]; then
        python scripts/runtime/generate_sbom.py || log_warning "SBOM generation failed (continuing)"
    elif [ -f scripts/generate_sbom.py ]; then
        python scripts/generate_sbom.py || log_warning "SBOM generation failed (continuing)"
    else
        log_warning "No SBOM generator available despite flag; skipping"
    fi
    SBOM_DIR="sbom_${TIMESTAMP}"
    mkdir -p "$SBOM_DIR" && mv sbom.spdx.json sbom.cyclonedx.json "$SBOM_DIR" 2>/dev/null || true
    tar -czf "${SBOM_DIR}.tar.gz" "$SBOM_DIR" || true
    scp "${SBOM_DIR}.tar.gz" "$SERVER_USER@$SERVER_HOST:/tmp/" || log_warning "Failed to transfer SBOM archive"
fi

# 9. Verify deployment
log_info "Verifying deployment..."

# Check API health
API_HEALTH=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s http://localhost:8000/health" || echo "failed")

if [[ "$API_HEALTH" == *"healthy"* ]]; then
    log_info "✓ API is healthy"
else
    log_error "API health check failed"
    log_error "Response: $API_HEALTH"
# Move SBOM archive into release directory & compute hash (on server)
if [ "$GENERATE_SBOM" = true ]; then
    ssh "$SERVER_USER@$SERVER_HOST" "bash -c 'set -e; REL_DIR=\"$DEPLOYMENT_PATH/releases/$TIMESTAMP\"; SBOM_TMP=\"/tmp/sbom_${TIMESTAMP}.tar.gz\"; if [ -f \"$SBOM_TMP\" ]; then mkdir -p \"$REL_DIR/sbom\"; mv \"$SBOM_TMP\" \"$REL_DIR/sbom/\"; cd \"$REL_DIR/sbom\"; SBOM_HASH=\"$(sha256sum sbom_${TIMESTAMP}.tar.gz 2>/dev/null | awk '{print $1}')\"; echo $SBOM_HASH > sbom.sha256 || true; fi'" || log_warning "Remote SBOM retention failed"
fi

# Compute sanitized env hash (server side, redacting secrets heuristically)
ENV_HASH=$(ssh "$SERVER_USER@$SERVER_HOST" "bash -c 'grep -v -E "(SECRET|PASSWORD|TOKEN|KEY)=" $DEPLOYMENT_PATH/releases/$TIMESTAMP/.env.production 2>/dev/null | sha256sum | awk '{print $1}' || true'" || true)
log_info "Env hash (sanitized): ${ENV_HASH:-n/a}"

# Emit deployment structured event (runtime first)
ssh "$SERVER_USER@$SERVER_HOST" "bash -c '
EVT_SCRIPT="$DEPLOYMENT_PATH/releases/$TIMESTAMP/scripts/runtime/emit_deployment_event.py"
if [ ! -f "$EVT_SCRIPT" ]; then EVT_SCRIPT="$DEPLOYMENT_PATH/releases/$TIMESTAMP/scripts/emit_deployment_event.py"; fi
SBOM_HASH_FILE="$DEPLOYMENT_PATH/releases/$TIMESTAMP/sbom/sbom.sha256"
SBOM_HASH=$(cat "$SBOM_HASH_FILE" 2>/dev/null | awk '{print $1}')
python "$EVT_SCRIPT" --version $TIMESTAMP --sbom-hash ${SBOM_HASH:-} --env-hash $ENV_HASH --release-dir $DEPLOYMENT_PATH/releases/$TIMESTAMP || true'" || log_warning "Deployment event emission failed"

# Optional vulnerability scan placeholder
if [ "$VULN_SCAN" = true ]; then
    log_info "Attempting vulnerability scan (trivy or grype if available)..."
    if command -v trivy >/dev/null 2>&1; then
        VULN_OUT=$(trivy image --format json --severity HIGH,CRITICAL ai-trading-system:latest 2>/dev/null || true)
        CRIT_COUNT=$(echo "$VULN_OUT" | grep -o 'CRITICAL' | wc -l | tr -d ' ')
        HIGH_COUNT=$(echo "$VULN_OUT" | grep -o 'HIGH' | wc -l | tr -d ' ')
        TOTAL_VULNS=$((CRIT_COUNT + HIGH_COUNT))
        if [ "$TOTAL_VULNS" -gt 0 ]; then
            log_warning "Trivy found HIGH/CRITICAL vulnerabilities: total=$TOTAL_VULNS (CRITICAL=$CRIT_COUNT HIGH=$HIGH_COUNT)"
            if [ "$ENFORCE_VULN" = true ]; then
                log_error "Failing due to vulnerabilities with --enforce-vuln"
                exit 1
            fi
        else
            log_info "Trivy: No HIGH/CRITICAL vulnerabilities"
        fi
    elif command -v grype >/dev/null 2>&1; then
        VULN_OUT=$(grype -o json ai-trading-system:latest 2>/dev/null || true)
        CRIT_COUNT=$(echo "$VULN_OUT" | grep -o 'Critical' | wc -l | tr -d ' ')
        HIGH_COUNT=$(echo "$VULN_OUT" | grep -o 'High' | wc -l | tr -d ' ')
        TOTAL_VULNS=$((CRIT_COUNT + HIGH_COUNT))
        if [ "$TOTAL_VULNS" -gt 0 ]; then
            log_warning "Grype found High/Critical vulnerabilities: total=$TOTAL_VULNS (Critical=$CRIT_COUNT High=$HIGH_COUNT)"
            if [ "$ENFORCE_VULN" = true ]; then
                log_error "Failing due to vulnerabilities with --enforce-vuln"
                exit 1
            fi
        else
            log_info "Grype: No High/Critical vulnerabilities"
        fi
    else
        # Fallback to runtime security scan script if present
        if [ -f scripts/runtime/security_scan.py ]; then
            log_info "Using fallback runtime security scan script"
            python scripts/runtime/security_scan.py ai-trading-system:latest || log_warning "Runtime security scan script issues"
        else
            log_warning "No vulnerability scanner (trivy/grype) installed locally; skipping"
        fi
    fi
fi

# SLO guard (simple 5xx rate check from Prometheus) placeholder
if [ "$SLO_GUARD" = true ]; then
    PROM_QUERY='sum(rate(api_request_total{status=~"5.."}[5m]))'
    RATE=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s 'http://localhost:9090/api/v1/query?query=${PROM_QUERY}' | grep -Eo '"value".*\[.*"([0-9.]+)"' | grep -Eo '[0-9.]+$' || echo 0")
    log_info "Recent 5xx rate (5m): ${RATE}" 
    awk -v r="$RATE" -v thr="$SLO_THRESHOLD" 'BEGIN{ if (r+0 > thr+0) exit 42 }'
    if [ $? -eq 42 ]; then
        log_warning "5xx rate ${RATE} exceeds threshold ${SLO_THRESHOLD}" 
        if [ "$SLO_GUARD" = true ]; then
            log_warning "(Informational unless --enforce-slo implemented)"
        fi
    fi
fi

# Canary test (basic endpoint + metrics endpoint)
if [ "$CANARY_TEST" = true ]; then
    CANARY=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/api/v1/health" || echo 000)
    if [ "$CANARY" != "200" ]; then
        log_warning "Canary health endpoint returned $CANARY"
    else
        log_info "Canary health endpoint OK"
    fi
    # Additional canary endpoints
    METRICS_CODE=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/metrics" || echo 000)
    if [ "$METRICS_CODE" != "200" ]; then
        log_warning "Metrics endpoint returned $METRICS_CODE"
    else
        log_info "Metrics endpoint OK"
    fi
    READY_CODE=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/ready" || echo 000)
    if [ "$READY_CODE" != "200" ]; then
        log_warning "Ready endpoint returned $READY_CODE"
    else
        log_info "Ready endpoint OK"
    fi
    # Optional simple DB query via API if such endpoint exists (/api/v1/db/ping)
    DB_PING=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/api/v1/db/ping" || echo 000)
    if [ "$DB_PING" = "200" ]; then
        log_info "DB ping endpoint OK"
    else
        log_warning "DB ping endpoint not available or returned $DB_PING"
    fi
fi
    
    # Show logs for debugging
    ssh "$SERVER_USER@$SERVER_HOST" "cd $DEPLOYMENT_PATH && docker-compose logs --tail=50"
    exit 1
fi

# Readiness endpoint check
if command -v alembic >/dev/null 2>&1 && [ -n "${LOCAL_HEAD:-}" ]; then
    log_info "Checking remote Alembic head for parity..."
    REMOTE_HEAD=$(ssh "$SERVER_USER@$SERVER_HOST" "bash -lc 'cd $DEPLOYMENT_PATH/current && alembic heads 2>/dev/null | awk \'{print \$1}\' | head -n1'" || true)
    if [ -n "$REMOTE_HEAD" ]; then
        log_info "Remote alembic head: $REMOTE_HEAD"
        if [ "$REMOTE_HEAD" != "$LOCAL_HEAD" ]; then
            log_warning "Remote DB migration head ($REMOTE_HEAD) differs from local ($LOCAL_HEAD)"
        else
            log_info "✓ Migration head parity confirmed"
        fi
    else
        log_warning "Could not determine remote alembic head"
    fi
fi

API_READY=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/ready" || echo "000")
if [ "$API_READY" = "200" ]; then
    log_info "✓ API readiness passed"
else
    log_warning "Readiness check returned $API_READY"
fi

# Check database connection
DB_CHECK=$(ssh "$SERVER_USER@$SERVER_HOST" "cd $DEPLOYMENT_PATH && docker-compose exec -T postgres pg_isready" || echo "failed")

if [[ "$DB_CHECK" == *"accepting connections"* ]]; then
    log_info "✓ Database is accepting connections"
else
    log_warning "Database check returned: $DB_CHECK"
fi

# 10. Setup monitoring and backups
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

# 11. Run post-deployment smoke tests
log_info "Running post-deployment smoke tests..."

# Test API endpoints
ENDPOINTS=(
    "http://localhost:8000/health"
    "http://localhost:8000/ready"
    "http://localhost:8000/api/v1/health"
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

# 12. Final deployment summary
log_info "========================================="
log_info "DEPLOYMENT COMPLETED SUCCESSFULLY"
log_info "========================================="
log_info "Server: $SERVER_HOST"
log_info "Path: $DEPLOYMENT_PATH"
log_info "Version: $TIMESTAMP"
if [ -f "sbom_${TIMESTAMP}.tar.gz" ]; then
    SBOM_HASH=$(sha256sum "sbom_${TIMESTAMP}.tar.gz" | awk '{print $1}')
    log_info "SBOM hash: $SBOM_HASH"
fi
log_info "Deployment metadata recorded (event + hashes)"
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
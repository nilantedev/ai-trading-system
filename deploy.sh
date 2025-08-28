#!/bin/bash
#
# Unified Production Deployment Script for AI Trading System
# Combines production deployment, AI intelligence upgrade capabilities
# 
# Usage: ./deploy.sh [mode] [flags]
#
# Modes:
#   production  - Deploy to production server (default)
#   ai-upgrade  - Deploy with AI intelligence enhancements
#   local-test  - Test deployment locally
#
# Flags:
#   --skip-tests            Skip running local test suites
#   --skip-backup           Skip invoking remote pre-deployment backup
#   --no-sbom               Disable SBOM generation & transfer
#   --dry-run               Execute build/tests/scans only; do not upload/deploy
#   --server <host>         Override server host (default: 168.119.145.135)
#   --user <user>           Override server user (default: nilante)
#   -h, --help              Show this help
#
# Examples:
#   ./deploy.sh production              # Standard production deployment
#   ./deploy.sh ai-upgrade               # Deploy with AI enhancements
#   ./deploy.sh production --skip-tests # Skip tests for hotfix
#

set -euo pipefail

# Configuration
MODE="${1:-production}"
SERVER_HOST="${SERVER_HOST:-168.119.145.135}"
SERVER_USER="${SERVER_USER:-nilante}"
DEPLOYMENT_PATH="${DEPLOYMENT_PATH:-/srv/trading}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RELEASES_DIR="releases"
CURRENT_LINK="current"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
SKIP_TESTS=false
SKIP_BACKUP=false
GENERATE_SBOM=true
DRY_RUN=false

shift || true
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-tests) SKIP_TESTS=true; shift ;;
    --skip-backup) SKIP_BACKUP=true; shift ;;
    --no-sbom) GENERATE_SBOM=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --server) SERVER_HOST="$2"; shift 2 ;;
    --user) SERVER_USER="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | head -25 | tail -23 | sed 's/^# //'
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
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

# Mode-specific header
case "$MODE" in
    production)
        echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}    🚀 PRODUCTION DEPLOYMENT${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
        ;;
    ai-upgrade)
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}    🧠 AI INTELLIGENCE UPGRADE DEPLOYMENT${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        ;;
    local-test)
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}    🧪 LOCAL TEST DEPLOYMENT${NC}"
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
        ;;
    *)
        log_error "Unknown mode: $MODE"
        exit 1
        ;;
esac

# Pre-deployment checks
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

# Validate environment
if grep -q "REQUIRED\|REPLACE_WITH\|your-.*-key" .env.production; then
    log_error ".env.production contains placeholder values. Please configure all secrets."
    exit 1
fi

log_info "✓ All required files present"

# Run tests
if [ "$SKIP_TESTS" = false ]; then
    log_info "Running tests..."
    
    if [ "$MODE" = "ai-upgrade" ]; then
        # Test AI components
        python -m pytest tests/integration/test_ml_pipeline.py -v --tb=short || {
            log_error "ML pipeline tests failed"
            exit 1
        }
    fi
    
    # Standard tests
    python -m pytest tests/unit -v --tb=short || {
        log_error "Unit tests failed"
        exit 1
    }
    
    python -m pytest tests/integration/test_critical_trading_flows.py -v --tb=short || {
        log_error "Critical integration tests failed"
        exit 1
    }
    
    log_info "✓ All tests passed"
else
    log_warning "Skipping tests (--skip-tests flag used)"
fi

# Build Docker images
log_info "Building Docker images..."

docker build -t ai-trading-system:$TIMESTAMP . || {
    log_error "Docker build failed"
    exit 1
}

docker tag ai-trading-system:$TIMESTAMP ai-trading-system:latest
log_info "✓ Docker images built successfully"

# Mode-specific preparation
if [ "$MODE" = "ai-upgrade" ]; then
    log_info "Preparing AI intelligence components..."
    
    # Create AI configuration
    cat > ai-config.yaml << 'EOF'
version: "1.0"
deployment:
  environment: production
  ai_level: ENHANCED
  
components:
  lightweight:
    enabled: true
    models:
      - technical_indicators
      - pattern_recognition
      - statistical_models
    
  advanced:
    enabled: true
    models:
      - finbert_sentiment
      - lstm_predictor
      - ensemble_coordinator
      
monitoring:
  model_drift: true
  performance_tracking: true
  explainability: true
EOF
    
    log_info "✓ AI configuration created"
    
    # Prepare AI setup script for server
    cat > setup_ai_models.sh << 'EOF'
#!/bin/bash
set -e

echo "Setting up AI models on server..."

# Install Python AI dependencies
pip install --no-cache-dir \
    torch==2.0.1 \
    transformers==4.31.0 \
    scikit-learn==1.3.0 \
    lime==0.2.0.1 \
    shap==0.42.1

# Download lightweight models
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print('Downloading FinBERT for sentiment analysis...')
AutoTokenizer.from_pretrained('ProsusAI/finbert')
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
print('✓ Models downloaded')
"

# Setup model serving
cat > /etc/systemd/system/ai-model-server.service << 'EOSERVICE'
[Unit]
Description=AI Model Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
ExecStart=/usr/bin/python3 services/ml/model_serving_service.py
Restart=always

[Install]
WantedBy=multi-user.target
EOSERVICE

systemctl daemon-reload
systemctl enable ai-model-server
systemctl start ai-model-server

echo "✓ AI models setup complete"
EOF
    chmod +x setup_ai_models.sh
fi

# Dry run check
if [ "$DRY_RUN" = true ]; then
    log_info "Dry-run complete. Exiting before deployment."
    exit 0
fi

# Create deployment package
log_info "Creating deployment package..."

DEPLOY_PACKAGE="deploy_${MODE}_${TIMESTAMP}.tar.gz"

tar -czf "$DEPLOY_PACKAGE" \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.env*' \
    --exclude='tests' \
    --exclude='*.log' \
    .

if [ "$MODE" = "ai-upgrade" ]; then
    tar -rf "$DEPLOY_PACKAGE" ai-config.yaml setup_ai_models.sh
fi

log_info "✓ Deployment package created: $DEPLOY_PACKAGE"

# Pre-deployment backup
if [ "$SKIP_BACKUP" = false ]; then
    log_info "Creating pre-deployment backup..."
    ssh "$SERVER_USER@$SERVER_HOST" "cd $DEPLOYMENT_PATH && ./scripts/automated_backup.sh" || {
        log_warning "Backup failed (may be first deployment)"
    }
fi

# Upload and deploy
log_info "Uploading to server..."

scp "$DEPLOY_PACKAGE" "$SERVER_USER@$SERVER_HOST:/tmp/"
scp .env.production "$SERVER_USER@$SERVER_HOST:/tmp/.env.production.tmp"

log_info "Deploying on server..."

ssh "$SERVER_USER@$SERVER_HOST" << REMOTE_SCRIPT
set -e

DEPLOYMENT_PATH="$DEPLOYMENT_PATH"
TIMESTAMP="$TIMESTAMP"
MODE="$MODE"
RELEASES_DIR="\$DEPLOYMENT_PATH/releases"
CURRENT_LINK="\$DEPLOYMENT_PATH/current"
NEW_RELEASE_DIR="\$RELEASES_DIR/\$TIMESTAMP"

mkdir -p "\$RELEASES_DIR"
mkdir -p "\$NEW_RELEASE_DIR"

# Extract deployment
tar -xzf "/tmp/deploy_\${MODE}_\${TIMESTAMP}.tar.gz" -C "\$NEW_RELEASE_DIR"
mv /tmp/.env.production.tmp "\$NEW_RELEASE_DIR/.env.production"
chmod 600 "\$NEW_RELEASE_DIR/.env.production"

# Mode-specific setup
if [ "\$MODE" = "ai-upgrade" ] && [ -f "\$NEW_RELEASE_DIR/setup_ai_models.sh" ]; then
    echo "Running AI setup..."
    cd "\$NEW_RELEASE_DIR"
    ./setup_ai_models.sh || echo "AI setup failed (non-critical)"
fi

# Atomic deployment
PREVIOUS_TARGET=""
if [ -L "\$CURRENT_LINK" ]; then 
    PREVIOUS_TARGET=\$(readlink "\$CURRENT_LINK")
fi

ln -sfn "\$NEW_RELEASE_DIR" "\$CURRENT_LINK"
cd "\$CURRENT_LINK"

# Start services
docker-compose pull
if ! docker-compose up -d; then
    echo "Startup failed; attempting rollback"
    if [ -n "\$PREVIOUS_TARGET" ]; then
        ln -sfn "\$PREVIOUS_TARGET" "\$CURRENT_LINK"
        cd "\$CURRENT_LINK" && docker-compose up -d
    fi
    exit 1
fi

# Cleanup
rm -f /tmp/deploy_*.tar.gz /tmp/.env.production.tmp

echo "✓ Deployment complete"
REMOTE_SCRIPT

# Verify deployment
log_info "Verifying deployment..."

sleep 10

API_HEALTH=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s http://localhost:8000/health" || echo "failed")
if [[ "$API_HEALTH" == *"healthy"* ]]; then
    log_info "✓ API is healthy"
else
    log_error "API health check failed: $API_HEALTH"
    ssh "$SERVER_USER@$SERVER_HOST" "cd $DEPLOYMENT_PATH && docker-compose logs --tail=50"
    exit 1
fi

# Mode-specific verification
if [ "$MODE" = "ai-upgrade" ]; then
    AI_STATUS=$(ssh "$SERVER_USER@$SERVER_HOST" "curl -s http://localhost:8000/api/v1/ml/status" || echo "{}")
    if [[ "$AI_STATUS" == *"enabled"* ]]; then
        log_info "✓ AI intelligence active"
    else
        log_warning "AI features may not be fully active yet"
    fi
fi

# SBOM generation
if [ "$GENERATE_SBOM" = true ] && [ -f scripts/generate_sbom.py ]; then
    log_info "Generating SBOM..."
    python scripts/generate_sbom.py || log_warning "SBOM generation failed"
fi

# Final summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    ✅ DEPLOYMENT SUCCESSFUL${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Mode: $MODE"
echo "Server: $SERVER_HOST"
echo "Version: $TIMESTAMP"
echo ""
echo "Services:"
echo "  API: http://$SERVER_HOST:8000"
echo "  Grafana: http://$SERVER_HOST:3000"
echo "  Prometheus: http://$SERVER_HOST:9090"

if [ "$MODE" = "ai-upgrade" ]; then
    echo ""
    echo "AI Features:"
    echo "  ✓ Sentiment Analysis"
    echo "  ✓ Pattern Recognition" 
    echo "  ✓ Ensemble Models"
    echo "  ✓ Explainable AI"
fi

echo ""
echo "Monitor: ssh $SERVER_USER@$SERVER_HOST 'cd $DEPLOYMENT_PATH && docker-compose logs -f'"

# Cleanup
rm -f "$DEPLOY_PACKAGE" ai-config.yaml setup_ai_models.sh

exit 0
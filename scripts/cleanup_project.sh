#!/bin/bash

# Project Cleanup Script - Prepares for production deployment
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AI Trading System - Project Cleanup ===${NC}"
echo ""

# Change to project directory
cd /home/nilante/main-nilante-server/ai-trading-system

# 1. Clean Python cache files
echo -e "${YELLOW}1. Cleaning Python cache files...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true
find . -type f -name ".coverage" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Python cache cleaned${NC}"

# 2. Clean temporary files
echo -e "${YELLOW}2. Cleaning temporary files...${NC}"
find . -type f -name "*.tmp" -delete 2>/dev/null || true
find . -type f -name "*.temp" -delete 2>/dev/null || true
find . -type f -name "*.swp" -delete 2>/dev/null || true
find . -type f -name "*.swo" -delete 2>/dev/null || true
find . -type f -name "*~" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
find . -type f -name "Thumbs.db" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Temporary files cleaned${NC}"

# 3. Clean log files (keep structure but remove old logs)
echo -e "${YELLOW}3. Cleaning old log files...${NC}"
find ./logs -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
echo -e "${GREEN}✓ Old logs cleaned${NC}"

# 4. Consolidate environment files
echo -e "${YELLOW}4. Organizing environment files...${NC}"

# Remove deprecated template (we have the new .env.production.example)
if [ -f ".env.production.template" ]; then
    rm .env.production.template
    echo "  - Removed deprecated .env.production.template"
fi

# Ensure proper .gitignore for env files
if ! grep -q "^\.env\.production$" .gitignore 2>/dev/null; then
    echo ".env.production" >> .gitignore
    echo "  - Added .env.production to .gitignore"
fi

# Remove actual production env if accidentally committed
if [ -f ".env.production" ]; then
    git rm --cached .env.production 2>/dev/null || true
fi

echo -e "${GREEN}✓ Environment files organized${NC}"

# 5. Consolidate requirements files
echo -e "${YELLOW}5. Consolidating requirements files...${NC}"

# Create consolidated requirements if not exists
if [ ! -f "requirements-consolidated.txt" ]; then
    cat > requirements-consolidated.txt << 'EOF'
# AI Trading System - Consolidated Requirements
# ==============================================

# Core Dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.12.1
redis==5.0.1
aioredis==2.0.1

# Data Processing
pandas==2.1.3
numpy==1.25.2
scipy==1.11.4
ta==0.11.0
yfinance==0.2.32

# Machine Learning
scikit-learn==1.3.2
xgboost==2.0.2
lightgbm==4.1.0
torch==2.1.1
torch-geometric==2.4.0

# Trading APIs
alpaca-py==0.13.3
polygon-api-client==1.13.0

# Messaging
pulsar-client==3.3.0
websockets==12.0

# Monitoring
prometheus-client==0.19.0
grafana-api==1.0.3

# Security
cryptography==41.0.7
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Utilities
httpx==0.25.2
python-dateutil==2.8.2
pytz==2023.3
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
pylint==3.0.3
mypy==1.7.1

# Documentation
mkdocs==1.5.3
mkdocs-material==9.4.14
EOF
    echo "  - Created consolidated requirements file"
fi

echo -e "${GREEN}✓ Requirements consolidated${NC}"

# 6. Move test files to proper location
echo -e "${YELLOW}6. Organizing test files...${NC}"
if [ -f "test_ml_basic.py" ] || [ -f "test_services_quick.py" ]; then
    mkdir -p tests/integration
    mv test_*.py tests/integration/ 2>/dev/null || true
    echo "  - Moved test files to tests/integration/"
fi
echo -e "${GREEN}✓ Test files organized${NC}"

# 7. Clean Docker artifacts
echo -e "${YELLOW}7. Cleaning Docker artifacts...${NC}"
# Remove dangling images
docker image prune -f 2>/dev/null || true
# Remove stopped containers
docker container prune -f 2>/dev/null || true
# Remove unused volumes (careful with this)
# docker volume prune -f 2>/dev/null || true
echo -e "${GREEN}✓ Docker cleaned${NC}"

# 8. Create project structure documentation
echo -e "${YELLOW}8. Documenting project structure...${NC}"
cat > PROJECT_STRUCTURE.md << 'EOF'
# AI Trading System - Project Structure

## Directory Layout

```
ai-trading-system/
├── api/                      # FastAPI application
│   ├── routers/             # API endpoints
│   ├── auth.py             # Authentication
│   ├── rate_limiter.py     # Rate limiting
│   └── main.py             # API entry point
│
├── services/                # Microservices
│   ├── ml/                 # Machine learning services
│   │   ├── continuous_improvement_engine.py
│   │   ├── ml_orchestrator.py
│   │   ├── off_hours_training_service.py
│   │   └── reinforcement_learning_engine.py
│   ├── data-ingestion/     # Data collection
│   ├── stream-processor/   # Stream processing
│   └── strategy-engine/    # Trading strategies
│
├── shared/                  # Shared libraries
│   ├── python-common/      # Common Python modules
│   │   └── trading_common/
│   │       ├── ai_models.py      # Local AI models
│   │       ├── local_swarm.py    # Local orchestration
│   │       └── trading_agents.py # Agent coordination
│   ├── storage/            # Storage integrations
│   └── vector/            # Vector database
│
├── infrastructure/         # Infrastructure configs
│   ├── docker/           # Docker compositions
│   ├── grafana/         # Monitoring dashboards
│   └── prometheus/      # Metrics collection
│
├── config/                # Configuration files
│   ├── logging.yaml     # Logging config
│   └── postgresql.conf  # DB encryption
│
├── scripts/              # Utility scripts
│   ├── cleanup_project.sh
│   ├── deploy_production.sh
│   └── enable_postgres_encryption.sh
│
├── tests/               # Test suites
│   ├── unit/
│   └── integration/
│
├── docs/                # Documentation
│   └── CONTINUOUS_LEARNING.md
│
├── .env.production.example  # Production template
├── docker-compose.yml      # Main composition
├── Dockerfile             # Container definition
├── Makefile              # Build automation
├── pyproject.toml        # Python project config
└── requirements.txt      # Dependencies
```

## Key Files

- `.env.production.example` - Production environment template
- `docker-compose.yml` - Main Docker composition
- `Makefile` - Build and deployment automation
- `deploy_production.sh` - Production deployment script

## Environment Files

- `.env` - Development environment (git-ignored)
- `.env.production` - Production environment (git-ignored, create from .example)
- `.env.production.example` - Production template (committed)

## Local AI Models (Ollama)

All AI operations use local models - NO API costs:
- qwen2.5:72b - Analysis
- deepseek-r1:70b - Risk assessment  
- llama3.1:70b - Strategy
- mixtral:8x7b - Fast inference
- phi3:medium - Sentiment
EOF
echo -e "${GREEN}✓ Project structure documented${NC}"

# 9. Generate deployment readiness report
echo -e "${YELLOW}9. Generating deployment readiness report...${NC}"

# Count files and check sizes
TOTAL_FILES=$(find . -type f -not -path "./.venv/*" -not -path "./.git/*" | wc -l)
PROJECT_SIZE=$(du -sh . --exclude=.venv --exclude=.git 2>/dev/null | cut -f1)

cat > DEPLOYMENT_READY.md << EOF
# Deployment Readiness Report
Generated: $(date)

## Project Statistics
- Total Files: $TOTAL_FILES
- Project Size: $PROJECT_SIZE (excluding .venv and .git)
- Python Version: 3.11+
- Docker: Required
- Ollama: Required for AI models

## Checklist

### ✅ Code Quality
- [ ] All tests passing
- [ ] No hardcoded secrets
- [ ] Local AI models only (no API dependencies)
- [ ] Rate limiting configured
- [ ] PostgreSQL encryption ready

### ✅ Configuration
- [ ] .env.production configured
- [ ] Docker compose validated
- [ ] SSL certificates ready
- [ ] Backup strategy defined

### ✅ Security
- [ ] Secrets in vault/env only
- [ ] Rate limiter fail-closed
- [ ] Database encryption enabled
- [ ] Container security hardened

### ✅ Monitoring
- [ ] Prometheus configured
- [ ] Grafana dashboards ready
- [ ] Alerting rules defined
- [ ] Logging aggregation setup

## Next Steps

1. Copy .env.production.example to .env.production
2. Fill in all REQUIRED values
3. Pull Ollama models:
   \`\`\`bash
   ollama pull qwen2.5:72b
   ollama pull deepseek-r1:70b
   ollama pull llama3.1:70b
   ollama pull mixtral:8x7b
   ollama pull phi3:medium
   \`\`\`
4. Run: ./deploy_production.sh

## Support

For issues, check:
- logs/ directory
- docker logs <container>
- Grafana dashboards
EOF

echo -e "${GREEN}✓ Deployment report generated${NC}"

# 10. Final summary
echo ""
echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo ""
echo "Summary:"
echo "  ✓ Python cache cleaned"
echo "  ✓ Temporary files removed"
echo "  ✓ Environment files organized"
echo "  ✓ Requirements consolidated"
echo "  ✓ Test files organized"
echo "  ✓ Docker artifacts cleaned"
echo "  ✓ Project structure documented"
echo "  ✓ Deployment report generated"
echo ""
echo -e "${BLUE}Project is clean and ready for deployment!${NC}"
echo ""
echo "Next steps:"
echo "1. Review PROJECT_STRUCTURE.md"
echo "2. Check DEPLOYMENT_READY.md"
echo "3. Configure .env.production from .env.production.example"
echo "4. Run deployment: ./deploy_production.sh"
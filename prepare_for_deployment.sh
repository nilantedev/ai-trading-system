#!/bin/bash

# AI Trading System - Deployment Preparation Script
# ==================================================

echo "🚀 Preparing AI Trading System for Server Deployment"
echo "====================================================="

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Not in the ai-trading-system directory${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Step 1: Verifying core files...${NC}"
required_files=(".env" "requirements.txt" "docker-compose.yml" "Dockerfile")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ✅ $file exists"
    else
        echo -e "  ❌ $file missing"
    fi
done

echo -e "\n${YELLOW}Step 2: Creating deployment package...${NC}"
# Create deployment directory
DEPLOY_DIR="deployment_package_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEPLOY_DIR"

# Files and directories to include in deployment
INCLUDE_FILES=(
    ".env"
    "requirements.txt"
    "pyproject.toml"
    "docker-compose.yml"
    "Dockerfile"
    "alembic.ini"
    "Makefile"
    "README.md"
    "DEPLOYMENT.md"
)

INCLUDE_DIRS=(
    "api"
    "services"
    "shared"
    "config"
    "infrastructure"
    "scripts"
    "migrations"
)

# Copy files
for file in "${INCLUDE_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$DEPLOY_DIR/"
        echo -e "  📦 Copied $file"
    fi
done

# Copy directories
for dir in "${INCLUDE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        cp -r "$dir" "$DEPLOY_DIR/"
        echo -e "  📦 Copied $dir/"
    fi
done

echo -e "\n${YELLOW}Step 3: Creating server setup script...${NC}"
cat > "$DEPLOY_DIR/server_setup.sh" << 'EOF'
#!/bin/bash

echo "🚀 Setting up AI Trading System on Server"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python 3.11+ if not present
if ! python3.11 --version &> /dev/null; then
    echo "Installing Python 3.11..."
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# Install Docker and Docker Compose if not present
if ! docker --version &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Install PostgreSQL client
sudo apt-get install -y postgresql-client

# Create virtual environment
echo "Creating Python virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install trading_common module
echo "Installing trading_common module..."
pip install -e shared/python-common/

# Set up environment
echo "Setting up environment..."
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Please configure it first."
    exit 1
fi

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p models

# Initialize database (if PostgreSQL is running)
echo "Checking database connection..."
if pg_isready -h localhost -p 5432; then
    echo "Database is ready. Running migrations..."
    alembic upgrade head
else
    echo "Database not running. Start it with: docker-compose up -d postgres"
fi

echo "✅ Server setup complete!"
echo ""
echo "Next steps:"
echo "1. Configure your .env file with production values"
echo "2. Start PostgreSQL and Redis: docker-compose up -d postgres redis"
echo "3. Run the API: uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo "4. Or use Docker: docker-compose up -d"
EOF

chmod +x "$DEPLOY_DIR/server_setup.sh"
echo -e "  ✅ Created server_setup.sh"

echo -e "\n${YELLOW}Step 4: Creating deployment checklist...${NC}"
cat > "$DEPLOY_DIR/DEPLOYMENT_CHECKLIST.md" << 'EOF'
# AI Trading System - Deployment Checklist

## Pre-Deployment (Local)
- [x] Fixed AsyncClient test compatibility
- [x] Created .env configuration file
- [x] Installed trading_common module
- [x] Set up data provider API keys
- [x] Basic smoke tests passing

## Server Setup
- [ ] Upload deployment package to server
- [ ] Run server_setup.sh script
- [ ] Configure production .env file with real API keys
- [ ] Set up SSL certificates (if needed)
- [ ] Configure firewall rules

## Database Setup
- [ ] Start PostgreSQL container
- [ ] Run database migrations
- [ ] Verify database connectivity

## Service Deployment
- [ ] Start Redis container
- [ ] Deploy API service
- [ ] Verify health endpoints
- [ ] Set up monitoring (Prometheus/Grafana)

## Post-Deployment Testing
- [ ] Test API endpoints
- [ ] Verify data provider connections
- [ ] Test trading functionality (paper mode)
- [ ] Monitor logs for errors
- [ ] Set up backup procedures

## Production Configuration
- [ ] Replace demo API keys with real ones
- [ ] Configure proper secret management
- [ ] Enable SSL/TLS
- [ ] Set up rate limiting
- [ ] Configure auto-restart on failure

## ML Model Deployment (Phase 2)
- [ ] Install Ollama on server
- [ ] Download and configure models
- [ ] Test model inference
- [ ] Enable advanced ML features
EOF

echo -e "  ✅ Created deployment checklist"

echo -e "\n${YELLOW}Step 5: Compressing deployment package...${NC}"
tar -czf "${DEPLOY_DIR}.tar.gz" "$DEPLOY_DIR"
echo -e "  ✅ Created ${DEPLOY_DIR}.tar.gz"

echo -e "\n${GREEN}=========================================="
echo -e "✅ DEPLOYMENT PACKAGE READY!"
echo -e "==========================================${NC}"
echo ""
echo -e "${YELLOW}Package created:${NC} ${DEPLOY_DIR}.tar.gz"
echo -e "${YELLOW}Size:${NC} $(du -h ${DEPLOY_DIR}.tar.gz | cut -f1)"
echo ""
echo -e "${GREEN}To deploy to your server:${NC}"
echo "1. Copy the package: scp ${DEPLOY_DIR}.tar.gz user@server:/path/to/deploy/"
echo "2. Extract on server: tar -xzf ${DEPLOY_DIR}.tar.gz"
echo "3. Run setup: cd ${DEPLOY_DIR} && ./server_setup.sh"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "- Update .env file with production values on the server"
echo "- Get real API keys from data providers"
echo "- Configure proper database passwords"
echo "- Set up SSL certificates for production"
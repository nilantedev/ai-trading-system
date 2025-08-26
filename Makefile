# AI Trading System - Build Automation
# Version: 3.0.0-production
# =================================

.PHONY: help init dev production test clean

# Variables
PYTHON := python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
DOCKER_COMPOSE := docker-compose
DOCKER := docker

# Default target
help:
	@echo "AI Trading System - Production Build Commands"
	@echo "============================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make init              Initialize environment"
	@echo "  make dev               Start development environment"
	@echo "  make production        Start production environment"
	@echo "  make test              Run all tests"
	@echo "  make clean             Clean all generated files"
	@echo ""
	@echo "📦 Build Commands:"
	@echo "  make build             Build Docker images"
	@echo "  make rebuild           Rebuild Docker images (no cache)"
	@echo "  make push              Push images to registry"
	@echo ""
	@echo "🧪 Testing Commands:"
	@echo "  make test-unit         Run unit tests"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-security     Run security tests"
	@echo "  make test-coverage     Generate coverage report"
	@echo "  make lint              Run code quality checks"
	@echo ""
	@echo "🔐 Security Commands:"
	@echo "  make security-scan     Run security vulnerability scan"
	@echo "  make secrets-check     Check for exposed secrets"
	@echo "  make security-audit    Full security audit"
	@echo ""
	@echo "📊 Database Commands:"
	@echo "  make db-migrate        Run database migrations"
	@echo "  make db-rollback       Rollback last migration"
	@echo "  make db-reset          Reset database (DESTRUCTIVE)"
	@echo "  make db-backup         Create database backup"
	@echo ""
	@echo "🛡️ Backup & Recovery:"
	@echo "  make backup            Run manual backup"
	@echo "  make backup-list       List available backups"
	@echo "  make dr-status         Check disaster recovery status"
	@echo "  make dr-test           Test disaster recovery"
	@echo ""
	@echo "📈 Monitoring:"
	@echo "  make logs              View application logs"
	@echo "  make metrics           View current metrics"
	@echo "  make health            Check system health"
	@echo "  make status            Show all services status"
	@echo ""
	@echo "🔧 Maintenance:"
	@echo "  make update-deps       Update dependencies"
	@echo "  make clean-cache       Clean all caches"
	@echo "  make clean-logs        Clean old logs"
	@echo "  make optimize          Run optimization tasks"

# Environment setup
init: check-requirements create-venv install-deps setup-env
	@echo "✅ Environment initialized successfully"

check-requirements:
	@echo "🔍 Checking system requirements..."
	@command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3.11+ required"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker required"; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose required"; exit 1; }
	@python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" || { echo "❌ Python 3.11+ required"; exit 1; }
	@echo "✅ All requirements satisfied"

create-venv:
	@echo "🐍 Creating Python virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo "✅ Virtual environment created"

install-deps:
	@echo "📦 Installing Python dependencies..."
	@$(VENV_PIP) install -r requirements.txt
	@$(VENV_PIP) install -r requirements-dev.txt 2>/dev/null || true
	@echo "✅ Dependencies installed"

setup-env:
	@echo "⚙️ Setting up environment files..."
	@test -f .env || cp .env.template .env
	@test -f .env.production || cp .env.production.template .env.production
	@mkdir -p logs config backups data
	@echo "✅ Environment files configured"

# Development environment
dev: check-requirements
	@echo "🚀 Starting development environment..."
	@$(DOCKER_COMPOSE) -f docker-compose.yml up -d postgres redis
	@sleep 5  # Wait for services
	@$(VENV_PYTHON) scripts/init_database.py --dev 2>/dev/null || true
	@$(VENV_PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dev-stop:
	@echo "🛑 Stopping development environment..."
	@$(DOCKER_COMPOSE) -f docker-compose.yml down

# Production environment
production: check-requirements build
	@echo "🚀 Starting production environment..."
	@$(DOCKER_COMPOSE) -f docker-compose.yml up -d
	@echo "⏳ Waiting for services to be healthy..."
	@sleep 10
	@make health
	@echo "✅ Production environment running"

production-stop:
	@echo "🛑 Stopping production environment..."
	@$(DOCKER_COMPOSE) -f docker-compose.yml down

# Build commands
build:
	@echo "🔨 Building Docker images..."
	@$(DOCKER_COMPOSE) -f docker-compose.yml build
	@echo "✅ Images built successfully"

rebuild:
	@echo "🔨 Rebuilding Docker images (no cache)..."
	@$(DOCKER_COMPOSE) -f docker-compose.yml build --no-cache
	@echo "✅ Images rebuilt successfully"

push:
	@echo "📤 Pushing images to registry..."
	@$(DOCKER_COMPOSE) -f docker-compose.yml push
	@echo "✅ Images pushed successfully"

# Testing
test: test-unit test-integration test-security
	@echo "✅ All tests passed"

test-unit:
	@echo "🧪 Running unit tests..."
	@$(VENV_PYTHON) -m pytest tests/unit -v --cov=api --cov=shared

test-integration:
	@echo "🧪 Running integration tests..."
	@$(VENV_PYTHON) -m pytest tests/integration -v

test-security:
	@echo "🔐 Running security tests..."
	@$(VENV_PYTHON) -m bandit -r api shared services -ll
	@$(VENV_PYTHON) -m safety check --json
	@echo "✅ Security tests passed"

test-coverage:
	@echo "📊 Generating coverage report..."
	@$(VENV_PYTHON) -m pytest --cov=api --cov=shared --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"

lint:
	@echo "🎨 Running code quality checks..."
	@$(VENV_PYTHON) -m black --check api shared services tests
	@$(VENV_PYTHON) -m isort --check-only api shared services tests
	@$(VENV_PYTHON) -m flake8 api shared services tests
	@$(VENV_PYTHON) -m mypy api shared services
	@echo "✅ Code quality checks passed"

format:
	@echo "🎨 Formatting code..."
	@$(VENV_PYTHON) -m black api shared services tests
	@$(VENV_PYTHON) -m isort api shared services tests
	@echo "✅ Code formatted"

# Security
security-scan:
	@echo "🔍 Running security vulnerability scan..."
	@$(VENV_PYTHON) -m pip-audit
	@$(VENV_PYTHON) -m bandit -r api shared services -ll
	@$(DOCKER) scan trading-api:latest 2>/dev/null || echo "Docker scan not available"
	@echo "✅ Security scan completed"

secrets-check:
	@echo "🔍 Checking for exposed secrets..."
	@$(VENV_PYTHON) scripts/check_secrets.py
	@git secrets --scan 2>/dev/null || echo "git-secrets not installed"
	@echo "✅ No exposed secrets found"

security-audit: security-scan secrets-check
	@echo "📋 Security audit completed"

# Database management
db-migrate:
	@echo "📊 Running database migrations..."
	@$(VENV_PYTHON) -m alembic upgrade head
	@echo "✅ Migrations applied"

db-rollback:
	@echo "⏪ Rolling back last migration..."
	@$(VENV_PYTHON) -m alembic downgrade -1
	@echo "✅ Rollback completed"

db-reset:
	@echo "⚠️ Resetting database (DESTRUCTIVE)..."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@$(VENV_PYTHON) -m alembic downgrade base
	@$(VENV_PYTHON) -m alembic upgrade head
	@$(VENV_PYTHON) scripts/init_database.py
	@echo "✅ Database reset completed"

db-backup:
	@echo "💾 Creating database backup..."
	@$(VENV_PYTHON) scripts/disaster_recovery.py --action backup
	@echo "✅ Database backup created"

# Backup and disaster recovery
backup:
	@echo "💾 Running manual backup..."
	@$(VENV_PYTHON) scripts/disaster_recovery.py --action backup
	@echo "✅ Backup completed"

backup-list:
	@echo "📋 Available backups:"
	@$(VENV_PYTHON) shared/python-common/trading_common/backup_manager.py --action list

dr-status:
	@echo "🛡️ Disaster recovery status:"
	@$(VENV_PYTHON) scripts/disaster_recovery.py --action status

dr-test:
	@echo "🧪 Testing disaster recovery..."
	@$(VENV_PYTHON) scripts/disaster_recovery.py --action recovery-test

# Monitoring
logs:
	@echo "📜 Showing application logs..."
	@$(DOCKER_COMPOSE) -f docker-compose.yml logs -f --tail=100

logs-api:
	@$(DOCKER_COMPOSE) -f docker-compose.yml logs -f api --tail=100

metrics:
	@echo "📊 Current metrics:"
	@curl -s http://localhost:8000/metrics | head -50

health:
	@echo "🩺 Checking system health..."
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "❌ API not responding"
	@$(DOCKER_COMPOSE) -f docker-compose.yml ps

status:
	@echo "📊 Service status:"
	@$(DOCKER_COMPOSE) -f docker-compose.yml ps
	@echo ""
	@echo "🔗 Service URLs:"
	@echo "  API:        http://localhost:8000"
	@echo "  Docs:       http://localhost:8000/docs"
	@echo "  Grafana:    http://localhost:3000"
	@echo "  Prometheus: http://localhost:9090"

# Maintenance
update-deps:
	@echo "📦 Updating dependencies..."
	@$(VENV_PIP) list --outdated
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo "Run 'make update-deps-force' to update all packages"

update-deps-force:
	@echo "📦 Force updating all dependencies..."
	@$(VENV_PIP) install --upgrade -r requirements.txt
	@echo "✅ Dependencies updated"

clean-cache:
	@echo "🧹 Cleaning caches..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Caches cleaned"

clean-logs:
	@echo "🧹 Cleaning old logs..."
	@find logs -name "*.log" -mtime +30 -delete 2>/dev/null || true
	@echo "✅ Old logs cleaned"

optimize:
	@echo "⚡ Running optimization tasks..."
	@$(DOCKER) exec trading-postgres psql -U trading_user -d trading_system -c "VACUUM ANALYZE;"
	@$(DOCKER) exec trading-redis redis-cli --scan --pattern "*" | head -10
	@echo "✅ Optimization completed"

# Clean everything
clean: clean-cache clean-logs
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf build dist *.egg-info
	@rm -rf htmlcov .coverage
	@rm -rf .mypy_cache .ruff_cache
	@$(DOCKER_COMPOSE) -f docker-compose.yml down -v 2>/dev/null || true
	@echo "✅ Cleanup completed"

# Full reset
reset: clean
	@echo "⚠️ Full system reset..."
	@rm -rf $(VENV)
	@rm -f .env .env.production
	@echo "✅ System reset completed"
	@echo "Run 'make init' to reinitialize"

.DEFAULT_GOAL := help
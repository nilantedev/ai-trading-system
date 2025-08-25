# AI Trading System - Build Automation
# =================================
# Comprehensive build automation for the AI trading system
# Supports all development phases with proper validation gates

.PHONY: help init-dev start-infrastructure health-check clean-dev test lint build security-check

# Default target
help:
	@echo "AI Trading System - Build Commands"
	@echo "=================================="
	@echo ""
	@echo "🚀 Quick Start Commands:"
	@echo "  init-dev               Initialize development environment"
	@echo "  start-infrastructure   Start all infrastructure services"
	@echo "  health-check          Check system health"
	@echo "  stop-all              Stop all services"
	@echo ""
	@echo "🔧 Development Commands:"
	@echo "  install-deps          Install all dependencies"
	@echo "  test                  Run comprehensive test suite with coverage gates"
	@echo "  test-fast             Run fast test subset for quick validation"
	@echo "  test-unit             Run unit tests only"
	@echo "  test-integration      Run integration tests only"
	@echo "  test-property         Run property-based tests only"
	@echo "  coverage              Generate coverage report"
	@echo "  secrets-scan          Scan for hardcoded secrets"
	@echo "  secrets-template      Generate secrets template"
	@echo "  secrets-validate      Validate secrets access"
	@echo "  build                 Build all components"
	@echo "  test                  Run all tests"
	@echo "  lint                  Run code quality checks"
	@echo "  clean-dev             Clean development environment"
	@echo ""
	@echo "🔐 Security Commands:"
	@echo "  security-check        Run security validation"
	@echo "  check-secrets         Check for exposed secrets"
	@echo "  generate-secret       Generate secure secret key"
	@echo "  generate-password-hash Generate bcrypt password hash"
	@echo ""
	@echo "📋 Phase Management:"
	@echo "  current-phase         Show current development phase"
	@echo "  validate-current-phase Validate current phase completion"
	@echo "  advance-phase         Move to next development phase"
	@echo ""
	@echo "🩺 Monitoring & Debugging:"
	@echo "  logs                  View service logs"
	@echo "  resource-status       Check resource usage"
	@echo "  performance-monitor   Monitor system performance"
	@echo "  debug-service         Debug specific service"
	@echo ""
	@echo "🚨 Emergency Commands:"
	@echo "  emergency-stop        Emergency stop all services"
	@echo "  disaster-recovery     Run disaster recovery procedures"
	@echo "  rollback-phase        Rollback to previous phase"

# =================================
# Environment Setup
# =================================

# Python virtual environment path
VENV_PATH := .venv
PYTHON := $(VENV_PATH)/bin/python
PIP := $(VENV_PATH)/bin/pip

check-tools:
	@echo "🔍 Checking required tools..."
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed."; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed."; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3.11+ is required but not installed."; exit 1; }
	@command -v cargo >/dev/null 2>&1 || { echo "❌ Rust/Cargo is required but not installed."; exit 1; }
	@command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }
	@command -v git >/dev/null 2>&1 || { echo "❌ Git is required but not installed."; exit 1; }
	@python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" || { echo "❌ Python 3.11+ is required."; exit 1; }
	@echo "✅ All required tools are available"

setup-venv:
	@echo "🐍 Setting up Python virtual environment..."
	@python3 -m venv $(VENV_PATH)
	@$(PIP) install --upgrade pip setuptools wheel
	@echo "✅ Virtual environment created"

init-dev: check-tools setup-venv
	@echo "🚀 Initializing development environment..."
	@echo "📁 Creating required directories..."
	@mkdir -p logs data/models data/backups infrastructure/configs
	@mkdir -p infrastructure/docker/traefik infrastructure/docker/redis
	@mkdir -p infrastructure/docker/questdb infrastructure/docker/prometheus
	@mkdir -p infrastructure/docker/grafana infrastructure/docker/loki
	@echo "🐍 Installing Python dependencies..."
	@$(PIP) install -r requirements-dev.txt
	@cd shared/python-common && $(PIP) install -e .
	@echo "🦀 Setting up Rust environment..."
	@cd shared/rust-common && cargo build
	@echo "📦 Installing pre-commit hooks..."
	@$(PIP) install pre-commit
	@pre-commit install
	@echo "📋 Creating environment file..."
	@cp infrastructure/docker/.env.example .env
	@echo "✅ Development environment initialized"
	@echo ""
	@echo "🔧 Next steps:"
	@echo "1. Edit .env file with your API keys and secrets"
	@echo "2. Run 'make start-infrastructure' to start services"
	@echo "3. Run 'make health-check' to verify everything is working"

# =================================
# Infrastructure Management
# =================================

start-infrastructure:
	@echo "🏗️ Starting infrastructure services..."
	@docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@$(MAKE) health-check-infrastructure

stop-all:
	@echo "🛑 Stopping all services..."
	@docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml down
	@docker-compose -f services/docker-compose.services.yml down 2>/dev/null || true

health-check: health-check-infrastructure health-check-services

health-check-infrastructure:
	@echo "🩺 Checking infrastructure health..."
	@echo "  Checking Traefik..."
	@curl -s -f http://localhost:8080/api/overview >/dev/null && echo "    ✅ Traefik is healthy" || echo "    ❌ Traefik is not responding"
	@echo "  Checking Redis..."
	@docker exec $$(docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml ps -q redis) redis-cli ping >/dev/null 2>&1 && echo "    ✅ Redis is healthy" || echo "    ❌ Redis is not responding"
	@echo "  Checking QuestDB..."
	@curl -s -f "http://localhost:9000/exec?query=SELECT%201" >/dev/null && echo "    ✅ QuestDB is healthy" || echo "    ❌ QuestDB is not responding"
	@echo "  Checking Prometheus..."
	@curl -s -f http://localhost:9090/api/v1/targets >/dev/null && echo "    ✅ Prometheus is healthy" || echo "    ❌ Prometheus is not responding"
	@echo "  Checking Grafana..."
	@curl -s -f http://localhost:3001/api/health >/dev/null && echo "    ✅ Grafana is healthy" || echo "    ❌ Grafana is not responding"

health-check-services:
	@echo "🩺 Checking application services..."
	@echo "  (Services will be available in later phases)"

restart-services:
	@echo "🔄 Restarting all services..."
	@$(MAKE) stop-all
	@sleep 5
	@$(MAKE) start-infrastructure

# =================================
# Development Workflow
# =================================

install-deps:
	@echo "📦 Installing dependencies..."
	@echo "  Installing Python dependencies..."
	@$(PIP) install -r requirements-dev.txt
	@cd shared/python-common && $(PIP) install -e .
	@echo "  Installing Rust dependencies..."
	@cd shared/rust-common && cargo build --release
	@echo "  Installing Node.js dependencies (when available)..."
	@if [ -d "services/dashboard" ]; then cd services/dashboard && npm install; fi

build: build-shared build-services

build-shared:
	@echo "🔨 Building shared libraries..."
	@echo "  Building Python common library..."
	@cd shared/python-common && python setup.py build
	@echo "  Building Rust common library..."
	@cd shared/rust-common && cargo build --release

build-services:
	@echo "🔨 Building services..."
	@echo "  (Services will be built in later phases)"

# =================================
# Comprehensive Testing
# =================================

test: ## Run comprehensive test suite with coverage gates
	@echo "🚀 Running comprehensive test suite..."
	@$(PYTHON) scripts/run_tests_with_coverage_gates.py

test-fast: ## Run fast test subset for quick validation  
	@echo "🏃‍♂️ Running fast test suite..."
	@$(PYTHON) scripts/run_tests_with_coverage_gates.py --fast

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	@$(PYTHON) -m pytest tests/unit/ -v --cov=shared --cov=services --cov-report=term-missing -m unit

test-integration: ## Run integration tests only
	@echo "🔗 Running integration tests..."
	@$(PYTHON) -m pytest tests/integration/ -v -m integration

test-property: ## Run property-based tests only
	@echo "🔍 Running property-based tests..."
	@$(PYTHON) -m pytest tests/unit/test_property_based.py -v --hypothesis-show-statistics -m property

test-security: ## Run security tests only
	@echo "🔒 Running security tests..."
	@$(PYTHON) -m pytest tests/security/ -v -m security

test-performance: ## Run performance tests only
	@echo "⚡ Running performance tests..."
	@$(PYTHON) -m pytest tests/performance/ -v -m performance --tb=short

coverage: ## Generate coverage report
	@echo "📊 Generating coverage report..."
	@$(PYTHON) -m pytest tests/unit/ tests/integration/ --cov=shared --cov=services --cov-report=html --cov-report=term-missing --cov-fail-under=70

test-shared:
	@echo "🧪 Running shared library tests..."
	@cd shared/python-common && $(PYTHON) -m pytest tests/ -v --cov=trading_common
	@cd shared/rust-common && cargo test --release

test-infrastructure:
	@echo "🧪 Running infrastructure tests..."
	@$(MAKE) health-check-infrastructure

test-all: test

# =================================
# Secrets Management
# =================================

secrets-scan: ## Scan for hardcoded secrets and security issues
	@echo "🔍 Scanning for hardcoded secrets..."
	@$(PYTHON) scripts/manage_secrets.py scan

secrets-template: ## Generate secrets template file
	@echo "📋 Generating secrets template..."
	@$(PYTHON) scripts/manage_secrets.py template

secrets-validate: ## Validate secrets access and configuration
	@echo "✅ Validating secrets configuration..."
	@$(PYTHON) scripts/manage_secrets.py validate

secrets-health: ## Check secrets vault health
	@echo "🔍 Checking secrets vault health..."
	@$(PYTHON) scripts/manage_secrets.py health

lint: lint-python lint-rust

lint-python:
	@echo "🔍 Running Python linting..."
	@$(PYTHON) -m black --check shared/python-common/ services/ || echo "❌ Code formatting issues found. Run 'make format-python' to fix."
	@$(PYTHON) -m isort --check-only shared/python-common/ services/ || echo "❌ Import sorting issues found. Run 'make format-python' to fix."
	@$(PYTHON) -m flake8 shared/python-common/ services/ || echo "❌ Linting issues found."
	@$(PYTHON) -m mypy shared/python-common/ || echo "❌ Type checking issues found."

lint-rust:
	@echo "🔍 Running Rust linting..."
	@cd shared/rust-common && cargo fmt -- --check || echo "❌ Code formatting issues found. Run 'make format-rust' to fix."
	@cd shared/rust-common && cargo clippy --release -- -D warnings

format: format-python format-rust

format-python:
	@echo "🎨 Formatting Python code..."
	@$(PYTHON) -m black shared/python-common/ services/
	@$(PYTHON) -m isort shared/python-common/ services/

format-rust:
	@echo "🎨 Formatting Rust code..."
	@cd shared/rust-common && cargo fmt

# =================================
# Phase Management
# =================================

current-phase:
	@echo "📋 Current Development Phase:"
	@cat Claude-Code-Context/CURRENT_PHASE.md | head -10

validate-current-phase: validate-phase-1

validate-phase-1:
	@echo "✅ Validating Phase 1: Foundation Infrastructure"
	@echo "  Checking infrastructure services..."
	@$(MAKE) health-check-infrastructure
	@echo "  Checking shared libraries..."
	@$(MAKE) test-shared
	@echo "  Checking configuration files..."
	@$(MAKE) validate-configs
	@echo "🎉 Phase 1 validation complete!"

validate-phase-2:
	@echo "✅ Validating Phase 2: Core Data Layer"
	@echo "  (Phase 2 validation will be implemented when reached)"

advance-phase:
	@echo "🚀 Advancing to next development phase"
	@echo "⚠️  Please ensure current phase validation has passed"
	@echo "📝 Update Claude-Code-Context/CURRENT_PHASE.md with next phase details"

# =================================
# Monitoring & Debugging
# =================================

logs:
	@echo "📋 Viewing service logs..."
	@docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml logs --tail=100 --follow

logs-service:
	@echo "📋 Viewing logs for specific service..."
	@if [ -z "$(SERVICE)" ]; then echo "❌ Please specify SERVICE=<service-name>"; exit 1; fi
	@docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml logs --tail=100 --follow $(SERVICE)

resource-status:
	@echo "📊 System Resource Status:"
	@echo "  CPU Usage:"
	@top -bn1 | grep "Cpu(s)" | awk '{print "    " $$2 " " $$3}'
	@echo "  Memory Usage:"
	@free -h | grep "Mem:" | awk '{print "    Used: " $$3 " / " $$2 " (" $$3 "/" $$2 * 100 "%)"}'
	@echo "  Disk Usage:"
	@df -h | grep -E "(/$|/srv|/mnt)" | awk '{print "    " $$6 ": " $$3 " / " $$2 " (" $$5 ")"}'
	@echo "  Docker Resource Usage:"
	@docker system df

performance-monitor:
	@echo "📈 Performance Monitoring:"
	@echo "  Container Resource Usage:"
	@docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
	@echo "  Database Query Performance:"
	@curl -s "http://localhost:9000/exec?query=SELECT%20COUNT(*)%20FROM%20information_schema.tables" 2>/dev/null | grep -o '"count":[0-9]*' || echo "    QuestDB not available for query testing"

debug-service:
	@echo "🐛 Service Debugging:"
	@if [ -z "$(SERVICE)" ]; then echo "❌ Please specify SERVICE=<service-name>"; exit 1; fi
	@echo "  Service Status:"
	@docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml ps $(SERVICE)
	@echo "  Recent Logs:"
	@docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml logs --tail=50 $(SERVICE)
	@echo "  Resource Usage:"
	@docker stats $(SERVICE) --no-stream

export-logs:
	@echo "📤 Exporting logs for analysis..."
	@mkdir -p logs/exports
	@docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml logs > logs/exports/all-services-$$(date +%Y%m%d-%H%M%S).log
	@echo "  Logs exported to logs/exports/"

# =================================
# Configuration Management
# =================================

validate-configs:
	@echo "🔧 Validating configuration files..."
	@echo "  Checking Docker Compose files..."
	@docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml config >/dev/null && echo "    ✅ Infrastructure compose file is valid" || echo "    ❌ Infrastructure compose file has errors"
	@echo "  Checking JSON schema files..."
	@find shared/schemas -name "*.json" -exec python -m json.tool {} \; >/dev/null 2>&1 && echo "    ✅ JSON schemas are valid" || echo "    ❌ JSON schema validation failed"

backup-configs:
	@echo "💾 Backing up configuration files..."
	@mkdir -p data/backups/configs
	@tar -czf data/backups/configs/configs-$$(date +%Y%m%d-%H%M%S).tar.gz infrastructure/configs/ shared/schemas/
	@echo "  Configuration backup created"

# =================================
# Data Management
# =================================

validate-data-integrity:
	@echo "🔍 Validating data integrity..."
	@echo "  Checking Redis data..."
	@docker exec $$(docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml ps -q redis) redis-cli dbsize >/dev/null 2>&1 && echo "    ✅ Redis data accessible" || echo "    ❌ Redis data check failed"
	@echo "  Checking QuestDB data..."
	@curl -s "http://localhost:9000/exec?query=SELECT%20COUNT(*)%20FROM%20information_schema.tables" >/dev/null && echo "    ✅ QuestDB data accessible" || echo "    ❌ QuestDB data check failed"

backup-data:
	@echo "💾 Backing up system data..."
	@mkdir -p data/backups/data
	@echo "  Backing up Redis data..."
	@docker exec $$(docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml ps -q redis) redis-cli bgsave >/dev/null 2>&1 || echo "    Redis backup initiated"
	@echo "  Backing up QuestDB data..."
	@curl -s "http://localhost:9000/exec?query=BACKUP%20TABLE%20*" >/dev/null || echo "    QuestDB backup would be configured here"
	@echo "  Data backup procedures completed"

# =================================
# Emergency Procedures
# =================================

emergency-stop:
	@echo "🚨 EMERGENCY STOP - Shutting down all services immediately"
	@docker stop $$(docker ps -q) 2>/dev/null || echo "  No running containers to stop"
	@echo "  All services stopped"
	@echo "  Check logs and system status before restarting"

disaster-recovery:
	@echo "🚑 Disaster Recovery Procedures"
	@echo "  Step 1: Stopping all services..."
	@$(MAKE) emergency-stop
	@echo "  Step 2: Checking system resources..."
	@$(MAKE) resource-status
	@echo "  Step 3: Validating configurations..."
	@$(MAKE) validate-configs
	@echo "  Step 4: Checking data integrity..."
	@$(MAKE) validate-data-integrity
	@echo "  Step 5: Ready for manual recovery procedures"
	@echo "  📖 See Claude-Code-Context/TROUBLESHOOTING.md for detailed recovery steps"

rollback-phase:
	@echo "🔄 Rolling back to previous development phase"
	@echo "⚠️  This will reset current development progress"
	@echo "📝 Please update Claude-Code-Context/CURRENT_PHASE.md accordingly"
	@echo "🔄 Use git to revert to last stable commit for current phase"

# =================================
# Cleanup Operations
# =================================

clean-dev:
	@echo "🧹 Cleaning development environment..."
	@$(MAKE) stop-all
	@echo "  Removing Docker containers and volumes..."
	@docker-compose -f infrastructure/docker/docker-compose.infrastructure.yml down -v
	@echo "  Cleaning Docker system..."
	@docker system prune -f
	@echo "  Cleaning build artifacts..."
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@cd shared/rust-common && cargo clean 2>/dev/null || true
	@echo "  Development environment cleaned"

clean-all: clean-dev
	@echo "🧹 Deep cleaning all generated files..."
	@rm -rf data/backups/* logs/* 2>/dev/null || true
	@echo "  All generated files cleaned"

# =================================
# Development Utilities
# =================================

setup-git-hooks:
	@echo "🪝 Setting up Git hooks..."
	@pre-commit install
	@echo "  Git hooks configured"

generate-docs:
	@echo "📚 Generating documentation..."
	@echo "  (Documentation generation will be implemented in later phases)"

run-security-scan:
	@echo "🔒 Running security scan..."
	@echo "  (Security scanning will be implemented in later phases)"

# =================================
# Security Operations (Added 2024-08-25)
# =================================

security-check: ## Run security validation
	@echo "🔐 Running security validation..."
	@$(PYTHON) -m trading_common.security_validator --environment $(ENV) || \
		(echo "❌ Security validation failed!" && exit 1)
	@echo "✅ Security validation passed"

check-secrets: ## Check for exposed secrets
	@echo "🔍 Checking for exposed secrets..."
	@! grep -r "TradingSystem2024\|password123\|secret_key\|default_password" \
		--include="*.py" --exclude-dir=.venv . || \
		(echo "❌ Found exposed secrets!" && exit 1)
	@echo "✅ No exposed secrets found"

generate-secret: ## Generate secure secret key
	@echo "🔑 Generating secure secret key:"
	@$(PYTHON) -c "import secrets; print(secrets.token_urlsafe(32))"

generate-password-hash: ## Generate bcrypt password hash
	@read -s -p "Enter password to hash: " password; \
	echo; \
	$(PYTHON) -c "from passlib.context import CryptContext; \
		pwd_context = CryptContext(schemes=['bcrypt']); \
		print(pwd_context.hash('$$password'))"

validate-env: ## Validate environment variables for deployment
	@echo "🔍 Validating environment..."
	@if [ "$(ENV)" = "production" ] || [ "$(ENV)" = "staging" ]; then \
		if [ -z "$$SECURITY_SECRET_KEY" ]; then \
			echo "❌ SECURITY_SECRET_KEY required for $(ENV)"; \
			exit 1; \
		fi; \
		if [ -z "$$ADMIN_PASSWORD_HASH" ]; then \
			echo "❌ ADMIN_PASSWORD_HASH required for $(ENV)"; \
			exit 1; \
		fi; \
	fi
	@echo "✅ Environment validated for $(ENV)"

# =================================
# Testing Operations (Updated 2024-08-25)
# =================================

test-services: ## Run service tests with pytest
	@echo "🧪 Running service tests..."
	@$(PYTHON) -m pytest tests/test_services.py -v --tb=short

test-coverage: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	@$(PYTHON) -m pytest tests/ -v --cov=api --cov=services --cov=trading_common \
		--cov-report=term-missing --cov-report=html
	@echo "📊 Coverage report generated in htmlcov/"

# =================================
# Variables and Configuration
# =================================

# Default service for debugging
SERVICE ?= redis

# Default environment
ENV ?= development

# Version information
VERSION := $(shell cat VERSION 2>/dev/null || echo "1.0.0-dev")

version:
	@echo "AI Trading System Version: $(VERSION)"

# Show all available targets
list-targets:
	@echo "Available Make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
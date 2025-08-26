# Development Environment Setup

This guide helps you set up a clean, reproducible development environment for the AI Trading System.

## 1. Prerequisites
- Python 3.11.x
- PostgreSQL 14+ (optional for core dev; required for full services)
- Redis 6+ (for rate limiting / security store)
- Make (optional convenience)
- Git

## 2. Clone Repository
```bash
git clone https://github.com/nilantedev/ai-trading-system.git
cd ai-trading-system
```

## 3. Create Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install Dependencies
Base (runtime + dev tooling):
```bash
pip install -e .[dev]
```
Optional groups:
```bash
pip install .[ai]
pip install .[trading]
```

If you only need a minimal footprint:
```bash
pip install -r requirements-minimal.txt
```

## 5. Internal Package `trading_common`
Some modules reference an internal package `trading_common` used in a monorepo layout. If it is not yet published or installed, add it to PYTHONPATH during development:
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```
Or create a lightweight editable package stub if located elsewhere.

mypy is configured to ignore missing imports for `trading_common.*` until the package is fully integrated.

## 6. Environment Variables
Create a `.env` file at the repo root (never commit secrets):
```
ENVIRONMENT=development
JWT_SECRET=dev-secret-change-me
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/trading
REDIS_URL=redis://localhost:6379/0
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
TRUSTED_HOSTS=localhost,127.0.0.1
```

## 7. Running the API
```bash
uvicorn api.main:app --reload --port 8000
```
Or via the built-in block in `api/main.py`:
```bash
python api/main.py
```

## 8. Code Quality Tooling
Run linters & type checks:
```bash
black .
isort .
flake8
mypy .
```

## 9. Tests
Current tests are being refactored. Run existing tests:
```bash
pytest -q --cov
```

## 10. Metrics & Observability
- Metrics endpoint: `/metrics`
- Health: `/health`
- Readiness: `/ready`
- ML Status: `/api/v1/ml/status`

## 11. Common Issues
| Problem | Cause | Fix |
|---------|-------|-----|
| Unresolved import `trading_common.*` | Package not installed | Add to PYTHONPATH or install package location editable |
| Broad exception warnings | Intentional during phased startup | Will be narrowed iteratively |
| Missing Redis/Postgres during startup | Optional in dev | Provide service or ignore if not needed |

## 12. Next Steps
- Improve test coverage to >=80%
- Add OpenTelemetry tracing configuration
- Harden exception scopes in startup phases

---
For any issues see `docs/TROUBLESHOOTING.md` or open an issue in the repository.

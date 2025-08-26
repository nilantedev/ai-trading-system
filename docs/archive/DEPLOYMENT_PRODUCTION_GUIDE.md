# Production Deployment Guide

This guide summarizes the minimal steps and configurations to deploy the AI Trading System API + ML intelligence layer to production.

## 1. Runtime Components
Required services:
- API Application: FastAPI (uvicorn or gunicorn+uvicorn workers)
- PostgreSQL: primary relational store (features, models, experiments, drift reports)
- Redis: rate limiting, security store, caching
- (Optional) Prometheus: scrape /metrics endpoint
- (Optional) Grafana: dashboards

## 2. Environment Variables (Sample)
```
ENVIRONMENT=production
PORT=8080
WORKERS=4
DRIFT_INTERVAL_SECONDS=3600
DEFAULT_FEATURE_VIEW=core_technical
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/trading
REDIS_URL=redis://redis:6379/0
JWT_SECRET=CHANGE_ME
ALLOWED_ORIGINS=https://yourdomain.com
TRUSTED_HOSTS=yourdomain.com
```

## 3. Startup Flow Summary
1. Security validation (`security_validator`) – fails fast if critical misconfigurations.
2. Security store init (Redis or DB fallbacks).
3. Middleware: metrics -> rate limiting -> auth (decorator order yields correct stack).
4. WebSocket streaming (non-fatal failure in prod).
5. ML Wiring:
   - Drift monitor started (interval configurable) with currently PRODUCTION models.
   - Default feature view registration + materialization if absent.
   - Startup banner logs counts (features, views, drift monitor status).

## 4. Health & Observability
Endpoints:
- `/health` basic health + circuit breaker snapshot.
- `/api/v1/health` extended service matrix.
- `/api/v1/ml/status` ML layer status (feature counts, drift monitor, experiment tracker).
- `/metrics` Prometheus exposition (latency, errors, domain counters).

## 5. Scaling Guidance
- Stateless API workers: scale horizontally behind load balancer.
- Ensure single drift monitor instance per model (current implementation is cooperative; deploy only one background-enabled replica or add leader election later).
- Database connection pool sizing: (#workers * average concurrent queries) <= max_connections * safety_factor.
- Redis: configure eviction policy (volatile-lru) for rate limit keys.

## 6. Security Hardening Checklist
- Enforce HTTPS + HSTS (already auto-enabled in production).
- Rotate JWT secret and store externally (Vault / Key Vault) – not committed.
- Set stricter CORS origins (avoid wildcard).
- Enable DB at-rest encryption and daily backups.
- Add WAF (e.g., Cloudflare / AWS ALB rules) in front of API endpoints.

## 7. Logging & Monitoring
- Structure logs (JSON) via logging config for ingestion (ELK / Loki).
- Prometheus scrape interval: 15s typical.
- Add custom alerts: high 5xx rate, drift severity HIGH, feature quality score < threshold.

## 8. Deployment Patterns
### Container (Recommended)
Use a Dockerfile (to be added) similar to:
```
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install . --no-cache-dir
COPY . .
ENV ENVIRONMENT=production
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```
Add build args for version stamping if desired.

### Process Manager
If running directly on VM:
- Use `gunicorn -k uvicorn.workers.UvicornWorker -w 4 api.main:app` (prefork reliability, graceful restarts).

## 9. Data Migration
Run Alembic migrations (if configured) before starting application. Feature store + registry create tables on-demand; prefer to codify them into migrations for reproducibility.

## 10. Post-Deployment Validation
1. `GET /health` returns healthy.
2. `GET /api/v1/ml/status` shows feature + view counts > 0.
3. `GET /metrics` exposes process + application metrics.
4. Drift monitor logs appear within first interval showing `Drift scan complete` or `no models`.
5. Execute minimal experiment tracking run in staging to confirm tables.

## 11. Known Deferred Items (See NON_GOALS.md)
- Automated retraining
- Full MLflow integration
- Distributed feature computation
- Lineage visualization UI
- Experiment permissioning

## 12. Future Hardening Roadmap
- Add leader election for background tasks (Redis-based lock).
- Integrate structured audit logging.
- Externalize model promotion policy configuration.
- Add SLA/SLO metrics (e.g., request success percentile latency distributions).

---
Document version: 1.0.0 | Last Updated: 2025-08-26

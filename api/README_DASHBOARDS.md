# Dashboards (Admin & Business)

This document describes the architecture, security posture, and extensibility model for the Mekoshi (mekoshi.com) dashboards integrated into the API service.

## Overview
Two dashboards have been added:
- **Admin Dashboard** (`/admin`): Deep operational visibility (system summary, trading performance placeholders, model status, runtime metrics aggregation).
- **Business Dashboard** (`/business`): Business KPIs, company list, and per-company profile pages with forecast/report placeholders.

All pages are server-side rendered via Jinja2 templates with client-side hydration fetching JSON endpoints.

## Security Posture
- **RBAC / Hard Restriction:** Both Admin and Business dashboards are now restricted to the single admin user `nilante` (enforced post role check). All dashboard HTML + JSON/SSE endpoints require an authenticated `admin` role token AND username match. (See `require_roles_cookie_or_bearer` + single-user guards in routers.)
- **CSP:** Strict Content Security Policy with per-request nonce. Only self-hosted scripts/styles and nonce-bound script tags.
- **Headers:** HSTS (production), X-Frame-Options=DENY, X-Content-Type-Options=nosniff, Referrer-Policy, Permissions-Policy minimized, X-Download-Options, X-Permitted-Cross-Domain-Policies.
- **CSRF Defense:** HttpOnly access/refresh cookies are paired with a separate `csrf_token` (non-HttpOnly) cookie. Mutating `fetch` requests (non-GET/HEAD/OPTIONS) must send header `X-CSRF-Token` = cookie value. Middleware enforces token presence & equality; auth endpoints and safe SSE streams exempted.
- **Silent Refresh:** A lightweight script (`static/js/auth_refresh.js`) renews the access token before expiry (interval jittered) by calling `/auth/refresh`. Failures trigger a single retry; persistent failures allow natural logout.
- **MFA Hooks:** Optional TOTP verification path for admin accounts; stored secrets isolated in Redis.
- **Deterministic Placeholder Data:** Prevents information leakage & supports stable diffing during audits.
- **No Inline Unsanitized Data:** Templates avoid embedding raw JSON; all dynamic large data fetched via credentialed `fetch()`.

## Directory Structure
```
api/
  templates/
    base.html
    admin/dashboard.html
    business/dashboard.html
    business/company_profile.html
  static/
    css/dashboard.css
    js/admin_dashboard.js
    js/business_dashboard.js
    js/company_profile.js
```

## Jinja2 Environment
Initialized in `api/main.py` during module import. If the templates directory is missing, the system logs a warning but continues serving API endpoints.

## JSON Endpoints (Placeholders + Enhanced)
- `/admin/api/system/summary`
- `/admin/api/trading/performance`
- `/admin/api/models/status`
- `/admin/api/metrics/summary` (scrapes subset of Prometheus metrics locally)
- `/business/api/kpis`
- `/business/api/companies`
- `/business/api/company/{symbol}/forecast`
- `/business/api/company/{symbol}/report`
- `/business/api/company/{symbol}/sparkline` (deterministic pseudo price series for small charts)
- `/business/api/coverage/summary` (equities/options coverage ratios; now also includes `history: [{ts, equities, options}]` for charting)
- `/business/api/ingestion/health` (latest ingestion timestamps & lag placeholders)
- `/business/api/news/sentiment` (1d vs 7d average sentiment + anomaly delta + `history: [{date, avg_sentiment}]` 30d daily averages)
- `/admin/api/latency/metrics` (HTTP & inference latency quantiles + circuit breaker snapshot)
- `/admin/api/pnl/timeseries` (Intraday deterministic PnL placeholder series with `points: [{ts, pnl}], summary: {current,min,max}`)
- `/admin/api/historical/coverage` (Executes verification script returning long-horizon coverage stats & ratios; used for compliance/monitoring)
- `/admin/api/data/verification` (Consolidated multi-dataset verification summary: equities/options/news/social earliest/latest, span, distinct trading days, row counts, missing day estimates, recent gap flags)
- `/admin/api/events/stream` (SSE snapshots combining latency + PnL at interval)
- `/admin/api/heartbeat/stream` (Low-cost 5s heartbeat SSE with latency metric presence flag)

All are intentionally lightweight and return deterministic JSON structures to simplify frontend integration and allow future replacement with real data sources.

## Caching & Deterministic Placeholders
Small in-memory TTL cache (`api/cache_utils.py`) wraps frequently polled data:
- System summary (5s)
- Trading performance (10s)
- Model status (15s)
- Metrics summary (5s)
- KPIs (10s)
- Company forecast (30s)
- Companies list (60s)
- Report (120s)
- Sparkline (60s)

Random placeholders replaced with deterministic seeded values (SHA-256 of key: date + context) for reproducibility.

## Audit Logging
Structured entries using logger names:
- `ADMIN_AUDIT`
- `BUSINESS_AUDIT`

Each records: user, user_id, action, and optional symbol.

## Rate Limiting
Dashboard JSON endpoints enforce rate limits via `EnhancedRateLimiter` (admin → `admin` bucket, business → `default`). If Redis unavailable, production behavior still respects fail-closed logic; development may bypass.

## Sparkline Endpoint
`/business/api/company/{symbol}/sparkline` returns a deterministic pseudo price series for lightweight charting.

## Updated Extensibility Roadmap
1. Replace placeholders with real data sources (QuestDB views, Prometheus aggregation layer).
2. Expand SSE or migrate to WebSocket for sub-5s latency updates where justified.
3. Move caching to distributed layer if horizontal scaling.
4. Harden session management (already added cookie access/refresh tokens + logout + refresh endpoint).
5. Add SVG/Canvas rendering micro-charts (client side) under existing CSP constraints.
6. Add portfolio & risk panels (live VaR, drawdown) once metrics available.

## Authentication Flow (Browser)
1. User visits `/auth/login` and submits credentials (+ optional MFA code).
2. Server sets HttpOnly `at` (access, ~15m) and `rt` (refresh, 7d) cookies.
3. Dashboards send credentialed fetch requests; access token auto-attached by HttpOnly cookie (cookie-or-bearer dependency handles extraction—no JS header injection required).
4. Silent refresh script periodically invokes `/auth/refresh` before expiration to renew `at` (access) cookie.
5. CSRF token cookie is issued alongside login; frontend includes `X-CSRF-Token` for POST/DELETE/PUT/PATCH.
6. `/auth/logout` revokes access token (best-effort) and clears cookies.

Security notes:
- Cookies are `SameSite=Strict`, `HttpOnly`, `Secure` (production) to mitigate CSRF & XSS token theft.
- MFA enforced for admin users if configured via Redis flags.
- Future: CSRF token binding for state-changing POST requests (current dashboard POST endpoints are admin-protected and low risk but will be upgraded with anti-CSRF token header).

## Password Rotation Procedure
Use `scripts/update_password.py --username <user> --password '<NewStrongPassword>'` (minimum 12 chars). Script upserts user with Argon2 hash (bcrypt fallback retained for legacy hashes). Future enhancement will include password version claims for immediate JWT invalidation; current revocation is best-effort via Redis namespace if available. Document rotation in ops runbook and require post-rotation login.

Virtual Environment Rationale: A `.venv` isolates cryptographic/auth libraries (argon2-cffi, passlib) on minimal production hosts lacking system-wide packages, ensuring reproducible security patching. Keep until container build fully encapsulates dependencies.

## Data Coverage & Ingestion Panels
Coverage and ingestion endpoints return placeholders if underlying QuestDB tables (`equities_coverage_daily`, `options_coverage_daily`, `market_data`, `options_data`, `news_events`) are absent. The coverage summary endpoint now exposes a short history window suitable for line charts without additional queries (seeded synthetic until real historical aggregation table exists).

### Historical Coverage Verification
Endpoint: `/admin/api/historical/coverage` wraps `scripts/verify_historical_coverage.py` via a subprocess. Output JSON schema (example):
```
{
  "run_ts": "2025-09-11T12:34:56.123456Z",
  "equities": {"first_date": "2005-01-03", "last_date": "2025-09-10", "trading_days": 5234, "span_years": 20.7, "ratio_vs_target": 0.99 },
  "options": {"first_date": "2024-08-15", "last_date": "2025-09-10", "trading_days": 275, "span_years": 1.1, "ratio_vs_target": 1.00 },
  "target_years": {"equities": 20, "options": 1}
}
```
Return codes from script indicate severity; endpoint always 200 with structured fields (`status`, `stderr_lines`) for UI resilience.

## Latency & Circuit Breakers
`/admin/api/latency/metrics` scrapes selected Prometheus metrics each poll to avoid exporting large metric payloads to the browser. Circuit breaker states are pulled via `trading_common.resilience.get_all_circuit_breakers()` (fail-silent if unavailable).

## SSE & Streaming
`/admin/api/events/stream` emits composite snapshots (latency + PnL) every ~15s for incremental hydration without multiple polling calls. Browser auto-reconnect provides resiliency.
Task output and logs use independent SSE channels (`/admin/api/tasks/{id}/stream`, `/admin/api/logs/stream`).

## Operational Considerations
- Failure to register dashboard routers does not block API startup (guarded try/except).
- Metrics endpoint summary intentionally curated to avoid high payload size or leaking sensitive labels.
- Business/company profile page purposely uses separate forecast/report endpoints for cache layering later.

## Hardening Checklist (Completed)
- [x] Strict CSP with nonce
- [x] Role restricted routes
- [x] No untrusted inline scripts
- [x] Security headers consolidation
- [x] Template autoescaping enabled
- [x] Cookie-based JWT (access + refresh) with silent renewal
- [x] CSRF token + header enforcement for mutating requests
- [x] MFA hook integration (extensible)
- [x] Deterministic placeholder data to avoid volatile output

## Future Hardening
- Expand rate limiting granularity (per-endpoint burst tracking)
- Add audit logging for admin dashboard access events (currently only action events)
- Signed nonces or session binding if frames are ever allowed (currently frame denied)
- Historical coverage table population job for real chart history (replace synthetic)
- WebSocket upgrade for high-frequency panels if justified by latency-sensitive metrics

## Chart Rendering & Shared Utilities
All charts use a shared `static/js/chart_utils.js` module providing:
- Linear scale helpers
- Multi-series line rendering
- Legend + hover tooltip hit detection
- Deterministic color palette

Implemented Charts:
- Business Coverage: equities/options ratios (`history` field)
- Business News Sentiment: 30d sentiment history (`history` field)
- Admin PnL Timeseries: intraday PnL points with summary stats

Accessibility: Each canvas has `role="img"` and `aria-label`; numeric summaries adjacent. Future enhancement: offscreen table for screen readers.

## Requirements Traceability (Recent Additions)
- Coverage history for charting: implemented via `history` field.
- PnL timeseries with summary stats: implemented (`points`, `summary`).
- Historical coverage verification endpoint: implemented.
- Consolidated data verification endpoint (`/admin/api/data/verification`): implemented.
- Heartbeat SSE stream (`/admin/api/heartbeat/stream`): implemented.
- News sentiment 30d history timeline: implemented.
- Shared chart utilities + legends & tooltips: implemented.
- Silent access token refresh: implemented.
- CSRF protection: implemented.

## Contact
For architecture questions or escalation paths, refer to system maintainers or ops runbook.

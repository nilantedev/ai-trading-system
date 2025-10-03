# Configuration Reference

Centralized environment variable and operational configuration reference for the AI Trading System.

## 1. Authentication & Security
| Variable | Default | Description | Notes |
|----------|---------|-------------|-------|
| ACCESS_TOKEN_EXPIRE_MINUTES | 15 | Access token lifetime (minutes). | Short-lived by design; do not raise significantly. |
| REFRESH_TOKEN_EXPIRE_DAYS | 7 | Refresh token lifetime (days). | Rotation and revocation supported. |
| JWT_KEY_ROTATION_DAYS | 30 | Lifetime for active signing key. | Rotate earlier if compromise suspected. |
| JWT_KEY_OVERLAP_HOURS | 24 | Overlap window for old key validity (verification only). | Allows in-flight tokens to drain. |
| REQUIRE_MFA_SENSITIVE | true | Enforce MFA for sensitive operations flag. | Toggled rarely. |
| REQUIRE_MFA_FOR_ADMIN | true | Force MFA for admin role logins. | Should remain true in production. |
| REVOKE_REFRESH_ON_PASSWORD_CHANGE | true | Revoke all refresh tokens after password reset/change. | Ensures global session invalidation. |
| MFA_BACKUP_CODE_PEPPER | (unset) | Pepper concatenated to backup codes before hashing. | Keep secret, rotate with caution. |
| COOKIE_SECURE | true | If true, sets Secure on auth cookies. | Must be true in production (TLS). |
| TRUSTED_HOSTS | * | FastAPI trusted hostnames list. | Set to explicit domains in production. |
| BUSINESS_HOST | biz.mekoshi.com | Expected host for business UI. | Host separation enforced. |
| ADMIN_HOST | admin.mekoshi.com | Expected host for admin UI. | Host separation enforced. |

## 2. Password Policy
- Enforced server-side: length ≥ 12, at least one uppercase, lowercase, digit, and symbol.
- Failure increments metric `auth_password_resets_total{event="change_failed"|"reset_failed"}`.

## 3. Password Reset Rate Limiting
| Limit | Window | Dimension | Notes |
|-------|--------|-----------|-------|
| 5 | 1 hour | user | Sliding window (Redis list or in-memory fallback). |
| 10 | 1 hour | IP | Helps deter bulk enumeration / spraying. |

## 4. Metrics (Auth Lifecycle)
`auth_password_resets_total{event=...}` label values:
- request
- request_rate_limited
- token_issued
- reset_success
- reset_failed
- reset_refresh_revoked
- change_success
- change_failed
- change_refresh_revoked

Key gauges & counters (other):
- auth_mfa_adoption_percent
- auth_failed_login_counters
- auth_key_rotation_remaining_seconds / auth_key_rotation_age_seconds

## 5. Streaming & Ingestion (Selected)
| Variable | Purpose |
|----------|---------|
| INGEST_CONCURRENCY_LIMIT | Max concurrent ingestion tasks. |
| STREAM_DLQ_ENABLED | Enable DLQ routing for malformed messages. |
| STREAM_FRESHNESS_INTERVAL | Symbol freshness sampling interval (seconds). |
| NEWS_RETENTION_DAYS | Hard age cutoff (days) for news_items window deletes (default 1825 = 5y). |
| SOCIAL_RETENTION_DAYS | Hard age cutoff (days) for social_events / social_signals (default 1825 = 5y). |
| OPTION_ZERO_VOL_DAYS | Additional prune window for zero-volume option rows (default 540). |
| NEWS_LOWRELEVANCE_DAYS | Age threshold for low-relevance news prune (default 30). |
| SOCIAL_LOWENGAGEMENT_DAYS | Age threshold for low-engagement social prune (default 14). |
| OPTIONS_BACKFILL_DAYS | Initial historical options backfill horizon (default 730). |
| NEWS_BACKFILL_DAYS | Historical news backfill horizon (default 1825). |
| SOCIAL_BACKFILL_DAYS | Historical social sentiment backfill horizon (default 1825). |
| EQUITIES_BACKFILL_DAYS | Historical equities backfill horizon (default 7300). |
| NEWS_INGEST_VALUE_MIN | Minimum value_score to keep a news article at ingestion (pre-persist). |
| SOCIAL_INGEST_VALUE_MIN | Minimum value_score to keep a social sentiment row at ingestion (pre-persist). |
| NEWSAPI_ENABLED | Enable NewsAPI provider (default true if NEWS_API_KEY set). |
| FINNHUB_NEWS_ENABLED | Enable Finnhub news endpoints (default true if FINNHUB_API_KEY set). |
| FINNHUB_SYMBOLS_PER_WINDOW | Max symbols per Finnhub call window (default 5). |
| FINNHUB_SYMBOL_PACING_SECONDS | Sleep between Finnhub symbol loops (default 0.25s). |
| POLYGON_NEWS_ENABLED | Enable Polygon v2/reference/news API (default true if key present). |
| POLYGON_FLATFILES_ENABLED | Enable Polygon S3 flatfiles bulk news (default true if creds present). |
| ALPHAVANTAGE_NEWS_ENABLED | Enable Alpha Vantage NEWS_SENTIMENT (default true if key present). |
| GDELT_ENABLED | Enable GDELT public news (default true). |
| NEWSAPI_MAX_SYMBOLS_PER_QUERY | Chunking to avoid NewsAPI 400s (default 20). |
| NEWS_STREAM_INTERVAL_SECONDS | Streaming cadence; raise to reduce 429s (default 300). |
| FINNHUB_WEBSOCKET_ENABLED | Enable Finnhub realtime websocket (default false to respect free tier). |
| POLYGON_WEBSOCKET_ENABLED | Enable Polygon realtime websocket (default true). |
| USE_TWELVEDATA_FALLBACK | Enable TwelveData daily-bars fallback when EODHD fails/missing (default false). |
| TWELVEDATA_API_KEY | TwelveData API key for fallback daily bars. |

### Historical Horizons vs Retention
| Domain | Initial Backfill Horizon | Retention Horizon | Notes |
|--------|--------------------------|-------------------|-------|
| Equities (market_data) | Up to 20y (gap fill) | 20y+ daily bars kept (older pruned) | Daily data effectively long-term archive. |
| Options | 2y (initial) | 5y | Intentionally capped backfill to reduce load after plan upgrade; forward accumulation continues to fill remaining 3y over time or via later bulk job. |
| News | 5y | 5y | Full 5y fetched (chunked); low-value & relevance pruning still applies within window. |
| Social | 5y | 5y | Low-engagement/value prune rules may delete earlier subsets; hard cap remains 5y. |

Important: The retention service enforces hard age cutoffs (DELETE) at the retention horizon, while additional value-based pruning (news relevance, social low engagement, value_score floors) can remove low-signal data sooner to conserve storage without impacting analytical quality.

### Canonical Backfill Endpoints
- POST /backfill/news/eodhd-60d: Bounded backfill over N days using EODHD-first provider flow. Body: { symbols?, days, batch_days, max_articles_per_batch }.
- POST /backfill/calendar/eodhd: Calendar (earnings, IPOs, splits, dividends optional) backfill over N years.
- POST /backfill/options/run: Options chain backfill over expiry window with historical bar window.
- POST /backfill/news/options-batch/run: Server-side batch scheduler that resolves distinct option underlyings from QuestDB, chunks them (default 25), and sequentially runs news backfill windows honoring pacing.

Deprecated/removed: All legacy /news/backfill variants have been removed to avoid duplication. Use the /backfill/news family exclusively.

Batch Scheduler Notes (/backfill/news/options-batch/run):
- Request fields: underlyings?, max_underlyings (default 500), chunk_size (default 25), years (default ENV NEWS_BACKFILL_YEARS), batch_days (default 14), max_articles_per_batch (default 80), inter_batch_delay_seconds (default 5).
- Returns: { status, underlyings, chunks, chunk_size, start, end, years } and runs asynchronously.
- Honors provider pacing and environment caps (e.g., EODHD_PACING_SECONDS) internally via per-window chunking and inter-batch delay.

Healthcheck Tuning:
- redis-exporter: Healthchecks use /metrics with relaxed timeout/retries to reduce flapping; functional metrics remain available even if transiently slow.
- Ollama: Healthchecks use /api/version with increased timeout/retries; model loading may take time after container start.

## 6. Observability
| Variable | Purpose |
|----------|---------|
| API_CONCURRENCY_LIMIT | API concurrency gauge baseline. |
| DISABLE_METRICS_WARMUP | Skip warm-up request emission if true. |
| PROMETHEUS_MULTIPROC_DIR | Multiprocess metrics directory (if using gunicorn). |

## 7. Email (Password Reset)
| Variable | Required | Description |
|----------|----------|-------------|
| SMTP_HOST | Y | SMTP server hostname. |
| SMTP_PORT | N (587) | SMTP port (STARTTLS). |
| SMTP_USER | Y | Auth user. |
| SMTP_PASSWORD | Y | Auth password. |
| SMTP_FROM | Y (or SMTP_USER) | From address. |
| APP_PUBLIC_URL | N (https://biz.mekoshi.com) | Base URL for reset links. |

If SMTP variables absent: API returns masked token preview (first 12 chars + ellipsis) for staging only.

## 8. Recommended Secrets Rotation Cadence
| Secret Type | Cadence | Method |
|-------------|---------|--------|
| JWT Signing Key | 30d (auto) | Built-in rotation scheduler. |
| MFA Backup Codes | User-initiated | Encourage after each admin privilege escalation. |
| Redis Password | 90d | Rotate, restart dependent services sequentially. |
| Database Passwords | 90d | Staged rotation w/ dual credentials window. |

## 9. Alert Names (Auth Segment)
| Alert | What It Indicates |
|-------|------------------|
| AuthPasswordResetAbuse | Elevated reset volume. |
| AuthPasswordResetAbuseCritical | Severe reset surge. |
| AuthPasswordResetRateLimitedSpike | High rate-limited counts (attack). |
| AuthPasswordResetFailuresSpike | Elevated invalid/expired token or policy failures. |
| AuthPasswordResetFailuresCritical | Severe failure surge. |
| AuthPasswordChangeFailuresSpike | Many failed change attempts. |
| AuthPasswordChangeFailuresCritical | Potential credential probing / abuse. |
| AuthPasswordResetRefreshRevocationsMissing | Resets without session revocation. |

## 10. Hardening Notes
- All password reset tokens single-use, TTL 30 minutes.
- Refresh token revocation scans Redis; falls back gracefully if unavailable (no crash path).
- Metrics emission guarded with try/except to avoid runtime disruption.
- Alert thresholds tuned for baseline; revisit after 2 weeks of production telemetry.

## 11. Future Roadmap
- Compromised password check (haveibeenpwned range API).
- Device / location anomaly scoring during reset.
- Adaptive thresholding (dynamic baselines) for auth-related alerts.

---
Update this file when adding new environment variables or altering security-relevant defaults.


## 12. Polygon Flat Files (S3) for Historical News

The data-ingestion service can load bulk historical news from Polygon's flat files (S3-compatible archive).

Required environment variables (set via your deployment environment or .env):

- POLYGON_S3_ENDPOINT: S3 endpoint for flat files. Default: https://files.polygon.io
- POLYGON_S3_BUCKET: Bucket name. Default: flatfiles
- POLYGON_S3_ACCESS_KEY_ID: Access key ID for Polygon flat files
- POLYGON_S3_SECRET_ACCESS_KEY: Secret access key for Polygon flat files
- POLYGON_S3_REGION: S3 region. Default: us-east-1

Behavior and tips:
- Without valid credentials, the Polygon S3 path is disabled and no news will ingest from flat files.
- On service startup, a lightweight S3 connectivity probe runs and logs success/failure.
- Smoke test ingestion using the data-ingestion API: POST /backfill/news with providers ["polygon_s3"] and a small symbol/date window.
- Ensure IAM/permissions allow ListObjects and GetObject for the news/ prefixes in the bucket.

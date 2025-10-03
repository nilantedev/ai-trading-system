# Operational Scripts Guide

Comprehensive reference for production operations scripts in the AI Trading System.

## Scripts Overview

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `scripts/start_full_system.sh` | Ordered, dependency-aware startup | Phased groups, adaptive waits, tracing wait option, metrics & dashboard readiness, summary diagnostics |
| `scripts/check_health_production.sh` | Comprehensive live health assessment | Infra/services/metrics/dashboards, resource & error analysis, P95 latency estimation, security & backup checks |
| `scripts/reset_environment.sh` | Safe environment reset / rebuild | Plan mode, soft restart, granular wipes (cache, timeseries, vectors, logs), purge Pulsar, selective rebuild |
| `scripts/logs_tail.sh` | Advanced multi-service log tailing | JSON parsing, highlight & squelch, severity colors, rate limiting, fail-on-error, container/time prefixes |
| `scripts/diagnose_service.sh` | Deep dive for a single service | Logs, metrics, health endpoints, network, env/image metadata |

## 1. Startup (`start_full_system.sh`)

```
./scripts/start_full_system.sh [--build] [--no-cache] [--fast] [--wait-tracing] \
  [--skip-final-health] [--no-dashboards-check] [--summary-only]
```

Phases (in order): Core Infra → Observability → App Phase1 (api) → Phase2 (ml) → Phase3 (ingestion, signal, risk) → Phase4 (execution, strategy, backtesting)

Readiness Enhancements:
- Metrics readiness: waits for `app_http_requests_total` exposure.
- Dashboard checks: `/admin` & `/business` with CSP header (can disable).
- Tracing collector optional gating.

## 2. Health (`check_health_production.sh`)
Run anytime; exit codes: 0 healthy, 1 degraded, 2 unhealthy.

Highlights:
- Component inventory & presence grid.
- Infrastructure, monitoring, application, provider stack.
- Shared metrics validation & dashboard availability.
- P95 HTTP latency estimation (histogram bucket snapshot).
- Container resource snapshot, image & volume hygiene.
- Drift / model readiness placeholders, AI model presence.
- Security (JWT, firewall heuristics, fail2ban), backups age, external API reachability.

Optional flags via environment:
- `LOKI_DEEP_CHECK=true` deep ingestion roundtrip.
- `PROVIDER_DEEP_TEST=true PROVIDER_TEST_SYMBOL=AAPL` provider quote probe.

## 3. Reset (`reset_environment.sh`)

```
./scripts/reset_environment.sh --force [options]
```

Key Flags:
- Safety / Planning: `--plan` (dry-run), `--soft` (skip destructive actions), `--force` required unless plan.
- Wipes: `--wipe-cache`, `--wipe-timeseries`, `--wipe-vectors`, `--wipe-logs`, `--wipe-volumes` (nuclear).
- Pulsar / Images: `--purge-pulsar`, `--prune-images`.
- Build: `--no-rebuild` (skip rebuild), default rebuild after destructive changes.
- Health: `--skip-health` skip final health run.

Recommended Patterns:
- Fresh but safe: `--force --purge-pulsar --prune-images`
- Deep data reset: `--force --wipe-cache --wipe-timeseries --wipe-vectors`
- Full nuclear: `--force --wipe-volumes`
- Fast restart: `--force --soft --no-rebuild`

## 4. Logs (`logs_tail.sh`)

```
./scripts/logs_tail.sh [services...] [--since 30m] [--grep REGEX] [--json] \
  [--json-field f1,f2] [--highlight REGEX] [--squelch REGEX] [--fail-on-error] [--rate N] [--no-follow]
```

Environment Variables:
- `LOG_DATETIME=yes` add timestamps
- `LOG_SHOW_CONTAINER=no` hide container name
- `LOG_COLOR=no` disable colors

Failure Integration:
- `--fail-on-error` returns exit code 2 if any ERROR/CRITICAL/FATAL lines matched (useful for CI smoke checks).

## 5. Diagnose (`diagnose_service.sh`)

```
./scripts/diagnose_service.sh <service> [--logs N] [--metrics] [--net] [--env] [--all]
```

Service Short Names: api, ml, data-ingestion, signal-generator, execution, risk-monitor, strategy-engine, backtesting, redis, postgres, questdb, pulsar, weaviate, minio, grafana, prometheus, loki.

Outputs:
- Container status, resource snapshot
- Health endpoints & codes
- Recent logs (configurable lines)
- Metrics sample (first 25 `app_` lines)
- Network IP, port mapping, socket summary
- Sanitized environment key visibility & image creation timestamp

## 6. CI Shell Lint
GitHub Action: `.github/workflows/shellcheck.yml`
- Runs ShellCheck on all `scripts/*.sh` on push & PR changes.
- Uses `--severity=style` (upgrade to `info` or `warning` as desired).

## 7. Best Practices & Safety
- Always run health script after resets for confirmation.
- Use `--plan` before destructive resets in production.
- Avoid `--wipe-volumes` unless performing a controlled re-seed.
- Investigate any non-zero shed counter or missing metrics promptly.
- Track latency trends; rising P95 may indicate saturation or drift.

## 9. Retention Tunables (Env)
These control table-specific retention in QuestDB via the Data Retention Service:
- `OPTION_ZERO_VOL_DAYS` (default 540): delete zero-volume option bars older than N days (windowed)
- `OPTION_EXPIRY_PAST_DAYS` (default 120): delete option_daily rows whose expiry is older than N days
- `NEWS_RETENTION_DAYS` (default 365): age-based delete for `news_items` (windowed)
- `NEWS_LOWRELEVANCE_DAYS` (default 30): delete low-relevance, near-neutral older than N days
- `NEWS_LOWRELEVANCE_SENTIMENT` (default 0.05): abs(sentiment) threshold
- `NEWS_LOWRELEVANCE_THRESHOLD` (default 0.2): relevance threshold
- `SOCIAL_RETENTION_DAYS` (default 180): age-based delete for `social_signals` (windowed)
- `SOCIAL_LOWENGAGEMENT_DAYS` (default 14): delete low-engagement, low-influence older than N days
- `SOCIAL_LOWENGAGEMENT_THRESHOLD` (default 0.05): engagement threshold
- `SOCIAL_LOWINFLUENCE_THRESHOLD` (default 0.1): influence threshold
- `RETENTION_WINDOW_DAYS` (default 30): size of delete windows
- `RETENTION_MAX_WINDOWS_PER_TABLE` (default 6): max windows per run (safety valve)

These are wired in `docker-compose.yml` under the `data-ingestion` service.

## 10. Options Coverage Report
Two ways to surface options coverage for Grafana and audits:
- API: `GET /coverage/options?underlyings=AAPL,MSFT` (FastAPI data-ingestion)
- Background export: enable `ENABLE_OPTIONS_COVERAGE_REPORT=true` to write a JSON to
  `/app/export/grafana-csv/options_coverage.json` (mounted to `/mnt/fastdrive/trading/grafana/csv`).
  Tunables:
  - `OPTIONS_COVERAGE_INTERVAL_SECONDS` (default 86400)
  - `OPTIONS_COVERAGE_OUTPUT_DIR` (default /app/export/grafana-csv)
  - `OPTIONS_COVERAGE_MAX_UNDERLYINGS` (default 200)
  - `OPTIONS_COVERAGE_UNDERLYINGS` (optional CSV override)

CLI script (ad-hoc run): `services/data_ingestion/scripts/options_coverage_report.py` with `--questdb-url` and `--underlyings`.

## 11. Backfill Driver (safe API client)
Use `scripts/backfill_driver.py` to avoid shell JSON quoting issues and run common backfill tasks:

- Export coverage artifacts:
  - `./scripts/backfill_driver.py coverage`
- Pilot options backfill (example: last 30 days, expiry -7d..+60d):
  - `./scripts/backfill_driver.py options --symbols AAPL,MSFT --start 2024-08-10 --end 2024-09-10 --start-expiry -7d --end-expiry +60d --max-contracts 300 --pace 0.2`
- Historical news backfill (windowed):
  - `./scripts/backfill_driver.py news --symbols AAPL,MSFT --start 2019-01-01 --end 2020-12-31 --batch-days 14 --max-articles 80`
- Historical social backfill:
  - `./scripts/backfill_driver.py social --symbols AAPL,MSFT --start 2022-01-01 --end 2022-02-01 --batch-hours 6`

Set `INGEST_BASE_URL` if the ingestion API isn’t at `http://127.0.0.1:8002`.

### 11.1 Horizon & Gap Enforcement (Historical Backfill)

The historical backfill orchestrator (`scripts/run_historical_backfill.py`) now supports automatic 20-year equity horizon enforcement and internal gap detection to guarantee durable coverage:

Core Capabilities (Expanded):
- Horizon Enforcement: compute effective start as (today - BACKFILL_TARGET_YEARS_EQUITIES years) when `--enforce-horizon` / `BACKFILL_ENFORCE_HORIZON=true`.
- Precise Internal Gaps (Equities): leading + trailing always; internal spans > `BACKFILL_MIN_GAP_DAYS` when `--internal-gaps` / `BACKFILL_INTERNAL_GAP_DETECTION=true`.
- Heuristic Internal Gaps (Non‑Equities): density-based detection for `options,news,social` via `--internal-gaps-others` or `BACKFILL_INTERNAL_GAP_HEURISTIC_OTHERS=true` (flags holes if observed_day_density < `BACKFILL_DENSITY_THRESHOLD`).
- QuestDB Coverage Queries: non-destructive reads of underlying asset tables (market_data, options_data, news_events, social_events) via HTTP `/exec`.
- Concurrency: symbol-level tasks + gap-level semaphore bounded by `--concurrency` / `BACKFILL_CONCURRENCY`.
- Plan Mode: `--plan` produces JSON (stdout + optional file) with missing ranges, heuristic density, and warnings.
- Gap Chunking: `--max-gap-days-chunk` splits wide ranges for provider friendliness.
- Large Gap Warnings: gaps > `BACKFILL_MAX_GAP_WARNING_DAYS` tagged in plan (`gap_exceeds_threshold`).
- Coverage Metrics: oldest/newest epoch timestamps, coverage span (days), and lag (days) per asset+symbol for Grafana & alerting.

Recommended Cron (daily early UTC):
```
ENABLE_HISTORICAL_BACKFILL=true \
BACKFILL_ENFORCE_HORIZON=true \
BACKFILL_INTERNAL_GAP_DETECTION=true \
python scripts/run_historical_backfill.py --symbols AAPL,MSFT,GOOG,AMZN,SPY --enforce-horizon --internal-gaps
```

Tunable Environment Variables:
- `BACKFILL_TARGET_YEARS_EQUITIES` (default 20)
- `BACKFILL_MIN_GAP_DAYS` (default 5)
- `BACKFILL_ENFORCE_HORIZON` true/false
- `BACKFILL_INTERNAL_GAP_DETECTION` true/false
- `BACKFILL_TARGET_YEARS_OTHERS` (default 5) horizon for options/news/social
- `BACKFILL_ASSETS` CSV assets to include (equities,options,news,social) default `equities`
- `BACKFILL_MAX_GAP_DAYS_CHUNK` (default 365) splits large gaps to smaller provider windows
 - `BACKFILL_INTERNAL_GAP_HEURISTIC_OTHERS` true/false enable heuristic gaps for non-equities
 - `BACKFILL_DENSITY_THRESHOLD` (default 0.6) density cutoff for heuristic activation
 - `BACKFILL_CONCURRENCY` (default 1) max simultaneous equity gap ingestions
 - `BACKFILL_PLAN_OUTPUT_DEFAULT` default plan file path if `--plan-output` omitted
 - `BACKFILL_MAX_GAP_WARNING_DAYS` threshold for warning injection in plan JSON

Additional Flags:
```
--plan                  Dry-run: emit JSON plan of missing ranges (no ingestion)
--assets equities,news  Limit or extend asset set
--max-gap-days-chunk N  Override chunk size for splitting big gaps
--concurrency N         Max concurrent equity gap ingestions (bounded by semaphore)
--plan-output FILE      Write JSON plan to FILE
--internal-gaps-others  Enable density heuristic internal gap detection (non-equities)
--density-threshold F   Override heuristic density threshold (default 0.6)
```

Exit Codes remain unchanged (0 success, 2 flag disabled, 3 init failure).

Operational Guidance:
- Use `--start/--end` for ad‑hoc bounded fills; combine `--enforce-horizon` to re-baseline fully.
- Enable heuristic (`--internal-gaps-others`) cautiously; review density & warnings in plan before real ingestion.
- Monitor `historical_backfill_progress_timestamp_seconds` for stall detection (no recent advances => investigate provider or Redis state).
- Coverage Panels:
  - `asset_symbol_coverage_timestamp_seconds{asset="equities",symbol="AAPL",bound="oldest|newest"}`
  - `asset_symbol_coverage_span_days{asset="equities",symbol="AAPL"}`
  - `asset_symbol_coverage_lag_days{asset="equities",symbol="AAPL"}`
- Alert Suggestions:
  - Coverage lag > 2 trading days on active equities.
  - Span days < expected horizon minus 30d tolerance.
  - Any `gap_exceeds_threshold` warning for priority symbols (review before ingestion).

  ## 12. Retention Enforcement

  `scripts/enforce_retention.py` applies rolling retention: 20y equities, 5y options/news/social.

  Usage (dry-run default):
  ```
  python scripts/enforce_retention.py
  ```
  Apply deletions & emit JSON metrics:
  ```
  python scripts/enforce_retention.py --apply --metrics-json /var/log/retention/last_run.json
  ```
  Locking:
  - Prevents concurrent runs via `/tmp/enforce_retention.lock` (override with `--lock-path` or env `RETENTION_LOCK_PATH`).
  - Stale lock (>3600s) auto-recovers.
  - Disable locking (not recommended) with `--no-lock`.

  Flags:
  | Flag | Description |
  |------|-------------|
  | `--apply` | Execute deletions (omit for dry-run) |
  | `--metrics-json FILE` | Write structured JSON summary for audits / ingestion |
  | `--lock-path FILE` | Override lock file path |
  | `--no-lock` | Disable locking (testing only) |

  JSON Structure:
  ```
  {
    "apply": false,
    "generated_at": "2025-09-12T00:00:00Z",
    "tables": [
      {"table": "equities_prices", "total": 1234567, "old": 3456, "deleted": 0, "cutoff": "2005-09-12T00:00:00Z"}
    ]
  }
  ```

  Operational Recommendations:
  - Run daily in dry-run, weekly with `--apply` after validating counts.
  - Track deletion volumes; sudden spikes may indicate upstream clock or ingestion issues.
  - Store JSON artifacts for compliance and space reclamation audits.

  Future Enhancements (planned):
  - Export metrics directly via Pushgateway (optional).
  - Per-table override windows via env variables.
  - Parallel segmented deletions for very large tables (bounded by I/O budget).


## 8. Future Enhancements (Backlog)
- Automated drift metrics extraction and threshold alerting from health script.
- Extended latency percentiles (P50/P99) via histogram calculation.
- Optional JSON report output for health script (machine-readable).
- Integration with alert routing (Pager / Slack) for critical failures.

---
Document last updated: $(date '+%Y-%m-%d')

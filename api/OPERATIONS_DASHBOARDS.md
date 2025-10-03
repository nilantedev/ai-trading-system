# Dashboards Operations: health checks, outage policy, trust lists, and metrics

This doc captures operational knobs and observability for the Admin and Business dashboards.

## Health checks
- External probes should use `GET` or `HEAD` on `/admin/availability`.
  - Response: `{ "ok": true, "ts": <epoch_ms> }` with HTTP 200 when the API is up.

## Rate limiting outage policy
- In production, if Redis is unavailable or the limiter enters FAIL_CLOSED, the following public paths remain accessible by default:
  - `/admin/availability`
  - `/business/api/kpis`
- You can extend the allowlist via env: `FAIL_CLOSED_PUBLIC_PATHS="/path1,/path2"`.

## Trusted IPs/CIDRs (signed configuration)
- To allowlist monitoring networks and bypass limiter blocks when safe:
  - `TRUSTED_IP_SECRET` (strong secret used for HMAC signing)
  - `RATE_LIMIT_TRUSTED_IPS`: `"ip1,ip2,...:<hmac_sha256_hex>"`
  - `RATE_LIMIT_TRUSTED_CIDRS`: `"cidr1,cidr2,...:<hmac_sha256_hex>"`
- Signature format: `hex(HMAC_SHA256(plaintext_list, key=TRUSTED_IP_SECRET))` where `plaintext_list` is the exact comma-joined string of IPs/CIDRs before the colon.

## Prometheus metrics (admin operations)
- Exposed at `/metrics` (default registry). Metrics appear after first use:
  - `admin_tasks_started_total{name}`
  - `admin_tasks_completed_total{name,status}` (status ∈ {success,error})
  - `admin_tasks_running` (gauge)
  - `admin_logs_stream_clients` (gauge)
  - `admin_task_stream_clients` (gauge)
- Sample Grafana queries:
  - Task throughput (5m): `sum by (name,status) (rate(admin_tasks_completed_total[5m]))`
  - Running tasks: `admin_tasks_running`
  - SSE clients: `admin_logs_stream_clients` and `admin_task_stream_clients`

  ## Grafana auto-provisioning
  - Datasource provisioning file (Prometheus): `trading/config/grafana/provisioning/datasources/prometheus.yml`
  - Dashboards provisioning file: `trading/config/grafana/provisioning/dashboards/dashboards.yml`
  - Dashboard JSONs directory: `trading/config/grafana/dashboards/`
  - To deploy on this server, ensure these repo files are synced to the mounted host paths used by docker-compose:
    - Sync to `/srv/trading/config/grafana/provisioning/` → mounted into Grafana at `/etc/grafana/provisioning/`
    - Sync to `/srv/trading/config/grafana/dashboards/` → mounted into Grafana at `/var/lib/grafana/dashboards/`
    - Then restart the Grafana container to pick up changes.

  ## Prometheus scraping (example)
  If your Prometheus config doesn’t already discover API/ML services, add scrape jobs similar to:

  job_name: trading-api
  metrics_path: /metrics
  scrape_interval: 15s
  static_configs:
    - targets: ['trading-api:8000']

  job_name: trading-ml
  metrics_path: /metrics
  scrape_interval: 15s
  static_configs:
    - targets: ['trading-ml:8001']

  job_name: data-ingestion
  metrics_path: /metrics
  scrape_interval: 30s
  static_configs:
    - targets: ['trading-data-ingestion:8002']

## Grafana panel
- A starter dashboard JSON is provided at `tools/grafana/admin_ops.dashboard.json`.
  - Import it in Grafana (Dashboards → Import) and select your Prometheus data source.

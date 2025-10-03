Ingestion & Retention Dashboards

- `ingestion_and_retention.json`: Overview panel for backfill status, retention deletions, retention runs/last run, and provider error/latency.
- JSON export (optional): options coverage written to `/mnt/fastdrive/trading/grafana/csv/options_coverage.json` by the data-ingestion service when `ENABLE_OPTIONS_COVERAGE_REPORT=true`.
  - You can add a JSON datasource/panel to parse and visualize contracts and gap days per underlying.

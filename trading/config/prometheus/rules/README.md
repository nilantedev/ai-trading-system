# Prometheus Alerting Rules

This directory contains custom alerting rules for the AI Trading System. They are mounted into Prometheus at `/etc/prometheus/rules` and loaded via the main `prometheus.yml` configuration.

## Groups Overview

1. api-slo
   - Error rate thresholds (warning 5%, critical 15%).
   - Latency SLO for p95 (warning 1s, critical 2s).
   - Concurrency saturation (>90%).
   - Request shedding detection.

2. ingestion-slo
   - Error rate & shedding for data ingestion service.
   - Historical backfill progress stall detection (15m warning / 60m critical).

3. pulsar-health
   - Managed ledger persistence errors (warning any non-zero over 5m, critical >10).

4. ml-governance
   - Silence / inactivity detection for governance metrics (>15m).
   - Auto rollback spike detection.
5. auth-security
   - MFA adoption thresholds (warning <60%, critical <40%).
   - JWT key rotation lifecycle (near-expiry & critical urgency).
   - Failed login counters (elevated & critical brute force indicators).
   - Password reset & change lifecycle abuse/failure alerts:
     * Reset volume abuse (warning/critical)
     * Rate-limited spike
     * Reset failure spike (warning/critical)
     * Change failure spike (warning/critical)
     * Missing refresh token revocation after reset
6. streaming
   - Consumer lag thresholds (market topic) & silence detection.
   - Publish skip volume (warning/critical) and circuit breaker open state.
   - Processing error bursts and DLQ surge + burn-rate (fast & persistent).
   - Topic gap recording & gap outage alerts (5m / 15m).

## Adding New Rules

1. Create or modify a YAML file in this directory.
2. Ensure the file extension is `.yml` or `.yaml`.
3. Follow Prometheus rule group structure (`groups: - name: ... rules: ...`).
4. Validate syntax locally (optional):
   `docker run --rm -v $(pwd):/etc/prometheus prom/prometheus:v2.48.0 --config.file=/etc/prometheus/prometheus.yml --log.level=error`.
5. Reload Prometheus (if web admin API enabled):
   `curl -X POST http://prometheus:9090/-/reload`.

## TODO / Future Enhancements
- Add SLO burn-rate multi-window error budget alerts.
- Include circuit breaker trip rate and drift detection anomalies.
- Create recording rules for normalized rate calculations.
- Integrate blackbox probe latency (external) for end-user latency SLO.
- Add anomaly detection for password reset volume vs historical baseline.
- Introduce per-dataset adaptive lag thresholds (market hours vs off-hours) for ingestion.

This directory contains template configuration files for local development and reference. In production, infrastructure services (Grafana, Prometheus, Loki, Redis, Postgres, MinIO, QuestDB, Pulsar, Weaviate) read their configs from host-mounted paths under /srv/trading/config.

Notes:
- The only repo-managed config currently mounted in production is blackbox-exporter: trading/config/blackbox/blackbox.yml.
- Prometheus and Grafana dashboards in this folder are reference copies; production instances load from /mnt/fastdrive/trading/grafana and /mnt/fastdrive/trading/prometheus.
- Do not edit secrets here. Use /srv/trading/config/secure_config.py or environment variables.
- If you need to promote a template to production, copy it to /srv/trading/config and reload the service.

Quick map:
- Repo templates: ./trading/config/** (this folder)
- Prod infra configs: /srv/trading/config/** (host persistent)
- Grafana data and dashboards: /mnt/fastdrive/trading/grafana/**
- Prometheus data: /mnt/fastdrive/trading/prometheus/**

Contact: leave a note in PRODUCTION_HARDENING_REPORT.md with any config changes.

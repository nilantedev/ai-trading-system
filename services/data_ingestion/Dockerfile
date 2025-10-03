FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*
# Use the service-specific requirements to avoid pulling heavy root requirements (e.g., TA-Lib)
COPY services/data_ingestion/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

FROM base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
	&& rm -rf /var/lib/apt/lists/* \
	&& useradd -u 1014 -m appuser \
	# Ensure application directory exists and is writable by non-root user
	&& mkdir -p /app/export/grafana-csv \
	&& chown -R appuser:appuser /app
USER appuser
COPY --from=build /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=build /usr/local/bin/python /usr/local/bin/python
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn

COPY --chown=appuser:appuser services/data_ingestion/ ./services/data_ingestion/
COPY --chown=appuser:appuser shared/ ./shared/
COPY --chown=appuser:appuser scripts/ ./scripts/
ENV PYTHONPATH=/app:/app/shared/python-common

# Inline Python probe to avoid curl dependency in slim images
HEALTHCHECK --interval=45s --timeout=15s --start-period=60s --retries=6 \
	CMD ["/bin/sh","-lc","python -c 'import urllib.request,sys; import socket; socket.setdefaulttimeout(12);\ntry:\n    with urllib.request.urlopen(\"http://localhost:8002/health\", timeout=12) as r:\n        sys.exit(0 if r.status==200 else 1)\nexcept Exception:\n    sys.exit(1)' "]

CMD ["uvicorn", "services.data_ingestion.main:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "info", "--workers", "1"]
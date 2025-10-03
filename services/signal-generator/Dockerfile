FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl wget gcc && rm -rf /var/lib/apt/lists/*
# Install TA-Lib C library so Python package can build wheels
RUN wget -q https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib-0.4.0-src.tar.gz \
	&& cd /tmp \
	&& tar -xzf ta-lib-0.4.0-src.tar.gz \
	&& cd ta-lib \
	&& ./configure --prefix=/usr \
	&& make \
	&& make install \
	&& cd / \
	&& rm -rf /tmp/ta-lib /tmp/ta-lib-0.4.0-src.tar.gz
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

FROM base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
	&& rm -rf /var/lib/apt/lists/* \
	&& useradd -u 1011 -m appuser
USER appuser
COPY --from=build /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=build /usr/local/bin/python /usr/local/bin/python
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn

COPY --chown=appuser:appuser services/signal-generator/ ./services/signal-generator/
COPY --chown=appuser:appuser shared/ ./shared/
ENV PYTHONPATH=/app:/app/shared/python-common

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 CMD curl -f http://localhost:8003/health || exit 1

CMD ["python", "services/signal-generator/main.py"]
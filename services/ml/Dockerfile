## ============================================================================
## ML Service Dockerfile (Hardened / Multi-Stage)
## ----------------------------------------------------------------------------
## Goals:
##  - Reproducible, minimal runtime surface
##  - Compile TA-Lib in isolated build stage
##  - Pre-warm FinBERT (avoids cold-start model download latency)
##  - Non-root runtime user
##  - Single virtual environment copied into final image
##  - Clear documentation of dependency divergence vs root requirements.txt
##
## NOTE: This service intentionally uses newer versions of several heavy ML
##       dependencies than the monorepo root `requirements.txt` (e.g. torch,
##       transformers, sentence-transformers). We keep them isolated here to
##       avoid forcing all lighter services to carry the heavier dependency
##       graph. A follow-up task will decide whether to uplift global pins or
##       introduce per-service constraints files.
## ============================================================================

########################
# Stage 1: Builder     #
########################
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /build

# System build deps (removed in final image)
RUN set -eux; \
        apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran make build-essential pkg-config \
    libopenblas-dev liblapack-dev curl ca-certificates \
    wget; \
        rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Core heavy ML deps - install torch first with CPU wheels index for determinism
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

# Remaining dependencies (grouped to maximize layer caching)
RUN pip install --no-cache-dir \
    torch-geometric==2.6.1 \
    networkx==3.3 \
    scipy==1.14.1 \
    statsmodels==0.14.4 \
    xgboost==2.1.2 \
    lightgbm==4.5.0 \
    backtesting==0.3.3 \
    quantstats==0.0.62 \
    empyrical==0.5.5 \
    hmmlearn==0.3.2 \
    transformers==4.46.3 \
    sentence-transformers==3.3.0 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scikit-learn==1.5.2 \
    ollama==0.4.4 \
    redis==5.2.1 \
    pydantic==2.9.0 \
    pydantic-settings==2.5.2 \
    aiohttp==3.11.10 \
    pyyaml==6.0.2 \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    structlog==24.1.0 \
    asyncpg==0.30.0 \
    pulsar-client==3.5.0 \
    psutil==5.9.8 \
    httpx==0.27.0 \
    sqlalchemy==2.0.25 \
    alembic==1.13.1 \
    optuna==4.5.0 \
    mlflow==2.18.0 \
    catboost==1.2.7 \
    bayesian-testing==0.6.2 \
    prometheus-client==0.21.0 \
    minio==7.2.7 \
    weaviate-client==4.6.2

# Install 'ta' library for technical indicators (used by our TA-Lib compat layer)
RUN pip install --no-cache-dir ta==0.11.0

# Pre-download FinBERT model (stored under HF cache) to avoid runtime cold starts
RUN python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print('Downloading FinBERT model...')
AutoTokenizer.from_pretrained('ProsusAI/finbert')
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
print('FinBERT model cached successfully.')
PY

# Prune bytecode caches to reduce image size (keep metadata intact)
RUN find /usr/local/lib/python3.11 -type d -name "__pycache__" -prune -exec rm -rf {} +

########################
# Stage 2: Runtime     #
########################
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PYTHONPATH=/app:/app/shared/python-common

WORKDIR /app

# Minimal runtime packages (curl only for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /root/.cache/huggingface ${HF_HOME}
# Create non-root user
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app ${VENV_PATH} ${HF_HOME}

# Refresh dynamic linker cache so TA-Lib is discoverable
RUN ldconfig

# Copy application code (after deps to leverage Docker layer caching)
COPY services/ml/ ./services/ml/
COPY shared/ ./shared/
# Model assets are provided at runtime via volume mounts; no need to bake them into the image

RUN chown -R appuser:appuser /app

USER appuser

ENV PATH="/usr/local/bin:${PATH}"

# Health check endpoint (fast-fail on connectivity issues)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

## Entrypoint (uvicorn FastAPI server)
EXPOSE 8001
CMD ["uvicorn", "services.ml.main:app", "--host", "0.0.0.0", "--port", "8001", "--proxy-headers"]
# Multi-stage build for AI Trading System API
# Build stage with minimal layers
FROM python:3.11-slim as builder

# Build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
# Install with specific optimizations for production
RUN pip install --no-cache-dir --user --compile --no-deps -r requirements.txt

# Production image - use distroless for smaller size and security
FROM python:3.11-slim

# Security: Add security-focused apt options
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Security: Create non-root user with specific UID/GID for consistency
RUN groupadd -g 1000 trading && \
    useradd -r -u 1000 -g trading -s /sbin/nologin trading && \
    mkdir -p /app && \
    chown -R trading:trading /app

WORKDIR /app

# Copy Python packages from builder with proper ownership
COPY --from=builder --chown=trading:trading /root/.local /home/trading/.local

# Copy only necessary application files (respecting .dockerignore)
COPY --chown=trading:trading api/ ./api/
COPY --chown=trading:trading services/ ./services/
COPY --chown=trading:trading shared/ ./shared/
COPY --chown=trading:trading config/*.py ./config/
COPY --chown=trading:trading config/logging.yaml ./config/logging.yaml
COPY --chown=trading:trading requirements.txt ./

# Security: Set restricted permissions
RUN chmod -R 755 /app && \
    find /app -type f -name "*.py" -exec chmod 644 {} \;

# Environment optimizations
ENV PATH=/home/trading/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Security: Switch to non-root user
USER trading

# Health check with proper timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Network optimizations
EXPOSE 8000

# Production server with optimized settings
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--access-log", \
     "--log-config", "/app/config/logging.yaml"]
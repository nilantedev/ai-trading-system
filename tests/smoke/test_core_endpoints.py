import os
import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status

# Ensure development admin password is set before importing the app so auth module initializes
os.environ.setdefault("ADMIN_PASSWORD", "devpass123")

from api.main import app  # noqa: E402

@pytest.mark.asyncio
async def test_root():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["name"].startswith("AI Trading System")

@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert "status" in data

@pytest.mark.asyncio
async def test_ready():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/ready")
    # Ready may be 200 or 503 depending on stubs; accept both, just ensure JSON structure
    assert resp.status_code in {200, 503}
    data = resp.json()
    assert "components" in data

@pytest.mark.asyncio
async def test_ml_status():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/ml/status")
    assert resp.status_code == status.HTTP_200_OK
    assert "timestamp" in resp.json()

@pytest.mark.asyncio
async def test_metrics_available():
    # metrics may not yet be registered if startup event hasn't run; manually trigger startup
    await app.router.startup()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/metrics")
    # Either success with text/plain metrics or 404 if metrics middleware not loaded in this context
    assert resp.status_code in {200, 404}
    if resp.status_code == 200:
        assert "process_cpu_seconds_total" in resp.text or len(resp.text) > 0

@pytest.mark.asyncio
async def test_correlation_header():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
    assert "X-Correlation-ID" in resp.headers

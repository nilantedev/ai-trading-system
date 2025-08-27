import asyncio
import os
import time
from types import SimpleNamespace
import pytest

# Import rate limiter
from api.rate_limiter import RedisRateLimiter


@pytest.mark.asyncio
async def test_degraded_mode_memory_fallback(monkeypatch):
    # Force development environment
    monkeypatch.setenv("ENVIRONMENT", "development")
    rl = RedisRateLimiter(redis_url="redis://invalid-host:6379")
    # Simulate failure to connect immediately by setting connected False and entering degraded mode manually
    rl.connected = False
    rl._enter_degraded_mode("test")  # type: ignore

    # Perform several requests; they should be allowed until limit reached
    allowed = 0
    for i in range(5):
        res = await rl.check_rate_limit(identifier=f"userX", limit_type="auth")
        if res["allowed"]:
            allowed += 1
    assert allowed > 0
    assert rl.degraded_mode is True

@pytest.mark.asyncio
async def test_cold_start_fail_closed_in_prod(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    rl = RedisRateLimiter(redis_url="redis://invalid-host:6379")
    rl.connected = False
    # Ensure not degraded yet (cold start)
    rl.degraded_mode = False
    res = await rl.check_rate_limit(identifier="ip1", limit_type="default")
    assert res["allowed"] is False
    assert res["error"].startswith("Rate limiter unavailable")

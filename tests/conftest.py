#!/usr/bin/env python3
"""
Simple pytest configuration without full app imports
"""

import asyncio
import pytest
import pytest_asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
import tempfile
import os


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    return {
        "symbol": "AAPL",
        "price": 150.25,
        "bid": 150.20,
        "ask": 150.30,
        "volume": 1000000,
        "timestamp": datetime.utcnow().isoformat(),
        "high": 151.00,
        "low": 149.50,
        "open": 150.00,
        "change": 0.25,
        "change_percent": 0.17
    }


@pytest.fixture
def sample_trading_signal():
    """Generate sample trading signal for testing."""
    return {
        "id": "signal_001",
        "symbol": "AAPL",
        "signal_type": "BUY",
        "strength": 0.85,
        "price_target": 155.00,
        "stop_loss": 145.00,
        "confidence": 0.92,
        "strategy": "momentum_breakout",
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": {
            "rsi": 65.2,
            "macd": 1.25,
            "volume_ratio": 1.8
        }
    }


@pytest.fixture
def sample_order():
    """Generate sample order for testing."""
    return {
        "id": "order_001",
        "symbol": "AAPL",
        "side": "BUY",
        "order_type": "MARKET",
        "quantity": 100,
        "price": 150.25,
        "status": "PENDING",
        "user_id": "test_user",
        "timestamp": datetime.utcnow().isoformat(),
        "filled_quantity": 0,
        "average_price": 0.0
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    mock.exists.return_value = False
    mock.keys.return_value = []
    return mock


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    mock = AsyncMock()
    mock.execute.return_value = MagicMock()
    mock.fetch.return_value = []
    mock.fetchrow.return_value = None
    mock.fetchval.return_value = None
    return mock


@pytest.fixture
def test_settings():
    """Provide test configuration settings."""
    return {
        "database": {
            "url": "postgresql://test:test@localhost:5433/test_db",
            "pool_size": 5
        },
        "redis": {
            "url": "redis://localhost:6380/0",
            "pool_size": 10
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8001,
            "admin_token": "test_admin_token"
        }
    }
#!/usr/bin/env python3
"""
pytest configuration and shared fixtures for the AI Trading System tests.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
import tempfile
import os
from datetime import datetime, timedelta
import json

# Import test dependencies
from fastapi.testclient import TestClient
import httpx

# Import application components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simplified imports for testing
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared', 'python-common'))
    from trading_common import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Import FastAPI app separately to avoid dependency issues during testing
try:
    from api.main import app
except ImportError:
    # Create minimal app for testing if main app import fails
    from fastapi import FastAPI
    app = FastAPI(title="Test API")

# Test configuration
TEST_SETTINGS = {
    "database": {
        "url": "postgresql://test:test@localhost:5433/test_trading_db",
        "pool_size": 5,
        "max_overflow": 10
    },
    "redis": {
        "url": "redis://localhost:6380/0",
        "pool_size": 10
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8001,
        "admin_token": "test_admin_token_123",
        "allowed_origins": ["*"],
        "allowed_hosts": ["*"]
    },
    "messaging": {
        "pulsar_url": "pulsar://localhost:6651"
    }
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Provide test configuration settings."""
    return TEST_SETTINGS.copy()


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    return MagicMock()


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_test_client():
    """Create async test client for FastAPI."""
    async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


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
def sample_portfolio():
    """Generate sample portfolio data for testing."""
    return {
        "user_id": "test_user",
        "total_value": 100000.00,
        "cash_balance": 25000.00,
        "invested_value": 75000.00,
        "day_change": 1250.50,
        "day_change_percent": 1.25,
        "positions": [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "average_price": 145.00,
                "current_price": 150.25,
                "market_value": 15025.00,
                "unrealized_pnl": 525.00,
                "unrealized_pnl_percent": 3.62
            }
        ]
    }


@pytest.fixture
def mock_market_data_service():
    """Mock market data service."""
    mock = AsyncMock()
    mock.get_current_data.return_value = {
        "AAPL": {
            "price": 150.25,
            "volume": 1000000,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    mock.get_historical_data.return_value = []
    mock.subscribe_to_symbol.return_value = True
    mock.unsubscribe_from_symbol.return_value = True
    return mock


@pytest.fixture
def mock_signal_service():
    """Mock signal generation service."""
    mock = AsyncMock()
    mock.get_current_signals.return_value = []
    mock.generate_signal.return_value = {
        "symbol": "AAPL",
        "signal_type": "BUY",
        "confidence": 0.85
    }
    mock.get_signal_history.return_value = []
    return mock


@pytest.fixture
def mock_order_service():
    """Mock order management service."""
    mock = AsyncMock()
    mock.place_order.return_value = {
        "id": "order_001",
        "status": "PENDING"
    }
    mock.get_order.return_value = None
    mock.cancel_order.return_value = True
    mock.get_user_orders.return_value = []
    return mock


@pytest.fixture
def mock_portfolio_service():
    """Mock portfolio service."""
    mock = AsyncMock()
    mock.get_portfolio.return_value = {
        "total_value": 100000.00,
        "positions": []
    }
    mock.get_positions.return_value = []
    mock.get_performance.return_value = {
        "total_return": 0.15,
        "sharpe_ratio": 1.2
    }
    return mock


@pytest.fixture
def mock_risk_service():
    """Mock risk monitoring service."""
    mock = AsyncMock()
    mock.check_risk_limits.return_value = True
    mock.get_risk_metrics.return_value = {
        "var": 0.05,
        "max_drawdown": 0.12
    }
    mock.validate_order.return_value = True
    return mock


@pytest.fixture
def temp_config_file():
    """Create temporary configuration file for testing."""
    config_data = {
        "database": {"url": "sqlite:///test.db"},
        "redis": {"url": "redis://localhost:6379/0"},
        "api": {"port": 8000}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def websocket_mock():
    """Mock WebSocket connection."""
    mock = AsyncMock()
    mock.accept.return_value = None
    mock.send_text.return_value = None
    mock.send_json.return_value = None
    mock.receive_text.return_value = '{"type": "test"}'
    mock.receive_json.return_value = {"type": "test"}
    mock.close.return_value = None
    return mock


# Async fixtures for services
@pytest_asyncio.fixture
async def mock_async_redis():
    """Async mock for Redis operations."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    mock.exists.return_value = False
    mock.close.return_value = None
    return mock


@pytest_asyncio.fixture
async def mock_async_database():
    """Async mock for database operations."""
    mock = AsyncMock()
    mock.execute.return_value = None
    mock.fetch.return_value = []
    mock.fetchrow.return_value = None
    mock.fetchval.return_value = None
    mock.close.return_value = None
    return mock


# Test data generators
def generate_historical_data(symbol: str, days: int = 30):
    """Generate historical market data for testing."""
    data = []
    base_price = 100.0
    
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=days-i)
        price = base_price + (i * 0.5) + ((-1) ** i * 2.0)  # Simple trend with noise
        
        data.append({
            "symbol": symbol,
            "timestamp": date.isoformat(),
            "open": price - 0.5,
            "high": price + 1.0,
            "low": price - 1.5,
            "close": price,
            "volume": 1000000 + (i * 10000)
        })
    
    return data


def generate_signal_data(symbol: str, count: int = 10):
    """Generate trading signals for testing."""
    signals = []
    signal_types = ["BUY", "SELL", "HOLD"]
    
    for i in range(count):
        signals.append({
            "id": f"signal_{i:03d}",
            "symbol": symbol,
            "signal_type": signal_types[i % len(signal_types)],
            "strength": 0.5 + (i * 0.05),
            "confidence": 0.7 + (i * 0.02),
            "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat()
        })
    
    return signals


# Markers for test categories
pytestmark = [
    pytest.mark.asyncio,
]
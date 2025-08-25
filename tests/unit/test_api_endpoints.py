#!/usr/bin/env python3
"""
Unit tests for API Endpoints
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import json

from fastapi.testclient import TestClient
import httpx

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.main import app


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest_asyncio.fixture
    async def async_client(self):
        """Create async test client."""
        async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
            yield client

    # Health Check Tests
    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AI Trading System API"
        assert "documentation" in data
        assert "endpoints" in data

    # Market Data API Tests
    @patch('api.routers.market_data.get_market_data_service')
    def test_get_market_data_success(self, mock_service, client, sample_market_data):
        """Test successful market data retrieval."""
        # Mock service
        mock_service_instance = AsyncMock()
        mock_service_instance.get_current_data.return_value = sample_market_data
        mock_service.return_value = mock_service_instance
        
        response = client.get("/api/v1/market-data/AAPL")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["symbol"] == "AAPL"

    @patch('api.routers.market_data.get_market_data_service')
    def test_get_market_data_not_found(self, mock_service, client):
        """Test market data not found."""
        mock_service_instance = AsyncMock()
        mock_service_instance.get_current_data.return_value = None
        mock_service.return_value = mock_service_instance
        
        response = client.get("/api/v1/market-data/INVALID")
        
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "DATA_NOT_FOUND"

    @patch('api.routers.market_data.get_market_data_service')
    def test_get_historical_data(self, mock_service, client):
        """Test historical data retrieval."""
        historical_data = [
            {
                "symbol": "AAPL",
                "timestamp": "2023-01-01T00:00:00",
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 152.0,
                "volume": 1000000
            }
        ]
        
        mock_service_instance = AsyncMock()
        mock_service_instance.get_historical_data.return_value = historical_data
        mock_service.return_value = mock_service_instance
        
        response = client.get("/api/v1/market-data/AAPL/history?days=1")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1

    # Trading Signals API Tests
    @patch('api.routers.trading.get_signal_service')
    def test_get_trading_signals(self, mock_service, client, sample_trading_signal):
        """Test trading signals retrieval."""
        mock_service_instance = AsyncMock()
        mock_service_instance.get_current_signals.return_value = [sample_trading_signal]
        mock_service.return_value = mock_service_instance
        
        response = client.get("/api/v1/signals")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["symbol"] == "AAPL"

    @patch('api.routers.trading.get_signal_service')
    def test_get_signals_by_symbol(self, mock_service, client, sample_trading_signal):
        """Test signals by symbol retrieval."""
        mock_service_instance = AsyncMock()
        mock_service_instance.get_signals_by_symbol.return_value = [sample_trading_signal]
        mock_service.return_value = mock_service_instance
        
        response = client.get("/api/v1/signals/AAPL")
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["symbol"] == "AAPL"

    # Order Management API Tests
    @patch('api.routers.trading.get_order_management_system')
    def test_place_order_success(self, mock_oms, client):
        """Test successful order placement."""
        mock_oms_instance = AsyncMock()
        mock_oms_instance.place_order.return_value = {
            "id": "order_001",
            "status": "PENDING",
            "symbol": "AAPL"
        }
        mock_oms.return_value = mock_oms_instance
        
        order_request = {
            "symbol": "AAPL",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": 100
        }
        
        response = client.post(
            "/api/v1/orders",
            json=order_request,
            headers={"Authorization": "Bearer demo_token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["symbol"] == "AAPL"

    def test_place_order_unauthorized(self, client):
        """Test order placement without authentication."""
        order_request = {
            "symbol": "AAPL", 
            "side": "BUY",
            "quantity": 100
        }
        
        response = client.post("/api/v1/orders", json=order_request)
        
        assert response.status_code == 403  # Should require authentication

    @patch('api.routers.trading.get_order_management_system')
    def test_get_order(self, mock_oms, client, sample_order):
        """Test order retrieval."""
        mock_oms_instance = AsyncMock()
        mock_oms_instance.get_order.return_value = sample_order
        mock_oms.return_value = mock_oms_instance
        
        response = client.get(
            "/api/v1/orders/order_001",
            headers={"Authorization": "Bearer demo_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["id"] == "order_001"

    @patch('api.routers.trading.get_order_management_system')
    def test_cancel_order(self, mock_oms, client):
        """Test order cancellation."""
        mock_oms_instance = AsyncMock()
        mock_oms_instance.cancel_order.return_value = True
        mock_oms.return_value = mock_oms_instance
        
        response = client.delete(
            "/api/v1/orders/order_001",
            headers={"Authorization": "Bearer demo_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    # Portfolio API Tests
    @patch('api.routers.portfolio.get_portfolio_service')
    def test_get_portfolio(self, mock_service, client, sample_portfolio):
        """Test portfolio retrieval."""
        mock_service_instance = AsyncMock()
        mock_service_instance.get_portfolio.return_value = sample_portfolio
        mock_service.return_value = mock_service_instance
        
        response = client.get(
            "/api/v1/portfolio",
            headers={"Authorization": "Bearer demo_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total_value"] == 100000.0

    @patch('api.routers.portfolio.get_portfolio_service')
    def test_get_positions(self, mock_service, client):
        """Test positions retrieval."""
        positions = [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "average_price": 150.0,
                "current_price": 155.0,
                "unrealized_pnl": 500.0
            }
        ]
        
        mock_service_instance = AsyncMock()
        mock_service_instance.get_positions.return_value = positions
        mock_service.return_value = mock_service_instance
        
        response = client.get(
            "/api/v1/positions",
            headers={"Authorization": "Bearer demo_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1

    # System API Tests
    def test_get_system_health(self, client):
        """Test system health endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data

    @patch('api.routers.system.get_all_services')
    def test_get_services_status(self, mock_get_services, client):
        """Test services status endpoint."""
        mock_services = {
            "market_data_service": {"status": "healthy"},
            "signal_service": {"status": "healthy"}
        }
        
        mock_get_services.return_value = mock_services
        
        response = client.get("/api/v1/services")
        
        assert response.status_code == 200
        data = response.json()
        assert "services" in data["data"]

    # Error Handling Tests
    def test_invalid_endpoint(self, client):
        """Test request to invalid endpoint."""
        response = client.get("/api/v1/invalid")
        
        assert response.status_code == 404

    def test_invalid_method(self, client):
        """Test invalid HTTP method."""
        response = client.patch("/api/v1/market-data/AAPL")
        
        assert response.status_code == 405  # Method not allowed

    def test_invalid_json(self, client):
        """Test request with invalid JSON."""
        response = client.post(
            "/api/v1/orders",
            data="invalid json",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer demo_token"
            }
        )
        
        assert response.status_code == 422

    # Rate Limiting Tests
    def test_rate_limiting(self, client):
        """Test API rate limiting."""
        # Make many requests quickly
        responses = []
        for i in range(65):  # Exceed rate limit of 60/minute
            response = client.get("/api/v1/market-data/AAPL")
            responses.append(response)
        
        # Should get rate limited
        rate_limited = any(r.status_code == 429 for r in responses[-5:])
        assert rate_limited

    # Authentication Tests
    def test_valid_token(self, client):
        """Test valid authentication token."""
        response = client.get(
            "/api/v1/portfolio",
            headers={"Authorization": "Bearer demo_token"}
        )
        
        # Should not be 401 (token is valid for demo)
        assert response.status_code != 401

    def test_invalid_token(self, client):
        """Test invalid authentication token."""
        response = client.get(
            "/api/v1/portfolio",
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == 401

    def test_missing_token(self, client):
        """Test missing authentication token."""
        response = client.get("/api/v1/portfolio")
        
        assert response.status_code == 403

    # Validation Tests
    def test_invalid_symbol_format(self, client):
        """Test invalid symbol format."""
        response = client.get("/api/v1/market-data/123INVALID")
        
        assert response.status_code == 422

    def test_invalid_order_data(self, client):
        """Test invalid order data validation."""
        invalid_order = {
            "symbol": "AAPL",
            "side": "INVALID_SIDE",  # Invalid side
            "quantity": -100  # Invalid negative quantity
        }
        
        response = client.post(
            "/api/v1/orders",
            json=invalid_order,
            headers={"Authorization": "Bearer demo_token"}
        )
        
        assert response.status_code == 422

    # CORS Tests
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/market-data/AAPL")
        
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "*"

    # Async Client Tests
    @pytest.mark.asyncio
    async def test_async_market_data(self, async_client):
        """Test async client for market data."""
        response = await async_client.get("/api/v1/market-data/AAPL")
        
        assert response.status_code in [200, 404]  # Either success or not found

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling of concurrent requests."""
        import asyncio
        
        tasks = [
            async_client.get("/api/v1/market-data/AAPL"),
            async_client.get("/api/v1/market-data/GOOGL"),
            async_client.get("/api/v1/market-data/MSFT")
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should complete
        assert len(responses) == 3
        assert all(r.status_code in [200, 404] for r in responses)

    # Content Type Tests
    def test_json_response_content_type(self, client):
        """Test that API returns JSON content type."""
        response = client.get("/health")
        
        assert response.headers["content-type"] == "application/json"

    def test_request_content_type_validation(self, client):
        """Test content type validation for POST requests."""
        response = client.post(
            "/api/v1/orders",
            data="not json",
            headers={
                "Content-Type": "text/plain",
                "Authorization": "Bearer demo_token"
            }
        )
        
        assert response.status_code == 422

    # Response Time Tests
    def test_response_time_header(self, client):
        """Test that response time header is included."""
        response = client.get("/health")
        
        assert "X-Response-Time" in response.headers
        assert response.headers["X-API-Version"] == "1.0.0"
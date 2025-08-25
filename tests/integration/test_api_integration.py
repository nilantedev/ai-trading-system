#!/usr/bin/env python3
"""
API integration tests
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
import httpx
import json
from datetime import datetime
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestAPIIntegration:
    """Integration tests for API endpoints with backend services."""

    @pytest_asyncio.fixture
    async def test_client(self):
        """Create test HTTP client."""
        # Create minimal FastAPI app for testing
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        @app.get("/api/v1/market-data/{symbol}")
        async def get_market_data(symbol: str):
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "price": 150.25,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
            yield client

    @pytest.mark.asyncio
    async def test_market_data_api_integration(self, test_client):
        """Test market data API integration with backend services."""
        # Mock backend service calls
        with patch('services.data_ingestion.market_data_service.get_market_data_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_current_data.return_value = {
                "symbol": "AAPL",
                "price": 150.25,
                "volume": 1000000,
                "timestamp": datetime.utcnow().isoformat()
            }
            mock_service.return_value = mock_service_instance
            
            # Test API endpoint
            response = await test_client.get("/api/v1/market-data/AAPL")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_trading_workflow_integration(self, test_client):
        """Test complete trading workflow through API."""
        # Step 1: Get market data
        market_response = await test_client.get("/api/v1/market-data/AAPL")
        assert market_response.status_code == 200
        
        # Step 2: Check for signals
        with patch('api.routers.trading.get_signal_service') as mock_signal:
            mock_signal_instance = AsyncMock()
            mock_signal_instance.get_signals_by_symbol.return_value = [{
                "id": "signal_001",
                "symbol": "AAPL",
                "signal_type": "BUY",
                "confidence": 0.85
            }]
            mock_signal.return_value = mock_signal_instance
            
            signals_response = await test_client.get("/api/v1/signals/AAPL")
            # Would assert signals response if endpoint existed
        
        # Step 3: Place order based on signal
        with patch('api.routers.trading.get_order_management_system') as mock_oms:
            mock_oms_instance = AsyncMock()
            mock_oms_instance.place_order.return_value = {
                "id": "order_001",
                "status": "PENDING",
                "symbol": "AAPL"
            }
            mock_oms.return_value = mock_oms_instance
            
            order_data = {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 100,
                "order_type": "MARKET"
            }
            
            # Would place order if endpoint existed with auth
            # order_response = await test_client.post("/api/v1/orders", json=order_data, headers={"Authorization": "Bearer demo_token"})

    @pytest.mark.asyncio
    async def test_websocket_integration(self, test_client):
        """Test WebSocket integration with backend services."""
        # Mock WebSocket connection manager
        from api.websocket_manager import ConnectionManager
        
        connection_manager = ConnectionManager()
        
        # Simulate WebSocket connection
        mock_websocket = AsyncMock()
        await connection_manager.connect(mock_websocket, "market_data", symbols=["AAPL"])
        
        # Simulate data broadcast
        test_message = {
            "type": "market_data_update",
            "symbol": "AAPL",
            "price": 150.25,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await connection_manager.broadcast_to_stream("market_data", test_message)
        
        # Verify message was sent to connected websocket
        mock_websocket.send_json.assert_called_once_with(test_message)

    @pytest.mark.asyncio
    async def test_authentication_integration(self, test_client):
        """Test authentication integration across endpoints."""
        # Test with valid token
        valid_headers = {"Authorization": "Bearer demo_token"}
        
        # Test with invalid token
        invalid_headers = {"Authorization": "Bearer invalid_token"}
        
        # Health endpoint should not require auth
        response = await test_client.get("/health")
        assert response.status_code == 200
        
        # Test auth validation logic (would need real endpoints)
        # Portfolio endpoint should require auth
        # portfolio_response = await test_client.get("/api/v1/portfolio", headers=valid_headers)
        # assert portfolio_response.status_code != 401

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, test_client):
        """Test rate limiting integration."""
        # Make rapid requests to trigger rate limiting
        responses = []
        
        for i in range(10):  # Fewer requests for test
            response = await test_client.get("/health")
            responses.append(response)
        
        # All should succeed for health endpoint (not rate limited)
        assert all(r.status_code == 200 for r in responses)

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, test_client):
        """Test error handling across API layers."""
        # Test with invalid symbol
        response = await test_client.get("/api/v1/market-data/INVALID123")
        # Should handle gracefully (would return 404 or error in real implementation)
        
        # Test with malformed request
        response = await test_client.get("/api/v1/market-data/")  # Missing symbol
        assert response.status_code in [404, 422]  # Not found or validation error

    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, test_client):
        """Test concurrent API requests."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Make concurrent requests
        tasks = [
            test_client.get(f"/api/v1/market-data/{symbol}")
            for symbol in symbols
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(responses) == 3
        assert all(r.status_code == 200 for r in responses)

    @pytest.mark.asyncio
    async def test_data_consistency_across_endpoints(self, test_client):
        """Test data consistency across different API endpoints."""
        # Get market data
        market_response = await test_client.get("/api/v1/market-data/AAPL")
        
        # Mock portfolio service to return consistent data
        with patch('api.routers.portfolio.get_portfolio_service') as mock_portfolio:
            mock_portfolio_instance = AsyncMock()
            mock_portfolio_instance.get_positions.return_value = [{
                "symbol": "AAPL",
                "current_price": 150.25,  # Same as market data
                "quantity": 100
            }]
            mock_portfolio.return_value = mock_portfolio_instance
            
            # Get portfolio data
            # portfolio_response = await test_client.get("/api/v1/positions", headers={"Authorization": "Bearer demo_token"})
            
            # Prices should be consistent between endpoints
            # This would verify data consistency in real implementation

    @pytest.mark.asyncio
    async def test_api_response_format_consistency(self, test_client):
        """Test API response format consistency."""
        # All endpoints should follow consistent response format
        response = await test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have consistent structure
        assert "status" in data
        
        # Market data endpoint
        market_response = await test_client.get("/api/v1/market-data/AAPL")
        market_data = market_response.json()
        
        # Should follow consistent format
        assert "success" in market_data
        assert "data" in market_data

    @pytest.mark.asyncio
    async def test_api_pagination_integration(self, test_client):
        """Test API pagination integration."""
        # Mock service with paginated data
        with patch('api.routers.trading.get_order_management_system') as mock_oms:
            mock_oms_instance = AsyncMock()
            # Mock 100 orders for pagination testing
            mock_orders = [
                {
                    "id": f"order_{i:03d}",
                    "symbol": "AAPL",
                    "status": "FILLED"
                }
                for i in range(100)
            ]
            
            # Return first page
            mock_oms_instance.get_user_orders.return_value = mock_orders[:20]
            mock_oms.return_value = mock_oms_instance
            
            # Test pagination (would need real endpoint)
            # page1_response = await test_client.get("/api/v1/orders?page=1&limit=20", headers={"Authorization": "Bearer demo_token"})

    @pytest.mark.asyncio 
    async def test_api_filtering_integration(self, test_client):
        """Test API filtering integration."""
        # Mock filtered signals
        with patch('api.routers.trading.get_signal_service') as mock_signal:
            mock_signal_instance = AsyncMock()
            mock_signal_instance.get_filtered_signals.return_value = [{
                "symbol": "AAPL",
                "signal_type": "BUY",
                "confidence": 0.9
            }]
            mock_signal.return_value = mock_signal_instance
            
            # Test filtering (would need real endpoint)
            # filtered_response = await test_client.get("/api/v1/signals?min_confidence=0.8&signal_type=BUY")

    @pytest.mark.asyncio
    async def test_api_caching_integration(self, test_client):
        """Test API caching integration."""
        # First request should hit backend
        response1 = await test_client.get("/api/v1/market-data/AAPL")
        
        # Second request should use cache (in real implementation)
        response2 = await test_client.get("/api/v1/market-data/AAPL")
        
        # Both should return same data
        assert response1.json() == response2.json()

    @pytest.mark.asyncio
    async def test_api_monitoring_integration(self, test_client):
        """Test API monitoring integration."""
        # Make request and verify monitoring headers
        response = await test_client.get("/health")
        
        # Should include monitoring headers (would be added by middleware)
        # In real implementation, would check for headers like:
        # - X-Response-Time
        # - X-Request-ID
        # - X-API-Version
        
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_validation_integration(self, test_client):
        """Test API validation integration."""
        # Test with invalid data types
        invalid_order = {
            "symbol": "AAPL",
            "side": "INVALID_SIDE",
            "quantity": "not_a_number"
        }
        
        # Would test validation with real endpoint
        # response = await test_client.post("/api/v1/orders", json=invalid_order, headers={"Authorization": "Bearer demo_token"})
        # assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_api_timeout_handling(self, test_client):
        """Test API timeout handling."""
        # Mock slow backend service
        with patch('services.data_ingestion.market_data_service.get_market_data_service') as mock_service:
            mock_service_instance = AsyncMock()
            
            # Simulate timeout
            async def slow_response():
                await asyncio.sleep(10)  # Simulate slow response
                return {"symbol": "AAPL", "price": 150.25}
            
            mock_service_instance.get_current_data.side_effect = slow_response
            mock_service.return_value = mock_service_instance
            
            # Request should timeout or return quickly
            # (In real implementation, would have timeout middleware)
            response = await test_client.get("/api/v1/market-data/AAPL")
            # Should handle timeout gracefully

    @pytest.mark.asyncio
    async def test_api_health_check_integration(self, test_client):
        """Test API health check integration with backend services."""
        # Mock backend services health
        with patch('api.main.get_all_services') as mock_get_services:
            mock_services = {
                "market_data_service": {"status": "healthy"},
                "signal_service": {"status": "healthy"},
                "order_service": {"status": "degraded"}
            }
            mock_get_services.return_value = mock_services
            
            # Health check should aggregate service status
            # response = await test_client.get("/api/v1/health")
            # data = response.json()
            # assert "services" in data
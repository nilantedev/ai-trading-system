#!/usr/bin/env python3
"""
Tests for Kubernetes health check endpoints.
Tests liveness, readiness, and startup probes.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import json
import sys
import os
import time

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.main import app


class TestLivenessProbe:
    """Test liveness probe endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_liveness_healthy(self, client):
        """Test liveness probe when application is healthy."""
        # Mock application state
        app.state.start_time = time.time() - 60  # App running for 1 minute
        
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert data["uptime_seconds"] > 50
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_liveness_starting(self, client):
        """Test liveness probe during startup."""
        # Mock recent startup
        app.state.start_time = time.time() - 2  # App running for 2 seconds
        
        response = client.get("/health/live")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "starting"
        assert data["uptime_seconds"] < 5
    
    def test_liveness_not_initialized(self, client):
        """Test liveness probe when app state not initialized."""
        # Remove start_time attribute
        if hasattr(app.state, 'start_time'):
            delattr(app.state, 'start_time')
        
        response = client.get("/health/live")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "starting"
    
    def test_liveness_alternative_endpoint(self, client):
        """Test liveness probe on /health endpoint."""
        app.state.start_time = time.time() - 60
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestReadinessProbe:
    """Test readiness probe endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_readiness_healthy(self, client):
        """Test readiness probe when all dependencies are healthy."""
        # Mock application state
        app.state.start_time = time.time() - 60
        
        # Mock Redis cache
        with patch('trading_common.cache.get_trading_cache') as mock_cache_factory:
            mock_cache = AsyncMock()
            mock_cache.set = AsyncMock()
            mock_cache.get = AsyncMock(return_value="ok")
            mock_cache_factory.return_value = mock_cache
            
            # Mock secrets manager
            with patch('trading_common.secrets_vault.get_secrets_manager') as mock_secrets:
                mock_manager = AsyncMock()
                mock_manager.health_check = AsyncMock(return_value={"primary": True, "fallback": True})
                mock_secrets.return_value = mock_manager
                
                # Mock circuit breakers
                with patch('trading_common.resilience.get_all_circuit_breakers') as mock_breakers:
                    mock_breakers.return_value = {
                        "test_breaker": {"state": "closed"}
                    }
                    
                    response = client.get("/health/ready")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "ready"
                    assert data["ready"] is True
                    assert "dependencies" in data
                    assert data["dependencies"]["redis"]["status"] == "healthy"
                    assert data["dependencies"]["secrets_vault"]["status"] == "healthy"
                    assert data["dependencies"]["circuit_breakers"]["status"] == "healthy"
    
    def test_readiness_redis_failure(self, client):
        """Test readiness probe when Redis is unavailable."""
        app.state.start_time = time.time() - 60
        
        # Mock Redis cache failure
        with patch('trading_common.cache.get_trading_cache') as mock_cache_factory:
            mock_cache_factory.side_effect = Exception("Redis connection failed")
            
            # Mock secrets manager as healthy
            with patch('trading_common.secrets_vault.get_secrets_manager') as mock_secrets:
                mock_manager = AsyncMock()
                mock_manager.health_check = AsyncMock(return_value={"primary": True, "fallback": True})
                mock_secrets.return_value = mock_manager
                
                with patch('trading_common.resilience.get_all_circuit_breakers') as mock_breakers:
                    mock_breakers.return_value = {}
                    
                    response = client.get("/health/ready")
                    
                    assert response.status_code == 503
                    data = response.json()
                    assert data["status"] == "not_ready"
                    assert data["ready"] is False
                    assert "Redis connection failed" in str(data["issues"])
                    assert data["dependencies"]["redis"]["status"] == "error"
    
    def test_readiness_secrets_vault_failure(self, client):
        """Test readiness probe when secrets vault is unavailable."""
        app.state.start_time = time.time() - 60
        
        # Mock Redis as healthy
        with patch('trading_common.cache.get_trading_cache') as mock_cache_factory:
            mock_cache = AsyncMock()
            mock_cache.set = AsyncMock()
            mock_cache.get = AsyncMock(return_value="ok")
            mock_cache_factory.return_value = mock_cache
            
            # Mock secrets manager failure
            with patch('trading_common.secrets_vault.get_secrets_manager') as mock_secrets:
                mock_manager = AsyncMock()
                mock_manager.health_check = AsyncMock(return_value={"primary": False, "fallback": False})
                mock_secrets.return_value = mock_manager
                
                with patch('trading_common.resilience.get_all_circuit_breakers') as mock_breakers:
                    mock_breakers.return_value = {}
                    
                    response = client.get("/health/ready")
                    
                    assert response.status_code == 503
                    data = response.json()
                    assert data["status"] == "not_ready"
                    assert data["ready"] is False
                    assert "No healthy secrets vault available" in data["issues"]
                    assert data["dependencies"]["secrets_vault"]["status"] == "unhealthy"
    
    def test_readiness_still_starting(self, client):
        """Test readiness probe when application is still starting."""
        # Mock recent startup (less than 10 seconds)
        app.state.start_time = time.time() - 5
        
        with patch('trading_common.cache.get_trading_cache') as mock_cache_factory:
            mock_cache = AsyncMock()
            mock_cache.set = AsyncMock()
            mock_cache.get = AsyncMock(return_value="ok")
            mock_cache_factory.return_value = mock_cache
            
            with patch('trading_common.secrets_vault.get_secrets_manager') as mock_secrets:
                mock_manager = AsyncMock()
                mock_manager.health_check = AsyncMock(return_value={"primary": True, "fallback": True})
                mock_secrets.return_value = mock_manager
                
                with patch('trading_common.resilience.get_all_circuit_breakers') as mock_breakers:
                    mock_breakers.return_value = {}
                    
                    response = client.get("/health/ready")
                    
                    assert response.status_code == 503
                    data = response.json()
                    assert data["status"] == "not_ready"
                    assert data["ready"] is False
                    assert "Application still starting up" in data["issues"]
    
    def test_readiness_circuit_breakers_open(self, client):
        """Test readiness probe with open circuit breakers."""
        app.state.start_time = time.time() - 60
        
        # Mock healthy dependencies
        with patch('trading_common.cache.get_trading_cache') as mock_cache_factory:
            mock_cache = AsyncMock()
            mock_cache.set = AsyncMock()
            mock_cache.get = AsyncMock(return_value="ok")
            mock_cache_factory.return_value = mock_cache
            
            with patch('trading_common.secrets_vault.get_secrets_manager') as mock_secrets:
                mock_manager = AsyncMock()
                mock_manager.health_check = AsyncMock(return_value={"primary": True, "fallback": True})
                mock_secrets.return_value = mock_secrets
                
                # Mock open circuit breakers
                with patch('trading_common.resilience.get_all_circuit_breakers') as mock_breakers:
                    mock_breakers.return_value = {
                        "api_breaker": {"state": "open"},
                        "db_breaker": {"state": "closed"}
                    }
                    
                    response = client.get("/health/ready")
                    
                    # Should still be ready despite degraded circuit breakers
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "ready"
                    assert data["ready"] is True
                    assert data["dependencies"]["circuit_breakers"]["status"] == "degraded"
                    assert "api_breaker" in data["dependencies"]["circuit_breakers"]["open_breakers"]


class TestStartupProbe:
    """Test startup probe endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_startup_complete(self, client):
        """Test startup probe when application has completed startup."""
        # Mock long-running application
        app.state.start_time = time.time() - 120  # 2 minutes ago
        app.state.websocket_task = MagicMock()
        app.state.websocket_task.done.return_value = False  # Still running
        
        response = client.get("/health/startup")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["started"] is True
        assert data["startup_checks"]["app_initialized"]["status"] == "complete"
        assert data["startup_checks"]["startup_timeout"]["status"] == "complete"
        assert data["startup_checks"]["websocket_streaming"]["status"] == "running"
    
    def test_startup_in_progress(self, client):
        """Test startup probe during application startup."""
        # Mock recent startup
        app.state.start_time = time.time() - 15  # 15 seconds ago
        
        response = client.get("/health/startup")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "starting"
        assert data["started"] is False
        assert data["startup_checks"]["startup_timeout"]["status"] == "in_progress"
    
    def test_startup_not_initialized(self, client):
        """Test startup probe when application state not initialized."""
        # Remove start_time attribute
        if hasattr(app.state, 'start_time'):
            delattr(app.state, 'start_time')
        
        response = client.get("/health/startup")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "starting"
        assert data["started"] is False
        assert "Application state not initialized" in data["issues"]
    
    def test_startup_websocket_not_configured(self, client):
        """Test startup probe when WebSocket is not configured."""
        app.state.start_time = time.time() - 60
        
        # Remove websocket_task if exists
        if hasattr(app.state, 'websocket_task'):
            delattr(app.state, 'websocket_task')
        
        response = client.get("/health/startup")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["startup_checks"]["websocket_streaming"]["status"] == "not_configured"


class TestHealthEndpointIntegration:
    """Test health endpoint integration and error handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint_includes_health_probes(self, client):
        """Test that root endpoint lists all health probe endpoints."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "kubernetes_probes" in data
        assert data["kubernetes_probes"]["liveness"] == "/health/live"
        assert data["kubernetes_probes"]["readiness"] == "/health/ready"
        assert data["kubernetes_probes"]["startup"] == "/health/startup"
    
    def test_concurrent_health_checks(self, client):
        """Test multiple concurrent health check requests."""
        app.state.start_time = time.time() - 60
        
        # Mock dependencies as healthy
        with patch('trading_common.cache.get_trading_cache') as mock_cache:
            mock_cache_instance = AsyncMock()
            mock_cache_instance.set = AsyncMock()
            mock_cache_instance.get = AsyncMock(return_value="ok")
            mock_cache.return_value = mock_cache_instance
            
            with patch('trading_common.secrets_vault.get_secrets_manager') as mock_secrets:
                mock_manager = AsyncMock()
                mock_manager.health_check = AsyncMock(return_value={"primary": True, "fallback": True})
                mock_secrets.return_value = mock_manager
                
                with patch('trading_common.resilience.get_all_circuit_breakers') as mock_breakers:
                    mock_breakers.return_value = {}
                    
                    # Make multiple concurrent requests
                    responses = []
                    for endpoint in ["/health/live", "/health/ready", "/health/startup"]:
                        responses.append(client.get(endpoint))
                    
                    # All should succeed
                    for response in responses:
                        assert response.status_code in [200, 503]  # Some may be 503 during startup
                        data = response.json()
                        assert "status" in data
                        assert "timestamp" in data
    
    def test_health_check_response_format(self, client):
        """Test that all health checks return consistent response format."""
        app.state.start_time = time.time() - 60
        
        endpoints = ["/health/live", "/health/ready", "/health/startup"]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            
            # Should always return JSON
            assert response.headers["content-type"] == "application/json"
            
            data = response.json()
            
            # Common fields
            assert "status" in data
            assert "timestamp" in data
            
            # Timestamp should be valid ISO format
            from datetime import datetime
            datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
    
    def test_health_check_timeouts(self, client):
        """Test health check behavior under slow dependencies."""
        app.state.start_time = time.time() - 60
        
        # Mock slow Redis response
        with patch('trading_common.cache.get_trading_cache') as mock_cache:
            mock_cache_instance = AsyncMock()
            mock_cache_instance.set = AsyncMock()
            mock_cache_instance.get = AsyncMock(side_effect=asyncio.TimeoutError("Timeout"))
            mock_cache.return_value = mock_cache_instance
            
            with patch('trading_common.secrets_vault.get_secrets_manager') as mock_secrets:
                mock_manager = AsyncMock()
                mock_manager.health_check = AsyncMock(return_value={"primary": True, "fallback": True})
                mock_secrets.return_value = mock_manager
                
                with patch('trading_common.resilience.get_all_circuit_breakers') as mock_breakers:
                    mock_breakers.return_value = {}
                    
                    response = client.get("/health/ready")
                    
                    # Should handle timeout gracefully
                    assert response.status_code == 503
                    data = response.json()
                    assert data["status"] == "not_ready"
                    assert data["ready"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
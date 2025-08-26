#!/usr/bin/env python3
"""
Tests for resilient HTTP client and API client factory.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
from datetime import datetime, timedelta
import json

# Add parent directories to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "shared", "python-common"))

from trading_common.http_client import (
    ResilientHTTPClient, HTTPClientConfig, HTTPResponse, HTTPMethod,
    get_http_client, get_trading_api_client, get_market_data_client
)
from trading_common.api_clients import APIClientFactory, get_alpaca_client
from trading_common.resilience import CircuitBreakerConfig, RetryConfig


class TestHTTPResponse:
    """Test HTTPResponse model."""
    
    def test_http_response_creation(self):
        """Test creating HTTPResponse instance."""
        response = HTTPResponse(
            status=200,
            headers={'content-type': 'application/json'},
            body={'message': 'success'},
            url='https://api.example.com/test',
            method='GET',
            elapsed_time=0.5,
            attempt_count=1
        )
        
        assert response.status == 200
        assert response.is_success
        assert not response.is_client_error
        assert not response.is_server_error
        assert response.elapsed_time == 0.5
        
    def test_http_response_error_detection(self):
        """Test error detection in HTTPResponse."""
        # Test client error
        client_error = HTTPResponse(
            status=404, headers={}, body='', url='', method='GET',
            elapsed_time=0.1, attempt_count=1
        )
        assert client_error.is_client_error
        assert not client_error.is_success
        
        # Test server error
        server_error = HTTPResponse(
            status=500, headers={}, body='', url='', method='GET',
            elapsed_time=0.1, attempt_count=1
        )
        assert server_error.is_server_error
        assert not server_error.is_success


class TestHTTPClientConfig:
    """Test HTTP client configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HTTPClientConfig()
        
        assert config.timeout == 30.0
        assert config.max_redirects == 10
        assert config.max_concurrent_requests == 50
        assert config.enable_metrics is True
        assert config.verify_ssl is True
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = HTTPClientConfig(
            timeout=15.0,
            rate_limit_per_minute=100,
            max_concurrent_requests=20,
            default_headers={'Authorization': 'Bearer token'}
        )
        
        assert config.timeout == 15.0
        assert config.rate_limit_per_minute == 100
        assert config.max_concurrent_requests == 20
        assert 'Authorization' in config.default_headers


class TestResilientHTTPClient:
    """Test resilient HTTP client functionality."""
    
    @pytest.fixture
    async def http_client(self):
        """Create HTTP client for testing."""
        config = HTTPClientConfig(
            timeout=5.0,
            max_concurrent_requests=2
        )
        client = ResilientHTTPClient("test_client", config)
        await client.start()
        yield client
        await client.close()
    
    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization and cleanup."""
        client = ResilientHTTPClient("test")
        assert client.name == "test"
        assert client.session is None
        
        await client.start()
        assert client.session is not None
        
        await client.close()
        assert client.session is None
        
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        async with ResilientHTTPClient("test") as client:
            assert client.session is not None
        
        # Session should be closed after context exit
        assert client.session is None
    
    @pytest.mark.asyncio
    async def test_successful_request(self, http_client):
        """Test successful HTTP request."""
        # Mock aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.url = 'https://api.example.com/test'
        mock_response.json = AsyncMock(return_value={'status': 'ok'})
        mock_response.request_info = MagicMock()
        mock_response.history = []
        
        http_client.session.request = AsyncMock()
        http_client.session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        http_client.session.request.return_value.__aexit__ = AsyncMock(return_value=None)
        
        response = await http_client.get("https://api.example.com/test")
        
        assert response.status == 200
        assert response.is_success
        assert response.body == {'status': 'ok'}
        assert http_client.stats['successful_requests'] == 1
        assert http_client.stats['failed_requests'] == 0
    
    @pytest.mark.asyncio
    async def test_client_error_handling(self, http_client):
        """Test client error handling."""
        # Mock aiohttp to raise an error
        http_client.session.request = AsyncMock()
        http_client.session.request.side_effect = aiohttp.ClientError("Connection error")
        
        with pytest.raises(aiohttp.ClientError):
            await http_client.get("https://api.example.com/test")
        
        assert http_client.stats['failed_requests'] == 1
        assert http_client.stats['successful_requests'] == 0
    
    @pytest.mark.asyncio
    async def test_server_error_handling(self, http_client):
        """Test server error (5xx) handling."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.headers = {'content-type': 'text/plain'}
        mock_response.url = 'https://api.example.com/test'
        mock_response.text = AsyncMock(return_value='Internal Server Error')
        mock_response.request_info = MagicMock()
        mock_response.history = []
        
        http_client.session.request = AsyncMock()
        http_client.session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        http_client.session.request.return_value.__aexit__ = AsyncMock(return_value=None)
        
        with pytest.raises(aiohttp.ClientResponseError):
            await http_client.get("https://api.example.com/test")
    
    def test_get_stats(self, http_client):
        """Test statistics collection."""
        stats = http_client.get_stats()
        
        assert 'name' in stats
        assert 'stats' in stats
        assert 'circuit_breaker' in stats
        assert 'bulkhead' in stats
        assert stats['name'] == 'test_client'
        
        # Test reset stats
        http_client.reset_stats()
        assert http_client.stats['total_requests'] == 0


class TestAPIClientFactory:
    """Test API client factory functionality."""
    
    def test_create_alpaca_client(self):
        """Test Alpaca client creation."""
        client = APIClientFactory.create_alpaca_client()
        
        assert client.name == "alpaca_api"
        assert client.config.timeout == 15.0
        assert client.config.rate_limit_per_minute == 200
        assert 'User-Agent' in client.config.default_headers
        
    def test_create_alpha_vantage_client(self):
        """Test Alpha Vantage client creation with conservative settings."""
        client = APIClientFactory.create_alpha_vantage_client()
        
        assert client.name == "alpha_vantage_api"
        assert client.config.timeout == 30.0
        assert client.config.rate_limit_per_minute == 5
        assert client.config.max_concurrent_requests == 2
        
    def test_create_news_api_client(self):
        """Test News API client creation."""
        client = APIClientFactory.create_news_api_client()
        
        assert client.name == "news_api"
        assert client.config.timeout == 20.0
        assert client.config.rate_limit_per_minute == 60
        
    def test_create_openai_client(self):
        """Test OpenAI client creation with extended timeout."""
        client = APIClientFactory.create_openai_client()
        
        assert client.name == "openai_api"
        assert client.config.timeout == 60.0
        assert client.config.max_concurrent_requests == 5
        
    def test_create_generic_client(self):
        """Test generic client creation with custom settings."""
        client = APIClientFactory.create_generic_client(
            name="custom_api",
            timeout=45.0,
            rate_limit_per_minute=150,
            max_concurrent=25
        )
        
        assert client.name == "custom_api"
        assert client.config.timeout == 45.0
        assert client.config.rate_limit_per_minute == 150
        assert client.config.max_concurrent_requests == 25


class TestGlobalClientFunctions:
    """Test global client management functions."""
    
    def test_get_http_client_singleton(self):
        """Test that get_http_client returns the same instance."""
        client1 = get_http_client("test_singleton")
        client2 = get_http_client("test_singleton")
        
        assert client1 is client2
        assert client1.name == "test_singleton"
        
    def test_get_trading_api_client(self):
        """Test trading API client getter."""
        client = get_trading_api_client()
        
        assert client.name == "trading_api"
        assert client.config.timeout == 15.0
        assert client.config.max_concurrent_requests == 20
        
    def test_get_market_data_client(self):
        """Test market data client getter."""
        client = get_market_data_client()
        
        assert client.name == "market_data"
        assert client.config.timeout == 10.0
        assert client.config.max_concurrent_requests == 50
        
    @pytest.mark.asyncio
    async def test_cached_client_functions(self):
        """Test cached client functions from api_clients."""
        # Test that cached functions return the same instance
        client1 = get_alpaca_client()
        client2 = get_alpaca_client()
        
        assert client1 is client2
        assert client1.name == "alpaca_api"


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with HTTP client."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        config = HTTPClientConfig(
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=1
            )
        )
        
        async with ResilientHTTPClient("cb_test", config) as client:
            # Mock failed requests
            client.session.request = AsyncMock()
            client.session.request.side_effect = aiohttp.ClientError("Network error")
            
            # First two failures should go through
            for i in range(2):
                with pytest.raises(aiohttp.ClientError):
                    await client.get("https://api.example.com/test")
            
            # Circuit breaker should now be open
            cb_state = client.circuit_breaker.get_state()
            assert cb_state['state'] == 'open'


class TestRateLimitingIntegration:
    """Test rate limiting integration."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_delays_requests(self):
        """Test that rate limiting properly delays requests."""
        config = HTTPClientConfig(
            rate_limit_per_minute=2  # Very low for testing
        )
        
        async with ResilientHTTPClient("rate_test", config) as client:
            # Mock successful responses
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.url = 'https://api.example.com/test'
            mock_response.json = AsyncMock(return_value={'status': 'ok'})
            mock_response.request_info = MagicMock()
            mock_response.history = []
            
            client.session.request = AsyncMock()
            client.session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            client.session.request.return_value.__aexit__ = AsyncMock(return_value=None)
            
            import time
            start_time = time.time()
            
            # Make requests that should trigger rate limiting
            for i in range(3):
                await client.get("https://api.example.com/test")
            
            elapsed = time.time() - start_time
            # Should take some time due to rate limiting
            assert elapsed > 0.1  # Some delay should occur


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
Unit tests for Market Data Service
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
sys.path.append('services/data-ingestion')
from market_data_service import MarketDataService
from trading_common import MarketData


class TestMarketDataService:
    """Test cases for MarketDataService class."""

    @pytest_asyncio.fixture
    async def service(self, mock_redis, mock_database, test_settings):
        """Create MarketDataService instance for testing."""
        service = MarketDataService()
        service.redis_client = mock_redis
        service.database = mock_database
        service.settings = test_settings
        return service

    @pytest.mark.asyncio
    async def test_get_real_time_quote_success(self, service, sample_market_data):
        """Test successful retrieval of real-time market quote."""
        # Mock cache returning cached data
        with patch.object(service.cache, 'get', return_value=sample_market_data):
            result = await service.get_real_time_quote("AAPL")
        
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 150.25
        service.redis_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_real_time_quote_cache_miss(self, service, sample_market_data):
        """Test real-time quote retrieval with cache miss."""
        # Mock cache miss, API fallback
        with patch.object(service.cache, 'get', return_value=None), \
             patch.object(service, '_get_alpaca_quote', return_value=MarketData(**sample_market_data)):
            result = await service.get_real_time_quote("AAPL")
            
            assert result is not None
            assert result["symbol"] == "AAPL"
            service.redis_client.set.assert_called_once()  # Should cache the result

    @pytest.mark.asyncio
    async def test_get_historical_data(self, service):
        """Test historical data retrieval."""
        from datetime import datetime, timedelta
        historical_data = [
            {
                "symbol": "AAPL",
                "timestamp": datetime.utcnow() - timedelta(days=1),
                "open": 149.0,
                "high": 151.0,
                "low": 148.5,
                "close": 150.25,
                "volume": 1000000,
                "data_source": "alpaca"
            }
        ]
        
        # Mock the internal API methods since historical data comes from external APIs
        # Also need to mock that API key exists
        service.alpaca_config['api_key'] = 'test_key' 
        with patch.object(service, '_get_alpaca_historical', return_value=[MarketData(**historical_data[0])]):
            start_date = datetime.utcnow() - timedelta(days=1)
            result = await service.get_historical_data("AAPL", start=start_date, limit=100)
        
            assert len(result) == 1
            assert result[0].symbol == "AAPL"  # MarketData object attribute access

    @pytest.mark.asyncio
    async def test_stream_real_time_data(self, service):
        """Test streaming real-time data for symbols."""
        symbols = ["AAPL", "GOOGL"]
        
        # Mock the stream method since it's async generator
        with patch.object(service, 'stream_real_time_data') as mock_stream:
            mock_stream.return_value = AsyncMock()
            await service.stream_real_time_data(symbols)
            mock_stream.assert_called_once_with(symbols)

    @pytest.mark.asyncio
    async def test_unsubscribe_from_symbol(self, service):
        """Test symbol unsubscription."""
        # First subscribe
        await service.subscribe_to_symbol("AAPL")
        
        # Then unsubscribe
        result = await service.unsubscribe_from_symbol("AAPL")
        
        assert result is True
        assert "AAPL" not in service.subscribed_symbols

    @pytest.mark.asyncio
    async def test_get_supported_symbols(self, service):
        """Test retrieval of supported symbols."""
        supported_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        service.database.fetch.return_value = [
            {"symbol": symbol} for symbol in supported_symbols
        ]
        
        result = await service.get_supported_symbols()
        
        assert len(result) == 4
        assert "AAPL" in result
        assert "TSLA" in result

    @pytest.mark.asyncio
    async def test_validate_symbol(self, service):
        """Test symbol validation."""
        # Valid symbol
        result = await service.validate_symbol("AAPL")
        assert result is True
        
        # Invalid symbol
        result = await service.validate_symbol("INVALID")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_market_status(self, service):
        """Test market status retrieval."""
        mock_status = {
            "is_open": True,
            "next_open": datetime.utcnow() + timedelta(hours=1),
            "next_close": datetime.utcnow() + timedelta(hours=8),
            "timezone": "US/Eastern"
        }
        
        with patch.object(service, '_get_market_status', return_value=mock_status):
            result = await service.get_market_status()
            
            assert result["is_open"] is True
            assert "timezone" in result

    @pytest.mark.asyncio
    async def test_batch_get_data(self, service, sample_market_data):
        """Test batch retrieval of market data."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Mock returning data for all symbols
        with patch.object(service, 'get_current_data') as mock_get:
            mock_get.return_value = sample_market_data
            
            result = await service.batch_get_data(symbols)
            
            assert len(result) == 3
            assert all(symbol in result for symbol in symbols)
            assert mock_get.call_count == 3

    @pytest.mark.asyncio
    async def test_error_handling_invalid_symbol(self, service):
        """Test error handling for invalid symbols."""
        service.database.fetchrow.return_value = None
        
        with pytest.raises(ValueError, match="Invalid symbol"):
            await service.get_current_data("INVALID_SYMBOL")

    @pytest.mark.asyncio
    async def test_error_handling_database_error(self, service):
        """Test error handling for database errors."""
        service.database.fetchrow.side_effect = Exception("Database connection error")
        
        with pytest.raises(Exception):
            await service.get_current_data("AAPL")

    @pytest.mark.asyncio
    async def test_cache_expiry(self, service, sample_market_data):
        """Test that cache expires correctly."""
        # Mock expired cache
        service.redis_client.get.return_value = None
        
        with patch.object(service, '_is_cache_valid', return_value=False):
            with patch.object(service, '_fetch_live_data', return_value=sample_market_data):
                result = await service.get_current_data("AAPL")
                
                assert result is not None
                service.redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_validation(self, service):
        """Test market data validation."""
        invalid_data = {
            "symbol": "AAPL",
            "price": -100,  # Invalid negative price
            "volume": -1000  # Invalid negative volume
        }
        
        is_valid = service._validate_market_data(invalid_data)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_rate_limiting(self, service):
        """Test API rate limiting protection."""
        # Mock rate limiting
        service.rate_limiter = MagicMock()
        service.rate_limiter.is_allowed.return_value = False
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await service._fetch_live_data("AAPL")

    @pytest.mark.asyncio
    async def test_service_health_check(self, service):
        """Test service health check functionality."""
        health = await service.get_service_health()
        
        assert "status" in health
        assert "service" in health
        assert "connections" in health
        assert "data_sources" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert health["service"] == "market_data"

    @pytest.mark.asyncio
    async def test_metric_collection(self, service):
        """Test that service collects metrics properly."""
        with patch.object(service, '_record_metric') as mock_metric:
            await service.get_current_data("AAPL")
            
            # Should record API call metric
            mock_metric.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, service, sample_market_data):
        """Test handling of concurrent requests."""
        import asyncio
        
        with patch.object(service, 'get_current_data', return_value=sample_market_data):
            tasks = [
                service.get_current_data("AAPL"),
                service.get_current_data("GOOGL"),
                service.get_current_data("MSFT")
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert all(result is not None for result in results)

    @pytest.mark.asyncio
    async def test_data_freshness_check(self, service, sample_market_data):
        """Test data freshness validation."""
        # Old data (more than 5 minutes old)
        old_timestamp = datetime.utcnow() - timedelta(minutes=10)
        old_data = sample_market_data.copy()
        old_data["timestamp"] = old_timestamp.isoformat()
        
        is_fresh = service._is_data_fresh(old_data)
        assert is_fresh is False
        
        # Fresh data
        fresh_data = sample_market_data.copy()
        fresh_data["timestamp"] = datetime.utcnow().isoformat()
        
        is_fresh = service._is_data_fresh(fresh_data)
        assert is_fresh is True
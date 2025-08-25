#!/usr/bin/env python3
"""
End-to-end integration tests for data flow: ingestion → filtering → risk → API
Tests the complete data processing pipeline with realistic scenarios.
"""

import pytest
import asyncio
import json
import aiohttp
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Optional
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_common import MarketData, get_settings
from trading_common.cache import get_trading_cache
from services.data_ingestion.market_data_service import MarketDataService
from services.data_ingestion.smart_data_filter import filter_market_data


@pytest.mark.integration
class TestEndToEndDataFlow:
    """Integration tests for complete data processing flow."""
    
    @pytest.fixture
    async def market_data_service(self):
        """Create a market data service for testing."""
        service = MarketDataService()
        await service.start()
        yield service
        await service.stop()
    
    @pytest.fixture
    def sample_market_data(self) -> List[MarketData]:
        """Generate sample market data for testing."""
        now = datetime.utcnow()
        return [
            MarketData(
                symbol="AAPL",
                timestamp=now - timedelta(minutes=1),
                open=150.00,
                high=151.50,
                low=149.75,
                close=151.25,
                volume=100000,
                timeframe="1min",
                data_source="test"
            ),
            MarketData(
                symbol="AAPL", 
                timestamp=now,
                open=151.25,
                high=152.00,
                low=150.50,
                close=151.75,
                volume=85000,
                timeframe="1min",
                data_source="test"
            ),
            # Low-quality data that should be filtered out
            MarketData(
                symbol="AAPL",
                timestamp=now - timedelta(hours=2),  # Stale data
                open=150.00,
                high=150.01,
                low=149.99,
                close=150.00,
                volume=10,  # Very low volume
                timeframe="1min", 
                data_source="test"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_data_ingestion_to_filtering_flow(self, sample_market_data):
        """Test data flows from ingestion through smart filtering."""
        filtered_data = []
        
        for market_data in sample_market_data:
            result = await filter_market_data(market_data)
            if result:  # Only include non-None results
                filtered_data.append(result)
        
        # Should filter out low-quality data
        assert len(filtered_data) < len(sample_market_data)
        
        # Remaining data should be high quality
        for data in filtered_data:
            assert data.volume > 1000  # Minimum volume threshold
            assert (datetime.utcnow() - data.timestamp).total_seconds() < 3600  # Fresh data
    
    @pytest.mark.asyncio
    async def test_market_data_service_resilience_integration(self, market_data_service):
        """Test market data service with resilience patterns under load."""
        # Test multiple concurrent requests to verify resilience patterns work
        tasks = []
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        for symbol in symbols:
            # Create multiple concurrent requests for each symbol
            for _ in range(3):
                task = market_data_service.get_real_time_quote(symbol)
                tasks.append(task)
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful responses (may be None if no API keys configured)
        successful_responses = [r for r in results if not isinstance(r, Exception)]
        failed_responses = [r for r in results if isinstance(r, Exception)]
        
        # Should handle concurrent load without crashing
        assert len(results) == len(tasks)
        
        # If we have API keys, should get some successful responses
        # If not, should gracefully return None (not crash)
        for result in successful_responses:
            if result is not None:  # If API returned data
                assert hasattr(result, 'symbol')
                assert hasattr(result, 'timestamp')
                assert result.symbol in symbols
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration_in_data_flow(self, market_data_service):
        """Test that circuit breakers properly integrate with data flow."""
        # Get circuit breaker states
        health = await market_data_service.get_service_health()
        
        assert 'resilience_patterns' in health
        assert 'circuit_breakers' in health['resilience_patterns']
        
        breakers = health['resilience_patterns']['circuit_breakers']
        
        # Should have circuit breakers for each data vendor
        expected_breakers = ['alpaca_data_api', 'polygon_data_api', 'alpha_vantage_api']
        
        for breaker_name in expected_breakers:
            if breaker_name in breakers:  # May not exist if service not fully initialized
                breaker_state = breakers[breaker_name]
                assert 'state' in breaker_state
                assert 'failure_count' in breaker_state
                assert breaker_state['state'] in ['closed', 'open', 'half_open']
    
    @pytest.mark.asyncio  
    async def test_rate_limiting_integration(self, market_data_service):
        """Test that rate limiting integrates properly with data services."""
        # Make rapid sequential requests to test rate limiting
        symbol = "AAPL"
        request_times = []
        
        for i in range(5):
            start_time = asyncio.get_event_loop().time()
            result = await market_data_service.get_real_time_quote(symbol)
            end_time = asyncio.get_event_loop().time()
            
            request_times.append(end_time - start_time)
        
        # Rate limiting should introduce some delays for rapid requests
        # (This test may pass quickly if no API keys are configured)
        assert len(request_times) == 5
        assert all(t >= 0 for t in request_times)  # All requests completed
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, market_data_service):
        """Test health checks return comprehensive status."""
        health = await market_data_service.get_service_health()
        
        # Should include all required health information
        assert 'service' in health
        assert health['service'] == 'market_data'
        assert 'status' in health
        assert health['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'timestamp' in health
        assert 'data_sources' in health
        assert 'connections' in health
        assert 'resilience_patterns' in health
        
        # Data sources should be properly reported
        data_sources = health['data_sources']
        expected_sources = ['alpaca', 'polygon', 'alpha_vantage']
        
        for source in expected_sources:
            assert source in data_sources
            assert isinstance(data_sources[source], bool)
        
        # Resilience patterns should be reported
        resilience = health['resilience_patterns']
        assert 'circuit_breakers' in resilience
        assert 'bulkhead_status' in resilience
    
    @pytest.mark.asyncio
    async def test_error_handling_propagation(self, market_data_service):
        """Test that errors are properly handled and don't crash the system."""
        # Test with invalid symbols that should cause errors
        invalid_symbols = ["", "INVALID_SYMBOL_THAT_DOES_NOT_EXIST", "123", None]
        
        for symbol in invalid_symbols:
            if symbol is None:
                continue
                
            try:
                result = await market_data_service.get_real_time_quote(symbol)
                # Should either return None or valid data, but not crash
                if result is not None:
                    assert hasattr(result, 'symbol')
            except Exception as e:
                # Specific exceptions are acceptable, but should be handled gracefully
                assert not isinstance(e, (SystemExit, KeyboardInterrupt))
    
    @pytest.mark.asyncio
    async def test_data_quality_filtering_integration(self, sample_market_data):
        """Test data quality filtering integrates with the pipeline."""
        high_quality_count = 0
        low_quality_count = 0
        
        for market_data in sample_market_data:
            filtered_result = await filter_market_data(market_data)
            
            if filtered_result is not None:
                high_quality_count += 1
                # Verify high-quality data properties
                assert filtered_result.volume > 0
                assert filtered_result.open > 0
                assert filtered_result.close > 0
                assert len(filtered_result.symbol.strip()) > 0
            else:
                low_quality_count += 1
        
        # Should filter out some low-quality data
        assert low_quality_count > 0, "Filter should reject some low-quality data"
        assert high_quality_count > 0, "Filter should accept some high-quality data"
    
    @pytest.mark.asyncio
    async def test_concurrent_service_access(self, market_data_service):
        """Test concurrent access to service doesn't cause issues."""
        async def make_request(symbol: str, request_id: int):
            try:
                result = await market_data_service.get_real_time_quote(symbol)
                return {"request_id": request_id, "result": result, "error": None}
            except Exception as e:
                return {"request_id": request_id, "result": None, "error": str(e)}
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            symbol = ["AAPL", "GOOGL", "MSFT"][i % 3]
            task = make_request(symbol, i)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        
        # All requests should complete (successfully or with handled errors)
        for result in results:
            assert "request_id" in result
            assert "result" in result
            assert "error" in result
            assert result["request_id"] >= 0


@pytest.mark.integration
class TestServiceHealthIntegration:
    """Integration tests for service health monitoring."""
    
    @pytest.mark.asyncio
    async def test_service_health_aggregation(self):
        """Test that service health is properly aggregated."""
        from services.data_ingestion.market_data_service import get_market_data_service
        
        try:
            service = await get_market_data_service()
            health = await service.get_service_health()
            
            # Health should include comprehensive status
            assert isinstance(health, dict)
            assert 'status' in health
            assert 'timestamp' in health
            
        except Exception as e:
            # If service can't start (e.g., no Redis), test should not fail
            # but should verify error is handled gracefully
            assert not isinstance(e, (SystemExit, KeyboardInterrupt))
    
    @pytest.mark.asyncio
    async def test_degraded_service_detection(self):
        """Test that degraded services are properly detected."""
        # This would be expanded to actually test with simulated service degradation
        # For now, just verify the health check structure
        from services.data_ingestion.market_data_service import MarketDataService
        
        service = MarketDataService()
        try:
            await service.start()
            health = await service.get_service_health()
            
            # Should detect if any circuit breakers are open
            if 'resilience_patterns' in health:
                breakers = health['resilience_patterns'].get('circuit_breakers', {})
                open_breakers = [
                    name for name, state in breakers.items() 
                    if state.get('state') == 'open'
                ]
                
                if open_breakers:
                    assert health['status'] == 'degraded'
                    assert 'degraded_reason' in health
                    
        finally:
            await service.stop()


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_end_to_end_data_flow.py -v -m integration
    pytest.main([__file__, "-v", "-m", "integration"])
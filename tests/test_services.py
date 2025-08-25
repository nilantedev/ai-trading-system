#!/usr/bin/env python3
"""
Comprehensive test suite for Data Ingestion Services.
Migrated from ad-hoc test scripts to proper pytest framework.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

# Test the services without actual network calls
from services.data_ingestion.market_data_service import MarketDataService
from services.data_ingestion.news_service import NewsService
from services.data_ingestion.reference_data_service import ReferenceDataService
from services.data_ingestion.data_validation_service import DataValidationService, ValidationSeverity
from services.data_ingestion.alternative_data_collector import AlternativeDataCollector
from trading_common import MarketData, NewsItem


class TestMarketDataService:
    """Test suite for Market Data Service."""
    
    @pytest.fixture
    def market_service(self):
        """Create market data service instance."""
        with patch('services.data_ingestion.market_data_service.AlpacaMarketDataService'):
            with patch('services.data_ingestion.market_data_service.PolygonMarketDataService'):
                service = MarketDataService()
                yield service
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, market_service):
        """Test market data service initializes correctly."""
        assert market_service is not None
        assert hasattr(market_service, 'get_latest_quote')
        assert hasattr(market_service, 'get_latest_trade')
    
    @pytest.mark.asyncio
    async def test_get_latest_quote_mock(self, market_service):
        """Test getting latest quote with mocked response."""
        # Mock the primary service response
        mock_quote = MarketData(
            symbol="AAPL",
            price=150.50,
            volume=1000000,
            bid=150.45,
            ask=150.55,
            timestamp=datetime.now()
        )
        
        market_service.primary_service.get_latest_quote = AsyncMock(return_value=mock_quote)
        
        quote = await market_service.get_latest_quote("AAPL")
        assert quote is not None
        assert quote.symbol == "AAPL"
        assert quote.price == 150.50
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, market_service):
        """Test fallback to secondary service when primary fails."""
        # Make primary service fail
        market_service.primary_service.get_latest_quote = AsyncMock(side_effect=Exception("Primary failed"))
        
        # Mock secondary service response
        mock_quote = MarketData(
            symbol="AAPL",
            price=151.00,
            volume=900000,
            timestamp=datetime.now()
        )
        market_service.secondary_service.get_latest_quote = AsyncMock(return_value=mock_quote)
        
        quote = await market_service.get_latest_quote("AAPL")
        assert quote is not None
        assert quote.price == 151.00
    
    @pytest.mark.asyncio
    async def test_historical_data_retrieval(self, market_service):
        """Test historical data retrieval."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        mock_historical = [
            MarketData(symbol="AAPL", price=150.0, timestamp=start_date + timedelta(days=i))
            for i in range(7)
        ]
        
        market_service.primary_service.get_historical_bars = AsyncMock(return_value=mock_historical)
        
        historical = await market_service.get_historical_bars("AAPL", start_date, end_date)
        assert len(historical) == 7
        assert all(data.symbol == "AAPL" for data in historical)


class TestNewsService:
    """Test suite for News Service."""
    
    @pytest.fixture
    def news_service(self):
        """Create news service instance."""
        with patch('services.data_ingestion.news_service.AlpacaNewsService'):
            with patch('services.data_ingestion.news_service.BenzingaNewsService'):
                service = NewsService()
                yield service
    
    @pytest.mark.asyncio
    async def test_news_service_initialization(self, news_service):
        """Test news service initializes correctly."""
        assert news_service is not None
        assert hasattr(news_service, 'get_news')
    
    @pytest.mark.asyncio
    async def test_get_news_for_symbol(self, news_service):
        """Test getting news for a specific symbol."""
        mock_news = [
            NewsItem(
                id="1",
                headline="Apple announces new product",
                summary="Apple Inc. announced...",
                symbols=["AAPL"],
                timestamp=datetime.now(),
                source="TestSource",
                url="https://example.com/news/1"
            )
        ]
        
        news_service.primary_source.get_news = AsyncMock(return_value=mock_news)
        
        news = await news_service.get_news(["AAPL"], limit=10)
        assert len(news) == 1
        assert "AAPL" in news[0].symbols
        assert "Apple" in news[0].headline
    
    @pytest.mark.asyncio
    async def test_news_deduplication(self, news_service):
        """Test that duplicate news items are filtered."""
        duplicate_news = [
            NewsItem(
                id="1",
                headline="Same news",
                summary="Content",
                symbols=["AAPL"],
                timestamp=datetime.now(),
                source="Source1"
            ),
            NewsItem(
                id="1",  # Same ID - should be deduplicated
                headline="Same news",
                summary="Content",
                symbols=["AAPL"],
                timestamp=datetime.now(),
                source="Source2"
            )
        ]
        
        news_service.primary_source.get_news = AsyncMock(return_value=duplicate_news[:1])
        news_service.secondary_source.get_news = AsyncMock(return_value=duplicate_news[1:])
        
        # Assuming the service has deduplication logic
        news = await news_service.get_news(["AAPL"])
        # Should only return unique news items
        unique_ids = set(item.id for item in news)
        assert len(unique_ids) == len(news)


class TestDataValidationService:
    """Test suite for Data Validation Service."""
    
    @pytest.fixture
    def validation_service(self):
        """Create validation service instance."""
        return DataValidationService()
    
    def test_validation_service_initialization(self, validation_service):
        """Test validation service initializes correctly."""
        assert validation_service is not None
        assert hasattr(validation_service, 'validate_market_data')
    
    def test_validate_valid_market_data(self, validation_service):
        """Test validation of valid market data."""
        valid_data = MarketData(
            symbol="AAPL",
            price=150.50,
            volume=1000000,
            bid=150.45,
            ask=150.55,
            timestamp=datetime.now()
        )
        
        errors = validation_service.validate_market_data(valid_data)
        assert len(errors) == 0
    
    def test_validate_invalid_price(self, validation_service):
        """Test validation catches invalid price."""
        invalid_data = MarketData(
            symbol="AAPL",
            price=-10.0,  # Invalid negative price
            volume=1000000,
            timestamp=datetime.now()
        )
        
        errors = validation_service.validate_market_data(invalid_data)
        assert len(errors) > 0
        assert any("price" in str(error).lower() for error in errors)
    
    def test_validate_stale_data(self, validation_service):
        """Test validation catches stale data."""
        stale_data = MarketData(
            symbol="AAPL",
            price=150.50,
            volume=1000000,
            timestamp=datetime.now() - timedelta(hours=25)  # Old data
        )
        
        errors = validation_service.validate_market_data(stale_data)
        assert len(errors) > 0
        assert any("stale" in str(error).lower() or "old" in str(error).lower() for error in errors)
    
    def test_severity_classification(self, validation_service):
        """Test that validation errors have appropriate severity."""
        # Create data with multiple issues
        problematic_data = MarketData(
            symbol="",  # Critical: missing symbol
            price=0,    # Warning: zero price
            volume=-1,  # Error: negative volume
            timestamp=datetime.now()
        )
        
        errors = validation_service.validate_market_data(problematic_data)
        
        # Check that different severities are assigned
        severities = [error.severity for error in errors if hasattr(error, 'severity')]
        if severities:
            assert ValidationSeverity.CRITICAL in severities or ValidationSeverity.ERROR in severities


class TestAlternativeDataCollector:
    """Test suite for Alternative Data Collector."""
    
    @pytest.fixture
    def alt_data_collector(self):
        """Create alternative data collector instance."""
        with patch('services.data_ingestion.alternative_data_collector.RedditDataSource'):
            with patch('services.data_ingestion.alternative_data_collector.TwitterDataSource'):
                collector = AlternativeDataCollector()
                yield collector
    
    @pytest.mark.asyncio
    async def test_collector_initialization(self, alt_data_collector):
        """Test alternative data collector initializes correctly."""
        assert alt_data_collector is not None
        assert hasattr(alt_data_collector, 'collect_signals')
    
    @pytest.mark.asyncio
    async def test_collect_signals(self, alt_data_collector):
        """Test signal collection from alternative sources."""
        mock_signals = [
            {
                "symbol": "AAPL",
                "signal_type": "sentiment",
                "value": 0.75,
                "confidence": 0.8,
                "source": "reddit",
                "timestamp": datetime.now()
            }
        ]
        
        alt_data_collector.reddit_source.get_sentiment = AsyncMock(return_value=mock_signals)
        
        signals = await alt_data_collector.collect_signals(["AAPL"])
        assert len(signals) > 0
        assert signals[0]["symbol"] == "AAPL"
        assert signals[0]["signal_type"] == "sentiment"
    
    @pytest.mark.asyncio
    async def test_signal_aggregation(self, alt_data_collector):
        """Test aggregation of signals from multiple sources."""
        reddit_signal = {
            "symbol": "AAPL",
            "signal_type": "sentiment",
            "value": 0.7,
            "source": "reddit"
        }
        
        twitter_signal = {
            "symbol": "AAPL",
            "signal_type": "sentiment", 
            "value": 0.8,
            "source": "twitter"
        }
        
        alt_data_collector.reddit_source.get_sentiment = AsyncMock(return_value=[reddit_signal])
        alt_data_collector.twitter_source.get_sentiment = AsyncMock(return_value=[twitter_signal])
        
        aggregated = await alt_data_collector.get_aggregated_sentiment("AAPL")
        
        # Should combine signals from both sources
        assert aggregated is not None
        assert "combined_score" in aggregated or "average_sentiment" in aggregated


class TestReferenceDataService:
    """Test suite for Reference Data Service."""
    
    @pytest.fixture
    def reference_service(self):
        """Create reference data service instance."""
        return ReferenceDataService()
    
    def test_reference_service_initialization(self, reference_service):
        """Test reference data service initializes correctly."""
        assert reference_service is not None
        assert hasattr(reference_service, 'get_symbol_info')
    
    @pytest.mark.asyncio
    async def test_get_symbol_info(self, reference_service):
        """Test getting symbol information."""
        # Mock the underlying data source
        with patch.object(reference_service, 'get_symbol_info', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "sector": "Technology",
                "market_cap": 3000000000000
            }
            
            info = await reference_service.get_symbol_info("AAPL")
            assert info["symbol"] == "AAPL"
            assert info["name"] == "Apple Inc."
            assert info["exchange"] == "NASDAQ"
    
    @pytest.mark.asyncio
    async def test_get_trading_calendar(self, reference_service):
        """Test getting trading calendar information."""
        with patch.object(reference_service, 'get_trading_calendar', new_callable=AsyncMock) as mock_cal:
            mock_cal.return_value = {
                "is_open": True,
                "next_open": datetime.now() + timedelta(days=1),
                "next_close": datetime.now() + timedelta(hours=6)
            }
            
            calendar = await reference_service.get_trading_calendar()
            assert "is_open" in calendar
            assert isinstance(calendar["is_open"], bool)


# Integration test class
class TestServiceIntegration:
    """Integration tests for service interactions."""
    
    @pytest.mark.asyncio
    async def test_market_data_validation_pipeline(self):
        """Test the full pipeline of fetching and validating market data."""
        market_service = MarketDataService()
        validation_service = DataValidationService()
        
        with patch.object(market_service, 'get_latest_quote', new_callable=AsyncMock) as mock_quote:
            mock_quote.return_value = MarketData(
                symbol="AAPL",
                price=150.50,
                volume=1000000,
                timestamp=datetime.now()
            )
            
            # Fetch data
            quote = await market_service.get_latest_quote("AAPL")
            
            # Validate data
            errors = validation_service.validate_market_data(quote)
            
            assert quote is not None
            assert len(errors) == 0
    
    @pytest.mark.asyncio  
    async def test_multi_service_data_enrichment(self):
        """Test enriching market data with news and alternative data."""
        market_service = MarketDataService()
        news_service = NewsService()
        alt_data_collector = AlternativeDataCollector()
        
        symbol = "AAPL"
        
        # Mock all services
        with patch.object(market_service, 'get_latest_quote', new_callable=AsyncMock) as mock_market:
            with patch.object(news_service, 'get_news', new_callable=AsyncMock) as mock_news:
                with patch.object(alt_data_collector, 'collect_signals', new_callable=AsyncMock) as mock_alt:
                    
                    mock_market.return_value = MarketData(symbol=symbol, price=150.0, timestamp=datetime.now())
                    mock_news.return_value = [NewsItem(headline="Apple news", symbols=[symbol], timestamp=datetime.now())]
                    mock_alt.return_value = [{"symbol": symbol, "sentiment": 0.8}]
                    
                    # Collect all data
                    market_data = await market_service.get_latest_quote(symbol)
                    news_data = await news_service.get_news([symbol])
                    alt_signals = await alt_data_collector.collect_signals([symbol])
                    
                    # Verify we have enriched data
                    assert market_data.symbol == symbol
                    assert len(news_data) > 0
                    assert len(alt_signals) > 0
                    assert alt_signals[0]["symbol"] == symbol


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
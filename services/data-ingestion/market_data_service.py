#!/usr/bin/env python3
"""Market Data Service - Real-time market data ingestion and processing."""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import asdict
import os

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_pulsar_client
from trading_common.config_secrets import get_enhanced_settings_async
from trading_common.secrets_vault import get_api_secrets
from trading_common.resilience import (
    get_circuit_breaker, CircuitBreakerConfig, RetryStrategy, RetryConfig,
    RateLimiter, BulkheadPool, with_retry, with_circuit_breaker
)
from .smart_data_filter import filter_market_data, get_filtering_performance

logger = get_logger(__name__)
settings = get_settings()


class MarketDataService:
    """Handles real-time market data ingestion from multiple sources with resilience patterns."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = None
        self.pulsar_client = None
        self.producer = None
        
        # API configurations - will be loaded from vault
        self.alpaca_config = {
            'api_key': None,
            'secret_key': None,
            'base_url': 'https://paper-api.alpaca.markets' if settings.trading.paper_trading else 'https://api.alpaca.markets',
            'data_url': 'https://data.alpaca.markets'
        }
        
        self.polygon_config = {
            'api_key': None,
            'base_url': 'https://api.polygon.io'
        }
        
        self.alpha_vantage_config = {
            'api_key': None,
            'base_url': 'https://www.alphavantage.co/query'
        }
        
        # Initialize resilience patterns for each data vendor
        self._init_resilience_patterns()
    
    def _init_resilience_patterns(self):
        """Initialize resilience patterns for external data vendors."""
        # Circuit breaker configurations for each vendor
        alpaca_cb_config = CircuitBreakerConfig(
            failure_threshold=3,  # Open after 3 failures
            recovery_timeout=30,  # Try recovery after 30 seconds
            success_threshold=2   # Close after 2 successes
        )
        polygon_cb_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=45,  # Polygon might need longer recovery
            success_threshold=2
        )
        alpha_vantage_cb_config = CircuitBreakerConfig(
            failure_threshold=2,  # Alpha Vantage has stricter limits
            recovery_timeout=60,  # Longer recovery due to rate limits
            success_threshold=1
        )
        
        # Initialize circuit breakers
        self.alpaca_circuit_breaker = get_circuit_breaker("alpaca_data_api", alpaca_cb_config)
        self.polygon_circuit_breaker = get_circuit_breaker("polygon_data_api", polygon_cb_config)
        self.alpha_vantage_circuit_breaker = get_circuit_breaker("alpha_vantage_api", alpha_vantage_cb_config)
        
        # Retry strategies for each vendor
        self.alpaca_retry = RetryStrategy(RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True
        ))
        self.polygon_retry = RetryStrategy(RetryConfig(
            max_attempts=3,
            initial_delay=1.5,
            max_delay=15.0,
            exponential_base=2.0,
            jitter=True
        ))
        self.alpha_vantage_retry = RetryStrategy(RetryConfig(
            max_attempts=2,  # Lower attempts due to stricter rate limits
            initial_delay=2.0,
            max_delay=30.0,
            exponential_base=2.5,
            jitter=True
        ))
        
        # Rate limiters based on each vendor's API limits
        self.alpaca_rate_limiter = RateLimiter(rate=200, per=60)  # 200 requests/minute
        self.polygon_rate_limiter = RateLimiter(rate=100, per=60)  # 100 requests/minute (basic plan)
        self.alpha_vantage_rate_limiter = RateLimiter(rate=5, per=60)  # 5 requests/minute (free tier)
        
        # Bulkhead pools for request isolation
        self.alpaca_bulkhead = BulkheadPool(max_concurrent=5)
        self.polygon_bulkhead = BulkheadPool(max_concurrent=3)
        self.alpha_vantage_bulkhead = BulkheadPool(max_concurrent=1)  # Very conservative
        
        logger.info("Resilience patterns initialized for all data vendors")
    
    async def _load_api_secrets(self):
        """Load API secrets from vault."""
        try:
            # Load API secrets from vault with fallback to environment
            api_secrets = await get_api_secrets()
            
            # Update configurations with vault secrets
            if api_secrets.get('alpaca_api_key'):
                self.alpaca_config['api_key'] = api_secrets['alpaca_api_key']
            else:
                # Fallback to environment
                self.alpaca_config['api_key'] = os.getenv('ALPACA_API_KEY')
                
            if api_secrets.get('alpaca_secret_key'):
                self.alpaca_config['secret_key'] = api_secrets['alpaca_secret_key']
            else:
                self.alpaca_config['secret_key'] = os.getenv('ALPACA_SECRET_KEY')
                
            if api_secrets.get('polygon_api_key'):
                self.polygon_config['api_key'] = api_secrets['polygon_api_key']
            else:
                self.polygon_config['api_key'] = os.getenv('POLYGON_API_KEY')
                
            if api_secrets.get('alpha_vantage_api_key'):
                self.alpha_vantage_config['api_key'] = api_secrets['alpha_vantage_api_key']
            else:
                self.alpha_vantage_config['api_key'] = os.getenv('ALPHA_VANTAGE_API_KEY')
            
            # Log which APIs are configured (without revealing keys)
            configured_apis = []
            if self.alpaca_config['api_key']:
                configured_apis.append("Alpaca")
            if self.polygon_config['api_key']:
                configured_apis.append("Polygon")
            if self.alpha_vantage_config['api_key']:
                configured_apis.append("Alpha Vantage")
            
            logger.info(f"✅ API secrets loaded from vault. Configured APIs: {', '.join(configured_apis) or 'None'}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load API secrets from vault, falling back to environment: {e}")
            # Fallback to environment variables
            self.alpaca_config.update({
                'api_key': os.getenv('ALPACA_API_KEY'),
                'secret_key': os.getenv('ALPACA_SECRET_KEY'),
            })
            self.polygon_config['api_key'] = os.getenv('POLYGON_API_KEY')
            self.alpha_vantage_config['api_key'] = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    async def start(self):
        """Initialize service connections."""
        logger.info("Starting Market Data Service")
        
        # Load API secrets from vault first
        await self._load_api_secrets()
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Trading-System/1.0'}
        )
        
        # Initialize cache
        self.cache = get_trading_cache()
        
        # Initialize message producer
        try:
            self.pulsar_client = get_pulsar_client()
            self.producer = self.pulsar_client.create_producer(
                topic='persistent://trading/development/market-data',
                producer_name='market-data-service'
            )
            logger.info("Connected to message system")
        except Exception as e:
            logger.warning(f"Failed to connect to message system: {e}")
    
    async def stop(self):
        """Cleanup service connections."""
        if self.session:
            await self.session.close()
        if self.producer:
            self.producer.close()
        if self.pulsar_client:
            self.pulsar_client.close()
        logger.info("Market Data Service stopped")
    
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote for a symbol."""
        # Try multiple data sources with fallback
        quote = None
        
        if self.alpaca_config['api_key']:
            quote = await self._get_alpaca_quote(symbol)
        
        if not quote and self.polygon_config['api_key']:
            quote = await self._get_polygon_quote(symbol)
        
        if not quote and self.alpha_vantage_config['api_key']:
            quote = await self._get_alpha_vantage_quote(symbol)
        
        if quote:
            # Apply smart filtering to only process high-value data
            filtered_quote = await filter_market_data(quote)
            
            if filtered_quote:
                # Cache the filtered data
                if self.cache:
                    await self.cache.set_market_data(filtered_quote)
                
                # Publish to message stream
                if self.producer:
                    try:
                        await self._publish_market_data(filtered_quote)
                    except Exception as e:
                        logger.warning(f"Failed to publish market data: {e}")
                
                return filtered_quote
            else:
                # Data was filtered out - log for monitoring
                logger.debug(f"Market data for {quote.symbol} filtered out as low-value")
                return None
        
        return quote
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1min',
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MarketData]:
        """Get historical market data."""
        if not start:
            start = datetime.utcnow() - timedelta(days=1)
        if not end:
            end = datetime.utcnow()
        
        # Try data sources in order of preference
        data = []
        
        if self.alpaca_config['api_key']:
            data = await self._get_alpaca_historical(symbol, timeframe, start, end, limit)
        
        if not data and self.polygon_config['api_key']:
            data = await self._get_polygon_historical(symbol, timeframe, start, end, limit)
        
        # Cache historical data
        if data and self.cache:
            for bar in data:
                await self.cache.set_market_data(bar)
        
        return data
    
    async def stream_real_time_data(
        self, 
        symbols: List[str]
    ) -> AsyncGenerator[MarketData, None]:
        """Stream real-time data for multiple symbols."""
        logger.info(f"Starting real-time stream for {len(symbols)} symbols: {symbols}")
        
        while True:
            try:
                # Poll each symbol for updates
                for symbol in symbols:
                    quote = await self.get_real_time_quote(symbol)
                    if quote:
                        yield quote
                
                # Wait before next poll (adjust based on API rate limits)
                await asyncio.sleep(1.0)  # 1 second interval
                
            except Exception as e:
                logger.error(f"Error in real-time stream: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def _get_alpaca_quote(self, symbol: str) -> Optional[MarketData]:
        """Get quote from Alpaca API with resilience patterns."""
        async def _make_alpaca_request():
            # Rate limiting
            if not await self.alpaca_rate_limiter.acquire():
                raise Exception(f"Alpaca rate limit exceeded for {symbol}")
            
            url = f"{self.alpaca_config['data_url']}/v2/stocks/{symbol}/quotes/latest"
            headers = {
                'APCA-API-KEY-ID': self.alpaca_config['api_key'],
                'APCA-API-SECRET-KEY': self.alpaca_config['secret_key']
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    quote_data = data.get('quote', {})
                    
                    return MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromisoformat(quote_data['t'].replace('Z', '+00:00')),
                        open=quote_data.get('o', 0.0),
                        high=quote_data.get('h', 0.0),
                        low=quote_data.get('l', 0.0),
                        close=quote_data.get('c', 0.0),
                        volume=quote_data.get('v', 0),
                        timeframe='quote',
                        data_source='alpaca'
                    )
                elif response.status == 429:
                    raise Exception(f"Alpaca rate limit hit for {symbol}")
                elif response.status >= 500:
                    raise Exception(f"Alpaca server error {response.status} for {symbol}")
                else:
                    logger.warning(f"Alpaca API error {response.status} for {symbol}")
                    return None
        
        try:
            # Execute with full resilience stack: bulkhead -> circuit breaker -> retry
            return await self.alpaca_bulkhead.execute(
                lambda: self.alpaca_circuit_breaker.call(
                    lambda: self.alpaca_retry.execute(_make_alpaca_request)
                )
            )
        except Exception as e:
            logger.error(f"Alpaca quote error for {symbol} after resilience patterns: {e}")
            return None
    
    async def _get_polygon_quote(self, symbol: str) -> Optional[MarketData]:
        """Get quote from Polygon API with resilience patterns."""
        async def _make_polygon_request():
            # Rate limiting
            if not await self.polygon_rate_limiter.acquire():
                raise Exception(f"Polygon rate limit exceeded for {symbol}")
            
            url = f"{self.polygon_config['base_url']}/v2/last/trade/{symbol}"
            params = {'apikey': self.polygon_config['api_key']}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', {})
                    
                    return MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(results.get('t', 0) / 1000),
                        open=results.get('p', 0.0),
                        high=results.get('p', 0.0),
                        low=results.get('p', 0.0),
                        close=results.get('p', 0.0),
                        volume=results.get('s', 0),
                        timeframe='quote',
                        data_source='polygon'
                    )
                elif response.status == 429:
                    raise Exception(f"Polygon rate limit hit for {symbol}")
                elif response.status >= 500:
                    raise Exception(f"Polygon server error {response.status} for {symbol}")
                else:
                    logger.warning(f"Polygon API error {response.status} for {symbol}")
                    return None
        
        try:
            # Execute with full resilience stack: bulkhead -> circuit breaker -> retry
            return await self.polygon_bulkhead.execute(
                lambda: self.polygon_circuit_breaker.call(
                    lambda: self.polygon_retry.execute(_make_polygon_request)
                )
            )
        except Exception as e:
            logger.error(f"Polygon quote error for {symbol} after resilience patterns: {e}")
            return None
    
    async def _get_alpha_vantage_quote(self, symbol: str) -> Optional[MarketData]:
        """Get quote from Alpha Vantage API with resilience patterns."""
        async def _make_alpha_vantage_request():
            # Rate limiting (very conservative for free tier)
            if not await self.alpha_vantage_rate_limiter.acquire():
                raise Exception(f"Alpha Vantage rate limit exceeded for {symbol}")
            
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_config['api_key']
            }
            
            async with self.session.get(self.alpha_vantage_config['base_url'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    quote = data.get('Global Quote', {})
                    
                    # Alpha Vantage returns error messages in response body
                    if 'Error Message' in data:
                        raise Exception(f"Alpha Vantage API error: {data['Error Message']}")
                    if 'Note' in data:
                        raise Exception(f"Alpha Vantage rate limit: {data['Note']}")
                    
                    if quote:
                        return MarketData(
                            symbol=symbol,
                            timestamp=datetime.utcnow(),  # Alpha Vantage doesn't provide exact timestamp
                            open=float(quote.get('02. open', 0.0)),
                            high=float(quote.get('03. high', 0.0)),
                            low=float(quote.get('04. low', 0.0)),
                            close=float(quote.get('05. price', 0.0)),
                            volume=int(quote.get('06. volume', 0)),
                            timeframe='quote',
                            data_source='alpha_vantage'
                        )
                    else:
                        logger.warning(f"Alpha Vantage: No quote data for {symbol}")
                        return None
                elif response.status == 429:
                    raise Exception(f"Alpha Vantage rate limit hit for {symbol}")
                elif response.status >= 500:
                    raise Exception(f"Alpha Vantage server error {response.status} for {symbol}")
                else:
                    logger.warning(f"Alpha Vantage API error {response.status} for {symbol}")
                    return None
        
        try:
            # Execute with full resilience stack: bulkhead -> circuit breaker -> retry
            return await self.alpha_vantage_bulkhead.execute(
                lambda: self.alpha_vantage_circuit_breaker.call(
                    lambda: self.alpha_vantage_retry.execute(_make_alpha_vantage_request)
                )
            )
        except Exception as e:
            logger.error(f"Alpha Vantage quote error for {symbol} after resilience patterns: {e}")
            return None
    
    async def _get_alpaca_historical(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime, 
        limit: int
    ) -> List[MarketData]:
        """Get historical data from Alpaca."""
        try:
            url = f"{self.alpaca_config['data_url']}/v2/stocks/{symbol}/bars"
            headers = {
                'APCA-API-KEY-ID': self.alpaca_config['api_key'],
                'APCA-API-SECRET-KEY': self.alpaca_config['secret_key']
            }
            params = {
                'timeframe': timeframe,
                'start': start.isoformat(),
                'end': end.isoformat(),
                'limit': limit,
                'adjustment': 'raw'
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    bars = data.get('bars', [])
                    
                    return [
                        MarketData(
                            symbol=symbol,
                            timestamp=datetime.fromisoformat(bar['t'].replace('Z', '+00:00')),
                            open=bar['o'],
                            high=bar['h'],
                            low=bar['l'],
                            close=bar['c'],
                            volume=bar['v'],
                            timeframe=timeframe,
                            data_source='alpaca'
                        ) for bar in bars
                    ]
                    
        except Exception as e:
            logger.error(f"Alpaca historical data error for {symbol}: {e}")
        
        return []
    
    async def _get_polygon_historical(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime, 
        limit: int
    ) -> List[MarketData]:
        """Get historical data from Polygon."""
        # Polygon has different endpoint structure - simplified implementation
        return []  # Would implement full Polygon historical API
    
    async def _publish_market_data(self, market_data: MarketData):
        """Publish market data to message stream."""
        if self.producer:
            message = json.dumps(asdict(market_data), default=str)
            self.producer.send(message.encode('utf-8'))
    
    async def get_service_health(self) -> Dict:
        """Get service health status including resilience patterns."""
        health = {
            'service': 'market_data',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'data_sources': {
                'alpaca': bool(self.alpaca_config['api_key']),
                'polygon': bool(self.polygon_config['api_key']),
                'alpha_vantage': bool(self.alpha_vantage_config['api_key'])
            },
            'connections': {
                'http_session': self.session is not None,
                'cache': self.cache is not None,
                'message_producer': self.producer is not None
            },
            'resilience_patterns': {
                'circuit_breakers': {
                    'alpaca': self.alpaca_circuit_breaker.get_state(),
                    'polygon': self.polygon_circuit_breaker.get_state(),
                    'alpha_vantage': self.alpha_vantage_circuit_breaker.get_state()
                },
                'bulkhead_status': {
                    'alpaca': self.alpaca_bulkhead.get_status(),
                    'polygon': self.polygon_bulkhead.get_status(), 
                    'alpha_vantage': self.alpha_vantage_bulkhead.get_status()
                }
            }
        }
        
        # Check if any circuit breakers are open (degraded service)
        open_breakers = [
            name for name, state in health['resilience_patterns']['circuit_breakers'].items()
            if state['state'] == 'open'
        ]
        if open_breakers:
            health['status'] = 'degraded'
            health['degraded_reason'] = f"Circuit breakers open: {', '.join(open_breakers)}"
        
        # Test data source connectivity (if circuit breakers allow)
        if self.session and self.alpaca_config['api_key']:
            try:
                test_symbol = 'SPY'
                quote = await self._get_alpaca_quote(test_symbol)
                health['data_sources']['alpaca_test'] = quote is not None
            except Exception as e:
                health['data_sources']['alpaca_test'] = False
                health['data_sources']['alpaca_test_error'] = str(e)
        
        return health


# Global service instance
market_data_service: Optional[MarketDataService] = None




async def get_market_data_service() -> MarketDataService:
    """Get or create market data service instance."""
    global market_data_service
    if market_data_service is None:
        market_data_service = MarketDataService()
        await market_data_service.start()
    return market_data_service
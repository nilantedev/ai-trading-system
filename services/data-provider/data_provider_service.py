#!/usr/bin/env python3
"""Data Provider Service - Multi-source market data integration."""

import asyncio
import json
import logging
import aiohttp
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer

logger = get_logger(__name__)
settings = get_settings()


class DataProvider(Enum):
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IEX_CLOUD = "iex_cloud"
    YAHOO_FINANCE = "yahoo_finance"
    FINNHUB = "finnhub"


class DataType(Enum):
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"
    OPTIONS = "options"


@dataclass
class DataRequest:
    """Data request configuration."""
    request_id: str
    provider: DataProvider
    data_type: DataType
    symbols: List[str]
    timeframe: str = "1min"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    parameters: Dict[str, Any] = None


@dataclass
class DataResponse:
    """Data provider response."""
    request_id: str
    provider: DataProvider
    data_type: DataType
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


@dataclass
class ProviderStatus:
    """Data provider status tracking."""
    provider: DataProvider
    is_available: bool
    last_success: Optional[datetime]
    last_error: Optional[datetime]
    error_count: int
    request_count: int
    rate_limit_remaining: int
    rate_limit_reset: Optional[datetime]


class DataProviderAPI(ABC):
    """Abstract base class for data provider APIs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        
        # Rate limiting
        self.rate_limit = asyncio.Semaphore(config.get('rate_limit', 5))
        self.min_request_interval = config.get('min_request_interval', 0.2)
        self.last_request_time = 0
        
        # Status tracking
        self.request_count = 0
        self.error_count = 0
        self.last_success = None
        self.last_error = None
    
    async def initialize(self):
        """Initialize the data provider connection."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Trading-System/1.0'}
        )
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Apply rate limiting."""
        async with self.rate_limit:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            self.last_request_time = time.time()
    
    @abstractmethod
    async def get_real_time_data(self, symbols: List[str]) -> List[MarketData]:
        """Get real-time market data."""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical market data."""
        pass
    
    @abstractmethod
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamental information."""
        pass
    
    async def get_status(self) -> ProviderStatus:
        """Get provider status."""
        return ProviderStatus(
            provider=DataProvider(self.config.get('type')),
            is_available=self.session is not None,
            last_success=self.last_success,
            last_error=self.last_error,
            error_count=self.error_count,
            request_count=self.request_count,
            rate_limit_remaining=self.rate_limit._value,
            rate_limit_reset=None
        )


class AlphaVantageAPI(DataProviderAPI):
    """Alpha Vantage data provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        config['base_url'] = 'https://www.alphavantage.co/query'
        config['rate_limit'] = 5  # 5 requests per minute
        config['min_request_interval'] = 12.0  # 12 seconds between requests
        super().__init__(config)
    
    async def get_real_time_data(self, symbols: List[str]) -> List[MarketData]:
        """Get real-time data from Alpha Vantage."""
        market_data_list = []
        
        for symbol in symbols:
            await self._rate_limit()
            
            try:
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                async with self.session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'Global Quote' in data:
                            quote = data['Global Quote']
                            
                            market_data = MarketData(
                                symbol=symbol,
                                timestamp=datetime.utcnow(),
                                open=float(quote['02. open']),
                                high=float(quote['03. high']),
                                low=float(quote['04. low']),
                                close=float(quote['05. price']),
                                volume=int(quote['06. volume']),
                                timeframe='1day',
                                data_source='alpha_vantage'
                            )
                            
                            market_data_list.append(market_data)
                            self.last_success = datetime.utcnow()
                        else:
                            logger.warning(f"No quote data for {symbol} from Alpha Vantage")
                    else:
                        self.error_count += 1
                        self.last_error = datetime.utcnow()
                        logger.error(f"Alpha Vantage API error for {symbol}: {response.status}")
                
                self.request_count += 1
                
            except Exception as e:
                self.error_count += 1
                self.last_error = datetime.utcnow()
                logger.error(f"Failed to get Alpha Vantage data for {symbol}: {e}")
        
        return market_data_list
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical data from Alpha Vantage."""
        await self._rate_limit()
        
        try:
            # Map timeframe to Alpha Vantage function
            function_map = {
                '1min': 'TIME_SERIES_INTRADAY',
                '5min': 'TIME_SERIES_INTRADAY',
                '15min': 'TIME_SERIES_INTRADAY',
                '30min': 'TIME_SERIES_INTRADAY',
                '60min': 'TIME_SERIES_INTRADAY',
                '1day': 'TIME_SERIES_DAILY'
            }
            
            function = function_map.get(timeframe, 'TIME_SERIES_DAILY')
            
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            if function == 'TIME_SERIES_INTRADAY':
                params['interval'] = timeframe
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Find the time series key
                    time_series_key = None
                    for key in data.keys():
                        if 'Time Series' in key:
                            time_series_key = key
                            break
                    
                    if time_series_key and time_series_key in data:
                        time_series = data[time_series_key]
                        market_data_list = []
                        
                        for timestamp_str, values in time_series.items():
                            timestamp = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                            
                            # Filter by date range
                            if start_date <= timestamp <= end_date:
                                market_data = MarketData(
                                    symbol=symbol,
                                    timestamp=timestamp,
                                    open=float(values['1. open']),
                                    high=float(values['2. high']),
                                    low=float(values['3. low']),
                                    close=float(values['4. close']),
                                    volume=int(values['5. volume']),
                                    timeframe=timeframe,
                                    data_source='alpha_vantage'
                                )
                                
                                market_data_list.append(market_data)
                        
                        # Sort by timestamp
                        market_data_list.sort(key=lambda x: x.timestamp)
                        
                        self.last_success = datetime.utcnow()
                        self.request_count += 1
                        
                        return market_data_list
                    else:
                        logger.warning(f"No time series data for {symbol} from Alpha Vantage")
                else:
                    self.error_count += 1
                    self.last_error = datetime.utcnow()
                    logger.error(f"Alpha Vantage historical data error for {symbol}: {response.status}")
            
        except Exception as e:
            self.error_count += 1
            self.last_error = datetime.utcnow()
            logger.error(f"Failed to get Alpha Vantage historical data for {symbol}: {e}")
        
        return []
    
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company overview from Alpha Vantage."""
        await self._rate_limit()
        
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data and 'Symbol' in data:
                        self.last_success = datetime.utcnow()
                        self.request_count += 1
                        return data
                    else:
                        logger.warning(f"No company info for {symbol} from Alpha Vantage")
                else:
                    self.error_count += 1
                    self.last_error = datetime.utcnow()
                    logger.error(f"Alpha Vantage company info error for {symbol}: {response.status}")
            
        except Exception as e:
            self.error_count += 1
            self.last_error = datetime.utcnow()
            logger.error(f"Failed to get Alpha Vantage company info for {symbol}: {e}")
        
        return {}


class PolygonAPI(DataProviderAPI):
    """Polygon.io data provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        config['base_url'] = 'https://api.polygon.io'
        config['rate_limit'] = 100  # Higher rate limit
        config['min_request_interval'] = 0.1
        super().__init__(config)
    
    async def get_real_time_data(self, symbols: List[str]) -> List[MarketData]:
        """Get real-time data from Polygon."""
        market_data_list = []
        
        for symbol in symbols:
            await self._rate_limit()
            
            try:
                # Get previous close for real-time comparison
                url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
                params = {'apikey': self.api_key}
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('resultsCount', 0) > 0:
                            result = data['results'][0]
                            
                            market_data = MarketData(
                                symbol=symbol,
                                timestamp=datetime.fromtimestamp(result['t'] / 1000),
                                open=result['o'],
                                high=result['h'],
                                low=result['l'],
                                close=result['c'],
                                volume=result['v'],
                                timeframe='1day',
                                data_source='polygon'
                            )
                            
                            market_data_list.append(market_data)
                            self.last_success = datetime.utcnow()
                        else:
                            logger.warning(f"No data for {symbol} from Polygon")
                    else:
                        self.error_count += 1
                        self.last_error = datetime.utcnow()
                        logger.error(f"Polygon API error for {symbol}: {response.status}")
                
                self.request_count += 1
                
            except Exception as e:
                self.error_count += 1
                self.last_error = datetime.utcnow()
                logger.error(f"Failed to get Polygon data for {symbol}: {e}")
        
        return market_data_list
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical data from Polygon."""
        await self._rate_limit()
        
        try:
            # Map timeframe to Polygon timespan
            timespan_map = {
                '1min': 'minute',
                '5min': 'minute',
                '15min': 'minute',
                '30min': 'minute',
                '60min': 'hour',
                '1day': 'day'
            }
            
            multiplier_map = {
                '1min': 1,
                '5min': 5,
                '15min': 15,
                '30min': 30,
                '60min': 1,
                '1day': 1
            }
            
            timespan = timespan_map.get(timeframe, 'day')
            multiplier = multiplier_map.get(timeframe, 1)
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
            params = {
                'apikey': self.api_key,
                'limit': 50000
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('resultsCount', 0) > 0:
                        market_data_list = []
                        
                        for result in data['results']:
                            timestamp = datetime.fromtimestamp(result['t'] / 1000)
                            
                            market_data = MarketData(
                                symbol=symbol,
                                timestamp=timestamp,
                                open=result['o'],
                                high=result['h'],
                                low=result['l'],
                                close=result['c'],
                                volume=result['v'],
                                timeframe=timeframe,
                                data_source='polygon'
                            )
                            
                            market_data_list.append(market_data)
                        
                        self.last_success = datetime.utcnow()
                        self.request_count += 1
                        
                        return market_data_list
                    else:
                        logger.warning(f"No historical data for {symbol} from Polygon")
                else:
                    self.error_count += 1
                    self.last_error = datetime.utcnow()
                    logger.error(f"Polygon historical data error for {symbol}: {response.status}")
            
        except Exception as e:
            self.error_count += 1
            self.last_error = datetime.utcnow()
            logger.error(f"Failed to get Polygon historical data for {symbol}: {e}")
        
        return []
    
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company details from Polygon."""
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/v3/reference/tickers/{symbol}"
            params = {'apikey': self.api_key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('results'):
                        self.last_success = datetime.utcnow()
                        self.request_count += 1
                        return data['results']
                    else:
                        logger.warning(f"No company info for {symbol} from Polygon")
                else:
                    self.error_count += 1
                    self.last_error = datetime.utcnow()
                    logger.error(f"Polygon company info error for {symbol}: {response.status}")
            
        except Exception as e:
            self.error_count += 1
            self.last_error = datetime.utcnow()
            logger.error(f"Failed to get Polygon company info for {symbol}: {e}")
        
        return {}


class YahooFinanceAPI(DataProviderAPI):
    """Yahoo Finance data provider implementation (free tier)."""
    
    def __init__(self, config: Dict[str, Any]):
        config['base_url'] = 'https://query1.finance.yahoo.com/v8/finance/chart'
        config['rate_limit'] = 20  # Conservative rate limit
        config['min_request_interval'] = 0.5
        super().__init__(config)
    
    async def get_real_time_data(self, symbols: List[str]) -> List[MarketData]:
        """Get real-time data from Yahoo Finance."""
        market_data_list = []
        
        # Yahoo Finance allows multiple symbols in one request
        symbol_string = ','.join(symbols)
        
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/{symbol_string}"
            params = {
                'interval': '1d',
                'range': '1d'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('chart') and data['chart'].get('result'):
                        for result in data['chart']['result']:
                            symbol = result['meta']['symbol']
                            
                            if result.get('indicators') and result['indicators'].get('quote'):
                                quote = result['indicators']['quote'][0]
                                
                                if quote.get('close') and len(quote['close']) > 0:
                                    # Get the latest data point
                                    timestamps = result['timestamp']
                                    latest_idx = -1
                                    
                                    market_data = MarketData(
                                        symbol=symbol,
                                        timestamp=datetime.fromtimestamp(timestamps[latest_idx]),
                                        open=quote['open'][latest_idx] or 0.0,
                                        high=quote['high'][latest_idx] or 0.0,
                                        low=quote['low'][latest_idx] or 0.0,
                                        close=quote['close'][latest_idx] or 0.0,
                                        volume=quote['volume'][latest_idx] or 0,
                                        timeframe='1day',
                                        data_source='yahoo_finance'
                                    )
                                    
                                    market_data_list.append(market_data)
                        
                        self.last_success = datetime.utcnow()
                    else:
                        logger.warning(f"No chart data from Yahoo Finance")
                else:
                    self.error_count += 1
                    self.last_error = datetime.utcnow()
                    logger.error(f"Yahoo Finance API error: {response.status}")
            
            self.request_count += 1
            
        except Exception as e:
            self.error_count += 1
            self.last_error = datetime.utcnow()
            logger.error(f"Failed to get Yahoo Finance data: {e}")
        
        return market_data_list
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical data from Yahoo Finance."""
        await self._rate_limit()
        
        try:
            # Map timeframe to Yahoo Finance interval
            interval_map = {
                '1min': '1m',
                '2min': '2m',
                '5min': '5m',
                '15min': '15m',
                '30min': '30m',
                '60min': '1h',
                '90min': '90m',
                '1day': '1d',
                '5day': '5d',
                '1week': '1wk',
                '1month': '1mo',
                '3month': '3mo'
            }
            
            interval = interval_map.get(timeframe, '1d')
            period1 = int(start_date.timestamp())
            period2 = int(end_date.timestamp())
            
            url = f"{self.base_url}/{symbol}"
            params = {
                'period1': period1,
                'period2': period2,
                'interval': interval
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('chart') and data['chart'].get('result'):
                        result = data['chart']['result'][0]
                        
                        if result.get('indicators') and result['indicators'].get('quote'):
                            timestamps = result['timestamp']
                            quote = result['indicators']['quote'][0]
                            
                            market_data_list = []
                            
                            for i, timestamp in enumerate(timestamps):
                                if (quote['open'][i] is not None and 
                                    quote['close'][i] is not None):
                                    
                                    market_data = MarketData(
                                        symbol=symbol,
                                        timestamp=datetime.fromtimestamp(timestamp),
                                        open=quote['open'][i],
                                        high=quote['high'][i],
                                        low=quote['low'][i],
                                        close=quote['close'][i],
                                        volume=quote['volume'][i] or 0,
                                        timeframe=timeframe,
                                        data_source='yahoo_finance'
                                    )
                                    
                                    market_data_list.append(market_data)
                            
                            self.last_success = datetime.utcnow()
                            self.request_count += 1
                            
                            return market_data_list
                        else:
                            logger.warning(f"No quote data for {symbol} from Yahoo Finance")
                    else:
                        logger.warning(f"No chart data for {symbol} from Yahoo Finance")
                else:
                    self.error_count += 1
                    self.last_error = datetime.utcnow()
                    logger.error(f"Yahoo Finance historical data error for {symbol}: {response.status}")
            
        except Exception as e:
            self.error_count += 1
            self.last_error = datetime.utcnow()
            logger.error(f"Failed to get Yahoo Finance historical data for {symbol}: {e}")
        
        return []
    
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company info from Yahoo Finance (limited free data)."""
        # Yahoo Finance doesn't provide comprehensive company data via free API
        # This would need to be implemented with web scraping or paid API
        return {
            'symbol': symbol,
            'name': f"Company {symbol}",
            'exchange': 'Unknown',
            'source': 'yahoo_finance_limited'
        }


class DataProviderService:
    """Service for managing multiple data providers."""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.cache = None
        
        self.providers: Dict[str, DataProviderAPI] = {}
        self.provider_priorities: List[str] = []
        self.is_running = False
        
        # Request tracking
        self.request_queue = asyncio.Queue(maxsize=10000)
        self.active_requests = {}
        self.request_counter = 1
        
        # Performance metrics
        self.requests_processed = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Symbols being tracked
        self.tracked_symbols: Set[str] = set()
        
    async def start(self):
        """Initialize and start data provider service."""
        logger.info("Starting Data Provider Service")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            self.cache = get_trading_cache()
            
            # Initialize data providers
            await self._initialize_providers()
            
            # Subscribe to data requests
            await self._setup_subscriptions()
            
            # Start processing tasks
            self.is_running = True
            
            tasks = [
                asyncio.create_task(self._process_request_queue()),
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._periodic_data_collection()),
                asyncio.create_task(self._health_monitoring()),
                asyncio.create_task(self._cache_maintenance())
            ]
            
            logger.info("Data provider service started with 5 concurrent tasks")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start data provider service: {e}")
            raise
    
    async def stop(self):
        """Stop data provider service gracefully."""
        logger.info("Stopping Data Provider Service")
        self.is_running = False
        
        # Clean up provider connections
        for provider in self.providers.values():
            await provider.cleanup()
        
        if self.consumer:
            await self.consumer.close()
        if self.producer:
            await self.producer.close()
        
        logger.info("Data Provider Service stopped")
    
    async def _initialize_providers(self):
        """Initialize configured data providers."""
        try:
            provider_configs = settings.get('data_providers', {})
            
            for provider_name, config in provider_configs.items():
                provider_type = DataProvider(config.get('type'))
                
                if provider_type == DataProvider.ALPHA_VANTAGE:
                    provider = AlphaVantageAPI(config)
                elif provider_type == DataProvider.POLYGON:
                    provider = PolygonAPI(config)
                elif provider_type == DataProvider.YAHOO_FINANCE:
                    provider = YahooFinanceAPI(config)
                else:
                    logger.warning(f"Unsupported data provider: {provider_type}")
                    continue
                
                await provider.initialize()
                self.providers[provider_name] = provider
                self.provider_priorities.append(provider_name)
                
                logger.info(f"Initialized {provider_type.value} provider: {provider_name}")
            
            if not self.providers:
                # Initialize default Yahoo Finance provider (free)
                yahoo_provider = YahooFinanceAPI({'type': 'yahoo_finance'})
                await yahoo_provider.initialize()
                self.providers['yahoo'] = yahoo_provider
                self.provider_priorities.append('yahoo')
                logger.info("Initialized default Yahoo Finance provider")
                
        except Exception as e:
            logger.error(f"Failed to initialize data providers: {e}")
            raise
    
    async def _setup_subscriptions(self):
        """Subscribe to data requests."""
        try:
            await self.consumer.subscribe_data_requests(
                self._handle_data_request,
                subscription_name="data-provider-requests"
            )
            
            logger.info("Subscribed to data request streams")
        except Exception as e:
            logger.warning(f"Subscription setup failed: {e}")
    
    async def _consume_messages(self):
        """Consume messages from subscribed topics."""
        try:
            await self.consumer.start_consuming()
        except Exception as e:
            logger.error(f"Message consumption failed: {e}")
    
    async def _handle_data_request(self, message):
        """Handle incoming data request."""
        try:
            request_data = json.loads(message) if isinstance(message, str) else message
            
            data_request = DataRequest(
                request_id=request_data.get('request_id', f"req_{self.request_counter}"),
                provider=DataProvider(request_data.get('provider', 'yahoo_finance')),
                data_type=DataType(request_data.get('data_type', 'real_time')),
                symbols=request_data.get('symbols', []),
                timeframe=request_data.get('timeframe', '1min'),
                start_date=datetime.fromisoformat(request_data['start_date']) if request_data.get('start_date') else None,
                end_date=datetime.fromisoformat(request_data['end_date']) if request_data.get('end_date') else None,
                parameters=request_data.get('parameters', {})
            )
            
            self.request_counter += 1
            
            # Add to processing queue
            await self.request_queue.put(data_request)
            
        except Exception as e:
            logger.error(f"Failed to handle data request: {e}")
    
    async def _process_request_queue(self):
        """Process data requests."""
        while self.is_running:
            try:
                # Wait for data request
                data_request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                # Process the request
                response = await self._execute_data_request(data_request)
                
                # Cache and publish response
                if response:
                    await self._cache_data_response(response)
                    await self._publish_data_response(response)
                
                self.requests_processed += 1
                self.request_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Request processing error: {e}")
                self.failed_requests += 1
    
    async def _execute_data_request(self, request: DataRequest) -> Optional[DataResponse]:
        """Execute a data request with provider fallback."""
        
        # Try providers in priority order
        for provider_name in self.provider_priorities:
            provider = self.providers.get(provider_name)
            
            if not provider:
                continue
            
            try:
                if request.data_type == DataType.REAL_TIME:
                    data_list = await provider.get_real_time_data(request.symbols)
                    
                    # Process each symbol's data
                    for data in data_list:
                        response = DataResponse(
                            request_id=request.request_id,
                            provider=DataProvider(provider.config.get('type')),
                            data_type=request.data_type,
                            symbol=data.symbol,
                            timestamp=datetime.utcnow(),
                            data=asdict(data),
                            success=True
                        )
                        
                        # Publish individual market data
                        await self._publish_market_data(data)
                        
                        self.successful_requests += 1
                        return response
                
                elif request.data_type == DataType.HISTORICAL:
                    for symbol in request.symbols:
                        data_list = await provider.get_historical_data(
                            symbol, 
                            request.timeframe,
                            request.start_date,
                            request.end_date
                        )
                        
                        if data_list:
                            response = DataResponse(
                                request_id=request.request_id,
                                provider=DataProvider(provider.config.get('type')),
                                data_type=request.data_type,
                                symbol=symbol,
                                timestamp=datetime.utcnow(),
                                data={'historical_data': [asdict(d) for d in data_list]},
                                success=True
                            )
                            
                            # Publish historical data
                            for data in data_list:
                                await self._publish_market_data(data)
                            
                            self.successful_requests += 1
                            return response
                
                elif request.data_type == DataType.FUNDAMENTALS:
                    for symbol in request.symbols:
                        company_info = await provider.get_company_info(symbol)
                        
                        if company_info:
                            response = DataResponse(
                                request_id=request.request_id,
                                provider=DataProvider(provider.config.get('type')),
                                data_type=request.data_type,
                                symbol=symbol,
                                timestamp=datetime.utcnow(),
                                data=company_info,
                                success=True
                            )
                            
                            self.successful_requests += 1
                            return response
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for request {request.request_id}: {e}")
                continue
        
        # All providers failed
        self.failed_requests += 1
        return DataResponse(
            request_id=request.request_id,
            provider=DataProvider.YAHOO_FINANCE,  # Default
            data_type=request.data_type,
            symbol=request.symbols[0] if request.symbols else 'UNKNOWN',
            timestamp=datetime.utcnow(),
            data={},
            success=False,
            error_message="All data providers failed"
        )
    
    async def _periodic_data_collection(self):
        """Collect data periodically for tracked symbols."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                if self.tracked_symbols:
                    # Create real-time data request
                    request = DataRequest(
                        request_id=f"periodic_{datetime.utcnow().timestamp()}",
                        provider=DataProvider.YAHOO_FINANCE,
                        data_type=DataType.REAL_TIME,
                        symbols=list(self.tracked_symbols)
                    )
                    
                    await self.request_queue.put(request)
                
            except Exception as e:
                logger.warning(f"Periodic data collection error: {e}")
    
    async def _health_monitoring(self):
        """Monitor provider health and availability."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for provider_name, provider in self.providers.items():
                    status = await provider.get_status()
                    
                    if status.error_count > 10:  # High error rate
                        logger.warning(f"Provider {provider_name} has high error rate: {status.error_count}")
                    
                    if status.last_error and status.last_success:
                        if status.last_error > status.last_success:
                            logger.warning(f"Provider {provider_name} last request failed")
                
            except Exception as e:
                logger.warning(f"Health monitoring error: {e}")
    
    async def _cache_maintenance(self):
        """Maintain data cache and cleanup old entries."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Clean up hourly
                
                if self.cache:
                    # Would implement cache cleanup logic
                    logger.debug("Performed cache maintenance")
                
            except Exception as e:
                logger.warning(f"Cache maintenance error: {e}")
    
    async def _cache_data_response(self, response: DataResponse):
        """Cache data response."""
        try:
            if self.cache:
                cache_key = f"data_response:{response.request_id}"
                response_data = asdict(response)
                response_data['timestamp'] = response.timestamp.isoformat()
                
                await self.cache.set_json(cache_key, response_data, ttl=3600)  # 1 hour
        except Exception as e:
            logger.warning(f"Failed to cache data response: {e}")
    
    async def _publish_data_response(self, response: DataResponse):
        """Publish data response."""
        try:
            if self.producer:
                response_message = {
                    'request_id': response.request_id,
                    'provider': response.provider.value,
                    'data_type': response.data_type.value,
                    'symbol': response.symbol,
                    'success': response.success,
                    'timestamp': response.timestamp.isoformat()
                }
                
                # Would publish to data response topic
                logger.debug(f"Publishing data response: {response.request_id}")
                
        except Exception as e:
            logger.warning(f"Failed to publish data response: {e}")
    
    async def _publish_market_data(self, market_data: MarketData):
        """Publish market data to stream."""
        try:
            if self.producer:
                market_data_dict = asdict(market_data)
                market_data_dict['timestamp'] = market_data.timestamp.isoformat()
                
                # Would publish to market data topic
                logger.debug(f"Publishing market data: {market_data.symbol}")
                
        except Exception as e:
            logger.warning(f"Failed to publish market data: {e}")
    
    async def add_tracked_symbol(self, symbol: str):
        """Add symbol to tracking list."""
        self.tracked_symbols.add(symbol.upper())
        logger.info(f"Added {symbol} to tracked symbols")
    
    async def remove_tracked_symbol(self, symbol: str):
        """Remove symbol from tracking list."""
        self.tracked_symbols.discard(symbol.upper())
        logger.info(f"Removed {symbol} from tracked symbols")
    
    async def get_provider_status(self) -> Dict[str, ProviderStatus]:
        """Get status of all providers."""
        statuses = {}
        
        for provider_name, provider in self.providers.items():
            status = await provider.get_status()
            statuses[provider_name] = status
        
        return statuses
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        provider_health = {}
        
        for provider_name, provider in self.providers.items():
            status = await provider.get_status()
            provider_health[provider_name] = {
                'available': status.is_available,
                'error_count': status.error_count,
                'request_count': status.request_count,
                'last_success': status.last_success.isoformat() if status.last_success else None
            }
        
        return {
            'service': 'data_provider_service',
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'requests_processed': self.requests_processed,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.successful_requests / max(self.requests_processed, 1) * 100,
                'tracked_symbols': len(self.tracked_symbols)
            },
            'providers': provider_health,
            'connections': {
                'consumer': self.consumer is not None,
                'producer': self.producer is not None,
                'cache': self.cache is not None
            }
        }


# Global service instance
data_provider_service: Optional[DataProviderService] = None


async def get_data_provider_service() -> DataProviderService:
    """Get or create data provider service instance."""
    global data_provider_service
    if data_provider_service is None:
        data_provider_service = DataProviderService()
    return data_provider_service
#!/usr/bin/env python3
"""Reference Data Service - Static and reference data management."""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import os

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.database import get_redis_client

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class SecurityInfo:
    """Security reference information."""
    symbol: str
    name: str
    exchange: str
    sector: str = ""
    industry: str = ""
    market_cap: Optional[float] = None
    currency: str = "USD"
    country: str = "US"
    isin: Optional[str] = None
    cusip: Optional[str] = None
    active: bool = True
    last_updated: datetime = datetime.utcnow()


@dataclass
class ExchangeInfo:
    """Exchange information."""
    code: str
    name: str
    timezone: str
    currency: str
    country: str
    open_time: str
    close_time: str
    active: bool = True


@dataclass
class EconomicEvent:
    """Economic calendar event."""
    event_id: str
    name: str
    country: str
    currency: str
    impact: str  # 'low', 'medium', 'high'
    scheduled_time: datetime
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None
    unit: str = ""
    source: str = ""


class ReferenceDataService:
    """Manages static reference data and economic calendar."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = None
        self.redis_client = None
        
        # API configurations
        self.polygon_config = {
            'api_key': os.getenv('POLYGON_API_KEY'),
            'base_url': 'https://api.polygon.io'
        }
        
        self.finnhub_config = {
            'api_key': os.getenv('FINNHUB_API_KEY'),
            'base_url': 'https://finnhub.io/api/v1'
        }
        
        self.fred_config = {
            'api_key': os.getenv('FRED_API_KEY'),
            'base_url': 'https://api.stlouisfed.org/fred'
        }
        
        # Default watchlist symbols
        self.default_symbols = [
            'SPY', 'QQQ', 'IWM', 'DIA',  # Major ETFs
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',  # Tech giants
            'JPM', 'BAC', 'GS', 'WFC',  # Banking
            'XOM', 'CVX', 'COP',  # Energy
            'JNJ', 'PFE', 'UNH',  # Healthcare
        ]
    
    async def start(self):
        """Initialize service connections."""
        logger.info("Starting Reference Data Service")
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Trading-System/1.0'}
        )
        
        # Initialize cache and Redis
        self.cache = get_trading_cache()
        self.redis_client = get_redis_client()
        
        # Initialize reference data
        await self._initialize_reference_data()
    
    async def stop(self):
        """Cleanup service connections."""
        if self.session:
            await self.session.close()
        logger.info("Reference Data Service stopped")
    
    async def get_security_info(self, symbol: str, refresh: bool = False) -> Optional[SecurityInfo]:
        """Get security information."""
        cache_key = f"security_info:{symbol}"
        
        # Try cache first unless refresh requested
        if not refresh and self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    return SecurityInfo(**data)
            except Exception as e:
                logger.warning(f"Cache error for {symbol}: {e}")
        
        # Fetch from API
        security_info = await self._fetch_security_info(symbol)
        
        # Cache the result
        if security_info and self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    24 * 3600,  # 24 hours
                    json.dumps(asdict(security_info), default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to cache security info for {symbol}: {e}")
        
        return security_info
    
    async def get_exchange_info(self, exchange_code: str) -> Optional[ExchangeInfo]:
        """Get exchange information."""
        cache_key = f"exchange_info:{exchange_code}"
        
        # Try cache first
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    return ExchangeInfo(**data)
            except Exception as e:
                logger.warning(f"Cache error for exchange {exchange_code}: {e}")
        
        # Return built-in exchange data
        exchange_info = self._get_builtin_exchange_info(exchange_code)
        
        # Cache the result
        if exchange_info and self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    7 * 24 * 3600,  # 7 days
                    json.dumps(asdict(exchange_info), default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to cache exchange info: {e}")
        
        return exchange_info
    
    async def get_watchlist_symbols(self) -> List[str]:
        """Get list of symbols to monitor."""
        # Try to get custom watchlist from Redis
        if self.redis_client:
            try:
                custom_symbols = await self.redis_client.smembers("watchlist:symbols")
                if custom_symbols:
                    return list(custom_symbols)
            except Exception as e:
                logger.warning(f"Failed to get custom watchlist: {e}")
        
        return self.default_symbols.copy()
    
    async def add_to_watchlist(self, symbols: List[str]) -> bool:
        """Add symbols to watchlist."""
        if self.redis_client:
            try:
                await self.redis_client.sadd("watchlist:symbols", *symbols)
                logger.info(f"Added {len(symbols)} symbols to watchlist")
                return True
            except Exception as e:
                logger.error(f"Failed to add symbols to watchlist: {e}")
        
        return False
    
    async def remove_from_watchlist(self, symbols: List[str]) -> bool:
        """Remove symbols from watchlist."""
        if self.redis_client:
            try:
                await self.redis_client.srem("watchlist:symbols", *symbols)
                logger.info(f"Removed {len(symbols)} symbols from watchlist")
                return True
            except Exception as e:
                logger.error(f"Failed to remove symbols from watchlist: {e}")
        
        return False
    
    async def get_economic_events(
        self, 
        start_date: datetime, 
        end_date: datetime,
        impact_filter: Optional[List[str]] = None
    ) -> List[EconomicEvent]:
        """Get economic calendar events."""
        cache_key = f"economic_events:{start_date.date()}:{end_date.date()}"
        
        # Try cache first
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    events_data = json.loads(cached_data)
                    events = [EconomicEvent(**event) for event in events_data]
                    
                    # Apply impact filter if specified
                    if impact_filter:
                        events = [e for e in events if e.impact in impact_filter]
                    
                    return events
            except Exception as e:
                logger.warning(f"Cache error for economic events: {e}")
        
        # Fetch from APIs
        events = await self._fetch_economic_events(start_date, end_date)
        
        # Cache the result
        if events and self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    6 * 3600,  # 6 hours
                    json.dumps([asdict(event) for event in events], default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to cache economic events: {e}")
        
        # Apply impact filter if specified
        if impact_filter:
            events = [e for e in events if e.impact in impact_filter]
        
        return events
    
    async def _initialize_reference_data(self):
        """Initialize reference data on startup."""
        logger.info("Initializing reference data...")
        
        # Pre-populate security info for default symbols
        tasks = []
        for symbol in self.default_symbols[:5]:  # Limit to avoid rate limits
            tasks.append(self.get_security_info(symbol))
        
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info("Reference data initialization complete")
            except Exception as e:
                logger.warning(f"Reference data initialization partial: {e}")
    
    async def _fetch_security_info(self, symbol: str) -> Optional[SecurityInfo]:
        """Fetch security information from APIs."""
        # Try Polygon first
        if self.polygon_config['api_key']:
            info = await self._fetch_from_polygon(symbol)
            if info:
                return info
        
        # Try Finnhub as fallback
        if self.finnhub_config['api_key']:
            info = await self._fetch_from_finnhub(symbol)
            if info:
                return info
        
        # Return basic info if no API data available
        return SecurityInfo(
            symbol=symbol,
            name=symbol,
            exchange="UNKNOWN",
            sector="Unknown",
            industry="Unknown"
        )
    
    async def _fetch_from_polygon(self, symbol: str) -> Optional[SecurityInfo]:
        """Fetch security info from Polygon API."""
        try:
            url = f"{self.polygon_config['base_url']}/v3/reference/tickers/{symbol}"
            params = {'apikey': self.polygon_config['api_key']}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', {})
                    
                    if results:
                        return SecurityInfo(
                            symbol=symbol,
                            name=results.get('name', symbol),
                            exchange=results.get('primary_exchange', 'UNKNOWN'),
                            sector=results.get('sic_description', ''),
                            industry=results.get('industry', ''),
                            market_cap=results.get('market_cap'),
                            currency=results.get('currency_name', 'USD'),
                            active=results.get('active', True)
                        )
                else:
                    logger.warning(f"Polygon API error for {symbol}: {response.status}")
                    
        except Exception as e:
            logger.error(f"Polygon fetch error for {symbol}: {e}")
        
        return None
    
    async def _fetch_from_finnhub(self, symbol: str) -> Optional[SecurityInfo]:
        """Fetch security info from Finnhub API."""
        try:
            # Get company profile
            url = f"{self.finnhub_config['base_url']}/stock/profile2"
            params = {
                'symbol': symbol,
                'token': self.finnhub_config['api_key']
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data and 'name' in data:
                        return SecurityInfo(
                            symbol=symbol,
                            name=data.get('name', symbol),
                            exchange=data.get('exchange', 'UNKNOWN'),
                            sector=data.get('finnhubIndustry', ''),
                            industry=data.get('gind', ''),
                            market_cap=data.get('marketCapitalization', 0) * 1000000 if data.get('marketCapitalization') else None,
                            currency=data.get('currency', 'USD'),
                            country=data.get('country', 'US')
                        )
                else:
                    logger.warning(f"Finnhub API error for {symbol}: {response.status}")
                    
        except Exception as e:
            logger.error(f"Finnhub fetch error for {symbol}: {e}")
        
        return None
    
    def _get_builtin_exchange_info(self, exchange_code: str) -> Optional[ExchangeInfo]:
        """Get built-in exchange information."""
        exchanges = {
            'NYSE': ExchangeInfo(
                code='NYSE',
                name='New York Stock Exchange',
                timezone='America/New_York',
                currency='USD',
                country='US',
                open_time='09:30',
                close_time='16:00'
            ),
            'NASDAQ': ExchangeInfo(
                code='NASDAQ',
                name='NASDAQ Stock Market',
                timezone='America/New_York',
                currency='USD',
                country='US',
                open_time='09:30',
                close_time='16:00'
            ),
            'LSE': ExchangeInfo(
                code='LSE',
                name='London Stock Exchange',
                timezone='Europe/London',
                currency='GBP',
                country='GB',
                open_time='08:00',
                close_time='16:30'
            )
        }
        
        return exchanges.get(exchange_code.upper())
    
    async def _fetch_economic_events(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[EconomicEvent]:
        """Fetch economic events from APIs."""
        events = []
        
        # Get FRED economic data if available
        if self.fred_config['api_key']:
            fred_events = await self._fetch_from_fred(start_date, end_date)
            events.extend(fred_events)
        
        # Add mock events for development
        events.extend(self._get_mock_economic_events(start_date, end_date))
        
        return events
    
    async def _fetch_from_fred(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[EconomicEvent]:
        """Fetch economic data from FRED API."""
        try:
            # This would implement FRED API calls for economic indicators
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"FRED fetch error: {e}")
        
        return []
    
    def _get_mock_economic_events(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[EconomicEvent]:
        """Generate mock economic events for development."""
        events = []
        current = start_date
        
        while current <= end_date:
            # Add some sample events
            if current.weekday() < 5:  # Weekdays only
                if current.day in [1, 15]:  # Monthly events
                    events.append(EconomicEvent(
                        event_id=f"cpi_{current.strftime('%Y%m%d')}",
                        name="Consumer Price Index",
                        country="US",
                        currency="USD",
                        impact="high",
                        scheduled_time=current.replace(hour=8, minute=30),
                        forecast_value=0.3,
                        source="mock"
                    ))
            
            current += timedelta(days=1)
        
        return events
    
    async def get_service_health(self) -> Dict:
        """Get service health status."""
        return {
            'service': 'reference_data',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'data_sources': {
                'polygon': bool(self.polygon_config['api_key']),
                'finnhub': bool(self.finnhub_config['api_key']),
                'fred': bool(self.fred_config['api_key'])
            },
            'connections': {
                'http_session': self.session is not None,
                'cache': self.cache is not None,
                'redis': self.redis_client is not None
            },
            'watchlist_size': len(await self.get_watchlist_symbols())
        }


# Global service instance
reference_data_service: Optional[ReferenceDataService] = None


async def get_reference_data_service() -> ReferenceDataService:
    """Get or create reference data service instance."""
    global reference_data_service
    if reference_data_service is None:
        reference_data_service = ReferenceDataService()
        await reference_data_service.start()
    return reference_data_service
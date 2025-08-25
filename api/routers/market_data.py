#!/usr/bin/env python3
"""
Market Data API Router - REST endpoints for market data access
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models import (
    MarketDataRequest, MarketDataResponse, MarketDataPoint,
    SymbolsResponse, SymbolInfo, BaseResponse, ErrorResponse,
    PaginationParams, PaginatedResponse
)
from api.main import verify_token, optional_auth, APIException
from trading_common import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/market-data/symbols",
    response_model=SymbolsResponse,
    summary="Get available trading symbols",
    description="Retrieve list of all available trading symbols with metadata"
)
async def get_symbols(
    search: Optional[str] = Query(None, description="Search term for symbol filtering"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    active_only: bool = Query(True, description="Return only actively traded symbols"),
    pagination: PaginationParams = Depends(),
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
):
    """Get available trading symbols with optional filtering."""
    try:
        # Mock symbol data - in production, fetch from reference data service
        mock_symbols = [
            SymbolInfo(
                symbol="AAPL",
                name="Apple Inc.",
                exchange="NASDAQ",
                sector="Technology",
                industry="Consumer Electronics",
                market_cap=3000000000000.0,
                active=True
            ),
            SymbolInfo(
                symbol="GOOGL",
                name="Alphabet Inc.",
                exchange="NASDAQ",
                sector="Technology",
                industry="Internet Services",
                market_cap=1800000000000.0,
                active=True
            ),
            SymbolInfo(
                symbol="TSLA",
                name="Tesla Inc.",
                exchange="NASDAQ",
                sector="Consumer Discretionary",
                industry="Auto Manufacturers",
                market_cap=800000000000.0,
                active=True
            ),
            SymbolInfo(
                symbol="SPY",
                name="SPDR S&P 500 ETF Trust",
                exchange="NYSE",
                sector="ETF",
                industry="Index Fund",
                market_cap=400000000000.0,
                active=True
            ),
            SymbolInfo(
                symbol="QQQ",
                name="Invesco QQQ Trust",
                exchange="NASDAQ",
                sector="ETF",
                industry="Technology ETF",
                market_cap=200000000000.0,
                active=True
            )
        ]
        
        # Apply filters
        filtered_symbols = mock_symbols
        
        if search:
            search_term = search.upper()
            filtered_symbols = [
                s for s in filtered_symbols 
                if search_term in s.symbol or search_term in s.name.upper()
            ]
        
        if exchange:
            filtered_symbols = [s for s in filtered_symbols if s.exchange == exchange.upper()]
        
        if sector:
            filtered_symbols = [s for s in filtered_symbols if s.sector == sector]
        
        if active_only:
            filtered_symbols = [s for s in filtered_symbols if s.active]
        
        # Apply pagination
        start_idx = pagination.offset
        end_idx = start_idx + pagination.size
        paginated_symbols = filtered_symbols[start_idx:end_idx]
        
        return SymbolsResponse(
            symbols=paginated_symbols,
            count=len(paginated_symbols),
            message=f"Retrieved {len(paginated_symbols)} symbols"
        )
        
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to fetch symbols",
            error_code="SYMBOLS_FETCH_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/market-data/{symbol}",
    response_model=MarketDataResponse,
    summary="Get current market data",
    description="Retrieve current/latest market data for a specific symbol"
)
async def get_current_market_data(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL)"),
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
):
    """Get current market data for a symbol."""
    try:
        symbol = symbol.upper()
        
        # Import data provider service
        from services.data_provider.data_provider_service import get_data_provider_service
        
        data_service = await get_data_provider_service()
        
        # Get real-time data
        market_data_list = []
        
        # Try to get data from service
        try:
            # Mock current market data for demonstration
            current_time = datetime.utcnow()
            mock_data = MarketDataPoint(
                symbol=symbol,
                timestamp=current_time,
                open=150.00,
                high=152.50,
                low=149.25,
                close=151.75,
                volume=1500000,
                timeframe="1min",
                data_source="mock"
            )
            market_data_list = [mock_data]
            
        except Exception as e:
            logger.warning(f"Failed to get real data for {symbol}, using mock: {e}")
            # Return mock data as fallback
            current_time = datetime.utcnow()
            mock_data = MarketDataPoint(
                symbol=symbol,
                timestamp=current_time,
                open=100.00,
                high=102.50,
                low=99.25,
                close=101.75,
                volume=1000000,
                timeframe="current",
                data_source="mock"
            )
            market_data_list = [mock_data]
        
        return MarketDataResponse(
            data=market_data_list,
            count=len(market_data_list),
            symbol=symbol,
            message=f"Retrieved current market data for {symbol}"
        )
        
    except Exception as e:
        logger.error(f"Error fetching current market data for {symbol}: {e}")
        raise APIException(
            status_code=500,
            detail=f"Failed to fetch market data for {symbol}",
            error_code="MARKET_DATA_ERROR",
            context={"symbol": symbol, "error": str(e)}
        )


@router.get(
    "/market-data/{symbol}/history",
    response_model=MarketDataResponse,
    summary="Get historical market data",
    description="Retrieve historical market data for a specific symbol with optional date range and timeframe"
)
async def get_historical_market_data(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL)"),
    timeframe: str = Query("1day", description="Data timeframe (1min, 5min, 15min, 30min, 1hour, 1day)"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, ge=1, le=5000, description="Maximum number of data points"),
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
):
    """Get historical market data for a symbol."""
    try:
        symbol = symbol.upper()
        
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        
        if not start_date:
            # Default to 30 days back
            start_date = end_date - timedelta(days=30)
        
        # Validate date range
        if start_date >= end_date:
            raise APIException(
                status_code=400,
                detail="Start date must be before end date",
                error_code="INVALID_DATE_RANGE"
            )
        
        # Validate timeframe
        valid_timeframes = ['1min', '5min', '15min', '30min', '1hour', '1day', '1week', '1month']
        if timeframe not in valid_timeframes:
            raise APIException(
                status_code=400,
                detail=f"Invalid timeframe. Must be one of: {valid_timeframes}",
                error_code="INVALID_TIMEFRAME"
            )
        
        # Generate mock historical data
        market_data_list = []
        
        try:
            # Import data provider service
            from services.data_provider.data_provider_service import get_data_provider_service
            
            data_service = await get_data_provider_service()
            
            # Try to get real historical data
            # data_list = await data_service.get_historical_data(symbol, timeframe, start_date, end_date)
            # For now, generate mock data
            data_list = _generate_mock_historical_data(symbol, start_date, end_date, timeframe, limit)
            
            for data in data_list[:limit]:
                market_data_list.append(MarketDataPoint(
                    symbol=data.get('symbol', symbol),
                    timestamp=data.get('timestamp'),
                    open=data.get('open'),
                    high=data.get('high'),
                    low=data.get('low'),
                    close=data.get('close'),
                    volume=data.get('volume'),
                    timeframe=timeframe,
                    data_source=data.get('data_source', 'mock')
                ))
                
        except Exception as e:
            logger.warning(f"Failed to get historical data for {symbol}, using mock: {e}")
            # Generate mock data as fallback
            mock_data = _generate_mock_historical_data(symbol, start_date, end_date, timeframe, limit)
            for data in mock_data:
                market_data_list.append(MarketDataPoint(
                    symbol=symbol,
                    timestamp=data['timestamp'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    volume=data['volume'],
                    timeframe=timeframe,
                    data_source='mock'
                ))
        
        return MarketDataResponse(
            data=market_data_list,
            count=len(market_data_list),
            symbol=symbol,
            message=f"Retrieved {len(market_data_list)} historical data points for {symbol}"
        )
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        raise APIException(
            status_code=500,
            detail=f"Failed to fetch historical data for {symbol}",
            error_code="HISTORICAL_DATA_ERROR",
            context={"symbol": symbol, "error": str(e)}
        )


@router.get(
    "/market-data/{symbol}/quote",
    response_model=Dict[str, Any],
    summary="Get detailed quote",
    description="Get detailed quote information including bid/ask, day range, and trading stats"
)
async def get_detailed_quote(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL)"),
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
):
    """Get detailed quote information for a symbol."""
    try:
        symbol = symbol.upper()
        
        # Mock detailed quote data
        quote = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "price": 151.75,
            "change": 1.25,
            "change_percent": 0.83,
            "volume": 1500000,
            "avg_volume": 50000000,
            "bid": 151.70,
            "ask": 151.80,
            "bid_size": 100,
            "ask_size": 200,
            "day_low": 149.25,
            "day_high": 152.50,
            "open": 150.00,
            "previous_close": 150.50,
            "market_cap": 2500000000000.0,
            "pe_ratio": 28.5,
            "dividend_yield": 0.44,
            "52_week_high": 180.00,
            "52_week_low": 120.00
        }
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "data": quote,
            "message": f"Retrieved detailed quote for {symbol}"
        }
        
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {e}")
        raise APIException(
            status_code=500,
            detail=f"Failed to fetch quote for {symbol}",
            error_code="QUOTE_ERROR",
            context={"symbol": symbol, "error": str(e)}
        )


@router.get(
    "/market-data/batch",
    response_model=Dict[str, MarketDataResponse],
    summary="Get batch market data",
    description="Get current market data for multiple symbols in a single request"
)
async def get_batch_market_data(
    symbols: str = Query(..., description="Comma-separated list of symbols (e.g., AAPL,GOOGL,TSLA)"),
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
):
    """Get market data for multiple symbols."""
    try:
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Limit number of symbols
        if len(symbol_list) > 50:
            raise APIException(
                status_code=400,
                detail="Too many symbols requested. Maximum 50 symbols per batch request.",
                error_code="TOO_MANY_SYMBOLS"
            )
        
        # Get data for each symbol
        batch_data = {}
        
        for symbol in symbol_list:
            try:
                # Mock data for each symbol
                current_time = datetime.utcnow()
                mock_data = MarketDataPoint(
                    symbol=symbol,
                    timestamp=current_time,
                    open=100.00 + hash(symbol) % 100,
                    high=105.00 + hash(symbol) % 100,
                    low=95.00 + hash(symbol) % 100,
                    close=102.50 + hash(symbol) % 100,
                    volume=1000000 + hash(symbol) % 5000000,
                    timeframe="current",
                    data_source="mock"
                )
                
                batch_data[symbol] = MarketDataResponse(
                    data=[mock_data],
                    count=1,
                    symbol=symbol,
                    message=f"Retrieved data for {symbol}"
                )
                
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
                batch_data[symbol] = MarketDataResponse(
                    success=False,
                    data=[],
                    count=0,
                    symbol=symbol,
                    message=f"Failed to retrieve data for {symbol}: {str(e)}"
                )
        
        return batch_data
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"Error in batch market data request: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to process batch market data request",
            error_code="BATCH_DATA_ERROR",
            context={"error": str(e)}
        )


def _generate_mock_historical_data(symbol: str, start_date: datetime, end_date: datetime, 
                                 timeframe: str, limit: int) -> List[Dict[str, Any]]:
    """Generate mock historical data for testing."""
    import random
    
    data_points = []
    current_date = start_date
    
    # Determine time delta based on timeframe
    if timeframe == '1min':
        delta = timedelta(minutes=1)
    elif timeframe == '5min':
        delta = timedelta(minutes=5)
    elif timeframe == '15min':
        delta = timedelta(minutes=15)
    elif timeframe == '30min':
        delta = timedelta(minutes=30)
    elif timeframe == '1hour':
        delta = timedelta(hours=1)
    elif timeframe == '1day':
        delta = timedelta(days=1)
    else:
        delta = timedelta(days=1)
    
    base_price = 100.0
    
    while current_date <= end_date and len(data_points) < limit:
        # Generate OHLCV data with some randomness
        open_price = base_price + random.uniform(-2, 2)
        close_price = open_price + random.uniform(-1, 1)
        high_price = max(open_price, close_price) + random.uniform(0, 1)
        low_price = min(open_price, close_price) - random.uniform(0, 1)
        volume = random.randint(100000, 5000000)
        
        data_points.append({
            'timestamp': current_date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        base_price = close_price  # Use close as next base price for continuity
        current_date += delta
    
    return data_points
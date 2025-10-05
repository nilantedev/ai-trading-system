#!/usr/bin/env python3
"""
QuestDB Data Loader for Backtesting - HTTP API Version

Efficiently loads historical market data from QuestDB using HTTP API.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "../../../shared/python-common"))
from trading_common import get_logger

logger = get_logger(__name__)


class QuestDBDataLoader:
    """Loads historical market data from QuestDB using HTTP API"""
    
    def __init__(self, host: str = "questdb", port: int = 9000):
        """Initialize QuestDB HTTP client"""
        self.base_url = f"http://{host}:{port}"
        logger.info(f"QuestDB HTTP loader initialized: {self.base_url}")
    
    def _execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute SQL query via HTTP API"""
        try:
            response = requests.get(
                f"{self.base_url}/exec",
                params={"query": query},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("dataset"):
                return None
            
            columns = [col["name"] for col in data["columns"]]
            df = pd.DataFrame(data["dataset"], columns=columns)
            return df
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None
    
    def load_ohlcv(self, symbol: str, start_date: datetime, end_date: datetime,
                   timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """Load OHLCV data for a symbol"""
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        query = f"""
        SELECT timestamp, first(price) as open, max(price) as high, min(price) as low,
               last(price) as close, sum(volume) as volume
        FROM market_data
        WHERE symbol = '{symbol}' AND timestamp >= '{start_str}' AND timestamp < '{end_str}'
        SAMPLE BY 1d ALIGN TO CALENDAR
        """
        
        df = self._execute_query(query)
        if df is None or len(df) == 0:
            logger.warning(f"No data for {symbol} from {start_date} to {end_date}")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        logger.info(f"Loaded {len(df)} bars for {symbol}")
        return df
    
    def load_multiple_symbols(self, symbols: List[str], start_date: datetime,
                             end_date: datetime, timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        """Load OHLCV data for multiple symbols"""
        data = {}
        for symbol in symbols:
            df = self.load_ohlcv(symbol, start_date, end_date, timeframe)
            if df is not None and len(df) > 0:
                data[symbol] = df
        logger.info(f"Loaded data for {len(data)}/{len(symbols)} symbols")
        return data
    
    def get_available_symbols(self, min_bars: int = 100) -> List[str]:
        """Get list of symbols with sufficient data"""
        query = f"""
        SELECT symbol, COUNT(*) as bar_count
        FROM market_data
        GROUP BY symbol
        HAVING COUNT(*) >= {min_bars}
        ORDER BY bar_count DESC
        """
        df = self._execute_query(query)
        if df is None:
            return []
        symbols = df['symbol'].tolist()
        logger.info(f"Found {len(symbols)} symbols with >= {min_bars} bars")
        return symbols
    
    def get_date_range(self, symbol: str) -> Optional[tuple]:
        """Get available date range for a symbol"""
        query = f"""
        SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date
        FROM market_data
        WHERE symbol = '{symbol}'
        """
        df = self._execute_query(query)
        if df is None or len(df) == 0:
            return None
        min_date = pd.to_datetime(df['min_date'].iloc[0])
        max_date = pd.to_datetime(df['max_date'].iloc[0])
        return (min_date, max_date)

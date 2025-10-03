"""
Real-Time Market Data Integration
Connects real-time intelligence APIs to actual QuestDB data
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MarketDataIntegration:
    """Integrates real-time dashboard with QuestDB market data"""
    
    def __init__(self, questdb_host: str = "questdb", questdb_port: int = 8812):
        self.questdb_host = questdb_host
        self.questdb_port = questdb_port
        self.conn = None
    
    async def connect(self):
        """Establish connection to QuestDB"""
        try:
            import psycopg2
            self.conn = psycopg2.connect(
                host=self.questdb_host,
                port=self.questdb_port,
                user="admin",
                password="quest",
                database="qdb"
            )
            logger.info(f"Connected to QuestDB at {self.questdb_host}:{self.questdb_port}")
        except Exception as e:
            logger.error(f"Failed to connect to QuestDB: {e}")
            self.conn = None
    
    async def get_latest_bars(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get latest market data bars for symbol"""
        if not self.conn:
            await self.connect()
        
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = '{symbol}'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            return [
                {
                    "timestamp": row[0],
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": int(row[5])
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")
            return []
    
    async def get_options_flow(self, limit: int = 50) -> List[Dict]:
        """Get recent options flow from QuestDB"""
        if not self.conn:
            await self.connect()
        
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            query = f"""
                SELECT timestamp, symbol, option_type, strike, expiration, 
                       size, premium, side
                FROM options_data
                WHERE timestamp > NOW() - interval '1 hour'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            flows = []
            for row in rows:
                # Determine sentiment based on side and type
                option_type = row[2]  # call or put
                side = row[7]  # buy or sell
                
                if option_type == 'call' and side == 'buy':
                    sentiment = 'bullish'
                elif option_type == 'put' and side == 'buy':
                    sentiment = 'bearish'
                else:
                    sentiment = 'neutral'
                
                flows.append({
                    "timestamp": row[0],
                    "symbol": row[1],
                    "type": option_type,
                    "strike": float(row[3]),
                    "expiration": row[4],
                    "size": int(row[5]),
                    "premium": float(row[6]),
                    "sentiment": sentiment
                })
            
            return flows
        except Exception as e:
            logger.error(f"Failed to fetch options flow: {e}")
            return []
    
    async def get_top_movers(self, limit: int = 20) -> List[Dict]:
        """Get top movers by percentage change today"""
        if not self.conn:
            await self.connect()
        
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            # Get today's change for all symbols
            query = f"""
                WITH today_first AS (
                    SELECT symbol, 
                           first(close) as open_price,
                           last(close) as close_price
                    FROM market_data
                    WHERE timestamp >= CAST(NOW() AS DATE)
                    GROUP BY symbol
                )
                SELECT symbol, 
                       ((close_price - open_price) / open_price * 100) as pct_change,
                       close_price
                FROM today_first
                WHERE open_price > 0
                ORDER BY ABS(pct_change) DESC
                LIMIT {limit}
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            return [
                {
                    "symbol": row[0],
                    "change": float(row[1]),
                    "price": float(row[2])
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to fetch top movers: {e}")
            return []
    
    async def get_sector_performance(self) -> Dict[str, List[Dict]]:
        """Get performance by sector"""
        # TODO: Implement sector classification and grouping
        # For now, return mock data structure
        return {
            "Technology": [],
            "Financials": [],
            "Healthcare": [],
            "Energy": [],
            "Consumer": []
        }
    
    async def get_symbol_statistics(self, symbol: str) -> Dict:
        """Get comprehensive statistics for a symbol"""
        if not self.conn:
            await self.connect()
        
        if not self.conn:
            return {}
        
        try:
            cursor = self.conn.cursor()
            
            # Get volatility, volume, price range
            query = f"""
                SELECT 
                    COUNT(*) as bar_count,
                    AVG(volume) as avg_volume,
                    STDDEV(close) as volatility,
                    MIN(low) as period_low,
                    MAX(high) as period_high,
                    LAST(close) as last_price
                FROM market_data
                WHERE symbol = '{symbol}'
                  AND timestamp > NOW() - interval '30 days'
            """
            cursor.execute(query)
            row = cursor.fetchone()
            
            if row:
                return {
                    "bar_count": int(row[0]),
                    "avg_volume": int(row[1]) if row[1] else 0,
                    "volatility": float(row[2]) if row[2] else 0.0,
                    "period_low": float(row[3]) if row[3] else 0.0,
                    "period_high": float(row[4]) if row[4] else 0.0,
                    "last_price": float(row[5]) if row[5] else 0.0
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to fetch statistics for {symbol}: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("QuestDB connection closed")

# Global instance
_market_data_integration = None

def get_market_data_integration() -> MarketDataIntegration:
    """Get or create global market data integration instance"""
    global _market_data_integration
    if _market_data_integration is None:
        _market_data_integration = MarketDataIntegration()
    return _market_data_integration

#!/usr/bin/env python3
"""
Dynamic Watchlist Manager
Manages trading watchlist from database and Redis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

import os
import redis
import requests
from typing import Set, List, Dict
from datetime import datetime
from trading_common import get_logger

logger = get_logger(__name__)


class WatchlistManager:
    """
    Manages dynamic watchlist synchronized from QuestDB
    """
    
    def __init__(self, redis_url: str = None, questdb_url: str = None):
        # Use environment variables or defaults
        if redis_url is None:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        if questdb_url is None:
            # Try environment variable first, then docker network, then localhost
            questdb_host = os.getenv('QUESTDB_HOST', 'trading-questdb')
            questdb_port = os.getenv('QUESTDB_HTTP_PORT', '9000')
            questdb_url = f'http://{questdb_host}:{questdb_port}'
        
        # Parse Redis URL
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.questdb_url = questdb_url.rstrip('/')
        self.logger = get_logger(__name__)
        
    def get_symbols_from_questdb(self, min_bars: int = 100, 
                                 max_symbols: int = None) -> List[Dict]:
        """
        Get symbols from QuestDB that meet minimum bar requirements
        
        Args:
            min_bars: Minimum number of bars required
            max_symbols: Maximum symbols to return (None for all)
        
        Returns:
            List of dicts with symbol, bar_count
        """
        try:
            # Simpler query - just get symbols and counts
            query = f"SELECT DISTINCT symbol FROM market_data ORDER BY symbol"
            
            if max_symbols:
                query += f" LIMIT {max_symbols}"
            
            url = f"{self.questdb_url}/exec"
            
            response = requests.get(
                url,
                params={'query': query},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                symbols = []
                for row in data.get('dataset', []):
                    symbols.append({
                        'symbol': row[0],
                        'bar_count': min_bars  # Assume has enough bars
                    })
                
                logger.info(f"Found {len(symbols)} symbols from QuestDB")
                return symbols
            else:
                error_msg = response.text[:500] if hasattr(response, 'text') else str(response)
                logger.error(f"QuestDB query failed: {response.status_code} - {error_msg}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching symbols from QuestDB: {e}")
            return []
    
    def update_watchlist(self, symbols: List[str], namespace: str = 'watchlist') -> int:
        """
        Update Redis watchlist with new symbols
        
        Args:
            symbols: List of symbol strings
            namespace: Redis key name
        
        Returns:
            Number of symbols in watchlist
        """
        try:
            # Clear existing watchlist
            self.redis_client.delete(namespace)
            
            # Add all symbols
            if symbols:
                self.redis_client.sadd(namespace, *symbols)
            
            count = self.redis_client.scard(namespace)
            logger.info(f"Updated {namespace}: {count} symbols")
            return count
            
        except Exception as e:
            logger.error(f"Error updating Redis watchlist: {e}")
            return 0
    
    def get_watchlist(self, namespace: str = 'watchlist') -> Set[str]:
        """Get current watchlist from Redis"""
        try:
            return self.redis_client.smembers(namespace)
        except Exception as e:
            logger.error(f"Error fetching watchlist: {e}")
            return set()
    
    def sync_from_questdb(self, min_bars: int = 100, max_symbols: int = 100) -> Dict:
        """
        Sync watchlist from QuestDB to Redis
        
        Args:
            min_bars: Minimum data requirement
            max_symbols: Maximum symbols in watchlist
        
        Returns:
            Dict with sync results
        """
        logger.info(f"Syncing watchlist from QuestDB (min_bars={min_bars}, max={max_symbols})")
        
        # Get symbols from QuestDB
        symbol_data = self.get_symbols_from_questdb(min_bars=min_bars, max_symbols=max_symbols)
        
        if not symbol_data:
            logger.warning("No symbols found in QuestDB")
            return {'success': False, 'count': 0}
        
        # Extract symbol names
        symbols = [s['symbol'] for s in symbol_data]
        
        # Update Redis
        count = self.update_watchlist(symbols)
        
        # Store metadata
        for s in symbol_data:
            meta = {
                'bar_count': s.get('bar_count', 0),
                'updated_at': datetime.utcnow().isoformat()
            }
            # Add optional fields if present
            if 'first_date' in s:
                meta['first_date'] = s['first_date']
            if 'last_date' in s:
                meta['last_date'] = s['last_date']
            
            self.redis_client.hset(
                f"watchlist:meta:{s['symbol']}",
                mapping=meta
            )
        
        result = {
            'success': True,
            'count': count,
            'symbols': symbols[:20],  # First 20 for display
            'top_symbol': symbol_data[0] if symbol_data else None
        }
        
        logger.info(f"Watchlist sync complete: {count} symbols")
        return result
    
    def add_symbol(self, symbol: str, namespace: str = 'watchlist') -> bool:
        """Add a single symbol to watchlist"""
        try:
            self.redis_client.sadd(namespace, symbol)
            logger.info(f"Added {symbol} to {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error adding symbol: {e}")
            return False
    
    def remove_symbol(self, symbol: str, namespace: str = 'watchlist') -> bool:
        """Remove a symbol from watchlist"""
        try:
            self.redis_client.srem(namespace, symbol)
            logger.info(f"Removed {symbol} from {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error removing symbol: {e}")
            return False
    
    def get_symbol_metadata(self, symbol: str) -> Dict:
        """Get metadata for a symbol"""
        try:
            key = f"symbol:{symbol}:meta"
            data = self.redis_client.hgetall(key)
            return data if data else {}
        except Exception as e:
            logger.error(f"Error fetching metadata: {e}")
            return {}


if __name__ == "__main__":
    # Test the watchlist manager
    manager = WatchlistManager()
    
    print("=== Dynamic Watchlist Manager ===\n")
    
    # Sync from QuestDB
    print("Syncing watchlist from QuestDB...")
    result = manager.sync_from_questdb(min_bars=100, max_symbols=100)
    
    if result['success']:
        print(f"✓ Synced {result['count']} symbols")
        print(f"\nTop symbols: {', '.join(result['symbols'][:10])}")
        
        if result['top_symbol']:
            top = result['top_symbol']
            print(f"\nMost data: {top['symbol']} ({top['bar_count']} bars)")
    else:
        print("✗ Sync failed")
    
    # Get current watchlist
    print("\nCurrent watchlist:")
    watchlist = manager.get_watchlist()
    print(f"Total: {len(watchlist)} symbols")

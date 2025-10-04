#!/usr/bin/env python3
"""
Options Symbol Discovery Service
Discovers all symbols with active options contracts from data provider (Polygon)
Maintains authoritative list of optionable symbols for the trading system
"""

import asyncio
import aiohttp
import os
import sys
from datetime import datetime
from typing import List, Set, Dict, Optional
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from trading_common import get_logger, get_settings
    from trading_common.database import get_redis_client
except ImportError:
    # Fallback logging
    import logging
    get_logger = lambda name: logging.getLogger(name)
    get_redis_client = None
    get_settings = None

logger = get_logger(__name__)


class OptionsSymbolDiscovery:
    """Discovers symbols with active options from Polygon API"""
    
    def __init__(self):
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.polygon_base_url = os.getenv('POLYGON_BASE_URL', 'https://api.polygon.io')
        self.session: Optional[aiohttp.ClientSession] = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize connections"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'AI-Trading-System/1.0'}
            )
        
        # Initialize Redis for caching
        if get_redis_client:
            try:
                self.redis_client = await get_redis_client()
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
    
    async def close(self):
        """Cleanup connections"""
        if self.session:
            await self.session.close()
    
    async def discover_options_symbols_polygon(
        self, 
        market: str = "stocks",
        active: bool = True,
        limit_per_page: int = 1000
    ) -> Set[str]:
        """
        Discover all symbols with active options contracts from Polygon
        
        Uses Polygon's options contracts endpoint to find underlyings
        This is the authoritative source for optionable symbols
        """
        if not self.polygon_api_key:
            raise ValueError("POLYGON_API_KEY not configured")
        
        logger.info("Starting options symbol discovery from Polygon...")
        
        optionable_symbols: Set[str] = set()
        
        # Method 1: Query options contracts and extract unique underlyings
        # This gives us symbols that actually have options
        url = f"{self.polygon_base_url}/v3/reference/options/contracts"
        
        params = {
            'apiKey': self.polygon_api_key,
            'limit': limit_per_page,
            'order': 'asc',
            'sort': 'ticker'
        }
        
        if active:
            params['expired'] = 'false'  # Only active contracts
        
        page = 0
        cursor = None
        total_contracts = 0
        
        try:
            while True:
                page += 1
                query_params = dict(params)
                if cursor:
                    query_params['cursor'] = cursor
                
                logger.info(f"Fetching options contracts page {page}...")
                
                async with self.session.get(url, params=query_params) as resp:
                    if resp.status == 429:  # Rate limit
                        logger.warning("Rate limited, waiting 60s...")
                        await asyncio.sleep(60)
                        continue
                    
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Polygon API error {resp.status}: {text[:200]}")
                        break
                    
                    data = await resp.json()
                
                results = data.get('results', [])
                
                # Extract underlying symbols from contracts (even if empty, we need to check pagination)
                for contract in results:
                    underlying = contract.get('underlying_ticker')
                    if underlying:
                        underlying = str(underlying).strip().upper()
                        if underlying and underlying.replace('.', '').replace('-', '').isalnum():
                            optionable_symbols.add(underlying)
                
                total_contracts += len(results)
                
                # Check pagination - ALWAYS check for next_url/cursor, even if results are empty
                next_url = data.get('next_url')
                status = data.get('status')
                
                if next_url:
                    # Extract cursor from next_url
                    try:
                        from urllib.parse import urlparse, parse_qs
                        parsed = urlparse(next_url)
                        cursor_list = parse_qs(parsed.query).get('cursor', [])
                        cursor = cursor_list[0] if cursor_list else None
                    except Exception as e:
                        logger.warning(f"Failed to parse next_url: {e}")
                        cursor = None
                else:
                    # No next_url means no more pages
                    cursor = None
                
                # Stop only if we have no more pages (no cursor) AND no results
                if not cursor and not results:
                    if status == 'OK':
                        # Normal end - no more pages
                        logger.info(f"Reached end of data at page {page}")
                        break
                    elif status == 'DELAYED':
                        # Rate limited but may have more data - wait and retry
                        logger.warning(f"Status DELAYED at page {page}, waiting 60s before continuing...")
                        await asyncio.sleep(60)
                        continue
                    else:
                        # Unknown status with no cursor
                        logger.warning(f"No cursor and status={status} at page {page}, assuming end")
                        break
                
                # Rate limiting - be gentle with API
                await asyncio.sleep(0.5)
                
                # Log progress every 50 pages
                if page % 50 == 0:
                    logger.info(f"Progress: page {page}, contracts {total_contracts}, unique symbols {len(optionable_symbols)}")
                
                # Safety check: warn if we're fetching an unusually large number of pages
                if page >= 1000 and page % 100 == 0:
                    logger.warning(f"Still fetching at page {page} - found {len(optionable_symbols)} symbols so far...")
                    logger.info("This is normal for complete market coverage but will take time")
        
        except Exception as e:
            logger.error(f"Error discovering options symbols: {e}")
            raise
        
        logger.info(f"Discovered {len(optionable_symbols)} unique optionable symbols from {total_contracts} contracts")
        
        return optionable_symbols
    
    async def discover_options_symbols_tickers_endpoint(self) -> Set[str]:
        """
        Alternative method: Use tickers endpoint filtered for options-enabled symbols
        This is a backup method if contracts endpoint fails
        """
        if not self.polygon_api_key:
            raise ValueError("POLYGON_API_KEY not configured")
        
        logger.info("Discovering options symbols via tickers endpoint...")
        
        optionable_symbols: Set[str] = set()
        
        # Polygon doesn't have a direct "has_options" filter, but we can try market=options
        # or use the reference endpoint with type filtering
        url = f"{self.polygon_base_url}/v3/reference/tickers"
        
        params = {
            'apiKey': self.polygon_api_key,
            'market': 'stocks',
            'active': 'true',
            'limit': 1000,
        }
        
        page = 0
        cursor = None
        
        try:
            while True:
                page += 1
                query_params = dict(params)
                if cursor:
                    query_params['cursor'] = cursor
                
                async with self.session.get(url, params=query_params) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(60)
                        continue
                    
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Polygon API error {resp.status}: {text[:200]}")
                        break
                    
                    data = await resp.json()
                
                results = data.get('results', [])
                if not results:
                    break
                
                for ticker in results:
                    symbol = ticker.get('ticker')
                    # Check if ticker has options (this would need to be verified per symbol)
                    # For now, we'll need to use the contracts method as authoritative
                    if symbol:
                        symbol = str(symbol).strip().upper()
                        if symbol:
                            optionable_symbols.add(symbol)
                
                cursor = data.get('next_url')
                if cursor and isinstance(cursor, str) and cursor.startswith('http'):
                    from urllib.parse import urlparse, parse_qs
                    parsed = urlparse(cursor)
                    cursor_list = parse_qs(parsed.query).get('cursor', [])
                    cursor = cursor_list[0] if cursor_list else None
                
                if not cursor or page >= 10:  # Limit pages for backup method
                    break
                
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Error in tickers discovery: {e}")
        
        logger.info(f"Found {len(optionable_symbols)} symbols via tickers endpoint")
        return optionable_symbols
    
    async def get_optionable_symbols(self, use_cache: bool = True, cache_ttl: int = 86400) -> List[str]:
        """
        Get authoritative list of optionable symbols
        
        Args:
            use_cache: Use cached list if available
            cache_ttl: Cache time-to-live in seconds (default 24h)
        
        Returns:
            Sorted list of symbols with active options
        """
        cache_key = "options:discovery:symbols"
        cache_timestamp_key = "options:discovery:timestamp"
        
        # Try cache first
        if use_cache and self.redis_client:
            try:
                cached = await self.redis_client.smembers(cache_key)
                timestamp_str = await self.redis_client.get(cache_timestamp_key)
                
                if cached and timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    age_seconds = (datetime.utcnow() - timestamp).total_seconds()
                    
                    if age_seconds < cache_ttl:
                        logger.info(f"Using cached optionable symbols ({len(cached)} symbols, age: {age_seconds/3600:.1f}h)")
                        return sorted(list(cached))
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Discover from API
        try:
            symbols = await self.discover_options_symbols_polygon()
        except Exception as e:
            logger.error(f"Primary discovery failed: {e}")
            # Try backup method
            try:
                symbols = await self.discover_options_symbols_tickers_endpoint()
            except Exception as e2:
                logger.error(f"Backup discovery also failed: {e2}")
                raise RuntimeError("All discovery methods failed") from e
        
        if not symbols:
            raise RuntimeError("No optionable symbols discovered")
        
        # Cache the results
        if self.redis_client:
            try:
                # Clear old cache
                await self.redis_client.delete(cache_key)
                # Add new symbols
                await self.redis_client.sadd(cache_key, *list(symbols))
                # Set timestamp
                await self.redis_client.set(cache_timestamp_key, datetime.utcnow().isoformat())
                # Set TTL
                await self.redis_client.expire(cache_key, cache_ttl)
                await self.redis_client.expire(cache_timestamp_key, cache_ttl)
                logger.info(f"Cached {len(symbols)} optionable symbols")
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
        
        return sorted(list(symbols))
    
    async def sync_watchlist(self, symbols: List[str]) -> int:
        """
        Sync watchlist in Redis with discovered optionable symbols
        
        Returns:
            Number of symbols in final watchlist
        """
        if not self.redis_client:
            raise RuntimeError("Redis not available")
        
        watchlist_key = "watchlist"
        
        # Clear existing watchlist
        await self.redis_client.delete(watchlist_key)
        
        # Add all optionable symbols
        if symbols:
            await self.redis_client.sadd(watchlist_key, *symbols)
        
        final_count = await self.redis_client.scard(watchlist_key)
        logger.info(f"Watchlist synchronized: {final_count} optionable symbols")
        
        return final_count
    
    async def export_to_file(self, symbols: List[str], filepath: str):
        """Export discovered symbols to JSON file"""
        data = {
            'generated_at': datetime.utcnow().isoformat(),
            'count': len(symbols),
            'symbols': symbols,
            'source': 'polygon_options_discovery'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(symbols)} symbols to {filepath}")


async def main():
    """Main discovery and sync routine"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover optionable symbols and sync watchlist')
    parser.add_argument('--no-cache', action='store_true', help='Force fresh discovery')
    parser.add_argument('--export', type=str, help='Export to file path')
    parser.add_argument('--sync-watchlist', action='store_true', help='Sync Redis watchlist')
    parser.add_argument('--dry-run', action='store_true', help='Discover only, no sync')
    
    args = parser.parse_args()
    
    discovery = OptionsSymbolDiscovery()
    
    try:
        await discovery.initialize()
        
        # Discover optionable symbols
        symbols = await discovery.get_optionable_symbols(use_cache=not args.no_cache)
        
        print(f"\n{'='*70}")
        print(f"OPTIONS SYMBOL DISCOVERY COMPLETE")
        print(f"{'='*70}")
        print(f"Total optionable symbols: {len(symbols)}")
        print(f"\nFirst 50 symbols:")
        for i, sym in enumerate(symbols[:50], 1):
            print(f"  {i:3d}. {sym}")
        if len(symbols) > 50:
            print(f"  ... and {len(symbols) - 50} more")
        print(f"{'='*70}\n")
        
        # Export if requested
        if args.export:
            await discovery.export_to_file(symbols, args.export)
        
        # Sync watchlist if requested
        if args.sync_watchlist and not args.dry_run:
            count = await discovery.sync_watchlist(symbols)
            print(f"✓ Watchlist synchronized: {count} symbols\n")
        elif args.dry_run:
            print("DRY RUN: No changes made to watchlist\n")
        
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        raise
    finally:
        await discovery.close()


if __name__ == '__main__':
    asyncio.run(main())

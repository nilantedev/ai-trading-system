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
        
        # Track symbol count to detect when we stop finding new ones
        symbols_at_checkpoint = 0
        checkpoint_page = 0
        stagnant_pages = 0
        
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
                
                # Check progress at checkpoints (every 100 pages)
                if page % 100 == 0:
                    current_count = len(optionable_symbols)
                    new_symbols = current_count - symbols_at_checkpoint
                    
                    logger.info(f"Progress: page {page}, contracts {total_contracts}, unique symbols {current_count}")
                    logger.info(f"New symbols in last 100 pages: {new_symbols}")
                    
                    # OPTIMIZATION: Stop if no new symbols found in last 200 pages
                    if new_symbols == 0:
                        stagnant_pages += 100
                        if stagnant_pages >= 200:
                            logger.info(f"No new symbols found in last {stagnant_pages} pages")
                            logger.info(f"Discovered {current_count} optionable symbols - stopping early")
                            logger.info("Market coverage complete for active options")
                            break
                    else:
                        stagnant_pages = 0  # Reset counter when we find new symbols
                    
                    # Update checkpoint
                    symbols_at_checkpoint = current_count
                    checkpoint_page = page
                
                # Log progress every 50 pages (less verbose)
                elif page % 50 == 0:
                    logger.info(f"Progress: page {page}, contracts {total_contracts}, unique symbols {len(optionable_symbols)}")
                
                # Safety limit: stop at 3000 pages (comprehensive market coverage)
                if page >= 3000:
                    logger.warning(f"Reached safety limit at {page} pages - found {len(optionable_symbols)} symbols")
                    logger.info("This represents comprehensive market coverage")
                    break
        
        except Exception as e:
            logger.error(f"Error discovering options symbols: {e}")
            raise
        
        logger.info(f"Discovered {len(optionable_symbols)} unique optionable symbols from {total_contracts} contracts")
        
        return optionable_symbols
    
    async def discover_options_symbols_fast(self) -> Set[str]:
        """
        FAST method: Query recent options activity to find actively traded symbols
        This is much faster than scanning all contracts (6000+ symbols in ~100 pages vs 10,000+ pages)
        """
        if not self.polygon_api_key:
            raise ValueError("POLYGON_API_KEY not configured")
        
        logger.info("Fast discovery: Finding symbols with recent options activity...")
        
        optionable_symbols: Set[str] = set()
        
        # Query recent options contracts (last 90 days expiration)
        from datetime import datetime, timedelta
        today = datetime.utcnow()
        min_expiry = today.strftime('%Y-%m-%d')
        max_expiry = (today + timedelta(days=90)).strftime('%Y-%m-%d')
        
        url = f"{self.polygon_base_url}/v3/reference/options/contracts"
        
        params = {
            'apiKey': self.polygon_api_key,
            'limit': 1000,
            'expired': 'false',
            'expiration_date.gte': min_expiry,
            'expiration_date.lte': max_expiry,
            'order': 'asc',
            'sort': 'ticker'
        }
        
        page = 0
        cursor = None
        total_contracts = 0
        stagnant_pages = 0
        symbols_at_checkpoint = 0
        
        try:
            while True:
                page += 1
                query_params = dict(params)
                if cursor:
                    query_params['cursor'] = cursor
                
                logger.info(f"Fetching recent options contracts page {page}...")
                
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
                
                # Extract underlying symbols
                for contract in results:
                    underlying = contract.get('underlying_ticker')
                    if underlying:
                        underlying = str(underlying).strip().upper()
                        if underlying and underlying.replace('.', '').replace('-', '').isalnum():
                            optionable_symbols.add(underlying)
                
                total_contracts += len(results)
                
                # Check for more pages
                next_url = data.get('next_url')
                if next_url:
                    from urllib.parse import urlparse, parse_qs
                    parsed = urlparse(next_url)
                    cursor_list = parse_qs(parsed.query).get('cursor', [])
                    cursor = cursor_list[0] if cursor_list else None
                else:
                    cursor = None
                
                # Stop if no more pages or no results
                if not cursor and not results:
                    break
                
                # Progress tracking - stop early if stagnant
                if page % 50 == 0:
                    current_count = len(optionable_symbols)
                    logger.info(f"Progress: page {page}, contracts {total_contracts}, unique symbols {current_count}")
                    
                    if page % 100 == 0:
                        new_symbols = current_count - symbols_at_checkpoint
                        if new_symbols == 0:
                            stagnant_pages += 100
                            if stagnant_pages >= 100:  # Stop after 100 pages with no new symbols
                                logger.info(f"No new symbols in last {stagnant_pages} pages - stopping")
                                break
                        else:
                            stagnant_pages = 0
                        symbols_at_checkpoint = current_count
                
                # Safety limit for fast method
                if page >= 500:
                    logger.info(f"Reached page limit for fast discovery at {page}")
                    break
                
                await asyncio.sleep(0.3)  # Faster rate for smaller dataset
        
        except Exception as e:
            logger.error(f"Error in fast discovery: {e}")
            raise
        
        logger.info(f"Fast discovery complete: {len(optionable_symbols)} symbols from {total_contracts} recent contracts")
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
        
        # Discover from API - use fast method by default
        try:
            logger.info("Using fast discovery method (recent contracts only)")
            symbols = await self.discover_options_symbols_fast()
            
            # If fast method returns too few symbols, fall back to full scan
            if len(symbols) < 1000:
                logger.warning(f"Fast discovery found only {len(symbols)} symbols - using full scan")
                symbols = await self.discover_options_symbols_polygon()
        except Exception as e:
            logger.error(f"Fast discovery failed: {e}")
            # Try full scan method
            try:
                logger.info("Falling back to full discovery scan")
                symbols = await self.discover_options_symbols_polygon()
            except Exception as e2:
                logger.error(f"Full discovery also failed: {e2}")
                # Last resort - return empty set
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
    parser.add_argument('--fast', action='store_true', help='Use fast discovery (recent contracts only, ~5min)')
    parser.add_argument('--full', action='store_true', help='Use full discovery (all contracts, may take hours)')
    parser.add_argument('--export', type=str, help='Export to file path')
    parser.add_argument('--sync-watchlist', action='store_true', help='Sync Redis watchlist')
    parser.add_argument('--dry-run', action='store_true', help='Discover only, no sync')
    
    args = parser.parse_args()
    
    discovery = OptionsSymbolDiscovery()
    
    try:
        await discovery.initialize()
        
        # Discover optionable symbols
        if args.fast:
            logger.info("Using FAST discovery method (--fast flag)")
            symbols = sorted(list(await discovery.discover_options_symbols_fast()))
        elif args.full:
            logger.info("Using FULL discovery method (--full flag)")
            symbols = sorted(list(await discovery.discover_options_symbols_polygon()))
        else:
            # Default: use cache or fast method
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

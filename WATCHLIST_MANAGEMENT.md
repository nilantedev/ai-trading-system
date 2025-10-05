# Watchlist Management System Documentation

## Overview

The trading system uses a Redis-based watchlist to track symbols for data collection, analysis, and trading. The watchlist contains **optionable symbols only** - stocks with active options contracts suitable for advanced trading strategies.

## System Components

### 1. Options Symbol Discovery (`options_symbol_discovery.py`)

**Purpose**: Discovers all symbols with active options contracts from Polygon API

**Key Features**:
- Queries Polygon `/v3/reference/options/contracts` endpoint
- Extracts unique underlying symbols from options contracts
- Caches results in Redis (24-hour TTL)
- Handles API rate limits and pagination
- Falls back to alternative discovery method if primary fails

**Methods**:
- `discover_options_symbols_polygon()` - Primary discovery via options contracts
- `get_optionable_symbols()` - Main entry point with caching
- `sync_watchlist()` - Syncs discovered symbols to Redis watchlist
- `export_to_file()` - Exports symbol list to JSON

**Performance**: 20-30 minutes for full discovery due to Polygon rate limits (need to paginate through thousands of options contracts)

### 2. Watchlist Sync Script (`sync_optionable_watchlist.sh`)

**Purpose**: Wrapper script for running discovery and syncing watchlist

**Usage**:
```bash
# Full sync (recommended to run weekly or monthly)
./scripts/sync_optionable_watchlist.sh

# Force fresh discovery (ignore cache)
./scripts/sync_optionable_watchlist.sh --no-cache

# Dry run (discover only, don't update watchlist)
./scripts/sync_optionable_watchlist.sh --dry-run
```

**Recommended Schedule**:
- **Weekly**: For maintaining current optionable symbols
- **Monthly**: For slower-paced updates
- **After major market events**: IPOs, delistings, etc.

### 3. Lightweight Auto-Refresh (Optional)

**Purpose**: Daily refresh using fast method (top liquid symbols only)

**Configuration** (`.env`):
```bash
ENABLE_WATCHLIST_AUTO_REFRESH=false  # Disabled by default
WATCHLIST_REFRESH_INTERVAL_SECONDS=86400  # Daily
```

**Note**: This lightweight method gets top symbols quickly but may not include all optionable symbols. Use the full discovery script for comprehensive coverage.

## Redis Structure

### Watchlist Key
- **Key**: `"watchlist"`
- **Type**: SET
- **Contents**: Symbol tickers (e.g., "AAPL", "MSFT", "TSLA")
- **Size**: ~1000-2000 symbols (optionable stocks only)

### Discovery Cache Keys
- **Symbols**: `"options:discovery:symbols"` (SET)
- **Timestamp**: `"options:discovery:timestamp"` (STRING)
- **TTL**: 24 hours

## Data Flow

```
┌─────────────────────────────────────┐
│  Polygon API                         │
│  /v3/reference/options/contracts     │
└──────────────┬──────────────────────┘
               │
               │ (Paginated queries)
               │
               ▼
┌─────────────────────────────────────┐
│  Options Symbol Discovery            │
│  - Extract unique underlyings        │
│  - Deduplicate                       │
│  - Cache in Redis (24h)              │
└──────────────┬──────────────────────┘
               │
               │ sync_watchlist()
               │
               ▼
┌─────────────────────────────────────┐
│  Redis: watchlist SET                │
│  ~1000-2000 optionable symbols       │
└──────────────┬──────────────────────┘
               │
               │ get_watchlist_symbols()
               │
               ▼
┌─────────────────────────────────────┐
│  Downstream Services                 │
│  - Data Ingestion                    │
│  - Backfill                          │
│  - Strategy Engine                   │
│  - Signal Generator                  │
└─────────────────────────────────────┘
```

## Integration Points

### Data Ingestion Service
- Reads watchlist on startup and periodically
- Fetches real-time quotes for all watchlist symbols
- Collects news and social sentiment for watchlist symbols
- Method: `reference_data_service.get_watchlist_symbols()`

### Historical Backfill
- Processes watchlist symbols for historical data collection
- Limited by `EQUITY_BACKFILL_MAX_SYMBOLS` (default: 2000)
- Reads from: Redis `"watchlist"` key

### Retention Service
- Removes data for symbols NOT on watchlist
- Runs daily at 2:00 AM
- Keeps data within retention bounds (default: 20 years for bars)

### Strategy Engine
- Receives signals for watchlist symbols only
- Watchlist manager maintains local cache
- Method: `watchlist_manager.get_watchlist_symbols()`

## Operational Procedures

### Initial Setup
1. Run full discovery to populate watchlist:
   ```bash
   ./scripts/sync_optionable_watchlist.sh --no-cache
   ```

2. Verify watchlist size:
   ```bash
   docker exec trading-redis redis-cli -a <password> scard watchlist
   # Expected: 1000-2000
   ```

3. Check sample symbols:
   ```bash
   docker exec trading-redis redis-cli -a <password> srandmember watchlist 20
   ```

### Regular Maintenance

**Weekly Sync** (Recommended):
```bash
# Add to crontab
0 2 * * 0 /srv/ai-trading-system/scripts/sync_optionable_watchlist.sh >> /var/log/watchlist-sync.log 2>&1
```

**Monthly Sync** (Minimum):
```bash
# Add to crontab
0 2 1 * * /srv/ai-trading-system/scripts/sync_optionable_watchlist.sh --no-cache >> /var/log/watchlist-sync.log 2>&1
```

### Troubleshooting

**Watchlist is empty or too small**:
```bash
# Force fresh discovery
./scripts/sync_optionable_watchlist.sh --no-cache

# Check Redis connectivity
docker exec trading-redis redis-cli -a <password> ping
```

**Discovery times out**:
```bash
# Check Polygon API key
docker exec trading-data-ingestion env | grep POLYGON_API_KEY

# Check API rate limit status
curl "https://api.polygon.io/v3/reference/options/contracts?apiKey=<key>&limit=1"
```

**Symbols not being processed**:
```bash
# Restart services to pick up new watchlist
docker-compose restart data-ingestion strategy-engine

# Verify backfill is reading watchlist
docker logs trading-data-ingestion 2>&1 | grep "watchlist"
```

## Performance Considerations

### Discovery Performance
- **Full discovery**: 20-30 minutes (Polygon rate limits)
- **API calls**: ~5000+ (1000 contracts per page, need all contracts)
- **Network**: ~50-100 MB total data transfer
- **Memory**: ~500 MB peak (parsing options contracts)

### Redis Performance
- **Watchlist size**: ~50 KB (1000 symbols @ ~50 bytes each)
- **Cache size**: ~100 KB (cached discovery results)
- **Operations**: O(1) for membership checks, O(N) for full list

### Scaling Considerations
- **Current**: 1000-2000 symbols (all U.S. optionable stocks)
- **Target**: 6000 symbols (includes all optionable ETFs, international)
- **Maximum**: ~10,000 symbols (entire options market)

To scale beyond 2000 symbols:
1. Increase `EQUITY_BACKFILL_MAX_SYMBOLS` in .env
2. Add more API workers: `API_WORKERS=16`
3. Increase Redis memory: `maxmemory 256gb`
4. Scale horizontal: Multiple data-ingestion containers

## Configuration Reference

### Environment Variables

```bash
# Discovery system (runs in data-ingestion container)
POLYGON_API_KEY=<your_key>
POLYGON_BASE_URL=https://api.polygon.io

# Watchlist auto-refresh (disabled by default - use manual sync instead)
ENABLE_WATCHLIST_AUTO_REFRESH=false
WATCHLIST_REFRESH_INTERVAL_SECONDS=86400  # 24 hours

# Backfill limits
EQUITY_BACKFILL_MAX_SYMBOLS=2000
HISTORICAL_BACKFILL_YEARS=20

# Redis
REDIS_URL=redis://:password@redis:6379/0
REDIS_PASSWORD=<password>
```

### Redis Keys

```bash
# Watchlist (authoritative)
watchlist (SET)

# Discovery cache
options:discovery:symbols (SET, TTL 24h)
options:discovery:timestamp (STRING, TTL 24h)

# Per-symbol metadata
watchlist:meta:<SYMBOL> (HASH)
```

## Migration Notes

### From Old System
The previous system used `populate_watchlist_from_polygon()` which retrieved ALL Polygon symbols, not just optionable ones. This caused the watchlist to balloon to 11,000+ symbols including penny stocks, inactive tickers, etc.

**Changes**:
1. Replaced generic ticker retrieval with options-specific discovery
2. Disabled automatic refresh loop (too slow for production)
3. Created manual sync script for weekly/monthly execution
4. Added Redis caching to avoid re-discovering every run

### Backward Compatibility
- Redis key `"watchlist"` remains unchanged (SET of symbols)
- All services read from same key: `get_watchlist_symbols()`
- No API changes required for downstream services

## Security Considerations

1. **API Key Protection**: Polygon API key stored in .env, never logged
2. **Redis Password**: Required for all Redis operations
3. **Rate Limits**: Discovery respects Polygon rate limits (12 req/min)
4. **Container Isolation**: Discovery runs inside data-ingestion container

## Future Enhancements

1. **Multi-source discovery**: Combine Polygon + IBKR + others
2. **Smart filtering**: Exclude low-volume options, wide spreads
3. **Historical tracking**: Track watchlist changes over time
4. **Alerting**: Notify when symbols added/removed
5. **UI Integration**: Grafana dashboard for watchlist management
6. **Incremental updates**: Only fetch changed symbols, not full list

## Support

For issues or questions:
1. Check logs: `docker logs trading-data-ingestion`
2. Verify Redis: `docker exec trading-redis redis-cli -a <password> scard watchlist`
3. Run discovery manually: `./scripts/sync_optionable_watchlist.sh --dry-run`
4. Review this documentation: `/srv/ai-trading-system/WATCHLIST_MANAGEMENT.md`

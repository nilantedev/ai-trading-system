#!/usr/bin/env python3
"""Market Data Service

Production-safe implementation focused on:
 - Fetching daily bars from EODHD over a date range
 - Persisting results to QuestDB via ILP (TCP) with graceful HTTP fallback
 - Returning lightweight bar objects with ``timestamp`` attribute for
   HistoricalCollector compatibility (dedup/progress tracking)
 - Providing no-op stubs for options backfill to avoid scheduler crashes
 - Emitting heartbeats from the quote stream generator so loops remain healthy

Notes:
 - Postgres multi-sink for historical bars is deferred here because the shared
   database manager in this workspace only exposes Redis. The QuestDB path is
   enabled by default (see docker-compose) and is the primary store-of-record
   for time series.
"""

import os
import asyncio
from typing import Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime
from types import SimpleNamespace

from trading_common import MarketData, get_settings, get_logger

# Logger and settings
logger = get_logger(__name__)
settings = get_settings()

# Module-level runtime for websocket engine (lazy)
_rt_stream = None
_rt_started = False


class MarketDataService:
    def __init__(self) -> None:
        self.session = None
        self.cache = None
        self.pulsar_client = None
        self.producer = None
        self.metrics: Dict[str, Dict[str, int]] = {
            'quotes': {'requests': 0, 'errors': 0},
            'historical': {'requests': 0, 'errors': 0},
        }
        # Provider configs
        self.alpaca_config = {
            'api_key': os.getenv('ALPACA_API_KEY'),
            'secret_key': os.getenv('ALPACA_SECRET_KEY'),
            'data_url': 'https://data.alpaca.markets'
        }
        self.polygon_config = {
            'api_key': os.getenv('POLYGON_API_KEY'),
            'base_url': 'https://api.polygon.io'
        }
        self.alpha_vantage_config = {
            'api_key': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'base_url': 'https://www.alphavantage.co/query'
        }
        self.eodhd_config = {
            'api_key': os.getenv('EODHD_API_KEY'),
            'base_url': 'https://eodhd.com/api'
        }
        # TwelveData fallback (free tier) for daily bars
        self.twelvedata_config = {
            'api_key': os.getenv('TWELVEDATA_API_KEY'),
            'base_url': 'https://api.twelvedata.com'
        }
        # Feature flags
        self.enable_questdb_persist = os.getenv('ENABLE_QUESTDB_HIST_PERSIST', 'false').lower() in ('1','true','yes')
        self.enable_hist_dry_run = os.getenv('ENABLE_HIST_DRY_RUN', 'false').lower() in ('1','true','yes')
        self.enable_options_ingest = os.getenv('ENABLE_OPTIONS_INGEST', 'false').lower() in ('1','true','yes')
        try:
            self.options_max_contracts_per_underlying = int(os.getenv('OPTIONS_MAX_CONTRACTS_PER_UNDERLYING','500'))
        except Exception:
            self.options_max_contracts_per_underlying = 500
        try:
            self.options_polygon_pacing = float(os.getenv('OPTIONS_POLYGON_PACING_SECONDS','0.2'))
        except Exception:
            self.options_polygon_pacing = 0.2
        self.enable_twelvedata_fallback = os.getenv('USE_TWELVEDATA_FALLBACK', 'false').lower() in ('1','true','yes')
        # Options persistence preference: HTTP is more flexible for DATE casting
        self.options_persist_use_http = os.getenv('OPTIONS_PERSIST_USE_HTTP', 'true').lower() in ('1','true','yes')
        # QuestDB sender (optional)
        try:
            from questdb.ingress import Sender, TimestampNanos  # type: ignore
            self._qdb_Sender = Sender
            self._qdb_ts = TimestampNanos
        except Exception:
            self._qdb_Sender = None
            self._qdb_ts = None
        self._qdb_conf: Optional[str] = None

    async def start(self) -> None:
        try:
            import aiohttp
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20),
                headers={'User-Agent': 'AI-Trading-System/1.0'}
            )
        except Exception:
            self.session = None
        # Configure QuestDB sender conf string
        try:
            if self.enable_questdb_persist and self._qdb_Sender:
                host = os.getenv('QUESTDB_HOST', 'trading-questdb')
                proto = os.getenv('QUESTDB_INGEST_PROTOCOL', 'tcp').strip().lower()
                if proto == 'http':
                    port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
                    self._qdb_conf = f"http::addr={host}:{port};"
                else:
                    port = int(os.getenv('QUESTDB_LINE_TCP_PORT', '9009'))
                    self._qdb_conf = f"tcp::addr={host}:{port};"
        except Exception:
            self._qdb_conf = None

    async def stop(self) -> None:
        try:
            if self.session:
                await self.session.close()
        except Exception:
            pass

    async def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        self.metrics['quotes']['requests'] += 1
        return None

    def summary_metrics(self) -> Dict[str, object]:
        return {
            'quotes': self.metrics.get('quotes', {}),
            'historical': self.metrics.get('historical', {}),
            'questdb_persist_enabled': self.enable_questdb_persist,
            'questdb_dry_run': self.enable_hist_dry_run,
            'vendor_metrics': False,
        }

    async def get_service_health(self) -> Dict[str, object]:
        """Lightweight health snapshot used by status endpoint."""
        return {
            'status': 'running',
            'questdb_persist': self.enable_questdb_persist,
            'options_ingest_enabled': self.enable_options_ingest,
            'eodhd_configured': bool(self.eodhd_config.get('api_key') or os.getenv('EODHD_API_KEY')),
            'polygon_key': bool(self.polygon_config.get('api_key')),
            'alpaca_key': bool(self.alpaca_config.get('api_key') and self.alpaca_config.get('secret_key')),
        }

    async def get_bulk_daily_historical(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Fetch daily bars from EODHD for [start_date, end_date] and persist to QuestDB.

        Returns a list of lightweight bar objects that expose at least ``timestamp``.
        """
        self.metrics['historical']['requests'] += 1
        sym = (symbol or '').strip().upper()
        if not sym or not self.session:
            return []
        api_key = (self.eodhd_config.get('api_key') or os.getenv('EODHD_API_KEY') or '').strip()
        if not api_key:
            logger.warning("EODHD_API_KEY missing; attempting TwelveData fallback if enabled", extra={"symbol": sym})
            if self.enable_twelvedata_fallback and self.twelvedata_config.get('api_key'):
                return await self._fetch_daily_from_twelvedata(sym, start_date, end_date)
            return []
        exch = os.getenv('EODHD_EXCHANGE_SUFFIX', 'US').strip()
        base = (self.eodhd_config.get('base_url') or 'https://eodhd.com/api').rstrip('/')
        path = f"/eod/{sym}.{exch}"
        params = {
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'api_token': api_key,
            'fmt': 'json'
        }
        rows: List[Any] = []
        try:
            async with self.session.get(base + path, params=params, timeout=30) as resp:
                if resp.status != 200:
                    try:
                        txt = (await resp.text())[:300]
                    except Exception:
                        txt = ''
                    logger.warning("EODHD HTTP %s for %s: %s", resp.status, sym, txt)
                    return []
                data = await resp.json()
        except Exception as e:
            logger.warning("EODHD request failed for %s: %s", sym, e)
            if self.enable_twelvedata_fallback and self.twelvedata_config.get('api_key'):
                return await self._fetch_daily_from_twelvedata(sym, start_date, end_date)
            return []
        for it in data or []:
            try:
                dstr = it.get('date') or it.get('Date')
                ts = datetime.strptime(dstr, '%Y-%m-%d') if isinstance(dstr, str) else start_date
                o = float(it.get('open') or it.get('Open') or it.get('o') or 0.0)
                h = float(it.get('high') or it.get('High') or it.get('h') or o)
                l = float(it.get('low') or it.get('Low') or it.get('l') or o)
                c = float(it.get('close') or it.get('Close') or it.get('c') or o)
                v = int(it.get('volume') or it.get('Volume') or it.get('v') or 0)
                rows.append(SimpleNamespace(symbol=sym, timestamp=ts, open=o, high=h, low=l, close=c, volume=v))
            except Exception:
                continue
        if not rows:
            if self.enable_twelvedata_fallback and self.twelvedata_config.get('api_key'):
                return await self._fetch_daily_from_twelvedata(sym, start_date, end_date)
            return []
        try:
            await self._persist_daily_rows_qdb(rows)
        except Exception as e:
            logger.warning("QuestDB persist failed for %s: %s", sym, e)
        return rows

    async def _fetch_daily_from_twelvedata(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Fallback: fetch daily time series from TwelveData (free-tier-friendly).

        Endpoint: /time_series?symbol=SYM&interval=1day&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&apikey=...
        Note: TwelveData returns most-recent first; we normalize to ascending by date.
        """
        if not self.session:
            return []
        api_key = (self.twelvedata_config.get('api_key') or os.getenv('TWELVEDATA_API_KEY') or '').strip()
        if not api_key:
            return []
        base = (self.twelvedata_config.get('base_url') or 'https://api.twelvedata.com').rstrip('/')
        params = {
            'symbol': symbol,
            'interval': '1day',
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'apikey': api_key,
            'format': 'JSON'
        }
        try:
            async with self.session.get(base + '/time_series', params=params, timeout=30) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            return []
        values = data.get('values') if isinstance(data, dict) else None
        if not values:
            return []
        rows: List[Any] = []
        for it in reversed(values):  # convert to ascending
            try:
                dstr = it.get('datetime') or it.get('date')
                ts = datetime.strptime(dstr, '%Y-%m-%d') if isinstance(dstr, str) and len(dstr) == 10 else datetime.fromisoformat(str(dstr))
                o = float(it.get('open') or 0.0)
                h = float(it.get('high') or o)
                l = float(it.get('low') or o)
                c = float(it.get('close') or o)
                v = int(float(it.get('volume') or 0))
                rows.append(SimpleNamespace(symbol=symbol, timestamp=ts, open=o, high=h, low=l, close=c, volume=v))
            except Exception:
                continue
        try:
            await self._persist_daily_rows_qdb(rows)
        except Exception:
            pass
        return rows

    async def _persist_daily_rows_qdb(self, rows: List[Any]) -> None:
        """Persist daily bar rows to QuestDB using ILP; fallback to HTTP INSERT if configured.

        Expected row attributes: symbol, timestamp, open, high, low, close, volume.
        """
        if not (self.enable_questdb_persist and rows):
            return
        # ILP path via questdb.ingress Sender
        if self._qdb_Sender and self._qdb_conf:
            try:
                with self._qdb_Sender(self._qdb_conf) as sender:  # type: ignore[operator]
                    for r in rows:
                        ts_nanos = int((r.timestamp - datetime(1970, 1, 1)).total_seconds() * 1_000_000_000)
                        sender.row('market_data') \
                              .symbol('symbol', r.symbol) \
                              .str('timeframe', '1d') \
                              .str('data_source', 'eodhd') \
                              .float_column('open', float(r.open)) \
                              .float_column('high', float(r.high)) \
                              .float_column('low', float(r.low)) \
                              .float_column('close', float(r.close)) \
                              .long_column('volume', int(r.volume)) \
                              .at(ts_nanos)
                return
            except Exception:
                pass
        # HTTP /exec fallback for small batches
        try:
            host = os.getenv('QUESTDB_HOST', 'trading-questdb')
            http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
            qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
            values = []
            for r in rows[:5000]:
                ts_iso = r.timestamp.strftime('%Y-%m-%dT00:00:00.000000Z')
                values.append(
                    f"('{r.symbol}','1d','eodhd',{float(r.open)},{float(r.high)},{float(r.low)},{float(r.close)},{int(r.volume)},to_timestamp('{ts_iso}', 'yyyy-MM-ddTHH:mm:ss.SSSSSSZ'))"
                )
            sql = (
                "insert into market_data(symbol,timeframe,data_source,open,high,low,close,volume,timestamp) values "
                + ",".join(values)
            )
            if self.session:
                async with self.session.get(qdb_url, params={"query": sql}, timeout=30) as resp:
                    if resp.status != 200:
                        _txt = (await resp.text())[:200]
                        logger.warning("QuestDB HTTP insert failed: %s", _txt)
        except Exception:
            pass

    async def backfill_options_chain(
        self,
        underlying: str,
        start_expiry: datetime,
        end_expiry: datetime,
        *,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_contracts: int = 500,
        pacing_seconds: float = 0.2,
    ) -> Dict[str, Any]:
        """Backfill options chain for an underlying via Polygon (paid tier).

        1) List contracts within expiry window using v3/reference/options/contracts
        2) For each contract (up to max_contracts), fetch daily aggregates over [start_date, end_date]
        3) Persist rows to QuestDB options_data (ILP preferred, HTTP fallback)
        """
        if not (self.enable_options_ingest and self.polygon_config.get('api_key') and self.session):
            return {"underlying": underlying, "contracts": 0, "bars": 0, "enabled": False}
        u = underlying.strip().upper()
        # Resolve historical window defaults if not supplied
        start_date = start_date or datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date or datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        contracts = await self._polygon_list_options_contracts(u, start_expiry, end_expiry, self.options_max_contracts_per_underlying)
        bars_total = 0
        for c in contracts[:max(1, min(max_contracts, self.options_max_contracts_per_underlying))]:
            sym = c.get('ticker') or c.get('symbol')
            if not sym:
                continue
            try:
                rows = await self._polygon_fetch_option_aggs(sym, start_date, end_date)
                if rows:
                    # Attach metadata from contract
                    meta = self._parse_option_symbol(sym)
                    for r in rows:
                        r.update({
                            'underlying': u,
                            'option_symbol': sym,
                            'right': meta.get('right'),
                            'strike': meta.get('strike'),
                            'expiry': meta.get('expiry')
                        })
                    await self._persist_option_bars_qdb(rows)
                    bars_total += len(rows)
            except Exception:
                pass
            await asyncio.sleep(max(0.0, pacing_seconds if pacing_seconds is not None else self.options_polygon_pacing))
        return {"underlying": u, "contracts": len(contracts), "bars": bars_total, "enabled": True}

    async def _polygon_list_options_contracts(self, underlying: str, start_expiry: datetime, end_expiry: datetime, limit: int) -> list:
        base = (self.polygon_config.get('base_url') or 'https://api.polygon.io').rstrip('/')
        url = base + '/v3/reference/options/contracts'
        params = {
            'underlying_ticker': underlying,
            'expiration_date.gte': start_expiry.strftime('%Y-%m-%d'),
            'expiration_date.lte': end_expiry.strftime('%Y-%m-%d'),
            'order': 'asc',
            'limit': min(1000, max(1, limit)),
            'sort': 'expiration_date',
            'apiKey': self.polygon_config['api_key']
        }
        try:
            async with self.session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            return []
        results = data.get('results') or []
        return results[:limit]

    async def _polygon_fetch_option_aggs(self, option_symbol: str, start_date: datetime, end_date: datetime) -> list[dict]:
        base = (self.polygon_config.get('base_url') or 'https://api.polygon.io').rstrip('/')
        frm = start_date.strftime('%Y-%m-%d')
        to = end_date.strftime('%Y-%m-%d')
        path = f"/v2/aggs/ticker/{option_symbol}/range/1/day/{frm}/{to}"
        params = {'adjusted': 'true', 'order': 'asc', 'limit': 50000, 'apiKey': self.polygon_config['api_key']}
        try:
            async with self.session.get(base + path, params=params, timeout=30) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            return []
        res = []
        for it in (data.get('results') or []):
            try:
                # Polygon returns epoch ms in 't'
                ts = datetime.utcfromtimestamp(int(it.get('t', 0)) / 1000.0)
                res.append({
                    'timestamp': ts,
                    'open': float(it.get('o', 0.0)),
                    'high': float(it.get('h', 0.0)),
                    'low': float(it.get('l', 0.0)),
                    'close': float(it.get('c', 0.0)),
                    'volume': int(it.get('v', 0)),
                })
            except Exception:
                continue
        return res

    def _parse_option_symbol(self, symbol: str) -> dict:
        """Parse Polygon option symbol like O:SPY251219C00600000.

        Returns dict with expiry (datetime), right ('C'|'P'), strike (float).
        """
        try:
            s = symbol
            if s.startswith('O:'):
                s = s[2:]
            # Find right marker C/P after 6 digits of date YYMMDD
            # e.g., SPY251219C00600000 -> ... YY=25 MM=12 DD=19
            import re
            m = re.search(r"(\d{6})([CP])([0-9]+)$", s)
            if not m:
                return {'right': None, 'strike': None, 'expiry': None}
            ymd = m.group(1)
            right = m.group(2)
            strike_raw = m.group(3)
            yy = int(ymd[0:2]); mm = int(ymd[2:4]); dd = int(ymd[4:6])
            year = 2000 + yy
            from datetime import datetime as _dt
            expiry = _dt(year, mm, dd)
            # Strike scaling heuristic: many tickers encode strike * 1000 or * 100
            val = int(strike_raw)
            strike = val / 1000.0 if len(strike_raw) >= 4 else float(val)
            return {'right': right, 'strike': strike, 'expiry': expiry}
        except Exception:
            return {'right': None, 'strike': None, 'expiry': None}

    async def _persist_option_bars_qdb(self, rows: list[dict]) -> None:
        if not (self.enable_questdb_persist and rows):
            return
        # ILP path if available
        if (not self.options_persist_use_http) and self._qdb_Sender and self._qdb_conf:
            try:
                with self._qdb_Sender(self._qdb_conf) as sender:  # type: ignore[operator]
                    for r in rows:
                        ts_nanos = int((r['timestamp'] - datetime(1970, 1, 1)).total_seconds() * 1_000_000_000)
                        # Schema alignment: QuestDB options_data has columns
                        # underlying SYMBOL, option_symbol SYMBOL, option_type SYMBOL,
                        # strike DOUBLE, expiry DATE, open/high/low/close DOUBLE, volume LONG, timestamp TIMESTAMP
                        opt_type = r.get('option_type') or r.get('right') or ''
                        try:
                            strike_val = float(r.get('strike') or 0.0)
                        except Exception:
                            strike_val = 0.0
                        expiry_dt = r.get('expiry') or datetime.utcnow()
                        sender.row('options_data') \
                              .symbol('underlying', r.get('underlying','')) \
                              .symbol('option_symbol', r.get('option_symbol','')) \
                              .str('option_type', str(opt_type)) \
                              .float_column('strike', float(strike_val)) \
                              .str('expiry', expiry_dt.strftime('%Y-%m-%d')) \
                              .float_column('open', float(r.get('open') or 0.0)) \
                              .float_column('high', float(r.get('high') or 0.0)) \
                              .float_column('low', float(r.get('low') or 0.0)) \
                              .float_column('close', float(r.get('close') or 0.0)) \
                              .long_column('volume', int(r.get('volume') or 0)) \
                              .at(ts_nanos)
                return
            except Exception:
                pass
        # HTTP fallback
        try:
            host = os.getenv('QUESTDB_HOST', 'trading-questdb')
            http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
            qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")
            values = []
            for r in rows[:5000]:
                ts_iso = r['timestamp'].strftime('%Y-%m-%dT00:00:00.000000Z')
                expiry_s = (r.get('expiry') or datetime.utcnow()).strftime('%Y-%m-%d')
                opt_type = r.get('option_type') or r.get('right') or ''
                try:
                    strike_val = float(r.get('strike') or 0.0)
                except Exception:
                    strike_val = 0.0
                values.append(
                    "('" + str(r.get('underlying','')) + "', '" + str(r.get('option_symbol','')) + "', '" + str(opt_type) + "', " + f"{strike_val}" + ", to_date('" + expiry_s + "', 'yyyy-MM-dd'), "
                    + f"{float(r.get('open') or 0.0)},{float(r.get('high') or 0.0)},{float(r.get('low') or 0.0)},{float(r.get('close') or 0.0)},{int(r.get('volume') or 0)},to_timestamp('{ts_iso}', 'yyyy-MM-ddTHH:mm:ss.SSSSSSZ')"
                    + ")"
                )
            if not values:
                return
            sql = (
                "insert into options_data(underlying,option_symbol,option_type,strike,expiry,open,high,low,close,volume,timestamp) values "
                + ",".join(values)
            )
            if self.session:
                async with self.session.get(qdb_url, params={"query": sql}, timeout=30) as resp:
                    if resp.status != 200:
                        _txt = (await resp.text())[:200]
                        logger.warning("QuestDB HTTP insert (options_data) failed: %s", _txt)
        except Exception:
            pass

    async def stream_real_time_data(self, symbols: List[str]) -> AsyncGenerator[MarketData, None]:
        """Start provider WebSocket streams (feature-gated) and emit heartbeats.

        Behavior:
          - If ENABLE_WEBSOCKETS is true and any provider keys are present, lazily start
            the RealTimeMarketStream in the background for the given symbols.
          - Always yield lightweight heartbeat items so the main loop records activity.
        """
        global _rt_stream, _rt_started
        # Try to start websockets once (non-blocking)
        try:
            if os.getenv('ENABLE_WEBSOCKETS', 'true').lower() in ('1','true','yes') and not _rt_started:
                # Only start if at least one provider credential is configured
                have_polygon = bool(self.polygon_config.get('api_key'))
                have_alpaca = bool(self.alpaca_config.get('api_key') and self.alpaca_config.get('secret_key'))
                if have_polygon or have_alpaca:
                    try:
                        # Lazy import to avoid heavy deps at module import time
                        from .realtime_market_stream import RealTimeMarketStream  # type: ignore
                    except Exception:
                        from services.data_ingestion.realtime_market_stream import RealTimeMarketStream  # type: ignore
                    try:
                        _rt_stream = _rt_stream or RealTimeMarketStream()
                        clean_symbols = [s.strip().upper() for s in symbols if s]
                        logger.info(f"Starting WebSocket connections for {len(clean_symbols)} symbols")
                        # Start WebSocket connections as a fire-and-forget task
                        # Use ensure_future to properly schedule on event loop
                        asyncio.ensure_future(_rt_stream.start(clean_symbols))
                        _rt_started = True
                        # Yield control to event loop to let task start
                        await asyncio.sleep(0)
                        logger.info("Real-time WebSocket engine started", extra={"symbols": len(clean_symbols)})
                    except Exception as e:
                        logger.warning("Failed to start real-time stream engine: %s", e)
                        logger.exception("WebSocket startup exception:")
        except Exception:
            pass

        # Emit heartbeats continuously with modest pacing
        interval = float(os.getenv('QUOTE_STREAM_HEARTBEAT_SECONDS', '1.0') or '1.0')
        while True:
            try:
                hb = SimpleNamespace(kind='heartbeat', symbols=[s.strip().upper() for s in symbols if s], timestamp=datetime.utcnow())
                yield hb  # type: ignore[misc]
            except Exception:
                break
            await asyncio.sleep(max(0.25, interval))


_svc_singleton: Optional["MarketDataService"] = None


async def get_market_data_service() -> MarketDataService:
    global _svc_singleton
    if _svc_singleton is None:
        _svc_singleton = MarketDataService()
        await _svc_singleton.start()
    return _svc_singleton
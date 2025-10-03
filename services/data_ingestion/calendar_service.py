#!/usr/bin/env python3
"""Calendar Service - Calendar collectors for earnings, IPOs, splits, dividends.

Supports providers:
    - Alpha Vantage (earnings calendar, IPO calendar; CSV endpoints)
    - EODHD (earnings, IPO, splits, dividends; JSON endpoints)

Feature-gated via env:
    Provider selection:
        - CALENDAR_PROVIDER = alphavantage | eodhd (default autodetect)
    Alpha Vantage:
        - ALPHAVANTAGE_API_KEY (required when CALENDAR_PROVIDER=alphavantage)
        - AV_EARNINGS_CALENDAR_HORIZON = 3month|6month|12month (default 3month)
    EODHD:
        - EODHD_CALENDAR_ENABLED (default true when key present)
        - EODHD_API_KEY (required)

Persists to QuestDB tables via HTTP /exec:
  - earnings_calendar
  - ipo_calendar
  - splits_calendar
  - dividends_calendar (optional)

All tables are created if missing, with TIMESTAMP(timestamp) PARTITION BY DAY.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import aiohttp
import csv
import io

from trading_common import get_logger

logger = get_logger(__name__)


class CalendarService:
    def __init__(self):
        # Provider selection
        self.provider = os.getenv('CALENDAR_PROVIDER', '').strip().lower()
        # Alpha Vantage config
        self.av_base_url = os.getenv('ALPHAVANTAGE_BASE_URL', 'https://www.alphavantage.co/query')
        # Support both env spellings: ALPHAVANTAGE_API_KEY (preferred) and ALPHA_VANTAGE_API_KEY (legacy)
        self.av_api_key = (
            os.getenv('ALPHAVANTAGE_API_KEY', '').strip()
            or os.getenv('ALPHA_VANTAGE_API_KEY', '').strip()
        )
        self.av_horizon = os.getenv('AV_EARNINGS_CALENDAR_HORIZON', '3month').strip() or '3month'
        # EODHD config
        self.eodhd_base_url = os.getenv('EODHD_BASE_URL', 'https://eodhd.com/api')
        self.eodhd_api_key = os.getenv('EODHD_API_KEY', '').strip()
        self.eodhd_enabled_flag = (os.getenv('EODHD_CALENDAR_ENABLED', 'true').lower() in ('1','true','yes'))
        # Compute availability flags
        self.av_enabled = bool(self.av_api_key)
        self.eodhd_enabled = self.eodhd_enabled_flag and bool(self.eodhd_api_key)
        # Determine effective provider if not explicitly set
        if not self.provider:
            if self.av_enabled:
                self.provider = 'alphavantage'
            elif self.eodhd_enabled:
                self.provider = 'eodhd'
        # If provider explicitly set but unavailable, fail over to the other if possible
        if self.provider == 'alphavantage' and not self.av_enabled and self.eodhd_enabled:
            # Fall back to EODHD silently to avoid stalling calendar ingestion
            self.provider = 'eodhd'
        elif self.provider == 'eodhd' and not self.eodhd_enabled and self.av_enabled:
            self.provider = 'alphavantage'
        # Service-wide enabled if any provider is available
        self.enabled = self.av_enabled or self.eodhd_enabled
        self.session: Optional[aiohttp.ClientSession] = None
        # QuestDB HTTP URL used for DDL/DML
        host = os.getenv('QUESTDB_HOST', 'trading-questdb')
        http_port = int(os.getenv('QUESTDB_HTTP_PORT', '9000'))
        self.qdb_url = os.getenv('QUESTDB_HTTP_URL', f"http://{host}:{http_port}/exec")

    async def start(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        # Ensure schemas upfront (best-effort)
        try:
            await self._ensure_schemas()
        except Exception as e:
            logger.warning("Calendar schema ensure failed (continuing): %s", e)

    async def stop(self):
        try:
            if self.session:
                await self.session.close()
        except Exception:
            pass

    # -------------------- QuestDB helpers -------------------- #
    async def _exec(self, sql: str, *, timeout: float = 20.0) -> dict:
        if not self.session:
            await self.start()
        assert self.session
        async with self.session.get(self.qdb_url, params={"query": sql}, timeout=timeout) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"QuestDB HTTP {resp.status}: {txt[:200]}")
            return await resp.json()

    async def _ensure_schemas(self) -> None:
        stmts = [
            # earnings_calendar
            ("create table if not exists earnings_calendar ("
             "symbol symbol, date date, time string, eps_estimate double, eps_actual double, "
             "revenue_estimate double, revenue_actual double, currency symbol, updated timestamp, "
             "timestamp timestamp) timestamp(timestamp) PARTITION BY DAY"),
            # ipo_calendar
            ("create table if not exists ipo_calendar ("
             "symbol symbol, company string, date date, exchange string, price_range string, "
             "shares long, updated timestamp, timestamp timestamp) timestamp(timestamp) PARTITION BY DAY"),
            # splits_calendar
            ("create table if not exists splits_calendar ("
             "symbol symbol, date date, numerator double, denominator double, ratio double, updated timestamp, "
             "timestamp timestamp) timestamp(timestamp) PARTITION BY DAY"),
            # dividends_calendar (optional)
            ("create table if not exists dividends_calendar ("
             "symbol symbol, ex_date date, declaration_date date, record_date date, payment_date date, "
             "dividend double, currency symbol, updated timestamp, timestamp timestamp) timestamp(timestamp) PARTITION BY DAY"),
        ]
        for sql in stmts:
            try:
                await self._exec(sql)
            except Exception:
                # Continue; individual table may fail on older QuestDB but others still apply
                continue

    # -------------------- Provider requests -------------------- #
    async def _get_json_eodhd(self, path: str, params: dict) -> list:
        if not self.session:
            await self.start()
        assert self.session
        url = f"{self.eodhd_base_url.rstrip('/')}/{path.lstrip('/')}"
        p = dict(params)
        p['api_token'] = self.eodhd_api_key
        p.setdefault('fmt', 'json')
        async with self.session.get(url, params=p) as resp:
            if resp.status != 200:
                try:
                    txt = (await resp.text())[:300]
                except Exception:
                    txt = ''
                raise RuntimeError(f"EODHD HTTP {resp.status}: {txt}")
            try:
                data = await resp.json()
            except Exception:
                data = []
        # EODHD returns objects or lists; normalize to list
        if isinstance(data, dict):
            return data.get('data') or data.get('results') or []
        return data or []

    async def _get_csv_alphavantage(self, function: str, params: dict) -> List[Dict[str, str]]:
        """Fetch CSV from Alpha Vantage and parse into list of dict rows."""
        if not self.session:
            await self.start()
        assert self.session
        p = dict(params)
        p['function'] = function
        p['apikey'] = self.av_api_key
        async with self.session.get(self.av_base_url, params=p) as resp:
            txt = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"AlphaVantage HTTP {resp.status}: {txt[:200]}")
            # Some AV error payloads are JSON; detect quickly
            if txt.strip().startswith('{'):
                # return empty to be safe
                return []
            # Parse CSV
            try:
                reader = csv.DictReader(io.StringIO(txt))
                return [ { (k or '').strip(): (v or '').strip() for (k,v) in row.items() } for row in reader ]
            except Exception:
                return []

    # -------------------- Collectors -------------------- #
    # -------------------- EODHD collectors -------------------- #
    async def collect_earnings_range(self, start: datetime, end: datetime) -> int:
        def _esc(s: str) -> str:
            try:
                return str(s).replace("'", "''")
            except Exception:
                return str(s)
        total = 0
        day = start.date()
        end_day = end.date()
        while day <= end_day:
            try:
                items = await self._get_json_eodhd('calendar/earnings', {
                    'from': day.strftime('%Y-%m-%d'),
                    'to': day.strftime('%Y-%m-%d')
                })
            except Exception:
                items = []
            rows = []
            for it in items:
                try:
                    sym = str(it.get('code') or it.get('symbol') or '').upper()
                    date_s = str(it.get('date') or it.get('report_date') or day.isoformat())
                    time_s = _esc(str(it.get('time') or ''))
                    eps_est = float(it.get('epsEstimate')) if it.get('epsEstimate') is not None else 'NaN'
                    eps_act = float(it.get('epsActual')) if it.get('epsActual') is not None else 'NaN'
                    rev_est = float(it.get('revenueEstimate')) if it.get('revenueEstimate') is not None else 'NaN'
                    rev_act = float(it.get('revenueActual')) if it.get('revenueActual') is not None else 'NaN'
                    ccy = _esc(str(it.get('currency') or ''))
                    upd = _esc(str(it.get('updated_at') or date_s))
                    # Insert; cast date to DATE, timestamp use updated or date
                    sql = (
                        "insert into earnings_calendar (symbol,date,time,eps_estimate,eps_actual,revenue_estimate,revenue_actual,currency,updated,timestamp) values ("
                        f"'{sym}', to_date('{date_s}','yyyy-MM-dd'), '{time_s}', {eps_est}, {eps_act}, {rev_est}, {rev_act}, '{ccy}', '{upd}', '{date_s}T00:00:00Z')"
                    )
                    rows.append(sql)
                except Exception:
                    continue
            if rows:
                try:
                    # Batch by concatenating statements separated by ';'
                    await self._exec(';'.join(rows))
                    total += len(rows)
                except Exception:
                    pass
            # pacing between days
            try:
                await asyncio.sleep(float(os.getenv('EODHD_PACING_SECONDS','0.2')))
            except Exception:
                pass
            day = day + timedelta(days=1)
        return total

    async def collect_ipo_range(self, start: datetime, end: datetime) -> int:
        def _esc(s: str) -> str:
            try:
                return str(s).replace("'", "''")
            except Exception:
                return str(s)
        total = 0
        day = start.date()
        end_day = end.date()
        while day <= end_day:
            try:
                items = await self._get_json_eodhd('calendar/ipos', {
                    'from': day.strftime('%Y-%m-%d'),
                    'to': day.strftime('%Y-%m-%d')
                })
            except Exception:
                items = []
            rows = []
            for it in items:
                try:
                    sym = str(it.get('code') or it.get('symbol') or '').upper()
                    company = _esc(str(it.get('company') or it.get('name') or ''))
                    date_s = str(it.get('date') or day.isoformat())
                    exch = _esc(str(it.get('exchange') or ''))
                    pr = _esc(str(it.get('priceRange') or it.get('price_range') or ''))
                    shares = int(float(it.get('shares') or 0))
                    upd = _esc(str(it.get('updated_at') or date_s))
                    sql = (
                        "insert into ipo_calendar (symbol,company,date,exchange,price_range,shares,updated,timestamp) values ("
                        f"'{sym}','{company}',to_date('{date_s}','yyyy-MM-dd'),'{exch}','{pr}',{shares},'{upd}','{date_s}T00:00:00Z')"
                    )
                    rows.append(sql)
                except Exception:
                    continue
            if rows:
                try:
                    await self._exec(';'.join(rows))
                    total += len(rows)
                except Exception:
                    pass
            try:
                await asyncio.sleep(float(os.getenv('EODHD_PACING_SECONDS','0.2')))
            except Exception:
                pass
            day = day + timedelta(days=1)
        return total

    async def collect_splits_range(self, start: datetime, end: datetime) -> int:
        def _esc(s: str) -> str:
            try:
                return str(s).replace("'", "''")
            except Exception:
                return str(s)
        total = 0
        day = start.date()
        end_day = end.date()
        while day <= end_day:
            try:
                items = await self._get_json_eodhd('calendar/splits', {
                    'from': day.strftime('%Y-%m-%d'),
                    'to': day.strftime('%Y-%m-%d')
                })
            except Exception:
                items = []
            rows = []
            for it in items:
                try:
                    sym = _esc(str(it.get('code') or it.get('symbol') or '')).upper()
                    date_s = str(it.get('date') or day.isoformat())
                    num = float(it.get('numerator') or it.get('toFactor') or 0)
                    den = float(it.get('denominator') or it.get('forFactor') or 1)
                    ratio = num/den if den else (num or 0)
                    upd = _esc(str(it.get('updated_at') or date_s))
                    sql = (
                        "insert into splits_calendar (symbol,date,numerator,denominator,ratio,updated,timestamp) values ("
                        f"'{sym}', to_date('{date_s}','yyyy-MM-dd'), {num}, {den}, {ratio}, '{upd}', '{date_s}T00:00:00Z')"
                    )
                    rows.append(sql)
                except Exception:
                    continue
            if rows:
                try:
                    await self._exec(';'.join(rows))
                    total += len(rows)
                except Exception:
                    pass
            try:
                await asyncio.sleep(float(os.getenv('EODHD_PACING_SECONDS','0.2')))
            except Exception:
                pass
            day = day + timedelta(days=1)
        return total

    async def collect_dividends_range(self, start: datetime, end: datetime) -> int:
        """Optional dividends calendar collection (best-effort)."""
        def _esc(s: str) -> str:
            try:
                return str(s).replace("'", "''")
            except Exception:
                return str(s)
        total = 0
        day = start.date()
        end_day = end.date()
        while day <= end_day:
            try:
                items = await self._get_json_eodhd('calendar/dividends', {
                    'from': day.strftime('%Y-%m-%d'),
                    'to': day.strftime('%Y-%m-%d')
                })
            except Exception:
                items = []
            rows = []
            for it in items:
                try:
                    sym = _esc(str(it.get('code') or it.get('symbol') or '')).upper()
                    exd = _esc(str(it.get('exDate') or it.get('ex_date') or day.isoformat()))
                    dec = _esc(str(it.get('declarationDate') or it.get('declaration_date') or ''))
                    rec = _esc(str(it.get('recordDate') or it.get('record_date') or ''))
                    pay = _esc(str(it.get('paymentDate') or it.get('payment_date') or ''))
                    div = float(it.get('dividend') or 0)
                    ccy = _esc(str(it.get('currency') or ''))
                    upd = _esc(str(it.get('updated_at') or exd))
                    # Build SQL fragments for optional dates to avoid backslashes in f-strings
                    dec_sql = f"to_date('{dec}','yyyy-MM-dd')" if dec else 'NaN'
                    rec_sql = f"to_date('{rec}','yyyy-MM-dd')" if rec else 'NaN'
                    pay_sql = f"to_date('{pay}','yyyy-MM-dd')" if pay else 'NaN'
                    sql = (
                        "insert into dividends_calendar (symbol,ex_date,declaration_date,record_date,payment_date,dividend,currency,updated,timestamp) values ("
                        f"'{sym}', to_date('{exd}','yyyy-MM-dd'), {dec_sql}, {rec_sql}, {pay_sql}, {div}, '{ccy}', '{upd}', '{exd}T00:00:00Z')"
                    )
                    rows.append(sql)
                except Exception:
                    continue
            if rows:
                try:
                    await self._exec(';'.join(rows))
                    total += len(rows)
                except Exception:
                    pass
            try:
                await asyncio.sleep(float(os.getenv('EODHD_PACING_SECONDS','0.25')))
            except Exception:
                pass
            day = day + timedelta(days=1)
        return total

    # -------------------- EODHD per-symbol feeds (Splits/Dividends) -------------------- #
    async def collect_eodhd_splits_symbol(self, symbol: str, start: datetime, end: datetime) -> int:
        """Collect splits for a single symbol using EODHD splits feed (non-calendar).

        Endpoint (typical): /api/splits/{SYMBOL}?from=YYYY-MM-DD&to=YYYY-MM-DD
        """
        if not self.eodhd_api_key:
            return 0
        sym = (symbol or '').strip().upper()
        if not sym:
            return 0
        def _esc(s: str) -> str:
            try:
                return str(s).replace("'", "''")
            except Exception:
                return str(s)
        total = 0
        items = []
        # Attempt plain symbol first
        try:
            path = f"splits/{sym}"
            items = await self._get_json_eodhd(path, {
                'from': start.strftime('%Y-%m-%d'),
                'to': end.strftime('%Y-%m-%d')
            })
        except Exception:
            items = []
        # Fallback to default exchange-qualified symbol e.g., AAPL.US
        if not items and '.' not in sym:
            try:
                ex = os.getenv('EODHD_DEFAULT_EXCHANGE', 'US').strip().upper() or 'US'
                sym_ex = f"{sym}.{ex}"
                path = f"splits/{sym_ex}"
                items = await self._get_json_eodhd(path, {
                    'from': start.strftime('%Y-%m-%d'),
                    'to': end.strftime('%Y-%m-%d')
                })
            except Exception:
                items = []
        rows = []
        for it in items or []:
            try:
                date_s = str(it.get('date') or it.get('split_date') or '')
                if not date_s:
                    continue
                num = float(it.get('numerator') or it.get('toFactor') or it.get('to') or 0)
                den = float(it.get('denominator') or it.get('forFactor') or it.get('from') or 1)
                ratio = (num/den) if den else (num or 0)
                upd = _esc(str(it.get('updated_at') or date_s))
                rows.append(
                    "insert into splits_calendar (symbol,date,numerator,denominator,ratio,updated,timestamp) values ("
                    f"'{sym}', to_date('{date_s}','yyyy-MM-dd'), {num}, {den}, {ratio}, '{upd}', '{date_s}T00:00:00Z')"
                )
            except Exception:
                continue
        if rows:
            try:
                await self._exec(';'.join(rows))
                total = len(rows)
            except Exception:
                total = 0
        return total

    async def collect_eodhd_dividends_symbol(self, symbol: str, start: datetime, end: datetime) -> int:
        """Collect dividends for a single symbol using EODHD dividends feed (non-calendar).

        Endpoint (typical): /api/dividends/{SYMBOL}?from=YYYY-MM-DD&to=YYYY-MM-DD
        """
        if not self.eodhd_api_key:
            return 0
        sym = (symbol or '').strip().upper()
        if not sym:
            return 0
        def _esc(s: str) -> str:
            try:
                return str(s).replace("'", "''")
            except Exception:
                return str(s)
        total = 0
        items = []
        try:
            path = f"dividends/{sym}"
            items = await self._get_json_eodhd(path, {
                'from': start.strftime('%Y-%m-%d'),
                'to': end.strftime('%Y-%m-%d')
            })
        except Exception:
            items = []
        if not items and '.' not in sym:
            try:
                ex = os.getenv('EODHD_DEFAULT_EXCHANGE', 'US').strip().upper() or 'US'
                sym_ex = f"{sym}.{ex}"
                path = f"dividends/{sym_ex}"
                items = await self._get_json_eodhd(path, {
                    'from': start.strftime('%Y-%m-%d'),
                    'to': end.strftime('%Y-%m-%d')
                })
            except Exception:
                items = []
        rows = []
        for it in items or []:
            try:
                exd = _esc(str(it.get('exDate') or it.get('date') or ''))
                if not exd:
                    continue
                dec = _esc(str(it.get('declarationDate') or it.get('declaration_date') or ''))
                rec = _esc(str(it.get('recordDate') or it.get('record_date') or ''))
                pay = _esc(str(it.get('paymentDate') or it.get('payment_date') or ''))
                div = float(it.get('dividend') or it.get('value') or 0)
                ccy = _esc(str(it.get('currency') or ''))
                upd = _esc(str(it.get('updated_at') or exd))
                dec_sql = f"to_date('{dec}','yyyy-MM-dd')" if dec else 'NaN'
                rec_sql = f"to_date('{rec}','yyyy-MM-dd')" if rec else 'NaN'
                pay_sql = f"to_date('{pay}','yyyy-MM-dd')" if pay else 'NaN'
                rows.append(
                    "insert into dividends_calendar (symbol,ex_date,declaration_date,record_date,payment_date,dividend,currency,updated,timestamp) values ("
                    f"'{sym}', to_date('{exd}','yyyy-MM-dd'), {dec_sql}, {rec_sql}, {pay_sql}, {div}, '{ccy}', '{upd}', '{exd}T00:00:00Z')"
                )
            except Exception:
                continue
        if rows:
            try:
                await self._exec(';'.join(rows))
                total = len(rows)
            except Exception:
                total = 0
        return total

    # -------------------- Alpha Vantage collectors -------------------- #
    async def collect_av_earnings_upcoming(self, *, symbol: Optional[str] = None, horizon: Optional[str] = None) -> int:
        """Collect upcoming earnings using Alpha Vantage EARNINGS_CALENDAR (CSV). Earnings are future-only.

        Note: AV does not include revenue fields in calendar; fill with NaN. Time/currency often absent.
        """
        if self.provider != 'alphavantage' or not self.av_api_key:
            return 0
        hz = (horizon or self.av_horizon or '3month').lower()
        if hz not in ('3month','6month','12month'):
            hz = '3month'

        def _esc(s: str) -> str:
            try:
                return str(s).replace("'", "''")
            except Exception:
                return str(s)

        params: Dict[str, str] = { 'horizon': hz }
        if symbol:
            params['symbol'] = symbol.strip().upper()
        rows = await self._get_csv_alphavantage('EARNINGS_CALENDAR', params)
        if not rows:
            return 0
        stmts: List[str] = []
        for r in rows:
            try:
                sym = _esc(r.get('symbol') or r.get('Symbol') or '')
                if not sym:
                    continue
                date_s = (r.get('reportDate') or r.get('ReportDate') or '')
                if not date_s:
                    continue
                time_s = _esc(r.get('time') or '')
                # EPS fields may be empty strings
                def _to_float(v):
                    try:
                        if v is None:
                            return 'NaN'
                        s = str(v).strip()
                        if s == '' or s.lower() in ('na','nan','null'):
                            return 'NaN'
                        return float(s)
                    except Exception:
                        return 'NaN'
                eps_est = _to_float(r.get('epsEstimate') or r.get('EpsEstimate'))
                eps_act = _to_float(r.get('epsActual') or r.get('EpsActual'))
                # Revenue not present in AV earnings calendar (leave NaN)
                rev_est = 'NaN'
                rev_act = 'NaN'
                currency = _esc(r.get('currency') or '')
                updated = _esc(r.get('updated') or date_s)
                stmts.append(
                    "insert into earnings_calendar (symbol,date,time,eps_estimate,eps_actual,revenue_estimate,revenue_actual,currency,updated,timestamp) values ("
                    f"'{sym}', to_date('{date_s}','yyyy-MM-dd'), '{time_s}', {eps_est}, {eps_act}, {rev_est}, {rev_act}, '{currency}', '{updated}', '{date_s}T00:00:00Z')"
                )
            except Exception:
                continue
        if not stmts:
            return 0
        try:
            await self._exec(';'.join(stmts))
            return len(stmts)
        except Exception:
            return 0

    async def collect_av_ipo_upcoming(self) -> int:
        """Collect upcoming IPOs using Alpha Vantage IPO_CALENDAR (CSV)."""
        if self.provider != 'alphavantage' or not self.av_api_key:
            return 0
        def _esc(s: str) -> str:
            try:
                return str(s).replace("'", "''")
            except Exception:
                return str(s)
        rows = await self._get_csv_alphavantage('IPO_CALENDAR', {})
        if not rows:
            return 0
        stmts: List[str] = []
        for r in rows:
            try:
                sym = _esc(r.get('symbol') or r.get('Symbol') or '')
                if not sym:
                    continue
                company = _esc(r.get('name') or r.get('Name') or '')
                date_s = (r.get('ipoDate') or r.get('IPODate') or r.get('date') or '').strip()
                if not date_s:
                    continue
                exch = _esc(r.get('exchange') or r.get('Exchange') or '')
                # priceRange may be present as single field or low/high
                pr = r.get('priceRange') or r.get('PriceRange')
                low = r.get('priceRangeLow') or r.get('PriceRangeLow')
                high = r.get('priceRangeHigh') or r.get('PriceRangeHigh')
                price_range = _esc(pr or (f"{low}-{high}" if (low and high) else ''))
                # shares may include commas
                shares_raw = (r.get('shares') or r.get('Shares') or '0').replace(',', '')
                try:
                    shares = int(float(shares_raw))
                except Exception:
                    shares = 0
                updated = _esc(r.get('updated') or date_s)
                stmts.append(
                    "insert into ipo_calendar (symbol,company,date,exchange,price_range,shares,updated,timestamp) values ("
                    f"'{sym}','{company}',to_date('{date_s}','yyyy-MM-dd'),'{exch}','{price_range}',{shares},'{updated}','{date_s}T00:00:00Z')"
                )
            except Exception:
                continue
        if not stmts:
            return 0
        try:
            await self._exec(';'.join(stmts))
            return len(stmts)
        except Exception:
            return 0


_CALENDAR_SINGLETON: Optional[CalendarService] = None


async def get_calendar_service() -> Optional[CalendarService]:
    global _CALENDAR_SINGLETON
    if _CALENDAR_SINGLETON is None:
        svc = CalendarService()
        if svc.enabled:
            try:
                await svc.start()
            except Exception:
                pass
        _CALENDAR_SINGLETON = svc
    return _CALENDAR_SINGLETON

#!/usr/bin/env python3
"""QuestDB Table Bootstrap (Idempotent)

Creates core time-series tables required for historical + live ingestion if they
are missing. Safe to run multiple times (will skip existing tables).

Output: JSON summary to stdout.
Exit codes:
 0 success (all created or existed)
 2 error (one or more failures)

Environment:
  QUESTDB_HTTP_URL  (default: http://127.0.0.1:9000/exec)

NOTE: We use the HTTP /exec endpoint for simplicity & portability (works even
when asyncpg / Postgres wire is unavailable inside the container). For each
table we attempt a trivial SELECT; on failure containing 'does not exist' we
issue the CREATE TABLE DDL.
"""
from __future__ import annotations
import os, json, time, urllib.parse as _up, urllib.request as _ur
from urllib.error import URLError, HTTPError
from datetime import datetime, timezone

HTTP_URL = os.getenv('QUESTDB_HTTP_URL', f"http://{os.getenv('QUESTDB_HOST','127.0.0.1')}:9000/exec")

TABLES: dict[str, str] = {
    'market_data': (
        "CREATE TABLE market_data ("  # Basic OHLCV per symbol
        "symbol SYMBOL, "
        "timestamp TIMESTAMP, "
        "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, "
        "volume LONG, source SYMBOL"  # origin provider tag
        ") timestamp (timestamp) PARTITION BY DAY WAL"),
    'options_data': (
        "CREATE TABLE options_data ("  # Options daily aggregates (extended)
        "underlying SYMBOL, option_symbol SYMBOL, timestamp TIMESTAMP, "
        "bid DOUBLE, ask DOUBLE, last DOUBLE, volume LONG, open_interest LONG, "
        "expiry DATE, strike DOUBLE, option_type SYMBOL, source SYMBOL, "
        "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE"  # Added OHLC for aggregate bars
        ") timestamp (timestamp) PARTITION BY DAY WAL"),
    'news_events': (
        "CREATE TABLE news_events ("  # Structured news feed events
        "timestamp TIMESTAMP, source SYMBOL, id SYMBOL, headline STRING, "
        "body STRING, symbols SYMBOL, sentiment DOUBLE"
        ") timestamp (timestamp) PARTITION BY DAY WAL"),
    'social_events': (
        "CREATE TABLE social_events ("  # Social / sentiment feed items
        "timestamp TIMESTAMP, platform SYMBOL, id SYMBOL, author SYMBOL, "
        "symbol SYMBOL, content STRING, sentiment DOUBLE"
        ") timestamp (timestamp) PARTITION BY DAY WAL"),
    # Align with ILP writers in news_service and social_media_collector
    'news_items': (
        "CREATE TABLE news_items ("
        "symbol SYMBOL, ts TIMESTAMP, "
        "title STRING, source STRING, url STRING, "
        "sentiment DOUBLE, relevance DOUBLE, provider STRING"
        ") timestamp (ts) PARTITION BY DAY WAL"),
    'social_signals': (
        "CREATE TABLE social_signals ("
        "symbol SYMBOL, source SYMBOL, ts TIMESTAMP, "
        "sentiment DOUBLE, engagement DOUBLE, influence DOUBLE, "
        "author STRING, url STRING, content STRING"
        ") timestamp (ts) PARTITION BY DAY WAL"),
    # Lightweight analytics time-series for system/trading KPIs
    # Flexible schema using metric name/value pairs to avoid frequent DDL
    'system_analytics': (
        "CREATE TABLE system_analytics ("
        "ts TIMESTAMP, component SYMBOL, metric SYMBOL, "
        "value DOUBLE, str_value STRING"
        ") timestamp (ts) PARTITION BY DAY WAL"),
    'trading_analytics': (
        "CREATE TABLE trading_analytics ("
        "ts TIMESTAMP, symbol SYMBOL, metric SYMBOL, "
        "value DOUBLE, str_value STRING"
        ") timestamp (ts) PARTITION BY DAY WAL"),
}


def _http_query(sql: str) -> tuple[int, str]:
    """Execute a query returning (status_code, body_text)."""
    params = {'query': sql}
    q = _up.urlencode(params)
    url = HTTP_URL + ("&" if "?" in HTTP_URL else "?") + q
    try:
        with _ur.urlopen(url, timeout=10) as resp:  # nosec - controlled URL
            body = resp.read().decode(errors='ignore')
            return resp.getcode(), body
    except HTTPError as he:  # capture body for diagnostics
        try:
            body = he.read().decode(errors='ignore')
        except Exception:  # noqa: BLE001
            body = ''
        return he.code, body
    except URLError as ue:  # network error
        return 599, str(ue)
    except Exception as e:  # noqa: BLE001
        return 598, str(e)


def table_exists(name: str) -> tuple[bool, str | None]:
    # Minimal existence probe
    status, body = _http_query(f"SELECT * FROM {name} LIMIT 1")
    if status == 200:
        return True, None
    lower = (body or '').lower()
    if 'does not exist' in lower or 'not exist' in lower:
        return False, None
    # Other errors (syntax, network) treated as non-existence error cause
    return False, body


def create_table(name: str, ddl: str) -> tuple[bool, str | None]:
    status, body = _http_query(ddl)
    if status == 200:
        return True, None
    return False, f"status={status} body_snippet={(body or '')[:160]}"


def main():
    results = []
    errors: list[str] = []
    created = 0
    skipped = 0
    start = time.time()
    # Track whether we patched options schema
    schema_patches: list[str] = []
    for table, ddl in TABLES.items():
        exists, probe_err = table_exists(table)
        if exists:
            results.append({'table': table, 'status': 'exists'})
            skipped += 1
            continue
        if probe_err and 'does not exist' not in (probe_err.lower()):
            # unexpected probe error; attempt create anyway but record context
            results.append({'table': table, 'status': 'probe_error', 'error': probe_err[:160]})
        ok, err = create_table(table, ddl)
        if ok:
            created += 1
            results.append({'table': table, 'status': 'created'})
        else:
            errors.append(f"create_failed:{table}:{err}")
            results.append({'table': table, 'status': 'error', 'error': err})
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'http_endpoint': HTTP_URL,
        'results': results,
        'created': created,
        'skipped': skipped,
        'schema_patches': schema_patches,
        'errors': errors,
        'duration_seconds': round(time.time() - start, 3),
        'status': 'ok' if not errors else 'error'
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

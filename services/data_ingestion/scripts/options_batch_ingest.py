#!/usr/bin/env python3
"""CLI helper to drive batch option chain ingestion via the data-ingestion API.

Usage:
  python options_batch_ingest.py --underlyings AAPL,SPY,QQQ --days 7 --ahead 45 --max-contracts 40 \
      --api http://localhost:8002 --pacing 0.15

Features:
  - Splits symbols into waves to avoid large single POST bodies if desired
  - Progress output with per-wave summary
  - Retries with exponential backoff for transient failures
  - Exit code non-zero if any wave hard-fails

This script intentionally keeps dependencies minimal (stdlib only) for production hygiene.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from typing import List


def post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted internal URL)
        return json.loads(resp.read().decode())


def run_wave(api_base: str, symbols: List[str], start: date, end: date, args) -> dict:
    payload = {
        "underlyings": symbols,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "max_contracts": args.max_contracts,
        "pacing_seconds": args.pacing,
        "expired": False,
        "include_recent_expired": True,
        "recent_expired_days": args.recent_expired_days,
    }
    url = api_base.rstrip('/') + '/market-data/options-chains'
    attempt = 0
    backoff = 2.0
    while True:
        attempt += 1
        try:
            return post_json(url, payload, timeout=args.timeout)
        except urllib.error.HTTPError as e:  # noqa: PERF203
            body = e.read().decode()[:400]
            print(f"HTTP {e.code} for wave symbols={symbols} body={body}", file=sys.stderr)
            if 500 <= e.code < 600 and attempt <= args.retries:
                time.sleep(backoff)
                backoff *= 1.8
                continue
            raise
        except Exception as e:  # noqa: BLE001
            print(f"Error wave symbols={symbols}: {e}", file=sys.stderr)
            if attempt <= args.retries:
                time.sleep(backoff)
                backoff *= 1.8
                continue
            raise


def main():  # noqa: D401
    parser = argparse.ArgumentParser(description="Batch option chain ingestion driver")
    parser.add_argument('--underlyings', required=True, help='Comma separated list of symbols')
    parser.add_argument('--api', default='http://localhost:8002', help='Base URL for ingestion API')
    parser.add_argument('--days', type=int, default=7, help='Historical window days back (start = today-days)')
    parser.add_argument('--ahead', type=int, default=45, help='Forward expiry window days (informational only; service uses its own env-driven default)')
    parser.add_argument('--max-contracts', type=int, default=50, help='Max contracts per underlying')
    parser.add_argument('--pacing', type=float, default=0.15, help='Pacing seconds between contracts')
    parser.add_argument('--recent-expired-days', type=int, default=5, help='Recent expired days to include')
    parser.add_argument('--wave-size', type=int, default=25, help='Symbols per API request wave')
    parser.add_argument('--retries', type=int, default=3, help='Retries for transient failures (5xx)')
    parser.add_argument('--timeout', type=float, default=420.0, help='HTTP timeout seconds per wave')
    args = parser.parse_args()

    syms = [s.strip().upper() for s in args.underlyings.split(',') if s.strip()]
    if not syms:
        print('No symbols provided', file=sys.stderr)
        return 2

    end = date.today()
    start = end - timedelta(days=args.days)

    waves = [syms[i:i + args.wave_size] for i in range(0, len(syms), args.wave_size)]
    grand = {"underlyings": 0, "contracts_processed": 0, "bars_ingested": 0, "errors": 0}
    t_start = time.time()

    for idx, wave in enumerate(waves, 1):
        print(f"[wave {idx}/{len(waves)}] ingesting {len(wave)} symbols: {wave}")
        try:
            resp = run_wave(args.api, wave, start, end, args)
        except Exception:
            print(f"Wave {idx} FAILED hard after retries", file=sys.stderr)
            return 3
        totals = resp.get('totals', {})
        grand['underlyings'] += int(totals.get('underlyings', 0) or 0)
        grand['contracts_processed'] += int(totals.get('contracts_processed', 0) or 0)
        grand['bars_ingested'] += int(totals.get('bars_ingested', 0) or 0)
        grand['errors'] += int(totals.get('errors', 0) or 0)
        print(f"  wave_totals contracts={totals.get('contracts_processed')} bars={totals.get('bars_ingested')} errors={totals.get('errors')} available={resp.get('results', [{}])[:1][0].get('total_contracts_available','?')}")
        # Small safety sleep to avoid bursty provider pressure
        time.sleep(1.0)

    dur = time.time() - t_start
    print("== SUMMARY ==")
    print(json.dumps({"totals": grand, "duration_seconds": round(dur, 2)}, indent=2))
    # Non-zero errors => exit 4
    if grand['errors'] > 0:
        return 4
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())

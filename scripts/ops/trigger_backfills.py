#!/usr/bin/env python3
"""
Trigger core backfills via Admin proxy endpoints.

Usage:
  python scripts/ops/trigger_backfills.py --api-url http://localhost:8000 --equities --options --news --calendar

Requires an authenticated session cookie or a Bearer token. For automation, you can pass a Bearer token via --token.
Alternatively, run this inside the API container with curl against the ingestion service directly.
"""
from __future__ import annotations
import argparse
import json
import sys

import httpx


def _auth_headers(token: str | None) -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def main() -> int:
    p = argparse.ArgumentParser(description="Trigger historical backfills")
    p.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the API service")
    p.add_argument("--token", default=None, help="Optional Bearer access token (admin)")
    p.add_argument("--equities", action="store_true", help="Trigger equities 20y backfill")
    p.add_argument("--options", action="store_true", help="Trigger options 5y rolling backfill")
    p.add_argument("--news", action="store_true", help="Trigger news backfill (3y)")
    p.add_argument("--calendar", action="store_true", help="Trigger calendar backfill (5y incl dividends)")
    args = p.parse_args()

    base = args.api_url.rstrip('/')
    actions: list[tuple[str, dict]] = []
    if args.equities:
        actions.append(("/admin/api/backfill/equities", {"years": 20, "max_symbols": 1000, "pacing_seconds": 0.2}))
    if args.options:
        actions.append(("/admin/api/backfill/options", {"max_underlyings": 200, "pacing_seconds": 0.2, "expiry_back_days": 365*2, "expiry_ahead_days": 90, "hist_lookback_days": 365*5}))
    if args.news:
        actions.append(("/admin/api/backfill/news", {"days": 365*3, "batch_days": 14, "max_articles_per_batch": 80}))
    if args.calendar:
        actions.append(("/admin/api/backfill/calendar", {"years": 5, "include_dividends": True, "pacing_seconds": 0.1}))

    if not actions:
        print("No backfill flags provided. Use --equities/--options/--news/--calendar.")
        return 1

    headers = _auth_headers(args.token)

    with httpx.Client(timeout=30.0) as client:
        for path, payload in actions:
            url = base + path
            try:
                r = client.post(url, json=payload, headers=headers)
            except Exception as e:
                print(f"ERROR: POST {url} failed: {e}")
                continue
            ok = r.status_code == 200
            print(f"POST {path} -> {r.status_code}")
            try:
                print(json.dumps(r.json(), indent=2)[:800])
            except Exception:
                print((r.text or "")[:400])
            if not ok and r.status_code in (401,403):
                print("Hint: This endpoint requires admin auth (and MFA). Use --token with a valid access token.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

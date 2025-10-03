#!/usr/bin/env python3
"""
Options Coverage Report
Summarizes QuestDB option_daily coverage by underlying. Connects to QuestDB HTTP API.

Outputs per-underlying:
- contracts: distinct option_ticker count
- rows: total bars
- first_day, last_day: min/max timestamp (UTC date)
- recent_gap_days_30d: days in last 30d window with no bars (approx)

Usage:
  python options_coverage_report.py --questdb-url http://trading-questdb:9000/exec --underlyings AAPL,MSFT,TSLA
"""

import argparse
import asyncio
from datetime import datetime, timedelta
import json
import sys
from typing import Dict, List, Optional

import aiohttp


async def qdb_query(session: aiohttp.ClientSession, url: str, sql: str) -> dict:
    params = {"query": sql}
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            txt = await resp.text()
            raise RuntimeError(f"QuestDB HTTP {resp.status}: {txt[:200]}")
        return await resp.json()


async def coverage_for_underlying(session: aiohttp.ClientSession, url: str, underlying: str) -> Dict:
    # Sanitize
    u = underlying.upper()
    # Total rows and contracts
    sql_summary = (
        "select count() as rows, count_distinct(option_ticker) as contracts, "
        "to_char(min(timestamp),'yyyy-MM-dd') as first_day, to_char(max(timestamp),'yyyy-MM-dd') as last_day "
        f"from option_daily where underlying = '{u}'"
    )
    data = await qdb_query(session, url, sql_summary)
    rows = data.get('dataset', [])
    if not rows:
        return {"underlying": u, "rows": 0, "contracts": 0, "first_day": None, "last_day": None, "recent_gap_days_30d": None}
    r = rows[0]
    # dataset rows are arrays; columns lists names
    cols = data.get('columns', [])
    col_idx = {c['name']: i for i, c in enumerate(cols)}
    total_rows = int(r[col_idx['rows']]) if 'rows' in col_idx else 0
    contracts = int(r[col_idx['contracts']]) if 'contracts' in col_idx else 0
    first_day = r[col_idx['first_day']] if 'first_day' in col_idx else None
    last_day = r[col_idx['last_day']] if 'last_day' in col_idx else None

    # Approximate missing days in last 30 days by counting distinct days present
    sql_recent_days = (
        "select count_distinct(to_char(timestamp,'yyyy-MM-dd')) as have_days "
        f"from option_daily where underlying = '{u}' and timestamp >= dateadd('d', -30, now())"
    )
    d2 = await qdb_query(session, url, sql_recent_days)
    have_days = 0
    if d2.get('dataset'):
        idx = {c['name']: i for i, c in enumerate(d2.get('columns', []))}
        try:
            have_days = int(d2['dataset'][0][idx['have_days']])
        except Exception:
            have_days = 0
    gap_days = max(0, 30 - have_days)

    return {
        "underlying": u,
        "rows": total_rows,
        "contracts": contracts,
        "first_day": first_day,
        "last_day": last_day,
        "recent_gap_days_30d": gap_days,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questdb-url", default="http://trading-questdb:9000/exec", help="QuestDB HTTP exec endpoint")
    ap.add_argument("--underlyings", default="", help="Comma-separated list; if empty will try a small built-in set")
    args = ap.parse_args()

    underlyings = [u.strip().upper() for u in args.underlyings.split(',') if u.strip()] or [
        'AAPL','MSFT','TSLA','NVDA','SPY'
    ]

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        out: List[Dict] = []
        for u in underlyings:
            try:
                cov = await coverage_for_underlying(session, args.questdb_url, u)
                out.append(cov)
            except Exception as e:
                out.append({"underlying": u, "error": str(e)})
        print(json.dumps({"generated_at": datetime.utcnow().isoformat(), "questdb": args.questdb_url, "coverage": out}, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)

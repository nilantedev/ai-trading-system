#!/usr/bin/env python3
"""
Backfill Driver - Safe CLI to drive coverage exports and pilot backfills

Why: Avoids curl/heredoc JSON pitfalls and provides sensible defaults/timeouts.

Targets the Data Ingestion Service API (default http://127.0.0.1:8002).

Subcommands:
  - coverage: Re-export equities and options coverage artifacts
  - options:  Pilot options-chain backfill for underlyings over a date window
  - news:     Historical news backfill in sliding windows
  - social:   Historical social backfill in small hour windows
  - all:      Run coverage, then options/news/social pilots in sequence

Examples:
  ./scripts/backfill_driver.py coverage
  ./scripts/backfill_driver.py options --symbols AAPL,MSFT --start 2024-08-10 --end 2024-09-10 \
      --start-expiry -7d --end-expiry +60d --max-contracts 300 --pace 0.2
  ./scripts/backfill_driver.py news --symbols AAPL,MSFT --start 2019-01-01 --end 2020-12-31 \
      --batch-days 14 --max-articles 80
  ./scripts/backfill_driver.py social --symbols AAPL,MSFT --start 2022-01-01 --end 2022-02-01 \
      --batch-hours 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional, Tuple

try:
    import httpx  # type: ignore
except Exception as e:  # noqa: BLE001
    print("This script requires the 'httpx' package. Install with: pip install httpx", file=sys.stderr)
    raise


DEFAULT_BASE_URL = os.getenv("INGEST_BASE_URL", "http://127.0.0.1:8002").rstrip("/")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_relative_day(s: str, anchor: Optional[date] = None) -> date:
    """Parse "+Nd" or "-Nd" relative to anchor (default today), or YYYY-MM-DD absolute."""
    if not s:
        raise ValueError("empty date")
    s = s.strip()
    if s[0] in "+-" and s.endswith("d"):
        try:
            n = int(s[:-1])
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"invalid relative days: {s}") from e
        base = anchor or date.today()
        return base + timedelta(days=n)
    return _parse_date(s)


def _split_symbols(csv: Optional[str]) -> List[str]:
    if not csv:
        return []
    return [s.strip().upper() for s in csv.split(',') if s.strip()]


def _pretty(obj) -> str:
    try:
        return json.dumps(obj, indent=2, sort_keys=True)
    except Exception:
        return str(obj)


@dataclass
class Client:
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    # ---------------- Coverage ---------------- #
    def export_equities_coverage(self, symbols: Optional[List[str]] = None) -> dict:
        qp = {}
        if symbols:
            qp["symbols"] = ",".join(symbols)
        with self._client() as c:
            r = c.post("/coverage/equities/export", params=qp, headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()

    def export_options_coverage(self, underlyings: Optional[List[str]] = None) -> dict:
        qp = {}
        if underlyings:
            qp["underlyings"] = ",".join(underlyings)
        with self._client() as c:
            r = c.post("/coverage/options/export", params=qp, headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()

    # ---------------- Options backfill ---------------- #
    def backfill_options_chain(
        self,
        underlying: str,
        start: date,
        end: date,
        start_expiry: Optional[date] = None,
        end_expiry: Optional[date] = None,
        max_contracts: int = 300,
        pace: float = 0.2,
    ) -> dict:
        qp = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "max_contracts": str(max_contracts),
            "pacing_seconds": str(pace),
        }
        if start_expiry:
            qp["start_expiry"] = start_expiry.isoformat()
        if end_expiry:
            qp["end_expiry"] = end_expiry.isoformat()
        with self._client() as c:
            r = c.post(f"/market-data/options-chain/{underlying.upper()}", params=qp, headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()

    # ---------------- News backfill ---------------- #
    def backfill_news(
        self,
        symbols: List[str],
        start: date,
        end: date,
        batch_days: int = 14,
        max_articles: int = 80,
    ) -> dict:
        body = {
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "batch_days": batch_days,
            "max_articles_per_batch": max_articles,
        }
        with self._client() as c:
            r = c.post("/news/backfill", json=body, headers={"Content-Type": "application/json", "Accept": "application/json"})
            r.raise_for_status()
            return r.json()

    # ---------------- Social backfill ---------------- #
    def backfill_social(
        self,
        symbols: List[str],
        start: date,
        end: date,
        batch_hours: int = 6,
    ) -> dict:
        body = {
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "batch_hours": batch_hours,
        }
        with self._client() as c:
            r = c.post("/social/backfill", json=body, headers={"Content-Type": "application/json", "Accept": "application/json"})
            r.raise_for_status()
            return r.json()


def cmd_coverage(args) -> int:
    client = Client(base_url=args.base_url, timeout=args.timeout)
    eq_syms = _split_symbols(args.symbols) if args.symbols else None
    op_syms = _split_symbols(args.underlyings) if args.underlyings else None
    print(f"Exporting equities coverage to artifacts (base {client.base_url})...")
    try:
        eq = client.export_equities_coverage(eq_syms)
        cov = eq.get("coverage", [])
        print(f"  equities: status={eq.get('status','?')} items={len(cov)}")
        if cov:
            print(f"  sample: {_pretty(cov[:2])}")
    except Exception as e:  # noqa: BLE001
        print(f"  equities export failed: {e}")
    print("Exporting options coverage to artifacts...")
    try:
        op = client.export_options_coverage(op_syms)
        cov = op.get("coverage", [])
        print(f"  options: status={op.get('status','?')} items={len(cov)}")
        if cov:
            print(f"  sample: {_pretty(cov[:2])}")
    except Exception as e:  # noqa: BLE001
        print(f"  options export failed: {e}")
    return 0


def cmd_options(args) -> int:
    client = Client(base_url=args.base_url, timeout=args.timeout)
    syms = _split_symbols(args.symbols)
    if not syms:
        print("--symbols is required (CSV)")
        return 2
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    # expiry window may be relative to today
    start_expiry = _parse_relative_day(args.start_expiry) if args.start_expiry else None
    end_expiry = _parse_relative_day(args.end_expiry) if args.end_expiry else None
    ok = 0
    for u in syms:
        try:
            res = client.backfill_options_chain(
                u, start=start, end=end, start_expiry=start_expiry, end_expiry=end_expiry,
                max_contracts=args.max_contracts, pace=args.pace,
            )
            print(f"{u}: {res.get('contracts_processed',0)} contracts, {res.get('bars_ingested',0)} bars")
            ok += 1
        except httpx.HTTPStatusError as he:  # typical: 403 if disabled
            print(f"{u}: HTTP {he.response.status_code} -> {he}")
        except Exception as e:  # noqa: BLE001
            print(f"{u}: error -> {e}")
    return 0 if ok > 0 else 1


def cmd_news(args) -> int:
    client = Client(base_url=args.base_url, timeout=args.timeout)
    syms = _split_symbols(args.symbols)
    if not syms:
        print("--symbols is required (CSV)")
        return 2
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    try:
        res = client.backfill_news(syms, start=start, end=end, batch_days=args.batch_days, max_articles=args.max_articles)
        print(f"news: windows={res.get('windows',0)} articles={res.get('articles',0)} symbols={res.get('symbols',0)}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"news backfill failed: {e}")
        return 1


def cmd_social(args) -> int:
    client = Client(base_url=args.base_url, timeout=args.timeout)
    syms = _split_symbols(args.symbols)
    if not syms:
        print("--symbols is required (CSV)")
        return 2
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    try:
        res = client.backfill_social(syms, start=start, end=end, batch_hours=args.batch_hours)
        print(f"social: windows={res.get('windows',0)} signals={res.get('signals',0)} symbols={res.get('symbols',0)}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"social backfill failed: {e}")
        return 1


def cmd_all(args) -> int:
    # Coverage
    rc = cmd_coverage(args)
    # Options pilot (only if symbols provided)
    if args.symbols and args.start and args.end:
        _ = cmd_options(args)
    else:
        print("(skip options pilot: provide --symbols, --start, --end)")
    # News pilot
    if args.symbols and args.news_start and args.news_end:
        ns = argparse.Namespace(
            base_url=args.base_url,
            timeout=args.timeout,
            symbols=args.symbols,
            start=args.news_start,
            end=args.news_end,
            batch_days=args.batch_days,
            max_articles=args.max_articles,
        )
        _ = cmd_news(ns)
    else:
        print("(skip news pilot: provide --symbols, --news-start, --news-end)")
    # Social pilot
    if args.symbols and args.social_start and args.social_end:
        ss = argparse.Namespace(
            base_url=args.base_url,
            timeout=args.timeout,
            symbols=args.symbols,
            start=args.social_start,
            end=args.social_end,
            batch_hours=args.batch_hours,
        )
        _ = cmd_social(ss)
    else:
        print("(skip social pilot: provide --symbols, --social-start, --social-end)")
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Drive coverage exports and pilot backfills")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Base URL for ingestion API (default {DEFAULT_BASE_URL})")
    p.add_argument("--timeout", type=float, default=45.0, help="HTTP timeout seconds")

    sub = p.add_subparsers(dest="cmd", required=True)

    # coverage
    pc = sub.add_parser("coverage", help="Export equities and options coverage artifacts")
    pc.add_argument("--symbols", help="CSV symbols for equities coverage (optional)")
    pc.add_argument("--underlyings", help="CSV underlyings for options coverage (optional)")
    pc.set_defaults(func=cmd_coverage)

    # options
    po = sub.add_parser("options", help="Pilot options-chain backfill")
    po.add_argument("--symbols", required=True, help="CSV underlyings e.g. AAPL,MSFT")
    po.add_argument("--start", required=True, help="Start date YYYY-MM-DD for aggregates window")
    po.add_argument("--end", required=True, help="End date YYYY-MM-DD for aggregates window")
    po.add_argument("--start-expiry", help="Start expiry YYYY-MM-DD or relative like -7d")
    po.add_argument("--end-expiry", help="End expiry YYYY-MM-DD or relative like +60d")
    po.add_argument("--max-contracts", type=int, default=300)
    po.add_argument("--pace", type=float, default=0.2, help="Seconds between contracts")
    po.set_defaults(func=cmd_options)

    # news
    pn = sub.add_parser("news", help="Historical news backfill in windows")
    pn.add_argument("--symbols", required=True, help="CSV symbols")
    pn.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    pn.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    pn.add_argument("--batch-days", type=int, default=14)
    pn.add_argument("--max-articles", type=int, default=80)
    pn.set_defaults(func=cmd_news)

    # social
    ps = sub.add_parser("social", help="Historical social backfill in hour windows")
    ps.add_argument("--symbols", required=True, help="CSV symbols")
    ps.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ps.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ps.add_argument("--batch-hours", type=int, default=6)
    ps.set_defaults(func=cmd_social)

    # all
    pall = sub.add_parser("all", help="Run coverage then options/news/social pilots")
    pall.add_argument("--symbols", help="CSV symbols for pilots (e.g., AAPL,MSFT)")
    # options window for ALL
    pall.add_argument("--start", help="Options aggregates start date YYYY-MM-DD")
    pall.add_argument("--end", help="Options aggregates end date YYYY-MM-DD")
    pall.add_argument("--start-expiry", help="Options start expiry (YYYY-MM-DD or +/-Nd)")
    pall.add_argument("--end-expiry", help="Options end expiry (YYYY-MM-DD or +/-Nd)")
    pall.add_argument("--max-contracts", type=int, default=300)
    pall.add_argument("--pace", type=float, default=0.2)
    # news window for ALL
    pall.add_argument("--news-start", help="News start date YYYY-MM-DD")
    pall.add_argument("--news-end", help="News end date YYYY-MM-DD")
    pall.add_argument("--batch-days", type=int, default=14)
    pall.add_argument("--max-articles", type=int, default=80)
    # social window for ALL
    pall.add_argument("--social-start", help="Social start date YYYY-MM-DD")
    pall.add_argument("--social-end", help="Social end date YYYY-MM-DD")
    pall.add_argument("--batch-hours", type=int, default=6)
    pall.set_defaults(func=cmd_all)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except httpx.ConnectError as ce:  # service likely not running
        print(f"Connection failed to {args.base_url}: {ce}")
        return 3
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

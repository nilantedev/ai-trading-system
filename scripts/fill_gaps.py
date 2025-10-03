#!/usr/bin/env python3
"""
Fill historical gaps by calling the data-ingestion service bulk daily endpoint.

Input CSV: symbol,start,end,type,note (as produced by generate_gap_targets.py)

Options:
    --ingest-url   Base URL for data-ingestion service (default: http://127.0.0.1:8002)
    --concurrency  Max concurrent requests (default: 4)
    --pacing       Seconds to sleep between task submissions (default: 0.1)
    --max-rows     Limit number of gap rows to process (0 = all)
    --targets -    Read gap targets from stdin (use '-' for stdin)
    --out -        Write results to stdout (use '-' for stdout)

Notes:
  - Uses POST /market-data/historical/{symbol}?timeframe=1d&start=YYYY-MM-DD&end=YYYY-MM-DD
  - Idempotent: service should dedupe or skip existing rows on persist.
  - Honors HISTORICAL_PROVIDER_PRIMARY_ONLY via service env.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from typing import Tuple, Iterable

import aiohttp


async def fetch_one(session: aiohttp.ClientSession, base: str, sym: str, start: str, end: str) -> Tuple[str, int, str | None]:
    url = f"{base.rstrip('/')}/market-data/historical/{sym}?timeframe=1d&start={start}&end={end}"
    try:
        async with session.post(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                txt = await resp.text()
                return sym, 0, f"HTTP {resp.status}: {txt[:300]}"
            data = await resp.json()
            return sym, int(data.get('count') or 0), None
    except Exception as e:
        return sym, 0, str(e)


async def worker(name: str, q: asyncio.Queue, base: str, result_q: asyncio.Queue, session: aiohttp.ClientSession, pacing: float = 0.0):
    while True:
        item = await q.get()
        if item is None:
            q.task_done()
            return
        sym, start, end = item
        sym_u = sym.upper()
        count, err = 0, None
        try:
            if pacing and pacing > 0:
                await asyncio.sleep(pacing)
            s, count, err = await fetch_one(session, base, sym_u, start, end)
        finally:
            await result_q.put((sym_u, start, end, count, err))
            q.task_done()


def _iter_rows_from_csv(fobj) -> Iterable[Tuple[str, str, str]]:
    r = csv.DictReader(fobj)
    for row in r:
        yield (row['symbol'], row['start'], row['end'])


async def main_async(args):
    tasks = []
    q: asyncio.Queue = asyncio.Queue()
    result_q: asyncio.Queue = asyncio.Queue()
    base = args.ingest_url
    sem = asyncio.Semaphore(max(1, args.concurrency))

    # Load targets (support file path or stdin when '-')
    if args.targets in ('-', '/dev/stdin'):
        rows = list(_iter_rows_from_csv(sys.stdin))
    else:
        with open(args.targets, newline='', encoding='utf-8') as f:
            rows = list(_iter_rows_from_csv(f))
    if args.max_rows > 0:
        rows = rows[:args.max_rows]

    # Enqueue
    for sym, start, end in rows:
        await q.put((sym, start, end))

    # Writer coroutine to stream results as they arrive
    async def writer_task():
        if args.out in ('-', '/dev/stdout'):
            out_f = sys.stdout
            close = False
        else:
            out_f = open(args.out, 'w', newline='', encoding='utf-8')
            close = True
        try:
            w = csv.writer(out_f)
            w.writerow(['symbol', 'start', 'end', 'rows_ingested', 'error'])
            out_f.flush()
            ok = 0
            total = 0
            while True:
                row = await result_q.get()
                if row is None:
                    result_q.task_done()
                    break
                total += 1
                _, _, _, c, e = row
                if (e is None and c >= 0):
                    ok += 1
                w.writerow(row)
                out_f.flush()
                result_q.task_done()
            # Print summary: to stderr if writing to stdout, else stdout
            summary = f"Gap fill completed: {ok}/{total} requests ok; wrote {args.out}"
            if args.out in ('-', '/dev/stdout'):
                print(summary, file=sys.stderr)
            else:
                print(summary)
        finally:
            if close:
                out_f.close()

    async with aiohttp.ClientSession() as session:
        # Start writer
        writer = asyncio.create_task(writer_task())

        # Start workers
        for i in range(max(1, args.concurrency)):
            tasks.append(asyncio.create_task(worker(f"w{i+1}", q, base, result_q, session, pacing=float(args.pacing))))

        # Wait for all work items to complete
        await q.join()

        # Stop workers
        for _ in tasks:
            await q.put(None)
        await asyncio.gather(*tasks)

        # Signal writer to finish
        await result_q.put(None)
        await writer


def main():
    ap = argparse.ArgumentParser(description='Fill historical gaps via ingestion service')
    ap.add_argument('--targets', required=True, help='CSV of gap targets (symbol,start,end,...) or - for stdin')
    ap.add_argument('--ingest-url', default='http://127.0.0.1:8002', help='Data ingestion service base URL')
    ap.add_argument('--concurrency', type=int, default=4, help='Max concurrent requests')
    ap.add_argument('--pacing', type=float, default=0.1, help='Seconds between submissions (not strict)')
    ap.add_argument('--max-rows', type=int, default=0, help='Process only first N rows (0 = all)')
    ap.add_argument('--out', default='/mnt/fastdrive/trading/gap_fill_results.csv', help='Output CSV for results or - for stdout')
    args = ap.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == '__main__':
    main()

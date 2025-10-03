#!/usr/bin/env python3
"""Data retention enforcement script.

Policies (rolling windows):
- Equities (stock) price/history: 20 years
- Options: 5 years
- News: 5 years
- Social: 5 years

Design:
- Reads cutoff dates relative to now (UTC)
- Supports --dry-run (default) and --apply
- Logs candidate deletions grouped by domain
- Uses environment variables for DSN / connection selection
- Modular backends (currently PostgreSQL + QuestDB placeholders)

Assumptions:
- PostgreSQL tables (example): equities_prices(date, symbol,...), options_quotes(ts,...), news_items(published_at,...), social_posts(published_at,...)
- QuestDB may store large time-series; we illustrate deletion via SQL ILP (if configured).

Exit Codes:
0 success, 1 partial failures, 2 fatal config error
"""
from __future__ import annotations
import argparse
import asyncio
import json
import fcntl
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, List, Optional

try:
    from trading_common.database_manager import get_database_manager  # type: ignore
except Exception:  # pragma: no cover
    get_database_manager = None  # type: ignore

RETENTION_WINDOWS = {
    'equities_prices': timedelta(days=365*20),
    'options_quotes': timedelta(days=365*5),
    'news_items': timedelta(days=365*5),
    'social_posts': timedelta(days=365*5),
}

TIMESTAMP_COLUMNS = {
    'equities_prices': 'date',
    'options_quotes': 'ts',
    'news_items': 'published_at',
    'social_posts': 'published_at',
}

async def compute_cutoffs(now: datetime) -> Dict[str, datetime]:
    return {table: now - window for table, window in RETENTION_WINDOWS.items()}

async def gather_counts(session, table: str, cutoff: datetime):  # type: ignore
    col = TIMESTAMP_COLUMNS[table]
    q_total = f"SELECT COUNT(*) AS c FROM {table}"
    q_old = f"SELECT COUNT(*) AS c FROM {table} WHERE {col} < :cutoff"
    res_total = await session.execute(q_total)
    res_old = await session.execute(q_old, {"cutoff": cutoff})
    total = res_total.scalar() or 0
    old = res_old.scalar() or 0
    return total, old

async def delete_old(session, table: str, cutoff: datetime):  # type: ignore
    col = TIMESTAMP_COLUMNS[table]
    stmt = f"DELETE FROM {table} WHERE {col} < :cutoff"
    await session.execute(stmt, {"cutoff": cutoff})

LOCK_PATH_DEFAULT = "/tmp/enforce_retention.lock"
LOCK_STALE_SECONDS = 3600

def acquire_lock(lock_path: str) -> Optional[int]:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        # Check staleness
        st = os.fstat(fd)
        if st.st_size == 0:
            os.write(fd, str(int(datetime.now(timezone.utc).timestamp())).encode())
        else:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                ts_raw = os.read(fd, 32).decode(errors='ignore').strip()
                if ts_raw.isdigit():
                    started = datetime.fromtimestamp(int(ts_raw), timezone.utc)
                    age = (datetime.now(timezone.utc) - started).total_seconds()
                    if age > LOCK_STALE_SECONDS:
                        # Stale lock, rewrite timestamp
                        os.ftruncate(fd, 0)
                        os.write(fd, str(int(datetime.now(timezone.utc).timestamp())).encode())
            except Exception:
                pass
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        print(f"Another retention process is running (lock {lock_path}).")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"ERROR acquiring lock {lock_path}: {e}")
        return None

def release_lock(fd: Optional[int]):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except Exception:
        pass

async def enforce_retention(apply: bool = False, metrics_path: Optional[str] = None, lock_path: Optional[str] = None, no_lock: bool = False) -> int:
    if get_database_manager is None:
        print("Database manager unavailable; aborting retention enforcement.")
        return 2
    dbm = await get_database_manager()
    lock_fd = None
    if not no_lock:
        lock_fp = lock_path or os.getenv('RETENTION_LOCK_PATH', LOCK_PATH_DEFAULT)
        lock_fd = acquire_lock(lock_fp)
        if lock_fd is None:
            return 0  # another process active or lock issue reported
    now = datetime.now(timezone.utc)
    cutoffs = await compute_cutoffs(now)
    summary: List[Tuple[str,int,int,int]] = []  # table, total, old, deleted
    async with dbm.get_postgres() as session:  # type: ignore[attr-defined]
        for table, cutoff in cutoffs.items():
            try:
                total, old = await gather_counts(session, table, cutoff)
                deleted = 0
                if apply and old > 0:
                    await delete_old(session, table, cutoff)
                    deleted = old
                summary.append((table, total, old, deleted))
            except Exception as e:  # noqa: BLE001
                print(f"ERROR processing {table}: {e}")
    print("Retention Summary (apply=%s)" % apply)
    metrics = {"apply": apply, "generated_at": now.isoformat(), "tables": []}
    for table, total, old, deleted in summary:
        cutoff = cutoffs[table]
        entry = {"table": table, "total": total, "old": old, "deleted": deleted, "cutoff": cutoff.isoformat()}
        metrics["tables"].append(entry)
        print(f" - {table}: total={total} old={old} cutoff<{cutoff.isoformat()} deleted={deleted}")
    if metrics_path:
        try:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            print(f"Wrote retention metrics JSON: {metrics_path}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR writing metrics file {metrics_path}: {e}")
    release_lock(lock_fd)
    return 0

async def amain(args):
    return await enforce_retention(apply=args.apply, metrics_path=args.metrics_json, lock_path=args.lock_path, no_lock=args.no_lock)

def parse_args():
    p = argparse.ArgumentParser(description="Retention enforcement")
    p.add_argument('--apply', action='store_true', help='Execute deletions (omit for dry-run)')
    p.add_argument('--metrics-json', help='Optional path to write JSON summary for monitoring')
    p.add_argument('--lock-path', help='Override lock file path (default /tmp/enforce_retention.lock)')
    p.add_argument('--no-lock', action='store_true', help='Disable file locking (use with caution)')
    return p.parse_args()

def main():
    args = parse_args()
    code = asyncio.run(amain(args))
    raise SystemExit(code)

if __name__ == '__main__':  # pragma: no cover
    main()

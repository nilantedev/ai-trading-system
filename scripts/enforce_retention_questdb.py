#!/usr/bin/env python3
"""QuestDB Retention Enforcement (Production-safe)

Policies (rolling windows):
- Equities (market_data): 20 years
- Options (options_data): 5 years
- News (news_items/news_events): 5 years
- Social (social_signals/social_events): 5 years

Behavior:
- Computes cutoffs relative to now (UTC midnight)
- For each known QuestDB table, checks if any partitions are older than cutoff
- In --dry-run (default), prints what would be dropped
- With --apply, issues ALTER TABLE ... DROP PARTITION WHERE ...

Notes:
- Requires QuestDB HTTP /exec endpoint (no Postgres wire dep)
- Idempotent and safe to run repeatedly
- Limits deletions to whole partitions using WHERE < cutoff (QuestDB evaluates on partition timestamp)

Exit codes: 0 success, 2 error
"""
from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import urllib.parse as _up
import urllib.request as _ur
from urllib.error import HTTPError, URLError


HTTP_URL = os.getenv('QUESTDB_HTTP_URL', f"http://{os.getenv('QUESTDB_HOST','127.0.0.1')}:9000/exec")


@dataclass
class TableSpec:
    table: str
    ts_col: str
    horizon_days: int
    label: str


SPECS: List[TableSpec] = [
    TableSpec('market_data', 'timestamp', 365*20, 'equities'),
    TableSpec('daily_bars', 'timestamp', 365*20, 'equities'),  # Add daily_bars to retention
    TableSpec('options_data', 'timestamp', 365*5, 'options'),
    # Prefer concrete ingestion tables when present
    TableSpec('news_items', 'ts', 365*5, 'news'),
    TableSpec('news_events', 'timestamp', 365*5, 'news'),
    TableSpec('social_signals', 'ts', 365*5, 'social'),
    TableSpec('social_events', 'timestamp', 365*5, 'social'),
]


def _http_get(sql: str) -> Tuple[int, str]:
    params = {'query': sql}
    q = _up.urlencode(params)
    url = HTTP_URL + ("&" if "?" in HTTP_URL else "?") + q
    try:
        with _ur.urlopen(url, timeout=20) as resp:  # nosec - controlled internal endpoint
            return resp.getcode(), resp.read().decode('utf-8', errors='ignore')
    except HTTPError as he:
        try:
            body = he.read().decode('utf-8', errors='ignore')
        except Exception:  # noqa: BLE001
            body = str(he)
        return he.code, body
    except URLError as ue:  # network errors
        return 599, str(ue)
    except Exception as e:  # noqa: BLE001
        return 598, str(e)


def _http_exec(sql: str) -> Tuple[bool, str]:
    code, body = _http_get(sql)
    return code == 200, (body or '')


def questdb_table_exists(name: str) -> bool:
    code, _ = _http_get(f"SELECT 1 FROM {name} LIMIT 1")
    return code == 200


def min_max_ts(table: str, ts_col: str) -> Tuple[Optional[str], Optional[str]]:
    code, body = _http_get(f"SELECT min({ts_col}) AS first_ts, max({ts_col}) AS last_ts FROM {table}")
    if code != 200:
        return None, None
    try:
        import json as _json
        js = _json.loads(body)
        cols = [c['name'] for c in js.get('columns', [])]
        rows = js.get('dataset') or []
        if not rows:
            return None, None
        row = rows[0]
        cidx = {cols[i]: i for i in range(len(cols))}
        return row[cidx.get('first_ts')], row[cidx.get('last_ts')]
    except Exception:  # noqa: BLE001
        return None, None


def drop_before(table: str, ts_col: str, cutoff_iso: str) -> Tuple[bool, str]:
    # Use WHERE {ts_col} < 'cutoff' to drop all partitions strictly earlier than cutoff
    sql = f"ALTER TABLE {table} DROP PARTITION WHERE {ts_col} < to_timestamp('{cutoff_iso}')"
    return _http_exec(sql)


def compute_cutoff(days: int) -> datetime:
    now = datetime.now(timezone.utc)
    mid = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return mid - timedelta(days=days)


def iso(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def main() -> int:
    ap = argparse.ArgumentParser(description='QuestDB retention enforcement (partition drops)')
    ap.add_argument('--apply', action='store_true', help='Execute partition drops (default dry-run)')
    ap.add_argument('--json-out', type=str, help='Optional write summary to JSON file')
    args = ap.parse_args()

    summary: Dict[str, dict] = {
        'generated_at': iso(datetime.now(timezone.utc)),
        'http_url': HTTP_URL,
        'apply': args.apply,
        'tables': {},
    }
    any_error = False
    for spec in SPECS:
        if not questdb_table_exists(spec.table):
            summary['tables'][spec.table] = {
                'label': spec.label,
                'exists': False,
                'action': 'skip',
                'reason': 'table_missing',
            }
            continue
        first_ts, last_ts = min_max_ts(spec.table, spec.ts_col)
        cutoff_dt = compute_cutoff(spec.horizon_days)
        cutoff_iso = iso(cutoff_dt)
        action = 'none'
        dropped = False
        error: Optional[str] = None
        # If dataset extends beyond horizon on the left (first_ts older than cutoff), schedule drop
        needs_drop = False
        try:
            if first_ts:
                # Normalize potential formats (YYYY-MM-DD.., may include Z)
                fst = str(first_ts).replace('Z','')
                # lexical compare safe for ISO-like strings
                needs_drop = fst < cutoff_iso.replace('Z','')
        except Exception:
            needs_drop = False
        if needs_drop:
            action = 'drop_partitions_before_cutoff'
            if args.apply:
                ok, body = drop_before(spec.table, spec.ts_col, cutoff_iso)
                dropped = ok
                if not ok:
                    any_error = True
                    error = (body or '')[:200]
        summary['tables'][spec.table] = {
            'label': spec.label,
            'exists': True,
            'first_ts': first_ts,
            'last_ts': last_ts,
            'cutoff': cutoff_iso,
            'needs_drop': needs_drop,
            'action': action,
            'applied': args.apply and needs_drop,
            'dropped': dropped,
            'error': error,
        }
    if args.json_out:
        try:
            with open(args.json_out, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
        except Exception:  # noqa: BLE001
            pass
    print(json.dumps(summary, indent=2))
    return 0 if not any_error else 2


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())

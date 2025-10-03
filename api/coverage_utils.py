#!/usr/bin/env python3
"""
Coverage utilities consumed by API endpoints and health checks.

This module provides minimal implementations backed by existing ingestion
artifacts and direct QuestDB queries so the API doesn't depend on the
ingestion service process at runtime. Best-effort with graceful degradation.
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any, Dict

# Best-effort QuestDB access via trading_common.database_manager (pgwire) with HTTP fallback
async def _qdb_summary() -> Dict[str, Any]:
    # Try pg-wire first via DatabaseManager
    try:
        from trading_common.database_manager import get_database_manager  # type: ignore
        dbm = await get_database_manager()
        async with dbm.get_questdb() as q:  # type: ignore[attr-defined]
            out: Dict[str, Any] = {}
            try:
                row = await q.fetchrow("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows, count_distinct(symbol) AS symbols FROM market_data")
                out['equities'] = {
                    'first_ts': row.get('first_ts') if isinstance(row, dict) else None,
                    'last_ts': row.get('last_ts') if isinstance(row, dict) else None,
                    'rows': int(row.get('rows') or 0) if isinstance(row, dict) else 0,
                    'symbols': int(row.get('symbols') or 0) if isinstance(row, dict) else 0,
                }
            except Exception:
                out['equities'] = {'first_ts': None, 'last_ts': None, 'rows': 0, 'symbols': 0}
            try:
                row = await q.fetchrow("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows, count_distinct(option_symbol) AS contracts, count_distinct(underlying) AS underlyings FROM options_data")
                out['options'] = {
                    'first_ts': row.get('first_ts') if isinstance(row, dict) else None,
                    'last_ts': row.get('last_ts') if isinstance(row, dict) else None,
                    'rows': int(row.get('rows') or 0) if isinstance(row, dict) else 0,
                    'contracts': int(row.get('contracts') or 0) if isinstance(row, dict) else 0,
                    'underlyings': int(row.get('underlyings') or 0) if isinstance(row, dict) else 0,
                }
            except Exception:
                out['options'] = {'first_ts': None, 'last_ts': None, 'rows': 0, 'contracts': 0, 'underlyings': 0}
            # News (prefer news_items ts, fallback to news_events timestamp)
            try:
                row = await q.fetchrow("SELECT min(ts) AS first_ts, max(ts) AS last_ts, count() AS rows FROM news_items")
                first_ts = row.get('first_ts') if isinstance(row, dict) else None
                last_ts = row.get('last_ts') if isinstance(row, dict) else None
                rows = int(row.get('rows') or 0) if isinstance(row, dict) else 0
                if (first_ts is None and last_ts is None) or rows == 0:
                    row2 = await q.fetchrow("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows FROM news_events")
                    first_ts = (row2.get('first_ts') if isinstance(row2, dict) else first_ts)
                    last_ts = (row2.get('last_ts') if isinstance(row2, dict) else last_ts)
                    rows = int(row2.get('rows') or 0) if isinstance(row2, dict) else rows
                out['news'] = {'first_ts': first_ts, 'last_ts': last_ts, 'rows': rows}
            except Exception:
                out['news'] = {'first_ts': None, 'last_ts': None, 'rows': 0}
            # Social (prefer social_signals ts, fallback to social_events timestamp)
            try:
                row = await q.fetchrow("SELECT min(ts) AS first_ts, max(ts) AS last_ts, count() AS rows FROM social_signals")
                first_ts = row.get('first_ts') if isinstance(row, dict) else None
                last_ts = row.get('last_ts') if isinstance(row, dict) else None
                rows = int(row.get('rows') or 0) if isinstance(row, dict) else 0
                if (first_ts is None and last_ts is None) or rows == 0:
                    row2 = await q.fetchrow("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows FROM social_events")
                    first_ts = (row2.get('first_ts') if isinstance(row2, dict) else first_ts)
                    last_ts = (row2.get('last_ts') if isinstance(row2, dict) else last_ts)
                    rows = int(row2.get('rows') or 0) if isinstance(row2, dict) else rows
                out['social'] = {'first_ts': first_ts, 'last_ts': last_ts, 'rows': rows}
            except Exception:
                out['social'] = {'first_ts': None, 'last_ts': None, 'rows': 0}
            # Calendar (look across common tables)
            try:
                row = await q.fetchrow("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows FROM calendar_events")
                first_ts = row.get('first_ts') if isinstance(row, dict) else None
                last_ts = row.get('last_ts') if isinstance(row, dict) else None
                rows = int(row.get('rows') or 0) if isinstance(row, dict) else 0
                if (first_ts is None and last_ts is None) or rows == 0:
                    row2 = await q.fetchrow("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows FROM corporate_actions")
                    if row2:
                        first_ts = first_ts or row2.get('first_ts')
                        last_ts = last_ts or row2.get('last_ts')
                        rows = rows or int(row2.get('rows') or 0)
                if (first_ts is None and last_ts is None) or rows == 0:
                    try:
                        r3 = await q.fetchrow("SELECT min(ex_date) AS first_ts, max(ex_date) AS last_ts, count() AS rows FROM dividends")
                        if r3 and r3.get('last_ts'):
                            first_ts = first_ts or r3.get('first_ts')
                            last_ts = last_ts or r3.get('last_ts')
                            rows = rows or int(r3.get('rows') or 0)
                    except Exception:
                        pass
                    try:
                        r4 = await q.fetchrow("SELECT min(date) AS first_ts, max(date) AS last_ts, count() AS rows FROM splits")
                        if r4 and r4.get('last_ts'):
                            first_ts = first_ts or r4.get('first_ts')
                            last_ts = last_ts or r4.get('last_ts')
                            rows = rows or int(r4.get('rows') or 0)
                    except Exception:
                        pass
                    try:
                        r5 = await q.fetchrow("SELECT min(date) AS first_ts, max(date) AS last_ts, count() AS rows FROM earnings_calendar")
                        if r5 and r5.get('last_ts'):
                            first_ts = first_ts or r5.get('first_ts')
                            last_ts = last_ts or r5.get('last_ts')
                            rows = rows or int(r5.get('rows') or 0)
                    except Exception:
                        pass
                out['calendar'] = {'first_ts': first_ts, 'last_ts': last_ts, 'rows': rows}
            except Exception:
                out['calendar'] = {'first_ts': None, 'last_ts': None, 'rows': 0}
            return out
    except Exception:
        pass
    # HTTP fallback via /exec
    try:
        import httpx
        base = os.getenv('DB_QUESTDB_HTTP_URL') or os.getenv('QUESTDB_HTTP_URL') or f"http://{os.getenv('DB_QUESTDB_HOST', os.getenv('QUESTDB_HOST','trading-questdb'))}:{os.getenv('DB_QUESTDB_HTTP_PORT', os.getenv('QUESTDB_HTTP_PORT','9000'))}/exec"
        async with httpx.AsyncClient(timeout=2.0) as client:
            out: Dict[str, Any] = {}
            async def q(sql: str) -> Dict[str, Any]:
                r = await client.get(base, params={'query': sql})
                if r.status_code != 200:
                    return {}
                js = r.json()
                cols = {c['name']: i for i,c in enumerate(js.get('columns', []))}
                ds = js.get('dataset') or []
                row = ds[0] if ds else []
                def _get(name: str):
                    idx = cols.get(name)
                    return row[idx] if idx is not None and idx < len(row) else None
                return {'first_ts': _get('first_ts') or _get('min'), 'last_ts': _get('last_ts') or _get('max'), 'rows': (_get('rows') or _get('count') or 0)}
            out['equities'] = await q("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows FROM market_data")
            op = await q("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows FROM options_data")
            out['options'] = op | {'contracts': 0, 'underlyings': 0}
            # news
            news = await q("SELECT min(ts) AS first_ts, max(ts) AS last_ts, count() AS rows FROM news_items")
            if not news.get('rows'):
                news = await q("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows FROM news_events")
            out['news'] = news
            # social
            social = await q("SELECT min(ts) AS first_ts, max(ts) AS last_ts, count() AS rows FROM social_signals")
            if not social.get('rows'):
                social = await q("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows FROM social_events")
            out['social'] = social
            # calendar
            cal = await q("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows FROM calendar_events")
            if not cal.get('rows'):
                cal2 = await q("SELECT min(timestamp) AS first_ts, max(timestamp) AS last_ts, count() AS rows FROM corporate_actions")
                for k in ('first_ts','last_ts','rows'):
                    cal[k] = cal.get(k) or cal2.get(k)
            out['calendar'] = cal
            return out
    except Exception:
        pass
    return {}

async def load_retention_metrics() -> Dict[str, Any]:
    """Load lightweight retention metrics.

    Strategy:
      - Attempt to read latest on-disk retention audit JSON artifacts if present
        (names like retention_*_questdb.json under repo root or /srv/).
      - If not found, return status 'missing'. This keeps /health/full fast.
    """
    import glob
    candidates = []
    try:
        # Look for recent retention artifacts produced by scripts or scheduled jobs
        for pat in (
            '/srv/ai-trading-system/retention_*_questdb.json',
            '/srv/retention_*_questdb.json',
            '/srv/ai-trading-system/storage_projection_*.json',
        ):
            candidates.extend(glob.glob(pat))
        # Pick the newest by mtime
        if candidates:
            newest = max(candidates, key=lambda p: os.path.getmtime(p))
            with open(newest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Normalize to minimal schema
            out: Dict[str, Any] = {'status': 'available', 'source': newest}
            # If this is a specific retention audit (violations by table)
            v = data.get('violations') if isinstance(data, dict) else None
            if isinstance(v, dict):
                out['violations'] = v
                out['summary'] = {'tables_with_violations': [k for k,val in v.items() if isinstance(val, dict) and (val.get('rows') or 0) > 0]}
                out['status'] = 'fail' if out['summary']['tables_with_violations'] else 'ok'
            else:
                out['raw'] = data
            return out
    except Exception:
        # fall through to missing
        pass
    return {'status': 'missing'}

async def compute_coverage() -> Dict[str, Any]:
    """Compose a minimal coverage snapshot using artifacts and QuestDB.

    Returns structure used by API health and business endpoints:
    {
      status: 'ok'|'degraded'|'error',
      latest: { equities|options|news|social: ISO8601 strings },
      ratios: { equities_total, options_total, news_total, social_total } (best-effort)
    }
    """
    status = 'ok'
    latest: Dict[str, Any] = {}
    ratios: Dict[str, Any] = {}

    # Prefer artifacts written by ingestion service
    artifacts_root = os.getenv('GRAFANA_CSV_DIR', '/mnt/fastdrive/trading/grafana/csv').rstrip('/')
    eq_art = os.path.join(artifacts_root, 'equities_coverage.json')
    op_art = os.path.join(artifacts_root, 'options_coverage.json')

    # QuestDB summaries for equities/options/news/social latest timestamps
    qdb = await _qdb_summary()
    try:
        eq_last = qdb.get('equities', {}).get('last_ts')
        if eq_last:
            latest['equities'] = str(eq_last)
    except Exception:
        pass
    try:
        op_last = qdb.get('options', {}).get('last_ts')
        if op_last:
            latest['options'] = str(op_last)
    except Exception:
        pass
    try:
        nw_last = qdb.get('news', {}).get('last_ts')
        if nw_last:
            latest['news'] = str(nw_last)
    except Exception:
        pass
    try:
        so_last = qdb.get('social', {}).get('last_ts')
        if so_last:
            latest['social'] = str(so_last)
    except Exception:
        pass
    try:
        cal_last = qdb.get('calendar', {}).get('last_ts')
        if cal_last:
            latest['calendar'] = str(cal_last)
    except Exception:
        pass

    # Attempt to read artifact ratios
    try:
        if os.path.isfile(eq_art):
            with open(eq_art, 'r') as f:
                data = json.load(f)
            # Support either top-level ratios or explicit fields
            r = data.get('ratios') if isinstance(data, dict) else None
            if isinstance(r, dict):
                ratios['equities_total'] = r.get('equities_total') or r.get('equities')
            if 'last_date' in data:
                latest['equities'] = data.get('last_date')
    except Exception:
        pass
    try:
        if os.path.isfile(op_art):
            with open(op_art, 'r') as f:
                data = json.load(f)
            cov_list = data.get('coverage') or []
            # Derive a crude ratio: fraction of underlyings with non-zero rows
            if isinstance(cov_list, list) and cov_list:
                total = len([x for x in cov_list if isinstance(x, dict)])
                nonzero = len([x for x in cov_list if isinstance(x, dict) and (x.get('rows') or 0) > 0])
                if total > 0:
                    ratios['options_total'] = round(nonzero / total, 3)
                # Latest options date across entries
                try:
                    last_dates = [x.get('last_day') for x in cov_list if x.get('last_day')]
                    if last_dates:
                        latest['options'] = max(last_dates)
                except Exception:
                    pass
    except Exception:
        pass

    # Derive rough ratios for news/social based on row presence (treat presence as partial completeness)
    try:
        n_rows = qdb.get('news', {}).get('rows') or 0
        if isinstance(n_rows, int):
            # Heuristic: assume 5y target; map to [0.0, 1.0] with a soft cap
            # If there are any rows, mark at least 0.1 to differentiate from zero
            ratios['news_total'] = 1.0 if n_rows > 1_000_000 else (0.1 if n_rows > 0 else 0.0)
    except Exception:
        pass
    try:
        s_rows = qdb.get('social', {}).get('rows') or 0
        if isinstance(s_rows, int):
            ratios['social_total'] = 1.0 if s_rows > 1_000_000 else (0.1 if s_rows > 0 else 0.0)
    except Exception:
        pass
    try:
        c_rows = qdb.get('calendar', {}).get('rows') or 0
        if isinstance(c_rows, int):
            ratios['calendar_total'] = 1.0 if c_rows > 100_000 else (0.1 if c_rows > 0 else 0.0)
    except Exception:
        pass

    return {
        'status': status,
        'generated_at': datetime.utcnow().isoformat(),
        'latest': latest,
        'ratios': ratios,
    }
# NOTE: A second, more complex implementation previously lived in this file and
# incorrectly referenced non-existent tables (equity_bars/option_bars) and different
# client interfaces. It has been removed to avoid overriding these minimal, correct
# helpers that align with our persisted tables market_data/options_data and the
# ingestion artifacts.

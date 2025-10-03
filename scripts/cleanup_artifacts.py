#!/usr/bin/env python3
"""
Safe artifacts cleanup utility (dry-run by default)

Removes old CSV/JSON/log files from well-known artifact directories while respecting
retention windows. Designed for cron/compose job usage. Prints a summary report.

Env vars:
  CLEAN_DRY_RUN          true|false (default true)
  RETENTION_DAYS        integer days (default 14)
  EXTRA_RETENTION_DAYS  integer days for large dirs (default 30)
  PATHS                 comma list of directories to scan (default preconfigured)
  GLOB_INCLUDE          comma list of glob patterns to include (default: *.csv,*.json,*.log,*.txt)
  GLOB_EXCLUDE          comma list of glob patterns to exclude (default: *.keep,*.md)

Exit codes: 0 on success
"""
from __future__ import annotations

import os
import sys
import time
import fnmatch
from dataclasses import dataclass
from typing import Iterable, List


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, '').strip().lower()
    if not v:
        return default
    return v in ('1','true','yes','on')


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, '').strip() or default)
    except Exception:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name, '').strip()
    if not raw:
        return default
    return [x.strip() for x in raw.split(',') if x.strip()]


@dataclass
class Stats:
    examined: int = 0
    matched: int = 0
    deleted: int = 0
    bytes_freed: int = 0


def glob_match(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def cleanup_dir(path: str, include: List[str], exclude: List[str], before_epoch: float, dry_run: bool, stats: Stats):
    if not os.path.isdir(path):
        return
    for root, dirs, files in os.walk(path):
        # Skip hidden directories like .git, .cache
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            stats.examined += 1
            # quick excludes
            if f.startswith('.'):
                continue
            if glob_match(f, exclude):
                continue
            if not glob_match(f, include):
                continue
            fp = os.path.join(root, f)
            try:
                st = os.stat(fp)
            except Exception:
                continue
            if st.st_mtime >= before_epoch:
                continue
            stats.matched += 1
            if not dry_run:
                try:
                    os.remove(fp)
                    stats.deleted += 1
                    stats.bytes_freed += st.st_size
                except Exception:
                    # Non-fatal
                    pass


def main():
    dry_run = _env_bool('CLEAN_DRY_RUN', True)
    retention_days = _env_int('RETENTION_DAYS', 14)
    extra_retention_days = _env_int('EXTRA_RETENTION_DAYS', 30)
    include = _env_list('GLOB_INCLUDE', ['*.csv','*.json','*.log','*.txt'])
    exclude = _env_list('GLOB_EXCLUDE', ['*.keep','*.md'])
    paths = _env_list('PATHS', [
        '/mnt/fastdrive/trading/grafana/csv',
        '/mnt/fastdrive/trading/grafana/png',
        '/mnt/fastdrive/trading/pulsar-logs',
        '/srv/ai-trading-system',
    ])

    now = time.time()
    before_main = now - retention_days * 86400
    before_large = now - extra_retention_days * 86400

    stats = Stats()
    for p in paths:
        # Use longer retention on big dirs like pulsar logs
        cutoff = before_large if any(seg in p for seg in ('pulsar','prometheus','weaviate')) else before_main
        cleanup_dir(p, include, exclude, cutoff, dry_run, stats)

    summary = {
        'dry_run': dry_run,
        'retention_days': retention_days,
        'extra_retention_days': extra_retention_days,
        'paths': paths,
        'examined': stats.examined,
        'matched': stats.matched,
        'deleted': stats.deleted,
        'bytes_freed': stats.bytes_freed,
    }
    import json
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('{"status":"cancelled"}')
        raise SystemExit(130)

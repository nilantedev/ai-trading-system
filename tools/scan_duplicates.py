#!/usr/bin/env python3
"""Scan repository for duplicate large files (>=100KB) by SHA256 hash.
Outputs JSON report listing groups of duplicates with paths and sizes.
Excludes common transient dirs (.__pycache__, .git, logs runtime files) and virtual environment patterns.
"""
from __future__ import annotations
import os, sys, hashlib, json, stat
from pathlib import Path

EXCLUDE_DIR_NAMES = {'.git', '__pycache__', '.mypy_cache', '.pytest_cache', '.ruff_cache', 'node_modules', '.venv', 'venv', 'env', '.idea', '.vscode', 'logs'}
MIN_SIZE = 100 * 1024  # 100KB threshold to reduce noise

ROOT = Path(__file__).resolve().parent.parent

hash_map: dict[str, list[dict]] = {}

for root, dirs, files in os.walk(ROOT):
    # Prune excluded dirs in-place
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]
    for fname in files:
        path = Path(root) / fname
        try:
            if not path.is_file():
                continue
            st = path.stat()
            if st.st_size < MIN_SIZE:
                continue
            # Skip obvious build artifacts or large data directories
            if any(seg in ('prometheus', 'grafana', 'postgres', 'questdb', 'pulsar', 'weaviate', 'minio') for seg in path.parts):
                continue
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            digest = h.hexdigest()
            hash_map.setdefault(digest, []).append({
                'path': str(path.relative_to(ROOT)),
                'size_bytes': st.st_size
            })
        except (OSError, PermissionError):
            continue

# Build report of duplicates (more than one file with same hash)
duplicates = [v for v in hash_map.values() if len(v) > 1]
report = {
    'root': str(ROOT),
    'threshold_bytes': MIN_SIZE,
    'duplicate_groups': duplicates,
    'total_groups': len(duplicates),
}
print(json.dumps(report, indent=2))

#!/usr/bin/env python3
"""Model & Feature Reproducibility Utilities.
Provides hashing helpers for datasets, feature graphs, configs, and git metadata extraction.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from datetime import datetime
from typing import Iterable, List, Dict, Any, Optional
import pandas as pd

_HASH_BLOCK_SIZE = 65536


def _sha256_iter(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def git_commit_hash(fallback_env: str = "GIT_COMMIT") -> str:
    """Attempt to retrieve current git commit hash; fall back to env or timestamp."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        if commit:
            return commit
    except Exception:
        pass
    return os.getenv(fallback_env, f"no-git-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")


def hash_training_config(cfg: Dict[str, Any]) -> str:
    """Hash a training config dict (stable ordering)."""
    stable_json = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(stable_json.encode()).hexdigest()[:16]


def hash_feature_definitions(feature_defs: List[Dict[str, Any]]) -> str:
    """Hash list of feature definition dicts (name + version + dependencies + logic)."""
    canonical = []
    for fd in feature_defs:
        canonical.append({
            "name": fd.get("name"),
            "version": fd.get("version"),
            "dependencies": sorted(fd.get("dependencies", [])),
            "logic": fd.get("transformation_logic")
        })
    stable_json = json.dumps(sorted(canonical, key=lambda x: x["name"]), sort_keys=True)
    return hashlib.sha256(stable_json.encode()).hexdigest()[:16]


def hash_dataset(df: pd.DataFrame, feature_cols: Optional[List[str]] = None, max_rows: int = 0) -> str:
    """Create deterministic hash of dataset contents (subset for scalability).
    Args:
        df: DataFrame with at least entity_id + timestamp + features
        feature_cols: restrict to these columns if provided
        max_rows: if >0, sample head and tail windows to bound cost
    """
    if feature_cols:
        subset = df[feature_cols].copy()
    else:
        subset = df.copy()
    # Stable ordering
    if "timestamp" in subset.columns:
        subset = subset.sort_values(by=["timestamp", "entity_id"], errors="ignore")
    if max_rows and len(subset) > max_rows:
        head_n = max_rows // 2
        tail_n = max_rows - head_n
        subset = pd.concat([subset.head(head_n), subset.tail(tail_n)])
    # Convert to CSV bytes (no index)
    csv_bytes = subset.to_csv(index=False).encode()
    return hashlib.sha256(csv_bytes).hexdigest()[:16]


def build_repro_manifest(**kwargs) -> Dict[str, Any]:
    """Create a manifest dict to embed into model artifact for provenance."""
    manifest = {k: v for k, v in kwargs.items() if v is not None}
    manifest["generated_at"] = datetime.utcnow().isoformat()
    return manifest

__all__ = [
    "git_commit_hash",
    "hash_training_config",
    "hash_feature_definitions",
    "hash_dataset",
    "build_repro_manifest",
]

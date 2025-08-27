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
import platform
import importlib.metadata as importlib_metadata

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
    """Create an enriched manifest dict to embed alongside a model artifact for provenance.

    Backwards-compatible: accepts arbitrary kwargs and adds structured sections:
      - schema_version
      - id (composite identifier: model:version:gitshort:traincfg)
      - environment (python/platform/git)
      - artifact (path,size,sha256) if artifact_path provided
      - dependencies (critical ML lib versions)
      - data_window (train_start/end) if present in config
    """
    manifest: Dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}
    manifest["schema_version"] = "1.0"
    manifest["generated_at"] = datetime.utcnow().isoformat()

    model_name = manifest.get("model_name") or manifest.get("config", {}).get("model_name")
    version = manifest.get("version") or manifest.get("config", {}).get("version")
    git_commit = manifest.get("git_commit") or git_commit_hash()
    training_config_hash = manifest.get("training_config_hash")
    short_commit = git_commit[:7] if isinstance(git_commit, str) else "unknown"
    if model_name and version and training_config_hash:
        manifest["id"] = f"{model_name}:{version}:{short_commit}:{training_config_hash[:8]}"

    manifest["environment"] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit,
        "git_short": short_commit,
    }

    critical_packages = ["numpy", "pandas", "scikit-learn", "torch", "transformers", "sentence-transformers"]
    deps: Dict[str, str] = {}
    for pkg in critical_packages:
        try:
            deps[pkg] = importlib_metadata.version(pkg)
        except Exception:  # pragma: no cover
            continue
    if deps:
        manifest["dependencies"] = deps

    artifact_path = manifest.get("artifact_path") or manifest.get("artifact")
    if isinstance(artifact_path, str):
        try:
            size = os.path.getsize(artifact_path)
            with open(artifact_path, "rb") as f:
                h = hashlib.sha256()
                while True:
                    chunk = f.read(_HASH_BLOCK_SIZE)
                    if not chunk:
                        break
                    h.update(chunk)
            manifest["artifact"] = {"path": artifact_path, "size_bytes": size, "sha256": h.hexdigest()}
        except Exception as e:  # pragma: no cover
            manifest.setdefault("warnings", []).append(f"artifact_hash_failed: {e}")

    cfg = manifest.get("config") or {}
    if isinstance(cfg, dict) and cfg.get("train_start") and cfg.get("train_end"):
        manifest["data_window"] = {"train_start": cfg.get("train_start"), "train_end": cfg.get("train_end")}

    return manifest

__all__ = [
    "git_commit_hash",
    "hash_training_config",
    "hash_feature_definitions",
    "hash_dataset",
    "build_repro_manifest",
]

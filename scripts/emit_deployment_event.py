#!/usr/bin/env python3
"""Emit a structured deployment event for auditing.

This script intentionally has no external dependencies beyond the project.
It attempts to import the existing structured event emitter; if unavailable,
falls back to stdout JSON.

Usage:
  python scripts/emit_deployment_event.py --version <version> \
      [--sbom-hash HASH] [--env-hash HASH] [--status success|failed]
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib, time
from datetime import datetime


def _emit_json(payload: dict):
    print(json.dumps(payload, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--sbom-hash", default="n/a")
    parser.add_argument("--env-hash", default="n/a")
    parser.add_argument("--status", default="success")
    parser.add_argument("--release-dir", default=".")
    args = parser.parse_args()

    event = {
        "type": "deployment.release",
        "version": args.version,
        "sbom_hash": args.sbom_hash,
        "env_hash": args.env_hash,
        "status": args.status,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Try structured emitter
    try:
        from trading_common.event_logging import emit_event  # type: ignore
        emit_event("deployment.release", {
            "version": args.version,
            "sbom_hash": args.sbom_hash,
            "env_hash": args.env_hash,
            "status": args.status,
        })
    except Exception:
        _emit_json(event)

    # Write metadata file for future audit
    meta_path = os.path.join(args.release_dir, "deployment.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(event, f, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to write deployment metadata: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

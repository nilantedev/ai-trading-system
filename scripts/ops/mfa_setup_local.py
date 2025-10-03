#!/usr/bin/env python3
"""
Local MFA setup helper (ops-only)

Usage examples:
  python scripts/ops/mfa_setup_local.py --username nilante --api-url http://localhost:8000 --qr-out qr.png

This calls /auth/admin/ops/mfa_setup_local which is IP-restricted to localhost/private ranges.
It prints the secret and otpauth URL, and optionally writes a QR PNG if provided by the API.
"""
from __future__ import annotations
import argparse
import base64
import sys
from pathlib import Path

import httpx


def main() -> int:
    p = argparse.ArgumentParser(description="Generate MFA secret + QR for a user (local ops)")
    p.add_argument("--username", "-u", default="nilante", help="Username to setup MFA for (default: nilante)")
    p.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the API service")
    p.add_argument("--qr-out", default=None, help="Optional path to write QR PNG image")
    args = p.parse_args()

    url = args.api_url.rstrip("/") + "/auth/admin/ops/mfa_setup_local"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(url, json={"username": args.username})
    except Exception as e:
        print(f"ERROR: Request failed: {e}")
        return 2

    if r.status_code != 200:
        try:
            j = r.json()
            detail = j.get("detail") if isinstance(j, dict) else None
        except Exception:
            detail = r.text
        print(f"ERROR: API responded with {r.status_code}: {detail}")
        if r.status_code == 403:
            print("Hint: This endpoint only works from localhost/private IPs. Run inside the API container or on the host network.")
        return 3

    j = r.json()
    secret = j.get("secret")
    otpauth = j.get("otpauth_url")
    qr_b64 = j.get("qr_png_base64")
    print("OK: MFA secret issued")
    print(f"  username : {j.get('username')}")
    print(f"  secret   : {secret}")
    print(f"  otpauth  : {otpauth}")

    if args.qr_out and qr_b64:
        try:
            data = base64.b64decode(qr_b64)
            out = Path(args.qr_out)
            out.write_bytes(data)
            print(f"  qr_png   : written to {out.resolve()}")
        except Exception as e:
            print(f"WARN: Failed to write QR PNG: {e}")

    # Provide next steps
    print("\nNext steps:")
    print("  1) Scan the QR with your authenticator app (Google Authenticator, Authy, etc.)")
    print("  2) Log in and enter your 6-digit code when prompted")
    print("  3) In Admin → Security Tools, regenerate backup codes and store them securely")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

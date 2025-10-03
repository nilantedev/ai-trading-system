#!/usr/bin/env python3
"""ML proxy endpoints exposed via API.

Routes:
 - POST /api/v1/analysis/network -> forwards to ML /analyze/network
 - POST /api/v1/backtests/intelligent -> forwards to ML /backtest/intelligent

Behavior:
 - Preserves request/response JSON (including data_availability)
 - Requires business auth (token cookie or bearer) with MFA
"""
from __future__ import annotations

import os
from typing import Any
from fastapi import APIRouter, Request, HTTPException, Depends

# Avoid circular import with api.main; implement our own strict auth dependency
from api.auth import get_auth_manager, TokenType  # type: ignore[import-not-found]

router = APIRouter(prefix="/api/v1", tags=["ml-proxy"])


def _ml_base() -> str:
    return os.getenv("ML_SERVICE_URL", "http://trading-ml:8001").rstrip("/")


async def _require_business_auth(request: Request):
    """Strict auth for business endpoints.
    Accepts bearer Authorization header or cookie 'at' (access token).
    Verifies token and enforces MFA-enabled user when possible.
    Returns minimal user context dict.
    """
    # 1) Extract token from Authorization header or cookie
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    token = None
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(None, 1)[1].strip()
    else:
        token = request.cookies.get("at")  # access token cookie set by login
    if not token:
        raise HTTPException(status_code=401, detail="not authenticated")
    # 2) Verify access token
    auth_manager = await get_auth_manager()
    try:
        payload = await auth_manager.verify_token(token, token_type=TokenType.ACCESS)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="invalid token")
    user_id = payload.get("sub") or ""
    username = payload.get("username") or ""
    roles = payload.get("roles") or []
    # 3) MFA enforcement: if configured, require user to have MFA enabled
    try:
        if auth_manager.enforce_mfa_for_admin:
            # If admin role present, MFA must be enabled
            if ("admin" in roles) and (not await auth_manager.mfa_is_enabled(user_id)):
                raise HTTPException(status_code=401, detail="mfa required for admin")
        # General MFA enforcement for business endpoints (best-effort)
        if os.getenv("REQUIRE_MFA_SENSITIVE", "true").lower() in ("1","true","yes","on"):
            if not await auth_manager.mfa_is_enabled(user_id):
                # If user hasn't enabled MFA, block access
                raise HTTPException(status_code=401, detail="mfa required")
    except HTTPException:
        raise
    except Exception:
        # Do not leak internal errors
        raise HTTPException(status_code=401, detail="authentication failure")
    return {"user_id": user_id, "username": username, "roles": roles}


@router.post("/analysis/network")
async def proxy_analysis_network(request: Request, user=Depends(_require_business_auth)) -> Any:
    import httpx
    body: Any
    try:
        body = await request.json()
    except Exception:
        body = None
    # Accept either array of symbols or object with symbols
    symbols = None
    analysis_type = request.query_params.get("analysis_type") or "correlation"
    if isinstance(body, list):
        symbols = body
        payload = body
        headers = {"content-type": "application/json"}
    elif isinstance(body, dict):
        symbols = body.get("symbols")
        payload = symbols if isinstance(symbols, list) else []
        headers = {"content-type": "application/json"}
        if body.get("analysis_type"):
            analysis_type = body.get("analysis_type")
    else:
        payload = []
        headers = {"content-type": "application/json"}
    if not isinstance(payload, list) or not payload:
        raise HTTPException(status_code=400, detail="symbols required")
    url = f"{_ml_base()}/analyze/network?analysis_type={analysis_type}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, json=payload, headers=headers)
    if r.status_code != 200:
        # Bubble up ML error but with API semantics
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:200]
        raise HTTPException(status_code=r.status_code, detail=detail)
    # Pass through JSON (includes data_availability)
    return r.json()


@router.post("/backtests/intelligent")
async def proxy_backtest_intelligent(request: Request, user=Depends(_require_business_auth)) -> Any:
    import httpx
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    url = f"{_ml_base()}/backtest/intelligent"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers={"content-type": "application/json"})
    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:200]
        raise HTTPException(status_code=r.status_code, detail=detail)
    return r.json()

__all__ = ["router"]

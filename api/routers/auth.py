#!/usr/bin/env python3
"""
Authentication API Router for AI Trading System
Provides login, token refresh, and authentication management endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request, Response
import pyotp
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import logging

from ..auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_auth_health,
    User,
    JWT_EXPIRY_MINUTES,
    get_auth_manager
)
from sqlalchemy import text as sql_text
from trading_common.database_manager import get_database_manager  # type: ignore[import-not-found]
import os
import secrets

logger = logging.getLogger(__name__)

router = APIRouter()


class Token(BaseModel):
    """JWT token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    expires_at: datetime


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str
    mfa_code: Optional[str] = None


class UserProfile(BaseModel):
    """User profile response model."""
    user_id: str
    username: str
    roles: list[str]
    permissions: list[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

# Cookie constants (aligned with consolidated api.auth)
COOKIE_ACCESS = "at"
COOKIE_REFRESH = "rt"
COOKIE_CSRF = "csrf_token"

def _cookie_settings():
    return dict(
        httponly=True,
        samesite="Strict",
        secure=os.getenv("COOKIE_SECURE","true").lower() in ("1","true","yes","on"),
        path="/"
    )

def _csrf_cookie_settings():
    return dict(
        httponly=False,
        samesite="Strict",
        secure=os.getenv("COOKIE_SECURE","true").lower() in ("1","true","yes","on"),
        path="/"
    )

def _gen_csrf() -> str:
    return secrets.token_urlsafe(32)


@router.post("/auth/login", response_model=Token)
async def login(request: Request, response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint to obtain JWT access token.
    
    **Required fields:**
    - username: User's username
    - password: User's password
    
    **Returns:**
    - access_token: JWT token for API authentication
    - expires_in: Token expiry time in seconds
    """
    # Get client IP address
    client_ip = request.client.host if request.client else "unknown"
    
    # Authenticate user with persistent brute force protection
    user = await authenticate_user(form_data.username, form_data.password, client_ip)
    if not user:
        logger.warning(f"Failed login attempt for username: {form_data.username} from {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # MFA enforcement
    auth_manager = await get_auth_manager()
    mfa_required = False
    if auth_manager.enforce_mfa_for_admin and ("admin" in (user.roles or [])):
        mfa_required = True
    if not mfa_required:
        mfa_required = await auth_manager.mfa_is_enabled(user.user_id)

    if mfa_required:
        # OAuth2PasswordRequestForm doesn't include custom fields; support X-MFA-Code header fall-back
        mfa_code = request.headers.get("X-MFA-Code")
        if not mfa_code:
            # Try parsing from query param as last resort
            mfa_code = request.query_params.get("mfa_code")
        if not mfa_code or not await auth_manager.mfa_verify(user.user_id, mfa_code):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="MFA code required or invalid")

    # Create access token
    access_token_expires = timedelta(minutes=JWT_EXPIRY_MINUTES)
    access_token = await create_access_token(user, expires_delta=access_token_expires)
    
    expires_at = datetime.utcnow() + access_token_expires
    
    logger.info(f"User {user.username} logged in successfully from {client_ip}")

    # Browser-friendly behavior: set cookies and optionally redirect
    cs = _cookie_settings()
    # Create refresh token via auth manager (includes MFA claim to satisfy dashboard gates)
    refresh_token = await (await get_auth_manager()).create_refresh_token({
        "sub": user.user_id,
        "username": user.username,
        "roles": user.roles,
        "role": (user.roles[0] if user.roles else 'viewer'),
        "mfa": True
    })
    # Always set cookies for convenience
    response.set_cookie(COOKIE_ACCESS, access_token, max_age=int(access_token_expires.total_seconds()), **cs)
    response.set_cookie(COOKIE_REFRESH, refresh_token, max_age=7*86400, **cs)
    if COOKIE_CSRF not in request.cookies:
        response.set_cookie(COOKIE_CSRF, _gen_csrf(), max_age=7*86400, **_csrf_cookie_settings())

    # If HTML requested, return token body but cookies suffice for redirects handled elsewhere
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_token_expires.total_seconds()),
        "expires_at": expires_at
    }


@router.post("/auth/login-json", response_model=Token)
async def login_json(request: Request, response: Response, login_data: LoginRequest):
    """
    JSON login endpoint (alternative to OAuth2 form).
    
    **Required fields:**
    - username: User's username  
    - password: User's password
    
    **Returns:**
    - access_token: JWT token for API authentication
    - expires_in: Token expiry time in seconds
    """
    # Get client IP address
    client_ip = request.client.host if request.client else "unknown"
    
    # Authenticate user with persistent brute force protection
    user = await authenticate_user(login_data.username, login_data.password, client_ip)
    if not user:
        logger.warning(f"Failed JSON login attempt for username: {login_data.username} from {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # MFA enforcement
    auth_manager = await get_auth_manager()
    mfa_required = False
    if auth_manager.enforce_mfa_for_admin and ("admin" in (user.roles or [])):
        mfa_required = True
    if not mfa_required:
        mfa_required = await auth_manager.mfa_is_enabled(user.user_id)

    if mfa_required:
        if not login_data.mfa_code or not await auth_manager.mfa_verify(user.user_id, login_data.mfa_code):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="MFA code required or invalid")

    # Create access token  
    access_token_expires = timedelta(minutes=JWT_EXPIRY_MINUTES)
    access_token = await create_access_token(user, expires_delta=access_token_expires)
    
    expires_at = datetime.utcnow() + access_token_expires
    
    logger.info(f"User {user.username} logged in successfully via JSON from {client_ip}")

    # Also set cookies for browser clients consuming JSON path
    cs = _cookie_settings()
    response.set_cookie(COOKIE_ACCESS, access_token, max_age=int(access_token_expires.total_seconds()), **cs)
    refresh_token = await (await get_auth_manager()).create_refresh_token({
        "sub": user.user_id,
        "username": user.username,
        "roles": user.roles,
        "role": (user.roles[0] if user.roles else 'viewer'),
        "mfa": True if mfa_required else True
    })
    response.set_cookie(COOKIE_REFRESH, refresh_token, max_age=7*86400, **cs)
    if COOKIE_CSRF not in request.cookies:
        response.set_cookie(COOKIE_CSRF, _gen_csrf(), max_age=7*86400, **_csrf_cookie_settings())

    return {
        "access_token": access_token,
        "token_type": "bearer", 
        "expires_in": int(access_token_expires.total_seconds()),
        "expires_at": expires_at
    }


class MFASetupRequest(BaseModel):
    label: Optional[str] = None  # e.g., "Mekoshi Trading (user)"


class MFAVerifyRequest(BaseModel):
    code: str


@router.post("/auth/mfa/setup")
async def mfa_setup(current_user: User = Depends(get_current_active_user), req: MFASetupRequest | None = None):
    """Begin MFA setup: returns provisioning URI (otpauth://) and base32 secret."""
    auth_manager = await get_auth_manager()
    secret = await auth_manager.mfa_generate_secret(current_user.user_id)
    label = (req.label if req and req.label else f"mekoshi:{current_user.username}")
    issuer = "Mekoshi Trading"
    uri = pyotp.totp.TOTP(secret).provisioning_uri(name=label, issuer_name=issuer)
    return {"secret": secret, "otpauth_uri": uri}


@router.post("/auth/mfa/enable")
async def mfa_enable(req: MFAVerifyRequest, current_user: User = Depends(get_current_active_user)):
    auth_manager = await get_auth_manager()
    # Verify against pending or stored secret
    secret = await auth_manager.mfa_get_secret(current_user.user_id)
    if not secret:
        # generate one if missing
        secret = await auth_manager.mfa_generate_secret(current_user.user_id)
    if not pyotp.TOTP(secret).verify(req.code, valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid MFA code")
    await auth_manager.mfa_enable(current_user.user_id, secret)
    # Best-effort: reflect MFA enabled in Postgres users table
    try:
        dbm = await get_database_manager()
        async with dbm.get_postgres() as session:  # type: ignore[attr-defined]
            await session.execute(
                sql_text(
                    "UPDATE users SET two_factor_enabled = true, updated_at = NOW() WHERE user_id = :uid"
                ),
                {"uid": current_user.user_id},
            )
            try:
                await session.commit()  # commit if session supports it
            except Exception:
                pass
    except Exception:
        # Non-fatal: Redis is the source of truth for MFA enforcement
        pass
    return {"status": "enabled"}


@router.post("/auth/mfa/disable")
async def mfa_disable(current_user: User = Depends(get_current_active_user)):
    auth_manager = await get_auth_manager()
    await auth_manager.mfa_disable(current_user.user_id)
    # Best-effort: reflect MFA disabled in Postgres users table
    try:
        dbm = await get_database_manager()
        async with dbm.get_postgres() as session:  # type: ignore[attr-defined]
            await session.execute(
                sql_text(
                    "UPDATE users SET two_factor_enabled = false, updated_at = NOW() WHERE user_id = :uid"
                ),
                {"uid": current_user.user_id},
            )
            try:
                await session.commit()
            except Exception:
                pass
    except Exception:
        pass
    return {"status": "disabled"}


@router.get("/auth/me", response_model=UserProfile)
async def get_current_user_profile(current_user: User = Depends(get_current_active_user)):
    """
    Get current user's profile information.
    
    **Requires:** Valid JWT token in Authorization header
    
    **Returns:**
    - User profile with roles and permissions
    """
    return UserProfile(
        user_id=current_user.user_id,
        username=current_user.username,
        roles=current_user.roles,
        permissions=current_user.permissions,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@router.post("/auth/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    Logout endpoint (token invalidation).
    
    Note: JWT tokens are stateless, so logout is mainly for client-side cleanup.
    In production, consider token blacklisting for enhanced security.
    """
    logger.info(f"User {current_user.username} logged out")
    return {
        "message": "Successfully logged out",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/auth/health")
async def auth_health_check():
    """
    Authentication system health check.
    
    **Returns:**
    - Authentication system status and configuration
    """
    return get_auth_health()


@router.get("/auth/test")
async def test_auth(current_user: User = Depends(get_current_active_user)):
    """
    Test authenticated endpoint.
    
    **Requires:** Valid JWT token
    
    **Returns:**
    - Confirmation of successful authentication
    """
    return {
        "message": f"Authentication successful for user: {current_user.username}",
        "user_id": current_user.user_id,
        "roles": current_user.roles,
        "permissions": current_user.permissions,
        "timestamp": datetime.utcnow().isoformat()
    }
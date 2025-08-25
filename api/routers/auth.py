#!/usr/bin/env python3
"""
Authentication API Router for AI Trading System
Provides login, token refresh, and authentication management endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
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
    JWT_EXPIRY_MINUTES
)

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


class UserProfile(BaseModel):
    """User profile response model."""
    user_id: str
    username: str
    roles: list[str]
    permissions: list[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]


@router.post("/auth/login", response_model=Token)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
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
    
    # Create access token
    access_token_expires = timedelta(minutes=JWT_EXPIRY_MINUTES)
    access_token = create_access_token(user, expires_delta=access_token_expires)
    
    expires_at = datetime.utcnow() + access_token_expires
    
    logger.info(f"User {user.username} logged in successfully from {client_ip}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_token_expires.total_seconds()),
        "expires_at": expires_at
    }


@router.post("/auth/login-json", response_model=Token)
async def login_json(request: Request, login_data: LoginRequest):
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
    
    # Create access token  
    access_token_expires = timedelta(minutes=JWT_EXPIRY_MINUTES)
    access_token = create_access_token(user, expires_delta=access_token_expires)
    
    expires_at = datetime.utcnow() + access_token_expires
    
    logger.info(f"User {user.username} logged in successfully via JSON from {client_ip}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer", 
        "expires_in": int(access_token_expires.total_seconds()),
        "expires_at": expires_at
    }


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
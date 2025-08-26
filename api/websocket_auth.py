"""WebSocket authentication and authorization helpers.

Centralizes token extraction, verification, and stream permission policy so the
router code stays slim. Integrates with existing JWT auth logic in api.auth.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from fastapi import WebSocket

from trading_common import get_logger
from api.auth import verify_access_token, UserRole, user_manager

logger = get_logger(__name__)

# Stream access policy: which streams require auth and minimum role / permission.
STREAM_POLICY = {
    "market_data": {"auth_required": False, "permission": None, "role": None},
    "alerts": {"auth_required": False, "permission": None, "role": None},
    "signals": {"auth_required": True, "permission": "read:market_data", "role": None},
    "orders": {"auth_required": True, "permission": "write:orders", "role": None},
    "portfolio": {"auth_required": True, "permission": "read:portfolio", "role": None},
    "system": {"auth_required": True, "permission": None, "role": UserRole.ADMIN},
}


async def authenticate_websocket(websocket: WebSocket, stream: str, token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Authenticate & authorize a WebSocket connection for a given stream.

    Returns user_info dict (sanitized) or None if unauthenticated (and allowed).
    If not allowed, closes the websocket and returns None.
    """
    policy = STREAM_POLICY.get(stream, {"auth_required": True})
    auth_required = policy.get("auth_required", True)

    if not token:
        if auth_required:
            await websocket.close(code=1008, reason="Authentication required")
            logger.info("WebSocket rejected - missing token", stream=stream)
            return None
        return None  # anonymous allowed

    # Verify token
    try:
        token_data = await verify_access_token(token)
    except Exception as e:  # token invalid
        if auth_required:
            await websocket.close(code=1008, reason="Invalid token")
            logger.info("WebSocket rejected - invalid token", stream=stream, error=str(e))
            return None
        logger.warning("Proceeding anonymously after token failure on non-auth stream", stream=stream)
        return None

    # Get user from user manager
    try:
        user_mgmt_user = await user_manager._get_user_by_username(token_data.username)
        if not user_mgmt_user or user_mgmt_user.status.value != "active":
            if auth_required:
                await websocket.close(code=1008, reason="Inactive user")
                logger.info("WebSocket rejected - inactive user", stream=stream, user=token_data.username)
                return None
            return None

        user_role = user_mgmt_user.role.value
        permissions = list(user_mgmt_user.permissions)

        # Authorization checks
        required_perm = policy.get("permission")
        required_role = policy.get("role")

        if required_perm and required_perm not in permissions and "admin:all" not in permissions:
            await websocket.close(code=1008, reason="Insufficient permission")
            logger.info("WebSocket rejected - missing permission", stream=stream, required=required_perm, user=token_data.username)
            return None
        
        if required_role and user_role not in [required_role.value, UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value]:
            await websocket.close(code=1008, reason="Insufficient role")
            logger.info("WebSocket rejected - missing role", stream=stream, required=str(required_role), user=token_data.username)
            return None

        user_info = {
            "user_id": user_mgmt_user.user_id,
            "username": token_data.username,
            "role": user_role,
            "permissions": permissions,
            "authenticated": True,
        }
        
    except Exception as e:
        logger.error(f"Error getting user for WebSocket auth: {e}")
        if auth_required:
            await websocket.close(code=1008, reason="Authentication error")
            return None
        return None

    logger.info("WebSocket authenticated", stream=stream, user=token_data.username, role=user_role)
    return user_info

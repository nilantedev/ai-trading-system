import asyncio
from typing import Any, Dict, Optional, Literal
from enum import Enum
from dataclasses import dataclass

class SecurityEventType(str, Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    ACCOUNT_LOCKED = "account_locked"
    TOKEN_REFRESH = "token_refresh"


@dataclass
class UserSession:
    user_id: str = "stub"
    username: str = "stub"
    created_at: str = "0"


@dataclass
class RefreshToken:
    token: str = "stub"
    user_id: str = "stub"
    expires_at: str = "0"


class SecurityStore:
    async def get_store_health(self) -> Dict[str, Any]:
        return {"status": "healthy"}
    async def close(self) -> None:  # noqa: D401
        return None
    # stubbed interface methods used by auth
    async def is_account_locked(self, username: str, max_attempts: int) -> bool:
        return False
    async def record_login_attempt(self, username: str, success: bool) -> None:
        return None
    async def store_refresh_token(self, refresh_token: str, data: Dict[str, Any]) -> None:
        return None
    async def get_refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        return None
    async def revoke_refresh_token(self, refresh_token: str) -> None:
        return None


async def log_security_event(event_type: SecurityEventType, success: bool, username: str | None = None, details: Dict[str, Any] | None = None, severity: Literal["INFO", "WARNING", "ERROR"] = "INFO") -> None:  # noqa: D401
    return None

_store: SecurityStore | None = None

async def get_security_store() -> SecurityStore:
    global _store
    if _store is None:
        # simulate async init
        await asyncio.sleep(0)
        _store = SecurityStore()
    return _store

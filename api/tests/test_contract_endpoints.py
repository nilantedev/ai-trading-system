import os
import jwt
import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta, timezone

os.environ.setdefault('ENV', 'test')
from api.main import app  # noqa: E402
from api.auth import get_auth_manager, TokenType


async def _make_access_cookie(username: str, roles: list[str]):
    auth = await get_auth_manager()
    payload = {
        'sub': f'user-{username}',
        'username': username,
        'roles': roles,
        'type': TokenType.ACCESS.value,
        'exp': datetime.now(timezone.utc) + timedelta(minutes=5),
        'iat': datetime.now(timezone.utc),
        'iss': auth.issuer,
        'aud': auth.audience,
        'jti': 'test-jti'
    }
    return jwt.encode(payload, auth.active_key.secret, algorithm=auth.active_key.algorithm, headers={'kid': auth.active_key.kid})  # type: ignore

@pytest.mark.asyncio
async def test_dashboard_unauthenticated_access_denied():
    async with AsyncClient(app=app, base_url="http://test") as client:
        r1 = await client.get('/admin')
        r2 = await client.get('/business')
        assert r1.status_code in (401,403)
        assert r2.status_code in (401,403)

@pytest.mark.asyncio
async def test_dashboard_wrong_user_forbidden():
    token = await _make_access_cookie('alice','admin'.split())
    async with AsyncClient(app=app, base_url="http://test") as client:
        client.cookies.set('at', token)
        r = await client.get('/admin')
        r2 = await client.get('/business')
        # Wrong user should be 403 (username not 'nilante')
        assert r.status_code == 403
        assert r2.status_code == 403

@pytest.mark.asyncio
async def test_dashboard_correct_user_allowed():
    token = await _make_access_cookie('nilante','admin'.split())
    async with AsyncClient(app=app, base_url="http://test") as client:
        client.cookies.set('at', token)
        r = await client.get('/admin')
        r2 = await client.get('/business')
        assert r.status_code == 200
        assert r2.status_code == 200

@pytest.mark.asyncio
async def test_protected_api_endpoints_cookie_auth():
    token = await _make_access_cookie('nilante','admin'.split())
    async with AsyncClient(app=app, base_url="http://test") as client:
        client.cookies.set('at', token)
        a = await client.get('/admin/api/pnl/timeseries')
        b = await client.get('/business/api/coverage/summary')
        assert a.status_code == 200
        assert b.status_code == 200

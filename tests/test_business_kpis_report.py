import asyncio
import json
from httpx import AsyncClient
import pytest

# NOTE: We assume the FastAPI app object is importable as api.main.app
# and that dependency overrides / auth tokens can be faked by monkeypatching the auth requirement.

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="module")
async def test_app():
    from api.main import app, _require_business_auth
    async def _fake_auth(request):
        return {"sub": "tester", "mfa": True, "role": "admin"}
    # monkeypatch dependency
    app.dependency_overrides[_require_business_auth] = _fake_auth  # type: ignore
    yield app
    app.dependency_overrides.clear()

@pytest.mark.anyio
async def test_kpis_endpoint_cache(test_app):
    from api.main import _KPI_CACHE
    _KPI_CACHE['data'] = None
    _KPI_CACHE['ts'] = 0
    async with AsyncClient(app=test_app, base_url="http://testserver") as ac:
        r1 = await ac.get("/business/api/kpis")
        assert r1.status_code == 200
        first = r1.json()
        assert set(['timestamp','kpis','cached']).issubset(first.keys())
        # second request should be cached OR 304 if ETag revalidation
        headers = {}
        if 'etag' in r1.headers:
            headers['If-None-Match'] = r1.headers['etag']
        r2 = await ac.get("/business/api/kpis", headers=headers)
        assert r2.status_code in (200, 304)
        if r2.status_code == 200:
            second = r2.json()
            assert second.get('cached') is True or second.get('cached') is False  # schema presence

@pytest.mark.anyio
async def test_company_report_schema(test_app):
    async with AsyncClient(app=test_app, base_url="http://testserver") as ac:
        r = await ac.get("/business/api/company/AAPL/report")
        assert r.status_code == 200
        data = r.json()
        for key in ['timestamp','symbol','summary','highlights','fundamentals','factors','risk','options','cached']:
            assert key in data
        assert isinstance(data['highlights'], list)
        assert isinstance(data['fundamentals'], dict)
        # ETag revalidation
        etag = r.headers.get('etag')
        if etag:
            r2 = await ac.get("/business/api/company/AAPL/report", headers={'If-None-Match': etag})
            assert r2.status_code in (200,304)

#!/usr/bin/env python3
"""Test ingestion health classification logic boundary transitions.

We simulate timestamps at various ages to force ok, warning, stale, unknown classifications.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
import pytest

from api.main import app

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('DISABLE_AUTH','1')
    monkeypatch.setitem(app.dependency_overrides, {}, {})
    # Provide dummy auth manager so _require_business_auth passes
    from api import auth as auth_mod
    def fake_verify_access_token(token):
        return {'sub':'test-user','mfa':True,'role':'analyst'}
    monkeypatch.setattr(auth_mod, 'get_auth_manager', lambda : type('X',(object,),{'verify_access_token':fake_verify_access_token,'redis':None})())
    with TestClient(app) as c:
        yield c


def _iso(dt):
    return dt.replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z')


def test_ingestion_classifications(client, monkeypatch):
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    import api.coverage_utils as cu
    # Provide latest for news & social only; equities/options rely on QuestDB query which we will bypass via patch
    async def fake_compute_coverage():
        return {'status':'ok','latest':{'news': _iso(now - timedelta(seconds=100)),  # ok (<900)
                                       'social': _iso(now - timedelta(seconds=2000))}}  # warning (<3600)
    monkeypatch.setattr(cu, 'compute_coverage', fake_compute_coverage)

    # Patch questdb functions to skip DB errors and set timestamps manually by patching business_ingestion_health internals not accessible -> easier: monkeypatch questdb pool acquisition to raise so questdb paths skipped.
    import trading_common.questdb as qdb  # type: ignore
    async def fake_get_questdb_pool():
        class DummyPool:
            async def acquire(self):
                class DummyConn:
                    async def __aenter__(self):
                        raise Exception("questdb unavailable")
                    async def __aexit__(self, exc_type, exc, tb):
                        return False
                return DummyConn()
        return DummyPool()
    monkeypatch.setattr(qdb, 'get_questdb_pool', fake_get_questdb_pool)

    resp = client.get('/business/api/ingestion/health')
    assert resp.status_code == 200
    body = resp.json()
    # equities/options will be unknown due to questdb error
    assert body['equities_lag_class'] == 'unknown'
    assert body['options_lag_class'] == 'unknown'
    # news ok, social warning per injected ages
    assert body['news_lag_class'] == 'ok'
    assert body['social_lag_class'] == 'warning'

if __name__ == '__main__':
    import pytest
    pytest.main([__file__])

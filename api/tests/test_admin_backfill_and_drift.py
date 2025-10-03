#!/usr/bin/env python3
"""Tests for admin backfill and drift summary endpoints.

These are lightweight structural tests that mock Redis and database interactions.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

# Import app
from api.main import app

# Utilities to patch

class DummyRedis:
    def __init__(self):
        self.hashes: Dict[str, Dict[str,str]] = {}
        self.sorted: Dict[str, Dict[str,float]] = {}
        self.values: Dict[str,str] = {}

    async def hset(self, key: str, mapping: Dict[str, Any]):  # type: ignore[override]
        self.hashes.setdefault(key, {}).update({k:str(v) for k,v in mapping.items()})

    async def zadd(self, zkey: str, mapping: Dict[str, float]):  # type: ignore[override]
        z = self.sorted.setdefault(zkey, {})
        for k,v in mapping.items():
            z[k] = v

    async def zrevrange(self, zkey: str, start: int, end: int):  # type: ignore[override]
        items = sorted(self.sorted.get(zkey, {}).items(), key=lambda x: x[1], reverse=True)
        return [k for k,_ in items[start:end+1]]

    async def hgetall(self, key: str):  # type: ignore[override]
        return self.hashes.get(key, {})

    async def get(self, key: str):  # type: ignore[override]
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):  # type: ignore[override]
        self.values[key] = value


@pytest.fixture
def client(monkeypatch):
    # Patch rate limiter to return object with redis attribute
    class DummyLimiter:
        def __init__(self, r):
            self.redis = r
            self.connected = True
        async def check_rate_limit(self, *a, **k):
            return {'allowed': True}
        async def close(self):
            pass
    dummy_redis = DummyRedis()

    async def get_rate_limiter():  # noqa: D401
        return DummyLimiter(dummy_redis)

    monkeypatch.setenv("DISABLE_AUTH", "1")  # if auth dependencies check env
    monkeypatch.setitem(app.dependency_overrides, {}, {})  # ensure clean
    monkeypatch.setenv('FORECAST_MODEL_NAME','returns_gbm')
    monkeypatch.setenv('PULSAR_TOPICS','')

    import api.rate_limiter as rl
    monkeypatch.setattr(rl, 'get_rate_limiter', get_rate_limiter)

    # Provide simple admin token bypass by overriding auth dependency functions
    from api import auth as auth_mod
    def fake_verify_access_token(token):
        return {'sub':'test-admin','mfa':True,'role':'admin'}
    monkeypatch.setattr(auth_mod, 'get_auth_manager', lambda : SimpleNamespace(verify_access_token=fake_verify_access_token, redis=dummy_redis))

    with TestClient(app) as c:
        yield c


def test_force_backfill_and_list(client: TestClient):
    # Create job
    resp = client.post('/admin/api/verification/force-backfill', json={'symbols':['AAPL','MSFT'],'dataset':'equities'})
    assert resp.status_code in (200,202) or resp.json()['status'] == 'accepted'
    data = resp.json()
    job_id = data['job_id']
    # List jobs
    list_resp = client.get('/admin/api/backfill/jobs')
    assert list_resp.status_code == 200
    jobs = list_resp.json()['jobs']
    assert any(j.get('job_id') == job_id for j in jobs)
    # Detail
    detail = client.get(f'/admin/api/backfill/jobs/{job_id}')
    assert detail.status_code == 200
    assert detail.json()['job']['job_id'] == job_id


def test_drift_summary_cache(client: TestClient, monkeypatch):
    # Patch DB manager to return sample drift rows
    class DummySession:
        async def fetch_all(self, query, params):  # type: ignore[override]
            now = datetime.utcnow()
            return [
                {'model_name':'returns_gbm','drift_type':'FEATURE','drift_score':0.12,'threshold_value':0.1,'detected_at': now},
                {'model_name':'returns_gbm','drift_type':'FEATURE','drift_score':0.05,'threshold_value':0.1,'detected_at': now - timedelta(hours=1)},
                {'model_name':'vol_model','drift_type':'PREDICTION','drift_score':0.25,'threshold_value':0.15,'detected_at': now}
            ]
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
    class DummyDBM:
        async def get_postgres(self):  # noqa: D401
            return DummySession()
    async def get_database_manager():
        return DummyDBM()
    import trading_common.database_manager as dbm
    monkeypatch.setattr(dbm, 'get_database_manager', get_database_manager)

    first = client.get('/admin/api/drift/summary')
    assert first.status_code == 200
    body = first.json()
    assert body['overall_severity'] in ('medium','high','low')
    # Cached path (should still succeed)
    second = client.get('/admin/api/drift/summary')
    assert second.status_code == 200


def test_ingestion_health(client: TestClient, monkeypatch):
    # Patch coverage_utils.compute_coverage to supply latest times
    import api.coverage_utils as cu
    async def fake_compute_coverage():
        return {'status':'ok','latest':{'news': datetime.utcnow().isoformat(), 'social': datetime.utcnow().isoformat()}}
    monkeypatch.setattr(cu, 'compute_coverage', fake_compute_coverage)
    resp = client.get('/business/api/ingestion/health')
    assert resp.status_code == 200
    data = resp.json()
    assert 'news_lag_class' in data and 'social_lag_class' in data

if __name__ == '__main__':
    pytest.main([__file__])

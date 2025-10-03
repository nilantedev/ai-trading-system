#!/usr/bin/env python3
"""Test forecast endpoint fallback behavior.

Ensures when model service forecast path raises or times out we return fallback baseline with fallback: True.
"""
from __future__ import annotations

import asyncio
from fastapi.testclient import TestClient
import pytest

from api.main import app

@pytest.fixture
def client(monkeypatch):
    # Disable auth for test and patch dependencies similar to other tests
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setitem(app.dependency_overrides, {}, {})  # reset

    # Provide dummy auth manager so _require_business_auth passes
    from api import auth as auth_mod
    def fake_verify_access_token(token):
        return {'sub':'test-user','mfa':True,'role':'analyst'}
    monkeypatch.setattr(auth_mod, 'get_auth_manager', lambda : type('X',(object,),{'verify_access_token':fake_verify_access_token,'redis':None})())

    # Patch model serving service to simulate timeout by never completing predict()
    class DummyPredictResponse:
        def __init__(self):
            self.prediction = 0.5
            self.confidence = 0.9
            self.model_version = 'v1'

    async def get_model_serving_service():  # pragma: no cover - structure only
        class DummySvc:
            async def predict(self, req):  # type: ignore[override]
                # Simulate long-running prediction beyond 2s timeout
                await asyncio.sleep(5)
                return DummyPredictResponse()
        return DummySvc()

    import services.ml.model_serving_service as mss  # type: ignore
    monkeypatch.setattr(mss, 'get_model_serving_service', get_model_serving_service)

    with TestClient(app) as c:
        yield c


def test_forecast_timeout_fallback(client: TestClient):
    resp = client.get('/business/api/company/AAPL/forecast')
    assert resp.status_code == 200
    data = resp.json()
    # Should indicate fallback True for next_1d_return
    assert data['forecasts']['next_1d_return']['fallback'] is True
    assert data['forecasts']['next_1d_return']['model'] == 'returns_gbm_v3'
    assert data['symbol'] == 'AAPL'

if __name__ == '__main__':
    pytest.main([__file__])

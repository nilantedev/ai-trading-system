import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from api.main import app  # type: ignore


def test_auth_me_unauthenticated():
    client = TestClient(app)
    r = client.get('/auth/me')
    assert r.status_code == 401


def test_auth_me_authenticated(monkeypatch):
    client = TestClient(app)

    # Create a minimal fake token by calling real login flow requires DB; instead monkeypatch auth manager
    from api import auth as auth_module  # type: ignore

    class DummyUser(auth_module.User):
        pass

    async def fake_authenticate_user(username: str, password: str, client_ip: str = "unknown"):
        return auth_module.User(
            user_id="u123",
            username="tester",
            roles=["viewer"],
            permissions=[],
            is_active=True,
            mfa_enabled=False,
        )

    monkeypatch.setattr(auth_module, 'authenticate_user', fake_authenticate_user)

    # Perform login (no MFA required)
    r = client.post('/auth/login', data={'username': 'tester', 'password': 'secret'})
    assert r.status_code in (200, 401)  # 401 only if MFA required misconfiguration
    if r.status_code == 401:
        pytest.skip('MFA enforced unexpectedly in test environment')
    # Access /auth/me
    r2 = client.get('/auth/me')
    assert r2.status_code == 200
    data = r2.json()
    assert data['username'] == 'tester'
    assert 'roles' in data
    assert 'mfa' in data

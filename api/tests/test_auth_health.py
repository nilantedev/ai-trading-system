import asyncio
import types
import pytest

# We will import the auth module and monkeypatch internals to simulate conditions
from api import auth as auth_module  # type: ignore

@pytest.mark.asyncio
async def test_auth_health_advisories(monkeypatch):
    # Get singleton manager
    manager = await auth_module.get_auth_manager()

    class FakeRedis:
        def __init__(self, keys):
            self._keys = keys
        async def scan(self, cursor=0, match=None, count=10):
            # Return all keys in one batch then end
            if cursor == 0:
                return (1, [k for k in self._keys if match is None or k.startswith(match.split(':*')[0])])
            else:
                return (0, [])
        async def ping(self):
            return True

    # Monkeypatch redis to simulate many failed attempts and low MFA adoption
    fake_keys = ["failed_attempts:user" + str(i) for i in range(30)] + ["mfa:enabled:user1"]
    manager.redis = FakeRedis(fake_keys)  # type: ignore

    # Monkeypatch database manager to simulate total users = 10
    class FakeSession:
        async def execute(self, sql):
            class R:
                def mappings(self_inner):
                    return types.SimpleNamespace(first=lambda: {"c": 10})
            return R()
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeDBM:
        def get_postgres(self):
            return FakeSession()

    async def fake_get_dbm():
        return FakeDBM()

    monkeypatch.setattr(auth_module, 'get_database_manager', fake_get_dbm)

    # Clear cache to force recompute
    auth_module._AUTH_HEALTH_CACHE['data'] = None
    auth_module._AUTH_HEALTH_CACHE['ts'] = 0

    snapshot = await auth_module.get_auth_health_async()
    assert 'mfa_adoption_percent' in snapshot
    assert snapshot['mfa_adoption_percent'] == 10.0  # 1 of 10 users
    # Expect advisories for low MFA and elevated failed login counters
    assert 'mfa_adoption_low' in snapshot['advisories']
    assert 'elevated_failed_login_counters' in snapshot['advisories']

@pytest.mark.asyncio
async def test_auth_health_cache(monkeypatch):
    # Ensure caching returns quickly without re-scanning
    auth_module._AUTH_HEALTH_CACHE['data'] = {'foo': 'bar'}
    auth_module._AUTH_HEALTH_CACHE['ts'] = auth_module.time.time()
    snap = await auth_module.get_auth_health_async()
    assert snap['foo'] == 'bar'
    assert snap.get('cache') == 'hit'

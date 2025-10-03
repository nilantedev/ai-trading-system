import json
import os
import tempfile
import pytest
from datetime import datetime, timezone

from scripts import enforce_retention as retention  # type: ignore

@pytest.mark.asyncio
async def test_retention_metrics_json(monkeypatch):
    # Monkeypatch database manager to avoid real DB dependency
    class FakeSession:
        async def execute(self, q, params=None):  # noqa: D401
            class R:
                def scalar(self_inner):
                    # Pretend there are 100 total rows and 5 old ones for every table
                    return 100 if 'COUNT' in q else None
            return R()
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False
    class FakeDBM:
        def get_postgres(self): return FakeSession()
    async def fake_get_dbm(): return FakeDBM()
    monkeypatch.setattr(retention, 'get_database_manager', fake_get_dbm)

    # Temporary metrics file
    fd, path = tempfile.mkstemp(prefix='retention_metrics_', suffix='.json')
    os.close(fd)

    code = await retention.enforce_retention(apply=False, metrics_path=path, no_lock=True)
    assert code == 0
    data = json.loads(open(path).read())
    assert 'tables' in data and isinstance(data['tables'], list)
    assert any(t['table'] == 'equities_prices' for t in data['tables'])
    # Cleanup
    os.remove(path)

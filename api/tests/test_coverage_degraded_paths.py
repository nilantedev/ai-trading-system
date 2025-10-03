import pytest

@pytest.mark.asyncio
async def test_coverage_handles_absence(monkeypatch):
    # Force questdb and weaviate helpers to raise
    async def fail(*a, **k):
        raise RuntimeError("unavailable")
    monkeypatch.setattr('api.coverage_utils._questdb_range', fail, raising=True)
    monkeypatch.setattr('api.coverage_utils._postgres_range', fail, raising=True)
    monkeypatch.setattr('api.coverage_utils._weaviate_latest', fail, raising=True)
    from api.coverage_utils import compute_coverage
    data = await compute_coverage()
    # Should still return structure
    assert 'datasets' in data
    assert any(d.get('status') == 'unavailable' for d in data['datasets'])

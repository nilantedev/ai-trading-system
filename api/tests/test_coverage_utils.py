import asyncio
import pytest

@pytest.mark.asyncio
async def test_compute_coverage_structure():
    from api.coverage_utils import compute_coverage
    data = await compute_coverage()
    assert 'datasets' in data
    assert 'ratios' in data
    assert isinstance(data['datasets'], list)
    # Ensure each dataset entry has dataset key
    for d in data['datasets']:
        assert 'dataset' in d

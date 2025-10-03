from datetime import datetime, timedelta, timezone
import pytest

from scripts import enforce_retention as retention  # type: ignore

@pytest.mark.asyncio
async def test_compute_cutoffs_windows():
    now = datetime(2025, 9, 12, tzinfo=timezone.utc)
    cutoffs = await retention.compute_cutoffs(now)
    assert 'equities_prices' in cutoffs
    # 20 year window approx 20*365 days within tolerance (leap years) -> between 7300 and 7310
    delta_days = (now - cutoffs['equities_prices']).days
    assert 7300 <= delta_days <= 7315
    # 5 year windows
    for k in ['options_quotes','news_items','social_posts']:
        d = (now - cutoffs[k]).days
        assert 1825 <= d <= 1830

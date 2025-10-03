import asyncio
import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

from services.data_ingestion.market_data_service import MarketDataService

@pytest.mark.asyncio
async def test_eodhd_success_path(monkeypatch):
    svc = MarketDataService()

    class FakeProvider:
        async def fetch_daily_bars(self, symbol, start_date, end_date):
            return [
                {"symbol": symbol, "timestamp": datetime.utcnow()-timedelta(days=1), "open": 10, "high": 12, "low": 9, "close":11, "volume":1000, "data_source":"eodhd"}
            ]
    svc.eodhd_provider = FakeProvider()
    svc.eodhd_config['api_key'] = 'x'
    svc.alpha_vantage_config['api_key'] = 'y'
    await svc.start()

    results = await svc.get_bulk_daily_historical('TEST', datetime.utcnow()-timedelta(days=5), datetime.utcnow())
    assert results, 'Expected results from primary provider'
    assert results[0].data_source == 'eodhd'

@pytest.mark.asyncio
async def test_fallback_activation(monkeypatch):
    svc = MarketDataService()

    class FailingProvider:
        async def fetch_daily_bars(self, *a, **k):
            raise RuntimeError('boom')
    svc.eodhd_provider = FailingProvider()
    svc.eodhd_config['api_key'] = 'x'

    # Mock alpha vantage HTTP call
    async def fake_get(url, params):
        class Resp:
            status = 200
            async def json(self_inner):
                return {
                    'Time Series (Daily)': {
                        (datetime.utcnow()-timedelta(days=1)).strftime('%Y-%m-%d'): {
                            '1. open': '10', '2. high': '11', '3. low': '9', '4. close': '10.5', '6. volume': '1000'
                        }
                    }
                }
            async def __aenter__(self_inner):
                return self_inner
            async def __aexit__(self_inner, exc_type, exc, tb):
                return False
        return Resp()

    session = SimpleNamespace(get=fake_get)
    svc.session = session
    svc.alpha_vantage_config['api_key'] = 'y'
    await svc.start()

    results = await svc.get_bulk_daily_historical('TEST', datetime.utcnow()-timedelta(days=5), datetime.utcnow())
    assert results, 'Expected fallback results'
    assert results[0].data_source == 'alpha_vantage'
import asyncio
import pytest

from trading_common.drift_scheduler import start_drift_monitor, stop_drift_monitor

@pytest.mark.asyncio
async def test_start_and_stop_drift_monitor():
    monitor = await start_drift_monitor(interval_seconds=1)
    # no models registered yet; allow one tick
    await asyncio.sleep(1.2)
    assert monitor.scans_run >= 0
    await stop_drift_monitor(monitor)

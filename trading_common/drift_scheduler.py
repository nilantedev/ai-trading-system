import asyncio
from typing import Any, List, Tuple

async def start_drift_monitor(interval_seconds: int, models: List[Tuple[str, str]]):
    async def _run():
        while True:
            await asyncio.sleep(interval_seconds)
    task = asyncio.create_task(_run())
    return task

async def stop_drift_monitor(task):
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

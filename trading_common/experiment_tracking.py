import asyncio

async def get_experiment_tracker():
    await asyncio.sleep(0)
    class Tracker: ...
    return Tracker()

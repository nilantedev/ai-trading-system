import asyncio
from typing import Any, List

class _DB:
    async def fetch_all(self, query: str) -> List[dict]:
        return []
    async def fetch_one(self, query: str, params=None):
        return {"c": 0}

class ModelRegistry:
    def __init__(self):
        self.db = _DB()

_registry: ModelRegistry | None = None

async def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        await asyncio.sleep(0)
        _registry = ModelRegistry()
    return _registry

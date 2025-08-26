import asyncio
from datetime import datetime
from typing import Any, Dict, List

class _DB:
    async def fetch_one(self, query: str, params=None):
        return None

class FeatureStore:
    def __init__(self):
        self.db = _DB()
        self.feature_definitions: Dict[str, Any] = {}
    async def register_feature_view(self, name: str, features: List[str], description: str):
        return None
    async def materialize_feature_view(self, name: str, entity_ids: List[str], as_of: datetime):
        return None

_store: FeatureStore | None = None

async def get_feature_store() -> FeatureStore:
    global _store
    if _store is None:
        await asyncio.sleep(0)
        _store = FeatureStore()
    return _store

"""Weaviate schema management scaffold.

Phase 1 Goals:
 - Provide a pure diff engine for desired vs current schema (no side effects at import time).
 - Safe client factory (lazy) with minimal requirements.
 - Allow future extension: hybrid search, replication, sharding, metrics.
 - Keep tests offline (no live Weaviate dependency required).

The structures used here are intentionally simplified representations of Weaviate classes:
Desired schema format (list of class specs):
[
  {
    "class": "NewsArticle",
    "description": "Normalized news article content",
    "vectorizer": "text2vec-transformers",
    "moduleConfig": {...},          # optional
    "properties": [
       {"name": "title", "dataType": ["text"], "description": "Headline"},
       {"name": "published_at", "dataType": ["date"]}
    ]
  },
]

Diff result example:
{
  'add_classes': [ ... full class specs ... ],
  'remove_classes': ['DeprecatedClass'],
  'add_properties': {
       'NewsArticle': [{...property spec...}]
  },
  'modify_classes': []  # reserved for future (vectorizer/config changes)
}
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
import os
from time import perf_counter

try:  # pragma: no cover - runtime import guarded for offline tests
    import weaviate  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    weaviate = None  # type: ignore


class WeaviateSchemaError(RuntimeError):
    """Domain error for schema management."""


@dataclass(frozen=True)
class WeaviateConfig:
    url: str = os.getenv("DB_WEAVIATE_URL", "http://localhost:8080")
    api_key: Optional[str] = os.getenv("WEAVIATE_API_KEY")
    timeout: int = int(os.getenv("WEAVIATE_TIMEOUT_SECONDS", "15"))

    def validate(self) -> None:
        if not self.url.startswith("http"):
            raise WeaviateSchemaError(f"Invalid Weaviate URL: {self.url}")


@lru_cache(maxsize=1)
def get_weaviate_client(cfg: Optional[WeaviateConfig] = None):  # pragma: no cover - network path not tested
    """Return a cached Weaviate client instance.
    In scaffold stage we avoid raising if library missing until actually used.
    """
    if weaviate is None:
        raise WeaviateSchemaError("weaviate-client not installed or failed to import")
    cfg = cfg or WeaviateConfig()
    cfg.validate()
    auth_config = None
    if cfg.api_key:
        try:
            from weaviate.auth import AuthApiKey  # type: ignore[import-not-found]
            auth_config = AuthApiKey(api_key=cfg.api_key)
        except ImportError:  # pragma: no cover
            auth_config = None
    return weaviate.Client(url=cfg.url, auth_client_secret=auth_config, timeout_config=(cfg.timeout, cfg.timeout))  # type: ignore


def desired_schema() -> List[Dict[str, Any]]:
    """Return the desired canonical class specs (initial minimal set).
    Extend this definition in future phases.
    """
    return [
        {
            "class": "NewsArticle",
            "description": "Normalized financial news article",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "title", "dataType": ["text"], "description": "Article headline"},
                {"name": "body", "dataType": ["text"], "description": "Full text"},
                {"name": "source", "dataType": ["text"], "description": "Source provider"},
                {"name": "published_at", "dataType": ["date"], "description": "Publication timestamp"},
                {"name": "tickers", "dataType": ["text[]"], "description": "Related symbols"},
            ],
        }
    ]


def index_classes_by_name(classes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {c["class"]: c for c in classes}


def property_names(class_spec: Dict[str, Any]) -> List[str]:
    return [p["name"] for p in class_spec.get("properties", [])]


def diff_schema(current: List[Dict[str, Any]], desired: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute difference between current and desired schema (non-destructive).

    We treat property removals & class modifications as advisory only in this scaffold (no automatic destructive ops).
    """
    cur_index = index_classes_by_name(current)
    des_index = index_classes_by_name(desired)

    add_classes: List[Dict[str, Any]] = []
    remove_classes: List[str] = []
    add_properties: Dict[str, List[Dict[str, Any]]] = {}
    modify_classes: List[Tuple[str, str]] = []  # (class_name, reason)

    # Classes to add
    for name, spec in des_index.items():
        if name not in cur_index:
            add_classes.append(spec)
        else:
            # Compare vectorizer / description for modification advisory
            cur_spec = cur_index[name]
            if cur_spec.get("vectorizer") != spec.get("vectorizer"):
                modify_classes.append((name, "vectorizer differs"))
            if cur_spec.get("description") != spec.get("description"):
                modify_classes.append((name, "description differs"))
            # Property additions
            cur_props = set(property_names(cur_spec))
            for prop in spec.get("properties", []):
                if prop["name"] not in cur_props:
                    add_properties.setdefault(name, []).append(prop)

    # Classes present currently but absent in desired -> removal advisory only
    for name in cur_index.keys():
        if name not in des_index:
            remove_classes.append(name)

    return {
        "add_classes": add_classes,
        "remove_classes": remove_classes,
        "add_properties": add_properties,
        "modify_classes": modify_classes,
    }


def fetch_current_schema(client) -> List[Dict[str, Any]]:  # pragma: no cover - network
    schema = client.schema.get()
    return schema.get("classes", []) if isinstance(schema, dict) else []


def apply_schema_changes(diff: Dict[str, Any], client) -> Dict[str, Any]:  # pragma: no cover - network
    """Apply additive schema changes ONLY (no destructive operations)."""
    applied = {"classes_created": 0, "properties_added": 0}
    start = perf_counter()
    for cls in diff.get("add_classes", []):
        client.schema.create_class(cls)
        applied["classes_created"] += 1
    for cls_name, props in diff.get("add_properties", {}).items():
        for p in props:
            client.schema.add_property(cls_name, p)
            applied["properties_added"] += 1
    _record_vector_metric('schema_apply', 'success', start)
    return applied


def _record_vector_metric(operation: str, status: str, start_time: float) -> None:
    try:  # pragma: no cover
        from api.metrics import vector_store_operations_total, vector_store_operation_duration_seconds
        vector_store_operations_total.labels(operation=operation, status=status).inc()
        vector_store_operation_duration_seconds.labels(operation=operation).observe(perf_counter() - start_time)
    except Exception:
        pass


__all__ = [
    "WeaviateConfig",
    "WeaviateSchemaError",
    "get_weaviate_client",
    "desired_schema",
    "diff_schema",
    "fetch_current_schema",
    "apply_schema_changes",
]

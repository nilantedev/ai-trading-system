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

try:
    import weaviate
    from weaviate.auth import AuthApiKey
    from weaviate.config import Config
except ImportError:
    weaviate = None


class WeaviateSchemaError(RuntimeError):
    """Domain error for schema management."""


@dataclass(frozen=True)
class WeaviateConfig:
    # Prefer service-level WEAVIATE_URL when present; fall back to legacy DB_WEAVIATE_URL, then sane default
    url: str = (
        os.getenv("WEAVIATE_URL")
        or os.getenv("DB_WEAVIATE_URL")
        or "http://weaviate:8080"
    )
    api_key: Optional[str] = os.getenv("WEAVIATE_API_KEY")
    timeout: int = int(os.getenv("WEAVIATE_TIMEOUT_SECONDS", "15"))

    def validate(self) -> None:
        if not self.url.startswith("http"):
            raise WeaviateSchemaError(f"Invalid Weaviate URL: {self.url}")


@lru_cache(maxsize=1)
def get_weaviate_client(cfg: Optional[WeaviateConfig] = None):
    """Return a cached Weaviate v4 client instance with safe auth fallback.

    Behavior:
    - If WEAVIATE_API_KEY is provided, first attempt authenticated connection.
    - On auth-related failures (401 / OIDC not configured), retry without auth
      to support clusters with anonymous access enabled.
    """
    if weaviate is None:
        raise WeaviateSchemaError("weaviate-client not installed or failed to import")

    cfg = cfg or WeaviateConfig()
    cfg.validate()

    # Extract host and port from URL
    from urllib.parse import urlparse
    parsed = urlparse(cfg.url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080

    # Build an authenticated client first (if api_key present)
    def _connect(auth):
        return weaviate.connect_to_local(
            host=host,
            port=port,
            auth_credentials=auth,
            skip_init_checks=True,
        )

    last_err: Optional[Exception] = None
    # Try with API key if provided
    if cfg.api_key:
        try:
            client = _connect(AuthApiKey(api_key=cfg.api_key))
            # Touch meta to validate auth where possible
            try:
                _ = client.cluster.get_nodes()  # cheap call; raises on auth issues in many setups
            except Exception:
                pass
            return client
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            # Known case: server has no OIDC/API key configured -> 401 with OIDC hint
            if "oidc auth is not configured" in msg.lower() or "401" in msg or "unauthorized" in msg.lower():
                # fall through to anonymous retry
                pass
            else:
                # Not an auth issue; propagate
                raise

    # Retry without auth (anonymous access)
    try:
        client = _connect(None)
        return client
    except Exception as e:  # noqa: BLE001
        # If we had an earlier auth error, surface that for operator clarity
        if last_err is not None:
            raise WeaviateSchemaError(f"Authenticated connect failed and anonymous retry failed: auth_error={last_err}; anon_error={e}")
        raise WeaviateSchemaError(str(e))


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
        },
        {
            "class": "SocialSentiment",
            "description": "Aggregated social sentiment observation per symbol & platform",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "symbol", "dataType": ["text"], "description": "Underlying symbol"},
                {"name": "platform", "dataType": ["text"], "description": "Source platform (reddit/twitter/stocktwits)"},
                {"name": "sentiment", "dataType": ["text"], "description": "Serialized numeric sentiment score"},
                {"name": "momentum", "dataType": ["text"], "description": "Serialized momentum score"},
                {"name": "topics", "dataType": ["text[]"], "description": "Key discussion topics"},
                {"name": "timestamp", "dataType": ["date"], "description": "Observation timestamp"},
                {"name": "influential_mentions", "dataType": ["text"], "description": "Influential mention count (stringified)"},
                {"name": "confidence", "dataType": ["text"], "description": "Confidence score (stringified)"},
            ],
        },
        {
            "class": "OptionContract",
            "description": "Daily aggregate / metadata for an option contract",
            "vectorizer": "none",
            "properties": [
                {"name": "underlying", "dataType": ["text"], "description": "Underlying symbol"},
                {"name": "option_symbol", "dataType": ["text"], "description": "Full option ticker (e.g. O:XYZ... )"},
                {"name": "expiry", "dataType": ["date"], "description": "Expiration date"},
                {"name": "right", "dataType": ["text"], "description": "Call or Put"},
                {"name": "strike", "dataType": ["text"], "description": "Strike price (stringified)"},
                {"name": "implied_vol", "dataType": ["text"], "description": "Implied volatility (stringified)"},
                {"name": "timestamp", "dataType": ["date"], "description": "Observation timestamp (bar date)"},
            ],
        },
        {
            "class": "EquityBar",
            "description": "Daily equity OHLCV bar snapshot for vector augmentation",
            "vectorizer": "none",
            "properties": [
                {"name": "symbol", "dataType": ["text"], "description": "Equity symbol"},
                {"name": "timestamp", "dataType": ["date"], "description": "Bar date"},
                {"name": "open", "dataType": ["text"], "description": "Open price (stringified)"},
                {"name": "high", "dataType": ["text"], "description": "High price (stringified)"},
                {"name": "low", "dataType": ["text"], "description": "Low price (stringified)"},
                {"name": "close", "dataType": ["text"], "description": "Close price (stringified)"},
                {"name": "volume", "dataType": ["text"], "description": "Volume (stringified)"},
            ],
        },
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


def fetch_current_schema(client) -> List[Dict[str, Any]]:
    """Fetch current schema using v4 client."""
    try:
        collections = client.collections.list_all()
        classes = []
        for name in collections:
            collection = client.collections.get(name)
            config = collection.config.get()
            class_spec = {
                "class": name,
                "description": config.description if hasattr(config, 'description') else "",
                "properties": []
            }
            if hasattr(config, 'properties'):
                for prop in config.properties:
                    class_spec["properties"].append({
                        "name": prop.name,
                        "dataType": [prop.data_type],
                        "description": prop.description if hasattr(prop, 'description') else ""
                    })
            classes.append(class_spec)
        return classes
    except Exception:
        return []


def apply_schema_changes(diff: Dict[str, Any], client) -> Dict[str, Any]:
    """Apply schema changes using v4 client."""
    applied = {"classes_created": 0, "properties_added": 0}
    start = perf_counter()
    
    for cls in diff.get("add_classes", []):
        try:
            # Map old dataType format to v4 DataType
            properties = []
            for p in cls.get("properties", []):
                data_type = p["dataType"][0]
                # Map broader set of simple types; fallback to TEXT
                if data_type == "text":
                    wv_data_type = weaviate.classes.config.DataType.TEXT
                elif data_type == "text[]":
                    wv_data_type = weaviate.classes.config.DataType.TEXT_ARRAY
                elif data_type == "date":
                    wv_data_type = weaviate.classes.config.DataType.DATE
                elif data_type in ("number", "float", "int"):
                    # We store numbers as TEXT in this scaffold (stringified); true numeric support can be added later.
                    wv_data_type = weaviate.classes.config.DataType.TEXT
                else:
                    wv_data_type = weaviate.classes.config.DataType.TEXT
                
                properties.append(
                    weaviate.classes.config.Property(
                        name=p["name"],
                        data_type=wv_data_type,
                        description=p.get("description", "")
                    )
                )
            
            client.collections.create(
                name=cls["class"],
                description=cls.get("description", ""),
                properties=properties
            )
            applied["classes_created"] += 1
        except Exception as e:
            print(f"Error creating class {cls['class']}: {e}")
    
    _record_vector_metric('schema_apply', 'success', start)
    return applied


def ensure_desired_schema(client) -> Dict[str, Any]:
    """Idempotently ensure desired schema is applied.

    Returns a dict summarizing actions taken (same shape as apply_schema_changes output + diff keys).
    Swallows errors (returns empty) if client not available. Intended for best-effort startup / fallback flows.
    """
    try:  # pragma: no cover
        current = fetch_current_schema(client)
        desired = desired_schema()
        diff = diff_schema(current, desired)
        # Fast path: nothing to add
        if not diff['add_classes'] and not diff['add_properties']:
            return {"changed": False, **diff}
        applied = apply_schema_changes(diff, client)
        return {"changed": True, **diff, **applied}
    except Exception:
        return {"changed": False, "error": True}


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
    "ensure_desired_schema",
]

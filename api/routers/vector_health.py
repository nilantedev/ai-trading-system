"""Vector store health & observability endpoints.

Provides an internal endpoint exposing:
 - Schema synchronization diff (add classes/properties summary)
 - Connectivity & latency to Weaviate (best-effort)
 - Primary vs fallback indexing counters (if metrics registry accessible)
 - Basic embedding quality proxy (sample recall of recent objects vs simple keyword query) best-effort.

All failures are tolerated; endpoint never raises.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from time import perf_counter
from typing import Dict, Any

try:  # Prometheus optional
    from prometheus_client import Gauge  # type: ignore
except Exception:  # noqa: BLE001
    Gauge = None  # type: ignore

_drift_gauge = None
if Gauge:
    try:
        _drift_gauge = Gauge('vector_embedding_keyword_recall_score','Surrogate embedding recall score (keyword coverage)')
    except Exception:  # noqa: BLE001
        _drift_gauge = None

try:
    from shared.vector.weaviate_schema import get_weaviate_client, ensure_desired_schema, diff_schema, fetch_current_schema, desired_schema
except Exception:  # noqa: BLE001
    get_weaviate_client = None  # type: ignore
    ensure_desired_schema = None  # type: ignore
    diff_schema = None  # type: ignore
    fetch_current_schema = None  # type: ignore
    desired_schema = None  # type: ignore

try:
    from api.auth import get_current_user_cookie_or_bearer  # type: ignore
except Exception:  # noqa: BLE001
    get_current_user_cookie_or_bearer = None  # type: ignore

async def _admin_guard(user=Depends(get_current_user_cookie_or_bearer)):
    # Single admin policy consistent with dashboards
    if not user or getattr(user, 'username', None) != 'nilante':  # pragma: no cover
        raise HTTPException(status_code=403, detail='Access restricted')
    return user

router = APIRouter(prefix="/internal/vector", tags=["vector-internal"])  # Guard applied per-endpoint


@router.get("/health")
async def vector_health(user=Depends(_admin_guard)) -> Dict[str, Any]:  # pragma: no cover - operational endpoint
    start = perf_counter()
    out: Dict[str, Any] = {
        "status": "degraded",
        "latency_ms": None,
        "schema": {},
        "metrics": {},
        "errors": [],
    }
    client = None
    try:
        if get_weaviate_client:
            client = get_weaviate_client()
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"client_init:{e}")
    if client:
        t0 = perf_counter()
        try:
            colls = client.collections.list_all()
            out["latency_ms"] = int((perf_counter() - t0) * 1000)
            out["status"] = "ok"
            # Schema diff summary
            try:
                current = fetch_current_schema(client) if fetch_current_schema else []
                desired = desired_schema() if desired_schema else []
                d = diff_schema(current, desired) if diff_schema else {}
                out["schema"] = {
                    "add_classes": [c.get("class") for c in d.get("add_classes", [])],
                    "add_properties": {k: [p['name'] for p in v] for k, v in d.get("add_properties", {}).items()},
                    "remove_classes": d.get("remove_classes", []),
                }
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"schema_diff:{e}")
            # Lightweight drift surrogate: count news articles containing keyword(s)
            try:  # pragma: no cover
                if 'NewsArticle' in colls:
                    coll = client.collections.get('NewsArticle')
                    # Simple keyword test; if zero matches repeatedly, embeddings/source ingestion might be degraded
                    kw = 'earnings'
                    # Weaviate v4 filtering may differ; fallback to scanning small iterator sample
                    sample_iter = coll.data.iterator(limit=50)  # type: ignore
                    total = 0; match = 0
                    for obj in sample_iter:  # type: ignore
                        total += 1
                        title = ''
                        body = ''
                        try:
                            title = str(obj.properties.get('title',''))  # type: ignore[attr-defined]
                            body = str(obj.properties.get('body',''))
                        except Exception:
                            pass
                        text = (title + ' ' + body).lower()
                        if kw in text:
                            match += 1
                        if total >= 50:
                            break
                    recall = (match / total) if total else 0.0
                    out.setdefault('metrics', {})['keyword_recall_surrogate'] = round(recall,4)
                    if _drift_gauge:
                        try:
                            _drift_gauge.set(recall)
                        except Exception:
                            pass
            except Exception as e:  # noqa: BLE001
                out['errors'].append(f'drift_sample:{e}')
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"list_all:{e}")
    # Metrics snapshot (read from Prometheus registry if process metrics accessible)
    try:
        from prometheus_client import REGISTRY
        metric_map = {
            'news_weaviate_indexed_total': 'news_primary_indexed',
            'news_weaviate_fallback_indexed_total': 'news_fallback_indexed',
            'vector_fallback_news_indexed_total': 'news_direct_fallback_indexed',
            'social_weaviate_indexed_total': 'social_primary_indexed',
            'social_weaviate_fallback_indexed_total': 'social_fallback_indexed',
            'vector_fallback_social_indexed_total': 'social_direct_fallback_indexed',
        }
        collected = {}
        for fam in REGISTRY.collect():  # type: ignore[attr-defined]
            mname = fam.name
            if mname in metric_map:
                # Sum all samples of that metric
                total_val = 0.0
                try:
                    for s in fam.samples:  # type: ignore[attr-defined]
                        total_val += float(s.value)
                except Exception:  # noqa: BLE001
                    pass
                collected[metric_map[mname]] = total_val
        if collected:
            out["metrics"].update(collected)
            # Compute ratios if denominators exist
            try:
                np = collected.get('news_primary_indexed', 0.0)
                nf = collected.get('news_fallback_indexed', 0.0) + collected.get('news_direct_fallback_indexed', 0.0)
                if (np + nf) > 0:
                    out['metrics']['news_fallback_ratio'] = round(nf / (np + nf), 4)
                sp = collected.get('social_primary_indexed', 0.0)
                sf = collected.get('social_fallback_indexed', 0.0) + collected.get('social_direct_fallback_indexed', 0.0)
                if (sp + sf) > 0:
                    out['metrics']['social_fallback_ratio'] = round(sf / (sp + sf), 4)
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"metrics:{e}")
    out["duration_ms"] = int((perf_counter() - start) * 1000)
    return out

__all__ = ["router"]

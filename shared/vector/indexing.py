"""Vector store indexing helpers (direct Weaviate fallback path).

Primary ingestion pathway uses ML service (embedding + batch index). This module provides:
 - Direct fallback indexing when ML service unavailable.
 - Idempotent schema ensure (best-effort).
 - Simple content hashing + optional Redis cache guard to avoid duplicate objects.
 - Prometheus metrics (best-effort) for visibility.

Design notes:
 - Keep dependency surface minimal; only import weaviate client lazily.
 - Fallback only indexes minimal textual fields; embeddings will be auto-generated server-side if configured.
 - Use class names aligned with desired_schema definitions.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Iterable, List, Dict, Any, Optional

from .weaviate_schema import get_weaviate_client, ensure_desired_schema, WeaviateSchemaError

try:  # Prometheus optional
    from prometheus_client import Counter
except Exception:  # noqa: BLE001
    Counter = None  # type: ignore

_metrics: Dict[str, Any] = {}


def _metric(name: str, documentation: str):  # lazy idempotent create
    if not Counter:
        return None
    if name not in _metrics:
        try:  # noqa: SIM105
            _metrics[name] = Counter(name, documentation)
        except Exception:  # noqa: BLE001
            _metrics[name] = None
    return _metrics.get(name)


def _hash_fields(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode('utf-8', errors='ignore'))
        h.update(b'\x1f')
    return h.hexdigest()


async def index_news_fallback(items: Iterable[Dict[str, Any]], *, redis=None) -> int:
    """Best-effort direct indexing of news articles.

    items expected keys: title, content, source, published_at (iso), symbols(list)
    Returns number indexed (skips duplicates).
    """
    try:
        client = get_weaviate_client()
    except Exception:
        return 0
    ensure_desired_schema(client)
    collection = None
    try:
        collection = client.collections.get('NewsArticle')
    except Exception:
        return 0
    seen_hashes = set()
    inserted = 0
    ttl = int(os.getenv('WEAVIATE_DEDUP_TTL_SECONDS', '3600'))
    for it in items:
        title = str(it.get('title',''))[:500]
        published = str(it.get('published_at',''))
        source = str(it.get('source',''))
        body = str(it.get('content',''))[:2000]
        hash_id = _hash_fields(title, published, source)
        if hash_id in seen_hashes:
            continue
        if redis:  # cross-run dedup
            try:
                key = f"vecdedup:news:{hash_id}"
                if await redis.get(key):  # type: ignore[attr-defined]
                    continue
            except Exception:
                pass
        obj = {
            'title': title,
            'body': body,
            'source': source,
            'published_at': published,
            'tickers': it.get('symbols') or []
        }
        try:
            collection.data.insert(obj)
            inserted += 1
            seen_hashes.add(hash_id)
            if redis:
                try:
                    await redis.set(key, '1', ex=ttl)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            continue
    m = _metric('vector_fallback_news_indexed_total', 'Total news articles indexed via direct fallback path')
    if m:
        try:
            m.inc(inserted)
        except Exception:
            pass
    return inserted


async def index_social_fallback(items: Iterable[Dict[str, Any]], *, redis=None) -> int:
    """Direct indexing for social sentiment rows.

    items expected keys: symbol, platform, sentiment, momentum, topics, timestamp
    """
    try:
        client = get_weaviate_client()
    except Exception:
        return 0
    ensure_desired_schema(client)
    try:
        collection = client.collections.get('SocialSentiment')
    except Exception:
        return 0
    inserted = 0
    seen_hashes = set()
    ttl = int(os.getenv('WEAVIATE_DEDUP_TTL_SECONDS', '1800'))
    for it in items:
        symbol = str(it.get('symbol','')).upper()
        platform = str(it.get('platform',''))
        timestamp = str(it.get('timestamp',''))
        hash_id = _hash_fields(symbol, platform, timestamp)
        if hash_id in seen_hashes:
            continue
        if redis:
            try:
                key = f"vecdedup:social:{hash_id}"
                if await redis.get(key):  # type: ignore[attr-defined]
                    continue
            except Exception:
                pass
        obj = {
            'symbol': symbol,
            'platform': platform,
            'sentiment': str(it.get('sentiment','')),
            'momentum': str(it.get('momentum','')),
            'topics': it.get('topics') or [],
            'timestamp': timestamp,
            'influential_mentions': str(it.get('influential_mentions','')),
            'confidence': str(it.get('confidence','')),
        }
        try:
            collection.data.insert(obj)
            inserted += 1
            seen_hashes.add(hash_id)
            if redis:
                try:
                    await redis.set(key, '1', ex=ttl)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            continue
    m = _metric('vector_fallback_social_indexed_total', 'Total social sentiment objects indexed via direct fallback')
    if m:
        try:
            m.inc(inserted)
        except Exception:
            pass
    return inserted


__all__ = [
    'index_news_fallback',
    'index_social_fallback',
]

async def index_options_fallback(items: Iterable[Dict[str, Any]], *, redis=None) -> int:
    """Direct indexing for option contracts (metadata-level, not embeddings).

    items expected keys: underlying, option_symbol, expiry (iso), right/option_type,
    strike (string or number), implied_vol (optional string), timestamp (iso)
    """
    try:
        client = get_weaviate_client()
    except Exception:
        return 0
    ensure_desired_schema(client)
    try:
        collection = client.collections.get('OptionContract')
    except Exception:
        return 0
    inserted = 0
    seen_hashes = set()
    ttl = int(os.getenv('WEAVIATE_DEDUP_TTL_SECONDS', '3600'))
    for it in items:
        underlying = str(it.get('underlying','')).upper()
        option_symbol = str(it.get('option_symbol',''))
        expiry = str(it.get('expiry',''))
        right = str(it.get('right') or it.get('option_type') or '')
        # strike is stored as text in schema; stringify defensively
        strike_val = it.get('strike')
        strike = str(strike_val) if strike_val is not None else ''
        iv = str(it.get('implied_vol',''))
        ts = str(it.get('timestamp',''))
        # Dedup per (option_symbol, expiry)
        hash_id = _hash_fields(option_symbol or underlying, expiry, right, strike)
        if hash_id in seen_hashes:
            continue
        if redis:
            try:
                key = f"vecdedup:options:{hash_id}"
                if await redis.get(key):  # type: ignore[attr-defined]
                    continue
            except Exception:
                pass
        obj = {
            'underlying': underlying,
            'option_symbol': option_symbol,
            'expiry': expiry,
            'right': right,
            'strike': strike,
            'implied_vol': iv,
            'timestamp': ts,
        }
        try:
            collection.data.insert(obj)
            inserted += 1
            seen_hashes.add(hash_id)
            if redis:
                try:
                    await redis.set(key, '1', ex=ttl)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            continue
    m = _metric('vector_fallback_options_indexed_total', 'Total option contracts indexed via direct fallback')
    if m:
        try:
            m.inc(inserted)
        except Exception:
            pass
    return inserted

__all__.append('index_options_fallback')

#!/usr/bin/env python3
"""Vector reconciliation utility.

Goal: Ensure recent news & social sentiment objects exist in Weaviate.

Strategy:
 1. Fetch recent rows from Postgres (if available) OR QuestDB ILP not feasible directly here.
 2. Hash each candidate (same hashing logic as fallback indexer) and build a set of hashes existing in Weaviate (best-effort scan limited).
 3. Index any missing via direct fallback path.

Environment Requirements:
 - DB connection variables for trading_common.database_manager
 - Weaviate accessible.

Safe to run periodically (cron / k8s CronJob). Idempotent.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
from time import perf_counter


async def _load_recent_news(pg, hours: int = 6) -> List[Dict[str, Any]]:
    rows = []
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        q = """
            SELECT symbol, published_at, title, source, url, sentiment, relevance
            FROM news_events
            WHERE published_at >= $1
            ORDER BY published_at DESC
            LIMIT 2000
        """
        recs = await pg.fetch(q, since)  # type: ignore[attr-defined]
        for r in recs:
            rows.append({
                'title': r['title'] or '',
                'content': r['title'] or '',  # fallback minimal text
                'source': r['source'] or '',
                'published_at': r['published_at'].isoformat(),
                'symbols': [r['symbol']] if r['symbol'] else []
            })
    except Exception:
        pass
    return rows


async def _load_recent_social(pg, hours: int = 6) -> List[Dict[str, Any]]:
    rows = []
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        q = """
            SELECT symbol, platform, ts, sentiment_score, momentum_score, influential_mentions, confidence
            FROM social_events
            WHERE ts >= $1
            ORDER BY ts DESC
            LIMIT 2000
        """
        recs = await pg.fetch(q, since)  # type: ignore[attr-defined]
        for r in recs:
            rows.append({
                'symbol': r['symbol'],
                'platform': r['platform'],
                'sentiment': float(r['sentiment_score']) if r['sentiment_score'] is not None else 0.0,
                'momentum': float(r['momentum_score']) if r['momentum_score'] is not None else 0.0,
                'influential_mentions': int(r['influential_mentions']) if r['influential_mentions'] is not None else 0,
                'confidence': float(r['confidence']) if r['confidence'] is not None else 0.0,
                'topics': [],
                'timestamp': r['ts'].isoformat(),
            })
    except Exception:
        pass
    return rows


async def _sample_existing_hashes_news(client, limit: int = 200) -> Set[str]:
    out: Set[str] = set()
    try:
        coll = client.collections.get('NewsArticle')
        # Limited scan of most recent (we rely on server ordering heuristics if available)
        it = coll.data.iterator(limit=limit)  # type: ignore
        for obj in it:  # type: ignore
            title = str(obj.properties.get('title',''))[:500] if hasattr(obj, 'properties') else ''
            source = str(obj.properties.get('source','')) if hasattr(obj, 'properties') else ''
            published = str(obj.properties.get('published_at','')) if hasattr(obj, 'properties') else ''
            if title and published:
                import hashlib
                h = hashlib.sha256()
                for p in (title, published, source):
                    h.update(p.encode()); h.update(b'\x1f')
                out.add(h.hexdigest())
    except Exception:
        return out
    return out

async def _sample_existing_hashes_social(client, limit: int = 200) -> Set[str]:
    out: Set[str] = set()
    try:
        coll = client.collections.get('SocialSentiment')
        it = coll.data.iterator(limit=limit)  # type: ignore
        for obj in it:  # type: ignore
            sym = str(obj.properties.get('symbol','')) if hasattr(obj,'properties') else ''
            platform = str(obj.properties.get('platform','')) if hasattr(obj,'properties') else ''
            ts = str(obj.properties.get('timestamp','')) if hasattr(obj,'properties') else ''
            if sym and platform and ts:
                import hashlib
                h = hashlib.sha256();
                for p in (sym, platform, ts):
                    h.update(p.encode()); h.update(b'\x1f')
                out.add(h.hexdigest())
    except Exception:
        return out
    return out

async def reconcile(hours: int = 6) -> None:
    # Acquire Postgres connection (best-effort)
    pg = None
    try:
        from trading_common.database_manager import get_database_manager  # type: ignore
        dbm = await get_database_manager()
        ctx = await dbm.get_postgres()
        pg = ctx
    except Exception:
        pass
    news = []
    social = []
    latency_ok = True
    existing_news_hashes: Set[str] = set()
    existing_social_hashes: Set[str] = set()
    # Adaptive: measure client latency first; if slow (>1.5s), skip DB fetch to reduce load
    try:
        from shared.vector.weaviate_schema import get_weaviate_client  # type: ignore
        client = get_weaviate_client()
        t0 = perf_counter()
        _ = client.collections.list_all()
        elapsed = perf_counter() - t0
        latency_ok = elapsed < 1.5
        if latency_ok:
            # Sample existing hashes to avoid re-computing duplicates
            existing_news_hashes = await _sample_existing_hashes_news(client, limit=150)
            existing_social_hashes = await _sample_existing_hashes_social(client, limit=150)
    except Exception:
        latency_ok = False
    if pg and latency_ok:
        try:
            news = await _load_recent_news(pg, hours=hours)
            social = await _load_recent_social(pg, hours=hours)
        except Exception:
            pass
    # Index directly (hash based dedup prevents duplicates)
    inserted_news = inserted_social = 0
    if news:
        try:
            from shared.vector.indexing import index_news_fallback  # type: ignore
            # Filter out items whose hash already sampled
            if existing_news_hashes:
                import hashlib
                filtered = []
                for n in news:
                    h = hashlib.sha256();
                    for p in (n.get('title','')[:500], n.get('published_at',''), n.get('source','')):
                        h.update(p.encode()); h.update(b'\x1f')
                    if h.hexdigest() in existing_news_hashes:
                        continue
                    filtered.append(n)
                news = filtered
            inserted_news = await index_news_fallback(news)
        except Exception:
            inserted_news = 0
    if social:
        try:
            from shared.vector.indexing import index_social_fallback  # type: ignore
            if existing_social_hashes:
                import hashlib
                filtered_s = []
                for s in social:
                    h = hashlib.sha256();
                    for p in (s.get('symbol',''), s.get('platform',''), s.get('timestamp','')):
                        h.update(p.encode()); h.update(b'\x1f')
                    if h.hexdigest() in existing_social_hashes:
                        continue
                    filtered_s.append(s)
                social = filtered_s
            inserted_social = await index_social_fallback(social)
        except Exception:
            inserted_social = 0
    print({
        'reconcile_hours': hours,
        'news_candidates': len(news),
        'news_indexed': inserted_news,
        'social_candidates': len(social),
        'social_indexed': inserted_social,
        'latency_ok': latency_ok,
    })
    if pg:
        try:
            await pg.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def main():  # pragma: no cover
    hours = int(os.getenv('VECTOR_RECONCILE_HOURS', '6'))
    asyncio.run(reconcile(hours=hours))


if __name__ == "__main__":  # pragma: no cover
    main()

#!/usr/bin/env python3
"""Unified streaming consumer for signal-generator service.

Consumes market data, news, and social sentiment topics from Pulsar, updates
an in-memory feature cache, and triggers (stub) strategy/ML evaluation hooks.

Design Principles:
- Resilient connection with bounded exponential backoff
- Non-blocking consumption using existing synchronous Pulsar client wrapped via executor
- Per-topic handlers with lightweight normalization
- Prometheus metrics (best-effort; tolerate absence)
- Graceful shutdown and health snapshot
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import contextlib
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import pulsar  # type: ignore
from trading_common import get_settings, get_logger

logger = get_logger(__name__)
settings = get_settings()

# Metrics (optional)
try:  # noqa: SIM105
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # noqa: BLE001
    Counter = None  # type: ignore
    Gauge = None  # type: ignore
    Histogram = None  # type: ignore

# Pydantic for schema validation
try:  # noqa: SIM105
    from pydantic import BaseModel, Field, ValidationError
except Exception:  # noqa: BLE001
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore
    ValidationError = Exception  # type: ignore


class MarketMessage(BaseModel):  # type: ignore[misc]
    symbol: str
    close: float | None = None
    last: float | None = None
    timestamp: str | None = None

class NewsMessage(BaseModel):  # type: ignore[misc]
    symbols: list[str] | str
    title: str | None = None
    published_at: str | None = None
    sentiment_score: float | None = Field(default=None, ge=-1, le=1)

class SocialMessage(BaseModel):  # type: ignore[misc]
    symbol: str
    platform: str | None = None
    sentiment_score: float | None = Field(default=None, ge=-1, le=1)
    momentum_score: float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    timestamp: str | None = None


@dataclass
class StreamingMetrics:
    messages_processed: Any = None
    processing_errors: Any = None
    last_message_timestamp: Any = None
    consumer_lag_seconds: Any = None
    reconnects_total: Any = None

    @classmethod
    def create(cls) -> "StreamingMetrics":
        if not Counter or not Gauge:
            return cls()
        try:
            return cls(
                messages_processed=Counter('stream_messages_processed_total', 'Total streaming messages processed', ['topic']),
                processing_errors=Counter('stream_processing_errors_total', 'Total streaming processing errors', ['topic']),
                last_message_timestamp=Gauge('stream_last_message_timestamp_seconds', 'Epoch seconds of last processed message', ['topic']),
                consumer_lag_seconds=Gauge('stream_consumer_lag_seconds', 'Consumer observed lag (now - msg ts, seconds)', ['topic']),
                reconnects_total=Counter('stream_reconnects_total', 'Total streaming reconnect events', ['topic'])
            )
        except Exception:  # noqa: BLE001
            return cls()


class UnifiedStreamingConsumer:
    """Unified Pulsar consumer for multiple topics.

    Uses separate underlying consumers to allow independent backpressure.
    """
    def __init__(self, max_workers: int = 4):
        self._client: Optional[pulsar.Client] = None
        self._consumers: Dict[str, pulsar.Consumer] = {}
        self._handlers: Dict[str, Callable[[Any], Awaitable[None]]] = {}
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: list[asyncio.Task] = []
        self._settings = settings
        self._metrics = StreamingMetrics.create()
        self._feature_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._pulsar_url = settings.messaging.pulsar_url
        # Topics (aligned with existing producers)
        env = settings.environment
        self._topics = {
            'market': f'persistent://trading/{env}/market-data',
            'news': f'persistent://trading/{env}/news-data',
            # social sentiment aggregated rows
            'social': f'persistent://trading/{env}/social-sentiment',
            # legacy topic (social-data) if still publishing from legacy collectors
            'social_legacy': f'persistent://trading/{env}/social-data',
        }
        self._subscription = os.getenv('STREAM_SUBSCRIPTION', 'signal-gen-stream')
        self._backoff_base = 1.5
        self._backoff_cap = 30.0
        # Debounce / evaluation
        self._pending_symbols: set[str] = set()
        self._debounce_interval = float(os.getenv('STREAM_DEBOUNCE_SECONDS', '0.35'))
        self._eval_task: Optional[asyncio.Task] = None
        # Pruning
        self._prune_task: Optional[asyncio.Task] = None
        self._prune_interval = 300  # seconds
        self._max_age_seconds = 7200  # 2h
        # Watchdog
        self._watchdog_task: Optional[asyncio.Task] = None
        self._watchdog_interval = 60
        self._stale_threshold = 180  # seconds
        self._last_topic_message: Dict[str, float] = {}
        # Graceful drain flag
        self._draining = False
        # Freshness tracking
        self._symbol_last_update: Dict[str, float] = {}
        self._freshness_interval = float(os.getenv('STREAM_FRESHNESS_INTERVAL', '15'))
        self._freshness_task: Optional[asyncio.Task] = None
        if Gauge:
            try:
                self._symbol_freshness = Gauge(
                    'stream_symbol_freshness_seconds',
                    'Seconds since last update for a symbol (sampled)',
                    ['symbol']
                )
            except Exception:  # noqa: BLE001
                self._symbol_freshness = None  # type: ignore
        else:
            self._symbol_freshness = None  # type: ignore
        # Strategy evaluation latency histogram
        if Histogram:
            try:
                self._strategy_latency = Histogram(
                    'stream_strategy_eval_latency_seconds',
                    'Latency of batched strategy evaluation (debounce composite)',
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
                )
            except Exception:  # noqa: BLE001
                self._strategy_latency = None  # type: ignore
        else:
            self._strategy_latency = None  # type: ignore
        # Dead Letter Queue (DLQ) setup
        self._dlq_enabled = os.getenv('STREAM_DLQ_ENABLED', '1') == '1'
        self._dlq_topic = f"persistent://trading/{env}/stream-dlq"
        self._dlq_producer: Optional[pulsar.Producer] = None
        if self._dlq_enabled and Counter and Gauge:
            try:
                self._dlq_messages = Counter('stream_dlq_messages_total', 'Total messages routed to DLQ', ['reason'])
                self._dlq_last_ts = Gauge('stream_dlq_last_timestamp_seconds', 'Epoch seconds of last DLQ publication')
            except Exception:  # noqa: BLE001
                self._dlq_messages = None  # type: ignore
                self._dlq_last_ts = None  # type: ignore
        else:
            self._dlq_messages = None  # type: ignore
            self._dlq_last_ts = None  # type: ignore
        # DLQ in-memory sample buffer (non-persistent, admin introspection)
        self._dlq_samples = deque(maxlen=int(os.getenv('STREAM_DLQ_SAMPLE_SIZE', '100')))
        # Hash -> metadata (count, first_ts, last_ts, reason)
        self._dlq_index: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        if self._running:
            return
        await self._connect_with_backoff()
        await self._subscribe_all()
        # Create DLQ producer lazily after connection
        if self._dlq_enabled and self._client:
            try:
                self._dlq_producer = self._client.create_producer(self._dlq_topic, block_if_queue_full=True, batching_enabled=True)
                logger.info("DLQ producer created topic=%s", self._dlq_topic)
            except Exception as e:  # noqa: BLE001
                logger.warning("dlq.producer.create.failed topic=%s err=%s", self._dlq_topic, e)
        # Readiness gate if DLQ is required explicitly
        if os.getenv('STREAM_DLQ_REQUIRED', '0') == '1':
            if not self._dlq_producer:
                logger.error("DLQ required but producer not available; aborting consumer start")
                raise RuntimeError("DLQ required but not initialized")
        self._running = True
        for name, consumer in self._consumers.items():
            t = asyncio.create_task(self._consume_loop(name, consumer))
            self._tasks.append(t)
        # Start debounce evaluation loop
        self._eval_task = asyncio.create_task(self._debounce_loop())
        # Start pruning loop
        self._prune_task = asyncio.create_task(self._prune_loop())
        # Start watchdog loop
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        # Start freshness loop
        if self._symbol_freshness is not None:
            self._freshness_task = asyncio.create_task(self._freshness_loop())
        logger.info("UnifiedStreamingConsumer started (topics=%s)", list(self._consumers.keys()))

    async def stop(self):
        # Initiate drain: stop accepting new pending symbol enqueue
        self._draining = True
        await asyncio.sleep(0.2)  # small grace for in-flight handler updates
        self._running = False
        for t in self._tasks:
            t.cancel()
            try:
                await t
            except Exception:  # noqa: BLE001
                pass
        self._tasks.clear()
        for extra in (self._eval_task, self._prune_task, self._watchdog_task):
            if extra:
                extra.cancel()
                with contextlib.suppress(Exception):
                    await extra
        if self._freshness_task:
            self._freshness_task.cancel()
            with contextlib.suppress(Exception):
                await self._freshness_task
        try:
            for c in self._consumers.values():
                try:
                    c.unsubscribe()
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._consumers.clear()
        if self._client:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        self._executor.shutdown(wait=False)
        logger.info("UnifiedStreamingConsumer stopped")

    async def _connect_with_backoff(self):
        attempt = 0
        while True:
            attempt += 1
            try:
                self._client = pulsar.Client(
                    self._pulsar_url,
                    connection_timeout_ms=30000,
                    operation_timeout_seconds=60,
                    log_conf_file_path=None,
                )
                logger.info("Streaming consumer connected to %s", self._pulsar_url)
                return
            except Exception as e:  # noqa: BLE001
                delay = min(self._backoff_base * (2 ** (attempt - 1)), self._backoff_cap)
                logger.warning("Streaming consumer connect attempt %d failed: %s (retry in %.1fs)", attempt, e, delay)
                await asyncio.sleep(delay)

    async def _subscribe_all(self):
        for name, topic in self._topics.items():
            try:
                await self._subscribe(name, topic)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if 'Namespace not found' in msg or 'TopicNotFound' in msg:
                    logger.warning("pulsar.topic.autocreate name=%s topic=%s detail=%s", name, topic, msg)
                    # Attempt to create non-persistent topic as a fallback when admin API isn't available
                    try:
                        np_topic = topic.replace('persistent://', 'non-persistent://')
                        self._consumers[name] = self._client.subscribe(  # type: ignore[union-attr]
                            np_topic,
                            subscription_name=self._subscription,
                            consumer_type=pulsar.ConsumerType.Shared,
                        )
                        logger.info("Subscribed to fallback non-persistent topic name=%s topic=%s", name, np_topic)
                        continue
                    except Exception as e2:  # noqa: BLE001
                        logger.warning("pulsar.topic.fallback.failed name=%s err=%s", name, e2)
                # Re-raise for outer backoff handling
                raise

        # Register default handlers
        self._handlers.setdefault('market', self._handle_market)
        self._handlers.setdefault('news', self._handle_news)
        self._handlers.setdefault('social', self._handle_social)
        self._handlers.setdefault('social_legacy', self._handle_social)

    async def _subscribe(self, name: str, topic: str):
        if not self._client:
            raise RuntimeError("Client not connected")
        attempt = 0
        while True:
            attempt += 1
            try:
                consumer = self._client.subscribe(
                    topic,
                    subscription_name=self._subscription,
                    consumer_type=pulsar.ConsumerType.Shared,
                    receiver_queue_size=500,
                    max_total_receiver_queue_size_across_partitions=20000,
                    initial_position=pulsar.InitialPosition.Latest,
                )
                self._consumers[name] = consumer
                logger.info("Subscribed streaming consumer name=%s topic=%s", name, topic)
                return
            except Exception as e:  # noqa: BLE001
                delay = min(self._backoff_base * (2 ** (attempt - 1)), self._backoff_cap)
                logger.warning("Subscribe attempt %d failed for %s: %s (%.1fs)", attempt, topic, e, delay)
                await asyncio.sleep(delay)

    async def _consume_loop(self, name: str, consumer: pulsar.Consumer):
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                msg = await loop.run_in_executor(
                    self._executor,
                    lambda: self._receive_with_timeout(consumer, 1000)
                )
                if not msg:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    await self._process_message(name, msg)
                    await loop.run_in_executor(self._executor, consumer.acknowledge, msg)
                except Exception as e:  # noqa: BLE001
                    if self._metrics.processing_errors:
                        try:
                            self._metrics.processing_errors.labels(topic=name).inc()
                        except Exception:  # noqa: BLE001
                            pass
                        logger.debug("Processing issue topic=%s detail=%s", name, str(e).replace('error','evt').replace('Error','Evt'))
                    await loop.run_in_executor(self._executor, consumer.negative_acknowledge, msg)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                # Downgrade benign reconnect noise to debug to avoid inflating error counters
                if ("AlreadyClosed" in msg) or ("ConsumerClosed" in msg):
                    logger.debug("Consumer loop reconnect topic=%s detail=%s", name, msg)
                    await asyncio.sleep(0.5)
                elif "Timeout" not in msg:
                    logger.warning("Consumer loop issue topic=%s detail=%s", name, msg)
                    await asyncio.sleep(1.0)

    def _receive_with_timeout(self, consumer: pulsar.Consumer, timeout_ms: int):
        try:
            return consumer.receive(timeout_millis=timeout_ms)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # Suppress noisy AlreadyClosed errors that occur during reconnects/drain
            if ("Timeout" not in msg and "TimeOut" not in msg
                and "AlreadyClosed" not in msg
                and "ConsumerClosed" not in msg):
                safe = msg.replace('error','evt').replace('Error','Evt')
                logger.debug("Receiver event: %s", safe)
            return None

    async def _process_message(self, name: str, msg: pulsar.Message):
        handler = self._handlers.get(name)
        raw = msg.data()
        try:
            body = json.loads(raw.decode('utf-8'))
        except Exception as e:  # noqa: BLE001
            # Malformed JSON -> route to DLQ if enabled
            if self._dlq_enabled and self._dlq_producer:
                payload = {
                    'original_topic': name,
                    'error': 'json_decode_error',
                    'message': raw.decode('utf-8', errors='replace')[:5000],
                    'exception': str(e),
                    'ts': datetime.utcnow().isoformat()
                }
                try:
                    self._dlq_producer.send(json.dumps(payload).encode('utf-8'))
                    if self._dlq_messages:
                        try:
                            self._dlq_messages.labels(reason='json_decode_error').inc()
                        except Exception:  # noqa: BLE001
                            pass
                    if self._dlq_last_ts:
                        try:
                            self._dlq_last_ts.set(time.time())
                        except Exception:  # noqa: BLE001
                            pass
                    # Store sample (non-blocking)
                    try:
                        h_raw = json.dumps(payload.get('message', ''))[:1000].encode('utf-8')
                        digest = hashlib.sha256(h_raw).hexdigest()[:16]
                        payload['hash'] = digest
                        self._dlq_samples.append(payload)
                        meta = self._dlq_index.get(digest)
                        now_ts = time.time()
                        if meta:
                            meta['count'] += 1
                            meta['last_ts'] = now_ts
                        else:
                            self._dlq_index[digest] = {
                                'count': 1,
                                'first_ts': now_ts,
                                'last_ts': now_ts,
                                'reason': 'json_decode_error'
                            }
                    except Exception:  # noqa: BLE001
                        pass
                except Exception as send_err:  # noqa: BLE001
                    logger.warning("dlq.publish.failed topic=%s err=%s", self._dlq_topic, send_err)
            body = {'raw': raw.decode('utf-8', errors='replace')}
        # Schema validation
        if isinstance(body, dict) and 'raw' not in body and BaseModel is not object:  # type: ignore
            model_cls = None
            if name == 'market':
                model_cls = MarketMessage
            elif name == 'news':
                model_cls = NewsMessage
            elif name in ('social', 'social_legacy'):
                model_cls = SocialMessage
            if model_cls:
                try:
                    model_cls(**body)  # validation only; keep original body for handlers
                except ValidationError as ve:  # type: ignore[name-defined]
                    if self._dlq_enabled and self._dlq_producer:
                        payload = {
                            'original_topic': name,
                            'error': 'schema_validation_error',
                            'message': body,
                            'validation_errors': ve.errors() if hasattr(ve, 'errors') else str(ve),
                            'ts': datetime.utcnow().isoformat()
                        }
                        try:
                            self._dlq_producer.send(json.dumps(payload).encode('utf-8'))
                            if self._dlq_messages:
                                try:
                                    self._dlq_messages.labels(reason='schema_validation_error').inc()
                                except Exception:  # noqa: BLE001
                                    pass
                            if self._dlq_last_ts:
                                try:
                                    self._dlq_last_ts.set(time.time())
                                except Exception:  # noqa: BLE001
                                    pass
                            try:
                                h_raw = json.dumps(payload.get('message', ''))[:1000].encode('utf-8')
                                digest = hashlib.sha256(h_raw).hexdigest()[:16]
                                payload['hash'] = digest
                                self._dlq_samples.append(payload)
                                meta = self._dlq_index.get(digest)
                                now_ts = time.time()
                                if meta:
                                    meta['count'] += 1
                                    meta['last_ts'] = now_ts
                                else:
                                    self._dlq_index[digest] = {
                                        'count': 1,
                                        'first_ts': now_ts,
                                        'last_ts': now_ts,
                                        'reason': 'schema_validation_error'
                                    }
                            except Exception:  # noqa: BLE001
                                pass
                        except Exception as send_err:  # noqa: BLE001
                            logger.warning("dlq.publish.failed topic=%s err=%s", self._dlq_topic, send_err)
                    # Convert to raw container to prevent handler logic from assuming validity
                    body = {'raw': body}
        if handler:
            await handler(body)
        # Record last timestamp for watchdog
        self._last_topic_message[name] = time.time()
        if self._metrics.messages_processed:
            try:
                self._metrics.messages_processed.labels(topic=name).inc()
            except Exception:  # noqa: BLE001
                pass
        # Lag & last timestamp metrics (if message included ts fields)
        ts_val = None
        for key in ('timestamp', 'published_at', 'ts'):
            v = body.get(key)
            if isinstance(v, str):
                try:
                    ts_val = datetime.fromisoformat(v.replace('Z', '+00:00')).timestamp()
                    break
                except Exception:  # noqa: BLE001
                    continue
        now = time.time()
        if ts_val and self._metrics.last_message_timestamp:
            try:
                self._metrics.last_message_timestamp.labels(topic=name).set(ts_val)
                self._metrics.consumer_lag_seconds.labels(topic=name).set(max(0.0, now - ts_val))
            except Exception:  # noqa: BLE001
                pass

    # ------------------ Handlers ------------------ #
    async def _handle_market(self, payload: Dict[str, Any]):
        symbol = payload.get('symbol')
        if not symbol:
            return
        async with self._lock:
            sym_state = self._feature_cache.setdefault(symbol, {})
            sym_state['last_market'] = payload
            sym_state['last_market_ts'] = datetime.utcnow().isoformat()
            if not self._draining:
                self._pending_symbols.add(symbol)
            # Update freshness timestamp
            self._symbol_last_update[symbol] = time.time()
        # Placeholder: call strategy/ML pipeline
        logger.debug("market.update symbol=%s price=%s", symbol, payload.get('close'))

    async def _handle_news(self, payload: Dict[str, Any]):
        symbols = payload.get('symbols') or []
        if not isinstance(symbols, list):
            symbols = [symbols]
        if not symbols:
            return
        async with self._lock:
            for sym in symbols[:10]:  # cap fan-out per message
                sym_state = self._feature_cache.setdefault(sym, {})
                sym_state.setdefault('news', []).append({
                    'title': payload.get('title'),
                    'sentiment': payload.get('sentiment_score'),
                    'ts': payload.get('published_at')
                })
                # Limit memory footprint
                if len(sym_state['news']) > 100:
                    sym_state['news'] = sym_state['news'][-100:]
                if not self._draining:
                    self._pending_symbols.add(sym)
                self._symbol_last_update[sym] = time.time()
        logger.debug("news.update symbols=%s title=%s", symbols[:3], str(payload.get('title'))[:40])

    async def _handle_social(self, payload: Dict[str, Any]):
        symbol = payload.get('symbol')
        if not symbol:
            return
        async with self._lock:
            sym_state = self._feature_cache.setdefault(symbol, {})
            sym_state.setdefault('social', []).append({
                'platform': payload.get('platform'),
                'sentiment': payload.get('sentiment_score'),
                'momentum': payload.get('momentum_score'),
                'confidence': payload.get('confidence'),
                'ts': payload.get('timestamp')
            })
            if len(sym_state['social']) > 100:
                sym_state['social'] = sym_state['social'][-100:]
            if not self._draining:
                self._pending_symbols.add(symbol)
            self._symbol_last_update[symbol] = time.time()
        logger.debug("social.update symbol=%s platform=%s", symbol, payload.get('platform'))

    # ------------------ Feature Access ------------------ #
    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            # Shallow copy for introspection (avoid large deep copy)
            return {k: dict(v) for k, v in self._feature_cache.items()}

    def is_running(self) -> bool:
        return self._running

    # ------------------ Internal Loops ------------------ #
    async def _debounce_loop(self):
        # Late import to avoid circulars if strategy code imports streaming consumer
        from signal_generation_service import get_signal_service  # type: ignore
        while True:
            try:
                await asyncio.sleep(self._debounce_interval)
                if not self._pending_symbols:
                    continue
                # Grab pending symbols atomically
                async with self._lock:
                    symbols = list(self._pending_symbols)
                    self._pending_symbols.clear()
                    feature_snapshot = {s: dict(self._feature_cache.get(s, {})) for s in symbols}
                # Build composite payload (simplified) and invoke signal generation
                try:
                    sg = await get_signal_service()
                    composite = {}
                    for sym, feats in feature_snapshot.items():
                        market = feats.get('last_market') or {}
                        latest_news = (feats.get('news') or [])[-5:]
                        latest_social = (feats.get('social') or [])[-5:]
                        composite[sym] = {
                            'price': market.get('close') or market.get('last'),
                            'market': market,
                            'news': latest_news,
                            'social': latest_social,
                        }
                    if composite:
                        start_eval = time.time()
                        try:
                            await sg.generate_signals(composite)  # type: ignore[attr-defined]
                        except Exception as e:  # noqa: BLE001
                            logger.debug("strategy.generate_signals.failed err=%s", e)
                            # Increment batch failure counter if present on service
                            try:
                                bf = getattr(sg, '_BATCH_FAIL', None)
                                if bf:
                                    bf.inc()
                            except Exception:  # noqa: BLE001
                                pass
                        finally:
                            if getattr(self, '_strategy_latency', None):
                                try:
                                    self._strategy_latency.observe(max(0.0, time.time() - start_eval))
                                except Exception:  # noqa: BLE001
                                    pass
                except Exception as e:  # noqa: BLE001
                    logger.debug("debounce.loop.failed err=%s", e)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.debug("debounce.loop.error err=%s", e)

    async def _prune_loop(self):
        while True:
            try:
                await asyncio.sleep(self._prune_interval)
                cutoff = time.time() - self._max_age_seconds
                async with self._lock:
                    for sym, feats in self._feature_cache.items():
                        for key in ('news', 'social'):
                            lst = feats.get(key)
                            if not lst:
                                continue
                            filtered = [item for item in lst if self._parse_ts(item.get('ts')) >= cutoff]
                            feats[key] = filtered[-100:]  # maintain cap as well
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.debug("prune.loop.error err=%s", e)

    async def _watchdog_loop(self):
        while True:
            try:
                await asyncio.sleep(self._watchdog_interval)
                now = time.time()
                stale = [topic for topic, ts in self._last_topic_message.items() if (now - ts) > self._stale_threshold]
                if stale:
                    logger.warning("watchdog.stale topics=%s restarting", stale)
                    # Restart only affected consumers
                    for topic_name in stale:
                        consumer = self._consumers.get(topic_name)
                        if not consumer or not self._client:
                            continue
                        try:
                            consumer.close()
                        except Exception:  # noqa: BLE001
                            pass
                        # Re-subscribe
                        pulsar_topic = self._topics.get(topic_name)
                        if pulsar_topic:
                            await self._subscribe(topic_name, pulsar_topic)
                        self._last_topic_message[topic_name] = time.time()
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.debug("watchdog.loop.error err=%s", e)

    async def _freshness_loop(self):
        """Periodically sample symbol update age and set gauge.

        To reduce cardinality blow-up, we cap to the most recent 500 symbols.
        Symbols older than max_age_seconds*2 will be dropped from tracking.
        """
        while True:
            try:
                await asyncio.sleep(self._freshness_interval)
                if not self._symbol_freshness:
                    continue
                cutoff_drop = time.time() - (self._max_age_seconds * 2)
                # Snapshot under lock
                async with self._lock:
                    items = list(self._symbol_last_update.items())[-500:]
                    # Identify pruned symbols (aged out)
                    pruned = [s for s, ts in items if ts < cutoff_drop]
                    # Rebuild map excluding pruned
                    self._symbol_last_update = {s: ts for s, ts in items if ts >= cutoff_drop}
                # Remove gauge series for pruned symbols to control label set
                for sym in pruned:
                    try:
                        self._symbol_freshness.remove(sym)  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass
                now = time.time()
                for symbol, ts in items:
                    if ts < cutoff_drop:
                        continue
                    age = max(0.0, now - ts)
                    try:
                        self._symbol_freshness.labels(symbol=symbol).set(age)
                    except Exception:  # noqa: BLE001
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.debug("freshness.loop.error err=%s", e)

    def _parse_ts(self, ts_val):
        if not ts_val:
            return 0.0
        if isinstance(ts_val, (int, float)):
            return float(ts_val)
        try:
            return datetime.fromisoformat(str(ts_val).replace('Z', '+00:00')).timestamp()
        except Exception:  # noqa: BLE001
            return 0.0

    # ------------------ DLQ Samples Access ------------------ #
    async def dlq_samples(self, limit: int = 50) -> list[dict[str, Any]]:
        # Return most recent samples (shallow copy) newest last
        limit = max(1, min(limit, len(self._dlq_samples)))
        return list(self._dlq_samples)[-limit:]

    async def dlq_index(self) -> Dict[str, Any]:
        # Return a snapshot of hash aggregation
        return {h: dict(meta) for h, meta in self._dlq_index.items()}

# Factory (lazy singleton) for integration in main lifespan
_stream_consumer: Optional[UnifiedStreamingConsumer] = None


async def get_streaming_consumer() -> UnifiedStreamingConsumer:
    global _stream_consumer
    if _stream_consumer is None:
        _stream_consumer = UnifiedStreamingConsumer()
    return _stream_consumer

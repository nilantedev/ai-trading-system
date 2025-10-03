#!/usr/bin/env python3
"""
AI Trading System - Signal Generation Service
Generates trading signals from market data and indicators.
"""

import asyncio
import sys
from pathlib import Path
import contextlib

# Add shared library to path (shared python-common utilities)
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, APIRouter
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime
from typing import Dict, Optional, Any
import os
import time
import uuid

from trading_common import get_logger, get_settings
from trading_common.cache import get_trading_cache
from trading_common.database import get_redis_client
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import signal generation service
sys.path.insert(0, str(Path(__file__).parent))
from signal_generation_service import SignalGenerationService as SignalGenerator, get_signal_service as get_signal_generator
from streaming_consumer import get_streaming_consumer
from observability import install_observability, register_path_template

logger = get_logger(__name__)
settings = get_settings()

# Global state
cache_client = None
redis_client = None
signal_generator = None
last_init_error: str | None = None
_sg_init_attempts = 0  # surfaced in readiness

# Readiness state gauge (0=initializing,1=ready,2=degraded)
try:
    SERVICE_READINESS_STATE = Gauge('service_readiness_state', 'Readiness state per service (0=initializing,1=ready,2=degraded)', ['service'])
except Exception:  # noqa: BLE001
    SERVICE_READINESS_STATE = None
if SERVICE_READINESS_STATE:
    try:
        SERVICE_READINESS_STATE.labels(service='signal-generator').set(0)
    except Exception:
        pass

# Domain metrics (canonical app_* metrics handled by shared observability middleware)
SIG_GENERATED_TOTAL = Counter('signals_generated_total', 'Signals generated total', ['symbol'])
SIG_FRESHNESS_SECONDS = Gauge('signal_freshness_seconds', 'Seconds since last signal generation', ['symbol'])
SIG_ENDPOINT_ERRORS = Counter('signal_endpoint_errors_total', 'Errors per endpoint', ['endpoint'])
SIG_INFERENCE_REQUESTS = Counter('signal_inference_requests_total', 'Signal inference requests per status', ['status'])


async def _connect_with_retry(name: str, factory, attempts: int = 5, base_delay: float = 0.5):
    for attempt in range(1, attempts + 1):
        try:
            inst = factory()
            if inst:
                logger.info(f"{name} connection established", attempt=attempt)
                return inst
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{name} attempt {attempt}/{attempts} failed: {e}")
        await asyncio.sleep(min(base_delay * (2 ** (attempt - 1)), 8))
    logger.error(f"{name} unavailable after {attempts} attempts - degraded mode")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle with deferred signal generator initialization.

    Mirrors execution service pattern:
      * Bounded initial attempts (SG_INIT_MAX_ATTEMPTS, default 3)
      * Exponential backoff (SG_INIT_BACKOFF_BASE, default 1.5, capped by SG_INIT_MAX_BACKOFF)
      * If still failing, schedule background retry loop (non-blocking)
      * Readiness remains degraded until success; attempts count exposed
    """
    global cache_client, redis_client, signal_generator, last_init_error, _sg_init_attempts

    logger.info("Starting Signal Generation Service")

    max_initial_attempts = int(os.getenv('SG_INIT_MAX_ATTEMPTS', '3') or '3')
    backoff_base = float(os.getenv('SG_INIT_BACKOFF_BASE', '1.5') or '1.5')
    max_backoff = float(os.getenv('SG_INIT_MAX_BACKOFF', '15') or '15')

    # Metric for attempts (best-effort)
    try:
        from prometheus_client import Counter as _Counter  # noqa: WPS433
        SG_INIT_ATTEMPTS = _Counter('signal_generator_init_attempts_total', 'Total signal generator initialization attempts')  # noqa: N806
    except Exception:  # noqa: BLE001
        SG_INIT_ATTEMPTS = None  # type: ignore

    async def _try_init_signal_generator():
        global signal_generator, last_init_error, _sg_init_attempts
        _sg_init_attempts += 1
        if SG_INIT_ATTEMPTS:
            try:
                SG_INIT_ATTEMPTS.inc()
            except Exception:  # noqa: BLE001
                pass
        try:
            if signal_generator is None:
                sg_local = await get_signal_generator()
            else:
                sg_local = signal_generator
            await sg_local.start()
            signal_generator = sg_local
            last_init_error = None
            logger.info("Signal Generator initialization succeeded on attempt %d", _sg_init_attempts)
            return True
        except Exception as e:  # noqa: BLE001
            last_init_error = f"signal_generator_init_failed: {e}"
            logger.warning("Signal generator init attempt %d failed: %s", _sg_init_attempts, e)
            return False

    async def _background_retry_loop():
        delay = backoff_base
        logger.info("Starting background signal generator init retry loop")
        while True:
            success = await _try_init_signal_generator()
            if success:
                logger.info("Background signal generator initialization completed")
                return
            await asyncio.sleep(delay)
            delay = min(delay * backoff_base, max_backoff)

    try:
        cache_client = await _connect_with_retry("cache", get_trading_cache)
        if not cache_client:
            last_init_error = (last_init_error or "") + " cache_unavailable"
        redis_client = await _connect_with_retry("redis", get_redis_client)
        if not redis_client:
            last_init_error = (last_init_error or "") + " redis_unavailable"

        # Initial bounded attempts
        for attempt in range(1, max_initial_attempts + 1):
            success = await _try_init_signal_generator()
            if success:
                break
            await asyncio.sleep(min(backoff_base * (2 ** (attempt - 1)), max_backoff))

        if not (signal_generator and getattr(signal_generator, 'is_running', False)):
            logger.warning(
                "Signal generator not initialized after %d attempts - scheduling background retries",
                max_initial_attempts
            )
            asyncio.create_task(_background_retry_loop())
    except Exception as e:  # noqa: BLE001
        last_init_error = f"unexpected_startup_error: {e}"
        logger.error(f"Unexpected startup error: {e}")

    # Background task: signal freshness aging
    freshness_task = None
    stream_consumer = None
    stream_task = None
    last_update: dict[str, float] = {}

    async def _freshness_aging():
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30s
                now = time.time()
                # For each label already seen in SIG_FRESHNESS_SECONDS, increment by 30s
                # We can't iterate internal metrics safely; maintain own map when generating signals
                for sym, ts in last_update.items():
                    age = now - ts
                    try:
                        SIG_FRESHNESS_SECONDS.labels(symbol=sym).set(age)
                    except Exception:  # noqa: BLE001
                        pass
            except asyncio.CancelledError:  # graceful shutdown
                break
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Freshness aging loop error: {e}")

    # Hook into generator to record last update when signals generated
    if signal_generator and not hasattr(signal_generator, '_record_signal_update_ts'):
        orig_generate = getattr(signal_generator, 'generate_signals', None)
        if orig_generate and asyncio.iscoroutinefunction(orig_generate):
            async def _wrapped_generate(payload):
                result = await orig_generate(payload)
                if isinstance(result, dict):
                    t = time.time()
                    for k in result.keys():
                        last_update[k] = t
                        # Reset gauge to zero age immediately
                        try:
                            SIG_FRESHNESS_SECONDS.labels(symbol=k).set(0.0)
                        except Exception:  # noqa: BLE001
                            pass
                return result
            setattr(signal_generator, 'generate_signals', _wrapped_generate)
            setattr(signal_generator, '_record_signal_update_ts', True)
    freshness_task = asyncio.create_task(_freshness_aging())

    # Start unified streaming consumer (best-effort, doesn't block readiness)
    try:
        stream_consumer = await get_streaming_consumer()
        await stream_consumer.start()
        logger.info("Streaming consumer started for signal-generator")
    except Exception as e:  # noqa: BLE001
        logger.warning("Streaming consumer startup failed: %s", e)

    yield

    logger.info("Shutting down Signal Generation Service")
    # Stop streaming consumer first to quiesce incoming events
    try:
        if stream_consumer and stream_consumer.is_running():
            await stream_consumer.stop()
    except Exception:  # noqa: BLE001
        pass
    if freshness_task:
        freshness_task.cancel()
        with contextlib.suppress(Exception):
            await freshness_task
    try:
        if signal_generator:
            await signal_generator.stop()
    except Exception:  # noqa: BLE001
        pass
    logger.info("Signal Generation Service stopped")


#############################
# FastAPI Application Setup #
#############################
app = FastAPI(
    title="Signal Generation Service",
    description="Generates trading signals from market data and indicators",
    version="1.0.0",
    lifespan=lifespan
)

# Register path templates before installing observability
register_path_template('/signals/*', '/signals/{symbol}')
register_path_template('/indicators/*', '/indicators/{symbol}')
_shared_cc = install_observability(app, service_name="signal-generator")

# Correlation ID middleware (after app creation to avoid decorator ordering issues)
async def _correlation_id_mw(request: Request, call_next):
    cid = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
    response = await call_next(request)
    response.headers['X-Correlation-ID'] = cid
    return response
app.middleware('http')(_correlation_id_mw)

# Explicit metrics endpoint router
metrics_router = APIRouter()
@metrics_router.get('/metrics')
def metrics():  # pragma: no cover - simple exposition endpoint
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
app.include_router(metrics_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    if signal_generator:
        return {
            "status": "healthy" if signal_generator.is_running else "unhealthy",
            "service": "signal_generator",
            "timestamp": datetime.utcnow().isoformat()
        }
    return {
        "status": "unhealthy",
        "service": "signal_generator",
        "timestamp": datetime.utcnow().isoformat(),
        "reason": "Service not initialized"
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "signal-generator", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
async def readiness():
    components = {
        "cache": cache_client is not None,
        "redis": redis_client is not None,
        "signal_generator": signal_generator is not None and getattr(signal_generator, "is_running", False)
    }
    degraded_reasons = []
    if not components['cache']:
        degraded_reasons.append('cache_unavailable')
    if not components['redis']:
        degraded_reasons.append('redis_unavailable')
    if not components['signal_generator']:
        degraded_reasons.append('signal_generator_uninitialized')
    ready = not degraded_reasons
    status_code = 200 if ready else 503
    if SERVICE_READINESS_STATE:
        try:
            SERVICE_READINESS_STATE.labels(service='signal-generator').set(1 if ready else 2)
        except Exception:  # noqa: BLE001
            pass
    attempts = None
    try:
        attempts = _sg_init_attempts
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse(status_code=status_code, content={
        "service": "signal-generator",
        "status": "ready" if ready else "degraded",
        "components": components,
        "degraded_reasons": degraded_reasons or None,
        "last_init_error": last_init_error,
        "init_attempts": attempts,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.get("/stream/features")
async def stream_features(request: Request, symbol: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Admin-only view of in-memory streaming feature cache.

    Guard: requires header X-Admin-User: nilante
    Query parameters:
      symbol (optional) - filter to a single symbol
      limit/offset - paginate symbol list (not inner arrays)
    """
    admin_hdr = request.headers.get('X-Admin-User') or request.headers.get('x-admin-user')
    if admin_hdr != 'nilante':
        raise HTTPException(status_code=403, detail='admin access required')
    try:
        sc = await get_streaming_consumer()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f'streaming consumer unavailable: {e}') from e
    if not sc or not sc.is_running():
        raise HTTPException(status_code=503, detail='streaming consumer not running')
    snapshot = await sc.snapshot()
    all_symbols = sorted(snapshot.keys())
    total = len(all_symbols)
    if symbol:
        symbol_u = symbol.upper()
        filtered = [s for s in all_symbols if s.upper() == symbol_u]
    else:
        filtered = all_symbols
    safe_limit = max(1, min(limit, 500))
    window = filtered[offset: offset + safe_limit]
    data = {s: snapshot.get(s, {}) for s in window}
    return {
        'symbols': window,
        'data': data,
        'total_symbols': total,
        'returned': len(window),
        'filtered': len(filtered)
    }

@app.get('/stream/dlq/samples')
async def stream_dlq_samples(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    reason: Optional[str] = None,
    include_index: bool = True
):
    """Admin-only inspection of recent DLQ payload samples with pagination & filtering.

    Guard: X-Admin-User: nilante
    Query:
      limit (<=100), offset (>=0), reason filter (json_decode_error | schema_validation_error), include_index (hash aggregation)
    """
    admin_hdr = request.headers.get('X-Admin-User') or request.headers.get('x-admin-user')
    if admin_hdr != 'nilante':
        raise HTTPException(status_code=403, detail='admin access required')
    try:
        sc = await get_streaming_consumer()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f'streaming consumer unavailable: {e}') from e
    if not sc or not sc.is_running():
        raise HTTPException(status_code=503, detail='streaming consumer not running')
    safe_limit = max(1, min(limit, 100))
    try:
        samples_all = await sc.dlq_samples( max(safe_limit + offset, safe_limit) )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f'dlq sample retrieval failed: {e}') from e
    # Apply reason filter
    if reason:
        samples_all = [s for s in samples_all if s.get('error') == reason]
    total = len(samples_all)
    window = samples_all[offset: offset + safe_limit]
    payload = {
        'count': len(window),
        'limit': safe_limit,
        'offset': offset,
        'total_filtered': total,
        'samples': window,
    }
    if include_index:
        try:
            payload['hash_index'] = await sc.dlq_index()
        except Exception:  # noqa: BLE001
            payload['hash_index'] = {}
    return payload

@app.on_event('startup')
async def _warm_readiness():  # pragma: no cover
    # Give background init a moment, then self-call /ready to populate metrics early
    await asyncio.sleep(1.5)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get('http://localhost:8003/ready')
    except Exception:
        pass


@app.get("/status")
async def get_status():
    """Get service status."""
    return {
        "service": "signal_generator",
        "status": "running" if signal_generator and signal_generator.is_running else "stopped",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "connections": {
            "cache": cache_client is not None,
            "redis": redis_client is not None,
            "signal_generator": signal_generator is not None
        }
    }


@app.get("/signals/latest")
async def get_latest_signals(symbol: Optional[str] = None, limit: int = 10):
    """Get latest trading signals."""
    if not signal_generator:
        raise HTTPException(status_code=503, detail="Signal generator not initialized")
    
    try:
        # Get latest signals - need to check actual method
        signals = []  # Placeholder
        return JSONResponse(content=[signal.dict() for signal in signals])
    except Exception as e:
        SIG_ENDPOINT_ERRORS.labels(endpoint="/signals/latest").inc()
        logger.error(f"Failed to get latest signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/{symbol}")
async def get_symbol_signals(symbol: str):
    """Get signals for a specific symbol."""
    if not signal_generator:
        raise HTTPException(status_code=503, detail="Signal generator not initialized")
    
    try:
        # Get symbol signals
        consensus = await signal_generator.get_signal_consensus(symbol)
        signals = consensus.__dict__ if consensus else {}
        return JSONResponse(content=signals)
    except Exception as e:
        SIG_ENDPOINT_ERRORS.labels(endpoint="/signals/{symbol}").inc()
        logger.error(f"Failed to get symbol signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/generate")
async def generate_signals(request: Dict[str, Any]):
    """Generate signals for given market data."""
    if not signal_generator:
        raise HTTPException(status_code=503, detail="Signal generator not initialized")
    
    try:
        signals = await signal_generator.generate_signals(request)
        # Update metrics
        now_ts = time.time()
        for sym, sig in (signals.items() if isinstance(signals, dict) else []):
            try:
                SIG_GENERATED_TOTAL.labels(symbol=sym).inc()
                SIG_FRESHNESS_SECONDS.labels(symbol=sym).set(0.0)
            except Exception:  # noqa: BLE001
                pass
        SIG_INFERENCE_REQUESTS.labels(status='success').inc()
        return JSONResponse(content=signals)
    except Exception as e:
        SIG_INFERENCE_REQUESTS.labels(status='error').inc()
        SIG_ENDPOINT_ERRORS.labels(endpoint="/signals/generate").inc()
        logger.error(f"Failed to generate signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indicators/{symbol}")
async def get_indicators(symbol: str):
    """Get technical indicators for a symbol."""
    if not signal_generator:
        raise HTTPException(status_code=503, detail="Signal generator not initialized")
    
    try:
        indicators = await signal_generator.get_indicators(symbol)
        return JSONResponse(content=indicators)
    except Exception as e:
        SIG_ENDPOINT_ERRORS.labels(endpoint="/indicators/{symbol}").inc()
        logger.error(f"Failed to get indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/performance")
async def get_signal_performance():
    """Get signal performance metrics."""
    if not signal_generator:
        raise HTTPException(status_code=503, detail="Signal generator not initialized")
    
    try:
        # Get performance metrics - placeholder
        performance = {"status": "not_implemented"}
        return JSONResponse(content=performance)
    except Exception as e:
        SIG_ENDPOINT_ERRORS.labels(endpoint="/signals/performance").inc()
        logger.error(f"Failed to get signal performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/backtest")
async def backtest_signals(request: Dict[str, Any]):
    """Backtest signals on historical data."""
    if not signal_generator:
        raise HTTPException(status_code=503, detail="Signal generator not initialized")
    
    try:
        # Backtest signals - placeholder
        results = {"status": "not_implemented"}
        return JSONResponse(content=results)
    except Exception as e:
        SIG_ENDPOINT_ERRORS.labels(endpoint="/signals/backtest").inc()
        logger.error(f"Failed to backtest signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
#!/usr/bin/env python3
"""Risk Monitoring Service - simplified stable startup."""

import asyncio
import sys
from pathlib import Path
from contextlib import asynccontextmanager, suppress
from typing import Dict, Any
from datetime import datetime
import os
import uuid

sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Request, APIRouter  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
import uvicorn  # noqa: E402
from prometheus_client import Histogram, Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST  # noqa: E402
from trading_common import get_logger, get_settings  # noqa: E402
from trading_common.cache import get_trading_cache  # noqa: E402
from trading_common.database import get_redis_client  # noqa: E402
from observability import install_observability, register_path_template  # noqa: E402
from resilience import AsyncCircuitBreaker, BreakerState, AdaptiveTokenBucket  # noqa: E402

try:
    from risk_monitoring_service import RiskMonitoringService as RiskMonitor, get_risk_service as get_risk_monitor  # noqa: E402
except Exception as e:  # noqa: BLE001
    # Defer raising until first readiness check to keep process alive
    RiskMonitor = None  # type: ignore
    get_risk_monitor = None  # type: ignore
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None

logger = get_logger(__name__)
settings = get_settings()

cache_client = None
redis_client = None
risk_monitor: RiskMonitor | None = None  # type: ignore
last_init_error: str | None = None
_rm_init_attempts = 0  # surfaced via readiness

try:
    SERVICE_READINESS_STATE = Gauge('service_readiness_state', 'Readiness state per service (0=initializing,1=ready,2=degraded)', ['service'])
except Exception:  # noqa: BLE001
    SERVICE_READINESS_STATE = None
if SERVICE_READINESS_STATE:
    with suppress(Exception):
        SERVICE_READINESS_STATE.labels(service='risk-monitor').set(0)

RISK_EVAL_LATENCY = Histogram('risk_evaluation_seconds', 'Risk evaluation latency seconds')
RISK_ALERTS_ACTIVE = Gauge('risk_alerts_active', 'Current active risk alerts')
RISK_SYMBOLS_TRACKED = Gauge('risk_symbols_tracked', 'Number of symbols tracked')
PULSAR_PRODUCER_ERRORS = Counter('pulsar_producer_errors_total', 'Pulsar producer errors', ['service','reason'])
PULSAR_LAST_SUCCESS_TS = Gauge('pulsar_last_success_epoch_seconds', 'Epoch timestamp of last successful Pulsar interaction')
CB_STATE = Gauge('circuit_breaker_state', 'Circuit breaker state (0=closed,1=half_open,2=open)', ['service','breaker'])
CB_TRANSITIONS = Counter('circuit_breaker_transitions_total', 'Circuit breaker transitions', ['service','breaker','from_state','to_state'])

breaker = AsyncCircuitBreaker('pulsar_producer')
token_bucket = AdaptiveTokenBucket(rate=50, capacity=100)  # baseline limit

TOKEN_BUCKET_RATE = Gauge('adaptive_token_bucket_rate', 'Current adaptive token bucket rate', ['service','bucket'])
TOKEN_BUCKET_TOKENS = Gauge('adaptive_token_bucket_tokens', 'Current available tokens in bucket', ['service','bucket'])
with suppress(Exception):
    TOKEN_BUCKET_RATE.labels(service='risk-monitor', bucket='pulsar_producer').set(50)
    TOKEN_BUCKET_TOKENS.labels(service='risk-monitor', bucket='pulsar_producer').set(100)
def _cb_listener(old, new):
    mapping = {BreakerState.CLOSED:0, BreakerState.HALF_OPEN:1, BreakerState.OPEN:2}
    with suppress(Exception):
        CB_STATE.labels(service='risk-monitor', breaker='pulsar_producer').set(mapping[new])
    CB_TRANSITIONS.labels(service='risk-monitor', breaker='pulsar_producer', from_state=old.value, to_state=new.value).inc()
breaker.add_state_listener(_cb_listener)
with suppress(Exception):
    CB_STATE.labels(service='risk-monitor', breaker='pulsar_producer').set(0)

async def _connect_with_retry(name: str, factory, attempts: int = 5, base_delay: float = 0.5):
    """Generic retry connector that supports both sync and coroutine factories.

    The provided factory may be:
      * a regular function returning an object
      * an async function returning an object (needs awaiting)
    """
    for attempt in range(1, attempts + 1):
        try:
            maybe = factory()
            inst = await maybe if asyncio.iscoroutine(maybe) else maybe
            if inst:
                logger.info(f"{name} connection established", attempt=attempt)
                return inst
            else:
                logger.warning(f"{name} attempt {attempt}/{attempts} returned empty/None")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{name} attempt {attempt}/{attempts} failed: {e}")
        await asyncio.sleep(min(base_delay * 2 ** (attempt - 1), 8))
    logger.error(f"{name} unavailable after {attempts} attempts - degraded mode")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle with deferred Risk Monitor initialization.

    Adds bounded initial attempts & background retry to avoid prolonged degraded readiness
    due to Pulsar timing issues (mirrors execution & signal services).
    """
    global cache_client, redis_client, risk_monitor, last_init_error, _rm_init_attempts
    logger.info("Starting Risk Monitoring Service")
    if IMPORT_ERROR:
        last_init_error = f"import_failed: {IMPORT_ERROR}"

    max_initial_attempts = int(os.getenv('RM_INIT_MAX_ATTEMPTS', '3') or '3')
    backoff_base = float(os.getenv('RM_INIT_BACKOFF_BASE', '1.5') or '1.5')
    max_backoff = float(os.getenv('RM_INIT_MAX_BACKOFF', '15') or '15')

    # Metric for attempts
    try:
        from prometheus_client import Counter as _Counter  # noqa: WPS433
        RM_INIT_ATTEMPTS = _Counter('risk_monitor_init_attempts_total', 'Total risk monitor initialization attempts')  # noqa: N806
    except Exception:  # noqa: BLE001
        RM_INIT_ATTEMPTS = None  # type: ignore

    async def _try_init_risk_monitor():
        global risk_monitor, last_init_error, _rm_init_attempts
        _rm_init_attempts += 1
        if RM_INIT_ATTEMPTS:
            with suppress(Exception):
                RM_INIT_ATTEMPTS.inc()
        if get_risk_monitor is None:
            return False
        try:
            rm = risk_monitor or await get_risk_monitor()
            if not rm:
                last_init_error = (last_init_error or "") + " risk_monitor_factory_none"
                return False
            if risk_monitor is None:
                risk_monitor = rm
            await risk_monitor.start()
            logger.info("Risk Monitor initialization succeeded on attempt %d", _rm_init_attempts)
            with suppress(Exception):
                PULSAR_LAST_SUCCESS_TS.set(int(datetime.utcnow().timestamp()))
            with suppress(Exception):
                if breaker.state != BreakerState.CLOSED:
                    await breaker.on_result(True)
            with suppress(Exception):
                token_bucket.record_result(True, 0.05)
                snap = token_bucket.metrics_snapshot()
                TOKEN_BUCKET_RATE.labels(service='risk-monitor', bucket='pulsar_producer').set(snap['rate'])
                TOKEN_BUCKET_TOKENS.labels(service='risk-monitor', bucket='pulsar_producer').set(snap['tokens'])
            last_init_error = None
            return True
        except Exception as e:  # noqa: BLE001
            last_init_error = f"risk_monitor_init_failed: {e}"
            logger.warning("Risk monitor init attempt %d failed: %s", _rm_init_attempts, e)
            if 'Pulsar' in str(e) or 'pulsar' in str(e):
                with suppress(Exception):
                    PULSAR_PRODUCER_ERRORS.labels(service='risk-monitor', reason='startup').inc()
            return False

    async def _background_retry_loop():
        delay = backoff_base
        logger.info("Starting background risk monitor init retry loop")
        while True:
            success = await _try_init_risk_monitor()
            if success:
                logger.info("Background risk monitor initialization completed")
                return
            await asyncio.sleep(delay)
            delay = min(delay * backoff_base, max_backoff)

    cache_client = await _connect_with_retry("cache", get_trading_cache)
    if not cache_client:
        last_init_error = (last_init_error or "") + " cache_unavailable"
    redis_client = await _connect_with_retry("redis", get_redis_client)
    if not redis_client:
        last_init_error = (last_init_error or "") + " redis_unavailable"

    # Initial bounded attempts
    for attempt in range(1, max_initial_attempts + 1):
        success = await _try_init_risk_monitor()
        if success:
            break
        await asyncio.sleep(min(backoff_base * (2 ** (attempt - 1)), max_backoff))

    if not (risk_monitor and getattr(risk_monitor, 'is_running', False)):
        logger.warning(
            "Risk monitor not initialized after %d attempts - scheduling background retries",
            max_initial_attempts
        )
        asyncio.create_task(_background_retry_loop())

    yield
    logger.info("Shutting down Risk Monitoring Service")
    if risk_monitor:
        with suppress(Exception):
            await risk_monitor.stop()
    logger.info("Risk Monitoring Service stopped")


app = FastAPI(title="Risk Monitoring Service", version="1.0.0", lifespan=lifespan)

register_path_template('/risk/*', '/risk/{id}')
try:
    concurrency_limit = int(os.getenv('SERVICE_CONCURRENCY_LIMIT', '0') or 0)
except ValueError:
    concurrency_limit = 0
install_observability(app, 'risk-monitor', concurrency_limit if concurrency_limit > 0 else None)


@app.middleware('http')
async def _corr_and_metrics(request: Request, call_next):
    cid = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
    response = await call_next(request)
    if risk_monitor and hasattr(risk_monitor, 'active_alerts'):
        with suppress(Exception):
            RISK_ALERTS_ACTIVE.set(len(risk_monitor.active_alerts))
    if risk_monitor and hasattr(risk_monitor, 'symbol_metrics'):
        with suppress(Exception):
            RISK_SYMBOLS_TRACKED.set(len(risk_monitor.symbol_metrics))
    response.headers['X-Correlation-ID'] = cid
    return response

metrics_router = APIRouter()
@metrics_router.get('/metrics')
def metrics():  # pragma: no cover
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
app.include_router(metrics_router)


@app.get('/healthz')
async def healthz():
    return {"status": "alive", "service": "risk-monitor", "timestamp": datetime.utcnow().isoformat()}


@app.get('/health')
async def health():
    status = 'healthy' if (risk_monitor and getattr(risk_monitor, 'is_running', False)) else 'unhealthy'
    return {"status": status, "service": "risk_monitor", "timestamp": datetime.utcnow().isoformat()}


@app.get('/ready')
async def ready():
    components = {
        'cache': cache_client is not None,
        'redis': redis_client is not None,
        'risk_monitor': risk_monitor is not None and getattr(risk_monitor, 'is_running', False)
    }
    mapping = {BreakerState.CLOSED:'closed', BreakerState.HALF_OPEN:'half_open', BreakerState.OPEN:'open'}
    breaker_state = mapping.get(breaker.state, 'unknown')
    degraded = [k + '_unavailable' for k, v in components.items() if not v]
    ok = not degraded
    if SERVICE_READINESS_STATE:
        with suppress(Exception):
            SERVICE_READINESS_STATE.labels(service='risk-monitor').set(1 if ok else 2)
    attempts = None
    try:
        attempts = _rm_init_attempts
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse(status_code=200 if ok else 503, content={
        'service': 'risk-monitor',
        'status': 'ready' if ok else 'degraded',
        'components': components,
        'breaker': {'pulsar_producer': breaker_state},
        'adaptive_bucket': token_bucket.metrics_snapshot(),
        'degraded_reasons': degraded or None,
        'last_init_error': last_init_error,
        'init_attempts': attempts,
        'timestamp': datetime.utcnow().isoformat()
    })


@app.get('/risk/alerts')
async def risk_alerts():
    if not risk_monitor:
        raise HTTPException(status_code=503, detail='Risk monitor not initialized')
    alerts = []
    if hasattr(risk_monitor, 'active_alerts'):
        for _aid, alert in risk_monitor.active_alerts.items():  # type: ignore
            alerts.append({
                'alert_id': alert.alert_id,
                'symbol': alert.symbol,
                'alert_type': alert.alert_type.value,
                'risk_level': alert.risk_level.value,
                'timestamp': alert.timestamp.isoformat(),
                'title': alert.title,
                'description': alert.description,
                'severity_score': getattr(alert, 'severity_score', None)
            })
    return {'alerts': alerts, 'count': len(alerts)}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8005, log_level='info')
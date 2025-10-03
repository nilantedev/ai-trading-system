#!/usr/bin/env python3
"""
AI Trading System - Order Execution Service
Handles order management, routing, and execution.
"""

import asyncio
import sys
from pathlib import Path
import contextlib

# Add shared library to path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, APIRouter
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import time
import uuid

from trading_common import get_logger, get_settings
from trading_common.cache import get_trading_cache
from trading_common.database import get_redis_client
from trading_common.resilience import get_circuit_breaker, CircuitBreakerConfig

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from observability import install_observability, register_path_template

# Import order management system
sys.path.insert(0, str(Path(__file__).parent))
try:
    from order_management_system import OrderManagementSystem, get_order_management_system
except ImportError as e:
    logger = get_logger(__name__)
    logger.error(f"Failed to import order_management_system: {e}")
    raise

logger = get_logger(__name__)
settings = get_settings()

# Global state
cache_client = None
redis_client = None
oms = None
oms_init_attempts_reported = 0  # surfaced in readiness
last_init_error: str | None = None

# Readiness state gauge (0=initializing,1=ready,2=degraded)
try:
    SERVICE_READINESS_STATE = Gauge('service_readiness_state', 'Readiness state per service (0=initializing,1=ready,2=degraded)', ['service'])
except Exception:  # noqa: BLE001
    SERVICE_READINESS_STATE = None
if SERVICE_READINESS_STATE:
    try:
        SERVICE_READINESS_STATE.labels(service='execution').set(0)
    except Exception:
        pass

# Legacy per-request metrics removed in favor of shared observability (app_* metrics)
# Domain-specific metrics retained below.
ORDERS_SUBMITTED = Counter('orders_submitted_total', 'Total orders submitted')
ORDERS_REJECTED = Counter('orders_rejected_total', 'Total orders rejected')
OPEN_ORDERS_GAUGE = Gauge('open_orders', 'Current open (non-final) orders')
CB_STATE_GAUGE = Gauge('execution_circuit_breaker_state', 'Circuit breaker state (0=closed,1=half-open,2=open)', ['name'])

# Circuit breaker for OMS submission
oms_cb = get_circuit_breaker('oms_submit', CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30,
    success_threshold=2
))

def _normalize_path(path: str) -> str:  # retained for any local usage (e.g., future domain metrics)
    if path.startswith('/orders/') and len(path.split('/')) == 3:
        return '/orders/{id}'
    return path

def _update_cb_metric():
    state_map = {'closed':0,'half_open':1,'open':2}
    try:
        state = oms_cb.get_state()['state']
        CB_STATE_GAUGE.labels(name='oms_submit').set(state_map.get(state,0))
    except Exception:  # noqa: BLE001
        pass


async def _connect_with_retry(name: str, factory, attempts: int = 5, base_delay: float = 0.5):
    """Resilient connection establishment with exponential backoff."""
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
    """Application lifecycle management with resilience & deferred OMS retry.

    Changes (resilience enhancement):
    - Adds bounded retry attempts for initial OMS start (handles Pulsar not yet ready).
    - If initial bounded retries fail, schedules a background retry loop that keeps
      attempting initialization without blocking service liveness; readiness stays degraded
      until success.
    - Exposes attempt count via metrics (if available) and clears last_init_error on success.
    """
    global cache_client, redis_client, oms, last_init_error

    logger.info("Starting Order Execution Service")
    oms_init_attempts = 0
    max_initial_attempts = int(os.getenv('OMS_INIT_MAX_ATTEMPTS', '3') or '3')
    backoff_base = float(os.getenv('OMS_INIT_BACKOFF_BASE', '1.5') or '1.5')
    max_backoff = float(os.getenv('OMS_INIT_MAX_BACKOFF', '15') or '15')

    # Metric for attempts (optional)
    try:
        from prometheus_client import Counter as _Counter
        OMS_INIT_ATTEMPTS = _Counter('oms_init_attempts_total', 'Total OMS initialization attempts')  # noqa: N806
    except Exception:  # noqa: BLE001
        OMS_INIT_ATTEMPTS = None  # type: ignore

    async def _try_init_oms():
        nonlocal oms_init_attempts
        global last_init_error, oms, oms_init_attempts_reported
        try:
            oms_init_attempts += 1
            if OMS_INIT_ATTEMPTS:
                try:
                    OMS_INIT_ATTEMPTS.inc()
                except Exception:  # noqa: BLE001
                    pass
            if oms is None:
                oms_local = await get_order_management_system()
            else:
                oms_local = oms
            await oms_local.start()
            oms = oms_local
            last_init_error = None
            oms_init_attempts_reported = oms_init_attempts
            logger.info("OMS initialization succeeded on attempt %d", oms_init_attempts)
            return True
        except Exception as e:  # noqa: BLE001
            last_init_error = f"oms_init_failed: {e}"
            logger.warning("OMS init attempt %d failed: %s", oms_init_attempts, e)
            oms_init_attempts_reported = oms_init_attempts
            return False

    async def _background_retry_loop():
        delay = backoff_base
        logger.info("Starting background OMS init retry loop")
        while True:
            success = await _try_init_oms()
            if success:
                logger.info("Background OMS initialization completed")
                return
            await asyncio.sleep(delay)
            delay = min(delay * backoff_base, max_backoff)

    try:
        # Core dependencies first
        cache_client = await _connect_with_retry("cache", get_trading_cache)
        if not cache_client:
            last_init_error = (last_init_error or "") + " cache_unavailable"
        redis_client = await _connect_with_retry("redis", get_redis_client)
        if not redis_client:
            last_init_error = (last_init_error or "") + " redis_unavailable"

        # Attempt bounded initial OMS starts
        for attempt in range(1, max_initial_attempts + 1):
            success = await _try_init_oms()
            if success:
                break
            # Exponential backoff for initial attempts
            await asyncio.sleep(min(backoff_base * (2 ** (attempt - 1)), max_backoff))

        if not (oms and getattr(oms, 'is_running', False)):
            # Schedule background retry without blocking startup
            logger.warning("OMS not initialized after %d attempts - scheduling background retries", max_initial_attempts)
            asyncio.create_task(_background_retry_loop())
        else:
            logger.info("OMS ready at startup")

    except Exception as e:  # noqa: BLE001
        last_init_error = f"unexpected_startup_error: {e}"
        logger.error(f"Unexpected startup error: {e}")

    yield

    # Cleanup
    logger.info("Shutting down Order Execution Service")
    try:
        if oms:
            await oms.stop()
    except Exception:  # noqa: BLE001
        pass
    logger.info("Order Execution Service stopped")


app = FastAPI(
    title="Order Execution Service",
    description="Handles order management, routing, and execution",
    version="1.0.0",
    lifespan=lifespan
)

# Install shared observability middleware & register path templates
register_path_template('/orders/*', '/orders/{id}')
try:
    concurrency_limit = int(os.getenv('SERVICE_CONCURRENCY_LIMIT', '0') or 0)
except ValueError:
    concurrency_limit = 0
install_observability(app, 'execution', concurrency_limit if concurrency_limit > 0 else None)

def _correlation_id_mw(request: Request, call_next):
    async def _inner():
        cid = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
        response = await call_next(request)
        response.headers['X-Correlation-ID'] = cid
        return response
    return _inner()
app.middleware('http')(_correlation_id_mw)

metrics_router = APIRouter()
@metrics_router.get('/metrics')
def metrics():  # pragma: no cover
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
app.include_router(metrics_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        if oms:
            return await oms.get_service_health()
    except AttributeError:
        # Fallback if get_service_health doesn't exist
        pass
    
    return {
        "status": "healthy" if oms and oms.is_running else "unhealthy",
        "service": "order_execution",
        "timestamp": datetime.utcnow().isoformat(),
        "reason": None if (oms and oms.is_running) else "Service not initialized"
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "execution", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
async def readiness():
    """Readiness endpoint: requires cache, redis and OMS running."""
    components = {
        "cache": cache_client is not None,
        "redis": redis_client is not None,
        "oms": oms is not None and getattr(oms, "is_running", False)
    }
    degraded_reasons = []
    if not components['cache']:
        degraded_reasons.append('cache_unavailable')
    if not components['redis']:
        degraded_reasons.append('redis_unavailable')
    if not components['oms']:
        degraded_reasons.append('oms_uninitialized')
    ready = not degraded_reasons
    status_code = 200 if ready else 503
    if SERVICE_READINESS_STATE:
        try:
            SERVICE_READINESS_STATE.labels(service='execution').set(1 if ready else 2)
        except Exception:  # noqa: BLE001
            pass
    global oms_init_attempts_reported
    # Attempt count may be updated by lifespan retry logic; we read metric if available
    try:
        attempts = oms_init_attempts_reported
    except Exception:  # noqa: BLE001
        attempts = None
    return JSONResponse(status_code=status_code, content={
        "service": "execution",
        "status": "ready" if ready else "degraded",
        "components": components,
        "degraded_reasons": degraded_reasons or None,
        "last_init_error": last_init_error,
        "oms_init_attempts": attempts,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.on_event('startup')
async def _warm_readiness():  # pragma: no cover
    await asyncio.sleep(1.5)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get('http://localhost:8004/ready')
    except Exception:
        pass


@app.get("/status")
async def get_status():
    """Get service status."""
    return {
        "service": "order_execution",
        "status": "running" if oms and oms.is_running else "stopped",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "connections": {
            "cache": cache_client is not None,
            "redis": redis_client is not None,
            "oms": oms is not None
        }
    }


@app.post("/orders/submit")
async def submit_order(order_request: Dict[str, Any]):
    """Submit a new order."""
    if not oms:
        raise HTTPException(status_code=503, detail="Order management system not initialized")
    
    try:
        # Process order submission - OMS doesn't have submit_order, it processes through queues
        # Add to order request queue
        from order_management_system import OrderRequest, OrderSide, OrderType, TimeInForce
        import uuid
        
        order_req = OrderRequest(
            request_id=order_request.get('request_id', str(uuid.uuid4())),
            symbol=order_request['symbol'],
            side=OrderSide(order_request.get('side', 'buy')),
            order_type=OrderType(order_request.get('order_type', 'market')),
            quantity=float(order_request['quantity']),
            price=float(order_request['price']) if order_request.get('price') else None,
            source='api'
        )
        
        async def _submit():
            await oms.order_request_queue.put(order_req)
            oms.order_requests[order_req.request_id] = order_req
        try:
            await oms_cb.call(_submit)
        except Exception as e:  # noqa: BLE001
            ORDERS_REJECTED.inc()
            _update_cb_metric()
            logger.error("Order submission failed", extra={"cid": order_req.request_id, "error": str(e)})
            raise HTTPException(status_code=503, detail="Order submission temporarily unavailable") from e
        ORDERS_SUBMITTED.inc()
        # Update open orders gauge
        try:
            open_count = sum(1 for o in oms.order_requests.values() if getattr(o,'status',None) not in ('filled','cancelled','rejected'))
            OPEN_ORDERS_GAUGE.set(open_count)
        except Exception:  # noqa: BLE001
            pass
        _update_cb_metric()
        
        return JSONResponse(content={
            "status": "submitted",
            "request_id": order_req.request_id,
            "symbol": order_req.symbol,
            "quantity": order_req.quantity
        })
    except Exception as e:
        logger.error(f"Failed to submit order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orders")
async def get_orders(symbol: Optional[str] = None, status: Optional[str] = None):
    """Get orders with optional filters."""
    if not oms:
        raise HTTPException(status_code=503, detail="Order management system not initialized")
    
    try:
        from order_management_system import OrderStatus
        status_enum = OrderStatus(status) if status else None
        orders = await oms.get_orders(symbol=symbol, status=status_enum)
        # Convert Order objects to dicts
        orders_dict = []
        for order in orders:
            order_dict = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'created_at': order.created_at.isoformat() if order.created_at else None
            }
            orders_dict.append(order_dict)
        return JSONResponse(content=orders_dict)
    except Exception as e:
        logger.error(f"Failed to get orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orders/active")
async def get_active_orders():
    """Get all active orders."""
    if not oms:
        raise HTTPException(status_code=503, detail="Order management system not initialized")
    
    try:
        orders = await oms.get_active_orders()
        # Convert Order objects to dicts
        orders_dict = []
        for order in orders:
            order_dict = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'created_at': order.created_at.isoformat() if order.created_at else None
            }
            orders_dict.append(order_dict)
        return JSONResponse(content=orders_dict)
    except Exception as e:
        logger.error(f"Failed to get active orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orders/{order_id}/cancel")
async def cancel_order(order_id: str):
    """Cancel an order."""
    if not oms:
        raise HTTPException(status_code=503, detail="Order management system not initialized")
    
    try:
        result = await oms.cancel_order(order_id)
        if result:
            return JSONResponse(content={"status": "success", "order_id": order_id})
        else:
            raise HTTPException(status_code=404, detail="Order not found or cannot be cancelled")
    except Exception as e:
        logger.error(f"Failed to cancel order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info"
    )
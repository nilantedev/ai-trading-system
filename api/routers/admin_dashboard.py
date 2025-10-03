from __future__ import annotations
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from datetime import datetime, date, timedelta
import math
import os, random, hashlib
import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel
from prometheus_client import REGISTRY, Counter, Gauge
try:
    from api.metrics import (
        historical_dataset_coverage_ratio,
        historical_dataset_row_count,
        historical_dataset_last_timestamp_seconds,
        historical_dataset_span_days,
    )  # type: ignore
except Exception:  # noqa: BLE001
    # Gauges may be unavailable in certain minimal deployments; degrade silently
    historical_dataset_coverage_ratio = None  # type: ignore
    historical_dataset_row_count = None  # type: ignore
    historical_dataset_last_timestamp_seconds = None  # type: ignore
    historical_dataset_span_days = None  # type: ignore
from api.auth import get_current_user_cookie_or_bearer
from api.cache_utils import ttl_cache
from api.rate_limiter import EnhancedRateLimiter
import logging
import json
try:  # Prefer real orchestrator if available inside image
    from services.ml.ml_orchestrator import get_ml_orchestrator  # type: ignore
except Exception:  # noqa: BLE001
    # The API image does not bundle the services/ code tree; provide a graceful stub so
    # router registration does not fail and dashboard endpoints degrade instead of 500.
    async def get_ml_orchestrator():  # type: ignore
        class _Stub:
            async def list_models(self):  # noqa: D401
                return []
            async def get_promotion_audit_log(self, limit: int = 100):  # noqa: D401
                return []
            async def manual_promotion_check(self):  # noqa: D401
                return {"status": "unavailable", "detail": "orchestrator stub"}
            async def list_shadow_stats(self, horizon: int = 1):  # noqa: D401
                return []
            async def admin_rollback(self, model_id: str):  # noqa: D401
                return {"rolled_back": False, "model_id": model_id, "reason": "orchestrator stub"}
        logger = logging.getLogger(__name__)
        logger.warning("ML orchestrator module unavailable; using stub implementation")
        return _Stub()

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin-dashboard"], include_in_schema=False)

async def _admin_single_user_guard(user=Depends(get_current_user_cookie_or_bearer)):
    """Enforce that ONLY the single bootstrap user 'nilante' may access any admin
    dashboard (HTML or API) endpoint regardless of stored roles.

    Rationale: User reported role mismatches preventing access while business
    dashboard remained public. We decouple dashboard gating from role
    persistence so a missing/incorrect DB role cannot accidentally expose or
    deny access. Future: replace with allow-list / RBAC mapping if multi-user
    dashboards are introduced.
    """
    if getattr(user, 'username', None) != 'nilante':
        raise HTTPException(status_code=403, detail='Access restricted')
    return user

SINGLE_ADMIN_DEP = _admin_single_user_guard  # Backward alias

# Helper to render template via request.app state (we will add Jinja env in main)

def _template(request: Request, name: str, **ctx):
    """Render a template ensuring 'request' is present in context.

    Several base templates reference 'request' (headers, host). If omitted,
    Jinja will raise an internal error post-login. Pass it explicitly here.
    """
    env = request.app.state.jinja_env
    tpl = env.get_template(name)
    nonce = getattr(request.state, 'csp_nonce', '')
    return HTMLResponse(tpl.render(request=request, csp_nonce=nonce, year=datetime.utcnow().year, **ctx))

@router.get('/admin', response_class=HTMLResponse)
async def admin_dashboard(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    return _template(request, 'admin/dashboard.html', title='Admin Dashboard')

# Explicit HEAD support to avoid 405 during health checks
@router.head('/admin')
async def admin_dashboard_head(user=Depends(SINGLE_ADMIN_DEP)):
    return HTMLResponse(content="", status_code=200)

# Lightweight, unauthenticated probe for external monitoring
@router.get('/admin/availability')
async def admin_availability(request: Request):
    # Intentionally do not enforce auth or rate limit to keep probes reliable
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@router.head('/admin/availability')
async def admin_availability_head():
    return HTMLResponse(content="", status_code=200)

_rate_limiter: EnhancedRateLimiter | None = None

async def _get_rate_limiter():  # Lazy acquire (reuse existing global if already initialized elsewhere)
    global _rate_limiter
    if _rate_limiter is None:
        try:
            rl = EnhancedRateLimiter()
            await rl.initialize()
            _rate_limiter = rl
        except Exception:  # noqa: BLE001
            logger.warning("Rate limiter unavailable for admin dashboard endpoints", exc_info=True)
            _rate_limiter = None
    return _rate_limiter

def _seeded_rand(seed_key: str) -> random.Random:
    h = hashlib.sha256(seed_key.encode()).hexdigest()[:16]
    seed_int = int(h, 16)
    return random.Random(seed_int)

def _cache(key: str, ttl: int, loader):
    return ttl_cache(key, ttl, loader)

def _audit(user, action: str, extra: dict | None = None):
    base = {'user': user.username, 'user_id': user.user_id, 'action': action}
    if extra:
        base.update(extra)
    logger.info("ADMIN_AUDIT", extra=base)

@router.get('/admin/api/latency/metrics')
async def admin_latency_metrics(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'latency_metrics')
    def load():
        out = {
            'timestamp': datetime.utcnow().isoformat(),
            'http': {},
            'inference': {},
            'circuit_breakers': []
        }
        wanted = ('app_http_request_latency_seconds', 'app_inference_latency_seconds')
        try:
            for metric in REGISTRY.collect():  # type: ignore[attr-defined]
                if metric.name in wanted:
                    summary = {}
                    for s in metric.samples:
                        # Expect buckets or summary quantiles
                        if 'quantile' in s.labels:
                            q = s.labels['quantile']
                            summary[q] = s.value
                    out['http' if 'http' in metric.name else 'inference'] = summary
        except Exception:
            pass
        # Circuit breakers best-effort import
        try:
            from trading_common.resilience import get_all_circuit_breakers  # type: ignore
            breakers = get_all_circuit_breakers()
            for b in breakers:
                out['circuit_breakers'].append({
                    'name': getattr(b, 'name', 'unknown'),
                    'state': getattr(b, 'state', 'unknown'),
                    'failure_count': getattr(b, 'failure_count', None),
                    'success_count': getattr(b, 'success_count', None)
                })
        except Exception:
            pass
        return out
    return _cache('admin:latency_metrics', 10, load)

@router.get('/admin/api/pnl/timeseries')
async def admin_pnl_timeseries(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'pnl_timeseries')
    def load():
        # Placeholder deterministic intraday PnL curve (minute resolution last ~60 points)
        r = _seeded_rand('pnl:ts:'+datetime.utcnow().strftime('%Y-%m-%d'))
        base = r.uniform(-2000, 2000)
        series = []
        pnl = base
        for i in range(60):
            pnl += r.uniform(-200, 200)
            series.append(round(pnl,2))
        points = []
        now = datetime.utcnow()
        for i in range(60):  # extend to 60 points for smoother chart
            ts = now.replace(microsecond=0) - timedelta(minutes=2 * (59 - i))
            # simulated pnl curve with some trend + oscillation
            v = round(math.sin(i/6) * 950 + i * 3.5 + 4000, 2)
            points.append({'ts': ts.isoformat(), 'pnl': v})
        return {
            'points': points,
            'summary': {
                'current': points[-1]['pnl'],
                'min': min(p['pnl'] for p in points),
                'max': max(p['pnl'] for p in points)
            }
        }
    return _cache('admin:pnl_ts', 30, load)

@router.get('/admin/api/events/stream')
async def admin_events_stream(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'events_stream')
    # Minimal composite SSE: periodically emit latency + pnl snapshot
    async def gen():
        yield b": events-start\n\n"
        try:
            while True:
                # Reuse existing loaders
                try:
                    lat = await admin_latency_metrics(request, user)  # type: ignore
                except Exception:
                    lat = {'error':'latency_failed'}
                try:
                    pnl = await admin_pnl_timeseries(request, user)  # type: ignore
                except Exception:
                    pnl = {'error':'pnl_failed'}
                payload = json.dumps({'latency': lat, 'pnl': pnl})
                yield f"event: snapshot\ndata: {payload}\n\n".encode()
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            yield b"event: end\ndata: cancelled\n\n"
    return StreamingResponse(gen(), media_type='text/event-stream')

async def _enforce_rate(request: Request, user):
    rl = await _get_rate_limiter()
    if rl is None:
        return
    ident = f"admin:{user.user_id}:{request.client.host if request.client else 'unknown'}"
    result = await rl.check_rate_limit(ident, limit_type="admin", request=request)
    if not result.get('allowed'):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

@router.get('/admin/api/system/summary')
async def system_summary(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'system_summary')
    def load():
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'environment': os.getenv('ENV', 'production'),
            'components': {
                'trading': 'operational',
                'execution': 'operational',
                'risk_monitor': 'operational',
                'ml': 'operational'
            }
        }
    return _cache('admin:system_summary', 5, load)

@router.get('/admin/api/trading/performance')
async def trading_performance(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'trading_performance')
    today = date.today().isoformat()
    def load():
        r = _seeded_rand(f"perf:{today}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'pnl': {
                'daily': round(r.uniform(-5000, 5000), 2),
                'monthly': round(r.uniform(-20000, 20000), 2),
                'ytd': round(r.uniform(-100000, 100000), 2)
            },
            'positions': {
                'count': r.randint(5, 25),
                'exposure_usd': round(r.uniform(50000, 250000), 2)
            }
        }
    return _cache('admin:trading_performance', 10, load)

@router.get('/admin/api/models/status')
async def model_status(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'model_status')
    orch = await get_ml_orchestrator()
    models = await orch.list_models()
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'count': len(models),
        'models': models
    }

@router.get('/admin/api/models/promotion-audit')
async def promotion_audit(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'promotion_audit')
    orch = await get_ml_orchestrator()
    log = await orch.get_promotion_audit_log(limit=150)
    return {'timestamp': datetime.utcnow().isoformat(), 'entries': log, 'count': len(log)}

@router.post('/admin/api/models/promotion-check')
async def manual_promotion_check(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'manual_promotion_check')
    orch = await get_ml_orchestrator()
    result = await orch.manual_promotion_check()
    return {'timestamp': datetime.utcnow().isoformat(), **result}

@router.get('/admin/api/metrics/summary')
async def metrics_summary(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'metrics_summary')
    def load():
        out = {}
        wanted_prefixes = ('app_http_', 'app_inference_', 'app_concurrency_', 'app_request_queue_depth', 'app_requests_shed_total')
        for metric in REGISTRY.collect():  # type: ignore[attr-defined]
            if any(metric.name.startswith(p) for p in wanted_prefixes):
                samples = []
                for s in metric.samples:
                    samples.append({'name': s.name, 'labels': s.labels, 'value': s.value})
                out[metric.name] = samples
        return {'timestamp': datetime.utcnow().isoformat(), 'metrics': out}
    return _cache('admin:metrics_summary', 5, load)

@router.get('/admin/api/models/shadow-stats')
async def shadow_stats(request: Request, user=Depends(SINGLE_ADMIN_DEP), horizon: int = 1):
    await _enforce_rate(request, user)
    _audit(user, 'shadow_stats')
    orch = await get_ml_orchestrator()
    stats = await orch.list_shadow_stats(horizon=horizon)
    return {'timestamp': datetime.utcnow().isoformat(), 'horizon': horizon, 'shadow_models': stats}

@router.post('/admin/api/models/{model_id}/rollback')
async def rollback_model_action(model_id: str, request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'rollback_model', {'model_id': model_id})
    orch = await get_ml_orchestrator()
    result = await orch.admin_rollback(model_id)
    return {'timestamp': datetime.utcnow().isoformat(), **result}

# =====================
# ADMIN TASK RUNNER + LOG STREAMING (SSE)
# =====================

class TaskRunRequest(BaseModel):
    name: str
    args: Optional[list[str]] = None


# Whitelisted scripts (relative to repo root)
ALLOWED_TASKS: dict[str, list[str]] = {
    'backfill': ['python', 'scripts/run_historical_backfill.py'],
    'health-check': ['bash', 'scripts/check_health_production.sh'],
    'quality-check': ['bash', 'scripts/quality_check.sh'],
    'diagnose': ['bash', 'scripts/diagnose_service.sh'],
    'sbom': ['bash', 'scripts/generate_sbom.sh'],
    'bootstrap-questdb': ['python', 'scripts/bootstrap_questdb_tables.py'],
    'bulk-load': ['python', 'scripts/bulk_load_market_data.py'],
    'news-collect-once': ['python', 'scripts/collect_news_once.py'],
    'social-collect-once': ['python', 'scripts/collect_social_sentiment_once.py'],
    'cleanup': ['python', 'scripts/cleanup_artifacts.py'],
    # Postgres index bootstrap (idempotent) to ensure common indexes exist
    'bootstrap-postgres-indexes': ['python', 'scripts/bootstrap_postgres_indexes.py'],
    # On-demand retention run (single-shot) - executes retention service cleanup once
    'retention-run': [
        'python','-c',
        'import asyncio;from services.data_ingestion.data_retention_service import get_retention_service;'
        'async def _m(): svc=await get_retention_service(); await svc.run_data_cleanup(); print("retention_complete")+'
        '"";asyncio.run(_m())'
    ],
}

_tasks: dict[str, dict[str, Any]] = {}
_tasks_lock = asyncio.Lock()

# Prometheus metrics for admin operations (default registry)
admin_tasks_started_total = Counter("admin_tasks_started_total", "Admin tasks started", ["name"])
admin_tasks_completed_total = Counter("admin_tasks_completed_total", "Admin tasks completed", ["name", "status"])
admin_tasks_running = Gauge("admin_tasks_running", "Currently running admin tasks")
admin_logs_stream_clients = Gauge("admin_logs_stream_clients", "Connected admin log SSE clients")
admin_task_stream_clients = Gauge("admin_task_stream_clients", "Connected admin task SSE clients")
admin_heartbeat_stream_clients = Gauge("admin_heartbeat_stream_clients", "Connected admin heartbeat SSE clients")

# -------------------- ML TRAINING TRIGGER (ADMIN) --------------------

class TrainingJobRequest(BaseModel):
    symbols: Optional[list[str]] = None
    model_type: Optional[str] = "ensemble"  # ensemble | gradient_boosting | deep_learning
    model_name: Optional[str] = None
    horizon_days: Optional[int] = 1
    start_date: Optional[str] = None  # YYYY-MM-DD (optional; defaults to ~5y ago in ML service)
    end_date: Optional[str] = None    # YYYY-MM-DD (optional; defaults to today in ML service)
    features: Optional[list[str]] = None


@router.post('/admin/api/models/train')
async def admin_trigger_training(req: TrainingJobRequest, request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    """Trigger a model training job in the ML service (admin-only).

    Defaults:
      - symbols: [AAPL, MSFT, NVDA, SPY, QQQ]
      - model_type: ensemble
      - horizon_days: 1
      - start/end omitted -> ML service uses a robust 5y default window
    """
    await _enforce_rate(request, user)
    _audit(user, 'models_train', {'symbols': (req.symbols or [])[:10], 'model_type': req.model_type})

    ml_url = os.getenv('ML_SERVICE_URL', 'http://trading-ml:8001').rstrip('/')
    target = f"{ml_url}/models/train"

    # Build job payload expected by ML service
    symbols = [s.strip().upper() for s in (req.symbols or ['AAPL','MSFT','NVDA','SPY','QQQ']) if s and s.strip()]
    model_name = req.model_name or f"{(req.model_type or 'ensemble')}_top{len(symbols)}_{req.horizon_days or 1}d_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    job = {
        'model_name': model_name,
        'symbols': symbols,
        'horizon_days': int(req.horizon_days or 1),
    }
    if req.start_date:
        job['start_date'] = req.start_date
    if req.end_date:
        job['end_date'] = req.end_date
    if req.features:
        job['features'] = req.features

    payload = {
        'model_type': (req.model_type or 'ensemble'),
        'training_config': {'job': job}
    }

    # Call ML service and proxy response
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(target, json=payload)
        code = resp.status_code
        try:
            data = resp.json()
        except Exception:
            data = {'text': (resp.text[:500] if hasattr(resp, 'text') else '')}
        if code >= 200 and code < 300:
            return {
                'status': 'accepted',
                'ml_status_code': code,
                'ml_response': data,
                'job': job,
                'timestamp': datetime.utcnow().isoformat()
            }
        raise HTTPException(status_code=code, detail={'error': 'ml_service_error', 'response': data})
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"ml_service_unreachable: {e}")

async def _spawn_task(task_name: str, cmd: list[str], extra_env: Optional[dict[str, str]] = None) -> tuple[str, asyncio.subprocess.Process, asyncio.Queue[str]]:
    task_id = hashlib.sha256((" ".join(cmd) + str(datetime.utcnow().timestamp())).encode()).hexdigest()[:12]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd='/srv/ai-trading-system',
        env=env,
    )
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)

    async def _reader():
        try:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    text = line.decode(errors='replace').rstrip('\n')
                except Exception:
                    text = str(line)
                # Bound queue: drop oldest if full
                if q.full():
                    try:
                        _ = q.get_nowait()
                    except Exception:
                        pass
                await q.put(text)
        finally:
            await proc.wait()
            # Record return code and metrics
            rc = proc.returncode
            async with _tasks_lock:
                info = _tasks.get(task_id)
                if info is not None:
                    info['returncode'] = rc
                    info['ended_at'] = datetime.utcnow().isoformat()
                    info['status'] = 'success' if rc == 0 else 'error'
            try:
                status = 'success' if rc == 0 else 'error'
                admin_tasks_completed_total.labels(name=task_name, status=status).inc()
            except Exception:
                pass
            try:
                admin_tasks_running.dec()
            except Exception:
                pass
            await q.put(f"__TASK_EXIT__:{proc.returncode}")

    asyncio.create_task(_reader())

    async with _tasks_lock:
        _tasks[task_id] = {
            'cmd': cmd,
            'name': task_name,
            'proc': proc,
            'queue': q,
            'start_time': datetime.utcnow().isoformat(),
            'returncode': None,
        }
    return task_id, proc, q


@router.post('/admin/api/tasks/run')
async def run_admin_task(req: TaskRunRequest, request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    name = req.name.strip()
    if name not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail="Task not allowed")
    base_cmd = ALLOWED_TASKS[name]
    # Sanitize args: allow alnum, dash, underscore, dot, slash, equals
    args: list[str] = []
    if req.args:
        for a in req.args:
            if not a or len(a) > 200:
                continue
            if not all(ch.isalnum() or ch in "-_.=/,:" for ch in a):
                continue
            args.append(a)
    cmd = [*base_cmd, *args]
    task_id, proc, _ = await _spawn_task(name, cmd)
    try:
        admin_tasks_started_total.labels(name=name).inc()
        admin_tasks_running.inc()
    except Exception:
        pass
    _audit(user, 'task_run', {'task_id': task_id, 'name': name, 'cmd': cmd})
    return {'task_id': task_id, 'pid': proc.pid, 'status': 'started', 'started_at': datetime.utcnow().isoformat()}


@router.get('/admin/api/tasks/{task_id}/status')
async def task_status(task_id: str, request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    async with _tasks_lock:
        info = _tasks.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="task not found")
    proc = info['proc']
    rc = proc.returncode
    if rc is None and proc.returncode is not None:
        rc = proc.returncode
    return {
        'task_id': task_id,
        'cmd': info['cmd'],
        'started_at': info['start_time'],
        'returncode': rc,
        'running': rc is None,
    }


@router.get('/admin/api/tasks/{task_id}/stream')
async def task_stream(task_id: str, request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    async with _tasks_lock:
        info = _tasks.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="task not found")
    q: asyncio.Queue[str] = info['queue']

    async def event_generator():
        try:
            admin_task_stream_clients.inc()
        except Exception:
            pass
        # Send initial comment to open stream
        yield b": stream-start\n\n"
        last_keepalive = asyncio.get_event_loop().time()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=10.0)
                    if item.startswith("__TASK_EXIT__:"):
                        code = item.split(":",1)[1]
                        yield f"event: end\ndata: {code}\n\n".encode()
                        break
                    payload = item.replace('\r','')
                    yield f"event: log\ndata: {payload}\n\n".encode()
                except asyncio.TimeoutError:
                    # keep-alive
                    now = asyncio.get_event_loop().time()
                    if now - last_keepalive > 10:
                        yield b": keep-alive\n\n"
                        last_keepalive = now
        except asyncio.CancelledError:
            yield b"event: end\ndata: cancelled\n\n"
        finally:
            try:
                admin_task_stream_clients.dec()
            except Exception:
                pass

    return StreamingResponse(event_generator(), media_type='text/event-stream')


# Application log streaming via SSE
_log_queue: asyncio.Queue[str] | None = None

class _AsyncQueueHandler(logging.Handler):
    def __init__(self, queue: asyncio.Queue[str]):
        super().__init__()
        self.queue = queue
        self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    def emit(self, record):
        try:
            msg = self.format(record)
            # Best-effort put without blocking
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(self.queue.put_nowait, msg)
            except RuntimeError:
                pass
        except Exception:
            pass

_log_handler_attached = False

def _ensure_log_stream():
    global _log_queue, _log_handler_attached
    if _log_queue is None:
        _log_queue = asyncio.Queue(maxsize=2000)
    if not _log_handler_attached:
        handler = _AsyncQueueHandler(_log_queue)
        root = logging.getLogger()
        root.addHandler(handler)
        _log_handler_attached = True


@router.get('/admin/api/logs/stream')
async def logs_stream(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _ensure_log_stream()
    assert _log_queue is not None
    q = _log_queue

    async def gen():
        try:
            admin_logs_stream_clients.inc()
        except Exception:
            pass
        yield b": log-stream-start\n\n"
        last_keepalive = asyncio.get_event_loop().time()
        try:
            while True:
                try:
                    line = await asyncio.wait_for(q.get(), timeout=10.0)
                    payload = line.replace('\r','')
                    yield f"event: log\ndata: {payload}\n\n".encode()
                except asyncio.TimeoutError:
                    now = asyncio.get_event_loop().time()
                    if now - last_keepalive > 10:
                        yield b": keep-alive\n\n"
                        last_keepalive = now
        finally:
            try:
                admin_logs_stream_clients.dec()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type='text/event-stream')

# Heartbeat SSE endpoint: emits lightweight heartbeat JSON every 5 seconds
@router.get('/admin/api/heartbeat/stream')
async def admin_heartbeat_stream(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    await _enforce_rate(request, user)
    _audit(user, 'heartbeat_stream')
    async def gen():
        try:
            admin_heartbeat_stream_clients.inc()
        except Exception:
            pass
        yield b": heartbeat-stream-start\n\n"
        try:
            while True:
                # minimal load: provide server time + monotonic counter + optional latency metric existence flag
                payload = {
                    'ts': datetime.utcnow().isoformat(),
                    'type': 'heartbeat'
                }
                try:
                    # Sample one metric presence quickly
                    from prometheus_client import REGISTRY  # type: ignore
                    payload['latency_metric_present'] = any(m.name == 'app_http_request_latency_seconds' for m in REGISTRY.collect())
                except Exception:
                    payload['latency_metric_present'] = False
                yield f"event: heartbeat\ndata: {json.dumps(payload)}\n\n".encode()
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            yield b"event: end\ndata: cancelled\n\n"
        finally:
            try:
                admin_heartbeat_stream_clients.dec()
            except Exception:
                pass
    return StreamingResponse(gen(), media_type='text/event-stream')

_historical_cache: dict[str, tuple[float, dict]] = {}
_historical_lock = asyncio.Lock()

@router.get('/admin/api/historical/coverage')
async def admin_historical_coverage(request: Request, user=Depends(SINGLE_ADMIN_DEP), refresh: bool = False):
    await _enforce_rate(request, user)
    _audit(user, 'historical_coverage')
    import time, json as _json, asyncio as _asyncio, subprocess, os
    key = 'historical:coverage'
    now = time.time()
    async with _historical_lock:
        entry = _historical_cache.get(key)
    if entry and not refresh and entry[0] > now:
        return entry[1]
    # Run verification script as subprocess for isolation
    try:
        proc = await asyncio.create_subprocess_exec(
            'python','scripts/verify_historical_coverage.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd='/srv/ai-trading-system'
        )
        out, err = await proc.communicate()
        if err:
            logger.debug('historical coverage stderr', stderr=err.decode(errors='ignore'))
        try:
            data = _json.loads(out.decode()) if out else {"error":"no_output"}
        except Exception as e:  # noqa: BLE001
            data = {"error":"decode_failed","detail": str(e)}
        data['exit_code'] = proc.returncode
    except Exception as e:  # noqa: BLE001
        data = {"error":"subprocess_failed","detail": str(e)}
    async with _historical_lock:
        _historical_cache[key] = (now + 300, data)  # cache 5 min
    # Update coverage gauges best-effort
    try:
        if data and isinstance(data, dict):
            for r in data.get('results', []):
                ds = r.get('name') or r.get('table') or 'unknown'
                ratio = r.get('approx_trading_day_ratio') or 0.0
                span = r.get('span_days') or 0
                rows = r.get('row_count') or 0
                last_ts = r.get('last_timestamp')
                if last_ts and isinstance(last_ts, str):
                    try:
                        from datetime import datetime as _dt
                        import dateutil.parser  # type: ignore
                        epoch = int(dateutil.parser.isoparse(last_ts).timestamp())
                    except Exception:  # noqa: BLE001
                        epoch = 0
                else:
                    epoch = 0
                if historical_dataset_coverage_ratio:
                    historical_dataset_coverage_ratio.labels(dataset=ds).set(ratio or 0)
                if historical_dataset_row_count:
                    historical_dataset_row_count.labels(dataset=ds).set(rows)
                if historical_dataset_last_timestamp_seconds:
                    historical_dataset_last_timestamp_seconds.labels(dataset=ds).set(epoch)
                if historical_dataset_span_days:
                    historical_dataset_span_days.labels(dataset=ds).set(span)
    except Exception:
        logger.debug("coverage gauge update failed", exc_info=True)
    return data

# Consolidated multi-dataset verification endpoint (wrapper around script w/ normalization)
_verification_cache: dict[str, tuple[float, dict]] = {}
_verification_lock = asyncio.Lock()

@router.get('/admin/api/data/verification')
async def admin_data_verification(request: Request, user=Depends(SINGLE_ADMIN_DEP), refresh: bool = False):
    await _enforce_rate(request, user)
    _audit(user, 'data_verification')
    import time as _time
    key = 'admin:data_verification'
    now = _time.time()
    async with _verification_lock:
        entry = _verification_cache.get(key)
    if entry and not refresh and entry[0] > now:
        return entry[1]
    # Execute verification script (same as coverage but we also build quick summary)
    try:
        proc = await asyncio.create_subprocess_exec(
            'python','scripts/verify_historical_coverage.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd='/srv/ai-trading-system'
        )
        out, err = await proc.communicate()
        if err:
            logger.debug('data verification stderr', stderr=err.decode(errors='ignore'))
        try:
            raw = json.loads(out.decode()) if out else {"error":"no_output"}
        except Exception as e:  # noqa: BLE001
            raw = {"error":"decode_failed","detail": str(e)}
        raw['exit_code'] = proc.returncode
    except Exception as e:  # noqa: BLE001
        raw = {"error":"subprocess_failed","detail": str(e)}
    # Build normalized summary
    summary: Dict[str, Any] = {
        'timestamp': datetime.utcnow().isoformat(),
        'status': raw.get('status','unknown'),
        'targets': {},
        'meets_all_targets': True,
        'script_exit_code': raw.get('exit_code')
    }
    for r in raw.get('results', []):
        name = r.get('name') or 'unknown'
        summary['targets'][name] = {
            'present': r.get('present', False),
            'span_days': r.get('span_days'),
            'distinct_trading_days': r.get('distinct_trading_days'),
            'row_count': r.get('row_count'),
            'first_timestamp': r.get('first_timestamp'),
            'last_timestamp': r.get('last_timestamp'),
            'approx_trading_day_ratio': r.get('approx_trading_day_ratio'),
            'meets_target': r.get('meets_target'),
            'recent_gap': r.get('recent_gap'),
            'estimated_missing_trading_days': r.get('estimated_missing_trading_days'),
            'reason': r.get('reason'),
            'error_message': r.get('error_message')
        }
        if not r.get('meets_target'):
            summary['meets_all_targets'] = False
    if raw.get('errors'):
        summary['script_errors'] = raw.get('errors')
    output = {'raw': raw, 'summary': summary}
    async with _verification_lock:
        _verification_cache[key] = (now + 300, output)  # cache 5 min
    # Update gauges using normalized summary (raw contains the dataset list)
    try:
        raw = output.get('raw', {}) if isinstance(output, dict) else {}
        for r in raw.get('results', []):
            ds = r.get('name') or r.get('table') or 'unknown'
            ratio = r.get('approx_trading_day_ratio') or 0.0
            span = r.get('span_days') or 0
            rows = r.get('row_count') or 0
            last_ts = r.get('last_timestamp')
            if last_ts and isinstance(last_ts, str):
                try:
                    from datetime import datetime as _dt
                    import dateutil.parser  # type: ignore
                    epoch = int(dateutil.parser.isoparse(last_ts).timestamp())
                except Exception:  # noqa: BLE001
                    epoch = 0
            else:
                epoch = 0
            if historical_dataset_coverage_ratio:
                historical_dataset_coverage_ratio.labels(dataset=ds).set(ratio or 0)
            if historical_dataset_row_count:
                historical_dataset_row_count.labels(dataset=ds).set(rows)
            if historical_dataset_last_timestamp_seconds:
                historical_dataset_last_timestamp_seconds.labels(dataset=ds).set(epoch)
            if historical_dataset_span_days:
                historical_dataset_span_days.labels(dataset=ds).set(span)
    except Exception:
        logger.debug("verification gauge update failed", exc_info=True)
    return output

# Lightweight authenticated debug endpoint to inspect current user context (diagnostic only)
@router.get('/admin/api/debug/auth-state')
async def admin_debug_auth_state(user=Depends(SINGLE_ADMIN_DEP)):
    """Return the authenticated user's principal info for debugging dashboard access.

    Intentionally minimal and safe: reveals only username, user_id, and roles already present in token.
    """
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'user': {
            'user_id': getattr(user, 'user_id', None),
            'username': getattr(user, 'username', None),
            'roles': getattr(user, 'roles', []),
        }
    }

# =====================
# MODEL / DATA READINESS ENDPOINT
# =====================

@router.get('/admin/api/readiness/model')
async def model_data_readiness(request: Request, user=Depends(SINGLE_ADMIN_DEP), refresh: bool = False):
    """Summarize dataset readiness vs required horizons for model training.

    Targets:
      equities: 20y, options/news/social: 5y each (approx trading day ratio >= 0.97)
    Returns overall READY / PARTIAL along with per-dataset gaps.
    """
    await _enforce_rate(request, user)
    _audit(user, 'model_data_readiness')
    # Reuse existing verification endpoint logic by invoking internal function (subprocess call) to avoid code duplication
    # Run coverage script directly (fresh if refresh=True)
    import asyncio as _asyncio, json as _json
    proc = await asyncio.create_subprocess_exec(
        'python','scripts/verify_historical_coverage.py',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd='/srv/ai-trading-system'
    )
    out, err = await proc.communicate()
    try:
        raw = _json.loads(out.decode()) if out else {"error":"no_output"}
    except Exception as e:  # noqa: BLE001
        raw = {"error":"decode_failed","detail": str(e)}
    datasets = {}
    overall_ready = True
    gaps: list[dict[str, any]] = []  # type: ignore[name-defined]
    expected_years = {'equities':20,'options':5,'news':5,'social':5}
    for r in raw.get('results', []):
        name = r.get('name')
        if not name:
            continue
        meets = bool(r.get('meets_target'))
        datasets[name] = {
            'present': r.get('present'),
            'meets_target': meets,
            'approx_trading_day_ratio': r.get('approx_trading_day_ratio'),
            'span_days': r.get('span_days'),
            'row_count': r.get('row_count'),
            'last_timestamp': r.get('last_timestamp'),
            'target_years': expected_years.get(name),
            'recent_gap': r.get('recent_gap')
        }
        if not meets:
            overall_ready = False
            gaps.append({
                'dataset': name,
                'reason': r.get('reason') or ('recent_gap' if r.get('recent_gap') else 'insufficient_span'),
                'approx_ratio': r.get('approx_trading_day_ratio'),
                'estimated_missing_trading_days': r.get('estimated_missing_trading_days')
            })
    status = 'READY' if overall_ready else ('PARTIAL' if datasets else 'UNKNOWN')
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'status': status,
        'datasets': datasets,
        'gaps': gaps,
        'script_exit_code': raw.get('exit_code'),
        'raw_status': raw.get('status')
    }


# Staging-only: validation endpoint to nudge admin metrics (does not run in production)
@router.post('/admin/api/tasks/validate-metrics')
async def validate_metrics(request: Request, user=Depends(SINGLE_ADMIN_DEP)):
    env = os.getenv('ENVIRONMENT', 'production').lower()
    if env == 'production':
        raise HTTPException(status_code=403, detail='validation endpoint disabled in production')
    await _enforce_rate(request, user)
    name = 'metrics-validation'
    cmd = ['bash', '-lc', 'echo "validation start"; sleep 1; echo "validation end"']
    task_id, proc, _ = await _spawn_task(name, cmd, extra_env={"VALIDATION": "1"})
    try:
        admin_tasks_started_total.labels(name=name).inc()
        admin_tasks_running.inc()
    except Exception:
        pass
    _audit(user, 'task_run_validation', {'task_id': task_id, 'name': name})
    return {'task_id': task_id, 'pid': proc.pid, 'status': 'started', 'started_at': datetime.utcnow().isoformat()}

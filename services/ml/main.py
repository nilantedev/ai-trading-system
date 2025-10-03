#!/usr/bin/env python3
"""
AI Trading System - Advanced ML Service
Orchestrates all machine learning models and intelligence systems.
"""

import asyncio
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import time
import uuid
import json
import numpy as np
import pandas as pd
import urllib.parse

# Add service and shared libraries to path (mirrors data-ingestion robustness)
_THIS_DIR = Path(__file__).parent
_SHARED = (_THIS_DIR / "../../shared/python-common").resolve()
for _p in (_THIS_DIR, _SHARED):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from trading_common import get_logger, get_settings, MarketData
from trading_common.cache import get_trading_cache
from trading_common.database import get_redis_client
from trading_common.database_manager import get_database_manager
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

# Import all advanced ML services
from production_llm_service import ProductionLLMService, AnalysisRequest, AnalysisType
from advanced_intelligence_coordinator import AdvancedIntelligenceCoordinator
from ensemble_intelligence_coordinator import EnsembleIntelligenceCoordinator
from market_regime_detector import MarketRegimeDetector
from advanced_factor_models import AdvancedFactorService
from garch_lstm_model import GARCHModel
from graph_neural_network import MarketGraphConstructor as MarketGraphNetwork
from stochastic_volatility_models import StochasticVolatilityService
from transfer_entropy_analysis import TransferEntropyCalculator
from multi_timeframe_intelligence import MultiTimeframeIntelligence
from drift_monitor import DriftMonitor
from continuous_improvement_engine import ContinuousImprovementEngine
from continuous_training_orchestrator import ContinuousTrainingOrchestrator
from performance_analytics_service import PerformanceAnalyticsService
from model_training_pipeline import ModelTrainingPipeline
from intelligent_backtesting_framework import IntelligentBacktestingFramework
from model_router import ModelRouter, TaskType, TaskUrgency
from ollama_service import OllamaService
from finbert_sentiment_analyzer import FinBERTSentimentAnalyzer
from observability import install_observability, timed_inference
from pydantic import BaseModel
from typing import Union
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore
try:
    # Optional Weaviate schema utilities
    from shared.vector.weaviate_schema import (
        get_weaviate_client,
        desired_schema,
        fetch_current_schema,
        diff_schema,
        apply_schema_changes,
        WeaviateSchemaError,
    )
except Exception:  # pragma: no cover - optional import
    get_weaviate_client = None  # type: ignore
    desired_schema = None  # type: ignore
    fetch_current_schema = None  # type: ignore
    diff_schema = None  # type: ignore
    apply_schema_changes = None  # type: ignore
    WeaviateSchemaError = Exception  # type: ignore

logger = get_logger(__name__)
settings = get_settings()

# Global state
cache_client = None
redis_client = None
llm_service = None
intelligence_coordinator = None
ensemble_coordinator = None
regime_detector = None
factor_model = None
volatility_model = None
graph_network = None
stochastic_vol = None
entropy_analyzer = None
timeframe_intelligence = None
drift_monitor = None
improvement_engine = None
training_orchestrator = None

analytics_service = None
training_pipeline = None
backtesting_framework = None
model_router = None
ollama_service = None
finbert_analyzer = None
_news_embedder = None
_weav_client_cached = None
_NIGHT_BACKFILL_DONE_DATE: Optional[str] = None

# ------------------------------ ML Exported Metrics (Grafana) ------------------------------
try:
    # Intelligence job last-run stats
    ML_INTEL_LAST_RUN_TS = Gauge('ml_intelligence_last_run_timestamp_seconds', 'Unix timestamp of last completed intelligence job', ['job'])
    ML_INTEL_LAST_DURATION = Gauge('ml_intelligence_last_duration_seconds', 'Duration of last intelligence job in seconds', ['job'])
    ML_INTEL_LAST_OK = Gauge('ml_intelligence_last_ok', '1 if last intelligence job succeeded, else 0', ['job'])
    # Per-symbol risk latest snapshot
    ML_RISK_LAST_SHARPE = Gauge('ml_risk_last_sharpe', 'Latest computed Sharpe ratio per symbol', ['symbol'])
    ML_RISK_LAST_MDD = Gauge('ml_risk_last_max_drawdown', 'Latest computed max drawdown per symbol', ['symbol'])
    # Market-hours indicator for alert gating
    ML_MARKET_HOURS = Gauge('ml_market_hours', '1 during configured market hours, else 0')
except Exception:
    ML_INTEL_LAST_RUN_TS = None  # type: ignore
    ML_INTEL_LAST_DURATION = None  # type: ignore
    ML_INTEL_LAST_OK = None  # type: ignore
    ML_RISK_LAST_SHARPE = None  # type: ignore
    ML_RISK_LAST_MDD = None  # type: ignore
    ML_MARKET_HOURS = None  # type: ignore

# ------------------------------ Scheduler Config ------------------------------
ENABLE_ML_SCHEDULER = os.getenv('ENABLE_ML_SCHEDULER', 'false').lower() in ('1','true','yes','on')
SCHED_WATCHLIST = [s.strip().upper() for s in os.getenv('ML_SCHEDULER_WATCHLIST', 'AAPL,MSFT,SPY,TSLA,NVDA,AMZN,QQQ').split(',') if s.strip()]
CORRELATION_INTERVAL = int(os.getenv('ML_SCHEDULER_CORRELATION_INTERVAL_SECONDS', '1800'))  # 30m
FACTORS_INTERVAL = int(os.getenv('ML_SCHEDULER_FACTORS_INTERVAL_SECONDS', '1800'))          # 30m
RISK_INTERVAL = int(os.getenv('ML_SCHEDULER_RISK_INTERVAL_SECONDS', '900'))                 # 15m
REDIS_PREFIX = os.getenv('ML_SCHEDULER_REDIS_PREFIX', 'ml:sched')
DEFAULT_TIMEFRAME = os.getenv('ML_SCHEDULER_TIMEFRAME', '1d')
# Dynamic watchlist settings (avoid hard-coded symbols)
WATCHLIST_SOURCE = os.getenv('ML_SCHEDULER_WATCHLIST_SOURCE', 'env').lower()  # env | questdb (default 'env')
WATCHLIST_LOOKBACK_DAYS = int(os.getenv('ML_SCHEDULER_DYNAMIC_LOOKBACK_DAYS', '180'))
WATCHLIST_MAX_SYMBOLS = int(os.getenv('ML_SCHEDULER_MAX_SYMBOLS', '200'))
_SCHED_TASK = None  # set at runtime
# --------- Day/Night Ollama Warm Scheduler Config ---------
MARKET_TZ = os.getenv('MARKET_HOURS_TZ', os.getenv('TRADING_HOURS_TZ', 'America/New_York'))
MARKET_OPEN = os.getenv('MARKET_HOURS_OPEN', os.getenv('TRADING_HOURS_OPEN', '09:30'))
MARKET_CLOSE = os.getenv('MARKET_HOURS_CLOSE', os.getenv('TRADING_HOURS_CLOSE', '16:00'))
DAY_HOT_MODELS = [s.strip() for s in os.getenv('ML_DAY_HOT_MODELS', 'solar:10.7b,phi3:14b,yi:34b').split(',') if s.strip()]
NIGHT_HEAVY_MODELS = [s.strip() for s in os.getenv('ML_NIGHT_HEAVY_MODELS', 'mixtral:8x22b,qwen2.5:72b,command-r-plus:104b,llama3.1:70b').split(',') if s.strip()]
ENABLE_WARM_SCHEDULER = os.getenv('ENABLE_WARM_SCHEDULER', 'true').lower() in ('1','true','yes','on')
# Replace DeepSeek toggle with Llama 3.1 70B night toggle
INCLUDE_LLAMA_NIGHT = os.getenv('ML_NIGHT_INCLUDE_LLAMA', 'true').lower() in ('1','true','yes','on')
LLAMA_NIGHT_MODEL = os.getenv('ML_LLAMA_NIGHT_MODEL', 'llama3.1:70b').strip()
LLAMA_NIGHT_DAYS = os.getenv('ML_LLAMA_NIGHT_DAYS', '').strip()  # e.g., "Sun,Tue,Thu"

def _is_llama_night_allowed() -> bool:
    if not INCLUDE_LLAMA_NIGHT:
        return False
    if not LLAMA_NIGHT_DAYS:
        return True
    try:
        days = [d.strip().lower() for d in LLAMA_NIGHT_DAYS.split(',') if d.strip()]
        names = ['mon','tue','wed','thu','fri','sat','sun']
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(MARKET_TZ)
        today = names[datetime.now(tz).weekday()]
        return today in days
    except Exception:
        return True

def _parse_hhmm_to_ints(val: str) -> tuple[int,int]:
    try:
        hh, mm = val.strip().split(':', 1)
        return int(hh), int(mm)
    except Exception:
        return 0, 0

def _is_market_hours() -> bool:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(MARKET_TZ)
        now = datetime.now(tz)
        if now.weekday() >= 5:
            return False
        oh, om = _parse_hhmm_to_ints(MARKET_OPEN)
        ch, cm = _parse_hhmm_to_ints(MARKET_CLOSE)
        start = now.replace(hour=oh, minute=om, second=0, microsecond=0)
        end = now.replace(hour=ch, minute=cm, second=0, microsecond=0)
        return start <= now <= end
    except Exception:
        return False
_SCHED_LAST_KEYS = {
    'correlation': 'last:correlation',
    'factors': 'last:factors',
    'risk': 'last:risk',
}

async def _warmup_ollama_models(router: ModelRouter, models: list[str]) -> dict:
    """Warm up selected Ollama models by issuing a tiny prompt to load them into memory.
    Does not attempt to pull models; only loads if already present.
    Returns a dict of model -> {loaded:bool, latency_ms:int|None, error:str|None}.
    """
    results: dict[str, dict] = {}
    if not router:
        return results
    tiny_prompt = os.getenv('OLLAMA_WARMUP_PROMPT', 'ready?')
    for name in models:
        try:
            # Skip if model marked unavailable
            info = router.models.get(name)
            if info is not None and not info.is_available:
                results[name] = {"loaded": False, "latency_ms": None, "error": "unavailable"}
                continue
            r = await router.execute_with_model(name, tiny_prompt, temperature=0.0, max_tokens=5)
            if r.get('success'):
                results[name] = {"loaded": True, "latency_ms": r.get('latency_ms'), "error": None}
            else:
                results[name] = {"loaded": False, "latency_ms": r.get('latency_ms'), "error": r.get('error')}
        except Exception as e:  # noqa: BLE001
            results[name] = {"loaded": False, "latency_ms": None, "error": str(e)}
    return results

def _redis_key(*parts: str) -> str:
    return ':'.join([REDIS_PREFIX, *[p for p in parts if p is not None]])

async def _cache_set_json(key: str, payload: dict, ttl: Optional[int] = None):
    try:
        if not redis_client:
            return False
        data = json.dumps({"data": payload, "cached_at": datetime.utcnow().isoformat()})
        client = getattr(redis_client, 'client', None) or redis_client
        # Call method; if it returns an awaitable, await it; else assume sync
        method_name = 'setex' if ttl and ttl > 0 else 'set'
        method = getattr(client, method_name)
        res = method(key, ttl, data) if method_name == 'setex' else method(key, data)
        import asyncio as _asyncio, inspect as _inspect
        if _inspect.isawaitable(res):
            await res
        elif callable(res):  # unexpected callable result; run in thread
            await asyncio.to_thread(res)
        return True
    except Exception:
        return False

async def _cache_get_json(key: str) -> Optional[dict]:
    try:
        if not redis_client:
            return None
        client = getattr(redis_client, 'client', None) or redis_client
        meth = getattr(client, 'get')
        raw_res = meth(key)
        import asyncio as _asyncio, inspect as _inspect
        if _inspect.isawaitable(raw_res):
            raw = await raw_res
        else:
            raw = await asyncio.to_thread(lambda: raw_res)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        return json.loads(raw)
    except Exception:
        return None

async def _set_last_run(job: str, started_at: float, finished_at: float, ok: bool, extra: Optional[dict] = None):
    meta = {
        'job': job,
        'started_at': datetime.utcfromtimestamp(started_at).isoformat() + 'Z',
        'finished_at': datetime.utcfromtimestamp(finished_at).isoformat() + 'Z',
        'duration_seconds': round(finished_at - started_at, 4),
        'ok': ok,
    }
    if isinstance(extra, dict):
        meta.update(extra)
    await _cache_set_json(_redis_key(_SCHED_LAST_KEYS.get(job, f'last:{job}')), meta)
    # Export to Prometheus for Grafana panels/alerts
    try:
        if ML_INTEL_LAST_RUN_TS:
            ML_INTEL_LAST_RUN_TS.labels(job=job).set(finished_at)
        if ML_INTEL_LAST_DURATION:
            ML_INTEL_LAST_DURATION.labels(job=job).set(max(0.0, finished_at - started_at))
        if ML_INTEL_LAST_OK:
            ML_INTEL_LAST_OK.labels(job=job).set(1 if ok else 0)
    except Exception:
        pass

async def _redis_ready() -> bool:
    try:
        if redis_client is None:
            return False
        # Prefer underlying async client when using our RedisClient wrapper
        client = getattr(redis_client, 'client', None) or redis_client
        res = getattr(client, 'ping')()
        import asyncio as _asyncio, inspect as _inspect
        if _inspect.isawaitable(res):
            pong = await res
        else:
            pong = await asyncio.to_thread(lambda: res)
        return bool(pong)
    except Exception:
        return False

async def _run_correlation_job(symbols: list[str]) -> Optional[dict]:
    started = time.time()
    try:
        result = await analyze_market_network(symbols, analysis_type='correlation')
        await _cache_set_json(_redis_key('correlation','latest'), result)
        await _set_last_run('correlation', started, time.time(), True, {'symbols_count': len(symbols)})
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Correlation job failed: {e}")
        await _set_last_run('correlation', started, time.time(), False, {'error': str(e)})
        return None

async def _run_factors_job(symbols: list[str]) -> Optional[dict]:
    started = time.time()
    try:
        # Limit to top symbols with most data to avoid query size limits
        # Factor model needs quality data - use known large-cap symbols first
        max_symbols = 50  # Limit to avoid HTTP query size issues
        symbols_subset = symbols[:max_symbols] if len(symbols) > max_symbols else symbols
        
        result = await analyze_factors(symbols_subset, factors=None)
        await _cache_set_json(_redis_key('factors','latest'), result)
        await _set_last_run('factors', started, time.time(), True, {'symbols_count': len(symbols_subset), 'symbols_total': len(symbols)})
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Factors job failed: {e}")
        await _set_last_run('factors', started, time.time(), False, {'error': str(e)})
        return None

async def _run_risk_job(symbols: list[str]) -> Optional[dict]:
    started = time.time()
    try:
        # Compute risk per first symbol vs SPY and also a compact portfolio summary
        risk_map: dict[str, dict] = {}
        for s in symbols:
            try:
                m = await calculate_risk_metrics([s], DEFAULT_TIMEFRAME)
                if m:
                    risk_map[s] = m
                    # also persist per-symbol latest risk for dashboards
                    try:
                        await _cache_set_json(_redis_key('risk', s, 'latest'), m)
                        # Export selected metrics
                        try:
                            sharpe = m.get('sharpe_ratio')
                            mdd = m.get('max_drawdown')
                            if sharpe is not None and ML_RISK_LAST_SHARPE:
                                ML_RISK_LAST_SHARPE.labels(symbol=s).set(float(sharpe))
                            if mdd is not None and ML_RISK_LAST_MDD:
                                ML_RISK_LAST_MDD.labels(symbol=s).set(float(mdd))
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass
        # crude aggregated portfolio risk metric
        portfolio = {}
        try:
            # average sharpe, worst drawdown among symbols
            sharpes = [v.get('sharpe_ratio') for v in risk_map.values() if v.get('sharpe_ratio') is not None]
            mdds = [v.get('max_drawdown') for v in risk_map.values() if v.get('max_drawdown') is not None]
            if sharpes:
                portfolio['avg_sharpe'] = float(sum(sharpes)/len(sharpes))
            if mdds:
                portfolio['worst_mdd'] = float(max(mdds))
        except Exception:
            pass
        payload = {"symbols": symbols, "risk": risk_map, "portfolio": portfolio, "timestamp": datetime.utcnow().isoformat()}
        await _cache_set_json(_redis_key('risk','latest'), payload)
        await _set_last_run('risk', started, time.time(), True, {'symbols_count': len(symbols)})
        return payload
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Risk job failed: {e}")
        await _set_last_run('risk', started, time.time(), False, {'error': str(e)})
        return None

async def _scheduler_loop():
    """Feature-flagged background scheduler for periodic intelligence computations."""
    if not ENABLE_ML_SCHEDULER:
        return
    # Determine initial watchlist (dynamic when configured)
    try:
        dynamic_list = await _resolve_watchlist()
        current_watch = dynamic_list if dynamic_list else SCHED_WATCHLIST
    except Exception:
        current_watch = SCHED_WATCHLIST
    logger.info("ML scheduler enabled", watchlist=current_watch, tf=DEFAULT_TIMEFRAME)
    # Stagger first runs to avoid thundering herd at startup
    t0 = time.time()
    last_corr = 0.0
    last_factors = 0.0
    last_risk = 0.0
    # small random offset to decorrelate instances if replicated
    try:
        offset = (uuid.uuid4().int % 7)
    except Exception:
        offset = 0
    await asyncio.sleep(offset)
    while True:
        now = time.time()
        tasks: list[asyncio.Task] = []
        try:
            # Ensure Redis is available before scheduling cache-producing jobs
            if not await _redis_ready():
                await asyncio.sleep(3)
                continue
            if (now - last_corr) >= max(60, CORRELATION_INTERVAL):
                if WATCHLIST_SOURCE in ('questdb','db'):
                    try:
                        current_watch = (await _resolve_watchlist()) or current_watch
                    except Exception:
                        pass
                tasks.append(asyncio.create_task(_run_correlation_job(current_watch)))
                last_corr = now
            if (now - last_factors) >= max(60, FACTORS_INTERVAL):
                tasks.append(asyncio.create_task(_run_factors_job(current_watch)))
                last_factors = now
            if (now - last_risk) >= max(60, RISK_INTERVAL):
                tasks.append(asyncio.create_task(_run_risk_job(current_watch)))
                last_risk = now
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Scheduler iteration error: {e}")
        # sleep a short cadence to check timers
        await asyncio.sleep(5)

async def _initial_fill():
    """Run one-time initial computations shortly after startup to prime caches."""
    if not ENABLE_ML_SCHEDULER:
        return
    try:
        # Wait until Redis is ready (up to ~30s) to ensure cache writes succeed
        for _ in range(30):
            if await _redis_ready():
                break
            await asyncio.sleep(1)
        try:
            wl = await _resolve_watchlist()
            wl_use = wl if wl else SCHED_WATCHLIST
        except Exception:
            wl_use = SCHED_WATCHLIST
        await _run_correlation_job(wl_use)
        await _run_factors_job(wl_use)
        await _run_risk_job(wl_use)
        logger.info("Initial intelligence cache fill completed")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Initial fill error: {e}")

async def _qdb_http_query(sql: str) -> list[dict]:
    """Execute SQL via QuestDB HTTP endpoint, returning list of dict rows.
    Uses QUESTDB_HTTP_URL env or defaults to http://trading-questdb:9000/exec.
    """
    import httpx
    base = os.getenv('QUESTDB_HTTP_URL', 'http://questdb:9000/exec')
    q = urllib.parse.urlencode({'fmt': 'json', 'query': sql})
    url = f"{base}?{q}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        j = r.json()
        cols = [c['name'] for c in j.get('columns', [])]
        out: list[dict] = []
        for row in j.get('dataset', []) or []:
            rec = {}
            for i, v in enumerate(row):
                key = cols[i] if i < len(cols) else f"c{i}"
                rec[key] = v
            out.append(rec)
        return out

async def _fetch_watchlist_from_redis() -> list[str]:
    """Fetch the production watchlist from Redis.
    This is the authoritative source - maintained by production_watchlist_manager.sh
    with optionable symbols discovered from Polygon.
    """
    try:
        if not redis_client:
            return []
        symbols = await redis_client.smembers('watchlist')
        if not symbols:
            return []
        out = sorted([s.strip().upper() for s in symbols if s and s.strip()])
        logger.info(f"Fetched {len(out)} symbols from Redis watchlist")
        return out
    except Exception as e:
        logger.warning(f"Failed to fetch Redis watchlist: {e}")
        return []

async def _fetch_watchlist_from_questdb() -> list[str]:
    """Fetch a dynamic watchlist from QuestDB based on recent activity.
    BACKUP method only - primary source is Redis.
    """
    try:
        start = (datetime.utcnow() - timedelta(days=max(1, WATCHLIST_LOOKBACK_DAYS))).date().isoformat() + 'T00:00:00Z'
        sql = (
            "SELECT DISTINCT symbol FROM daily_bars "
            f"WHERE timestamp >= '{start}' ORDER BY symbol ASC LIMIT {max(1, WATCHLIST_MAX_SYMBOLS)}"
        )
        rows = await _qdb_http_query(sql)
        out = []
        for r in rows:
            try:
                s = (r.get('symbol') or '').strip().upper()
                if s:
                    out.append(s)
            except Exception:
                continue
        return out
    except Exception:
        return []

async def _resolve_watchlist() -> list[str]:
    """Resolve the scheduler watchlist from configured source with safe fallback.
    
    Priority:
    1. Redis watchlist (production optionable symbols from Polygon)
    2. QuestDB (symbols with recent data)
    3. Env list (hardcoded fallback)
    """
    # Try Redis first (authoritative source)
    try:
        lst = await _fetch_watchlist_from_redis()
        if lst and len(lst) >= 100:  # Sanity check
            return lst
    except Exception:
        pass
    
    # Fallback to QuestDB if configured
    if WATCHLIST_SOURCE in ('questdb', 'db'):
        try:
            lst = await _fetch_watchlist_from_questdb()
            if lst:
                return lst
        except Exception:
            pass
    
    # Last resort: env-provided list
    return SCHED_WATCHLIST

class EquityBarVector(BaseModel):
    symbol: str
    timestamp: str  # YYYY-MM-DD
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | int | None = None

class EquityBarBatch(BaseModel):
    bars: list[EquityBarVector]

class OptionContractVector(BaseModel):
    underlying: str
    option_symbol: str | None = None
    expiry: str | None = None
    right: str | None = None
    strike: float | None = None
    implied_vol: float | None = None
    timestamp: str | None = None

class OptionContractBatch(BaseModel):
    contracts: list[OptionContractVector]

# Metrics (guarded against duplicate registration)


def _metric_exists(name: str) -> bool:
    if os.getenv('PROM_DISABLE_METRIC_REDEFINE'):
        return True  # short-circuit registration entirely if flag set
    try:


        if hasattr(REGISTRY, '_names'):
            return name in getattr(REGISTRY, '_names')  # type: ignore[attr-defined]
        for collector in getattr(REGISTRY, '_collector_to_names', {}).keys():  # type: ignore[attr-defined]
            if hasattr(collector, 'name') and getattr(collector, 'name') == name:
                return True
        return False
    except Exception:
        return False



ML_INFERENCE_LATENCY = Histogram('ml_inference_latency_seconds', 'Model inference latency', ['model']) if not _metric_exists('ml_inference_latency_seconds') else None  # type: ignore
ML_INFERENCE_TOKENS = Counter('ml_inference_tokens_total', 'Total tokens processed per model', ['model']) if not _metric_exists('ml_inference_tokens_total') else None  # type: ignore


ML_INFERENCE_CPU_SECONDS = Counter('ml_inference_cpu_seconds_total', 'Approx CPU seconds consumed per model', ['model']) if not _metric_exists('ml_inference_cpu_seconds_total') else None  # type: ignore
ML_COMPONENTS_LOADED = Gauge('ml_components_loaded', 'Number of initialized ML components') if not _metric_exists('ml_components_loaded') else None  # type: ignore
ML_DRIFT_EVENTS = Counter('ml_drift_events_total', 'Drift events detected') if not _metric_exists('ml_drift_events_total') else None  # type: ignore
ML_DRIFT_EVENTS_SEVERITY = Counter('ml_drift_events_severity_total', 'Drift events by severity and type', ['severity','drift_type']) if not _metric_exists('ml_drift_events_severity_total') else None  # type: ignore
ML_INFERENCE_REQUESTS = Counter('ml_inference_requests_total', 'Inference requests per model and status', ['model','status']) if not _metric_exists('ml_inference_requests_total') else None  # type: ignore
ML_ENDPOINT_ERRORS = Counter('ml_endpoint_errors_total', 'Errors per endpoint', ['endpoint']) if not _metric_exists('ml_endpoint_errors_total') else None  # type: ignore
ML_CANARY_REQUESTS = Counter('ml_canary_requests_total', 'Canary dual inference requests', ['primary','canary','status']) if not _metric_exists('ml_canary_requests_total') else None  # type: ignore
ML_CANARY_DIVERGENCE = Histogram('ml_canary_divergence_score', 'Divergence score between primary and canary output', buckets=(0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0)) if not _metric_exists('ml_canary_divergence_score') else None  # type: ignore
ML_CANARY_LATENCY_DELTA = Histogram('ml_canary_latency_delta_seconds', 'Canary latency minus primary latency', buckets=( -1.0,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1,2)) if not _metric_exists('ml_canary_latency_delta_seconds') else None  # type: ignore

# ----------------------------------------------------------------------------
# Governance / Lifecycle Canonical Metrics (placeholder emission at startup)
# These align with health script expectations (app_ml_* & app_inference_* names)
# and allow dashboards to stabilize even before real governance workflows fire.
# ----------------------------------------------------------------------------
_GOVERNANCE_METRICS_REGISTERED = False

def _register_governance_metrics():
    global _GOVERNANCE_METRICS_REGISTERED
    if _GOVERNANCE_METRICS_REGISTERED:
        return
    try:
        # State transitions across promotion lifecycle (e.g., shadow -> canary -> primary)
        if not _metric_exists('app_ml_state_transitions_total'):
            globals()['APP_ML_STATE_TRANSITIONS'] = Counter(
                'app_ml_state_transitions_total',
                'Model governance state transitions',
                ['from_state','to_state','model']
            )


        # Promotions accepted into production
        if not _metric_exists('app_ml_promotions_total'):
            globals()['APP_ML_PROMOTIONS'] = Counter(

                'app_ml_promotions_total',
                'Accepted model promotions into production',
                ['model','decision']
            )
        # Manual or automatic rollbacks
        if not _metric_exists('app_ml_rollbacks_total'):
            globals()['APP_ML_ROLLBACKS'] = Counter(
                'app_ml_rollbacks_total',
                'Model rollbacks executed',
                ['model','reason']
            )

        # Sequential Probability Ratio Test (SPRT) decisions for canary trials
        if not _metric_exists('app_inference_sprt_decisions_total'):
            globals()['APP_INFERENCE_SPRT_DECISIONS'] = Counter(
                'app_inference_sprt_decisions_total',
                'SPRT decisions (accept/reject) for candidate models',
                ['model','decision']
            )
        # Log-likelihood ratio gauge (instantaneous) for current SPRT window
        if not _metric_exists('app_inference_sprt_llr'):
            globals()['APP_INFERENCE_SPRT_LLR'] = Gauge(
                'app_inference_sprt_llr',
                'Current SPRT log-likelihood ratio',
                ['model']
            )
        # Shadow directional accuracy (label-based) gauge


        if not _metric_exists('app_inference_shadow_directional_accuracy'):
            globals()['APP_INFERENCE_SHADOW_DIRECTIONAL_ACCURACY'] = Gauge(
                'app_inference_shadow_directional_accuracy',
                'Directional accuracy of shadow model vs baseline',


                ['model','window']
            )
        # Shadow prediction totals (to compute coverage ratios)
        if not _metric_exists('app_inference_shadow_predictions_total'):
            globals()['APP_INFERENCE_SHADOW_PREDICTIONS'] = Counter(
                'app_inference_shadow_predictions_total',
                'Total shadow model predictions observed',
                ['model']
            )
        # Circuit breaker trip counts for ML safety guardrails
        if not _metric_exists('app_ml_circuit_breaker_trips_total'):
            globals()['APP_ML_CIRCUIT_BREAKER_TRIPS'] = Counter(
                'app_ml_circuit_breaker_trips_total',
                'Circuit breaker trips for ML model serving',
                ['breaker','model']
            )
        # Automatic rollback triggers
        if not _metric_exists('app_ml_auto_rollbacks_total'):
            globals()['APP_ML_AUTO_ROLLBACKS'] = Counter(
                'app_ml_auto_rollbacks_total',
                'Automatic rollback events (policy-driven)',
                ['model','policy']
            )
        # Validation / pre-promotion failures
        if not _metric_exists('app_ml_validation_failures_total'):
            globals()['APP_ML_VALIDATION_FAILURES'] = Counter(
                'app_ml_validation_failures_total',
                'Pre-promotion validation failures',
                ['model','stage']
            )
        # Governance last update timestamp gauge (used for silence detection)
        if not _metric_exists('app_ml_governance_last_update_timestamp_seconds'):
            globals()['APP_ML_GOVERNANCE_LAST_UPDATE_TIMESTAMP'] = Gauge(
                'app_ml_governance_last_update_timestamp_seconds',
                'Unix timestamp of last governance-related metric emission'
            )
        _GOVERNANCE_METRICS_REGISTERED = True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to register governance metrics: {e}")

def _zero_governance_metrics():
    """Emit zero samples so that metrics appear immediately after startup."""
    if not _GOVERNANCE_METRICS_REGISTERED:
        return
    try:
        # Use a generic placeholder model id so series exist; real labels will appear later.
        placeholder_model = 'uninitialized'
        # Counters: increment 0 by calling inc(0); Gauges: set(0)
        for name in [
            'APP_ML_STATE_TRANSITIONS','APP_ML_PROMOTIONS','APP_ML_ROLLBACKS',
            'APP_INFERENCE_SPRT_DECISIONS','APP_INFERENCE_SHADOW_PREDICTIONS',
            'APP_ML_CIRCUIT_BREAKER_TRIPS','APP_ML_AUTO_ROLLBACKS','APP_ML_VALIDATION_FAILURES'
        ]:
            metric = globals().get(name)
            if metric is not None:
                try:
                    if name == 'APP_ML_STATE_TRANSITIONS':
                        metric.labels(from_state='none', to_state='initializing', model=placeholder_model).inc(0)
                    elif name == 'APP_ML_PROMOTIONS':
                        metric.labels(model=placeholder_model, decision='accept').inc(0)
                    elif name == 'APP_ML_ROLLBACKS':
                        metric.labels(model=placeholder_model, reason='none').inc(0)
                    elif name == 'APP_INFERENCE_SPRT_DECISIONS':
                        metric.labels(model=placeholder_model, decision='accept').inc(0)
                        metric.labels(model=placeholder_model, decision='reject').inc(0)
                    elif name == 'APP_INFERENCE_SHADOW_PREDICTIONS':
                        metric.labels(model=placeholder_model).inc(0)
                    elif name == 'APP_ML_CIRCUIT_BREAKER_TRIPS':
                        metric.labels(breaker='quality', model=placeholder_model).inc(0)
                    elif name == 'APP_ML_AUTO_ROLLBACKS':
                        metric.labels(model=placeholder_model, policy='quality-guard').inc(0)
                    elif name == 'APP_ML_VALIDATION_FAILURES':
                        metric.labels(model=placeholder_model, stage='pre-promotion').inc(0)
                except Exception:  # noqa: BLE001
                    pass
        # Gauges
        for name in ['APP_INFERENCE_SPRT_LLR','APP_INFERENCE_SHADOW_DIRECTIONAL_ACCURACY']:
            metric = globals().get(name)
            if metric is not None:
                try:
                    if name == 'APP_INFERENCE_SPRT_LLR':
                        metric.labels(model=placeholder_model).set(0)
                    else:
                        metric.labels(model=placeholder_model, window='short').set(0)
                except Exception:  # noqa: BLE001
                    pass
        # Initialize governance last update timestamp to current time (so alert window starts now)
        try:
            ts_g = globals().get('APP_ML_GOVERNANCE_LAST_UPDATE_TIMESTAMP')
            if ts_g is not None:
                ts_g.set(time.time())
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to zero governance metrics: {e}")

# Gauge signaling governance metrics ready for early health scripts
try:
    if not any(c for c in getattr(REGISTRY, '_names', set()) if c == 'ml_governance_metrics_ready'):
        ML_GOVERNANCE_METRICS_READY = Gauge('ml_governance_metrics_ready', '1 when governance metrics initialized and zeroed')
    else:
        ML_GOVERNANCE_METRICS_READY = None
except Exception:  # noqa: BLE001
    ML_GOVERNANCE_METRICS_READY = None

# Register governance metrics immediately (safe; idempotent due to _metric_exists)
_register_governance_metrics()
"""Immediately emit zero samples so governance metrics are visible on first scrape.
This must occur before any Prometheus server scrape interval elapses to avoid
alert false-positives for 'missing series'."""
_zero_governance_metrics()
try:
    if ML_GOVERNANCE_METRICS_READY:
        ML_GOVERNANCE_METRICS_READY.set(1)
except Exception:  # noqa: BLE001
    pass

# Component initialization metrics (deferred startup tracking)
ML_COMPONENT_INIT_STATUS = Gauge('ml_component_init_status', 'Initialization status per component (0=pending,1=initializing,2=ready,3=failed)', ['component']) if not _metric_exists('ml_component_init_status') else None  # type: ignore
ML_COMPONENT_INIT_DURATION = Histogram('ml_component_init_duration_seconds', 'Initialization duration per component', ['component']) if not _metric_exists('ml_component_init_duration_seconds') else None  # type: ignore

# FastAPI app must be defined before decorators referencing @app.*
app = FastAPI(
    title="AI Trading System - Advanced ML Service",
    description="Orchestrates machine learning models and intelligence systems",
    version="2.0.0",
    lifespan=None  # temporary placeholder; replaced after lifespan defined
)

# Environment flag for canary enablement
CANARY_ENABLED = os.getenv('ML_CANARY_ENABLED', 'false').lower() in ('1','true','yes')
CANARY_MODEL = os.getenv('ML_CANARY_MODEL')  # optional override

# --------------------------- Ollama Mode Status Helpers --------------------------- #
_LAST_WARM_STATUS_KEY = _redis_key('ollama','last_warm_status')

async def _record_warm_status(mode: str, targets: list[str], results: dict):
    try:
        payload = {
            'mode': mode,
            'targets': targets,
            'results': results,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        await _cache_set_json(_LAST_WARM_STATUS_KEY, payload, ttl=24*3600)
    except Exception:
        pass

@app.get('/ollama/mode')
async def ollama_mode_status():
    """Report current day/night mode based on market hours and configured model targets.

    Returns:
      - mode: 'day' or 'night'
      - market_hours: bool flag
      - configured: {day_hot_models, night_heavy_models, deepseek_enabled, deepseek_model}
      - last_warm_status: most recent warmup attempt (if any)
    """
    try:
        is_market = _is_market_hours()
        mode = 'day' if is_market else 'night'
        last = await _cache_get_json(_LAST_WARM_STATUS_KEY)
        return {
            'mode': mode,
            'market_hours': is_market,
            'configured': {
                'day_hot_models': DAY_HOT_MODELS,
                'night_heavy_models': NIGHT_HEAVY_MODELS,
                'llama_night_enabled': _is_llama_night_allowed(),
                'llama_night_model': LLAMA_NIGHT_MODEL,
            },
            'last_warm_status': last.get('data') if isinstance(last, dict) else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _recount_components():
    count = sum(1 for name in [
        llm_service, intelligence_coordinator, ensemble_coordinator, regime_detector, factor_model,
        volatility_model, graph_network, stochastic_vol, entropy_analyzer, timeframe_intelligence,
        drift_monitor, improvement_engine, analytics_service, training_pipeline, backtesting_framework,
        model_router, ollama_service, finbert_analyzer
    ] if name is not None)
    ML_COMPONENTS_LOADED.set(count)

def _normalize_path(p: str) -> str:
    """Normalize path for metrics label cardinality (placeholder for future dynamic segments)."""
    return p

def _ensure_news_embedder():
    global _news_embedder
    if _news_embedder is not None:
        return _news_embedder
    model_name = os.getenv('NEWS_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed in ML service")
    _news_embedder = SentenceTransformer(model_name)
    return _news_embedder

async def _infer(model: str, awaitable):
    """Execute awaitable measuring latency, token & CPU usage, and success/error counts.

    Token attribution:
      - Expects returned object or value to possibly include a 'tokens_used' attribute or key.
      - Falls back to naive whitespace token approximation for string responses.
    CPU estimation:
      - Uses wall-clock as proxy (could be refined with psutil if available later).
    """
    start = time.perf_counter()
    status = 'success'
    result = None
    try:
        result = await awaitable
        return result
    except Exception:
        status = 'error'
        raise
    finally:
        elapsed = time.perf_counter() - start
        ML_INFERENCE_LATENCY.labels(model=model).observe(elapsed)
        ML_INFERENCE_REQUESTS.labels(model=model, status=status).inc()
        # Token extraction logic
        try:
            tokens = 0
            if isinstance(result, dict) and 'metadata' in result and isinstance(result['metadata'], dict):
                tokens = int(result['metadata'].get('tokens_used', 0) or 0)
            elif hasattr(result, 'metadata') and isinstance(getattr(result, 'metadata'), dict):
                tokens = int(getattr(result, 'metadata').get('tokens_used', 0) or 0)  # type: ignore[arg-type]
            elif isinstance(result, str):
                tokens = len(result.split())
            if tokens > 0:
                ML_INFERENCE_TOKENS.labels(model=model).inc(tokens)
        except Exception:
            pass
        try:
            ML_INFERENCE_CPU_SECONDS.labels(model=model).inc(elapsed)
        except Exception:
            pass

@app.middleware('http')
async def correlation_component_count_middleware(request: Request, call_next):
    """Attach correlation ID and update component count post-response (request metrics handled by shared layer)."""
    cid = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
    response = await call_next(request)
    try:
        _recount_components()
        # Touch governance timestamp on activity to avoid false silence alerts until orchestrator integrated
        try:
            ts_g = globals().get('APP_ML_GOVERNANCE_LAST_UPDATE_TIMESTAMP')
            if ts_g is not None:
                ts_g.set(time.time())
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass
    response.headers['X-Correlation-ID'] = cid
    return response

# ---------------------- Vector Store Supplemental Endpoints ---------------------- #
@app.post('/vector/schema/extend')
async def vector_schema_extend():
    """Ensure extended schema (OptionContract, EquityBar) applied."""
    if not (get_weaviate_client and desired_schema and fetch_current_schema and diff_schema and apply_schema_changes):
        raise HTTPException(status_code=503, detail='Weaviate utilities unavailable')
    try:
        client = await asyncio.to_thread(get_weaviate_client)
        current = await asyncio.to_thread(fetch_current_schema, client)
        target = desired_schema()
        diff = await asyncio.to_thread(diff_schema, current, target)
        changed = False
        if diff.get('add_classes') or diff.get('add_properties'):
            await asyncio.to_thread(apply_schema_changes, diff, client)
            changed = True
        return {"status":"ok","changed":changed,"diff":{k:v for k,v in diff.items() if v}}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------- Ollama Helper APIs --------------------------- #
@app.get('/ollama/models')
async def list_ollama_models():
    """List locally available Ollama models via the router/service."""
    try:
        # Always refresh availability to reflect current Ollama reachability (no pulls)
        if model_router and hasattr(model_router, 'check_model_availability'):
            try:
                await model_router.check_model_availability()
            except Exception:
                pass
        out = []
        if model_router and hasattr(model_router, 'models'):
            for name, info in model_router.models.items():
                try:
                    out.append({'name': name, 'available': bool(getattr(info, 'is_available', True))})
                except Exception:
                    out.append({'name': name, 'available': True})
        return {'models': out}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/ollama/refresh')
async def refresh_ollama_models():
    """Force a non-destructive refresh of model availability from Ollama."""
    try:
        if not model_router or not hasattr(model_router, 'check_model_availability'):
            raise HTTPException(status_code=503, detail='model_router_unavailable')
        availability = await model_router.check_model_availability()
        return {'status': 'ok', 'availability': availability, 'host': getattr(model_router, 'ollama_host', None)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class GeneratePayload(BaseModel):
    prompt: str
    task_type: str | None = None  # maps to TaskType
    urgency: str | None = None    # maps to TaskUrgency
    model: str | None = None      # optional explicit model id
    temperature: float | None = 0.3
    max_tokens: int | None = 512

def _map_task_type(name: str | None) -> TaskType:
    if not name:
        return TaskType.MARKET_ANALYSIS
    n = name.strip().lower()
    mapping = {
        'signal': TaskType.SIGNAL_GENERATION,
        'risk': TaskType.RISK_ASSESSMENT,
        'market': TaskType.MARKET_ANALYSIS,
        'document': TaskType.DOCUMENT_ANALYSIS,
        'metrics': TaskType.FINANCIAL_METRICS,
        'options': TaskType.OPTIONS_PRICING,
        'strategy': TaskType.STRATEGY_SELECTION,
        'sentiment': TaskType.SENTIMENT_ANALYSIS,
        'technical': TaskType.TECHNICAL_ANALYSIS,
        'news': TaskType.NEWS_PROCESSING,
    }
    return mapping.get(n, TaskType.MARKET_ANALYSIS)

def _map_urgency(name: str | None) -> TaskUrgency:
    if not name:
        return TaskUrgency.NORMAL
    n = name.strip().lower()
    mapping = {
        'realtime': TaskUrgency.REALTIME,
        'fast': TaskUrgency.FAST,
        'normal': TaskUrgency.NORMAL,
        'batch': TaskUrgency.BATCH,
        'deep': TaskUrgency.DEEP,
    }
    return mapping.get(n, TaskUrgency.NORMAL)

@app.post('/ollama/generate')
async def ollama_generate(payload: GeneratePayload):
    if not model_router:
        raise HTTPException(status_code=503, detail='model_router_unavailable')
    try:
        task = _map_task_type(payload.task_type)
        urg = _map_urgency(payload.urgency)
        temp = float(payload.temperature or 0.3)
        max_toks = int(payload.max_tokens or 512)
        if payload.model:
            res = await model_router.execute_with_model(payload.model, payload.prompt, temperature=temp, max_tokens=max_toks)
        else:
            res = await model_router.smart_execute(task, urg, payload.prompt, temperature=temp, max_tokens=max_toks, require_long_context=(task in (TaskType.DOCUMENT_ANALYSIS, TaskType.NEWS_PROCESSING)))
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SamplePayload(BaseModel):
    mode: str | None = None  # 'day' | 'night' | 'all'
    prompt: str | None = None
    max_tokens: int | None = 256
    temperature: float | None = 0.2
    concurrency: int | None = 3
    per_model_timeout_seconds: float | None = None
    include_unavailable: bool | None = False
    skip_heavy: bool | None = True
    heavy_memory_gb: int | None = None

@app.post('/ollama/sample')
async def ollama_sample(payload: SamplePayload | None = None):
    if not model_router:
        raise HTTPException(status_code=503, detail='model_router_unavailable')
    mode = (payload.mode if payload else None) or ('day' if _is_market_hours() else 'night')
    prompt = (payload.prompt if payload else None) or "Summarize today's US market action in one concise paragraph."
    max_tokens = int((payload.max_tokens if payload else None) or 256)
    temperature = float((payload.temperature if payload else None) or 0.2)
    concurrency = max(1, int((payload.concurrency if payload else None) or 2))
    per_model_timeout = payload.per_model_timeout_seconds if payload and payload.per_model_timeout_seconds is not None else None
    targets: list[str] = []
    if mode in ('day','all'):
        targets.extend(DAY_HOT_MODELS)
    if mode in ('night','all'):
        targets.extend(NIGHT_HEAVY_MODELS)
    # de-dup while preserving order
    seen = set(); ordered = []
    for m in targets:
        if m not in seen:
            seen.add(m); ordered.append(m)
    # Refresh availability and filter if requested
    try:
        availability = await model_router.check_model_availability()
    except Exception:
        availability = {name: True for name in ordered}
    filtered = []
    # Determine heavy threshold
    try:
        heavy_threshold = int(payload.heavy_memory_gb) if (payload and payload.heavy_memory_gb is not None) else int(os.getenv('ROUTER_HEAVY_MEMORY_GB', '40'))
    except Exception:
        heavy_threshold = 40
    for name in ordered:
        if not ((payload and payload.include_unavailable) or availability.get(name, True)):
            continue
        if payload and payload.skip_heavy:
            try:
                info = model_router.models.get(name)
                if info and getattr(info, 'memory_gb', 0) >= heavy_threshold:
                    continue
            except Exception:
                pass
        filtered.append(name)
    # Per-model timeout heuristic: heavier models need more time
    def _default_timeout_for(model_name: str) -> float:
        heavy = any(model_name.startswith(p) for p in ("llama3.1:", "mixtral", "qwen2.5:", "command-r-plus:"))
        return 45.0 if heavy else 15.0
    timeout_for = lambda m: (per_model_timeout if per_model_timeout is not None else _default_timeout_for(m))

    # Run concurrently with a semaphore
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def _run_one(name: str):
        async with sem:
            try:
                t = timeout_for(name)
                r = await asyncio.wait_for(
                    model_router.execute_with_model(name, prompt, temperature=temperature, max_tokens=max_tokens),
                    timeout=t
                )
                return r
            except asyncio.TimeoutError:
                return {"success": False, "model": name, "error": f"timeout_after_{int(timeout_for(name))}s"}
            except Exception as e:  # noqa: BLE001
                return {"success": False, "model": name, "error": str(e)}

    tasks = [asyncio.create_task(_run_one(name)) for name in filtered]
    if tasks:
        done = await asyncio.gather(*tasks, return_exceptions=False)
        for r in done:
            try:
                results.append(r)
            except Exception:
                continue
    # Compact summary
    summary = []
    for r in results:
        try:
            # Compute tokens/sec if not supplied by router
            tps = r.get('tokens_per_sec')
            if tps is None:
                try:
                    tokens = float(r.get('tokens_used') or 0)
                    # Try to read durations from router extra fields
                    eval_ms = float(r.get('eval_duration_ms') or 0)
                    if eval_ms > 0:
                        tps = tokens / (eval_ms / 1000.0)
                except Exception:
                    tps = None
            summary.append({
                'model': r.get('model'),
                'success': bool(r.get('success')),
                'latency_ms': r.get('latency_ms'),
                'tokens_used': r.get('tokens_used'),
                'tokens_per_sec': tps,
                'used_fallback': r.get('used_fallback', False),
                'error': r.get('error')
            })
        except Exception:
            continue
    return {'mode': mode, 'count': len(results), 'results': summary}

# --------------------------- FinBERT Sentiment Endpoints --------------------------- #
class FinBertAnalyzePayload(BaseModel):
    text: str


class FinBertBatchPayload(BaseModel):
    texts: List[str]
    batch_size: int | None = 32


@app.get('/sentiment/finbert/health')
async def finbert_health():
    try:
        global finbert_analyzer
        if finbert_analyzer is None:
            # Lazily initialize if background init not finished yet
            finbert_analyzer = await asyncio.to_thread(lambda: FinBERTSentimentAnalyzer())
        device = getattr(finbert_analyzer, 'device', 'cpu')
        return {"status": "ok", "device": str(device)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/sentiment/finbert/analyze')
async def finbert_analyze(payload: FinBertAnalyzePayload):
    if not payload or not (payload.text and payload.text.strip()):
        raise HTTPException(status_code=400, detail='empty_text')
    try:
        global finbert_analyzer
        if finbert_analyzer is None:
            finbert_analyzer = await asyncio.to_thread(lambda: FinBERTSentimentAnalyzer())
        res = await finbert_analyzer.analyze_sentiment_async(payload.text)
        return {
            'text': res.text,
            'sentiment': res.sentiment,
            'confidence': res.confidence,
            'scores': res.scores
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/sentiment/finbert/batch')
async def finbert_batch(payload: FinBertBatchPayload):
    try:
        global finbert_analyzer
        if finbert_analyzer is None:
            finbert_analyzer = await asyncio.to_thread(lambda: FinBERTSentimentAnalyzer())
        texts = [t for t in (payload.texts or []) if isinstance(t, str)]
        if not texts:
            return {'count': 0, 'results': []}
        batch_size = int(payload.batch_size or 32)
        results = await finbert_analyzer.batch_analyze_async(texts, batch_size=batch_size)
        out = [{
            'text': r.text,
            'sentiment': r.sentiment,
            'confidence': r.confidence,
            'scores': r.scores
        } for r in results]
        return {'count': len(out), 'results': out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post('/ollama/warmup')
async def ollama_warmup(request: Request):
    """Warm specific models (or a curated default set) through the model router.
    This triggers tiny prompts to load models into memory without heavy generation.
    """
    try:
        if not model_router:
            raise HTTPException(status_code=503, detail='model_router_unavailable')
        default_models_csv = os.getenv('OLLAMA_WARMUP_MODELS', 'solar:10.7b,phi3:14b,mixtral:8x22b')
        targets: list[str] = []
        try:
            body = await request.json()
        except Exception:
            body = None
        # Accept either array of strings or {"models": [..]}
        if isinstance(body, list):
            targets = [str(m).strip() for m in body if str(m).strip()]
        elif isinstance(body, dict) and isinstance(body.get('models'), list):
            targets = [str(m).strip() for m in body.get('models') if str(m).strip()]
        if not targets:
            targets = [m.strip() for m in default_models_csv.split(',') if m.strip()]
        res = await _warmup_ollama_models(model_router, targets)
        return {'status': 'ok', 'results': res}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/vector/index/equity/batch')
async def vector_index_equity_batch(payload: Union[List[EquityBarVector], EquityBarBatch, dict]):
    if not get_weaviate_client:
        raise HTTPException(status_code=503, detail='Weaviate client unavailable')
    try:
        client = await asyncio.to_thread(get_weaviate_client)
        coll = client.collections.get('EquityBar')
        if isinstance(payload, EquityBarBatch):
            bars_in = payload.bars
        elif isinstance(payload, list):
            bars_in = payload
        elif isinstance(payload, dict) and 'bars' in payload and isinstance(payload['bars'], list):
            bars_in = [EquityBarVector(**b) for b in payload['bars'] if isinstance(b, dict)]
        else:
            raise HTTPException(status_code=400, detail='Unsupported payload shape')
        inserted = 0
        errors = 0
        for b in bars_in:
            try:
                # Ensure RFC3339 date (append time if only YYYY-MM-DD)
                ts_val = b.timestamp
                if ts_val and len(ts_val) == 10:  # YYYY-MM-DD
                    ts_val = ts_val + 'T00:00:00Z'
                coll.data.insert({
                    'symbol': (b.symbol or '').upper(),
                    'timestamp': ts_val,
                    'open': '' if b.open is None else str(b.open),
                    'high': '' if b.high is None else str(b.high),
                    'low': '' if b.low is None else str(b.low),
                    'close': '' if b.close is None else str(b.close),
                    'volume': '' if b.volume is None else str(b.volume),
                })
                inserted += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                if errors <= 3:
                    logger.warning(f"EquityBar insert error: {e}")
        return {"status":"ok","indexed":inserted,"errors":errors}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/vector/index/options/batch')
async def vector_index_options_batch(payload: Union[List[OptionContractVector], OptionContractBatch, dict]):
    if not get_weaviate_client:
        raise HTTPException(status_code=503, detail='Weaviate client unavailable')
    try:
        client = await asyncio.to_thread(get_weaviate_client)
        coll = client.collections.get('OptionContract')
        if isinstance(payload, OptionContractBatch):
            contracts_in = payload.contracts
        elif isinstance(payload, list):
            contracts_in = payload
        elif isinstance(payload, dict) and 'contracts' in payload and isinstance(payload['contracts'], list):
            contracts_in = [OptionContractVector(**c) for c in payload['contracts'] if isinstance(c, dict)]
        else:
            raise HTTPException(status_code=400, detail='Unsupported payload shape')
        inserted = 0
        errors = 0
        for c in contracts_in:
            try:
                exp_val = c.expiry
                if exp_val and len(exp_val) == 10:
                    exp_val = exp_val + 'T00:00:00Z'
                ts_val = c.timestamp
                if ts_val and len(ts_val) == 10:
                    ts_val = ts_val + 'T00:00:00Z'
                coll.data.insert({
                    'underlying': (c.underlying or '').upper(),
                    'option_symbol': c.option_symbol,
                    'expiry': exp_val,
                    'right': (c.right or '').upper(),
                    'strike': '' if c.strike is None else str(c.strike),
                    'implied_vol': '' if c.implied_vol is None else str(c.implied_vol),
                    'timestamp': ts_val,
                })
                inserted += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                if errors <= 3:
                    logger.warning(f"OptionContract insert error: {e}")
        return {"status":"ok","indexed":inserted, "errors":errors}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/metrics')  # type: ignore[name-defined]
def metrics():  # type: ignore[name-defined]
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle with deferred heavy component initialization."""
    global cache_client, redis_client, llm_service, intelligence_coordinator
    global ensemble_coordinator, regime_detector, factor_model, volatility_model
    global graph_network, stochastic_vol, entropy_analyzer, timeframe_intelligence
    global drift_monitor, improvement_engine, analytics_service, training_pipeline
    global backtesting_framework, model_router, ollama_service, finbert_analyzer

    logger.info("Starting Advanced ML Service (deferred init mode)")

    COMPONENT_STATUSES: Dict[str, Dict[str, Any]] = {}
    STATUS_MAP = {"pending": 0, "initializing": 1, "ready": 2, "failed": 3}

    def _set_status(name: str, status: str, err: Optional[str] = None, duration: Optional[float] = None):
        now = time.time()
        rec = COMPONENT_STATUSES.get(name, {"started_at": now, "ended_at": None, "error": None})
        if status == "initializing":
            rec["started_at"] = now
        if status in ("ready", "failed"):
            rec["ended_at"] = now
            if duration is None:
                try:
                    duration = rec["ended_at"] - rec["started_at"]
                except Exception:
                    duration = None
        rec["status"] = status
        rec["error"] = err
        if duration is not None and ML_COMPONENT_INIT_DURATION:
            try:
                ML_COMPONENT_INIT_DURATION.labels(component=name).observe(duration)
            except Exception:
                pass
        COMPONENT_STATUSES[name] = rec
        if ML_COMPONENT_INIT_STATUS:
            try:
                ML_COMPONENT_INIT_STATUS.labels(component=name).set(STATUS_MAP.get(status, -1))
            except Exception:
                pass

    # Critical synchronous path
    try:
        _set_status("infrastructure", "initializing")
        # Ensure async cache and Redis connections are established
        cache_client = await get_trading_cache()
        redis_client = get_redis_client()
        try:
            await redis_client.connect()
        except Exception as re:  # noqa: BLE001
            logger.warning("Redis connect failed in ML lifespan", error=str(re))
        _set_status("infrastructure", "ready")
        logger.info("Infrastructure ready", component="infrastructure")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Infrastructure connections failed: {e}")
        _set_status("infrastructure", "failed", err=str(e))

    try:
        _set_status("llm_service", "initializing")
        llm_service = ProductionLLMService()
        _set_status("llm_service", "ready")
        logger.info("LLM service initialized", component="llm")
    except Exception as e:
        logger.warning(f"LLM service initialization failed: {e}")
        _set_status("llm_service", "failed", err=str(e))

    for comp_name, ctor in [
        ("advanced_intelligence", AdvancedIntelligenceCoordinator),
        ("ensemble_intelligence", EnsembleIntelligenceCoordinator),
        ("regime_detector", MarketRegimeDetector)
    ]:
        try:
            _set_status(comp_name, "initializing")
            instance = ctor()
            if comp_name == "advanced_intelligence":
                intelligence_coordinator = instance
            elif comp_name == "ensemble_intelligence":
                ensemble_coordinator = instance
            else:
                regime_detector = instance
            _set_status(comp_name, "ready")
            logger.info("Component initialized", component=comp_name)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Component {comp_name} failed: {e}")
            _set_status(comp_name, "failed", err=str(e))

    # Drift monitor early
    try:
        _set_status("drift_monitor", "initializing")
        drift_monitor = DriftMonitor(model_id="ml-service-primary")
        if hasattr(drift_monitor, 'detect_drift') and not hasattr(drift_monitor, '_metrics_wrapped'):
            orig_detect = drift_monitor.detect_drift
            async def _wrapped_detect(*a, **kw):
                alerts = await orig_detect(*a, **kw)
                if alerts:
                    ML_DRIFT_EVENTS.inc(len(alerts))
                    for al in alerts:
                        try:
                            sev = getattr(al, 'severity', None)
                            dtype = getattr(al, 'drift_type', 'unknown')
                            if sev:
                                sev_val = getattr(sev, 'value', str(sev))
                            else:
                                sev_val = 'unknown'
                            ML_DRIFT_EVENTS_SEVERITY.labels(severity=sev_val, drift_type=dtype).inc()
                        except Exception:  # noqa: BLE001
                            pass
                return alerts
            drift_monitor.detect_drift = _wrapped_detect  # type: ignore
            setattr(drift_monitor, '_metrics_wrapped', True)
        _set_status("drift_monitor", "ready")
        logger.info("Drift monitor initialized", component="drift_monitor")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Drift monitor init failed: {e}")
        _set_status("drift_monitor", "failed", err=str(e))

    # Optional: Initialize Weaviate schema (additive only, safe to skip if missing)
    async def init_weaviate_schema():
        name = "weaviate_schema"
        try:
            _set_status(name, "initializing")
            if get_weaviate_client and desired_schema and fetch_current_schema and diff_schema and apply_schema_changes:
                client = await asyncio.to_thread(get_weaviate_client)
                current = await asyncio.to_thread(fetch_current_schema, client)
                target = desired_schema()
                diff = await asyncio.to_thread(diff_schema, current, target)
                if diff.get("add_classes") or diff.get("add_properties"):
                    await asyncio.to_thread(apply_schema_changes, diff, client)
            _set_status(name, "ready")
            logger.info("Weaviate schema initialized (additive)", component=name)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Weaviate schema init failed: {e}")
            _set_status(name, "failed", err=str(e))

    # Deferred component initializers
    async def init_simple(name: str, ctor):
        start = time.perf_counter()
        _set_status(name, "initializing")
        try:
            instance = ctor()
            globals()[name] = instance
            _set_status(name, "ready", duration=time.perf_counter() - start)
            logger.info("Deferred component ready", component=name)
        except Exception as e:  # noqa: BLE001
            _set_status(name, "failed", err=str(e), duration=time.perf_counter() - start)
            logger.warning(f"Deferred component {name} failed: {e}")

    async def init_ollama():
        global ollama_service, model_router
        start = time.perf_counter()
        name = "ollama"
        _set_status(name, "initializing")
        try:
            ollama_service = OllamaService()
            ok = await asyncio.wait_for(ollama_service.initialize(), timeout=120)
            if ok:
                model_router = ModelRouter()
                await model_router.check_model_availability()
                # Optional warmup: load select models into memory for faster first-use
                if os.getenv('OLLAMA_WARMUP', 'false').lower() in ('1','true','yes'):
                    # Default warmup list favors fast + strategy models; avoid forcing huge CPUs by default
                    default_models = os.getenv('OLLAMA_WARMUP_MODELS', 'solar:10.7b,phi3:14b,mixtral:8x22b')
                    warm_list = [m.strip() for m in default_models.split(',') if m.strip()]
                    asyncio.create_task(_warmup_ollama_models(model_router, warm_list))
                # Start day/night warm scheduler
                if ENABLE_WARM_SCHEDULER:
                    async def _warm_loop():
                        last_mode = None  # 'day'|'night'
                        while True:
                            try:
                                is_market = _is_market_hours()
                                mode = 'day' if is_market else 'night'
                                try:
                                    if ML_MARKET_HOURS is not None:
                                        ML_MARKET_HOURS.set(1 if is_market else 0)
                                except Exception:
                                    pass
                                if mode != last_mode:
                                    targets = DAY_HOT_MODELS if mode == 'day' else NIGHT_HEAVY_MODELS
                                    res = await _warmup_ollama_models(model_router, targets)
                                    await _record_warm_status(mode, targets, res)
                                    # Optionally warm Llama 3.1 70B after core heavy models at night
                                    if mode == 'night' and _is_llama_night_allowed() and LLAMA_NIGHT_MODEL:
                                        try:
                                            res2 = await _warmup_ollama_models(model_router, [LLAMA_NIGHT_MODEL])
                                            await _record_warm_status(mode, [LLAMA_NIGHT_MODEL], res2)
                                        except Exception:
                                            pass
                                    last_mode = mode
                                    # If we just entered night mode, trigger a one-time nightly backfill
                                    if mode == 'night':
                                        try:
                                            today = datetime.utcnow().date().isoformat()
                                            global _NIGHT_BACKFILL_DONE_DATE
                                            if _NIGHT_BACKFILL_DONE_DATE != today:
                                                try:
                                                    # Resolve watchlist
                                                    try:
                                                        syms = await _resolve_watchlist()
                                                    except Exception:
                                                        syms = SCHED_WATCHLIST
                                                    if not syms:
                                                        syms = SCHED_WATCHLIST
                                                    # Run correlation, factors, and risk sequentially
                                                    await _run_correlation_job(syms)
                                                    await _run_factors_job(syms)
                                                    await _run_risk_job(syms)
                                                    _NIGHT_BACKFILL_DONE_DATE = today
                                                    logger.info("Nightly intelligence backfill completed", symbols=len(syms))
                                                except Exception as e:
                                                    logger.warning(f"Nightly backfill failed: {e}")
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            await asyncio.sleep(300)  # check every 5 min
                    asyncio.create_task(_warm_loop())
                _set_status(name, "ready", duration=time.perf_counter() - start)
                logger.info("Ollama + router ready", component="ollama")
            else:
                _set_status(name, "failed", err="initialize() returned False", duration=time.perf_counter() - start)
                logger.warning("Ollama service not available", component="ollama")
        except Exception as e:  # noqa: BLE001
            _set_status(name, "failed", err=str(e), duration=time.perf_counter() - start)
            logger.warning(f"Ollama integration failed: {e}")

    async def init_finbert():
        global finbert_analyzer
        start = time.perf_counter()
        name = "finbert"
        _set_status(name, "initializing")
        try:
            # Create an instance of the analyzer off the main loop
            finbert_analyzer = await asyncio.to_thread(lambda: FinBERTSentimentAnalyzer())
            _set_status(name, "ready", duration=time.perf_counter() - start)
            logger.info("FinBERT analyzer initialized", component="finbert")
        except Exception as e:  # noqa: BLE001
            _set_status(name, "failed", err=str(e), duration=time.perf_counter() - start)
            logger.warning(f"FinBERT initialization failed: {e}")

    async def init_optimization():
        global improvement_engine, analytics_service, training_pipeline, backtesting_framework, training_orchestrator
        name = "optimization"
        start = time.perf_counter()
        _set_status(name, "initializing")
        try:
            improvement_engine = ContinuousImprovementEngine()
            analytics_service = PerformanceAnalyticsService()
            from model_training_pipeline import TrainingConfig
            training_config = TrainingConfig(model_type='ensemble', task_type='price_prediction')
            training_pipeline = ModelTrainingPipeline(config=training_config)
            backtesting_framework = IntelligentBacktestingFramework()
            
            # Initialize autonomous training orchestrator
            training_orchestrator = ContinuousTrainingOrchestrator()
            await training_orchestrator.initialize()
            
            # Start autonomous training in background
            asyncio.create_task(training_orchestrator.run())
            
            _set_status(name, "ready", duration=time.perf_counter() - start)
            logger.info("Optimization stack with autonomous training initialized", component="optimization")
        except Exception as e:  # noqa: BLE001
            _set_status(name, "failed", err=str(e), duration=time.perf_counter() - start)
            logger.warning(f"Optimization stack init failed: {e}")

    deferred_factories = [
        ("factor_model", lambda: AdvancedFactorService()),
        ("volatility_model", lambda: GARCHModel()),
        ("graph_network", lambda: MarketGraphNetwork()),
        ("stochastic_vol", lambda: StochasticVolatilityService()),
        ("entropy_analyzer", lambda: TransferEntropyCalculator()),
        ("timeframe_intelligence", lambda: MultiTimeframeIntelligence()),
    ]

    semaphore = asyncio.Semaphore(3)
    background_tasks: List[asyncio.Task] = []

    async def _sem_wrapper(coro):
        async with semaphore:
            await coro

    for name, ctor in deferred_factories:
        _set_status(name, "pending")
        background_tasks.append(asyncio.create_task(_sem_wrapper(init_simple(name, ctor))))

    for heavy_fn in [init_ollama, init_finbert, init_optimization, init_weaviate_schema]:
        comp = heavy_fn.__name__.replace('init_', '')
        _set_status(comp, "pending")
        background_tasks.append(asyncio.create_task(_sem_wrapper(heavy_fn())))

    app.state.component_statuses = COMPONENT_STATUSES  # type: ignore[attr-defined]
    logger.info("Minimal ML service initialization complete; deferred tasks scheduled")

    # Ensure governance metrics appear immediately
    _zero_governance_metrics()

    # Start background scheduler if enabled
    global _SCHED_TASK
    if ENABLE_ML_SCHEDULER and _SCHED_TASK is None:
        try:
            _SCHED_TASK = asyncio.create_task(_scheduler_loop())
            logger.info("ML scheduler task started")
            # Also run initial fill in parallel
            asyncio.create_task(_initial_fill())
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to start ML scheduler: {e}")

    try:
        yield
    finally:
        try:
            if ollama_service:
                await ollama_service.cleanup()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error cleaning Ollama service: {e}")
        for t in background_tasks:
            if not t.done():
                t.cancel()
        if _SCHED_TASK and not _SCHED_TASK.done():
            _SCHED_TASK.cancel()
        logger.info("Advanced ML Service stopped")


app.router.lifespan_context = lifespan  # attach lifespan after definition

# Install shared observability middleware (non-breaking; existing metrics remain)
_shared_cc = install_observability(app, service_name="ml-service")

# Metrics warm-up: issue a self-call on startup so zero-governance metrics are scraped immediately.
@app.on_event("startup")
async def _metrics_warmup():  # pragma: no cover - side-effect only
    if os.getenv('DISABLE_ML_METRICS_WARMUP','false').lower() in ('1','true','yes'):
        return
    try:
        import httpx
        url = os.getenv('ML_WARMUP_URL', 'http://localhost:8001/healthz')
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(url)
        logger.info("ML metrics warm-up request executed", url=url)
    except Exception as e:  # noqa: BLE001
        logger.warning("ML metrics warm-up failed", error=str(e))

# Ensure /metrics endpoint exists (some earlier refactors might have removed it)
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # redundancy safe
    if not any(r.path == "/metrics" for r in app.routes):
        @app.get("/metrics")  # type: ignore[misc]
        def prometheus_metrics():  # noqa: D401
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
except Exception:  # noqa: BLE001
    pass


class NewsDoc(BaseModel):
    title: str
    content: str | None = None
    source: str | None = None
    published_at: str | None = None
    url: str | None = None
    symbols: List[str] | None = None


class IndexNewsRequest(BaseModel):
    items: List[NewsDoc]


INDEX_NEWS_LATENCY = Histogram('ml_index_news_latency_seconds', 'Latency to index news batch into vector store') if not _metric_exists('ml_index_news_latency_seconds') else None  # type: ignore
INDEX_NEWS_TOTAL = Counter('ml_index_news_total', 'Count of news indexing attempts', ['status']) if not _metric_exists('ml_index_news_total') else None  # type: ignore


@app.post('/vector/index/news')
async def vector_index_news(payload: IndexNewsRequest):
    """Index cleaned news articles into Weaviate with local embeddings.

    - Embedding: sentence-transformers on title + body (content)
    - Collection: NewsArticle (created additively on startup if missing)
    - Vector provided explicitly; no server-side vectorizer required
    """
    start = time.perf_counter()
    try:
        # Prepare client and embedder
        if not get_weaviate_client:
            raise HTTPException(status_code=503, detail='Weaviate client not available')
        client = await asyncio.to_thread(get_weaviate_client)
        collection = client.collections.get('NewsArticle')
        embedder = await asyncio.to_thread(_ensure_news_embedder)

        # Build batch payloads
        to_insert = []
        vectors = []
        for item in payload.items:
            text = (item.title or '')
            if item.content:
                text = f"{text}. {item.content}"
            vec = await asyncio.to_thread(embedder.encode, text)
            vectors.append(vec.tolist() if hasattr(vec, 'tolist') else vec)
            props = {
                'title': item.title or '',
                'body': item.content or '',
                'source': item.source or '',
                'published_at': item.published_at or datetime.utcnow().isoformat(),
                'tickers': item.symbols or [],
            }
            # Deterministic UUID from URL when available
            oid = None
            if item.url:
                try:
                    oid = str(uuid.uuid5(uuid.NAMESPACE_URL, item.url))
                except Exception:
                    oid = None
            to_insert.append((props, oid))

        # Insert in batches to avoid large payloads
        inserted = 0
        batch_size = int(os.getenv('WEAVIATE_INDEX_BATCH_SIZE', '64'))
        for i in range(0, len(to_insert), batch_size):
            chunk = to_insert[i:i+batch_size]
            vecs = vectors[i:i+batch_size]
            with collection.batch.dynamic() as batch:
                for (props, oid), v in zip(chunk, vecs):
                    batch.add_object(properties=props, vector=v, uuid=oid)
                    inserted += 1

        if INDEX_NEWS_TOTAL:
            INDEX_NEWS_TOTAL.labels(status='success').inc()
        if INDEX_NEWS_LATENCY:
            INDEX_NEWS_LATENCY.observe(time.perf_counter() - start)
        return {"indexed": inserted}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        if INDEX_NEWS_TOTAL:
            INDEX_NEWS_TOTAL.labels(status='error').inc()
        ML_ENDPOINT_ERRORS.labels(endpoint='/vector/index/news').inc()
        logger.error(f"Vector news indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/vector/news/count')
async def vector_news_count():
    """Return count of NewsArticle objects in Weaviate for quick verification.

    Graceful-degrade: if Weaviate is unavailable/misconfigured, return exists=false,count=0 (HTTP 200)
    rather than surfacing a 500 to avoid cascading health check failures in upstream services.
    """
    try:
        if not get_weaviate_client:
            return {"collection": "NewsArticle", "exists": False, "count": 0, "status": "unavailable"}
        client = await asyncio.to_thread(get_weaviate_client)
        try:
            coll = client.collections.get('NewsArticle')
            collection_exists = True
        except Exception:
            return {"collection": "NewsArticle", "exists": False, "count": 0}
        count = 0
        try:
            agg = coll.aggregate.over_all(total_count=True)
            count = int(getattr(agg, 'total_count', 0) or 0)
        except Exception:
            try:
                it = coll.query.fetch_objects(limit=1)
                count = int(getattr(it, 'total_count', 0) or (1 if getattr(it, 'objects', []) else 0))
            except Exception:
                count = 0
        return {"collection": "NewsArticle", "exists": collection_exists, "count": count}
    except Exception as e:
        try:
            ML_ENDPOINT_ERRORS.labels(endpoint='/vector/news/count').inc()
        except Exception:
            pass
        logger.warning(f"Vector news count degraded: {e}")
        return {"collection": "NewsArticle", "exists": False, "count": 0, "status": "degraded"}


@app.get('/vector/social/count')
async def vector_social_count():
    """Return count of SocialSentiment objects in Weaviate for quick verification (graceful-degrade)."""
    try:
        if not get_weaviate_client:
            return {"collection": "SocialSentiment", "exists": False, "count": 0, "status": "unavailable"}
        client = await asyncio.to_thread(get_weaviate_client)
        try:
            coll = client.collections.get('SocialSentiment')
            collection_exists = True
        except Exception:
            return {"collection": "SocialSentiment", "exists": False, "count": 0}
        count = 0
        try:
            agg = coll.aggregate.over_all(total_count=True)
            count = int(getattr(agg, 'total_count', 0) or 0)
        except Exception:
            try:
                it = coll.query.fetch_objects(limit=1)
                count = int(getattr(it, 'total_count', 0) or (1 if getattr(it, 'objects', []) else 0))
            except Exception:
                count = 0
        return {"collection": "SocialSentiment", "exists": collection_exists, "count": count}
    except Exception as e:
        try:
            ML_ENDPOINT_ERRORS.labels(endpoint='/vector/social/count').inc()
        except Exception:
            pass
        logger.warning(f"Vector social count degraded: {e}")
        return {"collection": "SocialSentiment", "exists": False, "count": 0, "status": "degraded"}


@app.get('/vector/equity/count')
async def vector_equity_count():
    """Return count of EquityBar objects in Weaviate for verification (graceful-degrade)."""
    try:
        if not get_weaviate_client:
            return {"collection": "EquityBar", "exists": False, "count": 0, "status": "unavailable"}
        client = await asyncio.to_thread(get_weaviate_client)
        try:
            coll = client.collections.get('EquityBar')
            collection_exists = True
        except Exception:
            return {"collection": "EquityBar", "exists": False, "count": 0}
        count = 0
        try:
            agg = coll.aggregate.over_all(total_count=True)
            count = int(getattr(agg, 'total_count', 0) or 0)
        except Exception:
            try:
                it = coll.query.fetch_objects(limit=1)
                count = int(getattr(it, 'total_count', 0) or (1 if getattr(it, 'objects', []) else 0))
            except Exception:
                count = 0
        return {"collection": "EquityBar", "exists": collection_exists, "count": count}
    except Exception as e:  # noqa: BLE001
        try:
            ML_ENDPOINT_ERRORS.labels(endpoint='/vector/equity/count').inc()
        except Exception:
            pass
        logger.warning(f"Vector equity count degraded: {e}")
        return {"collection": "EquityBar", "exists": False, "count": 0, "status": "degraded"}


@app.get('/vector/options/count')
async def vector_options_count():
    """Return count of OptionContract objects in Weaviate for verification (graceful-degrade)."""
    try:
        if not get_weaviate_client:
            return {"collection": "OptionContract", "exists": False, "count": 0, "status": "unavailable"}
        client = await asyncio.to_thread(get_weaviate_client)
        try:
            coll = client.collections.get('OptionContract')
            collection_exists = True
        except Exception:
            return {"collection": "OptionContract", "exists": False, "count": 0}
        count = 0
        try:
            agg = coll.aggregate.over_all(total_count=True)
            count = int(getattr(agg, 'total_count', 0) or 0)
        except Exception:
            try:
                it = coll.query.fetch_objects(limit=1)
                count = int(getattr(it, 'total_count', 0) or (1 if getattr(it, 'objects', []) else 0))
            except Exception:
                count = 0
        return {"collection": "OptionContract", "exists": collection_exists, "count": count}
    except Exception as e:  # noqa: BLE001
        try:
            ML_ENDPOINT_ERRORS.labels(endpoint='/vector/options/count').inc()
        except Exception:
            pass
        logger.warning(f"Vector options count degraded: {e}")
        return {"collection": "OptionContract", "exists": False, "count": 0, "status": "degraded"}


    # --------------------------- Training job APIs --------------------------- #
    from pydantic import BaseModel as _PydBaseModel
    class TrainingJobRequest(_PydBaseModel):
        model_name: str
        symbols: List[str]
        model_type: Optional[str] = None
        task_type: Optional[str] = None
        start_date: Optional[str] = None  # YYYY-MM-DD
        end_date: Optional[str] = None    # YYYY-MM-DD
        horizon_days: Optional[int] = 1
        features: Optional[List[str]] = None

    class TrainingJobStatus(_PydBaseModel):
        job_id: str
        status: str
        started_at: Optional[str] = None
        finished_at: Optional[str] = None
        error: Optional[str] = None
        metrics: Optional[dict] = None

    _TRAIN_JOBS: dict[str, TrainingJobStatus] = {}

    @app.post('/train/run')
    async def train_run(req: TrainingJobRequest):
        """Start an asynchronous training job over QuestDB data using the pipeline.

        Returns a job_id for polling via /train/status/{job_id}.
        """
        try:
            if not req.model_name or not req.symbols:
                raise HTTPException(status_code=400, detail='model_name and symbols are required')
            job_id = str(uuid.uuid4())
            _TRAIN_JOBS[job_id] = TrainingJobStatus(job_id=job_id, status='queued')

            async def _worker():
                _TRAIN_JOBS[job_id].status = 'running'
                _TRAIN_JOBS[job_id].started_at = datetime.utcnow().isoformat() + 'Z'
                try:
                    from model_training_pipeline import get_training_pipeline
                    pipe = await get_training_pipeline()
                    job = {
                        'model_name': req.model_name,
                        'symbols': [s.strip().upper() for s in req.symbols if s and s.strip()],
                        'model_type': req.model_type,
                        'task_type': req.task_type or 'price_prediction',
                        'start_date': req.start_date,
                        'end_date': req.end_date,
                        'horizon_days': req.horizon_days or 1,
                        'features': req.features,
                    }
                    res = await pipe.run_training_job(job)
                    _TRAIN_JOBS[job_id].status = 'completed'
                    _TRAIN_JOBS[job_id].finished_at = datetime.utcnow().isoformat() + 'Z'
                    # Store a summary of metrics only to keep memory small
                    _TRAIN_JOBS[job_id].metrics = res.get('metrics') if isinstance(res, dict) else None
                except Exception as e:  # noqa: BLE001
                    _TRAIN_JOBS[job_id].status = 'failed'
                    _TRAIN_JOBS[job_id].finished_at = datetime.utcnow().isoformat() + 'Z'
                    _TRAIN_JOBS[job_id].error = str(e)

            asyncio.create_task(_worker())
            return {"status": "started", "job_id": job_id}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/train/status/{job_id}')
    async def train_status(job_id: str):
        st = _TRAIN_JOBS.get(job_id)
        if not st:
            raise HTTPException(status_code=404, detail='job_not_found')
        return st.dict()


# --------------------------- Vector reconcile from QuestDB --------------------------- #
@app.post('/vector/reconcile/news')
async def vector_reconcile_news(days: int = 7, limit: int = 2000):
    """Pull recent news from QuestDB and index into Weaviate NewsArticle.

    - Uses sentence-transformers embeddings on title+content
    - De-duplicates by URL (deterministic UUID) when available
    """
    try:
        # Discover available columns to adapt to schema variations
        look_days = max(1, int(days))
        max_rows = max(1, int(limit))
        cols_info = []
        try:
            cols_info = await _qdb_http_query("show columns from news_items")
        except Exception:
            cols_info = []
        have = {str(r.get('column')).lower(): True for r in (cols_info or []) if r.get('column')}
        # Required
        select_cols: list[str] = ["title", "ts"]
        # Optional
        if have.get('source'): select_cols.append('source')
        if have.get('url'): select_cols.append('url')
        if have.get('symbol'): select_cols.append('symbol')
        if have.get('content'): select_cols.insert(1, 'content')  # keep near title when present
        select_list = ", ".join(select_cols)
        rows = await _qdb_http_query(
            (
                "SELECT " + select_list + " FROM news_items "
                f"WHERE ts >= dateadd('d', -{look_days}, now()) "
                "AND title IS NOT NULL AND length(title) > 0 "
                "ORDER BY ts DESC "
                f"LIMIT {max_rows}"
            )
        )
        if not rows:
            return {"indexed": 0, "items": 0}
        # Prepare payload
        items: List[NewsDoc] = []
        for r in rows:
            try:
                title = (r.get('title') or '').strip()
                if not title:
                    continue
                content = (r.get('content') or '') if 'content' in r else ''
                src = (r.get('source') or '') if 'source' in r else ''
                ts = r.get('ts') or None
                url = (r.get('url') or None) if 'url' in r else None
                sym = ((r.get('symbol') or '').strip().upper()) if 'symbol' in r else ''
                syms = [sym] if sym else []
                items.append(NewsDoc(title=title, content=content, source=src, published_at=str(ts) if ts else None, url=url, symbols=syms))
            except Exception:
                continue
        # Try embedding path first; if embedding model unavailable, fall back to direct insert (no vectors)
        inserted = 0
        embed_available = SentenceTransformer is not None
        if items and embed_available:
            try:
                inserted = (await vector_index_news(IndexNewsRequest(items=items))).get('indexed', 0)  # type: ignore[attr-defined]
            except Exception:
                inserted = 0
        if items and (not embed_available or inserted == 0):
            # Direct insert without vectors (depends on Weaviate class vectorizer config)
            if not get_weaviate_client:
                return {"status": "degraded", "items": len(items), "indexed": 0}
            client = await asyncio.to_thread(get_weaviate_client)
            coll = client.collections.get('NewsArticle')
            # Insert in small batches
            batch_size = int(os.getenv('WEAVIATE_INDEX_BATCH_SIZE', '64'))
            for i in range(0, len(items), batch_size):
                chunk = items[i:i+batch_size]
                with coll.batch.dynamic() as batch:
                    for it in chunk:
                        oid = None
                        if it.url:
                            try:
                                oid = str(uuid.uuid5(uuid.NAMESPACE_URL, it.url))  # type: ignore[name-defined]
                            except Exception:
                                oid = None
                        batch.add_object(properties={
                            'title': it.title or '',
                            'body': it.content or '',
                            'source': it.source or '',
                            'published_at': it.published_at or datetime.utcnow().isoformat(),
                            'tickers': it.symbols or [],
                        }, uuid=oid)
                        inserted += 1
        return {"status": "ok", "items": len(items), "indexed": int(inserted)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/vector/reconcile/equity')
async def vector_reconcile_equity(days: int = 30, limit: int = 10000):
    """Pull recent equity bars from QuestDB and index into Weaviate EquityBar."""
    try:
        look_days = max(1, int(days))
        max_rows = max(1, int(limit))
        rows = await _qdb_http_query(
            """
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM market_data
            WHERE timestamp >= dateadd('d', -{days}, now())
            ORDER BY timestamp DESC
            LIMIT {limit}
            """.format(days=look_days, limit=max_rows)
        )
        if not rows:
            return {"indexed": 0, "items": 0}
        bars: List[EquityBarVector] = []
        for r in rows:
            try:
                bars.append(EquityBarVector(
                    symbol=(r.get('symbol') or '').strip().upper(),
                    timestamp=str(r.get('timestamp') or ''),
                    open=float(r['open']) if r.get('open') is not None else None,
                    high=float(r['high']) if r.get('high') is not None else None,
                    low=float(r['low']) if r.get('low') is not None else None,
                    close=float(r['close']) if r.get('close') is not None else None,
                    volume=float(r['volume']) if r.get('volume') is not None else None,
                ))
            except Exception:
                continue
        # Index using existing batch endpoint logic
        result = await vector_index_equity_batch(EquityBarBatch(bars=bars))  # type: ignore[arg-type]
        # result is a dict with indexed/errors
        return {"status": "ok", **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------- Scheduler Read APIs --------------------------- #
@app.get('/intelligence/latest/correlation')
async def latest_correlation():
    data = await _cache_get_json(_redis_key('correlation','latest'))
    if not data:
        raise HTTPException(status_code=404, detail='no_correlation_cached')
    return data

@app.get('/intelligence/latest/factors')
async def latest_factors():
    data = await _cache_get_json(_redis_key('factors','latest'))
    if not data:
        raise HTTPException(status_code=404, detail='no_factors_cached')
    return data

@app.get('/intelligence/latest/risk')
async def latest_risk():
    data = await _cache_get_json(_redis_key('risk','latest'))
    if not data:
        raise HTTPException(status_code=404, detail='no_risk_cached')
    return data

@app.get('/intelligence/latest/risk/{symbol}')
async def latest_risk_for_symbol(symbol: str):
    sym = (symbol or '').strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail='invalid_symbol')
    data = await _cache_get_json(_redis_key('risk', sym, 'latest'))
    if not data:
        raise HTTPException(status_code=404, detail='no_risk_cached_for_symbol')
    return data

class SymbolsPayload(BaseModel):
    symbols: Optional[List[str]] = None

@app.post('/intelligence/run/correlation')
async def run_correlation_job(payload: Optional[SymbolsPayload] = None):
    # Prefer dynamic watchlist when configured; fallback to env list
    if not payload or not payload.symbols:
        try:
            syms = await _resolve_watchlist()
            if not syms:
                syms = SCHED_WATCHLIST
        except Exception:
            syms = SCHED_WATCHLIST
    else:
        syms = [s.strip().upper() for s in payload.symbols if s and s.strip()]
    res = await _run_correlation_job(syms)
    if not res:
        raise HTTPException(status_code=500, detail='correlation_job_failed')
    return res

@app.post('/intelligence/run/factors')
async def run_factors_job(payload: Optional[SymbolsPayload] = None):
    if not payload or not payload.symbols:
        try:
            syms = await _resolve_watchlist()
            if not syms:
                syms = SCHED_WATCHLIST
        except Exception:
            syms = SCHED_WATCHLIST
    else:
        syms = [s.strip().upper() for s in payload.symbols if s and s.strip()]
    res = await _run_factors_job(syms)
    if not res:
        raise HTTPException(status_code=500, detail='factors_job_failed')
    return res

@app.post('/intelligence/run/risk')
async def run_risk_job(payload: Optional[SymbolsPayload] = None):
    if not payload or not payload.symbols:
        try:
            syms = await _resolve_watchlist()
            if not syms:
                syms = SCHED_WATCHLIST
        except Exception:
            syms = SCHED_WATCHLIST
    else:
        syms = [s.strip().upper() for s in payload.symbols if s and s.strip()]
    res = await _run_risk_job(syms)
    if not res:
        raise HTTPException(status_code=500, detail='risk_job_failed')
    return res

@app.get('/intelligence/last-run')
async def intelligence_last_run_status():
    """Return last-run metadata for scheduler jobs and current scheduler config."""
    out: dict[str, Any] = {
        'config': {
            'enabled': ENABLE_ML_SCHEDULER,
            'watchlist': SCHED_WATCHLIST,
            'intervals_seconds': {
                'correlation': CORRELATION_INTERVAL,
                'factors': FACTORS_INTERVAL,
                'risk': RISK_INTERVAL,
            },
            'redis_prefix': REDIS_PREFIX,
            'timeframe': DEFAULT_TIMEFRAME,
        },
        'now': datetime.utcnow().isoformat() + 'Z'
    }
    for job, key in _SCHED_LAST_KEYS.items():
        try:
            wrapped = await _cache_get_json(_redis_key(key))
            out[job] = (wrapped or {}).get('data') if isinstance(wrapped, dict) else None
        except Exception:
            out[job] = None
    return out

@app.get('/intelligence/overview')
async def intelligence_overview():
    """Provide a compact market overview derived from latest cached intelligence results.

    Includes:
      - correlation: top pair by absolute value and average absolute correlation
      - risk: portfolio summary, top symbols by Sharpe, worst max drawdown
      - factors: best-effort signals summary if structure available
    """
    overview: dict[str, Any] = {'timestamp': datetime.utcnow().isoformat()}
    # Correlation summary
    try:
        corr_wrapped = await _cache_get_json(_redis_key('correlation','latest'))
        corr = (corr_wrapped or {}).get('data') if isinstance(corr_wrapped, dict) else None
        if corr and isinstance(corr.get('matrix'), list) and isinstance(corr.get('symbols'), list):
            syms = corr['symbols']
            mat = corr['matrix']
            top_val = 0.0
            top_pair = None
            abs_vals = []
            for i in range(len(syms)):
                for j in range(i+1, len(syms)):
                    try:
                        v = float(mat[i][j])
                        abs_vals.append(abs(v))
                        if abs(v) > abs(top_val):
                            top_val = v
                            top_pair = (syms[i], syms[j])
                    except Exception:
                        continue
            overview['correlation'] = {
                'top_pair': {'symbols': top_pair, 'value': top_val} if top_pair else None,
                'avg_abs_corr': (sum(abs_vals)/len(abs_vals)) if abs_vals else None,
                'data_availability': corr.get('data_availability')
            }
    except Exception:
        pass
    # Risk summary
    try:
        risk_wrapped = await _cache_get_json(_redis_key('risk','latest'))
        risk = (risk_wrapped or {}).get('data') if isinstance(risk_wrapped, dict) else None
        if risk and isinstance(risk.get('risk'), dict):
            rmap = risk['risk']
            sharpes = []
            mdds = []
            for sym, met in rmap.items():
                try:
                    if met.get('sharpe_ratio') is not None:
                        sharpes.append((sym, float(met['sharpe_ratio'])))
                except Exception:
                    pass
                try:
                    if met.get('max_drawdown') is not None:
                        mdds.append((sym, float(met['max_drawdown'])))
                except Exception:
                    pass
            sharpes.sort(key=lambda x: x[1], reverse=True)
            mdds.sort(key=lambda x: x[1], reverse=True)
            overview['risk'] = {
                'portfolio': risk.get('portfolio'),
                'top_by_sharpe': sharpes[:5] if sharpes else None,
                'worst_by_mdd': mdds[:5] if mdds else None,
            }
    except Exception:
        pass
    # Factors summary (best-effort)
    try:
        factors_wrapped = await _cache_get_json(_redis_key('factors','latest'))
        factors = (factors_wrapped or {}).get('data') if isinstance(factors_wrapped, dict) else None
        if factors and isinstance(factors.get('signals'), (dict, list)):
            summary: dict[str, Any] = {}
            sig = factors['signals']
            # attempt to count signals if dict[symbol]->dict[factor]->value format
            if isinstance(sig, dict):
                factor_counts: dict[str, int] = {}
                for _sym, fmap in sig.items():
                    if isinstance(fmap, dict):
                        for fname, fval in fmap.items():
                            try:
                                if isinstance(fval, (int, float)) and abs(float(fval)) > 0:
                                    factor_counts[fname] = factor_counts.get(fname, 0) + 1
                            except Exception:
                                continue
                if factor_counts:
                    summary['factor_counts_nonzero'] = factor_counts
            overview['factors'] = summary or None
    except Exception:
        pass
    return overview

# --------------------------- Watchlist Admin APIs --------------------------- #
@app.get('/intelligence/watchlist')
async def get_watchlist():
    """Return configured and resolved watchlist details."""
    try:
        resolved = await _resolve_watchlist()
        return {
            'source': WATCHLIST_SOURCE,
            'env_list': SCHED_WATCHLIST,
            'resolved': resolved,
            'counts': {
                'env': len(SCHED_WATCHLIST),
                'resolved': len(resolved or [])
            },
            'lookback_days': WATCHLIST_LOOKBACK_DAYS,
            'max_symbols': WATCHLIST_MAX_SYMBOLS,
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

class WatchlistRefreshPayload(BaseModel):
    force_source: Optional[str] = None  # 'env' | 'questdb'

@app.post('/intelligence/watchlist/refresh')
async def refresh_watchlist(payload: Optional[WatchlistRefreshPayload] = None):
    """Force a one-off watchlist resolution and return it (no persistent changes)."""
    try:
        src = (payload.force_source if payload else None)
        if src:
            src = src.lower().strip()
        # Temporarily override resolution when caller forces a source
        if src in ('questdb','db'):
            try:
                lst = await _fetch_watchlist_from_questdb()
                if lst:
                    return {'source': 'questdb', 'resolved': lst, 'count': len(lst)}
            except Exception:
                pass
        elif src == 'env':
            return {'source': 'env', 'resolved': SCHED_WATCHLIST, 'count': len(SCHED_WATCHLIST)}
        # Default behavior
        resolved = await _resolve_watchlist()
        return {'source': WATCHLIST_SOURCE, 'resolved': resolved, 'count': len(resolved or [])}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------- FinBERT Sentiment APIs --------------------------- #
class SentimentTextPayload(BaseModel):
    text: str

class SentimentBatchPayload(BaseModel):
    texts: List[str]
    batch_size: int | None = None

@app.get('/sentiment/finbert/health')
async def finbert_health():
    ok = finbert_analyzer is not None
    return {"status": "ok" if ok else "degraded", "initialized": ok}

@app.post('/sentiment/finbert/analyze')
async def finbert_analyze(payload: SentimentTextPayload):
    if not payload or not (payload.text and payload.text.strip()):
        raise HTTPException(status_code=400, detail='empty_text')
    if finbert_analyzer is None:
        raise HTTPException(status_code=503, detail='finbert_unavailable')
    try:
        # Run sync analysis off the loop
        res = await asyncio.to_thread(finbert_analyzer.analyze_sentiment, payload.text)
        return {
            'text': res.text,
            'sentiment': res.sentiment,
            'confidence': res.confidence,
            'scores': res.scores,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/sentiment/finbert/batch')
async def finbert_batch(payload: SentimentBatchPayload):
    if not payload or not isinstance(payload.texts, list) or not payload.texts:
        raise HTTPException(status_code=400, detail='empty_texts')
    if finbert_analyzer is None:
        raise HTTPException(status_code=503, detail='finbert_unavailable')
    try:
        bs = int(payload.batch_size) if payload.batch_size else 32
        results = await asyncio.to_thread(finbert_analyzer.batch_analyze, payload.texts, bs)
        out = []
        for r in results:
            out.append({
                'text': r.text,
                'sentiment': r.sentiment,
                'confidence': r.confidence,
                'scores': r.scores,
            })
        return {'count': len(out), 'results': out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/intelligence/backfill/all')
async def backfill_all_intelligence():
    """Run correlation, factors, and risk computations over the resolved watchlist now.
    Returns a compact summary with counts and durations.
    """
    try:
        try:
            syms = await _resolve_watchlist()
            if not syms:
                syms = SCHED_WATCHLIST
        except Exception:
            syms = SCHED_WATCHLIST
        t0 = time.time()
        corr = await _run_correlation_job(syms)
        fac = await _run_factors_job(syms)
        risk = await _run_risk_job(syms)
        return {
            'status': 'ok',
            'symbols': len(syms),
            'jobs': {
                'correlation': bool(corr),
                'factors': bool(fac),
                'risk': bool(risk),
            },
            'duration_seconds': round(time.time() - t0, 3)
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/vector/schema/init')
async def vector_schema_init(force: bool = False):
    """Explicitly (re)initialize Weaviate schema in an additive manner.

    Behavior:
      - Fetch current schema and desired schema definition.
      - Compute diff (add_classes / add_properties) via helper.
      - Apply only additive changes unless `force` specified (force currently
        still additive; placeholder for future destructive reconciliation).
      - Returns diff summary and status flags so orchestration can gate
        downstream vector indexing.

    Returns 503 if Weaviate integration utilities not available.
    """
    start = time.perf_counter()
    if not (get_weaviate_client and desired_schema and fetch_current_schema and diff_schema and apply_schema_changes):
        raise HTTPException(status_code=503, detail='Weaviate schema utilities not available')
    try:
        client = await asyncio.to_thread(get_weaviate_client)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f'Weaviate client init failed: {e}')
    try:
        current = await asyncio.to_thread(fetch_current_schema, client)
    except Exception as e:  # noqa: BLE001
        # Surface authorization separately for operator clarity
        code = 'unauthorized' if '401' in str(e) or 'Unauthorized' in str(e) else 'fetch_error'
        raise HTTPException(status_code=401 if code == 'unauthorized' else 502, detail=f'schema_fetch:{code}:{e}')
    try:
        target = desired_schema()
        diff = await asyncio.to_thread(diff_schema, current, target)
        applied = False
        if diff.get('add_classes') or diff.get('add_properties'):
            try:
                await asyncio.to_thread(apply_schema_changes, diff, client)
                applied = True
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f'schema_apply_failed:{e}')
        duration = time.perf_counter() - start
        return {
            'status': 'ok',
            'applied': applied,
            'force': force,
            'diff': {k: v for k, v in diff.items() if v},
            'duration_seconds': round(duration, 4)
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f'schema_init_unexpected:{e}')


@app.get("/health")
async def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "ml-service",
        "timestamp": datetime.utcnow().isoformat(),
        "connections": {
            "cache": cache_client is not None,
            "redis": redis_client is not None,
            "llm": llm_service is not None
        }
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "ml-service", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
async def ready():
    """Unified readiness probe.
    Criteria:
        - Cache & Redis clients established
        - Intelligence or ensemble coordinator present
        - NLP capability (Ollama+router or FinBERT)
    Returns 200 when all criteria met; 503 otherwise.
    """
    critical_infra = (cache_client is not None and redis_client is not None)
    intelligence_ready = (intelligence_coordinator is not None or ensemble_coordinator is not None)
    # Do not block readiness on heavy deferred components (ollama / finbert / optimization)
    llm_stack_ready = (llm_service is not None)
    degraded_reasons = []
    if not cache_client:
        degraded_reasons.append('cache_unavailable')
    if not redis_client:
        degraded_reasons.append('redis_unavailable')
    if not (intelligence_coordinator or ensemble_coordinator):
        degraded_reasons.append('intelligence_pipeline_unready')
    if not llm_service:
        degraded_reasons.append('llm_service_uninitialized')
    ready_flag = not degraded_reasons
    status_code = 200 if ready_flag else 503
    return JSONResponse({
        "service": "ml-service",
        "status": "ready" if ready_flag else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "cache": cache_client is not None,
            "redis": redis_client is not None,
            "intelligence": intelligence_coordinator is not None,
            "ensemble": ensemble_coordinator is not None,
            "ollama": ollama_service is not None,
            "model_router": model_router is not None,
            "finbert": finbert_analyzer is not None
        },
        "degraded_reasons": degraded_reasons or None
    }, status_code=status_code)


@app.get("/status")
async def get_status():
    """Get comprehensive service status."""
    try:
        status = {
            "service": "ml-service",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "initialization": getattr(app.state, 'component_statuses', {}),  # type: ignore[attr-defined]
            "models": {
                "llm": llm_service.get_model_status() if llm_service else None,
                "regime_detector": regime_detector.get_status() if regime_detector else None,
                "factor_model": {"initialized": factor_model is not None},
                "volatility_model": {"initialized": volatility_model is not None},
                "graph_network": {"initialized": graph_network is not None}
            },
            "intelligence_systems": {
                "advanced_coordinator": {"active": intelligence_coordinator is not None},
                "ensemble_coordinator": {"active": ensemble_coordinator is not None},
                "timeframe_intelligence": {"active": timeframe_intelligence is not None}
            },
            "monitoring": {
                "drift_monitor": drift_monitor.get_status() if drift_monitor else None,
                "analytics": {"active": analytics_service is not None},
                "improvement_engine": {"active": improvement_engine is not None}
            }
        }
        return status
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service status")


@app.get("/startup/status")
async def startup_status():
    """Detailed component initialization status with summary counts."""
    statuses = getattr(app.state, 'component_statuses', {})  # type: ignore[attr-defined]
    summary = {"pending": 0, "initializing": 0, "ready": 0, "failed": 0}
    for rec in statuses.values():
        st = rec.get("status", "pending")
        if st in summary:
            summary[st] += 1
    return {
        "service": "ml-service",
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary,
        "components": statuses
    }


@app.post("/analyze/market")
async def analyze_market(
    symbols: List[str],
    analysis_types: Optional[List[str]] = None,
    timeframe: str = "1h",
    lookback_periods: int = 100
):
    """Perform comprehensive market analysis."""
    try:
        if not analysis_types:
            analysis_types = ["technical", "sentiment", "regime", "risk"]
        
        results = {}
        
        # Market regime detection
        if "regime" in analysis_types and regime_detector:
            # The detector exposes get_current_regime(), returning RegimeCharacteristics
            rc = await _infer("regime_detector", regime_detector.get_current_regime())
            try:
                # Map volatility_level (0..1) to a coarse volatility_state label
                vol_state = (
                    "high" if getattr(rc, "volatility_level", 0.0) > 0.4
                    else "low" if getattr(rc, "volatility_level", 0.0) < 0.2
                    else "normal"
                )
                current_regime = getattr(rc, "regime", None)
                current_label = getattr(current_regime, "value", None) or str(current_regime)
                results["regime"] = {
                    "current": current_label,
                    "confidence": getattr(rc, "probability", None),
                    "volatility_state": vol_state,
                    "trend_strength": getattr(rc, "trend_strength", None)
                }
            except Exception:
                # Fallback to minimal regime info if unexpected structure
                results["regime"] = {"current": None}
        
        # Technical analysis via LLM
        if "technical" in analysis_types and llm_service:
            request = AnalysisRequest(
                type=AnalysisType.TECHNICAL,
                data={
                    "symbols": symbols,
                    "timeframe": timeframe,
                    "lookback": lookback_periods
                }
            )
            primary_model_label = "llm_technical"
            llm_result = await _infer(primary_model_label, llm_service.analyze(request))
            results["technical"] = {
                "analysis": llm_result.analysis,
                "confidence": llm_result.confidence,
                "recommendations": llm_result.recommendations
            }
            # Canary (non-blocking or inline) logic
            if CANARY_ENABLED and CANARY_MODEL and hasattr(llm_service, 'analyze'):
                async def _canary_flow():
                    c_status = 'success'
                    start_primary_len = len(llm_result.analysis.split()) if getattr(llm_result, 'analysis', None) else 0
                    t0 = time.perf_counter()
                    try:
                        # Duplicate request (could adjust urgency if needed)
                        c_req = AnalysisRequest(type=AnalysisType.TECHNICAL, data=request.data, context=request.context, urgency=request.urgency)
                        canary_raw = await llm_service.analyze(c_req)  # reuse same interface; assume routing respects CANARY_MODEL if env influences internal selection later
                        # Simple divergence score: 1 - token overlap ratio
                        overlap = 0.0
                        try:
                            primary_tokens = set(llm_result.analysis.lower().split())
                            canary_tokens = set(getattr(canary_raw, 'analysis', '').lower().split())
                            if primary_tokens and canary_tokens:
                                overlap_ratio = len(primary_tokens & canary_tokens) / max(len(primary_tokens | canary_tokens), 1)
                                overlap = 1 - overlap_ratio
                        except Exception:
                            pass
                        ML_CANARY_DIVERGENCE.observe(overlap)
                        ML_CANARY_LATENCY_DELTA.observe(time.perf_counter() - t0)
                        ML_CANARY_REQUESTS.labels(primary=primary_model_label, canary=CANARY_MODEL, status='success').inc()
                    except Exception:
                        ML_CANARY_REQUESTS.labels(primary=primary_model_label, canary=CANARY_MODEL, status='error').inc()
                asyncio.create_task(_canary_flow())
        
        # Risk assessment
        if "risk" in analysis_types:
            risk_metrics = await _infer("risk_metrics", calculate_risk_metrics(symbols, timeframe))
            results["risk"] = risk_metrics
        
        # Sentiment analysis
        if "sentiment" in analysis_types and llm_service:
            sentiment_request = AnalysisRequest(
                type=AnalysisType.SENTIMENT,
                data={"symbols": symbols}
            )
            primary_model_label = "llm_sentiment"
            sentiment_result = await _infer(primary_model_label, llm_service.analyze(sentiment_request))
            results["sentiment"] = {
                "score": sentiment_result.confidence,
                "analysis": sentiment_result.analysis
            }
            if CANARY_ENABLED and CANARY_MODEL and hasattr(llm_service, 'analyze'):
                async def _canary_flow_sent():
                    try:
                        c_req = AnalysisRequest(type=AnalysisType.SENTIMENT, data=sentiment_request.data, context=sentiment_request.context, urgency=sentiment_request.urgency)
                        canary_raw = await llm_service.analyze(c_req)
                        overlap = 0.0
                        try:
                            primary_tokens = set(sentiment_result.analysis.lower().split())
                            canary_tokens = set(getattr(canary_raw, 'analysis', '').lower().split())
                            if primary_tokens and canary_tokens:
                                overlap_ratio = len(primary_tokens & canary_tokens) / max(len(primary_tokens | canary_tokens), 1)
                                overlap = 1 - overlap_ratio
                        except Exception:
                            pass
                        ML_CANARY_DIVERGENCE.observe(overlap)
                        ML_CANARY_REQUESTS.labels(primary=primary_model_label, canary=CANARY_MODEL, status='success').inc()
                    except Exception:
                        ML_CANARY_REQUESTS.labels(primary=primary_model_label, canary=CANARY_MODEL, status='error').inc()
                asyncio.create_task(_canary_flow_sent())
        
        return {
            "symbols": symbols,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": results
        }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/analyze/market").inc()
        logger.error(f"Market analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/volatility")
async def predict_volatility(
    symbol: str,
    horizon: int = 5,
    method: str = "garch_lstm"
):
    """Predict future volatility."""
    try:
        if method == "garch_lstm" and volatility_model:
            prediction = await _infer("garch_lstm", volatility_model.predict(symbol, horizon))
            return {
                "symbol": symbol,
                "method": method,
                "horizon": horizon,
                "predictions": prediction.tolist(),
                "timestamp": datetime.utcnow().isoformat()
            }
        elif method == "stochastic" and stochastic_vol:
            prediction = await _infer("stochastic_vol", stochastic_vol.predict(symbol, horizon))
            return {
                "symbol": symbol,
                "method": method,
                "horizon": horizon,
                "predictions": prediction.tolist(),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
            
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/predict/volatility").inc()
        logger.error(f"Volatility prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/factors")
async def analyze_factors(
    symbols: List[str],
    factors: Optional[List[str]] = None
):
    """Analyze factor exposures and risks."""
    try:
        if not factor_model:
            raise HTTPException(status_code=503, detail="Factor model not available")
        # Build recent daily OHLCV from QuestDB as MarketData for model update
        market_map: Dict[str, List[MarketData]] = {s: [] for s in symbols}
        source_tag = 'pg'
        time_filter = 'exact_1600'
        resampled = False
        try:
            dbm = await get_database_manager()
            async with dbm.get_questdb() as conn:
                # Query daily bars table which has proper daily OHLCV data
                sql = (
                    "SELECT symbol, timestamp, open, high, low, close, volume FROM daily_bars "
                    "WHERE symbol = ANY($1::text[]) "
                    "ORDER BY symbol ASC, timestamp ASC"
                )
                rows = await conn.fetch(sql, symbols)
            for r in rows:
                sym = str(r["symbol"]) if isinstance(r, dict) else str(r[0])
                md = MarketData(
                    symbol=sym,
                    timestamp=(r["timestamp"] if isinstance(r, dict) else r[1]),
                    open=float(r["open"] if isinstance(r, dict) else r[2]),
                    high=float(r["high"] if isinstance(r, dict) else r[3]),
                    low=float(r["low"] if isinstance(r, dict) else r[4]),
                    close=float(r["close"] if isinstance(r, dict) else r[5]),
                    volume=float(r["volume"] if isinstance(r, dict) else r[6]),
                )
                market_map.setdefault(sym, []).append(md)
        except Exception as e:
            # Fallback via HTTP console API (supports to_char) if PG-wire fails
            try:
                in_list = ",".join(["'" + s.replace("'","''") + "'" for s in symbols])
                # Query daily bars table which has proper daily OHLCV data
                sql = (
                    "SELECT symbol, timestamp, open, high, low, close, volume FROM daily_bars "
                    f"WHERE symbol IN ({in_list}) "
                    "ORDER BY symbol ASC, timestamp ASC"
                )
                rows = await _qdb_http_query(sql)
                for r in rows:
                    sym = str(r.get("symbol"))
                    md = MarketData(
                        symbol=sym,
                        timestamp=pd.to_datetime(r.get("timestamp")),
                        open=float(r.get("open")),
                        high=float(r.get("high")),
                        low=float(r.get("low")),
                        close=float(r.get("close")),
                        volume=float(r.get("volume")),
                    )
                    market_map.setdefault(sym, []).append(md)
            except Exception as ee:
                raise HTTPException(status_code=502, detail=f"market_data_fetch_failed:{ee}")

        update_summary = await _infer("factor_model_update", factor_model.update_factor_models(symbols, market_map, {}))
        signals = await _infer("factor_model_signals", factor_model.get_factor_signals(symbols))

        return {
            "symbols": symbols,
            "requested_factors": factors or ["momentum", "value", "quality"],
            "signals": signals,
            "update_summary": update_summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/analyze/factors").inc()
        logger.error(f"Factor analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/network")
async def analyze_market_network(
    symbols: List[str],
    analysis_type: str = "correlation"
):
    """Analyze market network relationships."""
    try:
        if analysis_type == "correlation":
            # Prefer graph implementation if available
            if graph_network and hasattr(graph_network, 'compute_correlations'):
                correlations = await _infer("graph_correlation", graph_network.compute_correlations(symbols))
                try:
                    matrix = correlations.tolist()
                except Exception:
                    matrix = correlations
                return {
                    "type": "correlation",
                    "symbols": symbols,
                    "matrix": matrix,
                    "timestamp": datetime.utcnow().isoformat()
                }
            # Fallback: compute from QuestDB daily closes
            try:
                dbm = await get_database_manager()
                async with dbm.get_questdb() as conn:
                    sql = (
                        "SELECT symbol, timestamp, close FROM daily_bars "
                        "WHERE symbol = ANY($1::text[]) "
                        "ORDER BY timestamp ASC"
                    )
                    rows = await conn.fetch(sql, symbols)
                import pandas as pd
                df = pd.DataFrame([dict(r) for r in rows])
                source_tag = 'pg'
                time_filter = 'exact_1600'
                resampled = False
                if df.empty:
                    # Broader fallback: fetch all bars and resample to daily last
                    async with dbm.get_questdb() as conn:
                        sql2 = (
                            "SELECT symbol, timestamp, close FROM market_data "
                            "WHERE symbol = ANY($1::text[]) ORDER BY timestamp ASC"
                        )
                        rows2 = await conn.fetch(sql2, symbols)
                    df = pd.DataFrame([dict(r) for r in rows2])
                    time_filter = 'resampled_daily'
                    resampled = True
                if df.empty:
                    raise HTTPException(status_code=502, detail='no_data_for_correlation')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Robust daily pivot: last close per calendar day per symbol
                df['date'] = df['timestamp'].dt.date
                df = df.sort_values(['symbol','timestamp'])
                df_daily = df.groupby(['symbol','date'], as_index=False).last()
                piv = df_daily.pivot(index='date', columns='symbol', values='close').dropna(how='any')
                corr = piv.pct_change().dropna().corr().fillna(0.0)
                return {
                    "type": "correlation",
                    "symbols": list(corr.columns),
                    "matrix": corr.values.tolist(),
                    "timestamp": datetime.utcnow().isoformat(),
                    "data_availability": {
                        "source": source_tag,
                        "time_filter": time_filter,
                        "resampled": resampled
                    }
                }
            except HTTPException:
                raise
            except Exception:
                # HTTP fallback via console API
                try:
                    in_list = ",".join(["'" + s.replace("'","''") + "'" for s in symbols])
                    sql = (
                        "SELECT symbol, timestamp, close FROM market_data "
                        f"WHERE symbol IN ({in_list}) "
                        "AND extract(HOUR from timestamp)=16 AND extract(MINUTE from timestamp)=0 AND extract(SECOND from timestamp)=0 "
                        "ORDER BY timestamp ASC"
                    )
                    rows = await _qdb_http_query(sql)
                    import pandas as pd
                    df = pd.DataFrame(rows)
                    source_tag = 'http'
                    time_filter = 'exact_1600'
                    resampled = False
                    if df.empty:
                        # broader HTTP fallback without time filter
                        sql2 = (
                            "SELECT symbol, timestamp, close FROM market_data "
                            f"WHERE symbol IN ({in_list}) ORDER BY timestamp ASC"
                        )
                        rows = await _qdb_http_query(sql2)
                        df = pd.DataFrame(rows)
                        time_filter = 'resampled_daily'
                        resampled = True
                    if df.empty:
                        raise HTTPException(status_code=502, detail='no_data_for_correlation')
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['date'] = df['timestamp'].dt.date
                    df = df.sort_values(['symbol','timestamp'])
                    df_daily = df.groupby(['symbol','date'], as_index=False).last()
                    corr = df_daily.pivot(index='date', columns='symbol', values='close').dropna(how='any').pct_change().dropna().corr().fillna(0.0)
                    return {
                        "type": "correlation",
                        "symbols": list(corr.columns),
                        "matrix": corr.values.tolist(),
                        "timestamp": datetime.utcnow().isoformat(),
                        "data_availability": {
                            "source": source_tag,
                            "time_filter": time_filter,
                            "resampled": resampled
                        }
                    }
                except HTTPException:
                    raise
                except Exception as ee:
                    raise HTTPException(status_code=500, detail=f'correlation_fallback_failed:{ee}')
        elif analysis_type == "causality":
            if entropy_analyzer:
                causality = await _infer("entropy_causality", entropy_analyzer.analyze_causality(symbols))
                return {
                    "type": "causality",
                    "symbols": symbols,
                    "transfer_entropy": causality,
                    "timestamp": datetime.utcnow().isoformat()
                }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {analysis_type}")
            
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/analyze/network").inc()
        logger.error(f"Network analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/generate")
async def generate_signals(
    symbols: List[str],
    strategies: Optional[List[str]] = None,
    use_ensemble: bool = True
):
    """Generate trading signals using advanced models."""
    try:
        if use_ensemble and ensemble_coordinator:
            signals = await _infer("ensemble_signals", ensemble_coordinator.generate_signals(symbols, strategies))
        elif intelligence_coordinator:
            signals = await _infer("advanced_signals", intelligence_coordinator.generate_signals(symbols, strategies))
        else:
            raise HTTPException(status_code=503, detail="Signal generation not available")
        
        return {
            "symbols": symbols,
            "strategies": strategies,
            "signals": signals,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/signals/generate").inc()
        logger.error(f"Signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel as _BaseModel

class BacktestBody(_BaseModel):
    strategy_config: Dict[str, Any]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbols: Optional[List[str]] = None

@app.post("/backtest/intelligent")
async def run_intelligent_backtest(payload: BacktestBody, request: Request):
    """Run intelligent backtesting with ML optimization."""
    try:
        # Resolve parameters from JSON body (preferred) with query fallback for backwards compatibility
        strategy_config = payload.strategy_config
        start_date = payload.start_date or request.query_params.get('start_date')
        end_date = payload.end_date or request.query_params.get('end_date')
        symbols = payload.symbols
        if symbols is None:
            # Read multiple query params: symbols=A&symbols=B
            try:
                symbols = request.query_params.getlist('symbols')  # type: ignore[attr-defined]
            except Exception:
                one = request.query_params.get('symbols')
                symbols = [one] if one else None
        if not strategy_config:
            raise HTTPException(status_code=400, detail='strategy_config_required')
        if not start_date or not end_date or not symbols:
            raise HTTPException(status_code=400, detail='start_date_end_date_symbols_required')
        # Preferred path: framework method if available
        if backtesting_framework and hasattr(backtesting_framework, 'run_backtest'):
            results = await _infer("intelligent_backtest", backtesting_framework.run_backtest(
                strategy_config,
                symbols,
                start_date,
                end_date
            ))
            return {
                "strategy": strategy_config.get("name", "unnamed"),
                "symbols": symbols,
                "period": f"{start_date} to {end_date}",
                "results": results,
                "data_availability": {
                    "source": "framework",
                    "time_filter": "unknown",
                    "resampled": None
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        # Fallback: simple SMA crossover on daily closes from QuestDB
        import pandas as pd, numpy as np
        lookback = int(((strategy_config or {}).get('params') or {}).get('lookback', 20))
        try:
            dbm = await get_database_manager()
            async with dbm.get_questdb() as conn:
                sql = (
                    "SELECT symbol, timestamp, close FROM market_data "
                    "WHERE symbol = ANY($1::text[]) "
                    "AND extract(HOUR from timestamp)=16 AND extract(MINUTE from timestamp)=0 AND extract(SECOND from timestamp)=0 "
                    "AND timestamp >= $2 AND timestamp <= $3 ORDER BY timestamp ASC"
                )
                rows = await conn.fetch(sql, symbols, start_date, end_date)
            df = pd.DataFrame([dict(r) for r in rows])
            if df.empty:
                # broader range without time filter
                async with dbm.get_questdb() as conn:
                    sql2 = (
                        "SELECT symbol, timestamp, close FROM market_data "
                        "WHERE symbol = ANY($1::text[]) AND timestamp >= $2 AND timestamp <= $3 ORDER BY timestamp ASC"
                    )
                    rows2 = await conn.fetch(sql2, symbols, start_date, end_date)
                df = pd.DataFrame([dict(r) for r in rows2])
                time_filter = 'resampled_daily'
                resampled = True
        except Exception:
            # HTTP fallback
            source_tag = 'http'
            in_list = ",".join(["'" + s.replace("'","''") + "'" for s in symbols])
            sql = (
                "SELECT symbol, timestamp, close FROM market_data "
                f"WHERE symbol IN ({in_list}) "
                "AND extract(HOUR from timestamp)=16 AND extract(MINUTE from timestamp)=0 AND extract(SECOND from timestamp)=0 "
                f"AND timestamp >= '{start_date}' AND timestamp <= '{end_date}' ORDER BY timestamp ASC"
            )
            rows = await _qdb_http_query(sql)
            df = pd.DataFrame(rows)
            if df.empty:
                sql2 = (
                    "SELECT symbol, timestamp, close FROM market_data "
                    f"WHERE symbol IN ({in_list}) AND timestamp >= '{start_date}' AND timestamp <= '{end_date}' ORDER BY timestamp ASC"
                )
                rows2 = await _qdb_http_query(sql2)
                df = pd.DataFrame(rows2)
                time_filter = 'resampled_daily'
                resampled = True
        if df.empty:
            raise HTTPException(status_code=502, detail='no_data_for_backtest')
        # Robust daily last-close per symbol
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df = df.sort_values(['symbol','timestamp'])
        df_daily = df.groupby(['symbol','date'], as_index=False).last()
        piv = df_daily.pivot(index='date', columns='symbol', values='close').dropna(how='any')
        rets = piv.pct_change().fillna(0.0)
        sma = piv.rolling(lookback).mean()
        signal = (piv > sma).astype(float).shift(1).fillna(0.0)
        port_ret = (signal * rets).mean(axis=1)
        cumret = (1.0 + port_ret).cumprod() - 1.0
        sharpe = float(np.sqrt(252) * (port_ret.mean() / (port_ret.std() + 1e-12)))
        peak = cumret.cummax()
        mdd = float(((cumret - peak).min()) or 0.0)
        return {
            "strategy": strategy_config.get("name", "sma_cross_fallback"),
            "symbols": symbols,
            "period": f"{start_date} to {end_date}",
            "results": {
                "cumulative_return": float(cumret.iloc[-1]) if len(cumret) else 0.0,
                "sharpe_ratio": round(sharpe, 4),
                "max_drawdown": round(abs(mdd), 6),
                "daily_points": {
                    "timestamps": [ts.isoformat() for ts in cumret.index.to_list()],
                    "cumret": [float(x) for x in cumret.to_list()],
                }
            },
            "data_availability": {
                "source": source_tag,
                "time_filter": time_filter,
                "resampled": resampled
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/backtest/intelligent").inc()
        logger.error(f"Backtesting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/train")
async def train_model(
    background_tasks: BackgroundTasks,
    model_type: str,
    training_config: Dict[str, Any]
):
    """Train a new model."""
    try:
        if not training_pipeline:
            raise HTTPException(status_code=503, detail="Training pipeline not available")
        # Prefer QuestDB-backed job form if symbols/start/end provided
        job_cfg = training_config.get('job') if isinstance(training_config, dict) else None
        job = job_cfg if isinstance(job_cfg, dict) else training_config
        # Allow overriding model_type/task_type in job
        if isinstance(job, dict):
            job.setdefault('model_type', model_type)
        
        # Dispatch as background job; run_training_job will fetch data from QuestDB and train
        async def _run_job(job_payload: Dict[str, Any]):
            try:
                res = await training_pipeline.run_training_job(job_payload)  # type: ignore[attr-defined]
                logger.info("Training job completed", model_name=job_payload.get('model_name'), metrics=res.get('metrics'))
            except Exception as e:  # noqa: BLE001
                logger.error(f"Training job failed: {e}")
        background_tasks.add_task(_run_job, job)

        return {
            "status": "training_started",
            "model_type": model_type,
            "job": job,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/models/train").inc()
        logger.error(f"Model training failed to start: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitor/drift")
async def check_model_drift():
    """Check for model drift across all models."""
    try:
        if not drift_monitor:
            raise HTTPException(status_code=503, detail="Drift monitor not available")
        
        drift_report = await _infer("drift_monitor", drift_monitor.check_all_models())
        
        return {
            "drift_detected": drift_report.drift_detected,
            "models_affected": drift_report.affected_models,
            "severity": drift_report.severity,
            "recommendations": drift_report.recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/monitor/drift").inc()
        logger.error(f"Drift monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/analytics")
async def get_performance_analytics(
    lookback_days: int = 30
):
    """Get comprehensive performance analytics."""
    try:
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics not available")
        
        analytics = await _infer("performance_analytics", analytics_service.get_analytics(lookback_days))
        
        return {
            "period_days": lookback_days,
            "analytics": analytics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/performance/analytics").inc()
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/continuous")
async def trigger_optimization(
    background_tasks: BackgroundTasks,
    optimization_type: str = "full"
):
    """Trigger continuous improvement optimization."""
    try:
        if not improvement_engine:
            raise HTTPException(status_code=503, detail="Improvement engine not available")
        
        # Start optimization in background
        background_tasks.add_task(
            improvement_engine.optimize,
            optimization_type
        )
        
        return {
            "status": "optimization_started",
            "type": optimization_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/optimize/continuous").inc()
        logger.error(f"Optimization failed to start: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
async def calculate_risk_metrics(symbols: List[str], timeframe: str) -> Dict[str, Any]:
    """Calculate comprehensive risk metrics using QuestDB daily closes.

    Metrics per first symbol vs SPY benchmark:
      - var_95, cvar_95 on daily returns
      - sharpe_ratio (assumes 0 risk-free), annualized
      - max_drawdown on price series
      - beta and correlation_spy relative to SPY
    """
    try:
        if not symbols:
            return {}
        symbol = symbols[0]

        rows = None
        try:
            dbm = await get_database_manager()
            async with dbm.get_questdb() as conn:
                # Pull daily close series for target symbol and SPY as benchmark
                sql = (
                    "SELECT symbol, timestamp, close FROM daily_bars "
                    "WHERE symbol IN ($1,$2) "
                    "ORDER BY timestamp ASC"
                )
                rows = await conn.fetch(sql, symbol, 'SPY')
        except Exception:
            # HTTP fallback
            in_list = f"'{symbol}','SPY'"
            sql = (
                "SELECT symbol, timestamp, close FROM daily_bars "
                f"WHERE symbol IN ({in_list}) "
                "ORDER BY timestamp ASC"
            )
            rows = await _qdb_http_query(sql)

        import numpy as np
        import pandas as pd

        if not rows:
            return {}

        df = pd.DataFrame([dict(r) for r in rows]) if isinstance(rows, list) and rows and not isinstance(rows[0], dict) else pd.DataFrame(rows)
        if df.empty:
            return {}
        if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception:
                pass
        # Robust daily pivot (in case exact 16:00 bars are not present)
        df['date'] = df['timestamp'].dt.date
        df = df.sort_values(['symbol','timestamp'])
        df_daily = df.groupby(['symbol','date'], as_index=False).last()
        pivot = df_daily.pivot(index='date', columns='symbol', values='close').dropna(how='any')
        if symbol not in pivot.columns or 'SPY' not in pivot.columns:
            # Fallback: compute metrics without benchmark if SPY missing
            series = pivot[symbol].dropna() if symbol in pivot.columns else pd.Series(dtype=float)
            rets = series.pct_change().dropna()
            if rets.empty:
                return {}
            var_95 = float(np.percentile(rets, 5)) * -1.0
            cvar_95 = float(rets[rets <= np.percentile(rets, 5)].mean()) * -1.0
            sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(252))
            # Max drawdown
            roll_max = series.cummax()
            drawdown = (series / roll_max - 1.0)
            mdd = float(drawdown.min() * -1.0)
            return {
                "var_95": round(var_95, 6),
                "cvar_95": round(cvar_95, 6),
                "sharpe_ratio": round(sharpe, 4),
                "max_drawdown": round(mdd, 6),
                "beta": None,
                "correlation_spy": None,
            }

        # Compute returns
        sym = pivot[symbol].dropna()
        spy = pivot['SPY'].dropna()
        idx = sym.index.intersection(spy.index)
        sym = sym.loc[idx]
        spy = spy.loc[idx]
        if len(idx) < 20:
            return {}
        r_sym = sym.pct_change().dropna()
        r_spy = spy.pct_change().dropna()
        idx = r_sym.index.intersection(r_spy.index)
        r_sym = r_sym.loc[idx]
        r_spy = r_spy.loc[idx]

        # VaR and CVaR (left tail)
        var_95 = float(np.percentile(r_sym, 5)) * -1.0
        cvar_95 = float(r_sym[r_sym <= np.percentile(r_sym, 5)].mean()) * -1.0

        # Sharpe (annualized)
        sharpe = float(r_sym.mean() / (r_sym.std() + 1e-12) * np.sqrt(252))

        # Max drawdown
        roll_max = sym.cummax()
        drawdown = (sym / roll_max - 1.0)
        mdd = float(drawdown.min() * -1.0)

        # Beta and correlation
        cov = float(np.cov(r_sym, r_spy)[0, 1])
        var_spy = float(np.var(r_spy)) + 1e-12
        beta = cov / var_spy
        corr = float(np.corrcoef(r_sym, r_spy)[0, 1])

        return {
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(mdd, 6),
            "beta": round(beta, 4),
            "correlation_spy": round(corr, 4),
        }
    except Exception as e:
        logger.error(f"Risk calculation failed: {e}")
        return {}


@app.post("/ollama/generate")
async def ollama_generate(
    model: str,
    prompt: str,
    task_type: str = "market",
    urgency: str = "normal",
    temperature: float = 0.7,
    max_tokens: int = 2000
):
    """Generate response using Ollama models with intelligent routing"""
    try:
        if not model_router or not ollama_service:
            raise HTTPException(status_code=503, detail="Ollama service not available")
        
        # Parse task type and urgency
        try:
            task = TaskType(task_type)
        except:
            task = TaskType.MARKET_ANALYSIS
        
        try:
            urg = TaskUrgency[urgency.upper()]
        except:
            urg = TaskUrgency.NORMAL
        
        # Use model router for intelligent selection
        if model == "auto":
            result = await _infer("router_auto", model_router.smart_execute(
                task_type=task,
                urgency=urg,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            ))
        else:
            # Direct model selection
            response = await _infer(f"ollama_{model}", ollama_service.generate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            ))
            result = {
                "success": True,
                "model": model,
                "response": response.response,
                "latency_ms": response.latency_ms,
                "cached": response.cached
            }
        
        return result
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/ollama/generate").inc()
        logger.error(f"Ollama generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ollama/models")
async def get_ollama_models():
    """Get available Ollama models and their status"""
    try:
        if not ollama_service:
            raise HTTPException(status_code=503, detail="Ollama service not available")
        
        models = await ollama_service.refresh_models()
        
        if model_router:
            router_stats = model_router.get_stats()
            return {
                "available_models": models,
                "router_stats": router_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "available_models": models,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/ollama/models").inc()
        logger.error(f"Failed to get Ollama models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ollama/warmup")
async def ollama_warmup(models: Optional[list[str]] = None):
    """Warm up selected Ollama models (or defaults) to ensure they are loaded and responsive.
    Returns per-model status with latency.
    """
    try:
        if not model_router:
            raise HTTPException(status_code=503, detail="Model router not available")
        await model_router.check_model_availability()
        if not models:
            defaults = os.getenv('OLLAMA_WARMUP_MODELS', 'solar:10.7b,phi3:14b,mixtral:8x22b')
            models = [m.strip() for m in defaults.split(',') if m.strip()]
        results = await _warmup_ollama_models(model_router, models)
        return {"results": results, "timestamp": datetime.utcnow().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/ollama/warmup").inc()
        logger.error(f"Ollama warmup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class WarmModePayload(BaseModel):
    mode: str  # 'day' | 'night'

@app.post('/ollama/warm/switch')
async def ollama_warm_switch(payload: WarmModePayload):
    """Manually warm day or night model sets regardless of current market hours."""
    try:
        if not model_router:
            raise HTTPException(status_code=503, detail="Model router not available")
        mode = (payload.mode or '').strip().lower()
        if mode not in ('day','night'):
            raise HTTPException(status_code=400, detail='mode_must_be_day_or_night')
        targets = DAY_HOT_MODELS if mode == 'day' else NIGHT_HEAVY_MODELS
        await model_router.check_model_availability()
        res = await _warmup_ollama_models(model_router, targets)
        # Night mode may include DeepSeek warming as a separate step
        extra = None
        if mode == 'night' and _is_deepseek_night_allowed() and DEEPSEEK_MODEL:
            try:
                extra = await _warmup_ollama_models(model_router, [DEEPSEEK_MODEL])
            except Exception:
                extra = None
        return {'status': 'ok', 'mode': mode, 'targets': targets + (([DEEPSEEK_MODEL] if mode=='night' and _is_deepseek_night_allowed() and DEEPSEEK_MODEL else [])), 'results': res, 'extra': extra}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/sentiment/finbert")
async def analyze_sentiment_finbert(
    texts: List[str],
    aggregate: bool = True
):
    """Analyze sentiment using FinBERT"""
    try:
        if not finbert_analyzer:
            raise HTTPException(status_code=503, detail="FinBERT not available")
        
        # FinBERT is synchronous CPU-bound: run in a thread and measure via helper
        results = await _infer("finbert_batch", asyncio.to_thread(finbert_analyzer.batch_analyze, texts))
        
        # Convert SentimentResult objects to dicts
        results_dict = []
        for r in results:
            results_dict.append({
                "text": r.text,
                "sentiment": r.sentiment,
                "confidence": r.confidence,
                "scores": r.scores
            })
        
        if aggregate and len(results) > 0:
            # Calculate aggregate sentiment from SentimentResult objects
            total_positive = sum(r.scores['positive'] for r in results) / len(results)
            total_negative = sum(r.scores['negative'] for r in results) / len(results)
            total_neutral = sum(r.scores['neutral'] for r in results) / len(results)
            
            # Determine overall sentiment
            if total_positive > total_negative and total_positive > total_neutral:
                overall_sentiment = "positive"
            elif total_negative > total_positive and total_negative > total_neutral:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            return {
                "individual_results": results_dict,
                "aggregate": {
                    "positive": total_positive,
                    "negative": total_negative,
                    "neutral": total_neutral,
                    "overall_sentiment": overall_sentiment
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "results": results_dict,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/analyze/sentiment/finbert").inc()
        logger.error(f"FinBERT analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/tiered")
async def tiered_analysis(
    prompt: str,
    tier: int = 3,
    task_type: str = "market"
):
    """Perform tiered analysis based on urgency level"""
    try:
        if not model_router or not ollama_service:
            raise HTTPException(status_code=503, detail="Ollama service not available")
        
        # Map tier to urgency
        urgency_map = {
            1: TaskUrgency.REALTIME,
            2: TaskUrgency.FAST,
            3: TaskUrgency.NORMAL,
            4: TaskUrgency.BATCH,
            5: TaskUrgency.DEEP
        }
        
        urgency = urgency_map.get(tier, TaskUrgency.NORMAL)
        
        try:
            task = TaskType(task_type)
        except:
            task = TaskType.MARKET_ANALYSIS
        
        # Execute with appropriate model
        result = await _infer(f"tier_{tier}", model_router.smart_execute(
            task_type=task,
            urgency=urgency,
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000
        ))
        
        return {
            "tier": tier,
            "urgency": urgency.name,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        ML_ENDPOINT_ERRORS.labels(endpoint="/analyze/tiered").inc()
        logger.error(f"Tiered analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting Advanced ML Service...")
    # Run directly with app object to avoid second module import which caused
    # Prometheus metric double-registration (ValueError: Duplicated timeseries)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
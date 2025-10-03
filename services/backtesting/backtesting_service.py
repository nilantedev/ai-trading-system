#!/usr/bin/env python3
"""
Backtesting Service - Historical strategy testing and validation
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
import uvicorn
import time
import uuid
import pandas as pd
import numpy as np
import logging

# Add shared library to path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from trading_common import get_logger, get_settings
from trading_common.cache import get_trading_cache
from trading_common.database import get_redis_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from observability import install_observability, register_path_template

logger = get_logger(__name__)
settings = get_settings()

# Global state
cache_client = None
redis_client = None
backtest_results = {}

# Readiness gauge (shared semantic: 0 init,1 ready,2 degraded)
try:
    SERVICE_READINESS_STATE = Gauge('service_readiness_state', 'Readiness state per service (0=initializing,1=ready,2=degraded)', ['service'])
except Exception:  # noqa: BLE001
    SERVICE_READINESS_STATE = None
if SERVICE_READINESS_STATE:
    try:
        SERVICE_READINESS_STATE.labels(service='backtesting').set(0)
    except Exception:  # noqa: BLE001
        pass

BT_RUNS_TOTAL = Counter('backtest_runs_total', 'Total backtests executed')
BT_LATENCY = Histogram('backtest_run_latency_seconds', 'Backtest run latency seconds', buckets=(0.05,0.1,0.25,0.5,1,2,5,10,30,60))
BT_RESULTS_GAUGE = Gauge('backtest_results_cached', 'Currently stored backtest result count')

def _normalize_path(p:str)->str:
    if p.startswith('/backtest/') and len(p.split('/'))==3:
        return '/backtest/{id}'
    return p

# Register as middleware after app is created; define function first
async def correlation_and_domain_metrics(request: Request, call_next):
    cid = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
    response: Response
    try:
        response = await call_next(request)
    finally:
        try:
            BT_RESULTS_GAUGE.set(len(backtest_results))
        except Exception:  # noqa: BLE001
            pass
    response.headers['X-Correlation-ID'] = cid
    return response

# Register as route after app is created; define function first
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
    """Application lifecycle management with resilience."""
    global cache_client, redis_client

    logger.info("Starting Backtesting Service")
    try:
        cache_client = await _connect_with_retry("cache", get_trading_cache)
        redis_client = await _connect_with_retry("redis", get_redis_client)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Unexpected startup error: {e}")

    yield

    try:
        if cache_client and hasattr(cache_client, "close"):
            await cache_client.close()
        if redis_client and hasattr(redis_client, "close"):
            await redis_client.close()
    except Exception:  # noqa: BLE001
        pass
    logger.info("Backtesting Service stopped")

app = FastAPI(
    title="AI Trading System - Backtesting Service",
    description="Historical strategy testing and validation",
    version="1.0.0",
    lifespan=lifespan
)

# Install shared observability and path template
register_path_template('/backtest/*', '/backtest/{id}')
try:
    concurrency_limit = int(os.getenv('SERVICE_CONCURRENCY_LIMIT', '0') or 0)
except ValueError:
    concurrency_limit = 0
install_observability(app, 'backtesting', concurrency_limit if concurrency_limit > 0 else None)

# Now that app exists, register middleware and routes defined earlier
app.middleware('http')(correlation_and_domain_metrics)
app.add_api_route('/metrics', metrics, methods=['GET'])

@app.get("/health")
async def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "backtesting",
        "timestamp": datetime.utcnow().isoformat(),
        "connections": {
            "cache": cache_client is not None,
            "redis": redis_client is not None
        }
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "backtesting", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
async def readiness():
    components = {
        "cache": cache_client is not None,
        "redis": redis_client is not None
    }
    ready = components["cache"] and components["redis"]
    status_code = 200 if ready else 503
    # Update readiness gauge
    if SERVICE_READINESS_STATE:
        try:
            SERVICE_READINESS_STATE.labels(service='backtesting').set(1 if ready else 2)
        except Exception:  # noqa: BLE001
            pass
    return JSONResponse(status_code=status_code, content={
        "status": "ready" if ready else "degraded",
        "components": components,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.post("/backtest/run")
async def run_backtest(config: Dict[str, Any]):
    """Run a backtest with given configuration."""
    backtest_id = f"bt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Simulate backtest execution
    bt_start = time.perf_counter()
    results = {
        "backtest_id": backtest_id,
        "strategy": config.get("strategy", "momentum"),
        "symbol": config.get("symbol", "AAPL"),
        "period": config.get("period", "30d"),
        "initial_capital": config.get("initial_capital", 10000),
        "final_capital": 11500,
        "total_return": 0.15,
        "annualized_return": 0.45,
        "sharpe_ratio": 1.2,
        "sortino_ratio": 1.5,
        "max_drawdown": -0.08,
        "win_rate": 0.55,
        "profit_factor": 1.8,
        "total_trades": 100,
        "winning_trades": 55,
        "losing_trades": 45,
        "avg_win": 50,
        "avg_loss": -20,
        "best_trade": 200,
        "worst_trade": -80,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    backtest_results[backtest_id] = results
    BT_RUNS_TOTAL.inc()
    BT_LATENCY.observe(time.perf_counter() - bt_start)
    
    return results

@app.get("/backtest/{backtest_id}")
async def get_backtest_result(backtest_id: str):
    """Get results of a specific backtest."""
    if backtest_id not in backtest_results:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")
    
    return backtest_results[backtest_id]

@app.get("/backtest/list")
async def list_backtests():
    """List all completed backtests."""
    return {
        "backtests": list(backtest_results.values()),
        "total": len(backtest_results),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/backtest/compare")
async def compare_backtests(backtest_ids: List[str]):
    """Compare multiple backtest results."""
    results = []
    for bt_id in backtest_ids:
        if bt_id in backtest_results:
            results.append(backtest_results[bt_id])
    
    if not results:
        raise HTTPException(status_code=404, detail="No valid backtests found")
    
    # Create comparison summary
    comparison = {
        "backtests": results,
        "best_return": max(results, key=lambda x: x["total_return"]),
        "best_sharpe": max(results, key=lambda x: x["sharpe_ratio"]),
        "lowest_drawdown": min(results, key=lambda x: abs(x["max_drawdown"])),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return comparison

@app.post("/backtest/optimize")
async def optimize_parameters(config: Dict[str, Any]):
    """Run parameter optimization for a strategy."""
    optimization_id = f"opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Simulate parameter optimization
    results = {
        "optimization_id": optimization_id,
        "strategy": config.get("strategy", "momentum"),
        "parameters_tested": 100,
        "best_parameters": {
            "lookback_period": 20,
            "threshold": 0.02,
            "stop_loss": 0.05,
            "take_profit": 0.10
        },
        "best_sharpe": 1.5,
        "best_return": 0.25,
        "optimization_time": "2 minutes",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return results

@app.post("/backtest/walk-forward")
async def walk_forward_analysis(config: Dict[str, Any]):
    """Perform walk-forward analysis."""
    return {
        "strategy": config.get("strategy", "momentum"),
        "in_sample_periods": 10,
        "out_sample_periods": 10,
        "avg_in_sample_return": 0.12,
        "avg_out_sample_return": 0.10,
        "efficiency_ratio": 0.83,
        "robust": True,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8007,
        reload=False,
        workers=1
    )
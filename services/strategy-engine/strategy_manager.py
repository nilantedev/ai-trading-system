#!/usr/bin/env python3
"""
Strategy Engine Service - Manages and orchestrates trading strategies
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
# Ensure shared common is on path before importing observability utilities
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))
try:
    from observability import install_observability
except Exception:  # noqa: BLE001
    # Fallback to namespaced import if available
    from trading_common.observability import install_observability  # type: ignore
import logging
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from trading_common import get_logger, get_settings
from trading_common.cache import get_trading_cache
from trading_common.database import get_redis_client

logger = get_logger(__name__)
settings = get_settings()

# Global state
cache_client = None
redis_client = None
active_strategies = {}

# Prometheus domain metrics (per-request latency & counts now via shared app_* metrics)
try:
    STRATEGY_EVALUATIONS = Counter(
        "strategy_evaluations_total",
        "Number of strategy evaluations",
        ["strategy"]
    )
    STRATEGY_BACKTESTS = Counter(
        "strategy_backtests_total",
        "Number of strategy backtests",
        ["strategy"]
    )
    STRATEGY_ACTIVE = Gauge(
        "strategy_active_strategies",
        "Number of active strategies"
    )
    STRATEGY_LAST_SIGNAL_AGE = Gauge(
        "strategy_last_signal_age_seconds",
        "Age in seconds since last signal per strategy",
        ["strategy"]
    )
    STRATEGY_ERRORS = Counter(
        "strategy_engine_errors_total",
        "Number of errors encountered",
        ["endpoint"]
    )
except ValueError:
    # Duplicate registration (e.g., module re-import); fetch metrics from default registry
    from prometheus_client import REGISTRY
    def _get_metric(name):
        for c in REGISTRY._names_to_collectors.values():  # noqa: SLF001
            if getattr(c, '_name', None) == name:
                return c
        raise
    STRATEGY_EVALUATIONS = _get_metric('strategy_evaluations')  # type: ignore
    STRATEGY_BACKTESTS = _get_metric('strategy_backtests')  # type: ignore
    STRATEGY_ACTIVE = _get_metric('strategy_active_strategies')  # type: ignore
    STRATEGY_LAST_SIGNAL_AGE = _get_metric('strategy_last_signal_age_seconds')  # type: ignore
    STRATEGY_ERRORS = _get_metric('strategy_engine_errors')  # type: ignore


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

class PortfolioOptimizer:
    """Advanced portfolio optimization using Markowitz, Black-Litterman, and Risk Parity."""
    
    @staticmethod
    def calculate_returns_statistics(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate expected returns and covariance matrix."""
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
        
        expected_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        
        # Ensure covariance matrix is positive definite
        if len(cov_matrix.shape) == 0:
            cov_matrix = np.array([[max(0.0001, cov_matrix)]])
        else:
            min_eigenvalue = np.min(np.linalg.eigvalsh(cov_matrix))
            if min_eigenvalue < 0:
                cov_matrix = cov_matrix + (-min_eigenvalue + 0.0001) * np.eye(cov_matrix.shape[0])
        
        return expected_returns, cov_matrix
    
    @staticmethod
    def markowitz_optimization(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                              risk_free_rate: float = 0.02,
                              target_return: Optional[float] = None) -> np.ndarray:
        """Markowitz Mean-Variance Optimization.
        
        Returns optimal portfolio weights.
        """
        n_assets = len(expected_returns)
        
        # Optimization constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            # Add target return constraint
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(expected_returns, x) - target_return
            })
        
        # Bounds for weights (0 to 1 for long-only)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Objective function (minimize portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # If no target return, maximize Sharpe ratio
        if target_return is None:
            def neg_sharpe_ratio(weights):
                returns = np.dot(expected_returns, weights)
                volatility = np.sqrt(portfolio_variance(weights))
                return -(returns - risk_free_rate) / volatility if volatility > 0 else 0
            
            result = minimize(neg_sharpe_ratio, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints[0])
        else:
            result = minimize(portfolio_variance, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    @staticmethod
    def black_litterman(market_weights: np.ndarray, cov_matrix: np.ndarray,
                        views_matrix: np.ndarray, views_expected_returns: np.ndarray,
                        views_uncertainty: np.ndarray, risk_aversion: float = 2.5,
                        tau: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Black-Litterman model for combining market equilibrium with investor views.
        
        Args:
            market_weights: Market capitalization weights
            cov_matrix: Covariance matrix of returns
            views_matrix: P matrix mapping views to assets
            views_expected_returns: Q vector of expected returns for views
            views_uncertainty: Omega diagonal of view uncertainties
            risk_aversion: Risk aversion parameter
            tau: Scaling factor for prior uncertainty
            
        Returns:
            Tuple of (posterior expected returns, posterior covariance)
        """
        # Calculate equilibrium returns (reverse optimization)
        equilibrium_returns = risk_aversion * np.dot(cov_matrix, market_weights)
        
        # Prior covariance (scaled market covariance)
        prior_cov = tau * cov_matrix
        
        # Views uncertainty matrix
        if len(views_uncertainty.shape) == 1:
            omega = np.diag(views_uncertainty)
        else:
            omega = views_uncertainty
        
        # Black-Litterman posterior returns
        term1 = np.linalg.inv(np.linalg.inv(prior_cov) + 
                             np.dot(views_matrix.T, np.dot(np.linalg.inv(omega), views_matrix)))
        term2 = (np.dot(np.linalg.inv(prior_cov), equilibrium_returns) + 
                np.dot(views_matrix.T, np.dot(np.linalg.inv(omega), views_expected_returns)))
        
        posterior_returns = np.dot(term1, term2)
        
        # Posterior covariance
        posterior_cov = term1 + cov_matrix
        
        return posterior_returns, posterior_cov
    
    @staticmethod
    def risk_parity(cov_matrix: np.ndarray) -> np.ndarray:
        """Risk Parity portfolio - equal risk contribution from each asset."""
        n_assets = cov_matrix.shape[0]
        
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
        
        def objective(weights):
            contrib = risk_contribution(weights)
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.001, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    @staticmethod
    def hierarchical_risk_parity(returns: np.ndarray) -> np.ndarray:
        """Hierarchical Risk Parity (HRP) - uses hierarchical clustering."""
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
        
        n_assets = returns.shape[1]
        
        # Calculate correlation and distance matrices
        corr_matrix = np.corrcoef(returns.T)
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # For simplicity, use equal weights (full HRP requires clustering)
        # In production, would use scipy.cluster.hierarchy
        weights = np.array([1/n_assets] * n_assets)
        
        return weights
    
    @staticmethod
    def kelly_criterion(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                       risk_free_rate: float = 0.02) -> np.ndarray:
        """Kelly Criterion for optimal bet sizing."""
        n_assets = len(expected_returns)
        
        # Calculate excess returns
        excess_returns = expected_returns - risk_free_rate
        
        # Kelly weights (simplified for multiple assets)
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            kelly_weights = np.dot(inv_cov, excess_returns)
            
            # Normalize to sum to 1 (for fully invested portfolio)
            if np.sum(np.abs(kelly_weights)) > 0:
                kelly_weights = kelly_weights / np.sum(np.abs(kelly_weights))
            else:
                kelly_weights = np.array([1/n_assets] * n_assets)
                
        except np.linalg.LinAlgError:
            # Fallback to equal weights if matrix is singular
            kelly_weights = np.array([1/n_assets] * n_assets)
        
        return kelly_weights
    
    @staticmethod
    def calculate_efficient_frontier(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                    n_portfolios: int = 50) -> Dict[str, List]:
        """Calculate efficient frontier portfolios."""
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []
        
        for target_return in target_returns:
            weights = PortfolioOptimizer.markowitz_optimization(
                expected_returns, cov_matrix, target_return=target_return
            )
            
            portfolio_return = np.dot(expected_returns, weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            frontier_returns.append(portfolio_return)
            frontier_volatilities.append(portfolio_vol)
            frontier_weights.append(weights.tolist())
        
        return {
            'returns': frontier_returns,
            'volatilities': frontier_volatilities,
            'weights': frontier_weights
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management with resilience."""
    global cache_client, redis_client

    logger.info("Starting Strategy Engine Service")
    try:
        cache_client = await _connect_with_retry("cache", get_trading_cache)
        redis_client = await _connect_with_retry("redis", get_redis_client)
        try:
            await initialize_strategies()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Strategy initialization failed: {e}")
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
    logger.info("Strategy Engine Service stopped")

app = FastAPI(
    title="AI Trading System - Strategy Engine",
    description="Manages and orchestrates trading strategies",
    version="1.0.0",
    lifespan=lifespan
)

# Shared observability install (strategy-engine)
_shared_cc = install_observability(app, service_name="strategy-engine")

@app.middleware("http")
async def correlation_id_only(request: Request, call_next):
    """Attach correlation ID; shared observability layer records latency & counts."""
    correlation_id = request.headers.get("X-Correlation-ID") or f"strat-{int(asyncio.get_event_loop().time()*1000)}"
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        STRATEGY_ERRORS.labels(endpoint=request.url.path).inc()
        raise
    response.headers["X-Correlation-ID"] = correlation_id
    return response

async def initialize_strategies():
    """Initialize available trading strategies."""
    strategies = [
        "momentum",
        "mean_reversion", 
        "arbitrage",
        "pairs_trading",
        "ml_ensemble"
    ]
    
    for strategy in strategies:
        active_strategies[strategy] = {
            "name": strategy,
            "status": "ready",
            "last_signal": None,
            "performance": 0.0
        }
    
    logger.info(f"Initialized {len(strategies)} strategies")
    STRATEGY_ACTIVE.set(len(strategies))

@app.get("/health")
async def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "strategy-engine",
        "timestamp": datetime.utcnow().isoformat(),
        "connections": {
            "cache": cache_client is not None,
            "redis": redis_client is not None
        },
        "active_strategies": len(active_strategies)
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "strategy-engine", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
async def readiness():
    components = {
        "cache": cache_client is not None,
        "redis": redis_client is not None,
        "strategies_initialized": len(active_strategies) > 0
    }
    ready = components["cache"] and components["redis"] and components["strategies_initialized"]
    status_code = 200 if ready else 503
    return JSONResponse(status_code=status_code, content={
        "status": "ready" if ready else "degraded",
        "components": components,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.get("/strategies")
async def list_strategies():
    """List all available strategies."""
    return {
        "strategies": list(active_strategies.values()),
        "total": len(active_strategies),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/strategies/{strategy_name}/evaluate")
async def evaluate_strategy(strategy_name: str, data: Dict[str, Any]):
    """Evaluate a strategy with given market data."""
    if strategy_name not in active_strategies:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_name} not found")
    
    # Simulate strategy evaluation
    signal = {
        "strategy": strategy_name,
        "action": "buy" if data.get("price", 100) < 100 else "sell",
        "confidence": 0.75,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    active_strategies[strategy_name]["last_signal"] = signal
    STRATEGY_EVALUATIONS.labels(strategy=strategy_name).inc()
    
    return signal

@app.post("/strategies/{strategy_name}/backtest")
async def backtest_strategy(strategy_name: str, params: Dict[str, Any]):
    """Run backtest for a specific strategy."""
    if strategy_name not in active_strategies:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_name} not found")
    
    # Simulate backtest results
    results = {
        "strategy": strategy_name,
        "period": params.get("period", "30d"),
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.08,
        "win_rate": 0.55,
        "total_trades": 100,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    STRATEGY_BACKTESTS.labels(strategy=strategy_name).inc()
    return results

@app.post("/strategies/optimize")
async def optimize_strategies(params: Dict[str, Any]):
    """Optimize strategy parameters."""
    return {
        "status": "optimization_started",
        "strategies": list(active_strategies.keys()),
        "estimated_time": "5 minutes",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
async def _metrics():  # noqa: D401
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _signal_age_updater():
    """Background task updating last signal age gauge."""
    while True:
        try:
            now = datetime.utcnow()
            for name, data in active_strategies.items():
                last = data.get("last_signal")
                if last and isinstance(last, dict):
                    ts = last.get("timestamp")
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
                    except Exception:  # noqa: BLE001
                        dt = None
                    if dt:
                        age = (now - dt.replace(tzinfo=None)).total_seconds()
                        STRATEGY_LAST_SIGNAL_AGE.labels(strategy=name).set(age)
            await asyncio.sleep(5)
        except asyncio.CancelledError:  # graceful shutdown
            break
        except Exception as e:  # noqa: BLE001
            logger.warning(f"signal age updater error: {e}")
            await asyncio.sleep(5)


# Launch background task after startup
@app.on_event("startup")
async def _start_background_tasks():  # noqa: D401
    asyncio.create_task(_signal_age_updater())

if __name__ == "__main__":
    uvicorn.run(
        "strategy_manager:app",
        host="0.0.0.0",
        port=8006,
        reload=False,
        workers=1
    )
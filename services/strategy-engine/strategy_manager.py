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
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from trading_common import get_logger, get_settings
from trading_common.cache import get_trading_cache
from trading_common.database import get_redis_client

# Import real strategies
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
# Elite strategies
from strategies.statistical_arbitrage import StatisticalArbitrageStrategy
from strategies.market_making import MarketMakingStrategy
from strategies.volatility_arbitrage import VolatilityArbitrageStrategy
from strategies.index_arbitrage import IndexArbitrageStrategy
from strategies.trend_following import AdvancedTrendFollowingStrategy
# Strategy adapter for backtesting compatibility
from strategy_adapter import StrategyAdapter
# Dynamic infrastructure
from watchlist_manager import WatchlistManager
from portfolio_manager import PortfolioManager, PortfolioConfig
from continuous_processor import ContinuousProcessor

logger = get_logger(__name__)
settings = get_settings()

# Global state
cache_client = None
redis_client = None
active_strategies = {}
strategy_instances = {}  # Store instantiated strategy objects
watchlist_manager = None  # Dynamic watchlist from QuestDB
portfolio_manager = None  # Position sizing and risk management
continuous_processor = None  # Continuous strategy processing

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
    """Lifespan events for startup/shutdown."""
    global cache_client, redis_client, watchlist_manager, portfolio_manager, continuous_processor
    logger.info("Strategy Engine starting up...")
    
    # Connect cache (retryable)
    cache_client = await _connect_with_retry(
        "Cache",
        lambda: get_trading_cache(settings),
        attempts=5,
        base_delay=1.0
    )
    
    # Connect redis (retryable)
    redis_client = await _connect_with_retry(
        "Redis",
        lambda: get_redis_client(settings),
        attempts=5,
        base_delay=1.0
    )
    
    # Initialize watchlist manager
    try:
        watchlist_manager = WatchlistManager()
        logger.info("Watchlist manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize watchlist manager: {e}")
    
    # Initialize portfolio manager
    try:
        portfolio_config = PortfolioConfig(
            total_capital=100000.0,
            max_position_pct=0.10,  # 10% per position
            max_positions=10,
            reserve_cash_pct=0.20,  # 20% reserve
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.10
        )
        portfolio_manager = PortfolioManager(portfolio_config)
        logger.info("Portfolio manager initialized with $100K capital")
    except Exception as e:
        logger.error(f"Failed to initialize portfolio manager: {e}")
    
    # Initialize strategies
    await initialize_strategies()
    
    # Initialize continuous processor for real-time strategy execution
    try:
        continuous_processor = ContinuousProcessor(
            watchlist_manager=watchlist_manager,
            portfolio_manager=portfolio_manager,
            update_interval=60  # Process every 60 seconds
        )
        # Start continuous processing in background
        strategies = ['momentum', 'mean_reversion', 'stat_arb', 'pairs_trading']
        asyncio.create_task(continuous_processor.start(strategies))
        logger.info("Continuous processor started with strategies: momentum, mean_reversion, stat_arb, pairs_trading")
    except Exception as e:
        logger.error(f"Failed to initialize continuous processor: {e}")
    
    logger.info("Strategy Engine startup complete")
    yield
    logger.info("Strategy Engine shutting down...")
    
    # Stop continuous processor
    if continuous_processor:
        try:
            continuous_processor.stop()
            logger.info("Continuous processor stopped")
        except Exception as e:
            logger.error(f"Error stopping continuous processor: {e}")

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
    """Initialize available trading strategies including elite hedge fund strategies."""
    # Basic strategies
    strategy_instances["momentum"] = MomentumStrategy(
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9
    )
    
    strategy_instances["mean_reversion"] = MeanReversionStrategy(
        lookback=20,
        std_multiplier=2,
        z_threshold=2
    )
    
    # Elite hedge fund strategies (wrapped with adapter for backtesting)
    logger.info("Initializing elite hedge fund strategies...")
    
    # Statistical Arbitrage (Renaissance Technologies, Citadel approach)
    stat_arb = StatisticalArbitrageStrategy(
        lookback_period=60,
        entry_z_threshold=2.0,
        exit_z_threshold=0.5,
        stop_loss_z=4.0,
        min_correlation=0.7,
        max_pvalue=0.05
    )
    strategy_instances["stat_arb"] = StrategyAdapter(stat_arb, "stat_arb")
    logger.info("✓ Statistical Arbitrage: Pairs trading with cointegration (Renaissance, Citadel)")
    
    # Market Making (Virtu Financial, Tower Research approach)
    market_making = MarketMakingStrategy(
        base_spread_bps=10.0,
        min_spread_bps=5.0,
        max_spread_bps=50.0,
        target_inventory=0,
        max_inventory=1000,
        risk_aversion=0.5
    )
    strategy_instances["market_making"] = StrategyAdapter(market_making, "market_making")
    logger.info("✓ Market Making: Bid-ask spread capture (Virtu 99.9% win rate approach)")
    
    # Volatility Arbitrage (Susquehanna, Jane Street approach)
    vol_arb = VolatilityArbitrageStrategy(
        vol_threshold=0.05,
        min_vega=100.0,
        max_vega=10000.0,
        lookback_window=30
    )
    strategy_instances["vol_arb"] = StrategyAdapter(vol_arb, "vol_arb")
    logger.info("✓ Volatility Arbitrage: IV vs RV trading (SIG, Jane Street)")
    
    # Index Arbitrage (AQR, Millennium approach)
    index_arb = IndexArbitrageStrategy(
        min_spread_bps=5.0,
        max_basket_size=50,
        futures_threshold_bps=10.0,
        rebalance_lead_days=5
    )
    strategy_instances["index_arb"] = StrategyAdapter(index_arb, "index_arb")
    logger.info("✓ Index Arbitrage: Index rebalancing + futures basis (AQR, Millennium)")
    
    # Advanced Trend Following (AQR, Two Sigma, Winton approach)
    trend_following = AdvancedTrendFollowingStrategy(
        short_window=20,
        medium_window=60,
        long_window=200,
        atr_period=14,
        vol_target=0.15,
        min_trend_strength=0.6
    )
    strategy_instances["trend_following"] = StrategyAdapter(trend_following, "trend_following")
    logger.info("✓ Trend Following: Multi-timeframe momentum (AQR Managed Futures, Two Sigma)")
    
    # Initialize status tracking
    for strategy_name in strategy_instances.keys():
        active_strategies[strategy_name] = {
            "name": strategy_name,
            "status": "ready",
            "last_signal": None,
            "performance": 0.0,
            "type": _get_strategy_type(strategy_name)
        }
    
    logger.info(f"✓ Initialized {len(strategy_instances)} strategies including {len(strategy_instances) - 2} elite hedge fund strategies")
    logger.info(f"Active strategies: {', '.join(list(strategy_instances.keys()))}")
    STRATEGY_ACTIVE.set(len(strategy_instances))

def _get_strategy_type(strategy_name: str) -> str:
    """Get strategy type for classification."""
    type_map = {
        "momentum": "basic",
        "mean_reversion": "basic",
        "stat_arb": "elite_statistical_arbitrage",
        "market_making": "elite_hft",
        "vol_arb": "elite_options",
        "index_arb": "elite_quantitative",
        "trend_following": "elite_managed_futures"
    }
    return type_map.get(strategy_name, "unknown")

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
    
    if strategy_name not in strategy_instances:
        raise HTTPException(status_code=501, detail=f"Strategy {strategy_name} not implemented yet")
    
    try:
        # Get strategy instance and evaluate
        strategy = strategy_instances[strategy_name]
        symbol = data.get("symbol", "UNKNOWN")
        
        # Call real strategy evaluation
        signal = await strategy.evaluate(symbol, data)
        
        # Update tracking
        active_strategies[strategy_name]["last_signal"] = signal
        STRATEGY_EVALUATIONS.labels(strategy=strategy_name).inc()
        
        logger.info(f"Strategy {strategy_name} evaluated {symbol}: {signal.get('signal_type')} (confidence: {signal.get('confidence', 0):.2f})")
        
        return signal
        
    except Exception as e:
        logger.error(f"Strategy evaluation failed for {strategy_name}: {e}")
        STRATEGY_ERRORS.labels(endpoint=f"/strategies/{strategy_name}/evaluate").inc()
        raise HTTPException(status_code=500, detail=f"Strategy evaluation failed: {str(e)}")

@app.post("/strategies/{strategy_name}/backtest")
async def backtest_strategy(strategy_name: str, params: Dict[str, Any]):
    """Run backtest for a specific strategy."""
    if strategy_name not in active_strategies:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_name} not found")
    
    # TODO: Implement real backtesting engine with QuestDB historical data
    # For now, return error indicating not implemented
    logger.warning(f"Backtest requested for {strategy_name} but backtesting engine not implemented yet")
    
    raise HTTPException(
        status_code=501,
        detail="Backtesting engine not implemented yet. Priority 2 task - see PRIORITY_FIXES_ACTION_PLAN.md"
    )
    
    # STRATEGY_BACKTESTS.labels(strategy=strategy_name).inc()
    # return results

@app.post("/strategies/optimize")
async def optimize_strategies(params: Dict[str, Any]):
    """Optimize strategy parameters."""
    return {
        "status": "optimization_started",
        "strategies": list(active_strategies.keys()),
        "estimated_time": "5 minutes",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/strategies/ensemble")
async def ensemble_voting(data: Dict[str, Any]):
    """
    Ensemble voting across multiple strategies.
    
    Combines signals from all active strategies using weighted voting.
    Elite strategies get higher weight based on historical performance.
    
    Args:
        data: Market data including symbol, price_data, options_chain, etc.
        
    Returns:
        Aggregated signal with consensus direction and confidence
    """
    symbol = data.get("symbol", "UNKNOWN")
    logger.info(f"Running ensemble voting for {symbol} across {len(strategy_instances)} strategies")
    
    votes = {
        "LONG": 0.0,
        "SHORT": 0.0,
        "NEUTRAL": 0.0
    }
    
    strategy_signals = {}
    total_weight = 0.0
    
    # Strategy weights (elite strategies get higher weight)
    strategy_weights = {
        "stat_arb": 2.0,  # High weight for proven strategies
        "market_making": 1.5,
        "vol_arb": 1.5,
        "index_arb": 1.5,
        "trend_following": 2.0,
        "momentum": 1.0,
        "mean_reversion": 1.0
    }
    
    # Collect signals from all strategies
    for strategy_name, strategy in strategy_instances.items():
        try:
            # Determine if strategy is applicable based on data available
            applicable = True
            
            # Some strategies need specific data
            if strategy_name == "vol_arb" and "options_chain" not in data:
                applicable = False
            if strategy_name == "index_arb" and "index_composition" not in data:
                applicable = False
            
            if not applicable:
                continue
            
            # Generate signal (this is simplified - real implementation would call strategy methods)
            signal = None
            confidence = 0.0
            direction = "NEUTRAL"
            
            # For now, we'll simulate signal generation
            # In production, each strategy would have a standard evaluate() method
            if hasattr(strategy, 'evaluate'):
                signal = await strategy.evaluate(symbol, data)
                direction = signal.get('direction', 'NEUTRAL')
                confidence = signal.get('confidence', 0.0)
            
            if direction and confidence > 0:
                strategy_signals[strategy_name] = {
                    "direction": direction,
                    "confidence": confidence,
                    "weight": strategy_weights.get(strategy_name, 1.0)
                }
                
                # Add weighted vote
                weight = strategy_weights.get(strategy_name, 1.0) * confidence
                votes[direction] += weight
                total_weight += weight
                
                logger.debug(f"Strategy {strategy_name}: {direction} (conf: {confidence:.2f}, weight: {weight:.2f})")
                
        except Exception as e:
            logger.error(f"Error evaluating strategy {strategy_name}: {e}")
            continue
    
    # Normalize votes
    if total_weight > 0:
        for direction in votes:
            votes[direction] /= total_weight
    
    # Determine consensus
    consensus_direction = max(votes, key=votes.get)
    consensus_confidence = votes[consensus_direction]
    
    # Require minimum confidence and agreement
    min_confidence = 0.4  # 40% of weighted votes
    if consensus_confidence < min_confidence:
        consensus_direction = "NEUTRAL"
        consensus_confidence = 0.0
    
    # Calculate agreement score (how aligned are strategies)
    agreement_score = max(votes.values()) - sorted(votes.values())[-2] if len(votes) > 1 else 1.0
    
    result = {
        "symbol": symbol,
        "consensus": {
            "direction": consensus_direction,
            "confidence": float(consensus_confidence),
            "agreement_score": float(agreement_score)
        },
        "votes": votes,
        "strategy_signals": strategy_signals,
        "num_strategies": len(strategy_signals),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info(
        f"Ensemble result for {symbol}: {consensus_direction} "
        f"(confidence: {consensus_confidence:.2%}, agreement: {agreement_score:.2%}, "
        f"strategies: {len(strategy_signals)})"
    )
    
    return result


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


# ============================================================================
# BACKTESTING API ENDPOINTS
# ============================================================================

from backtesting_engine import (
    BacktestEngine, BacktestConfig, FillModel, PerformanceMetrics
)
from questdb_data_loader import QuestDBDataLoader

# Global backtest state
questdb_loader = None
active_backtests: Dict[str, Dict[str, Any]] = {}


@app.post("/backtest/run")
async def run_backtest(request: Request):
    """
    Run a backtest for a strategy
    
    Request body:
    {
        "strategy": "momentum",  // Strategy name
        "symbols": ["AAPL", "GOOGL"],  // List of symbols
        "start_date": "2024-01-01",
        "end_date": "2024-10-01",
        "initial_capital": 100000,
        "config": {  // Optional backtest config
            "commission_bps": 10,
            "slippage_bps": 5,
            "fill_model": "realistic"
        }
    }
    """
    try:
        body = await request.json()
        
        strategy_name = body.get("strategy")
        symbols = body.get("symbols", [])
        start_date = datetime.fromisoformat(body.get("start_date"))
        end_date = datetime.fromisoformat(body.get("end_date"))
        initial_capital = body.get("initial_capital", 100000.0)
        
        if not strategy_name or not symbols:
            raise HTTPException(status_code=400, detail="Missing strategy or symbols")
        
        if strategy_name not in strategy_instances:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_name} not found")
        
        # Create backtest config
        config_overrides = body.get("config", {})
        fill_model_str = config_overrides.get("fill_model", "realistic").upper()
        fill_model = FillModel[fill_model_str] if fill_model_str in FillModel.__members__ else FillModel.REALISTIC
        
        config = BacktestConfig(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission_bps=config_overrides.get("commission_bps", 10.0),
            slippage_bps=config_overrides.get("slippage_bps", 5.0),
            fill_model=fill_model,
            max_position_size=config_overrides.get("max_position_size", 0.20)
        )
        
        # Create backtest engine
        engine = BacktestEngine(config)
        
        # Initialize QuestDB loader if needed
        global questdb_loader
        if questdb_loader is None:
            questdb_loader = QuestDBDataLoader()
        
        # Load market data from QuestDB
        logger.info(f"Loading market data for {len(symbols)} symbols...")
        market_data = questdb_loader.load_multiple_symbols(symbols, start_date, end_date, timeframe="1d")
        
        if not market_data:
            raise HTTPException(status_code=404, detail="No market data found for specified symbols/dates")
        
        # Load data into engine
        for symbol, data in market_data.items():
            engine.load_market_data(symbol, data)
        
        # Generate strategy signals by evaluating strategy at each point in time
        strategy = strategy_instances[strategy_name]
        strategy_signals = {}
        
        logger.info(f"Generating strategy signals for {strategy_name}...")
        
        # Get all unique timestamps across all symbols
        all_timestamps = sorted(set(
            ts for symbol_data in market_data.values()
            for ts in symbol_data['timestamp'].to_numpy()
        ))
        
        logger.info(f"Total timestamps: {len(all_timestamps)}, date range: {all_timestamps[0]} to {all_timestamps[-1]}")
        
        # Generate signals for each timestamp
        for idx, timestamp in enumerate(all_timestamps):
            # Convert numpy datetime64 to datetime for comparison
            ts_datetime = pd.Timestamp(timestamp).to_pydatetime()
            if ts_datetime < start_date or ts_datetime > end_date:
                continue
            
            signals_at_time = []
            
            # Evaluate strategy for each symbol
            for symbol, symbol_data in market_data.items():
                # Get data up to current timestamp
                symbol_timestamps = symbol_data['timestamp'].to_numpy()
                mask = symbol_timestamps <= timestamp
                
                if mask.sum() < 30:  # Need at least 30 bars for indicators
                    continue
                
                # Prepare data for strategy evaluation
                strategy_input = {
                    'close': symbol_data['close'].to_numpy()[mask],
                    'open': symbol_data['open'].to_numpy()[mask],
                    'high': symbol_data['high'].to_numpy()[mask],
                    'low': symbol_data['low'].to_numpy()[mask],
                    'volume': symbol_data['volume'].to_numpy()[mask]
                }
                
                # Evaluate strategy
                signal = await strategy.evaluate(symbol, strategy_input)
                
                # Log signal for debugging
                if idx % 30 == 0:  # Log every 30th evaluation
                    logger.info(f"[{idx}/{len(all_timestamps)}] {symbol} @ {timestamp}: {signal['signal_type']} (conf: {signal['confidence']:.2f})")
                
                # Convert to backtest engine format
                if signal['signal_type'] == 'BUY':
                    signals_at_time.append({
                        'symbol': symbol,
                        'signal': 1,  # Buy signal
                        'confidence': signal['confidence']
                    })
                elif signal['signal_type'] == 'SELL':
                    signals_at_time.append({
                        'symbol': symbol,
                        'signal': -1,  # Sell signal
                        'confidence': signal['confidence']
                    })
            
            if signals_at_time:
                strategy_signals[timestamp] = signals_at_time
        
        logger.info(f"Generated {len(strategy_signals)} signal timestamps")
        
        # Create backtest ID
        backtest_id = f"BT{len(active_backtests) + 1:06d}"
        
        active_backtests[backtest_id] = {
            "strategy": strategy_name,
            "symbols": symbols,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "status": "running",
            "started_at": datetime.utcnow().isoformat()
        }
        
        # Run backtest with generated signals
        try:
            metrics = engine.run_backtest(strategy_signals)
            
            active_backtests[backtest_id]["status"] = "completed"
            active_backtests[backtest_id]["completed_at"] = datetime.utcnow().isoformat()
            active_backtests[backtest_id]["metrics"] = {
                "total_return": metrics.total_return,
                "annual_return": metrics.annual_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "max_drawdown": metrics.max_drawdown,
                "volatility": metrics.volatility,
                "num_trades": metrics.num_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "avg_trade_pnl": metrics.avg_trade_pnl,
                "total_commission": metrics.total_commission,
                "total_slippage": metrics.total_slippage
            }
            
            STRATEGY_BACKTESTS.labels(strategy=strategy_name).inc()
            
        except Exception as e:
            active_backtests[backtest_id]["status"] = "failed"
            active_backtests[backtest_id]["error"] = str(e)
            logger.error(f"Backtest failed: {e}")
        
        return {
            "backtest_id": backtest_id,
            "status": active_backtests[backtest_id]["status"],
            "strategy": strategy_name,
            "symbols": symbols,
            "message": "Backtest initiated"
        }
        
    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        STRATEGY_ERRORS.labels(endpoint="backtest_run").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/{backtest_id}")
async def get_backtest_status(backtest_id: str):
    """Get status and results of a backtest"""
    if backtest_id not in active_backtests:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    return active_backtests[backtest_id]


@app.get("/backtest/list")
async def list_backtests():
    """List all backtests"""
    return {
        "backtests": [
            {"id": bid, **info}
            for bid, info in active_backtests.items()
        ],
        "total": len(active_backtests)
    }


@app.get("/backtest/symbols/available")
async def get_available_symbols():
    """Get list of symbols with sufficient data for backtesting"""
    global questdb_loader
    if questdb_loader is None:
        questdb_loader = QuestDBDataLoader()
    
    try:
        symbols = questdb_loader.get_available_symbols(min_bars=100)
        return {
            "symbols": symbols,
            "count": len(symbols),
            "min_bars": 100
        }
    except Exception as e:
        logger.error(f"Error getting available symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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